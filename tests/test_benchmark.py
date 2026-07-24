import unittest

from gth import (
    load, rank_all, cohort_stats, weighted_total, build_index, similar, ask,
    BM25Index, TfidfIndex, reciprocal_rank_fusion,
)
from gth.rubric import WEIGHTS, DIMENSIONS
from gth import eval as ev


class TestRubric(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=9)

    def test_weighted_total_bounds(self):
        self.assertEqual(weighted_total({d: 0 for d in DIMENSIONS}), 0.0)
        self.assertEqual(weighted_total({d: 5 for d in DIMENSIONS}), 5.0)


class TestCorpus(unittest.TestCase):
    def setUp(self):
        self.c = load()

    def test_size(self):
        self.assertEqual(len(self.c), 192)
        self.assertEqual(len(self.c.winners), 11)

    def test_every_hero_has_all_dims(self):
        for h in self.c:
            for d in DIMENSIONS:
                self.assertIn(d, h.scores)

    def test_applicant_present(self):
        self.assertIsNotNone(self.c.applicant)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.c = load()

    def test_ranking_is_sorted(self):
        ranked = rank_all(self.c)
        totals = [r.total for r in ranked]
        self.assertEqual(totals, sorted(totals, reverse=True))
        self.assertEqual(ranked[0].rank, 1)

    def test_winner_stats_reasonable(self):
        s = cohort_stats(self.c, "Winner")
        self.assertEqual(s["n"], 11)
        self.assertLessEqual(s["min"], s["mean"])
        self.assertLessEqual(s["mean"], s["max"])


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.c = load()
        self.docs = [h.doc for h in self.c.heroes]

    def test_bm25_topical(self):
        bm = BM25Index(self.docs)
        top = bm.search("menstrual health hygiene")[0][0]
        self.assertIn("menstru", self.c.heroes[top].then.lower())

    def test_tfidf_vectors_normalized(self):
        tf = TfidfIndex(self.docs)
        import math
        for v in tf.vecs:
            if v:
                self.assertAlmostEqual(math.sqrt(sum(x * x for x in v.values())), 1.0, places=6)

    def test_rrf_fuses_ranks(self):
        fused = reciprocal_rank_fusion({"a": [3, 1, 2], "b": [1, 3, 4]})
        # doc 1 & 3 appear high in both -> should top the fusion
        top2 = {i for i, _ in fused[:2]}
        self.assertEqual(top2, {1, 3})


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.c = load()
        self.idx = build_index(self.c, backend="tfidf")

    def test_query_returns_k(self):
        hits = self.idx.query("astronomy space rocket", k=5)
        self.assertEqual(len(hits), 5)

    def test_similar_excludes_self(self):
        res = similar("Rahul Ranjan Sah", k=5, corpus=self.c, index=self.idx)
        self.assertTrue(all(r.hero.name != "Rahul Ranjan Sah" for r in res))

    def test_topical_retrieval(self):
        hits = self.idx.query("menstrual health hygiene periods", k=8)
        names = " ".join(r.hero.then.lower() for r in hits)
        self.assertIn("menstru", names)

    def test_ask_reports_config_and_provenance(self):
        out = ask("robotics and hardware", k=3, corpus=self.c, index=self.idx)
        self.assertIn("retrieved by", out)

    def test_ask_hybrid_opt_in_reports_hybrid_label(self):
        out = ask("robotics and hardware", k=3, corpus=self.c, index=self.idx, hybrid=True)
        self.assertIn("hybrid(", out)

    def test_default_methods_is_the_cv_selected_config(self):
        # The default is not a guess — python -m gth cv 5-fold cross-validates
        # every candidate and char-ngram alone wins every fold (see eval.py).
        self.assertEqual(self.idx.default_methods, ["char"])

    def test_describe_reflects_actual_call_params(self):
        self.assertEqual(self.idx.describe(["char"]), "char")
        self.assertIn("hybrid(", self.idx.describe(["bm25", "tfidf", "char"], rerank="mmr"))


class TestEval(unittest.TestCase):
    def test_metrics_in_range_and_hybrid_beats_baseline(self):
        c = load()
        rows = dict((name, m) for name, m in ev.run(c))
        for m in rows.values():
            for key, v in m.items():
                if key.startswith("_"):
                    continue
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
        # the full stack should beat the single-retriever tf-idf baseline on nDCG
        self.assertGreater(rows["hybrid+rm3+mmr"]["ndcg"], rows["tfidf"]["ndcg"])

    def test_bootstrap_ci_ordered(self):
        lo, hi = ev.bootstrap_ci([0.2, 0.4, 0.6, 0.8])
        self.assertLessEqual(lo, hi)

    def test_cross_validate_reports_held_out_scores_in_range(self):
        c = load()
        result = ev.cross_validate(c, n_folds=3)
        self.assertEqual(len(result["held_out_ndcg"]), 3)
        for v in result["held_out_ndcg"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertEqual(len(result["folds"]), 3)

    def test_paired_bootstrap_diff_symmetry(self):
        result = ev.paired_bootstrap_diff([0.8, 0.6, 0.7], [0.2, 0.4, 0.3])
        self.assertAlmostEqual(result["mean_diff"], 0.4, places=6)
        self.assertTrue(result["significant"])  # a uniformly dominates b

    def test_significance_runs_and_returns_ci(self):
        c = load()
        result = ev.significance(c)
        lo, hi = result["ci"]
        self.assertLessEqual(lo, hi)

    def test_nested_cross_validate_reports_held_out_scores_in_range(self):
        # small fold counts to keep the suite fast — full 5x4 grid search is
        # exercised via `python -m gth ncv`, not on every test run
        c = load()
        result = ev.nested_cross_validate(c, n_outer=2, n_inner=2)
        self.assertEqual(len(result["held_out_ndcg"]), 2)
        for v in result["held_out_ndcg"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
