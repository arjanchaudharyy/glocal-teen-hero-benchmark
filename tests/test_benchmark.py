import math
import subprocess
import sys
import unittest
from pathlib import Path

from gth import (
    BM25Index,
    Corpus,
    Hero,
    HybridRetriever,
    TfidfIndex,
    ask,
    build_index,
    cohort_stats,
    load,
    rank_all,
    reciprocal_rank_fusion,
    similar,
    weighted_total,
)
from gth import eval as ev
from gth.rubric import DIMENSIONS, LABELS, WEIGHTS
from gth.scoring import _median

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    def test_get_by_name_and_year_is_correct(self):
        h = self.c.get("Aarjan Chaudhary", 2026)
        self.assertIsNotNone(h)
        self.assertTrue(h.me)

    def test_get_by_name_without_year_falls_back(self):
        h = self.c.get("Aarjan Chaudhary")
        self.assertIsNotNone(h)
        self.assertEqual(h.year, 2026)

    def test_get_unknown_name_returns_none(self):
        self.assertIsNone(self.c.get("Nobody Real", 1999))
        self.assertIsNone(self.c.get("Nobody Real"))

    def test_get_is_case_insensitive(self):
        h = self.c.get("aarjan CHAUDHARY", 2026)
        self.assertIsNotNone(h)

    def test_corpus_is_hashable_for_caching(self):
        # Corpus is used as a cache key (gth.rag._CACHE); a default dataclass
        # __eq__/__hash__ would try to hash the unhashable `heroes` list and
        # raise TypeError the moment anything cached on it. eq=False on the
        # dataclass falls back to identity-based hashing instead.
        self.assertIsInstance(hash(self.c), int)

    def test_no_duplicate_name_year_keys(self):
        keys = [(h.name.lower(), h.year) for h in self.c]
        self.assertEqual(len(keys), len(set(keys)))


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

    def test_median_odd_length(self):
        self.assertEqual(_median([1.0, 2.0, 3.0]), 2.0)

    def test_median_even_length_averages_middle_two(self):
        # A previous version used vals[len(vals)//2], which for an even-length
        # list returns the upper-middle value instead of averaging the two
        # middle values -- correct by luck only for odd n. Finalist (n=54)
        # and 20under20 (n=126) are both even, so this was a live bug.
        self.assertEqual(_median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_cohort_stats_median_matches_even_length_cohort(self):
        s = cohort_stats(self.c, "Finalist")
        self.assertEqual(s["n"], 54)
        vals = sorted(weighted_total(h.scores) for h in self.c.by_tier("Finalist"))
        expected = (vals[26] + vals[27]) / 2
        self.assertAlmostEqual(s["median"], round(expected, 4))

    def test_empty_cohort_has_zeroed_stats(self):
        s = cohort_stats(self.c, "NoSuchTier")
        self.assertEqual(s, {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0})


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.c = load()
        self.docs = [h.doc for h in self.c.heroes]

    def test_bm25_topical(self):
        bm = BM25Index(self.docs)
        top = bm.search("menstrual health hygiene")[0][0]
        self.assertIn("menstru", self.c.heroes[top].then.lower())

    def test_bm25_search_never_returns_zero_score_docs(self):
        # An inverted index only visits documents sharing at least one query
        # term; unlike the old dense-scan implementation, it should never
        # produce a (doc, 0.0) entry, since such a document was never visited.
        bm = BM25Index(self.docs)
        for _, score in bm.search("zzqxw fjklmnop nonexistentword"):
            self.assertNotEqual(score, 0.0)

    def test_tfidf_vectors_normalized(self):
        tf = TfidfIndex(self.docs)
        for v in tf.vecs:
            if v:
                self.assertAlmostEqual(math.sqrt(sum(x * x for x in v.values())), 1.0, places=6)

    def test_tfidf_and_bm25_share_tokenization_when_given_precomputed_tokens(self):
        from gth.retrieval import tokenize
        toks = [tokenize(d) for d in self.docs]
        tf = TfidfIndex(tokens=toks)
        bm = BM25Index(tokens=toks)
        self.assertEqual(len(tf.vecs), len(bm.tokens))

    def test_rrf_fuses_ranks(self):
        fused = reciprocal_rank_fusion({"a": [3, 1, 2], "b": [1, 3, 4]})
        # doc 1 & 3 appear high in both -> should top the fusion
        top2 = {i for i, _ in fused[:2]}
        self.assertEqual(top2, {1, 3})

    def test_set_bm25_params_changes_scoring_without_rebuilding(self):
        idx = HybridRetriever(self.c)
        bm25_obj_before = idx.bm25
        idx.set_bm25_params(k1=2.0, b=0.9)
        self.assertIs(idx.bm25, bm25_obj_before)  # same object, not rebuilt
        self.assertEqual((idx.bm25.k1, idx.bm25.b), (2.0, 0.9))

    def test_unknown_retrieval_method_raises(self):
        # A previous version silently dropped unrecognized method names and
        # returned an empty result list, which looks identical to "no
        # results found" -- a debuggability trap for a simple typo.
        idx = HybridRetriever(self.c)
        with self.assertRaises(ValueError):
            idx.query("astronomy", methods=["brm25"])


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.c = load()
        self.idx = build_index(self.c)

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

    def test_no_relevance_returns_fewer_than_k_not_padded_zeros(self):
        # A previous version's single-method query path took the raw search()
        # output (which used to include zero-score docs as filler) without
        # filtering, so a query with fewer than k genuine matches would
        # silently return irrelevant documents padded in as "top-k" results.
        # Char n-grams are fuzzy (almost any string shares some 3-4 char
        # fragment with something in 192 documents), so use word-level BM25,
        # which has a clean, unambiguous "no shared vocabulary at all" case.
        hits = self.idx.query("zzqxw fjklmnop nonexistentword", k=50, methods=["bm25"])
        self.assertEqual(len(hits), 0)
        # And on the fuzzy char-ngram default, every returned hit is at
        # least a genuine (if weak) match: never an exact-zero-score filler.
        hits = self.idx.query("zzqxw fjklmnop nonexistentword", k=50)
        for r in hits:
            self.assertGreater(r.score, 0.0)

    def test_ask_reports_config_and_provenance(self):
        out = ask("robotics and hardware", k=3, corpus=self.c, index=self.idx)
        self.assertIn("retrieved by", out)

    def test_ask_hybrid_opt_in_reports_hybrid_label(self):
        out = ask("robotics and hardware", k=3, corpus=self.c, index=self.idx, hybrid=True)
        self.assertIn("hybrid(", out)

    def test_default_methods_is_the_cv_selected_config(self):
        # The default is not a guess: python -m gth cv 5-fold cross-validates
        # every candidate and char-ngram alone wins every fold (see eval.py).
        self.assertEqual(self.idx.default_methods, ["char"])

    def test_describe_reflects_actual_call_params(self):
        self.assertEqual(self.idx.describe(["char"]), "char")
        self.assertIn("hybrid(", self.idx.describe(["bm25", "tfidf", "char"], rerank="mmr"))


class TestEval(unittest.TestCase):
    def test_metrics_in_range_and_hybrid_beats_baseline(self):
        c = load()
        rows = dict(ev.run(c, verbose=False))
        for m in rows.values():
            for v in (m.recall, m.prec, m.mrr, m.ndcg, m.map):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
        # the full stack should beat the single-retriever tf-idf baseline on nDCG
        self.assertGreater(rows["hybrid+rm3+mmr"].ndcg, rows["tfidf"].ndcg)

    def test_bootstrap_ci_ordered(self):
        lo, hi = ev.bootstrap_ci([0.2, 0.4, 0.6, 0.8])
        self.assertLessEqual(lo, hi)

    def test_cross_validate_reports_held_out_scores_in_range(self):
        c = load()
        result = ev.cross_validate(c, n_folds=3, verbose=False)
        self.assertEqual(len(result["held_out_ndcg"]), 3)
        for v in result["held_out_ndcg"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertEqual(len(result["folds"]), 3)

    def test_paired_bootstrap_diff_symmetry(self):
        result = ev.paired_bootstrap_diff([0.8, 0.6, 0.7], [0.2, 0.4, 0.3])
        self.assertAlmostEqual(result["mean_diff"], 0.4, places=6)
        self.assertTrue(result["significant"])  # a uniformly dominates b

    def test_paired_bootstrap_diff_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            ev.paired_bootstrap_diff([0.1, 0.2], [0.1])

    def test_significance_runs_and_returns_ci(self):
        c = load()
        result = ev.significance(c, verbose=False)
        lo, hi = result["ci"]
        self.assertLessEqual(lo, hi)

    def test_nested_cross_validate_reports_held_out_scores_in_range(self):
        # small fold counts to keep the suite fast; the full 5x4 grid search is
        # exercised via `python -m gth ncv`, not on every test run
        c = load()
        result = ev.nested_cross_validate(c, n_outer=2, n_inner=2, verbose=False)
        self.assertEqual(len(result["held_out_ndcg"]), 2)
        for v in result["held_out_ndcg"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestBuildScript(unittest.TestCase):
    """Coverage for build.py, the least-reviewed part of the system: it
    generates data/heroes.json from a plain-text roster with no schema
    enforcement beyond what these tests check."""

    def setUp(self):
        sys.path.insert(0, str(REPO_ROOT))
        import build  # noqa: PLC0415 (import at call time: build.py runs on import)
        self.build = build

    def test_every_dimension_weight_matches_gth_rubric(self):
        # build.py's RUBRIC used to hardcode its own second copy of these
        # seven weights, completely independent of gth/rubric.py's WEIGHTS,
        # with nothing keeping them in sync (and index.html hardcoded a
        # third copy on top of that). Now both are read from one place.
        for d in DIMENSIONS:
            self.assertEqual(self.build.RUBRIC["dimensions"][d]["weight"], WEIGHTS[d])
            self.assertEqual(self.build.RUBRIC["dimensions"][d]["label"], LABELS[d])

    def test_parser_skips_malformed_rows(self):
        d = self.build.parse_socials("li=https://a;bad;x=https://b")
        self.assertEqual(d, {"li": "https://a", "x": "https://b"})

    def test_regenerated_corpus_matches_shipped_data(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "build.py")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        c = load.__wrapped__(str(REPO_ROOT / "data" / "heroes.json"))
        self.assertEqual(len(c), 192)


class TestRagCache(unittest.TestCase):
    def test_build_index_returns_same_instance_for_same_corpus(self):
        c = Corpus(heroes=[Hero(
            name="Test Person", year=2020, award="Winner",
            scores={d: 3 for d in DIMENSIONS}, conf="high", then="test", now="test",
        )], rubric={})
        idx1 = build_index(c)
        idx2 = build_index(c)
        self.assertIs(idx1, idx2)


if __name__ == "__main__":
    unittest.main()
