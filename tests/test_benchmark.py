import unittest

from gth import load, rank_all, cohort_stats, weighted_total, build_index, similar, ask
from gth.rubric import WEIGHTS, DIMENSIONS


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


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.c = load()
        self.idx = build_index(self.c, backend="tfidf")

    def test_query_returns_k(self):
        hits = self.idx.query("astronomy space rocket", k=5)
        self.assertEqual(len(hits), 5)
        self.assertGreaterEqual(hits[0].score, hits[-1].score)

    def test_similar_excludes_self(self):
        res = similar("Rahul Ranjan Sah", k=5, corpus=self.c, index=self.idx)
        self.assertTrue(all(r.hero.name != "Rahul Ranjan Sah" for r in res))

    def test_topical_retrieval(self):
        # a menstrual-health query should surface a menstrual-health honoree in top-k
        hits = self.idx.query("menstrual health hygiene periods", k=8)
        names = " ".join(r.hero.then.lower() for r in hits)
        self.assertIn("menstru", names)

    def test_ask_grounded(self):
        out = ask("robotics and hardware", k=3, corpus=self.c, index=self.idx)
        self.assertIn("backend: tfidf", out)


if __name__ == "__main__":
    unittest.main()
