"""
The retrieval benchmark: a real IR test collection over the corpus (gold
queries, relevance judgments, compared systems). This is the one part of
this project that earns the word "benchmark". The rubric scoring elsewhere
in this package is a documented methodology applied to a fixed cohort, not
a standardized task multiple systems are measured against.

The 39 gold queries are deliberately NOT copied phrases from the honorees'
bios. An earlier version used queries like "menstrual health hygiene periods
dignity" against bios that literally contain "menstrual hygiene", so any
substring-matching retriever won by construction. These queries describe
the same topics in different words, so a retriever actually has to bridge
vocabulary rather than find a copy of itself.

    python -m gth eval                 # comparison table + CI
    python -m gth eval --per-query     # per-query nDCG breakdown
    python -m gth cv                   # k-fold cross-validated config selection
    python -m gth ncv                  # nested CV, publish-safe estimate
    python -m gth sig                  # paired significance test
    python -m gth tune                 # BM25 (k1,b) grid search

Labels are judgment-based ground truth for this corpus: they measure whether
the engine finds the on-topic honorees, not the honorees' rubric scores.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .corpus import Corpus, load
from .retrieval import HybridRetriever

# query -> list of (name, year) a human judges clearly relevant.
# Queries paraphrase the topic; they avoid copying the bio's own wording.
GOLD: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("young stargazers who build flying machines and compete worldwide", [
        ("Rahul Ranjan Sah", 2022), ("Swaraj Sagar Pradhan", 2020),
        ("Lucky Sah", 2022), ("Aamod Paudel", 2022), ("Osish Niraula", 2025)]),
    ("teens breaking taboos around a private monthly health topic for girls", [
        ("Ruchi Ojha", 2025), ("Sinshiya K.C.", 2024),
        ("Bidhata Pathak", 2022), ("Manushi Neupane", 2025)]),
    ("self-taught tinkerers who build machines with their own hands", [
        ("Lov Panthi", 2019), ("Bikalpa Dhungana", 2018),
        ("Santosh Lamichhane", 2016), ("Lucky Sah", 2022), ("Sahil K Gupta", 2018)]),
    ("young organizers fighting to protect a warming planet", [
        ("Kaushal Niraula", 2024), ("Sushant Sapkota", 2020),
        ("Nischal Dhungana", 2025), ("Shrijana Gautam", 2022), ("Sagar Budha", 2023)]),
    ("coders building smart software that learns from data", [
        ("Aryan Basnet", 2024), ("Safal Poudel", 2025),
        ("Sagar Gupta", 2023), ("Aarjan Chaudhary", 2026)]),
    ("campaigners working to stop underage weddings", [
        ("Bipana Sharma", 2015), ("Prashansha KC", 2018),
        ("Namrata Dahal", 2020), ("Prakash Badu", 2023)]),
    ("teens who launched their own companies with paying customers", [
        ("Kunal Sah", 2022), ("Samir Phuyal", 2019), ("Grace Thapa", 2020),
        ("Nischal Singh Bista", 2022), ("Mohammad Aftab Sheikh", 2025)]),
    ("young directors capturing real lives on camera", [
        ("Aditya Khadka", 2015), ("Bimarsha Poudel", 2022), ("Phurwa Tsering Gurung", 2025)]),
    ("students who won top honors at international science contests", [
        ("Prakash Pant", 2023), ("Madhav Khanal", 2023), ("Aamod Paudel", 2022),
        ("Shakti K.C.", 2024), ("Swaraj Sagar Pradhan", 2020)]),
    ("teens helping peers cope with emotional struggles", [
        ("Anugraha Ghale", 2021), ("Diwash Sarraf", 2023)]),
    ("young builders shipping real mobile and web products", [
        ("Prashim Timsina", 2024), ("Safal Poudel", 2025), ("Sagar Gupta", 2023),
        ("Shreejay Subedi", 2024), ("Aarjan Chaudhary", 2026)]),
    ("advocates fighting for girls to be treated as equals", [
        ("Bipana Sharma", 2015), ("Prashansha KC", 2018), ("Namrata Dahal", 2020)]),
    ("teens working on tools and campaigns to keep people well", [
        ("Rachin Kalakheti", 2019), ("Safal Poudel", 2025),
        ("Sadiksha Ghimire", 2023), ("Dhurbesh Dhami", 2025)]),
    ("curious teens running their own research and hobbyist science groups", [
        ("Amit Timalsina", 2020), ("Rahul Ranjan Sah", 2022), ("Rachin Kalakheti", 2019),
        ("Sambridhi Deo", 2022), ("Sagar Gupta", 2023)]),
    ("young leaders trying to fix problems in their community", [
        ("Sushant Sapkota", 2020), ("Anugraha Ghale", 2021), ("Kunal Sah", 2022)]),
    ("teens finding flaws in digital systems before criminals do", [
        ("Aarjan Chaudhary", 2026), ("Sajan Adhikari", 2025),
        ("Nischal Bhattarai", 2024), ("Anil Pradhan", 2018)]),
    ("students spreading a love of hard science to other kids", [
        ("Aabiskar Thapa Kshetri", 2021), ("Abhishek Karna", 2020), ("Sambridhi Deo", 2022),
        ("Swaraj Sagar Pradhan", 2020), ("Rakshit Poudel", 2025)]),
    ("teens who founded clubs where kids build and race machines", [
        ("Ekraj Ghimire", 2020), ("Bikram Parajuli", 2019), ("Aashish Shah", 2022),
        ("Lucky Sah", 2022), ("Lov Panthi", 2019), ("Tanmay Chaudhary", 2017)]),
    ("teens who taught thousands of other kids to write software", [
        ("Samir Phuyal", 2019), ("Aashish Panthi", 2024), ("Aryan Sigdel", 2023),
        ("Sabhya Rai", 2021), ("Johnson Subedi", 2021)]),
    ("a system connecting donors to patients who need transfusions", [
        ("Prashim Timsina", 2024), ("Krishtina Khanal", 2024)]),
    ("teens who rescue dangerous animals instead of killing them", [
        ("Ganesh Sah Sudi", 2019), ("Khusbu Bhandari", 2021)]),
    ("young authors who published their own written work", [
        ("Madhav Khanal", 2023), ("Aryan Basnet", 2024), ("Samata Shrestha", 2016),
        ("Deepshikha Ghimire", 2018), ("Preeti Pantha", 2023)]),
    ("activists confronting caste-based discrimination", [
        ("Bishnu Mijar", 2019), ("Gobind Pajiyar", 2021), ("Rajaram Basnet", 2017)]),
    ("teens improving how farmers grow and harvest their crops", [
        ("Renuka Singh", 2025), ("Laxman Poudel", 2020), ("Safal Poudel", 2025),
        ("Kishor Shahi", 2024), ("Risham Kumar Sah", 2023)]),
    ("young innovators turning trash into something useful", [
        ("Shruti Tiwari", 2023), ("Sabina Shakya", 2021),
        ("Sumitra Acharya", 2023), ("Bidhi Mandal", 2019)]),
    ("peer educators teaching classmates about their bodies and relationships", [
        ("Mandira Shrestha", 2020), ("Sarwagya Bhattarai", 2022),
        ("Aayushman Puri", 2024), ("Sinshiya K.C.", 2024)]),
    ("confident young voices who present in front of large crowds", [
        ("Vaibhav Nahata", 2020), ("Nischal Bhattarai", 2024),
        ("Shivu Pandey", 2019), ("Krish Yadav", 2025)]),
    ("teens making free online videos to explain school subjects", [
        ("Anurag Chapagain", 2021), ("Atith Adhikari", 2023),
        ("Aryan Sigdel", 2023), ("Amit Timalsina", 2020)]),
    ("young founders connecting workers with paid opportunities", [
        ("Kunal Sah", 2022), ("Tushar Shah", 2024),
        ("Hangsam Nembang", 2024), ("Aryan Sigdel", 2023)]),
    ("teen performers making original songs", [
        ("Avinash Kumar Paswan", 2023), ("Mohammad Aftab Sheikh", 2025)]),
    ("teens bringing schooling to kids who would otherwise be left out", [
        ("Mohan Budha", 2021), ("Sabhya Rai", 2021), ("Ashish Banjara", 2024),
        ("Saurab Banstola", 2024), ("Reet Kafle", 2020)]),
    ("young chemists inventing new cleaning and lab materials", [
        ("Suyog Vardan Acharya", 2021), ("Sambridhi Deo", 2022)]),
    ("nonprofits started by teens to widen access to technical subjects", [
        ("Saksham Rupakheti", 2025), ("Ranjan Shankar", 2022), ("Sampanna Jyoti Tuladhar", 2022)]),
    ("young Nepali voices demanding action on a heating planet", [
        ("Sanif Kandel", 2020), ("Shrijana Gautam", 2022), ("Bidhata Pathak", 2022),
        ("Kovid Bhusan Pathak", 2019), ("Nischal Dhungana", 2025), ("Prajesh Khanal", 2017)]),
    ("advocates keeping young girls in school instead of getting married off", [
        ("Ganga Sah", 2022), ("Albina Prawin", 2017), ("Babita Pariyar", 2019),
        ("Namrata Dahal", 2020), ("Ashna Poudel", 2017)]),
    ("young Nepalis connected to America's space agency and rocketry programs", [
        ("Lucky Sah", 2022), ("Prashim Timsina", 2024), ("Risham Kumar Sah", 2023),
        ("Swaraj Sagar Pradhan", 2020), ("Aamod Paudel", 2022)]),
    ("athletes teaching others how to protect themselves physically", [
        ("Suraj Sapkota", 2021), ("Sanskriti Phuyal", 2020), ("Ruby Tamang", 2017)]),
    ("builders making daily life easier for people with disabilities", [
        ("Pranjal Chalise", 2021), ("Sulav Subedi", 2020), ("Shubham Jha", 2021)]),
    ("students who flew abroad to represent Nepal in academic competitions", [
        ("Prakash Pant", 2023), ("Madhav Khanal", 2023), ("Rahul Ranjan Sah", 2022),
        ("Osish Niraula", 2025), ("Shakti K.C.", 2024)]),
]

_HYBRID = ["bm25", "tfidf", "char"]
CONFIGS = [
    ("tfidf",           dict(methods=["tfidf"], expand=None, rerank=None)),
    ("bm25",            dict(methods=["bm25"], expand=None, rerank=None)),
    ("char-ngram",      dict(methods=["char"], expand=None, rerank=None)),
    ("hybrid(rrf)",     dict(methods=_HYBRID, fusion="rrf", expand=None, rerank=None)),
    ("hybrid+rm3",      dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank=None)),
    ("hybrid+rm3+mmr",  dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr")),
]


# --------------------------------------------------------------------------- metrics
def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / k if k else 0.0


def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked, rel):
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            ap += hits / i
    return ap / len(rel) if rel else 0.0


def ndcg_at_k(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def _resolve(corpus, pairs):
    return {h.key for h in (corpus.get(n, y) for n, y in pairs) if h is not None}


def evaluate(index, corpus, cfg, k=10, gold=None):
    gold = GOLD if gold is None else gold
    per = {"recall": [], "prec": [], "mrr": [], "ndcg": [], "map": []}
    for query, g in gold:
        rel = _resolve(corpus, g)
        if not rel:
            continue
        ranked = [r.hero.key for r in index.query(query, k=max(k, 20), **cfg)]
        per["recall"].append(recall_at_k(ranked, rel, k))
        per["prec"].append(precision_at_k(ranked, rel, k))
        per["mrr"].append(reciprocal_rank(ranked, rel))
        per["ndcg"].append(ndcg_at_k(ranked, rel, k))
        per["map"].append(average_precision(ranked, rel))
    means = {m: (sum(v) / len(v) if v else 0.0) for m, v in per.items()}
    means["_ndcg_list"] = per["ndcg"]
    return means


def bootstrap_ci(values, n=2000, seed=0, alpha=0.05):
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    means = []
    m = len(values)
    for _ in range(n):
        means.append(sum(values[rnd.randrange(m)] for _ in range(m)) / m)
    means.sort()
    return (means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)])


def paired_bootstrap_diff(a_values, b_values, n=5000, seed=0, alpha=0.05):
    """95% CI on mean(a) - mean(b), resampling query indices jointly (not each
    list independently) since both configs are scored on the same queries.
    The correct test for "does config A beat config B", as opposed to two
    separate bootstrap_ci calls that would ignore the pairing."""
    assert len(a_values) == len(b_values)
    m = len(a_values)
    diffs = [a - b for a, b in zip(a_values, b_values)]
    mean_diff = sum(diffs) / m if m else 0.0
    rnd = random.Random(seed)
    boot = []
    for _ in range(n):
        idxs = [rnd.randrange(m) for _ in range(m)]
        boot.append(sum(diffs[i] for i in idxs) / m)
    boot.sort()
    lo, hi = boot[int(alpha / 2 * n)], boot[int((1 - alpha / 2) * n)]
    return {"mean_diff": mean_diff, "ci": (lo, hi), "significant": lo > 0 or hi < 0}


def significance(corpus=None, k=10, name_a="char-ngram", name_b="hybrid+rm3+mmr"):
    """Paired bootstrap significance test between two named CONFIGS rows."""
    corpus = corpus or load()
    cfg_a = dict(next(c for n, c in CONFIGS if n == name_a))
    cfg_b = dict(next(c for n, c in CONFIGS if n == name_b))
    idx = HybridRetriever(corpus)
    a = evaluate(idx, corpus, cfg_a, k)["_ndcg_list"]
    b = evaluate(idx, corpus, cfg_b, k)["_ndcg_list"]
    result = paired_bootstrap_diff(a, b)
    print(f"\nPAIRED SIGNIFICANCE TEST  -  {name_a}  vs.  {name_b}  -  nDCG@{k}  -  n={len(a)}")
    print("=" * 68)
    print(f"  mean({name_a}) = {sum(a)/len(a):.4f}")
    print(f"  mean({name_b}) = {sum(b)/len(b):.4f}")
    print(f"  mean difference = {result['mean_diff']:+.4f}")
    lo, hi = result["ci"]
    print(f"  95% CI on the difference (paired bootstrap, n=5000) = [{lo:+.4f}, {hi:+.4f}]")
    verdict = "statistically significant" if result["significant"] else "NOT statistically significant"
    print(f"  -> {verdict} at alpha=0.05 (n={len(a)} queries)")
    return result


def run(corpus=None, k=10, per_query=False):
    corpus = corpus or load()
    index = HybridRetriever(corpus)
    rows = [(name, evaluate(index, corpus, cfg, k)) for name, cfg in CONFIGS]
    _print(rows, k, sum(1 for _, g in GOLD if _resolve(corpus, g)))
    best = max(rows, key=lambda r: r[1]["ndcg"])
    lo, hi = bootstrap_ci(best[1]["_ndcg_list"])
    print(f"\n  best: {best[0]}  -  nDCG@{k} = {best[1]['ndcg']:.3f}  (95% CI {lo:.3f}-{hi:.3f}, bootstrap)")
    if per_query:
        _per_query(index, corpus, CONFIGS, best[0], k)
    return rows


def _per_query(index, corpus, configs, name, k):
    print(f"\n  per-query nDCG@{k} for {name}:")
    cfg = dict(next(c for n, c in configs if n == name))
    for query, gold in GOLD:
        rel = _resolve(corpus, gold)
        if not rel:
            continue
        ranked = [r.hero.key for r in index.query(query, k=max(k, 20), **cfg)]
        print(f"    {ndcg_at_k(ranked, rel, k):.3f}  {query}")


def tune(corpus=None, k=10):
    """Grid-search BM25 (k1, b) on the hybrid+rm3+mmr config; report best nDCG@k."""
    corpus = corpus or load()
    cfg = dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr")
    grid = [(k1, b) for k1 in (1.0, 1.2, 1.5, 2.0) for b in (0.4, 0.6, 0.75)]
    print(f"\nBM25 GRID SEARCH  -  hybrid+rm3+mmr  -  nDCG@{k}\n" + "=" * 40)
    results = []
    for k1, b in grid:
        idx = HybridRetriever(corpus, bm25_k1=k1, bm25_b=b)
        m = evaluate(idx, corpus, cfg, k)
        results.append(((k1, b), m["ndcg"]))
        print(f"  k1={k1:<4} b={b:<5} -> nDCG@{k}={m['ndcg']:.3f}")
    (bk1, bb), bnd = max(results, key=lambda t: t[1])
    print("-" * 40)
    print(f"  best: k1={bk1}, b={bb}  ->  nDCG@{k}={bnd:.3f}")
    return results


CV_CANDIDATES = [
    ("tfidf",         dict(methods=["tfidf"], expand=None, rerank=None), {}),
    ("bm25",          dict(methods=["bm25"], expand=None, rerank=None), {}),
    ("char",          dict(methods=["char"], expand=None, rerank=None), {}),
    ("hybrid(rrf)",   dict(methods=_HYBRID, fusion="rrf", expand=None, rerank=None), {"char": 0.3}),
    ("hybrid+rm3",    dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank=None), {"char": 0.3}),
    ("hybrid+rm3+mmr w=0.3", dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.3}),
    ("hybrid+rm3+mmr w=0.5", dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.5}),
    ("hybrid+rm3+mmr w=0.7", dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.7}),
    ("hybrid+rm3+mmr w=1.0", dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr"), {"char": 1.0}),
]


def _folds(n_folds: int, gold=None):
    """Deterministic, non-contiguous k-fold split by index striping (no randomness)."""
    gold = GOLD if gold is None else gold
    return [[q for i, q in enumerate(gold) if i % n_folds == f] for f in range(n_folds)]


# Grids searched INSIDE nested_cross_validate, never fit on the full 39-query
# set, so the reported held-out score carries no tuning leakage at all. (The
# CV_CANDIDATES weights above, e.g. char=0.3, WERE picked from a manual sweep
# on the full gold set: fine for cross_validate's apples-to-apples comparison
# of named strategies, but not a clean generalization bound on hyperparameter
# selection. Nested CV re-derives them from scratch per fold.)
_BM25_GRID = [(k1, b) for k1 in (1.0, 1.5, 2.0) for b in (0.4, 0.6, 0.75)]
_CHAR_WEIGHT_GRID = (0.0, 0.3, 0.5, 0.7, 1.0)


def _inner_select(corpus, train_gold, k, n_inner):
    """Grid-search BM25 (k1,b) x char-fusion-weight, plus every plain single
    retriever, using ONLY an inner k-fold split of train_gold. Returns the
    winning (name, cfg, weights, k1, b) by mean inner-fold nDCG@k."""
    inner_folds = _folds(n_inner, train_gold)

    def inner_ndcg(cfg, weights, k1, b):
        idx = HybridRetriever(corpus, bm25_k1=k1, bm25_b=b)
        idx.weights.update(weights)
        scores = []
        for i in range(n_inner):
            val = inner_folds[i]
            if not val:
                continue
            scores.append(evaluate(idx, corpus, cfg, k, gold=val)["ndcg"])
        return sum(scores) / len(scores) if scores else 0.0

    best = None
    for name, methods in (("tfidf", ["tfidf"]), ("bm25", ["bm25"]), ("char", ["char"])):
        cfg = dict(methods=methods, expand=None, rerank=None)
        score = inner_ndcg(cfg, {}, 1.5, 0.4)  # bm25 k1/b only matter for the "bm25" row
        if best is None or score > best[0]:
            best = (score, name, cfg, {}, 1.5, 0.4)
    cfg = dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr")
    for k1, b in _BM25_GRID:
        for w in _CHAR_WEIGHT_GRID:
            score = inner_ndcg(cfg, {"char": w}, k1, b)
            if score > best[0]:
                best = (score, f"hybrid+rm3+mmr(k1={k1},b={b},w={w})", cfg, {"char": w}, k1, b)
    return best


def nested_cross_validate(corpus=None, k=10, n_outer=5, n_inner=4):
    """Nested k-fold CV: hyperparameters (BM25 k1/b, char fusion weight) are
    grid-searched INSIDE each outer training fold via an inner CV split, never
    touching the outer test fold and never fit on the full gold set. This is
    the unbiased generalization estimate, the number safe to publish, since
    cross_validate's weight grid was itself picked by eyeballing the full set.
    """
    corpus = corpus or load()
    folds = _folds(n_outer, GOLD)
    print(f"\nNESTED {n_outer}x{n_inner}-FOLD CROSS-VALIDATION  -  {len(GOLD)} queries  -  k={k}")
    print("=" * 68)
    winners, held_out = [], []
    for f in range(n_outer):
        test_gold = folds[f]
        train_gold = [q for i, fold in enumerate(folds) if i != f for q in fold]
        inner_score, name, cfg, weights, k1, b = _inner_select(corpus, train_gold, k, n_inner)
        idx = HybridRetriever(corpus, bm25_k1=k1, bm25_b=b)
        idx.weights.update(weights)
        test_m = evaluate(idx, corpus, cfg, k, gold=test_gold)
        winners.append(name)
        held_out.append(test_m["ndcg"])
        print(f"  fold {f+1}/{n_outer}: inner-selected={name:<30} inner_nDCG={inner_score:.3f}"
              f"  ->  held-out nDCG={test_m['ndcg']:.3f}  (n={len(test_gold)})")
    avg = sum(held_out) / len(held_out)
    lo, hi = bootstrap_ci(held_out)
    print("-" * 68)
    from collections import Counter as _C
    families = _C(w.split("(")[0] for w in winners)
    print(f"  winning family per fold: {dict(families)}")
    print(f"  nested cross-validated nDCG@{k} = {avg:.3f}  (95% CI {lo:.3f}-{hi:.3f} over folds)")
    return {"folds": winners, "held_out_ndcg": held_out, "mean": avg, "ci": (lo, hi)}


def cross_validate(corpus=None, k=10, n_folds=5):
    """K-fold CV: on each fold's TRAIN queries, pick the best candidate config by
    nDCG@k; score it on the held-out TEST queries. Reports the honest,
    out-of-fold generalization estimate, the number that actually justifies a
    shipped default, as opposed to a config tuned and reported on the same set.
    """
    corpus = corpus or load()
    folds = _folds(n_folds, GOLD)
    print(f"\n{n_folds}-FOLD CROSS-VALIDATION  -  {len(GOLD)} queries  -  cutoff k={k}")
    print("=" * 68)
    winners, held_out = [], []
    for f in range(n_folds):
        test_gold = folds[f]
        train_gold = [q for i, fold in enumerate(folds) if i != f for q in fold]
        best_name, best_train_ndcg, best_cfg = None, -1.0, None
        for name, cfg, weights in CV_CANDIDATES:
            idx = HybridRetriever(corpus)
            idx.weights.update(weights)
            m = evaluate(idx, corpus, cfg, k, gold=train_gold)
            if m["ndcg"] > best_train_ndcg:
                best_name, best_train_ndcg, best_cfg = name, m["ndcg"], (cfg, weights)
        cfg, weights = best_cfg
        idx = HybridRetriever(corpus)
        idx.weights.update(weights)
        test_m = evaluate(idx, corpus, cfg, k, gold=test_gold)
        winners.append(best_name)
        held_out.append(test_m["ndcg"])
        print(f"  fold {f+1}/{n_folds}: trained-best={best_name:<24} train_nDCG={best_train_ndcg:.3f}"
              f"  ->  held-out nDCG={test_m['ndcg']:.3f}  (n={len(test_gold)})")
    avg = sum(held_out) / len(held_out)
    lo, hi = bootstrap_ci(held_out)
    print("-" * 68)
    from collections import Counter as _C
    tally = _C(winners)
    print(f"  winner per fold: {dict(tally)}")
    print(f"  cross-validated nDCG@{k} = {avg:.3f}  (95% CI {lo:.3f}-{hi:.3f} over folds)")
    return {"folds": winners, "held_out_ndcg": held_out, "mean": avg, "ci": (lo, hi)}


def _print(rows, k, n_queries):
    print(f"\nRETRIEVAL EVALUATION  -  {n_queries} labeled queries  -  cutoff k={k}")
    print("=" * 68)
    print(f"{'config':<18}{'Recall@k':>10}{'Prec@k':>9}{'MRR':>8}{'nDCG@k':>9}{'MAP':>8}")
    print("-" * 68)
    best = max(r[1]["ndcg"] for r in rows)
    for name, m in rows:
        star = "  *" if m["ndcg"] == best else ""
        print(f"{name:<18}{m['recall']:>10.3f}{m['prec']:>9.3f}{m['mrr']:>8.3f}{m['ndcg']:>9.3f}{m['map']:>8.3f}{star}")
    print("-" * 68)
    print("  * = best nDCG@k. Higher is better on every metric (range 0-1).")


if __name__ == "__main__":
    run()
