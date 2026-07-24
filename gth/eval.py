"""
Retrieval evaluation harness.

A hand-labeled query set (query -> honorees a human judges relevant) plus
standard IR metrics — Recall@k, Precision@k, MRR, nDCG@k, MAP — computed for
every retrieval configuration, with a bootstrap 95% CI on the winner and an
optional per-query breakdown. A small grid-search (`tune`) is included too.

    python -m gth eval                 # comparison table + CI
    python -m gth eval --per-query     # per-query nDCG breakdown
    python -m gth tune                 # BM25 (k1,b) grid search

Labels are judgment-based ground truth for *this* corpus: they measure whether
the engine finds the on-topic honorees, not the benchmark scores.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .corpus import Corpus, load
from .retrieval import HybridRetriever

# query -> list of (name, year) a human judges clearly relevant
GOLD: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("astronomy space rocket satellite olympiad", [
        ("Rahul Ranjan Sah", 2022), ("Swaraj Sagar Pradhan", 2020),
        ("Lucky Sah", 2022), ("Aamod Paudel", 2022), ("Osish Niraula", 2025)]),
    ("menstrual health hygiene periods dignity", [
        ("Ruchi Ojha", 2025), ("Dipisha Bhujel", 2018),
        ("Sinshiya K.C.", 2024), ("Sushan Shrestha", 2023)]),
    ("robotics hardware engineering invention", [
        ("Lov Panthi", 2019), ("Bikalpa Dhungana", 2018),
        ("Santosh Lamichhane", 2016), ("Lucky Sah", 2022), ("Sahil K Gupta", 2018)]),
    ("climate environment tree planting sustainability", [
        ("Kaushal Niraula", 2024), ("Sushant Sapkota", 2020),
        ("Nischal Dhungana", 2025), ("Shrijana Gautam", 2022), ("Sagar Budha", 2023)]),
    ("artificial intelligence machine learning developer", [
        ("Aryan Basnet", 2024), ("Safal Poudel", 2025), ("Sagar Gupta", 2023),
        ("Amit Timalsina", 2020), ("Aarjan Chaudhary", 2026)]),
    ("child rights child marriage advocacy", [
        ("Bipana Sharma", 2015), ("Prashansha KC", 2018),
        ("Namrata Dahal", 2020), ("Prakash Badu", 2023)]),
    ("startup founder entrepreneur business", [
        ("Kunal Sah", 2022), ("Samir Phuyal", 2019), ("Grace Thapa", 2020),
        ("Nischal Singh Bista", 2022), ("Mohammad Aftab Sheikh", 2025)]),
    ("filmmaker documentary storytelling", [
        ("Aditya Khadka", 2015), ("Bimarsha Poudel", 2022), ("Phurwa Tsering Gurung", 2025)]),
    ("mathematics physics olympiad medal", [
        ("Prakash Pant", 2023), ("Madhav Khanal", 2023),
        ("Aamod Paudel", 2022), ("Abhishek Karna", 2020)]),
    ("mental health awareness youth wellbeing", [
        ("Manushi Neupane", 2025), ("Aayushman Puri", 2024), ("Sarwagya Bhattarai", 2022)]),
    ("programming app development software building", [
        ("Prashim Timsina", 2024), ("Safal Poudel", 2025), ("Sagar Gupta", 2023),
        ("Shreejay Subedi", 2024), ("Aarjan Chaudhary", 2026)]),
    ("girls empowerment gender equality women", [
        ("Bipana Sharma", 2015), ("Prashansha KC", 2018), ("Namrata Dahal", 2020)]),
    ("healthcare medical diagnosis disease", [
        ("Rachin Kalakheti", 2019), ("Safal Poudel", 2025)]),
    ("young scientist research science community", [
        ("Amit Timalsina", 2020), ("Rahul Ranjan Sah", 2022), ("Rachin Kalakheti", 2019)]),
    ("social impact community changemaker", [
        ("Sushant Sapkota", 2020), ("Anugraha Ghale", 2021), ("Kunal Sah", 2022)]),
    ("cybersecurity ethical hacking vulnerability", [
        ("Aarjan Chaudhary", 2026), ("Sajan Adhikari", 2025), ("Nischal Bhattarai", 2024)]),
    ("physics quantum research education", [
        ("Aabiskar Thapa Kshetri", 2021), ("Abhishek Karna", 2020),
        ("Sambridhi Deo", 2022), ("Swaraj Sagar Pradhan", 2020), ("Rakshit Poudel", 2025)]),
    ("robotics club robot competition building", [
        ("Ekraj Ghimire", 2020), ("Bikram Parajuli", 2019), ("Aashish Shah", 2022),
        ("Lucky Sah", 2022), ("Lov Panthi", 2019), ("Tanmay Chaudhary", 2017)]),
    ("coding education teaching students programming", [
        ("Samir Phuyal", 2019), ("Aashish Panthi", 2024), ("Aryan Sigdel", 2023),
        ("Sabhya Rai", 2021), ("Johnson Subedi", 2021)]),
    ("blood donation platform app", [
        ("Prashim Timsina", 2024), ("Krishtina Khanal", 2024)]),
    ("snake wildlife rescue conservation", [
        ("Ganesh Sah Sudi", 2019), ("Khusbu Bhandari", 2021)]),
    ("poetry writing author novel book", [
        ("Madhav Khanal", 2023), ("Aryan Basnet", 2024), ("Samata Shrestha", 2016),
        ("Deepshikha Ghimire", 2018), ("Preeti Pantha", 2023)]),
    ("anti-caste dalit discrimination equality", [
        ("Bishnu Mijar", 2019), ("Gobind Pajiyar", 2021), ("Rajaram Basnet", 2017)]),
    ("agriculture farming crops food", [
        ("Renuka Singh", 2025), ("Laxman Poudel", 2020), ("Safal Poudel", 2025),
        ("Kishor Shahi", 2024), ("Risham Kumar Sah", 2023)]),
    ("recycling waste plastic pollution", [
        ("Shruti Tiwari", 2023), ("Sabina Shakya", 2021), ("Sumitra Acharya", 2023),
        ("Bidhi Mandal", 2019)]),
    ("sexual reproductive health SRHR education", [
        ("Mandira Shrestha", 2020), ("Sarwagya Bhattarai", 2022),
        ("Aayushman Puri", 2024), ("Sinshiya K.C.", 2024)]),
    ("public speaking debate leadership", [
        ("Vaibhav Nahata", 2020), ("Nischal Bhattarai", 2024),
        ("Shivu Pandey", 2019), ("Krish Yadav", 2025)]),
    ("science education outreach youtube", [
        ("Anurag Chapagain", 2021), ("Atith Adhikari", 2023),
        ("Aryan Sigdel", 2023), ("Amit Timalsina", 2020)]),
    ("entrepreneurship internship jobs platform", [
        ("Kunal Sah", 2022), ("Tushar Shah", 2024), ("Hangsam Nembang", 2024),
        ("Aryan Sigdel", 2023)]),
    ("music rapper singer artist", [
        ("Avinash Kumar Paswan", 2023), ("Mohammad Aftab Sheikh", 2025)]),
    ("education access marginalized rural children", [
        ("Mohan Budha", 2021), ("Sabhya Rai", 2021), ("Ashish Banjara", 2024),
        ("Saurab Banstola", 2024), ("Reet Kafle", 2020)]),
    ("chemistry sanitizer materials polymer", [
        ("Suyog Vardan Acharya", 2021), ("Sambridhi Deo", 2022)]),
    ("STEM access foundation education", [
        ("Saksham Rupakheti", 2025), ("Ranjan Shankar", 2022),
        ("Sampanna Jyoti Tuladhar", 2022)]),
    ("climate activism youth organizing", [
        ("Sanif Kandel", 2020), ("Shrijana Gautam", 2022), ("Bidhata Pathak", 2022),
        ("Kovid Bhusan Pathak", 2019), ("Nischal Dhungana", 2025), ("Prajesh Khanal", 2017)]),
    ("girls education anti child marriage", [
        ("Ganga Sah", 2022), ("Albina Prawin", 2017), ("Babita Pariyar", 2019),
        ("Namrata Dahal", 2020), ("Ashna Poudel", 2017)]),
    ("space nasa rocket camp launch", [
        ("Lucky Sah", 2022), ("Prashim Timsina", 2024), ("Risham Kumar Sah", 2023),
        ("Swaraj Sagar Pradhan", 2020), ("Aamod Paudel", 2022)]),
    ("sports self defense taekwondo", [
        ("Suraj Sapkota", 2021), ("Sanskriti Phuyal", 2020), ("Ruby Tamang", 2017)]),
    ("disability blind accessibility assistive technology", [
        ("Pranjal Chalise", 2021), ("Sulav Subedi", 2020), ("Shubham Jha", 2021)]),
    ("olympiad international medal representation", [
        ("Prakash Pant", 2023), ("Madhav Khanal", 2023), ("Rahul Ranjan Sah", 2022),
        ("Osish Niraula", 2025), ("Shakti K.C.", 2024)]),
]

# NOTE: methods is always spelled out explicitly here (never None) — the
# default `HybridRetriever.default_methods` is itself the CV-selected winner
# (char-ngram alone, see cross_validate()), so leaving it implicit would make
# every "hybrid(...)" row silently collapse to a single-method run.
_HYBRID = ["bm25", "tfidf", "char"]
CONFIGS = [
    ("tfidf",           dict(methods=["tfidf"], expand=None, rerank=None)),
    ("bm25",            dict(methods=["bm25"], expand=None, rerank=None)),
    ("qlm",             dict(methods=["qlm"], expand=None, rerank=None)),
    ("char-ngram",      dict(methods=["char"], expand=None, rerank=None)),
    ("hybrid(rrf)",     dict(methods=_HYBRID, fusion="rrf", expand=None, rerank=None)),
    ("hybrid(combsum)", dict(methods=_HYBRID, fusion="combsum", expand=None, rerank=None)),
    ("hybrid(combmnz)", dict(methods=_HYBRID, fusion="combmnz", expand=None, rerank=None)),
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


def run(corpus=None, k=10, per_query=False):
    corpus = corpus or load()
    index = HybridRetriever(corpus)
    configs = list(CONFIGS)
    from .retrieval import _try_cross_encoder
    if _try_cross_encoder():  # add a neural-reranker row only when available
        configs.append(("hybrid+rm3+cross", dict(methods=None, fusion="rrf", expand="rm3", rerank="cross")))
    rows = [(name, evaluate(index, corpus, cfg, k)) for name, cfg in configs]
    _print(rows, k, sum(1 for _, g in GOLD if _resolve(corpus, g)))
    best = max(rows, key=lambda r: r[1]["ndcg"])
    lo, hi = bootstrap_ci(best[1]["_ndcg_list"])
    print(f"\n  best: {best[0]}  ·  nDCG@{k} = {best[1]['ndcg']:.3f}  (95% CI {lo:.3f}–{hi:.3f}, bootstrap)")
    if per_query:
        _per_query(index, corpus, configs, best[0], k)
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
    """Grid-search BM25 (k1, b) on the hybrid+rm3+mmr config; report best nDCG@k.

    (BM25 params only matter when bm25 is actually in play — the CV-selected
    HybridRetriever default is char-ngram alone, so this pins methods explicitly.)
    """
    corpus = corpus or load()
    cfg = dict(methods=_HYBRID, fusion="rrf", expand="rm3", rerank="mmr")
    grid = [(k1, b) for k1 in (1.0, 1.2, 1.5, 2.0) for b in (0.4, 0.6, 0.75)]
    print(f"\nBM25 GRID SEARCH  ·  hybrid+rm3+mmr  ·  nDCG@{k}\n" + "=" * 40)
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
    ("hybrid(rrf)",   dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand=None, rerank=None), {"char": 0.3}),
    ("hybrid+rm3",    dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand="rm3", rerank=None), {"char": 0.3}),
    ("hybrid+rm3+mmr w=0.3", dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.3}),
    ("hybrid+rm3+mmr w=0.5", dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.5}),
    ("hybrid+rm3+mmr w=0.7", dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand="rm3", rerank="mmr"), {"char": 0.7}),
    ("hybrid+rm3+mmr w=1.0", dict(methods=["bm25", "tfidf", "char"], fusion="rrf", expand="rm3", rerank="mmr"), {"char": 1.0}),
]


def _folds(n_folds: int, gold=None):
    """Deterministic, non-contiguous k-fold split by index striping (no randomness)."""
    gold = GOLD if gold is None else gold
    return [[q for i, q in enumerate(gold) if i % n_folds == f] for f in range(n_folds)]


def cross_validate(corpus=None, k=10, n_folds=5):
    """K-fold CV: on each fold's TRAIN queries, pick the best candidate config by
    nDCG@k; score it on the held-out TEST queries. Reports the honest,
    out-of-fold generalization estimate — the number that actually justifies a
    shipped default, as opposed to a config tuned and reported on the same set.
    """
    corpus = corpus or load()
    folds = _folds(n_folds, GOLD)
    print(f"\n{n_folds}-FOLD CROSS-VALIDATION  ·  {len(GOLD)} queries  ·  cutoff k={k}")
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
    print(f"  cross-validated nDCG@{k} = {avg:.3f}  (95% CI {lo:.3f}–{hi:.3f} over folds)")
    return {"folds": winners, "held_out_ndcg": held_out, "mean": avg, "ci": (lo, hi)}


def _print(rows, k, n_queries):
    print(f"\nRETRIEVAL EVALUATION  ·  {n_queries} labeled queries  ·  cutoff k={k}")
    print("=" * 68)
    print(f"{'config':<18}{'Recall@k':>10}{'Prec@k':>9}{'MRR':>8}{'nDCG@k':>9}{'MAP':>8}")
    print("-" * 68)
    best = max(r[1]["ndcg"] for r in rows)
    for name, m in rows:
        star = "  ★" if m["ndcg"] == best else ""
        print(f"{name:<18}{m['recall']:>10.3f}{m['prec']:>9.3f}{m['mrr']:>8.3f}{m['ndcg']:>9.3f}{m['map']:>8.3f}{star}")
    print("-" * 68)
    print("  ★ = best nDCG@k. Higher is better on every metric (range 0–1).")


if __name__ == "__main__":
    run()
