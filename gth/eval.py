"""
Retrieval evaluation harness.

A hand-labeled query set (query -> the honorees a human considers relevant) plus
standard IR metrics — Recall@k, Precision@k, MRR, nDCG@k, MAP — computed for
each retrieval configuration so the methods can be compared on equal footing.

    python -m gth eval            # full comparison table
    from gth.eval import run; run()

The labels are judgment-based ground truth for *this* corpus; they exist to
measure retrieval quality (does the engine find the on-topic honorees?), not to
re-litigate the benchmark scores.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .corpus import Corpus, load
from .retrieval import HybridRetriever

# query -> list of (name, year) that a human judges clearly relevant
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
]

CONFIGS = [
    ("tfidf",        dict(methods=["tfidf"], rerank=False, expand=False)),
    ("bm25",         dict(methods=["bm25"], rerank=False, expand=False)),
    ("hybrid(rrf)",  dict(methods=None, rerank=False, expand=False)),
    ("hybrid+prf",   dict(methods=None, rerank=False, expand=True)),
    ("hybrid+mmr",   dict(methods=None, rerank=True, expand=True)),
]


# --------------------------------------------------------------------------- metrics
def recall_at_k(ranked: Sequence[str], rel: set, k: int) -> float:
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked: Sequence[str], rel: set, k: int) -> float:
    return len(set(ranked[:k]) & rel) / k if k else 0.0


def reciprocal_rank(ranked: Sequence[str], rel: set) -> float:
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], rel: set) -> float:
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            ap += hits / i
    return ap / len(rel) if rel else 0.0


def ndcg_at_k(ranked: Sequence[str], rel: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def _resolve(corpus: Corpus, pairs) -> set:
    keys = set()
    for name, year in pairs:
        h = corpus.get(name, year)
        if h is not None:
            keys.add(h.key)
    return keys


def evaluate(index: HybridRetriever, corpus: Corpus, cfg: dict, k: int = 10) -> Dict[str, float]:
    R = P = MRR = NDCG = MAP = 0.0
    n = 0
    for query, gold in GOLD:
        rel = _resolve(corpus, gold)
        if not rel:
            continue
        ranked = [r.hero.key for r in index.query(query, k=max(k, 20), **cfg)]
        R += recall_at_k(ranked, rel, k)
        P += precision_at_k(ranked, rel, k)
        MRR += reciprocal_rank(ranked, rel)
        NDCG += ndcg_at_k(ranked, rel, k)
        MAP += average_precision(ranked, rel)
        n += 1
    n = n or 1
    return {"recall": R / n, "prec": P / n, "mrr": MRR / n, "ndcg": NDCG / n, "map": MAP / n}


def run(corpus: Corpus = None, k: int = 10) -> List[Tuple[str, Dict[str, float]]]:
    corpus = corpus or load()
    index = build = HybridRetriever(corpus)
    rows = [(name, evaluate(index, corpus, cfg, k)) for name, cfg in CONFIGS]
    _print(rows, k, len(GOLD))
    return rows


def _print(rows, k, n_queries):
    print(f"\nRETRIEVAL EVALUATION  ·  {n_queries} labeled queries  ·  cutoff k={k}")
    print("=" * 66)
    print(f"{'config':<14}{'Recall@k':>10}{'Prec@k':>9}{'MRR':>8}{'nDCG@k':>9}{'MAP':>8}")
    print("-" * 66)
    best = max(r[1]["ndcg"] for r in rows)
    for name, m in rows:
        star = "  ★" if m["ndcg"] == best else ""
        print(f"{name:<14}{m['recall']:>10.3f}{m['prec']:>9.3f}{m['mrr']:>8.3f}{m['ndcg']:>9.3f}{m['map']:>8.3f}{star}")
    print("-" * 66)
    print("  ★ = best nDCG@k. Higher is better on every metric (range 0–1).")


if __name__ == "__main__":
    run()
