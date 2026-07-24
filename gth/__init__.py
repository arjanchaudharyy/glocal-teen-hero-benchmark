"""
gth — Glocal Teen Hero at-selection benchmark + retrieval engine.

An open, reproducible benchmark of every Nepal Glocal Teen Hero honoree
(2015-2025), scored on the record they held at selection, with a hybrid
retrieval (RAG) layer over the corpus: BM25 + TF-IDF (+ optional MiniLM),
fused with Reciprocal Rank Fusion and re-ranked with MMR, plus an evaluation
harness (Recall@k / MRR / nDCG / MAP).
"""
from .corpus import Corpus, Hero, load
from .rubric import WEIGHTS, LABELS, DIMENSIONS, weighted_total
from .scoring import rank_all, cohort_stats, percentile_vs, rank_of, verdict
from .retrieval import (
    HybridRetriever, TfidfIndex, BM25Index, QLMIndex, CharNGramIndex, Retrieval,
    reciprocal_rank_fusion, comb_sum, mmr, rm3, prf_expand, tokenize, char_ngrams,
)
from .rag import build_index, similar, ask
from . import eval as evaluation

__version__ = "1.2.0"
__all__ = [
    "Corpus", "Hero", "load", "WEIGHTS", "LABELS", "DIMENSIONS", "weighted_total",
    "rank_all", "cohort_stats", "percentile_vs", "rank_of", "verdict",
    "HybridRetriever", "TfidfIndex", "BM25Index", "QLMIndex", "CharNGramIndex", "Retrieval",
    "reciprocal_rank_fusion", "comb_sum", "mmr", "rm3", "prf_expand", "tokenize", "char_ngrams",
    "build_index", "similar", "ask", "evaluation", "__version__",
]
