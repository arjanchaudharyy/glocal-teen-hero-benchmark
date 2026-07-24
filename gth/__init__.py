"""
gth — Glocal Teen Hero at-selection benchmark.

An open, reproducible benchmark of every Nepal Glocal Teen Hero honoree
(2015-2025), scored on the record they held at selection, with a small RAG
retrieval layer over the corpus.
"""
from .corpus import Corpus, Hero, load
from .rubric import WEIGHTS, LABELS, DIMENSIONS, weighted_total
from .scoring import rank_all, cohort_stats, percentile_vs, rank_of, verdict
from .rag import build_index, similar, ask

__version__ = "1.0.0"
__all__ = [
    "Corpus", "Hero", "load", "WEIGHTS", "LABELS", "DIMENSIONS", "weighted_total",
    "rank_all", "cohort_stats", "percentile_vs", "rank_of", "verdict",
    "build_index", "similar", "ask", "__version__",
]
