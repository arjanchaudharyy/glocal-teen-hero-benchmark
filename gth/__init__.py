"""
gth — Glocal Teen Hero corpus, at-selection rubric, and retrieval benchmark.

An open, reproducible corpus of every Nepal Glocal Teen Hero honoree
(2015-2025), scored via a documented rubric on the record they held at
selection, plus a full IR retrieval benchmark over the same corpus: 5
retrievers (BM25, TF-IDF, QLM, char n-gram, optional MiniLM), 3 fusion
strategies, RM3 expansion, MMR re-ranking. "Benchmark" here refers
specifically to the retrieval evaluation harness (gth.eval) — a real test
collection with gold queries and compared systems — not the rubric scoring,
which is a documented methodology applied to a fixed cohort, not a
standardized task multiple parties are measured against.

The default config is not a guess — `gth.eval.cross_validate` k-fold
cross-validates every candidate, and char-ngram alone generalizes best on
this corpus, beating the fancier hybrid on held-out queries. The full
ensemble is real and available (`hybrid=True`); it just doesn't win the
honest comparison, and we ship what the evidence says.
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

__version__ = "1.5.0"
__all__ = [
    "Corpus", "Hero", "load", "WEIGHTS", "LABELS", "DIMENSIONS", "weighted_total",
    "rank_all", "cohort_stats", "percentile_vs", "rank_of", "verdict",
    "HybridRetriever", "TfidfIndex", "BM25Index", "QLMIndex", "CharNGramIndex", "Retrieval",
    "reciprocal_rank_fusion", "comb_sum", "mmr", "rm3", "prf_expand", "tokenize", "char_ngrams",
    "build_index", "similar", "ask", "evaluation", "__version__",
]
