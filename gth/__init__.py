"""
gth: Glocal Teen Hero corpus, at-selection rubric, and retrieval benchmark.

An open, reproducible corpus of every Nepal Glocal Teen Hero honoree
(2015-2025), scored via a documented rubric on the record they held at
selection, plus a small IR retrieval benchmark over the same corpus: BM25,
TF-IDF, and character n-gram retrieval, RRF fusion, RM3 expansion, MMR
re-ranking. Kept deliberately small for a 192-document corpus rather than
stacked with methods that never won a comparison. "Benchmark" here refers
specifically to the retrieval evaluation harness (gth.eval), a real test
collection with gold queries and compared systems, not the rubric scoring,
which is a documented methodology applied to a fixed cohort, not a
standardized task multiple parties are measured against.

The default config is not a guess: `gth.eval.cross_validate` k-fold
cross-validates every candidate, and char n-gram alone generalizes best on
this corpus, beating the fancier hybrid on held-out queries in every fold,
under both flat and nested cross-validation. The full ensemble is real and
available (`hybrid=True`); it just does not win the honest comparison, so
it is not the default.
"""
from . import eval as evaluation
from .corpus import Corpus, Hero, load
from .eval import Metrics
from .rag import ask, build_index, similar
from .retrieval import (
    BM25Index,
    CharNGramIndex,
    HybridRetriever,
    Retrieval,
    TfidfIndex,
    char_ngrams,
    mmr,
    reciprocal_rank_fusion,
    rm3,
    tokenize,
)
from .rubric import DIMENSIONS, LABELS, WEIGHTS, weighted_total
from .scoring import cohort_stats, percentile_vs, rank_all, rank_of, verdict

__version__ = "1.7.1"
__all__ = [
    "Corpus", "Hero", "load", "WEIGHTS", "LABELS", "DIMENSIONS", "weighted_total",
    "rank_all", "cohort_stats", "percentile_vs", "rank_of", "verdict",
    "HybridRetriever", "TfidfIndex", "BM25Index", "CharNGramIndex", "Retrieval",
    "reciprocal_rank_fusion", "mmr", "rm3", "tokenize", "char_ngrams",
    "build_index", "similar", "ask", "evaluation", "Metrics", "__version__",
]
