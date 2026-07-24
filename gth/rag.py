"""
High-level RAG facade over the retrieval engine.

    build_index(corpus)  -> HybridRetriever
    similar(name, k=5)    -> nearest honorees to a given honoree
    ask(query, k=5)       -> grounded, cite-by-name answer block

Default config is char n-gram retrieval, the cross-validated winner (see
gth/eval.py). Pass hybrid=True for the full BM25+TF-IDF+char ensemble with
RRF fusion, RM3 expansion, and MMR re-ranking, real and available, just not
the default because it does not out-generalize the simple retriever here.
"""
from __future__ import annotations

import weakref
from typing import Any

from .corpus import Corpus, load
from .retrieval import HybridRetriever, Retrieval

# WeakKeyDictionary, not a plain dict keyed by id(corpus): id() values are
# reused by CPython after garbage collection, so an id-keyed cache can
# silently return a retriever built from a different, already-deallocated
# corpus. A WeakKeyDictionary keys on the object itself (via a weak
# reference) and evicts its entry automatically when that corpus is
# collected, so it can never return a stale answer for a live object and
# never grows unboundedly for corpora that get discarded.
_CACHE: weakref.WeakKeyDictionary[Corpus, HybridRetriever] = weakref.WeakKeyDictionary()


def build_index(corpus: Corpus | None = None) -> HybridRetriever:
    """Build (and memoize) a hybrid retriever."""
    corpus = corpus or load()
    idx = _CACHE.get(corpus)
    if idx is None:
        idx = _CACHE[corpus] = HybridRetriever(corpus)
    return idx


def _hybrid_kwargs(index: HybridRetriever) -> dict[str, Any]:
    return dict(methods=index.full_hybrid_methods, fusion="rrf", expand="rm3", rerank="mmr")


def similar(name: str, year: int | None = None, k: int = 5, corpus: Corpus | None = None,
            index: HybridRetriever | None = None, hybrid: bool = False, **kw: Any) -> list[Retrieval]:
    """Nearest honorees to `name`. Pass `year` if `name` matches more than one
    honoree (Corpus.get raises AmbiguousHeroError otherwise)."""
    corpus = corpus or load()
    index = index or build_index(corpus)
    hero = corpus.get(name, year)
    if hero is None:
        raise KeyError(f"no honoree matching {name!r}")
    if hybrid:
        kw = {**_hybrid_kwargs(index), **kw}
    return index.query(hero.doc, k=k, exclude=hero.key, **kw)


def ask(query: str, k: int = 5, corpus: Corpus | None = None,
        index: HybridRetriever | None = None, hybrid: bool = False, **kw: Any) -> str:
    """Retrieval-augmented answer. Defaults to the cross-validated best config
    (char n-gram); pass hybrid=True for the full BM25+TF-IDF+char+RM3+MMR stack."""
    corpus = corpus or load()
    index = index or build_index(corpus)
    if hybrid:
        kw = {**_hybrid_kwargs(index), **kw}
    hits = index.query(query, k=k, **kw)
    label = index.describe(kw.get("methods"), kw.get("fusion", "rrf"), kw.get("expand"), kw.get("rerank"))
    lines = [f'Top {len(hits)} honorees matching "{query}" (config: {label}):', ""]
    for n, r in enumerate(hits, 1):
        prov = ", ".join(f"{m}#{rk}" for m, rk in sorted(r.sources.items()))
        lines.append(f"{n}. {r.hero.name} ({r.hero.year}, {r.hero.award}) - score={r.score}")
        lines.append(f"   {r.hero.then[:110]}")
        if prov:
            lines.append(f"   retrieved by {prov}")
    return "\n".join(lines)
