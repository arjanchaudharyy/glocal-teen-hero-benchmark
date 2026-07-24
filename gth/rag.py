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

from typing import List, Optional

from .corpus import Corpus, load
from .retrieval import HybridRetriever, Retrieval

_CACHE = {}


def build_index(corpus: Optional[Corpus] = None) -> HybridRetriever:
    """Build (and memoize) a hybrid retriever."""
    corpus = corpus or load()
    key = id(corpus)
    if key not in _CACHE:
        _CACHE[key] = HybridRetriever(corpus)
    return _CACHE[key]


def _hybrid_kwargs(index: HybridRetriever) -> dict:
    return dict(methods=index.full_hybrid_methods, fusion="rrf", expand="rm3", rerank="mmr")


def similar(name: str, k: int = 5, corpus: Optional[Corpus] = None,
            index: Optional[HybridRetriever] = None, hybrid: bool = False, **kw) -> List[Retrieval]:
    corpus = corpus or load()
    index = index or build_index(corpus)
    hero = corpus.get(name)
    if hero is None:
        raise KeyError(f"no honoree matching {name!r}")
    if hybrid:
        kw = {**_hybrid_kwargs(index), **kw}
    return index.query(hero.doc, k=k, exclude=hero.key, **kw)


def ask(query: str, k: int = 5, corpus: Optional[Corpus] = None,
        index: Optional[HybridRetriever] = None, hybrid: bool = False, **kw) -> str:
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
