"""
High-level RAG facade over the retrieval engine.

    build_index(corpus, backend="auto")  -> HybridRetriever
    similar(name, k=5)                    -> nearest honorees to a given honoree
    ask(query, k=5)                       -> grounded, cite-by-name answer block

The retriever fuses BM25 + TF-IDF (and MiniLM dense when available) with
Reciprocal Rank Fusion, then re-ranks for diversity with MMR. Everything is
deterministic and, in the default configuration, dependency-free.
"""
from __future__ import annotations

import os
from typing import List, Optional

from .corpus import Corpus, load
from .retrieval import HybridRetriever, Retrieval

_CACHE = {}


def build_index(corpus: Optional[Corpus] = None, backend: str = "auto") -> HybridRetriever:
    """Build (and memoize) a hybrid retriever.

    backend: "auto"/"hybrid"/"tfidf" -> dependency-free BM25+TF-IDF fusion.
             "embeddings"/"dense"     -> additionally load MiniLM if available.
             Overridable via the GTH_BACKEND environment variable.
    """
    backend = os.environ.get("GTH_BACKEND", backend)
    corpus = corpus or load()
    use_dense = backend in ("embeddings", "dense")
    key = (id(corpus), use_dense)
    if key not in _CACHE:
        _CACHE[key] = HybridRetriever(corpus, use_dense=use_dense)
    return _CACHE[key]


def similar(name: str, k: int = 5, corpus: Optional[Corpus] = None,
            index: Optional[HybridRetriever] = None, **kw) -> List[Retrieval]:
    corpus = corpus or load()
    index = index or build_index(corpus)
    hero = corpus.get(name)
    if hero is None:
        raise KeyError(f"no honoree matching {name!r}")
    return index.query(hero.doc, k=k, exclude=hero.key, **kw)


def ask(query: str, k: int = 5, corpus: Optional[Corpus] = None,
        index: Optional[HybridRetriever] = None, expand: bool = True,
        rerank: bool = True) -> str:
    corpus = corpus or load()
    index = index or build_index(corpus)
    hits = index.query(query, k=k, expand=expand, rerank=rerank)
    lines = [f'Top {len(hits)} honorees matching "{query}" (backend: {index.backend}):', ""]
    for n, r in enumerate(hits, 1):
        prov = ", ".join(f"{m}#{rk}" for m, rk in sorted(r.sources.items()))
        lines.append(f"{n}. {r.hero.name} ({r.hero.year}, {r.hero.award}) · rrf={r.score}")
        lines.append(f"   {r.hero.then[:110]}")
        if prov:
            lines.append(f"   ↳ retrieved by {prov}")
    return "\n".join(lines)
