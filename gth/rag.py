"""
Retrieval over the honoree corpus.

Each honoree's at-selection record is a document; we build a vector index and
retrieve the most similar honorees to a free-text query or another honoree.
This powers `similar` and `ask` (retrieval-augmented answers).

Backends (auto-selected, graceful degradation):
  * "tfidf"      -> pure-Python TF-IDF + cosine. Zero dependencies. Default.
  * "embeddings" -> sentence-transformers ("all-MiniLM-L6-v2") if installed.
Set GTH_BACKEND=embeddings to prefer embeddings; otherwise TF-IDF is used.

The index is deliberately small and interpretable — the corpus is ~192 docs, so
exact cosine over sparse vectors is instant and needs no ANN structure.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Corpus, Hero, load

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for with on at from by is was were as his her their "
    "he she they it its him them who whom nepal year built ran led".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Retrieval:
    hero: Hero
    score: float


class TfidfIndex:
    """Sparse TF-IDF vector index with cosine similarity. Pure stdlib."""

    backend = "tfidf"

    def __init__(self, heroes: Sequence[Hero]):
        self.heroes = list(heroes)
        toks = [tokenize(h.doc) for h in self.heroes]
        n = len(toks)
        df: Counter = Counter()
        for tk in toks:
            df.update(set(tk))
        # smoothed idf
        self.idf: Dict[str, float] = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.vecs: List[Dict[str, float]] = [self._vec(tk) for tk in toks]

    def _vec(self, tk: Sequence[str]) -> Dict[str, float]:
        if not tk:
            return {}
        tf = Counter(tk)
        v = {w: (c / len(tk)) * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def _query_vec(self, text: str) -> Dict[str, float]:
        return self._vec(tokenize(text))

    def query(self, text: str, k: int = 5, exclude: Optional[str] = None) -> List[Retrieval]:
        q = self._query_vec(text)
        out: List[Tuple[int, float]] = []
        for i, v in enumerate(self.vecs):
            if exclude and self.heroes[i].key == exclude:
                continue
            small, big = (q, v) if len(q) < len(v) else (v, q)
            s = sum(val * big.get(w, 0.0) for w, val in small.items())
            out.append((i, s))
        out.sort(key=lambda t: t[1], reverse=True)
        return [Retrieval(self.heroes[i], round(s, 4)) for i, s in out[:k]]


def _try_embedding_index(heroes):
    """Return an embeddings-backed index if sentence-transformers is available, else None."""
    try:
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    class EmbeddingIndex:
        backend = "embeddings"

        def __init__(self, heroes):
            import numpy as np
            self.heroes = list(heroes)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            E = self.model.encode([h.doc for h in self.heroes], normalize_embeddings=True)
            self.E = np.asarray(E)

        def query(self, text, k=5, exclude=None):
            import numpy as np
            q = self.model.encode([text], normalize_embeddings=True)[0]
            sims = self.E @ q
            order = np.argsort(-sims)
            res = []
            for i in order:
                h = self.heroes[int(i)]
                if exclude and h.key == exclude:
                    continue
                res.append(Retrieval(h, round(float(sims[int(i)]), 4)))
                if len(res) >= k:
                    break
            return res

    return EmbeddingIndex(heroes)


def build_index(corpus: Optional[Corpus] = None, backend: str = "auto"):
    corpus = corpus or load()
    prefer = backend if backend != "auto" else os.environ.get("GTH_BACKEND", "tfidf")
    if prefer == "embeddings":
        idx = _try_embedding_index(corpus.heroes)
        if idx is not None:
            return idx
    return TfidfIndex(corpus.heroes)


def similar(name: str, k: int = 5, corpus: Optional[Corpus] = None, index=None) -> List[Retrieval]:
    corpus = corpus or load()
    h = corpus.get(name)
    if h is None:
        raise KeyError(f"unknown honoree: {name!r}")
    index = index or build_index(corpus)
    return index.query(h.doc, k=k, exclude=h.key)


def ask(query: str, k: int = 5, corpus: Optional[Corpus] = None, index=None) -> str:
    """Retrieval-augmented answer: grounded, extractive, cite-by-name.

    This returns the retrieved evidence formatted as an answer. To make it
    generative, pass the same context block to any LLM — the retrieval layer is
    the part that keeps it grounded in the real corpus.
    """
    corpus = corpus or load()
    index = index or build_index(corpus)
    hits = index.query(query, k=k)
    lines = [f'Top {len(hits)} honorees matching "{query}" (backend: {index.backend}):', ""]
    for i, r in enumerate(hits, 1):
        lines.append(f"{i}. {r.hero.name} ({r.hero.year}, {r.hero.award}) · sim={r.score}")
        lines.append(f"   {r.hero.then}")
    return "\n".join(lines)
