"""
Retrieval engine for the honoree corpus.

A small but real IR stack, dependency-free at the core:

  * Tokenizer         — lowercase, stopword + light-stemming pipeline.
  * TfidfIndex        — smoothed-IDF TF-IDF, L2-normalized sparse vectors, cosine.
  * BM25Index         — Okapi BM25 (k1=1.5, b=0.75).
  * DenseIndex        — optional MiniLM sentence-embeddings (sentence-transformers).
  * reciprocal_rank_fusion — rank-level fusion of any set of retrievers (RRF).
  * mmr               — Maximal Marginal Relevance re-ranking for diversity.
  * prf_expand        — Rocchio-style pseudo-relevance-feedback query expansion.
  * HybridRetriever   — fuses BM25 + TF-IDF (+ dense), optional PRF + MMR.

The corpus is ~192 docs, so exact search over sparse vectors is instant and
fully interpretable — no ANN index needed. Everything is deterministic.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Corpus, Hero, load

# ----------------------------------------------------------------------------- tokenizer
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for with on at from by is was were are be as its it he she "
    "they them his her their who whom this that these those i me my we our you your at into "
    "over under out up down not no nor so than then also very more most such can will".split()
)


def _stem(w: str) -> str:
    """Conservative suffix stripping — enough to bridge plural/tense, not aggressive."""
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# ----------------------------------------------------------------------------- result type
@dataclass
class Retrieval:
    hero: Hero
    score: float
    sources: Dict[str, int] = field(default_factory=dict)  # method -> rank (1-based)


def _rank_map(ranked_idx: Sequence[int]) -> Dict[int, int]:
    return {idx: r for r, idx in enumerate(ranked_idx, 1)}


# ----------------------------------------------------------------------------- TF-IDF
class TfidfIndex:
    name = "tfidf"

    def __init__(self, docs: Sequence[str]):
        toks = [tokenize(d) for d in docs]
        n = len(toks)
        df: Counter = Counter()
        for tk in toks:
            df.update(set(tk))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.vecs = [self._vec(tk) for tk in toks]

    def _vec(self, tk: Sequence[str]) -> Dict[str, float]:
        if not tk:
            return {}
        tf = Counter(tk)
        v = {w: (c / len(tk)) * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    @staticmethod
    def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        small, big = (a, b) if len(a) < len(b) else (b, a)
        return sum(v * big.get(w, 0.0) for w, v in small.items())

    def search(self, query: str) -> List[Tuple[int, float]]:
        q = self._vec(tokenize(query))
        scored = [(i, self.cosine(q, v)) for i, v in enumerate(self.vecs)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


# ----------------------------------------------------------------------------- BM25
class BM25Index:
    name = "bm25"

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.dl = [len(t) for t in self.toks]
        n = len(self.toks)
        self.avgdl = (sum(self.dl) / n) if n else 0.0
        df: Counter = Counter()
        for tk in self.toks:
            df.update(set(tk))
        # Okapi BM25 idf (with +1 to keep it non-negative)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(tk) for tk in self.toks]

    def search(self, query: str) -> List[Tuple[int, float]]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.dl[i] or 1
            for w in q:
                if w not in tf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(w, 0.0) * (f * (self.k1 + 1)) / denom
            scored.append((i, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


# ----------------------------------------------------------------------------- dense (optional)
def _try_dense(docs: Sequence[str]):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    class DenseIndex:
        name = "dense"

        def __init__(self, docs):
            import numpy as np
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.E = np.asarray(self.model.encode(list(docs), normalize_embeddings=True))

        def search(self, query):
            import numpy as np
            q = self.model.encode([query], normalize_embeddings=True)[0]
            sims = self.E @ q
            order = np.argsort(-sims)
            return [(int(i), float(sims[int(i)])) for i in order]

    return DenseIndex(docs)


# ----------------------------------------------------------------------------- fusion / rerank / expansion
def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """RRF: fused_score(d) = sum_i 1 / (k + rank_i(d)). Robust, scale-free rank fusion."""
    fused: Dict[int, float] = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda t: t[1], reverse=True)
    return out


def mmr(query_vec, cand_idx, tfidf: TfidfIndex, lam: float = 0.7, k: int = 10) -> List[int]:
    """Maximal Marginal Relevance re-ranking for diversity (uses TF-IDF vectors)."""
    selected: List[int] = []
    cands = list(cand_idx)
    rel = {i: tfidf.cosine(query_vec, tfidf.vecs[i]) for i in cands}
    while cands and len(selected) < k:
        best, best_s = None, -1e9
        for i in cands:
            div = max((tfidf.cosine(tfidf.vecs[i], tfidf.vecs[j]) for j in selected), default=0.0)
            s = lam * rel[i] - (1 - lam) * div
            if s > best_s:
                best, best_s = i, s
        selected.append(best)
        cands.remove(best)
    return selected


def prf_expand(query: str, tfidf: TfidfIndex, top_m: int = 4, n_terms: int = 8) -> str:
    """Pseudo-relevance feedback: append the strongest terms from the top-m docs (Rocchio-lite)."""
    top = [i for i, _ in tfidf.search(query)[:top_m]]
    weights: Counter = Counter()
    for i in top:
        for w, x in tfidf.vecs[i].items():
            weights[w] += x
    extra = [w for w, _ in weights.most_common(n_terms)]
    return query + " " + " ".join(extra)


# ----------------------------------------------------------------------------- hybrid
class HybridRetriever:
    def __init__(self, corpus: Corpus, use_dense: bool = False, pool: int = 50):
        self.corpus = corpus
        self.docs = [h.doc for h in corpus.heroes]
        self.tfidf = TfidfIndex(self.docs)
        self.bm25 = BM25Index(self.docs)
        self.dense = _try_dense(self.docs) if use_dense else None
        self.pool = pool
        methods = ["bm25", "tfidf"] + (["dense"] if self.dense else [])
        self.backend = "hybrid(" + "+".join(methods) + ")"

    def _indices(self, methods: Optional[Sequence[str]]):
        m = {"bm25": self.bm25, "tfidf": self.tfidf}
        if self.dense:
            m["dense"] = self.dense
        if methods:
            return {k: m[k] for k in methods if k in m}
        return m

    def query(
        self, text: str, k: int = 5, *, methods: Optional[Sequence[str]] = None,
        expand: bool = False, rerank: bool = True, exclude: Optional[str] = None,
    ) -> List[Retrieval]:
        idxs = self._indices(methods)
        q = text
        if expand:
            q = prf_expand(text, self.tfidf)

        rank_lists, rank_maps = [], {}
        for name, ix in idxs.items():
            ranked = [i for i, s in ix.search(q)[: self.pool] if s > 0]
            rank_lists.append(ranked)
            rank_maps[name] = _rank_map(ranked)

        if len(rank_lists) == 1:
            fused = [(i, 0.0) for i in rank_lists[0]]  # single retriever
            # keep original scores for single-method transparency
            base = dict(idxs[list(idxs)[0]].search(q))
            fused = sorted(((i, base.get(i, 0.0)) for i, _ in fused), key=lambda t: t[1], reverse=True)
        else:
            fused = reciprocal_rank_fusion(rank_lists)

        cand = [i for i, _ in fused]
        if exclude:
            cand = [i for i in cand if self.corpus.heroes[i].key != exclude]

        if rerank and cand:
            qv = self.tfidf._vec(tokenize(q))
            order = mmr(qv, cand[: max(k * 4, 20)], self.tfidf, k=k)
            cand = order + [i for i in cand if i not in order]

        score_of = dict(fused)
        out = []
        for i in cand[:k]:
            srcs = {name: rm[i] for name, rm in rank_maps.items() if i in rm}
            out.append(Retrieval(self.corpus.heroes[i], round(score_of.get(i, 0.0), 5), srcs))
        return out
