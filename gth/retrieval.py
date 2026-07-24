"""
Retrieval engine for the honoree corpus — a full, from-scratch IR stack.

Retrievers (each: .search(query) -> [(doc_idx, score)] sorted desc):
  * TfidfIndex    — smoothed-IDF TF-IDF, L2-normalized sparse vectors, cosine.
  * BM25Index     — Okapi BM25 (k1, b tunable).
  * QLMIndex      — query-likelihood language model, Dirichlet smoothing.
  * CharNGramIndex— character 3–4-gram TF-IDF cosine (typo / transliteration robust).
  * DenseIndex    — optional MiniLM sentence-embeddings (sentence-transformers).

Fusion (combine heterogeneous retrievers):
  * reciprocal_rank_fusion — rank-level, scale-free, weightable (RRF).
  * comb_sum / comb_mnz    — score-level, min-max normalized (CombSUM / CombMNZ).

Expansion & re-ranking:
  * rm3            — RM3 relevance-model pseudo-relevance feedback.
  * prf_expand     — lightweight Rocchio-style expansion (kept for comparison).
  * mmr            — Maximal Marginal Relevance diversity re-ranking.
  * cross-encoder  — optional neural re-ranker over the fused top-k.

The corpus is ~192 short docs, so exact search over sparse vectors is instant
and fully interpretable — no ANN index needed. Everything is deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Corpus, Hero, load

# ----------------------------------------------------------------------------- tokenizer
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for with on at from by is was were are be as its it he she "
    "they them his her their who whom this that these those i me my we our you your into "
    "over under out up down not no nor so than then also very more most such can will".split()
)


def _stem(w: str) -> str:
    """Conservative suffix stripping — bridges plural/tense without over-stemming."""
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


def char_ngrams(text: str, lo: int = 3, hi: int = 4) -> List[str]:
    s = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    grams: List[str] = []
    for tok in s.split():
        t = f"#{tok}#"
        for n in range(lo, hi + 1):
            grams.extend(t[i:i + n] for i in range(len(t) - n + 1))
    return grams


# ----------------------------------------------------------------------------- result type
@dataclass
class Retrieval:
    hero: Hero
    score: float
    sources: Dict[str, int] = field(default_factory=dict)  # method -> rank (1-based)
    snippet: str = ""


def _rank_map(ranked_idx: Sequence[int]) -> Dict[int, int]:
    return {idx: r for r, idx in enumerate(ranked_idx, 1)}


def _l2(v: Dict[str, float]) -> Dict[str, float]:
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    small, big = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * big.get(w, 0.0) for w, v in small.items())


# ----------------------------------------------------------------------------- TF-IDF
class TfidfIndex:
    name = "tfidf"

    def __init__(self, docs: Sequence[str], tok=tokenize):
        self._tok = tok
        toks = [tok(d) for d in docs]
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
        return _l2({w: (c / len(tk)) * self.idf.get(w, 0.0) for w, c in tf.items()})

    def search(self, query: str) -> List[Tuple[int, float]]:
        q = self._vec(self._tok(query))
        scored = [(i, _cos(q, v)) for i, v in enumerate(self.vecs)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


class CharNGramIndex(TfidfIndex):
    """TF-IDF cosine over character n-grams — robust to spelling / transliteration."""
    name = "char"

    def __init__(self, docs: Sequence[str]):
        super().__init__(docs, tok=char_ngrams)


# ----------------------------------------------------------------------------- BM25
class BM25Index:
    name = "bm25"

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.dl = [len(t) for t in self.tokens]
        n = len(self.tokens)
        self.avgdl = (sum(self.dl) / n) if n else 0.0
        df: Counter = Counter()
        for tk in self.tokens:
            df.update(set(tk))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(tk) for tk in self.tokens]

    def search(self, query: str) -> List[Tuple[int, float]]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            s, dl = 0.0, self.dl[i] or 1
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(w, 0.0) * (f * (self.k1 + 1)) / denom
            out.append((i, s))
        out.sort(key=lambda t: t[1], reverse=True)
        return out


# ----------------------------------------------------------------------------- Query-Likelihood LM
class QLMIndex:
    """Query-likelihood language model with Dirichlet smoothing."""
    name = "qlm"

    def __init__(self, docs: Sequence[str], mu: float = 200.0):
        self.mu = mu
        self.tokens = [tokenize(d) for d in docs]
        self.dl = [len(t) for t in self.tokens]
        self.tf = [Counter(tk) for tk in self.tokens]
        cf: Counter = Counter()
        for tk in self.tokens:
            cf.update(tk)
        self.total = sum(cf.values()) or 1
        self.pc = {w: c / self.total for w, c in cf.items()}

    def search(self, query: str) -> List[Tuple[int, float]]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            dl = self.dl[i]
            s = 0.0
            for w in q:
                pc = self.pc.get(w, 1e-9)
                s += math.log((tf.get(w, 0) + self.mu * pc) / (dl + self.mu))
            out.append((i, s))
        out.sort(key=lambda t: t[1], reverse=True)
        return out


# ----------------------------------------------------------------------------- dense (optional)
def _try_dense(docs: Sequence[str]):
    try:
        import numpy as np  # noqa
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
            return [(int(i), float(sims[int(i)])) for i in np.argsort(-sims)]

    return DenseIndex(docs)


def _try_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


# ----------------------------------------------------------------------------- fusion
def reciprocal_rank_fusion(rankings: Dict[str, Sequence[int]],
                           weights: Optional[Dict[str, float]] = None,
                           k: int = 60) -> List[Tuple[int, float]]:
    """RRF: score(d) = sum_r w_r / (k + rank_r(d)). Scale-free, weightable."""
    fused: Dict[int, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, idx in enumerate(ranked, 1):
            fused[idx] += w / (k + rank)
    return sorted(fused.items(), key=lambda t: t[1], reverse=True)


def _minmax(scored: Sequence[Tuple[int, float]]) -> Dict[int, float]:
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {i: (s - lo) / rng for i, s in scored}


def comb_sum(scored_lists: Dict[str, Sequence[Tuple[int, float]]],
             weights: Optional[Dict[str, float]] = None, mnz: bool = False) -> List[Tuple[int, float]]:
    """CombSUM / CombMNZ over min-max normalized scores."""
    acc: Dict[int, float] = defaultdict(float)
    hits: Dict[int, int] = defaultdict(int)
    for name, sl in scored_lists.items():
        w = (weights or {}).get(name, 1.0)
        for i, s in _minmax(sl).items():
            acc[i] += w * s
            if s > 0:
                hits[i] += 1
    if mnz:
        acc = {i: v * hits[i] for i, v in acc.items()}
    return sorted(acc.items(), key=lambda t: t[1], reverse=True)


# ----------------------------------------------------------------------------- expansion & rerank
def mmr(rel: Dict[int, float], cand_idx, tfidf: TfidfIndex, lam: float = 0.7, k: int = 10) -> List[int]:
    """Maximal Marginal Relevance re-ranking.

    `rel` maps candidate index -> its (fused) relevance in [0, 1]; diversity is
    measured with TF-IDF cosine between docs. This balances the *fused* ranking
    against redundancy, so every upstream retriever actually influences output.
    """
    selected: List[int] = []
    cands = list(cand_idx)
    while cands and len(selected) < k:
        best, best_s = cands[0], -1e18
        for i in cands:
            div = max((_cos(tfidf.vecs[i], tfidf.vecs[j]) for j in selected), default=0.0)
            s = lam * rel.get(i, 0.0) - (1 - lam) * div
            if s > best_s:
                best, best_s = i, s
        selected.append(best)
        cands.remove(best)
    return selected


def prf_expand(query: str, tfidf: TfidfIndex, top_m: int = 4, n_terms: int = 8) -> str:
    top = [i for i, _ in tfidf.search(query)[:top_m]]
    weights: Counter = Counter()
    for i in top:
        for w, x in tfidf.vecs[i].items():
            weights[w] += x
    return query + " " + " ".join(w for w, _ in weights.most_common(n_terms))


def rm3(query: str, bm25: BM25Index, top_m: int = 8, n_terms: int = 10,
        lam: float = 0.5, scale: int = 3) -> str:
    """RM3 relevance model: interpolate the query LM with a feedback LM from the
    top-m pseudo-relevant docs, then materialize as a boosted query string."""
    ranked = bm25.search(query)[:top_m]
    if not ranked:
        return query
    # relevance model P(t|R) weighted by (softmaxed) retrieval score
    scores = [s for _, s in ranked]
    lo = min(scores)
    ws = [math.exp(s - max(scores)) for s in scores]
    z = sum(ws) or 1.0
    fb: Dict[str, float] = defaultdict(float)
    for (i, _), w in zip(ranked, ws):
        dl = bm25.dl[i] or 1
        for term, f in bm25.tf[i].items():
            fb[term] += (w / z) * (f / dl)
    qter = tokenize(query)
    qmodel = {t: qter.count(t) / len(qter) for t in set(qter)} if qter else {}
    terms = set(fb) | set(qmodel)
    mixed = {t: (1 - lam) * qmodel.get(t, 0.0) + lam * fb.get(t, 0.0) for t in terms}
    # deterministic ordering: weight desc, then term asc (ties must not depend on
    # per-process set/hash ordering, or results become non-reproducible).
    top = sorted(mixed.items(), key=lambda kv: (-kv[1], kv[0]))[:n_terms]
    if not top:
        return query
    mx = top[0][1] or 1.0
    parts = [query]
    for term, wt in top:
        parts.extend([term] * max(1, round(scale * wt / mx)))
    return " ".join(parts)


def _snippet(text: str, query: str, width: int = 120) -> str:
    qset = set(tokenize(query))
    words = text.split()
    best, best_hits = 0, -1
    win = 14
    for i in range(max(1, len(words) - win + 1)):
        hits = sum(1 for w in words[i:i + win] if _stem(re.sub(r"[^a-z0-9]", "", w.lower())) in qset)
        if hits > best_hits:
            best_hits, best = hits, i
    seg = " ".join(words[best:best + win])
    return (("…" if best else "") + seg)[:width]


# ----------------------------------------------------------------------------- hybrid
# Weights + defaults tuned on the labeled gold set (see `python -m gth tune` and eval.py).
_DEFAULT_WEIGHTS = {"bm25": 1.0, "tfidf": 1.0, "qlm": 0.9, "dense": 1.1, "char": 0.3}


class HybridRetriever:
    def __init__(self, corpus: Corpus, use_dense: bool = False, pool: int = 50,
                 bm25_k1: float = 1.5, bm25_b: float = 0.4):
        self.corpus = corpus
        self.docs = [h.doc for h in corpus.heroes]
        self.tfidf = TfidfIndex(self.docs)
        self.bm25 = BM25Index(self.docs, k1=bm25_k1, b=bm25_b)
        self.qlm = QLMIndex(self.docs)
        self.char = CharNGramIndex(self.docs)
        self.dense = _try_dense(self.docs) if use_dense else None
        self._cross = None
        self.pool = pool
        self.weights = dict(_DEFAULT_WEIGHTS)
        self._all = {"bm25": self.bm25, "tfidf": self.tfidf, "qlm": self.qlm, "char": self.char}
        if self.dense:
            self._all["dense"] = self.dense
        self.default_methods = ["bm25", "tfidf", "char"] + (["dense"] if self.dense else [])
        self.backend = "hybrid(" + "+".join(self.default_methods) + ")"

    def _indices(self, methods):
        methods = methods or self.default_methods
        return {m: self._all[m] for m in methods if m in self._all}

    def query(self, text: str, k: int = 5, *, methods=None, fusion: str = "rrf",
              expand: Optional[str] = "rm3", rerank: Optional[str] = "mmr",
              exclude: Optional[str] = None, snippets: bool = False) -> List[Retrieval]:
        idxs = self._indices(methods)
        q = text
        if expand == "rm3":
            q = rm3(text, self.bm25)
        elif expand == "prf":
            q = prf_expand(text, self.tfidf)

        scored = {name: ix.search(q)[: self.pool] for name, ix in idxs.items()}
        rank_maps = {name: _rank_map([i for i, s in sl if s != 0]) for name, sl in scored.items()}

        if len(idxs) == 1:
            fused = list(scored[next(iter(idxs))])
        elif fusion == "rrf":
            fused = reciprocal_rank_fusion(
                {n: [i for i, s in sl if s != 0] for n, sl in scored.items()}, self.weights)
        elif fusion == "combsum":
            fused = comb_sum(scored, self.weights)
        elif fusion == "combmnz":
            fused = comb_sum(scored, self.weights, mnz=True)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        cand = [i for i, _ in fused]
        if exclude:
            cand = [i for i in cand if self.corpus.heroes[i].key != exclude]

        if rerank == "mmr" and cand:
            head = cand[: max(k * 4, 20)]
            rel = _minmax([(i, dict(fused).get(i, 0.0)) for i in head])
            order = mmr(rel, head, self.tfidf, k=k)
            cand = order + [i for i in cand if i not in order]
        elif rerank == "cross" and cand:
            cand = self._cross_rerank(text, cand, k)

        score_of = dict(fused)
        out = []
        for i in cand[:k]:
            h = self.corpus.heroes[i]
            srcs = {n: rm[i] for n, rm in rank_maps.items() if i in rm}
            snip = _snippet(h.then, text) if snippets else ""
            out.append(Retrieval(h, round(score_of.get(i, 0.0), 5), srcs, snip))
        return out

    def _cross_rerank(self, text, cand, k):
        if self._cross is None:
            self._cross = _try_cross_encoder() or False
        if not self._cross:
            return cand
        head = cand[: max(k * 5, 25)]
        pairs = [(text, self.corpus.heroes[i].then) for i in head]
        scores = self._cross.predict(pairs)
        order = [i for i, _ in sorted(zip(head, scores), key=lambda t: t[1], reverse=True)]
        return order + [i for i in cand if i not in set(order)]
