"""
Retrieval engine for the honoree corpus.

Deliberately small. At 192 short documents, this does not need five
retrievers, three fusion strategies, and a neural reranker; an earlier
version of this file had all of that, and most of it never won a single
comparison in gth/eval.py. What is here is what actually earned its place:

  * TfidfIndex     - smoothed-IDF TF-IDF, L2-normalized sparse vectors, cosine.
  * BM25Index      - Okapi BM25 (k1, b tunable at search time, not just at
                     construction; see HybridRetriever.set_bm25_params).
  * CharNGramIndex - character 3-4-gram TF-IDF cosine (typo / transliteration
                     robust, and the strongest single retriever on this corpus).
  * reciprocal_rank_fusion - the one fusion strategy tested and kept (RRF).
  * rm3            - RM3 relevance-model pseudo-relevance feedback.
  * mmr            - Maximal Marginal Relevance diversity re-ranking.

Every result carries provenance (which method(s) surfaced it, and at what
rank), so `describe()` on HybridRetriever always reflects what actually ran.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .corpus import Corpus, Hero

# ----------------------------------------------------------------------------- tokenizer
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in for with on at from by is was were are be as its it he she "
    "they them his her their who whom this that these those i me my we our you your into "
    "over under out up down not no nor so than then also very more most such can will".split()
)


def _stem(w: str) -> str:
    """Conservative suffix stripping, bridges plural/tense without over-stemming."""
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def char_ngrams(text: str, lo: int = 3, hi: int = 4) -> list[str]:
    s = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    grams: list[str] = []
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
    sources: dict[str, int] = field(default_factory=dict)  # method -> rank (1-based)
    snippet: str = ""


def _rank_map(ranked_idx: Sequence[int]) -> dict[int, int]:
    return {idx: r for r, idx in enumerate(ranked_idx, 1)}


def _l2(v: dict[str, float]) -> dict[str, float]:
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    small, big = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * big.get(w, 0.0) for w, v in small.items())


# ----------------------------------------------------------------------------- TF-IDF
class TfidfIndex:
    """TF-IDF cosine index.

    Pass either raw `docs` (tokenized internally with `tok`) or a
    precomputed `tokens` list. HybridRetriever tokenizes once and shares
    the result with BM25Index rather than each index retokenizing the same
    documents independently.

    search() uses an inverted index (term -> posting list of doc indices)
    so it only scores documents that share at least one term with the
    query, instead of computing cosine similarity against every document in
    the corpus on every call. A document with zero term overlap always
    scores exactly 0.0 under cosine similarity, so this changes nothing
    about which documents are considered relevant, only how many
    zero-relevance documents get scored for no reason.
    """
    name = "tfidf"

    def __init__(self, docs: Sequence[str] | None = None, tok=tokenize,
                 *, tokens: Sequence[list[str]] | None = None):
        self._tok = tok
        if tokens is not None:
            toks = list(tokens)
        else:
            if docs is None:
                raise ValueError("must pass either docs or tokens")
            toks = [tok(d) for d in docs]
        n = len(toks)
        df: Counter = Counter()
        for tk in toks:
            df.update(set(tk))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.vecs = [self._vec(tk) for tk in toks]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(self.vecs):
            for w in v:
                self.postings[w].append(i)

    def _vec(self, tk: Sequence[str]) -> dict[str, float]:
        if not tk:
            return {}
        tf = Counter(tk)
        return _l2({w: (c / len(tk)) * self.idf.get(w, 0.0) for w, c in tf.items()})

    def search(self, query: str) -> list[tuple[int, float]]:
        q = self._vec(self._tok(query))
        candidates: set = set()
        for w in q:
            candidates.update(self.postings.get(w, ()))
        scored = [(i, _cos(q, self.vecs[i])) for i in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


class CharNGramIndex(TfidfIndex):
    """TF-IDF cosine over character n-grams, robust to spelling and transliteration.
    Uses a different tokenizer than the word-level indices, so it cannot share
    their tokenization pass."""
    name = "char"

    def __init__(self, docs: Sequence[str]):
        super().__init__(docs, tok=char_ngrams)


# ----------------------------------------------------------------------------- BM25
class BM25Index:
    """Okapi BM25, scored via an inverted index.

    k1/b are read at search() time from self.k1/self.b, which callers may
    mutate directly (or via HybridRetriever.set_bm25_params) between calls.
    Tokenization, document frequencies, IDF, and the postings lists (the
    expensive, O(corpus) part of construction) do not depend on k1/b at
    all, so trying many (k1, b) combinations, as gth/eval.py's grid
    searches do, never requires rebuilding this index: only search()
    re-runs, and it only visits documents that share at least one query
    term (via the postings lists) rather than scanning the whole corpus.
    """
    name = "bm25"

    def __init__(self, docs: Sequence[str] | None = None, k1: float = 1.5, b: float = 0.75,
                 *, tokens: Sequence[list[str]] | None = None):
        self.k1, self.b = k1, b
        if tokens is not None:
            self.tokens = list(tokens)
        else:
            if docs is None:
                raise ValueError("must pass either docs or tokens")
            self.tokens = [tokenize(d) for d in docs]
        self.dl = [len(t) for t in self.tokens]
        n = len(self.tokens)
        self.avgdl = (sum(self.dl) / n) if n else 0.0
        df: Counter = Counter()
        for tk in self.tokens:
            df.update(set(tk))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(tk) for tk in self.tokens]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for term, f in tf.items():
                self.postings[term].append((i, f))

    def search(self, query: str) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings.get(w, ()):
                dl = self.dl[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        out = list(scores.items())
        out.sort(key=lambda t: t[1], reverse=True)
        return out


# ----------------------------------------------------------------------------- fusion
def reciprocal_rank_fusion(rankings: dict[str, Sequence[int]],
                           weights: dict[str, float] | None = None,
                           k: int = 60) -> list[tuple[int, float]]:
    """RRF: score(d) = sum_r w_r / (k + rank_r(d)). Scale-free, weightable."""
    fused: dict[int, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, idx in enumerate(ranked, 1):
            fused[idx] += w / (k + rank)
    return sorted(fused.items(), key=lambda t: t[1], reverse=True)


def _minmax(scored: Sequence[tuple[int, float]]) -> dict[int, float]:
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {i: (s - lo) / rng for i, s in scored}


# ----------------------------------------------------------------------------- expansion & rerank
def mmr(rel: dict[int, float], cand_idx, tfidf: TfidfIndex, lam: float = 0.7, k: int = 10) -> list[int]:
    """Maximal Marginal Relevance re-ranking.

    `rel` maps candidate index to its (fused) relevance in [0, 1]; diversity is
    measured with TF-IDF cosine between docs. This balances the fused ranking
    against redundancy, so every upstream retriever actually influences output.

    Maintains a running max-similarity-to-selected per candidate, updated
    incrementally against only the most recently selected item each round,
    instead of recomputing max cosine similarity against every selected item
    from scratch on every round. That naive version is O(k^2 * |candidates|)
    and was, empirically (profiled during gth/eval.py's grid searches), the
    single largest cost in this whole codebase, over 80% of wall time on a
    corpus of only 192 documents. This version is O(k * |candidates|) and
    returns mathematically identical output.
    """
    cands = list(cand_idx)
    max_sim = {i: 0.0 for i in cands}
    selected: list[int] = []
    while cands and len(selected) < k:
        best, best_s = cands[0], -1e18
        for i in cands:
            s = lam * rel.get(i, 0.0) - (1 - lam) * max_sim[i]
            if s > best_s:
                best, best_s = i, s
        selected.append(best)
        cands.remove(best)
        for i in cands:
            sim = _cos(tfidf.vecs[best], tfidf.vecs[i])
            if sim > max_sim[i]:
                max_sim[i] = sim
    return selected


def rm3(query: str, bm25: BM25Index, top_m: int = 8, n_terms: int = 10,
        lam: float = 0.5, scale: int = 3) -> str:
    """RM3 relevance model: interpolate the query LM with a feedback LM from the
    top-m pseudo-relevant docs, then materialize as a boosted query string."""
    ranked = bm25.search(query)[:top_m]
    if not ranked:
        return query
    # relevance model P(t|R), weighted by softmaxed retrieval score
    scores = [s for _, s in ranked]
    ws = [math.exp(s - max(scores)) for s in scores]
    z = sum(ws) or 1.0
    fb: dict[str, float] = defaultdict(float)
    for (i, _), w in zip(ranked, ws):
        dl = bm25.dl[i] or 1
        for term, f in bm25.tf[i].items():
            fb[term] += (w / z) * (f / dl)
    qter = tokenize(query)
    qmodel = {t: qter.count(t) / len(qter) for t in set(qter)} if qter else {}
    terms = set(fb) | set(qmodel)
    mixed = {t: (1 - lam) * qmodel.get(t, 0.0) + lam * fb.get(t, 0.0) for t in terms}
    # deterministic ordering: weight desc, then term asc. Ties must not depend
    # on per-process set/hash ordering, or results become non-reproducible.
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
    return (("..." if best else "") + seg)[:width]


# ----------------------------------------------------------------------------- hybrid
# Weight tuned on the labeled gold set (see gth/eval.py). BM25 and TF-IDF get
# equal weight; char n-gram is down-weighted in fusion because it dominates on
# its own and would otherwise drown out the other two.
_DEFAULT_WEIGHTS = {"bm25": 1.0, "tfidf": 1.0, "char": 0.3}


class HybridRetriever:
    def __init__(self, corpus: Corpus, pool: int = 50, bm25_k1: float = 1.5, bm25_b: float = 0.4):
        self.corpus = corpus
        self.docs = [h.doc for h in corpus.heroes]
        # Word-level tokenization happens once and is shared between TF-IDF
        # and BM25 (they used to each retokenize the same documents
        # independently). Char n-grams use a different tokenizer and need
        # their own pass.
        word_tokens = [tokenize(d) for d in self.docs]
        self.tfidf = TfidfIndex(tokens=word_tokens)
        self.bm25 = BM25Index(tokens=word_tokens, k1=bm25_k1, b=bm25_b)
        self.char = CharNGramIndex(self.docs)
        self.pool = pool
        self.weights = dict(_DEFAULT_WEIGHTS)
        self._all = {"bm25": self.bm25, "tfidf": self.tfidf, "char": self.char}
        # Default is the cross-validated winner, not a guess: `python -m gth cv`
        # (and, closing a leakage path, `python -m gth ncv`) both pick char
        # n-gram alone over the full BM25+TF-IDF+char+RM3+MMR ensemble on
        # held-out queries. The ensemble is real and available (hybrid=True);
        # it just doesn't generalize better here, so it isn't the default.
        self.default_methods = ["char"]
        self.full_hybrid_methods = ["bm25", "tfidf", "char"]

    def set_bm25_params(self, k1: float, b: float) -> None:
        """Change BM25's k1/b without rebuilding the index. Cheap: tokenization,
        document frequencies, and IDF don't depend on k1/b, only search() does.
        Grid-searching (k1, b), as gth/eval.py's tune/cv/ncv all do, should call
        this on one shared HybridRetriever rather than constructing a new one
        per grid point."""
        self.bm25.k1 = k1
        self.bm25.b = b

    def _indices(self, methods):
        methods = methods or self.default_methods
        unknown = [m for m in methods if m not in self._all]
        if unknown:
            raise ValueError(f"unknown retrieval method(s) {unknown!r}; valid: {sorted(self._all)}")
        return {m: self._all[m] for m in methods}

    def describe(self, methods=None, fusion: str = "rrf", expand=None, rerank=None) -> str:
        """Human-readable config label. Reflects the actual params of a call."""
        methods = methods or self.default_methods
        tag = "+".join(methods) if len(methods) == 1 else f"hybrid({'+'.join(methods)},{fusion})"
        if expand:
            tag += f"+{expand}"
        if rerank:
            tag += f"+{rerank}"
        return tag

    def query(self, text: str, k: int = 5, *, methods=None, fusion: str = "rrf",
              expand: str | None = None, rerank: str | None = None,
              exclude: str | None = None, snippets: bool = False,
              pool: int | None = None) -> list[Retrieval]:
        """
        pool: how many top hits each individual retriever contributes to
        fusion before RRF combines them (default self.pool=50). At this
        corpus's size (192 docs) that's over a quarter of the corpus, so
        truncation essentially never affects recall; if this corpus grows,
        a small pool can silently cap recall and should be raised.
        """
        idxs = self._indices(methods)
        pool = self.pool if pool is None else pool
        q = rm3(text, self.bm25) if expand == "rm3" else text

        scored = {name: ix.search(q)[:pool] for name, ix in idxs.items()}
        rank_maps = {name: _rank_map([i for i, s in sl if s != 0]) for name, sl in scored.items()}

        if len(idxs) == 1:
            fused = list(scored[next(iter(idxs))])
        elif fusion == "rrf":
            fused = reciprocal_rank_fusion(
                {n: [i for i, s in sl if s != 0] for n, sl in scored.items()}, self.weights)
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

        score_of = dict(fused)
        out = []
        for i in cand[:k]:
            h = self.corpus.heroes[i]
            srcs = {n: rm[i] for n, rm in rank_maps.items() if i in rm}
            snip = _snippet(h.then, text) if snippets else ""
            out.append(Retrieval(h, round(score_of.get(i, 0.0), 5), srcs, snip))
        return out
