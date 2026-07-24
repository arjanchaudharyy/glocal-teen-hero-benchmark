# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## 1.7.0

### Fixed
- **Real bug**: `HybridRetriever`'s single-retriever query path (the shipped default) silently padded results with zero-relevance documents when fewer than `k` genuine matches existed, presenting them as legitimate top-k hits. Fixed as a byproduct of the inverted-index rewrite below; regression test added.
- **Real bug**: `cohort_stats()`'s median used `vals[len(vals)//2]`, which is wrong for even-length cohorts (Finalist n=54, 20under20 n=126 are both even). Now averages the two middle values correctly.
- Module-level `assert` enforcing the rubric-weights-sum-to-1.0 invariant replaced with an explicit `raise`; `assert` is stripped under `python -O`, silently disabling the check.
- `HybridRetriever` now raises `ValueError` on an unrecognized retrieval method name instead of silently returning an empty result list.
- `Corpus.get()` was an O(n) linear scan on every call; now backed by a dict index built once at load time (year-qualified lookups are O(1)).
- `gth.rag.build_index`'s cache was keyed by `id(corpus)`, which CPython can reuse after garbage collection, and could silently return a stale retriever for a different, deallocated corpus. Replaced with a `WeakKeyDictionary` (`Corpus` made `eq=False`/identity-hashable to support this).
- Removed a stale `[project.optional-dependencies] embeddings` extra in `pyproject.toml` referencing a dense-retrieval code path (`DenseIndex`, `sentence-transformers`) that had already been deleted from `gth/retrieval.py` in a previous release.
- `index.html` had no `<!DOCTYPE html>`, `<html>`, `<head>`, or `<body>`, and no `<meta charset="utf-8">` - the live site rendered visible mojibake (garbled en-dashes/curly quotes) under some server configurations. Fixed.
- `index.html` hardcoded its own, third, independent copy of the seven rubric weights (alongside `gth/rubric.py` and `build.py`'s own second copy). All three are now derived from `gth/rubric.py` at build time via `corpus.js`'s `window.__RUBRIC__`.
- `index.html` also had an inline, hand-pasted copy of the corpus JSON, disconnected from the `corpus.js` file `build.py` actually generates - the two could silently drift. `index.html` now loads `corpus.js` via `<script src>`.
- HTML-escaped all honoree-derived fields interpolated into `innerHTML`-rendered markup (name, bio, award, trajectory note, social links), and validate link URL schemes before rendering them as `href` attributes. Not exploitable today (the corpus is a trusted, maintainer-curated file), but the pattern shouldn't depend on that staying true.

### Performance
- `mmr()` recomputed max-diversity-to-already-selected from scratch for every remaining candidate on every round (O(k² · candidates)). Profiling during a nested cross-validation run showed this was over 80% of total wall time on a 192-document corpus. Rewritten to maintain a running max-similarity per candidate, updated incrementally (O(k · candidates)), with mathematically identical output.
- `TfidfIndex`/`CharNGramIndex`/`BM25Index` scanned every document in the corpus on every `search()` call. Rewritten around inverted indices (postings lists), so a query only visits documents sharing at least one term with it.
- `TfidfIndex` and `BM25Index` each independently retokenized the same documents; `HybridRetriever` now tokenizes once and shares the result.
- `gth/eval.py`'s grid searches (`tune`, `cv`, `ncv`) constructed a brand-new `HybridRetriever` (full reindex) for every hyperparameter combination, even though BM25's `k1`/`b` only affect scoring, not indexing. `HybridRetriever.set_bm25_params()` now mutates them in place on one shared retriever.
- Combined effect: `python -m gth ncv` went from ~75s to ~16s, with byte-identical output.

### Changed
- `gth.eval`'s `run`/`cross_validate`/`nested_cross_validate`/`tune`/`significance` all take `verbose: bool = True` and always return their data (a `Metrics` dataclass or plain dict), so the module is usable from a test or script without capturing stdout.
- `evaluate()` now returns a `Metrics` dataclass instead of a `dict[str, float]` with one smuggled-in `list[float]` value under an underscore-prefixed key.
- `cross_validate` and `nested_cross_validate`'s duplicated fold-selection loops unified into a shared `_select_best` helper.
- Added `ruff` and `mypy` (both clean, zero suppressions) and a GitHub Actions CI workflow running lint, type-check, `build.py`, and the test suite on Python 3.9/3.11/3.12.
- Added `CONTRIBUTING.md` and this changelog.
- Test suite: 23 → 43 tests, including regression tests for every bug above and new coverage for `build.py`.

## 1.6.0
- Removed `QLMIndex`, `CombSUM`/`CombMNZ` fusion, the optional dense-embedding retriever, and the optional cross-encoder reranker: none ever won a single comparison in `gth/eval.py`. Shipped retrieval stack reduced to BM25 + TF-IDF + char n-gram, RRF, RM3, MMR.
- Rebuilt the 39-query gold set as paraphrases of the honoree bios rather than phrases copied from them; auditing every query against the full bio text caught and fixed a real labeling error along the way.
- Every em dash removed repo-wide (code, docs, data, the web app).

## 1.5.0
- Renamed the project from "...Benchmark" to "Glocal Teen Hero Corpus". The rubric scoring is a documented methodology applied by one person to a fixed cohort, not a standardized task others are measured against, so it isn't a benchmark; `gth/eval.py`'s retrieval test collection is, and keeps the name.

## 1.4.0
- Added nested cross-validation (`gth cv` → `gth ncv`) to close a hyperparameter-leakage path in the flat cross-validation, and a paired bootstrap significance test (`gth sig`). Added `PAPER.md`.

## 1.3.0
- Added 5-fold cross-validation (`gth cv`) to select the shipped retrieval default by held-out performance instead of an in-sample comparison table. Result: character n-gram retrieval, not the tuned hybrid ensemble, is the default.

## 1.2.0 and earlier
- Initial `gth` package: typed corpus/rubric/scoring, a from-scratch multi-retriever IR stack (BM25, TF-IDF, an optional dense backend, RRF/CombSUM/CombMNZ fusion, RM3, MMR), a CLI, and the first evaluation harness.
