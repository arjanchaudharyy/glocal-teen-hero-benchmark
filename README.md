<div align="center">

# Glocal Teen Hero Corpus

**An open dataset of every Glocal Teen Hero (Nepal) honoree, 2015-2025, scored on the record they held at selection via a documented rubric, with a small retrieval benchmark over the same corpus.**

[![CI](https://github.com/arjanchaudharyy/glocal-teen-hero-corpus/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml) [![tests](https://img.shields.io/badge/tests-48%20passing-2f7d54)](tests/) [![python](https://img.shields.io/badge/python-3.9%2B-16233e)](pyproject.toml) [![core deps](https://img.shields.io/badge/core%20deps-0-16233e)](pyproject.toml) [![types](https://img.shields.io/badge/mypy%20--strict-clean-16233e)](pyproject.toml) [![license](https://img.shields.io/badge/license-MIT-b3906a)](LICENSE)

[Live tool](https://arjanchaudharyy.github.io/glocal-teen-hero-corpus/) · [Paper](PAPER.md) · [Dataset card](DATA_CARD.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

</div>

---

## What this is

192 honorees, scored on a documented 7-dimension rubric, on the record they held **at selection**, not the career they built afterward. That's the whole rubric side: a dataset plus a scoring methodology, applied by one person to a fixed cohort. It isn't a standardized task other people are measured against, so it isn't called a benchmark.

The one part that is a benchmark: `gth/eval.py`, a real IR test collection (39 gold queries, several retrieval methods compared, cross-validated). That distinction matters enough that it's the reason this repo has this name instead of the one it used to have.

## Quickstart

```bash
git clone https://github.com/arjanchaudharyy/glocal-teen-hero-corpus
cd glocal-teen-hero-corpus

python -m gth rank                          # at-selection leaderboard
python -m gth ask "who works on climate?"   # retrieval, cross-validated default
python -m gth eval                          # IR metrics table
python -m gth cv                            # cross-validated config selection
python -m unittest discover -s tests
```

Zero runtime dependencies, pure standard library. `pip install -e ".[dev]"` for `ruff`/`mypy` if you're contributing (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## The retrieval benchmark

Three retrievers (BM25, TF-IDF cosine, character n-gram TF-IDF), RRF fusion, RM3 query expansion, MMR re-ranking, each backed by an inverted index rather than a full corpus scan per query. That's it. An earlier version of this stacked five retrievers, three fusion strategies, and a neural reranker onto a 192-document corpus; most of it never won a single comparison, which is its own kind of answer. What's here is what actually earned its place.

The 39 gold queries are paraphrases, not copies of the honorees' bios. A query like "menstrual health hygiene periods dignity" against a bio that literally contains "menstrual hygiene" tests substring matching, not retrieval. The current queries describe the same topics in different words ("teens breaking taboos around a private monthly health topic for girls"), so a retriever has to bridge vocabulary instead of finding a copy of itself. Auditing the old query set for this also surfaced a real labeling error: an honoree whose bio is about drug-addiction awareness had been marked relevant to a menstrual-health query. Fixed.

Under the harder queries, every method's absolute score drops sharply (nDCG@10 around 0.16-0.19, versus roughly 0.55 before), which is the honest reading: the earlier high numbers were mostly keyword overlap, not retrieval quality. Character n-gram retrieval is still the strongest and is what ships as the default (`python -m gth cv`, 5-fold, char wins all 5 folds), though a paired significance test shows its edge over the full ensemble is not statistically significant at n=39. The full ensemble is real and available (`hybrid=True` / `--hybrid`), just not the default, because it doesn't out-generalize the simple retriever here.

Full write-up, related work, and the exact numbers: [PAPER.md](PAPER.md).

## Methodology

Seven weighted dimensions (`gth/rubric.py`): social impact (.20), leadership (.20), innovation (.15), entrepreneurship (.15), recognition (.10), glocal fit (.10), character (.10). Each honoree scored 0-5 per dimension on their at-selection record. Weights are a documented prior, editable in `rubric.py` (`build.py` and the web app both read from it; nobody hardcodes a second copy of these numbers anymore). See [DATA_CARD.md](DATA_CARD.md) for schema and labeling protocol.

## Code quality

CI (`.github/workflows/ci.yml`) runs on Python 3.9, 3.11, and 3.12 for every push: `ruff` (lint), `mypy` (type-check, zero suppressions), the 48-test suite, and a full `build.py` regeneration. All four run clean, and are what a PR is expected to pass (see [CONTRIBUTING.md](CONTRIBUTING.md)). None of this existed for most of this project's life; the gaps it caught along the way (a wrong median formula for even-length cohorts, a cache keyed by `id()` that could return stale results after garbage collection, a query path that silently padded results with irrelevant zero-score documents, a `mmr()` implementation that was the dominant cost, over 80% of wall time, in every cross-validation run, a name lookup that could silently resolve to the wrong honoree on a name collision) are in [CHANGELOG.md](CHANGELOG.md).

## Layout

```
gth/
├── rubric.py      # dimensions, weights, weighted_total
├── corpus.py      # typed loader (Hero, Corpus), O(1) name+year lookup
├── scoring.py     # rank_all, cohort_stats, percentile, verdict
├── retrieval.py   # BM25, TF-IDF, char n-gram (inverted indices), RRF, RM3, MMR
├── rag.py         # build_index, similar, ask
├── eval.py        # gold queries, metrics, cv, ncv, sig, tune
├── cli.py         # python -m gth
build.py            # regenerates data/heroes.json + corpus.js from the scored roster
data/heroes.json     # the corpus
index.html           # the web app (loads corpus.js, no hardcoded duplicates)
tests/               # 48 unittest cases
.github/workflows/   # CI: lint, type-check, build, test
```

## Honesty and limitations

Rubric scores reflect public information plus a documented methodology applied by this project's author, not an independent jury. This is a self-assessment tool, and says so. Retrieval labels are one annotator's judgment. `now` career notes are shown for context and excluded from scoring. Not affiliated with Glocal Pvt. Ltd.

## License

MIT. Built by Aarjan Chaudhary.
