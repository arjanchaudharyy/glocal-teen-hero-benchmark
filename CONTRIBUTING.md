# Contributing

## Setup

```bash
git clone https://github.com/arjanchaudharyy/glocal-teen-hero-corpus
cd glocal-teen-hero-corpus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

No other dependencies: the package itself is pure standard library.

## Before opening a PR

```bash
ruff check gth/ tests/ build.py          # lint
mypy gth/ build.py                       # type-check
python -m unittest discover -s tests -v  # tests
python build.py                          # regenerate data/heroes.json + corpus.js
```

All four are also what CI runs (`.github/workflows/ci.yml`), on Python 3.9, 3.11, and 3.12. A PR that doesn't pass all four won't merge.

## Where things live

- `gth/rubric.py` is the single source of truth for the seven scoring weights. Do not hardcode a second copy anywhere; `build.py` and `index.html` both read from it (via `data/heroes.json`'s generated `rubric` field, injected into `corpus.js`).
- `gth/retrieval.py` is deliberately small. Before adding a retriever, fusion strategy, or reranker, run `python -m gth cv` or `python -m gth ncv` on your change and confirm it actually wins a held-out comparison. Two removed retrievers (a query-likelihood LM, two extra fusion strategies, a dense embedding backend, a cross-encoder reranker) never won a single comparison and were cut for exactly that reason. "It sounds more sophisticated" is not sufficient justification for a corpus this size.
- `gth/eval.py`'s `GOLD` query set must stay paraphrases of the honoree bios, not phrases copied from them (see the module docstring and `PAPER.md` Section 4 for why: a copied-phrase query set inflates every retriever's apparent quality by rewarding substring matching instead of retrieval, and previously caused a real labeling error to go unnoticed).
- `data/heroes.json` and `corpus.js` are generated files. Edit `build.py`'s `DATA` string, then run `python build.py`; do not hand-edit the generated files.

## Adding an honoree or correcting a record

Edit the `DATA` string in `build.py` (format documented at the top of the file), then run `python build.py` and `python -m unittest discover -s tests`.

## Reporting a bug

Open an issue with the exact command that reproduces it (`python -m gth ...` or a short Python snippet). If it's about a specific honoree's score or bio, include the source you're citing.
