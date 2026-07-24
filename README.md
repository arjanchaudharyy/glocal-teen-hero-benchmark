# Glocal Teen Hero — At-Selection Benchmark

An open, reproducible benchmark of the **Nepal** Glocal Teen Hero honorees (glocalteenhero.com), 2015–2025.

**The key idea:** everyone is scored on the record they held **at the time they were selected** — the teen record the jury actually saw — **not** the career they built in the years since. That's the only fair basis for the question *"would I be selected as a Glocal Teen Hero?"* Comparing a current 15-year-old to someone's post-award university/startup years would be apples-to-oranges. A separate **"now"** note on each person captures where they went afterward.

**Live tool:** `index.html` (or GitHub Pages) — paste your record, get an at-selection score, compare against any honoree, and browse all 192.

## Corpus

- **192** honorees · **11** winners · **54** finalists · **126** 20under20 (2015–2025)
- Deep public research per person (press, program pages, LinkedIn/GitHub/portfolios). Higher-footprint honorees are well-sourced; low-footprint ones get conservative estimates, flagged `est`.

## Top of the at-selection ranking

```
#   Name                Year  Tier        Score
1   Aarjan Chaudhary    2026  Applicant   4.65   <- applicant
2   Shreejay Subedi     2024  Finalist    4.10
3   Sushant Sapkota     2020  20under20   4.00
4   Darshana Rijal      2022  Finalist    4.00
5   Vaibhav Nahata      2020  Finalist    3.95
```

Note the difference between *at-selection* and *now*: e.g. Dipisha Bhujel (2018) was an awareness-campaign teen at selection (~2.3) but has since built Sparśa and won the Iris STEM Prize; Samir Phuyal (2019 winner) later built Karobar (300k+ downloads). Those belong in the "now" view, not the selection benchmark.

## Seven dimensions (weights)

Social impact 0.20 · Leadership 0.20 · Innovation 0.15 · Entrepreneurship 0.15 · Recognition 0.10 · Glocal fit 0.10 · Character 0.10

## Reproduce

```bash
python3 build.py    # rebuild data/heroes.json (+ corpus.js) from the scored roster
python3 score.py    # print the at-selection ranking
```

No dependencies. Edit scores in `build.py` or weights, and re-run.

## Honesty

Scores reflect public information + a documented rubric, scored **at selection**, not the jury's decision. Self-assessment tool built by a 2026 applicant. Not affiliated with Glocal Pvt. Ltd. MIT licensed.
