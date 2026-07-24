# Glocal Teen Hero Benchmark

An open, reproducible attempt to answer one question honestly: **what does a Glocal Teen Hero actually look like, and how does an applicant measure up?**

This covers the **Nepal edition** of Glocal Teen Hero (glocalteenhero.com). It compiles a corpus of **192 honorees** — every annual winner (2015–2025), plus the top-6 finalists and the full "20under20" cohorts (2017–2025) — derives a scoring rubric from the program's own stated criteria, and scores everyone transparently.

**Live tool:** open `index.html` (or the GitHub Pages URL) — paste your application and get scored live against the whole cohort.

## Corpus at a glance

- **192** total honorees · **11** winners · **34** finalists · **146** 20under20
- Winners and a handful of deeply-documented honorees are **hand-scored from public records**. The wider cohort is scored by a **documented tier + field heuristic** (`build.py`) and flagged `est: true` — modeled estimates, not per-person research. The UI labels them.

## Winners ranked (2015–2025)

| # | Winner | Year | Score |
|---|--------|------|-------|
| 1 | Rahul Ranjan Sah | 2022 | 4.05 |
| 2 | Bipana Sharma | 2015 | 3.80 |
| 3 | Prashansha KC | 2018 | 3.80 |
| 4 | Shruti Tiwari | 2023 | 3.80 |
| 5 | Ghanashyam Bishwakarma | 2024 | 3.80 |
| 6 | Sachin Dangi | 2017 | 3.70 |
| 7 | Samir Phuyal | 2019 | 3.70 |
| 8 | Krish Yadav | 2025 | 3.65 |
| 9 | Mandira Shrestha | 2020 | 3.60 |
| 10 | Pranjal Chalise | 2021 | 3.55 |
| 11 | Santosh Lamichhane | 2016 | 3.40 |

Applicant reference — **Aarjan Chaudhary (2026): 4.80**, which would rank **#1 of all 192 honorees** (+1.09 vs the winners' mean of 3.71, +0.75 vs the best winner). The tool shows this openly, including the one dimension (pure social impact) where the applicant sits *below* the winners' average.

## The seven dimensions

Drawn from the program's stated values (leadership, innovation & technology, social impact, creativity, community contribution, "beyond academics") and the organizer's write-ups of why winners won. Weighted, summing to 1.0:

| Dimension | Weight |
|---|---|
| Social impact | 0.20 |
| Leadership | 0.20 |
| Innovation | 0.15 |
| Entrepreneurship | 0.15 |
| Recognition | 0.10 |
| Glocal fit | 0.10 |
| Character | 0.10 |

## Reproduce

```bash
python3 build.py    # rebuild data/heroes.json from the roster + rubric
python3 score.py    # print the ranking + applicant vs the winners' benchmark
```

No dependencies. Edit the roster or the 0–5 scores in `build.py` / `data/heroes.json`, or change the weights, and re-run.

## Files

- `data/heroes.json` — the corpus (all 192 honorees, per-dimension scores, sources).
- `build.py` — roster + rubric + the tier/field heuristic that generates the corpus.
- `score.py` — reproducible ranking + benchmark.
- `index.html` — the interactive scoring platform (self-contained, embeds the corpus).

## Honesty / limitations

- Scores reflect **public information** and a documented rubric, not the actual jury's decision. Winner scores are hand-read; cohort scores are heuristic estimates (`est: true`).
- The rubric encodes a point of view. The weights are exposed so you can change them.
- This is a self-assessment tool built by a 2026 applicant, labeled as exactly that. It exists so the claim "I measure up to past winners" can be checked instead of just asserted.
- Not affiliated with Glocal Pvt. Ltd. Corpus facts belong to their sources.

## License

MIT.
