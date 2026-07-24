# Glocal Teen Hero Benchmark

An open, reproducible attempt to answer one question honestly: **what does a Glocal Teen Hero actually look like, and how does any given applicant measure up?**

I built a small corpus of past Glocal Teen Hero winners, derived a scoring rubric from the program's own stated criteria, and scored the cohort transparently. Everything here is public data and a 60-line Python script. Run it yourself, disagree with my scores, fork it, re-score anyone.

## TL;DR result

```
Rank  Name                    Year   Score
1     Aarjan Chaudhary        2026    4.80   <- applicant
2     Rahul Ranjan Sah        2022    4.05
3     Shruti Tiwari           2023    3.80
4     Ghanashyam Bishwakarma  2024    3.80
5     Pranjal Chalise         2021    3.55

Winners' mean: 3.80 | best winner: 4.05 | applicant: 4.80 (+1.00 vs mean, beats 4/4)
```

I score *below* the winners' average on pure social impact (4.0 vs 4.2) and ahead on innovation, entrepreneurship, recognition, leadership and glocal fit. That gap is the honest picture, not a rounding-up.

## Method

1. **Corpus** (`data/heroes.json`): the single-winner Glocal Teen Hero Nepal titleholders 2021-2024 (the format that matches the 2026 South Asia single-winner edition), each with a public summary and sources. 2025 moved to a "20under20" cohort format, so it's noted as context rather than scored head-to-head.
2. **Rubric** (`data/heroes.json` -> `rubric`): seven dimensions drawn straight from the program's stated values (leadership, innovation & technology, social impact, creativity, community contribution, "beyond academics") and from the organizer's own public write-ups of *why* past winners won. Each is scored 0-5 and weighted; weights sum to 1.0.

   | Dimension | Weight |
   |---|---|
   | Social impact | 0.20 |
   | Leadership | 0.20 |
   | Innovation | 0.15 |
   | Entrepreneurship | 0.15 |
   | Recognition | 0.10 |
   | Glocal fit | 0.10 |
   | Character | 0.10 |

3. **Scoring** (`score.py`): loads the corpus, applies the weights, ranks everyone, and reports the applicant against the winners' mean and best.

## Reproduce

```bash
python3 score.py
```

No dependencies. Edit the 0-5 scores in `data/heroes.json` if you'd weight things differently, and re-run.

## Honesty / limitations

- Scores are my judgment from **public information** (press coverage, the program's own pages, the applicants' sites). They're defensible, not objective truth.
- The rubric encodes a point of view. I documented the weights so you can change them; the applicant stays #1 under most reasonable reweightings, but not all (e.g. weighting social impact to 0.6 narrows it a lot).
- This is a self-assessment tool made by an applicant, and I'm labeling it as exactly that. It exists so the claim "I measure up to past winners" can be checked instead of just asserted.

## License

MIT. Corpus facts belong to their sources (linked in `data/heroes.json`).
