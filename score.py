#!/usr/bin/env python3
"""
Glocal Teen Hero benchmark scorer.

Loads data/heroes.json, applies the weighted rubric, and prints a ranking of
past single-winner Glocal Teen Heroes (2021-2024) plus the 2026 applicant.

The "benchmark" is the past winners' cohort: their mean and max weighted score.
Reproducible, dependency-free:  python3 score.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "data", "heroes.json")) as f:
    data = json.load(f)

DIMS = data["rubric"]["dimensions"]
WEIGHTS = {k: v["weight"] for k, v in DIMS.items()}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

def weighted(scores):
    return sum(scores[d] * WEIGHTS[d] for d in WEIGHTS)

rows = []
for h in data["heroes"]:
    rows.append({
        "name": h["name"],
        "year": h["year"],
        "applicant": h["year"] == 2026,
        "total": round(weighted(h["scores"]), 3),
        "scores": h["scores"],
    })

winners = [r for r in rows if not r["applicant"]]
applicant = next(r for r in rows if r["applicant"])

bench_mean = round(sum(r["total"] for r in winners) / len(winners), 3)
bench_max = max(r["total"] for r in winners)

rows.sort(key=lambda r: r["total"], reverse=True)

print("\nGLOCAL TEEN HERO BENCHMARK  (weighted rubric, 0-5 scale)\n" + "=" * 64)
hdr = f"{'Rank':<5}{'Name':<24}{'Year':<6}{'Score':>7}"
print(hdr + "\n" + "-" * 64)
for i, r in enumerate(rows, 1):
    tag = "  <- applicant" if r["applicant"] else ""
    print(f"{i:<5}{r['name']:<24}{r['year']:<6}{r['total']:>7.2f}{tag}")

print("\nBENCHMARK (past winners 2021-2024)\n" + "-" * 64)
print(f"  winners' mean score : {bench_mean:.2f}")
print(f"  winners' best score : {bench_max:.2f}  ({[w['name'] for w in winners if w['total']==bench_max][0]})")
print(f"  applicant score     : {applicant['total']:.2f}  ({applicant['name']})")
print(f"  delta vs mean       : {applicant['total']-bench_mean:+.2f}")
print(f"  delta vs best winner: {applicant['total']-bench_max:+.2f}")
beat = sum(1 for w in winners if applicant["total"] > w["total"])
print(f"  beats {beat}/{len(winners)} past winners")

print("\nPER-DIMENSION (applicant vs winners' average)\n" + "-" * 64)
print(f"{'Dimension':<20}{'Weight':>8}{'Applicant':>11}{'Win.avg':>9}")
for d in WEIGHTS:
    wa = sum(w2["scores"][d] for w2 in winners) / len(winners)
    print(f"{d:<20}{WEIGHTS[d]:>8.2f}{applicant['scores'][d]:>11.1f}{wa:>9.1f}")
print()
