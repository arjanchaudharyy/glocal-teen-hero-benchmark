#!/usr/bin/env python3
"""
Glocal Teen Hero benchmark scorer (Nepal edition).

Loads data/heroes.json (built by build.py), applies the weighted rubric, and
ranks the applicant against:
  * the WINNERS (the real bar), and
  * the full honoree pool (winners + finalists + 20under20).

Winners + a few honorees are hand-scored from public records; the rest use the
documented tier+field heuristic in build.py (flagged est=true). Dependency-free:
    python3 score.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data", "heroes.json")))
W = {k: v["weight"] for k, v in data["rubric"]["dimensions"].items()}
total = lambda s: round(sum(s[k] * W[k] for k in W), 3)

rows = [{"name": h["name"], "year": h["year"], "award": h["award"],
         "est": h.get("est", False), "t": total(h["s"]), "me": h.get("me", False)} for h in data["heroes"]]

winners = [r for r in rows if r["award"] == "Winner"]
applicant = next((r for r in rows if r["me"]), None)
wl = sorted(r["t"] for r in winners)
mean = sum(wl) / len(wl); mx = max(wl); mn = min(wl)

rows.sort(key=lambda r: r["t"], reverse=True)

print("\nGLOCAL TEEN HERO — NEPAL — BENCHMARK  (weighted rubric, 0-5)\n" + "=" * 70)
print(f"corpus: {len(rows)} honorees  |  {len(winners)} winners  |  "
      f"{sum(1 for r in rows if r['award']=='Finalist')} finalists  |  "
      f"{sum(1 for r in rows if r['award']=='20under20')} 20under20\n")

print("TOP 15 OF ALL HONOREES")
print(f"{'#':<4}{'Name':<26}{'Year':<6}{'Tier':<11}{'Score':>6}")
print("-" * 70)
for i, r in enumerate(rows[:15], 1):
    tag = " *" if r["est"] else ""
    me = "  <- applicant" if r["me"] else ""
    print(f"{i:<4}{r['name']:<26}{r['year']:<6}{r['award']:<11}{r['t']:>6.2f}{tag}{me}")
print("  (* = heuristic estimate, not hand-scored)")

if applicant:
    all_sorted = [r["t"] for r in rows]
    rank_all = sorted(all_sorted, reverse=True).index(applicant["t"]) + 1
    pctW = round(100 * sum(1 for x in wl if x <= applicant["t"]) / len(wl))
    print("\nAPPLICANT vs BENCHMARK (winners 2015-2025)\n" + "-" * 70)
    print(f"  applicant           : {applicant['t']:.2f}  ({applicant['name']})")
    print(f"  winners' mean       : {mean:.2f}")
    print(f"  winners' best       : {mx:.2f}")
    print(f"  delta vs mean       : {applicant['t']-mean:+.2f}")
    print(f"  delta vs best winner: {applicant['t']-mx:+.2f}")
    print(f"  beats {sum(1 for w in winners if applicant['t']>w['t'])}/{len(winners)} winners "
          f"| {pctW}th percentile vs winners")
    print(f"  rank among ALL {len(rows)} honorees: #{rank_all}")
print()
