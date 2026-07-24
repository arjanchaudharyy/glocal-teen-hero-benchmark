"""
Command-line interface.

    python -m gth rank [--tier winner|finalist|20under20] [--top N]
    python -m gth stats
    python -m gth similar "Aarjan Chaudhary" [--k 5]
    python -m gth ask "who worked on menstrual health?" [--k 5]
    python -m gth score --file profile.json      # {"social_impact":4, ...}
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .corpus import load
from .rubric import DIMENSIONS, LABELS, WEIGHTS, weighted_total, validate_scores
from .scoring import rank_all, cohort_stats, percentile_vs, rank_of, verdict
from .rag import build_index, similar, ask

_TIER = {"winner": "Winner", "finalist": "Finalist", "20under20": "20under20", "applicant": "Applicant"}


def _cmd_rank(a):
    corpus = load()
    ranked = rank_all(corpus)
    if a.tier:
        tier = _TIER[a.tier]
        ranked = [r for r in ranked if r.hero.award == tier]
    print(f"\nGLOCAL TEEN HERO — AT-SELECTION RANKING  ({len(corpus)} honorees)\n" + "=" * 62)
    print(f"{'#':<4}{'Name':<26}{'Year':<6}{'Tier':<11}{'Score':>6}")
    print("-" * 62)
    for i, r in enumerate(ranked[: a.top], 1):
        tag = " *" if r.hero.est else ""
        me = "  <- applicant" if r.hero.me else ""
        print(f"{i:<4}{r.hero.name:<26}{r.hero.year:<6}{r.hero.award:<11}{r.total:>6.2f}{tag}{me}")
    print("  (* = heuristic estimate)")


def _cmd_stats(a):
    corpus = load()
    for tier in ("Winner", "Finalist", "20under20"):
        s = cohort_stats(corpus, tier)
        print(f"{tier:<11} n={s['n']:<4} mean={s['mean']:.2f} min={s['min']:.2f} max={s['max']:.2f}")


def _cmd_similar(a):
    for r in similar(a.name, k=a.k, rerank=not a.no_rerank):
        prov = ",".join(f"{m}#{rk}" for m, rk in sorted(r.sources.items()))
        print(f"{r.score:>8}  {r.hero.name} ({r.hero.year}, {r.hero.award}) [{prov}] — {r.hero.then[:80]}")


def _cmd_ask(a):
    print(ask(a.query, k=a.k, expand=not a.no_expand, rerank=not a.no_rerank))


def _cmd_eval(a):
    from .eval import run
    run(k=a.k, per_query=a.per_query)


def _cmd_tune(a):
    from .eval import tune
    tune(k=a.k)


def _cmd_score(a):
    corpus = load()
    scores = json.load(open(a.file)) if a.file else json.loads(sys.stdin.read())
    validate_scores(scores)
    t = weighted_total(scores)
    print(f"\nWeighted total: {t:.2f} / 5")
    for d in DIMENSIONS:
        print(f"  {LABELS[d]:<16} {scores[d]}  (w={WEIGHTS[d]})")
    print(f"\nVerdict: {verdict(corpus, t)}")
    print(f"Rank:    #{rank_of(corpus, t)} of {len(corpus)} honorees")
    print(f"Winners percentile: {percentile_vs(corpus, t)}th")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gth", description="Glocal Teen Hero at-selection benchmark")
    p.add_argument("--version", action="version", version=f"gth {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank"); r.add_argument("--tier", choices=list(_TIER)); r.add_argument("--top", type=int, default=15); r.set_defaults(fn=_cmd_rank)
    sub.add_parser("stats").set_defaults(fn=_cmd_stats)
    s = sub.add_parser("similar"); s.add_argument("name"); s.add_argument("--k", type=int, default=5); s.add_argument("--no-rerank", action="store_true"); s.set_defaults(fn=_cmd_similar)
    q = sub.add_parser("ask"); q.add_argument("query"); q.add_argument("--k", type=int, default=5); q.add_argument("--no-expand", action="store_true"); q.add_argument("--no-rerank", action="store_true"); q.set_defaults(fn=_cmd_ask)
    e = sub.add_parser("eval"); e.add_argument("--k", type=int, default=10); e.add_argument("--per-query", action="store_true"); e.set_defaults(fn=_cmd_eval)
    t = sub.add_parser("tune"); t.add_argument("--k", type=int, default=10); t.set_defaults(fn=_cmd_tune)
    sc = sub.add_parser("score"); sc.add_argument("--file"); sc.set_defaults(fn=_cmd_score)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
