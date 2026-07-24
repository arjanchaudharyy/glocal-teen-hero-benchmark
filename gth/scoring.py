"""
Ranking + cohort statistics over the at-selection rubric-scored corpus.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .corpus import Corpus, Hero
from .rubric import WEIGHTS, weighted_total


@dataclass(frozen=True)
class Ranked:
    hero: Hero
    total: float
    rank: int


def rank_all(corpus: Corpus, weights: Mapping[str, float] = WEIGHTS) -> list[Ranked]:
    pairs = sorted(
        ((weighted_total(h.scores, weights), h) for h in corpus.heroes),
        key=lambda p: p[0], reverse=True,
    )
    return [Ranked(h, total, i + 1) for i, (total, h) in enumerate(pairs)]


def _median(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2, 4)


def cohort_stats(corpus: Corpus, tier: str = "Winner", weights: Mapping[str, float] = WEIGHTS) -> dict[str, float]:
    vals = sorted(weighted_total(h.scores, weights) for h in corpus.by_tier(tier))
    if not vals:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "min": vals[0],
        "max": vals[-1],
        "median": _median(vals),
    }


def percentile_vs(corpus: Corpus, score: float, tier: str = "Winner", weights: Mapping[str, float] = WEIGHTS) -> int:
    vals = [weighted_total(h.scores, weights) for h in corpus.by_tier(tier)]
    if not vals:
        return 0
    return round(100 * sum(1 for v in vals if v <= score) / len(vals))


def rank_of(corpus: Corpus, score: float, weights: Mapping[str, float] = WEIGHTS) -> int:
    return sum(1 for h in corpus.heroes if weighted_total(h.scores, weights) > score) + 1


def verdict(corpus: Corpus, score: float, weights: Mapping[str, float] = WEIGHTS) -> str:
    st = cohort_stats(corpus, "Winner", weights)
    if score <= 0:
        return "no record entered"
    if score > st["max"]:
        return "above every winner's teen record"
    if score >= st["mean"]:
        return "winner-level"
    if score >= st["min"]:
        return "finalist-level"
    return "below the winner floor"
