"""
Ranking + cohort statistics over the at-selection rubric-scored corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

from .corpus import Corpus, Hero
from .rubric import WEIGHTS, weighted_total


@dataclass(frozen=True)
class Ranked:
    hero: Hero
    total: float
    rank: int


def rank_all(corpus: Corpus, weights: Mapping[str, float] = WEIGHTS) -> List[Ranked]:
    scored = sorted(corpus.heroes, key=lambda h: weighted_total(h.scores, weights), reverse=True)
    return [Ranked(h, weighted_total(h.scores, weights), i + 1) for i, h in enumerate(scored)]


def cohort_stats(corpus: Corpus, tier: str = "Winner", weights: Mapping[str, float] = WEIGHTS) -> Dict[str, float]:
    vals = sorted(weighted_total(h.scores, weights) for h in corpus.by_tier(tier))
    if not vals:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "min": vals[0],
        "max": vals[-1],
        "median": vals[len(vals) // 2],
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
