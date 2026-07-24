"""
Scoring rubric for the Glocal Teen Hero corpus, applied on an at-selection basis.

Seven weighted dimensions derived from the program's own stated criteria
(leadership, innovation & technology, social impact, creativity, community
contribution, "beyond academics") and the organizer's public write-ups of why
past winners won. Weights are a documented prior — override them and re-run.
"""
from __future__ import annotations

from typing import Dict, Mapping

# dimension -> weight (must sum to 1.0)
WEIGHTS: Dict[str, float] = {
    "social_impact": 0.20,
    "leadership": 0.20,
    "innovation": 0.15,
    "entrepreneurship": 0.15,
    "recognition": 0.10,
    "glocal_fit": 0.10,
    "character": 0.10,
}

LABELS: Dict[str, str] = {
    "social_impact": "Social impact",
    "leadership": "Leadership",
    "innovation": "Innovation",
    "entrepreneurship": "Entrepreneurship",
    "recognition": "Recognition",
    "glocal_fit": "Glocal fit",
    "character": "Character",
}

DIMENSIONS = tuple(WEIGHTS.keys())
SCALE_MAX = 5.0

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "rubric weights must sum to 1.0"


def weighted_total(scores: Mapping[str, float], weights: Mapping[str, float] = WEIGHTS) -> float:
    """Weighted 0-5 score. Missing dimensions count as 0."""
    return round(sum(scores.get(d, 0.0) * w for d, w in weights.items()), 4)


def validate_scores(scores: Mapping[str, float]) -> None:
    for d in DIMENSIONS:
        v = scores.get(d)
        if v is None:
            raise ValueError(f"missing dimension: {d}")
        if not (0 <= v <= SCALE_MAX):
            raise ValueError(f"{d}={v} out of range [0,{SCALE_MAX}]")
