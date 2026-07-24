#!/usr/bin/env python3
"""Backwards-compatible entrypoint. Prefer:  python -m gth rank
Prints the at-selection ranking (top 15)."""
from gth.cli import main

if __name__ == "__main__":
    main(["rank", "--top", "15"])
