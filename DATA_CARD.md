# Dataset card — Glocal Teen Hero honorees (Nepal, 2015–2025)

**File:** `data/heroes.json` · **Records:** 192 · **Unit:** one honoree.

## Composition
| Tier | Count |
|------|------:|
| Winner (annual titleholder) | 11 |
| Finalist (top-6, non-winner) | 54 |
| 20under20 honoree | 126 |
| Applicant (2026, benchmarked) | 1 |

## Schema (per record)
| field | type | meaning |
|-------|------|---------|
| `name` | str | honoree name |
| `year` | int | selection year (2015–2025; 2026 = applicant) |
| `award` | enum | `Winner` \| `Finalist` \| `20under20` \| `Applicant` |
| `s` | obj | 0–5 score per dimension (7 dimensions) |
| `then` | str | **at-selection** record (what the jury saw) |
| `now` | str | later trajectory (not used in scoring) |
| `conf` | enum | source confidence: `high` \| `med` \| `low` |
| `est` | bool | heuristic estimate (low public footprint) |
| `links` | obj | source/social links |

## Labeling protocol
Scores are assigned from public information (press, the program's pages,
LinkedIn/GitHub/portfolios) on the **record held at selection**, not later
careers. Higher-footprint honorees are individually researched; low-footprint
honorees receive conservative estimates flagged `est=true`.

## Known limitations
- Judgment-based labels, not ground truth. The rubric encodes a documented prior.
- Common-name honorees with thin footprints are conservative estimates.
- `now` notes are provided for context and are **excluded** from all scoring.

## License
MIT for the code; corpus facts belong to their linked sources.
