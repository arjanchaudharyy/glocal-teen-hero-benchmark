# Dataset card - Glocal Teen Hero honorees (Nepal, 2015-2025)

**File:** `data/heroes.json` · **Records:** 192 · **Unit:** one honoree.

## Composition
| Tier | Count |
|------|------:|
| Winner (annual titleholder) | 11 |
| Finalist (top-6, non-winner) | 54 |
| 20under20 honoree | 126 |
| Applicant (2026, rubric-scored) | 1 |

## Schema (per record)
| field | type | meaning |
|-------|------|---------|
| `name` | str | honoree name |
| `year` | int | selection year (2015-2025; 2026 = applicant) |
| `award` | enum | `Winner` \| `Finalist` \| `20under20` \| `Applicant` |
| `s` | obj | 0-5 score per dimension (7 dimensions) |
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
- All labels are from a single annotator (this project's author). No
  independent inter-rater check has been performed.

## Retrieval evaluation labels (`gth/eval.py`)
A separate hand-labeled set of **39 topical queries**, each mapped to the
honorees a human judges clearly relevant, is used to evaluate the retrieval
engine (Recall@k, Precision@k, MRR, nDCG@k, MAP). These labels measure
whether the engine finds on-topic honorees; they are independent of the
honorees' rubric scores.

The queries are deliberately written as paraphrases, not phrases copied from
the bios they target. An earlier version used queries built from the bios'
own wording (e.g. "menstrual health hygiene periods" against a bio containing
"menstrual hygiene"), which rewards substring matching rather than retrieval
and also made a real labeling error easy to miss. Auditing every query
against the full bio text caught and fixed that error; see
[PAPER.md](PAPER.md) for the full account and the resulting drop in every
method's absolute score once the easy overlap was removed.

Labels remain a **sparse pool**: for broad queries, relevant honorees outside
the labeled set are counted as misses, so reported metrics are conservative
lower bounds. The valid signal is the *relative* comparison across retrieval
configs, which is why `gth.eval.cross_validate` (5-fold, held-out queries per
fold) and `gth.eval.nested_cross_validate` are used to select the shipped
default, not the single-table comparison.

## License
MIT for the code; corpus facts belong to their linked sources.
