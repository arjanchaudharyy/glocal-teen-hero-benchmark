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
Scores were assigned by an AI research process: Claude (via Claude Code,
using web search and page-fetch tools, dispatched as batches of research
agents per year/cohort) was directed to find each honoree's public record
(press, the program's pages, LinkedIn/GitHub/portfolios) **as it stood at
selection**, not their later career, and to score it 0-5 per dimension
against a fixed rubric and calibration guide, explicitly instructed not to
guess or inflate. Higher-footprint honorees were individually researched
this way; low-footprint honorees received conservative estimates flagged
`est=true` per that same instruction. This project's author (Aarjan
Chaudhary) set the research scope, reviewed the output, and made final
edits, but did not personally research or hand-score each record.

## Known limitations
- Judgment-based labels, not ground truth. The rubric encodes a documented prior.
- Common-name honorees with thin footprints are conservative estimates.
- `now` notes are provided for context and are **excluded** from all scoring.
- Labels were produced by a single AI research process (Claude), directed
  and reviewed by this project's author, not by independent human
  annotators. This carries AI-specific risk on top of the single-annotator
  risk: research quality is bounded by what is indexed and findable online,
  which likely compounds the existing skew toward honorees with an
  English-language digital footprint, and no individual claim was
  independently fact-checked beyond what its source page stated. No
  inter-rater check of any kind, human or AI, has been performed.
- The applicant's own record was researched and scored by this same AI
  process, which knew throughout that it was scoring the record being
  benchmarked against the rest of the corpus. This was not a blind or
  independent evaluation, and is the single most important limitation of
  this methodology overall.
- The `20under20` tier has no honorees at all for 2015-2016 (21 honorees/year
  from 2017 on; 6 and 5 respectively for 2015/2016). This reflects the
  award's own history, not a data-quality gap, and is **not** a normalization
  target: filling it in would fabricate honorees that did not exist. Any
  year-over-year comparison should account for this asymmetry rather than
  treat 2015-2016 as directly comparable cohorts.
- 14 honorees (7.3% of the corpus) share one of 5 exact-duplicate one-line
  `then` bios ("Social activist honoree" x5, "Innovator honoree" x3, "Coder
  honoree" x2, "Technology honoree" x2, "Child-rights (Nepalgunj)" x2) -
  `build.py` now prints a warning listing every such cluster on each rebuild
  (`find_duplicate_bios`), though it does not fail the build, since a generic
  descriptor is sometimes the genuine limit of what's on record for a
  low-footprint honoree. These honorees are textually indistinguishable to
  every retriever in `gth/retrieval.py`.
- Within those clusters, per-dimension rubric scores differ across honorees
  who share identical bio text, and nothing in the schema (no per-dimension
  notes or citation field) records a reason. This should be read as
  unverifiable, not as evidence of additional research: there is no
  recorded basis in this repository for treating those differences as
  meaningful rather than noise in a coarse 0-5 heuristic estimate.
- None of those 14 honorees appear as a relevant result for any of the 39
  `gth/eval.py` gold queries, so the retrieval benchmark's reported metrics
  never exercise the one scenario where lexical retrieval would most
  obviously fail (disambiguating honorees with identical text).

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
