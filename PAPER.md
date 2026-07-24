# Two Ways an Evaluation Lies to You: Copied Vocabulary and Same-Set Tuning

**Aarjan Chaudhary**
*Independent research note, accompanying the [Glocal Teen Hero Corpus](README.md) open-source repository*

---

## Abstract

We study retrieval over a small (N = 192 documents), domain-specific corpus of biographical records, comparing three retrieval methods (BM25, TF-IDF cosine, character n-gram TF-IDF), Reciprocal Rank Fusion, RM3 query expansion, and Maximal Marginal Relevance re-ranking. An earlier version of this evaluation used gold queries built from phrases copied out of the target bios themselves (e.g., a query containing "menstrual hygiene" against a bio containing the same words), which we found inflates every retriever's apparent quality by rewarding literal substring overlap rather than actual retrieval. Auditing that query set for this problem also surfaced a labeling error: an honoree whose bio concerns drug-addiction awareness was marked relevant to a menstrual-health query. We rebuilt the 39-query gold set as paraphrases with minimal vocabulary overlap with the target text, and corrected the labels. Under the corrected, harder queries, every method's absolute score drops sharply (nDCG@10 falls from roughly 0.55 to roughly 0.16-0.19), which we read as evidence that the earlier numbers measured keyword copying, not retrieval. Character n-gram retrieval remains the strongest method: flat 5-fold cross-validation selects it unanimously (5 of 5 folds, nDCG@10 = 0.194, 95% CI 0.142-0.228). A second, nested design that additionally re-derives every hyperparameter from scratch inside each outer fold selects it in 4 of 5 folds (nDCG@10 = 0.181, 95% CI 0.104-0.228) and, in the one fold it doesn't, shows the more elaborate ensemble collapsing from an inner score of 0.223 to 0.030 on held-out data, a direct illustration of the failure mode cross-validation exists to catch. A paired bootstrap test shows the gap between the simple retriever and the tuned ensemble is not statistically significant at this sample size (Δ nDCG@10 = 0.024, 95% CI [-0.014, 0.063]). We report all of it, including a subsequent correctness bug in our own retrieval code that changed several of these numbers between drafts, as a single case study in how easy it is for an evaluation to look better than the system actually is, and how cheap it is to check for both the vocabulary-overlap artifact and the overfitting artifact.

## 1. Introduction

Two habits reliably make a retrieval evaluation look better than it is. The first is building gold queries that share vocabulary with the documents they're meant to retrieve, since any retriever with a term-overlap component will do artificially well. The second is tuning a system's hyperparameters and architecture on the same set used to report its final performance, since a configuration can look best simply because it was fit to that data. Both are easy to do by accident, especially on a small corpus with a small labeled set, where the person writing the queries already knows what the target documents say.

This note is a record of catching both, on the same small system. Section 3 describes a 192-document corpus of biographical records and the retrieval system built over it. Section 5 shows that the original 39-query gold set had queries built from the same words as the bios they targeted, and that this had also produced a real labeling error. Section 6 reports evaluation on a rebuilt, harder query set: absolute performance collapses across every method, which is the expected result once literal overlap is removed. Section 7 then applies cross-validation to the harder set and finds that a more elaborate retrieval ensemble still doesn't out-generalize the simplest retriever, closing the second failure mode on top of the first.

All code, data, and evaluation harnesses are open source and executable. Every number here is reproducible with `python -m gth eval`, `python -m gth cv`, `python -m gth ncv`, and `python -m gth sig`, with no network access, no randomness beyond a fixed seed, and no dependencies beyond the Python standard library.

## 2. Related Work

BM25 (Robertson & Zaragoza, 2009) is a standard, length-normalization-aware term-based retrieval baseline. Reciprocal Rank Fusion (Cormack, Clarke, & Buettcher, 2009) combines ranked lists from heterogeneous scorers by summing reciprocal ranks, avoiding the need to calibrate scores across retrievers whose outputs live on different scales. RM3 (Lavrenko & Croft, 2001; Jaleel et al., 2004) interpolates a relevance model estimated from top-ranked pseudo-relevant documents with the original query model. Maximal Marginal Relevance (Carbonell & Goldstein, 1998) balances relevance against redundancy with previously selected results.

On validation: Kohavi (1995) documents how naive train-and-report evaluation overstates generalization when a system is tuned on the set it's reported against, which is exactly the second failure mode this note describes. Efron (1979) establishes the bootstrap as a general method for estimating sampling distributions, used here both for confidence intervals and, in paired form, for a significance test between two configurations scored on the same queries. The first failure mode, query sets built with vocabulary borrowed from target documents, is closer to a data quality problem than a modeling one; we are not aware of a standard name for it in the IR literature, though it is a specific instance of the general problem of gold labels correlating with a system's inductive bias rather than with true relevance.

This work does not propose a new retrieval or fusion method. Its contribution is a small, fully reproducible demonstration of two specific ways an evaluation can overstate what a system can do, in a system whose individual components are all standard.

## 3. Dataset and System

The corpus is 192 short biographical records (11 "Winner", 54 "Finalist", 126 "20under20", 1 applicant record), each an "at-selection" summary describing the record as it stood at the time of selection, not later career developments. See [DATA_CARD.md](DATA_CARD.md) for schema and labeling protocol.

The retrieval system is deliberately small for a 192-document corpus: BM25 (tunable *k1*, *b*), TF-IDF cosine with smoothed IDF and L2-normalized vectors, and a character 3-4-gram TF-IDF cosine retriever, robust to spelling and transliteration variance. Retriever outputs combine via Reciprocal Rank Fusion; queries may be expanded via RM3 before retrieval; the fused list may be re-ranked with MMR using the fused relevance score. An earlier version of this system additionally included a query-likelihood language model, two more fusion strategies (CombSUM, CombMNZ), an optional dense embedding retriever, and an optional neural cross-encoder reranker. None of them won a single comparison in this evaluation, so they were removed; what remains is what actually earned its place. Full implementation is in `gth/retrieval.py`.

## 4. The Vocabulary-Overlap Problem

The original 39-query gold set included queries such as *"menstrual health hygiene periods dignity"*, matched against bios that literally contain the phrase *"menstrual hygiene"*. A character n-gram or TF-IDF retriever will find that match trivially: it is not retrieval in any meaningful sense, it is confirming that a substring search works. Auditing every query against the full, untruncated bio text (rather than the query author's memory of what the bios said) surfaced a genuine labeling error downstream of this practice: an honoree whose bio concerns drug-addiction awareness, with no mention of menstrual health, had been included as relevant to the menstrual-health query, most likely because an earlier summarization of the corpus conflated two honorees. A second honoree with a directly on-topic bio ("Pyari Periods", a menstrual-health project literally named after the topic) had been left out, having been placed under an unrelated query instead.

We rebuilt all 39 queries as paraphrases: *"teens breaking taboos around a private monthly health topic for girls"* in place of the phrase above, and similarly for the rest of the set, describing each topic in different words than the bios use. We also re-verified every relevant-honoree list against the full bio text at this time, correcting several similar minor errors alongside the one described here. The rebuilt set and corrected labels are in `gth/eval.py`.

## 5. Results Under the Corrected, Harder Queries

**Table 1: in-sample evaluation (k=10, n=39 queries), corrected and paraphrased gold set**

| Config | Recall@10 | Prec@10 | MRR | nDCG@10 | MAP |
|---|--:|--:|--:|--:|--:|
| TF-IDF | 0.168 | 0.069 | 0.283 | 0.160 | 0.102 |
| BM25 | 0.163 | 0.067 | 0.289 | 0.160 | 0.103 |
| Character n-gram | 0.243 | 0.095 | 0.291 | 0.193 | 0.123 |
| Hybrid (RRF) | 0.157 | 0.064 | 0.306 | 0.157 | 0.120 |
| Hybrid + RM3 | 0.200 | 0.079 | 0.298 | 0.174 | 0.114 |
| Hybrid + RM3 + MMR | 0.186 | 0.077 | 0.309 | 0.170 | 0.114 |

Every absolute number is roughly a third of what the vocabulary-matched query set produced (nDCG@10 around 0.55 there, around 0.16-0.19 here). We read this as the honest measurement: once a retriever can no longer win by finding a literal copy of the query inside the target text, actual retrieval over paraphrased, semantically-related-but-lexically-different queries is considerably harder for every purely lexical method tested, which is unsurprising and exactly what should happen. Character n-gram retrieval remains the best performer on 4 of 5 metrics, though the margins between methods are small relative to their confidence intervals (Section 6).

These numbers differ slightly from an earlier draft of this table (TF-IDF and BM25 moved from nDCG@10 0.167/0.165 to 0.160/0.160; character n-gram and the hybrid rows are unchanged). The cause was a second, independent bug, not the vocabulary-overlap issue in Section 4: the single-retriever query path returned exactly-zero-relevance documents as filler whenever fewer than *k* genuine matches existed, which inflated TF-IDF and BM25's apparent recall specifically (they match fewer documents per query than the fuzzier character n-gram retriever, so they hit this padding path more often). Fixed by rewriting the retrievers around inverted indices, which structurally cannot return a document that shares no term with the query; a regression test now asserts this directly. We mention it because a table that silently changed between two versions of this document, with no note, would be exactly the kind of quiet correction this paper argues against.

## 6. Cross-Validation

5-fold cross-validation (candidates: each single retriever, and the RRF+RM3+MMR ensemble at four fusion weightings) selects the training-best candidate per fold and scores it on the held-out fold:

**Table 2a: flat 5-fold cross-validation**

| Fold | Selected (train) | Train nDCG@10 | Held-out nDCG@10 | n (test) |
|---|---|--:|--:|--:|
| 1 | char | 0.184 | 0.231 | 8 |
| 2 | char | 0.219 | 0.095 | 8 |
| 3 | char | 0.184 | 0.230 | 8 |
| 4 | char | 0.193 | 0.195 | 8 |
| 5 | char | 0.188 | 0.219 | 7 |

Character n-gram wins all 5 folds outright. Cross-validated nDCG@10 across all 5 folds = 0.194 (95% CI 0.142-0.228, bootstrap over fold scores).

Nested cross-validation re-derives BM25 *k1, b* and the fusion weight from scratch inside each outer training fold's own inner split, rather than trusting the fixed values above (themselves picked by an earlier sweep of the full set):

**Table 2b: nested 5x4-fold cross-validation**

| Fold | Inner-selected | Inner nDCG@10 | Held-out nDCG@10 | n (test) |
|---|---|--:|--:|--:|
| 1 | char | 0.184 | 0.231 | 8 |
| 2 | hybrid+rm3+mmr (k1=1.0, b=0.4, w=0.5) | 0.223 | **0.030** | 8 |
| 3 | char | 0.187 | 0.230 | 8 |
| 4 | char | 0.193 | 0.195 | 8 |
| 5 | char | 0.188 | 0.219 | 7 |

Character n-gram wins 4 of 5 folds here. Fold 2 is the clearest illustration in this dataset of the exact failure mode cross-validation exists to catch: across a 45-point inner grid, the ensemble variant that looked best on its own training partition (nDCG@10 = 0.223, the highest inner score of any fold) fell to 0.030, the single worst score in either table, on data it hadn't seen. Nested cross-validated nDCG@10 = 0.181 (95% CI 0.104-0.228).

The two designs no longer agree exactly (0.194, 5/5, versus 0.181, 4/5) the way an earlier draft of this paper reported; that earlier agreement was itself an artifact of the padding bug described in Section 5, which happened to affect fold 2's queries similarly under both designs. With the bug fixed, the two now diverge slightly, in the direction the nested design's larger search space would predict: searching 45 hyperparameter combinations per fold, instead of 9 fixed named candidates, gives more surface area for a fold-specific quirk to look good in-sample before failing out-of-sample. This is exactly why the nested number, not the flat one, is the one we treat as safe to publish, regardless of which direction they happen to disagree in.

A paired bootstrap test on the per-query nDCG difference between char n-gram and the full ensemble (both scored on the same 39 queries) gives a mean difference of +0.024 in char's favor, 95% CI [-0.014, 0.063], not significant at α = 0.05. As before, this and the cross-validation result are not in tension: cross-validation answers "which config should ship, given we must pick one," and its answer is character n-gram in every flat fold and 4 of 5 nested folds; the significance test answers "is the gap distinguishable from chance at this sample size," and at n=39 it is not. Both are true at once.

## 7. Discussion

**On the vocabulary-overlap finding.** This is, in retrospect, an obvious problem: writing gold queries by paraphrasing your own memory of what a document says will tend to reproduce that document's words, especially for a single annotator working from a corpus they built. The fix (write queries as if you did not know the target vocabulary, then verify labels against the actual text rather than memory) is cheap and worth doing before, not after, reporting numbers.

**On the overfitting finding.** It survives the harder queries essentially unchanged from an earlier version of this evaluation, which used the easier, vocabulary-matched query set. That earlier version also found character n-gram winning cross-validation while the significance test came back non-significant. The fact that both findings hold under a much harder, corrected task is some evidence the second finding wasn't an artifact of the first, though at n=39 queries this is not a strong claim either way.

**On the retrieval-code bug.** A third, smaller thing went wrong after both findings above were already written up: the single-retriever query path returned zero-relevance documents as filler when a query had fewer genuine matches than requested, inflating TF-IDF and BM25's apparent metrics specifically (Section 5). It was caught by a code audit unrelated to this paper's methodology, not by the evaluation design, which is itself worth noting: cross-validation and careful query construction catch measurement artifacts in the *evaluation*, not correctness bugs in the *system being evaluated*. Both kinds of checking are needed, and neither substitutes for the other.

**Limitations.** The 39-query set, corrected and paraphrased, remains small; per-fold test partitions of 7-8 queries produce the wide held-out variance visible in Tables 2a-2b. Absolute metrics under the harder queries are low in a way that likely also reflects the limits of purely lexical methods against genuinely paraphrased queries; a dense embedding retriever was not evaluated here, having been removed in the same pass that removed other components that hadn't earned their place, and whether it changes either finding is an open question. All labels, including the corrections in Section 4, were made by one annotator; no independent inter-rater check was performed.

## 8. Conclusion

Two ordinary things went wrong in an earlier version of this evaluation: the gold queries shared vocabulary with their target documents, and a retrieval ensemble's apparent advantage came from being tuned on the same set it was reported against. Neither is exotic, and both were caught with tools that cost nothing beyond writing them once: rewriting queries to paraphrase rather than copy, and running cross-validation instead of a single train-and-report table. Once both were corrected, the honest picture is a system with low absolute retrieval quality on genuinely hard queries, where the simplest retriever tested is at least as good as, and probably not distinguishable in quality from, a considerably more complicated one.

## Reproducing this paper

```bash
git clone https://github.com/arjanchaudharyy/glocal-teen-hero-corpus
cd glocal-teen-hero-corpus
python -m gth eval          # Table 1
python -m gth cv --folds 5  # Table 2a
python -m gth ncv           # Table 2b
python -m gth sig           # Section 6, significance test
```

Deterministic (invariant across `PYTHONHASHSEED`), no network access, no dependencies beyond the Python 3.9+ standard library.

## References

- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98.*
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR '09.*
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1-26.
- Jaleel, N. A., Allan, J., Croft, W. B., Diaz, F., Larkey, L. S., Li, X., Smucker, M. D., & Wade, C. (2004). UMass at TREC 2004: Novelty and HARD. *TREC 2004.*
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI '95.*
- Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. *SIGIR '01.*
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.
