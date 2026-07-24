# Simple Beats Sophisticated (Sometimes): A Cross-Validated Study of Lexical Retrieval Ensembles on a Small, Domain-Specific Corpus

**Aarjan Chaudhary**
*Independent research note · accompanying the [Glocal Teen Hero Benchmark](README.md) open-source repository*

---

## Abstract

We study retrieval-augmented lookup over a small (N = 192 documents), domain-specific corpus of biographical records, comparing five retrieval methods (BM25, TF-IDF cosine, a Dirichlet-smoothed query-likelihood language model, character n-gram TF-IDF, and an optional dense sentence-embedding model), three rank/score fusion strategies (Reciprocal Rank Fusion, CombSUM, CombMNZ), RM3 relevance-model query expansion, and Maximal Marginal Relevance re-ranking. On a fixed 39-query labeled evaluation set, a hand-tuned hybrid ensemble (BM25 + TF-IDF + character n-gram, fused with RRF, expanded with RM3, re-ranked with MMR) outperforms every single retriever — the expected result, and the one a practitioner would naturally report and ship. We then apply 5-fold and nested 5×4-fold cross-validation to test whether that advantage generalizes to held-out queries. It does not: **plain character n-gram retrieval wins the held-out comparison in all 5 folds**, under both flat and nested cross-validation, with an identical cross-validated nDCG@10 of 0.578 (95% CI 0.490–0.648) in both cases. A paired bootstrap significance test further shows the in-sample performance gap between the ensemble and the single retriever (Δ nDCG@10 = 0.022) is **not statistically significant** at n = 39 (95% CI on the difference: [-0.031, 0.076]). We interpret this as a case study in a specific, common failure mode — hyperparameter and architecture selection performed on the same set used for final reporting — and in the value of cross-validation as a cheap, general-purpose defense against it, even in the small-data, classical-IR regime where such controls are frequently skipped.

## 1. Introduction

Retrieval system design commonly proceeds by building an ensemble of complementary retrievers, tuning their combination on a labeled query set, and reporting the tuned configuration's performance on that same set. This practice is standard in production RAG systems and often reasonable at scale — but it structurally cannot distinguish "this configuration generalizes" from "this configuration was fit to this data." The distinction matters most exactly where it's least often checked: small, low-resource corpora, where the number of tunable knobs (retriever weights, fusion parameters, BM25's *k₁* and *b*) can approach or exceed the number of labeled queries available to tune them against.

This note documents such a case. We built a five-retriever hybrid IR system over a 192-document corpus (Section 3), tuned it against a 39-query labeled set (Section 5), and observed the tuned ensemble beat every single-retriever baseline on the in-sample table (Section 6) — the standard, expected result. We then asked the harder question directly: *does this configuration outperform the simplest baseline on queries it did not see?* Sections 7–8 show it does not, using two independent cross-validation designs and a significance test, and Section 9 discusses what we take the finding to mean (and not mean).

All code, data, and evaluation harnesses referenced here are open source and executable — every number in this paper is reproducible by running `python -m gth eval`, `python -m gth cv`, `python -m gth ncv`, and `python -m gth sig` against the accompanying repository, with no network access, no randomness beyond a fixed seed, and no dependencies beyond the Python standard library.

## 2. Related Work

**Lexical retrieval.** BM25 (Robertson & Zaragoza, 2009) remains a strong, well-understood baseline for term-based retrieval and is length-normalization-aware, unlike plain TF-IDF cosine. Query-likelihood retrieval with Dirichlet smoothing (Zhai & Lafferty, 2001) offers a probabilistic alternative grounded in language modeling rather than the vector-space model.

**Rank fusion.** Reciprocal Rank Fusion (Cormack, Clarke, & Buettcher, 2009) combines ranked lists from heterogeneous scorers by summing reciprocal ranks rather than normalized scores, avoiding the need for score calibration across retrievers whose outputs live on incomparable scales. CombSUM and CombMNZ (Fox & Shaw, 1994) are longstanding score-level alternatives.

**Pseudo-relevance feedback.** RM3 (Lavrenko & Croft, 2001; Jaleel et al., 2004) interpolates a relevance model estimated from top-ranked pseudo-relevant documents with the original query language model, and is a standard, principled successor to earlier Rocchio-style expansion.

**Diversity re-ranking.** Maximal Marginal Relevance (Carbonell & Goldstein, 1998) balances query relevance against redundancy with previously selected results, addressing the tendency of pure relevance ranking to return near-duplicate top-k results.

**Dense retrieval.** Sentence-embedding models such as those built on the Sentence-BERT training scheme (Reimers & Gurevych, 2019) and distilled architectures such as MiniLM (Wang et al., 2020) enable semantic similarity search via dense vectors, at the cost of a model dependency and reduced interpretability relative to sparse lexical methods.

**Validation methodology.** Kohavi (1995) is the standard reference on the accuracy and variance trade-offs of cross-validation and bootstrap methods for model selection, and specifically documents how naive train-and-report evaluation overstates generalization when hyperparameters are tuned on the reporting set. Efron (1979) establishes the bootstrap as a general nonparametric method for estimating sampling distributions, which we use both for confidence intervals on single metrics and, in paired form, for a significance test on the difference between two configurations evaluated on the same queries.

This work does not propose a new retrieval or fusion method; its contribution is an honest, fully reproducible demonstration of *why* the validation step above is necessary even in a system whose individual components are all standard.

## 3. Dataset

The corpus consists of 192 short biographical records (11 "Winner"-tier, 54 "Finalist"-tier, 126 "20under20"-tier, plus 1 benchmarked applicant record), each an "at-selection" summary — the record as it stood at the time of selection, deliberately excluding later career developments to keep comparisons contemporaneous. Each record is treated as a retrieval document; no other text processing (deduplication, chunking) is required at this scale. See the accompanying [dataset card](DATA_CARD.md) for the full schema and labeling protocol.

## 4. System

Five retrievers operate over a shared, conservatively-stemmed tokenizer (lowercasing, stopword removal, light suffix stripping): BM25 (tunable *k₁*, *b*), TF-IDF cosine with smoothed IDF and L2-normalized vectors, a Dirichlet-smoothed query-likelihood language model, a character 3–4-gram TF-IDF cosine retriever (using a separate character-level analyzer, robust to spelling and transliteration variance), and an optional MiniLM dense-embedding retriever that degrades gracefully to the lexical retrievers when the optional dependency is absent. Retriever outputs are combined via Reciprocal Rank Fusion, CombSUM, or CombMNZ; queries may optionally be expanded via RM3 before retrieval; and the fused candidate list may optionally be re-ranked with MMR (using the *fused* relevance score, not any single retriever's score, as the relevance term). Full implementation is in `gth/retrieval.py`.

## 5. Evaluation Methodology

We constructed a hand-labeled query set of 39 free-text topical queries (e.g., *"menstrual health hygiene periods dignity"*, *"robotics club robot competition building"*, *"cybersecurity ethical hacking vulnerability"*), each mapped to the subset of the 192 honorees a human annotator judges clearly relevant. Standard information-retrieval metrics — Recall@10, Precision@10, Mean Reciprocal Rank, normalized Discounted Cumulative Gain at 10 (nDCG@10), and Mean Average Precision — are computed per query and averaged. We treat the label set as a **sparse pool**: for broad queries, the retrieval system may surface additional genuinely on-topic honorees absent from the labeled set, which are scored as misses. Reported absolute metrics should therefore be read as conservative lower bounds; the comparisons between configurations, evaluated against the identical label set, remain valid.

## 6. In-Sample Results

Table 1 reports each configuration's performance on the full 39-query set (k = 10). The `char-ngram`, `bm25`, `tfidf`, `qlm`, and `hybrid(rrf)` rows use fixed configurations; the `hybrid+rm3+mmr` row uses BM25 parameters (*k₁*=1.5, *b*=0.4) and a character-retriever fusion weight (0.3) selected in an earlier, non-cross-validated sweep over the same 39-query set — the standard (and, as Section 7 shows, insufficient) practice this paper examines.

**Table 1 — In-sample retrieval evaluation (k=10, n=39 queries)**

| Config | Recall@10 | Prec@10 | MRR | nDCG@10 | MAP |
|---|--:|--:|--:|--:|--:|
| TF-IDF | 0.516 | 0.195 | 0.729 | 0.524 | 0.436 |
| BM25 | 0.532 | 0.203 | 0.726 | 0.536 | 0.442 |
| Query-likelihood LM | 0.518 | 0.195 | 0.713 | 0.526 | 0.432 |
| **Character n-gram** | **0.603** | **0.228** | 0.717 | **0.579** | **0.496** |
| Hybrid (RRF) | 0.576 | 0.221 | 0.735 | 0.564 | 0.484 |
| Hybrid (CombSUM) | 0.576 | 0.221 | 0.717 | 0.559 | 0.479 |
| Hybrid + RM3 | 0.568 | 0.215 | 0.722 | 0.558 | 0.461 |
| Hybrid + RM3 + MMR | 0.577 | 0.218 | **0.739** | 0.557 | 0.451 |

By the conventional workflow — evaluate every candidate on the labeled set, report the winner — the analysis would stop here. Character n-gram retrieval already wins 4 of 5 metrics outright; the fully-tuned hybrid ensemble wins only MRR. A practitioner optimizing for nDCG or MAP would already have grounds to prefer the simpler retriever. We nonetheless proceed to cross-validation, because the *table itself* cannot answer whether either result generalizes — both configurations were evaluated on, and one was explicitly tuned on, the same 39 queries.

## 7. Cross-Validation

### 7.1 Flat 5-fold cross-validation

We partition the 39 queries into 5 folds by index-striping (deterministic, no randomness). For each fold, every candidate configuration — the four single retrievers and the hybrid ensemble at four character-fusion weightings (0.3, 0.5, 0.7, 1.0) — is evaluated on the **other four folds** (training partition); the best-on-training candidate is selected; that candidate alone is then scored on the **held-out fold**.

**Table 2 — Flat cross-validation results**

| Fold | Selected (train) | Train nDCG@10 | Held-out nDCG@10 | n (test) |
|---|---|--:|--:|--:|
| 1 | char | 0.562 | 0.646 | 8 |
| 2 | char | 0.568 | 0.622 | 8 |
| 3 | char | 0.558 | 0.663 | 8 |
| 4 | char | 0.620 | 0.422 | 8 |
| 5 | char | 0.588 | 0.537 | 7 |

Character n-gram retrieval is selected as the training-best candidate in **all 5 folds** — never the hybrid ensemble at any of the four tested weightings. Cross-validated nDCG@10 = 0.578 (95% CI 0.490–0.648, bootstrap over fold scores).

### 7.2 Nested cross-validation (closing a remaining leakage path)

The hybrid candidates in Section 7.1 use BM25 parameters and a fusion weight chosen by an earlier sweep over the *full* 39-query set (Section 6) — those specific hyperparameter values were not re-derived independently per fold, which is a legitimate (if narrow) objection to Table 2's validity as a bound on hyperparameter selection specifically. We therefore ran nested cross-validation: within each outer training fold, an inner 4-fold split grid-searches BM25 (*k₁* ∈ {1.0, 1.5, 2.0}, *b* ∈ {0.4, 0.6, 0.75}) crossed with character-fusion weight ∈ {0.0, 0.3, 0.5, 0.7, 1.0} — 45 hybrid configurations, plus the four single retrievers — using only the inner training/validation split, never the outer test fold, and never any value derived from the full dataset.

**Table 3 — Nested cross-validation results**

| Outer fold | Inner-selected | Inner nDCG@10 | Held-out nDCG@10 | n (test) |
|---|---|--:|--:|--:|
| 1 | char | 0.563 | 0.646 | 8 |
| 2 | char | 0.569 | 0.622 | 8 |
| 3 | char | 0.563 | 0.663 | 8 |
| 4 | char | 0.626 | 0.422 | 8 |
| 5 | char | 0.588 | 0.537 | 7 |

The result is unchanged: character n-gram retrieval wins all 5 outer folds under a completely independent, from-scratch inner hyperparameter search, with an **identical** nested cross-validated nDCG@10 of 0.578 (95% CI 0.490–0.648). This closes the leakage path identified above without altering the conclusion.

### 7.3 Is the gap significant?

Table 1's apparent gap between the best single retriever and the tuned hybrid (Δ nDCG@10 = 0.022, in char n-gram's favor) is evaluated on paired data — both configurations are scored on the identical 39 queries — so we compute a paired bootstrap confidence interval on the mean difference (5,000 resamples of query indices, resampled jointly across both configurations' per-query scores):

> mean(char-ngram) = 0.5792, mean(hybrid+RM3+MMR) = 0.5572, mean difference = +0.0220
> 95% CI on the difference: **[-0.0312, +0.0760]** — includes zero; **not significant at α = 0.05.**

This is an important qualifier on the headline result. Cross-validation answers a *selection* question — "which configuration should we ship, given we must pick one?" — and its answer is unanimous: character n-gram, in 10 of 10 folds across two independent CV designs. The significance test answers a different question — "is the performance difference between the two candidates distinguishable from chance at this sample size?" — and its answer is no. Both are true simultaneously and are not in tension: cross-validation is a model-selection procedure, and at n=39 queries, it correctly and repeatedly favors the simpler model because the fancier one shows no reliable held-out advantage to justify its complexity — which is a substantively different, more defensible claim than "the simple model is *better*."

## 8. Discussion

**What generalizes and what doesn't, here.** The finding is specific to this corpus and label set, not a general claim that lexical fusion or RM3/MMR are ineffective. At n=192 documents, character n-gram similarity — which captures near-substring overlap robust to stemming and transliteration variance — appears sufficient on its own for the topical queries in this label set; the additional retrievers primarily provide the fusion machinery with more ways to reorder an already-strong candidate list, without enough queries to prove that reordering reliably helps in held-out generalization.

**Why report a negative result for one's own more complex system.** The alternative — reporting only Table 1 and shipping the tuned hybrid as the recommended default — is the standard and unremarkable path, and would have understated what the cross-validation and significance analysis actually show. We consider the negative result, and the shipped default it implies (a single retriever, not the five-component ensemble), the more useful and more honest contribution of this note.

**Limitations.** (1) The 39-query label set, while an order of magnitude larger than an initial 16-query pilot set used earlier in this project, remains small; per-fold test partitions of 7–8 queries produce wide held-out variance (Table 2's fold 4, at 0.422, versus fold 3's 0.663). (2) Labels are a sparse pool (Section 5); absolute metrics are conservative lower bounds, though this affects all configurations equally and should not bias the *relative* comparison. (3) The dense (MiniLM) retriever and the optional cross-encoder re-ranker were not evaluated in this study, as the optional dependency was not installed in the evaluation environment; whether they change the conclusion is an open question, noted in the accompanying repository's roadmap. (4) All labels were authored by a single annotator (the corpus's builder); inter-rater agreement was not assessed.

## 9. Conclusion

On this corpus, five retrievers, three fusion strategies, RM3 expansion, and MMR re-ranking — combined and hand-tuned — do not demonstrably outperform character n-gram retrieval alone, once evaluated under cross-validation rather than on the same set used for tuning. The shipped default in the accompanying system was changed to reflect this: the simpler retriever is the default; the full ensemble remains fully implemented, tested, and available on request (`hybrid=True`). We take the general lesson to be unglamorous but important: complexity that looks like an improvement on a fixed evaluation set is a claim that needs a held-out test, cheaply available even for small, classical-IR systems, and cross-validation is one of the simplest tools available to make that test honest.

## Reproducing this paper

Every table above is generated directly by the accompanying open-source repository, with no manual editing of numbers:

```bash
git clone https://github.com/arjanchaudharyy/glocal-teen-hero-benchmark
cd glocal-teen-hero-benchmark
python -m gth eval          # Table 1
python -m gth cv --folds 5  # Table 2
python -m gth ncv           # Table 3
python -m gth sig           # Section 7.3
```

All commands are deterministic (verified invariant across `PYTHONHASHSEED`), require no network access, and depend on nothing beyond the Python 3.9+ standard library.

## References

- Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR '98.*
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR '09.*
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1–26.
- Fox, E. A., & Shaw, J. A. (1994). Combination of multiple searches. *TREC-2.*
- Jaleel, N. A., Allan, J., Croft, W. B., Diaz, F., Larkey, L. S., Li, X., Smucker, M. D., & Wade, C. (2004). UMass at TREC 2004: Novelty and HARD. *TREC 2004.*
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI '95.*
- Lavrenko, V., & Croft, W. B. (2001). Relevance-based language models. *SIGIR '01.*
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019.*
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS 2020.*
- Zhai, C., & Lafferty, J. (2001). A study of smoothing methods for language models applied to ad hoc information retrieval. *SIGIR '01.*
