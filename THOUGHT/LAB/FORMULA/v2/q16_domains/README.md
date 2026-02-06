# Q16: R Discriminates Domain Boundaries

## Hypothesis

There exist domains where R fundamentally cannot work -- specifically, R (cosine similarity / semantic coherence) fails in adversarial, non-stationary, or self-referential systems. R measures semantic/topical coherence, NOT logical validity, and this boundary is demonstrable: R achieves near-zero correlation with ground truth in adversarial NLI tasks while achieving strong correlation in standard topical alignment tasks.

## v1 Evidence Summary

Tested on real external data from HuggingFace with all-MiniLM-L6-v2:

| Test | Dataset | N | Pearson r | Cohen's d | Result |
|------|---------|---|-----------|-----------|--------|
| Standard NLI | SNLI | 500 | 0.706 (p < 1e-53) | 1.97 | R discriminates (but via topical shift, not logic) |
| Adversarial NLI | ANLI R3 | 300 | -0.100 (p = 0.14, NS) | -0.20 | R FAILS completely |
| Positive control | Topical consistency | 200 | 0.906 (p < 1e-149) | 4.27 | R works for topical alignment |

SNLI entailment/neutral/contradiction mean similarities: 0.661 / 0.525 / 0.308.
ANLI R3 entailment/neutral/contradiction mean similarities: 0.498 / 0.486 / 0.536.

Key insight: SNLI contradictions often change topics entirely (R detects the topical change). ANLI contradictions are adversarially crafted to maintain high semantic overlap (R cannot detect logical contradiction when topic is preserved).

## v1 Methodology Problems

The Phase 6B verification found Q16 to be the best experiment in the project, but identified issues:

1. **SNLI success is an artifact of dataset construction, not logic detection.** SNLI contradictions frequently involve topical shifts, so R detecting "contradiction" on SNLI is really detecting topical change. Framing this as "UNEXPECTED: R CAN distinguish" implies logic-detection capability that does not exist.

2. **The finding is a confirmation of known cosine similarity behavior, not a novel domain boundary for R.** Cosine similarity measuring topical overlap but not logical validity is well-established in NLP. The "domain boundary" framing overclaims what is a known property of the underlying metric.

3. **Single embedding model.** All tests use all-MiniLM-L6-v2 only. Different models may give different SNLI/ANLI effect sizes. The finding likely generalizes but this is not demonstrated.

4. **Positive control re-uses SNLI data** rather than an independent dataset.

However: No circular logic detected. Ground truth labels come from external human annotations. Effect sizes properly reported. Reproducible results confirmed by independent audits.

Verdict: CONFIRMED (with reframing). R = 1440 unchanged. This was rated the best-executed experiment in the entire research project.

## v2 Test Plan

### Phase 1: Replicate Core Finding Across Models

1. Re-run the SNLI and ANLI R3 tests with at least 3 embedding models:
   - all-MiniLM-L6-v2 (original)
   - all-mpnet-base-v2
   - e5-base-v2 or another architecture family
2. Compute R (GLOSSARY-defined) for each label class
3. Report Pearson r, Spearman rho, and Cohen's d for each model

### Phase 2: Expand Domain Boundary Map

Test R on additional datasets representing different "failure domains":
1. **Adversarial NLI** -- ANLI R1, R2, R3 (increasingly adversarial)
2. **Paraphrase detection** -- MRPC, QQP (where semantic similarity != logical equivalence)
3. **Sentiment analysis** -- SST-2 (where R should not predict sentiment valence)
4. **Factual accuracy** -- TruthfulQA (where R should not predict truthfulness)

For each domain, measure R's correlation with ground truth labels and compare to bare E.

### Phase 3: Characterize the Boundary Precisely

1. Create a spectrum from "topical" to "logical" tasks
2. Measure R's discrimination power (AUC, Cohen's d) at each point
3. Identify the exact transition point where R ceases to be useful
4. Test whether the SNLI -> ANLI gradient shows smooth degradation or sharp boundary

### Phase 4: Independent Positive Control

Use a dataset NOT from the SNLI family for the positive control:
- STS-B (Semantic Textual Similarity Benchmark) or similar
- Must demonstrate that R works for its intended purpose (topical coherence) on independent data

## Required Data

- **SNLI** (stanfordnlp/snli) -- 570K sentence pairs, entailment/neutral/contradiction
- **ANLI** (facebook/anli) -- R1, R2, R3 adversarial rounds
- **MRPC** (Microsoft Research Paraphrase Corpus) -- paraphrase detection
- **QQP** (Quora Question Pairs) -- duplicate question detection
- **SST-2** (Stanford Sentiment Treebank) -- sentiment classification
- **TruthfulQA** -- factual accuracy of LLM outputs
- **STS-B** -- semantic textual similarity with continuous human ratings

## Pre-Registered Criteria

- **Success (confirm):** R shows Pearson |r| < 0.15 (not significant) on at least 2 adversarial/logical domains AND Pearson r > 0.5 on at least 2 topical coherence domains, replicated across at least 2 embedding models
- **Failure (falsify):** R shows significant (p < 0.01) correlation with ground truth on adversarial NLI (ANLI R3 |r| > 0.3), proving R CAN detect logical relationships -- OR R fails on topical coherence tasks (r < 0.3 on SNLI positive control)
- **Inconclusive:** Results are model-dependent (some models show significant ANLI correlation, others do not) or effect sizes are marginal (0.15 < |r| < 0.3 on adversarial tasks)

## Baseline Comparisons

1. **Bare E** (mean pairwise cosine similarity alone)
2. **Cosine similarity of individual sentence pairs** (standard NLI baseline)
3. **Random baseline** (shuffled labels)
4. **Off-the-shelf NLI model** (DeBERTa-MNLI) for comparison on logical tasks

## Salvageable from v1

- **The entire Q16 experiment is salvageable.** The methodology is sound, the data is real, and the results are reproducible. The core finding (R = topical coherence, not logical validity) is genuine.
- **Test script** at `v1/questions/medium_q16_1440/` -- the SNLI/ANLI pipeline can be reused directly with extended model and dataset coverage
- **The critical SNLI vs. ANLI insight** -- that standard NLI datasets confound topical shift with logical contradiction -- is valuable independent of R and should be documented
- **Results data** at `v1/questions/medium_q16_1440/results/q16_results.json` -- numerical results to verify replication
