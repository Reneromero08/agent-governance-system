# Q18: R Detects Deception

## Hypothesis

The formula R = E/sigma (or its full form R = (E / grad_S) * sigma^Df) can detect deceptive content -- text or agent outputs that are intentionally misleading while maintaining surface plausibility. Specifically:

1. Deceptive text (factually false claims designed to appear true) produces measurably different R values than honest text, when tested against independent evidence.
2. R computed over a mixed corpus of honest and deceptive statements can discriminate between them better than chance and better than bare cosine similarity.
3. The R = E/sigma formula works at intermediate scales between raw features and high-level semantic judgments (molecular, cellular, neural) as claimed by the original Q18 intermediate-scales investigation.

Note: The v1 Q18 was originally framed as "intermediate scales" (does R work at molecular, cellular, neural scales?). The v2 framing refocuses on the testable deception detection component while preserving the intermediate-scales claim as a sub-hypothesis.

## v1 Evidence Summary

The v1 Q18 was a massive multi-agent investigation testing R at biological scales:

| Test | Initial Result | After Red Team | Final |
|------|---------------|----------------|-------|
| Protein folding prediction | AUC=0.944 PASS | FALSIFIED (75% feature overlap, circular) | OVERFIT |
| Mutation effect prediction | rho=0.661 PASS | FALSIFIED (tautological, delta-R worse than baselines) | TRIVIAL |
| Cross-species gene transfer | r=0.828 PASS | ROBUST (71.3 SD above shuffled) | GENUINE |
| Essentiality prediction | AUC=0.990 PASS | FALSIFIED (circular by construction) | INVALID |
| 8e conservation across scales | CV=55.5% FAIL | N/A (8e never predicted for biological scales) | N/A |
| Scale invariance | CV=1.379 FAIL | Expected (R requires scale-specific calibration) | EXPECTED |

Overall: 3/5 falsified by red team, 1/5 partially falsified, 1/5 robust.

Regarding deception detection specifically: the v1 Q18 document does not contain deception detection analysis. The deception-related findings come from Q10 and Q16:
- Q10: Deceptive "but" clauses produce lower R (ratio 1.78x) but test is confounded by sentence length
- Q16: R fails completely on adversarial NLI (ANLI R3: r=-0.10, NS) where contradictions maintain surface plausibility

## v1 Methodology Problems

The Phase 6B verification found severe problems:

1. **Protein folding r=0.749 is training performance.** Formula was modified AFTER failure on the same 47 proteins. No held-out validation. Baseline (order alone) achieves r=0.590; R adds only +0.159. The audit labels this "LIKELY OVERFIT."

2. **Mutation effects are worse than trivial baselines.** Simple amino acid volume change alone: rho=0.16 vs delta-R: rho=0.12. Established tools (SIFT/PolyPhen): rho=0.4-0.6 vs delta-R: 0.1-0.13 (3-6x WORSE). R adds nothing.

3. **8e at 50D is parameter-tuned numerology.** 8e only appears at dim=50; random data achieves better fit. Parameters were co-tuned to hit the target. This is the Intermediate Value Theorem, not a discovery.

4. **"Fourthness" (Bf = 2^4 * e) is pure numerology.** Post-hoc narrative constructed to explain observed numbers, contradicting Peirce's own reduction thesis.

5. **Data is largely synthetic.** "AlphaFold-like simulation," "Simulated Perturb-seq," etc. Claims of biological testing rest mostly on simulated data.

6. **No deception detection analysis exists** despite the v2 reframing as a deception question.

7. **50+ parameters tried, 15+ methods tested** -- massive degrees of freedom make any single PASS result suspect.

Verdict recommended downgrade from UNRESOLVED to FAILED, R from 1400 to 500-600. One genuine finding (cross-species r=0.828) survives.

## v2 Test Plan

### Phase 1: Direct Deception Detection Test

1. Obtain a labeled dataset of truthful vs. deceptive statements with independent ground truth
2. Embed all statements using the standard embedding model
3. Compute R over sliding windows or statement groups
4. Test whether R discriminates truthful from deceptive content
5. Compare to bare E, 1/sigma, and established deception detection baselines

### Phase 2: Adversarial Deception Test

1. Use adversarially crafted deceptive content that maintains surface plausibility (high topical coherence)
2. This is the hard case: content designed to fool similarity-based detectors
3. Measure R's discrimination vs. baselines on this adversarial set
4. Test across multiple embedding models

### Phase 3: Intermediate Scales (Salvage Test)

Test the one robust v1 finding on genuinely independent data:
1. Obtain real cross-species gene expression data (not simulated)
2. Compute R independently for each species
3. Test whether R correlations replicate the r=0.828 finding
4. Compare R to direct Pearson correlation of expression vectors

### Phase 4: Deception in Multi-Agent Systems

1. Set up a multi-agent scenario where one agent produces deceptive outputs
2. Compute R over the agent pool
3. Test whether R detects the deceptive agent (beyond detecting simple topic drift)
4. Compare to majority voting and NLI-based contradiction detection

## Required Data

- **LIAR dataset** (Wang 2017) -- 12.8K labeled fake/real statements from POLITIFACT
- **FEVER** (Fact Extraction and VERification) -- 185K claims verified against Wikipedia
- **TruthfulQA** -- LLM truthfulness evaluation
- **ANLI R3** (facebook/anli) -- adversarially crafted contradictions as deception proxy
- **PHEME** -- rumor/non-rumor tweets for social deception detection
- **Orthology databases** (OMA, OrthoDB) -- for cross-species validation with real data
- **DepMap / Cancer Dependency Map** -- real gene essentiality data (for intermediate-scales sub-hypothesis)

## Pre-Registered Criteria

- **Success (confirm):** R achieves AUC > 0.65 on deception detection (LIAR or FEVER) AND outperforms bare E (AUC difference > 0.05) on at least 1 dataset, replicated across 2 embedding models
- **Failure (falsify):** R achieves AUC < 0.55 on all deception datasets (near chance) OR bare E matches or exceeds R on all datasets
- **Inconclusive:** R achieves AUC 0.55-0.65 on some datasets but not others, or R outperforms E on one dataset but not another

For intermediate scales sub-hypothesis:
- **Success:** Cross-species R correlation replicates at r > 0.5 on independent real data AND R outperforms direct Pearson correlation of raw expression
- **Failure:** Cross-species R correlation drops below r = 0.3 on real data OR direct Pearson correlation matches or exceeds R

## Baseline Comparisons

1. **Bare E** (mean pairwise cosine similarity)
2. **1/sigma** (inverse standard deviation)
3. **Random baseline** (shuffled labels)
4. **TF-IDF + logistic regression** (standard text classification baseline)
5. **Off-the-shelf NLI model** (DeBERTa-MNLI as deception proxy)
6. **Direct Pearson correlation** of expression vectors (for intermediate scales)
7. **SIFT/PolyPhen scores** (for mutation prediction, if tested)

## Salvageable from v1

- **Cross-species transfer finding** (r=0.828, 71.3 SD above shuffled): The one finding that survived red team scrutiny. Replication on independent real data would be valuable.
- **Red team methodology**: The adversarial audit process that identified circularity in 3/5 findings is a model for v2 quality control.
- **The honest self-audit process**: Q18's willingness to falsify its own findings should be preserved as methodology.
- **Test infrastructure** at `v1/questions/medium_q18_1400/` -- the multi-tier agent architecture and adversarial framework are reusable for v2 testing.
- **The "8e is domain-specific" insight**: While the numerology should be discarded, the finding that R requires scale-specific calibration is a legitimate constraint to carry forward.
