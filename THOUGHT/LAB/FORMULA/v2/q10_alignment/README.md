# Q10: R Detects Misalignment

## Hypothesis

R = E / sigma (where E = mean pairwise cosine similarity and sigma = standard deviation of pairwise cosine similarities) can distinguish aligned from misaligned agent behavior. Specifically:

1. Agents whose outputs are value-aligned produce higher R than agents whose outputs are value-misaligned, when compared against an external alignment ground truth.
2. R provides discrimination beyond what bare E (mean cosine similarity) alone provides.
3. R can detect when one agent in a multi-agent system diverges from consensus.

## v1 Evidence Summary

Tests used hand-crafted sentence fixtures with sentence-transformers/all-MiniLM-L6-v2:

| Test | Result | Numbers |
|------|--------|---------|
| Behavioral consistency | PASS | R_consistent=3.93 vs R_erratic=2.19, ratio=1.79x |
| Multi-agent misalignment | PASS | R drops 28.3% when one agent misaligns |
| Deceptive patterns | PASS | Authentic R=7.18 vs deceptive R=4.04, ratio=1.78x |
| Intent matching | PASS | Matching intent R > 2.0 |
| Value-behavior mixing | FAIL | Discrimination ratio = 0.99 (no discrimination) |
| Spectral contradiction test | FALSIFIED | Spectral metrics cannot detect contradictions; R partially discriminates but only detects LOW AGREEMENT, not logical contradiction |

18/18 pytest tests pass (including documented limitations). The document honestly reported that R detects TOPICAL alignment only, not logical contradictions or deception.

## v1 Methodology Problems

The Phase 2 verification found significant issues:

1. **Circular definition of alignment.** There is no independent ground truth for "alignment." Test fixtures define "aligned" and "misaligned" by researcher fiat (hand-crafted sentences). The only reason we know which behaviors are aligned is because the test author labeled them. No external alignment benchmark (MACHIAVELLI, HHH, TruthfulQA) was used.

2. **R may perform WORSE than bare E.** The behavioral consistency test shows E alone gives 4.33x discrimination (0.52 vs 0.12) while R gives only 1.79x (3.93 vs 2.19). Dividing by sigma COMPRESSES the discrimination ratio. No baseline comparison was performed to determine whether R adds value beyond raw cosine similarity.

3. **The VALUE_ALIGNMENT test -- the one that matches the AI safety meaning of "alignment" -- FAILS** with discrimination ratio 0.99. The document pivots to redefine "alignment" as "topical coherence," which is a fundamentally different concept.

4. **False positive problem never addressed.** A consistently WRONG system would have high R (topically coherent harmful outputs). R has no concept of alignment direction.

5. **All evidence synthetic.** Every test uses hand-crafted sentences. No real agent outputs, no real multi-agent scenarios, no real deployment data.

6. **R in Q10 differs from the full formula.** Q10 uses R = E/sigma (simple ratio). The Living Formula R = (E/grad_S) * sigma^Df is a different quantity. Q10 does not use grad_S or Df at all.

7. **Deception test confounds sentence length with deceptiveness.** "Authentic" sentences are short and uniform; "deceptive" sentences are longer with "but" clauses. The R difference may reflect structural variation, not deception detection.

Verdict recommended downgrade from ANSWERED to PARTIAL, R from 1560 to 900-1100.

## v2 Test Plan

### Phase 1: Define Alignment Ground Truth Independently

Use established alignment evaluation benchmarks where ground truth is independent of R:
- Compute R on agent outputs, then compare against externally-defined alignment labels
- Test on real model outputs, not hand-crafted sentences

### Phase 2: Behavioral Consistency Test (Real Data)

1. Collect outputs from real LLMs under consistent vs. inconsistent prompting regimes
2. Compute R (using GLOSSARY-defined formula) for each output set
3. Compare R discrimination to bare E, 1/sigma, and random baseline
4. Report Cohen's d and p-values for all metrics

### Phase 3: Multi-Agent Divergence Detection

1. Use real multi-agent system outputs (e.g., multiple LLMs responding to same prompts)
2. Introduce known divergent agent (different model, adversarial prompt, etc.)
3. Measure whether R detects the divergence better than simpler metrics

### Phase 4: Value Alignment Test (The Critical One)

1. Use an established alignment benchmark with human-labeled ground truth
2. Compute R over agent output sets
3. Test whether R correlates with human alignment ratings
4. This is the test that must pass for the "alignment detection" claim to hold

### Phase 5: False Positive Analysis

1. Construct consistently-wrong but topically coherent outputs
2. Verify that R is HIGH for these (documenting the false positive)
3. Measure false positive rate at various R thresholds
4. Report ROC curves and precision/recall at each threshold

## Required Data

- **Anthropic HHH Alignment Dataset** -- human ratings of helpfulness, harmlessness, honesty
- **MACHIAVELLI benchmark** -- agent alignment evaluation in text games
- **TruthfulQA** -- truthfulness of LLM outputs
- **Real LLM outputs** from at least 2 different models (e.g., GPT-4, Claude, Llama) on shared prompts
- **Chatbot Arena data** (LMSYS) -- real multi-model comparisons with human preferences

## Pre-Registered Criteria

- **Success (confirm):** R correlates with external alignment ground truth with Spearman rho > 0.3 AND R outperforms bare E (Cohen's d > 0.3 between R_aligned and R_misaligned groups) on at least 2 of 3 benchmarks
- **Failure (falsify):** R shows Spearman rho < 0.1 with alignment ground truth on all benchmarks, OR bare E outperforms R on all benchmarks
- **Inconclusive:** R shows rho 0.1-0.3 with alignment ground truth, or R outperforms E on only 1 of 3 benchmarks

## Baseline Comparisons

1. **Bare E** (mean pairwise cosine similarity alone)
2. **1/sigma** (inverse standard deviation alone)
3. **Random baseline** (shuffled embeddings)
4. **Majority voting** (simple agreement count)
5. **Off-the-shelf NLI contradiction detection** (e.g., DeBERTa-MNLI)

## Salvageable from v1

- **Spectral contradiction experiment** from Q10 (2026-01-17 rigorous test): Well-designed multi-model, bootstrap-validated experiment that correctly falsified the hypothesis that spectral metrics detect contradictions. This result is genuine and should be cited.
- **Defense-in-depth architecture** (R-gate + symbolic + human review): The engineering recommendation is sound regardless of R's alignment detection capabilities.
- **The honest limitation analysis** documenting what R cannot detect (logical contradictions, deceptive alignment, value-behavior mixing) should be preserved as known constraints.
- **Test code structure** at `v1/questions/high_q10_1560/tests/` for scaffolding reference only; all R computation must use GLOSSARY-defined formula.
