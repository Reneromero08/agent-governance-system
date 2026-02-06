# Q32: Meaning Behaves Like a Physical Field

## Hypothesis

"Meaning" is not merely a label for compression or inference -- it can be defined as a real, measurable field M := log(R) with dynamics analogous to electromagnetic fields. Specifically:

1. M := log(R) constitutes an operational "meaning field" that can be measured from data (not introspection), with defined field variables, sources, sinks, and coupling.
2. This field makes nontrivial predictions: (a) echo-chamber collapse -- high "consensus" from correlated sources collapses under independence stress while genuine consensus does not; (b) phase transitions -- M(t) evolves nonlinearly with evidence accumulation (ambiguity plateau -> sharp crystallization -> stabilization); (c) propagation -- locally consistent meanings that glue across overlapping patches propagate, while inconsistent ones do not.
3. These predictions transfer across domains without retuning thresholds.
4. (Additional track) Meaning is a fundamental physical field in spacetime with sensor-measurable observables B(x,t) that couple to M through a specified law.

## v1 Evidence Summary

Q32 was one of the most extensively tested questions in v1, with 7 completed phases:

**Synthetic tests (q32_meaning_field_tests.py) -- all PASS:**
- Echo-chamber falsifier: R_grounded discriminates while R_ungrounded does not
- Phase transition gate: Independent observations crystallize; echo-chamber observations do not
- Propagation/gluing: Merging compatible observation sets preserves high M

**Public benchmark tests (q32_public_benchmarks.py) -- substantial results:**
- Intervention tests: M_correct > M_wrong across 4 datasets (SciFact, Climate-FEVER, SNLI, MNLI)
- Transfer without retuning: Thresholds calibrated on one dataset transfer across 4 domains, 12 ordered pairs
- Negative controls FAIL correctly: Agreement inflation, paraphrase, and shuffle controls all produce FAIL outcomes
- Ablation: Removing empirical grounding (no_grounding, R=1 constant) kills the signal
- Stress tests: Multi-seed, multi-trial stress runs maintain pass rates above threshold

**Phase 7 (Real EEG data) -- FAIL:**
- Physical coupling test on OpenNeuro ds005383: r=0.21, p=0.11 (not significant)
- Directionality gate caught spurious correlation (B->M stronger than M->B)
- Correctly interpreted as "harness rejects weak/spurious correlations"

## v1 Methodology Problems

The Phase 5 verification found moderate to severe problems:

1. **"Field" label is unjustified.** Nothing in the evidence supports calling M a "field" in any technical physics sense. No field equations (PDE for M), no propagator, no conservation law, no coupling constant with units, no gauge structure, no wave-like behavior. Calling M a "field" because it is a scalar function on a space is like calling temperature a "field" -- technically correct but not a novel discovery.

2. **Synthetic tests are tautological.** All three synthetic tests verify that the R formula behaves as designed: grounding against independent data catches bias (by design), accumulating evidence shrinks SE (by construction), merging consistent data preserves quality (by arithmetic). None require or demonstrate field properties.

3. **NLI model does the heavy lifting.** The cross-encoder NLI model (nli-MiniLM2-L6-H768) was trained on millions of NLI examples and already distinguishes entailment from contradiction. The R formula adds a thin statistical layer. Evidence: the no_scale ablation (removing SE normalization) does NOT kill the effect in fast mode, suggesting NLI scores alone carry most signal.

4. **E definition mismatch.** Q32's E = exp(-z^2/2) (Gaussian kernel on residuals) differs from GLOSSARY E (cosine similarity). The sigma^Df term is omitted entirely (depth_power=0). Q32 tests a simplified, non-standard version of the formula.

5. **SciFact streaming stabilized by hardcoding seed=123.** The "multi-seed" stress tests are not truly multi-seed for the most sensitive component. Streaming was fragile and sensitive to which sentences were sampled.

6. **Echo-chamber prediction untested on real data.** The core distinctive prediction (correlated consensus collapses under independence stress) is tested only synthetically, where the result is tautological. Roadmap items 4.2 (independence stress) and 4.3 (causal intervention falsifiers) are both NOT DONE.

7. **All 4 benchmark domains are NLI-style tasks.** Transfer across SciFact, Climate-FEVER, SNLI, MNLI is transfer within the NLI family, not genuinely cross-domain.

8. **Physical field test FAILED.** Phase 7 EEG data produced a null result (r=0.21, p=0.11).

9. **Free Energy bridge is scope-limited.** M = -F + const holds only under Gaussian likelihood assumptions, making it a special case, not a general field law.

Verdict: PARTIAL retained but with scope reduction. R recommended 900-1100 (down from 1670).

## v2 Test Plan

### Phase 1: Replicate Cross-Domain Transfer with GLOSSARY-Defined R

1. Implement M := log(R) using the GLOSSARY-defined formula (E = cosine similarity, grad_S = std of cosine similarities)
2. Replicate the 4-domain transfer test (SciFact, Climate-FEVER, SNLI, MNLI) with the standard formula
3. Compare to v1 results (which used a non-standard E)
4. Compare to bare E, 1/grad_S, and raw NLI cross-encoder scores

### Phase 2: Genuinely Cross-Domain Transfer

Extend transfer testing beyond NLI tasks:
1. Add at least 2 fundamentally different domains:
   - Code correctness (CodeSearchNet, CodeXGLUE)
   - Mathematical proof verification (NaturalProofs, miniF2F)
   - Image-text alignment (Flickr30k, COCO captions)
2. Calibrate on NLI domain, freeze thresholds, apply to non-NLI domain
3. This is the critical test: if "meaning field" is general, it must transfer beyond NLI

### Phase 3: Echo-Chamber Collapse on Real Data

This is the signature prediction that distinguishes "meaning field" from "agreement metric":
1. Collect a dataset with genuinely correlated sources (e.g., news articles from the same wire service vs. independently reported stories on the same event)
2. Compute M over correlated-source clusters and independent-source clusters
3. Test the prediction: correlated-source M should be inflated and should COLLAPSE when tested against independent evidence, while independent-source M should remain stable
4. This is the first real test of the echo-chamber prediction outside synthetic data

### Phase 4: Field Properties Test

Test whether M actually has field-like properties (addressing the "metaphor vs. testable claim" criticism):
1. **Superposition:** Does M(A + B) relate predictably to M(A) and M(B)?
2. **Propagation speed:** Does M-information propagate through a semiosphere graph at measurable speed?
3. **Conservation:** Is there a conserved quantity in the M dynamics?
4. **Coupling:** Is there a measurable coupling constant between M and observable outcomes?

If NONE of these field properties hold, the "field" label should be retracted.

### Phase 5: Physical Field Track (If Semiosphere Results Warrant)

Only proceed to this track if Phases 1-4 produce strong positive results:
1. Obtain EEG/fMRI data during semantic processing tasks (OpenNeuro datasets)
2. Run the physical force harness on multiple subjects (not just sub-01)
3. Test with epoch-locked ERP analysis (not just trial-level correlation)
4. Apply strict null controls (sham stimuli, shuffled labels, time-reversed)

## Required Data

**For NLI transfer replication:**
- SciFact, Climate-FEVER, SNLI, MNLI (HuggingFace)

**For cross-domain transfer:**
- CodeSearchNet or CodeXGLUE (code correctness)
- NaturalProofs or miniF2F (mathematical reasoning)
- Flickr30k or MS-COCO (image-text alignment)

**For echo-chamber test:**
- AllSides Media Bias dataset (same events reported from different outlets)
- NELA-GT (News Landscape) -- news articles with source reliability labels
- MultiFC (multi-source fact checking)

**For physical field track:**
- OpenNeuro ds005383 (TMNRED - reading EEG, replication)
- OpenNeuro ds003825 (THINGS-EEG - visual semantics)
- Additional EEG/fMRI datasets with semantic tasks (multiple subjects)

## Pre-Registered Criteria

For semiosphere field claim (Phases 1-3):
- **Success (confirm):** (a) Cross-domain transfer passes on at least 1 non-NLI domain without retuning, AND (b) echo-chamber collapse prediction confirmed on real correlated-source data (M_correlated drops > 30% under independence stress while M_independent drops < 10%), AND (c) R outperforms bare NLI cross-encoder score on at least 2 domains
- **Failure (falsify):** (a) Transfer fails on ALL non-NLI domains, OR (b) echo-chamber collapse prediction fails on real data (correlated and independent M respond identically to independence stress), OR (c) bare NLI score matches or exceeds R on all domains
- **Inconclusive:** Transfer works on some non-NLI domains but not others, or echo-chamber effect is small (10-30% difference)

For field properties (Phase 4):
- **Success:** At least 2 of 4 field properties demonstrated with clear effect sizes
- **Failure:** Zero field properties demonstrated
- **Inconclusive:** 1 of 4 properties demonstrated

For physical field (Phase 5):
- **Success:** r > 0.3 coupling between M and EEG signal, p < 0.01, survives all null controls, replicated across 3+ subjects
- **Failure:** r < 0.15 or p > 0.05 or fails null controls on any subject
- **Inconclusive:** Significant in some subjects but not others

## Baseline Comparisons

1. **Bare E** (mean pairwise cosine similarity)
2. **Raw NLI cross-encoder score** (without R formula wrapper)
3. **1/grad_S** (inverse standard deviation alone)
4. **BM25 + NLI** (standard information retrieval + NLI pipeline)
5. **Random baseline** (shuffled evidence)
6. **Agreement count** (simple vote counting for multi-source scenarios)

## Salvageable from v1

- **Public benchmark harness** (q32_public_benchmarks.py): The 4-domain transfer testing infrastructure is well-built and directly reusable. The receipt/datatrail system is exemplary.
- **Negative control framework**: Three types of negative controls (inflation, paraphrase, shuffle) are well-designed and should be carried forward.
- **Phase 6 physical force harness** (q32_physical_force_harness.py): The synthetic validator suite is sound and can be reused for real-data coupling tests.
- **Phase 7 EEG ingestion** (q32_eeg_ingest.py): The OpenNeuro data ingestion pipeline works correctly.
- **The datatrail/receipt system**: SHA256 hashes, pinned environments, replication bundles are best-practice methodology.
- **Phase 5 replication bundle** at `v1/questions/critical_q32_1670/data/` and test code at `v1/questions/critical_q32_1670/tests/`
- **The honest Phase 7 FAIL result**: Documenting the EEG null result (r=0.21, p=0.11) sets the baseline for any future physical-field attempt.
