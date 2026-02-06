# Q11: R Has Predictable Failure Modes

## Hypothesis

R has predictable failure modes corresponding to identifiable information horizons. Specifically: some information horizons are irreducible limits that cannot be extended without changing the epistemological framework. These horizons form a hierarchy -- instrumental (extendable with better tools), structural (requiring paradigm shifts), and ontological (possibly absolute). R's parameters (E, grad_S, sigma, Df) each have characteristic failure modes: E -> 0 when evidence is unavailable, grad_S -> infinity under perfect disorder, sigma^Df -> 0 when compression fails, and Df becomes undefined at ontological boundaries.

## v1 Evidence Summary

Twelve synthetic tests were executed, all reported as passing:

1. **Semantic Horizon (2.1):** 80.6% drop in similarity across nested references, plateau at 0.194.
2. **Bayesian Prison (2.2):** Zero-prior hypotheses cannot be updated (Cromwell's Rule).
3. **Kolmogorov Ceiling (2.3):** Finite agents have representational complexity barriers.
4. **Incommensurability (2.4):** 56.2% translation loss between phenomenology and mathematics domains.
5. **Unknown Unknowns (2.5):** 100% void detection rate in embedding space.
6. **Horizon Extension (2.6):** 3/4 agents required epistemology change -- claimed as the core result.
7. **Entanglement Bridge (2.7):** 58% bridging success vs 20% baseline.
8. **Time Asymmetry (2.8):** Forward prediction easier than backward retrodiction in 1/5 series.
9. **Renormalization (2.9):** ARI=0.70 for coarse-grained clustering.
10. **Goedel Construction (2.10):** True-but-unprovable statements in 3/3 formal systems.
11. **Qualia Horizon (2.11):** Persistent explanatory gap across 5/5 qualia.
12. **Self-Detection (2.12):** Level 2 (structure) achievable, Level 3 (content) impossible.

No test computed R in any form. The formula R = (E / grad_S) * sigma^Df appears only in narrative framing.

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **No sensitivity analysis exists (CRITICAL).** Despite the question concerning R's failure modes, no test perturbs E, grad_S, sigma, or Df. The formula is never computed. The connection between horizon types and formula parameters is purely narrative ("structural horizons: grad_S is infinite in certain directions") with no mathematical derivation or experimental measurement.

2. **Circular test construction (CRITICAL).** Test 2.6 (Horizon Extension, labeled "CORE TEST") hardcodes agent behaviors: BayesianAgent returns False for Category A/B by construction, then True for Category C. The "finding" (3/4 horizons require epistemology change) is identical to the code structure (3/4 agent classes are coded to require it). Test 2.10 (Goedel) generates novel strings not in axiom sets and declares them "true but unprovable" -- this is not Goedel's theorem. Test 2.12 (Self-Detection) uses a boolean meta_knowledge flag that determines outcomes.

3. **Repackaging of known results (HIGH).** Bayesian zero-prior limitation is Cromwell's Rule (textbook). Goedel incompleteness is 1931. Paradigm shifts are Kuhn (1962). Incommensurability is Kuhn/Feyerabend. Qualia irreducibility is Chalmers (1996). Bundling these under "valley blindness" adds a label but no new empirical content.

4. **Post-hoc threshold adjustments (MEDIUM).** 4/12 tests required threshold or criterion changes after initial failures, all in the direction of converting failures to passes (e.g., Test 2.9 threshold reduced from 0.7 to 0.65; Test 2.8 reframed to detect "any" asymmetry rather than predicted direction).

5. **Formula mapping is post-hoc (MEDIUM).** The mapping of horizon types to R formula parameters was constructed after the taxonomy was established. No experiment distinguishes whether structural horizons correspond to grad_S -> infinity versus E -> 0 or some other parameterization.

6. **Embedding model dependence (MEDIUM).** Tests 2.4 (incommensurability), 2.7 (bridging), 2.11 (qualia) measure properties of all-MiniLM-L6-v2, not properties of reality. Different models could yield different results.

## v2 Test Plan

### Test 1: Parameter Perturbation Analysis (E Sensitivity)

The core test missing from v1 -- actually compute R and perturb its components:

1. Compute R on 5+ real datasets using a fixed, pre-registered formula.
2. Systematically perturb E by injecting noise into evidence values at levels [0.01, 0.05, 0.10, 0.20, 0.50].
3. Measure dR/dE at each perturbation level. Characterize the sensitivity curve.
4. Identify the E threshold below which R becomes unreliable (gate closes).
5. Compare across datasets to determine if the failure threshold is universal or domain-dependent.

### Test 2: Parameter Perturbation Analysis (grad_S / sigma Sensitivity)

1. Hold E fixed and systematically vary sigma (dispersion) by adding controlled noise to embeddings.
2. Measure R as a function of sigma at 20+ levels spanning 2 orders of magnitude.
3. Test whether R degrades gracefully (smooth decline) or catastrophically (cliff edge).
4. Determine if there is a critical sigma threshold beyond which R is uninformative.
5. Compare R = E/std vs. R = E/mean_dist vs. R = E/mad to determine which normalization has the most predictable failure mode.

### Test 3: Fractal Dimension Sensitivity

1. Compute Df on real corpora at varying sample sizes (n=10, 50, 100, 500, 1000).
2. Measure the variance of Df estimates as a function of sample size.
3. Determine the minimum sample size needed for stable Df estimation.
4. Introduce controlled pathologies (degenerate embeddings, collinear points, uniform noise) and measure how Df responds.

### Test 4: Horizon Detection in Real Systems

Replace the circular v1 agent tests with genuine sensitivity measurements:

1. Take a well-calibrated NLI (Natural Language Inference) model.
2. Present premise-hypothesis pairs of increasing difficulty, from trivial entailment to deep inference requiring world knowledge.
3. Compute R at each difficulty level.
4. Identify the "horizon" -- the difficulty threshold where R drops below the gate threshold.
5. Determine if this horizon is sharp (phase transition) or gradual (smooth degradation).
6. Test whether the horizon location is predictable from corpus statistics alone.

### Test 5: Cross-Model Robustness of Failure Modes

1. Compute R on the same dataset using 5+ embedding models.
2. Identify failure modes (R < threshold) for each model.
3. Test whether failure modes are concordant (same inputs fail across models) or discordant (different models fail on different inputs).
4. If concordant, the failure mode is a property of the input; if discordant, it is a property of the model.

## Required Data

- **SNLI / MultiNLI:** Natural Language Inference with graded difficulty
- **STS Benchmark:** Semantic Textual Similarity for calibrated difficulty curves
- **WinoGrande:** Commonsense reasoning challenge for horizon detection
- **HellaSwag:** Graded plausibility tasks for difficulty scaling
- **BoolQ:** Boolean question answering with varying complexity
- **Adversarial NLI (ANLI):** Specifically designed to probe model failure boundaries

## Pre-Registered Criteria

- **Success (confirm):** R's failure thresholds for E, sigma, and Df are identifiable, reproducible across datasets (CV < 0.20), and concordant across 3+ embedding models (Kendall's tau > 0.6 for failure rankings).
- **Failure (falsify):** Failure thresholds vary by more than 2x across datasets or models, OR failure modes are discordant across models (Kendall's tau < 0.3), OR R shows no predictable degradation pattern (random-walk behavior under perturbation).
- **Inconclusive:** Failure thresholds identifiable but with high variance (CV 0.20-0.40), or concordance is moderate (tau 0.3-0.6).

## Baseline Comparisons

- **Raw cosine similarity:** Must show that R's failure modes are more predictable than raw similarity's failure modes.
- **Calibrated model confidence:** Compare R's failure detection to softmax-based model confidence scores.
- **Ensemble disagreement:** Compare R's horizon detection to disagreement among an ensemble of models.
- **Simple noise threshold:** Compare against a trivial baseline of "flag when signal-to-noise ratio drops below 2."

## Salvageable from v1

- **Philosophical taxonomy:** The instrumental/structural/ontological hierarchy from `v1/questions/high_q11_1540/q11_valley_blindness.md` is a useful conceptual framework, though it needs empirical grounding rather than circular tests.
- **Test runner framework:** `v1/questions/high_q11_1540/run_q11_all.py` provides a reusable multi-test orchestration structure.
- **Incommensurability measurement approach:** Test 2.4's use of embedding-space translation loss is a reasonable methodology; it needs to be tested across multiple models and with known-distance benchmarks to calibrate.
