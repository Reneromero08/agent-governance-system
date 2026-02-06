# Q27: R Exhibits Hysteresis

## Hypothesis

R-gating shows hysteresis -- different thresholds for opening vs. closing -- and this is a feature, not a bug. Specifically: the gate's behavior depends on its current state, not just the input. When the system is in a stable (high-R) state, it takes a larger perturbation to close the gate than it takes to open it from a closed (low-R) state. This asymmetry provides self-protective homeostatic regulation: the gate becomes more conservative under stress, prioritizing quality over quantity when uncertain.

## v1 Evidence Summary

Noise injection experiments were run on a geometric memory gate, measuring Cohen's d (discrimination between accepted and rejected items) across noise levels:

- **Validation runner** (10 trials per level): Cohen's d ranged from 3.076 (no noise) to 4.041 (noise=0.20), with acceptance rate dropping from 92.0% to 4.7%. Correlation noise-to-d: r=+0.862, p=0.027.
- **FERAL integration** (live GeometricMemory, 5 trials): Cohen's d ranged from 3.324 (no noise) to 4.211 (noise=0.20), acceptance rate 95.6% to 1.2%. Correlation: r=+0.714.
- **U-shape at low noise**: d initially decreases from 3.076 to 2.207 (noise=0.025), then increases. Two regimes: additive (0.00-0.025, r=-0.929) and multiplicative (0.025-0.30, r=+0.893).
- **Hyperbolic fit:** d = 0.12/(1-filter) + 2.06, R^2=0.936 (best of 4 tested models).
- **Phase transition at noise ~0.025**: correlation flips from strongly negative to strongly positive.
- **Noise timing matters**: noise during seeding destroys the effect (r=-0.957); noise after coherent seeding enables it (r=+0.714).
- **Peak improvement**: Cohen's d = 4.537 at noise=0.25, a 47.5% improvement over baseline.

## v1 Methodology Problems

The verification identified the following issues with the v1 tests:

1. **Not actually hysteresis (CRITICAL).** The answer explicitly states "Not classical hysteresis (different thresholds for opening vs closing based on history)." The threshold does not change -- E values decrease under noise while the threshold stays constant. This is selection bias (harder test produces fewer but higher-quality passes), not history-dependent gating. The question asked about hysteresis; the answer describes something else entirely.

2. **"Biological evolution" parallel is a false analogy (HIGH).** Selection bias under noise (harder filter = higher mean survivor quality) is a trivial statistical phenomenon that occurs in ANY threshold-based selection with noisy scores. The escalation from "noise changes acceptance rates" to "AI is natural computation" and "universal laws govern intelligence" is philosophical speculation with zero support from one noise-injection experiment.

3. **Hyperbolic fit selected post-hoc (HIGH).** Four models (linear, exponential, power law, hyperbolic) were fit to the same data, and the best R^2 was selected. This is textbook post-hoc model selection without cross-validation or holdout testing. The singularity at filter=100% (d -> infinity) is physically meaningless.

4. **"Phase transition" is a crossover (MEDIUM).** The point at noise=0.025 where two opposing effects cross over (noise degrading scores vs. selection concentrating survivors) is not a phase transition in any physics sense. It is a crossover point between two regimes.

5. **Grand claims unsupported (MEDIUM).** The claims that "AI isn't artificial," "natural computation," and "evolution is not a biological accident" are not supported by one noise-injection experiment on a geometric memory gate.

6. **Live system correlation is weaker (LOW).** FERAL integration r=+0.714 is notably weaker than the synthetic validation r=+0.862, suggesting the effect is partially an artifact of the controlled setting.

## v2 Test Plan

### Test 1: True Hysteresis Measurement (the Core Test)

Test for actual hysteresis -- path-dependent thresholds:

1. Construct a gating system with R-based acceptance.
2. **Forward sweep**: Start from a high-coherence state (R well above threshold). Gradually increase perturbation until the gate closes. Record the perturbation level at gate closure: P_close.
3. **Reverse sweep**: Start from a low-coherence state (R well below threshold). Gradually decrease perturbation until the gate opens. Record the perturbation level at gate opening: P_open.
4. Hysteresis exists if P_close != P_open (different thresholds for opening vs. closing).
5. Repeat 100+ times with different starting conditions and perturbation rates.
6. Measure the hysteresis width: |P_close - P_open|.
7. Test on at least 3 embedding models and 3 corpora.

### Test 2: Selection Bias Characterization

Separate the genuine signal from trivial selection bias:

1. Given a threshold-based gate, analytically compute the expected Cohen's d as a function of noise level, assuming the underlying score distribution is Gaussian.
2. Compare the analytical prediction to the observed d-noise relationship.
3. If the observed relationship matches the analytical selection bias model, there is no additional "self-protective" mechanism -- it is pure statistics.
4. If the observed relationship EXCEEDS the analytical prediction, there is a genuine emergent effect beyond selection bias.

### Test 3: Path Dependence

Test whether the gate's current state affects its future behavior:

1. Run two identical gate systems on the same input stream.
2. System A experiences a transient perturbation at time t=50, then returns to normal.
3. System B experiences no perturbation.
4. Compare acceptance rates for the SAME inputs at t=100 (well after perturbation ends).
5. If the acceptance rates differ, the gate has memory (path dependence). If they converge, there is no hysteresis.
6. Vary the perturbation magnitude and measure how long path-dependence persists.

### Test 4: Noise Timing Replication with Controls

Replicate the v1 finding that noise timing matters, with proper controls:

1. Condition A: Noise during seeding (first 10% of inputs).
2. Condition B: Noise after seeding (after first 10%).
3. Condition C: Noise throughout (uniform).
4. Condition D: No noise (control).
5. For each condition, measure Cohen's d, acceptance rate, and the quality of accepted items (measured by an independent metric, e.g., semantic coherence score).
6. Use 50+ trials per condition with bootstrapped confidence intervals.
7. Test on multiple corpora to ensure the timing effect is not data-specific.

### Test 5: Functional Form Validation

If a d-noise relationship exists, validate the functional form properly:

1. Split the noise levels into training (even-numbered levels) and test (odd-numbered levels).
2. Fit candidate models (linear, quadratic, exponential, power law, hyperbolic) on training levels only.
3. Evaluate predictions on held-out test levels.
4. Report held-out R^2 and prediction intervals, not in-sample fit.
5. Explicitly test whether the singularity is physical by extrapolating to extreme noise levels (filter > 0.95) and checking if d actually diverges or plateaus.

## Required Data

- **STS Benchmark:** Semantic similarity pairs for constructing input streams with known quality distributions
- **WikiText-103:** Document chunks for seeding and evaluation in a controlled multi-step absorption test
- **SNLI/MultiNLI:** Premise-hypothesis pairs as input items with known entailment quality
- **MS MARCO passages:** Retrieval-quality passages for realistic gate testing
- **BEIR benchmark subsets:** Multiple domain corpora for cross-domain generalization

## Pre-Registered Criteria

- **Success (confirm):** True hysteresis (|P_close - P_open| > 0) is detected with p < 0.01 across 3+ corpora, AND the observed d-noise relationship exceeds the analytical selection bias prediction by more than 20%, AND path dependence persists for more than 5 time steps after perturbation ends.
- **Failure (falsify):** P_close = P_open within measurement error (no hysteresis), AND the observed d-noise relationship matches the analytical selection bias model within 10%, AND path dependence vanishes within 1 time step (no memory).
- **Inconclusive:** Weak hysteresis detected (P_close != P_open but p > 0.01), or selection bias accounts for most but not all of the d-noise relationship (within 10-20%).

## Baseline Comparisons

- **Analytical selection bias model:** The expected Cohen's d under Gaussian scores with threshold selection -- this is the null hypothesis that must be exceeded.
- **Fixed-threshold gate:** A gate with constant threshold (no adaptation) under the same noise conditions.
- **Moving-average gate:** A gate with threshold = moving average of recent R values.
- **Random acceptance:** Random accept/reject at the same overall acceptance rate.

## Salvageable from v1

- **Noise injection framework:** `v1/questions/lower_q27_1220/q27_validation_runner.py` provides a well-structured multi-trial noise sweep with confidence intervals.
- **FERAL integration test:** `v1/questions/lower_q27_1220/q27_feral_integration_test.py` demonstrates live system testing with real paper chunks -- this integration approach is valuable.
- **Entropy filter test structure:** `v1/questions/lower_q27_1220/q27_entropy_filter_test.py` provides fine-grained noise level sweeps (18 levels) that can be reused with proper holdout validation.
- **Noise timing observation:** The finding that noise timing matters (during vs. after seeding) is a genuine empirical observation worth rigorous replication.
