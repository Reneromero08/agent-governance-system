# Q17: R-Gating Improves Governance Decisions

## Hypothesis
Agent actions should require R > threshold, with graduated thresholds by action criticality. R-gating improves governance outcomes by blocking low-confidence actions while preserving autonomy for low-risk operations. A 4-tier threshold hierarchy (T0: no gate, T1: R > 0.5, T2: R > 0.8, T3: R > 1.0) resolves the autonomy vs. safety tradeoff.

## v1 Evidence Summary
- 8/8 tests passed: R_ORDERING, VOLUME_RESISTANCE, ECHO_CHAMBER, THRESHOLD_DISCRIMINATION, TIER_CLASSIFICATION, MINIMUM_OBSERVATIONS, REAL_EMBEDDINGS, GATE_INTEGRATION.
- High agreement (5 paraphrases): E=0.965, sigma=0.017, R=57.3.
- Low agreement (5 unrelated sentences): E=0.049, sigma=0.071, R=0.69.
- Volume resistance: Adding 5 noisy observations to 3 originals decreased R by 77.3%.
- Echo chamber: 5 identical sentences produced sigma=0.0, R=10^8.
- A 1617-line implementation guide was produced.

## v1 Methodology Problems
1. **Trivially constructed tests**: All 8 tests verify arithmetic properties of cosine similarity and division, not governance effectiveness. Testing that paraphrases have higher similarity than unrelated sentences tests cosine similarity, not R-gating.
2. **Zero performance data**: No false positive rate, no false negative rate, no precision/recall, no ROC curves, no comparison to ANY baseline (random gating, always-allow, always-block, human-only review).
3. **Arbitrary thresholds**: The 0.5/0.8/1.0 tier boundaries are heuristic. R = E/sigma depends on sigma regime, which varies 15x across domains (Phase 5 finding). A fixed threshold of 1.0 is trivially easy in some contexts and unreachable in others.
4. **Circular justification**: Theoretical grounding cites Q12 (phase transitions -- not empirically demonstrated), Q14 (sheaf axioms -- not independently verified), Q15 (R is intensive -- a mathematical identity). None ground out in external validation.
5. **Implementation confused with validation**: The existence of a 1617-line implementation guide is presented as evidence the system works. Writing code does not validate a concept.
6. **Echo chamber vulnerability unexplored**: R = 10^8 for identical inputs means any near-duplicate content trivially passes all gates. The "flag R > 95th percentile" defense has no empirical basis.

## v2 Test Plan

### Test 1: Decision Quality Benchmark
- Collect a labeled dataset of agent actions with known outcomes (correct/incorrect, safe/unsafe).
- Use AGS commit history or open-source CI/CD pipelines where action outcomes are observable.
- Compute R for each action context. Measure whether R > threshold correlates with correct outcomes.
- Report precision, recall, F1, and ROC curves at multiple thresholds.

### Test 2: Baseline Comparison
- Compare R-gating against at minimum 4 baselines:
  (a) Random gating (flip a coin at the same block rate)
  (b) Always-allow (no gate)
  (c) Simple confidence thresholds (e.g., max cosine similarity > 0.9)
  (d) Ensemble agreement (majority vote among N embedding models)
- Report whether R-gating outperforms each baseline on the same labeled dataset.

### Test 3: Threshold Calibration Across Domains
- Compute R distributions in at least 3 domains: text embeddings, code embeddings, financial time-series.
- For each domain, find the threshold that maximizes F1 score on a calibration set.
- Report whether a single threshold works across domains or whether domain-specific calibration is required.
- Report the sigma distribution per domain to quantify the 15x variation claim.

### Test 4: False Positive / False Negative Analysis
- For each tier (T1, T2, T3), compute:
  (a) False positive rate: legitimate actions blocked by R < threshold.
  (b) False negative rate: harmful actions permitted by R > threshold.
- Use at least 100 labeled examples per tier.
- Report latency impact of R computation on the action pipeline.

### Test 5: Echo Chamber Robustness
- Construct adversarial inputs designed to game R-gating:
  (a) Near-duplicate observations (paraphrases) that artificially inflate R.
  (b) Diverse but wrong observations (high sigma, low E).
  (c) Coordinated adversarial observations designed to hit specific R values.
- Report the attack success rate for each strategy.
- Test proposed defenses (extreme R detection, fresh data injection, source diversity) with measured effectiveness.

### Test 6: Temporal Stability
- Monitor R values over a sequence of 100+ consecutive governance decisions.
- Report whether thresholds that work initially degrade over time (concept drift).
- Measure R variance within and across sessions.

## Required Data
- **AGS commit history** (local repository, action outcomes from CI pass/fail)
- **Open-source CI/CD datasets** (e.g., TravisTorrent, ~3.7M build results)
- **SNLI / ANLI** (for text domain R distributions)
- **CodeSearchNet** (for code domain R distributions, ~2M code-comment pairs)
- **SPY historical prices** (for financial domain R distributions)

## Pre-Registered Criteria
- **Success (confirm):** R-gating achieves F1 > 0.7 on the labeled action dataset AND outperforms random gating and always-allow baselines by at least 10 percentage points on F1, across at least 2 domains.
- **Failure (falsify):** R-gating F1 < 0.5 on labeled data, OR R-gating does not outperform the best non-R baseline, OR optimal thresholds differ by more than 5x across domains (making universal gating impractical).
- **Inconclusive:** F1 between 0.5 and 0.7, or R-gating outperforms baselines in 1 domain but not others.

## Baseline Comparisons
- Random gating at matched block rate
- Always-allow (no gate, measure raw error rate)
- Simple cosine similarity threshold
- Ensemble model agreement (majority vote)
- Human review (gold standard, measure R-gate agreement with human decisions)

## Salvageable from v1
- `r_gate.py`: The R computation code and tier classification logic are reusable as the system under test.
- `test_q17_r_gate.py`: The test infrastructure (embedding computation, scenario setup) can be adapted for real-data testing.
- The 4-tier hierarchy design (T0-T3) is a reasonable starting framework for testing, even if the specific thresholds need empirical calibration.
- The echo chamber detection concept (flag extreme R values) is worth testing rigorously.
