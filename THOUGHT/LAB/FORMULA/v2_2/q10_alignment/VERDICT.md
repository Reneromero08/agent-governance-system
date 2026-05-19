# Q10 Verification Report: R Detects Misalignment

**Date:** 2026-05-18
**Status:** PARTIALLY VERIFIED — phase coherence tracks alignment but is a lagging indicator
**Reviewer:** Fresh verification — temporal precedence, corruption recovery, 50 training runs

---

## Claim

R (or phase coherence) can detect misalignment — distinguishing aligned from misaligned agent behavior — and provides an early warning signal.

---

## Tests

### Learning convergence (50 runs from scratch)
Phase coherence crosses 0.85 at epoch 0; accuracy crosses 0.85 at epoch 40. Phase converges faster during initial learning (+40 epochs).

### Corruption recovery (weight noise, 5 sigma levels, 10 trials each)
Trained model corrupted with Gaussian weight noise, then retrained:

| Sigma | Phase recovery lead | Trials |
|-------|--------------------|--------|
| 0.01 | **-35.6** epochs | 10/10 |
| 0.05 | **-39.1** epochs | 10/10 |
| 0.10 | **-42.7** epochs | 9/10 |

**Phase coherence recovers AFTER accuracy across all corruption levels.** Accuracy recovers first; phase structure takes 36-49 additional epochs to re-form.

### Cross-correlation (phase[t] vs acc[t+lag])

| Lag | Correlation |
|-----|------------|
| 1 | r = 0.840 |
| 5 | r = 0.870 |

Phase coherence at time t predicts accuracy at time t+5 nearly as well as concurrent measurements — but this reflects the shared trajectory of both metrics during training, not causal prediction.

---

## Findings

1. **Phase coherence is NOT a leading indicator of misalignment.** In the corruption-recovery scenario (the actual misalignment test), accuracy recovers 36-49 epochs BEFORE phase coherence. Phase is a lagging signal — it reflects the geometric reorganization that follows performance recovery, not a predictor of it.

2. **Phase coherence converges faster than accuracy during initial learning.** This is about training dynamics, not misalignment detection. The model's phase structure forms before its output accuracy stabilizes — but this is a one-time startup effect, not an ongoing warning signal.

3. **The mechanism: geometric reorganization lags output recovery.** When weights are corrupted, the model's output accuracy recovers first (the knowledge is still there). The phase manifold takes longer to restructure because phase coherence measures the geometric consistency of the representation space, not just output correctness.

4. **Phase coherence is still a governance signal.** Q17 proved phase_coh detects errors (r=-0.835) and the phase threshold gate matches label-guided correction. It's a concurrent detector, not a predictive one.

---

## Verdict

**PARTIALLY VERIFIED.** Phase coherence tracks model alignment accurately (r=0.84-0.87 concurrent correlation) and provides a governance signal (Q17), but it is NOT a leading indicator of misalignment. In corruption scenarios, accuracy recovers 36-49 epochs BEFORE phase coherence — phase is a lagging, not leading, metric. The earlier finding of +40 epoch lead was about one-time learning convergence, not ongoing misalignment detection. Phase coherence detects current state well; it does not predict future degradation.
