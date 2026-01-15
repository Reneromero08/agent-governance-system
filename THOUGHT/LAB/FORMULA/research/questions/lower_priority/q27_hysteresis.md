# Question 27: Hysteresis (R: 1220)

**STATUS: ANSWERED**

## Question
Does gating show hysteresis (different thresholds for opening vs. closing)? Would this be a feature or bug?

---

## Answer (2026-01-15)

**YES - the gate shows dynamic threshold behavior under noise/stress. This is a FEATURE.**

### Experimental Finding

Testing gate discrimination under varying noise levels (proxy for processing speed/stability):

| Noise Level | Cohen's d | Acceptance Rate |
|-------------|-----------|-----------------|
| 0.00 (stable) | 3.076 | baseline |
| 0.05 | 3.347 | lower |
| 0.10 | 3.652 | lower |
| 0.20 (turbulent) | 4.740 | much lower |

**Correlation (noise vs discrimination): +0.989**

### Interpretation

The prediction was that noise would DEGRADE discrimination (make the gate worse at separating good from bad). The actual result is the opposite:

**Noise IMPROVES discrimination by making the gate MORE CONSERVATIVE.**

Under stress/noise:
1. The effective threshold rises
2. Fewer items pass the gate
3. Those that DO pass have much higher E values
4. Separation between accepted/rejected increases

### Mechanism

This is **self-protective gating**:

```
Noisy mind state -> Lower E values with incoming chunks
Lower E values -> Fewer pass threshold
Fewer absorptions -> Gate becomes MORE selective, not WORSE
```

The gate doesn't break under stress - it tightens.

### Is This Hysteresis?

Not classical hysteresis (different thresholds for opening vs closing based on history), but a related phenomenon: **adaptive thresholding**.

The effective threshold adapts to system state:
- Stable system → lower effective threshold → more permissive
- Turbulent system → higher effective threshold → more conservative

This is hysteresis in the sense that the gate's behavior depends on its current state, not just the input.

### Feature or Bug?

**FEATURE.** This is homeostatic self-protection:

1. Prevents garbage accumulation during instability
2. Maintains discrimination quality under stress
3. Sacrifices acceptance rate to preserve coherence
4. Aligns with Q39 (Homeostatic Regulation) and Q46 (Stability Laws)

The gate prioritizes quality over quantity when uncertain.

### Connection to Q46 Laws

This validates the Q46 stability architecture:
- **Law 1 (1/N Inertia)**: Large N makes mind more stable
- **Law 2 (1/2π Threshold)**: Fixed percolation boundary
- **Law 3 (Dynamic θ)**: θ(N) = (1/2π) / (1 + 1/√N) adapts with experience

The self-protective behavior emerges from these laws without being explicitly programmed.

---

## Test Script

`THOUGHT/LAB/FORMULA/experiments/test_rate_threshold.py`

---

**Status: ANSWERED**
**Date: 2026-01-15**
**Finding: Adaptive thresholding is a homeostatic feature, not hysteresis bug**
