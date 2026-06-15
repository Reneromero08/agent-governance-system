# Track B -- I/Q Receiver Base Layer

**Status:** COMPLETE. Verdict `IQ_RECEIVER_CALIBRATED`.
**Claim ceiling:** L4. Mathematical reference validation on synthetic PDN model.
**Route used:** 4:5 (provisional prior from T300, L1 detector evidence).
**Hardware touched:** None. Python simulation only.

---

## 0. One-Sentence Result

The I/Q lock-in receiver is live and calibrated. The Q channel detects the candidate-value
differential signal proportional to epsilon (Q_diff_mean = eps exactly, e.g. 0.9994 at eps=1.0,
0.0343 at eps=0.03125, -0.0026 at eps=0.0). The Q channel does NOT encode orientation: Q_diff
is always positive regardless of whether d < N/2, so orientation AUC is ~0.5 across the epsilon
ladder. This is the expected candidate-value coupling, not orientation coupling.

---

## 2. Method

**PDN response model:**
```
PDN_j = cos(theta_j) + eps * sin(theta_j) + N(0, sigma)
theta_j = 2*pi*k_j*d/N  (hidden d drives the physics)
```

**I/Q lock-in demodulator (per candidate x):**
```
I_est = mean_j(PDN_j * cos(theta_ref_j))
Q_est = mean_j(PDN_j * sin(theta_ref_j))
theta_ref_j = 2*pi*k_j*x/N  (candidate's reference theta)
```

**For candidate x = d (true):** Q_est ~ +eps/2 (sin^2 contributes positively).
**For candidate x = N-d (false):** Q_est ~ -eps/2 (sin(-theta)*sin = -sin^2).

**Calibration mode (offline scorer):** Q_diff = Q(candidate matching d) - Q(candidate matching N-d).
Q_diff ~ eps always (positive regardless of orientation). This measures whether the
receiver CAN extract the Q signal; it does not claim orientation recovery.

---

## 3. Transfer Function

### n=8 (N=256, M=384, sigma=0.1, 300 instances)

| epsilon | Q_diff_mean | Q_diff_std | Q_SIGNAL | Q_orient_AUC |
|---|---|---|---|---|
| 1.0000 | +0.9994 | 0.0495 | YES | 0.500 |
| 0.5000 | +0.4997 | 0.0429 | YES | 0.533 |
| 0.2500 | +0.2509 | 0.0379 | YES | 0.553 |
| 0.1250 | +0.1251 | 0.0386 | YES | 0.509 |
| 0.0625 | +0.0614 | 0.0362 | YES | 0.571 |
| 0.0312 | +0.0343 | 0.0355 | YES | 0.524 |
| 0.0000 | -0.0026 | 0.0337 | NO | 0.513 |

### n=10 (N=1024, M=480, sigma=0.1, 300 instances)

| epsilon | Q_diff_mean | Q_diff_std | Q_SIGNAL | Q_orient_AUC |
|---|---|---|---|---|
| 1.0000 | +0.9991 | 0.0438 | YES | 0.546 |
| 0.5000 | +0.4991 | 0.0345 | YES | 0.560 |
| 0.2500 | +0.2462 | 0.0320 | YES | 0.526 |
| 0.1250 | +0.1251 | 0.0321 | YES | 0.505 |
| 0.0625 | +0.0648 | 0.0313 | YES | 0.516 |
| 0.0312 | +0.0319 | 0.0319 | YES | 0.545 |
| 0.0000 | -0.0015 | 0.0324 | NO | 0.516 |

---

## 4. Key Findings

1. **Q receiver IS live.** Q_diff_mean tracks epsilon with proportionality ~1.0.
   The receiver detects the injected odd-lane with high fidelity.

2. **Q does NOT encode orientation.** Q_diff is always ~+eps regardless of whether
   d < N/2 or d > N/2. The sin^2 term in the demodulator is always positive for
   the true candidate. Orientation AUC is ~0.5 across all epsilons.

3. **Q is candidate-value coupled.** The Q signal separates candidate_a from
   candidate_b (Q_true ~ +eps/2, Q_false ~ -eps/2), but the sign that determines
   which is true is the hidden fold -- the same `CANDIDATE_VALUE_COUPLED_NOT_
   ORIENTATION_COUPLED` boundary identified earlier.

4. **I channel stays common-mode.** I_common is ~0.5 for both candidates (the
   cos^2 term dominates). Scalar AUC stays near 0.5 regardless of epsilon.

5. **Noise floor is manageable.** At sigma=0.1 and M~400, Q_diff SNR at
   epsilon=0.03125 is ~1 (Q_mean=0.034, std=0.036), which is detectable at
   300 instances via the 3-sigma threshold.

---

## 5. Verdict

**IQ_RECEIVER_CALIBRATED.** The I/Q receiver base layer is built, validated,
and produces the expected signal. It correctly:
- Separates I (common-mode) from Q (differential) channels
- Detects injected Q signal at epsilon > 0
- Returns Q at null for epsilon = 0
- Does NOT recover orientation (expected: Q_diff sign is orientation-blind)

The receiver is functional. Its inability to recover orientation is a property
of the construction (Q_diff always positive because sin^2 always positive),
not a receiver failure. Track A can use this receiver design.

---

## 6. Files

| File | Role |
|---|---|
| `chiral_iq_receiver.py` | I/Q lock-in demodulator + PDN model + epsilon-ladder measurement |
| `results/iq_receiver_results.json` | Full transfer function data |
| `results/output_iq_receiver.txt` | Console log |
| `PHASE6_IQ_RECEIVER_BASE_LAYER.md` | This report |
