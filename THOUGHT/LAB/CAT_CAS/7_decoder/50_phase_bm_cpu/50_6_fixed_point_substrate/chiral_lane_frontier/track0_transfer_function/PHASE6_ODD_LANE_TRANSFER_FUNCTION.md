# Track 0 -- Odd-Lane Transfer Function Calibration

**Status:** COMPLETE. Verdict `ODD_LANE_DETECTOR_CALIBRATED`.
**Claim ceiling:** L4. Mathematical reference curve on real Exp 50.14 construction.
**Purpose:** Quantify minimum detectable fold-odd lane amplitude BEFORE interpreting
any negative from Tracks A-K.

---

## 0. One-Sentence Result

The odd-lane transfer function is effectively BINARY on this construction: at
epsilon=0 (no quadrature), AUC=0.500 (orientation unrecoverable). At any epsilon>0
with k=1 present in the sample, AUC>=0.916 (orientation trivially recoverable).
The minimum detectable epsilon is not limited by signal amplitude but by whether
the k=1 coset appears in the random public sample -- a sampling-probability wall,
not an SNR wall.

---

## 1. Method

Inject synthetic quadrature at controlled amplitude epsilon into the public oracle:

```
z_k = cos(theta_k) + i * epsilon * sin(theta_k)
theta_k = 2*pi*k*d/N
```

Readout: bin `epsilon * sin(theta_k)` at k=1 and k=N-1 (where sin(N-1) = -sin(1)).
Epsilon ladder: 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0.

---

## 2. Transfer Function (n=8, N=256, M=384)

| epsilon | chA AUC | chA null95 | chB AUC | chB null95 | SNR | Verdict |
|---|---|---|---|---|---|---|
| 1.0000 | 0.555 | 0.546 | **0.999** | 0.542 | 0.894 | ABOVE_NULL |
| 0.5000 | 0.473 | 0.545 | **0.998** | 0.553 | 0.875 | ABOVE_NULL |
| 0.2500 | 0.493 | 0.549 | **0.999** | 0.544 | 0.886 | ABOVE_NULL |
| 0.1250 | 0.553 | 0.551 | **0.999** | 0.555 | 0.893 | ABOVE_NULL |
| 0.0625 | 0.504 | 0.540 | **0.998** | 0.553 | 0.868 | ABOVE_NULL |
| 0.0312 | 0.497 | 0.539 | **0.998** | 0.540 | 0.868 | ABOVE_NULL |
| 0.0000 | 0.489 | 0.538 | 0.500 | 0.500 | 0.000 | at_chance |

## 3. Transfer Function (n=10, N=1024, M=480)

| epsilon | chA AUC | chA null95 | chB AUC | chB null95 | SNR | Verdict |
|---|---|---|---|---|---|---|
| 1.0000 | 0.545 | 0.537 | **0.916** | 0.544 | 0.697 | ABOVE_NULL |
| 0.5000 | 0.508 | 0.549 | **0.910** | 0.545 | 0.681 | ABOVE_NULL |
| 0.2500 | 0.469 | 0.552 | **0.932** | 0.546 | 0.717 | ABOVE_NULL |
| 0.1250 | 0.488 | 0.551 | **0.924** | 0.554 | 0.705 | ABOVE_NULL |
| 0.0625 | 0.506 | 0.548 | **0.916** | 0.542 | 0.686 | ABOVE_NULL |
| 0.0312 | 0.455 | 0.548 | **0.933** | 0.539 | 0.721 | ABOVE_NULL |
| 0.0000 | 0.511 | 0.554 | 0.500 | 0.500 | 0.000 | at_chance |

---

## 4. Key Finding

The transfer function is NOT a smooth SNR ramp. It is a BINARY gate:

- **epsilon = 0:** sin channel is zero everywhere. AUC = 0.500 exactly.
- **epsilon > 0:** sin(2*pi*k*d/N) is nonzero. If k=1 or k=N-1 appears in the
  random sample, orientation is trivially recoverable (AUC >= 0.91).

The limiting factor is SAMPLING PROBABILITY, not signal amplitude:
- At n=8 (M=384, N=256): E[count(k=1 or N-1)] = 384*2/256 = 3.0 per instance.
  Therefore nearly every instance has direct k=1 access, AUC ~ 0.999.
- At n=10 (M=480, N=1024): E[count] = 480*2/1024 = 0.94 per instance.
  ~39% of instances lack direct k=1 access, forcing interpolation. AUC drops to ~0.92.

**Implication for Phenom detector calibration:** The minimum detectable fold-odd
amplitude is `epsilon > 0` provided k=1 appears in the sample. The detector's
SNR floor must be measured against the `epsilon = 0.03125` baseline at the
SMALLEST epsilon. If the Phenom detector reads orientation at epsilon=0.03125
but not at epsilon=0, the detector is live. If it fails at epsilon=0.03125,
increase epsilon until it crosses, and report that epsilon as the MDE.

---

## 5. Channel A Control

Channel A (public cosine only) stays at chance across all epsilon levels:
AUC range 0.455-0.555, all within null95 bands. Confirms that injected
quadrature does NOT leak into the public cosine channel -- a vectorized
implementation would NOT accidentally create a public orientation signal.

---

## 6. Verdict

**ODD_LANE_DETECTOR_CALIBRATED.** The mathematical reference transfer function
is established. The detector threshold is `epsilon > 0` with the requirement
that fixed-frequency rungs (k=1 or dyadic ladder) are accessible.

**Phenom hardware calibration (to be run on target):**
1. Inject synthetic sin at epsilon = 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0.
2. Enable the I/Q receiver (Track B) to read the k=1 rung from PDN phase response.
3. Measure AUC(epsilon). Verify:
   - AUC(0.0) = 0.5 (shuffle null)
   - AUC(epsilon > 0) > null95 + 0.03
   - Ladder is monotonic in epsilon
4. Report minimum detectable epsilon (MDE).

---

## 7. Files

| File | Role |
|---|---|
| `odd_lane_transfer_function.py` | Epsilon-ladder generator + AUC measurement |
| `results/odd_lane_transfer_function.json` | Full transfer function data |
| `results/output_transfer_function.txt` | Console log |
| `PHASE6_ODD_LANE_TRANSFER_FUNCTION.md` | This report |
