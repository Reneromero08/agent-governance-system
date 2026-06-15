# Track A -- Dual-Lane Even Cancellation (Staged)

**Status:** STAGED. Verdict `TRACK_A_STAGED_NOT_RUN`.
**Claim ceiling:** L3 (mathematical reference, no hardware execution).
**Route:** 4:5 (PROVISIONAL_ROUTE_4_5_PRIOR from T300, read from config file, not hardcoded).
**Hardware touched:** None.

---

## 0. One-Sentence Result

The mathematical reference model of the integer-multiply Hamming weight differential
predicts zero DC signal: `mean_diff ~ 0` for both public and same-candidate modes,
with hidden_positive also at noise floor. The Hamming weight difference between
`a*k_j` and `(N-a)*k_j` is symmetric under averaging -- per-step fluctuations exist
but cancel to zero mean. The Track A hardware experiment on Phenom may differ because
real PDN responds to dynamic transition density, not static Hamming weight.

---

## 1. Method

**Odd-lane source (named by Track Z):** Intermediate integer register values during
phase walk accumulation. `a * k_j` and `(N-a) * k_j` produce different Hamming weight
bit patterns in the integer multiplier/ALU.

**Model:** For each step j:
```
diff_j = alpha * (hamming_weight(a * k_j mod N) - hamming_weight((N-a) * k_j mod N))
PDN_differential = diff + N(0, sigma)
```

**Finding:** `mean(diff_j)` ~ 0 because Hamming weights of `a*k` and `(N-a)*k`
are identically distributed over random k. The per-step variance is nonzero but
the DC component cancels.

---

## 2. Route Configuration

| Property | Value |
|---|---|
| Selected route | 4:5 |
| Status | PROVISIONAL_ROUTE_4_5_PRIOR |
| Source | T300 PDN Slot2: 6/6 seeds PASS |
| Track I dependency | OPEN |
| Hardcoded? | NO -- read from config/route_selection.json |

---

## 3. Results (mathematical reference, alpha=0.05, sigma=0.005)

| Mode | n | mean_diff | std_diff | has_signal |
|---|---|---|---|---|
| public | 8 | -0.0006 | 0.0056 | NO |
| same_candidate | 8 | 0.0000 | 0.0003 | NO |
| hidden_positive | 8 | 0.0013 | 0.0277 | NO |
| public | 10 | 0.0433 | 0.8397 | NO |
| same_candidate | 10 | 0.0000 | 0.0002 | NO |
| hidden_positive | 10 | 0.1740 | 4.0294 | NO |

**Candidate separation AUC ~0.5 across all modes.** Orientation AUC ~0.5.

---

## 4. Why Zero DC Signal (Mathematical)

For random `k`, the product `a*k mod N` is uniformly distributed over [0, N).
The product `(N-a)*k mod N = (-a*k) mod N` is also uniformly distributed.
Therefore `E[hw(a*k)] = E[hw((N-a)*k)]` and `E[diff] = 0`.

The differential has zero mean because the Hamming weight distribution is
symmetric under `x -> N-x` for uniform random operands. Any per-step asymmetry
cancels in the average.

**Hardware note:** Real PDN measures dynamic transition power (bit-flips between
consecutive operations), which may have nonzero asymmetry due to the SEQUENCE
dependence of `a*k_1, a*k_2, a*k_3...` vs `(N-a)*k_1, (N-a)*k_2, (N-a)*k_3...`.
The Python static model cannot capture this without knowing the Phenom II
multiplier microarchitecture.

---

## 5. Controls Status

| Control | Status |
|---|---|
| same-candidate PDN null | STAGED (zero-diff injection) |
| equal-sign schedule | STAGED |
| no-sender baseline | STAGED |
| lane-swap control | STAGED |
| core-swap control | STAGED |
| schedule-shuffle control | STAGED |
| hidden positive control | STAGED (5x alpha injection) |
| shuffle-label null | STAGED |
| candidate blinding check | PASS (c0/c1 labels only) |
| route-config check | PASS (read from config, not hardcoded) |
| no hardcoded core check | PASS (core pair in config file) |

---

## 6. Verdict

**TRACK_A_STAGED_NOT_RUN.** The mathematical reference model is built, controls
are defined, route configuration is externalized, and candidate blinding is
enforced. Hardware execution requires the Phenom II with the Rust PDN probe
(`chiral_pdn_native.rs` adapted for dual-lane operation).

The static integer multiply Hamming weight model predicts zero mean PDN
differential. The hardware result may differ due to dynamic transition power
effects not captured by the static model.

---

## 7. Files

| File | Role |
|---|---|
| `chiral_dual_lane.py` | Mathematical reference: dual-lane PDN differential model |
| `config/route_selection.json` | Route config (4:5 provisional, not hardcoded) |
| `results/trackA_dual_lane_results.json` | Simulation results |
| `results/output_trackA.txt` | Console log |
| `PHASE6_TRACK_A_DUAL_LANE_EVEN_CANCELLATION.md` | This report |
