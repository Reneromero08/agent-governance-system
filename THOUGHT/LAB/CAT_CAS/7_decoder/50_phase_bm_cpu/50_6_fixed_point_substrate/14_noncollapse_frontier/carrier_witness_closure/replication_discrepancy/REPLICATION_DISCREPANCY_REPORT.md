# Phase 6B.5A Replication-Discrepancy Report

Status: `HISTORICAL_RESULT_NOT_REPRODUCED`

Official verdicts remain unchanged: historical T300 route `4:5` is 6/6; the
raw-reconstructable T48 campaign is `PARTIAL`, with route `4:5` at 1/6 and route
`2:3` at 2/6.

## Facts

- Campaign: `phase6b5_t48_d32b1bed_20260619`
- Source: `d32b1bed0deae1b907a07eeed018b924244e9ea2`
- Campaign manifest: `cbcd2a19d6dd3bc478244f77888aa87eb043003a7685caa17ff13fe4d47e6487`
- Historical and current analyzer SHA-256: `aa1ca5b0ce911ce931b036d023981eb1589d316e0adf88b66eb494ada1c5a50c`
- Analysis tool SHA-256: `edbd70277ad962bec19334d26a78ed997798eccc78a2352785c7ecc305feb4e5`
- Generated UTC: `2026-06-19T20:00:00Z`
- All 12 matrix schedules regenerate exactly from the C xorshift algorithm.
- Historical CSV codewords and first 48 trial labels match the regenerated
  current schedule for every route and seed.
- Historical raw timing samples and acquisition-binary SHA-256 are absent.

The material protocol differences are 300 versus 48 trials per family and
absence versus presence of raw timing capture. The raw-writer/restoration source
revision changed, but the official analyzer and summary semantics did not.

## Official Metrics

| Route | Seed | Pass | Real accuracy | RvP floor | Pseudo reject | Floor mode |
|---|---:|---:|---:|---:|---:|---|
| 2:3 | 0 | no | 0.958 | 0.920 | 1.000 | residual |
| 2:3 | 1 | yes | 1.000 | 1.000 | 1.000 | basis |
| 2:3 | 2 | no | 0.875 | 0.923 | 1.000 | residual |
| 2:3 | 3 | yes | 0.917 | 1.000 | 1.000 | basis |
| 2:3 | 4 | no | 0.958 | 0.875 | 1.000 | mini |
| 2:3 | 5 | no | 1.000 | 0.889 | 1.000 | rotation |
| 4:5 | 0 | no | 1.000 | 0.700 | 1.000 | mini |
| 4:5 | 1 | no | 1.000 | 0.818 | 1.000 | rotation |
| 4:5 | 2 | no | 1.000 | 0.875 | 1.000 | mini |
| 4:5 | 3 | yes | 1.000 | 1.000 | 1.000 | basis |
| 4:5 | 4 | no | 0.375 | 0.800 | 0.500 | basis |
| 4:5 | 5 | no | 1.000 | 0.867 | 1.000 | mini |

Per-mode real test denominators range from 3 to 9 rows. Pseudo predicted-mode
denominators are similarly sparse. Exact counts, false accepts/rejects,
Clopper-Pearson 95% intervals, and row flips to each gate are in
`results/official_gate_decomposition.json`.

## Diagnostic Metrics

All counterfactuals are `DIAGNOSTIC_ONLY__NOT_OFFICIAL_VERDICT`. Route `4:5`
non-phase pass counts range from 0/6 to 3/6 across deterministic grouping,
threshold, partition, and centroid variants. Reversing odd/even gives 2/6;
four deterministic trial-block holdouts give 2/6 or 3/6; grouping pseudo rows by
declared or actual mode remains 1/6. These results demonstrate finite-partition
fragility but do not recover historical 6/6 and do not replace the official gate.

## Raw Nonstationarity

All matrix runs have 1,776 windows, 1,504-1,520 samples per window, fixed
1.6 GHz frequency/P-state proxies, and temperatures below the 68 C veto. The
dominant repeatable association is tone frequency with I/Q magnitude/off-bin
floor, which is expected from the physical readout and is not a session-failure
diagnosis.

Route `4:5` seed 4 differs materially from peers: mean window magnitude is about
0.042 versus about 0.051-0.054, dispersion is several times larger, and magnitude
has strong rank association with TSC interval mean/std. Its early and late halves
are both degraded, and temperature, sample count, P-state, and capture timing do
not isolate a simple monotonic cause. Correlations are descriptive, not causal.

## Seed 4

Seed 4 is not a schedule imbalance: its schedule regenerates exactly and its
mode/family counts are within the finite T48 allocations. It shows compressed
centroid separations, a broad real-mode confusion matrix, depressed/noisy I/Q,
and simultaneous real, rho, and wrong-family degradation. Classification:
`UNRESOLVED_ANOMALY`, with temporary physical degradation or route/session drift
remaining plausible; ordinary marginal threshold variation is insufficient.

## Historical Comparison

Across the 12 matrix sessions, current-minus-historical mean shifts are -0.071
for real accuracy, -0.128 for real-mode floor, -0.064 for real-vs-pseudo floor,
and -0.040 for pseudo-reject floor. Phase delta is stable (+0.006 mean shift),
and wrong-declared match remains near zero. Thus a strong phase carrier persists
while the frozen classification/null closure gate is not reproduced.

## Inferences

The retained evidence rules out analyzer revision, codebook revision, and the
first-48 label schedule as explanations. It does not identify one unique cause:
trial count is materially different, sparse partitions are fragile, seed 4 is a
physical/session anomaly, and historical raw equivalence cannot be tested.

## Unknowns

- Exact historical raw timing behavior and telemetry associations.
- Historical acquisition binary identity.
- Whether seed 4 is repeatable on the same route in a new session.
- Whether a same-N independent replication retains the current 1/6 result.

## Final Adjudication

Primary classification: `HISTORICAL_RESULT_NOT_REPRODUCED`.

This is narrower than claiming a proven implementation mismatch, decoder-only
failure, or causal physical nonstationarity. Secondary findings are sparse-gate
fragility and an unresolved seed-4 route/session anomaly.

## Next Recommendation

`REPEAT_T48_SAME_PROTOCOL` before authorizing higher N. Freeze the existing
analyzer and seven gates, repeat both routes/seeds and silent/scramble controls,
budget approximately 3.6 hours and 605 MB, and retain the existing 68 C,
affinity, P-state, TSC, disk, raw-writer, and process stop conditions. A second
result near 1/6 supports reproducible current-protocol failure; a result near 6/6
supports session/route nonstationarity and requires a third preregistered
adjudication. A 300-trial campaign is not yet justified by this packet.
