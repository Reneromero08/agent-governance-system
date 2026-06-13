# PHASE2B_6_CHANNEL_MATRIX

## Verdict

`PHASE2B_6_CHANNEL_MATRIX_REJECTED_BIASED`

The passive channel matrix was rerun on the Phenom II target. Shared two-worker channels can differ from single-worker nulls, but the effect is sign-biased and does not generalize across problem types. It is rejected as passive hidden-attractor evidence.

This is not `CPU_SINGS`, not `BYTE_READY_HUMAN_REVIEW`, and not `SOFTWARE_FIRMWARE_TRUE_WALL`.

## Command

```powershell
Get-Content -Raw 50_2b_blackbox\src\channel_matrix.c | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "cat > /tmp/channel_matrix.c && gcc -O2 /tmp/channel_matrix.c -lm -o /tmp/channel_matrix && timeout 80 /tmp/channel_matrix"
```

## Safety

- Pure userspace C.
- No MSR access.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No Tier 3 physical instrumentation.

## Channels Tested

| Channel | Mechanism |
|---|---|
| C1 | QR orthogonal subspace partition |
| C2 | Retrocausal 2-pass self-consistency |
| C3 | Warm-tape fingerprint contention |
| C4 | Detuned DID frequency coupling harness logic |

The harness states:

```text
CONTAMINATION: Workers NEVER access J_ij or compute energy.
```

## Result Summary

| Problem | Best shared result | Null comparison | Decision |
|---|---|---|---|
| Ferro | shared channels produce worse or equal mean energy than nulls | C1 `7.00` vs null `3.90`; C2 `7.00` vs null `2.08`; C4 equal | reject |
| Anti-ferro | shared channels strongly beat nulls | C1/C2/C4 hit `300/300`, C3 improves | biased candidate |
| Mixed | shared channels do not reach ground and are inconsistent | C1/C3 improve mean but no hits; C2 worse; C4 equal | reject |

## Interpretation

The anti-ferro success is not accepted as passive Phase 2B evidence because it does not survive cross-problem consistency:

- the same mechanism is actively bad for the ferro problem,
- the mixed-sign problem does not produce ground hits,
- C4 is saturated/equal in anti-ferro and mixed nulls,
- the workers do not know `J_ij`, so the anti-ferro improvement is best explained as a fixed update-rule bias, not problem-sensitive hidden substrate computation.

This closes the current passive channel matrix attempt.

## Route Impact

Phase 2B.6 advances to:

`PHASE2B_6_CHANNEL_MATRIX_REJECTED_BIASED`

The active phase-oracle branch remains useful CAT_CAS software, but passive shared-substrate evidence is still not met.

## Next Action

`PHASE2B_7_RESTORATION_GATE`

Run the restoration gate only as a classification step: verify whether any passive/shared-substrate attractor phase can be forward-applied and reversed/restored. If no passive channel candidate survives, classify the restoration gate as not applicable for passive evidence and keep active catalytic restoration under Phase 3/4 only.
