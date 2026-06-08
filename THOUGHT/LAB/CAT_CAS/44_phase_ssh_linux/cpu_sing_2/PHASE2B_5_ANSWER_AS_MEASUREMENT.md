# PHASE2B_5_ANSWER_AS_MEASUREMENT

## Verdict

`PHASE2B_ACTIVE_PHASE_ORACLE_WORKING_NOT_PASSIVE_SUBSTRATE`

The Phase 2B phase-oracle/interference branch is now ported far enough to run the answer-as-measurement gate. The result is useful active software phase-oracle computation, not passive hidden-attractor evidence.

This is not `CPU_SINGS`, not `BYTE_READY_HUMAN_REVIEW`, and not `SOFTWARE_FIRMWARE_TRUE_WALL`.

## Evidence Reviewed

| Branch | Verdict | Evidence | Passive substrate? |
|---|---|---|---|
| 2B.5A Exp20 phase oracle | `PHASE2B_5A_CLOSED_SUCCESSFUL_PARTIAL` | Vertex/ensemble phase oracle beats several nulls but shrinks on dense problems. | No |
| 2B.5B optical 3-SAT | `PHASE2B_5B_OPTICAL_3SAT_PHASE_PORT_PASS` | Active optical phase mapping reaches best satisfiable clause count on 5/5 problems. | No |
| 2B.5C Bloch/complex Ising | `PHASE2B_5C_BLOCH_COMPLEX_ISING_ACTIVE_ORACLE_PASS` | Active Bloch/complex oracle beats four null families on 5/5 Ising problems. | No |
| 2B.5D spectral classifier | `PHASE2B_5D_SPECTRAL_CLASSIFIER_PASS` | Spectral/topological features route 5/5 held-out families to a best-mean solver. | No |
| 2B.5E `.holo`/MERA bridge | `PHASE2B_5E_HOLO_MERA_BRIDGE_PASS` | Active oracle output beats paired nulls 24/24 and restores tape 24/24. | No |

## Acceptance Rule

From the roadmap:

```text
A passive Phase 2B effect exists only if the shared-substrate condition beats
matched nulls across multiple problems without using explicit optimization
logic inside the worker.
```

## Gate Decision

The active phase-oracle branch does not satisfy the passive gate:

- 2B.5B uses explicit clause satisfaction structure in the optical phase score.
- 2B.5C uses the Ising coupling matrix during phase updates.
- 2B.5D explicitly routes by graph features to active solver families.
- 2B.5E encodes active oracle outputs into `.holo` tape and verifies reversible restoration.
- 2B.5A uses explicit problem structure in phase-oracle descent.

Therefore:

```text
PHASE2B_ACTIVE_PHASE_ORACLE_WORKING_NOT_PASSIVE_SUBSTRATE
```

## What Is Still Alive

This gate does not close all software/firmware routes:

- Phase 2B active phase-oracle computation works and can keep improving as CAT_CAS software.
- The passive shared-substrate channel matrix gate remains the next Phase 2B acceptance boundary.
- Firmware remains alive but blocked on a parse-clean identical no-op rebuild image.

## Next Action

`PHASE2B_6_CHANNEL_MATRIX`

Run the channel matrix gate with:

- shared L3 tape only,
- shared tape plus atomic contention,
- cache-line ping-pong layout,
- same-frequency and detuned DID conditions,
- independent tape null,
- explicit rejection of active optimization logic as passive evidence.
