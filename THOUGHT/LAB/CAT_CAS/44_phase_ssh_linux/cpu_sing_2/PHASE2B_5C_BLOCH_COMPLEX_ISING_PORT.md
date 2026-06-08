# PHASE2B_5C_BLOCH_COMPLEX_ISING_PORT

## Verdict

`PHASE2B_5C_BLOCH_COMPLEX_ISING_ACTIVE_ORACLE_PASS`

The Exp07-style Bloch / complex-plane Ising port was implemented and run on the Phenom II target as a pure software phase-oracle harness. It beat four matched null families on mean energy across all five test problems.

This is active software phase-oracle evidence. It is not passive Kuramoto evidence, not physical phase lock, and not a claim that the CPU has sung.

## Command

```powershell
Get-Content -Raw session_scripts\phase2b\bloch_complex_ising.c | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "cat > /tmp/bloch_complex_ising.c && gcc -O2 /tmp/bloch_complex_ising.c -lm -o /tmp/bloch_complex_ising && timeout 40 /tmp/bloch_complex_ising"
```

## Safety

- Pure userspace C.
- No MSR access.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No Tier 3 physical instrumentation.

## Harness

Source:

```text
session_scripts/phase2b/bloch_complex_ising.c
```

Parameters:

| Parameter | Value |
|---|---:|
| paths | 24 |
| Bloch update steps | 48 |
| max spins | 16 |
| problem count | 5 |

Null hierarchy:

- random phase decode,
- random spin,
- sign-shuffled coupling matrix,
- edge-rewired coupling matrix.

Comparator:

- active sign-aware edge solver.

## Results

| Problem | Ground | Bloch best | Bloch mean | Ground hits | Null gate |
|---|---:|---:|---:|---:|---|
| `ferro_chain_n12` | `-11` | `-11` | `-9.500` | `9/24` | PASS |
| `anti_chain_n12` | `-11` | `-11` | `-9.417` | `9/24` | PASS |
| `frustrated_ring_n12` | `-10` | `-10` | `-10.000` | `24/24` | PASS |
| `random_sparse_n16` | `-18` | `-18` | `-17.167` | `16/24` | PASS |
| `planted_bipartite_n16` | `-120` | `-120` | `-120.000` | `24/24` | PASS |

Global result:

```text
Problems passing all four null means: 5/5
PHASE2B_5C_BLOCH_COMPLEX_ISING_ACTIVE_ORACLE_PASS
```

## Interpretation

The Bloch/complex-plane update is a valid active phase-oracle mechanism for the Phase 2B software branch:

- It keeps state in continuous phase angles rather than direct binary spin flips during the update.
- It decodes only at measurement.
- It beats random phase, random spin, sign-shuffled, and edge-rewired nulls on every tested problem.
- It outperforms the active edge solver on the frustrated ring and random sparse cases in this run.

This does not satisfy passive hidden-attractor criteria because the worker uses the problem coupling matrix during the phase update. It is therefore classified as active software phase-oracle progress, not passive substrate evidence.

## Route Impact

Phase 2B.5C advances from untested to:

`PHASE2B_5C_BLOCH_COMPLEX_ISING_ACTIVE_ORACLE_PASS`

The global Phase 2 goal remains active:

- not `CPU_SINGS`,
- not `BYTE_READY_HUMAN_REVIEW`,
- not `SOFTWARE_FIRMWARE_TRUE_WALL`,
- not `HUMAN_TOOL_REQUIRED_WITH_ALL_OTHER_ROUTES_EXHAUSTED`.

## Next Action

`PHASE2B_5D_SPECTRAL_PROBLEM_CLASSIFIER`

Port the spectral classifier branch to classify which problem families benefit from vertex phase oracle, Bloch/complex-plane oracle, or active edge solver before running the oracle. Acceptance requires deterministic problem features, held-out problem families, and null comparisons.
