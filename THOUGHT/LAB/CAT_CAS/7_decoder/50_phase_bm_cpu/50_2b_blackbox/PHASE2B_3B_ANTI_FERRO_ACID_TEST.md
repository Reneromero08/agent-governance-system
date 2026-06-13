# Phase 2B.3B Anti-Ferromagnetic Acid Test

**Date:** 2026-06-05
**Binary:** `/tmp/topology_attractor`
**Source:** `50_2b_blackbox/src/topology_attractor.c`

## Command

```bash
gcc -O2 -o /tmp/topology_attractor /tmp/topology_attractor.c -lm && /tmp/topology_attractor
```

## Output

```
Anti-ferromagnetic chain J=-1, ground=-7
MODE       WORKERS       HITS     MEAN_E
  P1:2w shared                     0/200     7.00
  P1:1w null                       0/200     7.00
  P2:2w shared                   200/200    -7.00
  P2:1w null                     200/200    -7.00
```

## Interpretation

**P1 (topology-only local rule):** The worker flips to align when two connected spins differ. This is an implicit ferromagnetic bias — it always pushes toward alignment regardless of the underlying coupling sign. On a ferromagnetic chain (J=+1), this accidentally produces the ground state (200/200). On an anti-ferromagnetic chain (J=-1), it produces the WORST possible energy (+7.00, all spins aligned = all edges violated). P1 is FALSIFIED as passive attractor evidence.

**P2 (sign-aware edge rule):** The worker knows edge signs and flips to satisfy each edge. This correctly solves both ferro and anti-ferro problems (200/200 both). However, shared-substrate (2 workers) performs identically to single-worker (1 worker) — the shared hardware provides zero additional benefit. P2 is a valid active local constraint solver, not passive hidden-attractor evidence.

## Contamination Checklist

- [x] Workers never access J_ij matrix
- [x] Workers never compute global energy
- [x] Workers never compute local field sum
- [x] Workers never use Metropolis/Glauber/descent
- [x] P1 only uses: load two spins, compare, flip if different
- [x] P2 only uses: load two spins + sign, compare, flip if unsatisfied
- [x] Energy scored only after run completion

## Verdicts

```
PHASE2B_3B_P1_FERRO_BIAS_FALSIFIED
PHASE2B_3B_P2_ACTIVE_EDGE_SOLVER_WORKING
PHASE2B_3B_SHARED_SUBSTRATE_NO_ADVANTAGE
PHASE2B_3B_PASSIVE_NULLS_FAILED
PHASE2B_PASSIVE_NULLS_FAILED_CURRENT_MECHANISMS
```

## Next Boundary

Either (A) continue Phase 2B channel matrix with new passive mechanisms (route perturbation, contention injection, DID mismatch coupling), or (B) freeze current passive mechanisms as negative and promote P2 into Phase 3 active catalytic Ising solver. Not global `PHASE2B_NEGATIVE` — passive route still has untested channel mechanisms.
