# Phase 5.9V to Phase 6 Basin Bridge

Verdict: `PHASE5_9V_DIRECTIONAL_BASIN_CONTROL__NOT_DETERMINISTIC_ENOUGH_FOR_MODE_C`

Latest result: `PHASE5_9V_TARGET_COUPLED_PUBLIC_SELECTOR_REJECTED`

## Purpose

Phase 5.9V is the physical carrier/basin feeder for Phase 6. The Phase 6 spec needs `precondition -> reproducible basin -> answer-predictive invariant`. Current 5.9V evidence reaches the first directional step, but not deterministic basin selection.

## Existing Evidence

- K10 P4 VID definition writes are live.
- P4 ladder/bracket reached decoded `1.1375V`.
- Restoration failures remained `0`.
- Carrier response amplified, collapsed, and switched basin under repeated VID+4/VID+5 bursts.
- VID+5 / decoded `1.1625V` basin selector:
  - `syscall_prelude`: 0 collapsed, 2 mid, 2 high
  - `cache_prelude`: 2 collapsed, 2 mid, 0 high
  - `branch_prelude`: 0 collapsed, 3 mid, 1 high

## Phase 6 Mapping

| Phase 6 Gate | 5.9V Contribution | Current State |
|---|---|---|
| G3 basin -> invariant | supplies physical basin label | directional only |
| G4 no-smuggle | must compare public-prelude vs d-oracle-prelude | not run |
| G5 controls | wrong/shuffled/prelude ladder | partial prelude ladder only |
| G6 scaling | basin stability across n | not run |

## Current Selector Read

| Selector | Read |
|---|---|
| `syscall_prelude` | avoids collapse in 4/4; splits high/mid |
| `cache_prelude` | avoids high in 4/4; useful negative/control selector |
| `branch_prelude` | mostly mid; avoids collapse |
| `quiet`, `reset_p0`, `collapse_prelude` | split basins; not selectors |

## Required Phase 6 Push

Run VID+5 with at least 10 repeats per selector, using measurement-core-only P4 VID definition writes by default:

- `syscall_prelude`
- `cache_prelude`
- `branch_prelude`
- `quiet`
- `public_kb_prelude`
- `shuffled_kb_prelude`
- `d_oracle_prelude`

Required output columns:

- `target_public_hash`
- `selector`
- `repeat`
- `vid_offset`
- `decoded_voltage`
- `restoration_failures`
- `boundary_thickness`
- `cycle_cv`
- `p99_p50`
- `basin_id`
- `fixed_point_d`
- `public_prelude_used_d` (`0` required)
- `oracle_control` (`1` only for d-oracle control)

## Acceptance For Mode C Handoff

Promote to Phase 6 Mode C only if a public-prelude selector reproducibly chooses an answer-predictive basin outside shuffled/null confidence intervals. If only `d_oracle_prelude` works, classify as smuggling and reject crossing.

## Attempt Log

`phase5_9/PHASE5_9V_PHASE6_REPRO_ATTEMPT.md` records the first Phase 6-facing reproducibility attempt. The target was reachable, `rdmsr`/`wrmsr` were present, and the matrix was launched with 10 repeats per selector, but the first runner attempted all-core P4 VID+5 setup. Every row failed the MSR-set gate and the target then stopped responding to SSH.

Runner hardening after the failed attempt:

- `DEF_CORES` now defaults to the measurement core only.
- P4 VID definition writes no longer target all cores by default.
- All-core VID+5 is classified as too aggressive for this feeder route unless console/power recovery is available.

Next exact action after power-cycle or SSH return:

```bash
scp session_scripts/phase5_9/run_phase5_9v_phase6_basin_repro.sh root@192.168.137.100:/root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh
ssh root@192.168.137.100 "chmod +x /root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh && cd /root/exp44_phase5_9 && DEF_CORES=3 REPEATS=10 ITERATIONS=30000 ./run_phase5_9v_phase6_basin_repro.sh"
```

## Completed Reproducibility Matrix

Artifact: `phase5_9/results/k10_voltage_probe/p4_vid5_phase6_basin_repro/PHASE5_9V_PHASE6_BASIN_REPRO.md`

The hardened measurement-core-only run completed:

- VID offset: `+5`
- Decoded voltage: `1.1625V`
- Rows analyzed: `70`
- Repeats per selector: `10`
- Restoration failures: `0`
- Core 3 P4 definition restored to stock `8000013540003440`
- Core 3 P-state restored to `0`

Selector summary:

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `branch_prelude` | 10 | 1 | 5 | 4 | mid | 0.500 | 0.900 | 0.600 |
| `cache_prelude` | 10 | 2 | 4 | 4 | high | 0.400 | 0.800 | 0.600 |
| `d_oracle_prelude` | 10 | 2 | 4 | 4 | mid | 0.400 | 0.800 | 0.600 |
| `public_kb_prelude` | 10 | 3 | 6 | 1 | mid | 0.600 | 0.700 | 0.900 |
| `quiet` | 10 | 3 | 6 | 1 | mid | 0.600 | 0.700 | 0.900 |
| `shuffled_kb_prelude` | 10 | 6 | 3 | 1 | collapsed | 0.600 | 0.400 | 0.900 |
| `syscall_prelude` | 10 | 2 | 1 | 7 | high | 0.700 | 0.800 | 0.300 |

Interpretation:

- Directional basin control reproduced.
- `syscall_prelude` is the strongest high-basin bias but remains below deterministic threshold.
- `public_kb_prelude` does not beat `quiet` on top-rate and does not separate from shuffled strongly enough.
- `d_oracle_prelude` does not dominate, so no smuggle-positive control appeared.
- This is still not sufficient for Phase 6 Mode C handoff.

## Additional Pushes

After the baseline 70-row matrix, follow-up pushes tested whether the weak public-prelude result could be rescued by coupling, longer conditioning, a different VID bracket, or direct Phase 6 target-coupled workload shaping.

### Public-Prelude Refinement

Artifact: `phase5_9/results/k10_voltage_probe/p4_vid5_phase6_public_refine/PHASE5_9V_PUBLIC_REFINE.md`

- VID offset: `+5`
- Rows analyzed: `90`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- `public_kb_syscall_prelude`: mid 4/10, collapsed 3/10, high 3/10.
- `shuffled_kb_syscall_prelude`: mid 6/10, collapsed 1/10, high 3/10.
- Result: public+syscall did not beat shuffled+syscall.

### Long Prelude

Artifact: `phase5_9/results/k10_voltage_probe/p4_vid5_phase6_long_prelude/PHASE5_9V_LONG_PRELUDE.md`

- VID offset: `+5`
- Prelude duration: `3.0s`
- Rows analyzed: `50`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- `public_kb_syscall_prelude`: collapsed 5/10, mid 4/10, high 1/10.
- `shuffled_kb_syscall_prelude`: collapsed 5/10, mid 3/10, high 2/10.
- Result: longer conditioning did not separate public from shuffled.

### VID Offset Comparison

Artifacts:

- `phase5_9/results/k10_voltage_probe/p4_vid4_phase6_offset_compare/PHASE5_9V_OFFSET_COMPARE.md`
- `phase5_9/results/k10_voltage_probe/p4_vid6_phase6_offset_compare/PHASE5_9V_OFFSET_COMPARE.md`
- `phase5_9/results/k10_voltage_probe/p4_vid6_phase6_public_candidate/PHASE5_9V_VID6_PUBLIC_CANDIDATE.md`

VID+4:

- Rows analyzed: `30`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- `public_kb_prelude`: mid 3/5, collapsed 2/5, high 0/5.
- `shuffled_kb_prelude`: collapsed 3/5, mid 2/5, high 0/5.
- Result: weak public-vs-shuffled direction, not enough sample size or determinism.

VID+6 exploratory:

- Rows analyzed: `30`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`
- `public_kb_syscall_prelude`: mid 4/5, collapsed 0/5, high 1/5.
- Result: possible public candidate, required confirmation.

VID+6 confirmation:

- Rows analyzed: `70`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`
- `public_kb_prelude`: mid 6/10, collapsed 3/10, high 1/10.
- `shuffled_kb_prelude`: mid 6/10, collapsed 1/10, high 3/10.
- `public_kb_syscall_prelude`: collapsed 6/10, mid 3/10, high 1/10.
- `d_oracle_syscall_prelude`: collapsed 8/10, mid 2/10, high 0/10.
- Result: the 4/5 public+syscall candidate did not confirm at 10 repeats. VID+6 gives nonpublic/voltage basin structure, not a Phase 6 public-prelude handoff.

Current pushed boundary:

`PHASE5_9V_PUBLIC_PRELUDE_NOT_DETERMINISTIC_AFTER_COUPLING_DURATION_VID_SWEEP_AND_TARGET_COUPLING`

### Target-Coupled Workload Push

Artifacts:

- `phase5_9/results/k10_voltage_probe/p4_vid5_phase6_target_coupled/PHASE5_9V_TARGET_COUPLED.md`
- `phase5_9/results/k10_voltage_probe/p4_vid6_phase6_target_coupled/PHASE5_9V_TARGET_COUPLED.md`

This push fixed the earlier coupling weakness: the `public_kb_*`, `shuffled_kb_*`, `wrong_kb_*`, and `d_oracle_*` selectors now derive a Phase 6-style public target payload and use it to shape both prelude dynamics and the measured workload. The run also added wrong-target controls.

VID+5 target-coupled:

- Rows analyzed: `40`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- `public_kb_prelude`: high 2/5, collapsed 2/5, mid 1/5, top rate 0.400.
- `public_kb_syscall_prelude`: mid 3/5, collapsed 1/5, high 1/5, top rate 0.600.
- `shuffled_kb_prelude`: collapsed 3/5, mid 2/5, high 0/5, top rate 0.600.
- `wrong_kb_syscall_prelude`: high 3/5, mid 2/5, collapsed 0/5, top rate 0.600.
- Result: public did not separate from shuffled or wrong-target controls.

VID+6 target-coupled:

- Rows analyzed: `40`
- Restoration failures: `0`
- Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`
- `public_kb_prelude`: high 3/5, collapsed 1/5, mid 1/5, top rate 0.600.
- `public_kb_syscall_prelude`: collapsed 3/5, mid 1/5, high 1/5, top rate 0.600.
- `shuffled_kb_prelude`: mid 4/5, collapsed 1/5, high 0/5, top rate 0.800.
- `wrong_kb_prelude`: mid 3/5, collapsed 2/5, high 0/5, top rate 0.600.
- Result: the strongest deterministic-looking selector was shuffled/nonpublic, not public. The VID+6 public hint is rejected under target-coupled workload shaping.

Current blocker:

`PUBLIC_TARGET_COUPLING_DOES_NOT_SELECT_PUBLIC_BASIN`
