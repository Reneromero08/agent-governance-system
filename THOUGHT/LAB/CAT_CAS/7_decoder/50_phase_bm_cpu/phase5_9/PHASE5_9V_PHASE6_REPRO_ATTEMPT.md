# Phase 5.9V Phase 6 Reproducibility Attempt

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

## Objective

Run the Phase 6-facing 5.9V basin reproducibility matrix at P4 VID+5 with 10 repeats per selector:

- `quiet`
- `syscall_prelude`
- `cache_prelude`
- `branch_prelude`
- `public_kb_prelude`
- `shuffled_kb_prelude`
- `d_oracle_prelude`

## Execution Attempt

Target host was reachable before launch:

- SSH: `root@192.168.137.100`
- `rdmsr`: present
- `wrmsr`: present
- `/dev/cpu/0/msr`: present

Command launched from the lab folder:

```bash
ssh root@192.168.137.100 "chmod +x /root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh && cd /root/exp44_phase5_9 && REPEATS=10 ITERATIONS=30000 ./run_phase5_9v_phase6_basin_repro.sh"
```

The first runner version attempted P4 VID+5 setup across all cores before each row. Every row failed the MSR-set readback gate, and immediately after the attempt the target stopped responding to SSH:

```text
ssh: connect to host 192.168.137.100 port 22: Connection timed out
```

## Interpretation

This is not a completed basin reproducibility matrix. It is a failed all-core VID+5 setup attempt.

The failure is still useful: all-core P4 VID+5 is too aggressive for the reproducibility runner as written, or it leaves the host in a low-power/off-network state before the matrix can begin. Treat all-core P4 VID definition writes as unsafe for this Phase 6 feeder path.

## Hardening Applied

The local runner was corrected after the failure:

- `DEF_CORES` now defaults to the measurement core only (`MEAS_CORE`, default core 3).
- P4 VID definition writes now target `DEF_CORES`, not all cores.
- P-state reset still attempts to return all cores to P0 on exit.
- The next run should use the safer measurement-core-only default.

Updated runner:

`session_scripts/phase5_9/run_phase5_9v_phase6_basin_repro.sh`

## Next Exact Action

After the Phenom is power-cycled or SSH returns, copy the hardened runner and retry with measurement-core-only VID control:

```bash
scp session_scripts/phase5_9/run_phase5_9v_phase6_basin_repro.sh root@192.168.137.100:/root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh
ssh root@192.168.137.100 "chmod +x /root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh && cd /root/exp44_phase5_9 && DEF_CORES=3 REPEATS=10 ITERATIONS=30000 ./run_phase5_9v_phase6_basin_repro.sh"
```

Do not retry the all-core VID+5 variant unless there is an explicit recovery plan and console/power access.

## Successful Retry

After power-cycle / SSH recovery, the runner was patched to pass P4 definition values to `wrmsr` with explicit `0x` prefixes and to default VID definition writes to the measurement core only.

Smoke test:

- `DEF_CORES=3`
- `REPEATS=1`
- `ITERATIONS=10000`
- Rows: `7`
- Restoration failures: `0`
- Smoke verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`

Full retry:

```bash
ssh root@192.168.137.100 "rm -rf /root/exp44_k10_voltage_probe/p4_vid5_phase6_basin_repro && chmod +x /root/exp44_phase5_9/run_phase5_9v_phase6_basin_repro.sh && cd /root/exp44_phase5_9 && OUTPUT_DIR=/root/exp44_k10_voltage_probe/p4_vid5_phase6_basin_repro DEF_CORES=3 REPEATS=10 ITERATIONS=30000 ./run_phase5_9v_phase6_basin_repro.sh"
```

Completed result:

- Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`
- Rows analyzed: `70`
- Restoration failures: `0`
- VID offset: `+5`
- Decoded voltage: `1.1625V`
- Core 3 restored after run:
  - P4 definition: `8000013540003440`
  - P-state control: `0`

Final artifact:

`phase5_9/results/k10_voltage_probe/p4_vid5_phase6_basin_repro/PHASE5_9V_PHASE6_BASIN_REPRO.md`
