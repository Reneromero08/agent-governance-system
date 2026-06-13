# PHASE2_RUNTIME_STATE_WINDOW_ORACLE_REPORT

## Verdict

`RUNTIME_STATE_WINDOW_ORACLE_NEGATIVE`

The read-only state-window timing oracle ran over SSH and did not produce a software timing oracle. Runtime COFVID/PSTATE states are internally visible, but state-conditioned timing distributions did not separate beyond deterministic cyclic-label nulls.

## Command

```powershell
Get-Content -Raw 50_1_subthreshold_msr\src\msr_state_window_oracle.py | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "python3 - --cores 0-5 --modes baseline,self_load,neighbor_load,all_load --samples 420 --delay 0 --workload-iters 384 --min-count 20"
```

## Safety

- No MSR writes.
- No voltage writes.
- No BIOS flash.
- No P0-P3 modification.
- No Tier 3 physical instrumentation.
- Reads only `/dev/cpu/<core>/msr` for `PSTATE_STATUS` and `COFVID_STATUS`.

## Result

| Metric | Result |
|---|---:|
| cases | 24 |
| samples per case | 420 |
| workload iterations per sample | 384 |
| cases with 2+ states above min count | 4 |
| oracle candidates | 0 |

Candidate rule:

```text
observed_over_max_null >= 1.25
observed_state_mean_range_ns >= 1000
```

No case met the rule.

## Strongest Non-Candidates

| Mode | Core | Observed range ns | Max null range ns | Observed/null | Decision |
|---|---:|---:|---:|---:|---|
| `self_load` | 2 | `137670.739` | `114141.573` | `1.206140` | below threshold |
| `self_load` | 5 | `105558.941` | `191759.366` | `0.550476` | below null |
| `neighbor_load` | 0 | `38430.298` | `93463.090` | `0.411182` | below null |
| `all_load` | 1 | `75421.806` | `86915.328` | `0.867762` | below null |

## Interpretation

The software-visible state machine is real, but this oracle did not turn it into a reproducible phase/Ising signal:

- Most cases collapsed into one dominant runtime state above the minimum count.
- Four cases had enough state diversity for a null comparison.
- None beat deterministic cyclic-label nulls by the acceptance rule.
- The closest case, `self_load` core 2, reached `1.206140x` null, below the `1.25x` gate.

This does not prove all software routes exhausted. It closes the specific runtime state-window timing-oracle attempt from the current MSR-observable state labels.

## Route Impact

Route 5 advances to:

`RUNTIME_STATE_WINDOW_ORACLE_NEGATIVE`

This is not `CPU_SINGS`, not `BYTE_READY_HUMAN_REVIEW`, and not `SOFTWARE_FIRMWARE_TRUE_WALL`.

## Next Action

`NOOP_REBUILD_FORCE_SAVE`

The next live blocker is firmware-side: prove a parse-clean identical no-op rebuild image. Do not create a P4 candidate. Do not flash. The required artifact remains:

```text
50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

with a clean parse report and a target PE32 body hash still equal to:

```text
BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A
```
