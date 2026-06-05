# PHASE2_MASTER_CPU_SING_OR_TRUE_WALL

## Verdict

`HUMAN_APPROVAL_REQUIRED`

The CPU has not produced accepted Kuramoto/Ising/phase behavior yet, and this is not a true wall. Routes A-D advanced the boundary:

- Route A recovered the `.dG3_DXE` heap/table consumer and classified the `0xFFF7371A` source path.
- Route B narrowed the rebuild blocker to one exact missing local replacer.
- Route C found public donor workflows but no local donor image pair.
- Route D is ready for non-invasive external measurement, and the offline waveform/marker analyzer is prepared. The physical setup still requires human approval.

## Route Table

| Route | Status | Deepest progress | Blocker or next action |
|---|---|---|---|
| A: AGESA dispatcher chase | `DISPATCH_SOURCE_FOUND` | Consumer `0xFFF72B3C` walks `.dG3_DXE` heap handles; `0xFFF8D11E -> 0xFFF7371A` is a static function-pointer entry; `arg_0C` is runtime/heap-selected table context, not a proven static P4 row. | Firmware route alive, but not byte-ready. Need caller/dispatcher invocation proof before any P4-only claim. |
| B: no-op rebuild toolchain | `MISSING_REBUILD_TOOLCHAIN` | Local search found only `UEFIExtract`; no replace/save-image tool. | Add `cpu_hack/tools/uefitool_rebuild/UEFITool.exe` or equivalent and run no-op replacement. |
| C: public BIOS-mod donor route | `PUBLIC_MOD_DONOR_FOUND` | Public GA-970A-DS3P NVMe-mod donor workflow identified; no local donor-vs-stock image pair exists. | Acquire exact revision-matched donor and stock image for structural diff only. |
| D: external observability / Pi sidecar | `EXTERNAL_MEASUREMENT_READY` | Existing marker harness can coordinate Core3/Core4/Core5 state with external waveform capture; `phase2_external_align.py` can analyze the first waveform CSV against marker states and shuffled nulls. | Human sets up non-invasive scope/logic analyzer capture and saves raw waveform plus marker CSV. |

## Artifacts Produced

- `cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`
- `cpu_sing_3/PHASE2_MASTER_A_DISPATCH_SOURCE.md`
- `cpu_sing_3/PHASE2_MASTER_B_REBUILD_TOOLCHAIN.md`
- `cpu_sing_3/PHASE2_MASTER_C_BIOS_MOD_DONORS.md`
- `cpu_sing_3/PHASE2_MASTER_D_EXTERNAL_OBSERVABILITY.md`
- `cpu_sing_3/PHASE2_MASTER_CPU_SING_OR_TRUE_WALL.md`
- `session_scripts/phase2_external_align.py`

## Deepest Progress

The strongest new RE result is the `.dG3_DXE` source classification:

- `0xFFF8D0EC-0xFFF8D104` is a heap-handle list.
- `0xFFF72B3C` consumes that list and rebuilds/copies heap state.
- `0xFFF8D11E -> 0xFFF7371A` is in a nearby static function-pointer array.
- `0xFFF7371A` still consumes a selected runtime/caller-supplied structure via `arg_0C`.

That makes the current firmware route alive but not byte-ready.

## Exact Next Action

`EXTERNAL_MEASUREMENT_READY`

Run the non-invasive external capture:

```sh
gcc -O2 -pthread phase2_marker_harness.c -o phase2_marker_harness
./phase2_marker_harness 256 50000 > phase2_marker_log.csv
```

Capture waveform at the VRM/output rail during the same run, then analyze marker-aligned waveform changes against idle and shuffled nulls.

Offline analysis command after capture:

```sh
python3 phase2_external_align.py --marker phase2_marker_log.csv --wave scope_waveform.csv --segment-us 50000 --out-csv phase2_external_summary.csv --out-report phase2_external_alignment_report.md
```

## Human Approval Needed

Yes: `HUMAN_APPROVAL_REQUIRED` for physical measurement setup only.

Reason: the next action requires attaching scope/logic-analyzer probes to the owned board. It does not authorize BIOS flash, voltage writes, board mods, unknown PCI writes, or patch bytes.

## Do-Not-Do List

- Do not flash.
- Do not run flash commands.
- Do not write blind voltage values.
- Do not modify P0-P3.
- Do not repeat the rejected global AGESA branch edit.
- Do not claim the 2.67 MHz line as phase behavior unless it is marker-modulated and survives nulls.
- Do not treat public donor BIOS images as flash candidates.
- Do not count `UEFIExtract` as a rebuild tool.
