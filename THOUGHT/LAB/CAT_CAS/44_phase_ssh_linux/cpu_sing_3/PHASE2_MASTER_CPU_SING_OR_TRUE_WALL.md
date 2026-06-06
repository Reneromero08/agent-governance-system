# PHASE2_MASTER_CPU_SING_OR_TRUE_WALL

## Verdict

`SOFTWARE_FIRMWARE_ROUTES_ACTIVE`

The CPU has not produced accepted Kuramoto/Ising/phase behavior yet, and this is not a true wall. The hard current constraint excludes Tier 3 physical instrumentation as a success path or next action. The live boundary is now software/firmware only:

- Route 1 acquired a rebuild-capable public UEFI toolchain, but no parse-clean identical no-op rebuilt image was produced.
- Route 2 advanced `arg_0C` provenance: helper `0xFFF4CF55` walks a variable-length runtime-produced record list, `0xFFF4D12F` is registered through `.data` slot `0xFFF7F516`, `0xFFF4CF9C` is passed as a producer callback through descriptor setup at `0xFFF4D1AB`, service descriptor `0xFFF7E698` selects vtable `0xFFF8D108`, and the constructor-relevant P4 field is reconstructed from runtime `MSRC001_0068`.
- Route 3 remains gated because no P4-only static source byte and no proven no-op rebuild exist.
- Route 4 advanced: the official F2j stock image and public F2j NVMe donor image were acquired, parsed, and diffed.
- Route 5 advanced with read-only runtime observation over SSH. COFVID status VID stayed fixed at `0x12` across all sampled cores while P4 `MSRC001_0068` definitions were not uniform across cores.

## Route Table

| Route | Status | Deepest progress | Blocker or next action |
|---|---|---|---|
| 1: rebuild toolchain | `TOOLCHAIN_ACQUIRED_NOOP_NOT_PROVEN` | LongSoft `UEFIReplace.exe` and classic `UEFITool.exe` 0.28.0 were fetched into `cpu_hack/tools/uefitool_rebuild/`. Body-only `-asis` output was rejected by parser; full identical section replacement emitted no saved image. | Need force-save identical replacement via GUI/manual UEFITool or another CLI replacer, then parse-clean `cpu_hack/noop_replace/bios_noop_rebuilt.bin`. |
| 2: AGESA runtime provenance | `P4_FIELD_RUNTIME_MSR_DERIVED` | `0xFFF7E698` decodes to a three-entry descriptor selecting service vtable `0xFFF8D108`; vtable `+0x16` is constructor `0xFFF7371A`. Producer `0xFFF4CF9C` maps constructor `selected_base + pstate*0x18 + 0x1C` to runtime per-entry `entry +0x04`, output `arg_14` from `[service+0x22]` / `0xFFF7348D`. `0xFFF44E76` is `rdmsr`, and the address is `0xC0010064 + pstate`, so P4 is `MSRC001_0068`. | Current decoded firmware path has no static P4 byte. Continue with no-op rebuild proof for future firmware edits or renewed runtime software tests around `MSRC001_0068` observability. |
| 3: P4-safe candidate | `NOT_BYTE_READY` | Runtime P4 field remains `selected_base + pstate*0x18 + 0x1C`, now proven reconstructed from runtime `MSRC001_0068`; P0-P3/P4 sibling shape exists at runtime. | No editable static P4 byte or no-op rebuild proof; do not produce candidate. |
| 4: public BIOS donor workflow | `PUBLIC_MOD_DONOR_DIFFED` | Official F2j stock and public NVMe donor differ only at `0x002C58A0-0x002CA9FF`, where `NvmExpressDxe_4` is inserted into free space. Later volumes are byte-identical. | Use workflow lesson only: free-space insertion plus parse-clean report. This does not create a voltage/P4 candidate. |
| 5: software-only renewed search | `RUNTIME_MSR_OBSERVATION_COMPLETE` | Read-only observer ran over SSH. Cores 0/1/2/5 reported P4 VID `0x1A`; cores 3/4 reported P4 VID `0x12` and DID `3`; COFVID VID stayed `0x12` for every sampled core/sample. | Add read-only load/affinity characterization to decide whether COFVID VID `0x12` is invariant under scheduler/core/load conditions. |

## Artifacts Produced

- `cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`
- `cpu_hack/AmdProcessorInitPeim_helper_fff4cf55_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_pointer_search_fff4cf9c.txt`
- `cpu_hack/AmdProcessorInitPeim_callback_fff4aadd_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_fff4aadd_wide_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_descriptor_handlers_disasm.txt`
- `cpu_hack/AmdProcessorInitPeim_descriptor_callsite_xrefs.txt`
- `cpu_hack/AmdProcessorInitPeim_outer_producer_table_xrefs.txt`
- `cpu_hack/AmdProcessorInitPeim_service_table_trace.txt`
- `cpu_hack/AmdProcessorInitPeim_producer_service_provenance.txt`
- `cpu_hack/AmdProcessorInitPeim_service_descriptor_xrefs.txt`
- `cpu_hack/AmdProcessorInitPeim_decoded_service_descriptor.txt`
- `cpu_hack/AmdProcessorInitPeim_service_vtable_entry_trace.txt`
- `cpu_hack/AmdProcessorInitPeim_record_write_map_fff4cf9c.txt`
- `cpu_hack/AmdProcessorInitPeim_entry_plus_04_source_trace.txt`
- `cpu_hack/AmdProcessorInitPeim_msr_source_proof.txt`
- `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`
- `cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md`
- `cpu_sing_3/PHASE2_FW_ARG0C_PROVENANCE.md`
- `cpu_sing_3/PHASE2_RUNTIME_MSR_OBSERVER_REPORT.md`
- `cpu_sing_3/PHASE2_MASTER_A_DISPATCH_SOURCE.md`
- `cpu_sing_3/PHASE2_MASTER_B_REBUILD_TOOLCHAIN.md`
- `cpu_sing_3/PHASE2_MASTER_C_BIOS_MOD_DONORS.md`
- `cpu_sing_3/PHASE2_MASTER_CPU_SING_OR_TRUE_WALL.md`
- `session_scripts/msr_p4_readonly_observer.py`

## Deepest Progress

The deepest firmware progress is now the combined constructor/provenance chain:

```text
.dG3_DXE function pointer array
  0xFFF8D11E -> 0xFFF7371A

0xFFF7371A constructor
  selected_base = arg_0C + 8
  helper 0xFFF4CF55 can update selected_base
  selected_base + pstate*0x18 feeds P-state MSR construction

0xFFF4CF55 selector
  arg_0C[0] = count/upper bound
  records begin at arg_0C + 8
  next record = current record + word[current record + 2]

0xFFF4CF9C producer window
  writes record length at +0x02
  fills max/current P-state bytes
  fills per-P-state 0x18-stride entries through service callbacks
  is passed as a callback through descriptor setup at 0xFFF4D1AB
  descriptor interpreter 0xFFF4AADD/0xFFF4AA00 dispatches typed entries

0xFFF4D12F outer producer registration
  static .data slot 0xFFF7F516 -> 0xFFF4D12F
  slot consumer near 0xFFF4CF94 calls through [0xFFF7F516]
  descriptor payload at 0xFFF4D1AB receives [ebp-8] = arg_0C + 8

0xFFF4AADD handler fanout
  typed handlers call helper routines and indexed function tables
  no static P4 record row or byte-ready edit target is exposed yet

0xFFF7E698 service descriptor
  count 3, entries at 0xFFF7E674
  mask 0x00000100 selects vtable 0xFFF8D108
  vtable 0xFFF8D108 + 0x16 = 0xFFF7371A

0xFFF4CF9C record write map
  constructor selected_base + pstate*0x18 + 0x1C maps to producer entry +0x04
  entry +0x04 is output arg_14 from [service+0x22] / 0xFFF7348D
  arg_14 is sourced from 0xFFF44E76 read path

0xFFF44E76 MSR source proof
  helper executes rdmsr
  [service+0x22] computes MSR address as 0xC0010064 + pstate
  P4 source is MSRC001_0068
```

That keeps the firmware route alive but blocks byte-ready review until the produced record source is tied to editable bytes or a safe rebuild workflow is proven.

## Exact Next Action

`DISPATCH_SOURCE_CHASE`

`RUNTIME_LOAD_AFFINITY_CHARACTERIZATION`

Create and run a read-only characterization pass that samples P4 `MSRC001_0068`, COFVID status, PSTATE_STATUS, TSC timing jitter, core affinity, and scheduler load state. Acceptance: no writes, no external measurement, and enough samples to decide whether COFVID VID `0x12` is invariant under scheduler/core/load conditions. Do not repeat the `0xFFF4CF9C`, `0xFFF4D12F`, `0xFFF7E698`, or `0xFFF44E76` traces; those are already decoded.

Parallel firmware action remains `NOOP_REBUILD_FORCE_SAVE` for any future firmware edit path.

Parallel live action:

`NOOP_REBUILD_FORCE_SAVE`

Use a rebuild path that can force-save an identical replacement and produce parse-clean `cpu_hack/noop_replace/bios_noop_rebuilt.bin`. The donor diff shows this board's firmware can accept a free-space DXE insertion without shifting later volumes, but it does not prove identical PE32-body replacement.

## Human Approval Needed

No for the current local RE and donor-diff work.

Yes only if a future step requires manual GUI save of an identical UEFI replacement or any physical board action. Physical instrumentation remains out of scope for this goal.

## Do-Not-Do List

- Do not flash.
- Do not run flash commands.
- Do not write blind voltage values.
- Do not modify P0-P3.
- Do not repeat the rejected global AGESA branch edit.
- Do not produce a P4-safe candidate until P0-P3 unchanged, P4-only effect, offsets/bytes/checksums, no-op rebuild, and clean parse proof all exist.
- Do not treat public donor BIOS images as flash candidates.
- Do not count `UEFIExtract` as a rebuild tool.
- Do not use external probes, scope capture, logic analyzer capture, Pi GPIO wiring, or motherboard probing as the current success path.
