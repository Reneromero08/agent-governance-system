# PHASE2_MASTER_CPU_SING_OR_TRUE_WALL

## Verdict

`SOFTWARE_FIRMWARE_ROUTES_ACTIVE`

The CPU has not produced accepted Kuramoto/Ising/phase behavior yet, and this is not a true wall. The hard current constraint excludes Tier 3 physical instrumentation as a success path or next action. The live boundary is now software/firmware only:

- Route 1 acquired and built a rebuild-capable public UEFI toolchain variant; a parse-clean identical no-op rebuilt image was produced, parsed, and verified byte-identical to stock.
- Route 2 advanced `arg_0C` provenance: helper `0xFFF4CF55` walks a variable-length runtime-produced record list, `0xFFF4D12F` is registered through `.data` slot `0xFFF7F516`, `0xFFF4CF9C` is passed as a producer callback through descriptor setup at `0xFFF4D1AB`, service descriptor `0xFFF7E698` selects vtable `0xFFF8D108`, and the constructor-relevant P4 field is reconstructed from runtime `MSRC001_0068`.
- Route 3 remains gated because no P4-only static source byte or edit target exists, even though no-op rebuild/save is now proven. A new helper-layer probe chased the next service/function-table layer behind `0xFFF7348D` / `0xFFF44E76` and found no editable static P4-only byte source.
- Route 4 advanced: the official F2j stock image and public F2j NVMe donor image were acquired, parsed, and diffed.
- Route 5 advanced with read-only runtime load/affinity characterization, transition/jitter characterization, state-window oracle testing, P4-asymmetry oracle testing, cacheline phase-coupling rejection, Phase 2B.5B optical 3-SAT, Phase 2B.5C Bloch/complex-plane Ising, Phase 2B.5D spectral classifier, Phase 2B.5E `.holo`/MERA bridge, the Phase 2B answer-as-measurement gate, Phase 2B.6 channel matrix, Phase 2B.7 restoration gate, Phase 2B.8 decision tree, Phase 3B four-snapshot catalytic invariant probing, Phase 4.3 residual-channel compression, Phase 4.4A operator/eigenvalue validation, Phase 4.5 `.holo` mini-model decode/restore, Phase 4.6 public harness packaging, and Phase 5.10D cache/address topology probing. Runtime MSR/state-window/cacheline/P4-asymmetry/state-label/scheduler tests did not produce accepted Kuramoto, Ising, phase lock, or CPU-sings evidence. Active phase-oracle software works, but passive substrate evidence remains rejected; active software explains the successful Phase 2B results. Phase 3B confirms an answer-predictive reversible relational carrier in a catalytic harness, Phase 4.3 compresses it into `.holo` residual tags, Phase 4.4A produces GOE-like operator-matrix spacing against nulls, Phase 4.5 decodes a readable mini-model while restoring tape, and Phase 4.6 packages Track A. Phase 5.10D found a reproducible cache/address topology timing witness, but it is classified as `VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL`; it does not open Phase 6 and is not physical Kuramoto.

## Route Table

| Route | Status | Deepest progress | Blocker or next action |
|---|---|---|---|
| 1: rebuild toolchain | `NOOP_REBUILD_PROVEN` | Public LongSoft `old_engine` source was fetched into ignored tool tree, a temporary Qt/qmake force-save UEFIReplace variant was built on the Linux target, and identical AmdProcessorInitPeim PE32 body replacement produced `cpu_hack/noop_replace/bios_noop_rebuilt.bin`. The rebuilt image parses cleanly, is byte-identical to stock, and preserves target body SHA-256 `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`. | Do not repeat no-op rebuild. Use this proof only if a future P4-only static edit target is found. |
| 2: AGESA runtime provenance | `P4_FIELD_RUNTIME_MSR_DERIVED_HELPER_LAYER_CLOSED` | `0xFFF7E698` decodes to a three-entry descriptor selecting service vtable `0xFFF8D108`; vtable `+0x16` is constructor `0xFFF7371A`. Producer `0xFFF4CF9C` maps constructor `selected_base + pstate*0x18 + 0x1C` to runtime per-entry `entry +0x04`, output `arg_14` from `[service+0x22]` / `0xFFF7348D`. `0xFFF44E76` is `rdmsr`, and the address is `0xC0010064 + pstate`, so P4 is `MSRC001_0068`. The follow-up helper-layer probe found 5 direct `0xC0010068` hits, all already-known MSR loop/read paths, and no static handler pointers or P0-P4 sibling bytes. | Current decoded firmware path has no static P4 byte. Leave this chain; search only separate P4-affecting sources or renew runtime software tests around `MSRC001_0068` observability. |
| 3: P4-safe candidate | `NOT_BYTE_READY` | Runtime P4 field remains `selected_base + pstate*0x18 + 0x1C`, now proven reconstructed from runtime `MSRC001_0068`; P0-P3/P4 sibling shape exists at runtime only. No-op rebuild proof exists. Helper-layer probe did not expose editable static bytes. Family 10h source confirms the same shape: P-state values are gathered from live `PS_REG_BASE + k` MSRs into runtime `PSTATE_LEVELING` buffers, then optionally used for leveling writes. | No editable static P4 byte or P4-only edit target; do not produce candidate. |
| 4: public BIOS donor workflow | `PUBLIC_MOD_DONOR_DIFFED` | Official F2j stock and public NVMe donor differ only at `0x002C58A0-0x002CA9FF`, where `NvmExpressDxe_4` is inserted into free space. Later volumes are byte-identical. | Use workflow lesson only: free-space insertion plus parse-clean report. This does not create a voltage/P4 candidate. |
| 5: software-only renewed search | `VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL` | Runtime state-window oracle was negative. P4-asymmetry oracle found the old per-core `MSRC001_0068` split had vanished after restart: all six cores now report `0x8000013540003440`, P4 VID `0x1A`, and COFVID VID `0x12`; no two-group runtime label remained to use. A follow-up read-only effective-state selector map found that ordinary load selectors move FID/DID/PSTATE labels across four states while VID remains fixed at `0x12`. The state-label phase-coupling probe produced a timing candidate in sparse windows, but dense narrowing, shuffled-answer hard-null sweeps, and modal validation collapsed the state-label route. Scheduler/topology phase-offset probing produced sparse candidates but did not reproduce across fresh seed windows or core pairs. Firmware separate-source search found CpuDxe/CpuPei/LegacyRegion P-state sibling constants, but raw context classifies them as MSR address initializers, not P4 value rows. Direct P4 value-pattern search found zero hits. Family 10h source provenance confirms the runtime MSR gather and runtime buffer write path, not a ROM value table. Phase 5.10D found a reproducible same-address cache/address timing channel across two core layouts, with median effects `36.9428405` and `32.3227925`, permutation p-value `0.0004997501`, and family sign agreement `1.0`. | Do not open Phase 6 from scalar timing witnesses. 5.10D is complete as a capped witness below the encoding wall; it is not CPU-sings evidence and not a fixed-point crossing. |

## Artifacts Produced

- `cpu_hack/agesa_trace/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_helper_fff4cf55_disasm.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_pointer_search_fff4cf9c.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_callback_fff4aadd_disasm.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff4aadd_wide_disasm.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_descriptor_handlers_disasm.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_descriptor_callsite_xrefs.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_outer_producer_table_xrefs.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_service_table_trace.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_producer_service_provenance.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_service_descriptor_xrefs.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_decoded_service_descriptor.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_service_vtable_entry_trace.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_record_write_map_fff4cf9c.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_entry_plus_04_source_trace.txt`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_msr_source_proof.txt`
- `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`
- `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
- `cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt`
- `cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md`
- `cpu_sing_3/PHASE2_FW_ARG0C_PROVENANCE.md`
- `cpu_sing_3/PHASE2_P4_EDIT_SOURCE_PROOF.md`
- `cpu_hack/agesa_trace/AmdProcessorInitPeim_p4_edit_source_probe.txt`
- `cpu_sing_3/PHASE2_RUNTIME_MSR_OBSERVER_REPORT.md`
- `cpu_sing_3/PHASE2_RUNTIME_LOAD_AFFINITY_REPORT.md`
- `cpu_sing_3/PHASE2_RUNTIME_TRANSITION_JITTER_REPORT.md`
- `cpu_sing_3/PHASE2_RUNTIME_STATE_WINDOW_ORACLE_REPORT.md`
- `cpu_sing_3/PHASE2_RUNTIME_P4_ASYMMETRY_ORACLE_REPORT.md`
- `cpu_sing_3/PHASE2_RUNTIME_P4_ASYMMETRY_ORACLE.json`
- `cpu_sing_3/PHASE2_EFFECTIVE_STATE_SELECTOR_MAP_REPORT.md`
- `cpu_sing_3/PHASE2_EFFECTIVE_STATE_SELECTOR_MAP.json`
- `cpu_sing_3/PHASE2_STATE_LABEL_PHASE_COUPLING_REPORT.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_BALANCED_SEED1000.json`
- `cpu_sing_3/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_BALANCED_SEED5000.json`
- `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP.csv`
- `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_NARROWING_SWEEP.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_HARDNULL_SWEEP.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_HARDNULL_FOCUS_SWEEP.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_TIMING_EDGE_HARDENED_REPORT.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_MODAL_FEATURE_SEARCH.md`
- `cpu_sing_3/PHASE2_STATE_LABEL_MODAL_VALIDATION.md`
- `cpu_sing_3/PHASE2_SCHEDULER_TOPOLOGY_RESONANCE_HARDENED.md`
- `cpu_sing_3/PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_SEARCH.md`
- `cpu_sing_3/PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_DEEPENED.md`
- `cpu_sing_3/PHASE2_FIRMWARE_PSTATE_VALUE_PATTERN_SEARCH.md`
- `cpu_sing_3/PHASE2_FIRMWARE_SOURCE_PROVENANCE_WALL_AUDIT.md`
- `cpu_sing_3/PHASE2_CPU_DXE_VALUE_CONSUMER_TRACE.md`
- `cpu_sing_3/PHASE2_CPU_PEI_VALUE_CONSUMER_TRACE.md`
- `cpu_sing_3/PHASE2_LEGACY_REGION_VALUE_CONSUMER_TRACE.md`
- `cpu_sing_3/PHASE2_F10_SOURCE_PSTATE_VALUE_PROVENANCE.md`
- `phase5_10/PHASE5_10D_CACHE_ADDRESS_TOPOLOGY_LIVE.md`
- `phase5_10/PHASE5_10D_CACHE_SET_ADDRESS_TOPOLOGY_PREP.md`
- `phase5_10/results/live_5_10d_cache_addr/phase5_10d_cache_address_topology_report.json`
- `phase5_10/results/live_5_10d_cache_addr_swap/phase5_10d_cache_address_topology_report.json`
- `phase5_10/phase5_10_cache_address_topology.c`
- `phase5_10/analyze_phase5_10d_cache_address.py`
- `cpu_sing_2/PHASE2_CACHELINE_PHASE_COUPLING.md`
- `cpu_sing_2/PHASE2B_5B_OPTICAL_3SAT_PORT.md`
- `cpu_sing_2/PHASE2B_5C_BLOCH_COMPLEX_ISING_PORT.md`
- `cpu_sing_2/PHASE2B_5D_SPECTRAL_PROBLEM_CLASSIFIER.md`
- `cpu_sing_2/PHASE2B_5E_HOLO_MERA_BRIDGE.md`
- `cpu_sing_2/PHASE2B_5_ANSWER_AS_MEASUREMENT.md`
- `cpu_sing_2/PHASE2B_6_CHANNEL_MATRIX.md`
- `cpu_sing_2/PHASE2B_7_RESTORATION_GATE.md`
- `cpu_sing_2/PHASE2B_8_DECISION_TREE.md`
- `phase3b/PHASE3B_CATALYTIC_SUBSTRATE_PRIMITIVE.md`
- `session_scripts/phase3b/catalytic_invariant_probe.c`
- `phase3b/results/invariant_probe_summary.csv`
- `phase4_holo/PHASE4_3_RESIDUAL_CHANNEL.md`
- `session_scripts/phase4_holo/residual_channel.c`
- `phase4_holo/PHASE4_4A_OPERATOR_GOE.md`
- `session_scripts/phase4_holo/operator_goe.c`
- `phase4_holo/PHASE4_5_HOLO_MINI_MODEL.md`
- `session_scripts/phase4_holo/holo_mini_model.c`
- `phase4_holo/PHASE4_6_PUBLIC_HOLO_HARNESS.md`
- `session_scripts/phase4_holo/catcas_holo_harness.c`
- `cpu_sing_3/PHASE2_MASTER_A_DISPATCH_SOURCE.md`
- `cpu_sing_3/PHASE2_MASTER_B_REBUILD_TOOLCHAIN.md`
- `cpu_sing_3/PHASE2_MASTER_C_BIOS_MOD_DONORS.md`
- `cpu_sing_3/PHASE2_MASTER_CPU_SING_OR_TRUE_WALL.md`
- `session_scripts/phase1_msr/msr_p4_readonly_observer.py`
- `session_scripts/phase1_msr/msr_load_affinity_characterizer.py`
- `session_scripts/phase1_msr/msr_transition_jitter_probe.py`
- `session_scripts/phase1_msr/msr_state_window_oracle.py`
- `session_scripts/phase1_msr/msr_effective_state_selector_map.py`
- `session_scripts/phase1_msr/msr_state_label_phase_coupling_probe.py`
- `session_scripts/phase1_msr/summarize_state_label_sweep.py`
- `session_scripts/phase1_msr/analyze_state_label_modal_features.py`
- `session_scripts/phase2_kuramoto/scheduler_topology_resonance.c`
- `session_scripts/phase2_kuramoto/analyze_scheduler_topology_resonance.py`
- `session_scripts/phase2_firmware/find_p4_sources_across_bios.py`
- `session_scripts/phase2_kuramoto/cacheline_phase_coupling.c`
- `session_scripts/phase2b/optical_3sat_phase_port.c`
- `session_scripts/phase2b/bloch_complex_ising.c`
- `session_scripts/phase2b/spectral_problem_classifier.c`
- `session_scripts/phase2b/holo_mera_bridge.c`

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

Family 10h source provenance
  PStateGatherMain gathers P-state data from live PS_REG_BASE + k MSRs
  P4 resolves to PS_REG_BASE + 4 / MSRC001_0068
  PSTATE_LEVELING buffers carry CoreFreq, Power, IddValue, IddDiv, enable
  F10PstateLevelingCoreMsrModify can write from runtime buffers
  no ROM-resident P4-only value row is exposed
```

That keeps the firmware route alive but blocks byte-ready review until a separate P4-affecting source is tied to editable bytes. The decoded `0xFFF4CF9C -> 0xFFF7348D -> 0xFFF44E76` chain is now closed as runtime-MSR-derived for byte-ready purposes. The safe no-op rebuild workflow is proven.

## Exact Next Action

`SOFTWARE_FIRMWARE_WALL_REVIEW_AFTER_5_10D_SCALAR_WITNESS`

Phase 2B has been classified: active phase-oracle software works, but passive substrate evidence is rejected because active software explains the successful results. Phase 3B confirmed a catalytic relational invariant in a controlled four-snapshot harness, Phase 4.3 compressed that carrier into `.holo` residual tags, Phase 4.4A produced GOE-like operator-matrix statistics against nulls, Phase 4.5 decoded a readable `.holo` mini-model while restoring tape, and Phase 4.6 packaged the public harness. The no-op rebuild artifact is now proven. Phase 5.6 polytope/positive-geometry work and later scalar witness work show software-readable carrier structure, but not accepted CPU-sings evidence. Do not repeat `0xFFF4CF9C`, `0xFFF4D12F`, `0xFFF7E698`, or `0xFFF44E76` traces; those are already decoded.

Parallel firmware action no longer repeats the decoded helper chain or raw address-table traces. Runtime observability around `MSRC001_0068` advanced: the old per-core P4 definition asymmetry is not stable after restart, and current COFVID VID is clamped at `0x12` across the P4-asymmetry matrix. The read-only effective-state selector map then found movable FID/DID/PSTATE labels under ordinary load selectors while VID stayed fixed. The elapsed-threshold timing route collapsed under dense narrowing, shuffled-answer nulls, and higher-row focus reruns; modal feature validation also failed to reproduce across three fresh seed starts. Scheduler/topology phase-offset probing produced sparse candidates but did not reproduce. Firmware separate-source search found P-state MSR address tables outside AmdProcessorInitPeim, but not P4 value rows; direct P4 value-pattern search was also negative. CpuDxe, CpuPei, and LegacyRegion raw consumer traces found no P4 value consumer. Family 10h source provenance now confirms the P4 value path is runtime MSR gather into `PSTATE_LEVELING`, then optional runtime leveling write. Phase 5.10D is complete as a capped scalar/cache-address witness below the encoding wall. Good pause point: next resume action is `SOFTWARE_FIRMWARE_WALL_REVIEW_AFTER_5_10D_SCALAR_WITNESS`.

The donor diff shows this board's firmware can accept a free-space DXE insertion without shifting later volumes, and the no-op rebuild proves identical PE32 body rebuild/save mechanics. Neither proves a voltage/P4 candidate.

## Human Approval Needed

No for the current local RE, donor-diff, and no-op rebuild proof work.

Yes only before any future firmware candidate construction or any action outside the current no-flash/no-voltage/no-hardware boundary. Physical instrumentation remains out of scope for this goal.

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
