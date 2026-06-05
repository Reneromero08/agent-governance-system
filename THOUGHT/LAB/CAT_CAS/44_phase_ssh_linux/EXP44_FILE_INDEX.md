# Exp44 File Index

Checkpoint date: 2026-06-04

This index preserves the existing Exp44 phase layout. No files are reorganized into generic category folders.

| File / path | Location | Phase / lab role | Short purpose | Keep in git | Reason |
|---|---|---|---|---|---|
| `SSH_ROADMAP.md` | Exp44 root | Roadmap | Experiment 44 SSH/bare-metal roadmap and current checkpoint status. | Yes | Correct roadmap for this lab. |
| `REPORT.md` | Exp44 root | Report | Initial SSH/Linux bring-up report. | Yes | Human-readable experiment evidence. |
| `EXP44_FILE_INDEX.md` | Exp44 root | Checkpoint | This phase-based file index. | Yes | Reproducibility and commit scope. |
| `GIT_CEREMONY_EXP44_CHECKPOINT.md` | Exp44 root | Checkpoint | Prepared git checkpoint ceremony without committing. | Yes | Human staging/commit guide. |
| `EXP44_CHECKPOINT_FINAL_PACK.md` | Exp44 root | Checkpoint | Final checkpoint summary and next action. | Yes | Closeout artifact for Exp44 checkpoint. |
| `cpu_sing_1/*.md` | `cpu_sing_1/` | Phase 1 / CPU_SING | Runtime VID clamp, route reports, catalytic tape, `.holo`, VRM, detuning, GOE, and final pack. | Yes | Markdown lab reports are primary evidence. |
| `cpu_sing_1/RUNTIME_VID_DECIDER_PACK.md` | `cpu_sing_1/` | Phase 1 / CPU_SING | Runtime VID decision pack showing lower-VID path clamped. | Yes | Supports `RUNTIME_VID_CLAMPED`. |
| `cpu_sing_1/CPU_SING_GOAL_FINAL_PACK.md` | `cpu_sing_1/` | Phase 1 / CPU_SING | Final CPU_SING route pack. | Yes | Summarizes catalytic tape and route outcomes. |
| `cpu_sing_1/GOAL_ROUTE_7_HOLO.md` | `cpu_sing_1/` | Phase 1 / CPU_SING | `.holo` tape route evidence. | Yes | Supports `CATALYTIC_TAPE_WORKING_NON_KURAMOTO`. |
| `cpu_sing_2/*.md` | `cpu_sing_2/` | Phase 2 | Kuramoto, active phase, deep control, AGESA P4-safe, GOE, Ising, detuning, markers, and external measurement reports. | Yes | Markdown lab reports are primary Phase 2 evidence. |
| `cpu_sing_2/PHASE2_BASELINE.md` | `cpu_sing_2/` | Phase 2 | Passive baseline and TSC route evidence. | Yes | Supports passive route closure. |
| `cpu_sing_2/PHASE2_ACTIVE_PHASE.md` | `cpu_sing_2/` | Phase 2 | Active phase attempt results. | Yes | Supports no-lock / no-null-separation finding. |
| `cpu_sing_2/PHASE2_KURAMOTO_FINAL_PACK.md` | `cpu_sing_2/` | Phase 2 | Kuramoto final report. | Yes | Supports software route exhaustion verdict. |
| `cpu_sing_2/PHASE2_KURAMOTO_METRIC.md` | `cpu_sing_2/` | Phase 2 | Metric analysis for phase lock claims. | Yes | Supports no reliable Kuramoto observation. |
| `cpu_sing_2/PHASE2_COUPLING_CHANNELS.md` | `cpu_sing_2/` | Phase 2 | Software-visible coupling channel inventory. | Yes | Supports coupling channel exhaustion. |
| `cpu_sing_2/PHASE2_DETUNING.md` | `cpu_sing_2/` | Phase 2 | Frequency/DID detuning route. | Yes | Supports detuning not reproducible. |
| `cpu_sing_2/PHASE2_GOE.md` | `cpu_sing_2/` | Phase 2 | GOE route report. | Yes | Supports GOE not observed. |
| `cpu_sing_2/PHASE2_ISING_MAP.md` | `cpu_sing_2/` | Phase 2 | Ising route map. | Yes | Supports Ising not observed. |
| `cpu_sing_2/PHASE2_AGESA_P4_SAFE_FINAL_PACK.md` | `cpu_sing_2/` | Phase 2 firmware route | P4-safe AGESA analysis final pack. | Yes | Supports `AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`. |
| `cpu_sing_2/PHASE2_DEEP_3_EXTERNAL_MEASURE.md` | `cpu_sing_2/` | Phase 2 external boundary | External measurement boundary report. | Yes | Supports `EXTERNAL_OBSERVABILITY_REQUIRED`. |
| `cpu_sing_3/AGESA_GATE*.md` | `cpu_sing_3/` | Phase 3 AGESA recursive gate | CFG, table hunt, no-op workflow, injection, and patch-candidate gate artifacts. | Yes | Records recursive AGESA route blockers and exact next artifacts. |
| `cpu_sing_3/AGESA_NEXT_*.md` | `cpu_sing_3/` | Phase 3 AGESA next gate | Constructor decompile/xrefs, table reopen, no-op rebuild, actionability, and final pack. | Yes | Records the next-gate progress: function `0xFFF7371A`, `.dG3_DXE` pointer `0xFFF8D11E`, and missing rebuild tool. |
| `cpu_sing_3/AGESA_RECURSIVE_FINAL_PACK.md` | `cpu_sing_3/` | Phase 3 AGESA recursive gate | Recursive AGESA final pack. | Yes | Preserves route-alive / missing-artifact verdict. |
| `cpu_sing_3/PHASE2_MASTER_A_DISPATCH_SOURCE.md` | `cpu_sing_3/` | Phase 2 master route A | Dispatch/source route report for `.dG3_DXE` consumer and `arg_0C` classification. | Yes | Records `DISPATCH_SOURCE_FOUND` without claiming byte-ready P4 edit target. |
| `cpu_sing_3/PHASE2_MASTER_B_REBUILD_TOOLCHAIN.md` | `cpu_sing_3/` | Phase 2 master route B | Rebuild toolchain search and exact missing replacer checklist. | Yes | Records why `UEFIExtract` is insufficient and what exact tool is needed. |
| `cpu_sing_3/PHASE2_MASTER_C_BIOS_MOD_DONORS.md` | `cpu_sing_3/` | Phase 2 master route C | Public BIOS-mod donor workflow classification. | Yes | Records donor workflow leads without treating donor images as flash candidates. |
| `cpu_sing_3/PHASE2_MASTER_D_EXTERNAL_OBSERVABILITY.md` | `cpu_sing_3/` | Phase 2 master route D | Non-invasive external measurement and marker-correlation plan. | Yes | Records `EXTERNAL_MEASUREMENT_READY`. |
| `cpu_sing_3/PHASE2_MASTER_CPU_SING_OR_TRUE_WALL.md` | `cpu_sing_3/` | Phase 2 master final | CPU sings or true wall master checkpoint after routes A-D. | Yes | Final route table and exact next action. |
| `gpt_research/*.md` | `gpt_research/` | Research pathway | Undervolt pathway research and evidence inventory. | Yes | Markdown research reports are small and reproducible. |
| `gpt_research/UNDERVOLT_PATHWAY_1_BIOS_AGESA.md` | `gpt_research/` | Research pathway | BIOS/AGESA undervolt pathway evidence. | Yes | Supports rejected global AGESA patch route. |
| `cpu_hack/PATCH_ANALYSIS.md` | `cpu_hack/` | Firmware lab | Patch analysis for AGESA/global branch/P4-safe route. | Yes | Primary firmware decision evidence. |
| `cpu_hack/bios_dump.bin.report.txt` | `cpu_hack/` | Firmware lab | Text report for BIOS dump. | Yes | Text report is small evidence; binary remains ignored. |
| `cpu_hack/pstate_mask_hits.txt` | `cpu_hack/` | Firmware lab | P-state mask search hits. | Yes | Small text disassembly/search evidence. |
| `cpu_hack/pstate_targeted_disasm.txt` | `cpu_hack/` | Firmware lab | Targeted disassembly text. | Yes | Small text disassembly evidence. |
| `cpu_hack/check_ics.ps1` | `cpu_hack/` | Firmware/board lab | Local board/clock inspection helper. | Yes | Source script, not generated artifact. |
| `cpu_hack/check_ics2.ps1` | `cpu_hack/` | Firmware/board lab | Second local board/clock inspection helper. | Yes | Source script, not generated artifact. |
| `cpu_hack/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt` | `cpu_hack/` | Firmware lab text evidence | Annotated local decompile/pseudocode for the function containing `0xFFF737A3`. | Yes | Small text evidence proving function entry and constructor base flow. |
| `cpu_hack/AmdProcessorInitPeim_fff737a3_xrefs.txt` | `cpu_hack/` | Firmware lab text evidence | Xref export for the constructor function and dispatch pointer. | Yes | Small text evidence proving no direct call xrefs and `.dG3_DXE` pointer reference. |
| `cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt` | `cpu_hack/` | Firmware lab text evidence | Annotated local decompile/pseudocode for the `.dG3_DXE` heap/table consumer. | Yes | Small text evidence proving the table consumer and `arg_0C` source classification. |
| `cpu_hack/AmdProcessorInitPeim_helper_fff4cf55_disasm.txt` | `cpu_hack/` | Firmware lab text evidence | Disassembly of selector helper `0xFFF4CF55`, producer window `0xFFF4CF9C`, and helper call sites. | Yes | Proves `arg_0C` is a variable-length runtime-produced record list. |
| `cpu_hack/AmdProcessorInitPeim_pointer_search_fff4cf9c.txt` | `cpu_hack/` | Firmware lab text evidence | Raw pointer search for `0xFFF4CF9C`, `0xFFF4CF55`, and `0xFFF7371A`. | Yes | Proves the producer pointer is in callback descriptor setup code, not a producer table. |
| `cpu_hack/AmdProcessorInitPeim_callback_fff4aadd_disasm.txt` | `cpu_hack/` | Firmware lab text evidence | Disassembly around callback descriptor setup and first `0xFFF4AADD` inspection. | Yes | Records the callback registration path for the producer. |
| `cpu_hack/AmdProcessorInitPeim_fff4aadd_wide_disasm.txt` | `cpu_hack/` | Firmware lab text evidence | Wider disassembly of descriptor interpreter `0xFFF4AA00` / `0xFFF4AADD` path. | Yes | Shows typed descriptor iteration and handler dispatch. |
| `cpu_hack/AmdProcessorInitPeim_descriptor_handlers_disasm.txt` | `cpu_hack/` | Firmware lab text evidence | Disassembly of `0xFFF4AADD` descriptor runner and typed handlers `0xFFF4A175`, `0xFFF4A34A`, `0xFFF4A540`, and `0xFFF4A676`. | Yes | Records the next service/handler layer behind the runtime record producer. |
| `cpu_hack/AmdProcessorInitPeim_descriptor_callsite_xrefs.txt` | `cpu_hack/` | Firmware lab text evidence | Xrefs for `0xFFF4D12F`, `0xFFF4CF9C`, and `0xFFF4AADD` callsites. | Yes | Proves `[ebp-8]` before `0xFFF4D1AB` is `arg_0C + 8` and records all direct descriptor-runner callsites. |
| `cpu_hack/AmdProcessorInitPeim_outer_producer_table_xrefs.txt` | `cpu_hack/` | Firmware lab text evidence | Raw pointer and section map for outer producer slot `0xFFF7F516 -> 0xFFF4D12F`. | Yes | Proves the producer is registered through `.data`, not as a static P-state row. |
| `cpu_hack/AmdProcessorInitPeim_service_table_trace.txt` | `cpu_hack/` | Firmware lab text evidence | Focused PE32 disassembly trace of slot consumer, outer producer, descriptor runner, and typed handler fanout. | Yes | Advances the route to service/function-table provenance without producing patch bytes. |
| `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt` | `cpu_hack/noop_replace/` | Firmware no-op rebuild gate | Documents acquired UEFIReplace/UEFITool 0.28.0 and no-op attempts. | Yes | Small text evidence; records why no parse-clean no-op rebuilt image exists yet. |
| `cpu_sing_3/PHASE2_FW_ARG0C_PROVENANCE.md` | `cpu_sing_3/` | Phase 2 firmware route | Route 2 provenance report for `arg_0C`, helper `0xFFF4CF55`, and producer window `0xFFF4CF9C`. | Yes | Advances firmware route without claiming byte-ready P4 candidate. |
| `cpu_sing_3/PHASE2_DONOR_DIFF_REPORT.md` | `cpu_sing_3/` | Phase 2 donor workflow | Official F2j stock vs public NVMe donor structural diff. | Yes | Records rebuild/checksum workflow lessons without treating donor as flash candidate. |
| `cpu_hack/noop_replace/*.bin` | `cpu_hack/noop_replace/` | Firmware no-op rebuild generated artifacts | Rebuilt/attempted BIOS images and generated PE32 section blobs. | No | Binary/generated artifacts; summarize through `NOOP_DIFF_SUMMARY.txt`. |
| `cpu_hack/mod_donors/*` | `cpu_hack/mod_donors/` | Firmware donor workflow generated/local artifacts | Downloaded stock/donor BIOS packages, extracted images, and parse reports. | No for binaries; yes only if summarized as markdown | Binary/generated artifacts remain local; `PHASE2_DONOR_DIFF_REPORT.md` is the commit-safe summary. |
| `cpu_hack/bios_dump.bin` | `cpu_hack/` | Firmware lab raw artifact | Raw BIOS dump image. | No | Binary/heavy/sensitive artifact; preserve locally and describe through reports. |
| `cpu_hack/bios_dump.bin.guids.csv` | `cpu_hack/` | Firmware lab generated artifact | GUID export from BIOS dump tooling. | No | Generated CSV; can be regenerated from local dump. |
| `cpu_hack/bios_dump.bin.dump/` | `cpu_hack/` | Firmware lab generated tree | Extracted BIOS dump tree. | No | Large generated extraction tree. |
| `cpu_hack/_tmp_coreboot_*/` | `cpu_hack/` | Firmware lab generated tree | Temporary coreboot/source extraction trees. | No | Generated/heavy local research trees. |
| `cpu_hack/tools/` | `cpu_hack/` | Firmware lab local tools | Downloaded/local tools. | No | Local tool binaries/sources should not enter this checkpoint. |
| `cpu_hack/catcas_*.log` | `cpu_hack/` | Firmware lab logs | Local probe and extraction logs. | No | Generated logs; keep local unless summarized in markdown. |
| `session_scripts/*` | `session_scripts/` | Lab scripts | Python, C, and shell harnesses used during Exp44 sessions. | Yes | Source scripts/harnesses are reproducibility material. |
| `session_scripts/phase2_external_align.py` | `session_scripts/` | Phase 2 external observability | Offline waveform/marker alignment analyzer for scope or logic-analyzer CSV captures. | Yes | Reproducibility material for the external measurement route. |

## Phase Layout Verdict

- Keep `cpu_sing_1/`, `cpu_sing_2/`, `gpt_research/`, `cpu_hack/`, and `session_scripts/`.
- Do not recreate generic `docs/`, `reports/`, `firmware/`, `voltage/`, `phase2/`, `scripts/`, `logs/`, `raw/`, or `archive/` folders for this checkpoint.
- Track markdown reports, source scripts, and small text evidence.
- Ignore raw BIOS images, extracted BIOS trees, generated logs, CSV exports, compiled binaries, and local tool trees.
