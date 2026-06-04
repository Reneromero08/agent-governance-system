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
| `gpt_research/*.md` | `gpt_research/` | Research pathway | Undervolt pathway research and evidence inventory. | Yes | Markdown research reports are small and reproducible. |
| `gpt_research/UNDERVOLT_PATHWAY_1_BIOS_AGESA.md` | `gpt_research/` | Research pathway | BIOS/AGESA undervolt pathway evidence. | Yes | Supports rejected global AGESA patch route. |
| `cpu_hack/PATCH_ANALYSIS.md` | `cpu_hack/` | Firmware lab | Patch analysis for AGESA/global branch/P4-safe route. | Yes | Primary firmware decision evidence. |
| `cpu_hack/bios_dump.bin.report.txt` | `cpu_hack/` | Firmware lab | Text report for BIOS dump. | Yes | Text report is small evidence; binary remains ignored. |
| `cpu_hack/pstate_mask_hits.txt` | `cpu_hack/` | Firmware lab | P-state mask search hits. | Yes | Small text disassembly/search evidence. |
| `cpu_hack/pstate_targeted_disasm.txt` | `cpu_hack/` | Firmware lab | Targeted disassembly text. | Yes | Small text disassembly evidence. |
| `cpu_hack/check_ics.ps1` | `cpu_hack/` | Firmware/board lab | Local board/clock inspection helper. | Yes | Source script, not generated artifact. |
| `cpu_hack/check_ics2.ps1` | `cpu_hack/` | Firmware/board lab | Second local board/clock inspection helper. | Yes | Source script, not generated artifact. |
| `cpu_hack/bios_dump.bin` | `cpu_hack/` | Firmware lab raw artifact | Raw BIOS dump image. | No | Binary/heavy/sensitive artifact; preserve locally and describe through reports. |
| `cpu_hack/bios_dump.bin.guids.csv` | `cpu_hack/` | Firmware lab generated artifact | GUID export from BIOS dump tooling. | No | Generated CSV; can be regenerated from local dump. |
| `cpu_hack/bios_dump.bin.dump/` | `cpu_hack/` | Firmware lab generated tree | Extracted BIOS dump tree. | No | Large generated extraction tree. |
| `cpu_hack/_tmp_coreboot_*/` | `cpu_hack/` | Firmware lab generated tree | Temporary coreboot/source extraction trees. | No | Generated/heavy local research trees. |
| `cpu_hack/tools/` | `cpu_hack/` | Firmware lab local tools | Downloaded/local tools. | No | Local tool binaries/sources should not enter this checkpoint. |
| `cpu_hack/catcas_*.log` | `cpu_hack/` | Firmware lab logs | Local probe and extraction logs. | No | Generated logs; keep local unless summarized in markdown. |
| `session_scripts/*` | `session_scripts/` | Lab scripts | Python, C, and shell harnesses used during Exp44 sessions. | Yes | Source scripts/harnesses are reproducibility material. |

## Phase Layout Verdict

- Keep `cpu_sing_1/`, `cpu_sing_2/`, `gpt_research/`, `cpu_hack/`, and `session_scripts/`.
- Do not recreate generic `docs/`, `reports/`, `firmware/`, `voltage/`, `phase2/`, `scripts/`, `logs/`, `raw/`, or `archive/` folders for this checkpoint.
- Track markdown reports, source scripts, and small text evidence.
- Ignore raw BIOS images, extracted BIOS trees, generated logs, CSV exports, compiled binaries, and local tool trees.
