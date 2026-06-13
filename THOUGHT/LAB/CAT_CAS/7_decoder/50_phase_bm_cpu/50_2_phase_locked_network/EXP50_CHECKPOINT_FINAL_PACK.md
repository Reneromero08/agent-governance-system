# Exp50 Checkpoint Final Pack

Checkpoint date: 2026-06-04

## A. Executive Verdict

`EXP50_PHASE_LAYOUT_CHECKPOINT_READY`

Exp50 is organized around the phase folders that were already present in the lab:

- `50_1_subthreshold_msr/`
- `50_2_phase_locked_network/`
- `50_2_undervolt_research/`
- `50_2_firmware/cpu_hack/`
- per-phase `src/` subdirs (former session_scripts harnesses, now folded into each phase dir)

Phase 2 software routes are closed from the current evidence:

- `PHASE2_SOFTWARE_ROUTES_EXHAUSTED`
- `RUNTIME_VID_CLAMPED`
- `AGESA_GLOBAL_PATCH_REJECTED`
- `AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`
- `EXTERNAL_OBSERVABILITY_REQUIRED`
- `CATALYTIC_TAPE_WORKING_NON_KURAMOTO`

No BIOS flash, voltage write, board modification, or hardware-changing command was run.

## B. Roadmap Changes

Only `ROADMAP.md` is the active Exp50 roadmap for this checkpoint.

The checkpoint section records:

- Phase 0 foundation complete.
- Phase 1 runtime VID floor confirmed.
- Runtime lower-VID MSR path clamped.
- Passive TSC route dominated by a fixed ~2.67 MHz VRM/infrastructure artifact.
- Active phase route found no lock and no null separation.
- Coupling software channels exhausted.
- Detuning route not reproducible.
- GOE not observed.
- Ising not observed.
- AGESA global branch edit rejected after no-boot outcome and backup BIOS recovery.
- P4-safe AGESA route not byte-ready from current artifacts.
- External observability is the next boundary.
- Catalytic tape and `.holo` tape restoration work but are not Phase 2 Kuramoto success.

`ROADMAP_3.md` is not the Exp50 roadmap and should not contain this checkpoint.

## C. File Moves / Organization

No evidence files were moved into generic category folders.

The intended Exp50 layout is:

- `50_1_subthreshold_msr/`: Phase 1 / CPU_SING runtime VID, route reports, catalytic tape, `.holo`, VRM, detuning, GOE, and final pack.
- `50_2_phase_locked_network/`: Phase 2 Kuramoto, active phase, deep control, AGESA P4-safe, GOE, Ising, detuning, markers, and external measurement reports.
- `50_2_undervolt_research/`: undervolt pathway research and evidence inventory.
- `50_2_firmware/cpu_hack/`: firmware dump, text disassembly/search evidence, BIOS report, patch analysis, local generated trees, local tools, and logs.
- per-phase `src/` subdirs: scripts and harnesses used during Exp50 sessions (formerly under a single session_scripts dir, now folded into each phase dir).

The mistaken empty category folders are not part of the final layout and are removed only if they contain no real lab evidence.

## D. Gitignore Changes

`.gitignore` keeps generated/heavy/local artifacts out of git:

- BIOS/images: `*.bin`, `*.rom`, `*.fd`, `*.cap`
- raw/generated captures: `*.csv`, `*.dat`, `*.raw`, `*.wav`, `*.npy`, `*.npz`, `*.pcap`, `*.vcd`
- compiled binaries: `*.o`, `*.so`, `*.dll`, `*.exe`, `a.out`
- logs/temp/local files: `*.log`, `tmp/`, `temp/`, `*_tmp.*`, `*.env`, `*.local`
- Exp50 generated trees: `50_2_firmware/cpu_hack/bios_dump.bin.dump/`, `50_2_firmware/cpu_hack/_tmp_coreboot_*/`, `50_2_firmware/cpu_hack/tools/`

Markdown reports, source scripts, roadmap files, and small text evidence are not ignored.

## E. Files Safe To Commit

- `.gitignore`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/ROADMAP.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/REPORT.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/EXP50_FILE_INDEX.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/GIT_CEREMONY_EXP50_CHECKPOINT.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/EXP50_CHECKPOINT_FINAL_PACK.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_1_subthreshold_msr/*.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_phase_locked_network/*.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_undervolt_research/*.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/agesa_trace/PATCH_ANALYSIS.md`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/bios_parse/bios_dump.bin.report.txt`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/agesa_trace/pstate_mask_hits.txt`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/agesa_trace/pstate_targeted_disasm.txt`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/board_probe/check_ics*.ps1`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_*/src/*`

## F. Files Excluded From Git

- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/bios_dump.bin`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/bios_dump.bin.guids.csv`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/bios_dump.bin.dump/`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/_tmp_coreboot_*/`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/tools/`
- `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu/50_2_firmware/cpu_hack/catcas_*.log`

## G. Open Risks

- Raw firmware evidence remains local and untracked; reproducibility depends on text reports and local retained artifacts.
- per-phase `src/*` scripts should be human-reviewed before commit if any script should stay local.
- Firmware route remains at `AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`; do not treat current artifacts as flash-ready.
- External observability is still required before stronger Phase 2 voltage/phase claims.

## H. Exact Next Human Action

Review `GIT_CEREMONY_EXP50_CHECKPOINT.md`, run the listed `git add` commands if the scope is acceptable, inspect `git status`, and approve one commit only after the staged file list is correct.
