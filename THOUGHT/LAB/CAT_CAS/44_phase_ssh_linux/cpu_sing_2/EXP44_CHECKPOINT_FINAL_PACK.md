# Exp44 Checkpoint Final Pack

Checkpoint date: 2026-06-04

## A. Executive Verdict

`EXP44_PHASE_LAYOUT_CHECKPOINT_READY`

Exp44 is organized around the phase folders that were already present in the lab:

- `cpu_sing_1/`
- `cpu_sing_2/`
- `gpt_research/`
- `cpu_hack/`
- `session_scripts/`

Phase 2 software routes are closed from the current evidence:

- `PHASE2_SOFTWARE_ROUTES_EXHAUSTED`
- `RUNTIME_VID_CLAMPED`
- `AGESA_GLOBAL_PATCH_REJECTED`
- `AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`
- `EXTERNAL_OBSERVABILITY_REQUIRED`
- `CATALYTIC_TAPE_WORKING_NON_KURAMOTO`

No BIOS flash, voltage write, board modification, or hardware-changing command was run.

## B. Roadmap Changes

Only `SSH_ROADMAP.md` is the active Exp44 roadmap for this checkpoint.

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

`ROADMAP_3.md` is not the Exp44 roadmap and should not contain this checkpoint.

## C. File Moves / Organization

No evidence files were moved into generic category folders.

The intended Exp44 layout is:

- `cpu_sing_1/`: Phase 1 / CPU_SING runtime VID, route reports, catalytic tape, `.holo`, VRM, detuning, GOE, and final pack.
- `cpu_sing_2/`: Phase 2 Kuramoto, active phase, deep control, AGESA P4-safe, GOE, Ising, detuning, markers, and external measurement reports.
- `gpt_research/`: undervolt pathway research and evidence inventory.
- `cpu_hack/`: firmware dump, text disassembly/search evidence, BIOS report, patch analysis, local generated trees, local tools, and logs.
- `session_scripts/`: scripts and harnesses used during Exp44 sessions.

The mistaken empty category folders are not part of the final layout and are removed only if they contain no real lab evidence.

## D. Gitignore Changes

`.gitignore` keeps generated/heavy/local artifacts out of git:

- BIOS/images: `*.bin`, `*.rom`, `*.fd`, `*.cap`
- raw/generated captures: `*.csv`, `*.dat`, `*.raw`, `*.wav`, `*.npy`, `*.npz`, `*.pcap`, `*.vcd`
- compiled binaries: `*.o`, `*.so`, `*.dll`, `*.exe`, `a.out`
- logs/temp/local files: `*.log`, `tmp/`, `temp/`, `*_tmp.*`, `*.env`, `*.local`
- Exp44 generated trees: `cpu_hack/bios_dump.bin.dump/`, `cpu_hack/_tmp_coreboot_*/`, `cpu_hack/tools/`

Markdown reports, source scripts, roadmap files, and small text evidence are not ignored.

## E. Files Safe To Commit

- `.gitignore`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/SSH_ROADMAP.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/REPORT.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/EXP44_FILE_INDEX.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/GIT_CEREMONY_EXP44_CHECKPOINT.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/EXP44_CHECKPOINT_FINAL_PACK.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_sing_1/*.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_sing_2/*.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/gpt_research/*.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/PATCH_ANALYSIS.md`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/bios_dump.bin.report.txt`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/pstate_mask_hits.txt`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/pstate_targeted_disasm.txt`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/check_ics*.ps1`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/session_scripts/*`

## F. Files Excluded From Git

- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/bios_dump.bin`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/bios_dump.bin.guids.csv`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/bios_dump.bin.dump/`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/_tmp_coreboot_*/`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/tools/`
- `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/cpu_hack/catcas_*.log`

## G. Open Risks

- Raw firmware evidence remains local and untracked; reproducibility depends on text reports and local retained artifacts.
- `session_scripts/*` should be human-reviewed before commit if any script should stay local.
- Firmware route remains at `AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`; do not treat current artifacts as flash-ready.
- External observability is still required before stronger Phase 2 voltage/phase claims.

## H. Exact Next Human Action

Review `GIT_CEREMONY_EXP44_CHECKPOINT.md`, run the listed `git add` commands if the scope is acceptable, inspect `git status`, and approve one commit only after the staged file list is correct.
