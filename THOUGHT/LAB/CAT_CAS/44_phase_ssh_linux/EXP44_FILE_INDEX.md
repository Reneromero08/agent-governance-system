# Exp44 File Index

Checkpoint date: 2026-06-05

This index preserves the existing Exp44 phase layout. No files are reorganized into generic category folders.

| File / path | Location | Phase / lab role | Short purpose | Keep in git | Reason |
|---|---|---|---|---|---|
| `SSH_ROADMAP.md` | Exp44 root | Roadmap | Experiment 44 SSH/bare-metal roadmap and current checkpoint status. | Yes | Correct roadmap for this lab. |
| `REPORT.md` | Exp44 root | Report | Initial SSH/Linux bring-up report. | Yes | Human-readable experiment evidence. |
| `EXP44_FILE_INDEX.md` | Exp44 root | Checkpoint | This phase-based file index. | Yes | Reproducibility and commit scope. |
| `PHASE3_EXPANSION_SUMMARY.md` | Exp44 root | Phase 3 summary | Phase 3 expansion summary: subphases, verdicts, next task. | Yes | Closeout artifact for Phase 3 catalytic ladder. |
| `PHASE4_INTEGRATION_SUMMARY.md` | Exp44 root | Phase 4 summary | Phase 4 integration summary: Track A/B split, verdicts, next task. | Yes | Closeout artifact for Phase 4 dual-track architecture. |
| `cpu_sing_1/*.md` | `cpu_sing_1/` | Phase 1 / CPU_SING | Runtime VID clamp, route reports, catalytic tape, `.holo`, VRM, detuning, GOE, and final pack. | Yes | Markdown lab reports are primary evidence. |
| `cpu_sing_2/*.md` | `cpu_sing_2/` | Phase 2 | Kuramoto, active phase, deep control, AGESA P4-safe, GOE, Ising, detuning, markers, and external measurement reports. | Yes | Markdown lab reports are primary Phase 2 evidence. |
| `cpu_sing_3/*.md` | `cpu_sing_3/` | Phase 2/3 firmware | AGESA recursive gate, master dispatch, donor diff, arg0C provenance. | Yes | Records AGESA route blockers and exact next artifacts. |
| `gpt_research/*.md` | `gpt_research/` | Research pathway | Undervolt pathway research and evidence inventory. | Yes | Markdown research reports are small and reproducible. |
| `cpu_hack/PATCH_ANALYSIS.md` | `cpu_hack/` | Firmware lab | Patch analysis for AGESA/global branch/P4-safe route. | Yes | Primary firmware decision evidence. |
| `cpu_hack/*.txt` | `cpu_hack/` | Firmware lab text evidence | Decompile, disasm, xref, service table, MSR source proof artifacts. | Yes | Small text evidence for firmware route. |
| `cpu_hack/check_ics.ps1` | `cpu_hack/` | Firmware/board lab | Local board/clock inspection helpers. | Yes | Source scripts, not generated artifacts. |
| `cpu_hack/bios_dump.bin` | `cpu_hack/` | Firmware lab raw artifact | Raw BIOS dump image. | No | Binary/sensitive; preserved locally. |
| `cpu_hack/bios_dump.bin.*` | `cpu_hack/` | Firmware lab generated | GUID CSV, report, extracted tree. | No | Generated artifacts. |
| `cpu_hack/_tmp_coreboot_*/` | `cpu_hack/` | Firmware lab generated | Temporary coreboot/source extraction trees. | No | Generated/heavy local research trees. |
| `cpu_hack/tools/` | `cpu_hack/` | Firmware lab local tools | Downloaded/local tools. | No | Local tool binaries. |
| `cpu_hack/mod_donors/` | `cpu_hack/` | Firmware lab local | Downloaded stock/donor BIOS packages. | No | Binary/generated artifacts. |
| `cpu_hack/noop_replace/*.bin` | `cpu_hack/noop_replace/` | Firmware lab local | Rebuilt/attempted BIOS images. | No | Binary/generated artifacts. |
| `cpu_hack/catcas_*.log` | `cpu_hack/` | Firmware lab logs | Local probe and extraction logs. | No | Generated logs. |

## Session Scripts (organized by phase)

### Phase 1: MSR / Voltage / Frequency (`session_scripts/phase1_msr/`)

| File | Purpose | Keep |
|------|---------|------|
| `baseline.py` | Pre-test MSR status and cpufreq governor config | Yes |
| `subthreshold.py` | First sub-threshold write attempt (MSR 0xC0010062, I/O error) | Yes |
| `subthreshold2.py` | COFVID_CTL write attempt (hardware-overridden) | Yes |
| `cofvid_check.py` | Decoded COFVID_CTL reads for all 6 cores | Yes |
| `pstate_test.py` | P-state definition decode and transition prototype | Yes |
| `sub_p4.py` | Sub-threshold P4 definition write + Core 4 request | Yes |
| `freq_sweep.py` | DID frequency sweep (100MHz-200MHz breakthrough) | Yes |
| `vid_attacks.py` | P-state limit and NB voltage attack surfaces | Yes |
| `vid_runtime_test.py` | Runtime VID=0x20 test (SVI clamped) | Yes |
| `core_state.py` | Core frequency state checker | Yes |
| `msr_deep_probe.py` | Deep MSR probe: HWCR bits, NB P-state, COFVID vs P-state VID | Yes |
| `msr_p4_readonly_observer.py` | Read-only Linux target observer for P-state MSRs | Yes |
| `recovery_check.py` | Post-CMOS-clear P-state verification | Yes |
| `match_cores.py` | Set Cores 3+4 to matching 200MHz | Yes |
| `smbus_scan.sh` | SMBus/I2C bus scan for VRM controller | Yes |
| `nb_probe.sh` | Northbridge PCI config space probe | Yes |

### Phase 2: Kuramoto / Coupling / Oscillators (`session_scripts/phase2_kuramoto/`)

| File | Purpose | Keep |
|------|---------|------|
| `oscillator.c` | C oscillator payload (LCG tight loop) | Yes |
| `oscillator_test.py` | Python oscillator test harness | Yes |
| `oscillator_c.sh` | C oscillator compile and 2-oscillator network script | Yes |
| `tsc_sampler.c` | Raw rdtsc sampler (2M samples, Core-affine) | Yes |
| `kuramoto_test.py` | Kuramoto phase diagram sweep (DID 0-4) | Yes |
| `lock_oscillator.c` | LOCK CMPXCHG coupling test | Yes |
| `phase_oscillator.c` | Active phase measurement (mmap shared cache line) | Yes |
| `phase2_analyze.py` | Phase 2 coupling analysis | Yes |
| `phase2_analyze_fast.py` | Fast phase 2 analysis | Yes |
| `phase2_external_align.py` | Offline waveform/marker alignment for external measurement | Yes |
| `phase2_marker_harness.c` | Marker harness for coupling detection | Yes |
| `phase2_probe.c` | Phase 2 probe | Yes |

### Phase 3: Catalytic Computing Ladder (`session_scripts/phase3_catalytic/`)

| File | Purpose | Keep |
|------|---------|------|
| `operator_library.c` | 7 reversible operators, 28/28 tests | Yes |
| `meaningful_compute.c` | Reversible parity, hash fragment, FSM transition | Yes |
| `catalytic_sign.c` | Single sign, two-sign interference, metadata isolation | Yes |
| `oracle_paths.c` | 3-path oracle, tiebreak, 4-seed randomized | Yes |
| `baseline_compare.c` | Reversible vs destructive benchmarking | Yes |
| `catcas_phase3.h` | Public API header (40 lines, 17 functions) | Yes |
| `catcas_phase3.c` | Public API implementation (130 lines) | Yes |
| `catcas_phase3_cli.c` | CLI tool (test/xor/oracle modes) | Yes |
| `catalytic_phase.c` | First catalytic forward-reverse cycle (6 hardening gates) | Yes |
| `catalytic_batch.c` | 100-cycle stress test (100/100 SHA-256 matches) | Yes |
| `holo_phase.c` | First .holo encoding (4 slots, XOR output) | Yes |
| `holo_metadata.c` | .holo with metadata (9 slots, basis metadata survival) | Yes |
| `holo_tape_goal.c` | .holo tape goal test | Yes |
| `audit1.py` | Fresh Python catalytic cycle verification | Yes |

### Phase 4: .holo Eigenbasis on Catalytic Silicon (`session_scripts/phase4_holo/`)

| File | Purpose | Keep |
|------|---------|------|
| `phase4_bridge.c` | Phase 4.0 bridge gate (5/5 gates pass) | Yes |
| `eigenbasis_tape.c` | Phase 4.1A shared eigenbasis (4 tests, basis read-only) | Yes |
| `rotation_chain.c` | Phase 4.2A rotation chain (3 layers, forward/reverse) | Yes |

---

## Phase Layout Verdict

- Keep `cpu_sing_1/`, `cpu_sing_2/`, `cpu_sing_3/`, `gpt_research/`, `cpu_hack/`, and `session_scripts/`.
- Session scripts organized by phase: `phase1_msr/`, `phase2_kuramoto/`, `phase3_catalytic/`, `phase4_holo/`.
- Do not recreate generic `docs/`, `reports/`, `firmware/`, or `archive/` folders.
- Track markdown reports, source scripts, and small text evidence.
- Ignore raw BIOS images, extracted BIOS trees, generated logs, CSV exports, compiled binaries, and local tool trees.
- Phase 3 COMPLETE (12 subphases). Phase 4 Track A in progress (4.0, 4.1A, 4.2A complete; 4.3 next). Phase 4 Track B pending Phase 2.
