# Exp50 File Index

Checkpoint date: 2026-06-05

This index preserves the existing Exp50 phase layout. No files are reorganized into generic category folders.

| File / path | Location | Phase / lab role | Short purpose | Keep in git | Reason |
|---|---|---|---|---|---|
| `ROADMAP.md` | Exp50 root | Roadmap | experiment 50 SSH/bare-metal roadmap and current checkpoint status. | Yes | Correct roadmap for this lab. |
| `REPORT.md` | Exp50 root | Report | Initial SSH/Linux bring-up report. | Yes | Human-readable experiment evidence. |
| `EXP50_FILE_INDEX.md` | Exp50 root | Checkpoint | This phase-based file index. | Yes | Reproducibility and commit scope. |
| `50_3_catalytic_ladder/PHASE3_EXPANSION_SUMMARY.md` | `50_3_catalytic_ladder/` | Phase 3 summary | Phase 3 expansion summary: subphases, verdicts, next task. | Yes | Closeout artifact for Phase 3 catalytic ladder. |
| `50_3b_substrate_primitive/PHASE3B_CATALYTIC_SUBSTRATE_PRIMITIVE.md` | `50_3b_substrate_primitive/` | Phase 3B | Four-snapshot invariant probe verdict and null-control summary. | Yes | Records the catalytic substrate primitive decision gate. |
| `50_3b_substrate_primitive/results/invariant_probe_summary.csv` | `50_3b_substrate_primitive/results/` | Phase 3B result | Target-run metric summary for four-snapshot invariant probe. | Yes | Small required result artifact for the primitive gate. |
| `50_4_holo_eigenbasis/PHASE4_INTEGRATION_SUMMARY.md` | `50_4_holo_eigenbasis/` | Phase 4 summary | Phase 4 integration summary: Track A/B split, verdicts, next task. | Yes | Closeout artifact for Phase 4 dual-track architecture. |
| `50_4_holo_eigenbasis/PHASE4_3_RESIDUAL_CHANNEL.md` | `50_4_holo_eigenbasis/` | Phase 4 Track A | Residual-channel target-run report and controls. | Yes | Records Phase 4.3 completion. |
| `50_4_holo_eigenbasis/PHASE4_4A_OPERATOR_GOE.md` | `50_4_holo_eigenbasis/` | Phase 4 Track A | Operator-matrix eigenvalue spacing report and null comparison. | Yes | Records Phase 4.4A completion. |
| `50_4_holo_eigenbasis/PHASE4_5_HOLO_MINI_MODEL.md` | `50_4_holo_eigenbasis/` | Phase 4 Track A | `.holo` mini-model target-run report and controls. | Yes | Records Phase 4.5 completion. |
| `50_4_holo_eigenbasis/PHASE4_6_PUBLIC_HOLO_HARNESS.md` | `50_4_holo_eigenbasis/` | Phase 4 Track A | Public `.holo` harness packaging report. | Yes | Records Phase 4.6 completion. |
| `50_1_subthreshold_msr/*.md` | `50_1_subthreshold_msr/` | Phase 1 / CPU_SING | Runtime VID clamp, route reports, catalytic tape, `.holo`, VRM, detuning, GOE, and final pack. | Yes | Markdown lab reports are primary evidence. |
| `50_2_phase_locked_network/*.md` | `50_2_phase_locked_network/` | Phase 2 | Kuramoto, active phase, deep control, AGESA P4-safe, GOE, Ising, detuning, markers, and external measurement reports. | Yes | Markdown lab reports are primary Phase 2 evidence. |
| `50_2b_blackbox/*.md` | `50_2b_blackbox/` | Phase 2B black-box attractor | Passive-mechanism closure, decision tree, 2B.5A status, and transition reports. | Yes | Existing named Phase 2B workspace; keep as-is. |
| `50_2_firmware/*.md` | `50_2_firmware/` | Phase 2/3 firmware | AGESA recursive gate, master dispatch, donor diff, arg0C provenance. | Yes | Records AGESA route blockers and exact next artifacts. |
| `50_2_undervolt_research/*.md` | `50_2_undervolt_research/` | Research pathway | Undervolt pathway research and evidence inventory. | Yes | Markdown research reports are small and reproducible. |
| `50_2_firmware/cpu_hack/CPU_HACK_INDEX.md` | `50_2_firmware/cpu_hack/` | Firmware lab index | Folder map for firmware evidence, local tools, generated trees, and no-op rebuild blocker. | Yes | Keeps the organized firmware workspace readable. |
| `50_2_firmware/cpu_hack/agesa_trace/PATCH_ANALYSIS.md` | `50_2_firmware/cpu_hack/agesa_trace/` | Firmware lab | Patch analysis for AGESA/global branch/P4-safe route. | Yes | Primary firmware decision evidence. |
| `50_2_firmware/cpu_hack/agesa_trace/*.txt` | `50_2_firmware/cpu_hack/agesa_trace/` | Firmware lab text evidence | Decompile, disasm, xref, service table, record map, and MSR source proof artifacts. | Yes | Small text evidence for firmware route. |
| `50_2_firmware/cpu_hack/bios_parse/bios_dump.bin.report.txt` | `50_2_firmware/cpu_hack/bios_parse/` | Firmware lab parse report | UEFIExtract parse report for owned BIOS dump. | Yes | Small text parse evidence. |
| `50_2_firmware/cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt` | `50_2_firmware/cpu_hack/noop_replace/` | Firmware no-op rebuild | Authoritative no-op rebuild proof and byte-difference summary. | Yes | Small text evidence; proves rebuild/save gate without tracking binaries. |
| `50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt` | `50_2_firmware/cpu_hack/noop_replace/` | Firmware no-op rebuild | UEFIExtract parse report for accepted no-op rebuilt image. | Yes | Small text parse evidence for `NOOP_REBUILD_PROVEN`. |
| `50_2_firmware/cpu_hack/board_probe/check_ics*.ps1` | `50_2_firmware/cpu_hack/board_probe/` | Firmware/board lab | Local board/clock inspection helpers. | Yes | Source scripts, not generated artifacts. |
| `50_2_firmware/cpu_hack/bios_dump.bin` | `50_2_firmware/cpu_hack/` | Firmware lab raw artifact | Raw BIOS dump image. | No | Binary/sensitive; preserved locally. |
| `50_2_firmware/cpu_hack/bios_parse/*.csv` | `50_2_firmware/cpu_hack/bios_parse/` | Firmware lab generated | GUID CSV exports. | No | Generated artifacts. |
| `50_2_firmware/cpu_hack/bios_dump.bin.dump/` | `50_2_firmware/cpu_hack/` | Firmware lab generated | UEFIExtract tree for owned dump. | No | Generated artifact tree. |
| `50_2_firmware/cpu_hack/_tmp_coreboot_*/` | `50_2_firmware/cpu_hack/` | Firmware lab generated | Temporary coreboot/source extraction trees. | No | Generated/heavy local research trees. |
| `50_2_firmware/cpu_hack/tools/` | `50_2_firmware/cpu_hack/` | Firmware lab local tools | Downloaded/local tools. | No | Local tool binaries. |
| `50_2_firmware/cpu_hack/mod_donors/` | `50_2_firmware/cpu_hack/` | Firmware lab local | Downloaded stock/donor BIOS packages. | No | Binary/generated artifacts. |
| `50_2_firmware/cpu_hack/noop_replace/*.bin` | `50_2_firmware/cpu_hack/noop_replace/` | Firmware lab local | Rebuilt/attempted BIOS images. | No | Binary/generated artifacts. |
| `50_2_firmware/cpu_hack/noop_replace/*/body.bin` | `50_2_firmware/cpu_hack/noop_replace/` | Firmware lab generated | Target body extraction from accepted no-op rebuilt image. | No | Generated binary verification artifact; hash is recorded in text reports. |
| `50_2_firmware/cpu_hack/local_logs/` | `50_2_firmware/cpu_hack/local_logs/` | Firmware lab logs | Local probe and extraction logs. | No | Generated logs. |

## Session Scripts (organized by phase)

### Phase 1: MSR / Voltage / Frequency (`50_1_subthreshold_msr/src/`)

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
| `msr_load_affinity_characterizer.py` | Read-only load/affinity characterization for P4, COFVID, PSTATE_STATUS, and local TSC jitter | Yes |
| `msr_transition_jitter_probe.py` | Read-only COFVID/PSTATE transition counter and timing-jitter summary probe | Yes |
| `msr_state_window_oracle.py` | Read-only state-conditioned timing oracle with deterministic label-rotation nulls | Yes |
| `recovery_check.py` | Post-CMOS-clear P-state verification | Yes |
| `match_cores.py` | Set Cores 3+4 to matching 200MHz | Yes |
| `smbus_scan.sh` | SMBus/I2C bus scan for VRM controller | Yes |
| `nb_probe.sh` | Northbridge PCI config space probe | Yes |

### Phase 2: Kuramoto / Coupling / Oscillators (`50_2_phase_locked_network/src/`)

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
| `cacheline_phase_coupling.c` | Core-pinned cacheline phase-coupling harness with isolated, false-shared, atomic same-line, and cyclic-shift null modes | Yes |

### Phase 2B: Black-Box Attractor / Phase Oracle (`50_2b_blackbox/src/`)

| File | Purpose | Keep |
|------|---------|------|
| `active_catalytic_ising.c` | Active catalytic Ising comparator / Phase 3 bridge | Yes |
| `channel_matrix.c` | Passive channel matrix across QR, retrocausal, fingerprint, and detuned harness channels | Yes |
| `passive_attractor.c` | Passive hidden-attractor random-flip harness | Yes |
| `phase_oracle_ising.c` | Exp20 vertex/ensemble phase-oracle Ising branch | Yes |
| `topology_attractor.c` | Topology-encoded attractor tests | Yes |
| `wormhole_correlation.c` | Wormhole/correlation phase-transfer test | Yes |
| `wormhole_protocol_transfer.c` | Exp32 protocol transfer to catalytic tape | Yes |
| `optical_3sat_phase_port.c` | Exp26-style optical 3-SAT phase mapping with random and ablated nulls | Yes |
| `bloch_complex_ising.c` | Exp07-style Bloch/complex-plane active phase-oracle Ising port with random, sign-shuffled, and edge-rewired nulls | Yes |
| `spectral_problem_classifier.c` | Exp31-style spectral/topological router for active edge, vertex phase, and Bloch/complex solver families | Yes |
| `holo_mera_bridge.c` | Exp33-style `.holo`/MERA bridge from active phase-oracle output into reversible catalytic tape slots | Yes |

### Phase 3: Catalytic Computing Ladder (`50_3_catalytic_ladder/src/`)

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

### Phase 3B: Catalytic Substrate Primitive (`50_3b_substrate_primitive/src/`)

| File | Purpose | Keep |
|------|---------|------|
| `catalytic_invariant_probe.c` | Four-snapshot invariant probe with destructive, random reversible, random-answer, shuffled-schedule, and same-final-hash/wrong-answer controls | Yes |

### Phase 4: .holo Eigenbasis on Catalytic Silicon (`50_4_holo_eigenbasis/src/`)

| File | Purpose | Keep |
|------|---------|------|
| `phase4_bridge.c` | Phase 4.0 bridge gate (5/5 gates pass) | Yes |
| `eigenbasis_tape.c` | Phase 4.1A shared eigenbasis (4 tests, basis read-only) | Yes |
| `rotation_chain.c` | Phase 4.2A rotation chain (3 layers, forward/reverse) | Yes |
| `residual_channel.c` | Phase 4.3 residual channel from Phase 3B carrier into 2-bit `.holo` residual tags | Yes |
| `operator_goe.c` | Phase 4.4A catalytic operator-matrix eigenvalue spacing and null comparison | Yes |
| `holo_mini_model.c` | Phase 4.5 tiny graph-class `.holo` mini-model with decode and restore controls | Yes |
| `catcas_holo_harness.c` | Phase 4.6 CLI-style public `.holo` harness for residual, mini, GOE, and all-test modes | Yes |
| `polytope_hypothesis.c` | Canonical hardened Phase 5.6 polytope/positive-geometry full-carrier hypothesis harness | Yes |

### Phase 5.7: Entropic Boundary Geometry (`50_5_7_entropic_boundary/`)

| File | Purpose | Keep |
|------|---------|------|
| `PHASE5_7_ENTROPIC_BOUNDARY_GEOMETRY.md` | Hardened Phase 5.7 report for measured-runtime load/entropy computational boundary deformation | Yes |
| `PHASE5_7_INTEGRITY_AUDIT.md` | Phase 5.7 leakage/tautology audit proving class-label and synthetic-load routes were removed | Yes |
| `results/phase5_7_stdout.txt` | Captured Phase 5.7 run summary and verdict | Yes |
| `results/*.csv` | Generated detailed Phase 5.7 tables; keep as local artifacts unless explicitly staged | No |

---

## Phase Layout Verdict

- Keep the per-phase dirs `50_1_subthreshold_msr/`, `50_2_phase_locked_network/`, `50_2_firmware/`, `50_2b_blackbox/`, `50_2_undervolt_research/`, and `50_2_firmware/cpu_hack/` (Scheme 2 merge: no standalone session_scripts dir remains).
- Session scripts are folded into each phase dir's `src/` subdir, e.g. `50_1_subthreshold_msr/src/`, `50_2_phase_locked_network/src/`, `50_3_catalytic_ladder/src/`, `50_4_holo_eigenbasis/src/`, `50_5_6_polytope_geometry/src/`, and `50_5_7_entropic_boundary/src/`.
- Do not recreate generic `docs/`, `reports/`, `firmware/`, or `archive/` folders.
- Track markdown reports, source scripts, and small text evidence.
- Ignore raw BIOS images, extracted BIOS trees, generated logs, CSV exports, compiled binaries, and local tool trees.
- Phase 3 COMPLETE. Phase 4 Track A complete. Phase 4 Track B pending Phase 2. Phase 5.6 confirmed as `PHASE5_6_POLYTOPE_GEOMETRY_CONFIRMED`: the canonical harness now generates full T0/T1/T2/T3 carrier rows, rejects same-final-hash wrong-answer controls, predicts held-out rows, passes static projection hierarchy, and passes fine residual-boundary deformation. Phase 5.7 confirmed as `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED`: bounded measured runtime load deforms the computational carrier boundary while null exclusion survives, and the decisive correlation is within-load carrier/boundary rather than raw jitter.
