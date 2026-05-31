# CAT_CAS VIOLATION REMEDIATION ROADMAP

**Date**: 2026-05-30
**Status**: 0 / 140 resolved
**Sources**: CODEBASE_AUDIT_REPORT.md, MANIFESTO_ENFORCEMENT_AUDIT.md, REFACTORING_AUDIT.md, AUDIT_REPORT_PHASE_47.md

---

## LEGEND

| Priority | Symbol | Meaning |
|----------|--------|---------|
| **BLOCKER** | `[B]` | Prevents experiments from running correctly. Fix before ANY other work. |
| **CRITICAL** | `[C]` | Produces false results, null claims, or violates lab paradigm. |
| **HIGH** | `[H]` | Undermines reproducibility, claims, or statistical rigor. |
| **MEDIUM** | `[M]` | Code quality, maintainability, portability. |
| **DEBT** | `[D]` | Widespread patterns requiring systematic cleanup. |
| **PROCESS** | `[P]` | Governance/process failures. Prevent recurrence. |

---

## SECTION A: BLOCKER — RUNTIME BUGS (4 items)

*Experiments crash or produce garbage. Must be fixed first.*

- [ ] **A-1** `[B]` Fix Feistel swap in `15_hdd_native_inference/experiment.py:166-170`
  - 2-step XOR swap produces `a^b` in both halves. Needs 3-step swap.
  - Evidence: 100/100 failures across random seeds.
  
- [ ] **A-2** `[B]` Fix F16 weight loading in `16_catalytic_27b_inference/experiment.py:236-237`
  - `uint16.astype(float32)` converts integer values, not IEEE 754 bit patterns.
  - Missing `np.frombuffer(uint16, dtype=np.float16).astype(np.float32)`.
  
- [ ] **A-3** `[B]` Fix undefined `k95_phase` in `16_catalytic_27b_inference/_test_phase.py:89`
  - Line 50 defines `k95`, line 89 references `k95_phase`. NameError at runtime.
  
- [ ] **A-4** `[B]` Fix 6 AttributeErrors in `30_boundary_stress/1_memory_collision.py:114-138`
  - `tape.barrier`, `tape.running`, `tape.collisions_active`, `tape.collisions_unalloc`, `tape.verify_all()` do not exist.
  - `catalytic_encrypt` is not defined.

---

## SECTION B: CRITICAL — NULL RESULTS AND FALSE CLAIMS (6 items)

*Experiments that run but produce invalid or unsupported results.*

- [ ] **B-1** `[C]` Deprecate or delete `47_phase_atom/47_4_lhc_overflow_exploit/`
  - COMPLETE NULL RESULT. Palindrome rate on 64-bit random strings has expected mean 0.5.
  - All observed values (0.375-0.594) are within +/-2 sigma of noise.
  - Boson/Fermion classification via arbitrary 0.55 threshold on noise.
  - No structural correspondence to particle generation. Every sensor reading = statistical artifact.
  - Reference: AUDIT_REPORT_PHASE_47.md verdict "FAILS."

- [ ] **B-2** `[C]` Reclassify `47_phase_atom/47_5_higgs_mechanism/` as WEAK
  - Claims Higgs mechanism at 512-bit boundary = cache-line crossing.
  - Fact: mpmath bigints use dynamically allocated digit arrays, not contiguous cache lines.
  - The 512-bit threshold is a post-hoc heuristic on 10 data points.
  - Timing data is valid (bigger operands cost more), but the Higgs mechanism claim is factually incorrect.
  - Reference: AUDIT_REPORT_PHASE_47.md verdict "WEAK / FAILS SPECIFIC CLAIM."

- [ ] **B-3** `[C]` Fix inflated PUSHED_REPORT claims (3 items)
  - `REPORTS/PUSHED_REPORT_FINAL_14.md`: "3076.9x KV compression" -- actual output shows 12.5x max.
  - `REPORTS/PUSHED_REPORT_INFINITY.md`: "Cross-talk exactly 0.000000" -- infinity version shows 18,214+ at 10 models.
  - `REPORTS/PUSHED_REPORT_INFINITY.md`: "4096x4096 classical dataset" -- actual experiment tests 18-qubit circuits (262,144 amplitudes), not a 16.7M parameter classical dataset.

- [ ] **B-4** `[C]` Fix Exp 13 infinity cross-talk in `13_orthogonal_multimodel/1_infinity_multimodel.py`
  - Base experiment (2 models): cross-talk 1.98e-16. Works correctly.
  - Infinity experiment: cross-talk 18,214+ at 10 models, 1M+ at 100 models.
  - Extraction formula `X_signed @ W_shared` is mathematically wrong.

- [ ] **B-5** `[C]` Fix Exp 13 snapshot drift comparison in `13_orthogonal_multimodel/experiment.py:275`
  - `snap_a_init = P_A @ (np.array(list(SharedTape().tape[:TAPE_DIM])` creates a NEW SharedTape with different seed.
  - Drift comparison is against wrong baseline.

- [ ] **B-6** `[C]` Fix Exp 7 non-determinism in `07_quantum_simulator/stealth_borrowing.py:95`
  - `measured_val = 0 if np.random.rand() < prob_0 else 1` -- non-deterministic measurement.
  - Violates determinism requirement (STEWARDSHIP.md, MANIFESTO.md Section 8).

---

## SECTION C: HIGH — CRITIC M-1 THROUGH M-4 (10 items)

*Hardcoded invariants, ceremonial tapes, arbitrary thresholds, proven-impossible approaches.*

### M-1: Hardcoded Invariants (1 item)

- [ ] **C-1** `[H]` Fix `46_phase_bio/validation_mandate4_null_models.py`
  - `residual = 5.0` assigned by `state == 'annihilated'` instead of being dynamically computed.
  - This is the residual pattern of the already-fixed 46.6 hardcoded Bott Index. Clean up.

### M-2: Tautological Tapes (4 items)

*Tape instantiated, verified, never XOR-modified. SHA-256 guaranteed to pass. Not catalytic.*

- [ ] **C-2** `[H]` Fix `46_phase_bio/validation_mandate4_null_models.py` — ceremonial tape
- [ ] **C-3** `[H]` Fix `46_phase_bio/validation_mandate5_conservation.py` — ceremonial tape
- [ ] **C-4** `[H]` Fix `46_phase_bio/46_5_neural_binding_oracle/validation_real_connectome.py` — ceremonial tape
- [ ] **C-5** `[H]` Fix `46_phase_bio/46_6_morphogenesis_oracle/validation_real_morphogenesis.py` — ceremonial tape

### M-3: Arbitrary Threshold (1 item)

- [ ] **C-6** `[H]` Fix `47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py`
  - Threshold 0.55 on palindrome match rate with expected mean 0.5.
  - MUST BE BLOCKED BY B-1 (deprecate experiment entirely). If kept, derive threshold from physical principle, not from data.

### M-4: NxN Compression (4 items)

*NxN matrix for SAT/NP-complete classification. Proven impossible in Phase 45.5.*

- [ ] **C-7** `[H]` Fix or remove `40_5d_floquet_oracle/40_sub/40_sub_1_temporal_sat/40_sub_1_temporal_sat.py`
- [ ] **C-8** `[H]` Fix or remove `40_5d_floquet_oracle/40_sub/40_sub_2_floquet_swarm/40_sub_2_floquet_swarm.py`
- [ ] **C-9** `[H]` Fix or remove `40_5d_floquet_oracle/40_sub/40_sub_4_sat_swarm/40_sub_4_sat_swarm.py`
- [ ] **C-10** `[H]` Fix or remove `45_phase_math/45_5_p_vs_np/45_5_p_vs_np_catalytic.py`
  - This experiment is its OWN NULL RESULT. Phase 45.5 master report admits "UNIVERSAL FAILURE. 0/4 hardening gates."
  - The NxN catalytic approach is proven impossible. These files MUST NOT mislead future agents.

---

## SECTION D: HIGH — CEREMONIAL TAPE CRISIS IN PHASE 47 (6 items)

*ALL Phase 47 experiments use BennettHistoryTape that appends to a Python list and pops it. The bytearray is never XOR'd. Verification is structurally guaranteed to pass. Not catalytic.*

- [ ] **D-1** `[H]` Fix `47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py`
- [ ] **D-2** `[H]` Fix `47_phase_atom/47_2_electron_edge_states/47_2_electron_edge_states.py`
- [ ] **D-3** `[H]` Fix `47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py`
  - Gap threshold 0.001 is near numerical noise for 225x225 matrix. Increase statistical rigor.
- [ ] **D-4** `[H]` Fix `47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py`
  - BLOCKED BY B-1. If experiment is deprecated, skip.
- [ ] **D-5** `[H]` Fix `47_phase_atom/47_5_higgs_mechanism/47_5_higgs_mechanism.py`
  - BLOCKED BY B-2. If reclassified as WEAK, tape fix still required.
- [ ] **D-6** `[H]` Fix `47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py`
  - Best-engineered experiment in Phase 47. Tape should be made genuinely catalytic.

**Cross-cutting fix**: Either make BennettHistoryTape XOR-modify the actual bytearray, or replace it with a real `CatalyticTape` that tracks `bytes_written` and FAILS verification if `bytes_written == 0`. The current implementation is a trap.

---

## SECTION E: HIGH — MISSING NULL MODELS (26 items)

*Hardening gates present but no null/shuffled/random baseline detected.*

### Phase 04-05 (3 items)
- [ ] **E-1** `[H]` Add null model to `04_thermodynamic_cpu/reversible_cpu.py`
- [ ] **E-2** `[H]` Add null model to `05_multibit_compiler/compiler_experiment.py`
- [ ] **E-3** `[H]` Add null model to `05_multibit_compiler/reversible_cpu.py`

### Phase 33 (1 item)
- [ ] **E-4** `[H]` Add null model to `33_mera_compression/20_tuneable_holo_model.py`

### Phase 40 (1 item)
- [ ] **E-5** `[H]` Add null model to `40_5d_floquet_oracle/40_sub/40_sub_3_quantum/40_sub_3_quantum.py`

### Phase 41 (1 item)
- [ ] **E-6** `[H]` Add null model to `41_toe_bulletproof/41d_transfer_clock.py`

### Phase 45 — PROVEN CLAIMS WITHOUT NULL BASELINES (5 items)
- [ ] **E-7** `[H]` Add null model to `45_phase_math/45_2_navier_stokes/45_2_navier_stokes_smoothness.py` — PROVEN claim
- [ ] **E-8** `[H]` Add null model to `45_phase_math/45_3_erdos_discrepancy/45_3_erdos_discrepancy.py` — PROVEN claim
- [ ] **E-9** `[H]` Add null model to `45_phase_math/45_3_erdos_discrepancy/45_3_erdos_spatial_upgrade.py` — PROVEN claim
- [ ] **E-10** `[H]` Add null model to `45_phase_math/45_4_riemann_hypothesis/45_4_riemann_hypothesis.py` — PROVEN claim
- [ ] **E-11** `[H]` Add null model to `45_phase_math/45_5_p_vs_np/45_5_p_vs_np_time_crystal.py`
- [ ] **E-12** `[H]` Add null model to `45_phase_math/45_6_yang_mills/45_6_yang_mills_gribov_gap.py` — PROVEN claim
- [ ] **E-13** `[H]` Add null model to `45_phase_math/45_6_yang_mills/45_6_yang_mills_mass_gap.py` — PROVEN claim

### Phase 46 (5 items)
- [ ] **E-14** `[H]` Add null model to `46_phase_bio/46_2_folding_pathway/46_2_folding_pathway_oracle.py`
- [ ] **E-15** `[H]` Add null model to `46_phase_bio/46_3_prion_contagion/46_3_prion_contagion_oracle.py`
- [ ] **E-16** `[H]` Add null model to `46_phase_bio/46_5_neural_binding_oracle/46_5_neural_binding_oracle.py`
- [ ] **E-17** `[H]` Add null model to `46_phase_bio/46_6_morphogenesis_oracle/46_6_morphogenesis_oracle.py`

### Phase 47 (6 items)
- [ ] **E-18** `[H]` Add null model to `47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py`
- [ ] **E-19** `[H]` Add null model to `47_phase_atom/47_2_electron_edge_states/47_2_electron_edge_states.py`
- [ ] **E-20** `[H]` Add null model to `47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py`
- [ ] **E-21** `[H]` Add null model to `47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py` — BLOCKED BY B-1
- [ ] **E-22** `[H]` Add null model to `47_phase_atom/47_5_higgs_mechanism/47_5_higgs_mechanism.py` — BLOCKED BY B-2
- [ ] **E-23** `[H]` Add null model to `47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py`

---

## SECTION F: HIGH — MISSING STATISTICS (46 items)

*Numeric results without p-value, CI, std, effect size, bootstrap. Grouped by phase for batch fixes.*

### Phase 07 (1 item)
- [ ] **F-1** `[H]` Add statistics to `07_quantum_simulator/1_infinity_quantum.py`

### Phase 08 (1 item)
- [ ] **F-2** `[H]` Add statistics to `08_catalytic_gpt/run_multi_outputs.py`

### Phase 11 (1 item)
- [ ] **F-3** `[H]` Add statistics to `11_grail_calorimeter/1_infinity_calorimeter.py`

### Phase 19 (1 item)
- [ ] **F-4** `[H]` Add statistics to `19_catalytic_computronium/experiment.py`

### Phase 23 (2 items)
- [ ] **F-5** `[H]` Add statistics to `23_temporal_catalysis/2_real_weights.py`
- [ ] **F-6** `[H]` Add statistics to `23_temporal_catalysis/5_temporal_attention.py`

### Phase 24 (2 items)
- [ ] **F-7** `[H]` Add statistics to `24_quantum_catalytic_entanglement/3_massive_scale.py`
- [ ] **F-8** `[H]` Add statistics to `24_quantum_catalytic_entanglement/7_dpr_scaling.py`

### Phase 33 (3 items)
- [ ] **F-9** `[H]` Add statistics to `33_mera_compression/15_temporal_calibration.py`
- [ ] **F-10** `[H]` Add statistics to `33_mera_compression/19_er_epr_verify.py`
- [ ] **F-11** `[H]` Add statistics to `33_mera_compression/_k_sweep.py`

### Phase 34 — Riemann Proof Chain (5 items)
- [ ] **F-12** `[H]` Add statistics to `34_zeta_eigenbasis/03_infinity_bootstrap/14_riemann_zero_telescope.py`
- [ ] **F-13** `[H]` Add statistics to `34_zeta_eigenbasis/03_infinity_bootstrap/18_googol_zero_telescope.py`
- [ ] **F-14** `[H]` Add statistics to `34_zeta_eigenbasis/04_catalytic_engines/16_catalytic_zero_engine.py`
- [ ] **F-15** `[H]` Add statistics to `34_zeta_eigenbasis/05_topological_proof/19_topological_zeta_winding.py`
  - W=+3 is exact (topological invariant). But empirical quantities in report need statistics.
- [ ] **F-16** `[H]` Add statistics to `34_zeta_eigenbasis/05_topological_proof/20_transcendent_winding_oracle.py`
- [ ] **F-17** `[H]` Add statistics to `34_zeta_eigenbasis/05_topological_proof/21_absolute_infinity_collapse.py`

### Phase 35 (2 items)
- [ ] **F-18** `[H]` Add statistics to `35_topological_halting_oracle/35.2_nonhermitian_oracle/36_nonhermitian_oracle.py`
- [ ] **F-19** `[H]` Add statistics to `35_topological_halting_oracle/35.3_skin_effect/35.3_hatano_nelson_skin_effect.py`

### Phase 36 (1 item)
- [ ] **F-20** `[H]` Add statistics to `36_bekenstein_godel/36_bekenstein_godel_singularity_catalytic.py`

### Phase 40 (8 items)
- [ ] **F-21** `[H]` Add statistics to `40_5d_floquet_oracle/40_5d_floquet_oracle.py`
- [ ] **F-22** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_11_nondtc/40_sub_11_nondtc_v2.py`
- [ ] **F-23** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_13_rust/40_sub_13_rust.py`
- [ ] **F-24** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_2_floquet_swarm/40_sub_2_tree_swarm.py`
- [ ] **F-25** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_3_quantum/40_sub_3_quantum.py`
- [ ] **F-26** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_4_temporal_signal/40_sub_4_temporal_signal.py`
- [ ] **F-27** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_5_pulseprog/40_sub_5_pulseprog_v2.py`
- [ ] **F-28** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_6_temporal_memory/40_sub_6_temporal_memory.py`
- [ ] **F-29** `[H]` Add statistics to `40_5d_floquet_oracle/40_sub/40_sub_8_addressing/40_sub_8_addressing.py`

### Phase 42 — Event Horizon / BLACK_HOLES / COSMOS / ULTRA (12 items)
- [ ] **F-30** `[H]` Add statistics to `42_computational_event_horizon/10_information_paradox.py`
- [ ] **F-31** `[H]` Add statistics to `42_computational_event_horizon/11_photon_sphere.py`
- [ ] **F-32** `[H]` Add statistics to `42_computational_event_horizon/2_wormhole_mutation_exploit.py`
- [ ] **F-33** `[H]` Add statistics to `42_computational_event_horizon/3_quantum_tunneling_exploit.py`
- [ ] **F-34** `[H]` Add statistics to `42_computational_event_horizon/4_page_curve_entropy.py`
- [ ] **F-35** `[H]` Add statistics to `42_computational_event_horizon/5_gravitational_waves.py`
- [ ] **F-36** `[H]` Add statistics to `42_computational_event_horizon/6_holographic_boundary.py`
- [ ] **F-37** `[H]` Add statistics to `42_computational_event_horizon/9_quantum_superposition.py`
- [ ] **F-38** `[H]` Add statistics to `42_computational_event_horizon/BLACK_HOLES/exp_20_amps_firewall/20_amps_firewall.py`
- [ ] **F-39** `[H]` Add statistics to `42_computational_event_horizon/BLACK_HOLES/exp_21_bekenstein_hawking/21_bekenstein_hawking_area_law.py`
- [ ] **F-40** `[H]` Add statistics to `42_computational_event_horizon/BLACK_HOLES/exp_22_kerr_ergosphere/22_kerr_ergosphere.py`
- [ ] **F-41** `[H]` Add statistics to `42_computational_event_horizon/BLACK_HOLES/exp_23_true_singularity/23_true_singularity_core_crush.py`
- [ ] **F-42** `[H]` Add statistics to `42_computational_event_horizon/COSMOS/exp_24_dark_matter/24_dark_matter_orphaned_pointers.py`
- [ ] **F-43** `[H]` Add statistics to `42_computational_event_horizon/COSMOS/exp_25_dark_energy/25_dark_energy_expansion.py`
- [ ] **F-44** `[H]` Add statistics to `42_computational_event_horizon/COSMOS/exp_27_arrow_of_time/42_27_arrow_of_time.py`
- [ ] **F-45** `[H]` Add statistics to `42_computational_event_horizon/ULTRA/exp_14_boltzmann_brain/rust/plot_entropy.py`
- [ ] **F-46** `[H]` Add statistics to `42_computational_event_horizon/ULTRA/exp_15_quantum_gravity_unification/rust/unification_proof.py`

### Phase 45 (1 item)
- [ ] **F-47** `[H]` Add statistics to `45_phase_math/45_1_collatz_oracle/45_1_collatz_oracle.py`

### Phase 47 (4 items)
- [ ] **F-48** `[H]` Add statistics to `47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py`
- [ ] **F-49** `[H]` Add statistics to `47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py`
- [ ] **F-50** `[H]` Add statistics to `47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py` — BLOCKED BY B-1
- [ ] **F-51** `[H]` Add statistics to `47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py`

---

## SECTION G: MEDIUM — HARDCODED OUTPUT PATHS (17 items)

*Paths containing `THOUGHT/LAB/CAT_CAS/` hardcoded. Replace with `os.path.dirname(__file__)` relative paths.*

### Phase 46 (8 items)
- [ ] **G-1** `[M]` Fix paths in `46_phase_bio/validation_mandate4_null_models.py`
- [ ] **G-2** `[M]` Fix paths in `46_phase_bio/46_1_protein_folding/46_1_protein_folding_oracle.py`
- [ ] **G-3** `[M]` Fix paths in `46_phase_bio/46_2_folding_pathway/46_2_folding_pathway_oracle.py`
- [ ] **G-4** `[M]` Fix paths in `46_phase_bio/46_3_prion_contagion/46_3_prion_contagion_oracle.py`
- [ ] **G-5** `[M]` Fix paths in `46_phase_bio/46_4_topological_genetic_code/46_4_topological_genetic_code_oracle.py`
- [ ] **G-6** `[M]` Fix paths in `46_phase_bio/46_5_neural_binding_oracle/46_5_neural_binding_oracle.py`
- [ ] **G-7** `[M]` Fix paths in `46_phase_bio/46_6_morphogenesis_oracle/46_6_morphogenesis_oracle.py`
- [ ] **G-8** `[M]` Fix paths in `46_phase_bio/46_6_morphogenesis_oracle/validation_real_morphogenesis.py`

### Phase 47 (9 items)
- [ ] **G-9** `[M]` Fix paths in `47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py`
- [ ] **G-10** `[M]` Fix paths in `47_phase_atom/47_2_electron_edge_states/47_2_electron_edge_states.py`
- [ ] **G-11** `[M]` Fix paths in `47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py`
- [ ] **G-12** `[M]` Fix paths in `47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py` — BLOCKED BY B-1
- [ ] **G-13** `[M]` Fix paths in `47_phase_atom/47_5_higgs_mechanism/47_5_higgs_mechanism.py` — BLOCKED BY B-2
- [ ] **G-14** `[M]` Fix paths in `47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py`

---

## SECTION H: MEDIUM — REMAINING CODEBASE AUDIT BUGS (10 items)

### High bugs from CODEBASE_AUDIT_REPORT.md
- [ ] **H-1** `[H]` Fix lm_head overwrite in `16_catalytic_27b_inference/experiment.py:398-414`
  - Line 398 saves result, lines 399-405 compute lm_head, line 414 overwrites with original. Dead computation.
  
- [ ] **H-2** `[H]` Fix `_ground_truth` side-effect in `11_grail_calorimeter/workloads.py:215,344`
  - `run_irreversible()` must be called before `run_reversible()` or assertion crashes.

- [ ] **H-3** `[H]` Fix 41b = 41a exact duplicate
  - `41_toe_bulletproof/41a_mpowinding.py` and `41_toe_bulletproof/41b_godel_ep.py` are byte-for-byte identical (MD5 match). Either merge, rename, or implement actual Godel-EP content.

### Medium bugs
- [ ] **H-4** `[M]` Fix floating-point equality in `04_thermodynamic_cpu/1_infinity_thermo.py:58`
  - `if heat_dissipated == 0.0 and mse == 0.0` — use tolerance comparison.

- [ ] **H-5** `[M]` Fix floating-point equality in `07_quantum_simulator/experiment.py:151`
  - `conserved = (prob_sample_post == prob_sample_pre)` — use `np.isclose()`.

### Deprecated APIs
- [ ] **H-6** `[M]` Replace `torch.svd()` with `torch.linalg.svd()` in:
  - `14_bekenstein_violator/1_infinity_violator.py`
  - `17_temporal_bootstrap/1_time_travel_compute.py`

- [ ] **H-7** `[M]` Migrate `np.random.RandomState` to `np.random.Generator` in 25 files (batch task)

### Missing/incomplete
- [ ] **H-8** `[M]` Add `finally` block to mmap in `14_bekenstein_violator/hdd_scale.py`

- [ ] **H-9** `[M]` Fix dead code in `33_mera_compression/_infinity_engine.py:280`
  - `x = x.float() * math.cos(phi) + x.float() * math.sin(phi)` is scaling `x*(cos+sin)`, not rotation.
  
- [ ] **H-10** `[M]` Fix dead code in `33_mera_compression/_tape_engine.py:215-218`
  - `rope()` function has no return statement. Falls through to comment "Skip RoPE for simplicity."

---

## SECTION I: DEBT — WIDESPREAD PATTERNS (3 batch tasks)

- [ ] **I-1** `[D]` Fix 46 bare `except:` clauses across the codebase
  - Locations: `generate_gold_data.py` (2), `3_recursive_rho.py` (3), 33+ in `33_mera_compression/`, `connes_scattering.py` (1), `hp_matrix_search.py` (1), `36d_scaling_sweep.py` (1), plus others.
  - Each must specify the actual exception type(s), never bare.
  - Reference: STEWARDSHIP.md explicitly forbids bare excepts.

- [ ] **I-2** `[D]` Fix `torch.load()` without `weights_only=True` (2 files)
  - `16_catalytic_27b_inference/_check_holo.py:7`
  - `25_lattice_holography/2_holographic_svp.py:7`
  - Security: unsafe deserialization. Must set `weights_only=True` or use `safetensors`.

- [ ] **I-3** `[D]` Fix 6+ hardcoded Windows paths (non-M-7, non-portable)
  - `08_catalytic_gpt/run_6_gemmas.py:8`
  - `14_bekenstein_violator/fractal_cache_exploit.py:24,69`
  - `14_bekenstein_violator/hdd_scale.py:27,48`
  - `15_hdd_native_inference/experiment.py:599`

---

## SECTION J: DEBT — DOCUMENTATION (6 items)

- [ ] **J-1** `[D]` Fix spelling: "Haydeng-Preskill" -> "Hayden-Preskill" in `README.md`
- [ ] **J-2** `[D]` Fix spelling: "Assesment" -> "Assessment" in `REPORTS/5-21-2026_Integrity_Assesment.md`
- [ ] **J-3** `[D]` Fix missing files referenced in README:
  - `06_catalytic_nn/catalytic_inference.py` — MISSING
  - `06_catalytic_nn/classical_inference.py` — MISSING
  - `06_catalytic_nn/generate_model_and_data.py` — MISSING
  - `06_catalytic_nn/report.md` — MISSING
  - `20_catalytic_eigen_shor/20.1/rust_ffi/` — MISSING
  - `_10_catalytic_27b.py` — MISSING (referenced in README line 296)
- [ ] **J-4** `[D]` Update `REPORTS/master_report.md` — covers only 9 experiments of 41+
- [ ] **J-5** `[D]` Remove or annotate unused imports (5 files):
  - `01_tree_evaluation/scale_experiment.py`: `hashlib`, `numpy`
  - `22_superconducting_inference/1_zero_power_attention.py`: `sys`
  - `23_temporal_catalysis/2_real_weights.py`: `sys`
  - `23_temporal_catalysis/5_temporal_attention.py`: `sys`
- [ ] **J-6** `[D]` Deduplicate `04_thermodynamic_cpu/reversible_cpu.py` and `05_multibit_compiler/reversible_cpu.py` (exact duplicate)

---

## SECTION K: PROCESS — PREVENT RECURRENCE (4 items)

*These address the ROOT CAUSE: why 99+ violations accumulated.*

- [ ] **K-1** `[P]` Enforce zero-violation pre-commit critic pass
  - Current: 99 violations committed without critic run.
  - Fix: Pre-commit hook must block any CAT_CAS commit with critic violations.
  - Exceptions require Lead Physicist waiver recorded in `REPORTS/VIOLATIONS/waivers.jsonl`.

- [ ] **K-2** `[P]` Deprecate BennettHistoryTape or make it fail-safe
  - If `verify()` is called and `bytes_written == 0`, verification must FAIL with explicit error.
  - "Ceremonial tape detected: no bytes were XOR-modified. This is not catalytic computing."
  - Reference: MANIFESTO.md Section "M-2: Tautological Tape Verification"

- [ ] **K-3** `[P]` Require manual isomorphism audit for each new phase
  - The critic cannot detect false isomorphisms (Exp 47.4), claim-code gaps (Phase 46.6 Bott Index), or null results.
  - Each new phase must produce an audit file modeled after `AUDIT_REPORT_PHASE_47.md`.
  - Audit must test: sensor validity, structural correspondence, claim-code alignment, gate integrity.

- [ ] **K-4** `[P]` Add `verify_tape_modification()` to the critic as M-8
  - Extend the critic to check: if `CatalyticTape`/`BennettHistoryTape` is instantiated AND `.verify()` is called, but `.write()` / `XOR` / `record_operation` is never called on the SAME tape instance, flag as M-8 CEREMONIAL TAPE.
  - Current M-2 detection is limited. The Phase 47 BennettHistoryTape pattern (list append/pop, no bytearray XOR) slips through because `record_operation` IS called — just on the wrong abstraction.
  - M-8 should detect: `BennettHistoryTape` instantiation without any `bytearray.__setitem__` or `xor` on the underlying bytes.

---

## SECTION L: QUALITATIVE — PHASE 46 REMAINING LIMITATIONS (4 items)

*From REFACTORING_AUDIT.md. Not critic-detectable. Not yet fixed.*

- [ ] **L-1** `[M]` Address IPR signal degradation at large L in `46_1_protein_folding/` and `46_2_folding_pathway/`
  - 2D contact map IPR ~ 1/L. Directional ordering holds but absolute classification degrades with scale.
  - Document limitation in experiment reports; do not claim scale-invariance without qualification.

- [ ] **L-2** `[M]` Document prion non-propagation in `46_3_prion_contagion/`
  - Prion seed is DETECTABLE as IPR impurity but does not propagate winding number to neighbors.
  - Contagion requires dynamical mechanisms beyond the static lattice model.
  - Update experiment claims to reflect honest physics: impurity detection, not contagion.

- [ ] **L-3** `[M]` Add biological connectome validation for `46_5_neural_binding_oracle/`
  - Current core oracle uses synthetic Watts-Strogatz graph.
  - `validation_real_connectome.py` already uses C. elegans data. Bridge the two — either replace synthetic or clearly label synthetic as phase 0 proof-of-concept.

- [ ] **L-4** `[M]` Fix Bott Index projector at Exceptional Points in `46_6_morphogenesis_oracle/`
  - Current workaround: 1D slice IPR bypasses the Bott Index because the spectral projector fails at EPs.
  - The 1D slice IPR is valid but is a workaround. A robust Bott Index computation at EPs remains an open engineering problem. Document this.

---

## PROGRESS SUMMARY

| Section | Category | Total | Completed | Remaining |
|---------|----------|-------|-----------|-----------|
| A | Blocker — Runtime bugs | 4 | 0 | 4 |
| B | Critical — Null results / false claims | 6 | 0 | 6 |
| C | High — Critic M-1 through M-4 | 10 | 0 | 10 |
| D | High — Ceremonial tape crisis (Phase 47) | 6 | 0 | 6 |
| E | High — Missing null models | 26 | 0 | 26 |
| F | High — Missing statistics | 46 | 0 | 46 |
| G | Medium — Hardcoded paths | 17 | 0 | 17 |
| H | Medium — Remaining codebase bugs | 10 | 0 | 10 |
| I | Debt — Widespread patterns | 3 | 0 | 3 |
| J | Debt — Documentation | 6 | 0 | 6 |
| K | Process — Prevent recurrence | 4 | 0 | 4 |
| L | Qualitative — Phase 46 limitations | 4 | 0 | 4 |
| | **TOTAL** | **142** | **0** | **142** |

---

## EXECUTION ORDER

Fix in this order. Each section unblocks the next.

1. **A (Blockers)**: Fix the 4 runtime crashes. Experiments must run before they can be evaluated.
2. **B (Critical)**: Deprecate null results (47.4), reclassify weak claims (47.5), fix inflated reports.
3. **C (M-1 to M-4)**: Hardcoded invariants, ceremonial tapes, arbitrary thresholds, NxN dead code.
4. **D (Phase 47 tapes)**: Make BennettHistoryTape genuinely catalytic or replace.
5. **K (Process)**: Prevent recurrence BEFORE fixing the remaining 88 items. Otherwise new violations accumulate.
6. **E-F (Null models + statistics)**: 72 items. Batch by phase. Largest volume.
7. **G (Paths)**: 17 mechanical fixes. Quick win.
8. **H (Remaining bugs)**: 10 items. Lower priority than null models/statistics.
9. **I-J (Debt + docs)**: 9 items. Cleanup after critical work.
10. **L (Phase 46 limitations)**: Document. No code changes needed for these 4.
