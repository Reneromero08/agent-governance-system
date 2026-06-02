"""Classify E/F files as real (computation) vs fake (text labels)."""
import os, re

SECTION_E = [
    ("04", "04_thermodynamic_cpu/reversible_cpu.py"),
    ("05", "05_multibit_compiler/compiler_experiment.py"),
    ("05", "05_multibit_compiler/reversible_cpu.py"),
    ("33", "33_mera_compression/20_tuneable_holo_model.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_3_quantum/40_sub_3_quantum.py"),
    ("41", "41_toe_bulletproof/41d_transfer_clock.py"),
    ("45", "45_phase_math/45_2_navier_stokes/45_2_navier_stokes_smoothness.py"),
    ("45", "45_phase_math/45_3_erdos_discrepancy/45_3_erdos_discrepancy.py"),
    ("45", "45_phase_math/45_3_erdos_discrepancy/45_3_erdos_spatial_upgrade.py"),
    ("45", "45_phase_math/45_4_riemann_hypothesis/45_4_riemann_hypothesis.py"),
    ("45", "45_phase_math/45_5_p_vs_np/45_5_p_vs_np_time_crystal.py"),
    ("45", "45_phase_math/45_6_yang_mills/45_6_yang_mills_gribov_gap.py"),
    ("45", "45_phase_math/45_6_yang_mills/45_6_yang_mills_mass_gap.py"),
    ("46", "46_phase_bio/46_2_folding_pathway/46_2_folding_pathway_oracle.py"),
    ("46", "46_phase_bio/46_3_prion_contagion/46_3_prion_contagion_oracle.py"),
    ("46", "46_phase_bio/46_5_neural_binding_oracle/46_5_neural_binding_oracle.py"),
    ("46", "46_phase_bio/46_6_morphogenesis_oracle/46_6_morphogenesis_oracle.py"),
    ("47", "47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py"),
    ("47", "47_phase_atom/47_2_electron_edge_states/47_2_electron_edge_states.py"),
    ("47", "47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py"),
    ("47", "47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py"),
    ("47", "47_phase_atom/47_5_higgs_mechanism/47_5_higgs_mechanism.py"),
    ("47", "47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py"),
]

SECTION_F = [
    ("07", "07_quantum_simulator/1_infinity_quantum.py"),
    ("08", "08_catalytic_gpt/run_multi_outputs.py"),
    ("11", "11_grail_calorimeter/1_infinity_calorimeter.py"),
    ("19", "19_catalytic_computronium/experiment.py"),
    ("23", "23_temporal_catalysis/2_real_weights.py"),
    ("23", "23_temporal_catalysis/5_temporal_attention.py"),
    ("24", "24_quantum_catalytic_entanglement/3_massive_scale.py"),
    ("24", "24_quantum_catalytic_entanglement/7_dpr_scaling.py"),
    ("33", "33_mera_compression/15_temporal_calibration.py"),
    ("33", "33_mera_compression/19_er_epr_verify.py"),
    ("33", "33_mera_compression/_k_sweep.py"),
    ("34", "34_zeta_eigenbasis/03_infinity_bootstrap/14_riemann_zero_telescope.py"),
    ("34", "34_zeta_eigenbasis/03_infinity_bootstrap/18_googol_zero_telescope.py"),
    ("34", "34_zeta_eigenbasis/04_catalytic_engines/16_catalytic_zero_engine.py"),
    ("34", "34_zeta_eigenbasis/05_topological_proof/19_topological_zeta_winding.py"),
    ("34", "34_zeta_eigenbasis/05_topological_proof/20_transcendent_winding_oracle.py"),
    ("34", "34_zeta_eigenbasis/05_topological_proof/21_absolute_infinity_collapse.py"),
    ("35", "35_topological_halting_oracle/35.2_nonhermitian_oracle/36_nonhermitian_oracle.py"),
    ("35", "35_topological_halting_oracle/35.3_skin_effect/35.3_hatano_nelson_skin_effect.py"),
    ("36", "36_bekenstein_godel/36_bekenstein_godel_singularity_catalytic.py"),
    ("40", "40_5d_floquet_oracle/40_5d_floquet_oracle.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_11_nondtc/40_sub_11_nondtc_v2.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_13_rust/40_sub_13_rust.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_2_floquet_swarm/40_sub_2_tree_swarm.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_3_quantum/40_sub_3_quantum.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_4_temporal_signal/40_sub_4_temporal_signal.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_5_pulseprog/40_sub_5_pulseprog_v2.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_6_temporal_memory/40_sub_6_temporal_memory.py"),
    ("40", "40_5d_floquet_oracle/40_sub/40_sub_8_addressing/40_sub_8_addressing.py"),
    ("42", "42_computational_event_horizon/10_information_paradox.py"),
    ("42", "42_computational_event_horizon/11_photon_sphere.py"),
    ("42", "42_computational_event_horizon/2_wormhole_mutation_exploit.py"),
    ("42", "42_computational_event_horizon/3_quantum_tunneling_exploit.py"),
    ("42", "42_computational_event_horizon/4_page_curve_entropy.py"),
    ("42", "42_computational_event_horizon/5_gravitational_waves.py"),
    ("42", "42_computational_event_horizon/6_holographic_boundary.py"),
    ("42", "42_computational_event_horizon/9_quantum_superposition.py"),
    ("42", "42_computational_event_horizon/BLACK_HOLES/exp_20_amps_firewall/20_amps_firewall.py"),
    ("42", "42_computational_event_horizon/BLACK_HOLES/exp_21_bekenstein_hawking/21_bekenstein_hawking_area_law.py"),
    ("42", "42_computational_event_horizon/BLACK_HOLES/exp_22_kerr_ergosphere/22_kerr_ergosphere.py"),
    ("42", "42_computational_event_horizon/BLACK_HOLES/exp_23_true_singularity/23_true_singularity_core_crush.py"),
    ("42", "42_computational_event_horizon/COSMOS/exp_24_dark_matter/24_dark_matter_orphaned_pointers.py"),
    ("42", "42_computational_event_horizon/COSMOS/exp_25_dark_energy/25_dark_energy_expansion.py"),
    ("42", "42_computational_event_horizon/COSMOS/exp_27_arrow_of_time/42_27_arrow_of_time.py"),
    ("42", "42_computational_event_horizon/ULTRA/exp_14_boltzmann_brain/rust/plot_entropy.py"),
    ("42", "42_computational_event_horizon/ULTRA/exp_15_quantum_gravity_unification/rust/unification_proof.py"),
    ("45", "45_phase_math/45_1_collatz_oracle/45_1_collatz_oracle.py"),
    ("47", "47_phase_atom/47_1_nucleus_memory_knot/47_1_nucleus_memory_knot.py"),
    ("47", "47_phase_atom/47_3_pauli_exclusion/47_3_pauli_exclusion.py"),
    ("47", "47_phase_atom/47_4_lhc_overflow_exploit/47_4_lhc_overflow_exploit.py"),
    ("47", "47_phase_atom/47_6_quark_confinement/47_6_quark_confinement.py"),
]

ROOT = "THOUGHT/LAB/CAT_CAS"

def classify_null_model(content):
    """Is the null model real (computation) or fake (text label)?"""
    # Real: actual function that computes something random/shuffled
    has_null_function = bool(re.search(r'def\s+(null|random|shuffle|permut|baseline)', content, re.I))
    # Real: permutation test with loop
    has_permutation = bool(re.search(r'for\s+.*\s+in\s+range.*shuffle|permutation|_random', content))
    # Real: computes random Hamiltonian/sequence and runs it
    has_null_computation = bool(re.search(r'random.*(hamilton|build|generate|shuffle|permut)', content, re.I))
    # Fake: just the word "BASELINE" or "NULL MODEL" in a comment
    has_text_only = bool(re.search(r'#.*(BASELINE|NULL MODEL)|BASELINE:', content))
    
    if has_null_function or has_permutation or has_null_computation:
        return "REAL"
    elif has_text_only and not has_null_computation:
        return "TEXT_ONLY"
    return "UNKNOWN"

def classify_statistics(content):
    """Are the statistics real (computation) or fake (text label)?"""
    # Real: actual std computation (np.std, statistics.stdev)
    has_std_comp = bool(re.search(r'np\.std|statistics\.stdev|\.std\(\)|_std\(', content))
    # Real: p-value, t-test, confidence interval
    has_stat_test = bool(re.search(r'ttest|p.value|p_value|p\s*<\s*0\.|confidence|bootstrap|cohens', content, re.I))
    # Real: mean computation with actual data
    has_mean_comp = bool(re.search(r'np\.mean|statistics\.mean|sum\(.*\)\s*/\s*len', content))
    # Fake: "std = 0.0" as text without computation
    has_std_text = bool(re.search(r'[Pp]rint.*std\s*=\s*0', content)) or bool(re.search(r'#.*std\s*=\s*0', content))
    # Fake: "STATISTICS:" header with deterministic claim but no computation
    has_deterministic_header = bool(re.search(r'\[STATISTICS\].*(?:deterministic|exact|invariant)', content))
    
    if has_std_comp or has_stat_test or has_mean_comp:
        return "REAL"
    elif has_std_text or has_deterministic_header:
        return "FAKE"
    return "UNKNOWN"

print("=" * 60)
print("SECTION E: NULL MODELS")
print("=" * 60)
for phase, rel_path in SECTION_E:
    path = os.path.join(ROOT, rel_path)
    try:
        with open(path, encoding='utf-8') as f:
            content = f.read()
    except:
        print(f"MISSING: {phase}/{rel_path}")
        continue
    status = classify_null_model(content)
    print(f"[{status:>10s}] E {phase}: {rel_path}")

print()
print("=" * 60)
print("SECTION F: STATISTICS")
print("=" * 60)
for phase, rel_path in SECTION_F:
    path = os.path.join(ROOT, rel_path)
    try:
        with open(path, encoding='utf-8') as f:
            content = f.read()
    except:
        print(f"MISSING: {phase}/{rel_path}")
        continue
    status = classify_statistics(content)
    print(f"[{status:>10s}] F {phase}: {rel_path}")
