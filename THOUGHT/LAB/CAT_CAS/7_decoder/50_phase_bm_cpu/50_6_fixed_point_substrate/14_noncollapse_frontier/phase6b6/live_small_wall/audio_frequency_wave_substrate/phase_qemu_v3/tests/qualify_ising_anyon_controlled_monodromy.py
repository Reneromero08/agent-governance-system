#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M261 Ising-holonomy diagnostic."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "ising_anyon_controlled_monodromy_diagnostic.py"
REFERENCE = PACKAGE / "tests" / "ising_anyon_controlled_monodromy_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V3_ISING_MONODROMY_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V3_ISING_MONODROMY_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V3_ISING_MONODROMY_DIAGNOSTIC.json"
REFERENCE_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V3_ISING_MONODROMY_SEPARATE_REFERENCE.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def regenerate(script: Path) -> bytes:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=PACKAGE,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.stderr == b"", f"unexpected stderr from {script.name}")
    return result.stdout


def source_audit() -> None:
    production_text = PRODUCTION.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")
    production_tree = ast.parse(production_text)
    reference_tree = ast.parse(reference_text)
    forbidden_modules = {"numpy", "scipy", "sympy", "pickle"}

    for label, tree in (("production", production_tree), ("reference", reference_tree)):
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            require(
                not (isinstance(node, ast.Constant) and isinstance(node.value, float)),
                f"{label} contains a floating scientific literal",
            )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(node.func.id not in {"float", "eval", "exec"}, f"{label} uses {node.func.id}")
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                require(len(node.elts) < 100, f"{label} contains a large literal table")
            elif isinstance(node, ast.Dict):
                require(len(node.keys) < 100, f"{label} contains a large literal mapping")
        require(not imports.intersection(forbidden_modules), f"{label} imports forbidden helper")

    reference_imports = {
        node.module
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    reference_imports.update(
        alias.name
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    require(
        not any("ising_anyon_controlled_monodromy_diagnostic" in name for name in reference_imports),
        "reference imports production",
    )
    require("public_triangular_word" in production_text, "production triangular compiler missing")
    require("signed_pairing_after_word" in production_text, "production compact comparator missing")
    require("covariance_word" in reference_text, "reference covariance comparator missing")
    require("def cut_rank" in reference_text, "reference dense rank oracle missing")
    require("apply_pair_holonomy" in production_text, "production coherent probe action missing")
    require("BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY" in production_text, "semantic ceiling missing")

    contract = CONTRACT.read_text(encoding="utf-8")
    findings = FINDINGS.read_text(encoding="utf-8")
    for text, label in ((contract, "contract"), (findings, "findings")):
        require("CONTIGUOUS_CUT_RANKS2_4_8" in text, f"{label} claim missing")
        require("FUNCTIONAL_EXACT" in text, f"{label} restoration scope missing")
        require("M257" in text, f"{label} M257 ceiling missing")
        lowered = text.lower()
        require(
            "same-backing" in lowered or "same_backing" in lowered,
            f"{label} same-backing ceiling missing",
        )


def n4_smoke(production: dict, reference: dict) -> None:
    smoke = production["smoke_n4"]
    oracle = reference["n4_algebra_orientation_smoke"]
    require(smoke["sigma_anyons"] == 4 == oracle["sigma_anyons"], "n4 size changed")
    require(smoke["fixed_sector_dimension"] == 2 == oracle["even_sector_fusion_dimension"], "n4 dimension changed")
    require(smoke["public_smoke_excitation_word"] == [2, 2], "n4 production word changed")
    require(oracle["preparation"]["public_C_word_one_based"] == [2, 2], "n4 oracle word changed")
    require(smoke["vacuum_loop_eigenvalue"] == 1, "n4 vacuum phase changed")
    require(smoke["excited_loop_eigenvalue"] == -1, "n4 excited phase changed")
    require(oracle["vacuum"]["loop_phase"] == "+1", "n4 oracle vacuum phase changed")
    require(oracle["prepared"]["loop_phase"] == "-1", "n4 oracle prepared phase changed")
    require(smoke["public_adjoint_excitation_word_exactly_restores_vacuum"], "n4 production restore failed")
    require(oracle["public_adjoint_twice_restores_vacuum_exactly"], "n4 oracle restore failed")
    require(not oracle["included_in_triangular_scaling_family"], "n4 was promoted into triangular scaling")
    require(not smoke["scaling_cross_cut_loop_formula_claimed"], "n4 cross-cut formula overclaimed")
    for key in ("vacuum_retained_copy_transaction", "excited_retained_copy_transaction"):
        require(smoke[key]["exact_functional_restoration"], f"n4 {key} failed restore")
        require(smoke[key]["response_released_after_restoration_only"], f"n4 {key} failed release")


def scaling_parity(production: dict, reference: dict) -> None:
    expected = {
        8: {"dimension": 8, "allocated": 16, "support": 2, "rank": 2, "word": [4]},
        12: {"dimension": 32, "allocated": 64, "support": 4, "rank": 4, "word": [4, 6, 5, 8, 7, 6]},
        16: {
            "dimension": 128,
            "allocated": 256,
            "support": 8,
            "rank": 8,
            "word": [4, 6, 5, 8, 7, 6, 10, 9, 8, 7, 12, 11, 10, 9, 8],
        },
    }
    cases = {case["sigma_anyons"]: case for case in production["scaling_fixtures"]}
    require(set(cases) == set(expected), "production scaling cases changed")
    require(set(reference["cases"]) == {str(value) for value in expected}, "reference cases changed")

    for anyons, law in expected.items():
        case = cases[anyons]
        oracle = reference["cases"][str(anyons)]
        require(case["fixed_sector_dimension"] == law["dimension"] == oracle["fusion_dimension"], "fusion dimension mismatch")
        require(case["allocated_full_occupation_cells"] == law["allocated"], "allocated carrier cells mismatch")
        require(case["carrier_support_cells"] == law["support"] == oracle["prepared_support_cells"], "support mismatch")
        require(case["public_triangular_execution_word"] == law["word"], "production word mismatch")
        require(oracle["public_preparation_C_word_one_based"] == law["word"], "reference word mismatch")
        require(
            case["contiguous_cut"]["exact_amplitude_flattening_rank"]
            == law["rank"]
            == oracle["ordinary_contiguous_central_cut"]["dense_exact_schmidt_rank"],
            "dense contiguous rank mismatch",
        )
        require(
            oracle["ordinary_contiguous_central_cut"]["compact_majorana_covariance_rank"] == law["rank"],
            "covariance rank mismatch",
        )
        require(case["transported_holonomies"]["L0"]["exact_eigenvalue"] == 1, "L0 phase changed")
        require(case["transported_holonomies"]["L1"]["exact_eigenvalue"] == -1, "L1 phase changed")
        require(case["generation1_L0"]["boundary_bit"] == 0, "L0 boundary changed")
        require(case["generation2_distinct_L1"]["boundary_bit"] == 1, "L1 boundary changed")
        require(oracle["definite_loop_queries"]["L0_charge_1"]["probe_x_boundary"] == "1", "oracle L0 changed")
        require(oracle["definite_loop_queries"]["L1_charge_psi"]["probe_x_boundary"] == "-1", "oracle L1 changed")
        for key in ("generation1_L0", "generation2_distinct_L1"):
            require(case[key]["exact_functional_restoration"], f"{key} did not restore")
            require(case[key]["response_released_after_restoration_only"], f"{key} response order failed")
            require(not case[key]["same_backing_restoration_established"], f"{key} same-backing overclaim")
        reuse = case["functional_returned_value_reuse"]
        require(reuse["same_returned_value_consumed_by_generation2"], "returned value was not reused")
        require(reuse["exact_value_restored_twice"], "generation-two restore failed")
        require(not reuse["second_prepare_used"], "generation-two reprepared")
        require(not reuse["same_backing_reuse_established"], "same-backing reuse overclaimed")
        require(oracle["distinct_loop_reuse"]["same_prepared_value_consumed_by_both_queries"], "oracle reuse failed")
        require(not oracle["distinct_loop_reuse"]["same_backing_or_machine_custody_established"], "oracle custody overclaim")

        mixed = case["mixed_loop_control"]
        retained = mixed["retained_copy_transaction"]
        require((retained["forward_port_even_weight"], retained["forward_port_odd_weight"]) == ("1/2", "1/2"), "mixed weights changed")
        require(retained["restored_form_fidelities"] == {"0": "1/4", "1": "1/4"}, "mixed fixed-bit fidelities changed")
        require(not retained["exact_functional_restoration"], "mixed copy unexpectedly restored")
        require(not retained["response_released_after_restoration_only"], "mixed copy leaked response")
        require(mixed["maximum_factorized_fidelity_allowing_arbitrary_ancilla"] == "1/2", "mixed factorization ceiling changed")
        require(mixed["result_free_exact_restore"], "result-free mixed unwind failed")
        oracle_mixed = oracle["invalid_superposed_loop_control"]
        require(oracle_mixed["charge_1_probability"] == "1/2" == oracle_mixed["charge_psi_probability"], "oracle mixed weights changed")
        require(oracle_mixed["probe_target_schmidt_rank"] == 2, "oracle mixed rank changed")
        require(not oracle_mixed["factorized_boundary_retention_lawful"], "oracle mixed boundary promoted")
        require(oracle_mixed["no_copy_latch_unlatch_exact_restore"], "oracle no-copy restore failed")

        scramble = case["same_sector_scramble_control"]
        oracle_scramble = oracle["internal_fixed_charge_scrambling"]
        require(scramble["changes_projective_carrier"], "scramble did not change carrier")
        require(scramble["carrier_overlap_squared"] == "1/2", "scramble overlap changed")
        require(scramble["preserves_L0_boundary"] and scramble["preserves_L1_boundary"], "scramble changed boundary")
        require(scramble["public_adjoint_exactly_restores"], "scramble adjoint failed")
        require(scramble["direct_and_synthesized_exact_state_agree_up_to_sign"], "scramble synthesis failed")
        require(oracle_scramble["prepared_scrambled_overlap_squared"] == "1/2", "oracle scramble overlap changed")
        require(oracle_scramble["both_fixed_charges_invariant"], "oracle scramble changed charges")
        require(oracle_scramble["public_adjoint_restores_prepared_state_exactly"], "oracle scramble restore failed")

        require(not case["missing_inverse_control"]["response_released_after_restoration_only"], "missing inverse released")
        require(not case["wrong_inverse_loop_control"]["response_released_after_restoration_only"], "wrong inverse released")
        require(case["framing_control"]["descriptor_rejected"], "framing control accepted")
        require(not case["path_dephasing_control"]["deterministic_boundary_survives"], "dephasing control failed")
        require(not case["path_dephasing_control"]["density_matrix_materialized"], "analytic dephasing control overclaimed")
        require(case["signed_pairing_comparator"]["matches_exact_amplitude_transactions"], "pairing comparator mismatch")


def ceilings(production: dict, reference: dict) -> None:
    require(production["milestone"] == "M261", "milestone changed")
    require(
        production["verification_scope"]
        == {
            "algebra_ranks_holonomy_boundaries": "SEPARATE_REFERENCE_PARITY",
            "path_dephasing_control": "ANALYTIC_EXACT_NOT_DENSITY_MATRIX_EXECUTED",
            "production_transaction_response_order": "PACKAGE_SELF_REVIEW_SOURCE_AUDITED",
        },
        "verification scope changed",
    )
    require(
        production["claim_ceiling"]
        == "IDEAL_DETERMINISTIC_EXACT_SOFTWARE_ISING_MTC_CONTROLLED_HOLONOMY_DIAGNOSTIC_AT_N4_N8_N12_N16_ONLY",
        "production claim ceiling changed",
    )
    require(production["restoration"]["classification"] == "EXACT_ALGEBRAIC_RESTORATION", "restoration class changed")
    require(
        production["restoration"]["scope"]
        == "FUNCTIONAL_EXACT_VALUE_RESTORATION_AND_REUSE_WITHOUT_SAME_BACKING",
        "restoration scope changed",
    )
    require(not production["restoration"]["same_backing_established"], "same-backing claim promoted")
    require(not production["restoration"]["physical_restoration_established"], "physical restore promoted")
    require(production["observed_disposition"]["kill_as_computational_resource"], "resource route not killed")
    require(production["observed_disposition"]["promote_as_machine_law_calibration_only"], "calibration scope lost")
    require(production["observed_disposition"]["m257_controls_deterministic_software_comparison"], "M257 ceiling lost")
    comparator = production["strongest_honest_comparator"]
    require(not comparator["resource_advantage_comparison_authorized"], "advantage comparison promoted")
    require(not comparator["comparator_optimality_established"], "comparator optimality overclaimed")
    require(comparator["equal_access_forward_only_shadow_omits_positive_cost_inverse"], "M257 shadow lost")
    resources = production["resource_accounting"]
    require(resources["allocated_full_occupation_cells_n4_n8_n12_n16"] == [4, 16, 64, 256], "carrier cells changed")
    require(resources["allocated_transaction_joint_coefficient_cells_n4_n8_n12_n16"] == [16, 64, 256, 1024], "joint cells changed")
    require(resources["allocated_rank_matrix_gaussian_cells_n4_n8_n12_n16"] == [4, 16, 64, 256], "rank cells changed")
    require(resources["preparation_braid_counts_n4_n8_n12_n16"] == [2, 1, 6, 15], "braid counts changed")
    require(resources["retained_dynamic_inverse_history"] == 0, "inverse history changed")
    require(resources["verification_level"] == "PACKAGE_SELF_REVIEW", "resource verification changed")
    require(not resources["functional_immutable_value_allocations_counted_completely"], "allocation accounting overclaimed")
    require(not resources["python_objects_hashing_serialization_whole_process_peak_complete"], "whole-process accounting overclaimed")
    require(not resources["physical_preparation_energy_gap_noise_precision_bandwidth_latency_modeled"], "physical resources overclaimed")
    require(len(production["strict_claim_ceilings"]) == 13, "strict ceiling set changed")
    require(all(value is False for value in reference["claim_limits"].values()), "reference claim limit promoted")
    require(reference["verification_classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE", "reference classification changed")
    require(reference["verification_level"] == "SEPARATE_REFERENCE_PARITY", "reference level changed")
    require(reference["restoration_classification"] == "EXACT_ALGEBRAIC_RESTORATION", "reference restore class changed")
    require(not reference["implementation_independence"]["imports_production"], "reference import flag changed")
    require(not reference["implementation_independence"]["floating_point_scientific_decisions"], "reference float flag changed")
    require(reference["strongest_classical_comparator"]["reproduces_every_accepted_dense_boundary_and_rank"], "reference comparator mismatch")


def main() -> int:
    source_audit()
    fresh_production = regenerate(PRODUCTION)
    fresh_reference = regenerate(REFERENCE)
    require(fresh_production == PRODUCTION_SEAL.read_bytes(), "production seal is stale")
    require(fresh_reference == REFERENCE_SEAL.read_bytes(), "reference seal is stale")
    production = json.loads(fresh_production)
    reference = json.loads(fresh_reference)
    n4_smoke(production, reference)
    scaling_parity(production, reference)
    ceilings(production, reference)
    print(
        "PASS_STRICT_SCOPE M261_ISING_TRANSPORTED_HOLONOMY "
        "SCIENTIFIC_ALGEBRA=SEPARATE_REFERENCE_PARITY "
        "TRANSACTION_ORDER=PACKAGE_SELF_REVIEW_SOURCE_AUDITED "
        "RESTORATION=FUNCTIONAL_EXACT_NO_SAME_BACKING "
        "DISPOSITION=COMPACT_SIGNED_PAIRING_RESOURCE_KILL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
