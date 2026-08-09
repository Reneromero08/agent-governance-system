#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M262 central-Wilson obstruction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Mapping


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "finite_mtc_closed_probe_obstruction.py"
REFERENCE = PACKAGE / "tests" / "finite_mtc_closed_probe_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V4_FINITE_MTC_CLOSED_PROBE_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V4_FINITE_MTC_CLOSED_PROBE_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V4_FINITE_MTC_CLOSED_PROBE_DIAGNOSTIC.json"
REFERENCE_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V4_FINITE_MTC_CLOSED_PROBE_SEPARATE_REFERENCE.json"

CLAIM = (
    "EXACT_FIXED_FINITE_UMTC_SINGLE_GLOBAL_CLOSED_SIMPLE_PROBE_DIAGNOSTIC_"
    "ESTABLISHES_MULTIPLICITY_BLIND_TOTAL_CHARGE_SCALAR_ACTION_"
    "DETERMINISTIC_UNIT_MODULUS_BOUNDARIES_AS_CONSTANT_SIZE_SIMPLE_OBJECT_"
    "LOOKUPS_AND_STRICTLY_INTERMEDIATE_VACUUM_RETURN_RETAINED_BOUNDARY_"
    "OBSTRUCTION_WITH_FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_"
    "RESTORATION_DISTINCT_PROBE_REUSE_AND_SEMION_ISING_FIBONACCI_FIXTURES"
)
CEILING = (
    "ABSTRACT_EXACT_FIXED_FINITE_UMTC_SINGLE_SIMPLE_PROBE_GLOBAL_DISK_"
    "ENCIRCLEMENT_WITH_DECLARED_TOTAL_CHARGE_AND_SEMION_ISING_FIBONACCI_"
    "FIXTURES_ONLY"
)
RESTORATION_SCOPE = (
    "FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_AND_DISTINCT_"
    "PROBE_REUSE_WITHOUT_SAME_BACKING"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def coordinates(encoded: Mapping[str, object]) -> dict[int, Fraction]:
    require(encoded["basis"] == "Q_ZETA40_POWER_MOD_PHI40", "field basis changed")
    return {
        int(entry["power"]): Fraction(int(entry["numerator"]), int(entry["denominator"]))
        for entry in encoded["nonzero"]
    }


def radical_tuple(encoded: Mapping[str, object]) -> tuple[Fraction, Fraction, int]:
    return (
        Fraction(int(encoded["rational_numerator"]), int(encoded["rational_denominator"])),
        Fraction(int(encoded["radical_numerator"]), int(encoded["radical_denominator"])),
        int(encoded["radicand"]),
    )


def source_audit() -> None:
    production_text = PRODUCTION.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")
    forbidden_modules = {"numpy", "scipy", "sympy", "pickle"}
    for label, text in (("production", production_text), ("reference", reference_text)):
        tree = ast.parse(text)
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
                require(node.func.id not in {"float", "complex", "eval", "exec"}, f"{label} uses {node.func.id}")
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                require(len(node.elts) < 100, f"{label} contains a large literal table")
            elif isinstance(node, ast.Dict):
                require(len(node.keys) < 100, f"{label} contains a large literal mapping")
        require(not imports.intersection(forbidden_modules), f"{label} imports forbidden helper")

    reference_tree = ast.parse(reference_text)
    imported_names = {
        node.module
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_names.update(
        alias.name
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    require(
        not any("finite_mtc_closed_probe_obstruction" in name for name in imported_names),
        "reference imports production",
    )
    require("common_full_monodromy_scalar_across_supported_channels" in production_text, "channel-scalar semantics missing")
    require("retained_response_register_unchanged_through_inverse" in production_text, "retained response is not modeled")
    require("five_loop_q5" in reference_text, "reference five-loop derivation missing")
    require("FORMAL_DERIVATION_SOURCE_AUDITED" in CONTRACT.read_text(encoding="utf-8"), "formal theorem scope missing")
    for path in (CONTRACT, FINDINGS):
        text = path.read_text(encoding="utf-8")
        require(CLAIM in text or path == FINDINGS, f"claim missing from {path.name}")
        require(CEILING in text, f"ceiling missing from {path.name}")
        require("M257" in text, f"M257 ceiling missing from {path.name}")
        require("noncentral" in text.lower(), f"noncentral exclusion missing from {path.name}")


def fixture_parity(production: dict, reference: dict) -> None:
    calibration = production["calibrations"]
    semion = calibration["semion_s_s_single_loop"]
    require(coordinates(semion["normalized_vacuum_return_amplitude"]) == {0: Fraction(-1)}, "Semion amplitude changed")
    require(coordinates(semion["vacuum_return_probability"]) == {0: Fraction(1)}, "Semion probability changed")
    require(reference["semion"]["s_s_single_loop_amplitude"] == -1, "reference Semion amplitude changed")
    require(reference["semion"]["s_s_single_loop_probability"] == 1, "reference Semion probability changed")

    sigma_psi = calibration["ising_sigma_psi_single_loop"]
    require(coordinates(sigma_psi["normalized_vacuum_return_amplitude"]) == {0: Fraction(-1)}, "Ising sigma-psi amplitude changed")
    require(coordinates(sigma_psi["vacuum_return_probability"]) == {0: Fraction(1)}, "Ising sigma-psi probability changed")
    require(reference["ising"]["sigma_psi_single_loop_amplitude"] == {"denominator": 1, "numerator": -1}, "reference Ising sigma-psi changed")
    psi_psi = calibration["ising_psi_psi_single_loop"]
    require(coordinates(psi_psi["normalized_vacuum_return_amplitude"]) == {0: Fraction(1)}, "Ising psi-psi amplitude changed")
    require(coordinates(psi_psi["vacuum_return_probability"]) == {0: Fraction(1)}, "Ising psi-psi probability changed")
    require(reference["ising"]["psi_psi_single_loop_amplitude"] == {"denominator": 1, "numerator": 1}, "reference Ising psi-psi changed")
    sigma_sigma = calibration["ising_sigma_sigma_single_loop"]
    require(coordinates(sigma_sigma["normalized_vacuum_return_amplitude"]) == {}, "Ising sigma-sigma amplitude changed")
    require(coordinates(sigma_sigma["vacuum_return_probability"]) == {}, "Ising sigma-sigma probability changed")
    require(not sigma_sigma["common_full_monodromy_scalar_across_supported_channels"], "Ising channels unexpectedly aligned")
    reference_sigma_sigma = reference["ising"]["sigma_sigma_single_loop_amplitude"]
    require(radical_tuple(reference_sigma_sigma["real"]) == (Fraction(0), Fraction(0), 2), "reference Ising sigma-sigma real part changed")
    require(radical_tuple(reference_sigma_sigma["imag"]) == (Fraction(0), Fraction(0), 2), "reference Ising sigma-sigma imaginary part changed")
    require(reference["ising"]["sigma_sigma_single_loop_probability"] == {"denominator": 1, "numerator": 0}, "reference Ising probability changed")

    fibonacci = calibration["fibonacci_tau_tau_single_loop"]
    require(
        coordinates(fibonacci["normalized_vacuum_return_amplitude"])
        == {0: Fraction(-1), 8: Fraction(1), 12: Fraction(-1)},
        "Fibonacci amplitude changed",
    )
    require(
        radical_tuple(reference["fibonacci"]["single_loop_amplitude"])
        == (Fraction(-3, 2), Fraction(1, 2), 5),
        "reference Fibonacci amplitude changed",
    )
    require(
        radical_tuple(reference["fibonacci"]["single_loop_probability"])
        == (Fraction(7, 2), Fraction(-3, 2), 5),
        "reference Fibonacci probability changed",
    )
    require(
        coordinates(fibonacci["vacuum_return_probability"])
        == {0: Fraction(2), 8: Fraction(-3), 12: Fraction(3)},
        "production Fibonacci probability changed",
    )
    require(not fibonacci["common_full_monodromy_scalar_across_supported_channels"], "Fibonacci channels unexpectedly aligned")

    ising_two = production["repeated_loop_controls"]["ising_sigma_sigma_two_loops"]
    require(
        ising_two["deterministic_vacuum_return"],
        "Ising two-loop realignment failed",
    )
    require(coordinates(ising_two["normalized_vacuum_return_amplitude"]) == {10: Fraction(-1)}, "Ising two-loop common phase changed")
    require(coordinates(ising_two["vacuum_return_probability"]) == {0: Fraction(1)}, "Ising two-loop probability changed")
    reference_two_phase = reference["ising"]["sigma_sigma_two_loop_common_phase"]
    require(radical_tuple(reference_two_phase["real"]) == (Fraction(0), Fraction(0), 2), "reference Ising two-loop real part changed")
    require(radical_tuple(reference_two_phase["imag"]) == (Fraction(-1), Fraction(0), 2), "reference Ising two-loop imaginary part changed")
    require(reference["ising"]["sigma_sigma_two_loop_probability"] == {"denominator": 1, "numerator": 1}, "reference Ising two-loop probability changed")
    fibonacci_five = production["repeated_loop_controls"]["fibonacci_tau_tau_five_loops"]
    require(
        fibonacci_five["deterministic_vacuum_return"],
        "Fibonacci five-loop realignment failed",
    )
    require(coordinates(fibonacci_five["normalized_vacuum_return_amplitude"]) == {0: Fraction(1)}, "Fibonacci five-loop amplitude changed")
    require(coordinates(fibonacci_five["vacuum_return_probability"]) == {0: Fraction(1)}, "Fibonacci five-loop probability changed")
    require(
        reference["fibonacci"]["five_loop_amplitude_zeta5_coordinates"]
        == [{"denominator": 1, "numerator": 1, "power": 0}],
        "reference Fibonacci five-loop derivation changed",
    )
    require(radical_tuple(reference["fibonacci"]["five_loop_amplitude"]) == (Fraction(1), Fraction(0), 5), "reference Fibonacci five-loop amplitude changed")
    require(radical_tuple(reference["fibonacci"]["five_loop_probability"]) == (Fraction(1), Fraction(0), 5), "reference Fibonacci five-loop probability changed")

    ranks = {entry["category"]: entry["character_table_rank"] for entry in production["character_tables"]}
    require(ranks == {"SEMION": 2, "ISING": 3, "FIBONACCI": 2}, "character ranks changed")
    require(reference["semion"]["character_table_rank"] == 2, "reference Semion rank changed")
    require(reference["ising"]["character_table_rank"] == 3, "reference Ising rank changed")
    require(reference["fibonacci"]["character_table_rank"] == 2, "reference Fibonacci rank changed")


def retained_boundary_control(production: dict, reference: dict) -> None:
    control = production["strictly_intermediate_return_control"]
    probability = coordinates(control["vacuum_return_probability"])
    complement = coordinates(control["complementary_outcome_probability"])
    combined = dict(probability)
    for power, value in complement.items():
        combined[power] = combined.get(power, Fraction(0)) + value
    combined = {power: value for power, value in combined.items() if value}
    require(combined == {0: Fraction(1)}, "retained-outcome probabilities do not sum to one")
    require(control["both_exact_outcome_weights_nonzero"], "strictly intermediate weights lost")
    require(control["coherently_copied_which_outcome_schmidt_rank"] == 2, "retained-outcome rank changed")
    require(control["carrier_probe_only_inverse_cannot_erase_retained_orthogonal_response"], "retained record obstruction lost")
    require(not control["response_release_with_exact_factorized_restoration_authorized"], "nondeterministic response released")
    require(control["zero_or_unit_probability_endpoints_excluded"], "endpoint scope lost")
    fib = reference["fibonacci"]
    require(fib["coherently_copied_which_outcome_schmidt_rank"] == 2, "reference retained rank changed")
    require(fib["carrier_probe_only_inverse_cannot_erase_retained_orthogonal_response"], "reference retained obstruction lost")
    require(not fib["response_release_with_exact_factorized_restoration_authorized"], "reference nondeterministic release promoted")


def transaction_parity(production: dict, reference: dict) -> None:
    transactions = production["deterministic_transactions"]
    records = reference["deterministic_transactions"]["records"]
    require(transactions["dimensions"] == [1, 2, 4, 8, 16], "production dimensions changed")
    require([record["dimension"] for record in records] == [1, 2, 4, 8, 16], "reference dimensions changed")
    require(transactions["boundaries_generation_1"] == [1] * 5, "generation-one boundaries changed")
    require(transactions["boundaries_generation_2"] == [0] * 5, "generation-two boundaries changed")
    require(not transactions["second_preparation_used"], "production reprepared")
    require(not reference["deterministic_transactions"]["second_preparation_used"], "reference reprepared")
    for case, oracle in zip(transactions["cases"], records):
        require(case["internal_multiplicity_dimension"] == oracle["dimension"], "transaction dimension mismatch")
        require(case["preparation_count"] == 1, "transaction preparation count changed")
        require(case["generation_1"]["boundary_bit"] == oracle["generation_1_bit"] == 1, "generation-one parity mismatch")
        require(case["generation_2"]["boundary_bit"] == oracle["generation_2_bit"] == 0, "generation-two parity mismatch")
        for generation in ("generation_1", "generation_2"):
            item = case[generation]
            require(item["functional_exact_restoration"], f"{generation} did not restore")
            require(item["initial_commitment"] == item["restored_commitment"], f"{generation} commitment mismatch")
            require(item["retained_response_register_unchanged_through_inverse"], f"{generation} response changed")
            require(not item["same_backing_restoration_established"], f"{generation} same-backing overclaim")
        require(case["functional_exact_returned_value_reuse"], "functional reuse failed")
        require(not case["same_backing_reuse_established"], "same-backing reuse overclaim")
        require(oracle["functional_exact_restoration_and_reuse"], "reference transaction failed")
        require(not oracle["same_backing_established"], "reference same-backing overclaim")


def scopes_and_resources(production: dict, reference: dict) -> None:
    require(production["milestone"] == reference["milestone"] == "M262", "milestone changed")
    require(production["claim"] == reference["claim"] == CLAIM, "claim changed")
    require(production["claim_ceiling"] == reference["claim_ceiling"] == CEILING, "ceiling changed")
    require(
        production["verification_scope"]
        == {
            "fixture_arithmetic_dimensions_character_ranks_and_transactions": "SEPARATE_REFERENCE_PARITY",
            "general_fixed_umtc_centrality_and_equality_theorem": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource_accounting": "PACKAGE_SELF_REVIEW",
        },
        "verification scope changed",
    )
    theorem = production["theorem"]
    require(theorem["fixed_charge_internal_multiplicity_observable_rank"] == 1, "observable rank changed")
    require(theorem["declared_internal_multiplicity_dimensions_without_materialized_identity_matrices"] == [1, 2, 4, 8, 16], "unverified dimensions entered theorem")
    require(theorem["all_simple_boundary_wilson_loops_span_k_dimensional_charge_projector_algebra"], "central algebra theorem lost")
    require(reference["fusion_dimensions"]["global_loop_scalar_on_every_enumerated_internal_basis_vector"], "reference multiplicity centrality failed")
    require(reference["fusion_dimensions"]["ising_even_sigma_total_psi_dimensions"] == [1, 2, 4, 8, 16], "reference Ising dimensions changed")
    require(reference["fusion_dimensions"]["ising_odd_sigma_total_sigma_dimensions"] == [1, 2, 4, 8, 16], "reference odd-Ising dimensions changed")
    require(reference["fusion_dimensions"]["fibonacci_even_tau_total_1_dimensions"] == [1, 2, 5, 13, 34], "reference Fibonacci total-one dimensions changed")
    require(reference["fusion_dimensions"]["fibonacci_even_tau_total_tau_dimensions"] == [1, 3, 8, 21, 55], "reference Fibonacci total-tau dimensions changed")
    require(reference["fusion_dimensions"]["maximum_enumerated_fixed_charge_basis_vectors"] == 55, "reference path enumeration ceiling changed")
    require(reference["theorem_checks"]["prepared_noncentral_eigenstate_is_an_explicit_exception"], "prepared eigenstate exception lost")
    require(reference["theorem_checks"]["repeated_loops_can_realign_root_of_unity_channel_phases"], "repeated-loop caveat lost")

    rejected = set(production["scope_controls"]["rejected_descriptor_classes"])
    require(rejected == set(reference["scope_rejections"]), "scope rejection parity changed")
    require(not production["scope_controls"]["noncentral_two_eigenphase_control"]["reference_complete_uniform_factorization"], "single eigenstate promoted to uniform factorization")
    require(production["scope_controls"]["indefinite_charge_control"]["fixed_charge_hypothesis_required"], "definite-charge hypothesis lost")

    restoration = production["restoration"]
    require(restoration["classification"] == reference["restoration"]["classification"] == "EXACT_ALGEBRAIC_RESTORATION", "restoration class changed")
    require(restoration["scope"] == reference["restoration"]["scope"] == RESTORATION_SCOPE, "restoration scope changed")
    require(not restoration["same_backing_restoration_or_reuse_established"], "same-backing claim promoted")
    require(restoration["vacuum_return_probability_is_not_catalytic_restoration"], "vacuum return promoted to restoration")

    require(all(value is False for value in production["claim_limits"].values()), "a forbidden claim limit was promoted")
    comparator = production["strongest_classical_comparator"]
    require(comparator["fixed_category_definite_charge_query"].startswith("O1_"), "O(1) table comparator lost")
    require(not comparator["arbitrary_internal_many_anyon_state_represented_by_comparator"], "comparator scope overclaimed")
    require(not comparator["arbitrary_growing_probe_link_network_constant_work_established"], "growing link work hidden")
    resources = production["resource_law"]
    require(resources["accepted_transaction_max_logical_carrier_field_cells"] == 16, "logical carrier count changed")
    require(resources["accepted_transaction_two_path_logical_field_cells"] == 32, "logical path count changed")
    require(resources["resource_verification_level"] == "PACKAGE_SELF_REVIEW", "resource evidence level changed")
    require(not resources["whole_process_liveness_complete"], "whole-process liveness overclaimed")


def main() -> None:
    source_audit()
    generated_production = regenerate(PRODUCTION)
    generated_reference = regenerate(REFERENCE)
    require(PRODUCTION_SEAL.read_bytes() == generated_production, "production seal is stale")
    require(REFERENCE_SEAL.read_bytes() == generated_reference, "reference seal is stale")
    production = json.loads(generated_production)
    reference = json.loads(generated_reference)
    require(
        production["source_dependencies"] == {PRODUCTION.name: sha256(PRODUCTION)},
        "production source dependency hash changed",
    )
    require(
        reference["source_dependencies"] == {REFERENCE.name: sha256(REFERENCE)},
        "reference source dependency hash changed",
    )
    fixture_parity(production, reference)
    retained_boundary_control(production, reference)
    transaction_parity(production, reference)
    scopes_and_resources(production, reference)
    print(
        "PASS_STRICT_SCOPE M262_FIXED_FINITE_UMTC_GLOBAL_CLOSED_PROBE "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "THEOREM=FORMAL_DERIVATION_SOURCE_AUDITED "
        "RESTORATION=FUNCTIONAL_EXACT_PLUS_MINUS_ONE_NO_SAME_BACKING "
        "DISPOSITION=FINITE_TOTAL_CHARGE_LOOKUP_RESOURCE_KILL"
    )


if __name__ == "__main__":
    main()
