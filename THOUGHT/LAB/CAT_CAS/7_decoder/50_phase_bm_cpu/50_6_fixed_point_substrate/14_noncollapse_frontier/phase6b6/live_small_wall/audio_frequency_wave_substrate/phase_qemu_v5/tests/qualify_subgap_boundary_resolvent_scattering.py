#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M263 subgap-resolvent diagnostic."""

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
PRODUCTION = PACKAGE / "subgap_boundary_resolvent_scattering.py"
REFERENCE = PACKAGE / "tests" / "subgap_boundary_resolvent_scattering_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V5_SUBGAP_RESOLVENT_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V5_SUBGAP_RESOLVENT_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V5_SUBGAP_BOUNDARY_RESOLVENT_DIAGNOSTIC.json"
REFERENCE_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V5_SUBGAP_BOUNDARY_RESOLVENT_SEPARATE_REFERENCE.json"

CLAIM = (
    "EXACT_RATIONAL_SINGLE_CHANNEL_SUBGAP_CAYLEY_RESOLVENT_DIAGNOSTIC_"
    "IMPLEMENTS_A_STIPULATED_FORMAL_STATIONARY_UNIT_MODULUS_BOUNDARY_LAW_"
    "AT_DECLARED_FIXTURES_WITH_DISTINCT_ENERGY_DESCRIPTOR_REUSE_GROWING_"
    "EXACT_KRYLOV_RANK_AND_PATH_ONLY_FIXED_MARGIN_EFFECTIVE_DEPTH_BOUND_"
    "PLUS_TILTED_FIELD_AND_BETHE_FACTORIZATION_CONTROLS"
)
CEILING = (
    "EXACT_DETERMINISTIC_SOFTWARE_FINITE_DIMENSIONAL_RATIONAL_ONE_CHANNEL_"
    "K_MATRIX_BOUNDARY_MODEL_WITH_FORMAL_STATIONARY_ASYMPTOTIC_RETURN_ONLY"
)
DISPOSITION = (
    "GROWING_EXACT_KRYLOV_RANK_ALONE_IS_NOT_AN_APPROXIMATION_LOWER_BOUND_"
    "PATH_FIXED_MARGIN_HAS_COMPACT_STREAMED_SHADOW_AND_BETHE_FACTORIZED_"
    "EIGENPHASE_IS_PUBLIC_RAPIDITY_PRODUCT_NEAR_THRESHOLD_TIME_DOMAIN_"
    "QUALIFICATION_REQUIRED"
)
NEXT = (
    "NEAR_THRESHOLD_NONINTEGRABLE_BOUNDARY_RESOLVENT_WITH_EXPLICIT_WIGNER_"
    "DELAY_FINITE_BANDWIDTH_PRECISION_PREPARATION_AMORTIZATION_AND_TENSOR_"
    "NETWORK_RESOURCE_CROSSOVER"
)
EXPECTED_HASHES = {
    PRODUCTION: "e0ae1309ec17b3db4943607c12a5fa3338b5a120ba9fcf72e55133ad2286b67f",
    REFERENCE: "5f315cfb03117f967ff7cd98d6374631d052117c29e4c7206d840f10f72f3235",
    CONTRACT: "5f42d99316d525cd339b4f26a17233ab706c343be5135b7a256ca2d11d74be6f",
    FINDINGS: "75168e25fd0748012971139b7ce2b4dc161dfd66f0dc39410968094a318c88a1",
}

RESTORATION_SCOPE = (
    "STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_"
    "DESCRIPTOR_REUSE_ONLY"
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
        [sys.executable, "-B", str(script), "--compact"],
        cwd=PACKAGE,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.stderr == b"", f"unexpected stderr from {script.name}")
    return result.stdout


def fraction(encoded: Mapping[str, object]) -> Fraction:
    return Fraction(int(encoded["numerator"]), int(encoded["denominator"]))


def complex_pair(encoded: Mapping[str, object]) -> tuple[Fraction, Fraction]:
    return fraction(encoded["real"]), fraction(encoded["imag"])


def source_and_document_audit() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(sha256(path) == expected, f"source dependency changed: {path.name}")

    forbidden_modules = {"numpy", "scipy", "sympy", "pickle", "decimal"}
    for label, path in (("production", PRODUCTION), ("reference", REFERENCE)):
        text = path.read_text(encoding="utf-8")
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
                require(
                    node.func.id not in {"float", "complex", "eval", "exec"},
                    f"{label} uses {node.func.id}",
                )
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                require(len(node.elts) < 128, f"{label} contains a large literal table")
            elif isinstance(node, ast.Dict):
                require(len(node.keys) < 128, f"{label} contains a large literal mapping")
        require(not imports.intersection(forbidden_modules), f"{label} imports forbidden helper")

    reference_text = REFERENCE.read_text(encoding="utf-8")
    require("subgap_boundary_resolvent_scattering.py" not in reference_text, "reference names production")
    require("import subgap_boundary_resolvent_scattering" not in reference_text, "reference imports production")

    for path in (CONTRACT, FINDINGS):
        text = path.read_text(encoding="utf-8")
        require(CLAIM in text, f"claim missing from {path.name}")
        require(CEILING in text, f"ceiling missing from {path.name}")
        require(DISPOSITION in text, f"disposition missing from {path.name}")
        require(NEXT in text, f"successor missing from {path.name}")
        require("M257" in text, f"M257 guardrail missing from {path.name}")
        require("physical restoration" in text.lower(), f"physical-restoration ceiling missing from {path.name}")
    findings_text = FINDINGS.read_text(encoding="utf-8")
    normalized_findings = " ".join(findings_text.split())
    require("NO_RESTORATION_CLAIM" in findings_text, "no-restoration classification missing")
    require("EXACT_ALGEBRAIC_RESTORATION" not in findings_text, "executed restoration overclaim returned")
    require("Only the declared path family has an executed compact streamed comparator" in normalized_findings, "path-only compact comparator scope missing")
    require("current recurrence vector may still occupy the full target-sector representation" in normalized_findings, "generic recurrence-state caveat missing")
    require("does not show that a generic interacting boundary can avoid dense state or work" in normalized_findings, "generic interacting work ceiling missing")


def metadata_and_scope(production: dict, reference: dict) -> None:
    metadata = production["metadata"]
    require(metadata["milestone"] == reference["metadata"]["milestone"] == "M263", "milestone changed")
    require(metadata["claim"] == CLAIM, "claim changed")
    require(reference["metadata"]["claim"] == CLAIM, "reference claim changed")
    require(metadata["claim_ceiling"] == CEILING, "ceiling changed")
    require(reference["metadata"]["claim_ceiling"] == CEILING, "reference ceiling changed")
    require(metadata["disposition"] == DISPOSITION, "disposition changed")
    require(reference["metadata"]["disposition"] == DISPOSITION, "reference disposition changed")
    require(metadata["next_mechanism"] == NEXT, "successor changed")
    require(reference["metadata"]["next_mechanism"] == NEXT, "reference successor changed")
    require(metadata["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "production self hash changed")
    require(reference["metadata"]["source_sha256"] == EXPECTED_HASHES[REFERENCE], "reference self hash changed")
    require(not reference["metadata"]["production_imported"], "reference imports production")
    require(reference["metadata"]["split_primes"] == [65537, 998244353], "reference primes changed")
    require(metadata["m257_guardrail"].startswith("PRESERVED_"), "M257 guardrail changed")
    require(
        production["verification_scope"]
        == {
            "science": "SEPARATE_REFERENCE_PARITY",
            "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource": "PACKAGE_SELF_REVIEW",
        },
        "production verification scope changed",
    )
    require(
        reference["verification_scope"]
        == {
            "scientific": "SEPARATE_REFERENCE_PARITY",
            "formal_derivation": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource_accounting": "PACKAGE_SELF_REVIEW",
        },
        "reference verification scope changed",
    )

    for evidence, label in ((production, "production"), (reference, "reference")):
        restoration = evidence["restoration"]
        require(restoration["classification"] == "NO_RESTORATION_CLAIM", f"{label} restoration class changed")
        require(restoration["scope"] == RESTORATION_SCOPE, f"{label} restoration scope changed")
        require(not restoration["same_backing_established"], f"{label} same-backing promoted")
        require(not restoration["physical_restoration_established"], f"{label} physical restoration promoted")

    exclusions = set(production["scope_exclusions"])
    required = {
        "NO_QEMU_DEVICE_EXECUTION",
        "NO_TIME_DOMAIN_SCATTERING",
        "NO_EXECUTED_INVERSE_OR_ECHO",
        "NO_PHYSICAL_CARRIER_OR_OBSERVATION",
        "NO_SAME_BACKING_CUSTODY",
        "NO_PHYSICAL_OR_EXACT_FINITE_TIME_RESTORATION",
        "NO_ADVANTAGE_OR_M257_ESCAPE",
        "NO_SMALL_WALL_CROSSING_OR_UNBOUNDED_COMPUTE",
        "NO_REPLACE_THE_BIT_WITH_PI_CLAIM",
    }
    require(exclusions == required, "scope exclusions changed")
    resources = reference["resource_accounting"]
    for key in (
        "classical_advantage_established",
        "m257_escape_established",
        "physical_resource_advantage_established",
        "replace_the_bit_with_pi_established",
        "unbounded_compute_established",
    ):
        require(not resources[key], f"reference {key} promoted")
    for key in (
        "boundary_krylov_quotient_is_controlling_exact_representation",
        "fixed_margin_effective_depth_is_precision_and_latency_dependent",
        "integrability_comparator_included",
    ):
        require(resources[key], f"reference {key} lost")


def fixture_parity(production: dict, reference: dict) -> None:
    expected = [
        ("ONE_MODE", 1, Fraction(-1, 3), (Fraction(4, 5), Fraction(3, 5))),
        ("COUPLED_TWO_MODE", 2, Fraction(-3, 8), (Fraction(55, 73), Fraction(48, 73))),
        ("TRIDIAGONAL_THREE_MODE", 3, Fraction(-15, 56), (Fraction(2911, 3361), Fraction(1680, 3361))),
    ]
    require(len(production["exact_mode_fixtures"]) == len(reference["exact_mode_fixtures"]) == 3, "fixture count changed")
    for case, oracle, (name, dimension, green, phase) in zip(
        production["exact_mode_fixtures"], reference["exact_mode_fixtures"], expected
    ):
        require(case["name"] == oracle["name"] == name, "fixture name mismatch")
        require(case["dimension"] == oracle["dimension"] == dimension, "fixture dimension mismatch")
        require(fraction(case["green"]) == fraction(oracle["green"]) == green, "fixture Green mismatch")
        require(complex_pair(case["cayley_s"]) == complex_pair(oracle["cayley_s"]) == phase, "fixture phase mismatch")
        require(case["inverse_product_exact_one"], "production inverse product failed")
        require(case["cayley_s"]["norm_squared"] == {"numerator": 1, "denominator": 1}, "production norm changed")
        require(oracle["cayley_unit_modulus_exact"], "reference phase norm failed")
        require(oracle["all_split_prime_ranks_match_exact"], "reference fixture ranks failed")

    case = production["nonunit_kappa_control"]
    oracle = reference["non_unit_kappa_control"]
    require(fraction(case["green"]) == fraction(oracle["green"]) == Fraction(-1, 3), "nonunit-kappa Green changed")
    require(fraction(case["kappa"]) == fraction(oracle["kappa"]) == Fraction(2), "nonunit kappa changed")
    require(complex_pair(case["cayley_s"]) == complex_pair(oracle["cayley_s"]) == (Fraction(5, 13), Fraction(12, 13)), "linear-kappa Cayley control failed")
    require(case["linear_kappa_green_rule_exact"] and oracle["linear_kappa_recurrence_exposed"], "linear-kappa rule not checked")


def path_and_interacting_parity(production: dict, reference: dict) -> None:
    expected_path_sizes = [2, 4, 8, 16, 32]
    require([entry["n"] for entry in production["path_family"]] == expected_path_sizes, "production path sizes changed")
    require([entry["n"] for entry in reference["path_family"]] == expected_path_sizes, "reference path sizes changed")
    for case, oracle, size in zip(production["path_family"], reference["path_family"], expected_path_sizes):
        require(fraction(case["green"]) == fraction(oracle["green"]), "path Green parity failed")
        require(fraction(case["continuant_green"]) == fraction(oracle["continuant_green"]), "path continuant parity failed")
        require(complex_pair(case["cayley_s"]) == complex_pair(oracle["cayley_s"]), "path phase parity failed")
        require(case["krylov_rank_certified_exact"] == oracle["krylov_rank_certified_exact"] == size, "path rank changed")
        require(case["krylov_rank_by_prime"] == oracle["krylov_rank_by_prime"] == {"65537": size, "998244353": size}, "path modular ranks changed")
        require(case["continuant_lanczos_dense_parity"], "production path parity failed")
        require(oracle["all_split_prime_ranks_match_exact"], "reference path parity failed")

    expected_dimensions = [4, 8, 16, 32, 64]
    for case, oracle, dimension in zip(
        production["interacting_flagged_blocks"], reference["interacting_flagged_blocks"], expected_dimensions
    ):
        require(case["dimension"] == oracle["dimension"] == dimension, "interacting dimension changed")
        require(case["krylov_rank_certified_exact"] == oracle["krylov_rank_certified_exact"] == dimension, "interacting rank changed")
        require(case["krylov_rank_by_prime"] == oracle["krylov_rank_by_prime"] == {"65537": dimension, "998244353": dimension}, "interacting modular rank changed")
        require(fraction(case["operator_norm_upper_bound_j"]) == fraction(oracle["coefficient_bound_J"]), "interaction bound mismatch")
        require(fraction(case["positive_offset_d"]) == fraction(oracle["diagonal_offset_D"]), "interaction offset mismatch")
        require(oracle["all_split_prime_ranks_match_exact"], "reference interacting certificate failed")
        if case["n"] <= 4:
            require(fraction(case["green"]) == fraction(oracle["green"]), "interacting exact Green mismatch")
            require(case["exact_resolvent_materialized"], "small interacting resolvent missing")
        else:
            require(case["green"] is None and not case["exact_resolvent_materialized"], "production materialization ceiling changed")
            require(oracle["green"] is None and oracle["resolvent_seal_sha256"] is None, "reference materialization ceiling changed")
        require(case["approximation_work_claim"] == "NONE", "interacting approximation-work overclaim returned")


def approximation_and_bandwidth_parity(production: dict, reference: dict) -> None:
    case = production["fixed_margin_neumann"]
    oracle = reference["fixed_margin_neumann"]
    require(case["smallest_truncation_order_k"] == oracle["minimal_truncation_order_K"] == 19, "Neumann order changed")
    require(case["retained_moment_count"] == 20, "moment count changed")
    require(fraction(case["q"]) == fraction(oracle["uniform_ratio_q"]) == Fraction(1, 2), "Neumann ratio changed")
    require(fraction(case["epsilon"]) == fraction(oracle["epsilon"]) == Fraction(1, 1 << 20), "epsilon changed")
    require(fraction(case["phase_tail_bound"]) == fraction(oracle["phase_tail_bound"]) == Fraction(1, 1 << 20), "tail bound changed")
    require(fraction(oracle["previous_order_phase_tail_bound"]) > fraction(oracle["epsilon"]), "minimality control failed")
    require(all(record["within_epsilon"] for record in oracle["records"]), "reference truncation failed")

    case_band = production["finite_bandwidth_control"]
    oracle_band = reference["finite_bandwidth_control"]
    require(case_band["energies"] == oracle_band["energies"], "bandwidth energies changed")
    require(
        [complex_pair(value) for value in case_band["cayley_phases"]]
        == [complex_pair(value) for value in oracle_band["cayley_phases"]],
        "bandwidth phase parity failed",
    )
    require(fraction(case_band["same_spectral_mode_probability"]) == fraction(oracle_band["same_spectral_mode_probability"]) == Fraction(39204, 39493), "mode fidelity changed")
    require(fraction(case_band["orthogonal_spectral_distortion_probability"]) == fraction(oracle_band["orthogonal_spectral_distortion_probability"]) == Fraction(289, 39493), "distortion changed")
    require(oracle_band["strictly_less_than_one"], "reference distortion control failed")


def bethe_and_stationary_scope(production: dict, reference: dict) -> None:
    expected_counts = [(2, 1), (4, 6), (8, 28)]
    for case, oracle, expected in zip(
        production["bethe_factorized_control"], reference["bethe_factorized_control"], expected_counts
    ):
        require((case["m"], case["pair_count"]) == (oracle["m"], oracle["pair_count"]) == expected, "Bethe count changed")
        require(case["rapidities"] == oracle["rapidities"], "Bethe rapidities changed")
        require(complex_pair(case["factorized_phase"]) == complex_pair(oracle["factorized_phase"]), "Bethe phase parity failed")
        require(case["exact_inverse_product_one"] and oracle["exact_inverse_product_one"], "Bethe inverse failed")
        require(oracle["classical_pair_product_work"] == expected[1], "Bethe comparator work changed")

    case = production["stationary_reuse_semantics"]
    oracle = reference["stationary_reuse_semantics"]
    require(case["classification"] == oracle["classification"] == RESTORATION_SCOPE, "stationary classification changed")
    require(case["distinct_energy"] and oracle["distinct_energy"], "distinct-energy control failed")
    require(case["second_query_uses_no_new_target_descriptor"] and oracle["second_query_uses_no_new_target_descriptor"], "descriptor reuse changed")
    require(len(case["transactions"]) == len(oracle["transactions"]) == 2, "formal query count changed")
    for transaction, reference_transaction in zip(case["transactions"], oracle["transactions"]):
        require(transaction["query"] == reference_transaction["query"], "formal query index changed")
        require(fraction(transaction["energy"]) == fraction(reference_transaction["energy"]), "formal energy mismatch")
        require(fraction(transaction["green"]) == fraction(reference_transaction["green"]), "formal reuse Green mismatch")
        require(complex_pair(transaction["phase"]) == complex_pair(reference_transaction["phase"]), "formal reuse phase mismatch")
    require(not case["same_backing_established"], "same-backing claim promoted")
    require(not case["executed_time_domain_restoration"], "time-domain restoration promoted")
    require(not case["physical_restoration_established"], "physical restoration promoted")
    for key in (
        "distinct_probe_reuse_executed",
        "finite_time_target_restoration_executed",
        "physical_ground_state_prepared",
        "physical_source_separation_executed",
        "same_backing_established",
        "time_domain_scattering_executed",
    ):
        require(not oracle[key], f"reference {key} promoted")

    resources = production["resource_accounting"]
    require(resources["classification"] == "PACKAGE_SELF_REVIEW", "resource verification level changed")
    require(resources["exact_dense_resolvent_limit"] == "INTERACTING_N_LE_4", "dense resolvent ceiling changed")
    require(resources["coefficient_payload_bits_scope"].startswith("INPUT_MATERIALIZED_"), "coefficient payload scope missing")
    required_uninstrumented = {
        "WHOLE_PROCESS_LIVENESS",
        "INTERMEDIATE_EXACT_PAYLOAD_HEIGHT",
        "OUTPUT_EXACT_PAYLOAD_HEIGHT",
        "EXACT_ARITHMETIC_OPERATION_WORK",
        "FORMULA_DESCRIPTOR_SIZE_AND_DESCRIPTOR_COPIES",
        "RETAINED_STATE_AND_RETAINED_HISTORY",
        "REMATERIALIZATION_WORK",
        "CONTROLLER_STATE_AND_DETECTOR_STATE",
        "PER_QUERY_AND_TOTAL_QUERY_WORK",
        "PRECISION_AND_SHOT_COUNT",
    }
    require(required_uninstrumented.issubset(set(resources["not_instrumented"])), "resource caveats incomplete")


def main() -> None:
    source_and_document_audit()
    generated_production = regenerate(PRODUCTION)
    generated_reference = regenerate(REFERENCE)
    require(PRODUCTION_SEAL.read_bytes() == generated_production, "production seal is stale")
    require(REFERENCE_SEAL.read_bytes() == generated_reference, "reference seal is stale")
    production = json.loads(generated_production)
    reference = json.loads(generated_reference)
    metadata_and_scope(production, reference)
    fixture_parity(production, reference)
    path_and_interacting_parity(production, reference)
    approximation_and_bandwidth_parity(production, reference)
    bethe_and_stationary_scope(production, reference)
    require(production["assertions"]["status"] == reference["assertions"]["status"] == "SOURCE_SELF_CHECK_PASS", "source assertions failed")
    require(not production["assertions"]["general_approximation_lower_bound_claimed"], "general approximation lower bound promoted")
    require(not production["assertions"]["terminal"], "M263 incorrectly marked terminal")
    print(
        "PASS_STRICT_SCOPE M263_SUBGAP_BOUNDARY_RESOLVENT "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "THEORY=FORMAL_DERIVATION_SOURCE_AUDITED "
        "RESTORATION=NO_RESTORATION_CLAIM "
        "RESOURCE=PACKAGE_SELF_REVIEW "
        "DISPOSITION=PATH_FIXED_MARGIN_AND_BETHE_FACTORIZATION_RESOURCE_KILL_NEAR_THRESHOLD_REQUIRED"
    )


if __name__ == "__main__":
    main()
