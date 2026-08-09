#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M268 phase-kickback result."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "phase_eigenstate_kickback_oracle.py"
REFERENCE = PACKAGE / "tests" / "phase_eigenstate_kickback_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_FINDINGS.md"
GITIGNORE = PACKAGE / ".gitignore"
PRODUCTION_SEAL = (
    PACKAGE / "evidence" / "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK.json"
)
REFERENCE_SEAL = (
    PACKAGE
    / "evidence"
    / "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE.json"
)

CLAIM = (
    "FINITE_QUDIT_PHASE_EIGENSTATE_KICKBACK_RETURNS_A_SECRET_INDEPENDENT_"
    "CHARACTER_CARRIER_EXACTLY_FOR_TWO_DISTINCT_COHERENT_ORACLE_QUERIES_"
    "WHILE_PUBLIC_LAWS_ADMIT_DIRECT_PHASE_COMPILATION_SECRET_DEPENDENT_"
    "REUSABLE_PROGRAM_STATES_REQUIRE_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_"
    "ORACLE_ACCESS_ERASES_ANY_UNIQUE_PHASE_QEMU_ADVANTAGE"
)
CEILING = (
    "DETERMINISTIC_EXACT_FINITE_DIMENSIONAL_SOFTWARE_ORACLE_DIGITAL_TWIN_"
    "WITH_STIPULATED_EXTERNAL_COHERENT_QUERY_INTERFACE_NO_PHYSICAL_ORACLE_"
    "CARRIER_CUSTODY_QUERY_SEPARATION_OR_TOTAL_RESOURCE_ADVANTAGE"
)
PRODUCTION_RESTORATION = "EXACT_ALGEBRAIC_RESTORATION"
PRODUCTION_RESTORATION_SCOPE = (
    "EXACT_CYCLOTOMIC_LOGICAL_CARRIER_AND_INERT_REFERENCE_RETURN_FOR_TWO_"
    "DISTINCT_STIPULATED_COHERENT_ORACLE_QUERIES_ON_ONE_RESIDENT_SOFTWARE_"
    "ALLOCATION_WITHOUT_PHYSICAL_ORACLE_OR_CARRIER_CUSTODY"
)
REFERENCE_RESTORATION = "NO_RESTORATION_CLAIM"
REFERENCE_RESTORATION_SCOPE = (
    "FORMAL_EXACT_CHARACTER_DENSITY_AND_SPECTATOR_REFERENCE_IDENTITIES_ONLY_"
    "WITHOUT_EXECUTED_SAME_BACKING_OR_PHYSICAL_RESTORATION"
)
DISPOSITION = (
    "EXACT_KICKBACK_AND_LOGICAL_CARRIER_REUSE_ARE_VALID_BUT_PUBLIC_"
    "DESCRIPTORS_COMPILE_DIRECTLY_SECRET_DEPENDENT_REUSABLE_PROGRAM_STATES_"
    "PAY_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_ORACLE_ACCESS_RUNS_THE_"
    "IDENTICAL_QUERY_SO_NO_UNIQUE_PHASE_RESOURCE_TOTAL_ADVANTAGE_OR_M257_"
    "ESCAPE_IS_ESTABLISHED"
)
SUCCESSOR = (
    "COMPACT_PHYSICAL_COHERENT_ORACLE_GENERATION_LAW_WITH_SECRET_INDEPENDENT_"
    "FINITE_ENERGY_EIGENSTATE_CARRIER_AND_EQUAL_INTERFACE_TOTAL_RESOURCE_"
    "ACCOUNTING"
)
M257_GUARDRAIL = (
    "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_"
    "COUNTED_AS_A_PHASE_RESOURCE"
)
ARCHITECTURE_CLASSIFICATION = "MECHANISM_SEARCH_DIGITAL_TWIN_OUTSIDE_QEMU_DEVICE"
INTEGRATION_GATE = (
    "RETURN_TO_COMMON_PHASE_QEMU_GUEST_DEVICE_OR_BACKEND_AND_DEMONSTRATE_"
    "LIFECYCLE_CUSTODY_BOUNDARY_ORDERING_RESTORATION_REUSE_AND_SNAPSHOT_"
    "SHAM_LINEAGE_BEFORE_MACHINE_ARCHITECTURE_PROMOTION"
)

EXPECTED_HASHES = {
    PRODUCTION: "dfb5b606323e6fb61bda90c03329ab5238011b2a3f0941e1e7b7619c2ad40eb4",
    REFERENCE: "206410817cf440482b26fe7c2b80a3c20033731df5e70fd3303ddad0073ae6b0",
    CONTRACT: "1aca8b34b0a70cdb76408f6603db2ac527da00d05d2eb5d5d7b7808768b3a267",
    FINDINGS: "273f560d2f44812370084e9223178f59fa277294f4b5de28a62f5ca797c921c7",
    GITIGNORE: "9b5f20010d181ae39a79ca90cf70f6615567aeb087ae229fc5004ddeecee3b39",
}
EXPECTED_STDOUT_HASHES = {
    PRODUCTION: "4146d6a8efab1cc466262ef87b9662a2dff2a64a35d897f13ab59d9d05c68bf0",
    REFERENCE: "77f67cc11261897ec12283ae14f3758c94798371668b74981cca32d176978cc5",
}
EXPECTED_PRODUCTION_PAYLOAD_HASH = (
    "9ae57e6bffe7cf175cd32ca5e6a87484460292afc2a21065119f9989c822a33f"
)
EXPECTED_REFERENCE_PAYLOAD_HASH = (
    "1f5396aec4ab0754d9772eb76ec3bf9d422d503f3431925c629430983b381524"
)

FIXTURE_NAMES = {
    "branch_record_environment",
    "computational_basis_entanglement_controls",
    "cyclotomic_authority",
    "equal_coherent_oracle_access_collapse",
    "exact_reusable_program_orthogonality",
    "forrelation_oracle_cost_caveat",
    "m241_m242_linear_calibration_negative",
    "mixed_character_marginal_return_and_eta_dephasing",
    "nonproportional_program_control",
    "program_a",
    "program_b",
    "public_law_direct_compiler",
    "same_client_sequential_composition_diagnostic",
    "secret_independent_character_carrier",
    "spectator_reference_bell_preservation",
    "two_fresh_client_same_logical_carrier_transactions",
}
CHECK_NAMES = {
    "architecture_scope_fails_closed_outside_qemu_device",
    "basis_a_entangles_and_does_not_return",
    "basis_b_entangles_and_does_not_return",
    "branch_record_is_orthogonal",
    "chi1_density_shift_invariant",
    "chi1_hermitian",
    "chi1_pure",
    "chi1_shift_eigenvector_law",
    "chi1_trace_one",
    "client_reference_completeness_test_preserved",
    "client_reference_kickback_factorization",
    "cyclotomic_polynomial_is_exact",
    "equal_access_runs_identical_queries",
    "first_fresh_client_transaction_exact",
    "forrelation_is_prospective_only",
    "inert_qutrit_spectator_reference_sentinel_exact",
    "inert_sentinel_and_client_reference_tests_are_dimensionally_distinct",
    "m241_m242_calibration_is_negative",
    "mixed_character_carrier_marginal_returns",
    "mixed_character_client_dephases",
    "mixed_character_joint_does_not_return",
    "nielsen_chuang_classes_exact",
    "program_a_carrier_return",
    "program_a_exact_kickback",
    "program_a_program_b_phase_gates_are_nonproportional",
    "program_b_carrier_return",
    "program_b_exact_kickback",
    "public_direct_compiler_matches_all_boundaries",
    "same_client_sequential_identity_is_diagnostic_only",
    "two_fresh_client_boundaries_are_distinct",
    "two_fresh_client_carrier_return",
    "two_fresh_client_combined_boundary_exact",
    "two_fresh_client_supply_is_explicit",
    "two_oracles_are_distinct",
    "uniform_character_eta_is_delta",
}
REFERENCE_ONLY_CHECK_NAMES = {
    "no_programming_zero_overlap_is_theorem_requirement_not_measurement",
    "resource_scope_excludes_unmeasured_production_backing",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite_json(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def regenerate(script: Path) -> bytes:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=PACKAGE,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode == 0, f"{script.name} exited {result.returncode}")
    require(result.stderr == b"", f"unexpected stderr from {script.name}")
    require(result.stdout.endswith(b"\n"), f"unterminated JSON from {script.name}")
    require(result.stdout.count(b"\n") == 1, f"non-single-line JSON from {script.name}")
    parsed = json.loads(result.stdout)
    require(isinstance(parsed, dict), f"non-object JSON from {script.name}")
    require(finite_json(parsed), f"non-finite JSON from {script.name}")
    require(
        sha256_bytes(result.stdout) == EXPECTED_STDOUT_HASHES[script],
        f"stdout bytes changed for {script.name}",
    )
    return result.stdout


def scrub_nonclaim_diagnostics(value: Any) -> Any:
    """Scrub only diagnostics whose names explicitly disclaim claim authority."""
    if isinstance(value, dict):
        return {
            key: (
                0.0
                if key.endswith("_not_used_for_claim")
                else scrub_nonclaim_diagnostics(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_nonclaim_diagnostics(item) for item in value]
    return value


def seal_bytes(raw: bytes) -> bytes:
    normalized = scrub_nonclaim_diagnostics(json.loads(raw))
    return (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def normalized_document(path: Path) -> str:
    return " ".join(
        path.read_text(encoding="utf-8")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def source_and_document_audit() -> None:
    require(
        EXPECTED_HASHES[FINDINGS] != "PENDING_CORRECTED_FINDINGS_SHA256",
        "corrected findings hash has not been frozen",
    )
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing dependency: {path.name}")
        require(sha256(path) == expected, f"dependency changed: {path.name}")

    production_source = PRODUCTION.read_text(encoding="utf-8")
    reference_source = REFERENCE.read_text(encoding="utf-8")
    ast.parse(production_source, filename=str(PRODUCTION))
    reference_tree = ast.parse(reference_source, filename=str(REFERENCE))
    require(REFERENCE.name not in production_source, "production names reference")
    require(PRODUCTION.name not in reference_source, "reference names production")
    require(PRODUCTION.stem not in reference_source, "reference names production module")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    require(
        not imported.intersection({"importlib", "runpy", "subprocess"}),
        "reference has dynamic or subprocess coupling",
    )
    for forbidden in ("exec(", "eval(", "compile(", "__import__("):
        require(forbidden not in reference_source, f"reference contains {forbidden}")
    for anchor in (
        "class Cyclo3:",
        "class ResidentBacking:",
        "def oracle_permutation(",
        "def computational_basis_control(",
        "def mixed_character_control(",
        "def branch_environment_control(",
        "def cross_kerr_fock_compiler_control(",
        "def reusable_program_theorem(",
        '"M268_RESIDENT_ALLOCATION_0001"',
        '"phase_qemu_layer_classification"',
        '"matrix_materialization_ledger": {',
        '"theorem_required_program_overlap_if_fixed_exact_processor": ZERO',
        '"program_states_materialized": False',
        '"program_overlap_executed_or_measured": False',
    ):
        require(anchor in production_source, f"production source anchor missing: {anchor}")
    for anchor in (
        "class Cyclo3:",
        "def oracle_permutation(",
        "def two_client_target_permutation(",
        "def client_reference_target_permutation(",
        "def branch_record_gram(",
        "def diagonal_exponents_proportional(",
        '"restoration_classification": RESTORATION_CLASSIFICATION',
        '"phase_qemu_layer_classification"',
        '"theorem_required_program_overlap_if_fixed_exact_processor": ZERO',
        '"program_states_materialized": False',
        '"program_overlap_executed_or_measured": False',
        '"resource_scope_accounting"',
    ):
        require(anchor in reference_source, f"reference source anchor missing: {anchor}")

    for document in (CONTRACT, FINDINGS):
        text = document.read_text(encoding="utf-8")
        for value, label in (
            (CLAIM, "claim"),
            (CEILING, "ceiling"),
            (PRODUCTION_RESTORATION, "production restoration class"),
            (PRODUCTION_RESTORATION_SCOPE, "production restoration scope"),
            (DISPOSITION, "resource disposition"),
            (SUCCESSOR, "successor"),
            (ARCHITECTURE_CLASSIFICATION, "architecture classification"),
        ):
            require(value in text, f"{label} missing from {document.name}")
        if document == FINDINGS:
            require(REFERENCE_RESTORATION in text, "reference restoration class missing from findings")
            require(REFERENCE_RESTORATION_SCOPE in text, "reference restoration scope missing from findings")
        normalized = normalized_document(document)
        for anchor in (
            "standalone",
            "digital twin",
            "qemu device implemented false",
            "common guest visible device contract",
            "eligible for mechanism kill",
            "eligible for architecture promotion",
            "lifecycle",
            "custody",
            "boundary ordering",
            "restoration before response release",
            "same backing reuse",
            "snapshot",
            "reload",
            "sham",
            "lineage",
            "promotion",
            "m257",
            "m241",
            "m242",
            "forrelation",
            "public descriptor",
            "equal access",
            "physical oracle",
            "total resource advantage",
            "program states materialized",
            "returned carrier marginal",
            "fresh public algebraic carrier verifier",
            "expected client",
            "expected joint",
            "saved baseline",
        ):
            require(anchor in normalized, f"scope anchor {anchor!r} missing from {document.name}")
        if document == CONTRACT:
            require(
                "13859" in normalized or "13,859" in normalized,
                "13,859-cell floor missing from contract",
            )
            require("reconstructed from the public" in normalized, "public algebraic verifier reconstruction missing from contract")
            require("directly materialized comparator and control matrices" in normalized, "comparator/control matrix accounting missing from contract")
            require("theorem required program overlap if fixed exact processor" in normalized, "theorem-only program overlap missing from contract")
            require("does not execute or measure" in normalized, "program overlap nonexecution missing from contract")
            require("loaded joint transient" in normalized and "permuted joint transient" in normalized, "joint transient accounting missing from contract")
            require("whole process peak" in normalized, "unknown whole-process peak caveat missing from contract")
        else:
            require("verifier rematerialization" in normalized, "public algebraic verifier rematerialization missing from findings")
            require("controls" in normalized and "comparators" in normalized, "comparator/control matrix accounting missing from findings")
            require("zero overlap is the theorem required overlap for a hypothetical processor" in normalized, "theorem-only program overlap missing from findings")
            require("not an executed or measured overlap" in normalized, "program overlap nonexecution missing from findings")
            require("loaded client plus carrier density" in normalized and "permuted joint density" in normalized, "joint transient accounting missing from findings")
            require("not a package peak memory measurement" in normalized, "unknown package peak caveat missing from findings")
    findings_text = FINDINGS.read_text(encoding="utf-8")
    require(
        "77,711 source bytes" in findings_text,
        "findings production source-byte charge does not match frozen source",
    )
    production_check_row = (
        "| Production | `PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_ORACLE_V1` "
        "| `PASS_INTERNAL_EXACT_QUTRIT_KICKBACK_SELF_CHECK` | 35/35 |"
    )
    reference_check_row = (
        "| Separate reference | "
        "`PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE_V1` "
        "| `PASS_INDEPENDENT_EXACT_QUTRIT_KICKBACK_REFERENCE` | 37/37 |"
    )
    stale_reference_check_row = reference_check_row.replace("37/37", "35/35")
    require(
        findings_text.count(production_check_row) == 1,
        "findings must report the exact production 35/35 check row once",
    )
    require(
        findings_text.count(reference_check_row) == 1,
        "findings must report the exact separate-reference 37/37 check row once",
    )
    require(
        stale_reference_check_row not in findings_text,
        "findings retain the stale separate-reference 35/35 check row",
    )


def metadata_audit(production: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    require(
        set(production)
        == {
            "architecture_authority", "architecture_scope", "checks", "claim",
            "claim_ceiling", "claim_payload", "claim_payload_sha256", "fixtures",
            "m257", "mathematical_conventions", "milestone", "negative_claims",
            "next_mechanism", "public_boundary", "resource_disposition",
            "resource_ledger", "restoration_classification", "restoration_scope",
            "schema", "scope_exclusions", "source_self_assertion", "source_sha256",
            "status", "strongest_honest_comparators", "terminal", "theorem",
        },
        "production top-level schema changed",
    )
    require(
        set(reference)
        == {
            "architecture_authority", "architecture_scope", "ceiling", "checks",
            "claim", "claim_payload", "claim_payload_sha256",
            "expected_production_restoration_classification",
            "expected_production_restoration_scope", "fixtures", "m257",
            "mathematical_conventions", "milestone", "next_mechanism", "nonclaims",
            "reference_id", "reference_self_assertion", "resource_disposition",
            "resource_scope_accounting", "resources", "restoration_classification",
            "restoration_scope", "schema", "source_sha256", "status", "terminal",
        },
        "reference top-level schema changed",
    )
    require(
        production["schema"] == "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_ORACLE_V1",
        "production schema changed",
    )
    require(
        reference["schema"]
        == "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE_V1",
        "reference schema changed",
    )
    require(production["milestone"] == reference["milestone"] == "M268", "milestone changed")
    require(
        production["status"] == "PASS_INTERNAL_EXACT_QUTRIT_KICKBACK_SELF_CHECK",
        "production status changed",
    )
    require(
        production["source_self_assertion"] == "PASS_INTERNAL_CONSISTENCY_ONLY",
        "production self-assertion changed",
    )
    require("INDEPENDENT" not in production["status"], "production self-promoted")
    require(
        reference["reference_id"]
        == "M268_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE_V1",
        "reference id changed",
    )
    require(
        reference["status"] == "PASS_INDEPENDENT_EXACT_QUTRIT_KICKBACK_REFERENCE"
        and reference["reference_self_assertion"] == reference["status"],
        "reference status changed",
    )
    require(production["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "embedded production hash changed")
    require(reference["source_sha256"] == EXPECTED_HASHES[REFERENCE], "embedded reference hash changed")
    require(production["claim_payload_sha256"] == EXPECTED_PRODUCTION_PAYLOAD_HASH, "production claim payload changed")
    require(reference["claim_payload_sha256"] == EXPECTED_REFERENCE_PAYLOAD_HASH, "reference claim payload changed")
    require(production["claim"] == reference["claim"] == CLAIM, "claim authority changed")
    require(production["claim_ceiling"] == reference["ceiling"] == CEILING, "ceiling authority changed")
    require(production["resource_disposition"] == reference["resource_disposition"] == DISPOSITION, "resource disposition changed")
    require(production["next_mechanism"] == reference["next_mechanism"] == SUCCESSOR, "successor changed")
    require(production["restoration_classification"] == PRODUCTION_RESTORATION, "production restoration changed")
    require(production["restoration_scope"] == PRODUCTION_RESTORATION_SCOPE, "production restoration scope changed")
    require(reference["expected_production_restoration_classification"] == PRODUCTION_RESTORATION, "reference production-restoration expectation changed")
    require(reference["expected_production_restoration_scope"] == PRODUCTION_RESTORATION_SCOPE, "reference production-restoration scope expectation changed")
    require(reference["restoration_classification"] == REFERENCE_RESTORATION, "reference restoration claim changed")
    require(reference["restoration_scope"] == REFERENCE_RESTORATION_SCOPE, "reference restoration scope changed")
    require(not production["terminal"] and not reference["terminal"], "long-term goal terminated")
    require(set(production["fixtures"]) == set(reference["fixtures"]) == FIXTURE_NAMES, "fixture set changed")
    require(set(production["checks"]) == CHECK_NAMES, "production check-key set changed")
    require(
        set(reference["checks"]) == CHECK_NAMES | REFERENCE_ONLY_CHECK_NAMES,
        "reference check-key set changed",
    )
    require(len(production["checks"]) == 35, "production check count changed from 35/35")
    require(len(reference["checks"]) == 37, "reference check count changed from 37/37")
    require(all(production["checks"].values()) and all(reference["checks"].values()), "internal check failed")


def cyclotomic_and_kickback_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]
    pc = prod["cyclotomic_authority"]
    rc = ref["cyclotomic_authority"]
    require(pc["field"] == rc["field"] == "Q(omega)", "cyclotomic field changed")
    require(pc["dimension"] == rc["dimension"] == 3, "cyclotomic dimension changed")
    require(pc["minimal_polynomial"] == rc["minimal_polynomial"] == "omega^2+omega+1=0", "minimal polynomial changed")
    require(pc["omega"] == rc["omega"], "omega representation differs")
    require(pc["omega_squared"] == rc["omega_squared"], "omega squared differs")
    require(pc["omega_cubed"] == rc["omega_cubed"], "omega cubed differs")
    require(pc["polynomial_residual"] == rc["one_plus_omega_plus_omega_squared"], "cyclotomic residual differs")
    require(pc["floating_point_values_on_decision_path"] == 0, "production used floating decisions")
    require(rc["all_decision_paths_are_fraction_and_cyclotomic_equality"], "reference exactness changed")

    ps = prod["secret_independent_character_carrier"]
    rs = ref["secret_independent_character_carrier"]
    require(ps["character_label"] == rs["character_label"] == 1, "character label changed")
    require(ps["secret_independent"] and rs["secret_independent"], "carrier depends on secret")
    require(ps["projector"] == rs["density_matrix"], "chi1 projector parity failed")
    require(ps["unnormalized_sqrt3_times_state"] == rs["unnormalized_numerator_sqrt3_times_chi1"], "chi1 numerator parity failed")
    require(ps["trace"] == rs["trace"] == pc["omega_cubed"], "chi1 trace changed")
    require(ps["purity"] == rs["purity"] == pc["omega_cubed"], "chi1 purity changed")
    require(ps["hermitian"] and rs["hermitian"], "chi1 Hermiticity failed")
    require(ps["density_is_invariant_under_every_target_shift"] and rs["density_is_invariant_under_every_target_shift"], "chi1 shift invariance failed")
    require(ps["all_shifted_numerators_match_eigenvalue_law"] and rs["all_shifted_numerators_match_eigenvalue_law"], "chi1 shift eigenlaw failed")
    require(ps["shift_eigenvalues_0_1_2"] == rs["shift_eigenvalues"], "chi1 shift eigenvalues differ")

    for name, table in (("program_a", [0, 1]), ("program_b", [0, 2])):
        pp = prod[name]
        rp = ref[name]
        require(pp["function_table"] == rp["function_table"] == table, f"{name} table changed")
        require(pp["exact_kickback_factorization"] and rp["exact_kickback_factorization"], f"{name} factorization failed")
        require(pp["exact_character_carrier_marginal_return"] and rp["exact_character_carrier_marginal_return"], f"{name} carrier return failed")
        require(pp["boundary_density"] == rp["client_boundary_density"], f"{name} client boundary differs")
        require(pp["compiled_client_diagonal"] == rp["compiled_client_diagonal"], f"{name} diagonal differs")
        require(rp["target_density_after"] == rs["density_matrix"], f"{name} reference target changed")
        require(rp["coherent_oracle_query_count"] == 1, f"{name} reference query count changed")
        receipt = pp["receipt"]
        require(receipt["oracle_is_basis_permutation"], f"{name} did not execute permutation")
        require(receipt["oracle_permutation_basis_rows"] == 54, f"{name} permutation size changed")
        require(receipt["density_permutation_qomega_reads"] == receipt["density_permutation_qomega_writes"] == 2916, f"{name} density action changed")
        require(receipt["exact_factorization_before_release"], f"{name} release preceded factorization")
        require(receipt["target_spectator_reference_return_before_release"], f"{name} release preceded carrier return")
        require(receipt["boundary_released_after_return"], f"{name} boundary ordering failed")

    pn = prod["nonproportional_program_control"]
    rn = ref["nonproportional_program_control"]
    require(set(pn) == set(rn), "nonproportional A/B control schema differs")
    for key in set(pn) - {"interpretation"}:
        require(pn[key] == rn[key], f"nonproportional A/B parity failed: {key}")
    require(pn["a_and_b_phase_diagonals_are_nonproportional"], "A/B diagonals became proportional")
    require(pn["a_and_b_fresh_client_density_channels_are_distinct"], "A/B client boundaries collapsed")
    require(pn["oracles_remain_distinct_permutations"], "A/B permutations collapsed")


def reference_completeness_and_reuse_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]
    ps = prod["spectator_reference_bell_preservation"]
    rs = ref["spectator_reference_bell_preservation"]
    for key in (
        "bell_density_before",
        "bell_density_after_both_queries",
        "client_reference_bell_density_before",
        "client_reference_bell_density_after_program_a",
        "client_reference_marginal_after_program_a",
        "client_reference_target_after_program_a",
        "spectator_reference_marginal_after_both_queries",
    ):
        require(ps[key] == rs[key], f"spectator/client-reference parity failed: {key}")
    for key in (
        "exactly_unchanged",
        "bell_entanglement_preserved_by_local_kickback_unitary",
        "client_reference_marginal_is_maximally_mixed",
        "client_reference_target_exact_factorization",
        "spectator_reference_marginal_is_maximally_mixed",
    ):
        require(ps[key] and rs[key], f"reference completeness check failed: {key}")
    require(ps["spectator_dimension"] == rs["spectator_dimension"] == 3, "spectator dimension changed")
    require(ps["reference_dimension"] == rs["reference_dimension"] == 3, "qutrit reference dimension changed")
    require(ps["client_reference_purity_after_program_a"] == rs["client_reference_purity_after_program_a"], "client-reference purity differs")
    require(ps["spectator_marginal"] == ps["reference_marginal"], "qutrit Bell marginals differ")

    pt = prod["two_fresh_client_same_logical_carrier_transactions"]
    rt = ref["two_fresh_client_same_logical_carrier_transactions"]
    require(
        pt["transaction_model"]
        == "TWO_FRESH_CLIENT_TRANSACTIONS_ON_ONE_RESIDENT_TARGET_SPECTATOR_REFERENCE_CARRIER_ALLOCATION",
        "production transaction model changed",
    )
    require(
        rt["transaction_model"]
        == "TWO_FRESH_CLIENT_TRANSACTIONS_ON_ONE_LOGICAL_CHARACTER_CARRIER",
        "reference logical transaction model changed",
    )
    require(pt["executed_same_backing_claim"], "production same-backing execution claim lost")
    require(not rt["executed_same_backing_claim"], "reference claimed executed same backing")
    require(pt["query_order"] == rt["query_order"] == ["A", "B"], "query order changed")
    require(pt["function_tables"] == [[0, 1], [0, 2]], "production function table receipt changed")
    require(pt["client_supply_count"] == rt["client_supply_count"] == 2, "fresh-client supply changed")
    require(pt["fresh_client_transactions"] == 2, "fresh transaction count changed")
    require(pt["coherent_oracle_query_count"] == rt["coherent_oracle_query_count"] == 2, "query count changed")
    require(pt["program_a_fresh_client_boundary"] == rt["program_a_fresh_client_boundary"], "fresh A boundary differs")
    require(pt["program_b_fresh_client_boundary"] == rt["program_b_fresh_client_boundary"], "fresh B boundary differs")
    require(pt["combined_fresh_client_boundary_density"] == rt["combined_fresh_client_boundary_density"], "combined fresh-client boundary differs")
    require(pt["combined_fresh_client_boundary_exact_factorization"] and rt["combined_fresh_client_boundary_exact_factorization"], "combined boundary factorization failed")
    require(pt["carrier_returns_after_each_query"] and rt["carrier_returns_after_each_query"], "carrier did not return twice")
    require(pt["target_spectator_reference_returns_after_a"] and pt["target_spectator_reference_returns_after_b"], "production full carrier/reference return failed")
    require(pt["algebraic_carrier_reference_density_digest"] == pt["final_carrier_reference_density_digest"], "final carrier/reference digest changed")
    require(pt["allocation_id"] == "M268_RESIDENT_ALLOCATION_0001", "allocation ID changed")
    require(pt["carrier_base_id"] == "M268_QUTRIT_CHARACTER_SENTINEL_BASE_0001", "carrier base ID changed")
    for key in (
        "carrier_object_stable",
        "carrier_row_objects_stable",
        "joint_workspace_object_stable",
        "joint_workspace_row_objects_stable",
        "basis_object_stable",
        "same_logical_carrier_semantics",
        "executed_same_backing_claim",
        "only_declared_client_boundaries_released",
    ):
        require(pt[key], f"stable production backing check failed: {key}")
    require(pt["generation_sequence"] == [0, 1, 2], "generation sequence changed")
    for key in (
        "snapshot_count", "reload_count", "reseed_count", "carrier_swap_count",
        "carrier_reprepare_count", "baseline_read_count", "retained_history_entries",
    ):
        require(pt[key] == 0, f"accepted path used {key}")
    require(not pt["carrier_or_intermediate_joint_projection_released"], "carrier/intermediate escaped")
    for key, classification in (
        ("snapshot_reload", "REJECTED_HISTORY_BASED_RESET_NOT_RESTORATION"),
        ("fresh_carrier_swap", "REJECTED_EXTERNAL_REPLACEMENT_NOT_REUSE"),
        ("carrier_reprepare", "REJECTED_NEW_PREPARATION_NOT_RETURN"),
    ):
        control = pt["reset_controls"][key]
        require(not control["executed_on_accepted_path"], f"reset control executed: {key}")
        require(control["classification"] == classification, f"reset classification changed: {key}")

    boundary = production["public_boundary"]
    require(boundary["release_policy"] == "EACH_CLIENT_BOUNDARY_RELEASED_ONLY_AFTER_EXACT_TARGET_SPECTATOR_REFERENCE_RETURN_AND_FACTORIZATION", "boundary release policy changed")
    require(boundary["generation_at_release_a"] == 1 and boundary["generation_at_release_b"] == 2, "boundary release generations changed")
    require(boundary["release_count"] == 2, "boundary release count changed")
    require(not boundary["carrier_density_released"] and not boundary["joint_workspace_released"] and not boundary["oracle_branch_history_released"], "private state crossed boundary")
    require(boundary["program_a_client_density"] == prod["program_a"]["boundary_density"], "released A density differs")
    require(boundary["program_b_client_density"] == prod["program_b"]["boundary_density"], "released B density differs")

    pd = prod["same_client_sequential_composition_diagnostic"]
    rd = ref["same_client_sequential_composition_diagnostic"]
    require(pd == rd, "same-client diagnostic parity failed")
    require(pd["sequential_phase_is_identity"] and pd["result_free_global_identity_only"], "same-client diagnostic changed")
    require(not pd["is_reuse_boundary"] and pd["same_client_supply_count"] == 1, "same-client diagnostic counted as reuse")


def killer_controls_and_comparators_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]
    pb = prod["computational_basis_entanglement_controls"]
    rb = ref["computational_basis_entanglement_controls"]
    for name, table in (("program_a", [0, 1]), ("program_b", [0, 2])):
        require(pb[name]["function_table"] == rb[name]["function_table"] == table, f"basis {name} table changed")
        require(pb[name]["client_purity"] == rb[name]["client_purity"], f"basis {name} client purity differs")
        require(pb[name]["joint_purity"] == rb[name]["joint_purity"], f"basis {name} joint purity differs")
        require(pb[name]["client_target_entangled"] and rb[name]["client_target_entangled"], f"basis {name} did not entangle")
        require(not pb[name]["joint_is_product_of_marginals"], f"production basis {name} became product")
        require(rb[name]["schmidt_rank"] == 2 and rb[name]["distinct_target_images"], f"reference basis {name} entanglement witness changed")
        require(not pb[name]["target_returned"] and not rb[name]["target_returned"], f"basis {name} target returned")

    pm = prod["mixed_character_marginal_return_and_eta_dephasing"]
    rm = ref["mixed_character_marginal_return_and_eta_dephasing"]
    for pkey, rkey in (
        ("uniform_character_mixture", "uniform_character_mixture"),
        ("uniform_eta_delta_0_1_2", "uniform_eta_delta_0_1_2"),
        ("program_a_client_marginal_after_uniform_mixture", "program_a_client_marginal_after_uniform_mixture"),
        ("program_a_eta_prediction", "program_a_eta_prediction"),
    ):
        require(pm[pkey] == rm[rkey], f"mixed-character parity failed: {pkey}")
    require(pm["carrier_marginal_returns"] and rm["carrier_marginal_returns"], "mixed carrier marginal did not return")
    require(pm["uniform_mixture_equals_identity_over_three"] and rm["uniform_mixture_equals_identity_over_three"], "uniform mixture changed")
    require(pm["client_is_fully_dephased"] and rm["program_a_client_is_fully_dephased"], "eta dephasing failed")
    require(pm["client_density_after"] == pm["eta_prediction"], "production eta prediction differs")
    require(not pm["joint_state_returns"] and not rm["joint_state_returns"], "mixed joint state returned")
    require(pm["client_carrier_joint_is_correlated"], "mixed control lost correlation")

    pe = prod["branch_record_environment"]
    re = ref["branch_record_environment"]
    require(pe["program_a_uniform_record_gram"] == re["program_a_uniform_record_gram"], "environment record Gram differs")
    require(pe["program_a_branch_records_are_orthogonal"] and re["program_a_branch_records_are_orthogonal"], "branch records not orthogonal")
    require(pe["environment_recording_permutation_executed"], "environment permutation not executed")
    require(pe["environment_dimension"] == 2 and pe["environment_contains_two_branch_record"], "environment record dimension changed")
    require(pe["named_character_carrier_marginal_returns"], "named carrier marginal did not return")
    require(pe["client_is_fully_dephased"] and pe["client_dephasing_matches_record_gram"] and re["client_dephasing_matches_record_gram"], "environment dephasing law failed")
    require(not pe["reference_complete_joint_return"] and not re["reference_complete_joint_return"], "branch-record joint returned")
    require(not pe["environment_reset_executed"], "environment reset executed")

    pp = prod["public_law_direct_compiler"]
    rp = ref["public_law_direct_compiler"]
    for pkey, rkey in (
        ("program_a_compiled_exponents", "program_a_compiled_exponents"),
        ("program_b_compiled_exponents", "program_b_compiled_exponents"),
        ("combined_fresh_client_compiled_diagonal", "combined_fresh_client_compiled_diagonal"),
    ):
        require(pp[pkey] == rp[rkey], f"public compiler parity failed: {pkey}")
    for key in (
        "function_descriptors_are_public", "compilation_and_descriptor_costs_are_charged",
        "direct_program_a_boundary_exact", "direct_program_b_boundary_exact",
        "direct_combined_boundary_exact",
    ):
        require(pp[key] and rp[key], f"public compiler check failed: {key}")
    require(pp["compiler_law"] == rp["compiler_law"] == "D_F=DIAG_X_omega^F_X", "compiler law changed")
    require(pp["direct_compiler_coherent_oracle_queries"] == rp["direct_compiler_coherent_oracle_queries"] == 0, "direct compiler queried oracle")
    require(pp["carrier_dimension_retained"] == 0 and not pp["restoration_stage_executed"], "direct compiler retained carrier/restoration")
    kerr = pp["cross_kerr_fock_compiler_kill"]
    kerr_semantics = dict(kerr)
    kerr_ledger = kerr_semantics.pop("matrix_materialization_ledger")
    require(kerr_semantics == {
        "compiled_client_exponents_mod_3": [0, 2],
        "direct_compiler_matches": True,
        "exact_factorization": True,
        "fock_carrier_returns": True,
        "public_fock_number": 2,
        "public_law": "U_KERR|X,N>=omega^(X*N)|X,N>",
        "unique_carrier_advantage": False,
    }, "cross-Kerr/Fock compiler kill changed")
    require(
        kerr_ledger
        == production["resource_ledger"]
        ["directly_materialized_comparator_and_control_matrices"]
        ["cross_kerr_fock_compiler_control"]["matrix_materializations"],
        "cross-Kerr fixture/resource materialization ledgers differ",
    )

    po = prod["exact_reusable_program_orthogonality"]
    ro = ref["exact_reusable_program_orthogonality"]
    require(
        po["theorem"]
        == "A_FIXED_DETERMINISTIC_EXACT_PROCESSOR_FOR_NONPROPORTIONAL_CLIENT_UNITARIES_REQUIRES_ORTHOGONAL_PROGRAM_STATES",
        "production program theorem changed",
    )
    require(
        ro["theorem"]
        == "A_FIXED_EXACT_PROCESSOR_FOR_NONPROPORTIONAL_UNITARIES_REQUIRES_ORTHOGONAL_PROGRAM_STATES",
        "reference program theorem changed",
    )
    require(po["two_client_basis_phase_classes"] == ro["two_client_basis_phase_classes"] == [[0, 0], [0, 1], [0, 2]], "phase classes changed")
    require("program_a_program_b_exact_program_overlap" not in po, "production overlap mislabeled as executed evidence")
    require("program_a_program_b_exact_program_overlap" not in ro, "reference overlap mislabeled as executed evidence")
    zero = {"basis": ["1", "omega"], "coefficients": ["0", "0"]}
    require(
        po["theorem_required_program_overlap_if_fixed_exact_processor"]
        == ro["theorem_required_program_overlap_if_fixed_exact_processor"]
        == zero,
        "theorem-required program overlap changed",
    )
    for name, record in (("production", po), ("reference", ro)):
        require(not record["program_states_materialized"], f"{name} claims materialized program states")
        require(not record["program_overlap_executed_or_measured"], f"{name} claims executed/measured program overlap")
    require(
        po["overlap_value_status"]
        == "THEOREM_REQUIRED_SYMBOLIC_CONCLUSION_NOT_EXECUTED_OR_MEASURED",
        "production overlap status changed",
    )
    require(
        ro["zero_overlap_is_hypothetical_no_programming_theorem_requirement"],
        "reference promoted symbolic overlap to evidence",
    )
    require(po["minimum_program_dimension_for_two_basis_inputs"] == ro["minimum_program_dimension_for_two_basis_inputs"] == 3, "program dimension bound changed")
    require(po["program_a_and_b_are_nonproportional_and_witness_orthogonality"] and ro["program_a_and_b_are_nonproportional_and_witness_orthogonality"], "A/B orthogonality witness failed")
    rows = po["general_exact_dimension_lower_bound"]
    require([(row["client_basis_size"], row["nonproportional_phase_classes"], row["minimum_program_hilbert_dimension"]) for row in rows] == [(m, 3 ** (m - 1), 3 ** (m - 1)) for m in range(1, 7)], "general program dimension law changed")
    reference_rows = ro["general_qutrit_phase_class_count_and_dimension_lower_bound"]
    require(
        [
            (
                row["client_basis_size"],
                row["phase_classes_modulo_global_phase"],
                row["minimum_exact_program_hilbert_dimension"],
            )
            for row in reference_rows
        ]
        == [(m, 3 ** (m - 1), 3 ** (m - 1)) for m in range(1, 7)],
        "reference program dimension law changed",
    )
    comparator = production["strongest_honest_comparators"]["secret_reusable_program"]
    require(comparator == po, "secret-program comparator differs from theorem fixture")

    peq = prod["equal_coherent_oracle_access_collapse"]
    req = ref["equal_coherent_oracle_access_collapse"]
    for key in (
        "phase_route_coherent_queries", "equal_access_comparator_coherent_queries",
        "query_sequences_identical", "exact_boundaries_identical",
        "total_resource_advantage", "unique_phase_qemu_query_advantage",
    ):
        require(peq[key] == req[key], f"equal-access parity failed: {key}")
    require(peq["phase_route_coherent_queries"] == peq["equal_access_comparator_coherent_queries"] == 2, "equal-access query count changed")
    require(peq["phase_route_query_sequence"] == peq["equal_access_comparator_query_sequence"] == ["O_A", "O_B"], "equal-access calls changed")
    require(req["phase_route_query_sequence"] == req["equal_access_comparator_query_sequence"] == ["Q_A_ON_CHI1", "Q_B_ON_CHI1"], "reference equal-access calls changed")
    require(peq["query_sequences_identical"] and peq["exact_boundaries_identical"], "equal access no longer identical")
    require(not peq["total_resource_advantage"] and not peq["unique_phase_qemu_query_advantage"], "equal-access advantage promoted")

    for name in ("m241_m242_linear_calibration_negative", "forrelation_oracle_cost_caveat"):
        pitem = prod[name]
        ritem = ref[name]
        common = set(pitem).intersection(ritem) - {"reason", "required_change"}
        for key in common:
            require(pitem[key] == ritem[key], f"{name} parity failed: {key}")
    negative = prod["m241_m242_linear_calibration_negative"]
    require(not negative["m241_escape"] and not negative["m242_escape"], "M241/M242 escape promoted")
    require(not negative["linear_secret_query_lower_bound_changed"], "linear query law changed")
    prospective = prod["forrelation_oracle_cost_caveat"]
    reference_prospective = ref["forrelation_oracle_cost_caveat"]
    require(
        prospective["required_change"]
        == "RESTRICTED_PROMISE_BLACK_BOX_INTERFACE_WITH_PREDECLARED_EQUAL_COMPARATOR_ACCESS",
        "production Forrelation access-model change changed",
    )
    require(
        reference_prospective["required_change"]
        == "REPLACE_PUBLIC_TABLES_WITH_A_RESTRICTED_PROMISE_BLACK_BOX_INTERFACE_AND_PREDECLARE_THE_COMPARATOR_ACCESS",
        "reference Forrelation access-model change changed",
    )
    require(prospective["prospective_only"] and not prospective["implemented_here"], "Forrelation promoted from prospective")
    require(not prospective["query_separation_claimed"] and not prospective["forrelation_query_separation_claimed"], "Forrelation separation claimed")
    for key in (
        "carrier_preparation_and_precision_costs_must_be_charged",
        "oracle_custody_cost_must_be_charged", "oracle_generation_cost_must_be_charged",
        "total_resource_accounting_required",
    ):
        require(prospective[key], f"Forrelation resource charge lost: {key}")


def resources_nonclaims_architecture_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    ledger = production["resource_ledger"]
    require(set(ledger) == {
        "accepted_transact_matrix_materializations",
        "descriptor_and_oracle_internal_size", "dimensions",
        "directly_materialized_comparator_and_control_matrices",
        "environment_controller_and_history", "equal_coherent_oracle_comparator",
        "preparation_and_certification", "public_direct_phase_comparator",
        "qomega_storage_and_precision", "query_action_and_bandwidth",
        "software_total_accounting_boundary",
    }, "resource-ledger sections changed")
    dimensions = ledger["dimensions"]
    require(dimensions == {
        "client": 2,
        "inert_reference_qutrit": 3,
        "joint_transaction_hilbert_dimension": 54,
        "resident_carrier_density_qomega_cells": 729,
        "resident_target_spectator_reference_hilbert_dimension": 27,
        "reused_joint_density_workspace_qomega_cells": 2916,
        "spectator_qutrit": 3,
        "target_character_qutrit": 3,
    }, "dimension ledger changed")

    expected_transaction_materializations = {
        "baseline_read_count": 0,
        "client_boundary_qomega_cells": 4,
        "expected_client_verifier_qomega_cells": 4,
        "expected_joint_verifier_qomega_cells": 2916,
        "fresh_public_algebraic_carrier_verifier_qomega_cells": 729,
        "listed_concurrent_qomega_cell_floor": 13859,
        "loaded_joint_transient_qomega_cells": 2916,
        "permuted_joint_transient_qomega_cells": 2916,
        "public_algebraic_carrier_verifier_provenance": (
            "FRESH_PUBLIC_FORMULA_REMATERIALIZATION_NOT_SAVED_BASELINE_READ"
        ),
        "python_allocator_rss_and_liveness_instrumented": False,
        "resident_carrier_qomega_cells": 729,
        "retained_history_entries": 0,
        "returned_carrier_qomega_cells": 729,
        "reused_joint_workspace_qomega_cells": 2916,
        "saved_baseline_matrix_materialized": False,
        "whole_process_peak_qomega_cells_established": False,
    }
    for name in ("program_a", "program_b"):
        receipt_ledger = production["fixtures"][name]["receipt"][
            "matrix_materialization_ledger"
        ]
        require(
            receipt_ledger == expected_transaction_materializations,
            f"{name} matrix-materialization receipt changed",
        )
    transact = ledger["accepted_transact_matrix_materializations"]
    require(
        transact["per_transaction"] == expected_transaction_materializations,
        "accepted transact matrix enumeration changed",
    )
    require(transact["transaction_count"] == 2, "accepted transaction count changed")
    require(transact["program_a_and_program_b_ledgers_identical"], "A/B materialization ledgers differ")
    require(transact["persistent_resident_qomega_cells_allocated_once"] == 3645, "persistent cell count changed")
    require(transact["transient_and_verifier_qomega_cell_materializations_per_transaction"] == 10214, "per-transaction transient/verifier count changed")
    require(transact["transient_and_verifier_qomega_cell_materializations_for_two_transactions"] == 20428, "two-transaction transient/verifier count changed")
    require(transact["primary_persistent_plus_two_transaction_materialization_events"] == 24073, "materialization-event accounting changed")
    require(transact["listed_concurrent_qomega_cell_floor_per_transaction"] == 13859, "transaction liveness floor changed")
    require(not transact["listed_concurrent_floor_is_total_process_peak"], "liveness floor promoted to process peak")
    require(transact["whole_process_peak_qomega_cells"] == "NOT_ESTABLISHED_FAIL_CLOSED", "unknown peak no longer fails closed")
    require(transact["public_algebraic_verifier_is_fresh_formula_rematerialization"], "algebraic verifier provenance weakened")
    require(not transact["public_algebraic_verifier_is_saved_baseline"], "algebraic verifier became saved baseline")
    require(transact["baseline_read_count"] == transact["retained_history_entries"] == 0, "baseline/history read entered transact ledger")

    controls = ledger["directly_materialized_comparator_and_control_matrices"]
    require(set(controls) == {
        "branch_record_environment_control",
        "client_reference_completeness_control_qomega_cells",
        "computational_basis_controls",
        "cross_kerr_fock_compiler_control",
        "equal_coherent_oracle_comparator",
        "matrix_lists_are_logical_object_counts_not_allocator_or_rss_measurements",
        "mixed_character_control",
        "public_direct_phase_comparator_qomega_cells",
        "same_client_sequential_diagnostic_qomega_cells",
        "top_level_public_fixture_objects_qomega_cells",
        "whole_process_matrix_liveness_instrumented",
        "whole_process_peak_claimed",
    }, "materialized comparator/control sections changed")
    require(controls["matrix_lists_are_logical_object_counts_not_allocator_or_rss_measurements"], "logical matrix-count scope weakened")
    require(not controls["whole_process_matrix_liveness_instrumented"], "matrix liveness falsely instrumented")
    require(not controls["whole_process_peak_claimed"], "control matrix list promoted to peak")
    basis_controls = controls["computational_basis_controls"]
    require(basis_controls["fixture_count"] == 2 and basis_controls["program_a_and_program_b_ledgers_identical"], "basis-control materialization scope changed")
    require(basis_controls["per_fixture_matrix_materializations"] == {
        "client_marginal_density_qomega_cells": 4,
        "client_plus_density_qomega_cells": 4,
        "client_purity_product_verifier_qomega_cells": 4,
        "client_target_input_density_qomega_cells": 36,
        "client_target_output_density_qomega_cells": 36,
        "joint_purity_product_verifier_qomega_cells": 36,
        "materialized_qomega_cell_events": 174,
        "product_of_marginals_verifier_qomega_cells": 36,
        "target_basis_density_qomega_cells": 9,
        "target_marginal_density_qomega_cells": 9,
    }, "basis-control materialization enumeration changed")
    for name in ("program_a", "program_b"):
        require(
            production["fixtures"]["computational_basis_entanglement_controls"]
            [name]["matrix_materialization_ledger"]
            == basis_controls["per_fixture_matrix_materializations"],
            f"basis {name} fixture/resource materialization ledgers differ",
        )
    require(controls["mixed_character_control"]["matrix_materializations"] == {
        "client_marginal_density_qomega_cells": 4,
        "client_plus_density_qomega_cells": 4,
        "client_target_input_density_qomega_cells": 36,
        "client_target_output_density_qomega_cells": 36,
        "eta_predicted_client_density_qomega_cells": 4,
        "fully_dephased_client_constructor_qomega_cell_events": 8,
        "identity_over_three_constructor_qomega_cell_events": 18,
        "materialized_qomega_cell_events": 245,
        "product_of_marginals_verifier_qomega_cells": 36,
        "target_marginal_density_qomega_cells": 9,
        "three_character_projectors_qomega_cells": 27,
        "weighted_sum_constructor_qomega_cell_events_including_mixed_result": 63,
    }, "mixed-character materialization enumeration changed")
    require(
        production["fixtures"]["mixed_character_marginal_return_and_eta_dephasing"]
        ["matrix_materialization_ledger"]
        == controls["mixed_character_control"]["matrix_materializations"],
        "mixed-character fixture/resource materialization ledgers differ",
    )
    require(controls["branch_record_environment_control"]["matrix_materializations"] == {
        "client_marginal_density_qomega_cells": 4,
        "client_plus_density_qomega_cells": 4,
        "client_target_environment_input_density_qomega_cells": 144,
        "client_target_environment_output_density_qomega_cells": 144,
        "client_target_intermediate_density_qomega_cells": 36,
        "environment_basis_density_qomega_cells": 4,
        "environment_marginal_density_qomega_cells": 4,
        "half_identity_two_qomega_cells": 4,
        "identity_two_qomega_cells": 4,
        "materialized_qomega_cell_events": 366,
        "named_character_carrier_density_qomega_cells": 9,
        "target_marginal_density_qomega_cells": 9,
    }, "branch-record materialization enumeration changed")
    require(
        production["fixtures"]["branch_record_environment"]
        ["matrix_materialization_ledger"]
        == controls["branch_record_environment_control"]["matrix_materializations"],
        "branch-record fixture/resource materialization ledgers differ",
    )
    require(controls["cross_kerr_fock_compiler_control"]["matrix_materializations"] == {
        "client_marginal_density_qomega_cells": 4,
        "client_plus_density_qomega_cells": 4,
        "client_target_input_density_qomega_cells": 36,
        "client_target_output_density_qomega_cells": 36,
        "expected_client_density_qomega_cells": 4,
        "expected_factorized_density_qomega_cells": 36,
        "fock_basis_density_qomega_cells": 9,
        "materialized_qomega_cell_events": 138,
        "target_marginal_density_qomega_cells": 9,
    }, "cross-Kerr materialization enumeration changed")
    require(controls["client_reference_completeness_control_qomega_cells"] == {
        "client_reference_bell_density": 16,
        "client_reference_target_input_density": 144,
        "client_reference_target_output_density": 144,
        "expected_factorized_density": 144,
        "phased_client_reference_density": 16,
        "reference_marginal_density": 4,
    }, "client-reference control materializations changed")
    equal_control = controls["equal_coherent_oracle_comparator"]
    require(equal_control["per_transaction_matrix_materializations"] == expected_transaction_materializations, "equal-comparator transaction enumeration changed")
    require(equal_control["transaction_count"] == 2 and equal_control["program_a_and_program_b_ledgers_identical"], "equal-comparator materialization scope changed")
    require(
        equal_control["resident_carrier_qomega_cells"] == 729
        and equal_control["reused_joint_workspace_qomega_cells"] == 2916
        and equal_control["program_a_client_boundary_qomega_cells"] == 4
        and equal_control["program_b_client_boundary_qomega_cells"] == 4,
        "equal-comparator direct matrix counts changed",
    )
    comparator_receipts = production["fixtures"][
        "equal_coherent_oracle_access_collapse"
    ]["comparator_receipts"]
    require(len(comparator_receipts) == 2, "equal-comparator receipt count changed")
    for generation, receipt in enumerate(comparator_receipts, start=1):
        require(receipt["generation"] == generation, "equal-comparator generation changed")
        require(receipt["matrix_materialization_ledger"] == expected_transaction_materializations, "equal-comparator receipt materialization ledger changed")
        require(receipt["fresh_client_supplied"] and receipt["coherent_query_count_this_transaction"] == 1, "equal-comparator transaction law changed")
        require(receipt["exact_factorization_before_release"] and receipt["target_spectator_reference_return_before_release"] and receipt["boundary_released_after_return"], "equal-comparator release ordering changed")
    require(controls["public_direct_phase_comparator_qomega_cells"] == {
        "program_a_client_boundary": 4,
        "program_a_client_plus_input": 4,
        "program_b_client_boundary": 4,
        "program_b_client_plus_input": 4,
    }, "public direct comparator materializations changed")
    require(controls["same_client_sequential_diagnostic_qomega_cells"] == {
        "after_program_a_density": 36,
        "after_program_b_density": 36,
        "compiled_client_density": 4,
        "initial_client_target_density": 36,
    }, "same-client diagnostic materializations changed")
    require(controls["top_level_public_fixture_objects_qomega_cells"] == {
        "chi1_projector": 9,
        "post_transaction_public_algebraic_carrier": 729,
        "spectator_reference_bell_projector": 81,
    }, "top-level public fixture materializations changed")
    action = ledger["query_action_and_bandwidth"]
    require(action["coherent_query_count"] == action["logical_query_rounds"] == 2, "query ledger changed")
    require(action["basis_permutation_rows_total"] == 108, "permutation action changed")
    require(action["density_qomega_reads_total"] == action["density_qomega_writes_total"] == 5832, "density action ledger changed")
    require(action["physical_bandwidth_hz"] == action["physical_latency_seconds"] == "UNINSTANTIATED_NOT_FREE", "physical timing cost invented/free")
    require(action["wall_clock_latency_evidence"] == "NONE", "wall time entered claim")
    history = ledger["environment_controller_and_history"]
    require(history["generation_sequence"] == [0, 1, 2], "ledger generations changed")
    require(history["retained_dynamic_history_entries"] == 0, "history retained")
    require(history["snapshot_reload_reseed_swap_reprepare_baseline_reads"] == 0, "sham restoration operation entered ledger")
    direct = ledger["public_direct_phase_comparator"]
    require(direct["coherent_oracle_queries"] == direct["carrier_preparation"] == direct["carrier_qomega_cells"] == direct["restoration_actions"] == 0, "direct comparator resource law changed")
    equal = ledger["equal_coherent_oracle_comparator"]
    require(equal["coherent_queries"] == 2 and equal["client_supply_count"] == 2 and equal["outputs_identical"], "equal comparator ledger changed")
    require(equal["same_preparation_certification_and_restoration_categories_charged"], "equal comparator costs omitted")
    precision = ledger["qomega_storage_and_precision"]
    require(precision["exact_fraction_and_cyclotomic_equality_only"] and precision["arbitrary_precision_integer_cost_is_charged"], "exact arithmetic resource law changed")
    require(precision["floating_point_decision_operations"] == 0, "floating decision operation appeared")
    total = ledger["software_total_accounting_boundary"]
    require(not total["total_resource_advantage_claimed"] and not total["unmeasured_costs_treated_as_zero"], "total resource claim inflated")
    require(total["json_serialization_and_hashing_charged"] and total["source_hash_operations_charged"], "software accounting omitted")
    require(total["source_bytes"] == len(PRODUCTION.read_bytes()) == 77711, "production source-byte accounting changed")
    require(not total["2916_cell_workspace_is_total_peak"], "2916-cell workspace promoted to total peak")
    require(total["whole_process_peak_logical_or_physical_memory"] == "NOT_ESTABLISHED_FAIL_CLOSED", "whole-process memory peak no longer fails closed")
    for key in (
        "matrix_object_liveness",
        "python_objects_allocator_interpreter_and_process_costs",
        "whole_process_rss",
    ):
        require(total[key] == "UNINSTRUMENTED_NONZERO_FAIL_CLOSED", f"uninstrumented cost no longer fails closed: {key}")

    reference_scope = reference["resource_scope_accounting"]
    require(reference_scope["scope"] == "INDEPENDENT_EXACT_ALGEBRA_REFERENCE_MATERIALIZATIONS_ONLY_NO_PRODUCTION_RESIDENT_TRANSIENT_OR_PEAK_BACKING_MEASUREMENT", "reference resource scope changed")
    require(reference_scope["reference_structural_matrix_entry_counts_declared"], "reference matrix counts not declared")
    require(reference_scope["reference_exact_arithmetic_and_hashing_work_charged"], "reference exact work uncharged")
    require(reference_scope["reference_public_table_entries_accounted"] == 4, "reference table accounting changed")
    require(reference_scope["reference_stipulated_coherent_queries_accounted"] == 2, "reference query accounting changed")
    require(reference_scope["reference_largest_materialized_matrix_dimension"] == 12, "reference largest matrix dimension changed")
    require(reference_scope["reference_largest_materialized_matrix_entries"] == 144, "reference largest matrix entries changed")
    require(reference_scope["reference_exact_matrix_materializations"] == {
        "character_density": {"dimension": 3, "matrix_entries": 9},
        "client_reference_target_density": {"dimension": 12, "matrix_entries": 144},
        "qutrit_spectator_reference_density": {"dimension": 9, "matrix_entries": 81},
        "single_client_target_density": {"dimension": 6, "matrix_entries": 36},
        "two_fresh_clients_target_density": {"dimension": 12, "matrix_entries": 144},
    }, "reference local materialization accounting changed")
    for key in (
        "production_peak_backing_cells_independently_measured",
        "production_peak_bytes_independently_measured",
        "production_resident_backing_cells_independently_measured",
        "production_resident_backing_materialized_by_reference",
        "production_resource_parity_claimed",
        "production_transient_backing_cells_independently_measured",
        "production_transient_backing_materialized_by_reference",
        "reference_python_object_byte_cost_measured",
        "reference_runtime_peak_bytes_measured",
    ):
        require(not reference_scope[key], f"reference overclaimed resource measurement: {key}")
    require(reference_scope["production_resident_transient_or_peak_claims_are_out_of_scope"], "reference production-resource exclusion weakened")
    require(reference["resources"] == {
        "carrier_preparation_physical_cost_accounted": False,
        "character_carrier_dimension": 3,
        "client_dimension": 2,
        "client_supply_count": 2,
        "client_target_joint_dimension": 6,
        "combined_fresh_client_boundary_dimension": 4,
        "equal_access_comparator_oracle_queries": 2,
        "exact_arithmetic_field": "Q(omega)",
        "floating_point_operations_on_decision_path": 0,
        "oracle_generation_physical_cost_accounted": False,
        "physical_energy_accounted": False,
        "program_count": 2,
        "public_direct_compiler_oracle_queries": 0,
        "public_table_entries": 4,
        "secret_program_dimension_lower_bound_for_two_client_basis_states": 3,
        "source_bytes_and_json_hashing_are_charged_software_work": True,
        "stipulated_coherent_oracle_queries": 2,
        "two_fresh_clients_target_joint_dimension": 12,
    }, "reference resource ledger changed")

    require(set(production["scope_exclusions"]) == {
        "complexity_lower_bound", "enforced_secret_oracle_custody",
        "forrelation_or_other_promise_problem", "growing_problem_family",
        "physical_carrier_or_reference", "physical_oracle_generation",
        "physical_preparation_certification_energy_bandwidth_and_latency",
    }, "production scope exclusions changed")
    require(all(production["scope_exclusions"].values()), "production scope exclusion weakened")
    require(all(value is False for value in production["negative_claims"].values()), "production nonclaim promoted")
    require(all(value is False for value in reference["nonclaims"].values()), "reference nonclaim promoted")

    pm = production["m257"]
    rm = reference["m257"]
    require(pm["guardrail"] == rm["guardrail"] == M257_GUARDRAIL, "M257 guardrail changed")
    require(pm["guardrail_remains_intact"] and rm["guardrail_remains_intact"], "M257 weakened")
    require(not pm["escape_established"] and not rm["escape_established"], "M257 escape promoted")
    require(pm["equal_coherent_oracle_comparator_runs_identical_queries"], "production equal-access M257 check lost")
    require(rm["equal_coherent_oracle_case_runs_the_identical_two_queries"], "reference equal-access M257 check lost")

    for output_name, output in (("production", production), ("reference", reference)):
        scope = output["architecture_scope"]
        require(scope["phase_qemu_layer_classification"] == ARCHITECTURE_CLASSIFICATION, f"{output_name} architecture class changed")
        require(not scope["qemu_device_implemented"], f"{output_name} invented QEMU device")
        require(not scope["common_guest_visible_device_contract_exercised"], f"{output_name} invented common contract")
        require(scope["eligible_for_mechanism_kill"], f"{output_name} mechanism kill disabled")
        require(not scope["eligible_for_architecture_promotion"], f"{output_name} architecture promotion enabled")
        require(scope["promotion_requires_common_phase_qemu_device_or_backend"], f"{output_name} integration gate removed")
    require(production["architecture_scope"]["production_executes_bounded_logical_restoration"], "bounded production restoration erased")
    require(not production["architecture_scope"]["production_can_promote_architecture"], "production self-promoted architecture")
    require(reference["architecture_scope"]["reference_verifies_algebra_only"], "reference scope expanded")
    require(not reference["architecture_scope"]["reference_can_promote_architecture"], "reference promoted architecture")

    authority = production["architecture_authority"]
    require(authority["phase_qemu_layer_classification"] == ARCHITECTURE_CLASSIFICATION, "production architecture authority changed")
    require(authority["survivor_integration_gate"] == INTEGRATION_GATE, "survivor integration gate changed")
    require(authority["successor_requires_integration_gate"], "successor integration gate removed")
    require(authority["successor_if_physical_oracle_survives"] == SUCCESSOR, "gated successor changed")
    require(authority["bounded_exact_logical_restoration_evidence_preserved"], "bounded restoration evidence discarded")
    require(not authority["mechanism_proof_is_device_integration"], "mechanism mislabeled device integration")
    ra = reference["architecture_authority"]
    require(ra["reference_is_independent_exact_algebra"], "reference independence weakened")
    for key in (
        "reference_asserts_same_backing_identity", "reference_executes_restoration",
        "reference_imports_package_code", "reference_is_physical_evidence",
        "reference_reads_external_artifacts",
    ):
        require(not ra[key], f"reference authority inflated: {key}")


def seal_audit(
    production_bytes: bytes,
    reference_bytes: bytes,
    *,
    write: bool,
    preseal: bool,
) -> None:
    production_seal = seal_bytes(production_bytes)
    reference_seal = seal_bytes(reference_bytes)
    require(production_seal == production_bytes, "production stdout is not canonical compact sorted JSON")
    require(reference_seal == reference_bytes, "reference stdout is not canonical compact sorted JSON")
    if preseal:
        return
    if write:
        PRODUCTION_SEAL.parent.mkdir(parents=True, exist_ok=True)
        PRODUCTION_SEAL.write_bytes(production_seal)
        REFERENCE_SEAL.write_bytes(reference_seal)
    require(PRODUCTION_SEAL.is_file(), "production seal missing")
    require(REFERENCE_SEAL.is_file(), "reference seal missing")
    require(PRODUCTION_SEAL.read_bytes() == production_seal, "production seal drift")
    require(REFERENCE_SEAL.read_bytes() == reference_seal, "reference seal drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-seals", action="store_true")
    mode.add_argument("--preseal", action="store_true")
    arguments = parser.parse_args()

    source_and_document_audit()
    production_bytes = regenerate(PRODUCTION)
    reference_bytes = regenerate(REFERENCE)
    require(regenerate(PRODUCTION) == production_bytes, "production regeneration is nondeterministic")
    require(regenerate(REFERENCE) == reference_bytes, "reference regeneration is nondeterministic")
    production = json.loads(production_bytes)
    reference = json.loads(reference_bytes)

    metadata_audit(production, reference)
    cyclotomic_and_kickback_audit(production, reference)
    reference_completeness_and_reuse_audit(production, reference)
    killer_controls_and_comparators_audit(production, reference)
    resources_nonclaims_architecture_audit(production, reference)
    seal_audit(
        production_bytes,
        reference_bytes,
        write=arguments.write_seals,
        preseal=arguments.preseal,
    )

    print(
        "PASS_STRICT_SCOPE M268_PHASE_EIGENSTATE_KICKBACK "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "RESTORATION=EXACT_ALGEBRAIC_RESTORATION "
        "SCOPE=LOGICAL_QUTRIT_CARRIER_REFERENCE_RETURN "
        "RESOURCE=PACKAGE_SELF_REVIEW "
        "PHASE_QEMU=MECHANISM_SEARCH_NOT_DEVICE M257=INTACT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
