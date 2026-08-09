#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M267 conditional-Gaussian result."""

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
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
PRODUCTION = PACKAGE / "conditional_gaussian_closed_loop_obstruction.py"
REFERENCE = PACKAGE / "tests" / "conditional_gaussian_closed_loop_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_FINDINGS.md"
PRODUCTION_SEAL = (
    PACKAGE
    / "evidence"
    / "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_OBSTRUCTION.json"
)
REFERENCE_SEAL = (
    PACKAGE
    / "evidence"
    / "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_SEPARATE_REFERENCE.json"
)

CLAIM = (
    "FINITE_MODE_PUBLIC_FIXED_AXIS_CONDITIONAL_GAUSSIAN_LOOPS_WITH_EXACT_"
    "FAITHFUL_CARRIER_REFERENCE_IDENTITY_REDUCE_TO_A_DIRECT_CLIENT_DIAGONAL_"
    "PHASE_OR_DECLARED_DILATION_SCHUR_CHANNEL_WHILE_POSITIVE_ACCUMULATED_CP_"
    "DIVISIBLE_MARKOV_DIFFUSION_ON_A_CLAIMED_CARRIER_SUBSPACE_PRECLUDES_"
    "EXACT_SAME_MODE_CHANNEL_RETURN_ON_THAT_SUBSPACE"
)
CEILING = (
    "FINITE_MODE_FINITE_JOINT_CLIENT_LABEL_PUBLIC_PIECEWISE_QUADRATIC_OR_"
    "AFFINE_GAUSSIAN_DYNAMICS_WITH_FIXED_COMMUTING_CLIENT_OBSERVABLES_"
    "DECLARED_COMMON_DILATION_AND_EXACT_GAUSSIAN_MOMENT_OR_LIFTED_AFFINE_"
    "SYMPLECTIC_SEMANTICS_ONLY_NO_NONCOMMUTING_AXES_NONQUADRATIC_INTERACTIONS_"
    "NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_QEC_RESTRICTED_ACCESS_NONMARKOV_"
    "RECOHERENCE_INFINITE_MODE_OR_PHYSICAL_CUSTODY"
)
RESTORATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "FORMAL_REFERENCE_COMPLETE_GAUSSIAN_CHANNEL_IDENTITY_CRITERION_AND_"
    "POSITIVE_DIFFUSION_NO_RETURN_ON_DECLARED_SUPPORT_WITHOUT_EXECUTED_OR_"
    "PHYSICAL_CARRIER_RESTORATION"
)
DISPOSITION = (
    "GENERAL_SECTOR_DIRECT_CLIENT_SHADOW_EXISTS_WITH_EXPLICIT_L_OR_L_SQUARED_"
    "COST_AND_THE_AFFINE_LABEL_COROLLARY_IS_POLYNOMIALLY_COMPACT_WHILE_"
    "POSITIVE_DIFFUSION_ON_CLAIMED_SUPPORT_FORBIDS_EXACT_REFERENCE_COMPLETE_"
    "RETURN_SO_NO_CATALYTIC_BUS_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
SUCCESSOR = (
    "RESTRICTED_ACCESS_NON_GAUSSIAN_PHASE_EIGENSTATE_KICKBACK_ORACLE_WITH_"
    "FAITHFUL_CARRIER_RETURN_PREPARATION_PRECISION_QUERY_AND_CUSTODY_COSTS"
)

EXPECTED_HASHES = {
    PRODUCTION: "1333b6990644df6349cf87280fbc868bd5f019ac6b057afa0d1c71811463b6e9",
    REFERENCE: "fe63c91736c62e96cae111986fe5a79f93fd38d1b9a3389857a2d2b760a7c3d8",
    CONTRACT: "907d5b2c2a59bb016083f20410cf5c09fd6781e4ec638786a117d0c17318c48d",
    FINDINGS: "8894225443914f03c82a546750629acd501d7cf36231e8da5dfee63a2b9b4c63",
}
EXPECTED_STDOUT_HASHES = {
    PRODUCTION: "bb74c78c267329856f1bb03e21eeec6161966a17961b073384b91b0ffa7a950a",
    REFERENCE: "d35ea2eac12a993fab2d14b12af25f1d65383f140c9e6096a15f4728a321d363",
}
EXPECTED_REFERENCE_PAYLOAD_HASH = (
    "7c4345d38f24a295d01643c98840d2595d9d6a4c2f58e7e074042fbe4cad4e7c"
)

FIXTURE_NAMES = {
    "metaplectic_2pi_vs_zero",
    "metaplectic_4pi_control",
    "weyl_rectangle_cocycle",
    "vacuum_rotation_marginal_false_positive",
    "additive_diffusion",
    "pure_loss_fixed_point",
    "rank_deficient_dark_mode",
    "finite_environment_recurrence",
    "common_nontrivial_bus_evolution",
    "declared_environment_schur",
    "sector_scaling",
}
PRODUCTION_CHECKS = {
    "exact_claim_authority",
    "metaplectic_2pi_lift_detected",
    "metaplectic_4pi_returns_lift",
    "weyl_rectangle_closes_with_cocycle",
    "marginal_false_positive_caught_by_reference",
    "positive_additive_diffusion_rejects_return",
    "pure_loss_fixed_point_not_channel_identity",
    "dark_kernel_scopes_no_return",
    "finite_environment_recurrence_outside_scope",
    "factorization_not_misclassified_as_restoration",
    "declared_dilation_schur_requires_overlap",
    "general_cost_not_hidden_by_affine_corollary",
    "affine_weyl_corollary_is_explicit_and_fail_closed",
    "no_physical_result",
    "no_same_backing_result",
    "m257_intact",
}
REFERENCE_CHECKS = {
    "metaplectic_2pi_affine_identity_lifted_minus_one",
    "metaplectic_4pi_returns_to_zero_lift",
    "weyl_rectangle_closes_with_nontrivial_cocycle",
    "vacuum_marginal_test_is_not_reference_complete",
    "positive_additive_diffusion_rejects_identity",
    "pure_loss_vacuum_fixed_but_reference_fails",
    "rank_deficient_diffusion_preserves_only_dark_kernel",
    "finite_environment_recurrence_is_outside_cp_divisible_scope",
    "common_nontrivial_map_is_not_carrier_return",
    "declared_environment_kernel_is_a_channel",
    "sector_counts_are_exact",
    "affine_weyl_force_q_squared_corollary_is_exactly_scoped",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def number(value: Any) -> float:
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        require(value["denominator"] != 0, "zero fraction denominator")
        return float(value["numerator"] / value["denominator"])
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"not a numeric value: {value!r}",
    )
    return float(value)


def close(
    first: Any,
    second: Any,
    *,
    absolute: float = 1e-12,
    relative: float = 0.0,
    label: str,
) -> None:
    left = number(first)
    right = number(second)
    require(
        math.isclose(left, right, abs_tol=absolute, rel_tol=relative),
        f"{label}: {left!r} != {right!r}",
    )


def close_sequence(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    absolute: float = 1e-12,
    label: str,
) -> None:
    require(len(first) == len(second), f"{label}: length changed")
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            close_sequence(left, right, absolute=absolute, label=f"{label}[{index}]")
        else:
            close(left, right, absolute=absolute, label=f"{label}[{index}]")


def complex_value(value: Mapping[str, Any]) -> complex:
    require(set(value) == {"real", "imag"}, "complex schema changed")
    return complex(number(value["real"]), number(value["imag"]))


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


def source_and_document_audit() -> None:
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
        "return -pair_sum / 2",
        '"identity_channel_gaussian_conditions": "X_S=I_D_S=0_Y_S=0"',
        '"accumulated_diffusion_gramian_Y"',
        '"general_sector_descriptor_cost": "EXPLICIT_L_OR_L_SQUARED_NO_BLANKET_COMPACTNESS"',
    ):
        require(anchor in production_source, f"production source anchor missing: {anchor}")
    for anchor in (
        "def apply_signal_channel_to_reference(",
        "def finite_environment_recurrence_fixture(",
        "np.linalg.eigvalsh",
        '"reference_complete_identity_criterion"',
        '"generic_direct_shadow_is_polynomially_compact": False',
    ):
        require(anchor in reference_source, f"reference source anchor missing: {anchor}")

    for document in (CONTRACT, FINDINGS):
        text = document.read_text(encoding="utf-8")
        for value, label in (
            (CLAIM, "claim"),
            (CEILING, "ceiling"),
            (RESTORATION, "restoration class"),
            (RESTORATION_SCOPE, "restoration scope"),
            (DISPOSITION, "resource disposition"),
            (SUCCESSOR, "successor"),
        ):
            require(value in text, f"{label} missing from {document.name}")
        normalized = " ".join(
            text.lower().replace("_", " ").replace("-", " ").split()
        )
        for anchor in (
            "m257",
            "reference complete",
            "faithful",
            "first moment",
            "covariance",
            "common dilation",
            "environment overlap",
            "fixed point",
            "dark",
            "finite environment",
            "non markov",
            "environment reversal",
            "noncommuting",
            "nonquadratic",
            "non gaussian",
            "measurement",
            "qec",
            "restricted",
            "infinite mode",
            "physical custody",
            "snapshot",
            "reset",
            "resource advantage",
            "unbounded",
            "l^2",
            "affine label",
        ):
            require(anchor in normalized, f"scope anchor {anchor!r} missing from {document.name}")
        require(
            any(anchor in normalized for anchor in ("same backing", "same carrier", "same mode")),
            f"same-carrier caveat missing from {document.name}",
        )
        require(
            any(
                anchor in normalized
                for anchor in (
                    "complexity lower bound",
                    "no asymptotic separation",
                    "every gaussian quantum process is classically easy",
                )
            ),
            f"complexity-scope caveat missing from {document.name}",
        )
        require(
            "replace the bit with pi" in normalized
            or "replacement of bits with pi" in normalized,
            f"bit-replacement caveat missing from {document.name}",
        )
        compact = normalized.replace(" ", "").replace("`", "")
        raw_compact = "".join(text.lower().replace("`", "").split())
        require("v_k(z)=v_k0+sum_iz_iv_ki" in raw_compact, f"affine-force law missing from {document.name}")
        require(
            (
                "common label independent symplectic" in normalized
                or "fixed, label independent symplectic/quadratic" in normalized
            ),
            f"common-symplectic restriction missing from {document.name}",
        )
        require(
            "label dependent quadratic generator" in normalized
            or "no quadratic generator depends on a client label" in normalized,
            f"label-dependent quadratic-generator exclusion missing from {document.name}",
        )
        require(
            "closes for every joint label" in normalized
            or "every branch displacement closes" in normalized,
            f"all-label closure hypothesis missing from {document.name}",
        )
        require("o(k(m^2+qm))" in compact, f"affine-force input cost missing from {document.name}")
        require("o(q^2)" in compact, f"compiled affine-force cost missing from {document.name}")
        require("compilation" in normalized and "application" in normalized, f"compile/apply charges missing from {document.name}")
        require("public" in normalized and "charged" in normalized, f"public/charged K scope missing from {document.name}")
        require(
            "finite per program" in normalized or "fixed for one program" in normalized,
            f"finite-per-program K scope missing from {document.name}",
        )
        require(
            "not assumed constant across a growing family" in normalized,
            f"K scaling-family charge missing from {document.name}",
        )


def metadata_audit(production: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    require(
        set(production)
        == {
            "schema", "milestone", "status", "terminal", "claim", "claim_ceiling",
            "restoration_classification", "restoration_scope", "resource_disposition",
            "next_mechanism", "source_self_assertion", "theorem", "fixtures",
            "strongest_honest_classical_comparator", "resource_ledger",
            "scope_exclusions", "negative_claims", "checks", "source_sha256",
        },
        "production top-level schema changed",
    )
    require(
        set(reference)
        == {
            "schema", "reference_id", "milestone", "claim", "ceiling",
            "restoration_classification", "restoration_scope", "resource_disposition",
            "next_mechanism", "mathematical_conventions", "class_theorem", "fixtures",
            "checks", "claims", "architecture_authority", "reference_self_assertion",
            "status", "terminal", "claim_payload_sha256", "source_sha256",
        },
        "reference top-level schema changed",
    )
    require(
        production["schema"]
        == "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_OBSTRUCTION_V1",
        "production schema changed",
    )
    require(
        reference["schema"]
        == "PHASE_QEMU_V9_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_SEPARATE_REFERENCE_V1",
        "reference schema changed",
    )
    require(production["milestone"] == reference["milestone"] == "M267", "milestone changed")
    require(
        production["status"] == "PASS_INTERNAL_FORMAL_OBSTRUCTION_SELF_CHECK",
        "production internal status changed",
    )
    require(
        production["source_self_assertion"] == "PASS_INTERNAL_CONSISTENCY_ONLY",
        "production self-assertion changed",
    )
    require("INDEPENDENT" not in production["status"], "production self-promoted")
    require(
        reference["reference_id"]
        == "M267_CONDITIONAL_GAUSSIAN_CLOSED_LOOP_SEPARATE_REFERENCE_V1",
        "reference id changed",
    )
    require(
        reference["status"] == "PASS_INDEPENDENT_ANALYTIC_CLASS_REFERENCE"
        and reference["reference_self_assertion"] == reference["status"],
        "reference status changed",
    )
    require(production["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "embedded production hash changed")
    require(reference["source_sha256"] == EXPECTED_HASHES[REFERENCE], "embedded reference hash changed")
    require(reference["claim_payload_sha256"] == EXPECTED_REFERENCE_PAYLOAD_HASH, "reference claim payload changed")
    for key, expected in (
        ("claim", CLAIM),
        ("restoration_classification", RESTORATION),
        ("restoration_scope", RESTORATION_SCOPE),
        ("resource_disposition", DISPOSITION),
        ("next_mechanism", SUCCESSOR),
    ):
        require(production[key] == reference[key] == expected, f"authority {key} changed")
    require(production["claim_ceiling"] == reference["ceiling"] == CEILING, "ceiling changed")
    require(not production["terminal"] and not reference["terminal"], "long-term goal terminated")
    require(set(production["fixtures"]) == set(reference["fixtures"]) == FIXTURE_NAMES, "fixture set changed")
    require(
        all(record.get("fixture") == name for name, record in production["fixtures"].items()),
        "production fixture self-names changed",
    )
    require(set(production["checks"]) == PRODUCTION_CHECKS, "production check-key set changed")
    require(len(production["checks"]) == 16 and all(production["checks"].values()), "production checks failed")
    require(set(reference["checks"]) == REFERENCE_CHECKS, "reference check-key set changed")
    require(len(reference["checks"]) == 12 and all(reference["checks"].values()), "reference checks failed")


def lifted_and_reference_fixture_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]
    identity2 = ((1.0, 0.0), (0.0, 1.0))

    p2 = prod["metaplectic_2pi_vs_zero"]
    r2 = ref["metaplectic_2pi_vs_zero"]
    require(p2["projected_endpoints_equal"] and p2["carrier_reference_identity"], "2pi carrier endpoint failed")
    require(p2["lifted_metaplectic_signs"] == [1, -1], "2pi lifted signs changed")
    close_sequence(p2["projected_symplectic_endpoints"][0], identity2, label="production zero endpoint")
    close_sequence(p2["projected_symplectic_endpoints"][1], r2["two_pi_affine_symplectic_map"], label="2pi endpoint parity")
    require(r2["affine_symplectic_maps_equal"] and not r2["lifted_scalars_equal"], "reference 2pi lift failed")
    close(r2["angle_radians"], 2.0 * math.pi, label="2pi angle")
    require(complex_value(r2["zero_metaplectic_scalar"]) == 1 + 0j, "zero lift changed")
    require(complex_value(r2["two_pi_metaplectic_scalar"]) == -1 + 0j, "2pi lift changed")

    p4 = prod["metaplectic_4pi_control"]
    r4 = ref["metaplectic_4pi_control"]
    require(p4["matches_zero_lift"] and p4["lifted_metaplectic_sign"] == 1, "4pi control failed")
    require(r4["returns_to_zero_lift"], "reference 4pi control failed")
    close(r4["angle_radians"], 4.0 * math.pi, label="4pi angle")
    close_sequence(p4["projected_symplectic_endpoint"], r4["four_pi_affine_symplectic_map"], label="4pi endpoint parity")

    pw = prod["weyl_rectangle_cocycle"]
    rw = ref["weyl_rectangle_cocycle"]
    sector_order = ["++", "+-", "-+", "--"]
    spin_labels = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    sector_phases = [-1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0]
    symbolic_phases = ["EXP(-I/6)", "EXP(+I/6)", "EXP(+I/6)", "EXP(-I/6)"]
    close(pw["a"], rw["a"], label="Weyl a")
    close(pw["b"], rw["b"], label="Weyl b")
    close(pw["base_signed_symplectic_area"], 1.0 / 6.0, label="Weyl base area")
    require(pw["sector_order"] == rw["sector_order"] == sector_order, "Weyl sector order changed")
    require(pw["sector_spin_labels"] == spin_labels, "Weyl spin labels changed")
    require(pw["client_qubits"] == 2, "Weyl client width changed")
    require(pw["affine_force_xi"] == "(A*Z0,0)" and pw["affine_force_eta"] == "(0,B*Z1)", "production affine-force law changed")
    require(pw["weyl_convention"] == "W_V_W_W=EXP_MINUS_I_SIGMA_V_W_OVER_2_W_V_PLUS_W", "production Weyl convention changed")
    require(rw["weyl_convention"] == "W_XI_W_ETA=EXP_MINUS_I_OVER_2_XI_TRANSPOSE_OMEGA_ETA_TIMES_W_XI_PLUS_ETA", "reference Weyl convention changed")
    require(pw["exact_sector_phase_law_radians"] == "PHI(Z0,Z1)=-Z0*Z1/6", "production Weyl phase law changed")
    require(rw["exact_client_phase_law"] == "EXP_MINUS_I_Z0_Z1_OVER_6", "reference Weyl phase law changed")
    require(
        rw["exact_sector_phase_sequence"]
        == "[EXP_MINUS_I_OVER_6,EXP_PLUS_I_OVER_6,EXP_PLUS_I_OVER_6,EXP_MINUS_I_OVER_6]",
        "reference Weyl symbolic sequence changed",
    )
    require(pw["direct_client_diagonal_in_declared_order"] == symbolic_phases, "production Weyl symbolic diagonal changed")
    close_sequence(pw["sector_phase_radians_in_declared_order"], sector_phases, label="production Weyl sector phase sequence")
    require(pw["phase_survives_closed_carrier_loop"] and rw["lifted_client_diagonal_is_nontrivial"], "Weyl lifted client phase lost")
    require(pw["all_sector_displacements_close"] and pw["carrier_reference_identity_all_sectors"], "production all-sector Weyl closure failed")
    require(rw["all_sector_carrier_affine_maps_are_identity"] and rw["all_sector_lifted_scalars_match_exact_zz_law"], "reference all-sector Weyl law failed")
    require(len(pw["sector_records"]) == len(rw["sectors"]) == 4, "Weyl sector count changed")
    for index, (precord, rrecord) in enumerate(
        zip(pw["sector_records"], rw["sectors"], strict=True)
    ):
        sector = sector_order[index]
        z0, z1 = spin_labels[index]
        phase = sector_phases[index]
        require(precord["sector"] == rrecord["name"] == sector, f"Weyl sector {index} name changed")
        require(precord["z0"] == rrecord["z0"] == z0, f"Weyl sector {sector} z0 changed")
        require(precord["z1"] == rrecord["z1"] == z1, f"Weyl sector {sector} z1 changed")
        close(precord["lifted_loop_phase_radians"], phase, label=f"Weyl sector {sector} production phase")
        close(rrecord["cocycle_exponent_radians"], phase, label=f"Weyl sector {sector} reference phase")
        close(rrecord["expected_zz_exponent_radians"], phase, label=f"Weyl sector {sector} expected phase")
        require(precord["lifted_loop_phase_symbolic"] == symbolic_phases[index], f"Weyl sector {sector} symbolic phase changed")
        close_sequence(precord["ordered_vectors"], rrecord["ordered_rectangle"], label=f"Weyl sector {sector} path")
        close_sequence(precord["net_displacement"], (0.0, 0.0), absolute=0.0, label=f"Weyl sector {sector} production closure")
        close_sequence(rrecord["net_displacement"], (0.0, 0.0), absolute=0.0, label=f"Weyl sector {sector} reference closure")
        require(rrecord["carrier_affine_map_is_identity"] and rrecord["matches_exact_zz_law"], f"Weyl sector {sector} reference law failed")
        scalar = complex(math.cos(phase), math.sin(phase))
        require(abs(complex_value(rrecord["lifted_rectangle_scalar"]) - scalar) <= 2e-15, f"Weyl sector {sector} scalar changed")
        require(abs(complex_value(rrecord["expected_zz_scalar"]) - scalar) <= 2e-15, f"Weyl sector {sector} expected scalar changed")
        for row in range(4):
            value = complex_value(rw["direct_client_diagonal"][row][index])
            expected = scalar if row == index else 0j
            require(abs(value - expected) <= 2e-15, f"Weyl direct diagonal[{row},{index}] changed")

    pv = prod["vacuum_rotation_marginal_false_positive"]
    rv = ref["vacuum_rotation_marginal_false_positive"]
    close_sequence(pv["bus_symplectic"], rv["rotation"], label="rotation parity")
    close_sequence(pv["input_covariance"], rv["faithful_tmsv_covariance_before"], label="TMSV input covariance")
    close_sequence(pv["output_covariance"], rv["faithful_tmsv_covariance_after"], label="TMSV output covariance")
    require(pv["bus_marginal_unchanged"] and rv["marginal_test_false_positive"], "marginal false positive lost")
    require(pv["reference_complete_identity_rejected"] and not rv["reference_complete_identity"], "faithful reference failed")
    require(rv["faithful_tmsv_frobenius_change"] > 1.0, "faithful TMSV did not detect rotation")
    close(pv["tmsv_c"], rv["tmsv_parameters"]["c"], label="TMSV c")
    close(pv["tmsv_s"], rv["tmsv_parameters"]["s"], label="TMSV s")
    close(pv["covariance_condition_number"], 4.0, label="TMSV conditioning")
    close(pv["input_mean_quanta_per_mode"], 1.0 / 8.0, label="TMSV energy")

    theorem = production["theorem"]
    reference_theorem = reference["class_theorem"]
    require(theorem["identity_channel_gaussian_conditions"] == "X_S=I_D_S=0_Y_S=0", "production first-moment/channel criterion changed")
    require("X=I_D=0_Y=0" in reference_theorem["reference_complete_identity_criterion"], "reference first-moment/channel criterion changed")
    require(
        reference["mathematical_conventions"]["gaussian_channel"]
        == "M_MAPS_TO_X_M_PLUS_D_AND_V_MAPS_TO_X_V_XT_PLUS_Y",
        "Gaussian mean/covariance convention changed",
    )


def noise_and_scope_fixture_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]

    pa = prod["additive_diffusion"]
    ra = ref["additive_diffusion"]
    close_sequence(pa["X"], ra["x"], label="additive X")
    close_sequence(pa["accumulated_diffusion_gramian_Y"], ra["y"], label="additive Y")
    close_sequence(pa["vacuum_input_covariance"], ra["vacuum_covariance_before"], label="additive vacuum input")
    close_sequence(pa["vacuum_output_covariance"], ra["vacuum_covariance_after"], label="additive vacuum output")
    require(pa["gramian_rank"] == ra["diffusion_rank"] == 2, "additive rank changed")
    close(pa["vacuum_output_determinant"], ra["vacuum_output_covariance_determinant"], label="additive determinant")
    close(pa["vacuum_output_purity"], ra["vacuum_output_gaussian_purity"], label="additive purity")
    close(pa["added_mean_quanta"], ra["added_mean_occupation"], label="additive occupation")
    require(not pa["exact_reference_complete_return"] and not ra["reference_complete_identity"], "additive return promoted")
    require(ra["faithful_tmsv_frobenius_change"] > 0.17, "additive faithful reference insensitive")

    pl = prod["pure_loss_fixed_point"]
    rl = ref["pure_loss_fixed_point"]
    close(pl["eta"], rl["eta"], label="loss eta")
    close_sequence(pl["vacuum_input_covariance"], rl["vacuum_covariance_before"], label="loss vacuum input")
    close_sequence(pl["vacuum_output_covariance"], rl["vacuum_covariance_after"], label="loss vacuum output")
    require(pl["prepared_vacuum_is_fixed_point"] and rl["vacuum_is_exact_fixed_point"], "loss fixed point lost")
    require(not pl["channel_is_identity"] and not rl["reference_complete_identity"], "loss channel promoted")
    close(pl["tmsv_output_bus_variance"], rl["faithful_tmsv_signal_variance_after"], label="loss TMSV variance")
    close(pl["tmsv_output_cross_correlation_squared"], rl["faithful_tmsv_cross_entry_squared"], label="loss TMSV cross square")
    require(rl["faithful_tmsv_frobenius_change"] > 0.23, "loss faithful reference insensitive")

    pd = prod["rank_deficient_dark_mode"]
    rd = ref["rank_deficient_dark_mode"]
    close_sequence(pd["accumulated_diffusion_gramian_Y"], rd["diffusion_y"], label="dark-mode Y")
    require(pd["gramian_rank"] == rd["diffusion_rank"] == 2, "dark-mode rank changed")
    require(pd["dark_kernel_dimension"] == rd["dark_kernel_dimension"] == 2, "dark kernel dimension changed")
    require(not pd["no_return_asserted_on_dark_kernel"] and rd["declared_dark_subspace_is_noiseless"], "dark kernel overclaimed")
    require(not rd["full_two_mode_reference_complete_identity"], "full dark fixture return promoted")
    close(rd["dark_kernel_residual"], 0.0, absolute=1e-15, label="dark kernel residual")

    pf = prod["finite_environment_recurrence"]
    rf = ref["finite_environment_recurrence"]
    require(not pf["cp_divisible_markov_diffusion_model"] and not rf["cp_divisible_markov_scope"], "finite environment entered Markov scope")
    require(pf["full_recurrence"]["joint_heisenberg_return"] and rf["four_quarter_joint_recurrence_is_identity"], "finite recurrence failed")
    require(rf["quarter_turn_reduced_map_erases_input"] and rf["half_turn_reduced_map_recovers_input_dependence"], "finite-memory revival control failed")
    require(pf["outside_positive_accumulated_diffusion_theorem_scope"], "finite recurrence scope flag lost")
    require(not pf["intermediate_noise_never_recoheres_claimed"], "finite recurrence overclaim")
    require(len(rf["steps"]) == 5, "finite recurrence step count changed")
    for index, step in enumerate(rf["steps"]):
        require(step["quarter_turn"] == index, "finite recurrence index changed")
        close(step["symplectic_residual"], 0.0, absolute=1e-15, label=f"finite step {index} symplecticity")
    close_sequence(rf["steps"][4]["joint_symplectic"], ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)), absolute=0.0, label="finite recurrence endpoint")


def direct_shadow_and_scaling_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    prod = production["fixtures"]
    ref = reference["fixtures"]

    pc = prod["common_nontrivial_bus_evolution"]
    rc = ref["common_nontrivial_bus_evolution"]
    require(pc["client_labels"] == 2 and rc["client_labels"] == [0, 1], "common fixture labels changed")
    require(rc["all_sector_maps_equal"] and pc["branch_relative_bus_action_identity"], "common branch maps differ")
    for map_index in range(2):
        close_sequence(pc["sector_bus_symplectics"][map_index], rc["common_bus_symplectic"], label=f"common map {map_index}")
    close(pc["sector_lifted_phases_pi_units"][0], rc["client_lifted_phases_radians"][0], label="common phase zero")
    close(number(pc["sector_lifted_phases_pi_units"][1]) * math.pi, rc["client_lifted_phases_radians"][1], label="common phase pi/3")
    require(pc["client_direct_shadow_exists"] and rc["factorized_common_bus_and_client_diagonal"], "common direct shadow lost")
    require(not pc["bus_endpoint_is_identity"] and not rc["common_map_is_identity"], "common bus return promoted")
    require(not pc["carrier_restoration"] and not rc["carrier_reference_complete_identity"], "common restoration promoted")
    require(rc["direct_client_diagonal_nontrivial"] and not rc["client_channel_is_identity"], "common client phase erased")
    close(complex_value(rc["direct_client_diagonal"][1][1]).real, 0.5, label="common diagonal real")
    close(complex_value(rc["direct_client_diagonal"][1][1]).imag, math.sqrt(3.0) / 2.0, label="common diagonal imag")

    ps = prod["declared_environment_schur"]
    rs = ref["declared_environment_schur"]
    close(ps["environment_overlap_kappa"], rs["kappa"], label="Schur kappa")
    close_sequence(ps["client_schur_kernel"], rs["declared_environment_overlap_kernel"], label="Schur kernel")
    require(ps["kernel_rank"] == 2 and rs["positive_semidefinite"] and rs["trace_preserving"], "Schur channel invalid")
    close(ps["plus_client_output_purity"], rs["uniform_client_output_purity"], label="Schur purity")
    close(ps["plus_client_output_purity"], 17.0 / 25.0, label="Schur exact purity")
    require(rs["plus_output_purity_exact"] == "17_OVER_25", "Schur exact receipt changed")
    require(not ps["sector_triples_determine_cross_branch_channel"], "sector triples overclaimed")
    require(ps["declared_common_dilation_overlap_data_required"], "common dilation data not charged")
    require(not ps["strong_carrier_environment_return"] and ps["carrier_reference_identity"], "carrier/environment return distinction lost")

    pg = prod["sector_scaling"]
    rg = ref["sector_scaling"]
    q_values = [1, 2, 4, 8, 12]
    l_values = [2, 4, 16, 256, 4096]
    l2_values = [4, 16, 256, 65536, 16777216]
    q2_values = [1, 4, 16, 64, 144]
    require(len(pg["samples"]) == len(rg["rows"]) == 5, "scaling row count changed")
    for index, (prow, rrow) in enumerate(zip(pg["samples"], rg["rows"], strict=True)):
        require(prow["q_client_bits"] == rrow["q"] == q_values[index], "q scaling changed")
        require(prow["L_joint_labels"] == rrow["L"] == l_values[index], "L scaling changed")
        require(prow["general_diagonal_phase_entries"] == rrow["generic_explicit_sector_map_entries"] == l_values[index], "diagonal L cost changed")
        require(prow["general_declared_schur_entries"] == rrow["generic_explicit_schur_kernel_entries"] == rrow["L_squared"] == l2_values[index], "Schur L^2 cost changed")
        require(
            prow["affine_label_coarse_o_q_squared_count"]
            == rrow["affine_weyl_force_compiled_q_squared_upper_bound"]
            == rrow["affine_weyl_force_pair_query_q_squared_upper_bound"]
            == q2_values[index],
            "affine-Weyl q^2 corollary changed",
        )
        require(prow["display_public_segment_count_K"] == 4, "display K changed")
        require(prow["display_carrier_modes_M"] == 1, "display M changed")
        require(
            prow["display_total_input_descriptor_scalars"]
            == prow["display_common_symplectic_descriptor_scalars"]
            + prow["display_affine_force_descriptor_scalars"],
            "display input accounting changed",
        )
    require(pg["coarse_o_q_squared_counts"] == q2_values, "production q^2 receipt changed")
    require(not pg["blanket_polynomial_classical_efficiency_claimed"], "blanket polynomial claim promoted")
    require(pg["arbitrary_phase_or_environment_kernel_may_require_exponential_label_data"], "generic exponential descriptor cost hidden")

    pcoro = pg["affine_weyl_force_corollary"]
    rcoro = rg["affine_weyl_force_corollary"]
    phyp = pcoro["hypotheses"]
    rhyp = rcoro["hypotheses"]
    require(
        set(pcoro)
        == {
            "K_assumed_constant_in_scaling", "application_work_charged",
            "compilation_work_charged", "compiled_phase_coefficient_scalars",
            "conclusion", "degree_bound_reason", "dense_compilation_arithmetic_upper_bound",
            "fails_if_displacement_not_closed_for_every_label",
            "fails_if_force_law_has_label_degree_above_1",
            "fails_if_label_dependent_quadratic_generator_present", "hypotheses",
            "input_descriptor_scalars", "per_label_application_arithmetic",
            "public_segment_count_K_charged", "status",
        },
        "production affine-Weyl corollary keys changed",
    )
    require(
        set(rcoro)
        == {
            "application_work_charged", "compilation_work_charged",
            "compiled_phase_descriptor_scaling", "dense_L_or_L_squared_materialization_required",
            "dense_compilation_arithmetic_upper_bound", "derivation",
            "fixed_public_K_is_still_charged", "hypotheses",
            "input_descriptor_scaling", "per_label_application_arithmetic",
            "phase_polynomial_degree_upper_bound", "scope_exclusion",
        },
        "reference affine-Weyl corollary keys changed",
    )
    require(
        set(phyp)
        == {
            "final_displacement_closed_for_every_label",
            "finite_carrier_modes_M",
            "finite_commuting_binary_labels_Q",
            "label_dependent_quadratic_generator_absent",
            "phase_source_is_bilinear_weyl_cocycle",
            "public_finite_segment_structure_K",
            "quadratic_or_symplectic_propagation_G_K_or_S_K_is_common",
            "quadratic_or_symplectic_propagation_is_label_independent",
            "segment_force_is_label_affine",
            "segment_force_law",
        },
        "production affine-Weyl hypothesis keys changed",
    )
    require(
        set(rhyp)
        == {
            "K_assumed_constant_across_scaling_family",
            "carrier_mode_count_symbol",
            "client_labels",
            "common_quadratic_generator_is_label_independent",
            "common_symplectic_propagation",
            "endpoint_displacement_closed_for_every_label",
            "label_dependent_quadratic_generator",
            "segment_count_is_charged",
            "segment_count_is_finite_per_program",
            "segment_count_is_public",
            "segment_count_symbol",
            "segment_force_law",
        },
        "reference affine-Weyl hypothesis keys changed",
    )
    for key in set(phyp) - {"segment_force_law"}:
        require(phyp[key] is True, f"production affine-Weyl hypothesis failed: {key}")
    require(phyp["segment_force_law"] == "V_K(Z)=V_K0+SUM_I_Z_I*V_KI", "production affine-force law changed")
    require(rhyp["segment_force_law"] == "V_K_Z=V_K0_PLUS_SUM_I_Z_I_V_KI", "reference affine-force law changed")
    require(rhyp["common_symplectic_propagation"] == "S_K_OR_G_K_INDEPENDENT_OF_Z", "reference common symplectic law changed")
    require(rhyp["common_quadratic_generator_is_label_independent"], "reference common quadratic law lost")
    require(not rhyp["label_dependent_quadratic_generator"], "reference label-dependent quadratic generator admitted")
    require(rhyp["endpoint_displacement_closed_for_every_label"], "reference all-label closure lost")
    require(rhyp["segment_count_is_public"] and rhyp["segment_count_is_charged"], "reference K not public and charged")
    require(rhyp["segment_count_is_finite_per_program"], "reference finite-per-program K lost")
    require(not rhyp["K_assumed_constant_across_scaling_family"], "reference K hidden as scaling constant")
    require(rhyp["segment_count_symbol"] == "K" and rhyp["carrier_mode_count_symbol"] == "M", "reference symbols changed")

    require(pcoro["input_descriptor_scalars"] == "O(K*(M^2+Q*M))", "production affine input cost changed")
    require(rcoro["input_descriptor_scaling"] == "O(K*(M^2+q*M))", "reference affine input cost changed")
    require(pcoro["compiled_phase_coefficient_scalars"] == "O(Q^2)", "production compiled q^2 cost changed")
    require(rcoro["compiled_phase_descriptor_scaling"] == "O(q^2)", "reference compiled q^2 cost changed")
    require(pcoro["per_label_application_arithmetic"] == "O(Q^2)", "production apply cost changed")
    require(rcoro["per_label_application_arithmetic"] == "O(q^2)", "reference apply cost changed")
    require(not rcoro["dense_L_or_L_squared_materialization_required"], "reference dense materialization promoted")
    require(rcoro["phase_polynomial_degree_upper_bound"] == 2, "reference phase degree changed")
    require(pcoro["public_segment_count_K_charged"] and rcoro["fixed_public_K_is_still_charged"], "K charge lost")
    require(not pcoro["K_assumed_constant_in_scaling"], "production K hidden as scaling constant")
    require(pcoro["compilation_work_charged"] and pcoro["application_work_charged"], "production compile/apply work not charged")
    require(rcoro["compilation_work_charged"] and rcoro["application_work_charged"], "reference compile/apply work not charged")
    require(pcoro["dense_compilation_arithmetic_upper_bound"] == "O(K*M^3+K*Q*M^2+K*Q^2*M)", "production compile cost changed")
    require(rcoro["dense_compilation_arithmetic_upper_bound"] == "O(K*M^3+K*q*M^2+K*q^2*M)", "reference compile cost changed")
    require(pcoro["status"] == "FAIL_CLOSED_UNLESS_EVERY_HYPOTHESIS_IS_TRUE", "production corollary no longer fail-closed")
    require(pcoro["fails_if_displacement_not_closed_for_every_label"], "production closure failure gate lost")
    require(pcoro["fails_if_force_law_has_label_degree_above_1"], "production affine-force degree gate lost")
    require(pcoro["fails_if_label_dependent_quadratic_generator_present"], "production quadratic-generator gate lost")

    named = pg["named_corollary_parameters"]
    require(named == {
        "K": 4,
        "M": 1,
        "Q": 2,
        "compiled_coarse_q_squared_coefficient_count": 4,
        "exact_nonzero_quadratic_phase_coefficients": 1,
        "exact_phase_polynomial_radians": "-Z0*Z1/6",
        "input_descriptor_scalar_upper_count": 40,
    }, "named affine-Weyl corollary parameters changed")


def resources_nonclaims_and_scope_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    scope = production["scope_exclusions"]
    expected_scope = {
        "noncommuting_client_axes",
        "nonquadratic_or_non_gaussian_interactions",
        "non_gaussian_boundary_measurements",
        "measurement_feedback_or_qec",
        "restricted_or_exogenous_access",
        "nonmarkov_environment_recoherence",
        "infinite_mode_limits",
        "dark_or_noiseless_subspaces_not_on_declared_support",
        "physical_carrier_custody",
    }
    require(set(scope) == expected_scope and all(scope.values()), "production scope exclusions changed")
    reference_exclusions = set(reference["class_theorem"]["excluded_or_separately_scoped_cases"])
    require(
        reference_exclusions
        == {
            "DARK_OR_NOISELESS_SUBSPACE_OUTSIDE_DECLARED_NOISY_SUPPORT",
            "FINITE_ENVIRONMENT_NONMARKOV_RECOHERENCE_OR_ENVIRONMENT_REVERSAL",
            "NONCOMMUTING_CLIENT_AXES",
            "NONQUADRATIC_OR_NON_GAUSSIAN_INTERACTIONS",
            "NON_GAUSSIAN_BOUNDARY_MEASUREMENTS_OR_QEC",
            "RESTRICTED_OR_EXOGENOUS_ACCESS",
            "INFINITE_MODE_LIMITS",
        },
        "reference theorem exclusions changed",
    )

    negative = production["negative_claims"]
    require(
        set(negative)
        == {
            "physical_execution", "physical_same_backing_custody", "physical_restoration",
            "same_backing_software_execution", "qemu_device_execution",
            "all_noise_changes_all_states", "intermediate_noise_never_recoheres",
            "blanket_classical_polynomial_efficiency", "complexity_lower_bound",
            "resource_advantage", "unbounded_compute",
        }
        and not any(negative.values()),
        "production nonclaims changed",
    )
    claims = reference["claims"]
    require(
        set(claims)
        == {
            "formal_reference_complete_gaussian_identity_criterion",
            "positive_markov_diffusion_no_return_on_declared_support",
            "general_direct_client_shadow_exists",
            "generic_direct_shadow_is_polynomially_compact",
            "restricted_affine_weyl_force_common_quadratic_propagation_corollary_is_polynomially_compact",
            "unrestricted_affine_label_direct_shadow_is_polynomially_compact",
            "executed_carrier_restoration", "same_backing_reuse", "physical_execution",
            "physical_carrier_custody", "physical_restoration", "computational_advantage",
            "m257_escape", "unbounded_compute", "bit_replaced_with_pi",
        },
        "reference claim-key set changed",
    )
    for key in (
        "formal_reference_complete_gaussian_identity_criterion",
        "positive_markov_diffusion_no_return_on_declared_support",
        "general_direct_client_shadow_exists",
        "restricted_affine_weyl_force_common_quadratic_propagation_corollary_is_polynomially_compact",
    ):
        require(claims[key], f"reference positive bounded claim lost: {key}")
    for key in set(claims) - {
        "formal_reference_complete_gaussian_identity_criterion",
        "positive_markov_diffusion_no_return_on_declared_support",
        "general_direct_client_shadow_exists",
        "restricted_affine_weyl_force_common_quadratic_propagation_corollary_is_polynomially_compact",
    }:
        require(not claims[key], f"reference forbidden claim promoted: {key}")

    authority = reference["architecture_authority"]
    require(
        set(authority)
        == {
            "reference_is_physical_evidence", "reference_executes_restoration",
            "reference_asserts_same_backing_identity", "generic_sector_shadow_cost_is_explicit",
            "q_squared_compactness_requires_affine_weyl_forces_common_label_independent_quadratic_propagation_fixed_public_charged_K_and_closed_displacement",
            "m257_equal_access_guardrail_remains_intact",
        },
        "reference architecture-authority keys changed",
    )
    require(not authority["reference_is_physical_evidence"], "reference promoted to physical evidence")
    require(not authority["reference_executes_restoration"], "reference promoted to restoration")
    require(not authority["reference_asserts_same_backing_identity"], "reference promoted to same backing")
    require(authority["generic_sector_shadow_cost_is_explicit"], "generic shadow cost hidden")
    require(
        authority[
            "q_squared_compactness_requires_affine_weyl_forces_common_label_independent_quadratic_propagation_fixed_public_charged_K_and_closed_displacement"
        ],
        "affine-Weyl scope lost",
    )
    require(authority["m257_equal_access_guardrail_remains_intact"], "reference weakened M257")

    comparator = production["strongest_honest_classical_comparator"]
    require(
        set(comparator)
        == {
            "general_reference_complete_closed_sector_case", "general_sector_descriptor_cost",
            "affine_weyl_force_compiled_phase_coefficients",
            "affine_weyl_force_corollary_hypotheses",
            "affine_weyl_force_dense_compilation_arithmetic_upper_bound",
            "affine_weyl_force_input_descriptor_cost",
            "affine_weyl_force_per_label_application_arithmetic",
            "public_segment_count_K_charged", "bus_modes_retained",
            "restoration_stage_executed", "arbitrary_client_phase_function_proved_easy",
            "m257_escape_established",
        },
        "comparator keys changed",
    )
    require(comparator["bus_modes_retained"] == 0, "comparator retained bus modes")
    require(not comparator["restoration_stage_executed"], "comparator executed restoration")
    require(not comparator["arbitrary_client_phase_function_proved_easy"], "arbitrary client phase promoted")
    require(not comparator["m257_escape_established"], "M257 escape promoted")
    require(comparator["general_sector_descriptor_cost"] == "EXPLICIT_L_OR_L_SQUARED_NO_BLANKET_COMPACTNESS", "comparator descriptor cost changed")
    require(comparator["affine_weyl_force_input_descriptor_cost"] == "O(K*(M^2+Q*M))", "affine comparator input law changed")
    require(comparator["affine_weyl_force_compiled_phase_coefficients"] == "O(Q^2)", "affine comparator compiled law changed")
    require(comparator["affine_weyl_force_per_label_application_arithmetic"] == "O(Q^2)", "affine comparator apply law changed")
    require(comparator["affine_weyl_force_dense_compilation_arithmetic_upper_bound"] == "O(K*M^3+K*Q*M^2+K*Q^2*M)", "affine comparator compile law changed")
    require(comparator["public_segment_count_K_charged"], "comparator hid K")
    require(
        comparator["affine_weyl_force_corollary_hypotheses"]
        == "V_K(Z)=V_K0+SUM_I_Z_I*V_KI;_COMMON_LABEL_INDEPENDENT_G_K_OR_S_K;_PUBLIC_K;_CLOSED_DISPLACEMENT;_NO_LABEL_DEPENDENT_QUADRATIC_GENERATOR",
        "affine comparator hypotheses changed",
    )

    ledger = production["resource_ledger"]
    require(ledger["formal_carrier_modes"] == 1 and ledger["largest_control_carrier_modes"] == 2, "formal mode ledger changed")
    require(ledger["executed_physical_modes"] == 0, "physical mode execution invented")
    require(ledger["joint_client_label_scaling_samples_q"] == [1, 2, 4, 8, 12], "q ledger changed")
    require(ledger["retained_dynamic_trajectory_history"] == 0, "hidden trajectory history appeared")
    require(ledger["maximum_exact_fixture_denominator"] == 128 and ledger["maximum_exact_fixture_denominator_bits"] == 8, "precision ledger changed")
    require(ledger["largest_named_lifted_winding_turns"] == 2 and ledger["largest_named_lifted_winding_descriptor_bits"] == 2, "lift ledger changed")
    require(ledger["physical_energy_joules"] == ledger["bandwidth_hz"] == ledger["latency_s"] == "UNINSTANTIATED", "physical resource invented")
    close(ledger["tmsv_mean_quanta_per_mode"], 1.0 / 8.0, label="ledger TMSV energy")
    close(ledger["additive_diffusion_added_mean_quanta"], 1.0 / 8.0, label="ledger additive occupation")
    require(ledger["affine_weyl_corollary_public_segment_count_K"] == 4, "ledger affine K changed")
    require(ledger["affine_weyl_corollary_input_descriptor_scalar_upper_count"] == 40, "ledger affine input count changed")
    require(ledger["affine_weyl_corollary_compiled_coarse_q_squared_count"] == 4, "ledger affine compiled count changed")
    require(ledger["affine_weyl_corollary_compilation_work_charged"], "ledger compile cost lost")
    require(ledger["affine_weyl_corollary_application_work_charged"], "ledger apply cost lost")
    require(not ledger["affine_weyl_corollary_K_assumed_constant_in_scaling"], "ledger K hidden as constant")


def seal_audit(production_bytes: bytes, reference_bytes: bytes, write: bool) -> None:
    production_seal = seal_bytes(production_bytes)
    reference_seal = seal_bytes(reference_bytes)
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
    parser.add_argument("--write-seals", action="store_true")
    arguments = parser.parse_args()

    source_and_document_audit()
    production_bytes = regenerate(PRODUCTION)
    reference_bytes = regenerate(REFERENCE)
    require(regenerate(PRODUCTION) == production_bytes, "production regeneration is nondeterministic")
    require(regenerate(REFERENCE) == reference_bytes, "reference regeneration is nondeterministic")
    production = json.loads(production_bytes)
    reference = json.loads(reference_bytes)

    metadata_audit(production, reference)
    lifted_and_reference_fixture_audit(production, reference)
    noise_and_scope_fixture_audit(production, reference)
    direct_shadow_and_scaling_audit(production, reference)
    resources_nonclaims_and_scope_audit(production, reference)
    seal_audit(production_bytes, reference_bytes, arguments.write_seals)

    print(
        "PASS_STRICT_SCOPE M267_CONDITIONAL_GAUSSIAN_CLASS_OBSTRUCTION "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "RESTORATION=NO_RESTORATION_CLAIM "
        "SCOPE=REFERENCE_COMPLETE_CHANNEL_AND_DIFFUSION_SUPPORT "
        "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
