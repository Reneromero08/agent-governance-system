#!/usr/bin/env python3
"""Independent exact reference for the hardware-disconnected V12 adapter stub.

This file deliberately imports no project module.  It proves only the internal
symbolic Q(omega) law for two fresh clients and all nine residue pairs.  It is
not evidence of QEMU execution, authentication strength, a physical adapter,
physical return, carrier custody, restoration, reuse, or resource advantage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence


SCHEMA = "M270_V12_AUTHENTICATED_ADAPTER_SEPARATE_REFERENCE_V1"
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
PHYSICAL_EVIDENCE_CLASS = "NONE"
LAYER_CLASSIFICATION = "INDEPENDENT_SYMBOLIC_REFERENCE_OUTSIDE_QEMU_EXECUTION"
TARGET_ARCHITECTURE_STIPULATION = (
    "V12_COMMON_DEVICE_EXTERNAL_ADAPTER_HARDWARE_DISCONNECTED"
)
INTEGRITY_TAG_SCOPE = "OUTSIDE_INDEPENDENT_SCIENTIFIC_PARITY"
M257_STATUS = "INTACT"

# a + b*omega is represented by the integer pair (a, b), with
# omega**2 + omega + 1 == 0.  No floating-point value participates in a
# decision or in the emitted evidence.
QOmega = tuple[int, int]
ONE: QOmega = (1, 0)
OMEGA: QOmega = (0, 1)
ZERO: QOmega = (0, 0)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"FAIL_CLOSED {message}")


def q_add(left: QOmega, right: QOmega) -> QOmega:
    return (left[0] + right[0], left[1] + right[1])


def q_mul(left: QOmega, right: QOmega) -> QOmega:
    a, b = left
    c, d = right
    return (a * c - b * d, a * d + b * c - b * d)


def q_conjugate(value: QOmega) -> QOmega:
    # conjugate(omega) = omega**2 = -1 - omega
    a, b = value
    return (a - b, -b)


def q_pow(exponent: int) -> QOmega:
    result = ONE
    for _ in range(exponent % 3):
        result = q_mul(result, OMEGA)
    return result


def q_json(value: QOmega) -> list[int]:
    return [value[0], value[1]]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def basis_label(c_a: int, c_b: int, rail: int) -> list[Any]:
    return [c_a, c_a, c_b, c_b, "a" if rail == 0 else "b", rail]


def choi_density(residue: int) -> dict[str, Any]:
    # |J_s> = (|00> + omega**s |11>)/sqrt(2).
    return {
        "basis_order": ["00", "11"],
        "denominator": 2,
        "numerators": [
            [q_json(ONE), q_json(q_pow(-residue))],
            [q_json(q_pow(residue)), q_json(ONE)],
        ],
    }


def carrier_reference_density() -> dict[str, Any]:
    # (|a,0> + |b,1>)/sqrt(2), represented on its two-dimensional support.
    return {
        "basis_order": [["a", 0], ["b", 1]],
        "denominator": 2,
        "numerators": [
            [q_json(ONE), q_json(ONE)],
            [q_json(ONE), q_json(ONE)],
        ],
    }


def density_numerator(
    s_a: int,
    s_b: int,
    ket: tuple[int, int, int],
    bra: tuple[int, int, int],
) -> QOmega:
    c_a, c_b, _rail = ket
    c_a_prime, c_b_prime, _rail_prime = bra
    return q_pow(s_a * (c_a - c_a_prime) + s_b * (c_b - c_b_prime))


def expected_factorized_numerator(
    s_a: int,
    s_b: int,
    ket: tuple[int, int, int],
    bra: tuple[int, int, int],
) -> QOmega:
    c_a, c_b, _rail = ket
    c_a_prime, c_b_prime, _rail_prime = bra
    client_a = q_mul(q_pow(s_a * c_a), q_conjugate(q_pow(s_a * c_a_prime)))
    client_b = q_mul(q_pow(s_b * c_b), q_conjugate(q_pow(s_b * c_b_prime)))
    return q_mul(q_mul(client_a, client_b), ONE)


def build_case(s_a: int, s_b: int) -> tuple[dict[str, Any], dict[str, bool]]:
    support = [
        (c_a, c_b, rail)
        for c_a in range(2)
        for c_b in range(2)
        for rail in range(2)
    ]
    amplitude_numerators = [q_pow(s_a * c_a + s_b * c_b) for c_a, c_b, _ in support]

    density_rows: list[dict[str, Any]] = []
    hermitian = True
    factorized = True
    rail_independent = True
    trace_numerator = ZERO
    for ket in support:
        for bra in support:
            numerator = density_numerator(s_a, s_b, ket, bra)
            factorized = factorized and numerator == expected_factorized_numerator(
                s_a, s_b, ket, bra
            )
            reverse = density_numerator(s_a, s_b, bra, ket)
            hermitian = hermitian and numerator == q_conjugate(reverse)
            if ket == bra:
                trace_numerator = q_add(trace_numerator, numerator)
            density_rows.append(
                {
                    "bra": basis_label(*bra),
                    "ket": basis_label(*ket),
                    "numerator": q_json(numerator),
                }
            )

    for c_a in range(2):
        for c_b in range(2):
            rail_independent = rail_independent and (
                amplitude_numerators[support.index((c_a, c_b, 0))]
                == amplitude_numerators[support.index((c_a, c_b, 1))]
            )

    direct_compiler = [
        q_pow(s_a * c_a + s_b * c_b)
        for c_a in range(2)
        for c_b in range(2)
        for _rail in range(2)
    ]
    density_digest = sha256_hex(canonical_bytes(density_rows))
    case = {
        "fixture": f"ideal_pair_{s_a}_{s_b}",
        "residues": {"a": s_a, "b": s_b},
        "formal_fresh_client_count": 2,
        "formal_query_slot_order": ["A", "B"],
        "formal_query_slot_count": 2,
        "begin_reuse_count": 0,
        "joint_density_denominator": 8,
        "support": [
            {
                "basis": basis_label(*state),
                "amplitude_numerator": q_json(amplitude),
            }
            for state, amplitude in zip(support, amplitude_numerators)
        ],
        "joint_density_numerators_sha256": density_digest,
        "client_a_choi": choi_density(s_a),
        "client_b_choi": choi_density(s_b),
        "carrier_reference_formal_factor": carrier_reference_density(),
        "internal_symbolic_complete_factorization": factorized,
        "internal_symbolic_carrier_reference_factor_unchanged": rail_independent,
        "direct_equal_access_compiler_amplitude_numerators": [
            q_json(value) for value in direct_compiler
        ],
        "direct_equal_access_compiler_parity": direct_compiler == amplitude_numerators,
        "physical_return_inference": False,
        "same_carrier_inference": False,
    }
    properties = {
        "amplitude_count_is_eight": len(amplitude_numerators) == 8,
        "density_entry_count_is_sixty_four": len(density_rows) == 64,
        "trace_is_one": trace_numerator == (8, 0),
        "hermitian": hermitian,
        "factorized": factorized,
        "rail_independent": rail_independent,
        "direct_compiler_parity": direct_compiler == amplitude_numerators,
        "two_fresh_clients": case["formal_fresh_client_count"] == 2,
        "no_begin_reuse": case["begin_reuse_count"] == 0,
    }
    return case, properties


def all_true(values: Iterable[bool]) -> bool:
    return all(values)


def main() -> None:
    checks: list[dict[str, Any]] = []

    def require(identifier: str, condition: bool) -> None:
        if not condition:
            fail(identifier)
        checks.append({"id": identifier, "pass": True})

    require(
        "ring_omega_squared_plus_omega_plus_one_is_zero",
        q_add(q_add(q_mul(OMEGA, OMEGA), OMEGA), ONE) == ZERO,
    )
    require("ring_omega_cubed_is_one", q_pow(3) == ONE)
    require("ring_conjugate_omega_is_omega_squared", q_conjugate(OMEGA) == q_pow(2))

    cases: list[dict[str, Any]] = []
    case_properties: list[dict[str, bool]] = []
    for s_a in range(3):
        for s_b in range(3):
            case, properties = build_case(s_a, s_b)
            cases.append(case)
            case_properties.append(properties)

    expected_fixture_names = [f"ideal_pair_{a}_{b}" for a in range(3) for b in range(3)]
    require("exactly_nine_residue_pairs", len(cases) == 9)
    require(
        "residue_pairs_are_lexicographically_complete",
        [case["fixture"] for case in cases] == expected_fixture_names,
    )
    for property_name in (
        "amplitude_count_is_eight",
        "density_entry_count_is_sixty_four",
        "trace_is_one",
        "hermitian",
        "factorized",
        "rail_independent",
        "direct_compiler_parity",
        "two_fresh_clients",
        "no_begin_reuse",
    ):
        require(
            f"all_nine_{property_name}",
            all_true(properties[property_name] for properties in case_properties),
        )

    unknown_physical_resources = {
        name: {
            "status": "UNKNOWN",
            "value": None,
            "unit": unit,
            "reason": "NO_PHYSICAL_ADAPTER_CONNECTED_OR_MEASURED_BY_REFERENCE",
        }
        for name, unit in (
            ("authentication_compute_energy", "J"),
            ("calibration_attempts", "count"),
            ("control_rf_energy", "J"),
            ("cryogenic_wall_energy", "J"),
            ("external_adapter_energy", "J"),
            ("external_adapter_wall_time", "ns"),
            ("physical_carrier_preparation_attempts", "count"),
            ("physical_loss_probability", "dimensionless"),
        )
    }
    unknown_resource_law_holds = all(
        entry["status"] == "UNKNOWN"
        and entry["value"] is None
        and bool(entry["reason"])
        for entry in unknown_physical_resources.values()
    )
    require("unknown_physical_resources_are_null_not_zero", unknown_resource_law_holds)

    finite_evidence_theorem = {
        "name": "FINITE_EVIDENCE_DOES_NOT_ENTAIL_EXACT_PHYSICAL_RETURN",
        "domain": "FINITE_STATISTICAL_OR_APPROXIMATE_PHYSICAL_EVIDENCE",
        "argument_classification": (
            "SYMBOLIC_COUNTERMODEL_ARGUMENT_NOT_NUMERICAL_CONVERGENCE_TEST"
        ),
        "premises_machine_derived": False,
        "countermodel_family": (
            "FOR_EACH_INTEGER_N_AT_LEAST_2_MIX_THE_IDEAL_MAP_WITH_WEIGHT_1_MINUS_1_OVER_N_"
            "AND_A_NONRETURN_MAP_WITH_WEIGHT_1_OVER_N"
        ),
        "every_finite_n_is_nonexact": True,
        "countermodels_converge_to_ideal_without_becoming_ideal": True,
        "conclusion": (
            "PASSING_ANY_POSITIVE_TOLERANCE_OR_FINITE_STATISTICAL_TEST_CANNOT_BY_ITSELF_"
            "ESTABLISH_EXACT_PHYSICAL_RETURN_OR_SAME_CARRIER_IDENTITY"
        ),
        "threshold_used_by_reference": None,
    }
    require(
        "finite_evidence_theorem_denies_exact_physical_return",
        finite_evidence_theorem["every_finite_n_is_nonexact"]
        and finite_evidence_theorem["countermodels_converge_to_ideal_without_becoming_ideal"]
        and finite_evidence_theorem["threshold_used_by_reference"] is None,
    )

    classification = {
        "phase_qemu_layer_classification": LAYER_CLASSIFICATION,
        "stipulated_target_architecture": TARGET_ARCHITECTURE_STIPULATION,
        "target_architecture_independently_verified_by_reference": False,
        "common_guest_visible_device_contract_exercised_by_reference": False,
        "qemu_execution_performed_by_reference": False,
        "external_hardware_connected": False,
        "algebra_evidence_class": "EXACT_SYMBOLIC_CYCLOTOMIC_IDENTITY",
        "physical_evidence_class": PHYSICAL_EVIDENCE_CLASS,
        "allowed_future_physical_receipt_classes": ["APPROX_MODEL", "STATISTICAL_ONLY"],
        "physical_evidence_present": False,
        "physical_evidence_claim": False,
        "physical_measurement_performed": False,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "physical_restoration_claim": False,
        "same_carrier_claim": False,
        "carrier_custody_claim": False,
    }
    require(
        "reference_makes_no_restoration_claim",
        classification["restoration_classification"] == "NO_RESTORATION_CLAIM"
        and not classification["physical_restoration_claim"],
    )
    require(
        "reference_makes_no_qemu_execution_claim",
        not classification["qemu_execution_performed_by_reference"]
        and not classification["common_guest_visible_device_contract_exercised_by_reference"],
    )
    require(
        "reference_makes_no_same_carrier_or_custody_claim",
        not classification["same_carrier_claim"] and not classification["carrier_custody_claim"],
    )
    require(
        "reference_has_no_physical_evidence",
        classification["physical_evidence_class"] == "NONE"
        and classification["allowed_future_physical_receipt_classes"]
        == ["APPROX_MODEL", "STATISTICAL_ONLY"]
        and not classification["physical_evidence_present"]
        and not classification["physical_evidence_claim"]
        and not classification["physical_measurement_performed"],
    )

    integrity_tag = {
        "width_bits": 21,
        "deterministic": True,
        "scope": INTEGRITY_TAG_SCOPE,
        "computed_or_verified_by_reference": False,
        "included_in_independent_scientific_parity": False,
        "cryptographic_authentication_claim": False,
        "collision_resistance_claim": False,
        "unforgeability_claim": False,
    }
    require(
        "twenty_one_bit_integrity_tag_is_outside_scientific_parity",
        integrity_tag["width_bits"] == 21
        and integrity_tag["scope"] == "OUTSIDE_INDEPENDENT_SCIENTIFIC_PARITY"
        and not integrity_tag["computed_or_verified_by_reference"]
        and not integrity_tag["included_in_independent_scientific_parity"],
    )
    require(
        "reference_makes_no_cryptographic_claim",
        not integrity_tag["cryptographic_authentication_claim"]
        and not integrity_tag["collision_resistance_claim"]
        and not integrity_tag["unforgeability_claim"],
    )

    equal_access_comparator = {
        "name": "DIRECT_SECRET_CONTROLLED_PHASE_COMPILER",
        "formal_secret_residue_accesses_per_pair": 2,
        "adapter_formal_secret_residue_accesses_per_pair": 2,
        "all_nine_symbolic_outputs_identical": all(
            case["direct_equal_access_compiler_parity"] for case in cases
        ),
        "unique_query_advantage": False,
        "total_resource_advantage": "UNDETERMINED",
        "resource_advantage_claim": False,
    }
    require(
        "direct_compiler_has_equal_residue_access",
        equal_access_comparator["formal_secret_residue_accesses_per_pair"]
        == equal_access_comparator["adapter_formal_secret_residue_accesses_per_pair"],
    )
    require(
        "direct_compiler_matches_all_nine_symbolic_outputs",
        equal_access_comparator["all_nine_symbolic_outputs_identical"],
    )
    require(
        "reference_makes_no_query_or_resource_advantage_claim",
        not equal_access_comparator["unique_query_advantage"]
        and not equal_access_comparator["resource_advantage_claim"]
        and equal_access_comparator["total_resource_advantage"] == "UNDETERMINED",
    )

    promotion_gate = {
        "mechanism_twins_can_kill": True,
        "mechanism_twins_can_nominate": True,
        "nomination_is_architecture_promotion": False,
        "eligible_for_architecture_promotion": False,
        "architecture_promotion": False,
        "required_for_promotion": [
            "INDEPENDENTLY_QUALIFIED_COMMON_COMPILED_BACKEND",
            "REAL_EXTERNAL_ADAPTER_STATISTICAL_VALIDATION",
        ],
        "requirements_satisfied_by_this_reference": [],
    }
    require(
        "mechanism_twin_can_kill_or_nominate_but_not_promote",
        promotion_gate["mechanism_twins_can_kill"]
        and promotion_gate["mechanism_twins_can_nominate"]
        and not promotion_gate["nomination_is_architecture_promotion"]
        and not promotion_gate["eligible_for_architecture_promotion"]
        and not promotion_gate["architecture_promotion"],
    )
    require(
        "promotion_requires_compiled_backend_and_real_adapter_statistics",
        promotion_gate["required_for_promotion"]
        == [
            "INDEPENDENTLY_QUALIFIED_COMMON_COMPILED_BACKEND",
            "REAL_EXTERNAL_ADAPTER_STATISTICAL_VALIDATION",
        ]
        and promotion_gate["requirements_satisfied_by_this_reference"] == [],
    )

    m257 = {"status": M257_STATUS, "escape_established": False}
    require("m257_remains_intact", m257["status"] == "INTACT" and not m257["escape_established"])

    nonclaims = [
        "NO_ADAPTER_HARDWARE_EXECUTION_CLAIM",
        "NO_ARCHITECTURE_PROMOTION_CLAIM",
        "NO_CARRIER_CUSTODY_CLAIM",
        "NO_CRYPTOGRAPHIC_AUTHENTICATION_CLAIM",
        "NO_EXACT_PHYSICAL_EVIDENCE_CLAIM",
        "NO_M257_ESCAPE_CLAIM",
        "NO_PHYSICAL_EVIDENCE_CLAIM",
        "NO_PHYSICAL_RESOURCE_ADVANTAGE_CLAIM",
        "NO_PHYSICAL_RETURN_CLAIM",
        "NO_QEMU_EXECUTION_CLAIM",
        "NO_QUERY_ADVANTAGE_CLAIM",
        "NO_RESTORATION_CLAIM",
        "NO_SAME_CARRIER_CLAIM",
    ]
    require("nonclaims_are_unique_and_sorted", nonclaims == sorted(set(nonclaims)))

    # Check count is included only after all fail-closed checks have executed.
    payload = {
        "authority": {
            "scope": "NINE_EXACT_INTERNAL_SYMBOLIC_RESIDUE_PAIRS_ONLY",
            "restoration_classification": RESTORATION_CLASSIFICATION,
            "physical_evidence_class": "NONE",
            "future_physical_receipt_ceiling": "APPROX_MODEL_OR_STATISTICAL_ONLY",
        },
        "classification": classification,
        "cyclotomic_model": {
            "dimension": 3,
            "relation": "omega^2+omega+1=0",
            "representation": "a_plus_b_omega_as_two_integers",
            "floating_point_decision_count": 0,
            "client_dimensions": [2, 2],
            "client_reference_dimensions": [2, 2],
            "carrier_dimension": 3,
            "carrier_reference_dimension": 2,
            "formal_joint_dimension": 96,
            "prepared_support_component_count": 8,
            "prepared_density_denominator": 8,
        },
        "ideal_pairs": cases,
        "finite_evidence_theorem": finite_evidence_theorem,
        "integrity_tag": integrity_tag,
        "equal_access_comparator": equal_access_comparator,
        "resource_accounting": {
            "known_formal_counts": {
                "residue_pairs": 9,
                "fresh_clients_per_pair": 2,
                "formal_query_slots_per_pair": 2,
                "formal_query_slots_total": 18,
                "begin_reuse": 0,
            },
            "unknown_physical_resources": unknown_physical_resources,
            "unknown_value_encoding": "NULL_NEVER_NUMERIC_ZERO",
            "unknown_physical_total_precludes_advantage_claim": True,
            "production_resident_or_transient_backing_measured": False,
        },
        "promotion_gate": promotion_gate,
        "m257": m257,
        "nonclaims": nonclaims,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_pass": True,
    }
    require_count_before_output = len(checks)
    if payload["check_count"] != require_count_before_output:
        fail("check_count_changed_before_output")

    payload_hash = sha256_hex(canonical_bytes(payload))
    source_hash = sha256_hex(Path(__file__).read_bytes())
    document = {
        "schema": SCHEMA,
        "source_sha256": source_hash,
        "payload_sha256": payload_hash,
        "payload": payload,
    }
    print(canonical_bytes(document).decode("ascii"))


if __name__ == "__main__":
    main()
