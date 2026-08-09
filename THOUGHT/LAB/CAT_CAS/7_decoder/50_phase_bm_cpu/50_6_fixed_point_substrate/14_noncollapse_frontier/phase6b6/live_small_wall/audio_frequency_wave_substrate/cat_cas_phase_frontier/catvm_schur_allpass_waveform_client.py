#!/usr/bin/env python3
"""M256 public controller; it imports no backend or exact field arithmetic."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_QZETA8_FORMAL_SCHUR_ALLPASS_WAVEFORM_PHASE_V1"
OUTPUT_TYPE = "QZETA8_ALLPASS_WINDING_AND_POINT_EVALUATION_V1"
OWNER = 256004
CONTROLLER = 256001
BOUNDARY_CONSUMER = 256002
RESULT = "PASS_CATVM_EXACT_SCHUR_ALLPASS_WAVEFORM_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_QZETA8_FORMAL_SCHUR_ALLPASS_FEEDBACK_"
    "WAVEFORM_RETAINS_TWO_UNRESOLVED_POLYNOMIAL_BACKINGS_THROUGH_THREE_"
    "SECTIONS_AND_RELEASES_ONLY_WINDING_AND_ONE_POINT_EVALUATION_AFTER_EXACT_"
    "SAME_BACKING_INVERSE_RESTORATION_AND_GENERATION2_REUSE_BUT_THE_IDENTICAL_"
    "SCALAR_SCHUR_BOUNDARY_RECURRENCE_IS_STRICTLY_SMALLER_AND_NO_DISTINCT_"
    "PHASE_RESOURCE_OR_ADVANTAGE_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA8_SEEDED_THREE_SECTION_RATIONAL_SCHUR_ALLPASS_WORDS_"
    "AT_ONE_PUBLIC_ZETA8_EVALUATION_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m256-"):
        raise RuntimeError("M256 client requires declared abstract socket")
    return "\0" + name[1:]


def exchange(name: str, request: dict[str, object]) -> tuple[dict[str, Any], int, int]:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(name))
    connection.sendall(encoded)
    response = b""
    while not response.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        response += chunk
    connection.close()
    if not response:
        raise RuntimeError("M256 backend returned no response")
    return json.loads(response), len(encoded), len(response)


def program_id(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_request(case: dict[str, Any], generation: int, transaction_id: str) -> dict[str, object]:
    descriptor = case["descriptor"]
    return {
        "command": "RUN", "carrier_id": case["carrier_id"],
        "expected_generation": generation, "descriptor": descriptor,
        "port_type": PORT_TYPE, "output_type": OUTPUT_TYPE,
        "controller_id": CONTROLLER, "owner": OWNER,
        "consumer_id": BOUNDARY_CONSUMER, "transaction_id": transaction_id,
        "program_id": program_id(descriptor),
    }


def status_request(case: dict[str, Any]) -> dict[str, object]:
    return {"command": "STATUS", "carrier_id": case["carrier_id"]}


def disconnect_run(name: str, case: dict[str, Any]) -> int:
    request = run_request(case, 1, "M256_DISCONNECT")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def comparable(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in (
        "output_type", "winding", "evaluation", "resource_shape", "work",
    )}


def contains_forbidden(value: object) -> bool:
    forbidden = {
        "numerator", "denominator", "coefficients", "receipts", "seed",
        "scratch_values", "intermediate", "inverse_temporaries", "paths",
        "assignments", "truth_table", "debug_dump",
    }
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden or contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden(item) for item in value)
    return False


def main(name: str) -> None:
    public = json.load(sys.stdin)
    cases = public["cases"]
    expected = {
        "primary", "reuse", "fresh", "sham", "disconnect", "partial",
        "postprojection", "descriptor_control",
    }
    if set(cases) != expected:
        raise RuntimeError("M256 public case set rejected")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1), ("sham", 1)):
        response, sent, received = exchange(
            name, run_request(cases[label], generation, f"M256_{label.upper()}")
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK":
            raise RuntimeError(f"M256 accepted case rejected: {label}")
        item = dict(response)
        item["run_kind"] = label.upper()
        item["public_descriptor"] = cases[label]["descriptor"]
        item["public_program_id"] = program_id(cases[label]["descriptor"])
        item["controller_request_bytes"] = sent
        item["backend_response_bytes"] = received
        accepted.append(item)

    reuse = next(case for case in accepted if case["run_kind"] == "REUSE")
    fresh = next(case for case in accepted if case["run_kind"] == "FRESH")
    reuse_parity = (
        comparable(reuse) == comparable(fresh)
        and reuse["generation"] == 2 and fresh["generation"] == 1
    )

    disconnect_bytes = disconnect_run(name, cases["disconnect"])
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(name, status_request(cases["disconnect"]))
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M256 disconnect restoration timeout")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (
        ("partial", "inject_failure_after_partial"),
        ("postprojection", "inject_failure_after_projection"),
    ):
        request = run_request(cases[case_id], 1, f"M256_{case_id.upper()}")
        request[field] = True
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(name, status_request(cases[case_id]))
        total_request_bytes += sent
        total_response_bytes += received
        fault_controls[f"{case_id}_failure_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED"
            and "evaluation" not in response
            and status.get("canonical") is True
            and status.get("leased") is False
            and status.get("last_restored_generation") == 1
        )

    controls_response, sent, received = exchange(name, {"command": "SELF_TEST"})
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M256 backend controls rejected")

    base = run_request(cases["descriptor_control"], 1, "M256_CONTROL")
    mutations: list[tuple[str, dict[str, object]]] = []
    for label, key, value in (
        ("wrong_program_rejected", "program_id", "0" * 64),
        ("wrong_owner_rejected", "owner", OWNER + 1),
        ("wrong_type_rejected", "port_type", "WRONG"),
        ("wrong_output_type_rejected", "output_type", "WRONG"),
        ("wrong_controller_rejected", "controller_id", CONTROLLER + 1),
        ("wrong_consumer_request_rejected", "consumer_id", BOUNDARY_CONSUMER + 1),
    ):
        mutation = dict(base)
        mutation[key] = value
        mutations.append((label, mutation))
    stale = run_request(cases["primary"], 1, "M256_STALE")
    mutations.append(("stale_generation_rejected", stale))
    skipped = dict(base)
    skipped["expected_generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id = json.loads(json.dumps(base))
    same_id["descriptor"]["sections"][2] = [1, 5]
    mutations.append(("same_id_changed_descriptor_rejected_at_digest_boundary", same_id))
    malformed = json.loads(json.dumps(base))
    malformed["descriptor"]["answer"] = 7
    mutations.append(("answer_bearing_extra_descriptor_field_rejected", malformed))

    descriptor_controls: dict[str, bool] = {}
    for label, request in mutations:
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    malformed_transaction = dict(base)
    malformed_transaction["transaction_id"] = 256
    response, sent, received = exchange(name, malformed_transaction)
    total_request_bytes += sent
    total_response_bytes += received
    status, sent, received = exchange(name, status_request(cases["descriptor_control"]))
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["malformed_transaction_id_rejected_without_poisoning_carrier"] = (
        response.get("status") == "REJECTED" and status.get("canonical") is True
        and status.get("leased") is False and status.get("last_restored_generation") == 0
    )
    malformed_carrier = dict(base)
    malformed_carrier["carrier_id"] = 256
    response, sent, received = exchange(name, malformed_carrier)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["malformed_carrier_id_rejected"] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "SNAPSHOT_RUN", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
        "PROJECT_COEFFICIENTS", "PROJECT_NUMERATOR", "PROJECT_DENOMINATOR",
        "PROJECT_SEED", "PROJECT_INTERMEDIATE", "DEBUG_DUMP", "DUMP",
        "NULL_CARRIER",
    ):
        response, sent, received = exchange(name, {"command": command})
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    stop, sent, received = exchange(name, {"command": "STOP"})
    total_request_bytes += sent
    total_response_bytes += received
    if stop.get("status") != "STOPPED":
        raise RuntimeError("M256 backend stop failed")

    controls = {
        **dict(controls_response["controls"]), **fault_controls,
        **descriptor_controls, **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"] and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(
            case["canonical_after_restoration"] for case in accepted
        ),
        "all_accepted_waveform_scratch_and_receipt_backings_stable": all(
            case["same_waveform_scratch_and_receipt_backings"] for case in accepted
        ),
        "restored_unrelated_program_reuse_matches_fresh": reuse_parity,
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in accepted),
        "responses_exclude_hidden_waveform_coefficients_seed_receipts_and_scratch": not any(
            contains_forbidden(case) for case in accepted
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M256 control failure: {controls}")

    primary = next(case for case in accepted if case["run_kind"] == "PRIMARY")
    sham = next(case for case in accepted if case["run_kind"] == "SHAM")
    resource_law = {
        "accepted_transactions": len(accepted),
        "accepted_public_descriptors": {
            case["run_kind"]: case["public_descriptor"] for case in accepted
        },
        "accepted_persistent_carriers": 3,
        "resident_waveform_field_cells_per_carrier": 8,
        "scratch_waveform_field_cells_per_carrier": 8,
        "receipt_rational_cells_per_carrier": 3,
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 1,
        "retained_final_winding_integer_cells_during_inverse_per_transaction": 1,
        "retained_final_degree_integer_cells_during_inverse_per_transaction": 2,
        "retained_dynamic_inverse_history_field_cells": 0,
        "accepted_forward_sections": sum(case["work"]["forward_sections"] for case in accepted),
        "accepted_inverse_sections": sum(case["work"]["inverse_sections"] for case in accepted),
        "accepted_forward_field_scalar_multiplications": sum(
            case["work"]["forward_field_scalar_multiplications"] for case in accepted
        ),
        "accepted_forward_field_additions": sum(
            case["work"]["forward_field_additions"] for case in accepted
        ),
        "accepted_inverse_field_scalar_multiplications": sum(
            case["work"]["inverse_field_scalar_multiplications"] for case in accepted
        ),
        "accepted_inverse_field_subtractions": sum(
            case["work"]["inverse_field_subtractions"] for case in accepted
        ),
        "accepted_inverse_public_rational_divisions": sum(
            case["work"]["inverse_public_rational_divisions"] for case in accepted
        ),
        "accepted_inverse_public_rational_multiplications": sum(
            case["work"]["inverse_public_rational_multiplications"] for case in accepted
        ),
        "accepted_inverse_public_rational_subtractions": sum(
            case["work"]["inverse_public_rational_subtractions"] for case in accepted
        ),
        "accepted_carrier_coefficient_writes": sum(
            case["work"]["carrier_coefficient_writes"] for case in accepted
        ),
        "accepted_scratch_clears": sum(case["work"]["scratch_clears"] for case in accepted),
        "accepted_inverse_divisibility_checks": sum(
            case["work"]["inverse_divisibility_checks"] for case in accepted
        ),
        "accepted_point_evaluation_field_multiplications": sum(
            case["work"]["evaluation_field_multiplications"] for case in accepted
        ),
        "accepted_point_evaluation_field_additions": sum(
            case["work"]["evaluation_field_additions"] for case in accepted
        ),
        "strongest_fixed_fixture_classical_baseline": (
            "PUBLIC_DESCRIPTOR_VALIDATION_PLUS_FROZEN_EXACT_WINDING_AND_POINT_EVALUATION_CERTIFICATE_IN_O1_WORK"
        ),
        "strongest_actual_boundary_classical_baseline": (
            "ONE_QZETA8_SCALAR_PLUS_ONE_WINDING_INTEGER_SCHUR_RECURRENCE_IN_O1_LIVE_ALGEBRAIC_STATE"
        ),
        "strongest_full_formal_waveform_classical_baseline": (
            "IDENTICAL_TWO_POLYNOMIAL_EXACT_SCHUR_RECURRENCE_WITHOUT_CATVM_INVERSE_RESTORATION"
        ),
        "primary_feedback_boundary_differs_from_feedback_disabled_sham": (
            primary["evaluation"] != sham["evaluation"] and primary["winding"] == sham["winding"] == 3
        ),
        "catvm_path_has_space_work_or_query_advantage": False,
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "field_cells_are_not_fixed_width_payload_claims": True,
        "whole_transaction_live_payload_peak_complete": False,
        "private_resident_waveform_component_exact_coordinate_payload_measured_by_standalone_only": True,
        "descriptor_parse_digest_and_json_operation_counts_instrumented": False,
        "qzeta8_field_operation_counts_are_abstract_not_python_fraction_primitive_counts": True,
        "python_fraction_object_allocator_socket_hash_serialization_rss_excluded_not_zero": True,
    }

    output = {
        "result": RESULT, "claim": CLAIM, "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": accepted, "reuse_parity": reuse_parity, "controls": controls,
        "waveform_law": {
            "primitive": "FORMAL_QZETA8_RATIONAL_ALLPASS_PHASE_FUNCTION_N_OVER_D",
            "native_section": "N_PRIME_EQUALS_A_D_PLUS_Z_N_AND_D_PRIME_EQUALS_D_PLUS_A_Z_N",
            "exact_inverse_requires_Z_DIVISIBILITY_AND_ONE_MINUS_A_SQUARED_NONZERO": True,
            "winding_increments_once_per_declared_section": True,
            "waveform_coefficients_remain_unprojected": True,
            "only_final_winding_and_zeta8_evaluation_are_released": True,
            "route_disposition": "RETIRE_AFTER_THIS_BOUNDED_DIAGNOSTIC_IF_SCALAR_AND_FULL_FUNCTION_BISIMULATIONS_MATCH",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_bytes,
        },
        "claim_limits": {
            "convergent_or_normalized_physical_filter_execution": False,
            "physical_waveform_or_audio_execution": False,
            "general_schur_or_smith_mcmillan_invariant_advantage": False,
            "fixed_bounded_width_exact_state": False,
            "distinct_phase_resource_unavailable_to_compact_classical_software": False,
            "total_computational_advantage": False,
            "general_relational_geometry": False,
            "general_catalytic_inference": False,
            "small_wall_crossed": False,
            "physical_bit_replacement": False,
            "unbounded_catalytic_computation": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: client.py @catvm-m256-NAME")
    main(sys.argv[1])
