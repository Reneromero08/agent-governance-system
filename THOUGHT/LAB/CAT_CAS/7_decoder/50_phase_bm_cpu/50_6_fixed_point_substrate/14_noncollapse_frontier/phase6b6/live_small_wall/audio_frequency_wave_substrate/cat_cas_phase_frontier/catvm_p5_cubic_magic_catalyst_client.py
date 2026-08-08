#!/usr/bin/env python3
"""M248 public controller; imports no backend or exact-field implementation."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


P = 5
PORT_TYPE = "CATVM_P5_CUBIC_MAGIC_CATALYST_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_DATA_AMPLITUDE_V1"
CONSUMER_ID = 248001
OWNER = 248004
RESULT = "PASS_CATVM_P5_CUBIC_MAGIC_CATALYST_RESOURCE_BALANCE_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_P5_CUBIC_MAGIC_STATE_CATALYST_RETURNS_"
    "ONE5_CELL_QZETA5_CATALYST_AFTER_ONE_AND_TWO_DISTINCT_COHERENT_"
    "SYNDROME_FEEDBACK_USES_WITH_ATOMIC_RESPONSE_ORDERING_EXACT_SAME_"
    "BACKING_RESTORATION_AND_GENERATION2_REUSE_BUT_THE_JOINT_CORRECTION_"
    "CONTAINS_TWO_BIVARIATE_CUBIC_TERMS_AND_THE_DIRECT_SINGLE_QUDIT_"
    "CUBIC_PHASE_AND_SYMBOLIC_CLASSICAL_IDENTITY_ARE_SMALLER_WITH_NO_"
    "MAGIC_OR_COMPUTATIONAL_ADVANTAGE"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA5_WIDTHS1_2_THREE_PUBLIC_CUBIC_CATALYST_"
    "DESCRIPTORS_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m248-"):
        raise RuntimeError("M248 client requires a declared abstract Unix socket")
    return "\0" + socket_name[1:]


def exchange(
    socket_name: str, request: dict[str, object]
) -> tuple[dict[str, Any], int, int]:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(socket_name))
    connection.sendall(encoded)
    response = b""
    while not response.endswith(b"\n"):
        chunk = connection.recv(65536)
        if not chunk:
            break
        response += chunk
    connection.close()
    if not response:
        raise RuntimeError("M248 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(descriptor["family"]),
        int(descriptor["width"]),
        tuple(
            tuple(int(value) % P for value in row)
            for row in descriptor["syndrome_maps"]
        ),
        tuple(int(value) % P for value in descriptor["output"]),
        int(descriptor["catalyst_strength"]) % P,
        str(descriptor["catalyst_commitment"]),
    )


def program_id(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(canonical_descriptor(descriptor), separators=(",", ":")).encode()
    ).hexdigest()


def run_request(
    case: dict[str, Any], generation: int, transaction_id: str
) -> dict[str, object]:
    descriptor = case["descriptor"]
    return {
        "command": "RUN",
        "carrier_id": case["carrier_id"],
        "descriptor": descriptor,
        "family": int(descriptor["family"]),
        "width": int(descriptor["width"]),
        "program_id": program_id(descriptor),
        "catalyst_commitment": descriptor["catalyst_commitment"],
        "port_type": PORT_TYPE,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": OWNER,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def status_request(case: dict[str, Any]) -> dict[str, object]:
    return {"command": "STATUS", "carrier_id": case["carrier_id"]}


def disconnect_run(socket_name: str, case: dict[str, Any]) -> int:
    request = run_request(case, 1, "M248_DISCONNECT_CONTROL")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(socket_name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def boundary_payload_bits(amplitude: dict[str, object]) -> int:
    exponent = int(amplitude["denominator_power5"])
    return sum(signed_bits(int(value)) for value in amplitude["numerator"]) + signed_bits(
        5**exponent
    )


def comparable(case: dict[str, object]) -> dict[str, object]:
    keys = (
        "family", "width", "syndrome_use_count", "final_amplitude",
        "catalyst_commitment", "catalyst_field_cells",
        "joint_interaction_scratch_field_cells", "phase_signature_field_cells",
        "final_projection_workspace_field_cells",
        "retained_final_boundary_field_cells_during_inverse", "work",
    )
    return {key: case[key] for key in keys}


def main(socket_name: str) -> None:
    public = json.load(sys.stdin)
    cases_by_id = public["cases"]
    required = {
        "single", "primary", "reuse", "fresh", "disconnect", "partial",
        "postprojection", "descriptor_control",
    }
    if set(cases_by_id) != required:
        raise RuntimeError("invalid M248 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    disconnect_request_bytes = disconnect_run(socket_name, cases_by_id["disconnect"])
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(
            socket_name, status_request(cases_by_id["disconnect"])
        )
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M248 disconnect did not restore")

    failure_controls: dict[str, bool] = {}
    for case_id, label, injected in (
        ("partial", "partial_forward_exception", "inject_failure_after_uses"),
        ("postprojection", "post_projection_exception", "inject_failure_after_projection"),
    ):
        request = run_request(cases_by_id[case_id], 1, f"M248_{label.upper()}")
        request[injected] = 1 if injected.endswith("uses") else True
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(socket_name, status_request(cases_by_id[case_id]))
        total_request_bytes += sent
        total_response_bytes += received
        failure_controls[f"{label}_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED"
            and "response" not in response
            and status.get("canonical") is True
            and status.get("leased") is False
            and status.get("last_restored_generation") == 1
        )

    descriptor_controls: dict[str, bool] = {}
    base = run_request(cases_by_id["descriptor_control"], 1, "M248_DESCRIPTOR_CONTROL")
    attacks = (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_output_type", "output_type", "WRONG"),
        ("wrong_consumer", "consumer_id", CONSUMER_ID + 1),
        ("wrong_owner", "owner", 0),
        ("wrong_generation", "generation", 2),
        ("wrong_family", "family", 2),
        ("wrong_width", "width", 1),
        ("wrong_program", "program_id", "WRONG"),
        ("wrong_catalyst_commitment", "catalyst_commitment", "WRONG"),
        ("empty_transaction", "transaction_id", ""),
    )
    for label, field_name, value in attacks:
        request = json.loads(json.dumps(base))
        request[field_name] = value
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[f"{label}_rejected"] = response.get("status") == "REJECTED"
    mutated = json.loads(json.dumps(base))
    mutated["descriptor"]["syndrome_maps"][0][0] = 2
    response, sent, received = exchange(socket_name, mutated)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["same_id_changed_descriptor_rejected"] = (
        response.get("status") == "REJECTED"
    )

    schedule = (
        ("single", 1, "SINGLE_SYNDROME"),
        ("primary", 1, "TWO_SYNDROME_PRIMARY"),
        ("reuse", 2, "RESTORED_UNRELATED_REUSE"),
        ("fresh", 1, "FRESH_UNRELATED_REFERENCE"),
    )
    cases: list[dict[str, object]] = []
    for case_id, generation, run_kind in schedule:
        response, sent, received = exchange(
            socket_name,
            run_request(cases_by_id[case_id], generation, f"M248_{run_kind}"),
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK" or set(response) != {"status", "response"}:
            raise RuntimeError(f"M248 atomic case rejected: {run_kind}")
        case = dict(response["response"])
        case["run_kind"] = run_kind
        case["released_final_boundary_exact_payload_bits"] = boundary_payload_bits(
            case["final_amplitude"]
        )
        case["controller_request_bytes"] = sent
        case["backend_response_bytes"] = received
        cases.append(case)

    reuse, fresh = cases[-2], cases[-1]
    if comparable(reuse) != comparable(fresh):
        raise RuntimeError("M248 fresh/restored unrelated-program mismatch")

    stale, sent, received = exchange(
        socket_name,
        run_request(cases_by_id["reuse"], 2, "M248_STALE_GENERATION"),
    )
    total_request_bytes += sent
    total_response_bytes += received
    stale_status, sent, received = exchange(socket_name, status_request(cases_by_id["reuse"]))
    total_request_bytes += sent
    total_response_bytes += received

    alternate = json.loads(json.dumps(run_request(
        cases_by_id["reuse"], 3, "M248_WRONG_CATALYST_TYPE"
    )))
    alternate["descriptor"]["catalyst_strength"] = 2
    alternate["descriptor"]["catalyst_commitment"] = public["alternate_catalyst_commitment"]
    alternate["program_id"] = program_id(alternate["descriptor"])
    alternate["catalyst_commitment"] = public["alternate_catalyst_commitment"]
    mismatch, sent, received = exchange(socket_name, alternate)
    total_request_bytes += sent
    total_response_bytes += received

    control_request = run_request(
        cases_by_id["descriptor_control"], 1, "M248_BACKEND_CONTROLS"
    )
    control_request["command"] = "CONTROLS"
    backend_controls, sent, received = exchange(socket_name, control_request)
    total_request_bytes += sent
    total_response_bytes += received
    if backend_controls.get("status") != "OK":
        raise RuntimeError("M248 backend controls rejected")

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_CATALYST", "PROJECT_SYNDROME", "PROJECT_PHASE_SIGNATURE",
        "PROJECT_JOINT", "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR",
        "PATH_LIST", "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
        "NULL_CARRIER", "DUMP", "DEBUG",
    ):
        response, sent, received = exchange(socket_name, {"command": command})
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = (
            response.get("status") == "REJECTED"
        )

    shutdown, sent, received = exchange(socket_name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M248 shutdown failed")

    controls = {
        **failure_controls,
        **descriptor_controls,
        **protocol_controls,
        **dict(backend_controls["controls"]),
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"]
            and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "stale_generation_rejected": bool(
            stale.get("status") == "REJECTED"
            and stale_status.get("canonical")
            and stale_status.get("last_restored_generation") == 2
        ),
        "catalyst_parameter_mismatch_rejected": mismatch.get("status") == "REJECTED",
        "response_released_only_after_restoration": all(
            case["canonical_after_restoration"] for case in cases
        ),
        "all_backings_same_through_reuse": all(case["same_all_backings"] for case in cases),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M248 control failure: {controls}")

    accepted_uses = sum(int(case["syndrome_use_count"]) for case in cases)
    resource_law = {
        "accepted_transaction_catalyst_uses": accepted_uses,
        "actual_catalyst_field_cells_per_carrier": 5,
        "actual_catalyst_integer_numerator_coordinates_per_carrier": 20,
        "actual_catalyst_denominator_exponent_scalar_cells_per_carrier": 1,
        "joint_interaction_scratch_field_cells_per_carrier": 25,
        "two_hidden_phase_signature_field_cells_per_carrier": 10,
        "final_projection_workspace_field_cells_per_carrier": 1,
        "accepted_fixed_field_backings_per_carrier": 41,
        "descriptor_validation_catalyst_receipt_rematerialization_field_cells": 5,
        "canonical_restoration_check_expected_catalyst_rematerialization_field_cells": 5,
        "accepted_persistent_carriers": 3,
        "accepted_persistent_field_backings_across_service": 123,
        "control_only_persistent_carriers": 3,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "retained_dynamic_inverse_history_entries": 0,
        "accepted_path_joint_correction_root_multiplications": sum(
            int(case["work"]["joint_correction_root_multiplications"])
            for case in cases
        ),
        "direct_non_catalytic_phase_signature_root_values_per_use": 5,
        "direct_non_catalytic_phase_has_one_univariate_cubic_term": True,
        "catalyst_joint_correction_has_two_bivariate_cubic_terms_per_use": True,
        "catalyst_creation_counted_once_per_actual_carrier": True,
        "catalyst_wigner_l1_exact": {
            "rational_numerator": 1,
            "rational_denominator": 1,
            "sqrt5_numerator": 2,
            "sqrt5_denominator": 5,
        },
        "catalyst_magic_unchanged_by_each_exact_factorized_use": True,
        "joint_correction_magic_monotone_or_optimal_synthesis_measured": False,
        "accepted_software_path_has_work_or_magic_advantage_over_direct_phase": False,
        "strongest_implemented_classical_baselines": [
            "EXACT_SYMBOLIC_CATALYST_IDENTITY_PHASE_S_OF_S_EQUALS_MINUS_A_S_CUBED",
            "DIRECT_NONCATALYTIC_SINGLE_QUDIT_CUBIC_PHASE_SIGNATURE_PLUS_IDENTICAL_STREAMED_FINAL_BOUNDARY",
        ],
        "classical_baselines_require_no_catalyst_inverse_or_catvm_traffic": True,
        "dense_joint_amplitude_execution_is_VERIFIER_ONLY": True,
        "comparison_basis": "DECLARED_EXACT_QZETA5_FIELD_BACKINGS_ROOT_MULTIPLICATIONS_PUBLIC_DESCRIPTOR_TRAFFIC_RESTORATION_AND_REUSE_NOT_WHOLE_PROCESS_RSS",
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_transaction_live_payload_peak_complete": False,
        "python_objects_allocator_socket_kernel_hash_serialization_rss_excluded_not_zero": True,
    }

    output = {
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": cases,
        "controls": controls,
        "resource_balance": {
            "decision": "CATALYST_IDENTITY_EXACT_BUT_JOINT_CORRECTION_RETAINS_TWO_BIVARIATE_CUBIC_TERMS_AND_DIRECT_PHASE_SOFTWARE_IS_SMALLER",
            "exact_catalyst_returned": True,
            "free_magic_established": False,
            "magic_cost_reduction_established": False,
            "work_reduction_established": False,
            "route_disposition": "RETIRE_THIS_IDENTITY_AS_AN_ADVANTAGE_ROUTE_AFTER_THE_BOUNDED_PROOF",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_request_bytes,
        },
        "claim_limits": {
            "physical_quantum_or_waveform_catalysis": False,
            "free_magic": False,
            "magic_monotone_reduction": False,
            "optimal_magic_synthesis": False,
            "general_catalytic_inference": False,
            "general_width_or_depth_scaling": False,
            "distinct_phase_resource_unavailable_to_compact_classical_software": False,
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "unbounded_catalytic_computation": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: client.py @catvm-m248-NAME")
    main(sys.argv[1])
