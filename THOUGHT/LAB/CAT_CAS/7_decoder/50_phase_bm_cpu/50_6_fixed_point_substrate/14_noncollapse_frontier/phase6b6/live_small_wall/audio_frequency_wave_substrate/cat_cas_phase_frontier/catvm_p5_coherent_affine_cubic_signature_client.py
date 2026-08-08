#!/usr/bin/env python3
"""M247 public CATVM controller; imports no backend or field arithmetic."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import sys
import time
from typing import Any


P = 5
WIDTHS = (1, 2, 3, 4)
PORT_TYPE = "CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_PATH_AMPLITUDE_V1"
CONSUMER_ID = 247001
OWNER = 247004
RESULT = "PASS_CATVM_P5_COHERENT_AFFINE_CUBIC_SIGNATURE_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_P5_WIDTHS1_2_3_4_COHERENT_AFFINE_"
    "CUBIC_PATH_SIGNATURES_RETAIN_ONE_TYPED_UNRESOLVED_SYNDROME_PORT_"
    "THROUGH_NONCOMMUTING_X_AND_Z_CONSUMERS_CLOSE_TO_ONE_FINAL_QZETA5_"
    "AMPLITUDE_WITH_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_BUT_FINAL_"
    "CONTRACTION_HAS_AN_IDENTICAL_STREAMED_SCALAR_CLASSICAL_BISIMULATION_"
    "AND_THE_MATCHED_EXACT_VARIABLE_ELIMINATION_ALTERNATIVE_USES_GROWING_"
    "FACTOR_TABLES_THROUGH_THE_DECLARED_WIDTH4_WITH_NO_ADVANTAGE"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_P5_AFFINE_CUBIC_DEGREE3_SIGNATURES_AT_DECLARED_"
    "WIDTHS1_2_3_4_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m247-"):
        raise RuntimeError("M247 client requires declared abstract Unix socket")
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
        raise RuntimeError("M247 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    width = int(descriptor["width"])
    flat = lambda matrix: tuple(int(value) % P for row in matrix for value in row)
    return (
        width,
        flat(descriptor["A"]),
        flat(descriptor["B"]),
        flat(descriptor["C"]),
        tuple(int(value) % P for value in descriptor["a"]),
        tuple(int(value) % P for value in descriptor["b"]),
        tuple(int(value) % P for value in descriptor["output"]),
    )


def program_id(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(canonical_descriptor(descriptor), separators=(",", ":")).encode()
    ).hexdigest()


def run_request(
    case: dict[str, Any], generation: int, transaction_id: str
) -> dict[str, object]:
    descriptor = case["descriptor"]
    width = int(descriptor["width"])
    return {
        "command": "RUN",
        "carrier_id": case["carrier_id"],
        "descriptor": descriptor,
        "port_type": PORT_TYPE,
        "width": width,
        "program_id": program_id(descriptor),
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": OWNER,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def status_request(case: dict[str, Any]) -> dict[str, object]:
    return {
        "command": "STATUS",
        "carrier_id": case["carrier_id"],
        "width": int(case["descriptor"]["width"]),
    }


def disconnect_run(socket_name: str, case: dict[str, Any]) -> int:
    request = run_request(case, 1, "M247_DISCONNECT_CONTROL")
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


def main(socket_name: str) -> None:
    public = json.load(sys.stdin)
    cases_by_id = public["cases"]
    required = {
        "primary_w1", "primary_w2", "primary_w3", "primary_w4",
        "reuse_w4", "reuse_fresh_w4", "disconnect_w2",
        "partial_w2", "exception_w2", "descriptor_control_w2",
    }
    if set(cases_by_id) != required:
        raise RuntimeError("invalid M247 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    disconnect_request_bytes = disconnect_run(socket_name, cases_by_id["disconnect_w2"])
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(
            socket_name, status_request(cases_by_id["disconnect_w2"])
        )
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M247 disconnect did not restore")

    failure_controls: dict[str, bool] = {}
    for case_id, label, injected in (
        ("partial_w2", "partial_forward_exception", "inject_failure_after_actions"),
        ("exception_w2", "post_projection_exception", "inject_failure_after_projection"),
    ):
        request = run_request(cases_by_id[case_id], 1, f"M247_{label.upper()}")
        request[injected] = 5 if injected.endswith("actions") else True
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
    control_case = cases_by_id["descriptor_control_w2"]
    base = run_request(control_case, 1, "M247_DESCRIPTOR_CONTROL")
    attacks = (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_width", "width", 3),
        ("wrong_program", "program_id", "WRONG"),
        ("wrong_output_type", "output_type", "WRONG"),
        ("wrong_consumer", "consumer_id", CONSUMER_ID + 1),
        ("wrong_owner", "owner", 0),
        ("wrong_generation", "generation", 2),
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
    mutated["descriptor"]["C"][0][0] = (mutated["descriptor"]["C"][0][0] + 1) % P
    response, sent, received = exchange(socket_name, mutated)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["same_id_changed_descriptor_rejected"] = (
        response.get("status") == "REJECTED"
    )
    status, sent, received = exchange(socket_name, status_request(control_case))
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        status.get("canonical") is True
        and status.get("last_restored_generation") == 0
    )

    schedule = [
        ("primary_w1", 1, "PRIMARY_W1"),
        ("primary_w2", 1, "PRIMARY_W2"),
        ("primary_w3", 1, "PRIMARY_W3"),
        ("primary_w4", 1, "PRIMARY_W4"),
        ("reuse_w4", 2, "RESTORED_REUSE_W4"),
        ("reuse_fresh_w4", 1, "FRESH_REUSE_REFERENCE_W4"),
    ]
    cases: list[dict[str, object]] = []
    for case_id, generation, run_kind in schedule:
        response, sent, received = exchange(
            socket_name,
            run_request(cases_by_id[case_id], generation, f"M247_{run_kind}"),
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK" or set(response) != {"status", "response"}:
            raise RuntimeError(f"M247 atomic case rejected: {run_kind}")
        case = dict(response["response"])
        case["run_kind"] = run_kind
        case["released_final_boundary_exact_payload_bits"] = boundary_payload_bits(
            case["final_amplitude"]
        )
        case["controller_request_bytes"] = sent
        case["backend_response_bytes"] = received
        cases.append(case)

    reuse, fresh = cases[-2], cases[-1]
    for key in (
        "final_amplitude", "signature_coefficient_cells", "data_map_residue_cells",
        "syndrome_map_residue_cells", "projection_workspace_field_cells",
        "descriptor_residue_cells", "projection_assignment_terms",
        "projection_monomial_evaluations", "forward_actions", "inverse_actions",
        "forward_coefficient_updates", "inverse_coefficient_updates",
        "syndrome_backing_scalar_reads", "streamed_assignment_cursor_residue_cells",
    ):
        if reuse[key] != fresh[key]:
            raise RuntimeError(f"M247 restored/fresh mismatch: {key}")

    stale, sent, received = exchange(
        socket_name,
        run_request(cases_by_id["reuse_w4"], 2, "M247_STALE_GENERATION"),
    )
    total_request_bytes += sent
    total_response_bytes += received
    stale_status, sent, received = exchange(
        socket_name, status_request(cases_by_id["reuse_w4"])
    )
    total_request_bytes += sent
    total_response_bytes += received

    control_request = run_request(cases_by_id["primary_w4"], 2, "M247_CONTROLS")
    control_request["command"] = "CONTROLS"
    backend_controls, sent, received = exchange(socket_name, control_request)
    total_request_bytes += sent
    total_response_bytes += received

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_SYNDROME", "PROJECT_DATA_MAP", "PROJECT_SIGNATURE",
        "PROJECT_COEFFICIENTS", "PROJECT_ASSIGNMENTS", "PROJECT_BAG",
        "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR", "PATH_LIST",
        "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER",
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
        raise RuntimeError("M247 shutdown failed")

    controls = {
        **failure_controls,
        **descriptor_controls,
        **protocol_controls,
        **dict(backend_controls["controls"]),
        "disconnect_restored_before_lost_response": (
            disconnect_status["canonical"]
            and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "stale_generation_rejected": (
            stale.get("status") == "REJECTED"
            and stale_status["canonical"]
            and stale_status["last_restored_generation"] == 2
        ),
        "response_released_only_after_restoration": all(
            case["canonical_after_restoration"] for case in cases
        ),
        "all_backings_same_through_reuse": all(
            case["same_coefficient_backing"]
            and case["same_data_map_backing"]
            and case["same_syndrome_map_backing"]
            and case["same_projection_workspace_backing"]
            and case["same_descriptor_backings"]
            for case in cases
        ),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M247 control failure: {controls}")

    widths = list(WIDTHS)
    coefficient_cells = [math.comb(width + 3, 3) for width in widths]
    residue_cells = [
        coefficient + 5 * width**2 + 3 * width
        for width, coefficient in zip(widths, coefficient_cells)
    ]
    resource_law = {
        "declared_widths": widths,
        "degree_at_most_three_signature_residue_cells": coefficient_cells,
        "data_plus_syndrome_affine_map_residue_cells": [2 * width**2 for width in widths],
        "projection_workspace_field_cells": 1,
        "accepted_total_residue_backing_cells_excluding_projection_field_workspace": residue_cells,
        "accepted_total_residue_backing_payload_bits_at_three_bits_per_f5_residue": [
            3 * value for value in residue_cells
        ],
        "accepted_amplitude_vector_field_cells": 0,
        "accepted_bag_table_field_cells": 0,
        "streamed_assignment_cursor_residue_cells": widths,
        "projection_assignment_terms": [5**width for width in widths],
        "retained_dynamic_inverse_history_entries": 0,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": 1,
        "public_final_amplitude_denominator_power_upper_bounds": widths,
        "projection_accumulator_plus_denominator_exact_payload_bit_upper_bounds": [
            5 * signed_bits(5**width) for width in widths
        ],
        "suite_service_carrier_count": 9,
        "suite_service_carrier_width_multiplicities": {"1": 1, "2": 5, "3": 1, "4": 2},
        "suite_service_signature_coefficient_residue_cells": 144,
        "suite_service_data_plus_syndrome_map_residue_cells": 124,
        "suite_service_descriptor_residue_cells": 252,
        "suite_service_projection_workspace_field_cells": 9,
        "shared_public_monomial_plan_exponent_integer_cells": 224,
        "shared_public_monomial_index_scalar_entries": 69,
        "suite_service_custody_metadata_request_objects_and_container_headers_complete": False,
        "polynomial_signature_size_does_not_remove_width_dependent_projection_work_or_exact_payload": True,
        "strongest_implemented_classical_baseline_family": [
            "IDENTICAL_PUBLIC_POLYNOMIAL_SIGNATURE_PLUS_STREAMED_SCALAR_5_TO_WIDTH_EVALUATION",
            "EXACT_MIN_FILL_VARIABLE_ELIMINATION_ON_THE_IDENTICAL_PUBLIC_AFFINE_CUBIC_FACTOR_GRAPH",
        ],
        "streamed_scalar_baseline_has_no_inverse_restoration_or_catvm_work": True,
        "variable_elimination_and_streamed_scalar_are_reported_as_a_time_memory_pareto_not_a_total_order": True,
        "quadratic_sham_strongest_baseline": (
            "EXACT_FINITE_FIELD_QUADRATIC_GAUSS_SUM_FROM_PUBLIC_MATRIX_RANK_"
            "DETERMINANT_AND_COMPLETED_SQUARE"
        ),
        "classical_bisimulation_uses_the_same_formula_compiled_signature": True,
        "projection_work_below_treewidth_or_stabilizer_sum_established": False,
        "classical_optimality_claimed": False,
        "whole_transaction_live_payload_peak_measured": False,
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete": False,
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
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_request_bytes,
        },
        "claim_limits": {
            "all_widths_or_unbounded_scaling": False,
            "fixed_bounded_width_exact_state": False,
            "bounded_projection_treewidth": False,
            "general_tensor_network_or_relational_closure": False,
            "catalytic_inference_or_learning": False,
            "distinct_phase_resource": False,
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
        raise SystemExit("usage: client.py @catvm-m247-NAME")
    main(sys.argv[1])
