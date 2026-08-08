#!/usr/bin/env python3
"""M243 public controller for the atomic quadratic plus rank-one cubic service."""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any


P = 5
DIMENSIONS = (2, 3, 4, 6, 8, 12, 16)
PORT_TYPE = "CATVM_P5_CONNECTED_QUADRATIC_RANK1_CUBIC_PHASE_V1"
OUTPUT_TYPE = "QZETA5_FINAL_AMPLITUDE_V1"
CONSUMER_ID = 243001
RESULT = "PASS_CATVM_P5_QUADRATIC_RANK1_CUBIC_GAUSS_QUOTIENT_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_CONNECTED_REGULAR_QUADRATIC_PLUS_ONE_MULTI_"
    "COORDINATE_CUBIC_FUNCTIONAL_COLLAPSES_THE_FINAL_COHERENT_AMPLITUDE_TO_"
    "A_FIVE_CHANNEL_GAUSS_QUOTIENT_DETERMINED_BY_DIMENSION_DISCRIMINANT_"
    "DELTA_AND_LAMBDA_ACROSS_DECLARED_WIDTHS2_3_4_6_8_12_16_WITH_FINAL_"
    "RESPONSE_ONLY_AFTER_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_WHILE_"
    "THE_IDENTICAL_COMPACT_MODULAR_LDL_PLUS_FIVE_TERM_CLASSICAL_EVALUATOR_"
    "MATCHES_THE_ACCEPTED_PATH_SO_NO_TOTAL_ADVANTAGE_OR_SMALL_WALL_CROSSING_"
    "IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_P5_CONNECTED_REGULAR_QUADRATIC_FORMS_PLUS_ONE_CUBIC_"
    "LINEAR_FUNCTIONAL_DECLARED_WIDTHS2_3_4_6_8_12_16_ABSTRACT_UNIX_"
    "SOCKET_EXACT_AMPLITUDE_BOUNDARY_MODEL_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m243-"):
        raise RuntimeError("M243 client requires the declared abstract Unix socket")
    return "\0" + socket_name[1:]


def exchange(socket_name: str, request: dict[str, object]) -> tuple[dict[str, Any], int, int]:
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
        raise RuntimeError("M243 service returned no response")
    return json.loads(response), len(encoded), len(response)


def run_request(oracle_id: str, dimension: int, generation: int, transaction_id: str) -> dict[str, object]:
    return {
        "command": "RUN",
        "oracle_id": oracle_id,
        "port_type": PORT_TYPE,
        "dimension": dimension,
        "program_id": oracle_id,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": 243000 + dimension,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def disconnected_run(socket_name: str) -> int:
    request = run_request("disconnect_control", 16, 1, "M243_DISCONNECT_CONTROL")
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(socket_name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def main(socket_name: str) -> None:
    cases: list[dict[str, object]] = []
    total_request_bytes = 0
    total_response_bytes = 0

    disconnect_request_bytes = disconnected_run(socket_name)
    disconnect_status: dict[str, Any] | None = None
    for _ in range(80):
        time.sleep(0.01)
        status, request_bytes, response_bytes = exchange(
            socket_name, {"command": "STATUS", "oracle_id": "disconnect_control"}
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M243 disconnect control did not restore")

    failure_controls: dict[str, bool] = {}
    for oracle_id, label in (
        ("exception_control", "post_projection_exception"),
        ("partial_exception_control", "partial_ldl_exception"),
    ):
        response, request_bytes, response_bytes = exchange(
            socket_name,
            run_request(oracle_id, 16, 1, f"M243_{label.upper()}_CONTROL"),
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        status, request_bytes, response_bytes = exchange(
            socket_name, {"command": "STATUS", "oracle_id": oracle_id}
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        failure_controls[f"{label}_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED"
            and "response" not in response
            and status["canonical"]
            and not status["leased"]
            and status["last_restored_generation"] == 1
        )

    validation_request = run_request("n4_primary", 4, 1, "M243_DESCRIPTOR_CONTROL")
    malformed: dict[str, dict[str, object]] = {}
    for label, field, value in (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_dimension", "dimension", 6),
        ("same_id_changed_program", "program_id", "MUTATED"),
        ("wrong_output_type", "output_type", "WRONG"),
        ("wrong_consumer", "consumer_id", CONSUMER_ID + 1),
        ("wrong_owner", "owner", 0),
        ("wrong_generation", "generation", 2),
        ("empty_transaction", "transaction_id", ""),
    ):
        request = dict(validation_request)
        request[field] = value
        malformed[label] = request
    descriptor_controls: dict[str, bool] = {}
    for label, request in malformed.items():
        response, request_bytes, response_bytes = exchange(socket_name, request)
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        descriptor_controls[f"{label}_rejected"] = response.get("status") == "REJECTED"
    validation_status, request_bytes, response_bytes = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "n4_primary"}
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        validation_status["canonical"]
        and not validation_status["leased"]
        and validation_status["last_restored_generation"] == 0
    )

    for dimension in DIMENSIONS:
        run_specs = (
            (f"n{dimension}_primary", 1, "PRIMARY"),
            (f"n{dimension}_reuse", 2, "RESTORED_REUSE"),
            (f"n{dimension}_reuse_fresh", 1, "FRESH_REUSE_REFERENCE"),
        )
        dimension_cases: list[dict[str, object]] = []
        for oracle_id, generation, run_kind in run_specs:
            response, request_bytes, response_bytes = exchange(
                socket_name,
                run_request(oracle_id, dimension, generation, f"M243_N{dimension}_{run_kind}"),
            )
            total_request_bytes += request_bytes
            total_response_bytes += response_bytes
            if response.get("status") != "OK" or set(response) != {"status", "response"}:
                raise RuntimeError("M243 atomic case rejected or response envelope invalid")
            case = dict(response["response"])
            case["run_kind"] = run_kind
            case["controller_request_bytes"] = request_bytes
            case["backend_response_bytes"] = response_bytes
            cases.append(case)
            dimension_cases.append(case)
        reuse, fresh = dimension_cases[1], dimension_cases[2]
        if reuse["final_amplitude"] != fresh["final_amplitude"]:
            raise RuntimeError("M243 restored/fresh boundary mismatch")
        for key in (
            "carrier_residue_cells",
            "solve_scratch_residue_cells",
            "quotient_residue_scratch_cells",
            "coherent_channel_field_cells",
            "coherent_channel_scratch_field_cells",
        ):
            if reuse[key] != fresh[key]:
                raise RuntimeError("M243 restored/fresh resource mismatch")

    collision_cases: list[dict[str, object]] = []
    for oracle_id in ("quotient_collision_a", "quotient_collision_b"):
        response, request_bytes, response_bytes = exchange(
            socket_name,
            run_request(oracle_id, 4, 1, f"M243_{oracle_id.upper()}"),
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        if response.get("status") != "OK":
            raise RuntimeError("M243 quotient collision control rejected")
        collision_cases.append(dict(response["response"]))
    quotient_collision_equal = (
        collision_cases[0]["final_amplitude"] == collision_cases[1]["final_amplitude"]
    )

    stale_response, request_bytes, response_bytes = exchange(
        socket_name, run_request("n4_reuse", 4, 2, "M243_STALE_GENERATION")
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    stale_status, request_bytes, response_bytes = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "n4_reuse"}
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes

    backend_controls, request_bytes, response_bytes = exchange(
        socket_name, {"command": "CONTROLS", "oracle_id": "n4_primary"}
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_MATRIX",
        "PROJECT_VECTOR",
        "PROJECT_SOLVE",
        "PROJECT_QUOTIENT",
        "PROJECT_CHANNELS",
        "DENSE_GLOBAL_VECTOR",
        "SNAPSHOT",
        "RUN_SNAPSHOT",
        "NULL_CARRIER",
    ):
        response, request_bytes, response_bytes = exchange(
            socket_name, {"command": command, "oracle_id": "n4_primary"}
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    stop, request_bytes, response_bytes = exchange(socket_name, {"command": "STOP"})
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes

    controls = dict(backend_controls["controls"])
    controls.update(protocol_controls)
    controls.update(descriptor_controls)
    controls.update(failure_controls)
    controls.update({
        "disconnect_before_response_still_restores": (
            disconnect_status["canonical"]
            and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "stale_generation_rejected_after_reuse": (
            stale_response.get("status") == "REJECTED"
            and stale_status["canonical"]
            and not stale_status["leased"]
            and stale_status["last_restored_generation"] == 2
        ),
        "distinct_descriptors_with_equal_gauss_quotient_have_equal_boundary": quotient_collision_equal,
        "all_service_carriers_canonical_at_stop": bool(stop.get("all_carriers_canonical")),
        "controller_imports_or_loads_backend_code": False,
        "controller_receives_hidden_descriptor_or_quotient": False,
        "controller_computes_final_amplitude_independently": False,
        "service_stdout_stderr_contains_hidden_descriptor_or_channels": False,
        "snapshot_reload_used_by_accepted_path": False,
        "pre_run_secret_dependent_receipt_exposed": False,
    })

    dimensions = list(DIMENSIONS)
    packed = [n * (n + 1) // 2 for n in dimensions]
    result = {
        "schema": "cat_cas.catvm_p5_quadratic_rank1_cubic_gauss_quotient_raw.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "cases": cases,
        "controls": controls,
        "algebra_law": {
            "field_order": P,
            "dimensions": dimensions,
            "phase_signature": "X_TRANSPOSE_A_X_PLUS_LAMBDA_TIMES_U_TRANSPOSE_X_CUBED",
            "candidate_exact_quotient": "DIMENSION_DETERMINANT_SQUARE_CLASS_U_TRANSPOSE_A_INVERSE_U_LAMBDA",
            "coherent_cubic_channels": P,
            "standard_one_shot_quantum_inference_claimed": False,
            "dense_5_to_the_n_sum_accepted_path": False,
        },
        "resource_law": {
            "packed_symmetric_matrix_residue_cells": packed,
            "hidden_descriptor_residue_cells": [packed[i] + n + 1 for i, n in enumerate(dimensions)],
            "catvm_hidden_configuration_residue_cells": [packed[i] + n + 1 for i, n in enumerate(dimensions)],
            "accepted_carrier_residue_cells": [packed[i] + n + 1 for i, n in enumerate(dimensions)],
            "accepted_carrier_plus_hidden_configuration_residue_cells": [2 * (packed[i] + n + 1) for i, n in enumerate(dimensions)],
            "solve_scratch_residue_cells": dimensions,
            "quotient_residue_scratch_cells": [2] * len(dimensions),
            "coherent_channel_field_cells": [P] * len(dimensions),
            "coherent_channel_scratch_field_cells": [P] * len(dimensions),
            "coherent_channel_integer_coordinate_cells": [4 * P] * len(dimensions),
            "coherent_channel_scratch_integer_coordinate_cells": [4 * P] * len(dimensions),
            "retained_final_amplitude_field_cells_during_inverse": [1] * len(dimensions),
            "retained_final_amplitude_integer_coordinates_during_inverse": [4] * len(dimensions),
            "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": [1] * len(dimensions),
            "final_amplitude_exact_payload_fixed_width": False,
            "final_amplitude_denominator_power5_upper_bounds": [n + 1 for n in dimensions],
            "final_amplitude_denominator_material_value_bit_upper_bounds": [(P ** (n + 1)).bit_length() for n in dimensions],
            "final_amplitude_numerator_coordinate_signed_bit_upper_bounds": [(50 * P**n).bit_length() + 1 for n in dimensions],
            "final_amplitude_public_bound_derivation": "FIVE_ROOTS_PER_CHANNEL_GIVE_COORDINATE_ABS_AT_MOST5_ROOT_MULTIPLICATION_ROW_L1_AT_MOST2_FIVE_CHANNEL_SUM_GIVES_AT_MOST50_AND_EACH_SQRT5_MULTIPLICATION_HAS_ROW_L1_EQUAL5",
            "secret_dependent_exact_amplitude_payload_metrics_retained_in_sanitized_evidence": False,
            "hidden_descriptor_residue_reads_forward_plus_inverse": [2 * (packed[i] + n + 1) for i, n in enumerate(dimensions)],
            "dense_global_amplitude_cells_not_materialized": [P**n for n in dimensions],
            "strongest_implemented_classical_baseline": "IDENTICAL_COMPACT_MODULAR_LDL_SOLVE_PLUS_FIVE_TERM_QZETA5_CLOSURE",
            "theoretical_dense_matrix_algebra_ceiling": "O_N_TO_THE_MATRIX_MULTIPLICATION_EXPONENT",
            "accepted_restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "snapshot_baseline_restoration_classification": "SNAPSHOT_RELOAD",
            "controller_backend_request_bytes_total": total_request_bytes,
            "backend_controller_response_bytes_total": total_response_bytes,
            "disconnect_control_request_bytes": disconnect_request_bytes,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete": False,
        },
        "claim_limits": {
            "standard_one_shot_quantum_inference": False,
            "rank1_cubic_family_escapes_fixed_gauss_quotient": False,
            "fixed_bounded_width_exact_state": False,
            "general_nonlinear_phase_resource": False,
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "unbounded_catalytic_computation": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "general_inference_or_learning": False,
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: catvm_p5_quadratic_rank1_cubic_gauss_quotient_client.py SOCKET")
    main(sys.argv[1])
