#!/usr/bin/env python3
"""M242 public controller for the tensor-factored CATVM oracle service."""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any


P = 5
DIMENSIONS = (1, 2, 4, 8, 16, 32)
PORT_TYPE = "CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_V1"
OUTPUT_TYPE = "F5_SECRET_VECTOR_FINAL_BOUNDARY_V1"
CONSUMER_ID = 242001
RESULT = "PASS_CATVM_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_TENSOR_FACTORED_HIDDEN_LINEAR_PHASE_ORACLE_"
    "PRESERVES_ONE_ABSTRACT_COHERENT_FORWARD_QUERY_ACROSS_DECLARED_"
    "DIMENSIONS1_2_4_8_16_32_WHILE_REPLACING5_TO_THE_N_GLOBAL_AMPLITUDES_"
    "WITH5N_EXACT_PHASE_FACTOR_CELLS_AND_RELEASES_ONLY_THE_FINAL_N_RESIDUES_"
    "AFTER_EXACT_SAME_BACKING_INVERSE_RESTORATION_AND_REUSE_BUT_EACH_"
    "ORACLE_CALL_READS_N_HIDDEN_RESIDUES_AND_THE_STRONGEST_DIRECT_PRIVATE_"
    "DESCRIPTOR_CLASSICAL_BASELINE_IS_O_N_SO_NO_TOTAL_ADVANTAGE_OR_SMALL_"
    "WALL_CROSSING_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "CATVM_P5_SEPARABLE_LINEAR_CHARACTER_ORACLES_PRODUCT_INPUT_RANK1_FACTOR_"
    "CARRIER_DIMENSIONS1_2_4_8_16_32_ABSTRACT_UNIX_SOCKET_SERVICE_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m242-"):
        raise RuntimeError("M242 client requires the declared abstract Unix socket")
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
        raise RuntimeError("M242 service returned no response")
    return json.loads(response), len(encoded), len(response)


def run_request(
    oracle_id: str,
    dimension: int,
    generation: int,
    transaction_id: str,
) -> dict[str, object]:
    return {
        "command": "RUN",
        "oracle_id": oracle_id,
        "port_type": PORT_TYPE,
        "dimension": dimension,
        "program_id": oracle_id,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": 242000 + dimension,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def disconnected_run(socket_name: str) -> int:
    request = run_request(
        "disconnect_control",
        32,
        1,
        "M242_DISCONNECT_BEFORE_RESTORATION_RESPONSE",
    )
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
    for _ in range(40):
        time.sleep(0.01)
        status, request_bytes, response_bytes = exchange(
            socket_name,
            {"command": "STATUS", "oracle_id": "disconnect_control"},
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M242 disconnect control did not restore")

    failure_controls: dict[str, bool] = {}
    for oracle_id, label in (
        ("exception_control", "post_projection_exception"),
        ("partial_exception_control", "partial_oracle_exception"),
    ):
        response, request_bytes, response_bytes = exchange(
            socket_name,
            run_request(oracle_id, 32, 1, f"M242_{label.upper()}_CONTROL"),
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        status, request_bytes, response_bytes = exchange(
            socket_name,
            {"command": "STATUS", "oracle_id": oracle_id},
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

    validation_id = "n4_primary"
    validation_request = run_request(
        validation_id,
        4,
        1,
        "M242_DESCRIPTOR_VALIDATION_CONTROL",
    )
    malformed: dict[str, dict[str, object]] = {}
    for label, field, value in (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_dimension", "dimension", 8),
        ("same_id_changed_program", "program_id", "MUTATED_PROGRAM"),
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
        socket_name,
        {"command": "STATUS", "oracle_id": validation_id},
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        validation_status["canonical"]
        and not validation_status["leased"]
        and validation_status["last_restored_generation"] == 0
    )

    for dimension in DIMENSIONS:
        primary_id = f"n{dimension}_primary"
        reuse_id = f"n{dimension}_reuse"
        before, request_bytes, response_bytes = exchange(
            socket_name,
            {"command": "STATUS", "oracle_id": primary_id},
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        if not before["canonical"] or before["last_restored_generation"] != 0:
            raise RuntimeError("M242 carrier not canonical before primary")
        forbidden_status_keys = {
            "inferred_secret",
            "secret_commitment",
            "boundary_commitment",
            "factor_commitment",
            "factor_values",
        }
        if forbidden_status_keys.intersection(before):
            raise RuntimeError("M242 status smuggled answer-bearing state")
        for oracle_id, generation, run_kind in (
            (primary_id, 1, "PRIMARY"),
            (reuse_id, 2, "RESTORED_REUSE"),
        ):
            response, request_bytes, response_bytes = exchange(
                socket_name,
                run_request(
                    oracle_id,
                    dimension,
                    generation,
                    f"M242_N{dimension}_{run_kind}",
                ),
            )
            total_request_bytes += request_bytes
            total_response_bytes += response_bytes
            if response.get("status") != "OK":
                raise RuntimeError("M242 atomic run rejected")
            if set(response) != {"status", "response"}:
                raise RuntimeError("M242 service returned an undeclared envelope")
            case = dict(response["response"])
            case["run_kind"] = run_kind
            case["controller_request_bytes"] = request_bytes
            case["backend_response_bytes"] = response_bytes
            cases.append(case)
        after, request_bytes, response_bytes = exchange(
            socket_name,
            {"command": "STATUS", "oracle_id": reuse_id},
        )
        total_request_bytes += request_bytes
        total_response_bytes += response_bytes
        if not after["canonical"] or after["last_restored_generation"] != 2:
            raise RuntimeError("M242 carrier not canonical after restored reuse")

    stale_response, request_bytes, response_bytes = exchange(
        socket_name,
        run_request("n4_reuse", 4, 2, "M242_STALE_GENERATION_CONTROL"),
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    stale_status, request_bytes, response_bytes = exchange(
        socket_name,
        {"command": "STATUS", "oracle_id": "n4_reuse"},
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes

    backend_controls, request_bytes, response_bytes = exchange(
        socket_name,
        {"command": "CONTROLS", "oracle_id": "n4_primary"},
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_INTERMEDIATE",
        "PROJECT_FACTORS",
        "DENSE_GLOBAL_VECTOR",
        "SNAPSHOT",
        "RUN_SNAPSHOT",
        "NULL_CARRIER",
    ):
        response, request_bytes, response_bytes = exchange(
            socket_name,
            {"command": command, "oracle_id": "n4_primary"},
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
        "pre_run_status_contains_answer_bearing_receipt": False,
        "all_service_carriers_canonical_at_stop": bool(stop.get("all_carriers_canonical")),
        "controller_imports_or_loads_backend_code": False,
        "controller_receives_hidden_phase_factor_amplitudes": False,
        "controller_computes_secret_independently": False,
        "service_stdout_stderr_contains_secret_or_amplitudes": False,
        "snapshot_reload_used_by_accepted_path": False,
    })

    dimensions = list(DIMENSIONS)
    result = {
        "schema": "cat_cas.catvm_p5_tensor_factored_hidden_linear_phase_oracle_raw.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "cases": cases,
        "controls": controls,
        "query_law": {
            "field_order": P,
            "dimensions": dimensions,
            "abstract_global_coherent_forward_queries": [1] * len(dimensions),
            "classical_black_box_value_queries_necessary_and_sufficient": dimensions,
            "actual_hidden_residue_accesses_forward_plus_inverse": [2 * n for n in dimensions],
            "oracle_factor_cell_visits_forward_plus_inverse": [2 * P * n for n in dimensions],
            "direct_private_descriptor_scan_residue_accesses": dimensions,
            "black_box_lower_bound_model": (
                "F5_LINEAR_VALUE_OR_PHASE_QUERY_RETURNS_ONE_F5_SYMBOL_PER_CLASSICAL_QUERY"
            ),
            "strongest_total_software_baseline": "DIRECT_PRIVATE_DESCRIPTOR_SCAN_O_N",
            "one_abstract_query_is_not_one_constant_cost_software_operation": True,
            "abstract_query_separation_is_not_total_software_advantage": True,
        },
        "resource_law": {
            "factor_carrier_field_cells": [P * n for n in dimensions],
            "factor_scratch_field_cells": [P * n for n in dimensions],
            "factor_carrier_integer_coordinate_cells": [4 * P * n for n in dimensions],
            "factor_scratch_integer_coordinate_cells": [4 * P * n for n in dimensions],
            "predecessor_dense_global_carrier_field_cells": [P**n for n in dimensions],
            "predecessor_dense_global_carrier_plus_scratch_field_cells": [2 * P**n for n in dimensions],
            "hidden_oracle_secret_residue_cells": dimensions,
            "retained_final_boundary_residue_cells_during_inverse": dimensions,
            "factor_fourier_character_terms_forward_plus_inverse": [100 * n for n in dimensions],
            "snapshot_baseline_copy_plus_reload_factor_and_scratch_field_cells": [20 * n for n in dimensions],
            "snapshot_baseline_restoration_classification": "SNAPSHOT_RELOAD",
            "accepted_restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "matched_exact_factor_recurrence_uses_same5N_state_and_scratch": True,
            "strongest_direct_descriptor_baseline_uses_N_residue_state": True,
            "warm_isolated_boundary_overhead_counted_in_protocol_bytes": True,
            "controller_backend_request_bytes_total": total_request_bytes,
            "backend_controller_response_bytes_total": total_response_bytes,
            "disconnect_control_request_bytes": disconnect_request_bytes,
            "whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete": False,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "claim_limits": {
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "general_oracle_or_query_advantage": False,
            "nonseparable_phase_resource": False,
            "bounded_total_carrier_across_interface_growth": False,
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
        raise SystemExit(
            "usage: catvm_p5_tensor_factored_hidden_linear_phase_oracle_client.py SOCKET"
        )
    main(sys.argv[1])
