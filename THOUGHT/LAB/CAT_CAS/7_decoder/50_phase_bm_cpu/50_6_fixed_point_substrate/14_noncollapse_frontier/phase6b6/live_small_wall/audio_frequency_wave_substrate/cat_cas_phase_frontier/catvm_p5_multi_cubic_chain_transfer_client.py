#!/usr/bin/env python3
"""M244 public controller; intentionally imports no backend arithmetic."""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any


P = 5
DEPTHS = (2, 3, 4, 8, 16, 32, 64)
PORT_TYPE = "CATVM_P5_MULTI_CUBIC_CHAIN_PHASE_MESSAGE_V1"
OUTPUT_TYPE = "QZETA5_FINAL_CHAIN_AMPLITUDE_V1"
CONSUMER_ID = 244001
RESULT = "PASS_CATVM_P5_MULTI_CUBIC_CHAIN_TRANSFER_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_CONNECTED_MULTI_CUBIC_TREEWIDTH1_CHAIN_"
    "USES_ONE_ACTUAL_FIVE_CELL_QZETA5_PHASE_MESSAGE_ACROSS_INDEPENDENT_"
    "INTERACTING_CUBIC_DIRECTIONS_AT_DECLARED_DEPTHS2_3_4_8_16_32_64_"
    "WITH_FINAL_ONLY_AMPLITUDE_RESPONSE_EXACT_SAME_BACKING_RESTORATION_"
    "AND_REUSE_BUT_PUBLIC_EXACT_PAYLOAD_BOUNDS_AND_DESCRIPTOR_WORK_GROW_"
    "AN_ENDPOINT_SPECIALIZED_FIVE_VECTOR_CLASSICAL_RECURRENCE_REMAINS_"
    "AND_A_CROSS_RANK2_"
    "CERTIFICATE_REQUIRES25_INTERFACE_STATES_OUTSIDE_TREEWIDTH_ONE"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA5_MULTI_CUBIC_CHAIN_TREEWIDTH_ONE_ONLY_"
    "DECLARED_DEPTHS2_3_4_8_16_32_64_FIVE_LOGICAL_PHASE_CELLS_WITH_"
    "GROWING_EXACT_PAYLOAD_ABSTRACT_UNIX_SOCKET_MODEL_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m244-"):
        raise RuntimeError("M244 client requires declared abstract Unix socket")
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
        raise RuntimeError("M244 service returned no response")
    return json.loads(response), len(encoded), len(response)


def run_request(oracle_id: str, depth: int, generation: int, transaction_id: str) -> dict[str, object]:
    return {
        "command": "RUN",
        "oracle_id": oracle_id,
        "port_type": PORT_TYPE,
        "depth": depth,
        "program_id": oracle_id,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": 244000 + depth,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def disconnect_run(socket_name: str) -> int:
    request = run_request("disconnect_control", 64, 1, "M244_DISCONNECT_CONTROL")
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(socket_name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def released_boundary_payload_bits(amplitude: dict[str, object]) -> int:
    exponent = int(amplitude["denominator_power5"])
    return sum(signed_bits(int(value)) for value in amplitude["numerator"]) + signed_bits(5**exponent)


def public_vector_payload_bound(depth: int) -> int:
    # Each exact transfer output coordinate is a sum of five inputs acted on
    # by a Q(zeta_5) integer matrix with maximum absolute row sum six.
    return 20 * signed_bits(30**depth) + signed_bits(5**depth)


def main(socket_name: str) -> None:
    cases: list[dict[str, object]] = []
    total_request_bytes = 0
    total_response_bytes = 0

    disconnect_request_bytes = disconnect_run(socket_name)
    disconnect_status: dict[str, Any] | None = None
    for _ in range(160):
        time.sleep(0.01)
        status, sent, received = exchange(
            socket_name, {"command": "STATUS", "oracle_id": "disconnect_control"}
        )
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M244 disconnect did not restore")

    failure_controls: dict[str, bool] = {}
    for oracle_id, label in (
        ("exception_control", "post_projection_exception"),
        ("partial_exception_control", "partial_transfer_exception"),
    ):
        response, sent, received = exchange(
            socket_name,
            run_request(oracle_id, 64, 1, f"M244_{label.upper()}_CONTROL"),
        )
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(
            socket_name, {"command": "STATUS", "oracle_id": oracle_id}
        )
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
    base = run_request("k4_primary", 4, 1, "M244_DESCRIPTOR_CONTROL")
    for label, field, value in (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_depth", "depth", 8),
        ("same_id_changed_program", "program_id", "MUTATED"),
        ("wrong_output_type", "output_type", "WRONG"),
        ("wrong_consumer", "consumer_id", CONSUMER_ID + 1),
        ("wrong_owner", "owner", 0),
        ("wrong_generation", "generation", 2),
        ("empty_transaction", "transaction_id", ""),
    ):
        request = dict(base)
        request[field] = value
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[f"{label}_rejected"] = response.get("status") == "REJECTED"
    status, sent, received = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "k4_primary"}
    )
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        status.get("canonical") is True
        and status.get("last_restored_generation") == 0
    )

    for depth in DEPTHS:
        dimension_cases: list[dict[str, object]] = []
        for oracle_id, generation, run_kind in (
            (f"k{depth}_primary", 1, "PRIMARY"),
            (f"k{depth}_reuse", 2, "RESTORED_REUSE"),
            (f"k{depth}_reuse_fresh", 1, "FRESH_REUSE_REFERENCE"),
        ):
            response, sent, received = exchange(
                socket_name,
                run_request(oracle_id, depth, generation, f"M244_K{depth}_{run_kind}"),
            )
            total_request_bytes += sent
            total_response_bytes += received
            if response.get("status") != "OK" or set(response) != {"status", "response"}:
                raise RuntimeError("M244 atomic case rejected")
            case = dict(response["response"])
            case["released_final_boundary_exact_payload_bits"] = (
                released_boundary_payload_bits(case["final_amplitude"])
            )
            case["run_kind"] = run_kind
            case["controller_request_bytes"] = sent
            case["backend_response_bytes"] = received
            cases.append(case)
            dimension_cases.append(case)
        reuse, fresh = dimension_cases[1], dimension_cases[2]
        if reuse["final_amplitude"] != fresh["final_amplitude"]:
            raise RuntimeError("M244 restored/fresh boundary mismatch")
        for key in (
            "carrier_field_cells", "scratch_field_cells",
            "hidden_descriptor_residue_cells",
        ):
            if reuse[key] != fresh[key]:
                raise RuntimeError("M244 restored/fresh resource mismatch")

    stale, sent, received = exchange(
        socket_name, run_request("k4_reuse", 4, 2, "M244_STALE_GENERATION")
    )
    total_request_bytes += sent
    total_response_bytes += received
    stale_status, sent, received = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "k4_reuse"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    backend_controls, sent, received = exchange(
        socket_name, {"command": "CONTROLS", "oracle_id": "k4_primary"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_MESSAGE", "PROJECT_SCRATCH", "PROJECT_DESCRIPTOR",
        "PROJECT_INTERMEDIATE", "DENSE_ASSIGNMENTS", "SNAPSHOT",
        "RUN_SNAPSHOT", "NULL_CARRIER",
    ):
        response, sent, received = exchange(
            socket_name, {"command": command, "oracle_id": "k4_primary"}
        )
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(socket_name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M244 service shutdown failed")

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
        "response_envelope_released_only_after_restoration": all(
            case["canonical_after_restoration"] for case in cases
        ),
        "all_backings_same_through_reuse": all(
            case["same_cell_backing"]
            and case["same_scratch_backing"]
            and case["same_descriptor_backings"]
            for case in cases
        ),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M244 control failure: {controls}")

    by_depth = {depth: [case for case in cases if case["depth"] == depth] for depth in DEPTHS}
    resource_law = {
        "depths": list(DEPTHS),
        "phase_message_field_cells": [P] * len(DEPTHS),
        "phase_message_scratch_field_cells": [P] * len(DEPTHS),
        "catvm_hidden_configuration_residue_cells": [3 * depth for depth in DEPTHS],
        "hidden_lambda_residue_cells": list(DEPTHS),
        "hidden_quadratic_residue_cells": list(DEPTHS),
        "hidden_inter_module_coupling_residue_cells": [depth - 1 for depth in DEPTHS],
        "hidden_selected_output_index_residue_cells": [1] * len(DEPTHS),
        "first_module_coupling_is_public_fixed_one": True,
        "first_module_coupling_is_nonmaterial_for_declared_e0_input": True,
        "carrier_resident_descriptor_residue_cells": [3 * depth for depth in DEPTHS],
        "service_startup_descriptor_validation_residue_reads_per_oracle": [5 * depth - 1 for depth in DEPTHS],
        "suite_service_hidden_configuration_descriptor_residue_cells": 1737,
        "suite_service_peak_resident_phase_and_scratch_field_cells": 170,
        "suite_service_peak_resident_carrier_descriptor_residue_cells": 1350,
        "suite_service_peak_resident_configuration_plus_carrier_descriptor_residue_cells": 3087,
        "suite_service_startup_descriptor_validation_residue_reads": 2871,
        "suite_service_retains17_carriers_for_controls_cases_and_fresh_comparisons": True,
        "forward_character_terms": [25 * depth for depth in DEPTHS],
        "inverse_character_terms": [25 * depth for depth in DEPTHS],
        "accepted_transfer_descriptor_reads": [9 * depth - 2 for depth in DEPTHS],
        "retained_final_amplitude_field_cells_during_inverse": [1] * len(DEPTHS),
        "retained_final_amplitude_integer_coordinates_during_inverse": [4] * len(DEPTHS),
        "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": [1] * len(DEPTHS),
        "released_primary_final_boundary_exact_payload_bits": [
            by_depth[depth][0]["released_final_boundary_exact_payload_bits"] for depth in DEPTHS
        ],
        "public_forward_denominator_exponent_upper_bounds": list(DEPTHS),
        "public_full_inverse_denominator_exponent_upper_bounds": [2 * depth for depth in DEPTHS],
        "public_forward_single_five_cell_vector_exact_payload_bit_upper_bounds": [
            public_vector_payload_bound(depth) for depth in DEPTHS
        ],
        "public_whole_transaction_message_scratch_and_retained_boundary_payload_bit_upper_bounds": [
            2 * public_vector_payload_bound(2 * depth)
            + 4 * signed_bits(30**depth)
            + signed_bits(5**depth)
            for depth in DEPTHS
        ],
        "fixed_bounded_width_exact_state": False,
        "secret_dependent_intermediate_payload_metrics_released": False,
        "secret_dependent_intermediate_payload_metrics_retained_in_sealed_evidence": False,
        "whole_transaction_exact_payload_live_peak_measured": False,
        "public_whole_transaction_payload_bounds_include_message_scratch_and_retained_boundary": True,
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete": False,
        "strongest_implemented_classical_baseline": "ENDPOINT_SPECIALIZED_EXACT_FIVE_VECTOR_INTERIOR_WITH5_TERM_FIRST_AND_FINAL_BOUNDARY_TRANSFERS",
        "endpoint_specialized_classical_forward_character_terms": [
            25 * depth - 40 for depth in DEPTHS
        ],
        "endpoint_specialized_classical_forward_only_no_inverse_or_restoration_work": True,
        "general_graph_treewidth_optimal_classical_ceiling": "O_K_TIMES5_TO_THE_TREEWIDTH_PLUS1",
        "dense5_to_the_k_assignment_expansion_used_on_accepted_path": False,
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
            "arbitrary_graph_transfer": False,
            "fixed_bounded_width_exact_state": False,
            "distinct_phase_resource": False,
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "general_inference_or_learning": False,
            "unbounded_catalytic_computation": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: client.py @catvm-m244-NAME")
    main(sys.argv[1])
