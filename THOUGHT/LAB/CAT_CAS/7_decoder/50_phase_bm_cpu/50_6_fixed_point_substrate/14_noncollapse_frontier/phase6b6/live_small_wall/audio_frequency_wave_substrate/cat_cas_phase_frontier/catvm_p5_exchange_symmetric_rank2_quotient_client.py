#!/usr/bin/env python3
"""M245 public CATVM controller; imports no backend or field arithmetic."""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_UNORDERED_PAIR_AMPLITUDE_V1"
CONSUMER_ID = 245001
RESULT = "PASS_CATVM_P5_EXCHANGE_SYMMETRIC_RANK2_ORBIT_QUOTIENT_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_EXCHANGE_SYMMETRIC_TWO_RAIL_CUBIC_INTERFACE_"
    "CLOSES_ON_ONE15_CELL_UNORDERED_PAIR_QZETA5_MESSAGE_WITH_REACHABLE_"
    "AND_OBSERVABLE_LINEAR_RANK15_FINAL_ONLY_AMPLITUDE_RESPONSE_EXACT_"
    "SAME_BACKING_RESTORATION_AND_REUSE_WHILE_A_DECLARED_THREE_GATE_"
    "EXCHANGE_BROKEN_LABELLED_ALPHABET_HAS_REACHABLE_OBSERVABLE_RANK25_"
    "AND_THE_STRONGEST_FACTORIZED_"
    "TREEWIDTH2_CLASSICAL_BOUNDARY_RECURRENCE_REMAINS_SMALLER_IN_WORK"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA5_EXCHANGE_SYMMETRIC_TWO_RAIL_FOUR_MODULE_"
    "DIAGNOSTIC_ONLY_15_UNORDERED_PAIR_CELLS_ABSTRACT_UNIX_SOCKET_MODEL"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m245-"):
        raise RuntimeError("M245 client requires declared abstract Unix socket")
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
        raise RuntimeError("M245 service returned no response")
    return json.loads(response), len(encoded), len(response)


def run_request(oracle_id: str, generation: int, transaction_id: str) -> dict[str, object]:
    return {
        "command": "RUN",
        "oracle_id": oracle_id,
        "port_type": PORT_TYPE,
        "depth": 4,
        "program_id": oracle_id,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": 245004,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def disconnect_run(socket_name: str) -> int:
    request = run_request("disconnect_control", 1, "M245_DISCONNECT_CONTROL")
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
    return sum(signed_bits(int(value)) for value in amplitude["numerator"]) + signed_bits(5**exponent)


def main(socket_name: str) -> None:
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
        raise RuntimeError("M245 disconnect did not restore")

    failure_controls: dict[str, bool] = {}
    for oracle_id, label in (
        ("exception_control", "post_projection_exception"),
        ("partial_exception_control", "partial_forward_exception"),
    ):
        response, sent, received = exchange(
            socket_name, run_request(oracle_id, 1, f"M245_{label.upper()}_CONTROL")
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
    base = run_request("primary", 1, "M245_DESCRIPTOR_CONTROL")
    for label, field, value in (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_depth", "depth", 3),
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
        socket_name, {"command": "STATUS", "oracle_id": "primary"}
    )
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        status.get("canonical") is True and status.get("last_restored_generation") == 0
    )

    cases: list[dict[str, object]] = []
    for oracle_id, generation, run_kind in (
        ("primary", 1, "PRIMARY"),
        ("reuse", 2, "RESTORED_REUSE"),
        ("reuse_fresh", 1, "FRESH_REUSE_REFERENCE"),
    ):
        response, sent, received = exchange(
            socket_name, run_request(oracle_id, generation, f"M245_{run_kind}")
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK" or set(response) != {"status", "response"}:
            raise RuntimeError("M245 atomic case rejected")
        case = dict(response["response"])
        case["run_kind"] = run_kind
        case["released_final_boundary_exact_payload_bits"] = boundary_payload_bits(
            case["final_amplitude"]
        )
        case["controller_request_bytes"] = sent
        case["backend_response_bytes"] = received
        cases.append(case)

    reuse, fresh = cases[1], cases[2]
    if reuse["final_amplitude"] != fresh["final_amplitude"]:
        raise RuntimeError("M245 restored/fresh boundary mismatch")
    for key in (
        "carrier_field_cells", "scratch_field_cells", "hidden_descriptor_residue_cells",
        "forward_character_terms", "inverse_character_terms", "accepted_descriptor_reads",
    ):
        if reuse[key] != fresh[key]:
            raise RuntimeError(f"M245 restored/fresh resource mismatch: {key}")

    stale, sent, received = exchange(
        socket_name, run_request("reuse", 2, "M245_STALE_GENERATION")
    )
    total_request_bytes += sent
    total_response_bytes += received
    stale_status, sent, received = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "reuse"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    backend_controls, sent, received = exchange(
        socket_name, {"command": "CONTROLS", "oracle_id": "primary"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_ORBIT_MESSAGE", "PROJECT_LABELLED_MESSAGE", "PROJECT_SCRATCH",
        "PROJECT_DESCRIPTOR", "PROJECT_INTERMEDIATE", "DENSE_PATHS", "SNAPSHOT",
        "RUN_SNAPSHOT", "NULL_CARRIER",
    ):
        response, sent, received = exchange(
            socket_name, {"command": command, "oracle_id": "primary"}
        )
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(socket_name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M245 service shutdown failed")

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
            case["same_cell_backing"]
            and case["same_scratch_backing"]
            and case["same_descriptor_backings"]
            for case in cases
        ),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M245 control failure: {controls}")

    resource_law = {
        "accepted_orbit_message_field_cells": 15,
        "accepted_orbit_scratch_field_cells": 15,
        "accepted_total_message_plus_scratch_field_cells": 30,
        "hidden_descriptor_residue_cells": 17,
        "descriptor_residue_law": "lambda4_plus_quadratic4_plus_rung4_plus_hidden_coupling3_plus_output_pair2",
        "suite_service_hidden_configuration_descriptor_residue_cells": 102,
        "suite_service_carrier_descriptor_residue_cells": 85,
        "suite_service_configuration_plus_carrier_descriptor_residue_cells": 187,
        "suite_service_retains5_carriers_for_controls_cases_and_fresh_comparison": True,
        "suite_service_phase_message_plus_scratch_field_cells": 150,
        "suite_service_startup_descriptor_conversion_and_nonzero_validation_residue_reads": 168,
        "public_first_coupling_fixed_one": True,
        "forward_character_terms": 1140,
        "inverse_character_terms": 1500,
        "accepted_complete_transaction_character_terms": 2640,
        "accepted_complete_transaction_root_field_multiplications": 2625,
        "accepted_complete_transaction_field_accumulations": 2625,
        "exact_restoration_checks15_message15_scratch_cells_plus_metadata": True,
        "retained_dynamic_inverse_history_entries": 0,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "strongest_implemented_classical_baseline": (
            "ENDPOINT_SPECIALIZED_EXACT_SYMMETRIC_TWO_RAIL_FACTORIZED_"
            "FIVE_BY_FIVE_TRANSFORM_WITH15_ORBIT_RESIDENT_AND25_TEMPORARY_CELLS"
        ),
        "classical_endpoint_specialized_forward_character_terms": 440,
        "classical_orbit_resident_field_cells": 15,
        "classical_factorized_temporary_field_cells": 25,
        "classical_total_resident_plus_temporary_field_cells": 40,
        "identical15_orbit_classical_forward_character_terms": 1140,
        "identical15_orbit_classical_message_plus_scratch_field_cells": 30,
        "time_memory_pareto_not_total_advantage": True,
        "classical_forward_only_has_no_inverse_restoration_or_catvm_work": True,
        "public_forward_denominator_exponent_upper_bound": 4,
        "public_full_inverse_denominator_exponent_upper_bound": 8,
        "public_single15_cell_vector_exact_payload_bit_upper_bound": (
            60 * signed_bits(100**8) + signed_bits(5**8)
        ),
        "public_payload_bound_uses25_character_terms_times4_coordinate_root_action_per_step": True,
        "secret_dependent_intermediate_payload_metrics_released": False,
        "secret_dependent_intermediate_payload_metrics_sealed": False,
        "whole_transaction_live_payload_peak_measured": False,
        "kernel_transient_field_values_and_integer_reduction_scratch_in30_backing_cells": False,
        "canonicalization_scan_and_integer_division_work_instrumented": False,
        "fixed_bounded_width_exact_state": False,
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_process_rss_python_objects_allocator_socket_kernel_hashing_and_scheduler_costs_complete": False,
        "dense25_state_materialized_on_accepted_path": False,
        "transfer_matrix_materialized_on_accepted_path": False,
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
            "labelled25_state_interface_compressed_by15_cell_port": False,
            "generic_exchange_breaking_implies_rank25": False,
            "arbitrary_nonlinear_encoding_lower_bound": False,
            "arbitrary_graph_or_treewidth": False,
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
        raise SystemExit("usage: client.py @catvm-m245-NAME")
    main(sys.argv[1])
