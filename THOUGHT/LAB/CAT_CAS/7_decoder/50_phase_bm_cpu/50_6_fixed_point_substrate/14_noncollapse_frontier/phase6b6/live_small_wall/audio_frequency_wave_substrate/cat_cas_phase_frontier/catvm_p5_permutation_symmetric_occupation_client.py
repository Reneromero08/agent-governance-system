#!/usr/bin/env python3
"""M246 public controller; imports no backend or field arithmetic."""

from __future__ import annotations

import json
import math
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_PORT_V1"
OUTPUT_TYPE = "QZETA5_FINAL_OCCUPATION_AMPLITUDE_V1"
CONSUMER_ID = 246001
OWNER = 246004
DEPTH = 3
RAILS = (2, 3, 4, 6)
RESULT = "PASS_CATVM_P5_PERMUTATION_SYMMETRIC_OCCUPATION_QUOTIENT_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_PERMUTATION_SYMMETRIC_RAILS2_3_4_6_"
    "CUBIC_OCCUPATION_INTERFACES_CLOSE_ON15_35_70_210_QZETA5_CELLS_"
    "WITH_SPLIT_PRIME_EXACT_DEPTH3_FORWARD_DESCRIPTOR_FAMILY_REACHABLE_"
    "AND_ALL_PUBLIC_OUTPUT_SELECTOR_OBSERVABLE_LINEAR_RANKS_EQUAL_TO_EACH_"
    "DECLARED_OCCUPATION_DIMENSION_FINAL_ONLY_AMPLITUDE_RESPONSE_EXACT_"
    "SAME_BACKING_RESTORATION_AND_REUSE_BUT_RANK_GROWS_AS_NPLUS4_"
    "CHOOSE4_AND_THE_MATCHED_MATRIX_FREE_OCCUPATION_CLASSICAL_"
    "RECURRENCE_USES_FEWER_DECLARED_COEFFICIENT_UPDATE_PLUS_DOT_TERMS_"
    "WHILE_TOTAL_WORK_REMAINS_UNMEASURED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QZETA5_PERMUTATION_SYMMETRIC_DEPTH3_RAILS2_3_4_6_"
    "OCCUPATION_QUOTIENT_ABSTRACT_UNIX_SOCKET_MODEL_ONLY"
)


def width_for(n: int) -> int:
    return math.comb(n + 4, 4)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m246-"):
        raise RuntimeError("M246 client requires declared abstract Unix socket")
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
        raise RuntimeError("M246 service returned no response")
    return json.loads(response), len(encoded), len(response)


def rails_for(oracle_id: str) -> int:
    if "n2" in oracle_id:
        return 2
    if "n3" in oracle_id:
        return 3
    if "n4" in oracle_id:
        return 4
    if "n6" in oracle_id:
        return 6
    raise RuntimeError("M246 oracle id lacks rail count")


def run_request(oracle_id: str, generation: int, transaction_id: str) -> dict[str, object]:
    return {
        "command": "RUN",
        "oracle_id": oracle_id,
        "port_type": PORT_TYPE,
        "rails": rails_for(oracle_id),
        "depth": DEPTH,
        "program_id": oracle_id,
        "output_type": OUTPUT_TYPE,
        "consumer_id": CONSUMER_ID,
        "owner": OWNER,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def disconnect_run(socket_name: str) -> int:
    request = run_request("disconnect_n2", 1, "M246_DISCONNECT_CONTROL")
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


def public_vector_payload_bound(n: int) -> int:
    exponent = 6 * ((n + 1) // 2)
    coordinate = (32 * (5**n)) ** 6
    return 4 * width_for(n) * signed_bits(coordinate) + signed_bits(5**exponent)


def main(socket_name: str) -> None:
    total_request_bytes = 0
    total_response_bytes = 0
    disconnect_request_bytes = disconnect_run(socket_name)
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(
            socket_name, {"command": "STATUS", "oracle_id": "disconnect_n2"}
        )
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M246 disconnect did not restore")

    failure_controls: dict[str, bool] = {}
    for oracle_id, label in (
        ("exception_n2", "post_projection_exception"),
        ("partial_exception_n2", "partial_forward_exception"),
    ):
        response, sent, received = exchange(
            socket_name, run_request(oracle_id, 1, f"M246_{label.upper()}")
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
    base = run_request("primary_n2", 1, "M246_DESCRIPTOR_CONTROL")
    for label, field_name, value in (
        ("wrong_port_type", "port_type", "WRONG"),
        ("wrong_rails", "rails", 3),
        ("wrong_depth", "depth", 4),
        ("same_id_changed_program", "program_id", "MUTATED"),
        ("wrong_output_type", "output_type", "WRONG"),
        ("wrong_consumer", "consumer_id", CONSUMER_ID + 1),
        ("wrong_owner", "owner", 0),
        ("wrong_generation", "generation", 2),
        ("empty_transaction", "transaction_id", ""),
    ):
        request = dict(base)
        request[field_name] = value
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[f"{label}_rejected"] = response.get("status") == "REJECTED"
    status, sent, received = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "primary_n2"}
    )
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["descriptor_attacks_leave_carrier_canonical"] = (
        status.get("canonical") is True and status.get("last_restored_generation") == 0
    )

    cases: list[dict[str, object]] = []
    schedule = [
        ("primary_n2", 1, "PRIMARY_N2"),
        ("primary_n3", 1, "PRIMARY_N3"),
        ("primary_n4", 1, "PRIMARY_N4"),
        ("primary_n6", 1, "PRIMARY_N6"),
        ("reuse_n6", 2, "RESTORED_REUSE_N6"),
        ("reuse_fresh_n6", 1, "FRESH_REUSE_REFERENCE_N6"),
    ]
    for oracle_id, generation, run_kind in schedule:
        response, sent, received = exchange(
            socket_name, run_request(oracle_id, generation, f"M246_{run_kind}")
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK" or set(response) != {"status", "response"}:
            raise RuntimeError(f"M246 atomic case rejected: {run_kind}")
        case = dict(response["response"])
        case["run_kind"] = run_kind
        case["released_final_boundary_exact_payload_bits"] = boundary_payload_bits(
            case["final_amplitude"]
        )
        case["controller_request_bytes"] = sent
        case["backend_response_bytes"] = received
        cases.append(case)

    reuse, fresh = cases[-2], cases[-1]
    if reuse["final_amplitude"] != fresh["final_amplitude"]:
        raise RuntimeError("M246 restored/fresh boundary mismatch")
    for key in (
        "occupation_dimension", "message_field_cells", "output_scratch_field_cells",
        "coefficient_row_field_cells", "total_fixed_field_backing_cells",
        "public_plan_integer_cells", "hidden_descriptor_residue_cells",
        "forward_kernel_coefficient_terms", "inverse_kernel_coefficient_terms",
        "forward_orbit_dot_terms", "inverse_orbit_dot_terms",
        "normalization_field_multiplications", "accepted_descriptor_reads",
    ):
        if reuse[key] != fresh[key]:
            raise RuntimeError(f"M246 restored/fresh resource mismatch: {key}")

    stale, sent, received = exchange(
        socket_name, run_request("reuse_n6", 2, "M246_STALE_GENERATION")
    )
    total_request_bytes += sent
    total_response_bytes += received
    stale_status, sent, received = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "reuse_n6"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    backend_controls, sent, received = exchange(
        socket_name, {"command": "CONTROLS", "oracle_id": "primary_n3"}
    )
    total_request_bytes += sent
    total_response_bytes += received

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_OCCUPATION_MESSAGE", "PROJECT_LABELLED_MESSAGE", "PROJECT_SCRATCH",
        "PROJECT_COEFFICIENT_ROW", "PROJECT_DESCRIPTOR", "PROJECT_INTERMEDIATE",
        "DENSE_ASSIGNMENTS", "DENSE_KERNEL", "SNAPSHOT", "RUN_SNAPSHOT",
        "NULL_CARRIER",
    ):
        response, sent, received = exchange(
            socket_name, {"command": command, "oracle_id": "primary_n2"}
        )
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(socket_name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M246 service shutdown failed")

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
            case["same_message_backing"]
            and case["same_output_scratch_backing"]
            and case["same_coefficient_row_backing"]
            and case["same_descriptor_backings"]
            for case in cases
        ),
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in cases),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M246 control failure: {controls}")

    dimensions = [width_for(n) for n in RAILS]
    resource_law = {
        "declared_rails": list(RAILS),
        "occupation_dimensions": dimensions,
        "dimension_law": "binomial_n_plus4_choose4",
        "accepted_message_field_cells": dimensions,
        "accepted_output_scratch_field_cells": dimensions,
        "accepted_coefficient_row_field_cells": dimensions,
        "accepted_total_fixed_field_backing_cells": [3 * value for value in dimensions],
        "hidden_descriptor_residue_cells_per_program": 16,
        "descriptor_residue_law": "rails_plus_depth_public_metadata_not_counted_lambda3_quadratic3_rung3_hidden_coupling2_output_occupation5",
        "retained_dynamic_inverse_history_entries": 0,
        "retained_final_amplitude_field_cells_during_inverse": 1,
        "retained_final_amplitude_denominator_exponent_scalar_cells_during_inverse": 1,
        "retained_final_amplitude_denominator_material_bit_upper_bounds": [
            signed_bits(5 ** (3 * ((n + 1) // 2))) for n in RAILS
        ],
        "suite_service_configuration_descriptor_residue_cells": 144,
        "suite_service_carrier_descriptor_residue_cells": 128,
        "suite_service_configuration_plus_carrier_descriptor_residue_cells": 272,
        "suite_service_retains8_carriers_for_controls_cases_and_fresh_comparison": True,
        "suite_service_total_fixed_field_backing_cells": 1755,
        "suite_service_shared_public_occupation_and_index_plan_integer_cells": 2640,
        "suite_service_startup_descriptor_conversion_and_validation_reads_instrumented": False,
        "public_single_vector_exact_payload_bit_upper_bounds": [
            public_vector_payload_bound(n) for n in RAILS
        ],
        "public_payload_bounds_are_conservative_not_measured_secret_metrics": True,
        "secret_dependent_intermediate_payload_metrics_released": False,
        "secret_dependent_intermediate_payload_metrics_sealed": False,
        "strongest_implemented_classical_baseline": (
            "ENDPOINT_SPECIALIZED_MATRIX_FREE_OCCUPATION_MULTINOMIAL_"
            "COEFFICIENT_RECURRENCE"
        ),
        "unimplemented_stronger_exact_classical_ceiling": (
            "SYMMETRIC_POWER_FIVE_MODE_FOURIER_FACTORIZATION_INTO_TWO_MODE_"
            "BINOMIAL_KRAWTCHOUK_BLOCKS_WITH_AT_MOST_NPLUS1_BLOCK_SCRATCH"
        ),
        "classical_optimality_claimed": False,
        "declared_coefficient_update_plus_dot_terms_are_not_total_work": True,
        "multinomial_integer_arithmetic_and_python_operation_cost_comparison_complete": False,
        "total_forward_work_comparison_authorized": False,
        "classical_forward_only_has_no_inverse_restoration_or_catvm_work": True,
        "labelled5_to_n_assignment_materialization_on_accepted_path": False,
        "dense_occupation_transfer_matrix_on_accepted_path": False,
        "coefficient_row_rematerialized_and_cleared_per_output": True,
        "coefficient_row_recursion_transient_integer_cells_upper_bound": 35,
        "canonicalization_scan_and_integer_division_work_instrumented": False,
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
            "labelled5_to_n_interfaces_compressed": False,
            "fixed_rank_or_fixed_width_exact_state": False,
            "all_n_rank_growth_theorem": False,
            "arbitrary_nonlinear_encoding_lower_bound": False,
            "arbitrary_graph_or_treewidth": False,
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
        raise SystemExit("usage: client.py @catvm-m246-NAME")
    main(sys.argv[1])
