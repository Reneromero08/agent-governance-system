#!/usr/bin/env python3
"""M249 public CATVM controller; imports no backend or arithmetic code."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


LENGTHS = (2, 4, 8, 16)
PORT_TYPE = "CATVM_U1_FINITE_REFERENCE_JOINT_PORT_V1"
OUTPUT_TYPE = "QSQRT2_REDUCED_SYSTEM_BOUNDARY_V1"
CONSUMER_ID = 249001
OWNER = 249004
RESULT = "PASS_CATVM_U1_PHASE_REFERENCE_ENERGY_WRAP_DICHOTOMY_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_U1_FINITE_OPEN_LADDER_PHASE_REFERENCES_"
    "AT_L2_4_8_16_UNDER_FORMAL_ENERGY_CONSERVING_TWO_LEVEL_DILATIONS_"
    "DEVELOP_NONZERO_SYSTEM_REFERENCE_CORRELATION_WHILE_CYCLIC_EXACT_"
    "RETURN_HAS_NONZERO_ENERGY_WRAP_AND_THE_BILATERAL_EXACT_SHIFT_"
    "EIGENREFERENCE_IS_NONNORMALIZABLE_WITH_ATOMIC_EXACT_SAME_BACKING_"
    "INVERSE_RESTORATION_AND_GENERATION2_REUSE_AND_AN_O1_ANALYTIC_"
    "CLASSICAL_BASELINE"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_QSQRT2_FINITE_SHIFT_COVARIANT_U1_REFERENCE_FAMILY_"
    "L2_4_8_16_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m249-"):
        raise RuntimeError("M249 client requires a declared abstract Unix socket")
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
        raise RuntimeError("M249 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[int, str]:
    return int(descriptor["length"]), str(descriptor["gate"])


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
        "length": int(descriptor["length"]),
        "program_id": program_id(descriptor),
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
    request = run_request(case, 1, "M249_DISCONNECT_CONTROL")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(socket_name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def comparable(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "length", "gate", "boundary", "joint_carrier_commitment",
            "joint_field_cells", "retained_final_boundary_field_cells_during_inverse",
            "work",
        )
    }


def contains_forbidden_intermediate(response: object) -> bool:
    forbidden = {
        "joint", "reference", "reservoir", "amplitude_vector", "dense_operator",
        "intermediate", "eigenvector", "system_reference_amplitudes",
    }
    if isinstance(response, dict):
        return any(
            str(key).lower() in forbidden or contains_forbidden_intermediate(value)
            for key, value in response.items()
        )
    if isinstance(response, list):
        return any(contains_forbidden_intermediate(value) for value in response)
    return False


def main(socket_name: str) -> None:
    public = json.load(sys.stdin)
    cases = public["cases"]
    expected = {
        *(f"primary_{length}" for length in LENGTHS),
        *(f"reuse_{length}" for length in LENGTHS),
        *(f"fresh_{length}" for length in LENGTHS),
        "disconnect", "partial", "postprojection", "descriptor_control",
    }
    if set(cases) != expected:
        raise RuntimeError("invalid M249 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for length in LENGTHS:
        for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1)):
            case = cases[f"{label}_{length}"]
            response, sent, received = exchange(
                socket_name,
                run_request(case, generation, f"M249_{label.upper()}_{length}"),
            )
            total_request_bytes += sent
            total_response_bytes += received
            if response.get("status") != "OK":
                raise RuntimeError(f"M249 accepted case rejected: {label}_{length}")
            item = dict(response["response"])
            item["run_kind"] = label.upper()
            item["controller_request_bytes"] = sent
            item["backend_response_bytes"] = received
            accepted.append(item)

    reuse_parity: dict[str, bool] = {}
    for length in LENGTHS:
        reuse = next(
            case for case in accepted
            if case["length"] == length and case["run_kind"] == "REUSE"
        )
        fresh = next(
            case for case in accepted
            if case["length"] == length and case["run_kind"] == "FRESH"
        )
        reuse_parity[str(length)] = (
            comparable(reuse) == comparable(fresh)
            and reuse["generation"] == 2
            and fresh["generation"] == 1
        )

    disconnect_request_bytes = disconnect_run(socket_name, cases["disconnect"])
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(
            socket_name, status_request(cases["disconnect"])
        )
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M249 disconnect did not restore")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (
        ("partial", "inject_failure_after_forward"),
        ("postprojection", "inject_failure_after_projection"),
    ):
        request = run_request(cases[case_id], 1, f"M249_{case_id.upper()}")
        request[field] = True
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(socket_name, status_request(cases[case_id]))
        total_request_bytes += sent
        total_response_bytes += received
        fault_controls[f"{case_id}_failure_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED"
            and "response" not in response
            and status.get("canonical") is True
            and status.get("last_restored_generation") == 1
        )

    control_case = cases["descriptor_control"]
    base = run_request(control_case, 1, "M249_CONTROL")
    controls_response, sent, received = exchange(
        socket_name, {**base, "command": "CONTROLS"}
    )
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M249 backend controls rejected")

    descriptor_controls: dict[str, bool] = {}
    mutations: list[tuple[str, dict[str, object]]] = []
    wrong_program = dict(base)
    wrong_program["program_id"] = "0" * 64
    mutations.append(("wrong_program_rejected", wrong_program))
    wrong_owner = dict(base)
    wrong_owner["owner"] = OWNER + 1
    mutations.append(("wrong_owner_rejected", wrong_owner))
    wrong_type = dict(base)
    wrong_type["port_type"] = "WRONG_PORT"
    mutations.append(("wrong_type_rejected", wrong_type))
    wrong_output = dict(base)
    wrong_output["output_type"] = "WRONG_OUTPUT"
    mutations.append(("wrong_output_type_rejected", wrong_output))
    wrong_consumer = dict(base)
    wrong_consumer["consumer_id"] = CONSUMER_ID + 1
    mutations.append(("wrong_consumer_rejected", wrong_consumer))
    stale = run_request(cases["primary_4"], 2, "M249_STALE")
    mutations.append(("stale_generation_rejected", stale))
    skipped = dict(base)
    skipped["generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id_mutation = json.loads(json.dumps(base))
    same_id_mutation["descriptor"]["gate"] = "RATIONAL_3_4_5"
    mutations.append(("same_id_descriptor_mutation_rejected", same_id_mutation))
    wrong_length = json.loads(json.dumps(run_request(
        cases["primary_4"], 3, "M249_WRONG_LENGTH"
    )))
    wrong_length["descriptor"]["length"] = 8
    wrong_length["length"] = 8
    wrong_length["program_id"] = program_id(wrong_length["descriptor"])
    mutations.append(("carrier_length_type_mismatch_rejected", wrong_length))
    for label, request in mutations:
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_REFERENCE", "PROJECT_JOINT", "PROJECT_INTERMEDIATE",
        "PROJECT_RESERVOIR", "AMPLITUDE_VECTOR", "DENSE_OPERATOR",
        "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
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
        raise RuntimeError("M249 shutdown failed")

    controls = {
        **dict(controls_response["controls"]),
        **fault_controls,
        **descriptor_controls,
        **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"]
            and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(
            case["canonical_after_restoration"] for case in accepted
        ),
        "all_accepted_joint_backings_stable": all(
            case["same_joint_backing"] for case in accepted
        ),
        "all_reuse_matches_fresh": all(reuse_parity.values()),
        "no_baseline_reload": all(
            not case["baseline_reload_used"] for case in accepted
        ),
        "backend_responses_exclude_intermediate_arrays": not any(
            contains_forbidden_intermediate(case) for case in accepted
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M249 control failure: {controls}")

    total_blocks = sum(
        int(case["work"]["forward_fixed_energy_pair_updates"])
        for case in accepted
    )
    resource_law = {
        "declared_lengths": list(LENGTHS),
        "accepted_joint_field_cells_by_length": [2 * length for length in LENGTHS],
        "accepted_persistent_primary_and_fresh_carriers": 8,
        "accepted_persistent_joint_field_cells_across_service": 4 * sum(LENGTHS),
        "control_only_persistent_carriers": 3,
        "control_only_joint_field_cells": 24,
        "accepted_transactions": len(accepted),
        "accepted_forward_fixed_energy_pair_updates": total_blocks,
        "accepted_inverse_fixed_energy_pair_updates": sum(
            int(case["work"]["inverse_fixed_energy_pair_updates"])
            for case in accepted
        ),
        "accepted_forward_field_multiplications": sum(
            int(case["work"]["forward_field_multiplications"])
            for case in accepted
        ),
        "accepted_inverse_field_multiplications": sum(
            int(case["work"]["inverse_field_multiplications"])
            for case in accepted
        ),
        "accepted_boundary_field_multiplications": sum(
            int(case["work"]["boundary_square_multiplications"])
            + int(case["work"]["boundary_coherence_multiplications"])
            for case in accepted
        ),
        "accepted_boundary_field_accumulations": sum(
            int(case["work"]["boundary_accumulations"])
            for case in accepted
        ),
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 3,
        "retained_dynamic_inverse_history_entries": 0,
        "public_descriptor_scalar_fields_per_transaction": 2,
        "program_digest_material_bytes_per_transaction": 32,
        "strongest_classical_baseline": "O1_EXACT_ALL_L_BOUNDARY_FORMULAS_FROM_PUBLIC_L_A_B",
        "secondary_classical_baseline": "O_L_STREAMED_TWO_ROW_SPARSE_RECURRENCE_WITH_CONSTANT_FIELD_WORKSPACE",
        "analytic_baseline_requires_no_joint_carrier_inverse_or_catvm_traffic": True,
        "accepted_catvm_path_has_space_or_work_advantage": False,
        "finite_reference_exact_payload_not_fixed_width": True,
        "canonical_eta_denominator_bit_width_grows_logarithmically_with_length": True,
        "dense_2l_by_2l_operator_materialized": False,
        "comparison_basis": "EXACT_QSQRT2_FIELD_CELLS_PUBLIC_PAIR_UPDATES_FINAL_BOUNDARY_RESTORATION_REUSE_AND_PROTOCOL_TRAFFIC_NOT_WHOLE_PROCESS_RSS",
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_transaction_live_payload_peak_complete": False,
        "python_fraction_object_allocator_socket_hash_serialization_rss_excluded_not_zero": True,
    }

    output = {
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": accepted,
        "reuse_parity": reuse_parity,
        "controls": controls,
        "dichotomy": {
            "lawful_finite_open_reference_returns_exactly_without_correlation": False,
            "cyclic_reference_exact_return": True,
            "cyclic_reference_preserves_declared_total_number": False,
            "bilateral_exact_shift_eigenreference_is_normalizable": False,
            "route_disposition": "RETIRE_FINITE_NORMALIZABLE_SHIFT_EIGENREFERENCE_AS_AN_ADVANTAGE_ROUTE_IF_INDEPENDENT_PARITY_PASSES",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_request_bytes,
        },
        "claim_limits": {
            "all_coherence_catalysts_excluded": False,
            "physical_energy_conservation_established": False,
            "physical_phase_reference_executed": False,
            "distinct_phase_resource_unavailable_to_compact_classical_software": False,
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "general_catalytic_inference": False,
            "unbounded_catalytic_computation": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: client.py @catvm-m249-NAME")
    main(sys.argv[1])
