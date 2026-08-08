#!/usr/bin/env python3
"""M251 public controller; imports no backend or exact matrix implementation."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_DQC1_OPERATOR_COHERENCE_DENSITY_PORT_V1"
OUTPUT_TYPE = "DQC1_NORMALIZED_TRACE_BOUNDARY_V1"
OWNER = 251004
CONTROLLER_ID = 251001
RESULT = "PASS_CATVM_DQC1_OPERATOR_COHERENCE_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_DQC1_ONE_CLEAN_CONTROL_AND_TWO_QUBIT_"
    "MAXIMALLY_MIXED_OPERATOR_COHERENCE_RETURNS_THE_QZETA8_NORMALIZED_TRACE_"
    "OF_A_DECLARED_NONCOMMUTING_PUBLIC_WORD_WITH_FINAL_ONLY_RESPONSE_ATOMIC_"
    "EXACT_SAME_BACKING_INVERSE_RESTORATION_AND_GENERATION2_REUSE_BUT_AN_O1_"
    "FIXED_FIXTURE_INVARIANT_AND_THE_DIRECT_FOUR_BY_FOUR_PUBLIC_MATRIX_TRACE_"
    "RECURRENCE_ARE_SMALLER_AND_NO_COMPUTATIONAL_ADVANTAGE_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_ONE_CLEAN_CONTROL_TWO_MAXIMALLY_MIXED_DATA_QUBITS_"
    "PUBLIC_WORD_LENGTH_AT_MOST8_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m251-"):
        raise RuntimeError("M251 client requires declared abstract socket")
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
        raise RuntimeError("M251 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_word(descriptor: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in descriptor["word"])


def program_id(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(canonical_word(descriptor), separators=(",", ":")).encode()).hexdigest()


def run_request(case: dict[str, Any], generation: int, transaction_id: str) -> dict[str, object]:
    descriptor = case["descriptor"]
    return {
        "command": "RUN", "carrier_id": case["carrier_id"], "descriptor": descriptor,
        "program_id": program_id(descriptor), "port_type": PORT_TYPE,
        "output_type": OUTPUT_TYPE, "controller_id": CONTROLLER_ID,
        "owner": OWNER, "generation": generation, "transaction_id": transaction_id,
    }


def status_request(case: dict[str, Any]) -> dict[str, object]:
    return {"command": "STATUS", "carrier_id": case["carrier_id"]}


def disconnect_run(name: str, case: dict[str, Any]) -> int:
    request = run_request(case, 1, "M251_DISCONNECT")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def comparable(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in (
        "word", "normalized_trace", "hidden_density_field_cells",
        "hidden_scratch_field_cells", "retained_final_boundary_field_cells_during_inverse",
        "work",
    )}


def contains_forbidden(response: object) -> bool:
    forbidden = {
        "density_values", "density_matrix", "scratch_values", "operator_blocks",
        "gate_matrices", "amplitudes", "paths", "assignments", "intermediate",
    }
    if isinstance(response, dict):
        return any(str(key).lower() in forbidden or contains_forbidden(value) for key, value in response.items())
    if isinstance(response, list):
        return any(contains_forbidden(value) for value in response)
    return False


def main(name: str) -> None:
    public = json.load(sys.stdin)
    cases = public["cases"]
    expected = {"primary", "reuse", "fresh", "disconnect", "partial", "postprojection", "descriptor_control"}
    if set(cases) != expected:
        raise RuntimeError("invalid M251 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1)):
        response, sent, received = exchange(name, run_request(cases[label], generation, f"M251_{label.upper()}"))
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK":
            raise RuntimeError(f"M251 accepted case rejected: {label}")
        item = dict(response["response"])
        item["run_kind"] = label.upper()
        item["controller_request_bytes"] = sent
        item["backend_response_bytes"] = received
        accepted.append(item)

    reuse = next(case for case in accepted if case["run_kind"] == "REUSE")
    fresh = next(case for case in accepted if case["run_kind"] == "FRESH")
    reuse_parity = comparable(reuse) == comparable(fresh) and reuse["generation"] == 2 and fresh["generation"] == 1

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
        raise RuntimeError("M251 disconnect restoration timeout")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (("partial", "inject_failure_after_partial"), ("postprojection", "inject_failure_after_projection")):
        request = run_request(cases[case_id], 1, f"M251_{case_id.upper()}")
        request[field] = True
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(name, status_request(cases[case_id]))
        total_request_bytes += sent
        total_response_bytes += received
        fault_controls[f"{case_id}_failure_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED" and "response" not in response
            and status.get("canonical") is True and status.get("last_restored_generation") == 1
        )

    base = run_request(cases["descriptor_control"], 1, "M251_CONTROL")
    controls_response, sent, received = exchange(name, {**base, "command": "CONTROLS"})
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M251 backend controls rejected")

    descriptor_controls: dict[str, bool] = {}
    mutations: list[tuple[str, dict[str, object]]] = []
    for label, key, value in (
        ("wrong_program_rejected", "program_id", "0" * 64),
        ("wrong_owner_rejected", "owner", OWNER + 1),
        ("wrong_type_rejected", "port_type", "WRONG"),
        ("wrong_output_type_rejected", "output_type", "WRONG"),
        ("wrong_controller_rejected", "controller_id", CONTROLLER_ID + 1),
    ):
        mutated = dict(base)
        mutated[key] = value
        mutations.append((label, mutated))
    mutations.append(("stale_generation_rejected", run_request(cases["primary"], 2, "M251_STALE")))
    skipped = dict(base)
    skipped["generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id = json.loads(json.dumps(base))
    same_id["descriptor"]["word"][0] = "T0"
    mutations.append(("same_id_word_mutation_rejected", same_id))
    invalid = json.loads(json.dumps(base))
    invalid["descriptor"]["word"] = ["DENSE_U"]
    mutations.append(("undeclared_gate_rejected", invalid))
    too_long = json.loads(json.dumps(base))
    too_long["descriptor"]["word"] = ["H0"] * 9
    mutations.append(("word_longer_than8_rejected", too_long))
    for label, request in mutations:
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    malformed_transaction = dict(base)
    malformed_transaction["transaction_id"] = 123
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
    malformed_carrier["carrier_id"] = 251
    response, sent, received = exchange(name, malformed_carrier)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["malformed_carrier_id_rejected"] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_DENSITY", "PROJECT_BLOCK", "PROJECT_OPERATOR_COHERENCE",
        "DENSITY_MATRIX", "GATE_MATRICES", "SNAPSHOT", "RUN_SNAPSHOT",
        "RUN_INPLACE_ON_SNAPSHOT", "NULL_CARRIER", "DUMP", "DEBUG",
    ):
        response, sent, received = exchange(name, {"command": command})
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M251 shutdown failed")

    controls = {
        **dict(controls_response["controls"]), **fault_controls, **descriptor_controls, **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"] and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(case["canonical_after_restoration"] for case in accepted),
        "all_accepted_density_and_scratch_backings_stable": all(case["same_density_and_scratch_backings"] for case in accepted),
        "restored_reuse_matches_fresh": reuse_parity,
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in accepted),
        "backend_responses_exclude_hidden_density_and_operator_blocks": not any(contains_forbidden(case) for case in accepted),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M251 control failure: {controls}")

    resource_law = {
        "declared_public_word_lengths": [len(case["word"]) for case in accepted],
        "accepted_persistent_carriers": 2,
        "accepted_hidden_density_field_cells_per_carrier": 64,
        "accepted_hidden_scratch_field_cells_per_carrier": 4,
        "accepted_transactions": 3,
        "service_static_public_gate_library_field_cells": 96,
        "accepted_compiled_public_gate_plan_matrix_references": sum(case["work"]["compiled_public_gate_plan_matrix_references"] for case in accepted),
        "accepted_forward_controlled_gates": sum(case["work"]["forward_controlled_gates"] for case in accepted),
        "accepted_inverse_controlled_gates": sum(case["work"]["inverse_controlled_gates"] for case in accepted),
        "accepted_forward_field_multiply_terms": sum(case["work"]["forward_field_multiply_terms"] for case in accepted),
        "accepted_inverse_field_multiply_terms": sum(case["work"]["inverse_field_multiply_terms"] for case in accepted),
        "accepted_forward_density_field_writes": sum(case["work"]["forward_density_field_writes"] for case in accepted),
        "accepted_inverse_density_field_writes": sum(case["work"]["inverse_density_field_writes"] for case in accepted),
        "accepted_scratch_result_and_clear_writes_forward": [
            sum(case["work"]["forward_scratch_result_writes"] for case in accepted),
            sum(case["work"]["forward_scratch_clear_writes"] for case in accepted),
        ],
        "accepted_scratch_result_and_clear_writes_inverse": [
            sum(case["work"]["inverse_scratch_result_writes"] for case in accepted),
            sum(case["work"]["inverse_scratch_clear_writes"] for case in accepted),
        ],
        "accepted_boundary_field_multiplications": sum(case["work"]["boundary_field_multiplications"] for case in accepted),
        "accepted_boundary_field_accumulations": sum(case["work"]["boundary_field_accumulations"] for case in accepted),
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 1,
        "retained_dynamic_inverse_history_entries": 0,
        "strongest_fixed_fixture_classical_baseline": "PUBLIC_WORD_VALIDATION_PLUS_FROZEN_EXACT_NORMALIZED_TRACE_IN_O1_WORK",
        "strongest_transferable_descriptor_level_classical_baseline": "DIRECT_EXACT_FOUR_BY_FOUR_QZETA8_PUBLIC_WORD_MATRIX_RECURRENCE_PLUS_TRACE_WITH16_RESIDENT_MATRIX_FIELD_CELLS_PLUS_DECLARED_TRANSIENT_MULTIPLICATION_SCRATCH_AND_NO_CATVM_INVERSE",
        "direct_transferable_baseline_resident_matrix_field_cells": 16,
        "direct_transferable_baseline_transient_peak_complete": False,
        "independent_verifier_only_baseline": "EXACT_DENSE_EIGHT_BY_EIGHT_QZETA8_DENSITY_CONJUGATION",
        "accepted_catvm_path_has_space_work_or_query_advantage": False,
        "comparison_basis": "QZETA8_FIELD_CELLS_GATE_TERMS_DENSITY_AND_SCRATCH_WRITES_FINAL_BOUNDARY_RESTORATION_REUSE_AND_PROTOCOL_TRAFFIC_NOT_WHOLE_PROCESS_RSS",
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_transaction_live_payload_peak_complete": False,
        "exact_qzeta8_coordinate_payload_instrumented": False,
        "field_cell_counts_are_not_fixed_bit_payload_claims": True,
        "python_fraction_object_allocator_socket_hash_serialization_rss_excluded_not_zero": True,
    }

    output = {
        "result": RESULT, "claim": CLAIM, "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": accepted, "reuse_parity": reuse_parity, "controls": controls,
        "operator_coherence_law": {
            "one_clean_control": True, "maximally_mixed_data_qubits": 2,
            "final_boundary_is_normalized_trace": True,
            "data_marginal_unchanged_but_full_joint_state_correlated": True,
            "operator_coherence_is_causally_required_for_xy_boundary": True,
            "route_disposition": "RETIRE_AFTER_ONE_BOUNDED_GRAMMAR_IF_DIRECT_FOUR_BY_FOUR_TRACE_RECURRENCE_MATCHES",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_bytes,
        },
        "claim_limits": {
            "physical_mixed_state_execution": False,
            "dqc1_hardness_established": False,
            "oracle_or_query_separation": False,
            "distinct_phase_resource_unavailable_to_compact_classical_software": False,
            "total_computational_advantage": False,
            "general_catalytic_inference": False,
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
        raise SystemExit("usage: client.py @catvm-m251-NAME")
    main(sys.argv[1])
