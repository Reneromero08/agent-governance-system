#!/usr/bin/env python3
"""M252 public controller; imports no backend or exact arithmetic code."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_QZETA8_COHERENT_ORDER_BRANCH_PORT_V1"
OUTPUT_TYPE = "QZETA8_ORDER_COMMUTATOR_BOUNDARY_V1"
OWNER = 252004
CONTROLLER_ID = 252001
RESULT = "PASS_CATVM_COHERENT_ORDER_COMMUTATOR_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_QZETA8_COHERENT_ORDER_PORT_ROUTES_ONE_"
    "TARGET_QUBIT_THROUGH_VU_AND_UV_BRANCH_CONSUMERS_AND_CLOSES_ONLY_TO_THE_"
    "COMMUTATOR_PHASE_BOUNDARY_WITH_ATOMIC_EXACT_SAME_BACKING_INVERSE_"
    "RESTORATION_AND_GENERATION2_REUSE_BUT_AN_O1_FIXED_FIXTURE_INVARIANT_AND_"
    "THE_STREAMED_ONE_VECTOR_CLASSICAL_COMMUTATOR_RECURRENCE_ARE_SMALLER_AND_NO_"
    "ADVANTAGE_OR_PHYSICAL_INDEFINITE_ORDER_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_ONE_HIDDEN_TWO_BRANCH_ORDER_PORT_ONE_FIXED_ZERO_TARGET_QUBIT_"
    "PUBLIC_GATE_PAIRS_FROM_X_Z_H_T_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m252-"):
        raise RuntimeError("M252 client requires declared abstract socket")
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
        raise RuntimeError("M252 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_pair(descriptor: dict[str, Any]) -> tuple[str, str]:
    return (str(descriptor["u"]), str(descriptor["v"]))


def program_id(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(canonical_pair(descriptor), separators=(",", ":")).encode()).hexdigest()


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
    request = run_request(case, 1, "M252_DISCONNECT")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def comparable(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in (
        "pair", "commutator_boundary", "hidden_branch_field_cells",
        "hidden_scratch_field_cells", "hidden_order_consumer_receipt_cells",
        "retained_final_boundary_field_cells_during_inverse", "work",
    )}


def contains_forbidden(value: object) -> bool:
    forbidden = {
        "amplitudes", "branch_values", "scratch_values", "order_values",
        "gate_matrices", "paths", "assignments", "intermediate",
    }
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden or contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden(item) for item in value)
    return False


def main(name: str) -> None:
    public = json.load(sys.stdin)
    cases = public["cases"]
    expected = {"primary", "reuse", "fresh", "disconnect", "partial", "postprojection", "descriptor_control"}
    if set(cases) != expected:
        raise RuntimeError("invalid M252 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1)):
        response, sent, received = exchange(name, run_request(cases[label], generation, f"M252_{label.upper()}"))
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK":
            raise RuntimeError(f"M252 accepted case rejected: {label}")
        item = dict(response["response"])
        item["run_kind"] = label.upper()
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
        raise RuntimeError("M252 disconnect restoration timeout")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (("partial", "inject_failure_after_partial"), ("postprojection", "inject_failure_after_projection")):
        request = run_request(cases[case_id], 1, f"M252_{case_id.upper()}")
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

    base = run_request(cases["descriptor_control"], 1, "M252_CONTROL")
    controls_response, sent, received = exchange(name, {**base, "command": "CONTROLS"})
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M252 backend controls rejected")

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
    mutations.append(("stale_generation_rejected", run_request(cases["primary"], 2, "M252_STALE")))
    skipped = dict(base)
    skipped["generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id = json.loads(json.dumps(base))
    same_id["descriptor"]["v"] = "H"
    mutations.append(("same_id_pair_mutation_rejected", same_id))
    undeclared = json.loads(json.dumps(base))
    undeclared["descriptor"]["u"] = "DENSE_U"
    mutations.append(("undeclared_gate_rejected", undeclared))
    malformed = json.loads(json.dumps(base))
    malformed["descriptor"]["extra"] = "answer"
    mutations.append(("answer_bearing_extra_descriptor_field_rejected", malformed))
    for label, request in mutations:
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    malformed_transaction = dict(base)
    malformed_transaction["transaction_id"] = 252
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
    malformed_carrier["carrier_id"] = 252
    response, sent, received = exchange(name, malformed_carrier)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["malformed_carrier_id_rejected"] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_ORDER", "PROJECT_BRANCH", "PROJECT_AMPLITUDES", "PROJECT_SCRATCH",
        "GATE_MATRICES", "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
        "NULL_CARRIER", "DUMP", "DEBUG",
    ):
        response, sent, received = exchange(name, {"command": command})
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M252 shutdown failed")

    controls = {
        **dict(controls_response["controls"]), **fault_controls,
        **descriptor_controls, **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"] and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(case["canonical_after_restoration"] for case in accepted),
        "all_accepted_branch_scratch_and_receipt_backings_stable": all(
            case["same_branch_scratch_and_receipt_backings"] for case in accepted
        ),
        "restored_reuse_matches_fresh": reuse_parity,
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in accepted),
        "backend_responses_exclude_hidden_branch_order_and_scratch_values": not any(
            contains_forbidden(case) for case in accepted
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M252 control failure: {controls}")

    resource_law = {
        "declared_public_gate_pairs": [case["pair"] for case in accepted],
        "accepted_persistent_carriers": 2,
        "accepted_hidden_branch_field_cells_per_carrier": 4,
        "accepted_hidden_scratch_field_cells_per_carrier": 2,
        "accepted_hidden_order_consumer_receipt_cells_per_carrier": 2,
        "accepted_transactions": 3,
        "service_static_public_gate_library_field_cells": 16,
        "accepted_compiled_public_gate_plan_matrix_references": sum(
            case["work"]["compiled_public_gate_plan_matrix_references"] for case in accepted
        ),
        "accepted_forward_branch_gate_actions": sum(case["work"]["forward_branch_gate_actions"] for case in accepted),
        "accepted_inverse_branch_gate_actions": sum(case["work"]["inverse_branch_gate_actions"] for case in accepted),
        "accepted_forward_field_multiply_terms": sum(case["work"]["forward_field_multiply_terms"] for case in accepted),
        "accepted_inverse_field_multiply_terms": sum(case["work"]["inverse_field_multiply_terms"] for case in accepted),
        "accepted_forward_field_accumulations": sum(case["work"]["forward_field_accumulations"] for case in accepted),
        "accepted_inverse_field_accumulations": sum(case["work"]["inverse_field_accumulations"] for case in accepted),
        "accepted_forward_branch_field_writes": sum(case["work"]["forward_branch_field_writes"] for case in accepted),
        "accepted_inverse_branch_field_writes": sum(case["work"]["inverse_branch_field_writes"] for case in accepted),
        "accepted_forward_scratch_clear_writes": sum(case["work"]["forward_scratch_clear_writes"] for case in accepted),
        "accepted_inverse_scratch_clear_writes": sum(case["work"]["inverse_scratch_clear_writes"] for case in accepted),
        "accepted_boundary_field_multiply_terms": sum(case["work"]["boundary_field_multiply_terms"] for case in accepted),
        "accepted_boundary_field_accumulations": sum(case["work"]["boundary_field_accumulations"] for case in accepted),
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 1,
        "retained_dynamic_inverse_history_entries": 0,
        "strongest_fixed_fixture_classical_baseline": "PUBLIC_PAIR_VALIDATION_PLUS_FROZEN_EXACT_COMMUTATOR_BOUNDARY_IN_O1_WORK",
        "strongest_transferable_descriptor_level_classical_baseline": "DIRECT_EXACT_ONE_TWO_COMPONENT_QZETA8_VECTOR_COMMUTATOR_WORD_V_THEN_U_THEN_V_DAGGER_THEN_U_DAGGER_WITH2_RESIDENT_FIELD_CELLS_PLUS2_REUSABLE_SCRATCH_CELLS_AND_NO_CATVM_RESTORATION",
        "transferable_baseline_resident_target_field_cells": 2,
        "transferable_baseline_reusable_scratch_field_cells": 2,
        "transferable_baseline_gate_actions_per_case": 4,
        "transferable_baseline_field_multiply_terms_per_case": 16,
        "accepted_catvm_path_has_space_work_or_query_advantage": False,
        "comparison_basis": "QZETA8_FIELD_CELLS_GATE_TERMS_WRITES_FINAL_BOUNDARY_RESTORATION_REUSE_AND_PROTOCOL_TRAFFIC_NOT_WHOLE_PROCESS_RSS",
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
        "coherent_order_law": {
            "hidden_order_branch_count": 2,
            "same_target_consumed_in_orders": ["VU", "UV"],
            "final_boundary_is_order_commutator_expectation": True,
            "order_port_remains_unprojected_until_final_overlap": True,
            "order_coherence_is_causally_required_for_boundary": True,
            "route_disposition": "RETIRE_AFTER_ONE_BOUNDED_GATE_PAIR_SUITE_IF_STREAMED_CLASSICAL_RECURRENCE_MATCHES",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_bytes,
        },
        "claim_limits": {
            "physical_indefinite_causal_order": False,
            "quantum_switch_or_oracle_query_separation": False,
            "general_process_matrix_execution": False,
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
        raise SystemExit("usage: client.py @catvm-m252-NAME")
    main(sys.argv[1])
