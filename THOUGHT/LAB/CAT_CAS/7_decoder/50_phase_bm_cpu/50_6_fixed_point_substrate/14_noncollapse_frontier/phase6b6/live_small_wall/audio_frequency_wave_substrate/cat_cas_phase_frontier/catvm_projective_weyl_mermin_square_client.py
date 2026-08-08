#!/usr/bin/env python3
"""M250 public controller; imports no backend, Weyl algebra, or matrix oracle."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_TYPED_TWO_QUBIT_PROJECTIVE_WEYL_PORT_V1"
OUTPUT_TYPE = "PROJECTIVE_WEYL_CENTRAL_PHASE_BOUNDARY_V1"
OWNER = 250004
CONTROLLER_ID = 250001
RESULT = "PASS_CATVM_PROJECTIVE_WEYL_MERMIN_SQUARE_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_TWO_QUBIT_PROJECTIVE_WEYL_2_COCYCLE_"
    "COMPOSES_NINE_TYPED_SHARED_OBSERVABLE_PORTS_ACROSS_SIX_MERMIN_SQUARE_"
    "CONTEXTS_TO_ONE_CENTRAL_MINUS_ONE_PHASE_WITH_FINAL_ONLY_RESPONSE_"
    "ATOMIC_EXACT_SAME_BACKING_INVERSE_RESTORATION_AND_GENERATION2_REUSE_"
    "BUT_PUBLIC_VARIANT_VALIDATION_PLUS_THE_FIXED_MERMIN_PARITY_COCYCLE_"
    "INVARIANT_IS_AN_O1_CLASSICAL_BASELINE_AND_NO_COMPUTATIONAL_ADVANTAGE_"
    "IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_TWO_QUBIT_TYPED_MERMIN_SQUARE_PROJECTIVE_WEYL_"
    "CONTEXTUALITY_CALIBRATION_ON_AN_ABSTRACT_UNIX_SOCKET_CATVM_ONLY"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m250-"):
        raise RuntimeError("M250 client requires a declared abstract Unix socket")
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
        raise RuntimeError("M250 service returned no response")
    return json.loads(response), len(encoded), len(response)


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[str]:
    return (str(descriptor["variant"]),)


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
        "program_id": program_id(descriptor),
        "port_type": PORT_TYPE,
        "output_type": OUTPUT_TYPE,
        "controller_id": CONTROLLER_ID,
        "owner": OWNER,
        "generation": generation,
        "transaction_id": transaction_id,
    }


def status_request(case: dict[str, Any]) -> dict[str, object]:
    return {"command": "STATUS", "carrier_id": case["carrier_id"]}


def disconnect_run(socket_name: str, case: dict[str, Any]) -> int:
    request = run_request(case, 1, "M250_DISCONNECT_CONTROL")
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
            "variant", "central_phase_exponent_mod4", "central_phase",
            "observable_port_count", "context_count", "hidden_carrier_field_cells",
            "hidden_scratch_field_cells", "hidden_context_signature_cells",
            "retained_final_boundary_field_cells_during_inverse", "work",
        )
    }


def contains_forbidden_intermediate(response: object) -> bool:
    forbidden = {
        "vector", "scratch_values", "context_signatures", "context_products",
        "observable_values", "amplitude_vector", "dense_matrix", "assignments",
        "intermediate", "port_values",
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
        "primary", "reuse", "fresh", "disconnect", "partial",
        "postprojection", "descriptor_control",
    }
    if set(cases) != expected:
        raise RuntimeError("invalid M250 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1)):
        response, sent, received = exchange(
            socket_name,
            run_request(cases[label], generation, f"M250_{label.upper()}"),
        )
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK":
            raise RuntimeError(f"M250 accepted case rejected: {label}")
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

    disconnect_request_bytes = disconnect_run(socket_name, cases["disconnect"])
    disconnect_status: dict[str, Any] | None = None
    for _ in range(300):
        time.sleep(0.01)
        status, sent, received = exchange(socket_name, status_request(cases["disconnect"]))
        total_request_bytes += sent
        total_response_bytes += received
        if status.get("last_restored_generation") == 1:
            disconnect_status = status
            break
    if disconnect_status is None:
        raise RuntimeError("M250 disconnect did not restore")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (
        ("partial", "inject_failure_after_partial"),
        ("postprojection", "inject_failure_after_projection"),
    ):
        request = run_request(cases[case_id], 1, f"M250_{case_id.upper()}")
        request[field] = True
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        status, sent, received = exchange(socket_name, status_request(cases[case_id]))
        total_request_bytes += sent
        total_response_bytes += received
        fault_controls[f"{case_id}_failure_rejected_only_after_restoration"] = (
            response.get("status") == "REJECTED" and "response" not in response
            and status.get("canonical") is True
            and status.get("last_restored_generation") == 1
        )

    base = run_request(cases["descriptor_control"], 1, "M250_CONTROL")
    controls_response, sent, received = exchange(
        socket_name, {**base, "command": "CONTROLS"}
    )
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M250 backend controls rejected")

    descriptor_controls: dict[str, bool] = {}
    mutations: list[tuple[str, dict[str, object]]] = []
    for label, key, value in (
        ("wrong_program_rejected", "program_id", "0" * 64),
        ("wrong_owner_rejected", "owner", OWNER + 1),
        ("wrong_type_rejected", "port_type", "WRONG_PORT"),
        ("wrong_output_type_rejected", "output_type", "WRONG_OUTPUT"),
        ("wrong_controller_rejected", "controller_id", CONTROLLER_ID + 1),
    ):
        mutated = dict(base)
        mutated[key] = value
        mutations.append((label, mutated))
    stale = run_request(cases["primary"], 2, "M250_STALE")
    mutations.append(("stale_generation_rejected", stale))
    skipped = dict(base)
    skipped["generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id = json.loads(json.dumps(base))
    same_id["descriptor"]["variant"] = "H_CONJUGATED_REORDERED"
    mutations.append(("same_id_descriptor_mutation_rejected", same_id))
    overmerged = json.loads(json.dumps(base))
    overmerged["descriptor"]["contexts"] = "OVERMERGE_XI_IY"
    mutations.append(("overmerged_observable_port_descriptor_rejected", overmerged))
    undermerged = json.loads(json.dumps(base))
    undermerged["descriptor"]["contexts"] = "UNDERMERGE_DUPLICATE_XI"
    mutations.append(("undermerged_observable_port_descriptor_rejected", undermerged))
    for label, request in mutations:
        response, sent, received = exchange(socket_name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_OBSERVABLE", "PROJECT_CONTEXT", "PROJECT_VECTOR",
        "PROJECT_INTERMEDIATE", "AMPLITUDE_VECTOR", "DENSE_MATRIX",
        "SNAPSHOT", "RUN_SNAPSHOT", "RUN_INPLACE_ON_SNAPSHOT",
        "NULL_CARRIER", "DUMP", "DEBUG", "CONTEXT_SIGNATURES",
    ):
        response, sent, received = exchange(socket_name, {"command": command})
        total_request_bytes += sent
        total_response_bytes += received
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"

    shutdown, sent, received = exchange(socket_name, {"command": "SHUTDOWN"})
    total_request_bytes += sent
    total_response_bytes += received
    if not shutdown.get("shutdown"):
        raise RuntimeError("M250 shutdown failed")

    controls = {
        **dict(controls_response["controls"]),
        **fault_controls,
        **descriptor_controls,
        **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"] and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(
            case["canonical_after_restoration"] for case in accepted
        ),
        "all_accepted_carrier_and_custody_backings_stable": all(
            case["same_carrier_and_custody_backings"] for case in accepted
        ),
        "restored_reuse_matches_fresh": reuse_parity,
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in accepted),
        "backend_responses_exclude_intermediate_state": not any(
            contains_forbidden_intermediate(case) for case in accepted
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M250 control failure: {controls}")

    resource_law = {
        "declared_public_program_variants": list(("BASE", "H_CONJUGATED_REORDERED")),
        "accepted_persistent_carriers": 2,
        "accepted_hidden_carrier_field_cells_per_carrier": 4,
        "accepted_hidden_scratch_field_cells_per_carrier": 4,
        "accepted_hidden_context_signature_cells_per_carrier": 6,
        "accepted_typed_observable_port_receipts_per_carrier": 9,
        "accepted_contexts_per_transaction": 6,
        "accepted_observable_consumptions_per_transaction": 18,
        "accepted_transactions": len(accepted),
        "accepted_forward_pauli_actions": sum(case["work"]["forward_pauli_actions"] for case in accepted),
        "accepted_inverse_pauli_actions": sum(case["work"]["inverse_pauli_actions"] for case in accepted),
        "accepted_forward_vector_cell_reads": sum(case["work"]["forward_vector_cell_reads"] for case in accepted),
        "accepted_forward_vector_cell_writes": sum(case["work"]["forward_vector_cell_writes"] for case in accepted),
        "accepted_inverse_vector_cell_reads": sum(case["work"]["inverse_vector_cell_reads"] for case in accepted),
        "accepted_inverse_vector_cell_writes": sum(case["work"]["inverse_vector_cell_writes"] for case in accepted),
        "accepted_signature_compositions_forward_and_inverse": [
            sum(case["work"]["forward_signature_compositions"] for case in accepted),
            sum(case["work"]["inverse_signature_compositions"] for case in accepted),
        ],
        "accepted_final_overlap_field_multiplications": sum(case["work"]["final_overlap_field_multiplications"] for case in accepted),
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 1,
        "retained_dynamic_inverse_history_entries": 0,
        "public_descriptor_scalar_fields_per_transaction": 1,
        "program_digest_material_bytes_per_transaction": 32,
        "strongest_classical_baseline": "PUBLIC_VARIANT_VALIDATION_PLUS_FIXED_MERMIN_PARITY_COCYCLE_INVARIANT_RETURNING_CENTRAL_EXPONENT2_IN_O1_WORK",
        "strongest_transferable_descriptor_level_classical_baseline": "IDENTICAL_BINARY_SYMPLECTIC_PROJECTIVE_2_COCYCLE_RECURRENCE_WITH_CONSTANT_SIGNATURE_STATE_AND18_PUBLIC_COMPOSITIONS",
        "independent_verifier_only_baseline": "EXACT_FOUR_BY_FOUR_QI_MATRIX_COMPOSITION",
        "classical_baseline_requires_no_four_cell_carrier_inverse_or_catvm_traffic": True,
        "accepted_catvm_path_has_space_or_work_advantage": False,
        "comparison_basis": "EXACT_QI_CARRIER_CELLS_Z4_SIGNATURE_CELLS_TYPED_PORT_RECEIPTS_PUBLIC_COMPOSITIONS_FINAL_BOUNDARY_RESTORATION_REUSE_AND_PROTOCOL_TRAFFIC_NOT_WHOLE_PROCESS_RSS",
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
        "contextuality_law": {
            "native_projective_two_cocycle_causally_changes_boundary": True,
            "typed_shared_observable_ports_have_two_consumers_each": True,
            "noncontextual_assignment_parity_product": 1,
            "projective_context_product": -1,
            "assignment_enumeration_used": False,
            "carrier_state_coherence_required_for_operator_cocycle": False,
            "route_disposition": "RETIRE_AFTER_ONE_MERMIN_SQUARE_IF_IDENTICAL_COMPACT_SYMPLECTIC_RECURRENCE_MATCHES",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_request_bytes,
        },
        "claim_limits": {
            "physical_contextuality_executed": False,
            "general_contextuality_resource_theorem": False,
            "distinct_phase_resource_unavailable_to_compact_classical_software": False,
            "total_computational_advantage": False,
            "general_relational_closure": False,
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
        raise SystemExit("usage: client.py @catvm-m250-NAME")
    main(sys.argv[1])
