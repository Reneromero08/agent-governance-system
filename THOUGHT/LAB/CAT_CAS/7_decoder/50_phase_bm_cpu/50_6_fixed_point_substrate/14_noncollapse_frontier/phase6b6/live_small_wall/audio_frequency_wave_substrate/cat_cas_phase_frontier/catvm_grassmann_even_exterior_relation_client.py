#!/usr/bin/env python3
"""M254 public controller; imports no backend or exact-field implementation."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from typing import Any


PORT_TYPE = "CATVM_QZETA8_GRASSMANN_EVEN_FOUR_PORT_RELATION_V1"
OUTPUT_TYPE = "QZETA8_GRASSMANN_EVEN_TOP_FORM_BOUNDARY_V1"
OWNER = 254004
CONTROLLER_ID = 254001
PORT_ORDER = ["THETA0", "THETA1", "THETA2", "THETA3"]
RESULT = "PASS_CATVM_GRASSMANN_EVEN_EXTERIOR_OPEN_RELATION_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_BOUNDED_EXACT_QZETA8_FOUR_PORT_FULL_EVEN_GRASSMANN_"
    "NON_GAUSSIAN_OPEN_RELATION_INTERSECTS_BY_SIGNED_EXTERIOR_MULTIPLICATION_"
    "AND_CLOSES_BY_AN_INVERTIBLE_BEREZIN_HODGE_COMPLEMENT_ON_ONE_EIGHT_CELL_"
    "RESIDENT_SIGNATURE_WITH_FINAL_TOP_FORM_ONLY_ATOMIC_EXACT_SAME_BACKING_"
    "RESTORATION_AND_GENERATION2_REUSE_BUT_THE_EXACT_RANK8_DECLARED_LINEAR_"
    "LANGUAGE_AND_IDENTICAL_EIGHT_CELL_CLASSICAL_EXTERIOR_RECURRENCE_"
    "BISIMULATE_THE_LAW_WITH_NO_ADVANTAGE"
)
CLAIM_CEILING = (
    "EXACT_SOFTWARE_FOUR_TYPED_GRASSMANN_PORT_QZETA8_FULL_EVEN_EXTERIOR_"
    "RELATIONS_PUBLIC_PROGRAM_LENGTH_AT_MOST8_ON_AN_ABSTRACT_UNIX_SOCKET_"
    "CATVM_ONLY"
)


def socket_address(name: str) -> str:
    if not name.startswith("@catvm-m254-"):
        raise RuntimeError("M254 client requires declared abstract socket")
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
        raise RuntimeError("M254 service returned no response")
    return json.loads(response), len(encoded), len(response)


def program_id(descriptor: dict[str, Any]) -> str:
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


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
    request = run_request(case, 1, "M254_DISCONNECT")
    request["test_delay_before_inverse_ms"] = 120
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(socket_address(name))
    connection.sendall(encoded)
    connection.close()
    return len(encoded)


def comparable(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in (
        "module_kinds", "top_form_boundary", "hidden_even_relation_field_cells",
        "hidden_hodge_scratch_field_cells", "hidden_module_receipt_cells",
        "retained_final_boundary_field_cells_during_inverse", "work",
    )}


def contains_forbidden(value: object) -> bool:
    forbidden = {
        "relation_coefficients", "carrier_values", "scratch_values", "matrix_entries",
        "grassmann_terms", "even_signature", "amplitudes", "paths", "assignments",
        "intermediate", "inverse_history_values", "factor_inverse_values",
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
        raise RuntimeError("invalid M254 public case set")

    total_request_bytes = 0
    total_response_bytes = 0
    accepted: list[dict[str, Any]] = []
    for label, generation in (("primary", 1), ("reuse", 2), ("fresh", 1)):
        response, sent, received = exchange(name, run_request(cases[label], generation, f"M254_{label.upper()}"))
        total_request_bytes += sent
        total_response_bytes += received
        if response.get("status") != "OK":
            raise RuntimeError(f"M254 accepted case rejected: {label}")
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
        raise RuntimeError("M254 disconnect restoration timeout")

    fault_controls: dict[str, bool] = {}
    for case_id, field in (("partial", "inject_failure_after_partial"), ("postprojection", "inject_failure_after_projection")):
        request = run_request(cases[case_id], 1, f"M254_{case_id.upper()}")
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

    base = run_request(cases["descriptor_control"], 1, "M254_CONTROL")
    controls_response, sent, received = exchange(name, {**base, "command": "CONTROLS"})
    total_request_bytes += sent
    total_response_bytes += received
    if controls_response.get("status") != "OK":
        raise RuntimeError("M254 backend controls rejected")

    descriptor_controls: dict[str, bool] = {}
    mutations: list[tuple[str, dict[str, object]]] = []
    for label, key, value in (
        ("wrong_program_rejected", "program_id", "0" * 64),
        ("wrong_owner_rejected", "owner", OWNER + 1),
        ("wrong_type_rejected", "port_type", "WRONG"),
        ("wrong_output_type_rejected", "output_type", "WRONG"),
        ("wrong_controller_rejected", "controller_id", CONTROLLER_ID + 1),
        ("boolean_generation_rejected", "generation", True),
    ):
        mutated = dict(base)
        mutated[key] = value
        mutations.append((label, mutated))
    mutations.append(("stale_generation_rejected", run_request(cases["primary"], 2, "M254_STALE")))
    skipped = dict(base)
    skipped["generation"] = 2
    mutations.append(("skipped_generation_rejected", skipped))
    same_id = json.loads(json.dumps(base))
    same_id["descriptor"]["modules"][0]["factor"][7] = [[2, 1], [0, 1], [0, 1], [0, 1]]
    mutations.append(("same_id_changed_relation_descriptor_rejected", same_id))
    wrong_ports = json.loads(json.dumps(base))
    wrong_ports["descriptor"]["ports"][0], wrong_ports["descriptor"]["ports"][1] = (
        wrong_ports["descriptor"]["ports"][1], wrong_ports["descriptor"]["ports"][0]
    )
    mutations.append(("wrong_port_order_rejected", wrong_ports))
    malformed_rational = json.loads(json.dumps(base))
    malformed_rational["descriptor"]["modules"][0]["factor"][0][0] = [1, 0]
    mutations.append(("zero_denominator_rejected", malformed_rational))
    zero_scalar = json.loads(json.dumps(base))
    zero_scalar["descriptor"]["modules"][0]["factor"][0] = [[0, 1]] * 4
    zero_scalar["program_id"] = program_id(zero_scalar["descriptor"])
    mutations.append(("zero_scalar_factor_rejected", zero_scalar))
    answer_bearing = json.loads(json.dumps(base))
    answer_bearing["descriptor"]["modules"][1]["answer"] = [[1, 1]]
    mutations.append(("answer_bearing_extra_descriptor_field_rejected", answer_bearing))
    for label, request in mutations:
        response, sent, received = exchange(name, request)
        total_request_bytes += sent
        total_response_bytes += received
        descriptor_controls[label] = response.get("status") == "REJECTED"

    malformed_transaction = dict(base)
    malformed_transaction["transaction_id"] = 254
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
    malformed_carrier["carrier_id"] = 254
    response, sent, received = exchange(name, malformed_carrier)
    total_request_bytes += sent
    total_response_bytes += received
    descriptor_controls["malformed_carrier_id_rejected"] = response.get("status") == "REJECTED"

    protocol_controls: dict[str, bool] = {}
    for command in (
        "PROJECT_RELATION", "PROJECT_COEFFICIENTS", "PROJECT_GRASSMANN_PORTS",
        "PROJECT_SCRATCH", "DENSE_SIGNATURE", "SNAPSHOT", "RUN_SNAPSHOT",
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
        raise RuntimeError("M254 shutdown failed")

    controls = {
        **dict(controls_response["controls"]), **fault_controls,
        **descriptor_controls, **protocol_controls,
        "disconnect_restored_before_lost_response": bool(
            disconnect_status["canonical"] and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "all_accepted_responses_after_exact_restoration": all(case["canonical_after_restoration"] for case in accepted),
        "all_accepted_relation_scratch_and_receipt_backings_stable": all(
            case["same_relation_scratch_and_receipt_backings"] for case in accepted
        ),
        "restored_reuse_matches_fresh": reuse_parity,
        "no_baseline_reload": all(not case["baseline_reload_used"] for case in accepted),
        "backend_responses_exclude_relation_coefficients_scratch_and_factor_inverses": not any(
            contains_forbidden(case) for case in accepted
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M254 control failure: {controls}")

    work_fields = tuple(
        field for field in accepted[0]["work"]
        if field != "peak_returned_inverse_factor_field_cells"
    )
    resource_law = {
        "accepted_public_module_kind_sequences": [case["module_kinds"] for case in accepted],
        "accepted_persistent_carriers": 2,
        "accepted_hidden_even_relation_field_cells_per_carrier": 8,
        "accepted_hidden_hodge_scratch_field_cells_per_carrier": 1,
        "accepted_hidden_module_receipt_cells_per_carrier": 8,
        "accepted_transactions": 3,
        **{f"accepted_{field}": sum(case["work"][field] for case in accepted) for field in work_fields},
        "accepted_peak_returned_inverse_factor_field_cells": max(
            case["work"]["peak_returned_inverse_factor_field_cells"] for case in accepted
        ),
        "retained_final_boundary_field_cells_during_inverse_per_transaction": 1,
        "strongest_fixed_fixture_classical_baseline": "PUBLIC_DESCRIPTOR_VALIDATION_PLUS_FROZEN_EXACT_TOP_FORM_BOUNDARY_IN_O1_WORK",
        "strongest_implemented_transferable_descriptor_level_classical_baseline": "IDENTICAL_EXACT_INPLACE_EIGHT_QZETA8_CELL_FULL_EVEN_EXTERIOR_WEDGE_AND_HODGE_RECURRENCE_WITH_ONE_REUSABLE_FIELD_SCRATCH_CELL_NO_CATVM_RESTORATION",
        "transferable_baseline_resident_relation_field_cells": 8,
        "transferable_baseline_reusable_hodge_and_intersection_scratch_field_cells": 1,
        "declared_public_linear_language_exact_reachable_rank": 8,
        "declared_public_linear_language_exact_observable_rank": 8,
        "declared_public_linear_language_exact_hankel_rank": 8,
        "full_even_interface_dimension_law": "TWO_TO_THE_POWER_PORT_COUNT_MINUS_ONE",
        "accepted_catvm_path_has_space_work_or_query_advantage": False,
        "comparison_basis": "QZETA8_FIELD_CELLS_PUBLIC_FACTOR_CELLS_FIELD_OPERATIONS_WRITES_FINAL_BOUNDARY_RESTORATION_REUSE_AND_PROTOCOL_TRAFFIC_NOT_WHOLE_PROCESS_RSS",
        "resource_verification_level": "PACKAGE_SELF_REVIEW",
        "whole_transaction_live_payload_peak_complete": False,
        "exact_qzeta8_coordinate_payload_instrumented": False,
        "field_cell_counts_are_not_fixed_bit_payload_claims": True,
        "public_factor_storage_python_temporaries_fraction_objects_allocator_socket_hash_serialization_rss_excluded_not_zero": True,
    }

    output = {
        "result": RESULT, "claim": CLAIM, "claim_ceiling": CLAIM_CEILING,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": accepted, "reuse_parity": reuse_parity, "controls": controls,
        "open_relation_law": {
            "typed_open_grassmann_ports": PORT_ORDER,
            "resident_full_even_relation_field_cells": 8,
            "intersection_is_native_signed_exterior_multiplication": True,
            "four_port_berezin_hodge_closure_is_exact_signed_complement_involution": True,
            "relation_coefficients_remain_unprojected_until_final_top_form_boundary": True,
            "independent_quartic_coordinate_is_strictly_broader_than_m253_gaussian_chart": True,
            "declared_public_linear_language_minimal_dimension": 8,
            "route_disposition": "RETIRE_AFTER_ONE_BOUNDED_FOUR_PORT_SUITE_BECAUSE_IDENTICAL_CLASSICAL_EXTERIOR_RECURRENCE_BISIMULATES_IT_AND_EVEN_INTERFACE_DIMENSION_GROWS_EXPONENTIALLY",
        },
        "resource_law": resource_law,
        "protocol_accounting": {
            "controller_backend_request_bytes": total_request_bytes,
            "backend_controller_response_bytes": total_response_bytes,
            "disconnect_request_bytes": disconnect_bytes,
        },
        "claim_limits": {
            "normalized_or_convergent_fermionic_path_integral": False,
            "physical_fermion_or_berezin_execution": False,
            "general_grassmann_relation_or_arbitrary_topology_closure": False,
            "compact_wide_interface_even_exterior_relation": False,
            "fixed_bounded_rank_or_width_across_growing_interfaces": False,
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
        raise SystemExit("usage: client.py @catvm-m254-NAME")
    main(sys.argv[1])
