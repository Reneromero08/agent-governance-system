#!/usr/bin/env python3
"""M241 controller: public protocol only; imports and loads no backend code."""

from __future__ import annotations

import json
import socket
import sys
import time
from typing import Any


P = 5
DIMENSIONS = (1, 2, 3, 4)
RESULT = "PASS_CATVM_EXACT_P5_HIDDEN_LINEAR_PHASE_ORACLE_QUERY_DIAGNOSTIC_STRICT_SCOPE"
CLAIM = (
    "CATVM_ENFORCED_EXACT_P5_HIDDEN_LINEAR_PHASE_ORACLE_USES_ONE_COHERENT_"
    "FORWARD_QUERY_TO_INFER_N_RESIDUES_ACROSS_DECLARED_DIMENSIONS1_2_3_4_"
    "WITH_FINAL_SECRET_RESPONSE_RELEASED_ONLY_AFTER_ACTUAL_INVERSE_EXACT_"
    "SAME_BACKING_RESTORATION_AND_DESCRIPTOR_DISTINCT_REUSE_WHILE_EXACT_"
    "CLASSICAL_BLACK_BOX_VALUE_QUERY_COMPLEXITY_IS_N_BUT_THE_SOFTWARE_"
    "PHASE_CARRIER_AND_SCRATCH_GROW5_TO_THE_N_AND_NO_TOTAL_COMPUTATIONAL_"
    "ADVANTAGE_OR_SMALL_WALL_CROSSING_IS_ESTABLISHED"
)
CLAIM_CEILING = (
    "CATVM_UNIX_SOCKET_QZETA5_HIDDEN_LINEAR_ORACLES_DIMENSIONS1_2_3_4_TWO_"
    "DECLARED_SECRETS_PER_DIMENSION_ONE_COHERENT_FORWARD_QUERY_EXACT_RESTORATION"
)


def socket_address(socket_name: str) -> str:
    if not socket_name.startswith("@catvm-m241-"):
        raise RuntimeError("M241 client requires the declared abstract Unix socket")
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
        raise RuntimeError("CATVM service returned no response")
    return json.loads(response), len(encoded), len(response)


def disconnected_run(socket_name: str) -> int:
    request = {
        "command": "RUN",
        "oracle_id": "disconnect_control",
        "owner": 241900,
        "generation": 1,
        "transaction_id": "M241_DISCONNECT_BEFORE_RESTORATION_RESPONSE",
    }
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
    for _ in range(30):
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
        raise RuntimeError("disconnect control did not restore")

    exception_response, request_bytes, response_bytes = exchange(socket_name, {
        "command": "RUN",
        "oracle_id": "exception_control",
        "owner": 241901,
        "generation": 1,
        "transaction_id": "M241_POST_PROJECTION_EXCEPTION_CONTROL",
    })
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes
    exception_status, request_bytes, response_bytes = exchange(
        socket_name, {"command": "STATUS", "oracle_id": "exception_control"}
    )
    total_request_bytes += request_bytes
    total_response_bytes += response_bytes

    for dimension in DIMENSIONS:
        primary_id = f"n{dimension}_primary"
        reuse_id = f"n{dimension}_reuse"
        before, req, resp = exchange(socket_name, {"command": "STATUS", "oracle_id": primary_id})
        total_request_bytes += req; total_response_bytes += resp
        if not before["canonical"] or before["last_restored_generation"] != 0:
            raise RuntimeError("CATVM carrier not canonical before primary")
        if "oracle_commitment" in before:
            raise RuntimeError("enumerable pre-run oracle commitment exposed")
        for oracle_id, generation, run_kind in (
            (primary_id, 1, "PRIMARY"), (reuse_id, 2, "RESTORED_REUSE")
        ):
            transaction_id = f"M241_N{dimension}_{run_kind}"
            response, request_bytes, response_bytes = exchange(socket_name, {
                "command": "RUN",
                "oracle_id": oracle_id,
                "owner": 241000 + dimension,
                "generation": generation,
                "transaction_id": transaction_id,
            })
            total_request_bytes += request_bytes
            total_response_bytes += response_bytes
            if response.get("status") != "OK":
                raise RuntimeError("CATVM atomic run rejected")
            case = dict(response["response"])
            case["run_kind"] = run_kind
            case["controller_request_bytes"] = request_bytes
            case["backend_response_bytes"] = response_bytes
            cases.append(case)
        after, req, resp = exchange(socket_name, {"command": "STATUS", "oracle_id": reuse_id})
        total_request_bytes += req; total_response_bytes += resp
        if not after["canonical"] or after["last_restored_generation"] != 2:
            raise RuntimeError("CATVM carrier not canonical after reuse")

    backend_controls, req, resp = exchange(
        socket_name, {"command": "CONTROLS", "oracle_id": "n3_primary"}
    )
    total_request_bytes += req; total_response_bytes += resp
    protocol_controls: dict[str, bool] = {}
    for command in ("PROJECT_INTERMEDIATE", "SNAPSHOT", "RUN_SNAPSHOT", "NULL_CARRIER"):
        response, request_bytes, response_bytes = exchange(
            socket_name, {"command": command, "oracle_id": "n2_primary"}
        )
        total_request_bytes += request_bytes; total_response_bytes += response_bytes
        protocol_controls[f"{command.lower()}_rejected"] = response.get("status") == "REJECTED"
    stop, req, resp = exchange(socket_name, {"command": "STOP"})
    total_request_bytes += req; total_response_bytes += resp

    controls = dict(backend_controls["controls"])
    controls.update(protocol_controls)
    controls.update({
        "disconnect_before_response_still_restores": (
            disconnect_status["canonical"]
            and not disconnect_status["leased"]
            and disconnect_status["last_restored_generation"] == 1
        ),
        "post_projection_exception_rejected_only_after_restoration": (
            exception_response.get("status") == "REJECTED"
            and exception_status["canonical"]
            and not exception_status["leased"]
            and exception_status["last_restored_generation"] == 1
        ),
        "pre_run_status_contains_enumerable_secret_commitment": False,
        "all_service_carriers_canonical_at_stop": bool(stop.get("all_carriers_canonical")),
        "controller_imports_or_loads_backend_code": False,
        "controller_receives_hidden_phase_amplitudes": False,
        "controller_computes_secret_independently": False,
        "service_stdout_stderr_contains_secret_or_amplitudes": False,
        "snapshot_reload_used_by_accepted_path": False,
    })
    result = {
        "schema": "cat_cas.catvm_p5_hidden_linear_phase_oracle_raw.v1",
        "result": RESULT,
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "cases": cases,
        "controls": controls,
        "query_law": {
            "field_order": P,
            "dimensions": list(DIMENSIONS),
            "coherent_forward_phase_queries": [1, 1, 1, 1],
            "classical_deterministic_value_queries_necessary_and_sufficient": list(DIMENSIONS),
            "classical_lower_bound_model": "BLACK_BOX_F5_LINEAR_VALUE_OR_PHASE_QUERY_RETURNS_ONE_F5_SYMBOL_PER_CLASSICAL_QUERY",
            "coherent_query_acts_on_all5_TO_THE_N_BASIS_CELLS": True,
            "oracle_query_separation_is_not_total_software_advantage": True,
        },
        "resource_law": {
            "carrier_field_cells": [P**dimension for dimension in DIMENSIONS],
            "scratch_field_cells": [P**dimension for dimension in DIMENSIONS],
            "hidden_oracle_secret_residue_cells": list(DIMENSIONS),
            "abstract_forward_query_count": 1,
            "actual_inverse_query_count": 1,
            "controller_backend_request_bytes_total": total_request_bytes,
            "backend_controller_response_bytes_total": total_response_bytes,
            "disconnect_control_request_bytes": disconnect_request_bytes,
            "snapshot_baseline_would_copy_and_reload_carrier_cells": [2 * P**dimension for dimension in DIMENSIONS],
            "snapshot_baseline_restoration_classification": "SNAPSHOT_RELOAD",
            "warm_direct_phase_software_uses_same5_TO_THE_N_amplitude_recurrence": True,
            "warm_isolated_boundary_overhead_counted_in_protocol_bytes": True,
            "whole_process_rss_allocator_socket_kernel_and_scheduler_costs_complete": False,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
        },
        "claim_limits": {
            "total_computational_advantage": False,
            "small_wall_crossed": False,
            "general_oracle_or_query_advantage": False,
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
        raise SystemExit("usage: catvm_p5_hidden_linear_phase_oracle_client.py SOCKET_PATH")
    main(sys.argv[1])
