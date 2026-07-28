#!/usr/bin/env python3
"""Warm direct-process compact baseline for the four-rotor CATVM triad."""

from __future__ import annotations

import json
import time

import catvm_four_rotor_incremental_backend as backend
import catvm_four_rotor_incremental_protocol as protocol
import four_rotor_kicked_phase_tt as reference


def response(
    program: str, transaction_id: int, creation_count: int
) -> dict[str, object]:
    transaction_start = time.perf_counter_ns()
    creation_start = time.perf_counter_ns()
    carrier = reference.product_zero_carrier(reference.MODE_RADIUS)
    carrier_creation_ns = time.perf_counter_ns() - creation_start
    result = backend.forward_only(
        "DIRECT_PROCESS", carrier, program, transaction_id
    )
    resources = result["resources"]
    if not isinstance(resources, dict):
        raise RuntimeError("direct baseline resources malformed")
    value = {
        "version": protocol.PROTOCOL_VERSION,
        "status": "PASS",
        "arm": "DIRECT_PROCESS",
        "program": program,
        "transaction_id": transaction_id,
        "final_boundary": result["final_boundary"],
        "actual_inverse_restoration": False,
        "canonical_restoration": False,
        "restoration_error": None,
        "restoration_generation": 0,
        "snapshot_loaded": False,
        "carrier_creation_count": creation_count,
        "custody_receipt": result["custody_receipt"],
        "resources": {
            **resources,
            "snapshot_copy_bytes_cumulative": 0,
            "service_init_carrier_creation_ns": 0,
            "snapshot_creation_ns": 0,
            "transaction_carrier_creation_ns": carrier_creation_ns,
            "snapshot_execution_load_ns": 0,
            "snapshot_restoration_reload_ns": 0,
            "logical_request_bytes": protocol.REQUEST_BYTES,
            "logical_response_bytes": protocol.RESPONSE_BYTES,
        },
    }
    value["resources"]["service_transaction_ns"] = (
        time.perf_counter_ns() - transaction_start
    )
    protocol.fixed_packet(value, protocol.RESPONSE_BYTES)
    return value


def main() -> None:
    backend.warm_runtime()
    primary = response("PRIMARY", 1, 1)
    reuse = response("REUSE", 2, 2)
    protocol_equivalent_count = 7
    for transaction_id, command in enumerate(
        (
            "HELLO",
            "RUN",
            "RUN",
            "PROJECT_INTERMEDIATE",
            "NULL_CARRIER",
            "STATUS",
            "STOP",
        )
    ):
        protocol.fixed_packet(
            protocol.request(command, transaction_id),
            protocol.REQUEST_BYTES,
        )
        protocol.fixed_packet(
            {
                "version": protocol.PROTOCOL_VERSION,
                "status": "SYNTHETIC_PROTOCOL_SHAPE",
                "transaction_id": transaction_id,
            },
            protocol.RESPONSE_BYTES,
        )
    print(
        json.dumps(
            {
                "result": "PASS",
                "arm": "DIRECT_PROCESS",
                "primary": primary,
                "reuse": reuse,
                "protocol_equivalent_request_count": (
                    protocol_equivalent_count
                ),
                "protocol_equivalent_response_count": (
                    protocol_equivalent_count
                ),
                "request_bytes_each": protocol.REQUEST_BYTES,
                "response_bytes_each": protocol.RESPONSE_BYTES,
                "simulated_matched_protocol_traffic": True,
                "protocol_equivalent_logical_bytes": (
                    protocol_equivalent_count
                    * (
                        protocol.REQUEST_BYTES
                        + protocol.RESPONSE_BYTES
                    )
                ),
                "actual_controller_backend_traffic_bytes": 0,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
