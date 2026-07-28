#!/usr/bin/env python3
"""Protocol-only controller for the incremental four-rotor CATVM."""

from __future__ import annotations

import json
import socket
import sys
import time

import catvm_four_rotor_incremental_protocol as protocol


def exchange(
    socket_path: str, request: dict[str, object]
) -> dict[str, object]:
    packet = protocol.fixed_packet(request, protocol.REQUEST_BYTES)
    with socket.socket(
        socket.AF_UNIX, socket.SOCK_SEQPACKET
    ) as connection:
        connection.connect(socket_path)
        connection.sendall(packet)
        response = connection.recv(protocol.RESPONSE_BYTES + 1)
    return protocol.parse_packet(response, protocol.RESPONSE_BYTES)


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError(
            "usage: catvm_four_rotor_incremental_controller.py SOCKET"
        )
    socket_path = sys.argv[1]
    requests = [
        protocol.request("HELLO", 0),
        protocol.request("RUN", 1, program="PRIMARY"),
        protocol.request("RUN", 2, program="REUSE"),
        protocol.request("PROJECT_INTERMEDIATE", 3),
        protocol.request("NULL_CARRIER", 4),
        protocol.request("STATUS", 5),
        protocol.request("STOP", 6),
    ]
    controller_start = time.perf_counter_ns()
    responses = []
    roundtrip_ns = []
    for request in requests:
        exchange_start = time.perf_counter_ns()
        responses.append(exchange(socket_path, request))
        roundtrip_ns.append(
            time.perf_counter_ns() - exchange_start
        )
    controller_total_ns = time.perf_counter_ns() - controller_start
    (
        hello,
        primary,
        reuse,
        projection,
        null_carrier,
        status,
        stopped,
    ) = responses
    if not (
        hello.get("status") == "PASS"
        and primary.get("status") == "PASS"
        and reuse.get("status") == "PASS"
        and projection.get("status") == "DENIED"
        and null_carrier.get("status") == "DENIED"
        and status.get("status") == "PASS"
        and status.get("transactions") == 2
        and stopped.get("status") == "STOP"
    ):
        raise RuntimeError("CATVM four-rotor controller gate failed")
    print(
        json.dumps(
            {
                "result": "PASS",
                "arm": primary["arm"],
                "primary": primary,
                "reuse": reuse,
                "projection_control": projection,
                "null_carrier_control": null_carrier,
                "status": status,
                "request_count": len(requests),
                "response_count": len(responses),
                "request_bytes_each": protocol.REQUEST_BYTES,
                "response_bytes_each": protocol.RESPONSE_BYTES,
                "roundtrip_ns": roundtrip_ns,
                "primary_roundtrip_ns": roundtrip_ns[1],
                "reuse_roundtrip_ns": roundtrip_ns[2],
                "controller_total_ns": controller_total_ns,
                "actual_controller_backend_traffic_bytes": (
                    len(requests)
                    * (
                        protocol.REQUEST_BYTES
                        + protocol.RESPONSE_BYTES
                    )
                ),
                "controller_phase_engine_loaded": False,
                "controller_service_module_loaded": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
