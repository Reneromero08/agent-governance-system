#!/usr/bin/env python3
"""Fixed-traffic controller for the cyclotomic TT CATVM service."""

from __future__ import annotations

import json
import socket
import sys

import catvm_cyclotomic_f5_tt_protocol as protocol


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
            "usage: catvm_cyclotomic_f5_tt_controller.py SOCKET"
        )
    socket_path = sys.argv[1]
    primary = exchange(
        socket_path, {"command": "RUN", "program": "PRIMARY"}
    )
    reuse = exchange(
        socket_path, {"command": "RUN", "program": "REUSE"}
    )
    status = exchange(socket_path, {"command": "STATUS"})
    if not (
        primary.get("status") == "PASS"
        and primary.get("actual_inverse_restoration") is True
        and primary.get("snapshot_loaded") is False
        and primary.get("restoration_generation") == 1
        and reuse.get("status") == "PASS"
        and reuse.get("actual_inverse_restoration") is True
        and reuse.get("snapshot_loaded") is False
        and reuse.get("restoration_generation") == 2
        and status.get("transactions") == 2
        and status.get("restoration_generation") == 2
    ):
        raise RuntimeError("CATVM cyclotomic custody control failed")
    print(
        json.dumps(
            {
                "result": "PASS",
                "claim_candidate": (
                    "CATVM_ENFORCED_CYCLOTOMIC_CUBIC_TT_HIDDEN_"
                    "BOND_COMPOSITION_WITH_ACTUAL_RESTORATION_"
                    "AND_REUSE"
                ),
                "primary": primary,
                "reuse": reuse,
                "status": status,
                "request_bytes_each": protocol.REQUEST_BYTES,
                "response_bytes_each": protocol.RESPONSE_BYTES,
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
