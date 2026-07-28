#!/usr/bin/env python3
"""Minimal CATVM custody service for the exact cyclotomic cubic TT carrier."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import socket
import struct
import sys
from copy import deepcopy

import catvm_cyclotomic_f5_tt_protocol as protocol
import cyclotomic_f5_cubic_tt_phase as phase


WIDTH = 4
PR_SET_DUMPABLE = 4


def fail(message: str) -> None:
    raise RuntimeError(message)


def recursive_size(value: object) -> int:
    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(item)
        if isinstance(item, dict):
            total += sum(
                visit(key) + visit(member)
                for key, member in item.items()
            )
        elif isinstance(item, (list, tuple, set, frozenset)):
            total += sum(visit(member) for member in item)
        elif hasattr(item, "__dict__"):
            total += visit(vars(item))
        return total

    return visit(value)


def peer_uid(connection: socket.socket) -> int:
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, 12
    )
    _, uid, _ = struct.unpack("3i", credentials)
    return uid


def boundary_from_result(
    result: dict[str, object],
) -> tuple[list[int], list[int]]:
    return (
        [int(value) for value in result["boundary_numerators"]],
        [int(value) for value in result["boundary_denominators"]],
    )


def receipt(
    program: str,
    generation: int,
    numerators: list[int],
    denominators: list[int],
) -> str:
    payload = json.dumps(
        {
            "program": program,
            "generation": generation,
            "numerators": numerators,
            "denominators": denominators,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class Service:
    def __init__(self, mode: str) -> None:
        if mode not in ("IN_PLACE", "SNAPSHOT"):
            fail("CATVM cyclotomic service mode invalid")
        self.mode = mode
        self.carrier = phase.product_zero_state(WIDTH)
        self.snapshot_image = (
            deepcopy(self.carrier)
            if mode == "SNAPSHOT"
            else None
        )
        self.snapshot_logical_payload_bytes = (
            phase.logical_payload_bytes(self.snapshot_image)
            if self.snapshot_image is not None
            else 0
        )
        self.snapshot_image_python_resident_bytes = (
            recursive_size(self.snapshot_image)
            if self.snapshot_image is not None
            else 0
        )
        self.generation = 0
        self.transactions = 0

    def run_in_place(
        self, program: str
    ) -> dict[str, object]:
        if self.mode != "IN_PLACE":
            fail("in-place command denied in snapshot service")
        if program == "PRIMARY":
            rounds, program_id = 4, 0
        elif program == "REUSE":
            rounds, program_id = 3, 1
        else:
            fail("CATVM cyclotomic program invalid")
        result = phase.transaction(
            self.carrier, WIDTH, rounds, program_id
        )
        if not result["actual_inverse_restoration"]:
            fail("CATVM cyclotomic restoration failed")
        self.generation += 1
        self.transactions += 1
        numerators, denominators = boundary_from_result(result)
        return {
            "status": "PASS",
            "program": program,
            "boundary_numerators": numerators,
            "boundary_denominators": denominators,
            "restoration_generation": self.generation,
            "actual_inverse_restoration": True,
            "snapshot_loaded": False,
            "custody_receipt": receipt(
                program,
                self.generation,
                numerators,
                denominators,
            ),
        }

    def run_snapshot_sham(self) -> dict[str, object]:
        if self.mode != "SNAPSHOT" or self.snapshot_image is None:
            fail("snapshot command denied in in-place service")
        sham = deepcopy(self.snapshot_image)
        working_python_resident_bytes = recursive_size(sham)
        stats = phase.Stats()
        schedule = phase.bond_schedule(WIDTH, 4, 0)
        for operation in schedule:
            phase.apply_operation(sham, operation, False, stats)
        boundary = phase.boundary_amplitude(sham, stats)
        self.carrier = deepcopy(self.snapshot_image)
        restored_python_resident_bytes = recursive_size(self.carrier)
        numerators = [
            coefficient.numerator for coefficient in boundary
        ]
        denominators = [
            coefficient.denominator for coefficient in boundary
        ]
        self.transactions += 1
        return {
            "status": "PASS",
            "program": "PRIMARY",
            "boundary_numerators": numerators,
            "boundary_denominators": denominators,
            "restoration_generation": 0,
            "actual_inverse_restoration": False,
            "snapshot_loaded": True,
            "snapshot_logical_payload_bytes": (
                self.snapshot_logical_payload_bytes
            ),
            "snapshot_creation_logical_copy_bytes": (
                self.snapshot_logical_payload_bytes
            ),
            "snapshot_execution_load_logical_copy_bytes": (
                self.snapshot_logical_payload_bytes
            ),
            "snapshot_restoration_reload_logical_copy_bytes": (
                self.snapshot_logical_payload_bytes
            ),
            "snapshot_total_logical_copy_bytes": (
                3 * self.snapshot_logical_payload_bytes
            ),
            "snapshot_image_python_resident_bytes": (
                self.snapshot_image_python_resident_bytes
            ),
            "snapshot_working_python_resident_bytes": (
                working_python_resident_bytes
            ),
            "snapshot_restored_python_resident_bytes": (
                restored_python_resident_bytes
            ),
            "custody_receipt": receipt(
                "SNAPSHOT_PRIMARY", 0, numerators, denominators
            ),
        }

    def dispatch(
        self, request: dict[str, object]
    ) -> dict[str, object]:
        command = request.get("command")
        if command == "RUN":
            if self.mode != "IN_PLACE":
                return {
                    "status": "DENIED",
                    "error": (
                        "in-place command denied in snapshot service"
                    ),
                }
            program = request.get("program")
            if not isinstance(program, str):
                fail("CATVM cyclotomic program missing")
            return self.run_in_place(program)
        if command == "SNAPSHOT_PRIMARY":
            if self.mode != "SNAPSHOT":
                return {
                    "status": "DENIED",
                    "error": (
                        "snapshot command denied in in-place service"
                    ),
                }
            return self.run_snapshot_sham()
        if command == "PROJECT_INTERMEDIATE":
            return {
                "status": "DENIED",
                "error": "cyclotomic TT intermediate projection denied",
            }
        if command == "NULL_CARRIER":
            return {
                "status": "DENIED",
                "error": "invalid cyclotomic TT carrier",
            }
        if command == "STATUS":
            return {
                "status": "PASS",
                "transactions": self.transactions,
                "restoration_generation": self.generation,
            }
        if command == "STOP":
            return {"status": "STOP"}
        fail("CATVM cyclotomic command invalid")


def main() -> None:
    if len(sys.argv) != 3:
        fail(
            "usage: catvm_cyclotomic_f5_tt_service.py SOCKET MODE"
        )
    socket_path = sys.argv[1]
    mode = sys.argv[2]
    if os.path.exists(socket_path):
        fail("CATVM cyclotomic socket already exists")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        fail("CATVM cyclotomic non-dumpable setup failed")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.bind(socket_path)
    os.chmod(socket_path, 0o600)
    listener.listen(1)
    service = Service(mode)
    try:
        running = True
        while running:
            connection, _ = listener.accept()
            with connection:
                if peer_uid(connection) != os.getuid():
                    connection.sendall(
                        protocol.fixed_packet(
                            {
                                "status": "DENIED",
                                "error": "peer credential denied",
                            },
                            protocol.RESPONSE_BYTES,
                        )
                    )
                    continue
                request = protocol.parse_packet(
                    connection.recv(protocol.REQUEST_BYTES + 1),
                    protocol.REQUEST_BYTES,
                )
                response = service.dispatch(request)
                connection.sendall(
                    protocol.fixed_packet(
                        response, protocol.RESPONSE_BYTES
                    )
                )
                running = response.get("status") != "STOP"
    finally:
        listener.close()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
