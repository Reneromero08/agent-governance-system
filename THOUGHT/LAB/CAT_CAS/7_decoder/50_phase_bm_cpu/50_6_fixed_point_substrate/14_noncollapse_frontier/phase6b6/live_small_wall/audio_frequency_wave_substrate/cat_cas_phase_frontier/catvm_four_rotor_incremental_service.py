#!/usr/bin/env python3
"""Isolated CATVM custody service for the incremental four-rotor carrier."""

from __future__ import annotations

import ctypes
import os
import resource
import socket
import struct
import sys
import time

import catvm_four_rotor_incremental_backend as backend
import catvm_four_rotor_incremental_protocol as protocol
import four_rotor_kicked_phase_tt as reference


PR_GET_DUMPABLE = 3
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_SET_PTRACER = 0x59616D61
MODES = {"ISOLATED", "SNAPSHOT", "IN_PLACE"}
DENIED_COMMANDS = {
    "PROJECT_INTERMEDIATE",
    "PROJECT_TENSORS",
    "PROJECT_SCHMIDT",
    "PROJECT_RANKS",
    "SERIALIZE_INTERMEDIATE",
    "DEBUG_DUMP",
    "DUMP",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def harden_process() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        fail("CATVM four-rotor non-dumpable setup failed")
    if libc.prctl(PR_SET_PTRACER, 0, 0, 0, 0) != 0:
        fail("CATVM four-rotor ptracer setup failed")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail("CATVM four-rotor no-new-privileges setup failed")
    if libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        fail("CATVM four-rotor dumpable verification failed")


def peer_uid(connection: socket.socket) -> int:
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, 12
    )
    _, uid, _ = struct.unpack("3i", credentials)
    return uid


class Service:
    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            fail("CATVM four-rotor service mode invalid")
        self.mode = mode
        self.transactions = 0
        self.carrier_creation_count = 0
        self.restoration_generation = 0
        self.snapshot_copy_bytes = 0
        self.carrier_creation_ns = 0
        self.snapshot_creation_ns = 0
        self.carrier = None
        self.snapshot_image = None
        self.snapshot_payload_bytes = 0
        if mode == "IN_PLACE":
            creation_start = time.perf_counter_ns()
            self.carrier = reference.product_zero_carrier(
                reference.MODE_RADIUS
            )
            self.carrier_creation_ns = (
                time.perf_counter_ns() - creation_start
            )
            self.carrier_creation_count = 1
        elif mode == "SNAPSHOT":
            creation_start = time.perf_counter_ns()
            baseline = reference.product_zero_carrier(
                reference.MODE_RADIUS
            )
            self.carrier_creation_count = 1
            self.snapshot_image = reference.copy_carrier(baseline)
            self.carrier_creation_ns = (
                time.perf_counter_ns() - creation_start
            )
            self.snapshot_creation_ns = self.carrier_creation_ns
            self.snapshot_payload_bytes = backend.carrier_payload_bytes(
                self.snapshot_image
            )
            self.snapshot_copy_bytes = self.snapshot_payload_bytes
        backend.warm_runtime()

    def run(
        self,
        program: str,
        transaction_id: int,
    ) -> dict[str, object]:
        transaction_start = time.perf_counter_ns()
        transaction_carrier_creation_ns = 0
        snapshot_execution_load_ns = 0
        snapshot_restoration_reload_ns = 0
        if program not in backend.PROGRAMS:
            fail("CATVM four-rotor program invalid")
        if self.mode == "IN_PLACE":
            if self.carrier is None:
                fail("CATVM four-rotor carrier missing")
            result = backend.in_place(
                self.carrier, program, transaction_id
            )
            self.restoration_generation = int(
                result["restoration_generation"]
            )
            snapshot_loaded = False
        elif self.mode == "SNAPSHOT":
            if self.snapshot_image is None:
                fail("CATVM four-rotor snapshot missing")
            load_start = time.perf_counter_ns()
            working = reference.copy_carrier(self.snapshot_image)
            snapshot_execution_load_ns = (
                time.perf_counter_ns() - load_start
            )
            self.snapshot_copy_bytes += self.snapshot_payload_bytes
            result = backend.forward_only(
                self.mode,
                working,
                program,
                transaction_id,
                self.snapshot_payload_bytes,
            )
            reload_start = time.perf_counter_ns()
            restored = reference.copy_carrier(self.snapshot_image)
            snapshot_restoration_reload_ns = (
                time.perf_counter_ns() - reload_start
            )
            self.snapshot_copy_bytes += self.snapshot_payload_bytes
            del working, restored
            snapshot_loaded = True
        else:
            creation_start = time.perf_counter_ns()
            working = reference.product_zero_carrier(
                reference.MODE_RADIUS
            )
            transaction_carrier_creation_ns = (
                time.perf_counter_ns() - creation_start
            )
            self.carrier_creation_count += 1
            result = backend.forward_only(
                self.mode,
                working,
                program,
                transaction_id,
            )
            del working
            snapshot_loaded = False
        self.transactions += 1
        resources = result["resources"]
        if not isinstance(resources, dict):
            fail("CATVM four-rotor resource receipt malformed")
        resources = {
            **resources,
            "snapshot_copy_bytes_cumulative": self.snapshot_copy_bytes,
            "service_init_carrier_creation_ns": (
                self.carrier_creation_ns
            ),
            "snapshot_creation_ns": self.snapshot_creation_ns,
            "transaction_carrier_creation_ns": (
                transaction_carrier_creation_ns
            ),
            "snapshot_execution_load_ns": snapshot_execution_load_ns,
            "snapshot_restoration_reload_ns": (
                snapshot_restoration_reload_ns
            ),
            "logical_request_bytes": protocol.REQUEST_BYTES,
            "logical_response_bytes": protocol.RESPONSE_BYTES,
        }
        response = {
            "version": protocol.PROTOCOL_VERSION,
            "status": "PASS",
            "arm": self.mode,
            "program": program,
            "transaction_id": transaction_id,
            "final_boundary": result["final_boundary"],
            "actual_inverse_restoration": result[
                "actual_inverse_restoration"
            ],
            "canonical_restoration": result[
                "canonical_restoration"
            ],
            "restoration_error": result["restoration_error"],
            "restoration_generation": result[
                "restoration_generation"
            ],
            "snapshot_loaded": snapshot_loaded,
            "carrier_creation_count": self.carrier_creation_count,
            "custody_receipt": result["custody_receipt"],
            "resources": resources,
        }
        resources["service_transaction_ns"] = (
            time.perf_counter_ns() - transaction_start
        )
        return response

    def dispatch(
        self, request: dict[str, object]
    ) -> dict[str, object]:
        command = request.get("command")
        transaction_id = request.get("transaction_id")
        if not isinstance(transaction_id, int):
            fail("CATVM four-rotor transaction id invalid")
        if command == "HELLO":
            return {
                "version": protocol.PROTOCOL_VERSION,
                "status": "PASS",
                "arm": self.mode,
                "transaction_id": transaction_id,
            }
        if command == "RUN":
            program = request.get("program")
            if not isinstance(program, str):
                fail("CATVM four-rotor program missing")
            return self.run(program, transaction_id)
        if command in DENIED_COMMANDS:
            return {
                "version": protocol.PROTOCOL_VERSION,
                "status": "DENIED",
                "error": "four-rotor intermediate projection denied",
                "transaction_id": transaction_id,
            }
        if command == "NULL_CARRIER":
            return {
                "version": protocol.PROTOCOL_VERSION,
                "status": "DENIED",
                "error": "four-rotor null carrier denied",
                "transaction_id": transaction_id,
            }
        if command == "STATUS":
            return {
                "version": protocol.PROTOCOL_VERSION,
                "status": "PASS",
                "arm": self.mode,
                "transaction_id": transaction_id,
                "transactions": self.transactions,
                "restoration_generation": self.restoration_generation,
                "carrier_creation_count": self.carrier_creation_count,
            }
        if command == "STOP":
            return {
                "version": protocol.PROTOCOL_VERSION,
                "status": "STOP",
                "transaction_id": transaction_id,
            }
        fail("CATVM four-rotor command invalid")


def main() -> None:
    if len(sys.argv) != 3:
        fail(
            "usage: catvm_four_rotor_incremental_service.py SOCKET MODE"
        )
    socket_path, mode = sys.argv[1:]
    if os.path.exists(socket_path):
        fail("CATVM four-rotor socket already exists")
    harden_process()
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
                    response = {
                        "version": protocol.PROTOCOL_VERSION,
                        "status": "DENIED",
                        "error": "four-rotor peer credential denied",
                        "transaction_id": -1,
                    }
                else:
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
