#!/usr/bin/env python3
"""Atomic CATVM service for exact S3 translation-relation phase programs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import catvm_s3_relation_protocol as protocol


FIELD = 103
ZETA6 = 57
DEPTHS = (1, 4, 16, 64, 256, 1024)
ELEMENTS = tuple(itertools.permutations(range(3)))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
IDENTITY = INDEX[(0, 1, 2)]
PORT_TYPE = 0x53335245
OWNER_A = 0xA103
OWNER_B = 0xB103
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38


def fail(message: str) -> None:
    raise RuntimeError(message)


def multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[position]] for position in range(3))


def inverse(element: tuple[int, ...]) -> tuple[int, ...]:
    result = [0, 0, 0]
    for source, target in enumerate(element):
        result[target] = source
    return tuple(result)


def phase(exponent: int) -> int:
    return pow(ZETA6, exponent % 6, FIELD)


def family_name(family: int) -> str:
    if family == 1:
        return "PRIMARY"
    if family == 2:
        return "ALTERNATE"
    fail("invalid public family")


def seed(register: int) -> list[int]:
    code = 1
    offset = 1 if register == 0 else 4
    return [phase(offset + code * (position + 1) + position * position) for position in range(6)]


def public_vector(index: int, family: int, kind: int) -> tuple[int, ...]:
    code = family
    return tuple(phase((kind + 1) * index + (2 * kind + code) * position + position * position + code) for position in range(6))


def scalar(index: int, family: int, offset: int) -> int:
    return phase((offset + 1) * index + offset * family + 1)


@dataclass(frozen=True)
class Stage:
    target: int
    operation: str
    scalar: int
    public_operand: tuple[int, ...]


def stage(index: int, family: int, position: int) -> Stage:
    if family not in (1, 2) or position not in range(4):
        fail("invalid public stage")
    base_position = position if family == 1 else 3 - position
    targets = (1, 0, 1, 0)
    operations = ("RIGHT_COMPOSE", "INTERSECT", "LEFT_COMPOSE", "INTERSECT")
    kinds = (1, 2, 3, 4)
    offsets = (1, 2, 4, 5)
    return Stage(
        targets[base_position],
        operations[base_position],
        scalar(index, family, offsets[base_position]),
        public_vector(index, family, kinds[base_position]),
    )


def configure_process_boundary() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        fail("PR_SET_DUMPABLE failed")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail("PR_SET_NO_NEW_PRIVS failed")


class Service:
    def __init__(self, mode: str, audit_path: Path) -> None:
        if mode not in ("inplace", "snapshot"):
            fail("invalid CATVM mode")
        self.mode = mode
        self.cells = seed(0) + seed(1)
        self.port_owners = [OWNER_A, OWNER_B]
        self.lease_generations = [0, 0]
        self.stage_name = "IDLE"
        self.active_depth = 0
        self.active_family = 0
        self.pending_operations = 0
        self.generation = 0
        self.nonce = 1
        self.snapshot = tuple(self.cells) if mode == "snapshot" else None
        self.audit_path = audit_path

    def audit(self, event: str) -> None:
        with self.audit_path.open("a", encoding="ascii") as handle:
            handle.write(f"{self.generation}:{event}\n")

    def canonical_state(self) -> tuple[object, ...]:
        return (
            tuple(self.cells),
            tuple(self.port_owners),
            tuple(self.lease_generations),
            self.stage_name,
            self.active_depth,
            self.active_family,
            self.pending_operations,
            self.generation,
            self.nonce,
            self.mode,
        )

    def restorable_state(self) -> tuple[object, ...]:
        state = self.canonical_state()
        return state[:2] + state[3:7] + state[8:]

    def backing_identity(self) -> tuple[int, int, int]:
        return id(self.cells), id(self.port_owners), id(self.lease_generations)

    def validate_idle_custody(self) -> None:
        if self.stage_name != "IDLE" or self.pending_operations != 0:
            fail("carrier is not idle")
        if self.port_owners != [OWNER_A, OWNER_B]:
            fail("port owner corruption")
        if self.lease_generations != [self.generation, self.generation]:
            fail("port lease generation mismatch")

    def validate_consumer(self, source_port: int, declared_type: int, declared_owner: int) -> None:
        if declared_type != PORT_TYPE:
            fail("wrong typed port")
        if declared_owner != self.port_owners[source_port]:
            fail("wrong port owner")
        if self.lease_generations[source_port] != self.generation:
            fail("stale port lease")

    def compose_delta(self, source_offset: int, public: tuple[int, ...], public_on_left: bool) -> list[int]:
        result = [0] * 6
        for target, element in enumerate(ELEMENTS):
            total = 0
            for source, source_element in enumerate(ELEMENTS):
                residual = INDEX[multiply(inverse(source_element), element)]
                if public_on_left:
                    left = public[source]
                    right = self.cells[source_offset + residual]
                else:
                    left = self.cells[source_offset + source]
                    right = public[residual]
                total += left * right
            result[target] = total % FIELD
        return result

    def stage_delta(self, current: Stage) -> list[int]:
        source_port = 0 if current.target == 1 else 1
        source_offset = 6 * source_port
        self.validate_consumer(source_port, PORT_TYPE, self.port_owners[source_port])
        if current.operation == "RIGHT_COMPOSE":
            value = self.compose_delta(source_offset, current.public_operand, False)
        elif current.operation == "LEFT_COMPOSE":
            value = self.compose_delta(source_offset, current.public_operand, True)
        elif current.operation == "INTERSECT":
            value = [self.cells[source_offset + position] * current.public_operand[position] % FIELD for position in range(6)]
        else:
            fail("unknown public S3 relation operation")
        return [current.scalar * item % FIELD for item in value]

    def apply_stage(self, current: Stage, subtracting: bool) -> None:
        delta = self.stage_delta(current)
        target_offset = 6 * current.target
        direction = -1 if subtracting else 1
        for position, value in enumerate(delta):
            self.cells[target_offset + position] = (self.cells[target_offset + position] + direction * value) % FIELD

    def forward(self, depth: int, family: int) -> None:
        self.validate_idle_custody()
        self.stage_name = "FORWARD"
        self.active_depth = depth
        self.active_family = family
        self.pending_operations = 4 * depth
        for index in range(depth):
            for position in range(4):
                self.apply_stage(stage(index, family, position), False)
                self.pending_operations -= 1
        self.stage_name = "FORWARD_COMPLETE"

    def reverse(self, depth: int, family: int, mutation: str | None = None) -> None:
        if self.stage_name != "FORWARD_COMPLETE":
            fail("carrier has no forward state")
        indices = list(reversed(range(depth)))
        positions = (3, 2, 1, 0)
        if mutation == "REORDER":
            indices = list(range(depth))
            positions = (0, 1, 2, 3)
        self.stage_name = "INVERSE"
        self.pending_operations = 4 * depth
        wrong_applied = False
        for index in indices:
            for position in positions:
                current = stage(index, family, position)
                if mutation == "WRONG" and not wrong_applied:
                    current = Stage(current.target, current.operation, (current.scalar + 1) % FIELD, current.public_operand)
                    wrong_applied = True
                self.apply_stage(current, True)
                self.pending_operations -= 1
        self.stage_name = "IDLE"
        self.active_depth = 0
        self.active_family = 0

    def forward_receipt(self, depth: int, family: int) -> int:
        digest = hashlib.sha256()
        digest.update(bytes(self.cells))
        digest.update(depth.to_bytes(2, "little"))
        digest.update(family.to_bytes(1, "little"))
        return int.from_bytes(digest.digest()[:8], "little")

    def boundary(self, family: int) -> int:
        if self.stage_name != "FORWARD_COMPLETE":
            fail("boundary projection outside final forward stage")
        total = 0
        for position in range(6):
            weight = phase(family + 2 * position + position * position)
            total += weight * self.cells[6 + position]
        return total % FIELD

    def hidden_projection(self) -> None:
        if self.stage_name != "FORWARD_COMPLETE":
            fail("hidden projection attack outside forward residency")
        fail("hidden S3 relation projection denied")

    def advance_generation(self) -> None:
        self.generation += 1
        self.lease_generations[:] = [self.generation, self.generation]

    def verify_restoration(self, before_restorable: tuple[object, ...], backing: tuple[int, int, int]) -> None:
        if self.restorable_state() != before_restorable:
            fail("exact S3 carrier restoration failed")
        if self.backing_identity() != backing:
            fail("S3 carrier backing identity changed")

    def success_response(self, command: int, boundary: int, receipt: int, depth: int, reuse: bool, snapshot: bool) -> bytes:
        flags = protocol.BOUNDARY_VALID
        flags |= protocol.SNAPSHOT_RELOADED if snapshot else protocol.RESTORED
        if reuse:
            flags |= protocol.REUSE_FLAG
        resource_signature = (4 * depth << 32) | len(self.cells)
        return protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, command, self.generation, boundary, flags, receipt, resource_signature)

    def run_inplace(self, depth: int, family: int, reuse: bool) -> bytes:
        before = self.restorable_state()
        backing = self.backing_identity()
        self.audit("FORWARD_BEGIN")
        self.forward(depth, family)
        receipt = self.forward_receipt(depth, family)
        boundary = self.boundary(family)
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        self.reverse(depth, family)
        self.verify_restoration(before, backing)
        self.advance_generation()
        self.audit("RESTORATION_VERIFIED")
        response = self.success_response(protocol.REUSE_INPLACE if reuse else protocol.RUN_INPLACE, boundary, receipt, depth, reuse, False)
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def run_snapshot(self, depth: int, family: int) -> bytes:
        if self.snapshot is None:
            fail("snapshot backing unavailable")
        before = self.restorable_state()
        backing = self.backing_identity()
        self.audit("FORWARD_BEGIN")
        self.forward(depth, family)
        receipt = self.forward_receipt(depth, family)
        boundary = self.boundary(family)
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        self.cells[:] = self.snapshot
        self.stage_name = "IDLE"
        self.active_depth = 0
        self.active_family = 0
        self.pending_operations = 0
        self.verify_restoration(before, backing)
        self.advance_generation()
        self.audit("SNAPSHOT_RELOADED")
        response = self.success_response(protocol.RUN_SNAPSHOT, boundary, receipt, depth, False, True)
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def resident_denial(self, command: int, depth: int, family: int) -> bytes:
        before = self.restorable_state()
        backing = self.backing_identity()
        self.audit("FORWARD_BEGIN")
        self.forward(depth, family)
        self.audit("FORWARD_RESIDENT")
        denied = False
        try:
            if command == protocol.PROJECT_HIDDEN_DURING_FORWARD:
                self.hidden_projection()
            elif command == protocol.EARLY_RESPONSE_DURING_FORWARD:
                fail("response release before restoration denied")
            elif command == protocol.WRONG_TYPE_DURING_FORWARD:
                self.validate_consumer(0, PORT_TYPE ^ 1, OWNER_A)
            elif command == protocol.WRONG_OWNER_DURING_FORWARD:
                self.validate_consumer(0, PORT_TYPE, OWNER_B)
            else:
                fail("unknown resident denial")
        except RuntimeError:
            denied = True
        if not denied:
            fail("resident attack unexpectedly accepted")
        events = {
            protocol.PROJECT_HIDDEN_DURING_FORWARD: "HIDDEN_PROJECTION_DENIED_DURING_FORWARD",
            protocol.EARLY_RESPONSE_DURING_FORWARD: "EARLY_RESPONSE_DENIED_DURING_FORWARD",
            protocol.WRONG_TYPE_DURING_FORWARD: "WRONG_TYPE_DENIED_DURING_FORWARD",
            protocol.WRONG_OWNER_DURING_FORWARD: "WRONG_OWNER_DENIED_DURING_FORWARD",
        }
        self.audit(events[command])
        self.reverse(depth, family)
        self.verify_restoration(before, backing)
        self.advance_generation()
        self.audit("RESTORATION_VERIFIED")
        response = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_DENIED, command, self.generation, 0, protocol.RESTORED, 0, len(self.cells))
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def inverse_control(self, command: int, depth: int, family: int) -> int:
        before = self.restorable_state()
        self.audit("FORWARD_BEGIN")
        self.forward(depth, family)
        self.audit("FORWARD_RESIDENT")
        if command == protocol.MISSING_INVERSE:
            self.audit("INVERSE_OMITTED_AFTER_FORWARD")
        else:
            self.audit("MUTATED_INVERSE_EXECUTED")
            self.reverse(depth, family, "WRONG" if command == protocol.WRONG_INVERSE else "REORDER")
        restored = self.restorable_state() == before
        self.audit("MUTATED_RESTORATION_UNEXPECTEDLY_PASSED" if restored else "RESTORATION_FAILED_CONTROL")
        return 24 if restored else 23

    def denied(self, command: int) -> bytes:
        return protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_DENIED, command, self.generation, 0, 0, 0, len(self.cells))

    def clean(self) -> bool:
        return self.stage_name == "IDLE" and self.pending_operations == 0 and self.active_depth == 0 and self.active_family == 0 and self.lease_generations == [self.generation, self.generation]


def read_exact(size: int) -> bytes:
    payload = sys.stdin.buffer.read(size)
    if not payload:
        return b""
    if len(payload) != size:
        fail("truncated request")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inplace", "snapshot"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    args.audit.write_text("", encoding="ascii")
    configure_process_boundary()
    service = Service(args.mode, args.audit)
    while True:
        payload = read_exact(protocol.REQUEST.size)
        if not payload:
            return 0
        magic, command, generation, packed, nonce = protocol.REQUEST.unpack(payload)
        depth, family = packed & 0xFFFF, packed >> 16
        valid = magic == protocol.MAGIC and generation == service.generation and nonce == service.nonce
        public_program = depth in DEPTHS and family in (1, 2)
        if not valid:
            response = service.denied(command)
        elif command == protocol.PING:
            response = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, command, service.generation, 0, 0, 0, len(service.cells))
            service.nonce += 1
        elif command in (protocol.RUN_INPLACE, protocol.REUSE_INPLACE):
            if service.mode != "inplace" or not public_program:
                response = service.denied(command)
            else:
                response = service.run_inplace(depth, family, command == protocol.REUSE_INPLACE)
            service.nonce += 1
        elif command == protocol.RUN_SNAPSHOT:
            if service.mode != "snapshot" or not public_program:
                response = service.denied(command)
            else:
                response = service.run_snapshot(depth, family)
            service.nonce += 1
        elif command in (
            protocol.PROJECT_HIDDEN_DURING_FORWARD,
            protocol.EARLY_RESPONSE_DURING_FORWARD,
            protocol.WRONG_TYPE_DURING_FORWARD,
            protocol.WRONG_OWNER_DURING_FORWARD,
        ):
            response = service.resident_denial(command, depth, family) if service.mode == "inplace" and public_program else service.denied(command)
            service.nonce += 1
        elif command in (protocol.MISSING_INVERSE, protocol.WRONG_INVERSE, protocol.REORDERED_INVERSE):
            return service.inverse_control(command, depth, family) if service.mode == "inplace" and public_program else 22
        elif command == protocol.NULL_CARRIER:
            response = service.denied(command)
            service.nonce += 1
        elif command == protocol.STOP:
            response = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, command, service.generation, 0, protocol.RESTORED if service.mode == "inplace" else protocol.SNAPSHOT_RELOADED, 0, len(service.cells)) if service.clean() else service.denied(command)
            service.audit("STOP_RESPONSE_WRITE_ATTEMPT")
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
            return 0
        else:
            response = service.denied(command)
            service.nonce += 1
        try:
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
