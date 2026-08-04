#!/usr/bin/env python3
"""Separate-process atomic CATVM for scheduled C17 Paley relation DAGs."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import catvm_paley_relation_protocol as protocol


FIELD = 103
ZETA = 72


def phase(exponent: int) -> int:
    return pow(ZETA, exponent % 17, FIELD)


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def intersect(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def compose(left: list[int], right: list[int]) -> list[int]:
    a, b, c = left
    d, e, f = right
    return [
        (a * d + 8 * b * e + 8 * c * f) % FIELD,
        (a * e + b * d + 3 * b * e + 4 * b * f + 4 * c * e + 4 * c * f) % FIELD,
        (a * f + c * d + 4 * b * e + 4 * b * f + 4 * c * e + 3 * c * f) % FIELD,
    ]


def seed(register: int) -> list[int]:
    offset = 5 if register == 0 else 19
    return [phase(offset + 2 * (coordinate + 1) + coordinate * coordinate) for coordinate in range(3)]


def descriptor(topology: int, family: int) -> list[tuple[int, str, int, int]]:
    if topology not in (1, 2) or family not in (1, 2):
        raise RuntimeError("invalid public descriptor")
    edges = ((2, 0, 1), (3, 0, 1), (4, 2, 3), (5, 2, 3), (6, 4, 5), (7, 3, 6), (8, 5, 7))
    first = ("H", "C", "C", "H", "C", "H", "C")
    second = ("C", "H", "H", "C", "H", "C", "H")
    operations = first if topology == 1 else second
    return [(target, operation, left, right) for (target, left, right), operation in zip(edges, operations, strict=True)]


def compile_schedule(nodes: list[tuple[int, str, int, int]]) -> tuple[list[tuple[int, str, int, int]], dict[int, int]]:
    pending = list(nodes)
    available = {0, 1}
    schedule: list[tuple[int, str, int, int]] = []
    while pending:
        ready = [node for node in pending if node[2] in available and node[3] in available]
        if not ready:
            raise RuntimeError("cyclic or unresolved public topology")
        ready.sort(key=lambda item: item[0])
        for node in ready:
            schedule.append(node)
            available.add(node[0])
            pending.remove(node)
    uses = {node: 0 for node in available}
    for _, _, left, right in schedule:
        uses[left] += 1
        uses[right] += 1
    return schedule, uses


class Service:
    def __init__(self, audit_path: Path) -> None:
        self.cells = seed(0) + seed(1) + [0] * 21
        self.generation = 0
        self.nonce = 1
        self.audit_path = audit_path

    def audit(self, event: str) -> None:
        with self.audit_path.open("a", encoding="ascii") as handle:
            handle.write(f"{self.generation}:{event}\n")

    def receipt(self, schedule: list[tuple[int, str, int, int]]) -> int:
        payload = repr((schedule, self.generation, tuple(self.cells))).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def slot(self, index: int) -> list[int]:
        return self.cells[3 * index : 3 * index + 3]

    def write_slot(self, index: int, value: list[int]) -> None:
        self.cells[3 * index : 3 * index + 3] = value

    def run(self, topology: int, family: int, reuse: bool) -> bytes:
        before = tuple(self.cells)
        backing = id(self.cells)
        schedule, _uses = compile_schedule(descriptor(topology, family))
        self.audit("FORWARD_BEGIN")
        for target, operation, left, right in schedule:
            value = intersect(self.slot(left), self.slot(right)) if operation == "H" else compose(self.slot(left), self.slot(right))
            self.write_slot(target, add(self.slot(target), value))
        boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), self.slot(8), strict=True)) % FIELD
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        for target, operation, left, right in reversed(schedule):
            value = intersect(self.slot(left), self.slot(right)) if operation == "H" else compose(self.slot(left), self.slot(right))
            self.write_slot(target, subtract(self.slot(target), value))
        if tuple(self.cells) != before or id(self.cells) != backing:
            raise RuntimeError("actual carrier restoration failed")
        self.generation += 1
        self.audit("RESTORATION_VERIFIED")
        flags = protocol.BOUNDARY_VALID | protocol.RESTORED | (protocol.REUSE_FLAG if reuse else 0)
        resource = (len(schedule) << 32) | len(self.cells)
        payload = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, protocol.REUSE if reuse else protocol.RUN, self.generation, boundary, flags, self.receipt(schedule), resource)
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return payload

    def denied(self, command: int) -> bytes:
        return protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_DENIED, command, self.generation, 0, protocol.RESTORED, 0, len(self.cells))


def read_exact(size: int) -> bytes:
    payload = sys.stdin.buffer.read(size)
    if not payload:
        return b""
    if len(payload) != size:
        raise RuntimeError("truncated request")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    args.audit.write_text("", encoding="ascii")
    service = Service(args.audit)
    while True:
        payload = read_exact(protocol.REQUEST.size)
        if not payload:
            return 0
        magic, command, generation, packed, nonce = protocol.REQUEST.unpack(payload)
        topology, family = packed & 0xFFFF, packed >> 16
        if magic != protocol.MAGIC or generation != service.generation or nonce != service.nonce:
            response = service.denied(command)
        elif command in (protocol.RUN, protocol.REUSE):
            response = service.run(topology, family, command == protocol.REUSE)
            service.nonce += 1
        elif command in (protocol.PROJECT_HIDDEN, protocol.NULL_CARRIER, protocol.SNAPSHOT):
            response = service.denied(command)
            service.nonce += 1
        elif command in (protocol.MISSING_INVERSE, protocol.WRONG_INVERSE, protocol.REORDERED_INVERSE):
            service.audit("MUTATED_INVERSE_REJECTED_WITHOUT_RESPONSE")
            return 23
        elif command == protocol.STOP:
            response = service.denied(command) if any(service.cells[6:]) else protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, command, service.generation, 0, protocol.RESTORED, 0, len(service.cells))
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
            return 0
        else:
            response = service.denied(command)
        try:
            sys.stdout.buffer.write(response)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
