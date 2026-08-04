#!/usr/bin/env python3
"""Atomic CATVM with topology-derived reversible Paley-DAG pebbling."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Node:
    target: int
    operation: str
    left: int
    right: int


@dataclass(frozen=True)
class Step:
    kind: str
    node: int
    slot: int
    epoch: int


@dataclass(frozen=True)
class Plan:
    nodes: tuple[Node, ...]
    steps: tuple[Step, ...]
    capacity: int
    sink: int
    unreachable_state_counts_below_capacity: tuple[int, ...]


def descriptor(topology: int, family: int) -> tuple[Node, ...]:
    if topology not in (1, 2) or family not in (1, 2):
        raise RuntimeError("invalid public descriptor")
    edges = ((2, 0, 1), (3, 0, 1), (4, 2, 3), (5, 2, 3), (6, 4, 5), (7, 3, 6), (8, 5, 7))
    operations = ("H", "C", "C", "H", "C", "H", "C") if topology == 1 else ("C", "H", "H", "C", "H", "C", "H")
    return tuple(Node(target, operation, left, right) for (target, left, right), operation in zip(edges, operations, strict=True))


def validate_public_topology(nodes: tuple[Node, ...]) -> tuple[tuple[int, ...], int]:
    targets = {node.target for node in nodes}
    if len(targets) != len(nodes):
        raise RuntimeError("duplicate public target")
    parents = {parent for node in nodes for parent in (node.left, node.right)}
    sources = tuple(sorted(parents - targets))
    if sources != (0, 1):
        raise RuntimeError("unexpected public sources")
    available = set(sources)
    pending = list(nodes)
    while pending:
        ready = sorted((node for node in pending if node.left in available and node.right in available), key=lambda item: item.target)
        if not ready:
            raise RuntimeError("cyclic or unresolved public topology")
        for node in ready:
            available.add(node.target)
            pending.remove(node)
    sinks = targets - {parent for node in nodes for parent in (node.left, node.right) if parent in targets}
    if len(sinks) != 1:
        raise RuntimeError("public topology must have one sink")
    return tuple(sorted(targets)), next(iter(sinks))


def search_toggles(nodes: tuple[Node, ...], capacity: int, sink: int) -> tuple[tuple[int, ...] | None, int]:
    targets = tuple(sorted(node.target for node in nodes))
    bits = {node: 1 << index for index, node in enumerate(targets)}
    parents = {node.target: (node.left, node.right) for node in nodes}
    start = 0
    goal = bits[sink]
    queue = deque([start])
    previous: dict[int, tuple[int, int] | None] = {start: None}
    while queue:
        state = queue.popleft()
        if state == goal:
            toggles: list[int] = []
            while previous[state] is not None:
                prior, node = previous[state]
                toggles.append(node)
                state = prior
            return tuple(reversed(toggles)), len(previous)
        for node in targets:
            if all(parent not in bits or state & bits[parent] for parent in parents[node]):
                successor = state ^ bits[node]
                if successor.bit_count() <= capacity and successor not in previous:
                    previous[successor] = (state, node)
                    queue.append(successor)
    return None, len(previous)


def compile_plan(nodes: tuple[Node, ...]) -> Plan:
    targets, sink = validate_public_topology(nodes)
    unreachable_counts: list[int] = []
    toggles: tuple[int, ...] | None = None
    capacity = 0
    for candidate in range(1, len(targets) + 1):
        toggles, explored = search_toggles(nodes, candidate, sink)
        if toggles is not None:
            capacity = candidate
            break
        unreachable_counts.append(explored)
    if toggles is None:
        raise RuntimeError("no clean reversible public schedule")

    free = list(range(capacity))
    live: dict[int, tuple[int, int]] = {}
    epoch_counts = {node: 0 for node in targets}
    steps: list[Step] = []
    for node in toggles:
        if node in live:
            slot, epoch = live.pop(node)
            steps.append(Step("REMOVE", node, slot, epoch))
            free.append(slot)
            free.sort()
        else:
            slot = free.pop(0)
            epoch = epoch_counts[node]
            epoch_counts[node] += 1
            live[node] = (slot, epoch)
            steps.append(Step("PLACE", node, slot, epoch))
    if set(live) != {sink}:
        raise RuntimeError("compiled public plan did not clean its garbage")
    return Plan(nodes, tuple(steps), capacity, sink, tuple(unreachable_counts))


class Service:
    def __init__(self, audit_path: Path) -> None:
        initial_plan = compile_plan(descriptor(1, 1))
        self.cells = seed(0) + seed(1) + [0] * (3 * initial_plan.capacity)
        self.owner_nodes = [-1] * initial_plan.capacity
        self.owner_epochs = [-1] * initial_plan.capacity
        self.pending_actions = 0
        self.active_topology = 0
        self.active_family = 0
        self.generation = 0
        self.nonce = 1
        self.audit_path = audit_path

    def audit(self, event: str) -> None:
        with self.audit_path.open("a", encoding="ascii") as handle:
            handle.write(f"{self.generation}:{event}\n")

    def public_node_map(self, plan: Plan) -> dict[int, Node]:
        return {node.target: node for node in plan.nodes}

    def find_slot(self, node: int) -> int:
        try:
            return self.owner_nodes.index(node)
        except ValueError as exc:
            raise RuntimeError("required public-plan parent is not resident") from exc

    def relation(self, node: int) -> list[int]:
        if node in (0, 1):
            return self.cells[3 * node : 3 * node + 3]
        slot = self.find_slot(node)
        start = 6 + 3 * slot
        return self.cells[start : start + 3]

    def write_pool(self, slot: int, value: list[int]) -> None:
        start = 6 + 3 * slot
        self.cells[start : start + 3] = value

    def pool(self, slot: int) -> list[int]:
        start = 6 + 3 * slot
        return self.cells[start : start + 3]

    def node_value(self, node: Node, operation_override: str | None = None) -> list[int]:
        operation = node.operation if operation_override is None else operation_override
        if operation == "H":
            return intersect(self.relation(node.left), self.relation(node.right))
        if operation == "C":
            return compose(self.relation(node.left), self.relation(node.right))
        raise RuntimeError("unknown public relation operation")

    def place(self, step: Step, node: Node) -> None:
        if self.owner_nodes[step.slot] != -1 or any(self.pool(step.slot)):
            raise RuntimeError("public plan attempted to place into a live slot")
        if step.node in self.owner_nodes:
            raise RuntimeError("public plan duplicated a live node")
        self.write_pool(step.slot, self.node_value(node))
        self.owner_nodes[step.slot] = step.node
        self.owner_epochs[step.slot] = step.epoch

    def remove(self, step: Step, node: Node, operation_override: str | None = None) -> None:
        if self.owner_nodes[step.slot] != step.node or self.owner_epochs[step.slot] != step.epoch:
            raise RuntimeError("public plan violated node lease ownership")
        remainder = subtract(self.pool(step.slot), self.node_value(node, operation_override))
        self.write_pool(step.slot, remainder)
        if any(remainder):
            raise RuntimeError("inverse did not clear the leased relation slot")
        self.owner_nodes[step.slot] = -1
        self.owner_epochs[step.slot] = -1

    def forward(self, plan: Plan) -> None:
        node_map = self.public_node_map(plan)
        self.pending_actions = len(plan.steps)
        for step in plan.steps:
            if step.kind == "PLACE":
                self.place(step, node_map[step.node])
            else:
                self.remove(step, node_map[step.node])
            self.pending_actions -= 1

    def reverse(self, plan: Plan, wrong: bool = False, reordered: bool = False) -> bool:
        node_map = self.public_node_map(plan)
        actions = list(reversed(plan.steps))
        if reordered:
            swap_at = next(
                (
                    index
                    for index, (first, second) in enumerate(zip(actions, actions[1:], strict=False))
                    if first.kind == "REMOVE"
                    and second.kind == "REMOVE"
                    and first.node in (node_map[second.node].left, node_map[second.node].right)
                ),
                None,
            )
            if swap_at is None:
                raise RuntimeError("no applicable noncommuting inverse pair")
            actions[swap_at], actions[swap_at + 1] = actions[swap_at + 1], actions[swap_at]
        self.pending_actions = len(actions)
        wrong_applied = False
        try:
            for original in actions:
                node = node_map[original.node]
                if original.kind == "PLACE":
                    override = None
                    if wrong and not wrong_applied and original.node == plan.sink:
                        override = "C" if node.operation == "H" else "H"
                        wrong_applied = True
                    self.remove(original, node, override)
                else:
                    self.place(original, node)
                self.pending_actions -= 1
        except RuntimeError:
            return False
        return not wrong or wrong_applied

    def structural_state(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int, int, int, int]:
        return (
            tuple(self.cells),
            tuple(self.owner_nodes),
            tuple(self.owner_epochs),
            self.pending_actions,
            self.active_topology,
            self.active_family,
            self.generation,
            self.nonce,
        )

    def receipt(self, plan: Plan) -> int:
        toggles = ",".join(str(step.node) for step in plan.steps)
        restored = ",".join(str(value) for value in self.cells)
        payload = f"{plan.capacity}|{toggles}|{self.generation}|{restored}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def run(self, topology: int, family: int, reuse: bool) -> bytes:
        plan = compile_plan(descriptor(topology, family))
        if plan.capacity != len(self.owner_nodes):
            raise RuntimeError("public descriptor exceeds sealed carrier capacity")
        before = self.structural_state()
        backing = (id(self.cells), id(self.owner_nodes), id(self.owner_epochs))
        generation_before = self.generation
        self.active_topology = topology
        self.active_family = family
        self.audit("FORWARD_BEGIN")
        self.forward(plan)
        boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), self.relation(plan.sink), strict=True)) % FIELD
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        if not self.reverse(plan):
            raise RuntimeError("public inverse schedule failed")
        self.active_topology = 0
        self.active_family = 0
        if self.structural_state() != before or (id(self.cells), id(self.owner_nodes), id(self.owner_epochs)) != backing or self.generation != generation_before:
            raise RuntimeError("actual rematerialized carrier restoration failed")
        self.generation += 1
        self.audit("RESTORATION_VERIFIED")
        flags = protocol.BOUNDARY_VALID | protocol.RESTORED | (protocol.REUSE_FLAG if reuse else 0)
        resource = (len(plan.steps) << 48) | (plan.capacity << 32) | len(self.cells)
        response = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, protocol.REUSE if reuse else protocol.RUN, self.generation, boundary, flags, self.receipt(plan), resource)
        self.audit("RESPONSE_WRITE_ATTEMPT")
        return response

    def run_inverse_control(self, topology: int, family: int, command: int) -> int:
        plan = compile_plan(descriptor(topology, family))
        before = self.structural_state()
        self.active_topology = topology
        self.active_family = family
        self.audit("FORWARD_BEGIN")
        self.forward(plan)
        _boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), self.relation(plan.sink), strict=True)) % FIELD
        self.audit("BOUNDARY_RETAINED_INTERNAL")
        restored = False
        if command == protocol.MISSING_INVERSE:
            self.audit("INVERSE_OMITTED_AFTER_FORWARD")
            restored = False
        elif command == protocol.WRONG_INVERSE:
            self.audit("MUTATED_INVERSE_EXECUTED")
            restored = self.reverse(plan, wrong=True)
        elif command == protocol.REORDERED_INVERSE:
            self.audit("MUTATED_INVERSE_EXECUTED")
            restored = self.reverse(plan, reordered=True)
        self.active_topology = 0
        self.active_family = 0
        restored = restored and self.structural_state() == before
        self.audit("MUTATED_RESTORATION_UNEXPECTEDLY_PASSED" if restored else "RESTORATION_FAILED_CONTROL")
        return 24 if restored else 23

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
            return service.run_inverse_control(topology, family, command)
        elif command == protocol.STOP:
            clean = not any(service.cells[6:]) and all(owner == -1 for owner in service.owner_nodes) and service.pending_actions == 0
            response = protocol.RESPONSE.pack(protocol.MAGIC, protocol.STATUS_OK, command, service.generation, 0, protocol.RESTORED, 0, len(service.cells)) if clean else service.denied(command)
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
