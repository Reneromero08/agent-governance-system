#!/usr/bin/env python3
"""Independent algebra and black-box oracle for the S3 relation CATVM."""

from __future__ import annotations

import argparse
import errno
import hashlib
import itertools
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


FIELD = 103
ZETA6 = 57
ELEMENTS = tuple(itertools.permutations(range(3)))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
IDENTITY = INDEX[(0, 1, 2)]
MAGIC = 0x53334341
REQUEST = struct.Struct("<IIIIQ")
RESPONSE = struct.Struct("<IIIIiQQQ")
RUN_INPLACE = 1
REUSE_INPLACE = 2
PROJECT_HIDDEN = 3
REORDERED_INVERSE = 6
RUN_SNAPSHOT = 8
STOP = 9
PING = 13
STATUS_OK = 0
STATUS_DENIED = 1
BOUNDARY_VALID = 1
RESTORED = 2
SNAPSHOT_RELOADED = 8


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


def seed(register: int) -> list[int]:
    offset = 1 if register == 0 else 4
    return [phase(offset + position + 1 + position * position) for position in range(6)]


def public_vector(index: int, family: int, kind: int) -> list[int]:
    return [phase((kind + 1) * index + (2 * kind + family) * position + position * position + family) for position in range(6)]


def scalar(index: int, family: int, offset: int) -> int:
    return phase((offset + 1) * index + offset * family + 1)


def descriptor(index: int, family: int, position: int) -> tuple[int, str, int, list[int]]:
    base = position if family == 1 else 3 - position
    targets = (1, 0, 1, 0)
    operations = ("RIGHT", "HADAMARD", "LEFT", "HADAMARD")
    kinds = (1, 2, 3, 4)
    offsets = (1, 2, 4, 5)
    return targets[base], operations[base], scalar(index, family, offsets[base]), public_vector(index, family, kinds[base])


def compose(left: list[int], right: list[int]) -> list[int]:
    result = [0] * 6
    for target, element in enumerate(ELEMENTS):
        for source, source_element in enumerate(ELEMENTS):
            residual = INDEX[multiply(inverse(source_element), element)]
            result[target] = (result[target] + left[source] * right[residual]) % FIELD
    return result


def apply_compact(a: list[int], b: list[int], item: tuple[int, str, int, list[int]], subtracting: bool) -> tuple[list[int], list[int]]:
    target, operation, factor, public = item
    source = a if target == 1 else b
    if operation == "RIGHT":
        delta = compose(source, public)
    elif operation == "LEFT":
        delta = compose(public, source)
    else:
        delta = [left * right % FIELD for left, right in zip(source, public, strict=True)]
    delta = [factor * value % FIELD for value in delta]
    direction = -1 if subtracting else 1
    updated = [(value + direction * change) % FIELD for value, change in zip(a if target == 0 else b, delta, strict=True)]
    return (updated, b) if target == 0 else (a, updated)


def compact_forward(depth: int, family: int) -> tuple[list[int], list[int]]:
    a, b = seed(0), seed(1)
    for index in range(depth):
        for position in range(4):
            a, b = apply_compact(a, b, descriptor(index, family, position), False)
    return a, b


def compact_reverse(a: list[int], b: list[int], depth: int, family: int, reordered: bool = False) -> tuple[list[int], list[int]]:
    indices = range(depth) if reordered else reversed(range(depth))
    positions = range(4) if reordered else (3, 2, 1, 0)
    for index in indices:
        for position in positions:
            a, b = apply_compact(a, b, descriptor(index, family, position), True)
    return a, b


def relation(compact: list[int]) -> list[int]:
    return [compact[INDEX[multiply(inverse(left), right)]] for left in ELEMENTS for right in ELEMENTS]


def matrix_multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * 36
    for row in range(6):
        for column in range(6):
            result[6 * row + column] = sum(left[6 * row + inner] * right[6 * inner + column] for inner in range(6)) % FIELD
    return result


def apply_full(a: list[int], b: list[int], item: tuple[int, str, int, list[int]], subtracting: bool) -> tuple[list[int], list[int]]:
    target, operation, factor, public_compact = item
    source = a if target == 1 else b
    public = relation(public_compact)
    if operation == "RIGHT":
        delta = matrix_multiply(source, public)
    elif operation == "LEFT":
        delta = matrix_multiply(public, source)
    else:
        delta = [left * right % FIELD for left, right in zip(source, public, strict=True)]
    direction = -1 if subtracting else 1
    delta = [factor * value % FIELD for value in delta]
    updated = [(value + direction * change) % FIELD for value, change in zip(a if target == 0 else b, delta, strict=True)]
    return (updated, b) if target == 0 else (a, updated)


def full_forward(depth: int, family: int) -> tuple[list[int], list[int]]:
    a, b = relation(seed(0)), relation(seed(1))
    for index in range(depth):
        for position in range(4):
            a, b = apply_full(a, b, descriptor(index, family, position), False)
    return a, b


def full_reverse(a: list[int], b: list[int], depth: int, family: int) -> tuple[list[int], list[int]]:
    for index in reversed(range(depth)):
        for position in (3, 2, 1, 0):
            a, b = apply_full(a, b, descriptor(index, family, position), True)
    return a, b


def compact_from_relation(value: list[int]) -> list[int]:
    return value[6 * IDENTITY : 6 * IDENTITY + 6]


def boundary(b: list[int], family: int) -> int:
    return sum(phase(family + 2 * position + position * position) * value for position, value in enumerate(b)) % FIELD


def receipt(a: list[int], b: list[int], depth: int, family: int) -> int:
    digest = hashlib.sha256(bytes(a + b) + depth.to_bytes(2, "little") + family.to_bytes(1, "little")).digest()
    return int.from_bytes(digest[:8], "little")


def request(command: int, generation: int, depth: int, family: int, nonce: int) -> bytes:
    return REQUEST.pack(MAGIC, command, generation, depth | (family << 16), nonce)


def response(payload: bytes) -> dict[str, int]:
    if len(payload) != RESPONSE.size:
        fail("black-box response size mismatch")
    magic, status, command, generation, result, flags, custody, resources = RESPONSE.unpack(payload)
    if magic != MAGIC:
        fail("black-box response magic mismatch")
    return {"status": status, "command": command, "generation": generation, "boundary": result, "flags": flags, "receipt": custody, "resource": resources}


class BlackBox:
    def __init__(self, service: Path, evidence: Path, label: str, mode: str) -> None:
        self.audit = evidence / f"oracle-{label}.audit.log"
        self.process = subprocess.Popen(
            [sys.executable, str(service), "--mode", mode, "--audit", str(self.audit)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.generation = 0
        self.nonce = 1
        ready = self.call(PING, 0, 0)
        if ready["status"] != STATUS_OK:
            fail("independent black-box readiness failed")

    def call(self, command: int, depth: int, family: int) -> dict[str, int]:
        if self.process.stdin is None or self.process.stdout is None:
            fail("independent black-box pipe unavailable")
        self.process.stdin.write(request(command, self.generation, depth, family, self.nonce))
        self.process.stdin.flush()
        parsed = response(self.process.stdout.read(RESPONSE.size))
        if parsed["flags"] & (RESTORED | SNAPSHOT_RELOADED):
            self.generation = parsed["generation"]
        self.nonce += 1
        return parsed

    def memory_denied(self) -> bool:
        try:
            descriptor = os.open(f"/proc/{self.process.pid}/mem", os.O_RDONLY)
        except OSError as exc:
            return exc.errno in (errno.EACCES, errno.EPERM)
        os.close(descriptor)
        return False

    def stop(self) -> None:
        stopped = self.call(STOP, 0, 0)
        if stopped["status"] != STATUS_OK:
            fail("independent black-box stop failed")
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=5)
        stderr = self.process.stderr.read() if self.process.stderr is not None else b""
        if stderr:
            fail("independent black-box emitted stderr")


def semantic_case(depth: int, family: int) -> dict[str, Any]:
    compact_a, compact_b = compact_forward(depth, family)
    full_a, full_b = full_forward(depth, family)
    compact_restored = compact_reverse(list(compact_a), list(compact_b), depth, family)
    full_restored = full_reverse(list(full_a), list(full_b), depth, family)
    return {
        "depth": depth,
        "family": family,
        "boundary": boundary(compact_b, family),
        "receipt": receipt(compact_a, compact_b, depth, family),
        "compact_matches_full72_cells": compact_a == compact_from_relation(full_a) and compact_b == compact_from_relation(full_b),
        "compact_restores_exactly": compact_restored == (seed(0), seed(1)),
        "full72_cells_restore_exactly": full_restored == (relation(seed(0)), relation(seed(1))),
    }


def black_box_checks(service: Path, evidence: Path, semantic: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    accepted = BlackBox(service, evidence, "accepted", "inplace")
    memory_denied = accepted.memory_denied()
    first = accepted.call(RUN_INPLACE, 1, 1)
    second = accepted.call(REUSE_INPLACE, 256, 2)
    accepted.stop()
    names = [line.split(":", 1)[1] for line in accepted.audit.read_text(encoding="ascii").splitlines()]

    projection = BlackBox(service, evidence, "projection", "inplace")
    projection_memory_denied = projection.memory_denied()
    projection_response = projection.call(PROJECT_HIDDEN, 64, 1)
    projection.stop()
    projection_names = [line.split(":", 1)[1] for line in projection.audit.read_text(encoding="ascii").splitlines()]

    snapshot = BlackBox(service, evidence, "snapshot", "snapshot")
    snapshot_memory_denied = snapshot.memory_denied()
    snapshot_response = snapshot.call(RUN_SNAPSHOT, 1, 1)
    mode_mismatch = snapshot.call(RUN_INPLACE, 1, 1)
    snapshot.stop()

    reordered = BlackBox(service, evidence, "reordered", "inplace")
    reordered_memory_denied = reordered.memory_denied()
    if reordered.process.stdin is None or reordered.process.stdout is None:
        fail("independent reordered pipe unavailable")
    reordered.process.stdin.write(request(REORDERED_INVERSE, 0, 64, 1, reordered.nonce))
    reordered.process.stdin.flush()
    reordered_payload = reordered.process.stdout.read(RESPONSE.size)
    reordered.process.wait(timeout=10)
    reordered_stderr = reordered.process.stderr.read() if reordered.process.stderr is not None else b""
    reordered_names = [line.split(":", 1)[1] for line in reordered.audit.read_text(encoding="ascii").splitlines()]

    return {
        "accepted_process_memory_denied": memory_denied,
        "accepted_first_matches_oracle": first["boundary"] == semantic[(1, 1)]["boundary"] and first["receipt"] == semantic[(1, 1)]["receipt"],
        "accepted_reuse_matches_oracle": second["boundary"] == semantic[(256, 2)]["boundary"] and second["receipt"] == semantic[(256, 2)]["receipt"],
        "accepted_atomic_event_order": names == [
            "FORWARD_BEGIN", "BOUNDARY_RETAINED_INTERNAL", "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT",
            "FORWARD_BEGIN", "BOUNDARY_RETAINED_INTERNAL", "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT",
            "STOP_RESPONSE_WRITE_ATTEMPT",
        ],
        "projection_process_memory_denied": projection_memory_denied,
        "projection_denied_after_forward_and_restoration": projection_response["status"] == STATUS_DENIED and projection_response["boundary"] == 0 and projection_response["flags"] & RESTORED and projection_names == [
            "FORWARD_BEGIN", "FORWARD_RESIDENT", "HIDDEN_PROJECTION_DENIED_DURING_FORWARD", "RESTORATION_VERIFIED", "RESPONSE_WRITE_ATTEMPT", "STOP_RESPONSE_WRITE_ATTEMPT",
        ],
        "snapshot_process_memory_denied": snapshot_memory_denied,
        "snapshot_matches_oracle_but_is_reload": snapshot_response["boundary"] == semantic[(1, 1)]["boundary"] and snapshot_response["receipt"] == semantic[(1, 1)]["receipt"] and snapshot_response["flags"] & SNAPSHOT_RELOADED and not snapshot_response["flags"] & RESTORED,
        "inplace_command_on_snapshot_denied": mode_mismatch["status"] == STATUS_DENIED and mode_mismatch["boundary"] == 0,
        "reordered_process_memory_denied": reordered_memory_denied,
        "reordered_inverse_fails_after_forward_without_response": len(reordered_payload) == 0 and reordered.process.returncode == 23 and not reordered_stderr and reordered_names == ["FORWARD_BEGIN", "FORWARD_RESIDENT", "MUTATED_INVERSE_EXECUTED", "RESTORATION_FAILED_CONTROL"],
    }


def build_result(service: Path, production_result: Path, evidence: Path) -> dict[str, Any]:
    production = json.loads(production_result.read_text(encoding="utf-8"))
    cases = [semantic_case(1, 1), semantic_case(256, 2)]
    semantic = {(item["depth"], item["family"]): item for item in cases}
    black_box = black_box_checks(service, evidence, semantic)
    first, second = cases
    production_match = (
        production["inplace"]["first_boundary"] == first["boundary"]
        and production["inplace"]["first_receipt"] == first["receipt"]
        and production["inplace"]["second_boundary"] == second["boundary"]
        and production["inplace"]["second_receipt"] == second["receipt"]
        and production["snapshot_sham"]["boundary"] == first["boundary"]
        and production["snapshot_sham"]["receipt"] == first["receipt"]
    )
    basis = [[1 if position == index else 0 for position in range(6)] for index in range(6)]
    basis_checks = [
        compose(basis[left], basis[right]) == basis[INDEX[multiply(ELEMENTS[left], ELEMENTS[right])]]
        and relation(compose(basis[left], basis[right])) == matrix_multiply(relation(basis[left]), relation(basis[right]))
        for left in range(6)
        for right in range(6)
    ]
    noncommuting_pair = next((left, right) for left in range(6) for right in range(6) if compose(basis[left], basis[right]) != compose(basis[right], basis[left]))
    forward64 = compact_forward(64, 1)
    reordered64 = compact_reverse(list(forward64[0]), list(forward64[1]), 64, 1, True)
    if not all(item["compact_matches_full72_cells"] and item["compact_restores_exactly"] and item["full72_cells_restore_exactly"] for item in cases):
        fail("independent semantic restoration failure")
    if not all(basis_checks) or not all(black_box.values()) or not production_match or reordered64 == (seed(0), seed(1)):
        fail("independent CATVM comparison failure")
    return {
        "schema": "CATVM_S3_NONCOMMUTATIVE_RELATION_ORACLE_RESULTS_V1",
        "claim": production["claim_candidate"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_service": False,
        "imports_controller": False,
        "imports_protocol": False,
        "imports_numpy": False,
        "semantic_cases": cases,
        "production_boundary_and_receipt_fields_match": production_match,
        "black_box_service_reexecution": black_box,
        "algebra": {
            "all36_group_and_full_matrix_basis_products_checked": sum(basis_checks),
            "noncommuting_basis_pair": list(noncommuting_pair),
            "noncommuting_products_differ": True,
            "reordered_depth64_compact_inverse_fails": reordered64 != (seed(0), seed(1)),
        },
        "resource_scope": {
            "accepted_carrier_field_cells": 12,
            "accepted_temporary_verification_baseline_field_cells": 12,
            "accepted_streamed_public_operand_delta_scalar_field_slots": 13,
            "accepted_conservative_field_value_slots_peak": 37,
            "matched_streamed_group_classical_field_value_slots_peak": 25,
            "oracle_full_relation_field_cells": 72,
            "retained_inverse_history_records": 0,
            "snapshot_sham_field_cells": 12,
            "physical_process_peak_unmeasured": True,
        },
        "matched_baseline": "IDENTICAL_STREAMED_SIX_COORDINATE_S3_GROUP_RECURRENCE",
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "ATOMIC_RESPONSE_RELEASE_AFTER_EXACT_RESTORATION_ON_THE_ACCEPTED_PATH",
            "NON_DUMPABLE_CHILD_PROCESS_MEMORY_OPEN_IS_DENIED_TO_BOTH_CONTROLLERS",
            "HIDDEN_PROJECTION_IS_ATTEMPTED_AND_DENIED_DURING_FORWARD_RESIDENCY_THEN_RESTORED",
            "REORDERED_NONCOMMUTING_INVERSE_IS_EXECUTED_AFTER_FORWARD_RESIDENCY_AND_WITHHOLDS_RESPONSE",
            "INDEPENDENT_FULL72_CELL_RELATION_BOUNDARIES_AND_EXACT_REVERSE_CLEARING",
            "SAME_SERVICE_UNRELATED_REUSE_MATCHES_FRESH_AND_INDEPENDENT_RECURRENCES",
            "SNAPSHOT_RELOAD_IS_SEPARATELY_CLASSIFIED",
        ],
        "rejected_interpretations": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", type=Path)
    parser.add_argument("production_result", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_result(args.service.resolve(), args.production_result.resolve(), args.evidence.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
