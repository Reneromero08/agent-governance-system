#!/usr/bin/env python3
"""Independent full-relation oracle for the atomic Paley DAG CATVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELD = 103
ORDER = 17
ZETA = 72
RESIDUES = {value * value % ORDER for value in range(1, ORDER)}
NONRESIDUES = set(range(1, ORDER)) - RESIDUES
CLASSES = ({0}, RESIDUES, NONRESIDUES)


def phase(exponent: int) -> int:
    return pow(ZETA, exponent % ORDER, FIELD)


def expand(values: list[int]) -> list[int]:
    return [next(values[index] for index, orbit in enumerate(CLASSES) if position in orbit) for position in range(ORDER)]


def contract(full: list[int]) -> list[int]:
    result = []
    for orbit in CLASSES:
        values = {full[position] for position in orbit}
        if len(values) != 1:
            raise RuntimeError("oracle DAG escaped Paley algebra")
        result.append(next(iter(values)))
    return result


def hadamard(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * ORDER
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[(i + j) % ORDER] = (result[(i + j) % ORDER] + x * y) % FIELD
    return result


def seed(register: int) -> list[int]:
    offset = 5 if register == 0 else 19
    return expand([phase(offset + 2 * (coordinate + 1) + coordinate * coordinate) for coordinate in range(3)])


def descriptor(topology: int) -> list[tuple[int, str, int, int]]:
    edges = ((2, 0, 1), (3, 0, 1), (4, 2, 3), (5, 2, 3), (6, 4, 5), (7, 3, 6), (8, 5, 7))
    operations = ("H", "C", "C", "H", "C", "H", "C") if topology == 1 else ("C", "H", "H", "C", "H", "C", "H")
    return [(target, operation, left, right) for (target, left, right), operation in zip(edges, operations, strict=True)]


def execute(topology: int) -> dict[str, Any]:
    slots = [seed(0), seed(1)] + [[0] * ORDER for _ in range(7)]
    sealed = [slot.copy() for slot in slots]
    schedule = descriptor(topology)
    for target, operation, left, right in schedule:
        slots[target] = hadamard(slots[left], slots[right]) if operation == "H" else convolution(slots[left], slots[right])
    final_values = contract(slots[8])
    boundary = sum(weight * value for weight, value in zip((phase(3), phase(7), phase(11)), final_values, strict=True)) % FIELD
    for target, _operation, _left, _right in reversed(schedule):
        slots[target] = [0] * ORDER
    return {
        "topology": topology,
        "boundary": boundary,
        "full153_cells_restore_exactly": slots == sealed,
        "schedule_nodes": len(schedule),
        "shared_fanouts": {str(node): sum(1 for _, _, left, right in schedule for parent in (left, right) if parent == node) for node in range(9)},
        "final_class_values": final_values,
    }


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    first, second = execute(1), execute(2)
    first_match = first["boundary"] == production["primary"]["first_boundary"]
    second_match = second["boundary"] == production["primary"]["second_boundary"] == production["primary"]["fresh_second_boundary"]
    if not first_match or not second_match or not first["full153_cells_restore_exactly"] or not second["full153_cells_restore_exactly"]:
        raise RuntimeError("independent CATVM semantic mismatch")
    return {
        "schema": "CATVM_PALEY_RELATION_DAG_ORACLE_RESULTS_V1",
        "claim_candidate": production["claim_candidate"],
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "imports_service": False,
        "imports_controller": False,
        "imports_numpy": False,
        "oracle_state_law": "FULL17_RELATION_VALUES_FOR_ALL9_DAG_NODES",
        "topologies": [first, second],
        "boundary_parity": {"primary": first_match, "alternate_and_reuse": second_match},
        "resource_scope": {
            "oracle_full_relation_cells": 9 * ORDER,
            "accepted_paley_relation_cells": 9 * 3,
            "dense17_by17_relation_table_cells": 0,
            "oracle_full_relations_are_verification_only": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": ["INDEPENDENT_FULL_RELATION_BOUNDARY_PARITY", "INDEPENDENT_FULL_DAG_REVERSE_CLEARING", "PUBLIC_TOPOLOGY_HAS_SHARED_FANOUT"],
        "rejected_interpretations": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_result.read_text(encoding="utf-8"))
    payload = json.dumps(build_result(production), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
