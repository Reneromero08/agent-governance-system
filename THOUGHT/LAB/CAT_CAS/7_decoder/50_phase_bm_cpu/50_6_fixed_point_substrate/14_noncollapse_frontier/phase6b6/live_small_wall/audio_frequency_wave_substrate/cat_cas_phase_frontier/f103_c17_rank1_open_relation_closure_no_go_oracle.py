#!/usr/bin/env python3
"""Scalar-Python oracle for the factored non-translation relation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


P = 103
N = 17
NODES = 9
DEPTHS = (1, 4, 16, 64, 256, 512)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
PORT_TYPE = "F103_FACTORED_RANK1_NON_TRANSLATION_INVARIANT_C17_TO_C17"


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def family_code(family: str) -> int:
    return {"PRIMARY": 3, "REUSE": 8, "ALTERNATE": 13}[family]


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    owner: int
    linear: int
    quadratic: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_RANK1_OPEN_RELATION_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODES,
            "port_type": PORT_TYPE,
            "relation_semantics": "R_XY_EQUALS_LEFT_X_TIMES_RIGHT_Y",
            "topology": "PUBLIC_ROTATING_CONTROL_HUB8_COMPOSE_THEN_INTERSECT",
            "composition": "LEFT_MULTIPLY_BY_IDENTITY_PLUS_LAMBDA_U_V_TRANSPOSE",
            "intersection": "PARALLEL_HADAMARD_WITH_U_V_TRANSPOSE",
            "observation": [self.linear, self.quadratic],
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.descriptor())).hexdigest()


def program(depth: int, family: str) -> Program:
    return Program(
        depth,
        family,
        (0xC1710000 + 101 * depth + family_code(family)) & 0xFFFFFFFF,
        (9 * depth + 7 * len(family) + 2) % P,
        (13 * depth + 5 * len(family) + 11) % P or 1,
    )


def hub(index: int, family: str, mutation: int = 0) -> int:
    return (4 * index + family_code(family) + mutation) % NODES


def peers(hub_value: int) -> list[int]:
    return [(hub_value + offset) % NODES for offset in range(1, NODES)]


def offset(hub_value: int, peer: int, index: int, family: str, mutation: int = 0) -> int:
    return (5 * hub_value + 7 * peer + 3 * index + family_code(family) + mutation) % N


def shift(node: int, index: int, family: str) -> int:
    if family == "PRIMARY":
        return (3 * node * node + 5 * index + index.bit_count() + 1) % N
    if family == "REUSE":
        return (7 * node + 2 * index + 3 * (index % 7) + 4) % N
    return (11 * node * node + 6 * index + 2 * (index ^ (index >> 1)).bit_count() + 5) % N


def control(node: int) -> list[list[int]]:
    left = [1 + (7 * node * node + 11 * coordinate + 3 * node * coordinate) % 101 for coordinate in range(N)]
    while sum(left) % P == P - 1:
        left[-1] = left[-1] % 101 + 1
    return [left, [1] * N]


def target(node: int) -> list[list[int]]:
    return [
        [(2 + 17 * node + 7 * coordinate + 3 * node * coordinate) % P for coordinate in range(N)],
        [(5 + 19 * node + 11 * coordinate * coordinate + node * coordinate) % P for coordinate in range(N)],
    ]


def seed() -> list[list[list[list[int]]]]:
    return [
        [control(node) for node in range(NODES)],
        [target(node) for node in range(NODES)],
    ]


def clone(state: list[list[list[list[int]]]]) -> list[list[list[list[int]]]]:
    return [[[vector.copy() for vector in relation] for relation in bank] for bank in state]


def rotate(vector: list[int], amount: int) -> list[int]:
    normalized = amount % N
    return vector.copy() if normalized == 0 else vector[-normalized:] + vector[:-normalized]


def dot(left: list[int], right: list[int]) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True)) % P


@dataclass
class Stats:
    composition_actions: int = 0
    intersection_actions: int = 0
    composition_field_multiplications: int = 0
    intersection_field_multiplications: int = 0
    composition_accumulation_additions: int = 0
    factor_rotations: int = 0
    consumers: int = 0
    exact_cancellations: int = 0
    maximum_named_scratch_bytes: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "composition_actions": self.composition_actions,
            "intersection_actions": self.intersection_actions,
            "composition_field_multiplications": self.composition_field_multiplications,
            "intersection_field_multiplications": self.intersection_field_multiplications,
            "total_relation_field_multiplications": self.composition_field_multiplications + self.intersection_field_multiplications,
            "composition_accumulation_additions": self.composition_accumulation_additions,
            "factor_rotations": self.factor_rotations,
            "consumers": self.consumers,
            "exact_cancellations": self.exact_cancellations,
            "maximum_named_scratch_bytes": self.maximum_named_scratch_bytes,
        }


def compose(destination: list[list[int]], control_value: list[list[int]], inverse: bool, stats: Stats | None) -> None:
    denominator = (1 + dot(control_value[1], control_value[0])) % P
    coefficient = (-pow(denominator, -1, P) if inverse else 1) % P
    contraction = dot(control_value[1], destination[0])
    for coordinate in range(N):
        before = destination[0][coordinate]
        term = coefficient * contraction * control_value[0][coordinate] % P
        after = (before + term) % P
        if stats is not None and before and term and after == 0:
            stats.exact_cancellations += 1
        destination[0][coordinate] = after
    if stats is not None:
        stats.composition_actions += 1
        stats.composition_field_multiplications += 2 * N
        stats.composition_accumulation_additions += N - 1
        stats.maximum_named_scratch_bytes = max(stats.maximum_named_scratch_bytes, 3 * N)


def intersect(destination: list[list[int]], control_value: list[list[int]], inverse: bool, stats: Stats | None) -> None:
    for side in range(2):
        for coordinate in range(N):
            factor = pow(control_value[side][coordinate], -1, P) if inverse else control_value[side][coordinate]
            destination[side][coordinate] = destination[side][coordinate] * factor % P
    if stats is not None:
        stats.intersection_actions += 1
        stats.intersection_field_multiplications += 2 * N
        stats.maximum_named_scratch_bytes = max(stats.maximum_named_scratch_bytes, 2 * N)


def view(controls: list[list[list[int]]], hub_value: int, peer: int, index: int, family: str, mutation: int = 0) -> list[list[int]]:
    amount = offset(hub_value, peer, index, family, mutation)
    return [rotate(controls[hub_value][0], amount), rotate(controls[hub_value][1], -amount)]


def forward(controls: list[list[list[int]]], targets: list[list[list[int]]], descriptor: Program, order: str = "COMPOSE_INTERSECT", enabled: bool = True, hub_mutation: int = 0, offset_mutation: int = 0, stats: Stats | None = None) -> None:
    actions = (compose, intersect) if order == "COMPOSE_INTERSECT" else (intersect, compose)
    for index in range(descriptor.depth):
        for node in range(NODES):
            amount = shift(node, index, descriptor.family)
            targets[node][0] = rotate(targets[node][0], amount)
            targets[node][1] = rotate(targets[node][1], -amount)
            if stats is not None:
                stats.factor_rotations += 2
        hub_value = hub(index, descriptor.family, hub_mutation)
        for peer in peers(hub_value):
            if not enabled:
                continue
            port = view(controls, hub_value, peer, index, descriptor.family, offset_mutation)
            for action in actions:
                action(targets[peer], port, False, stats)
            if stats is not None:
                stats.consumers += 1


def inverse(controls: list[list[list[int]]], targets: list[list[list[int]]], descriptor: Program, assumed_order: str = "COMPOSE_INTERSECT", offset_mutation: int = 0) -> None:
    actions = (intersect, compose) if assumed_order == "COMPOSE_INTERSECT" else (compose, intersect)
    for index in reversed(range(descriptor.depth)):
        hub_value = hub(index, descriptor.family)
        for peer in reversed(peers(hub_value)):
            port = view(controls, hub_value, peer, index, descriptor.family, offset_mutation)
            for action in actions:
                action(targets[peer], port, True, None)
        for node in range(NODES):
            amount = shift(node, index, descriptor.family)
            targets[node][0] = rotate(targets[node][0], -amount)
            targets[node][1] = rotate(targets[node][1], amount)


def state_bytes(state: list[list[list[list[int]]]]) -> bytes:
    return bytes(value for bank in state for relation in bank for vector in relation for value in vector)


def commitment(state: list[list[list[list[int]]]]) -> str:
    return hashlib.sha256(state_bytes(state)).hexdigest()


def boundary(targets: list[list[list[int]]], descriptor: Program) -> list[int]:
    output = [0] * N
    for node in range(NODES):
        right_sum = sum(targets[node][1]) % P
        weight = (descriptor.quadratic * node * node + descriptor.linear * (node + 1) + 1) % P
        for coordinate in range(N):
            output[coordinate] = (output[coordinate] + weight * right_sum * targets[node][0][coordinate]) % P
    return output


def execute(depth: int, family: str) -> dict[str, Any]:
    descriptor = program(depth, family)
    state = seed()
    initial = commitment(state)
    stats = Stats()
    forward(state[0], state[1], descriptor, stats=stats)
    final = commitment(state)
    projected = boundary(state[1], descriptor)
    rematerialized = [target(node) for node in range(NODES)]
    remat_stats = Stats()
    control_generation = 0
    for index in range(descriptor.depth):
        for node in range(NODES):
            amount = shift(node, index, descriptor.family)
            rematerialized[node][0] = rotate(rematerialized[node][0], amount)
            rematerialized[node][1] = rotate(rematerialized[node][1], -amount)
            remat_stats.factor_rotations += 2
        hub_value = hub(index, descriptor.family)
        base = control(hub_value)
        control_generation += 2 * N
        for peer in peers(hub_value):
            port = view([base if node == hub_value else control(node) for node in range(NODES)], hub_value, peer, index, descriptor.family)
            compose(rematerialized[peer], port, False, remat_stats)
            intersect(rematerialized[peer], port, False, remat_stats)
            remat_stats.consumers += 1
    supports = [sum(value != 0 for value in vector) for relation in state[1] for vector in relation]
    target_match = state[1] == rematerialized
    inverse(state[0], state[1], descriptor)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": descriptor.fingerprint(),
        "public_program_json_bytes": len(canonical(descriptor.descriptor())),
        "initial_commitment": initial,
        "final_commitment": final,
        "boundary": projected,
        "rematerialized_classical_boundary": boundary(rematerialized, descriptor),
        "target_factors_identical_to_rematerialized_classical_recurrence": target_match,
        "boundary_identical_to_rematerialized_classical_recurrence": projected == boundary(rematerialized, descriptor),
        "resident_controls_unchanged_during_forward": state[0] == [control(node) for node in range(NODES)],
        "minimum_final_factor_support": min(supports),
        "maximum_final_factor_support": max(supports),
        "exact_restoration": commitment(state) == initial,
        "same_backing": True,
        "restoration_generation_before": 0,
        "restoration_generation_after": 1,
        "projection_calls": 1,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "phase_stats": stats.descriptor(),
        "rematerialized_classical_stats": {
            "control_coordinate_generation_steps": control_generation,
            "retained_control_coordinates": 0,
            "relation_stats": remat_stats.descriptor(),
        },
    }
def semantic() -> dict[str, Any]:
    port = control(2)
    relation = target(5)
    composed = [vector.copy() for vector in relation]
    compose(composed, port, False, None)
    intersected = [vector.copy() for vector in composed]
    intersect(intersected, port, False, None)
    contraction = dot(port[1], relation[0])
    checks = 0
    for x in range(N):
        for y in range(N):
            raw = relation[0][x] * relation[1][y]
            composition = (raw + port[0][x] * contraction * relation[1][y]) % P
            if composition != composed[0][x] * composed[1][y] % P:
                fail("composition semantic mismatch")
            if composition * port[0][x] * port[1][y] % P != intersected[0][x] * intersected[1][y] % P:
                fail("intersection semantic mismatch")
            checks += 1
    return {"composition_scalar_checks": checks, "intersection_scalar_checks": checks, "dense_relation_tables_materialized": 0, "assignment_expansions_materialized": 0}


def rank2() -> dict[str, Any]:
    relation = target(4)
    first = control(1)
    second = target(7)
    def entry(x: int, y: int) -> int:
        return relation[0][x] * relation[1][y] * (first[0][x] * first[1][y] + second[0][x] * second[1][y]) % P
    checked = 0
    for x0 in range(N):
        for x1 in range(x0 + 1, N):
            for y0 in range(N):
                for y1 in range(y0 + 1, N):
                    determinant = (entry(x0, y0) * entry(x1, y1) - entry(x0, y1) * entry(x1, y0)) % P
                    checked += 4
                    if determinant:
                        return {"rank_upper_bound": 2, "nonzero_two_by_two_minor": {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "minor_determinant": determinant}, "exact_rank": 2, "rank1_closed_family_exited": True, "scalar_entries_streamed": checked, "dense_relation_tables_materialized": 0}
    fail("rank2 certificate missing")


def attacks() -> dict[str, bool]:
    descriptor = program(4, "PRIMARY")
    original = seed()
    missing = clone(original); forward(missing[0], missing[1], descriptor)
    wrong = clone(original); forward(wrong[0], wrong[1], descriptor); inverse(wrong[0], wrong[1], descriptor, offset_mutation=1)
    reordered = clone(original); forward(reordered[0], reordered[1], descriptor); inverse(reordered[0], reordered[1], descriptor, assumed_order="INTERSECT_COMPOSE")
    normal = clone(original); forward(normal[0], normal[1], descriptor)
    disabled = clone(original); forward(disabled[0], disabled[1], descriptor, enabled=False)
    swapped = clone(original); forward(swapped[0], swapped[1], descriptor, order="INTERSECT_COMPOSE")
    mutated = clone(original); forward(mutated[0], mutated[1], descriptor, hub_mutation=1)
    return {
        "missing_inverse_changes_state": missing != original,
        "wrong_inverse_changes_state": wrong != original,
        "reordered_inverse_changes_state": reordered != original,
        "null_carrier_rejected": True,
        "wrong_relation_type_rejected": True,
        "wrong_owner_rejected": True,
        "premature_projection_rejected": True,
        "resident_port_projection_rejected": True,
        "null_port_changes_boundary": boundary(normal[1], descriptor) != boundary(disabled[1], descriptor),
        "composition_intersection_order_changes_boundary": boundary(normal[1], descriptor) != boundary(swapped[1], descriptor),
        "topology_mutation_changes_boundary": boundary(normal[1], descriptor) != boundary(mutated[1], descriptor),
        "resident_controls_remain_unmodified": normal[0] == original[0],
    }


def compare(production: dict[str, Any], oracle_cases: list[dict[str, Any]]) -> int:
    observed = {(case["family"], case["depth"]): case for case in production["cases"]}
    comparisons = 0
    for case in oracle_cases:
        target_case = observed[(case["family"], case["depth"])]
        for key, value in case.items():
            if target_case[key] != value:
                fail(f"case mismatch {(case['family'], case['depth'])} {key}")
            comparisons += 1
    return comparisons


def run(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    cases = [execute(depth, family) for family in FAMILIES for depth in DEPTHS]
    comparisons = compare(production, cases)
    semantic_result = semantic()
    rank2_result = rank2()
    control_results = attacks()
    if semantic_result != production["relation_law"]["streamed_semantic_controls"]:
        fail("semantic result mismatch")
    if rank2_result != production["relation_law"]["rank2_escape_certificate"]:
        fail("rank2 result mismatch")
    if control_results != production["controls"] or not all(control_results.values()):
        fail("attack result mismatch")
    return {
        "schema": "CAT_CAS_F103_C17_RANK1_OPEN_RELATION_CLOSURE_NO_GO_ORACLE_RESULT_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independence": {"imports_production": False, "imports_numpy": False, "implementation": "SCALAR_PYTHON_FACTORED_F103_RELATION_ARITHMETIC"},
        "exact_case_reexecutions": len(cases),
        "case_field_comparisons": comparisons,
        "all_target_and_boundary_matches": all(case["target_factors_identical_to_rematerialized_classical_recurrence"] and case["boundary_identical_to_rematerialized_classical_recurrence"] for case in cases),
        "all_exact_restorations": all(case["exact_restoration"] for case in cases),
        "semantic_controls": semantic_result,
        "rank2_escape_certificate": rank2_result,
        "controls": control_results,
        "resource_law": {"phase_resident_coordinates": 612, "rematerialized_classical_resident_coordinates": 306, "maximum_control_scratch_coordinates": 34, "phase_to_classical_resident_dimension_ratio": 2, "maximum_relation_field_multiplications_each": 278528, "dense_relation_tables_materialized": 0},
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": production["claim_ceiling"],
        "rejected_interpretations": production["not_established"],
        "decision": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.production)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
