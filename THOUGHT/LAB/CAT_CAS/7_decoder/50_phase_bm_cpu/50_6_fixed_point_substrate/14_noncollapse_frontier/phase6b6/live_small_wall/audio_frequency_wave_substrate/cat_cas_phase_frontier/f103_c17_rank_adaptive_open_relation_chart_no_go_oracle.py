#!/usr/bin/env python3
"""Independent scalar oracle for the F103 C17 rank-adaptive relation chart.

This file imports neither the production implementation nor NumPy.  It
reconstructs the public controls, dense relation semantics, final boundary,
exact ranks, full inverse, mutations, and production-result comparisons using
plain Python modular arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


P = 103
N = 17
NODES = 9
DEPTHS = (1, 2, 4, 8, 32, 128)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
PORT_TYPE = "F103_RANK_ADAPTIVE_NONTRANSLATION_C17_TO_C17"
CLAIM = (
    "BOUNDED_EXACT_RANK_ADAPTIVE_NON_TRANSLATION_INVARIANT_F103_C17_"
    "OPEN_RELATION_CHART_CLOSES_IDENTITY_PLUS_RANK2_COMPOSITION_AND_"
    "RECIPROCAL_RANK2_INTERSECTION_ON_ONE_SHARED_UNRESOLVED_PORT_"
    "ACROSS8_NONCOMMUTING_CONSUMERS_WITH_EXACT_RANK_ADAPTIVE_"
    "CANONICALIZATION_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_"
    "THROUGH_DEPTH128_BUT_RANK_SATURATES17_THE_CHART_BECOMES_DENSE_"
    "EQUIVALENT_AND_AN_EXECUTED_REMATERIALIZED_CONTROL_HYBRID_CLASSICAL_"
    "RECURRENCE_RETAINS_FEWER_RESIDENT_COORDINATES"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def family_code(family: str) -> int:
    return {"PRIMARY": 5, "REUSE": 11, "ALTERNATE": 17}[family]


def program(depth: int, family: str) -> dict[str, Any]:
    code = family_code(family)
    owner = (0xC1720000 + 131 * depth + code) & 0xFFFFFFFF
    descriptor = {
        "schema": "CAT_CAS_F103_C17_RANK_ADAPTIVE_RELATION_PROGRAM_V1",
        "depth": depth,
        "family": family,
        "owner": owner,
        "node_count": NODES,
        "port_type": PORT_TYPE,
        "topology": "PUBLIC_ROTATING_RANK2_CONTROL_HUB8",
        "composition": "LEFT_ACTION_BY_IDENTITY_PLUS_RANK2",
        "intersection": "HADAMARD_WITH_RECIPROCAL_RANK2_CONTROL",
        "observation": [
            (7 * depth + 3 * code + 2) % P,
            (11 * depth + 5 * code + 1) % P,
        ],
    }
    return {"descriptor": descriptor, "fingerprint": digest_json(descriptor)}


def rank_mod(matrix: list[list[int]]) -> int:
    if not matrix:
        return 0
    work = [[value % P for value in row] for row in matrix]
    rows = len(work)
    columns = len(work[0])
    pivot = 0
    for column in range(columns):
        candidate = next(
            (row for row in range(pivot, rows) if work[row][column]), None
        )
        if candidate is None:
            continue
        work[pivot], work[candidate] = work[candidate], work[pivot]
        inverse = pow(work[pivot][column], -1, P)
        work[pivot] = [(value * inverse) % P for value in work[pivot]]
        for row in range(rows):
            if row == pivot or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                (work[row][index] - factor * work[pivot][index]) % P
                for index in range(columns)
            ]
        pivot += 1
        if pivot == rows:
            break
    return pivot


def inverse_2(matrix: list[list[int]]) -> list[list[int]]:
    determinant = (
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    ) % P
    if not determinant:
        fail("singular two by two kernel")
    scale = pow(determinant, -1, P)
    return [
        [matrix[1][1] * scale % P, -matrix[0][1] * scale % P],
        [-matrix[1][0] * scale % P, matrix[0][0] * scale % P],
    ]


def control(node: int) -> tuple[list[list[int]], list[list[int]]]:
    q = (7 + 9 * node) % P
    if q in (0, P - 1):
        q = 3
    a = [1 + ((11 * node + 7 * x + 3 * node * x) % 101) for x in range(N)]
    b = [1 + ((13 * node + 5 * y + 4 * node * y) % 101) for y in range(N)]
    s = [1 if ((x + 2 * node) % 5) < 2 else 0 for x in range(N)]
    t = [1 if ((3 * y + node) % 7) < 3 else 0 for y in range(N)]
    left = [[a[x] % P, a[x] * s[x] % P] for x in range(N)]
    right = [[b[y] % P, b[y] * q * t[y] % P] for y in range(N)]
    return left, right


def factor_entry(
    factors: tuple[list[list[int]], list[list[int]]], x: int, y: int
) -> int:
    left, right = factors
    return sum(left[x][term] * right[y][term] for term in range(2)) % P


def reciprocal(
    factors: tuple[list[list[int]], list[list[int]]]
) -> tuple[list[list[int]], list[list[int]]]:
    left, right = factors
    a_inverse = [pow(row[0], -1, P) for row in left]
    b_inverse = [pow(row[0], -1, P) for row in right]
    s = [left[index][1] * a_inverse[index] % P for index in range(N)]
    z = [right[index][1] * b_inverse[index] % P for index in range(N)]
    nonzero = sorted({value for value in z if value})
    if len(nonzero) != 1:
        fail("oracle reciprocal structure failed")
    q = nonzero[0]
    coefficient = -pow(1 + q, -1, P) % P
    inverse_left = [
        [a_inverse[index], a_inverse[index] * s[index] % P]
        for index in range(N)
    ]
    inverse_right = [
        [b_inverse[index], b_inverse[index] * coefficient * z[index] % P]
        for index in range(N)
    ]
    return inverse_left, inverse_right


def rotate_vector(vector: list[list[int]], shift: int) -> list[list[int]]:
    shift %= N
    return [row[:] for row in vector[-shift:] + vector[:-shift]] if shift else [row[:] for row in vector]


def control_view(
    hub: int, peer: int, index: int, family: str, mutation: int = 0
) -> tuple[list[list[int]], list[list[int]]]:
    left, right = control(hub)
    offset = (
        7 * hub
        + 11 * peer
        + 3 * index
        + family_code(family)
        + mutation
    ) % N
    return rotate_vector(left, offset), rotate_vector(right, -offset)


def nonsingular_coupling(
    factors: tuple[list[list[int]], list[list[int]]]
) -> int:
    left, right = factors
    pairing = [
        [sum(right[x][row] * left[x][column] for x in range(N)) % P for column in range(2)]
        for row in range(2)
    ]
    for coupling in range(1, P):
        kernel = [
            [
                ((1 if row == column else 0) + coupling * pairing[row][column]) % P
                for column in range(2)
            ]
            for row in range(2)
        ]
        if (kernel[0][0] * kernel[1][1] - kernel[0][1] * kernel[1][0]) % P:
            return coupling
    fail("oracle found no nonsingular coupling")


def target(node: int) -> list[list[int]]:
    left = [
        [
            (2 + 7 * node + 5 * x + node * x) % P,
            (3 + 11 * node + 9 * x * x) % P,
        ]
        for x in range(N)
    ]
    right = [
        [
            (5 + 13 * node + 7 * y * y) % P,
            (8 + 17 * node + 3 * y + 2 * node * y) % P,
        ]
        for y in range(N)
    ]
    return [
        [sum(left[x][term] * right[y][term] for term in range(2)) % P for y in range(N)]
        for x in range(N)
    ]


def seed() -> list[list[list[int]]]:
    return [target(node) for node in range(NODES)]


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODES


def peers(hub: int) -> list[int]:
    return [(hub + offset) % NODES for offset in range(1, NODES)]


def rotation_shift(node: int, index: int, family: str) -> int:
    return (
        3 * node * node
        + 5 * index
        + family_code(family) * (1 + index.bit_count())
    ) % N


def rotate_relation(relation: list[list[int]], shift: int) -> list[list[int]]:
    return [
        [relation[(x - shift) % N][(y + shift) % N] for y in range(N)]
        for x in range(N)
    ]


def compose(
    relation: list[list[int]],
    factors: tuple[list[list[int]], list[list[int]]],
    *,
    inverse: bool,
) -> list[list[int]]:
    left, right = factors
    coupling = nonsingular_coupling(factors)
    contraction = [
        [sum(right[x][row] * relation[x][y] for x in range(N)) % P for y in range(N)]
        for row in range(2)
    ]
    if inverse:
        pairing = [
            [sum(right[x][row] * left[x][column] for x in range(N)) % P for column in range(2)]
            for row in range(2)
        ]
        kernel = [
            [
                ((1 if row == column else 0) + coupling * pairing[row][column]) % P
                for column in range(2)
            ]
            for row in range(2)
        ]
        kernel_inverse = inverse_2(kernel)
        solved = [
            [sum(kernel_inverse[row][inner] * contraction[inner][y] for inner in range(2)) % P for y in range(N)]
            for row in range(2)
        ]
        sign = -1
        terms = solved
    else:
        sign = 1
        terms = contraction
    return [
        [
            (
                relation[x][y]
                + sign
                * coupling
                * sum(left[x][term] * terms[term][y] for term in range(2))
            )
            % P
            for y in range(N)
        ]
        for x in range(N)
    ]


def intersect(
    relation: list[list[int]],
    factors: tuple[list[list[int]], list[list[int]]],
    *,
    inverse: bool,
) -> list[list[int]]:
    multiplier = reciprocal(factors) if inverse else factors
    return [
        [relation[x][y] * factor_entry(multiplier, x, y) % P for y in range(N)]
        for x in range(N)
    ]


def forward(
    depth: int,
    family: str,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    enabled: bool = True,
    hub_mutation: int = 0,
) -> list[list[list[int]]]:
    state = seed()
    for index in range(depth):
        state = [
            rotate_relation(state[node], rotation_shift(node, index, family))
            for node in range(NODES)
        ]
        hub = hub_index(index, family, hub_mutation)
        for peer in peers(hub):
            if not enabled:
                continue
            factors = control_view(hub, peer, index, family)
            if action_order == "COMPOSE_INTERSECT":
                state[peer] = intersect(compose(state[peer], factors, inverse=False), factors, inverse=False)
            else:
                state[peer] = compose(intersect(state[peer], factors, inverse=False), factors, inverse=False)
    return state


def reverse(
    state: list[list[list[int]]], depth: int, family: str, *, wrong_offset: bool = False, reordered: bool = False
) -> list[list[list[int]]]:
    restored = [[row[:] for row in relation] for relation in state]
    for index in reversed(range(depth)):
        hub = hub_index(index, family)
        for peer in reversed(peers(hub)):
            factors = control_view(hub, peer, index, family, 1 if wrong_offset else 0)
            if reordered:
                restored[peer] = intersect(compose(restored[peer], factors, inverse=True), factors, inverse=True)
            else:
                restored[peer] = compose(intersect(restored[peer], factors, inverse=True), factors, inverse=True)
        restored = [
            rotate_relation(restored[node], -rotation_shift(node, index, family))
            for node in range(NODES)
        ]
    return restored


def boundary(state: list[list[list[int]]], depth: int, family: str) -> list[int]:
    descriptor = program(depth, family)["descriptor"]
    observation_left, observation_right = descriptor["observation"]
    values: list[int] = []
    for coordinate in range(N):
        value = 0
        for node in range(NODES):
            x = (observation_left + coordinate + 3 * node) % N
            y = (observation_right + 2 * coordinate + 5 * node) % N
            value += (1 + node + coordinate * coordinate) * state[node][x][y]
        values.append(value % P)
    return values


def controls() -> dict[str, bool]:
    normal = forward(2, "PRIMARY")
    initial = seed()
    disabled = forward(2, "PRIMARY", enabled=False)
    swapped = forward(2, "PRIMARY", action_order="INTERSECT_COMPOSE")
    mutated = forward(2, "PRIMARY", hub_mutation=1)
    wrong = reverse(normal, 2, "PRIMARY", wrong_offset=True)
    reordered = reverse(normal, 2, "PRIMARY", reordered=True)
    reciprocal_ok = True
    for node in range(NODES):
        factors = control(node)
        inverse = reciprocal(factors)
        for x in range(N):
            for y in range(N):
                reciprocal_ok &= (
                    factor_entry(factors, x, y) * factor_entry(inverse, x, y) % P
                    == 1
                )
    return {
        "missing_inverse_changes_state": normal != initial,
        "wrong_inverse_changes_state": wrong != initial,
        "reordered_inverse_changes_state": reordered != initial,
        "null_port_changes_boundary": boundary(normal, 2, "PRIMARY") != boundary(disabled, 2, "PRIMARY"),
        "composition_intersection_order_changes_boundary": boundary(normal, 2, "PRIMARY") != boundary(swapped, 2, "PRIMARY"),
        "topology_mutation_changes_boundary": boundary(normal, 2, "PRIMARY") != boundary(mutated, 2, "PRIMARY"),
        "rank2_reciprocal_exact": reciprocal_ok,
    }


def run(production_path: Path) -> dict[str, Any]:
    production_bytes = production_path.read_bytes()
    production = json.loads(production_bytes)
    if production.get("claim") != CLAIM:
        fail("production claim differs from independent oracle scope")
    production_cases = {
        (case["family"], case["depth"]): case for case in production["cases"]
    }
    comparisons = 0
    cases: list[dict[str, Any]] = []
    for family in FAMILIES:
        for depth in DEPTHS:
            public_program = program(depth, family)
            final = forward(depth, family)
            final_boundary = boundary(final, depth, family)
            final_ranks = [rank_mod(relation) for relation in final]
            restored = reverse(final, depth, family)
            exact_restoration = restored == seed()
            production_case = production_cases[(family, depth)]
            for index in range(N):
                if final_boundary[index] != production_case["boundary"][index]:
                    fail("independent boundary mismatch")
                comparisons += 1
            for node in range(NODES):
                if final_ranks[node] != production_case["final_ranks"][node]:
                    fail("independent rank mismatch")
                comparisons += 1
            if public_program["fingerprint"] != production_case["program_fingerprint"]:
                fail("independent program fingerprint mismatch")
            comparisons += 1
            if exact_restoration != production_case["exact_restoration"] or not exact_restoration:
                fail("independent restoration mismatch")
            comparisons += 1
            if not production_case["boundary_identical_to_hybrid_classical_recurrence"]:
                fail("production hybrid parity field rejected")
            comparisons += 1
            cases.append(
                {
                    "family": family,
                    "depth": depth,
                    "program_fingerprint": public_program["fingerprint"],
                    "boundary": final_boundary,
                    "final_ranks": final_ranks,
                    "exact_dense_forward_inverse_restoration": exact_restoration,
                }
            )
    control_results = controls()
    if not all(control_results.values()):
        fail("independent mutation controls failed")
    for value in control_results.values():
        comparisons += 1
    maximum_rank = max(max(case["final_ranks"]) for case in cases)
    if maximum_rank != N:
        fail("independent rank-seventeen saturation missing")
    if production["matched_classical_recurrence"]["maximum_resident_value_coordinates"] != NODES * N * N:
        fail("production classical resident ceiling mismatch")
    comparisons += 1
    if production["carrier_law"]["resident_total_bytes"] != 3375:
        fail("production carrier byte count mismatch")
    comparisons += 1
    return {
        "schema": "CAT_CAS_F103_C17_RANK_ADAPTIVE_RELATION_CHART_NO_GO_ORACLE_V1",
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_result_sha256": hashlib.sha256(production_bytes).hexdigest(),
        "oracle_imports_production": False,
        "oracle_imports_numpy": False,
        "independent_case_count": len(cases),
        "independent_comparisons": comparisons,
        "independent_dense_resident_field_coordinates": NODES * N * N,
        "maximum_rank_observed": maximum_rank,
        "controls": control_results,
        "cases": cases,
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_DENSE_SEMANTIC_PARITY",
            "IDENTITY_PLUS_RANK2_COMPOSITION",
            "RECIPROCAL_RANK2_INTERSECTION",
            "RANK_GROWTH_TO17",
            "FINAL_BOUNDARY_PARITY",
            "EXACT_FORWARD_INVERSE_RESTORATION",
            "MUTATION_CONTROL_SEPARATION",
        ],
        "rejected_interpretations": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.production_result)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
