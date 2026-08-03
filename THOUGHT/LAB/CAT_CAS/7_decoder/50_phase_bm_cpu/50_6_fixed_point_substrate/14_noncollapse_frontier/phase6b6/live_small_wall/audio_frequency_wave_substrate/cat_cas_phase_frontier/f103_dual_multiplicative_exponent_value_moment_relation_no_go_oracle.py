#!/usr/bin/env python3
"""Production-independent scalar oracle for the dual exponent/moment no-go.

This file imports neither the production module nor NumPy.  It reconstructs
the public descriptors, exact dense F103 recurrence, exact inverse, CRT ranks,
canonical factor-chart commitments, boundary commitments, rank law, resident
byte formulas, and semantic mutations from first principles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELD = 103
GENERATOR = 5
EXPONENT_MODULUS = 102
MODULI = (2, 3, 17)
INTERFACES = (5, 7, 11, 17)
DEPTHS = (1, 2, 4, 8)
FAMILIES = ("PRIMARY", "ALTERNATE")
NODE_COUNT = 9


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def family_code(family: str) -> int:
    return {"PRIMARY": 5, "ALTERNATE": 17}[family]


POWERS = tuple(pow(GENERATOR, exponent, FIELD) for exponent in range(102))
LOG = {value: exponent for exponent, value in enumerate(POWERS)}
if len(LOG) != 102:
    fail("oracle generator is not primitive")


def descriptor(interface: int, depth: int, family: str) -> dict[str, Any]:
    code = family_code(family)
    return {
        "schema": "CAT_CAS_F103_DUAL_EXPONENT_MOMENT_RELATION_PROGRAM_V1",
        "interface": interface,
        "depth": depth,
        "family": family,
        "owner": (0xE11D0000 + 257 * interface + 131 * depth + code) & 0xFFFFFFFF,
        "node_count": NODE_COUNT,
        "port_type": f"F103_DUAL_EXPONENT_MOMENT_C{interface}_TO_C{interface}",
        "topology": "PUBLIC_ROTATING_CONTROL_HUB8",
        "composition": "REVERSIBLE_RANK1_LEFT_ACTION_THROUGH_STREAMED_VALUE_MOMENT",
        "intersection": "NATIVE_C102_EXPONENT_ADDITION_WITH_ZERO_MASK",
        "observation": [
            (7 * depth + 3 * code + interface) % interface,
            (11 * depth + 5 * code + 2 * interface) % interface,
        ],
    }


def seed_exponent(node: int, family: str, row: int, column: int) -> int:
    code = family_code(family)
    left1 = (code + 3 * node + 2 * row + row * row) % 102
    right1 = (7 + node + 5 * column + column**3) % 102
    left2 = (11 + 5 * node + 3 * row + row**3) % 102
    right2 = (13 + 2 * node + 7 * column + column * column) % 102
    return (left1 * right1 + left2 * right2) % 102


def seed_matrices(interface: int, family: str) -> list[list[list[int]]]:
    return [
        [
            [POWERS[seed_exponent(node, family, row, column)] for column in range(interface)]
            for row in range(interface)
        ]
        for node in range(NODE_COUNT)
    ]


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int, family: str) -> list[int]:
    peers = [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]
    return peers if family == "PRIMARY" else list(reversed(peers))


def shift(interface: int, node: int, index: int, family: str) -> int:
    return (
        3 * node * node
        + 5 * index
        + family_code(family) * (1 + index.bit_count())
    ) % interface


def intersection_exponent(
    hub: int,
    peer: int,
    index: int,
    family: str,
    row: int,
    column: int,
    mutation: int = 0,
) -> int:
    code = family_code(family) + mutation
    left1 = (3 + code + 2 * hub + row + row * row) % 102
    right1 = (5 + peer + 3 * index + 2 * column + column**3) % 102
    left2 = (7 + peer + index + 3 * row + row**3) % 102
    right2 = (11 + hub + code + column + column * column) % 102
    return (left1 * right1 + left2 * right2) % 102


def composition_vectors(
    interface: int,
    hub: int,
    peer: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> tuple[list[int], list[int], int]:
    code = family_code(family) + mutation
    left = [
        POWERS[(17 + code + hub + 3 * x + 2 * x * x) % 102]
        for x in range(interface)
    ]
    right = [
        POWERS[(23 + 2 * peer + 5 * x + 3 * x**3 + index) % 102]
        for x in range(interface)
    ]
    pairing = sum(right[x] * left[x] for x in range(interface)) % FIELD
    coupling = next(value for value in range(1, FIELD) if (1 + value * pairing) % FIELD)
    return left, right, coupling


def rotate(matrix: list[list[int]], amount: int) -> list[list[int]]:
    interface = len(matrix)
    return [
        [matrix[(row - amount) % interface][(column - amount) % interface] for column in range(interface)]
        for row in range(interface)
    ]


def compose(
    matrix: list[list[int]],
    left: list[int],
    right: list[int],
    coupling: int,
) -> list[list[int]]:
    interface = len(matrix)
    moments = [
        sum(right[row] * matrix[row][column] for row in range(interface)) % FIELD
        for column in range(interface)
    ]
    return [
        [
            (matrix[row][column] + coupling * left[row] * moments[column]) % FIELD
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def intersect(
    matrix: list[list[int]],
    hub: int,
    peer: int,
    index: int,
    family: str,
    inverse: bool = False,
    mutation: int = 0,
) -> list[list[int]]:
    interface = len(matrix)
    sign = -1 if inverse else 1
    return [
        [
            matrix[row][column]
            * POWERS[
                sign
                * intersection_exponent(
                    hub, peer, index, family, row, column, mutation
                )
                % 102
            ]
            % FIELD
            for column in range(interface)
        ]
        for row in range(interface)
    ]


def forward(
    interface: int,
    depth: int,
    family: str,
    *,
    action_order: str = "COMPOSE_INTERSECT",
    mutation: int = 0,
    enabled: bool = True,
) -> list[list[list[int]]]:
    matrices = seed_matrices(interface, family)
    if not enabled:
        return matrices
    for index in range(depth):
        hub = hub_index(index, family, mutation)
        for peer in peer_order(hub, family):
            current = rotate(matrices[peer], shift(interface, peer, index, family))
            actions = ("COMPOSE", "INTERSECT")
            if action_order == "INTERSECT_COMPOSE":
                actions = tuple(reversed(actions))
            for action in actions:
                if action == "COMPOSE":
                    left, right, coupling = composition_vectors(
                        interface, hub, peer, index, family, mutation
                    )
                    current = compose(current, left, right, coupling)
                else:
                    current = intersect(
                        current, hub, peer, index, family, mutation=mutation
                    )
            matrices[peer] = current
    return matrices


def reverse(
    matrices: list[list[list[int]]],
    interface: int,
    depth: int,
    family: str,
    *,
    inverse_order: str = "INTERSECT_COMPOSE",
    mutation: int = 0,
) -> list[list[list[int]]]:
    result = [[row[:] for row in matrix] for matrix in matrices]
    for index in reversed(range(depth)):
        hub = hub_index(index, family, mutation)
        for peer in reversed(peer_order(hub, family)):
            current = result[peer]
            actions = ("INTERSECT", "COMPOSE")
            if inverse_order == "COMPOSE_INTERSECT":
                actions = tuple(reversed(actions))
            for action in actions:
                if action == "INTERSECT":
                    current = intersect(
                        current,
                        hub,
                        peer,
                        index,
                        family,
                        inverse=True,
                        mutation=mutation,
                    )
                else:
                    left, right, coupling = composition_vectors(
                        interface, hub, peer, index, family, mutation
                    )
                    pairing = sum(right[x] * left[x] for x in range(interface)) % FIELD
                    inverse_coupling = (
                        -coupling * pow((1 + coupling * pairing) % FIELD, -1, FIELD)
                    ) % FIELD
                    current = compose(current, left, right, inverse_coupling)
            result[peer] = rotate(current, -shift(interface, peer, index, family))
    return result


def boundary(
    matrices: list[list[list[int]]], interface: int, depth: int, family: str
) -> tuple[int, ...]:
    observation = descriptor(interface, depth, family)["observation"]
    return tuple(
        matrix[
            (observation[0] + node) % interface
        ][
            (observation[1] + 2 * node) % interface
        ]
        for node, matrix in enumerate(matrices)
    )


def rank_factor(matrix: list[list[int]], modulus: int) -> tuple[list[list[int]], list[list[int]], list[int]]:
    interface = len(matrix)
    basis: dict[int, list[int]] = {}
    for column in range(interface):
        vector = [matrix[row][column] % modulus for row in range(interface)]
        for pivot in sorted(basis):
            scale = vector[pivot]
            if scale:
                vector = [
                    (vector[row] - scale * basis[pivot][row]) % modulus
                    for row in range(interface)
                ]
        pivot = next((row for row, value in enumerate(vector) if value), None)
        if pivot is None:
            continue
        scale = pow(vector[pivot], -1, modulus)
        vector = [value * scale % modulus for value in vector]
        for old_pivot in list(basis):
            scale = basis[old_pivot][pivot]
            if scale:
                basis[old_pivot] = [
                    (basis[old_pivot][row] - scale * vector[row]) % modulus
                    for row in range(interface)
                ]
        basis[pivot] = vector
    pivots = sorted(basis)
    rank = len(pivots)
    if rank == 0:
        return ([[] for _ in range(interface)], [[] for _ in range(interface)], [])
    left = [[basis[pivot][row] for pivot in pivots] for row in range(interface)]
    right = [[matrix[pivot][column] % modulus for pivot in pivots] for column in range(interface)]
    for row in range(interface):
        for column in range(interface):
            reconstructed = sum(
                left[row][component] * right[column][component]
                for component in range(rank)
            ) % modulus
            if reconstructed != matrix[row][column] % modulus:
                fail("oracle rank factor reconstruction failed")
    return left, right, pivots


def factor_commitment(matrix: list[list[int]], modulus: int) -> tuple[str, int]:
    interface = len(matrix)
    left, right, pivots = rank_factor(matrix, modulus)
    rank = len(pivots)
    nonpivots = [row for row in range(interface) if row not in pivots]
    values = [left[row][component] for row in nonpivots for component in range(rank)]
    values.extend(
        right[column][component]
        for column in range(interface)
        for component in range(rank)
    )
    payload = bytearray(interface * interface)
    payload[: len(values)] = bytes(values)
    pivot_payload = bytearray([255] * interface)
    pivot_payload[:rank] = bytes(pivots)
    digest = hashlib.sha256()
    digest.update(bytes((modulus, rank)))
    digest.update(payload)
    digest.update(pivot_payload)
    return digest.hexdigest(), rank


def dual_chart_receipt(matrix: list[list[int]]) -> tuple[str, dict[str, int]]:
    interface = len(matrix)
    zero = [[int(matrix[row][column] == 0) for column in range(interface)] for row in range(interface)]
    exponents = [
        [0 if matrix[row][column] == 0 else LOG[matrix[row][column]] for column in range(interface)]
        for row in range(interface)
    ]
    _zero_commitment, zero_rank = factor_commitment(zero, 2)
    ranks = {"ZERO_F2": zero_rank}
    for modulus in MODULI:
        _commitment, rank = factor_commitment(exponents, modulus)
        ranks[f"EXP_F{modulus}"] = rank
    # Reconstruct the exact part-byte sequence used by production.
    exact = hashlib.sha256()
    for source, modulus in ((zero, 2), *((exponents, modulus) for modulus in MODULI)):
        left, right, pivots = rank_factor(source, modulus)
        rank = len(pivots)
        nonpivots = [row for row in range(interface) if row not in pivots]
        values = [left[row][component] for row in nonpivots for component in range(rank)]
        values.extend(
            right[column][component]
            for column in range(interface)
            for component in range(rank)
        )
        payload = bytearray(interface * interface)
        payload[: len(values)] = bytes(values)
        pivot_payload = bytearray([255] * interface)
        pivot_payload[:rank] = bytes(pivots)
        exact.update(bytes((modulus, rank)))
        exact.update(payload)
        exact.update(pivot_payload)
    return exact.hexdigest(), ranks


def charts_receipt(matrices: list[list[list[int]]]) -> tuple[str, list[dict[str, int]]]:
    commitments: list[str] = []
    ranks: list[dict[str, int]] = []
    for matrix in matrices:
        commitment, component_ranks = dual_chart_receipt(matrix)
        commitments.append(commitment)
        ranks.append(component_ranks)
    return digest_json(commitments), ranks


def verify_equal(actual: Any, expected: Any, label: str, counter: list[int]) -> None:
    counter[0] += 1
    if actual != expected:
        fail(f"{label}: expected {expected!r}, received {actual!r}")


def run(production: dict[str, Any]) -> dict[str, Any]:
    comparisons = [0]
    verify_equal(production["experiment"]["case_count"], 32, "case count", comparisons)
    indexed = {
        (case["interface"], case["depth"], case["family"]): case
        for case in production["cases"]
    }
    depth1_forward_ranks: dict[tuple[int, str], list[dict[str, int]]] = {}
    depth1_seed_ranks: dict[tuple[int, str], list[dict[str, int]]] = {}
    exact_inverse_cases = 0
    for interface in INTERFACES:
        for family in FAMILIES:
            seeds = seed_matrices(interface, family)
            seed_commitment, seed_ranks = charts_receipt(seeds)
            del seed_commitment
            for depth in DEPTHS:
                case = indexed[(interface, depth, family)]
                public_descriptor = descriptor(interface, depth, family)
                verify_equal(
                    case["program_fingerprint"],
                    digest_json(public_descriptor),
                    "program fingerprint",
                    comparisons,
                )
                matrices = forward(interface, depth, family)
                forward_commitment, forward_ranks = charts_receipt(matrices)
                verify_equal(
                    case["boundary_commitment"],
                    digest_json(list(boundary(matrices, interface, depth, family))),
                    "boundary commitment",
                    comparisons,
                )
                verify_equal(
                    case["forward_chart_commitment"],
                    forward_commitment,
                    "forward chart commitment",
                    comparisons,
                )
                verify_equal(case["seed_component_ranks"], seed_ranks, "seed ranks", comparisons)
                verify_equal(case["forward_component_ranks"], forward_ranks, "forward ranks", comparisons)
                verify_equal(
                    case["phase_resident_bytes"],
                    36 * (interface * interface + interface) + 72 + 16 + 205,
                    "phase resident bytes",
                    comparisons,
                )
                verify_equal(
                    case["classical_resident_bytes"],
                    9 * interface * interface,
                    "classical resident bytes",
                    comparisons,
                )
                restored = reverse(matrices, interface, depth, family)
                verify_equal(restored, seeds, "exact dense inverse", comparisons)
                exact_inverse_cases += 1
                verify_equal(case["payload_restored_exactly"], True, "reported restoration", comparisons)
                verify_equal(case["same_backing_restored"], True, "reported backing", comparisons)
                verify_equal(case["restoration_generation"], 1, "generation", comparisons)
                if depth == 1:
                    depth1_forward_ranks[(interface, family)] = forward_ranks
                    depth1_seed_ranks[(interface, family)] = seed_ranks

    floors: dict[str, dict[str, int]] = {}
    ceilings: dict[str, dict[str, int]] = {}
    seed_ceilings: dict[str, dict[str, int]] = {}
    components = ("EXP_F2", "EXP_F3", "EXP_F17", "ZERO_F2")
    for interface in INTERFACES:
        converted: list[dict[str, int]] = []
        seed_values: list[dict[str, int]] = []
        for family in FAMILIES:
            hub = hub_index(0, family)
            converted.extend(
                ranks
                for node, ranks in enumerate(depth1_forward_ranks[(interface, family)])
                if node != hub
            )
            seed_values.extend(depth1_seed_ranks[(interface, family)])
        floors[str(interface)] = {
            component: min(ranks[component] for ranks in converted)
            for component in components
        }
        ceilings[str(interface)] = {
            component: max(ranks[component] for ranks in converted)
            for component in components
        }
        seed_ceilings[str(interface)] = {
            component: max(ranks[component] for ranks in seed_values)
            for component in components
        }
    verify_equal(
        production["rank_law"]["depth1_converted_node_rank_floor"], floors, "rank floors", comparisons
    )
    verify_equal(
        production["rank_law"]["depth1_converted_node_rank_ceiling"], ceilings, "rank ceilings", comparisons
    )
    verify_equal(
        production["rank_law"]["seed_component_rank_ceiling"], seed_ceilings, "seed ceilings", comparisons
    )

    control_interface, control_depth, control_family = 5, 2, "PRIMARY"
    control_seed = seed_matrices(control_interface, control_family)
    normal = forward(control_interface, control_depth, control_family)
    missing_changes = normal != control_seed
    wrong_changes = reverse(
        normal, control_interface, control_depth, control_family, mutation=1
    ) != control_seed
    reordered_changes = reverse(
        normal,
        control_interface,
        control_depth,
        control_family,
        inverse_order="COMPOSE_INTERSECT",
    ) != control_seed
    normal_boundary = boundary(normal, control_interface, control_depth, control_family)
    disabled_boundary = boundary(
        forward(control_interface, control_depth, control_family, enabled=False),
        control_interface,
        control_depth,
        control_family,
    )
    swapped_boundary = boundary(
        forward(
            control_interface,
            control_depth,
            control_family,
            action_order="INTERSECT_COMPOSE",
        ),
        control_interface,
        control_depth,
        control_family,
    )
    mutated_boundary = boundary(
        forward(control_interface, control_depth, control_family, mutation=1),
        control_interface,
        control_depth,
        control_family,
    )
    semantic_controls = {
        "missing_inverse_changes_payload": missing_changes,
        "wrong_inverse_changes_payload": wrong_changes,
        "reordered_inverse_changes_payload": reordered_changes,
        "disabled_port_changes_boundary": disabled_boundary != normal_boundary,
        "composition_intersection_order_changes_boundary": swapped_boundary != normal_boundary,
        "topology_mutation_changes_boundary": mutated_boundary != normal_boundary,
    }
    for key, value in semantic_controls.items():
        verify_equal(production["controls"][key], value, key, comparisons)

    return {
        "schema": "CAT_CAS_F103_DUAL_EXPONENT_MOMENT_RELATION_NO_GO_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "imports_production_module": False,
        "imports_numpy": False,
        "case_count": 32,
        "comparison_count": comparisons[0],
        "exact_inverse_cases": exact_inverse_cases,
        "reconstructed_program_descriptors": True,
        "reconstructed_dense_forward_and_inverse": True,
        "reconstructed_canonical_chart_commitments": True,
        "reconstructed_boundary_commitments": True,
        "reconstructed_crt_rank_law": True,
        "reconstructed_resident_byte_law": True,
        "semantic_controls": semantic_controls,
        "rank_floor": floors,
        "package_local_fields": [
            "SAME_NATIVE_ARRAY_BACKING_IDENTITY",
            "DECLARED_STREAMED_OPERATION_TOTALS",
            "DECLARED_TRANSIENT_SCRATCH_MAXIMA",
            "PYTHON_NUMPY_RUNTIME_EXCLUSIONS",
            "CUSTODY_REJECTION_CONTROL_IMPLEMENTATION",
        ],
        "claim_ceiling": production["claim_ceiling"],
        "not_verified": [
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_EXECUTION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("production", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    production = json.loads(arguments.production.read_text(encoding="utf-8"))
    result = run(production)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "case_count": result["case_count"],
                "comparison_count": result["comparison_count"],
                "rank_floor": result["rank_floor"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
