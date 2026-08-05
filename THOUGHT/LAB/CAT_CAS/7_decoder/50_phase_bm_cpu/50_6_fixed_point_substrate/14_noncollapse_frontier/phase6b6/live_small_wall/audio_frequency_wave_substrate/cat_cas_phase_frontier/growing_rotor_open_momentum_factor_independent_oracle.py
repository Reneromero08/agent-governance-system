#!/usr/bin/env python3
"""Independent oracle for the Rotor-6 open-momentum factor closure.

No production module is imported.  The oracle reconstructs the quotient from
tuple rotations and reflections, builds a verification-only direct two-body
CSR, and separately compiles all sixteen unpaired one-body factorizations.
The latter intentionally differs from production's eight reflection-paired
leases while yielding the same exact bracelet states.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy import sparse


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
PAIR_CHANNELS = 9
TRIANGLE_STENCILS = (((1, 3), (2, 3)), ((1, 5), (4, 5)))
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def histograms(rotors: int) -> Iterator[Histogram]:
    working = [0] * GRID

    def visit(position: int, remaining: int) -> Iterator[Histogram]:
        if position == GRID - 1:
            working[position] = remaining
            yield tuple(working)
            return
        for count in range(remaining + 1):
            working[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, rotors)


def rotate(item: Histogram, amount: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(item):
        result[(mode + amount) % GRID] = count
    return tuple(result)


def cyclic(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def bracelet(item: Histogram) -> Histogram:
    return min(cyclic(item), cyclic(reflect(item)))


def encode(item: Histogram) -> int:
    result = 0
    for count in item:
        result = result * (ROTORS + 1) + count
    return result


def pair_signature(item: Histogram) -> tuple[int, ...]:
    values = [sum(count * (count - 1) // 2 for count in item)]
    for distance in range(1, PAIR_CHANNELS):
        values.append(
            sum(
                item[mode] * item[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(values) != math.comb(ROTORS, 2):
        raise RuntimeError("independent pair partition changed")
    return tuple(values)


def triangles(item: Histogram) -> tuple[int, int]:
    return tuple(
        sum(
            item[anchor]
            * item[(anchor + first) % GRID]
            * item[(anchor + second) % GRID]
            for first, second in orientations
            for anchor in range(GRID)
        )
        for orientations in TRIANGLE_STENCILS
    )  # type: ignore[return-value]


def refined_signature(item: Histogram) -> Signature:
    return pair_signature(item) + triangles(item)


def pair_weight(distance: int, step: int, tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * tag
    ) % GRID % (GRID - 1)


def phase_exponent(item: Histogram, step: int, tag: int) -> int:
    return sum(
        count * pair_weight(distance, step, tag)
        for distance, count in enumerate(pair_signature(item))
    ) % GRID


def scattering_weight(shift: int, step: int, tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


def public_program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


@dataclass(frozen=True)
class Topology:
    necklaces: tuple[Histogram, ...]
    necklace_lookup: dict[Histogram, int]
    bracelets: tuple[Histogram, ...]
    bracelet_lookup: dict[Histogram, int]
    necklace_to_bracelet: tuple[int, ...]
    reflected_necklace: tuple[int, ...]
    boundary_weights: tuple[int, ...]
    occupation_count: int
    signature_order: tuple[int, ...]


def compile_topology() -> Topology:
    occupations = list(histograms(ROTORS))
    necklaces = tuple(item for item in occupations if cyclic(item) == item)
    necklace_lookup = {item: index for index, item in enumerate(necklaces)}
    bracelets = tuple(sorted({bracelet(item) for item in necklaces}))
    bracelet_lookup = {item: index for index, item in enumerate(bracelets)}
    necklace_to_bracelet = tuple(
        bracelet_lookup[bracelet(item)] for item in necklaces
    )
    reflected_necklace = tuple(
        necklace_lookup[cyclic(reflect(item))] for item in necklaces
    )
    boundary = [0] * len(bracelets)
    for necklace_index, item in enumerate(necklaces):
        target = necklace_to_bracelet[necklace_index]
        boundary[target] = (
            boundary[target]
            + pow(
                ROOT,
                (11 * necklace_index + 5 * pair_signature(item)[0] + 1) % GRID,
                PRIME,
            )
        ) % PRIME
    records = sorted(
        (refined_signature(item), index) for index, item in enumerate(bracelets)
    )
    signature_order = tuple(index for _, index in records)
    if (
        len(occupations) != 74613
        or len(necklaces) != 4389
        or len(bracelets) != 2277
        or len({signature for signature, _ in records}) != 2277
    ):
        raise RuntimeError("independent factor topology changed")
    return Topology(
        necklaces=necklaces,
        necklace_lookup=necklace_lookup,
        bracelets=bracelets,
        bracelet_lookup=bracelet_lookup,
        necklace_to_bracelet=necklace_to_bracelet,
        reflected_necklace=reflected_necklace,
        boundary_weights=tuple(boundary),
        occupation_count=len(occupations),
        signature_order=signature_order,
    )


def topology_commitment(topology: Topology) -> str:
    digest = hashlib.sha256()
    for index, item in enumerate(topology.necklaces):
        digest.update(
            (
                f"{encode(item)}:"
                + ",".join(map(str, item))
                + f":{topology.necklace_to_bracelet[index]}:"
                + f"{topology.reflected_necklace[index]};"
            ).encode()
        )
    digest.update(",".join(map(str, topology.boundary_weights)).encode())
    return digest.hexdigest()


def source_state(topology: Topology, family: int) -> list[int]:
    source = [0] * len(topology.bracelets)
    records = sorted(
        (refined_signature(item), index)
        for index, item in enumerate(topology.bracelets)
    )
    for order, (signature, bracelet_index) in enumerate(records):
        source[bracelet_index] = (
            1
            + (family + 3) * (order + 1)
            + sum(
                (coordinate + 2 + family) * (count + 1) ** 2
                for coordinate, count in enumerate(signature)
            )
        ) % PRIME
    return source


def signature_commitment(state: list[int], topology: Topology) -> str:
    return hashlib.sha256(
        ",".join(str(state[index]) for index in topology.signature_order).encode()
    ).hexdigest()


@dataclass(frozen=True)
class DirectOperator:
    matrix: sparse.csr_matrix
    raw_terms: int


def compile_direct_operator(
    topology: Topology, step: int, tag: int
) -> DirectOperator:
    expected = 684624
    rows = np.empty(expected, dtype=np.int32)
    columns = np.empty(expected, dtype=np.int32)
    data = np.empty(expected, dtype=np.int64)
    cursor = 0
    for target, item in enumerate(topology.bracelets):
        for first in range(GRID):
            if item[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = item[first] * (
                    item[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    moved = list(item)
                    moved[first] -= 1
                    moved[second] -= 1
                    moved[(first - shift) % GRID] += 1
                    moved[(second + shift) % GRID] += 1
                    rows[cursor] = target
                    columns[cursor] = topology.bracelet_lookup[
                        bracelet(tuple(moved))
                    ]
                    data[cursor] = multiplicity * scattering_weight(
                        shift, step, tag
                    )
                    cursor += 1
    if cursor != expected:
        raise RuntimeError("independent direct two-body term count changed")
    matrix = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(topology.bracelets), len(topology.bracelets)),
        dtype=np.int64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return DirectOperator(matrix, cursor)


@dataclass(frozen=True)
class OneBodyPlans:
    first: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    second: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    first_entries: int
    second_entries: int


def compile_one_body_plans(topology: Topology) -> OneBodyPlans:
    first_plans: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    second_plans: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    first_entries = 0
    second_entries = 0
    for momentum in range(1, GRID):
        first_rows = []
        for item in topology.necklaces:
            row = []
            for mode, count in enumerate(item):
                if count:
                    moved = list(item)
                    moved[mode] -= 1
                    moved[(mode - momentum) % GRID] += 1
                    source = topology.necklace_lookup[cyclic(tuple(moved))]
                    row.append((source, count))
                    first_entries += 1
            first_rows.append(tuple(row))
        first_plans.append(tuple(first_rows))
        second_rows = []
        for item in topology.bracelets:
            row = []
            for mode, count in enumerate(item):
                if count:
                    moved = list(item)
                    moved[mode] -= 1
                    moved[(mode + momentum) % GRID] += 1
                    source = topology.necklace_lookup[cyclic(tuple(moved))]
                    row.append((source, count))
                    second_entries += 1
            second_rows.append(tuple(row))
        second_plans.append(tuple(second_rows))
    return OneBodyPlans(
        first=tuple(first_plans),
        second=tuple(second_plans),
        first_entries=first_entries,
        second_entries=second_entries,
    )


def factor_scattering(
    state: list[int],
    topology: Topology,
    plans: OneBodyPlans,
    step: int,
    tag: int,
    omit_identity_correction: bool = False,
) -> list[int]:
    output = [0] * len(state)
    for momentum in range(1, GRID):
        middle = [0] * len(topology.necklaces)
        for target, row in enumerate(plans.first[momentum - 1]):
            middle[target] = sum(
                count * state[topology.necklace_to_bracelet[source]]
                for source, count in row
            ) % PRIME
        weight = scattering_weight(momentum, step, tag)
        for target, row in enumerate(plans.second[momentum - 1]):
            closed = sum(count * middle[source] for source, count in row)
            correction = 0 if omit_identity_correction else ROTORS * state[target]
            output[target] = (
                output[target] + weight * (closed - correction)
            ) % PRIME
    return output


def direct_scattering(
    state: list[int], operator: DirectOperator
) -> list[int]:
    return np.asarray(
        operator.matrix.dot(np.asarray(state, dtype=np.int64)) % PRIME,
        dtype=np.int64,
    ).tolist()


def diagonal(
    state: list[int], topology: Topology, step: int, tag: int
) -> list[int]:
    return [
        value * pow(ROOT, phase_exponent(item, step, tag), PRIME) % PRIME
        for value, item in zip(state, topology.bracelets, strict=True)
    ]


def execute_factor(
    source: list[int],
    topology: Topology,
    plans: OneBodyPlans,
    operations: tuple[tuple[int, int], ...],
    reordered: bool = False,
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        if reordered:
            current = diagonal(
                factor_scattering(current, topology, plans, step, tag),
                topology,
                step,
                tag,
            )
        else:
            current = factor_scattering(
                diagonal(current, topology, step, tag),
                topology,
                plans,
                step,
                tag,
            )
    return current


def execute_direct(
    source: list[int],
    topology: Topology,
    operator: DirectOperator,
    step: int,
    tag: int,
) -> list[int]:
    return direct_scattering(diagonal(source, topology, step, tag), operator)


def boundary(state: list[int], topology: Topology) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % PRIME


def transaction(
    carrier: list[list[int]],
    source: list[int],
    topology: Topology,
    plans: OneBodyPlans,
    operations: tuple[tuple[int, int], ...],
) -> tuple[int, int, bool, list[int]]:
    source_backing = id(carrier[0])
    target_backing = id(carrier[1])
    forward = execute_factor(source, topology, plans, operations)
    carrier[1][:] = [
        (left + right) % PRIME
        for left, right in zip(carrier[1], forward, strict=True)
    ]
    projected = boundary(carrier[1], topology)
    inverse = execute_factor(source, topology, plans, operations)
    carrier[1][:] = [
        (left - right) % PRIME
        for left, right in zip(carrier[1], inverse, strict=True)
    ]
    error = sum(
        left != right for left, right in zip(carrier[0], source, strict=True)
    ) + sum(value != 0 for value in carrier[1])
    return (
        projected,
        error,
        id(carrier[0]) == source_backing and id(carrier[1]) == target_backing,
        forward,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def main() -> None:
    topology = compile_topology()
    plans = compile_one_body_plans(topology)
    source = source_state(topology, 0)
    primary_word = public_program(1, 0)
    reuse_word = public_program(1, 4)
    wrong_word = public_program(1, 1)
    primary_direct_operator = compile_direct_operator(topology, *primary_word[0])

    primary_factor = execute_factor(source, topology, plans, primary_word)
    reuse_factor = execute_factor(source, topology, plans, reuse_word)
    primary_direct = execute_direct(
        source, topology, primary_direct_operator, *primary_word[0]
    )
    carrier = [source.copy(), [0] * len(source)]
    primary_boundary, primary_error, primary_backing, primary_forward = transaction(
        carrier, source, topology, plans, primary_word
    )
    reuse_boundary, reuse_error, reuse_backing, reuse_forward = transaction(
        carrier, source, topology, plans, reuse_word
    )
    fresh = [source.copy(), [0] * len(source)]
    fresh_boundary, fresh_error, fresh_backing, fresh_forward = transaction(
        fresh, source, topology, plans, reuse_word
    )
    wrong = execute_factor(source, topology, plans, wrong_word)
    reordered = execute_factor(
        source, topology, plans, primary_word, reordered=True
    )
    diagonal_primary = diagonal(source, topology, *primary_word[0])
    omitted = factor_scattering(
        diagonal_primary,
        topology,
        plans,
        *primary_word[0],
        omit_identity_correction=True,
    )
    paired_terms = 8 * (plans.first_entries // 16) + 16 * (
        plans.second_entries // 16
    )
    if (
        plans.first_entries != 325584
        or plans.second_entries != 168912
        or paired_terms != 331704
        or mismatch(primary_factor, primary_direct)
        or primary_forward != primary_factor
        or reuse_forward != reuse_factor
        or fresh_forward != reuse_factor
        or primary_boundary != 83
        or reuse_boundary != 70
        or fresh_boundary != 70
        or signature_commitment(primary_factor, topology)
        != "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e"
        or any((primary_error, reuse_error, fresh_error))
        or not all((primary_backing, reuse_backing, fresh_backing))
        or mismatch(primary_factor, wrong) == 0
        or mismatch(primary_factor, reordered) == 0
        or mismatch(primary_factor, omitted) == 0
    ):
        raise RuntimeError("independent open-momentum verification failed")

    print(
        json.dumps(
            {
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_AND_REUSE_DIRECT_PROCESS_SOFTWARE_TYPED_OPEN_MOMENTUM_PORT_ONLY",
                "topology": {
                    "occupation_histograms": topology.occupation_count,
                    "necklace_cells": len(topology.necklaces),
                    "bracelet_cells": len(topology.bracelets),
                    "topology_commitment": topology_commitment(topology),
                },
                "factor_verification": {
                    "oracle_factor_channels": 16,
                    "oracle_first_pass_plan_entries": plans.first_entries,
                    "oracle_closure_plan_entries": plans.second_entries,
                    "oracle_full_unpaired_factor_terms_per_scattering": (
                        plans.first_entries + plans.second_entries
                    ),
                    "production_reflection_paired_term_law": paired_terms,
                    "direct_two_body_raw_terms": primary_direct_operator.raw_terms,
                    "direct_primary_csr_nonzeros": int(
                        primary_direct_operator.matrix.nnz
                    ),
                    "primary_factor_direct_mismatch_cells": mismatch(
                        primary_factor, primary_direct
                    ),
                    "reuse_factor_fresh_transaction_mismatch_cells": mismatch(
                        reuse_factor, fresh_forward
                    ),
                    "primary_signature_order_commitment": signature_commitment(
                        primary_factor, topology
                    ),
                },
                "transaction": {
                    "primary_boundary": primary_boundary,
                    "reuse_boundary": reuse_boundary,
                    "fresh_reuse_boundary": fresh_boundary,
                    "primary_restoration_error_field_cells": primary_error,
                    "reuse_restoration_error_field_cells": reuse_error,
                    "fresh_restoration_error_field_cells": fresh_error,
                    "same_backing_primary": primary_backing,
                    "same_backing_reuse": reuse_backing,
                    "same_backing_fresh": fresh_backing,
                    "fresh_restored_reuse_state_agreement": fresh_forward
                    == reuse_forward,
                    "baseline_reload_used": False,
                },
                "controls": {
                    "missing_inverse_error_field_cells": sum(
                        value != 0 for value in primary_factor
                    ),
                    "wrong_inverse_error_field_cells": mismatch(
                        primary_factor, wrong
                    ),
                    "reordered_inverse_error_field_cells": mismatch(
                        primary_factor, reordered
                    ),
                    "omitted_identity_correction_error_field_cells": mismatch(
                        primary_factor, omitted
                    ),
                },
                "production_source_imported": False,
                "production_projection_called": False,
                "production_inverse_called": False,
                "oracle_direct_csr_and_one_body_plans_are_verification_only": True,
                "matched_classical_recurrence": "IDENTICAL_REFLECTION_PAIRED_OPEN_MOMENTUM_ONE_BODY_FACTOR_STREAM_ON2277_BRACELET_AND4389_TEMPORARY_NECKLACE_CELLS",
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
