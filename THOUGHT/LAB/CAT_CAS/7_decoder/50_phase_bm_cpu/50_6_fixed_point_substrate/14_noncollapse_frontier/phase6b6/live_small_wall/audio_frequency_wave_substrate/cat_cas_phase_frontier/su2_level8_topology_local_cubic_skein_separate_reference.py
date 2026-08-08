#!/usr/bin/env python3
"""Standalone exact reference for the M219 cubic-skein diagnostic.

This file intentionally imports no CAT_CAS production module.  It rebuilds
the cyclotomic field, noncrossing link patterns, local skein action, triangular
cubic shear, Markov boundary, inverse, and leading-term propagation from the
public formulas.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)
FIELD_DEGREE = 16
ROOT_ORDER = 40
EXACT_CASES = (
    *((4, rounds, 0) for rounds in range(1, 7)),
    *((6, rounds, 0) for rounds in range(1, 4)),
    *((8, rounds, 0) for rounds in range(1, 3)),
)
GENERIC_CASES = (
    *((4, rounds, 0) for rounds in range(1, 8)),
    *((6, rounds, 0) for rounds in range(1, 5)),
    *((8, rounds, 0) for rounds in range(1, 3)),
)


def reduce_root(values: list[Fraction]) -> tuple[Fraction, ...]:
    work = values.copy()
    work.extend([Fraction(0)] * max(0, FIELD_DEGREE - len(work)))
    for degree in range(len(work) - 1, FIELD_DEGREE - 1, -1):
        value = work[degree]
        work[degree] = Fraction(0)
        work[degree - 4] += value
        work[degree - 8] -= value
        work[degree - 12] += value
        work[degree - 16] -= value
    return tuple(work[:FIELD_DEGREE])


@dataclass(frozen=True)
class E:
    coordinates: tuple[Fraction, ...]

    @staticmethod
    def integer(value: int) -> "E":
        return E((Fraction(value),) + (Fraction(0),) * 15)

    @staticmethod
    def root(power: int) -> "E":
        exponent = power % ROOT_ORDER
        values = [Fraction(0)] * (exponent + 1)
        values[exponent] = Fraction(1)
        return E(reduce_root(values))

    def __add__(self, other: "E") -> "E":
        return E(tuple(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "E") -> "E":
        return E(tuple(a - b for a, b in zip(self.coordinates, other.coordinates)))

    def __mul__(self, other: "E") -> "E":
        values = [Fraction(0)] * 31
        for left_index, left in enumerate(self.coordinates):
            if not left:
                continue
            for right_index, right in enumerate(other.coordinates):
                if right:
                    values[left_index + right_index] += left * right
        return E(reduce_root(values))

    def token(self) -> str:
        return ":".join(
            f"{value.numerator}/{value.denominator}" for value in self.coordinates
        )


ZERO = E.integer(0)
ONE = E.integer(1)
DELTA = E.root(2) + E.root(38)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: list[E]) -> int:
    return sum(
        signed_bits(coordinate.numerator) + coordinate.denominator.bit_length()
        for value in values
        for coordinate in value.coordinates
    )


def coordinate_bits(values: list[E]) -> dict[str, int]:
    coordinates = [item for value in values for item in value.coordinates]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(item.numerator) for item in coordinates
        ),
        "maximum_denominator_bits": max(
            item.denominator.bit_length() for item in coordinates
        ),
    }


def state_commitment(values: list[E]) -> str:
    return hashlib.sha256("|".join(value.token() for value in values).encode("ascii")).hexdigest()


def boundary_commitment(value: E) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def enumerate_paths(strands: int) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []

    def visit(position: int, height: int, path: tuple[int, ...]) -> None:
        if position == strands:
            if height == 0:
                result.append(path)
            return
        remaining_after = strands - position - 1
        if height > 0 and height - 1 <= remaining_after:
            visit(position + 1, height - 1, path + (height - 1,))
        if height < 8 and height + 1 <= remaining_after:
            visit(position + 1, height + 1, path + (height + 1,))

    visit(0, 0, (0,))
    return tuple(result)


def path_to_pairing(path: tuple[int, ...]) -> tuple[int, ...]:
    pairing = [-1] * (len(path) - 1)
    stack: list[int] = []
    for position, (left, right) in enumerate(zip(path, path[1:])):
        if right == left + 1:
            stack.append(position)
        else:
            peer = stack.pop()
            pairing[position] = peer
            pairing[peer] = position
    if stack or any(peer < 0 for peer in pairing):
        raise RuntimeError("standalone Dyck path did not close")
    return tuple(pairing)


@dataclass(frozen=True)
class Topology:
    strands: int
    pairings: tuple[tuple[int, ...], ...]
    index: dict[tuple[int, ...], int]
    cup_index: int
    targets: tuple[tuple[int, ...], ...]
    cup_flags: tuple[tuple[bool, ...], ...]

    @staticmethod
    def compile(strands: int) -> "Topology":
        pairings = tuple(path_to_pairing(path) for path in enumerate_paths(strands))
        index = {pairing: ordinal for ordinal, pairing in enumerate(pairings)}
        cups = tuple(position ^ 1 for position in range(strands))
        targets: list[tuple[int, ...]] = [tuple()]
        flags: list[tuple[bool, ...]] = [tuple()]
        for generator in range(1, strands):
            left, right = generator - 1, generator
            local_targets = []
            local_flags = []
            for pairing in pairings:
                if pairing[left] == right:
                    local_targets.append(index[pairing])
                    local_flags.append(True)
                    continue
                changed = list(pairing)
                left_peer, right_peer = changed[left], changed[right]
                changed[left], changed[right] = right, left
                changed[left_peer], changed[right_peer] = right_peer, left_peer
                local_targets.append(index[tuple(changed)])
                local_flags.append(False)
            targets.append(tuple(local_targets))
            flags.append(tuple(local_flags))
        return Topology(strands, pairings, index, index[cups], tuple(targets), tuple(flags))

    @property
    def dimension(self) -> int:
        return len(self.pairings)


@dataclass(frozen=True)
class Operation:
    generator: int
    exponent: int


def operations(strands: int, rounds: int, family: int) -> tuple[Operation, ...]:
    result = []
    for round_index in range(rounds):
        for offset in range(strands - 1):
            generator = (
                strands - 1 - offset
                if (round_index + family) % 2
                else offset + 1
            )
            exponent = -1 if (3 * round_index + generator + family) % 5 == 0 else 1
            result.append(Operation(generator, exponent))
    return tuple(result)


def local_scalars(exponent: int) -> tuple[E, E]:
    return (E.root(11), E.root(29)) if exponent == 1 else (E.root(29), E.root(11))


def shear_power(operation: Operation) -> int:
    return (
        7 + 2 * operation.generator
        if operation.exponent == 1
        else 13 + 2 * operation.generator
    ) % ROOT_ORDER


def apply_gate(state: list[E], scratch: list[E], topology: Topology, operation: Operation) -> None:
    alpha, beta = local_scalars(operation.exponent)
    scratch[:] = [ZERO] * topology.dimension
    targets = topology.targets[operation.generator]
    flags = topology.cup_flags[operation.generator]
    for column, value in enumerate(state):
        scratch[column] = scratch[column] + alpha * value
        e_value = DELTA * value if flags[column] else value
        scratch[targets[column]] = scratch[targets[column]] + beta * e_value
    state[:] = scratch


def apply_shear(
    state: list[E], topology: Topology, operation: Operation, *, inverse: bool = False
) -> None:
    phase = E.root(shear_power(operation))
    if inverse:
        phase = ZERO - phase
    targets = topology.targets[operation.generator]
    flags = topology.cup_flags[operation.generator]
    sources = range(topology.dimension - 1, -1, -1) if inverse else range(topology.dimension)
    for source in sources:
        if not flags[source]:
            value = state[source]
            state[targets[source]] = state[targets[source]] + phase * value * value * value


def forward(
    strands: int, rounds: int, family: int
) -> tuple[Topology, list[E], list[E]]:
    topology = Topology.compile(strands)
    state = [ZERO] * topology.dimension
    state[topology.cup_index] = ONE
    scratch = [ZERO] * topology.dimension
    for operation in operations(strands, rounds, family):
        apply_gate(state, scratch, topology, operation)
        apply_shear(state, topology, operation)
    return topology, state, scratch


def reverse(
    state: list[E], scratch: list[E], topology: Topology, strands: int, rounds: int, family: int
) -> None:
    for operation in reversed(operations(strands, rounds, family)):
        apply_shear(state, topology, operation, inverse=True)
        apply_gate(state, scratch, topology, Operation(operation.generator, -operation.exponent))


def loop_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    seen: set[int] = set()
    loops = 0
    for start in range(len(left)):
        if start in seen:
            continue
        loops += 1
        pending = [start]
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            pending.extend((left[item], right[item]))
    return loops


def power(value: E, exponent: int) -> E:
    result = ONE
    for _ in range(exponent):
        result = result * value
    return result


def markov_boundary(state: list[E], topology: Topology) -> E:
    cups = topology.pairings[topology.cup_index]
    # DELTA^(n/2) is a cyclotomic unit in the declared cases.  Its inverse is
    # DELTA^(-n/2), represented here through the exact identity used by the
    # production field.  A small standalone Gaussian elimination keeps the
    # reference independent.
    normalization = power(DELTA, topology.strands // 2)
    columns = [(normalization * E.root(index)).coordinates for index in range(16)]
    matrix = [
        [columns[column][row] for column in range(16)] + [Fraction(row == 0)]
        for row in range(16)
    ]
    for column in range(16):
        pivot = next(row for row in range(column, 16) if matrix[row][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(16):
            if row == column or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * basis
                for value, basis in zip(matrix[row], matrix[column])
            ]
    inverse_normalization = E(tuple(matrix[row][-1] for row in range(16)))
    result = ZERO
    for value, pairing in zip(state, topology.pairings):
        result = result + value * power(DELTA, loop_count(cups, pairing))
    return result * inverse_normalization


Leading = tuple[tuple[int, ...], E]


def add_leading(left: Leading | None, right: Leading) -> Leading:
    if left is None:
        return right
    left_key, right_key = (sum(left[0]), left[0]), (sum(right[0]), right[0])
    if left_key > right_key:
        return left
    if right_key > left_key:
        return right
    coefficient = left[1] + right[1]
    if coefficient == ZERO:
        raise RuntimeError("standalone leading term cancelled")
    return left[0], coefficient


def generic_case(strands: int, rounds: int, family: int) -> dict[str, object]:
    topology = Topology.compile(strands)
    leading: list[Leading] = []
    for index in range(topology.dimension):
        exponent = [0] * topology.dimension
        exponent[index] = 1
        leading.append((tuple(exponent), ONE))
    for operation in operations(strands, rounds, family):
        alpha, beta = local_scalars(operation.exponent)
        scratch: list[Leading | None] = [None] * topology.dimension
        targets = topology.targets[operation.generator]
        flags = topology.cup_flags[operation.generator]
        for column, term in enumerate(leading):
            scratch[column] = add_leading(scratch[column], (term[0], alpha * term[1]))
            factor = DELTA if flags[column] else ONE
            scratch[targets[column]] = add_leading(
                scratch[targets[column]], (term[0], beta * factor * term[1])
            )
        leading = [term for term in scratch if term is not None]
        phase = E.root(shear_power(operation))
        for source in range(topology.dimension):
            if flags[source]:
                continue
            term = leading[source]
            cubed = tuple(3 * value for value in term[0])
            coefficient = phase * term[1] * term[1] * term[1]
            target = targets[source]
            leading[target] = add_leading(leading[target], (cubed, coefficient))
    degrees = [sum(term[0]) for term in leading]
    digest = hashlib.sha256(
        "|".join(
            f"{','.join(map(str, term[0]))}:{term[1].token()}" for term in leading
        ).encode("ascii")
    ).hexdigest()
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "link_pattern_cells": topology.dimension,
        "coordinate_total_degrees": degrees,
        "maximum_total_degree": max(degrees),
        "expected_maximum_total_degree": 3 ** ((strands - 2) * rounds + 1),
        "leading_term_digest": digest,
        "leading_cancellation_encountered": False,
        "leading_exponent_integer_cells": topology.dimension * topology.dimension,
        "leading_coefficient_field_cells": topology.dimension,
    }


def exact_case(strands: int, rounds: int, family: int) -> dict[str, object]:
    topology, state, _ = forward(strands, rounds, family)
    boundary = markov_boundary(state, topology)
    return {
        "strands": strands,
        "rounds": rounds,
        "family": family,
        "steps": rounds * (strands - 1),
        "link_pattern_cells": topology.dimension,
        "forward_nonzero_field_cells": sum(value != ZERO for value in state),
        "forward_payload_bits": payload_bits(state),
        **coordinate_bits(state),
        "forward_state_commitment": state_commitment(state),
        "boundary_commitment": boundary_commitment(boundary),
    }


def transaction_case(strands: int, rounds: int, family: int) -> dict[str, object]:
    topology, state, scratch = forward(strands, rounds, family)
    state_backing, scratch_backing = id(state), id(scratch)
    source = [ZERO] * topology.dimension
    source[topology.cup_index] = ONE
    boundary = markov_boundary(state, topology)
    forward_commitment = state_commitment(state)
    forward_payload = payload_bits(state)
    reverse(state, scratch, topology, strands, rounds, family)
    scratch[:] = [ZERO] * topology.dimension
    return {
        "boundary_commitment": boundary_commitment(boundary),
        "forward_state_commitment": forward_commitment,
        "forward_payload_bits": forward_payload,
        "restoration_error_field_cells": sum(a != b for a, b in zip(state, source)),
        "canonical_post_restoration_state_exact": state == source and all(x == ZERO for x in scratch),
        "same_backing_identity": id(state) == state_backing and id(scratch) == scratch_backing,
        "restored_state": state,
        "topology": topology,
        "scratch": scratch,
    }


def main() -> None:
    exact_cases = [exact_case(*case) for case in EXACT_CASES]
    generic_cases = [generic_case(*case) for case in GENERIC_CASES]
    primary = transaction_case(6, 3, 0)
    topology = primary.pop("topology")
    state = primary.pop("restored_state")
    scratch = primary.pop("scratch")
    state_backing, scratch_backing = id(state), id(scratch)
    for operation in operations(6, 2, 1):
        apply_gate(state, scratch, topology, operation)
        apply_shear(state, topology, operation)
    reuse_boundary = markov_boundary(state, topology)
    reuse_commitment = state_commitment(state)
    reuse_payload = payload_bits(state)
    reverse(state, scratch, topology, 6, 2, 1)
    scratch[:] = [ZERO] * topology.dimension
    source = [ZERO] * topology.dimension
    source[topology.cup_index] = ONE
    reuse = {
        "boundary_commitment": boundary_commitment(reuse_boundary),
        "forward_state_commitment": reuse_commitment,
        "forward_payload_bits": reuse_payload,
        "restoration_error_field_cells": sum(a != b for a, b in zip(state, source)),
        "canonical_post_restoration_state_exact": state == source and all(x == ZERO for x in scratch),
        "same_backing_identity": id(state) == state_backing and id(scratch) == scratch_backing,
    }
    fresh = transaction_case(6, 2, 1)
    fresh.pop("topology")
    fresh.pop("restored_state")
    fresh.pop("scratch")
    result = {
        "schema": "cat_cas.su2_level8_topology_local_cubic_skein_reference.v1",
        "imports_m219_or_m218_production": False,
        "exact_cases": exact_cases,
        "generic_degree_cases": generic_cases,
        "transaction": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
            "fresh_restored_reuse_state_agreement": reuse["forward_state_commitment"] == fresh["forward_state_commitment"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
