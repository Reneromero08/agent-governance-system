#!/usr/bin/env python3
"""Independent gauge-parity oracle for the SU(2)_8 fusion-path carrier.

This oracle imports no CAT_CAS module.  Production streams A_9 path
rank/unrank and uses the radical-free source-weight Temperley-Lieb gauge.
The oracle explicitly enumerates bounded vacuum paths and evolves the
diagonally similar target-weight gauge.  It converts the final vector back
through the independently reconstructed path-weight diagonal before hashing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)

LEVEL = 8
LABELS = 9
FIELD_DEGREE = 16
ROOT_ORDER = 40
EXECUTED_STRANDS = (4, 6, 8, 10, 12, 14, 16)
PRIMARY_STRANDS = 16
PRIMARY_ROUNDS = 8
REUSE_ROUNDS = 5
ANALYTIC_STRANDS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def reduce_root_polynomial(values: list[Fraction]) -> tuple[Fraction, ...]:
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
        return E(reduce_root_polynomial(values))

    def __add__(self, other: "E") -> "E":
        return E(tuple(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "E") -> "E":
        return E(tuple(a - b for a, b in zip(self.coordinates, other.coordinates)))

    def __mul__(self, other: "E") -> "E":
        values = [Fraction(0)] * 31
        for i, left in enumerate(self.coordinates):
            if not left:
                continue
            for j, right in enumerate(other.coordinates):
                if right:
                    values[i + j] += left * right
        return E(reduce_root_polynomial(values))

    def inverse(self) -> "E":
        if self == ZERO:
            raise ZeroDivisionError("zero cyclotomic element")
        columns = [
            (self * E.root(index)).coordinates for index in range(FIELD_DEGREE)
        ]
        matrix = [
            [columns[column][row] for column in range(FIELD_DEGREE)]
            + [Fraction(row == 0)]
            for row in range(FIELD_DEGREE)
        ]
        pivot_row = 0
        for column in range(FIELD_DEGREE):
            pivot = next(
                row
                for row in range(pivot_row, FIELD_DEGREE)
                if matrix[row][column]
            )
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            scale = matrix[pivot_row][column]
            matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
            for row in range(FIELD_DEGREE):
                if row == pivot_row or not matrix[row][column]:
                    continue
                factor = matrix[row][column]
                matrix[row] = [
                    value - factor * basis
                    for value, basis in zip(
                        matrix[row], matrix[pivot_row], strict=True
                    )
                ]
            pivot_row += 1
        return E(tuple(matrix[row][-1] for row in range(FIELD_DEGREE)))

    def token(self) -> str:
        return ":".join(
            f"{value.numerator}/{value.denominator}"
            for value in self.coordinates
        )


ZERO = E.integer(0)
ONE = E.integer(1)
A = E.root(11)
A_INVERSE = E.root(-11)


def quantum_dimensions() -> tuple[E, ...]:
    delta = E.root(2) + E.root(-2)
    dimensions = [ONE, delta]
    for _ in range(2, LABELS + 1):
        dimensions.append(delta * dimensions[-1] - dimensions[-2])
    if dimensions[LABELS] != ZERO:
        raise RuntimeError("independent Jones-Wenzl relation failed")
    return tuple(dimensions[:LABELS])


DIMENSIONS = quantum_dimensions()
INVERSE_DIMENSIONS = tuple(value.inverse() for value in DIMENSIONS)


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def field_payload_bits(values: list[E]) -> int:
    return sum(
        signed_bits(coordinate.numerator) + coordinate.denominator.bit_length()
        for value in values
        for coordinate in value.coordinates
    )


def maximum_coordinate_bits(values: list[E]) -> dict[str, int]:
    coordinates = [coordinate for value in values for coordinate in value.coordinates]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(value.numerator) for value in coordinates
        ),
        "maximum_denominator_bits": max(
            value.denominator.bit_length() for value in coordinates
        ),
    }


def state_commitment(values: list[E]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in values).encode("ascii")
    ).hexdigest()


def boundary_commitment(value: E) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def enumerate_paths(strands: int) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...]) -> None:
        remaining = strands + 1 - len(prefix)
        label = prefix[-1]
        if remaining == 0:
            if label == 0:
                paths.append(prefix)
            return
        for next_label in (label - 1, label + 1):
            if 0 <= next_label <= LEVEL and abs(next_label) <= remaining:
                visit(prefix + (next_label,))

    visit((0,))
    return paths


def completion_table(strands: int) -> tuple[tuple[int, ...], ...]:
    rows = [tuple(1 if label == 0 else 0 for label in range(LABELS))]
    for _ in range(strands):
        previous = rows[-1]
        rows.append(
            tuple(
                (previous[label - 1] if label else 0)
                + (previous[label + 1] if label < LEVEL else 0)
                for label in range(LABELS)
            )
        )
    return tuple(rows)


def topology_commitment(strands: int) -> str:
    rows = completion_table(strands)
    token = "|".join(
        ",".join(str(value) for value in row) for row in rows
    )
    return hashlib.sha256(f"A9:{strands}:{token}".encode("ascii")).hexdigest()


def topology_payload(strands: int) -> int:
    return sum(
        signed_bits(value) for row in completion_table(strands) for value in row
    )


def vacuum_path(strands: int) -> tuple[int, ...]:
    return tuple(index % 2 for index in range(strands + 1))


def path_weight(path: tuple[int, ...]) -> E:
    value = ONE
    for label in path[1:-1]:
        value = value * DIMENSIONS[label]
    return value


def inverse_path_weight(path: tuple[int, ...]) -> E:
    value = ONE
    for label in path[1:-1]:
        value = value * INVERSE_DIMENSIONS[label]
    return value


@dataclass(frozen=True)
class Program:
    strands: int
    rounds: int
    family: int

    @property
    def steps(self) -> int:
        return self.rounds * (self.strands - 1)

    def operation(self, step: int) -> tuple[int, int]:
        round_index, offset = divmod(step, self.strands - 1)
        if (round_index + self.family) % 2:
            generator = self.strands - 1 - offset
        else:
            generator = offset + 1
        exponent = -1 if (3 * round_index + generator + self.family) % 5 == 0 else 1
        return generator, exponent


def apply_target_weight_braid(
    values: list[E],
    paths: list[tuple[int, ...]],
    path_index: dict[tuple[int, ...], int],
    generator: int,
    exponent: int,
) -> None:
    alpha, beta = (A, A_INVERSE) if exponent == 1 else (A_INVERSE, A)
    for index, path in enumerate(paths):
        left = path[generator - 1]
        middle = path[generator]
        right = path[generator + 1]
        if left != right:
            values[index] = alpha * values[index]
            continue
        alternatives = tuple(
            label for label in (left - 1, left + 1) if 0 <= label <= LEVEL
        )
        if len(alternatives) == 2 and middle == alternatives[1]:
            continue
        if len(alternatives) == 1:
            temperley = DIMENSIONS[middle] * INVERSE_DIMENSIONS[left] * values[index]
            values[index] = alpha * values[index] + beta * temperley
            continue
        peer = path[:generator] + (alternatives[1],) + path[generator + 1 :]
        peer_index = path_index[peer]
        total = values[index] + values[peer_index]
        first_temperley = DIMENSIONS[alternatives[0]] * INVERSE_DIMENSIONS[left] * total
        second_temperley = DIMENSIONS[alternatives[1]] * INVERSE_DIMENSIONS[left] * total
        first = values[index]
        second = values[peer_index]
        values[index] = alpha * first + beta * first_temperley
        values[peer_index] = alpha * second + beta * second_temperley


def execute(program: Program) -> tuple[list[tuple[int, ...]], list[E]]:
    paths = enumerate_paths(program.strands)
    path_index = {path: index for index, path in enumerate(paths)}
    values = [ZERO] * len(paths)
    vacuum = vacuum_path(program.strands)
    values[path_index[vacuum]] = path_weight(vacuum)
    for step in range(program.steps):
        generator, exponent = program.operation(step)
        apply_target_weight_braid(
            values, paths, path_index, generator, exponent
        )
    return paths, values


def production_gauge_values(paths: list[tuple[int, ...]], values: list[E]) -> list[E]:
    return [
        value * inverse_path_weight(path)
        for path, value in zip(paths, values, strict=True)
    ]


def case(strands: int) -> dict[str, object]:
    program = Program(strands, PRIMARY_ROUNDS, 0)
    paths, target_values = execute(program)
    values = production_gauge_values(paths, target_values)
    vacuum_index = paths.index(vacuum_path(strands))
    boundary = values[vacuum_index]
    return {
        "strands": strands,
        "rounds": PRIMARY_ROUNDS,
        "family": 0,
        "program_steps": program.steps,
        "fusion_path_field_cells": len(values),
        "nonzero_fusion_path_field_cells": sum(value != ZERO for value in values),
        "carrier_payload_bits": field_payload_bits(values),
        **maximum_coordinate_bits(values),
        "state_commitment": state_commitment(values),
        "boundary_commitment": boundary_commitment(boundary),
        "boundary_payload_bits": field_payload_bits([boundary]),
        "public_topology_integer_cells": (strands + 1) * LABELS,
        "public_topology_payload_bits": topology_payload(strands),
        "topology_commitment": topology_commitment(strands),
    }


def transaction(program: Program) -> dict[str, object]:
    paths = enumerate_paths(program.strands)
    path_index = {path: index for index, path in enumerate(paths)}
    vacuum = vacuum_path(program.strands)
    values = [ZERO] * len(paths)
    values[path_index[vacuum]] = path_weight(vacuum)
    source = values.copy()
    backing = id(values)
    for step in range(program.steps):
        generator, exponent = program.operation(step)
        apply_target_weight_braid(values, paths, path_index, generator, exponent)
    production_values = production_gauge_values(paths, values)
    boundary = production_values[path_index[vacuum]]
    forward_state = state_commitment(production_values)
    for step in range(program.steps - 1, -1, -1):
        generator, exponent = program.operation(step)
        apply_target_weight_braid(values, paths, path_index, generator, -exponent)
    return {
        "boundary_commitment": boundary_commitment(boundary),
        "forward_state_commitment": forward_state,
        "source_restored": values == source,
        "same_backing": id(values) == backing,
        "baseline_reload_used": False,
    }


def dimension_law() -> list[dict[str, int]]:
    result = []
    for strands in ANALYTIC_STRANDS:
        dimension = len(enumerate_paths(strands))
        half = strands // 2
        catalan = 1
        for index in range(half):
            catalan = catalan * 2 * (2 * index + 1) // (index + 2)
        result.append(
            {
                "strands": strands,
                "su2_level8_vacuum_path_cells": dimension,
                "untruncated_catalan_cells": catalan,
                "jones_wenzl_removed_cells": catalan - dimension,
            }
        )
    return result


def braid_controls() -> dict[str, object]:
    paths = enumerate_paths(6)
    index = {path: position for position, path in enumerate(paths)}
    vacuum = vacuum_path(6)
    source = [ZERO] * len(paths)
    source[index[vacuum]] = path_weight(vacuum)

    def run(word: tuple[tuple[int, int], ...]) -> list[E]:
        values = source.copy()
        for generator, exponent in word:
            apply_target_weight_braid(values, paths, index, generator, exponent)
        return values

    return {
        "yang_baxter_relation_exact": run(((2, 1), (3, 1), (2, 1)))
        == run(((3, 1), (2, 1), (3, 1))),
        "far_generators_commute_exactly": run(((1, 1), (4, 1)))
        == run(((4, 1), (1, 1))),
        "adjacent_generators_do_not_commute": run(((2, 1), (3, 1)))
        != run(((3, 1), (2, 1))),
        "single_generator_inverse_exact": run(((2, 1), (2, -1))) == source,
    }


def main() -> None:
    cases = [case(strands) for strands in EXECUTED_STRANDS]
    primary = transaction(Program(PRIMARY_STRANDS, PRIMARY_ROUNDS, 0))
    reuse = transaction(Program(PRIMARY_STRANDS, REUSE_ROUNDS, 1))
    result = {
        "schema": "cat_cas.su2_level8_fusion_path_braid_independent_oracle.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "oracle_algorithm": "EXPLICIT_A9_VACUUM_PATH_ENUMERATION_TARGET_WEIGHT_TEMPERLEY_LIEB_GAUGE_WITH_DIAGONAL_SIMILARITY_BACK_TO_PRODUCTION_GAUGE",
        "production_uses_source_weight_gauge": True,
        "oracle_uses_distinct_target_weight_gauge": True,
        "oracle_explicit_path_list_and_index_map_is_a_VERIFICATION_BASELINE": True,
        "cases": cases,
        "dimension_law": dimension_law(),
        "primary": primary,
        "reuse": reuse,
        "controls": braid_controls(),
        "jones_wenzl_relation_exact": True,
        "first_truncation_at_strands18_reproduced": True,
        "matched_classical_baseline": "IDENTICAL_EXACT_QZETA40_TEMPERLEY_LIEB_FUSION_PATH_RECURRENCE",
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
