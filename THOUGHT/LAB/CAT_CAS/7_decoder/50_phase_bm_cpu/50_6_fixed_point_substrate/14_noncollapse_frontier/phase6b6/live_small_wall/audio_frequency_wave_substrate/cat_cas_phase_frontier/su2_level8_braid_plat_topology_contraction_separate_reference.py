#!/usr/bin/env python3
"""Independent column-frontier parity for the M216 plat contraction.

This reference imports the independently checked M214 exact field and braid
program, but no M216 production code.  It independently reconstructs the
local spacetime factors and contracts them in a fixed column-major order,
distinct from production's topology-compiled min-fill order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product

import su2_level8_fusion_path_braid_phase_relation as braid


Variable = tuple[int, int]
Assignment = tuple[int, ...]
STRANDS = (4, 6, 8, 10, 12, 14, 16)
FAMILIES = (0, 1)
ROUNDS = 8


@dataclass
class Factor:
    variables: tuple[Variable, ...]
    values: dict[Assignment, braid.K]


@dataclass
class Work:
    initial_factor_records: int = 0
    initial_field_cells: int = 0
    join_assignment_pairs: int = 0
    elimination_rows: int = 0
    field_additions: int = 0
    field_multiplications: int = 0
    peak_single_factor_field_cells: int = 0
    peak_live_factor_field_cells: int = 0

    def observe(self, factors: list[Factor], extra: tuple[Factor, ...] = ()) -> None:
        live = tuple(factors) + extra
        self.peak_single_factor_field_cells = max(
            self.peak_single_factor_field_cells,
            max((len(factor.values) for factor in live), default=0),
        )
        self.peak_live_factor_field_cells = max(
            self.peak_live_factor_field_cells,
            sum(len(factor.values) for factor in live),
        )


def domain(strands: int, position: int) -> tuple[int, ...]:
    maximum = min(position, strands - position, braid.LEVEL)
    return tuple(range(position % 2, maximum + 1, 2))


def fixed(strands: int, rounds: int, coordinate: Variable) -> int | None:
    row, position = coordinate
    if position in (0, strands):
        return 0
    if row in (0, rounds):
        return position % 2
    return None


def coordinates(
    round_index: int, generator: int, reverse: bool
) -> tuple[Variable, Variable, Variable, Variable]:
    # The independent derivation follows data dependencies through one sweep:
    # a forward sweep reads the new left and old right neighbors; a reverse
    # sweep reads the old left and new right neighbors.
    if reverse:
        return (
            (round_index, generator - 1),
            (round_index, generator),
            (round_index + 1, generator + 1),
            (round_index + 1, generator),
        )
    return (
        (round_index + 1, generator - 1),
        (round_index, generator),
        (round_index, generator + 1),
        (round_index + 1, generator),
    )


def gate_value(
    left: int, middle: int, right: int, output: int, exponent: int, work: Work
) -> braid.K:
    admissible = (
        abs(left - middle) == 1
        and abs(middle - right) == 1
        and abs(left - output) == 1
        and abs(output - right) == 1
        and (left == right or output == middle)
    )
    if not admissible:
        return braid.ZERO
    alpha, beta = braid.local_braid_scalars(exponent)
    if left != right:
        return alpha
    temperley = braid.QUANTUM_DIMENSIONS[middle] * braid.INVERSE_DIMENSIONS[left]
    work.field_multiplications += 1
    value = beta * temperley
    work.field_multiplications += 1
    if output == middle:
        value = alpha + value
        work.field_additions += 1
    return value


def make_leaf(
    program: braid.BraidProgram,
    round_index: int,
    generator: int,
    exponent: int,
    reverse: bool,
    work: Work,
) -> Factor:
    gate = coordinates(round_index, generator, reverse)
    variables = tuple(
        dict.fromkeys(coordinate for coordinate in gate if fixed(program.strands, program.rounds, coordinate) is None)
    )
    values: dict[Assignment, braid.K] = {}
    for assignment in product(*(domain(program.strands, position) for _row, position in variables)):
        lookup = dict(zip(variables, assignment, strict=True))
        labels = tuple(
            fixed(program.strands, program.rounds, coordinate)
            if fixed(program.strands, program.rounds, coordinate) is not None
            else lookup[coordinate]
            for coordinate in gate
        )
        value = gate_value(*labels, exponent, work)
        if not value.is_zero():
            values[assignment] = value
    return Factor(variables, values)


def multiply(left: Factor, right: Factor, work: Work) -> Factor:
    common = tuple(variable for variable in left.variables if variable in right.variables)
    left_common = tuple(left.variables.index(variable) for variable in common)
    right_common = tuple(right.variables.index(variable) for variable in common)
    right_tail = tuple(
        index for index, variable in enumerate(right.variables) if variable not in common
    )
    right_index: dict[Assignment, list[tuple[Assignment, braid.K]]] = {}
    for assignment, value in right.values.items():
        key = tuple(assignment[index] for index in right_common)
        right_index.setdefault(key, []).append((assignment, value))
    values: dict[Assignment, braid.K] = {}
    for left_assignment, left_value in left.values.items():
        key = tuple(left_assignment[index] for index in left_common)
        for right_assignment, right_value in right_index.get(key, ()):
            output = left_assignment + tuple(right_assignment[index] for index in right_tail)
            value = left_value * right_value
            work.field_multiplications += 1
            work.join_assignment_pairs += 1
            if output in values:
                value = values[output] + value
                work.field_additions += 1
            if value.is_zero():
                values.pop(output, None)
            else:
                values[output] = value
    return Factor(
        left.variables + tuple(variable for variable in right.variables if variable not in common),
        values,
    )


def eliminate(factor: Factor, variable: Variable, work: Work) -> Factor:
    position = factor.variables.index(variable)
    values: dict[Assignment, braid.K] = {}
    for assignment, value in factor.values.items():
        output = assignment[:position] + assignment[position + 1 :]
        work.elimination_rows += 1
        if output in values:
            value = values[output] + value
            work.field_additions += 1
        if value.is_zero():
            values.pop(output, None)
        else:
            values[output] = value
    return Factor(
        factor.variables[:position] + factor.variables[position + 1 :], values
    )


def contract(program: braid.BraidProgram) -> tuple[braid.K, Work]:
    work = Work()
    factors = []
    for step in range(program.steps):
        round_index, _offset = divmod(step, program.strands - 1)
        operation = program.operation(step)
        reverse = (round_index + program.family) % 2 == 1
        factors.append(
            make_leaf(
                program,
                round_index,
                operation.generator,
                operation.exponent,
                reverse,
                work,
            )
        )
    work.initial_factor_records = len(factors)
    work.initial_field_cells = sum(len(factor.values) for factor in factors)
    work.observe(factors)

    variables = sorted(
        {variable for factor in factors for variable in factor.variables},
        key=lambda variable: (variable[1], variable[0]),
    )
    for variable in variables:
        bucket = [factor for factor in factors if variable in factor.variables]
        factors = [factor for factor in factors if variable not in factor.variables]
        bucket.sort(key=lambda factor: (len(factor.values), factor.variables))
        while len(bucket) > 1:
            left = bucket.pop(0)
            right = bucket.pop(0)
            output = multiply(left, right, work)
            work.observe(factors + bucket, (left, right, output))
            bucket.append(output)
            bucket.sort(key=lambda factor: (len(factor.values), factor.variables))
        output = eliminate(bucket.pop(), variable, work)
        work.observe(factors, (output,))
        factors.append(output)

    factors.sort(key=lambda factor: (len(factor.values), factor.variables))
    while len(factors) > 1:
        left = factors.pop(0)
        right = factors.pop(0)
        output = multiply(left, right, work)
        work.observe(factors, (left, right, output))
        factors.append(output)
        factors.sort(key=lambda factor: (len(factor.values), factor.variables))
    if len(factors) != 1 or factors[0].variables or set(factors[0].values) != {()}:
        raise RuntimeError("column-frontier contraction did not close to one scalar")
    return factors[0].values[()], work


def boundary_commitment(value: braid.K) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def case(strands: int, family: int) -> dict[str, object]:
    program = braid.BraidProgram(strands, ROUNDS, family)
    value, work = contract(program)
    topology, vector, _direct_work = braid.execute_forward(program)
    direct = vector[topology.rank(braid.vacuum_path(strands))]
    if value != direct:
        raise RuntimeError("column-frontier reference differs from direct boundary")
    return {
        "strands": strands,
        "rounds": ROUNDS,
        "family": family,
        "boundary_commitment": boundary_commitment(value),
        "exact_direct_boundary_agreement": True,
        "initial_factor_records": work.initial_factor_records,
        "initial_field_cells": work.initial_field_cells,
        "peak_single_factor_field_cells": work.peak_single_factor_field_cells,
        "peak_live_factor_field_cells": work.peak_live_factor_field_cells,
        "join_assignment_pairs": work.join_assignment_pairs,
        "elimination_rows": work.elimination_rows,
        "field_additions": work.field_additions,
        "field_multiplications": work.field_multiplications,
    }


def main() -> None:
    cases = [case(strands, family) for family in FAMILIES for strands in STRANDS]
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_braid_plat_topology_contraction_separate_reference.v1",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "SEPARATE_REFERENCE_PARITY",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "reference_imports_m216_production": False,
                "reference_algorithm": "INDEPENDENT_FIXED_COLUMN_MAJOR_SPARSE_VARIABLE_ELIMINATION",
                "underlying_m214_exact_field_and_braid_program_reused": True,
                "cases": cases,
                "all_cases_exact_direct_boundary_agreement": True,
                "fixed_eight_sweep_strand_family_scope_only": True,
                "distinct_phase_resource_established": False,
                "terminal": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
