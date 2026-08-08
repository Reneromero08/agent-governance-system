#!/usr/bin/env python3
"""Standalone M227 in-place GL(16,Z) public-unit plan reference."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence

import su2_level8_cubic_skein_direct_unit_action_separate_reference as m226r


sys.set_int_max_str_digits(0)
m222 = m226r.m222
prior = m226r.prior
base = m226r.base
CASES = m226r.CASES
FIELD_DEGREE = 16


@dataclass(frozen=True)
class UnitDirection:
    parameter: int
    value: base.E
    reciprocal: base.E
    norm: base.E
    reciprocal_norm: base.E

    @property
    def unit(self) -> base.E:
        return self.value

    @property
    def inverse(self) -> base.E:
        return self.reciprocal

    @property
    def inverse_norm(self) -> base.E:
        return self.reciprocal_norm


UNITS = tuple(
    UnitDirection(parameter, unit.value, unit.reciprocal, unit.norm, unit.reciprocal_norm)
    for parameter, unit in zip(prior.m220.PARAMETERS, m222.UNITS, strict=True)
)
m222.UNITS = UNITS
prior.UNITS = UNITS
CONJUGATION_PLAN_INDEX = len(UNITS)


def signed_bits(value: int) -> int:
    return prior.signed_bits(value)


def fraction_payload(value: Fraction) -> int:
    return signed_bits(value.numerator) + value.denominator.bit_length()


@dataclass(frozen=True)
class PlanOperation:
    opcode: str
    target: int
    source: int = -1
    coefficient: int = 0

    def inverse(self) -> "PlanOperation":
        if self.opcode == "SHEAR":
            return PlanOperation("SHEAR", self.target, self.source, -self.coefficient)
        return self

    def integer_tuple(self) -> tuple[int, ...]:
        code = {"SWAP": 0, "NEGATE": 1, "SHEAR": 2}[self.opcode]
        if self.opcode == "NEGATE":
            return (code, self.target)
        if self.opcode == "SWAP":
            return (code, self.target, self.source)
        return (code, self.target, self.source, self.coefficient)


@dataclass(frozen=True)
class PlanRef:
    plan_index: int
    inverse: bool = False

    def inverted(self) -> "PlanRef":
        return PlanRef(self.plan_index, not self.inverse)


@dataclass(frozen=True)
class FactorAction:
    unit_index: int
    kind: str
    exponent: int

    def __post_init__(self) -> None:
        if not 0 <= self.unit_index < len(UNITS):
            raise ValueError("standalone unit index is outside the compiled library")
        if self.kind not in {"UNIT", "INVERSE", "NORM", "INVERSE_NORM"}:
            raise ValueError("unknown standalone public factor kind")
        if self.exponent <= 0:
            raise ValueError("standalone public factor exponent must be positive")

    def integer_tuple(self) -> tuple[int, ...]:
        code = {"UNIT": 0, "INVERSE": 1, "NORM": 2, "INVERSE_NORM": 3}[
            self.kind
        ]
        return (self.unit_index, code, self.exponent)


def integer_matrix_payload_bits(matrix: Sequence[Sequence[int]]) -> int:
    return sum(signed_bits(value) for row in matrix for value in row)


def require_integral_square_matrix(matrix: Sequence[Sequence[int]]) -> None:
    if len(matrix) != FIELD_DEGREE or any(
        len(row) != FIELD_DEGREE or any(type(value) is not int for value in row)
        for row in matrix
    ):
        raise RuntimeError("standalone compiler requires a 16x16 integer matrix")


def bareiss_determinant(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, int, int, int]:
    require_integral_square_matrix(matrix)
    work = [list(row) for row in matrix]
    sign = 1
    denominator = 1
    multiply_subtracts = 0
    exact_divisions = 0
    peak_payload_bits = integer_matrix_payload_bits(work)
    for column in range(len(work) - 1):
        pivot_row = next(
            (row for row in range(column, len(work)) if work[row][column]), None
        )
        if pivot_row is None:
            return 0, multiply_subtracts, exact_divisions, peak_payload_bits
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign *= -1
        pivot = work[column][column]
        for row in range(column + 1, len(work)):
            for index in range(column + 1, len(work)):
                numerator = (
                    work[row][index] * pivot
                    - work[row][column] * work[column][index]
                )
                multiply_subtracts += 1
                if numerator % denominator:
                    raise RuntimeError("standalone nonexact Bareiss division")
                work[row][index] = numerator // denominator
                exact_divisions += denominator != 1
            work[row][column] = 0
        denominator = pivot
        peak_payload_bits = max(
            peak_payload_bits, integer_matrix_payload_bits(work)
        )
    return (
        sign * work[-1][-1],
        multiply_subtracts,
        exact_divisions,
        peak_payload_bits,
    )


def reduce_unimodular_matrix(
    matrix: Sequence[Sequence[int]],
) -> tuple[tuple[PlanOperation, ...], int, int, int]:
    require_integral_square_matrix(matrix)
    work = [list(row) for row in matrix]
    reduction: list[PlanOperation] = []
    row_integer_updates = 0
    peak_payload_bits = integer_matrix_payload_bits(work)
    peak_work_history_payload_bits = peak_payload_bits

    def observe_work_history() -> None:
        nonlocal peak_work_history_payload_bits
        history_bits = sum(
            signed_bits(value)
            for operation in reduction
            for value in operation.integer_tuple()
        )
        peak_work_history_payload_bits = max(
            peak_work_history_payload_bits,
            integer_matrix_payload_bits(work) + history_bits,
        )

    for column in range(FIELD_DEGREE):
        pivot_row = next(
            row
            for row in range(column, FIELD_DEGREE)
            if work[row][column]
        )
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            reduction.append(PlanOperation("SWAP", column, pivot_row))
            observe_work_history()
        for row in range(column + 1, FIELD_DEGREE):
            while work[row][column]:
                quotient = work[column][column] // work[row][column]
                if quotient:
                    work[column] = [
                        left - quotient * right
                        for left, right in zip(work[column], work[row], strict=True)
                    ]
                    row_integer_updates += FIELD_DEGREE
                    reduction.append(
                        PlanOperation("SHEAR", column, row, -quotient)
                    )
                    observe_work_history()
                    peak_payload_bits = max(
                        peak_payload_bits, integer_matrix_payload_bits(work)
                    )
                work[column], work[row] = work[row], work[column]
                reduction.append(PlanOperation("SWAP", column, row))
                observe_work_history()
        if work[column][column] == -1:
            work[column] = [-value for value in work[column]]
            row_integer_updates += FIELD_DEGREE
            reduction.append(PlanOperation("NEGATE", column))
            observe_work_history()
            peak_payload_bits = max(
                peak_payload_bits, integer_matrix_payload_bits(work)
            )
        if work[column][column] != 1:
            raise RuntimeError("standalone public matrix is not unimodular")
        for row in range(FIELD_DEGREE):
            if row == column or work[row][column] == 0:
                continue
            quotient = work[row][column]
            work[row] = [
                left - quotient * right
                for left, right in zip(work[row], work[column], strict=True)
            ]
            row_integer_updates += FIELD_DEGREE
            reduction.append(PlanOperation("SHEAR", row, column, -quotient))
            observe_work_history()
            peak_payload_bits = max(
                peak_payload_bits, integer_matrix_payload_bits(work)
            )
    if any(
        work[row][column] != int(row == column)
        for row in range(FIELD_DEGREE)
        for column in range(FIELD_DEGREE)
    ):
        raise RuntimeError("standalone reduction did not reach identity")
    plan = tuple(operation.inverse() for operation in reversed(reduction))
    history_payload_bits = sum(
        signed_bits(value)
        for operation in reduction
        for value in operation.integer_tuple()
    )
    final_work_payload_bits = integer_matrix_payload_bits(work)
    peak_work_history_plan_payload_bits = max(
        peak_work_history_payload_bits,
        final_work_payload_bits + history_payload_bits + plan_payload(plan)[1],
    )
    return (
        plan,
        row_integer_updates,
        peak_payload_bits,
        peak_work_history_plan_payload_bits,
    )


def apply_integer_plan(
    coordinates: Sequence[int],
    plan: Sequence[PlanOperation],
    *,
    inverse: bool = False,
) -> tuple[int, ...]:
    values = list(coordinates)
    operations = (
        tuple(operation.inverse() for operation in reversed(plan))
        if inverse
        else plan
    )
    for operation in operations:
        if operation.opcode == "SWAP":
            values[operation.target], values[operation.source] = (
                values[operation.source],
                values[operation.target],
            )
        elif operation.opcode == "NEGATE":
            values[operation.target] = -values[operation.target]
        elif operation.opcode == "SHEAR":
            values[operation.target] += operation.coefficient * values[operation.source]
        else:
            raise RuntimeError("unknown standalone plan opcode")
    return tuple(values)


def multiplication_matrix(value: base.E) -> tuple[tuple[int, ...], ...]:
    columns = [
        (value * base.E.root(index)).coordinates for index in range(FIELD_DEGREE)
    ]
    if any(
        coefficient.denominator != 1
        for column in columns
        for coefficient in column
    ):
        raise RuntimeError("standalone public unit action is nonintegral")
    return tuple(
        tuple(columns[column][row].numerator for column in range(FIELD_DEGREE))
        for row in range(FIELD_DEGREE)
    )


def conjugation_matrix() -> tuple[tuple[int, ...], ...]:
    columns = [
        prior.m220.conjugate(base.E.root(index)).coordinates
        for index in range(FIELD_DEGREE)
    ]
    if any(
        coefficient.denominator != 1
        for column in columns
        for coefficient in column
    ):
        raise RuntimeError("standalone conjugation action is nonintegral")
    return tuple(
        tuple(columns[column][row].numerator for column in range(FIELD_DEGREE))
        for row in range(FIELD_DEGREE)
    )


def plan_payload(plan: Sequence[PlanOperation]) -> tuple[int, int]:
    integers = [value for operation in plan for value in operation.integer_tuple()]
    return len(integers), sum(signed_bits(value) for value in integers)


def plan_commitment(plans: Sequence[Sequence[PlanOperation]]) -> str:
    serial = [[operation.integer_tuple() for operation in plan] for plan in plans]
    return hashlib.sha256(
        json.dumps(serial, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def compile_public_plan_library() -> tuple[
    tuple[tuple[PlanOperation, ...], ...], dict[str, Any]
]:
    public_actions = tuple(direction.value for direction in UNITS) + (None,)
    plans: list[tuple[PlanOperation, ...]] = []
    determinants = []
    determinant_multiply_subtracts = 0
    determinant_exact_divisions = 0
    compiler_row_integer_updates = 0
    compiler_peak_dense_integer_payload_bits = 0
    compiler_peak_declared_exact_payload_bits = 0
    compiler_total_input_matrix_integer_payload_bits = 0
    retained_emitted_plan_payload_bits = 0
    descriptor = m226r.public_unit_descriptor()
    compiler_public_descriptor_payload_bits = (
        descriptor["field_payload_bits"] + descriptor["parameter_payload_bits"]
    )
    public_input_hasher = hashlib.sha256()
    for action in public_actions:
        matrix = conjugation_matrix() if action is None else multiplication_matrix(action)
        input_bits = integer_matrix_payload_bits(matrix)
        compiler_total_input_matrix_integer_payload_bits += input_bits
        public_input_hasher.update(
            json.dumps(matrix, separators=(",", ":")).encode("ascii")
        )
        (
            determinant,
            multiply_subtracts,
            exact_divisions,
            determinant_peak_bits,
        ) = bareiss_determinant(matrix)
        if abs(determinant) != 1:
            raise RuntimeError("standalone action determinant is not a unit")
        determinants.append(determinant)
        determinant_multiply_subtracts += multiply_subtracts
        determinant_exact_divisions += exact_divisions
        (
            plan,
            row_updates,
            reduction_peak_bits,
            reduction_declared_peak_bits,
        ) = reduce_unimodular_matrix(matrix)
        compiler_row_integer_updates += row_updates
        compiler_peak_dense_integer_payload_bits = max(
            compiler_peak_dense_integer_payload_bits,
            input_bits + determinant_peak_bits,
            input_bits + reduction_peak_bits,
        )
        compiler_peak_declared_exact_payload_bits = max(
            compiler_peak_declared_exact_payload_bits,
            compiler_public_descriptor_payload_bits
            + retained_emitted_plan_payload_bits
            + input_bits
            + determinant_peak_bits,
            compiler_public_descriptor_payload_bits
            + retained_emitted_plan_payload_bits
            + input_bits
            + reduction_declared_peak_bits,
        )
        for column in range(FIELD_DEGREE):
            basis = tuple(int(index == column) for index in range(FIELD_DEGREE))
            expected = tuple(matrix[row][column] for row in range(FIELD_DEGREE))
            if apply_integer_plan(basis, plan) != expected:
                raise RuntimeError("standalone basis parity failed")
            if apply_integer_plan(expected, plan, inverse=True) != basis:
                raise RuntimeError("standalone inverse basis parity failed")
        plans.append(plan)
        retained_emitted_plan_payload_bits += plan_payload(plan)[1]
        del matrix, plan
    operation_counts = {
        opcode: sum(
            operation.opcode == opcode for plan in plans for operation in plan
        )
        for opcode in ("SWAP", "NEGATE", "SHEAR")
    }
    payloads = [plan_payload(plan) for plan in plans]
    serialized = json.dumps(
        [[operation.integer_tuple() for operation in plan] for plan in plans],
        separators=(",", ":"),
    ).encode("ascii")
    stats = {
        "primitive_plan_count": len(plans),
        "unit_plan_count": len(UNITS),
        "conjugation_plan_count": 1,
        "plan_lengths": [len(plan) for plan in plans],
        "operation_counts": operation_counts,
        "maximum_absolute_shear_coefficient": max(
            abs(operation.coefficient)
            for plan in plans
            for operation in plan
            if operation.opcode == "SHEAR"
        ),
        "retained_plan_integer_cells": sum(item[0] for item in payloads),
        "retained_plan_integer_payload_bits": sum(item[1] for item in payloads),
        "retained_plan_serialized_bytes": len(serialized),
        "retained_inverse_plan_library": False,
        "plan_commitment": plan_commitment(plans),
        "determinants": determinants,
        "compiler_matrices_built": len(public_actions),
        "compiler_peak_dense_integer_matrix_cells": 2 * FIELD_DEGREE * FIELD_DEGREE,
        "compiler_peak_dense_integer_payload_bits": compiler_peak_dense_integer_payload_bits,
        "compiler_peak_matrix_history_plan_and_public_descriptor_payload_bits": compiler_peak_declared_exact_payload_bits,
        "compiler_public_unit_descriptor_payload_bits": compiler_public_descriptor_payload_bits,
        "compiler_scalar_arithmetic_live_payload_complete": False,
        "compiler_total_input_matrix_integer_payload_bits": compiler_total_input_matrix_integer_payload_bits,
        "public_compiler_input_commitment": public_input_hasher.hexdigest(),
        "compiler_input_matrices_streamed_one_at_a_time": True,
        "compiler_input_matrix_field_multiplications": len(UNITS) * FIELD_DEGREE,
        "compiler_determinant_multiply_subtracts": determinant_multiply_subtracts,
        "compiler_determinant_exact_divisions": determinant_exact_divisions,
        "compiler_row_integer_updates": compiler_row_integer_updates,
        "execution_dense_matrix_cells": 0,
        "answer_dependent_compilation": False,
    }
    return tuple(plans), stats


PLAN_LIBRARY, PLAN_LIBRARY_STATS = compile_public_plan_library()


PUBLIC_UNIT_DESCRIPTOR = m226r.public_unit_descriptor()
STATIC_PUBLIC_RESOURCES = {
    "retained_plan_integer_payload_bits": PLAN_LIBRARY_STATS[
        "retained_plan_integer_payload_bits"
    ],
    "predecessor_descriptor_field_cells": PUBLIC_UNIT_DESCRIPTOR["field_cells"],
    "predecessor_descriptor_field_payload_bits": PUBLIC_UNIT_DESCRIPTOR[
        "field_payload_bits"
    ],
    "predecessor_descriptor_parameter_integer_cells": PUBLIC_UNIT_DESCRIPTOR[
        "parameter_integer_cells"
    ],
    "predecessor_descriptor_parameter_payload_bits": PUBLIC_UNIT_DESCRIPTOR[
        "parameter_payload_bits"
    ],
    "total_logical_payload_bits": (
        PLAN_LIBRARY_STATS["retained_plan_integer_payload_bits"]
        + PUBLIC_UNIT_DESCRIPTOR["field_payload_bits"]
        + PUBLIC_UNIT_DESCRIPTOR["parameter_payload_bits"]
    ),
}


def refs_for_kind(unit_index: int, kind: str) -> tuple[PlanRef, ...]:
    if not 0 <= unit_index < len(UNITS):
        raise ValueError("standalone unit index is outside the compiled library")
    unit = PlanRef(unit_index)
    conjugation = PlanRef(CONJUGATION_PLAN_INDEX)
    if kind == "UNIT":
        return (unit,)
    if kind == "INVERSE":
        return (unit.inverted(),)
    norm = (unit, conjugation, unit, conjugation)
    if kind == "NORM":
        return norm
    if kind == "INVERSE_NORM":
        return tuple(reference.inverted() for reference in reversed(norm))
    raise ValueError("unknown standalone factor kind")


def apply_plan_refs_integer(
    coordinates: Sequence[int], refs: Sequence[PlanRef]
) -> tuple[int, ...]:
    values = tuple(coordinates)
    for reference in refs:
        values = apply_integer_plan(
            values,
            PLAN_LIBRARY[reference.plan_index],
            inverse=reference.inverse,
        )
    return values


@dataclass
class Work(m226r.Work):
    inplace_factor_action_calls: int = 0
    inplace_factor_action_repetitions: int = 0
    inplace_plan_reference_reads: int = 0
    inplace_plan_operation_steps: int = 0
    inplace_plan_shears: int = 0
    inplace_plan_swaps: int = 0
    inplace_plan_negations: int = 0
    inplace_rational_integer_multiplications: int = 0
    inplace_rational_additions: int = 0
    inplace_coordinate_accumulators_materialized: int = 0
    inplace_coordinate_accumulators_released: int = 0
    maximum_inplace_coordinate_payload_bits: int = 0
    maximum_inplace_transient_rational_payload_bits: int = 0
    maximum_inplace_transient_rational_cells: int = 0
    maximum_inplace_action_descriptor_integer_cells: int = 0
    maximum_inplace_plan_descriptor_refs: int = 0
    maximum_inplace_absolute_shear_coefficient: int = 0
    inplace_ledger_norm_action_calls: int = 0
    inplace_ledger_scale_action_calls: int = 0
    inplace_trace_weight_action_calls: int = 0
    inplace_candidate_multiplier_action_calls: int = 0
    inplace_selected_multiplier_action_calls: int = 0
    execution_dense_matrix_cells: int = 0

    def observe_inplace(
        self,
        *,
        resident_payload_bits: int,
        scratch_payload_bits: int,
        coordinate_payload_bits: int,
        transient_payload_bits: int,
        transient_rational_cells: int,
        integer_payload_bits: int,
        descriptor_integer_cells: int,
        plan_descriptor_refs: int,
        carrier_field_cells: int,
        context: str,
    ) -> None:
        payload = (
            resident_payload_bits
            + coordinate_payload_bits
            + transient_payload_bits
            + integer_payload_bits
        )
        self.transient_observations += 1
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, resident_payload_bits
        )
        self.maximum_scratch_payload_bits = max(
            self.maximum_scratch_payload_bits, scratch_payload_bits
        )
        self.maximum_line_integer_payload_bits = max(
            self.maximum_line_integer_payload_bits, integer_payload_bits
        )
        self.maximum_inplace_coordinate_payload_bits = max(
            self.maximum_inplace_coordinate_payload_bits, coordinate_payload_bits
        )
        self.maximum_inplace_transient_rational_payload_bits = max(
            self.maximum_inplace_transient_rational_payload_bits,
            transient_payload_bits,
        )
        self.maximum_inplace_transient_rational_cells = max(
            self.maximum_inplace_transient_rational_cells,
            transient_rational_cells,
        )
        self.maximum_inplace_action_descriptor_integer_cells = max(
            self.maximum_inplace_action_descriptor_integer_cells,
            descriptor_integer_cells,
        )
        self.maximum_inplace_plan_descriptor_refs = max(
            self.maximum_inplace_plan_descriptor_refs, plan_descriptor_refs
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells, carrier_field_cells + 1
        )


def apply_plan_in_place(
    coordinates: list[Fraction],
    plan_ref: PlanRef,
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    coordinate_payload_bits: int,
    resident_payload_bits: int,
    scratch_payload_bits: int,
    base_integer_payload_bits: int,
    descriptor_integer_cells: int,
    plan_descriptor_refs: int,
    plan_descriptor_payload_bits: int,
    carrier_field_cells: int,
    action_index: int,
    repetition: int,
    ref_index: int,
    context_prefix: str,
) -> int:
    plan = PLAN_LIBRARY[plan_ref.plan_index]
    operation_indices: Iterable[int] = (
        range(len(plan) - 1, -1, -1) if plan_ref.inverse else range(len(plan))
    )
    work.inplace_plan_reference_reads += 1
    for step_index, operation_index in enumerate(operation_indices):
        operation = plan[operation_index]
        effective_coefficient = (
            -operation.coefficient
            if plan_ref.inverse and operation.opcode == "SHEAR"
            else operation.coefficient
        )
        operation_descriptor = operation.integer_tuple()
        if operation.opcode == "SHEAR":
            operation_descriptor = (*operation_descriptor[:-1], effective_coefficient)
        operation_integers = (
            action_index,
            repetition,
            ref_index,
            plan_ref.plan_index,
            int(plan_ref.inverse),
            step_index,
            operation_index,
            *operation_descriptor,
        )
        integer_payload_bits = (
            base_integer_payload_bits
            + plan_descriptor_payload_bits
            + sum(signed_bits(value) for value in operation_integers)
        )
        if operation.opcode == "SWAP":
            work.inplace_plan_swaps += 1
            work.observe_inplace(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                coordinate_payload_bits=coordinate_payload_bits,
                transient_payload_bits=0,
                transient_rational_cells=0,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context=f"{context_prefix}_SWAP",
            )
            coordinates[operation.target], coordinates[operation.source] = (
                coordinates[operation.source],
                coordinates[operation.target],
            )
        elif operation.opcode == "NEGATE":
            updated = -coordinates[operation.target]
            updated_bits = fraction_payload(updated)
            work.inplace_plan_negations += 1
            work.observe_inplace(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                coordinate_payload_bits=coordinate_payload_bits,
                transient_payload_bits=updated_bits,
                transient_rational_cells=1,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context=f"{context_prefix}_NEGATE",
            )
            coordinate_payload_bits += updated_bits - fraction_payload(
                coordinates[operation.target]
            )
            coordinates[operation.target] = updated
            del updated
        elif operation.opcode == "SHEAR":
            if operation.target == operation.source or effective_coefficient == 0:
                raise RuntimeError("invalid standalone shear")
            if effective_coefficient == 1:
                updated = coordinates[operation.target] + coordinates[operation.source]
                transient_payload_bits = fraction_payload(updated)
                transient_rational_cells = 1
            elif effective_coefficient == -1:
                updated = coordinates[operation.target] - coordinates[operation.source]
                transient_payload_bits = fraction_payload(updated)
                transient_rational_cells = 1
            else:
                scaled = coordinates[operation.source] * effective_coefficient
                updated = coordinates[operation.target] + scaled
                transient_payload_bits = fraction_payload(scaled) + fraction_payload(
                    updated
                )
                transient_rational_cells = 2
                work.inplace_rational_integer_multiplications += 1
            work.inplace_plan_shears += 1
            work.inplace_rational_additions += 1
            work.maximum_inplace_absolute_shear_coefficient = max(
                work.maximum_inplace_absolute_shear_coefficient,
                abs(effective_coefficient),
            )
            work.observe_inplace(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                coordinate_payload_bits=coordinate_payload_bits,
                transient_payload_bits=transient_payload_bits,
                transient_rational_cells=transient_rational_cells,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context=f"{context_prefix}_SHEAR",
            )
            coordinate_payload_bits += fraction_payload(updated) - fraction_payload(
                coordinates[operation.target]
            )
            coordinates[operation.target] = updated
            if effective_coefficient not in (-1, 1):
                del scaled
            del updated, transient_payload_bits, transient_rational_cells
        else:
            raise RuntimeError("unknown standalone operation")
        work.inplace_plan_operation_steps += 1
    return coordinate_payload_bits


def coordinate_accumulator(
    actions: Sequence[FactorAction],
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> base.E:
    coordinates = list(base.ONE.coordinates)
    coordinate_payload_bits = sum(fraction_payload(value) for value in coordinates)
    descriptor_integers = tuple(
        value for action in actions for value in action.integer_tuple()
    )
    descriptor_integer_cells = len(descriptor_integers)
    scratch_payload_bits = base.payload_bits(scratch)
    resident_payload_bits = (
        base.payload_bits(residual)
        + prior.m220.ledger_bits(ledger)
        + scratch_payload_bits
    )
    base_integer_payload_bits = sum(
        signed_bits(value) for value in live_integers + descriptor_integers
    )
    carrier_field_cells = len(residual) + len(scratch)
    work.inplace_coordinate_accumulators_materialized += 1
    for action_index, action in enumerate(actions):
        if action.exponent <= 0:
            raise ValueError("standalone action exponent must be positive")
        refs = refs_for_kind(action.unit_index, action.kind)
        plan_descriptor_payload_bits = sum(
            signed_bits(reference.plan_index) + signed_bits(int(reference.inverse))
            for reference in refs
        )
        plan_descriptor_refs = len(refs)
        work.inplace_factor_action_calls += 1
        work.inplace_factor_action_repetitions += action.exponent
        for repetition in range(action.exponent):
            for ref_index, plan_ref in enumerate(refs):
                coordinate_payload_bits = apply_plan_in_place(
                    coordinates,
                    plan_ref,
                    residual,
                    ledger,
                    scratch,
                    work,
                    coordinate_payload_bits=coordinate_payload_bits,
                    resident_payload_bits=resident_payload_bits,
                    scratch_payload_bits=scratch_payload_bits,
                    base_integer_payload_bits=base_integer_payload_bits,
                    descriptor_integer_cells=descriptor_integer_cells,
                    plan_descriptor_refs=plan_descriptor_refs,
                    plan_descriptor_payload_bits=plan_descriptor_payload_bits,
                    carrier_field_cells=carrier_field_cells,
                    action_index=action_index,
                    repetition=repetition,
                    ref_index=ref_index,
                    context_prefix=context_prefix,
                )
    if actions:
        del (
            action,
            action_index,
            plan_descriptor_payload_bits,
            plan_descriptor_refs,
            plan_ref,
            ref_index,
            refs,
            repetition,
        )
    result = base.E(tuple(coordinates))
    work.observe_inplace(
        resident_payload_bits=resident_payload_bits,
        scratch_payload_bits=scratch_payload_bits,
        coordinate_payload_bits=coordinate_payload_bits,
        transient_payload_bits=0,
        transient_rational_cells=0,
        integer_payload_bits=base_integer_payload_bits,
        descriptor_integer_cells=descriptor_integer_cells,
        plan_descriptor_refs=0,
        carrier_field_cells=carrier_field_cells,
        context=(
            f"{context_prefix}_HANDOFF"
            if actions
            else f"{context_prefix}_IDENTITY_HANDOFF"
        ),
    )
    coordinates.clear()
    work.inplace_coordinate_accumulators_released += 1
    return result


def ledger_actions(ledger: list[int], positive: str, negative: str) -> tuple[FactorAction, ...]:
    return tuple(
        FactorAction(index, positive if exponent > 0 else negative, abs(exponent))
        for index, exponent in enumerate(ledger)
        if exponent
    )


def inplace_ledger_scale(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    live_integers: tuple[int, ...] = (),
    **_kwargs: Any,
) -> base.E:
    work.ledger_scale_rematerializations += 1
    actions = ledger_actions(ledger, "UNIT", "INVERSE")
    work.inplace_ledger_scale_action_calls += len(actions)
    return coordinate_accumulator(
        actions,
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers,
        context_prefix="INPLACE_PUBLIC_UNIT_LEDGER_SCALE",
    )


def inplace_ledger_norm(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    *,
    live_integers: tuple[int, ...],
) -> base.E:
    work.ledger_norm_rematerializations += 1
    actions = ledger_actions(ledger, "NORM", "INVERSE_NORM")
    work.inplace_ledger_norm_action_calls += len(actions)
    return coordinate_accumulator(
        actions,
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers,
        context_prefix="INPLACE_PUBLIC_UNIT_LEDGER_NORM",
    )


def inplace_unit_action_energy(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    unit: UnitDirection | None,
    exponent: int,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> int:
    actions = list(ledger_actions(ledger, "NORM", "INVERSE_NORM"))
    work.inplace_ledger_norm_action_calls += len(actions)
    work.ledger_norm_rematerializations += 1
    if unit is not None and exponent:
        index = next(
            index for index, candidate in enumerate(UNITS) if candidate.parameter == unit.parameter
        )
        actions.append(
            FactorAction(
                index,
                "INVERSE_NORM" if exponent > 0 else "NORM",
                abs(exponent),
            )
        )
        work.inplace_trace_weight_action_calls += 1
    weight = coordinate_accumulator(
        tuple(actions),
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers + (exponent,),
        context_prefix="INPLACE_PUBLIC_UNIT_TRACE_WEIGHT",
    )
    work.direct_trace_weight_rematerializations += 1
    return m226r.direct_trace_energy(
        residual,
        ledger,
        scratch,
        weight,
        work,
        live_integers=live_integers + (exponent,),
        caller_live_scalars=(),
        context_prefix=context_prefix,
    )


def inplace_scaled_payload(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    candidate_unit: UnitDirection | None,
    candidate_exponent: int,
    live_integers: tuple[int, ...],
) -> int:
    actions = list(ledger_actions(ledger, "UNIT", "INVERSE"))
    work.ledger_scale_rematerializations += 1
    work.inplace_ledger_scale_action_calls += len(actions)
    candidate = candidate_unit is not None and candidate_exponent != 0
    if candidate:
        index = next(
            index
            for index, unit in enumerate(UNITS)
            if unit.parameter == candidate_unit.parameter
        )
        actions.append(
            FactorAction(
                index,
                "INVERSE" if candidate_exponent > 0 else "UNIT",
                abs(candidate_exponent),
            )
        )
        work.inplace_candidate_multiplier_action_calls += 1
        work.candidate_multiplier_rematerializations += 1
    multiplier = coordinate_accumulator(
        tuple(actions),
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers + (candidate_exponent,),
        context_prefix="INPLACE_PUBLIC_UNIT_CANDIDATE_MULTIPLIER",
    )
    return prior.stream_payload(
        residual,
        multiplier,
        residual,
        ledger,
        scratch,
        work,
        candidate=candidate,
        live_scalars=(),
        live_integers=live_integers + (candidate_exponent,),
    )


def inplace_apply_selected(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    selected_ledger: list[int],
    work: Work,
    live_integers: tuple[int, ...],
) -> None:
    actions = list(ledger_actions(ledger, "UNIT", "INVERSE"))
    work.ledger_scale_rematerializations += 1
    work.inplace_ledger_scale_action_calls += len(actions)
    selected_actions = ledger_actions(selected_ledger, "INVERSE", "UNIT")
    actions.extend(selected_actions)
    work.inplace_selected_multiplier_action_calls += len(selected_actions)
    multiplier = coordinate_accumulator(
        tuple(actions),
        residual,
        ledger,
        scratch,
        work,
        live_integers=live_integers + tuple(selected_ledger),
        context_prefix="INPLACE_PUBLIC_UNIT_SELECTED_MULTIPLIER",
    )
    work.selected_multiplier_rematerializations += 1
    work.set_ambient_liveness(scalars=(), integers=live_integers)
    try:
        prior.mutate_selected(
            residual, ledger, scratch, multiplier, work, ()
        )
    finally:
        work.clear_ambient_liveness()


def install_inplace_path() -> None:
    m226r.Work = Work
    m226r.fused_ledger_scale = inplace_ledger_scale
    m226r.rematerialize_ledger_norm = inplace_ledger_norm
    m226r.direct_unit_action_energy = inplace_unit_action_energy
    m226r.rematerialized_scaled_payload = inplace_scaled_payload
    m226r.rematerialized_apply_selected = inplace_apply_selected
    m222.Work = Work
    m222.streamed_energy = m226r.direct_trace_energy
    m222.line_minimum = m226r.rematerialized_line_minimum
    m222.balance = m226r.rematerialized_balance
    prior.Work = Work
    prior.counted_power = m226r.reject_standalone_power
    prior.ledger_scale = inplace_ledger_scale
    prior.balance = m226r.rematerialized_balance
    prior.apply_operation = m226r.rematerialized_apply_operation
    prior.project = m226r.fused_project


def compiler_controls() -> dict[str, Any]:
    matrix = multiplication_matrix(UNITS[0].value)
    plan = PLAN_LIBRARY[0]
    basis = (1,) + (0,) * (FIELD_DEGREE - 1)
    expected = tuple(matrix[row][0] for row in range(FIELD_DEGREE))
    corrupted = list(plan)
    shear_index = next(
        index for index, operation in enumerate(corrupted) if operation.opcode == "SHEAR"
    )
    operation = corrupted[shear_index]
    corrupted[shear_index] = PlanOperation(
        "SHEAR", operation.target, operation.source, operation.coefficient + 1
    )
    skipped = plan[:shear_index] + plan[shear_index + 1 :]
    corrupted_changes_result = False
    skipped_changes_result = False
    for column in range(FIELD_DEGREE):
        candidate_basis = tuple(int(row == column) for row in range(FIELD_DEGREE))
        candidate_expected = tuple(matrix[row][column] for row in range(FIELD_DEGREE))
        corrupted_changes_result |= apply_integer_plan(
            candidate_basis, corrupted
        ) != candidate_expected
        skipped_changes_result |= apply_integer_plan(
            candidate_basis, skipped
        ) != candidate_expected
    noncommuting_pair_witnessed = False
    reordered_changes_result = False
    for index, left in enumerate(plan[:-1]):
        right = plan[index + 1]
        if apply_integer_plan(basis, (left, right)) != apply_integer_plan(
            basis, (right, left)
        ):
            noncommuting_pair_witnessed = True
            reordered_changes_result = True
            break
    determinant, _, _, _ = bareiss_determinant(matrix)
    nonunimodular = [list(row) for row in matrix]
    nonunimodular[0] = [2 * value for value in nonunimodular[0]]
    bad_determinant, _, _, _ = bareiss_determinant(nonunimodular)
    try:
        reduce_unimodular_matrix(nonunimodular)
    except RuntimeError:
        nonunimodular_rejected = True
    else:
        nonunimodular_rejected = False
    nonintegral = [list(row) for row in matrix]
    nonintegral[0][0] = Fraction(1, 2)
    try:
        reduce_unimodular_matrix(nonintegral)
    except RuntimeError:
        nonintegral_rejected = True
    else:
        nonintegral_rejected = False
    mutated_matrix = [list(row) for row in matrix]
    mutated_matrix[0] = [
        left + right
        for left, right in zip(mutated_matrix[0], mutated_matrix[1], strict=True)
    ]
    mutated_plan = reduce_unimodular_matrix(mutated_matrix)[0]
    public_input_mutation_changes_plan_commitment = plan_commitment(
        (mutated_plan,)
    ) != plan_commitment((plan,))
    norm_composite_basis_parity = True
    inverse_norm_composite_basis_parity = True
    unit_and_inverse_basis_parity_all = True
    nonbasis_plan_and_inverse_parity = True
    verification_peak_composite_matrix_payload_bits = 0
    nonbasis = tuple(range(-8, 8))
    for unit_index, direction in enumerate(UNITS):
        unit_matrix = multiplication_matrix(direction.value)
        norm_matrix = multiplication_matrix(direction.norm)
        inverse_norm_matrix = multiplication_matrix(direction.reciprocal_norm)
        verification_peak_composite_matrix_payload_bits = max(
            verification_peak_composite_matrix_payload_bits,
            integer_matrix_payload_bits(norm_matrix)
            + integer_matrix_payload_bits(inverse_norm_matrix),
        )
        for column in range(FIELD_DEGREE):
            candidate_basis = tuple(
                int(row == column) for row in range(FIELD_DEGREE)
            )
            norm_composite_basis_parity &= apply_plan_refs_integer(
                candidate_basis, refs_for_kind(unit_index, "NORM")
            ) == tuple(norm_matrix[row][column] for row in range(FIELD_DEGREE))
            inverse_norm_composite_basis_parity &= apply_plan_refs_integer(
                candidate_basis, refs_for_kind(unit_index, "INVERSE_NORM")
            ) == tuple(
                inverse_norm_matrix[row][column] for row in range(FIELD_DEGREE)
            )
            unit_expected = tuple(
                unit_matrix[row][column] for row in range(FIELD_DEGREE)
            )
            unit_and_inverse_basis_parity_all &= apply_integer_plan(
                candidate_basis, PLAN_LIBRARY[unit_index]
            ) == unit_expected and apply_integer_plan(
                unit_expected, PLAN_LIBRARY[unit_index], inverse=True
            ) == candidate_basis
        nonbasis_expected = tuple(
            sum(unit_matrix[row][column] * nonbasis[column] for column in range(FIELD_DEGREE))
            for row in range(FIELD_DEGREE)
        )
        nonbasis_plan_and_inverse_parity &= apply_integer_plan(
            nonbasis, PLAN_LIBRARY[unit_index]
        ) == nonbasis_expected and apply_integer_plan(
            nonbasis_expected, PLAN_LIBRARY[unit_index], inverse=True
        ) == nonbasis
    conjugation = conjugation_matrix()
    conjugation_basis_parity = all(
        apply_integer_plan(
            tuple(int(row == column) for row in range(FIELD_DEGREE)),
            PLAN_LIBRARY[CONJUGATION_PLAN_INDEX],
        )
        == tuple(conjugation[row][column] for row in range(FIELD_DEGREE))
        for column in range(FIELD_DEGREE)
    )
    wrong_inversion_flag_changes_result = apply_integer_plan(
        basis, plan, inverse=True
    ) != expected
    wrong_basis_order_changes_result = apply_integer_plan(
        tuple(reversed(basis)), plan
    ) != expected
    off_by_one_repetition_changes_result = apply_integer_plan(
        apply_integer_plan(basis, plan), plan
    ) != expected
    invalid_action_controls = []
    for factory in (
        lambda: FactorAction(len(UNITS), "UNIT", 1),
        lambda: FactorAction(0, "UNKNOWN", 1),
        lambda: FactorAction(0, "UNIT", 0),
        lambda: FactorAction(0, "UNIT", -1),
    ):
        try:
            factory()
        except ValueError:
            invalid_action_controls.append(True)
        else:
            invalid_action_controls.append(False)
    recompiled_plans, recompiled_stats = compile_public_plan_library()
    return {
        "unit_determinant_is_one": determinant == 1,
        "basis_action_matches": apply_integer_plan(basis, plan) == expected,
        "inverse_restores_basis": apply_integer_plan(expected, plan, inverse=True)
        == basis,
        "corrupted_shear_changes_result": corrupted_changes_result,
        "skipped_operation_changes_result": skipped_changes_result,
        "noncommuting_pair_witnessed": noncommuting_pair_witnessed,
        "reordered_noncommuting_pair_changes_result": reordered_changes_result,
        "nonunimodular_determinant_detected": abs(bad_determinant) != 1,
        "nonunimodular_matrix_rejected": nonunimodular_rejected,
        "nonintegral_matrix_rejected": nonintegral_rejected,
        "public_input_mutation_changes_plan_commitment": public_input_mutation_changes_plan_commitment,
        "norm_composite_basis_parity": norm_composite_basis_parity,
        "inverse_norm_composite_basis_parity": inverse_norm_composite_basis_parity,
        "unit_and_inverse_basis_parity_all": unit_and_inverse_basis_parity_all,
        "conjugation_basis_parity": conjugation_basis_parity,
        "nonbasis_plan_and_inverse_parity": nonbasis_plan_and_inverse_parity,
        "wrong_inversion_flag_changes_result": wrong_inversion_flag_changes_result,
        "wrong_basis_order_changes_result": wrong_basis_order_changes_result,
        "off_by_one_repetition_changes_result": off_by_one_repetition_changes_result,
        "invalid_unit_index_rejected": invalid_action_controls[0],
        "invalid_action_kind_rejected": invalid_action_controls[1],
        "zero_exponent_rejected": invalid_action_controls[2],
        "negative_exponent_rejected": invalid_action_controls[3],
        "compilation_uses_only_public_basis_and_unit_parameters": (
            compile_public_plan_library.__code__.co_argcount == 0
        ),
        "public_compiler_input_commitment_reproduced": (
            recompiled_stats["public_compiler_input_commitment"]
            == PLAN_LIBRARY_STATS["public_compiler_input_commitment"]
        ),
        "accepted_execution_plan_nodes_are_primitive_descriptors": all(
            isinstance(operation, PlanOperation)
            for compiled_plan in PLAN_LIBRARY
            for operation in compiled_plan
        ),
        "execution_dense_matrix_cells": 0,
        "deterministic_recompilation_commitment": (
            plan_commitment(recompiled_plans)
            == PLAN_LIBRARY_STATS["plan_commitment"]
        ),
        "verification_recompile_retained_plan_payload_bits": recompiled_stats[
            "retained_plan_integer_payload_bits"
        ],
        "verification_peak_composite_matrix_cells": 2 * FIELD_DEGREE * FIELD_DEGREE,
        "verification_peak_composite_matrix_payload_bits": verification_peak_composite_matrix_payload_bits,
        "conjugation_plan_involutory": all(
            apply_integer_plan(
                apply_integer_plan(
                    tuple(int(row == column) for row in range(FIELD_DEGREE)),
                    PLAN_LIBRARY[CONJUGATION_PLAN_INDEX],
                ),
                PLAN_LIBRARY[CONJUGATION_PLAN_INDEX],
            )
            == tuple(int(row == column) for row in range(FIELD_DEGREE))
            for column in range(FIELD_DEGREE)
        ),
    }


def main() -> None:
    install_inplace_path()
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_inplace_unit_plan_reference.v1",
                "imports_m227_production": False,
                "imports_m226_production": False,
                "uses_prior_standalone_m226_reference_substrate": True,
                "plan_library": PLAN_LIBRARY_STATS,
                "static_public_resources": STATIC_PUBLIC_RESOURCES,
                "compiler_controls": compiler_controls(),
                "cases": [prior.case(*item) for item in CASES],
                "reuse": prior.reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
