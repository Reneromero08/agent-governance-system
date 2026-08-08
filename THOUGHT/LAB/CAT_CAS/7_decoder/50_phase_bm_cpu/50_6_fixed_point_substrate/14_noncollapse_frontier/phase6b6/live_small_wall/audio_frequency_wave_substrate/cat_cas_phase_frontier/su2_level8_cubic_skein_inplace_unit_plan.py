#!/usr/bin/env python3
"""M227 in-place GL(16,Z) public-unit coordinate actions.

M226 removes binary powers but keeps a whole old cyclotomic accumulator beside
its whole product.  M227 compiles the seven public unit multiplications and
cyclotomic conjugation into exact swaps, signs, and integer shears on the
sixteen rational coordinates.  One owned coordinate vector is updated in
place; norms are composed from unit and conjugation plan references rather
than decomposed as dense matrices.

The compiler and retained plans are public, answer-independent resources.  The
identical compiler and coordinate recurrence remain the strongest classical
baseline, so this experiment can establish only a bounded liveness repair or
another exact-height/resource obstruction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import su2_level8_cubic_skein_direct_unit_action as m226


sys.set_int_max_str_digits(0)
m222 = m226.m222
m221 = m226.m221
braid = m226.braid
CASES = m226.CASES
FIELD_DEGREE = 16
CONJUGATION_PLAN_INDEX = len(m222.UNITS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return m221.signed_bits(value)


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
        if not 0 <= self.unit_index < len(m222.UNITS):
            raise ValueError("public unit index is outside the compiled library")
        if self.kind not in {"UNIT", "INVERSE", "NORM", "INVERSE_NORM"}:
            raise ValueError("unknown public factor kind")
        if self.exponent <= 0:
            raise ValueError("public factor exponent must be positive")

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
        raise RuntimeError("public action compiler requires a 16x16 integer matrix")


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
                    raise RuntimeError("nonexact Bareiss division")
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
            raise RuntimeError("public action matrix is not unimodular")
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
        raise RuntimeError("unimodular reduction did not reach identity")
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
            values[operation.target] += (
                operation.coefficient * values[operation.source]
            )
        else:
            raise RuntimeError("unknown public plan opcode")
    return tuple(values)


def multiplication_matrix(value: braid.K) -> tuple[tuple[int, ...], ...]:
    columns = [
        (value * braid.K.zeta(index)).coefficients
        for index in range(FIELD_DEGREE)
    ]
    if any(
        coefficient.denominator != 1
        for column in columns
        for coefficient in column
    ):
        raise RuntimeError("public unit action left the integral coordinate ring")
    return tuple(
        tuple(columns[column][row].numerator for column in range(FIELD_DEGREE))
        for row in range(FIELD_DEGREE)
    )


def conjugation_matrix() -> tuple[tuple[int, ...], ...]:
    columns = [
        m221.m220.conjugate(braid.K.zeta(index)).coefficients
        for index in range(FIELD_DEGREE)
    ]
    if any(
        coefficient.denominator != 1
        for column in columns
        for coefficient in column
    ):
        raise RuntimeError("conjugation action left the integral coordinate ring")
    return tuple(
        tuple(columns[column][row].numerator for column in range(FIELD_DEGREE))
        for row in range(FIELD_DEGREE)
    )


def plan_payload(plan: Sequence[PlanOperation]) -> tuple[int, int]:
    integers = [value for operation in plan for value in operation.integer_tuple()]
    return len(integers), sum(signed_bits(value) for value in integers)


def plan_commitment(plans: Sequence[Sequence[PlanOperation]]) -> str:
    serial = [
        [operation.integer_tuple() for operation in plan]
        for plan in plans
    ]
    return hashlib.sha256(
        json.dumps(serial, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def compile_public_plan_library() -> tuple[
    tuple[tuple[PlanOperation, ...], ...], dict[str, Any]
]:
    public_actions = tuple(direction.unit for direction in m222.UNITS) + (None,)
    plans: list[tuple[PlanOperation, ...]] = []
    determinants = []
    determinant_multiply_subtracts = 0
    determinant_exact_divisions = 0
    compiler_row_integer_updates = 0
    compiler_peak_dense_integer_payload_bits = 0
    compiler_peak_declared_exact_payload_bits = 0
    compiler_total_input_matrix_integer_payload_bits = 0
    retained_emitted_plan_payload_bits = 0
    compiler_public_descriptor_payload_bits = sum(
        m221.scalar_payload(value)
        for direction in m222.UNITS
        for value in (
            direction.unit,
            direction.inverse,
            direction.norm,
            direction.inverse_norm,
        )
    ) + sum(signed_bits(direction.parameter) for direction in m222.UNITS)
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
            raise RuntimeError("public coordinate action determinant is not a unit")
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
                raise RuntimeError("compiled public plan failed basis parity")
            if apply_integer_plan(expected, plan, inverse=True) != basis:
                raise RuntimeError("compiled public plan inverse failed")
        plans.append(plan)
        retained_emitted_plan_payload_bits += plan_payload(plan)[1]
        del matrix, plan
    if apply_integer_plan(
        apply_integer_plan((1,) + (0,) * 15, plans[CONJUGATION_PLAN_INDEX]),
        plans[CONJUGATION_PLAN_INDEX],
    ) != (1,) + (0,) * 15:
        raise RuntimeError("conjugation plan is not involutory")
    operation_counts = {
        opcode: sum(
            operation.opcode == opcode for plan in plans for operation in plan
        )
        for opcode in ("SWAP", "NEGATE", "SHEAR")
    }
    integer_cells, integer_payload_bits = zip(
        *(plan_payload(plan) for plan in plans), strict=True
    )
    serialized = json.dumps(
        [[operation.integer_tuple() for operation in plan] for plan in plans],
        separators=(",", ":"),
    ).encode("ascii")
    stats = {
        "primitive_plan_count": len(plans),
        "unit_plan_count": len(m222.UNITS),
        "conjugation_plan_count": 1,
        "plan_lengths": [len(plan) for plan in plans],
        "operation_counts": operation_counts,
        "maximum_absolute_shear_coefficient": max(
            abs(operation.coefficient)
            for plan in plans
            for operation in plan
            if operation.opcode == "SHEAR"
        ),
        "retained_plan_integer_cells": sum(integer_cells),
        "retained_plan_integer_payload_bits": sum(integer_payload_bits),
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
        "compiler_input_matrix_field_multiplications": len(m222.UNITS)
        * FIELD_DEGREE,
        "compiler_determinant_multiply_subtracts": determinant_multiply_subtracts,
        "compiler_determinant_exact_divisions": determinant_exact_divisions,
        "compiler_row_integer_updates": compiler_row_integer_updates,
        "execution_dense_matrix_cells": 0,
        "answer_dependent_compilation": False,
    }
    return tuple(plans), stats


PLAN_LIBRARY, PLAN_LIBRARY_STATS = compile_public_plan_library()


def public_unit_descriptor() -> dict[str, int]:
    field_values = tuple(
        value
        for direction in m222.UNITS
        for value in (
            direction.unit,
            direction.inverse,
            direction.norm,
            direction.inverse_norm,
        )
    )
    parameters = tuple(direction.parameter for direction in m222.UNITS)
    return {
        "field_cells": len(field_values),
        "field_payload_bits": sum(m221.scalar_payload(value) for value in field_values),
        "parameter_integer_cells": len(parameters),
        "parameter_payload_bits": sum(signed_bits(value) for value in parameters),
    }


PUBLIC_UNIT_DESCRIPTOR = public_unit_descriptor()
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
    if not 0 <= unit_index < len(m222.UNITS):
        raise ValueError("public unit index is outside the compiled library")
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
    raise ValueError("unknown public factor kind")


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
class Work(m226.Work):
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
            self.maximum_declared_live_field_cells,
            carrier_field_cells + 1,
        )


def apply_plan_in_place(
    coordinates: list[Fraction],
    plan_ref: PlanRef,
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
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
                raise RuntimeError("invalid public shear operation")
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
            raise RuntimeError("unknown public plan operation")
        work.inplace_plan_operation_steps += 1
    return coordinate_payload_bits


def coordinate_accumulator(
    actions: Sequence[FactorAction],
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> braid.K:
    coordinates = list(braid.ONE.coefficients)
    coordinate_payload_bits = sum(fraction_payload(value) for value in coordinates)
    descriptor_integers = tuple(
        value for action in actions for value in action.integer_tuple()
    )
    descriptor_integer_cells = len(descriptor_integers)
    scratch_payload_bits = braid.field_payload_bits(scratch)
    resident_payload_bits = (
        braid.field_payload_bits(residual)
        + m221.m220.ledger_payload_bits(ledger)
        + scratch_payload_bits
    )
    base_integer_payload_bits = sum(
        signed_bits(value) for value in live_integers + descriptor_integers
    )
    carrier_field_cells = len(residual) + len(scratch)
    work.inplace_coordinate_accumulators_materialized += 1
    for action_index, action in enumerate(actions):
        if action.exponent <= 0:
            raise ValueError("in-place action exponent must be positive")
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
    result = braid.K(tuple(coordinates))
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    live_scalars: tuple[braid.K, ...] = (),
    live_integers: tuple[int, ...] = (),
) -> braid.K:
    if live_scalars:
        raise RuntimeError("M227 ledger scale requires released caller field scalars")
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    live_integers: tuple[int, ...],
) -> braid.K:
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    direction: m221.m220.UnitDirection | None,
    exponent: int,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> int:
    actions = list(ledger_actions(ledger, "NORM", "INVERSE_NORM"))
    work.inplace_ledger_norm_action_calls += len(actions)
    work.ledger_norm_rematerializations += 1
    if direction is not None and exponent:
        index = next(
            index
            for index, candidate in enumerate(m222.UNITS)
            if candidate.parameter == direction.parameter
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
    return m226.direct_trace_energy(
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    candidate_direction: m221.m220.UnitDirection | None,
    candidate_exponent: int,
    live_integers: tuple[int, ...],
) -> int:
    actions = list(ledger_actions(ledger, "UNIT", "INVERSE"))
    work.ledger_scale_rematerializations += 1
    work.inplace_ledger_scale_action_calls += len(actions)
    candidate = candidate_direction is not None and candidate_exponent != 0
    if candidate:
        index = next(
            index
            for index, direction in enumerate(m222.UNITS)
            if direction.parameter == candidate_direction.parameter
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
    return m221.stream_scaled_payload(
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
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    selected_ledger: list[int],
    work: Work,
    *,
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
        m221.apply_selected_net(
            residual,
            ledger,
            scratch,
            multiplier,
            work,
            live_scalars=(),
        )
    finally:
        work.clear_ambient_liveness()


def install_inplace_path() -> None:
    m226.Work = Work
    m226.fused_ledger_scale = inplace_ledger_scale
    m226.rematerialize_ledger_norm = inplace_ledger_norm
    m226.direct_unit_action_energy = inplace_unit_action_energy
    m226.rematerialized_scaled_payload = inplace_scaled_payload
    m226.rematerialized_apply_selected = inplace_apply_selected
    m226.install_direct_trace_path()


def compiler_controls() -> dict[str, Any]:
    unit = m222.UNITS[0].unit
    matrix = multiplication_matrix(unit)
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
        candidate_basis = tuple(
            int(row == column) for row in range(FIELD_DEGREE)
        )
        candidate_expected = tuple(
            matrix[row][column] for row in range(FIELD_DEGREE)
        )
        corrupted_changes_result |= (
            apply_integer_plan(candidate_basis, corrupted) != candidate_expected
        )
        skipped_changes_result |= (
            apply_integer_plan(candidate_basis, skipped) != candidate_expected
        )
    noncommuting_pair_witnessed = False
    reordered_changes_result = False
    for index, left in enumerate(plan[:-1]):
        right = plan[index + 1]
        if apply_integer_plan(
            basis, (left, right)
        ) != apply_integer_plan(basis, (right, left)):
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
    for unit_index, direction in enumerate(m222.UNITS):
        unit_matrix = multiplication_matrix(direction.unit)
        norm_matrix = multiplication_matrix(direction.norm)
        inverse_norm_matrix = multiplication_matrix(direction.inverse_norm)
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
        lambda: FactorAction(len(m222.UNITS), "UNIT", 1),
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


def reference_case_view(case: dict[str, Any]) -> dict[str, Any]:
    return m221.reference_case_view(case)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_inplace_unit_plan.py "
            "SEPARATE_REFERENCE_JSON"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M227 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_cubic_skein_inplace_unit_plan_reference.v1":
        raise RuntimeError("M227 separate-reference schema changed")
    if reference.get("plan_library") != PLAN_LIBRARY_STATS:
        raise RuntimeError("M227 independent public plan-library parity failed")
    if reference.get("static_public_resources") != STATIC_PUBLIC_RESOURCES:
        raise RuntimeError("M227 independent static public-resource parity failed")
    controls = compiler_controls()
    if controls != reference.get("compiler_controls"):
        raise RuntimeError("M227 independent compiler control parity failed")
    install_inplace_path()
    cases = [m221.execute_case(*case) for case in CASES]
    if [reference_case_view(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M227 independent case/resource parity failed")
    reuse = m221.reuse_result()
    reference_reuse = reference.get("reuse")
    for section in ("primary", "reuse", "fresh_reuse"):
        if reference_case_view(reuse[section]) != reference_reuse[section]:
            raise RuntimeError(f"M227 independent reuse parity failed: {section}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse[key] != reference_reuse[key]:
            raise RuntimeError(f"M227 independent reuse parity failed: {key}")
    predecessor_path = here / "SU2_LEVEL8_CUBIC_SKEIN_DIRECT_UNIT_ACTION_RESULTS.json"
    predecessor = json.loads(predecessor_path.read_text())
    semantic_keys = (
        "strands",
        "rounds",
        "family",
        "boundary_commitment",
        "forward_state_commitment",
        "forward_raw_payload_bits",
        "final_balance",
        "same_residual_backing",
        "same_unit_ledger_backing",
        "same_scratch_backing",
        "restoration_error_field_cells",
        "canonical_post_restoration_state_exact",
        "baseline_reload_used",
        "restoration_generation",
        "matched_raw_recurrence",
    )
    for case, old_case in zip(cases, predecessor["cases"], strict=True):
        for key in semantic_keys:
            if case[key] != old_case[key]:
                raise RuntimeError(f"M227 predecessor semantic parity failed: {key}")
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in semantic_keys[3:]:
            if reuse[section][key] != predecessor["reuse"][section][key]:
                raise RuntimeError(
                    f"M227 predecessor reuse semantic parity failed: {section}.{key}"
                )
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse[key] != predecessor["reuse"][key]:
            raise RuntimeError(f"M227 predecessor reuse parity failed: {key}")
    primary = next(
        case
        for case in cases
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    old_primary = next(
        case
        for case in predecessor["cases"]
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    primary_bits = primary["work"]["maximum_declared_live_payload_bits"]
    old_primary_bits = old_primary["work"]["maximum_declared_live_payload_bits"]
    raw_bits = primary["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
    inclusive_bits = primary_bits + STATIC_PUBLIC_RESOURCES["total_logical_payload_bits"]
    every_dynamic_below_raw = all(
        case["work"]["maximum_declared_live_payload_bits"]
        < case["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
        for case in cases
    )
    every_inclusive_below_raw = all(
        case["work"]["maximum_declared_live_payload_bits"]
        + STATIC_PUBLIC_RESOURCES["total_logical_payload_bits"]
        < case["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
        for case in cases
    )
    contexts = sorted({case["work"]["maximum_declared_live_context"] for case in cases})
    if every_inclusive_below_raw:
        result_name = "PASS_BOUNDED_EXACT_INPLACE_UNIT_PLAN_LIVE_REDUCTION"
    else:
        result_name = "PASS_BOUNDED_EXACT_INPLACE_UNIT_PLAN_PERSISTING_RESOURCE_NO_GO"
    claim = (
        "BOUNDED_EXACT_IN_PLACE_GL16Z_PUBLIC_UNIT_COORDINATE_ACTION_"
        "ELIMINATES_SIMULTANEOUS_WHOLE_OLD_ACCUMULATOR_AND_WHOLE_NEW_PRODUCT_"
        f"AND_REDUCES_PRIMARY_DECLARED_EXACT_DYNAMIC_LIVE_PAYLOAD_FROM{old_primary_bits}_TO{primary_bits}_BITS_"
        f"WITH{inclusive_bits}_BITS_INCLUDING_RETAINED_PUBLIC_PLAN_AND_UNIT_DESCRIPTOR_PAYLOAD_VERSUS{raw_bits}_MATCHED_RAW_BITS_"
        "WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_BUT_THE_IDENTICAL_"
        "CLASSICAL_IN_PLACE_COORDINATE_RECURRENCE_AND_PUBLIC_PLAN_LIBRARY_REMAIN"
    )
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_inplace_unit_plan.v1",
        "result": result_name,
        "claim": claim,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_M220_M221_M222_M223_M224_M225_M226_PARAMETERS3_7_9_11_13_17_19_GL16Z_INPLACE_PUBLIC_UNIT_AND_CONJUGATION_PLANS_POWER_BASIS_RAMANUJAN_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "plan_library": PLAN_LIBRARY_STATS,
        "static_public_resources": STATIC_PUBLIC_RESOURCES,
        "compiler_controls": controls,
        "cases": cases,
        "reuse": reuse,
        "predecessor_comparison": {
            "m226_semantic_parity": True,
            "m226_primary_dynamic_live_payload_bits": old_primary_bits,
            "m227_primary_dynamic_live_payload_bits": primary_bits,
            "m227_retained_public_plan_payload_bits": PLAN_LIBRARY_STATS[
                "retained_plan_integer_payload_bits"
            ],
            "m227_retained_predecessor_descriptor_payload_bits": (
                PUBLIC_UNIT_DESCRIPTOR["field_payload_bits"]
                + PUBLIC_UNIT_DESCRIPTOR["parameter_payload_bits"]
            ),
            "m227_total_static_public_resource_payload_bits": (
                STATIC_PUBLIC_RESOURCES["total_logical_payload_bits"]
            ),
            "m227_primary_dynamic_plus_static_public_payload_bits": inclusive_bits,
            "matched_raw_payload_bits": raw_bits,
            "dynamic_reduction_bits": old_primary_bits - primary_bits,
            "m226_direct_action_steps": old_primary["work"][
                "direct_unit_action_steps"
            ],
            "m227_plan_operation_steps": primary["work"][
                "inplace_plan_operation_steps"
            ],
        },
        "lifecycle_law": {
            "every_case_dynamic_below_matched_raw": every_dynamic_below_raw,
            "every_case_dynamic_plus_static_public_below_matched_raw": every_inclusive_below_raw,
            "dominant_contexts": contexts,
            "one_owned_coordinate_accumulator": all(
                case["work"]["inplace_coordinate_accumulators_materialized"]
                == case["work"]["inplace_coordinate_accumulators_released"]
                for case in cases
            ),
            "execution_dense_matrix_cells_zero": all(
                case["work"]["execution_dense_matrix_cells"] == 0 for case in cases
            ),
            "whole_old_and_new_field_product_coexistence_eliminated": True,
            "source_complete_scope": "INPLACE_PUBLIC_PLAN_OPERATION_INTERVALS_ONLY",
            "projected_boundary_retention_during_inverse_instrumented": False,
            "whole_transaction_live_payload_complete": False,
        },
        "separate_reference": {
            "imports_m227_production": reference.get("imports_m227_production"),
            "imports_m226_production": reference.get("imports_m226_production"),
            "uses_prior_standalone_m226_reference_substrate": reference.get(
                "uses_prior_standalone_m226_reference_substrate"
            ),
            "plan_compiler_and_resource_parity": True,
            "case_state_boundary_balance_resource_restoration_parity": True,
            "reuse_parity": True,
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_DETERMINISTIC_GL16Z_PLAN_COMPILER_AND_INPLACE_RATIONAL_COORDINATE_RECURRENCE",
            "matched_raw": "IDENTICAL_RAW_LINK_PATTERN_CUBIC_SKEIN_RECURRENCE_WITH_THE_SAME_DECLARED_FIELD_TEMPORARY_LAW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "compiler_matrices_and_work_counted": True,
            "retained_public_plan_library_counted": True,
            "retained_predecessor_public_unit_descriptor_counted": True,
            "inverse_plan_library_eliminated": True,
            "compiler_input_matrices_streamed_one_at_a_time": True,
            "dynamic_and_inclusive_payloads_reported_separately": True,
            "every_plan_operation_and_rational_temporary_counted": True,
            "verification_compiler_reexecution_excluded_from_accepted_path": True,
            "warm_runtime_measured": False,
            "whole_transaction_live_payload_complete": False,
            "excluded_not_zero": "PYTHON_FRACTION_OBJECT_CONTAINER_CAPACITY_ALLOCATOR_INTERPRETER_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_RSS",
        },
        "claim_limits": {
            "asymptotic_height_bound": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m226_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_direct_unit_action.py"
            ),
            "m226_result_sha256": sha256_file(predecessor_path),
            "m227_production_sha256": sha256_file(Path(__file__).resolve()),
            "m227_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_inplace_unit_plan_separate_reference.py"
            ),
            "m227_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
