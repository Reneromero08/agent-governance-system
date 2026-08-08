#!/usr/bin/env python3
"""Standalone M228 transposed-plan Hermitian trace-functional reference."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import su2_level8_cubic_skein_inplace_unit_plan_separate_reference as m227r


sys.set_int_max_str_digits(0)
m226r = m227r.m226r
m222 = m227r.m222
prior = m227r.prior
base = m227r.base
CASES = m227r.CASES
UNITS = m227r.UNITS
FIELD_DEGREE = m227r.FIELD_DEGREE


def fraction_payload(value: Fraction) -> int:
    return m227r.fraction_payload(value)


@dataclass
class Work(m227r.Work):
    trace_functional_evaluations: int = 0
    trace_functional_carrier_cells_scanned: int = 0
    trace_functional_coordinates_materialized: int = 0
    trace_functional_coordinates_released: int = 0
    trace_functional_nonzero_coordinate_pairs: int = 0
    trace_functional_nonzero_ramanujan_terms: int = 0
    trace_functional_zero_ramanujan_terms_skipped: int = 0
    trace_functional_rational_multiplications: int = 0
    trace_functional_rational_additions: int = 0
    transpose_factor_action_calls: int = 0
    transpose_factor_action_repetitions: int = 0
    transpose_plan_reference_reads: int = 0
    transpose_plan_operation_steps: int = 0
    transpose_plan_shears: int = 0
    transpose_plan_swaps: int = 0
    transpose_plan_negations: int = 0
    transpose_rational_integer_multiplications: int = 0
    transpose_rational_additions: int = 0
    maximum_trace_functional_payload_bits: int = 0
    maximum_trace_functional_transient_payload_bits: int = 0
    maximum_trace_functional_transient_rational_cells: int = 0
    maximum_trace_functional_integer_payload_bits: int = 0
    maximum_transpose_plan_descriptor_refs: int = 0
    maximum_transpose_action_descriptor_integer_cells: int = 0
    maximum_transpose_absolute_shear_coefficient: int = 0
    materialized_trace_weight_field_cells: int = 0

    def observe_trace_functional(
        self,
        *,
        resident_payload_bits: int,
        scratch_payload_bits: int,
        functional_payload_bits: int,
        transient_payload_bits: int,
        transient_rational_cells: int,
        integer_payload_bits: int,
        carrier_field_cells: int,
        context: str,
    ) -> None:
        payload = (
            resident_payload_bits
            + functional_payload_bits
            + transient_payload_bits
            + integer_payload_bits
        )
        self.coordinate_transient_observations += 1
        self.transient_observations += 1
        self.maximum_resident_payload_bits = max(
            self.maximum_resident_payload_bits, resident_payload_bits
        )
        self.maximum_scratch_payload_bits = max(
            self.maximum_scratch_payload_bits, scratch_payload_bits
        )
        self.maximum_trace_functional_payload_bits = max(
            self.maximum_trace_functional_payload_bits, functional_payload_bits
        )
        self.maximum_trace_functional_transient_payload_bits = max(
            self.maximum_trace_functional_transient_payload_bits,
            transient_payload_bits,
        )
        self.maximum_trace_functional_transient_rational_cells = max(
            self.maximum_trace_functional_transient_rational_cells,
            transient_rational_cells,
        )
        self.maximum_trace_functional_integer_payload_bits = max(
            self.maximum_trace_functional_integer_payload_bits,
            integer_payload_bits,
        )
        self.maximum_line_integer_payload_bits = max(
            self.maximum_line_integer_payload_bits, integer_payload_bits
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells, carrier_field_cells
        )

    def observe_transpose(
        self,
        *,
        resident_payload_bits: int,
        scratch_payload_bits: int,
        functional_payload_bits: int,
        transient_payload_bits: int,
        transient_rational_cells: int,
        integer_payload_bits: int,
        descriptor_integer_cells: int,
        plan_descriptor_refs: int,
        carrier_field_cells: int,
        context: str,
    ) -> None:
        self.maximum_transpose_plan_descriptor_refs = max(
            self.maximum_transpose_plan_descriptor_refs, plan_descriptor_refs
        )
        self.maximum_transpose_action_descriptor_integer_cells = max(
            self.maximum_transpose_action_descriptor_integer_cells,
            descriptor_integer_cells,
        )
        self.observe_trace_functional(
            resident_payload_bits=resident_payload_bits,
            scratch_payload_bits=scratch_payload_bits,
            functional_payload_bits=functional_payload_bits,
            transient_payload_bits=transient_payload_bits,
            transient_rational_cells=transient_rational_cells,
            integer_payload_bits=integer_payload_bits,
            carrier_field_cells=carrier_field_cells,
            context=context,
        )


def resident_metrics(
    residual: list[base.E], ledger: list[int], scratch: list[base.E]
) -> tuple[int, int, int]:
    scratch_bits = base.payload_bits(scratch)
    resident_bits = (
        base.payload_bits(residual) + prior.m220.ledger_bits(ledger) + scratch_bits
    )
    return resident_bits, scratch_bits, len(residual) + len(scratch)


def stream_hermitian_trace_functional(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    actions: Sequence[m227r.FactorAction],
    live_integers: tuple[int, ...],
) -> tuple[list[Fraction], int, int, int, int, int]:
    functional = [Fraction(0) for _ in range(FIELD_DEGREE)]
    functional_payload_bits = sum(fraction_payload(value) for value in functional)
    descriptor_integers = tuple(
        value for action in actions for value in action.integer_tuple()
    )
    base_integer_payload_bits = sum(
        m227r.signed_bits(value) for value in live_integers + descriptor_integers
    )
    resident_bits, scratch_bits, carrier_field_cells = resident_metrics(
        residual, ledger, scratch
    )
    work.exact_trace_energy_evaluations += 1
    work.streamed_embedding_energy_evaluations += 1
    work.trace_functional_evaluations += 1
    work.trace_functional_coordinates_materialized += FIELD_DEGREE
    for cell_index, value in enumerate(residual):
        work.streamed_embedding_cells_scanned += 1
        work.trace_functional_carrier_cells_scanned += 1
        for functional_index in range(FIELD_DEGREE):
            for left_index, left_coordinate in enumerate(value.coordinates):
                if not left_coordinate:
                    continue
                for right_index, right_coordinate in enumerate(value.coordinates):
                    if not right_coordinate:
                        continue
                    trace_coefficient = m226r.ramanujan_trace(
                        functional_index + left_index - right_index
                    )
                    if not trace_coefficient:
                        work.trace_functional_zero_ramanujan_terms_skipped += 1
                        integer_payload_bits = base_integer_payload_bits + sum(
                            m227r.signed_bits(integer)
                            for integer in (
                                cell_index,
                                functional_index,
                                left_index,
                                right_index,
                                trace_coefficient,
                            )
                        )
                        work.observe_trace_functional(
                            resident_payload_bits=resident_bits,
                            scratch_payload_bits=scratch_bits,
                            functional_payload_bits=functional_payload_bits,
                            transient_payload_bits=0,
                            transient_rational_cells=0,
                            integer_payload_bits=integer_payload_bits,
                            carrier_field_cells=carrier_field_cells,
                            context="STREAMED_HERMITIAN_TRACE_FUNCTIONAL_ZERO_SKIP",
                        )
                        continue
                    product = left_coordinate * right_coordinate
                    scaled = product * trace_coefficient
                    updated = functional[functional_index] + scaled
                    transient_payload_bits = (
                        fraction_payload(product)
                        + fraction_payload(scaled)
                        + fraction_payload(updated)
                    )
                    integer_payload_bits = base_integer_payload_bits + sum(
                        m227r.signed_bits(integer)
                        for integer in (
                            cell_index,
                            functional_index,
                            left_index,
                            right_index,
                            trace_coefficient,
                        )
                    )
                    work.trace_functional_nonzero_coordinate_pairs += 1
                    work.trace_functional_nonzero_ramanujan_terms += 1
                    work.trace_functional_rational_multiplications += 2
                    work.trace_functional_rational_additions += 1
                    work.observe_trace_functional(
                        resident_payload_bits=resident_bits,
                        scratch_payload_bits=scratch_bits,
                        functional_payload_bits=functional_payload_bits,
                        transient_payload_bits=transient_payload_bits,
                        transient_rational_cells=3,
                        integer_payload_bits=integer_payload_bits,
                        carrier_field_cells=carrier_field_cells,
                        context="STREAMED_HERMITIAN_TRACE_FUNCTIONAL_ACCUMULATE",
                    )
                    functional_payload_bits += fraction_payload(updated) - fraction_payload(
                        functional[functional_index]
                    )
                    functional[functional_index] = updated
                    del product, scaled, updated, transient_payload_bits
    if residual:
        del (
            cell_index,
            functional_index,
            left_coordinate,
            left_index,
            right_coordinate,
            right_index,
            trace_coefficient,
            value,
        )
    return (
        functional,
        functional_payload_bits,
        base_integer_payload_bits,
        len(descriptor_integers),
        resident_bits,
        scratch_bits,
    )


def apply_transposed_plan_in_place(
    functional: list[Fraction],
    plan_ref: m227r.PlanRef,
    work: Work,
    *,
    functional_payload_bits: int,
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
) -> int:
    plan = m227r.PLAN_LIBRARY[plan_ref.plan_index]
    operation_indices: Iterable[int] = (
        range(len(plan))
        if plan_ref.inverse
        else range(len(plan) - 1, -1, -1)
    )
    work.transpose_plan_reference_reads += 1
    for step_index, operation_index in enumerate(operation_indices):
        operation = plan[operation_index]
        effective_coefficient = (
            -operation.coefficient
            if plan_ref.inverse and operation.opcode == "SHEAR"
            else operation.coefficient
        )
        target = operation.source if operation.opcode == "SHEAR" else operation.target
        source = operation.target if operation.opcode == "SHEAR" else operation.source
        operation_descriptor = (
            {"SWAP": 0, "NEGATE": 1, "SHEAR": 2}[operation.opcode],
            target,
            source,
            effective_coefficient,
        )
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
            + sum(m227r.signed_bits(value) for value in operation_integers)
        )
        if operation.opcode == "SWAP":
            work.transpose_plan_swaps += 1
            work.observe_transpose(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                functional_payload_bits=functional_payload_bits,
                transient_payload_bits=0,
                transient_rational_cells=0,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context="TRANSPOSE_PUBLIC_UNIT_TRACE_FUNCTIONAL_SWAP",
            )
            functional[target], functional[source] = functional[source], functional[target]
        elif operation.opcode == "NEGATE":
            updated = -functional[target]
            transient_payload_bits = fraction_payload(updated)
            work.transpose_plan_negations += 1
            work.observe_transpose(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                functional_payload_bits=functional_payload_bits,
                transient_payload_bits=transient_payload_bits,
                transient_rational_cells=1,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context="TRANSPOSE_PUBLIC_UNIT_TRACE_FUNCTIONAL_NEGATE",
            )
            functional_payload_bits += fraction_payload(updated) - fraction_payload(
                functional[target]
            )
            functional[target] = updated
            del updated, transient_payload_bits
        elif operation.opcode == "SHEAR":
            if target == source or effective_coefficient == 0:
                raise RuntimeError("invalid standalone transposed shear")
            if effective_coefficient == 1:
                updated = functional[target] + functional[source]
                transient_payload_bits = fraction_payload(updated)
                transient_rational_cells = 1
            elif effective_coefficient == -1:
                updated = functional[target] - functional[source]
                transient_payload_bits = fraction_payload(updated)
                transient_rational_cells = 1
            else:
                scaled = functional[source] * effective_coefficient
                updated = functional[target] + scaled
                transient_payload_bits = fraction_payload(scaled) + fraction_payload(
                    updated
                )
                transient_rational_cells = 2
                work.transpose_rational_integer_multiplications += 1
            work.transpose_plan_shears += 1
            work.transpose_rational_additions += 1
            work.maximum_transpose_absolute_shear_coefficient = max(
                work.maximum_transpose_absolute_shear_coefficient,
                abs(effective_coefficient),
            )
            work.observe_transpose(
                resident_payload_bits=resident_payload_bits,
                scratch_payload_bits=scratch_payload_bits,
                functional_payload_bits=functional_payload_bits,
                transient_payload_bits=transient_payload_bits,
                transient_rational_cells=transient_rational_cells,
                integer_payload_bits=integer_payload_bits,
                descriptor_integer_cells=descriptor_integer_cells,
                plan_descriptor_refs=plan_descriptor_refs,
                carrier_field_cells=carrier_field_cells,
                context="TRANSPOSE_PUBLIC_UNIT_TRACE_FUNCTIONAL_SHEAR",
            )
            functional_payload_bits += fraction_payload(updated) - fraction_payload(
                functional[target]
            )
            functional[target] = updated
            if effective_coefficient not in (-1, 1):
                del scaled
            del updated, transient_payload_bits, transient_rational_cells
        else:
            raise RuntimeError("unknown standalone transposed operation")
        work.transpose_plan_operation_steps += 1
    return functional_payload_bits


def transpose_unit_action_energy(
    residual: list[base.E],
    ledger: list[int],
    scratch: list[base.E],
    work: Work,
    unit: m227r.UnitDirection | None,
    exponent: int,
    live_integers: tuple[int, ...],
    context_prefix: str,
) -> int:
    actions = list(m227r.ledger_actions(ledger, "NORM", "INVERSE_NORM"))
    work.inplace_ledger_norm_action_calls += len(actions)
    if unit is not None and exponent:
        unit_index = next(
            index for index, candidate in enumerate(UNITS)
            if candidate.parameter == unit.parameter
        )
        actions.append(
            m227r.FactorAction(
                unit_index,
                "INVERSE_NORM" if exponent > 0 else "NORM",
                abs(exponent),
            )
        )
    (
        functional,
        functional_payload_bits,
        base_integer_payload_bits,
        descriptor_integer_cells,
        resident_payload_bits,
        scratch_payload_bits,
    ) = stream_hermitian_trace_functional(
        residual,
        ledger,
        scratch,
        work,
        actions,
        live_integers + (exponent,),
    )
    carrier_field_cells = len(residual) + len(scratch)
    for action_index in range(len(actions) - 1, -1, -1):
        action = actions[action_index]
        refs = m227r.refs_for_kind(action.unit_index, action.kind)
        plan_descriptor_payload_bits = sum(
            m227r.signed_bits(reference.plan_index)
            + m227r.signed_bits(int(reference.inverse))
            for reference in refs
        )
        work.transpose_factor_action_calls += 1
        work.transpose_factor_action_repetitions += action.exponent
        for repetition in range(action.exponent - 1, -1, -1):
            for ref_index in range(len(refs) - 1, -1, -1):
                functional_payload_bits = apply_transposed_plan_in_place(
                    functional,
                    refs[ref_index],
                    work,
                    functional_payload_bits=functional_payload_bits,
                    resident_payload_bits=resident_payload_bits,
                    scratch_payload_bits=scratch_payload_bits,
                    base_integer_payload_bits=base_integer_payload_bits,
                    descriptor_integer_cells=descriptor_integer_cells,
                    plan_descriptor_refs=len(refs),
                    plan_descriptor_payload_bits=plan_descriptor_payload_bits,
                    carrier_field_cells=carrier_field_cells,
                    action_index=action_index,
                    repetition=repetition,
                    ref_index=ref_index,
                )
    if actions:
        del action, action_index, plan_descriptor_payload_bits, ref_index, refs, repetition
    result = functional[0]
    work.observe_trace_functional(
        resident_payload_bits=resident_payload_bits,
        scratch_payload_bits=scratch_payload_bits,
        functional_payload_bits=functional_payload_bits,
        transient_payload_bits=0,
        transient_rational_cells=0,
        integer_payload_bits=base_integer_payload_bits,
        carrier_field_cells=carrier_field_cells,
        context="TRANSPOSE_PUBLIC_UNIT_TRACE_FUNCTIONAL_HANDOFF",
    )
    functional.clear()
    work.trace_functional_coordinates_released += FIELD_DEGREE
    if result.denominator != 1 or result.numerator < 0:
        raise RuntimeError("standalone transposed trace functional invalid")
    return result.numerator


def apply_ref_plain(
    coordinates: Sequence[Fraction], reference: m227r.PlanRef
) -> tuple[Fraction, ...]:
    return tuple(
        m227r.apply_integer_plan(
            coordinates,
            m227r.PLAN_LIBRARY[reference.plan_index],
            inverse=reference.inverse,
        )
    )


def apply_transpose_plain(
    coordinates: Sequence[Fraction], reference: m227r.PlanRef
) -> tuple[Fraction, ...]:
    values = list(coordinates)
    plan = m227r.PLAN_LIBRARY[reference.plan_index]
    indices = range(len(plan)) if reference.inverse else range(len(plan) - 1, -1, -1)
    for index in indices:
        operation = plan[index]
        coefficient = (
            -operation.coefficient
            if reference.inverse and operation.opcode == "SHEAR"
            else operation.coefficient
        )
        if operation.opcode == "SWAP":
            values[operation.target], values[operation.source] = (
                values[operation.source], values[operation.target]
            )
        elif operation.opcode == "NEGATE":
            values[operation.target] = -values[operation.target]
        else:
            values[operation.source] += coefficient * values[operation.target]
    return tuple(values)


def controls() -> dict[str, bool]:
    left = tuple(Fraction(index + 1, 3) for index in range(FIELD_DEGREE))
    right = tuple(
        Fraction((index + 2) * (-1 if index % 3 else 1), 5)
        for index in range(FIELD_DEGREE)
    )
    transpose_identity_all = True
    inverse_transpose_identity_all = True
    for plan_index in range(len(m227r.PLAN_LIBRARY)):
        for inverse in (False, True):
            reference = m227r.PlanRef(plan_index, inverse)
            transformed = apply_ref_plain(right, reference)
            transpose_identity_all &= sum(
                a * b for a, b in zip(left, transformed, strict=True)
            ) == sum(
                a * b
                for a, b in zip(
                    apply_transpose_plain(left, reference), right, strict=True
                )
            )
            inverse_transpose_identity_all &= apply_transpose_plain(
                apply_transpose_plain(left, reference), reference.inverted()
            ) == left
    refs = (m227r.PlanRef(0), m227r.PlanRef(m227r.CONJUGATION_PLAN_INDEX))
    weight = (Fraction(1),) + (Fraction(0),) * (FIELD_DEGREE - 1)
    for reference in refs:
        weight = apply_ref_plain(weight, reference)
    dual = left
    for reference in reversed(refs):
        dual = apply_transpose_plain(dual, reference)
    wrong = left
    for reference in refs:
        wrong = apply_transpose_plain(wrong, reference)
    direct = sum(a * b for a, b in zip(left, weight, strict=True))
    control_value = base.E(
        tuple(Fraction((index % 5) - 2) for index in range(FIELD_DEGREE))
    )
    control_norm = control_value * prior.m220.conjugate(control_value)
    streamed_control = tuple(
        sum(
            left_coordinate
            * right_coordinate
            * m226r.ramanujan_trace(
                functional_index + left_index - right_index
            )
            for left_index, left_coordinate in enumerate(control_value.coordinates)
            for right_index, right_coordinate in enumerate(control_value.coordinates)
        )
        for functional_index in range(FIELD_DEGREE)
    )
    field_trace_control = tuple(
        prior.m220.trace(
            base.E(
                tuple(
                    Fraction(int(index == functional_index))
                    for index in range(FIELD_DEGREE)
                )
            )
            * control_norm
        )
        for functional_index in range(FIELD_DEGREE)
    )
    return {
        "transpose_dot_identity_all_public_plans": transpose_identity_all,
        "inverse_transpose_restores_all_public_plans": inverse_transpose_identity_all,
        "composite_reverse_order_matches": dual[0] == direct,
        "wrong_composite_order_changes_result": wrong[0] != direct,
        "hermitian_functional_matches_field_trace_basis": (
            streamed_control == field_trace_control
        ),
        "functional_coordinate_count_is_field_degree": FIELD_DEGREE == 16,
        "public_plan_library_reused_without_answer_compilation": (
            not m227r.PLAN_LIBRARY_STATS["answer_dependent_compilation"]
        ),
    }


def install_transpose_path() -> None:
    m227r.install_inplace_path()
    m226r.Work = Work
    m226r.direct_unit_action_energy = transpose_unit_action_energy
    m222.Work = Work
    prior.Work = Work


def main() -> None:
    install_transpose_path()
    print(
        json.dumps(
            {
                "schema": "cat_cas.su2_level8_cubic_skein_transpose_trace_functional_reference.v1",
                "imports_m228_production": False,
                "imports_m227_production": False,
                "uses_prior_standalone_m227_reference_substrate": True,
                "controls": controls(),
                "cases": [prior.case(*case) for case in CASES],
                "reuse": prior.reuse(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
