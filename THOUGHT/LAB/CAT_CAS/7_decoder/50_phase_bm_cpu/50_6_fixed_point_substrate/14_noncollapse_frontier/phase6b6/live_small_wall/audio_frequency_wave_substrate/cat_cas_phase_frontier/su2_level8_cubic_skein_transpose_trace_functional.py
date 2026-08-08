#!/usr/bin/env python3
"""M228 exact transposed-plan Hermitian trace-functional diagnostic.

M227 removes whole-field old/new product coexistence but still constructs a
transformed cyclotomic weight and then evaluates one exact trace accumulator.
M228 uses linearity instead: it streams the sixteen power-basis Hermitian trace
coordinates of the resident carrier and applies the transposes of the same
public GL(16,Z) plans in reverse composition order.  Coordinate zero is the
same final energy.  The complete functional, every exact transient, retained
public plans, and the identical classical recurrence remain counted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import su2_level8_cubic_skein_inplace_unit_plan as m227


sys.set_int_max_str_digits(0)
m226 = m227.m226
m222 = m227.m222
m221 = m227.m221
braid = m227.braid
CASES = m227.CASES
FIELD_DEGREE = m227.FIELD_DEGREE


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction_payload(value: Fraction) -> int:
    return m227.fraction_payload(value)


@dataclass
class Work(m227.Work):
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
    residual: list[braid.K], ledger: list[int], scratch: list[braid.K]
) -> tuple[int, int, int]:
    scratch_bits = braid.field_payload_bits(scratch)
    resident_bits = (
        braid.field_payload_bits(residual)
        + m221.m220.ledger_payload_bits(ledger)
        + scratch_bits
    )
    return resident_bits, scratch_bits, len(residual) + len(scratch)


def stream_hermitian_trace_functional(
    residual: list[braid.K],
    ledger: list[int],
    scratch: list[braid.K],
    work: Work,
    *,
    actions: Sequence[m227.FactorAction],
    live_integers: tuple[int, ...],
) -> tuple[list[Fraction], int, int, int, int, int]:
    functional = [Fraction(0) for _ in range(FIELD_DEGREE)]
    functional_payload_bits = sum(fraction_payload(value) for value in functional)
    descriptor_integers = tuple(
        value for action in actions for value in action.integer_tuple()
    )
    base_integer_payload_bits = sum(
        m227.signed_bits(value) for value in live_integers + descriptor_integers
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
            for left_index, left_coordinate in enumerate(value.coefficients):
                if not left_coordinate:
                    continue
                for right_index, right_coordinate in enumerate(value.coefficients):
                    if not right_coordinate:
                        continue
                    trace_coefficient = m221.m220.ramanujan_trace(
                        functional_index + left_index - right_index
                    )
                    if not trace_coefficient:
                        work.trace_functional_zero_ramanujan_terms_skipped += 1
                        integer_payload_bits = base_integer_payload_bits + sum(
                            m227.signed_bits(integer)
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
                        m227.signed_bits(integer)
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
    plan_ref: m227.PlanRef,
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
    plan = m227.PLAN_LIBRARY[plan_ref.plan_index]
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
            + sum(m227.signed_bits(value) for value in operation_integers)
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
                raise RuntimeError("invalid transposed public shear")
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
            raise RuntimeError("unknown transposed public operation")
        work.transpose_plan_operation_steps += 1
    return functional_payload_bits


def transpose_unit_action_energy(
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
    actions = list(m227.ledger_actions(ledger, "NORM", "INVERSE_NORM"))
    work.inplace_ledger_norm_action_calls += len(actions)
    if direction is not None and exponent:
        unit_index = next(
            index
            for index, candidate in enumerate(m222.UNITS)
            if candidate.parameter == direction.parameter
        )
        actions.append(
            m227.FactorAction(
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
        actions=actions,
        live_integers=live_integers + (exponent,),
    )
    carrier_field_cells = len(residual) + len(scratch)
    for action_index in range(len(actions) - 1, -1, -1):
        action = actions[action_index]
        refs = m227.refs_for_kind(action.unit_index, action.kind)
        plan_descriptor_payload_bits = sum(
            m227.signed_bits(reference.plan_index)
            + m227.signed_bits(int(reference.inverse))
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
    if result.denominator != 1:
        raise RuntimeError("transposed trace functional is not integral")
    if result.numerator < 0:
        raise RuntimeError("transposed trace functional became negative")
    return result.numerator


def apply_ref_plain(
    coordinates: Sequence[Fraction], reference: m227.PlanRef
) -> tuple[Fraction, ...]:
    return tuple(
        m227.apply_integer_plan(
            coordinates,
            m227.PLAN_LIBRARY[reference.plan_index],
            inverse=reference.inverse,
        )
    )


def apply_transpose_plain(
    coordinates: Sequence[Fraction], reference: m227.PlanRef
) -> tuple[Fraction, ...]:
    values = list(coordinates)
    plan = m227.PLAN_LIBRARY[reference.plan_index]
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
                values[operation.source],
                values[operation.target],
            )
        elif operation.opcode == "NEGATE":
            values[operation.target] = -values[operation.target]
        else:
            values[operation.source] += coefficient * values[operation.target]
    return tuple(values)


def controls() -> dict[str, bool]:
    left = tuple(Fraction(index + 1, 3) for index in range(FIELD_DEGREE))
    right = tuple(Fraction((index + 2) * (-1 if index % 3 else 1), 5) for index in range(FIELD_DEGREE))
    transpose_identity_all = True
    inverse_transpose_identity_all = True
    for plan_index in range(len(m227.PLAN_LIBRARY)):
        for inverse in (False, True):
            reference = m227.PlanRef(plan_index, inverse)
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
    refs = (m227.PlanRef(0), m227.PlanRef(m227.CONJUGATION_PLAN_INDEX))
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
    control_value = braid.K(
        tuple(Fraction((index % 5) - 2) for index in range(FIELD_DEGREE))
    )
    control_norm = control_value * m221.m220.conjugate(control_value)
    streamed_control = tuple(
        sum(
            left_coordinate
            * right_coordinate
            * m221.m220.ramanujan_trace(
                functional_index + left_index - right_index
            )
            for left_index, left_coordinate in enumerate(control_value.coefficients)
            for right_index, right_coordinate in enumerate(control_value.coefficients)
        )
        for functional_index in range(FIELD_DEGREE)
    )
    field_trace_control = tuple(
        m221.m220.field_trace(
            braid.K(
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
            not m227.PLAN_LIBRARY_STATS["answer_dependent_compilation"]
        ),
    }


def install_transpose_path() -> None:
    m227.install_inplace_path()
    m226.Work = Work
    m226.direct_unit_action_energy = transpose_unit_action_energy
    m222.Work = Work
    m221.Work = Work


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_cubic_skein_transpose_trace_functional.py "
            "SEPARATE_REFERENCE_JSON"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M228 reference is forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_cubic_skein_transpose_trace_functional_reference.v1":
        raise RuntimeError("M228 separate-reference schema changed")
    if reference.get("controls") != controls():
        raise RuntimeError("M228 independent transpose controls differ")
    install_transpose_path()
    cases = [m221.execute_case(*case) for case in CASES]
    if [m221.reference_case_view(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M228 independent case/resource parity failed")
    reuse = m221.reuse_result()
    reference_reuse = reference["reuse"]
    for section in ("primary", "reuse", "fresh_reuse"):
        if m221.reference_case_view(reuse[section]) != reference_reuse[section]:
            raise RuntimeError(f"M228 independent reuse parity failed: {section}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse[key] != reference_reuse[key]:
            raise RuntimeError(f"M228 independent reuse parity failed: {key}")
    predecessor_path = here / "SU2_LEVEL8_CUBIC_SKEIN_INPLACE_UNIT_PLAN_RESULTS.json"
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
                raise RuntimeError(f"M228 predecessor semantic parity failed: {key}")
    for section in ("primary", "reuse", "fresh_reuse"):
        for key in semantic_keys[3:]:
            if reuse[section][key] != predecessor["reuse"][section][key]:
                raise RuntimeError(f"M228 predecessor reuse parity failed: {section}.{key}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_state_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse[key] != predecessor["reuse"][key]:
            raise RuntimeError(f"M228 predecessor top-level reuse parity failed: {key}")
    primary = next(
        case for case in cases
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    old_primary = next(
        case for case in predecessor["cases"]
        if (case["strands"], case["rounds"], case["family"]) == (4, 4, 0)
    )
    primary_bits = primary["work"]["maximum_declared_live_payload_bits"]
    old_primary_bits = old_primary["work"]["maximum_declared_live_payload_bits"]
    raw_bits = primary["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
    static_bits = m227.STATIC_PUBLIC_RESOURCES["total_logical_payload_bits"]
    inclusive_bits = primary_bits + static_bits
    every_dynamic_below_raw = all(
        case["work"]["maximum_declared_live_payload_bits"]
        < case["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
        for case in cases
    )
    every_inclusive_below_raw = all(
        case["work"]["maximum_declared_live_payload_bits"] + static_bits
        < case["matched_raw_recurrence"]["maximum_declared_live_payload_bits"]
        for case in cases
    )
    result_name = (
        "PASS_BOUNDED_EXACT_TRANSPOSE_TRACE_FUNCTIONAL_LIVE_REDUCTION"
        if every_inclusive_below_raw
        else "PASS_BOUNDED_EXACT_TRANSPOSE_TRACE_FUNCTIONAL_PERSISTING_RESOURCE_NO_GO"
    )
    direction = "REDUCES" if primary_bits < old_primary_bits else "INCREASES"
    claim = (
        "BOUNDED_EXACT_TRANSPOSE_GL16Z_PUBLIC_UNIT_PLAN_ON_STREAMED_HERMITIAN_TRACE_FUNCTIONAL_"
        "ELIMINATES_MATERIALIZED_TRACE_WEIGHT_FIELD_"
        f"BUT_{direction}_PRIMARY_DECLARED_EXACT_DYNAMIC_LIVE_PAYLOAD_FROM{old_primary_bits}_TO{primary_bits}_BITS_"
        f"WITH{inclusive_bits}_BITS_INCLUDING_RETAINED_PUBLIC_PLAN_AND_UNIT_DESCRIPTOR_PAYLOAD_VERSUS{raw_bits}_MATCHED_RAW_BITS_"
        "WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_AND_THE_IDENTICAL_CLASSICAL_TRANSPOSE_FUNCTIONAL_RECURRENCE_REMAINS"
    )
    result = {
        "schema": "cat_cas.su2_level8_cubic_skein_transpose_trace_functional.v1",
        "result": result_name,
        "claim": claim,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_QZETA40_M219_THROUGH_M227_PARAMETERS3_7_9_11_13_17_19_TRANSPOSE_GL16Z_PUBLIC_UNIT_PLANS_STREAMED16_COORDINATE_HERMITIAN_TRACE_FUNCTIONAL_POWER_BASIS_RAMANUJAN_TRACE_LINE_SEARCH_FAMILY0_STRANDS4_DEPTH1TO4_STRANDS6_DEPTH1TO2_STRANDS8_DEPTH1_PRIMARY4_DEPTH4_REUSE4_DEPTH2_FAMILY1_DIRECT_PROCESS_ONLY",
        "controls": controls(),
        "plan_library": m227.PLAN_LIBRARY_STATS,
        "static_public_resources": m227.STATIC_PUBLIC_RESOURCES,
        "cases": cases,
        "reuse": reuse,
        "predecessor_comparison": {
            "m227_semantic_parity": True,
            "m227_primary_dynamic_live_payload_bits": old_primary_bits,
            "m228_primary_dynamic_live_payload_bits": primary_bits,
            "m228_primary_dynamic_plus_static_public_payload_bits": inclusive_bits,
            "matched_raw_payload_bits": raw_bits,
            "dynamic_delta_bits": primary_bits - old_primary_bits,
            "m227_materialized_trace_weight_rematerializations": old_primary["work"][
                "direct_trace_weight_rematerializations"
            ],
            "m228_materialized_trace_weight_field_cells": primary["work"][
                "materialized_trace_weight_field_cells"
            ],
            "m228_trace_functional_coordinates_materialized": primary["work"][
                "trace_functional_coordinates_materialized"
            ],
            "m228_transpose_plan_operation_steps": primary["work"][
                "transpose_plan_operation_steps"
            ],
        },
        "lifecycle_law": {
            "every_case_dynamic_below_matched_raw": every_dynamic_below_raw,
            "every_case_dynamic_plus_static_public_below_matched_raw": every_inclusive_below_raw,
            "materialized_trace_weight_field_cells": 0,
            "functional_coordinates_owned_and_released": all(
                case["work"]["trace_functional_coordinates_materialized"]
                == case["work"]["trace_functional_coordinates_released"]
                for case in cases
            ),
            "source_complete_scope": "STREAMED_FUNCTIONAL_ALL_TERM_AND_TRANSPOSE_PLAN_INTERVALS_ONLY",
            "zero_ramanujan_skip_intervals_instrumented": True,
            "projected_boundary_retention_during_inverse_instrumented": False,
            "whole_transaction_live_payload_complete": False,
        },
        "separate_reference": {
            "imports_m228_production": reference.get("imports_m228_production"),
            "imports_m227_production": reference.get("imports_m227_production"),
            "uses_prior_standalone_m227_reference_substrate": reference.get(
                "uses_prior_standalone_m227_reference_substrate"
            ),
            "transpose_formula_controls_parity": True,
            "case_resource_restoration_reuse_parity": True,
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_STREAMED16_COORDINATE_HERMITIAN_FUNCTIONAL_AND_TRANSPOSE_GL16Z_PLAN_RECURRENCE",
            "matched_raw": "IDENTICAL_RAW_LINK_PATTERN_CUBIC_SKEIN_RECURRENCE_WITH_THE_SAME_DECLARED_FIELD_TEMPORARY_LAW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "resource_law": {
            "complete_functional_vector_counted": True,
            "every_nonzero_functional_update_transient_counted": True,
            "every_transpose_plan_operation_and_transient_counted": True,
            "retained_public_plan_and_descriptor_counted": True,
            "compiler_resources_inherited_from_m227": True,
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
            "m227_production_sha256": sha256_file(
                here / "su2_level8_cubic_skein_inplace_unit_plan.py"
            ),
            "m227_result_sha256": sha256_file(predecessor_path),
            "m228_production_sha256": sha256_file(Path(__file__).resolve()),
            "m228_separate_reference_code_sha256": sha256_file(
                here / "su2_level8_cubic_skein_transpose_trace_functional_separate_reference.py"
            ),
            "m228_separate_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
