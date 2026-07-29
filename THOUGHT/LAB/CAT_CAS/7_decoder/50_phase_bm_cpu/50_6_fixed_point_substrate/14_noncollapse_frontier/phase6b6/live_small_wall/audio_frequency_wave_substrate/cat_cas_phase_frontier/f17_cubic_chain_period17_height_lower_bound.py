#!/usr/bin/env python3
"""Certify a pi-adic height lower bound for the period-17 boundary orbit.

Let pi = 1 - zeta_17.  Exact omitted-root regauging preserves the represented
element and its pi-valuation; pi-content factoring only transfers valuation to
an exponent ledger.  This diagnostic derives a tropical lower bound from the
exact characteristic recurrence, then reduces the equality cases modulo pi.

The finite-field recurrence is used only to certify that infinitely many
normalized residues are nonzero.  It does not replace exact boundary
semantics and does not establish a physical or computational resource.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

import f17_cubic_chain_period17_cyclotomic_module as cyclo


PRIME = cyclo.PRIME
DIMENSION = cyclo.DIMENSION
INITIAL_BOUNDARIES = DIMENSION + 1
RESTORATION_PERIODS = DIMENSION
CYCLE_STEP_CAP = 10_000_000

RingElement = cyclo.RingElement


def fail(message: str) -> None:
    raise RuntimeError(message)


def height_lower_bound(periods: int) -> int:
    if periods < 1:
        fail("height bound requires a positive period")
    return (272 * periods + 18) // 3


def pi_element() -> RingElement:
    return cyclo.ring_subtract(
        cyclo.ring_one(),
        cyclo.ring_monomial(1),
    )


PI = pi_element()


def divide_pi_exact(element: RingElement) -> RingElement:
    total = sum(element)
    if total % PRIME:
        fail("cyclotomic integer is not divisible by pi")
    last = total // PRIME
    prefix = 0
    quotient = []
    for index, coefficient in enumerate(element):
        prefix += coefficient
        quotient.append(prefix - (index + 1) * last)
    result = tuple(quotient)
    if cyclo.ring_multiply(PI, result) != element:
        fail("exact pi division identity failed")
    return result


def pi_valuation(element: RingElement) -> int | None:
    if element == cyclo.ring_zero():
        return None
    current = element
    valuation = 0
    while sum(current) % PRIME == 0:
        current = divide_pi_exact(current)
        valuation += 1
    return valuation


def residue_after_pi_division(
    element: RingElement,
    exponent: int,
) -> int:
    current = element
    for _ in range(exponent):
        current = divide_pi_exact(current)
    return sum(current) % PRIME


def direct_boundaries(
    block: cyclo.CompiledBlock,
) -> tuple[list[RingElement], int]:
    current = cyclo.seed_vector(block.public_program)
    stats = cyclo.Stats()
    boundaries = []
    for _ in range(INITIAL_BOUNDARIES):
        current = cyclo.apply_operator(
            block.operator,
            current,
            stats,
        )
        boundaries.append(cyclo.project_boundary(current))
    return boundaries, stats.ring_multiply_accumulations


def coefficient_data(
    characteristic: list[RingElement],
) -> list[dict[str, Any]]:
    if len(characteristic) != PRIME + 1:
        fail("unexpected characteristic width")
    if characteristic[0] != cyclo.ring_one():
        fail("characteristic is not monic")
    if characteristic[-1] != cyclo.ring_zero():
        fail("whole characteristic constant is not zero")
    data = []
    for lag in range(1, DIMENSION + 1):
        coefficient = characteristic[lag]
        valuation = pi_valuation(coefficient)
        requirements = [
            height_lower_bound(300 + phase)
            - height_lower_bound(300 + phase - lag)
            for phase in (0, 1, 2)
        ]
        maximum_required = max(requirements)
        if valuation is not None and valuation < maximum_required:
            fail("coefficient valuation does not prove height induction")
        residues = []
        for phase, requirement in zip(
            (0, 1, 2),
            requirements,
            strict=True,
        ):
            residues.append(
                residue_after_pi_division(
                    coefficient,
                    requirement,
                )
                if valuation == requirement
                else 0
            )
        data.append(
            {
                "lag": lag,
                "valuation": valuation,
                "required_by_target_mod3": {
                    str(phase): requirement
                    for phase, requirement in zip(
                        (0, 1, 2),
                        requirements,
                        strict=True,
                    )
                },
                "normalized_residue_by_target_mod3": {
                    str(phase): residue
                    for phase, residue in zip(
                        (0, 1, 2),
                        residues,
                        strict=True,
                    )
                },
            }
        )
    return data


State = tuple[int, tuple[int, ...]]


def transition(
    state: State,
    coefficient_rows: dict[int, tuple[int, ...]],
) -> State:
    target_mod3, values = state
    if len(values) != DIMENSION:
        fail("normalized recurrence state width changed")
    coefficients = coefficient_rows[target_mod3]
    next_value = -sum(
        coefficient * values[-lag]
        for lag, coefficient in enumerate(coefficients, start=1)
    )
    next_value %= PRIME
    return (
        (target_mod3 + 1) % 3,
        (*values[1:], next_value),
    )


def cycle_certificate(
    initial_residues: list[int],
    coefficient_rows: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    if len(initial_residues) != DIMENSION:
        fail("cycle seed must contain periods 1 through 16")
    start: State = (
        INITIAL_BOUNDARIES % 3,
        tuple(initial_residues),
    )
    tortoise = transition(start, coefficient_rows)
    hare = transition(
        transition(start, coefficient_rows),
        coefficient_rows,
    )
    steps = 1
    while tortoise != hare and steps < CYCLE_STEP_CAP:
        tortoise = transition(tortoise, coefficient_rows)
        hare = transition(
            transition(hare, coefficient_rows),
            coefficient_rows,
        )
        steps += 1
    if tortoise != hare:
        fail("normalized residue cycle was not found below cap")

    prefix = 0
    tortoise = start
    while tortoise != hare:
        tortoise = transition(tortoise, coefficient_rows)
        hare = transition(hare, coefficient_rows)
        prefix += 1
        if prefix >= CYCLE_STEP_CAP:
            fail("normalized residue cycle prefix exceeded cap")

    cycle_length = 1
    hare = transition(tortoise, coefficient_rows)
    while tortoise != hare:
        hare = transition(hare, coefficient_rows)
        cycle_length += 1
        if cycle_length >= CYCLE_STEP_CAP:
            fail("normalized residue cycle length exceeded cap")

    cursor = tortoise
    nonzero_outputs = 0
    cycle_sha = hashlib.sha256()
    for _ in range(cycle_length):
        next_state = transition(cursor, coefficient_rows)
        output = next_state[1][-1]
        nonzero_outputs += int(output != 0)
        cycle_sha.update(bytes([output]))
        cursor = next_state
    if cursor != tortoise:
        fail("normalized residue cycle did not close")
    return {
        "algorithm": "FLOYD_CONSTANT_STATE_CYCLE_DETECTION",
        "step_cap": CYCLE_STEP_CAP,
        "meeting_steps": steps,
        "prefix_length": prefix,
        "cycle_length": cycle_length,
        "cycle_nonzero_outputs": nonzero_outputs,
        "cycle_zero_outputs": cycle_length - nonzero_outputs,
        "cycle_output_sha256": cycle_sha.hexdigest(),
        "cycle_has_nonzero_output": nonzero_outputs > 0,
        "retained_floyd_core_recurrence_residue_cells": (
            3 * DIMENSION
        ),
        "retained_floyd_core_phase_scalar_cells": 3,
        "retained_cycle_dictionary_entries": 0,
    }


def family_result(
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    boundaries, direct_accumulations = direct_boundaries(block)
    valuations = [pi_valuation(boundary) for boundary in boundaries]
    if any(value is None for value in valuations):
        fail("initial boundary unexpectedly vanished")
    finite_valuations = [int(value) for value in valuations]
    bounds = [
        height_lower_bound(period)
        for period in range(1, INITIAL_BOUNDARIES + 1)
    ]
    if any(
        value < bound
        for value, bound in zip(
            finite_valuations,
            bounds,
            strict=True,
        )
    ):
        fail("initial boundary violates height lower bound")
    initial_residues = [
        residue_after_pi_division(boundary, bound)
        for boundary, bound in zip(boundaries, bounds, strict=True)
    ]
    coefficient_records = coefficient_data(block.characteristic)
    coefficient_rows = {
        phase: tuple(
            record["normalized_residue_by_target_mod3"][str(phase)]
            for record in coefficient_records
        )
        for phase in (0, 1, 2)
    }
    recurrence_state: State = (
        INITIAL_BOUNDARIES % 3,
        tuple(initial_residues[:DIMENSION]),
    )
    generated_period17 = transition(
        recurrence_state,
        coefficient_rows,
    )[1][-1]
    if generated_period17 != initial_residues[DIMENSION]:
        fail("normalized recurrence did not reproduce period 17")
    cycle = cycle_certificate(
        initial_residues[:DIMENSION],
        coefficient_rows,
    )
    if not cycle["cycle_has_nonzero_output"]:
        fail("normalized recurrence cycle has no nonzero output")
    return {
        "family": block.family,
        "public_program_sha256": hashlib.sha256(
            cyclo.adaptive.encoded_program(block.public_program)
        ).hexdigest(),
        "operator_sha256": block.operator_sha256,
        "characteristic_sha256": block.characteristic_sha256,
        "characteristic_identity_exact": (
            block.characteristic_identity_exact
        ),
        "initial_periods": list(range(1, INITIAL_BOUNDARIES + 1)),
        "initial_boundary_valuations": finite_valuations,
        "initial_height_lower_bounds": bounds,
        "initial_normalized_residues_mod17": initial_residues,
        "period17_recurrence_residue_equal": True,
        "coefficient_records": coefficient_records,
        "coefficient_valuations_prove_induction": True,
        "height_lower_bound_formula": "CEIL((272*N+16)/3)",
        "direct_ring_multiply_accumulations": direct_accumulations,
        "cycle": cycle,
    }


def restoration_reuse(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = cyclo.Carrier.create(RESTORATION_PERIODS)
    backing = carrier.backing_identity()
    primary_transaction = cyclo.execute_transaction(carrier, primary)
    reuse_transaction = cyclo.execute_transaction(carrier, reuse)
    fresh_transaction = cyclo.execute_transaction(
        cyclo.Carrier.create(RESTORATION_PERIODS),
        reuse,
    )
    return {
        "periods": RESTORATION_PERIODS,
        "restoration_scope": (
            "EXACT_MESSAGE_PAYLOAD_ZERO_AND_SAME_BACKING_WITH_"
            "MONOTONE_GENERATION_AND_LEASE_METADATA"
        ),
        "primary_restored_exactly": (
            primary_transaction.restored_exactly
        ),
        "reuse_restored_exactly": reuse_transaction.restored_exactly,
        "same_original_backing": (
            primary_transaction.same_backing
            and reuse_transaction.same_backing
            and carrier.backing_identity() == backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse_transaction.boundary == fresh_transaction.boundary
        ),
        "generation": carrier.generation,
        "lease": carrier.lease,
        "all_messages_zero": carrier.all_zero(),
        "full_carrier_object_state_equal": False,
        "repeated_use_metadata_width_bounded": False,
        "retained_inverse_history_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail(
            "usage: f17_cubic_chain_period17_"
            "height_lower_bound.py"
        )
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    families = [
        family_result(blocks[family])
        for family in ("primary", "reuse")
    ]
    restored = restoration_reuse(
        blocks["primary"],
        blocks["reuse"],
    )
    exact_cycle_densities = {}
    for family in families:
        numerator = family["cycle"]["cycle_nonzero_outputs"]
        denominator = family["cycle"]["cycle_length"]
        common = math.gcd(numerator, denominator)
        exact_cycle_densities[family["family"].lower()] = {
            "numerator": numerator // common,
            "denominator": denominator // common,
        }
    result = {
        "result": "PASS",
        "experiment": (
            "EXACT_F17_PERIOD17_BOUNDARY_HEIGHT_LOWER_BOUND_"
            "FOR_LOSSLESS_EXACT_BOUNDARY_ENCODING"
        ),
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "uniformizer": "PI_EQUALS_1_MINUS_ZETA17",
        "pi_ramification_identity": "17_IS_A_UNIT_TIMES_PI_TO_POWER16",
        "height_lower_bound_formula": "CEIL((272*N+16)/3)",
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": (
                    block.characteristic_sha256
                ),
                "characteristic_identity_exact": (
                    block.characteristic_identity_exact
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "families": families,
        "all_initial_boundaries_satisfy_bound": True,
        "all_characteristic_identities_exact": all(
            block.characteristic_identity_exact
            for block in blocks.values()
        ),
        "all_coefficient_valuations_prove_induction": all(
            family["coefficient_valuations_prove_induction"]
            for family in families
        ),
        "all_normalized_cycles_have_nonzero_outputs": all(
            family["cycle"]["cycle_has_nonzero_output"]
            for family in families
        ),
        "exact_cycle_nonzero_densities": exact_cycle_densities,
        "infinitely_many_exact_nonzero_boundaries_certified": True,
        "infinitely_many_distinct_pi_valuations_certified": True,
        "fixed_finite_lossless_discrete_boundary_alphabet_rejected": (
            True
        ),
        "asymptotic_lossless_exact_boundary_encoding_horizon_lower_bound": (
            "MAX_CODE_WIDTH_THROUGH_N_IS_OMEGA_LOG_N_WITHOUT_"
            "FREE_PERIOD_INDEX"
        ),
        "lower_bound_scope": (
            "WORST_CASE_INJECTIVE_EXACT_BOUNDARY_OR_VALUATION_"
            "ENCODING_THROUGH_PERIOD_N_DECODER_GETS_NO_FREE_N"
        ),
        "lower_bound_derivation": (
            "POSITIVE_NORMALIZED_CYCLE_DENSITY_GIVES_OMEGA_N_"
            "DISTINCT_EXACT_BOUNDARY_VALUATIONS_THROUGH_N_AND_"
            "THUS_A_LOG2_CARDINALITY_HORIZON_BOUND"
        ),
        "exact_regauging_preserves_pi_valuation": True,
        "pi_content_factoring_transfers_valuation_to_exponent_ledger": (
            True
        ),
        "cyclotomic_units_can_remove_pi_valuation": False,
        "compact_exponent_ledger_upper_bound": (
            "O_LOG_N_BITS_FOR_THE_PI_EXPONENT_ALONE"
        ),
        "restoration_reuse_case": restored,
        "matched_classical": {
            "identical_characteristic_recurrence_available": True,
            "identical_pi_valuation_certificate_available": True,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "accounting_scope": (
                "NAMED_LOGICAL_COMPONENTS_NOT_AN_EXACT_TEMPORARY_"
                "OR_WHOLE_PROCESS_PEAK"
            ),
            "production_operator_integer_cells_two_families": (
                2 * cyclo.OPERATOR_INTEGER_CELLS
            ),
            "production_characteristic_integer_cells_two_families": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "direct_stream_two_message_integer_cells": (
                2 * cyclo.MESSAGE_INTEGER_CELLS
            ),
            "stored_initial_exact_boundary_integer_cells_per_family": (
                INITIAL_BOUNDARIES * DIMENSION
            ),
            "normalized_coefficient_table_field_cells_per_family": (
                3 * DIMENSION
            ),
            "normalized_initial_residue_field_cells_per_family": (
                INITIAL_BOUNDARIES
            ),
            "floyd_core_recurrence_residue_cells_per_family": (
                3 * DIMENSION
            ),
            "floyd_core_phase_scalar_cells_per_family": 3,
            "cycle_dictionary_entries": 0,
            "restoration_carrier_message_slots": len(
                cyclo.Carrier.create(RESTORATION_PERIODS).messages
            ),
            "restoration_carrier_integer_cells": (
                len(
                    cyclo.Carrier.create(
                        RESTORATION_PERIODS
                    ).messages
                )
                * cyclo.MESSAGE_INTEGER_CELLS
            ),
            "exact_temporary_peak_bounded": False,
            "python_object_overhead_bounded": False,
            "sympy_internal_temporaries_bounded": False,
            "allocator_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "LINEAR_BIT_LOWER_BOUND",
            "NO_COMPACT_VARIABLE_LENGTH_EXACT_ENCODING",
            "GENERIC_MACHINE_STATE_MEMORY_LOWER_BOUND",
            "ONLINE_SPACE_LOWER_BOUND",
            "POINTWISE_OMEGA_LOG_N_BITS_AT_EVERY_PERIOD",
            "LOWER_BOUND_WITH_PERIOD_INDEX_OR_EXTERNAL_COUNTER_FREE",
            "NO_COMPACT_INDEXED_GENERATOR",
            "FULL_CARRIER_OBJECT_STATE_EQUALITY_AFTER_RESTORATION",
            "BOUNDED_REPEATED_USE_GENERATION_AND_LEASE_METADATA",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": (
            "EXACT_BOUNDARY_PI_VALUATION_REQUIRES_AN_UNBOUNDED_"
            "LOSSLESS_DISCRETE_BOUNDARY_ALPHABET_BUT_ITS_"
            "EXPONENT_ALONE_HAS_A_COMPACT_LOGARITHMIC_LEDGER_"
            "AND_THE_IDENTICAL_CLASSICAL_RECURRENCE_REMAINS"
        ),
        "terminal": False,
    }
    if (
        not result["all_coefficient_valuations_prove_induction"]
        or not result["all_characteristic_identities_exact"]
        or not result["all_normalized_cycles_have_nonzero_outputs"]
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("height lower-bound qualification failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
