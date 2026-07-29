#!/usr/bin/env python3
"""Execute the exact period-17 Cayley-Hamilton recurrence.

The predecessor certifies a degree-17 annihilator over Q(zeta_17), but its
runtime still applies the dense period block at every step.  Here the zero
constant coefficient is made explicit: the characteristic polynomial is
x*q(x), with monic q of degree 16.  For n >= 1, A**n is represented as
A times x**(n-1) modulo q.

The carrier builds the fixed resident basis A*seed through A**16*seed once,
executes binary polynomial powering over Z[zeta_17], and contracts the
resulting 16 coefficient registers into one boundary message.  The output,
coefficient registers, basis, and seed are then exactly subtracted in lawful
reverse order.  This uses 18 resident message slots plus 16 cyclotomic
coefficient registers across the declared periods; integer width remains
part of the material state, so this is not a fixed total footprint.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_cyclotomic_module as cyclo


PRIME = cyclo.PRIME
DIMENSION = cyclo.DIMENSION
MESSAGE_INTEGER_CELLS = cyclo.MESSAGE_INTEGER_CELLS
BASIS_MESSAGES = DIMENSION
MESSAGE_SLOTS = 1 + BASIS_MESSAGES + 1
OUTPUT_SLOT = MESSAGE_SLOTS - 1
COEFFICIENT_REGISTERS = DIMENSION
COEFFICIENT_INTEGER_CELLS = COEFFICIENT_REGISTERS * DIMENSION
TESTED_PERIODS = (1, 4, 16, 64)

RingElement = cyclo.RingElement
RingVector = cyclo.RingVector


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    return cyclo.signed_bits(value)


def ring_negate(value: RingElement) -> RingElement:
    return tuple(-coefficient for coefficient in value)


def polynomial_zero() -> list[RingElement]:
    return [cyclo.ring_zero() for _ in range(DIMENSION)]


def polynomial_one() -> list[RingElement]:
    result = polynomial_zero()
    result[0] = cyclo.ring_one()
    return result


def polynomial_x() -> list[RingElement]:
    result = polynomial_zero()
    result[1] = cyclo.ring_one()
    return result


@dataclass
class RecurrenceStats:
    polynomial_multiplications: int = 0
    polynomial_ring_multiplications: int = 0
    polynomial_reduction_updates: int = 0
    output_ring_multiplications: int = 0
    basis_forward_block_applications: int = 0
    basis_inverse_block_applications: int = 0
    basis_ring_multiply_accumulations: int = 0
    maximum_carrier_payload_bits: int = 0
    maximum_coefficient_signed_bits: int = 1
    maximum_nonzero_message_slots: int = 0
    maximum_nonzero_coefficient_registers: int = 0


def multiply_mod_q(
    left: list[RingElement],
    right: list[RingElement],
    characteristic: list[RingElement],
    stats: RecurrenceStats,
) -> list[RingElement]:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        fail("invalid recurrence polynomial width")
    if len(characteristic) != PRIME + 1:
        fail("invalid characteristic width")
    if characteristic[-1] != cyclo.ring_zero():
        fail("fixed integral recurrence requires zero constant coefficient")
    product = [
        cyclo.ring_zero()
        for _ in range(2 * DIMENSION - 1)
    ]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            product[left_degree + right_degree] = cyclo.ring_add(
                product[left_degree + right_degree],
                cyclo.ring_multiply(left_value, right_value),
            )
            stats.polynomial_ring_multiplications += 1
    q_low_to_high = [
        characteristic[DIMENSION - degree]
        for degree in range(DIMENSION)
    ]
    for degree in range(2 * DIMENSION - 2, DIMENSION - 1, -1):
        factor = product[degree]
        if factor == cyclo.ring_zero():
            continue
        for q_degree, q_coefficient in enumerate(q_low_to_high):
            target_degree = degree - DIMENSION + q_degree
            product[target_degree] = cyclo.ring_subtract(
                product[target_degree],
                cyclo.ring_multiply(factor, q_coefficient),
            )
            stats.polynomial_ring_multiplications += 1
            stats.polynomial_reduction_updates += 1
        product[degree] = cyclo.ring_zero()
    stats.polynomial_multiplications += 1
    return product[:DIMENSION]


def recurrence_coefficients(
    periods: int,
    characteristic: list[RingElement],
    stats: RecurrenceStats,
) -> list[RingElement]:
    if periods < 1:
        fail("declared recurrence requires at least one period")
    exponent = periods - 1
    result = polynomial_one()
    base = polynomial_x()
    while exponent:
        if exponent & 1:
            result = multiply_mod_q(
                result,
                base,
                characteristic,
                stats,
            )
        exponent >>= 1
        if exponent:
            base = multiply_mod_q(
                base,
                base,
                characteristic,
                stats,
            )
    return result


@dataclass
class Carrier:
    messages: list[RingVector]
    coefficients: list[RingElement]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"

    @classmethod
    def create(cls) -> "Carrier":
        return cls(
            messages=[
                cyclo.zero_vector()
                for _ in range(MESSAGE_SLOTS)
            ],
            coefficients=polynomial_zero(),
        )

    def all_zero(self) -> bool:
        return (
            all(cyclo.vector_is_zero(row) for row in self.messages)
            and all(
                value == cyclo.ring_zero()
                for value in self.coefficients
            )
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.messages),
            *(id(message) for message in self.messages),
            id(self.coefficients),
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "message_slots": len(self.messages),
            "coefficient_registers": len(self.coefficients),
            "all_state_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
        }


def record_peak(carrier: Carrier, stats: RecurrenceStats) -> None:
    payload = 0
    maximum_bits = 1
    nonzero_messages = 0
    nonzero_coefficients = 0
    for message in carrier.messages:
        if not cyclo.vector_is_zero(message):
            nonzero_messages += 1
        for element in message:
            for coefficient in element:
                bits = signed_bits(coefficient)
                payload += bits
                maximum_bits = max(maximum_bits, bits)
    for element in carrier.coefficients:
        if element != cyclo.ring_zero():
            nonzero_coefficients += 1
        for coefficient in element:
            bits = signed_bits(coefficient)
            payload += bits
            maximum_bits = max(maximum_bits, bits)
    stats.maximum_carrier_payload_bits = max(
        stats.maximum_carrier_payload_bits,
        payload,
    )
    stats.maximum_coefficient_signed_bits = max(
        stats.maximum_coefficient_signed_bits,
        maximum_bits,
    )
    stats.maximum_nonzero_message_slots = max(
        stats.maximum_nonzero_message_slots,
        nonzero_messages,
    )
    stats.maximum_nonzero_coefficient_registers = max(
        stats.maximum_nonzero_coefficient_registers,
        nonzero_coefficients,
    )


def build_basis(
    carrier: Carrier,
    block: cyclo.CompiledBlock,
    stats: RecurrenceStats,
) -> None:
    block_stats = cyclo.Stats()
    seed = cyclo.seed_vector(block.public_program)
    cyclo.copy_vector_into(carrier.messages[0], seed)
    for basis_index in range(1, BASIS_MESSAGES + 1):
        cyclo.compute_block_into(
            carrier,
            block.operator,
            basis_index - 1,
            basis_index,
            block_stats,
            False,
        )
    stats.basis_forward_block_applications += (
        block_stats.forward_block_applications
    )
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    record_peak(carrier, stats)


def load_coefficients(
    carrier: Carrier,
    coefficients: list[RingElement],
) -> None:
    if any(value != cyclo.ring_zero() for value in carrier.coefficients):
        fail("coefficient registers were not clean")
    carrier.coefficients[:] = coefficients


def output_from_resident_basis(
    carrier: Carrier,
    stats: RecurrenceStats,
) -> RingVector:
    target = cyclo.zero_vector()
    for basis_index, scalar in enumerate(carrier.coefficients, start=1):
        for phase_index, value in enumerate(
            carrier.messages[basis_index]
        ):
            target[phase_index] = cyclo.ring_add(
                target[phase_index],
                cyclo.ring_multiply(scalar, value),
            )
            stats.output_ring_multiplications += 1
    return target


def populate_forward(
    carrier: Carrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, RecurrenceStats]:
    if carrier.active or carrier.pending_operations or not carrier.all_zero():
        fail("recurrence carrier was not restored")
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    carrier.phase = "BUILD_BASIS"
    stats = RecurrenceStats()
    build_basis(carrier, block, stats)
    coefficients = recurrence_coefficients(
        periods,
        block.characteristic,
        stats,
    )
    load_coefficients(carrier, coefficients)
    carrier.phase = "COEFFICIENTS_RESIDENT"
    output = output_from_resident_basis(carrier, stats)
    cyclo.copy_vector_into(carrier.messages[OUTPUT_SLOT], output)
    carrier.phase = "OUTPUT_RESIDENT"
    record_peak(carrier, stats)
    return cyclo.project_boundary(output), stats


def restore_forward(
    carrier: Carrier,
    block: cyclo.CompiledBlock,
    periods: int,
    stats: RecurrenceStats,
) -> None:
    if carrier.phase != "OUTPUT_RESIDENT":
        fail("recurrence inverse was reordered")
    expected_output = output_from_resident_basis(carrier, stats)
    cyclo.subtract_vector_exact(
        carrier.messages[OUTPUT_SLOT],
        expected_output,
    )
    carrier.phase = "COEFFICIENTS_RESIDENT"
    expected_coefficients = recurrence_coefficients(
        periods,
        block.characteristic,
        stats,
    )
    if carrier.coefficients != expected_coefficients:
        fail("recurrence inverse coefficient registers did not match")
    for index, value in enumerate(expected_coefficients):
        carrier.coefficients[index] = cyclo.ring_subtract(
            carrier.coefficients[index],
            value,
        )
    carrier.phase = "BASIS_RESIDENT"
    block_stats = cyclo.Stats()
    for basis_index in range(BASIS_MESSAGES, 0, -1):
        cyclo.uncompute_block_from(
            carrier,
            block.operator,
            basis_index - 1,
            basis_index,
            block_stats,
        )
    stats.basis_inverse_block_applications += (
        block_stats.inverse_block_applications
    )
    stats.basis_ring_multiply_accumulations += (
        block_stats.ring_multiply_accumulations
    )
    seed = cyclo.seed_vector(block.public_program)
    cyclo.subtract_vector_exact(carrier.messages[0], seed)
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.generation += 1
    record_peak(carrier, stats)
    if not carrier.all_zero():
        fail("recurrence carrier did not restore exactly")


@dataclass
class Transaction:
    boundary: RingElement
    stats: RecurrenceStats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: Carrier,
    block: cyclo.CompiledBlock,
    periods: int,
) -> Transaction:
    if not isinstance(carrier, Carrier):
        fail("null or invalid recurrence carrier")
    backing = carrier.backing_identity()
    boundary, stats = populate_forward(carrier, block, periods)
    restore_forward(carrier, block, periods, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def dense_boundary(
    block: cyclo.CompiledBlock,
    periods: int,
) -> tuple[RingElement, dict[str, int]]:
    vector = cyclo.seed_vector(block.public_program)
    stats = cyclo.Stats()
    maximum_payload_bits = 0
    maximum_coefficient_signed_bits = 1
    for _ in range(periods):
        target = cyclo.apply_operator(block.operator, vector, stats)
        vector_payload_bits = sum(
            signed_bits(coefficient)
            for element in vector
            for coefficient in element
        )
        target_payload_bits = sum(
            signed_bits(coefficient)
            for element in target
            for coefficient in element
        )
        maximum_payload_bits = max(
            maximum_payload_bits,
            vector_payload_bits + target_payload_bits,
        )
        maximum_coefficient_signed_bits = max(
            maximum_coefficient_signed_bits,
            max(
                signed_bits(coefficient)
                for message in (vector, target)
                for element in message
                for coefficient in element
            ),
        )
        vector = target
    return cyclo.project_boundary(vector), {
        "ring_multiply_accumulations": (
            stats.ring_multiply_accumulations
        ),
        "maximum_two_message_payload_bits": maximum_payload_bits,
        "maximum_coefficient_signed_bits": (
            maximum_coefficient_signed_bits
        ),
        "projection_ring_additions": PRIME,
    }


def stats_json(stats: RecurrenceStats) -> dict[str, int]:
    return {
        "polynomial_multiplications": stats.polynomial_multiplications,
        "polynomial_ring_multiplications": (
            stats.polynomial_ring_multiplications
        ),
        "polynomial_reduction_updates": (
            stats.polynomial_reduction_updates
        ),
        "output_ring_multiplications": stats.output_ring_multiplications,
        "basis_forward_block_applications": (
            stats.basis_forward_block_applications
        ),
        "basis_inverse_block_applications": (
            stats.basis_inverse_block_applications
        ),
        "basis_ring_multiply_accumulations": (
            stats.basis_ring_multiply_accumulations
        ),
        "maximum_carrier_payload_bits": (
            stats.maximum_carrier_payload_bits
        ),
        "maximum_coefficient_signed_bits": (
            stats.maximum_coefficient_signed_bits
        ),
        "maximum_nonzero_message_slots": (
            stats.maximum_nonzero_message_slots
        ),
        "maximum_nonzero_coefficient_registers": (
            stats.maximum_nonzero_coefficient_registers
        ),
    }


def case_result(
    periods: int,
    block: cyclo.CompiledBlock,
) -> dict[str, Any]:
    carrier = Carrier.create()
    transaction = execute_transaction(carrier, block, periods)
    direct, direct_stats = dense_boundary(block, periods)
    return {
        "periods": periods,
        "equivalent_edges": periods * cyclo.PERIOD,
        "family": block.family,
        "boundary": transaction.boundary,
        "boundary_sha256": hashlib.sha256(
            cyclo.encoded_ring_object(transaction.boundary)
        ).hexdigest(),
        "boundary_payload_bits": sum(
            signed_bits(value)
            for value in transaction.boundary
        ),
        "dense_direct_boundary_equal": transaction.boundary == direct,
        "dense_direct_stats": direct_stats,
        "message_slots": MESSAGE_SLOTS,
        "message_integer_cells": (
            MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
        ),
        "coefficient_register_integer_cells": (
            COEFFICIENT_INTEGER_CELLS
        ),
        "stats": stats_json(transaction.stats),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
    }


def restoration_reuse_case(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, Any]:
    periods = max(TESTED_PERIODS)
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(
        carrier,
        primary,
        periods,
    )
    reuse_transaction = execute_transaction(carrier, reuse, periods)
    fresh_transaction = execute_transaction(
        Carrier.create(),
        reuse,
        periods,
    )
    return {
        "periods": periods,
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
        "canonical_restored_state": carrier.canonical_state(),
        "retained_forward_inverse_enabling_basis_integer_cells": (
            BASIS_MESSAGES * MESSAGE_INTEGER_CELLS
        ),
        "retained_seed_plus_basis_integer_cells": (
            (1 + BASIS_MESSAGES) * MESSAGE_INTEGER_CELLS
        ),
        "separate_inverse_operation_log_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def controls(
    primary: cyclo.CompiledBlock,
    reuse: cyclo.CompiledBlock,
) -> dict[str, bool]:
    missing = Carrier.create()
    populate_forward(missing, primary, 4)
    missing_inverse_leaves_nonzero = not missing.all_zero()

    wrong = Carrier.create()
    _, wrong_stats = populate_forward(wrong, primary, 4)
    wrong_inverse_rejected = False
    try:
        restore_forward(wrong, reuse, 4, wrong_stats)
    except RuntimeError:
        wrong_inverse_rejected = True

    reordered = Carrier.create()
    _, reordered_stats = populate_forward(reordered, primary, 4)
    reordered.phase = "BASIS_RESIDENT"
    reordered_inverse_rejected = False
    try:
        restore_forward(reordered, primary, 4, reordered_stats)
    except RuntimeError:
        reordered_inverse_rejected = True

    null_carrier_rejected = False
    try:
        execute_transaction(None, primary, 4)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True

    primary_boundary = execute_transaction(
        Carrier.create(),
        primary,
        4,
    ).boundary
    reuse_boundary = execute_transaction(
        Carrier.create(),
        reuse,
        4,
    ).boundary
    return {
        "missing_inverse_leaves_nonzero": (
            missing_inverse_leaves_nonzero
        ),
        "wrong_inverse_rejected": wrong_inverse_rejected,
        "reordered_inverse_rejected": reordered_inverse_rejected,
        "null_carrier_rejected": null_carrier_rejected,
        "semantic_family_perturbation_changes_boundary": (
            primary_boundary != reuse_boundary
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_cubic_chain_period17_executed_recurrence.py")
    blocks = {
        family.lower(): cyclo.build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    constant_coefficients_zero = all(
        block.characteristic[-1] == cyclo.ring_zero()
        for block in blocks.values()
    )
    native_k_monic_factor_constant_zero = {
        family: block.characteristic[DIMENSION] == cyclo.ring_zero()
        for family, block in blocks.items()
    }
    cases = [
        case_result(periods, blocks[family])
        for periods in TESTED_PERIODS
        for family in ("primary", "reuse")
    ]
    restored = restoration_reuse_case(
        blocks["primary"],
        blocks["reuse"],
    )
    control_results = controls(blocks["primary"], blocks["reuse"])
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_PERIOD17_CUBIC_CHAIN_EXECUTED_"
            "NATIVE_Q_ZETA17_CAYLEY_HAMILTON_RECURRENCE_USES_"
            "FIXED_18_RESIDENT_MESSAGE_SLOTS_PLUS16_CYCLOTOMIC_"
            "COEFFICIENT_REGISTERS_ACROSS_PERIODS1_4_16_64_"
            "WITH_EXACT_RESTORATION_AND_REUSE_BUT_GROWING_"
            "INTEGER_WIDTH_AND_IDENTICAL_CLASSICAL_EXECUTION"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_"
            "CUBIC_PATH_FAMILIES_Q_ZETA17_CHARACTERISTIC_"
            "RECURRENCE_PERIODS1_4_16_64_FIXED18_RESIDENT_"
            "MESSAGE_SLOTS_PLUS16_CYCLOTOMIC_COEFFICIENT_"
            "REGISTERS_DIRECT_DENSE_BOUNDARY_PARITY_EXACT_"
            "SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY"
        ),
        "classification_candidate": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level_candidate": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "coefficient_field": "Q_ZETA17",
        "integral_carrier_ring": "Z_ZETA17",
        "characteristic_order": PRIME,
        "characteristic_constant_coefficients_zero": (
            constant_coefficients_zero
        ),
        "native_k_monic_factor_degree": DIMENSION,
        "native_k_recurrence_basis_messages": DIMENSION,
        "native_k_monic_factor_constant_zero": (
            native_k_monic_factor_constant_zero
        ),
        "native_k_monic_factor_integral_unit_certified": {
            "primary": False,
            "reuse": False,
        },
        "scalar_q_order16_established": False,
        "reversible_integral_rolling_window_established": False,
        "fixed_resident_basis_message_bank_executed": True,
        "runtime_executes_characteristic_recurrence": True,
        "runtime_dense_block_forward_applications_per_transaction": (
            BASIS_MESSAGES
        ),
        "runtime_dense_block_inverse_applications_per_transaction": (
            BASIS_MESSAGES
        ),
        "runtime_dense_block_total_applications_per_transaction": (
            2 * BASIS_MESSAGES
        ),
        "block_certificates": {
            family: {
                "public_program_sha256": hashlib.sha256(
                    cyclo.adaptive.encoded_program(block.public_program)
                ).hexdigest(),
                "operator_sha256": block.operator_sha256,
                "characteristic_sha256": (
                    block.characteristic_sha256
                ),
                "characteristic": block.characteristic,
            }
            for family, block in blocks.items()
        },
        "tested_periods": list(TESTED_PERIODS),
        "cases": cases,
        "all_dense_direct_boundaries_equal": all(
            case["dense_direct_boundary_equal"]
            for case in cases
        ),
        "all_cases_restored_exactly": all(
            case["restored_exactly"]
            and case["same_backing"]
            and case["canonical_restored_state"]["all_state_zero"]
            for case in cases
        ),
        "restoration_reuse_case": restored,
        "controls": control_results,
        "matched_classical": {
            "identical_cyclotomic_characteristic_recurrence": True,
            "identical_fixed_basis_contraction": True,
            "identical_recurrence_lifecycle_and_restoration_available": (
                True
            ),
            "direct_streaming_dense_block_message_integer_cells": (
                2 * MESSAGE_INTEGER_CELLS
            ),
            "direct_streaming_dense_block_work_linear_in_periods": True,
            "direct_streaming_dense_block_comparison_is_forward_only": (
                True
            ),
            "direct_streaming_dense_block_restoration_classification": (
                "NO_RESTORATION_CLAIM"
            ),
            "polynomial_multiplication_count_is_logarithmic_in_periods_after_basis": True,
            "growing_integer_bit_operation_work_bounded": False,
            "comparison_establishes_advantage": False,
            "strongest_family_specific_method_established": False,
        },
        "resource_law": {
            "accounting_scope": (
                "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_"
                "EXACT_PROCESS_PEAK"
            ),
            "fixed_message_slots": MESSAGE_SLOTS,
            "fixed_message_slots_are_not_fixed_total_footprint": True,
            "fixed_message_bank_integer_cells": (
                MESSAGE_SLOTS * MESSAGE_INTEGER_CELLS
            ),
            "coefficient_register_integer_cells": (
                COEFFICIENT_INTEGER_CELLS
            ),
            "compiled_operator_integer_cells_per_family": (
                cyclo.OPERATOR_INTEGER_CELLS
            ),
            "compiled_characteristic_integer_cells_per_family": (
                (PRIME + 1) * DIMENSION
            ),
            "compiled_operator_integer_cells_two_families": (
                2 * cyclo.OPERATOR_INTEGER_CELLS
            ),
            "compiled_characteristic_integer_cells_two_families": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "both_compiled_families_retained_during_full_run": True,
            "retained_forward_inverse_enabling_basis_integer_cells_per_transaction": (
                BASIS_MESSAGES * MESSAGE_INTEGER_CELLS
            ),
            "retained_seed_plus_basis_integer_cells_per_transaction": (
                (1 + BASIS_MESSAGES) * MESSAGE_INTEGER_CELLS
            ),
            "temporary_polynomial_product_integer_cells": (
                (2 * DIMENSION - 1) * DIMENSION
            ),
            "temporary_ring_convolution_integer_cells": (
                2 * DIMENSION - 1
            ),
            "temporary_output_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "temporary_inverse_expected_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "projection_integer_cells": DIMENSION,
            "separate_inverse_operation_log_bytes": 0,
            "baseline_reload_bytes": 0,
            "integer_width_counted": True,
            "polynomial_operation_counts_recorded": True,
            "dense_direct_baseline_work_recorded": True,
            "dense_direct_baseline_width_and_payload_recorded": True,
            "dense_direct_projection_counted": True,
            "named_logical_cell_accounting_is_exact_total": False,
            "sympy_characteristic_internal_temporaries_bounded": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "REVERSIBLE_INTEGRAL_ROLLING_WINDOW",
            "SCALAR_Q_ORDER17_RECURRENCE",
            "FIXED_INTEGER_WIDTH",
            "CONSTANT_TOTAL_BIT_STORAGE",
            "STRONGEST_FAMILY_SPECIFIC_COMPACT_BASELINE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": (
            "EXECUTING_THE_NATIVE_Q_ZETA17_RECURRENCE_FIXES_"
            "RESIDENT_SLOT_AND_REGISTER_COUNTS_IN_THE_TESTED_"
            "MECHANISM_BUT_INTEGER_WIDTH_GROWS_AN_INTEGRALLY_"
            "REVERSIBLE_ROLLING_WINDOW_IS_NOT_ESTABLISHED_AND_"
            "COMPACT_CLASSICAL_SOFTWARE_EXECUTES_THE_IDENTICAL_"
            "RECURRENCE"
        ),
        "terminal": False,
    }
    if (
        not constant_coefficients_zero
        or not result["all_dense_direct_boundaries_equal"]
        or not result["all_cases_restored_exactly"]
        or not all(control_results.values())
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("executed recurrence qualification condition failed")
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
