#!/usr/bin/env python3
"""Exact period-17 block closure over the native cyclotomic module.

The earlier modular diagnostic views a 17-vector over Q(zeta_17) as 272
rational coordinates.  This successor keeps the native 17-state
Q(zeta_17)-module structure explicit.  One public period compiles to a
17-by-17 cyclotomic matrix, whose exact characteristic polynomial gives an
order-at-most-17 recurrence over the native phase field.

The software carrier uses integer coefficients in the canonical
Z[zeta_17] basis.  Reversible pebbling applies the fixed public block without
retaining inverse history, then exact subtraction restores the borrowed
carrier for an unrelated second family.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

from sympy import QQ
from sympy.polys.matrices import DomainMatrix

import f17_cubic_chain_adaptive_gauge as adaptive


PRIME = 17
DIMENSION = 16
PERIOD = 17
MESSAGE_INTEGER_CELLS = PRIME * DIMENSION
OPERATOR_INTEGER_CELLS = PRIME * PRIME * DIMENSION
TESTED_PERIODS = (1, 2, 4, 8)

RingElement = tuple[int, ...]
RingVector = list[RingElement]
RingMatrix = list[list[RingElement]]


def fail(message: str) -> None:
    raise RuntimeError(message)


def signed_bits(value: int) -> int:
    if value == 0:
        return 1
    return abs(value).bit_length() + 1


def ring_zero() -> RingElement:
    return (0,) * DIMENSION


def ring_one() -> RingElement:
    return (1,) + (0,) * (DIMENSION - 1)


def ring_add(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def ring_subtract(left: RingElement, right: RingElement) -> RingElement:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def ring_multiply(left: RingElement, right: RingElement) -> RingElement:
    work = [0] * (2 * DIMENSION - 1)
    for left_degree, left_value in enumerate(left):
        if left_value == 0:
            continue
        for right_degree, right_value in enumerate(right):
            if right_value:
                work[left_degree + right_degree] += (
                    left_value * right_value
                )
    for degree in range(len(work) - 1, DIMENSION - 1, -1):
        value = work[degree]
        if value == 0:
            continue
        offset = degree - DIMENSION
        for reduced_degree in range(offset, offset + DIMENSION):
            work[reduced_degree] -= value
        work[degree] = 0
    return tuple(work[:DIMENSION])


def ring_monomial(exponent: int) -> RingElement:
    exponent %= PRIME
    if exponent < DIMENSION:
        result = [0] * DIMENSION
        result[exponent] = 1
        return tuple(result)
    return (-1,) * DIMENSION


def ring_identity_matrix() -> RingMatrix:
    return [
        [
            ring_one() if row == column else ring_zero()
            for column in range(PRIME)
        ]
        for row in range(PRIME)
    ]


def ring_matrix_multiply(
    left: RingMatrix,
    right: RingMatrix,
) -> RingMatrix:
    target = [
        [ring_zero() for _ in range(PRIME)]
        for _ in range(PRIME)
    ]
    for row in range(PRIME):
        for column in range(PRIME):
            accumulator = ring_zero()
            for shared in range(PRIME):
                accumulator = ring_add(
                    accumulator,
                    ring_multiply(
                        left[row][shared],
                        right[shared][column],
                    ),
                )
            target[row][column] = accumulator
    return target


def check_annihilator(
    operator: RingMatrix,
    characteristic: list[RingElement],
) -> bool:
    residual = ring_identity_matrix()
    for coefficient in characteristic[1:]:
        residual = ring_matrix_multiply(operator, residual)
        for diagonal in range(PRIME):
            residual[diagonal][diagonal] = ring_add(
                residual[diagonal][diagonal],
                coefficient,
            )
    return all(
        element == ring_zero()
        for row in residual
        for element in row
    )


def ring_element_from_domain(element: Any) -> RingElement:
    descending = list(element.to_list())
    integers: list[int] = []
    for value in descending:
        if int(value.denominator) != 1:
            fail("cyclotomic domain element was not integral")
        integers.append(int(value.numerator))
    if len(integers) > DIMENSION:
        fail("cyclotomic domain element exceeded canonical degree")
    integers = [0] * (DIMENSION - len(integers)) + integers
    return tuple(reversed(integers))


def encoded_ring_object(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass
class CompiledBlock:
    family: str
    public_program: adaptive.ChainProgram
    operator: RingMatrix
    characteristic: list[RingElement]
    operator_sha256: str
    characteristic_sha256: str
    characteristic_identity_exact: bool


def build_compiled_block(family: str) -> CompiledBlock:
    program = adaptive.compile_program(PERIOD + 1, family)
    field = QQ.cyclotomic_field(PRIME)
    zeta = field.convert(field.ext)
    powers = [field.one]
    for _ in range(1, PRIME):
        powers.append(powers[-1] * zeta)
    identity_rows = [
        [
            field.one if row == column else field.zero
            for column in range(PRIME)
        ]
        for row in range(PRIME)
    ]
    operator_domain = DomainMatrix(
        identity_rows,
        (PRIME, PRIME),
        field,
    )
    for edge_index in range(PERIOD):
        edge_rows = []
        for right in range(PRIME):
            row = []
            for left in range(PRIME):
                shift = (
                    adaptive.unary_phase(
                        program.unary_coefficients[edge_index + 1],
                        right,
                    )
                    + adaptive.edge_phase(
                        program.edge_coefficients[edge_index],
                        left,
                        right,
                    )
                ) % PRIME
                row.append(powers[shift])
            edge_rows.append(row)
        edge_matrix = DomainMatrix(
            edge_rows,
            (PRIME, PRIME),
            field,
        )
        operator_domain = edge_matrix * operator_domain
    operator = [
        [
            ring_element_from_domain(element)
            for element in row
        ]
        for row in operator_domain.to_list()
    ]
    characteristic_domain = operator_domain.charpoly()
    characteristic = [
        ring_element_from_domain(element)
        for element in characteristic_domain
    ]
    operator_bytes = encoded_ring_object(operator)
    characteristic_bytes = encoded_ring_object(characteristic)
    return CompiledBlock(
        family=family,
        public_program=program,
        operator=operator,
        characteristic=characteristic,
        operator_sha256=hashlib.sha256(operator_bytes).hexdigest(),
        characteristic_sha256=hashlib.sha256(
            characteristic_bytes
        ).hexdigest(),
        characteristic_identity_exact=check_annihilator(
            operator,
            characteristic,
        ),
    )


def seed_vector(program: adaptive.ChainProgram) -> RingVector:
    return [
        ring_monomial(
            adaptive.unary_phase(
                program.unary_coefficients[0],
                value,
            )
        )
        for value in range(PRIME)
    ]


@dataclass
class Stats:
    forward_block_applications: int = 0
    inverse_block_applications: int = 0
    ring_multiply_accumulations: int = 0
    maximum_carrier_payload_bits: int = 0
    maximum_coefficient_signed_bits: int = 1
    maximum_nonzero_slots: int = 0


def zero_vector() -> RingVector:
    return [ring_zero() for _ in range(PRIME)]


def vector_is_zero(vector: RingVector) -> bool:
    return all(element == ring_zero() for element in vector)


def vector_equals(left: RingVector, right: RingVector) -> bool:
    return left == right


def apply_operator(
    operator: RingMatrix,
    source: RingVector,
    stats: Stats,
) -> RingVector:
    target = zero_vector()
    for right in range(PRIME):
        accumulator = ring_zero()
        for left in range(PRIME):
            accumulator = ring_add(
                accumulator,
                ring_multiply(
                    operator[right][left],
                    source[left],
                ),
            )
            stats.ring_multiply_accumulations += 1
        target[right] = accumulator
    return target


def copy_vector_into(target: RingVector, source: RingVector) -> None:
    if not vector_is_zero(target):
        fail("cyclotomic target was not clean")
    for index, element in enumerate(source):
        target[index] = element


def subtract_vector_exact(
    target: RingVector,
    expected: RingVector,
) -> None:
    for index in range(PRIME):
        target[index] = ring_subtract(target[index], expected[index])
    if not vector_is_zero(target):
        fail("exact cyclotomic subtraction left a nonzero target")


@dataclass
class Carrier:
    capacity_periods: int
    messages: list[RingVector]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0

    @classmethod
    def create(cls, periods: int) -> "Carrier":
        if periods < 1 or periods & (periods - 1):
            fail("declared block carrier requires a power-of-two period count")
        return cls(
            capacity_periods=periods,
            messages=[
                zero_vector()
                for _ in range(adaptive.slot_count(periods + 1))
            ],
        )

    def all_zero(self) -> bool:
        return all(vector_is_zero(message) for message in self.messages)

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.messages),
            *(id(message) for message in self.messages),
        )

    def canonical_restored_state(self) -> dict[str, Any]:
        return {
            "capacity_periods": self.capacity_periods,
            "message_slots": len(self.messages),
            "all_messages_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
        }


def record_peak(carrier: Carrier, stats: Stats) -> None:
    payload = 0
    maximum_bits = 1
    nonzero_slots = 0
    for message in carrier.messages:
        if not vector_is_zero(message):
            nonzero_slots += 1
        for element in message:
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
    stats.maximum_nonzero_slots = max(
        stats.maximum_nonzero_slots,
        nonzero_slots,
    )


def compute_block_into(
    carrier: Carrier,
    operator: RingMatrix,
    source_slot: int,
    target_slot: int,
    stats: Stats,
    inverse_accounting: bool,
) -> None:
    if not vector_is_zero(carrier.messages[target_slot]):
        fail("block target was not clean")
    expected = apply_operator(
        operator,
        carrier.messages[source_slot],
        stats,
    )
    copy_vector_into(carrier.messages[target_slot], expected)
    if inverse_accounting:
        stats.inverse_block_applications += 1
    else:
        stats.forward_block_applications += 1
    record_peak(carrier, stats)


def uncompute_block_from(
    carrier: Carrier,
    operator: RingMatrix,
    source_slot: int,
    target_slot: int,
    stats: Stats,
) -> None:
    expected = apply_operator(
        operator,
        carrier.messages[source_slot],
        stats,
    )
    if not vector_equals(carrier.messages[target_slot], expected):
        fail("inverse block did not match the resident target")
    subtract_vector_exact(carrier.messages[target_slot], expected)
    stats.inverse_block_applications += 1
    record_peak(carrier, stats)


def compute_segment(
    carrier: Carrier,
    operator: RingMatrix,
    first: int,
    last: int,
    source_slot: int,
    target_slot: int,
    scratch_slots: tuple[int, ...],
    stats: Stats,
) -> None:
    count = last - first
    if count == 1:
        compute_block_into(
            carrier,
            operator,
            source_slot,
            target_slot,
            stats,
            False,
        )
        return
    if count < 1 or not scratch_slots:
        fail("invalid forward block segment")
    middle = first + count // 2
    middle_slot = scratch_slots[0]
    remaining = scratch_slots[1:]
    compute_segment(
        carrier,
        operator,
        first,
        middle,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )
    compute_segment(
        carrier,
        operator,
        middle,
        last,
        middle_slot,
        target_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        operator,
        first,
        middle,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )


def uncompute_segment(
    carrier: Carrier,
    operator: RingMatrix,
    first: int,
    last: int,
    source_slot: int,
    target_slot: int,
    scratch_slots: tuple[int, ...],
    stats: Stats,
) -> None:
    count = last - first
    if count == 1:
        uncompute_block_from(
            carrier,
            operator,
            source_slot,
            target_slot,
            stats,
        )
        return
    if count < 1 or not scratch_slots:
        fail("invalid inverse block segment")
    middle = first + count // 2
    middle_slot = scratch_slots[0]
    remaining = scratch_slots[1:]
    compute_segment(
        carrier,
        operator,
        first,
        middle,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        operator,
        middle,
        last,
        middle_slot,
        target_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        operator,
        first,
        middle,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )


def project_boundary(message: RingVector) -> RingElement:
    result = ring_zero()
    for element in message:
        result = ring_add(result, element)
    return result


@dataclass
class Transaction:
    boundary: RingElement
    stats: Stats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: Carrier,
    block: CompiledBlock,
) -> Transaction:
    if not isinstance(carrier, Carrier):
        fail("null or invalid cyclotomic block carrier")
    if carrier.active or carrier.pending_operations:
        fail("carrier already has an active transaction")
    if not carrier.all_zero():
        fail("carrier was not restored")
    backing = carrier.backing_identity()
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    stats = Stats()
    seed = seed_vector(block.public_program)
    copy_vector_into(carrier.messages[0], seed)
    record_peak(carrier, stats)
    scratch = tuple(range(2, len(carrier.messages)))
    compute_segment(
        carrier,
        block.operator,
        0,
        carrier.capacity_periods,
        0,
        1,
        scratch,
        stats,
    )
    boundary = project_boundary(carrier.messages[1])
    uncompute_segment(
        carrier,
        block.operator,
        0,
        carrier.capacity_periods,
        0,
        1,
        scratch,
        stats,
    )
    if not vector_equals(carrier.messages[0], seed):
        fail("seed changed before exact release")
    subtract_vector_exact(carrier.messages[0], seed)
    carrier.pending_operations = 0
    restored = carrier.all_zero()
    same_backing = carrier.backing_identity() == backing
    if not restored or not same_backing:
        fail("cyclotomic block carrier did not restore")
    carrier.generation += 1
    carrier.active = False
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=restored,
        same_backing=same_backing,
    )


def stats_json(stats: Stats) -> dict[str, Any]:
    return {
        "forward_block_applications": stats.forward_block_applications,
        "inverse_block_applications": stats.inverse_block_applications,
        "ring_multiply_accumulations": stats.ring_multiply_accumulations,
        "maximum_carrier_payload_bits": (
            stats.maximum_carrier_payload_bits
        ),
        "maximum_coefficient_signed_bits": (
            stats.maximum_coefficient_signed_bits
        ),
        "maximum_nonzero_slots": stats.maximum_nonzero_slots,
    }


def block_summary(block: CompiledBlock) -> dict[str, Any]:
    operator_payload = sum(
        signed_bits(coefficient)
        for row in block.operator
        for element in row
        for coefficient in element
    )
    characteristic_payload = sum(
        signed_bits(coefficient)
        for element in block.characteristic
        for coefficient in element
    )
    return {
        "family": block.family,
        "public_program_sha256": hashlib.sha256(
            adaptive.encoded_program(block.public_program)
        ).hexdigest(),
        "public_program_descriptor_bytes": len(
            adaptive.encoded_program(block.public_program)
        ),
        "operator_sha256": block.operator_sha256,
        "operator_integer_cells": OPERATOR_INTEGER_CELLS,
        "operator_payload_bits": operator_payload,
        "operator_maximum_coefficient_signed_bits": max(
            signed_bits(coefficient)
            for row in block.operator
            for element in row
            for coefficient in element
        ),
        "characteristic_sha256": block.characteristic_sha256,
        "characteristic_order": len(block.characteristic) - 1,
        "characteristic_integer_cells": (
            len(block.characteristic) * DIMENSION
        ),
        "characteristic_payload_bits": characteristic_payload,
        "characteristic_maximum_coefficient_signed_bits": max(
            signed_bits(coefficient)
            for element in block.characteristic
            for coefficient in element
        ),
        "characteristic_monic": (
            block.characteristic[0] == ring_one()
        ),
        "characteristic_identity_exact": (
            block.characteristic_identity_exact
        ),
        "operator": block.operator,
        "characteristic": block.characteristic,
    }


def case_result(
    periods: int,
    block: CompiledBlock,
) -> dict[str, Any]:
    carrier = Carrier.create(periods)
    transaction = execute_transaction(carrier, block)
    return {
        "periods": periods,
        "equivalent_edges": periods * PERIOD,
        "equivalent_nodes": periods * PERIOD + 1,
        "family": block.family,
        "boundary": transaction.boundary,
        "boundary_sha256": hashlib.sha256(
            encoded_ring_object(transaction.boundary)
        ).hexdigest(),
        "boundary_payload_bits": sum(
            signed_bits(value)
            for value in transaction.boundary
        ),
        "message_slots": len(carrier.messages),
        "carrier_integer_cells": (
            len(carrier.messages) * MESSAGE_INTEGER_CELLS
        ),
        "stats": stats_json(transaction.stats),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_restored_state(),
    }


def restoration_reuse_case(
    primary: CompiledBlock,
    reuse: CompiledBlock,
) -> dict[str, Any]:
    periods = max(TESTED_PERIODS)
    carrier = Carrier.create(periods)
    backing = carrier.backing_identity()
    primary_transaction = execute_transaction(carrier, primary)
    reuse_transaction = execute_transaction(carrier, reuse)
    fresh_transaction = execute_transaction(
        Carrier.create(periods),
        reuse,
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
        "canonical_restored_state": carrier.canonical_restored_state(),
        "primary_stats": stats_json(primary_transaction.stats),
        "reuse_stats": stats_json(reuse_transaction.stats),
        "fresh_reuse_stats": stats_json(fresh_transaction.stats),
        "retained_inverse_history_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def controls(
    primary: CompiledBlock,
    reuse: CompiledBlock,
) -> dict[str, Any]:
    carrier = Carrier.create(1)
    seed = seed_vector(primary.public_program)
    copy_vector_into(carrier.messages[0], seed)
    stats = Stats()
    compute_block_into(
        carrier,
        primary.operator,
        0,
        1,
        stats,
        False,
    )
    missing_inverse_leaves_nonzero = not carrier.all_zero()
    wrong_inverse_rejected = False
    try:
        uncompute_block_from(
            carrier,
            reuse.operator,
            0,
            1,
            stats,
        )
    except RuntimeError:
        wrong_inverse_rejected = True
    reordered_inverse_rejected = False
    reordered = Carrier.create(1)
    reordered_seed = seed_vector(primary.public_program)
    copy_vector_into(reordered.messages[0], reordered_seed)
    try:
        uncompute_block_from(
            reordered,
            primary.operator,
            0,
            1,
            Stats(),
        )
    except RuntimeError:
        reordered_inverse_rejected = True
    null_carrier_rejected = False
    try:
        execute_transaction(None, primary)  # type: ignore[arg-type]
    except RuntimeError:
        null_carrier_rejected = True
    primary_boundary = execute_transaction(
        Carrier.create(1),
        primary,
    ).boundary
    reuse_boundary = execute_transaction(
        Carrier.create(1),
        reuse,
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
        fail("usage: f17_cubic_chain_period17_cyclotomic_module.py")
    blocks = {
        family.lower(): build_compiled_block(family)
        for family in ("PRIMARY", "REUSE")
    }
    summaries = {
        family: block_summary(block)
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
    control_results = controls(
        blocks["primary"],
        blocks["reuse"],
    )
    all_characteristic_identities = all(
        summary["characteristic_identity_exact"]
        and summary["characteristic_monic"]
        and summary["characteristic_order"] == PRIME
        for summary in summaries.values()
    )
    all_cases_restored = all(
        case["restored_exactly"]
        and case["same_backing"]
        and case["canonical_restored_state"]["all_messages_zero"]
        for case in cases
    )
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_PERIOD17_CUBIC_CHAIN_NATIVE_"
            "CYCLOTOMIC_17_STATE_BLOCK_MODULE_HAS_ORDER_AT_MOST17_"
            "CAYLEY_HAMILTON_CLOSURE_WITH_EXACT_RESTORATION_AND_"
            "REUSE_BUT_RETAINS_GROWING_INTEGER_WIDTH"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_SYMPY_EXACT_TWO_PUBLIC_F17_"
            "PERIOD17_UNARY_CUBIC_AND_NEAREST_NEIGHBOR_MIXED_"
            "CUBIC_PATH_FAMILIES_17_STATE_Q_ZETA17_MODULE_"
            "PERIODS1_2_4_8_EXACT_CAYLEY_HAMILTON_AND_"
            "SUBTRACTIVE_RESTORATION_SOFTWARE_ONLY"
        ),
        "classification_candidate": (
            "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
        "verification_level_candidate": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "native_module": {
            "coefficient_field": "Q_ZETA17",
            "integral_carrier_ring": "Z_ZETA17",
            "phase_states": PRIME,
            "integer_coefficients_per_phase_state": DIMENSION,
            "message_integer_cells": MESSAGE_INTEGER_CELLS,
            "operator_shape": [PRIME, PRIME],
            "operator_integer_cells": OPERATOR_INTEGER_CELLS,
            "null_relation": "1+ZETA+...+ZETA^16=0",
        },
        "blocks": summaries,
        "all_characteristic_identities_exact": (
            all_characteristic_identities
        ),
        "exact_native_cyclotomic_recurrence_order_upper_bound": (
            PRIME
        ),
        "exact_native_minimal_order_established": False,
        "native_cyclotomic_recurrence_certified_not_executed": True,
        "runtime_executes_dense_17_by_17_cyclotomic_block": True,
        "prior_modular_q_rank_lower_bounds": {
            "primary": 241,
            "reuse": 256,
        },
        "prior_modular_dependencies_lifted_to_q_recurrence": False,
        "q_scalar_recurrence_order_upper_bound17_established": False,
        "restriction_of_scalars_dimension_per_k_coefficient": (
            DIMENSION
        ),
        "native_k_coefficient_maps_expand_to_16_by_16_q_linear_operators": (
            True
        ),
        "coefficient_ring_change_is_not_q_order_reduction": True,
        "coefficient_ring_changed_to_native_phase_field": True,
        "cases": cases,
        "all_cases_restored_exactly": all_cases_restored,
        "restoration_reuse_case": restored,
        "controls": control_results,
        "matched_classical": {
            "identical_17_state_cyclotomic_matrix_recurrence": True,
            "identical_characteristic_recurrence": True,
            "period_block_operator_integer_cells": (
                OPERATOR_INTEGER_CELLS
            ),
            "two_message_streaming_integer_cells": (
                2 * MESSAGE_INTEGER_CELLS
            ),
            "strongest_family_specific_method_established": False,
        },
        "resource_law": {
            "accounting_scope": (
                "COMPONENT_LEVEL_NAMED_LOGICAL_INTEGER_CELLS_NOT_"
                "EXACT_PROCESS_PEAK"
            ),
            "named_logical_cell_accounting_is_exact_total": False,
            "compiled_block_operators": 2,
            "compiled_block_operator_integer_cells": (
                2 * OPERATOR_INTEGER_CELLS
            ),
            "compiled_characteristic_integer_cells": (
                2 * (PRIME + 1) * DIMENSION
            ),
            "retained_operator_and_characteristic_integer_cells": (
                2
                * (
                    OPERATOR_INTEGER_CELLS
                    + (PRIME + 1) * DIMENSION
                )
            ),
            "public_program_descriptor_bytes": sum(
                summary["public_program_descriptor_bytes"]
                for summary in summaries.values()
            ),
            "maximum_carrier_integer_cells": max(
                case["carrier_integer_cells"]
                for case in cases
            ),
            "two_carrier_verification_integer_cells": (
                2
                * max(
                    case["carrier_integer_cells"]
                    for case in cases
                )
            ),
            "temporary_operator_product_integer_cells": (
                OPERATOR_INTEGER_CELLS
            ),
            "temporary_ring_convolution_integer_cells": (
                2 * DIMENSION - 1
            ),
            "temporary_expected_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "temporary_seed_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "boundary_integer_cells": DIMENSION,
            "sympy_operator_build_three_matrix_logical_cells": (
                3 * OPERATOR_INTEGER_CELLS
            ),
            "sympy_characteristic_three_matrix_plus_polynomial_logical_cells": (
                3 * OPERATOR_INTEGER_CELLS
                + (PRIME + 1) * DIMENSION
            ),
            "compilation_named_component_lower_bound_integer_cells": (
                3 * OPERATOR_INTEGER_CELLS
                + OPERATOR_INTEGER_CELLS
                + (PRIME + 1) * DIMENSION
            ),
            "execution_two_carrier_named_component_sum_integer_cells": (
                2
                * (
                    OPERATOR_INTEGER_CELLS
                    + (PRIME + 1) * DIMENSION
                )
                + 2
                * max(
                    case["carrier_integer_cells"]
                    for case in cases
                )
                + MESSAGE_INTEGER_CELLS
                + (2 * DIMENSION - 1)
                + DIMENSION
            ),
            "program_descriptors_counted": True,
            "projection_counted": True,
            "restoration_and_reuse_counted": True,
            "retained_inverse_history_bytes": 0,
            "baseline_reload_bytes": 0,
            "retained_domain_matrix_after_compilation": False,
            "sympy_domain_matrix_logical_storage_during_compilation_bounded": (
                False
            ),
            "sympy_domain_object_overhead_bounded": False,
            "sympy_characteristic_internal_temporaries_bounded": False,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "EXACT_NATIVE_MINIMAL_ORDER17",
            "Q_SCALAR_ORDER17_RECURRENCE",
            "EXACT_Q_RECURRENCE_LIFT_OF_MODULAR_DEPENDENCIES",
            "FIXED_INTEGER_WIDTH",
            "CONSTANT_TOTAL_REVERSIBLE_STORAGE",
            "ARBITRARY_GRAPH_TOPOLOGY",
            "GENERAL_NON_GAUSSIAN_COMPOSITION",
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
            "THE_NATIVE_Q_ZETA17_COEFFICIENT_ORDER_AT_MOST17_"
            "CLOSURE_DOES_NOT_LIFT_THE_PRIOR_Q_DEPENDENCIES_OR_"
            "ESTABLISH_A_SCALAR_Q_ORDER17_RECURRENCE_WHILE_"
            "INTEGER_WIDTH_STILL_GROWS_AND_THE_IDENTICAL_"
            "CYCLOTOMIC_MATRIX_EXECUTION_IS_CLASSICAL"
        ),
        "terminal": False,
    }
    if (
        not all_characteristic_identities
        or not all_cases_restored
        or not all(control_results.values())
        or not restored["primary_restored_exactly"]
        or not restored["reuse_restored_exactly"]
        or not restored["same_original_backing"]
        or not restored["fresh_restored_reuse_boundary_equal"]
    ):
        fail("native cyclotomic module qualification condition failed")
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
