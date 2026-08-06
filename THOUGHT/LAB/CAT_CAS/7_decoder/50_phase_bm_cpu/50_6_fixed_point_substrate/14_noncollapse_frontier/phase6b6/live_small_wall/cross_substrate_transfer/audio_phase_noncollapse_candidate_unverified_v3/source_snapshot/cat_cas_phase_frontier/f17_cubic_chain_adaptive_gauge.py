#!/usr/bin/env python3
"""Exact adaptive cyclotomic gauge and 17-content quotient on an F17 chain.

Each row stores an element of Z[zeta_17] in a 16-coefficient basis, but the
omitted root is selected independently and deterministically for that row.
Changing the omitted root only adds a multiple of
1 + zeta + ... + zeta**16 = 0, so the operation is exact.  The transfer acts
directly on this representation.  Exact common powers of 17 are divided from
each resident message and retained as an exponent in that message.  Neither
operation expands a global assignment table.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any


PRIME = 17
DIMENSION = 16
MESSAGE_INTEGER_CELLS = PRIME * DIMENSION
PIVOT_BITS_PER_MESSAGE = PRIME * 5
TESTED_NODES = (2, 3, 5, 9, 17, 33, 65)


def fail(message: str) -> None:
    raise RuntimeError(message)


def mod(value: int) -> int:
    return value % PRIME


def signed_bits(value: int) -> int:
    if value == 0:
        return 1
    return abs(value).bit_length() + 1


@dataclass(frozen=True)
class ChainProgram:
    family: str
    nodes: int
    unary_coefficients: tuple[tuple[int, int, int], ...]
    edge_coefficients: tuple[tuple[int, int, int], ...]


def compile_program(nodes: int, family: str) -> ChainProgram:
    if nodes < 2:
        fail("a chain program requires at least two nodes")
    if family not in {"PRIMARY", "REUSE"}:
        fail("unknown public program family")
    shift = 0 if family == "PRIMARY" else 7
    unary = tuple(
        (
            1 + mod(3 * index + shift) % 16,
            mod(5 * index + 2 + 2 * shift),
            mod(7 * index + 4 + shift),
        )
        for index in range(nodes)
    )
    edges = tuple(
        (
            1 + mod(5 * index + 2 + shift) % 16,
            1 + mod(7 * index + 4 + 2 * shift) % 16,
            mod(3 * index + 6 + shift),
        )
        for index in range(nodes - 1)
    )
    return ChainProgram(
        family=family,
        nodes=nodes,
        unary_coefficients=unary,
        edge_coefficients=edges,
    )


def program_json(program: ChainProgram) -> dict[str, Any]:
    return {
        "field": "F17",
        "phase_root_order": PRIME,
        "topology": "PUBLIC_PATH_GRAPH",
        "family": program.family,
        "nodes": program.nodes,
        "unary_coefficients": [
            list(coefficients)
            for coefficients in program.unary_coefficients
        ],
        "edge_coefficients": [
            list(coefficients)
            for coefficients in program.edge_coefficients
        ],
    }


def encoded_program(program: ChainProgram) -> bytes:
    return json.dumps(
        program_json(program),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def unary_phase(
    coefficients: tuple[int, int, int],
    value: int,
) -> int:
    cubic, quadratic, linear = coefficients
    return mod(
        cubic * value**3
        + quadratic * value**2
        + linear * value
    )


def edge_phase(
    coefficients: tuple[int, int, int],
    left: int,
    right: int,
) -> int:
    left_square_right, left_right_square, bilinear = coefficients
    return mod(
        left_square_right * left * left * right
        + left_right_square * left * right * right
        + bilinear * left * right
    )


def slot_count(nodes: int) -> int:
    return math.ceil(math.log2(nodes - 1)) + 2


def pebble_applications(edges: int) -> int:
    if edges < 1:
        fail("pebble segment must contain an edge")
    if edges == 1:
        return 1
    left = edges // 2
    return (
        2 * pebble_applications(left)
        + pebble_applications(edges - left)
    )


def coefficient_indices(pivot: int) -> tuple[int, ...]:
    return tuple(index for index in range(PRIME) if index != pivot)


def choose_row_gauge(
    redundant: list[int],
) -> tuple[int, list[int]]:
    if len(redundant) != PRIME:
        fail("cyclotomic row must have 17 redundant coefficients")
    best: tuple[int, int] | None = None
    for pivot in range(PRIME):
        reference = redundant[pivot]
        coefficients = [
            redundant[index] - reference
            for index in range(PRIME)
            if index != pivot
        ]
        cost = sum(signed_bits(value) for value in coefficients)
        candidate = (cost, pivot)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        fail("adaptive gauge selection failed")
    # Release the final trial list before reconstructing the selected row;
    # only one 16-cell gauge candidate is resident at a time.
    del coefficients
    pivot = best[1]
    reference = redundant[pivot]
    return pivot, [
        redundant[index] - reference
        for index in range(PRIME)
        if index != pivot
    ]


def canonical_row(
    pivot: int,
    coefficients: list[int],
) -> list[int]:
    if not 0 <= pivot < PRIME or len(coefficients) != DIMENSION:
        fail("invalid adaptive cyclotomic row")
    redundant = [0] * PRIME
    for index, coefficient in zip(
        coefficient_indices(pivot),
        coefficients,
        strict=True,
    ):
        redundant[index] = coefficient
    reference = redundant[16]
    return [
        redundant[index] - reference
        for index in range(DIMENSION)
    ]


@dataclass
class GaugeMessage:
    pivots: list[int]
    coefficients: list[list[int]]
    scale_17_exponent: int = 0

    @classmethod
    def create(cls) -> "GaugeMessage":
        return cls(
            pivots=[16 for _ in range(PRIME)],
            coefficients=[
                [0 for _ in range(DIMENSION)]
                for _ in range(PRIME)
            ],
            scale_17_exponent=0,
        )

    def is_zero(self) -> bool:
        return self.scale_17_exponent == 0 and all(
            pivot == 16
            and all(value == 0 for value in row)
            for pivot, row in zip(
                self.pivots,
                self.coefficients,
                strict=True,
            )
        )

    def backing_identity(self) -> tuple[int, ...]:
        return (
            id(self.pivots),
            id(self.coefficients),
            *(id(row) for row in self.coefficients),
        )

    def canonical_quotient(self) -> list[list[int]]:
        return [
            canonical_row(pivot, row)
            for pivot, row in zip(
                self.pivots,
                self.coefficients,
                strict=True,
            )
        ]

    def canonical_semantic(self) -> list[list[int]]:
        scale = PRIME**self.scale_17_exponent
        return [
            [scale * value for value in row]
            for row in self.canonical_quotient()
        ]

    def equals(self, other: "GaugeMessage") -> bool:
        return (
            self.scale_17_exponent == other.scale_17_exponent
            and self.pivots == other.pivots
            and self.coefficients == other.coefficients
        )


def write_row(
    message: GaugeMessage,
    value: int,
    redundant: list[int],
) -> None:
    pivot, coefficients = choose_row_gauge(redundant)
    message.pivots[value] = pivot
    target = message.coefficients[value]
    for index, coefficient in enumerate(coefficients):
        target[index] = coefficient


def integer_valuation(value: int, prime: int) -> int:
    if value == 0:
        return sys.maxsize
    result = 0
    magnitude = abs(value)
    while magnitude % prime == 0:
        magnitude //= prime
        result += 1
    return result


def extract_17_content(message: GaugeMessage) -> int:
    valuation = min(
        integer_valuation(value, PRIME)
        for row in message.coefficients
        for value in row
        if value != 0
    )
    if valuation == sys.maxsize:
        fail("cannot extract content from a zero message")
    if valuation:
        divisor = PRIME**valuation
        for row in message.coefficients:
            for index, value in enumerate(row):
                if value % divisor:
                    fail("declared 17-content did not divide message")
                row[index] = value // divisor
    message.scale_17_exponent += valuation
    return valuation


def regauge_message(message: GaugeMessage) -> None:
    for value in range(PRIME):
        canonical = canonical_row(
            message.pivots[value],
            message.coefficients[value],
        )
        write_row(message, value, canonical + [0])


def seed_message(program: ChainProgram) -> GaugeMessage:
    message = GaugeMessage.create()
    for value in range(PRIME):
        redundant = [0] * PRIME
        phase = unary_phase(program.unary_coefficients[0], value)
        redundant[phase] = 1
        write_row(message, value, redundant)
    return message


@dataclass
class Stats:
    forward_transfer_applications: int = 0
    inverse_transfer_applications: int = 0
    transfer_scalar_accumulations: int = 0
    projection_scalar_accumulations: int = 0
    maximum_adaptive_payload_bits: int = 0
    maximum_fixed_basis_payload_bits: int = 0
    maximum_adaptive_coefficient_signed_bits: int = 1
    maximum_fixed_basis_coefficient_signed_bits: int = 1
    maximum_nonzero_slots: int = 0
    full_carrier_scans: int = 0


@dataclass
class Carrier:
    capacity_nodes: int
    messages: list[GaugeMessage]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0

    @classmethod
    def create(cls, nodes: int) -> "Carrier":
        return cls(
            capacity_nodes=nodes,
            messages=[
                GaugeMessage.create()
                for _ in range(slot_count(nodes))
            ],
        )

    def all_zero(self) -> bool:
        return all(message.is_zero() for message in self.messages)

    def backing_identity(self) -> tuple[int, ...]:
        result = [id(self.messages)]
        for message in self.messages:
            result.extend(message.backing_identity())
        return tuple(result)

    def canonical_restored_state(self) -> dict[str, Any]:
        return {
            "capacity_nodes": self.capacity_nodes,
            "message_slots": len(self.messages),
            "all_messages_zero": self.all_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
        }


def record_peak(carrier: Carrier, stats: Stats) -> None:
    stats.full_carrier_scans += 1
    adaptive_bits = 0
    fixed_bits = 0
    nonzero_slots = 0
    for message in carrier.messages:
        if not message.is_zero():
            nonzero_slots += 1
        adaptive_bits += PIVOT_BITS_PER_MESSAGE
        adaptive_bits += max(
            1,
            message.scale_17_exponent.bit_length(),
        )
        semantic_scale = PRIME**message.scale_17_exponent
        for pivot, row in zip(
            message.pivots,
            message.coefficients,
            strict=True,
        ):
            for coefficient in row:
                bits = signed_bits(coefficient)
                adaptive_bits += bits
                stats.maximum_adaptive_coefficient_signed_bits = max(
                    stats.maximum_adaptive_coefficient_signed_bits,
                    bits,
                )
            fixed = [
                semantic_scale * value
                for value in canonical_row(pivot, row)
            ]
            for coefficient in fixed:
                bits = signed_bits(coefficient)
                fixed_bits += bits
                stats.maximum_fixed_basis_coefficient_signed_bits = max(
                    stats.maximum_fixed_basis_coefficient_signed_bits,
                    bits,
                )
    stats.maximum_adaptive_payload_bits = max(
        stats.maximum_adaptive_payload_bits,
        adaptive_bits,
    )
    stats.maximum_fixed_basis_payload_bits = max(
        stats.maximum_fixed_basis_payload_bits,
        fixed_bits,
    )
    stats.maximum_nonzero_slots = max(
        stats.maximum_nonzero_slots,
        nonzero_slots,
    )


def copy_message_into(
    target: GaugeMessage,
    source: GaugeMessage,
) -> None:
    if not target.is_zero():
        fail("target message was not clean")
    target.scale_17_exponent = source.scale_17_exponent
    for value in range(PRIME):
        target.pivots[value] = source.pivots[value]
        for basis in range(DIMENSION):
            target.coefficients[value][basis] = (
                source.coefficients[value][basis]
            )


def subtract_message_exact(
    target: GaugeMessage,
    expected: GaugeMessage,
) -> None:
    if target.scale_17_exponent != expected.scale_17_exponent:
        fail("inverse content exponent did not match")
    target.scale_17_exponent -= expected.scale_17_exponent
    for value in range(PRIME):
        if target.pivots[value] != expected.pivots[value]:
            fail("inverse adaptive gauge did not match")
        for basis in range(DIMENSION):
            target.coefficients[value][basis] -= (
                expected.coefficients[value][basis]
            )
        # Pivot 16 is the exact zero-message identity.  The offset subtraction
        # is the inverse metadata operation on the 17-value pivot register.
        target.pivots[value] = (
            target.pivots[value]
            - expected.pivots[value]
            + 16
        ) % PRIME
    if not target.is_zero():
        fail("exact subtractive inverse left a nonzero message")


def transfer_row(
    source: GaugeMessage,
    program: ChainProgram,
    edge_index: int,
    right: int,
    stats: Stats,
) -> list[int]:
    redundant = [0] * PRIME
    unary = program.unary_coefficients[edge_index + 1]
    edge = program.edge_coefficients[edge_index]
    for left in range(PRIME):
        shift = mod(
            unary_phase(unary, right)
            + edge_phase(edge, left, right)
        )
        pivot = source.pivots[left]
        for exponent, coefficient in zip(
            coefficient_indices(pivot),
            source.coefficients[left],
            strict=True,
        ):
            redundant[mod(exponent + shift)] += coefficient
            stats.transfer_scalar_accumulations += 1
    return redundant


def compute_transfer_into(
    source: GaugeMessage,
    target: GaugeMessage,
    program: ChainProgram,
    edge_index: int,
    stats: Stats,
    inverse_accounting: bool,
) -> None:
    if not target.is_zero():
        fail("transfer target was not clean")
    for right in range(PRIME):
        write_row(
            target,
            right,
            transfer_row(source, program, edge_index, right, stats),
        )
    inherited = source.scale_17_exponent
    extract_17_content(target)
    target.scale_17_exponent += inherited
    regauge_message(target)
    if inverse_accounting:
        stats.inverse_transfer_applications += 1
    else:
        stats.forward_transfer_applications += 1


def uncompute_transfer_from(
    source: GaugeMessage,
    target: GaugeMessage,
    program: ChainProgram,
    edge_index: int,
    stats: Stats,
) -> None:
    expected = GaugeMessage.create()
    compute_transfer_into(
        source,
        expected,
        program,
        edge_index,
        stats,
        True,
    )
    if not target.equals(expected):
        fail("inverse transfer did not match the resident target")
    subtract_message_exact(target, expected)


def compute_segment(
    carrier: Carrier,
    program: ChainProgram,
    first_edge: int,
    last_edge: int,
    source_slot: int,
    target_slot: int,
    scratch_slots: tuple[int, ...],
    stats: Stats,
) -> None:
    edge_count = last_edge - first_edge
    if edge_count < 1:
        fail("empty forward segment")
    if edge_count == 1:
        compute_transfer_into(
            carrier.messages[source_slot],
            carrier.messages[target_slot],
            program,
            first_edge,
            stats,
            False,
        )
        record_peak(carrier, stats)
        return
    if not scratch_slots:
        fail("insufficient forward pebble slots")
    middle_edge = first_edge + edge_count // 2
    middle_slot = scratch_slots[0]
    remaining = scratch_slots[1:]
    compute_segment(
        carrier,
        program,
        first_edge,
        middle_edge,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )
    compute_segment(
        carrier,
        program,
        middle_edge,
        last_edge,
        middle_slot,
        target_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        program,
        first_edge,
        middle_edge,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )


def uncompute_segment(
    carrier: Carrier,
    program: ChainProgram,
    first_edge: int,
    last_edge: int,
    source_slot: int,
    target_slot: int,
    scratch_slots: tuple[int, ...],
    stats: Stats,
) -> None:
    edge_count = last_edge - first_edge
    if edge_count < 1:
        fail("empty inverse segment")
    if edge_count == 1:
        uncompute_transfer_from(
            carrier.messages[source_slot],
            carrier.messages[target_slot],
            program,
            first_edge,
            stats,
        )
        record_peak(carrier, stats)
        return
    if not scratch_slots:
        fail("insufficient inverse pebble slots")
    middle_edge = first_edge + edge_count // 2
    middle_slot = scratch_slots[0]
    remaining = scratch_slots[1:]
    compute_segment(
        carrier,
        program,
        first_edge,
        middle_edge,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        program,
        middle_edge,
        last_edge,
        middle_slot,
        target_slot,
        remaining,
        stats,
    )
    uncompute_segment(
        carrier,
        program,
        first_edge,
        middle_edge,
        source_slot,
        middle_slot,
        remaining,
        stats,
    )


def project_boundary(
    message: GaugeMessage,
    nodes: int,
    stats: Stats,
) -> dict[str, Any]:
    redundant = [0] * PRIME
    for pivot, row in zip(
        message.pivots,
        message.coefficients,
        strict=True,
    ):
        for exponent, coefficient in zip(
            coefficient_indices(pivot),
            row,
            strict=True,
        ):
            redundant[exponent] += coefficient
            stats.projection_scalar_accumulations += 1
    canonical_unfactored = [
        redundant[index] - redundant[16]
        for index in range(DIMENSION)
    ]
    boundary_content = min(
        integer_valuation(value, PRIME)
        for value in canonical_unfactored
        if value != 0
    )
    if boundary_content == sys.maxsize:
        fail("zero final boundary is outside the declared cases")
    if boundary_content:
        divisor = PRIME**boundary_content
        canonical_unfactored = [
            value // divisor
            for value in canonical_unfactored
        ]
    pivot, coefficients = choose_row_gauge(
        canonical_unfactored + [0]
    )
    total_content = (
        message.scale_17_exponent + boundary_content
    )
    canonical_quotient = canonical_row(pivot, coefficients)
    semantic_scale = PRIME**total_content
    canonical_semantic = [
        semantic_scale * value
        for value in canonical_quotient
    ]
    effective_sqrt_power = nodes - 2 * total_content
    return {
        "root_order": PRIME,
        "normalization_denominator_base": PRIME,
        "original_normalization_denominator_sqrt_power": nodes,
        "content_17_exponent": total_content,
        "effective_normalization_denominator_sqrt_power": (
            effective_sqrt_power
        ),
        "adaptive_omitted_root": pivot,
        "adaptive_cyclotomic_coefficients": coefficients,
        "adaptive_payload_bits": (
            5
            + max(1, total_content.bit_length())
            + sum(signed_bits(value) for value in coefficients)
        ),
        "canonical_quotient_coefficients": canonical_quotient,
        "canonical_quotient_payload_bits": sum(
            signed_bits(value)
            for value in canonical_quotient
        ),
        "verification_reconstructed_canonical_coefficients": (
            canonical_semantic
        ),
        "unfactored_canonical_payload_bits": sum(
            signed_bits(value)
            for value in canonical_semantic
        ),
    }


@dataclass
class Transaction:
    boundary: dict[str, Any]
    stats: Stats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: Carrier,
    program: ChainProgram,
) -> Transaction:
    if not isinstance(carrier, Carrier):
        fail("null or invalid adaptive gauge carrier")
    if carrier.capacity_nodes != program.nodes:
        fail("program topology mismatches carrier capacity")
    if carrier.active or carrier.pending_operations:
        fail("carrier already has an active transaction")
    if not carrier.all_zero():
        fail("carrier was not restored")
    backing = carrier.backing_identity()
    carrier.active = True
    carrier.lease += 1
    carrier.pending_operations = 1
    stats = Stats()
    seed = seed_message(program)
    copy_message_into(carrier.messages[0], seed)
    record_peak(carrier, stats)
    scratch = tuple(range(2, len(carrier.messages)))
    compute_segment(
        carrier,
        program,
        0,
        program.nodes - 1,
        0,
        1,
        scratch,
        stats,
    )
    boundary = project_boundary(
        carrier.messages[1],
        program.nodes,
        stats,
    )
    uncompute_segment(
        carrier,
        program,
        0,
        program.nodes - 1,
        0,
        1,
        scratch,
        stats,
    )
    if not carrier.messages[0].equals(seed):
        fail("seed changed before inverse release")
    subtract_message_exact(carrier.messages[0], seed)
    carrier.pending_operations = 0
    restored = carrier.all_zero()
    same_backing = carrier.backing_identity() == backing
    if not restored or not same_backing:
        fail("adaptive carrier did not restore on original backing")
    carrier.generation += 1
    carrier.active = False
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=restored,
        same_backing=same_backing,
    )


def mutate_program(program: ChainProgram) -> ChainProgram:
    edges = list(program.edge_coefficients)
    changed = list(edges[0])
    changed[0] = mod(changed[0] + 1) or 1
    edges[0] = tuple(changed)
    return ChainProgram(
        family=f"{program.family}_PERTURBED",
        nodes=program.nodes,
        unary_coefficients=program.unary_coefficients,
        edge_coefficients=tuple(edges),
    )


def inverse_control(program: ChainProgram, mode: str) -> bool:
    carrier = Carrier.create(program.nodes)
    seed = seed_message(program)
    copy_message_into(carrier.messages[0], seed)
    stats = Stats()
    scratch = tuple(range(2, len(carrier.messages)))
    if mode == "REORDERED":
        try:
            uncompute_segment(
                carrier,
                program,
                0,
                program.nodes - 1,
                0,
                1,
                scratch,
                stats,
            )
        except RuntimeError:
            return True
        return False
    compute_segment(
        carrier,
        program,
        0,
        program.nodes - 1,
        0,
        1,
        scratch,
        stats,
    )
    if mode == "MISSING":
        return not carrier.all_zero()
    try:
        uncompute_segment(
            carrier,
            mutate_program(program) if mode == "WRONG" else program,
            0,
            program.nodes - 1,
            0,
            1,
            scratch,
            stats,
        )
    except RuntimeError:
        return True
    return False


def divide_one_minus_zeta(
    row: list[int],
) -> list[int] | None:
    total = sum(row)
    if total % PRIME:
        return None
    last = total // PRIME
    quotient: list[int] = []
    prefix = 0
    for index, coefficient in enumerate(row):
        prefix += coefficient
        quotient.append(prefix - (index + 1) * last)
    return quotient


def content_diagnostic(message: GaugeMessage) -> dict[str, Any]:
    canonical = message.canonical_semantic()
    raw_bits = sum(
        signed_bits(value)
        for row in canonical
        for value in row
    )
    common_integer = 0
    for row in canonical:
        for value in row:
            common_integer = math.gcd(common_integer, abs(value))
    quotient = [row[:] for row in canonical]
    valuation = 0
    while True:
        divided = [divide_one_minus_zeta(row) for row in quotient]
        if any(row is None for row in divided):
            break
        quotient = [
            row
            for row in divided
            if row is not None
        ]
        valuation += 1
    quotient_bits = sum(
        signed_bits(value)
        for row in quotient
        for value in row
    )
    valuation_bits = max(1, valuation.bit_length())
    return {
        "fixed_basis_payload_bits": raw_bits,
        "common_integer_content": common_integer,
        "stored_17_content_exponent": message.scale_17_exponent,
        "one_minus_zeta_valuation": valuation,
        "one_minus_zeta_factored_payload_bits": (
            quotient_bits + valuation_bits
        ),
        "one_minus_zeta_factorization_reduces_payload": (
            quotient_bits + valuation_bits < raw_bits
        ),
    }


def streaming_final(
    program: ChainProgram,
) -> GaugeMessage:
    current = seed_message(program)
    stats = Stats()
    for edge_index in range(program.nodes - 1):
        target = GaugeMessage.create()
        compute_transfer_into(
            current,
            target,
            program,
            edge_index,
            stats,
            False,
        )
        current = target
    return current


def stats_json(stats: Stats) -> dict[str, Any]:
    return {
        "forward_transfer_applications": (
            stats.forward_transfer_applications
        ),
        "inverse_transfer_applications": (
            stats.inverse_transfer_applications
        ),
        "transfer_scalar_accumulations": (
            stats.transfer_scalar_accumulations
        ),
        "projection_scalar_accumulations": (
            stats.projection_scalar_accumulations
        ),
        "maximum_adaptive_payload_bits": (
            stats.maximum_adaptive_payload_bits
        ),
        "maximum_fixed_basis_payload_bits": (
            stats.maximum_fixed_basis_payload_bits
        ),
        "maximum_adaptive_coefficient_signed_bits": (
            stats.maximum_adaptive_coefficient_signed_bits
        ),
        "maximum_fixed_basis_coefficient_signed_bits": (
            stats.maximum_fixed_basis_coefficient_signed_bits
        ),
        "maximum_nonzero_slots": stats.maximum_nonzero_slots,
        "full_carrier_scans": stats.full_carrier_scans,
    }


def periodic_baseline(nodes: int) -> dict[str, Any]:
    edges = nodes - 1
    block = 17
    complete_blocks = edges // block
    tail = edges % block
    dense_block_build_transfer_equivalents = (
        block * MESSAGE_INTEGER_CELLS
    )
    return {
        "public_transfer_period": block,
        "complete_blocks": complete_blocks,
        "tail_transfers": tail,
        "streaming_transfer_applications": edges,
        "dense_block_integer_cells": (
            MESSAGE_INTEGER_CELLS**2
        ),
        "dense_block_build_transfer_equivalents": (
            dense_block_build_transfer_equivalents
        ),
        "dense_block_build_exceeds_streaming_at_case": (
            dense_block_build_transfer_equivalents > edges
        ),
        "powering_executed": False,
        "reason": (
            "SINGLE_QUERY_BUILD_COST_EXCEEDS_STREAMING_FOR_ALL_"
            "DECLARED_CASES"
        ),
    }


def run_case(nodes: int) -> dict[str, Any]:
    primary_program = compile_program(nodes, "PRIMARY")
    reuse_program = compile_program(nodes, "REUSE")
    carrier = Carrier.create(nodes)
    backing = carrier.backing_identity()
    primary = execute_transaction(carrier, primary_program)
    reuse = execute_transaction(carrier, reuse_program)
    fresh = execute_transaction(
        Carrier.create(nodes),
        reuse_program,
    )
    if reuse.boundary != fresh.boundary:
        fail("restored reuse boundary differs from fresh")
    if carrier.backing_identity() != backing:
        fail("reuse did not use the original backing")
    expected = pebble_applications(nodes - 1)
    for transaction in (primary, reuse, fresh):
        if (
            transaction.stats.forward_transfer_applications != expected
            or transaction.stats.inverse_transfer_applications != expected
        ):
            fail("pebble application law mismatch")
    primary_content = content_diagnostic(
        streaming_final(primary_program)
    )
    reuse_content = content_diagnostic(
        streaming_final(reuse_program)
    )
    primary_descriptor_bytes = len(encoded_program(primary_program))
    reuse_descriptor_bytes = len(encoded_program(reuse_program))
    primary_boundary_bytes = len(
        json.dumps(
            primary.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reuse_boundary_bytes = len(
        json.dumps(
            reuse.boundary,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "nodes": nodes,
        "edges": nodes - 1,
        "primary_public_program": program_json(primary_program),
        "reuse_public_program": program_json(reuse_program),
        "primary_program_sha256": hashlib.sha256(
            encoded_program(primary_program)
        ).hexdigest(),
        "reuse_program_sha256": hashlib.sha256(
            encoded_program(reuse_program)
        ).hexdigest(),
        "primary_program_descriptor_bytes": primary_descriptor_bytes,
        "reuse_program_descriptor_bytes": reuse_descriptor_bytes,
        "primary_boundary_descriptor_bytes": primary_boundary_bytes,
        "reuse_boundary_descriptor_bytes": reuse_boundary_bytes,
        "primary_boundary": primary.boundary,
        "reuse_boundary": reuse.boundary,
        "primary_stats": stats_json(primary.stats),
        "reuse_stats": stats_json(reuse.stats),
        "primary_final_content_diagnostic": primary_content,
        "reuse_final_content_diagnostic": reuse_content,
        "message_slots": len(carrier.messages),
        "message_integer_cells": (
            len(carrier.messages) * MESSAGE_INTEGER_CELLS
        ),
        "message_pivot_metadata_bits": (
            len(carrier.messages) * PIVOT_BITS_PER_MESSAGE
        ),
        "pebble_forward_applications": expected,
        "restored_exactly": (
            primary.restored_exactly
            and reuse.restored_exactly
        ),
        "same_original_backing": (
            primary.same_backing
            and reuse.same_backing
            and carrier.backing_identity() == backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse.boundary == fresh.boundary
        ),
        "restoration_generation": carrier.generation,
        "restoration_lease": carrier.lease,
        "canonical_restored_state": carrier.canonical_restored_state(),
        "periodic_block_baseline": periodic_baseline(nodes),
    }


def controls() -> dict[str, Any]:
    program = compile_program(3, "PRIMARY")
    null_rejected = False
    try:
        execute_transaction(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True
    primary = execute_transaction(
        Carrier.create(3),
        program,
    ).boundary
    perturbed_program = mutate_program(program)
    perturbed = execute_transaction(
        Carrier.create(3),
        perturbed_program,
    ).boundary
    return {
        "missing_inverse_rejected": inverse_control(
            program,
            "MISSING",
        ),
        "wrong_inverse_rejected": inverse_control(
            program,
            "WRONG",
        ),
        "reordered_inverse_rejected": inverse_control(
            program,
            "REORDERED",
        ),
        "null_carrier_rejected": null_rejected,
        "semantic_edge_perturbation_changes_boundary": (
            primary != perturbed
        ),
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_cubic_chain_adaptive_gauge.py")
    cases = [run_case(nodes) for nodes in TESTED_NODES]
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_EXACT_F17_ADAPTIVE_OMITTED_ROOT_AND_17_CONTENT_"
            "CYCLOTOMIC_QUOTIENT_CHAIN_TRANSFER_REDUCES_INTEGER_"
            "PAYLOAD_WITH_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS_"
            "DEPTH_GROWING_WIDTH"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_F17_PUBLIC_PATH_GRAPH_"
            "UNARY_CUBIC_AND_NEAREST_NEIGHBOR_MIXED_CUBIC_FACTORS_"
            "NODES2_3_5_9_17_33_65_TWO_PROGRAM_FAMILIES_PER_ROW_"
            "ADAPTIVE_OMITTED_ROOT_AND_EXACT_COMMON_17_CONTENT_"
            "Z_ZETA17_QUOTIENT_TOPOLOGY_DERIVED_REVERSIBLE_"
            "RECURSIVE_PEBBLING_SOFTWARE_ONLY"
        ),
        "classification_candidate": (
            "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
        "verification_level_candidate": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "restoration_mechanism": {
            "transfer_inverse": (
                "RECOMPUTE_EXPECTED_TRANSFER_THEN_EXACTLY_SUBTRACT_"
                "COEFFICIENTS_FROM_RESIDENT_TARGET"
            ),
            "pivot_metadata_inverse": (
                "OFFSET_SUBTRACTION_MOD17_WITH_ZERO_IDENTITY_16"
            ),
            "content_exponent_inverse": "EXACT_INTEGER_SUBTRACTION",
            "seed_inverse": (
                "EXACT_COEFFICIENT_PIVOT_AND_CONTENT_SUBTRACTION"
            ),
            "validated_destructive_erasure_used": False,
        },
        "representation": {
            "ring": "Z[ZETA17]",
            "exact_null_relation": (
                "1+ZETA+...+ZETA^16=0"
            ),
            "integer_cells_per_message": MESSAGE_INTEGER_CELLS,
            "pivot_metadata_bits_per_message": (
                PIVOT_BITS_PER_MESSAGE
            ),
            "pivot_selection": (
                "MINIMUM_SIGNED_BIT_PAYLOAD_THEN_LOWEST_ROOT"
            ),
            "content_quotient": (
                "MAXIMAL_COMMON_INTEGER_POWER_OF_17_PER_MESSAGE"
            ),
            "content_exponent_resident_per_message": True,
            "root_table_materialized": False,
            "assignment_table_materialized": False,
            "relation_table_materialized": False,
        },
        "cases": cases,
        "controls": controls(),
        "observed_law": {
            "tested_nodes": list(TESTED_NODES),
            "adaptive_peak_payload_bits": [
                max(
                    case["primary_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                    case["reuse_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                )
                for case in cases
            ],
            "content_quotient_peak_payload_bits": [
                max(
                    case["primary_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                    case["reuse_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                )
                for case in cases
            ],
            "fixed_basis_peak_payload_bits": [
                max(
                    case["primary_stats"][
                        "maximum_fixed_basis_payload_bits"
                    ],
                    case["reuse_stats"][
                        "maximum_fixed_basis_payload_bits"
                    ],
                )
                for case in cases
            ],
            "adaptive_peak_coefficient_signed_bits": [
                max(
                    case["primary_stats"][
                        "maximum_adaptive_coefficient_signed_bits"
                    ],
                    case["reuse_stats"][
                        "maximum_adaptive_coefficient_signed_bits"
                    ],
                )
                for case in cases
            ],
            "fixed_basis_peak_coefficient_signed_bits": [
                max(
                    case["primary_stats"][
                        "maximum_fixed_basis_coefficient_signed_bits"
                    ],
                    case["reuse_stats"][
                        "maximum_fixed_basis_coefficient_signed_bits"
                    ],
                )
                for case in cases
            ],
            "final_common_integer_content": [
                max(
                    case["primary_final_content_diagnostic"][
                        "common_integer_content"
                    ],
                    case["reuse_final_content_diagnostic"][
                        "common_integer_content"
                    ],
                )
                for case in cases
            ],
            "final_stored_17_content_exponents": [
                max(
                    case["primary_final_content_diagnostic"][
                        "stored_17_content_exponent"
                    ],
                    case["reuse_final_content_diagnostic"][
                        "stored_17_content_exponent"
                    ],
                )
                for case in cases
            ],
            "final_stored_17_content_exponents_match_floor_edges_over_3": (
                all(
                    max(
                        case["primary_final_content_diagnostic"][
                            "stored_17_content_exponent"
                        ],
                        case["reuse_final_content_diagnostic"][
                            "stored_17_content_exponent"
                        ],
                    )
                    == (case["nodes"] - 1) // 3
                    for case in cases
                )
            ),
            "boundary_17_content_exponents": [
                max(
                    case["primary_boundary"][
                        "content_17_exponent"
                    ],
                    case["reuse_boundary"][
                        "content_17_exponent"
                    ],
                )
                for case in cases
            ],
            "final_one_minus_zeta_valuations": [
                max(
                    case["primary_final_content_diagnostic"][
                        "one_minus_zeta_valuation"
                    ],
                    case["reuse_final_content_diagnostic"][
                        "one_minus_zeta_valuation"
                    ],
                )
                for case in cases
            ],
            "one_minus_zeta_factorization_reduces_any_final_payload": (
                any(
                    case["primary_final_content_diagnostic"][
                        "one_minus_zeta_factorization_reduces_payload"
                    ]
                    or case["reuse_final_content_diagnostic"][
                        "one_minus_zeta_factorization_reduces_payload"
                    ]
                    for case in cases
                )
            ),
            "fixed_integer_width_established": False,
            "constant_reversible_storage_established": False,
            "content_quotient_reduces_depth65_peak_payload": (
                max(
                    cases[-1]["primary_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                    cases[-1]["reuse_stats"][
                        "maximum_adaptive_payload_bits"
                    ],
                )
                < max(
                    cases[-1]["primary_stats"][
                        "maximum_fixed_basis_payload_bits"
                    ],
                    cases[-1]["reuse_stats"][
                        "maximum_fixed_basis_payload_bits"
                    ],
                )
            ),
        },
        "matched_classical": {
            "identical_adaptive_gauge_and_content_recurrence": True,
            "two_message_integer_cells": (
                2 * MESSAGE_INTEGER_CELLS
            ),
            "two_message_pivot_metadata_bits": (
                2 * PIVOT_BITS_PER_MESSAGE
            ),
            "periodic_transfer_period": 17,
            "dense_block_powering_applicability_gated": True,
            "dense_block_powering_executed": False,
            "strongest_family_specific_method_established": False,
        },
        "resource_law": {
            "accepted_carriers": 1,
            "fresh_reuse_verification_carriers": 1,
            "verification_peak_carriers": 2,
            "temporary_transfer_redundant_integer_cells": PRIME,
            "temporary_gauge_candidate_integer_cells": DIMENSION,
            "temporary_regauge_canonical_integer_cells": DIMENSION,
            "temporary_regauge_combined_peak_integer_cells": (
                DIMENSION + PRIME + DIMENSION
            ),
            "temporary_coefficient_index_integer_cells": DIMENSION,
            "temporary_seed_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "temporary_seed_pivot_metadata_bits": (
                PIVOT_BITS_PER_MESSAGE
            ),
            "temporary_inverse_expected_message_integer_cells": (
                MESSAGE_INTEGER_CELLS
            ),
            "temporary_inverse_expected_pivot_metadata_bits": (
                PIVOT_BITS_PER_MESSAGE
            ),
            "final_factorized_boundary_integer_cells": DIMENSION,
            "final_reconstructed_verification_boundary_integer_cells": (
                DIMENSION
            ),
            "verification_content_diagnostic_source_and_list_integer_cells": (
                4 * MESSAGE_INTEGER_CELLS
            ),
            "full_carrier_instrumentation_row_integer_cells": DIMENSION,
            "program_descriptors_counted": True,
            "projection_counted": True,
            "restoration_and_reuse_counted": True,
            "retained_inverse_history_bytes": 0,
            "baseline_reload_bytes": 0,
            "python_integer_object_overhead_bounded": False,
            "python_container_allocator_peak_bounded": False,
            "bit_operation_complexity_bounded": False,
            "recursive_call_stack_bytes_bounded": False,
            "whole_process_peak_bounded": False,
        },
        "not_established": [
            "FIXED_COEFFICIENT_WIDTH",
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
            "EXACT_17_CONTENT_QUOTIENT_REDUCES_THE_TESTED_CHAIN_"
            "INTEGER_PAYLOAD_BUT_RESIDUAL_COEFFICIENT_WIDTH_STILL_"
            "GROWS_WITH_DEPTH_AND_THE_IDENTICAL_CLASSICAL_"
            "RECURRENCE_INHERITS_THE_REDUCTION"
        ),
        "terminal": False,
    }
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
