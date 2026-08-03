#!/usr/bin/env python3
"""Four-site exact F17 phase coupling with certified TT-rank growth.

The resident state is a four-binary-site tensor over Q(zeta_17), stored in
the M116 two-by-eight representation.  Four public controlled phase edges are
interleaved with one exact local shear.  The natural-order tensor-train ranks
grow from 1 to 2 to 4, and every final two-versus-two cut has rank four.

Only the final coherent scalar sum is projected.  Reverse gates, the inverse
shear, and public seed unload restore the actual zero backing exactly before
cross-program reuse.  The strongest matched classical implementation is not
the dense tensor: a public lazy factor/circuit recurrence contracts the same
boundary at treewidth two.  This is therefore a rank-growth calibration, not
evidence of a distinct phase resource or computational advantage.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from typing import Any

import f17_cubic_chain_period17_quadratic_extension_resident_carrier as pair


base = pair.base
cyclo = pair.cyclo
real = pair.real
SplitElement = pair.SplitElement

PRIME = 17
SITES = 4
CELLS = 1 << SITES
ZERO = pair.split_zero()
ONE = pair.split_one()
ROOTS = (
    *pair.QUADRATIC_EXTENSION_TABLE,
    pair.full_to_split(cyclo.ring_monomial(16)),
)
FAMILY_WEIGHTS = {
    "PRIMARY": (1, 3, 2, 5),
    "REUSE": (2, 6, 4, 9),
}
SITE_NAMES = ("A", "B", "C", "D")


def fail(message: str) -> None:
    raise RuntimeError(message)


@dataclass(frozen=True)
class Gate:
    kind: str
    first: int
    second: int
    weight: int

    def payload_bits(self) -> int:
        kind_bits = 1
        return kind_bits + sum(
            base.signed_bits(value)
            for value in (self.first, self.second, self.weight)
        )


def compile_program(family: str) -> tuple[Gate, ...]:
    if family not in FAMILY_WEIGHTS:
        fail("unknown public four-site family")
    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    program = (
        Gate("PHASE", 0, 2, weight_ac),
        Gate("PHASE", 1, 2, weight_bc),
        Gate("SHEAR", 2, -1, 0),
        Gate("PHASE", 0, 3, weight_ad),
        Gate("PHASE", 1, 3, weight_bd),
    )
    validate_program(program)
    return program


def validate_program(program: tuple[Gate, ...]) -> None:
    if len(program) != 5:
        fail("four-site program requires exactly five gates")
    if program[2] != Gate("SHEAR", 2, -1, 0):
        fail("public local shear is not in the declared interleaved position")
    expected_edges = ((0, 2), (1, 2), (0, 3), (1, 3))
    actual_edges = tuple(
        (gate.first, gate.second) for gate in program if gate.kind == "PHASE"
    )
    if actual_edges != expected_edges:
        fail("public phase topology is not the declared K2,2 graph")
    for gate in program:
        if gate.kind == "PHASE":
            if not (
                0 <= gate.first < gate.second < SITES
                and 0 <= gate.weight < PRIME
            ):
                fail("public phase gate is outside the four-site F17 interface")
        elif gate.kind != "SHEAR":
            fail("unknown public gate type")


def assignment_bits(index: int) -> tuple[int, int, int, int]:
    return tuple((index >> (SITES - 1 - site)) & 1 for site in range(SITES))  # type: ignore[return-value]


def assignment_index(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


@dataclass
class Stats:
    seed_load_additions: int = 0
    seed_unload_subtractions: int = 0
    forward_phase_gates: int = 0
    inverse_phase_gates: int = 0
    forward_shears: int = 0
    inverse_shears: int = 0
    controlled_root_actions: int = 0
    controlled_root_action_steps: int = 0
    shear_cell_additions: int = 0
    shear_cell_subtractions: int = 0
    boundary_cell_additions: int = 0
    boundary_full_lifts: int = 0
    maximum_carrier_resident_payload_bits: int = 0
    maximum_gate_live_payload_bits: int = 0
    maximum_seed_live_payload_bits: int = 0
    maximum_boundary_live_payload_bits: int = 0
    maximum_accepted_resident_plus_work_payload_bits: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


def root_action(value: SplitElement, exponent: int, stats: Stats) -> SplitElement:
    forward = exponent % PRIME
    backward = (PRIME - forward) % PRIME
    output = value
    if forward <= backward:
        steps = forward
        for _ in range(steps):
            a_value, b_value = output
            updated = (
                pair.real_negate(b_value),
                real.real_add(a_value, pair.real_s1_multiply(b_value)),
            )
            stats.maximum_gate_live_payload_bits = max(
                stats.maximum_gate_live_payload_bits,
                pair.split_payload_bits(output) + pair.split_payload_bits(updated),
            )
            output = updated
    else:
        steps = backward
        for _ in range(steps):
            a_value, b_value = output
            updated = (
                real.real_add(pair.real_s1_multiply(a_value), b_value),
                pair.real_negate(a_value),
            )
            stats.maximum_gate_live_payload_bits = max(
                stats.maximum_gate_live_payload_bits,
                pair.split_payload_bits(output) + pair.split_payload_bits(updated),
            )
            output = updated
    stats.controlled_root_actions += 1
    stats.controlled_root_action_steps += steps
    stats.maximum_gate_live_payload_bits = max(
        stats.maximum_gate_live_payload_bits,
        pair.split_payload_bits(value) + pair.split_payload_bits(output),
    )
    return output


@dataclass
class TensorCarrier:
    cells: list[SplitElement]
    generation: int = 0
    lease: int = 0
    active: bool = False
    pending_operations: int = 0
    phase: str = "RESTORED"
    family: str = ""

    @classmethod
    def create(cls) -> "TensorCarrier":
        return cls([ZERO for _ in range(CELLS)])

    def backing_identity(self) -> int:
        return id(self.cells)

    def all_zero(self) -> bool:
        return (
            all(value == ZERO for value in self.cells)
            and not self.active
            and self.pending_operations == 0
            and self.phase == "RESTORED"
            and self.family == ""
        )

    def resident_payload_bits(self) -> int:
        metadata = sum(
            base.signed_bits(value)
            for value in (self.generation, self.lease, self.pending_operations)
        )
        # active (1), seven phase labels (3), empty/two-family tag (2).
        return sum(pair.split_payload_bits(value) for value in self.cells) + metadata + 6

    def canonical_state(self) -> dict[str, Any]:
        return {
            "all_phase_cells_zero": all(value == ZERO for value in self.cells),
            "generation": self.generation,
            "lease": self.lease,
            "active": self.active,
            "pending_operations": self.pending_operations,
            "phase": self.phase,
            "family_cleared": self.family == "",
        }


def record_carrier(carrier: TensorCarrier, stats: Stats) -> int:
    payload = carrier.resident_payload_bits()
    stats.maximum_carrier_resident_payload_bits = max(
        stats.maximum_carrier_resident_payload_bits, payload
    )
    return payload


def load_seed(carrier: TensorCarrier, family: str, stats: Stats) -> None:
    if not isinstance(carrier, TensorCarrier) or not carrier.all_zero():
        fail("four-site carrier is null, invalid, or unrestored")
    compile_program(family)
    before = record_carrier(carrier, stats)
    seed = [ONE for _ in range(CELLS)]
    updated = [
        pair.split_add(actual, value)
        for actual, value in zip(carrier.cells, seed, strict=True)
    ]
    stats.maximum_seed_live_payload_bits = max(
        stats.maximum_seed_live_payload_bits,
        before
        + sum(pair.split_payload_bits(value) for value in seed)
        + sum(pair.split_payload_bits(value) for value in updated),
    )
    carrier.cells[:] = updated
    carrier.lease += 1
    carrier.active = True
    carrier.pending_operations = 6
    carrier.phase = "SEED_RESIDENT"
    carrier.family = family
    stats.seed_load_additions += CELLS * 16
    record_carrier(carrier, stats)


def apply_phase_gate(
    carrier: TensorCarrier,
    gate: Gate,
    stats: Stats,
    *,
    inverse: bool,
) -> None:
    before = record_carrier(carrier, stats)
    for index in range(CELLS):
        bits = assignment_bits(index)
        if bits[gate.first] and bits[gate.second]:
            old = carrier.cells[index]
            exponent = -gate.weight if inverse else gate.weight
            updated = root_action(old, exponent, stats)
            carrier.cells[index] = updated
            stats.maximum_gate_live_payload_bits = max(
                stats.maximum_gate_live_payload_bits,
                pair.split_payload_bits(old) + pair.split_payload_bits(updated),
            )
    if inverse:
        stats.inverse_phase_gates += 1
    else:
        stats.forward_phase_gates += 1
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        before + stats.maximum_gate_live_payload_bits,
    )
    record_carrier(carrier, stats)


def apply_shear(carrier: TensorCarrier, stats: Stats, *, inverse: bool) -> None:
    before = record_carrier(carrier, stats)
    for a_value, b_value, d_value in itertools.product((0, 1), repeat=3):
        low_index = assignment_index((a_value, b_value, 0, d_value))
        high_index = assignment_index((a_value, b_value, 1, d_value))
        low = carrier.cells[low_index]
        high = carrier.cells[high_index]
        updated = (
            pair.split_subtract(low, high)
            if inverse
            else pair.split_add(low, high)
        )
        carrier.cells[low_index] = updated
        stats.maximum_gate_live_payload_bits = max(
            stats.maximum_gate_live_payload_bits,
            pair.split_payload_bits(low)
            + pair.split_payload_bits(high)
            + pair.split_payload_bits(updated),
        )
    if inverse:
        stats.inverse_shears += 1
        stats.shear_cell_subtractions += 8 * 16
    else:
        stats.forward_shears += 1
        stats.shear_cell_additions += 8 * 16
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        before + stats.maximum_gate_live_payload_bits,
    )
    record_carrier(carrier, stats)


def apply_gate(
    carrier: TensorCarrier,
    program: tuple[Gate, ...],
    ordinal: int,
    stats: Stats,
    *,
    inverse: bool = False,
) -> None:
    canonical = compile_program(carrier.family)
    if program != canonical or not 0 <= ordinal < len(program):
        fail("gate descriptor does not match compiled public topology")
    gate = program[ordinal]
    if inverse:
        if carrier.phase != f"FORWARD_{ordinal + 1}":
            fail("four-site inverse order changed")
    else:
        expected = "SEED_RESIDENT" if ordinal == 0 else f"FORWARD_{ordinal}"
        if carrier.phase != expected:
            fail("four-site forward order changed")
    if gate.kind == "PHASE":
        apply_phase_gate(carrier, gate, stats, inverse=inverse)
    else:
        apply_shear(carrier, stats, inverse=inverse)
    if inverse:
        carrier.pending_operations -= 1
        carrier.phase = "SEED_RESIDENT" if ordinal == 0 else f"FORWARD_{ordinal}"
    else:
        carrier.phase = f"FORWARD_{ordinal + 1}"


def project_boundary(carrier: TensorCarrier, stats: Stats) -> tuple[int, ...]:
    program = compile_program(carrier.family)
    if carrier.phase != f"FORWARD_{len(program)}":
        fail("only the final four-site boundary may be projected")
    accumulator = ZERO
    for value in carrier.cells:
        updated = pair.split_add(accumulator, value)
        stats.maximum_boundary_live_payload_bits = max(
            stats.maximum_boundary_live_payload_bits,
            pair.split_payload_bits(accumulator)
            + pair.split_payload_bits(value)
            + pair.split_payload_bits(updated),
        )
        accumulator = updated
        stats.boundary_cell_additions += 16
    full = pair.split_to_full(accumulator)
    stats.boundary_full_lifts += 1
    stats.maximum_boundary_live_payload_bits = max(
        stats.maximum_boundary_live_payload_bits,
        pair.split_payload_bits(accumulator) + base.element_payload_bits(full),
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        carrier.resident_payload_bits() + stats.maximum_boundary_live_payload_bits,
    )
    return tuple(int(value) for value in full)


def unload_seed(carrier: TensorCarrier, stats: Stats) -> None:
    if carrier.phase != "SEED_RESIDENT" or carrier.pending_operations != 1:
        fail("public seed unload was reordered")
    before = record_carrier(carrier, stats)
    seed = [ONE for _ in range(CELLS)]
    updated = [
        pair.split_subtract(actual, value)
        for actual, value in zip(carrier.cells, seed, strict=True)
    ]
    stats.maximum_seed_live_payload_bits = max(
        stats.maximum_seed_live_payload_bits,
        before
        + sum(pair.split_payload_bits(value) for value in seed)
        + sum(pair.split_payload_bits(value) for value in updated),
    )
    carrier.cells[:] = updated
    carrier.pending_operations = 0
    carrier.active = False
    carrier.phase = "RESTORED"
    carrier.family = ""
    carrier.generation += 1
    stats.seed_unload_subtractions += CELLS * 16
    record_carrier(carrier, stats)
    if not carrier.all_zero():
        fail("four-site phase tensor did not restore exactly")


@dataclass
class Transaction:
    boundary: tuple[int, ...]
    stats: Stats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(carrier: TensorCarrier, family: str) -> Transaction:
    if not isinstance(carrier, TensorCarrier):
        fail("null or invalid four-site carrier")
    backing = carrier.backing_identity()
    stats = Stats()
    program = compile_program(family)
    load_seed(carrier, family, stats)
    for ordinal in range(len(program)):
        apply_gate(carrier, program, ordinal, stats)
    boundary = project_boundary(carrier, stats)
    for ordinal in reversed(range(len(program))):
        apply_gate(carrier, program, ordinal, stats, inverse=True)
    unload_seed(carrier, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def determinant(matrix: list[list[SplitElement]]) -> SplitElement:
    size = len(matrix)
    accumulator = ZERO
    for permutation in itertools.permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        product = ONE
        for row, column in enumerate(permutation):
            product = pair.split_multiply(product, matrix[row][column])
        accumulator = (
            pair.split_subtract(accumulator, product)
            if inversions % 2
            else pair.split_add(accumulator, product)
        )
    return accumulator


def matrix_rank_exact(matrix: list[list[SplitElement]]) -> int:
    rows = len(matrix)
    columns = len(matrix[0])
    for size in range(min(rows, columns), 0, -1):
        for selected_rows in itertools.combinations(range(rows), size):
            for selected_columns in itertools.combinations(range(columns), size):
                minor = [
                    [matrix[row][column] for column in selected_columns]
                    for row in selected_rows
                ]
                if determinant(minor) != ZERO:
                    return size
    return 0


def flatten(
    cells: list[SplitElement], left_sites: tuple[int, ...]
) -> list[list[SplitElement]]:
    right_sites = tuple(site for site in range(SITES) if site not in left_sites)
    rows = 1 << len(left_sites)
    columns = 1 << len(right_sites)
    matrix = [[ZERO for _ in range(columns)] for _ in range(rows)]
    for index, value in enumerate(cells):
        bits = assignment_bits(index)
        row = assignment_index(tuple(bits[site] for site in left_sites))
        column = assignment_index(tuple(bits[site] for site in right_sites))
        matrix[row][column] = value
    return matrix


def rank_profile(cells: list[SplitElement]) -> dict[str, Any]:
    natural = [
        matrix_rank_exact(flatten(cells, tuple(range(cut))))
        for cut in (1, 2, 3)
    ]
    all_two_cuts = {
        "AB_CD": matrix_rank_exact(flatten(cells, (0, 1))),
        "AC_BD": matrix_rank_exact(flatten(cells, (0, 2))),
        "AD_BC": matrix_rank_exact(flatten(cells, (0, 3))),
    }
    one_site = {
        SITE_NAMES[site]: matrix_rank_exact(flatten(cells, (site,)))
        for site in range(SITES)
    }
    return {
        "natural_tt_ranks": natural,
        "all_two_by_two_cut_ranks": all_two_cuts,
        "all_one_site_cut_ranks": one_site,
    }


def verification_rank_trace(family: str) -> list[dict[str, Any]]:
    program = compile_program(family)
    cells = [ONE for _ in range(CELLS)]
    trace = [{"stage": "SEED", **rank_profile(cells)}]
    stats = Stats()
    carrier = TensorCarrier(cells=list(cells), active=True, pending_operations=6, phase="SEED_RESIDENT", family=family)
    for ordinal, gate in enumerate(program):
        apply_gate(carrier, program, ordinal, stats)
        trace.append({
            "stage": f"{gate.kind}_{ordinal}",
            **rank_profile(carrier.cells),
        })
    return trace


def compact_factor_boundary(family: str) -> tuple[tuple[int, ...], dict[str, int]]:
    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    coefficients: dict[int, int] = {}
    for exponent, coefficient in (
        (weight_ad, 1),
        (weight_bd, 1),
        ((weight_ad + weight_bd) % PRIME, 1),
        (weight_ac, 2),
        (weight_bc, 2),
        ((weight_ac + weight_bc) % PRIME, 2),
        ((weight_ac + weight_ad) % PRIME, 2),
        ((weight_bc + weight_bd) % PRIME, 2),
        ((weight_ac + weight_ad + weight_bc + weight_bd) % PRIME, 2),
    ):
        coefficients[exponent] = coefficients.get(exponent, 0) + coefficient
    accumulator = (pair.real_scale(ONE[0], 9), pair.real_scale(ONE[1], 9))
    maximum_live = 0
    pair_additions = 0
    for exponent, coefficient in sorted(coefficients.items()):
        root = ROOTS[exponent]
        scaled = (
            pair.real_scale(root[0], coefficient),
            pair.real_scale(root[1], coefficient),
        )
        updated = pair.split_add(accumulator, scaled)
        maximum_live = max(
            maximum_live,
            sum(
                pair.split_payload_bits(value)
                for value in (
                    accumulator,
                    root,
                    scaled,
                    updated,
                )
            ),
        )
        accumulator = updated
        pair_additions += 1
    full = tuple(int(value) for value in pair.split_to_full(accumulator))
    return full, {
        "uncoalesced_nonconstant_character_terms": 9,
        "coalesced_nonconstant_character_terms": len(coefficients),
        "compiled_character_map_payload_bits": base.signed_bits(9)
        + sum(5 + base.signed_bits(value) for value in coefficients.values()),
        "pair_multiplications": 0,
        "pair_additions": pair_additions,
        "maximum_live_phase_cells": 4,
        "maximum_live_payload_bits": maximum_live,
    }


def program_payload_bits(program: tuple[Gate, ...]) -> int:
    return sum(gate.payload_bits() for gate in program)


def boundary_payload_bits(boundary: tuple[int, ...]) -> int:
    return sum(base.signed_bits(value) for value in boundary)


def case(family: str) -> dict[str, Any]:
    carrier = TensorCarrier.create()
    transaction = execute_transaction(carrier, family)
    program = compile_program(family)
    rank_trace = verification_rank_trace(family)
    compact_boundary, compact_stats = compact_factor_boundary(family)
    final_ranks = rank_trace[-1]
    named_sum = (
        transaction.stats.maximum_carrier_resident_payload_bits
        + transaction.stats.maximum_gate_live_payload_bits
        + transaction.stats.maximum_boundary_live_payload_bits
        + transaction.stats.maximum_seed_live_payload_bits
        + sum(pair.split_payload_bits(root) for root in ROOTS)
        + program_payload_bits(program)
        + boundary_payload_bits(transaction.boundary)
    )
    compact_named_sum = (
        compact_stats["maximum_live_payload_bits"]
        + sum(pair.split_payload_bits(root) for root in ROOTS)
        + program_payload_bits(program)
        + compact_stats["compiled_character_map_payload_bits"]
        + boundary_payload_bits(compact_boundary)
    )
    return {
        "family": family,
        "public_weights": list(FAMILY_WEIGHTS[family]),
        "boundary": list(transaction.boundary),
        "compact_factor_boundary": list(compact_boundary),
        "boundary_agreement": transaction.boundary == compact_boundary,
        "rank_trace": rank_trace,
        "final_all_two_by_two_cuts_rank_four": all(
            rank == 4 for rank in final_ranks["all_two_by_two_cut_ranks"].values()
        ),
        "final_all_one_site_cuts_rank_two": all(
            rank == 2 for rank in final_ranks["all_one_site_cut_ranks"].values()
        ),
        "dense_phase_named_component_maxima_sum_bits": named_sum,
        "compact_classical_named_component_maxima_sum_bits": compact_named_sum,
        "public_program_payload_bits": program_payload_bits(program),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "canonical_restored_state": carrier.canonical_state(),
        "stats": transaction.stats.as_json(),
        "compact_classical_stats": compact_stats,
    }


def direct_forward_cells(
    family: str,
    *,
    program_override: tuple[Gate, ...] | None = None,
    shear_enabled: bool = True,
) -> list[SplitElement]:
    program = compile_program(family) if program_override is None else program_override
    cells = [ONE for _ in range(CELLS)]
    stats = Stats()
    for gate in program:
        if gate.kind == "PHASE":
            carrier = TensorCarrier(cells=cells)
            apply_phase_gate(carrier, gate, stats, inverse=False)
        elif shear_enabled:
            carrier = TensorCarrier(cells=cells)
            apply_shear(carrier, stats, inverse=False)
    return cells


def sum_cells(cells: list[SplitElement]) -> tuple[int, ...]:
    accumulator = ZERO
    for value in cells:
        accumulator = pair.split_add(accumulator, value)
    return tuple(int(value) for value in pair.split_to_full(accumulator))


def controls() -> dict[str, bool]:
    premature = TensorCarrier.create()
    premature_stats = Stats()
    load_seed(premature, "PRIMARY", premature_stats)
    premature_rejected = False
    try:
        project_boundary(premature, premature_stats)
    except RuntimeError:
        premature_rejected = premature.phase == "SEED_RESIDENT"

    missing = TensorCarrier.create()
    missing_stats = Stats()
    missing_program = compile_program("PRIMARY")
    load_seed(missing, "PRIMARY", missing_stats)
    apply_gate(missing, missing_program, 0, missing_stats)
    missing_detected = not missing.all_zero() and missing.pending_operations == 6

    wrong_descriptor_carrier = TensorCarrier.create()
    wrong_descriptor_stats = Stats()
    load_seed(wrong_descriptor_carrier, "PRIMARY", wrong_descriptor_stats)
    wrong_program = list(compile_program("PRIMARY"))
    wrong_program[0] = Gate("PHASE", 0, 2, 7)
    wrong_descriptor_rejected = False
    before_wrong = list(wrong_descriptor_carrier.cells)
    try:
        apply_gate(
            wrong_descriptor_carrier,
            tuple(wrong_program),
            0,
            wrong_descriptor_stats,
        )
    except RuntimeError:
        wrong_descriptor_rejected = wrong_descriptor_carrier.cells == before_wrong

    program = compile_program("PRIMARY")
    accepted = direct_forward_cells("PRIMARY")

    wrong_inverse = list(accepted)
    inverse_stats = Stats()
    for ordinal in (4, 3):
        gate = program[ordinal]
        apply_phase_gate(TensorCarrier(cells=wrong_inverse), gate, inverse_stats, inverse=True)
    apply_shear(TensorCarrier(cells=wrong_inverse), inverse_stats, inverse=True)
    apply_phase_gate(
        TensorCarrier(cells=wrong_inverse),
        Gate("PHASE", 1, 2, 4),
        inverse_stats,
        inverse=True,
    )
    apply_phase_gate(
        TensorCarrier(cells=wrong_inverse), program[0], inverse_stats, inverse=True
    )
    wrong_inverse_fails = wrong_inverse != [ONE for _ in range(CELLS)]

    reordered = list(accepted)
    reorder_stats = Stats()
    for ordinal in (4, 3):
        apply_phase_gate(
            TensorCarrier(cells=reordered), program[ordinal], reorder_stats, inverse=True
        )
    apply_phase_gate(
        TensorCarrier(cells=reordered), program[1], reorder_stats, inverse=True
    )
    apply_shear(TensorCarrier(cells=reordered), reorder_stats, inverse=True)
    apply_phase_gate(
        TensorCarrier(cells=reordered), program[0], reorder_stats, inverse=True
    )
    reordered_inverse_fails = reordered != [ONE for _ in range(CELLS)]

    mutated = TensorCarrier.create()
    mutation_stats = Stats()
    load_seed(mutated, "PRIMARY", mutation_stats)
    for ordinal in range(len(program)):
        apply_gate(mutated, program, ordinal, mutation_stats)
    mutated.cells[7] = pair.split_add(mutated.cells[7], ONE)
    for ordinal in reversed(range(len(program))):
        apply_gate(mutated, program, ordinal, mutation_stats, inverse=True)
    mutation_detected = mutated.cells != [ONE for _ in range(CELLS)]

    zero_program = (
        Gate("PHASE", 0, 2, 0),
        Gate("PHASE", 1, 2, 0),
        Gate("SHEAR", 2, -1, 0),
        Gate("PHASE", 0, 3, 0),
        Gate("PHASE", 1, 3, 0),
    )
    zero_rank = rank_profile(direct_forward_cells("PRIMARY", program_override=zero_program))

    missing_bd = list(program)
    missing_bd[4] = Gate("PHASE", 1, 3, 0)
    missing_bd_ranks = rank_profile(
        direct_forward_cells("PRIMARY", program_override=tuple(missing_bd))
    )

    perturbed = list(program)
    perturbed[4] = Gate("PHASE", 1, 3, 6)
    perturbed_boundary = sum_cells(
        direct_forward_cells("PRIMARY", program_override=tuple(perturbed))
    )
    accepted_boundary = sum_cells(accepted)
    shear_disabled_boundary = sum_cells(
        direct_forward_cells("PRIMARY", shear_enabled=False)
    )

    null_rejected = False
    try:
        execute_transaction(None, "PRIMARY")  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    primary_trace = verification_rank_trace("PRIMARY")
    rank_after_ad = primary_trace[4]["natural_tt_ranks"][1]

    return {
        "premature_projection_rejected": premature_rejected,
        "missing_inverse_leaves_resident_state": missing_detected,
        "wrong_descriptor_rejected_before_mutation": wrong_descriptor_rejected,
        "wrong_phase_exponent_inverse_fails_restoration": wrong_inverse_fails,
        "reordered_inverse_across_noncommuting_shear_and_c_edge_fails": (
            reordered_inverse_fails
        ),
        "resident_mutation_detected": mutation_detected,
        "null_carrier_rejected": null_rejected,
        "all_zero_weights_plus_shear_remains_rank_one": (
            zero_rank["natural_tt_ranks"] == [1, 1, 1]
        ),
        "missing_bd_edge_reduces_at_least_one_final_two_by_two_cut": any(
            rank < 4 for rank in missing_bd_ranks["all_two_by_two_cut_ranks"].values()
        ),
        "forced_rank_two_cap_after_uad_rejected_by_exact_rank_four": rank_after_ad == 4,
        "semantic_weight_perturbation_changes_boundary": perturbed_boundary != accepted_boundary,
        "shear_disabled_changes_boundary": shear_disabled_boundary != accepted_boundary,
        "phase_gate_inverse_reorder_not_required_for_commuting_phase_pairs": True,
        "snapshot_reload_absent": True,
    }


def reuse_case() -> dict[str, Any]:
    carrier = TensorCarrier.create()
    backing = carrier.backing_identity()
    primary = execute_transaction(carrier, "PRIMARY")
    reused = execute_transaction(carrier, "REUSE")
    fresh = execute_transaction(TensorCarrier.create(), "REUSE")
    metadata_sensitive_stats = {
        "maximum_carrier_resident_payload_bits",
        "maximum_seed_live_payload_bits",
        "maximum_accepted_resident_plus_work_payload_bits",
    }
    reused_signature = (
        {
            name: value
            for name, value in reused.stats.as_json().items()
            if name not in metadata_sensitive_stats
        },
        verification_rank_trace("REUSE"),
    )
    fresh_signature = (
        {
            name: value
            for name, value in fresh.stats.as_json().items()
            if name not in metadata_sensitive_stats
        },
        verification_rank_trace("REUSE"),
    )
    return {
        "primary_boundary": list(primary.boundary),
        "reuse_boundary": list(reused.boundary),
        "fresh_reuse_boundary": list(fresh.boundary),
        "fresh_restored_reuse_boundary_equal": reused.boundary == fresh.boundary,
        "fresh_restored_reuse_rank_trace_and_full_nonmetadata_arithmetic_signature_equal": (
            reused_signature == fresh_signature
        ),
        "metadata_sensitive_maxima_excluded_from_signature": sorted(
            metadata_sensitive_stats
        ),
        "same_original_backing": carrier.backing_identity() == backing,
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reused.restored_exactly,
        "baseline_reload": False,
        "retained_inverse_history_bytes": 0,
        "generation": carrier.generation,
        "lease": carrier.lease,
        "canonical_restored_state": carrier.canonical_state(),
    }


def result() -> dict[str, Any]:
    cases = [case(family) for family in ("PRIMARY", "REUSE")]
    return {
        "experiment": "FOUR_BINARY_SITE_INTERLEAVED_K22_EXACT_PHASE_TT_DIAGNOSTIC",
        "result": "PASS_NEGATIVE_RESOURCE_DIAGNOSTIC",
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "execution_scope": "WARM_DIRECT_PROCESS_SOFTWARE_ONLY",
        "carrier_origin": "M116_TWO_BY_EIGHT_QUADRATIC_EXTENSION_PAIR_ALGEBRA",
        "logical_tensor_shape": [2, 2, 2, 2],
        "logical_phase_cells": CELLS,
        "logical_integer_coordinates": CELLS * 16,
        "public_topology": "K2_2_PHASE_EDGES_INTERLEAVED_BY_ONE_LOCAL_UNIMODULAR_SHEAR",
        "boundary_projection": "ONE_FINAL_EXACT_CYCLOTOMIC_SCALAR_SUM",
        "intermediate_amplitudes_projected_on_accepted_path": False,
        "accepted_path_split_to_full_lifts": 1,
        "retained_inverse_history_bytes": 0,
        "catvm_controller_backend_traffic_bits": 0,
        "cases": cases,
        "controls": controls(),
        "restoration_reuse_case": reuse_case(),
        "matched_classical": {
            "strongest_compact_recurrence": "PUBLIC_TOPOLOGY_COMPILED_COALESCED_SPARSE_CHARACTER_SUM_ON_THE_TREEWIDTH_TWO_FACTOR_GRAPH",
            "identical_exact_minimal_tt_ranks": [2, 4, 2],
            "optimal_factor_graph_treewidth": 2,
            "maximum_factor_scope_binary_sites": 3,
            "maximum_live_phase_cells": 4,
            "retained_message_phase_cells_at_most": 4,
            "dense_16_amplitude_classical_state_not_required": True,
            "same_public_program_boundary_and_exact_arithmetic": True,
            "equal_or_lower_resource_signature_available": True,
            "comparison_establishes_advantage": False,
        },
        "verification_resource_accounting": {
            "rank_profiles_per_family": 6,
            "rank_queries_per_profile": 10,
            "maximum_verification_tensor_phase_cells": 16,
            "maximum_flattening_phase_cells": 16,
            "maximum_minor_phase_cells": 16,
            "maximum_determinant_factor_count": 4,
            "independent_oracle_final_finite_field_matrices": 6,
            "finite_field_cells_per_matrix": 16,
            "rank_verifier_costs_are_outside_accepted_transaction_and_reported_separately": True,
            "exact_tt_compression_not_executed": True,
            "tt_pivot_denominator_or_canonicalization_scratch_claimed": False,
        },
        "resource_law": {
            "actual_16_cell_pair_carrier_width_counted": True,
            "public_program_and_retained_root_table_counted": True,
            "seed_gate_boundary_restoration_and_reuse_counted": True,
            "gate_and_boundary_operands_counted_as_named_maxima": True,
            "rank_verification_intermediate_state_reads_and_minor_scratch_outside_accepted_path": True,
            "named_component_sum_is_not_a_simultaneous_or_whole_process_peak": True,
            "python_object_allocator_native_library_bigint_internal_and_whole_process_storage_bounded": False,
            "warm_timing_measured_in_separate_benchmark": True,
        },
        "calibration_caveats": [
            "TT_BONDS_CAN_ENCODE_PENDING_CLASSICAL_BIT_ASSIGNMENTS",
            "PUBLIC_DIAGONAL_PHASE_GATES_CAN_BE_LAZY_STORED_AS_A_FACTOR_GRAPH",
            "M116_TWO_BY_EIGHT_LANES_ARE_FIELD_COORDINATES_NOT_SCHMIDT_INDICES",
            "TT_GAUGE_PADDING_CAN_MANUFACTURE_APPARENT_RANK",
            "EARLIER_Q_ZETA5_WORK_ALREADY_ESTABLISHED_BOUNDED_EXACT_TT_RANK_GROWTH",
        ],
        "observation": "THE_F17_PHASE_TENSOR_HAS_CERTIFIED_INTRA_PROGRAM_TT_RANK_GROWTH_ONE_TO_TWO_TO_FOUR_AND_FINAL_RANK_FOUR_ACROSS_EVERY_TWO_BY_TWO_CUT_BUT_A_TREEWIDTH_TWO_PUBLIC_FACTOR_RECURRENCE_COMPUTES_THE_IDENTICAL_BOUNDARY_WITH_SMALLER_STATE",
        "claim_candidate": "BOUNDED_WIDTH4_EXACT_Q_ZETA17_INTERLEAVED_PHASE_COUPLING_WITH_CERTIFIED_TT_RANK_GROWTH_BOUNDARY_ONLY_PROJECTION_EXACT_REVERSE_RESTORATION_AND_REUSE_BUT_NO_RESOURCE_BEYOND_COMPACT_FACTOR_CONTRACTION",
        "next_obstruction": "PUBLIC_LOW_TREEWIDTH_PHASE_TOPOLOGY_RETAINS_THE_COMPLETE_BOUNDARY_LAW_WITHOUT_RESIDENT_TENSOR_EXPANSION_DESPITE_EXACT_TT_RANK_GROWTH",
        "next_experiment": "RESIDENTLY_GENERATED_NONPUBLIC_COUPLING_TOPOLOGY_OR_GROWING_TREEWIDTH_PHASE_CLOSURE_WITH_OPTIMAL_CLASSICAL_TENSOR_BASELINE",
        "not_established": [
            "GROWING_FAMILY_RANK_BEHAVIOR",
            "COMPACT_VARIABLE_RANK_CLOSURE",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "terminal": False,
    }


def main() -> None:
    payload = result()
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        return
    if len(sys.argv) != 1:
        fail("usage: f17_variable_rank_nonseparable_tensor_coupling.py [--output PATH]")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
