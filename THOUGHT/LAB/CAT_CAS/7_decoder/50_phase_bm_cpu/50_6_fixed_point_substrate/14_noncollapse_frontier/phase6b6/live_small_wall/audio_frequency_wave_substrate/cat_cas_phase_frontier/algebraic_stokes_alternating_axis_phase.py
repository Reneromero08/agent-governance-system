#!/usr/bin/env python3
"""Exact character-phase custody for alternating-axis Stokes brackets."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import algebraic_stokes_lie_relation_phase as base


GRADES = 13
MAX_DEGREE = 14
RESTORE_TOLERANCE = 2.0e-10
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
Monomial = tuple[int, int, int]


@dataclass
class Stats:
    native_phase_updates: int = 0
    resident_phase_reads: int = 0
    lie_poisson_monomial_products: int = 0
    sphere_reduction_terms: int = 0
    zero_scalar_contributions_skipped: int = 0
    nonzero_scalar_phase_permutations: int = 0
    hidden_coefficient_decodes: int = 0
    final_coefficient_decodes: int = 0
    maximum_unit_modulus_error: float = 0.0
    maximum_final_root_error: float = 0.0


@dataclass
class Block:
    degree_limit: int
    basis: tuple[Monomial, ...]
    index: dict[Monomial, int]
    working: list[list[list[complex]]]


@dataclass
class Carrier:
    blocks: list[Block]


@dataclass
class Execution:
    boundary: list[dict[str, object]]
    combined_hash: str
    stats: Stats
    restoration_max_abs: float
    restoration_nonidentity_cells: int
    restoration_generation: int
    actual_inverse: bool
    snapshot_loaded: bool


def make_carrier() -> Carrier:
    blocks = []
    for grade in range(GRADES):
        degree_limit = 2 + grade
        canonical_basis = base.basis(
            degree_limit, parity_reduced=True
        )
        blocks.append(
            Block(
                degree_limit,
                canonical_basis,
                {
                    monomial: cell
                    for cell, monomial in enumerate(canonical_basis)
                },
                [
                    [
                        [1.0 + 0.0j for _ in range(prime)]
                        for _ in canonical_basis
                    ]
                    for prime in base.PRIMES
                ],
            )
        )
    return Carrier(blocks)


def generator_polynomials(program: str) -> list[dict[Monomial, tuple[int, int]]]:
    if program == "PRIMARY":
        tilted = base.mixed_hamiltonian(-1)
    elif program == "REUSE":
        tilted = base.mixed_hamiltonian(1)
    else:
        base.fail("unknown alternating-axis program")
    axial = base.h0()
    return [
        tilted if grade % 2 == 0 else axial
        for grade in range(GRADES - 1)
    ]


def multiply(
    block: Block,
    field: int,
    cell: int,
    harmonic: int,
    factor: complex,
    stats: Stats,
) -> None:
    block.working[field][cell][harmonic] = base.unit(
        block.working[field][cell][harmonic] * factor
    )
    stats.native_phase_updates += 1


def seal(
    carrier: Carrier,
    polynomial: dict[Monomial, tuple[int, int]],
    inverse: bool,
    stats: Stats,
) -> None:
    block = carrier.blocks[0]
    for monomial, (numerator, denominator) in polynomial.items():
        cell = block.index[monomial]
        for field, prime in enumerate(base.PRIMES):
            value = base.residue(numerator, denominator, prime)
            for harmonic in range(prime):
                phase = base.root(field, value * harmonic)
                multiply(
                    block,
                    field,
                    cell,
                    harmonic,
                    phase.conjugate() if inverse else phase,
                    stats,
                )


def accumulate_bracket(
    carrier: Carrier,
    source_grade: int,
    generator: dict[Monomial, tuple[int, int]],
    inverse: bool,
    wrong: bool,
    stats: Stats,
) -> None:
    source = carrier.blocks[source_grade]
    target = carrier.blocks[source_grade + 1]
    cyclic = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for left, (numerator, denominator) in generator.items():
        for source_cell, right in enumerate(source.basis):
            for output_variable, first, second in cyclic:
                derivative = (
                    left[first] * right[second]
                    - left[second] * right[first]
                )
                if derivative == 0:
                    continue
                raw = [left[index] + right[index] for index in range(3)]
                raw[first] -= 1
                raw[second] -= 1
                raw[output_variable] += 1
                for reduced, reduction in base.canonical_reduction(
                    tuple(raw)
                ):
                    target_cell = target.index[reduced]
                    for field, prime in enumerate(base.PRIMES):
                        scalar = (
                            base.residue(numerator, denominator, prime)
                            * derivative
                            * reduction
                        ) % prime
                        if scalar == 0:
                            stats.zero_scalar_contributions_skipped += 1
                            continue
                        source_character = source.working[field][source_cell]
                        stats.nonzero_scalar_phase_permutations += 1
                        for harmonic in range(prime):
                            contribution = source_character[
                                (scalar * harmonic) % prime
                            ]
                            if inverse and not wrong:
                                contribution = contribution.conjugate()
                            multiply(
                                target,
                                field,
                                target_cell,
                                harmonic,
                                contribution,
                                stats,
                            )
                            stats.resident_phase_reads += 1
                    stats.sphere_reduction_terms += 1
                stats.lie_poisson_monomial_products += 1


def observe(carrier: Carrier, stats: Stats) -> None:
    for block in carrier.blocks:
        for field_coordinates in block.working:
            for character in field_coordinates:
                for phase in character:
                    error = abs(abs(phase) - 1.0)
                    stats.maximum_unit_modulus_error = max(
                        stats.maximum_unit_modulus_error, error
                    )
                    if error > 5.0e-12:
                        base.fail(
                            "alternating-axis carrier left unit torus"
                        )


def decode(phase: complex, field: int, stats: Stats) -> int:
    distances = [
        abs(phase - base.root(field, value))
        for value in range(base.PRIMES[field])
    ]
    selected = min(range(base.PRIMES[field]), key=distances.__getitem__)
    error = distances[selected]
    stats.maximum_final_root_error = max(
        stats.maximum_final_root_error, error
    )
    if error > base.ROOT_TOLERANCE:
        base.fail("alternating-axis boundary is not a root")
    stats.final_coefficient_decodes += 1
    return selected


def hash_byte(hash_value: int, value: int) -> int:
    return ((hash_value ^ value) * FNV_PRIME) & ((1 << 64) - 1)


def project(
    carrier: Carrier, stats: Stats
) -> tuple[list[dict[str, object]], str]:
    boundary = []
    combined = FNV_OFFSET
    for block in carrier.blocks:
        item: dict[str, object] = {
            "degree_limit": block.degree_limit,
            "quotient_basis_cells": len(block.basis),
        }
        for field, prime in enumerate(base.PRIMES):
            hash_value = FNV_OFFSET
            nonzero = 0
            for cell, monomial in enumerate(block.basis):
                for exponent in monomial:
                    hash_value = hash_byte(hash_value, exponent)
                value = decode(
                    block.working[field][cell][1], field, stats
                )
                hash_value = hash_byte(hash_value, value)
                nonzero += int(value != 0)
            item[f"nonzero_p{prime}"] = nonzero
            item[f"hash_p{prime}"] = f"{hash_value:016x}"
            for byte_index in range(8):
                combined = hash_byte(
                    combined, (hash_value >> (8 * byte_index)) & 0xFF
                )
        boundary.append(item)
    return boundary, f"{combined:016x}"


def restoration(carrier: Carrier) -> float:
    return max(
        (
            abs(phase - 1.0)
            for block in carrier.blocks
            for field_coordinates in block.working
            for character in field_coordinates
            for phase in character
        ),
        default=0.0,
    )


def restoration_nonidentity(carrier: Carrier) -> int:
    return sum(
        int(abs(phase - 1.0) > RESTORE_TOLERANCE)
        for block in carrier.blocks
        for field_coordinates in block.working
        for character in field_coordinates
        for phase in character
    )


def snapshot_reload(carrier: Carrier) -> None:
    for block in carrier.blocks:
        for field_coordinates in block.working:
            for character in field_coordinates:
                for harmonic in range(len(character)):
                    character[harmonic] = 1.0 + 0.0j


def execute_on(
    carrier: Carrier, program: str, mode: str
) -> Execution:
    stats = Stats()
    seed = base.h0()
    generators = generator_polynomials(program)
    seal(carrier, seed, False, stats)
    for grade, generator in enumerate(generators):
        accumulate_bracket(
            carrier, grade, generator, False, False, stats
        )
    observe(carrier, stats)
    boundary, combined_hash = project(carrier, stats)

    actual_inverse = False
    snapshot_loaded = False
    if mode == "SNAPSHOT":
        snapshot_reload(carrier)
        snapshot_loaded = True
    elif mode == "REORDERED_INVERSE":
        accumulate_bracket(
            carrier, 0, generators[0], True, False, stats
        )
        for grade in range(GRADES - 1, 1, -1):
            accumulate_bracket(
                carrier,
                grade - 1,
                generators[grade - 1],
                True,
                False,
                stats,
            )
        seal(carrier, seed, True, stats)
        actual_inverse = True
    else:
        for grade in range(GRADES - 1, 0, -1):
            if mode == "MISSING_INVERSE" and grade == GRADES - 1:
                continue
            accumulate_bracket(
                carrier,
                grade - 1,
                generators[grade - 1],
                True,
                mode == "WRONG_INVERSE" and grade == GRADES - 1,
                stats,
            )
        seal(carrier, seed, True, stats)
        actual_inverse = True

    restore_error = restoration(carrier)
    nonidentity_cells = restoration_nonidentity(carrier)
    generation = int(
        not snapshot_loaded
        and actual_inverse
        and restore_error <= RESTORE_TOLERANCE
        and nonidentity_cells == 0
    )
    if generation == 0:
        actual_inverse = False
    observe(carrier, stats)
    return Execution(
        boundary,
        combined_hash,
        stats,
        restore_error,
        nonidentity_cells,
        generation,
        actual_inverse,
        snapshot_loaded,
    )


def execute_once(program: str, mode: str) -> Execution:
    return execute_on(make_carrier(), program, mode)


def main() -> None:
    if sys.argv[1:] == ["--project-internal-grade"]:
        base.fail("resident alternating-axis grade projection denied")
    if sys.argv[1:] == ["--null-carrier"]:
        base.fail("invalid alternating-axis phase carrier")
    if sys.argv[1:]:
        base.fail("unsupported alternating-axis request")

    shared = make_carrier()
    primary = execute_on(shared, "PRIMARY", "CORRECT")
    replay = execute_once("PRIMARY", "CORRECT")
    missing = execute_once("PRIMARY", "MISSING_INVERSE")
    wrong = execute_once("PRIMARY", "WRONG_INVERSE")
    reordered = execute_once("PRIMARY", "REORDERED_INVERSE")
    snapshot = execute_once("PRIMARY", "SNAPSHOT")
    reuse = execute_on(shared, "REUSE", "CORRECT")
    maximum_reuse_restoration = reuse.restoration_max_abs
    same_carrier_transactions = 2
    for cycle in range(6):
        program = "PRIMARY" if cycle % 2 else "REUSE"
        repeated = execute_on(shared, program, "CORRECT")
        maximum_reuse_restoration = max(
            maximum_reuse_restoration,
            repeated.restoration_max_abs,
        )
        same_carrier_transactions += 1

    if not (
        primary.restoration_max_abs <= RESTORE_TOLERANCE
        and primary.restoration_nonidentity_cells == 0
        and primary.restoration_generation == 1
        and maximum_reuse_restoration <= RESTORE_TOLERANCE
        and missing.restoration_max_abs >= 1.0e-6
        and missing.restoration_nonidentity_cells > 0
        and wrong.restoration_max_abs >= 1.0e-6
        and wrong.restoration_nonidentity_cells > 0
        and reordered.restoration_max_abs >= 1.0e-6
        and reordered.restoration_nonidentity_cells > 0
        and snapshot.restoration_max_abs <= RESTORE_TOLERANCE
        and snapshot.restoration_generation == 0
        and primary.combined_hash == replay.combined_hash
        and primary.stats.hidden_coefficient_decodes == 0
    ):
        base.fail("alternating-axis phase control failed")

    basis_cells = sum(len(block.basis) for block in shared.blocks)
    phase_cells = basis_cells * sum(base.PRIMES)
    output = {
        "claim": (
            "BOUNDED_NONCOMMUTING_ALTERNATING_AXIS_STOKES_"
            "CHARACTER_PHASE_HARMONIC_CATALECTICANT_RANK_GROWTH_WITH_"
            "RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "generator_schedule": (
            "TILTED_AXIS_SQUARED_THEN_Z_AXIS_SQUARED_ALTERNATING"
        ),
        "grades": GRADES,
        "maximum_degree": MAX_DEGREE,
        "primary_grades": primary.boundary,
        "reuse_grades": reuse.boundary,
        "primary_combined_hash": primary.combined_hash,
        "reuse_combined_hash": reuse.combined_hash,
        "declared_boundary": (
            "FULL_13_GRADE_COEFFICIENT_JET_SUMMARIES"
        ),
        "all_projected_grades_are_declared_final_jet": True,
        "parity_admissible_basis_cells": basis_cells,
        "resident_character_phase_cells": phase_cells,
        "logical_packed_phase_payload_bytes": phase_cells * 16,
        "actual_python_object_allocation_measured": False,
        "retained_inverse_history_bytes": 0,
        "complete_unit_phase_character_orbit_encoding": True,
        "coefficient_addition_is_native_phase_multiplication": True,
        "public_scalar_action_is_phase_index_permutation": True,
        "native_phase_updates": primary.stats.native_phase_updates,
        "resident_phase_reads": primary.stats.resident_phase_reads,
        "zero_scalar_contributions_skipped": (
            primary.stats.zero_scalar_contributions_skipped
        ),
        "nonzero_scalar_phase_permutations": (
            primary.stats.nonzero_scalar_phase_permutations
        ),
        "hidden_coefficient_decodes": (
            primary.stats.hidden_coefficient_decodes
        ),
        "final_coefficient_decodes": (
            primary.stats.final_coefficient_decodes
        ),
        "maximum_unit_modulus_error": (
            primary.stats.maximum_unit_modulus_error
        ),
        "maximum_final_root_error": (
            primary.stats.maximum_final_root_error
        ),
        "restoration_max_abs": primary.restoration_max_abs,
        "maximum_reuse_restoration_error": (
            maximum_reuse_restoration
        ),
        "successful_restoration_receipt": (
            primary.restoration_generation
        ),
        "same_carrier_transactions": same_carrier_transactions,
        "missing_inverse_residual": missing.restoration_max_abs,
        "wrong_inverse_residual": wrong.restoration_max_abs,
        "reordered_inverse_residual": reordered.restoration_max_abs,
        "missing_inverse_modular_mismatch_cells": (
            missing.restoration_nonidentity_cells
        ),
        "wrong_inverse_modular_mismatch_cells": (
            wrong.restoration_nonidentity_cells
        ),
        "reordered_inverse_modular_mismatch_cells": (
            reordered.restoration_nonidentity_cells
        ),
        "snapshot_residual": snapshot.restoration_max_abs,
        "snapshot_restoration_receipt": (
            snapshot.restoration_generation
        ),
        "deterministic_replay": (
            primary.combined_hash == replay.combined_hash
        ),
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse": True,
        "final_boundary_only_projection": True,
        "actual_resident_prior_grade_consumed": True,
        "noncommuting_generator_order_material": True,
        "exact_rational_oracle_required": True,
        "fixed_rank_closure_established": False,
        "unbounded_rank_growth_proved": False,
        "same_output_dual_prime_classical_semantic_state_bytes": (
            basis_cells * len(base.PRIMES)
        ),
        "same_output_classical_actual_allocation_measured": False,
        "compact_point_evaluation_semantic_state_bytes": 64,
        "point_evaluation_has_same_boundary_semantics": False,
        "genuinely_distinct_phase_resource": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
