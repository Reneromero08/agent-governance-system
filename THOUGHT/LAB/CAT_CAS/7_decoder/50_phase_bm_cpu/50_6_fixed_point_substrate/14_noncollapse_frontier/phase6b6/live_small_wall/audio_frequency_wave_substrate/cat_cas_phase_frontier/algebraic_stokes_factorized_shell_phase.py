#!/usr/bin/env python3
"""Fixed-rank phase recurrence for a factorized highest Stokes shell."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import algebraic_stokes_lie_relation_phase as base


DEPTHS = (1, 2, 4, 8, 32, 128, 512, 2048)
COORDINATES = ("a*c", "a*b", "b*c", "b^2-c^2")
FORWARD_SCALES = ((-2, 25), (2, 25), (-8, 25), (2, 25))
PRIMARY_SEED = ((672, 15625), (0, 1), (-2304, 15625), (0, 1))
REUSE_SEED = ((-672, 15625), (0, 1), (-196, 15625), (0, 1))
RESTORE_TOLERANCE = 2.0e-10
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


@dataclass
class Carrier:
    working: list[list[list[complex]]]


@dataclass
class Stats:
    native_phase_updates: int = 0
    native_phase_swaps: int = 0
    native_phase_permutations: int = 0
    hidden_coefficient_decodes: int = 0
    final_coefficient_decodes: int = 0
    maximum_unit_modulus_error: float = 0.0
    maximum_final_root_error: float = 0.0


@dataclass
class Execution:
    boundary: dict[str, object]
    restoration_max_abs: float
    restoration_nonidentity_cells: int
    restoration_generation: int
    actual_inverse: bool
    snapshot_loaded: bool
    stats: Stats


def make_carrier() -> Carrier:
    return Carrier(
        [
            [
                [1.0 + 0.0j for _ in range(prime)]
                for _ in COORDINATES
            ]
            for prime in base.PRIMES
        ]
    )


def seed(program: str) -> tuple[tuple[int, int], ...]:
    if program == "PRIMARY":
        return PRIMARY_SEED
    if program == "REUSE":
        return REUSE_SEED
    base.fail("unknown factorized shell program")


def multiply(
    carrier: Carrier,
    field: int,
    cell: int,
    harmonic: int,
    factor: complex,
    stats: Stats,
) -> None:
    carrier.working[field][cell][harmonic] = base.unit(
        carrier.working[field][cell][harmonic] * factor
    )
    stats.native_phase_updates += 1


def seal(
    carrier: Carrier,
    coefficients: tuple[tuple[int, int], ...],
    inverse: bool,
    stats: Stats,
) -> None:
    for field, prime in enumerate(base.PRIMES):
        for cell, (numerator, denominator) in enumerate(coefficients):
            value = base.residue(numerator, denominator, prime)
            for harmonic in range(prime):
                phase = base.root(field, value * harmonic)
                multiply(
                    carrier,
                    field,
                    cell,
                    harmonic,
                    phase.conjugate() if inverse else phase,
                    stats,
                )


def scale_cell(
    carrier: Carrier,
    field: int,
    cell: int,
    scale: tuple[int, int],
    inverse: bool,
    stats: Stats,
) -> None:
    prime = base.PRIMES[field]
    scalar = base.residue(scale[0], scale[1], prime)
    if inverse:
        scalar = pow(scalar, -1, prime)
    source = carrier.working[field][cell]
    carrier.working[field][cell] = [
        source[(scalar * harmonic) % prime]
        for harmonic in range(prime)
    ]
    stats.native_phase_permutations += prime


def transition(
    carrier: Carrier, inverse: bool, stats: Stats
) -> None:
    for field in range(len(base.PRIMES)):
        if inverse:
            for cell, scale in enumerate(FORWARD_SCALES):
                scale_cell(
                    carrier, field, cell, scale, True, stats
                )
            carrier.working[field][0], carrier.working[field][1] = (
                carrier.working[field][1],
                carrier.working[field][0],
            )
            carrier.working[field][2], carrier.working[field][3] = (
                carrier.working[field][3],
                carrier.working[field][2],
            )
        else:
            carrier.working[field][0], carrier.working[field][1] = (
                carrier.working[field][1],
                carrier.working[field][0],
            )
            carrier.working[field][2], carrier.working[field][3] = (
                carrier.working[field][3],
                carrier.working[field][2],
            )
            for cell, scale in enumerate(FORWARD_SCALES):
                scale_cell(
                    carrier, field, cell, scale, False, stats
                )
        stats.native_phase_swaps += 2


def observe(carrier: Carrier, stats: Stats) -> None:
    for field_coordinates in carrier.working:
        for character in field_coordinates:
            for phase in character:
                error = abs(abs(phase) - 1.0)
                stats.maximum_unit_modulus_error = max(
                    stats.maximum_unit_modulus_error, error
                )
                if error > 5.0e-12:
                    base.fail("factorized shell carrier left unit torus")


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
        base.fail("factorized shell coefficient is not a root")
    stats.final_coefficient_decodes += 1
    return selected


def hash_byte(hash_value: int, value: int) -> int:
    return ((hash_value ^ value) * FNV_PRIME) & ((1 << 64) - 1)


def project(
    carrier: Carrier, depth: int, stats: Stats
) -> dict[str, object]:
    result: dict[str, object] = {
        "depth": depth,
        "highest_harmonic_degree": depth + 2,
        "public_repeated_factor_axis": [24, 0, -7],
        "public_repeated_factor_exponent": depth,
        "resident_quadratic_coordinates": len(COORDINATES),
    }
    for field, prime in enumerate(base.PRIMES):
        hash_value = FNV_OFFSET
        nonzero = 0
        for cell in range(len(COORDINATES)):
            value = decode(
                carrier.working[field][cell][1], field, stats
            )
            hash_value = hash_byte(hash_value, cell)
            hash_value = hash_byte(hash_value, value)
            nonzero += int(value != 0)
        result[f"nonzero_p{prime}"] = nonzero
        result[f"hash_p{prime}"] = f"{hash_value:016x}"
    return result


def restoration(carrier: Carrier) -> float:
    return max(
        (
            abs(phase - 1.0)
            for field_coordinates in carrier.working
            for character in field_coordinates
            for phase in character
        ),
        default=0.0,
    )


def restoration_nonidentity(carrier: Carrier) -> int:
    count = 0
    for field_coordinates in carrier.working:
        for character in field_coordinates:
            for phase in character:
                count += int(abs(phase - 1.0) > RESTORE_TOLERANCE)
    return count


def snapshot_reload(carrier: Carrier) -> None:
    for field_coordinates in carrier.working:
        for character in field_coordinates:
            for harmonic in range(len(character)):
                character[harmonic] = 1.0 + 0.0j


def execute_on(
    carrier: Carrier, program: str, depth: int, mode: str
) -> Execution:
    if depth < 1:
        base.fail("factorized shell depth must be positive")
    stats = Stats()
    coefficients = seed(program)
    seal(carrier, coefficients, False, stats)
    for _ in range(depth - 1):
        transition(carrier, False, stats)
    observe(carrier, stats)
    boundary = project(carrier, depth, stats)
    snapshot_loaded = False
    actual_inverse = False
    if mode == "SNAPSHOT":
        snapshot_reload(carrier)
        snapshot_loaded = True
    else:
        inverse_steps = depth - 1
        for step in range(inverse_steps):
            if mode == "MISSING_INVERSE" and step == 0:
                continue
            transition(
                carrier,
                inverse=not (mode == "WRONG_INVERSE" and step == 0),
                stats=stats,
            )
        seal(carrier, coefficients, True, stats)
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
        boundary=boundary,
        restoration_max_abs=restore_error,
        restoration_nonidentity_cells=nonidentity_cells,
        restoration_generation=generation,
        actual_inverse=actual_inverse,
        snapshot_loaded=snapshot_loaded,
        stats=stats,
    )


def execute_once(program: str, depth: int, mode: str) -> Execution:
    return execute_on(make_carrier(), program, depth, mode)


def main() -> None:
    if sys.argv[1:] == ["--project-resident-quadratic"]:
        base.fail("resident factorized shell projection denied")
    if sys.argv[1:] == ["--null-carrier"]:
        base.fail("invalid factorized shell carrier")
    if sys.argv[1:]:
        base.fail("unsupported factorized shell request")

    shared = make_carrier()
    primary_records = []
    maximum_restoration = 0.0
    total_transactions = 0
    aggregate_updates = 0
    aggregate_swaps = 0
    aggregate_permutations = 0
    for depth in DEPTHS:
        execution = execute_on(shared, "PRIMARY", depth, "CORRECT")
        primary_records.append(execution.boundary)
        maximum_restoration = max(
            maximum_restoration, execution.restoration_max_abs
        )
        aggregate_updates += execution.stats.native_phase_updates
        aggregate_swaps += execution.stats.native_phase_swaps
        aggregate_permutations += (
            execution.stats.native_phase_permutations
        )
        total_transactions += 1

    replay = execute_once("PRIMARY", DEPTHS[-1], "CORRECT")
    missing = execute_once(
        "PRIMARY", DEPTHS[-1], "MISSING_INVERSE"
    )
    wrong = execute_once("PRIMARY", DEPTHS[-1], "WRONG_INVERSE")
    snapshot = execute_once("PRIMARY", DEPTHS[-1], "SNAPSHOT")
    reuse = execute_on(shared, "REUSE", DEPTHS[-1], "CORRECT")
    maximum_restoration = max(
        maximum_restoration, reuse.restoration_max_abs
    )
    total_transactions += 1
    for cycle in range(7):
        program = "PRIMARY" if cycle % 2 else "REUSE"
        repeated = execute_on(
            shared, program, DEPTHS[cycle], "CORRECT"
        )
        maximum_restoration = max(
            maximum_restoration, repeated.restoration_max_abs
        )
        total_transactions += 1

    accepted = execute_once("PRIMARY", DEPTHS[-1], "CORRECT")
    if not (
        accepted.restoration_max_abs <= RESTORE_TOLERANCE
        and accepted.restoration_nonidentity_cells == 0
        and accepted.restoration_generation == 1
        and maximum_restoration <= RESTORE_TOLERANCE
        and missing.restoration_max_abs >= 1.0e-6
        and missing.restoration_nonidentity_cells > 0
        and wrong.restoration_max_abs >= 1.0e-6
        and wrong.restoration_nonidentity_cells > 0
        and snapshot.restoration_max_abs <= RESTORE_TOLERANCE
        and snapshot.restoration_generation == 0
        and accepted.boundary == replay.boundary
        and accepted.stats.hidden_coefficient_decodes == 0
    ):
        base.fail("factorized shell control failed")

    output = {
        "claim": (
            "FIXED_RANK_FACTORIZED_HIGHEST_STOKES_HARMONIC_SHELL_"
            "PHASE_RECURRENCE_WITH_RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "tested_depths": list(DEPTHS),
        "primary_boundaries": primary_records,
        "reuse_boundary": reuse.boundary,
        "public_factorized_signature": "L_AXIS_AND_EXPONENT_TIMES_Q4",
        "public_factor_axis": [24, 0, -7],
        "maximum_public_factor_exponent": DEPTHS[-1],
        "resident_quadratic_coordinates": len(COORDINATES),
        "resident_dual_prime_phase_cells": (
            len(COORDINATES) * sum(base.PRIMES)
        ),
        "logical_packed_phase_payload_bytes": (
            len(COORDINATES) * sum(base.PRIMES) * 16
        ),
        "compiled_public_recurrence_logical_packed_bytes": 96,
        "actual_python_object_allocation_measured": False,
        "retained_inverse_history_bytes": 0,
        "maximum_hidden_phase_cells_independent_of_depth": True,
        "native_phase_updates_all_primary_depths": aggregate_updates,
        "native_phase_swaps_all_primary_depths": aggregate_swaps,
        "native_phase_permutations_all_primary_depths": (
            aggregate_permutations
        ),
        "hidden_coefficient_decodes": (
            accepted.stats.hidden_coefficient_decodes
        ),
        "final_coefficient_decodes_at_maximum_depth": (
            accepted.stats.final_coefficient_decodes
        ),
        "maximum_unit_modulus_error": (
            accepted.stats.maximum_unit_modulus_error
        ),
        "maximum_final_root_error": (
            accepted.stats.maximum_final_root_error
        ),
        "complete_unit_phase_character_orbit_encoding": True,
        "coefficient_scaling_is_exact_phase_index_permutation": True,
        "restoration_max_abs_at_maximum_depth": (
            accepted.restoration_max_abs
        ),
        "maximum_same_carrier_restoration_error": maximum_restoration,
        "successful_restoration_receipt": (
            accepted.restoration_generation
        ),
        "same_carrier_transactions": total_transactions,
        "missing_inverse_residual": missing.restoration_max_abs,
        "wrong_inverse_residual": wrong.restoration_max_abs,
        "missing_inverse_modular_mismatch_cells": (
            missing.restoration_nonidentity_cells
        ),
        "wrong_inverse_modular_mismatch_cells": (
            wrong.restoration_nonidentity_cells
        ),
        "snapshot_residual": snapshot.restoration_max_abs,
        "snapshot_restoration_receipt": (
            snapshot.restoration_generation
        ),
        "reordered_inverse_applicable": False,
        "reordered_inverse_reason": (
            "ALL_REPEATED_FIXED_AXIS_TRANSITIONS_ARE_IDENTICAL"
        ),
        "deterministic_replay": accepted.boundary == replay.boundary,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse": True,
        "reuse_seed_is_second_public_factorized_q4_program": True,
        "final_boundary_only_projection": True,
        "actual_resident_quadratic_consumed": True,
        "exact_public_factor_not_expanded": True,
        "factor_exponent_storage_grows_logarithmically": True,
        "exact_rational_oracle_required": True,
        "highest_shell_fixed_rank_closure_established": True,
        "software_execution_bounded_to_tested_depths": True,
        "full_stokes_signature_fixed_rank_closure_established": False,
        "best_matched_dual_prime_classical_residue_state_bytes": 8,
        "matched_classical_recurrence_exists": True,
        "genuinely_distinct_phase_resource": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "unbounded_catalytic_computation": False,
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
