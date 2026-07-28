#!/usr/bin/env python3
"""Dual-prime phase carrier for Stokes-sphere reduced Kerr Lie closure."""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass, field
from enum import IntEnum


PRIMES = (17, 19)
GRADES = 5
MAX_DEGREE = 6
ROOT_TOLERANCE = 1.0e-3
RESTORE_TOLERANCE = 2.0e-10
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
REUSE_CYCLES = 8
Monomial = tuple[int, int, int]


class Program(IntEnum):
    PRIMARY = 0
    REUSE = 1
    IDENTITY = 2
    ZERO = 3
    SWAPPED = 4


class Mode(IntEnum):
    CORRECT = 0
    MISSING_INVERSE = 1
    WRONG_INVERSE = 2
    REORDERED_INVERSE = 3
    SNAPSHOT = 4


@dataclass
class Stats:
    native_phase_updates: int = 0
    resident_phase_reads: int = 0
    field_product_interpolations: int = 0
    lie_poisson_monomial_products: int = 0
    sphere_reduction_terms: int = 0
    hidden_coefficient_decodes: int = 0
    final_coefficient_decodes: int = 0
    maximum_unit_modulus_error: float = 0.0
    maximum_final_root_error: float = 0.0


@dataclass
class Block:
    degree_limit: int
    basis: tuple[Monomial, ...]
    index: dict[Monomial, int]
    working: list[list[complex]]


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


def fail(message: str) -> None:
    raise RuntimeError(message)


def unit(value: complex) -> complex:
    magnitude = abs(value)
    if not magnitude > 0.0 or not math.isfinite(magnitude):
        fail("nonfinite Stokes relation phase")
    return value / magnitude


def root(field: int, value: int) -> complex:
    prime = PRIMES[field]
    return cmath.exp(2j * math.pi * (value % prime) / prime)


def phase_power(value: complex, exponent: int, prime: int) -> complex:
    power = exponent % prime
    result = 1.0 + 0.0j
    factor = value
    while power:
        if power & 1:
            result *= factor
        factor *= factor
        power >>= 1
    return unit(result)


def field_product(left: complex, right: complex, field: int, stats: Stats) -> complex:
    prime = PRIMES[field]
    left_power = [1.0 + 0.0j]
    right_power = [1.0 + 0.0j]
    for _ in range(1, prime):
        left_power.append(left_power[-1] * left)
        right_power.append(right_power[-1] * right)
    total = 0.0 + 0.0j
    for left_index in range(prime):
        for right_index in range(prime):
            total += (
                root(field, -left_index * right_index)
                * left_power[left_index]
                * right_power[right_index]
            )
    stats.field_product_interpolations += 1
    return unit(total / prime)


def basis(degree_limit: int) -> tuple[Monomial, ...]:
    result: list[Monomial] = []
    for x_exponent in range(degree_limit + 1):
        for y_exponent in range(degree_limit - x_exponent + 1):
            for z_exponent in range(2):
                if x_exponent + y_exponent + z_exponent <= degree_limit:
                    result.append((x_exponent, y_exponent, z_exponent))
    expected = (degree_limit + 1) ** 2
    if len(result) != expected:
        fail("Stokes quotient basis cardinality mismatch")
    return tuple(result)


def make_carrier() -> Carrier:
    blocks: list[Block] = []
    for grade in range(GRADES):
        degree_limit = 2 + grade
        canonical_basis = basis(degree_limit)
        blocks.append(
            Block(
                degree_limit=degree_limit,
                basis=canonical_basis,
                index={
                    monomial: index
                    for index, monomial in enumerate(canonical_basis)
                },
                working=[
                    [1.0 + 0.0j for _ in canonical_basis]
                    for _ in PRIMES
                ],
            )
        )
    return Carrier(blocks)


def observe(carrier: Carrier, stats: Stats) -> None:
    for block in carrier.blocks:
        for phase_cells in block.working:
            for phase in phase_cells:
                error = abs(abs(phase) - 1.0)
                stats.maximum_unit_modulus_error = max(
                    stats.maximum_unit_modulus_error, error
                )
                if error > 5.0e-12:
                    fail("Stokes relation carrier left unit torus")


def multiply(
    block: Block, field: int, cell: int, factor: complex, stats: Stats
) -> None:
    block.working[field][cell] = unit(
        block.working[field][cell] * factor
    )
    stats.native_phase_updates += 1


def residue(numerator: int, denominator: int, prime: int) -> int:
    return numerator * pow(denominator, -1, prime) % prime


def h0() -> dict[Monomial, tuple[int, int]]:
    # z^2 == 1 - x^2 - y^2 on the unit Stokes sphere.
    return {(0, 0, 0): (1, 1), (2, 0, 0): (-1, 1), (0, 2, 0): (-1, 1)}


def mixed_hamiltonian(cross_sign: int) -> dict[Monomial, tuple[int, int]]:
    # (24*x +/- 7*z)^2 / 625, reduced by z^2=1-x^2-y^2.
    return {
        (0, 0, 0): (49, 625),
        (2, 0, 0): (527, 625),
        (0, 2, 0): (-49, 625),
        (1, 0, 1): (cross_sign * 336, 625),
    }


def compile_program(
    program: Program,
) -> tuple[dict[Monomial, tuple[int, int]], dict[Monomial, tuple[int, int]]]:
    if program == Program.PRIMARY:
        return h0(), mixed_hamiltonian(-1)
    if program == Program.REUSE:
        return h0(), mixed_hamiltonian(1)
    if program == Program.IDENTITY:
        return h0(), h0()
    if program == Program.ZERO:
        return {}, {}
    if program == Program.SWAPPED:
        return mixed_hamiltonian(-1), h0()
    fail("unknown Stokes relation program")


def encode_polynomial(
    polynomial: dict[Monomial, tuple[int, int]]
) -> list[tuple[Monomial, tuple[complex, complex]]]:
    result = []
    for monomial in sorted(polynomial):
        numerator, denominator = polynomial[monomial]
        result.append(
            (
                monomial,
                tuple(
                    root(
                        field,
                        residue(
                            numerator, denominator, PRIMES[field]
                        ),
                    )
                    for field in range(len(PRIMES))
                ),
            )
        )
    return result


def seal(
    carrier: Carrier,
    seed: list[tuple[Monomial, tuple[complex, complex]]],
    inverse: bool,
    stats: Stats,
) -> None:
    block = carrier.blocks[0]
    for monomial, phases in seed:
        cell = block.index[monomial]
        for field, phase in enumerate(phases):
            multiply(
                block,
                field,
                cell,
                phase.conjugate() if inverse else phase,
                stats,
            )


def canonical_reduction(monomial: Monomial) -> tuple[tuple[Monomial, int], ...]:
    pending = {monomial: 1}
    canonical: dict[Monomial, int] = {}
    while pending:
        current, coefficient = pending.popitem()
        x_exponent, y_exponent, z_exponent = current
        if z_exponent < 2:
            canonical[current] = canonical.get(current, 0) + coefficient
            continue
        branches = (
            ((x_exponent, y_exponent, z_exponent - 2), coefficient),
            ((x_exponent + 2, y_exponent, z_exponent - 2), -coefficient),
            ((x_exponent, y_exponent + 2, z_exponent - 2), -coefficient),
        )
        for branch, branch_coefficient in branches:
            pending[branch] = (
                pending.get(branch, 0) + branch_coefficient
            )
    return tuple(
        (reduced, coefficient)
        for reduced, coefficient in sorted(canonical.items())
        if coefficient != 0
    )


def accumulate_bracket(
    carrier: Carrier,
    source_grade: int,
    generator: list[tuple[Monomial, tuple[complex, complex]]],
    inverse: bool,
    wrong: bool,
    stats: Stats,
) -> None:
    source = carrier.blocks[source_grade]
    target = carrier.blocks[source_grade + 1]
    # (output variable, first derivative, second derivative, sign order)
    cyclic = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for left, generator_phases in generator:
        for source_cell, right in enumerate(source.basis):
            for output_variable, first, second in cyclic:
                factor = (
                    left[first] * right[second]
                    - left[second] * right[first]
                )
                if factor == 0:
                    continue
                raw = [left[index] + right[index] for index in range(3)]
                raw[first] -= 1
                raw[second] -= 1
                raw[output_variable] += 1
                reductions = canonical_reduction(tuple(raw))
                for reduced, reduction_factor in reductions:
                    target_cell = target.index[reduced]
                    for field, generator_phase in enumerate(generator_phases):
                        source_phase = source.working[field][source_cell]
                        stats.resident_phase_reads += 1
                        contribution = field_product(
                            generator_phase, source_phase, field, stats
                        )
                        contribution = phase_power(
                            contribution,
                            factor * reduction_factor,
                            PRIMES[field],
                        )
                        if inverse:
                            contribution = contribution.conjugate()
                        if wrong:
                            contribution = contribution.conjugate()
                        multiply(
                            target,
                            field,
                            target_cell,
                            contribution,
                            stats,
                        )
                    stats.sphere_reduction_terms += 1
                stats.lie_poisson_monomial_products += 1


def decode(phase: complex, field: int, stats: Stats) -> int:
    distances = [abs(phase - root(field, value)) for value in range(PRIMES[field])]
    selected = min(range(PRIMES[field]), key=distances.__getitem__)
    error = distances[selected]
    stats.maximum_final_root_error = max(
        stats.maximum_final_root_error, error
    )
    if error > ROOT_TOLERANCE:
        fail("final Stokes signature coefficient is not a root")
    stats.final_coefficient_decodes += 1
    return selected


def hash_byte(hash_value: int, value: int) -> int:
    return ((hash_value ^ value) * FNV_PRIME) & ((1 << 64) - 1)


def project(carrier: Carrier, stats: Stats) -> tuple[list[dict[str, object]], str]:
    boundary: list[dict[str, object]] = []
    combined = FNV_OFFSET
    for block in carrier.blocks:
        item: dict[str, object] = {
            "degree_limit": block.degree_limit,
            "quotient_basis_cells": len(block.basis),
        }
        for field, prime in enumerate(PRIMES):
            hash_value = FNV_OFFSET
            nonzero = 0
            for cell, monomial in enumerate(block.basis):
                for exponent in monomial:
                    hash_value = hash_byte(hash_value, exponent)
                value = decode(block.working[field][cell], field, stats)
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
        (abs(phase - 1.0) for block in carrier.blocks
         for field_cells in block.working for phase in field_cells),
        default=0.0,
    )

def restoration_nonidentity(carrier: Carrier) -> int:
    count = 0
    for block in carrier.blocks:
        for field, field_cells in enumerate(block.working):
            for phase in field_cells:
                selected = min(
                    range(PRIMES[field]),
                    key=lambda value: abs(phase - root(field, value)),
                )
                count += int(selected != 0)
    return count


def snapshot_reload(carrier: Carrier) -> None:
    for block in carrier.blocks:
        for field_cells in block.working:
            for cell in range(len(field_cells)):
                field_cells[cell] = 1.0 + 0.0j


def execute_on(carrier: Carrier, program: Program, mode: Mode) -> Execution:
    stats = Stats()
    seed_polynomial, generator_polynomial = compile_program(program)
    seed = encode_polynomial(seed_polynomial)
    generator = encode_polynomial(generator_polynomial)
    seal(carrier, seed, False, stats)
    for grade in range(GRADES - 1):
        accumulate_bracket(
            carrier, grade, generator, False, False, stats
        )
    observe(carrier, stats)
    boundary, combined_hash = project(carrier, stats)
    actual_inverse = False
    snapshot_loaded = False
    generation = 0
    if mode == Mode.SNAPSHOT:
        snapshot_reload(carrier)
        snapshot_loaded = True
    elif mode == Mode.REORDERED_INVERSE:
        accumulate_bracket(carrier, 0, generator, True, False, stats)
        for grade in range(GRADES - 1, 1, -1):
            accumulate_bracket(
                carrier, grade - 1, generator, True, False, stats
            )
        seal(carrier, seed, True, stats)
        actual_inverse = True
    else:
        for grade in range(GRADES - 1, 0, -1):
            if mode == Mode.MISSING_INVERSE and grade == GRADES - 1:
                continue
            accumulate_bracket(
                carrier,
                grade - 1,
                generator,
                True,
                mode == Mode.WRONG_INVERSE and grade == GRADES - 1,
                stats,
            )
        seal(carrier, seed, True, stats)
        actual_inverse = True
    restore_error = restoration(carrier)
    nonidentity_cells = restoration_nonidentity(carrier)
    if (
        not snapshot_loaded
        and actual_inverse
        and restore_error <= RESTORE_TOLERANCE
        and nonidentity_cells == 0
    ):
        generation = 1
    else:
        actual_inverse = False
        generation = 0
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


def execute_once(program: Program, mode: Mode) -> Execution:
    return execute_on(make_carrier(), program, mode)


def main() -> None:
    import sys

    if sys.argv[1:] == ["--project-shared-port"]:
        fail("resident Stokes shared wave port projection denied")
    if sys.argv[1:] == ["--null-carrier"]:
        fail("invalid Stokes relation carrier")
    if sys.argv[1:]:
        fail("unsupported Stokes relation request")

    shared_carrier = make_carrier()
    primary = execute_on(shared_carrier, Program.PRIMARY, Mode.CORRECT)
    replay = execute_once(Program.PRIMARY, Mode.CORRECT)
    missing = execute_once(Program.PRIMARY, Mode.MISSING_INVERSE)
    wrong = execute_once(Program.PRIMARY, Mode.WRONG_INVERSE)
    reordered = execute_once(Program.PRIMARY, Mode.REORDERED_INVERSE)
    snapshot = execute_once(Program.PRIMARY, Mode.SNAPSHOT)
    identity = execute_once(Program.IDENTITY, Mode.CORRECT)
    zero = execute_once(Program.ZERO, Mode.CORRECT)
    swapped = execute_once(Program.SWAPPED, Mode.CORRECT)
    reuse = execute_on(shared_carrier, Program.REUSE, Mode.CORRECT)

    maximum_reuse_restoration = reuse.restoration_max_abs
    reuse_generation = 1
    for cycle in range(1, REUSE_CYCLES):
        program = Program.REUSE if cycle % 2 == 0 else Program.PRIMARY
        repeated = execute_on(shared_carrier, program, Mode.CORRECT)
        maximum_reuse_restoration = max(
            maximum_reuse_restoration, repeated.restoration_max_abs
        )
        reuse_generation += 1

    identity_higher_zero = all(
        grade["nonzero_p17"] == 0 and grade["nonzero_p19"] == 0
        for grade in identity.boundary[1:]
    )
    zero_all_zero = all(
        grade["nonzero_p17"] == 0 and grade["nonzero_p19"] == 0
        for grade in zero.boundary
    )
    swap_discriminated = (
        primary.boundary[1]["hash_p17"] != swapped.boundary[1]["hash_p17"]
        and primary.boundary[1]["hash_p19"] != swapped.boundary[1]["hash_p19"]
    )
    if not (
        primary.restoration_max_abs <= RESTORE_TOLERANCE
        and missing.restoration_max_abs >= 1.0e-6
        and wrong.restoration_max_abs > RESTORE_TOLERANCE
        and reordered.restoration_max_abs >= 1.0e-6
        and primary.restoration_nonidentity_cells == 0
        and missing.restoration_nonidentity_cells > 0
        and wrong.restoration_nonidentity_cells > 0
        and reordered.restoration_nonidentity_cells > 0
        and snapshot.restoration_max_abs <= RESTORE_TOLERANCE
        and identity_higher_zero
        and zero_all_zero
        and swap_discriminated
        and primary.combined_hash == replay.combined_hash
        and primary.stats.hidden_coefficient_decodes == 0
        and maximum_reuse_restoration <= RESTORE_TOLERANCE
    ):
        fail("Stokes relation control failed")

    quotient_cells = sum(
        len(block.basis) for block in shared_carrier.blocks
    ) * len(PRIMES)
    output = {
        "claim": (
            "BOUNDED_STOKES_SPHERE_REDUCED_DUAL_PRIME_PHASE_RESIDENT_"
            "NONLINEAR_SYMPLECTIC_LIE_SIGNATURE_WITH_RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "port_type": "NORMALIZED_TWO_MODE_STOKES_SPHERE_WAVE_PORT",
        "poisson_canonical_index_contraction": True,
        "shared_wave_port_elimination_established": False,
        "grades": GRADES,
        "maximum_degree": MAX_DEGREE,
        "primary_grades": primary.boundary,
        "reuse_grades": reuse.boundary,
        "primary_combined_hash": primary.combined_hash,
        "reuse_combined_hash": reuse.combined_hash,
        "resident_dual_prime_coefficient_cells": quotient_cells,
        "logical_packed_phase_payload_bytes": quotient_cells * 16,
        "implicit_identity_baseline_bytes": 0,
        "actual_python_object_allocation_measured": False,
        "retain_all_grade_blocks": GRADES,
        "retained_inverse_kernels": 0,
        "native_phase_updates": primary.stats.native_phase_updates,
        "resident_phase_reads": primary.stats.resident_phase_reads,
        "field_product_interpolations": (
            primary.stats.field_product_interpolations
        ),
        "lie_poisson_monomial_products": (
            primary.stats.lie_poisson_monomial_products
        ),
        "sphere_reduction_terms": primary.stats.sphere_reduction_terms,
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
        "maximum_reuse_restoration_error": maximum_reuse_restoration,
        "successful_restoration_receipt": primary.restoration_generation,
        "same_carrier_reuse_transactions": reuse_generation,
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
        "snapshot_restoration_receipt": snapshot.restoration_generation,
        "identity_mixer_higher_grades_zero": identity_higher_zero,
        "zero_kerr_all_grades_zero": zero_all_zero,
        "swapped_order_discriminated": swap_discriminated,
        "deterministic_replay": primary.combined_hash == replay.combined_hash,
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse": True,
        "final_boundary_only_projection": True,
        "actual_resident_grade_consumed": True,
        "public_topology_inverse_rematerialization": True,
        "exact_norm_casimir_quotient_reduction": True,
        "global_phase_removed_by_stokes_map": True,
        "mathematically_exact_dual_prime_phase_algebra": True,
        "rational_oracle_required": True,
        "exact_rank_reduction_established": True,
        "fixed_rank_closure_established": False,
        "remaining_harmonic_rank_growth": True,
        "unbounded_growth_proved": False,
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
        print(str(error), file=__import__("sys").stderr)
        raise SystemExit(2) from error
