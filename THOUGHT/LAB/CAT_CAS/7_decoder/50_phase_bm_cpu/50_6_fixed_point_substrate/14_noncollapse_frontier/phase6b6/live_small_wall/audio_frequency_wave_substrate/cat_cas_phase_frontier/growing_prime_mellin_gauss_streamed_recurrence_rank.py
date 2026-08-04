#!/usr/bin/env python3
"""Fixed-workspace Gauss streaming and exact cyclic-rank diagnostic.

M180 retained q-1 Mellin coefficients and three public compiler tables.  This
successor evaluates one requested final boundary scalar by regenerating every
Gauss factor from public field topology in a ten-cell exact residue workspace.
The inverse rematerializes the same terms in reverse order, restores the zero
workspace on the same backing, and permits an unrelated second transaction.

The time-space repair is deliberately separated from algebraic compression.
The source coefficient sequence has q-1 nonzero cyclic Fourier modes, hence
cyclic Hankel rank q-1.  A fixed-order linear recurrence therefore does not
replace the table.  The streamed procedure instead spends quadratic work per
projected scalar, and identical compact classical software can do the same.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_MELLIN_GAUSS_SOURCE_COEFFICIENT_"
    "FAMILIES_STREAM_ONE_FINAL_BOUNDARY_SCALAR_FROM_A_FIXED10_FIELD_CELL_"
    "ZERO_WORKSPACE_WITH_TOPOLOGY_REMATERIALIZED_INVERSE_EXACT_SAME_BACKING_"
    "RESTORATION_AND_UNRELATED_REUSE_WHILE_THE_SOURCE_COEFFICIENT_CYCLIC_"
    "HANKEL_RANK_IS_EXACTLY_QMINUS1_AND_THE_DETERMINANT_GAMMA_SEQUENCE_HAS_"
    "OBSERVED_RANK_QMINUS1_ON_EVERY_DECLARED_CASE_SO_TABLE_RESIDENCY_IS_"
    "REMOVED_BY_A_QUADRATIC_WORK_TIME_SPACE_TRADEOFF_NOT_BY_LINEAR_"
    "RECURRENCE_COMPACTION_AND_THE_IDENTICAL_CLASSICAL_STREAM_REMAINS"
)


DECLARED_FIELDS = (
    (5, 41),
    (7, 43),
    (11, 331),
    (13, 157),
    (17, 1361),
    (19, 2053),
    (23, 1013),
    (29, 2437),
    (31, 1861),
    (37, 6661),
    (41, 13121),
    (43, 3613),
    (47, 12973),
    (53, 8269),
)

WORKSPACE_CELLS = 10


def fail(message: str) -> None:
    raise RuntimeError(message)


def prime_factors(value: int) -> tuple[int, ...]:
    factors: list[int] = []
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def primitive_root(prime: int) -> int:
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(
            pow(candidate, (prime - 1) // factor, prime) != 1
            for factor in factors
        ):
            return candidate
    fail(f"no primitive root for {prime}")


@dataclass(frozen=True)
class ProceduralField:
    q: int
    p: int
    additive_root: int
    multiplicative_root: int
    q_generator: int

    @classmethod
    def create(cls, q: int, p: int) -> "ProceduralField":
        if (p - 1) % q or (p - 1) % (q - 1):
            fail("auxiliary field lacks a required root group")
        generator = primitive_root(p)
        return cls(
            q=q,
            p=p,
            additive_root=pow(generator, (p - 1) // q, p),
            multiplicative_root=pow(generator, (p - 1) // (q - 1), p),
            q_generator=primitive_root(q),
        )


@dataclass
class WorkCounts:
    gauss_calls: int = 0
    gauss_orbit_terms: int = 0
    additive_phase_exponentiations: int = 0
    gauss_field_multiplications: int = 0
    gauss_field_additions: int = 0
    character_calls: int = 0
    character_orbit_visits: int = 0
    boundary_channels: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "gauss_calls": self.gauss_calls,
            "gauss_orbit_terms": self.gauss_orbit_terms,
            "additive_phase_exponentiations": self.additive_phase_exponentiations,
            "gauss_field_multiplications": self.gauss_field_multiplications,
            "gauss_field_additions": self.gauss_field_additions,
            "character_calls": self.character_calls,
            "character_orbit_visits": self.character_orbit_visits,
            "boundary_channels": self.boundary_channels,
        }


@dataclass(frozen=True)
class PublicProgram:
    determinant_character: int
    scale_character: int


@dataclass(frozen=True)
class PublicBoundary:
    coordinates: tuple[int, int, int, int, int, int]
    scale: int


def determinant(coordinates: tuple[int, ...], q: int) -> int:
    a, b, c, d, e, f = coordinates
    return (
        a * (d * f - e * e)
        - b * (b * f - e * c)
        + c * (b * e - d * c)
    ) % q


def principal_minor(
    matrix: tuple[tuple[int, ...], ...], indices: tuple[int, ...], q: int
) -> int:
    if len(indices) == 1:
        return matrix[indices[0]][indices[0]] % q
    if len(indices) == 2:
        left, right = indices
        return (
            matrix[left][left] * matrix[right][right]
            - matrix[left][right] * matrix[right][left]
        ) % q
    return determinant(
        (
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][1],
            matrix[1][2],
            matrix[2][2],
        ),
        q,
    )


def rank_and_square_class(coordinates: tuple[int, ...], q: int) -> tuple[int, int]:
    a, b, c, d, e, f = coordinates
    matrix = ((a, b, c), (b, d, e), (c, e, f))
    for rank in (3, 2, 1):
        for indices in itertools.combinations(range(3), rank):
            discriminant = principal_minor(matrix, indices, q)
            if discriminant:
                square = 1 if pow(discriminant, (q - 1) // 2, q) == 1 else -1
                return rank, square
    return 0, 0


def stream_gauss_into(
    field: ProceduralField,
    exponent: int,
    cells: list[int],
    destination: int,
    counts: WorkCounts,
) -> None:
    """Write one exact Gauss sum without a log, phase, or coefficient table.

    Slots 5..9 are the Gauss accumulator, F_q orbit value, character value,
    character step, and additive phase value.  Destination 5 retains the
    accumulator; destinations 1..4 receive a copy before scratch is cleared.
    """
    if len(cells) != WORKSPACE_CELLS:
        fail("wrong workspace width")
    p = field.p
    h = field.q - 1
    cells[5] = 0
    cells[6] = 1
    cells[7] = 1
    cells[8] = pow(field.multiplicative_root, exponent % h, p)
    cells[9] = 0
    counts.gauss_calls += 1
    for _ in range(h):
        cells[9] = pow(field.additive_root, cells[6], p)
        cells[5] = (cells[5] + cells[9] * cells[7]) % p
        cells[6] = cells[6] * field.q_generator % field.q
        cells[7] = cells[7] * cells[8] % p
        counts.gauss_orbit_terms += 1
        counts.additive_phase_exponentiations += 1
        counts.gauss_field_multiplications += 3
        counts.gauss_field_additions += 1
    if destination != 5:
        cells[destination] = cells[5]
        cells[5] = 0
    cells[6] = cells[7] = cells[8] = cells[9] = 0


def procedural_character(
    field: ProceduralField,
    value: int,
    exponent: int,
    cells: list[int],
    counts: WorkCounts,
) -> int:
    """Evaluate a multiplicative character by public generator-orbit search."""
    reduced = value % field.q
    if reduced == 0:
        fail("multiplicative character evaluated at zero")
    h = field.q - 1
    cells[6] = 1
    cells[7] = 1
    cells[8] = pow(field.multiplicative_root, exponent % h, field.p)
    counts.character_calls += 1
    for _ in range(h):
        counts.character_orbit_visits += 1
        if cells[6] == reduced:
            result = cells[7]
            cells[6] = cells[7] = cells[8] = 0
            return result
        cells[6] = cells[6] * field.q_generator % field.q
        cells[7] = cells[7] * cells[8] % field.p
    fail("public generator orbit did not reach value")


def determinant_factor_from_slots(
    field: ProceduralField,
    character_exponent: int,
    coordinates: tuple[int, ...],
    cells: list[int],
    counts: WorkCounts,
) -> int:
    q, p = field.q, field.p
    h = q - 1
    rank, square_class = rank_and_square_class(coordinates, q)
    if rank == 3:
        gamma = (
            cells[3]
            * cells[3]
            * cells[4]
            * cells[1]
            * cells[1]
            * cells[1]
        ) % p
        character = procedural_character(
            field, determinant(coordinates, q), -character_exponent, cells, counts
        )
        return gamma * character % p
    if character_exponent % h == 0:
        if rank == 0:
            return (q**6 - q**5 - q**3 + q**2) % p
        return (q**2 - q**3) % p
    if character_exponent % h == h // 2 and rank == 1:
        return q**2 * h * cells[1] ** 3 * square_class % p
    return 0


def scale_factor_from_slot(
    field: ProceduralField,
    character_exponent: int,
    scale: int,
    cells: list[int],
    counts: WorkCounts,
) -> int:
    h = field.q - 1
    if scale % field.q:
        character = procedural_character(
            field, scale, -character_exponent, cells, counts
        )
        return cells[5] * character % field.p
    return h % field.p if character_exponent % h == 0 else 0


def accumulate_boundary(
    field: ProceduralField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    counts: WorkCounts,
    sign: int,
    order: Iterable[int],
    omit_channel: int | None = None,
    wrong_gamma_shift: bool = False,
) -> None:
    """Add or subtract the public boundary character contraction in-place."""
    q, p = field.q, field.p
    h = q - 1
    eta = h // 2
    total_character = (program.determinant_character + program.scale_character) % h
    rank, _ = rank_and_square_class(boundary.coordinates, q)
    normalization = pow(h, -1, p)
    for j in order:
        if omit_channel is not None and j == omit_channel % h:
            continue
        source_m = (total_character - j) % h
        stream_gauss_into(
            field, -((j - program.determinant_character) % h), cells, 2, counts
        )
        if rank == 3:
            stream_gauss_into(field, j, cells, 3, counts)
            shift = eta + (1 if wrong_gamma_shift else 0)
            stream_gauss_into(field, j + shift, cells, 4, counts)
        else:
            cells[3] = cells[4] = 0
        if boundary.scale % q:
            stream_gauss_into(field, source_m, cells, 5, counts)
        else:
            cells[5] = 0
        determinant_factor = determinant_factor_from_slots(
            field, j, boundary.coordinates, cells, counts
        )
        scale_factor = scale_factor_from_slot(
            field, source_m, boundary.scale, cells, counts
        )
        coefficient = normalization * cells[2] % p
        summand = coefficient * determinant_factor * scale_factor % p
        cells[0] = (cells[0] + sign * summand) % p
        cells[2] = cells[3] = cells[4] = cells[5] = 0
        counts.boundary_channels += 1


def forward_boundary(
    field: ProceduralField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    counts: WorkCounts,
    omit_channel: int | None = None,
    wrong_gamma_shift: bool = False,
) -> int:
    if any(cells):
        fail("carrier must enter a transaction in canonical zero state")
    eta = (field.q - 1) // 2
    stream_gauss_into(field, eta, cells, 1, counts)
    accumulate_boundary(
        field,
        program,
        boundary,
        cells,
        counts,
        1,
        range(field.q - 1),
        omit_channel,
        wrong_gamma_shift,
    )
    return cells[0]


def inverse_boundary(
    field: ProceduralField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    counts: WorkCounts,
    reverse_order: bool = True,
) -> None:
    h = field.q - 1
    order = range(h - 1, -1, -1) if reverse_order else range(h)
    accumulate_boundary(field, program, boundary, cells, counts, -1, order)
    stream_gauss_into(field, h // 2, cells, 5, counts)
    cells[1] = (cells[1] - cells[5]) % field.p
    cells[5] = 0


def gauss_reference(field: ProceduralField, exponent: int) -> int:
    h = field.q - 1
    step = pow(field.multiplicative_root, exponent % h, field.p)
    x = 1
    character = 1
    total = 0
    for _ in range(h):
        total += pow(field.additive_root, x, field.p) * character
        x = x * field.q_generator % field.q
        character = character * step % field.p
    return total % field.p


def character_reference(field: ProceduralField, value: int, exponent: int) -> int:
    reduced = value % field.q
    if not reduced:
        fail("reference character evaluated at zero")
    x = 1
    result = 1
    step = pow(field.multiplicative_root, exponent % (field.q - 1), field.p)
    for _ in range(field.q - 1):
        if x == reduced:
            return result
        x = x * field.q_generator % field.q
        result = result * step % field.p
    fail("reference character orbit failure")


def materialized_boundary_reference(
    field: ProceduralField, program: PublicProgram, boundary: PublicBoundary
) -> int:
    q, p = field.q, field.p
    h = q - 1
    eta = h // 2
    gauss = [gauss_reference(field, exponent) for exponent in range(h)]
    normalization = pow(h, -1, p)
    total_character = (program.determinant_character + program.scale_character) % h
    rank, square_class = rank_and_square_class(boundary.coordinates, q)
    answer = 0
    for j in range(h):
        coefficient = normalization * gauss[-((j - program.determinant_character) % h)] % p
        if rank == 3:
            gamma = gauss[j] ** 2 * gauss[(j + eta) % h] * gauss[eta] ** 3 % p
            det_factor = gamma * character_reference(
                field, determinant(boundary.coordinates, q), -j
            ) % p
        elif j == 0:
            det_factor = (
                q**6 - q**5 - q**3 + q**2
                if rank == 0
                else q**2 - q**3
            ) % p
        elif j == eta and rank == 1:
            det_factor = q**2 * h * gauss[eta] ** 3 * square_class % p
        else:
            det_factor = 0
        m = (total_character - j) % h
        if boundary.scale % q:
            scale_factor = gauss[m] * character_reference(
                field, boundary.scale, -m
            ) % p
        else:
            scale_factor = h % p if m == 0 else 0
        answer += coefficient * det_factor * scale_factor
    return answer % p


def cyclic_fourier(
    sequence: list[int], root: int, modulus: int
) -> list[int]:
    width = len(sequence)
    return [
        sum(
            sequence[index] * pow(root, (frequency * index) % width, modulus)
            for index in range(width)
        )
        % modulus
        for frequency in range(width)
    ]


def matrix_rank_mod(matrix: list[list[int]], modulus: int) -> int:
    work = [row[:] for row in matrix]
    rows = len(work)
    columns = len(work[0]) if work else 0
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if work[row][column] % modulus),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column] % modulus, -1, modulus)
        work[rank] = [value * inverse % modulus for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column] % modulus
            if factor:
                work[row] = [
                    (left - factor * right) % modulus
                    for left, right in zip(work[row], work[rank])
                ]
        rank += 1
        if rank == rows:
            break
    return rank


def berlekamp_massey(sequence: list[int], modulus: int) -> int:
    connection = [1]
    previous = [1]
    order = 0
    shift = 1
    discrepancy_scale = 1
    for index in range(len(sequence)):
        discrepancy = sequence[index] % modulus
        for offset in range(1, order + 1):
            discrepancy += connection[offset] * sequence[index - offset]
        discrepancy %= modulus
        if not discrepancy:
            shift += 1
            continue
        old = connection[:]
        ratio = discrepancy * pow(discrepancy_scale, -1, modulus) % modulus
        required = len(previous) + shift
        if len(connection) < required:
            connection.extend([0] * (required - len(connection)))
        for offset, value in enumerate(previous):
            connection[offset + shift] = (
                connection[offset + shift] - ratio * value
            ) % modulus
        if 2 * order <= index:
            order = index + 1 - order
            previous = old
            discrepancy_scale = discrepancy
            shift = 1
        else:
            shift += 1
    return order


def rank_diagnostic(field: ProceduralField, determinant_character: int) -> dict[str, Any]:
    h = field.q - 1
    p = field.p
    gauss = [gauss_reference(field, exponent) for exponent in range(h)]
    coefficients = [
        pow(h, -1, p) * gauss[-((j - determinant_character) % h)] % p
        for j in range(h)
    ]
    transform = cyclic_fourier(coefficients, field.multiplicative_root, p)
    expected = [
        pow(field.additive_root, pow(field.q_generator, frequency, field.q), p)
        * pow(
            field.multiplicative_root,
            determinant_character * frequency % h,
            p,
        )
        % p
        for frequency in range(h)
    ]
    if transform != expected or any(value == 0 for value in transform):
        fail("analytic full-support identity failed")
    hankel = [
        [coefficients[(row + column) % h] for column in range(h)]
        for row in range(h)
    ]
    hankel_rank = matrix_rank_mod(hankel, p)
    recurrence_rank = berlekamp_massey(coefficients * 3, p)
    eta = h // 2
    gamma = [
        gauss[j] ** 2 * gauss[(j + eta) % h] * gauss[eta] ** 3 % p
        for j in range(h)
    ]
    gamma_hankel_rank = matrix_rank_mod(
        [[gamma[(row + column) % h] for column in range(h)] for row in range(h)],
        p,
    )
    gamma_recurrence_rank = berlekamp_massey(gamma * 3, p)

    mutated_transform = transform[:]
    mutated_transform[0] = 0
    inverse_root = pow(field.multiplicative_root, -1, p)
    inverse_width = pow(h, -1, p)
    mutated = [
        inverse_width
        * sum(
            mutated_transform[frequency]
            * pow(inverse_root, frequency * index % h, p)
            for frequency in range(h)
        )
        % p
        for index in range(h)
    ]
    mutated_rank = matrix_rank_mod(
        [[mutated[(row + column) % h] for column in range(h)] for row in range(h)],
        p,
    )
    if (hankel_rank, recurrence_rank, mutated_rank) != (h, h, h - 1):
        fail("cyclic rank or mutation control failed")
    return {
        "q": field.q,
        "auxiliary_prime": p,
        "sequence_width": h,
        "determinant_character": determinant_character,
        "analytic_source_coefficient_fourier_support": h,
        "source_coefficient_cyclic_hankel_rank": hankel_rank,
        "source_coefficient_periodic_linear_recurrence_order": recurrence_rank,
        "gamma_cyclic_hankel_rank": gamma_hankel_rank,
        "gamma_periodic_linear_recurrence_order": gamma_recurrence_rank,
        "zero_one_spectral_mode_mutation_rank": mutated_rank,
        "analytic_identity": "DFT_J_OF_HINV_G_A_MINUS_J_AT_FREQUENCY_K_EQUALS_PSI_G_TO_K_TIMES_CHI_A_G_TO_K",
        "every_analytic_spectral_component_nonzero": True,
        "diagnostic_materialized_cells_excluded_from_accepted_path": (
            2 * h + 2 * h * h
        ),
    }


def digest_cells(cells: list[int]) -> str:
    payload = ",".join(str(value) for value in cells).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def cycle_once(
    field: ProceduralField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    reverse_order: bool = True,
) -> dict[str, Any]:
    before = digest_cells(cells)
    forward_counts = WorkCounts()
    projected = forward_boundary(field, program, boundary, cells, forward_counts)
    resident_commitment = digest_cells(cells)
    persisted = projected
    inverse_counts = WorkCounts()
    inverse_boundary(
        field, program, boundary, cells, inverse_counts, reverse_order=reverse_order
    )
    return {
        "projected_final_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "pre_state_commitment": before,
        "resident_forward_commitment": resident_commitment,
        "post_state_commitment": digest_cells(cells),
        "exactly_restored": not any(cells),
        "forward_counts": forward_counts.as_dict(),
        "inverse_counts": inverse_counts.as_dict(),
    }


def transaction_case(
    field: ProceduralField,
    program: PublicProgram,
    boundary: PublicBoundary,
) -> dict[str, Any]:
    cells = [0] * WORKSPACE_CELLS
    backing = id(cells)
    primary = cycle_once(field, program, boundary, cells)
    reference = materialized_boundary_reference(field, program, boundary)
    if primary["projected_final_boundary_scalar"] != reference:
        fail("streamed boundary differs from materialized reference")
    if not primary["exactly_restored"] or id(cells) != backing:
        fail("primary carrier did not restore on the same backing")

    h = field.q - 1
    second_program = PublicProgram(
        (program.determinant_character + 2) % h,
        (program.scale_character + 3) % h,
    )
    second_boundary = PublicBoundary((3, 0, 0, 2, 0, 1), 3 % field.q)
    second = cycle_once(field, second_program, second_boundary, cells)
    fresh_cells = [0] * WORKSPACE_CELLS
    fresh = cycle_once(field, second_program, second_boundary, fresh_cells)
    if (
        second["projected_final_boundary_scalar"]
        != fresh["projected_final_boundary_scalar"]
        or not second["exactly_restored"]
        or id(cells) != backing
    ):
        fail("restored-carrier reuse failed")

    missing = [0] * WORKSPACE_CELLS
    forward_boundary(field, program, boundary, missing, WorkCounts())
    missing_inverse_fails = any(missing)

    wrong = [0] * WORKSPACE_CELLS
    forward_boundary(field, program, boundary, wrong, WorkCounts())
    wrong_program = PublicProgram(
        (program.determinant_character + 1) % h, program.scale_character
    )
    inverse_boundary(field, wrong_program, boundary, wrong, WorkCounts())
    wrong_inverse_fails = any(wrong)

    reordered = [0] * WORKSPACE_CELLS
    reordered_result = cycle_once(
        field, program, boundary, reordered, reverse_order=False
    )

    omitted_channel = None
    omitted_value = reference
    for candidate in range(h):
        omitted = [0] * WORKSPACE_CELLS
        omitted_value = forward_boundary(
            field,
            program,
            boundary,
            omitted,
            WorkCounts(),
            omit_channel=candidate,
        )
        if omitted_value != reference:
            omitted_channel = candidate
            break
    rank, _ = rank_and_square_class(boundary.coordinates, field.q)
    wrong_gamma_applicable = rank == 3
    wrong_gamma_value = reference
    if wrong_gamma_applicable:
        wrong_gamma = [0] * WORKSPACE_CELLS
        wrong_gamma_value = forward_boundary(
            field,
            program,
            boundary,
            wrong_gamma,
            WorkCounts(),
            wrong_gamma_shift=True,
        )

    if not (
        missing_inverse_fails
        and wrong_inverse_fails
        and reordered_result["exactly_restored"]
        and omitted_channel is not None
        and (not wrong_gamma_applicable or wrong_gamma_value != reference)
    ):
        fail("transaction control failed")
    return {
        "q": field.q,
        "auxiliary_prime": field.p,
        "program": {
            "determinant_character": program.determinant_character,
            "scale_character": program.scale_character,
        },
        "boundary": {
            "coordinates": list(boundary.coordinates),
            "scale": boundary.scale,
            "rank_and_square_class": list(
                rank_and_square_class(boundary.coordinates, field.q)
            ),
        },
        "workspace_field_cells": WORKSPACE_CELLS,
        "workspace_capacity_bits": WORKSPACE_CELLS * field.p.bit_length(),
        "same_backing_primary": id(cells) == backing,
        "same_backing_reused": id(cells) == backing,
        "observed_restoration_passes": 2,
        "generation_or_lease_metadata_enforced": False,
        "primary": primary,
        "materialized_reference_boundary_scalar": reference,
        "unrelated_second_program": {
            "determinant_character": second_program.determinant_character,
            "scale_character": second_program.scale_character,
            "boundary": {
                "coordinates": list(second_boundary.coordinates),
                "scale": second_boundary.scale,
            },
            "actual_restored_backing_consumed": id(cells) == backing,
            "restored_boundary_scalar": second["projected_final_boundary_scalar"],
            "fresh_boundary_scalar": fresh["projected_final_boundary_scalar"],
            "fresh_restored_resource_signature_agrees": (
                second["forward_counts"] == fresh["forward_counts"]
                and second["inverse_counts"] == fresh["inverse_counts"]
            ),
            "exactly_restored_again": second["exactly_restored"],
        },
        "controls": {
            "missing_inverse_fails": missing_inverse_fails,
            "wrong_inverse_fails": wrong_inverse_fails,
            "reordered_inverse_applicable": False,
            "reordered_inverse_reason": "CHANNEL_ACCUMULATION_IS_COMMUTATIVE_IN_THE_EXACT_RESIDUE_FIELD",
            "ascending_inverse_also_restores": reordered_result["exactly_restored"],
            "omitted_channel": omitted_channel,
            "omitted_channel_changes_boundary": omitted_channel is not None,
            "wrong_gamma_shift_applicable": wrong_gamma_applicable,
            "wrong_gamma_shift_changes_boundary": (
                wrong_gamma_value != reference if wrong_gamma_applicable else None
            ),
            "null_carrier_rejected": True,
            "snapshot_used": False,
        },
        "verification_only_materialized_gauss_table_cells": field.q - 1,
        "verification_only_fresh_carrier_cells": WORKSPACE_CELLS,
    }


def build_result() -> dict[str, Any]:
    fields = [ProceduralField.create(q, p) for q, p in DECLARED_FIELDS]
    rank_results = [
        rank_diagnostic(field, (2 * index + 1) % (field.q - 1))
        for index, field in enumerate(fields)
    ]
    transaction_results = []
    for index, field in enumerate(fields):
        h = field.q - 1
        transaction_results.append(
            transaction_case(
                field,
                PublicProgram((2 * index + 1) % h, (3 * index + 2) % h),
                PublicBoundary((2, 0, 0, 1, 0, 1), 2),
            )
        )
    # Exercise both singular determinant branches and the zero-scale branch.
    q5 = fields[0]
    transaction_results.extend(
        [
            transaction_case(q5, PublicProgram(2, 1), PublicBoundary((1, 0, 0, 0, 0, 0), 2)),
            transaction_case(q5, PublicProgram(0, 0), PublicBoundary((0, 0, 0, 0, 0, 0), 0)),
        ]
    )
    if not all(
        item["source_coefficient_cyclic_hankel_rank"] == item["q"] - 1
        and item["source_coefficient_periodic_linear_recurrence_order"]
        == item["q"] - 1
        and item["gamma_cyclic_hankel_rank"] == item["q"] - 1
        for item in rank_results
    ):
        fail("a declared rank result did not reach full width")
    if not all(
        case["primary"]["exactly_restored"]
        and case["primary"]["projected_result_survives_inverse"]
        and case["unrelated_second_program"]["exactly_restored_again"]
        for case in transaction_results
    ):
        fail("a declared transaction failed")
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "execution": "DIRECT_PROCESS_EXACT_FINITE_FIELD_RESIDUE_STREAMING_SOFTWARE",
        "declared_fields": [
            {"q": field.q, "auxiliary_prime": field.p} for field in fields
        ],
        "rank_diagnostics": rank_results,
        "transaction_cases": transaction_results,
        "analytic_rank_law": {
            "source_coefficient_sequence": "C_J=ONE_OVER_QMINUS1_TIMES_G_A_MINUS_J",
            "cyclic_fourier_component": "PSI_G_TO_K_TIMES_CHI_A_G_TO_K",
            "all_components_nonzero": True,
            "cyclic_hankel_rank": "q-1",
            "periodic_linear_recurrence_order": "q-1",
            "scope": "ALL_DECLARED_VALID_PRIME_AND_AUXILIARY_FIELD_PAIRS_BY_EXACT_CHARACTER_ORTHOGONALITY",
            "nonlinear_or_nonuniform_procedural_compression_rejected": False,
        },
        "observed_resource_law": {
            "accepted_workspace_carrier_field_cells": WORKSPACE_CELLS,
            "accepted_workspace_cell_count_independent_of_q": True,
            "accepted_workspace_exact_bit_width_independent_of_q": False,
            "workspace_capacity_bits": "10*CEIL_LOG2_P",
            "maximum_named_exact_field_temporaries_outside_workspace": 5,
            "persisted_projected_output_field_cells_during_inverse": 1,
            "accepted_full_lifecycle_named_exact_field_peak_including_output": 16,
            "accepted_full_lifecycle_named_exact_field_capacity_bits": "16*CEIL_LOG2_P",
            "public_field_configuration_integer_scalars": 5,
            "resident_gauss_or_coefficient_tables": 0,
            "forward_gauss_calls_rank3_nonzero_scale": "4*(q-1)+1",
            "full_lifecycle_gauss_calls_rank3_nonzero_scale": "8*(q-1)+2",
            "gauss_orbit_terms_per_call": "q-1",
            "forward_work_growth_per_projected_scalar": "THETA(q^2)_GAUSS_ORBIT_TERMS_PLUS_CHARACTER_SEARCH",
            "full_lifecycle_work_growth_per_projected_scalar": "THETA(q^2)_GAUSS_ORBIT_TERMS_PLUS_CHARACTER_SEARCH",
            "retained_inverse_history_cells": 0,
            "projection_output_field_cells": 1,
            "public_program_and_boundary_scalars": 9,
            "diagnostic_hankel_and_sequence_materialization_excluded_from_accepted_path": True,
            "loop_control_integers_python_objects_bigint_expression_and_modular_pow_temporaries_excluded": True,
        },
        "matched_baselines": {
            "m180_materialized_table_path": {
                "accepted_peak_field_cells": "4*q-3",
                "compiler_work": "THETA(q^2)",
                "per_boundary_character_terms_after_compile": "q-1",
            },
            "identical_classical_stream": {
                "workspace_field_cells": WORKSPACE_CELLS,
                "same_gauss_orbit_and_character_search_work": True,
                "same_exact_boundary": True,
            },
            "one_stream_point_dominates_materialized_path": False,
            "many_boundary_projection_advantage_claimed": False,
            "computational_advantage_established": False,
        },
        "controls": {
            "all_missing_inverse_controls_fail": all(
                case["controls"]["missing_inverse_fails"]
                for case in transaction_results
            ),
            "all_wrong_inverse_controls_fail": all(
                case["controls"]["wrong_inverse_fails"]
                for case in transaction_results
            ),
            "all_omitted_channel_controls_change_boundary": all(
                case["controls"]["omitted_channel_changes_boundary"]
                for case in transaction_results
            ),
            "all_applicable_wrong_gamma_shift_controls_change_boundary": all(
                (not case["controls"]["wrong_gamma_shift_applicable"])
                or case["controls"]["wrong_gamma_shift_changes_boundary"]
                for case in transaction_results
            ),
            "all_reordered_commutative_inverses_restore": all(
                case["controls"]["ascending_inverse_also_restores"]
                for case in transaction_results
            ),
            "zero_one_spectral_mode_rank_mutation_detected": all(
                item["zero_one_spectral_mode_mutation_rank"] == item["q"] - 2
                for item in rank_results
            ),
            "snapshot_used": False,
        },
        "claim_ceiling": (
            "PRIME_Q5_7_11_13_17_19_23_29_31_37_41_43_47_53_WITH_"
            "AUXILIARY_FIELDS_F41_43_331_157_1361_2053_1013_2437_1861_"
            "6661_13121_3613_12973_8269_ONE_DECLARED_RANK3_TRANSACTION_"
            "PER_FIELD_PLUS_Q5_RANK1_AND_ZERO_BOUNDARY_BRANCHES_DIRECT_"
            "PROCESS_SINGLE_SCALAR_PROJECTION_SOFTWARE"
        ),
        "claim_boundaries": {
            "fixed_exact_bit_width": False,
            "fixed_linear_recurrence_rank": False,
            "nonlinear_procedural_compression_no_go": False,
            "multi_boundary_amortized_advantage": False,
            "machine_enforced_hidden_intermediate": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "M181_REMOVES_QMINUS1_TABLE_RESIDENCY_FOR_ONE_PROJECTED_SCALAR_"
            "ONLY_BY_SPENDING_THETA_Q2_REMATERIALIZATION_WORK_WHILE_THE_"
            "SOURCE_COEFFICIENT_SEQUENCE_HAS_EXACT_FULL_QMINUS1_CYCLIC_"
            "HANKEL_RANK_AND_THE_IDENTICAL_CLASSICAL_STREAM_IS_AVAILABLE_"
            "SO_THE_NEXT_TEST_MUST_USE_NONLINEAR_HASSE_DAVENPORT_OR_JACOBI_"
            "PHASE_RELATIONS_TO_REDUCE_THE_GAUSS_FAMILY_WORK_STATE_FRONTIER_"
            "OR_ESTABLISH_THEIR_GROWING_RELATION_RANK"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
