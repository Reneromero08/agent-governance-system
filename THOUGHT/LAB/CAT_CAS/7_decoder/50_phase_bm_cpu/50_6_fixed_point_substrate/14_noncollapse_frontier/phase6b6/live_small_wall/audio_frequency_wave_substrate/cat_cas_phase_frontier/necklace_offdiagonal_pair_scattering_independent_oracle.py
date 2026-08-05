#!/usr/bin/env python3
"""Independent occupation-space oracle for necklace pair scattering.

This verifier imports no production module.  It reconstructs all 4,845
four-boson occupation histograms, the public diagonal pair phase, and the
off-diagonal momentum-conserving quartic generator.  Sparse SciPy
``expm_multiply`` is intentionally distinct from the production necklace
Chebyshev recurrence.  Occupation and sparse-matrix state is verification-only
and is not part of the accepted 285-cell path.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


GRID = 17
ROTORS = 4
PAIR_CHANNELS = 9
PRIMARY_DEPTH = 3
REUSE_DEPTH = 2
CONTROL_FLOOR = 1.0e-6


def rotate(histogram: tuple[int, ...], shift: int) -> tuple[int, ...]:
    result = [0] * GRID
    for mode, count in enumerate(histogram):
        result[(mode + shift) % GRID] = count
    return tuple(result)


def canonical(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate(histogram, shift) for shift in range(GRID))


def generate_histograms() -> list[tuple[int, ...]]:
    output: list[tuple[int, ...]] = []
    working = [0] * GRID

    def visit(position: int, remaining: int) -> None:
        if position == GRID - 1:
            working[position] = remaining
            output.append(tuple(working))
            return
        for value in range(remaining + 1):
            working[position] = value
            visit(position + 1, remaining - value)

    visit(0, ROTORS)
    if len(output) != math.comb(ROTORS + GRID - 1, ROTORS):
        raise RuntimeError("independent occupation topology failed")
    return output


def collision_count(histogram: tuple[int, ...]) -> int:
    return sum(count * (count - 1) // 2 for count in histogram)


def multinomial(histogram: tuple[int, ...]) -> int:
    denominator = math.prod(math.factorial(count) for count in histogram)
    return math.factorial(ROTORS) // denominator


def pair_signature(histogram: tuple[int, ...]) -> tuple[int, ...]:
    result = [collision_count(histogram)]
    for distance in range(1, PAIR_CHANNELS):
        result.append(
            sum(
                histogram[mode] * histogram[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(result) != math.comb(ROTORS, 2):
        raise RuntimeError("independent pair signature failed")
    return tuple(result)


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def pair_phase_exponent(
    signature: tuple[int, ...], step: int, program_tag: int
) -> int:
    return sum(
        signature[distance] * public_pair_weight(
            distance, step, program_tag
        )
        for distance in range(PAIR_CHANNELS)
    ) % GRID


def public_scattering_weight(
    signed_shift: int, step: int, program_tag: int
) -> float:
    positive = signed_shift % GRID
    if positive == 0:
        raise RuntimeError("zero shift reached independent scattering law")
    distance = min(positive, GRID - positive)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    sign = -1 if (distance + step + program_tag) % GRID % 3 == 0 else 1
    return 0.01 * sign * magnitude


def compile_scattering_bases(
    histograms: list[tuple[int, ...]],
    index: dict[tuple[int, ...], int],
) -> tuple[list[sparse.csr_matrix], int]:
    rows: list[list[int]] = [[] for _ in range(GRID // 2)]
    columns: list[list[int]] = [[] for _ in range(GRID // 2)]
    values: list[list[float]] = [[] for _ in range(GRID // 2)]
    streamed_terms = 0
    for target_index, histogram in enumerate(histograms):
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    source = list(histogram)
                    source[first] -= 1
                    source[second] -= 1
                    source[(first - shift) % GRID] += 1
                    source[(second + shift) % GRID] += 1
                    distance = min(shift, GRID - shift) - 1
                    rows[distance].append(target_index)
                    columns[distance].append(index[tuple(source)])
                    values[distance].append(0.5 * multiplicity)
                    streamed_terms += 1
    bases: list[sparse.csr_matrix] = []
    shape = (len(histograms), len(histograms))
    for distance in range(GRID // 2):
        matrix = sparse.coo_matrix(
            (values[distance], (rows[distance], columns[distance])),
            shape=shape,
            dtype=np.float64,
        ).tocsr()
        matrix.sum_duplicates()
        bases.append(matrix)
    return bases, streamed_terms


def compile_generator(
    bases: list[sparse.csr_matrix], step: int, program_tag: int
) -> sparse.csr_matrix:
    result = bases[0] * public_scattering_weight(1, step, program_tag)
    for distance in range(2, PAIR_CHANNELS):
        result = result + bases[distance - 1] * public_scattering_weight(
            distance, step, program_tag
        )
    result.sum_duplicates()
    return result.tocsr()


def apply_pair_phase(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    step: int,
    program_tag: int,
    adjoint: bool,
) -> None:
    weights = np.array(
        [
            public_pair_weight(distance, step, program_tag)
            for distance in range(PAIR_CHANNELS)
        ],
        dtype=np.int64,
    )
    exponents = signatures @ weights % GRID
    if adjoint:
        exponents = (-exponents) % GRID
    state *= roots[exponents]


def apply_scattering(
    state: np.ndarray,
    generator: sparse.csr_matrix,
    adjoint: bool,
) -> np.ndarray:
    sign = -1.0 if adjoint else 1.0
    operator = (1j * sign) * generator
    trace = complex(operator.diagonal().sum())
    return np.asarray(expm_multiply(operator, state, traceA=trace))


def forward_step(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    generator: sparse.csr_matrix,
    step: int,
    program_tag: int,
    zero_scattering: bool = False,
) -> np.ndarray:
    apply_pair_phase(state, signatures, roots, step, program_tag, False)
    if not zero_scattering:
        state = apply_scattering(state, generator, False)
    return state


def inverse_step(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    generator: sparse.csr_matrix,
    step: int,
    program_tag: int,
    reordered: bool,
) -> np.ndarray:
    if reordered:
        apply_pair_phase(state, signatures, roots, step, program_tag, True)
    state = apply_scattering(state, generator, True)
    if not reordered:
        apply_pair_phase(state, signatures, roots, step, program_tag, True)
    return state


def weighted_l2(
    left: np.ndarray, right: np.ndarray, multiplicities: np.ndarray
) -> float:
    return float(np.sqrt(np.sum(multiplicities * np.abs(left - right) ** 2)))


def project_boundary(
    state: np.ndarray,
    collisions: np.ndarray,
    multiplicities: np.ndarray,
) -> list[float]:
    result = np.zeros(math.comb(ROTORS, 2) + 1, dtype=np.float64)
    np.add.at(result, collisions, multiplicities * np.abs(state) ** 2)
    return [float(value) for value in result]


def transaction(
    initial: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    collisions: np.ndarray,
    multiplicities: np.ndarray,
    generator_for: object,
    depth: int,
    program_tag: int,
    control: Literal["correct", "missing", "wrong", "reordered"],
) -> dict[str, object]:
    state = initial.copy()
    for step in range(depth):
        state = forward_step(
            state,
            signatures,
            roots,
            generator_for(step, program_tag),
            step,
            program_tag,
        )
    boundary = project_boundary(state, collisions, multiplicities)
    norm_error = abs(float(np.sum(multiplicities * np.abs(state) ** 2)) - 1.0)
    minimum_step = 1 if control == "missing" else 0
    for step in range(depth - 1, minimum_step - 1, -1):
        inverse_tag = program_tag + 1 if control == "wrong" and step == depth - 1 else program_tag
        state = inverse_step(
            state,
            signatures,
            roots,
            generator_for(step, inverse_tag),
            step,
            inverse_tag,
            control == "reordered",
        )
    return {
        "boundary": boundary,
        "norm_error": norm_error,
        "restoration_error": weighted_l2(state, initial, multiplicities),
    }


def main() -> None:
    histograms = generate_histograms()
    histogram_index = {histogram: index for index, histogram in enumerate(histograms)}
    necklaces = [histogram for histogram in histograms if canonical(histogram) == histogram]
    if len(necklaces) != 285:
        raise RuntimeError("independent necklace count failed")
    necklace_index = {histogram: index for index, histogram in enumerate(necklaces)}
    roots = np.exp(2j * np.pi * np.arange(GRID) / GRID)
    collisions = np.array([collision_count(item) for item in histograms], dtype=np.int8)
    multiplicities = np.array([multinomial(item) for item in histograms], dtype=np.float64)
    signatures = np.array([pair_signature(item) for item in histograms], dtype=np.int64)
    initial = np.array(
        [
            roots[(7 * necklace_index[canonical(item)] + 3 * collision_count(item)) % GRID]
            / 289.0
            for item in histograms
        ],
        dtype=np.complex128,
    )
    if abs(float(np.sum(multiplicities * np.abs(initial) ** 2)) - 1.0) > 1e-14:
        raise RuntimeError("independent initial norm failed")

    bases, streamed_basis_terms = compile_scattering_bases(
        histograms, histogram_index
    )

    @lru_cache(maxsize=None)
    def generator_for(step: int, program_tag: int) -> sparse.csr_matrix:
        return compile_generator(bases, step, program_tag)

    generator_zero = generator_for(0, 0)
    diagonal_weights = sparse.diags(multiplicities, format="csr")
    weighted_difference = diagonal_weights @ generator_zero - generator_zero.T @ diagonal_weights
    weighted_hermitian_max_error = (
        0.0 if weighted_difference.nnz == 0 else float(np.max(np.abs(weighted_difference.data)))
    )
    if weighted_hermitian_max_error > 2e-13:
        raise RuntimeError("independent weighted Hermitian check failed")

    primary = transaction(
        initial,
        signatures,
        roots,
        collisions,
        multiplicities,
        generator_for,
        PRIMARY_DEPTH,
        0,
        "correct",
    )
    reuse = transaction(
        initial,
        signatures,
        roots,
        collisions,
        multiplicities,
        generator_for,
        REUSE_DEPTH,
        3,
        "correct",
    )
    missing = transaction(
        initial, signatures, roots, collisions, multiplicities,
        generator_for, 2, 0, "missing"
    )
    wrong = transaction(
        initial, signatures, roots, collisions, multiplicities,
        generator_for, 2, 0, "wrong"
    )
    reordered = transaction(
        initial, signatures, roots, collisions, multiplicities,
        generator_for, 2, 0, "reordered"
    )

    full = initial.copy()
    zero = initial.copy()
    for step in range(PRIMARY_DEPTH):
        full = forward_step(
            full, signatures, roots, generator_for(step, 0), step, 0
        )
        zero = forward_step(
            zero, signatures, roots, generator_for(step, 0), step, 0, True
        )
    zero_difference = max(
        abs(left - right)
        for left, right in zip(
            project_boundary(full, collisions, multiplicities),
            project_boundary(zero, collisions, multiplicities),
            strict=True,
        )
    )

    ordered = initial.copy()
    ordered = forward_step(
        ordered, signatures, roots, generator_zero, 0, 0
    )
    swapped = apply_scattering(initial.copy(), generator_zero, False)
    apply_pair_phase(swapped, signatures, roots, 0, 0, False)
    swapped_difference = max(
        abs(left - right)
        for left, right in zip(
            project_boundary(ordered, collisions, multiplicities),
            project_boundary(swapped, collisions, multiplicities),
            strict=True,
        )
    )

    canonical_groups: dict[tuple[int, ...], list[int]] = {}
    for index, histogram in enumerate(histograms):
        canonical_groups.setdefault(canonical(histogram), []).append(index)
    rotation_invariance_error = 0.0
    for indices in canonical_groups.values():
        reference = full[indices[0]]
        rotation_invariance_error = max(
            rotation_invariance_error,
            max(abs(full[index] - reference) for index in indices),
        )

    if (
        float(primary["restoration_error"]) > 2e-10
        or float(reuse["restoration_error"]) > 2e-10
        or float(primary["norm_error"]) > 2e-10
        or float(missing["restoration_error"]) < CONTROL_FLOOR
        or float(wrong["restoration_error"]) < CONTROL_FLOOR
        or float(reordered["restoration_error"]) < CONTROL_FLOOR
        or zero_difference < 1e-8
        or swapped_difference < 1e-8
        or rotation_invariance_error > 2e-10
    ):
        raise RuntimeError("independent scattering oracle gate failed")

    output = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "oracle_representation": "FULL_4845_CELL_EXCHANGE_SYMMETRIC_OCCUPATION_SPACE_VERIFICATION_ONLY",
        "occupation_cells": len(histograms),
        "necklace_cells_reconstructed": len(necklaces),
        "scattering_basis_count": len(bases),
        "scattering_basis_streamed_terms": streamed_basis_terms,
        "scattering_basis_total_nnz": sum(matrix.nnz for matrix in bases),
        "weighted_hermitian_max_error": weighted_hermitian_max_error,
        "rotation_invariance_error": rotation_invariance_error,
        "primary": primary,
        "reuse": reuse,
        "controls": {
            "missing_inverse_error": missing["restoration_error"],
            "wrong_inverse_error": wrong["restoration_error"],
            "reordered_inverse_error": reordered["restoration_error"],
            "zero_scattering_boundary_difference": zero_difference,
            "swapped_phase_scattering_boundary_difference": swapped_difference,
        },
        "accepted_path_occupation_cells": 0,
        "accepted_path_sparse_generator_cells": 0,
        "accepted_path_dense_necklace_operator_cells": 0,
        "matched_classical_recurrence": "IDENTICAL_285_COMPLEX_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE",
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catvm_custody": False,
        "physical_waveform_execution": False,
        "unbounded_computation_established": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
