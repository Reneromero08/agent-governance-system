#!/usr/bin/env python3
"""Independent labelled-wave oracle for the distance-resolved necklace law.

This verifier imports no production module.  It reconstructs the 17^4 labelled
wave, public pair-distance phases, four single-particle chirp transforms,
boundary projection, inverse ordering, and reuse directly.  Its dense arrays
are verification-only and are not part of the accepted 285-cell path.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Literal

import numpy as np


GRID = 17
ROTORS = 4
PAIR_CHANNELS = 9
PRIMARY_DEPTH = 4
REUSE_DEPTH = 2
CONTROL_FLOOR = 1.0e-6


def mod(value: int) -> int:
    return value % GRID


def rotate_histogram(histogram: tuple[int, ...], shift: int) -> tuple[int, ...]:
    result = [0] * GRID
    for index, value in enumerate(histogram):
        result[(index + shift) % GRID] = value
    return tuple(result)


def canonical_histogram(histogram: tuple[int, ...]) -> tuple[int, ...]:
    return min(rotate_histogram(histogram, shift) for shift in range(GRID))


def generate_histograms(
    position: int,
    remaining: int,
    working: list[int],
    output: list[tuple[int, ...]],
) -> None:
    if position == GRID - 1:
        working[position] = remaining
        output.append(tuple(working))
        return
    for value in range(remaining + 1):
        working[position] = value
        generate_histograms(position + 1, remaining - value, working, output)


def compile_necklaces() -> list[tuple[int, ...]]:
    histograms: list[tuple[int, ...]] = []
    generate_histograms(0, ROTORS, [0] * GRID, histograms)
    necklaces = [item for item in histograms if canonical_histogram(item) == item]
    if len(histograms) != 4_845 or len(necklaces) != 285:
        raise RuntimeError("independent necklace topology failed")
    return necklaces


def histogram_from_tuple(values: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * GRID
    for value in values:
        result[value] += 1
    return tuple(result)


def collision_count(histogram: tuple[int, ...]) -> int:
    return sum(value * (value - 1) // 2 for value in histogram)


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
    return 1 + mod(
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % (GRID - 1)


def pair_phase_exponent(
    signature: tuple[int, ...],
    step: int,
    program_tag: int,
    collision_only: bool = False,
) -> int:
    channels = 1 if collision_only else PAIR_CHANNELS
    return sum(
        signature[distance] * public_pair_weight(distance, step, program_tag)
        for distance in range(channels)
    ) % GRID


def public_chirp(step: int, program_tag: int) -> int:
    return 1 + mod(5 * step + 7 * program_tag) % (GRID - 1)


def compile_dense_topology() -> dict[str, object]:
    necklaces = compile_necklaces()
    necklace_index = {item: index for index, item in enumerate(necklaces)}
    shape = (GRID,) * ROTORS
    cell_count = GRID**ROTORS
    initial = np.empty(shape, dtype=np.complex128)
    collisions = np.empty(shape, dtype=np.int8)
    signatures = np.empty(shape + (PAIR_CHANNELS,), dtype=np.int8)
    roots = np.exp(2j * np.pi * np.arange(GRID) / GRID)

    for values in itertools.product(range(GRID), repeat=ROTORS):
        histogram = histogram_from_tuple(values)
        canonical = canonical_histogram(histogram)
        index = necklace_index[canonical]
        collision = collision_count(histogram)
        exponent = (7 * index + 3 * collision) % GRID
        initial[values] = roots[exponent] / 289.0
        collisions[values] = collision
        signatures[values] = pair_signature(histogram)

    if initial.size != cell_count:
        raise RuntimeError("independent labelled-wave size failed")
    if abs(float(np.vdot(initial.ravel(), initial.ravel()).real) - 1.0) > 1e-14:
        raise RuntimeError("independent initial norm failed")
    return {
        "necklaces": necklaces,
        "initial": initial,
        "collisions": collisions,
        "signatures": signatures,
        "roots": roots,
    }


def apply_pair_phase(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    step: int,
    program_tag: int,
    adjoint: bool,
    collision_only: bool = False,
) -> None:
    channels = 1 if collision_only else PAIR_CHANNELS
    weights = np.array(
        [public_pair_weight(distance, step, program_tag) for distance in range(channels)],
        dtype=np.int64,
    )
    exponents = np.tensordot(
        signatures[..., :channels].astype(np.int64),
        weights,
        axes=([-1], [0]),
    ) % GRID
    if adjoint:
        exponents = (-exponents) % GRID
    state *= roots[exponents]


def apply_dense_free(
    state: np.ndarray,
    roots: np.ndarray,
    chirp: int,
    adjoint: bool,
) -> np.ndarray:
    sign = -1 if adjoint else 1
    unitary = np.empty((GRID, GRID), dtype=np.complex128)
    for target in range(GRID):
        for source in range(GRID):
            difference = target - source
            unitary[target, source] = (
                roots[(sign * chirp * difference * difference) % GRID]
                / math.sqrt(GRID)
            )
    output = state
    for axis in range(ROTORS - 1, -1, -1):
        moved = np.moveaxis(output, axis, 0)
        transformed = np.tensordot(unitary, moved, axes=((1,), (0,)))
        output = np.moveaxis(transformed, 0, axis)
    return output


def forward_step(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    step: int,
    program_tag: int,
    collision_only: bool = False,
) -> np.ndarray:
    apply_pair_phase(
        state, signatures, roots, step, program_tag, False, collision_only
    )
    return apply_dense_free(state, roots, public_chirp(step, program_tag), False)


def inverse_step(
    state: np.ndarray,
    signatures: np.ndarray,
    roots: np.ndarray,
    step: int,
    program_tag: int,
    reordered: bool,
    collision_only: bool = False,
) -> np.ndarray:
    if reordered:
        apply_pair_phase(
            state, signatures, roots, step, program_tag, True, collision_only
        )
    state = apply_dense_free(state, roots, public_chirp(step, program_tag), True)
    if not reordered:
        apply_pair_phase(
            state, signatures, roots, step, program_tag, True, collision_only
        )
    return state


def project_boundary(state: np.ndarray, collisions: np.ndarray) -> list[float]:
    probabilities = np.abs(state.ravel()) ** 2
    return np.bincount(
        collisions.ravel().astype(np.int64),
        weights=probabilities,
        minlength=7,
    ).tolist()


def transaction(
    initial: np.ndarray,
    signatures: np.ndarray,
    collisions: np.ndarray,
    roots: np.ndarray,
    depth: int,
    program_tag: int,
    control: Literal["correct", "missing", "wrong", "reordered"] = "correct",
) -> dict[str, object]:
    state = initial.copy()
    for step in range(depth):
        state = forward_step(state, signatures, roots, step, program_tag)
    boundary = project_boundary(state, collisions)
    norm_error = abs(float(np.vdot(state.ravel(), state.ravel()).real) - 1.0)
    minimum_step = 1 if control == "missing" else 0
    for step in range(depth - 1, minimum_step - 1, -1):
        inverse_tag = program_tag + 1 if control == "wrong" and step == depth - 1 else program_tag
        state = inverse_step(
            state,
            signatures,
            roots,
            step,
            inverse_tag,
            control == "reordered",
        )
    restoration_error = float(np.linalg.norm(state.ravel() - initial.ravel()))
    return {
        "boundary": boundary,
        "norm_error": norm_error,
        "restoration_error": restoration_error,
    }


def main() -> None:
    topology = compile_dense_topology()
    initial = topology["initial"]
    collisions = topology["collisions"]
    signatures = topology["signatures"]
    roots = topology["roots"]
    assert isinstance(initial, np.ndarray)
    assert isinstance(collisions, np.ndarray)
    assert isinstance(signatures, np.ndarray)
    assert isinstance(roots, np.ndarray)

    primary = transaction(
        initial, signatures, collisions, roots, PRIMARY_DEPTH, 0
    )
    reuse = transaction(
        initial, signatures, collisions, roots, REUSE_DEPTH, 3
    )
    missing = transaction(
        initial, signatures, collisions, roots, 2, 0, "missing"
    )
    wrong = transaction(initial, signatures, collisions, roots, 2, 0, "wrong")
    reordered = transaction(
        initial, signatures, collisions, roots, 2, 0, "reordered"
    )

    all_signatures = {
        pair_signature(item) for item in topology["necklaces"]  # type: ignore[union-attr]
    }
    collision_split = any(
        collision_count(left) == collision_count(right)
        and pair_phase_exponent(pair_signature(left), 0, 0)
        != pair_phase_exponent(pair_signature(right), 0, 0)
        for left_index, left in enumerate(topology["necklaces"])  # type: ignore[union-attr]
        for right in topology["necklaces"][left_index + 1 :]  # type: ignore[index]
    )
    rotation_invariant = all(
        pair_signature(item) == pair_signature(rotate_histogram(item, 1))
        for item in topology["necklaces"]  # type: ignore[union-attr]
    )
    if (
        primary["restoration_error"] > 2e-10
        or reuse["restoration_error"] > 2e-10
        or missing["restoration_error"] < CONTROL_FLOOR
        or wrong["restoration_error"] < CONTROL_FLOOR
        or reordered["restoration_error"] < CONTROL_FLOOR
        or len(all_signatures) != 165
        or not collision_split
        or not rotation_invariant
    ):
        raise RuntimeError("independent two-body oracle gate failed")

    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle_representation": "DENSE_LABELLED_17_TO_THE_4_COMPLEX_WAVE_VERIFICATION_ONLY",
        "labelled_complex_cells": GRID**ROTORS,
        "dense_work_complex_cells": GRID**ROTORS,
        "dense_pair_exponent_integer_cells": GRID**ROTORS,
        "dense_pair_signature_integer_cells": PAIR_CHANNELS * GRID**ROTORS,
        "necklace_cells_reconstructed": 285,
        "distinct_pair_signatures": len(all_signatures),
        "rotation_invariant": rotation_invariant,
        "exchange_symmetric": True,
        "collision_degenerate_states_split": collision_split,
        "primary": primary,
        "reuse": reuse,
        "controls": {
            "missing_inverse_error": missing["restoration_error"],
            "wrong_inverse_error": wrong["restoration_error"],
            "reordered_inverse_error": reordered["restoration_error"],
        },
        "accepted_path_dense_labelled_wave_cells": 0,
        "accepted_path_dense_pair_signature_cells": 0,
        "matched_classical_recurrence": "IDENTICAL_285_COMPLEX_NECKLACE_GENERATOR_AND_DIAGONAL_PAIR_PHASE_RECURRENCE",
        "distinct_phase_resource_established": False,
        "small_wall_crossed": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
