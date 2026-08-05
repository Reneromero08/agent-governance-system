#!/usr/bin/env python3
"""Independent centered-finite-difference oracle for M188.

This file imports neither the analytic diagnostic nor M187 production.  It
reconstructs the local phase map, resident charts, continuation descriptors,
norm-one tangent bases, centered derivatives, and rank controls directly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


WIDTHS = (4, 8, 16)
RANK_RELATIVE_TOLERANCE = 1.0e-9
FINITE_DIFFERENCE_STEP = 1.0e-6
STATE_TOLERANCE = 2.0e-10
REUSE_CYCLES = 64


@dataclass(frozen=True)
class Program:
    width: int
    depth: int
    even_angle: float
    odd_angle: float
    feedback_strength: float
    phase_seed: int
    weight_shift: int


def prefix(width: int, variant: int) -> Program:
    return Program(
        width,
        width,
        0.173 + 0.019 * variant,
        0.287 - 0.013 * variant,
        0.71 + 0.11 * variant,
        3 + 2 * variant,
        3 + variant,
    )


def suffix(width: int, index: int) -> Program:
    return Program(
        width,
        1,
        0.101 + 0.137 * (index + 1) / width,
        0.233 + 0.071 * ((index * index + 3) % (width + 1)) / width,
        0.43 + 1.17 * (index + 1) / width,
        2 * index + 1,
        3,
    )


def initial_state(width: int) -> np.ndarray:
    state = np.empty(width, dtype=np.complex128)
    for index in range(width):
        radius = 1.0 + 0.19 * math.cos(2.0 * math.pi * (index + 2) / width)
        angle = 2.0 * math.pi * (index * index + 3 * index + 1) / (width + 3)
        state[index] = radius * complex(math.cos(angle), math.sin(angle))
    state /= np.linalg.norm(state)
    return state


def layer_values(item: Program, layer: int) -> tuple[float, float, float]:
    return (
        item.even_angle + 0.003 * ((3 * layer + item.phase_seed) % 11),
        item.odd_angle - 0.002 * ((5 * layer + 2 * item.phase_seed) % 13),
        item.feedback_strength * (1.0 + 0.01 * ((layer + item.phase_seed) % 7)),
    )


def pair_layer(state: np.ndarray, offset: int, angle: float) -> None:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for left in range(offset, len(state), 2):
        right = (left + 1) % len(state)
        a = complex(state[left])
        b = complex(state[right])
        state[left] = cosine * a + 1j * sine * b
        state[right] = 1j * sine * a + cosine * b


def feedback_layer(state: np.ndarray, strength: float) -> None:
    for index, value_ in enumerate(state):
        value = complex(value_)
        angle = strength * abs(value) ** 2
        state[index] = value * complex(math.cos(angle), math.sin(angle))


def execute(state: np.ndarray, item: Program) -> None:
    for layer in range(item.depth):
        even, odd, feedback = layer_values(item, layer)
        pair_layer(state, 0, even)
        pair_layer(state, 1, odd)
        feedback_layer(state, feedback)


def reverse(state: np.ndarray, item: Program) -> None:
    for layer in reversed(range(item.depth)):
        even, odd, feedback = layer_values(item, layer)
        feedback_layer(state, -feedback)
        pair_layer(state, 1, -odd)
        pair_layer(state, 0, -even)


def projection(state: np.ndarray, item: Program) -> complex:
    total = 0j
    scale = math.sqrt(item.width)
    for index, value in enumerate(state):
        angle = 2.0 * math.pi * item.weight_shift * index / item.width
        weight = complex(math.cos(angle), math.sin(angle)) / scale
        total += weight.conjugate() * complex(value)
    return total


def tangent_basis(state: np.ndarray) -> np.ndarray:
    coordinates = np.empty(2 * len(state), dtype=np.float64)
    coordinates[0::2] = state.real
    coordinates[1::2] = state.imag
    coordinates /= np.linalg.norm(coordinates)
    _, _, right = np.linalg.svd(coordinates.reshape(1, -1), full_matrices=True)
    return right[1:].T.copy()


def perturb(state: np.ndarray, direction: np.ndarray, amount: float) -> np.ndarray:
    changed = state.copy()
    changed.real += amount * direction[0::2]
    changed.imag += amount * direction[1::2]
    changed /= np.linalg.norm(changed)
    return changed


def finite_difference_matrix(
    resident: np.ndarray, suffixes: list[Program]
) -> tuple[np.ndarray, int]:
    basis = tangent_basis(resident)
    rows: list[np.ndarray] = []
    forward_evaluations = 0
    for continuation in suffixes:
        derivatives = np.empty(basis.shape[1], dtype=np.complex128)
        for column in range(basis.shape[1]):
            plus = perturb(resident, basis[:, column], FINITE_DIFFERENCE_STEP)
            minus = perturb(resident, basis[:, column], -FINITE_DIFFERENCE_STEP)
            execute(plus, continuation)
            execute(minus, continuation)
            derivatives[column] = (
                projection(plus, continuation) - projection(minus, continuation)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)
            forward_evaluations += 2
        rows.extend((derivatives.real.copy(), derivatives.imag.copy()))
    return np.vstack(rows), forward_evaluations


def rank_certificate(matrix: np.ndarray) -> tuple[int, list[float], float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = RANK_RELATIVE_TOLERANCE * float(singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    return rank, [float(value) for value in singular], threshold


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return max(abs(complex(a) - complex(b)) for a, b in zip(left, right, strict=True))


def run_case(width: int) -> dict[str, Any]:
    primary = prefix(width, 0)
    second = prefix(width, 1)
    suffixes = [suffix(width, index) for index in range(width)]
    initial = initial_state(width)
    resident = initial.copy()
    execute(resident, primary)

    matrix, evaluations = finite_difference_matrix(resident, suffixes)
    rank, singular, threshold = rank_certificate(matrix)
    target = 2 * width - 1
    removed_rank, _, _ = rank_certificate(matrix[:-2, :])
    duplicate_rank, _, _ = rank_certificate(np.vstack([matrix[:2, :]] * width))

    zero_suffixes = [replace(item, feedback_strength=0.0) for item in suffixes]
    zero_matrix, zero_evaluations = finite_difference_matrix(resident, zero_suffixes)

    carrier = initial.copy()
    execute(carrier, primary)
    execute(carrier, suffixes[0])
    final_boundary = projection(carrier, suffixes[0])
    reverse(carrier, suffixes[0])
    reverse(carrier, primary)
    restoration_error = max_error(carrier, initial)

    execute(carrier, second)
    execute(carrier, suffixes[-1])
    reuse_boundary = projection(carrier, suffixes[-1])
    reverse(carrier, suffixes[-1])
    reverse(carrier, second)
    reuse_error = max_error(carrier, initial)

    fresh = initial.copy()
    execute(fresh, second)
    execute(fresh, suffixes[-1])
    fresh_boundary = projection(fresh, suffixes[-1])

    repeated_max = 0.0
    for cycle in range(REUSE_CYCLES):
        active_prefix = primary if cycle % 2 == 0 else second
        active_suffix = suffixes[cycle % width]
        execute(carrier, active_prefix)
        execute(carrier, active_suffix)
        reverse(carrier, active_suffix)
        reverse(carrier, active_prefix)
        repeated_max = max(repeated_max, max_error(carrier, initial))

    if rank != target or restoration_error > STATE_TOLERANCE or reuse_error > STATE_TOLERANCE:
        raise RuntimeError("independent oracle failed declared scope")
    if abs(reuse_boundary - fresh_boundary) > 2.0e-11 or repeated_max > STATE_TOLERANCE:
        raise RuntimeError("independent oracle reuse failed")

    return {
        "width": width,
        "tangent_dimension": target,
        "public_suffix_count": width,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "rank_threshold": threshold,
        "finite_difference_rank": rank,
        "full_tangent_rank": rank == target,
        "singular_values": singular,
        "minimum_retained_singular_value": singular[target - 1],
        "remove_one_suffix_rank": removed_rank,
        "remove_one_suffix_underobserves": removed_rank < target,
        "duplicate_one_suffix_rank": duplicate_rank,
        "duplicate_suffix_underobserves": duplicate_rank < target,
        "zero_feedback_changes_finite_difference_matrix": float(
            np.max(np.abs(matrix - zero_matrix))
        )
        > 2.0e-11,
        "finite_difference_forward_evaluations": evaluations,
        "zero_feedback_control_forward_evaluations": zero_evaluations,
        "final_boundary": [float(final_boundary.real), float(final_boundary.imag)],
        "reuse_boundary": [float(reuse_boundary.real), float(reuse_boundary.imag)],
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_error,
        "reuse_matches_fresh": abs(reuse_boundary - fresh_boundary) <= 2.0e-11,
        "repeated_reuse_cycles": REUSE_CYCLES,
        "repeated_reuse_max_error": repeated_max,
    }


def build_result() -> dict[str, Any]:
    cases = [run_case(width) for width in WIDTHS]
    return {
        "schema": "CAT_CAS_LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_INDEPENDENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "oracle": "STANDALONE_CENTERED_FINITE_DIFFERENCE_NO_PRODUCTION_IMPORT",
        "cases": cases,
        "all_full_tangent_rank": all(case["full_tangent_rank"] for case in cases),
        "all_controls_discriminate": all(
            case["remove_one_suffix_underobserves"]
            and case["duplicate_suffix_underobserves"]
            and case["zero_feedback_changes_finite_difference_matrix"]
            for case in cases
        ),
        "all_restorations_within_tolerance": all(
            case["restoration_error"] <= STATE_TOLERANCE
            and case["reuse_restoration_error"] <= STATE_TOLERANCE
            and case["repeated_reuse_max_error"] <= STATE_TOLERANCE
            for case in cases
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
