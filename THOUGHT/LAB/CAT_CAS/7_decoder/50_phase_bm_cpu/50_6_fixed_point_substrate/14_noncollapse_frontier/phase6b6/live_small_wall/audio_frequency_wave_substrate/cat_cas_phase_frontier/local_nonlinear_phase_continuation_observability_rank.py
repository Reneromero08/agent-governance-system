#!/usr/bin/env python3
"""Analytic continuation-observability diagnostic for the M187 phase law.

The accepted transaction remains an M187 local phase carrier.  Separately,
this verifier differentiates a public family of one-layer continuation
boundaries on the unit-norm carrier manifold.  Full rank rejects only a
smaller regular differentiable quotient that must preserve every declared
continuation near the tested resident state; it is not a global or algorithmic
lower bound.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from controlled_local_interference_feedback_phase_coupling import (
    BOUNDARY_TOLERANCE,
    PublicProgram,
    boundary_close,
    commitment,
    complex_pair,
    forward_in_place,
    inverse_in_place,
    layer_parameters,
    make_program,
    maximum_error,
    project_boundary,
    source_state,
)


CLAIM = (
    "BOUNDED_ANALYTIC_LOCAL_CONTINUATION_OBSERVABILITY_DIAGNOSTIC_FOR_"
    "THE_REVERSIBLE_INTERFERENCE_FEEDBACK_PHASE_LAW_REACHES_FULL_"
    "TANGENT_RANK_2NMINUS1_ON_THE_DECLARED_N4_N8_N16_NORM_ONE_"
    "RESIDENT_CHARTS_UNDER_N_PUBLIC_ONE_LAYER_SUFFIX_BOUNDARIES_WITH_"
    "NUMERICAL_SAME_BACKING_RESTORATION_AND_REUSE_REJECTING_ONLY_"
    "SMALLER_REGULAR_DIFFERENTIABLE_LOCAL_QUOTIENTS_FOR_THOSE_"
    "CONTINUATIONS_WHILE_QUADRATIC_VERIFICATION_STATE_AND_THE_"
    "IDENTICAL_CLASSICAL_FULL_STATE_RECURRENCE_REMAIN"
)

WIDTHS = (4, 8, 16)
STATE_TOLERANCE = 2.0e-10
RANK_RELATIVE_TOLERANCE = 1.0e-9
REUSE_CYCLES = 64


def suffix_program(width: int, index: int) -> PublicProgram:
    return PublicProgram(
        width=width,
        depth=1,
        even_angle=0.101 + 0.137 * (index + 1) / width,
        odd_angle=0.233 + 0.071 * ((index * index + 3) % (width + 1)) / width,
        feedback_strength=0.43 + 1.17 * (index + 1) / width,
        phase_seed=2 * index + 1,
        weight_shift=3,
    )


def real_coordinates(state: np.ndarray) -> np.ndarray:
    coordinates = np.empty(2 * len(state), dtype=np.float64)
    coordinates[0::2] = state.real
    coordinates[1::2] = state.imag
    return coordinates


def tangent_basis(state: np.ndarray) -> np.ndarray:
    coordinates = real_coordinates(state)
    coordinates /= np.linalg.norm(coordinates)
    _, _, right = np.linalg.svd(coordinates.reshape(1, -1), full_matrices=True)
    basis = right[1:].T.copy()
    if basis.shape != (2 * len(state), 2 * len(state) - 1):
        raise RuntimeError("unexpected tangent basis shape")
    return basis


def basis_deltas(basis: np.ndarray) -> np.ndarray:
    real = basis[0::2, :].T
    imag = basis[1::2, :].T
    return (real + 1j * imag).astype(np.complex128)


def tangent_coupler_sublayer(
    state: np.ndarray, deltas: np.ndarray, offset: int, angle: float
) -> None:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for left in range(offset, len(state), 2):
        right = (left + 1) % len(state)
        a = complex(state[left])
        b = complex(state[right])
        da = deltas[:, left].copy()
        db = deltas[:, right].copy()
        state[left] = cosine * a + 1j * sine * b
        state[right] = 1j * sine * a + cosine * b
        deltas[:, left] = cosine * da + 1j * sine * db
        deltas[:, right] = 1j * sine * da + cosine * db


def tangent_feedback_sublayer(
    state: np.ndarray, deltas: np.ndarray, strength: float
) -> None:
    for index in range(len(state)):
        value = complex(state[index])
        phase = strength * abs(value) ** 2
        multiplier = complex(math.cos(phase), math.sin(phase))
        perturbations = deltas[:, index].copy()
        radius_derivatives = 2.0 * np.real(np.conjugate(value) * perturbations)
        deltas[:, index] = multiplier * (
            perturbations + 1j * strength * value * radius_derivatives
        )
        state[index] = value * multiplier


def analytic_suffix_derivatives(
    resident: np.ndarray, basis: np.ndarray, suffix: PublicProgram
) -> tuple[complex, np.ndarray, float]:
    state = resident.copy()
    deltas = basis_deltas(basis)
    even, odd, feedback = layer_parameters(suffix, 0)
    tangent_coupler_sublayer(state, deltas, 0, even)
    tangent_coupler_sublayer(state, deltas, 1, odd)
    tangent_feedback_sublayer(state, deltas, feedback)

    direct = resident.copy()
    forward_in_place(direct, suffix)
    forward_parity_error = maximum_error(state, direct)

    scale = math.sqrt(suffix.width)
    derivatives = np.zeros(deltas.shape[0], dtype=np.complex128)
    boundary = 0j
    for index, value in enumerate(state):
        angle = 2.0 * math.pi * suffix.weight_shift * index / suffix.width
        weight = complex(math.cos(angle), math.sin(angle)) / scale
        boundary += weight.conjugate() * complex(value)
        derivatives += weight.conjugate() * deltas[:, index]
    return boundary, derivatives, forward_parity_error


def observability_certificate(
    resident: np.ndarray, suffixes: list[PublicProgram]
) -> dict[str, Any]:
    basis = tangent_basis(resident)
    rows: list[np.ndarray] = []
    boundaries: list[complex] = []
    forward_parity_max = 0.0
    for suffix in suffixes:
        boundary, derivatives, parity = analytic_suffix_derivatives(
            resident, basis, suffix
        )
        boundaries.append(boundary)
        rows.append(derivatives.real.copy())
        rows.append(derivatives.imag.copy())
        forward_parity_max = max(forward_parity_max, parity)
    matrix = np.vstack(rows)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = RANK_RELATIVE_TOLERANCE * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    target = 2 * len(resident) - 1

    removed_matrix = matrix[:-2, :]
    removed_singular = np.linalg.svd(removed_matrix, compute_uv=False)
    removed_threshold = RANK_RELATIVE_TOLERANCE * float(removed_singular[0])
    removed_rank = int(np.count_nonzero(removed_singular > removed_threshold))

    duplicate_matrix = np.vstack([matrix[:2, :]] * len(suffixes))
    duplicate_singular = np.linalg.svd(duplicate_matrix, compute_uv=False)
    duplicate_threshold = RANK_RELATIVE_TOLERANCE * float(duplicate_singular[0])
    duplicate_rank = int(np.count_nonzero(duplicate_singular > duplicate_threshold))

    zero_suffixes = [replace(suffix, feedback_strength=0.0) for suffix in suffixes]
    zero_rows: list[np.ndarray] = []
    for suffix in zero_suffixes:
        _, derivatives, _ = analytic_suffix_derivatives(resident, basis, suffix)
        zero_rows.extend((derivatives.real.copy(), derivatives.imag.copy()))
    zero_matrix = np.vstack(zero_rows)

    n = len(resident)
    verification_reals = 6 * n * (2 * n - 1) + 10 * n - 1
    descriptor_bytes = sum(
        len(
            json.dumps(
                suffix.as_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        for suffix in suffixes
    )
    return {
        "tangent_dimension": target,
        "public_suffix_count": len(suffixes),
        "observability_matrix_shape": list(matrix.shape),
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "rank_threshold": threshold,
        "observed_rank": rank,
        "full_tangent_rank": rank == target,
        "singular_values": [float(value) for value in singular_values],
        "minimum_retained_singular_value": float(singular_values[target - 1]),
        "condition_number_on_tangent": float(
            singular_values[0] / singular_values[target - 1]
        ),
        "analytic_forward_parity_max_error": forward_parity_max,
        "remove_one_suffix_rank": removed_rank,
        "remove_one_suffix_forces_rank_below_tangent_dimension": removed_rank < target,
        "duplicate_one_suffix_rank": duplicate_rank,
        "duplicate_suffix_underobserves": duplicate_rank < target,
        "zero_feedback_changes_analytic_matrix": float(
            np.max(np.abs(matrix - zero_matrix))
        )
        > BOUNDARY_TOLERANCE,
        "boundary_commitment": __import__("hashlib").sha256(
            np.asarray(boundaries, dtype=np.complex128).tobytes()
        ).hexdigest(),
        "accepted_carrier_logical_complex_cells": n + 2,
        "analytic_verification_conservative_named_real_scalars": verification_reals,
        "analytic_verification_scaling": "THETA_N_SQUARED",
        "public_suffix_descriptor_scalars": 7 * n,
        "public_suffix_descriptor_json_bytes": descriptor_bytes,
        "analytic_state_coupler_operations": 2 * n * n,
        "analytic_state_feedback_operations": 2 * n * n,
        "analytic_tangent_coupler_operations": n * n * (2 * n - 1),
        "analytic_tangent_feedback_operations": n * n * (2 * n - 1),
        "analytic_boundary_projection_terms": n * n,
        "analytic_tangent_projection_terms": n * n * (2 * n - 1),
    }


def execute_transaction(width: int) -> dict[str, Any]:
    prefix = make_program(width, width, 0)
    second_prefix = make_program(width, width, 1)
    suffixes = [suffix_program(width, index) for index in range(width)]

    carrier = source_state(prefix)
    initial = carrier.copy()
    backing = id(carrier)
    pre_commitment = commitment(carrier)
    forward_in_place(carrier, prefix)
    resident_commitment = commitment(carrier)
    certificate = observability_certificate(carrier, suffixes)

    selected = suffixes[0]
    forward_in_place(carrier, selected)
    boundary = project_boundary(carrier, selected)
    persisted = complex(boundary)
    inverse_in_place(carrier, selected)
    inverse_in_place(carrier, prefix)
    restoration_error = maximum_error(carrier, initial)
    if restoration_error > STATE_TOLERANCE or id(carrier) != backing:
        raise RuntimeError("primary observability carrier restoration failed")

    reuse_suffix = suffixes[-1]
    forward_in_place(carrier, second_prefix)
    forward_in_place(carrier, reuse_suffix)
    reuse_boundary = project_boundary(carrier, reuse_suffix)
    inverse_in_place(carrier, reuse_suffix)
    inverse_in_place(carrier, second_prefix)
    reuse_restoration_error = maximum_error(carrier, initial)
    fresh = source_state(second_prefix)
    forward_in_place(fresh, second_prefix)
    forward_in_place(fresh, reuse_suffix)
    fresh_boundary = project_boundary(fresh, reuse_suffix)
    if (
        reuse_restoration_error > STATE_TOLERANCE
        or not boundary_close(reuse_boundary, fresh_boundary)
        or id(carrier) != backing
    ):
        raise RuntimeError("restored observability carrier reuse failed")

    repeated_max = 0.0
    for cycle in range(REUSE_CYCLES):
        active_prefix = prefix if cycle % 2 == 0 else second_prefix
        active_suffix = suffixes[cycle % len(suffixes)]
        forward_in_place(carrier, active_prefix)
        forward_in_place(carrier, active_suffix)
        inverse_in_place(carrier, active_suffix)
        inverse_in_place(carrier, active_prefix)
        repeated_max = max(repeated_max, maximum_error(carrier, initial))
    if repeated_max > STATE_TOLERANCE:
        raise RuntimeError("observability carrier reuse drift exceeds tolerance")

    # Controls operate on copies and do not alter the accepted backing.
    missing = initial.copy()
    forward_in_place(missing, prefix)
    forward_in_place(missing, selected)
    missing_error = maximum_error(missing, initial)
    wrong = missing.copy()
    inverse_in_place(wrong, replace(selected, feedback_strength=selected.feedback_strength + 0.09))
    inverse_in_place(wrong, prefix)
    wrong_error = maximum_error(wrong, initial)
    reordered = missing.copy()
    inverse_in_place(reordered, prefix)
    inverse_in_place(reordered, selected)
    reordered_error = maximum_error(reordered, initial)
    unit_program_work = width * width + width

    return {
        "width": width,
        "prefix_depth": width,
        "public_suffix_count": width,
        "final_boundary": complex_pair(persisted),
        "reuse_boundary": complex_pair(reuse_boundary),
        "pre_state_commitment": pre_commitment,
        "resident_intermediate_commitment": resident_commitment,
        "same_backing": id(carrier) == backing,
        "restoration_error": restoration_error,
        "reuse_restoration_error": reuse_restoration_error,
        "reuse_matches_fresh": boundary_close(reuse_boundary, fresh_boundary),
        "repeated_reuse_cycles": REUSE_CYCLES,
        "repeated_reuse_max_error": repeated_max,
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "response_order_machine_enforced": False,
        "accepted_carrier_logical_complex_cells": width + 2,
        "accepted_primary_reuse_coupler_operations": 4 * unit_program_work,
        "accepted_primary_reuse_feedback_operations": 4 * unit_program_work,
        "accepted_primary_reuse_projection_terms": 2 * width,
        "verification_reuse_and_inverse_control_coupler_operations": (
            2 * REUSE_CYCLES + 4
        )
        * unit_program_work,
        "verification_reuse_and_inverse_control_feedback_operations": (
            2 * REUSE_CYCLES + 4
        )
        * unit_program_work,
        "verification_fresh_projection_terms": width,
        "commitment_serialization_bytes": 2 * 16 * width,
        "public_prefix_descriptor_scalars": 14,
        "certificate": certificate,
        "controls": {
            "missing_inverse_fails": missing_error > 1.0e-8,
            "wrong_inverse_fails": wrong_error > 1.0e-8,
            "reordered_prefix_suffix_inverse_fails": reordered_error > 1.0e-8,
            "remove_one_suffix_underobserves": certificate[
                "remove_one_suffix_forces_rank_below_tangent_dimension"
            ],
            "duplicate_suffix_underobserves": certificate[
                "duplicate_suffix_underobserves"
            ],
            "zero_feedback_changes_analytic_matrix": certificate[
                "zero_feedback_changes_analytic_matrix"
            ],
        },
    }


def build_result() -> dict[str, Any]:
    cases = [execute_transaction(width) for width in WIDTHS]
    all_controls = all(all(case["controls"].values()) for case in cases)
    if not all_controls:
        raise RuntimeError("observability control failed")
    if not all(case["certificate"]["full_tangent_rank"] for case in cases):
        raise RuntimeError("declared continuation family did not reach full tangent rank")
    return {
        "schema": "CAT_CAS_LOCAL_NONLINEAR_PHASE_CONTINUATION_OBSERVABILITY_RANK_V1",
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "execution_scope": "LINUX_DIRECT_PROCESS_COMPLEX128_VIRTUAL_PHASE_SOFTWARE",
        "predeclared_state_tolerance": STATE_TOLERANCE,
        "predeclared_rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "cases": cases,
        "controls": {
            "all_full_tangent_rank": True,
            "all_controls_discriminate": all_controls,
            "all_restorations_within_tolerance": all(
                case["restoration_error"] <= STATE_TOLERANCE
                and case["reuse_restoration_error"] <= STATE_TOLERANCE
                and case["repeated_reuse_max_error"] <= STATE_TOLERANCE
                for case in cases
            ),
        },
        "claim_ceiling": {
            "rejected_quotient": "REGULAR_DIFFERENTIABLE_LOCAL_QUOTIENT_BELOW_2N_MINUS_1_REAL_DIMENSIONS_PRESERVING_ALL_DECLARED_SUFFIX_BOUNDARIES_AT_TESTED_CHARTS",
            "global_nonlinear_quotient_rejected": False,
            "arbitrary_continuation_family_rejected": False,
            "algorithmic_state_lower_bound_established": False,
            "distinct_phase_resource_established": False,
            "small_wall_crossing_established": False,
            "catvm_custody": False,
            "physical_waveform_execution": False,
        },
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
