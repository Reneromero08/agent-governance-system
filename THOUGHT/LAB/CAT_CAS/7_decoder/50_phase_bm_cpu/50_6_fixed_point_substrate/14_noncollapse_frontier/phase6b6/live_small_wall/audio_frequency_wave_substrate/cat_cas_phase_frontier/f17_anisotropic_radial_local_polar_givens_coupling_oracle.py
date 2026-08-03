#!/usr/bin/env python3
"""Independent oracle for the bounded F17 local-polar Givens package.

This source imports neither the production module nor any predecessor.  It
reconstructs the weighted radial operator in long-double arithmetic, compiles
a separate float64 Givens plan from the public formula, reexecutes all declared
phase-pair cases and reuse paths, and attacks plan, weighting, and inverse
ordering.  The dense operator exists only in this verification oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


P = 17
TAU = 2.0 * math.pi
LTAU = 2 * np.arccos(np.longdouble(-1))
SHELL_COUNTS = (1, *([18] * 16))
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
ROTATION_COUNT = 136
RESTORATION_TOLERANCE = 2.0e-11
STATE_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 1.0e-10
PLAN_TOLERANCE = 5.0e-14
ZERO_FLOOR = 1.0e-14
RADIUS_SLACK = 2.0e-12
CONTROL_FLOOR = 1.0e-5
REUSE_DEPTH = 1537
REPEATED_DEPTH = 64
REPEATED_CYCLES = 100


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def wrap_scalar(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def complex_from_pair(value: list[float]) -> complex:
    return complex(float(value[0]), float(value[1]))


def phasor_pair(left: float, right: float) -> complex:
    return 0.5 * (
        complex(math.cos(left), math.sin(left))
        + complex(math.cos(right), math.sin(right))
    )


def chart(real: float, imag: float) -> tuple[float, float, float, bool]:
    magnitude = math.hypot(real, imag)
    if not math.isfinite(magnitude) or magnitude > 1.0 + RADIUS_SLACK:
        fail("independent local chart left unit disk")
    if magnitude <= ZERO_FLOOR:
        return math.pi / 2.0, -math.pi / 2.0, magnitude, True
    phase = math.atan2(imag, real)
    delta = math.acos(min(1.0, max(0.0, magnitude)))
    return wrap_scalar(phase + delta), wrap_scalar(phase - delta), magnitude, False


def scale(shell: int) -> float:
    return math.sqrt(float(SHELL_COUNTS[shell]))


def seed_angles() -> np.ndarray:
    result = np.empty((P, 2), dtype=np.float64)
    for shell in range(P):
        result[shell] = chart(scale(shell) / P, 0.0)[:2]
    return result


def angle_state(angles: np.ndarray) -> np.ndarray:
    return np.asarray(
        [phasor_pair(float(row[0]), float(row[1])) for row in angles],
        dtype=np.complex128,
    )


def physical_error(left: np.ndarray, right: np.ndarray) -> float:
    maximum = 0.0
    for shell in range(P):
        for slot in range(2):
            maximum = max(
                maximum,
                abs(
                    complex(
                        math.cos(float(left[shell, slot])),
                        math.sin(float(left[shell, slot])),
                    )
                    - complex(
                        math.cos(float(right[shell, slot])),
                        math.sin(float(right[shell, slot])),
                    )
                ),
            )
    return maximum


def gate_parameters(index: int, family: str) -> tuple[int, int, int]:
    bit_weight = index.bit_count()
    gray_weight = (index ^ (index >> 1)).bit_count()
    ternary_weight = 0
    remaining = index
    while remaining:
        ternary_weight += remaining % 3
        remaining //= 3
    if family == "PRIMARY":
        values = (
            3 * index + 5 * bit_weight + 1,
            7 * index + 2 * bit_weight + 2,
            11 * index + bit_weight + 4,
        )
    elif family == "REUSE":
        values = (
            5 * index + 2 * ternary_weight + 3,
            4 * index + 3 * ternary_weight + 6,
            9 * index + ternary_weight + 8,
        )
    elif family == "ALTERNATE":
        values = (
            7 * index + 3 * gray_weight + 2,
            8 * index + 2 * gray_weight + 5,
            6 * index + gray_weight + 1,
        )
    else:
        fail("oracle family outside declared set")
    return values[0] % P or 1, values[1] % P, values[2] % P


def phase_exponent(shell: int, index: int, family: str) -> int:
    quadratic, linear, cubic = gate_parameters(index, family)
    return (
        quadratic * shell**4 + linear * shell**2 + cubic * shell
    ) % P


def observation(depth: int, family: str) -> tuple[int, int]:
    return (
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )


def elimination_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 1):
        for row in range(P - 1, column, -1):
            yield row - 1, row


def reverse_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 2, -1, -1):
        for row in range(column + 1, P):
            yield row - 1, row


def inverse_half() -> tuple[int, ...]:
    return tuple(pow((4 * parameter) % P, -1, P) for parameter in range(1, 9))


def kernel_float(target: int, source: int) -> float:
    value = 1.0 if target == 0 else 0.0
    for offset, inverse in enumerate(inverse_half()):
        parameter = offset + 1
        exponent = (-source * parameter - target * inverse) % P
        value -= (2.0 / P) * math.cos(TAU * exponent / P)
    return value


def kernel_long(target: int, source: int) -> np.longdouble:
    value = np.longdouble(1 if target == 0 else 0)
    for offset, inverse in enumerate(inverse_half()):
        parameter = offset + 1
        exponent = (-source * parameter - target * inverse) % P
        value -= (np.longdouble(2) / P) * np.cos(LTAU * exponent / P)
    return value


def dense_weighted_long() -> np.ndarray:
    weights = np.sqrt(np.asarray(SHELL_COUNTS, dtype=np.longdouble))
    matrix = np.empty((P, P), dtype=np.longdouble)
    for target in range(P):
        for source in range(P):
            matrix[target, source] = (
                weights[target]
                * kernel_long(target, source)
                / weights[source]
            )
    return matrix


def compile_float_plan() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    weights = np.sqrt(np.asarray(SHELL_COUNTS, dtype=np.float64))
    working = np.empty((P, P), dtype=np.float64)
    for target in range(P):
        for source in range(P):
            working[target, source] = (
                weights[target] * kernel_float(target, source) / weights[source]
            )
    source = working.copy()
    coefficients = np.empty((ROTATION_COUNT, 2), dtype=np.float64)
    ordinal = 0
    for column in range(P - 1):
        for row in range(P - 1, column, -1):
            upper = row - 1
            lower = row
            radius = math.hypot(
                float(working[upper, column]),
                float(working[lower, column]),
            )
            cosine = float(working[upper, column]) / radius
            sine = float(working[lower, column]) / radius
            coefficients[ordinal] = cosine, sine
            upper_row = working[upper].copy()
            lower_row = working[lower].copy()
            working[upper] = cosine * upper_row + sine * lower_row
            working[lower] = -sine * upper_row + cosine * lower_row
            ordinal += 1
    diagonal = np.diag(working)
    signs = np.where(diagonal >= 0.0, 1.0, -1.0).astype(np.float64)
    off = working.copy()
    np.fill_diagonal(off, 0.0)
    reconstructed = np.diag(signs)
    ordinal = ROTATION_COUNT - 1
    for upper, lower in reverse_pairs():
        cosine = float(coefficients[ordinal, 0])
        sine = float(coefficients[ordinal, 1])
        upper_row = reconstructed[upper].copy()
        lower_row = reconstructed[lower].copy()
        reconstructed[upper] = cosine * upper_row - sine * lower_row
        reconstructed[lower] = sine * upper_row + cosine * lower_row
        ordinal -= 1
    metrics = {
        "orthogonality_error": float(
            np.max(np.abs(source.T @ source - np.eye(P)))
        ),
        "symmetry_error": float(np.max(np.abs(source - source.T))),
        "involution_error": float(
            np.max(np.abs(source @ source - np.eye(P)))
        ),
        "triangular_off_diagonal_error": float(np.max(np.abs(off))),
        "triangular_diagonal_magnitude_error": float(
            np.max(np.abs(np.abs(diagonal) - 1.0))
        ),
        "reconstruction_error": float(np.max(np.abs(reconstructed - source))),
    }
    return coefficients, signs, metrics


def local_coupler(
    angles: np.ndarray,
    upper: int,
    lower: int,
    cosine: float,
    sine: float,
    *,
    transpose: bool,
) -> int:
    upper_value = phasor_pair(
        float(angles[upper, 0]), float(angles[upper, 1])
    )
    lower_value = phasor_pair(
        float(angles[lower, 0]), float(angles[lower, 1])
    )
    if transpose:
        next_upper = cosine * upper_value - sine * lower_value
        next_lower = sine * upper_value + cosine * lower_value
    else:
        next_upper = cosine * upper_value + sine * lower_value
        next_lower = -sine * upper_value + cosine * lower_value
    upper_chart = chart(next_upper.real, next_upper.imag)
    lower_chart = chart(next_lower.real, next_lower.imag)
    angles[upper] = upper_chart[:2]
    angles[lower] = lower_chart[:2]
    return int(upper_chart[3]) + int(lower_chart[3])


def apply_signs(angles: np.ndarray, signs: np.ndarray) -> None:
    for shell in range(P):
        if float(signs[shell]) < 0.0:
            angles[shell, 0] = wrap_scalar(float(angles[shell, 0]) + math.pi)
            angles[shell, 1] = wrap_scalar(float(angles[shell, 1]) + math.pi)


def local_fourier(
    angles: np.ndarray,
    coefficients: np.ndarray,
    signs: np.ndarray,
    *,
    inverse: bool = False,
) -> int:
    zero_count = 0
    if inverse:
        for ordinal, (upper, lower) in enumerate(elimination_pairs()):
            zero_count += local_coupler(
                angles,
                upper,
                lower,
                float(coefficients[ordinal, 0]),
                float(coefficients[ordinal, 1]),
                transpose=False,
            )
        apply_signs(angles, signs)
    else:
        apply_signs(angles, signs)
        ordinal = ROTATION_COUNT - 1
        for upper, lower in reverse_pairs():
            zero_count += local_coupler(
                angles,
                upper,
                lower,
                float(coefficients[ordinal, 0]),
                float(coefficients[ordinal, 1]),
                transpose=True,
            )
            ordinal -= 1
    return zero_count


def phase_update(
    angles: np.ndarray, index: int, family: str, *, inverse: bool = False
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * phase_exponent(shell, index, family) / P
        angles[shell, 0] = wrap_scalar(float(angles[shell, 0]) + delta)
        angles[shell, 1] = wrap_scalar(float(angles[shell, 1]) + delta)


def project(angles: np.ndarray, depth: int, family: str) -> complex:
    quadratic, linear = observation(depth, family)
    result = 0.0j
    for shell in range(P):
        exponent = (quadratic * shell * shell + linear * shell) % P
        angle = TAU * exponent / P
        result += (
            scale(shell)
            * complex(math.cos(angle), math.sin(angle))
            * phasor_pair(float(angles[shell, 0]), float(angles[shell, 1]))
        )
    return result


def local_forward(
    angles: np.ndarray,
    depth: int,
    family: str,
    coefficients: np.ndarray,
    signs: np.ndarray,
) -> int:
    zero_count = 0
    for index in range(depth):
        phase_update(angles, index, family)
        zero_count += local_fourier(angles, coefficients, signs)
    return zero_count


def local_inverse(
    angles: np.ndarray,
    depth: int,
    family: str,
    coefficients: np.ndarray,
    signs: np.ndarray,
    *,
    reordered: bool = False,
) -> int:
    zero_count = 0
    for index in range(depth - 1, -1, -1):
        if reordered:
            phase_update(angles, index, family, inverse=True)
        zero_count += local_fourier(
            angles, coefficients, signs, inverse=True
        )
        if not reordered:
            phase_update(angles, index, family, inverse=True)
    return zero_count


def local_transaction(
    angles: np.ndarray,
    depth: int,
    family: str,
    coefficients: np.ndarray,
    signs: np.ndarray,
) -> tuple[complex, float, int]:
    zeros = local_forward(angles, depth, family, coefficients, signs)
    boundary = project(angles, depth, family)
    zeros += local_inverse(angles, depth, family, coefficients, signs)
    restoration = physical_error(angles, seed_angles())
    return boundary, restoration, zeros


def dense_case(
    matrix: np.ndarray, depth: int, family: str
) -> tuple[complex, np.ndarray]:
    weights = np.sqrt(np.asarray(SHELL_COUNTS, dtype=np.longdouble))
    state = np.asarray(weights / P, dtype=np.clongdouble)
    for index in range(depth):
        for shell in range(P):
            angle = LTAU * phase_exponent(shell, index, family) / P
            state[shell] *= np.cos(angle) + 1j * np.sin(angle)
        state = matrix @ state
    quadratic, linear = observation(depth, family)
    boundary = np.clongdouble(0)
    for shell in range(P):
        exponent = (quadratic * shell * shell + linear * shell) % P
        angle = LTAU * exponent / P
        boundary += weights[shell] * (
            np.cos(angle) + 1j * np.sin(angle)
        ) * state[shell]
    return complex(boundary), state


def run(package_path: Path) -> dict[str, Any]:
    package_bytes = package_path.read_bytes()
    package = json.loads(package_bytes)
    production_source = Path(__file__).with_name(
        "f17_anisotropic_radial_local_polar_givens_coupling.py"
    )
    if hashlib.sha256(production_source.read_bytes()).hexdigest() != package.get(
        "source_sha256"
    ):
        fail("package source hash does not match production source")
    if package.get("execution_scope", {}).get("case_count") != 21:
        fail("package case count changed")
    if package.get("classification") != "SOURCE_AUDITED_PACKAGE_LOCAL":
        fail("production package authority changed")
    coefficients, signs, plan_metrics = compile_float_plan()
    if max(plan_metrics.values()) > PLAN_TOLERANCE:
        fail("independent public plan exceeded tolerance")
    dense_matrix = dense_weighted_long()
    dense_orthogonality = float(
        np.max(
            np.abs(
                dense_matrix.T @ dense_matrix
                - np.eye(P, dtype=np.longdouble)
            )
        )
    )
    dense_symmetry = float(np.max(np.abs(dense_matrix - dense_matrix.T)))
    dense_involution = float(
        np.max(
            np.abs(
                dense_matrix @ dense_matrix
                - np.eye(P, dtype=np.longdouble)
            )
        )
    )
    case_map = {
        (case["family"], int(case["depth"])): case
        for case in package["cases"]
    }
    comparisons = 0
    maximum_production_oracle_phase_boundary_error = 0.0
    maximum_production_oracle_dense_boundary_error = 0.0
    maximum_independent_local_dense_state_error = 0.0
    maximum_independent_local_dense_boundary_error = 0.0
    maximum_independent_restoration_error = 0.0
    total_zero_canonicalizations = 0
    for family in FAMILIES:
        for depth in DEPTHS:
            case = case_map[(family, depth)]
            angles = seed_angles()
            zeros = local_forward(angles, depth, family, coefficients, signs)
            local_state = angle_state(angles)
            local_boundary = project(angles, depth, family)
            dense_boundary, dense_state = dense_case(dense_matrix, depth, family)
            restoration_zeros = local_inverse(
                angles, depth, family, coefficients, signs
            )
            restoration = physical_error(angles, seed_angles())
            total_zero_canonicalizations += zeros + restoration_zeros
            production_boundary = complex_from_pair(case["final_boundary"])
            production_matched = complex_from_pair(
                case["matched_complex_givens_boundary"]
            )
            phase_error = abs(production_boundary - local_boundary)
            dense_error = abs(production_matched - dense_boundary)
            local_dense_state = float(
                np.max(np.abs(local_state.astype(np.clongdouble) - dense_state))
            )
            local_dense_boundary = abs(local_boundary - dense_boundary)
            maximum_production_oracle_phase_boundary_error = max(
                maximum_production_oracle_phase_boundary_error, phase_error
            )
            maximum_production_oracle_dense_boundary_error = max(
                maximum_production_oracle_dense_boundary_error, dense_error
            )
            maximum_independent_local_dense_state_error = max(
                maximum_independent_local_dense_state_error, local_dense_state
            )
            maximum_independent_local_dense_boundary_error = max(
                maximum_independent_local_dense_boundary_error,
                local_dense_boundary,
            )
            maximum_independent_restoration_error = max(
                maximum_independent_restoration_error, restoration
            )
            checks = [
                phase_error <= BOUNDARY_TOLERANCE,
                dense_error <= BOUNDARY_TOLERANCE,
                local_dense_state <= STATE_TOLERANCE,
                local_dense_boundary <= BOUNDARY_TOLERANCE,
                restoration <= RESTORATION_TOLERANCE,
                case["same_backing"] is True,
                case["restoration_generation_before"] == 0,
                case["restoration_generation_after"] == 1,
                case["snapshot_reload_used"] is False,
                case["inverse_history_cells"] == 0,
                case["resident_restoration_class"]
                == "NUMERICAL_PHYSICAL_STATE_RESTORATION",
                case["transient_buffer_restoration_class"]
                == "NO_RESTORATION_CLAIM",
            ]
            comparisons += len(checks)
            if not all(checks):
                fail(f"independent case verification failed for {family}/{depth}")
    restored = seed_angles()
    local_transaction(restored, 37, "PRIMARY", coefficients, signs)
    reuse_boundary, reuse_restoration, reuse_zeros = local_transaction(
        restored, REUSE_DEPTH, "REUSE", coefficients, signs
    )
    fresh_boundary, _, fresh_zeros = local_transaction(
        seed_angles(), REUSE_DEPTH, "REUSE", coefficients, signs
    )
    reuse_error = abs(reuse_boundary - fresh_boundary)
    repeated = seed_angles()
    repeated_maximum = 0.0
    repeated_zeros = 0
    for _ in range(REPEATED_CYCLES):
        _, restoration, zeros = local_transaction(
            repeated, REPEATED_DEPTH, "ALTERNATE", coefficients, signs
        )
        repeated_maximum = max(repeated_maximum, restoration)
        repeated_zeros += zeros
    total_zero_canonicalizations += reuse_zeros + fresh_zeros + repeated_zeros
    reuse_record = package["reuse"]
    repeated_record = package["repeated_reuse"]
    reuse_checks = [
        reuse_error <= BOUNDARY_TOLERANCE,
        reuse_restoration <= RESTORATION_TOLERANCE,
        reuse_record["same_original_backing"] is True,
        reuse_record["restoration_generation"] == 2,
        reuse_record["snapshot_reload_used"] is False,
        reuse_record["inverse_history_cells"] == 0,
        abs(reuse_record["fresh_restored_boundary_error"] - reuse_error)
        <= BOUNDARY_TOLERANCE,
        repeated_maximum <= RESTORATION_TOLERANCE,
        repeated_record["same_backing"] is True,
        repeated_record["restoration_generation"] == 100,
        repeated_record["snapshot_reload_used"] is False,
        repeated_record["inverse_history_cells"] == 0,
        abs(
            repeated_record["maximum_restoration_error"] - repeated_maximum
        )
        <= RESTORATION_TOLERANCE,
    ]
    comparisons += len(reuse_checks)
    if not all(reuse_checks):
        fail("independent reuse verification failed")
    resources = package["resource_law"]
    matched = package["matched_classical_baseline"]
    resource_checks = [
        package["plan"]["rotation_count"] == 136,
        package["plan"]["retained_plan_bytes"] == 2312,
        package["plan"]["stored_index_cells"] == 0,
        package["plan"]["compilation_maximum_named_bytes"] == 4960,
        resources["resident_phase_angle_bytes"] == 272,
        resources["retained_public_givens_plan_bytes"] == 2312,
        resources["maximum_named_update_bytes"] == 96,
        resources["maximum_named_restoration_verification_bytes"] == 64,
        resources["maximum_named_warm_execution_live_bytes_including_program_json"]
        == 3055,
        resources["maximum_named_full_lifecycle_live_bytes"] == 4960,
        resources["retained_complex_state_cells"] == 0,
        resources["retained_dense_kernel_cells"] == 0,
        resources["inverse_history_cells"] == 0,
        matched["method"]
        == "IDENTICAL_PUBLIC_PLAN17_COMPLEX_IN_PLACE_GIVENS_RECURRENCE",
        matched["maximum_named_warm_execution_live_bytes_including_program_json"]
        == 2959,
        matched["maximum_named_full_lifecycle_live_bytes"] == 4960,
        matched["comparison_establishes_distinct_phase_resource"] is False,
        matched["comparison_establishes_computational_advantage"] is False,
        package["execution_scope"]["public_topology_compilation_reads_final_answers"]
        is False,
    ]
    comparisons += len(resource_checks)
    if not all(resource_checks):
        fail("independent resource-law verification failed")

    control_depth = 3
    control_family = "ALTERNATE"
    valid_angles = seed_angles()
    local_forward(
        valid_angles, control_depth, control_family, coefficients, signs
    )
    valid_boundary = project(valid_angles, control_depth, control_family)
    mutated_coefficients = coefficients.copy()
    mutated_coefficients[0, 0] += 0.03125
    mutated_angles = seed_angles()
    local_forward(
        mutated_angles,
        control_depth,
        control_family,
        mutated_coefficients,
        signs,
    )
    mutated_boundary = project(mutated_angles, control_depth, control_family)
    wrong_inverse = valid_angles.copy()
    local_inverse(
        wrong_inverse,
        control_depth,
        control_family,
        mutated_coefficients,
        signs,
    )
    reordered = valid_angles.copy()
    local_inverse(
        reordered,
        control_depth,
        control_family,
        coefficients,
        signs,
        reordered=True,
    )
    missing_inverse_error = physical_error(valid_angles, seed_angles())
    unweighted = np.empty((P, P), dtype=np.longdouble)
    for target in range(P):
        for source in range(P):
            unweighted[target, source] = kernel_long(target, source)
    unweighted_boundary, _ = dense_case(unweighted, control_depth, control_family)
    dense_valid_boundary, _ = dense_case(
        dense_matrix, control_depth, control_family
    )
    mutation_controls = {
        "plan_coefficient_mutation_detected": (
            abs(mutated_boundary - valid_boundary) > CONTROL_FLOOR
        ),
        "missing_inverse_detected": missing_inverse_error > CONTROL_FLOOR,
        "wrong_inverse_detected": (
            physical_error(wrong_inverse, seed_angles()) > CONTROL_FLOOR
        ),
        "reordered_inverse_detected": (
            physical_error(reordered, seed_angles()) > CONTROL_FLOOR
        ),
        "shell_weight_normalization_mutation_detected": (
            abs(unweighted_boundary - dense_valid_boundary) > CONTROL_FLOOR
        ),
    }
    comparisons += len(mutation_controls)
    if not all(mutation_controls.values()):
        fail("one or more independent mutation controls failed")
    return {
        "schema": "CAT_CAS_F17_LOCAL_POLAR_GIVENS_ORACLE_V1",
        "package_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "oracle_method": (
            "SEPARATE_LONG_DOUBLE_DENSE_WEIGHTED_RADIAL_OPERATOR_AND_"
            "SEPARATE_FLOAT64_QR_LOCAL_PHASE_PAIR_EXECUTION"
        ),
        "case_count": 21,
        "metric_field_and_control_comparisons": comparisons,
        "all_cases_within_predeclared_tolerances": True,
        "dense_weighted_orthogonality_error": dense_orthogonality,
        "dense_weighted_symmetry_error": dense_symmetry,
        "dense_weighted_involution_error": dense_involution,
        "independent_plan_metrics": plan_metrics,
        "maximum_production_oracle_phase_boundary_error": (
            maximum_production_oracle_phase_boundary_error
        ),
        "maximum_production_oracle_dense_boundary_error": (
            maximum_production_oracle_dense_boundary_error
        ),
        "maximum_independent_local_dense_state_error": (
            maximum_independent_local_dense_state_error
        ),
        "maximum_independent_local_dense_boundary_error": (
            maximum_independent_local_dense_boundary_error
        ),
        "maximum_independent_restoration_error": (
            maximum_independent_restoration_error
        ),
        "independent_unrelated_reuse_boundary_error": reuse_error,
        "independent_unrelated_reuse_restoration_error": reuse_restoration,
        "independent_100_cycle_maximum_restoration_error": repeated_maximum,
        "total_independent_zero_canonicalizations": (
            total_zero_canonicalizations
        ),
        "production_restoration_reuse_and_resource_fields_verified": True,
        "mutation_controls": mutation_controls,
        "claim_ceiling": (
            "21_DECLARED_WEIGHTED_LOCAL_POLAR_GIVENS_CASES_THROUGH_"
            "DEPTH4096_PLUS_DECLARED_REUSE_CONTROLS_IN_LINUX_DIRECT_"
            "PROCESS_SOFTWARE"
        ),
        "preserved_subclaims": [
            "PUBLIC_WEIGHTED_REAL_ORTHOGONAL_RADIAL_OPERATOR",
            "DETERMINISTIC136_LOCAL_GIVENS_FACTORIZATION",
            "DIRECT_TWO_PHASE_CELL_COUPLING_WITHOUT_ACCEPTED_COMPLEX_STATE",
            "FINAL_ONLY_BOUNDARY_PARITY",
            "SAME_BACKING_NUMERICAL_RESTORATION_AND_REUSE",
        ],
        "rejected_interpretations": [
            "CARTESIAN_REGISTER_FREE_LOCAL_COUPLING",
            "EXACT_ALGEBRAIC_RESTORATION",
            "UNBOUNDED_DEPTH_NUMERICAL_STABILITY",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = run(Path(arguments.package))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
