#!/usr/bin/env python3
"""Independent dense long-double oracle for native angle interference.

This file imports neither the production mechanism nor any predecessor.  It
reconstructs the public real radial kernel from paired F17 characters, checks
the exact exponent pairing and involution, and separately executes the
two-angle chart in long-double Cartesian verifier space.  The unweighted
radial coefficient matrix is not claimed symmetric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


P = 17
TAU = np.longdouble("6.2831853071795864769252867665590057683943387987502")
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = np.asarray((1, *([18] * 16)), dtype=np.longdouble)
BOUNDARY_TOLERANCE = np.longdouble("5e-11")
RESTORATION_TOLERANCE = np.longdouble("1e-11")
MUTATION_FLOOR = np.longdouble("1e-5")
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100


def fail(message: str) -> None:
    raise RuntimeError(message)


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


def phase_exponents(index: int, family: str) -> np.ndarray:
    quadratic, linear, cubic = gate_parameters(index, family)
    shells = np.arange(P, dtype=np.int64)
    return (
        quadratic * shells**4 + linear * shells**2 + cubic * shells
    ) % P


def observation(depth: int, family: str) -> tuple[int, int]:
    return (
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )


def wrap(values: np.ndarray) -> np.ndarray:
    return np.remainder(values + np.longdouble(np.pi), TAU) - np.longdouble(
        np.pi
    )


def seed_angles() -> np.ndarray:
    delta = np.arccos(np.longdouble(1) / np.longdouble(P))
    result = np.empty((P, 2), dtype=np.longdouble)
    result[:, 0] = delta
    result[:, 1] = -delta
    return result


def phasor_distance(left: np.ndarray, right: np.ndarray) -> np.longdouble:
    real = np.cos(left) - np.cos(right)
    imag = np.sin(left) - np.sin(right)
    return np.max(np.hypot(real, imag))


def compile_dense_kernel() -> tuple[np.ndarray, int, int]:
    inverse = [pow((4 * parameter) % P, -1, P) for parameter in range(1, 9)]
    kernel = np.empty((P, P), dtype=np.longdouble)
    visits = 0
    pairing_violations = 0
    for target in range(P):
        for source in range(P):
            value = np.longdouble(1 if target == 0 else 0)
            for parameter, reciprocal in zip(range(1, 9), inverse):
                exponent = (-source * parameter - target * reciprocal) % P
                negative_parameter = P - parameter
                negative_reciprocal = P - reciprocal
                negative_exponent = (
                    -source * negative_parameter
                    - target * negative_reciprocal
                ) % P
                if negative_exponent != (-exponent) % P:
                    pairing_violations += 1
                value -= (
                    np.longdouble(2)
                    / np.longdouble(P)
                    * np.cos(TAU * np.longdouble(exponent) / np.longdouble(P))
                )
                visits += 1
            kernel[target, source] = value
    return kernel, visits, pairing_violations


def decode(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.longdouble("0.5") * (
        np.cos(angles[:, 0]) + np.cos(angles[:, 1])
    )
    imag = np.longdouble("0.5") * (
        np.sin(angles[:, 0]) + np.sin(angles[:, 1])
    )
    return real, imag


def encode(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    magnitude = np.hypot(real, imag)
    if np.min(magnitude) <= np.longdouble("1e-7"):
        fail("oracle chart reached zero")
    if np.max(magnitude) > np.longdouble(1) + np.longdouble("2e-12"):
        fail("oracle chart exceeded unit disk")
    phase = np.arctan2(imag, real)
    delta = np.arccos(np.clip(magnitude, 0, 1))
    return wrap(np.stack((phase + delta, phase - delta), axis=1))


def apply_phase(
    angles: np.ndarray, index: int, family: str, inverse: bool = False
) -> None:
    sign = np.longdouble(-1 if inverse else 1)
    delta = (
        sign
        * TAU
        * phase_exponents(index, family).astype(np.longdouble)
        / np.longdouble(P)
    )
    angles[:] = wrap(angles + delta[:, None])


def apply_fourier(angles: np.ndarray, kernel: np.ndarray) -> None:
    real, imag = decode(angles)
    angles[:] = encode(kernel @ real, kernel @ imag)


def project(angles: np.ndarray, depth: int, family: str) -> complex:
    real, imag = decode(angles)
    quadratic, linear = observation(depth, family)
    shells = np.arange(P, dtype=np.int64)
    exponents = (quadratic * shells * shells + linear * shells) % P
    phases = TAU * exponents.astype(np.longdouble) / np.longdouble(P)
    cosine = np.cos(phases)
    sine = np.sin(phases)
    boundary_real = np.sum(SHELL_COUNTS * (cosine * real - sine * imag))
    boundary_imag = np.sum(SHELL_COUNTS * (sine * real + cosine * imag))
    return complex(float(boundary_real), float(boundary_imag))


def transaction(
    angles: np.ndarray, depth: int, family: str, kernel: np.ndarray
) -> tuple[complex, np.longdouble]:
    for index in range(depth):
        apply_phase(angles, index, family)
        apply_fourier(angles, kernel)
    boundary = project(angles, depth, family)
    for index in range(depth - 1, -1, -1):
        apply_fourier(angles, kernel)
        apply_phase(angles, index, family, inverse=True)
    return boundary, phasor_distance(angles, seed_angles())


def execute_case(depth: int, family: str, kernel: np.ndarray) -> dict[str, Any]:
    angles = seed_angles()
    backing = int(angles.__array_interface__["data"][0])
    for index in range(depth):
        apply_phase(angles, index, family)
        apply_fourier(angles, kernel)
    boundary = project(angles, depth, family)
    for index in range(depth - 1, -1, -1):
        apply_fourier(angles, kernel)
        apply_phase(angles, index, family, inverse=True)
    return {
        "depth": depth,
        "family": family,
        "boundary": [boundary.real, boundary.imag],
        "restoration_error": float(phasor_distance(angles, seed_angles())),
        "same_backing": int(angles.__array_interface__["data"][0]) == backing,
    }


def reuse_control(kernel: np.ndarray) -> dict[str, Any]:
    angles = seed_angles()
    backing = int(angles.__array_interface__["data"][0])
    transaction(angles, 64, "PRIMARY", kernel)
    restored_boundary, restoration_error = transaction(
        angles, REUSE_DEPTH, "REUSE", kernel
    )
    fresh = seed_angles()
    fresh_boundary, _ = transaction(fresh, REUSE_DEPTH, "REUSE", kernel)
    return {
        "boundary_error": abs(restored_boundary - fresh_boundary),
        "restoration_error": float(restoration_error),
        "same_backing": int(angles.__array_interface__["data"][0]) == backing,
    }


def repeated_reuse_control(kernel: np.ndarray) -> dict[str, Any]:
    angles = seed_angles()
    backing = int(angles.__array_interface__["data"][0])
    maximum_error = np.longdouble(0)
    for _ in range(REPEATED_REUSE_CYCLES):
        _, error = transaction(
            angles, REPEATED_REUSE_DEPTH, "PRIMARY", kernel
        )
        maximum_error = max(maximum_error, error)
    return {
        "cycles": REPEATED_REUSE_CYCLES,
        "maximum_restoration_error": float(maximum_error),
        "same_backing": int(angles.__array_interface__["data"][0]) == backing,
    }


def run(package_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    kernel, coordinate_visits, pairing_violations = compile_dense_kernel()
    identity = np.eye(P, dtype=np.longdouble)
    symmetry_error = float(np.max(np.abs(kernel - kernel.T)))
    involution_error = float(np.max(np.abs(kernel @ kernel - identity)))
    cases = [
        execute_case(depth, family, kernel)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    by_key = {(item["family"], item["depth"]): item for item in cases}
    maximum_phase_error = 0.0
    maximum_classical_error = 0.0
    comparisons = 0
    for production in package["cases"]:
        independent = by_key[(production["family"], production["depth"])]
        if independent["family"] != production["family"]:
            fail("oracle family mismatch")
        if independent["depth"] != production["depth"]:
            fail("oracle depth mismatch")
        phase_boundary = complex(*production["final_boundary"])
        classical_boundary = complex(*production["matched_classical_boundary"])
        streamed_boundary = complex(
            *production["matched_streamed_classical_boundary"]
        )
        oracle_boundary = complex(*independent["boundary"])
        maximum_phase_error = max(
            maximum_phase_error, abs(phase_boundary - oracle_boundary)
        )
        maximum_classical_error = max(
            maximum_classical_error,
            abs(classical_boundary - oracle_boundary),
            abs(streamed_boundary - oracle_boundary),
        )
        if production["resident_phase_angle_cells"] != 34:
            fail("oracle resident cell mismatch")
        if production["resident_phase_angle_bytes"] != 272:
            fail("oracle resident byte mismatch")
        if production["resident_restoration_class"] != (
            "NUMERICAL_PHYSICAL_STATE_RESTORATION"
        ):
            fail("oracle restoration class mismatch")
        if not production["same_backing"] or not independent["same_backing"]:
            fail("oracle backing mismatch")
        if production["restoration_error"] > float(RESTORATION_TOLERANCE):
            fail("production restoration exceeds declared tolerance")
        if independent["restoration_error"] > float(RESTORATION_TOLERANCE):
            fail("oracle restoration exceeds declared tolerance")
        if production["restoration_generation_before"] != 0:
            fail("production restoration generation did not start at zero")
        if production["restoration_generation_after"] != 1:
            fail("production restoration generation did not advance once")
        if production["snapshot_reload_used"]:
            fail("production case used snapshot reload")
        if production["inverse_history_cells"] != 0:
            fail("production case retained inverse history")
        if production["transient_buffer_restoration_class"] != (
            "NO_RESTORATION_CLAIM"
        ):
            fail("production transient restoration class mismatch")
        if production["matched_classical_frontier_boundary_error"] > float(
            BOUNDARY_TOLERANCE
        ):
            fail("production classical frontier boundary mismatch")
        if production["matched_classical_frontier_state_error"] > float(
            BOUNDARY_TOLERANCE
        ):
            fail("production classical frontier state mismatch")
        comparisons += 22
    package_reuse = package["reuse"]
    if not (
        package_reuse["same_original_backing"]
        and package_reuse["fresh_restored_boundary_error"]
        <= float(BOUNDARY_TOLERANCE)
        and package_reuse["restoration_error"]
        <= float(RESTORATION_TOLERANCE)
        and package_reuse["restoration_generation"] == 2
        and not package_reuse["snapshot_reload_used"]
        and package_reuse["inverse_history_cells"] == 0
    ):
        fail("production unrelated reuse fields failed")
    comparisons += 6
    package_repeated = package["repeated_reuse"]
    if not (
        package_repeated["cycles"] == 100
        and package_repeated["same_backing"]
        and package_repeated["maximum_restoration_error"]
        <= float(RESTORATION_TOLERANCE)
        and package_repeated["restoration_generation"] == 100
        and not package_repeated["snapshot_reload_used"]
        and package_repeated["inverse_history_cells"] == 0
    ):
        fail("production repeated reuse fields failed")
    comparisons += 6
    reuse = reuse_control(kernel)
    repeated = repeated_reuse_control(kernel)
    mutated = np.array(kernel, copy=True)
    mutated[0, 0] += np.longdouble("1e-3")
    reference_angles = seed_angles()
    mutated_angles = seed_angles()
    reference_boundary, _ = transaction(reference_angles, 4, "PRIMARY", kernel)
    mutated_boundary, _ = transaction(mutated_angles, 4, "PRIMARY", mutated)
    mutation_detected = abs(reference_boundary - mutated_boundary) > MUTATION_FLOOR
    all_cases_within = all(
        item["restoration_error"] <= float(RESTORATION_TOLERANCE)
        and item["same_backing"]
        for item in cases
    )
    if not (
        pairing_violations == 0
        and involution_error < 1.0e-15
        and maximum_phase_error <= float(BOUNDARY_TOLERANCE)
        and maximum_classical_error <= float(BOUNDARY_TOLERANCE)
        and all_cases_within
        and reuse["same_backing"]
        and reuse["boundary_error"] <= float(BOUNDARY_TOLERANCE)
        and reuse["restoration_error"] <= float(RESTORATION_TOLERANCE)
        and repeated["same_backing"]
        and repeated["maximum_restoration_error"]
        <= float(RESTORATION_TOLERANCE)
        and mutation_detected
    ):
        fail(
            "native-angle independent oracle failed: "
            f"pairing_violations={pairing_violations} "
            f"asymmetry={symmetry_error} involution={involution_error} "
            f"phase={maximum_phase_error} classical={maximum_classical_error} "
            f"cases={all_cases_within} reuse={reuse} repeated={repeated} "
            f"mutation={mutation_detected}"
        )
    return {
        "schema": "CAT_CAS_F17_NATIVE_ANGLE_INTERFERENCE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "oracle_method": "INDEPENDENT_LONG_DOUBLE_DENSE_REAL_KERNEL_AND_SEPARATE_TWO_ANGLE_CHART",
        "case_count": len(cases),
        "metric_and_boundary_comparisons": comparisons,
        "production_restoration_and_reuse_fields_verified": True,
        "dense_public_coordinate_visits": coordinate_visits,
        "public_character_pairing_violations": pairing_violations,
        "dense_kernel_asymmetry_max_abs": symmetry_error,
        "dense_kernel_involution_max_abs": involution_error,
        "maximum_production_oracle_phase_boundary_error": maximum_phase_error,
        "maximum_production_oracle_classical_boundary_error": maximum_classical_error,
        "oracle_unrelated_reuse_boundary_error": reuse["boundary_error"],
        "oracle_unrelated_reuse_restoration_error": reuse["restoration_error"],
        "oracle_100_cycle_maximum_restoration_error": repeated[
            "maximum_restoration_error"
        ],
        "all_cases_within_predeclared_tolerances": all_cases_within,
        "kernel_mutation_detected": bool(mutation_detected),
        "claim_ceiling": "21_DECLARED_NATIVE_ANGLE_CASES_THROUGH_DEPTH4096_PLUS_DECLARED_REUSE_CONTROLS_IN_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "INDEPENDENT_REAL_KERNEL_PAIRING_AND_INVOLUTION",
            "INDEPENDENT_LONG_DOUBLE_PHASE_CHART_BOUNDARY_PARITY",
            "INDEPENDENT_LONG_DOUBLE_RESTORATION_AND_REUSE_TOLERANCE_CHECKS",
            "PRODUCTION_RESTORATION_GENERATION_SNAPSHOT_HISTORY_AND_REUSE_FIELD_CHECKS",
        ],
        "rejected_interpretations": [
            "CARTESIAN_ACCUMULATOR_FREE_INTERFERENCE",
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
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.package)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
