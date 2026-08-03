#!/usr/bin/env python3
"""Independent dense long-double oracle for the phase-pair chart result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


P = 17
NONSQUARE = 3
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = np.asarray((1, *([18] * 16)), dtype=np.int64)
REAL = np.longdouble
COMPLEX = np.clongdouble
PI = np.arccos(REAL(-1))
TAU = REAL(2) * PI
CHART_ZERO_FLOOR = REAL("1e-7")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


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
        raise AssertionError("oracle family outside declared set")
    return values[0] % P or 1, values[1] % P, values[2] % P


def descriptor(depth: int, family: str) -> dict[str, Any]:
    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_STREAMED_PROGRAM_V1",
        "depth": depth,
        "family": family,
        "gate_generator": "PUBLIC_INDEX_BIT_GRAY_TERNARY_WEIGHT_FORMULA",
        "observation": [
            (3 * depth + 2 * len(family) + 1) % P or 1,
            (5 * depth + len(family) + 4) % P,
        ],
    }


def norm(x: int, y: int) -> int:
    return (x * x - NONSQUARE * y * y) % P


def pairing(u: int, v: int, x: int, y: int) -> int:
    return (u * x - NONSQUARE * v * y) % P


def roots() -> np.ndarray:
    return np.exp(
        COMPLEX(1j) * TAU * np.arange(P, dtype=REAL) / REAL(P)
    ).astype(COMPLEX)


def compile_dense_fourier(root_table: np.ndarray) -> tuple[np.ndarray, int]:
    representatives: list[tuple[int, int] | None] = [None] * P
    counts = [0] * P
    coordinates = [(x, y) for x in range(P) for y in range(P)]
    for x, y in coordinates:
        shell = norm(x, y)
        counts[shell] += 1
        if representatives[shell] is None:
            representatives[shell] = (x, y)
    if counts != [1, *([18] * 16)] or any(
        item is None for item in representatives
    ):
        raise AssertionError("oracle anisotropic shell law changed")
    matrix = np.zeros((P, P), dtype=COMPLEX)
    visits = 0
    for target, representative in enumerate(representatives):
        assert representative is not None
        u, v = representative
        for x, y in coordinates:
            source = norm(x, y)
            matrix[target, source] += root_table[
                pairing(u, v, x, y)
            ] / REAL(P)
            visits += 1
    return matrix, visits


def wrap(values: np.ndarray) -> np.ndarray:
    return np.remainder(values + PI, TAU) - PI


def encode(values: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(values).astype(REAL)
    if (
        not np.all(np.isfinite(values))
        or np.min(magnitudes) <= CHART_ZERO_FLOOR
        or np.max(magnitudes) > REAL(1) + REAL("1e-16")
    ):
        raise AssertionError("oracle phase-pair chart left unit disk")
    phases = np.arctan2(values.imag, values.real).astype(REAL)
    deltas = np.arccos(np.clip(magnitudes, REAL(0), REAL(1)))
    return wrap(np.stack((phases + deltas, phases - deltas), axis=1))


def decode(angles: np.ndarray) -> np.ndarray:
    return (
        np.exp(COMPLEX(1j) * angles[:, 0])
        + np.exp(COMPLEX(1j) * angles[:, 1])
    ) / REAL(2)


def phase_exponents(index: int, family: str) -> np.ndarray:
    quadratic, linear, cubic = gate_parameters(index, family)
    shells = np.arange(P, dtype=np.int64)
    return (
        quadratic * shells**4 + linear * shells**2 + cubic * shells
    ) % P


def phase_angles(
    angles: np.ndarray, index: int, family: str, inverse: bool = False
) -> np.ndarray:
    sign = REAL(-1) if inverse else REAL(1)
    delta = (
        sign
        * TAU
        * phase_exponents(index, family).astype(REAL)
        / REAL(P)
    )
    return wrap(angles + delta[:, None])


def phase_vector(
    state: np.ndarray,
    index: int,
    family: str,
    root_table: np.ndarray,
) -> np.ndarray:
    return state * root_table[phase_exponents(index, family)]


def boundary(
    state: np.ndarray,
    depth: int,
    family: str,
    root_table: np.ndarray,
) -> COMPLEX:
    public = descriptor(depth, family)["observation"]
    shells = np.arange(P, dtype=np.int64)
    exponents = (public[0] * shells * shells + public[1] * shells) % P
    return np.sum(
        SHELL_COUNTS.astype(REAL) * root_table[exponents] * state,
        dtype=COMPLEX,
    )


def phase_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.exp(COMPLEX(1j) * left)
                - np.exp(COMPLEX(1j) * right)
            )
        )
    )


def execute_case(
    matrix: np.ndarray,
    root_table: np.ndarray,
    depth: int,
    family: str,
) -> dict[str, Any]:
    seed = np.full(P, REAL(1) / REAL(P), dtype=COMPLEX)
    seed_chart = encode(seed)
    chart = np.array(seed_chart, copy=True)
    classical = np.array(seed, copy=True)
    for index in range(depth):
        chart = phase_angles(chart, index, family)
        chart = encode(matrix @ decode(chart))
        classical = matrix @ phase_vector(
            classical, index, family, root_table
        )
    chart_boundary = boundary(decode(chart), depth, family, root_table)
    classical_boundary = boundary(classical, depth, family, root_table)
    state_error = float(np.max(np.abs(decode(chart) - classical)))
    for index in range(depth - 1, -1, -1):
        chart = encode(matrix @ decode(chart))
        chart = phase_angles(chart, index, family, inverse=True)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": digest_json(descriptor(depth, family)),
        "chart_boundary": [
            float(chart_boundary.real),
            float(chart_boundary.imag),
        ],
        "classical_boundary": [
            float(classical_boundary.real),
            float(classical_boundary.imag),
        ],
        "chart_classical_state_error": state_error,
        "chart_classical_boundary_error": float(
            abs(chart_boundary - classical_boundary)
        ),
        "restoration_error": phase_distance(chart, seed_chart),
    }


def sequential_reuse(
    matrix: np.ndarray, root_table: np.ndarray
) -> tuple[float, float]:
    seed = np.full(P, REAL(1) / REAL(P), dtype=COMPLEX)
    seed_chart = encode(seed)

    def transaction(
        chart: np.ndarray, depth: int, family: str
    ) -> tuple[np.ndarray, COMPLEX, float]:
        for index in range(depth):
            chart = phase_angles(chart, index, family)
            chart = encode(matrix @ decode(chart))
        result = boundary(decode(chart), depth, family, root_table)
        for index in range(depth - 1, -1, -1):
            chart = encode(matrix @ decode(chart))
            chart = phase_angles(chart, index, family, inverse=True)
        return chart, result, phase_distance(chart, seed_chart)

    restored, _, _ = transaction(np.array(seed_chart, copy=True), 64, "PRIMARY")
    restored, reused_boundary, restoration_error = transaction(
        restored, 1537, "REUSE"
    )
    _, fresh_boundary, _ = transaction(
        np.array(seed_chart, copy=True), 1537, "REUSE"
    )
    return float(abs(reused_boundary - fresh_boundary)), restoration_error


def repeated_reuse(
    matrix: np.ndarray, root_table: np.ndarray
) -> float:
    seed = np.full(P, REAL(1) / REAL(P), dtype=COMPLEX)
    seed_chart = encode(seed)
    chart = np.array(seed_chart, copy=True)
    maximum = 0.0
    for _ in range(100):
        for index in range(64):
            chart = phase_angles(chart, index, "PRIMARY")
            chart = encode(matrix @ decode(chart))
        for index in range(63, -1, -1):
            chart = encode(matrix @ decode(chart))
            chart = phase_angles(chart, index, "PRIMARY", inverse=True)
        maximum = max(maximum, phase_distance(chart, seed_chart))
    return maximum


def pair(value: list[float]) -> complex:
    return complex(value[0], value[1])


def run(package_path: Path) -> dict[str, Any]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    root_table = roots()
    matrix, coordinate_visits = compile_dense_fourier(root_table)
    involution_error = float(
        np.max(np.abs(matrix @ matrix - np.eye(P, dtype=COMPLEX)))
    )
    if involution_error > 1.0e-16:
        raise AssertionError("oracle dense radial Fourier is not involutive")
    oracle_cases = [
        execute_case(matrix, root_table, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    if len(package["cases"]) != len(oracle_cases):
        raise AssertionError("package/oracle case count changed")
    comparisons = 0
    tolerance = package["predeclared_tolerances"]["boundary_max_abs"]
    restoration_tolerance = package["predeclared_tolerances"][
        "restoration_phasor_max_abs"
    ]
    maximum_production_oracle_boundary_error = 0.0
    maximum_production_oracle_classical_error = 0.0
    for candidate, oracle in zip(
        package["cases"], oracle_cases, strict=True
    ):
        if candidate["depth"] != oracle["depth"] or candidate["family"] != oracle["family"]:
            raise AssertionError("package/oracle case ordering changed")
        comparisons += 2
        if candidate["program_fingerprint"] != oracle["program_fingerprint"]:
            raise AssertionError("public program fingerprint changed")
        comparisons += 1
        production_boundary_error = abs(
            pair(candidate["final_boundary"]) - pair(oracle["chart_boundary"])
        )
        classical_error = abs(
            pair(candidate["matched_classical_boundary"])
            - pair(oracle["classical_boundary"])
        )
        maximum_production_oracle_boundary_error = max(
            maximum_production_oracle_boundary_error,
            production_boundary_error,
        )
        maximum_production_oracle_classical_error = max(
            maximum_production_oracle_classical_error, classical_error
        )
        comparisons += 2
        if production_boundary_error > tolerance or classical_error > tolerance:
            raise AssertionError("production/oracle boundary tolerance failed")
        if (
            oracle["chart_classical_boundary_error"] > tolerance
            or oracle["restoration_error"] > restoration_tolerance
        ):
            raise AssertionError("oracle chart control failed")
        comparisons += 2

    reuse_boundary_error, reuse_restoration_error = sequential_reuse(
        matrix, root_table
    )
    repeated_maximum = repeated_reuse(matrix, root_table)
    if (
        reuse_boundary_error > tolerance
        or reuse_restoration_error > restoration_tolerance
        or repeated_maximum > restoration_tolerance
    ):
        raise AssertionError("oracle reuse tolerance failed")

    mutated = json.loads(json.dumps(package))
    mutated["cases"][-1]["final_boundary"][0] += 1.0e-6
    mutation_detected = abs(
        pair(mutated["cases"][-1]["final_boundary"])
        - pair(oracle_cases[-1]["chart_boundary"])
    ) > tolerance
    if not mutation_detected:
        raise AssertionError("oracle failed boundary mutation")

    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_UNIT_PHASOR_PAIR_CHART_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "oracle_method": "INDEPENDENT_COMPLEX_LONG_DOUBLE_DENSE289_COORDINATE_ANISOTROPIC_FOURIER_AND_SEPARATE_PHASE_PAIR_CHART",
        "case_count": len(oracle_cases),
        "metric_and_boundary_comparisons": comparisons,
        "dense_public_coordinate_visits": coordinate_visits,
        "dense_fourier_involution_max_abs": involution_error,
        "maximum_production_oracle_phase_boundary_error": maximum_production_oracle_boundary_error,
        "maximum_production_oracle_classical_boundary_error": maximum_production_oracle_classical_error,
        "oracle_unrelated_reuse_boundary_error": reuse_boundary_error,
        "oracle_unrelated_reuse_restoration_error": reuse_restoration_error,
        "oracle_100_cycle_maximum_restoration_error": repeated_maximum,
        "all_cases_within_predeclared_tolerances": True,
        "boundary_mutation_detected": mutation_detected,
        "claim_ceiling": "21_DECLARED_NUMERICAL_PHASE_PAIR_CASES_THROUGH_DEPTH4096_AND_DECLARED_REUSE_CONTROLS_ON_LINUX_DIRECT_PROCESS_SOFTWARE",
        "preserved_subclaims": [
            "INDEPENDENT_DENSE_COORDINATE_FOURIER_PARITY",
            "INDEPENDENT_LONG_DOUBLE_PHASE_PAIR_CHART_AND_CLASSICAL_BOUNDARY_PARITY",
            "INDEPENDENT_LONG_DOUBLE_INVERSE_AND_REUSE_TOLERANCE_CHECKS",
        ],
        "rejected_interpretations": [
            "EXACT_ALGEBRAIC_RESTORATION",
            "UNBOUNDED_DEPTH_NUMERICAL_STABILITY",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
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
