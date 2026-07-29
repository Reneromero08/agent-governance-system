#!/usr/bin/env python3
"""Separate exact and dense oracle for the F17 Gaussian phase quotient.

The oracle consumes public module descriptors from a production result and
reexecutes the coefficient recurrence with Python integer lists at every
tested port. It additionally executes explicit complex phase vectors at ports
two and four. It does not import the compact backend or call its compiler or
projection.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PRIME = 17
INV2 = pow(2, -1, PRIME)
TESTED_PORTS = (2, 4, 8, 16, 32)
DENSE_PORTS = (2, 4)
TOLERANCE = 2.0e-11
ROOTS = np.exp(
    2.0j * math.pi * np.arange(PRIME, dtype=np.float64) / PRIME
)
FOURIER = (
    ROOTS[
        (
            np.arange(PRIME)[:, None]
            * np.arange(PRIME)[None, :]
        )
        % PRIME
    ]
    / math.sqrt(PRIME)
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def encoded_program(program: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        program,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reduced(value: int) -> int:
    return int(value) % PRIME


def field_inverse(value: int) -> int:
    value = reduced(value)
    if value == 0:
        fail("independent exact oracle reached a zero pivot")
    return pow(value, -1, PRIME)


def quadratic_character(value: int) -> int:
    value = reduced(value)
    if value == 0:
        return 0
    return 1 if pow(value, (PRIME - 1) // 2, PRIME) == 1 else -1


def reference_determinant(matrix: list[list[int]]) -> int:
    work = [[reduced(value) for value in row] for row in matrix]
    size = len(work)
    result = 1
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if work[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        pivot_value = work[column][column]
        result = reduced(result * pivot_value)
        pivot_inverse = field_inverse(pivot_value)
        for row in range(column + 1, size):
            factor = reduced(work[row][column] * pivot_inverse)
            if factor:
                work[row] = [
                    reduced(left - factor * right)
                    for left, right in zip(
                        work[row],
                        work[column],
                        strict=True,
                    )
                ]
    return reduced(result)


def reference_inverse(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    work = [
        [reduced(value) for value in row]
        + [1 if row_index == column else 0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if work[row][column] != 0
            ),
            None,
        )
        if pivot is None:
            fail("independent exact oracle found a singular boundary")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        pivot_inverse = field_inverse(work[column][column])
        work[column] = [
            reduced(value * pivot_inverse) for value in work[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    reduced(left - factor * right)
                    for left, right in zip(
                        work[row],
                        work[column],
                        strict=True,
                    )
                ]
    return [row[size:] for row in work]


def make_exact_state(ports: int) -> dict[str, Any]:
    return {
        "quadratic": [
            [1 if row == column else 0 for column in range(ports)]
            for row in range(ports)
        ],
        "linear": [reduced(3 * port + 1) for port in range(ports)],
        "constant": reduced(ports),
        "sign": 1,
    }


def apply_exact_shear(
    state: dict[str, Any],
    module: dict[str, Any],
    direction: int,
) -> None:
    quadratic = state["quadratic"]
    linear = state["linear"]
    for left, right, value in module["quadratic"]:
        left = int(left)
        right = int(right)
        value = reduced(direction * int(value))
        quadratic[left][right] = reduced(
            quadratic[left][right] + value
        )
        if left != right:
            quadratic[right][left] = reduced(
                quadratic[right][left] + value
            )
    for port, value in module["linear"]:
        port = int(port)
        linear[port] = reduced(
            linear[port] + direction * int(value)
        )
    state["constant"] = reduced(
        state["constant"] + direction * int(module["constant"])
    )


def apply_exact_fourier(
    state: dict[str, Any],
    port: int,
    kernel_sign: int,
) -> None:
    quadratic = state["quadratic"]
    linear = state["linear"]
    size = len(linear)
    pivot = quadratic[port][port]
    pivot_inverse = field_inverse(pivot)
    others = [index for index in range(size) if index != port]
    vector = [quadratic[port][other] for other in others]
    bias = linear[port]
    updated = [row.copy() for row in quadratic]
    for left_offset, left in enumerate(others):
        for right_offset, right in enumerate(others):
            updated[left][right] = reduced(
                quadratic[left][right]
                - pivot_inverse
                * vector[left_offset]
                * vector[right_offset]
            )
    updated[port][port] = reduced(-pivot_inverse)
    for offset, other in enumerate(others):
        cross = reduced(
            -kernel_sign * pivot_inverse * vector[offset]
        )
        updated[port][other] = cross
        updated[other][port] = cross
    updated_linear = linear.copy()
    for offset, other in enumerate(others):
        updated_linear[other] = reduced(
            linear[other] - pivot_inverse * bias * vector[offset]
        )
    updated_linear[port] = reduced(
        -kernel_sign * pivot_inverse * bias
    )
    state["constant"] = reduced(
        state["constant"] - INV2 * pivot_inverse * bias * bias
    )
    state["sign"] *= quadratic_character(pivot * INV2)
    state["quadratic"] = updated
    state["linear"] = updated_linear


def apply_exact_module(
    state: dict[str, Any],
    module: dict[str, Any],
) -> None:
    apply_exact_shear(state, module, 1)
    port = int(module["fourier_port"])
    if port >= 0:
        apply_exact_fourier(state, port, 1)


def inverse_exact_module(
    state: dict[str, Any],
    module: dict[str, Any],
) -> None:
    port = int(module["fourier_port"])
    if port >= 0:
        apply_exact_fourier(state, port, -1)
    apply_exact_shear(state, module, -1)


def exact_execute(
    state: dict[str, Any],
    program: list[dict[str, Any]],
) -> None:
    for module in program:
        apply_exact_module(state, module)


def exact_restore(
    state: dict[str, Any],
    program: list[dict[str, Any]],
) -> None:
    for module in reversed(program):
        inverse_exact_module(state, module)


def reference_boundary(state: dict[str, Any]) -> dict[str, int]:
    quadratic = state["quadratic"]
    linear = state["linear"]
    size = len(linear)
    inverse_quadratic = reference_inverse(quadratic)
    completed = sum(
        linear[row] * inverse_quadratic[row][column] * linear[column]
        for row in range(size)
        for column in range(size)
    )
    determinant = reference_determinant(quadratic)
    sign = quadratic_character(
        determinant * pow(INV2, size, PRIME)
    )
    return {
        "overlap_probability_numerator": 1,
        "overlap_probability_denominator_base": PRIME,
        "overlap_probability_denominator_power": size,
        "root_order": PRIME,
        "root_exponent": reduced(
            state["constant"] - INV2 * completed
        ),
        "real_sign": int(state["sign"]) * sign,
        "quadratic_rank": size,
    }


def coordinate_grids(ports: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.indices((PRIME,) * ports, sparse=True, dtype=np.int64)
    )


def make_state(ports: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    coordinates = coordinate_grids(ports)
    exponent: np.ndarray | int = ports
    for port, coordinate in enumerate(coordinates):
        exponent = (
            exponent
            + INV2 * coordinate * coordinate
            + (3 * port + 1) * coordinate
        )
    state = ROOTS[np.asarray(exponent, dtype=np.int64) % PRIME]
    state = state / math.sqrt(PRIME**ports)
    return np.asarray(state, dtype=np.complex128), coordinates


def shear_exponent(
    module: dict[str, Any],
    coordinates: tuple[np.ndarray, ...],
) -> np.ndarray:
    result: np.ndarray | int = int(module["constant"])
    for left, right, value in module["quadratic"]:
        left = int(left)
        right = int(right)
        value = int(value)
        factor = INV2 if left == right else 1
        result = (
            result
            + factor
            * value
            * coordinates[left]
            * coordinates[right]
        )
    for port, value in module["linear"]:
        result = (
            result
            + int(value) * coordinates[int(port)]
        )
    return np.asarray(result, dtype=np.int64) % PRIME


def apply_module(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    module: dict[str, Any],
) -> np.ndarray:
    state = state * ROOTS[shear_exponent(module, coordinates)]
    port = int(module["fourier_port"])
    if port < 0:
        return state
    moved = np.moveaxis(state, port, 0)
    transformed = np.tensordot(FOURIER, moved, axes=([1], [0]))
    return np.moveaxis(transformed, 0, port)


def inverse_module(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    module: dict[str, Any],
) -> np.ndarray:
    port = int(module["fourier_port"])
    if port >= 0:
        moved = np.moveaxis(state, port, 0)
        transformed = np.tensordot(
            np.conjugate(FOURIER),
            moved,
            axes=([1], [0]),
        )
        state = np.moveaxis(transformed, 0, port)
    return state * np.conjugate(
        ROOTS[shear_exponent(module, coordinates)]
    )


def execute(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    program: list[dict[str, Any]],
) -> np.ndarray:
    for module in program:
        state = apply_module(state, coordinates, module)
    return state


def restore(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    program: list[dict[str, Any]],
) -> np.ndarray:
    for module in reversed(program):
        state = inverse_module(state, coordinates, module)
    return state


def projected_overlap(state: np.ndarray) -> complex:
    return complex(np.sum(state) / math.sqrt(state.size))


def exact_boundary_complex(boundary: dict[str, Any]) -> complex:
    if (
        int(boundary["overlap_probability_numerator"]) != 1
        or int(boundary["overlap_probability_denominator_base"])
        != PRIME
        or int(boundary["root_order"]) != PRIME
    ):
        fail("production exact boundary format mismatch")
    power = int(boundary["overlap_probability_denominator_power"])
    magnitude = PRIME ** (-0.5 * power)
    return (
        int(boundary["real_sign"])
        * magnitude
        * ROOTS[int(boundary["root_exponent"])]
    )


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_gaussian_multi_port_dense_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    production_cases = {
        int(case["ports"]): case for case in production["cases"]
    }
    cases: list[dict[str, object]] = []
    for ports in TESTED_PORTS:
        case = production_cases[ports]
        primary = case["public_primary_program"]
        reuse = case["public_reuse_program"]
        if (
            hashlib.sha256(encoded_program(primary)).hexdigest()
            != case["primary_program_sha256"]
            or hashlib.sha256(encoded_program(reuse)).hexdigest()
            != case["reuse_program_sha256"]
        ):
            fail("public descriptor hash mismatch")
        exact_baseline = make_exact_state(ports)
        exact_primary = make_exact_state(ports)
        exact_execute(exact_primary, primary)
        if reference_boundary(exact_primary) != case["primary_boundary"]:
            fail(f"independent exact primary boundary mismatch at {ports}")
        exact_restore(exact_primary, primary)
        primary_exact_restoration = exact_primary == exact_baseline
        exact_execute(exact_primary, reuse)
        if reference_boundary(exact_primary) != case["reuse_boundary"]:
            fail(f"independent exact reuse boundary mismatch at {ports}")
        exact_restore(exact_primary, reuse)
        reuse_exact_restoration = exact_primary == exact_baseline
        if not primary_exact_restoration or not reuse_exact_restoration:
            fail(f"independent exact restoration failed at {ports}")
        case_result: dict[str, object] = {
            "ports": ports,
            "primary_program_sha256": case["primary_program_sha256"],
            "reuse_program_sha256": case["reuse_program_sha256"],
            "exact_primary_boundary_equal": True,
            "exact_primary_restoration_equal": True,
            "exact_reuse_boundary_equal": True,
            "exact_reuse_restoration_equal": True,
            "exact_reference_f17_field_cells": ports * ports + ports + 1,
            "dense_phase_vector_materialized": ports in DENSE_PORTS,
            "verification_dense_logical_complex_cells": (
                PRIME**ports if ports in DENSE_PORTS else 0
            ),
        }
        if ports in DENSE_PORTS:
            baseline, coordinates = make_state(ports)
            forward = execute(baseline.copy(), coordinates, primary)
            norm_error = abs(float(np.linalg.norm(forward)) - 1.0)
            boundary_error = abs(
                projected_overlap(forward)
                - exact_boundary_complex(case["primary_boundary"])
            )
            restored = restore(forward.copy(), coordinates, primary)
            restoration_error = max_abs(restored, baseline)
            reuse_forward = execute(restored, coordinates, reuse)
            reuse_boundary_error = abs(
                projected_overlap(reuse_forward)
                - exact_boundary_complex(case["reuse_boundary"])
            )
            reuse_restored = restore(
                reuse_forward.copy(),
                coordinates,
                reuse,
            )
            reuse_restoration_error = max_abs(
                reuse_restored,
                baseline,
            )
            if max(
                norm_error,
                boundary_error,
                restoration_error,
                reuse_boundary_error,
                reuse_restoration_error,
            ) > TOLERANCE:
                fail(
                    f"independent dense parity failed at {ports} ports"
                )
            case_result.update(
                {
                    "forward_norm_error": norm_error,
                    "primary_boundary_complex_abs_error": (
                        boundary_error
                    ),
                    "primary_restoration_max_abs_error": (
                        restoration_error
                    ),
                    "reuse_boundary_complex_abs_error": (
                        reuse_boundary_error
                    ),
                    "reuse_restoration_max_abs_error": (
                        reuse_restoration_error
                    ),
                }
            )
        cases.append(case_result)
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PYTHON_INTEGER_F17_GAUSSIAN_RECURRENCE_"
            "PLUS_EXPLICIT_COMPLEX_PHASE_VECTOR_PARITY"
        ),
        "predeclared_tolerance": TOLERANCE,
        "tested_ports": list(TESTED_PORTS),
        "dense_tested_ports": list(DENSE_PORTS),
        "cases": cases,
        "production_backend_imported": False,
        "production_projection_called": False,
        "production_compiler_called": False,
        "public_descriptors_consumed": True,
        "verification_assignment_expansion_materialized": True,
        "verification_dense_logical_vector_cells_counted": True,
        "dense_logical_vector_cells_are_not_process_peak": True,
        "numpy_dense_oracle_process_peak_bounded": False,
        "maximum_verification_dense_logical_complex_cells": max(
            int(case["verification_dense_logical_complex_cells"])
            for case in cases
        ),
        "dense_oracle_is_accepted_compact_carrier_path": False,
        "exact_reference_uses_compact_quadratic_coefficients": True,
        "exact_reference_reexecutes_all_tested_ports": True,
        "all_exact_boundaries_equal": True,
        "all_exact_restorations_equal": True,
        "exact_formula_supported_by_numerical_parity_at_dense_ports": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
