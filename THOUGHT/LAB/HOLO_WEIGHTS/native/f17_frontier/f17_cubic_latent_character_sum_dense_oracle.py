#!/usr/bin/env python3
"""Separate exact and dense oracle for the F17 cubic-latent quotient.

The oracle consumes hashed public module descriptors. It reexecutes the exact
coefficient recurrence with Python integer lists at all tested widths and
additionally materializes the explicit complex public-port state at ports two
and four. At two ports a second dense construction retains the latent
coordinate as an explicit independent axis through every public module before
the final trace. It does not import the production backend or call its compiler
or projection.
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
DELAYED_LATENT_AXIS_PORTS = (2,)
TOLERANCE = 3.0e-11
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


def encoded_program(program: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        program,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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
        "latent": [
            1 + ((7 * port) % (PRIME - 1))
            for port in range(ports)
        ],
        "constant": reduced(ports),
        "latent_cubic": 1,
        "latent_quadratic": 4,
        "latent_linear": 6,
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
    latent_coupling = state["latent"]
    size = len(linear)
    pivot = quadratic[port][port]
    pivot_inverse = field_inverse(pivot)
    others = [index for index in range(size) if index != port]
    vector = [quadratic[port][other] for other in others]
    bias = linear[port]
    latent = latent_coupling[port]
    updated = [row.copy() for row in quadratic]
    updated_linear = linear.copy()
    updated_latent = latent_coupling.copy()
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
        updated_linear[other] = reduced(
            linear[other] - pivot_inverse * bias * vector[offset]
        )
        updated_latent[other] = reduced(
            latent_coupling[other]
            - pivot_inverse * latent * vector[offset]
        )
    updated_linear[port] = reduced(
        -kernel_sign * pivot_inverse * bias
    )
    updated_latent[port] = reduced(
        -kernel_sign * pivot_inverse * latent
    )
    state["constant"] = reduced(
        state["constant"] - INV2 * pivot_inverse * bias * bias
    )
    state["latent_quadratic"] = reduced(
        state["latent_quadratic"]
        - pivot_inverse * latent * latent
    )
    state["latent_linear"] = reduced(
        state["latent_linear"]
        - pivot_inverse * bias * latent
    )
    state["sign"] *= quadratic_character(pivot * INV2)
    state["quadratic"] = updated
    state["linear"] = updated_linear
    state["latent"] = updated_latent


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


def bilinear(
    left: list[int],
    matrix: list[list[int]],
    right: list[int],
) -> int:
    return sum(
        left[row] * matrix[row][column] * right[column]
        for row in range(len(left))
        for column in range(len(right))
    )


def reference_boundary(state: dict[str, Any]) -> dict[str, object]:
    quadratic = state["quadratic"]
    linear = state["linear"]
    latent = state["latent"]
    size = len(linear)
    inverse_quadratic = reference_inverse(quadratic)
    reduced_constant = reduced(
        state["constant"]
        - INV2 * bilinear(linear, inverse_quadratic, linear)
    )
    reduced_quadratic = reduced(
        state["latent_quadratic"]
        - bilinear(latent, inverse_quadratic, latent)
    )
    reduced_linear = reduced(
        state["latent_linear"]
        - bilinear(linear, inverse_quadratic, latent)
    )
    determinant = reference_determinant(quadratic)
    sign = state["sign"] * quadratic_character(
        determinant * pow(INV2, size, PRIME)
    )
    histogram = [0] * PRIME
    for latent_value in range(PRIME):
        exponent = reduced(
            state["latent_cubic"] * latent_value**3
            + INV2 * reduced_quadratic * latent_value**2
            + reduced_linear * latent_value
            + reduced_constant
        )
        histogram[exponent] += 1
    canonical = [
        sign * (histogram[index] - histogram[PRIME - 1])
        for index in range(PRIME - 1)
    ]
    return {
        "root_order": PRIME,
        "normalization_denominator_base": PRIME,
        "normalization_denominator_sqrt_power": size + 1,
        "canonical_cyclotomic_coefficients": canonical,
        "canonical_nonzero_coefficients": sum(
            value != 0 for value in canonical
        ),
        "canonical_l1_coefficient_weight": sum(
            abs(value) for value in canonical
        ),
        "latent_trace_field_evaluations": PRIME,
        "public_gaussian_rank": size,
    }


def coordinate_grids(ports: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.indices((PRIME,) * ports, sparse=True, dtype=np.int64)
    )


def make_dense_state(
    ports: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    coordinates = coordinate_grids(ports)
    state: np.ndarray | complex = 0.0j
    for latent in range(PRIME):
        exponent: np.ndarray | int = (
            ports
            + latent**3
            + INV2 * 4 * latent**2
            + 6 * latent
        )
        for port, coordinate in enumerate(coordinates):
            exponent = (
                exponent
                + INV2 * coordinate * coordinate
                + (3 * port + 1) * coordinate
                + latent
                * (1 + ((7 * port) % (PRIME - 1)))
                * coordinate
            )
        state = state + ROOTS[
            np.asarray(exponent, dtype=np.int64) % PRIME
        ]
    state = state / math.sqrt(PRIME ** (ports + 1))
    return np.asarray(state, dtype=np.complex128), coordinates


def make_delayed_latent_state(
    ports: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if ports not in DELAYED_LATENT_AXIS_PORTS:
        fail("delayed latent-axis construction requested outside its scope")
    values = np.arange(PRIME, dtype=np.int64)
    coordinates = tuple(
        values.reshape(
            (1,) * port + (PRIME,) + (1,) * (ports - port)
        )
        for port in range(ports)
    )
    latent = values.reshape((1,) * ports + (PRIME,))
    exponent: np.ndarray | int = (
        ports
        + latent**3
        + INV2 * 4 * latent**2
        + 6 * latent
    )
    for port, coordinate in enumerate(coordinates):
        exponent = (
            exponent
            + INV2 * coordinate * coordinate
            + (3 * port + 1) * coordinate
            + latent
            * (1 + ((7 * port) % (PRIME - 1)))
            * coordinate
        )
    state = ROOTS[np.asarray(exponent, dtype=np.int64) % PRIME]
    state = state / math.sqrt(PRIME ** (ports + 1))
    expected_shape = (PRIME,) * (ports + 1)
    if state.shape != expected_shape:
        fail("delayed latent-axis construction has an unexpected shape")
    return np.asarray(state, dtype=np.complex128), coordinates


def shear_exponent(
    module: dict[str, Any],
    coordinates: tuple[np.ndarray, ...],
) -> np.ndarray:
    result: np.ndarray | int = int(module["constant"])
    for left, right, value in module["quadratic"]:
        left = int(left)
        right = int(right)
        factor = INV2 if left == right else 1
        result = (
            result
            + factor
            * int(value)
            * coordinates[left]
            * coordinates[right]
        )
    for port, value in module["linear"]:
        result = (
            result + int(value) * coordinates[int(port)]
        )
    return np.asarray(result, dtype=np.int64) % PRIME


def apply_dense_module(
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


def inverse_dense_module(
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


def dense_execute(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    program: list[dict[str, Any]],
) -> np.ndarray:
    for module in program:
        state = apply_dense_module(state, coordinates, module)
    return state


def dense_restore(
    state: np.ndarray,
    coordinates: tuple[np.ndarray, ...],
    program: list[dict[str, Any]],
) -> np.ndarray:
    for module in reversed(program):
        state = inverse_dense_module(state, coordinates, module)
    return state


def projected_overlap(state: np.ndarray) -> complex:
    return complex(np.sum(state) / math.sqrt(state.size))


def projected_delayed_latent_overlap(
    state: np.ndarray,
    ports: int,
) -> complex:
    return complex(np.sum(state) / math.sqrt(PRIME**ports))


def exact_boundary_complex(boundary: dict[str, Any]) -> complex:
    if (
        int(boundary["normalization_denominator_base"]) != PRIME
        or int(boundary["root_order"]) != PRIME
    ):
        fail("production exact boundary format mismatch")
    coefficients = boundary["canonical_cyclotomic_coefficients"]
    cyclotomic = sum(
        int(value) * ROOTS[index]
        for index, value in enumerate(coefficients)
    )
    power = int(boundary["normalization_denominator_sqrt_power"])
    return cyclotomic * PRIME ** (-0.5 * power)


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_cubic_latent_character_sum_dense_oracle.py "
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
        exact_state = make_exact_state(ports)
        exact_execute(exact_state, primary)
        if reference_boundary(exact_state) != case["primary_boundary"]:
            fail(f"independent exact primary boundary mismatch at {ports}")
        exact_restore(exact_state, primary)
        primary_restored = exact_state == exact_baseline
        exact_execute(exact_state, reuse)
        if reference_boundary(exact_state) != case["reuse_boundary"]:
            fail(f"independent exact reuse boundary mismatch at {ports}")
        exact_restore(exact_state, reuse)
        reuse_restored = exact_state == exact_baseline
        if not primary_restored or not reuse_restored:
            fail(f"independent exact restoration failed at {ports}")
        case_result: dict[str, object] = {
            "ports": ports,
            "primary_program_sha256": case["primary_program_sha256"],
            "reuse_program_sha256": case["reuse_program_sha256"],
            "exact_primary_boundary_equal": True,
            "exact_primary_restoration_equal": True,
            "exact_reuse_boundary_equal": True,
            "exact_reuse_restoration_equal": True,
            "exact_reference_f17_field_cells": (
                ports * ports + 2 * ports + 4
            ),
            "dense_public_port_state_materialized": ports in DENSE_PORTS,
            "verification_dense_public_port_logical_complex_cells": (
                PRIME**ports if ports in DENSE_PORTS else 0
            ),
            "verification_dense_phase_term_evaluations": (
                PRIME ** (ports + 1) if ports in DENSE_PORTS else 0
            ),
            "verification_dense_public_fourier_transforms": (
                2
                * (
                    sum(
                        int(module["fourier_port"]) >= 0
                        for module in primary
                    )
                    + sum(
                        int(module["fourier_port"]) >= 0
                        for module in reuse
                    )
                )
                if ports in DENSE_PORTS
                else 0
            ),
            "delayed_latent_axis_materialized": (
                ports in DELAYED_LATENT_AXIS_PORTS
            ),
            "verification_delayed_latent_logical_complex_cells": (
                PRIME ** (ports + 1)
                if ports in DELAYED_LATENT_AXIS_PORTS
                else 0
            ),
            "verification_delayed_latent_public_fourier_transforms": (
                2
                * (
                    sum(
                        int(module["fourier_port"]) >= 0
                        for module in primary
                    )
                    + sum(
                        int(module["fourier_port"]) >= 0
                        for module in reuse
                    )
                )
                if ports in DELAYED_LATENT_AXIS_PORTS
                else 0
            ),
        }
        if ports in DENSE_PORTS:
            baseline, coordinates = make_dense_state(ports)
            forward = dense_execute(
                baseline.copy(),
                coordinates,
                primary,
            )
            boundary_error = abs(
                projected_overlap(forward)
                - exact_boundary_complex(case["primary_boundary"])
            )
            restored = dense_restore(
                forward.copy(),
                coordinates,
                primary,
            )
            restoration_error = max_abs(restored, baseline)
            reuse_forward = dense_execute(restored, coordinates, reuse)
            reuse_boundary_error = abs(
                projected_overlap(reuse_forward)
                - exact_boundary_complex(case["reuse_boundary"])
            )
            reuse_restored_state = dense_restore(
                reuse_forward.copy(),
                coordinates,
                reuse,
            )
            reuse_restoration_error = max_abs(
                reuse_restored_state,
                baseline,
            )
            if max(
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
        if ports in DELAYED_LATENT_AXIS_PORTS:
            delayed_baseline, delayed_coordinates = (
                make_delayed_latent_state(ports)
            )
            delayed_forward = dense_execute(
                delayed_baseline.copy(),
                delayed_coordinates,
                primary,
            )
            if delayed_forward.shape != (PRIME,) * (ports + 1):
                fail("public modules contracted the delayed latent axis")
            delayed_boundary_error = abs(
                projected_delayed_latent_overlap(
                    delayed_forward,
                    ports,
                )
                - exact_boundary_complex(case["primary_boundary"])
            )
            delayed_restored = dense_restore(
                delayed_forward.copy(),
                delayed_coordinates,
                primary,
            )
            delayed_restoration_error = max_abs(
                delayed_restored,
                delayed_baseline,
            )
            delayed_reuse_forward = dense_execute(
                delayed_restored,
                delayed_coordinates,
                reuse,
            )
            delayed_reuse_boundary_error = abs(
                projected_delayed_latent_overlap(
                    delayed_reuse_forward,
                    ports,
                )
                - exact_boundary_complex(case["reuse_boundary"])
            )
            delayed_reuse_restored = dense_restore(
                delayed_reuse_forward.copy(),
                delayed_coordinates,
                reuse,
            )
            delayed_reuse_restoration_error = max_abs(
                delayed_reuse_restored,
                delayed_baseline,
            )
            if max(
                delayed_boundary_error,
                delayed_restoration_error,
                delayed_reuse_boundary_error,
                delayed_reuse_restoration_error,
            ) > TOLERANCE:
                fail(
                    "independent delayed latent-axis parity failed "
                    f"at {ports} ports"
                )
            case_result.update(
                {
                    "delayed_latent_axis_size_after_forward": (
                        delayed_forward.shape[-1]
                    ),
                    "delayed_latent_primary_boundary_complex_abs_error": (
                        delayed_boundary_error
                    ),
                    "delayed_latent_primary_restoration_max_abs_error": (
                        delayed_restoration_error
                    ),
                    "delayed_latent_reuse_boundary_complex_abs_error": (
                        delayed_reuse_boundary_error
                    ),
                    "delayed_latent_reuse_restoration_max_abs_error": (
                        delayed_reuse_restoration_error
                    ),
                }
            )
        cases.append(case_result)
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PYTHON_INTEGER_F17_CUBIC_LATENT_RECURRENCE_"
            "PLUS_EXPLICIT_COMPLEX_PUBLIC_PORT_STATE_PARITY"
        ),
        "predeclared_tolerance": TOLERANCE,
        "tested_ports": list(TESTED_PORTS),
        "dense_tested_ports": list(DENSE_PORTS),
        "delayed_latent_axis_tested_ports": list(
            DELAYED_LATENT_AXIS_PORTS
        ),
        "cases": cases,
        "production_backend_imported": False,
        "production_compiler_called": False,
        "production_projection_called": False,
        "public_descriptors_consumed": True,
        "exact_reference_reexecutes_all_tested_ports": True,
        "all_exact_boundaries_equal": True,
        "all_exact_restorations_equal": True,
        "dense_expansion_is_verification_only": True,
        "independent_delayed_latent_residency_parity": True,
        "dense_logical_cells_are_not_process_peak": True,
        "numpy_dense_oracle_process_peak_bounded": False,
        "exact_reference_python_allocator_peak_bounded": False,
        "maximum_verification_dense_public_port_logical_complex_cells": max(
            int(case[
                "verification_dense_public_port_logical_complex_cells"
            ])
            for case in cases
        ),
        "maximum_verification_dense_phase_term_evaluations": max(
            int(case["verification_dense_phase_term_evaluations"])
            for case in cases
        ),
        "exact_formula_supported_by_dense_parity_at_dense_ports": True,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
