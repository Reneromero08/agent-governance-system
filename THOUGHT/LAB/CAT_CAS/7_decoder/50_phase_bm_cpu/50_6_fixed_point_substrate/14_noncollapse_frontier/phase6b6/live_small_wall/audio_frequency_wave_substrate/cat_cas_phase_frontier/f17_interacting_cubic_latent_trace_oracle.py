#!/usr/bin/env python3
"""Separate exact oracle for the interacting-cubic-latent diagnostic.

The oracle consumes public module descriptors and independently reimplements
the F17 coefficient recurrence, final character trace, inverse, and reuse with
Python integer lists. For the declared ports-two/latents-two case it also
retains both latent coordinates as explicit dense axes throughout every
public module and compares the final complex boundary and restoration.

It does not import the production diagnostic, compiler, or projection.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PRIME = 17
INV2 = pow(2, -1, PRIME)
FIXED_PORTS = 4
TESTED_LATENTS = (1, 2, 3, 4)
DENSE_CASE = (2, 2)
TEST_CASES = tuple((FIXED_PORTS, count) for count in TESTED_LATENTS) + (
    DENSE_CASE,
)
TOLERANCE = 5.0e-11
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


def quadratic_pairs(latents: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(latents)
        for right in range(left, latents)
    )


def cubic_triples(
    latents: int,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (first, second, third)
        for first in range(latents)
        for second in range(first, latents)
        for third in range(second, latents)
    )


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
        + [
            1 if row_index == column else 0
            for column in range(size)
        ]
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


def make_exact_state(ports: int, latents: int) -> dict[str, Any]:
    return {
        "public_quadratic": [
            [
                1 if row == column else 0
                for column in range(ports)
            ]
            for row in range(ports)
        ],
        "public_linear": [
            reduced(3 * port + 1) for port in range(ports)
        ],
        "latent_coupling": [
            [
                1
                + (
                    (7 * port + 5 * latent)
                    % (PRIME - 1)
                )
                for latent in range(latents)
            ]
            for port in range(ports)
        ],
        "latent_quadratic": [
            1
            + (
                (5 * left + 9 * right)
                % (PRIME - 1)
            )
            for left, right in quadratic_pairs(latents)
        ],
        "latent_linear": [
            reduced(6 + 7 * latent)
            for latent in range(latents)
        ],
        "latent_cubic": [
            1
            + (
                (3 * first + 5 * second + 7 * third)
                % (PRIME - 1)
            )
            for first, second, third in cubic_triples(latents)
        ],
        "constant": reduced(ports + 2 * latents),
        "sign": 1,
    }


def apply_exact_shear(
    state: dict[str, Any],
    module: dict[str, Any],
    direction: int,
) -> None:
    quadratic = state["public_quadratic"]
    linear = state["public_linear"]
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
    quadratic = state["public_quadratic"]
    linear = state["public_linear"]
    coupling = state["latent_coupling"]
    ports = len(linear)
    latents = len(state["latent_linear"])
    pivot = quadratic[port][port]
    pivot_inverse = field_inverse(pivot)
    others = [index for index in range(ports) if index != port]
    vector = [quadratic[port][other] for other in others]
    bias = linear[port]
    latent_row = coupling[port].copy()
    updated_quadratic = [row.copy() for row in quadratic]
    updated_linear = linear.copy()
    updated_coupling = [row.copy() for row in coupling]
    updated_latent_quadratic = state["latent_quadratic"].copy()
    for left_offset, left in enumerate(others):
        for right_offset, right in enumerate(others):
            updated_quadratic[left][right] = reduced(
                quadratic[left][right]
                - pivot_inverse
                * vector[left_offset]
                * vector[right_offset]
            )
    updated_quadratic[port][port] = reduced(-pivot_inverse)
    for offset, other in enumerate(others):
        cross = reduced(
            -kernel_sign * pivot_inverse * vector[offset]
        )
        updated_quadratic[port][other] = cross
        updated_quadratic[other][port] = cross
        updated_linear[other] = reduced(
            linear[other] - pivot_inverse * bias * vector[offset]
        )
        for latent in range(latents):
            updated_coupling[other][latent] = reduced(
                coupling[other][latent]
                - pivot_inverse
                * vector[offset]
                * latent_row[latent]
            )
    updated_linear[port] = reduced(
        -kernel_sign * pivot_inverse * bias
    )
    for latent in range(latents):
        updated_coupling[port][latent] = reduced(
            -kernel_sign * pivot_inverse * latent_row[latent]
        )
        state["latent_linear"][latent] = reduced(
            state["latent_linear"][latent]
            - pivot_inverse * bias * latent_row[latent]
        )
    for index, (left, right) in enumerate(
        quadratic_pairs(latents)
    ):
        updated_latent_quadratic[index] = reduced(
            state["latent_quadratic"][index]
            - pivot_inverse
            * latent_row[left]
            * latent_row[right]
        )
    state["constant"] = reduced(
        state["constant"]
        - INV2 * pivot_inverse * bias * bias
    )
    state["sign"] *= quadratic_character(pivot * INV2)
    state["public_quadratic"] = updated_quadratic
    state["public_linear"] = updated_linear
    state["latent_coupling"] = updated_coupling
    state["latent_quadratic"] = updated_latent_quadratic


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


def latent_exponent(
    values: tuple[int, ...],
    quadratic: list[int],
    linear: list[int],
    cubic: list[int],
) -> int:
    exponent = 0
    for coefficient, (left, right) in zip(
        quadratic,
        quadratic_pairs(len(values)),
        strict=True,
    ):
        factor = INV2 if left == right else 1
        exponent += (
            factor * coefficient * values[left] * values[right]
        )
    exponent += sum(
        coefficient * value
        for coefficient, value in zip(
            linear,
            values,
            strict=True,
        )
    )
    exponent += sum(
        coefficient
        * values[first]
        * values[second]
        * values[third]
        for coefficient, (first, second, third) in zip(
            cubic,
            cubic_triples(len(values)),
            strict=True,
        )
    )
    return reduced(exponent)


def reference_boundary(state: dict[str, Any]) -> dict[str, object]:
    quadratic = state["public_quadratic"]
    linear = state["public_linear"]
    coupling = state["latent_coupling"]
    ports = len(linear)
    latents = len(state["latent_linear"])
    inverse_quadratic = reference_inverse(quadratic)
    reduced_constant = reduced(
        state["constant"]
        - INV2 * bilinear(
            linear,
            inverse_quadratic,
            linear,
        )
    )
    reduced_linear = [
        reduced(
            state["latent_linear"][latent]
            - bilinear(
                linear,
                inverse_quadratic,
                [
                    coupling[port][latent]
                    for port in range(ports)
                ],
            )
        )
        for latent in range(latents)
    ]
    reduced_quadratic = []
    for index, (left, right) in enumerate(
        quadratic_pairs(latents)
    ):
        left_vector = [
            coupling[port][left] for port in range(ports)
        ]
        right_vector = [
            coupling[port][right] for port in range(ports)
        ]
        reduced_quadratic.append(
            reduced(
                state["latent_quadratic"][index]
                - bilinear(
                    left_vector,
                    inverse_quadratic,
                    right_vector,
                )
            )
        )
    determinant = reference_determinant(quadratic)
    sign = state["sign"] * quadratic_character(
        determinant * pow(INV2, ports, PRIME)
    )
    histogram = [0] * PRIME
    for values in itertools.product(range(PRIME), repeat=latents):
        exponent = reduced(
            latent_exponent(
                values,
                reduced_quadratic,
                reduced_linear,
                state["latent_cubic"],
            )
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
        "normalization_denominator_sqrt_power": ports + latents,
        "canonical_cyclotomic_coefficients": canonical,
        "canonical_nonzero_coefficients": sum(
            value != 0 for value in canonical
        ),
        "canonical_l1_coefficient_weight": sum(
            abs(value) for value in canonical
        ),
        "canonical_signed_bit_width": max(
            (abs(value).bit_length() + 1 for value in canonical),
            default=1,
        ),
        "latent_trace_assignments_streamed": PRIME**latents,
        "latent_trace_histogram_cells": PRIME,
        "latent_trace_histogram_integer_payload_bits": sum(
            max(1, int(value).bit_length()) for value in histogram
        ),
        "canonical_cyclotomic_integer_payload_bits": sum(
            max(1, abs(value).bit_length() + 1)
            for value in canonical
        ),
        "public_gaussian_rank": ports,
    }


def make_dense_state(
    ports: int,
    latents: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    dimensions = ports + latents
    values = np.arange(PRIME, dtype=np.int64)
    coordinates = tuple(
        values.reshape(
            (1,) * axis
            + (PRIME,)
            + (1,) * (dimensions - axis - 1)
        )
        for axis in range(dimensions)
    )
    public = coordinates[:ports]
    latent = coordinates[ports:]
    exponent: np.ndarray | int = ports + 2 * latents
    for port, coordinate in enumerate(public):
        exponent = (
            exponent
            + INV2 * coordinate * coordinate
            + (3 * port + 1) * coordinate
        )
        for latent_index, latent_coordinate in enumerate(latent):
            exponent = (
                exponent
                + (
                    1
                    + (
                        (7 * port + 5 * latent_index)
                        % (PRIME - 1)
                    )
                )
                * coordinate
                * latent_coordinate
            )
    for coefficient, (left, right) in zip(
        make_exact_state(ports, latents)["latent_quadratic"],
        quadratic_pairs(latents),
        strict=True,
    ):
        factor = INV2 if left == right else 1
        exponent = (
            exponent
            + factor
            * coefficient
            * latent[left]
            * latent[right]
        )
    for index, coordinate in enumerate(latent):
        exponent = exponent + (6 + 7 * index) * coordinate
    for coefficient, (first, second, third) in zip(
        make_exact_state(ports, latents)["latent_cubic"],
        cubic_triples(latents),
        strict=True,
    ):
        exponent = (
            exponent
            + coefficient
            * latent[first]
            * latent[second]
            * latent[third]
        )
    state = ROOTS[np.asarray(exponent, dtype=np.int64) % PRIME]
    state = state / math.sqrt(PRIME ** (ports + latents))
    expected_shape = (PRIME,) * dimensions
    if state.shape != expected_shape:
        fail("dense latent-axis state has an unexpected shape")
    return np.asarray(state, dtype=np.complex128), public


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


def exact_boundary_complex(boundary: dict[str, Any]) -> complex:
    coefficients = boundary["canonical_cyclotomic_coefficients"]
    cyclotomic = sum(
        int(value) * ROOTS[index]
        for index, value in enumerate(coefficients)
    )
    power = int(boundary["normalization_denominator_sqrt_power"])
    return cyclotomic * PRIME ** (-0.5 * power)


def dense_projected_boundary(state: np.ndarray, ports: int) -> complex:
    return complex(np.sum(state) / math.sqrt(PRIME**ports))


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> int:
    if len(sys.argv) != 2:
        fail(
            "usage: f17_interacting_cubic_latent_trace_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    production_cases = {
        (int(case["ports"]), int(case["latents"])): case
        for case in production["cases"]
    }
    cases: list[dict[str, object]] = []
    for ports, latents in TEST_CASES:
        case = production_cases[(ports, latents)]
        primary = case["public_primary_program"]
        reuse = case["public_reuse_program"]
        if (
            hashlib.sha256(encoded_program(primary)).hexdigest()
            != case["primary_program_sha256"]
            or hashlib.sha256(encoded_program(reuse)).hexdigest()
            != case["reuse_program_sha256"]
        ):
            fail("public descriptor hash mismatch")
        exact_baseline = make_exact_state(ports, latents)
        exact_state = make_exact_state(ports, latents)
        exact_execute(exact_state, primary)
        if reference_boundary(exact_state) != case["primary_boundary"]:
            fail(
                "independent exact primary boundary mismatch "
                f"at ports={ports}, latents={latents}"
            )
        exact_restore(exact_state, primary)
        primary_restored = exact_state == exact_baseline
        exact_execute(exact_state, reuse)
        if reference_boundary(exact_state) != case["reuse_boundary"]:
            fail(
                "independent exact reuse boundary mismatch "
                f"at ports={ports}, latents={latents}"
            )
        exact_restore(exact_state, reuse)
        reuse_restored = exact_state == exact_baseline
        if not primary_restored or not reuse_restored:
            fail(
                "independent exact restoration failed "
                f"at ports={ports}, latents={latents}"
            )
        field_cells = (
            ports * ports
            + ports
            + ports * latents
            + math.comb(latents + 1, 2)
            + latents
            + math.comb(latents + 2, 3)
            + 1
        )
        case_result: dict[str, object] = {
            "ports": ports,
            "latents": latents,
            "primary_program_sha256": case["primary_program_sha256"],
            "reuse_program_sha256": case["reuse_program_sha256"],
            "exact_primary_boundary_equal": True,
            "exact_primary_restoration_equal": True,
            "exact_reuse_boundary_equal": True,
            "exact_reuse_restoration_equal": True,
            "exact_reference_f17_field_cells": field_cells,
            "exact_reference_trace_assignments": PRIME**latents,
            "dense_latent_axes_materialized": (
                (ports, latents) == DENSE_CASE
            ),
            "verification_dense_logical_complex_cells": (
                PRIME ** (ports + latents)
                if (ports, latents) == DENSE_CASE
                else 0
            ),
        }
        if (ports, latents) == DENSE_CASE:
            baseline, coordinates = make_dense_state(ports, latents)
            forward = dense_execute(
                baseline.copy(),
                coordinates,
                primary,
            )
            expected_shape = (PRIME,) * (ports + latents)
            if forward.shape != expected_shape:
                fail("dense public modules contracted a latent axis")
            boundary_error = abs(
                dense_projected_boundary(forward, ports)
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
                dense_projected_boundary(reuse_forward, ports)
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
                fail("dense two-latent parity exceeded tolerance")
            fourier_count = 2 * (
                sum(
                    int(module["fourier_port"]) >= 0
                    for module in primary
                )
                + sum(
                    int(module["fourier_port"]) >= 0
                    for module in reuse
                )
            )
            case_result.update(
                {
                    "dense_latent_axis_sizes_after_forward": (
                        list(forward.shape[ports:])
                    ),
                    "dense_primary_boundary_complex_abs_error": (
                        boundary_error
                    ),
                    "dense_primary_restoration_max_abs_error": (
                        restoration_error
                    ),
                    "dense_reuse_boundary_complex_abs_error": (
                        reuse_boundary_error
                    ),
                    "dense_reuse_restoration_max_abs_error": (
                        reuse_restoration_error
                    ),
                    "verification_dense_public_fourier_transforms": (
                        fourier_count
                    ),
                }
            )
        cases.append(case_result)
    result = {
        "result": "PASS",
        "oracle": (
            "SEPARATE_PYTHON_INTEGER_F17_INTERACTING_CUBIC_LATENT_"
            "RECURRENCE_PLUS_EXPLICIT_TWO_LATENT_AXIS_COMPLEX_PARITY"
        ),
        "predeclared_tolerance": TOLERANCE,
        "tested_cases": [
            {"ports": ports, "latents": latents}
            for ports, latents in TEST_CASES
        ],
        "dense_tested_case": {
            "ports": DENSE_CASE[0],
            "latents": DENSE_CASE[1],
        },
        "cases": cases,
        "production_backend_imported": False,
        "production_compiler_called": False,
        "production_projection_called": False,
        "public_descriptors_consumed": True,
        "exact_reference_reexecutes_all_tested_cases": True,
        "all_exact_boundaries_equal": True,
        "all_exact_restorations_equal": True,
        "independent_two_latent_axis_residency_parity": True,
        "dense_expansion_is_verification_only": True,
        "dense_logical_cells_are_not_process_peak": True,
        "numpy_dense_oracle_process_peak_bounded": False,
        "exact_reference_python_allocator_peak_bounded": False,
        "maximum_verification_dense_logical_complex_cells": (
            PRIME ** sum(DENSE_CASE)
        ),
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
