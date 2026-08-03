#!/usr/bin/env python3
"""Independent oracle for the bounded stateful gauge-phasor lift.

This source imports neither the production successor nor its M145
predecessor.  It independently reconstructs the public weighted radial
operator in long-double arithmetic, compiles a separate float64 Givens plan,
reexecutes the 51-angle gauge carrier, and checks the sealed package fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


P = 17
TAU = 2.0 * math.pi
PI_LONG_DOUBLE = np.longdouble(
    "3.1415926535897932384626433832795028841971693993751"
)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = (1, *([18] * 16))
ROTATION_COUNT = P * (P - 1) // 2
GAUGE_WEIGHT = 1.0 / 32.0
RESIDUAL_WEIGHT = 1.0 - GAUGE_WEIGHT
SUPPORTED_BASE_MAGNITUDE = 15.0 / 16.0
RESTORATION_TOLERANCE = 3.0e-11
STATE_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 1.0e-10
PLAN_TOLERANCE = 5.0e-14
CHART_RADIUS_TOLERANCE = 2.0e-12
CHART_ZERO_FLOOR = 1.0e-14
CONTROL_FLOOR = 1.0e-5
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
EXPECTED_CLAIM = (
    "BOUNDED_STATEFUL_WEIGHTED_THREE_PHASOR_GAUGE_CHART_LIFTS_LOCAL_"
    "F17_GIVENS_COUPLING_ACROSS_BASE_ZERO_IN_FIXED51_RESIDENT_PHASE_"
    "ANGLES_FOR_THE_DECLARED_MAGNITUDE_ENVELOPE_ACROSS21_CASES_THROUGH_"
    "DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_"
    "ADDS17_GAUGE_CELLS_RETAINS_LOCAL_CARTESIAN_RECHARTING_AND_HAS_THE_"
    "IDENTICAL_SMALLER_COMPLEX_GIVENS_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def wrap(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def shell_scale(shell: int) -> float:
    return math.sqrt(float(SHELL_COUNTS[shell]))


def gauge_seed(shell: int) -> float:
    return wrap(TAU * ((5 * shell + 3) % P) / P + 0.137)


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
        fail("unknown oracle family")
    return values[0] % P or 1, values[1] % P, values[2] % P


def phase_exponent(shell: int, index: int, family: str) -> int:
    quadratic, linear, cubic = gate_parameters(index, family)
    return (
        quadratic * shell**4 + linear * shell**2 + cubic * shell
    ) % P


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_LOCAL_POLAR_GIVENS_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "gate_generator": "PUBLIC_INDEX_BIT_GRAY_TERNARY_WEIGHT_FORMULA",
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if depth < 1 or depth > 4096 or family not in FAMILIES:
        fail("oracle program outside declared scope")
    return Program(
        depth,
        family,
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )


def elimination_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 1):
        for row in range(P - 1, column, -1):
            yield row - 1, row


def reverse_elimination_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 2, -1, -1):
        for row in range(column + 1, P):
            yield row - 1, row


def public_kernel(target: int, source: int) -> np.longdouble:
    value = np.longdouble(1.0 if target == 0 else 0.0)
    for parameter in range(1, 9):
        inverse = pow((4 * parameter) % P, -1, P)
        exponent = (-source * parameter - target * inverse) % P
        angle = (
            np.longdouble(2)
            * PI_LONG_DOUBLE
            * np.longdouble(exponent)
            / np.longdouble(P)
        )
        value -= np.longdouble(2) * np.cos(angle) / np.longdouble(P)
    return value


def dense_weighted_operator() -> np.ndarray:
    operator = np.empty((P, P), dtype=np.longdouble)
    weights = [np.sqrt(np.longdouble(count)) for count in SHELL_COUNTS]
    for target in range(P):
        for source in range(P):
            operator[target, source] = (
                weights[target]
                * public_kernel(target, source)
                / weights[source]
            )
    return operator


@dataclass(frozen=True)
class Plan:
    cosine_sine: np.ndarray
    signs: np.ndarray
    orthogonality_error: float
    off_diagonal_error: float
    diagonal_error: float


def compile_plan(operator: np.ndarray) -> Plan:
    float_operator = np.asarray(operator, dtype=np.float64)
    working = np.array(float_operator, copy=True)
    orthogonality = float(
        np.max(np.abs(float_operator.T @ float_operator - np.eye(P)))
    )
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
            upper_row = np.array(working[upper], copy=True)
            lower_row = np.array(working[lower], copy=True)
            working[upper] = cosine * upper_row + sine * lower_row
            working[lower] = -sine * upper_row + cosine * lower_row
            ordinal += 1
    diagonal = np.diag(working)
    signs = np.where(diagonal >= 0.0, 1.0, -1.0)
    off_diagonal = np.array(working, copy=True)
    np.fill_diagonal(off_diagonal, 0.0)
    off_error = float(np.max(np.abs(off_diagonal)))
    diagonal_error = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
    if max(orthogonality, off_error, diagonal_error) > PLAN_TOLERANCE:
        fail("independent plan exceeded tolerance")
    return Plan(
        coefficients, signs, orthogonality, off_error, diagonal_error
    )


def encode(value: complex, gauge: float) -> tuple[float, float, float]:
    if abs(value) > SUPPORTED_BASE_MAGNITUDE + CHART_RADIUS_TOLERANCE:
        fail("oracle value outside declared envelope")
    residual = (
        value
        - GAUGE_WEIGHT * complex(math.cos(gauge), math.sin(gauge))
    ) / RESIDUAL_WEIGHT
    radius = abs(residual)
    if radius > 1.0 + CHART_RADIUS_TOLERANCE:
        fail("oracle residual outside unit disk")
    if radius <= CHART_ZERO_FLOOR:
        return wrap(gauge), math.pi / 2.0, -math.pi / 2.0
    phase = math.atan2(residual.imag, residual.real)
    delta = math.acos(min(1.0, max(0.0, radius)))
    return wrap(gauge), wrap(phase + delta), wrap(phase - delta)


def decode(row: np.ndarray) -> complex:
    gauge, left, right = map(float, row)
    return (
        GAUGE_WEIGHT * complex(math.cos(gauge), math.sin(gauge))
        + 0.5
        * RESIDUAL_WEIGHT
        * (
            complex(math.cos(left), math.sin(left))
            + complex(math.cos(right), math.sin(right))
        )
    )


def seed_angles() -> np.ndarray:
    state = np.empty((P, 3), dtype=np.float64)
    for shell in range(P):
        state[shell] = encode(
            complex(shell_scale(shell) / P), gauge_seed(shell)
        )
    return state


def phase_error(left: np.ndarray, right: np.ndarray) -> float:
    return max(
        abs(
            complex(math.cos(float(observed)), math.sin(float(observed)))
            - complex(math.cos(float(expected)), math.sin(float(expected)))
        )
        for observed, expected in zip(left.flat, right.flat)
    )


def apply_phase(
    state: np.ndarray, index: int, family: str, *, inverse: bool = False
) -> None:
    direction = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = direction * TAU * phase_exponent(shell, index, family) / P
        for slot in range(3):
            state[shell, slot] = wrap(float(state[shell, slot]) + delta)


def apply_signs(state: np.ndarray, plan: Plan) -> None:
    for shell in range(P):
        if float(plan.signs[shell]) < 0.0:
            for slot in range(3):
                state[shell, slot] = wrap(
                    float(state[shell, slot]) + math.pi
                )


def couple(
    state: np.ndarray,
    upper: int,
    lower: int,
    cosine: float,
    sine: float,
    *,
    transpose: bool,
    gauge_direction_override: float | None = None,
) -> None:
    upper_value = decode(state[upper])
    lower_value = decode(state[lower])
    theta = math.atan2(sine, cosine)
    if transpose:
        next_upper = cosine * upper_value - sine * lower_value
        next_lower = sine * upper_value + cosine * lower_value
        direction = 1.0
    else:
        next_upper = cosine * upper_value + sine * lower_value
        next_lower = -sine * upper_value + cosine * lower_value
        direction = -1.0
    if gauge_direction_override is not None:
        direction = gauge_direction_override
    upper_gauge = wrap(float(state[upper, 0]) + direction * theta)
    lower_gauge = wrap(float(state[lower, 0]) - direction * theta)
    state[upper] = encode(next_upper, upper_gauge)
    state[lower] = encode(next_lower, lower_gauge)


def fourier(
    state: np.ndarray,
    plan: Plan,
    *,
    inverse: bool = False,
    wrong_gauge_direction: bool = False,
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(elimination_pairs()):
            couple(
                state,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=False,
                gauge_direction_override=(
                    1.0 if wrong_gauge_direction else None
                ),
            )
        apply_signs(state, plan)
    else:
        apply_signs(state, plan)
        ordinal = ROTATION_COUNT - 1
        for upper, lower in reverse_elimination_pairs():
            couple(
                state,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=True,
            )
            ordinal -= 1


def project(state: np.ndarray, program: Program) -> complex:
    boundary = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        angle = TAU * exponent / P
        boundary += (
            shell_scale(shell)
            * complex(math.cos(angle), math.sin(angle))
            * decode(state[shell])
        )
    return boundary


def run_gauge(
    program: Program, plan: Plan
) -> tuple[complex, np.ndarray, float]:
    state = seed_angles()
    seed = np.array(state, copy=True)
    for index in range(program.depth):
        apply_phase(state, index, program.family)
        fourier(state, plan)
    boundary = project(state, program)
    final_state = np.asarray([decode(row) for row in state])
    for index in range(program.depth - 1, -1, -1):
        fourier(state, plan, inverse=True)
        apply_phase(state, index, program.family, inverse=True)
    return boundary, final_state, phase_error(state, seed)


def dense_reference(
    program: Program, operator: np.ndarray
) -> tuple[complex, np.ndarray]:
    state = np.asarray(
        [np.clongdouble(shell_scale(shell) / P) for shell in range(P)],
        dtype=np.clongdouble,
    )
    for index in range(program.depth):
        for shell in range(P):
            angle = (
                np.longdouble(2)
                * PI_LONG_DOUBLE
                * np.longdouble(phase_exponent(shell, index, program.family))
                / np.longdouble(P)
            )
            state[shell] *= np.cos(angle) + np.clongdouble(1j) * np.sin(angle)
        state = operator @ state
    boundary = np.clongdouble(0.0j)
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        angle = (
            np.longdouble(2)
            * PI_LONG_DOUBLE
            * np.longdouble(exponent)
            / np.longdouble(P)
        )
        boundary += (
            np.longdouble(shell_scale(shell))
            * (np.cos(angle) + np.clongdouble(1j) * np.sin(angle))
            * state[shell]
        )
    return complex(boundary), np.asarray(state, dtype=np.complex128)


def independent_zero_controls(plan: Plan) -> dict[str, Any]:
    upper, lower = next(elimination_pairs())
    cosine, sine = map(float, plan.cosine_sine[0])
    lower_value = 0.25 * complex(math.cos(0.4), math.sin(0.4))
    legacy_outputs: list[np.ndarray] = []
    legacy_inputs: list[np.ndarray] = []
    repaired_forward: list[np.ndarray] = []
    restored_errors: list[float] = []
    for gauge in (0.2, 1.7):
        pair = np.asarray(
            [wrap(gauge + math.pi / 2), wrap(gauge - math.pi / 2)]
        )
        legacy_inputs.append(pair)
        zero = 0.5 * (
            complex(math.cos(pair[0]), math.sin(pair[0]))
            + complex(math.cos(pair[1]), math.sin(pair[1]))
        )
        output = []
        for value in (
            cosine * zero - sine * lower_value,
            sine * zero + cosine * lower_value,
        ):
            magnitude = abs(value)
            phase = math.atan2(value.imag, value.real)
            delta = math.acos(magnitude)
            output.extend((wrap(phase + delta), wrap(phase - delta)))
        legacy_outputs.append(np.asarray(output))

        state = seed_angles()
        state[upper] = encode(0.0j, gauge)
        seed = np.array(state, copy=True)
        couple(state, upper, lower, cosine, sine, transpose=True)
        repaired_forward.append(np.array(state, copy=True))
        couple(state, upper, lower, cosine, sine, transpose=False)
        restored_errors.append(phase_error(state, seed))

    legacy_input_separation = phase_error(legacy_inputs[0], legacy_inputs[1])
    legacy_output_collision = phase_error(
        legacy_outputs[0], legacy_outputs[1]
    )
    repaired_base_error = max(
        abs(decode(repaired_forward[0][index]) - decode(repaired_forward[1][index]))
        for index in (upper, lower)
    )
    repaired_carrier_separation = phase_error(
        repaired_forward[0], repaired_forward[1]
    )
    return {
        "input_nonzero_coordinate_magnitude": abs(lower_value),
        "output_coordinate_magnitudes": [
            abs(sine * lower_value),
            abs(cosine * lower_value),
        ],
        "zero_input_fiber_is_uncountable": True,
        "two_generic_nonzero_output_fiber_cardinality": 4,
        "fiber_cardinality_mismatch_proves_any_injective_lift_impossible": True,
        "legacy_input_separation": legacy_input_separation,
        "legacy_output_collision": legacy_output_collision,
        "legacy_collision_confirmed": bool(
            legacy_input_separation > CONTROL_FLOOR
            and legacy_output_collision <= 10.0 * np.finfo(np.float64).eps
        ),
        "exact_zero_residual_radius": GAUGE_WEIGHT / RESIDUAL_WEIGHT,
        "repaired_same_base_output_error": repaired_base_error,
        "repaired_carrier_separation": repaired_carrier_separation,
        "repaired_restoration_errors": restored_errors,
        "repaired_both_actual_gauges_restored": all(
            error <= RESTORATION_TOLERANCE for error in restored_errors
        ),
    }


def mutation_controls(plan: Plan) -> dict[str, bool]:
    program = compile_program(3, "ALTERNATE")
    valid_boundary, _, _ = run_gauge(program, plan)
    changed = np.array(plan.cosine_sine, copy=True)
    changed[0, 0] += 0.03125
    changed_plan = Plan(
        changed,
        plan.signs,
        plan.orthogonality_error,
        plan.off_diagonal_error,
        plan.diagonal_error,
    )
    changed_boundary, _, _ = run_gauge(program, changed_plan)

    state = seed_angles()
    seed = np.array(state, copy=True)
    for index in range(program.depth):
        apply_phase(state, index, program.family)
        fourier(state, plan)
    for index in range(program.depth - 1, -1, -1):
        fourier(state, plan, inverse=True, wrong_gauge_direction=True)
        apply_phase(state, index, program.family, inverse=True)
    wrong_gauge_error = phase_error(state, seed)

    envelope_rejected = False
    try:
        encode(complex(SUPPORTED_BASE_MAGNITUDE + 0.01), 0.0)
    except RuntimeError:
        envelope_rejected = True
    return {
        "plan_mutation_changes_boundary": (
            abs(changed_boundary - valid_boundary) > CONTROL_FLOOR
        ),
        "wrong_gauge_inverse_changes_actual_carrier": (
            wrong_gauge_error > CONTROL_FLOOR
        ),
        "out_of_envelope_state_rejected": envelope_rejected,
    }


def run_reuse(plan: Plan) -> tuple[dict[str, Any], dict[str, Any]]:
    state = seed_angles()
    original = np.array(state, copy=True)

    def transaction(program: Program) -> tuple[complex, float]:
        for index in range(program.depth):
            apply_phase(state, index, program.family)
            fourier(state, plan)
        boundary = project(state, program)
        for index in range(program.depth - 1, -1, -1):
            fourier(state, plan, inverse=True)
            apply_phase(state, index, program.family, inverse=True)
        return boundary, phase_error(state, original)

    transaction(compile_program(37, "PRIMARY"))
    reuse_program = compile_program(REUSE_DEPTH, "REUSE")
    restored_boundary, restoration = transaction(reuse_program)
    fresh_boundary, _, _ = run_gauge(reuse_program, plan)
    reuse = {
        "fresh_restored_boundary_error": abs(
            restored_boundary - fresh_boundary
        ),
        "restoration_error": restoration,
    }

    repeated_state = seed_angles()
    repeated_seed = np.array(repeated_state, copy=True)
    repeated_program = compile_program(REPEATED_REUSE_DEPTH, "ALTERNATE")
    maximum = 0.0
    for _ in range(REPEATED_REUSE_CYCLES):
        for index in range(repeated_program.depth):
            apply_phase(repeated_state, index, repeated_program.family)
            fourier(repeated_state, plan)
        for index in range(repeated_program.depth - 1, -1, -1):
            fourier(repeated_state, plan, inverse=True)
            apply_phase(
                repeated_state, index, repeated_program.family, inverse=True
            )
        maximum = max(maximum, phase_error(repeated_state, repeated_seed))
    repeated = {"maximum_restoration_error": maximum}
    return reuse, repeated


def verify(
    package_path: Path, production_path: Path, predecessor_path: Path
) -> dict[str, Any]:
    package = json.loads(package_path.read_text())
    if package.get("claim") != EXPECTED_CLAIM:
        fail("production claim mismatch")
    if hashlib.sha256(production_path.read_bytes()).hexdigest() != package.get(
        "source_sha256"
    ):
        fail("production source hash mismatch")
    if hashlib.sha256(predecessor_path.read_bytes()).hexdigest() != package.get(
        "production_dependency", {}
    ).get("sha256"):
        fail("production dependency hash mismatch")

    operator = dense_weighted_operator()
    identity = np.eye(P, dtype=np.longdouble)
    symmetry_error = float(np.max(np.abs(operator - operator.T)))
    orthogonality_error = float(
        np.max(np.abs(operator.T @ operator - identity))
    )
    involution_error = float(np.max(np.abs(operator @ operator - identity)))
    if max(symmetry_error, orthogonality_error, involution_error) > 2.0e-15:
        fail("independent long-double operator check failed")
    plan = compile_plan(operator)

    case_index = {
        (case["family"], case["depth"]): case for case in package["cases"]
    }
    comparisons = 0
    maximum_production_boundary_error = 0.0
    maximum_dense_boundary_error = 0.0
    maximum_dense_state_error = 0.0
    maximum_restoration_error = 0.0
    for family in FAMILIES:
        for depth in DEPTHS:
            program = compile_program(depth, family)
            production_case = case_index[(family, depth)]
            if production_case["program_fingerprint"] != program.fingerprint():
                fail("independent program fingerprint mismatch")
            boundary, state, restoration = run_gauge(program, plan)
            dense_boundary, dense_state = dense_reference(program, operator)
            production_boundary = complex(*production_case["final_boundary"])
            production_matrix_free_boundary = complex(
                *production_case["matched_matrix_free_complex_boundary"]
            )
            production_streamed_boundary = complex(
                *production_case[
                    "matched_streamed_real_kernel_complex_boundary"
                ]
            )
            production_error = abs(boundary - production_boundary)
            dense_boundary_error = abs(boundary - dense_boundary)
            dense_state_error = float(np.max(np.abs(state - dense_state)))
            maximum_production_boundary_error = max(
                maximum_production_boundary_error, production_error
            )
            maximum_dense_boundary_error = max(
                maximum_dense_boundary_error, dense_boundary_error
            )
            maximum_dense_state_error = max(
                maximum_dense_state_error, dense_state_error
            )
            maximum_restoration_error = max(
                maximum_restoration_error, restoration
            )
            if (
                production_error > BOUNDARY_TOLERANCE
                or dense_boundary_error > BOUNDARY_TOLERANCE
                or dense_state_error > STATE_TOLERANCE
                or restoration > RESTORATION_TOLERANCE
                or abs(production_matrix_free_boundary - dense_boundary)
                > BOUNDARY_TOLERANCE
                or abs(production_streamed_boundary - dense_boundary)
                > BOUNDARY_TOLERANCE
                or production_case[
                    "boundary_error_against_matrix_free_complex"
                ]
                > BOUNDARY_TOLERANCE
                or production_case[
                    "boundary_error_against_streamed_real_kernel_complex"
                ]
                > BOUNDARY_TOLERANCE
            ):
                fail("independent case reexecution exceeded tolerance")
            comparisons += 9

    zero = independent_zero_controls(plan)
    if not (
        zero["legacy_collision_confirmed"]
        and zero["repaired_both_actual_gauges_restored"]
        and zero["repaired_same_base_output_error"] <= STATE_TOLERANCE
        and zero["repaired_carrier_separation"] > CONTROL_FLOOR
    ):
        fail("independent zero-fiber control failed")
    production_zero = package["zero_fiber_obstruction"]
    if not (
        production_zero[
            "fiber_cardinality_mismatch_proves_any_injective_lift_impossible"
        ]
        and production_zero["generic_nonzero_ordered_pair_fiber_cardinality"]
        == 2
        and production_zero["two_generic_nonzero_output_fiber_cardinality"]
        == 4
        and abs(
            production_zero["canonical_output_phase_cell_collision"]
            - zero["legacy_output_collision"]
        )
        <= 10.0 * np.finfo(np.float64).eps
    ):
        fail("production zero-fiber certificate mismatch")
    comparisons += 7

    mutations = mutation_controls(plan)
    if not all(mutations.values()):
        fail("independent mutation control failed")
    comparisons += len(mutations)

    reuse, repeated = run_reuse(plan)
    if (
        reuse["fresh_restored_boundary_error"] > BOUNDARY_TOLERANCE
        or reuse["restoration_error"] > RESTORATION_TOLERANCE
        or repeated["maximum_restoration_error"] > RESTORATION_TOLERANCE
    ):
        fail("independent reuse control failed")
    if abs(
        reuse["fresh_restored_boundary_error"]
        - package["reuse"]["fresh_restored_boundary_error"]
    ) > BOUNDARY_TOLERANCE:
        fail("production reuse field mismatch")
    comparisons += 4

    expected_resources = {
        "resident_phase_angle_float64_cells": 51,
        "resident_phase_angle_bytes": 408,
        "added_gauge_phase_angle_cells_against_m145": 17,
        "retained_public_givens_plan_float64_cells": 289,
        "retained_public_givens_plan_bytes": 2312,
        "retained_complex_state_cells": 0,
        "retained_dense_kernel_cells": 0,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "maximum_named_commitment_live_bytes_including_program_json": 3191,
        "commitment_input_copy_bytes": 0,
        "commitment_public_hexdigest_bytes": 64,
        "commitment_logical_sha256_state_and_block_bytes": 96,
    }
    for key, expected in expected_resources.items():
        if package["resource_law"].get(key) != expected:
            fail(f"production resource field mismatch: {key}")
        comparisons += 1
    if not all(package["controls"].values()):
        fail("production controls are not all true")
    frontiers = package["matched_classical_frontiers"]
    if not (
        frontiers["identical_local_plan"][
            "maximum_named_warm_execution_live_bytes_including_program_json"
        ]
        == 3055
        and frontiers["work_frontier"][
            "maximum_named_warm_execution_live_bytes_including_program_json"
        ]
        == 1767
        and frontiers["memory_frontier"][
            "maximum_named_warm_execution_live_bytes_including_program_json"
        ]
        == 983
        and frontiers["identical_local_plan"][
            "phase_module_cosine_evaluations_per_step"
        ]
        == 17
        and frontiers["memory_frontier"][
            "kernel_cosine_evaluations_per_fourier"
        ]
        == 2312
        and not frontiers["comparison_establishes_distinct_phase_resource"]
        and not frontiers["comparison_establishes_computational_advantage"]
    ):
        fail("production matched classical frontier mismatch")
    if package["restoration"].get("class") != (
        "NUMERICAL_PHYSICAL_STATE_RESTORATION"
    ):
        fail("production restoration classification mismatch")
    comparisons += len(package["controls"]) + 8

    return {
        "schema": "CAT_CAS_F17_STATEFUL_GAUGE_PHASOR_LIFT_ORACLE_V1",
        "result": "PASS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "claim": EXPECTED_CLAIM,
        "production_source_sha256": hashlib.sha256(
            production_path.read_bytes()
        ).hexdigest(),
        "production_result_sha256": hashlib.sha256(
            package_path.read_bytes()
        ).hexdigest(),
        "independence": {
            "imports_production_module": False,
            "imports_predecessor_module": False,
            "long_double_dense_operator_reconstruction": True,
            "separate_float64_qr_compilation": True,
            "separate51_angle_forward_inverse": True,
            "separate_zero_fiber_collision_and_repair": True,
            "separate_reuse_reexecution": True,
        },
        "operator_checks": {
            "symmetry_error": symmetry_error,
            "orthogonality_error": orthogonality_error,
            "involution_error": involution_error,
            "float64_qr_orthogonality_error": plan.orthogonality_error,
            "float64_qr_off_diagonal_error": plan.off_diagonal_error,
            "float64_qr_diagonal_error": plan.diagonal_error,
        },
        "case_checks": {
            "case_count": len(case_index),
            "maximum_boundary_error_against_production": (
                maximum_production_boundary_error
            ),
            "maximum_boundary_error_against_long_double_dense": (
                maximum_dense_boundary_error
            ),
            "maximum_state_error_against_long_double_dense": (
                maximum_dense_state_error
            ),
            "maximum_restoration_error": maximum_restoration_error,
        },
        "zero_fiber_checks": zero,
        "mutation_controls": mutations,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "comparison_count": comparisons,
        "restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "claim_ceiling": (
            "21_DECLARED_WEIGHTED_GAUGE_PHASOR_CASES_THROUGH_DEPTH4096_"
            "PLUS_DECLARED_ZERO_AND_REUSE_CONTROLS_IN_LINUX_DIRECT_PROCESS_"
            "SOFTWARE"
        ),
        "not_established": [
            "FULL_WEIGHTED_UNIT_SPHERE_GLOBAL_CHART",
            "LOCAL_CARTESIAN_REGISTER_FREE_COUPLING",
            "EXACT_ALGEBRAIC_SEMANTICS",
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
    parser.add_argument("--production", required=True)
    parser.add_argument("--predecessor", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = verify(
        Path(arguments.package),
        Path(arguments.production),
        Path(arguments.predecessor),
    )
    Path(arguments.output).write_bytes(canonical_json(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
