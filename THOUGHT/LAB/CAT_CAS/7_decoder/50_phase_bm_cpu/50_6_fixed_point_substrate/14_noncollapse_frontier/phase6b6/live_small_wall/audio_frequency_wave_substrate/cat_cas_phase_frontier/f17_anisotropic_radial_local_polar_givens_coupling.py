#!/usr/bin/env python3
"""Factor the weighted F17 radial phase Fourier into local polar couplers.

The accepted carrier stores two unit-phasor angles for each of 17
shell-normalized radial coefficients.  Public shell multiplicities conjugate
the radial Fourier into a real symmetric orthogonal matrix.  A deterministic
QR schedule compiles that matrix into 136 adjacent-row Givens rotations and
17 terminal signs.  Forward and inverse execution then touch only two phase
cells at a time; no full complex state or dense kernel is retained.

This is a bounded local phase-law result, not an advantage claim.  Every
coupler still uses local Cartesian registers and canonical polar charting, and
the identical compiled complex-Givens recurrence is smaller and uses far less
trigonometric work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Predecessor use is restricted to a non-accepted matrix-free verification
# baseline.  The carrier, public plan, local couplers, inverse, and matched
# complex-Givens execution below are implemented in this source.
import f17_anisotropic_radial_native_angle_interference as m144


P = 17
TAU = 2.0 * math.pi
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = (1, *([18] * 16))
ROTATION_COUNT = P * (P - 1) // 2
RESTORATION_TOLERANCE = 2.0e-11
STATE_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 1.0e-10
PLAN_TOLERANCE = 5.0e-14
CHART_ZERO_CANONICAL_FLOOR = 1.0e-14
CHART_RADIUS_TOLERANCE = 2.0e-12
CONTROL_FLOOR = 1.0e-5
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
CLAIM = (
    "BOUNDED_TOPOLOGY_COMPILED_LOCAL_POLAR_GIVENS_PHASE_COUPLING_"
    "FACTORS_WEIGHTED_ANISOTROPIC_F17_RADIAL_INTERFERENCE_IN136_"
    "TWO_CELL_COUPLERS_WITH_FIXED34_RESIDENT_PHASE_ANGLES_ACROSS21_"
    "CASES_THROUGH_DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_"
    "AND_REUSE_BUT_REQUIRES_A2312_BYTE_PUBLIC_PLAN_LOCAL_CARTESIAN_"
    "REGISTERS_AND_HAS_THE_IDENTICAL_COMPLEX_GIVENS_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def wrap_scalar(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def phasor_pair_value(left: float, right: float) -> tuple[float, float]:
    return (
        0.5 * (math.cos(left) + math.cos(right)),
        0.5 * (math.sin(left) + math.sin(right)),
    )


def chart_pair(
    real: float, imag: float
) -> tuple[float, float, float, bool]:
    magnitude = math.hypot(real, imag)
    if not math.isfinite(magnitude) or magnitude > 1.0 + CHART_RADIUS_TOLERANCE:
        fail("local polar chart left its declared unit disk")
    if magnitude <= CHART_ZERO_CANONICAL_FLOOR:
        return math.pi / 2.0, -math.pi / 2.0, magnitude, True
    phase = math.atan2(imag, real)
    delta = math.acos(min(1.0, max(0.0, magnitude)))
    return (
        wrap_scalar(phase + delta),
        wrap_scalar(phase - delta),
        magnitude,
        False,
    )


def shell_scale(shell: int) -> float:
    return math.sqrt(float(SHELL_COUNTS[shell]))


def seed_angles() -> np.ndarray:
    angles = np.empty((P, 2), dtype=np.float64)
    for shell in range(P):
        left, right, _, _ = chart_pair(shell_scale(shell) / P, 0.0)
        angles[shell, 0] = left
        angles[shell, 1] = right
    return angles


def seed_state() -> np.ndarray:
    return np.asarray(
        [shell_scale(shell) / P for shell in range(P)],
        dtype=np.complex128,
    )


def physical_angle_error(left: np.ndarray, right: np.ndarray) -> float:
    maximum = 0.0
    for shell in range(P):
        for slot in range(2):
            observed = float(left[shell, slot])
            expected = float(right[shell, slot])
            maximum = max(
                maximum,
                math.hypot(
                    math.cos(observed) - math.cos(expected),
                    math.sin(observed) - math.sin(expected),
                ),
            )
    return maximum


def physical_error_from_seed(angles: np.ndarray) -> float:
    maximum = 0.0
    for shell in range(P):
        expected_left, expected_right, _, _ = chart_pair(
            shell_scale(shell) / P, 0.0
        )
        for slot, expected in ((0, expected_left), (1, expected_right)):
            observed = float(angles[shell, slot])
            maximum = max(
                maximum,
                math.hypot(
                    math.cos(observed) - math.cos(expected),
                    math.sin(observed) - math.sin(expected),
                ),
            )
    return maximum


def angles_state(angles: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            complex(
                *phasor_pair_value(
                    float(angles[shell, 0]), float(angles[shell, 1])
                )
            )
            for shell in range(P)
        ],
        dtype=np.complex128,
    )


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
        fail("unknown local-polar public family")
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
    if not isinstance(depth, int) or not 1 <= depth <= 4096:
        fail("local-polar program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("local-polar program family outside declared set")
    program = Program(
        depth,
        family,
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )
    gate_parameters(0, family)
    gate_parameters(depth - 1, family)
    return program


def elimination_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 1):
        for row in range(P - 1, column, -1):
            yield row - 1, row


def reverse_elimination_pairs() -> Iterator[tuple[int, int]]:
    for column in range(P - 2, -1, -1):
        for row in range(column + 1, P):
            yield row - 1, row


def public_real_kernel(
    target: int, source: int, inverse_half: tuple[int, ...]
) -> float:
    value = 1.0 if target == 0 else 0.0
    for offset, inverse in enumerate(inverse_half):
        parameter = offset + 1
        exponent = (-source * parameter - target * inverse) % P
        value -= (2.0 / P) * math.cos(TAU * exponent / P)
    return value


@dataclass(frozen=True)
class GivensPlan:
    cosine_sine: np.ndarray
    diagonal_signs: np.ndarray
    source_orthogonality_error: float
    triangular_off_diagonal_error: float
    triangular_diagonal_magnitude_error: float
    compilation_maximum_named_bytes: int

    @classmethod
    def compile(cls) -> "GivensPlan":
        inverse_half = tuple(
            pow((4 * parameter) % P, -1, P) for parameter in range(1, 9)
        )
        weights = np.asarray(
            [shell_scale(shell) for shell in range(P)], dtype=np.float64
        )
        working = np.empty((P, P), dtype=np.float64)
        for target in range(P):
            for source in range(P):
                working[target, source] = (
                    weights[target]
                    * public_real_kernel(target, source, inverse_half)
                    / weights[source]
                )
        del inverse_half, weights
        orthogonality_error = 0.0
        for left in range(P):
            for right in range(P):
                product = 0.0
                for row in range(P):
                    product += working[row, left] * working[row, right]
                orthogonality_error = max(
                    orthogonality_error,
                    abs(product - (1.0 if left == right else 0.0)),
                )
        cosine_sine = np.empty((ROTATION_COUNT, 2), dtype=np.float64)
        ordinal = 0
        for column in range(P - 1):
            for row in range(P - 1, column, -1):
                upper = row - 1
                lower = row
                upper_value = float(working[upper, column])
                lower_value = float(working[lower, column])
                radius = math.hypot(upper_value, lower_value)
                if radius == 0.0:
                    fail("public Givens compilation encountered zero pivot")
                cosine = upper_value / radius
                sine = lower_value / radius
                cosine_sine[ordinal, 0] = cosine
                cosine_sine[ordinal, 1] = sine
                upper_row = working[upper].copy()
                lower_row = working[lower].copy()
                working[upper] = cosine * upper_row + sine * lower_row
                working[lower] = -sine * upper_row + cosine * lower_row
                ordinal += 1
        if ordinal != ROTATION_COUNT:
            fail("public Givens compilation produced the wrong plan length")
        del upper_row, lower_row
        diagonal = np.diag(working)
        signs = np.where(diagonal >= 0.0, 1.0, -1.0).astype(np.float64)
        off_error = 0.0
        for row in range(P):
            for column in range(P):
                if row != column:
                    off_error = max(off_error, abs(float(working[row, column])))
        diagonal_error = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
        if (
            orthogonality_error > PLAN_TOLERANCE
            or off_error > PLAN_TOLERANCE
            or diagonal_error > PLAN_TOLERANCE
        ):
            fail("public Givens plan exceeded predeclared tolerance")
        cosine_sine.setflags(write=False)
        signs.setflags(write=False)
        # Peak logical compilation state: 289 working floats, 272 plan
        # floats, 17 shell-scale floats, 34 row-copy floats, and eight
        # inverse-parameter int64 values.  Python/NumPy internals are excluded.
        compilation_bytes = (289 + 272 + 17 + 34) * 8 + 8 * 8
        return cls(
            cosine_sine,
            signs,
            orthogonality_error,
            off_error,
            diagonal_error,
            compilation_bytes,
        )


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    streamed_gate_rematerializations: int = 0
    local_coupler_calls: int = 0
    sign_updates: int = 0
    input_phasor_cosine_evaluations: int = 0
    input_phasor_sine_evaluations: int = 0
    chart_hypot_evaluations: int = 0
    chart_atan2_evaluations: int = 0
    chart_acos_evaluations: int = 0
    zero_canonicalizations: int = 0
    projection_phasor_cosine_evaluations: int = 0
    projection_phasor_sine_evaluations: int = 0
    maximum_named_update_float64_cells: int = 0
    maximum_named_update_bytes: int = 0
    maximum_named_projection_float64_cells: int = 0
    maximum_named_projection_bytes: int = 0
    minimum_nonzero_chart_magnitude: float = math.inf
    maximum_chart_magnitude: float = 0.0

    def observe_update(self, cells: int) -> None:
        self.maximum_named_update_float64_cells = max(
            self.maximum_named_update_float64_cells, cells
        )
        self.maximum_named_update_bytes = max(
            self.maximum_named_update_bytes, cells * 8
        )

    def observe_projection(self, cells: int) -> None:
        self.maximum_named_projection_float64_cells = max(
            self.maximum_named_projection_float64_cells, cells
        )
        self.maximum_named_projection_bytes = max(
            self.maximum_named_projection_bytes, cells * 8
        )

    def observe_chart(self, magnitude: float, canonical_zero: bool) -> None:
        self.chart_hypot_evaluations += 1
        if canonical_zero:
            self.zero_canonicalizations += 1
        else:
            self.chart_atan2_evaluations += 1
            self.chart_acos_evaluations += 1
            self.minimum_nonzero_chart_magnitude = min(
                self.minimum_nonzero_chart_magnitude, magnitude
            )
        self.maximum_chart_magnitude = max(
            self.maximum_chart_magnitude, magnitude
        )


@dataclass
class Carrier:
    angles: np.ndarray
    stage: str = "RESTORED"
    active_program: str | None = None
    forward_index: int = 0
    inverse_index: int = 0
    projection_calls: int = 0
    restoration_generation: int = 0
    stats: Stats = field(default_factory=Stats)

    @classmethod
    def create(cls) -> "Carrier":
        return cls(seed_angles())

    def backing_identity(self) -> tuple[int, int]:
        return id(self), int(self.angles.__array_interface__["data"][0])

    def restored_error(self) -> float:
        return physical_error_from_seed(self.angles)


def apply_phase(
    carrier: Carrier, index: int, family: str, *, inverse: bool = False
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * phase_exponent(shell, index, family) / P
        carrier.angles[shell, 0] = wrap_scalar(
            float(carrier.angles[shell, 0]) + delta
        )
        carrier.angles[shell, 1] = wrap_scalar(
            float(carrier.angles[shell, 1]) + delta
        )
    carrier.stats.streamed_gate_rematerializations += 1
    carrier.stats.observe_update(4)


def apply_signs(carrier: Carrier, plan: GivensPlan) -> None:
    for shell in range(P):
        if float(plan.diagonal_signs[shell]) < 0.0:
            carrier.angles[shell, 0] = wrap_scalar(
                float(carrier.angles[shell, 0]) + math.pi
            )
            carrier.angles[shell, 1] = wrap_scalar(
                float(carrier.angles[shell, 1]) + math.pi
            )
            carrier.stats.sign_updates += 1


def apply_local_coupler(
    carrier: Carrier,
    upper: int,
    lower: int,
    cosine: float,
    sine: float,
    *,
    transpose: bool,
) -> None:
    upper_real, upper_imag = phasor_pair_value(
        float(carrier.angles[upper, 0]),
        float(carrier.angles[upper, 1]),
    )
    lower_real, lower_imag = phasor_pair_value(
        float(carrier.angles[lower, 0]),
        float(carrier.angles[lower, 1]),
    )
    carrier.stats.input_phasor_cosine_evaluations += 4
    carrier.stats.input_phasor_sine_evaluations += 4
    if transpose:
        next_upper_real = cosine * upper_real - sine * lower_real
        next_upper_imag = cosine * upper_imag - sine * lower_imag
        next_lower_real = sine * upper_real + cosine * lower_real
        next_lower_imag = sine * upper_imag + cosine * lower_imag
    else:
        next_upper_real = cosine * upper_real + sine * lower_real
        next_upper_imag = cosine * upper_imag + sine * lower_imag
        next_lower_real = -sine * upper_real + cosine * lower_real
        next_lower_imag = -sine * upper_imag + cosine * lower_imag
    upper_left, upper_right, upper_magnitude, upper_zero = chart_pair(
        next_upper_real, next_upper_imag
    )
    lower_left, lower_right, lower_magnitude, lower_zero = chart_pair(
        next_lower_real, next_lower_imag
    )
    carrier.angles[upper, 0] = upper_left
    carrier.angles[upper, 1] = upper_right
    carrier.angles[lower, 0] = lower_left
    carrier.angles[lower, 1] = lower_right
    carrier.stats.observe_chart(upper_magnitude, upper_zero)
    carrier.stats.observe_chart(lower_magnitude, lower_zero)
    carrier.stats.local_coupler_calls += 1
    # Four decoded Cartesian registers, four mixed registers, and a
    # conservative four-register chart scratch; carrier inputs/outputs alias.
    carrier.stats.observe_update(12)


def apply_givens_fourier(
    carrier: Carrier, plan: GivensPlan, *, inverse: bool = False
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(elimination_pairs()):
            apply_local_coupler(
                carrier,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=False,
            )
        apply_signs(carrier, plan)
    else:
        apply_signs(carrier, plan)
        ordinal = ROTATION_COUNT - 1
        for upper, lower in reverse_elimination_pairs():
            apply_local_coupler(
                carrier,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=True,
            )
            ordinal -= 1
        if ordinal != -1:
            fail("reverse public Givens schedule did not consume the plan")


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong local-polar carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("local-polar carrier owner or stage changed")


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null local-polar carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or carrier.restored_error() > RESTORATION_TOLERANCE
    ):
        fail("local-polar carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def forward(carrier: Carrier, program: Program, plan: GivensPlan) -> None:
    require_owned(carrier, program, "FORWARD")
    for index in range(program.depth):
        apply_phase(carrier, index, program.family)
        apply_givens_fourier(carrier, plan)
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_LOCAL_POLAR_STATE_RESIDENT"


def project(carrier: Carrier, program: Program) -> complex:
    require_owned(carrier, program, "FINAL_LOCAL_POLAR_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("local-polar final projection order changed")
    boundary_real = 0.0
    boundary_imag = 0.0
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        observation = TAU * exponent / P
        left = observation + float(carrier.angles[shell, 0])
        right = observation + float(carrier.angles[shell, 1])
        weight = 0.5 * shell_scale(shell)
        boundary_real += weight * (math.cos(left) + math.cos(right))
        boundary_imag += weight * (math.sin(left) + math.sin(right))
        carrier.stats.projection_phasor_cosine_evaluations += 2
        carrier.stats.projection_phasor_sine_evaluations += 2
    carrier.stats.observe_projection(10)
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return complex(boundary_real, boundary_imag)


def inverse(carrier: Carrier, program: Program, plan: GivensPlan) -> float:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("local-polar inverse requires one final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_givens_fourier(carrier, plan, inverse=True)
        apply_phase(carrier, index, program.family, inverse=True)
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("local-polar actual inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(memoryview(carrier.angles).cast("B")).hexdigest()


def complex_givens_fourier(
    state: np.ndarray, plan: GivensPlan, *, inverse: bool = False
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(elimination_pairs()):
            cosine = float(plan.cosine_sine[ordinal, 0])
            sine = float(plan.cosine_sine[ordinal, 1])
            upper_value = complex(state[upper])
            lower_value = complex(state[lower])
            state[upper] = cosine * upper_value + sine * lower_value
            state[lower] = -sine * upper_value + cosine * lower_value
        state *= plan.diagonal_signs
    else:
        state *= plan.diagonal_signs
        ordinal = ROTATION_COUNT - 1
        for upper, lower in reverse_elimination_pairs():
            cosine = float(plan.cosine_sine[ordinal, 0])
            sine = float(plan.cosine_sine[ordinal, 1])
            upper_value = complex(state[upper])
            lower_value = complex(state[lower])
            state[upper] = cosine * upper_value - sine * lower_value
            state[lower] = sine * upper_value + cosine * lower_value
            ordinal -= 1


def complex_givens_boundary(
    program: Program, plan: GivensPlan
) -> tuple[complex, np.ndarray]:
    state = seed_state()
    for index in range(program.depth):
        for shell in range(P):
            angle = TAU * phase_exponent(shell, index, program.family) / P
            state[shell] *= complex(math.cos(angle), math.sin(angle))
        complex_givens_fourier(state, plan)
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
            * state[shell]
        )
    return complex(boundary), state


def matrix_free_boundary(program: Program) -> tuple[complex, np.ndarray]:
    predecessor_program = m144.compile_program(program.depth, program.family)
    boundary, unweighted_state = m144.classical_boundary(predecessor_program)
    weighted_state = np.asarray(
        [
            shell_scale(shell) * unweighted_state[shell]
            for shell in range(P)
        ],
        dtype=np.complex128,
    )
    return boundary, weighted_state


def execute_case(plan: GivensPlan, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    commitment = state_commitment(carrier)
    boundary = project(carrier, program)
    matched_boundary, matched_state = complex_givens_boundary(program, plan)
    reference_boundary, reference_state = matrix_free_boundary(program)
    observed_state = angles_state(carrier.angles)
    state_error = float(np.max(np.abs(observed_state - matched_state)))
    boundary_error = abs(boundary - matched_boundary)
    frontier_boundary_error = abs(matched_boundary - reference_boundary)
    frontier_state_error = float(
        np.max(np.abs(matched_state - reference_state))
    )
    if state_error > STATE_TOLERANCE or boundary_error > BOUNDARY_TOLERANCE:
        fail("local-polar execution exceeded predeclared tolerance")
    if (
        frontier_boundary_error > BOUNDARY_TOLERANCE
        or frontier_state_error > STATE_TOLERANCE
    ):
        fail("compiled complex-Givens frontier disagreed with matrix-free path")
    restoration_error = inverse(carrier, program, plan)
    if carrier.backing_identity() != backing:
        fail("local-polar carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_complex_givens_boundary": complex_pair(matched_boundary),
        "matrix_free_frontier_boundary": complex_pair(reference_boundary),
        "maximum_state_error_against_matched_complex_givens": state_error,
        "boundary_error_against_matched_complex_givens": boundary_error,
        "matched_frontier_boundary_error": frontier_boundary_error,
        "matched_frontier_state_error": frontier_state_error,
        "restoration_error": restoration_error,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "resident_restoration_class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
        "transient_buffer_restoration_class": "NO_RESTORATION_CLAIM",
        "resident_phase_angle_cells": int(carrier.angles.size),
        "resident_phase_angle_bytes": int(carrier.angles.nbytes),
        "stats": vars(carrier.stats),
    }


def run_transaction(
    carrier: Carrier, program: Program, plan: GivensPlan
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    boundary = project(carrier, program)
    restoration_error = inverse(carrier, program, plan)
    return boundary, restoration_error


def reuse_control(plan: GivensPlan) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    first = compile_program(37, "PRIMARY")
    second = compile_program(REUSE_DEPTH, "REUSE")
    run_transaction(carrier, first, plan)
    restored_boundary, restoration_error = run_transaction(
        carrier, second, plan
    )
    fresh_boundary, _ = run_transaction(Carrier.create(), second, plan)
    error = abs(restored_boundary - fresh_boundary)
    if error > BOUNDARY_TOLERANCE:
        fail("local-polar unrelated reuse disagreed with fresh execution")
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "fresh_restored_boundary_error": error,
        "restoration_error": restoration_error,
        "same_original_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control(plan: GivensPlan) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    program = compile_program(REPEATED_REUSE_DEPTH, "ALTERNATE")
    maximum = 0.0
    for _ in range(REPEATED_REUSE_CYCLES):
        _, error = run_transaction(carrier, program, plan)
        maximum = max(maximum, error)
    return {
        "cycles": REPEATED_REUSE_CYCLES,
        "depth_per_cycle": REPEATED_REUSE_DEPTH,
        "maximum_restoration_error": maximum,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def raw_reverse(
    carrier: Carrier,
    program: Program,
    plan: GivensPlan,
    *,
    omit_fourier: bool = False,
    wrong_plan: GivensPlan | None = None,
    reordered: bool = False,
) -> None:
    active_plan = wrong_plan if wrong_plan is not None else plan
    for index in range(program.depth - 1, -1, -1):
        if reordered:
            apply_phase(carrier, index, program.family, inverse=True)
        if not omit_fourier:
            apply_givens_fourier(carrier, active_plan, inverse=True)
        if not reordered:
            apply_phase(carrier, index, program.family, inverse=True)


def mutated_plan(plan: GivensPlan) -> GivensPlan:
    coefficients = np.array(plan.cosine_sine, copy=True)
    coefficients[0, 0] += 0.03125
    coefficients.setflags(write=False)
    return GivensPlan(
        coefficients,
        plan.diagonal_signs,
        plan.source_orthogonality_error,
        plan.triangular_off_diagonal_error,
        plan.triangular_diagonal_magnitude_error,
        plan.compilation_maximum_named_bytes,
    )


def controls(plan: GivensPlan) -> dict[str, bool]:
    program = compile_program(3, "ALTERNATE")
    valid = Carrier.create()
    begin_forward(valid, program)
    forward(valid, program, plan)
    valid_boundary = project(valid, program)

    changed_plan = mutated_plan(plan)
    mutated = Carrier.create()
    begin_forward(mutated, program)
    forward(mutated, program, changed_plan)
    mutated_boundary = project(mutated, program)

    missing = Carrier.create()
    begin_forward(missing, program)
    forward(missing, program, plan)
    raw_reverse(missing, program, plan, omit_fourier=True)

    wrong = Carrier.create()
    begin_forward(wrong, program)
    forward(wrong, program, plan)
    raw_reverse(wrong, program, plan, wrong_plan=changed_plan)

    reordered_carrier = Carrier.create()
    begin_forward(reordered_carrier, program)
    forward(reordered_carrier, program, plan)
    raw_reverse(reordered_carrier, program, plan, reordered=True)

    phase_disabled = Carrier.create()
    begin_forward(phase_disabled, program)
    for _ in range(program.depth):
        apply_givens_fourier(phase_disabled, plan)
    phase_disabled.forward_index = program.depth
    phase_disabled.stage = "FINAL_LOCAL_POLAR_STATE_RESIDENT"
    phase_disabled_boundary = project(phase_disabled, program)

    premature_rejected = False
    try:
        premature = Carrier.create()
        begin_forward(premature, program)
        project(premature, program)
    except RuntimeError:
        premature_rejected = True

    null_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    return {
        "plan_coefficient_mutation_changes_boundary": (
            abs(mutated_boundary - valid_boundary) > CONTROL_FLOOR
        ),
        "missing_inverse_changes_state": (
            missing.restored_error() > CONTROL_FLOOR
        ),
        "wrong_inverse_changes_state": wrong.restored_error() > CONTROL_FLOOR,
        "reordered_inverse_changes_state": (
            reordered_carrier.restored_error() > CONTROL_FLOOR
        ),
        "phase_disabled_changes_boundary": (
            abs(phase_disabled_boundary - valid_boundary) > CONTROL_FLOOR
        ),
        "premature_projection_rejected": premature_rejected,
        "null_carrier_rejected": null_rejected,
    }


def run() -> dict[str, Any]:
    plan = GivensPlan.compile()
    cases = [
        execute_case(plan, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_within = all(
        case["maximum_state_error_against_matched_complex_givens"]
        <= STATE_TOLERANCE
        and case["boundary_error_against_matched_complex_givens"]
        <= BOUNDARY_TOLERANCE
        and case["matched_frontier_boundary_error"] <= BOUNDARY_TOLERANCE
        and case["matched_frontier_state_error"] <= STATE_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_before"] == 0
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        for case in cases
    )
    if not all_within:
        fail("one or more local-polar cases failed the declared envelope")
    control_results = controls(plan)
    if not all(control_results.values()):
        fail("one or more local-polar controls failed")
    reuse = reuse_control(plan)
    repeated = repeated_reuse_control(plan)
    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_update_bytes = max(
        case["stats"]["maximum_named_update_bytes"] for case in cases
    )
    maximum_projection_bytes = max(
        case["stats"]["maximum_named_projection_bytes"] for case in cases
    )
    resident_bytes = 34 * 8
    plan_bytes = (272 + 17) * 8
    shell_count_bytes = 17 * 8
    native_update_live = (
        resident_bytes
        + plan_bytes
        + shell_count_bytes
        + maximum_update_bytes
        + maximum_program_bytes
    )
    native_commitment_live = (
        resident_bytes
        + plan_bytes
        + shell_count_bytes
        + 96
        + 64
        + maximum_program_bytes
    )
    native_projection_live = (
        resident_bytes
        + plan_bytes
        + shell_count_bytes
        + maximum_projection_bytes
        + maximum_program_bytes
    )
    native_maximum_live = max(
        native_update_live, native_commitment_live, native_projection_live
    )
    # Identical plan and state plus four complex local scratch values.  Phase
    # roots are rematerialized from one angle and two scalar trig registers.
    matched_complex_live = (
        resident_bytes + plan_bytes + shell_count_bytes + 64 + maximum_program_bytes
    )
    return {
        "schema": "CAT_CAS_F17_LOCAL_POLAR_GIVENS_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_PHASE_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_within_predeclared_tolerances": all_within,
            "public_topology_compilation_reads_final_answers": False,
        },
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "state_max_abs": STATE_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
            "plan_max_abs": PLAN_TOLERANCE,
            "chart_zero_canonical_floor": CHART_ZERO_CANONICAL_FLOOR,
            "chart_radius_slack": CHART_RADIUS_TOLERANCE,
        },
        "plan": {
            "rotation_count": ROTATION_COUNT,
            "implicit_index_schedule": True,
            "stored_index_cells": 0,
            "cosine_sine_float64_cells": int(plan.cosine_sine.size),
            "diagonal_sign_float64_cells": int(plan.diagonal_signs.size),
            "retained_plan_bytes": plan_bytes,
            "source_orthogonality_error": plan.source_orthogonality_error,
            "triangular_off_diagonal_error": (
                plan.triangular_off_diagonal_error
            ),
            "triangular_diagonal_magnitude_error": (
                plan.triangular_diagonal_magnitude_error
            ),
            "compilation_maximum_named_bytes": (
                plan.compilation_maximum_named_bytes
            ),
        },
        "resource_law": {
            "resident_phase_angle_float64_cells": 34,
            "resident_phase_angle_bytes": resident_bytes,
            "retained_public_givens_plan_float64_cells": 289,
            "retained_public_givens_plan_bytes": plan_bytes,
            "implicit_givens_index_schedule_stored_cells": 0,
            "retained_shell_count_int64_equivalent_cells": 17,
            "retained_shell_count_bytes": shell_count_bytes,
            "maximum_named_update_bytes": maximum_update_bytes,
            "maximum_named_projection_bytes": maximum_projection_bytes,
            "maximum_named_restoration_verification_bytes": 64,
            "maximum_public_program_json_bytes": maximum_program_bytes,
            "maximum_named_warm_execution_live_bytes_including_program_json": (
                native_maximum_live
            ),
            "maximum_named_full_lifecycle_live_bytes": max(
                native_maximum_live, plan.compilation_maximum_named_bytes
            ),
            "maximum_named_commitment_live_bytes_including_program_json": (
                native_commitment_live
            ),
            "public_plan_compilation_maximum_named_bytes": (
                plan.compilation_maximum_named_bytes
            ),
            "retained_complex_state_cells": 0,
            "retained_dense_kernel_cells": 0,
            "compiled_gate_sequence_cells": 0,
            "inverse_history_cells": 0,
            "local_cartesian_registers_per_coupler": 8,
            "local_chart_scratch_float64_cells": 4,
            "input_phasor_cosine_evaluations_per_fourier": 544,
            "input_phasor_sine_evaluations_per_fourier": 544,
            "chart_hypot_evaluations_per_fourier": 272,
            "chart_atan2_and_acos_upper_bound_each_per_fourier": 272,
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
            "hashlib_internal_representation_excluded_beyond_declared_logical96_bytes": True,
        },
        "matched_classical_baseline": {
            "method": "IDENTICAL_PUBLIC_PLAN17_COMPLEX_IN_PLACE_GIVENS_RECURRENCE",
            "resident_complex128_cells": 17,
            "resident_bytes": resident_bytes,
            "retained_public_plan_bytes": plan_bytes,
            "maximum_named_warm_execution_live_bytes_including_program_json": (
                matched_complex_live
            ),
            "maximum_named_full_lifecycle_live_bytes": max(
                matched_complex_live, plan.compilation_maximum_named_bytes
            ),
            "local_complex_scratch_bytes": 64,
            "fourier_input_phasor_or_chart_trigonometry": 0,
            "phase_module_cosine_and_sine_evaluations_per_step": 34,
            "all_case_boundary_and_state_frontiers_within_tolerance": True,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "cases": cases,
        "controls": control_results,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "restoration": {
            "class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
            "same_backing": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "post_inverse_state_reset_or_canonical_reload_used": False,
            "intrinsic_local_chart_canonicalization_used_during_forward_and_inverse": True,
            "generation_is_package_local_not_catvm_lease": True,
            "transient_buffers": "NO_RESTORATION_CLAIM",
        },
        "claim_boundary": {
            "established": [
                "WEIGHTED_RADIAL_TRANSFORM_IS_PUBLIC_REAL_SYMMETRIC_ORTHOGONAL_WITHIN_TOLERANCE",
                "DETERMINISTIC136_TWO_CELL_GIVENS_PLUS17_SIGN_FACTORIZATION",
                "LOCAL_PHASE_PAIR_CONSUMPTION_WITHOUT17_COMPLEX_ACCEPTED_STATE",
                "ZERO_RETAINED_DENSE_KERNEL_AND_ZERO_INVERSE_HISTORY",
                "FINAL_ONLY_BOUNDARY_PROJECTION",
                "HISTORY_FREE_NUMERICAL_INVERSE_RESTORATION_ON_SAME_BACKING",
                "UNRELATED_AND100_CYCLE_REUSE_WITHIN_PREDECLARED_TOLERANCE",
            ],
            "not_established": [
                "CARTESIAN_REGISTER_FREE_LOCAL_COUPLING",
                "EXACT_ALGEBRAIC_SEMANTICS",
                "UNBOUNDED_DEPTH_NUMERICAL_STABILITY",
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_EXECUTION",
                "PHYSICAL_BIT_REPLACEMENT",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "next_obstruction": (
            "LOCAL_POLAR_GIVENS_REMOVES_GLOBAL_CARTESIAN_ACCUMULATION_BUT_"
            "RETAINS_LOCAL_CARTESIAN_REGISTERS_A2312_BYTE_PUBLIC_PLAN_AND_"
            "THE_SMALLER_LOWER_TRIGONOMETRY_IDENTICAL_COMPLEX_GIVENS_RECURRENCE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
