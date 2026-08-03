#!/usr/bin/env python3
"""Stream the anisotropic F17 radial Fourier directly from phase angles.

The accepted carrier stores two unit-phasor angles per normalized radial
coefficient.  The Fourier update consumes those angles directly.  Public
``p,-p`` character pairs make every radial kernel entry real, so one output
is accumulated at a time with two Cartesian scalar registers and returned to
the two-angle chart.  No 17-complex state vector, dense 17 by 17 kernel, gate
tape, or inverse history is retained by the accepted path.

This is a bounded numerical state-law repair, not evidence of an advantage:
the scalar accumulators are a streamed complex scalar and the strongest
matched 17-complex recurrence uses much less trigonometric work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


P = 17
TAU = 2.0 * math.pi
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
SHELL_COUNTS = (1, *([18] * 16))
RESTORATION_TOLERANCE = 1.0e-11
BOUNDARY_TOLERANCE = 5.0e-11
CHART_ZERO_FLOOR = 1.0e-7
CHART_RADIUS_TOLERANCE = 2.0e-12
CONTROL_FLOOR = 1.0e-5
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
CLAIM = (
    "BOUNDED_STREAMED_ANGLE_DOMAIN_ANISOTROPIC_F17_RADIAL_INTERFERENCE_"
    "CONSUMES_FIXED34_RESIDENT_PHASE_ANGLES_WITHOUT17_COMPLEX_STATE_"
    "DECODE_OR_RETAINED_DENSE_KERNEL_ACROSS21_CASES_THROUGH_DEPTH4096_"
    "WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_USES_TWO_"
    "CARTESIAN_ACCUMULATORS_PER_OUTPUT_AND_IS_STRICTLY_MORE_"
    "TRIGONOMETRIC_WORK_THAN_THE_IDENTICAL17_COMPLEX_MATRIX_FREE_"
    "RECURRENCE"
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


def phasor_pair_value(left: float, right: float) -> tuple[float, float]:
    return (
        0.5 * (math.cos(left) + math.cos(right)),
        0.5 * (math.sin(left) + math.sin(right)),
    )


def chart_pair(real: float, imag: float) -> tuple[float, float, float]:
    magnitude = math.hypot(real, imag)
    if (
        not math.isfinite(magnitude)
        or magnitude <= CHART_ZERO_FLOOR
        or magnitude > 1.0 + CHART_RADIUS_TOLERANCE
    ):
        fail("native angle chart left its declared nonzero unit disk")
    phase = math.atan2(imag, real)
    delta = math.acos(min(1.0, max(0.0, magnitude)))
    return wrap_scalar(phase + delta), wrap_scalar(phase - delta), magnitude


def seed_pair() -> tuple[float, float]:
    delta = math.acos(1.0 / P)
    return delta, -delta


def seed_angles() -> np.ndarray:
    left, right = seed_pair()
    result = np.empty((P, 2), dtype=np.float64)
    result[:, 0] = left
    result[:, 1] = right
    return result


def phasor_distance_from_seed(angles: np.ndarray) -> float:
    seed_left, seed_right = seed_pair()
    maximum = 0.0
    for shell in range(P):
        for slot, expected in ((0, seed_left), (1, seed_right)):
            observed = float(angles[shell, slot])
            distance = math.hypot(
                math.cos(observed) - math.cos(expected),
                math.sin(observed) - math.sin(expected),
            )
            maximum = max(maximum, distance)
    return maximum


def angles_state_error(angles: np.ndarray, state: np.ndarray) -> float:
    maximum = 0.0
    for shell in range(P):
        real, imag = phasor_pair_value(
            float(angles[shell, 0]), float(angles[shell, 1])
        )
        maximum = max(maximum, abs(complex(real, imag) - state[shell]))
    return maximum


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


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
        fail("unknown native-angle public family")
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
            "schema": "CAT_CAS_F17_NATIVE_ANGLE_STREAMED_PROGRAM_V1",
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
        fail("native-angle program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("native-angle program family outside declared set")
    program = Program(
        depth,
        family,
        (3 * depth + 2 * len(family) + 1) % P or 1,
        (5 * depth + len(family) + 4) % P,
    )
    gate_parameters(0, family)
    gate_parameters(depth - 1, family)
    return program


@dataclass(frozen=True)
class Geometry:
    inverse_four_parameters_half: tuple[int, ...]

    @classmethod
    def compile(cls) -> "Geometry":
        return cls(
            tuple(pow((4 * parameter) % P, -1, P) for parameter in range(1, 9))
        )


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    streamed_gate_rematerializations: int = 0
    angle_domain_fourier_calls: int = 0
    kernel_cosine_evaluations: int = 0
    input_phasor_cosine_evaluations: int = 0
    input_phasor_sine_evaluations: int = 0
    chart_hypot_evaluations: int = 0
    chart_atan2_evaluations: int = 0
    chart_acos_evaluations: int = 0
    projection_phasor_cosine_evaluations: int = 0
    projection_phasor_sine_evaluations: int = 0
    maximum_named_update_float64_cells: int = 0
    maximum_named_update_bytes: int = 0
    maximum_named_projection_float64_cells: int = 0
    maximum_named_projection_bytes: int = 0
    minimum_chart_magnitude: float = math.inf
    maximum_chart_magnitude: float = 0.0

    def observe_update(self, float64_cells: int) -> None:
        self.maximum_named_update_float64_cells = max(
            self.maximum_named_update_float64_cells, float64_cells
        )
        self.maximum_named_update_bytes = max(
            self.maximum_named_update_bytes,
            float64_cells * np.dtype(np.float64).itemsize,
        )

    def observe_projection(self, float64_cells: int) -> None:
        self.maximum_named_projection_float64_cells = max(
            self.maximum_named_projection_float64_cells, float64_cells
        )
        self.maximum_named_projection_bytes = max(
            self.maximum_named_projection_bytes,
            float64_cells * np.dtype(np.float64).itemsize,
        )


def apply_phase(
    angles: np.ndarray,
    index: int,
    family: str,
    stats: Stats,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * phase_exponent(shell, index, family) / P
        angles[shell, 0] = wrap_scalar(float(angles[shell, 0]) + delta)
        angles[shell, 1] = wrap_scalar(float(angles[shell, 1]) + delta)
    stats.streamed_gate_rematerializations += 1
    # Logical scalar exponent/delta registers; the output aliases the carrier.
    stats.observe_update(4)


def real_radial_kernel(
    target: int,
    source: int,
    geometry: Geometry,
    stats: Stats | None,
    *,
    omit_last_pair: bool = False,
) -> float:
    value = 1.0 if target == 0 else 0.0
    stop = 7 if omit_last_pair else 8
    for offset in range(stop):
        parameter = offset + 1
        inverse = geometry.inverse_four_parameters_half[offset]
        exponent = (-source * parameter - target * inverse) % P
        value -= (2.0 / P) * math.cos(TAU * exponent / P)
        if stats is not None:
            stats.kernel_cosine_evaluations += 1
    return value


def apply_angle_fourier(
    carrier: "Carrier",
    geometry: Geometry,
    stats: Stats,
    *,
    omit_last_kernel_pair: bool = False,
) -> None:
    output_angles = np.empty((P, 2), dtype=np.float64)
    for target in range(P):
        accumulator_real = 0.0
        accumulator_imag = 0.0
        for source in range(P):
            coefficient = real_radial_kernel(
                target,
                source,
                geometry,
                stats,
                omit_last_pair=omit_last_kernel_pair,
            )
            left = float(carrier.angles[source, 0])
            right = float(carrier.angles[source, 1])
            source_real, source_imag = phasor_pair_value(left, right)
            stats.input_phasor_cosine_evaluations += 2
            stats.input_phasor_sine_evaluations += 2
            accumulator_real += coefficient * source_real
            accumulator_imag += coefficient * source_imag
        left, right, magnitude = chart_pair(
            accumulator_real, accumulator_imag
        )
        output_angles[target, 0] = left
        output_angles[target, 1] = right
        stats.chart_hypot_evaluations += 1
        stats.chart_atan2_evaluations += 1
        stats.chart_acos_evaluations += 1
        stats.minimum_chart_magnitude = min(
            stats.minimum_chart_magnitude, magnitude
        )
        stats.maximum_chart_magnitude = max(
            stats.maximum_chart_magnitude, magnitude
        )
    carrier.angles[:] = output_angles
    stats.angle_domain_fourier_calls += 1
    # 34 output angles plus twelve logical scalar float64 registers.  Python
    # object/allocator representation is excluded and reported separately.
    stats.observe_update(34 + 12)


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
        return phasor_distance_from_seed(self.angles)


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong native-angle carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("native-angle carrier owner or stage changed")


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null native-angle carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or carrier.restored_error() > RESTORATION_TOLERANCE
    ):
        fail("native-angle carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def forward(carrier: Carrier, program: Program, geometry: Geometry) -> None:
    require_owned(carrier, program, "FORWARD")
    for index in range(program.depth):
        apply_phase(carrier.angles, index, program.family, carrier.stats)
        apply_angle_fourier(carrier, geometry, carrier.stats)
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_NATIVE_ANGLE_STATE_RESIDENT"


def project(carrier: Carrier, program: Program) -> complex:
    require_owned(carrier, program, "FINAL_NATIVE_ANGLE_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("native-angle final projection order changed")
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
        weight = SHELL_COUNTS[shell] * 0.5
        boundary_real += weight * (math.cos(left) + math.cos(right))
        boundary_imag += weight * (math.sin(left) + math.sin(right))
        carrier.stats.projection_phasor_cosine_evaluations += 2
        carrier.stats.projection_phasor_sine_evaluations += 2
    carrier.stats.observe_projection(10)
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return complex(boundary_real, boundary_imag)


def inverse(carrier: Carrier, program: Program, geometry: Geometry) -> float:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("native-angle inverse requires one final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_angle_fourier(carrier, geometry, carrier.stats)
        apply_phase(
            carrier.angles,
            index,
            program.family,
            carrier.stats,
            inverse=True,
        )
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("native-angle actual inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def state_commitment(carrier: Carrier) -> str:
    # Hash a zero-copy byte view.  The 64-byte public hexadecimal digest and
    # logical SHA-256 state/buffer are counted; hashlib internals remain an
    # explicit native-library exclusion.
    byte_view = memoryview(carrier.angles).cast("B")
    return hashlib.sha256(byte_view).hexdigest()


def classical_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roots = np.exp(2j * math.pi * np.arange(P) / P).astype(np.complex128)
    inverse = np.asarray(
        [pow((4 * value) % P, -1, P) for value in range(1, P)],
        dtype=np.int64,
    )
    sources = np.arange(P, dtype=np.int64)
    return roots, inverse, sources


def classical_fourier(
    state: np.ndarray,
    roots: np.ndarray,
    inverse: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    spectrum = np.empty(P - 1, dtype=np.complex128)
    for offset, parameter in enumerate(range(1, P)):
        spectrum[offset] = np.dot(
            roots[(-sources * parameter) % P], state
        )
    state_sum = np.sum(state)
    output = np.empty(P, dtype=np.complex128)
    for target in range(P):
        accumulator = np.dot(roots[(-target * inverse) % P], spectrum)
        output[target] = (
            state_sum if target == 0 else 0.0j
        ) - accumulator / P
    return output


def classical_streamed_fourier(
    state: np.ndarray, geometry: Geometry
) -> np.ndarray:
    output = np.empty(P, dtype=np.complex128)
    for target in range(P):
        accumulator = 0.0j
        for source in range(P):
            coefficient = real_radial_kernel(
                target, source, geometry, None
            )
            accumulator += coefficient * state[source]
        output[target] = accumulator
    return output


def classical_boundary(program: Program) -> tuple[complex, np.ndarray]:
    roots, inverse_parameters, sources = classical_geometry()
    state = np.full(P, 1.0 / P, dtype=np.complex128)
    for index in range(program.depth):
        exponents = np.asarray(
            [phase_exponent(shell, index, program.family) for shell in range(P)],
            dtype=np.int64,
        )
        state *= roots[exponents]
        state = classical_fourier(state, roots, inverse_parameters, sources)
    boundary = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        boundary += SHELL_COUNTS[shell] * roots[exponent] * state[shell]
    return complex(boundary), state


def classical_streamed_boundary(
    program: Program, geometry: Geometry
) -> tuple[complex, np.ndarray]:
    state = np.full(P, 1.0 / P, dtype=np.complex128)
    for index in range(program.depth):
        for shell in range(P):
            angle = TAU * phase_exponent(shell, index, program.family) / P
            state[shell] *= complex(math.cos(angle), math.sin(angle))
        state = classical_streamed_fourier(state, geometry)
    boundary = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        angle = TAU * exponent / P
        boundary += (
            SHELL_COUNTS[shell]
            * complex(math.cos(angle), math.sin(angle))
            * state[shell]
        )
    return complex(boundary), state


def execute_case(geometry: Geometry, depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program, geometry)
    commitment = state_commitment(carrier)
    boundary = project(carrier, program)
    classical, classical_state = classical_boundary(program)
    streamed_classical, streamed_classical_state = classical_streamed_boundary(
        program, geometry
    )
    state_error = angles_state_error(carrier.angles, classical_state)
    boundary_error = abs(boundary - classical)
    streamed_boundary_error = abs(boundary - streamed_classical)
    classical_frontier_boundary_error = abs(classical - streamed_classical)
    classical_frontier_state_error = float(
        np.max(np.abs(classical_state - streamed_classical_state))
    )
    if boundary_error > BOUNDARY_TOLERANCE:
        fail("native-angle boundary exceeded matched classical tolerance")
    if (
        streamed_boundary_error > BOUNDARY_TOLERANCE
        or classical_frontier_boundary_error > BOUNDARY_TOLERANCE
        or classical_frontier_state_error > BOUNDARY_TOLERANCE
    ):
        fail("native-angle matched classical frontier disagreed")
    restoration_error = inverse(carrier, program, geometry)
    if carrier.backing_identity() != backing:
        fail("native-angle carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_classical_boundary": complex_pair(classical),
        "matched_streamed_classical_boundary": complex_pair(streamed_classical),
        "maximum_state_error_against_matched_classical": state_error,
        "boundary_error_against_matched_classical": boundary_error,
        "boundary_error_against_streamed_classical": streamed_boundary_error,
        "matched_classical_frontier_boundary_error": (
            classical_frontier_boundary_error
        ),
        "matched_classical_frontier_state_error": classical_frontier_state_error,
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
    carrier: Carrier, program: Program, geometry: Geometry
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program, geometry)
    boundary = project(carrier, program)
    restoration_error = inverse(carrier, program, geometry)
    return boundary, restoration_error


def reuse_control(geometry: Geometry) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, compile_program(64, "PRIMARY"), geometry)
    restored_boundary, restoration_error = run_transaction(
        carrier, compile_program(REUSE_DEPTH, "REUSE"), geometry
    )
    fresh = Carrier.create()
    fresh_boundary, _ = run_transaction(
        fresh, compile_program(REUSE_DEPTH, "REUSE"), geometry
    )
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "same_original_backing": carrier.backing_identity() == backing,
        "fresh_restored_boundary_error": abs(
            fresh_boundary - restored_boundary
        ),
        "restoration_error": restoration_error,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control(geometry: Geometry) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    program = compile_program(REPEATED_REUSE_DEPTH, "PRIMARY")
    maximum_error = 0.0
    for _ in range(REPEATED_REUSE_CYCLES):
        _, error = run_transaction(carrier, program, geometry)
        maximum_error = max(maximum_error, error)
    return {
        "cycles": REPEATED_REUSE_CYCLES,
        "depth_per_cycle": REPEATED_REUSE_DEPTH,
        "maximum_restoration_error": maximum_error,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def raw_reverse(
    angles: np.ndarray,
    program: Program,
    geometry: Geometry,
    *,
    reordered: bool = False,
) -> np.ndarray:
    carrier = Carrier(np.array(angles, copy=True), stage="RAW")
    for index in range(program.depth - 1, -1, -1):
        if reordered:
            apply_phase(
                carrier.angles,
                index,
                program.family,
                carrier.stats,
                inverse=True,
            )
            apply_angle_fourier(carrier, geometry, carrier.stats)
        else:
            apply_angle_fourier(carrier, geometry, carrier.stats)
            apply_phase(
                carrier.angles,
                index,
                program.family,
                carrier.stats,
                inverse=True,
            )
    return carrier.angles


def controls(geometry: Geometry) -> dict[str, bool]:
    primary = compile_program(4, "PRIMARY")
    carrier = Carrier.create()
    begin_forward(carrier, primary)
    premature_projection_rejected = False
    try:
        project(carrier, primary)
    except RuntimeError:
        premature_projection_rejected = True
    forward(carrier, primary, geometry)
    final_angles = np.array(carrier.angles, copy=True)
    missing_inverse_changes_state = (
        phasor_distance_from_seed(final_angles) > CONTROL_FLOOR
    )
    wrong = raw_reverse(final_angles, compile_program(4, "REUSE"), geometry)
    wrong_inverse_changes_state = (
        phasor_distance_from_seed(wrong) > CONTROL_FLOOR
    )
    reordered = raw_reverse(final_angles, primary, geometry, reordered=True)
    reordered_inverse_changes_state = (
        phasor_distance_from_seed(reordered) > CONTROL_FLOOR
    )
    reference = project(carrier, primary)
    inverse(carrier, primary, geometry)

    roots, inverse_parameters, sources = classical_geometry()
    disabled_state = np.full(P, 1.0 / P, dtype=np.complex128)
    for _ in range(primary.depth):
        disabled_state = classical_fourier(
            disabled_state, roots, inverse_parameters, sources
        )
    disabled_boundary = 0.0j
    for shell in range(P):
        exponent = (
            primary.observation_quadratic * shell * shell
            + primary.observation_linear * shell
        ) % P
        disabled_boundary += (
            SHELL_COUNTS[shell] * roots[exponent] * disabled_state[shell]
        )
    phase_disabled_changes_boundary = bool(
        abs(disabled_boundary - reference) > CONTROL_FLOOR
    )

    mutated = Carrier.create()
    begin_forward(mutated, primary)
    try:
        for index in range(primary.depth):
            apply_phase(mutated.angles, index, primary.family, mutated.stats)
            apply_angle_fourier(
                mutated,
                geometry,
                mutated.stats,
                omit_last_kernel_pair=True,
            )
        mutated.stage = "FINAL_NATIVE_ANGLE_STATE_RESIDENT"
        mutated.forward_index = primary.depth
        mutated_boundary = project(mutated, primary)
        kernel_pairing_mutation_changes_boundary = (
            abs(mutated_boundary - reference) > CONTROL_FLOOR
        )
    except RuntimeError:
        kernel_pairing_mutation_changes_boundary = True

    null_rejected = False
    try:
        begin_forward(None, primary)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    perturbed = Carrier.create()
    begin_forward(perturbed, primary)
    forward(perturbed, primary, geometry)
    project(perturbed, primary)
    perturbed.angles[0, 0] += 1.0e-3
    perturbation_rejected = False
    try:
        inverse(perturbed, primary, geometry)
    except RuntimeError:
        perturbation_rejected = True

    result = {
        "premature_projection_rejected": premature_projection_rejected,
        "missing_inverse_changes_state": missing_inverse_changes_state,
        "wrong_inverse_changes_state": wrong_inverse_changes_state,
        "reordered_inverse_changes_state": reordered_inverse_changes_state,
        "phase_disabled_changes_boundary": phase_disabled_changes_boundary,
        "kernel_pairing_mutation_changes_boundary": (
            kernel_pairing_mutation_changes_boundary
        ),
        "null_carrier_rejected": null_rejected,
        "phase_perturbation_rejected_by_restoration": perturbation_rejected,
    }
    if not all(result.values()):
        fail("native-angle control failed")
    return result


def run() -> dict[str, Any]:
    geometry = Geometry.compile()
    cases = [
        execute_case(geometry, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    if not all(
        case["boundary_error_against_matched_classical"] <= BOUNDARY_TOLERANCE
        and case["boundary_error_against_streamed_classical"]
        <= BOUNDARY_TOLERANCE
        and case["matched_classical_frontier_boundary_error"]
        <= BOUNDARY_TOLERANCE
        and case["matched_classical_frontier_state_error"]
        <= BOUNDARY_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        for case in cases
    ):
        fail("native-angle declared case failed")
    reuse = reuse_control(geometry)
    repeated = repeated_reuse_control(geometry)
    if not (
        reuse["same_original_backing"]
        and reuse["fresh_restored_boundary_error"] <= BOUNDARY_TOLERANCE
        and reuse["restoration_error"] <= RESTORATION_TOLERANCE
        and reuse["restoration_generation"] == 2
        and repeated["same_backing"]
        and repeated["maximum_restoration_error"] <= RESTORATION_TOLERANCE
        and repeated["restoration_generation"] == REPEATED_REUSE_CYCLES
    ):
        fail("native-angle reuse control failed")
    controls_result = controls(geometry)
    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_update_bytes = max(
        case["stats"]["maximum_named_update_bytes"] for case in cases
    )
    maximum_projection_bytes = max(
        case["stats"]["maximum_named_projection_bytes"] for case in cases
    )
    return {
        "schema": "CAT_CAS_F17_ANISOTROPIC_RADIAL_NATIVE_ANGLE_INTERFERENCE_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "execution_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_SOFTWARE",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
            "chart_zero_floor": CHART_ZERO_FLOOR,
            "chart_radius_slack": CHART_RADIUS_TOLERANCE,
        },
        "source_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "unrelated_reuse_depth": REUSE_DEPTH,
            "repeated_reuse_depth": REPEATED_REUSE_DEPTH,
            "repeated_reuse_cycles": REPEATED_REUSE_CYCLES,
        },
        "cases": cases,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "controls": controls_result,
        "resource_law": {
            "resident_phase_angle_float64_cells": 34,
            "resident_phase_angle_bytes": 272,
            "retained_inverse_half_parameter_int64_cells": 8,
            "retained_inverse_half_parameter_bytes": 64,
            "retained_shell_count_int64_equivalent_cells": 17,
            "retained_shell_count_bytes": 136,
            "retained_complex_state_cells": 0,
            "retained_dense_kernel_cells": 0,
            "compiled_gate_sequence_cells": 0,
            "inverse_history_cells": 0,
            "cartesian_accumulator_float64_cells_per_output": 2,
            "output_angle_buffer_float64_cells": 34,
            "maximum_named_update_bytes": maximum_update_bytes,
            "maximum_named_projection_bytes": maximum_projection_bytes,
            "maximum_public_program_json_bytes": maximum_program_bytes,
            "maximum_named_accepted_live_bytes_including_program_json": (
                272 + 64 + 136 + maximum_update_bytes + maximum_program_bytes
            ),
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "maximum_named_commitment_live_bytes_including_program_json": (
                272 + 64 + 136 + 64 + 96 + maximum_program_bytes
            ),
            "kernel_cosine_evaluations_per_fourier": P * P * 8,
            "input_phasor_cosine_evaluations_per_fourier": P * P * 2,
            "input_phasor_sine_evaluations_per_fourier": P * P * 2,
            "chart_hypot_evaluations_per_fourier": P,
            "chart_atan2_evaluations_per_fourier": P,
            "chart_acos_evaluations_per_fourier": P,
            "matched_complex_character_products_per_fourier": 2 * P * (P - 1),
            "matched_matrix_free_retained_geometry_bytes": 536,
            "matched_matrix_free_maximum_transform_live_bytes_including_geometry": 1592,
            "matched_matrix_free_maximum_live_bytes_including_program_json": (
                1592 + maximum_program_bytes
            ),
            "matched_streamed_complex_maximum_named_update_bytes": 336,
            "matched_streamed_complex_maximum_live_bytes_including_program_json": (
                272 + 64 + 136 + 336 + maximum_program_bytes
            ),
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
            "hashlib_internal_representation_excluded_beyond_declared_logical96_bytes": True,
            "logical_scalar_register_accounting_is_not_python_object_peak": True,
            "verification_baseline_is_not_accepted_phase_state": True,
        },
        "matched_classical_baseline": {
            "work_minimizing_method": "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE",
            "equal_memory_method": "STREAMED_REAL_KERNEL_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE",
            "resident_complex128_cells": 17,
            "resident_bytes": 272,
            "resident_bytes_equal": True,
            "all_case_frontier_boundaries_and_states_within_predeclared_tolerance": True,
            "complex_character_products_per_fourier": 544,
            "matrix_free_retained_geometry_bytes": 536,
            "matrix_free_maximum_live_bytes_including_program_json": (
                1592 + maximum_program_bytes
            ),
            "streamed_complex_maximum_live_bytes_including_program_json": (
                272 + 64 + 136 + 336 + maximum_program_bytes
            ),
            "native_angle_maximum_live_bytes_including_program_json": (
                272 + 64 + 136 + maximum_update_bytes + maximum_program_bytes
            ),
            "native_angle_kernel_cosines_per_fourier": 2312,
            "native_angle_input_phasor_sin_cos_per_fourier": 1156,
            "native_angle_chart_hypot_atan2_acos_per_fourier": 51,
            "native_angle_path_has_strictly_more_trigonometric_work": True,
            "equal_memory_streamed_complex_uses_no_input_phasor_or_chart_trigonometry": True,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "restoration": {
            "class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
            "post_inverse_state_reset_or_canonical_reload_used": False,
            "transient_buffers": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "fresh_restored_reuse_equal_within_tolerance": True,
            "repeated_reuse_tested": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "generation_is_package_local_not_catvm_lease": True,
        },
        "claim_boundary": {
            "established": [
                "DIRECT_ANGLE_CONSUMPTION_WITHOUT17_COMPLEX_STATE_DECODE",
                "PUBLIC_P_NEG_P_REAL_KERNEL_PAIRING",
                "TWO_CARTESIAN_SCALAR_ACCUMULATORS_PER_OUTPUT",
                "ZERO_RETAINED_DENSE_KERNEL_AND_ZERO_INVERSE_HISTORY",
                "FINAL_ONLY_STREAMED_BOUNDARY_PROJECTION",
                "HISTORY_FREE_NUMERICAL_INVERSE_RESTORATION_ON_SAME_BACKING",
                "UNRELATED_AND100_CYCLE_REUSE_WITHIN_PREDECLARED_TOLERANCE",
            ],
            "not_established": [
                "CARTESIAN_ACCUMULATOR_FREE_INTERFERENCE",
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
            "DIRECT_ANGLE_INTERFERENCE_REMOVES_THE17_COMPLEX_STATE_DECODE_"
            "BUT_EACH_OUTPUT_STILL_USES_TWO_CARTESIAN_ACCUMULATORS_AND_"
            "THE_IDENTICAL_MATRIX_FREE_COMPLEX_RECURRENCE_HAS_EQUAL_"
            "RESIDENT_BYTES_AND_MUCH_LOWER_TRIGONOMETRIC_WORK"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
