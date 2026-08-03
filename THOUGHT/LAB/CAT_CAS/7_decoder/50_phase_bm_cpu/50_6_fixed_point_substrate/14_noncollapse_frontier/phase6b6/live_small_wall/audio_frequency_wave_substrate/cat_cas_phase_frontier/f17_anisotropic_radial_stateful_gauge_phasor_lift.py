#!/usr/bin/env python3
"""Exercise a stateful gauge lift of the local F17 radial couplers.

M145 stores each complex radial coefficient as the mean of two unit phasors.
That chart has a circle of representatives at zero but only finitely many
representatives away from zero, so a nontrivial Givens rotation cannot lift
injectively on the fixed four-angle two-coordinate carrier across zero.

This bounded successor adds one small, causally active gauge phasor to every
coordinate.  For epsilon=1/32 it stores

    z = epsilon exp(i g) + (1-epsilon)/2 * (exp(i a) + exp(i b)).

For every |z| <= 15/16 and every resident gauge g, the residual two-phasor
chart is valid by the triangle inequality.  Phase modules rotate all three
angles.  Each local Givens operation transports the two gauge angles by an
invertible public counter-rotation before recharting the residual phasors.
The actual gauge therefore survives a base-amplitude zero and is restored by
the inverse without a snapshot or retained history.

This is not a phase-resource or advantage claim.  The accepted path has 51
resident angles rather than 34 and retains local Cartesian scratch and
transcendental chart arithmetic.  The comparison set includes the identical
local-plan complex recurrence plus the lower-memory matrix-free and streamed
real-kernel complex frontiers retained from M144/M145.
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

import f17_anisotropic_radial_local_polar_givens_coupling as m145


P = m145.P
TAU = m145.TAU
FAMILIES = m145.FAMILIES
DEPTHS = m145.DEPTHS
ROTATION_COUNT = m145.ROTATION_COUNT
GAUGE_WEIGHT = 1.0 / 32.0
RESIDUAL_WEIGHT = 1.0 - GAUGE_WEIGHT
SUPPORTED_BASE_MAGNITUDE = 15.0 / 16.0
RESTORATION_TOLERANCE = 3.0e-11
STATE_TOLERANCE = 2.0e-11
BOUNDARY_TOLERANCE = 1.0e-10
CHART_RADIUS_TOLERANCE = 2.0e-12
CHART_ZERO_FLOOR = 1.0e-14
CONTROL_FLOOR = 1.0e-5
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
CLAIM = (
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


def wrap_scalar(value: float) -> float:
    return (value + math.pi) % TAU - math.pi


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def phase_cell_error(left: np.ndarray, right: np.ndarray) -> float:
    maximum = 0.0
    for observed, expected in zip(left.flat, right.flat):
        maximum = max(
            maximum,
            math.hypot(
                math.cos(float(observed)) - math.cos(float(expected)),
                math.sin(float(observed)) - math.sin(float(expected)),
            ),
        )
    return maximum


def gauge_seed(shell: int) -> float:
    return wrap_scalar(TAU * ((5 * shell + 3) % P) / P + 0.137)


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    local_coupler_calls: int = 0
    gauge_transport_updates: int = 0
    sign_updates: int = 0
    input_phasor_cosine_evaluations: int = 0
    input_phasor_sine_evaluations: int = 0
    output_gauge_cosine_evaluations: int = 0
    output_gauge_sine_evaluations: int = 0
    gauge_rotation_atan2_evaluations: int = 0
    chart_hypot_evaluations: int = 0
    chart_atan2_evaluations: int = 0
    chart_acos_evaluations: int = 0
    residual_zero_canonicalizations: int = 0
    projection_phasor_cosine_evaluations: int = 0
    projection_phasor_sine_evaluations: int = 0
    maximum_named_update_float64_cells: int = 0
    maximum_named_update_bytes: int = 0
    maximum_named_projection_float64_cells: int = 0
    maximum_named_projection_bytes: int = 0
    minimum_residual_chart_magnitude: float = math.inf
    maximum_residual_chart_magnitude: float = 0.0
    maximum_base_magnitude: float = 0.0

    def observe_update(self, cells: int) -> None:
        self.maximum_named_update_float64_cells = max(
            self.maximum_named_update_float64_cells, cells
        )
        self.maximum_named_update_bytes = max(
            self.maximum_named_update_bytes, 8 * cells
        )

    def observe_projection(self, cells: int) -> None:
        self.maximum_named_projection_float64_cells = max(
            self.maximum_named_projection_float64_cells, cells
        )
        self.maximum_named_projection_bytes = max(
            self.maximum_named_projection_bytes, 8 * cells
        )

    def observe_chart(
        self, residual_magnitude: float, base_magnitude: float, zero: bool
    ) -> None:
        self.chart_hypot_evaluations += 2
        self.maximum_base_magnitude = max(
            self.maximum_base_magnitude, base_magnitude
        )
        self.maximum_residual_chart_magnitude = max(
            self.maximum_residual_chart_magnitude, residual_magnitude
        )
        if zero:
            self.residual_zero_canonicalizations += 1
        else:
            self.chart_atan2_evaluations += 1
            self.chart_acos_evaluations += 1
            self.minimum_residual_chart_magnitude = min(
                self.minimum_residual_chart_magnitude, residual_magnitude
            )


def decode_triplet(row: np.ndarray, stats: Stats | None = None) -> complex:
    gauge, left, right = (float(row[0]), float(row[1]), float(row[2]))
    if stats is not None:
        stats.input_phasor_cosine_evaluations += 3
        stats.input_phasor_sine_evaluations += 3
    return (
        GAUGE_WEIGHT * complex(math.cos(gauge), math.sin(gauge))
        + 0.5
        * RESIDUAL_WEIGHT
        * (
            complex(math.cos(left), math.sin(left))
            + complex(math.cos(right), math.sin(right))
        )
    )


def encode_triplet(
    value: complex, gauge: float, stats: Stats | None = None
) -> tuple[float, float, float, float, bool]:
    base_magnitude = abs(value)
    if (
        not math.isfinite(base_magnitude)
        or base_magnitude > SUPPORTED_BASE_MAGNITUDE + CHART_RADIUS_TOLERANCE
    ):
        fail("stateful gauge chart left its declared base-magnitude envelope")
    gauge_phasor = complex(math.cos(gauge), math.sin(gauge))
    if stats is not None:
        stats.output_gauge_cosine_evaluations += 1
        stats.output_gauge_sine_evaluations += 1
    residual = (value - GAUGE_WEIGHT * gauge_phasor) / RESIDUAL_WEIGHT
    residual_magnitude = abs(residual)
    if residual_magnitude > 1.0 + CHART_RADIUS_TOLERANCE:
        fail("stateful gauge residual left the unit disk")
    zero = residual_magnitude <= CHART_ZERO_FLOOR
    if zero:
        left = math.pi / 2.0
        right = -math.pi / 2.0
    else:
        phase = math.atan2(residual.imag, residual.real)
        delta = math.acos(min(1.0, max(0.0, residual_magnitude)))
        left = wrap_scalar(phase + delta)
        right = wrap_scalar(phase - delta)
    if stats is not None:
        stats.observe_chart(residual_magnitude, base_magnitude, zero)
    return wrap_scalar(gauge), left, right, residual_magnitude, zero


def seed_angles() -> np.ndarray:
    angles = np.empty((P, 3), dtype=np.float64)
    for shell in range(P):
        gauge = gauge_seed(shell)
        encoded = encode_triplet(
            complex(m145.shell_scale(shell) / P, 0.0), gauge
        )
        angles[shell] = encoded[:3]
    return angles


def decoded_state(angles: np.ndarray) -> np.ndarray:
    return np.asarray(
        [decode_triplet(angles[shell]) for shell in range(P)],
        dtype=np.complex128,
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
        maximum = 0.0
        for shell in range(P):
            gauge = gauge_seed(shell)
            expected = encode_triplet(
                complex(m145.shell_scale(shell) / P, 0.0), gauge
            )
            for slot in range(3):
                observed = float(self.angles[shell, slot])
                target = float(expected[slot])
                maximum = max(
                    maximum,
                    math.hypot(
                        math.cos(observed) - math.cos(target),
                        math.sin(observed) - math.sin(target),
                    ),
                )
        return maximum


def apply_phase(
    carrier: Carrier, index: int, family: str, *, inverse: bool = False
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * m145.phase_exponent(shell, index, family) / P
        for slot in range(3):
            carrier.angles[shell, slot] = wrap_scalar(
                float(carrier.angles[shell, slot]) + delta
            )
    carrier.stats.observe_update(5)


def apply_signs(carrier: Carrier, plan: m145.GivensPlan) -> None:
    for shell in range(P):
        if float(plan.diagonal_signs[shell]) < 0.0:
            for slot in range(3):
                carrier.angles[shell, slot] = wrap_scalar(
                    float(carrier.angles[shell, slot]) + math.pi
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
    transport_gauge: bool = True,
) -> None:
    upper_value = decode_triplet(carrier.angles[upper], carrier.stats)
    lower_value = decode_triplet(carrier.angles[lower], carrier.stats)
    theta = math.atan2(sine, cosine)
    carrier.stats.gauge_rotation_atan2_evaluations += 1
    upper_gauge = float(carrier.angles[upper, 0])
    lower_gauge = float(carrier.angles[lower, 0])
    if transpose:
        next_upper = cosine * upper_value - sine * lower_value
        next_lower = sine * upper_value + cosine * lower_value
        gauge_sign = 1.0
    else:
        next_upper = cosine * upper_value + sine * lower_value
        next_lower = -sine * upper_value + cosine * lower_value
        gauge_sign = -1.0
    if transport_gauge:
        next_upper_gauge = wrap_scalar(upper_gauge + gauge_sign * theta)
        next_lower_gauge = wrap_scalar(lower_gauge - gauge_sign * theta)
        carrier.stats.gauge_transport_updates += 2
    else:
        next_upper_gauge = upper_gauge
        next_lower_gauge = lower_gauge
    upper_encoded = encode_triplet(
        next_upper, next_upper_gauge, carrier.stats
    )
    lower_encoded = encode_triplet(
        next_lower, next_lower_gauge, carrier.stats
    )
    carrier.angles[upper] = upper_encoded[:3]
    carrier.angles[lower] = lower_encoded[:3]
    carrier.stats.local_coupler_calls += 1
    # Two decoded complex values, two mixed values, two gauge phasors, two
    # residual values, two output triplets, and conservative chart scratch.
    carrier.stats.observe_update(24)


def apply_givens_fourier(
    carrier: Carrier,
    plan: m145.GivensPlan,
    *,
    inverse: bool = False,
    transport_gauge: bool = True,
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(m145.elimination_pairs()):
            apply_local_coupler(
                carrier,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=False,
                transport_gauge=transport_gauge,
            )
        apply_signs(carrier, plan)
    else:
        apply_signs(carrier, plan)
        ordinal = ROTATION_COUNT - 1
        for upper, lower in m145.reverse_elimination_pairs():
            apply_local_coupler(
                carrier,
                upper,
                lower,
                float(plan.cosine_sine[ordinal, 0]),
                float(plan.cosine_sine[ordinal, 1]),
                transpose=True,
                transport_gauge=transport_gauge,
            )
            ordinal -= 1
        if ordinal != -1:
            fail("stateful gauge schedule did not consume the public plan")


def require_owned(
    carrier: Carrier, program: m145.Program, stage: str
) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong stateful gauge carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("stateful gauge carrier owner or stage changed")


def begin_forward(carrier: Carrier, program: m145.Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null stateful gauge carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or carrier.restored_error() > RESTORATION_TOLERANCE
    ):
        fail("stateful gauge carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def forward(
    carrier: Carrier, program: m145.Program, plan: m145.GivensPlan
) -> None:
    require_owned(carrier, program, "FORWARD")
    for index in range(program.depth):
        apply_phase(carrier, index, program.family)
        apply_givens_fourier(carrier, plan)
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_STATEFUL_GAUGE_STATE_RESIDENT"


def project(carrier: Carrier, program: m145.Program) -> complex:
    require_owned(carrier, program, "FINAL_STATEFUL_GAUGE_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("stateful gauge final projection order changed")
    boundary_real = 0.0
    boundary_imag = 0.0
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        observation = TAU * exponent / P
        gauge, left, right = (
            observation + float(carrier.angles[shell, slot])
            for slot in range(3)
        )
        scale = m145.shell_scale(shell)
        boundary_real += scale * (
            GAUGE_WEIGHT * math.cos(gauge)
            + 0.5
            * RESIDUAL_WEIGHT
            * (math.cos(left) + math.cos(right))
        )
        boundary_imag += scale * (
            GAUGE_WEIGHT * math.sin(gauge)
            + 0.5
            * RESIDUAL_WEIGHT
            * (math.sin(left) + math.sin(right))
        )
        carrier.stats.projection_phasor_cosine_evaluations += 3
        carrier.stats.projection_phasor_sine_evaluations += 3
    carrier.stats.observe_projection(14)
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED"
    return complex(boundary_real, boundary_imag)


def inverse(
    carrier: Carrier, program: m145.Program, plan: m145.GivensPlan
) -> float:
    require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("stateful gauge inverse requires one final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        apply_givens_fourier(carrier, plan, inverse=True)
        apply_phase(carrier, index, program.family, inverse=True)
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("stateful gauge inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(memoryview(carrier.angles).cast("B")).hexdigest()


def streamed_complex_boundary(
    program: m145.Program,
) -> tuple[complex, np.ndarray]:
    predecessor_program = m145.m144.compile_program(
        program.depth, program.family
    )
    geometry = m145.m144.Geometry.compile()
    boundary, unweighted_state = m145.m144.classical_streamed_boundary(
        predecessor_program, geometry
    )
    weighted_state = np.asarray(
        [
            m145.shell_scale(shell) * unweighted_state[shell]
            for shell in range(P)
        ],
        dtype=np.complex128,
    )
    return boundary, weighted_state


def execute_case(
    plan: m145.GivensPlan, depth: int, family: str
) -> dict[str, Any]:
    program = m145.compile_program(depth, family)
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    commitment = state_commitment(carrier)
    boundary = project(carrier, program)
    matched_boundary, matched_state = m145.complex_givens_boundary(
        program, plan
    )
    matrix_free_boundary, matrix_free_state = m145.matrix_free_boundary(
        program
    )
    streamed_boundary, streamed_state = streamed_complex_boundary(program)
    observed_state = decoded_state(carrier.angles)
    state_error = float(np.max(np.abs(observed_state - matched_state)))
    boundary_error = abs(boundary - matched_boundary)
    matrix_free_boundary_error = abs(boundary - matrix_free_boundary)
    matrix_free_state_error = float(
        np.max(np.abs(observed_state - matrix_free_state))
    )
    streamed_boundary_error = abs(boundary - streamed_boundary)
    streamed_state_error = float(
        np.max(np.abs(observed_state - streamed_state))
    )
    if (
        state_error > STATE_TOLERANCE
        or boundary_error > BOUNDARY_TOLERANCE
        or matrix_free_boundary_error > BOUNDARY_TOLERANCE
        or matrix_free_state_error > STATE_TOLERANCE
        or streamed_boundary_error > BOUNDARY_TOLERANCE
        or streamed_state_error > STATE_TOLERANCE
    ):
        fail("stateful gauge execution exceeded predeclared tolerance")
    restoration_error = inverse(carrier, program, plan)
    if carrier.backing_identity() != backing:
        fail("stateful gauge carrier backing changed")
    if carrier.stats.residual_zero_canonicalizations:
        fail("declared stateful gauge path reached residual-chart zero")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_complex_givens_boundary": complex_pair(matched_boundary),
        "matched_matrix_free_complex_boundary": complex_pair(
            matrix_free_boundary
        ),
        "matched_streamed_real_kernel_complex_boundary": complex_pair(
            streamed_boundary
        ),
        "maximum_state_error_against_matched_complex_givens": state_error,
        "boundary_error_against_matched_complex_givens": boundary_error,
        "maximum_state_error_against_matrix_free_complex": (
            matrix_free_state_error
        ),
        "boundary_error_against_matrix_free_complex": (
            matrix_free_boundary_error
        ),
        "maximum_state_error_against_streamed_real_kernel_complex": (
            streamed_state_error
        ),
        "boundary_error_against_streamed_real_kernel_complex": (
            streamed_boundary_error
        ),
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
    carrier: Carrier, program: m145.Program, plan: m145.GivensPlan
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    boundary = project(carrier, program)
    restoration_error = inverse(carrier, program, plan)
    return boundary, restoration_error


def reuse_control(plan: m145.GivensPlan) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, m145.compile_program(37, "PRIMARY"), plan)
    second = m145.compile_program(REUSE_DEPTH, "REUSE")
    restored_boundary, restoration_error = run_transaction(
        carrier, second, plan
    )
    fresh_boundary, _ = run_transaction(Carrier.create(), second, plan)
    error = abs(restored_boundary - fresh_boundary)
    if error > BOUNDARY_TOLERANCE:
        fail("stateful gauge reuse disagreed with fresh execution")
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "fresh_restored_boundary_error": error,
        "restoration_error": restoration_error,
        "same_original_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control(plan: m145.GivensPlan) -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    program = m145.compile_program(REPEATED_REUSE_DEPTH, "ALTERNATE")
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
    program: m145.Program,
    plan: m145.GivensPlan,
    *,
    omit_fourier: bool = False,
    wrong_plan: m145.GivensPlan | None = None,
    reordered: bool = False,
    transport_gauge: bool = True,
) -> None:
    active_plan = wrong_plan if wrong_plan is not None else plan
    for index in range(program.depth - 1, -1, -1):
        if reordered:
            apply_phase(carrier, index, program.family, inverse=True)
        if not omit_fourier:
            apply_givens_fourier(
                carrier,
                active_plan,
                inverse=True,
                transport_gauge=transport_gauge,
            )
        if not reordered:
            apply_phase(carrier, index, program.family, inverse=True)


def legacy_pair(gauge: float) -> tuple[float, float]:
    return wrap_scalar(gauge + math.pi / 2.0), wrap_scalar(
        gauge - math.pi / 2.0
    )


def legacy_zero_fiber_collision(plan: m145.GivensPlan) -> dict[str, Any]:
    upper, lower = next(m145.elimination_pairs())
    cosine = float(plan.cosine_sine[0, 0])
    sine = float(plan.cosine_sine[0, 1])
    lower_value = 0.25 * complex(math.cos(0.4), math.sin(0.4))
    gauges = (0.2, 1.7)
    initial_pairs = [legacy_pair(gauge) for gauge in gauges]
    outputs: list[np.ndarray] = []
    for pair in initial_pairs:
        zero_value = complex(*m145.phasor_pair_value(*pair))
        next_upper = cosine * zero_value - sine * lower_value
        next_lower = sine * zero_value + cosine * lower_value
        upper_chart = m145.chart_pair(next_upper.real, next_upper.imag)[:2]
        lower_chart = m145.chart_pair(next_lower.real, next_lower.imag)[:2]
        outputs.append(np.asarray((*upper_chart, *lower_chart)))
    input_separation = phase_cell_error(
        np.asarray(initial_pairs[0]), np.asarray(initial_pairs[1])
    )
    output_collision = phase_cell_error(outputs[0], outputs[1])
    return {
        "actual_public_plan_pair": [upper, lower],
        "actual_public_plan_cosine": cosine,
        "actual_public_plan_sine": sine,
        "input_nonzero_coordinate_magnitude": abs(lower_value),
        "output_coordinate_magnitudes": [
            abs(sine * lower_value),
            abs(cosine * lower_value),
        ],
        "zero_fiber_topology": "S1_CONTINUUM_PER_ZERO_COORDINATE",
        "generic_nonzero_ordered_pair_fiber_cardinality": 2,
        "two_generic_nonzero_output_fiber_cardinality": 4,
        "fiber_cardinality_mismatch_proves_any_injective_lift_impossible": True,
        "distinct_zero_input_phase_cell_separation": input_separation,
        "canonical_output_phase_cell_collision": output_collision,
        "fixed_four_angle_injective_lift_across_zero_possible": False,
        "collision_detected": bool(
            input_separation > CONTROL_FLOOR
            and output_collision <= 10.0 * np.finfo(np.float64).eps
        ),
    }


def stateful_zero_probe(plan: m145.GivensPlan) -> dict[str, Any]:
    upper, lower = next(m145.elimination_pairs())
    cosine = float(plan.cosine_sine[0, 0])
    sine = float(plan.cosine_sine[0, 1])
    records: list[dict[str, float]] = []
    forward_states: list[np.ndarray] = []
    for gauge in (0.2, 1.7):
        carrier = Carrier.create()
        encoded = encode_triplet(0.0j, gauge)
        carrier.angles[upper] = encoded[:3]
        original = np.array(carrier.angles, copy=True)
        apply_local_coupler(
            carrier, upper, lower, cosine, sine, transpose=True
        )
        forward_states.append(np.array(carrier.angles, copy=True))
        output_base = np.asarray(
            [decode_triplet(carrier.angles[index]) for index in (upper, lower)]
        )
        apply_local_coupler(
            carrier, upper, lower, cosine, sine, transpose=False
        )
        records.append(
            {
                "input_gauge": gauge,
                "exact_zero_residual_chart_radius": encoded[3],
                "forward_output_base_norm": float(np.linalg.norm(output_base)),
                "inverse_actual_phase_cell_error": phase_cell_error(
                    carrier.angles, original
                ),
            }
        )
    base_difference = max(
        abs(
            decode_triplet(forward_states[0][index])
            - decode_triplet(forward_states[1][index])
        )
        for index in (upper, lower)
    )
    carrier_difference = phase_cell_error(forward_states[0], forward_states[1])
    all_restored = all(
        record["inverse_actual_phase_cell_error"] <= RESTORATION_TOLERANCE
        for record in records
    )
    return {
        "records": records,
        "expected_exact_zero_residual_chart_radius": (
            GAUGE_WEIGHT / RESIDUAL_WEIGHT
        ),
        "same_base_output_maximum_error": base_difference,
        "distinct_forward_carrier_phase_cell_separation": carrier_difference,
        "both_actual_gauges_restored": all_restored,
        "gauge_survives_while_base_output_is_identical": (
            base_difference <= STATE_TOLERANCE
            and carrier_difference > CONTROL_FLOOR
        ),
    }


def controls(plan: m145.GivensPlan) -> dict[str, bool]:
    program = m145.compile_program(3, "ALTERNATE")
    valid = Carrier.create()
    begin_forward(valid, program)
    forward(valid, program, plan)
    valid_boundary = project(valid, program)

    changed_plan = m145.mutated_plan(plan)
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

    gauge_disabled = Carrier.create()
    begin_forward(gauge_disabled, program)
    forward(gauge_disabled, program, plan)
    raw_reverse(gauge_disabled, program, plan, transport_gauge=False)

    phase_disabled_outside_envelope = False
    try:
        phase_disabled = Carrier.create()
        begin_forward(phase_disabled, program)
        for _ in range(program.depth):
            apply_givens_fourier(phase_disabled, plan)
    except RuntimeError as error:
        phase_disabled_outside_envelope = (
            "base-magnitude envelope" in str(error)
        )

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

    envelope_rejected = False
    try:
        encode_triplet(complex(SUPPORTED_BASE_MAGNITUDE + 0.01), 0.0)
    except RuntimeError:
        envelope_rejected = True

    return {
        "plan_coefficient_mutation_changes_boundary": (
            abs(mutated_boundary - valid_boundary) > CONTROL_FLOOR
        ),
        "missing_inverse_changes_state": missing.restored_error() > CONTROL_FLOOR,
        "wrong_inverse_changes_state": wrong.restored_error() > CONTROL_FLOOR,
        "reordered_inverse_changes_state": (
            reordered_carrier.restored_error() > CONTROL_FLOOR
        ),
        "missing_gauge_inverse_changes_actual_carrier": (
            gauge_disabled.restored_error() > CONTROL_FLOOR
        ),
        "phase_disabled_path_rejected_outside_declared_envelope": (
            phase_disabled_outside_envelope
        ),
        "premature_projection_rejected": premature_rejected,
        "null_carrier_rejected": null_rejected,
        "out_of_envelope_state_rejected": envelope_rejected,
    }


def run() -> dict[str, Any]:
    plan = m145.GivensPlan.compile()
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
        and case["maximum_state_error_against_matrix_free_complex"]
        <= STATE_TOLERANCE
        and case["boundary_error_against_matrix_free_complex"]
        <= BOUNDARY_TOLERANCE
        and case[
            "maximum_state_error_against_streamed_real_kernel_complex"
        ]
        <= STATE_TOLERANCE
        and case["boundary_error_against_streamed_real_kernel_complex"]
        <= BOUNDARY_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_before"] == 0
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["stats"]["maximum_base_magnitude"]
        <= SUPPORTED_BASE_MAGNITUDE + CHART_RADIUS_TOLERANCE
        and case["stats"]["residual_zero_canonicalizations"] == 0
        for case in cases
    )
    if not all_within:
        fail("one or more stateful gauge cases failed the declared envelope")
    control_results = controls(plan)
    if not all(control_results.values()):
        fail("one or more stateful gauge controls failed")
    legacy = legacy_zero_fiber_collision(plan)
    zero_probe = stateful_zero_probe(plan)
    if not legacy["collision_detected"]:
        fail("legacy two-phasor zero-fiber collision was not reproduced")
    if not (
        zero_probe["both_actual_gauges_restored"]
        and zero_probe["gauge_survives_while_base_output_is_identical"]
    ):
        fail("stateful gauge zero probe failed")
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
    resident_bytes = 51 * 8
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
    matched_local_plan_update_live = (
        17 * 16 + plan_bytes + shell_count_bytes + 64 + maximum_program_bytes
    )
    matched_local_plan_commitment_live = (
        17 * 16
        + plan_bytes
        + shell_count_bytes
        + 96
        + 64
        + maximum_program_bytes
    )
    matched_local_plan_maximum_live = max(
        matched_local_plan_update_live, matched_local_plan_commitment_live
    )
    matched_matrix_free_live = 1592 + maximum_program_bytes
    matched_streamed_complex_live = (
        272 + 64 + shell_count_bytes + 336 + maximum_program_bytes
    )
    maximum_state_error = max(
        case["maximum_state_error_against_matched_complex_givens"]
        for case in cases
    )
    maximum_boundary_error = max(
        case["boundary_error_against_matched_complex_givens"]
        for case in cases
    )
    maximum_matrix_free_boundary_error = max(
        case["boundary_error_against_matrix_free_complex"]
        for case in cases
    )
    maximum_matrix_free_state_error = max(
        case["maximum_state_error_against_matrix_free_complex"]
        for case in cases
    )
    maximum_streamed_boundary_error = max(
        case["boundary_error_against_streamed_real_kernel_complex"]
        for case in cases
    )
    maximum_streamed_state_error = max(
        case[
            "maximum_state_error_against_streamed_real_kernel_complex"
        ]
        for case in cases
    )
    maximum_restoration_error = max(
        case["restoration_error"] for case in cases
    )
    minimum_residual = min(
        case["stats"]["minimum_residual_chart_magnitude"] for case in cases
    )
    maximum_residual = max(
        case["stats"]["maximum_residual_chart_magnitude"] for case in cases
    )
    maximum_base = max(
        case["stats"]["maximum_base_magnitude"] for case in cases
    )

    return {
        "schema": "CAT_CAS_F17_STATEFUL_GAUGE_PHASOR_LIFT_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m145.__file__).name,
            "sha256": hashlib.sha256(Path(m145.__file__).read_bytes()).hexdigest(),
            "used_for": (
                "PUBLIC_PROGRAM_AND_GIVENS_PLAN_COMPILATION_PLUS_MATCHED_"
                "COMPLEX_FRONTIERS"
            ),
        },
        "source_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_PHASE_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_within_predeclared_tolerances": all_within,
            "public_topology_compilation_reads_final_answers": False,
            "supported_base_magnitude_ceiling": SUPPORTED_BASE_MAGNITUDE,
            "full_weighted_unit_sphere_supported": False,
        },
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "state_max_abs": STATE_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
            "chart_radius_slack": CHART_RADIUS_TOLERANCE,
            "residual_chart_zero_floor": CHART_ZERO_FLOOR,
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
        "carrier_law": {
            "per_coordinate": (
                "Z=EPSILON_EXP_IG_PLUS_ONE_MINUS_EPSILON_OVER2_"
                "TIMES_EXP_IA_PLUS_EXP_IB"
            ),
            "gauge_weight": GAUGE_WEIGHT,
            "residual_weight": RESIDUAL_WEIGHT,
            "resident_phase_angles_per_coordinate": 3,
            "resident_phase_angle_cells": 51,
            "phase_module_rotates_all_three_angles": True,
            "local_givens_transports_gauge_by_public_counter_rotation": True,
            "gauge_transport_is_locally_invertible": True,
            "declared_envelope_triangle_inequality_residual_bound": 1.0,
            "base_zero_residual_radius": GAUGE_WEIGHT / RESIDUAL_WEIGHT,
            "base_zero_requires_canonicalization": False,
            "residual_chart_zero_encountered_in_declared_cases": False,
        },
        "zero_fiber_obstruction": legacy,
        "stateful_zero_probe": zero_probe,
        "observed_envelope": {
            "maximum_base_magnitude": maximum_base,
            "minimum_nonzero_residual_chart_magnitude": minimum_residual,
            "maximum_residual_chart_magnitude": maximum_residual,
        },
        "maximum_errors": {
            "state_against_matched_complex_givens": maximum_state_error,
            "boundary_against_matched_complex_givens": maximum_boundary_error,
            "state_against_matrix_free_complex": (
                maximum_matrix_free_state_error
            ),
            "boundary_against_matrix_free_complex": (
                maximum_matrix_free_boundary_error
            ),
            "state_against_streamed_real_kernel_complex": (
                maximum_streamed_state_error
            ),
            "boundary_against_streamed_real_kernel_complex": (
                maximum_streamed_boundary_error
            ),
            "single_transaction_restoration": maximum_restoration_error,
        },
        "resource_law": {
            "resident_phase_angle_float64_cells": 51,
            "resident_phase_angle_bytes": resident_bytes,
            "added_gauge_phase_angle_cells_against_m145": 17,
            "added_resident_bytes_against_m145": 17 * 8,
            "retained_public_givens_plan_float64_cells": 289,
            "retained_public_givens_plan_bytes": plan_bytes,
            "retained_shell_count_bytes": shell_count_bytes,
            "maximum_named_update_bytes": maximum_update_bytes,
            "maximum_named_projection_bytes": maximum_projection_bytes,
            "maximum_named_restoration_verification_bytes": 64,
            "maximum_public_program_json_bytes": maximum_program_bytes,
            "maximum_named_warm_execution_live_bytes_including_program_json": (
                native_maximum_live
            ),
            "maximum_named_commitment_live_bytes_including_program_json": (
                native_commitment_live
            ),
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "hashlib_internal_representation_excluded_beyond_declared_logical96_bytes": True,
            "maximum_named_full_lifecycle_live_bytes": max(
                native_maximum_live, plan.compilation_maximum_named_bytes
            ),
            "public_plan_compilation_maximum_named_bytes": (
                plan.compilation_maximum_named_bytes
            ),
            "retained_complex_state_cells": 0,
            "retained_dense_kernel_cells": 0,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "restoration_seed_rematerialized_one_coordinate_at_a_time": True,
            "local_cartesian_and_chart_scratch_float64_cells": 24,
            "input_phasor_cosine_evaluations_per_fourier": 816,
            "input_phasor_sine_evaluations_per_fourier": 816,
            "output_gauge_cosine_evaluations_per_fourier": 272,
            "output_gauge_sine_evaluations_per_fourier": 272,
            "gauge_rotation_atan2_evaluations_per_fourier": 136,
            "chart_hypot_evaluations_per_fourier": 544,
            "chart_atan2_and_acos_upper_bound_each_per_fourier": 272,
            "phase_module_trigonometric_evaluations_per_step": 0,
            "projection_phasor_cosine_evaluations": 51,
            "projection_phasor_sine_evaluations": 51,
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
        },
        "matched_classical_frontiers": {
            "resident_complex128_cells": 17,
            "resident_bytes": 272,
            "gauge_cells": 0,
            "all_case_boundary_and_state_frontiers_within_tolerance": True,
            "identical_local_plan": {
                "method": "IDENTICAL_PUBLIC_PLAN17_COMPLEX_IN_PLACE_GIVENS_RECURRENCE",
                "retained_public_plan_bytes": plan_bytes,
                "local_complex_scratch_bytes": 64,
                "maximum_named_update_live_bytes_including_program_json": (
                    matched_local_plan_update_live
                ),
                "maximum_named_commitment_live_bytes_including_program_json": (
                    matched_local_plan_commitment_live
                ),
                "maximum_named_warm_execution_live_bytes_including_program_json": (
                    matched_local_plan_maximum_live
                ),
                "maximum_named_full_lifecycle_live_bytes": max(
                    matched_local_plan_maximum_live,
                    plan.compilation_maximum_named_bytes,
                ),
                "fourier_input_phasor_chart_or_kernel_trigonometry": 0,
                "phase_module_cosine_evaluations_per_step": 17,
                "phase_module_sine_evaluations_per_step": 17,
                "projection_cosine_evaluations": 17,
                "projection_sine_evaluations": 17,
            },
            "work_frontier": {
                "method": "IDENTICAL_MATRIX_FREE_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE",
                "retained_geometry_bytes": 536,
                "complex_character_products_per_fourier": 544,
                "maximum_named_transform_live_bytes_including_geometry": 1592,
                "maximum_named_warm_execution_live_bytes_including_program_json": (
                    matched_matrix_free_live
                ),
                "phase_and_projection_roots_retained_in_public_geometry": True,
                "runtime_phase_module_and_projection_trigonometry": 0,
            },
            "memory_frontier": {
                "method": "STREAMED_REAL_KERNEL_NORMALIZED17_COMPLEX_RADIAL_RECURRENCE",
                "retained_inverse_parameter_bytes": 64,
                "maximum_named_update_bytes": 336,
                "maximum_named_warm_execution_live_bytes_including_program_json": (
                    matched_streamed_complex_live
                ),
                "kernel_cosine_evaluations_per_fourier": 2312,
                "phase_module_cosine_evaluations_per_step": 17,
                "phase_module_sine_evaluations_per_step": 17,
                "projection_cosine_evaluations": 17,
                "projection_sine_evaluations": 17,
                "input_phasor_or_chart_trigonometry": 0,
            },
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
        },
        "cases": cases,
        "controls": control_results,
        "reuse": reuse,
        "repeated_reuse": repeated,
        "restoration": {
            "class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
            "all51_resident_phase_cells_compared": True,
            "same_backing": True,
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "post_inverse_state_reset_or_canonical_reload_used": False,
            "generation_is_package_local_not_catvm_lease": True,
            "transient_buffers": "NO_RESTORATION_CLAIM",
        },
        "claim_boundary": {
            "established": [
                "FIXED_FOUR_ANGLE_TWO_COORDINATE_PAIR_CHART_HAS_NO_INJECTIVE_LIFT_ACROSS_BASE_ZERO",
                "FIXED51_PHASE_ANGLE_STATEFUL_GAUGE_CARRIER_FOR_DECLARED_INTERIOR_ENVELOPE",
                "NONTRIVIAL_LOCAL_INVERTIBLE_GAUGE_COUNTER_ROTATION_PER_GIVENS_COUPLER",
                "BASE_ZERO_GAUGE_SURVIVES_FORWARD_AND_ACTUAL_INVERSE",
                "ZERO_RETAINED_DENSE_KERNEL_AND_ZERO_INVERSE_HISTORY",
                "FINAL_ONLY_BOUNDARY_PROJECTION",
                "HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_ON_SAME_BACKING",
            ],
            "not_established": [
                "FULL_WEIGHTED_UNIT_SPHERE_GLOBAL_CHART",
                "GLOBAL_CONTINUOUS_GAUGE_SECTION",
                "RESIDUAL_CHART_ZERO_FREE_BEYOND_DECLARED_CASES",
                "LOCAL_CARTESIAN_REGISTER_FREE_COUPLING",
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
            "STATEFUL_ZERO_FIBER_REPAIR_ADDS17_PHASE_CELLS_AND_MORE_"
            "TRIGONOMETRIC_WORK_WHILE_LOCAL_CARTESIAN_RECHARTING_AND_THE_"
            "IDENTICAL_SMALLER_COMPLEX_GIVENS_RECURRENCE_REMAIN"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output)
    result = run()
    output.write_bytes(canonical_json(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
