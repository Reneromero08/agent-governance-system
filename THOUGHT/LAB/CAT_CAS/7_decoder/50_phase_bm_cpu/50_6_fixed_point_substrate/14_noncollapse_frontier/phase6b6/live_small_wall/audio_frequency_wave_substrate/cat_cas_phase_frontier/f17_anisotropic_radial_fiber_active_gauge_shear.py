#!/usr/bin/env python3
"""Exercise a causally active phase fiber on the M146 carrier.

The M146 gauge survives zero but its value does not affect the decoded
17-complex base recurrence.  This successor adds a reversible diagonal shear
whose angle is a function of the actual resident gauge.  The shear preserves
base magnitude, uses no relation table, and is interleaved with the public
phase and Givens modules.  Equal-base/different-gauge carriers can therefore
produce different final boundaries.

The matched classical path stores the identical 17 complex bases plus 17
gauge scalars.  It is intentionally retained: base-quotient insufficiency is
not evidence of a resource beyond compact classical software.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import f17_anisotropic_radial_stateful_gauge_phasor_lift as m146


m145 = m146.m145
P = 17
TAU = 2.0 * math.pi
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
SHEAR_STRENGTH = 3.0 / 16.0
RESTORATION_TOLERANCE = 5.0e-11
STATE_TOLERANCE = 3.0e-11
GAUGE_TOLERANCE = 3.0e-11
BOUNDARY_TOLERANCE = 2.0e-10
CONTROL_FLOOR = 1.0e-6
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
CLAIM = (
    "BOUNDED_FIBER_ACTIVE_GAUGE_DEPENDENT_PHASE_SHEAR_MAKES_EQUAL_BASE_"
    "DIFFERENT_GAUGE_F17_CARRIERS_BOUNDARY_DISTINGUISHABLE_WHILE_"
    "INTERLEAVING_NONCOMMUTING_PHASE_AND_LOCAL_GIVENS_MODULES_ON_FIXED51_"
    "RESIDENT_PHASE_ANGLES_ACROSS21_CASES_THROUGH_DEPTH4096_WITH_HISTORY_"
    "FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_AN_EXECUTED_"
    "MATCHED17_COMPLEX_PLUS17_GAUGE_SCALAR_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def shear_offset(shell: int, index: int, family: str) -> float:
    family_code = {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]
    exponent = (5 * shell + 3 * index + family_code) % P
    return TAU * exponent / P


def shear_angle(
    gauge: float,
    shell: int,
    index: int,
    family: str,
    *,
    strength: float = SHEAR_STRENGTH,
) -> float:
    return strength * math.sin(gauge + shear_offset(shell, index, family))


def apply_fiber_shear(
    carrier: m146.Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    strength: float = SHEAR_STRENGTH,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        gauge = float(carrier.angles[shell, 0])
        value = m146.decode_triplet(carrier.angles[shell], carrier.stats)
        angle = sign * shear_angle(
            gauge, shell, index, family, strength=strength
        )
        rotated = value * complex(math.cos(angle), math.sin(angle))
        encoded = m146.encode_triplet(rotated, gauge, carrier.stats)
        carrier.angles[shell] = encoded[:3]
    # One decoded triplet, one shear angle/phasor, one rotated complex value,
    # one gauge phasor, one residual value, one output triplet, and chart
    # scratch are conservatively covered by the M146 24-cell update ceiling.
    carrier.stats.observe_update(24)


def begin_forward(carrier: m146.Carrier, program: m145.Program) -> None:
    m146.begin_forward(carrier, program)


def forward(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
    *,
    shear_first: bool = False,
    shear_after_givens: bool = False,
    strength: float = SHEAR_STRENGTH,
) -> None:
    m146.require_owned(carrier, program, "FORWARD")
    for index in range(program.depth):
        if shear_first and shear_after_givens:
            fail("only one forward-order mutation may be selected")
        if shear_first:
            apply_fiber_shear(
                carrier, index, program.family, strength=strength
            )
            m146.apply_phase(carrier, index, program.family)
            m146.apply_givens_fourier(carrier, plan)
        elif shear_after_givens:
            m146.apply_phase(carrier, index, program.family)
            m146.apply_givens_fourier(carrier, plan)
            apply_fiber_shear(
                carrier, index, program.family, strength=strength
            )
        else:
            m146.apply_phase(carrier, index, program.family)
            apply_fiber_shear(
                carrier, index, program.family, strength=strength
            )
            m146.apply_givens_fourier(carrier, plan)
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_FIBER_ACTIVE_GAUGE_STATE_RESIDENT"


def project(carrier: m146.Carrier, program: m145.Program) -> complex:
    m146.require_owned(
        carrier, program, "FINAL_FIBER_ACTIVE_GAUGE_STATE_RESIDENT"
    )
    carrier.stage = "FINAL_STATEFUL_GAUGE_STATE_RESIDENT"
    return m146.project(carrier, program)


def inverse(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
) -> float:
    m146.require_owned(carrier, program, "PROJECTED")
    if carrier.projection_calls != 1:
        fail("fiber-active inverse requires one final projection")
    carrier.stage = "INVERSE"
    for index in range(program.depth - 1, -1, -1):
        m146.apply_givens_fourier(carrier, plan, inverse=True)
        apply_fiber_shear(carrier, index, program.family, inverse=True)
        m146.apply_phase(carrier, index, program.family, inverse=True)
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("fiber-active inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def boundary_from_state(state: np.ndarray, program: m145.Program) -> complex:
    boundary = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        angle = TAU * exponent / P
        boundary += (
            m145.shell_scale(shell)
            * complex(math.cos(angle), math.sin(angle))
            * state[shell]
        )
    return complex(boundary)


def classical_apply_phase(
    state: np.ndarray,
    gauges: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * m145.phase_exponent(shell, index, family) / P
        state[shell] *= complex(math.cos(delta), math.sin(delta))
        gauges[shell] = m146.wrap_scalar(float(gauges[shell]) + delta)


def classical_apply_shear(
    state: np.ndarray,
    gauges: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    strength: float = SHEAR_STRENGTH,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        angle = sign * shear_angle(
            float(gauges[shell]),
            shell,
            index,
            family,
            strength=strength,
        )
        state[shell] *= complex(math.cos(angle), math.sin(angle))


def classical_apply_signs(
    state: np.ndarray, gauges: np.ndarray, plan: m145.GivensPlan
) -> None:
    for shell in range(P):
        if float(plan.diagonal_signs[shell]) < 0.0:
            state[shell] = -state[shell]
            gauges[shell] = m146.wrap_scalar(
                float(gauges[shell]) + math.pi
            )


def classical_apply_givens(
    state: np.ndarray,
    gauges: np.ndarray,
    plan: m145.GivensPlan,
    *,
    inverse: bool = False,
) -> None:
    if inverse:
        for ordinal, (upper, lower) in enumerate(m145.elimination_pairs()):
            cosine = float(plan.cosine_sine[ordinal, 0])
            sine = float(plan.cosine_sine[ordinal, 1])
            theta = math.atan2(sine, cosine)
            upper_value = complex(state[upper])
            lower_value = complex(state[lower])
            state[upper] = cosine * upper_value + sine * lower_value
            state[lower] = -sine * upper_value + cosine * lower_value
            gauges[upper] = m146.wrap_scalar(float(gauges[upper]) - theta)
            gauges[lower] = m146.wrap_scalar(float(gauges[lower]) + theta)
        classical_apply_signs(state, gauges, plan)
        return
    classical_apply_signs(state, gauges, plan)
    ordinal = m146.ROTATION_COUNT - 1
    for upper, lower in m145.reverse_elimination_pairs():
        cosine = float(plan.cosine_sine[ordinal, 0])
        sine = float(plan.cosine_sine[ordinal, 1])
        theta = math.atan2(sine, cosine)
        upper_value = complex(state[upper])
        lower_value = complex(state[lower])
        state[upper] = cosine * upper_value - sine * lower_value
        state[lower] = sine * upper_value + cosine * lower_value
        gauges[upper] = m146.wrap_scalar(float(gauges[upper]) + theta)
        gauges[lower] = m146.wrap_scalar(float(gauges[lower]) - theta)
        ordinal -= 1
    if ordinal != -1:
        fail("matched gauge recurrence did not consume the public plan")


def classical_forward(
    program: m145.Program,
    plan: m145.GivensPlan,
    *,
    initial_state: np.ndarray | None = None,
    initial_gauges: np.ndarray | None = None,
) -> tuple[complex, np.ndarray, np.ndarray]:
    state = (
        np.array(m145.seed_state(), copy=True)
        if initial_state is None
        else np.array(initial_state, dtype=np.complex128, copy=True)
    )
    gauges = (
        np.asarray([m146.gauge_seed(i) for i in range(P)], dtype=np.float64)
        if initial_gauges is None
        else np.array(initial_gauges, dtype=np.float64, copy=True)
    )
    for index in range(program.depth):
        classical_apply_phase(state, gauges, index, program.family)
        classical_apply_shear(state, gauges, index, program.family)
        classical_apply_givens(state, gauges, plan)
    return boundary_from_state(state, program), state, gauges


def gauge_error(observed: np.ndarray, expected: np.ndarray) -> float:
    maximum = 0.0
    for left, right in zip(observed, expected, strict=True):
        maximum = max(
            maximum,
            math.hypot(
                math.cos(float(left)) - math.cos(float(right)),
                math.sin(float(left)) - math.sin(float(right)),
            ),
        )
    return maximum


def execute_case(
    plan: m145.GivensPlan, depth: int, family: str
) -> dict[str, Any]:
    program = m145.compile_program(depth, family)
    carrier = m146.Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    commitment = m146.state_commitment(carrier)
    boundary = project(carrier, program)
    observed_state = m146.decoded_state(carrier.angles)
    observed_gauges = np.array(carrier.angles[:, 0], copy=True)
    matched_boundary, matched_state, matched_gauges = classical_forward(
        program, plan
    )
    state_error = float(np.max(np.abs(observed_state - matched_state)))
    boundary_error = abs(boundary - matched_boundary)
    matched_gauge_error = gauge_error(observed_gauges, matched_gauges)
    if (
        state_error > STATE_TOLERANCE
        or boundary_error > BOUNDARY_TOLERANCE
        or matched_gauge_error > GAUGE_TOLERANCE
    ):
        fail("fiber-active carrier disagreed with matched scalar recurrence")
    restoration_error = inverse(carrier, program, plan)
    if carrier.backing_identity() != backing:
        fail("fiber-active carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_scalar_boundary": complex_pair(matched_boundary),
        "maximum_state_error_against_matched_scalar": state_error,
        "maximum_gauge_error_against_matched_scalar": matched_gauge_error,
        "boundary_error_against_matched_scalar": boundary_error,
        "restoration_error": restoration_error,
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "resident_phase_angle_cells": int(carrier.angles.size),
        "stats": vars(carrier.stats),
    }


def run_transaction(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program, plan)
    boundary = project(carrier, program)
    return boundary, inverse(carrier, program, plan)


def reuse_control(plan: m145.GivensPlan) -> dict[str, Any]:
    carrier = m146.Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, m145.compile_program(37, "PRIMARY"), plan)
    program = m145.compile_program(REUSE_DEPTH, "REUSE")
    restored_boundary, restoration_error = run_transaction(
        carrier, program, plan
    )
    fresh_boundary, _ = run_transaction(m146.Carrier.create(), program, plan)
    error = abs(restored_boundary - fresh_boundary)
    if error > BOUNDARY_TOLERANCE:
        fail("fiber-active unrelated reuse disagreed with fresh execution")
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
    carrier = m146.Carrier.create()
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


def raw_forward(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
) -> None:
    for index in range(program.depth):
        m146.apply_phase(carrier, index, program.family)
        apply_fiber_shear(carrier, index, program.family)
        m146.apply_givens_fourier(carrier, plan)


def raw_inverse(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
) -> None:
    for index in range(program.depth - 1, -1, -1):
        m146.apply_givens_fourier(carrier, plan, inverse=True)
        apply_fiber_shear(carrier, index, program.family, inverse=True)
        m146.apply_phase(carrier, index, program.family, inverse=True)


def equal_base_different_gauge_witness(
    plan: m145.GivensPlan,
) -> dict[str, Any]:
    program = m145.compile_program(4, "PRIMARY")
    base = m145.seed_state()
    gauge_sets = (
        np.asarray([m146.gauge_seed(i) for i in range(P)], dtype=np.float64),
        np.asarray(
            [m146.wrap_scalar(m146.gauge_seed(i) + 0.73) for i in range(P)],
            dtype=np.float64,
        ),
    )
    boundaries: list[complex] = []
    final_states: list[np.ndarray] = []
    restoration_errors: list[float] = []
    initial_arrays: list[np.ndarray] = []
    for gauges in gauge_sets:
        angles = np.empty((P, 3), dtype=np.float64)
        for shell in range(P):
            angles[shell] = m146.encode_triplet(
                complex(base[shell]), float(gauges[shell])
            )[:3]
        carrier = m146.Carrier(angles)
        initial = np.array(angles, copy=True)
        initial_arrays.append(initial)
        raw_forward(carrier, program, plan)
        state = m146.decoded_state(carrier.angles)
        final_states.append(state)
        boundaries.append(boundary_from_state(state, program))
        raw_inverse(carrier, program, plan)
        restoration_errors.append(m146.phase_cell_error(carrier.angles, initial))
    initial_base_error = float(
        np.max(
            np.abs(
                m146.decoded_state(initial_arrays[0])
                - m146.decoded_state(initial_arrays[1])
            )
        )
    )
    final_state_separation = float(
        np.max(np.abs(final_states[0] - final_states[1]))
    )
    boundary_separation = abs(boundaries[0] - boundaries[1])
    return {
        "program_depth": program.depth,
        "initial_base_state_maximum_error": initial_base_error,
        "initial_gauge_phase_separation": gauge_error(
            gauge_sets[0], gauge_sets[1]
        ),
        "final_base_state_maximum_separation": final_state_separation,
        "final_boundary_separation": boundary_separation,
        "restoration_errors": restoration_errors,
        "same_public_program": True,
        "same_initial_base_state": initial_base_error <= STATE_TOLERANCE,
        "different_actual_gauge_state": gauge_error(
            gauge_sets[0], gauge_sets[1]
        ) > CONTROL_FLOOR,
        "gauge_changes_final_boundary": boundary_separation > CONTROL_FLOOR,
        "both_actual_carriers_restored": max(restoration_errors)
        <= RESTORATION_TOLERANCE,
    }


def raw_reverse_control(
    carrier: m146.Carrier,
    program: m145.Program,
    plan: m145.GivensPlan,
    mode: str,
) -> None:
    for index in range(program.depth - 1, -1, -1):
        if mode == "REORDERED":
            apply_fiber_shear(carrier, index, program.family, inverse=True)
            m146.apply_givens_fourier(carrier, plan, inverse=True)
        else:
            m146.apply_givens_fourier(carrier, plan, inverse=True)
            if mode != "MISSING_SHEAR":
                strength = (
                    SHEAR_STRENGTH * 1.125
                    if mode == "WRONG_SHEAR"
                    else SHEAR_STRENGTH
                )
                apply_fiber_shear(
                    carrier,
                    index,
                    program.family,
                    inverse=True,
                    strength=strength,
                )
        m146.apply_phase(carrier, index, program.family, inverse=True)


def controls(plan: m145.GivensPlan) -> dict[str, bool]:
    program = m145.compile_program(4, "ALTERNATE")

    valid = m146.Carrier.create()
    begin_forward(valid, program)
    forward(valid, program, plan)
    valid_boundary = project(valid, program)
    inverse(valid, program, plan)

    disabled = m146.Carrier.create()
    begin_forward(disabled, program)
    forward(disabled, program, plan, strength=0.0)
    disabled_boundary = project(disabled, program)

    phase_order = m146.Carrier.create()
    begin_forward(phase_order, program)
    forward(phase_order, program, plan, shear_first=True)
    phase_order_boundary = project(phase_order, program)

    givens_order = m146.Carrier.create()
    begin_forward(givens_order, program)
    forward(givens_order, program, plan, shear_after_givens=True)
    givens_order_boundary = project(givens_order, program)

    reverse_errors: dict[str, float] = {}
    for mode in ("MISSING_SHEAR", "WRONG_SHEAR", "REORDERED"):
        candidate = m146.Carrier.create()
        initial = np.array(candidate.angles, copy=True)
        raw_forward(candidate, program, plan)
        raw_reverse_control(candidate, program, plan, mode)
        reverse_errors[mode] = m146.phase_cell_error(candidate.angles, initial)

    premature_rejected = False
    premature = m146.Carrier.create()
    try:
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
        m146.encode_triplet(0.95 + 0.0j, 0.2)
    except RuntimeError:
        envelope_rejected = True

    witness = equal_base_different_gauge_witness(plan)
    return {
        "phase_fiber_shear_changes_boundary": abs(
            valid_boundary - disabled_boundary
        )
        > CONTROL_FLOOR,
        "phase_and_shear_order_changes_boundary": abs(
            valid_boundary - phase_order_boundary
        )
        > CONTROL_FLOOR,
        "shear_and_givens_order_changes_boundary": abs(
            valid_boundary - givens_order_boundary
        )
        > CONTROL_FLOOR,
        "missing_shear_inverse_changes_actual_carrier": reverse_errors[
            "MISSING_SHEAR"
        ]
        > CONTROL_FLOOR,
        "wrong_shear_inverse_changes_actual_carrier": reverse_errors[
            "WRONG_SHEAR"
        ]
        > CONTROL_FLOOR,
        "reordered_inverse_changes_actual_carrier": reverse_errors["REORDERED"]
        > CONTROL_FLOOR,
        "premature_projection_rejected": premature_rejected,
        "null_carrier_rejected": null_rejected,
        "out_of_envelope_carrier_rejected": envelope_rejected,
        "equal_base_different_gauge_changes_boundary": witness[
            "gauge_changes_final_boundary"
        ],
        "equal_base_different_gauge_both_restored": witness[
            "both_actual_carriers_restored"
        ],
    }


def run() -> dict[str, Any]:
    plan = m145.GivensPlan.compile()
    cases = [
        execute_case(plan, depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_within = all(
        case["maximum_state_error_against_matched_scalar"] <= STATE_TOLERANCE
        and case["maximum_gauge_error_against_matched_scalar"]
        <= GAUGE_TOLERANCE
        and case["boundary_error_against_matched_scalar"]
        <= BOUNDARY_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_before"] == 0
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        and case["stats"]["maximum_base_magnitude"]
        <= m146.SUPPORTED_BASE_MAGNITUDE + m146.CHART_RADIUS_TOLERANCE
        and case["stats"]["residual_zero_canonicalizations"] == 0
        for case in cases
    )
    if not all_within:
        fail("one or more fiber-active cases failed the declared scope")
    witness = equal_base_different_gauge_witness(plan)
    if not all(
        witness[key]
        for key in (
            "same_initial_base_state",
            "different_actual_gauge_state",
            "gauge_changes_final_boundary",
            "both_actual_carriers_restored",
        )
    ):
        fail("equal-base different-gauge causality witness failed")
    control_results = controls(plan)
    if not all(control_results.values()):
        fail("one or more fiber-active controls failed")
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
    plan_bytes = 289 * 8
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
    native_warm = max(
        native_update_live, native_commitment_live, native_projection_live
    )
    matched_update_live = (
        resident_bytes
        + plan_bytes
        + shell_count_bytes
        + 96
        + maximum_program_bytes
    )
    matched_commitment_live = native_commitment_live
    matched_projection_live = (
        resident_bytes
        + plan_bytes
        + shell_count_bytes
        + 80
        + maximum_program_bytes
    )
    matched_warm = max(
        matched_update_live, matched_commitment_live, matched_projection_live
    )

    return {
        "schema": "CAT_CAS_F17_FIBER_ACTIVE_GAUGE_SHEAR_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m146.__file__).name,
            "sha256": hashlib.sha256(Path(m146.__file__).read_bytes()).hexdigest(),
            "used_for": "M146_51_ANGLE_CARRIER_CHART_AND_LOCAL_GIVENS_FOUNDATION",
        },
        "source_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_PHASE_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_within_predeclared_tolerances": all_within,
            "supported_base_magnitude_ceiling": m146.SUPPORTED_BASE_MAGNITUDE,
            "full_weighted_unit_sphere_supported": False,
            "public_topology_compilation_reads_final_answers": False,
        },
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "state_max_abs": STATE_TOLERANCE,
            "gauge_phasor_max_abs": GAUGE_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
        },
        "carrier_law": {
            "resident_phase_angle_cells": 51,
            "gauge_weight": m146.GAUGE_WEIGHT,
            "gauge_dependent_shear_strength": SHEAR_STRENGTH,
            "gauge_dependent_shear": "Z_J_TO_Z_J_EXP_I_LAMBDA_SIN_G_J_PLUS_PUBLIC_OFFSET",
            "base_magnitude_preserved_by_shear": True,
            "gauge_unchanged_by_shear": True,
            "same_gauge_consumed_by_successive_shears_and_givens_transport": True,
            "phase_and_shear_modules_noncommute": control_results[
                "phase_and_shear_order_changes_boundary"
            ],
            "shear_and_local_givens_modules_noncommute": control_results[
                "shear_and_givens_order_changes_boundary"
            ],
            "equal_base_different_gauge_boundary_distinguishable": witness[
                "gauge_changes_final_boundary"
            ],
            "no_relation_table_or_assignment_expansion": True,
        },
        "equal_base_different_gauge_witness": witness,
        "maximum_errors": {
            "state_against_matched_scalar": max(
                case["maximum_state_error_against_matched_scalar"]
                for case in cases
            ),
            "gauge_against_matched_scalar": max(
                case["maximum_gauge_error_against_matched_scalar"]
                for case in cases
            ),
            "boundary_against_matched_scalar": max(
                case["boundary_error_against_matched_scalar"]
                for case in cases
            ),
            "single_transaction_restoration": max(
                case["restoration_error"] for case in cases
            ),
        },
        "resource_law": {
            "resident_phase_angle_float64_cells": 51,
            "resident_phase_angle_bytes": resident_bytes,
            "retained_public_givens_plan_float64_cells": 289,
            "retained_public_givens_plan_bytes": plan_bytes,
            "retained_shell_count_bytes": shell_count_bytes,
            "maximum_named_update_bytes": maximum_update_bytes,
            "maximum_named_projection_bytes": maximum_projection_bytes,
            "maximum_named_warm_execution_live_bytes_including_program_json": native_warm,
            "maximum_named_commitment_live_bytes_including_program_json": native_commitment_live,
            "maximum_named_full_lifecycle_live_bytes": max(
                native_warm, plan.compilation_maximum_named_bytes
            ),
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "retained_complex_state_cells": 0,
            "retained_dense_kernel_cells": 0,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "local_cartesian_and_chart_scratch_float64_cells": 24,
            "fiber_shear_calls_per_step": 17,
            "fiber_shear_gauge_sine_evaluations_per_step": 17,
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
        },
        "matched_classical_recurrence": {
            "method": "IDENTICAL17_COMPLEX_BASE_PLUS17_FLOAT64_GAUGE_SCALAR_RECURRENCE",
            "executed_in_every_case": True,
            "resident_complex128_cells": 17,
            "resident_gauge_float64_cells": 17,
            "resident_float64_scalar_equivalent_cells": 51,
            "resident_bytes": resident_bytes,
            "retained_public_givens_plan_bytes": plan_bytes,
            "maximum_named_update_live_bytes_including_program_json": matched_update_live,
            "maximum_named_commitment_live_bytes_including_program_json": matched_commitment_live,
            "maximum_named_warm_execution_live_bytes_including_program_json": matched_warm,
            "avoids_phase_chart_decode_and_reencode": True,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
            "optimal_compact_classical_recurrence_claimed": False,
        },
        "restoration": {
            "class": "NUMERICAL_PHYSICAL_STATE_RESTORATION",
            "transient_buffers": "NO_RESTORATION_CLAIM",
            "all51_resident_phase_cells_compared": True,
            "same_backing": all(case["same_backing"] for case in cases),
            "snapshot_reload_used": False,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "post_inverse_state_reset_or_canonical_reload_used": False,
            "generation_is_package_local_not_catvm_lease": True,
        },
        "reuse": reuse,
        "repeated_reuse": repeated,
        "controls": control_results,
        "cases": cases,
        "claim_boundary": {
            "established": [
                "BASE_ONLY17_COMPLEX_QUOTIENT_IS_INSUFFICIENT_FOR_DECLARED_VARIABLE_GAUGE_CARRIER",
                "FIBER_ACTIVE_GAUGE_SHEAR_INTERLEAVED_WITH_NONCOMMUTING_LOCAL_GIVENS",
                "FIXED51_PHASE_ANGLE_CARRIER_FOR_DECLARED_INTERIOR_ENVELOPE",
                "FINAL_ONLY_BOUNDARY_PROJECTION",
                "HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_ON_SAME_BACKING",
            ],
            "not_established": [
                "CLASSICAL_STATE_BEYOND51_FLOAT64_SCALARS",
                "OPTIMAL_CLASSICAL_BASELINE",
                "FULL_WEIGHTED_UNIT_SPHERE_GLOBAL_CHART",
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    payload = canonical_json(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
