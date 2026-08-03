#!/usr/bin/env python3
"""Exercise a direct phase-angle shared-gauge coupling on the M146 carrier.

Unlike M147, the accepted update never decodes a complex base value and never
reconstructs a phasor chart.  A public triangular schedule uses one resident
gauge as the controller of sixteen target phase triplets, then uses the
sixteen updated target gauges as controllers of the hub.  Every primitive is
an in-place phase translation and is inverted in exact reverse order.

The matched software recurrence stores and updates the identical 51 angles.
The mechanism therefore calibrates a direct phase primitive but does not
establish a resource beyond its compact classical angle recurrence.
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

import f17_anisotropic_radial_stateful_gauge_phasor_lift as m146


m145 = m146.m145
P = 17
TAU = 2.0 * math.pi
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
COUPLING_STRENGTH = 1.0 / 32.0
REJECTED_OVERSTRONG_COUPLING = 5.0 / 32.0
RESTORATION_TOLERANCE = 2.0e-11
STATE_TOLERANCE = 2.0e-12
BOUNDARY_TOLERANCE = 2.0e-11
CONTROL_FLOOR = 1.0e-6
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
CLAIM = (
    "BOUNDED_DIRECT_PHASE_ANGLE_TRIANGULAR_SHARED_GAUGE_COUPLING_USES_ONE_"
    "RESIDENT_GAUGE_ACROSS16_TARGETS_AND_RECIPROCAL_NONCOMMUTING_HUB_"
    "UPDATES_WITHOUT_COMPLEX_DECODE_CARTESIAN_SCRATCH_CHART_RECONSTRUCTION_"
    "OR_RETAINED_GIVENS_PLAN_ON_FIXED51_PHASE_ANGLES_ACROSS21_CASES_THROUGH_"
    "DEPTH4096_WITH_HISTORY_FREE_NUMERICAL_RESTORATION_AND_REUSE_BUT_IS_"
    "BYTE_BISIMULATED_BY_AN_EXECUTED_IDENTICAL51_ANGLE_CLASSICAL_RECURRENCE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_DIRECT_ANGLE_GAUGE_COUPLING_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "topology": "PUBLIC_ROTATING_HUB16_OUT16_IN_TRIANGULAR_SCHEDULE",
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 4096:
        fail("direct-angle program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("direct-angle program family outside declared set")
    return Program(
        depth=depth,
        family=family,
        observation_quadratic=(7 * depth + 3 * len(family) + 1) % P or 1,
        observation_linear=(11 * depth + len(family) + 5) % P,
    )


def family_code(family: str) -> int:
    return {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]


def hub_index(index: int, family: str, *, mutation: int = 0) -> int:
    return (3 * index + family_code(family) + mutation) % P


def target_order(hub: int) -> Iterator[int]:
    for offset in range(1, P):
        yield (hub + offset) % P


def public_offset(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
) -> float:
    exponent = (
        5 * controller
        + 7 * target
        + 3 * index
        + 2 * layer
        + family_code(family)
    ) % P
    return TAU * exponent / P


def coupling_delta(
    control_gauge: float,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    strength: float = COUPLING_STRENGTH,
) -> float:
    return strength * math.sin(
        control_gauge
        + public_offset(controller, target, index, family, layer)
    )


def rotate_triplet_in_place(row: np.ndarray, delta: float) -> None:
    for slot in range(3):
        row[slot] = m146.wrap_scalar(float(row[slot]) + delta)


def apply_edge(
    carrier: m146.Carrier,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    strength: float = COUPLING_STRENGTH,
) -> None:
    control_gauge = float(carrier.angles[controller, 0])
    delta = coupling_delta(
        control_gauge,
        controller,
        target,
        index,
        family,
        layer,
        strength=strength,
    )
    rotate_triplet_in_place(
        carrier.angles[target], -delta if inverse else delta
    )
    carrier.stats.observe_update(4)


def apply_out_layer(
    carrier: m146.Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    strength: float = COUPLING_STRENGTH,
    hub_mutation: int = 0,
) -> None:
    hub = hub_index(index, family, mutation=hub_mutation)
    targets = list(target_order(hub))
    if inverse:
        targets.reverse()
    for target in targets:
        apply_edge(
            carrier,
            hub,
            target,
            index,
            family,
            0,
            inverse=inverse,
            strength=strength,
        )


def apply_in_layer(
    carrier: m146.Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    strength: float = COUPLING_STRENGTH,
    hub_mutation: int = 0,
) -> None:
    hub = hub_index(index, family, mutation=hub_mutation)
    controllers = list(target_order(hub))
    if inverse:
        controllers.reverse()
    for controller in controllers:
        apply_edge(
            carrier,
            controller,
            hub,
            index,
            family,
            1,
            inverse=inverse,
            strength=strength,
        )


def begin_forward(carrier: m146.Carrier, program: Program) -> None:
    if not isinstance(carrier, m146.Carrier):
        fail("null direct-angle carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or carrier.restored_error() > RESTORATION_TOLERANCE
    ):
        fail("direct-angle carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD_DIRECT_ANGLE"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def require_owned(
    carrier: m146.Carrier, program: Program, stage: str
) -> None:
    if not isinstance(carrier, m146.Carrier):
        fail("null or wrong direct-angle carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("direct-angle carrier owner or stage changed")


def apply_public_phase(
    carrier: m146.Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * m145.phase_exponent(shell, index, family) / P
        rotate_triplet_in_place(carrier.angles[shell], delta)
    carrier.stats.observe_update(4)


def forward(
    carrier: m146.Carrier,
    program: Program,
    *,
    layer_order: str = "OUT_IN",
    strength: float = COUPLING_STRENGTH,
    hub_mutation: int = 0,
) -> None:
    require_owned(carrier, program, "FORWARD_DIRECT_ANGLE")
    for index in range(program.depth):
        apply_public_phase(carrier, index, program.family)
        if layer_order == "OUT_IN":
            apply_out_layer(
                carrier,
                index,
                program.family,
                strength=strength,
                hub_mutation=hub_mutation,
            )
            apply_in_layer(
                carrier,
                index,
                program.family,
                strength=strength,
                hub_mutation=hub_mutation,
            )
        elif layer_order == "IN_OUT":
            apply_in_layer(
                carrier,
                index,
                program.family,
                strength=strength,
                hub_mutation=hub_mutation,
            )
            apply_out_layer(
                carrier,
                index,
                program.family,
                strength=strength,
                hub_mutation=hub_mutation,
            )
        else:
            fail("unknown direct-angle layer order")
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_DIRECT_ANGLE_STATE_RESIDENT"


def project(carrier: m146.Carrier, program: Program) -> complex:
    require_owned(carrier, program, "FINAL_DIRECT_ANGLE_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("direct-angle final projection order changed")
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
            m146.GAUGE_WEIGHT * math.cos(gauge)
            + 0.5
            * m146.RESIDUAL_WEIGHT
            * (math.cos(left) + math.cos(right))
        )
        boundary_imag += scale * (
            m146.GAUGE_WEIGHT * math.sin(gauge)
            + 0.5
            * m146.RESIDUAL_WEIGHT
            * (math.sin(left) + math.sin(right))
        )
        carrier.stats.projection_phasor_cosine_evaluations += 3
        carrier.stats.projection_phasor_sine_evaluations += 3
    carrier.stats.observe_projection(14)
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED_DIRECT_ANGLE"
    return complex(boundary_real, boundary_imag)


def inverse(carrier: m146.Carrier, program: Program) -> float:
    require_owned(carrier, program, "PROJECTED_DIRECT_ANGLE")
    if carrier.projection_calls != 1:
        fail("direct-angle inverse requires one final projection")
    carrier.stage = "INVERSE_DIRECT_ANGLE"
    for index in range(program.depth - 1, -1, -1):
        apply_in_layer(carrier, index, program.family, inverse=True)
        apply_out_layer(carrier, index, program.family, inverse=True)
        apply_public_phase(carrier, index, program.family, inverse=True)
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    restoration_error = carrier.restored_error()
    if restoration_error > RESTORATION_TOLERANCE:
        fail("direct-angle inverse exceeded restoration tolerance")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1
    return restoration_error


def reference_rotate(row: np.ndarray, delta: float) -> None:
    row[:] = [m146.wrap_scalar(float(value) + delta) for value in row]


def reference_phase(
    angles: np.ndarray, index: int, family: str, *, inverse: bool = False
) -> None:
    sign = -1.0 if inverse else 1.0
    for shell in range(P):
        delta = sign * TAU * m145.phase_exponent(shell, index, family) / P
        reference_rotate(angles[shell], delta)


def reference_edge(
    angles: np.ndarray,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    strength: float = COUPLING_STRENGTH,
) -> None:
    delta = coupling_delta(
        float(angles[controller, 0]),
        controller,
        target,
        index,
        family,
        layer,
        strength=strength,
    )
    reference_rotate(angles[target], -delta if inverse else delta)


def reference_layer(
    angles: np.ndarray,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    strength: float = COUPLING_STRENGTH,
) -> None:
    hub = hub_index(index, family)
    actors = list(target_order(hub))
    if inverse:
        actors.reverse()
    for actor in actors:
        controller, target = (hub, actor) if layer == 0 else (actor, hub)
        reference_edge(
            angles,
            controller,
            target,
            index,
            family,
            layer,
            inverse=inverse,
            strength=strength,
        )


def reference_forward(program: Program) -> np.ndarray:
    angles = np.array(m146.seed_angles(), copy=True)
    for index in range(program.depth):
        reference_phase(angles, index, program.family)
        reference_layer(angles, index, program.family, 0)
        reference_layer(angles, index, program.family, 1)
    return angles


def boundary_from_angles(angles: np.ndarray, program: Program) -> complex:
    value = 0.0j
    for shell in range(P):
        exponent = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        observation = TAU * exponent / P
        value += (
            m145.shell_scale(shell)
            * complex(math.cos(observation), math.sin(observation))
            * m146.decode_triplet(angles[shell])
        )
    return complex(value)


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = m146.Carrier.create()
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment = m146.state_commitment(carrier)
    final_angles = np.array(carrier.angles, copy=True)
    boundary = project(carrier, program)
    reference = reference_forward(program)
    reference_boundary = boundary_from_angles(reference, program)
    byte_equal = final_angles.tobytes() == reference.tobytes()
    state_error = m146.phase_cell_error(final_angles, reference)
    boundary_error = abs(boundary - reference_boundary)
    if not byte_equal or state_error != 0.0 or boundary_error > BOUNDARY_TOLERANCE:
        fail("direct-angle execution disagreed with identical recurrence")
    restoration_error = inverse(carrier, program)
    if carrier.backing_identity() != backing:
        fail("direct-angle carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": complex_pair(boundary),
        "matched_identical_angle_boundary": complex_pair(reference_boundary),
        "final_angle_bytes_identical_to_matched_recurrence": byte_equal,
        "maximum_phase_cell_error_against_matched_recurrence": state_error,
        "boundary_error_against_matched_recurrence": boundary_error,
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
    carrier: m146.Carrier, program: Program
) -> tuple[complex, float]:
    begin_forward(carrier, program)
    forward(carrier, program)
    boundary = project(carrier, program)
    return boundary, inverse(carrier, program)


def reuse_control() -> dict[str, Any]:
    carrier = m146.Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, compile_program(37, "PRIMARY"))
    program = compile_program(REUSE_DEPTH, "REUSE")
    restored_boundary, restoration_error = run_transaction(carrier, program)
    fresh_boundary, _ = run_transaction(m146.Carrier.create(), program)
    error = abs(restored_boundary - fresh_boundary)
    if error > BOUNDARY_TOLERANCE:
        fail("direct-angle unrelated reuse disagreed with fresh execution")
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "fresh_restored_boundary_error": error,
        "restoration_error": restoration_error,
        "same_original_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control() -> dict[str, Any]:
    carrier = m146.Carrier.create()
    backing = carrier.backing_identity()
    program = compile_program(REPEATED_REUSE_DEPTH, "ALTERNATE")
    maximum = 0.0
    for _ in range(REPEATED_REUSE_CYCLES):
        _, error = run_transaction(carrier, program)
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
    program: Program,
    *,
    strength: float = COUPLING_STRENGTH,
) -> None:
    for index in range(program.depth):
        apply_public_phase(carrier, index, program.family)
        apply_out_layer(carrier, index, program.family, strength=strength)
        apply_in_layer(carrier, index, program.family, strength=strength)


def raw_inverse(
    carrier: m146.Carrier,
    program: Program,
    *,
    strength: float = COUPLING_STRENGTH,
) -> None:
    for index in range(program.depth - 1, -1, -1):
        apply_in_layer(
            carrier,
            index,
            program.family,
            inverse=True,
            strength=strength,
        )
        apply_out_layer(
            carrier,
            index,
            program.family,
            inverse=True,
            strength=strength,
        )
        apply_public_phase(carrier, index, program.family, inverse=True)


def inverse_conditioning_control() -> dict[str, Any]:
    program = compile_program(4096, "PRIMARY")
    errors: dict[str, float] = {}
    for label, strength in (
        ("accepted_1_over_32", COUPLING_STRENGTH),
        ("rejected_5_over_32", REJECTED_OVERSTRONG_COUPLING),
    ):
        carrier = m146.Carrier.create()
        initial = np.array(carrier.angles, copy=True)
        raw_forward(carrier, program, strength=strength)
        raw_inverse(carrier, program, strength=strength)
        errors[label] = m146.phase_cell_error(carrier.angles, initial)
    return {
        "depth": program.depth,
        "family": program.family,
        "accepted_strength": COUPLING_STRENGTH,
        "accepted_restoration_error": errors["accepted_1_over_32"],
        "accepted_within_tolerance": errors["accepted_1_over_32"]
        <= RESTORATION_TOLERANCE,
        "rejected_overstrong_strength": REJECTED_OVERSTRONG_COUPLING,
        "rejected_overstrong_restoration_error": errors[
            "rejected_5_over_32"
        ],
        "rejected_overstrong_exceeds_tolerance": errors[
            "rejected_5_over_32"
        ]
        > RESTORATION_TOLERANCE,
    }


def equal_base_different_gauge_witness() -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    base = m145.seed_state()
    gauge_sets = (
        np.asarray([m146.gauge_seed(i) for i in range(P)], dtype=np.float64),
        np.asarray(
            [m146.wrap_scalar(m146.gauge_seed(i) + 0.73) for i in range(P)],
            dtype=np.float64,
        ),
    )
    initial_arrays: list[np.ndarray] = []
    final_states: list[np.ndarray] = []
    boundaries: list[complex] = []
    restoration_errors: list[float] = []
    for gauges in gauge_sets:
        angles = np.empty((P, 3), dtype=np.float64)
        for shell in range(P):
            angles[shell] = m146.encode_triplet(
                complex(base[shell]), float(gauges[shell])
            )[:3]
        initial = np.array(angles, copy=True)
        initial_arrays.append(initial)
        carrier = m146.Carrier(angles)
        raw_forward(carrier, program)
        state = m146.decoded_state(carrier.angles)
        final_states.append(state)
        boundaries.append(boundary_from_angles(carrier.angles, program))
        raw_inverse(carrier, program)
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
    gauge_separation = m146.phase_cell_error(
        initial_arrays[0][:, :1], initial_arrays[1][:, :1]
    )
    return {
        "program_depth": program.depth,
        "initial_base_state_maximum_error": initial_base_error,
        "initial_gauge_phase_separation": gauge_separation,
        "final_base_state_maximum_separation": final_state_separation,
        "final_boundary_separation": boundary_separation,
        "restoration_errors": restoration_errors,
        "same_public_program": True,
        "same_initial_base_state": initial_base_error <= STATE_TOLERANCE,
        "different_actual_gauge_state": gauge_separation > CONTROL_FLOOR,
        "gauge_changes_final_boundary": boundary_separation > CONTROL_FLOOR,
        "both_actual_carriers_restored": max(restoration_errors)
        <= RESTORATION_TOLERANCE,
    }


def raw_reverse_control(
    carrier: m146.Carrier, program: Program, mode: str
) -> None:
    for index in range(program.depth - 1, -1, -1):
        if mode == "REORDERED":
            apply_out_layer(carrier, index, program.family, inverse=True)
            apply_in_layer(carrier, index, program.family, inverse=True)
        else:
            apply_in_layer(carrier, index, program.family, inverse=True)
            if mode != "MISSING_OUT":
                strength = (
                    COUPLING_STRENGTH * 1.125
                    if mode == "WRONG_OUT"
                    else COUPLING_STRENGTH
                )
                apply_out_layer(
                    carrier,
                    index,
                    program.family,
                    inverse=True,
                    strength=strength,
                )
        apply_public_phase(carrier, index, program.family, inverse=True)


def controls() -> dict[str, bool]:
    program = compile_program(4, "ALTERNATE")
    valid = m146.Carrier.create()
    begin_forward(valid, program)
    forward(valid, program)
    valid_boundary = project(valid, program)
    inverse(valid, program)

    disabled = m146.Carrier.create()
    begin_forward(disabled, program)
    forward(disabled, program, strength=0.0)
    disabled_boundary = project(disabled, program)

    order = m146.Carrier.create()
    begin_forward(order, program)
    forward(order, program, layer_order="IN_OUT")
    order_boundary = project(order, program)

    topology = m146.Carrier.create()
    begin_forward(topology, program)
    forward(topology, program, hub_mutation=1)
    topology_boundary = project(topology, program)

    reverse_errors: dict[str, float] = {}
    for mode in ("MISSING_OUT", "WRONG_OUT", "REORDERED"):
        carrier = m146.Carrier.create()
        initial = np.array(carrier.angles, copy=True)
        raw_forward(carrier, program)
        raw_reverse_control(carrier, program, mode)
        reverse_errors[mode] = m146.phase_cell_error(carrier.angles, initial)

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

    witness = equal_base_different_gauge_witness()
    return {
        "direct_angle_coupling_changes_boundary": abs(
            valid_boundary - disabled_boundary
        )
        > CONTROL_FLOOR,
        "out_in_layer_order_changes_boundary": abs(
            valid_boundary - order_boundary
        )
        > CONTROL_FLOOR,
        "public_hub_topology_mutation_changes_boundary": abs(
            valid_boundary - topology_boundary
        )
        > CONTROL_FLOOR,
        "missing_out_layer_inverse_changes_actual_carrier": reverse_errors[
            "MISSING_OUT"
        ]
        > CONTROL_FLOOR,
        "wrong_out_layer_inverse_changes_actual_carrier": reverse_errors[
            "WRONG_OUT"
        ]
        > CONTROL_FLOOR,
        "reordered_inverse_changes_actual_carrier": reverse_errors["REORDERED"]
        > CONTROL_FLOOR,
        "premature_projection_rejected": premature_rejected,
        "null_carrier_rejected": null_rejected,
        "equal_base_different_gauge_changes_boundary": witness[
            "gauge_changes_final_boundary"
        ],
        "equal_base_different_gauge_both_restored": witness[
            "both_actual_carriers_restored"
        ],
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_within = all(
        case["final_angle_bytes_identical_to_matched_recurrence"]
        and case["maximum_phase_cell_error_against_matched_recurrence"] == 0.0
        and case["boundary_error_against_matched_recurrence"]
        <= BOUNDARY_TOLERANCE
        and case["restoration_error"] <= RESTORATION_TOLERANCE
        and case["same_backing"]
        and case["restoration_generation_before"] == 0
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        for case in cases
    )
    if not all_within:
        fail("one or more direct-angle cases failed the declared scope")
    witness = equal_base_different_gauge_witness()
    if not all(
        witness[key]
        for key in (
            "same_initial_base_state",
            "different_actual_gauge_state",
            "gauge_changes_final_boundary",
            "both_actual_carriers_restored",
        )
    ):
        fail("direct-angle causality witness failed")
    control_results = controls()
    if not all(control_results.values()):
        fail("one or more direct-angle controls failed")
    reuse = reuse_control()
    repeated = repeated_reuse_control()
    inverse_conditioning = inverse_conditioning_control()
    if not (
        inverse_conditioning["accepted_within_tolerance"]
        and inverse_conditioning["rejected_overstrong_exceeds_tolerance"]
    ):
        fail("direct-angle inverse conditioning control failed")

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
    commitment_live = resident_bytes + 96 + 64 + maximum_program_bytes
    update_live = resident_bytes + maximum_update_bytes + maximum_program_bytes
    projection_live = (
        resident_bytes + maximum_projection_bytes + maximum_program_bytes
    )
    warm_live = max(commitment_live, update_live, projection_live)

    return {
        "schema": "CAT_CAS_F17_DIRECT_ANGLE_TRIANGULAR_GAUGE_COUPLING_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m146.__file__).name,
            "sha256": hashlib.sha256(Path(m146.__file__).read_bytes()).hexdigest(),
            "used_for": "M146_51_ANGLE_CARRIER_SEED_OWNERSHIP_AND_FINAL_PROJECTION_FOUNDATION",
        },
        "source_scope": "LINUX_DIRECT_PROCESS_NUMERICAL_PHASE_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_within_predeclared_tolerances": all_within,
            "public_topology_compilation_reads_final_answers": False,
            "full_weighted_unit_sphere_supported": False,
        },
        "predeclared_tolerances": {
            "restoration_phasor_max_abs": RESTORATION_TOLERANCE,
            "matched_phase_cell_max_abs": STATE_TOLERANCE,
            "boundary_max_abs": BOUNDARY_TOLERANCE,
        },
        "carrier_law": {
            "resident_phase_angle_cells": 51,
            "coupling_strength": COUPLING_STRENGTH,
            "rejected_overstrong_coupling": REJECTED_OVERSTRONG_COUPLING,
            "out_layer_edges_per_step": 16,
            "in_layer_edges_per_step": 16,
            "shared_hub_gauge_consumers_per_out_layer": 16,
            "primitive": "TARGET_TRIPLET_COMMON_PHASE_TRANSLATION_BY_SINE_OF_RESIDENT_CONTROLLER_GAUGE_PLUS_PUBLIC_OFFSET",
            "complex_base_decode_per_update": False,
            "cartesian_update_scratch_cells": 0,
            "chart_reconstruction_per_update": False,
            "retained_givens_plan_cells": 0,
            "base_magnitude_preserved_per_edge": True,
            "phase_only_update": True,
            "out_and_in_layers_noncommute": control_results[
                "out_in_layer_order_changes_boundary"
            ],
            "equal_base_different_gauge_boundary_distinguishable": witness[
                "gauge_changes_final_boundary"
            ],
            "no_relation_table_or_assignment_expansion": True,
        },
        "equal_base_different_gauge_witness": witness,
        "maximum_errors": {
            "phase_cell_against_identical_angle_recurrence": max(
                case["maximum_phase_cell_error_against_matched_recurrence"]
                for case in cases
            ),
            "boundary_against_identical_angle_recurrence": max(
                case["boundary_error_against_matched_recurrence"]
                for case in cases
            ),
            "single_transaction_restoration": max(
                case["restoration_error"] for case in cases
            ),
        },
        "resource_law": {
            "resident_phase_angle_float64_cells": 51,
            "resident_phase_angle_bytes": resident_bytes,
            "retained_public_plan_cells": 0,
            "retained_public_plan_bytes": 0,
            "maximum_named_update_float64_scratch_cells": 4,
            "maximum_named_update_bytes": maximum_update_bytes,
            "maximum_named_projection_bytes": maximum_projection_bytes,
            "maximum_public_program_json_bytes": maximum_program_bytes,
            "maximum_named_commitment_live_bytes_including_program_json": commitment_live,
            "maximum_named_warm_execution_live_bytes_including_program_json": warm_live,
            "maximum_named_full_lifecycle_live_bytes": warm_live,
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "retained_complex_state_cells": 0,
            "retained_dense_kernel_cells": 0,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "complex_decode_calls_per_update": 0,
            "chart_calls_per_update": 0,
            "coupling_sine_evaluations_per_step": 32,
            "phase_module_trigonometric_evaluations_per_step": 0,
            "projection_phasor_cosine_evaluations": 51,
            "projection_phasor_sine_evaluations": 51,
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
        },
        "matched_classical_recurrence": {
            "method": "IDENTICAL51_FLOAT64_ANGLE_TRIANGULAR_RECURRENCE",
            "executed_in_every_case": True,
            "final_angle_bytes_identical_in_every_case": all(
                case["final_angle_bytes_identical_to_matched_recurrence"]
                for case in cases
            ),
            "resident_float64_cells": 51,
            "resident_bytes": resident_bytes,
            "retained_public_plan_bytes": 0,
            "maximum_named_warm_execution_live_bytes_including_program_json": warm_live,
            "same_update_and_projection_operation_counts": True,
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
        "inverse_conditioning_control": inverse_conditioning,
        "controls": control_results,
        "cases": cases,
        "claim_boundary": {
            "established": [
                "DIRECT_PHASE_ANGLE_SHARED_GAUGE_COUPLING_WITHOUT_COMPLEX_DECODE_OR_RECHART",
                "ONE_RESIDENT_GAUGE_CONTROLS16_TARGET_PHASE_TRIPLETS",
                "NONCOMMUTING_RECIPROCAL_OUT_AND_IN_LAYERS",
                "BASE_ONLY17_COMPLEX_QUOTIENT_INSUFFICIENT_FOR_DECLARED_VARIABLE_GAUGE_CARRIER",
                "FIXED51_PHASE_ANGLE_CARRIER_WITH_FINAL_ONLY_PROJECTION_RESTORATION_AND_REUSE",
            ],
            "not_established": [
                "RESOURCE_BEYOND_IDENTICAL51_ANGLE_RECURRENCE",
                "OPTIMAL_CLASSICAL_BASELINE",
                "M147_RADIAL_GIVENS_TASK_EQUIVALENCE",
                "GENERAL_RELATIONAL_CONTRACTION",
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
