#!/usr/bin/env python3
"""Exact C17 phase-fiber port convolution on the M148 topology.

The carrier stores every phase factor as a one-hot coordinate in the group
algebra of C17.  A rotating hub phase is consumed directly by sixteen target
triplets through cyclic convolution; the updated target phases then control
reciprocal hub convolutions.  The accepted path never decodes a resident port
to an exponent, angle, truth table, or assignment list.

An executed comparison stores the same phases as 51 residues and applies the
identical public shear recurrence.  The comparison is intentionally the
strong compact baseline: this package tests whether direct exact phase-orbit
custody supplies anything beyond the residue quotient, not whether it beats
an expanded classical representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import f17_anisotropic_radial_local_polar_givens_coupling as m145


P = 17
CYCLIC_INDEX = np.asarray(
    [[(output - source) % P for source in range(P)] for output in range(P)],
    dtype=np.uint8,
)
DEPTHS = (1, 4, 16, 64, 256, 1024, 4096)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
REUSE_DEPTH = 1537
REPEATED_REUSE_DEPTH = 64
REPEATED_REUSE_CYCLES = 100
COMPLEX_UNIT_NORM_FAILURE_TOLERANCE = 1.0e-8
CLAIM = (
    "BOUNDED_EXACT_C17_ONE_HOT_PHASE_FIBER_PORT_CONVOLUTION_CONSUMES_ONE_"
    "RESIDENT_PORT_ACROSS16_TARGETS_AND_RECIPROCAL_NONCOMMUTING_UPDATES_"
    "WITHOUT_SCALAR_ANGLE_OR_EXPONENT_READOUT_RELATION_TABLE_ASSIGNMENT_"
    "EXPANSION_OR_RETAINED_PLAN_ON_FIXED51_BY17_PHASE_ORBIT_COORDINATES_"
    "THROUGH_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_"
    "AN_EXECUTED_IDENTICAL51_RESIDUE_RECURRENCE_WITH17X_LESS_RESIDENT_STATE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def one_hot(exponent: int) -> np.ndarray:
    value = np.zeros(P, dtype=np.uint8)
    value[exponent % P] = 1
    return value


def valid_phase(value: np.ndarray) -> bool:
    return (
        value.shape == (P,)
        and value.dtype == np.uint8
        and int(np.sum(value, dtype=np.uint16)) == 1
        and bool(np.all((value == 0) | (value == 1)))
    )


def phase_inverse(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    result[0] = value[0]
    for index in range(1, P):
        result[index] = value[P - index]
    return result


@dataclass
class Stats:
    forward_steps: int = 0
    inverse_steps: int = 0
    cyclic_convolutions: int = 0
    convolution_coordinate_multiplications: int = 0
    convolution_coordinate_additions: int = 0
    resident_port_scalar_reads: int = 0
    resident_port_exponent_decodes: int = 0
    final_boundary_coordinate_additions: int = 0
    maximum_named_update_uint8_cells: int = 0
    maximum_named_projection_integer_cells: int = 0

    def observe_convolution(
        self, logical_convolutions: int, live_uint8_cells: int
    ) -> None:
        self.cyclic_convolutions += logical_convolutions
        self.convolution_coordinate_multiplications += (
            logical_convolutions * P * P
        )
        self.convolution_coordinate_additions += logical_convolutions * P * P
        self.maximum_named_update_uint8_cells = max(
            self.maximum_named_update_uint8_cells, live_uint8_cells
        )


def convolve_rows(
    left_rows: np.ndarray,
    right: np.ndarray,
    stats: Stats | None = None,
) -> np.ndarray:
    rows = np.asarray(left_rows, dtype=np.uint8)
    if (
        rows.ndim != 2
        or rows.shape[1] != P
        or right.shape != (P,)
        or right.dtype != np.uint8
    ):
        fail("cyclic phase convolution requires C17 orbit rows and one factor")
    matrix = right[CYCLIC_INDEX]
    result = rows @ matrix.T
    if stats is not None:
        stats.observe_convolution(
            rows.shape[0],
            P * P + rows.size + result.size,
        )
    return result


def convolve(
    left: np.ndarray,
    right: np.ndarray,
    stats: Stats | None = None,
) -> np.ndarray:
    if left.shape != (P,) or left.dtype != np.uint8:
        fail("cyclic phase convolution requires a one-hot left factor")
    return convolve_rows(left.reshape(1, P), right, stats)[0]


def seed_exponents() -> np.ndarray:
    result = np.empty((P, 3), dtype=np.uint8)
    for shell in range(P):
        result[shell] = (
            (5 * shell + 3) % P,
            (7 * shell * shell + 2 * shell + 1) % P,
            (11 * shell * shell + 4 * shell + 6) % P,
        )
    return result


def phases_from_exponents(exponents: np.ndarray) -> np.ndarray:
    phases = np.zeros((P, 3, P), dtype=np.uint8)
    for shell in range(P):
        for slot in range(3):
            phases[shell, slot] = one_hot(int(exponents[shell, slot]))
    return phases


def seed_phases() -> np.ndarray:
    return phases_from_exponents(seed_exponents())


def carrier_valid(phases: np.ndarray) -> bool:
    return phases.shape == (P, 3, P) and all(
        valid_phase(phases[shell, slot])
        for shell in range(P)
        for slot in range(3)
    )


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F17_EXACT_C17_FIBER_PORT_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "topology": "PUBLIC_ROTATING_HUB16_OUT16_IN_TRIANGULAR_SCHEDULE",
            "port_type": "ONE_HOT_C17_GROUP_ALGEBRA_PHASE",
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 4096:
        fail("C17 convolution program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("C17 convolution program family outside declared set")
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


def public_offset_exponent(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    mutation: int = 0,
) -> int:
    return (
        5 * controller
        + 7 * target
        + 3 * index
        + 2 * layer
        + family_code(family)
        + mutation
    ) % P


def shell_weight(shell: int) -> int:
    return 1 + ((shell * shell + 3 * shell + 2) % 7)


@dataclass
class Carrier:
    phases: np.ndarray
    seed: np.ndarray
    stage: str = "RESTORED"
    active_program: str | None = None
    forward_index: int = 0
    inverse_index: int = 0
    projection_calls: int = 0
    restoration_generation: int = 0
    stats: Stats = field(default_factory=Stats)

    @classmethod
    def create(cls) -> "Carrier":
        phases = seed_phases()
        return cls(phases=phases, seed=np.array(phases, copy=True))

    def backing_identity(self) -> tuple[int, int]:
        return id(self), int(self.phases.__array_interface__["data"][0])

    def exact_restored(self) -> bool:
        return bool(np.array_equal(self.phases, self.seed))

    def state_commitment(self) -> str:
        return hashlib.sha256(self.phases.tobytes(order="C")).hexdigest()


def reset_stats(carrier: Carrier) -> None:
    carrier.stats = Stats()


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null C17 phase carrier")
    if (
        carrier.stage != "RESTORED"
        or carrier.active_program is not None
        or not carrier.exact_restored()
        or not carrier_valid(carrier.phases)
    ):
        fail("C17 phase carrier is not restored")
    carrier.active_program = program.fingerprint()
    carrier.stage = "FORWARD_C17_CONVOLUTION"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong C17 phase carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("C17 phase carrier owner or stage changed")


def apply_public_phase(
    carrier: Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    for shell in range(P):
        exponent = m145.phase_exponent(shell, index, family)
        factor = one_hot(-exponent if inverse else exponent)
        carrier.phases[shell] = convolve_rows(
            carrier.phases[shell], factor, carrier.stats
        )


def apply_edge(
    carrier: Carrier,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    offset_mutation: int = 0,
) -> None:
    offset = one_hot(
        public_offset_exponent(
            controller,
            target,
            index,
            family,
            layer,
            mutation=offset_mutation,
        )
    )
    if port_enabled:
        factor = convolve(carrier.phases[controller, 0], offset, carrier.stats)
    else:
        factor = offset
    if inverse:
        factor = phase_inverse(factor)
    carrier.phases[target] = convolve_rows(
        carrier.phases[target], factor, carrier.stats
    )


def apply_out_layer(
    carrier: Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
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
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
        )


def apply_in_layer(
    carrier: Carrier,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
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
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
        )


def forward(
    carrier: Carrier,
    program: Program,
    *,
    layer_order: str = "OUT_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
) -> None:
    require_owned(carrier, program, "FORWARD_C17_CONVOLUTION")
    for index in range(program.depth):
        apply_public_phase(carrier, index, program.family)
        if layer_order == "OUT_IN":
            apply_out_layer(
                carrier,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
            )
            apply_in_layer(
                carrier,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
            )
        elif layer_order == "IN_OUT":
            apply_in_layer(
                carrier,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
            )
            apply_out_layer(
                carrier,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
            )
        else:
            fail("unknown C17 phase layer order")
        carrier.forward_index = index + 1
        carrier.stats.forward_steps += 1
    carrier.stage = "FINAL_C17_PHASE_STATE_RESIDENT"


def project_port(carrier: Carrier, program: Program) -> tuple[int, ...]:
    del carrier, program
    fail("resident C17 phase-fiber port is not a projectable boundary")


def canonical_cyclotomic_boundary(coefficients: list[int]) -> tuple[int, ...]:
    if len(coefficients) != P:
        fail("C17 boundary coefficient width changed")
    tail = coefficients[P - 1]
    return tuple(value - tail for value in coefficients[: P - 1])


def boundary_from_phase_orbit(
    phases: np.ndarray, program: Program
) -> tuple[int, ...]:
    if not carrier_valid(phases):
        fail("C17 boundary requires one-hot resident phase factors")
    coefficients = [0 for _ in range(P)]
    for shell in range(P):
        observation = one_hot(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        for slot, slot_weight in enumerate((3, 1, 1)):
            observed = convolve(phases[shell, slot], observation)
            weight = shell_weight(shell) * slot_weight
            for coordinate in range(P):
                coefficients[coordinate] += weight * int(observed[coordinate])
    return canonical_cyclotomic_boundary(coefficients)


def project(carrier: Carrier, program: Program) -> tuple[int, ...]:
    require_owned(carrier, program, "FINAL_C17_PHASE_STATE_RESIDENT")
    if carrier.forward_index != program.depth or carrier.projection_calls:
        fail("C17 final projection order changed")
    if not carrier_valid(carrier.phases):
        fail("C17 final state left the one-hot phase orbit")
    coefficients = [0 for _ in range(P)]
    slot_weights = (3, 1, 1)
    for shell in range(P):
        observation = one_hot(
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        )
        for slot, slot_weight in enumerate(slot_weights):
            observed = convolve(
                carrier.phases[shell, slot], observation, carrier.stats
            )
            weight = shell_weight(shell) * slot_weight
            for coordinate in range(P):
                coefficients[coordinate] += weight * int(observed[coordinate])
                carrier.stats.final_boundary_coordinate_additions += 1
    carrier.stats.maximum_named_projection_integer_cells = P
    carrier.projection_calls = 1
    carrier.stage = "PROJECTED_C17_PHASE_BOUNDARY"
    return canonical_cyclotomic_boundary(coefficients)


def inverse(carrier: Carrier, program: Program) -> None:
    require_owned(carrier, program, "PROJECTED_C17_PHASE_BOUNDARY")
    if carrier.projection_calls != 1:
        fail("C17 inverse requires one final projection")
    carrier.stage = "INVERSE_C17_CONVOLUTION"
    for index in range(program.depth - 1, -1, -1):
        apply_in_layer(carrier, index, program.family, inverse=True)
        apply_out_layer(carrier, index, program.family, inverse=True)
        apply_public_phase(carrier, index, program.family, inverse=True)
        carrier.inverse_index += 1
        carrier.stats.inverse_steps += 1
    if not carrier.exact_restored():
        fail("C17 phase inverse failed exact restoration")
    carrier.active_program = None
    carrier.stage = "RESTORED"
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.restoration_generation += 1


def reference_public(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
) -> None:
    sign = -1 if inverse else 1
    for shell in range(P):
        state[shell] = (
            state[shell] + sign * m145.phase_exponent(shell, index, family)
        ) % P


def reference_edge(
    state: np.ndarray,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
) -> None:
    factor = (
        int(state[controller, 0])
        + public_offset_exponent(controller, target, index, family, layer)
    ) % P
    sign = -1 if inverse else 1
    state[target] = (state[target] + sign * factor) % P


def reference_layer(
    state: np.ndarray,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
) -> None:
    hub = hub_index(index, family)
    actors = list(target_order(hub))
    if inverse:
        actors.reverse()
    for actor in actors:
        controller, target = (hub, actor) if layer == 0 else (actor, hub)
        reference_edge(
            state,
            controller,
            target,
            index,
            family,
            layer,
            inverse=inverse,
        )


def reference_forward(program: Program, seed: np.ndarray | None = None) -> np.ndarray:
    state = np.array(seed_exponents() if seed is None else seed, copy=True)
    for index in range(program.depth):
        reference_public(state, index, program.family)
        reference_layer(state, index, program.family, 0)
        reference_layer(state, index, program.family, 1)
    return state.astype(np.uint8)


def reference_boundary(state: np.ndarray, program: Program) -> tuple[int, ...]:
    coefficients = [0 for _ in range(P)]
    slot_weights = (3, 1, 1)
    for shell in range(P):
        observation = (
            program.observation_quadratic * shell * shell
            + program.observation_linear * shell
        ) % P
        for slot, slot_weight in enumerate(slot_weights):
            coordinate = (int(state[shell, slot]) + observation) % P
            coefficients[coordinate] += shell_weight(shell) * slot_weight
    return canonical_cyclotomic_boundary(coefficients)


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.create()
    reset_stats(carrier)
    backing = carrier.backing_identity()
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program)
    commitment = carrier.state_commitment()
    final_phases = np.array(carrier.phases, copy=True)
    boundary = project(carrier, program)
    reference = reference_forward(program)
    reference_phases = phases_from_exponents(reference)
    reference_result = reference_boundary(reference, program)
    byte_equal = final_phases.tobytes(order="C") == reference_phases.tobytes(
        order="C"
    )
    if not byte_equal or boundary != reference_result:
        fail("C17 phase execution disagreed with compact residue recurrence")
    inverse(carrier, program)
    if carrier.backing_identity() != backing:
        fail("C17 phase carrier backing changed")
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "final_state_commitment": commitment,
        "final_boundary": list(boundary),
        "matched_residue_boundary": list(reference_result),
        "final_phase_orbit_bytes_identical_to_expanded_matched_recurrence": byte_equal,
        "boundary_identical_to_matched_recurrence": boundary == reference_result,
        "exact_restoration": carrier.exact_restored(),
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "resident_phase_orbit_uint8_cells": int(carrier.phases.size),
        "stats": vars(carrier.stats),
    }


def run_transaction(
    carrier: Carrier, program: Program
) -> tuple[tuple[int, ...], None]:
    reset_stats(carrier)
    begin_forward(carrier, program)
    forward(carrier, program)
    boundary = project(carrier, program)
    inverse(carrier, program)
    return boundary, None


def reuse_control() -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    run_transaction(carrier, compile_program(37, "PRIMARY"))
    program = compile_program(REUSE_DEPTH, "REUSE")
    restored_boundary, _ = run_transaction(carrier, program)
    fresh_boundary, _ = run_transaction(Carrier.create(), program)
    if restored_boundary != fresh_boundary:
        fail("restored C17 phase carrier reuse disagreed with fresh execution")
    return {
        "unrelated_reuse_depth": REUSE_DEPTH,
        "fresh_restored_boundary_equal": restored_boundary == fresh_boundary,
        "same_original_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def repeated_reuse_control() -> dict[str, Any]:
    carrier = Carrier.create()
    backing = carrier.backing_identity()
    program = compile_program(REPEATED_REUSE_DEPTH, "ALTERNATE")
    for _ in range(REPEATED_REUSE_CYCLES):
        run_transaction(carrier, program)
    return {
        "cycles": REPEATED_REUSE_CYCLES,
        "depth_per_cycle": REPEATED_REUSE_DEPTH,
        "exact_restoration_every_cycle": carrier.exact_restored(),
        "same_backing": carrier.backing_identity() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
    }


def raw_forward(carrier: Carrier, program: Program) -> None:
    for index in range(program.depth):
        apply_public_phase(carrier, index, program.family)
        apply_out_layer(carrier, index, program.family)
        apply_in_layer(carrier, index, program.family)


def raw_inverse(carrier: Carrier, program: Program) -> None:
    for index in range(program.depth - 1, -1, -1):
        apply_in_layer(carrier, index, program.family, inverse=True)
        apply_out_layer(carrier, index, program.family, inverse=True)
        apply_public_phase(carrier, index, program.family, inverse=True)


def port_causality_witness() -> dict[str, Any]:
    program = compile_program(4, "PRIMARY")
    seeds = [seed_phases(), seed_phases()]
    for shell in range(P):
        seeds[1][shell, 0] = convolve(seeds[1][shell, 0], one_hot(5))
    boundaries: list[tuple[int, ...]] = []
    final_equal = True
    restored = []
    for seed in seeds:
        carrier = Carrier(phases=np.array(seed, copy=True), seed=np.array(seed, copy=True))
        raw_forward(carrier, program)
        boundaries.append(boundary_from_phase_orbit(carrier.phases, program))
        final_equal = final_equal and carrier_valid(carrier.phases)
        raw_inverse(carrier, program)
        restored.append(carrier.exact_restored())
    return {
        "program_depth": program.depth,
        "only_resident_gauge_ports_differ_initially": bool(
            np.array_equal(seeds[0][:, 1:], seeds[1][:, 1:])
            and not np.array_equal(seeds[0][:, :1], seeds[1][:, :1])
        ),
        "both_forward_states_remain_on_phase_orbit": final_equal,
        "final_boundaries_differ": boundaries[0] != boundaries[1],
        "both_actual_carriers_restore_exactly": all(restored),
    }


def raw_reverse_control(carrier: Carrier, program: Program, mode: str) -> None:
    for index in range(program.depth - 1, -1, -1):
        if mode == "REORDERED":
            apply_out_layer(carrier, index, program.family, inverse=True)
            apply_in_layer(carrier, index, program.family, inverse=True)
        else:
            apply_in_layer(carrier, index, program.family, inverse=True)
            if mode != "MISSING_OUT":
                apply_out_layer(
                    carrier,
                    index,
                    program.family,
                    inverse=True,
                    offset_mutation=1 if mode == "WRONG_OUT" else 0,
                )
        apply_public_phase(carrier, index, program.family, inverse=True)


def controls() -> dict[str, bool]:
    program = compile_program(4, "ALTERNATE")
    valid = Carrier.create()
    begin_forward(valid, program)
    forward(valid, program)
    valid_boundary = project(valid, program)
    inverse(valid, program)

    disabled = Carrier.create()
    begin_forward(disabled, program)
    forward(disabled, program, port_enabled=False)
    disabled_boundary = project(disabled, program)

    order = Carrier.create()
    begin_forward(order, program)
    forward(order, program, layer_order="IN_OUT")
    order_boundary = project(order, program)

    topology = Carrier.create()
    begin_forward(topology, program)
    forward(topology, program, hub_mutation=1)
    topology_boundary = project(topology, program)

    reverse_failures: dict[str, bool] = {}
    for mode in ("MISSING_OUT", "WRONG_OUT", "REORDERED"):
        carrier = Carrier.create()
        raw_forward(carrier, program)
        raw_reverse_control(carrier, program, mode)
        reverse_failures[mode] = not carrier.exact_restored()

    premature_rejected = False
    premature = Carrier.create()
    try:
        begin_forward(premature, program)
        project(premature, program)
    except RuntimeError:
        premature_rejected = True

    port_projection_rejected = False
    hidden = Carrier.create()
    try:
        begin_forward(hidden, program)
        forward(hidden, program)
        project_port(hidden, program)
    except RuntimeError:
        port_projection_rejected = True

    wrong_owner_rejected = False
    wrong_owner = Carrier.create()
    try:
        begin_forward(wrong_owner, program)
        forward(wrong_owner, program)
        project(wrong_owner, program)
        inverse(wrong_owner, compile_program(4, "PRIMARY"))
    except RuntimeError:
        wrong_owner_rejected = True

    null_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    causality = port_causality_witness()
    return {
        "resident_port_factor_changes_boundary": valid_boundary != disabled_boundary,
        "out_in_layer_order_changes_boundary": valid_boundary != order_boundary,
        "public_hub_topology_mutation_changes_boundary": valid_boundary
        != topology_boundary,
        "missing_out_layer_inverse_changes_actual_carrier": reverse_failures[
            "MISSING_OUT"
        ],
        "wrong_out_layer_inverse_changes_actual_carrier": reverse_failures[
            "WRONG_OUT"
        ],
        "reordered_inverse_changes_actual_carrier": reverse_failures["REORDERED"],
        "premature_final_projection_rejected": premature_rejected,
        "resident_port_projection_rejected": port_projection_rejected,
        "wrong_program_ownership_rejected": wrong_owner_rejected,
        "null_carrier_rejected": null_rejected,
        "gauge_only_port_perturbation_changes_boundary": causality[
            "final_boundaries_differ"
        ],
        "gauge_only_port_perturbation_both_restore": causality[
            "both_actual_carriers_restore_exactly"
        ],
    }


def unrenormalized_complex_coordinate_attempt() -> dict[str, Any]:
    roots = np.exp(2.0j * np.pi * np.arange(P) / P).astype(np.complex128)
    observed = []
    for depth in (1, 4, 16, 64):
        state = roots[seed_exponents()]
        finite = True
        with np.errstate(over="ignore", invalid="ignore"):
            for index in range(depth):
                for shell in range(P):
                    state[shell] *= roots[
                        m145.phase_exponent(shell, index, "PRIMARY")
                    ]
                hub = hub_index(index, "PRIMARY")
                for target in target_order(hub):
                    factor = state[hub, 0] * roots[
                        public_offset_exponent(
                            hub, target, index, "PRIMARY", 0
                        )
                    ]
                    state[target] *= factor
                for controller in target_order(hub):
                    factor = state[controller, 0] * roots[
                        public_offset_exponent(
                            controller, hub, index, "PRIMARY", 1
                        )
                    ]
                    state[hub] *= factor
                if not bool(np.all(np.isfinite(state))):
                    finite = False
                    break
        norm_error = (
            float(np.max(np.abs(np.abs(state) - 1.0))) if finite else None
        )
        observed.append(
            {
                "depth": depth,
                "all_coordinates_finite": finite,
                "maximum_unit_norm_error": norm_error,
                "within_failure_tolerance": finite
                and norm_error is not None
                and norm_error <= COMPLEX_UNIT_NORM_FAILURE_TOLERANCE,
            }
        )
    failures = [
        item["depth"] for item in observed if not item["within_failure_tolerance"]
    ]
    if not failures or failures[0] != 64:
        fail("unrenormalized complex phase coordinate diagnostic changed")
    return {
        "accepted_path": False,
        "tested_depths": [item["depth"] for item in observed],
        "predeclared_unit_norm_failure_tolerance": COMPLEX_UNIT_NORM_FAILURE_TOLERANCE,
        "observed": observed,
        "observed_first_unit_norm_failure_depth": failures[0],
        "reason": "FLOATING_RADIAL_DRIFT_IS_REUSED_AS_CONTROL_MAGNITUDE_AND_AMPLIFIES; CANONICAL_NORMALIZATION_WOULD_CHANGE_THE_RESTORATION_CLASS",
        "normalization_silently_added": False,
        "claim_promoted": False,
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_within = all(
        case["final_phase_orbit_bytes_identical_to_expanded_matched_recurrence"]
        and case["boundary_identical_to_matched_recurrence"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_before"] == 0
        and case["restoration_generation_after"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        and case["stats"]["resident_port_scalar_reads"] == 0
        and case["stats"]["resident_port_exponent_decodes"] == 0
        for case in cases
    )
    if not all_within:
        fail("one or more exact C17 phase cases failed the declared scope")
    causality = port_causality_witness()
    if not all(causality.values()):
        fail("C17 phase port causality witness failed")
    control_results = controls()
    if not all(control_results.values()):
        fail("one or more C17 phase controls failed")
    reuse = reuse_control()
    repeated = repeated_reuse_control()
    complex_attempt = unrenormalized_complex_coordinate_attempt()

    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_update_cells = max(
        case["stats"]["maximum_named_update_uint8_cells"] for case in cases
    )
    maximum_projection_cells = max(
        case["stats"]["maximum_named_projection_integer_cells"] for case in cases
    )
    resident_cells = 51 * P
    resident_bytes = resident_cells
    update_live = resident_bytes + maximum_update_cells + maximum_program_bytes
    projection_live = (
        resident_bytes + 8 * maximum_projection_cells + maximum_program_bytes
    )
    commitment_live = resident_bytes + 96 + 64 + maximum_program_bytes
    warm_live = max(update_live, projection_live, commitment_live)
    baseline_resident_bytes = 51
    baseline_warm_live = baseline_resident_bytes + 4 + maximum_program_bytes

    return {
        "schema": "CAT_CAS_F17_EXACT_C17_FIBER_PORT_CONVOLUTION_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m145.__file__).name,
            "sha256": hashlib.sha256(Path(m145.__file__).read_bytes()).hexdigest(),
            "used_for": "PUBLIC_F17_PHASE_EXPONENT_SCHEDULE_ONLY",
        },
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_DISCRETE_PHASE_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_exact": all_within,
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "carrier_law": {
            "logical_phase_factors": 51,
            "resident_phase_orbit_coordinates_per_factor": P,
            "resident_phase_orbit_uint8_cells": resident_cells,
            "port_type": "ONE_HOT_C17_GROUP_ALGEBRA_PHASE",
            "primitive": "NATIVE17_COORDINATE_CYCLIC_CONVOLUTION_WITH_ACTUAL_RESIDENT_CONTROLLER_PHASE_FACTOR",
            "out_layer_edges_per_step": 16,
            "in_layer_edges_per_step": 16,
            "shared_hub_port_consumers_per_out_layer": 16,
            "resident_port_scalar_readout_per_update": False,
            "resident_port_exponent_decode_per_update": False,
            "retained_public_plan_cells": 0,
            "port_is_unprojected_until_final_boundary": True,
            "port_projection_api_rejects_resident_port": control_results[
                "resident_port_projection_rejected"
            ],
            "out_and_in_layers_noncommute": control_results[
                "out_in_layer_order_changes_boundary"
            ],
            "gauge_only_port_perturbation_changes_boundary": causality[
                "final_boundaries_differ"
            ],
            "no_relation_table_or_assignment_expansion": True,
            "phase_orbit_coordinate_expansion_is_counted": True,
        },
        "port_causality_witness": causality,
        "resource_law": {
            "resident_phase_orbit_uint8_cells": resident_cells,
            "resident_phase_orbit_bytes": resident_bytes,
            "retained_public_plan_cells": 0,
            "retained_public_plan_bytes": 0,
            "maximum_named_update_uint8_scratch_cells": maximum_update_cells,
            "maximum_named_projection_int64_cells": maximum_projection_cells,
            "maximum_public_program_json_bytes": maximum_program_bytes,
            "maximum_named_commitment_live_bytes_including_program_json": commitment_live,
            "maximum_named_warm_execution_live_bytes_including_program_json": warm_live,
            "maximum_named_full_lifecycle_live_bytes": warm_live,
            "commitment_input_copy_bytes": 0,
            "commitment_public_hexdigest_bytes": 64,
            "commitment_logical_sha256_state_and_block_bytes": 96,
            "forward_cyclic_convolutions_per_step": 179,
            "forward_convolution_coordinate_multiplications_per_step": 179
            * P
            * P,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "python_numpy_allocator_native_library_and_whole_process_memory_excluded": True,
        },
        "matched_classical_recurrence": {
            "method": "IDENTICAL51_UINT8_C17_RESIDUE_TRIANGULAR_RECURRENCE",
            "executed_in_every_case": True,
            "final_expanded_phase_bytes_identical_in_every_case": all(
                case[
                    "final_phase_orbit_bytes_identical_to_expanded_matched_recurrence"
                ]
                for case in cases
            ),
            "boundary_identical_in_every_case": all(
                case["boundary_identical_to_matched_recurrence"] for case in cases
            ),
            "resident_uint8_cells": 51,
            "resident_bytes": baseline_resident_bytes,
            "maximum_named_update_uint8_scratch_cells": 4,
            "maximum_named_warm_execution_live_bytes_including_program_json": baseline_warm_live,
            "phase_to_classical_resident_byte_ratio": resident_bytes
            / baseline_resident_bytes,
            "same_public_topology_and_boundary_semantics": True,
            "comparison_establishes_distinct_phase_resource": False,
            "comparison_establishes_computational_advantage": False,
            "optimal_compact_classical_recurrence_claimed": False,
        },
        "restoration": {
            "class": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_buffers": "NO_RESTORATION_CLAIM",
            "all867_resident_phase_orbit_cells_compared": True,
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
                "EXACT_ONE_HOT_C17_PHASE_PORT_CONSUMED_BY_NATIVE_CYCLIC_CONVOLUTION_WITHOUT_RESIDENT_EXPONENT_READOUT",
                "ONE_RESIDENT_PHASE_PORT_CONTROLS16_TARGET_TRIPLETS",
                "NONCOMMUTING_RECIPROCAL_OUT_AND_IN_CONVOLUTION_LAYERS",
                "FINAL_ONLY_BOUNDARY_PROJECTION_WITH_EXACT_SAME_BACKING_RESTORATION_AND_REUSE",
                "PURE_C17_PHASE_ORBIT_COLLAPSES_TO51_RESIDUE_CLASSICAL_RECURRENCE",
            ],
            "not_established": [
                "COHERENT_SUPERPOSITION_OVER_MULTIPLE_PHASE_ORBIT_COORDINATES",
                "RESOURCE_BEYOND_IDENTICAL51_RESIDUE_RECURRENCE",
                "OPTIMAL_CLASSICAL_BASELINE",
                "GENERAL_RELATIONAL_CONTRACTION",
                "CATVM_MACHINE_ENFORCED_CUSTODY",
                "DISTINCT_PHASE_RESOURCE",
                "COMPUTATIONAL_ADVANTAGE",
                "SMALL_WALL_CROSSING",
                "PHYSICAL_WAVEFORM_EXECUTION",
                "PHYSICAL_BIT_REPLACEMENT",
                "UNBOUNDED_CATALYTIC_COMPUTATION",
            ],
        },
        "rejected_unrenormalized_complex_coordinate_attempt": complex_attempt,
        "next_obstruction": "EXACT_PHASE_ORBIT_CONVOLUTION_REMOVES_SCALAR_PORT_READOUT_AND_FLOATING_RADIAL_DRIFT_BUT_ONE_HOT_C17_GROUP_ELEMENTS_HAVE_A51_RESIDUE_QUOTIENT_WITH17X_LESS_RESIDENT_STATE",
        "next_experiment": "COHERENT_SUPERPOSITION_PORT_OVER_MULTIPLE_C17_PHASE_FIBERS_WITH_NATIVE_INTERFERENCE_CLOSURE_OR_A_MATCHED_FIXED_RANK_CLASSICAL_FACTOR_NO_GO",
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
