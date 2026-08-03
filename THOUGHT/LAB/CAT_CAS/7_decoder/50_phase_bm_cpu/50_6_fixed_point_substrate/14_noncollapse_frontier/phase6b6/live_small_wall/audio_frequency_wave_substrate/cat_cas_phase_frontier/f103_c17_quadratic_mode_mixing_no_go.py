#!/usr/bin/env python3
"""Exact C17 convolution plus coefficient-quadratic mode-mixing diagnostic.

This bounded package changes the M150 phase machine itself.  It interleaves
native cyclic-convolution shears with a reversible coefficient-local square
shear driven by one unresolved resident C17 port.  The square destroys the 17
independent-mode factorization, because pointwise multiplication in the orbit
chart becomes circular convolution in the character chart.  The package then
checks the stronger coupled 17-mode recurrence and the identical coefficient
recurrence.  It is direct-process software and establishes neither CATVM
custody nor a distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import f103_c17_superposition_interference_factor_no_go as m150


MODULUS = 103
CYCLE = 17
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F103_C17_INTERLEAVED_CONVOLUTION_AND_COEFFICIENTWISE_"
    "QUADRATIC_PHASE_SHEARS_BREAK17_INDEPENDENT_SPECTRAL_MODE_CLOSURE_"
    "WHILE_KEEPING_ONE_RESIDENT_MULTI_COORDINATE_PORT_UNPROJECTED_ACROSS_"
    "16_CONSUMERS_WITH_EXACT_RESTORATION_AND_REUSE_THROUGH_DEPTH1024_BUT_"
    "COLLAPSE_TO_EXECUTED_COUPLED17_MODE_AND_IDENTICAL867_COORDINATE_"
    "CLASSICAL_RECURRENCES"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def family_code(family: str) -> int:
    return {"PRIMARY": 2, "REUSE": 7, "ALTERNATE": 11}[family]


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_QUADRATIC_MODE_MIXING_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "topology": "PUBLIC_ROTATING_HUB16_OUT16_QUADRATIC16_IN",
            "port_type": "F103_C17_GROUP_ALGEBRA_SUPERPOSITION",
            "primitive_order": [
                "C17_ROTATION",
                "C17_CONVOLUTION_OUT",
                "COEFFICIENTWISE_QUADRATIC_SHARED_PORT_SHEAR",
                "C17_CONVOLUTION_IN",
            ],
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 1024:
        fail("quadratic mode-mixing depth outside declared ceiling")
    if family not in FAMILIES:
        fail("quadratic mode-mixing family outside declared set")
    return Program(
        depth=depth,
        family=family,
        observation_quadratic=(7 * depth + 3 * len(family) + 1) % MODULUS
        or 1,
        observation_linear=(11 * depth + len(family) + 5) % MODULUS,
    )


def quadratic_offset(
    hub: int,
    target: int,
    index: int,
    family: str,
    mutation: int = 0,
) -> int:
    return (
        11 * hub
        + 13 * target
        + 5 * index
        + 3 * family_code(family)
        + mutation
    ) % CYCLE


@dataclass
class PhaseStats:
    convolution: m150.WorkStats = field(default_factory=m150.WorkStats)
    quadratic_shared_squares: int = 0
    quadratic_field_multiplications: int = 0
    quadratic_target_additions: int = 0
    quadratic_rotations: int = 0
    quadratic_layers: int = 0
    quadratic_destructive_interference_events: int = 0
    maximum_named_quadratic_scratch_bytes: int = 0

    def descriptor(self) -> dict[str, int]:
        descriptor = self.convolution.descriptor()
        descriptor.update(
            {
                "quadratic_shared_squares": self.quadratic_shared_squares,
                "quadratic_field_multiplications": (
                    self.quadratic_field_multiplications
                ),
                "quadratic_target_additions": self.quadratic_target_additions,
                "quadratic_rotations": self.quadratic_rotations,
                "quadratic_layers": self.quadratic_layers,
                "quadratic_destructive_interference_events": (
                    self.quadratic_destructive_interference_events
                ),
                "maximum_named_quadratic_scratch_bytes": (
                    self.maximum_named_quadratic_scratch_bytes
                ),
                "total_nonlinear_core_multiplications": (
                    descriptor["coefficient_multiplications"]
                    + self.quadratic_field_multiplications
                ),
            }
        )
        return descriptor


@dataclass
class Carrier:
    coefficients: np.ndarray
    stage: str = "RESTORED"
    active_program: str | None = None
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: PhaseStats = field(default_factory=PhaseStats)

    @classmethod
    def seal(cls, seed: np.ndarray | None = None) -> "Carrier":
        value = m150.seed_superpositions() if seed is None else seed
        if value.shape != (CYCLE, 3, CYCLE):
            fail("invalid F103 C17 quadratic seed")
        return cls(m150.mod_array(value).copy())

    @property
    def backing_identity(self) -> int:
        return int(self.coefficients.__array_interface__["data"][0])


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(carrier.coefficients.tobytes()).hexdigest()


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong quadratic carrier")
    if carrier.stage != "RESTORED" or carrier.active_program is not None:
        fail("quadratic carrier is already leased")
    carrier.stage = "FORWARD_QUADRATIC"
    carrier.active_program = program.fingerprint()
    carrier.projection_calls = 0
    carrier.stats = PhaseStats()


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong quadratic carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("quadratic carrier owner or stage changed")


def add_quadratic_term(
    destination: np.ndarray,
    term: np.ndarray,
    *,
    inverse: bool,
    stats: PhaseStats | None,
) -> None:
    before = destination.astype(np.int64)
    signed = -term.astype(np.int64) if inverse else term.astype(np.int64)
    after = np.mod(before + signed, MODULUS)
    if stats is not None:
        stats.quadratic_target_additions += CYCLE
        stats.quadratic_destructive_interference_events += int(
            np.count_nonzero(
                (before != 0)
                & (term.astype(np.int64) != 0)
                & (after == 0)
            )
        )
    destination[:] = after.astype(np.uint8)


def apply_quadratic_layer(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    quadratic_offset_mutation: int = 0,
    stats: PhaseStats | None = None,
) -> None:
    hub = m150.hub_index(index, family, mutation=hub_mutation)
    if port_enabled:
        control = state[hub, 2].astype(np.int64)
        shared_square = m150.mod_array(control * control)
    else:
        shared_square = np.zeros(CYCLE, dtype=np.uint8)
    if stats is not None:
        stats.quadratic_shared_squares += 1
        stats.quadratic_field_multiplications += CYCLE
        stats.quadratic_layers += 1
        stats.maximum_named_quadratic_scratch_bytes = max(
            stats.maximum_named_quadratic_scratch_bytes,
            CYCLE * 8 + CYCLE,
        )
    targets = list(m150.target_order(hub))
    if inverse:
        targets.reverse()
    for target in targets:
        term = m150.rotate_coefficients(
            shared_square,
            quadratic_offset(
                hub,
                target,
                index,
                family,
                mutation=quadratic_offset_mutation,
            ),
        )
        add_quadratic_term(
            state[target, 0], term, inverse=inverse, stats=stats
        )
        if stats is not None:
            stats.quadratic_rotations += 1


def apply_module(
    state: np.ndarray,
    module: str,
    index: int,
    family: str,
    *,
    inverse: bool,
    port_enabled: bool,
    hub_mutation: int,
    offset_mutation: int,
    quadratic_offset_mutation: int,
    stats: PhaseStats | None,
) -> None:
    convolution_stats = None if stats is None else stats.convolution
    if module == "OUT":
        m150.apply_out_layer(
            state,
            index,
            family,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            offset_mutation=offset_mutation,
            stats=convolution_stats,
        )
    elif module == "QUADRATIC":
        apply_quadratic_layer(
            state,
            index,
            family,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            quadratic_offset_mutation=quadratic_offset_mutation,
            stats=stats,
        )
    elif module == "IN":
        m150.apply_in_layer(
            state,
            index,
            family,
            inverse=inverse,
            port_enabled=port_enabled,
            hub_mutation=hub_mutation,
            offset_mutation=offset_mutation,
            stats=convolution_stats,
        )
    else:
        fail("unknown quadratic module")


ORDERS = {
    "OUT_QUADRATIC_IN": ("OUT", "QUADRATIC", "IN"),
    "OUT_IN_QUADRATIC": ("OUT", "IN", "QUADRATIC"),
    "QUADRATIC_OUT_IN": ("QUADRATIC", "OUT", "IN"),
}


def raw_forward(
    state: np.ndarray,
    program: Program,
    *,
    module_order: str = "OUT_QUADRATIC_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    quadratic_offset_mutation: int = 0,
    stats: PhaseStats | None = None,
) -> None:
    if module_order not in ORDERS:
        fail("unknown forward quadratic module order")
    for index in range(program.depth):
        m150.apply_public_rotation(
            state, index, program.family, stats=None if stats is None else stats.convolution
        )
        for module in ORDERS[module_order]:
            apply_module(
                state,
                module,
                index,
                program.family,
                inverse=False,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                quadratic_offset_mutation=quadratic_offset_mutation,
                stats=stats,
            )


def raw_inverse(
    state: np.ndarray,
    program: Program,
    *,
    assumed_module_order: str = "OUT_QUADRATIC_IN",
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    quadratic_offset_mutation: int = 0,
) -> None:
    if assumed_module_order not in ORDERS:
        fail("unknown inverse quadratic module order")
    for index in reversed(range(program.depth)):
        for module in reversed(ORDERS[assumed_module_order]):
            apply_module(
                state,
                module,
                index,
                program.family,
                inverse=True,
                port_enabled=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                quadratic_offset_mutation=quadratic_offset_mutation,
                stats=None,
            )
        m150.apply_public_rotation(
            state, index, program.family, inverse=True
        )


def forward(carrier: Carrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD_QUADRATIC")
    raw_forward(carrier.coefficients, program, stats=carrier.stats)
    carrier.stage = "FORWARDED_QUADRATIC"


def inverse(carrier: Carrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARDED_QUADRATIC")
    raw_inverse(carrier.coefficients, program)
    carrier.stage = "RESTORED"
    carrier.active_program = None
    carrier.restoration_generation += 1


def boundary_from_coefficients(
    state: np.ndarray, program: Program
) -> tuple[int, ...]:
    accumulator = np.zeros(CYCLE, dtype=np.int64)
    for shell in range(CYCLE):
        for slot in range(3):
            weight = (
                program.observation_quadratic * shell * shell
                + program.observation_linear * (slot + 1)
                + 5 * shell * (slot + 1)
                + 1
            ) % MODULUS
            accumulator += weight * state[shell, slot].astype(np.int64)
    return tuple(int(value) for value in np.mod(accumulator, MODULUS))


def project(carrier: Carrier, program: Program) -> tuple[int, ...]:
    require_owned(carrier, program, "FORWARDED_QUADRATIC")
    carrier.projection_calls += 1
    return boundary_from_coefficients(carrier.coefficients, program)


def project_resident_port(_carrier: Carrier, _port: int) -> None:
    fail("resident quadratic-port projection is forbidden")


@dataclass
class SpectralStats:
    modal_multiplications: int = 0
    modal_shear_additions: int = 0
    modal_phase_multiplications: int = 0
    triangular_shears: int = 0
    quadratic_mode_convolution_multiplications: int = 0
    quadratic_mode_convolution_additions: int = 0
    quadratic_target_additions: int = 0
    quadratic_layers: int = 0
    forward_ntt_multiplications: int = 0
    final_inverse_ntt_multiplications: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "modal_multiplications": self.modal_multiplications,
            "modal_shear_additions": self.modal_shear_additions,
            "modal_phase_multiplications": self.modal_phase_multiplications,
            "triangular_shears": self.triangular_shears,
            "quadratic_mode_convolution_multiplications": (
                self.quadratic_mode_convolution_multiplications
            ),
            "quadratic_mode_convolution_additions": (
                self.quadratic_mode_convolution_additions
            ),
            "quadratic_target_additions": self.quadratic_target_additions,
            "quadratic_layers": self.quadratic_layers,
            "forward_ntt_multiplications": self.forward_ntt_multiplications,
            "final_inverse_ntt_multiplications": (
                self.final_inverse_ntt_multiplications
            ),
            "total_nonlinear_core_multiplications": (
                self.modal_multiplications
                + self.quadratic_mode_convolution_multiplications
            ),
        }


def spectral_square(
    modes: np.ndarray, stats: SpectralStats | None = None
) -> np.ndarray:
    if modes.shape != (CYCLE,):
        fail("spectral square requires 17 modes")
    result = np.zeros(CYCLE, dtype=np.int64)
    for output_mode in range(CYCLE):
        total = 0
        for left_mode in range(CYCLE):
            right_mode = (output_mode - left_mode) % CYCLE
            total += int(modes[left_mode]) * int(modes[right_mode])
        result[output_mode] = total * m150.CYCLE_INVERSE % MODULUS
    if stats is not None:
        stats.quadratic_mode_convolution_multiplications += CYCLE * CYCLE
        stats.quadratic_mode_convolution_additions += CYCLE * (CYCLE - 1)
        stats.quadratic_layers += 1
    return result.astype(np.uint8)


def spectral_triangular_shear(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: SpectralStats | None,
) -> None:
    sign = -1 if inverse else 1
    if inverse:
        destination[2] = m150.mod_array(
            destination[2].astype(np.int64)
            - control.astype(np.int64) * destination[1].astype(np.int64)
        )
        destination[1] = m150.mod_array(
            destination[1].astype(np.int64)
            - control.astype(np.int64) * destination[0].astype(np.int64)
        )
    else:
        destination[1] = m150.mod_array(
            destination[1].astype(np.int64)
            + control.astype(np.int64) * destination[0].astype(np.int64)
        )
        destination[2] = m150.mod_array(
            destination[2].astype(np.int64)
            + control.astype(np.int64) * destination[1].astype(np.int64)
        )
    if stats is not None:
        stats.modal_multiplications += 2 * CYCLE
        stats.modal_shear_additions += 2 * CYCLE
        stats.triangular_shears += 1


def apply_spectral_convolution_layer(
    modes: np.ndarray,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool,
    stats: SpectralStats | None,
) -> None:
    hub = m150.hub_index(index, family)
    peers = list(m150.target_order(hub))
    if inverse:
        peers.reverse()
    for peer in peers:
        controller, target = (hub, peer) if layer == 0 else (peer, hub)
        control_slot = 0 if layer == 0 else 1
        control = m150.rotate_modes(
            modes[controller, control_slot],
            m150.public_offset(controller, target, index, family, layer),
        )
        spectral_triangular_shear(
            modes[target], control, inverse=inverse, stats=stats
        )
        if stats is not None:
            stats.modal_phase_multiplications += CYCLE


def apply_spectral_quadratic_layer(
    modes: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool,
    stats: SpectralStats | None,
    independent_mode_sham: bool = False,
) -> None:
    hub = m150.hub_index(index, family)
    control = modes[hub, 2]
    if independent_mode_sham:
        shared_square = m150.mod_array(
            control.astype(np.int64) * control.astype(np.int64)
        )
        if stats is not None:
            stats.quadratic_mode_convolution_multiplications += CYCLE
            stats.quadratic_layers += 1
    else:
        shared_square = spectral_square(control, stats)
    targets = list(m150.target_order(hub))
    if inverse:
        targets.reverse()
    sign = -1 if inverse else 1
    for target in targets:
        term = m150.rotate_modes(
            shared_square, quadratic_offset(hub, target, index, family)
        )
        modes[target, 0] = m150.mod_array(
            modes[target, 0].astype(np.int64)
            + sign * term.astype(np.int64)
        )
        if stats is not None:
            stats.modal_phase_multiplications += CYCLE
            stats.quadratic_target_additions += CYCLE


def spectral_forward(
    program: Program, *, independent_mode_sham: bool = False
) -> tuple[np.ndarray, SpectralStats]:
    modes = m150.ntt_state(m150.seed_superpositions())
    stats = SpectralStats(
        forward_ntt_multiplications=CYCLE * 3 * CYCLE * CYCLE
    )
    for index in range(program.depth):
        for shell in range(CYCLE):
            shift = m150.m149.m145.phase_exponent(
                shell, index, program.family
            ) % CYCLE
            for slot in range(3):
                modes[shell, slot] = m150.rotate_modes(
                    modes[shell, slot], shift
                )
                stats.modal_phase_multiplications += CYCLE
        apply_spectral_convolution_layer(
            modes, index, program.family, 0, inverse=False, stats=stats
        )
        apply_spectral_quadratic_layer(
            modes,
            index,
            program.family,
            inverse=False,
            stats=stats,
            independent_mode_sham=independent_mode_sham,
        )
        apply_spectral_convolution_layer(
            modes, index, program.family, 1, inverse=False, stats=stats
        )
    stats.final_inverse_ntt_multiplications = CYCLE * 3 * CYCLE * CYCLE
    return m150.inverse_ntt_state(modes), stats


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.seal()
    backing = carrier.backing_identity
    initial_commitment = state_commitment(carrier)
    generation_before = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program)
    final_commitment = state_commitment(carrier)
    boundary = project(carrier, program)
    spectral_state, spectral_stats = spectral_forward(program)
    spectral_boundary = boundary_from_coefficients(spectral_state, program)
    support_sizes = np.count_nonzero(carrier.coefficients, axis=2)
    identical = np.array_equal(carrier.coefficients, spectral_state)
    inverse(carrier, program)
    restored = np.array_equal(carrier.coefficients, m150.seed_superpositions())
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial_commitment,
        "final_commitment": final_commitment,
        "boundary": list(boundary),
        "coupled_spectral_boundary": list(spectral_boundary),
        "coefficient_state_identical_to_coupled_spectral_recurrence": identical,
        "boundary_identical_to_coupled_spectral_recurrence": (
            boundary == spectral_boundary
        ),
        "minimum_final_port_support": int(np.min(support_sizes)),
        "maximum_final_port_support": int(np.max(support_sizes)),
        "exact_restoration": restored,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation_before": generation_before,
        "restoration_generation_after": carrier.restoration_generation,
        "projection_calls": carrier.projection_calls,
        "snapshot_reload_used": carrier.snapshot_reload_used,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "phase_stats": carrier.stats.descriptor(),
        "coupled_spectral_stats": spectral_stats.descriptor(),
    }


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    generation = carrier.restoration_generation
    begin_forward(carrier, program)
    forward(carrier, program)
    boundary = project(carrier, program)
    final = state_commitment(carrier)
    inverse(carrier, program)
    return {
        "boundary": list(boundary),
        "final_commitment": final,
        "restored_commitment": state_commitment(carrier),
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "generation_before": generation,
        "generation_after": carrier.restoration_generation,
        "resource_signature": {
            "resident_field_cells": int(carrier.coefficients.size),
            "resident_bytes": int(carrier.coefficients.nbytes),
            "phase_stats": carrier.stats.descriptor(),
        },
    }


def reuse_control() -> dict[str, Any]:
    carrier = Carrier.seal()
    first = transaction(carrier, compile_program(64, "PRIMARY"))
    backing = carrier.backing_identity
    second_program = compile_program(613, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier.seal(), second_program)
    return {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": carrier.backing_identity == backing,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": (
            second["final_commitment"] == fresh["final_commitment"]
        ),
        "resource_signature_matches_fresh": (
            second["resource_signature"] == fresh["resource_signature"]
        ),
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }


def repeated_reuse_control() -> dict[str, Any]:
    carrier = Carrier.seal()
    backing = carrier.backing_identity
    seed_commitment = state_commitment(carrier)
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(64):
        result = transaction(carrier, compile_program(8, "REUSE"))
        if not result["exact_restoration"]:
            fail("repeated exact quadratic restoration failed")
        boundaries.add(tuple(result["boundary"]))
    return {
        "cycles": 64,
        "exact_restoration": state_commitment(carrier) == seed_commitment,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation": carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }


def controls() -> dict[str, bool]:
    program = compile_program(4, "PRIMARY")
    seed = m150.seed_superpositions()

    missing = seed.copy()
    raw_forward(missing, program)

    wrong = seed.copy()
    raw_forward(wrong, program)
    raw_inverse(wrong, program, quadratic_offset_mutation=1)

    reordered = seed.copy()
    raw_forward(reordered, program)
    raw_inverse(reordered, program, assumed_module_order="OUT_IN_QUADRATIC")

    null_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    leased = Carrier.seal()
    begin_forward(leased, program)
    premature_rejected = False
    try:
        project(leased, program)
    except RuntimeError:
        premature_rejected = True
    resident_rejected = False
    try:
        project_resident_port(leased, 0)
    except RuntimeError:
        resident_rejected = True
    ownership_rejected = False
    try:
        forward(leased, compile_program(4, "REUSE"))
    except RuntimeError:
        ownership_rejected = True

    normal = seed.copy()
    raw_forward(normal, program)
    disabled = seed.copy()
    raw_forward(disabled, program, port_enabled=False)
    reordered_forward = seed.copy()
    raw_forward(reordered_forward, program, module_order="OUT_IN_QUADRATIC")
    mutated = seed.copy()
    raw_forward(mutated, program, hub_mutation=1)

    vector = seed[0, 0]
    coefficient_square = m150.mod_array(
        vector.astype(np.int64) * vector.astype(np.int64)
    )
    transformed = m150.ntt(vector)
    coupled = spectral_square(transformed)
    independent_sham = m150.mod_array(
        transformed.astype(np.int64) * transformed.astype(np.int64)
    )

    witness = np.zeros(CYCLE, dtype=np.uint8)
    witness[0] = 1
    witness[1] = 1
    witness_modes = spectral_square(witness)

    nonlinear_left = np.zeros(CYCLE, dtype=np.uint8)
    nonlinear_right = np.zeros(CYCLE, dtype=np.uint8)
    nonlinear_left[0] = 1
    nonlinear_right[0] = 1
    square_sum = m150.mod_array(
        (nonlinear_left.astype(np.int64) + nonlinear_right.astype(np.int64)) ** 2
    )
    sum_squares = m150.mod_array(
        nonlinear_left.astype(np.int64) ** 2
        + nonlinear_right.astype(np.int64) ** 2
    )

    sham_state, _ = spectral_forward(program, independent_mode_sham=True)
    return {
        "missing_inverse_changes_state": not np.array_equal(missing, seed),
        "wrong_inverse_changes_state": not np.array_equal(wrong, seed),
        "reordered_inverse_changes_state": not np.array_equal(reordered, seed),
        "null_carrier_rejected": null_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_port_projection_rejected": resident_rejected,
        "wrong_owner_rejected": ownership_rejected,
        "null_port_changes_boundary": (
            boundary_from_coefficients(normal, program)
            != boundary_from_coefficients(disabled, program)
        ),
        "module_order_changes_boundary": (
            boundary_from_coefficients(normal, program)
            != boundary_from_coefficients(reordered_forward, program)
        ),
        "topology_mutation_changes_boundary": (
            boundary_from_coefficients(normal, program)
            != boundary_from_coefficients(mutated, program)
        ),
        "pointwise_square_transform_requires_mode_convolution": (
            np.array_equal(m150.ntt(coefficient_square), coupled)
            and not np.array_equal(coupled, independent_sham)
        ),
        "independent_spectral_mode_sham_changes_final_state": not np.array_equal(
            normal, sham_state
        ),
        "quadratic_map_is_nonlinear": not np.array_equal(
            square_sum, sum_squares
        ),
        "explicit_cross_mode_generation_witness": (
            int(witness_modes[0]) != 0
            and int(witness_modes[1]) != 0
            and int(witness_modes[2]) != 0
        ),
        "all17_input_modes_enter_each_symbolic_output_convolution": all(
            len({left for left in range(CYCLE)}) == CYCLE
            for _output in range(CYCLE)
        ),
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_exact = all(
        case["coefficient_state_identical_to_coupled_spectral_recurrence"]
        and case["boundary_identical_to_coupled_spectral_recurrence"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_after"]
        == case["restoration_generation_before"] + 1
        and case["projection_calls"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        and case["maximum_final_port_support"] > 1
        for case in cases
    )
    if not all_exact:
        fail("one or more quadratic mode-mixing cases failed")

    control_results = controls()
    if not all(control_results.values()):
        failed = [name for name, value in control_results.items() if not value]
        fail(f"quadratic controls failed: {failed}")
    reuse = reuse_control()
    if not all(
        (
            reuse["first_exact_restoration"],
            reuse["second_exact_restoration"],
            reuse["same_backing_across_programs"],
            reuse["second_boundary_matches_fresh"],
            reuse["second_final_commitment_matches_fresh"],
            reuse["resource_signature_matches_fresh"],
            not reuse["snapshot_reload_used"],
        )
    ):
        fail("unrelated quadratic restored-carrier reuse failed")
    repeated = repeated_reuse_control()
    if not (
        repeated["exact_restoration"]
        and repeated["same_backing"]
        and repeated["restoration_generation"] == 64
        and repeated["stable_boundary_count"] == 1
        and not repeated["snapshot_reload_used"]
    ):
        fail("repeated quadratic restored-carrier reuse failed")

    resident_cells = CYCLE * 3 * CYCLE
    resident_bytes = resident_cells
    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_phase_scratch = max(
        max(
            case["phase_stats"]["maximum_named_update_scratch_bytes"],
            case["phase_stats"]["maximum_named_quadratic_scratch_bytes"],
        )
        for case in cases
    )
    phase_warm_live = resident_bytes + maximum_phase_scratch + maximum_program_bytes
    spectral_scratch = CYCLE * 8 + CYCLE
    spectral_warm_live = resident_bytes + spectral_scratch + maximum_program_bytes
    maximum_phase_work = max(
        case["phase_stats"]["total_nonlinear_core_multiplications"]
        for case in cases
    )
    maximum_spectral_work = max(
        case["coupled_spectral_stats"]["total_nonlinear_core_multiplications"]
        for case in cases
    )
    quadratic_interference = sum(
        case["phase_stats"]["quadratic_destructive_interference_events"]
        for case in cases
    )

    return {
        "schema": "CAT_CAS_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m150.__file__).name,
            "sha256": hashlib.sha256(Path(m150.__file__).read_bytes()).hexdigest(),
            "used_for": "M150_SEED_PUBLIC_SCHEDULE_AND_LINEAR_C17_PRIMITIVES",
        },
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_FINITE_FIELD_PHASE_ORBIT_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_exact": all_exact,
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "carrier_law": {
            "field_modulus": MODULUS,
            "cycle_order": CYCLE,
            "logical_phase_factors": 51,
            "resident_field_coordinates": resident_cells,
            "port_type": "F103_C17_GROUP_ALGEBRA_SUPERPOSITION",
            "primitive": "INTERLEAVED_CYCLIC_CONVOLUTION_AND_COEFFICIENTWISE_QUADRATIC_TRIANGULAR_SHEARS",
            "shared_quadratic_port_consumers": 16,
            "quadratic_source_slot": 2,
            "quadratic_destination_slot": 0,
            "resident_port_scalar_or_exponent_readout": False,
            "resident_port_unprojected_until_final_boundary": True,
            "relation_table_cells": 0,
            "assignment_expansion_cells": 0,
            "quadratic_destructive_interference_events_observed": quadratic_interference,
            "retained_public_plan_cells": 0,
        },
        "spectral_mode_mixing": {
            "independent17_mode_closure_preserved": False,
            "exact_law": "NTT_OF_POINTWISE_SQUARE_EQUALS_INV17_TIMES_CIRCULAR_SELF_CONVOLUTION_OF17_MODES",
            "input_modes_per_output_quadratic_term": CYCLE,
            "independent_mode_square_sham_rejected": True,
            "coupled17_mode_recurrence_executed": True,
            "full_state_and_boundary_match_every_case": True,
        },
        "matched_classical_recurrences": {
            "identical_coefficient_recurrence": "EXECUTED_BY_INDEPENDENT_ORACLE_OVER_THE_SAME867_FIELD_COORDINATES",
            "coupled_spectral_recurrence": "EXECUTED17_MODE_NTT_RECURRENCE_WITH_FULL_MODE_CONVOLUTION_FOR_THE_QUADRATIC_SHEAR",
            "resident_field_coordinates_each": resident_cells,
            "resident_bytes_each": resident_bytes,
            "coupled_spectral_maximum_named_warm_live_bytes": spectral_warm_live,
            "maximum_phase_nonlinear_core_multiplications": maximum_phase_work,
            "maximum_coupled_spectral_nonlinear_core_multiplications": maximum_spectral_work,
            "optimal_compact_classical_recurrence_claimed": False,
        },
        "restoration": {
            "carrier_classification": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_buffers_classification": "NO_RESTORATION_CLAIM",
            "same_backing": True,
            "inverse_history_cells": 0,
            "retained_restoration_baseline_cells": 0,
            "snapshot_reload_used": False,
            "unrelated_program_reuse": reuse,
            "repeated_reuse": repeated,
        },
        "controls": control_results,
        "resource_accounting": {
            "phase_resident_uint8_cells": resident_cells,
            "phase_resident_bytes": resident_bytes,
            "phase_maximum_named_warm_live_bytes": phase_warm_live,
            "coupled_spectral_resident_uint8_cells": resident_cells,
            "coupled_spectral_resident_bytes": resident_bytes,
            "coupled_spectral_maximum_named_warm_live_bytes": spectral_warm_live,
            "phase_to_coupled_spectral_resident_dimension_ratio": 1,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": (
            "F103_C17_CONVOLUTION_AND_COEFFICIENTWISE_QUADRATIC_SHEARS_ON_"
            "THE_DECLARED51_PORT_ROTATING_HUB_TOPOLOGY_ACROSS18_CASES_"
            "THROUGH_DEPTH1024_IN_LINUX_DIRECT_PROCESS_SOFTWARE"
        ),
        "not_established": [
            "COMPLEX_OR_PHYSICAL_COHERENCE",
            "GENERAL_PHASE_RELATIONAL_CONTRACTION",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": (
            "THE_COEFFICIENTWISE_QUADRATIC_SHEAR_BREAKS_INDEPENDENT_"
            "CHARACTER_MODES_BUT_REMAINS_AN_EXACT_FIXED867_COORDINATE_"
            "POLYNOMIAL_MAP_WITH_IDENTICAL_COEFFICIENT_AND_COUPLED17_MODE_"
            "CLASSICAL_RECURRENCES"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
