#!/usr/bin/env python3
"""Exact C17 phase-orbit superposition and spectral-factor diagnostic.

This package is intentionally bounded.  It replaces M149's one-hot phase
orbit factors with general F103[C17] coefficient vectors, applies reversible
triangular convolution shears through one resident port, and compares every
accepted state with the exact 17-mode number-theoretic transform factorization.
It is direct-process software and establishes neither CATVM custody nor a
distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import f17_exact_c17_fiber_port_convolution as m149


MODULUS = 103
CYCLE = 17
ROOT = 72
ROOT_INVERSE = pow(ROOT, -1, MODULUS)
CYCLE_INVERSE = pow(CYCLE, -1, MODULUS)
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F103_C17_PHASE_ORBIT_SUPERPOSITION_CONVOLUTION_SHEARS_"
    "KEEP_ONE_MULTI_COORDINATE_RESIDENT_PORT_UNPROJECTED_ACROSS16_TARGETS_"
    "AND_RECIPROCAL_NONCOMMUTING_CONSUMERS_WITH_EXACT_DESTRUCTIVE_"
    "INTERFERENCE_RESTORATION_AND_REUSE_THROUGH_DEPTH1024_BUT_FACTORIZE_"
    "EXACTLY_INTO_AN_EXECUTED17_MODE_CLASSICAL_RECURRENCE_WITH_EQUAL_"
    "RESIDENT_DIMENSION_AND_LOWER_CONVOLUTION_WORK"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def mod_array(value: np.ndarray) -> np.ndarray:
    return np.mod(value, MODULUS).astype(np.uint8)


def rotate_coefficients(value: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(value, shift % CYCLE).astype(np.uint8, copy=False)


@dataclass
class WorkStats:
    cyclic_convolutions: int = 0
    coefficient_multiplications: int = 0
    convolution_accumulation_additions: int = 0
    shear_additions: int = 0
    rotations: int = 0
    triangular_shears: int = 0
    destructive_interference_events: int = 0
    maximum_named_update_scratch_bytes: int = 0

    def record_convolution(self) -> None:
        self.cyclic_convolutions += 1
        self.coefficient_multiplications += CYCLE * CYCLE
        self.convolution_accumulation_additions += CYCLE * (CYCLE - 1)
        self.maximum_named_update_scratch_bytes = max(
            self.maximum_named_update_scratch_bytes,
            CYCLE * 8 + CYCLE,
        )

    def descriptor(self) -> dict[str, int]:
        return {
            "cyclic_convolutions": self.cyclic_convolutions,
            "coefficient_multiplications": self.coefficient_multiplications,
            "convolution_accumulation_additions": (
                self.convolution_accumulation_additions
            ),
            "shear_additions": self.shear_additions,
            "rotations": self.rotations,
            "triangular_shears": self.triangular_shears,
            "destructive_interference_events": (
                self.destructive_interference_events
            ),
            "maximum_named_update_scratch_bytes": (
                self.maximum_named_update_scratch_bytes
            ),
        }


def cyclic_convolution(
    left: np.ndarray,
    right: np.ndarray,
    stats: WorkStats | None = None,
) -> np.ndarray:
    if left.shape != (CYCLE,) or right.shape != (CYCLE,):
        fail("C17 convolution requires two 17-coordinate factors")
    accumulator = np.zeros(CYCLE, dtype=np.int64)
    for index in range(CYCLE):
        shifted = np.roll(right, index).astype(np.int64)
        accumulator += int(left[index]) * shifted
    if stats is not None:
        stats.record_convolution()
    return mod_array(accumulator)


def seed_superpositions() -> np.ndarray:
    state = np.zeros((CYCLE, 3, CYCLE), dtype=np.uint8)
    for shell in range(CYCLE):
        for slot in range(3):
            positions = (
                (5 * shell + 3 * slot + 1) % CYCLE,
                (7 * shell * shell + 2 * slot + 4) % CYCLE,
                (11 * shell + 5 * slot + 9) % CYCLE,
            )
            amplitudes = (
                1 + (3 * shell + 5 * slot) % 31,
                1 + (7 * shell + 2 * slot) % 29,
                MODULUS - (1 + (11 * shell + slot) % 23),
            )
            for position, amplitude in zip(positions, amplitudes, strict=True):
                state[shell, slot, position] = (
                    int(state[shell, slot, position]) + amplitude
                ) % MODULUS
    return state


@dataclass(frozen=True)
class Program:
    depth: int
    family: str
    observation_quadratic: int
    observation_linear: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_SUPERPOSITION_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "topology": "PUBLIC_ROTATING_HUB16_OUT16_IN_TRIANGULAR_SHEARS",
            "port_type": "F103_C17_GROUP_ALGEBRA_SUPERPOSITION",
            "observation": [
                self.observation_quadratic,
                self.observation_linear,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 1024:
        fail("superposition program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("superposition program family outside declared set")
    return Program(
        depth=depth,
        family=family,
        observation_quadratic=(7 * depth + 3 * len(family) + 1) % MODULUS or 1,
        observation_linear=(11 * depth + len(family) + 5) % MODULUS,
    )


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return m149.hub_index(index, family, mutation=mutation)


def target_order(hub: int) -> Iterator[int]:
    return m149.target_order(hub)


def public_offset(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    mutation: int = 0,
) -> int:
    return m149.public_offset_exponent(
        controller,
        target,
        index,
        family,
        layer,
        mutation=mutation,
    )


@dataclass
class Carrier:
    coefficients: np.ndarray
    stage: str = "RESTORED"
    active_program: str | None = None
    forward_index: int = 0
    inverse_index: int = 0
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def seal(cls, seed: np.ndarray | None = None) -> "Carrier":
        value = seed_superpositions() if seed is None else seed
        if value.shape != (CYCLE, 3, CYCLE):
            fail("invalid F103 C17 superposition seed")
        return cls(mod_array(value).copy())

    @property
    def backing_identity(self) -> int:
        return int(self.coefficients.__array_interface__["data"][0])


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(carrier.coefficients.tobytes()).hexdigest()


def begin_forward(carrier: Carrier, program: Program) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong superposition carrier")
    if carrier.stage != "RESTORED" or carrier.active_program is not None:
        fail("superposition carrier is already leased")
    carrier.stage = "FORWARD_SUPERPOSITION"
    carrier.active_program = program.fingerprint()
    carrier.forward_index = 0
    carrier.inverse_index = 0
    carrier.projection_calls = 0
    carrier.stats = WorkStats()


def require_owned(carrier: Carrier, program: Program, stage: str) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong superposition carrier")
    if carrier.stage != stage or carrier.active_program != program.fingerprint():
        fail("superposition carrier owner or stage changed")


def apply_public_rotation(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    stats: WorkStats | None = None,
) -> None:
    for shell in range(CYCLE):
        shift = m149.m145.phase_exponent(shell, index, family) % CYCLE
        if inverse:
            shift = -shift
        for slot in range(3):
            state[shell, slot] = rotate_coefficients(
                state[shell, slot], shift
            )
            if stats is not None:
                stats.rotations += 1


def add_term(
    destination: np.ndarray,
    term: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    before = destination.astype(np.int64)
    signed = -term.astype(np.int64) if inverse else term.astype(np.int64)
    after = np.mod(before + signed, MODULUS)
    if stats is not None:
        stats.shear_additions += CYCLE
        stats.destructive_interference_events += int(
            np.count_nonzero(
                (before != 0)
                & (term.astype(np.int64) != 0)
                & (after == 0)
            )
        )
    destination[:] = after.astype(np.uint8)


def triangular_shear(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    if destination.shape != (3, CYCLE) or control.shape != (CYCLE,):
        fail("invalid triangular C17 shear operands")
    if inverse:
        term_two = cyclic_convolution(control, destination[1], stats)
        add_term(destination[2], term_two, inverse=True, stats=stats)
        term_one = cyclic_convolution(control, destination[0], stats)
        add_term(destination[1], term_one, inverse=True, stats=stats)
    else:
        term_one = cyclic_convolution(control, destination[0], stats)
        add_term(destination[1], term_one, inverse=False, stats=stats)
        term_two = cyclic_convolution(control, destination[1], stats)
        add_term(destination[2], term_two, inverse=False, stats=stats)
    if stats is not None:
        stats.triangular_shears += 1


def apply_edge(
    state: np.ndarray,
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    if port_enabled:
        control_slot = 0 if layer == 0 else 1
        control = rotate_coefficients(
            state[controller, control_slot],
            public_offset(
                controller,
                target,
                index,
                family,
                layer,
                mutation=offset_mutation,
            ),
        )
    else:
        control = np.zeros(CYCLE, dtype=np.uint8)
    triangular_shear(
        state[target], control, inverse=inverse, stats=stats
    )


def apply_out_layer(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    hub = hub_index(index, family, mutation=hub_mutation)
    targets = list(target_order(hub))
    if inverse:
        targets.reverse()
    for target in targets:
        apply_edge(
            state,
            hub,
            target,
            index,
            family,
            0,
            inverse=inverse,
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
            stats=stats,
        )


def apply_in_layer(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool = False,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    hub = hub_index(index, family, mutation=hub_mutation)
    controllers = list(target_order(hub))
    if inverse:
        controllers.reverse()
    for controller in controllers:
        apply_edge(
            state,
            controller,
            hub,
            index,
            family,
            1,
            inverse=inverse,
            port_enabled=port_enabled,
            offset_mutation=offset_mutation,
            stats=stats,
        )


def raw_forward(
    state: np.ndarray,
    program: Program,
    *,
    layer_order: str = "OUT_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    for index in range(program.depth):
        apply_public_rotation(
            state, index, program.family, stats=stats
        )
        if layer_order == "OUT_IN":
            apply_out_layer(
                state,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )
            apply_in_layer(
                state,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )
        elif layer_order == "IN_OUT":
            apply_in_layer(
                state,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )
            apply_out_layer(
                state,
                index,
                program.family,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )
        else:
            fail("unknown superposition layer order")


def raw_inverse(
    state: np.ndarray,
    program: Program,
    *,
    assumed_layer_order: str = "OUT_IN",
    hub_mutation: int = 0,
    offset_mutation: int = 0,
) -> None:
    for index in reversed(range(program.depth)):
        if assumed_layer_order == "OUT_IN":
            apply_in_layer(
                state,
                index,
                program.family,
                inverse=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )
            apply_out_layer(
                state,
                index,
                program.family,
                inverse=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )
        elif assumed_layer_order == "IN_OUT":
            apply_out_layer(
                state,
                index,
                program.family,
                inverse=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )
            apply_in_layer(
                state,
                index,
                program.family,
                inverse=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
            )
        else:
            fail("unknown assumed inverse layer order")
        apply_public_rotation(state, index, program.family, inverse=True)


def forward(carrier: Carrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARD_SUPERPOSITION")
    raw_forward(carrier.coefficients, program, stats=carrier.stats)
    carrier.forward_index = program.depth
    carrier.stage = "FORWARDED_SUPERPOSITION"


def inverse(carrier: Carrier, program: Program) -> None:
    require_owned(carrier, program, "FORWARDED_SUPERPOSITION")
    raw_inverse(carrier.coefficients, program)
    carrier.inverse_index = program.depth
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
    require_owned(carrier, program, "FORWARDED_SUPERPOSITION")
    carrier.projection_calls += 1
    return boundary_from_coefficients(carrier.coefficients, program)


def project_resident_port(_carrier: Carrier, _port: int) -> None:
    fail("resident superposition-port projection is forbidden")


def ntt(vector: np.ndarray) -> np.ndarray:
    output = np.zeros(CYCLE, dtype=np.int64)
    for mode in range(CYCLE):
        total = 0
        for coordinate in range(CYCLE):
            total += int(vector[coordinate]) * pow(
                ROOT, mode * coordinate, MODULUS
            )
        output[mode] = total % MODULUS
    return output.astype(np.uint8)


def inverse_ntt(vector: np.ndarray) -> np.ndarray:
    output = np.zeros(CYCLE, dtype=np.int64)
    for coordinate in range(CYCLE):
        total = 0
        for mode in range(CYCLE):
            total += int(vector[mode]) * pow(
                ROOT_INVERSE, mode * coordinate, MODULUS
            )
        output[coordinate] = total * CYCLE_INVERSE % MODULUS
    return output.astype(np.uint8)


def ntt_state(state: np.ndarray) -> np.ndarray:
    modes = np.empty_like(state)
    for shell in range(CYCLE):
        for slot in range(3):
            modes[shell, slot] = ntt(state[shell, slot])
    return modes


def inverse_ntt_state(modes: np.ndarray) -> np.ndarray:
    state = np.empty_like(modes)
    for shell in range(CYCLE):
        for slot in range(3):
            state[shell, slot] = inverse_ntt(modes[shell, slot])
    return state


@dataclass
class SpectralStats:
    modal_multiplications: int = 0
    modal_shear_additions: int = 0
    modal_phase_multiplications: int = 0
    triangular_shears: int = 0
    forward_ntt_multiplications: int = 0
    final_inverse_ntt_multiplications: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "modal_multiplications": self.modal_multiplications,
            "modal_shear_additions": self.modal_shear_additions,
            "modal_phase_multiplications": self.modal_phase_multiplications,
            "triangular_shears": self.triangular_shears,
            "forward_ntt_multiplications": self.forward_ntt_multiplications,
            "final_inverse_ntt_multiplications": (
                self.final_inverse_ntt_multiplications
            ),
        }


def rotate_modes(vector: np.ndarray, shift: int) -> np.ndarray:
    return np.array(
        [
            int(vector[mode]) * pow(ROOT, mode * (shift % CYCLE), MODULUS)
            % MODULUS
            for mode in range(CYCLE)
        ],
        dtype=np.uint8,
    )


def spectral_triangular_shear(
    destination: np.ndarray,
    control: np.ndarray,
    stats: SpectralStats,
) -> None:
    term_one = mod_array(
        control.astype(np.int64) * destination[0].astype(np.int64)
    )
    destination[1] = mod_array(
        destination[1].astype(np.int64) + term_one.astype(np.int64)
    )
    term_two = mod_array(
        control.astype(np.int64) * destination[1].astype(np.int64)
    )
    destination[2] = mod_array(
        destination[2].astype(np.int64) + term_two.astype(np.int64)
    )
    stats.modal_multiplications += 2 * CYCLE
    stats.modal_shear_additions += 2 * CYCLE
    stats.triangular_shears += 1


def spectral_forward(program: Program) -> tuple[np.ndarray, SpectralStats]:
    modes = ntt_state(seed_superpositions())
    stats = SpectralStats(
        forward_ntt_multiplications=CYCLE * 3 * CYCLE * CYCLE
    )
    for index in range(program.depth):
        for shell in range(CYCLE):
            shift = m149.m145.phase_exponent(
                shell, index, program.family
            ) % CYCLE
            for slot in range(3):
                modes[shell, slot] = rotate_modes(
                    modes[shell, slot], shift
                )
                stats.modal_phase_multiplications += CYCLE
        hub = hub_index(index, program.family)
        for target in target_order(hub):
            control = rotate_modes(
                modes[hub, 0],
                public_offset(hub, target, index, program.family, 0),
            )
            stats.modal_phase_multiplications += CYCLE
            spectral_triangular_shear(modes[target], control, stats)
        for controller in target_order(hub):
            control_slot = 1
            control = rotate_modes(
                modes[controller, control_slot],
                public_offset(controller, hub, index, program.family, 1),
            )
            stats.modal_phase_multiplications += CYCLE
            spectral_triangular_shear(modes[hub], control, stats)
    stats.final_inverse_ntt_multiplications = CYCLE * 3 * CYCLE * CYCLE
    return inverse_ntt_state(modes), stats


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
    full_state_equal = np.array_equal(carrier.coefficients, spectral_state)
    spectral_boundary = boundary_from_coefficients(spectral_state, program)
    support_sizes = np.count_nonzero(carrier.coefficients, axis=2)
    inverse(carrier, program)
    restored = np.array_equal(carrier.coefficients, seed_superpositions())
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial_commitment,
        "final_commitment": final_commitment,
        "boundary": list(boundary),
        "spectral_boundary": list(spectral_boundary),
        "coefficient_state_identical_to_spectral_factor_recurrence": (
            full_state_equal
        ),
        "boundary_identical_to_spectral_factor_recurrence": (
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
        "spectral_stats": spectral_stats.descriptor(),
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
    for _ in range(100):
        result = transaction(carrier, compile_program(16, "REUSE"))
        if not result["exact_restoration"]:
            fail("repeated exact superposition restoration failed")
        boundaries.add(tuple(result["boundary"]))
    return {
        "cycles": 100,
        "exact_restoration": state_commitment(carrier) == seed_commitment,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation": carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }


def controls() -> dict[str, bool]:
    program = compile_program(4, "PRIMARY")
    seed = seed_superpositions()

    missing = seed.copy()
    raw_forward(missing, program)

    wrong = seed.copy()
    raw_forward(wrong, program)
    raw_inverse(wrong, program, offset_mutation=1)

    reordered = seed.copy()
    raw_forward(reordered, program)
    raw_inverse(reordered, program, assumed_layer_order="IN_OUT")

    null_rejected = False
    try:
        begin_forward(None, program)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True

    premature_rejected = False
    leased = Carrier.seal()
    begin_forward(leased, program)
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
    raw_forward(reordered_forward, program, layer_order="IN_OUT")
    mutated = seed.copy()
    raw_forward(mutated, program, hub_mutation=1)

    basis_roundtrip = True
    for coordinate in range(CYCLE):
        basis = np.zeros(CYCLE, dtype=np.uint8)
        basis[coordinate] = 1
        basis_roundtrip &= np.array_equal(inverse_ntt(ntt(basis)), basis)

    witness_left = np.zeros(CYCLE, dtype=np.uint8)
    witness_right = np.zeros(CYCLE, dtype=np.uint8)
    witness_left[0] = 1
    witness_left[1] = 1
    witness_right[0] = 1
    witness_right[1] = MODULUS - 1
    witness_product = cyclic_convolution(witness_left, witness_right)

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
        "out_in_order_changes_boundary": (
            boundary_from_coefficients(normal, program)
            != boundary_from_coefficients(reordered_forward, program)
        ),
        "topology_mutation_changes_boundary": (
            boundary_from_coefficients(normal, program)
            != boundary_from_coefficients(mutated, program)
        ),
        "ntt_roundtrip_all17_basis_vectors": basis_roundtrip,
        "explicit_destructive_interference_witness": (
            int(witness_product[1]) == 0
            and int(witness_product[0]) == 1
            and int(witness_product[2]) == MODULUS - 1
        ),
    }


def run() -> dict[str, Any]:
    if pow(ROOT, CYCLE, MODULUS) != 1:
        fail("declared C17 root does not close")
    if len({pow(ROOT, index, MODULUS) for index in range(CYCLE)}) != CYCLE:
        fail("declared C17 root is not primitive")

    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_exact = all(
        case["coefficient_state_identical_to_spectral_factor_recurrence"]
        and case["boundary_identical_to_spectral_factor_recurrence"]
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
        fail("one or more C17 superposition cases failed")

    control_results = controls()
    if not all(control_results.values()):
        fail("one or more C17 superposition controls failed")
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
        fail("unrelated restored-carrier reuse failed")
    repeated = repeated_reuse_control()
    if not (
        repeated["exact_restoration"]
        and repeated["same_backing"]
        and repeated["restoration_generation"] == 100
        and repeated["stable_boundary_count"] == 1
        and not repeated["snapshot_reload_used"]
    ):
        fail("repeated restored-carrier reuse failed")

    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_phase_scratch = max(
        case["phase_stats"]["maximum_named_update_scratch_bytes"]
        for case in cases
    )
    resident_cells = CYCLE * 3 * CYCLE
    resident_bytes = resident_cells
    phase_warm_live = resident_bytes + maximum_phase_scratch + maximum_program_bytes
    spectral_scratch_bytes = CYCLE * 8
    spectral_warm_live = resident_bytes + spectral_scratch_bytes + maximum_program_bytes
    maximum_phase_work = max(
        case["phase_stats"]["coefficient_multiplications"] for case in cases
    )
    maximum_spectral_work = max(
        case["spectral_stats"]["modal_multiplications"] for case in cases
    )
    destructive_events = sum(
        case["phase_stats"]["destructive_interference_events"] for case in cases
    )

    return {
        "schema": "CAT_CAS_F103_C17_SUPERPOSITION_INTERFERENCE_FACTOR_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m149.__file__).name,
            "sha256": hashlib.sha256(Path(m149.__file__).read_bytes()).hexdigest(),
            "used_for": "PUBLIC_ROTATING_HUB_AND_PHASE_SCHEDULE_ONLY",
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
            "primitive_c17_root": ROOT,
            "logical_phase_factors": 51,
            "resident_field_coordinates": resident_cells,
            "port_type": "F103_C17_GROUP_ALGEBRA_SUPERPOSITION",
            "primitive": "REVERSIBLE_TRIANGULAR_NATIVE_CYCLIC_CONVOLUTION_SHEAR",
            "shared_hub_consumers_per_out_layer": 16,
            "reciprocal_hub_consumers_per_in_layer": 16,
            "resident_port_scalar_or_exponent_readout": False,
            "resident_port_unprojected_until_final_boundary": True,
            "general_multi_coordinate_superposition": True,
            "destructive_interference_events_observed": destructive_events,
            "retained_public_plan_cells": 0,
        },
        "exact_spectral_factorization": {
            "method": "EXECUTED17_MODE_F103_NUMBER_THEORETIC_FACTOR_RECURRENCE",
            "algebra_law": "F103_C17_GROUP_ALGEBRA_ISOMORPHIC_TO_F103_POWER17",
            "reason": "X17_MINUS1_SPLITS_INTO17_DISTINCT_ROOTS_OVER_F103",
            "full_state_and_boundary_match_every_case": True,
            "resident_field_coordinates": resident_cells,
            "resident_bytes": resident_bytes,
            "maximum_named_warm_live_bytes": spectral_warm_live,
            "maximum_forward_modal_multiplications": maximum_spectral_work,
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
            "spectral_resident_uint8_cells": resident_cells,
            "spectral_resident_bytes": resident_bytes,
            "spectral_maximum_named_warm_live_bytes": spectral_warm_live,
            "maximum_phase_forward_coefficient_multiplications": maximum_phase_work,
            "maximum_spectral_forward_modal_multiplications": maximum_spectral_work,
            "phase_to_spectral_resident_dimension_ratio": 1,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": (
            "F103_C17_GROUP_ALGEBRA_ROTATION_ADDITION_AND_CONVOLUTION_SHEARS_"
            "ON_THE_DECLARED51_PORT_ROTATING_HUB_TOPOLOGY_ACROSS18_CASES_"
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
            "GENERAL_F103_C17_SUPERPOSITIONS_PRESERVE_EXACT_INTERFERENCE_AND_"
            "RESTORE_WITHOUT_HISTORY_BUT_THE_ENTIRE_DECLARED_PRIMITIVE_"
            "ALGEBRA_FACTORIZES_INTO17_INDEPENDENT_CLASSICAL_MODES"
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
