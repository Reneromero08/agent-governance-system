#!/usr/bin/env python3
"""Typed open C17 weighted-relation composition/intersection diagnostic.

A 17-coordinate factor r represents the translation-invariant weighted open
relation R(x, y) = r[y-x] over F103.  Native hidden-interface composition is
cyclic convolution, while native parallel intersection is Hadamard product.
The accepted carrier interleaves both laws on one unresolved typed port,
projects only the final boundary relation, reverses the actual operations, and
reuses the same carrier.  No 17x17 relation table is materialized.

This is bounded direct-process software.  It does not establish CATVM custody,
a distinct phase resource, or an advantage over compact classical recurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import f103_c17_quadratic_mode_mixing_no_go as m151


MODULUS = 103
CYCLE = 17
NODE_COUNT = 9
SLOTS = 3
DEPTHS = (1, 4, 16, 64, 256, 512)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
RELATION_TYPE = "F103_WEIGHTED_TRANSLATION_INVARIANT_C17_TO_C17"
CLAIM = (
    "BOUNDED_EXACT_TYPED_OPEN_F103_C17_WEIGHTED_PHASE_RELATIONS_COMPOSE_BY_"
    "CYCLIC_CONVOLUTION_AND_INTERSECT_BY_HADAMARD_PRODUCT_ON_ONE_SHARED_"
    "UNRESOLVED_PORT_ACROSS8_NONCOMMUTING_CONSUMERS_WITHOUT_RELATION_TABLE_"
    "MATERIALIZATION_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_"
    "THROUGH_DEPTH512_BUT_COLLAPSE_TO_IDENTICAL459_COORDINATE_COEFFICIENT_"
    "AND_DUAL_SPECTRAL_CLASSICAL_RECURRENCES"
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
    owner: int
    observation_linear: int
    observation_quadratic: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": "CAT_CAS_F103_C17_OPEN_RELATION_BIALGEBRA_PROGRAM_V1",
            "depth": self.depth,
            "family": self.family,
            "owner": self.owner,
            "node_count": NODE_COUNT,
            "slots_per_node": SLOTS,
            "port_type": RELATION_TYPE,
            "topology": "PUBLIC_ROTATING_HUB8_COMPOSE_INTERSECT_OUT8_INTERSECT_COMPOSE_IN",
            "relation_semantics": "R_XY_EQUALS_SIGNATURE_Y_MINUS_X",
            "composition": "SUM_OVER_HIDDEN_Y_EQUALS_CYCLIC_CONVOLUTION",
            "intersection": "PARALLEL_PRODUCT_EQUALS_HADAMARD_PRODUCT",
            "observation": [
                self.observation_linear,
                self.observation_quadratic,
            ],
        }

    def fingerprint(self) -> str:
        return digest_json(self.descriptor())


def compile_program(depth: int, family: str) -> Program:
    if not isinstance(depth, int) or not 1 <= depth <= 512:
        fail("open-relation program depth outside declared ceiling")
    if family not in FAMILIES:
        fail("open-relation program family outside declared set")
    return Program(
        depth=depth,
        family=family,
        owner=(0xC1700000 + 97 * depth + family_code(family)) & 0xFFFFFFFF,
        observation_linear=(7 * depth + 5 * len(family) + 1) % MODULUS,
        observation_quadratic=(11 * depth + 3 * len(family) + 7) % MODULUS
        or 1,
    )


def hub_index(index: int, family: str, mutation: int = 0) -> int:
    return (5 * index + family_code(family) + mutation) % NODE_COUNT


def peer_order(hub: int) -> list[int]:
    return [(hub + offset) % NODE_COUNT for offset in range(1, NODE_COUNT)]


def relation_offset(
    controller: int,
    target: int,
    index: int,
    family: str,
    layer: int,
    mutation: int = 0,
) -> int:
    return (
        7 * controller
        + 11 * target
        + 3 * index
        + 5 * layer
        + family_code(family)
        + mutation
    ) % CYCLE


def relation_phase_exponent(node: int, index: int, family: str) -> int:
    if family == "PRIMARY":
        return (3 * node * node + 5 * index + 2 * index.bit_count() + 1) % CYCLE
    if family == "REUSE":
        return (7 * node + 4 * index + 3 * (index % 5) + 2) % CYCLE
    return (
        11 * node * node
        + 6 * index
        + 2 * (index ^ (index >> 1)).bit_count()
        + 3
    ) % CYCLE


def seed_relations() -> np.ndarray:
    state = np.zeros((NODE_COUNT, SLOTS, CYCLE), dtype=np.uint8)
    for node in range(NODE_COUNT):
        for slot in range(SLOTS):
            positions = (
                (2 * node + 3 * slot + 1) % CYCLE,
                (5 * node * node + 7 * slot + 4) % CYCLE,
                (11 * node + 2 * slot + 9) % CYCLE,
            )
            amplitudes = (
                1 + (7 * node + 3 * slot) % 37,
                1 + (11 * node + 5 * slot) % 31,
                MODULUS - (1 + (13 * node + 2 * slot) % 29),
            )
            for position, amplitude in zip(positions, amplitudes, strict=True):
                state[node, slot, position] = (
                    int(state[node, slot, position]) + amplitude
                ) % MODULUS
    return state


@dataclass
class WorkStats:
    composition_convolutions: int = 0
    intersection_hadamard_products: int = 0
    composition_field_multiplications: int = 0
    intersection_field_multiplications: int = 0
    accumulation_additions: int = 0
    shear_additions: int = 0
    rotations: int = 0
    out_modules: int = 0
    in_modules: int = 0
    destructive_interference_events: int = 0
    maximum_named_scratch_bytes: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "composition_convolutions": self.composition_convolutions,
            "intersection_hadamard_products": self.intersection_hadamard_products,
            "composition_field_multiplications": self.composition_field_multiplications,
            "intersection_field_multiplications": self.intersection_field_multiplications,
            "total_bialgebra_field_multiplications": (
                self.composition_field_multiplications
                + self.intersection_field_multiplications
            ),
            "accumulation_additions": self.accumulation_additions,
            "shear_additions": self.shear_additions,
            "rotations": self.rotations,
            "out_modules": self.out_modules,
            "in_modules": self.in_modules,
            "destructive_interference_events": self.destructive_interference_events,
            "maximum_named_scratch_bytes": self.maximum_named_scratch_bytes,
        }


@dataclass
class Carrier:
    relations: np.ndarray
    port_type: str = RELATION_TYPE
    stage: str = "RESTORED"
    active_program: str | None = None
    active_owner: int | None = None
    restoration_generation: int = 0
    projection_calls: int = 0
    snapshot_reload_used: bool = False
    stats: WorkStats = field(default_factory=WorkStats)

    @classmethod
    def seal(cls, seed: np.ndarray | None = None) -> "Carrier":
        value = seed_relations() if seed is None else seed
        if value.shape != (NODE_COUNT, SLOTS, CYCLE):
            fail("invalid open-relation carrier seed")
        return cls(m151.m150.mod_array(value).copy())

    @property
    def backing_identity(self) -> int:
        return int(self.relations.__array_interface__["data"][0])


def state_commitment(carrier: Carrier) -> str:
    return hashlib.sha256(carrier.relations.tobytes()).hexdigest()


def require_relation_type(type_name: str) -> None:
    if type_name != RELATION_TYPE:
        fail("open-relation port type mismatch")


def begin_forward(carrier: Carrier, program: Program, owner: int) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong open-relation carrier")
    require_relation_type(carrier.port_type)
    if owner != program.owner:
        fail("open-relation owner mismatch")
    if carrier.stage != "RESTORED" or carrier.active_program is not None:
        fail("open-relation carrier is already leased")
    carrier.stage = "FORWARD_OPEN_RELATION"
    carrier.active_program = program.fingerprint()
    carrier.active_owner = owner
    carrier.projection_calls = 0
    carrier.stats = WorkStats()


def require_owned(
    carrier: Carrier, program: Program, owner: int, stage: str
) -> None:
    if not isinstance(carrier, Carrier):
        fail("null or wrong open-relation carrier")
    require_relation_type(carrier.port_type)
    if (
        carrier.stage != stage
        or carrier.active_program != program.fingerprint()
        or carrier.active_owner != owner
        or owner != program.owner
    ):
        fail("open-relation owner, program, or stage changed")


def relation_compose(
    left: np.ndarray, right: np.ndarray, stats: WorkStats | None
) -> np.ndarray:
    result = m151.m150.cyclic_convolution(left, right)
    if stats is not None:
        stats.composition_convolutions += 1
        stats.composition_field_multiplications += CYCLE * CYCLE
        stats.accumulation_additions += CYCLE * (CYCLE - 1)
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, CYCLE * 8 + CYCLE
        )
    return result


def relation_intersect(
    left: np.ndarray, right: np.ndarray, stats: WorkStats | None
) -> np.ndarray:
    result = m151.m150.mod_array(
        left.astype(np.int64) * right.astype(np.int64)
    )
    if stats is not None:
        stats.intersection_hadamard_products += 1
        stats.intersection_field_multiplications += CYCLE
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, CYCLE * 8 + CYCLE
        )
    return result


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


def out_relation_module(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    require_relation_type(RELATION_TYPE)
    if inverse:
        intersection = relation_intersect(control, destination[1], stats)
        add_term(destination[2], intersection, inverse=True, stats=stats)
        composition = relation_compose(control, destination[0], stats)
        add_term(destination[1], composition, inverse=True, stats=stats)
    else:
        composition = relation_compose(control, destination[0], stats)
        add_term(destination[1], composition, inverse=False, stats=stats)
        intersection = relation_intersect(control, destination[1], stats)
        add_term(destination[2], intersection, inverse=False, stats=stats)
    if stats is not None:
        stats.out_modules += 1


def in_relation_module(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    require_relation_type(RELATION_TYPE)
    if inverse:
        composition = relation_compose(control, destination[1], stats)
        add_term(destination[2], composition, inverse=True, stats=stats)
        intersection = relation_intersect(control, destination[0], stats)
        add_term(destination[1], intersection, inverse=True, stats=stats)
    else:
        intersection = relation_intersect(control, destination[0], stats)
        add_term(destination[1], intersection, inverse=False, stats=stats)
        composition = relation_compose(control, destination[1], stats)
        add_term(destination[2], composition, inverse=False, stats=stats)
    if stats is not None:
        stats.in_modules += 1


def rotate_state(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    for node in range(NODE_COUNT):
        shift = relation_phase_exponent(node, index, family)
        if inverse:
            shift = -shift
        for slot in range(SLOTS):
            state[node, slot] = m151.m150.rotate_coefficients(
                state[node, slot], shift
            )
            if stats is not None:
                stats.rotations += 1


def apply_out_layer(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool,
    port_enabled: bool,
    hub_mutation: int,
    offset_mutation: int,
    stats: WorkStats | None,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    peers = peer_order(hub)
    if inverse:
        peers.reverse()
    for peer in peers:
        if port_enabled:
            control = m151.m150.rotate_coefficients(
                state[hub, 0],
                relation_offset(
                    hub, peer, index, family, 0, offset_mutation
                ),
            )
        else:
            control = np.zeros(CYCLE, dtype=np.uint8)
        out_relation_module(
            state[peer], control, inverse=inverse, stats=stats
        )


def apply_in_layer(
    state: np.ndarray,
    index: int,
    family: str,
    *,
    inverse: bool,
    port_enabled: bool,
    hub_mutation: int,
    offset_mutation: int,
    stats: WorkStats | None,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    peers = peer_order(hub)
    if inverse:
        peers.reverse()
    for peer in peers:
        if port_enabled:
            control = m151.m150.rotate_coefficients(
                state[peer, 2],
                relation_offset(
                    peer, hub, index, family, 1, offset_mutation
                ),
            )
        else:
            control = np.zeros(CYCLE, dtype=np.uint8)
        in_relation_module(
            state[hub], control, inverse=inverse, stats=stats
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
        rotate_state(
            state, index, program.family, inverse=False, stats=stats
        )
        if layer_order == "OUT_IN":
            layers = (apply_out_layer, apply_in_layer)
        elif layer_order == "IN_OUT":
            layers = (apply_in_layer, apply_out_layer)
        else:
            fail("unknown open-relation layer order")
        for layer in layers:
            layer(
                state,
                index,
                program.family,
                inverse=False,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )


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
            layers = (apply_in_layer, apply_out_layer)
        elif assumed_layer_order == "IN_OUT":
            layers = (apply_out_layer, apply_in_layer)
        else:
            fail("unknown inverse open-relation layer order")
        for layer in layers:
            layer(
                state,
                index,
                program.family,
                inverse=True,
                port_enabled=True,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=None,
            )
        rotate_state(
            state, index, program.family, inverse=True, stats=None
        )


def forward(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner, "FORWARD_OPEN_RELATION")
    raw_forward(carrier.relations, program, stats=carrier.stats)
    carrier.stage = "FORWARDED_OPEN_RELATION"


def inverse(carrier: Carrier, program: Program, owner: int) -> None:
    require_owned(carrier, program, owner, "FORWARDED_OPEN_RELATION")
    raw_inverse(carrier.relations, program)
    carrier.stage = "RESTORED"
    carrier.active_program = None
    carrier.active_owner = None
    carrier.restoration_generation += 1


def boundary_from_relations(
    state: np.ndarray, program: Program
) -> tuple[int, ...]:
    accumulator = np.zeros(CYCLE, dtype=np.int64)
    for node in range(NODE_COUNT):
        for slot in range(SLOTS):
            weight = (
                program.observation_quadratic * node * node
                + program.observation_linear * (slot + 1)
                + 7 * node * (slot + 1)
                + 1
            ) % MODULUS
            accumulator += weight * state[node, slot].astype(np.int64)
    return tuple(int(value) for value in np.mod(accumulator, MODULUS))


def project(
    carrier: Carrier, program: Program, owner: int
) -> tuple[int, ...]:
    require_owned(carrier, program, owner, "FORWARDED_OPEN_RELATION")
    carrier.projection_calls += 1
    return boundary_from_relations(carrier.relations, program)


def project_resident_relation(_carrier: Carrier, _port: int) -> None:
    fail("resident open-relation projection is forbidden")


@dataclass
class SpectralStats:
    composition_pointwise_products: int = 0
    intersection_mode_convolutions: int = 0
    composition_field_multiplications: int = 0
    intersection_field_multiplications: int = 0
    shear_additions: int = 0
    rotations: int = 0

    def descriptor(self) -> dict[str, int]:
        return {
            "composition_pointwise_products": self.composition_pointwise_products,
            "intersection_mode_convolutions": self.intersection_mode_convolutions,
            "composition_field_multiplications": self.composition_field_multiplications,
            "intersection_field_multiplications": self.intersection_field_multiplications,
            "total_bialgebra_field_multiplications": (
                self.composition_field_multiplications
                + self.intersection_field_multiplications
            ),
            "shear_additions": self.shear_additions,
            "rotations": self.rotations,
        }


def spectral_intersection(
    left: np.ndarray, right: np.ndarray, stats: SpectralStats | None
) -> np.ndarray:
    result = np.zeros(CYCLE, dtype=np.int64)
    for output in range(CYCLE):
        total = 0
        for left_mode in range(CYCLE):
            total += int(left[left_mode]) * int(
                right[(output - left_mode) % CYCLE]
            )
        result[output] = total * m151.m150.CYCLE_INVERSE % MODULUS
    if stats is not None:
        stats.intersection_mode_convolutions += 1
        stats.intersection_field_multiplications += CYCLE * CYCLE
    return result.astype(np.uint8)


def spectral_compose(
    left: np.ndarray, right: np.ndarray, stats: SpectralStats | None
) -> np.ndarray:
    result = m151.m150.mod_array(
        left.astype(np.int64) * right.astype(np.int64)
    )
    if stats is not None:
        stats.composition_pointwise_products += 1
        stats.composition_field_multiplications += CYCLE
    return result


def spectral_add(
    destination: np.ndarray,
    term: np.ndarray,
    *,
    inverse: bool,
    stats: SpectralStats | None,
) -> None:
    sign = -1 if inverse else 1
    destination[:] = m151.m150.mod_array(
        destination.astype(np.int64) + sign * term.astype(np.int64)
    )
    if stats is not None:
        stats.shear_additions += CYCLE


def spectral_out_module(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: SpectralStats | None,
) -> None:
    if inverse:
        term = spectral_intersection(control, destination[1], stats)
        spectral_add(destination[2], term, inverse=True, stats=stats)
        term = spectral_compose(control, destination[0], stats)
        spectral_add(destination[1], term, inverse=True, stats=stats)
    else:
        term = spectral_compose(control, destination[0], stats)
        spectral_add(destination[1], term, inverse=False, stats=stats)
        term = spectral_intersection(control, destination[1], stats)
        spectral_add(destination[2], term, inverse=False, stats=stats)


def spectral_in_module(
    destination: np.ndarray,
    control: np.ndarray,
    *,
    inverse: bool,
    stats: SpectralStats | None,
) -> None:
    if inverse:
        term = spectral_compose(control, destination[1], stats)
        spectral_add(destination[2], term, inverse=True, stats=stats)
        term = spectral_intersection(control, destination[0], stats)
        spectral_add(destination[1], term, inverse=True, stats=stats)
    else:
        term = spectral_intersection(control, destination[0], stats)
        spectral_add(destination[1], term, inverse=False, stats=stats)
        term = spectral_compose(control, destination[1], stats)
        spectral_add(destination[2], term, inverse=False, stats=stats)


def spectral_forward(program: Program) -> tuple[np.ndarray, SpectralStats]:
    modes = np.empty((NODE_COUNT, SLOTS, CYCLE), dtype=np.uint8)
    seed = seed_relations()
    for node in range(NODE_COUNT):
        for slot in range(SLOTS):
            modes[node, slot] = m151.m150.ntt(seed[node, slot])
    stats = SpectralStats()
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            shift = relation_phase_exponent(node, index, program.family)
            for slot in range(SLOTS):
                modes[node, slot] = m151.m150.rotate_modes(
                    modes[node, slot], shift
                )
                stats.rotations += 1
        hub = hub_index(index, program.family)
        for peer in peer_order(hub):
            control = m151.m150.rotate_modes(
                modes[hub, 0],
                relation_offset(hub, peer, index, program.family, 0),
            )
            spectral_out_module(
                modes[peer], control, inverse=False, stats=stats
            )
        for peer in peer_order(hub):
            control = m151.m150.rotate_modes(
                modes[peer, 2],
                relation_offset(peer, hub, index, program.family, 1),
            )
            spectral_in_module(
                modes[hub], control, inverse=False, stats=stats
            )
    state = np.empty_like(modes)
    for node in range(NODE_COUNT):
        for slot in range(SLOTS):
            state[node, slot] = m151.m150.inverse_ntt(
                modes[node, slot]
            )
    return state, stats


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    carrier = Carrier.seal()
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    final = state_commitment(carrier)
    boundary = project(carrier, program, program.owner)
    spectral_state, spectral_stats = spectral_forward(program)
    spectral_boundary = boundary_from_relations(spectral_state, program)
    support = np.count_nonzero(carrier.relations, axis=2)
    identical = np.array_equal(carrier.relations, spectral_state)
    inverse(carrier, program, program.owner)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial,
        "final_commitment": final,
        "boundary": list(boundary),
        "dual_spectral_boundary": list(spectral_boundary),
        "coefficient_state_identical_to_dual_spectral_recurrence": identical,
        "boundary_identical_to_dual_spectral_recurrence": boundary
        == spectral_boundary,
        "minimum_final_relation_support": int(np.min(support)),
        "maximum_final_relation_support": int(np.max(support)),
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation_before": generation,
        "restoration_generation_after": carrier.restoration_generation,
        "projection_calls": carrier.projection_calls,
        "snapshot_reload_used": carrier.snapshot_reload_used,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "phase_stats": carrier.stats.descriptor(),
        "dual_spectral_stats": spectral_stats.descriptor(),
    }


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    generation = carrier.restoration_generation
    begin_forward(carrier, program, program.owner)
    forward(carrier, program, program.owner)
    boundary = project(carrier, program, program.owner)
    final = state_commitment(carrier)
    inverse(carrier, program, program.owner)
    return {
        "boundary": list(boundary),
        "final_commitment": final,
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "generation_before": generation,
        "generation_after": carrier.restoration_generation,
        "resource_signature": {
            "resident_field_cells": int(carrier.relations.size),
            "resident_bytes": int(carrier.relations.nbytes),
            "phase_stats": carrier.stats.descriptor(),
        },
    }


def reuse_control() -> dict[str, Any]:
    carrier = Carrier.seal()
    first = transaction(carrier, compile_program(37, "PRIMARY"))
    backing = carrier.backing_identity
    second_program = compile_program(311, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier.seal(), second_program)
    return {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": carrier.backing_identity == backing,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": second["final_commitment"]
        == fresh["final_commitment"],
        "resource_signature_matches_fresh": second["resource_signature"]
        == fresh["resource_signature"],
        "restoration_generation": carrier.restoration_generation,
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }


def repeated_reuse_control() -> dict[str, Any]:
    carrier = Carrier.seal()
    backing = carrier.backing_identity
    initial = state_commitment(carrier)
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(64):
        result = transaction(carrier, compile_program(8, "REUSE"))
        boundaries.add(tuple(result["boundary"]))
    return {
        "cycles": 64,
        "exact_restoration": state_commitment(carrier) == initial,
        "same_backing": carrier.backing_identity == backing,
        "restoration_generation": carrier.restoration_generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": carrier.snapshot_reload_used,
    }


def streamed_semantic_controls() -> dict[str, Any]:
    left = seed_relations()[0, 0]
    right = seed_relations()[1, 1]
    composed = relation_compose(left, right, None)
    intersected = relation_intersect(left, right, None)
    composition_checks = 0
    intersection_checks = 0
    for x_value in range(CYCLE):
        for z_value in range(CYCLE):
            difference = (z_value - x_value) % CYCLE
            streamed_composition = 0
            for hidden_y in range(CYCLE):
                streamed_composition += int(
                    left[(hidden_y - x_value) % CYCLE]
                ) * int(right[(z_value - hidden_y) % CYCLE])
            if streamed_composition % MODULUS != int(composed[difference]):
                fail("streamed relation composition law failed")
            composition_checks += 1
            streamed_intersection = int(left[difference]) * int(
                right[difference]
            ) % MODULUS
            if streamed_intersection != int(intersected[difference]):
                fail("streamed relation intersection law failed")
            intersection_checks += 1
    return {
        "composition_scalar_checks": composition_checks,
        "intersection_scalar_checks": intersection_checks,
        "relation_tables_materialized": 0,
        "assignment_expansions_materialized": 0,
    }


def controls() -> dict[str, bool]:
    program = compile_program(4, "PRIMARY")
    seed = seed_relations()
    missing = seed.copy()
    raw_forward(missing, program)
    wrong = seed.copy()
    raw_forward(wrong, program)
    raw_inverse(wrong, program, offset_mutation=1)
    reordered = seed.copy()
    raw_forward(reordered, program)
    raw_inverse(reordered, program, assumed_layer_order="IN_OUT")
    normal = seed.copy()
    raw_forward(normal, program)
    disabled = seed.copy()
    raw_forward(disabled, program, port_enabled=False)
    swapped = seed.copy()
    raw_forward(swapped, program, layer_order="IN_OUT")
    mutated = seed.copy()
    raw_forward(mutated, program, hub_mutation=1)

    null_rejected = False
    try:
        begin_forward(None, program, program.owner)  # type: ignore[arg-type]
    except RuntimeError:
        null_rejected = True
    wrong_type_rejected = False
    wrong_type = Carrier.seal()
    wrong_type.port_type = "F103_SCALAR"
    try:
        begin_forward(wrong_type, program, program.owner)
    except RuntimeError:
        wrong_type_rejected = True
    wrong_owner_rejected = False
    try:
        begin_forward(Carrier.seal(), program, program.owner ^ 1)
    except RuntimeError:
        wrong_owner_rejected = True
    leased = Carrier.seal()
    begin_forward(leased, program, program.owner)
    premature_rejected = False
    try:
        project(leased, program, program.owner)
    except RuntimeError:
        premature_rejected = True
    resident_rejected = False
    try:
        project_resident_relation(leased, 0)
    except RuntimeError:
        resident_rejected = True

    control = seed[0, 0]
    destination = seed[1].copy()
    compose_then_intersect = destination.copy()
    out_relation_module(
        compose_then_intersect, control, inverse=False, stats=None
    )
    intersect_then_compose = destination.copy()
    in_relation_module(
        intersect_then_compose, control, inverse=False, stats=None
    )
    return {
        "missing_inverse_changes_state": not np.array_equal(missing, seed),
        "wrong_inverse_changes_state": not np.array_equal(wrong, seed),
        "reordered_inverse_changes_state": not np.array_equal(reordered, seed),
        "null_carrier_rejected": null_rejected,
        "wrong_relation_type_rejected": wrong_type_rejected,
        "wrong_owner_rejected": wrong_owner_rejected,
        "premature_projection_rejected": premature_rejected,
        "resident_relation_projection_rejected": resident_rejected,
        "null_port_changes_boundary": boundary_from_relations(normal, program)
        != boundary_from_relations(disabled, program),
        "out_in_order_changes_boundary": boundary_from_relations(normal, program)
        != boundary_from_relations(swapped, program),
        "topology_mutation_changes_boundary": boundary_from_relations(normal, program)
        != boundary_from_relations(mutated, program),
        "composition_intersection_order_noncommutes": not np.array_equal(
            compose_then_intersect, intersect_then_compose
        ),
        "composition_not_intersection": not np.array_equal(
            relation_compose(seed[0, 0], seed[1, 0], None),
            relation_intersect(seed[0, 0], seed[1, 0], None),
        ),
    }


def run() -> dict[str, Any]:
    cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    all_exact = all(
        case["coefficient_state_identical_to_dual_spectral_recurrence"]
        and case["boundary_identical_to_dual_spectral_recurrence"]
        and case["exact_restoration"]
        and case["same_backing"]
        and case["restoration_generation_after"]
        == case["restoration_generation_before"] + 1
        and case["projection_calls"] == 1
        and not case["snapshot_reload_used"]
        and case["inverse_history_cells"] == 0
        and case["retained_restoration_baseline_cells"] == 0
        for case in cases
    )
    if not all_exact:
        fail("one or more open-relation cases failed")
    control_results = controls()
    if not all(control_results.values()):
        failed = [name for name, value in control_results.items() if not value]
        fail(f"open-relation controls failed: {failed}")
    semantic = streamed_semantic_controls()
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
        fail("open-relation unrelated reuse failed")
    repeated = repeated_reuse_control()
    if not (
        repeated["exact_restoration"]
        and repeated["same_backing"]
        and repeated["restoration_generation"] == 64
        and repeated["stable_boundary_count"] == 1
        and not repeated["snapshot_reload_used"]
    ):
        fail("open-relation repeated reuse failed")

    resident_cells = NODE_COUNT * SLOTS * CYCLE
    resident_bytes = resident_cells
    maximum_program_bytes = max(
        case["public_program_json_bytes"] for case in cases
    )
    maximum_scratch = max(
        case["phase_stats"]["maximum_named_scratch_bytes"]
        for case in cases
    )
    maximum_phase_work = max(
        case["phase_stats"]["total_bialgebra_field_multiplications"]
        for case in cases
    )
    maximum_spectral_work = max(
        case["dual_spectral_stats"]["total_bialgebra_field_multiplications"]
        for case in cases
    )
    cancellations = sum(
        case["phase_stats"]["destructive_interference_events"]
        for case in cases
    )
    warm_live = resident_bytes + maximum_scratch + maximum_program_bytes
    return {
        "schema": "CAT_CAS_F103_C17_OPEN_RELATION_BIALGEBRA_NO_GO_RESULT_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_dependency": {
            "file": Path(m151.__file__).name,
            "sha256": hashlib.sha256(Path(m151.__file__).read_bytes()).hexdigest(),
            "used_for": "EXACT_C17_ARITHMETIC_AND_TRANSFORM_PRIMITIVES_ONLY",
        },
        "source_scope": "LINUX_DIRECT_PROCESS_EXACT_FINITE_FIELD_TYPED_OPEN_RELATION_SOFTWARE",
        "execution_scope": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "all_cases_exact": all_exact,
            "public_topology_compilation_reads_final_answers": False,
            "catvm_machine_boundary_used": False,
        },
        "relation_law": {
            "field_modulus": MODULUS,
            "cycle_order": CYCLE,
            "port_type": RELATION_TYPE,
            "open_boundary_arity": 2,
            "signature_coordinates_per_relation": CYCLE,
            "implicit_dense_relation_cells_per_signature": CYCLE * CYCLE,
            "materialized_dense_relation_table_cells": 0,
            "materialized_assignment_expansion_cells": 0,
            "hidden_interface_composition": "CYCLIC_CONVOLUTION",
            "parallel_intersection": "HADAMARD_PRODUCT",
            "shared_unresolved_port_consumers_per_out_layer": 8,
            "reciprocal_consumers_per_in_layer": 8,
            "resident_relation_projection_before_boundary": False,
            "destructive_interference_events_observed": cancellations,
            "streamed_semantic_controls": semantic,
        },
        "carrier_law": {
            "logical_relation_factors": NODE_COUNT * SLOTS,
            "resident_field_coordinates": resident_cells,
            "resident_bytes": resident_bytes,
            "typed_owner_checked": True,
            "exact_generation_checked": True,
            "retained_public_plan_cells": 0,
        },
        "matched_classical_recurrences": {
            "identical_coefficient_recurrence": "EXECUTED_BY_INDEPENDENT_ORACLE_OVER459_FIELD_COORDINATES",
            "dual_spectral_recurrence": "EXECUTED17_MODE_RECURRENCE_SWAPPING_CONVOLUTION_AND_HADAMARD_COSTS",
            "full_state_and_boundary_match_every_case": True,
            "resident_field_coordinates_each": resident_cells,
            "resident_bytes_each": resident_bytes,
            "maximum_named_warm_live_bytes_each": warm_live,
            "maximum_phase_bialgebra_multiplications": maximum_phase_work,
            "maximum_dual_spectral_bialgebra_multiplications": maximum_spectral_work,
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
            "dual_spectral_resident_uint8_cells": resident_cells,
            "dual_spectral_resident_bytes": resident_bytes,
            "maximum_named_warm_live_bytes_each": warm_live,
            "phase_to_dual_spectral_resident_dimension_ratio": 1,
            "excluded": [
                "PYTHON_CONTAINER_OVERHEAD",
                "PYTHON_OBJECT_ALLOCATOR",
                "NUMPY_AND_NATIVE_LIBRARY_INTERNAL_STORAGE",
                "WHOLE_PROCESS_PEAK",
            ],
        },
        "cases": cases,
        "claim_ceiling": (
            "F103_TRANSLATION_INVARIANT_WEIGHTED_C17_TO_C17_OPEN_RELATIONS_"
            "ON_THE_DECLARED9_NODE_THREE_SLOT_ROTATING_HUB_TOPOLOGY_ACROSS_"
            "18_CASES_THROUGH_DEPTH512_IN_LINUX_DIRECT_PROCESS_SOFTWARE"
        ),
        "not_established": [
            "NON_TRANSLATION_INVARIANT_RELATIONS",
            "ARBITRARY_PORT_ARITY_OR_GRAPH_TOPOLOGY",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "PHYSICAL_BIT_REPLACEMENT",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": (
            "THE_TYPED_OPEN_RELATION_BIALGEBRA_AVOIDS_DENSE_TABLES_AND_"
            "COMPOSES_AND_INTERSECTS_NATIVELY_BUT_TRANSLATION_INVARIANCE_"
            "REDUCES_EVERY_RELATION_TO17_CLASSICAL_COEFFICIENTS"
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
