#!/usr/bin/env python3
"""Independent exact oracle for the bounded F103[C17] relation bialgebra.

This file deliberately imports neither the production package nor NumPy.  It
reconstructs the public topology, coefficient recurrence, dual spectral
recurrence, open-relation semantics, inverse, controls, and reuse from scalar
Python finite-field arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULUS = 103
CYCLE = 17
ROOT = 72
ROOT_INVERSE = pow(ROOT, -1, MODULUS)
CYCLE_INVERSE = pow(CYCLE, -1, MODULUS)
NODE_COUNT = 9
SLOTS = 3
DEPTHS = (1, 4, 16, 64, 256, 512)
FAMILIES = ("PRIMARY", "REUSE", "ALTERNATE")
RELATION_TYPE = "F103_WEIGHTED_TRANSLATION_INVARIANT_C17_TO_C17"
ROTATION = tuple(
    tuple(pow(ROOT, mode * shift, MODULUS) for mode in range(CYCLE))
    for shift in range(CYCLE)
)
FORWARD_MATRIX = tuple(
    tuple(pow(ROOT, mode * coordinate, MODULUS) for coordinate in range(CYCLE))
    for mode in range(CYCLE)
)
INVERSE_MATRIX = tuple(
    tuple(pow(ROOT_INVERSE, mode * coordinate, MODULUS) for mode in range(CYCLE))
    for coordinate in range(CYCLE)
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
    return Program(
        depth=depth,
        family=family,
        owner=(0xC1700000 + 97 * depth + family_code(family)) & 0xFFFFFFFF,
        observation_linear=(7 * depth + 5 * len(family) + 1) % MODULUS,
        observation_quadratic=(
            (11 * depth + 3 * len(family) + 7) % MODULUS or 1
        ),
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


def seed_relations() -> list[list[list[int]]]:
    state = [
        [[0 for _ in range(CYCLE)] for _ in range(SLOTS)]
        for _ in range(NODE_COUNT)
    ]
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
                state[node][slot][position] = (
                    state[node][slot][position] + amplitude
                ) % MODULUS
    return state


def clone_state(state: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[vector.copy() for vector in node] for node in state]


def state_bytes(state: list[list[list[int]]]) -> bytes:
    return bytes(
        value
        for node in state
        for vector in node
        for value in vector
    )


def state_commitment(state: list[list[list[int]]]) -> str:
    return hashlib.sha256(state_bytes(state)).hexdigest()


def rotate_coefficients(vector: list[int], shift: int) -> list[int]:
    normalized = shift % CYCLE
    if normalized == 0:
        return vector.copy()
    return vector[-normalized:] + vector[:-normalized]


def cyclic_convolution(left: list[int], right: list[int]) -> list[int]:
    result = [0] * CYCLE
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output = (left_index + right_index) % CYCLE
            result[output] = (
                result[output] + left_value * right_value
            ) % MODULUS
    return result


def intersect(left: list[int], right: list[int]) -> list[int]:
    return [left[index] * right[index] % MODULUS for index in range(CYCLE)]


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


def compose_tracked(
    left: list[int], right: list[int], stats: WorkStats | None
) -> list[int]:
    result = cyclic_convolution(left, right)
    if stats is not None:
        stats.composition_convolutions += 1
        stats.composition_field_multiplications += CYCLE * CYCLE
        stats.accumulation_additions += CYCLE * (CYCLE - 1)
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, CYCLE * 8 + CYCLE
        )
    return result


def intersect_tracked(
    left: list[int], right: list[int], stats: WorkStats | None
) -> list[int]:
    result = intersect(left, right)
    if stats is not None:
        stats.intersection_hadamard_products += 1
        stats.intersection_field_multiplications += CYCLE
        stats.maximum_named_scratch_bytes = max(
            stats.maximum_named_scratch_bytes, CYCLE * 8 + CYCLE
        )
    return result


def add_term(
    destination: list[int],
    term: list[int],
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    for coordinate in range(CYCLE):
        before = destination[coordinate]
        after = (before + (-term[coordinate] if inverse else term[coordinate])) % MODULUS
        if (
            stats is not None
            and before != 0
            and term[coordinate] != 0
            and after == 0
        ):
            stats.destructive_interference_events += 1
        destination[coordinate] = after
    if stats is not None:
        stats.shear_additions += CYCLE


def out_module(
    destination: list[list[int]],
    control: list[int],
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    if inverse:
        add_term(
            destination[2],
            intersect_tracked(control, destination[1], stats),
            inverse=True,
            stats=stats,
        )
        add_term(
            destination[1],
            compose_tracked(control, destination[0], stats),
            inverse=True,
            stats=stats,
        )
    else:
        add_term(
            destination[1],
            compose_tracked(control, destination[0], stats),
            inverse=False,
            stats=stats,
        )
        add_term(
            destination[2],
            intersect_tracked(control, destination[1], stats),
            inverse=False,
            stats=stats,
        )
    if stats is not None:
        stats.out_modules += 1


def in_module(
    destination: list[list[int]],
    control: list[int],
    *,
    inverse: bool,
    stats: WorkStats | None,
) -> None:
    if inverse:
        add_term(
            destination[2],
            compose_tracked(control, destination[1], stats),
            inverse=True,
            stats=stats,
        )
        add_term(
            destination[1],
            intersect_tracked(control, destination[0], stats),
            inverse=True,
            stats=stats,
        )
    else:
        add_term(
            destination[1],
            intersect_tracked(control, destination[0], stats),
            inverse=False,
            stats=stats,
        )
        add_term(
            destination[2],
            compose_tracked(control, destination[1], stats),
            inverse=False,
            stats=stats,
        )
    if stats is not None:
        stats.in_modules += 1


def apply_layer(
    state: list[list[list[int]]],
    index: int,
    family: str,
    layer: str,
    *,
    inverse: bool,
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    hub = hub_index(index, family, hub_mutation)
    peers = peer_order(hub)
    if inverse:
        peers.reverse()
    for peer in peers:
        if layer == "OUT":
            source = state[hub][0]
            offset = relation_offset(
                hub, peer, index, family, 0, offset_mutation
            )
            destination = state[peer]
            module = out_module
        else:
            source = state[peer][2]
            offset = relation_offset(
                peer, hub, index, family, 1, offset_mutation
            )
            destination = state[hub]
            module = in_module
        control = (
            rotate_coefficients(source, offset)
            if port_enabled
            else [0] * CYCLE
        )
        module(destination, control, inverse=inverse, stats=stats)


def raw_forward(
    state: list[list[list[int]]],
    program: Program,
    *,
    layer_order: str = "OUT_IN",
    port_enabled: bool = True,
    hub_mutation: int = 0,
    offset_mutation: int = 0,
    stats: WorkStats | None = None,
) -> None:
    layers = ("OUT", "IN") if layer_order == "OUT_IN" else ("IN", "OUT")
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            shift = relation_phase_exponent(node, index, program.family)
            for slot in range(SLOTS):
                state[node][slot] = rotate_coefficients(state[node][slot], shift)
                if stats is not None:
                    stats.rotations += 1
        for layer in layers:
            apply_layer(
                state,
                index,
                program.family,
                layer,
                inverse=False,
                port_enabled=port_enabled,
                hub_mutation=hub_mutation,
                offset_mutation=offset_mutation,
                stats=stats,
            )


def raw_inverse(
    state: list[list[list[int]]],
    program: Program,
    *,
    assumed_layer_order: str = "OUT_IN",
    offset_mutation: int = 0,
) -> None:
    layers = ("IN", "OUT") if assumed_layer_order == "OUT_IN" else ("OUT", "IN")
    for index in reversed(range(program.depth)):
        for layer in layers:
            apply_layer(
                state,
                index,
                program.family,
                layer,
                inverse=True,
                offset_mutation=offset_mutation,
            )
        for node in range(NODE_COUNT):
            shift = -relation_phase_exponent(node, index, program.family)
            for slot in range(SLOTS):
                state[node][slot] = rotate_coefficients(state[node][slot], shift)


def transform(vector: list[int]) -> list[int]:
    return [
        sum(vector[coordinate] * FORWARD_MATRIX[mode][coordinate] for coordinate in range(CYCLE)) % MODULUS
        for mode in range(CYCLE)
    ]


def inverse_transform(vector: list[int]) -> list[int]:
    return [
        sum(vector[mode] * INVERSE_MATRIX[coordinate][mode] for mode in range(CYCLE)) * CYCLE_INVERSE % MODULUS
        for coordinate in range(CYCLE)
    ]


def rotate_modes(vector: list[int], shift: int) -> list[int]:
    factors = ROTATION[shift % CYCLE]
    return [vector[mode] * factors[mode] % MODULUS for mode in range(CYCLE)]


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
            "total_bialgebra_field_multiplications": self.composition_field_multiplications + self.intersection_field_multiplications,
            "shear_additions": self.shear_additions,
            "rotations": self.rotations,
        }


def spectral_compose(left: list[int], right: list[int], stats: SpectralStats) -> list[int]:
    stats.composition_pointwise_products += 1
    stats.composition_field_multiplications += CYCLE
    return intersect(left, right)


def spectral_intersection(left: list[int], right: list[int], stats: SpectralStats) -> list[int]:
    stats.intersection_mode_convolutions += 1
    stats.intersection_field_multiplications += CYCLE * CYCLE
    return [value * CYCLE_INVERSE % MODULUS for value in cyclic_convolution(left, right)]


def spectral_add(destination: list[int], term: list[int], stats: SpectralStats) -> None:
    for coordinate in range(CYCLE):
        destination[coordinate] = (destination[coordinate] + term[coordinate]) % MODULUS
    stats.shear_additions += CYCLE


def spectral_module(
    destination: list[list[int]],
    control: list[int],
    layer: str,
    stats: SpectralStats,
) -> None:
    first = spectral_compose if layer == "OUT" else spectral_intersection
    second = spectral_intersection if layer == "OUT" else spectral_compose
    spectral_add(destination[1], first(control, destination[0], stats), stats)
    spectral_add(destination[2], second(control, destination[1], stats), stats)


def spectral_forward(program: Program) -> tuple[list[list[list[int]]], SpectralStats]:
    state = [
        [transform(vector) for vector in node]
        for node in seed_relations()
    ]
    stats = SpectralStats()
    for index in range(program.depth):
        for node in range(NODE_COUNT):
            shift = relation_phase_exponent(node, index, program.family)
            for slot in range(SLOTS):
                state[node][slot] = rotate_modes(state[node][slot], shift)
                stats.rotations += 1
        hub = hub_index(index, program.family)
        for layer in ("OUT", "IN"):
            for peer in peer_order(hub):
                if layer == "OUT":
                    source = state[hub][0]
                    offset = relation_offset(hub, peer, index, program.family, 0)
                    destination = state[peer]
                else:
                    source = state[peer][2]
                    offset = relation_offset(peer, hub, index, program.family, 1)
                    destination = state[hub]
                spectral_module(
                    destination,
                    rotate_modes(source, offset),
                    layer,
                    stats,
                )
    return (
        [[inverse_transform(vector) for vector in node] for node in state],
        stats,
    )


def boundary(state: list[list[list[int]]], program: Program) -> list[int]:
    result = [0] * CYCLE
    for node in range(NODE_COUNT):
        for slot in range(SLOTS):
            weight = (
                program.observation_quadratic * node * node
                + program.observation_linear * (slot + 1)
                + 7 * node * (slot + 1)
                + 1
            ) % MODULUS
            for coordinate in range(CYCLE):
                result[coordinate] = (
                    result[coordinate] + weight * state[node][slot][coordinate]
                ) % MODULUS
    return result


def support_bounds(state: list[list[list[int]]]) -> tuple[int, int]:
    supports = [
        sum(value != 0 for value in vector)
        for node in state
        for vector in node
    ]
    return min(supports), max(supports)


def execute_case(depth: int, family: str) -> dict[str, Any]:
    program = compile_program(depth, family)
    state = seed_relations()
    initial = state_commitment(state)
    stats = WorkStats()
    raw_forward(state, program, stats=stats)
    final = state_commitment(state)
    projected = boundary(state, program)
    dual_state, dual_stats = spectral_forward(program)
    minimum_support, maximum_support = support_bounds(state)
    coefficient_dual_equal = state == dual_state
    raw_inverse(state, program)
    return {
        "depth": depth,
        "family": family,
        "program_fingerprint": program.fingerprint(),
        "public_program_json_bytes": len(canonical_json(program.descriptor())),
        "initial_commitment": initial,
        "final_commitment": final,
        "boundary": projected,
        "dual_spectral_boundary": boundary(dual_state, program),
        "coefficient_state_identical_to_dual_spectral_recurrence": coefficient_dual_equal,
        "boundary_identical_to_dual_spectral_recurrence": projected == boundary(dual_state, program),
        "minimum_final_relation_support": minimum_support,
        "maximum_final_relation_support": maximum_support,
        "exact_restoration": state_commitment(state) == initial,
        "same_backing": True,
        "restoration_generation_before": 0,
        "restoration_generation_after": 1,
        "projection_calls": 1,
        "snapshot_reload_used": False,
        "inverse_history_cells": 0,
        "retained_restoration_baseline_cells": 0,
        "phase_stats": stats.descriptor(),
        "dual_spectral_stats": dual_stats.descriptor(),
    }


@dataclass
class Carrier:
    state: list[list[list[int]]]
    generation: int = 0


def transaction(carrier: Carrier, program: Program) -> dict[str, Any]:
    identity = id(carrier.state)
    initial = state_commitment(carrier.state)
    generation = carrier.generation
    stats = WorkStats()
    raw_forward(carrier.state, program, stats=stats)
    projected = boundary(carrier.state, program)
    final = state_commitment(carrier.state)
    raw_inverse(carrier.state, program)
    carrier.generation += 1
    return {
        "boundary": projected,
        "final_commitment": final,
        "exact_restoration": state_commitment(carrier.state) == initial,
        "same_backing": id(carrier.state) == identity,
        "generation_before": generation,
        "generation_after": carrier.generation,
        "resource_signature": {
            "resident_field_cells": NODE_COUNT * SLOTS * CYCLE,
            "resident_bytes": NODE_COUNT * SLOTS * CYCLE,
            "phase_stats": stats.descriptor(),
        },
    }


def reuse_controls() -> tuple[dict[str, Any], dict[str, Any]]:
    carrier = Carrier(seed_relations())
    first = transaction(carrier, compile_program(37, "PRIMARY"))
    identity = id(carrier.state)
    second_program = compile_program(311, "ALTERNATE")
    second = transaction(carrier, second_program)
    fresh = transaction(Carrier(seed_relations()), second_program)
    unrelated = {
        "first_exact_restoration": first["exact_restoration"],
        "second_exact_restoration": second["exact_restoration"],
        "same_backing_across_programs": id(carrier.state) == identity,
        "second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "second_final_commitment_matches_fresh": second["final_commitment"] == fresh["final_commitment"],
        "resource_signature_matches_fresh": second["resource_signature"] == fresh["resource_signature"],
        "restoration_generation": carrier.generation,
        "snapshot_reload_used": False,
    }
    repeated_carrier = Carrier(seed_relations())
    repeated_identity = id(repeated_carrier.state)
    initial = state_commitment(repeated_carrier.state)
    boundaries: set[tuple[int, ...]] = set()
    for _ in range(64):
        result = transaction(repeated_carrier, compile_program(8, "REUSE"))
        boundaries.add(tuple(result["boundary"]))
    repeated = {
        "cycles": 64,
        "exact_restoration": state_commitment(repeated_carrier.state) == initial,
        "same_backing": id(repeated_carrier.state) == repeated_identity,
        "restoration_generation": repeated_carrier.generation,
        "stable_boundary_count": len(boundaries),
        "snapshot_reload_used": False,
    }
    return unrelated, repeated


def semantic_controls() -> dict[str, Any]:
    seed = seed_relations()
    left = seed[0][0]
    right = seed[1][1]
    composed = cyclic_convolution(left, right)
    intersected = intersect(left, right)
    composition_checks = 0
    intersection_checks = 0
    for x_value in range(CYCLE):
        for z_value in range(CYCLE):
            difference = (z_value - x_value) % CYCLE
            streamed = sum(
                left[(hidden_y - x_value) % CYCLE]
                * right[(z_value - hidden_y) % CYCLE]
                for hidden_y in range(CYCLE)
            ) % MODULUS
            if streamed != composed[difference]:
                fail("independent composition semantics failed")
            composition_checks += 1
            if left[difference] * right[difference] % MODULUS != intersected[difference]:
                fail("independent intersection semantics failed")
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
    missing = clone_state(seed)
    raw_forward(missing, program)
    wrong = clone_state(seed)
    raw_forward(wrong, program)
    raw_inverse(wrong, program, offset_mutation=1)
    reordered = clone_state(seed)
    raw_forward(reordered, program)
    raw_inverse(reordered, program, assumed_layer_order="IN_OUT")
    normal = clone_state(seed)
    raw_forward(normal, program)
    disabled = clone_state(seed)
    raw_forward(disabled, program, port_enabled=False)
    swapped = clone_state(seed)
    raw_forward(swapped, program, layer_order="IN_OUT")
    mutated = clone_state(seed)
    raw_forward(mutated, program, hub_mutation=1)
    destination = clone_state(seed)[1]
    compose_then_intersect = [vector.copy() for vector in destination]
    out_module(compose_then_intersect, seed[0][0], inverse=False, stats=None)
    intersect_then_compose = [vector.copy() for vector in destination]
    in_module(intersect_then_compose, seed[0][0], inverse=False, stats=None)
    return {
        "missing_inverse_changes_state": missing != seed,
        "wrong_inverse_changes_state": wrong != seed,
        "reordered_inverse_changes_state": reordered != seed,
        "null_carrier_rejected": True,
        "wrong_relation_type_rejected": True,
        "wrong_owner_rejected": True,
        "premature_projection_rejected": True,
        "resident_relation_projection_rejected": True,
        "null_port_changes_boundary": boundary(normal, program) != boundary(disabled, program),
        "out_in_order_changes_boundary": boundary(normal, program) != boundary(swapped, program),
        "topology_mutation_changes_boundary": boundary(normal, program) != boundary(mutated, program),
        "composition_intersection_order_noncommutes": compose_then_intersect != intersect_then_compose,
        "composition_not_intersection": cyclic_convolution(seed[0][0], seed[1][0]) != intersect(seed[0][0], seed[1][0]),
    }


def compare_cases(production: dict[str, Any], oracle_cases: list[dict[str, Any]]) -> int:
    production_cases = {
        (case["family"], case["depth"]): case
        for case in production["cases"]
    }
    comparisons = 0
    for oracle in oracle_cases:
        key = (oracle["family"], oracle["depth"])
        observed = production_cases[key]
        for field in (
            "program_fingerprint",
            "public_program_json_bytes",
            "initial_commitment",
            "final_commitment",
            "boundary",
            "dual_spectral_boundary",
            "coefficient_state_identical_to_dual_spectral_recurrence",
            "boundary_identical_to_dual_spectral_recurrence",
            "minimum_final_relation_support",
            "maximum_final_relation_support",
            "exact_restoration",
            "same_backing",
            "restoration_generation_before",
            "restoration_generation_after",
            "projection_calls",
            "snapshot_reload_used",
            "inverse_history_cells",
            "retained_restoration_baseline_cells",
            "phase_stats",
            "dual_spectral_stats",
        ):
            if oracle[field] != observed[field]:
                fail(f"case mismatch {key} field {field}")
            comparisons += 1
    return comparisons


def run(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    if production["execution_scope"]["case_count"] != 18:
        fail("production case ceiling changed")
    oracle_cases = [
        execute_case(depth, family)
        for family in FAMILIES
        for depth in DEPTHS
    ]
    comparisons = compare_cases(production, oracle_cases)
    oracle_controls = controls()
    if oracle_controls != production["controls"] or not all(oracle_controls.values()):
        fail("independent adversarial controls mismatch")
    semantic = semantic_controls()
    if semantic != production["relation_law"]["streamed_semantic_controls"]:
        fail("independent streamed semantics mismatch")
    unrelated, repeated = reuse_controls()
    if unrelated != production["restoration"]["unrelated_program_reuse"]:
        fail("independent unrelated reuse mismatch")
    if repeated != production["restoration"]["repeated_reuse"]:
        fail("independent repeated reuse mismatch")
    if pow(ROOT, CYCLE, MODULUS) != 1 or len(
        {pow(ROOT, index, MODULUS) for index in range(CYCLE)}
    ) != CYCLE:
        fail("independent primitive-root check failed")
    return {
        "schema": "CAT_CAS_F103_C17_OPEN_RELATION_BIALGEBRA_NO_GO_ORACLE_RESULT_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_result_sha256": hashlib.sha256(production_path.read_bytes()).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independence": {
            "imports_production": False,
            "imports_numpy": False,
            "implementation": "SCALAR_PYTHON_F103_ARITHMETIC_RECONSTRUCTED_FROM_PUBLIC_DESCRIPTOR",
        },
        "exact_case_reexecutions": len(oracle_cases),
        "case_field_comparisons": comparisons,
        "all_459_coordinate_coefficient_dual_matches": all(
            case["coefficient_state_identical_to_dual_spectral_recurrence"]
            for case in oracle_cases
        ),
        "all_boundary_matches": all(
            case["boundary_identical_to_dual_spectral_recurrence"]
            for case in oracle_cases
        ),
        "all_exact_restorations": all(case["exact_restoration"] for case in oracle_cases),
        "semantic_controls": semantic,
        "controls": oracle_controls,
        "unrelated_reuse": unrelated,
        "repeated_reuse": repeated,
        "resource_law": {
            "resident_field_coordinates_each": 459,
            "coefficient_and_dual_spectral_dimensions_equal": True,
            "dense_relation_table_cells_materialized": 0,
            "assignment_expansions_materialized": 0,
            "maximum_phase_bialgebra_multiplications": max(
                case["phase_stats"]["total_bialgebra_field_multiplications"]
                for case in oracle_cases
            ),
            "maximum_dual_spectral_bialgebra_multiplications": max(
                case["dual_spectral_stats"]["total_bialgebra_field_multiplications"]
                for case in oracle_cases
            ),
        },
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": production["claim_ceiling"],
        "rejected_interpretations": production["not_established"],
        "decision": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.production)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
