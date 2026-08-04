#!/usr/bin/env python3
"""Exact growing-dihedral irrep-sparse relation diagnostic over F103."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


FIELD = 103
GENERATOR = 5
ROTATION_ORDERS = (3, 6, 17, 34, 51)
MAX_DEPTH = 16
CHECKPOINTS = (1, 2, 4, 8, 16)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_GROWING_F103_DIHEDRAL_TRANSLATION_RELATION_IRREP_"
    "SPARSE_PHASE_ALGEBRA_NO_GO_FINDS_ADAPTIVE_BLOCK_SUPPORT_SATURATES_"
    "THE_FULL_TWO_PORT_GROUP_ALGEBRA_CAPACITY_ON_D6_D12_D34_D68_D102_"
    "BY_DEPTH16_UNDER_NONCOMMUTATIVE_COMPOSITION_AND_HADAMARD_INTERSECTION_"
    "WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def phase(exponent: int) -> int:
    return pow(GENERATOR, exponent % (FIELD - 1), FIELD)


@lru_cache(maxsize=None)
def root_for(rotation_order: int) -> int:
    if (FIELD - 1) % rotation_order:
        fail("rotation order does not split over F103")
    root = pow(GENERATOR, (FIELD - 1) // rotation_order, FIELD)
    if pow(root, rotation_order, FIELD) != 1 or any(
        pow(root, exponent, FIELD) == 1 for exponent in range(1, rotation_order)
    ):
        fail("invalid primitive rotation root")
    return root


Element = tuple[int, int]


@lru_cache(maxsize=None)
def elements(rotation_order: int) -> tuple[Element, ...]:
    return tuple((rotation, reflection) for reflection in (0, 1) for rotation in range(rotation_order))


def multiply(rotation_order: int, left: Element, right: Element) -> Element:
    rotation = (left[0] + (-1 if left[1] else 1) * right[0]) % rotation_order
    return rotation, left[1] ^ right[1]


def inverse(rotation_order: int, element: Element) -> Element:
    return ((element[0] if element[1] else -element[0]) % rotation_order, element[1])


@dataclass(frozen=True)
class Irrep:
    label: str
    dimension: int
    rotation_sign: int = 1
    reflection_sign: int = 1
    frequency: int = 0


@lru_cache(maxsize=None)
def irreps(rotation_order: int) -> tuple[Irrep, ...]:
    result: list[Irrep] = []
    rotation_signs = (1, -1) if rotation_order % 2 == 0 else (1,)
    for rotation_sign in rotation_signs:
        for reflection_sign in (1, -1):
            r_label = "P" if rotation_sign == 1 else "M"
            s_label = "P" if reflection_sign == 1 else "M"
            result.append(Irrep(f"L{r_label}{s_label}", 1, rotation_sign, reflection_sign))
    maximum = rotation_order // 2 - 1 if rotation_order % 2 == 0 else (rotation_order - 1) // 2
    result.extend(Irrep(f"K{frequency}", 2, frequency=frequency) for frequency in range(1, maximum + 1))
    if sum(irrep.dimension * irrep.dimension for irrep in result) != 2 * rotation_order:
        fail("dihedral irrep dimensions do not sum to group order")
    return tuple(result)


def representation(rotation_order: int, irrep: Irrep, element: Element) -> list[int]:
    rotation, reflection = element
    if irrep.dimension == 1:
        return [
            pow(irrep.rotation_sign % FIELD, rotation, FIELD)
            * pow(irrep.reflection_sign % FIELD, reflection, FIELD)
            % FIELD
        ]
    root = root_for(rotation_order)
    positive = pow(root, irrep.frequency * rotation, FIELD)
    negative = pow(positive, -1, FIELD)
    if reflection:
        return [0, positive, negative, 0]
    return [positive, 0, 0, negative]


def matrix_multiply(left: list[int], right: list[int], dimension: int) -> list[int]:
    if dimension == 1:
        return [left[0] * right[0] % FIELD]
    return [
        (left[0] * right[0] + left[1] * right[2]) % FIELD,
        (left[0] * right[1] + left[1] * right[3]) % FIELD,
        (left[2] * right[0] + left[3] * right[2]) % FIELD,
        (left[2] * right[1] + left[3] * right[3]) % FIELD,
    ]


def matrix_trace_product(left: list[int], right: list[int], dimension: int) -> int:
    if dimension == 1:
        return left[0] * right[0] % FIELD
    return (
        left[0] * right[0]
        + left[1] * right[2]
        + left[2] * right[1]
        + left[3] * right[3]
    ) % FIELD


Blocks = dict[str, list[int]]


def canonical_blocks(blocks: Blocks) -> Blocks:
    return {label: [entry % FIELD for entry in value] for label, value in blocks.items() if any(entry % FIELD for entry in value)}


def blocks_state(blocks: Blocks) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((label, tuple(value)) for label, value in sorted(canonical_blocks(blocks).items()))


def active_capacity(rotation_order: int, blocks: Blocks) -> int:
    dimensions = {irrep.label: irrep.dimension for irrep in irreps(rotation_order)}
    return sum(dimensions[label] ** 2 for label in canonical_blocks(blocks))


def nonzero_coordinates(blocks: Blocks) -> int:
    return sum(sum(entry % FIELD != 0 for entry in value) for value in canonical_blocks(blocks).values())


def evaluate_blocks(rotation_order: int, blocks: Blocks, element: Element) -> int:
    order = 2 * rotation_order
    inverse_order = pow(order, -1, FIELD)
    total = 0
    inverted = inverse(rotation_order, element)
    for irrep in irreps(rotation_order):
        block = blocks.get(irrep.label)
        if block is None:
            continue
        rho = representation(rotation_order, irrep, inverted)
        total += irrep.dimension * matrix_trace_product(block, rho, irrep.dimension)
    return total * inverse_order % FIELD


def transform_vector(rotation_order: int, vector: list[int]) -> Blocks:
    result: Blocks = {}
    group = elements(rotation_order)
    for irrep in irreps(rotation_order):
        block = [0] * (irrep.dimension * irrep.dimension)
        for value, element in zip(vector, group, strict=True):
            rho = representation(rotation_order, irrep, element)
            for position, entry in enumerate(rho):
                block[position] = (block[position] + value * entry) % FIELD
        if any(block):
            result[irrep.label] = block
    return result


def seed_blocks(rotation_order: int, family: str, register: str) -> Blocks:
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 4
    if rotation_order < 3:
        fail("rotation order too small")
    return {
        "K1": [
            phase(code + offset + 1),
            phase(2 * code + offset + 3),
            phase(3 * code + offset + 5),
            phase(5 * code + offset + 7),
        ]
    }


def public_mask_blocks(rotation_order: int, index: int, family: str, kind: int) -> Blocks:
    code = 1 if family == "PRIMARY" else 2
    return {
        "LPP": [phase(index + code + 3 * kind)],
        "K1": [
            phase(index + code + kind + 1),
            phase(2 * index + code + 2 * kind + 3),
            phase(3 * index + 2 * code + kind + 5),
            phase(index + 3 * code + 4 * kind + 7),
        ],
    }


def public_convolution_block(irrep: Irrep, index: int, family: str, kind: int) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    if irrep.dimension == 1:
        return [phase(index + code + kind + irrep.rotation_sign + 2 * irrep.reflection_sign)]
    entry = phase(index + code + kind + irrep.frequency)
    if kind % 2:
        return [1, entry, 0, 1]
    return [1, 0, entry, 1]


def scalar_for(index: int, family: str, kind: int) -> int:
    code = 1 if family == "PRIMARY" else 2
    return phase((kind + 1) * index + kind * code + 1)


@dataclass(frozen=True)
class Shear:
    target: str
    operation: str
    index: int
    family: str
    kind: int
    scalar: int


def descriptors(index: int, family: str) -> tuple[Shear, ...]:
    stages = (
        Shear("B", "RIGHT_COMPOSE", index, family, 1, scalar_for(index, family, 1)),
        Shear("A", "INTERSECT", index, family, 2, scalar_for(index, family, 2)),
        Shear("B", "LEFT_COMPOSE", index, family, 3, scalar_for(index, family, 3)),
        Shear("A", "INTERSECT", index, family, 4, scalar_for(index, family, 4)),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def update_block(target: Blocks, irrep: Irrep, delta: list[int], scalar: int, subtracting: bool) -> None:
    sign = -1 if subtracting else 1
    current = target.get(irrep.label, [0] * (irrep.dimension * irrep.dimension))
    updated = [(value + sign * scalar * change) % FIELD for value, change in zip(current, delta, strict=True)]
    if any(updated):
        target[irrep.label] = updated
    else:
        target.pop(irrep.label, None)


def convolution_update(rotation_order: int, target: Blocks, source: Blocks, stage: Shear, subtracting: bool) -> None:
    by_label = {irrep.label: irrep for irrep in irreps(rotation_order)}
    for label, source_block in tuple(source.items()):
        irrep = by_label[label]
        public = public_convolution_block(irrep, stage.index, stage.family, stage.kind)
        if stage.operation == "RIGHT_COMPOSE":
            delta = matrix_multiply(source_block, public, irrep.dimension)
        else:
            delta = matrix_multiply(public, source_block, irrep.dimension)
        update_block(target, irrep, delta, stage.scalar, subtracting)


def hadamard_coefficient(
    rotation_order: int,
    source: Blocks,
    public: Blocks,
    output_irrep: Irrep,
    position: int,
) -> int:
    total = 0
    for element in elements(rotation_order):
        source_value = evaluate_blocks(rotation_order, source, element)
        public_value = evaluate_blocks(rotation_order, public, element)
        rho = representation(rotation_order, output_irrep, element)
        total = (total + source_value * public_value * rho[position]) % FIELD
    return total


def hadamard_update(rotation_order: int, target: Blocks, source: Blocks, stage: Shear, subtracting: bool) -> None:
    public = public_mask_blocks(rotation_order, stage.index, stage.family, stage.kind)
    for irrep in irreps(rotation_order):
        delta = [
            hadamard_coefficient(rotation_order, source, public, irrep, position)
            for position in range(irrep.dimension * irrep.dimension)
        ]
        update_block(target, irrep, delta, stage.scalar, subtracting)


def apply_shear(rotation_order: int, a: Blocks, b: Blocks, stage: Shear, subtracting: bool = False) -> None:
    target, source = (a, b) if stage.target == "A" else (b, a)
    if stage.operation == "INTERSECT":
        hadamard_update(rotation_order, target, source, stage, subtracting)
    elif stage.operation in ("RIGHT_COMPOSE", "LEFT_COMPOSE"):
        convolution_update(rotation_order, target, source, stage, subtracting)
    else:
        fail("unknown dihedral relation operation")


@dataclass
class Carrier:
    rotation_order: int
    seed_family: str
    a: Blocks
    b: Blocks
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, rotation_order: int, family: str) -> "Carrier":
        if rotation_order not in ROTATION_ORDERS or family not in FAMILIES:
            fail("invalid carrier descriptor")
        return cls(rotation_order, family, seed_blocks(rotation_order, family, "A"), seed_blocks(rotation_order, family, "B"))

    def canonical_state(self) -> tuple[Any, ...]:
        return (
            self.rotation_order,
            self.seed_family,
            blocks_state(self.a),
            blocks_state(self.b),
            self.stage,
        )

    def backing_ids(self) -> tuple[int, int]:
        return id(self.a), id(self.b)


def support_record(carrier: Carrier, depth: int) -> dict[str, int]:
    a_capacity = active_capacity(carrier.rotation_order, carrier.a)
    b_capacity = active_capacity(carrier.rotation_order, carrier.b)
    return {
        "depth": depth,
        "a_active_block_capacity": a_capacity,
        "b_active_block_capacity": b_capacity,
        "total_active_block_capacity": a_capacity + b_capacity,
        "a_nonzero_coordinates": nonzero_coordinates(carrier.a),
        "b_nonzero_coordinates": nonzero_coordinates(carrier.b),
    }


def forward(carrier: Carrier, depth: int, family: str, reverse_module_order: bool = False) -> list[dict[str, int]]:
    if carrier.stage != "IDLE":
        fail("carrier is not idle")
    history: list[dict[str, int]] = []
    for index in range(depth):
        stages = descriptors(index, family)
        if reverse_module_order:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply_shear(carrier.rotation_order, carrier.a, carrier.b, stage)
        if index + 1 in CHECKPOINTS:
            history.append(support_record(carrier, index + 1))
    carrier.stage = "FORWARD_COMPLETE"
    return history


def reverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier has no forward state")
    sequence = [(index, stage) for index in reversed(range(depth)) for stage in reversed(descriptors(index, family))]
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    wrong_applied = False
    for _index, stage in sequence:
        if mutation == "WRONG" and not wrong_applied:
            stage = Shear(stage.target, stage.operation, stage.index, stage.family, stage.kind, (stage.scalar + 1) % FIELD)
            wrong_applied = True
        apply_shear(carrier.rotation_order, carrier.a, carrier.b, stage, subtracting=True)
    carrier.stage = "IDLE"


def boundary_value(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary is not available")
    code = 1 if family == "PRIMARY" else 2
    element = ((7 * code + carrier.rotation_order // 3) % carrier.rotation_order, code % 2)
    return evaluate_blocks(carrier.rotation_order, carrier.b, element)


def digest_group_state(rotation_order: int, a: Blocks, b: Blocks) -> str:
    digest = hashlib.sha256()
    for blocks in (a, b):
        for element in elements(rotation_order):
            digest.update(evaluate_blocks(rotation_order, blocks, element).to_bytes(2, "big"))
    return digest.hexdigest()


def transaction(carrier: Carrier, depth: int, family: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    history = forward(carrier, depth, family)
    boundary = boundary_value(carrier, family)
    commitment = digest_group_state(carrier.rotation_order, carrier.a, carrier.b)
    maximum_capacity = max((record["total_active_block_capacity"] for record in history), default=0)
    final_support = support_record(carrier, depth)
    reverse(carrier, depth, family)
    if carrier.canonical_state() != before or carrier.backing_ids() != backing or carrier.restoration_generation != generation:
        fail("exact dihedral carrier restoration failed")
    carrier.restoration_generation += 1
    return {
        "rotation_order": carrier.rotation_order,
        "group_order": 2 * carrier.rotation_order,
        "family": family,
        "depth": depth,
        "boundary": boundary,
        "forward_commitment": commitment,
        "support_history": history,
        "final_support": final_support,
        "maximum_total_active_block_capacity": maximum_capacity,
        "exact_canonical_state_restored": carrier.canonical_state() == before,
        "same_backing_restored": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_relation_values_serialized": False,
    }


def public_group_vector(rotation_order: int, stage: Shear) -> list[int]:
    if stage.operation == "INTERSECT":
        blocks = public_mask_blocks(rotation_order, stage.index, stage.family, stage.kind)
    else:
        blocks = {
            irrep.label: public_convolution_block(irrep, stage.index, stage.family, stage.kind)
            for irrep in irreps(rotation_order)
        }
    return [evaluate_blocks(rotation_order, blocks, element) for element in elements(rotation_order)]


def group_seed(rotation_order: int, family: str, register: str) -> list[int]:
    blocks = seed_blocks(rotation_order, family, register)
    return [evaluate_blocks(rotation_order, blocks, element) for element in elements(rotation_order)]


def group_compose(rotation_order: int, left: list[int], right: list[int]) -> list[int]:
    group = elements(rotation_order)
    index = {element: position for position, element in enumerate(group)}
    result: list[int] = []
    for target in group:
        total = 0
        for source_position, source_element in enumerate(group):
            residual = multiply(rotation_order, inverse(rotation_order, source_element), target)
            total += left[source_position] * right[index[residual]]
        result.append(total % FIELD)
    return result


def group_apply_shear(rotation_order: int, a: list[int], b: list[int], stage: Shear, subtracting: bool = False) -> tuple[list[int], list[int]]:
    target, source = (a, b) if stage.target == "A" else (b, a)
    public = public_group_vector(rotation_order, stage)
    if stage.operation == "INTERSECT":
        delta = [left * right % FIELD for left, right in zip(source, public, strict=True)]
    elif stage.operation == "RIGHT_COMPOSE":
        delta = group_compose(rotation_order, source, public)
    elif stage.operation == "LEFT_COMPOSE":
        delta = group_compose(rotation_order, public, source)
    else:
        fail("unknown group recurrence operation")
    sign = -1 if subtracting else 1
    updated = [(value + sign * stage.scalar * change) % FIELD for value, change in zip(target, delta, strict=True)]
    return (updated, b) if stage.target == "A" else (a, updated)


def group_reference(rotation_order: int, depth: int, family: str) -> dict[str, Any]:
    a = group_seed(rotation_order, family, "A")
    b = group_seed(rotation_order, family, "B")
    sealed = (tuple(a), tuple(b))
    for index in range(depth):
        for stage in descriptors(index, family):
            a, b = group_apply_shear(rotation_order, a, b, stage)
    code = 1 if family == "PRIMARY" else 2
    boundary_element = ((7 * code + rotation_order // 3) % rotation_order, code % 2)
    boundary_index = elements(rotation_order).index(boundary_element)
    boundary = b[boundary_index]
    digest = hashlib.sha256()
    for vector in (a, b):
        for value in vector:
            digest.update(value.to_bytes(2, "big"))
    commitment = digest.hexdigest()
    for index in reversed(range(depth)):
        for stage in reversed(descriptors(index, family)):
            a, b = group_apply_shear(rotation_order, a, b, stage, subtracting=True)
    return {
        "boundary": boundary,
        "forward_commitment": commitment,
        "exact_group_coordinate_restoration": (tuple(a), tuple(b)) == sealed,
        "carrier_field_cells": 4 * rotation_order,
        "current_public_operand_field_cells": 2 * rotation_order,
    }


def one_case(rotation_order: int, depth: int, family: str) -> dict[str, Any]:
    receipt = transaction(Carrier.seal(rotation_order, family), depth, family)
    reference = group_reference(rotation_order, depth, family)
    receipt["matches_streamed_group_coordinate_boundary"] = receipt["boundary"] == reference["boundary"]
    receipt["matches_streamed_group_coordinate_commitment"] = receipt["forward_commitment"] == reference["forward_commitment"]
    receipt["group_coordinate_reference_restores_exactly"] = reference["exact_group_coordinate_restoration"]
    return receipt


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    rotation_order, depth, family = 6, 4, "PRIMARY"
    original = Carrier.seal(rotation_order, family)
    before = original.canonical_state()
    missing = Carrier.seal(rotation_order, family)
    forward(missing, depth, family)
    reverse(missing, depth, family, mutation="MISSING")
    wrong = Carrier.seal(rotation_order, family)
    forward(wrong, depth, family)
    reverse(wrong, depth, family, mutation="WRONG")
    reordered = Carrier.seal(rotation_order, family)
    forward(reordered, depth, family)
    reverse(reordered, depth, family, mutation="REORDER")
    normal = Carrier.seal(rotation_order, family)
    forward(normal, depth, family)
    altered = Carrier.seal(rotation_order, family)
    forward(altered, depth, family, reverse_module_order=True)
    first_two = next(irrep for irrep in irreps(rotation_order) if irrep.dimension == 2)
    source = seed_blocks(rotation_order, family, "A")[first_two.label]
    left = public_convolution_block(first_two, 0, family, 3)
    right = public_convolution_block(first_two, 0, family, 1)
    left_right = matrix_multiply(left, matrix_multiply(source, right, 2), 2)
    right_left = matrix_multiply(right, matrix_multiply(source, left, 2), 2)
    idle = Carrier.seal(rotation_order, family)
    return {
        "missing_inverse_fails_restoration": missing.canonical_state() != before,
        "wrong_inverse_fails_restoration": wrong.canonical_state() != before,
        "reordered_inverse_fails_restoration": reordered.canonical_state() != before,
        "same_modules_reordered_change_boundary": boundary_value(normal, family) != boundary_value(altered, family),
        "left_and_right_two_dimensional_irrep_products_differ": left_right != right_left,
        "premature_boundary_projection_rejected": raises(lambda: boundary_value(idle, family)),
        "invalid_rotation_order_rejected": raises(lambda: Carrier.seal(5, family)),
        "all_declared_roots_have_exact_order": all(
            pow(root_for(order), order, FIELD) == 1
            and all(pow(root_for(order), exponent, FIELD) != 1 for exponent in range(1, order))
            for order in ROTATION_ORDERS
        ),
    }


def algebra_checks() -> dict[str, bool]:
    representation_checks = []
    roundtrip_checks = []
    convolution_checks = []
    for rotation_order in ROTATION_ORDERS:
        group = elements(rotation_order)
        declared_irreps = irreps(rotation_order)
        representation_checks.append(
            all(
                representation(rotation_order, irrep, multiply(rotation_order, left, right))
                == matrix_multiply(
                    representation(rotation_order, irrep, left),
                    representation(rotation_order, irrep, right),
                    irrep.dimension,
                )
                for irrep in declared_irreps
                for left in group
                for right in group
            )
        )
        selected_positions = sorted({0, 1, len(group) // 2, len(group) - 1})
        for position in selected_positions:
            basis = [0] * len(group)
            basis[position] = 1
            transformed = transform_vector(rotation_order, basis)
            reconstructed = [evaluate_blocks(rotation_order, transformed, element) for element in group]
            roundtrip_checks.append(reconstructed == basis)
        selected = selected_positions[:3]
        for left_position in selected:
            for right_position in selected:
                left = [0] * len(group)
                right = [0] * len(group)
                left[left_position] = 1
                right[right_position] = 1
                compact_left = transform_vector(rotation_order, left)
                compact_right = transform_vector(rotation_order, right)
                compact_product: Blocks = {}
                for irrep in declared_irreps:
                    if irrep.label in compact_left and irrep.label in compact_right:
                        compact_product[irrep.label] = matrix_multiply(
                            compact_left[irrep.label], compact_right[irrep.label], irrep.dimension
                        )
                reconstructed = [evaluate_blocks(rotation_order, compact_product, element) for element in group]
                convolution_checks.append(reconstructed == group_compose(rotation_order, left, right))
    return {
        "all_declared_representations_are_homomorphisms": all(representation_checks),
        "selected_group_basis_fourier_roundtrips": all(roundtrip_checks),
        "selected_group_basis_convolutions_match_irrep_products": all(convolution_checks),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(17, "PRIMARY")
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 8, "ALTERNATE")
    fresh = Carrier.seal(17, "PRIMARY")
    reference = transaction(fresh, 8, "ALTERNATE")
    return {
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_canonical_state_restored_after_reuse": carrier.canonical_state() == before,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == reference["boundary"],
        "unrelated_second_commitment_matches_fresh": second["forward_commitment"] == reference["forward_commitment"],
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "snapshot_used": False,
    }


def build_result() -> dict[str, Any]:
    cases = [one_case(rotation_order, MAX_DEPTH, "PRIMARY") for rotation_order in ROTATION_ORDERS]
    cases.append(one_case(17, 8, "ALTERNATE"))
    checks = controls()
    algebra = algebra_checks()
    reuse = reuse_check()
    if not all(
        case["matches_streamed_group_coordinate_boundary"]
        and case["matches_streamed_group_coordinate_commitment"]
        and case["group_coordinate_reference_restores_exactly"]
        and case["exact_canonical_state_restored"]
        and case["same_backing_restored"]
        for case in cases
    ):
        fail("dihedral accepted/reference mismatch")
    if not all(checks.values()) or not all(algebra.values()):
        fail("dihedral control or algebra check failed")
    if not all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "exact_canonical_state_restored_after_reuse",
            "unrelated_second_boundary_matches_fresh",
            "unrelated_second_commitment_matches_fresh",
        )
    ):
        fail("dihedral reuse check failed")
    full_capacity_cases = [
        case["final_support"]["total_active_block_capacity"] == 2 * case["group_order"] for case in cases[:-1]
    ]
    return {
        "schema": "CAT_CAS_F103_GROWING_DIHEDRAL_IRREP_SPARSE_RELATION_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": FIELD,
            "multiplicative_generator": GENERATOR,
            "rotation_orders": list(ROTATION_ORDERS),
            "group_orders": [2 * order for order in ROTATION_ORDERS],
            "maximum_depth": MAX_DEPTH,
            "checkpoints": list(CHECKPOINTS),
            "accepted_state": "ADAPTIVE_NONZERO_DIHEDRAL_IRREP_BLOCKS",
            "native_composition": "BLOCKWISE_NONCOMMUTATIVE_IRREP_MATRIX_MULTIPLICATION",
            "native_intersection": "ONE_OUTPUT_COEFFICIENT_AT_A_TIME_STREAMED_GROUP_EVALUATION_WITHOUT_GROUP_VECTOR_RETENTION",
            "final_boundary_only": True,
            "hidden_relation_values_serialized": False,
            "full_group_vector_materialized_on_accepted_path": False,
            "dense_relation_table_materialized_on_accepted_path": False,
            "answer_bearing_lookup_table_materialized": False,
            "public_topology_reads_final_answer": False,
        },
        "cases": cases,
        "all_primary_cases_reach_full_two_port_irrep_block_capacity_by_depth16": all(full_capacity_cases),
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "accepted_initial_active_block_capacity_per_two_port_carrier": 8,
            "accepted_full_two_port_irrep_block_capacity_by_group_order": {
                str(2 * order): 4 * order for order in ROTATION_ORDERS
            },
            "matched_group_coordinate_two_port_carrier_field_cells_by_group_order": {
                str(2 * order): 4 * order for order in ROTATION_ORDERS
            },
            "accepted_intersection_retains_full_group_vector": False,
            "accepted_intersection_output_coefficient_group_scans_per_operation_by_group_order": {
                str(2 * order): (2 * order) ** 2 for order in ROTATION_ORDERS
            },
            "matched_group_coordinate_current_public_operand_field_cells_by_group_order": {
                str(2 * order): 2 * order for order in ROTATION_ORDERS
            },
            "retained_inverse_history_field_cells": 0,
            "retained_compiled_irrep_or_fusion_table_field_cells": 0,
            "snapshot_cells": 0,
            "group_topology_records_and_python_runtime_costs_excluded": True,
            "physical_process_peak_unmeasured": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL_ADAPTIVE_DIHEDRAL_IRREP_BLOCK_RECURRENCE",
            "strongest_law_identical_to_accepted": True,
            "executed_group_coordinate": "FULL_TWO_N_GROUP_VECTOR_RECURRENCE_WITH_CURRENT_PUBLIC_OPERAND_VECTOR",
            "executed_group_coordinate_matches_all_cases": True,
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "F103_SPLIT_DIHEDRAL_TRANSLATION_INVARIANT_RELATIONS_ROTATION_ORDERS3_6_17_34_51_ONE_PRIMARY_DEPTH16_CASE_EACH_ONE_ALTERNATE_D34_DEPTH8_CASE_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "GENERAL_FINITE_GROUP_COMPILER",
            "GENERAL_NON_TRANSLATION_INVARIANT_RELATIONS",
            "FIXED_RANK_UNBOUNDED_DEPTH_CLOSURE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": (
            "ADAPTIVE_NONABELIAN_IRREP_SPARSITY_DOES_NOT_SUPPLY_FIXED_RANK_CLOSURE_"
            "FOR_THE_TESTED_GROWING_DIHEDRAL_FAMILY;_TEST_THE_SMALLEST_EXACT_"
            "FUSION_CLOSED_NONCOMMUTATIVE_RELATION_SUBALGEBRA_OR_AN_EXACT_"
            "STRUCTURE_PRESERVING_REDUCTION_WITHOUT_MOVING_GROWTH_INTO_"
            "COEFFICIENT_WIDTH_OR_RETAINED_HISTORY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
