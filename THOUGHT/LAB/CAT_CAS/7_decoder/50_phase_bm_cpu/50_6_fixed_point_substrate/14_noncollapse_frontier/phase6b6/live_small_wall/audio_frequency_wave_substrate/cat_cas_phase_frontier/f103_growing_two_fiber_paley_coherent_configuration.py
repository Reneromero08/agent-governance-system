#!/usr/bin/env python3
"""Exact fixed-12 two-fiber Paley coherent-configuration experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


FIELD = 103
GENERATOR = 5
PALEY_ORDERS = (5, 13, 17, 29, 37, 53, 73, 97)
PRIMARY_DEPTH = 256
MAXIMUM_DEPTH = 1024
CHECKPOINTS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLASS_COUNT = 3
FIBER_COUNT = 2
RELATION_CELLS = FIBER_COUNT * FIBER_COUNT * CLASS_COUNT
CLAIM = (
    "BOUNDED_EXACT_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_"
    "FIXED12_NONCOMMUTATIVE_F103_PHASE_RELATION_ALGEBRA_CLOSES_NATIVE_"
    "COMPOSITION_AND_HADAMARD_INTERSECTION_ON_VERTEX_COUNTS10_26_34_58_"
    "74_106_146_194_THROUGH_DEPTH1024_WITH_FINAL_ONLY_BOUNDARY_EXACT_"
    "RESTORATION_AND_REUSE_BUT_THE_IDENTICAL12_COORDINATE_CLASSICAL_"
    "RECURRENCE_REMAINS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def phase(exponent: int) -> int:
    return pow(GENERATOR, exponent % (FIELD - 1), FIELD)


def relation_index(source_fiber: int, target_fiber: int, difference_class: int) -> int:
    return (source_fiber * FIBER_COUNT + target_fiber) * CLASS_COUNT + difference_class


def validate_order(order: int) -> None:
    if order not in PALEY_ORDERS or order % 4 != 1 or order >= FIELD:
        fail("unsupported Paley order")


def paley_class(order: int, value: int) -> int:
    value %= order
    if value == 0:
        return 0
    return 1 if pow(value, (order - 1) // 2, order) == 1 else 2


def class_product(order: int, left: int, right: int) -> tuple[int, int, int]:
    """Structure constants for D_left * D_right in the Paley class basis."""
    validate_order(order)
    if left == 0:
        return tuple(1 if output == right else 0 for output in range(CLASS_COUNT))
    if right == 0:
        return tuple(1 if output == left else 0 for output in range(CLASS_COUNT))
    valency = (order - 1) // 2
    lam = (order - 5) // 4
    mu = (order - 1) // 4
    if left == right == 1:
        return valency, lam, mu
    if left == right == 2:
        return valency, mu, lam
    return 0, mu, mu


def class_convolution(order: int, left: list[int], right: list[int]) -> list[int]:
    output = [0] * CLASS_COUNT
    for left_class, left_value in enumerate(left):
        for right_class, right_value in enumerate(right):
            constants = class_product(order, left_class, right_class)
            for output_class, multiplicity in enumerate(constants):
                output[output_class] = (
                    output[output_class] + left_value * right_value * multiplicity
                ) % FIELD
    return output


def compose(order: int, left: list[int], right: list[int]) -> list[int]:
    output = [0] * RELATION_CELLS
    for source in range(FIBER_COUNT):
        for target in range(FIBER_COUNT):
            accumulated = [0] * CLASS_COUNT
            for shared in range(FIBER_COUNT):
                left_block = [left[relation_index(source, shared, c)] for c in range(CLASS_COUNT)]
                right_block = [right[relation_index(shared, target, c)] for c in range(CLASS_COUNT)]
                product = class_convolution(order, left_block, right_block)
                accumulated = [(x + y) % FIELD for x, y in zip(accumulated, product, strict=True)]
            for difference_class, value in enumerate(accumulated):
                output[relation_index(source, target, difference_class)] = value
    return output


def intersect(left: list[int], right: list[int]) -> list[int]:
    return [x * y % FIELD for x, y in zip(left, right, strict=True)]


def seed(order: int, family: str, register: str) -> list[int]:
    validate_order(order)
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 19
    return [phase(order + code * 7 + offset + position * (code + 2)) for position in range(RELATION_CELLS)]


def public_relation(order: int, index: int, family: str, kind: int) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        phase(order + (index + 1) * (position + kind + 1) + code * (kind + 3) + position * position)
        for position in range(RELATION_CELLS)
    ]


def scalar(index: int, family: str, kind: int) -> int:
    code = 1 if family == "PRIMARY" else 2
    return phase((kind + 1) * (index + 1) + code * (kind + 5))


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
        Shear("B", "RIGHT_COMPOSE", index, family, 1, scalar(index, family, 1)),
        Shear("A", "INTERSECT", index, family, 2, scalar(index, family, 2)),
        Shear("B", "LEFT_COMPOSE", index, family, 3, scalar(index, family, 3)),
        Shear("A", "INTERSECT", index, family, 4, scalar(index, family, 4)),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def delta(order: int, source: list[int], stage: Shear) -> list[int]:
    public = public_relation(order, stage.index, stage.family, stage.kind)
    if stage.operation == "INTERSECT":
        return intersect(source, public)
    if stage.operation == "RIGHT_COMPOSE":
        return compose(order, source, public)
    if stage.operation == "LEFT_COMPOSE":
        return compose(order, public, source)
    fail("unknown relation operation")


def apply_shear(order: int, a: list[int], b: list[int], stage: Shear, subtracting: bool = False) -> None:
    target, source = (a, b) if stage.target == "A" else (b, a)
    change = delta(order, source, stage)
    sign = -1 if subtracting else 1
    for position, value in enumerate(change):
        target[position] = (target[position] + sign * stage.scalar * value) % FIELD


@dataclass
class Carrier:
    order: int
    seed_family: str
    a: list[int]
    b: list[int]
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, order: int, family: str) -> "Carrier":
        if family not in FAMILIES:
            fail("unknown family")
        return cls(order, family, seed(order, family, "A"), seed(order, family, "B"))

    def canonical_state(self) -> tuple[Any, ...]:
        return self.order, self.seed_family, tuple(self.a), tuple(self.b), self.stage

    def backing_ids(self) -> tuple[int, int]:
        return id(self.a), id(self.b)


def forward(carrier: Carrier, depth: int, family: str, reverse_modules: bool = False) -> list[dict[str, int]]:
    if carrier.stage != "IDLE" or depth < 1 or depth > MAXIMUM_DEPTH:
        fail("invalid forward request")
    checkpoints: list[dict[str, int]] = []
    for index in range(depth):
        stages = descriptors(index, family)
        if reverse_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply_shear(carrier.order, carrier.a, carrier.b, stage)
        if index + 1 in CHECKPOINTS:
            checkpoints.append(
                {
                    "depth": index + 1,
                    "resident_relation_cells": len(carrier.a) + len(carrier.b),
                    "represented_vertices": 2 * carrier.order,
                    "represented_dense_relation_entries_per_port": (2 * carrier.order) ** 2,
                }
            )
    carrier.stage = "FORWARD_COMPLETE"
    return checkpoints


def reverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier has no forward state")
    stages = [stage for index in reversed(range(depth)) for stage in reversed(descriptors(index, family))]
    if mutation == "MISSING":
        stages = stages[1:]
    elif mutation == "REORDER":
        stages = list(reversed(stages))
    for position, stage in enumerate(stages):
        if mutation == "WRONG" and position == 0:
            stage = Shear(stage.target, stage.operation, stage.index, stage.family, stage.kind, stage.scalar + 1)
        apply_shear(carrier.order, carrier.a, carrier.b, stage, subtracting=True)
    carrier.stage = "IDLE"


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    code = 1 if family == "PRIMARY" else 2
    return carrier.b[relation_index(code % 2, (code + 1) % 2, code % CLASS_COUNT)]


def commitment(carrier: Carrier) -> str:
    digest = hashlib.sha256()
    for relation in (carrier.a, carrier.b):
        for value in relation:
            digest.update(value.to_bytes(2, "big"))
    return digest.hexdigest()


def transaction(carrier: Carrier, depth: int, family: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    checkpoints = forward(carrier, depth, family)
    projected = boundary(carrier, family)
    state_commitment = commitment(carrier)
    reverse(carrier, depth, family)
    if carrier.canonical_state() != before or carrier.backing_ids() != backing:
        fail("exact coherent-configuration restoration failed")
    if carrier.restoration_generation != generation:
        fail("generation changed before restoration verification")
    carrier.restoration_generation += 1
    return {
        "paley_order": carrier.order,
        "represented_vertices": 2 * carrier.order,
        "family": family,
        "depth": depth,
        "boundary": projected,
        "forward_commitment": state_commitment,
        "checkpoints": checkpoints,
        "resident_relation_cells": RELATION_CELLS * 2,
        "represented_dense_relation_entries_per_port": (2 * carrier.order) ** 2,
        "exact_canonical_state_restored": carrier.canonical_state() == before,
        "same_backing_restored": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_relation_coefficients_serialized": False,
    }


def enumerate_structure_constants(order: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    classes = [[value for value in range(order) if paley_class(order, value) == label] for label in range(3)]
    result: dict[tuple[int, int], tuple[int, int, int]] = {}
    for left in range(3):
        for right in range(3):
            counts: list[int] = []
            for output in range(3):
                representative = classes[output][0]
                count = sum(
                    1 for x in classes[left] if (representative - x) % order in set(classes[right])
                )
                counts.append(count)
            result[left, right] = tuple(counts)
    return result


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    order, depth, family = 13, 4, "PRIMARY"
    before = Carrier.seal(order, family).canonical_state()
    damaged: dict[str, Carrier] = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        carrier = Carrier.seal(order, family)
        forward(carrier, depth, family)
        reverse(carrier, depth, family, mutation)
        damaged[mutation] = carrier
    normal = Carrier.seal(order, family)
    forward(normal, depth, family)
    reordered = Carrier.seal(order, family)
    forward(reordered, depth, family, reverse_modules=True)
    left = [0] * RELATION_CELLS
    right = [0] * RELATION_CELLS
    left[relation_index(0, 1, 0)] = 1
    right[relation_index(1, 0, 0)] = 1
    idle = Carrier.seal(order, family)
    return {
        "missing_inverse_fails_restoration": damaged["MISSING"].canonical_state() != before,
        "wrong_inverse_fails_restoration": damaged["WRONG"].canonical_state() != before,
        "reordered_inverse_fails_restoration": damaged["REORDER"].canonical_state() != before,
        "module_reordering_changes_boundary": boundary(normal, family) != boundary(reordered, family),
        "left_and_right_composition_differ": compose(order, left, right) != compose(order, right, left),
        "premature_boundary_projection_rejected": raises(lambda: boundary(idle, family)),
        "invalid_order_rejected": raises(lambda: Carrier.seal(9, family)),
        "all_formula_structure_constants_match_enumeration": all(
            enumerate_structure_constants(q)[left, right] == class_product(q, left, right)
            for q in PALEY_ORDERS
            for left in range(CLASS_COUNT)
            for right in range(CLASS_COUNT)
        ),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(37, "PRIMARY")
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 64, "ALTERNATE")
    fresh = transaction(Carrier.seal(37, "PRIMARY"), 64, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_canonical_state_restored_after_reuse": carrier.canonical_state() == before,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "unrelated_second_commitment_matches_fresh": second["forward_commitment"] == fresh["forward_commitment"],
        "snapshot_used": False,
    }


def build_result() -> dict[str, Any]:
    cases = [transaction(Carrier.seal(order, "PRIMARY"), PRIMARY_DEPTH, "PRIMARY") for order in PALEY_ORDERS]
    cases.append(transaction(Carrier.seal(97, "PRIMARY"), MAXIMUM_DEPTH, "PRIMARY"))
    cases.append(transaction(Carrier.seal(37, "ALTERNATE"), 64, "ALTERNATE"))
    checks = controls()
    reuse = reuse_check()
    if not all(checks.values()) or not all(
        case["exact_canonical_state_restored"] and case["same_backing_restored"] for case in cases
    ):
        fail("coherent-configuration qualification failed")
    if not all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "exact_canonical_state_restored_after_reuse",
            "unrelated_second_boundary_matches_fresh",
            "unrelated_second_commitment_matches_fresh",
        )
    ):
        fail("coherent-configuration reuse failed")
    return {
        "schema": "CAT_CAS_F103_GROWING_TWO_FIBER_PALEY_COHERENT_CONFIGURATION_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "coefficient_field": FIELD,
            "paley_orders": list(PALEY_ORDERS),
            "represented_vertex_counts": [2 * order for order in PALEY_ORDERS],
            "relation_basis": "TWO_BY_TWO_FIBER_BLOCKS_TENSOR_IDENTITY_QUADRATIC_RESIDUE_NONRESIDUE_DIFFERENCE_CLASSES",
            "resident_coefficients_per_port": RELATION_CELLS,
            "native_composition": "TWO_BY_TWO_MATRIX_PRODUCT_OVER_PALEY_CLASS_CONVOLUTION_ALGEBRA",
            "native_intersection": "COEFFICIENTWISE_HADAMARD_PRODUCT_OF_DISJOINT_BASIS_RELATIONS",
            "composition_noncommutative": True,
            "final_boundary_only": True,
            "ordinary_relation_table_materialized_on_accepted_path": False,
            "assignment_or_witness_expansion_materialized": False,
            "hidden_relation_coefficients_serialized": False,
            "public_topology_reads_final_answer": False,
        },
        "cases": cases,
        "controls": checks,
        "restoration_and_reuse": reuse,
        "fixed_rank_law": {
            "resident_coefficients_per_port": RELATION_CELLS,
            "two_port_carrier_field_cells": RELATION_CELLS * 2,
            "maximum_represented_vertices": 194,
            "maximum_dense_relation_entries_per_port": 194 * 194,
            "fixed12_across_all_declared_orders_and_depths": all(
                case["resident_relation_cells"] == 24 for case in cases
            ),
            "structural_family_restriction": "TWO_FIBER_PALEY_COHERENT_CONFIGURATION_ONLY",
        },
        "resource_accounting": {
            "accepted_carrier_field_cells": 24,
            "accepted_public_operand_field_cells": 12,
            "accepted_operation_delta_field_cells": 12,
            "accepted_restoration_verification_baseline_field_cells": 24,
            "accepted_conservative_named_field_cell_peak": 72,
            "retained_inverse_history_cells": 0,
            "retained_compiled_structure_table_cells": 0,
            "snapshot_cells": 0,
            "matched_identical_classical_conservative_named_field_cell_peak": 72,
            "python_container_allocator_runtime_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL12_COORDINATE_TWO_BY_TWO_PALEY_CLASS_ALGEBRA_RECURRENCE",
            "strongest_law_identical_to_accepted": True,
            "dense_relation_expansion_used_as_matched_baseline": False,
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "F103_TWO_FIBER_PALEY_COHERENT_CONFIGURATIONS_ORDERS5_13_17_29_37_53_73_97_PRIMARY_DEPTH256_ALL_ORDERS_PRIMARY_DEPTH1024_ORDER97_ALTERNATE_DEPTH64_ORDER37_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "GENERAL_COHERENT_CONFIGURATION_COMPILER",
            "GENERAL_NON_TRANSLATION_INVARIANT_RELATIONS",
            "ARBITRARY_FIBER_COUNT_OR_PALEY_ORDER",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": (
            "FIXED12_CLOSURE_IS_A_PUBLIC_STRUCTURE_FUSION_AND_HAS_THE_IDENTICAL12_"
            "COORDINATE_CLASSICAL_RECURRENCE;_TEST_WHETHER_A_PHASE_NATIVE_COUPLING_"
            "CAN_BREAK_THIS_CLASSICAL_BISIMULATION_WITHOUT_EXPANDING_THE_RELATION_BASIS"
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
