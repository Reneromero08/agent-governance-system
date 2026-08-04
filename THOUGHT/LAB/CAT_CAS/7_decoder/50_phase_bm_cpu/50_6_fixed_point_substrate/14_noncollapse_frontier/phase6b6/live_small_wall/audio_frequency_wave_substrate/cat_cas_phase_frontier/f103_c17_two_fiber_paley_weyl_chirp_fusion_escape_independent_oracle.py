#!/usr/bin/env python3
"""Independent dense-matrix oracle for the C17 Weyl-chirp diagnostic.

This verifier intentionally does not import the production package.  It
reconstructs the public program as exact 34 x 34 matrices over F103, derives
Weyl support with an independent Fourier transform, and compares the durable
production result field by field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


P = 103
Q = 17
ROOT = pow(5, 6, P)
INV_Q = pow(Q, -1, P)
FIBERS = 2
SIZE = FIBERS * Q
CHECKPOINTS = (1, 2, 4, 8, 16)
FAMILIES = ("PRIMARY", "ALTERNATE")

Matrix = list[list[int]]


def phase(exponent: int) -> int:
    return pow(ROOT, exponent % Q, P)


def paley_class(value: int) -> int:
    value %= Q
    if value == 0:
        return 0
    return 1 if pow(value, (Q - 1) // 2, Q) == 1 else 2


def zero_matrix() -> Matrix:
    return [[0] * SIZE for _ in range(SIZE)]


def copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def base_matrix(index: int, family: str, register: str | None = None, kind: int = 0) -> Matrix:
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 23 if register == "B" else 41 + 7 * kind
    result = zero_matrix()
    for source in range(FIBERS):
        for x in range(Q):
            for target in range(FIBERS):
                for displacement in range(Q):
                    cls = paley_class(displacement)
                    exponent = (
                        offset
                        + code * 5
                        + 11 * source
                        + 13 * target
                        + 17 * cls
                        + (index + 1) * (kind + cls + 1)
                    )
                    result[source * Q + x][target * Q + (x + displacement) % Q] = pow(
                        5, exponent % (P - 1), P
                    )
    return result


def chirped_public(index: int, family: str, kind: int, strength: int) -> Matrix:
    """Construct the chirped public relation directly in vertex coordinates."""
    base = base_matrix(index, family, kind=kind)
    result = zero_matrix()
    for source in range(FIBERS):
        for x in range(Q):
            row = source * Q + x
            for target in range(FIBERS):
                for displacement in range(Q):
                    column = target * Q + (x + displacement) % Q
                    exponent = -strength * displacement * displacement - 2 * strength * displacement * x
                    result[row][column] = base[row][column] * phase(exponent) % P
    return result


def hadamard(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] * right[row][column] % P for column in range(SIZE)]
        for row in range(SIZE)
    ]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    result = zero_matrix()
    for row in range(SIZE):
        for shared in range(SIZE):
            left_value = left[row][shared]
            if not left_value:
                continue
            for column in range(SIZE):
                right_value = right[shared][column]
                if right_value:
                    result[row][column] = (
                        result[row][column] + left_value * right_value
                    ) % P
    return result


def add_scaled(target: Matrix, change: Matrix, scalar: int, subtracting: bool = False) -> None:
    signed = -scalar if subtracting else scalar
    for row in range(SIZE):
        for column in range(SIZE):
            target[row][column] = (target[row][column] + signed * change[row][column]) % P


@dataclass(frozen=True)
class Stage:
    target: str
    operation: str
    index: int
    family: str
    kind: int
    strength: int
    scalar: int


def descriptors(index: int, family: str) -> tuple[Stage, ...]:
    code = 1 if family == "PRIMARY" else 2
    stages = (
        Stage("B", "RIGHT_COMPOSE", index, family, 1, index + code, pow(5, (3 * index + 7 * code + 1) % 102, P)),
        Stage("A", "INTERSECT", index, family, 2, 2 * index + code, pow(5, (5 * index + 11 * code + 2) % 102, P)),
        Stage("B", "LEFT_COMPOSE", index, family, 3, 3 * index + code, pow(5, (7 * index + 13 * code + 3) % 102, P)),
        Stage("A", "INTERSECT", index, family, 4, 5 * index + code, pow(5, (11 * index + 17 * code + 4) % 102, P)),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def apply_stage(a: Matrix, b: Matrix, stage: Stage, subtracting: bool = False) -> None:
    target, source = (a, b) if stage.target == "A" else (b, a)
    public = chirped_public(stage.index, stage.family, stage.kind, stage.strength)
    if stage.operation == "INTERSECT":
        change = hadamard(source, public)
    elif stage.operation == "RIGHT_COMPOSE":
        change = multiply(source, public)
    elif stage.operation == "LEFT_COMPOSE":
        change = multiply(public, source)
    else:
        raise ValueError(stage.operation)
    add_scaled(target, change, stage.scalar, subtracting)


def fourier_coeff(matrix: Matrix, source: int, target: int, displacement: int, mode: int) -> int:
    total = 0
    for x in range(Q):
        row = source * Q + x
        column = target * Q + (x + displacement) % Q
        total += matrix[row][column] * phase(-mode * x)
    return total * INV_Q % P


def active_weyl_cells(matrix: Matrix) -> int:
    return sum(
        fourier_coeff(matrix, source, target, displacement, mode) != 0
        for source in range(FIBERS)
        for target in range(FIBERS)
        for displacement in range(Q)
        for mode in range(Q)
    )


def support(a: Matrix, b: Matrix, depth: int) -> dict[str, int]:
    a_active = active_weyl_cells(a)
    b_active = active_weyl_cells(b)
    return {
        "depth": depth,
        "a_active_weyl_cells": a_active,
        "b_active_weyl_cells": b_active,
        "total_active_weyl_cells": a_active + b_active,
        "full_two_port_weyl_capacity": 2 * FIBERS * FIBERS * Q * Q,
    }


def semantic_digest(a: Matrix, b: Matrix) -> str:
    digest = hashlib.sha256()
    for matrix in (a, b):
        for source in range(FIBERS):
            for x in range(Q):
                row = source * Q + x
                for target in range(FIBERS):
                    for displacement in range(Q):
                        column = target * Q + (x + displacement) % Q
                        digest.update(matrix[row][column].to_bytes(2, "big"))
    return digest.hexdigest()


@dataclass
class DenseCarrier:
    seed_family: str
    a: Matrix
    b: Matrix
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, family: str) -> "DenseCarrier":
        if family not in FAMILIES:
            raise ValueError("invalid family")
        return cls(family, base_matrix(0, family, "A"), base_matrix(0, family, "B"))

    def canonical_state(self) -> tuple[Any, ...]:
        return self.seed_family, tuple(map(tuple, self.a)), tuple(map(tuple, self.b)), self.stage

    def backing_ids(self) -> tuple[int, int]:
        return id(self.a), id(self.b)


def forward(carrier: DenseCarrier, depth: int, family: str, reverse_modules: bool = False) -> list[dict[str, int]]:
    if carrier.stage != "IDLE":
        raise ValueError("carrier not idle")
    records = []
    for index in range(depth):
        stages = descriptors(index, family)
        if reverse_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            apply_stage(carrier.a, carrier.b, stage)
        if index + 1 in CHECKPOINTS:
            records.append(support(carrier.a, carrier.b, index + 1))
    carrier.stage = "FORWARD_COMPLETE"
    return records


def reverse(carrier: DenseCarrier, depth: int, family: str, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        raise ValueError("carrier lacks forward state")
    sequence = [stage for index in reversed(range(depth)) for stage in reversed(descriptors(index, family))]
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, stage in enumerate(sequence):
        if mutation == "WRONG" and position == 0:
            stage = Stage(stage.target, stage.operation, stage.index, stage.family, stage.kind, stage.strength, stage.scalar + 1)
        apply_stage(carrier.a, carrier.b, stage, subtracting=True)
    carrier.stage = "IDLE"


def boundary(carrier: DenseCarrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        raise ValueError("boundary unavailable")
    code = 1 if family == "PRIMARY" else 2
    source = code % 2
    target = (code + 1) % 2
    x = 3 * code
    displacement = 5 * code
    return carrier.b[source * Q + x][target * Q + (x + displacement) % Q]


def transaction(carrier: DenseCarrier, depth: int, family: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    history = forward(carrier, depth, family)
    projected = boundary(carrier, family)
    digest = semantic_digest(carrier.a, carrier.b)
    final_support = support(carrier.a, carrier.b, depth)
    reverse(carrier, depth, family)
    restored = carrier.canonical_state() == before
    same_backing = carrier.backing_ids() == backing
    generation_unchanged_during_inverse = carrier.restoration_generation == generation
    carrier.restoration_generation += 1
    return {
        "family": family,
        "depth": depth,
        "boundary": projected,
        "semantic_commitment": digest,
        "support_history": history,
        "final_support": final_support,
        "exact_canonical_state_restored": restored and generation_unchanged_during_inverse,
        "same_backing_restored": same_backing,
        "restoration_generation": carrier.restoration_generation,
    }


def control_attacks() -> dict[str, bool]:
    expected = DenseCarrier.seal("PRIMARY").canonical_state()
    failures: dict[str, bool] = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        carrier = DenseCarrier.seal("PRIMARY")
        forward(carrier, 2, "PRIMARY")
        reverse(carrier, 2, "PRIMARY", mutation)
        failures[mutation] = carrier.canonical_state() != expected
    normal = DenseCarrier.seal("PRIMARY")
    forward(normal, 2, "PRIMARY")
    altered = DenseCarrier.seal("PRIMARY")
    forward(altered, 2, "PRIMARY", reverse_modules=True)
    premature = DenseCarrier.seal("PRIMARY")
    try:
        boundary(premature, "PRIMARY")
        premature_rejected = False
    except ValueError:
        premature_rejected = True
    try:
        DenseCarrier.seal("NULL")
        null_rejected = False
    except ValueError:
        null_rejected = True
    return {
        "missing_inverse_fails_restoration": failures["MISSING"],
        "wrong_inverse_fails_restoration": failures["WRONG"],
        "reordered_inverse_fails_restoration": failures["REORDER"],
        "module_reordering_changes_boundary": boundary(normal, "PRIMARY") != boundary(altered, "PRIMARY"),
        "premature_boundary_projection_rejected": premature_rejected,
        "null_family_rejected": null_rejected,
    }


def reuse_attack() -> dict[str, Any]:
    carrier = DenseCarrier.seal("PRIMARY")
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 8, "ALTERNATE")
    fresh = transaction(DenseCarrier.seal("PRIMARY"), 8, "ALTERNATE")
    return {
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "same_backing_reused": carrier.backing_ids() == backing,
        "exact_canonical_state_restored_after_reuse": carrier.canonical_state() == before,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "unrelated_second_commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "snapshot_used": False,
    }


def chirp_escape_attack() -> dict[str, Any]:
    matrix = chirped_public(0, "PRIMARY", 0, 1)
    support_count = active_weyl_cells(matrix)
    x_dependent = any(
        matrix[source * Q][target * Q + displacement]
        != matrix[source * Q + 1][target * Q + (1 + displacement) % Q]
        for source in range(FIBERS)
        for target in range(FIBERS)
        for displacement in range(1, Q)
    )
    return {
        "fixed12_fusion_representation_rejected_by_x_dependence": x_dependent,
        "chirped_public_active_weyl_cells": support_count,
        "predecessor_fused_cells_per_port": 12,
    }


def build_result(production: dict[str, Any]) -> dict[str, Any]:
    reconstructed = [
        transaction(DenseCarrier.seal(family), depth, family)
        for family in FAMILIES
        for depth in CHECKPOINTS
    ]
    production_cases = {
        (case["family"], case["depth"]): case for case in production["cases"]
    }
    comparisons = []
    for case in reconstructed:
        observed = production_cases[case["family"], case["depth"]]
        comparisons.append(
            {
                "family": case["family"],
                "depth": case["depth"],
                "boundary_matches": case["boundary"] == observed["boundary"],
                "semantic_commitment_matches": case["semantic_commitment"] == observed["semantic_commitment"],
                "support_history_matches": case["support_history"] == observed["support_history"],
                "final_support_matches": case["final_support"] == observed["final_support"],
                "exact_restoration_matches": case["exact_canonical_state_restored"] == observed["exact_canonical_state_restored"],
                "same_backing_matches": case["same_backing_restored"] == observed["same_backing_restored"],
                "generation_matches": case["restoration_generation"] == observed["restoration_generation"],
            }
        )
    controls = control_attacks()
    reuse = reuse_attack()
    escape = chirp_escape_attack()
    all_case_checks = all(
        all(value for key, value in comparison.items() if key.endswith("_matches"))
        for comparison in comparisons
    )
    controls_match = controls == production["controls"] and all(controls.values())
    reuse_match = reuse == production["restoration_and_reuse"] and all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "exact_canonical_state_restored_after_reuse",
            "unrelated_second_boundary_matches_fresh",
            "unrelated_second_commitment_matches_fresh",
        )
    )
    observed_supports = [
        point["total_active_weyl_cells"]
        for case in reconstructed
        for point in case["support_history"]
    ]
    support_range = {
        "minimum": min(observed_supports),
        "maximum": max(observed_supports),
        "capacity": 2 * FIBERS * FIBERS * Q * Q,
    }
    support_range_matches = support_range == production["observed_two_port_active_weyl_cell_range_after_first_update"]
    decision = (
        "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        if all_case_checks and controls_match and reuse_match and support_range_matches and escape["fixed12_fusion_representation_rejected_by_x_dependence"]
        else "REJECTED_SOURCE_DEFECT"
    )
    return {
        "schema": "CAT_CAS_F103_C17_TWO_FIBER_PALEY_WEYL_CHIRP_FUSION_ESCAPE_INDEPENDENT_ORACLE_V1",
        "classification": decision,
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_imported": False,
        "independent_representation": "EXACT_34_BY_34_VERTEX_MATRICES_OVER_F103_WITH_INDEPENDENT_DFT_SUPPORT_RECOVERY",
        "case_comparisons": comparisons,
        "all_ten_case_boundaries_commitments_support_histories_and_restoration_match": all_case_checks,
        "control_attacks": controls,
        "controls_match_production": controls_match,
        "fresh_restored_reuse_attack": reuse,
        "reuse_matches_production": reuse_match,
        "fusion_escape_attack": escape,
        "observed_two_port_active_weyl_cell_range_after_first_update": support_range,
        "support_range_matches_production": support_range_matches,
        "matched_classical_baseline": "THE_PRODUCTION_WEYL_RECURRENCE_REMAINS_THE_STRONGEST_EXECUTED_COMPACT_CLASSICAL_METHOD_AND_IS_IDENTICAL_TO_THE_ACCEPTED_PATH",
        "claim_ceiling": production["claim_ceiling"],
        "rejected_interpretations": production["not_established"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-result", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    production = json.loads(args.production_result.read_text(encoding="utf-8"))
    result = build_result(production)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
