#!/usr/bin/env python3
"""Exact C17 Weyl-phase diagnostic beyond the fixed-12 Paley fusion chart."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


P = 103
Q = 17
ROOT = pow(5, 6, P)
FIBERS = 2
MAX_DEPTH = 16
CHECKPOINTS = (1, 2, 4, 8, 16)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F103_C17_TWO_FIBER_PALEY_PHASE_WEYL_CHIRP_COUPLING_"
    "ESCAPES_THE_FIXED12_FUSION_CHART_AND_EXPANDS_THE_INITIAL136_ACTIVE_"
    "TWO_PORT_CELLS_TO_AT_LEAST2278_OF2312_WEYL_CELLS_THROUGH_DEPTH16_"
    "UNDER_NATIVE_TWISTED_COMPOSITION_"
    "AND_PHASE_MODE_INTERSECTION_WITH_FINAL_ONLY_BOUNDARY_EXACT_"
    "RESTORATION_AND_REUSE_BUT_THE_MATCHED_WEYL_CLASSICAL_RECURRENCE_IS_IDENTICAL"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def phase(exponent: int) -> int:
    return pow(ROOT, exponent % Q, P)


def paley_class(value: int) -> int:
    value %= Q
    if value == 0:
        return 0
    return 1 if pow(value, (Q - 1) // 2, Q) == 1 else 2


Key = tuple[int, int, int, int]
Weyl = dict[Key, int]


def canonical(state: Weyl) -> Weyl:
    return {key: value % P for key, value in state.items() if value % P}


def state_tuple(state: Weyl) -> tuple[tuple[Key, int], ...]:
    return tuple(sorted(canonical(state).items()))


def base_relation(index: int, family: str, register: str | None = None, kind: int = 0) -> Weyl:
    code = 1 if family == "PRIMARY" else 2
    offset = 0 if register == "A" else 23 if register == "B" else 41 + 7 * kind
    result: Weyl = {}
    for source in range(FIBERS):
        for target in range(FIBERS):
            for displacement in range(Q):
                cls = paley_class(displacement)
                exponent = offset + code * 5 + 11 * source + 13 * target + 17 * cls + (index + 1) * (kind + cls + 1)
                result[source, target, displacement, 0] = pow(5, exponent % (P - 1), P)
    return result


def chirp(state: Weyl, strength: int) -> Weyl:
    result: Weyl = {}
    for (source, target, displacement, mode), value in state.items():
        shifted = (mode - 2 * strength * displacement) % Q
        key = source, target, displacement, shifted
        result[key] = (result.get(key, 0) + value * phase(-strength * displacement * displacement)) % P
    return canonical(result)


def intersect(left: Weyl, right: Weyl) -> Weyl:
    right_by_block: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for (source, target, displacement, mode), value in right.items():
        right_by_block.setdefault((source, target, displacement), []).append((mode, value))
    result: Weyl = {}
    for (source, target, displacement, left_mode), left_value in left.items():
        for right_mode, right_value in right_by_block.get((source, target, displacement), ()):
            key = source, target, displacement, (left_mode + right_mode) % Q
            result[key] = (result.get(key, 0) + left_value * right_value) % P
    return canonical(result)


def compose(left: Weyl, right: Weyl) -> Weyl:
    right_by_source: dict[int, list[tuple[Key, int]]] = {}
    for key, value in right.items():
        right_by_source.setdefault(key[0], []).append((key, value))
    result: Weyl = {}
    for (source, shared, left_d, left_m), left_value in left.items():
        for (_, target, right_d, right_m), right_value in right_by_source.get(shared, ()):
            key = source, target, (left_d + right_d) % Q, (left_m + right_m) % Q
            twist = phase(right_m * left_d)
            result[key] = (result.get(key, 0) + left_value * right_value * twist) % P
    return canonical(result)


def evaluate(state: Weyl, source: int, target: int, x: int, displacement: int) -> int:
    return sum(
        value * phase(mode * x)
        for (left, right, d, mode), value in state.items()
        if left == source and right == target and d == displacement
    ) % P


def semantic_digest(a: Weyl, b: Weyl) -> str:
    digest = hashlib.sha256()
    for state in (a, b):
        for source in range(FIBERS):
            for x in range(Q):
                for target in range(FIBERS):
                    for displacement in range(Q):
                        digest.update(evaluate(state, source, target, x, displacement).to_bytes(2, "big"))
    return digest.hexdigest()


@dataclass(frozen=True)
class Shear:
    target: str
    operation: str
    index: int
    family: str
    kind: int
    strength: int
    scalar: int


def descriptors(index: int, family: str) -> tuple[Shear, ...]:
    code = 1 if family == "PRIMARY" else 2
    stages = (
        Shear("B", "RIGHT_COMPOSE", index, family, 1, index + code, pow(5, (3 * index + 7 * code + 1) % 102, P)),
        Shear("A", "INTERSECT", index, family, 2, 2 * index + code, pow(5, (5 * index + 11 * code + 2) % 102, P)),
        Shear("B", "LEFT_COMPOSE", index, family, 3, 3 * index + code, pow(5, (7 * index + 13 * code + 3) % 102, P)),
        Shear("A", "INTERSECT", index, family, 4, 5 * index + code, pow(5, (11 * index + 17 * code + 4) % 102, P)),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def public_operand(stage: Shear) -> Weyl:
    return chirp(base_relation(stage.index, stage.family, kind=stage.kind), stage.strength)


def add_scaled(target: Weyl, change: Weyl, scalar: int, subtracting: bool) -> None:
    sign = -1 if subtracting else 1
    for key, value in change.items():
        updated = (target.get(key, 0) + sign * scalar * value) % P
        if updated:
            target[key] = updated
        else:
            target.pop(key, None)


def apply_shear(a: Weyl, b: Weyl, stage: Shear, subtracting: bool = False) -> int:
    target, source = (a, b) if stage.target == "A" else (b, a)
    public = public_operand(stage)
    if stage.operation == "INTERSECT":
        change = intersect(source, public)
    elif stage.operation == "RIGHT_COMPOSE":
        change = compose(source, public)
    elif stage.operation == "LEFT_COMPOSE":
        change = compose(public, source)
    else:
        fail("unknown Weyl operation")
    add_scaled(target, change, stage.scalar, subtracting)
    return len(public) + len(change)


@dataclass
class Carrier:
    seed_family: str
    a: Weyl
    b: Weyl
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, family: str) -> "Carrier":
        if family not in FAMILIES:
            fail("invalid family")
        return cls(family, base_relation(0, family, "A"), base_relation(0, family, "B"))

    def canonical_state(self) -> tuple[Any, ...]:
        return self.seed_family, state_tuple(self.a), state_tuple(self.b), self.stage

    def backing_ids(self) -> tuple[int, int]:
        return id(self.a), id(self.b)


def support(carrier: Carrier, depth: int) -> dict[str, int]:
    return {
        "depth": depth,
        "a_active_weyl_cells": len(canonical(carrier.a)),
        "b_active_weyl_cells": len(canonical(carrier.b)),
        "total_active_weyl_cells": len(canonical(carrier.a)) + len(canonical(carrier.b)),
        "full_two_port_weyl_capacity": 2 * FIBERS * FIBERS * Q * Q,
    }


def forward(carrier: Carrier, depth: int, family: str, reverse_modules: bool = False) -> tuple[list[dict[str, int]], int]:
    if carrier.stage != "IDLE" or depth < 1 or depth > MAX_DEPTH:
        fail("invalid forward request")
    records = []
    transient_peak = 0
    for index in range(depth):
        stages = descriptors(index, family)
        if reverse_modules:
            stages = tuple(reversed(stages))
        for stage in stages:
            transient_peak = max(transient_peak, apply_shear(carrier.a, carrier.b, stage))
        if index + 1 in CHECKPOINTS:
            records.append(support(carrier, index + 1))
    carrier.stage = "FORWARD_COMPLETE"
    return records, transient_peak


def reverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier lacks forward state")
    sequence = [stage for index in reversed(range(depth)) for stage in reversed(descriptors(index, family))]
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, stage in enumerate(sequence):
        if mutation == "WRONG" and position == 0:
            stage = Shear(stage.target, stage.operation, stage.index, stage.family, stage.kind, stage.strength, stage.scalar + 1)
        apply_shear(carrier.a, carrier.b, stage, subtracting=True)
    carrier.stage = "IDLE"


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary unavailable")
    code = 1 if family == "PRIMARY" else 2
    return evaluate(carrier.b, code % 2, (code + 1) % 2, 3 * code, 5 * code)


def transaction(carrier: Carrier, depth: int, family: str) -> dict[str, Any]:
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    generation = carrier.restoration_generation
    records, transient_peak = forward(carrier, depth, family)
    projected = boundary(carrier, family)
    digest = semantic_digest(carrier.a, carrier.b)
    final_support = support(carrier, depth)
    reverse(carrier, depth, family)
    if carrier.canonical_state() != before or carrier.backing_ids() != backing or carrier.restoration_generation != generation:
        fail("Weyl carrier restoration failed")
    carrier.restoration_generation += 1
    return {
        "family": family,
        "depth": depth,
        "boundary": projected,
        "semantic_commitment": digest,
        "support_history": records,
        "final_support": final_support,
        "maximum_public_plus_delta_cells": transient_peak,
        "exact_canonical_state_restored": carrier.canonical_state() == before,
        "same_backing_restored": carrier.backing_ids() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_weyl_coefficients_serialized": False,
    }


def dense_matrix(state: Weyl) -> list[list[int]]:
    size = FIBERS * Q
    result = [[0] * size for _ in range(size)]
    for source in range(FIBERS):
        for x in range(Q):
            for target in range(FIBERS):
                for y in range(Q):
                    result[source * Q + x][target * Q + y] = evaluate(state, source, target, x, (y - x) % Q)
    return result


def matrix_multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    size = len(left)
    return [[sum(left[i][j] * right[j][k] for j in range(size)) % P for k in range(size)] for i in range(size)]


def algebra_checks() -> dict[str, bool]:
    left: Weyl = {(0, 1, 3, 4): 7, (1, 0, 5, 2): 11}
    right: Weyl = {(1, 0, 6, 9): 13, (0, 1, 7, 8): 17}
    dense_left, dense_right = dense_matrix(left), dense_matrix(right)
    dense_hadamard = [[x * y % P for x, y in zip(a, b, strict=True)] for a, b in zip(dense_left, dense_right, strict=True)]
    chirped = chirp(left, 5)
    restored = chirp(chirped, -5)
    paley = base_relation(0, "PRIMARY", "A")
    return {
        "root_has_exact_order17": pow(ROOT, Q, P) == 1 and all(pow(ROOT, k, P) != 1 for k in range(1, Q)),
        "twisted_composition_matches_dense_matrix_product": dense_matrix(compose(left, right)) == matrix_multiply(dense_left, dense_right),
        "phase_mode_intersection_matches_dense_hadamard": dense_matrix(intersect(left, right)) == dense_hadamard,
        "chirp_inverse_restores_exactly": state_tuple(restored) == state_tuple(left),
        "chirp_escapes_fixed12_paley_fusion_chart": any(mode != 0 for (_, _, _, mode) in chirp(paley, 1)),
    }


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    before = Carrier.seal("PRIMARY").canonical_state()
    damaged = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        carrier = Carrier.seal("PRIMARY")
        forward(carrier, 2, "PRIMARY")
        reverse(carrier, 2, "PRIMARY", mutation)
        damaged[mutation] = carrier.canonical_state() != before
    normal = Carrier.seal("PRIMARY")
    forward(normal, 2, "PRIMARY")
    altered = Carrier.seal("PRIMARY")
    forward(altered, 2, "PRIMARY", reverse_modules=True)
    idle = Carrier.seal("PRIMARY")
    return {
        "missing_inverse_fails_restoration": damaged["MISSING"],
        "wrong_inverse_fails_restoration": damaged["WRONG"],
        "reordered_inverse_fails_restoration": damaged["REORDER"],
        "module_reordering_changes_boundary": boundary(normal, "PRIMARY") != boundary(altered, "PRIMARY"),
        "premature_boundary_projection_rejected": raises(lambda: boundary(idle, "PRIMARY")),
        "null_family_rejected": raises(lambda: Carrier.seal("NULL")),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal("PRIMARY")
    before = carrier.canonical_state()
    backing = carrier.backing_ids()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 8, "ALTERNATE")
    fresh = transaction(Carrier.seal("PRIMARY"), 8, "ALTERNATE")
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


def build_result() -> dict[str, Any]:
    cases = [transaction(Carrier.seal(family), depth, family) for family in FAMILIES for depth in CHECKPOINTS]
    checks = controls()
    algebra = algebra_checks()
    reuse = reuse_check()
    if not all(checks.values()) or not all(algebra.values()) or not all(
        case["exact_canonical_state_restored"] and case["same_backing_restored"] for case in cases
    ):
        fail("Weyl chirp qualification failed")
    if not all(reuse[key] for key in ("same_backing_reused", "exact_canonical_state_restored_after_reuse", "unrelated_second_boundary_matches_fresh", "unrelated_second_commitment_matches_fresh")):
        fail("Weyl chirp reuse failed")
    full_capacity = 2 * FIBERS * FIBERS * Q * Q
    return {
        "schema": "CAT_CAS_F103_C17_TWO_FIBER_PALEY_WEYL_CHIRP_FUSION_ESCAPE_RESULTS_V1",
        "claim_candidate": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": P,
            "cyclic_order": Q,
            "fibers": FIBERS,
            "primitive_phase_root": ROOT,
            "native_state": "ADAPTIVE_DISPLACEMENT_PHASE_MODE_WEYL_COEFFICIENTS",
            "native_chirp": "MODE_SHIFT_M_MINUS_2KD_WITH_PHASE_OMEGA_MINUS_KD2",
            "native_composition": "TWISTED_DISPLACEMENT_PHASE_MODE_CONVOLUTION",
            "native_intersection": "PHASE_MODE_CONVOLUTION_AT_FIXED_DISPLACEMENT",
            "fixed12_predecessor_cells_per_port": 12,
            "full_weyl_cells_per_port": FIBERS * FIBERS * Q * Q,
            "final_boundary_only": True,
            "ordinary_relation_table_materialized_on_accepted_path": False,
            "hidden_weyl_coefficients_serialized": False,
            "public_topology_reads_final_answer": False,
        },
        "cases": cases,
        "observed_two_port_active_weyl_cell_range_after_first_update": {
            "minimum": min(
                point["total_active_weyl_cells"]
                for case in cases
                for point in case["support_history"]
            ),
            "maximum": max(
                point["total_active_weyl_cells"]
                for case in cases
                for point in case["support_history"]
            ),
            "capacity": full_capacity,
        },
        "all_observed_post_update_states_use_at_least_2278_of_2312_cells": all(
            point["total_active_weyl_cells"] >= 2278
            for case in cases
            for point in case["support_history"]
        ),
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "initial_two_port_expanded_weyl_cells": 2 * FIBERS * FIBERS * Q,
            "full_two_port_weyl_capacity_field_cells": full_capacity,
            "matched_dense_two_port_relation_field_cells": 2 * (FIBERS * Q) ** 2,
            "matched_identical_weyl_two_port_field_cells": full_capacity,
            "retained_inverse_history_cells": 0,
            "retained_compiled_plan_cells": 0,
            "snapshot_cells": 0,
            "python_container_allocator_runtime_and_whole_process_peak_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest_executed": "IDENTICAL_ADAPTIVE_WEYL_DISPLACEMENT_PHASE_MODE_RECURRENCE",
            "strongest_law_identical_to_accepted": True,
            "dense_matrix_semantics_executed_for_algebra_PARITY": True,
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "F103_C17_TWO_FIBER_PALEY_SEEDS_TWO_FIXED_PUBLIC_WEYL_CHIRP_PROGRAM_FAMILIES_DEPTHS1_2_4_8_16_DIRECT_PROCESS_SOFTWARE",
        "not_established": [
            "GROWING_CYCLIC_ORDER_FOURIER_CHIRP_CLOSURE",
            "GENERAL_WEYL_RELATION_COMPILER",
            "COMPACT_CLOSURE_BELOW_FULL_MATRIX_DIMENSION",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "PHASE_CHIRP_BREAKS_THE_FIXED12_FUSION_BUT_THE_EXACT_WEYL_CHART_BECOMES_NEAR_DENSE_AT_2278_TO_2308_OF_2312_TWO_PORT_CELLS_AND_HAS_AN_IDENTICAL_CLASSICAL_RECURRENCE",
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
