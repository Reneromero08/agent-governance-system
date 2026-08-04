#!/usr/bin/env python3
"""Exact three-class C17 Paley phase-relation algebra over F103.

The relation kernels are constant on the three difference classes {0}, the
nonzero quadratic residues, and the nonresidues of C17.  This automorphism
orbit subspace is closed under both cyclic relation composition and pointwise
relation intersection.  It is a restricted subalgebra, not a quotient that
represents arbitrary C17 relations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


FIELD = 103
ORDER = 17
ZETA17 = 72
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_F103_C17_PALEY_THREE_CLASS_ASSOCIATION_PHASE_RELATION_"
    "ALGEBRA_CLOSES_NATIVE_CYCLIC_COMPOSITION_AND_HADAMARD_INTERSECTION_"
    "ON_ONE_SHARED_UNRESOLVED_RELATION_PORT_THROUGH_DEPTH1024_WITH_FINAL_"
    "ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_IS_THE_IDENTICAL_THREE_"
    "COORDINATE_COMPACT_CLASSICAL_RECURRENCE_AND_DOES_NOT_COVER_GENERAL_"
    "C17_RELATIONS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * x) % FIELD for x in value]


def intersect(left: list[int], right: list[int]) -> list[int]:
    """Pointwise relation intersection in the orbit-class value basis."""
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def compose(left: list[int], right: list[int]) -> list[int]:
    """Cyclic convolution using the exact C17 Paley intersection numbers."""
    a, b, c = left
    d, e, f = right
    return [
        (a * d + 8 * b * e + 8 * c * f) % FIELD,
        (a * e + b * d + 3 * b * e + 4 * b * f + 4 * c * e + 4 * c * f) % FIELD,
        (a * f + c * d + 4 * b * e + 4 * b * f + 4 * c * e + 3 * c * f) % FIELD,
    ]


def phase(exponent: int) -> int:
    return pow(ZETA17, exponent % ORDER, FIELD)


def parameters(index: int, family: str) -> tuple[int, int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    return (
        phase(3 * index + code),
        phase(5 * index + 2 * code + 1),
        phase(7 * index + 3 * code + 2),
        phase(11 * index + 4 * code + 3),
    )


def public_kernel(index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        phase(index + code),
        phase(3 * index + 2 * code + 1),
        phase(5 * index + 4 * code + 2),
    ]


def public_mask(index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        phase(2 * index + code + 1),
        phase(7 * index + 3 * code + 2),
        phase(13 * index + 5 * code + 3),
    ]


def seed(family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 5 if register == "A" else 19
    return [phase(offset + code * (coordinate + 1) + coordinate * coordinate) for coordinate in range(3)]


def boundary_value(b_relation: list[int], family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [phase(2 * code + 1), phase(5 * code + 2), phase(9 * code + 3)]
    return sum(weight * value for weight, value in zip(weights, b_relation, strict=True)) % FIELD


@dataclass
class Carrier:
    cells: list[int]
    seed_family: str
    stage: str = "SEALED"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, family: str) -> "Carrier":
        if family not in FAMILIES:
            fail("invalid seed family")
        return cls(seed(family, "A") + seed(family, "B"), family)

    def a(self) -> list[int]:
        return self.cells[:3]

    def b(self) -> list[int]:
        return self.cells[3:]

    def write(self, a_relation: list[int], b_relation: list[int]) -> None:
        self.cells[:3] = a_relation
        self.cells[3:] = b_relation

    def backing_id(self) -> int:
        return id(self.cells)


def primary_step(a: list[int], b: list[int], index: int) -> tuple[list[int], list[int]]:
    alpha, beta, gamma, delta = parameters(index, "PRIMARY")
    kernel = public_kernel(index, "PRIMARY")
    mask = public_mask(index, "PRIMARY")
    a = add(a, scale(intersect(b, b), alpha))
    b = add(b, scale(compose(a, a), beta))
    a = add(a, scale(compose(b, kernel), gamma))
    b = add(b, scale(intersect(a, mask), delta))
    return a, b


def alternate_step(a: list[int], b: list[int], index: int) -> tuple[list[int], list[int]]:
    alpha, beta, gamma, delta = parameters(index, "ALTERNATE")
    kernel = public_kernel(index, "ALTERNATE")
    mask = public_mask(index, "ALTERNATE")
    b = add(b, scale(intersect(a, mask), delta))
    a = add(a, scale(compose(b, kernel), gamma))
    b = add(b, scale(compose(a, a), beta))
    a = add(a, scale(intersect(b, b), alpha))
    return a, b


def forward(carrier: Carrier, depth: int, family: str, enabled: bool = True) -> None:
    if not enabled:
        carrier.stage = "FORWARD_COMPLETE"
        return
    a, b = carrier.a(), carrier.b()
    step = primary_step if family == "PRIMARY" else alternate_step
    for index in range(depth):
        a, b = step(a, b, index)
    carrier.write(a, b)
    carrier.stage = "FORWARD_COMPLETE"


def inverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    a, b = carrier.a(), carrier.b()
    indices = list(reversed(range(depth)))
    if mutation == "REORDER":
        indices = list(range(depth))
    for index in indices:
        alpha, beta, gamma, delta = parameters(index, family)
        if mutation == "WRONG" and index == indices[0]:
            alpha = (alpha + 1) % FIELD
        kernel = public_kernel(index, family)
        mask = public_mask(index, family)
        if family == "PRIMARY":
            b = subtract(b, scale(intersect(a, mask), delta))
            a = subtract(a, scale(compose(b, kernel), gamma))
            b = subtract(b, scale(compose(a, a), beta))
            a = subtract(a, scale(intersect(b, b), alpha))
        else:
            a = subtract(a, scale(intersect(b, b), alpha))
            b = subtract(b, scale(compose(a, a), beta))
            a = subtract(a, scale(compose(b, kernel), gamma))
            b = subtract(b, scale(intersect(a, mask), delta))
    carrier.write(a, b)
    carrier.stage = "RESTORED"


def project_boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("projection allowed only after forward completion")
    return boundary_value(carrier.b(), family)


def project_hidden_a(_carrier: Carrier) -> list[int]:
    fail("hidden A relation projection is forbidden")


def transaction(carrier: Carrier | None, depth: int, family: str) -> dict[str, Any]:
    if carrier is None:
        fail("null carrier")
    if family not in FAMILIES or depth not in DEPTHS:
        fail("invalid program")
    before = tuple(carrier.cells)
    backing = carrier.backing_id()
    forward(carrier, depth, family)
    forward_a, forward_b = carrier.a(), carrier.b()
    projected = project_boundary(carrier, family)
    inverse(carrier, depth, family)
    if tuple(carrier.cells) != before:
        fail("exact restoration failed")
    carrier.restoration_generation += 1
    return {
        "depth": depth,
        "family": family,
        "boundary": projected,
        "forward_a": forward_a,
        "forward_b": forward_b,
        "forward_commitment": digest_json([forward_a, forward_b]),
        "exact_cells_restored": tuple(carrier.cells) == before,
        "same_backing_restored": carrier.backing_id() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_a_serialized": False,
    }


def compact_classical_reference(depth: int, family: str) -> dict[str, Any]:
    a, b = seed(family, "A"), seed(family, "B")
    for index in range(depth):
        alpha, beta, gamma, delta = parameters(index, family)
        kernel, mask = public_kernel(index, family), public_mask(index, family)
        if family == "PRIMARY":
            bb = [(item * item) % FIELD for item in b]
            a = [(left + alpha * right) % FIELD for left, right in zip(a, bb, strict=True)]
            aa = compose(a, a)
            b = [(left + beta * right) % FIELD for left, right in zip(b, aa, strict=True)]
            bk = compose(b, kernel)
            a = [(left + gamma * right) % FIELD for left, right in zip(a, bk, strict=True)]
            am = [(left * right) % FIELD for left, right in zip(a, mask, strict=True)]
            b = [(left + delta * right) % FIELD for left, right in zip(b, am, strict=True)]
        else:
            am = [(left * right) % FIELD for left, right in zip(a, mask, strict=True)]
            b = [(left + delta * right) % FIELD for left, right in zip(b, am, strict=True)]
            bk = compose(b, kernel)
            a = [(left + gamma * right) % FIELD for left, right in zip(a, bk, strict=True)]
            aa = compose(a, a)
            b = [(left + beta * right) % FIELD for left, right in zip(b, aa, strict=True)]
            bb = [(item * item) % FIELD for item in b]
            a = [(left + alpha * right) % FIELD for left, right in zip(a, bb, strict=True)]
    return {
        "forward_a": a,
        "forward_b": b,
        "forward_commitment": digest_json([a, b]),
        "boundary": boundary_value(b, family),
    }


def one_case(depth: int, family: str) -> dict[str, Any]:
    receipt = transaction(Carrier.seal(family), depth, family)
    classical = compact_classical_reference(depth, family)
    receipt["matches_identical_compact_classical_recurrence"] = all(
        receipt[key] == classical[key]
        for key in ("forward_a", "forward_b", "forward_commitment", "boundary")
    )
    return receipt


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    depth, family = 64, "PRIMARY"
    sealed = Carrier.seal("ALTERNATE")
    before = tuple(sealed.cells)
    missing = Carrier(list(before), "ALTERNATE")
    forward(missing, depth, family)
    wrong = Carrier(list(before), "ALTERNATE")
    forward(wrong, depth, family)
    inverse(wrong, depth, family, mutation="WRONG")
    reordered = Carrier(list(before), "ALTERNATE")
    forward(reordered, depth, family)
    inverse(reordered, depth, family, mutation="REORDER")
    premature = Carrier(list(before), "ALTERNATE")
    disabled = Carrier(list(before), "ALTERNATE")
    forward(disabled, depth, family, enabled=False)
    active = Carrier(list(before), "ALTERNATE")
    forward(active, depth, family)
    merged = Carrier([before[0], before[1], before[1], before[3], before[4], before[4]], "ALTERNATE")
    forward(merged, depth, family)
    return {
        "missing_inverse_fails_restoration": tuple(missing.cells) != before,
        "wrong_inverse_fails_restoration": tuple(wrong.cells) != before,
        "reordered_inverse_fails_for_noncommuting_program": tuple(reordered.cells) != before,
        "premature_projection_rejected": raises(lambda: project_boundary(premature, family)),
        "hidden_a_projection_rejected": raises(lambda: project_hidden_a(sealed)),
        "null_carrier_rejected": raises(lambda: transaction(None, depth, family)),
        "invalid_depth_rejected": raises(lambda: transaction(sealed, 3, family)),
        "carrier_disabled_path_changes_boundary": project_boundary(disabled, family) != project_boundary(active, family),
        "quadratic_residue_nonresidue_overmerge_changes_boundary": project_boundary(merged, family) != project_boundary(active, family),
        "zeta_has_exact_order17": pow(ZETA17, ORDER, FIELD) == 1
        and all(pow(ZETA17, exponent, FIELD) != 1 for exponent in range(1, ORDER)),
    }


def algebra_checks() -> dict[str, bool]:
    zero, residues, nonresidues = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    return {
        "identity_composition": all(compose(zero, basis) == basis and compose(basis, zero) == basis for basis in (zero, residues, nonresidues)),
        "residue_square_intersection_numbers": compose(residues, residues) == [8, 3, 4],
        "residue_nonresidue_intersection_numbers": compose(residues, nonresidues) == [0, 4, 4],
        "nonresidue_square_intersection_numbers": compose(nonresidues, nonresidues) == [8, 4, 3],
        "hadamard_class_idempotents": all(intersect(basis, basis) == basis for basis in (zero, residues, nonresidues)),
        "hadamard_distinct_classes_disjoint": all(intersect(left, right) == [0, 0, 0] for left, right in ((zero, residues), (zero, nonresidues), (residues, nonresidues))),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal("PRIMARY")
    sealed = tuple(carrier.cells)
    backing = carrier.backing_id()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 256, "ALTERNATE")
    fresh = Carrier(list(sealed), "PRIMARY")
    reference = transaction(fresh, 256, "ALTERNATE")
    return {
        "same_backing_reused": carrier.backing_id() == backing,
        "exact_cells_restored_after_reuse": tuple(carrier.cells) == sealed,
        "restoration_generation": carrier.restoration_generation,
        "unrelated_second_boundary_matches_fresh": second["boundary"] == reference["boundary"],
        "unrelated_second_commitment_matches_fresh": second["forward_commitment"] == reference["forward_commitment"],
        "first_boundary": first["boundary"],
        "second_boundary": second["boundary"],
        "snapshot_used": False,
    }


def build_result() -> dict[str, Any]:
    cases = [one_case(depth, family) for family in FAMILIES for depth in DEPTHS]
    checks, algebra, reuse = controls(), algebra_checks(), reuse_check()
    if not all(case["matches_identical_compact_classical_recurrence"] for case in cases):
        fail("matched compact recurrence mismatch")
    if not all(case["exact_cells_restored"] and case["same_backing_restored"] for case in cases):
        fail("restoration failure")
    if not all(checks.values()) or not all(algebra.values()):
        fail("control or algebra failure")
    if not all(reuse[key] for key in ("same_backing_reused", "exact_cells_restored_after_reuse", "unrelated_second_boundary_matches_fresh", "unrelated_second_commitment_matches_fresh")):
        fail("reuse failure")
    return {
        "schema": "CAT_CAS_F103_C17_PALEY_ASSOCIATION_PHASE_RELATION_RESULTS_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": FIELD,
            "group": "C17",
            "phase_root": ZETA17,
            "orbit_classes": ["ZERO", "QUADRATIC_RESIDUE", "QUADRATIC_NONRESIDUE"],
            "resident_relation_coordinates_per_port": 3,
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "native_composition": "PALEY_INTERSECTION_NUMBER_CONVOLUTION",
            "native_intersection": "ORBIT_CLASS_COORDINATEWISE_HADAMARD",
            "shared_unresolved_a_relation_consumers_per_step": 2,
            "shared_unresolved_b_relation_consumers_per_step": 2,
            "final_boundary_only": True,
            "hidden_a_serialized": False,
            "full17_relation_materialized_on_accepted_path": False,
            "dense17_by17_relation_table_materialized": False,
            "public_topology_compilation_reads_final_answer": False,
        },
        "cases": cases,
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "accepted_carrier_field_cells": 6,
            "maximum_public_kernel_and_mask_field_cells_per_step": 6,
            "maximum_logical_working_field_cells_peak": 15,
            "retained_inverse_history_field_cells": 0,
            "full17_two_register_semantic_reference_field_cells": 34,
            "dense_relation_table_cells": 0,
            "matched_compact_classical_carrier_field_cells": 6,
            "matched_compact_classical_logical_working_field_cells_peak": 15,
            "accepted_over_matched_compact_classical_carrier_ratio": 1,
            "accepted_over_matched_compact_classical_working_ratio": 1,
            "phase_and_matched_classical_operation_law_identical": True,
            "python_allocator_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL_THREE_COORDINATE_PALEY_ASSOCIATION_ALGEBRA_RECURRENCE",
            "strongest_executed": True,
            "semantic_reference": "FULL17_RELATION_RECURRENCE_RESERVED_FOR_INDEPENDENT_ORACLE",
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "C17_TRANSLATION_INVARIANT_RELATIONS_CONSTANT_ON_ZERO_QUADRATIC_RESIDUE_AND_NONRESIDUE_DIFFERENCE_CLASSES_OVER_F103_TWO_PROGRAM_FAMILIES_THROUGH_DEPTH1024",
        "outside_family": [
            "GENERAL_C17_TRANSLATION_INVARIANT_RELATIONS",
            "NON_TRANSLATION_INVARIANT_RELATIONS",
            "ARBITRARY_GRAPH_TOPOLOGY",
            "GENERAL_ASSOCIATION_SCHEMES",
        ],
        "not_established": [
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "THE_THREE_CLASS_PALEY_RELATION_ALGEBRA_CLOSES_COMPOSITION_AND_INTERSECTION_COMPACTLY_BUT_IS_A_FIXED_STRUCTURED_FAMILY_WITH_AN_IDENTICAL_THREE_COORDINATE_CLASSICAL_RECURRENCE",
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
