#!/usr/bin/env python3
"""Exact noncommutative S3 translation-relation phase algebra over F103."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


FIELD = 103
ZETA6 = 57
DEPTHS = (1, 4, 16, 64, 256, 1024)
FAMILIES = ("PRIMARY", "ALTERNATE")
ELEMENTS = tuple(itertools.permutations(range(3)))
INDEX = {element: index for index, element in enumerate(ELEMENTS)}
CLAIM = (
    "BOUNDED_EXACT_F103_S3_NONCOMMUTATIVE_TRANSLATION_RELATION_PHASE_"
    "ALGEBRA_CLOSES_NATIVE_NONABELIAN_COMPOSITION_AND_HADAMARD_"
    "INTERSECTION_ON_ONE_SHARED_UNRESOLVED_SIX_CELL_PORT_THROUGH_"
    "DEPTH1024_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_"
    "BUT_IDENTICAL_SIX_COORDINATE_GROUP_AND_IRREP_CLASSICAL_"
    "RECURRENCES_REMAIN"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[position]] for position in range(3))


def inverse(element: tuple[int, ...]) -> tuple[int, ...]:
    result = [0, 0, 0]
    for source, target in enumerate(element):
        result[target] = source
    return tuple(result)


def sign(element: tuple[int, ...]) -> int:
    inversions = sum(element[left] > element[right] for left in range(3) for right in range(left + 1, 3))
    return -1 if inversions % 2 else 1


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * item) % FIELD for item in value]


def intersect(left: list[int], right: list[int]) -> list[int]:
    return [(x * y) % FIELD for x, y in zip(left, right, strict=True)]


def compose(left: list[int], right: list[int]) -> list[int]:
    result = [0] * len(ELEMENTS)
    for target, element in enumerate(ELEMENTS):
        for source, source_element in enumerate(ELEMENTS):
            residual = multiply(inverse(source_element), element)
            result[target] = (result[target] + left[source] * right[INDEX[residual]]) % FIELD
    return result


def matrix_multiply(left: list[int], right: list[int]) -> list[int]:
    a, b, c, d = left
    e, f, g, h = right
    return [
        (a * e + b * g) % FIELD,
        (a * f + b * h) % FIELD,
        (c * e + d * g) % FIELD,
        (c * f + d * h) % FIELD,
    ]


def representation(element: tuple[int, ...]) -> list[int]:
    columns = []
    for basis in ((1, 0, -1), (0, 1, -1)):
        image = [0, 0, 0]
        for source, value in enumerate(basis):
            image[element[source]] = value
        columns.append((image[0] % FIELD, image[1] % FIELD))
    return [columns[0][0], columns[1][0], columns[0][1], columns[1][1]]


def fourier(value: list[int]) -> tuple[int, int, list[int]]:
    trivial = sum(value) % FIELD
    signed = sum(item * sign(element) for item, element in zip(value, ELEMENTS, strict=True)) % FIELD
    standard = [0, 0, 0, 0]
    for item, element in zip(value, ELEMENTS, strict=True):
        rho = representation(element)
        standard = [(entry + item * matrix_entry) % FIELD for entry, matrix_entry in zip(standard, rho, strict=True)]
    return trivial, signed, standard


def inverse_fourier(value: tuple[int, int, list[int]]) -> list[int]:
    trivial, signed, standard = value
    inverse_six = pow(6, -1, FIELD)
    result = []
    for element in ELEMENTS:
        rho_inverse = representation(inverse(element))
        product = matrix_multiply(rho_inverse, standard)
        trace = (product[0] + product[3]) % FIELD
        result.append(inverse_six * (trivial + sign(element) * signed + 2 * trace) % FIELD)
    return result


def irrep_compose(left: list[int], right: list[int]) -> list[int]:
    left_trivial, left_sign, left_standard = fourier(left)
    right_trivial, right_sign, right_standard = fourier(right)
    return inverse_fourier(
        (
            left_trivial * right_trivial % FIELD,
            left_sign * right_sign % FIELD,
            matrix_multiply(left_standard, right_standard),
        )
    )


def phase(exponent: int) -> int:
    return pow(ZETA6, exponent % 6, FIELD)


def public_vector(index: int, family: str, kind: int) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [phase((kind + 1) * index + (2 * kind + code) * position + position * position + code) for position in range(6)]


def parameters(index: int, family: str) -> tuple[int, int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    return tuple(phase((offset + 1) * index + offset * code + 1) for offset in (1, 2, 4, 5))


def seed(family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 1 if register == "A" else 4
    return [phase(offset + code * (position + 1) + position * position) for position in range(6)]


def boundary_value(b_relation: list[int], family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [phase(code + 2 * position + position * position) for position in range(6)]
    return sum(weight * value for weight, value in zip(weights, b_relation, strict=True)) % FIELD


@dataclass(frozen=True)
class Shear:
    target: str
    operation: str
    scalar: int
    public_operand: tuple[int, ...]


def descriptors(index: int, family: str) -> tuple[Shear, ...]:
    alpha, beta, gamma, delta = parameters(index, family)
    right_kernel = tuple(public_vector(index, family, 1))
    first_mask = tuple(public_vector(index, family, 2))
    left_kernel = tuple(public_vector(index, family, 3))
    second_mask = tuple(public_vector(index, family, 4))
    stages = (
        Shear("B", "RIGHT_COMPOSE", alpha, right_kernel),
        Shear("A", "INTERSECT", beta, first_mask),
        Shear("B", "LEFT_COMPOSE", gamma, left_kernel),
        Shear("A", "INTERSECT", delta, second_mask),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def shear_delta(a: list[int], b: list[int], stage: Shear, composition: Callable[[list[int], list[int]], list[int]] = compose) -> list[int]:
    public = list(stage.public_operand)
    source = a if stage.target == "B" else b
    if stage.operation == "RIGHT_COMPOSE":
        value = composition(source, public)
    elif stage.operation == "LEFT_COMPOSE":
        value = composition(public, source)
    elif stage.operation == "INTERSECT":
        value = intersect(source, public)
    else:
        fail("unknown public S3 relation shear")
    return scale(value, stage.scalar)


def apply_shear(a: list[int], b: list[int], stage: Shear, subtracting: bool = False, composition: Callable[[list[int], list[int]], list[int]] = compose) -> tuple[list[int], list[int]]:
    delta = shear_delta(a, b, stage, composition)
    combine = subtract if subtracting else add
    if stage.target == "A":
        return combine(a, delta), b
    return a, combine(b, delta)


@dataclass
class Carrier:
    cells: list[int]
    seed_family: str
    stage: str = "IDLE"
    restoration_generation: int = 0

    @classmethod
    def seal(cls, family: str) -> "Carrier":
        if family not in FAMILIES:
            fail("invalid seed family")
        return cls(seed(family, "A") + seed(family, "B"), family)

    def a(self) -> list[int]:
        return self.cells[:6]

    def b(self) -> list[int]:
        return self.cells[6:]

    def write(self, a_relation: list[int], b_relation: list[int]) -> None:
        self.cells[:6] = a_relation
        self.cells[6:] = b_relation

    def backing_id(self) -> int:
        return id(self.cells)

    def canonical_state(self) -> tuple[tuple[int, ...], str, str]:
        return tuple(self.cells), self.seed_family, self.stage


def forward(carrier: Carrier, depth: int, family: str, enabled: bool = True, reverse_module_order: bool = False) -> None:
    if carrier.stage != "IDLE":
        fail("carrier is not idle")
    a, b = carrier.a(), carrier.b()
    if enabled:
        for index in range(depth):
            stages = descriptors(index, family)
            if reverse_module_order:
                stages = tuple(reversed(stages))
            for stage in stages:
                a, b = apply_shear(a, b, stage)
    carrier.write(a, b)
    carrier.stage = "FORWARD_COMPLETE"


def reverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("carrier has no forward state")
    a, b = carrier.a(), carrier.b()
    indices = list(reversed(range(depth)))
    if mutation == "REORDER":
        indices = list(range(depth))
    wrong_applied = False
    for index in indices:
        stages = list(reversed(descriptors(index, family)))
        if mutation == "REORDER":
            stages.reverse()
        for stage in stages:
            if mutation == "WRONG" and not wrong_applied:
                stage = Shear(stage.target, stage.operation, (stage.scalar + 1) % FIELD, stage.public_operand)
                wrong_applied = True
            a, b = apply_shear(a, b, stage, subtracting=True)
    carrier.write(a, b)
    carrier.stage = "IDLE"


def project_boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("projection allowed only after forward completion")
    return boundary_value(carrier.b(), family)


def project_hidden(_carrier: Carrier) -> list[int]:
    fail("hidden S3 relation projection is forbidden")


def transaction(carrier: Carrier | None, depth: int, family: str) -> dict[str, Any]:
    if carrier is None:
        fail("null carrier")
    if family not in FAMILIES or depth not in DEPTHS:
        fail("invalid public program")
    before = carrier.canonical_state()
    generation_before = carrier.restoration_generation
    backing = carrier.backing_id()
    forward(carrier, depth, family)
    commitment = digest_json([carrier.a(), carrier.b()])
    boundary = project_boundary(carrier, family)
    reverse(carrier, depth, family)
    if carrier.canonical_state() != before or carrier.backing_id() != backing or carrier.restoration_generation != generation_before:
        fail("exact S3 carrier restoration failed")
    carrier.restoration_generation += 1
    return {
        "depth": depth,
        "family": family,
        "boundary": boundary,
        "forward_commitment": commitment,
        "exact_canonical_state_restored": carrier.canonical_state() == before,
        "same_backing_restored": carrier.backing_id() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_relation_values_serialized": False,
    }


def classical_reference(depth: int, family: str, composition: Callable[[list[int], list[int]], list[int]]) -> dict[str, Any]:
    a, b = seed(family, "A"), seed(family, "B")
    for index in range(depth):
        for stage in descriptors(index, family):
            a, b = apply_shear(a, b, stage, composition=composition)
    return {
        "boundary": boundary_value(b, family),
        "forward_commitment": digest_json([a, b]),
    }


def one_case(depth: int, family: str) -> dict[str, Any]:
    receipt = transaction(Carrier.seal(family), depth, family)
    group_reference = classical_reference(depth, family, compose)
    irrep_reference = classical_reference(depth, family, irrep_compose)
    receipt["matches_identical_group_coordinate_classical_recurrence"] = all(receipt[key] == group_reference[key] for key in ("boundary", "forward_commitment"))
    receipt["matches_irrep_convolution_classical_recurrence"] = all(receipt[key] == irrep_reference[key] for key in ("boundary", "forward_commitment"))
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
    before = sealed.canonical_state()
    missing = Carrier(list(before[0]), "ALTERNATE")
    forward(missing, depth, family)
    wrong = Carrier(list(before[0]), "ALTERNATE")
    forward(wrong, depth, family)
    reverse(wrong, depth, family, mutation="WRONG")
    reordered = Carrier(list(before[0]), "ALTERNATE")
    forward(reordered, depth, family)
    reverse(reordered, depth, family, mutation="REORDER")
    normal = Carrier(list(before[0]), "ALTERNATE")
    forward(normal, depth, family)
    altered_order = Carrier(list(before[0]), "ALTERNATE")
    forward(altered_order, depth, family, reverse_module_order=True)
    disabled = Carrier(list(before[0]), "ALTERNATE")
    forward(disabled, depth, family, enabled=False)
    premature = Carrier(list(before[0]), "ALTERNATE")

    classes: dict[tuple[int, ...], list[int]] = {}
    for position, element in enumerate(ELEMENTS):
        cycle_signature = tuple(sorted(sum(1 for _ in cycle) for cycle in permutation_cycles(element)))
        classes.setdefault(cycle_signature, []).append(position)
    merged_cells = list(before[0])
    for offset in (0, 6):
        for positions in classes.values():
            for position in positions:
                merged_cells[offset + position] = merged_cells[offset + positions[0]]
    merged = Carrier(merged_cells, "ALTERNATE")
    forward(merged, depth, family)

    first, second = noncommuting_pair()
    left_then_right = compose(delta(first), delta(second))
    right_then_left = compose(delta(second), delta(first))
    return {
        "missing_inverse_fails_restoration": missing.canonical_state() != before,
        "wrong_inverse_fails_restoration": wrong.canonical_state() != before,
        "reordered_inverse_fails_for_noncommuting_program": reordered.canonical_state() != before,
        "same_public_modules_reordered_change_boundary": project_boundary(normal, family) != project_boundary(altered_order, family),
        "premature_projection_rejected": raises(lambda: project_boundary(premature, family)),
        "hidden_relation_projection_rejected": raises(lambda: project_hidden(sealed)),
        "null_carrier_rejected": raises(lambda: transaction(None, depth, family)),
        "invalid_depth_rejected": raises(lambda: transaction(sealed, 3, family)),
        "carrier_disabled_path_changes_boundary": project_boundary(disabled, family) != project_boundary(normal, family),
        "conjugacy_class_overmerge_changes_boundary": project_boundary(merged, family) != project_boundary(normal, family),
        "left_and_right_group_composition_differ": left_then_right != right_then_left,
        "zeta_has_exact_order6": pow(ZETA6, 6, FIELD) == 1 and all(pow(ZETA6, exponent, FIELD) != 1 for exponent in range(1, 6)),
    }


def permutation_cycles(element: tuple[int, ...]) -> list[list[int]]:
    cycles = []
    unseen = set(range(3))
    while unseen:
        start = min(unseen)
        cycle = []
        current = start
        while current in unseen:
            unseen.remove(current)
            cycle.append(current)
            current = element[current]
        cycles.append(cycle)
    return cycles


def delta(position: int) -> list[int]:
    result = [0] * 6
    result[position] = 1
    return result


def noncommuting_pair() -> tuple[int, int]:
    for left in range(6):
        for right in range(6):
            if multiply(ELEMENTS[left], ELEMENTS[right]) != multiply(ELEMENTS[right], ELEMENTS[left]):
                return left, right
    fail("S3 unexpectedly commutative")


def algebra_checks() -> dict[str, bool]:
    identity = INDEX[(0, 1, 2)]
    basis = [delta(position) for position in range(6)]
    representation_homomorphism = all(
        representation(multiply(left, right)) == matrix_multiply(representation(left), representation(right))
        for left in ELEMENTS
        for right in ELEMENTS
    )
    return {
        "group_identity_composition": all(compose(basis[identity], item) == item and compose(item, basis[identity]) == item for item in basis),
        "all36_basis_compositions_are_group_products": all(compose(basis[left], basis[right]) == basis[INDEX[multiply(ELEMENTS[left], ELEMENTS[right])]] for left in range(6) for right in range(6)),
        "all36_hadamard_basis_products_exact": all(intersect(basis[left], basis[right]) == (basis[left] if left == right else [0] * 6) for left in range(6) for right in range(6)),
        "standard_representation_is_homomorphism": representation_homomorphism,
        "fourier_roundtrip_all_basis": all(inverse_fourier(fourier(item)) == item for item in basis),
        "irrep_convolution_matches_all36_basis_products": all(irrep_compose(basis[left], basis[right]) == compose(basis[left], basis[right]) for left in range(6) for right in range(6)),
        "noncommutative_basis_witness": compose(basis[noncommuting_pair()[0]], basis[noncommuting_pair()[1]]) != compose(basis[noncommuting_pair()[1]], basis[noncommuting_pair()[0]]),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal("PRIMARY")
    sealed = carrier.canonical_state()
    backing = carrier.backing_id()
    first = transaction(carrier, 1, "PRIMARY")
    second = transaction(carrier, 256, "ALTERNATE")
    fresh = Carrier(list(sealed[0]), "PRIMARY")
    reference = transaction(fresh, 256, "ALTERNATE")
    return {
        "same_backing_reused": carrier.backing_id() == backing,
        "exact_canonical_state_restored_after_reuse": carrier.canonical_state() == sealed,
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
    if not all(case["matches_identical_group_coordinate_classical_recurrence"] and case["matches_irrep_convolution_classical_recurrence"] for case in cases):
        fail("matched compact classical recurrence mismatch")
    if not all(case["exact_canonical_state_restored"] and case["same_backing_restored"] for case in cases):
        fail("restoration failure")
    if not all(checks.values()) or not all(algebra.values()):
        fail("control or algebra failure")
    if not all(reuse[key] for key in ("same_backing_reused", "exact_canonical_state_restored_after_reuse", "unrelated_second_boundary_matches_fresh", "unrelated_second_commitment_matches_fresh")):
        fail("reuse failure")
    return {
        "schema": "CAT_CAS_F103_S3_NONCOMMUTATIVE_PHASE_RELATION_RESULTS_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "experiment": {
            "field": FIELD,
            "group": "S3",
            "group_order": 6,
            "phase_root": ZETA6,
            "resident_relation_coordinates_per_port": 6,
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "case_count": len(cases),
            "native_composition": "NONABELIAN_S3_GROUP_CONVOLUTION",
            "native_intersection": "GROUP_COORDINATEWISE_HADAMARD",
            "shared_unresolved_a_relation_consumers_per_step": 2,
            "shared_unresolved_b_relation_consumers_per_step": 2,
            "final_boundary_only": True,
            "hidden_relation_values_serialized": False,
            "full6_by6_relation_materialized_on_accepted_path": False,
            "answer_bearing_lookup_table_materialized": False,
            "public_group_topology_reads_final_answer": False,
        },
        "cases": cases,
        "controls": checks,
        "algebra_checks": algebra,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "accepted_carrier_field_cells": 12,
            "abstract_atomic_operation_scratch_field_cells": 6,
            "abstract_relation_state_working_field_cells_peak": 18,
            "compiled_current_step_public_operand_field_cells": 24,
            "compiled_current_step_public_scalar_field_cells": 4,
            "conservative_accepted_explicit_python_field_value_slots_peak": 70,
            "retained_inverse_history_field_cells": 0,
            "full6_by6_two_register_semantic_reference_field_cells": 72,
            "dense_relation_table_cells_on_accepted_path": 0,
            "matched_group_coordinate_classical_carrier_field_cells": 12,
            "matched_group_coordinate_classical_abstract_working_field_cells_peak": 18,
            "conservative_matched_group_coordinate_explicit_python_field_value_slots_peak": 58,
            "matched_irrep_coefficient_cells": 6,
            "accepted_over_strongest_compact_classical_carrier_ratio": 1,
            "phase_and_group_coordinate_classical_operation_law_identical": True,
            "irrep_decomposition_reduces_information_dimension": False,
            "python_object_headers_allocator_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL_SIX_COORDINATE_S3_GROUP_RECURRENCE",
            "strongest_executed": True,
            "irrep_crosscheck": "ONE_TRIVIAL_PLUS_ONE_SIGN_PLUS_ONE_TWO_BY_TWO_STANDARD_BLOCK_EQUALS_SIX_FIELD_COORDINATES",
            "irrep_crosscheck_executed": True,
            "semantic_reference": "FULL6_BY6_RELATION_RECURRENCE_RESERVED_FOR_INDEPENDENT_ORACLE",
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "S3_TRANSLATION_INVARIANT_F103_RELATIONS_TWO_SIX_CELL_PORTS_TWO_DECLARED_FOUR_SHEAR_PROGRAM_FAMILIES_THROUGH_DEPTH1024_DIRECT_PROCESS_SOFTWARE",
        "outside_family": ["GENERAL_RELATIONS_ON_SIX_LABELS", "NON_TRANSLATION_INVARIANT_RELATIONS", "ARBITRARY_FINITE_GROUP_COMPILER", "CATVM_CUSTODY"],
        "not_established": ["DISTINCT_PHASE_RESOURCE", "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING", "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION", "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI", "UNBOUNDED_CATALYTIC_COMPUTATION"],
        "next_obstruction": "THE_NONCOMMUTATIVE_RELATION_ALGEBRA_IS_STRICTLY_BROADER_THAN_THE_COMMUTATIVE_PALEY_FAMILY_BUT_ITS_SIX_COORDINATES_AND_IRREP_BLOCKS_ARE_AN_IDENTICAL_COMPACT_CLASSICAL_RECURRENCE",
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
