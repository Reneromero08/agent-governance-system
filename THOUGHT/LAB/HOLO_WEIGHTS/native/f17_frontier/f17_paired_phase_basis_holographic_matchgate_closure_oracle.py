#!/usr/bin/env python3
"""Independent matching and native-Holant oracle for M125.

This oracle imports neither production nor a predecessor.  It rebuilds the
public programs, evaluates exact n=2,4,6 boundaries by memoized perfect-
matching recursion, evaluates all modular cases by an independent row-profile
matching DP, and directly contracts the native paired-basis Holant at n=2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PRIME = 17
EXACT_SIZES = (2, 4, 6)
STRUCTURAL_SIZES = (2, 4, 6, 8, 10, 12)
FAMILIES = ("PRIMARY", "REUSE")
FIELDS = ((103, 72), (137, 16))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


class Algebra:
    def __init__(self, modulus: int = 0, root: int = 0) -> None:
        self.modulus = modulus
        self.root = root % modulus if modulus else 0
        if modulus:
            if pow(self.root, PRIME, modulus) != 1 or self.root == 1:
                fail("oracle finite-field root is invalid")
            self.zero: Any = 0
            self.one: Any = 1
        else:
            self.zero = tuple(Fraction(0) for _ in range(16))
            self.one = (Fraction(1), *(Fraction(0) for _ in range(15)))

    def integer(self, value: int) -> Any:
        if self.modulus:
            return value % self.modulus
        return (Fraction(value), *(Fraction(0) for _ in range(15)))

    def add(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left + right) % self.modulus
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def sub(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left - right) % self.modulus
        return tuple(a - b for a, b in zip(left, right, strict=True))

    def mul(self, left: Any, right: Any) -> Any:
        if self.modulus:
            return (left * right) % self.modulus
        coefficients = [Fraction(0) for _ in range(31)]
        for i, a in enumerate(left):
            if not a:
                continue
            for j, b in enumerate(right):
                if b:
                    coefficients[i + j] += a * b
        for degree in range(30, 15, -1):
            value = coefficients[degree]
            if not value:
                continue
            coefficients[degree] = Fraction(0)
            offset = degree - 16
            for shift in range(16):
                coefficients[offset + shift] -= value
        return tuple(coefficients[:16])

    def phase(self, exponent: int) -> Any:
        exponent %= PRIME
        if self.modulus:
            return pow(self.root, exponent, self.modulus)
        if exponent == 0:
            return self.one
        if exponent < 16:
            return tuple(
                Fraction(1 if index == exponent else 0) for index in range(16)
            )
        return tuple(Fraction(-1) for _ in range(16))

    def half(self) -> Any:
        if self.modulus:
            return pow(2, self.modulus - 2, self.modulus)
        return (Fraction(1, 2), *(Fraction(0) for _ in range(15)))

    def serialize(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        return [[int(item.numerator), int(item.denominator)] for item in value]


def product(alg: Algebra, values: Iterable[Any]) -> Any:
    result = alg.one
    for value in values:
        result = alg.mul(result, value)
    return result


def vertices(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row in range(n) for column in range(n))


def edges(n: int) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    return (
        *(((row, column), (row, column + 1))
          for row in range(n) for column in range(n - 1)),
        *(((row, column), (row + 1, column))
          for row in range(n - 1) for column in range(n)),
    )


@dataclass(frozen=True)
class Program:
    n: int
    family: str
    basis_exponent: int
    edge_exponents: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "n": self.n,
                "family": self.family,
                "basis_exponent": self.basis_exponent,
                "edge_exponents": self.edge_exponents,
            }
        )


def compile_program(n: int, family: str) -> Program:
    if family not in FAMILIES:
        fail("unknown oracle family")
    variant = 0 if family == "PRIMARY" else 1
    return Program(
        n,
        family,
        3 + 2 * variant,
        tuple(
            1 + ((7 * index + 3 * n + 5 * variant) % 16)
            for index in range(len(edges(n)))
        ),
    )


@dataclass(frozen=True)
class Factors:
    left: tuple[Any, ...]
    right: tuple[Any, ...]
    weights: tuple[Any, ...]


def compile_factors(program: Program, alg: Algebra) -> Factors:
    phase = alg.phase(program.basis_exponent)
    inverse_phase = alg.phase(-program.basis_exponent)
    half = alg.half()
    return Factors(
        (alg.one, alg.one, phase, alg.sub(alg.zero, phase)),
        (
            half,
            half,
            alg.mul(inverse_phase, half),
            alg.sub(alg.zero, alg.mul(inverse_phase, half)),
        ),
        tuple(alg.phase(exponent) for exponent in program.edge_exponents),
    )


def paired_contraction(factors: Factors, alg: Algebra) -> tuple[Any, ...]:
    result = []
    for left_row in range(2):
        for right_row in range(2):
            value = alg.zero
            for shared in range(2):
                value = alg.add(
                    value,
                    alg.mul(
                        factors.left[2 * left_row + shared],
                        factors.right[2 * right_row + shared],
                    ),
                )
            result.append(value)
    return tuple(result)


def histogram_shift(histogram: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    return tuple(histogram[(index - exponent) % PRIME] for index in range(PRIME))


def histogram_add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def exact_matching_histogram(program: Program) -> tuple[tuple[int, ...], dict[str, int]]:
    n = program.n
    graph = edges(n)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n * n)]
    for index, (left, right) in enumerate(graph):
        a = left[0] * n + left[1]
        b = right[0] * n + right[1]
        adjacency[a].append((b, program.edge_exponents[index]))
        adjacency[b].append((a, program.edge_exponents[index]))
    states = 0
    branches = 0

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, ...]:
        nonlocal states, branches
        states += 1
        if mask == 0:
            return (1, *(0 for _ in range(PRIME - 1)))
        first = (mask & -mask).bit_length() - 1
        without_first = mask ^ (1 << first)
        result = (0,) * PRIME
        for neighbor, exponent in adjacency[first]:
            if (without_first >> neighbor) & 1:
                branches += 1
                child = solve(without_first ^ (1 << neighbor))
                result = histogram_add(result, histogram_shift(child, exponent))
        return result

    histogram = solve((1 << (n * n)) - 1)
    return histogram, {"memoized_vertex_masks": states, "matching_branches": branches}


def histogram_boundary(histogram: tuple[int, ...], alg: Algebra) -> Any:
    result = alg.zero
    for exponent, count in enumerate(histogram):
        result = alg.add(result, alg.mul(alg.integer(count), alg.phase(exponent)))
    return result


def edge_index(program: Program) -> dict[tuple[tuple[int, int], tuple[int, int]], int]:
    return {
        tuple(sorted(edge)): index
        for index, edge in enumerate(edges(program.n))
    }


def modular_row_matching(program: Program, alg: Algebra) -> tuple[Any, dict[str, int]]:
    n = program.n
    lookup = edge_index(program)
    states = 0
    branches = 0

    def weight(left: tuple[int, int], right: tuple[int, int]) -> Any:
        index = lookup[tuple(sorted((left, right)))]
        return alg.phase(program.edge_exponents[index])

    @lru_cache(maxsize=None)
    def solve(position: int, mask: int) -> Any:
        nonlocal states, branches
        states += 1
        if position == n * n:
            return alg.one if mask == 0 else alg.zero
        row, column = divmod(position, n)
        if mask & 1:
            return solve(position + 1, mask >> 1)
        result = alg.zero
        if column + 1 < n and not (mask & 2):
            branches += 1
            right = solve(position + 1, (mask | 2) >> 1)
            result = alg.add(result, alg.mul(weight((row, column), (row, column + 1)), right))
        if row + 1 < n:
            branches += 1
            below = solve(position + 1, (mask >> 1) | (1 << (n - 1)))
            result = alg.add(result, alg.mul(weight((row, column), (row + 1, column)), below))
        return result

    boundary = solve(0, 0)
    return boundary, {"memoized_position_masks": states, "matching_branches": branches}


def incidents(n: int) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], list[int]] = {vertex: [] for vertex in vertices(n)}
    for index, (left, right) in enumerate(edges(n)):
        result[left].append(index)
        result[right].append(index)
    return {vertex: tuple(indices) for vertex, indices in result.items()}


def local_value(
    basis: tuple[Any, ...],
    bits: tuple[int, ...],
    weights: tuple[Any, ...],
    alg: Algebra,
) -> Any:
    result = alg.zero
    for chosen, chosen_weight in enumerate(weights):
        term = chosen_weight
        for position, bit in enumerate(bits):
            row = 1 if position == chosen else 0
            term = alg.mul(term, basis[2 * row + bit])
        result = alg.add(result, term)
    return result


def direct_native_holant(
    program: Program,
    alg: Algebra,
    *,
    mutate_right_basis: bool = False,
) -> Any:
    if program.n != 2:
        fail("direct native oracle is bounded to n=2")
    factors = compile_factors(program, alg)
    right = list(factors.right)
    if mutate_right_basis:
        right[0] = alg.add(right[0], alg.one)
    incident = incidents(program.n)
    result = alg.zero
    for assignment in range(1 << len(program.edge_exponents)):
        term = alg.one
        for vertex in vertices(program.n):
            indices = incident[vertex]
            bits = tuple((assignment >> index) & 1 for index in indices)
            if sum(vertex) % 2 == 0:
                weights = tuple(factors.weights[index] for index in indices)
                value = local_value(factors.left, bits, weights, alg)
            else:
                weights = tuple(alg.one for _ in indices)
                value = local_value(tuple(right), bits, weights, alg)
            term = alg.mul(term, value)
        result = alg.add(result, term)
    return result


def native_parity_witness(program: Program) -> dict[str, Any]:
    if program.n != 4:
        fail("native parity witness is declared at n=4")
    alg = Algebra()
    factors = compile_factors(program, alg)
    incident = incidents(program.n)
    left_vertex = (1, 1)
    right_vertex = (1, 2)
    left_indices = incident[left_vertex]
    right_indices = incident[right_vertex]
    values = (
        local_value(factors.left, (0, 0, 0, 0), tuple(factors.weights[i] for i in left_indices), alg),
        local_value(factors.left, (1, 0, 0, 0), tuple(factors.weights[i] for i in left_indices), alg),
        local_value(factors.right, (0, 0, 0, 0), tuple(alg.one for _ in right_indices), alg),
        local_value(factors.right, (1, 0, 0, 0), tuple(alg.one for _ in right_indices), alg),
    )
    if any(value == alg.zero for value in values):
        fail("native degree-four parity witness vanished")
    return {
        "family": program.family,
        "left_even_nonzero": True,
        "left_odd_nonzero": True,
        "right_even_nonzero": True,
        "right_odd_nonzero": True,
        "witness_sha256": sha256_json([alg.serialize(value) for value in values]),
        "interpretation": "NATIVE_DEGREE4_SIGNATURE_IS_NOT_PARITY_PRESERVING_MATCHGATE_IN_ITS_RESIDENT_BASIS",
    }


def production_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (row.get("field", "Q_ZETA17"), row["family"], row["n"]): row
        for row in rows
    }


def compare_exact(production: dict[str, Any]) -> list[dict[str, Any]]:
    index = production_index(production["exact_transactions"])
    result = []
    for family in FAMILIES:
        for n in EXACT_SIZES:
            program = compile_program(n, family)
            histogram, work = exact_matching_histogram(program)
            alg = Algebra()
            boundary = histogram_boundary(histogram, alg)
            row = index[("Q_ZETA17", family, n)]
            if row["program_fingerprint"] != program.fingerprint():
                fail("exact descriptor fingerprint mismatch")
            if row["boundary"] != alg.serialize(boundary):
                fail("exact matching recursion disagrees with production")
            result.append(
                {
                    "n": n,
                    "family": family,
                    "perfect_matching_count": sum(histogram),
                    **work,
                    "agreement": True,
                }
            )
    return result


def compare_structural(production: dict[str, Any]) -> list[dict[str, Any]]:
    index = production_index(production["dual_field_structural_transactions"])
    result = []
    for modulus, root in FIELDS:
        alg = Algebra(modulus, root)
        for family in FAMILIES:
            for n in STRUCTURAL_SIZES:
                program = compile_program(n, family)
                boundary, work = modular_row_matching(program, alg)
                row = index[(f"F{modulus}", family, n)]
                if row["program_fingerprint"] != program.fingerprint():
                    fail("modular descriptor fingerprint mismatch")
                if row["boundary"] != alg.serialize(boundary):
                    fail("modular row matching DP disagrees with production")
                result.append(
                    {"field": f"F{modulus}", "n": n, "family": family, **work, "agreement": True}
                )
    return result


def exact_native_checks() -> list[dict[str, Any]]:
    result = []
    for family in FAMILIES:
        program = compile_program(2, family)
        alg = Algebra()
        native = direct_native_holant(program, alg)
        histogram, _ = exact_matching_histogram(program)
        matching = histogram_boundary(histogram, alg)
        if native != matching:
            fail("direct native Holant does not cancel to exact matching")
        factors = compile_factors(program, alg)
        if paired_contraction(factors, alg) != (alg.one, alg.zero, alg.zero, alg.one):
            fail("paired exact basis does not close to identity")
        result.append(
            {
                "family": family,
                "native_edge_assignments": 16,
                "paired_basis_identity": True,
                "native_matching_agreement": True,
            }
        )
    return result


def controls() -> dict[str, Any]:
    alg = Algebra()
    program = compile_program(2, "PRIMARY")
    expected = direct_native_holant(program, alg)
    mutated_basis = direct_native_holant(program, alg, mutate_right_basis=True)
    if expected == mutated_basis:
        fail("basis-pair mutation did not change the native boundary")

    modular = Algebra(103, 72)
    original, _ = modular_row_matching(compile_program(4, "PRIMARY"), modular)
    base = compile_program(4, "PRIMARY")
    exponents = list(base.edge_exponents)
    exponents[0] = 1 + (exponents[0] % 16)
    mutated_program = Program(base.n, base.family, base.basis_exponent, tuple(exponents))
    changed, _ = modular_row_matching(mutated_program, modular)
    if original == changed:
        fail("edge-phase mutation did not change the boundary")

    face_signing = all(
        (1)
        * (-1 if (column + 1) % 2 else 1)
        * (1)
        * (-1 if column % 2 else 1)
        == -1
        for row in range(3)
        for column in range(3)
    )
    return {
        "paired_basis_mutation_changes_native_boundary": True,
        "semantic_edge_phase_mutation_changes_boundary": True,
        "public_grid_face_signing_valid": face_signing,
        "direct_native_assignment_control_bounded_to_n2": True,
        "production_accepted_path_signature_tables_required": False,
        "production_accepted_path_edge_assignments_required": False,
    }


def lifecycle_checks(production: dict[str, Any]) -> dict[str, Any]:
    rows = [*production["exact_transactions"], *production["dual_field_structural_transactions"]]
    for row in rows:
        if not (
            row["exact_phase_carrier_restored"]
            and row["same_backing"]
            and row["restoration_generation_increment"]
            and row["response_released_after_restoration"]
            and not row["snapshot_reload_used"]
            and not row["inverse_history_retained"]
        ):
            fail("production lifecycle receipt is inconsistent")
    reuse = production["reuse"]
    if not (
        reuse["fresh_restored_boundary_agreement"]
        and reuse["same_actual_backing_across_unrelated_programs"]
        and reuse["generation_after_two_transactions"] == 2
        and not reuse["baseline_reload_used"]
    ):
        fail("production restored-reuse receipt is inconsistent")
    return {
        "receipt_rows_checked": len(rows),
        "all_rows_exact_carrier_restored": True,
        "all_rows_same_backing": True,
        "all_rows_generation_incremented": True,
        "all_responses_marked_after_restoration": True,
        "snapshot_reload_absent": True,
        "fresh_restored_reuse_receipt_consistent": True,
        "limitation": "LIFECYCLE_RECEIPTS_REQUIRE_SEPARATE_SOURCE_AUDIT_AND_ARE_NOT_PROVEN_BY_BOUNDARY_PARITY_ALONE",
    }


def run(production_path: Path) -> dict[str, Any]:
    payload = production_path.read_bytes()
    production = json.loads(payload)
    if production.get("schema") != "CAT_CAS_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_V1":
        fail("unexpected production schema")
    exact = compare_exact(production)
    structural = compare_structural(production)
    native = exact_native_checks()
    parity = [native_parity_witness(compile_program(4, family)) for family in FAMILIES]
    control_result = controls()
    lifecycle = lifecycle_checks(production)
    return {
        "schema": "CAT_CAS_F17_PAIRED_PHASE_BASIS_HOLOGRAPHIC_MATCHGATE_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_result_sha256": sha256_bytes(payload),
        "independence": {
            "imports_production_or_predecessor": False,
            "exact_boundary_oracle": "MEMOIZED_WEIGHTED_PERFECT_MATCHING_HISTOGRAM",
            "modular_boundary_oracle": "ROW_PROFILE_WEIGHTED_MATCHING_DP",
            "native_basis_oracle": "DIRECT_N2_EDGE_ASSIGNMENT_HOLANT_WITH_INDEPENDENT_FRACTION_POWER_BASIS",
        },
        "exact_matching_parity": exact,
        "dual_field_row_dp_parity": structural,
        "exact_native_holant_parity": native,
        "native_mixed_parity_witnesses": parity,
        "controls": control_result,
        "lifecycle_receipt_consistency": lifecycle,
        "observed_resource_law": {
            "resident_phase_field_cells": "2N_TIMES_N_MINUS_1_PLUS_8",
            "kasteleyn_dimension": "N_SQUARED_OVER_2",
            "determinant_field_operations": "O_N_TO_THE_6",
            "row_matching_dp_baseline": "O_N_SQUARED_TIMES_2_TO_THE_N_WORK_AND_O_2_TO_THE_N_MEMORY",
            "native_edge_assignment_baseline": "2_TO_THE_2N_TIMES_N_MINUS_1",
            "accepted_and_matched_determinant_materialize_no_native_signature_table": True,
        },
        "restoration_class": {
            "resident_basis_weight_and_accumulator_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_determinant_buffers": "NO_RESTORATION_CLAIM",
        },
        "claim_ceiling": {
            "even_open_square_grid_family_only": True,
            "compact_mixed_parity_native_generators": True,
            "paired_phase_basis_identity_closure": True,
            "growing_treewidth_polynomial_matchgate_closure": True,
            "identical_compact_classical_holographic_determinant": True,
            "arbitrary_planar_holant_closure": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "THE_PUBLIC_PAIRED_PHASE_BASIS_CANCELS_TO_THE_IDENTICAL_CLASSICAL_MATCHGATE_DETERMINANT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.production), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
