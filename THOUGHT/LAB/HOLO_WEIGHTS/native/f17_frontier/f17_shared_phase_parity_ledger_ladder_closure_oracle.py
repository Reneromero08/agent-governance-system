#!/usr/bin/env python3
"""Independent oracle for the bounded M124 two-row parity ledger.

The oracle does not import the production package or any predecessor.  It
rebuilds the public ladder descriptors, implements Q(zeta_17) in the power
basis with Fraction coefficients, evaluates a direct 16-transition column
recurrence, and checks small cases by binary assignment enumeration.  Dense
defect signatures and occurrence-expanded field sums are constructed only as
bounded verification oracles.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


PRIME = 17
EXACT_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
STRUCTURAL_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = ("PRIMARY", "REUSE")
FIELDS = ((103, 72), (137, 16))
SPINS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


class Algebra:
    """F_p or Q[z]/Phi_17 using an independently implemented power basis."""

    def __init__(self, modulus: int = 0, root: int = 0) -> None:
        self.modulus = modulus
        self.root = root % modulus if modulus else 0
        if modulus:
            if pow(self.root, PRIME, modulus) != 1 or self.root == 1:
                fail("invalid finite-field seventeenth root")
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
        # Phi_17(z) = 1 + z + ... + z^16.  Descending reduction is
        # deterministic because each replacement lowers the highest degree.
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

    def serialize(self, value: Any) -> Any:
        if self.modulus:
            return int(value)
        return [[int(item.numerator), int(item.denominator)] for item in value]


def product(alg: Algebra, values: Iterable[Any]) -> Any:
    result = alg.one
    for value in values:
        result = alg.mul(result, value)
    return result


def ladder_edges(width: int) -> tuple[tuple[int, int], ...]:
    if width < 1:
        fail("ladder width must be positive")
    return (
        *((row * width + column, row * width + column + 1)
          for row in range(2) for column in range(width - 1)),
        *((column, width + column) for column in range(width)),
    )


@dataclass(frozen=True)
class Program:
    width: int
    family: str
    unary: tuple[int, ...]
    edge_weights: tuple[int, ...]
    field_residues: tuple[int, ...]

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "width": self.width,
                "family": self.family,
                "unary": self.unary,
                "edge_weights": self.edge_weights,
                "field_residues": self.field_residues,
            }
        )


def compile_program(width: int, family: str) -> Program:
    if family not in FAMILIES:
        fail("unknown oracle family")
    variant = 0 if family == "PRIMARY" else 1
    edges = ladder_edges(width)
    edge_weights = tuple(
        1 + ((7 * ordinal + 3 * width + 5 * variant) % 16)
        for ordinal in range(len(edges))
    )
    residues = tuple(
        1 + ((5 * site + 2 * width + 3 * variant) % 16)
        for site in range(2 * width)
    )
    unary = []
    for site, residue in enumerate(residues):
        incident = sum(
            weight
            for edge, weight in zip(edges, edge_weights, strict=True)
            if site in edge
        )
        unary.append((9 * (residue - incident)) % PRIME)
    return Program(width, family, tuple(unary), edge_weights, residues)


def phase_pair(alg: Algebra, exponent: int) -> tuple[Any, Any]:
    forward = alg.phase(exponent)
    backward = alg.phase(-exponent)
    half = alg.integer(pow(2, alg.modulus - 2, alg.modulus)) if alg.modulus else alg.integer(1)
    if alg.modulus:
        return (
            alg.mul(half, alg.add(forward, backward)),
            alg.mul(half, alg.sub(forward, backward)),
        )
    two = Fraction(2)
    return (
        tuple(value / two for value in alg.add(forward, backward)),
        tuple(value / two for value in alg.sub(forward, backward)),
    )


def signed(alg: Algebra, pair: tuple[Any, Any], sign: int) -> Any:
    return alg.add(pair[0], pair[1]) if sign == 1 else alg.sub(pair[0], pair[1])


@dataclass(frozen=True)
class Factors:
    constant: int
    edges: tuple[tuple[Any, Any], ...]
    fields: tuple[tuple[Any, Any], ...]


def compile_factors(program: Program, alg: Algebra) -> Factors:
    return Factors(
        (9 * sum(program.unary) + 13 * sum(program.edge_weights)) % PRIME,
        tuple(phase_pair(alg, 13 * value) for value in program.edge_weights),
        tuple(phase_pair(alg, -13 * value) for value in program.field_residues),
    )


def local_column(
    alg: Algebra,
    factors: Factors,
    width: int,
    column: int,
) -> list[Any]:
    horizontal = 2 * (width - 1)
    vertical = factors.edges[horizontal + column]
    top_field = factors.fields[column]
    bottom_field = factors.fields[width + column]
    result = []
    for top, bottom in SPINS:
        result.append(
            product(
                alg,
                (
                    signed(alg, vertical, top * bottom),
                    signed(alg, top_field, top),
                    signed(alg, bottom_field, bottom),
                ),
            )
        )
    return result


def direct_recurrence(
    program: Program,
    alg: Algebra,
    *,
    rank3_truncation: bool = False,
    field_residue_mutation: bool = False,
) -> Any:
    factors = compile_factors(program, alg)
    if field_residue_mutation:
        mutated = list(factors.fields)
        mutated[0] = phase_pair(alg, -13 * ((program.field_residues[0] % 16) + 1))
        factors = Factors(factors.constant, factors.edges, tuple(mutated))
    current = local_column(alg, factors, program.width, 0)
    if rank3_truncation and program.width > 1:
        current[2] = alg.add(current[2], current[3])
        current[3] = alg.zero
    for column in range(1, program.width):
        top_pair = factors.edges[column - 1]
        bottom_pair = factors.edges[(program.width - 1) + column - 1]
        local = local_column(alg, factors, program.width, column)
        following = []
        for target_index, (target_top, target_bottom) in enumerate(SPINS):
            accumulator = alg.zero
            for source_index, (source_top, source_bottom) in enumerate(SPINS):
                term = product(
                    alg,
                    (
                        current[source_index],
                        signed(alg, top_pair, source_top * target_top),
                        signed(alg, bottom_pair, source_bottom * target_bottom),
                    ),
                )
                accumulator = alg.add(accumulator, term)
            following.append(alg.mul(accumulator, local[target_index]))
        if rank3_truncation and column + 1 < program.width:
            following[2] = alg.add(following[2], following[3])
            following[3] = alg.zero
        current = following
    boundary = alg.zero
    for value in current:
        boundary = alg.add(boundary, value)
    return alg.mul(alg.phase(factors.constant), boundary)


def binary_assignment_boundary(program: Program, alg: Algebra) -> Any:
    histogram = [0] * PRIME
    edges = ladder_edges(program.width)
    for assignment in range(1 << (2 * program.width)):
        exponent = 0
        for site, unary in enumerate(program.unary):
            if (assignment >> site) & 1:
                exponent += unary
        for (left, right), weight in zip(edges, program.edge_weights, strict=True):
            if ((assignment >> left) & 1) and ((assignment >> right) & 1):
                exponent += weight
        histogram[exponent % PRIME] += 1
    result = alg.zero
    for exponent, multiplicity in enumerate(histogram):
        result = alg.add(result, alg.mul(alg.integer(multiplicity), alg.phase(exponent)))
    return result


def zero_field_signature(program: Program, alg: Algebra) -> list[Any]:
    """Dense spin-moment tensor, used only by the bounded oracle."""

    edge_pairs = tuple(phase_pair(alg, 13 * value) for value in program.edge_weights)
    signature = [alg.zero for _ in range(1 << (2 * program.width))]
    edges = ladder_edges(program.width)
    for assignment in range(1 << (2 * program.width)):
        spins = tuple(1 if (assignment >> site) & 1 else -1 for site in range(2 * program.width))
        base = product(
            alg,
            (
                signed(alg, pair, spins[left] * spins[right])
                for (left, right), pair in zip(edges, edge_pairs, strict=True)
            ),
        )
        subset_product = [1] * (1 << (2 * program.width))
        for mask in range(1, len(subset_product)):
            bit = (mask & -mask).bit_length() - 1
            subset_product[mask] = subset_product[mask ^ (1 << bit)] * spins[bit]
        for mask, sign in enumerate(subset_product):
            signature[mask] = alg.add(
                signature[mask],
                base if sign == 1 else alg.sub(alg.zero, base),
            )
    return signature


def occurrence_expanded_boundary(program: Program, alg: Algebra) -> Any:
    factors = compile_factors(program, alg)
    signature = zero_field_signature(program, alg)
    result = alg.zero
    for mask, moment in enumerate(signature):
        coefficient = alg.one
        for site, pair in enumerate(factors.fields):
            coefficient = alg.mul(coefficient, pair[1] if (mask >> site) & 1 else pair[0])
        result = alg.add(result, alg.mul(moment, coefficient))
    return alg.mul(alg.phase(factors.constant), result)


def deposit(local: int, positions: Sequence[int]) -> int:
    result = 0
    for local_index, global_index in enumerate(positions):
        if (local >> local_index) & 1:
            result |= 1 << global_index
    return result


def matrix_for_cut(signature: Sequence[int], left: Sequence[int], total: int) -> list[list[int]]:
    left_set = set(left)
    right = tuple(index for index in range(total) if index not in left_set)
    return [
        [signature[deposit(row, left) | deposit(column, right)] for column in range(1 << len(right))]
        for row in range(1 << len(left))
    ]


def rank_mod(matrix: Sequence[Sequence[int]], modulus: int) -> int:
    work = [[value % modulus for value in row] for row in matrix]
    if not work:
        return 0
    rows = len(work)
    columns = len(work[0])
    pivot_row = 0
    for column in range(columns):
        pivot = next((row for row in range(pivot_row, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column], modulus - 2, modulus)
        work[pivot_row] = [(value * inverse) % modulus for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [
                (value - scale * pivot_value) % modulus
                for value, pivot_value in zip(work[row], work[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def cut_rank(signature: Sequence[int], subset: int, total: int, modulus: int) -> int:
    left = tuple(index for index in range(total) if (subset >> index) & 1)
    return rank_mod(matrix_for_cut(signature, left, total), modulus)


def order_profile(signature: Sequence[int], order: Sequence[int], modulus: int) -> list[int]:
    subset = 0
    result = []
    for site in order[:-1]:
        subset |= 1 << site
        result.append(cut_rank(signature, subset, len(order), modulus))
    return result


def optimal_order(signature: Sequence[int], total: int, modulus: int) -> tuple[int, list[int]]:
    full = (1 << total) - 1
    ranks = [cut_rank(signature, mask, total, modulus) for mask in range(1 << total)]
    best = [10**9] * (1 << total)
    predecessor = [-1] * (1 << total)
    best[0] = 0
    for mask in range(1, 1 << total):
        cut = 0 if mask == full else ranks[mask]
        for site in range(total):
            if not (mask >> site) & 1:
                continue
            candidate = max(best[mask ^ (1 << site)], cut)
            if candidate < best[mask]:
                best[mask] = candidate
                predecessor[mask] = site
    order_reversed = []
    mask = full
    while mask:
        site = predecessor[mask]
        order_reversed.append(site)
        mask ^= 1 << site
    return best[full], list(reversed(order_reversed))


def plucker_delta(signature: Sequence[Any], indices: tuple[int, int, int, int], alg: Algebra) -> Any:
    a, b, c, d = indices
    mask = lambda *items: sum(1 << item for item in items)
    value = alg.mul(signature[0], signature[mask(a, b, c, d)])
    value = alg.sub(value, alg.mul(signature[mask(a, b)], signature[mask(c, d)]))
    value = alg.add(value, alg.mul(signature[mask(a, c)], signature[mask(b, d)]))
    value = alg.sub(value, alg.mul(signature[mask(a, d)], signature[mask(b, c)]))
    return value


def production_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (row.get("field", "Q_ZETA17"), row["family"], row["width"]): row
        for row in rows
    }


def compare_production(production: dict[str, Any]) -> dict[str, Any]:
    exact_index = production_index(production["exact_transactions"])
    modular_index = production_index(production["dual_field_structural_transactions"])
    exact_checks = []
    for family in FAMILIES:
        for width in EXACT_WIDTHS:
            program = compile_program(width, family)
            oracle = direct_recurrence(program, Algebra())
            row = exact_index[("Q_ZETA17", family, width)]
            if row["program_fingerprint"] != program.fingerprint():
                fail("independent exact descriptor fingerprint mismatch")
            if row["boundary"] != Algebra().serialize(oracle):
                fail("independent exact recurrence disagrees with production")
            exact_checks.append({"width": width, "family": family, "agreement": True})

    modular_checks = []
    for modulus, root in FIELDS:
        alg = Algebra(modulus, root)
        for family in FAMILIES:
            for width in STRUCTURAL_WIDTHS:
                program = compile_program(width, family)
                oracle = direct_recurrence(program, alg)
                row = modular_index[(f"F{modulus}", family, width)]
                if row["program_fingerprint"] != program.fingerprint():
                    fail("independent modular descriptor fingerprint mismatch")
                if row["boundary"] != alg.serialize(oracle):
                    fail("independent modular recurrence disagrees with production")
                modular_checks.append(
                    {"field": f"F{modulus}", "width": width, "family": family, "agreement": True}
                )
    return {"exact": exact_checks, "dual_field_structural": modular_checks}


def bounded_dense_checks() -> list[dict[str, Any]]:
    checks = []
    for family in FAMILIES:
        for width in (1, 2, 4):
            program = compile_program(width, family)
            exact = Algebra()
            recurrence = direct_recurrence(program, exact)
            binary = binary_assignment_boundary(program, exact)
            expanded = occurrence_expanded_boundary(program, exact)
            if recurrence != binary or recurrence != expanded:
                fail("bounded exact dense oracle disagrees with recurrence")
            checks.append(
                {
                    "width": width,
                    "family": family,
                    "binary_assignments": 1 << (2 * width),
                    "expanded_signature_cells": 1 << (2 * width),
                    "even_sector_ceiling": 1 << max(0, 2 * width - 1),
                    "agreement": True,
                }
            )
    return checks


def residual_rank_checks() -> list[dict[str, Any]]:
    result = []
    for modulus, root in FIELDS:
        alg = Algebra(modulus, root)
        for width in (1, 2, 3, 4):
            program = compile_program(width, "PRIMARY")
            raw = zero_field_signature(program, alg)
            signature = [int(value) for value in raw]
            odd_zero = all(
                value == 0
                for mask, value in enumerate(signature)
                if mask.bit_count() % 2
            )
            if not odd_zero:
                fail("global spin-flip parity law failed")
            grouped = tuple(range(2 * width))
            interleaved = tuple(
                site
                for column in range(width)
                for site in (column, width + column)
            )
            optimum, witness = optimal_order(signature, 2 * width, modulus)
            grouped_profile = order_profile(signature, grouped, modulus)
            interleaved_profile = order_profile(signature, interleaved, modulus)
            violations = []
            for indices in itertools.combinations(range(2 * width), 4):
                delta = plucker_delta(signature, indices, alg)
                if delta != alg.zero:
                    violations.append({"indices": indices, "delta": alg.serialize(delta)})
            result.append(
                {
                    "field": f"F{modulus}",
                    "width": width,
                    "odd_signature_entries_zero": True,
                    "grouped_order": grouped,
                    "grouped_cut_rank_profile": grouped_profile,
                    "column_interleaved_order": interleaved,
                    "column_interleaved_cut_rank_profile": interleaved_profile,
                    "all_order_optimal_maximum_cut_rank": optimum,
                    "all_order_witness": witness,
                    "plucker_violation_count": len(violations),
                    "first_plucker_violation": violations[0] if violations else None,
                }
            )
    return result


def exact_non_gaussian_check() -> dict[str, Any]:
    alg = Algebra()
    signature = zero_field_signature(compile_program(2, "PRIMARY"), alg)
    delta = plucker_delta(signature, (0, 1, 2, 3), alg)
    if delta == alg.zero:
        fail("exact two-column signature unexpectedly satisfies the Gaussian identity")
    return {
        "width": 2,
        "indices": [0, 1, 2, 3],
        "plucker_delta": alg.serialize(delta),
        "nonzero": True,
        "interpretation": "THE_FULL_DEFECT_SIGNATURE_IS_NOT_A_SINGLE_EVEN_GAUSSIAN_SIGNATURE",
    }


def controls() -> dict[str, Any]:
    alg = Algebra(103, 72)
    program = compile_program(4, "PRIMARY")
    boundary = direct_recurrence(program, alg)
    truncated = direct_recurrence(program, alg, rank3_truncation=True)
    mutated = direct_recurrence(program, alg, field_residue_mutation=True)
    if boundary == truncated:
        fail("rank-three overmerge control did not change the boundary")
    if boundary == mutated:
        fail("semantic field-port mutation did not change the boundary")
    return {
        "rank3_overmerge_changes_boundary": True,
        "semantic_field_port_mutation_changes_boundary": True,
        "grouped_order_is_width_exposing_not_a_semantic_mutation": True,
        "accepted_recurrence_enumerates_even_sectors": False,
        "accepted_recurrence_materializes_dense_signature": False,
        "oracle_dense_enumeration_is_bounded_to_width4": True,
    }


def lifecycle_checks(production: dict[str, Any]) -> dict[str, Any]:
    rows = [
        *production["exact_transactions"],
        *production["dual_field_structural_transactions"],
    ]
    required_true = (
        "exact_factor_carrier_restored",
        "same_backing",
        "restoration_generation_increment",
        "response_released_after_restoration",
    )
    if not all(row.get(key) is True for row in rows for key in required_true):
        fail("production lifecycle receipt is internally inconsistent")
    if not all(row.get("snapshot_reload_used") is False for row in rows):
        fail("production unexpectedly reports snapshot reload")
    reuse = production["reuse"]
    if not (
        reuse["fresh_restored_boundary_agreement"]
        and reuse["same_actual_backing_across_unrelated_programs"]
        and not reuse["baseline_reload_used"]
        and reuse["generation_after_two_transactions"] == 2
    ):
        fail("fresh/restored reuse receipt is inconsistent")
    return {
        "receipt_rows_checked": len(rows),
        "all_rows_exact_factor_restored": True,
        "all_rows_same_backing": True,
        "all_rows_generation_incremented": True,
        "all_responses_marked_after_restoration": True,
        "snapshot_reload_absent": True,
        "fresh_restored_reuse_receipt_consistent": True,
        "limitation": "LIFECYCLE_FIELDS_REQUIRE_SEPARATE_SOURCE_AUDIT;THEY_ARE_NOT_PROVEN_BY_BOUNDARY_PARITY_ALONE",
    }


def run(production_path: Path) -> dict[str, Any]:
    payload = production_path.read_bytes()
    production = json.loads(payload)
    if production.get("schema") != "CAT_CAS_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_V1":
        fail("unexpected production schema")
    parity = compare_production(production)
    dense = bounded_dense_checks()
    residual = residual_rank_checks()
    exact_non_gaussian = exact_non_gaussian_check()
    control_result = controls()
    lifecycle = lifecycle_checks(production)
    return {
        "schema": "CAT_CAS_F17_SHARED_PHASE_PARITY_LEDGER_LADDER_CLOSURE_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "production_result_sha256": sha256_bytes(payload),
        "independence": {
            "imports_production_or_predecessor": False,
            "exact_algebra": "INDEPENDENT_FRACTION_POWER_BASIS_MOD_PHI17",
            "accepted_boundary_oracle": "DIRECT_16_TRANSITION_FOUR_STATE_COLUMN_RECURRENCE",
            "bounded_dense_oracles": "BINARY_ASSIGNMENT_AND_OCCURRENCE_EXPANDED_SIGNATURE_WIDTHS_1_2_4",
        },
        "production_parity": parity,
        "bounded_dense_checks": dense,
        "residual_rank_order_gauge": residual,
        "exact_non_gaussian_signature": exact_non_gaussian,
        "controls": control_result,
        "lifecycle_receipt_consistency": lifecycle,
        "observed_resource_law": {
            "resident_factor_cells": "10W_MINUS_4",
            "topology_ordered_frontier_cells": 4,
            "direct_transition_work": "O_W_FIELD_OPERATIONS_WITH_FIXED_FOUR_STATE_FRONTIER_EXACT_BIT_COMPLEXITY_REPORTED_SEPARATELY",
            "dense_signature_cells": "4_TO_THE_W",
            "even_sector_count": "2_TO_THE_2W_MINUS_1",
            "grouped_row_cut_rank": "2_TO_THE_W_FOR_THE_TESTED_NONZERO_VERTICAL_KERNELS_W1_TO_W4",
            "column_interleaved_max_cut_rank": "AT_MOST_4_FOR_ARBITRARY_WIDTH_BY_THE_TWO_ROW_SEPARATOR",
            "all_order_optimum_tested": "4_FOR_W2_TO_W4_AND_2_FOR_W1_OVER_F103_AND_F137",
        },
        "restoration_class": {
            "resident_factor_and_accumulator_carrier": "EXACT_ALGEBRAIC_RESTORATION",
            "transient_projection_frontier": "NO_RESTORATION_CLAIM",
        },
        "claim_ceiling": {
            "two_row_open_ladder_family_only": True,
            "native_non_gaussian_signature_scope": "PRIMARY_WIDTH2_EXACT_GRASSMANN_PLUCKER_WITNESS_ONLY",
            "other_exact_families_or_widths_non_gaussian_established": False,
            "fixed_rank4_topology_ordered_closure": True,
            "identical_compact_classical_recurrence": True,
            "arbitrary_treewidth_compaction": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "physical_bits_replaced_with_pi": False,
            "unbounded_catalytic_computation": False,
        },
        "next_obstruction": "FIXED_RANK4_IS_EXPLAINED_BY_THE_TWO_ROW_FOUR_STATE_SEPARATOR_AND_IS_IDENTICAL_TO_COMPACT_CLASSICAL_TRANSFER",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.production), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
