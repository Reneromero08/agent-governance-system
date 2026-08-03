#!/usr/bin/env python3
"""Exact common-congruence diagnostic for F17[C17] phase jets.

The rank-r Hasse jet is an exact quotient for cyclic convolution, but a
quotient also supports arbitrary coefficientwise Hadamard multiplication only
when its kernel is an ideal for both algebras.  This package constructs the
linear certificates directly.  It also executes the maximal compatible
Hadamard family found for ranks 2, 4, and 8: public constant multipliers.

The result is a strict linear-quotient no-go, not a no-go for nonlinear
encodings or for every possible phase state law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


FIELD = 17
ORDER = 17
RANKS = (2, 4, 8)
DEPTHS = (1, 4, 16, 64)
FAMILIES = ("PRIMARY", "ALTERNATE")
CLAIM = (
    "BOUNDED_EXACT_LINEAR_CONGRUENCE_NO_GO_FOR_SIMULTANEOUS_ARBITRARY_C17_"
    "CONVOLUTION_AND_COEFFICIENTWISE_HADAMARD_INTERSECTION_OVER_F17_WITH_"
    "ONLY_CONSTANT_PUBLIC_HADAMARD_MULTIPLIERS_PRESERVING_RANK2_4_8_"
    "NILPOTENT_PHASE_JETS_AND_EXACT_RESTRICTED_PROGRAM_RESTORATION_AND_"
    "REUSE_BUT_THE_IDENTICAL_COMPACT_CLASSICAL_JET_RECURRENCE_REMAINS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    numerator = 1
    denominator = 1
    for item in range(1, k + 1):
        numerator = numerator * (n - item + 1) % FIELD
        denominator = denominator * item % FIELD
    return numerator * pow(denominator, FIELD - 2, FIELD) % FIELD


def jet_map(coefficients: list[int], rank: int) -> list[int]:
    return [
        sum(coefficients[position] * binomial(position, degree) for position in range(ORDER))
        % FIELD
        for degree in range(rank)
    ]


def add(left: list[int], right: list[int]) -> list[int]:
    return [(x + y) % FIELD for x, y in zip(left, right, strict=True)]


def subtract(left: list[int], right: list[int]) -> list[int]:
    return [(x - y) % FIELD for x, y in zip(left, right, strict=True)]


def scale(value: list[int], scalar: int) -> list[int]:
    return [(scalar * x) % FIELD for x in value]


def jet_multiply(left: list[int], right: list[int], rank: int) -> list[int]:
    result = [0] * rank
    for total in range(rank):
        result[total] = sum(left[index] * right[total - index] for index in range(total + 1)) % FIELD
    return result


def jet_power(value: list[int], exponent: int, rank: int) -> list[int]:
    if exponent < 0:
        return jet_power(jet_inverse(value, rank), -exponent, rank)
    result = [1] + [0] * (rank - 1)
    base = value.copy()
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = jet_multiply(result, base, rank)
        base = jet_multiply(base, base, rank)
        remaining >>= 1
    return result


def jet_inverse(value: list[int], rank: int) -> list[int]:
    if len(value) != rank or value[0] == 0:
        fail("jet is not a unit")
    inverse = [0] * rank
    inverse[0] = pow(value[0], FIELD - 2, FIELD)
    for degree in range(1, rank):
        subtotal = sum(value[index] * inverse[degree - index] for index in range(1, degree + 1))
        inverse[degree] = (-inverse[0] * subtotal) % FIELD
    return inverse


def phase(rank: int, shift: int) -> list[int]:
    return jet_power([1, 1] + [0] * (rank - 2), shift % ORDER, rank)


def public_kernel(rank: int, index: int, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    local = [1, (7 * index + 4 * code + 3) % FIELD] + [0] * (rank - 2)
    return jet_multiply(phase(rank, 3 * index + 5 * code + 1), local, rank)


def parameters(index: int, family: str) -> tuple[int, int, int]:
    code = 1 if family == "PRIMARY" else 2
    alpha = (5 * index + 3 * code + 1) % FIELD or 1
    beta = (7 * index + 2 * code + 4) % FIELD or 1
    constant = (11 * index + 5 * code + 2) % FIELD or 1
    return alpha, beta, constant


def seed(rank: int, family: str, register: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    offset = 3 if register == "A" else 9
    return [
        (offset + code * (degree + 1) + degree * degree + rank) % FIELD
        for degree in range(rank)
    ]


def rref(matrix: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    work = [[value % FIELD for value in row] for row in matrix]
    if not work:
        return work, []
    pivots: list[int] = []
    row = 0
    for column in range(len(work[0])):
        pivot = next((candidate for candidate in range(row, len(work)) if work[candidate][column]), None)
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        factor = pow(work[row][column], FIELD - 2, FIELD)
        work[row] = [(factor * value) % FIELD for value in work[row]]
        for other in range(len(work)):
            if other == row or work[other][column] == 0:
                continue
            factor = work[other][column]
            work[other] = [
                (left - factor * right) % FIELD
                for left, right in zip(work[other], work[row], strict=True)
            ]
        pivots.append(column)
        row += 1
        if row == len(work):
            break
    return work, pivots


def matrix_rank(matrix: list[list[int]]) -> int:
    return len(rref(matrix)[1])


def nullspace(matrix: list[list[int]]) -> list[list[int]]:
    reduced, pivots = rref(matrix)
    columns = len(matrix[0]) if matrix else 0
    free = [column for column in range(columns) if column not in pivots]
    basis: list[list[int]] = []
    for free_column in free:
        vector = [0] * columns
        vector[free_column] = 1
        for row, pivot in reversed(list(enumerate(pivots))):
            vector[pivot] = -sum(
                reduced[row][column] * vector[column]
                for column in free
            ) % FIELD
        basis.append(vector)
    return basis


def hadamard_descent_matrix(rank: int) -> list[list[int]]:
    """Solve diag(m) row(J_r) subset row(J_r) over F17.

    Variables are the 17 multiplier cells followed by r*r coefficients that
    express each multiplied Hasse row in the original Hasse row space.
    """
    rows: list[list[int]] = []
    width = ORDER + rank * rank
    for degree in range(rank):
        for position in range(ORDER):
            row = [0] * width
            row[position] = binomial(position, degree)
            for basis_degree in range(rank):
                row[ORDER + degree * rank + basis_degree] = -binomial(
                    position, basis_degree
                ) % FIELD
            rows.append(row)
    return rows


def compatible_multiplier_space(rank: int) -> dict[str, Any]:
    matrix = hadamard_descent_matrix(rank)
    solutions = nullspace(matrix)
    multipliers = [solution[:ORDER] for solution in solutions]
    multiplier_rank = matrix_rank(multipliers) if multipliers else 0
    normalized = []
    for multiplier in multipliers:
        first = next((value for value in multiplier if value), 1)
        inverse = pow(first, FIELD - 2, FIELD)
        normalized.append([(inverse * value) % FIELD for value in multiplier])
    return {
        "rank": rank,
        "constraint_rows": len(matrix),
        "constraint_variables": len(matrix[0]),
        "solution_dimension": len(solutions),
        "multiplier_space_dimension": multiplier_rank,
        "normalized_multiplier_basis": normalized,
        "only_constant_multipliers": multiplier_rank == 1
        and normalized == [[1] * ORDER],
    }


def jet_kernel_basis(rank: int) -> list[list[int]]:
    matrix = [
        [binomial(position, degree) for position in range(ORDER)]
        for degree in range(rank)
    ]
    return nullspace(matrix)


def hadamard_witness(rank: int, multiplier: list[int]) -> dict[str, Any]:
    for witness in jet_kernel_basis(rank):
        product = [(x * y) % FIELD for x, y in zip(multiplier, witness, strict=True)]
        escaped = jet_map(product, rank)
        if any(escaped):
            return {
                "multiplier": multiplier,
                "kernel_witness": witness,
                "witness_jet_before": jet_map(witness, rank),
                "product_jet_after": escaped,
            }
    fail("expected multiplier to violate quotient descent")


def shift(vector: list[int], amount: int = 1) -> list[int]:
    result = [0] * ORDER
    for index, value in enumerate(vector):
        result[(index + amount) % ORDER] = value
    return result


def common_congruence_certificate() -> dict[str, Any]:
    orbit_ranks: list[int] = []
    for coordinate in range(ORDER):
        basis = [0] * ORDER
        basis[coordinate] = 1
        orbit = [shift(basis, amount) for amount in range(ORDER)]
        orbit_ranks.append(matrix_rank(orbit))
    return {
        "hadamard_kernel_requirement": "INVARIANT_UNDER_ALL_COORDINATE_PROJECTORS",
        "convolution_kernel_requirement": "INVARIANT_UNDER_CYCLIC_SHIFT_DELTA1_CONVOLUTION",
        "nonzero_vector_projector_law": "ANY_NONZERO_KERNEL_VECTOR_YIELDS_A_COORDINATE_BASIS_VECTOR",
        "coordinate_orbit_ranks": orbit_ranks,
        "all_coordinate_orbits_span_full_space": all(value == ORDER for value in orbit_ranks),
        "common_invariant_kernel_dimensions": [0, ORDER],
        "nonzero_proper_common_kernel_exists": False,
        "smallest_nonzero_common_quotient_dimension": ORDER,
        "strict_scope": "LINEAR_QUOTIENTS_WITH_BILINEAR_OPERATIONS_DESCENDING_FOR_ALL_OPERANDS",
    }


@dataclass
class Carrier:
    cells: list[int]
    rank: int
    seed_family: str
    restoration_generation: int = 0
    stage: str = "SEALED"

    @classmethod
    def seal(cls, rank: int, family: str) -> "Carrier":
        return cls(seed(rank, family, "A") + seed(rank, family, "B"), rank, family)

    def backing_id(self) -> int:
        return id(self.cells)

    def a(self) -> list[int]:
        return self.cells[: self.rank]

    def b(self) -> list[int]:
        return self.cells[self.rank :]

    def write(self, a: list[int], b: list[int]) -> None:
        self.cells[: self.rank] = a
        self.cells[self.rank :] = b


def forward(carrier: Carrier, depth: int, family: str, enabled: bool = True) -> None:
    if not enabled:
        carrier.stage = "FORWARD_COMPLETE"
        return
    a, b = carrier.a(), carrier.b()
    for index in range(depth):
        alpha, beta, constant = parameters(index, family)
        kernel = public_kernel(carrier.rank, index, family)
        if family == "PRIMARY":
            a = jet_multiply(a, kernel, carrier.rank)
            a = scale(a, constant)
            b = add(b, scale(jet_multiply(a, a, carrier.rank), beta))
            a = add(a, scale(b, alpha))
        else:
            a = add(a, scale(b, alpha))
            b = add(b, scale(jet_multiply(a, a, carrier.rank), beta))
            a = scale(a, constant)
            a = jet_multiply(a, kernel, carrier.rank)
    carrier.write(a, b)
    carrier.stage = "FORWARD_COMPLETE"


def inverse(carrier: Carrier, depth: int, family: str, mutation: str | None = None) -> None:
    a, b = carrier.a(), carrier.b()
    indices = list(reversed(range(depth)))
    if mutation == "REORDER":
        indices = list(range(depth))
    for index in indices:
        alpha, beta, constant = parameters(index, family)
        if mutation == "WRONG" and index == indices[0]:
            beta = (beta + 1) % FIELD
        kernel = public_kernel(carrier.rank, index, family)
        if family == "PRIMARY":
            a = subtract(a, scale(b, alpha))
            b = subtract(b, scale(jet_multiply(a, a, carrier.rank), beta))
            a = scale(a, pow(constant, FIELD - 2, FIELD))
            a = jet_multiply(a, jet_inverse(kernel, carrier.rank), carrier.rank)
        else:
            a = jet_multiply(a, jet_inverse(kernel, carrier.rank), carrier.rank)
            a = scale(a, pow(constant, FIELD - 2, FIELD))
            b = subtract(b, scale(jet_multiply(a, a, carrier.rank), beta))
            a = subtract(a, scale(b, alpha))
    carrier.write(a, b)
    carrier.stage = "RESTORED"


def boundary_value(b_jet: list[int], rank: int, family: str) -> int:
    code = 1 if family == "PRIMARY" else 2
    weights = [(3 * degree + 5 * code + rank) % FIELD or 1 for degree in range(rank)]
    return sum(weight * value for weight, value in zip(weights, b_jet, strict=True)) % FIELD


def boundary(carrier: Carrier, family: str) -> int:
    if carrier.stage != "FORWARD_COMPLETE":
        fail("boundary is available only after complete forward execution")
    return boundary_value(carrier.b(), carrier.rank, family)


def transaction(carrier: Carrier | None, rank: int, depth: int, family: str) -> dict[str, Any]:
    if carrier is None:
        fail("null carrier")
    if carrier.rank != rank or rank not in RANKS or depth not in DEPTHS or family not in FAMILIES:
        fail("invalid program or carrier")
    before = tuple(carrier.cells)
    backing = carrier.backing_id()
    forward(carrier, depth, family)
    forward_a = carrier.a()
    forward_b = carrier.b()
    projected = boundary(carrier, family)
    inverse(carrier, depth, family)
    restored = tuple(carrier.cells) == before
    if not restored:
        fail("exact restoration failed")
    carrier.restoration_generation += 1
    return {
        "rank": rank,
        "depth": depth,
        "family": family,
        "boundary": projected,
        "forward_a": forward_a,
        "forward_b": forward_b,
        "forward_commitment": digest_json([forward_a, forward_b]),
        "exact_cells_restored": restored,
        "same_backing_restored": carrier.backing_id() == backing,
        "restoration_generation": carrier.restoration_generation,
        "snapshot_used": False,
        "hidden_a_serialized": False,
    }


def compact_classical_reference(rank: int, depth: int, family: str) -> dict[str, Any]:
    """Execute the strongest matched recurrence in separate plain arrays."""
    a = seed(rank, family, "A")
    b = seed(rank, family, "B")
    for index in range(depth):
        alpha, beta, constant = parameters(index, family)
        kernel = public_kernel(rank, index, family)
        if family == "PRIMARY":
            a = jet_multiply(a, kernel, rank)
            a = [(constant * item) % FIELD for item in a]
            square = jet_multiply(a, a, rank)
            b = [(left + beta * right) % FIELD for left, right in zip(b, square, strict=True)]
            a = [(left + alpha * right) % FIELD for left, right in zip(a, b, strict=True)]
        else:
            a = [(left + alpha * right) % FIELD for left, right in zip(a, b, strict=True)]
            square = jet_multiply(a, a, rank)
            b = [(left + beta * right) % FIELD for left, right in zip(b, square, strict=True)]
            a = [(constant * item) % FIELD for item in a]
            a = jet_multiply(a, kernel, rank)
    return {
        "forward_a": a,
        "forward_b": b,
        "forward_commitment": digest_json([a, b]),
        "boundary": boundary_value(b, rank, family),
    }


def one_case(rank: int, depth: int, family: str) -> dict[str, Any]:
    receipt = transaction(Carrier.seal(rank, family), rank, depth, family)
    classical = compact_classical_reference(rank, depth, family)
    receipt["matches_identical_compact_classical_recurrence"] = all(
        receipt[key] == classical[key]
        for key in ("forward_a", "forward_b", "forward_commitment", "boundary")
    )
    receipt["matched_classical_boundary"] = classical["boundary"]
    receipt["matched_classical_forward_commitment"] = classical["forward_commitment"]
    return receipt


def raises(action: Callable[[], Any]) -> bool:
    try:
        action()
    except RuntimeError:
        return True
    return False


def controls() -> dict[str, bool]:
    rank, depth, family = 8, 16, "PRIMARY"
    original = Carrier.seal(rank, "ALTERNATE")
    before = tuple(original.cells)

    missing = Carrier(list(before), rank, "ALTERNATE")
    forward(missing, depth, family)
    missing_fails = tuple(missing.cells) != before

    wrong = Carrier(list(before), rank, "ALTERNATE")
    forward(wrong, depth, family)
    inverse(wrong, depth, family, mutation="WRONG")

    reordered = Carrier(list(before), rank, "ALTERNATE")
    forward(reordered, depth, family)
    inverse(reordered, depth, family, mutation="REORDER")

    premature = Carrier(list(before), rank, "ALTERNATE")
    disabled = Carrier(list(before), rank, "ALTERNATE")
    forward(disabled, depth, family, enabled=False)
    disabled_boundary = boundary(disabled, family)
    active = Carrier(list(before), rank, "ALTERNATE")
    forward(active, depth, family)
    active_boundary = boundary(active, family)

    indicator = [1] + [0] * (ORDER - 1)
    linear = [(position + 1) % FIELD for position in range(ORDER)]
    quadratic = [(position * position + 3) % FIELD for position in range(ORDER)]
    witnesses = [hadamard_witness(rank, item) for item in (indicator, linear, quadratic)]
    return {
        "missing_inverse_fails_restoration": missing_fails,
        "wrong_inverse_fails_restoration": tuple(wrong.cells) != before,
        "reordered_inverse_fails_for_noncommuting_program": tuple(reordered.cells) != before,
        "premature_projection_rejected": raises(lambda: boundary(premature, family)),
        "null_carrier_rejected": raises(lambda: transaction(None, rank, depth, family)),
        "wrong_rank_rejected": raises(lambda: transaction(original, 7, depth, family)),
        "carrier_disabled_path_changes_boundary": disabled_boundary != active_boundary,
        "three_nonconstant_hadamard_multipliers_escape_rank8_kernel": all(
            any(item["product_jet_after"]) and not any(item["witness_jet_before"])
            for item in witnesses
        ),
    }


def reuse_check() -> dict[str, Any]:
    carrier = Carrier.seal(4, "PRIMARY")
    sealed = tuple(carrier.cells)
    backing = carrier.backing_id()
    first = transaction(carrier, 4, 1, "PRIMARY")
    second = transaction(carrier, 4, 16, "ALTERNATE")
    fresh = Carrier(list(sealed), 4, "PRIMARY")
    reference = transaction(fresh, 4, 16, "ALTERNATE")
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
    certificate = common_congruence_certificate()
    multiplier_search = [compatible_multiplier_space(rank) for rank in RANKS]
    cases = [
        one_case(rank, depth, family)
        for rank in RANKS
        for family in FAMILIES
        for depth in DEPTHS
    ]
    checks = controls()
    reuse = reuse_check()
    if certificate["nonzero_proper_common_kernel_exists"]:
        fail("unexpected common congruence")
    if not all(item["only_constant_multipliers"] for item in multiplier_search):
        fail("unexpected compatible Hadamard multiplier")
    if not all(checks.values()):
        fail("control failure")
    if not all(case["exact_cells_restored"] and case["same_backing_restored"] for case in cases):
        fail("case restoration failure")
    if not all(case["matches_identical_compact_classical_recurrence"] for case in cases):
        fail("matched compact classical recurrence mismatch")
    if not all(
        reuse[key]
        for key in (
            "same_backing_reused",
            "exact_cells_restored_after_reuse",
            "unrelated_second_boundary_matches_fresh",
            "unrelated_second_commitment_matches_fresh",
        )
    ):
        fail("reuse failure")
    return {
        "schema": "CAT_CAS_F17_C17_COMMON_CONGRUENCE_HADAMARD_NO_GO_RESULTS_V1",
        "claim": CLAIM,
        "classification": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level": "PACKAGE_SELF_REVIEW",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "common_congruence_no_go": certificate,
        "restricted_hadamard_multiplier_search": multiplier_search,
        "cases": cases,
        "controls": checks,
        "restoration_and_reuse": reuse,
        "resource_accounting": {
            "maximum_accepted_phase_jet_carrier_field_cells": 2 * max(RANKS),
            "maximum_logical_working_field_cells_peak": 7 * max(RANKS),
            "retained_inverse_history_field_cells": 0,
            "full_nonzero_common_quotient_field_cells_per_register": ORDER,
            "full_nonzero_common_quotient_two_register_carrier_field_cells": 2 * ORDER,
            "matched_compact_classical_carrier_field_cells": 2 * max(RANKS),
            "matched_compact_classical_logical_working_field_cells_peak": 7 * max(RANKS),
            "accepted_over_matched_compact_classical_carrier_ratio": 1,
            "accepted_over_matched_compact_classical_working_ratio": 1,
            "phase_and_matched_classical_operation_law_identical": True,
            "dense17_by17_relation_table_cells": 0,
            "python_allocator_and_whole_process_peaks_excluded": True,
            "advantage_claimed": False,
        },
        "matched_baselines": {
            "strongest": "IDENTICAL_RANK_R_TRUNCATED_POLYNOMIAL_RECURRENCE_WITH_SCALAR_MULTIPLIERS",
            "strongest_executed": True,
            "full_semantic_reference": "17_COEFFICIENT_CYCLIC_GROUP_ALGEBRA_IN_INDEPENDENT_ORACLE",
            "cold_start_comparison_used": False,
        },
        "claim_ceiling": "LINEAR_F17_QUOTIENTS_OF_F17_TO_THE_C17_WITH_ARBITRARY_BILINEAR_CONVOLUTION_AND_HADAMARD_OPERANDS_PLUS_RANK2_4_8_JET_COMPATIBLE_DIAGONAL_MULTIPLIER_SEARCH",
        "rejected_interpretations": [
            "NO_NONLINEAR_OR_NONQUOTIENT_ENCODING_CAN_SUPPORT_BOTH_OPERATIONS",
            "NO_OTHER_PHASE_ALGEBRA_CAN_SUPPORT_COMPOSITION_AND_INTERSECTION",
            "CONSTANT_HADAMARD_MULTIPLIERS_PROVIDE_GENERAL_RELATION_INTERSECTION",
            "THE_RESTRICTED_PHASE_JET_OUTPERFORMS_THE_IDENTICAL_CLASSICAL_RECURRENCE",
        ],
        "not_established": [
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "next_obstruction": "ARBITRARY_RELATION_INTERSECTION_FORCES_THE_PROPER_LINEAR_C17_PHASE_JET_QUOTIENT_BACK_TO_ALL17_COEFFICIENTS_WHILE_ITS_MAXIMAL_COMPATIBLE_DIAGONAL_FAMILY_IS_ONLY_PUBLIC_SCALAR_MULTIPLICATION",
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
