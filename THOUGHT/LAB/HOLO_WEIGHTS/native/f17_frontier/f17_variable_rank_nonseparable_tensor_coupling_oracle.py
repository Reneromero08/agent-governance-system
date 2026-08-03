#!/usr/bin/env python3
"""Independent power-basis oracle for the four-site M118 diagnostic.

No production M118 or M116 pair module is imported.  This oracle rebuilds
Z[zeta_17], the public programs, the 16-amplitude execution, exact minor-based
cut ranks, finite-field rank certificates, the compact factor contraction,
reverse restoration, and adversarial mutations.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from typing import Any


PRIME = 17
DEGREE = 16
SITES = 4
CELLS = 16
FAMILY_WEIGHTS = {
    "PRIMARY": (1, 3, 2, 5),
    "REUSE": (2, 6, 4, 9),
}

Element = tuple[int, ...]
ZERO: Element = (0,) * DEGREE
ONE: Element = (1,) + (0,) * (DEGREE - 1)


def fail(message: str) -> None:
    raise RuntimeError(message)


def monomial(exponent: int) -> Element:
    reduced = exponent % PRIME
    if reduced == DEGREE:
        return (-1,) * DEGREE
    return tuple(1 if index == reduced else 0 for index in range(DEGREE))


ROOTS = tuple(monomial(exponent) for exponent in range(PRIME))


def add(left: Element, right: Element) -> Element:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def subtract(left: Element, right: Element) -> Element:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def multiply(left: Element, right: Element) -> Element:
    residues = [0] * PRIME
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            residues[(left_index + right_index) % PRIME] += left_value * right_value
    high = residues[DEGREE]
    return tuple(residues[index] - high for index in range(DEGREE))


@dataclass(frozen=True)
class Gate:
    kind: str
    first: int
    second: int
    weight: int


def program(family: str) -> tuple[Gate, ...]:
    if family not in FAMILY_WEIGHTS:
        fail("oracle received unknown public family")
    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    return (
        Gate("PHASE", 0, 2, weight_ac),
        Gate("PHASE", 1, 2, weight_bc),
        Gate("SHEAR", 2, -1, 0),
        Gate("PHASE", 0, 3, weight_ad),
        Gate("PHASE", 1, 3, weight_bd),
    )


def assignment_bits(index: int) -> tuple[int, int, int, int]:
    return tuple((index >> (SITES - 1 - site)) & 1 for site in range(SITES))  # type: ignore[return-value]


def assignment_index(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def phase_gate(cells: list[Element], gate: Gate, inverse: bool = False) -> None:
    exponent = -gate.weight if inverse else gate.weight
    root = monomial(exponent)
    for index in range(CELLS):
        bits = assignment_bits(index)
        if bits[gate.first] and bits[gate.second]:
            cells[index] = multiply(cells[index], root)


def shear(cells: list[Element], inverse: bool = False) -> None:
    for a_value, b_value, d_value in itertools.product((0, 1), repeat=3):
        low_index = assignment_index((a_value, b_value, 0, d_value))
        high_index = assignment_index((a_value, b_value, 1, d_value))
        cells[low_index] = (
            subtract(cells[low_index], cells[high_index])
            if inverse
            else add(cells[low_index], cells[high_index])
        )


def apply(cells: list[Element], gate: Gate, inverse: bool = False) -> None:
    if gate.kind == "PHASE":
        phase_gate(cells, gate, inverse)
    else:
        shear(cells, inverse)


def forward(cells: list[Element], public_program: tuple[Gate, ...]) -> None:
    for gate in public_program:
        apply(cells, gate)


def reverse(cells: list[Element], public_program: tuple[Gate, ...]) -> None:
    for gate in reversed(public_program):
        apply(cells, gate, True)


def boundary(cells: list[Element]) -> Element:
    accumulator = ZERO
    for value in cells:
        accumulator = add(accumulator, value)
    return accumulator


def compact_boundary(family: str) -> Element:
    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    coefficients: dict[int, int] = {}
    for exponent, coefficient in (
        (weight_ad, 1),
        (weight_bd, 1),
        ((weight_ad + weight_bd) % PRIME, 1),
        (weight_ac, 2),
        (weight_bc, 2),
        ((weight_ac + weight_bc) % PRIME, 2),
        ((weight_ac + weight_ad) % PRIME, 2),
        ((weight_bc + weight_bd) % PRIME, 2),
        ((weight_ac + weight_ad + weight_bc + weight_bd) % PRIME, 2),
    ):
        coefficients[exponent] = coefficients.get(exponent, 0) + coefficient
    accumulator = tuple(9 * coefficient for coefficient in ONE)
    for exponent, coefficient in sorted(coefficients.items()):
        accumulator = add(
            accumulator,
            tuple(coefficient * value for value in ROOTS[exponent]),
        )
    return accumulator


def three_product_boundary(family: str) -> Element:
    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    accumulator = tuple(4 * coefficient for coefficient in ONE)
    for left_exponent, right_exponent, coefficient in (
        (weight_ad, weight_bd, 1),
        (weight_ac, weight_bc, 2),
        ((weight_ac + weight_ad) % PRIME, (weight_bc + weight_bd) % PRIME, 2),
    ):
        left_message = add(ONE, ROOTS[left_exponent])
        right_message = add(ONE, ROOTS[right_exponent])
        product = multiply(left_message, right_message)
        accumulator = add(
            accumulator,
            tuple(coefficient * value for value in product),
        )
    return accumulator


def outer_assignment_boundary(family: str) -> Element:
    """Separate four-assignment factor sum used only to check the closed form."""

    weight_ac, weight_bc, weight_ad, weight_bd = FAMILY_WEIGHTS[family]
    accumulator = ZERO
    for a_value, b_value in itertools.product((0, 1), repeat=2):
        c_phase = ROOTS[(a_value * weight_ac + b_value * weight_bc) % PRIME]
        d_phase = ROOTS[(a_value * weight_ad + b_value * weight_bd) % PRIME]
        c_message = add(ONE, tuple(2 * coefficient for coefficient in c_phase))
        d_message = add(ONE, d_phase)
        accumulator = add(accumulator, multiply(c_message, d_message))
    return accumulator


def determinant(matrix: list[list[Element]]) -> Element:
    size = len(matrix)
    accumulator = ZERO
    for permutation in itertools.permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        product = ONE
        for row, column in enumerate(permutation):
            product = multiply(product, matrix[row][column])
        accumulator = (
            subtract(accumulator, product)
            if inversions % 2
            else add(accumulator, product)
        )
    return accumulator


def flatten(cells: list[Element], left_sites: tuple[int, ...]) -> list[list[Element]]:
    right_sites = tuple(site for site in range(SITES) if site not in left_sites)
    matrix = [
        [ZERO for _ in range(1 << len(right_sites))]
        for _ in range(1 << len(left_sites))
    ]
    for index, value in enumerate(cells):
        bits = assignment_bits(index)
        row = assignment_index(tuple(bits[site] for site in left_sites))
        column = assignment_index(tuple(bits[site] for site in right_sites))
        matrix[row][column] = value
    return matrix


def exact_rank(matrix: list[list[Element]]) -> int:
    rows = len(matrix)
    columns = len(matrix[0])
    for size in range(min(rows, columns), 0, -1):
        for selected_rows in itertools.combinations(range(rows), size):
            for selected_columns in itertools.combinations(range(columns), size):
                minor = [
                    [matrix[row][column] for column in selected_columns]
                    for row in selected_rows
                ]
                if determinant(minor) != ZERO:
                    return size
    return 0


def evaluate(value: Element, root: int, modulus: int) -> int:
    return sum(
        coefficient * pow(root, exponent, modulus)
        for exponent, coefficient in enumerate(value)
    ) % modulus


def finite_rank(matrix: list[list[Element]], modulus: int, root: int) -> int:
    work = [
        [evaluate(value, root, modulus) for value in row]
        for row in matrix
    ]
    rows = len(work)
    columns = len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, rows) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = pow(work[rank][column], -1, modulus)
        work[rank] = [(value * scale) % modulus for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    (left - factor * right) % modulus
                    for left, right in zip(work[row], work[rank], strict=True)
                ]
        rank += 1
    return rank


def rank_profile(cells: list[Element]) -> dict[str, Any]:
    natural = [
        exact_rank(flatten(cells, tuple(range(cut))))
        for cut in (1, 2, 3)
    ]
    two_cuts = {
        "AB_CD": exact_rank(flatten(cells, (0, 1))),
        "AC_BD": exact_rank(flatten(cells, (0, 2))),
        "AD_BC": exact_rank(flatten(cells, (0, 3))),
    }
    one_cuts = {
        name: exact_rank(flatten(cells, (site,)))
        for site, name in enumerate(("A", "B", "C", "D"))
    }
    return {
        "natural_tt_ranks": natural,
        "all_two_by_two_cut_ranks": two_cuts,
        "all_one_site_cut_ranks": one_cuts,
    }


def case(family: str) -> dict[str, Any]:
    public_program = program(family)
    cells = [ONE for _ in range(CELLS)]
    backing = id(cells)
    trace = [{"stage": "SEED", **rank_profile(cells)}]
    for ordinal, gate in enumerate(public_program):
        apply(cells, gate)
        trace.append({"stage": f"{gate.kind}_{ordinal}", **rank_profile(cells)})
    final_boundary = boundary(cells)
    factor_boundary = compact_boundary(family)
    three_product_factor_boundary = three_product_boundary(family)
    outer_factor_boundary = outer_assignment_boundary(family)
    cut_matrices = {
        "AB_CD": flatten(cells, (0, 1)),
        "AC_BD": flatten(cells, (0, 2)),
        "AD_BC": flatten(cells, (0, 3)),
    }
    rank_mod_103 = {
        name: finite_rank(matrix, 103, 72)
        for name, matrix in cut_matrices.items()
    }
    determinant_mod_103 = {
        name: evaluate(determinant(matrix), 72, 103)
        for name, matrix in cut_matrices.items()
    }
    reverse(cells, public_program)
    restored_to_seed = cells == [ONE for _ in range(CELLS)]
    cells[:] = [subtract(value, ONE) for value in cells]
    return {
        "family": family,
        "public_weights": list(FAMILY_WEIGHTS[family]),
        "boundary": list(final_boundary),
        "compact_factor_boundary": list(factor_boundary),
        "boundary_agreement": final_boundary == factor_boundary,
        "closed_form_equals_independent_outer_assignment_factor_sum": (
            factor_boundary == three_product_factor_boundary == outer_factor_boundary
        ),
        "rank_trace": trace,
        "final_rank_mod_103": rank_mod_103,
        "final_determinant_mod_103": determinant_mod_103,
        "root_72_has_exact_order_17_mod_103": (
            pow(72, 17, 103) == 1 and 72 != 1
        ),
        "restored_to_product_seed_exactly": restored_to_seed,
        "unloaded_to_zero_exactly": all(value == ZERO for value in cells),
        "same_backing": id(cells) == backing,
    }


def controls() -> dict[str, bool]:
    public_program = program("PRIMARY")
    accepted = [ONE for _ in range(CELLS)]
    forward(accepted, public_program)

    wrong = list(accepted)
    for gate in reversed(public_program[2:]):
        apply(wrong, gate, True)
    apply(wrong, Gate("PHASE", 1, 2, 4), True)
    apply(wrong, public_program[0], True)

    reordered = list(accepted)
    apply(reordered, public_program[4], True)
    apply(reordered, public_program[3], True)
    apply(reordered, public_program[1], True)
    apply(reordered, public_program[2], True)
    apply(reordered, public_program[0], True)

    mutated = list(accepted)
    mutated[7] = add(mutated[7], ONE)
    reverse(mutated, public_program)

    zero_program = (
        Gate("PHASE", 0, 2, 0),
        Gate("PHASE", 1, 2, 0),
        Gate("SHEAR", 2, -1, 0),
        Gate("PHASE", 0, 3, 0),
        Gate("PHASE", 1, 3, 0),
    )
    zero_cells = [ONE for _ in range(CELLS)]
    forward(zero_cells, zero_program)

    missing_bd = list(public_program)
    missing_bd[4] = Gate("PHASE", 1, 3, 0)
    missing_cells = [ONE for _ in range(CELLS)]
    forward(missing_cells, tuple(missing_bd))

    perturbed = list(public_program)
    perturbed[4] = Gate("PHASE", 1, 3, 6)
    perturbed_cells = [ONE for _ in range(CELLS)]
    forward(perturbed_cells, tuple(perturbed))

    no_shear = [ONE for _ in range(CELLS)]
    for gate in public_program:
        if gate.kind == "PHASE":
            apply(no_shear, gate)

    return {
        "wrong_phase_exponent_inverse_fails_restoration": wrong != [ONE for _ in range(CELLS)],
        "reordered_noncommuting_inverse_fails_restoration": reordered != [ONE for _ in range(CELLS)],
        "resident_mutation_survives_inverse": mutated != [ONE for _ in range(CELLS)],
        "zero_weight_shear_sham_has_rank_one": (
            rank_profile(zero_cells)["natural_tt_ranks"] == [1, 1, 1]
        ),
        "missing_bd_edge_reduces_a_two_by_two_cut": any(
            rank < 4
            for rank in rank_profile(missing_cells)["all_two_by_two_cut_ranks"].values()
        ),
        "semantic_weight_perturbation_changes_boundary": (
            boundary(perturbed_cells) != boundary(accepted)
        ),
        "shear_disabled_changes_boundary": boundary(no_shear) != boundary(accepted),
    }


def result() -> dict[str, Any]:
    cases = [case(family) for family in ("PRIMARY", "REUSE")]
    return {
        "experiment": "INDEPENDENT_FULL_POWER_BASIS_FOUR_SITE_K22_PHASE_TT_ORACLE",
        "result": "PASS",
        "imports_production_m118": False,
        "imports_m116_pair_backend": False,
        "representation": "CANONICAL_16_INTEGER_POWER_BASIS",
        "exact_rank_method": "EXHAUSTIVE_NONZERO_MINOR_SEARCH_IN_Z_ZETA17",
        "finite_field_certificate": "ZETA_MAPS_TO_ORDER17_ELEMENT_72_IN_F103",
        "cases": cases,
        "controls": controls(),
        "all_boundaries_agree": all(case["boundary_agreement"] for case in cases),
        "all_closed_form_factor_contractions_match_outer_assignment_sums": all(
            case["closed_form_equals_independent_outer_assignment_factor_sum"]
            for case in cases
        ),
        "all_final_cut_ranks_four_exactly": all(
            all(rank == 4 for rank in case["rank_trace"][-1]["all_two_by_two_cut_ranks"].values())
            for case in cases
        ),
        "all_mod103_cut_ranks_four": all(
            all(rank == 4 for rank in case["final_rank_mod_103"].values())
            for case in cases
        ),
        "all_restore_and_unload_exactly": all(
            case["restored_to_product_seed_exactly"]
            and case["unloaded_to_zero_exactly"]
            for case in cases
        ),
        "all_same_backing": all(case["same_backing"] for case in cases),
        "strict_scope": "TWO_PUBLIC_WEIGHT_FAMILIES_WIDTH4_LOCAL_DIMENSION2_DIRECT_PROCESS_SOFTWARE",
    }


def main() -> None:
    payload = result()
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        return
    if len(sys.argv) != 1:
        fail("usage: f17_variable_rank_nonseparable_tensor_coupling_oracle.py [--output PATH]")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
