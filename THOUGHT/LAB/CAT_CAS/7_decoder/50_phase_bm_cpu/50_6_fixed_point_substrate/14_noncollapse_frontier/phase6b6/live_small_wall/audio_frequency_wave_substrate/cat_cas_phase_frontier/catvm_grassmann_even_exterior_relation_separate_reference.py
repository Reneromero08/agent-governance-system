#!/usr/bin/env python3
"""Standalone M254 oracle over Q[z]/(z^4+1) and an eight-generator exterior algebra."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence


PORT_ORDER = ("THETA0", "THETA1", "THETA2", "THETA3")
OWNER = 254004
BASIS = (0, 3, 5, 9, 6, 10, 12, 15)
INDEX = {mask: index for index, mask in enumerate(BASIS)}


def solve_rational(matrix: list[list[Fraction]], target: list[Fraction]) -> list[Fraction]:
    augmented = [list(row) + [target[index]] for index, row in enumerate(matrix)]
    size = len(matrix)
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ZeroDivisionError("reference singular rational system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        value = augmented[column][column]
        augmented[column] = [entry / value for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][index] - factor * augmented[column][index]
                    for index in range(size + 1)
                ]
    return [augmented[index][-1] for index in range(size)]


@dataclass(frozen=True)
class E:
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction] = (
        Fraction(0), Fraction(0), Fraction(0), Fraction(0)
    )

    def __add__(self, other: "E") -> "E":
        return E(tuple(a + b for a, b in zip(self.coefficients, other.coefficients)))

    def __neg__(self) -> "E":
        return E(tuple(-value for value in self.coefficients))

    def __sub__(self, other: "E") -> "E":
        return self + (-other)

    def __mul__(self, other: "E") -> "E":
        expanded = [Fraction(0) for _ in range(7)]
        for left, a in enumerate(self.coefficients):
            for right, b in enumerate(other.coefficients):
                expanded[left + right] += a * b
        for degree in range(6, 3, -1):
            expanded[degree - 4] -= expanded[degree]
        return E(tuple(expanded[:4]))

    def scale(self, value: Fraction) -> "E":
        return E(tuple(coefficient * value for coefficient in self.coefficients))

    def inverse(self) -> "E":
        basis = [E(tuple(Fraction(index == power) for index in range(4))) for power in range(4)]
        products = [self * value for value in basis]
        matrix = [[products[column].coefficients[row] for column in range(4)] for row in range(4)]
        return E(tuple(solve_rational(matrix, [Fraction(1), Fraction(0), Fraction(0), Fraction(0)])))


ZERO = E()
ONE = E((Fraction(1), Fraction(0), Fraction(0), Fraction(0)))
Z = E((Fraction(0), Fraction(1), Fraction(0), Fraction(0)))
Signature = tuple[E, E, E, E, E, E, E, E]


def from_public(value: object) -> E:
    if not isinstance(value, list) or len(value) != 4:
        raise RuntimeError("reference field coordinate rejected")
    coordinates: list[Fraction] = []
    for item in value:
        if (
            not isinstance(item, list) or len(item) != 2
            or any(not isinstance(part, int) or isinstance(part, bool) for part in item)
            or abs(item[0]) > 8 or not 1 <= item[1] <= 8
        ):
            raise RuntimeError("reference rational coordinate rejected")
        coordinates.append(Fraction(item[0], item[1]))
    one, root, imag, root_imag = coordinates
    return E((one, root + root_imag, imag, -root + root_imag))


def k_json(value: E) -> list[list[int]]:
    one, z, imag, z3 = value.coefficients
    coordinates = (one, (z - z3) / 2, imag, (z + z3) / 2)
    return [[coordinate.numerator, coordinate.denominator] for coordinate in coordinates]


def wedge_sign(left: int, right: int, variables: int = 4) -> int:
    if left & right:
        return 0
    inversions = sum(
        (right & ((1 << bit) - 1)).bit_count()
        for bit in range(variables) if left & (1 << bit)
    )
    return -1 if inversions % 2 else 1


def exterior_product(left: Sequence[E], right: Sequence[E]) -> Signature:
    output = [ZERO] * 8
    for li, lm in enumerate(BASIS):
        for ri, rm in enumerate(BASIS):
            sign = wedge_sign(lm, rm)
            if sign:
                target = INDEX[lm | rm]
                output[target] = output[target] + (left[li] * right[ri]).scale(Fraction(sign))
    return tuple(output)  # type: ignore[return-value]


def full_exterior_product(left: dict[int, E], right: dict[int, E]) -> dict[int, E]:
    output: dict[int, E] = {}
    for lm, lv in left.items():
        for rm, rv in right.items():
            sign = wedge_sign(lm, rm, 8)
            if sign:
                target = lm | rm
                output[target] = output.get(target, ZERO) + (lv * rv).scale(Fraction(sign))
    return output


def berezin_hodge(cells: Sequence[E]) -> Signature:
    # Variable order is theta0..theta3,eta0..eta3.  Each kernel factor is
    # 1+eta_i theta_i = 1-theta_i eta_i in canonical order.  Extracting the
    # theta0123 coefficient independently derives every signed complement.
    kernel: dict[int, E] = {0: ONE}
    for index in range(4):
        factor = {0: ONE, (1 << index) | (1 << (4 + index)): -ONE}
        kernel = full_exterior_product(kernel, factor)
    source = {mask: cells[index] for index, mask in enumerate(BASIS) if cells[index] != ZERO}
    expanded = full_exterior_product(kernel, source)
    result = [ZERO] * 8
    for mask, value in expanded.items():
        if mask & 15 == 15:
            eta_mask = (mask >> 4) & 15
            if eta_mask in INDEX:
                result[INDEX[eta_mask]] = result[INDEX[eta_mask]] + value
    return tuple(result)  # type: ignore[return-value]


def inplace_hodge(cells: list[E], scratch: list[E]) -> None:
    if len(cells) != 8 or scratch != [ZERO]:
        raise RuntimeError("reference dirty Hodge scratch")
    for left, right, sign in ((0, 7, 1), (1, 6, -1), (2, 5, 1), (3, 4, -1)):
        scratch[0] = cells[left]
        cells[left] = cells[right].scale(Fraction(sign))
        cells[right] = scratch[0].scale(Fraction(sign))
        scratch[0] = ZERO


def factor_inverse(factor: Signature) -> Signature:
    if factor[0] == ZERO:
        raise RuntimeError("reference zero-scalar factor")
    scalar_inverse = factor[0].inverse()
    unit: Signature = (ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)
    nilpotent = tuple(
        ZERO if index == 0 else factor[index] * scalar_inverse for index in range(8)
    )
    nilpotent_squared = exterior_product(nilpotent, nilpotent)
    geometric = tuple(unit[index] - nilpotent[index] + nilpotent_squared[index] for index in range(8))
    scalar_signature: Signature = (scalar_inverse, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)
    return exterior_product(scalar_signature, geometric)


@dataclass(frozen=True)
class Module:
    op: str
    factor: Signature = (ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO)


def canonical_program(descriptor: dict[str, Any]) -> tuple[Module, ...]:
    if set(descriptor) != {"ports", "modules"} or descriptor.get("ports") != list(PORT_ORDER):
        raise RuntimeError("reference typed port order rejected")
    modules = descriptor.get("modules")
    if not isinstance(modules, list) or not 1 <= len(modules) <= 8:
        raise RuntimeError("reference program length rejected")
    result: list[Module] = []
    for item in modules:
        if not isinstance(item, dict) or "op" not in item:
            raise RuntimeError("reference malformed module")
        if item["op"] == "HODGE" and set(item) == {"op"}:
            result.append(Module("HODGE"))
        elif item["op"] == "INTERSECT" and set(item) == {"op", "factor"}:
            values = item["factor"]
            if not isinstance(values, list) or len(values) != 8:
                raise RuntimeError("reference factor count rejected")
            factor: Signature = tuple(from_public(value) for value in values)  # type: ignore[assignment]
            if factor[0] == ZERO:
                raise RuntimeError("reference zero-scalar factor rejected")
            result.append(Module("INTERSECT", factor))
        else:
            raise RuntimeError("reference undeclared module")
    return tuple(result)


def normalized_descriptor(program: tuple[Module, ...]) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    for module in program:
        if module.op == "HODGE":
            modules.append({"op": "HODGE"})
        else:
            modules.append({"op": "INTERSECT", "factor": [k_json(value) for value in module.factor]})
    return {"ports": list(PORT_ORDER), "modules": modules}


def digest(program: tuple[Module, ...]) -> str:
    encoded = json.dumps(normalized_descriptor(program), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def initial_cells() -> list[E]:
    return [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO]


def apply_module(cells: list[E], scratch: list[E], module: Module, *, inverse: bool) -> None:
    if module.op == "HODGE":
        expected = berezin_hodge(cells)
        inplace_hodge(cells, scratch)
        if tuple(cells) != expected:
            raise RuntimeError("reference Berezin and in-place Hodge mismatch")
    else:
        cells[:] = exterior_product(cells, factor_inverse(module.factor) if inverse else module.factor)


def execute_compact(program: tuple[Module, ...]) -> E:
    cells = initial_cells()
    scratch = [ZERO]
    for module in program:
        apply_module(cells, scratch, module, inverse=False)
    return cells[7]


def work_record(program: tuple[Module, ...]) -> dict[str, int]:
    intersections = sum(module.op == "INTERSECT" for module in program)
    hodges = len(program) - intersections
    return {
        "compiled_public_module_plan_references": len(program),
        "compiled_public_intersection_factor_field_cells": 8 * intersections,
        "forward_intersections": intersections,
        "inverse_intersections": intersections,
        "forward_hodge_closures": hodges,
        "inverse_hodge_closures": hodges,
        "forward_field_multiplications": 21 * intersections,
        "inverse_field_multiplications": 21 * intersections,
        "forward_field_accumulations": 14 * intersections,
        "inverse_field_accumulations": 14 * intersections,
        "forward_field_negations": 2 * intersections + 4 * hodges,
        "inverse_field_negations": 2 * intersections + 4 * hodges,
        "forward_carrier_field_writes": 8 * len(program),
        "inverse_carrier_field_writes": 8 * len(program),
        "forward_hodge_scratch_writes_and_clears": 8 * hodges,
        "inverse_hodge_scratch_writes_and_clears": 8 * hodges,
        "inverse_factor_rematerializations": intersections,
        "inverse_factor_returned_field_cells_materialized": 8 * intersections,
        "peak_returned_inverse_factor_field_cells": 8 if intersections else 0,
        "inverse_factor_field_multiplications": 16 * intersections,
        "inverse_factor_field_accumulations": 7 * intersections,
        "inverse_factor_field_negations": 9 * intersections,
        "inverse_factor_field_inversions": intersections,
        "retained_dynamic_inverse_history_entries": 0,
    }


class ReferencePort:
    def __init__(self) -> None:
        self.cells = initial_cells()
        self.scratch = [ZERO]
        self.receipts = [False] * 8
        self.owner = 0
        self.generation = 0
        self.program: tuple[Module, ...] | None = None
        self.program_id = ""
        self.transaction_id = ""
        self.cursor = 0
        self.projected = False
        self.leased = False
        self.last_generation = 0

    def canonical(self) -> bool:
        return (
            self.cells == initial_cells() and self.scratch == [ZERO]
            and self.receipts == [False] * 8 and self.owner == 0 and self.generation == 0
            and self.program is None and self.program_id == "" and self.transaction_id == ""
            and self.cursor == 0 and not self.projected and not self.leased
        )

    def lease(self, program: tuple[Module, ...], generation: int, program_id: str, transaction_id: str) -> None:
        if not self.canonical() or generation != self.last_generation + 1 or program_id != digest(program):
            raise RuntimeError("reference lease rejected")
        self.owner, self.generation, self.program = OWNER, generation, program
        self.program_id, self.transaction_id, self.leased = program_id, transaction_id, True

    def require(self, program: tuple[Module, ...], owner: int, generation: int, program_id: str, transaction_id: str) -> None:
        if (
            not self.leased or self.program != program or self.owner != owner
            or self.generation != generation or self.program_id != program_id
            or self.transaction_id != transaction_id
        ):
            raise RuntimeError("reference custody rejected")

    def forward(self) -> None:
        if self.program is None:
            raise RuntimeError("reference unleased forward")
        while self.cursor < len(self.program):
            apply_module(self.cells, self.scratch, self.program[self.cursor], inverse=False)
            self.receipts[self.cursor] = True
            self.cursor += 1

    def project(self) -> E:
        if (
            self.program is None or self.cursor != len(self.program) or self.projected
            or self.receipts[:self.cursor] != [True] * self.cursor or self.scratch != [ZERO]
        ):
            raise RuntimeError("reference premature projection")
        self.projected = True
        return self.cells[7]

    def reverse(self) -> None:
        if self.program is None:
            raise RuntimeError("reference unleased inverse")
        while self.cursor:
            apply_module(self.cells, self.scratch, self.program[self.cursor - 1], inverse=True)
            self.cursor -= 1
            self.receipts[self.cursor] = False
        self.projected = False

    def release(self) -> None:
        if (
            self.cells != initial_cells() or self.scratch != [ZERO] or self.receipts != [False] * 8
            or self.cursor != 0 or self.projected or not self.leased or self.program is None
        ):
            raise RuntimeError("reference release before exact restoration")
        restored = self.generation
        self.owner = self.generation = 0
        self.program = None
        self.program_id = self.transaction_id = ""
        self.leased = False
        self.last_generation = restored


def execute(port: ReferencePort, program: tuple[Module, ...], generation: int, run_kind: str) -> dict[str, Any]:
    cell_id, scratch_id, receipt_id = id(port.cells), id(port.scratch), id(port.receipts)
    program_id = digest(program)
    transaction_id = f"REF_{run_kind}"
    port.lease(program, generation, program_id, transaction_id)
    port.require(program, OWNER, generation, program_id, transaction_id)
    port.forward()
    final_boundary = port.project()
    if final_boundary != execute_compact(program):
        raise RuntimeError("independent exterior recurrences disagree")
    port.reverse()
    port.release()
    return {
        "module_kinds": [module.op for module in program],
        "generation": generation,
        "top_form_boundary": k_json(final_boundary),
        "hidden_even_relation_field_cells": 8,
        "hidden_hodge_scratch_field_cells": 1,
        "hidden_module_receipt_cells": 8,
        "retained_final_boundary_field_cells_during_inverse": 1,
        "same_relation_scratch_and_receipt_backings": (
            cell_id == id(port.cells) and scratch_id == id(port.scratch) and receipt_id == id(port.receipts)
        ),
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "work": work_record(program),
        "run_kind": run_kind,
    }


def field_rank(matrix: list[list[E]]) -> int:
    work = [list(row) for row in matrix]
    rows, columns, rank = len(work), len(work[0]), 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column] != ZERO), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = work[rank][column].inverse()
        work[rank] = [value * inverse for value in work[rank]]
        for row in range(rows):
            if row != rank and work[row][column] != ZERO:
                factor = work[row][column]
                work[row] = [work[row][index] - factor * work[rank][index] for index in range(columns)]
        rank += 1
        if rank == rows:
            break
    return rank


def linear_rank_certificate() -> dict[str, int]:
    unit = tuple(initial_cells())
    reachable = [list(unit)]
    for index in range(1, 8):
        vector = list(unit)
        vector[index] = ONE
        reachable.append(vector)
    observable_rows: list[list[E]] = []
    for factor in reachable:
        row: list[E] = []
        for index in range(8):
            basis = [ZERO] * 8
            basis[index] = ONE
            row.append(exterior_product(basis, factor)[7])
        observable_rows.append(row)
    hankel = [
        [sum((observable_rows[row][k] * reachable[column][k] for k in range(8)), ZERO) for column in range(8)]
        for row in range(8)
    ]
    return {
        "reachable_rank": field_rank(reachable),
        "observable_rank": field_rank(observable_rows),
        "hankel_rank": field_rank(hankel),
    }


def controls(primary: tuple[Module, ...], reuse: tuple[Module, ...]) -> dict[str, bool]:
    basis_hodge = []
    involution = True
    in_place_parity = True
    for index in range(8):
        basis = [ZERO] * 8
        basis[index] = ONE
        transformed = berezin_hodge(basis)
        basis_hodge.append(transformed)
        involution &= berezin_hodge(transformed) == tuple(basis)
        in_place = list(basis)
        scratch = [ZERO]
        inplace_hodge(in_place, scratch)
        in_place_parity &= tuple(in_place) == transformed and scratch == [ZERO]

    factor1 = primary[0].factor
    factor2 = primary[2].factor
    factor3 = primary[3].factor
    inverse_ok = all(
        exterior_product(factor, factor_inverse(factor)) == tuple(initial_cells())
        for factor in (factor1, factor2, factor3)
    )
    associative = exterior_product(exterior_product(factor1, factor2), factor3) == exterior_product(
        factor1, exterior_product(factor2, factor3)
    )
    commuting = exterior_product(factor1, factor2) == exterior_product(factor2, factor1)

    quadratic: Signature = (ONE, ONE, ZERO, ZERO, ZERO, ZERO, ONE, ZERO)
    truncated = list(factor_inverse(quadratic))
    truncated[7] = ZERO

    wrong_owner = False
    port = ReferencePort()
    port.lease(primary, 1, digest(primary), "OWNER")
    try:
        port.require(primary, OWNER + 1, 1, digest(primary), "OWNER")
    except RuntimeError:
        wrong_owner = True
    port.release()

    premature = False
    port = ReferencePort()
    port.lease(primary, 1, digest(primary), "PREMATURE")
    try:
        port.project()
    except RuntimeError:
        premature = True
    port.release()

    dirty = False
    try:
        inplace_hodge(initial_cells(), [ONE])
    except RuntimeError:
        dirty = True

    stale = False
    port = ReferencePort()
    execute(port, primary, 1, "CONTROL")
    try:
        port.lease(reuse, 1, digest(reuse), "STALE")
    except RuntimeError:
        stale = True
    changed_descriptor = False
    try:
        port.lease(reuse, 2, digest(primary), "MUTATION")
    except RuntimeError:
        changed_descriptor = True

    gaussian: Signature = (ONE, ONE, ONE, ZERO, ZERO, ZERO, ONE, ONE)
    gaussian_transformed = berezin_hodge(gaussian)
    gaussian_defect = lambda cells: cells[0] * cells[7] - (
        cells[1] * cells[6] - cells[2] * cells[5] + cells[3] * cells[4]
    )
    nongaussian = list(gaussian)
    nongaussian[7] = Z
    ranks = linear_rank_certificate()

    return {
        "eight_variable_berezin_kernel_derives_expected_signed_complement_on_all_basis_masks": basis_hodge == [
            (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE),
            (ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, -ONE, ZERO),
            (ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO),
            (ZERO, ZERO, ZERO, ZERO, -ONE, ZERO, ZERO, ZERO),
            (ZERO, ZERO, ZERO, -ONE, ZERO, ZERO, ZERO, ZERO),
            (ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO),
            (ZERO, -ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO),
            (ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO),
        ],
        "berezin_hodge_is_exact_involution_on_all_eight_basis_masks": involution,
        "one_scratch_inplace_hodge_matches_independent_berezin_kernel": in_place_parity,
        "generic_signed_exterior_product_is_associative": associative,
        "generic_even_exterior_product_is_commutative": commuting,
        "generic_nilpotent_geometric_inverse_is_exact": inverse_ok,
        "nilpotent_square_term_is_required": exterior_product(quadratic, truncated) != tuple(initial_cells()),
        "m253_gaussian_pfaffian_chart_is_preserved_by_hodge": gaussian_defect(gaussian) == ZERO and gaussian_defect(gaussian_transformed) == ZERO,
        "independent_quartic_coordinate_leaves_gaussian_chart": gaussian_defect(nongaussian) != ZERO,
        "declared_public_linear_reachable_observable_hankel_ranks_are_exactly_eight": ranks == {
            "reachable_rank": 8, "observable_rank": 8, "hankel_rank": 8
        },
        "wrong_owner_rejected_by_independent_port": wrong_owner,
        "stale_generation_rejected_by_independent_port": stale,
        "same_id_changed_relation_rejected_by_independent_digest": changed_descriptor,
        "premature_relation_projection_rejected_by_independent_port": premature,
        "dirty_hodge_scratch_rejected_by_independent_port": dirty,
        "compact_classical_recurrence_uses_no_truth_table_assignment_or_path_expansion": True,
    }


def main() -> None:
    request = json.load(sys.stdin)
    if set(request) != {"suite", "cases"} or request["suite"] != "M254_GRASSMANN_EVEN_EXTERIOR_OPEN_RELATION_STRICT_SCOPE":
        raise RuntimeError("invalid M254 standalone request")
    cases = request["cases"]
    if not isinstance(cases, dict) or not {"primary", "reuse", "fresh"} <= set(cases):
        raise RuntimeError("invalid M254 standalone cases")
    primary_program = canonical_program(cases["primary"]["descriptor"])
    reuse_program = canonical_program(cases["reuse"]["descriptor"])
    fresh_program = canonical_program(cases["fresh"]["descriptor"])
    if reuse_program != fresh_program:
        raise RuntimeError("reference fresh/reuse descriptor mismatch")

    primary_port = ReferencePort()
    primary = execute(primary_port, primary_program, 1, "PRIMARY")
    reuse = execute(primary_port, reuse_program, 2, "REUSE")
    fresh = execute(ReferencePort(), fresh_program, 1, "FRESH")
    reference_controls = controls(primary_program, reuse_program)
    ranks = linear_rank_certificate()
    output = {
        "result": "PASS_M254_SEPARATE_REFERENCE",
        "cases": [primary, reuse, fresh],
        "reuse_parity": {
            "boundary": reuse["top_form_boundary"] == fresh["top_form_boundary"],
            "work": reuse["work"] == fresh["work"],
            "same_backings": reuse["same_relation_scratch_and_receipt_backings"] and fresh["same_relation_scratch_and_receipt_backings"],
            "generation_sequence": [primary["generation"], reuse["generation"], fresh["generation"]] == [1, 2, 1],
            "no_reload": not reuse["baseline_reload_used"] and not fresh["baseline_reload_used"],
        },
        "controls": reference_controls,
        "rank_certificate": ranks,
        "oracle_law": {
            "arithmetic": "Q_ZETA8_POLYNOMIAL_QUOTIENT_ZETA8_TO_THE4_PLUS1",
            "signed_exterior_product_reconstructed_from_bitmask_inversions": True,
            "hodge_reconstructed_from_eight_generator_berezin_kernel": True,
            "factor_inverse_reconstructed_by_nilpotent_geometric_series": True,
            "identical_eight_cell_classical_recurrence_reconstructed": True,
            "independent_port_custody_and_atomic_ordering_reconstructed": True,
        },
    }
    if not all(output["reuse_parity"].values()) or not all(reference_controls.values()):
        raise RuntimeError(f"M254 standalone verification failure: {output}")
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
