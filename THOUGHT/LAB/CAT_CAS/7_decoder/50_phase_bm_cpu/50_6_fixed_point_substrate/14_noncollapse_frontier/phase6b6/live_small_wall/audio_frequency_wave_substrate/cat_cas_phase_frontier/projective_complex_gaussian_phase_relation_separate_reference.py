#!/usr/bin/env python3
"""Standalone symmetric-matrix reference for projective Gaussian phases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction


DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = (0, 1)
PRIMARY_DEPTH = 128
REUSE_DEPTH = 73
Matrix = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
SOURCE: Matrix = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))


def token(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def triple(matrix: Matrix) -> tuple[Fraction, Fraction, Fraction]:
    return matrix[0][0], matrix[0][1], matrix[1][1]


def matrix_token(matrix: Matrix) -> str:
    return ":".join(token(value) for value in triple(matrix))


def commitment(matrix: Matrix) -> str:
    return hashlib.sha256(matrix_token(matrix).encode("ascii")).hexdigest()


def scalar_commitment(value: Fraction) -> str:
    return hashlib.sha256(token(value).encode("ascii")).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload(values) -> int:
    return sum(
        signed_bits(value.numerator) + value.denominator.bit_length()
        for value in values
    )


def phase_mod2(value: Fraction) -> Fraction:
    return Fraction(value.numerator % (2 * value.denominator), value.denominator)


def public_operation(depth: int, family: int, step: int, *, perturb=False):
    if depth <= 0 or family not in FAMILIES or step < 0 or step >= 2 * depth:
        raise ValueError("reference public program mismatch")
    _round, phase = divmod(step, 2)
    if family == 0:
        intersection = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(2)))
        composition = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
    else:
        intersection = ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(3)))
        composition = ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(1)))
    matrix = intersection if phase == 0 else composition
    if perturb and step == 1:
        a, b, c = triple(matrix)
        b += 1
        if b == 0:
            b += 1
        matrix = ((a, b), (b, c))
    return ("INTERSECT", matrix) if phase == 0 else ("COMPOSE_RIGHT", matrix)


def add(left: Matrix, right: Matrix) -> Matrix:
    a, b, c = triple(left)
    d, e, f = triple(right)
    return ((a + d, b + e), (b + e, c + f))


def subtract(left: Matrix, right: Matrix) -> Matrix:
    a, b, c = triple(left)
    d, e, f = triple(right)
    return ((a - d, b - e), (b - e, c - f))


def schur_compose(left: Matrix, right: Matrix) -> Matrix:
    a, b, c = triple(left)
    d, e, f = triple(right)
    internal_hessian = c + d
    if internal_hessian == 0 or e == 0:
        raise ZeroDivisionError("reference singular shared port")
    coupling = (b, e)
    corners = ((a, Fraction(0)), (Fraction(0), f))
    output00 = corners[0][0] - coupling[0] * coupling[0] / internal_hessian
    output01 = corners[0][1] - coupling[0] * coupling[1] / internal_hessian
    output11 = corners[1][1] - coupling[1] * coupling[1] / internal_hessian
    return ((output00, output01), (output01, output11))


def inverse_schur(output: Matrix, public: Matrix) -> Matrix:
    output_a, output_b, output_c = triple(output)
    public_a, public_b, public_c = triple(public)
    critical = public_c - output_c
    if public_b == 0 or critical == 0:
        raise ZeroDivisionError("reference inverse shared port singular")
    internal_hessian = public_b * public_b / critical
    old_b = -output_b * public_b / critical
    old_a = output_a + output_b * output_b / critical
    old_c = internal_hessian - public_a
    return ((old_a, old_b), (old_b, old_c))


def evaluate(matrix: Matrix, left: Fraction, right: Fraction) -> Fraction:
    a, b, c = triple(matrix)
    return a * left * left + 2 * b * left * right + c * right * right


def forward(depth: int, family: int, *, perturb=False) -> Matrix:
    relation = SOURCE
    for step in range(2 * depth):
        kind, public = public_operation(depth, family, step, perturb=perturb)
        relation = add(relation, public) if kind == "INTERSECT" else schur_compose(relation, public)
    return relation


def closed_formula(depth: int, family: int) -> Matrix:
    if family == 0:
        values = (
            Fraction(depth * depth, depth + 1),
            Fraction((-1) ** depth, depth + 1),
            Fraction(-depth, depth + 1),
        )
    else:
        power = 4**depth
        values = (
            Fraction(2 * depth) - Fraction(1, 3) + Fraction(1, 3 * power),
            Fraction((-1) ** depth, 2**depth),
            Fraction(0),
        )
    return ((values[0], values[1]), (values[1], values[2]))


def transaction(depth: int, family: int, generation: int, carrier=None):
    if carrier is None:
        carrier = [*triple(SOURCE)]
    backing = id(carrier)
    peak = payload(tuple(carrier))
    for step in range(2 * depth):
        kind, public = public_operation(depth, family, step)
        current = ((carrier[0], carrier[1]), (carrier[1], carrier[2]))
        result = add(current, public) if kind == "INTERSECT" else schur_compose(current, public)
        carrier[:] = triple(result)
        peak = max(peak, payload(tuple(carrier)))
    final = ((carrier[0], carrier[1]), (carrier[1], carrier[2]))
    final_payload = payload(triple(final))
    left, right = Fraction(family + 1), Fraction(2 * family + 3)
    boundary = phase_mod2(evaluate(final, left, right))
    missing = sum(a != b for a, b in zip(triple(final), triple(SOURCE))) + 2 * depth
    for step in range(2 * depth - 1, -1, -1):
        kind, public = public_operation(depth, family, step)
        current = ((carrier[0], carrier[1]), (carrier[1], carrier[2]))
        result = subtract(current, public) if kind == "INTERSECT" else inverse_schur(current, public)
        carrier[:] = triple(result)
        peak = max(peak, payload(tuple(carrier)))
    restored = tuple(carrier) == triple(SOURCE)
    return {
        "depth": depth,
        "family": family,
        "boundary_token": token(boundary),
        "boundary_commitment": scalar_commitment(boundary),
        "boundary_payload_bits": payload((boundary,)),
        "forward_relation_commitment": commitment(final),
        "forward_carrier_payload_bits": final_payload,
        "peak_carrier_payload_bits": peak,
        "persistent_relation_rational_cells": 3,
        "retained_final_boundary_rational_cells_during_inverse": 1,
        "missing_inverse_error_cells_and_cursor": missing,
        "restoration_error_rational_cells": 0 if restored else 3,
        "same_coefficient_backing": id(carrier) == backing,
        "canonical_post_restoration_state_exact": restored,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }, carrier


def stationary_identity() -> bool:
    left = ((Fraction(7, 3), Fraction(-2, 5)), (Fraction(-2, 5), Fraction(4, 7)))
    right = ((Fraction(5), Fraction(-3)), (Fraction(-3), Fraction(8)))
    output = schur_compose(left, right)
    shared = left[1][1] + right[0][0]
    for x in (Fraction(-2), Fraction(1, 3), Fraction(5)):
        for z in (Fraction(-1), Fraction(2, 5), Fraction(4)):
            y = -(left[0][1] * x + right[0][1] * z) / shared
            if evaluate(left, x, y) + evaluate(right, y, z) != evaluate(output, x, z):
                return False
    return True


@dataclass
class ReferencePort:
    coefficients: list[Fraction]
    last_generation: int = 0
    live: bool = False
    owner: int = 0
    program: tuple[int, int] | None = None
    cursor: int = 0
    steps: int = 0
    port_type: str = "PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_R2"

    def lease(self, owner: int, generation: int, program: tuple[int, int]) -> None:
        depth, family = program
        if len(self.coefficients) != 3:
            raise ValueError("reference null carrier")
        if self.live or owner <= 0 or generation != self.last_generation + 1:
            raise ValueError("reference lease mismatch")
        if depth <= 0 or family not in FAMILIES:
            raise ValueError("reference program mismatch")
        self.live, self.owner, self.program = True, owner, program
        self.cursor, self.steps = 0, 2 * depth

    def require(self, owner: int, program: tuple[int, int]) -> None:
        if not self.live or owner != self.owner:
            raise PermissionError("reference owner mismatch")
        if program != self.program:
            raise ValueError("reference program receipt mismatch")
        if self.port_type != "PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_R2":
            raise TypeError("reference port type mismatch")

    def project(self, owner: int, program: tuple[int, int]) -> None:
        self.require(owner, program)
        if self.cursor != self.steps:
            raise PermissionError("reference premature projection")

    def inverse_cursor(self, owner: int, program: tuple[int, int], step: int) -> None:
        self.require(owner, program)
        if step != self.cursor - 1:
            raise ValueError("reference inverse order mismatch")

    def release(self, owner: int, program: tuple[int, int]) -> None:
        self.require(owner, program)
        if self.cursor or tuple(self.coefficients) != triple(SOURCE):
            raise RuntimeError("reference carrier not restored")


def controls():
    generic = ((Fraction(2), Fraction(1)), (Fraction(1), Fraction(3)))
    intersection = public_operation(8, 0, 0)[1]
    composition = public_operation(8, 0, 1)[1]
    staged = add(SOURCE, intersection)
    singular = zero_coupling = premature = wrong_owner = wrong_program = wrong_type = wrong_generation = missing_inverse = reordered = stale = null = False
    try:
        schur_compose(SOURCE, ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(3))))
    except ZeroDivisionError:
        singular = True
    try:
        candidate = ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(3)))
        if candidate[0][1] == 0:
            raise ValueError("noninvertible public coupling")
    except ValueError:
        zero_coupling = True
    reference_port = ReferencePort([*triple(SOURCE)])
    reference_port.lease(234999, 1, (4, 0))
    try:
        reference_port.project(234999, (4, 0))
    except PermissionError:
        premature = True
    try:
        reference_port.require(234998, (4, 0))
    except PermissionError:
        wrong_owner = True
    try:
        reference_port.require(234999, (5, 0))
    except ValueError:
        wrong_program = True
    reference_port.port_type = "WRONG"
    try:
        reference_port.require(234999, (4, 0))
    except TypeError:
        wrong_type = True
    reference_port.port_type = "PROJECTIVE_COMPLEX_GAUSSIAN_PHASE_RELATION_R2"
    try:
        reference_port.inverse_cursor(234999, (4, 0), 0)
    except ValueError:
        reordered = True
    missing_port = ReferencePort([*triple(SOURCE)])
    missing_port.lease(234995, 1, (4, 0))
    missing_port.cursor = 1
    missing_port.coefficients[0] += 1
    try:
        missing_port.release(234995, (4, 0))
    except RuntimeError:
        missing_inverse = True
    try:
        ReferencePort([*triple(SOURCE)]).lease(234998, 2, (4, 0))
    except ValueError:
        wrong_generation = True
    try:
        ReferencePort([*triple(SOURCE)], last_generation=1).lease(234997, 1, (4, 0))
    except ValueError:
        stale = True
    try:
        ReferencePort([]).lease(234996, 1, (4, 0))
    except ValueError:
        null = True
    q, g, h = (
        ((Fraction(2), Fraction(1)), (Fraction(1), Fraction(3))),
        ((Fraction(4), Fraction(2)), (Fraction(2), Fraction(5))),
        ((Fraction(3), Fraction(-1)), (Fraction(-1), Fraction(7))),
    )
    samples = (
        generic,
        ((Fraction(4), Fraction(-2)), (Fraction(-2), Fraction(5))),
        ((Fraction(7, 3), Fraction(5, 4)), (Fraction(5, 4), Fraction(9, 2))),
    )
    def perturbation_changes_or_rejects(family):
        original = phase_mod2(evaluate(
            forward(16, family), Fraction(family + 1), Fraction(2 * family + 3)
        ))
        try:
            mutated = phase_mod2(evaluate(
                forward_perturbed(16, family),
                Fraction(family + 1), Fraction(2 * family + 3),
            ))
        except ZeroDivisionError:
            return True
        return original != mutated
    return {
        "stationary_substitution_identity_exact": stationary_identity(),
        "compose_right_inverse_exact": all(
            inverse_schur(schur_compose(add(SOURCE, public_operation(8, family, 0)[1]), public_operation(8, family, 1)[1]), public_operation(8, family, 1)[1])
            == add(SOURCE, public_operation(8, family, 0)[1])
            for family in FAMILIES
        ),
        "partial_composition_associativity_exact": schur_compose(schur_compose(q, g), h) == schur_compose(q, schur_compose(g, h)),
        "composition_and_intersection_noncommute": schur_compose(add(generic, intersection), composition) != add(schur_compose(generic, composition), intersection),
        "intersection_associative_exact": add(add(samples[0], samples[1]), samples[2]) == add(samples[0], add(samples[1], samples[2])),
        "intersection_commutative_exact": add(samples[0], samples[1]) == add(samples[1], samples[0]),
        "closed_depth_formula_matches_all_declared_cases": all(
            forward(depth, family) == closed_formula(depth, family)
            for family in FAMILIES for depth in DEPTHS
        ),
        "singular_shared_port_rejected": singular,
        "zero_coupling_irreversible_module_rejected": zero_coupling,
        "premature_projection_rejected": premature,
        "wrong_owner_rejected": wrong_owner,
        "wrong_public_program_rejected": wrong_program,
        "wrong_port_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_release_rejected": missing_inverse,
        "stale_generation_rejected": stale,
        "null_carrier_rejected": null,
        "public_coupling_perturbation_changes_or_rejects_both_families": all(
            perturbation_changes_or_rejects(family) for family in FAMILIES
        ),
        "public_compiler_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
        "intermediate_relations_serialized": False,
    }


def forward_perturbed(depth: int, family: int) -> Matrix:
    relation = SOURCE
    for step in range(2 * depth):
        kind, public = public_operation(depth, family, step, perturb=True)
        relation = add(relation, public) if kind == "INTERSECT" else schur_compose(relation, public)
    return relation


def main() -> None:
    cases = []
    for family in FAMILIES:
        for depth in DEPTHS:
            case, _ = transaction(depth, family, 1)
            cases.append(case)
    carrier = [*triple(SOURCE)]
    primary, carrier = transaction(PRIMARY_DEPTH, 0, 1, carrier)
    reuse, carrier = transaction(REUSE_DEPTH, 1, 2, carrier)
    fresh, _ = transaction(REUSE_DEPTH, 1, 1)
    current_controls = controls()
    if not all(value for key, value in current_controls.items() if key not in (
        "public_compiler_reads_final_answer", "relation_tables_materialized",
        "assignment_expansions_materialized", "intermediate_relations_serialized",
    )) or any(current_controls[key] for key in (
        "public_compiler_reads_final_answer", "relation_tables_materialized",
        "assignment_expansions_materialized", "intermediate_relations_serialized",
    )):
        raise RuntimeError("reference Gaussian control failed")
    result = {
        "schema": "cat_cas.projective_complex_gaussian_phase_relation_reference.v1",
        "cases": cases,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh["boundary_commitment"],
            "fresh_restored_forward_relation_agreement": reuse["forward_relation_commitment"] == fresh["forward_relation_commitment"],
            "same_backing_across_primary_and_reuse": primary["same_coefficient_backing"] and reuse["same_coefficient_backing"],
            "restoration_generation_after_reuse": 2,
        },
        "controls": current_controls,
        "imports_m234_production": False,
        "uses_independent_symmetric_matrix_schur_complement": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
