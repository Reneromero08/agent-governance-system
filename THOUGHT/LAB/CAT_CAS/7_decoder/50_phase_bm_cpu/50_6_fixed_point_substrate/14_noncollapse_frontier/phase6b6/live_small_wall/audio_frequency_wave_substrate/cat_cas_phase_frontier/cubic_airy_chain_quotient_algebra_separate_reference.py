#!/usr/bin/env python3
"""Standalone sparse-dictionary oracle for M236.

This source intentionally imports no M236 production implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable


TYPE = "CUBIC_AIRY_CHAIN_QUOTIENT_ALGEBRA_V1"
DEPTHS = (2, 3, 4, 5, 6)
BOUNDARIES = ((-3, 1), (-2, 1))
FAMILIES = ((1, -1, 1, 1, -1, 1), (-1, 1, 1, -1, -1, 1))
Poly = dict[int, int]


def clean(poly: Poly) -> Poly:
    return {degree: value for degree, value in poly.items() if value}


def plus(left: Poly, right: Poly) -> Poly:
    result = dict(left)
    for degree, value in right.items():
        result[degree] = result.get(degree, 0) + value
    return clean(result)


def minus(left: Poly, right: Poly) -> Poly:
    return plus(left, {degree: -value for degree, value in right.items()})


def times(left: Poly, right: Poly) -> Poly:
    result: Poly = {}
    for i, a in left.items():
        for j, b in right.items():
            result[i + j] = result.get(i + j, 0) + a * b
    return clean(result)


def scaled(poly: Poly, value: int) -> Poly:
    return clean({degree: value * coefficient for degree, coefficient in poly.items()})


def degree(poly: Poly) -> int:
    return max(poly) if poly else -1


def dense(poly: Poly, size: int | None = None) -> list[int]:
    required = degree(poly) + 1 if size is None else size
    result = [0] * max(1, required)
    for exponent, value in poly.items():
        if exponent >= len(result):
            result.extend([0] * (exponent + 1 - len(result)))
        result[exponent] = value
    while size is None and len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def commitment(poly: Poly, size: int | None = None) -> str:
    return hashlib.sha256(json.dumps(dense(poly, size), separators=(",", ":")).encode()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload(poly: Poly, size: int | None = None) -> int:
    return sum(signed_bits(value) for value in dense(poly, size))


@dataclass(frozen=True)
class Spec:
    family: int
    length: int
    x: int
    z: int
    alphas: tuple[int, ...]
    program_id: str

    def __post_init__(self) -> None:
        if self.family not in (0, 1) or self.length not in DEPTHS:
            raise ValueError("invalid reference descriptor")
        if len(self.alphas) != self.length or any(
            not isinstance(alpha, int) or alpha == 0 for alpha in self.alphas
        ):
            raise ValueError("invalid cubic coefficient")
        if self.program_id != spec_digest(
            self.family, self.length, self.x, self.z, self.alphas
        ):
            raise ValueError("reference program identity is not descriptor-bound")


def spec_digest(
    family: int, length: int, x: int, z: int, alphas: tuple[int, ...]
) -> str:
    encoded = json.dumps(
        [family, length, x, z, list(alphas)], separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compile_spec(family: int, length: int) -> Spec:
    if family not in (0, 1) or length not in DEPTHS:
        raise ValueError("invalid reference program")
    x, z = BOUNDARIES[family]
    alphas = FAMILIES[family][:length]
    return Spec(family, length, x, z, alphas, spec_digest(family, length, x, z, alphas))


def coordinates(spec: Spec) -> list[Poly]:
    result: list[Poly] = [{0: spec.x}, {1: 1}]
    for index in range(1, spec.length):
        result.append(
            scaled(
                plus(result[index - 1], scaled(times(result[index], result[index]), spec.alphas[index - 1])),
                -1,
            )
        )
    return result


def endpoint(spec: Spec) -> tuple[Poly, list[Poly]]:
    paths = coordinates(spec)
    modulus = minus(paths[-1], {0: spec.z})
    lead = modulus[degree(modulus)]
    if lead == -1:
        modulus = scaled(modulus, -1)
    elif lead != 1:
        raise RuntimeError("reference endpoint is not integral monic")
    if degree(modulus) != 1 << (spec.length - 1):
        raise RuntimeError("reference endpoint degree mismatch")
    return modulus, paths


def residue(poly: Poly, modulus: Poly) -> Poly:
    result = dict(poly)
    target = degree(modulus)
    while degree(result) >= target:
        exponent = degree(result)
        coefficient = result[exponent]
        offset = exponent - target
        for m_degree, value in modulus.items():
            key = offset + m_degree
            result[key] = result.get(key, 0) - coefficient * value
        result = clean(result)
    return result


def qadd(left: Poly, right: Poly, modulus: Poly) -> Poly:
    return residue(plus(left, right), modulus)


def qsub(left: Poly, right: Poly, modulus: Poly) -> Poly:
    return residue(minus(left, right), modulus)


def qmul(left: Poly, right: Poly, modulus: Poly) -> Poly:
    return residue(times(left, right), modulus)


def qscale(poly: Poly, value: int, modulus: Poly) -> Poly:
    return residue(scaled(poly, value), modulus)


def action_term(left: Poly, right: Poly, alpha: int, modulus: Poly) -> Poly:
    pair = qmul(left, right, modulus)
    cube = qmul(qmul(right, right, modulus), right, modulus)
    return qadd(qscale(pair, 3, modulus), qscale(cube, alpha, modulus), modulus)


def multiplication_matrix(element: Poly, modulus: Poly) -> list[list[int]]:
    size = degree(modulus)
    matrix = [[0] * size for _ in range(size)]
    basis: Poly = {0: 1}
    for column in range(size):
        product = qmul(element, basis, modulus)
        for row, value in product.items():
            matrix[row][column] = value
        basis = qmul(basis, {1: 1}, modulus)
    return matrix


def matmul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    size = len(left)
    result = [[0] * size for _ in range(size)]
    for row in range(size):
        for column in range(size):
            result[row][column] = sum(left[row][inner] * right[inner][column] for inner in range(size))
    return result


def charpoly(matrix: list[list[int]]) -> Poly:
    size = len(matrix)
    current = [[int(row == column) for column in range(size)] for row in range(size)]
    high = [1]
    for order in range(1, size + 1):
        product = matmul(matrix, current)
        numerator = -sum(product[index][index] for index in range(size))
        if numerator % order:
            raise RuntimeError("reference characteristic coefficient is nonintegral")
        coefficient = numerator // order
        high.append(coefficient)
        for index in range(size):
            product[index][index] += coefficient
        current = product
    return clean({size - index: value for index, value in enumerate(high)})


def divmod_q(left: Poly, right: Poly) -> tuple[dict[int, Fraction], dict[int, Fraction]]:
    numerator = {key: Fraction(value) for key, value in left.items()}
    denominator = {key: Fraction(value) for key, value in right.items()}
    quotient: dict[int, Fraction] = {}
    while numerator and max(numerator) >= max(denominator):
        offset = max(numerator) - max(denominator)
        value = numerator[max(numerator)] / denominator[max(denominator)]
        quotient[offset] = quotient.get(offset, Fraction(0)) + value
        for key, coefficient in denominator.items():
            target = offset + key
            numerator[target] = numerator.get(target, Fraction(0)) - value * coefficient
            if numerator[target] == 0:
                numerator.pop(target)
    return quotient, numerator


def square_free(poly: Poly) -> bool:
    derivative = {key - 1: key * value for key, value in poly.items() if key}
    left = {key: Fraction(value) for key, value in poly.items()}
    right = {key: Fraction(value) for key, value in derivative.items()}
    while right:
        _, remainder = divmod_q(left, right)  # type: ignore[arg-type]
        left, right = right, remainder
    return max(left) == 0


def evaluate_relation(relation: Poly, value: Poly, modulus: Poly) -> Poly:
    result: Poly = {}
    for exponent in range(degree(relation), -1, -1):
        result = qmul(result, value, modulus)
        result = qadd(result, {0: relation.get(exponent, 0)}, modulus)
    return result


def direct(spec: Spec) -> dict[str, object]:
    modulus, paths = endpoint(spec)
    reduced = [residue(item, modulus) for item in paths]
    action: Poly = {}
    for index in range(1, spec.length):
        action = qadd(action, action_term(reduced[index - 1], reduced[index], spec.alphas[index - 1], modulus), modulus)
    action = qadd(action, action_term(reduced[-2], {0: spec.z}, spec.alphas[-1], modulus), modulus)
    relation = charpoly(multiplication_matrix(action, modulus))
    if not square_free(modulus) or not square_free(relation):
        raise RuntimeError("reference fixture is not square-free")
    if evaluate_relation(relation, action, modulus):
        raise RuntimeError("reference characteristic relation failed")
    return {"modulus": modulus, "action": action, "relation": relation}


class ReferencePort:
    def __init__(self, spec: Spec) -> None:
        self.modulus, _ = endpoint(spec)
        self.r: Poly = {0: spec.x}
        self.s: Poly = {1: 1}
        self.h: Poly = {}
        self.initial_x = spec.x
        self.owner: int | None = None
        self.program_id: str | None = None
        self.program_descriptor: tuple[object, ...] | None = None
        self.port_type: str | None = None
        self.generation = 0
        self.last = 0
        self.cursor = 0
        self.final = False
        self.leased = False

    def canonical_r(self) -> Poly:
        return {0: self.initial_x}

    @staticmethod
    def canonical_s() -> Poly:
        return {1: 1}

    def lease(self, owner: int, spec: Spec, generation: int, port_type: str = TYPE) -> None:
        expected, _ = endpoint(spec)
        if self.leased or self.modulus != expected or self.r != self.canonical_r() or self.s != self.canonical_s() or self.h:
            raise RuntimeError("reference carrier not canonical")
        if owner <= 0 or port_type != TYPE or generation != self.last + 1:
            raise RuntimeError("reference lease mismatch")
        self.owner, self.program_id, self.port_type = owner, spec.program_id, port_type
        self.program_descriptor = (
            spec.family, spec.length, spec.x, spec.z, spec.alphas
        )
        self.generation, self.leased = generation, True

    def require(self, owner: int, spec: Spec, generation: int) -> None:
        if not self.leased or (owner, spec.program_id, TYPE, generation) != (
            self.owner, self.program_id, self.port_type, self.generation
        ) or self.program_descriptor != (
            spec.family, spec.length, spec.x, spec.z, spec.alphas
        ):
            raise RuntimeError("reference custody mismatch")

    @staticmethod
    def replace(target: Poly, value: Poly) -> None:
        target.clear()
        target.update(value)

    def forward(self, owner: int, spec: Spec, generation: int, index: int) -> None:
        self.require(owner, spec, generation)
        if index != self.cursor or index >= spec.length - 1 or self.final:
            raise RuntimeError("reference forward order")
        alpha = spec.alphas[index]
        self.replace(self.h, qadd(self.h, action_term(self.r, self.s, alpha, self.modulus), self.modulus))
        old_r, old_s = dict(self.r), dict(self.s)
        self.replace(self.r, old_s)
        self.replace(self.s, qsub(qscale(old_r, -1, self.modulus), qscale(qmul(old_s, old_s, self.modulus), alpha, self.modulus), self.modulus))
        self.cursor += 1

    def add_final(self, owner: int, spec: Spec, generation: int) -> None:
        self.require(owner, spec, generation)
        if self.cursor != spec.length - 1 or self.final or self.s != {0: spec.z}:
            raise RuntimeError("reference final order")
        self.replace(self.h, qadd(self.h, action_term(self.r, {0: spec.z}, spec.alphas[-1], self.modulus), self.modulus))
        self.final = True

    def project(self, owner: int, spec: Spec, generation: int) -> Poly:
        self.require(owner, spec, generation)
        if self.cursor != spec.length - 1 or not self.final:
            raise RuntimeError("reference premature projection")
        return charpoly(multiplication_matrix(self.h, self.modulus))

    def project_stationary(self) -> None:
        raise RuntimeError("reference stationary projection forbidden")

    def remove_final(self, owner: int, spec: Spec, generation: int) -> None:
        self.require(owner, spec, generation)
        if not self.final or self.cursor != spec.length - 1:
            raise RuntimeError("reference final inverse order")
        self.replace(self.h, qsub(self.h, action_term(self.r, {0: spec.z}, spec.alphas[-1], self.modulus), self.modulus))
        self.final = False

    def inverse(self, owner: int, spec: Spec, generation: int, index: int) -> None:
        self.require(owner, spec, generation)
        if index != self.cursor - 1 or self.final:
            raise RuntimeError("reference inverse order")
        alpha = spec.alphas[index]
        old_r, old_s = dict(self.r), dict(self.s)
        recovered_r = qsub(qscale(old_s, -1, self.modulus), qscale(qmul(old_r, old_r, self.modulus), alpha, self.modulus), self.modulus)
        self.replace(self.r, recovered_r)
        self.replace(self.s, old_r)
        self.replace(self.h, qsub(self.h, action_term(self.r, self.s, alpha, self.modulus), self.modulus))
        self.cursor -= 1

    def release(self, owner: int, spec: Spec, generation: int) -> None:
        self.require(owner, spec, generation)
        if self.cursor or self.final or self.h or self.r != self.canonical_r() or self.s != self.canonical_s():
            raise RuntimeError("reference carrier not restored")
        self.last = generation
        self.owner = self.program_id = self.port_type = None
        self.program_descriptor = None
        self.generation, self.leased = 0, False


def run_case(port: ReferencePort, spec: Spec, generation: int, owner: int = 23601) -> dict[str, object]:
    if port is None:
        raise TypeError("null reference cubic Airy-chain carrier")
    backing = (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.lease(owner, spec, generation)
    for index in range(spec.length - 1):
        port.forward(owner, spec, generation, index)
    port.add_final(owner, spec, generation)
    size = degree(port.modulus)
    resident_commitment = commitment(port.h, size)
    relation = port.project(owner, spec, generation)
    port.remove_final(owner, spec, generation)
    for index in range(spec.length - 2, -1, -1):
        port.inverse(owner, spec, generation, index)
    exact = port.r == port.canonical_r() and port.s == port.canonical_s() and not port.h
    same = backing == (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.release(owner, spec, generation)
    baseline = direct(spec)
    if relation != baseline["relation"] or resident_commitment != commitment(baseline["action"], size):
        raise RuntimeError("post-restoration reference direct parity failed")
    return {
        "family": spec.family,
        "length": spec.length,
        "quotient_dimension": size,
        "endpoint_polynomial_degree": size,
        "critical_value_relation_degree": degree(relation),
        "endpoint_polynomial_square_free": square_free(port.modulus),
        "critical_value_relation_square_free": square_free(relation),
        "critical_value_relation_annihilates_resident_action": not evaluate_relation(
            relation, baseline["action"], port.modulus
        ),
        "modulus_commitment": commitment(port.modulus),
        "resident_critical_value_commitment": resident_commitment,
        "critical_value_relation_commitment": commitment(relation),
        "modulus_coefficient_cells": size + 1,
        "carrier_quotient_coefficient_cells": 3 * size,
        "retained_final_relation_coefficient_cells_during_inverse": degree(relation) + 1,
        "modulus_payload_bits": payload(port.modulus),
        "resident_critical_value_payload_bits": payload(baseline["action"], size),
        "critical_value_relation_payload_bits": payload(relation),
        "canonical_post_inverse_state_exact": exact,
        "same_r_s_h_modulus_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
        "retained_dynamic_inverse_history_entries": 0,
        "matched_classical_relation_commitment": commitment(baseline["relation"]),
    }


def sentinel(port: ReferencePort, spec: Spec, generation: int, gamma: int = 2, owner: int = 23602) -> dict[str, object]:
    backing = (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.lease(owner, spec, generation)
    product = qmul(port.r, port.s, port.modulus)
    ReferencePort.replace(port.h, qadd(port.h, product, port.modulus))
    square = qmul(port.s, port.s, port.modulus)
    ReferencePort.replace(port.r, qadd(port.r, qscale(square, gamma, port.modulus), port.modulus))
    boundary = charpoly(multiplication_matrix(port.h, port.modulus))
    ReferencePort.replace(port.r, qsub(port.r, qscale(square, gamma, port.modulus), port.modulus))
    ReferencePort.replace(port.h, qsub(port.h, product, port.modulus))
    exact = port.r == port.canonical_r() and port.s == port.canonical_s() and not port.h
    same = backing == (id(port.r), id(port.s), id(port.h), id(port.modulus))
    port.release(owner, spec, generation)
    return {
        "boundary_commitment": commitment(boundary),
        "boundary_payload_bits": payload(boundary),
        "canonical_post_inverse_state_exact": exact,
        "same_r_s_h_modulus_backings": same,
        "restoration_generation": generation,
        "baseline_reload_used": False,
    }


def rejected(callback: Callable[[], object]) -> bool:
    try:
        callback()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def controls() -> dict[str, bool]:
    spec = compile_spec(0, 3)
    n2 = compile_spec(0, 2)
    relation = direct(n2)["relation"]
    alpha1, alpha2 = n2.alphas
    expected = {0: alpha1 * alpha2**2 * n2.z**6 + 4 * (n2.x + n2.z) ** 3, 1: -2 * alpha1 * alpha2 * n2.z**3, 2: alpha1}
    if expected[2] == -1:
        expected = scaled(expected, -1)
    port = ReferencePort(spec)
    port.lease(23611, spec, 1)
    premature = rejected(lambda: port.project(23611, spec, 1))
    stationary = rejected(port.project_stationary)
    wrong_owner = rejected(lambda: port.forward(23612, spec, 1, 0))
    wrong_program = rejected(lambda: port.forward(23611, compile_spec(1, 3), 1, 0))
    forged_program_identity = rejected(
        lambda: Spec(
            spec.family, spec.length, spec.x, spec.z,
            spec.alphas[:-1] + (-spec.alphas[-1],), spec.program_id,
        )
    )
    wrong_generation = rejected(lambda: port.forward(23611, spec, 2, 0))
    wrong_type = rejected(lambda: ReferencePort(spec).lease(23611, spec, 1, "WRONG"))
    null = rejected(lambda: run_case(None, spec, 1))  # type: ignore[arg-type]
    all_stationary_equations = True
    for family in (0, 1):
        for length in DEPTHS:
            declared = compile_spec(family, length)
            declared_modulus, declared_coordinates = endpoint(declared)
            for index in range(1, declared.length):
                stationary_law = plus(
                    declared_coordinates[index - 1],
                    plus(
                        scaled(times(declared_coordinates[index], declared_coordinates[index]), declared.alphas[index - 1]),
                        declared_coordinates[index + 1],
                    ),
                )
                all_stationary_equations &= not residue(stationary_law, declared_modulus)

    missing = ReferencePort(spec)
    missing.lease(23611, spec, 1)
    for index in range(spec.length - 1):
        missing.forward(23611, spec, 1, index)
    missing.add_final(23611, spec, 1)
    missing_inverse = rejected(lambda: missing.release(23611, spec, 1))
    reordered_port = ReferencePort(spec)
    reordered_port.lease(23611, spec, 1)
    for index in range(spec.length - 1):
        reordered_port.forward(23611, spec, 1, index)
    reordered_port.add_final(23611, spec, 1)
    reordered_port.remove_final(23611, spec, 1)
    reordered_inverse = rejected(lambda: reordered_port.inverse(23611, spec, 1, 0))
    wrong_inverse_port = ReferencePort(spec)
    wrong_inverse_port.lease(23611, spec, 1)
    for index in range(spec.length - 1):
        wrong_inverse_port.forward(23611, spec, 1, index)
    wrong_inverse_port.add_final(23611, spec, 1)
    wrong_final = action_term(
        wrong_inverse_port.r, {0: spec.z}, -spec.alphas[-1], wrong_inverse_port.modulus
    )
    ReferencePort.replace(
        wrong_inverse_port.h,
        qsub(wrong_inverse_port.h, wrong_final, wrong_inverse_port.modulus),
    )
    wrong_inverse_port.final = False
    for index in range(spec.length - 2, -1, -1):
        wrong_inverse_port.inverse(23611, spec, 1, index)
    wrong_alpha = rejected(lambda: wrong_inverse_port.release(23611, spec, 1))
    dirty_port = ReferencePort(spec)
    dirty_port.h[0] = 1
    dirty_carrier = rejected(lambda: dirty_port.lease(23611, spec, 1))
    stale = ReferencePort(spec)
    run_case(stale, spec, 1, 23611)
    sentinel(stale, spec, 2, owner=23611)
    stale_generation = rejected(lambda: stale.lease(23611, spec, 2))
    zero = rejected(lambda: Spec(0, 2, -3, 1, (0, 1), "zero"))
    base = compile_spec(0, 4)
    original = commitment(direct(base)["relation"])
    reordered_alphas = (base.alphas[1], base.alphas[0]) + base.alphas[2:]
    reordered = Spec(
        0, 4, base.x, base.z, reordered_alphas,
        spec_digest(0, 4, base.x, base.z, reordered_alphas),
    )
    perturbed_alphas = (
        base.alphas[0], base.alphas[1], -base.alphas[2], base.alphas[3]
    )
    perturbed = Spec(
        0, 4, base.x, base.z, perturbed_alphas,
        spec_digest(0, 4, base.x, base.z, perturbed_alphas),
    )
    return {
        "direct_n2_eliminated_critical_value_formula": relation == expected,
        "all_declared_stationary_equations_vanish_in_endpoint_quotient": all_stationary_equations,
        "premature_final_projection_rejected": premature,
        "stationary_port_projection_rejected": stationary,
        "wrong_owner_rejected": wrong_owner,
        "wrong_program_rejected": wrong_program,
        "same_id_changed_descriptor_rejected": forged_program_identity,
        "wrong_type_rejected": wrong_type,
        "wrong_generation_rejected": wrong_generation,
        "stale_generation_rejected": stale_generation,
        "null_carrier_rejected": null,
        "missing_inverse_rejected": missing_inverse,
        "reordered_inverse_rejected": reordered_inverse,
        "wrong_alpha_inverse_rejected": wrong_alpha,
        "dirty_carrier_rejected": dirty_carrier,
        "zero_cubic_coefficient_rejected": zero,
        "module_reorder_changes_critical_value_relation": commitment(direct(reordered)["relation"]) != original,
        "semantic_alpha_perturbation_changes_critical_value_relation": commitment(direct(perturbed)["relation"]) != original,
        "stationary_roots_or_branch_lists_materialized": False,
        "intermediate_quotient_elements_serialized": False,
        "public_compiler_reads_final_answer": False,
    }


def main() -> None:
    cases = []
    for family in (0, 1):
        for length in DEPTHS:
            spec = compile_spec(family, length)
            cases.append(run_case(ReferencePort(spec), spec, 1))
    primary_spec = compile_spec(0, 5)
    shared = ReferencePort(primary_spec)
    primary = run_case(shared, primary_spec, 1)
    reuse = sentinel(shared, primary_spec, 2)
    fresh_reuse = sentinel(ReferencePort(primary_spec), primary_spec, 1)
    output = {
        "schema": "cat_cas.cubic_airy_chain_quotient_algebra_reference.v1",
        "cases": cases,
        "controls": controls(),
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh_reuse,
            "restoration_generation_after_reuse": shared.last,
            "fresh_restored_boundary_agreement": reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"],
            "same_backing_across_primary_and_reuse": primary["same_r_s_h_modulus_backings"] and reuse["same_r_s_h_modulus_backings"],
        },
        "imports_m236_production": False,
        "uses_independent_sparse_dictionary_quotient_arithmetic": True,
        "implements_independent_custody_state_machine": True,
        "source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
