#!/usr/bin/env python3
"""Independent oracle for the cubic-deformed growing-prime Weil package.

No production module is imported.  The oracle separately reconstructs the
Weyl-component recurrence, final-boundary q-vector word recurrence, exact
inverse, and small-order dense matrix semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)
CASES = tuple((q, 2, "PRIMARY") for q in ORDERS) + ((113, 4, "PRIMARY"), (41, 3, "ALTERNATE"))
CHECKPOINTS = {1, 2, 3, 4}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return n == d
        d += 1
    return True


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    omega: int
    gauss: int


def field(q: int) -> Field:
    p = 2 * q + 1
    require(is_prime(q) and is_prime(p), "invalid safe-prime pair")
    primitive = next(g for g in range(2, p) if pow(g, 2, p) != 1 and pow(g, q, p) != 1)
    omega = pow(primitive, 2, p)
    require(pow(omega, q, p) == 1 and all(pow(omega, k, p) != 1 for k in range(1, q)), "bad phase root")
    gauss = sum(pow(omega, x * x % q, p) for x in range(q)) % p
    require(gauss != 0, "vanishing Gauss sum")
    return Field(q, p, omega, gauss)


def phase(f: Field, exponent: int) -> int:
    return pow(f.omega, exponent % f.q, f.p)


def mm(left: tuple[int, ...], right: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = left
    e, g, h, i = right
    return ((a * e + b * h) % modulus, (a * g + b * i) % modulus, (c * e + d * h) % modulus, (c * g + d * i) % modulus)


def mi(matrix: tuple[int, ...], modulus: int) -> tuple[int, ...]:
    a, b, c, d = matrix
    determinant = (a * d - b * c) % modulus
    require(determinant != 0, "singular matrix")
    scale = pow(determinant, -1, modulus)
    return d * scale % modulus, -b * scale % modulus, -c * scale % modulus, a * scale % modulus


def kernel(symplectic: tuple[int, ...], x: int, y: int, f: Field) -> int:
    a, b, c, d = symplectic
    if b % f.q:
        exponent = (d * x * x - 2 * x * y + a * y * y) * pow(2 * b % f.q, -1, f.q)
        return phase(f, exponent)
    if y % f.q != d * x % f.q:
        return 0
    return phase(f, c * d * x * x * pow(2, -1, f.q))


def streamed(left: tuple[int, ...], right: tuple[int, ...], f: Field) -> int:
    return sum(kernel(left, 0, y, f) * kernel(right, y, 0, f) for y in range(f.q)) % f.p


def closed(left: tuple[int, ...], right: tuple[int, ...], f: Field) -> int:
    a, b, _, _ = left
    _, g, _, i = right
    if b % f.q == 0 or g % f.q == 0:
        return 1
    coefficient = (a * pow(2 * b % f.q, -1, f.q) + i * pow(2 * g % f.q, -1, f.q)) % f.q
    if coefficient == 0:
        return f.q % f.p
    return f.gauss if pow(coefficient, (f.q - 1) // 2, f.q) == 1 else -f.gauss % f.p


# Relation layout: [symplectic-list, fiber-list, coefficient-dict].
def relation(symplectic: tuple[int, ...], fiber: tuple[int, ...], coefficients: dict[tuple[int, int], int]) -> list[Any]:
    return [list(symplectic), list(fiber), dict(coefficients)]


def canon(item: list[Any], f: Field) -> tuple[Any, ...]:
    return tuple(v % f.q for v in item[0]), tuple(v % f.p for v in item[1]), tuple(sorted((r % f.q, s % f.q, value % f.p) for (r, s), value in item[2].items() if value % f.p))


def overwrite(target: list[Any], source: list[Any]) -> None:
    target[0][:] = source[0]
    target[1][:] = source[1]
    target[2].clear()
    target[2].update(source[2])


def covariance(symplectic: tuple[int, ...], key: tuple[int, int], q: int) -> tuple[int, int]:
    a, b, c, d = symplectic
    r, s = key
    return (a * r - b * s) % q, (-c * r + d * s) % q


def dphase(left: tuple[int, int], right: tuple[int, int], f: Field) -> int:
    return phase(f, (left[0] * right[1] - left[1] * right[0]) * pow(2, -1, f.q))


def compose(left: list[Any], right: list[Any], f: Field, audit: dict[str, int]) -> list[Any]:
    scalar_closed = closed(tuple(left[0]), tuple(right[0]), f)
    scalar_streamed = streamed(tuple(left[0]), tuple(right[0]), f)
    require(scalar_closed == scalar_streamed, "cocycle parity failure")
    audit["trajectory_cocycle_parity_checks"] += 1
    result: dict[tuple[int, int], int] = {}
    for u, a in left[2].items():
        for v, b in right[2].items():
            moved = covariance(tuple(left[0]), v, f.q)
            key = ((u[0] + moved[0]) % f.q, (u[1] + moved[1]) % f.q)
            result[key] = (result.get(key, 0) + scalar_closed * a * b * dphase(u, moved, f)) % f.p
    result = {key: value for key, value in result.items() if value}
    return relation(mm(tuple(left[0]), tuple(right[0]), f.q), mm(tuple(left[1]), tuple(right[1]), f.p), result)


def gaussian_inverse(item: list[Any], f: Field, audit: dict[str, int]) -> list[Any]:
    require(set(item[2]) == {(0, 0)}, "non-Gaussian inverse request")
    inverse_s = mi(tuple(item[0]), f.q)
    scalar = closed(tuple(item[0]), inverse_s, f)
    require(scalar == streamed(tuple(item[0]), inverse_s, f), "inverse cocycle parity failure")
    audit["trajectory_cocycle_parity_checks"] += 1
    coefficient = item[2][(0, 0)] * scalar % f.p
    return relation(inverse_s, mi(tuple(item[1]), f.p), {(0, 0): pow(coefficient, -1, f.p)})


def cubic_coefficients(strength: int, f: Field) -> dict[tuple[int, int], int]:
    inverse_q = pow(f.q, -1, f.p)
    result = {}
    for mode in range(f.q):
        value = inverse_q * sum(phase(f, strength * x**3 - mode * x) for x in range(f.q)) % f.p
        if value:
            result[(0, mode)] = value
    return result


def cubic(item: list[Any], row: int, column: int, f: Field, audit: dict[str, int]) -> list[Any]:
    identity = (1, 0, 0, 1)
    left = relation(identity, identity, cubic_coefficients(row, f))
    right = relation(identity, identity, cubic_coefficients(column, f))
    return compose(compose(left, item, f, audit), right, f, audit)


def chirp(item: list[Any], row: int, column: int, f: Field) -> list[Any]:
    require(set(item[2]) == {(0, 0)}, "non-Gaussian chirp request")
    left = (1, 0, 2 * row % f.q, 1)
    right = (1, 0, 2 * column % f.q, 1)
    return relation(mm(mm(left, tuple(item[0]), f.q), right, f.q), tuple(item[1]), dict(item[2]))


def conjugate(deformed: list[Any], resident: list[Any], inverse: bool, f: Field, audit: dict[str, int]) -> list[Any]:
    resident_inverse = gaussian_inverse(resident, f, audit)
    return compose(compose(resident_inverse, deformed, f, audit), resident, f, audit) if inverse else compose(compose(resident, deformed, f, audit), resident_inverse, f, audit)


def seal(q: int, family: str = "PRIMARY") -> dict[str, Any]:
    require(family in {"PRIMARY", "ALTERNATE"}, "invalid family")
    f = field(q)
    code = 1 if family == "PRIMARY" else 2
    a = relation((1, 1, 1, 2), (1, 2 + code, 3, 7 + 3 * code), {(0, 0): 1})
    b = relation((2, 1, 1, 1), (2, 1 + code, 5, 3 + 3 * code), {(0, 0): 1})
    mi(tuple(a[1]), f.p)
    mi(tuple(b[1]), f.p)
    return {"field": f, "family": family, "a": a, "b": b, "stage": "IDLE", "generation": 0}


def state(carrier: dict[str, Any]) -> tuple[Any, ...]:
    f = carrier["field"]
    return f.q, f.p, carrier["family"], canon(carrier["a"], f), canon(carrier["b"], f), carrier["stage"]


def backing(carrier: dict[str, Any]) -> tuple[int, ...]:
    return tuple(id(value) for item in (carrier["a"], carrier["b"]) for value in (item, item[0], item[1], item[2]))


def descriptors(index: int, family: str, q: int) -> tuple[tuple[Any, ...], ...]:
    require(family in {"PRIMARY", "ALTERNATE"}, "invalid descriptor family")
    code = 1 if family == "PRIMARY" else 2
    stages = (
        ("CHIRP", (2 * index + code + 1) % q, (3 * index + code + 2) % q),
        ("CONJUGATE", 0, 0),
        ("CUBIC", (5 * index + 2 * code + 1) % q, (7 * index + 3 * code + 2) % q),
    )
    return stages if family == "PRIMARY" else tuple(reversed(stages))


def checkpoint(carrier: dict[str, Any], depth: int) -> dict[str, int]:
    f = carrier["field"]
    sa, sb = len(carrier["a"][2]), len(carrier["b"][2])
    return {
        "depth": depth,
        "deformed_port_gaussian_component_support": sa,
        "resident_gaussian_port_component_support": sb,
        "deformed_port_component_capacity": f.q * f.q,
        "packed_component_coordinate_cells": 3 * (sa + sb),
        "resident_chart_and_fiber_cells": 16,
        "resident_packed_field_cells": 3 * (sa + sb) + 16,
        "logical_resident_payload_bits": (sa + sb) * (2 * (f.q - 1).bit_length() + (f.p - 1).bit_length()) + 8 * (f.q - 1).bit_length() + 8 * (f.p - 1).bit_length(),
        "ordinary_dense_two_port_relation_cells": 8 * f.q * f.q,
    }


def stages_forward(carrier: dict[str, Any], depth: int, family: str, audit: dict[str, int], reorder: bool = False) -> tuple[list[dict[str, int]], list[tuple[Any, ...]]]:
    require(carrier["stage"] == "IDLE", "non-idle forward")
    f = carrier["field"]
    records, executed = [], []
    for index in range(depth):
        stages = descriptors(index, family, f.q)
        if reorder:
            stages = tuple(reversed(stages))
        for operation, row, column in stages:
            if operation == "CHIRP":
                overwrite(carrier["b"], chirp(carrier["b"], row, column, f))
            elif operation == "CONJUGATE":
                overwrite(carrier["a"], conjugate(carrier["a"], carrier["b"], False, f, audit))
            else:
                overwrite(carrier["a"], cubic(carrier["a"], row, column, f, audit))
            executed.append((operation, row, column))
        if index + 1 in CHECKPOINTS:
            records.append(checkpoint(carrier, index + 1))
    carrier["stage"] = "FORWARD_COMPLETE"
    return records, executed


def stages_reverse(carrier: dict[str, Any], executed: list[tuple[Any, ...]], audit: dict[str, int], mutation: str | None = None) -> None:
    f = carrier["field"]
    sequence = list(reversed(executed))
    if mutation == "MISSING":
        sequence = sequence[1:]
    elif mutation == "REORDER":
        sequence = list(reversed(sequence))
    for position, (operation, row, column) in enumerate(sequence):
        wrong = mutation == "WRONG" and position == 0
        if operation == "CHIRP":
            overwrite(carrier["b"], chirp(carrier["b"], -row + int(wrong), -column, f))
        elif operation == "CONJUGATE":
            updated = conjugate(carrier["a"], carrier["b"], True, f, audit)
            if wrong:
                key = next(iter(updated[2]))
                updated[2][key] = (updated[2][key] + 1) % f.p
            overwrite(carrier["a"], updated)
        else:
            overwrite(carrier["a"], cubic(carrier["a"], -row + int(wrong), -column, f, audit))
    carrier["stage"] = "IDLE"


def displaced_entry(item: list[Any], key: tuple[int, int], x: int, y: int, f: Field) -> int:
    r, s = key
    return phase(f, r * s * pow(2, -1, f.q) + s * x) * kernel(tuple(item[0]), (x + r) % f.q, y, f) % f.p


def kentry(item: list[Any], x: int, y: int, f: Field) -> int:
    return sum(value * displaced_entry(item, key, x, y, f) for key, value in item[2].items()) % f.p


def public_probes(q: int, family: str) -> tuple[tuple[int, int, int, int, int], ...]:
    code = 1 if family == "PRIMARY" else 2
    return ((0, 1, (3 * code + 1) % q, (5 * code + 2) % q, 1), (1, 0, (7 * code + 2) % q, (11 * code + 3) % q, 2), (0, 0, (13 * code + 1) % q, (17 * code + 4) % q, 3), (1, 1, (19 * code + 2) % q, (23 * code + 5) % q, 5))


def boundary(carrier: dict[str, Any], family: str) -> int:
    require(carrier["stage"] == "FORWARD_COMPLETE", "premature projection")
    f, item = carrier["field"], carrier["a"]
    return sum(weight * item[1][2 * source + target] * kentry(item, x, y, f) for source, target, x, y, weight in public_probes(f.q, family)) % f.p


def digest(carrier: dict[str, Any]) -> str:
    f = carrier["field"]
    result = hashlib.sha256()
    result.update(repr(canon(carrier["a"], f)).encode("ascii"))
    result.update(repr(canon(carrier["b"], f)).encode("ascii"))
    return result.hexdigest()


def vector_gaussian(vector: list[int], symplectic: tuple[int, ...], coefficient: int, f: Field) -> list[int]:
    return [coefficient * sum(kernel(symplectic, x, y, f) * vector[y] for y in range(f.q)) % f.p for x in range(f.q)]


def vector_cubic(vector: list[int], strength: int, f: Field) -> list[int]:
    return [value * phase(f, strength * x**3) % f.p for x, value in enumerate(vector)]


def q_vector_boundary(q: int, depth: int, family: str, audit: dict[str, int]) -> int:
    seed = seal(q)
    f = seed["field"]
    resident = relation(tuple(seed["b"][0]), tuple(seed["b"][1]), dict(seed["b"][2]))
    left_ops, right_ops = [], []
    fiber = tuple(seed["a"][1])
    for index in range(depth):
        for operation, row, column in descriptors(index, family, q):
            if operation == "CHIRP":
                resident = chirp(resident, row, column, f)
            elif operation == "CONJUGATE":
                inverse = gaussian_inverse(resident, f, audit)
                left_ops.append(("G", (tuple(resident[0]), resident[2][(0, 0)])))
                right_ops.append(("G", (tuple(inverse[0]), inverse[2][(0, 0)])))
                fiber = mm(mm(tuple(resident[1]), fiber, f.p), tuple(inverse[1]), f.p)
            else:
                left_ops.append(("C", row))
                right_ops.append(("C", column))
    total = 0
    for source, target, x, y, weight in public_probes(q, family):
        vector = [0] * q
        vector[y] = 1
        for operation, payload in reversed(right_ops):
            vector = vector_gaussian(vector, payload[0], payload[1], f) if operation == "G" else vector_cubic(vector, payload, f)
        vector = vector_gaussian(vector, tuple(seed["a"][0]), 1, f)
        for operation, payload in left_ops:
            vector = vector_gaussian(vector, payload[0], payload[1], f) if operation == "G" else vector_cubic(vector, payload, f)
        total = (total + weight * fiber[2 * source + target] * vector[x]) % f.p
    return total


def execute(carrier: dict[str, Any], depth: int, family: str, audit: dict[str, int], mutation: str | None = None, reorder: bool = False) -> dict[str, Any]:
    before, ids = state(carrier), backing(carrier)
    records, executed = stages_forward(carrier, depth, family, audit, reorder)
    projected, commitment, final_point = boundary(carrier, family), digest(carrier), checkpoint(carrier, depth)
    stages_reverse(carrier, executed, audit, mutation)
    restored = state(carrier) == before
    same_backing = backing(carrier) == ids
    if mutation is None:
        require(restored and same_backing, "independent restoration failure")
        carrier["generation"] += 1
    return {"boundary": projected, "semantic_commitment": commitment, "checkpoints": records, "final_checkpoint": final_point, "restored": restored, "same_backing": same_backing, "generation": carrier["generation"]}


def dense(item: list[Any], f: Field) -> list[list[int]]:
    return [[kentry(item, x, y, f) for y in range(f.q)] for x in range(f.q)]


def dense_mm(left: list[list[int]], right: list[list[int]], p: int) -> list[list[int]]:
    q = len(left)
    return [[sum(left[x][z] * right[z][y] for z in range(q)) % p for y in range(q)] for x in range(q)]


def dense_diag(strength: int, f: Field, degree: int) -> list[list[int]]:
    return [[phase(f, strength * x**degree) if x == y else 0 for y in range(f.q)] for x in range(f.q)]


def dense_program_parity(q: int, depth: int, family: str) -> bool:
    carrier = seal(q)
    f = carrier["field"]
    component = seal(q)
    audit = {"trajectory_cocycle_parity_checks": 0}
    stages_forward(component, depth, family, audit)
    a_dense, b_dense = dense(carrier["a"], f), dense(carrier["b"], f)
    for index in range(depth):
        for operation, row, column in descriptors(index, family, q):
            if operation == "CHIRP":
                b_dense = dense_mm(dense_diag(row, f, 2), dense_mm(b_dense, dense_diag(column, f, 2), f.p), f.p)
            elif operation == "CONJUGATE":
                b_relation = relation((1, 0, 0, 1), (1, 0, 0, 1), {(0, 0): 1})
                # Reconstruct current B chart separately from its dense matrix.
                b_chart = seal(q)["b"]
                for prior in range(index + (1 if family == "PRIMARY" else 0)):
                    chirp_stage = next(stage for stage in descriptors(prior, family, q) if stage[0] == "CHIRP")
                    b_chart = chirp(b_chart, chirp_stage[1], chirp_stage[2], f)
                b_inverse = dense(gaussian_inverse(b_chart, f, audit), f)
                a_dense = dense_mm(b_dense, dense_mm(a_dense, b_inverse, f.p), f.p)
            else:
                a_dense = dense_mm(dense_diag(row, f, 3), dense_mm(a_dense, dense_diag(column, f, 3), f.p), f.p)
    return a_dense == dense(component["a"], f) and b_dense == dense(component["b"], f)


def dense_algebra_checks() -> dict[str, bool]:
    f = field(5)
    elements = tuple((a, b, c, d) for a in range(5) for b in range(5) for c in range(5) for d in range(5) if (a * d - b * c) % 5 == 1)
    covariance_ok = True
    identity = (1, 0, 0, 1)
    audit = {"trajectory_cocycle_parity_checks": 0}
    for symplectic in elements:
        base = relation(symplectic, identity, {(0, 0): 1})
        for r in range(5):
            for s in range(5):
                displacement = relation(identity, identity, {(r, s): 1})
                if dense(compose(base, displacement, f, audit), f) != dense_mm(dense(base, f), dense(displacement, f), f.p):
                    covariance_ok = False
                    break
    cubic_ok = True
    for q in (5, 11):
        qf = field(q)
        for strength in range(q):
            operator = relation(identity, identity, cubic_coefficients(strength, qf))
            if dense(operator, qf) != dense_diag(strength, qf, 3):
                cubic_ok = False
                break
    return {
        "all_120_sl2_q5_elements_enumerated": len(elements) == 120,
        "all_3000_q5_gaussian_displacement_products_match_dense_semantics": covariance_ok,
        "all_q5_q11_cubic_spectra_reconstruct_exact_diagonals": cubic_ok,
        "q5_depth2_full_program_component_dense_parity": dense_program_parity(5, 2, "PRIMARY"),
        "q11_depth1_full_program_component_dense_parity": dense_program_parity(11, 1, "PRIMARY"),
    }


def compare_cases(production: dict[str, Any], audit: dict[str, int]) -> list[dict[str, Any]]:
    indexed = {(case["q"], case["depth"], case["family"]): case for case in production["cases"]}
    output = []
    for q, depth, family in CASES:
        observed = execute(seal(q), depth, family, audit)
        baseline = q_vector_boundary(q, depth, family, audit)
        package = indexed[(q, depth, family)]
        checks = {
            "boundary": observed["boundary"] == package["boundary"] == baseline,
            "semantic_commitment": observed["semantic_commitment"] == package["semantic_commitment"],
            "checkpoints": observed["checkpoints"] == package["checkpoints"],
            "final_checkpoint": observed["final_checkpoint"] == package["final_checkpoint"],
            "restoration": observed["restored"] and package["exact_canonical_state_restored"],
            "same_backing": observed["same_backing"] and package["same_backing_restored"],
            "generation": observed["generation"] == package["restoration_generation"],
        }
        require(all(checks.values()), f"case mismatch q={q} depth={depth} family={family}: {checks}")
        output.append({"q": q, "depth": depth, "family": family, "checks": checks})
    return output


def controls(audit: dict[str, int]) -> dict[str, bool]:
    failures = {}
    for mutation in ("MISSING", "WRONG", "REORDER"):
        failures[mutation] = not execute(seal(11), 1, "PRIMARY", audit, mutation)["restored"]
    normal = execute(seal(11), 1, "PRIMARY", audit)
    altered = execute(seal(11), 1, "PRIMARY", audit, reorder=True)
    idle = seal(11)
    premature = False
    try:
        boundary(idle, "PRIMARY")
    except AssertionError:
        premature = True
    null_source = seal(11)["a"]
    null_result = cubic(null_source, 0, 0, idle["field"], audit)
    invalid_family = False
    try:
        seal(11, "NULL")
    except AssertionError:
        invalid_family = True
    invalid_order = False
    try:
        seal(17)
    except AssertionError:
        invalid_order = True
    result = {
        "missing_inverse_fails": failures["MISSING"],
        "wrong_inverse_fails": failures["WRONG"],
        "reordered_inverse_fails": failures["REORDER"],
        "module_reorder_changes_boundary": normal["boundary"] != altered["boundary"],
        "premature_projection_rejected": premature,
        "null_cubic_is_identity": canon(null_source, idle["field"]) == canon(null_result, idle["field"]),
        "null_family_rejected": invalid_family,
        "non_safe_prime_rejected": invalid_order,
    }
    require(all(result.values()), f"control failure: {result}")
    return result


def reuse(production: dict[str, Any], audit: dict[str, int]) -> dict[str, bool]:
    carrier = seal(23)
    initial, ids = state(carrier), backing(carrier)
    first = execute(carrier, 1, "PRIMARY", audit)
    second = execute(carrier, 2, "ALTERNATE", audit)
    fresh = execute(seal(23), 2, "ALTERNATE", audit)
    result = {
        "generation_one": first["generation"] == 1,
        "generation_two": second["generation"] == 2,
        "same_backing": backing(carrier) == ids,
        "exact_state": state(carrier) == initial,
        "boundary_matches_fresh": second["boundary"] == fresh["boundary"],
        "commitment_matches_fresh": second["semantic_commitment"] == fresh["semantic_commitment"],
        "production_second_boundary": second["boundary"] == production["restoration_and_reuse"]["second_boundary"],
        "production_no_snapshot": production["restoration_and_reuse"]["snapshot_used"] is False,
    }
    require(all(result.values()), f"reuse failure: {result}")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(production_path: Path) -> dict[str, Any]:
    production = json.loads(production_path.read_text(encoding="utf-8"))
    audit = {"trajectory_cocycle_parity_checks": 0}
    comparisons = compare_cases(production, audit)
    control_results = controls(audit)
    reuse_results = reuse(production, audit)
    algebra = dense_algebra_checks()
    require(all(algebra.values()), f"dense algebra failure: {algebra}")
    first_supports = [case["checkpoints"][0]["deformed_port_gaussian_component_support"] for case in production["cases"]]
    require(all(support >= (case["q"] - 3) ** 2 for support, case in zip(first_supports, production["cases"])), "quadratic lower-bound failure")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_INDEPENDENT_ORACLE_V1",
        "production_result": "GROWING_PRIME_TWO_FIBER_CUBIC_DEFORMED_WEIL_COMPONENT_RANK_RESULTS.json",
        "production_result_sha256": sha256(production_path),
        "independence": {
            "imports_production_module": False,
            "imports_gaussian_predecessor_module": False,
            "separate_component_recurrence": True,
            "separate_q_vector_boundary_recurrence": True,
            "production_projection_or_inverse_called": False,
            "production_result_used_only_as_comparison_target": True,
        },
        "case_comparisons": comparisons,
        "all_11_cases_reconstructed": len(comparisons) == 11,
        "trajectory_cocycle_parity_checks": audit["trajectory_cocycle_parity_checks"],
        "dense_semantic_checks": algebra,
        "controls": control_results,
        "restoration_and_reuse": reuse_results,
        "observed_resource_law": {
            "first_cubic_component_supports": first_supports,
            "all_first_supports_at_least_q_minus_3_squared": True,
            "accepted_component_coordinates_are_quadratic_in_q": True,
            "matched_q_vector_dynamic_state": "2*Q_FIELD_CELLS",
            "matched_public_word_plan": "12*DEPTH_CELLS",
            "fixed_bit_width_across_unbounded_q": False,
        },
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "EXACT_WEIL_WEYL_COMPONENT_COMPOSITION",
            "EXACT_SEPARABLE_CUBIC_PHASE_INTERSECTION",
            "Q_MINUS3_SQUARED_FIRST_LAYER_COMPONENT_LOWER_BOUND_ON_DECLARED_ORDERS",
            "EXACT_IN_PLACE_RESTORATION_AND_SAME_BACKING_REUSE",
            "EXECUTED_LINEAR_Q_DYNAMIC_STATE_FINAL_BOUNDARY_BASELINE",
        ],
        "rejected_interpretations": production["not_established"],
        "qualified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build(args.production), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
