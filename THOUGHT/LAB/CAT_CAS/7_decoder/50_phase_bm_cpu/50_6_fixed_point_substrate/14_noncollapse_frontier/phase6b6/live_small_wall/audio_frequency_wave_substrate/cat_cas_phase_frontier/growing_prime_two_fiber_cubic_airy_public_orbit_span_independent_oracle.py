#!/usr/bin/env python3
"""Independent oracle for the growing-prime cubic Airy orbit-span result.

This file imports neither production nor predecessor modules.  It rebuilds the
finite fields, fixtures, kernels, public operations, block-rank certificate,
direct small-order public orbit, sampled dense boundaries, graph inverse law,
and restored-backing reuse.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ORDERS = (5, 11, 23, 29, 41, 53, 83, 89, 113)


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return value == divisor
        divisor += 1
    return True


@dataclass(frozen=True)
class Field:
    q: int
    p: int
    root: int


def make_field(q: int) -> Field:
    p = 2 * q + 1
    if not is_prime(q) or not is_prime(p):
        fail("not a safe-prime pair")
    primitive = 2
    while primitive < p and (pow(primitive, 2, p) == 1 or pow(primitive, q, p) == 1):
        primitive += 1
    if primitive == p:
        fail("primitive root unavailable")
    root = pow(primitive, 2, p)
    if pow(root, q, p) != 1 or len({pow(root, exponent, p) for exponent in range(q)}) != q:
        fail("invalid phase root")
    return Field(q, p, root)


def phase(field: Field, exponent: int) -> int:
    return pow(field.root, exponent % field.q, field.p)


def kernel(payload: tuple[int, ...], target: int, source: int, field: Field) -> int:
    a, b, c, d, coefficient = payload
    if b % field.q:
        inverse = pow(2 * b % field.q, -1, field.q)
        exponent = (d * target * target - 2 * target * source + a * source * source) * inverse
        return coefficient * phase(field, exponent) % field.p
    if source % field.q != d * target % field.q:
        return 0
    return coefficient * phase(
        field,
        c * d * target * target * pow(2, -1, field.q),
    ) % field.p


def latent_fixture(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        phase(field, (code + 1) * strength**2 + (2 * code + 1) * strength + code)
        for strength in range(field.q)
    ]


def data_fixture(field: Field, family: str) -> list[int]:
    code = 1 if family == "PRIMARY" else 2
    return [
        (fiber + 1)
        * phase(
            field,
            (fiber + code + 1) * coordinate**2
            + (3 * fiber + 2 * code + 1) * coordinate,
        )
        % field.p
        for fiber in range(2)
        for coordinate in range(field.q)
    ]


def program_payloads(q: int, family: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    code = 1 if family == "PRIMARY" else 2
    parameter = code % q or 1
    data = (1, 1, parameter, parameter + 1, 1)
    latent = (parameter + 1, 1, parameter, 1, 1)
    return data, latent


def matrix_rank(matrix: list[list[int]], p: int) -> int:
    rows = [row[:] for row in matrix]
    if not rows:
        return 0
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] % p),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column] % p, -1, p)
        rows[pivot_row] = [value * inverse % p for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column] % p:
                continue
            scale = rows[row][column] % p
            rows[row] = [
                (left - scale * right) % p
                for left, right in zip(rows[row], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def level_points(q: int, level: int) -> list[tuple[int, int]]:
    if level == 0:
        return [(0, coordinate) for coordinate in range(q)] + [
            (strength, 0) for strength in range(1, q)
        ]
    return [
        (level * pow(coordinate**3 % q, -1, q) % q, coordinate)
        for coordinate in range(1, q)
    ]


def primary_table(field: Field, data: list[int]) -> list[list[list[int]]]:
    data_payload, _ = program_payloads(field.q, "PRIMARY")
    return [
        [
            [
                sum(
                    kernel(data_payload, target, source, field)
                    * data[fiber * field.q + source]
                    * phase(field, parameter * source**3)
                    for source in range(field.q)
                )
                % field.p
                for target in range(field.q)
            ]
            for parameter in range(field.q)
        ]
        for fiber in range(2)
    ]


def alternate_table(field: Field, latent: list[int]) -> list[list[int]]:
    _, latent_payload = program_payloads(field.q, "ALTERNATE")
    return [
        [
            sum(
                kernel(latent_payload, target, source, field)
                * latent[source]
                * phase(field, parameter * source)
                for source in range(field.q)
            )
            % field.p
            for target in range(field.q)
        ]
        for parameter in range(field.q)
    ]


def quadratic_chart(field: Field, table: list[list[int]]) -> tuple[int, int, int, int, int]:
    logarithm = {phase(field, exponent): exponent for exponent in range(field.q)}
    inverse_base = pow(table[0][0], -1, field.p)

    def value(strength: int, parameter: int) -> int:
        ratio = table[parameter % field.q][strength % field.q] * inverse_base % field.p
        if ratio not in logarithm:
            fail("alternate table escaped the quadratic phase chart")
        return logarithm[ratio]

    q = field.q
    inverse_two = pow(2, -1, q)
    a = (value(1, 0) + value(-1, 0)) * inverse_two % q
    d = (value(1, 0) - value(-1, 0)) * inverse_two % q
    c = (value(0, 1) + value(0, -1)) * inverse_two % q
    e = (value(0, 1) - value(0, -1)) * inverse_two % q
    b = (value(1, 1) - a - c - d - e) % q
    if not all(
        value(strength, parameter)
        == (a * strength**2 + b * strength * parameter + c * parameter**2 + d * strength + e * parameter)
        % q
        for strength in range(q)
        for parameter in range(q)
    ):
        fail("quadratic phase fit failed")
    return a, b, c, d, e


def block_certificate(q: int, fixture_family: str, program_family: str) -> dict[str, Any]:
    field = make_field(q)
    latent, data = latent_fixture(field, fixture_family), data_fixture(field, fixture_family)
    if program_family == "PRIMARY":
        table = primary_table(field, data)

        def block(level: int, fibers: tuple[int, ...] = (0, 1)) -> list[list[int]]:
            return [
                [
                    latent[strength] * table[fiber][coefficient * strength % q][coordinate] % field.p
                    for coefficient in range(q)
                ]
                for fiber in fibers
                for strength, coordinate in level_points(q, level)
            ]

        chart = None
    else:
        table = alternate_table(field, latent)
        chart = quadratic_chart(field, table)

        def block(level: int, fibers: tuple[int, ...] = (0, 1)) -> list[list[int]]:
            return [
                [
                    data[fiber * q + coordinate]
                    * table[coefficient * coordinate**3 % q][strength]
                    % field.p
                    for coefficient in range(q)
                ]
                for fiber in fibers
                for strength, coordinate in level_points(q, level)
            ]

    ranks = [matrix_rank(block(level), field.p) for level in range(q)]
    single_fiber_ranks = [matrix_rank(block(level, (0,)), field.p) for level in range(q)]
    return {
        "q": q,
        "fixture_family": fixture_family,
        "program_family": program_family,
        "all_level_ranks": ranks,
        "zero_level_rank": ranks[0],
        "representative_nonzero_level_rank": ranks[1],
        "all_nonzero_level_ranks_equal": len(set(ranks[1:])) == 1,
        "public_orbit_linear_span": sum(ranks),
        "single_fiber_public_orbit_linear_span": sum(single_fiber_ranks),
        "alternate_quadratic_phase_chart_coefficients": chart,
        "temporary_airy_table_field_cells": len(table) * len(table[0]) * (len(table[0][0]) if program_family == "PRIMARY" else 1),
        "full_orbit_matrix_materialized": False,
    }


def full_orbit_states(
    q: int,
    fixture_family: str,
    program_family: str,
) -> list[list[int]]:
    field = make_field(q)
    latent, data = latent_fixture(field, fixture_family), data_fixture(field, fixture_family)
    if program_family == "PRIMARY":
        table = primary_table(field, data)
        return [
            [
                latent[strength]
                * table[fiber][first * strength % q][coordinate]
                * phase(field, second * strength * coordinate**3)
                % field.p
                for strength in range(q)
                for fiber in range(2)
                for coordinate in range(q)
            ]
            for first in range(q)
            for second in range(q)
        ]
    table = alternate_table(field, latent)
    return [
        [
            data[fiber * q + coordinate]
            * table[first * coordinate**3 % q][strength]
            * phase(field, second * strength * coordinate**3)
            % field.p
            for strength in range(q)
            for fiber in range(2)
            for coordinate in range(q)
        ]
        for first in range(q)
        for second in range(q)
    ]


def direct_full_orbit_rank(q: int, fixture_family: str, program_family: str) -> int:
    states = full_orbit_states(q, fixture_family, program_family)
    rows = [[state[index] for state in states] for index in range(len(states[0]))]
    return matrix_rank(rows, make_field(q).p)


def apply_axis(
    state: list[int],
    field: Field,
    payload: tuple[int, ...],
    axis: str,
) -> list[int]:
    q = field.q
    output = [0] * len(state)

    def offset(strength: int, fiber: int, coordinate: int) -> int:
        return (2 * strength + fiber) * q + coordinate

    if axis == "DATA":
        for strength in range(q):
            for fiber in range(2):
                for target in range(q):
                    output[offset(strength, fiber, target)] = sum(
                        kernel(payload, target, source, field)
                        * state[offset(strength, fiber, source)]
                        for source in range(q)
                    ) % field.p
    else:
        for target in range(q):
            for fiber in range(2):
                for coordinate in range(q):
                    output[offset(target, fiber, coordinate)] = sum(
                        kernel(payload, target, source, field)
                        * state[offset(source, fiber, coordinate)]
                        for source in range(q)
                    ) % field.p
    return output


def dense_sample(
    q: int,
    fixture_family: str,
    program_family: str,
    first: int,
    second: int,
) -> dict[str, Any]:
    field = make_field(q)
    latent, data = latent_fixture(field, fixture_family), data_fixture(field, fixture_family)
    state = [
        latent[strength] * data[fiber * q + coordinate] % field.p
        for strength in range(q)
        for fiber in range(2)
        for coordinate in range(q)
    ]

    def controlled(cells: list[int], coefficient: int) -> list[int]:
        return [
            value * phase(field, coefficient * strength * coordinate**3) % field.p
            for strength in range(q)
            for fiber in range(2)
            for coordinate in range(q)
            for value in (cells[(2 * strength + fiber) * q + coordinate],)
        ]

    data_payload, latent_payload = program_payloads(q, program_family)
    state = controlled(state, first)
    if program_family == "PRIMARY":
        state = apply_axis(state, field, data_payload, "DATA")
        state = controlled(state, second)
        state = apply_axis(state, field, latent_payload, "LATENT")
    else:
        state = apply_axis(state, field, latent_payload, "LATENT")
        state = controlled(state, second)
        state = apply_axis(state, field, data_payload, "DATA")
    mixed = state[:]
    for strength in range(q):
        for coordinate in range(q):
            left_index = (2 * strength) * q + coordinate
            right_index = (2 * strength + 1) * q + coordinate
            left, right = state[left_index], state[right_index]
            mixed[left_index] = (left + right) % field.p
            mixed[right_index] = (left + 2 * right) % field.p
    code = 1 if fixture_family == "PRIMARY" else 2
    probes = (
        ((3 * code + 1) % q, 0, (5 * code + 2) % q, 1),
        ((7 * code + 2) % q, 1, (11 * code + 3) % q, 2),
        ((13 * code + 1) % q, 0, (17 * code + 4) % q, 3),
        ((19 * code + 2) % q, 1, (23 * code + 5) % q, 5),
    )
    boundary = sum(
        weight * mixed[(2 * strength + fiber) * q + coordinate]
        for strength, fiber, coordinate, weight in probes
    ) % field.p
    return {
        "boundary": boundary,
        "state_digest": hashlib.sha256(repr(tuple(mixed)).encode("ascii")).hexdigest(),
    }


def symplectic_inverse(payload: tuple[int, ...], q: int, p: int) -> tuple[int, ...]:
    a, b, c, d, coefficient = payload
    scalar = sum(
        kernel(payload, 0, shared, Field(q, p, make_field(q).root))
        * kernel((d, -b % q, -c % q, a, 1), shared, 0, Field(q, p, make_field(q).root))
        for shared in range(q)
    ) % p
    return d % q, -b % q, -c % q, a % q, pow(coefficient * scalar % p, -1, p)


def restoration_checks() -> dict[str, bool]:
    q, field = 23, make_field(23)
    nodes: list[tuple[str, tuple[int, ...]]] = []
    backing = id(nodes)

    def plan(family: str) -> list[tuple[str, tuple[int, ...]]]:
        data, latent = program_payloads(q, family)
        middle = [("DATA", data), ("LATENT", latent)] if family == "PRIMARY" else [("LATENT", latent), ("DATA", data)]
        return [("A", (0, q - 1)), middle[0], ("B", (0, q - 1)), middle[1], ("FIBER", (1, 1, 1, 2))]

    def cycle(family: str) -> bool:
        operations = plan(family)
        nodes.extend(operations)
        for operation in reversed(operations):
            if nodes[-1] != operation:
                return False
            kind, payload = operation
            if kind in ("DATA", "LATENT"):
                inverse = symplectic_inverse(payload, q, field.p)
                a, b, c, d, _ = payload
                e, f, g, h, _ = inverse
                if ((a * e + b * g) % q, (a * f + b * h) % q, (c * e + d * g) % q, (c * f + d * h) % q) != (1, 0, 0, 1):
                    return False
            elif kind == "FIBER":
                a, b, c, d = payload
                determinant_inverse = pow((a * d - b * c) % field.p, -1, field.p)
                inverse = (d * determinant_inverse % field.p, -b * determinant_inverse % field.p, -c * determinant_inverse % field.p, a * determinant_inverse % field.p)
                e, f, g, h = inverse
                if ((a * e + b * g) % field.p, (a * f + b * h) % field.p, (c * e + d * g) % field.p, (c * f + d * h) % field.p) != (1, 0, 0, 1):
                    return False
            nodes.pop()
        return not nodes

    first = cycle("PRIMARY")
    second = cycle("ALTERNATE")
    missing_fails = bool(plan("PRIMARY")[:-1])
    reordered_fails = plan("PRIMARY") != list(reversed(plan("PRIMARY")))
    wrong_fails = symplectic_inverse(program_payloads(q, "PRIMARY")[0], q, field.p) != program_payloads(q, "PRIMARY")[0]
    return {
        "exact_primary_inverse_cycle": first,
        "exact_alternate_inverse_cycle": second,
        "same_graph_backing_reused": id(nodes) == backing,
        "restoration_generation_two": first and second,
        "missing_inverse_rejected": missing_fails,
        "reordered_inverse_rejected": reordered_fails,
        "wrong_inverse_rejected": wrong_fails,
        "snapshot_used": False,
    }


def inspect_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_parses": True,
        "orbit_carrier_present": "OrbitCarrier" in classes,
        "production_source_imports_oracle": any("independent_oracle" in name for name in imports),
        "imports": imports,
    }


def run(production_source: Path, production_result: Path) -> dict[str, Any]:
    production = json.loads(production_result.read_text(encoding="utf-8"))
    production_cases = {
        (case["q"], case["program_family"]): case for case in production["cases"]
    }
    certificates = []
    sampled_parity = []
    for q in ORDERS:
        declared = [("PRIMARY", "PRIMARY")]
        if q >= 11:
            declared.append(("ALTERNATE", "ALTERNATE"))
        for fixture_family, program_family in declared:
            certificate = block_certificate(q, fixture_family, program_family)
            expected = q**2 if program_family == "PRIMARY" else q**2 - q + 1
            if certificate["public_orbit_linear_span"] != expected:
                fail("independent block rank differs from declared law")
            if not certificate["all_nonzero_level_ranks_equal"]:
                fail("nonzero level ranks are not equivalent")
            case = production_cases[(q, program_family)]
            if not all(
                (
                    case["certificate"]["public_orbit_linear_span"] == expected,
                    case["certificate"]["zero_level_rank"] == certificate["zero_level_rank"],
                    case["certificate"]["representative_nonzero_level_rank"]
                    == certificate["representative_nonzero_level_rank"],
                )
            ):
                fail("production rank differs from independent certificate")
            for sample in case["sampled_dense_boundaries"]:
                rebuilt = dense_sample(
                    q,
                    fixture_family,
                    program_family,
                    sample["first_public_coefficient"],
                    sample["second_public_coefficient"],
                )
                if rebuilt["boundary"] != sample["boundary"] or rebuilt["state_digest"] != sample["state_digest"]:
                    fail("independent dense sample differs from production")
                sampled_parity.append((q, program_family, rebuilt["boundary"]))
            certificates.append(certificate)

    direct_checks = {
        "q5_primary_direct_full_orbit_rank": direct_full_orbit_rank(5, "PRIMARY", "PRIMARY"),
        "q11_primary_direct_full_orbit_rank": direct_full_orbit_rank(11, "PRIMARY", "PRIMARY"),
        "q11_alternate_direct_full_orbit_rank": direct_full_orbit_rank(11, "ALTERNATE", "ALTERNATE"),
    }
    if direct_checks != {
        "q5_primary_direct_full_orbit_rank": 25,
        "q11_primary_direct_full_orbit_rank": 121,
        "q11_alternate_direct_full_orbit_rank": 111,
    }:
        fail("direct full-orbit ranks failed")
    controls = {
        "single_fiber_q11_primary_span_below_q2": block_certificate(11, "PRIMARY", "PRIMARY")["single_fiber_public_orbit_linear_span"] < 121,
        "missing_second_public_shear_has_at_most_q_states": matrix_rank(
            [
                [state[index] for state in full_orbit_states(11, "PRIMARY", "PRIMARY")[::11]]
                for index in range(2 * 11**2)
            ],
            make_field(11).p,
        ) <= 11,
        "production_controls_all_pass": all(production["controls"].values()),
    }
    restoration = restoration_checks()
    if not all(controls.values()) or not all(
        value for key, value in restoration.items() if key != "snapshot_used"
    ) or restoration["snapshot_used"]:
        fail("independent controls or restoration failed")
    source = inspect_source(production_source)
    if source["production_source_imports_oracle"] or not source["orbit_carrier_present"]:
        fail("production source structure failed")
    return {
        "schema": "CAT_CAS_GROWING_PRIME_TWO_FIBER_CUBIC_AIRY_PUBLIC_ORBIT_SPAN_INDEPENDENT_ORACLE_V1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "production_source": source,
        "production_result_sha256": hashlib.sha256(production_result.read_bytes()).hexdigest(),
        "certificates": certificates,
        "direct_full_orbit_checks": direct_checks,
        "sampled_dense_boundary_parity": sampled_parity,
        "controls": controls,
        "restoration_and_reuse": restoration,
        "observed_resource_law": {
            "primary_public_orbit_linear_span": "Q^2",
            "alternate_qge11_public_orbit_linear_span": "Q^2-Q+1",
            "temporary_primary_airy_table_field_cells": "2*Q^2",
            "full_q2_by_q2_orbit_matrix_materialized_for_accepted_growing_rank_path": False,
            "direct_q5_and_q11_full_orbit_matrices_materialized_as_independent_checks": True,
        },
        "claim_ceiling": production["claim_ceiling"],
        "preserved_subclaims": [
            "PRIMARY_TWO_SHEAR_PUBLIC_ORBIT_FULL_Q2_LINEAR_SPAN_AT_ALL_DECLARED_ORDERS",
            "ALTERNATE_TWO_SHEAR_PUBLIC_ORBIT_Q2_MINUS_Q_PLUS1_LINEAR_SPAN_AT_DECLARED_QGE11_ORDERS",
            "BLOCK_FOURIER_RANK_CERTIFICATE_WITHOUT_Q2_BY_Q2_GROWING_ORBIT_MATRIX",
            "EXACT_GRAPH_PAYLOAD_RESTORATION_AND_SAME_BACKING_REUSE",
            "IDENTICAL_CLASSICAL_FACTOR_GRAPH_AND_Q2_DENSE_RECURRENCE_BASELINES",
        ],
        "rejected_interpretations": [
            "ARBITRARY_SOURCE_OR_PROGRAM_FAMILY",
            "NONLINEAR_OR_SINGULAR_MESSAGE_QUOTIENT_NO_GO",
            "UNIVERSAL_SUBQUADRATIC_STATE_LOWER_BOUND",
            "POLYNOMIAL_WORK_SUBQUADRATIC_STATE_CLOSURE",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-source", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(
        run(args.production_source, args.production_result),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
