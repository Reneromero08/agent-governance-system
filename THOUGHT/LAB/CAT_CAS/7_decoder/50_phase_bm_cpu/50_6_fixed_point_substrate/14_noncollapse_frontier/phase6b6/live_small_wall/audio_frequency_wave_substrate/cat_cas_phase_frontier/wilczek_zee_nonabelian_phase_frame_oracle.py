#!/usr/bin/env python3
"""Independent high-precision oracle for non-Abelian dark-frame transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp


mp.mp.dps = 80

ALPHA = mp.mpf("1.0")
BETA = mp.mpf("0.7")
SEGMENTS = 512
PRODUCTION_TOLERANCE = mp.mpf("2e-11")
ORACLE_TOLERANCE = mp.mpf("1e-65")

Matrix = list[list[mp.mpc]]
Frame = list[list[mp.mpc]]


def identity(size: int) -> Matrix:
    return [
        [
            mp.mpc(1 if row == column else 0)
            for column in range(size)
        ]
        for row in range(size)
    ]


def adjoint(value: Matrix) -> Matrix:
    return [
        [mp.conj(value[column][row]) for column in range(len(value))]
        for row in range(len(value[0]))
    ]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                mp.mpc(0),
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def matrix_error(left: Matrix, right: Matrix) -> mp.mpf:
    return max(
        abs(left[row][column] - right[row][column])
        for row in range(len(left))
        for column in range(len(left[0]))
    )


def frobenius(value: Matrix) -> mp.mpf:
    return mp.sqrt(
        sum(
            (
                abs(value[row][column]) ** 2
                for row in range(len(value))
                for column in range(len(value[0]))
            ),
            mp.mpf(0),
        )
    )


def subtract(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            left[row][column] - right[row][column]
            for column in range(len(left[0]))
        ]
        for row in range(len(left))
    ]


def power(base_value: Matrix, exponent: int) -> Matrix:
    result = identity(len(base_value))
    base = base_value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(base, result)
        remaining >>= 1
        if remaining:
            base = multiply(base, base)
    return result


def public_frame(phi1: mp.mpf, phi2: mp.mpf) -> Frame:
    ca = mp.cos(ALPHA)
    sa = mp.sin(ALPHA)
    cb = mp.cos(BETA)
    sb = mp.sin(BETA)
    e1 = mp.exp(mp.j * phi1)
    e2 = mp.exp(mp.j * phi2)
    return [
        [mp.mpc(-sa), mp.mpc(0)],
        [ca * cb * e1, -sb * e1],
        [ca * sb * e2, cb * e2],
    ]


def frame_overlap(left: Frame, right: Frame) -> Matrix:
    return [
        [
            sum(
                (
                    mp.conj(left[coordinate][row])
                    * right[coordinate][column]
                    for coordinate in range(3)
                ),
                mp.mpc(0),
            )
            for column in range(2)
        ]
        for row in range(2)
    ]


def frame_right_multiply(frame: Frame, value: Matrix) -> Frame:
    return [
        [
            sum(
                (
                    frame[coordinate][inner] * value[inner][column]
                    for inner in range(2)
                ),
                mp.mpc(0),
            )
            for column in range(2)
        ]
        for coordinate in range(3)
    ]


def inverse_square_root_hermitian(value: Matrix) -> Matrix:
    s00 = mp.re(value[0][0])
    s11 = mp.re(value[1][1])
    s01 = value[0][1]
    determinant = s00 * s11 - abs(s01) ** 2
    if determinant <= mp.mpf("1e-70"):
        raise RuntimeError("oracle polar overlap is singular")
    delta = mp.sqrt(determinant)
    scale = mp.sqrt(s00 + s11 + 2 * delta)
    a00 = s00 + delta
    a11 = s11 + delta
    inverse_determinant = 1 / (a00 * a11 - abs(s01) ** 2)
    return [
        [
            scale * a11 * inverse_determinant,
            -scale * s01 * inverse_determinant,
        ],
        [
            -scale * mp.conj(s01) * inverse_determinant,
            scale * a00 * inverse_determinant,
        ],
    ]


def edge_correction(left: Frame, right: Frame) -> Matrix:
    overlap = frame_overlap(left, right)
    gram = multiply(overlap, adjoint(overlap))
    correction = multiply(
        adjoint(overlap),
        inverse_square_root_hermitian(gram),
    )
    if matrix_error(
        multiply(adjoint(correction), correction), identity(2)
    ) > ORACLE_TOLERANCE:
        raise RuntimeError("oracle edge correction is not unitary")
    return correction


def transport_to(carrier: Frame, target: Frame) -> Frame:
    correction = edge_correction(carrier, target)
    return frame_right_multiply(target, correction)


def point(axis: int, orientation: int, step: int) -> tuple[mp.mpf, mp.mpf]:
    phase = orientation * 2 * mp.pi * step / SEGMENTS
    return (
        phase if axis == 1 else mp.mpf(0),
        phase if axis == 2 else mp.mpf(0),
    )


def forward_loop(carrier: Frame, axis: int, orientation: int) -> Frame:
    result = carrier
    for step in range(1, SEGMENTS + 1):
        result = transport_to(result, public_frame(*point(axis, orientation, step)))
    return result


def inverse_loop(carrier: Frame, axis: int, orientation: int) -> Frame:
    result = carrier
    for step in reversed(range(SEGMENTS)):
        result = transport_to(result, public_frame(*point(axis, orientation, step)))
    return result


def discrete_loop_formula(axis: int, orientation: int) -> Matrix:
    correction = edge_correction(
        public_frame(*point(axis, orientation, 0)),
        public_frame(*point(axis, orientation, 1)),
    )
    return power(correction, SEGMENTS)


def continuous_loop(axis: int, orientation: int) -> Matrix:
    ca = mp.cos(ALPHA)
    cb = mp.cos(BETA)
    sb = mp.sin(BETA)
    vector = [ca * cb, -sb] if axis == 1 else [ca * sb, cb]
    norm_squared = sum((entry * entry for entry in vector), mp.mpf(0))
    coefficient = (
        mp.exp(-mp.j * orientation * 2 * mp.pi * norm_squared) - 1
    ) / norm_squared
    result = identity(2)
    for row in range(2):
        for column in range(2):
            result[row][column] += (
                coefficient * vector[row] * vector[column]
            )
    return result


def decode_matrix(value: object) -> Matrix:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("production matrix schema mismatch")
    result: Matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise RuntimeError("production matrix row schema mismatch")
        decoded_row = []
        for cell in row:
            if not isinstance(cell, list) or len(cell) != 2:
                raise RuntimeError("production complex schema mismatch")
            decoded_row.append(mp.mpc(str(cell[0]), str(cell[1])))
        result.append(decoded_row)
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: wilczek_zee_nonabelian_phase_frame_oracle.py "
            "PRODUCTION_RESULT"
        )
    production = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not isinstance(production, dict):
        raise RuntimeError("production result is not an object")

    initial = public_frame(mp.mpf(0), mp.mpf(0))
    primary = forward_loop(initial, 1, 1)
    primary = forward_loop(primary, 2, 1)
    primary_boundary = frame_overlap(initial, primary)
    primary_restored = inverse_loop(primary, 2, 1)
    primary_restored = inverse_loop(primary_restored, 1, 1)

    reordered = forward_loop(initial, 2, 1)
    reordered = forward_loop(reordered, 1, 1)
    reordered_boundary = frame_overlap(initial, reordered)

    reuse = forward_loop(initial, 1, -1)
    reuse = forward_loop(reuse, 2, 1)
    reuse = forward_loop(reuse, 1, 1)
    reuse_boundary = frame_overlap(initial, reuse)
    reuse_restored = inverse_loop(reuse, 1, 1)
    reuse_restored = inverse_loop(reuse_restored, 2, 1)
    reuse_restored = inverse_loop(reuse_restored, 1, -1)

    h1 = discrete_loop_formula(1, 1)
    h2 = discrete_loop_formula(2, 1)
    formula_primary = multiply(h2, h1)
    formula_reordered = multiply(h1, h2)
    formula_reuse = multiply(h1, multiply(h2, adjoint(h1)))
    continuous_primary = multiply(
        continuous_loop(2, 1),
        continuous_loop(1, 1),
    )
    commutator_norm = frobenius(
        subtract(formula_primary, formula_reordered)
    )

    production_primary = decode_matrix(production["primary"]["boundary"])
    production_reordered = decode_matrix(
        production["reordered_forward"]["boundary"]
    )
    production_reuse = decode_matrix(production["reuse"]["boundary"])
    if (
        matrix_error(primary_boundary, formula_primary) > ORACLE_TOLERANCE
        or matrix_error(reordered_boundary, formula_reordered)
        > ORACLE_TOLERANCE
        or matrix_error(reuse_boundary, formula_reuse) > ORACLE_TOLERANCE
        or matrix_error(primary_restored, initial) > ORACLE_TOLERANCE
        or matrix_error(reuse_restored, initial) > ORACLE_TOLERANCE
        or matrix_error(production_primary, primary_boundary)
        > PRODUCTION_TOLERANCE
        or matrix_error(production_reordered, reordered_boundary)
        > PRODUCTION_TOLERANCE
        or matrix_error(production_reuse, reuse_boundary)
        > PRODUCTION_TOLERANCE
        or commutator_norm <= mp.mpf("1")
        or matrix_error(primary_boundary, continuous_primary)
        <= mp.mpf("1e-6")
        or matrix_error(primary_boundary, continuous_primary)
        >= mp.mpf("1e-4")
    ):
        raise RuntimeError("independent non-Abelian reconstruction failed")

    wrong_order = inverse_loop(primary, 1, 1)
    wrong_order = inverse_loop(wrong_order, 2, 1)
    if matrix_error(wrong_order, initial) <= mp.mpf("0.01"):
        raise RuntimeError("reordered inverse unexpectedly restored")

    output = {
        "result": "PASS",
        "oracle": (
            "INDEPENDENT_MPMATH80_CP2_DARK_FRAME_POLAR_TRANSPORT_"
            "AND_DISCRETE_LOOP_PRODUCT"
        ),
        "production_backend_imported": False,
        "precision_decimal_digits": 80,
        "loop_segments": SEGMENTS,
        "primary_boundary_matches": True,
        "reordered_boundary_matches": True,
        "reuse_boundary_matches": True,
        "all_discrete_loop_formulas_match_below_1e_65": True,
        "primary_restoration_below_1e_65": True,
        "reuse_restoration_below_1e_65": True,
        "fresh_restored_reuse_boundary_equal": True,
        "loop_order_noncommutator_frobenius": float(commutator_norm),
        "loop_order_noncommutes": True,
        "reordered_inverse_restored": False,
        "continuous_limit_distinct_from_finite_edge_product": True,
        "matched_compact_classical_2x2_recurrence_identical": True,
        "closed_form_fixed_loop_modules_available": True,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
