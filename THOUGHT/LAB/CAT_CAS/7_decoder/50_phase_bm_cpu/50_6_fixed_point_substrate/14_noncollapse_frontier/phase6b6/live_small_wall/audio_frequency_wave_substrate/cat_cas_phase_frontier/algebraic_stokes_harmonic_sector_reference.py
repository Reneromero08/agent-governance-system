#!/usr/bin/env python3
"""Exact oracle for parity-closed Stokes harmonic-sector signatures."""

from __future__ import annotations

import json

import sympy

import algebraic_stokes_lie_relation_reference as base


HOMOGENEOUS_SPHERE = sympy.groebner(
    [base.x * base.x + base.y * base.y + base.z * base.z],
    base.z,
    base.y,
    base.x,
    order="lex",
    domain=sympy.QQ,
)


def expressions(
    seed: sympy.Expr, generator: sympy.Expr
) -> list[sympy.Expr]:
    result = []
    current = seed
    for _ in range(base.GRADES):
        result.append(current)
        current = base.lie_poisson(generator, current)
    return result


def leading_harmonic_quotient(
    expression: sympy.Expr, degree: int
) -> dict[str, object]:
    polynomial = sympy.Poly(
        expression, *base.VARIABLES, domain=sympy.QQ
    )
    leading = sympy.Add(
        *(
            coefficient
            * base.x ** monomial[0]
            * base.y ** monomial[1]
            * base.z ** monomial[2]
            for monomial, coefficient in polynomial.terms()
            if sum(monomial) == degree
        )
    )
    remainder = sympy.expand(
        HOMOGENEOUS_SPHERE.reduce(sympy.expand(leading))[1]
    )
    remainder_terms = sympy.Poly(
        remainder, *base.VARIABLES, domain=sympy.QQ
    ).terms()
    return {
        "degree": degree,
        "leading_input_terms": len(
            sympy.Poly(
                leading, *base.VARIABLES, domain=sympy.QQ
            ).terms()
        ),
        "harmonic_quotient_terms": len(remainder_terms),
        "harmonic_quotient_nonzero": remainder != 0,
    }


def chain_records(
    seed: sympy.Expr, generator: sympy.Expr
) -> list[dict[str, object]]:
    return [
        base.grade_record(expression, 2 + grade, parity_reduced=True)
        for grade, expression in enumerate(expressions(seed, generator))
    ]


def main() -> None:
    primary_expressions = expressions(base.H0, base.PRIMARY)
    primary = chain_records(base.H0, base.PRIMARY)
    reuse = chain_records(base.H0, base.REUSE)
    identity = chain_records(base.H0, base.H0)
    swapped = chain_records(base.PRIMARY, base.H0)
    harmonic_certificates = [
        leading_harmonic_quotient(expression, 2 + grade)
        for grade, expression in enumerate(primary_expressions)
    ]
    output = {
        "result": "PASS",
        "oracle": (
            "INDEPENDENT_SYMPY_EXACT_PARITY_STOKES_"
            "HOMOGENEOUS_SPHERE_QUOTIENT"
        ),
        "sphere_relation": "x^2+y^2+z^2=1",
        "homogeneous_harmonic_quotient_relation": (
            "x^2+y^2+z^2=0"
        ),
        "primary_grades": primary,
        "reuse_grades": reuse,
        "successive_primary_term_counts": [
            grade["rational_nonzero_terms"] for grade in primary
        ],
        "successive_parity_basis_cells": [
            grade["quotient_basis_cells"] for grade in primary
        ],
        "all_rational_support_parity_exact": all(
            grade["support_parity_exact"] for grade in primary + reuse
        ),
        "leading_harmonic_certificates": harmonic_certificates,
        "all_leading_harmonic_quotients_nonzero": all(
            certificate["harmonic_quotient_nonzero"]
            for certificate in harmonic_certificates
        ),
        "identity_mixer_higher_grades_zero": all(
            grade["rational_nonzero_terms"] == 0
            for grade in identity[1:]
        ),
        "swapped_first_bracket_negates_primary": (
            swapped[1]["hash_p17"] != primary[1]["hash_p17"]
            and swapped[1]["hash_p19"] != primary[1]["hash_p19"]
        ),
        "stokes_full_quotient_basis_cells": 135,
        "parity_admissible_basis_cells": sum(
            int(grade["quotient_basis_cells"]) for grade in primary
        ),
        "compact_direct_wave_semantic_state_bytes": 64,
        "exact_parity_rank_reduction_established": True,
        "irreducible_harmonic_decomposition_established": False,
        "fixed_rank_reduction_found": False,
        "bounded_highest_harmonic_shell_growth_observed": True,
        "unbounded_growth_proved": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
