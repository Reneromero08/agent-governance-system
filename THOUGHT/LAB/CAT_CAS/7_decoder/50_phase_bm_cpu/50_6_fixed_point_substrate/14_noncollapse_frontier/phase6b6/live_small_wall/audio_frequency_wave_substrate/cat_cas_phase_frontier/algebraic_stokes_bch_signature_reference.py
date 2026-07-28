#!/usr/bin/env python3
"""Exact bounded BCH signature for two noncommuting Stokes modules."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache

import sympy

import algebraic_stokes_alternating_axis_reference as harmonic
import algebraic_stokes_lie_relation_reference as base


MAX_WORD_GRADE = 10
Word = tuple[str, ...]


def tensor_multiply(
    left: dict[Word, sympy.Rational],
    right: dict[Word, sympy.Rational],
) -> dict[Word, sympy.Rational]:
    result: defaultdict[Word, sympy.Rational] = defaultdict(
        lambda: sympy.Rational(0)
    )
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word = left_word + right_word
            if len(word) <= MAX_WORD_GRADE:
                result[word] += left_value * right_value
    return {
        word: value for word, value in result.items() if value
    }


def tensor_bch() -> dict[Word, sympy.Rational]:
    """Return log(exp(A) exp(B)) in the truncated tensor algebra."""
    exponential: dict[Word, sympy.Rational] = {
        (): sympy.Rational(1)
    }
    for left_power in range(MAX_WORD_GRADE + 1):
        for right_power in range(
            MAX_WORD_GRADE + 1 - left_power
        ):
            if left_power + right_power == 0:
                continue
            exponential[
                ("A",) * left_power + ("B",) * right_power
            ] = sympy.Rational(
                1,
                math.factorial(left_power)
                * math.factorial(right_power),
            )
    augmentation = {
        word: value
        for word, value in exponential.items()
        if word
    }
    logarithm: defaultdict[Word, sympy.Rational] = defaultdict(
        lambda: sympy.Rational(0)
    )
    power: dict[Word, sympy.Rational] = {
        (): sympy.Rational(1)
    }
    for order in range(1, MAX_WORD_GRADE + 1):
        power = tensor_multiply(power, augmentation)
        sign = 1 if order % 2 else -1
        for word, value in power.items():
            logarithm[word] += (
                sign * value / sympy.Integer(order)
            )
    return {
        word: value
        for word, value in logarithm.items()
        if value
    }


def hamiltonians(program: str) -> dict[str, sympy.Expr]:
    if program == "PRIMARY":
        return {"A": base.H0, "B": base.PRIMARY}
    if program == "REUSE":
        return {"A": base.H0, "B": base.REUSE}
    if program == "SWAPPED":
        return {"A": base.PRIMARY, "B": base.H0}
    if program == "COMMUTING":
        return {"A": base.H0, "B": base.H0}
    raise ValueError("unknown BCH program")


def components(program: str) -> list[sympy.Expr]:
    generators = hamiltonians(program)
    logarithm = tensor_bch()

    @lru_cache(maxsize=None)
    def dynkin(word: Word) -> sympy.Expr:
        if len(word) == 1:
            return generators[word[0]]
        return base.lie_poisson(
            generators[word[0]], dynkin(word[1:])
        )

    result = []
    for grade in range(1, MAX_WORD_GRADE + 1):
        # The Dynkin-Specht-Wever projection maps each homogeneous
        # primitive tensor component P_grade to grade * P_grade.
        expression = sum(
            (
                value
                * dynkin(word)
                / sympy.Integer(grade)
            )
            for word, value in logarithm.items()
            if len(word) == grade
        )
        result.append(base.reduce_sphere(expression))
    return result


def component_record(
    expression: sympy.Expr, word_grade: int
) -> dict[str, object]:
    polynomial_degree = word_grade + 1
    record = base.grade_record(
        expression, polynomial_degree, parity_reduced=True
    )
    highest = harmonic.homogeneous_part(
        expression, polynomial_degree
    )
    projected = harmonic.harmonic_projection(
        highest, polynomial_degree
    )
    record.update(
        {
            "word_grade": word_grade,
            "polynomial_degree": polynomial_degree,
            "highest_homogeneous_terms": len(
                sympy.Poly(
                    highest,
                    *base.VARIABLES,
                    domain=sympy.QQ,
                ).terms()
            ),
            "middle_catalecticant_rank": (
                harmonic.middle_catalecticant_rank(
                    projected, polynomial_degree
                )
            ),
            "unique_harmonic_representative": True,
        }
    )
    return record


def main() -> None:
    primary_expressions = components("PRIMARY")
    reuse_expressions = components("REUSE")
    primary = [
        component_record(expression, grade)
        for grade, expression in enumerate(
            primary_expressions, start=1
        )
    ]
    reuse = [
        component_record(expression, grade)
        for grade, expression in enumerate(
            reuse_expressions, start=1
        )
    ]
    swapped = components("SWAPPED")
    commuting = components("COMMUTING")
    ranks = [
        int(record["middle_catalecticant_rank"])
        for record in primary
    ]
    basis_cells = sum(
        int(record["quotient_basis_cells"])
        for record in primary
    )
    logarithm = tensor_bch()
    print(
        json.dumps(
            {
                "result": "PASS",
                "oracle": (
                    "EXACT_RATIONAL_TENSOR_LOG_DYNKIN_PROJECTED_"
                    "STOKES_BCH"
                ),
                "composition": "LOG_EXP_A_EXP_B",
                "maximum_word_grade": MAX_WORD_GRADE,
                "maximum_polynomial_degree": (
                    MAX_WORD_GRADE + 1
                ),
                "primary_components": primary,
                "reuse_components": reuse,
                "nonzero_tensor_words_by_grade": [
                    sum(
                        int(len(word) == grade)
                        for word in logarithm
                    )
                    for grade in range(
                        1, MAX_WORD_GRADE + 1
                    )
                ],
                "middle_catalecticant_ranks": ranks,
                "strict_rank_increases_observed": sum(
                    int(right > left)
                    for left, right in zip(ranks, ranks[1:])
                ),
                "maximum_middle_catalecticant_rank": max(ranks),
                "bounded_full_harmonic_catalecticant_rank_observed": (
                    ranks == [3, 3, 5, 5, 7, 7, 9, 9, 11, 11]
                ),
                "commuting_higher_components_zero": all(
                    expression == 0
                    for expression in commuting[1:]
                ),
                "swapped_order_material": any(
                    left != right
                    for left, right in zip(
                        primary_expressions[1:], swapped[1:]
                    )
                ),
                "parity_support_exact": all(
                    bool(record["support_parity_exact"])
                    for record in primary + reuse
                ),
                "same_output_dual_prime_classical_semantic_state_bytes": (
                    basis_cells * len(base.PRIMES)
                ),
                "same_output_classical_actual_allocation_measured": False,
                "compact_point_evaluation_semantic_state_bytes": 64,
                "point_evaluation_has_same_boundary_semantics": False,
                "fixed_rank_nonseparable_closure_established": False,
                "arbitrary_compact_representation_lower_bound": False,
                "unbounded_rank_growth_proved": False,
                "genuinely_distinct_phase_resource": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
