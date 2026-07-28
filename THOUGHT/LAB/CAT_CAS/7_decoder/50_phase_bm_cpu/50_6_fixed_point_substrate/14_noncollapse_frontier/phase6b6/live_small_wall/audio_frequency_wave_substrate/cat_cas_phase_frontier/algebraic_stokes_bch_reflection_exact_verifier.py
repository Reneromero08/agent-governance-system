#!/usr/bin/env python3
"""Non-emitting exact verifier for reflection-graded phase BCH closure."""

from __future__ import annotations

import json

import sympy

import algebraic_stokes_bch_reflection_phase as phase
import algebraic_stokes_bch_signature_reference as reference
import algebraic_stokes_lie_relation_reference as rational


def verify(program: str) -> tuple[int, int, float]:
    carrier = phase.make_carrier()
    stats = phase.Stats()
    compiled = phase.bch_words()
    program_generators = phase.generators(program)
    for word, coefficient in compiled:
        phase.apply_word(
            carrier,
            word,
            coefficient,
            program_generators,
            False,
            stats,
        )

    comparisons = 0
    excluded_zero_checks = 0
    expressions = reference.components(program)[
        : phase.MAX_WORD_GRADE
    ]
    for block, expression in zip(carrier.finals, expressions):
        coefficients = {
            monomial: sympy.Rational(value)
            for monomial, value in sympy.Poly(
                expression,
                *rational.VARIABLES,
                domain=sympy.QQ,
            ).as_dict().items()
        }
        full_basis = phase.base.basis(
            block.degree_limit, parity_reduced=True
        )
        excluded = set(full_basis) - set(block.basis)
        if any(coefficients.get(monomial, 0) for monomial in excluded):
            raise RuntimeError(
                "excluded BCH reflection sector is nonzero"
            )
        excluded_zero_checks += len(excluded)
        for field, prime in enumerate(phase.base.PRIMES):
            for cell, monomial in enumerate(block.basis):
                observed = phase.character.decode(
                    block.working[field][cell][1],
                    field,
                    phase.character.Stats(),
                )
                expected = rational.coefficient_mod(
                    coefficients.get(monomial, 0), prime
                )
                if observed != expected:
                    raise RuntimeError(
                        "reflection-graded BCH exact cell mismatch"
                    )
                comparisons += 1

    for word, coefficient in reversed(compiled):
        phase.apply_word(
            carrier,
            word,
            coefficient,
            program_generators,
            True,
            stats,
        )
    restore_error = phase.residual(carrier)
    if restore_error > phase.RESTORE_TOLERANCE:
        raise RuntimeError(
            "reflection BCH verifier failed to restore carrier"
        )
    return comparisons, excluded_zero_checks, restore_error


def main() -> None:
    primary_count, primary_excluded, primary_restore = verify(
        "PRIMARY"
    )
    reuse_count, reuse_excluded, reuse_restore = verify("REUSE")
    print(
        json.dumps(
            {
                "result": "PASS",
                "exact_per_cell_dual_prime_comparison": True,
                "coefficient_values_emitted": False,
                "primary_comparisons": primary_count,
                "reuse_comparisons": reuse_count,
                "total_comparisons": (
                    primary_count + reuse_count
                ),
                "primary_excluded_zero_cells": primary_excluded,
                "reuse_excluded_zero_cells": reuse_excluded,
                "reflection_excluded_sector_exactly_zero": True,
                "maximum_verifier_restoration_error": max(
                    primary_restore, reuse_restore
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
