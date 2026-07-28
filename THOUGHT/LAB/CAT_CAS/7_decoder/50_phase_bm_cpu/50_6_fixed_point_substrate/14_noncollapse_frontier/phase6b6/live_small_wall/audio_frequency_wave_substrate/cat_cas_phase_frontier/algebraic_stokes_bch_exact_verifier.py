#!/usr/bin/env python3
"""Non-emitting exact verifier for rematerialized phase BCH closure."""

from __future__ import annotations

import json

import sympy

import algebraic_stokes_bch_rematerialized_phase as phase
import algebraic_stokes_bch_signature_reference as reference
import algebraic_stokes_lie_relation_reference as rational


def verify(program: str) -> tuple[int, float]:
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
                        "rematerialized BCH exact cell mismatch"
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
            "BCH exact verifier failed to restore carrier"
        )
    return comparisons, restore_error


def main() -> None:
    primary_count, primary_restore = verify("PRIMARY")
    reuse_count, reuse_restore = verify("REUSE")
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
