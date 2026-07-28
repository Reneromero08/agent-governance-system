#!/usr/bin/env python3
"""Non-emitting exact per-cell verifier for alternating-axis boundaries."""

from __future__ import annotations

import json

import sympy

import algebraic_stokes_alternating_axis_phase as phase
import algebraic_stokes_alternating_axis_reference as reference
import algebraic_stokes_lie_relation_reference as rational


def verify_program(program: str) -> tuple[int, float]:
    carrier = phase.make_carrier()
    stats = phase.Stats()
    generators = phase.generator_polynomials(program)
    phase.seal(carrier, phase.base.h0(), False, stats)
    for grade, generator in enumerate(generators):
        phase.accumulate_bracket(
            carrier, grade, generator, False, False, stats
        )

    comparisons = 0
    exact_expressions = reference.expressions(program)
    for block, expression in zip(carrier.blocks, exact_expressions):
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
                observed = phase.decode(
                    block.working[field][cell][1], field, stats
                )
                expected = rational.coefficient_mod(
                    coefficients.get(monomial, 0), prime
                )
                if observed != expected:
                    raise RuntimeError(
                        "alternating-axis exact cell mismatch"
                    )
                comparisons += 1

    for grade in range(phase.GRADES - 1, 0, -1):
        phase.accumulate_bracket(
            carrier,
            grade - 1,
            generators[grade - 1],
            True,
            False,
            stats,
        )
    phase.seal(carrier, phase.base.h0(), True, stats)
    restore_error = phase.restoration(carrier)
    if (
        restore_error > phase.RESTORE_TOLERANCE
        or phase.restoration_nonidentity(carrier) != 0
    ):
        raise RuntimeError(
            "exact verifier failed to restore phase carrier"
        )
    return comparisons, restore_error


def main() -> None:
    primary_count, primary_restore = verify_program("PRIMARY")
    reuse_count, reuse_restore = verify_program("REUSE")
    print(
        json.dumps(
            {
                "result": "PASS",
                "exact_per_cell_dual_prime_comparison": True,
                "coefficient_values_emitted": False,
                "primary_comparisons": primary_count,
                "reuse_comparisons": reuse_count,
                "total_comparisons": primary_count + reuse_count,
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
