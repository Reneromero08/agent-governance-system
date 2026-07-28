#!/usr/bin/env python3
"""Reflection-graded rematerialized character-phase Stokes BCH closure."""

from __future__ import annotations

import json
import sys

import algebraic_stokes_bch_rematerialized_phase as full


base = full.base
character = full.character
Stats = full.Stats
Carrier = full.Carrier
MAX_WORD_GRADE = full.MAX_WORD_GRADE
RESTORE_TOLERANCE = full.RESTORE_TOLERANCE
bch_words = full.bch_words
generators = full.generators
apply_word = full.apply_word
execute_on = full.execute_on
residual = full.residual
project = full.project
snapshot_reload = full.snapshot_reload


def reflection_basis(
    polynomial_degree: int,
) -> tuple[base.Monomial, ...]:
    # A and B are even under y -> -y. The Lie-Poisson bracket
    # changes reflection parity, so word length n has parity n-1.
    required_y_parity = (polynomial_degree - 2) % 2
    return tuple(
        monomial
        for monomial in base.basis(
            polynomial_degree, parity_reduced=True
        )
        if monomial[1] % 2 == required_y_parity
    )


def make_block(degree: int) -> character.Block:
    canonical = reflection_basis(degree)
    return character.Block(
        degree,
        canonical,
        {
            monomial: index
            for index, monomial in enumerate(canonical)
        },
        [
            [
                [1.0 + 0.0j for _ in range(prime)]
                for _ in canonical
            ]
            for prime in base.PRIMES
        ],
    )


def make_carrier() -> Carrier:
    return Carrier(
        finals=[
            make_block(grade + 1)
            for grade in range(1, MAX_WORD_GRADE + 1)
        ],
        scratch=[
            make_block(degree)
            for degree in range(2, MAX_WORD_GRADE + 2)
        ],
    )


def main() -> None:
    if sys.argv[1:] == ["--project-scratch"]:
        raise RuntimeError(
            "resident reflection-graded BCH scratch projection denied"
        )
    if sys.argv[1:] == ["--null-carrier"]:
        raise RuntimeError(
            "invalid reflection-graded BCH phase carrier"
        )
    if sys.argv[1:]:
        raise RuntimeError(
            "unsupported reflection-graded BCH phase request"
        )

    carrier = make_carrier()
    primary, primary_stats, primary_residual = execute_on(
        carrier, "PRIMARY"
    )
    reuse, _, reuse_residual = execute_on(carrier, "REUSE")
    swapped_carrier = make_carrier()
    swapped, _, swapped_residual = execute_on(
        swapped_carrier, "SWAPPED"
    )
    missing_carrier = make_carrier()
    _, _, missing_residual = execute_on(
        missing_carrier, "PRIMARY", omit_last_inverse=True
    )
    wrong_carrier = make_carrier()
    _, _, wrong_residual = execute_on(
        wrong_carrier, "PRIMARY", wrong_first_inverse=True
    )

    snapshot_carrier = make_carrier()
    snapshot_words = bch_words()
    snapshot_generators = generators("PRIMARY")
    snapshot_stats = Stats()
    for word, coefficient in snapshot_words:
        apply_word(
            snapshot_carrier,
            word,
            coefficient,
            snapshot_generators,
            False,
            snapshot_stats,
        )
    snapshot_boundary = project(
        snapshot_carrier, snapshot_stats
    )
    snapshot_reload(snapshot_carrier)
    snapshot_residual = residual(snapshot_carrier)
    snapshot_reuse, _, snapshot_reuse_residual = execute_on(
        snapshot_carrier, "REUSE"
    )

    if not (
        primary_residual <= RESTORE_TOLERANCE
        and reuse_residual <= RESTORE_TOLERANCE
        and swapped_residual <= RESTORE_TOLERANCE
        and missing_residual >= 1.0e-6
        and wrong_residual >= 1.0e-6
        and swapped != primary
        and snapshot_boundary == primary
        and snapshot_residual == 0.0
        and snapshot_reuse == reuse
        and snapshot_reuse_residual <= RESTORE_TOLERANCE
        and primary_stats.hidden_decodes == 0
    ):
        raise RuntimeError(
            "reflection-graded BCH phase control failed"
        )

    final_cells = sum(len(block.basis) for block in carrier.finals)
    scratch_cells = sum(
        len(block.basis) for block in carrier.scratch
    )
    packed_bytes = (
        (final_cells + scratch_cells)
        * sum(base.PRIMES)
        * 16
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "claim": (
                    "BOUNDED_REFLECTION_GRADED_TOPOLOGY_"
                    "REMATERIALIZED_NONCOMMUTING_STOKES_BCH_"
                    "CHARACTER_PHASE_QUOTIENT_WITH_RESTORATION_"
                    "AND_REUSE"
                ),
                "maximum_word_grade": MAX_WORD_GRADE,
                "maximum_polynomial_degree": MAX_WORD_GRADE + 1,
                "declared_boundary": (
                    "FULL_REFLECTION_GRADED_GRADE1_TO6_BCH_"
                    "COEFFICIENT_SIGNATURE"
                ),
                "primary_components": primary,
                "reuse_components": reuse,
                "reflection_law": (
                    "Y_PARITY_EQUALS_LIE_WORD_LENGTH_MINUS_ONE_MOD2"
                ),
                "reflection_law_topology_compiled": True,
                "compiled_nonzero_lie_words": len(bch_words()),
                "compiled_public_word_letter_bytes": sum(
                    len(word) for word, _ in bch_words()
                ),
                "compiled_public_coefficient_logical_bytes": (
                    len(bch_words()) * 16
                ),
                "compiled_public_topology_actual_allocation_measured": (
                    False
                ),
                "full_parity_basis_cells": 116,
                "final_coefficient_cells": final_cells,
                "full_parity_scratch_cells": 116,
                "reusable_scratch_coefficient_cells": scratch_cells,
                "basis_cells_removed": 232
                - final_cells
                - scratch_cells,
                "resident_character_phase_cells": (
                    (final_cells + scratch_cells)
                    * sum(base.PRIMES)
                ),
                "logical_packed_phase_payload_bytes": packed_bytes,
                "same_output_dual_prime_classical_semantic_state_bytes": (
                    final_cells * len(base.PRIMES)
                ),
                "same_output_classical_actual_allocation_measured": False,
                "retained_word_history_blocks": 0,
                "maximum_live_scratch_blocks": (
                    primary_stats.maximum_live_scratch_blocks
                ),
                "native_phase_updates": primary_stats.phase_updates,
                "resident_phase_reads": primary_stats.phase_reads,
                "hidden_decodes": primary_stats.hidden_decodes,
                "final_decodes": primary_stats.final_decodes,
                "maximum_final_root_error": (
                    primary_stats.maximum_root_error
                ),
                "restoration_max_abs": primary_residual,
                "reuse_restoration_max_abs": reuse_residual,
                "missing_inverse_residual": missing_residual,
                "wrong_inverse_residual": wrong_residual,
                "swapped_order_boundary_differs": swapped != primary,
                "snapshot_creation_traffic_bytes": packed_bytes,
                "snapshot_reload_traffic_bytes": packed_bytes,
                "snapshot_sham_loaded": True,
                "snapshot_sham_residual": snapshot_residual,
                "snapshot_sham_reuse_boundary_matches": (
                    snapshot_reuse == reuse
                ),
                "snapshot_sham_reuse_residual": (
                    snapshot_reuse_residual
                ),
                "snapshot_restoration_receipt": 0,
                "actual_inverse_restoration": True,
                "actual_restored_carrier_reuse": True,
                "topology_derived_rematerialization": True,
                "final_boundary_only_projection": True,
                "fixed_rank_nonseparable_closure_established": False,
                "genuinely_distinct_phase_resource": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
