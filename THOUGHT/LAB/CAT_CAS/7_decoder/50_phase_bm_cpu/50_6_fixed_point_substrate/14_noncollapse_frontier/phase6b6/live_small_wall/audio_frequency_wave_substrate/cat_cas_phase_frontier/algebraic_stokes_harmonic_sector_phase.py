#!/usr/bin/env python3
"""Parity-closed Stokes harmonic-sector phase carrier experiment."""

from __future__ import annotations

import json
import sys

import algebraic_stokes_lie_relation_phase as base


def execute_once(
    program: base.Program, mode: base.Mode
) -> base.Execution:
    return base.execute_once(program, mode, parity_reduced=True)


def main() -> None:
    if sys.argv[1:] == ["--project-shared-port"]:
        base.fail("resident Stokes harmonic shared wave port projection denied")
    if sys.argv[1:] == ["--null-carrier"]:
        base.fail("invalid Stokes harmonic sector carrier")
    if sys.argv[1:]:
        base.fail("unsupported Stokes harmonic sector request")

    shared_carrier = base.make_carrier(parity_reduced=True)
    primary = base.execute_on(
        shared_carrier, base.Program.PRIMARY, base.Mode.CORRECT
    )
    replay = execute_once(base.Program.PRIMARY, base.Mode.CORRECT)
    missing = execute_once(
        base.Program.PRIMARY, base.Mode.MISSING_INVERSE
    )
    wrong = execute_once(base.Program.PRIMARY, base.Mode.WRONG_INVERSE)
    reordered = execute_once(
        base.Program.PRIMARY, base.Mode.REORDERED_INVERSE
    )
    snapshot = execute_once(base.Program.PRIMARY, base.Mode.SNAPSHOT)
    identity = execute_once(base.Program.IDENTITY, base.Mode.CORRECT)
    zero = execute_once(base.Program.ZERO, base.Mode.CORRECT)
    swapped = execute_once(base.Program.SWAPPED, base.Mode.CORRECT)
    reuse = base.execute_on(
        shared_carrier, base.Program.REUSE, base.Mode.CORRECT
    )

    maximum_reuse_restoration = reuse.restoration_max_abs
    reuse_generation = 1
    for cycle in range(1, base.REUSE_CYCLES):
        program = (
            base.Program.REUSE
            if cycle % 2 == 0
            else base.Program.PRIMARY
        )
        repeated = base.execute_on(
            shared_carrier, program, base.Mode.CORRECT
        )
        maximum_reuse_restoration = max(
            maximum_reuse_restoration,
            repeated.restoration_max_abs,
        )
        reuse_generation += 1

    identity_higher_zero = all(
        grade["nonzero_p17"] == 0 and grade["nonzero_p19"] == 0
        for grade in identity.boundary[1:]
    )
    zero_all_zero = all(
        grade["nonzero_p17"] == 0 and grade["nonzero_p19"] == 0
        for grade in zero.boundary
    )
    swap_discriminated = (
        primary.boundary[1]["hash_p17"]
        != swapped.boundary[1]["hash_p17"]
        and primary.boundary[1]["hash_p19"]
        != swapped.boundary[1]["hash_p19"]
    )
    if not (
        primary.restoration_max_abs <= base.RESTORE_TOLERANCE
        and missing.restoration_max_abs >= 1.0e-6
        and wrong.restoration_max_abs > base.RESTORE_TOLERANCE
        and reordered.restoration_max_abs >= 1.0e-6
        and primary.restoration_nonidentity_cells == 0
        and missing.restoration_nonidentity_cells > 0
        and wrong.restoration_nonidentity_cells > 0
        and reordered.restoration_nonidentity_cells > 0
        and snapshot.restoration_max_abs <= base.RESTORE_TOLERANCE
        and identity_higher_zero
        and zero_all_zero
        and swap_discriminated
        and primary.combined_hash == replay.combined_hash
        and primary.stats.hidden_coefficient_decodes == 0
        and maximum_reuse_restoration <= base.RESTORE_TOLERANCE
    ):
        base.fail("Stokes harmonic sector control failed")

    basis_cells = sum(
        len(block.basis) for block in shared_carrier.blocks
    )
    phase_cells = basis_cells * len(base.PRIMES)
    output = {
        "claim": (
            "BOUNDED_PARITY_ADMISSIBLE_STOKES_HARMONIC_SECTOR_"
            "DUAL_PRIME_PHASE_SIGNATURE_REDUCTION_WITH_"
            "RESTORATION_AND_REUSE"
        ),
        "result": "PASS",
        "port_type": "NORMALIZED_TWO_MODE_STOKES_SPHERE_WAVE_PORT",
        "grades": base.GRADES,
        "maximum_degree": base.MAX_DEGREE,
        "primary_grades": primary.boundary,
        "reuse_grades": reuse.boundary,
        "primary_combined_hash": primary.combined_hash,
        "reuse_combined_hash": reuse.combined_hash,
        "public_degree_parity_basis_compilation": True,
        "parity_admissible_basis_cells": basis_cells,
        "resident_dual_prime_coefficient_cells": phase_cells,
        "logical_packed_phase_payload_bytes": phase_cells * 16,
        "actual_python_object_allocation_measured": False,
        "retained_inverse_kernels": 0,
        "native_phase_updates": primary.stats.native_phase_updates,
        "resident_phase_reads": primary.stats.resident_phase_reads,
        "field_product_interpolations": (
            primary.stats.field_product_interpolations
        ),
        "lie_poisson_monomial_products": (
            primary.stats.lie_poisson_monomial_products
        ),
        "sphere_reduction_terms": primary.stats.sphere_reduction_terms,
        "hidden_coefficient_decodes": (
            primary.stats.hidden_coefficient_decodes
        ),
        "final_coefficient_decodes": (
            primary.stats.final_coefficient_decodes
        ),
        "maximum_unit_modulus_error": (
            primary.stats.maximum_unit_modulus_error
        ),
        "maximum_final_root_error": (
            primary.stats.maximum_final_root_error
        ),
        "restoration_max_abs": primary.restoration_max_abs,
        "maximum_reuse_restoration_error": (
            maximum_reuse_restoration
        ),
        "successful_restoration_receipt": (
            primary.restoration_generation
        ),
        "same_carrier_reuse_transactions": reuse_generation,
        "missing_inverse_residual": missing.restoration_max_abs,
        "wrong_inverse_residual": wrong.restoration_max_abs,
        "reordered_inverse_residual": reordered.restoration_max_abs,
        "missing_inverse_modular_mismatch_cells": (
            missing.restoration_nonidentity_cells
        ),
        "wrong_inverse_modular_mismatch_cells": (
            wrong.restoration_nonidentity_cells
        ),
        "reordered_inverse_modular_mismatch_cells": (
            reordered.restoration_nonidentity_cells
        ),
        "snapshot_residual": snapshot.restoration_max_abs,
        "snapshot_restoration_receipt": (
            snapshot.restoration_generation
        ),
        "identity_mixer_higher_grades_zero": identity_higher_zero,
        "zero_kerr_all_grades_zero": zero_all_zero,
        "swapped_order_discriminated": swap_discriminated,
        "deterministic_replay": (
            primary.combined_hash == replay.combined_hash
        ),
        "actual_inverse_restoration": True,
        "actual_restored_carrier_reuse": True,
        "final_boundary_only_projection": True,
        "actual_resident_grade_consumed": True,
        "public_topology_inverse_rematerialization": True,
        "exact_norm_casimir_quotient_reduction": True,
        "mathematically_exact_dual_prime_phase_algebra": True,
        "exact_rank_reduction_established": True,
        "irreducible_harmonic_decomposition_established": False,
        "fixed_rank_closure_established": False,
        "unbounded_growth_proved": False,
        "genuinely_distinct_phase_resource": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
