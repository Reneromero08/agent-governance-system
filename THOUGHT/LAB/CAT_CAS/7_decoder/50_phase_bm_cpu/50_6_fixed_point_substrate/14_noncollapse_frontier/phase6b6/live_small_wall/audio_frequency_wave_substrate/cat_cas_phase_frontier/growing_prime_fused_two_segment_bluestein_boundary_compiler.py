#!/usr/bin/env python3
"""Fuse final-boundary contraction into a reversible two-segment Bluestein path.

M183 retained q-1 compiled Gauss cells and used three M-cell transform
segments.  This successor uses the observed nonzero public kernel spectrum to
multiply the source spectrum in place.  It contracts the final scalar while
the convolution is transient, then reverses both M-cell segments.  Only the
scalar remains resident after forward.  The identical classical transform has
the same law, and allocated state remains linear in q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from growing_prime_reversible_bluestein_ntt_gauss_phase_compiler import (
    DECLARED_BOUNDARY,
    DECLARED_FIELDS,
    DECLARED_PROGRAMS,
    ProjectionCounts,
    TransformCounts,
    TransformField,
    boundary_from_descriptor,
    direct_gauss_table,
    ntt_segment,
    procedural_character,
)
from growing_prime_mellin_gauss_streamed_recurrence_rank import (
    PublicBoundary,
    PublicProgram,
    determinant,
    rank_and_square_class,
)


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_"
    "FINAL_BOUNDARY_COMPILER_USES_THE_NONZERO_PUBLIC_KERNEL_SPECTRUM_"
    "TO_REPLACE_THE_QMINUS1_RESIDENT_GAUSS_DESCRIPTOR_AND_THREE_M_"
    "SCRATCH_SEGMENTS_WITH_ONE_RESIDENT_SCALAR_AND_TWO_M_REVERSIBLY_"
    "CLEARED_TRANSFORM_SEGMENTS_WITH_THETA_M_LOG_M_PLUS_Q_WORK_EXACT_"
    "SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE_BUT_ALLOCATED_STATE_"
    "REMAINS_LINEAR_EXACT_WIDTH_GROWS_AND_THE_IDENTICAL_CLASSICAL_"
    "FUSED_BLUESTEIN_COMPILER_REMAINS"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def layout(field: TransformField) -> dict[str, int]:
    width = field.convolution_width
    return {
        "scalar_offset": 0,
        "resident_scalar_cells": 1,
        "left_offset": 1,
        "kernel_offset": 1 + width,
        "scratch_segment_cells": width,
        "total_carrier_field_cells": 1 + 2 * width,
    }


@dataclass
class FusedCounts:
    transform: TransformCounts
    in_place_spectral_multiplications: int = 0
    public_kernel_spectrum_inversions: int = 0
    transient_gauss_reads: int = 0
    transient_gauss_chirp_multiplications: int = 0

    @classmethod
    def create(cls) -> "FusedCounts":
        return cls(TransformCounts())

    def as_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.as_dict(),
            "in_place_spectral_multiplications": self.in_place_spectral_multiplications,
            "public_kernel_spectrum_inversions": self.public_kernel_spectrum_inversions,
            "transient_gauss_reads": self.transient_gauss_reads,
            "transient_gauss_chirp_multiplications": self.transient_gauss_chirp_multiplications,
        }


def initialize_segments(
    field: TransformField,
    cells: list[int],
    counts: TransformCounts,
    sign: int,
) -> None:
    q, p = field.base.q, field.base.p
    h = q - 1
    width = field.convolution_width
    offsets = layout(field)
    left = offsets["left_offset"]
    kernel = offsets["kernel_offset"]
    orbit = 1
    for index in range(h):
        phase = pow(field.base.additive_root, orbit, p)
        chirp = pow(field.chirp_root, index * index, p)
        cells[left + index] = (cells[left + index] + sign * phase * chirp) % p
        anti_chirp = pow(field.chirp_root, -index * index, p)
        cells[kernel + index] = (
            cells[kernel + index] + sign * anti_chirp
        ) % p
        if index:
            cells[kernel + width - index] = (
                cells[kernel + width - index] + sign * anti_chirp
            ) % p
        orbit = orbit * field.base.q_generator % q
        counts.source_phase_exponentiations += 1
        counts.chirp_exponentiations += 2
        counts.initialization_multiplications += 1
        counts.initialization_additions += 2 + int(index != 0)


def transient_gauss_reader(
    field: TransformField,
    cells: list[int],
    counts: FusedCounts,
) -> Callable[[int], int]:
    h = field.base.q - 1
    p = field.base.p
    left = layout(field)["left_offset"]

    def read(index: int) -> int:
        frequency = index % h
        counts.transient_gauss_reads += 1
        counts.transient_gauss_chirp_multiplications += 1
        return (
            pow(field.chirp_root, frequency * frequency, p)
            * cells[left + frequency]
        ) % p

    return read


def boundary_from_transient_convolution(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    counts: FusedCounts,
) -> tuple[int, ProjectionCounts]:
    q, p = field.base.q, field.base.p
    h = q - 1
    eta = h // 2
    normalization = pow(h, -1, p)
    total_character = (program.determinant_character + program.scale_character) % h
    rank, square_class = rank_and_square_class(boundary.coordinates, q)
    projection = ProjectionCounts()
    determinant_value = determinant(boundary.coordinates, q)
    determinant_character = (
        procedural_character(field.base, determinant_value, -1, projection)
        if rank == 3
        else 0
    )
    running_determinant_character = 1
    if boundary.scale % q:
        running_scale_character = procedural_character(
            field.base, boundary.scale, -total_character, projection
        )
        scale_character_step = procedural_character(
            field.base, boundary.scale, 1, projection
        )
    else:
        running_scale_character = 0
        scale_character_step = 0
    gauss = transient_gauss_reader(field, cells, counts)
    answer = 0
    for channel in range(h):
        coefficient = (
            normalization
            * gauss(-((channel - program.determinant_character) % h))
        ) % p
        if rank == 3:
            gamma = (
                gauss(channel) ** 2
                * gauss(channel + eta)
                * gauss(eta) ** 3
            ) % p
            determinant_factor = gamma * running_determinant_character % p
        elif channel == 0:
            determinant_factor = (
                q**6 - q**5 - q**3 + q**2
                if rank == 0
                else q**2 - q**3
            ) % p
        elif channel == eta and rank == 1:
            determinant_factor = q**2 * h * gauss(eta) ** 3 * square_class % p
        else:
            determinant_factor = 0
        scale_index = (total_character - channel) % h
        if boundary.scale % q:
            scale_factor = gauss(scale_index) * running_scale_character % p
        else:
            scale_factor = h % p if scale_index == 0 else 0
        answer += coefficient * determinant_factor * scale_factor
        running_determinant_character = (
            running_determinant_character * determinant_character % p
            if rank == 3
            else 0
        )
        if boundary.scale % q:
            running_scale_character = (
                running_scale_character * scale_character_step % p
            )
        projection.channel_field_multiplications += 12 if rank == 3 else 5
        projection.channel_field_additions += 1
        projection.channels += 1
    return answer % p, projection


def fused_scalar_shear(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    scalar_sign: int,
    omit_spectral_frequency: int | None = None,
    force_singular_kernel_frequency: int | None = None,
) -> tuple[FusedCounts, ProjectionCounts, str]:
    offsets = layout(field)
    if len(cells) != offsets["total_carrier_field_cells"]:
        fail("carrier size does not match fused two-segment layout")
    if any(cells[index] for index in range(1, len(cells))):
        fail("fused scratch must enter in canonical zero state")
    p = field.base.p
    width = field.convolution_width
    left = offsets["left_offset"]
    kernel = offsets["kernel_offset"]
    counts = FusedCounts.create()
    initialize_segments(field, cells, counts.transform, 1)
    ntt_segment(
        cells, left, width, field.convolution_root, p, False, counts.transform
    )
    ntt_segment(
        cells, kernel, width, field.convolution_root, p, False, counts.transform
    )
    if force_singular_kernel_frequency is not None:
        cells[kernel + force_singular_kernel_frequency % width] = 0
    if any(cells[kernel + index] == 0 for index in range(width)):
        fail("public Bluestein kernel spectrum is not invertible")
    spectrum_commitment = digest_region(cells, kernel, width)
    for index in range(width):
        if omit_spectral_frequency is not None and index == omit_spectral_frequency % width:
            continue
        cells[left + index] = cells[left + index] * cells[kernel + index] % p
        counts.in_place_spectral_multiplications += 1
    ntt_segment(
        cells, left, width, field.convolution_root, p, True, counts.transform
    )
    scalar, projection = boundary_from_transient_convolution(
        field, program, boundary, cells, counts
    )
    cells[0] = (cells[0] + scalar_sign * scalar) % p

    # Reverse the temporary convolution exactly.  The only surviving cell is
    # the public final scalar at offset zero.
    ntt_segment(
        cells, left, width, field.convolution_root, p, False, counts.transform
    )
    for index in range(width):
        if omit_spectral_frequency is not None and index == omit_spectral_frequency % width:
            continue
        cells[left + index] = (
            cells[left + index] * pow(cells[kernel + index], -1, p)
        ) % p
        counts.in_place_spectral_multiplications += 1
        counts.public_kernel_spectrum_inversions += 1
    ntt_segment(
        cells, kernel, width, field.convolution_root, p, True, counts.transform
    )
    ntt_segment(
        cells, left, width, field.convolution_root, p, True, counts.transform
    )
    initialize_segments(field, cells, counts.transform, -1)
    if any(cells[index] for index in range(1, len(cells))):
        fail("fused Bluestein scratch did not uncompute to zero")
    return counts, projection, spectrum_commitment


def digest(cells: list[int]) -> str:
    return digest_region(cells, 0, len(cells))


def digest_region(cells: list[int], offset: int, width: int) -> str:
    hasher = hashlib.sha256()
    for index in range(width):
        if index:
            hasher.update(b",")
        hasher.update(str(cells[offset + index]).encode("ascii"))
    return hasher.hexdigest()


def execute_cycle(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    expected_boundary: int,
) -> dict[str, Any]:
    if any(cells):
        fail("carrier must enter in canonical zero state")
    backing = id(cells)
    before = digest(cells)
    forward, projection, spectrum_commitment = fused_scalar_shear(
        field, program, boundary, cells, 1
    )
    projected = cells[0]
    resident_commitment = digest(cells)
    if projected != expected_boundary:
        fail("fused final boundary differs from descriptor reference")
    persisted = projected
    inverse, inverse_projection, inverse_spectrum_commitment = fused_scalar_shear(
        field, program, boundary, cells, -1
    )
    if any(cells) or id(cells) != backing:
        fail("fused carrier did not restore exactly on the same backing")
    if spectrum_commitment != inverse_spectrum_commitment:
        fail("public kernel spectrum changed across inverse rematerialization")
    return {
        "projected_final_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "pre_state_commitment": before,
        "resident_scalar_commitment": resident_commitment,
        "post_state_commitment": digest(cells),
        "public_kernel_spectrum_commitment": spectrum_commitment,
        "exactly_restored": True,
        "same_backing": id(cells) == backing,
        "forward_counts": forward.as_dict(),
        "inverse_counts": inverse.as_dict(),
        "projection_counts": projection.as_dict(),
        "inverse_projection_counts": inverse_projection.as_dict(),
    }


def transaction_case(q: int, p: int) -> dict[str, Any]:
    field = TransformField.create(q, p)
    offsets = layout(field)
    program = DECLARED_PROGRAMS[q]
    boundary = replace(DECLARED_BOUNDARY, scale=2 % q)
    h = q - 1
    second_program = PublicProgram(
        (program.determinant_character + 2) % h,
        (program.scale_character + 3) % h,
    )
    second_boundary = PublicBoundary((3, 0, 0, 2, 0, 1), 3 % q)

    # Package verification is deliberately completed before carrier allocation.
    # It is not part of the accepted fused transaction and is separately counted.
    verification_gauss = direct_gauss_table(field)
    primary_reference, _ = boundary_from_descriptor(
        field, program, boundary, verification_gauss
    )
    second_reference, _ = boundary_from_descriptor(
        field, second_program, second_boundary, verification_gauss
    )
    del verification_gauss

    cells = [0] * offsets["total_carrier_field_cells"]
    backing = id(cells)
    primary = execute_cycle(
        field, program, boundary, cells, primary_reference
    )
    reused = execute_cycle(
        field, second_program, second_boundary, cells, second_reference
    )
    fresh_cells = [0] * offsets["total_carrier_field_cells"]
    fresh = execute_cycle(
        field, second_program, second_boundary, fresh_cells, second_reference
    )
    if (
        reused["projected_final_boundary_scalar"]
        != fresh["projected_final_boundary_scalar"]
        or reused["forward_counts"] != fresh["forward_counts"]
        or id(cells) != backing
    ):
        fail("fused restored-carrier reuse differs from fresh execution")

    missing = [0] * offsets["total_carrier_field_cells"]
    fused_scalar_shear(field, program, boundary, missing, 1)
    missing_inverse_fails = any(missing)
    wrong = missing[:]
    wrong_program = PublicProgram(
        (program.determinant_character + 1) % h,
        program.scale_character,
    )
    fused_scalar_shear(field, wrong_program, boundary, wrong, -1)
    wrong_inverse_fails = any(wrong)
    omitted = [0] * offsets["total_carrier_field_cells"]
    fused_scalar_shear(
        field, program, boundary, omitted, 1, omit_spectral_frequency=0
    )
    frequency_zero_omission_changes_boundary = omitted[0] != primary[
        "projected_final_boundary_scalar"
    ]
    singular_kernel_rejected = False
    try:
        fused_scalar_shear(
            field,
            program,
            boundary,
            [0] * offsets["total_carrier_field_cells"],
            1,
            force_singular_kernel_frequency=0,
        )
    except RuntimeError:
        singular_kernel_rejected = True
    null_carrier_rejected = False
    try:
        fused_scalar_shear(field, program, boundary, [], 1)
    except RuntimeError:
        null_carrier_rejected = True
    if not all(
        (
            missing_inverse_fails,
            wrong_inverse_fails,
            frequency_zero_omission_changes_boundary,
            singular_kernel_rejected,
            null_carrier_rejected,
        )
    ):
        fail("fused transaction mutation control failed")
    forward_transform = primary["forward_counts"]["transform"]
    expected_butterflies = 3 * field.convolution_width * (
        field.convolution_width.bit_length() - 1
    )
    if (
        forward_transform["ntt_calls"] != 6
        or forward_transform["ntt_butterflies"] != expected_butterflies
    ):
        fail("fused transform work law mismatch")
    return {
        "q": q,
        "auxiliary_prime": p,
        "program": {
            "determinant_character": program.determinant_character,
            "scale_character": program.scale_character,
        },
        "boundary": {
            "coordinates": list(boundary.coordinates),
            "scale": boundary.scale,
            "rank_and_square_class": list(rank_and_square_class(boundary.coordinates, q)),
        },
        "layout": offsets,
        "carrier_capacity_bits": offsets["total_carrier_field_cells"] * p.bit_length(),
        "m183_carrier_field_cells": h + 3 * field.convolution_width,
        "m181_stream_workspace_field_cells": 10,
        "verification_only": {
            "executed_before_carrier_allocation": True,
            "direct_gauss_table_peak_field_cells": h,
            "direct_gauss_generation_field_terms": h * h,
            "descriptor_boundary_projection_channels": 2 * h,
            "included_in_accepted_transaction_carrier": False,
        },
        "primary": primary,
        "unrelated_reuse": {
            "actual_restored_backing_consumed": id(cells) == backing,
            "restored_boundary": reused["projected_final_boundary_scalar"],
            "fresh_boundary": fresh["projected_final_boundary_scalar"],
            "fresh_restored_boundary_agrees": reused["projected_final_boundary_scalar"]
            == fresh["projected_final_boundary_scalar"],
            "fresh_restored_resource_signature_agrees": reused["forward_counts"]
            == fresh["forward_counts"],
            "exactly_restored_again": reused["exactly_restored"],
            "same_backing_after_reuse": id(cells) == backing,
        },
        "controls": {
            "missing_inverse_fails": missing_inverse_fails,
            "wrong_program_inverse_fails": wrong_inverse_fails,
            "frequency_zero_omission_changes_boundary": frequency_zero_omission_changes_boundary,
            "singular_kernel_spectrum_rejected": singular_kernel_rejected,
            "singular_rejection_atomic_rollback_claimed": False,
            "null_carrier_rejected": null_carrier_rejected,
            "snapshot_used": False,
            "generation_or_lease_metadata_enforced": False,
        },
    }


def build_result() -> dict[str, Any]:
    cases = [transaction_case(q, p) for q, p in DECLARED_FIELDS]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "FOURTEEN_DECLARED_PRIME_AND_AUXILIARY_FIELD_PAIRS_Q5_"
            "THROUGH_Q53_ONE_RANK3_NONZERO_SCALE_TRANSACTION_PER_FIELD_"
            "UNRELATED_SAME_FIELD_REUSE_DIRECT_PROCESS_EXACT_RESIDUE_"
            "SOFTWARE_WITH_ONE_SCALAR_PLUS_TWO_M_ALLOCATED_CARRIER_CELLS"
        ),
        "transaction_cases": cases,
        "observed_resource_law": {
            "convolution_width": "LEAST_POWER_OF_TWO_AT_LEAST_2Q_MINUS3",
            "logical_carrier_field_cells": "ONE_PLUS2M",
            "resident_after_forward_field_cells": 1,
            "accepted_transaction_gauss_descriptor_cells": 0,
            "temporary_convolution_cells": "M",
            "temporary_kernel_spectrum_cells": "M",
            "ntt_calls_per_forward_or_inverse_compiler": 6,
            "butterflies_per_compiler": "3M_LOG2_M",
            "in_place_spectral_multiplications_per_compiler": "2M",
            "public_kernel_spectrum_inversions_per_compiler": "M",
            "accepted_forward_compiler_plus_projection_abstract_field_operation_work": "THETA_M_LOG_M_PLUS_Q",
            "modular_pow_bit_complexity_included_in_abstract_field_operation_law": False,
            "exact_bit_width_independent_of_q": False,
            "retained_inverse_history_cells": 0,
            "verification_only_direct_gauss_table_peak_cells": "Q_MINUS1",
            "verification_only_direct_gauss_generation_field_terms": "Q_MINUS1_SQUARED",
            "verification_runs_before_carrier_allocation": True,
            "streamed_commitments_avoid_full_spectrum_and_joined_digest_buffers": True,
            "python_integer_allocator_and_native_library_memory_included": False,
            "whole_process_memory_claimed": False,
        },
        "matched_baselines": {
            "m181_single_scalar_stream": "10_CARRIER_CELLS_AND_THETA_Q2_PER_BOUNDARY",
            "m183_descriptor_bluestein": "Q_MINUS1_PLUS3M_CELLS_AND_THETA_M_LOG_M_PLUS_Q_WORK",
            "strongest_compact_classical": "IDENTICAL_ONE_PLUS2M_FUSED_REVERSIBLE_BLUESTEIN_COMPILER",
            "state_advantage": False,
            "work_advantage": False,
        },
        "controls": {
            "all_boundaries_match_m183_descriptor_reference": True,
            "all_public_kernel_spectra_nonzero": True,
            "all_two_segment_scratch_clears": True,
            "all_missing_inverses_fail": True,
            "all_wrong_program_inverses_fail": True,
            "frequency_zero_omission_changes_all_declared_boundaries": True,
            "all_forced_singular_kernel_spectra_rejected": True,
            "all_null_carriers_rejected": True,
            "all_same_backing_restorations_pass": True,
            "all_unrelated_reuses_match_fresh": True,
        },
        "strict_boundaries": {
            "catvm_custody": False,
            "machine_enforced_hidden_intermediate": False,
            "sublinear_allocated_state": False,
            "fixed_exact_bit_width": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "FINAL_BOUNDARY_FUSION_REMOVES_DESCRIPTOR_RESIDENCY_AND_ONE_"
            "M_CELL_PRODUCT_SEGMENT_BUT_TWO_M_TRANSFORM_SEGMENTS_REMAIN_"
            "LINEAR_WITH_GROWING_EXACT_WIDTH_AND_THE_IDENTICAL_CLASSICAL_"
            "COMPILER_SO_THE_NEXT_REPAIR_MUST_REDUCE_REVERSIBLE_TRANSFORM_"
            "WORKSPACE_WITHOUT_MOVING_IT_INTO_A_RETAINED_PUBLIC_SPECTRUM_"
            "OR_ACCEPT_A_BOUNDED_LINEAR_STATE_OBSTRUCTION"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
