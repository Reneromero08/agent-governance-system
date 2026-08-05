#!/usr/bin/env python3
"""Execute the fused Bluestein boundary with one transform segment.

The public zero-padded chirp-kernel spectrum is rematerialized one frequency
at a time and never retained.  This removes M184's second M-cell segment, but
turns the exact kernel work back into theta(M*q).  The experiment measures
that state/work Pareto point against M181 and the identical classical path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from growing_prime_fused_two_segment_bluestein_boundary_compiler import (
    boundary_from_transient_convolution,
    digest,
)
from growing_prime_reversible_bluestein_ntt_gauss_phase_compiler import (
    DECLARED_BOUNDARY,
    DECLARED_FIELDS,
    DECLARED_PROGRAMS,
    TransformCounts,
    TransformField,
    boundary_from_descriptor,
    direct_gauss_table,
    ntt_segment,
)
from growing_prime_mellin_gauss_streamed_recurrence_rank import (
    PublicBoundary,
    PublicProgram,
    rank_and_square_class,
)


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_ONE_SEGMENT_STREAMED_KERNEL_"
    "BLUESTEIN_FINAL_BOUNDARY_COMPILER_REDUCES_THE_ACCEPTED_BACKING_TO_"
    "ONE_RESIDENT_SCALAR_PLUS_ONE_M_CELL_REVERSIBLY_CLEARED_TRANSFORM_"
    "SEGMENT_WITHOUT_RETAINING_THE_PUBLIC_KERNEL_SPECTRUM_WITH_EXACT_"
    "SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE_BUT_STREAMING_TWO_M_"
    "SPECTRUM_VALUES_COSTS_2M_TIMES_2QMINUS3_KERNEL_TERMS_M_REMAINS_"
    "LINEAR_THE_M181_FIXED10_CELL_QUADRATIC_STREAM_REMAINS_AND_THE_"
    "IDENTICAL_CLASSICAL_ONE_SEGMENT_COMPILER_HAS_THE_SAME_LAW"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def layout(field: TransformField) -> dict[str, int]:
    width = field.convolution_width
    return {
        "scalar_offset": 0,
        "resident_scalar_cells": 1,
        "transform_offset": 1,
        "transform_segment_cells": width,
        "total_carrier_field_cells": 1 + width,
    }


@dataclass
class StreamedKernelCounts:
    transform: TransformCounts
    streamed_kernel_spectrum_values: int = 0
    streamed_kernel_nonzero_terms: int = 0
    streamed_kernel_twiddle_exponentiations: int = 0
    streamed_kernel_chirp_exponentiations: int = 0
    streamed_kernel_field_multiplications: int = 0
    streamed_kernel_field_additions: int = 0
    in_place_spectral_multiplications: int = 0
    public_kernel_spectrum_inversions: int = 0
    transient_gauss_reads: int = 0
    transient_gauss_chirp_multiplications: int = 0

    @classmethod
    def create(cls) -> "StreamedKernelCounts":
        return cls(TransformCounts())

    def as_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.as_dict(),
            "streamed_kernel_spectrum_values": self.streamed_kernel_spectrum_values,
            "streamed_kernel_nonzero_terms": self.streamed_kernel_nonzero_terms,
            "streamed_kernel_twiddle_exponentiations": self.streamed_kernel_twiddle_exponentiations,
            "streamed_kernel_chirp_exponentiations": self.streamed_kernel_chirp_exponentiations,
            "streamed_kernel_field_multiplications": self.streamed_kernel_field_multiplications,
            "streamed_kernel_field_additions": self.streamed_kernel_field_additions,
            "in_place_spectral_multiplications": self.in_place_spectral_multiplications,
            "public_kernel_spectrum_inversions": self.public_kernel_spectrum_inversions,
            "transient_gauss_reads": self.transient_gauss_reads,
            "transient_gauss_chirp_multiplications": self.transient_gauss_chirp_multiplications,
        }


def initialize_source(
    field: TransformField,
    cells: list[int],
    counts: TransformCounts,
    sign: int,
) -> None:
    q, p = field.base.q, field.base.p
    h = q - 1
    offset = layout(field)["transform_offset"]
    orbit = 1
    for index in range(h):
        phase = pow(field.base.additive_root, orbit, p)
        chirp = pow(field.chirp_root, index * index, p)
        cells[offset + index] = (
            cells[offset + index] + sign * phase * chirp
        ) % p
        orbit = orbit * field.base.q_generator % q
        counts.source_phase_exponentiations += 1
        counts.chirp_exponentiations += 1
        counts.initialization_multiplications += 1
        counts.initialization_additions += 1


def streamed_kernel_spectrum_value(
    field: TransformField,
    frequency: int,
    counts: StreamedKernelCounts,
) -> int:
    """Return one exact DFT coefficient of the zero-padded chirp kernel."""
    q, p = field.base.q, field.base.p
    h = q - 1
    width = field.convolution_width
    answer = 1
    counts.streamed_kernel_spectrum_values += 1
    counts.streamed_kernel_nonzero_terms += 1
    counts.streamed_kernel_field_additions += 1
    for index in range(1, h):
        anti_chirp = pow(field.chirp_root, -index * index, p)
        left_twiddle = pow(field.convolution_root, frequency * index, p)
        right_twiddle = pow(
            field.convolution_root, frequency * (width - index), p
        )
        answer = (answer + anti_chirp * left_twiddle) % p
        answer = (answer + anti_chirp * right_twiddle) % p
        counts.streamed_kernel_nonzero_terms += 2
        counts.streamed_kernel_twiddle_exponentiations += 2
        counts.streamed_kernel_chirp_exponentiations += 1
        counts.streamed_kernel_field_multiplications += 2
        counts.streamed_kernel_field_additions += 2
    return answer


def update_spectrum_commitment(hasher: Any, index: int, value: int) -> None:
    if index:
        hasher.update(b",")
    hasher.update(str(value).encode("ascii"))


def one_segment_scalar_shear(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
    scalar_sign: int,
    omit_spectral_frequency: int | None = None,
    force_singular_kernel_frequency: int | None = None,
) -> tuple[StreamedKernelCounts, Any, str]:
    offsets = layout(field)
    if len(cells) != offsets["total_carrier_field_cells"]:
        fail("carrier size does not match one-segment layout")
    if any(cells[index] for index in range(1, len(cells))):
        fail("one-segment transform workspace must enter at zero")
    p = field.base.p
    width = field.convolution_width
    transform = offsets["transform_offset"]
    counts = StreamedKernelCounts.create()
    initialize_source(field, cells, counts.transform, 1)
    ntt_segment(
        cells, transform, width, field.convolution_root, p, False, counts.transform
    )
    forward_commitment = hashlib.sha256()
    for index in range(width):
        spectrum = streamed_kernel_spectrum_value(field, index, counts)
        if force_singular_kernel_frequency is not None and index == (
            force_singular_kernel_frequency % width
        ):
            spectrum = 0
        if spectrum == 0:
            fail("streamed public Bluestein kernel spectrum is not invertible")
        update_spectrum_commitment(forward_commitment, index, spectrum)
        if omit_spectral_frequency is not None and index == (
            omit_spectral_frequency % width
        ):
            continue
        cells[transform + index] = cells[transform + index] * spectrum % p
        counts.in_place_spectral_multiplications += 1
    ntt_segment(
        cells, transform, width, field.convolution_root, p, True, counts.transform
    )
    scalar, projection = boundary_from_transient_convolution(
        field, program, boundary, cells, counts
    )
    cells[0] = (cells[0] + scalar_sign * scalar) % p

    ntt_segment(
        cells, transform, width, field.convolution_root, p, False, counts.transform
    )
    inverse_commitment = hashlib.sha256()
    for index in range(width):
        spectrum = streamed_kernel_spectrum_value(field, index, counts)
        if spectrum == 0:
            fail("inverse rematerialized a singular public kernel spectrum")
        update_spectrum_commitment(inverse_commitment, index, spectrum)
        if omit_spectral_frequency is not None and index == (
            omit_spectral_frequency % width
        ):
            continue
        cells[transform + index] = (
            cells[transform + index] * pow(spectrum, -1, p)
        ) % p
        counts.in_place_spectral_multiplications += 1
        counts.public_kernel_spectrum_inversions += 1
    ntt_segment(
        cells, transform, width, field.convolution_root, p, True, counts.transform
    )
    initialize_source(field, cells, counts.transform, -1)
    if any(cells[index] for index in range(1, len(cells))):
        fail("one-segment transform workspace did not uncompute to zero")
    if forward_commitment.digest() != inverse_commitment.digest():
        fail("streamed public kernel spectrum changed across rematerialization")
    return counts, projection, forward_commitment.hexdigest()


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
    forward, projection, spectrum_commitment = one_segment_scalar_shear(
        field, program, boundary, cells, 1
    )
    projected = cells[0]
    resident_commitment = digest(cells)
    if projected != expected_boundary:
        fail("one-segment boundary differs from descriptor reference")
    persisted = projected
    inverse, inverse_projection, inverse_commitment = one_segment_scalar_shear(
        field, program, boundary, cells, -1
    )
    if any(cells) or id(cells) != backing:
        fail("one-segment carrier did not restore on the same backing")
    if spectrum_commitment != inverse_commitment:
        fail("kernel spectrum commitment differs across inverse execution")
    return {
        "projected_final_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "pre_state_commitment": before,
        "resident_scalar_commitment": resident_commitment,
        "post_state_commitment": digest(cells),
        "streamed_kernel_spectrum_commitment": spectrum_commitment,
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
    primary = execute_cycle(field, program, boundary, cells, primary_reference)
    reused = execute_cycle(
        field, second_program, second_boundary, cells, second_reference
    )
    fresh = execute_cycle(
        field,
        second_program,
        second_boundary,
        [0] * offsets["total_carrier_field_cells"],
        second_reference,
    )
    if (
        reused["projected_final_boundary_scalar"]
        != fresh["projected_final_boundary_scalar"]
        or reused["forward_counts"] != fresh["forward_counts"]
        or id(cells) != backing
    ):
        fail("restored-carrier reuse differs from fresh one-segment execution")

    missing = [0] * offsets["total_carrier_field_cells"]
    one_segment_scalar_shear(field, program, boundary, missing, 1)
    missing_inverse_fails = any(missing)
    wrong = missing[:]
    wrong_program = PublicProgram(
        (program.determinant_character + 1) % h,
        program.scale_character,
    )
    one_segment_scalar_shear(field, wrong_program, boundary, wrong, -1)
    wrong_inverse_fails = any(wrong)
    omitted = [0] * offsets["total_carrier_field_cells"]
    one_segment_scalar_shear(
        field, program, boundary, omitted, 1, omit_spectral_frequency=0
    )
    frequency_zero_omission_changes_boundary = omitted[0] != primary[
        "projected_final_boundary_scalar"
    ]
    singular_kernel_rejected = False
    try:
        one_segment_scalar_shear(
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
        one_segment_scalar_shear(field, program, boundary, [], 1)
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
        fail("one-segment mutation control failed")

    expected_butterflies = 2 * field.convolution_width * (
        field.convolution_width.bit_length() - 1
    )
    expected_kernel_terms = 2 * field.convolution_width * (2 * h - 1)
    forward_counts = primary["forward_counts"]
    if (
        forward_counts["transform"]["ntt_calls"] != 4
        or forward_counts["transform"]["ntt_butterflies"] != expected_butterflies
        or forward_counts["streamed_kernel_nonzero_terms"]
        != expected_kernel_terms
    ):
        fail("one-segment state/work law mismatch")
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
            "rank_and_square_class": list(
                rank_and_square_class(boundary.coordinates, q)
            ),
        },
        "layout": offsets,
        "carrier_capacity_bits": offsets["total_carrier_field_cells"]
        * p.bit_length(),
        "m184_carrier_field_cells": 1 + 2 * field.convolution_width,
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
            "fresh_restored_boundary_agrees": reused[
                "projected_final_boundary_scalar"
            ]
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
            "SOFTWARE_WITH_ONE_SCALAR_PLUS_ONE_M_CELL_TRANSFORM_SEGMENT_"
            "AND_STREAMED_PUBLIC_KERNEL_SPECTRUM"
        ),
        "transaction_cases": cases,
        "observed_resource_law": {
            "convolution_width": "LEAST_POWER_OF_TWO_AT_LEAST_2Q_MINUS3",
            "logical_carrier_field_cells": "ONE_PLUS_M",
            "resident_after_forward_field_cells": 1,
            "retained_public_kernel_spectrum_cells": 0,
            "retained_gauss_descriptor_cells": 0,
            "ntt_calls_per_forward_or_inverse_compiler": 4,
            "butterflies_per_compiler": "2M_LOG2_M",
            "streamed_kernel_spectrum_values_per_compiler": "2M",
            "streamed_kernel_nonzero_terms_per_compiler": "2M_TIMES_2Q_MINUS3",
            "accepted_forward_compiler_plus_projection_work": "THETA_MQ_PLUS_M_LOG_M_PLUS_Q",
            "exact_bit_width_independent_of_q": False,
            "retained_inverse_history_cells": 0,
            "verification_runs_before_carrier_allocation": True,
            "python_integer_allocator_and_native_library_memory_included": False,
            "whole_process_memory_claimed": False,
        },
        "matched_baselines": {
            "m181_single_scalar_stream": "10_CARRIER_CELLS_AND_THETA_Q2_PER_BOUNDARY",
            "m184_fused_two_segment": "ONE_PLUS2M_CELLS_AND_THETA_M_LOG_M_PLUS_Q_WORK",
            "strongest_compact_classical": "IDENTICAL_ONE_PLUSM_STREAMED_KERNEL_BLUESTEIN_COMPILER_WITH_M181_RETAINED_AS_A_SMALLER_STATE_QUADRATIC_WORK_POINT",
            "state_advantage": False,
            "work_advantage": False,
            "new_asymptotic_pareto_point": False,
        },
        "controls": {
            "all_boundaries_match_descriptor_reference": True,
            "all_streamed_kernel_spectra_nonzero": True,
            "all_one_segment_workspaces_clear": True,
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
            "subquadratic_work": False,
            "fixed_exact_bit_width": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "ONE_SEGMENT_EXECUTION_IS_EXACT_BUT_RECOVERS_LINEAR_STATE_ONLY_"
            "BY_STREAMING_2M_PUBLIC_SPECTRUM_VALUES_AT_2M_TIMES_2QMINUS3_"
            "KERNEL_TERMS_THE_CARRIER_STILL_GROWS_AS_M_AND_THE_M181_FIXED10_"
            "CELL_QUADRATIC_STREAM_AND_IDENTICAL_CLASSICAL_PATH_REMAIN_SO_"
            "THIS_BLUESTEIN_ROUTE_HAS_NO_DISTINCT_PHASE_RESOURCE"
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
