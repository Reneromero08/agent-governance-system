#!/usr/bin/env python3
"""Independent one-segment oracle using direct public-kernel DFT sums.

This file imports only the frozen M184 *oracle* field and boundary primitives,
never production code.  It reconstructs the successor transaction separately:
recursive out-of-place source transforms plus direct per-frequency kernel DFT
sums, with fresh/restored parity and q=5/q=7 direct relation checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from growing_prime_fused_two_segment_bluestein_boundary_compiler_independent_oracle import (
    CASES,
    PRIMARY_POINT,
    Field,
    boundary,
    digest,
    direct_gauss,
    direct_relation_sum,
    fail,
    log_map,
    recursive_transform,
    seed_segments,
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


def direct_kernel_spectrum_value(
    field: Field, kernel: list[int], frequency: int
) -> int:
    return sum(
        value
        * pow(field.convolution_root, frequency * index, field.p)
        for index, value in enumerate(kernel)
        if value
    ) % field.p


def commitment(values: list[int]) -> str:
    hasher = hashlib.sha256()
    for index, value in enumerate(values):
        if index:
            hasher.update(b",")
        hasher.update(str(value).encode("ascii"))
    return hasher.hexdigest()


def one_segment_update(
    field: Field,
    logs: dict[int, int],
    determinant_index: int,
    scale_index: int,
    point: tuple[int, ...],
    target_scale: int,
    cells: list[int],
    scalar_sign: int,
    omit_frequency: int | None = None,
    force_singular_frequency: int | None = None,
) -> dict[str, Any]:
    width = field.convolution_width
    if len(cells) != 1 + width:
        fail("independent one-segment carrier size mismatch")
    if any(cells[1:]):
        fail("independent one-segment workspace did not enter zero")
    left_seed, kernel_seed = seed_segments(field)
    left_hat = recursive_transform(
        left_seed, field.convolution_root, field.p
    )
    cells[1:] = left_hat
    spectrum_values: list[int] = []
    for frequency in range(width):
        spectrum = direct_kernel_spectrum_value(
            field, kernel_seed, frequency
        )
        if force_singular_frequency is not None and frequency == (
            force_singular_frequency % width
        ):
            spectrum = 0
        if spectrum == 0:
            fail("independent streamed public kernel spectrum is singular")
        spectrum_values.append(spectrum)
        if omit_frequency is None or frequency != omit_frequency % width:
            cells[1 + frequency] = (
                cells[1 + frequency] * spectrum
            ) % field.p
    convolution = recursive_transform(
        cells[1:], field.convolution_root, field.p, inverse=True
    )
    cells[1:] = convolution
    h = field.q - 1
    gauss = [
        pow(field.chirp_root, index * index, field.p)
        * convolution[index]
        % field.p
        for index in range(h)
    ]
    scalar = boundary(
        field,
        logs,
        gauss,
        determinant_index,
        scale_index,
        point,
        target_scale,
    )
    cells[0] = (cells[0] + scalar_sign * scalar) % field.p

    restored_hat = recursive_transform(
        convolution, field.convolution_root, field.p
    )
    inverse_spectrum_values: list[int] = []
    for frequency in range(width):
        spectrum = direct_kernel_spectrum_value(
            field, kernel_seed, frequency
        )
        inverse_spectrum_values.append(spectrum)
        if omit_frequency is None or frequency != omit_frequency % width:
            restored_hat[frequency] = (
                restored_hat[frequency] * pow(spectrum, -1, field.p)
            ) % field.p
    restored_left = recursive_transform(
        restored_hat, field.convolution_root, field.p, inverse=True
    )
    cells[1:] = [
        (value - seed) % field.p
        for value, seed in zip(restored_left, left_seed)
    ]
    if any(cells[1:]):
        fail("independent one-segment workspace did not uncompute")
    if spectrum_values != inverse_spectrum_values:
        fail("independent streamed spectrum changed across inverse")
    return {
        "scalar": scalar,
        "gauss": gauss,
        "streamed_kernel_spectrum_commitment": commitment(spectrum_values),
        "streamed_kernel_spectrum_nonzero": all(spectrum_values),
        "ntt_calls": 4,
        "ntt_butterflies": 2 * width * (width.bit_length() - 1),
        "streamed_kernel_spectrum_values": 2 * width,
        "streamed_kernel_nonzero_terms": 2 * width * (2 * h - 1),
    }


def oracle_case(
    q: int, p: int, determinant_index: int, scale_index: int
) -> dict[str, Any]:
    field = Field.make(q, p)
    logs = log_map(field)
    expected_gauss = direct_gauss(field, logs)
    width = field.convolution_width
    cells = [0] * (1 + width)
    backing = id(cells)
    primary = one_segment_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        cells,
        1,
    )
    if primary["gauss"] != expected_gauss:
        fail("independent one-segment Gauss values differ from direct sums")
    projected = cells[0]
    resident_commitment = digest(cells)
    persisted = projected
    inverse = one_segment_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        cells,
        -1,
    )
    if any(cells) or id(cells) != backing:
        fail("independent one-segment primary did not restore")
    if (
        primary["streamed_kernel_spectrum_commitment"]
        != inverse["streamed_kernel_spectrum_commitment"]
    ):
        fail("independent inverse spectrum commitment mismatch")

    second_det = (determinant_index + 2) % (q - 1)
    second_scale = (scale_index + 3) % (q - 1)
    second_point = (3, 0, 0, 2, 0, 1)
    second_target = 3 % q
    second_expected = boundary(
        field,
        logs,
        expected_gauss,
        second_det,
        second_scale,
        second_point,
        second_target,
    )
    reused = one_segment_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        second_target,
        cells,
        1,
    )
    reused_boundary = cells[0]
    one_segment_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        second_target,
        cells,
        -1,
    )
    fresh_cells = [0] * (1 + width)
    fresh = one_segment_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        second_target,
        fresh_cells,
        1,
    )
    fresh_boundary = fresh_cells[0]
    one_segment_update(
        field,
        logs,
        second_det,
        second_scale,
        second_point,
        second_target,
        fresh_cells,
        -1,
    )
    if (
        reused_boundary != second_expected
        or fresh_boundary != second_expected
        or any(cells)
        or any(fresh_cells)
        or id(cells) != backing
    ):
        fail("independent restored reuse differs from fresh")

    missing = [0] * (1 + width)
    one_segment_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        missing,
        1,
    )
    missing_inverse_fails = any(missing)
    wrong = missing[:]
    one_segment_update(
        field,
        logs,
        (determinant_index + 1) % (q - 1),
        scale_index,
        PRIMARY_POINT,
        2 % q,
        wrong,
        -1,
    )
    wrong_inverse_fails = any(wrong)
    omitted = [0] * (1 + width)
    one_segment_update(
        field,
        logs,
        determinant_index,
        scale_index,
        PRIMARY_POINT,
        2 % q,
        omitted,
        1,
        omit_frequency=0,
    )
    omitted_changes = omitted[0] != projected
    singular_rejected = False
    try:
        one_segment_update(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
            [0] * (1 + width),
            1,
            force_singular_frequency=0,
        )
    except RuntimeError:
        singular_rejected = True
    null_rejected = False
    try:
        one_segment_update(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
            [],
            1,
        )
    except RuntimeError:
        null_rejected = True
    if not all(
        (
            missing_inverse_fails,
            wrong_inverse_fails,
            omitted_changes,
            singular_rejected,
            null_rejected,
        )
    ):
        fail("independent mutation control failed")

    direct_check = None
    if q in (5, 7):
        direct_value, terms = direct_relation_sum(
            field,
            logs,
            determinant_index,
            scale_index,
            PRIMARY_POINT,
            2 % q,
        )
        direct_check = {
            "direct_boundary_scalar": direct_value,
            "nonzero_source_terms": terms,
            "matches_one_segment_boundary": direct_value == projected,
        }
        if direct_value != projected:
            fail("direct original relation differs from one-segment boundary")

    return {
        "q": q,
        "auxiliary_prime": p,
        "convolution_width": width,
        "logical_carrier_field_cells": 1 + width,
        "carrier_capacity_bits": (1 + width) * p.bit_length(),
        "projected_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "resident_scalar_commitment": resident_commitment,
        "streamed_kernel_spectrum_commitment": primary[
            "streamed_kernel_spectrum_commitment"
        ],
        "direct_gauss_commitment": commitment(expected_gauss),
        "one_segment_gauss_matches_direct": primary["gauss"] == expected_gauss,
        "expected_ntt_calls_per_compiler": primary["ntt_calls"],
        "expected_butterflies_per_compiler": primary["ntt_butterflies"],
        "expected_streamed_kernel_values_per_compiler": primary[
            "streamed_kernel_spectrum_values"
        ],
        "expected_streamed_kernel_terms_per_compiler": primary[
            "streamed_kernel_nonzero_terms"
        ],
        "exact_same_backing_restoration": True,
        "actual_restored_backing_reuse_matches_fresh": reused_boundary
        == fresh_boundary,
        "reuse_resource_signature_matches_fresh": reused["ntt_calls"]
        == fresh["ntt_calls"],
        "reused_boundary_scalar": reused_boundary,
        "fresh_boundary_scalar": fresh_boundary,
        "missing_inverse_fails": missing_inverse_fails,
        "wrong_program_inverse_fails": wrong_inverse_fails,
        "frequency_zero_omission_changes_boundary": omitted_changes,
        "singular_kernel_spectrum_rejected": singular_rejected,
        "singular_rejection_atomic_rollback_claimed": False,
        "null_carrier_rejected": null_rejected,
        "direct_relation_check": direct_check,
    }


def build_result() -> dict[str, Any]:
    cases = [oracle_case(*case) for case in CASES]
    return {
        "claim": CLAIM,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": (
            "FOURTEEN_DECLARED_FIELD_PROGRAM_CASES_SEPARATE_RECURSIVE_NTT_"
            "DIRECT_PER_FREQUENCY_KERNEL_DFT_DIRECT_GAUSS_AND_ONE_SEGMENT_"
            "TRANSACTION_PARITY_WITH_Q5_Q7_DIRECT_ORIGINAL_RELATION_"
            "BOUNDARY_REEXECUTION"
        ),
        "oracle_dependency": (
            "IMPORTS_ONLY_FROZEN_M184_INDEPENDENT_ORACLE_FIELD_BOUNDARY_"
            "AND_DIRECT_RELATION_PRIMITIVES_NEVER_PRODUCTION_CODE"
        ),
        "oracle_cases": cases,
        "controls": {
            "all_one_segment_gauss_values_match_direct": True,
            "all_streamed_kernel_spectra_nonzero": True,
            "all_exact_same_backing_restorations_pass": True,
            "all_reuses_match_fresh": True,
            "all_missing_inverses_fail": True,
            "all_wrong_program_inverses_fail": True,
            "frequency_zero_omission_changes_all_declared_boundaries": True,
            "all_forced_singular_kernel_spectra_rejected": True,
            "all_null_carriers_rejected": True,
            "q5_q7_direct_original_relation_boundaries_match": True,
        },
        "observed_resource_law": {
            "production_logical_carrier_field_cells": "ONE_PLUS_M",
            "resident_after_forward_field_cells": 1,
            "retained_public_kernel_spectrum_cells": 0,
            "ntt_calls_per_compiler": 4,
            "butterflies_per_compiler": "2M_LOG2_M",
            "streamed_kernel_spectrum_values_per_compiler": "2M",
            "streamed_kernel_nonzero_terms_per_compiler": "2M_TIMES_2Q_MINUS3",
            "oracle_materializes_kernel_seed_for_verification_only": True,
            "oracle_uses_recursive_out_of_place_transform_buffers": True,
            "oracle_reports_production_logical_schedule_not_oracle_peak_memory": True,
            "allocated_state_is_sublinear_in_q": False,
            "subquadratic_work": False,
        },
        "strict_boundaries": {
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "catvm_custody": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
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
