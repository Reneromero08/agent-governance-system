#!/usr/bin/env python3
"""Reversible exact Bluestein/NTT compiler for the M180 Gauss descriptor.

M181 removed Gauss-table residency by spending quadratic work for each final
scalar. M182 found that standard multiplicative Gauss relations still leave a
growing formal generator family. This successor changes the update law: the
additive phase orbit is transformed by Bluestein's chirp identity and a
power-of-two exact NTT convolution.

The carrier is one backing with h=q-1 resident output cells and three M-cell
scratch segments, where M is the least power of two at least 2h-1. A
compute-copy-uncompute schedule returns every scratch cell to zero after the
forward compiler. The inverse rematerializes the same transform and subtracts
the actual resident descriptor. The result is a time/state tradeoff shared by
the identical classical NTT; it is not a distinct phase resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from growing_prime_mellin_gauss_streamed_recurrence_rank import (
    ProceduralField,
    PublicBoundary,
    PublicProgram,
    determinant,
    primitive_root,
    rank_and_square_class,
)


CLAIM = (
    "BOUNDED_EXACT_FOURTEEN_DECLARED_PRIME_BLUESTEIN_POWER_OF_TWO_NTT_"
    "GAUSS_PHASE_COMPILER_REPLACES_QUADRATIC_ALL_GAUSS_CHARACTER_SUM_"
    "COMPILATION_WITH_SUBQUADRATIC_M_LOG_M_EXACT_FIELD_WORK_ON_ONE_"
    "QMINUS1_RESIDENT_DESCRIPTOR_PLUS_THREE_M_CELL_REVERSIBLY_CLEARED_"
    "SCRATCH_SEGMENTS_WITH_FINAL_BOUNDARY_PROJECTION_TOPOLOGY_"
    "REMATERIALIZED_EXACT_SAME_BACKING_RESTORATION_AND_UNRELATED_REUSE_"
    "BUT_M_IS_LINEAR_IN_Q_EXACT_BIT_WIDTH_GROWS_AND_THE_IDENTICAL_"
    "CLASSICAL_BLUESTEIN_NTT_COMPILER_REMAINS"
)


# Each auxiliary prime contains roots of order q, 2(q-1), and M.
DECLARED_FIELDS = (
    (5, 41),
    (7, 337),
    (11, 5281),
    (13, 1249),
    (17, 5441),
    (19, 32833),
    (23, 16193),
    (29, 38977),
    (31, 29761),
    (37, 127873),
    (41, 78721),
    (43, 231169),
    (47, 691841),
    (53, 264577),
)


DECLARED_PROGRAMS = {
    5: PublicProgram(1, 2),
    7: PublicProgram(3, 5),
    11: PublicProgram(5, 8),
    13: PublicProgram(7, 11),
    17: PublicProgram(9, 14),
    19: PublicProgram(11, 17),
    23: PublicProgram(13, 20),
    29: PublicProgram(15, 23),
    31: PublicProgram(17, 26),
    37: PublicProgram(19, 29),
    41: PublicProgram(21, 32),
    43: PublicProgram(23, 35),
    47: PublicProgram(25, 38),
    53: PublicProgram(27, 41),
}


DECLARED_BOUNDARY = PublicBoundary((2, 0, 0, 1, 0, 1), 2)


def fail(message: str) -> None:
    raise RuntimeError(message)


def next_power_of_two(value: int) -> int:
    answer = 1
    while answer < value:
        answer *= 2
    return answer


@dataclass(frozen=True)
class TransformField:
    base: ProceduralField
    generator: int
    chirp_root: int
    convolution_root: int
    convolution_width: int

    @classmethod
    def create(cls, q: int, p: int) -> "TransformField":
        h = q - 1
        width = next_power_of_two(2 * h - 1)
        required = (q, 2 * h, width)
        if any((p - 1) % order for order in required):
            fail("auxiliary field lacks a declared transform root")
        generator = primitive_root(p)
        base = ProceduralField.create(q, p)
        chirp = pow(generator, (p - 1) // (2 * h), p)
        convolution = pow(generator, (p - 1) // width, p)
        if chirp * chirp % p != base.multiplicative_root:
            fail("chirp square does not match the character root")
        return cls(base, generator, chirp, convolution, width)


@dataclass
class TransformCounts:
    ntt_calls: int = 0
    ntt_butterflies: int = 0
    ntt_field_additions: int = 0
    ntt_field_multiplications: int = 0
    ntt_inverse_scalings: int = 0
    pointwise_shear_multiplications: int = 0
    pointwise_shear_additions: int = 0
    descriptor_shear_multiplications: int = 0
    descriptor_shear_additions: int = 0
    source_phase_exponentiations: int = 0
    chirp_exponentiations: int = 0
    initialization_multiplications: int = 0
    initialization_additions: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))

    def field_operation_total(self) -> int:
        return (
            self.ntt_field_additions
            + self.ntt_field_multiplications
            + self.ntt_inverse_scalings
            + self.pointwise_shear_multiplications
            + self.pointwise_shear_additions
            + self.descriptor_shear_multiplications
            + self.descriptor_shear_additions
            + self.initialization_multiplications
            + self.initialization_additions
        )


@dataclass
class ProjectionCounts:
    character_calls: int = 0
    character_orbit_visits: int = 0
    channel_field_multiplications: int = 0
    channel_field_additions: int = 0
    channels: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def procedural_character(
    field: ProceduralField,
    value: int,
    exponent: int,
    counts: ProjectionCounts,
) -> int:
    reduced = value % field.q
    if reduced == 0:
        fail("multiplicative character evaluated at zero")
    point = 1
    phase = 1
    step = pow(field.multiplicative_root, exponent % (field.q - 1), field.p)
    counts.character_calls += 1
    for _ in range(field.q - 1):
        counts.character_orbit_visits += 1
        if point == reduced:
            return phase
        point = point * field.q_generator % field.q
        phase = phase * step % field.p
    fail("public character orbit did not reach the requested value")


def ntt_segment(
    cells: list[int],
    offset: int,
    width: int,
    root: int,
    modulus: int,
    inverse: bool,
    counts: TransformCounts,
) -> None:
    if width & (width - 1):
        fail("radix-two NTT width is not a power of two")
    counts.ntt_calls += 1
    target = 0
    for source in range(1, width):
        bit = width >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if source < target:
            left = offset + source
            right = offset + target
            cells[left], cells[right] = cells[right], cells[left]

    stage = 2
    while stage <= width:
        stage_root = pow(root, width // stage, modulus)
        if inverse:
            stage_root = pow(stage_root, -1, modulus)
        half = stage // 2
        for block in range(0, width, stage):
            twiddle = 1
            for inner in range(half):
                left = offset + block + inner
                right = left + half
                low = cells[left]
                high = cells[right] * twiddle % modulus
                cells[left] = (low + high) % modulus
                cells[right] = (low - high) % modulus
                twiddle = twiddle * stage_root % modulus
                counts.ntt_butterflies += 1
                counts.ntt_field_additions += 2
                counts.ntt_field_multiplications += 2
        stage *= 2
    if inverse:
        inverse_width = pow(width, -1, modulus)
        for index in range(offset, offset + width):
            cells[index] = cells[index] * inverse_width % modulus
            counts.ntt_inverse_scalings += 1


def carrier_layout(field: TransformField) -> dict[str, int]:
    h = field.base.q - 1
    width = field.convolution_width
    return {
        "descriptor_offset": 0,
        "descriptor_cells": h,
        "left_offset": h,
        "kernel_offset": h + width,
        "product_offset": h + 2 * width,
        "scratch_segment_cells": width,
        "total_carrier_field_cells": h + 3 * width,
    }


def initialize_chirps(
    field: TransformField,
    cells: list[int],
    counts: TransformCounts,
    sign: int,
) -> None:
    q, p = field.base.q, field.base.p
    h = q - 1
    width = field.convolution_width
    layout = carrier_layout(field)
    left = layout["left_offset"]
    kernel = layout["kernel_offset"]
    orbit = 1
    for index in range(h):
        phase = pow(field.base.additive_root, orbit, p)
        chirp = pow(field.chirp_root, index * index, p)
        cells[left + index] = (cells[left + index] + sign * phase * chirp) % p
        anti_chirp = pow(field.chirp_root, -index * index, p)
        cells[kernel + index] = (cells[kernel + index] + sign * anti_chirp) % p
        if index:
            cells[kernel + width - index] = (
                cells[kernel + width - index] + sign * anti_chirp
            ) % p
        orbit = orbit * field.base.q_generator % q
        counts.source_phase_exponentiations += 1
        counts.chirp_exponentiations += 2
        counts.initialization_multiplications += 1
        counts.initialization_additions += 2 + int(index != 0)


def compile_descriptor_shear(
    field: TransformField,
    cells: list[int],
    descriptor_sign: int,
    counts: TransformCounts,
    omit_frequency: int | None = None,
) -> None:
    """Add or subtract the Gauss descriptor, leaving all scratch cells zero."""
    p = field.base.p
    h = field.base.q - 1
    width = field.convolution_width
    layout = carrier_layout(field)
    left = layout["left_offset"]
    kernel = layout["kernel_offset"]
    product = layout["product_offset"]
    initialize_chirps(field, cells, counts, 1)
    ntt_segment(cells, left, width, field.convolution_root, p, False, counts)
    ntt_segment(cells, kernel, width, field.convolution_root, p, False, counts)
    for index in range(width):
        cells[product + index] = (
            cells[product + index]
            + cells[left + index] * cells[kernel + index]
        ) % p
        counts.pointwise_shear_multiplications += 1
        counts.pointwise_shear_additions += 1
    ntt_segment(cells, product, width, field.convolution_root, p, True, counts)
    for frequency in range(h):
        if omit_frequency is not None and frequency == omit_frequency % h:
            continue
        chirp = pow(field.chirp_root, frequency * frequency, p)
        cells[frequency] = (
            cells[frequency]
            + descriptor_sign * chirp * cells[product + frequency]
        ) % p
        counts.chirp_exponentiations += 1
        counts.descriptor_shear_multiplications += 1
        counts.descriptor_shear_additions += 1

    # Reverse the compute portion exactly; only the descriptor shear survives.
    ntt_segment(cells, product, width, field.convolution_root, p, False, counts)
    for index in range(width):
        cells[product + index] = (
            cells[product + index]
            - cells[left + index] * cells[kernel + index]
        ) % p
        counts.pointwise_shear_multiplications += 1
        counts.pointwise_shear_additions += 1
    ntt_segment(cells, kernel, width, field.convolution_root, p, True, counts)
    ntt_segment(cells, left, width, field.convolution_root, p, True, counts)
    initialize_chirps(field, cells, counts, -1)
    if any(cells[h:]):
        fail("Bluestein scratch did not uncompute to zero")


def direct_gauss_table(field: TransformField) -> list[int]:
    q, p = field.base.q, field.base.p
    h = q - 1
    source: list[int] = []
    orbit = 1
    for _ in range(h):
        source.append(pow(field.base.additive_root, orbit, p))
        orbit = orbit * field.base.q_generator % q
    root = field.base.multiplicative_root
    return [
        sum(
            source[index] * pow(root, frequency * index % h, p)
            for index in range(h)
        )
        % p
        for frequency in range(h)
    ]


def boundary_from_descriptor(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    gauss: list[int],
) -> tuple[int, ProjectionCounts]:
    q, p = field.base.q, field.base.p
    h = q - 1
    eta = h // 2
    normalization = pow(h, -1, p)
    total_character = (program.determinant_character + program.scale_character) % h
    rank, square_class = rank_and_square_class(boundary.coordinates, q)
    counts = ProjectionCounts()
    determinant_value = determinant(boundary.coordinates, q)
    determinant_character = (
        procedural_character(field.base, determinant_value, -1, counts)
        if rank == 3
        else 0
    )
    running_determinant_character = 1
    if boundary.scale % q:
        running_scale_character = procedural_character(
            field.base, boundary.scale, -total_character, counts
        )
        scale_character_step = procedural_character(
            field.base, boundary.scale, 1, counts
        )
    else:
        running_scale_character = 0
        scale_character_step = 0
    answer = 0
    for channel in range(h):
        coefficient = (
            normalization
            * gauss[-((channel - program.determinant_character) % h)]
        ) % p
        if rank == 3:
            gamma = (
                gauss[channel] ** 2
                * gauss[(channel + eta) % h]
                * gauss[eta] ** 3
            ) % p
            determinant_factor = gamma * running_determinant_character % p
        elif channel == 0:
            determinant_factor = (
                q**6 - q**5 - q**3 + q**2
                if rank == 0
                else q**2 - q**3
            ) % p
        elif channel == eta and rank == 1:
            determinant_factor = q**2 * h * gauss[eta] ** 3 * square_class % p
        else:
            determinant_factor = 0
        scale_index = (total_character - channel) % h
        if boundary.scale % q:
            scale_factor = gauss[scale_index] * running_scale_character % p
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
        counts.channel_field_multiplications += 12 if rank == 3 else 5
        counts.channel_field_additions += 1
        counts.channels += 1
    return answer % p, counts


def digest_cells(cells: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, cells)).encode("ascii")).hexdigest()


def execute_cycle(
    field: TransformField,
    program: PublicProgram,
    boundary: PublicBoundary,
    cells: list[int],
) -> dict[str, Any]:
    if any(cells):
        fail("carrier must enter in canonical zero state")
    backing = id(cells)
    before = digest_cells(cells)
    forward_counts = TransformCounts()
    compile_descriptor_shear(field, cells, 1, forward_counts)
    h = field.base.q - 1
    descriptor = cells[:h]
    reference = direct_gauss_table(field)
    if descriptor != reference:
        fail("Bluestein descriptor differs from direct Gauss transform")
    resident_commitment = digest_cells(cells)
    projected, projection_counts = boundary_from_descriptor(
        field, program, boundary, descriptor
    )
    persisted = projected
    inverse_counts = TransformCounts()
    compile_descriptor_shear(field, cells, -1, inverse_counts)
    if any(cells) or id(cells) != backing:
        fail("carrier did not restore exactly on the same backing")
    return {
        "projected_final_boundary_scalar": projected,
        "projected_result_survives_inverse": persisted == projected,
        "pre_state_commitment": before,
        "resident_descriptor_commitment": resident_commitment,
        "post_state_commitment": digest_cells(cells),
        "exactly_restored": True,
        "same_backing": id(cells) == backing,
        "forward_counts": forward_counts.as_dict(),
        "forward_counted_field_operations": forward_counts.field_operation_total(),
        "inverse_counts": inverse_counts.as_dict(),
        "inverse_counted_field_operations": inverse_counts.field_operation_total(),
        "projection_counts": projection_counts.as_dict(),
    }


def transaction_case(q: int, p: int) -> dict[str, Any]:
    field = TransformField.create(q, p)
    layout = carrier_layout(field)
    program = DECLARED_PROGRAMS[q]
    boundary = replace(DECLARED_BOUNDARY, scale=2 % q)
    cells = [0] * layout["total_carrier_field_cells"]
    backing = id(cells)
    primary = execute_cycle(field, program, boundary, cells)

    h = q - 1
    second_program = PublicProgram(
        (program.determinant_character + 2) % h,
        (program.scale_character + 3) % h,
    )
    second_boundary = PublicBoundary((3, 0, 0, 2, 0, 1), 3 % q)
    reused = execute_cycle(field, second_program, second_boundary, cells)
    fresh_cells = [0] * layout["total_carrier_field_cells"]
    fresh = execute_cycle(field, second_program, second_boundary, fresh_cells)
    if (
        reused["projected_final_boundary_scalar"]
        != fresh["projected_final_boundary_scalar"]
        or reused["forward_counts"] != fresh["forward_counts"]
        or id(cells) != backing
    ):
        fail("restored-carrier reuse differs from fresh execution")

    missing = [0] * layout["total_carrier_field_cells"]
    compile_descriptor_shear(field, missing, 1, TransformCounts())
    missing_inverse_fails = any(missing)

    wrong = [0] * layout["total_carrier_field_cells"]
    compile_descriptor_shear(field, wrong, 1, TransformCounts())
    wrong_field = replace(
        field,
        base=replace(
            field.base,
            additive_root=pow(field.base.additive_root, 2, p),
        ),
    )
    compile_descriptor_shear(wrong_field, wrong, -1, TransformCounts())
    wrong_inverse_fails = any(wrong)

    omitted = [0] * layout["total_carrier_field_cells"]
    compile_descriptor_shear(field, omitted, 1, TransformCounts())
    compile_descriptor_shear(
        field, omitted, -1, TransformCounts(), omit_frequency=1
    )
    omitted_frequency_fails = any(omitted)

    null_carrier_rejected = False
    try:
        compile_descriptor_shear(field, [], 1, TransformCounts())
    except (IndexError, RuntimeError):
        null_carrier_rejected = True
    if not (
        missing_inverse_fails
        and wrong_inverse_fails
        and omitted_frequency_fails
        and null_carrier_rejected
    ):
        fail("transaction mutation control failed")

    forward = primary["forward_counts"]
    expected_butterflies = (
        6
        * field.convolution_width
        // 2
        * (field.convolution_width.bit_length() - 1)
    )
    if forward["ntt_calls"] != 6 or forward["ntt_butterflies"] != expected_butterflies:
        fail("transform work law mismatch")
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
        "layout": layout,
        "carrier_capacity_bits": layout["total_carrier_field_cells"] * p.bit_length(),
        "m180_materialized_table_peak_field_cells": 4 * q - 3,
        "m181_stream_workspace_field_cells": 10,
        "m181_rank3_nonzero_scale_forward_gauss_orbit_terms": (4 * h + 1) * h,
        "primary": primary,
        "unrelated_reuse": {
            "actual_restored_backing_consumed": id(cells) == backing,
            "restored_boundary": reused["projected_final_boundary_scalar"],
            "fresh_boundary": fresh["projected_final_boundary_scalar"],
            "fresh_restored_boundary_agrees": (
                reused["projected_final_boundary_scalar"]
                == fresh["projected_final_boundary_scalar"]
            ),
            "fresh_restored_resource_signature_agrees": (
                reused["forward_counts"] == fresh["forward_counts"]
            ),
            "exactly_restored_again": reused["exactly_restored"],
            "same_backing_after_reuse": id(cells) == backing,
        },
        "controls": {
            "missing_inverse_fails": missing_inverse_fails,
            "wrong_additive_phase_inverse_fails": wrong_inverse_fails,
            "omitted_frequency_inverse_fails": omitted_frequency_fails,
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
            "FOURTEEN_DECLARED_Q5_THROUGH_Q53_PRIME_FIELDS_WITH_NEW_"
            "AUXILIARY_ROOT_FIELDS_ONE_RANK3_NONZERO_SCALE_PROGRAM_PER_"
            "FIELD_DIRECT_PROCESS_EXACT_BLUESTEIN_NTT_DESCRIPTOR_"
            "TRANSACTION_AND_UNRELATED_SAME_FIELD_REUSE"
        ),
        "declared_fields": [
            {"q": q, "auxiliary_prime": p} for q, p in DECLARED_FIELDS
        ],
        "transaction_cases": cases,
        "observed_resource_law": {
            "convolution_width": "LEAST_POWER_OF_TWO_AT_LEAST_2Q_MINUS3",
            "carrier_field_cells": "Q_MINUS1_PLUS3M",
            "resident_after_forward_field_cells": "Q_MINUS1_WITH3M_ZERO_SCRATCH",
            "ntt_calls_per_forward_or_inverse_compiler": 6,
            "butterflies_per_compiler": "3M_LOG2_M",
            "compiler_field_work": "THETA_M_LOG_M",
            "boundary_projection_terms": "Q_MINUS1",
            "boundary_projection_character_calls": 3,
            "boundary_projection_character_orbit_visits_upper_bound": "3_TIMES_Q_MINUS1",
            "boundary_projection_work": "THETA_Q_WITHOUT_A_LOG_TABLE",
            "accepted_forward_compiler_plus_projection_work": "THETA_M_LOG_M_PLUS_Q",
            "full_lifecycle_compiler_work": "TWO_TIMES_FORWARD_COMPILER",
            "exact_bit_width_independent_of_q": False,
            "twiddle_table_cells": 0,
            "retained_inverse_history_cells": 0,
            "whole_process_memory_claimed": False,
            "python_object_allocator_and_pow_internals_counted": False,
        },
        "matched_baselines": {
            "m180_materialized_direct_gauss_table": "4Q_MINUS3_CELLS_AND_THETA_Q2_COMPILATION",
            "m181_single_scalar_stream": "10_CARRIER_CELLS_AND_THETA_Q2_PER_BOUNDARY",
            "strongest_compact_classical": "IDENTICAL_Q_MINUS1_PLUS3M_REVERSIBLE_BLUESTEIN_NTT_COMPILER",
            "state_advantage": False,
            "work_advantage": False,
        },
        "controls": {
            "all_direct_gauss_tables_match": True,
            "all_scratch_segments_zero_after_forward": True,
            "all_projection_character_scans_are_linear": True,
            "all_missing_inverses_fail": True,
            "all_wrong_additive_phase_inverses_fail": True,
            "all_omitted_frequency_inverses_fail": True,
            "all_null_carriers_rejected": True,
            "all_same_backing_restorations_pass": True,
            "all_unrelated_reuses_match_fresh": True,
        },
        "strict_boundaries": {
            "catvm_custody": False,
            "machine_enforced_hidden_intermediate": False,
            "fixed_exact_bit_width": False,
            "fixed_rank_or_fixed_field_cell_count": False,
            "sublinear_state": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "small_wall_crossing": False,
            "physical_waveform_execution": False,
            "replacement_of_physical_bits_with_pi": False,
            "unbounded_computation": False,
        },
        "next_obstruction": (
            "BLUESTEIN_NTT_REPAIRS_QUADRATIC_GAUSS_COMPILATION_WORK_BUT_"
            "MOVES_THE_ACCEPTED_PATH_TO_LINEAR_DESCRIPTOR_AND_SCRATCH_"
            "STATE_WITH_GROWING_EXACT_BIT_WIDTH_AND_AN_IDENTICAL_CLASSICAL_"
            "TRANSFORM_SO_THE_NEXT_REPAIR_MUST_FUSE_FINAL_BOUNDARY_"
            "PROJECTION_WITH_THE_ADDITIVE_TRANSFORM_OR_FIND_A_PHASE_NATIVE_"
            "OPERATION_NOT_IDENTITY_SIMULABLE_BY_THE_SAME_NTT"
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
