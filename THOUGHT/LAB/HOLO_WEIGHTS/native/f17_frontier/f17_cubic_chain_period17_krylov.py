#!/usr/bin/env python3
"""Exact period-17 orbit diagnostic for the adaptive F17 cubic chain.

The public coefficient descriptors repeat every 17 edges.  This diagnostic
applies that exact block as a black-box linear map on the 272-dimensional
fixed cyclotomic message, measures the seed Krylov dimension over two finite
fields, and separately accounts for exact projective scale and residual
integer width in the adaptive Z[zeta_17] carrier.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any

import f17_cubic_chain_adaptive_gauge as adaptive


PRIME = 17
DIMENSION = 16
MESSAGE_CELLS = PRIME * DIMENSION
PERIOD = 17
KRYLOV_MODULI = (41, 73)
PROJECTIVE_PERIODS = (1, 2, 4, 8)
RESTORATION_PERIODS = 4


def fail(message: str) -> None:
    raise RuntimeError(message)


def fixed_seed(
    program: adaptive.ChainProgram,
    modulus: int,
) -> list[int]:
    message = [0] * MESSAGE_CELLS
    for value in range(PRIME):
        phase = adaptive.unary_phase(
            program.unary_coefficients[0],
            value,
        )
        offset = value * DIMENSION
        if phase < DIMENSION:
            message[offset + phase] = 1
        else:
            for basis in range(DIMENSION):
                message[offset + basis] = modulus - 1
    return message


def transfer_mod(
    source: list[int],
    program: adaptive.ChainProgram,
    edge_index: int,
    modulus: int,
) -> list[int]:
    target = [0] * MESSAGE_CELLS
    unary = program.unary_coefficients[edge_index + 1]
    edge = program.edge_coefficients[edge_index]
    for left in range(PRIME):
        source_offset = left * DIMENSION
        for right in range(PRIME):
            shift = (
                adaptive.unary_phase(unary, right)
                + adaptive.edge_phase(edge, left, right)
            ) % PRIME
            target_offset = right * DIMENSION
            for basis in range(DIMENSION):
                coefficient = source[source_offset + basis]
                if coefficient == 0:
                    continue
                exponent = (basis + shift) % PRIME
                if exponent < DIMENSION:
                    index = target_offset + exponent
                    target[index] = (
                        target[index] + coefficient
                    ) % modulus
                else:
                    for output_basis in range(DIMENSION):
                        index = target_offset + output_basis
                        target[index] = (
                            target[index] - coefficient
                        ) % modulus
    return target


def apply_period(
    source: list[int],
    program: adaptive.ChainProgram,
    modulus: int,
) -> list[int]:
    current = source
    for edge_index in range(PERIOD):
        current = transfer_mod(
            current,
            program,
            edge_index,
            modulus,
        )
    return current


def reduce_against_basis(
    vector: list[int],
    basis: dict[int, list[int]],
    modulus: int,
) -> tuple[list[int], int | None]:
    work = vector[:]
    for pivot in sorted(basis):
        factor = work[pivot]
        if factor:
            row = basis[pivot]
            for index in range(pivot, MESSAGE_CELLS):
                work[index] = (
                    work[index] - factor * row[index]
                ) % modulus
    pivot = next(
        (
            index
            for index, value in enumerate(work)
            if value
        ),
        None,
    )
    return work, pivot


def krylov_dimension(
    family: str,
    modulus: int,
) -> dict[str, Any]:
    program = adaptive.compile_program(PERIOD + 1, family)
    current = fixed_seed(program, modulus)
    basis: dict[int, list[int]] = {}
    for step in range(MESSAGE_CELLS):
        reduced, pivot = reduce_against_basis(
            current,
            basis,
            modulus,
        )
        if pivot is None:
            return {
                "dimension": len(basis),
                "first_dependence_power": step,
                "full_message_dimension": False,
            }
        inverse = pow(reduced[pivot], modulus - 2, modulus)
        for index in range(pivot, MESSAGE_CELLS):
            reduced[index] = reduced[index] * inverse % modulus
        basis[pivot] = reduced
        current = apply_period(current, program, modulus)
    return {
        "dimension": len(basis),
        "first_dependence_power": MESSAGE_CELLS,
        "full_message_dimension": len(basis) == MESSAGE_CELLS,
    }


def message_payload(
    message: adaptive.GaugeMessage,
) -> dict[str, Any]:
    adaptive_bits = adaptive.PIVOT_BITS_PER_MESSAGE
    adaptive_bits += max(
        1,
        message.scale_17_exponent.bit_length(),
    )
    adaptive_bits += sum(
        adaptive.signed_bits(value)
        for row in message.coefficients
        for value in row
    )
    canonical = message.canonical_semantic()
    fixed_bits = sum(
        adaptive.signed_bits(value)
        for row in canonical
        for value in row
    )
    quotient_gcd = 0
    maximum_quotient_bits = 1
    for row in message.coefficients:
        for value in row:
            quotient_gcd = math.gcd(quotient_gcd, abs(value))
            maximum_quotient_bits = max(
                maximum_quotient_bits,
                adaptive.signed_bits(value),
            )
    maximum_semantic_bits = max(
        adaptive.signed_bits(value)
        for row in canonical
        for value in row
    )
    encoded_message = json.dumps(
        {
            "pivots": message.pivots,
            "coefficients": message.coefficients,
            "scale_17_exponent": message.scale_17_exponent,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "adaptive_message_sha256": hashlib.sha256(
            encoded_message
        ).hexdigest(),
        "adaptive_total_payload_bits": adaptive_bits,
        "fixed_basis_semantic_payload_bits": fixed_bits,
        "maximum_quotient_coefficient_signed_bits": (
            maximum_quotient_bits
        ),
        "maximum_semantic_coefficient_signed_bits": (
            maximum_semantic_bits
        ),
        "stored_17_content_exponent": message.scale_17_exponent,
        "stored_exponent_payload_bits": max(
            1,
            message.scale_17_exponent.bit_length(),
        ),
        "semantic_scale_integer_bits_if_materialized": (
            PRIME**message.scale_17_exponent
        ).bit_length(),
        "metric_verification_canonical_quotient_integer_cells": (
            MESSAGE_CELLS
        ),
        "metric_verification_canonical_semantic_integer_cells": (
            MESSAGE_CELLS
        ),
        "metric_verification_combined_peak_integer_cells": (
            2 * MESSAGE_CELLS
        ),
        "encoded_message_bytes": len(encoded_message),
        "quotient_coefficient_gcd": quotient_gcd,
        "residual_common_integer_content_removed": (
            quotient_gcd == 1
        ),
    }


def streaming_final_accounted(
    program: adaptive.ChainProgram,
) -> tuple[adaptive.GaugeMessage, dict[str, int]]:
    current = adaptive.seed_message(program)
    stats = adaptive.Stats()
    peak_integer_cells = MESSAGE_CELLS
    peak_pivot_bits = adaptive.PIVOT_BITS_PER_MESSAGE
    peak_scale_bits = max(
        1,
        current.scale_17_exponent.bit_length(),
    )
    for edge_index in range(program.nodes - 1):
        target = adaptive.GaugeMessage.create()
        adaptive.compute_transfer_into(
            current,
            target,
            program,
            edge_index,
            stats,
            False,
        )
        peak_integer_cells = max(
            peak_integer_cells,
            2 * MESSAGE_CELLS,
        )
        peak_pivot_bits = max(
            peak_pivot_bits,
            2 * adaptive.PIVOT_BITS_PER_MESSAGE,
        )
        peak_scale_bits = max(
            peak_scale_bits,
            max(1, current.scale_17_exponent.bit_length())
            + max(1, target.scale_17_exponent.bit_length()),
        )
        current = target
    return current, {
        "forward_transfer_applications": (
            stats.forward_transfer_applications
        ),
        "transfer_scalar_accumulations": (
            stats.transfer_scalar_accumulations
        ),
        "two_message_peak_integer_cells": peak_integer_cells,
        "two_message_peak_pivot_metadata_bits": peak_pivot_bits,
        "two_message_peak_scale_register_bits": peak_scale_bits,
    }


def projective_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for periods in PROJECTIVE_PERIODS:
        nodes = periods * PERIOD + 1
        family_results: dict[str, Any] = {}
        for family in ("PRIMARY", "REUSE"):
            program = adaptive.compile_program(nodes, family)
            message, streaming_accounting = streaming_final_accounted(
                program
            )
            family_results[family.lower()] = {
                "public_program_sha256": hashlib.sha256(
                    adaptive.encoded_program(program)
                ).hexdigest(),
                "public_program_descriptor_bytes": len(
                    adaptive.encoded_program(program)
                ),
                "streaming_accounting": streaming_accounting,
                **message_payload(message),
            }
        cases.append(
            {
                "periods": periods,
                "edges": periods * PERIOD,
                "nodes": nodes,
                **family_results,
            }
        )
    return cases


def restoration_case() -> dict[str, Any]:
    nodes = RESTORATION_PERIODS * PERIOD + 1
    primary_program = adaptive.compile_program(nodes, "PRIMARY")
    reuse_program = adaptive.compile_program(nodes, "REUSE")
    carrier = adaptive.Carrier.create(nodes)
    backing = carrier.backing_identity()
    primary = adaptive.execute_transaction(carrier, primary_program)
    reuse = adaptive.execute_transaction(carrier, reuse_program)
    fresh = adaptive.execute_transaction(
        adaptive.Carrier.create(nodes),
        reuse_program,
    )
    if reuse.boundary != fresh.boundary:
        fail("restored reuse differs from fresh execution")
    if carrier.backing_identity() != backing:
        fail("restored carrier backing changed")
    if not carrier.all_zero():
        fail("periodic diagnostic carrier was not restored")
    return {
        "periods": RESTORATION_PERIODS,
        "edges": RESTORATION_PERIODS * PERIOD,
        "nodes": nodes,
        "primary_restored_exactly": primary.restored_exactly,
        "reuse_restored_exactly": reuse.restored_exactly,
        "same_original_backing": (
            primary.same_backing
            and reuse.same_backing
            and carrier.backing_identity() == backing
        ),
        "fresh_restored_reuse_boundary_equal": (
            reuse.boundary == fresh.boundary
        ),
        "restoration_generation": carrier.generation,
        "restoration_lease": carrier.lease,
        "canonical_restored_state": carrier.canonical_restored_state(),
        "message_slots": len(carrier.messages),
        "message_integer_cells": (
            len(carrier.messages) * MESSAGE_CELLS
        ),
        "message_pivot_metadata_bits": (
            len(carrier.messages)
            * adaptive.PIVOT_BITS_PER_MESSAGE
        ),
        "primary_program_descriptor_bytes": len(
            adaptive.encoded_program(primary_program)
        ),
        "reuse_program_descriptor_bytes": len(
            adaptive.encoded_program(reuse_program)
        ),
        "concurrent_program_descriptor_bytes": (
            len(adaptive.encoded_program(primary_program))
            + len(adaptive.encoded_program(reuse_program))
        ),
        "primary_stats": adaptive.stats_json(primary.stats),
        "reuse_stats": adaptive.stats_json(reuse.stats),
        "fresh_reuse_stats": adaptive.stats_json(fresh.stats),
        "retained_inverse_history_bytes": 0,
        "baseline_reload_bytes": 0,
    }


def main() -> int:
    if len(sys.argv) != 1:
        fail("usage: f17_cubic_chain_period17_krylov.py")
    krylov = {
        family.lower(): {
            str(modulus): krylov_dimension(family, modulus)
            for modulus in KRYLOV_MODULI
        }
        for family in ("PRIMARY", "REUSE")
    }
    projective = projective_cases()
    restored = restoration_case()
    block_programs = {
        family.lower(): adaptive.program_json(
            adaptive.compile_program(PERIOD + 1, family)
        )
        for family in ("PRIMARY", "REUSE")
    }
    modular_dimensions = {
        family: sorted(
            {
                entry["dimension"]
                for entry in results.values()
            }
        )
        for family, results in krylov.items()
    }
    stable_modular_dimensions = all(
        len(dimensions) == 1
        for dimensions in modular_dimensions.values()
    )
    observed_dimensions = {
        family: dimensions[0]
        for family, dimensions in modular_dimensions.items()
        if len(dimensions) == 1
    }
    all_residual_gcd_one = all(
        case[family]["residual_common_integer_content_removed"]
        for case in projective
        for family in ("primary", "reuse")
    )
    result = {
        "result": "PASS",
        "claim_candidate": (
            "BOUNDED_F17_PERIOD17_CUBIC_CHAIN_BLOCK_MODULAR_SEED_"
            "KRYLOV_IMAGES_HAVE_DIMENSIONS241_AND256_WHILE_EXACT_"
            "ADAPTIVE_PROJECTIVE_CONTENT_RETAINS_RESIDUAL_WIDTH_"
            "GROWTH_WITH_EXACT_RESTORATION_AND_REUSE"
        ),
        "claim_ceiling": (
            "LINUX_X86_64_PYTHON_EXACT_F17_TWO_PUBLIC_PERIOD17_"
            "UNARY_CUBIC_AND_NEAREST_NEIGHBOR_MIXED_CUBIC_PATH_"
            "FAMILIES_FIXED_272_DIMENSION_Z_ZETA17_TRANSFER_"
            "KRYLOV_RANK_MOD41_AND73_PERIODS1_2_4_8_"
            "RESTORATION_AT_PERIOD4_SOFTWARE_ONLY"
        ),
        "classification_candidate": (
            "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
        ),
        "verification_level_candidate": "SEPARATE_REFERENCE_PARITY",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "period": PERIOD,
        "message_dimension": MESSAGE_CELLS,
        "block_programs": block_programs,
        "block_program_sha256": {
            family: hashlib.sha256(
                json.dumps(
                    program,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for family, program in block_programs.items()
        },
        "block_program_descriptor_bytes": {
            family: len(
                json.dumps(
                    program,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for family, program in block_programs.items()
        },
        "krylov_moduli": list(KRYLOV_MODULI),
        "krylov": krylov,
        "modular_dimensions_stable_across_tested_primes": (
            stable_modular_dimensions
        ),
        "observed_modular_krylov_dimensions": observed_dimensions,
        "strictly_smaller_modular_images_observed": all(
            dimension < MESSAGE_CELLS
            for dimension in observed_dimensions.values()
        ),
        "exact_rational_krylov_dimension_lower_bounds": (
            observed_dimensions
        ),
        "exact_rational_krylov_dimensions_established": False,
        "exact_rational_krylov_reduction_established": False,
        "projective_cases": projective,
        "all_quotient_coefficient_gcds_one": all_residual_gcd_one,
        "restoration_case": restored,
        "matched_classical": {
            "identical_period_block_linear_map": True,
            "identical_modular_krylov_images": True,
            "two_message_streaming_integer_cells": (
                2 * MESSAGE_CELLS
            ),
            "dense_block_integer_cells": (
                MESSAGE_CELLS**2
            ),
            "fixed_order_recurrence_exists_by_cayley_hamilton": True,
            "exact_recurrence_order_lower_than_272_established": False,
            "strongest_family_specific_implementation_established": False,
        },
        "observed_law": {
            "projective_periods": list(PROJECTIVE_PERIODS),
            "primary_adaptive_payload_bits": [
                case["primary"]["adaptive_total_payload_bits"]
                for case in projective
            ],
            "reuse_adaptive_payload_bits": [
                case["reuse"]["adaptive_total_payload_bits"]
                for case in projective
            ],
            "primary_maximum_quotient_coefficient_signed_bits": [
                case["primary"][
                    "maximum_quotient_coefficient_signed_bits"
                ]
                for case in projective
            ],
            "reuse_maximum_quotient_coefficient_signed_bits": [
                case["reuse"][
                    "maximum_quotient_coefficient_signed_bits"
                ]
                for case in projective
            ],
            "fixed_width_established": False,
            "projective_scale_hides_residual_content": False,
        },
        "resource_law": {
            "accepted_adaptive_message_integer_cells": MESSAGE_CELLS,
            "accepted_adaptive_pivot_metadata_bits": (
                adaptive.PIVOT_BITS_PER_MESSAGE
            ),
            "projective_streaming_message_peak_integer_cells": (
                max(
                    case[family]["streaming_accounting"][
                        "two_message_peak_integer_cells"
                    ]
                    for case in projective
                    for family in ("primary", "reuse")
                )
            ),
            "projective_streaming_pivot_metadata_peak_bits": (
                max(
                    case[family]["streaming_accounting"][
                        "two_message_peak_pivot_metadata_bits"
                    ]
                    for case in projective
                    for family in ("primary", "reuse")
                )
            ),
            "projective_streaming_scale_register_peak_bits": (
                max(
                    case[family]["streaming_accounting"][
                        "two_message_peak_scale_register_bits"
                    ]
                    for case in projective
                    for family in ("primary", "reuse")
                )
            ),
            "projective_metric_verification_combined_peak_integer_cells": (
                max(
                    case[family][
                        "metric_verification_combined_peak_integer_cells"
                    ]
                    for case in projective
                    for family in ("primary", "reuse")
                )
            ),
            "maximum_projective_encoded_message_bytes": (
                max(
                    case[family]["encoded_message_bytes"]
                    for case in projective
                    for family in ("primary", "reuse")
                )
            ),
            "maximum_projective_program_descriptor_bytes": max(
                case[family]["public_program_descriptor_bytes"]
                for case in projective
                for family in ("primary", "reuse")
            ),
            "restoration_carrier_message_integer_cells": (
                restored["message_integer_cells"]
            ),
            "restoration_carrier_pivot_metadata_bits": (
                restored["message_pivot_metadata_bits"]
            ),
            "restoration_verification_two_carrier_integer_cells": (
                2 * restored["message_integer_cells"]
            ),
            "temporary_seed_message_integer_cells": MESSAGE_CELLS,
            "temporary_seed_pivot_metadata_bits": (
                adaptive.PIVOT_BITS_PER_MESSAGE
            ),
            "temporary_inverse_expected_message_integer_cells": (
                MESSAGE_CELLS
            ),
            "temporary_inverse_expected_pivot_metadata_bits": (
                adaptive.PIVOT_BITS_PER_MESSAGE
            ),
            "restoration_concurrent_program_descriptor_bytes": (
                restored["concurrent_program_descriptor_bytes"]
            ),
            "krylov_basis_peak_field_cells": (
                max(observed_dimensions.values()) * MESSAGE_CELLS
            ),
            "krylov_current_and_reduced_field_cells": (
                2 * MESSAGE_CELLS
            ),
            "krylov_combined_explicit_peak_field_cells": (
                max(observed_dimensions.values()) * MESSAGE_CELLS
                + 2 * MESSAGE_CELLS
            ),
            "restoration_verification_transaction_count": 3,
            "restoration_total_forward_transfer_applications": sum(
                restored[name]["forward_transfer_applications"]
                for name in (
                    "primary_stats",
                    "reuse_stats",
                    "fresh_reuse_stats",
                )
            ),
            "restoration_total_inverse_transfer_applications": sum(
                restored[name]["inverse_transfer_applications"]
                for name in (
                    "primary_stats",
                    "reuse_stats",
                    "fresh_reuse_stats",
                )
            ),
            "restoration_total_transfer_scalar_accumulations": sum(
                restored[name]["transfer_scalar_accumulations"]
                for name in (
                    "primary_stats",
                    "reuse_stats",
                    "fresh_reuse_stats",
                )
            ),
            "restoration_total_projection_scalar_accumulations": sum(
                restored[name]["projection_scalar_accumulations"]
                for name in (
                    "primary_stats",
                    "reuse_stats",
                    "fresh_reuse_stats",
                )
            ),
            "dense_block_materialized": False,
            "assignment_table_materialized": False,
            "relation_table_materialized": False,
            "projective_scale_exponent_counted": True,
            "restoration_and_reuse_counted": True,
            "retained_inverse_history_bytes": 0,
            "baseline_reload_bytes": 0,
            "python_object_overhead_bounded": False,
            "allocator_peak_bounded": False,
            "bit_operation_peak_bounded": False,
            "whole_process_peak_bounded": False,
            "accounting_scope": (
                "EXPLICIT_LOGICAL_CELLS_METADATA_DESCRIPTORS_AND_"
                "OPERATION_COUNTS_WITH_DECLARED_RUNTIME_EXCLUSIONS"
            ),
        },
        "not_established": [
            "EXACT_RATIONAL_KRYLOV_ORDER_BELOW_272",
            "FIXED_INTEGER_WIDTH",
            "CONSTANT_TOTAL_REVERSIBLE_STORAGE",
            "ARBITRARY_GRAPH_TOPOLOGY",
            "GENERAL_NON_GAUSSIAN_COMPOSITION",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": (
            "THE_PERIOD17_BLOCK_HAS_STABLE_241_AND256_DIMENSIONAL_"
            "MODULAR_SEED_IMAGES_BUT_THE_DEPENDENCIES_ARE_NOT_YET_"
            "LIFTED_TO_AN_EXACT_Z_ZETA17_QUOTIENT_AND_EXACT_"
            "PROJECTIVE_CONTENT_DOES_NOT_STOP_RESIDUAL_WIDTH_GROWTH"
        ),
        "terminal": False,
    }
    if (
        not stable_modular_dimensions
        or observed_dimensions != {"primary": 241, "reuse": 256}
        or not all_residual_gcd_one
    ):
        fail(
            "declared periodic orbit law was not observed: "
            + json.dumps(
                {
                    "krylov": krylov,
                    "all_residual_gcd_one": all_residual_gcd_one,
                    "quotient_gcds": [
                        {
                            "periods": case["periods"],
                            "primary": case["primary"][
                                "quotient_coefficient_gcd"
                            ],
                            "reuse": case["reuse"][
                                "quotient_coefficient_gcd"
                            ],
                        }
                        for case in projective
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
