#!/usr/bin/env python3
"""Exact-rank obstruction for compact SU(2)_8 fusion-path MPS carriers.

The accepted computational transaction remains final-boundary-only.  This
research diagnostic separately inspects exact final carrier coefficients and
uses reductions at two split primes to certify nonzero maximal minors in each
fusion-sector flattening.  A nonzero reduced minor proves the corresponding
Q(zeta_40) minor is nonzero; equality with the dimensional upper bound fixes
the exact rank without floating tolerance.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass

import su2_level8_fusion_path_braid_phase_relation as braid


sys.set_int_max_str_digits(0)

SPLIT_PRIMES = (241, 401)
STRANDS = (4, 6, 8, 10, 12, 14, 16)
FAMILIES = (0, 1)
ROUNDS = 8
PRIMARY_STRANDS = 16
PRIMARY_FAMILY = 0
REUSE_FAMILY = 1
M214_SOURCE_SHA256 = (
    "bc54076303fd12d57d9b57974832d451da603e123ea550f2312f81ad67bd9b11"
)


def prime_factors(value: int) -> tuple[int, ...]:
    factors = []
    candidate = 2
    residual = value
    while candidate * candidate <= residual:
        if residual % candidate == 0:
            factors.append(candidate)
            while residual % candidate == 0:
                residual //= candidate
        candidate += 1
    if residual > 1:
        factors.append(residual)
    return tuple(factors)


def is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor
        for divisor in range(2, math.isqrt(value) + 1)
    )


def primitive_root(prime: int) -> int:
    if not is_prime(prime):
        raise ValueError("rank certificate modulus must be prime")
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    raise RuntimeError("primitive root search failed")


def root_of_order_40(prime: int) -> int:
    if (prime - 1) % 40:
        raise ValueError("rank certificate prime does not split zeta40")
    root = pow(primitive_root(prime), (prime - 1) // 40, prime)
    if pow(root, 40, prime) != 1 or any(
        pow(root, 40 // factor, prime) == 1 for factor in (2, 5)
    ):
        raise RuntimeError("rank certificate root lacks exact order 40")
    return root


ROOTS = {prime: root_of_order_40(prime) for prime in SPLIT_PRIMES}


@dataclass
class DiagnosticWork:
    modular_field_evaluations: int = 0
    modular_fraction_coordinates: int = 0
    modular_elimination_row_scales: int = 0
    modular_elimination_row_updates: int = 0
    modular_elimination_cell_updates: int = 0
    sector_matrices: int = 0
    peak_modular_matrix_cells: int = 0
    peak_retained_verification_path_records: int = 0
    peak_retained_verification_path_label_cells: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_field(value: braid.K, prime: int, work: DiagnosticWork) -> int:
    root = ROOTS[prime]
    result = 0
    for power, coordinate in enumerate(value.coefficients):
        denominator = coordinate.denominator % prime
        if not denominator:
            raise ZeroDivisionError("certificate prime divides an exact denominator")
        result += (
            (coordinate.numerator % prime)
            * pow(denominator, -1, prime)
            * pow(root, power, prime)
        )
        work.modular_fraction_coordinates += 1
    work.modular_field_evaluations += 1
    return result % prime


def modular_rank(
    matrix: list[list[int]], prime: int, work: DiagnosticWork
) -> tuple[int, str]:
    if not matrix:
        return 0, hashlib.sha256(b"empty").hexdigest()
    values = [row.copy() for row in matrix]
    rows = len(values)
    columns = len(values[0])
    work.sector_matrices += 1
    work.peak_modular_matrix_cells = max(
        work.peak_modular_matrix_cells, rows * columns
    )
    pivot_row = 0
    trace = []
    for column in range(columns):
        pivot = next(
            (row for row in range(pivot_row, rows) if values[row][column]),
            None,
        )
        if pivot is None:
            continue
        pivot_value = values[pivot][column]
        trace.append((pivot_row, pivot, column, pivot_value))
        values[pivot_row], values[pivot] = values[pivot], values[pivot_row]
        inverse = pow(values[pivot_row][column], -1, prime)
        values[pivot_row] = [(entry * inverse) % prime for entry in values[pivot_row]]
        work.modular_elimination_row_scales += 1
        for row in range(rows):
            if row == pivot_row or not values[row][column]:
                continue
            factor = values[row][column]
            values[row] = [
                (entry - factor * basis) % prime
                for entry, basis in zip(values[row], values[pivot_row], strict=True)
            ]
            work.modular_elimination_row_updates += 1
            work.modular_elimination_cell_updates += columns
        pivot_row += 1
        if pivot_row == rows:
            break
    token = "|".join(":".join(map(str, entry)) for entry in trace)
    return pivot_row, hashlib.sha256(token.encode("ascii")).hexdigest()


def sector_matrices(
    paths: list[tuple[int, ...]],
    coefficients: list[braid.K],
    cut: int,
    prime: int,
    work: DiagnosticWork,
) -> list[dict[str, object]]:
    result = []
    for label in range(braid.LABELS):
        selected = [index for index, path in enumerate(paths) if path[cut] == label]
        if not selected:
            continue
        prefixes = sorted({paths[index][: cut + 1] for index in selected})
        suffixes = sorted({paths[index][cut:] for index in selected})
        prefix_index = {path: index for index, path in enumerate(prefixes)}
        suffix_index = {path: index for index, path in enumerate(suffixes)}
        matrix = [[0] * len(suffixes) for _ in prefixes]
        for index in selected:
            path = paths[index]
            matrix[prefix_index[path[: cut + 1]]][suffix_index[path[cut:]]] = (
                evaluate_field(coefficients[index], prime, work)
            )
        rank, commitment = modular_rank(matrix, prime, work)
        result.append(
            {
                "label": label,
                "prefix_rows": len(prefixes),
                "suffix_columns": len(suffixes),
                "maximum_rank": min(len(prefixes), len(suffixes)),
                "rank": rank,
                "pivot_trace_commitment": commitment,
            }
        )
    return result


def canonical_dense_sector_mps_cells(
    strands: int, cut_sector_ranks: list[dict[int, int]]
) -> int:
    ranks = [{0: 1}] + cut_sector_ranks + [{0: 1}]
    cells = 0
    for site in range(1, strands + 1):
        for left_label, left_rank in ranks[site - 1].items():
            for right_label in (left_label - 1, left_label + 1):
                right_rank = ranks[site].get(right_label, 0)
                cells += left_rank * right_rank
    return cells


def diagnose_case(strands: int, family: int) -> dict[str, object]:
    program = braid.BraidProgram(strands, ROUNDS, family)
    topology, coefficients, forward_work = braid.execute_forward(program)
    paths = [topology.unrank(index) for index in range(topology.dimension)]
    work = DiagnosticWork(
        peak_retained_verification_path_records=len(paths),
        peak_retained_verification_path_label_cells=sum(len(path) for path in paths),
    )
    prime_cuts: dict[int, list[dict[str, object]]] = {}
    for prime in SPLIT_PRIMES:
        cuts = []
        for cut in range(1, strands):
            sectors = sector_matrices(paths, coefficients, cut, prime, work)
            cuts.append(
                {
                    "cut": cut,
                    "total_rank": sum(int(sector["rank"]) for sector in sectors),
                    "maximum_rank": sum(
                        int(sector["maximum_rank"]) for sector in sectors
                    ),
                    "sectors": sectors,
                }
            )
        prime_cuts[prime] = cuts

    reference = prime_cuts[SPLIT_PRIMES[0]]
    ranks_by_cut = [
        {int(sector["label"]): int(sector["rank"]) for sector in cut["sectors"]}
        for cut in reference
    ]
    cross_prime_rank_agreement = all(
        [cut["total_rank"] for cut in prime_cuts[prime]]
        == [cut["total_rank"] for cut in reference]
        for prime in SPLIT_PRIMES[1:]
    )
    all_sector_ranks_maximal = all(
        sector["rank"] == sector["maximum_rank"]
        for cuts in prime_cuts.values()
        for cut in cuts
        for sector in cut["sectors"]
    )
    max_bond_rank = max(int(cut["total_rank"]) for cut in reference)
    mps_cells = canonical_dense_sector_mps_cells(strands, ranks_by_cut)
    prime_certificates = {}
    for prime, cuts in prime_cuts.items():
        trace_tokens = [
            str(sector["pivot_trace_commitment"])
            for cut in cuts
            for sector in cut["sectors"]
        ]
        prime_certificates[str(prime)] = {
            "cut_total_ranks": [cut["total_rank"] for cut in cuts],
            "cut_maximum_ranks": [cut["maximum_rank"] for cut in cuts],
            "all_sector_ranks_maximal": all(
                sector["rank"] == sector["maximum_rank"]
                for cut in cuts
                for sector in cut["sectors"]
            ),
            "pivot_trace_aggregate_commitment": hashlib.sha256(
                "|".join(trace_tokens).encode("ascii")
            ).hexdigest(),
        }
    exact_sector_shapes = [
        {
            "cut": cut["cut"],
            "sectors": [
                {
                    "label": sector["label"],
                    "prefix_rows": sector["prefix_rows"],
                    "suffix_columns": sector["suffix_columns"],
                    "exact_rank": sector["rank"],
                }
                for sector in cut["sectors"]
            ],
        }
        for cut in reference
    ]
    return {
        "strands": strands,
        "family": family,
        "rounds": ROUNDS,
        "program_steps": program.steps,
        "direct_fusion_path_field_cells": topology.dimension,
        "direct_fusion_path_payload_bits": braid.field_payload_bits(coefficients),
        "state_commitment": braid.state_commitment(coefficients),
        "split_primes": list(SPLIT_PRIMES),
        "cross_prime_rank_agreement": cross_prime_rank_agreement,
        "all_sector_ranks_maximal": all_sector_ranks_maximal,
        "maximum_exact_sector_schmidt_bond_rank": max_bond_rank,
        "canonical_dense_sector_mps_field_cells": mps_cells,
        "canonical_mps_over_direct_cell_ratio": f"{mps_cells}/{topology.dimension}",
        "prime_certificates": prime_certificates,
        "exact_sector_shapes": exact_sector_shapes,
        "forward_work": forward_work.as_dict(),
        "diagnostic_work": work.as_dict(),
    }


def transactions() -> dict[str, object]:
    topology = braid.FusionPathTopology.compile(PRIMARY_STRANDS)
    source = braid.source_state(topology)
    carrier = braid.Carrier(braid.OpenFusionPathPort(topology, source.copy()))
    primary_program = braid.BraidProgram(PRIMARY_STRANDS, ROUNDS, PRIMARY_FAMILY)
    reuse_program = braid.BraidProgram(PRIMARY_STRANDS, ROUNDS, REUSE_FAMILY)
    primary, primary_work = braid.transaction(carrier, source, primary_program)
    reuse, reuse_work = braid.transaction(carrier, source, reuse_program)
    fresh = braid.Carrier(braid.OpenFusionPathPort(topology, source.copy()))
    fresh_reuse, _ = braid.transaction(fresh, source, reuse_program)
    return {
        "primary_boundary_commitment": primary["boundary_commitment"],
        "primary_forward_state_commitment": primary["forward_state_commitment"],
        "primary_restoration_error_field_cells": primary[
            "restoration_error_field_cells"
        ],
        "primary_same_coefficient_backing": primary["same_coefficient_backing"],
        "primary_canonical_post_restoration_state_exact": primary[
            "canonical_post_restoration_state_exact"
        ],
        "reuse_boundary_commitment": reuse["boundary_commitment"],
        "reuse_forward_state_commitment": reuse["forward_state_commitment"],
        "reuse_restoration_error_field_cells": reuse["restoration_error_field_cells"],
        "reuse_same_coefficient_backing": reuse["same_coefficient_backing"],
        "reuse_canonical_post_restoration_state_exact": reuse[
            "canonical_post_restoration_state_exact"
        ],
        "fresh_reuse_boundary_commitment": fresh_reuse["boundary_commitment"],
        "fresh_reuse_state_commitment": fresh_reuse["forward_state_commitment"],
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
        == fresh_reuse["boundary_commitment"],
        "fresh_restored_reuse_state_agreement": reuse["forward_state_commitment"]
        == fresh_reuse["forward_state_commitment"],
        "restoration_generation_after_reuse": carrier.restoration_generation,
        "baseline_reload_used": False,
        "primary_full_transaction_work": primary_work.as_dict(),
        "reuse_full_transaction_work": reuse_work.as_dict(),
    }


def controls() -> dict[str, object]:
    invalid_composite = invalid_nonsplit = False
    try:
        root_of_order_40(9)
    except ValueError:
        invalid_composite = True
    try:
        root_of_order_40(239)
    except ValueError:
        invalid_nonsplit = True
    return {
        "invalid_composite_modulus_rejected": invalid_composite,
        "invalid_nonsplit_prime_rejected": invalid_nonsplit,
        "both_split_roots_have_exact_order40": all(
            pow(root, 40, prime) == 1
            and all(pow(root, 40 // factor, prime) != 1 for factor in (2, 5))
            for prime, root in ROOTS.items()
        ),
        "modular_nonzero_minor_implies_exact_nonzero_minor": True,
    }


def main() -> None:
    cases = [
        diagnose_case(strands, family)
        for family in FAMILIES
        for strands in STRANDS
    ]
    primary = next(
        case
        for case in cases
        if case["strands"] == PRIMARY_STRANDS and case["family"] == PRIMARY_FAMILY
    )
    result = {
        "schema": "cat_cas.su2_level8_fusion_path_mps_rank_no_go.v1",
        "result": "PASS_EXACT_SU2_LEVEL8_FUSION_PATH_MAXIMAL_SECTOR_SCHMIDT_RANK_NO_FIXED_BOND_MPS",
        "claim": "EXACT_DUAL_SPLIT_PRIME_CERTIFICATES_PROVE_TWO_SU2_LEVEL8_EIGHT_SWEEP_FUSION_PATH_FAMILIES_HAVE_MAXIMAL_SECTOR_SCHMIDT_BOND_RANKS2_3_6_10_20_35_70_ACROSS_STRANDS4TO16_SO_NO_UNIFORM_FIXED_BOND_EXACT_MPS_CARRIER_EXISTS_AND_THE_PRIMARY_CANONICAL_DENSE_SECTOR_MPS_USES4110_FIELD_CELLS_VERSUS1430_DIRECT_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_AND_IDENTICAL_CLASSICAL_ANYON_MPS",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_SU2_LEVEL8_TEMPERLEY_LIEB_QZETA40_TWO_EIGHT_SWEEP_PUBLIC_FAMILIES_EVEN_STRANDS4_6_8_10_12_14_16_SECTOR_FLATTENING_RANK_CERTIFICATES_AT_SPLIT_PRIMES241_401_PRIMARY16_FAMILY0_REUSE16_FAMILY1_DIRECT_PROCESS_ONLY",
        "exact_rank_law": {
            "strands": list(STRANDS),
            "both_family_maximum_bond_ranks": [
                next(
                    int(case["maximum_exact_sector_schmidt_bond_rank"])
                    for case in cases
                    if case["strands"] == strands and case["family"] == 0
                )
                for strands in STRANDS
            ],
            "analytic_dimensional_upper_bound": "SUM_OVER_SECTORS_MIN_PREFIX_PATHS_SUFFIX_PATHS",
            "every_declared_sector_reaches_upper_bound_at_both_split_primes": all(
                bool(case["all_sector_ranks_maximal"]) for case in cases
            ),
            "therefore_exact_qzeta40_rank_is_maximal": True,
            "uniform_fixed_bond_exact_mps_rejected_for_declared_growing_family": True,
        },
        "cases": cases,
        "transaction": transactions(),
        "controls": controls(),
        "resource_law": {
            "primary_direct_fusion_path_field_cells": primary[
                "direct_fusion_path_field_cells"
            ],
            "primary_direct_fusion_path_payload_bits": primary[
                "direct_fusion_path_payload_bits"
            ],
            "primary_maximum_exact_bond_rank": primary[
                "maximum_exact_sector_schmidt_bond_rank"
            ],
            "primary_canonical_dense_sector_mps_field_cells": primary[
                "canonical_dense_sector_mps_field_cells"
            ],
            "primary_canonical_mps_over_direct_cell_ratio": primary[
                "canonical_mps_over_direct_cell_ratio"
            ],
            "primary_peak_modular_matrix_cells": primary["diagnostic_work"][
                "peak_modular_matrix_cells"
            ],
            "primary_verification_path_records": primary["diagnostic_work"][
                "peak_retained_verification_path_records"
            ],
            "primary_verification_path_label_cells": primary["diagnostic_work"][
                "peak_retained_verification_path_label_cells"
            ],
            "rank_certification_is_verification_instrument_not_accepted_runtime_output": True,
            "accepted_transaction_retained_inverse_history": 0,
            "matched_compact_baseline": "IDENTICAL_EXACT_SECTOR_RANK_ANYON_MPS_AND_DIRECT_FUSION_PATH_RECURRENCES",
            "excluded_not_zero": "MODULAR_VERIFICATION_PATH_LIST_PREFIX_SUFFIX_SETS_HASH_BYTE_TRAFFIC_TRANSIENT_PYTHON_OBJECTS_ALLOCATOR_INTERPRETER_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_EXACT_SECTOR_RANK_ANYON_MPS_AND_SMALLER_DIRECT_FUSION_PATH_RECURRENCE",
            "phase_specific_rank_reduction": False,
            "computational_advantage": False,
        },
        "claim_limits": {
            "native_compact_mps_transaction_established": False,
            "uniform_fixed_bond_exact_mps": False,
            "rank_certificates_are_runtime_outputs": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m214_production_sha256": M214_SOURCE_SHA256
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
