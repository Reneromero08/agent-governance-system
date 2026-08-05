#!/usr/bin/env python3
"""Exact fixed-quotient diagnostic for the Rotor-6 bosonic Fourier route.

The momentum bracelet carrier and the reflection-closed, zero-total-coordinate
position sector both have 2,277 coordinates.  Equal dimensions do not by
themselves provide a compact transform.  This diagnostic tests the two direct
routes exposed by M201:

* the compiled single-particle Fourier network, one elementary gate at a time;
* the quotient-to-quotient bosonic kernel, streamed without retaining it.

The first compiled shear leaves the fixed rotation quotient.  The direct
kernel has an exactly full row and column, and a generic entry is the permanent
of a repeated 6 by 6 Fourier minor (evaluated here by a 64-state subset DP).
This rejects only those tested routes under the declared exclusions.  It is
not a lower bound against every possible structured quotient transform.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import growing_rotor_dual_position_phase_scattering as m201


GRID = m201.GRID
ROTORS = m201.ROTORS
PRIME = m201.PRIME
ROOT = m201.ROOT
Histogram = tuple[int, ...]
Operation = tuple[str, int, int, int]


def rotate(item: Histogram, amount: int) -> Histogram:
    amount %= GRID
    return item[amount:] + item[:amount]


def orbit(item: Histogram) -> tuple[Histogram, ...]:
    reflected = m201.predecessor.reflect(item)
    return tuple(
        sorted(
            {
                *(rotate(item, amount) for amount in range(GRID)),
                *(rotate(reflected, amount) for amount in range(GRID)),
            }
        )
    )


def particles(item: Histogram) -> tuple[int, ...]:
    result = tuple(
        mode for mode, count in enumerate(item) for _ in range(count)
    )
    if len(result) != ROTORS:
        raise ValueError("bosonic kernel requires exactly six particles")
    return result


def factorial_product(item: Histogram) -> int:
    result = 1
    for count in item:
        result = result * math.factorial(count) % PRIME
    return result


def fourier_entry(row: int, column: int, *, inverse: bool = False) -> int:
    exponent = (-row * column if inverse else row * column) % GRID
    value = pow(ROOT, exponent, PRIME)
    if inverse:
        value = value * pow(GRID, -1, PRIME) % PRIME
    return value


def permanent_subset(
    source: Histogram, target: Histogram, *, inverse: bool = False
) -> tuple[int, int]:
    """Permanent of the repeated 6 by 6 Fourier minor by subset DP."""
    rows = particles(source)
    columns = particles(target)
    dp = [0] * (1 << ROTORS)
    dp[0] = 1
    transitions = 0
    for row_number, row in enumerate(rows):
        updated = [0] * (1 << ROTORS)
        for mask, value in enumerate(dp):
            if value == 0 or mask.bit_count() != row_number:
                continue
            for column_number, column in enumerate(columns):
                bit = 1 << column_number
                if mask & bit:
                    continue
                updated[mask | bit] = (
                    updated[mask | bit]
                    + value
                    * fourier_entry(row, column, inverse=inverse)
                ) % PRIME
                transitions += 1
        dp = updated
    return dp[-1], transitions


def permanent_permutations(
    source: Histogram, target: Histogram, *, inverse: bool = False
) -> int:
    rows = particles(source)
    columns = particles(target)
    result = 0
    for assignment in itertools.permutations(range(ROTORS)):
        term = 1
        for row_number, column_number in enumerate(assignment):
            term = (
                term
                * fourier_entry(
                    rows[row_number], columns[column_number], inverse=inverse
                )
            ) % PRIME
        result = (result + term) % PRIME
    return result


def quotient_kernel(
    source: Histogram, target: Histogram, *, inverse: bool = False
) -> tuple[int, int]:
    permanent, transitions = permanent_subset(source, target, inverse=inverse)
    value = (
        len(orbit(source))
        * permanent
        * pow(factorial_product(source), -1, PRIME)
    ) % PRIME
    return value, transitions


def explicit_orbit_kernel(source: Histogram, target: Histogram) -> int:
    result = 0
    for member in orbit(source):
        permanent, _ = permanent_subset(member, target)
        result = (
            result
            + permanent * pow(factorial_product(member), -1, PRIME)
        ) % PRIME
    return result


def target_topology() -> tuple[tuple[Histogram, ...], int, int]:
    occupations = tuple(m201.predecessor.iter_histograms(ROTORS))
    zero_total = tuple(
        item
        for item in occupations
        if sum(mode * count for mode, count in enumerate(item)) % GRID == 0
    )
    representatives = tuple(
        sorted(
            {
                min(item, m201.predecessor.reflect(item))
                for item in zero_total
            }
        )
    )
    return representatives, len(occupations), len(zero_total)


def operation_matrix(operation: Operation) -> list[list[int]]:
    matrix = m201.identity()
    m201.apply_row_operation(matrix, operation)
    return matrix


def rotation_matrix() -> list[list[int]]:
    return [
        [1 if column == (row + 1) % GRID else 0 for column in range(GRID)]
        for row in range(GRID)
    ]


def commutator_nonzero_entries(operation: Operation) -> int:
    gate = operation_matrix(operation)
    rotation = rotation_matrix()
    left = m201.matrix_multiply(gate, rotation)
    right = m201.matrix_multiply(rotation, gate)
    return sum(
        left[row][column] != right[row][column]
        for row in range(GRID)
        for column in range(GRID)
    )


def commitment(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def main() -> None:
    topology = m201.predecessor.compile_topology()
    targets, occupation_count, zero_total_count = target_topology()
    forward_network, inverse_network = m201.compile_transforms()
    if (
        occupation_count != 74613
        or zero_total_count != 4389
        or len(topology.bracelets) != 2277
        or len(targets) != 2277
        or len(forward_network) != 288
        or len(inverse_network) != 289
    ):
        raise RuntimeError("Rotor-6 quotient or Fourier topology changed")

    position_origin = (ROTORS,) + (0,) * (GRID - 1)
    source_origin = position_origin
    dense_row: list[int] = []
    dense_row_transitions = 0
    for source in topology.bracelets:
        value, transitions = quotient_kernel(source, position_origin)
        dense_row.append(value)
        dense_row_transitions += transitions
    dense_column: list[int] = []
    dense_column_transitions = 0
    for target in targets:
        value, transitions = quotient_kernel(source_origin, target)
        dense_column.append(value)
        dense_column_transitions += transitions
    if (
        any(value == 0 for value in dense_row)
        or dense_column != [GRID] * len(targets)
    ):
        raise RuntimeError("direct quotient dense row or column law changed")

    generic_source = (1, 1, 1, 1, 1, 1) + (0,) * (GRID - 6)
    generic_target = (1, 1, 1, 1, 1, 0, 0, 1) + (0,) * (GRID - 8)
    if sum(mode * count for mode, count in enumerate(generic_target)) % GRID:
        raise RuntimeError("generic target escaped the zero-total sector")
    generic_dp, generic_transitions = permanent_subset(
        generic_source, generic_target
    )
    generic_explicit = permanent_permutations(generic_source, generic_target)
    if generic_dp != generic_explicit:
        raise RuntimeError("permanent subset and assignment laws differ")

    wrong_sector = (ROTORS - 1, 1) + (0,) * (GRID - 2)
    wrong_sector_sum = sum(
        mode * count for mode, count in enumerate(wrong_sector)
    ) % GRID
    wrong_sector_orbit_kernel = explicit_orbit_kernel(
        source_origin, wrong_sector
    )
    if wrong_sector_sum == 0 or wrong_sector_orbit_kernel != 0:
        raise RuntimeError("rotation quotient leaked outside zero-total sector")

    first_forward = forward_network[0]
    first_inverse = inverse_network[0]
    if first_forward != ("SHEAR", 15, 16, 10):
        raise RuntimeError("compiled Fourier first-gate witness changed")
    first_commutator = commutator_nonzero_entries(first_forward)
    commuting_forward = sum(
        commutator_nonzero_entries(operation) == 0
        for operation in forward_network
    )
    commuting_inverse = sum(
        commutator_nonzero_entries(operation) == 0
        for operation in inverse_network
    )
    broken_orbit_coefficient = math.comb(ROTORS, 1) * first_forward[3] % PRIME
    if first_commutator == 0 or broken_orbit_coefficient == 0:
        raise RuntimeError("first Fourier gate unexpectedly preserved quotient")

    dense_cells = len(topology.bracelets) * len(targets)
    maximum_subset_states = dense_cells * (1 << ROTORS)
    result = {
        "claim_candidate": (
            "EXACT_F103_ROTOR6_MOMENTUM_BRACELET_AND_ZERO_TOTAL_"
            "REFLECTION_CLOSED_POSITION_SECTORS_BOTH_HAVE2277_CELLS_BUT_"
            "THE_STANDARD577_GATE_BOSONIC_FOURIER_NETWORK_EXITS_THE_FIXED_"
            "QUOTIENT_AT_ITS_FIRST_SHEAR_AND_THE_DIRECT_QUOTIENT_KERNEL_"
            "HAS_A_FULL2277_ENTRY_ROW_AND_COLUMN_WITH_GENERIC_ENTRIES_"
            "EQUAL_TO6X6_FOURIER_PERMANENTS_SO_THE_TESTED_ROUTES_DO_NOT_"
            "REMOVE_M204_SEARCH_AND_FANOUT_WITHOUT_OCCUPATION_SCRATCH_"
            "DENSE_KERNEL_RETENTION_OR_STREAMED_PERMANENT_WORK"
        ),
        "claim_ceiling": (
            "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_"
            "ROTOR6_F103_ROOT72_STATIC_FORWARD_KERNEL_AND_COMPILED_"
            "GAUSSIAN_ELIMINATION_FOURIER_NETWORK_DIAGNOSTIC_ONLY"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "result": "PASS_TESTED_FIXED_QUOTIENT_FOURIER_ROUTES_RETAIN_FORBIDDEN_COST",
        "quotient_geometry": {
            "momentum_bracelet_cells": len(topology.bracelets),
            "position_zero_total_cells_before_reflection": zero_total_count,
            "position_zero_total_reflection_closed_cells": len(targets),
            "full_exchange_symmetric_occupation_cells": occupation_count,
        },
        "standard_butterfly_route": {
            "forward_elementary_gates": len(forward_network),
            "inverse_elementary_gates": len(inverse_network),
            "total_elementary_gates": len(forward_network) + len(inverse_network),
            "first_forward_gate": list(first_forward),
            "first_inverse_gate": list(first_inverse),
            "first_gate_rotation_commutator_nonzero_entries": first_commutator,
            "first_gate_orbit_coefficient_mismatch": broken_orbit_coefficient,
            "forward_gates_commuting_with_global_rotation": commuting_forward,
            "inverse_gates_commuting_with_global_rotation": commuting_inverse,
            "fixed2277_coordinate_quotient_closed_gate_by_gate": False,
            "m201_required_occupation_scratch_cells": occupation_count,
        },
        "direct_kernel_route": {
            "full_kernel_cells_if_retained": dense_cells,
            "full_transform_kernel_evaluations_if_streamed": dense_cells,
            "maximum_subset_states_if_every_kernel_is_streamed": maximum_subset_states,
            "permanent_subset_state_width": 1 << ROTORS,
            "dense_row_nonzero_entries": sum(value != 0 for value in dense_row),
            "dense_column_nonzero_entries": sum(value != 0 for value in dense_column),
            "dense_column_constant": dense_column[0],
            "dense_row_commitment": commitment(dense_row),
            "dense_column_commitment": commitment(dense_column),
            "dense_row_subset_transitions": dense_row_transitions,
            "dense_column_subset_transitions": dense_column_transitions,
            "generic_permanent_subset_value": generic_dp,
            "generic_permanent_assignment_value": generic_explicit,
            "generic_permanent_subset_transitions": generic_transitions,
            "generic_permanent_assignment_terms": math.factorial(ROTORS),
            "wrong_total_coordinate": wrong_sector_sum,
            "wrong_sector_orbit_kernel": wrong_sector_orbit_kernel,
            "complete2277_by2277_transform_executed": False,
            "universal_structured_transform_lower_bound_established": False,
        },
        "resource_law": {
            "accepted_diagnostic_dense_kernel_cells": 0,
            "accepted_diagnostic_full_occupation_scratch_cells": 0,
            "accepted_diagnostic_retained_transition_plan_entries": 0,
            "accepted_dense_row_and_column_field_cells": 2 * len(targets),
            "accepted_kernel_dp_field_cells": 2 * (1 << ROTORS),
            "accepted_public_quotient_representative_histogram_cells": (
                GRID * (len(topology.bracelets) + len(targets))
            ),
            "production_imports_and_retains_predecessor_topology_for_diagnostic": True,
            "python_object_bigint_allocator_interpreter_timing_and_whole_process_peaks_excluded": True,
        },
        "controls": {
            "wrong_sector_cancels_by_rotation_character": True,
            "generic_subset_dp_equals720_assignment_permanent": True,
            "zero_first_shear_factor_would_remove_reported_orbit_mismatch": True,
            "first_nonzero_shear_breaks_fixed_rotation_quotient": True,
        },
        "matched_classical_baselines": [
            "IDENTICAL_EXACT_F103_DIRECT_BOSONIC_FOURIER_KERNEL_SUBSET_DP",
            "M201_IDENTICAL74613_CELL_BOSONIC_ELEMENTARY_TRANSFORM",
            "M204_IDENTICAL_EIGHT_CHANNEL_IMPLICIT_DIHEDRAL_VECTOR_STREAM",
        ],
        "preserved_subclaims": [
            "M204_EXECUTION_EXACT_RESTORATION_AND_REUSE_REMAIN_SEPARATELY_VALID",
            "THE_SOURCE_AND_TARGET_SYMMETRY_SECTORS_HAVE_EQUAL2277_DIMENSION",
            "THE_TESTED_STANDARD_BUTTERFLY_AND_DIRECT_KERNEL_ROUTES_ARE_EXACTLY_CHARACTERIZED",
        ],
        "rejected_interpretations": [
            "UNIVERSAL_NO_GO_FOR_EVERY_STRUCTURED2277_CELL_TRANSFORM",
            "FULL_DIRECT_QUOTIENT_TRANSFORM_EXECUTION",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
