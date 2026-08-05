#!/usr/bin/env python3
"""Independent oracle for the Rotor-6 position-dual scattering diagnostic.

The oracle never imports the production position-dual implementation.  It
uses the separately verified direct/factor reference from M199 for the final
Rotor-6 state, reconstructs the Fourier networks independently, exhaustively
checks the position-diagonal identity on all 74,613 occupations, and derives
the lifted-network work law combinatorially.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import growing_rotor_open_momentum_factor_independent_oracle as reference


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
Operation = tuple[str, int, int, int]
Histogram = tuple[int, ...]


def identity() -> list[list[int]]:
    return [
        [1 if row == column else 0 for column in range(GRID)]
        for row in range(GRID)
    ]


def multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(GRID))
            % PRIME
            for column in range(GRID)
        ]
        for row in range(GRID)
    ]


def fourier(root: int, inverse: bool, *, normalize: bool = True) -> list[list[int]]:
    scale = pow(GRID, -1, PRIME) if inverse and normalize else 1
    sign = -1 if inverse else 1
    return [
        [
            scale * pow(root, (sign * row * column) % GRID, PRIME) % PRIME
            for column in range(GRID)
        ]
        for row in range(GRID)
    ]


def row_operation(matrix: list[list[int]], operation: Operation) -> None:
    kind, target, source, factor = operation
    if kind == "SWAP":
        matrix[target], matrix[source] = matrix[source], matrix[target]
    elif kind == "SCALE":
        matrix[target] = [factor * value % PRIME for value in matrix[target]]
    elif kind == "SHEAR":
        matrix[target] = [
            (value + factor * source_value) % PRIME
            for value, source_value in zip(
                matrix[target], matrix[source], strict=True
            )
        ]
    else:
        raise ValueError("unknown row operation")


def invert_operation(operation: Operation) -> Operation:
    kind, target, source, factor = operation
    if kind == "SWAP":
        return operation
    if kind == "SCALE":
        return kind, target, source, pow(factor, -1, PRIME)
    if kind == "SHEAR":
        return kind, target, source, (-factor) % PRIME
    raise ValueError("unknown row operation")


def independent_network(matrix: list[list[int]]) -> tuple[Operation, ...]:
    working = [row.copy() for row in matrix]
    reduction: list[Operation] = []
    for column in range(GRID):
        pivot = min(
            row
            for row in range(column, GRID)
            if working[row][column] % PRIME
        )
        if pivot != column:
            operation = ("SWAP", column, pivot, 0)
            row_operation(working, operation)
            reduction.append(operation)
        factor = pow(working[column][column], -1, PRIME)
        if factor != 1:
            operation = ("SCALE", column, column, factor)
            row_operation(working, operation)
            reduction.append(operation)
        for row in range(GRID):
            if row == column or working[row][column] == 0:
                continue
            operation = (
                "SHEAR",
                row,
                column,
                (-working[row][column]) % PRIME,
            )
            row_operation(working, operation)
            reduction.append(operation)
    if working != identity():
        raise RuntimeError("independent Fourier elimination failed")
    network = tuple(invert_operation(item) for item in reversed(reduction))
    rebuilt = identity()
    for operation in network:
        row_operation(rebuilt, operation)
    if rebuilt != matrix:
        raise RuntimeError("independent Fourier network reconstruction failed")
    return network


def network_commitment(network: tuple[Operation, ...]) -> str:
    return hashlib.sha256(
        ";".join(":".join(map(str, item)) for item in network).encode()
    ).hexdigest()


def matrix_mismatch(left: list[list[int]], right: list[list[int]]) -> int:
    return sum(
        left[row][column] != right[row][column]
        for row in range(GRID)
        for column in range(GRID)
    )


def spectral_pair(item: Histogram, momentum: int) -> int:
    positive = sum(
        count * pow(ROOT, momentum * coordinate % GRID, PRIME)
        for coordinate, count in enumerate(item)
    )
    negative = sum(
        count * pow(ROOT, -momentum * coordinate % GRID, PRIME)
        for coordinate, count in enumerate(item)
    )
    return (positive * negative - ROTORS) % PRIME


def explicit_ordered_pair(item: Histogram, momentum: int) -> int:
    result = 0
    occupied = [
        (coordinate, count)
        for coordinate, count in enumerate(item)
        if count
    ]
    for left, left_count in occupied:
        for right, right_count in occupied:
            multiplicity = left_count * (
                right_count - (1 if left == right else 0)
            )
            result += multiplicity * pow(
                ROOT, momentum * (left - right) % GRID, PRIME
            )
    return result % PRIME


@dataclass(frozen=True)
class SpectralAudit:
    occupations: int
    zero_total_coordinate: int
    pair_identity_mismatches: int
    primary_weighted_identity_mismatches: int
    reuse_weighted_identity_mismatches: int
    primary_omit_particle_correction_mutations: int
    reuse_omit_particle_correction_mutations: int


def spectral_audit() -> SpectralAudit:
    occupations = 0
    zero_total = 0
    pair_mismatches = 0
    primary_mismatches = 0
    reuse_mismatches = 0
    primary_omit_mutations = 0
    reuse_omit_mutations = 0
    for item in reference.histograms(ROTORS):
        occupations += 1
        zero_total += sum(
            coordinate * count for coordinate, count in enumerate(item)
        ) % GRID == 0
        spectral = []
        explicit = []
        uncorrected = []
        for momentum in range(1, GRID):
            spectral_value = spectral_pair(item, momentum)
            explicit_value = explicit_ordered_pair(item, momentum)
            spectral.append(spectral_value)
            explicit.append(explicit_value)
            uncorrected.append((spectral_value + ROTORS) % PRIME)
            pair_mismatches += spectral_value != explicit_value
        for tag, mismatch_name in ((0, "primary"), (4, "reuse")):
            weights = [
                reference.scattering_weight(momentum, 0, tag)
                for momentum in range(1, GRID)
            ]
            left = sum(
                weight * value for weight, value in zip(weights, spectral)
            ) % PRIME
            right = sum(
                weight * value for weight, value in zip(weights, explicit)
            ) % PRIME
            omitted = sum(
                weight * value for weight, value in zip(weights, uncorrected)
            ) % PRIME
            if mismatch_name == "primary":
                primary_mismatches += left != right
                primary_omit_mutations += left != omitted
            else:
                reuse_mismatches += left != right
                reuse_omit_mutations += left != omitted
    return SpectralAudit(
        occupations=occupations,
        zero_total_coordinate=zero_total,
        pair_identity_mismatches=pair_mismatches,
        primary_weighted_identity_mismatches=primary_mismatches,
        reuse_weighted_identity_mismatches=reuse_mismatches,
        primary_omit_particle_correction_mutations=primary_omit_mutations,
        reuse_omit_particle_correction_mutations=reuse_omit_mutations,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def main() -> None:
    if pow(ROOT, GRID, PRIME) != 1 or any(
        pow(ROOT, exponent, PRIME) == 1 for exponent in range(1, GRID)
    ):
        raise RuntimeError("independent phase root check failed")
    forward_matrix = fourier(ROOT, False)
    inverse_matrix = fourier(ROOT, True)
    forward_network = independent_network(forward_matrix)
    inverse_network = independent_network(inverse_matrix)
    inverse_mismatch = matrix_mismatch(
        multiply(forward_matrix, inverse_matrix), identity()
    )
    missing_normalization_mismatch = matrix_mismatch(
        multiply(forward_matrix, fourier(ROOT, True, normalize=False)),
        identity(),
    )
    nonprimitive_root_mismatch = matrix_mismatch(
        multiply(fourier(1, False), fourier(1, True)), identity()
    )

    audit = spectral_audit()
    topology = reference.compile_topology()
    plans = reference.compile_one_body_plans(topology)
    source = reference.source_state(topology, 0)
    primary_word = reference.public_program(1, 0)
    reuse_word = reference.public_program(1, 4)
    wrong_word = reference.public_program(1, 1)
    direct_operator = reference.compile_direct_operator(
        topology, *primary_word[0]
    )
    primary = reference.execute_factor(source, topology, plans, primary_word)
    direct = reference.execute_direct(
        source, topology, direct_operator, *primary_word[0]
    )
    carrier = [source.copy(), [0] * len(source)]
    primary_boundary, primary_error, primary_backing, primary_forward = (
        reference.transaction(carrier, source, topology, plans, primary_word)
    )
    reuse_boundary, reuse_error, reuse_backing, reuse_forward = (
        reference.transaction(carrier, source, topology, plans, reuse_word)
    )
    fresh = [source.copy(), [0] * len(source)]
    fresh_boundary, fresh_error, fresh_backing, fresh_forward = (
        reference.transaction(fresh, source, topology, plans, reuse_word)
    )
    wrong = reference.execute_factor(source, topology, plans, wrong_word)
    reordered = reference.execute_factor(
        source, topology, plans, primary_word, reordered=True
    )

    shears = sum(
        operation[0] == "SHEAR"
        for operation in forward_network + inverse_network
    )
    scales = sum(
        operation[0] == "SCALE"
        for operation in forward_network + inverse_network
    )
    swaps = sum(
        operation[0] == "SWAP"
        for operation in forward_network + inverse_network
    )
    blocks_per_shear = math.comb(ROTORS + GRID - 2, ROTORS) - math.comb(
        ROTORS + GRID - 3, ROTORS
    )
    terms_per_shear = sum(
        math.comb(ROTORS - degree + GRID - 3, GRID - 3)
        * (degree + 1)
        * (degree + 2)
        // 2
        for degree in range(1, ROTORS + 1)
    )
    occupation_topology_descriptors = audit.occupations * GRID + audit.occupations
    predecessor_topology_descriptors = (
        len(topology.necklaces) * GRID
        + len(topology.necklaces)
        + len(topology.necklace_lookup)
        + len(topology.bracelets)
        + len(topology.necklace_to_bracelet)
        + len(topology.reflected_necklace)
        + len(topology.boundary_weights)
    )
    network_descriptors = 4 * (len(forward_network) + len(inverse_network))
    active_numeric_cells = 3 * len(topology.bracelets) + audit.occupations

    if (
        inverse_mismatch
        or audit.occupations != 74613
        or audit.zero_total_coordinate != 4389
        or audit.pair_identity_mismatches
        or audit.primary_weighted_identity_mismatches
        or audit.reuse_weighted_identity_mismatches
        or audit.primary_omit_particle_correction_mutations == 0
        or audit.reuse_omit_particle_correction_mutations == 0
        or len(forward_network) != 288
        or len(inverse_network) != 289
        or shears != 544
        or scales != 33
        or swaps != 0
        or blocks_per_shear != 15504
        or terms_per_shear != 62187
        or mismatch(primary, direct)
        or primary != primary_forward
        or primary_boundary != 83
        or reuse_boundary != 70
        or fresh_boundary != 70
        or reuse_forward != fresh_forward
        or any((primary_error, reuse_error, fresh_error))
        or not all((primary_backing, reuse_backing, fresh_backing))
        or mismatch(primary, wrong) == 0
        or mismatch(primary, reordered) == 0
        or missing_normalization_mismatch == 0
        or nonprimitive_root_mismatch == 0
    ):
        raise RuntimeError("independent position-dual verification failed")

    print(
        json.dumps(
            {
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS_NEGATIVE_COMPACTION_RESULT",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_EXACT_BOSONIC_GAUSSIAN_ELIMINATION_FOURIER_NETWORK_ONLY",
                "fourier_verification": {
                    "forward_inverse_matrix_mismatch_cells": inverse_mismatch,
                    "forward_elementary_operations": len(forward_network),
                    "inverse_elementary_operations": len(inverse_network),
                    "forward_network_commitment": network_commitment(
                        forward_network
                    ),
                    "inverse_network_commitment": network_commitment(
                        inverse_network
                    ),
                    "forward_inverse_shears": shears,
                    "forward_inverse_scales": scales,
                    "forward_inverse_swaps": swaps,
                },
                "spectral_verification": {
                    "occupation_histograms": audit.occupations,
                    "zero_total_coordinate_cells": audit.zero_total_coordinate,
                    "pair_identity_mismatches_across_all_nonzero_momenta": audit.pair_identity_mismatches,
                    "primary_weighted_identity_mismatch_cells": audit.primary_weighted_identity_mismatches,
                    "reuse_weighted_identity_mismatch_cells": audit.reuse_weighted_identity_mismatches,
                    "primary_omit_particle_correction_mutated_occupations": audit.primary_omit_particle_correction_mutations,
                    "reuse_omit_particle_correction_mutated_occupations": audit.reuse_omit_particle_correction_mutations,
                },
                "independent_final_state": {
                    "direct_two_body_raw_terms": direct_operator.raw_terms,
                    "direct_two_body_csr_nonzeros": int(direct_operator.matrix.nnz),
                    "direct_factor_mismatch_cells": mismatch(primary, direct),
                    "primary_boundary": primary_boundary,
                    "reuse_boundary": reuse_boundary,
                    "fresh_reuse_boundary": fresh_boundary,
                    "primary_signature_order_commitment": reference.signature_commitment(
                        primary, topology
                    ),
                    "topology_commitment": reference.topology_commitment(topology),
                },
                "transaction": {
                    "primary_restoration_error_field_cells": primary_error,
                    "reuse_restoration_error_field_cells": reuse_error,
                    "fresh_reuse_restoration_error_field_cells": fresh_error,
                    "primary_same_backing": primary_backing,
                    "reuse_same_backing": reuse_backing,
                    "fresh_reuse_same_backing": fresh_backing,
                    "fresh_restored_reuse_state_agreement": reuse_forward
                    == fresh_forward,
                    "restoration_generation_after_reuse": 2,
                    "baseline_reload_used": False,
                },
                "controls": {
                    "missing_inverse_error_field_cells": sum(
                        value != 0 for value in primary
                    ),
                    "wrong_inverse_error_field_cells": mismatch(primary, wrong),
                    "reordered_noncommuting_error_field_cells": mismatch(
                        primary, reordered
                    ),
                    "missing_fourier_inverse_normalization_matrix_mismatch_cells": missing_normalization_mismatch,
                    "nonprimitive_phase_root_matrix_mismatch_cells": nonprimitive_root_mismatch,
                    "underreported_4389_cell_position_basis_shortfall_cells": audit.occupations
                    - audit.zero_total_coordinate,
                },
                "resource_derivation": {
                    "active_numeric_field_cells": active_numeric_cells,
                    "occupation_topology_descriptor_integers": occupation_topology_descriptors,
                    "predecessor_topology_descriptor_integers": predecessor_topology_descriptors,
                    "transform_network_descriptor_integers": network_descriptors,
                    "named_algorithm_field_and_descriptor_slots": active_numeric_cells
                    + occupation_topology_descriptors
                    + predecessor_topology_descriptors
                    + network_descriptors,
                    "shear_blocks_per_shear": blocks_per_shear,
                    "shear_polynomial_terms_per_shear": terms_per_shear,
                    "forward_inverse_shear_blocks": shears * blocks_per_shear,
                    "forward_inverse_shear_polynomial_terms": shears
                    * terms_per_shear,
                    "diagonal_character_terms_per_scattering": audit.occupations
                    * (GRID - 1)
                    * GRID
                    * 2,
                    "full_occupation_basis_is_explicitly_enumerated": True,
                },
                "production_position_dual_module_imported": False,
                "production_position_dual_transform_called": False,
                "prior_independent_factor_reference_reused": True,
                "network_boundary_nonzero_counts_independently_reexecuted": False,
                "matched_classical_baselines": [
                    "IDENTICAL_EXACT_BOSONIC_ELEMENTARY_TRANSFORM_AND_POSITION_DIAGONAL_STREAM",
                    "M199_REFLECTION_PAIRED331704_TERM_OPEN_MOMENTUM_FACTOR_STREAM",
                ],
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
