#!/usr/bin/env python3
"""Exact position-dual phase diagonalization diagnostic for Rotor-6.

The momentum-pair scattering operator is diagonal after the exact 17-mode
Fourier transform.  This implementation lifts a topology-rematerialized
single-particle elimination network to the degree-six exchange-symmetric
polynomial.  It constructs no dense 2,277 by 2,277 operator and enumerates no
permanent assignments, but the network leaves the rotation quotient during
intermediate steps and therefore materializes all 74,613 occupation cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import growing_rotor_open_momentum_factor_closure as predecessor


GRID = predecessor.GRID
ROTORS = predecessor.ROTORS
PRIME = predecessor.PRIME
ROOT = predecessor.ROOT
Histogram = tuple[int, ...]
Operation = tuple[str, int, int, int]


@dataclass
class Work:
    transforms: int = 0
    elementary_operations: int = 0
    shear_blocks: int = 0
    shear_polynomial_terms: int = 0
    scale_cells: int = 0
    swap_cell_pairs: int = 0
    diagonal_occupation_cells: int = 0
    diagonal_character_terms: int = 0
    diagonal_pair_signature_mode_terms: int = 0
    expanded_occupation_cells: int = 0
    closed_occupation_cells: int = 0
    maximum_network_boundary_nonzero_occupation_cells: int = 0

    def add(self, other: "Work") -> None:
        for name in self.__dataclass_fields__:
            if name == "maximum_network_boundary_nonzero_occupation_cells":
                setattr(
                    self,
                    name,
                    max(getattr(self, name), getattr(other, name, 0)),
                )
            else:
                setattr(
                    self,
                    name,
                    getattr(self, name) + getattr(other, name, 0),
                )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def matrix_multiply(
    left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(GRID))
            % PRIME
            for column in range(GRID)
        ]
        for row in range(GRID)
    ]


def identity() -> list[list[int]]:
    return [
        [1 if row == column else 0 for column in range(GRID)]
        for row in range(GRID)
    ]


def apply_row_operation(matrix: list[list[int]], operation: Operation) -> None:
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
        raise ValueError("unknown elementary operation")


def inverse_operation(operation: Operation) -> Operation:
    kind, target, source, factor = operation
    if kind == "SWAP":
        return operation
    if kind == "SCALE":
        return kind, target, source, pow(factor, -1, PRIME)
    if kind == "SHEAR":
        return kind, target, source, (-factor) % PRIME
    raise ValueError("unknown elementary operation")


def compile_network(matrix: list[list[int]]) -> tuple[Operation, ...]:
    reduced = [row.copy() for row in matrix]
    reductions: list[Operation] = []
    for column in range(GRID):
        pivot = next(
            row
            for row in range(column, GRID)
            if reduced[row][column] % PRIME
        )
        if pivot != column:
            operation = ("SWAP", column, pivot, 0)
            apply_row_operation(reduced, operation)
            reductions.append(operation)
        scale = pow(reduced[column][column], -1, PRIME)
        if scale != 1:
            operation = ("SCALE", column, column, scale)
            apply_row_operation(reduced, operation)
            reductions.append(operation)
        for row in range(GRID):
            if row == column or reduced[row][column] == 0:
                continue
            operation = (
                "SHEAR",
                row,
                column,
                (-reduced[row][column]) % PRIME,
            )
            apply_row_operation(reduced, operation)
            reductions.append(operation)
    if reduced != identity():
        raise RuntimeError("single-particle elimination failed")
    network = tuple(inverse_operation(item) for item in reversed(reductions))
    reconstructed = identity()
    for operation in network:
        apply_row_operation(reconstructed, operation)
    if reconstructed != matrix:
        raise RuntimeError("single-particle network reconstruction failed")
    return network


def fourier_matrix(inverse: bool) -> list[list[int]]:
    normalization = pow(GRID, -1, PRIME) if inverse else 1
    direction = -1 if inverse else 1
    return [
        [
            normalization
            * pow(ROOT, (direction * row * column) % GRID, PRIME)
            % PRIME
            for column in range(GRID)
        ]
        for row in range(GRID)
    ]


def compile_transforms() -> tuple[tuple[Operation, ...], tuple[Operation, ...]]:
    if pow(ROOT, GRID, PRIME) != 1 or any(
        pow(ROOT, exponent, PRIME) == 1 for exponent in range(1, GRID)
    ):
        raise RuntimeError("declared phase root is not primitive order 17")
    forward = fourier_matrix(False)
    inverse = fourier_matrix(True)
    if matrix_multiply(forward, inverse) != identity():
        raise RuntimeError("exact Fourier inverse law failed")
    return compile_network(forward), compile_network(inverse)


def multinomial(item: Histogram) -> int:
    value = math.factorial(ROTORS)
    for count in item:
        value //= math.factorial(count)
    return value % PRIME


def occupation_topology() -> tuple[tuple[Histogram, ...], dict[Histogram, int]]:
    occupations = tuple(predecessor.iter_histograms(ROTORS))
    if len(occupations) != math.comb(ROTORS + GRID - 1, ROTORS):
        raise RuntimeError("occupation topology changed")
    return occupations, {item: index for index, item in enumerate(occupations)}


def expand_bracelets(
    state: list[int],
    topology: predecessor.FactorTopology,
    occupations: tuple[Histogram, ...],
    work: Work,
) -> list[int]:
    polynomial = [0] * len(occupations)
    for index, item in enumerate(occupations):
        necklace = topology.necklace_lookup[
            predecessor.canonical_code(predecessor.encode(item))
        ]
        bracelet = topology.necklace_to_bracelet[necklace]
        polynomial[index] = multinomial(item) * state[bracelet] % PRIME
        work.expanded_occupation_cells += 1
    work.maximum_network_boundary_nonzero_occupation_cells = max(
        work.maximum_network_boundary_nonzero_occupation_cells,
        sum(value != 0 for value in polynomial),
    )
    return polynomial


def apply_scale(
    polynomial: list[int],
    occupations: tuple[Histogram, ...],
    mode: int,
    factor: int,
    work: Work,
) -> None:
    powers = [pow(factor, count, PRIME) for count in range(ROTORS + 1)]
    for index, item in enumerate(occupations):
        polynomial[index] = polynomial[index] * powers[item[mode]] % PRIME
        work.scale_cells += 1


def apply_swap(
    polynomial: list[int],
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    left: int,
    right: int,
    work: Work,
) -> None:
    for index, item in enumerate(occupations):
        swapped = list(item)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        other = occupation_index[tuple(swapped)]
        if index < other:
            polynomial[index], polynomial[other] = (
                polynomial[other],
                polynomial[index],
            )
            work.swap_cell_pairs += 1


def apply_shear(
    polynomial: list[int],
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    target: int,
    source: int,
    factor: int,
    work: Work,
) -> None:
    powers = [pow(factor, count, PRIME) for count in range(ROTORS + 1)]
    for item in occupations:
        if item[source] != 0:
            continue
        total = item[target]
        # A two-mode degree-zero fibre is invariant under every shear.  It is
        # the dominant fibre count, so omitting it is both exact and material
        # to the measured implementation cost.
        if total == 0:
            continue
        old = [0] * (total + 1)
        for target_count in range(total + 1):
            member = list(item)
            member[target] = target_count
            member[source] = total - target_count
            old[target_count] = polynomial[occupation_index[tuple(member)]]
        updated = [0] * (total + 1)
        for input_target, value in enumerate(old):
            input_source = total - input_target
            for transferred in range(input_source + 1):
                output_target = input_target + transferred
                coefficient = (
                    math.comb(input_source, transferred)
                    * powers[transferred]
                    % PRIME
                )
                updated[output_target] = (
                    updated[output_target] + value * coefficient
                ) % PRIME
                work.shear_polynomial_terms += 1
        for target_count, value in enumerate(updated):
            member = list(item)
            member[target] = target_count
            member[source] = total - target_count
            polynomial[occupation_index[tuple(member)]] = value
        work.shear_blocks += 1


def apply_network(
    polynomial: list[int],
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    network: tuple[Operation, ...],
    work: Work,
) -> None:
    for kind, target, source, factor in network:
        if kind == "SWAP":
            apply_swap(
                polynomial,
                occupations,
                occupation_index,
                target,
                source,
                work,
            )
        elif kind == "SCALE":
            apply_scale(polynomial, occupations, target, factor, work)
        elif kind == "SHEAR":
            apply_shear(
                polynomial,
                occupations,
                occupation_index,
                target,
                source,
                factor,
                work,
            )
        else:
            raise RuntimeError("unknown compiled phase transform")
        work.elementary_operations += 1
    work.transforms += 1
    work.maximum_network_boundary_nonzero_occupation_cells = max(
        work.maximum_network_boundary_nonzero_occupation_cells,
        sum(value != 0 for value in polynomial),
    )


def scattering_eigenvalue(item: Histogram, step: int, tag: int, work: Work) -> int:
    result = 0
    for momentum in range(1, GRID):
        positive = 0
        negative = 0
        for coordinate, count in enumerate(item):
            positive += count * pow(
                ROOT, (momentum * coordinate) % GRID, PRIME
            )
            negative += count * pow(
                ROOT, (-momentum * coordinate) % GRID, PRIME
            )
            work.diagonal_character_terms += 2
        weight = predecessor.public_law.public_scattering_integer(
            momentum, step, tag
        )
        result += weight * (positive * negative - ROTORS)
    return result % PRIME


def apply_position_diagonal(
    polynomial: list[int],
    occupations: tuple[Histogram, ...],
    step: int,
    tag: int,
    work: Work,
) -> None:
    for index, item in enumerate(occupations):
        polynomial[index] = (
            polynomial[index] * scattering_eigenvalue(item, step, tag, work)
        ) % PRIME
        work.diagonal_occupation_cells += 1


def close_bracelets(
    polynomial: list[int],
    topology: predecessor.FactorTopology,
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    work: Work,
) -> tuple[list[int], int]:
    result = [0] * len(topology.bracelets)
    for bracelet, item in enumerate(topology.bracelets):
        result[bracelet] = (
            polynomial[occupation_index[item]]
            * pow(multinomial(item), -1, PRIME)
            % PRIME
        )
    closure_error = 0
    for index, item in enumerate(occupations):
        necklace = topology.necklace_lookup[
            predecessor.canonical_code(predecessor.encode(item))
        ]
        bracelet = topology.necklace_to_bracelet[necklace]
        expected = multinomial(item) * result[bracelet] % PRIME
        closure_error += polynomial[index] != expected
        work.closed_occupation_cells += 1
    return result, closure_error


def total_coordinate(item: Histogram) -> int:
    return sum(index * count for index, count in enumerate(item)) % GRID


def dual_scattering(
    state: list[int],
    topology: predecessor.FactorTopology,
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    forward_network: tuple[Operation, ...],
    inverse_network: tuple[Operation, ...],
    step: int,
    tag: int,
) -> tuple[list[int], Work, dict[str, int]]:
    work = Work()
    polynomial = expand_bracelets(state, topology, occupations, work)
    apply_network(
        polynomial, occupations, occupation_index, forward_network, work
    )
    final_sector_nonzero = sum(
        value != 0 and total_coordinate(item) == 0
        for value, item in zip(polynomial, occupations, strict=True)
    )
    final_off_sector_nonzero = sum(
        value != 0 and total_coordinate(item) != 0
        for value, item in zip(polynomial, occupations, strict=True)
    )
    apply_position_diagonal(polynomial, occupations, step, tag, work)
    apply_network(
        polynomial, occupations, occupation_index, inverse_network, work
    )
    result, closure_error = close_bracelets(
        polynomial, topology, occupations, occupation_index, work
    )
    return result, work, {
        "position_zero_total_sector_nonzero_cells": final_sector_nonzero,
        "position_off_sector_nonzero_cells": final_off_sector_nonzero,
        "bracelet_closure_error_cells": closure_error,
    }


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def execute_word(
    source: list[int],
    topology: predecessor.FactorTopology,
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    forward_network: tuple[Operation, ...],
    inverse_network: tuple[Operation, ...],
    operations: tuple[tuple[int, int], ...],
    *,
    reordered: bool = False,
) -> tuple[list[int], Work, dict[str, int]]:
    current = source.copy()
    total = Work()
    diagnostics: dict[str, int] = {}
    for step, tag in operations:
        if reordered:
            current, scattering, diagnostics = dual_scattering(
                current,
                topology,
                occupations,
                occupation_index,
                forward_network,
                inverse_network,
                step,
                tag,
            )
            current, diagonal = predecessor.apply_diagonal(
                current, topology, step, tag
            )
        else:
            current, diagonal = predecessor.apply_diagonal(
                current, topology, step, tag
            )
            current, scattering, diagnostics = dual_scattering(
                current,
                topology,
                occupations,
                occupation_index,
                forward_network,
                inverse_network,
                step,
                tag,
            )
        total.add(diagonal)
        total.add(scattering)
    return current, total, diagnostics


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: predecessor.FactorTopology,
    occupations: tuple[Histogram, ...],
    occupation_index: dict[Histogram, int],
    forward_network: tuple[Operation, ...],
    inverse_network: tuple[Operation, ...],
    operations: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], Work, dict[str, int]]:
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward, forward_work, diagnostics = execute_word(
        carrier.source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        operations,
    )
    carrier.target[:] = [
        (left + right) % PRIME
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    projected = predecessor.boundary(carrier.target, topology)
    missing_error = sum(value != 0 for value in carrier.target)
    inverse, inverse_work, inverse_diagnostics = execute_word(
        carrier.source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        operations,
    )
    carrier.target[:] = [
        (left - right) % PRIME
        for left, right in zip(carrier.target, inverse, strict=True)
    ]
    restoration_error = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    total = Work()
    total.add(forward_work)
    total.add(inverse_work)
    if diagnostics != inverse_diagnostics:
        raise RuntimeError("dual inverse rematerialization diagnostics differ")
    return (
        {
            "boundary": projected,
            "missing_inverse_error_field_cells": missing_error,
            "restoration_error_field_cells": restoration_error,
            "same_backing": id(carrier.source) == source_backing
            and id(carrier.target) == target_backing,
            "generation": carrier.generation,
        },
        total,
        diagnostics,
    )


def mismatch(left: list[int], right: list[int]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def network_commitment(network: tuple[Operation, ...]) -> str:
    return hashlib.sha256(
        ";".join(":".join(map(str, item)) for item in network).encode()
    ).hexdigest()


def predecessor_topology_descriptor_integers(
    topology: predecessor.FactorTopology,
) -> int:
    """Count logical scalar descriptors; aliased bracelet histograms count once."""
    return (
        sum(map(len, topology.necklaces))
        + len(topology.necklace_codes)
        + len(topology.necklace_lookup)
        + len(topology.bracelet_codes)
        + len(topology.necklace_to_bracelet)
        + len(topology.reflected_necklace)
        + len(topology.boundary_weights)
    )


def main() -> None:
    topology = predecessor.compile_topology()
    occupations, occupation_index = occupation_topology()
    forward_network, inverse_network = compile_transforms()
    source, signature_order = predecessor.source_and_signature_order(topology, 0)
    primary_word = predecessor.public_law.public_program(1, 0)
    reuse_word = predecessor.public_law.public_program(1, 4)

    primary_dual, primary_forward_work, primary_diagnostics = execute_word(
        source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        primary_word,
    )
    primary_factor, primary_factor_work = predecessor.execute_word(
        source, topology, primary_word
    )
    if mismatch(primary_dual, primary_factor):
        raise RuntimeError("position-dual scattering differs from factor path")

    carrier = Carrier(source.copy(), [0] * len(source))
    primary, primary_work, primary_tx_diagnostics = transaction(
        carrier,
        source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        primary_word,
    )
    reuse, reuse_work, reuse_diagnostics = transaction(
        carrier,
        source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        reuse_word,
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse, fresh_work, fresh_diagnostics = transaction(
        fresh,
        source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        reuse_word,
    )
    reordered, reordered_work, _ = execute_word(
        source,
        topology,
        occupations,
        occupation_index,
        forward_network,
        inverse_network,
        primary_word,
        reordered=True,
    )
    primary_commitment = predecessor.signature_order_commitment(
        primary_dual, signature_order
    )
    if (
        primary["boundary"] != predecessor.EXPECTED_PRIMARY_BOUNDARY
        or reuse["boundary"] != predecessor.EXPECTED_REUSE_BOUNDARY
        or fresh_reuse["boundary"] != reuse["boundary"]
        or primary_commitment != predecessor.EXPECTED_PRIMARY_COMMITMENT
        or primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or fresh_reuse["restoration_error_field_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or not fresh_reuse["same_backing"]
        or carrier.generation != 2
        or primary_diagnostics != primary_tx_diagnostics
        or reuse_diagnostics != fresh_diagnostics
        or mismatch(primary_dual, reordered) == 0
    ):
        raise RuntimeError("position-dual transaction or controls failed")

    active_numeric_cells = 3 * len(topology.bracelets) + len(occupations)
    occupation_topology_descriptors = len(occupations) * GRID + len(
        occupation_index
    )
    transform_network_descriptors = 4 * (
        len(forward_network) + len(inverse_network)
    )
    predecessor_topology_descriptors = predecessor_topology_descriptor_integers(
        topology
    )

    result = {
        "claim_candidate": "EXACT_F103_ROTOR6_POSITION_DUAL_PHASE_DIAGONALIZATION_REPRODUCES_THE2277_CELL_REFLECTION_PAIRED_SCATTERING_WITHOUT_A4389_CELL_OPEN_PORT_DENSE2277_SQUARED_OPERATOR_OR_PERMANENT_ENUMERATION_BUT_THE_STANDARD_EXACT_BOSONIC_TRANSFORM_MATERIALIZES_ALL74613_OCCUPATION_CELLS_AND_IS_STRICTLY_WORSE_THAN_THE_IDENTICAL_CLASSICAL_FACTOR_STREAM",
        "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_F103_ROOT72_DEPTH1_PRIMARY_REUSE_DIRECT_PROCESS_EXACT_BOSONIC_GAUSSIAN_ELIMINATION_FOURIER_NETWORK_ONLY",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_NEGATIVE_COMPACTION_RESULT",
        "phase_law": {
            "single_particle_transform": "EXACT_F103_ORDER17_PHASE_FOURIER",
            "scattering_diagonal_eigenvalue": "SUM_Q_W_Q_TIMES_S_Q_S_MINUS_Q_MINUS_ROTOR_COUNT",
            "forward_elementary_operations": len(forward_network),
            "inverse_elementary_operations": len(inverse_network),
            "forward_network_commitment": network_commitment(forward_network),
            "inverse_network_commitment": network_commitment(inverse_network),
            "dense_2277_squared_operator_cells": 0,
            "open_momentum_port_cells": 0,
            "permanent_assignment_terms": 0,
            "full_occupation_scratch_cells": len(occupations),
            "position_zero_total_sector_cells": sum(
                total_coordinate(item) == 0 for item in occupations
            ),
            "position_reflection_closed_cells": len(topology.bracelets),
            "predecessor_topology_commitment": predecessor.topology_commitment(
                topology
            ),
        },
        "parity": {
            "primary_dual_factor_mismatch_cells": mismatch(
                primary_dual, primary_factor
            ),
            "primary_boundary": primary["boundary"],
            "primary_signature_order_commitment": primary_commitment,
            "reuse_boundary": reuse["boundary"],
            "fresh_reuse_boundary": fresh_reuse["boundary"],
            "fresh_restored_reuse_boundary_parity": reuse["boundary"]
            == fresh_reuse["boundary"],
            "primary_position_diagnostics": primary_diagnostics,
            "reuse_position_diagnostics": reuse_diagnostics,
        },
        "transaction": {
            "primary_restoration_error_field_cells": primary[
                "restoration_error_field_cells"
            ],
            "reuse_restoration_error_field_cells": reuse[
                "restoration_error_field_cells"
            ],
            "fresh_reuse_restoration_error_field_cells": fresh_reuse[
                "restoration_error_field_cells"
            ],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "fresh_reuse_same_backing": fresh_reuse["same_backing"],
            "restoration_generation_after_reuse": carrier.generation,
            "restoration_method": "EXACT_TOPOLOGY_REMATERIALIZE_AND_SUBTRACT_ON_SAME_TARGET_BACKING",
            "baseline_reload_used": False,
        },
        "controls": {
            "missing_inverse_error_field_cells": primary[
                "missing_inverse_error_field_cells"
            ],
            "reordered_noncommuting_error_field_cells": mismatch(
                primary_dual, reordered
            ),
        },
        "resource_law": {
            "accepted_source_target_carrier_field_cells": 2
            * len(topology.bracelets),
            "accepted_occupation_scratch_field_cells": len(occupations),
            "accepted_output_bracelet_field_cells": len(topology.bracelets),
            "accepted_active_numeric_field_cells": active_numeric_cells,
            "public_occupation_topology_descriptor_integers": (
                occupation_topology_descriptors
            ),
            "predecessor_public_topology_descriptor_integers": (
                predecessor_topology_descriptors
            ),
            "retained_transform_network_operations": len(forward_network)
            + len(inverse_network),
            "retained_transform_network_descriptor_integers": (
                transform_network_descriptors
            ),
            "named_algorithm_field_and_descriptor_slots": active_numeric_cells
            + occupation_topology_descriptors
            + predecessor_topology_descriptors
            + transform_network_descriptors,
            "transform_compilation_peak_matrix_field_cells_lower_bound": 4
            * GRID
            * GRID,
            "primary_forward_work": primary_forward_work.as_dict(),
            "primary_forward_inverse_work": primary_work.as_dict(),
            "reuse_forward_inverse_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_work.as_dict(),
            "reordered_control_work": reordered_work.as_dict(),
            "m199_factor_forward_work": primary_factor_work.as_dict(),
            "m199_factor_conservative_named_field_cells": 11220,
            "m200_catvm_conservative_named_field_cells": 15774,
            "retained_dense_transform_cells": 0,
            "retained_inverse_history_bytes": 0,
            "full_occupation_basis_is_explicitly_enumerated": True,
            "validation_driver_retained_states_and_expression_temporaries_excluded_from_named_algorithm_slot_sum": True,
            "python_tuple_dict_bigint_allocator_interpreter_timing_and_whole_process_peaks_excluded": True,
        },
        "matched_classical_baselines": [
            "IDENTICAL_EXACT_BOSONIC_ELEMENTARY_TRANSFORM_AND_POSITION_DIAGONAL_STREAM",
            "M199_REFLECTION_PAIRED331704_TERM_OPEN_MOMENTUM_FACTOR_STREAM",
        ],
        "phase_resource_established": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "catvm_custody": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
