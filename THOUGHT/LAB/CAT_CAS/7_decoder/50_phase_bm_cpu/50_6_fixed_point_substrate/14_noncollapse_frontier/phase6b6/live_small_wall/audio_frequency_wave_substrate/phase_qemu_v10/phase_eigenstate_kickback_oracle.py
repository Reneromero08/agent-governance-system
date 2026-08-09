#!/usr/bin/env python3
"""M268 exact qutrit phase-eigenstate kickback oracle digital twin.

Every state decision is performed in Q(omega), where
omega**2 + omega + 1 = 0.  Normalized character and Bell states are carried
as exact density matrices, so no square root or floating-point tolerance is
part of the decision path.  The coherent oracle is an actual basis
permutation, not a precompiled phase substituted into the accepted route.

This executable is deterministic software evidence only.  Its coherent
oracle interface is stipulated.  It does not establish a physical oracle,
physical carrier custody, a query separation, or a total resource advantage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


CLAIM = "FINITE_QUDIT_PHASE_EIGENSTATE_KICKBACK_RETURNS_A_SECRET_INDEPENDENT_CHARACTER_CARRIER_EXACTLY_FOR_TWO_DISTINCT_COHERENT_ORACLE_QUERIES_WHILE_PUBLIC_LAWS_ADMIT_DIRECT_PHASE_COMPILATION_SECRET_DEPENDENT_REUSABLE_PROGRAM_STATES_REQUIRE_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_ORACLE_ACCESS_ERASES_ANY_UNIQUE_PHASE_QEMU_ADVANTAGE"
CEILING = "DETERMINISTIC_EXACT_FINITE_DIMENSIONAL_SOFTWARE_ORACLE_DIGITAL_TWIN_WITH_STIPULATED_EXTERNAL_COHERENT_QUERY_INTERFACE_NO_PHYSICAL_ORACLE_CARRIER_CUSTODY_QUERY_SEPARATION_OR_TOTAL_RESOURCE_ADVANTAGE"
RESTORATION_CLASSIFICATION = "EXACT_ALGEBRAIC_RESTORATION"
RESTORATION_SCOPE = "EXACT_CYCLOTOMIC_LOGICAL_CARRIER_AND_INERT_REFERENCE_RETURN_FOR_TWO_DISTINCT_STIPULATED_COHERENT_ORACLE_QUERIES_ON_ONE_RESIDENT_SOFTWARE_ALLOCATION_WITHOUT_PHYSICAL_ORACLE_OR_CARRIER_CUSTODY"
DISPOSITION = "EXACT_KICKBACK_AND_LOGICAL_CARRIER_REUSE_ARE_VALID_BUT_PUBLIC_DESCRIPTORS_COMPILE_DIRECTLY_SECRET_DEPENDENT_REUSABLE_PROGRAM_STATES_PAY_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_ORACLE_ACCESS_RUNS_THE_IDENTICAL_QUERY_SO_NO_UNIQUE_PHASE_RESOURCE_TOTAL_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
SUCCESSOR = "COMPACT_PHYSICAL_COHERENT_ORACLE_GENERATION_LAW_WITH_SECRET_INDEPENDENT_FINITE_ENERGY_EIGENSTATE_CARRIER_AND_EQUAL_INTERFACE_TOTAL_RESOURCE_ACCOUNTING"

SCHEMA = "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_ORACLE_V1"
MILESTONE = "M268"
D = 3
CLIENT_DIMENSION = 2
SPECTATOR_DIMENSION = 3
REFERENCE_DIMENSION = 3
PROGRAM_A = (0, 1)
PROGRAM_B = (0, 2)
JOINT_DIMS = (CLIENT_DIMENSION, D, SPECTATOR_DIMENSION, REFERENCE_DIMENSION)
CARRIER_DIMS = (D, SPECTATOR_DIMENSION, REFERENCE_DIMENSION)


@dataclass(frozen=True)
class Cyclo3:
    """The exact element ``one + omega*omega`` in Q(omega)."""

    one: Fraction
    omega: Fraction

    def __add__(self, other: Cyclo3) -> Cyclo3:
        return Cyclo3(self.one + other.one, self.omega + other.omega)

    def __neg__(self) -> Cyclo3:
        return Cyclo3(-self.one, -self.omega)

    def __sub__(self, other: Cyclo3) -> Cyclo3:
        return self + (-other)

    def __mul__(self, other: Cyclo3) -> Cyclo3:
        # (a+b*w)(c+d*w)=(ac-bd)+(ad+bc-bd)w because w^2=-1-w.
        ac = self.one * other.one
        bd = self.omega * other.omega
        cross = self.one * other.omega + self.omega * other.one
        return Cyclo3(ac - bd, cross - bd)

    def scaled(self, scalar: int | Fraction) -> Cyclo3:
        factor = Fraction(scalar)
        return Cyclo3(self.one * factor, self.omega * factor)

    def conjugate(self) -> Cyclo3:
        # conjugate(w)=w^2=-1-w.
        return Cyclo3(self.one - self.omega, -self.omega)


ZERO = Cyclo3(Fraction(0), Fraction(0))
ONE = Cyclo3(Fraction(1), Fraction(0))
OMEGA = Cyclo3(Fraction(0), Fraction(1))
OMEGA2 = Cyclo3(Fraction(-1), Fraction(-1))
Matrix = list[list[Cyclo3]]


def root(power: int) -> Cyclo3:
    return (ONE, OMEGA, OMEGA2)[power % D]


def zero_matrix(rows: int, columns: int) -> Matrix:
    return [[ZERO for _ in range(columns)] for _ in range(rows)]


def identity(dimension: int) -> Matrix:
    result = zero_matrix(dimension, dimension)
    for index in range(dimension):
        result[index][index] = ONE
    return result


def matrix_scale(matrix: Matrix, scalar: int | Fraction) -> Matrix:
    return [[value.scaled(scalar) for value in row] for row in matrix]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    result = zero_matrix(len(left), len(right[0]))
    for row in range(len(left)):
        for column in range(len(right[0])):
            total = ZERO
            for inner in range(len(right)):
                total = total + left[row][inner] * right[inner][column]
            result[row][column] = total
    return result


def matrix_dagger(matrix: Matrix) -> Matrix:
    return [
        [matrix[column][row].conjugate() for column in range(len(matrix))]
        for row in range(len(matrix[0]))
    ]


def matrix_trace(matrix: Matrix) -> Cyclo3:
    total = ZERO
    for index in range(len(matrix)):
        total = total + matrix[index][index]
    return total


def tensor(left: Matrix, right: Matrix) -> Matrix:
    result = zero_matrix(len(left) * len(right), len(left[0]) * len(right[0]))
    for lr in range(len(left)):
        for lc in range(len(left[0])):
            for rr in range(len(right)):
                for rc in range(len(right[0])):
                    result[lr * len(right) + rr][lc * len(right[0]) + rc] = (
                        left[lr][lc] * right[rr][rc]
                    )
    return result


def matrix_copy_in_place(destination: Matrix, source: Matrix) -> None:
    if len(destination) != len(source) or len(destination[0]) != len(source[0]):
        raise ValueError("in-place matrix dimensions differ")
    for row in range(len(destination)):
        destination[row][:] = source[row]


def basis_density(dimension: int, index: int) -> Matrix:
    result = zero_matrix(dimension, dimension)
    result[index][index] = ONE
    return result


def client_plus_density() -> Matrix:
    return matrix_scale([[ONE, ONE], [ONE, ONE]], Fraction(1, 2))


def character_numerator(character: int) -> list[Cyclo3]:
    """Return sqrt(3)|chi_character> without adjoining sqrt(3)."""

    return [root(-character * value) for value in range(D)]


def character_density(character: int) -> Matrix:
    numerator = character_numerator(character)
    return [
        [
            (numerator[row] * numerator[column].conjugate()).scaled(
                Fraction(1, D)
            )
            for column in range(D)
        ]
        for row in range(D)
    ]


def shifted_density(matrix: Matrix, amount: int) -> Matrix:
    return [
        [
            matrix[(row - amount) % D][(column - amount) % D]
            for column in range(D)
        ]
        for row in range(D)
    ]


def bell_density(dimension: int) -> Matrix:
    """Exact projector onto sum_j |j,j>/sqrt(dimension)."""

    result = zero_matrix(dimension * dimension, dimension * dimension)
    for left in range(dimension):
        for right in range(dimension):
            result[left * dimension + left][right * dimension + right] = ONE.scaled(
                Fraction(1, dimension)
            )
    return result


def carrier_reference_law() -> Matrix:
    return tensor(character_density(1), bell_density(SPECTATOR_DIMENSION))


def flat_index(coordinates: Sequence[int], dimensions: Sequence[int]) -> int:
    index = 0
    for coordinate, dimension in zip(coordinates, dimensions):
        index = index * dimension + coordinate
    return index


def coordinates(index: int, dimensions: Sequence[int]) -> tuple[int, ...]:
    values = [0 for _ in dimensions]
    for position in range(len(dimensions) - 1, -1, -1):
        values[position] = index % dimensions[position]
        index //= dimensions[position]
    return tuple(values)


def partial_trace(matrix: Matrix, dimensions: Sequence[int], keep: Sequence[int]) -> Matrix:
    keep_tuple = tuple(keep)
    traced = tuple(index for index in range(len(dimensions)) if index not in keep_tuple)
    kept_dimensions = tuple(dimensions[index] for index in keep_tuple)
    output_dimension = 1
    for dimension in kept_dimensions:
        output_dimension *= dimension
    result = zero_matrix(output_dimension, output_dimension)
    total_dimension = len(matrix)
    for row in range(total_dimension):
        row_coordinates = coordinates(row, dimensions)
        output_row = flat_index(
            tuple(row_coordinates[index] for index in keep_tuple), kept_dimensions
        )
        for column in range(total_dimension):
            column_coordinates = coordinates(column, dimensions)
            if any(
                row_coordinates[index] != column_coordinates[index]
                for index in traced
            ):
                continue
            output_column = flat_index(
                tuple(column_coordinates[index] for index in keep_tuple),
                kept_dimensions,
            )
            result[output_row][output_column] = (
                result[output_row][output_column] + matrix[row][column]
            )
    return result


def apply_permutation(matrix: Matrix, permutation: Sequence[int]) -> Matrix:
    if len(matrix) != len(permutation) or len(matrix[0]) != len(permutation):
        raise ValueError("density and permutation dimensions disagree")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("oracle action is not a basis permutation")
    result = zero_matrix(len(matrix), len(matrix))
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            result[permutation[row]][permutation[column]] = matrix[row][column]
    return result


def oracle_permutation(function: Sequence[int], dimensions: Sequence[int]) -> list[int]:
    if len(function) != CLIENT_DIMENSION:
        raise ValueError("oracle table must contain two qutrit residues")
    if tuple(dimensions[:2]) != (CLIENT_DIMENSION, D):
        raise ValueError("oracle dimensions must begin with client,target")
    permutation: list[int] = []
    total = 1
    for dimension in dimensions:
        total *= dimension
    for source in range(total):
        coordinate = list(coordinates(source, dimensions))
        client = coordinate[0]
        coordinate[1] = (coordinate[1] + int(function[client])) % D
        permutation.append(flat_index(coordinate, dimensions))
    return permutation


def apply_client_phase(matrix: Matrix, function: Sequence[int]) -> Matrix:
    return [
        [
            matrix[left][right]
            * root(int(function[left]) - int(function[right]))
            for right in range(CLIENT_DIMENSION)
        ]
        for left in range(CLIENT_DIMENSION)
    ]


def phase_diagonal(function: Sequence[int]) -> Matrix:
    result = zero_matrix(CLIENT_DIMENSION, CLIENT_DIMENSION)
    for index, value in enumerate(function):
        result[index][index] = root(int(value))
    return result


def phase_client_reference_bell(function: Sequence[int]) -> Matrix:
    result = zero_matrix(CLIENT_DIMENSION**2, CLIENT_DIMENSION**2)
    for left in range(CLIENT_DIMENSION):
        for right in range(CLIENT_DIMENSION):
            result[left * CLIENT_DIMENSION + left][
                right * CLIENT_DIMENSION + right
            ] = root(int(function[left]) - int(function[right])).scaled(
                Fraction(1, CLIENT_DIMENSION)
            )
    return result


def client_reference_target_permutation(function: Sequence[int]) -> list[int]:
    dimensions = (CLIENT_DIMENSION, CLIENT_DIMENSION, D)
    permutation: list[int] = []
    for source in range(CLIENT_DIMENSION * CLIENT_DIMENSION * D):
        client, reference, target = coordinates(source, dimensions)
        permutation.append(
            flat_index(
                (client, reference, (target + int(function[client])) % D),
                dimensions,
            )
        )
    return permutation


def proportional_phase_diagonals(
    left: Sequence[int], right: Sequence[int]
) -> tuple[bool, int | None]:
    differences = [
        (int(right[index]) - int(left[index])) % D
        for index in range(CLIENT_DIMENSION)
    ]
    if len(set(differences)) == 1:
        return True, differences[0]
    return False, None


def weighted_sum(matrices: Sequence[Matrix], weights: Sequence[Fraction]) -> Matrix:
    if not matrices or len(matrices) != len(weights):
        raise ValueError("weighted matrix family is empty or mismatched")
    result = zero_matrix(len(matrices[0]), len(matrices[0][0]))
    for matrix, weight in zip(matrices, weights):
        result = matrix_add(result, matrix_scale(matrix, weight))
    return result


def eta(weights: Sequence[Fraction], delta: int) -> Cyclo3:
    total = ZERO
    for character, weight in enumerate(weights):
        total = total + root(character * delta).scaled(weight)
    return total


def eta_client_density(function: Sequence[int], weights: Sequence[Fraction]) -> Matrix:
    return [
        [
            eta(weights, int(function[left]) - int(function[right])).scaled(
                Fraction(1, 2)
            )
            for right in range(CLIENT_DIMENSION)
        ]
        for left in range(CLIENT_DIMENSION)
    ]


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def json_exact(value: object) -> object:
    """Convert exact values and reject every floating-point payload value."""

    if isinstance(value, Cyclo3):
        return {
            "basis": ["1", "omega"],
            "coefficients": [
                fraction_text(value.one),
                fraction_text(value.omega),
            ],
        }
    if isinstance(value, Fraction):
        return fraction_text(value)
    if isinstance(value, dict):
        return {str(key): json_exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_exact(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError("floating-point value rejected from M268 exact payload")
    raise TypeError(f"unsupported exact payload type: {type(value).__name__}")


def exact_bytes(value: object) -> bytes:
    return json.dumps(
        json_exact(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def matrix_digest(matrix: Matrix) -> str:
    return hashlib.sha256(exact_bytes(matrix)).hexdigest()


def coefficient_profile(matrix: Matrix) -> dict[str, int]:
    coefficients = [
        coefficient
        for row in matrix
        for value in row
        for coefficient in (value.one, value.omega)
    ]
    return {
        "qomega_field_cells": len(matrix) * len(matrix[0]),
        "rational_coefficient_slots": len(coefficients),
        "nonzero_qomega_field_cells": sum(value != ZERO for row in matrix for value in row),
        "maximum_absolute_numerator": max(abs(value.numerator) for value in coefficients),
        "maximum_numerator_bits": max(
            max(1, abs(value.numerator).bit_length()) for value in coefficients
        ),
        "maximum_denominator": max(value.denominator for value in coefficients),
        "maximum_denominator_bits": max(
            value.denominator.bit_length() for value in coefficients
        ),
    }


@dataclass
class ResidentBacking:
    """One resident carrier/reference allocation plus one reused joint workspace."""

    allocation_id: str
    carrier_base_id: str
    basis: tuple[str, ...]
    carrier_density: Matrix
    joint_workspace: Matrix
    generation: int = 0
    client_supply_count: int = 0
    coherent_query_count: int = 0
    snapshot_count: int = 0
    reload_count: int = 0
    reseed_count: int = 0
    carrier_swap_count: int = 0
    carrier_reprepare_count: int = 0
    baseline_read_count: int = 0
    retained_history_entries: int = 0

    @classmethod
    def prepared(cls, allocation_id: str) -> ResidentBacking:
        carrier = carrier_reference_law()
        joint_dimension = CLIENT_DIMENSION * len(carrier)
        return cls(
            allocation_id=allocation_id,
            carrier_base_id="M268_QUTRIT_CHARACTER_SENTINEL_BASE_0001",
            basis=("client", "target_qutrit", "spectator_qutrit", "reference_qutrit"),
            carrier_density=carrier,
            joint_workspace=zero_matrix(joint_dimension, joint_dimension),
        )

    def transact(self, function: Sequence[int]) -> tuple[Matrix, dict[str, object]]:
        """Supply a fresh client, execute O_f, certify return, then release it."""

        if tuple(int(value) % D for value in function) != tuple(function):
            raise ValueError("oracle residues must be canonical modulo three")
        self.client_supply_count += 1
        loaded = tensor(client_plus_density(), self.carrier_density)
        matrix_copy_in_place(self.joint_workspace, loaded)
        permutation = oracle_permutation(function, JOINT_DIMS)
        permuted = apply_permutation(self.joint_workspace, permutation)
        matrix_copy_in_place(self.joint_workspace, permuted)
        self.coherent_query_count += 1

        client_boundary = partial_trace(self.joint_workspace, JOINT_DIMS, (0,))
        returned_carrier = partial_trace(self.joint_workspace, JOINT_DIMS, (1, 2, 3))
        algebraic_carrier = carrier_reference_law()
        expected_client = apply_client_phase(client_plus_density(), function)
        expected_joint = tensor(expected_client, algebraic_carrier)
        factorized = self.joint_workspace == expected_joint
        returned = returned_carrier == algebraic_carrier
        if not factorized or not returned:
            raise AssertionError("carrier/reference return failed before boundary release")

        # This copies the actual returned carrier marginal into the same carrier
        # object.  It is a transaction handoff, not a snapshot or reprepare.
        matrix_copy_in_place(self.carrier_density, returned_carrier)
        self.generation += 1
        receipt = {
            "generation": self.generation,
            "fresh_client_supplied": True,
            "coherent_query_count_this_transaction": 1,
            "oracle_is_basis_permutation": sorted(permutation)
            == list(range(len(permutation))),
            "oracle_permutation_basis_rows": len(permutation),
            "density_permutation_qomega_reads": len(permutation) ** 2,
            "density_permutation_qomega_writes": len(permutation) ** 2,
            "exact_factorization_before_release": factorized,
            "target_spectator_reference_return_before_release": returned,
            "boundary_released_after_return": True,
            "matrix_materialization_ledger": {
                "resident_carrier_qomega_cells": 729,
                "reused_joint_workspace_qomega_cells": 2916,
                "loaded_joint_transient_qomega_cells": 2916,
                "permuted_joint_transient_qomega_cells": 2916,
                "client_boundary_qomega_cells": 4,
                "returned_carrier_qomega_cells": 729,
                "fresh_public_algebraic_carrier_verifier_qomega_cells": 729,
                "expected_client_verifier_qomega_cells": 4,
                "expected_joint_verifier_qomega_cells": 2916,
                "listed_concurrent_qomega_cell_floor": 13859,
                "public_algebraic_carrier_verifier_provenance": "FRESH_PUBLIC_FORMULA_REMATERIALIZATION_NOT_SAVED_BASELINE_READ",
                "saved_baseline_matrix_materialized": False,
                "baseline_read_count": self.baseline_read_count,
                "retained_history_entries": self.retained_history_entries,
                "whole_process_peak_qomega_cells_established": False,
                "python_allocator_rss_and_liveness_instrumented": False,
            },
        }
        return client_boundary, receipt


def computational_basis_control(function: Sequence[int]) -> dict[str, object]:
    target_zero = basis_density(D, 0)
    client_plus = client_plus_density()
    input_density = tensor(client_plus, target_zero)
    output = apply_permutation(input_density, oracle_permutation(function, (2, 3)))
    client = partial_trace(output, (2, 3), (0,))
    target = partial_trace(output, (2, 3), (1,))
    product_of_marginals = tensor(client, target)
    client_purity_product = matrix_multiply(client, client)
    joint_purity_product = matrix_multiply(output, output)
    client_purity = matrix_trace(client_purity_product)
    joint_purity = matrix_trace(joint_purity_product)
    return {
        "function_table": list(function),
        "client_purity": client_purity,
        "joint_purity": joint_purity,
        "client_target_entangled": (
            joint_purity == ONE
            and client_purity == ONE.scaled(Fraction(1, 2))
        ),
        "target_returned": target == target_zero,
        "joint_is_product_of_marginals": output == product_of_marginals,
        "matrix_materialization_ledger": {
            "client_plus_density_qomega_cells": 4,
            "target_basis_density_qomega_cells": 9,
            "client_target_input_density_qomega_cells": 36,
            "client_target_output_density_qomega_cells": 36,
            "client_marginal_density_qomega_cells": 4,
            "target_marginal_density_qomega_cells": 9,
            "product_of_marginals_verifier_qomega_cells": 36,
            "client_purity_product_verifier_qomega_cells": 4,
            "joint_purity_product_verifier_qomega_cells": 36,
            "materialized_qomega_cell_events": 174,
        },
    }


def mixed_character_control(function: Sequence[int]) -> dict[str, object]:
    weights = (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
    character_projectors = [
        character_density(character) for character in range(D)
    ]
    mixed = weighted_sum(character_projectors, weights)
    client_plus = client_plus_density()
    initial = tensor(client_plus, mixed)
    output = apply_permutation(initial, oracle_permutation(function, (2, 3)))
    client = partial_trace(output, (2, 3), (0,))
    target = partial_trace(output, (2, 3), (1,))
    predicted_client = eta_client_density(function, weights)
    identity_over_three = matrix_scale(identity(D), Fraction(1, 3))
    fully_dephased_client = matrix_scale(
        identity(CLIENT_DIMENSION), Fraction(1, 2)
    )
    product_of_marginals = tensor(client, target)
    return {
        "eta_law": "ETA_DELTA=SUM_K_P_K_omega^(K*DELTA)",
        "client_law": "RHO_X_XPRIME_MAPS_TO_RHO_X_XPRIME_TIMES_ETA_(F_X-F_XPRIME)",
        "uniform_character_weights": weights,
        "uniform_weights": weights,
        "mixed_carrier": mixed,
        "uniform_character_mixture": mixed,
        "mixed_carrier_equals_identity_over_three": mixed == identity_over_three,
        "uniform_mixture_equals_identity_over_three": mixed == identity_over_three,
        "eta_delta_0_1_2": [eta(weights, delta) for delta in range(D)],
        "uniform_eta_delta_0_1_2": [eta(weights, delta) for delta in range(D)],
        "client_density_after": client,
        "program_a_client_marginal_after_uniform_mixture": client,
        "eta_prediction": predicted_client,
        "program_a_eta_prediction": predicted_client,
        "carrier_marginal_returns": target == mixed,
        "client_is_fully_dephased": client == fully_dephased_client,
        "program_a_client_is_fully_dephased": client == fully_dephased_client,
        "client_carrier_joint_is_correlated": output != product_of_marginals,
        "joint_state_returned": output == initial,
        "joint_state_returns": output == initial,
        "matrix_materialization_ledger": {
            "three_character_projectors_qomega_cells": 27,
            "weighted_sum_constructor_qomega_cell_events_including_mixed_result": 63,
            "client_plus_density_qomega_cells": 4,
            "client_target_input_density_qomega_cells": 36,
            "client_target_output_density_qomega_cells": 36,
            "client_marginal_density_qomega_cells": 4,
            "target_marginal_density_qomega_cells": 9,
            "eta_predicted_client_density_qomega_cells": 4,
            "identity_over_three_constructor_qomega_cell_events": 18,
            "fully_dephased_client_constructor_qomega_cell_events": 8,
            "product_of_marginals_verifier_qomega_cells": 36,
            "materialized_qomega_cell_events": 245,
        },
    }


def branch_environment_control(function: Sequence[int]) -> dict[str, object]:
    """Record x in an environment while the named chi1 carrier still returns."""

    environment_dimension = CLIENT_DIMENSION
    dimensions = (CLIENT_DIMENSION, D, environment_dimension)
    client_plus = client_plus_density()
    named_carrier = character_density(1)
    client_target = tensor(client_plus, named_carrier)
    environment_basis = basis_density(environment_dimension, 0)
    initial = tensor(client_target, environment_basis)
    permutation: list[int] = []
    for source in range(CLIENT_DIMENSION * D * environment_dimension):
        client, target, environment = coordinates(source, dimensions)
        destination = (
            client,
            (target + int(function[client])) % D,
            environment ^ client,
        )
        permutation.append(flat_index(destination, dimensions))
    output = apply_permutation(initial, permutation)
    client = partial_trace(output, dimensions, (0,))
    target = partial_trace(output, dimensions, (1,))
    environment = partial_trace(output, dimensions, (2,))
    identity_two = identity(CLIENT_DIMENSION)
    half_identity_two = matrix_scale(identity_two, Fraction(1, 2))
    return {
        "environment_dimension": environment_dimension,
        "environment_recording_permutation_executed": True,
        "named_character_carrier_marginal_returns": target == named_carrier,
        "client_is_fully_dephased": client == half_identity_two,
        "environment_contains_two_branch_record": environment == half_identity_two,
        "program_a_uniform_record_gram": identity_two,
        "program_a_branch_records_are_orthogonal": environment == half_identity_two,
        "client_dephasing_matches_record_gram": client == half_identity_two,
        "carrier_marginal_return_does_not_erase_environment_record": target
        == named_carrier,
        "reference_complete_joint_return": False,
        "environment_reset_executed": False,
        "matrix_materialization_ledger": {
            "client_plus_density_qomega_cells": 4,
            "named_character_carrier_density_qomega_cells": 9,
            "client_target_intermediate_density_qomega_cells": 36,
            "environment_basis_density_qomega_cells": 4,
            "client_target_environment_input_density_qomega_cells": 144,
            "client_target_environment_output_density_qomega_cells": 144,
            "client_marginal_density_qomega_cells": 4,
            "target_marginal_density_qomega_cells": 9,
            "environment_marginal_density_qomega_cells": 4,
            "identity_two_qomega_cells": 4,
            "half_identity_two_qomega_cells": 4,
            "materialized_qomega_cell_events": 366,
        },
    }


def cross_kerr_fock_compiler_control() -> dict[str, object]:
    """Exact public qutrit cross-Kerr/Fock eigenstate compiler kill."""

    fock_number = 2
    client_plus = client_plus_density()
    fock_basis = basis_density(D, fock_number)
    input_density = tensor(client_plus, fock_basis)
    phases = [
        root(client * target)
        for client in range(CLIENT_DIMENSION)
        for target in range(D)
    ]
    output = [
        [
            input_density[row][column]
            * phases[row]
            * phases[column].conjugate()
            for column in range(CLIENT_DIMENSION * D)
        ]
        for row in range(CLIENT_DIMENSION * D)
    ]
    compiled_function = (0, fock_number)
    expected_client = apply_client_phase(client_plus, compiled_function)
    expected = tensor(expected_client, fock_basis)
    target_marginal = partial_trace(output, (2, 3), (1,))
    client_marginal = partial_trace(output, (2, 3), (0,))
    return {
        "public_law": "U_KERR|X,N>=omega^(X*N)|X,N>",
        "public_fock_number": fock_number,
        "compiled_client_exponents_mod_3": compiled_function,
        "exact_factorization": output == expected,
        "fock_carrier_returns": target_marginal == fock_basis,
        "direct_compiler_matches": client_marginal == expected_client,
        "unique_carrier_advantage": False,
        "matrix_materialization_ledger": {
            "client_plus_density_qomega_cells": 4,
            "fock_basis_density_qomega_cells": 9,
            "client_target_input_density_qomega_cells": 36,
            "client_target_output_density_qomega_cells": 36,
            "expected_client_density_qomega_cells": 4,
            "expected_factorized_density_qomega_cells": 36,
            "target_marginal_density_qomega_cells": 9,
            "client_marginal_density_qomega_cells": 4,
            "materialized_qomega_cell_events": 138,
        },
    }


def reusable_program_theorem() -> dict[str, object]:
    phase_classes = [(0, residue) for residue in range(D)]
    pairwise_nonproportional = all(
        not proportional_phase_diagonals(left, right)[0]
        for position, left in enumerate(phase_classes)
        for right in phase_classes[position + 1 :]
    )
    a_b_proportional, _ = proportional_phase_diagonals(PROGRAM_A, PROGRAM_B)
    return {
        "theorem": "A_FIXED_DETERMINISTIC_EXACT_PROCESSOR_FOR_NONPROPORTIONAL_CLIENT_UNITARIES_REQUIRES_ORTHOGONAL_PROGRAM_STATES",
        "unitarity_inner_product_law": "<P_F|P_G><PSI|PHI>=<P_F_PRIME|P_G_PRIME><PSI|U_F_DAGGER_U_G|PHI>",
        "exact_reuse_specialization": "P_F_PRIME=P_F_AND_P_G_PRIME=P_G",
        "conclusion": "NONZERO_PROGRAM_OVERLAP_IMPLIES_U_F_DAGGER_U_G_IS_A_SCALAR_IDENTITY_SO_NONPROPORTIONAL_UNITARIES_FORCE_ZERO_OVERLAP",
        "program_a_exponents": PROGRAM_A,
        "program_b_exponents": PROGRAM_B,
        "program_a_and_b_are_nonproportional": not a_b_proportional,
        "program_a_and_b_program_states_must_be_orthogonal": not a_b_proportional,
        "program_a_and_b_are_nonproportional_and_witness_orthogonality": not a_b_proportional,
        "theorem_required_program_overlap_if_fixed_exact_processor": ZERO,
        "program_states_materialized": False,
        "program_overlap_executed_or_measured": False,
        "overlap_value_status": "THEOREM_REQUIRED_SYMBOLIC_CONCLUSION_NOT_EXECUTED_OR_MEASURED",
        "minimum_program_dimension_for_a_and_b": 2,
        "qutrit_phase_class_representatives": phase_classes,
        "class_representatives_pairwise_nonproportional": pairwise_nonproportional,
        "minimum_program_dimension_for_all_two_basis_qutrit_phase_classes": D,
        "minimum_program_dimension_for_two_basis_inputs": D,
        "two_client_basis_phase_classes": phase_classes,
        "class_representatives_are_pairwise_nonproportional": pairwise_nonproportional,
        "general_exact_dimension_lower_bound": [
            {
                "client_basis_size": size,
                "nonproportional_phase_classes": D ** (size - 1),
                "minimum_program_hilbert_dimension": D ** (size - 1),
            }
            for size in range(1, 7)
        ],
        "program_preparation_and_certification_are_not_free": True,
    }


def run() -> dict[str, object]:
    source_path = Path(__file__).resolve()
    chi1 = character_density(1)
    bell = bell_density(SPECTATOR_DIMENSION)
    backing = ResidentBacking.prepared("M268_RESIDENT_ALLOCATION_0001")
    carrier_object = backing.carrier_density
    carrier_rows = tuple(backing.carrier_density)
    workspace_object = backing.joint_workspace
    workspace_rows = tuple(backing.joint_workspace)
    basis_object = backing.basis
    initial_generation = backing.generation

    boundary_a, receipt_a = backing.transact(PROGRAM_A)
    generation_after_a = backing.generation
    boundary_b, receipt_b = backing.transact(PROGRAM_B)
    generation_after_b = backing.generation

    carrier_object_stable = backing.carrier_density is carrier_object
    carrier_rows_stable = all(
        current is original
        for current, original in zip(backing.carrier_density, carrier_rows)
    )
    workspace_object_stable = backing.joint_workspace is workspace_object
    workspace_rows_stable = all(
        current is original
        for current, original in zip(backing.joint_workspace, workspace_rows)
    )
    basis_object_stable = backing.basis is basis_object
    algebraic_carrier = carrier_reference_law()
    final_carrier_exact = backing.carrier_density == algebraic_carrier

    direct_a = apply_client_phase(client_plus_density(), PROGRAM_A)
    direct_b = apply_client_phase(client_plus_density(), PROGRAM_B)
    sequential_same_client_function = tuple(
        (PROGRAM_A[index] + PROGRAM_B[index]) % D
        for index in range(CLIENT_DIMENSION)
    )
    sequential_same_client = apply_client_phase(
        apply_client_phase(client_plus_density(), PROGRAM_A), PROGRAM_B
    )
    sequential_initial = tensor(client_plus_density(), chi1)
    sequential_after_a = apply_permutation(
        sequential_initial, oracle_permutation(PROGRAM_A, (2, 3))
    )
    sequential_after_b = apply_permutation(
        sequential_after_a, oracle_permutation(PROGRAM_B, (2, 3))
    )

    client_reference = bell_density(CLIENT_DIMENSION)
    client_reference_target = tensor(client_reference, chi1)
    client_reference_after_a = apply_permutation(
        client_reference_target,
        client_reference_target_permutation(PROGRAM_A),
    )
    phased_client_reference = phase_client_reference_bell(PROGRAM_A)
    client_reference_expected = tensor(phased_client_reference, chi1)
    client_reference_marginal = partial_trace(
        phased_client_reference,
        (CLIENT_DIMENSION, CLIENT_DIMENSION),
        (1,),
    )

    comparator = ResidentBacking.prepared("M268_EQUAL_ACCESS_COMPARATOR_0001")
    comparator_a, comparator_receipt_a = comparator.transact(PROGRAM_A)
    comparator_b, comparator_receipt_b = comparator.transact(PROGRAM_B)

    basis_controls = {
        "program_a": computational_basis_control(PROGRAM_A),
        "program_b": computational_basis_control(PROGRAM_B),
    }
    mixed_control = mixed_character_control(PROGRAM_A)
    environment_control = branch_environment_control(PROGRAM_A)
    cross_kerr = cross_kerr_fock_compiler_control()
    program_theorem = reusable_program_theorem()

    joint_dimension = 1
    for dimension in JOINT_DIMS:
        joint_dimension *= dimension
    carrier_dimension = 1
    for dimension in CARRIER_DIMS:
        carrier_dimension *= dimension
    per_query_density_cells = joint_dimension * joint_dimension
    descriptor_payload = {"A": PROGRAM_A, "B": PROGRAM_B}
    final_profile = coefficient_profile(backing.carrier_density)

    fixtures: dict[str, object] = {
        "cyclotomic_authority": {
            "dimension": D,
            "field": "Q(omega)",
            "minimal_polynomial": "omega^2+omega+1=0",
            "omega": OMEGA,
            "omega_squared": OMEGA2,
            "omega_cubed": root(3),
            "polynomial_residual": OMEGA * OMEGA + OMEGA + ONE,
            "floating_point_values_on_decision_path": 0,
        },
        "secret_independent_character_carrier": {
            "character_label": 1,
            "secret_independent": True,
            "state_law": "|CHI_1>=SUM_Y_omega^(-Y)|Y>/SQRT_3",
            "density_entry_law": "RHO_YZ=omega^(Z-Y)/3",
            "unnormalized_sqrt3_times_state": character_numerator(1),
            "projector": chi1,
            "trace": matrix_trace(chi1),
            "purity": matrix_trace(matrix_multiply(chi1, chi1)),
            "hermitian": matrix_dagger(chi1) == chi1,
            "shift_eigenvalues_0_1_2": [root(amount) for amount in range(D)],
            "all_shifted_numerators_match_eigenvalue_law": all(
                [
                    character_numerator(1)[(index - amount) % D]
                    for index in range(D)
                ]
                == [root(amount) * value for value in character_numerator(1)]
                for amount in range(D)
            ),
            "density_is_invariant_under_every_target_shift": all(
                shifted_density(chi1, amount) == chi1 for amount in range(D)
            ),
        },
        "program_a": {
            "name": "A",
            "function_table": PROGRAM_A,
            "compiled_client_diagonal": phase_diagonal(PROGRAM_A),
            "boundary_density": boundary_a,
            "boundary_matches_direct_compile": boundary_a == direct_a,
            "exact_kickback_factorization": receipt_a[
                "exact_factorization_before_release"
            ],
            "exact_character_carrier_marginal_return": receipt_a[
                "target_spectator_reference_return_before_release"
            ],
            "receipt": receipt_a,
        },
        "program_b": {
            "name": "B",
            "function_table": PROGRAM_B,
            "compiled_client_diagonal": phase_diagonal(PROGRAM_B),
            "boundary_density": boundary_b,
            "boundary_matches_direct_compile": boundary_b == direct_b,
            "exact_kickback_factorization": receipt_b[
                "exact_factorization_before_release"
            ],
            "exact_character_carrier_marginal_return": receipt_b[
                "target_spectator_reference_return_before_release"
            ],
            "receipt": receipt_b,
        },
        "two_fresh_client_same_logical_carrier_transactions": {
            "query_order": ["A", "B"],
            "function_tables": [PROGRAM_A, PROGRAM_B],
            "distinct_oracle_permutations": oracle_permutation(PROGRAM_A, JOINT_DIMS)
            != oracle_permutation(PROGRAM_B, JOINT_DIMS),
            "fresh_client_transactions": 2,
            "transaction_model": "TWO_FRESH_CLIENT_TRANSACTIONS_ON_ONE_RESIDENT_TARGET_SPECTATOR_REFERENCE_CARRIER_ALLOCATION",
            "client_supply_count": backing.client_supply_count,
            "coherent_oracle_query_count": backing.coherent_query_count,
            "phase_diagonals_are_nonproportional": not proportional_phase_diagonals(
                PROGRAM_A, PROGRAM_B
            )[0],
            "fresh_client_boundaries_are_distinct": boundary_a != boundary_b,
            "program_a_fresh_client_boundary": boundary_a,
            "program_b_fresh_client_boundary": boundary_b,
            "combined_fresh_client_boundary_density": tensor(
                boundary_a, boundary_b
            ),
            "program_a_fresh_transaction_exact_factorization": receipt_a[
                "exact_factorization_before_release"
            ],
            "combined_fresh_client_boundary_exact_factorization": receipt_a[
                "exact_factorization_before_release"
            ]
            and receipt_b["exact_factorization_before_release"],
            "carrier_returns_after_each_query": receipt_a[
                "target_spectator_reference_return_before_release"
            ]
            and receipt_b["target_spectator_reference_return_before_release"],
            "generation_sequence": [
                initial_generation,
                generation_after_a,
                generation_after_b,
            ],
            "allocation_id": backing.allocation_id,
            "carrier_base_id": backing.carrier_base_id,
            "carrier_object_stable": carrier_object_stable,
            "carrier_row_objects_stable": carrier_rows_stable,
            "joint_workspace_object_stable": workspace_object_stable,
            "joint_workspace_row_objects_stable": workspace_rows_stable,
            "basis_object_stable": basis_object_stable,
            "target_spectator_reference_returns_after_a": receipt_a[
                "target_spectator_reference_return_before_release"
            ],
            "target_spectator_reference_returns_after_b": receipt_b[
                "target_spectator_reference_return_before_release"
            ],
            "final_carrier_reference_density_digest": matrix_digest(
                backing.carrier_density
            ),
            "algebraic_carrier_reference_density_digest": matrix_digest(
                algebraic_carrier
            ),
            "snapshot_count": backing.snapshot_count,
            "reload_count": backing.reload_count,
            "reseed_count": backing.reseed_count,
            "carrier_swap_count": backing.carrier_swap_count,
            "carrier_reprepare_count": backing.carrier_reprepare_count,
            "baseline_read_count": backing.baseline_read_count,
            "retained_history_entries": backing.retained_history_entries,
            "reset_controls": {
                "snapshot_reload": {
                    "classification": "REJECTED_HISTORY_BASED_RESET_NOT_RESTORATION",
                    "executed_on_accepted_path": False,
                },
                "fresh_carrier_swap": {
                    "classification": "REJECTED_EXTERNAL_REPLACEMENT_NOT_REUSE",
                    "executed_on_accepted_path": False,
                },
                "carrier_reprepare": {
                    "classification": "REJECTED_NEW_PREPARATION_NOT_RETURN",
                    "executed_on_accepted_path": False,
                },
            },
            "only_declared_client_boundaries_released": True,
            "carrier_or_intermediate_joint_projection_released": False,
            "same_logical_carrier_semantics": True,
            "executed_same_backing_claim": True,
        },
        "nonproportional_program_control": {
            "program_a_exponents": PROGRAM_A,
            "program_b_exponents": PROGRAM_B,
            "a_and_b_phase_diagonals_are_nonproportional": not proportional_phase_diagonals(
                PROGRAM_A, PROGRAM_B
            )[0],
            "a_and_b_fresh_client_density_channels_are_distinct": boundary_a
            != boundary_b,
            "oracles_remain_distinct_permutations": oracle_permutation(
                PROGRAM_A, JOINT_DIMS
            )
            != oracle_permutation(PROGRAM_B, JOINT_DIMS),
            "interpretation": "A_AND_B_INDUCE_DISTINCT_NONPROPORTIONAL_CLIENT_PHASE_GATES_AND_DISTINCT_FRESH_CLIENT_BOUNDARIES",
        },
        "same_client_sequential_composition_diagnostic": {
            "is_reuse_boundary": False,
            "same_client_supply_count": 1,
            "sequential_phase_exponents_mod_3": sequential_same_client_function,
            "sequential_client_diagonal": phase_diagonal(
                sequential_same_client_function
            ),
            "sequential_client_boundary_density": sequential_same_client,
            "joint_density_after_a_then_b_on_same_client": sequential_after_b,
            "sequential_exact_factorization": sequential_after_b
            == tensor(sequential_same_client, chi1),
            "sequential_phase_is_identity": phase_diagonal(
                sequential_same_client_function
            )
            == identity(CLIENT_DIMENSION),
            "result_free_global_identity_only": sequential_same_client
            == client_plus_density(),
            "interpretation": "A_THEN_B_ON_ONE_CLIENT_CANCELS_TO_IDENTITY_AND_IS_ONLY_A_RESULT_FREE_DIAGNOSTIC_NOT_THE_TWO_FRESH_CLIENT_REUSE_BOUNDARY",
        },
        "spectator_reference_bell_preservation": {
            "spectator_dimension": SPECTATOR_DIMENSION,
            "reference_dimension": REFERENCE_DIMENSION,
            "bell_projector": bell,
            "bell_density_before": bell,
            "bell_density_after_both_queries": bell,
            "exactly_unchanged": True,
            "bell_trace": matrix_trace(bell),
            "bell_purity": matrix_trace(matrix_multiply(bell, bell)),
            "spectator_marginal": partial_trace(
                bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (0,)
            ),
            "reference_marginal": partial_trace(
                bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (1,)
            ),
            "marginals_equal_identity_over_three": (
                partial_trace(
                    bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (0,)
                )
                == matrix_scale(identity(D), Fraction(1, D))
                and partial_trace(
                    bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (1,)
                )
                == matrix_scale(identity(D), Fraction(1, D))
            ),
            "spectator_reference_marginal_after_both_queries": partial_trace(
                bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (0,)
            ),
            "spectator_reference_marginal_is_maximally_mixed": partial_trace(
                bell, (SPECTATOR_DIMENSION, REFERENCE_DIMENSION), (0,)
            )
            == matrix_scale(identity(D), Fraction(1, D)),
            "client_reference_bell_density_before": client_reference,
            "client_reference_bell_density_after_program_a": phased_client_reference,
            "client_reference_target_after_program_a": client_reference_after_a,
            "client_reference_target_exact_factorization": client_reference_after_a
            == client_reference_expected,
            "client_reference_marginal_after_program_a": client_reference_marginal,
            "client_reference_marginal_is_maximally_mixed": client_reference_marginal
            == matrix_scale(identity(CLIENT_DIMENSION), Fraction(1, 2)),
            "client_reference_purity_after_program_a": matrix_trace(
                matrix_multiply(phased_client_reference, phased_client_reference)
            ),
            "bell_entanglement_preserved_by_local_kickback_unitary": True,
            "inert_under_both_oracle_permutations": True,
            "exact_carrier_reference_return_after_both_transactions": final_carrier_exact,
        },
        "computational_basis_entanglement_controls": basis_controls,
        "mixed_character_marginal_return_and_eta_dephasing": mixed_control,
        "branch_record_environment": environment_control,
        "public_law_direct_compiler": {
            "function_descriptors_are_public_in_this_fixture": True,
            "function_descriptors_are_public": True,
            "compiler_law": "D_F=DIAG_X_omega^F_X",
            "program_a_exact": boundary_a == direct_a,
            "program_b_exact": boundary_b == direct_b,
            "program_a_compiled_exponents": PROGRAM_A,
            "program_b_compiled_exponents": PROGRAM_B,
            "combined_fresh_client_compiled_diagonal": tensor(
                phase_diagonal(PROGRAM_A), phase_diagonal(PROGRAM_B)
            ),
            "direct_program_a_boundary_exact": boundary_a == direct_a,
            "direct_program_b_boundary_exact": boundary_b == direct_b,
            "direct_combined_boundary_exact": tensor(boundary_a, boundary_b)
            == tensor(direct_a, direct_b),
            "coherent_oracle_queries": 0,
            "direct_compiler_coherent_oracle_queries": 0,
            "descriptor_residue_reads": len(PROGRAM_A) + len(PROGRAM_B),
            "descriptor_reads_for_a_and_b": len(PROGRAM_A) + len(PROGRAM_B),
            "compiled_diagonal_entries": len(PROGRAM_A) + len(PROGRAM_B),
            "compiled_diagonal_entries_for_a_and_b": len(PROGRAM_A)
            + len(PROGRAM_B),
            "compilation_and_descriptor_costs_are_charged": True,
            "carrier_dimension_retained": 0,
            "restoration_stage_executed": False,
            "cross_kerr_fock_compiler_kill": cross_kerr,
        },
        "exact_reusable_program_orthogonality": program_theorem,
        "equal_coherent_oracle_access_collapse": {
            "phase_route_query_sequence": ["O_A", "O_B"],
            "equal_access_comparator_query_sequence": ["O_A", "O_B"],
            "query_sequences_identical": True,
            "phase_route_coherent_queries": backing.coherent_query_count,
            "equal_access_comparator_coherent_queries": comparator.coherent_query_count,
            "program_a_boundary_identical": comparator_a == boundary_a,
            "program_b_boundary_identical": comparator_b == boundary_b,
            "exact_boundaries_identical": comparator_a == boundary_a
            and comparator_b == boundary_b,
            "carrier_reference_final_identical": comparator.carrier_density
            == backing.carrier_density,
            "comparator_receipts": [comparator_receipt_a, comparator_receipt_b],
            "unique_phase_qemu_query_advantage": False,
            "total_resource_advantage": False,
        },
        "m241_m242_linear_calibration_negative": {
            "m241_hidden_linear_phase_calibration_improved": False,
            "m242_tensor_factored_linear_route_improved": False,
            "m242_tensor_factored_hidden_linear_phase_calibration_improved": False,
            "linear_secret_query_lower_bound_changed": False,
            "reason": "FIXED_D3_TWO_POINT_ADDITION_FIXTURES_DO_NOT_IMPROVE_THE_P5_HIDDEN_LINEAR_QUERY_LAW_OR_ESTABLISH_A_SCALING_SEPARATION",
            "m241_escape": False,
            "m242_escape": False,
        },
        "forrelation_oracle_cost_caveat": {
            "prospective_only": True,
            "implemented_here": False,
            "query_separation_claimed": False,
            "forrelation_query_separation_claimed": False,
            "required_change": "RESTRICTED_PROMISE_BLACK_BOX_INTERFACE_WITH_PREDECLARED_EQUAL_COMPARATOR_ACCESS",
            "oracle_generation_cost_must_be_charged": True,
            "oracle_custody_cost_must_be_charged": True,
            "carrier_preparation_and_precision_must_be_charged": True,
            "carrier_preparation_and_precision_costs_must_be_charged": True,
            "total_resource_accounting_required": True,
            "public_descriptor_access_restores_direct_compilation": True,
            "public_descriptor_access_would_restore_direct_compilation": True,
        },
    }

    resource_ledger: dict[str, object] = {
        "dimensions": {
            "client": CLIENT_DIMENSION,
            "target_character_qutrit": D,
            "spectator_qutrit": SPECTATOR_DIMENSION,
            "inert_reference_qutrit": REFERENCE_DIMENSION,
            "resident_target_spectator_reference_hilbert_dimension": carrier_dimension,
            "joint_transaction_hilbert_dimension": joint_dimension,
            "resident_carrier_density_qomega_cells": carrier_dimension**2,
            "reused_joint_density_workspace_qomega_cells": joint_dimension**2,
        },
        "accepted_transact_matrix_materializations": {
            "transaction_count": backing.coherent_query_count,
            "per_transaction": receipt_a["matrix_materialization_ledger"],
            "program_a_and_program_b_ledgers_identical": receipt_a[
                "matrix_materialization_ledger"
            ]
            == receipt_b["matrix_materialization_ledger"],
            "persistent_resident_qomega_cells_allocated_once": 729 + 2916,
            "transient_and_verifier_qomega_cell_materializations_per_transaction": 10214,
            "transient_and_verifier_qomega_cell_materializations_for_two_transactions": 20428,
            "primary_persistent_plus_two_transaction_materialization_events": 24073,
            "listed_concurrent_qomega_cell_floor_per_transaction": 13859,
            "listed_concurrent_floor_is_total_process_peak": False,
            "whole_process_peak_qomega_cells": "NOT_ESTABLISHED_FAIL_CLOSED",
            "public_algebraic_verifier_is_saved_baseline": False,
            "public_algebraic_verifier_is_fresh_formula_rematerialization": True,
            "baseline_read_count": backing.baseline_read_count,
            "retained_history_entries": backing.retained_history_entries,
        },
        "directly_materialized_comparator_and_control_matrices": {
            "top_level_public_fixture_objects_qomega_cells": {
                "chi1_projector": 9,
                "spectator_reference_bell_projector": 81,
                "post_transaction_public_algebraic_carrier": 729,
            },
            "public_direct_phase_comparator_qomega_cells": {
                "program_a_client_plus_input": 4,
                "program_a_client_boundary": 4,
                "program_b_client_plus_input": 4,
                "program_b_client_boundary": 4,
            },
            "equal_coherent_oracle_comparator": {
                "resident_carrier_qomega_cells": 729,
                "reused_joint_workspace_qomega_cells": 2916,
                "program_a_client_boundary_qomega_cells": 4,
                "program_b_client_boundary_qomega_cells": 4,
                "transaction_count": comparator.coherent_query_count,
                "per_transaction_matrix_materializations": comparator_receipt_a[
                    "matrix_materialization_ledger"
                ],
                "program_a_and_program_b_ledgers_identical": comparator_receipt_a[
                    "matrix_materialization_ledger"
                ]
                == comparator_receipt_b["matrix_materialization_ledger"],
            },
            "same_client_sequential_diagnostic_qomega_cells": {
                "compiled_client_density": 4,
                "initial_client_target_density": 36,
                "after_program_a_density": 36,
                "after_program_b_density": 36,
            },
            "client_reference_completeness_control_qomega_cells": {
                "client_reference_bell_density": 16,
                "client_reference_target_input_density": 144,
                "client_reference_target_output_density": 144,
                "phased_client_reference_density": 16,
                "expected_factorized_density": 144,
                "reference_marginal_density": 4,
            },
            "computational_basis_controls": {
                "fixture_count": 2,
                "per_fixture_matrix_materializations": basis_controls[
                    "program_a"
                ]["matrix_materialization_ledger"],
                "program_a_and_program_b_ledgers_identical": basis_controls[
                    "program_a"
                ]["matrix_materialization_ledger"]
                == basis_controls["program_b"]["matrix_materialization_ledger"],
            },
            "mixed_character_control": {
                "matrix_materializations": mixed_control[
                    "matrix_materialization_ledger"
                ],
            },
            "branch_record_environment_control": {
                "matrix_materializations": environment_control[
                    "matrix_materialization_ledger"
                ],
            },
            "cross_kerr_fock_compiler_control": {
                "matrix_materializations": cross_kerr[
                    "matrix_materialization_ledger"
                ],
            },
            "matrix_lists_are_logical_object_counts_not_allocator_or_rss_measurements": True,
            "whole_process_matrix_liveness_instrumented": False,
            "whole_process_peak_claimed": False,
        },
        "qomega_storage_and_precision": {
            "field_basis": ["1", "omega"],
            "rational_coefficients_per_qomega_cell": 2,
            "resident_carrier_profile_after_two_queries": final_profile,
            "joint_workspace_profile_after_program_b": coefficient_profile(
                backing.joint_workspace
            ),
            "joint_workspace_qomega_cells": per_query_density_cells,
            "joint_workspace_rational_coefficient_slots": 2
            * per_query_density_cells,
            "exact_fraction_and_cyclotomic_equality_only": True,
            "floating_point_decision_operations": 0,
            "arbitrary_precision_integer_cost_is_charged": True,
        },
        "descriptor_and_oracle_internal_size": {
            "program_count": 2,
            "function_table_entries": len(PROGRAM_A) + len(PROGRAM_B),
            "logical_bits_per_qutrit_residue": 2,
            "logical_function_table_bits": 2
            * (len(PROGRAM_A) + len(PROGRAM_B)),
            "compact_descriptor_json_bytes": len(exact_bytes(descriptor_payload)),
            "oracle_permutation_rows_per_query": joint_dimension,
            "oracle_permutation_index_bits": (joint_dimension - 1).bit_length(),
            "materialized_permutation_workspace_bits": joint_dimension
            * (joint_dimension - 1).bit_length(),
            "oracle_generation_is_stipulated_not_physically_accounted": True,
        },
        "preparation_and_certification": {
            "client_plus_density_preparations": backing.client_supply_count,
            "client_plus_density_qomega_writes_each": CLIENT_DIMENSION**2,
            "character_projector_qomega_writes": D**2,
            "spectator_reference_bell_density_qomega_writes": (
                SPECTATOR_DIMENSION * REFERENCE_DIMENSION
            ) ** 2,
            "resident_carrier_density_qomega_writes": carrier_dimension**2,
            "joint_workspace_load_qomega_writes": backing.client_supply_count
            * per_query_density_cells,
            "exact_trace_purity_hermiticity_shift_and_factorization_certification_executed": True,
            "physical_preparation_energy": "UNINSTANTIATED_NOT_FREE",
            "physical_certification_device": "UNINSTANTIATED_NOT_FREE",
        },
        "query_action_and_bandwidth": {
            "coherent_query_count": backing.coherent_query_count,
            "basis_permutation_rows_total": backing.coherent_query_count
            * joint_dimension,
            "density_qomega_reads_total": backing.coherent_query_count
            * per_query_density_cells,
            "density_qomega_writes_total": backing.coherent_query_count
            * per_query_density_cells,
            "density_qomega_cell_transfers_total": 2
            * backing.coherent_query_count
            * per_query_density_cells,
            "oracle_table_residue_reads_total": len(PROGRAM_A) + len(PROGRAM_B),
            "logical_query_rounds": 2,
            "physical_bandwidth_hz": "UNINSTANTIATED_NOT_FREE",
            "wall_clock_latency_evidence": "NONE",
            "physical_latency_seconds": "UNINSTANTIATED_NOT_FREE",
        },
        "environment_controller_and_history": {
            "accepted_active_environment_dimension": 1,
            "accepted_spectator_dimension": SPECTATOR_DIMENSION,
            "accepted_inert_reference_dimension": REFERENCE_DIMENSION,
            "branch_record_control_environment_dimension": CLIENT_DIMENSION,
            "generation_sequence": [0, 1, 2],
            "client_supply_count": backing.client_supply_count,
            "generation_counter_peak_bits": backing.generation.bit_length(),
            "query_counter_peak_bits": backing.coherent_query_count.bit_length(),
            "retained_dynamic_history_entries": backing.retained_history_entries,
            "snapshot_reload_reseed_swap_reprepare_baseline_reads": 0,
            "controller_ids_utf8_bytes": len(backing.allocation_id.encode("utf-8"))
            + len(backing.carrier_base_id.encode("utf-8")),
            "environment_reset_executed": False,
        },
        "public_direct_phase_comparator": {
            "coherent_oracle_queries": 0,
            "descriptor_residue_reads": len(PROGRAM_A) + len(PROGRAM_B),
            "compiled_diagonal_qomega_entries": len(PROGRAM_A) + len(PROGRAM_B),
            "client_density_phase_multiplications": 2 * CLIENT_DIMENSION**2,
            "carrier_qomega_cells": 0,
            "carrier_preparation": 0,
            "restoration_actions": 0,
            "boundaries_exactly_equal": boundary_a == direct_a
            and boundary_b == direct_b,
        },
        "equal_coherent_oracle_comparator": {
            "coherent_queries": comparator.coherent_query_count,
            "client_supply_count": comparator.client_supply_count,
            "joint_density_qomega_reads": comparator.coherent_query_count
            * per_query_density_cells,
            "joint_density_qomega_writes": comparator.coherent_query_count
            * per_query_density_cells,
            "same_preparation_certification_and_restoration_categories_charged": True,
            "outputs_identical": comparator_a == boundary_a
            and comparator_b == boundary_b,
        },
        "software_total_accounting_boundary": {
            "source_bytes": len(source_path.read_bytes()),
            "source_hash_operations_charged": True,
            "json_serialization_and_hashing_charged": True,
            "python_objects_allocator_interpreter_and_process_costs": "UNINSTRUMENTED_NONZERO_FAIL_CLOSED",
            "whole_process_rss": "UNINSTRUMENTED_NONZERO_FAIL_CLOSED",
            "matrix_object_liveness": "UNINSTRUMENTED_NONZERO_FAIL_CLOSED",
            "whole_process_peak_logical_or_physical_memory": "NOT_ESTABLISHED_FAIL_CLOSED",
            "scheduler_and_storage_io": "NONZERO_NOT_MEASURED",
            "unmeasured_costs_treated_as_zero": False,
            "2916_cell_workspace_is_total_peak": False,
            "total_resource_advantage_claimed": False,
        },
    }

    public_boundary = {
        "release_policy": "EACH_CLIENT_BOUNDARY_RELEASED_ONLY_AFTER_EXACT_TARGET_SPECTATOR_REFERENCE_RETURN_AND_FACTORIZATION",
        "release_count": 2,
        "generation_at_release_a": receipt_a["generation"],
        "generation_at_release_b": receipt_b["generation"],
        "program_a_client_density": boundary_a,
        "program_b_client_density": boundary_b,
        "program_a_boundary_sha256": matrix_digest(boundary_a),
        "program_b_boundary_sha256": matrix_digest(boundary_b),
        "carrier_density_released": False,
        "joint_workspace_released": False,
        "oracle_branch_history_released": False,
    }

    negative_claims = {
        "qemu_device_implementation": False,
        "common_guest_visible_device_contract": False,
        "physical_oracle": False,
        "physical_oracle_generation": False,
        "physical_carrier_custody": False,
        "physical_character_state_preparation": False,
        "physical_restoration": False,
        "external_access_enforcement": False,
        "query_separation": False,
        "total_resource_advantage": False,
        "unique_phase_qemu_advantage": False,
        "m241_escape": False,
        "m242_escape": False,
        "m257_escape": False,
        "forrelation_implemented": False,
        "small_wall_crossing": False,
        "unbounded_computation": False,
        "bit_replaced_with_pi": False,
    }

    architecture_scope = {
        "phase_qemu_layer_classification": "MECHANISM_SEARCH_DIGITAL_TWIN_OUTSIDE_QEMU_DEVICE",
        "qemu_device_implemented": False,
        "common_guest_visible_device_contract_exercised": False,
        "eligible_for_mechanism_kill": True,
        "eligible_for_architecture_promotion": False,
        "promotion_requires_common_phase_qemu_device_or_backend": True,
        "production_executes_bounded_logical_restoration": True,
        "production_can_promote_architecture": False,
        "scope_statement": "EXACT_MECHANISM_AND_BOUNDED_LOGICAL_RESTORATION_EVIDENCE_OUTSIDE_THE_COMMON_QEMU_DEVICE_CANNOT_PROMOTE_PHASE_QEMU_ARCHITECTURE",
    }

    transactions = fixtures["two_fresh_client_same_logical_carrier_transactions"]
    sentinel = fixtures["spectator_reference_bell_preservation"]
    checks = {
        "cyclotomic_polynomial_is_exact": OMEGA * OMEGA + OMEGA + ONE == ZERO,
        "chi1_trace_one": matrix_trace(chi1) == ONE,
        "chi1_pure": matrix_trace(matrix_multiply(chi1, chi1)) == ONE,
        "chi1_hermitian": matrix_dagger(chi1) == chi1,
        "chi1_shift_eigenvector_law": fixtures[
            "secret_independent_character_carrier"
        ]["all_shifted_numerators_match_eigenvalue_law"],
        "chi1_density_shift_invariant": fixtures[
            "secret_independent_character_carrier"
        ]["density_is_invariant_under_every_target_shift"],
        "program_a_exact_kickback": receipt_a["oracle_is_basis_permutation"]
        and fixtures["program_a"]["exact_kickback_factorization"],
        "program_b_exact_kickback": receipt_b["oracle_is_basis_permutation"]
        and fixtures["program_b"]["exact_kickback_factorization"],
        "program_a_carrier_return": fixtures["program_a"][
            "exact_character_carrier_marginal_return"
        ],
        "program_b_carrier_return": fixtures["program_b"][
            "exact_character_carrier_marginal_return"
        ],
        "two_oracles_are_distinct": transactions["distinct_oracle_permutations"],
        "program_a_program_b_phase_gates_are_nonproportional": fixtures[
            "nonproportional_program_control"
        ]["a_and_b_phase_diagonals_are_nonproportional"],
        "two_fresh_client_boundaries_are_distinct": transactions[
            "fresh_client_boundaries_are_distinct"
        ],
        "two_fresh_client_supply_is_explicit": transactions["client_supply_count"]
        == 2
        and transactions["generation_sequence"] == [0, 1, 2]
        and transactions["carrier_object_stable"]
        and transactions["carrier_row_objects_stable"]
        and transactions["joint_workspace_object_stable"]
        and transactions["joint_workspace_row_objects_stable"]
        and transactions["basis_object_stable"]
        and receipt_a["matrix_materialization_ledger"][
            "listed_concurrent_qomega_cell_floor"
        ]
        == 13859
        and receipt_b["matrix_materialization_ledger"]
        == receipt_a["matrix_materialization_ledger"]
        and not receipt_a["matrix_materialization_ledger"][
            "whole_process_peak_qomega_cells_established"
        ]
        and not receipt_a["matrix_materialization_ledger"][
            "saved_baseline_matrix_materialized"
        ]
        and all(
            count == 0
            for count in (
                backing.snapshot_count,
                backing.reload_count,
                backing.reseed_count,
                backing.carrier_swap_count,
                backing.carrier_reprepare_count,
                backing.baseline_read_count,
                backing.retained_history_entries,
            )
        ),
        "first_fresh_client_transaction_exact": transactions[
            "program_a_fresh_transaction_exact_factorization"
        ],
        "two_fresh_client_combined_boundary_exact": transactions[
            "combined_fresh_client_boundary_exact_factorization"
        ]
        and tensor(boundary_a, boundary_b) == tensor(direct_a, direct_b),
        "two_fresh_client_carrier_return": transactions[
            "carrier_returns_after_each_query"
        ]
        and final_carrier_exact,
        "same_client_sequential_identity_is_diagnostic_only": fixtures[
            "same_client_sequential_composition_diagnostic"
        ]["sequential_exact_factorization"]
        and fixtures["same_client_sequential_composition_diagnostic"][
            "sequential_phase_is_identity"
        ]
        and fixtures["same_client_sequential_composition_diagnostic"][
            "result_free_global_identity_only"
        ]
        and not fixtures["same_client_sequential_composition_diagnostic"][
            "is_reuse_boundary"
        ],
        "inert_qutrit_spectator_reference_sentinel_exact": sentinel[
            "exactly_unchanged"
        ]
        and sentinel["spectator_reference_marginal_is_maximally_mixed"]
        and sentinel["exact_carrier_reference_return_after_both_transactions"],
        "client_reference_kickback_factorization": sentinel[
            "client_reference_target_exact_factorization"
        ],
        "client_reference_completeness_test_preserved": sentinel[
            "client_reference_purity_after_program_a"
        ]
        == ONE
        and sentinel["client_reference_marginal_is_maximally_mixed"],
        "inert_sentinel_and_client_reference_tests_are_dimensionally_distinct": len(
            sentinel["spectator_reference_marginal_after_both_queries"]
        )
        == D
        and len(sentinel["client_reference_marginal_after_program_a"])
        == CLIENT_DIMENSION,
        "basis_a_entangles_and_does_not_return": basis_controls["program_a"][
            "client_target_entangled"
        ]
        and not basis_controls["program_a"]["target_returned"],
        "basis_b_entangles_and_does_not_return": basis_controls["program_b"][
            "client_target_entangled"
        ]
        and not basis_controls["program_b"]["target_returned"],
        "uniform_character_eta_is_delta": mixed_control["uniform_eta_delta_0_1_2"]
        == [ONE, ZERO, ZERO],
        "mixed_character_carrier_marginal_returns": mixed_control[
            "carrier_marginal_returns"
        ],
        "mixed_character_client_dephases": mixed_control[
            "program_a_client_is_fully_dephased"
        ],
        "mixed_character_joint_does_not_return": not mixed_control[
            "joint_state_returns"
        ]
        and mixed_control["client_carrier_joint_is_correlated"],
        "branch_record_is_orthogonal": environment_control[
            "program_a_branch_records_are_orthogonal"
        ]
        and environment_control["named_character_carrier_marginal_returns"]
        and not environment_control["reference_complete_joint_return"],
        "public_direct_compiler_matches_all_boundaries": fixtures[
            "public_law_direct_compiler"
        ]["direct_program_a_boundary_exact"]
        and fixtures["public_law_direct_compiler"]["direct_program_b_boundary_exact"]
        and fixtures["public_law_direct_compiler"]["direct_combined_boundary_exact"]
        and cross_kerr["direct_compiler_matches"],
        "nielsen_chuang_classes_exact": program_theorem[
            "program_a_and_b_are_nonproportional_and_witness_orthogonality"
        ]
        and program_theorem["class_representatives_are_pairwise_nonproportional"]
        and program_theorem["minimum_program_dimension_for_two_basis_inputs"]
        == D
        and program_theorem[
            "theorem_required_program_overlap_if_fixed_exact_processor"
        ]
        == ZERO
        and not program_theorem["program_states_materialized"]
        and not program_theorem["program_overlap_executed_or_measured"],
        "equal_access_runs_identical_queries": fixtures[
            "equal_coherent_oracle_access_collapse"
        ]["query_sequences_identical"]
        and fixtures["equal_coherent_oracle_access_collapse"][
            "exact_boundaries_identical"
        ]
        and comparator.carrier_density == backing.carrier_density,
        "m241_m242_calibration_is_negative": not fixtures[
            "m241_m242_linear_calibration_negative"
        ]["m241_hidden_linear_phase_calibration_improved"]
        and not fixtures["m241_m242_linear_calibration_negative"][
            "m242_tensor_factored_hidden_linear_phase_calibration_improved"
        ],
        "forrelation_is_prospective_only": fixtures[
            "forrelation_oracle_cost_caveat"
        ]["prospective_only"]
        and not fixtures["forrelation_oracle_cost_caveat"]["implemented_here"]
        and not fixtures["forrelation_oracle_cost_caveat"][
            "forrelation_query_separation_claimed"
        ]
        and not negative_claims["m257_escape"]
        and not negative_claims["physical_oracle"]
        and not negative_claims["total_resource_advantage"]
        and public_boundary["release_count"] == 2
        and not public_boundary["carrier_density_released"]
        and not public_boundary["joint_workspace_released"],
        "architecture_scope_fails_closed_outside_qemu_device": architecture_scope[
            "phase_qemu_layer_classification"
        ]
        == "MECHANISM_SEARCH_DIGITAL_TWIN_OUTSIDE_QEMU_DEVICE"
        and not architecture_scope["qemu_device_implemented"]
        and not architecture_scope[
            "common_guest_visible_device_contract_exercised"
        ]
        and architecture_scope["eligible_for_mechanism_kill"]
        and not architecture_scope["eligible_for_architecture_promotion"]
        and architecture_scope[
            "promotion_requires_common_phase_qemu_device_or_backend"
        ]
        and architecture_scope["production_executes_bounded_logical_restoration"]
        and not architecture_scope["production_can_promote_architecture"],
    }

    claim_payload = {
        "milestone": MILESTONE,
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
    }
    result: dict[str, object] = {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "status": "PASS_INTERNAL_EXACT_QUTRIT_KICKBACK_SELF_CHECK",
        "terminal": False,
        "claim": CLAIM,
        "claim_ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "claim_payload": claim_payload,
        "mathematical_conventions": {
            "target_shift": "X_A|Y>=|Y+A_MOD_3>",
            "character_state": "|CHI_K>=SUM_Y_omega^(-K*Y)|Y>/SQRT_3",
            "kickback": "X_A|CHI_K>=omega^(K*A)|CHI_K>",
            "addition_oracle": "O_F|X,Y>=|X,Y+F_X_MOD_3>",
            "client_phase": "D_F=DIAG_X_omega^F_X",
            "eta_law": "ETA_DELTA=TR(RHO_TARGET*X^DELTA)=SUM_K_P_K_omega^(K*DELTA)",
            "density_field": "Q(omega)_WITH_RATIONAL_NORMALIZATION",
        },
        "theorem": {
            "accepted_process": "TWO_FRESH_CLIENT_TRANSACTIONS_ON_ONE_RESIDENT_TARGET_SPECTATOR_REFERENCE_CARRIER_ALLOCATION",
            "carrier_return": "O_F(RHO_CLIENT_TENSOR_P_CHI1)O_F_DAGGER=D_F_RHO_CLIENT_D_F_DAGGER_TENSOR_P_CHI1",
            "sentinel_return": "IDENTITY_SPECTATOR_REFERENCE_ACTS_ON_ONE_QUTRIT_BELL_PROJECTOR_AND_RETURNS_IT_EXACTLY",
            "public_descriptor_case": "READ_F_AND_APPLY_D_F_DIRECTLY_WITHOUT_CARRIER",
            "secret_program_case": program_theorem["conclusion"],
            "equal_oracle_access_case": "RUN_THE_IDENTICAL_O_F_QUERY_SO_NO_UNIQUE_PHASE_QEMU_QUERY_RESOURCE_REMAINS",
        },
        "fixtures": fixtures,
        "public_boundary": public_boundary,
        "strongest_honest_comparators": {
            "public_descriptor": fixtures["public_law_direct_compiler"],
            "secret_reusable_program": program_theorem,
            "equal_coherent_oracle_access": fixtures[
                "equal_coherent_oracle_access_collapse"
            ],
        },
        "architecture_scope": architecture_scope,
        "architecture_authority": {
            "phase_qemu_layer_classification": "MECHANISM_SEARCH_DIGITAL_TWIN_OUTSIDE_QEMU_DEVICE",
            "qemu_device_implemented": False,
            "common_guest_visible_device_contract_exercised": False,
            "eligible_for_mechanism_kill": True,
            "eligible_for_architecture_promotion": False,
            "promotion_requires_common_phase_qemu_device_or_backend": True,
            "bounded_exact_logical_restoration_evidence_preserved": True,
            "mechanism_proof_is_device_integration": False,
            "survivor_integration_gate": "RETURN_TO_COMMON_PHASE_QEMU_GUEST_DEVICE_OR_BACKEND_AND_DEMONSTRATE_LIFECYCLE_CUSTODY_BOUNDARY_ORDERING_RESTORATION_REUSE_AND_SNAPSHOT_SHAM_LINEAGE_BEFORE_MACHINE_ARCHITECTURE_PROMOTION",
            "successor_if_physical_oracle_survives": SUCCESSOR,
            "successor_requires_integration_gate": True,
        },
        "resource_ledger": resource_ledger,
        "scope_exclusions": {
            "physical_oracle_generation": True,
            "physical_carrier_or_reference": True,
            "physical_preparation_certification_energy_bandwidth_and_latency": True,
            "enforced_secret_oracle_custody": True,
            "growing_problem_family": True,
            "forrelation_or_other_promise_problem": True,
            "complexity_lower_bound": True,
        },
        "negative_claims": negative_claims,
        "m257": {
            "guardrail": "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE",
            "public_descriptor_direct_compiler_present": True,
            "equal_coherent_oracle_comparator_runs_identical_queries": True,
            "guardrail_remains_intact": True,
            "escape_established": False,
        },
        "source_self_assertion": "PASS_INTERNAL_CONSISTENCY_ONLY",
        "checks": checks,
    }

    # Fail before emission.  Exact conversion rejects floats anywhere in the
    # claim-bearing structure, including resource and control records.
    exact_bytes(result)
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise AssertionError(f"M268 internal self-check failure: {failed}")
    result["claim_payload_sha256"] = hashlib.sha256(
        exact_bytes(claim_payload)
    ).hexdigest()
    result["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    exact_bytes(result)
    return result


def main() -> int:
    print(exact_bytes(run()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
