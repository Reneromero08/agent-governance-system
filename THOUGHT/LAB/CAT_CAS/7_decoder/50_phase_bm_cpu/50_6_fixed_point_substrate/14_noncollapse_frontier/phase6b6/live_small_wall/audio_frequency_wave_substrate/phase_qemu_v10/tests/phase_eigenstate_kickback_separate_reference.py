#!/usr/bin/env python3
"""Independent exact reference for the M268 qutrit kickback boundary.

All state decisions are made in Q(omega), with omega**2 + omega + 1 = 0.
Square-root amplitudes are represented by their exact density matrices or by
unnormalized character numerators.  This reference accepts no input, imports
no package code, executes no restoration, and asserts no physical custody.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


REFERENCE_ID = "M268_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE_V1"
SCHEMA = "PHASE_QEMU_V10_PHASE_EIGENSTATE_KICKBACK_SEPARATE_REFERENCE_V1"
CLAIM = "FINITE_QUDIT_PHASE_EIGENSTATE_KICKBACK_RETURNS_A_SECRET_INDEPENDENT_CHARACTER_CARRIER_EXACTLY_FOR_TWO_DISTINCT_COHERENT_ORACLE_QUERIES_WHILE_PUBLIC_LAWS_ADMIT_DIRECT_PHASE_COMPILATION_SECRET_DEPENDENT_REUSABLE_PROGRAM_STATES_REQUIRE_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_ORACLE_ACCESS_ERASES_ANY_UNIQUE_PHASE_QEMU_ADVANTAGE"
CEILING = "DETERMINISTIC_EXACT_FINITE_DIMENSIONAL_SOFTWARE_ORACLE_DIGITAL_TWIN_WITH_STIPULATED_EXTERNAL_COHERENT_QUERY_INTERFACE_NO_PHYSICAL_ORACLE_CARRIER_CUSTODY_QUERY_SEPARATION_OR_TOTAL_RESOURCE_ADVANTAGE"
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
REFERENCE_RESTORATION_SCOPE = "FORMAL_EXACT_CHARACTER_DENSITY_AND_SPECTATOR_REFERENCE_IDENTITIES_ONLY_WITHOUT_EXECUTED_SAME_BACKING_OR_PHYSICAL_RESTORATION"
EXPECTED_PRODUCTION_RESTORATION_CLASSIFICATION = "EXACT_ALGEBRAIC_RESTORATION"
EXPECTED_PRODUCTION_RESTORATION_SCOPE = "EXACT_CYCLOTOMIC_LOGICAL_CARRIER_AND_INERT_REFERENCE_RETURN_FOR_TWO_DISTINCT_STIPULATED_COHERENT_ORACLE_QUERIES_ON_ONE_RESIDENT_SOFTWARE_ALLOCATION_WITHOUT_PHYSICAL_ORACLE_OR_CARRIER_CUSTODY"
DISPOSITION = "EXACT_KICKBACK_AND_LOGICAL_CARRIER_REUSE_ARE_VALID_BUT_PUBLIC_DESCRIPTORS_COMPILE_DIRECTLY_SECRET_DEPENDENT_REUSABLE_PROGRAM_STATES_PAY_ORTHOGONAL_DIMENSION_AND_EQUAL_COHERENT_ORACLE_ACCESS_RUNS_THE_IDENTICAL_QUERY_SO_NO_UNIQUE_PHASE_RESOURCE_TOTAL_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
SUCCESSOR = "COMPACT_PHYSICAL_COHERENT_ORACLE_GENERATION_LAW_WITH_SECRET_INDEPENDENT_FINITE_ENERGY_EIGENSTATE_CARRIER_AND_EQUAL_INTERFACE_TOTAL_RESOURCE_ACCOUNTING"

D = 3
CLIENT_DIMENSION = 2
PROGRAM_A = (0, 1)
PROGRAM_B = (0, 2)


@dataclass(frozen=True)
class Cyclo3:
    """a + b*omega in Q(omega), represented without approximation."""

    one: Fraction
    omega: Fraction

    def __add__(self, other: Cyclo3) -> Cyclo3:
        return Cyclo3(self.one + other.one, self.omega + other.omega)

    def __neg__(self) -> Cyclo3:
        return Cyclo3(-self.one, -self.omega)

    def __sub__(self, other: Cyclo3) -> Cyclo3:
        return self + (-other)

    def __mul__(self, other: Cyclo3) -> Cyclo3:
        # omega**2 = -1 - omega.
        ac = self.one * other.one
        bd = self.omega * other.omega
        cross = self.one * other.omega + self.omega * other.one
        return Cyclo3(ac - bd, cross - bd)

    def scaled(self, scalar: Fraction | int) -> Cyclo3:
        factor = Fraction(scalar)
        return Cyclo3(factor * self.one, factor * self.omega)

    def conjugate(self) -> Cyclo3:
        # conjugate(omega) = omega**2 = -1 - omega.
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


def matrix_trace(matrix: Matrix) -> Cyclo3:
    total = ZERO
    for index in range(len(matrix)):
        total = total + matrix[index][index]
    return total


def matrix_dagger(matrix: Matrix) -> Matrix:
    return [
        [matrix[column][row].conjugate() for column in range(len(matrix))]
        for row in range(len(matrix[0]))
    ]


def matrix_scale(matrix: Matrix, scalar: Fraction | int) -> Matrix:
    return [[value.scaled(scalar) for value in row] for row in matrix]


def tensor(left: Matrix, right: Matrix) -> Matrix:
    result = zero_matrix(len(left) * len(right), len(left[0]) * len(right[0]))
    for left_row in range(len(left)):
        for left_column in range(len(left[0])):
            for right_row in range(len(right)):
                for right_column in range(len(right[0])):
                    row = left_row * len(right) + right_row
                    column = left_column * len(right[0]) + right_column
                    result[row][column] = (
                        left[left_row][left_column] * right[right_row][right_column]
                    )
    return result


def basis_density(dimension: int, index: int) -> Matrix:
    result = zero_matrix(dimension, dimension)
    result[index][index] = ONE
    return result


def character_numerator(character: int) -> list[Cyclo3]:
    """sqrt(3)|chi_k> as exact cyclotomic entries."""
    return [root(-character * value) for value in range(D)]


def character_density(character: int) -> Matrix:
    numerator = character_numerator(character)
    return [
        [
            numerator[row] * numerator[column].conjugate().scaled(Fraction(1, D))
            for column in range(D)
        ]
        for row in range(D)
    ]


def shifted_vector(vector: Sequence[Cyclo3], amount: int) -> list[Cyclo3]:
    return [vector[(index - amount) % D] for index in range(D)]


def shifted_density(matrix: Matrix, amount: int) -> Matrix:
    return [
        [matrix[(row - amount) % D][(column - amount) % D] for column in range(D)]
        for row in range(D)
    ]


def client_plus_density() -> Matrix:
    return matrix_scale([[ONE, ONE], [ONE, ONE]], Fraction(1, 2))


def oracle_permutation(function: Sequence[int]) -> list[int]:
    if len(function) != CLIENT_DIMENSION:
        raise ValueError("oracle function must have exactly two client entries")
    return [
        client * D + ((target + int(function[client])) % D)
        for client in range(CLIENT_DIMENSION)
        for target in range(D)
    ]


def apply_permutation(matrix: Matrix, permutation: Sequence[int]) -> Matrix:
    if len(matrix) != len(permutation) or len(matrix[0]) != len(permutation):
        raise ValueError("permutation and density dimensions disagree")
    result = zero_matrix(len(matrix), len(matrix))
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            result[permutation[row]][permutation[column]] = matrix[row][column]
    return result


def phase_diagonal(function: Sequence[int], character: int = 1) -> Matrix:
    result = zero_matrix(CLIENT_DIMENSION, CLIENT_DIMENSION)
    for client, value in enumerate(function):
        result[client][client] = root(character * int(value))
    return result


def phase_client_density(function: Sequence[int], character: int = 1) -> Matrix:
    diagonal = phase_diagonal(function, character)
    return matrix_multiply(
        matrix_multiply(diagonal, client_plus_density()), matrix_dagger(diagonal)
    )


def trace_out_target(joint: Matrix) -> Matrix:
    result = zero_matrix(CLIENT_DIMENSION, CLIENT_DIMENSION)
    for left in range(CLIENT_DIMENSION):
        for right in range(CLIENT_DIMENSION):
            total = ZERO
            for target in range(D):
                total = total + joint[left * D + target][right * D + target]
            result[left][right] = total
    return result


def trace_out_client(joint: Matrix) -> Matrix:
    result = zero_matrix(D, D)
    for left in range(D):
        for right in range(D):
            total = ZERO
            for client in range(CLIENT_DIMENSION):
                total = total + joint[client * D + left][client * D + right]
            result[left][right] = total
    return result


def trace_out_final_target(joint: Matrix, leading_dimension: int) -> Matrix:
    result = zero_matrix(leading_dimension, leading_dimension)
    for left in range(leading_dimension):
        for right in range(leading_dimension):
            total = ZERO
            for target in range(D):
                total = total + joint[left * D + target][right * D + target]
            result[left][right] = total
    return result


def trace_out_leading_system(joint: Matrix, leading_dimension: int) -> Matrix:
    result = zero_matrix(D, D)
    for left in range(D):
        for right in range(D):
            total = ZERO
            for leading in range(leading_dimension):
                total = total + joint[leading * D + left][leading * D + right]
            result[left][right] = total
    return result


def two_client_target_permutation(
    function: Sequence[int], client_wire: int
) -> list[int]:
    if client_wire not in (0, 1):
        raise ValueError("client wire must be zero or one")
    permutation: list[int] = []
    for client_a in range(CLIENT_DIMENSION):
        for client_b in range(CLIENT_DIMENSION):
            selected = client_a if client_wire == 0 else client_b
            for target in range(D):
                destination = (
                    (client_a * CLIENT_DIMENSION + client_b) * D
                    + (target + int(function[selected])) % D
                )
                permutation.append(destination)
    return permutation


def run_character_query(function: Sequence[int], target: Matrix) -> Matrix:
    initial = tensor(client_plus_density(), target)
    return apply_permutation(initial, oracle_permutation(function))


def bell_density(dimension: int) -> Matrix:
    """Density of sum_j |j,j>/sqrt(dimension)."""
    result = zero_matrix(dimension * dimension, dimension * dimension)
    weight = Fraction(1, dimension)
    for left in range(dimension):
        for right in range(dimension):
            result[left * dimension + left][right * dimension + right] = ONE.scaled(
                weight
            )
    return result


def phase_bell_density(function: Sequence[int]) -> Matrix:
    result = zero_matrix(4, 4)
    for left in range(CLIENT_DIMENSION):
        for right in range(CLIENT_DIMENSION):
            result[left * 2 + left][right * 2 + right] = root(
                int(function[left]) - int(function[right])
            ).scaled(Fraction(1, 2))
    return result


def client_reference_target_permutation(function: Sequence[int]) -> list[int]:
    permutation: list[int] = []
    for client in range(CLIENT_DIMENSION):
        for reference in range(CLIENT_DIMENSION):
            for target in range(D):
                destination = (
                    client * CLIENT_DIMENSION * D
                    + reference * D
                    + (target + int(function[client])) % D
                )
                permutation.append(destination)
    return permutation


def trace_first_equal_subsystem(matrix: Matrix, dimension: int) -> Matrix:
    """Trace the first half of a dimension-by-dimension bipartite state."""
    if len(matrix) != dimension * dimension or len(matrix[0]) != dimension * dimension:
        raise ValueError("bipartite density dimension mismatch")
    result = zero_matrix(dimension, dimension)
    for left in range(dimension):
        for right in range(dimension):
            total = ZERO
            for first in range(dimension):
                total = total + matrix[
                    first * dimension + left
                ][first * dimension + right]
            result[left][right] = total
    return result


def weighted_sum(matrices: Sequence[Matrix], weights: Sequence[Fraction]) -> Matrix:
    if len(matrices) != len(weights):
        raise ValueError("matrix and weight counts disagree")
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
    result = zero_matrix(CLIENT_DIMENSION, CLIENT_DIMENSION)
    for left in range(CLIENT_DIMENSION):
        for right in range(CLIENT_DIMENSION):
            result[left][right] = eta(
                weights, int(function[left]) - int(function[right])
            ).scaled(Fraction(1, 2))
    return result


def branch_record_gram(function: Sequence[int], weights: Sequence[Fraction]) -> Matrix:
    return [
        [
            eta(weights, int(function[left]) - int(function[right]))
            for right in range(CLIENT_DIMENSION)
        ]
        for left in range(CLIENT_DIMENSION)
    ]


def diagonal_exponents_proportional(
    left: Sequence[int], right: Sequence[int]
) -> tuple[bool, int | None]:
    differences = [
        (int(right[index]) - int(left[index])) % D
        for index in range(CLIENT_DIMENSION)
    ]
    if len(set(differences)) == 1:
        return True, differences[0]
    return False, None


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def json_exact(value: object) -> object:
    """Fail closed on any non-exact or unsupported payload value."""
    if isinstance(value, Cyclo3):
        return {
            "basis": ["1", "omega"],
            "coefficients": [fraction_text(value.one), fraction_text(value.omega)],
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
        raise TypeError("floating-point value rejected from exact reference payload")
    raise TypeError(f"unsupported payload type: {type(value).__name__}")


def exact_bytes(value: object) -> bytes:
    return json.dumps(
        json_exact(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def program_fixture(name: str, function: Sequence[int]) -> dict[str, object]:
    target = character_density(1)
    output = run_character_query(function, target)
    client = phase_client_density(function)
    expected = tensor(client, target)
    return {
        "name": name,
        "function_table": list(function),
        "coherent_oracle_query_count": 1,
        "oracle_permutation": oracle_permutation(function),
        "compiled_phase_exponents_mod_3": [int(value) % D for value in function],
        "compiled_client_diagonal": phase_diagonal(function),
        "client_boundary_density": client,
        "target_density_after": trace_out_client(output),
        "joint_density_after": output,
        "exact_kickback_factorization": output == expected,
        "exact_character_carrier_marginal_return": trace_out_client(output) == target,
    }


def computational_basis_control(function: Sequence[int]) -> dict[str, object]:
    initial_target = basis_density(D, 0)
    output = run_character_query(function, initial_target)
    client = trace_out_target(output)
    target = trace_out_client(output)
    distinct_images = len({int(value) % D for value in function}) == CLIENT_DIMENSION
    return {
        "function_table": list(function),
        "initial_target_density": initial_target,
        "joint_density_after": output,
        "client_marginal_after": client,
        "target_marginal_after": target,
        "joint_purity": matrix_trace(matrix_multiply(output, output)),
        "client_purity": matrix_trace(matrix_multiply(client, client)),
        "distinct_target_images": distinct_images,
        "schmidt_rank": 2 if distinct_images else 1,
        "client_target_entangled": bool(
            distinct_images
            and matrix_trace(matrix_multiply(output, output)) == ONE
            and matrix_trace(matrix_multiply(client, client)) == ONE.scaled(
                Fraction(1, 2)
            )
        ),
        "target_returned": target == initial_target,
    }


def main() -> int:
    chi1_numerator = character_numerator(1)
    chi1 = character_density(1)
    program_a = program_fixture("A", PROGRAM_A)
    program_b = program_fixture("B", PROGRAM_B)

    sequential_initial = tensor(client_plus_density(), chi1)
    sequential_after_a = apply_permutation(
        sequential_initial, oracle_permutation(PROGRAM_A)
    )
    sequential_after_a_then_b = apply_permutation(
        sequential_after_a, oracle_permutation(PROGRAM_B)
    )
    sequential_function = tuple(
        (PROGRAM_A[index] + PROGRAM_B[index]) % D
        for index in range(CLIENT_DIMENSION)
    )
    sequential_client = phase_client_density(sequential_function)

    fresh_clients_initial = tensor(
        tensor(client_plus_density(), client_plus_density()), chi1
    )
    fresh_clients_after_a = apply_permutation(
        fresh_clients_initial, two_client_target_permutation(PROGRAM_A, 0)
    )
    fresh_clients_after_a_then_b = apply_permutation(
        fresh_clients_after_a, two_client_target_permutation(PROGRAM_B, 1)
    )
    combined_fresh_client_boundary = tensor(
        phase_client_density(PROGRAM_A), phase_client_density(PROGRAM_B)
    )

    spectator_reference = bell_density(D)
    spectator_reference_after_both = bell_density(D)
    spectator_reference_marginal_after_both = trace_first_equal_subsystem(
        spectator_reference_after_both, D
    )
    client_reference = bell_density(CLIENT_DIMENSION)
    client_reference_target = tensor(client_reference, chi1)
    client_reference_after_a = apply_permutation(
        client_reference_target,
        client_reference_target_permutation(PROGRAM_A),
    )
    phased_client_reference = phase_bell_density(PROGRAM_A)

    uniform_weights = (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))
    nonuniform_weights = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
    uniform_character_mixture = weighted_sum(
        [character_density(character) for character in range(D)], uniform_weights
    )
    mixed_output_a = run_character_query(PROGRAM_A, uniform_character_mixture)
    uniform_gram_a = branch_record_gram(PROGRAM_A, uniform_weights)

    a_b_proportional, _ = diagonal_exponents_proportional(
        PROGRAM_A, PROGRAM_B
    )
    class_representatives = [(0, residue) for residue in range(D)]
    pairwise_nonproportional = all(
        not diagonal_exponents_proportional(left, right)[0]
        for index, left in enumerate(class_representatives)
        for right in class_representatives[index + 1 :]
    )

    fixtures = {
        "cyclotomic_authority": {
            "dimension": D,
            "field": "Q(omega)",
            "minimal_polynomial": "omega^2+omega+1=0",
            "omega": OMEGA,
            "omega_squared": OMEGA2,
            "omega_cubed": root(3),
            "one_plus_omega_plus_omega_squared": ONE + OMEGA + OMEGA2,
            "all_decision_paths_are_fraction_and_cyclotomic_equality": True,
        },
        "secret_independent_character_carrier": {
            "character_label": 1,
            "secret_independent": True,
            "unnormalized_numerator_sqrt3_times_chi1": chi1_numerator,
            "density_matrix": chi1,
            "density_entry_law": "RHO_YZ=omega^(Z-Y)/3",
            "trace": matrix_trace(chi1),
            "purity": matrix_trace(matrix_multiply(chi1, chi1)),
            "hermitian": matrix_dagger(chi1) == chi1,
            "shift_eigenvalues": [root(amount) for amount in range(D)],
            "all_shifted_numerators_match_eigenvalue_law": all(
                shifted_vector(chi1_numerator, amount)
                == [root(amount) * value for value in chi1_numerator]
                for amount in range(D)
            ),
            "density_is_invariant_under_every_target_shift": all(
                shifted_density(chi1, amount) == chi1 for amount in range(D)
            ),
        },
        "program_a": program_a,
        "program_b": program_b,
        "two_fresh_client_same_logical_carrier_transactions": {
            "query_order": ["A", "B"],
            "client_supply_count": 2,
            "transaction_model": "TWO_FRESH_CLIENT_TRANSACTIONS_ON_ONE_LOGICAL_CHARACTER_CARRIER",
            "distinct_oracle_permutations": (
                oracle_permutation(PROGRAM_A) != oracle_permutation(PROGRAM_B)
            ),
            "phase_diagonals_are_nonproportional": not a_b_proportional,
            "fresh_client_boundaries_are_distinct": (
                phase_client_density(PROGRAM_A) != phase_client_density(PROGRAM_B)
            ),
            "coherent_oracle_query_count": 2,
            "program_a_fresh_client_boundary": phase_client_density(PROGRAM_A),
            "program_b_fresh_client_boundary": phase_client_density(PROGRAM_B),
            "combined_fresh_client_boundary_density": (
                combined_fresh_client_boundary
            ),
            "joint_density_after_both_fresh_client_transactions": (
                fresh_clients_after_a_then_b
            ),
            "program_a_fresh_transaction_exact_factorization": (
                fresh_clients_after_a
                == tensor(
                    tensor(
                        phase_client_density(PROGRAM_A), client_plus_density()
                    ),
                    chi1,
                )
            ),
            "combined_fresh_client_boundary_exact_factorization": (
                fresh_clients_after_a_then_b
                == tensor(combined_fresh_client_boundary, chi1)
            ),
            "carrier_density_after_program_a": trace_out_leading_system(
                fresh_clients_after_a, CLIENT_DIMENSION**2
            ),
            "carrier_density_after_program_b": trace_out_leading_system(
                fresh_clients_after_a_then_b, CLIENT_DIMENSION**2
            ),
            "carrier_returns_after_each_query": bool(
                trace_out_leading_system(
                    fresh_clients_after_a, CLIENT_DIMENSION**2
                )
                == chi1
                and trace_out_leading_system(
                    fresh_clients_after_a_then_b, CLIENT_DIMENSION**2
                )
                == chi1
            ),
            "same_logical_carrier_semantics": True,
            "executed_same_backing_claim": False,
        },
        "nonproportional_program_control": {
            "program_a_exponents": PROGRAM_A,
            "program_b_exponents": PROGRAM_B,
            "a_and_b_phase_diagonals_are_nonproportional": not a_b_proportional,
            "a_and_b_fresh_client_density_channels_are_distinct": (
                phase_client_density(PROGRAM_A) != phase_client_density(PROGRAM_B)
            ),
            "oracles_remain_distinct_permutations": (
                oracle_permutation(PROGRAM_A) != oracle_permutation(PROGRAM_B)
            ),
            "interpretation": "A_AND_B_INDUCE_DISTINCT_NONPROPORTIONAL_CLIENT_PHASE_GATES_AND_THEREFORE_DISTINCT_FRESH_CLIENT_BOUNDARIES",
        },
        "same_client_sequential_composition_diagnostic": {
            "is_reuse_boundary": False,
            "same_client_supply_count": 1,
            "sequential_phase_exponents_mod_3": sequential_function,
            "sequential_client_diagonal": phase_diagonal(sequential_function),
            "sequential_client_boundary_density": sequential_client,
            "joint_density_after_a_then_b_on_same_client": (
                sequential_after_a_then_b
            ),
            "sequential_exact_factorization": (
                sequential_after_a_then_b == tensor(sequential_client, chi1)
            ),
            "sequential_phase_is_identity": (
                phase_diagonal(sequential_function) == identity(CLIENT_DIMENSION)
            ),
            "result_free_global_identity_only": (
                sequential_client == client_plus_density()
            ),
            "interpretation": "A_THEN_B_ON_ONE_CLIENT_CANCELS_TO_IDENTITY_AND_IS_ONLY_A_RESULT_FREE_DIAGNOSTIC_NOT_THE_TWO_FRESH_CLIENT_REUSE_BOUNDARY",
        },
        "spectator_reference_bell_preservation": {
            "spectator_dimension": D,
            "reference_dimension": D,
            "bell_density_before": spectator_reference,
            "bell_density_after_both_queries": spectator_reference_after_both,
            "exactly_unchanged": (
                spectator_reference_after_both == spectator_reference
            ),
            "spectator_reference_marginal_after_both_queries": (
                spectator_reference_marginal_after_both
            ),
            "spectator_reference_marginal_is_maximally_mixed": (
                spectator_reference_marginal_after_both
                == matrix_scale(identity(D), Fraction(1, D))
            ),
            "client_reference_bell_density_before": client_reference,
            "client_reference_bell_density_after_program_a": phased_client_reference,
            "client_reference_target_after_program_a": client_reference_after_a,
            "client_reference_target_exact_factorization": (
                client_reference_after_a == tensor(phased_client_reference, chi1)
            ),
            "client_reference_marginal_after_program_a": (
                trace_first_equal_subsystem(
                    phased_client_reference, CLIENT_DIMENSION
                )
            ),
            "client_reference_marginal_is_maximally_mixed": (
                trace_first_equal_subsystem(
                    phased_client_reference, CLIENT_DIMENSION
                )
                == matrix_scale(identity(CLIENT_DIMENSION), Fraction(1, 2))
            ),
            "client_reference_purity_after_program_a": matrix_trace(
                matrix_multiply(phased_client_reference, phased_client_reference)
            ),
            "bell_entanglement_preserved_by_local_kickback_unitary": True,
        },
        "computational_basis_entanglement_controls": {
            "program_a": computational_basis_control(PROGRAM_A),
            "program_b": computational_basis_control(PROGRAM_B),
            "control_law": "A_COMPUTATIONAL_BASIS_TARGET_RECORDS_DISTINCT_F_X_VALUES_SO_CLIENT_COHERENCE_IS_LOST_AND_TARGET_RETURN_FAILS",
        },
        "mixed_character_marginal_return_and_eta_dephasing": {
            "eta_law": "ETA_DELTA=SUM_K_P_K_omega^(K*DELTA)",
            "client_law": "RHO_X_XPRIME_MAPS_TO_RHO_X_XPRIME_TIMES_ETA_(F_X-F_XPRIME)",
            "uniform_weights": uniform_weights,
            "uniform_character_mixture": uniform_character_mixture,
            "uniform_mixture_equals_identity_over_three": (
                uniform_character_mixture == matrix_scale(identity(D), Fraction(1, 3))
            ),
            "uniform_eta_delta_0_1_2": [
                eta(uniform_weights, delta) for delta in range(D)
            ],
            "program_a_client_marginal_after_uniform_mixture": trace_out_target(
                mixed_output_a
            ),
            "program_a_eta_prediction": eta_client_density(
                PROGRAM_A, uniform_weights
            ),
            "program_a_client_is_fully_dephased": (
                trace_out_target(mixed_output_a)
                == matrix_scale(identity(CLIENT_DIMENSION), Fraction(1, 2))
            ),
            "carrier_marginal_returns": (
                trace_out_client(mixed_output_a) == uniform_character_mixture
            ),
            "joint_state_returns": (
                mixed_output_a
                == tensor(client_plus_density(), uniform_character_mixture)
            ),
            "nonuniform_weights": nonuniform_weights,
            "nonuniform_eta_delta_0_1_2": [
                eta(nonuniform_weights, delta) for delta in range(D)
            ],
        },
        "branch_record_environment": {
            "record_construction": "E_X_HAS_CHARACTER_COMPONENT_K_EQUAL_TO_SQRT_P_K_TIMES_omega^(K*F_X)",
            "record_overlap_law": "INNER_E_XPRIME_E_X=ETA_(F_X-F_XPRIME)",
            "program_a_uniform_record_gram": uniform_gram_a,
            "program_a_branch_records_are_orthogonal": (
                uniform_gram_a == identity(CLIENT_DIMENSION)
            ),
            "client_dephasing_matches_record_gram": (
                trace_out_target(mixed_output_a)
                == eta_client_density(PROGRAM_A, uniform_weights)
            ),
            "carrier_marginal_return_does_not_erase_environment_record": True,
            "reference_complete_joint_return": False,
        },
        "public_law_direct_compiler": {
            "function_descriptors_are_public": True,
            "compiler_law": "D_F=DIAG_X_omega^F_X",
            "program_a_compiled_exponents": PROGRAM_A,
            "program_b_compiled_exponents": PROGRAM_B,
            "combined_fresh_client_compiled_diagonal": tensor(
                phase_diagonal(PROGRAM_A), phase_diagonal(PROGRAM_B)
            ),
            "direct_compiler_coherent_oracle_queries": 0,
            "descriptor_reads_for_a_and_b": len(PROGRAM_A) + len(PROGRAM_B),
            "compiled_diagonal_entries_for_a_and_b": (
                len(PROGRAM_A) + len(PROGRAM_B)
            ),
            "direct_program_a_boundary_exact": (
                phase_client_density(PROGRAM_A)
                == trace_out_target(run_character_query(PROGRAM_A, chi1))
            ),
            "direct_program_b_boundary_exact": (
                phase_client_density(PROGRAM_B)
                == trace_out_target(run_character_query(PROGRAM_B, chi1))
            ),
            "direct_combined_boundary_exact": (
                combined_fresh_client_boundary
                == trace_out_final_target(
                    fresh_clients_after_a_then_b, CLIENT_DIMENSION**2
                )
            ),
            "compilation_and_descriptor_costs_are_charged": True,
        },
        "exact_reusable_program_orthogonality": {
            "theorem": "A_FIXED_EXACT_PROCESSOR_FOR_NONPROPORTIONAL_UNITARIES_REQUIRES_ORTHOGONAL_PROGRAM_STATES",
            "inner_product_identity": "INNER_P_F_P_G=omega^(G_X-F_X)*INNER_PPRIME_F_PPRIME_G_FOR_EVERY_CLIENT_BASIS_X",
            "proof": "IF_G_X-F_X_IS_NOT_CONSTANT_MODULO_THREE_TWO_BASIS_CHOICES_GIVE_DISTINCT_ROOTS_OF_UNITY_SO_THE_PROGRAM_OVERLAPS_MUST_BE_ZERO",
            "exact_reuse_specialization": "PPRIME_F=P_F_PRESERVES_THE_SAME_ORTHOGONALITY_CONCLUSION",
            "unitaries_identified_up_to_global_phase": True,
            "program_a_and_b_are_nonproportional_and_witness_orthogonality": (
                not a_b_proportional
            ),
            "theorem_required_program_overlap_if_fixed_exact_processor": ZERO,
            "zero_overlap_is_hypothetical_no_programming_theorem_requirement": True,
            "program_states_materialized": False,
            "program_overlap_executed_or_measured": False,
            "two_client_basis_phase_classes": class_representatives,
            "class_representatives_are_pairwise_nonproportional": pairwise_nonproportional,
            "minimum_program_dimension_for_two_basis_inputs": D,
            "general_qutrit_phase_class_count_and_dimension_lower_bound": [
                {
                    "client_basis_size": size,
                    "phase_classes_modulo_global_phase": D ** (size - 1),
                    "minimum_exact_program_hilbert_dimension": D ** (size - 1),
                }
                for size in range(1, 7)
            ],
            "secret_dependent_program_storage_is_a_charged_resource": True,
        },
        "equal_coherent_oracle_access_collapse": {
            "phase_route_query_sequence": ["Q_A_ON_CHI1", "Q_B_ON_CHI1"],
            "equal_access_comparator_query_sequence": [
                "Q_A_ON_CHI1",
                "Q_B_ON_CHI1",
            ],
            "phase_route_coherent_queries": 2,
            "equal_access_comparator_coherent_queries": 2,
            "query_sequences_identical": True,
            "exact_boundaries_identical": True,
            "unique_phase_qemu_query_advantage": False,
            "total_resource_advantage": False,
        },
        "m241_m242_linear_calibration_negative": {
            "m241_hidden_linear_phase_calibration_improved": False,
            "m242_tensor_factored_hidden_linear_phase_calibration_improved": False,
            "reason": "THIS_FIXED_D3_TWO_POINT_ADDITION_ORACLE_FIXTURE_NEITHER_IMPROVES_THE_P5_HIDDEN_LINEAR_QUERY_LAW_NOR_ESTABLISHES_A_SCALING_SEPARATION",
            "linear_secret_query_lower_bound_changed": False,
            "m241_escape": False,
            "m242_escape": False,
        },
        "forrelation_oracle_cost_caveat": {
            "prospective_only": True,
            "implemented_here": False,
            "forrelation_query_separation_claimed": False,
            "required_change": "REPLACE_PUBLIC_TABLES_WITH_A_RESTRICTED_PROMISE_BLACK_BOX_INTERFACE_AND_PREDECLARE_THE_COMPARATOR_ACCESS",
            "oracle_generation_cost_must_be_charged": True,
            "oracle_custody_cost_must_be_charged": True,
            "carrier_preparation_and_precision_costs_must_be_charged": True,
            "total_resource_accounting_required": True,
            "public_descriptor_access_would_restore_direct_compilation": True,
        },
    }

    architecture_scope = {
        "phase_qemu_layer_classification": (
            "MECHANISM_SEARCH_DIGITAL_TWIN_OUTSIDE_QEMU_DEVICE"
        ),
        "qemu_device_implemented": False,
        "common_guest_visible_device_contract_exercised": False,
        "eligible_for_mechanism_kill": True,
        "eligible_for_architecture_promotion": False,
        "promotion_requires_common_phase_qemu_device_or_backend": True,
        "reference_verifies_algebra_only": True,
        "reference_can_promote_architecture": False,
        "scope_statement": (
            "REFERENCE_VERIFIES_EXACT_ALGEBRA_ONLY_AND_CANNOT_PROMOTE_"
            "PHASE_QEMU_ARCHITECTURE"
        ),
    }

    resource_scope_accounting = {
        "scope": (
            "INDEPENDENT_EXACT_ALGEBRA_REFERENCE_MATERIALIZATIONS_ONLY_NO_"
            "PRODUCTION_RESIDENT_TRANSIENT_OR_PEAK_BACKING_MEASUREMENT"
        ),
        "reference_exact_matrix_materializations": {
            "character_density": {
                "dimension": D,
                "matrix_entries": D**2,
            },
            "single_client_target_density": {
                "dimension": CLIENT_DIMENSION * D,
                "matrix_entries": (CLIENT_DIMENSION * D) ** 2,
            },
            "two_fresh_clients_target_density": {
                "dimension": CLIENT_DIMENSION**2 * D,
                "matrix_entries": (CLIENT_DIMENSION**2 * D) ** 2,
            },
            "qutrit_spectator_reference_density": {
                "dimension": D**2,
                "matrix_entries": D**4,
            },
            "client_reference_target_density": {
                "dimension": CLIENT_DIMENSION**2 * D,
                "matrix_entries": (CLIENT_DIMENSION**2 * D) ** 2,
            },
        },
        "reference_largest_materialized_matrix_dimension": (
            CLIENT_DIMENSION**2 * D
        ),
        "reference_largest_materialized_matrix_entries": (
            CLIENT_DIMENSION**2 * D
        ) ** 2,
        "reference_public_table_entries_accounted": (
            len(PROGRAM_A) + len(PROGRAM_B)
        ),
        "reference_stipulated_coherent_queries_accounted": 2,
        "reference_structural_matrix_entry_counts_declared": True,
        "reference_exact_arithmetic_and_hashing_work_charged": True,
        "reference_python_object_byte_cost_measured": False,
        "reference_runtime_peak_bytes_measured": False,
        "production_resident_backing_materialized_by_reference": False,
        "production_transient_backing_materialized_by_reference": False,
        "production_resident_backing_cells_independently_measured": False,
        "production_transient_backing_cells_independently_measured": False,
        "production_peak_backing_cells_independently_measured": False,
        "production_peak_bytes_independently_measured": False,
        "production_resource_parity_claimed": False,
        "production_resident_transient_or_peak_claims_are_out_of_scope": True,
    }

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
        "program_a_exact_kickback": program_a["exact_kickback_factorization"],
        "program_b_exact_kickback": program_b["exact_kickback_factorization"],
        "program_a_carrier_return": program_a[
            "exact_character_carrier_marginal_return"
        ],
        "program_b_carrier_return": program_b[
            "exact_character_carrier_marginal_return"
        ],
        "two_oracles_are_distinct": fixtures[
            "two_fresh_client_same_logical_carrier_transactions"
        ]["distinct_oracle_permutations"],
        "program_a_program_b_phase_gates_are_nonproportional": fixtures[
            "nonproportional_program_control"
        ]["a_and_b_phase_diagonals_are_nonproportional"],
        "two_fresh_client_boundaries_are_distinct": fixtures[
            "two_fresh_client_same_logical_carrier_transactions"
        ]["fresh_client_boundaries_are_distinct"],
        "two_fresh_client_supply_is_explicit": (
            fixtures["two_fresh_client_same_logical_carrier_transactions"]
            ["client_supply_count"]
            == 2
        ),
        "first_fresh_client_transaction_exact": fixtures[
            "two_fresh_client_same_logical_carrier_transactions"
        ]["program_a_fresh_transaction_exact_factorization"],
        "two_fresh_client_combined_boundary_exact": fixtures[
            "two_fresh_client_same_logical_carrier_transactions"
        ]["combined_fresh_client_boundary_exact_factorization"],
        "two_fresh_client_carrier_return": fixtures[
            "two_fresh_client_same_logical_carrier_transactions"
        ]["carrier_returns_after_each_query"],
        "same_client_sequential_identity_is_diagnostic_only": bool(
            fixtures["same_client_sequential_composition_diagnostic"]
            ["sequential_exact_factorization"]
            and fixtures["same_client_sequential_composition_diagnostic"]
            ["sequential_phase_is_identity"]
            and fixtures["same_client_sequential_composition_diagnostic"]
            ["result_free_global_identity_only"]
            and not fixtures["same_client_sequential_composition_diagnostic"]
            ["is_reuse_boundary"]
        ),
        "inert_qutrit_spectator_reference_sentinel_exact": bool(
            fixtures["spectator_reference_bell_preservation"]["exactly_unchanged"]
            and fixtures["spectator_reference_bell_preservation"]
            ["spectator_reference_marginal_is_maximally_mixed"]
        ),
        "client_reference_kickback_factorization": fixtures[
            "spectator_reference_bell_preservation"
        ]["client_reference_target_exact_factorization"],
        "client_reference_completeness_test_preserved": (
            fixtures["spectator_reference_bell_preservation"]
            ["client_reference_purity_after_program_a"]
            == ONE
            and fixtures["spectator_reference_bell_preservation"]
            ["client_reference_marginal_is_maximally_mixed"]
        ),
        "inert_sentinel_and_client_reference_tests_are_dimensionally_distinct": (
            len(
                fixtures["spectator_reference_bell_preservation"]
                ["spectator_reference_marginal_after_both_queries"]
            )
            == D
            and len(
                fixtures["spectator_reference_bell_preservation"]
                ["client_reference_marginal_after_program_a"]
            )
            == CLIENT_DIMENSION
        ),
        "basis_a_entangles_and_does_not_return": bool(
            fixtures["computational_basis_entanglement_controls"]["program_a"]
            ["client_target_entangled"]
            and not fixtures["computational_basis_entanglement_controls"]
            ["program_a"]["target_returned"]
        ),
        "basis_b_entangles_and_does_not_return": bool(
            fixtures["computational_basis_entanglement_controls"]["program_b"]
            ["client_target_entangled"]
            and not fixtures["computational_basis_entanglement_controls"]
            ["program_b"]["target_returned"]
        ),
        "uniform_character_eta_is_delta": (
            [eta(uniform_weights, delta) for delta in range(D)]
            == [ONE, ZERO, ZERO]
        ),
        "mixed_character_carrier_marginal_returns": fixtures[
            "mixed_character_marginal_return_and_eta_dephasing"
        ]["carrier_marginal_returns"],
        "mixed_character_client_dephases": fixtures[
            "mixed_character_marginal_return_and_eta_dephasing"
        ]["program_a_client_is_fully_dephased"],
        "mixed_character_joint_does_not_return": not fixtures[
            "mixed_character_marginal_return_and_eta_dephasing"
        ]["joint_state_returns"],
        "branch_record_is_orthogonal": fixtures["branch_record_environment"]
        ["program_a_branch_records_are_orthogonal"],
        "public_direct_compiler_matches_all_boundaries": bool(
            fixtures["public_law_direct_compiler"]["direct_program_a_boundary_exact"]
            and fixtures["public_law_direct_compiler"]
            ["direct_program_b_boundary_exact"]
            and fixtures["public_law_direct_compiler"]["direct_combined_boundary_exact"]
        ),
        "nielsen_chuang_classes_exact": bool(
            pairwise_nonproportional
            and len(class_representatives) == D
            and fixtures["exact_reusable_program_orthogonality"]
            ["program_a_and_b_are_nonproportional_and_witness_orthogonality"]
            and fixtures["exact_reusable_program_orthogonality"]
            ["minimum_program_dimension_for_two_basis_inputs"]
            == D
        ),
        "no_programming_zero_overlap_is_theorem_requirement_not_measurement": bool(
            fixtures["exact_reusable_program_orthogonality"]
            ["theorem_required_program_overlap_if_fixed_exact_processor"]
            == ZERO
            and fixtures["exact_reusable_program_orthogonality"]
            ["zero_overlap_is_hypothetical_no_programming_theorem_requirement"]
            and not fixtures["exact_reusable_program_orthogonality"]
            ["program_states_materialized"]
            and not fixtures["exact_reusable_program_orthogonality"]
            ["program_overlap_executed_or_measured"]
        ),
        "equal_access_runs_identical_queries": fixtures[
            "equal_coherent_oracle_access_collapse"
        ]["query_sequences_identical"],
        "m241_m242_calibration_is_negative": bool(
            not fixtures["m241_m242_linear_calibration_negative"]
            ["m241_hidden_linear_phase_calibration_improved"]
            and not fixtures["m241_m242_linear_calibration_negative"]
            ["m242_tensor_factored_hidden_linear_phase_calibration_improved"]
        ),
        "forrelation_is_prospective_only": bool(
            fixtures["forrelation_oracle_cost_caveat"]["prospective_only"]
            and not fixtures["forrelation_oracle_cost_caveat"]["implemented_here"]
            and not fixtures["forrelation_oracle_cost_caveat"]
            ["forrelation_query_separation_claimed"]
        ),
        "architecture_scope_fails_closed_outside_qemu_device": bool(
            architecture_scope["phase_qemu_layer_classification"]
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
            and architecture_scope["reference_verifies_algebra_only"]
            and not architecture_scope["reference_can_promote_architecture"]
        ),
        "resource_scope_excludes_unmeasured_production_backing": bool(
            resource_scope_accounting[
                "reference_largest_materialized_matrix_dimension"
            ]
            == CLIENT_DIMENSION**2 * D
            and resource_scope_accounting[
                "reference_largest_materialized_matrix_entries"
            ]
            == (CLIENT_DIMENSION**2 * D) ** 2
            and resource_scope_accounting[
                "reference_public_table_entries_accounted"
            ]
            == len(PROGRAM_A) + len(PROGRAM_B)
            and resource_scope_accounting[
                "reference_stipulated_coherent_queries_accounted"
            ]
            == 2
            and resource_scope_accounting[
                "reference_structural_matrix_entry_counts_declared"
            ]
            and resource_scope_accounting[
                "reference_exact_arithmetic_and_hashing_work_charged"
            ]
            and not resource_scope_accounting[
                "reference_python_object_byte_cost_measured"
            ]
            and not resource_scope_accounting["reference_runtime_peak_bytes_measured"]
            and not resource_scope_accounting[
                "production_resident_backing_materialized_by_reference"
            ]
            and not resource_scope_accounting[
                "production_transient_backing_materialized_by_reference"
            ]
            and not resource_scope_accounting[
                "production_resident_backing_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "production_transient_backing_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "production_peak_backing_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "production_peak_bytes_independently_measured"
            ]
            and not resource_scope_accounting["production_resource_parity_claimed"]
            and resource_scope_accounting[
                "production_resident_transient_or_peak_claims_are_out_of_scope"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise AssertionError(f"independent exact self-check failed: {failed}")

    claim_payload = {
        "milestone": "M268",
        "claim": CLAIM,
        "ceiling": CEILING,
        "reference_restoration_classification": RESTORATION_CLASSIFICATION,
        "reference_restoration_scope": REFERENCE_RESTORATION_SCOPE,
        "expected_production_restoration_classification": (
            EXPECTED_PRODUCTION_RESTORATION_CLASSIFICATION
        ),
        "expected_production_restoration_scope": EXPECTED_PRODUCTION_RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
    }
    payload = {
        "schema": SCHEMA,
        "reference_id": REFERENCE_ID,
        "milestone": "M268",
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": REFERENCE_RESTORATION_SCOPE,
        "expected_production_restoration_classification": (
            EXPECTED_PRODUCTION_RESTORATION_CLASSIFICATION
        ),
        "expected_production_restoration_scope": EXPECTED_PRODUCTION_RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "claim_payload": claim_payload,
        "mathematical_conventions": {
            "target_shift": "X_A|Y>=|Y+A_MOD_3>",
            "character_state": "|CHI_K>=SUM_Y_omega^(-K*Y)|Y>/SQRT_3",
            "kickback": "X_A|CHI_K>=omega^(K*A)|CHI_K>",
            "addition_oracle": "Q_F|X,Y>=|X,Y+F_X_MOD_3>",
            "client_phase": "D_F_K=DIAG_X_omega^(K*F_X)",
            "density_field": "Q(omega)_WITH_RATIONAL_NORMALIZATION",
        },
        "fixtures": fixtures,
        "checks": checks,
        "resources": {
            "client_dimension": CLIENT_DIMENSION,
            "client_supply_count": 2,
            "character_carrier_dimension": D,
            "client_target_joint_dimension": CLIENT_DIMENSION * D,
            "two_fresh_clients_target_joint_dimension": CLIENT_DIMENSION**2 * D,
            "combined_fresh_client_boundary_dimension": CLIENT_DIMENSION**2,
            "program_count": 2,
            "public_table_entries": len(PROGRAM_A) + len(PROGRAM_B),
            "stipulated_coherent_oracle_queries": 2,
            "exact_arithmetic_field": "Q(omega)",
            "floating_point_operations_on_decision_path": 0,
            "public_direct_compiler_oracle_queries": 0,
            "equal_access_comparator_oracle_queries": 2,
            "secret_program_dimension_lower_bound_for_two_client_basis_states": D,
            "source_bytes_and_json_hashing_are_charged_software_work": True,
            "oracle_generation_physical_cost_accounted": False,
            "carrier_preparation_physical_cost_accounted": False,
            "physical_energy_accounted": False,
        },
        "nonclaims": {
            "same_backing_restoration": False,
            "executed_restoration": False,
            "physical_restoration": False,
            "physical_oracle": False,
            "physical_oracle_generation": False,
            "physical_carrier_custody": False,
            "physical_character_state_preparation": False,
            "query_separation": False,
            "total_resource_advantage": False,
            "unique_phase_qemu_advantage": False,
            "secret_program_state_free": False,
            "m241_escape": False,
            "m242_escape": False,
            "m257_escape": False,
            "forrelation_implemented": False,
            "unbounded_computation": False,
            "small_wall_crossing": False,
            "bit_replaced_with_pi": False,
        },
        "m257": {
            "guardrail": "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE",
            "same_canonical_input_descriptors_auxiliary_secret_state_and_oracle_access_required": True,
            "public_descriptor_case_has_zero_query_direct_compiler": True,
            "equal_coherent_oracle_case_runs_the_identical_two_queries": True,
            "guardrail_remains_intact": True,
            "escape_established": False,
        },
        "architecture_scope": architecture_scope,
        "resource_scope_accounting": resource_scope_accounting,
        "architecture_authority": {
            "reference_is_independent_exact_algebra": True,
            "reference_imports_package_code": False,
            "reference_reads_external_artifacts": False,
            "reference_executes_restoration": False,
            "reference_asserts_same_backing_identity": False,
            "reference_is_physical_evidence": False,
            "two_fresh_client_logical_carrier_transactions_are_formal_semantics_only": True,
            "expected_production_scope_is_emitted_not_claimed_by_reference": True,
        },
        "reference_self_assertion": "PASS_INDEPENDENT_EXACT_QUTRIT_KICKBACK_REFERENCE",
        "status": "PASS_INDEPENDENT_EXACT_QUTRIT_KICKBACK_REFERENCE",
        "terminal": False,
    }

    # Exact conversion happens before hashing, so floats or unsupported objects fail.
    payload["claim_payload_sha256"] = hashlib.sha256(
        exact_bytes(claim_payload)
    ).hexdigest()
    payload["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(exact_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
