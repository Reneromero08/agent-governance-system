#!/usr/bin/env python3
"""Independent exact reference for the M263 subgap-resolvent diagnostic.

The reference intentionally imports no production module.  It reconstructs
the finite stationary one-channel fixtures with ``Fraction`` arithmetic and a
small Q(i) implementation, certifies the Krylov ranks independently over Q
and two split prime fields, and emits deterministic JSON on stdout.

This is a stationary algebraic oracle.  It does not execute a time-domain
scattering process, target restoration, device reuse, or physical hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


SPLIT_PRIMES = (65537, 998244353)
PATH_SIZES = (2, 4, 8, 16, 32)
INTERACTING_SIZES = range(2, 7)
BETHE_PARTICLE_COUNTS = (2, 4, 8)
CLAIM = (
    "EXACT_RATIONAL_SINGLE_CHANNEL_SUBGAP_CAYLEY_RESOLVENT_DIAGNOSTIC_"
    "IMPLEMENTS_A_STIPULATED_FORMAL_STATIONARY_UNIT_MODULUS_BOUNDARY_LAW_"
    "AT_DECLARED_FIXTURES_WITH_DISTINCT_ENERGY_DESCRIPTOR_REUSE_GROWING_"
    "EXACT_KRYLOV_RANK_AND_PATH_ONLY_FIXED_MARGIN_EFFECTIVE_DEPTH_BOUND_"
    "PLUS_TILTED_FIELD_AND_BETHE_FACTORIZATION_CONTROLS"
)
CLAIM_CEILING = (
    "EXACT_DETERMINISTIC_SOFTWARE_FINITE_DIMENSIONAL_RATIONAL_ONE_CHANNEL_"
    "K_MATRIX_BOUNDARY_MODEL_WITH_FORMAL_STATIONARY_ASYMPTOTIC_RETURN_ONLY"
)
DISPOSITION = (
    "GROWING_EXACT_KRYLOV_RANK_ALONE_IS_NOT_AN_APPROXIMATION_LOWER_BOUND_"
    "PATH_FIXED_MARGIN_HAS_COMPACT_STREAMED_SHADOW_AND_BETHE_FACTORIZED_"
    "EIGENPHASE_IS_PUBLIC_RAPIDITY_PRODUCT_NEAR_THRESHOLD_TIME_DOMAIN_"
    "QUALIFICATION_REQUIRED"
)
NEXT_MECHANISM = (
    "NEAR_THRESHOLD_NONINTEGRABLE_BOUNDARY_RESOLVENT_WITH_EXPLICIT_WIGNER_"
    "DELAY_FINITE_BANDWIDTH_PRECISION_PREPARATION_AMORTIZATION_AND_TENSOR_"
    "NETWORK_RESOURCE_CROSSOVER"
)


def fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class GaussianQ:
    """A deliberately minimal exact Q(i) value."""

    real: Fraction
    imag: Fraction

    @staticmethod
    def make(real: int | Fraction, imag: int | Fraction = 0) -> "GaussianQ":
        return GaussianQ(Fraction(real), Fraction(imag))

    def __add__(self, other: "GaussianQ") -> "GaussianQ":
        return GaussianQ(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "GaussianQ") -> "GaussianQ":
        return GaussianQ(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "GaussianQ") -> "GaussianQ":
        return GaussianQ(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def conjugate(self) -> "GaussianQ":
        return GaussianQ(self.real, -self.imag)

    def norm_squared(self) -> Fraction:
        return self.real * self.real + self.imag * self.imag

    def inverse(self) -> "GaussianQ":
        denominator = self.norm_squared()
        if denominator == 0:
            raise ZeroDivisionError("zero Gaussian rational")
        conjugate = self.conjugate()
        return GaussianQ(conjugate.real / denominator, conjugate.imag / denominator)

    def __truediv__(self, other: "GaussianQ") -> "GaussianQ":
        return self * other.inverse()

    def json(self) -> dict[str, object]:
        return {"real": fraction_json(self.real), "imag": fraction_json(self.imag)}


QI_ONE = GaussianQ.make(1)


def cayley_phase(green: Fraction, kappa: Fraction = Fraction(1)) -> GaussianQ:
    """Return (1-i kappa G)/(1+i kappa G) exactly."""

    reaction = kappa * green
    return GaussianQ.make(1, -reaction) / GaussianQ.make(1, reaction)


def lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


def matvec(matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]) -> list[Fraction]:
    return [sum((entry * value for entry, value in zip(row, vector)), Fraction(0)) for row in matrix]


def matvec_int(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    return [sum(entry * value for entry, value in zip(row, vector)) for row in matrix]


def solve_linear(
    matrix: Sequence[Sequence[Fraction]],
    right_hand_side: Sequence[Fraction],
) -> list[Fraction]:
    """Exact pivoted elimination for the small general fixtures."""

    dimension = len(matrix)
    work = [list(row) + [right_hand_side[index]] for index, row in enumerate(matrix)]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ArithmeticError("singular exact resolvent fixture")
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(dimension):
            if row == column or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                current - factor * pivot_entry
                for current, pivot_entry in zip(work[row], work[column])
            ]
    return [work[row][-1] for row in range(dimension)]


def solve_spd_ldlt(
    matrix: Sequence[Sequence[Fraction]],
    right_hand_side: Sequence[Fraction],
) -> tuple[list[Fraction], Fraction]:
    """Exact LDL^T solve and determinant for a positive-definite block."""

    dimension = len(matrix)
    lower = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    diagonal = [Fraction(0) for _ in range(dimension)]
    for row in range(dimension):
        lower[row][row] = Fraction(1)
        for column in range(row):
            residual = matrix[row][column] - sum(
                lower[row][index] * diagonal[index] * lower[column][index]
                for index in range(column)
            )
            if diagonal[column] == 0:
                raise ArithmeticError("zero LDL pivot")
            lower[row][column] = residual / diagonal[column]
        diagonal[row] = matrix[row][row] - sum(
            lower[row][index] * lower[row][index] * diagonal[index]
            for index in range(row)
        )
        if diagonal[row] <= 0:
            raise ArithmeticError("declared excitation block is not positive definite")

    forward = [Fraction(0) for _ in range(dimension)]
    for row in range(dimension):
        forward[row] = right_hand_side[row] - sum(
            lower[row][column] * forward[column] for column in range(row)
        )
    scaled = [forward[index] / diagonal[index] for index in range(dimension)]
    solution = [Fraction(0) for _ in range(dimension)]
    for row in range(dimension - 1, -1, -1):
        solution[row] = scaled[row] - sum(
            lower[column][row] * solution[column]
            for column in range(row + 1, dimension)
        )
    determinant = math.prod(diagonal, start=Fraction(1))
    return solution, determinant


def boundary_green_solve(
    hamiltonian: Sequence[Sequence[Fraction]],
    energy: Fraction,
) -> Fraction:
    dimension = len(hamiltonian)
    resolvent_matrix = [
        [
            (energy if row == column else Fraction(0)) - hamiltonian[row][column]
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    boundary = [Fraction(1)] + [Fraction(0)] * (dimension - 1)
    solution = solve_linear(resolvent_matrix, boundary)
    residual = matvec(resolvent_matrix, solution)
    if residual != boundary:
        raise ArithmeticError("exact resolvent residual failed")
    return solution[0]


def boundary_green_continued_fraction(
    diagonal: Sequence[Fraction],
    off_diagonal: Sequence[Fraction],
    energy: Fraction,
) -> Fraction:
    if len(off_diagonal) + 1 != len(diagonal):
        raise ValueError("tridiagonal dimensions disagree")
    denominator = energy - diagonal[-1]
    for index in range(len(diagonal) - 2, -1, -1):
        denominator = (
            energy
            - diagonal[index]
            - off_diagonal[index] * off_diagonal[index] / denominator
        )
    return Fraction(1) / denominator


def bareiss_determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Fraction-free exact determinant with row pivoting."""

    dimension = len(matrix)
    if dimension == 0:
        return 1
    if dimension == 1:
        return matrix[0][0]
    work = [list(row) for row in matrix]
    previous_pivot = 1
    sign = 1
    for column in range(dimension - 1):
        pivot = next(
            (row for row in range(column, dimension) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign = -sign
        pivot_value = work[column][column]
        for row in range(column + 1, dimension):
            for target_column in range(column + 1, dimension):
                numerator = (
                    work[row][target_column] * pivot_value
                    - work[row][column] * work[column][target_column]
                )
                if numerator % previous_pivot:
                    raise ArithmeticError("Bareiss division was not exact")
                work[row][target_column] = numerator // previous_pivot
            work[row][column] = 0
        previous_pivot = pivot_value
    return sign * work[-1][-1]


def scaled_integer_matrix(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[list[list[int]], int]:
    scale = 1
    for row in matrix:
        for value in row:
            scale = lcm(scale, value.denominator)
    return [[int(value * scale) for value in row] for row in matrix], scale


def krylov_matrix_int(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    dimension = len(matrix)
    vector = [1] + [0] * (dimension - 1)
    columns: list[list[int]] = []
    for _ in range(dimension):
        columns.append(vector)
        vector = matvec_int(matrix, vector)
    return [[columns[column][row] for column in range(dimension)] for row in range(dimension)]


def signed_integer_sha256(value: int) -> str:
    sign = b"-" if value < 0 else b"+"
    magnitude = abs(value)
    encoded = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
    return hashlib.sha256(sign + encoded).hexdigest()


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    factor = 3
    limit = math.isqrt(value)
    while factor <= limit:
        if value % factor == 0:
            return False
        factor += 2
    return True


def fraction_mod(value: Fraction, prime: int) -> int:
    denominator = value.denominator % prime
    if denominator == 0:
        raise ArithmeticError("prime divides an exact denominator")
    return (value.numerator % prime) * pow(denominator, -1, prime) % prime


def rank_mod_prime(matrix: Sequence[Sequence[int]], prime: int) -> int:
    work = [[value % prime for value in row] for row in matrix]
    rank = 0
    columns = len(work[0]) if work else 0
    for column in range(columns):
        pivot = next(
            (row for row in range(rank, len(work)) if work[row][column] % prime),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, prime)
        work[rank] = [(value * inverse) % prime for value in work[rank]]
        for row in range(len(work)):
            if row == rank or work[row][column] == 0:
                continue
            factor = work[row][column]
            work[row] = [
                (left - factor * right) % prime
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == len(work):
            break
    return rank


def krylov_rank_mod_prime(
    matrix: Sequence[Sequence[Fraction]],
    prime: int,
) -> int:
    modular = [[fraction_mod(value, prime) for value in row] for row in matrix]
    dimension = len(modular)
    vector = [1] + [0] * (dimension - 1)
    columns: list[list[int]] = []
    for _ in range(dimension):
        columns.append(vector)
        vector = [
            sum(entry * value for entry, value in zip(row, vector)) % prime
            for row in modular
        ]
    krylov = [
        [columns[column][row] for column in range(dimension)]
        for row in range(dimension)
    ]
    return rank_mod_prime(krylov, prime)


def exact_krylov_certificate(
    matrix: Sequence[Sequence[Fraction]],
) -> dict[str, object]:
    integer_matrix, scale = scaled_integer_matrix(matrix)
    determinant = bareiss_determinant(krylov_matrix_int(integer_matrix))
    if determinant == 0:
        raise ArithmeticError("expected cyclic boundary vector was not cyclic")
    dimension = len(matrix)
    prime_ranks = {
        str(prime): krylov_rank_mod_prime(matrix, prime) for prime in SPLIT_PRIMES
    }
    return {
        "krylov_rank_certified_exact": dimension,
        "krylov_rank_by_prime": prime_ranks,
        "all_split_prime_ranks_match_exact": all(
            rank == dimension for rank in prime_ranks.values()
        ),
        "integer_scaling_factor": scale,
        "nonzero_krylov_determinant_sign": 1 if determinant > 0 else -1,
        "krylov_determinant_bit_length": abs(determinant).bit_length(),
        "krylov_determinant_sha256": signed_integer_sha256(determinant),
    }


def tridiagonal_matrix(
    diagonal: Sequence[Fraction],
    off_diagonal: Sequence[Fraction],
) -> list[list[Fraction]]:
    dimension = len(diagonal)
    matrix = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    for index, value in enumerate(diagonal):
        matrix[index][index] = value
    for index, value in enumerate(off_diagonal):
        matrix[index][index + 1] = value
        matrix[index + 1][index] = value
    return matrix


def mode_fixture(
    name: str,
    diagonal: Sequence[int],
    off_diagonal: Sequence[int],
) -> dict[str, object]:
    diagonal_q = [Fraction(value) for value in diagonal]
    off_q = [Fraction(value) for value in off_diagonal]
    hamiltonian = tridiagonal_matrix(diagonal_q, off_q)
    energy = Fraction(0)
    green = boundary_green_solve(hamiltonian, energy)
    continued = boundary_green_continued_fraction(diagonal_q, off_q, energy)
    if green != continued:
        raise ArithmeticError("direct and continued-fraction mode resolvents disagree")
    phase = cayley_phase(green)
    if phase.norm_squared() != 1:
        raise ArithmeticError("one-channel Cayley phase lost unit modulus")
    certificate = exact_krylov_certificate(hamiltonian)
    return {
        "name": name,
        "dimension": len(diagonal),
        "energy": fraction_json(energy),
        "kappa": fraction_json(Fraction(1)),
        "diagonal": diagonal,
        "off_diagonal": off_diagonal,
        "green": fraction_json(green),
        "continuant_green": fraction_json(continued),
        "cayley_s": phase.json(),
        "cayley_unit_modulus_exact": True,
        "parity": True,
        **certificate,
    }


def exact_mode_fixtures() -> list[dict[str, object]]:
    fixtures = [
        mode_fixture("ONE_MODE", (3,), ()),
        mode_fixture("COUPLED_TWO_MODE", (3, 3), (1,)),
        mode_fixture("TRIDIAGONAL_THREE_MODE", (4, 4, 4), (1, 1)),
    ]
    if len({json.dumps(item["cayley_s"], sort_keys=True) for item in fixtures}) != len(fixtures):
        raise ArithmeticError("mode fixtures did not produce distinct phases")
    return fixtures


def non_unit_kappa_control() -> dict[str, object]:
    green = Fraction(-1, 3)
    kappa = Fraction(2)
    phase = cayley_phase(green, kappa)
    expected = GaussianQ.make(Fraction(5, 13), Fraction(12, 13))
    if phase != expected or phase.norm_squared() != 1:
        raise ArithmeticError("linear non-unit-kappa Cayley control failed")
    return {
        "green": fraction_json(green),
        "kappa": fraction_json(kappa),
        "cayley_s": phase.json(),
        "expected_cayley_s": expected.json(),
        "linear_kappa_recurrence_exposed": True,
        "unit_modulus_exact": True,
    }


def path_hamiltonian(size: int) -> list[list[Fraction]]:
    return tridiagonal_matrix(
        [Fraction(4)] * size,
        [Fraction(1)] * (size - 1),
    )


def path_family() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    previous_rank = 0
    for size in PATH_SIZES:
        hamiltonian = path_hamiltonian(size)
        green = boundary_green_continued_fraction(
            [Fraction(4)] * size,
            [Fraction(1)] * (size - 1),
            Fraction(0),
        )
        if green != boundary_green_solve(hamiltonian, Fraction(0)):
            raise ArithmeticError("path resolvent implementations disagree")
        phase = cayley_phase(green)
        certificate = exact_krylov_certificate(hamiltonian)
        rank = int(certificate["krylov_rank_certified_exact"])
        if rank != size or rank <= previous_rank:
            raise ArithmeticError("path Krylov rank did not grow with path size")
        previous_rank = rank
        records.append(
            {
                "n": size,
                "energy": fraction_json(Fraction(0)),
                "green": fraction_json(green),
                "continuant_green": fraction_json(green),
                "cayley_s": phase.json(),
                "cayley_unit_modulus_exact": phase.norm_squared() == 1,
                "boundary_vector": "e1",
                "path_krylov_triangular_frontier_coefficient": 1,
                "parity": True,
                **certificate,
            }
        )
    return records


def path_adjacency_matvec(vector: Sequence[int]) -> list[int]:
    output = [0] * len(vector)
    for index in range(len(vector)):
        if index:
            output[index] += vector[index - 1]
        if index + 1 < len(vector):
            output[index] += vector[index + 1]
    return output


def truncated_path_green(size: int, order: int, diagonal: int = 4) -> Fraction:
    vector = [1] + [0] * (size - 1)
    result = Fraction(0)
    for power in range(order + 1):
        moment = vector[0]
        result += Fraction((-1) ** (power + 1) * moment, diagonal ** (power + 1))
        vector = path_adjacency_matvec(vector)
    return result


def minimum_neumann_order(
    epsilon: Fraction,
    diagonal_bound: Fraction,
    ratio: Fraction,
) -> int:
    order = 0
    while (
        Fraction(2, 1)
        / diagonal_bound
        * ratio ** (order + 1)
        / (Fraction(1) - ratio)
        > epsilon
    ):
        order += 1
    return order


def fixed_margin_neumann() -> dict[str, object]:
    adjacency_bound = Fraction(2)
    diagonal = Fraction(4)
    ratio = adjacency_bound / diagonal
    epsilon = Fraction(1, 2**20)
    order = minimum_neumann_order(epsilon, diagonal, ratio)
    phase_tail_bound = (
        Fraction(2) / diagonal * ratio ** (order + 1) / (Fraction(1) - ratio)
    )
    previous_bound = (
        Fraction(2) / diagonal * ratio**order / (Fraction(1) - ratio)
    )
    if phase_tail_bound > epsilon or previous_bound <= epsilon:
        raise ArithmeticError("Neumann order is not minimal")

    records = []
    for size in PATH_SIZES:
        exact_green = boundary_green_continued_fraction(
            [diagonal] * size,
            [Fraction(1)] * (size - 1),
            Fraction(0),
        )
        approximate_green = truncated_path_green(size, order, int(diagonal))
        exact_phase = cayley_phase(exact_green)
        approximate_phase = cayley_phase(approximate_green)
        actual_phase_error_squared = (exact_phase - approximate_phase).norm_squared()
        if actual_phase_error_squared > phase_tail_bound * phase_tail_bound:
            raise ArithmeticError("actual Cayley error exceeded fixed-margin bound")
        records.append(
            {
                "n": size,
                "truncated_green": fraction_json(approximate_green),
                "exact_green": fraction_json(exact_green),
                "actual_phase_error_squared": fraction_json(actual_phase_error_squared),
                "phase_tail_bound": fraction_json(phase_tail_bound),
                "resolvent_vector_effective_sites": min(size, order + 1),
                "boundary_scalar_effective_sites": min(size, order // 2 + 1),
                "boundary_scalar_max_graph_distance": min(size - 1, order // 2),
                "within_epsilon": actual_phase_error_squared <= epsilon * epsilon,
            }
        )
    return {
        "operator_bound_J": fraction_json(adjacency_bound),
        "diagonal_D": fraction_json(diagonal),
        "uniform_ratio_q": fraction_json(ratio),
        "epsilon": fraction_json(epsilon),
        "minimal_truncation_order_K": order,
        "phase_tail_bound": fraction_json(phase_tail_bound),
        "previous_order_phase_tail_bound": fraction_json(previous_bound),
        "records": records,
        "fixed_margin_not_near_threshold": True,
    }


def tilted_field_excitation_block(
    qubits: int,
) -> tuple[list[list[Fraction]], Fraction, Fraction]:
    dimension = 1 << qubits
    interaction = [
        [Fraction(0) for _ in range(dimension)] for _ in range(dimension)
    ]
    for state in range(dimension):
        z_values = [Fraction(1 if not (state >> site) & 1 else -1) for site in range(qubits)]
        zz_diagonal = sum(
            (z_values[site] * z_values[site + 1] / 2 for site in range(qubits - 1)),
            Fraction(0),
        )
        tilted_diagonal = sum(
            (
                Fraction((site + 1) ** 2, qubits + 2) * z_values[site]
                for site in range(qubits)
            ),
            Fraction(0),
        )
        interaction[state][state] += zz_diagonal + tilted_diagonal
        for site in range(qubits):
            interaction[state][state ^ (1 << site)] += 1

    coefficient_bound = (
        Fraction(qubits)
        + Fraction(qubits - 1, 2)
        + sum(
            (Fraction((site + 1) ** 2, qubits + 2) for site in range(qubits)),
            Fraction(0),
        )
    )
    diagonal_offset = 2 * coefficient_bound + 1
    excitation = [row[:] for row in interaction]
    for index in range(dimension):
        excitation[index][index] += diagonal_offset
    if any(excitation[row][column] != excitation[column][row] for row in range(dimension) for column in range(dimension)):
        raise ArithmeticError("tilted-field excitation block lost symmetry")
    return excitation, coefficient_bound, diagonal_offset


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def interacting_flagged_blocks() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for qubits in INTERACTING_SIZES:
        hamiltonian, coefficient_bound, diagonal_offset = tilted_field_excitation_block(qubits)
        dimension = len(hamiltonian)
        certificate = exact_krylov_certificate(hamiltonian)
        if certificate["krylov_rank_certified_exact"] != dimension:
            raise ArithmeticError("tilted-field block did not have full Krylov rank")
        record: dict[str, object] = {
            "n": qubits,
            "dimension": dimension,
            "energy": fraction_json(Fraction(0)),
            "kappa": fraction_json(Fraction(1)),
            "coefficient_bound_J": fraction_json(coefficient_bound),
            "diagonal_offset_D": fraction_json(diagonal_offset),
            "strict_subgap_margin_D_minus_J": fraction_json(
                diagonal_offset - coefficient_bound
            ),
            "has_x_flip_terms": True,
            "has_nearest_neighbor_zz_terms": True,
            "has_rational_tilted_z_terms": True,
            "noncommuting_interaction_terms_flagged": True,
            "nonintegrability_not_proved": True,
            "parity": True,
            **certificate,
        }
        if qubits <= 4:
            boundary = [Fraction(1)] + [Fraction(0)] * (dimension - 1)
            inverse_column, determinant = solve_spd_ldlt(hamiltonian, boundary)
            if matvec(hamiltonian, inverse_column) != boundary:
                raise ArithmeticError("tilted-field LDL resolvent residual failed")
            green = -inverse_column[0]
            phase = cayley_phase(green)
            if phase.norm_squared() != 1:
                raise ArithmeticError("tilted-field Cayley phase lost unit modulus")
            seal_payload = {
                "qubits": qubits,
                "dimension": dimension,
                "coefficient_bound_J": fraction_json(coefficient_bound),
                "diagonal_offset_D": fraction_json(diagonal_offset),
                "green": fraction_json(green),
                "cayley_s": phase.json(),
                "hamiltonian_determinant": fraction_json(determinant),
                "krylov_rank_certified_exact": certificate[
                    "krylov_rank_certified_exact"
                ],
                "krylov_rank_by_prime": certificate["krylov_rank_by_prime"],
            }
            record.update(
                {
                    "green": fraction_json(green),
                    "cayley_s": phase.json(),
                    "cayley_unit_modulus_exact": True,
                    "hamiltonian_determinant": fraction_json(determinant),
                    "resolvent_seal_sha256": canonical_sha256(seal_payload),
                    "exact_resolvent_materialized": True,
                    "exact_resolvent_policy": "DENSE_EXACT_MATERIALIZATION_N_LE_4",
                }
            )
        else:
            record.update(
                {
                    "green": None,
                    "cayley_s": None,
                    "cayley_unit_modulus_exact": None,
                    "hamiltonian_determinant": None,
                    "resolvent_seal_sha256": None,
                    "exact_resolvent_materialized": False,
                    "exact_resolvent_policy": "NOT_MATERIALIZED_ABOVE_N_4_PACKAGE_CEILING",
                }
            )
        records.append(record)
    return records


def finite_bandwidth_control() -> dict[str, object]:
    hamiltonian = tridiagonal_matrix([Fraction(3), Fraction(3)], [Fraction(1)])
    energies = [Fraction(0), Fraction(1, 2)]
    greens = [boundary_green_solve(hamiltonian, energy) for energy in energies]
    phases = [cayley_phase(green) for green in greens]
    coherent_overlap = (phases[0] + phases[1]) * GaussianQ.make(Fraction(1, 2))
    mode_fidelity = coherent_overlap.norm_squared()
    if mode_fidelity >= 1 or any(phase.norm_squared() != 1 for phase in phases):
        raise ArithmeticError("two-energy finite-bandwidth control failed")
    return {
        "hamiltonian": "H2_DIAGONAL_3_OFFDIAGONAL_1",
        "energies": [fraction_json(value) for value in energies],
        "greens": [fraction_json(value) for value in greens],
        "cayley_phases": [value.json() for value in phases],
        "equal_weight_coherent_phase_overlap": coherent_overlap.json(),
        "mode_fidelity": fraction_json(mode_fidelity),
        "same_spectral_mode_probability": fraction_json(mode_fidelity),
        "orthogonal_spectral_distortion_probability": fraction_json(
            Fraction(1) - mode_fidelity
        ),
        "strictly_less_than_one": True,
        "interpretation": "ENERGY_DEPENDENT_UNIT_PHASES_DISTORT_A_FINITE_BANDWIDTH_MODE",
    }


def bethe_factorized_control() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    imaginary_coupling = GaussianQ.make(0, 1)
    for particles in BETHE_PARTICLE_COUNTS:
        rapidities = [index * (index + 1) for index in range(particles)]
        phase = QI_ONE
        factors = []
        for left in range(particles):
            for right in range(left + 1, particles):
                difference = rapidities[left] - rapidities[right]
                factor = (
                    GaussianQ.make(difference) + imaginary_coupling
                ) / (
                    GaussianQ.make(difference) - imaginary_coupling
                )
                if factor.norm_squared() != 1:
                    raise ArithmeticError("Bethe pair phase lost unit modulus")
                phase = phase * factor
                factors.append(
                    {
                        "left": left,
                        "right": right,
                        "rapidity_difference": difference,
                        "factor": factor.json(),
                    }
                )
        if phase.norm_squared() != 1:
            raise ArithmeticError("factorized Bethe product lost unit modulus")
        records.append(
            {
                "M": particles,
                "m": particles,
                "c": 1,
                "rapidities": [fraction_json(Fraction(value)) for value in rapidities],
                "pair_count": len(factors),
                "pair_factors": factors,
                "factorized_phase_product": phase.json(),
                "factorized_phase": phase.json(),
                "inverse": phase.conjugate().json(),
                "exact_inverse_product_one": phase * phase.conjugate() == QI_ONE,
                "factorized_phase_unit_modulus_exact": True,
                "rapidity_descriptor_size": len(rapidities),
                "classical_pair_product_work": len(factors),
                "collapse": "PUBLIC_RAPIDITY_LIST_AND_PAIRWISE_PHASE_PRODUCT",
            }
        )
    return records


def stationary_reuse_semantics() -> dict[str, object]:
    hamiltonian = tridiagonal_matrix([Fraction(3), Fraction(3)], [Fraction(1)])
    energies = [Fraction(0), Fraction(1, 2)]
    transactions = []
    for query, energy in enumerate(energies, start=1):
        green = boundary_green_solve(hamiltonian, energy)
        phase = cayley_phase(green)
        transactions.append(
            {
                "query": query,
                "energy": fraction_json(energy),
                "green": fraction_json(green),
                "phase": phase.json(),
                "formal_stationary_target_input": "GROUND",
                "formal_stationary_target_output": "GROUND",
            }
        )
    return {
        "transactions": transactions,
        "distinct_energy": energies[0] != energies[1],
        "second_query_uses_no_new_target_descriptor": True,
        "classification": "STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY",
        "formal_stationary_one_channel_boundary_stipulated": True,
        "formal_basis": "EXACT_UNIT_MODULUS_CAYLEY_S_FOR_REAL_SUBGAP_BOUNDARY_RESOLVENT",
        "time_domain_scattering_executed": False,
        "finite_time_target_restoration_executed": False,
        "distinct_probe_reuse_executed": False,
        "same_backing_established": False,
        "physical_restoration_established": False,
        "physical_ground_state_prepared": False,
        "physical_source_separation_executed": False,
        "scope": "FORMAL_STATIONARY_ONE_CHANNEL_UNIT_MODULUS_ONLY",
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_result() -> dict[str, object]:
    for prime in SPLIT_PRIMES:
        if prime % 4 != 1 or not is_prime(prime):
            raise ArithmeticError("declared modular rank field is not a split prime")

    modes = exact_mode_fixtures()
    non_unit_kappa = non_unit_kappa_control()
    paths = path_family()
    flagged = interacting_flagged_blocks()
    neumann = fixed_margin_neumann()
    bandwidth = finite_bandwidth_control()
    bethe = bethe_factorized_control()
    reuse = stationary_reuse_semantics()
    source = Path(__file__).resolve()
    return {
        "metadata": {
            "schema": "PHASE_QEMU_V5_SUBGAP_BOUNDARY_RESOLVENT_SEPARATE_REFERENCE_V1",
            "milestone": "M263",
            "arithmetic": "STDLIB_FRACTION_AND_INDEPENDENT_Q_I",
            "production_imported": False,
            "split_primes": list(SPLIT_PRIMES),
            "source_sha256": sha256_file(source),
            "claim": CLAIM,
            "claim_ceiling": CLAIM_CEILING,
            "disposition": DISPOSITION,
            "next_mechanism": NEXT_MECHANISM,
            "scientific_verification": "SEPARATE_REFERENCE_PARITY",
            "theorem_verification": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource_verification": "PACKAGE_SELF_REVIEW",
        },
        "verification_scope": {
            "scientific": "SEPARATE_REFERENCE_PARITY",
            "formal_derivation": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource_accounting": "PACKAGE_SELF_REVIEW",
        },
        "exact_mode_fixtures": modes,
        "non_unit_kappa_control": non_unit_kappa,
        "path_family": paths,
        "interacting_flagged_blocks": flagged,
        "fixed_margin_neumann": neumann,
        "finite_bandwidth_control": bandwidth,
        "bethe_factorized_control": bethe,
        "stationary_reuse_semantics": reuse,
        "restoration": {
            "classification": "NO_RESTORATION_CLAIM",
            "scope": "STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY",
            "executed_restoration": False,
            "time_domain_restoration_executed": False,
            "distinct_probe_reuse_executed": False,
            "same_backing_established": False,
            "physical_restoration_established": False,
        },
        "resource_accounting": {
            "boundary_krylov_quotient_is_controlling_exact_representation": True,
            "full_hilbert_dimension_is_not_automatically_observable_rank": True,
            "fixed_margin_effective_depth_is_precision_and_latency_dependent": True,
            "integrability_comparator_included": True,
            "classical_advantage_established": False,
            "physical_resource_advantage_established": False,
            "m257_escape_established": False,
            "unbounded_compute_established": False,
            "replace_the_bit_with_pi_established": False,
        },
        "assertions": {
            "status": "SOURCE_SELF_CHECK_PASS",
            "all_mode_fixture_parity": all(item["parity"] for item in modes),
            "non_unit_kappa_linear_recurrence": non_unit_kappa[
                "linear_kappa_recurrence_exposed"
            ],
            "path_krylov_rank_grows_with_n": [
                item["krylov_rank_certified_exact"] for item in paths
            ] == list(PATH_SIZES),
            "all_interacting_blocks_have_full_exact_and_modular_krylov_rank": all(
                item["krylov_rank_certified_exact"] == item["dimension"]
                and item["all_split_prime_ranks_match_exact"]
                for item in flagged
            ),
            "fixed_margin_tail_is_below_epsilon": (
                Fraction(
                    neumann["phase_tail_bound"]["numerator"],
                    neumann["phase_tail_bound"]["denominator"],
                )
                <= Fraction(1, 2**20)
            ),
            "two_energy_wavepacket_is_not_one_scalar_mode": bandwidth[
                "strictly_less_than_one"
            ],
            "bethe_control_is_explicit_pair_product": all(
                item["pair_count"] == item["M"] * (item["M"] - 1) // 2
                for item in bethe
            ),
            "stationary_return_is_not_executed_restoration": True,
            "no_physical_or_resource_claim": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args()
    result = build_result()
    if arguments.compact:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
