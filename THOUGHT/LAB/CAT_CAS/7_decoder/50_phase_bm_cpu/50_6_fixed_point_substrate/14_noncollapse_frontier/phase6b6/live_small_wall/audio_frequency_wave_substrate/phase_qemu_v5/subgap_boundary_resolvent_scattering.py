#!/usr/bin/env python3
"""Exact stationary subgap boundary-resolvent scattering diagnostic (M263).

This deterministic software model uses only exact rational and Gaussian-
rational arithmetic.  It implements a stipulated one-channel K-matrix
boundary law and an executed compact path-only Krylov/continued-fraction
shadow.  It does not execute time-domain scattering, QEMU, a physical
carrier, or restoration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


CLAIM = (
    "EXACT_RATIONAL_SINGLE_CHANNEL_SUBGAP_CAYLEY_RESOLVENT_DIAGNOSTIC_"
    "IMPLEMENTS_A_STIPULATED_FORMAL_STATIONARY_UNIT_MODULUS_BOUNDARY_LAW_AT_"
    "DECLARED_FIXTURES_WITH_DISTINCT_ENERGY_DESCRIPTOR_REUSE_GROWING_EXACT_"
    "KRYLOV_RANK_AND_PATH_ONLY_FIXED_MARGIN_EFFECTIVE_DEPTH_BOUND_PLUS_"
    "TILTED_FIELD_AND_BETHE_FACTORIZATION_CONTROLS"
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
EPSILON = Fraction(1, 1 << 20)
SPLIT_PRIMES = (65_537, 998_244_353)


def _f(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class GaussianRational:
    real: Fraction
    imag: Fraction = Fraction(0)

    @staticmethod
    def scalar(value: int | Fraction) -> "GaussianRational":
        return GaussianRational(_f(value), Fraction(0))

    def __add__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        rhs = coerce_gaussian(other)
        return GaussianRational(self.real + rhs.real, self.imag + rhs.imag)

    def __radd__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        return self + other

    def __sub__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        rhs = coerce_gaussian(other)
        return GaussianRational(self.real - rhs.real, self.imag - rhs.imag)

    def __rsub__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        return coerce_gaussian(other) - self

    def __neg__(self) -> "GaussianRational":
        return GaussianRational(-self.real, -self.imag)

    def __mul__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        rhs = coerce_gaussian(other)
        return GaussianRational(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )

    def __rmul__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        return self * other

    def __truediv__(
        self, other: "GaussianRational" | int | Fraction
    ) -> "GaussianRational":
        rhs = coerce_gaussian(other)
        denominator = rhs.norm_squared()
        if denominator == 0:
            raise ZeroDivisionError("division by zero in Q(i)")
        numerator = self * rhs.conjugate()
        return GaussianRational(
            numerator.real / denominator, numerator.imag / denominator
        )

    def conjugate(self) -> "GaussianRational":
        return GaussianRational(self.real, -self.imag)

    def norm_squared(self) -> Fraction:
        return self.real * self.real + self.imag * self.imag


def coerce_gaussian(
    value: GaussianRational | int | Fraction,
) -> GaussianRational:
    return value if isinstance(value, GaussianRational) else GaussianRational.scalar(value)


GR_ONE = GaussianRational.scalar(1)
GR_I = GaussianRational(Fraction(0), Fraction(1))


def fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def gaussian_json(value: GaussianRational) -> dict[str, object]:
    return {
        "field": "Q_I",
        "real": fraction_json(value.real),
        "imag": fraction_json(value.imag),
        "norm_squared": fraction_json(value.norm_squared()),
    }


def cayley_s(green: Fraction, kappa: Fraction = Fraction(1)) -> GaussianRational:
    k_value = kappa * green
    return (GR_ONE - GR_I * k_value) / (GR_ONE + GR_I * k_value)


def zero_matrix(size: int) -> list[list[Fraction]]:
    return [[Fraction(0) for _ in range(size)] for _ in range(size)]


def solve_fraction_matrix(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> list[Fraction]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or len(rhs) != size:
        raise ValueError("fraction solve requires one nonempty square system")
    work = [list(row) + [rhs[row_index]] for row_index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ArithmeticError("singular exact matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[column])
                ]
    return [work[row][size] for row in range(size)]


def boundary_green(
    hamiltonian: Sequence[Sequence[Fraction]],
    energy: Fraction,
    chi_index: int = 0,
) -> Fraction:
    size = len(hamiltonian)
    shifted = zero_matrix(size)
    for row in range(size):
        for column in range(size):
            shifted[row][column] = (
                (energy if row == column else Fraction(0))
                - hamiltonian[row][column]
            )
    rhs = [Fraction(1 if row == chi_index else 0) for row in range(size)]
    solution = solve_fraction_matrix(shifted, rhs)
    return solution[chi_index]


def matrix_vector(
    matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]
) -> list[Fraction]:
    return [sum((value * vector[column] for column, value in enumerate(row)), Fraction(0)) for row in matrix]


def exact_krylov_rank(
    matrix: Sequence[Sequence[Fraction]], chi_index: int = 0
) -> int:
    size = len(matrix)
    basis: dict[int, list[Fraction]] = {}
    vector = [Fraction(1 if row == chi_index else 0) for row in range(size)]
    for _ in range(size):
        reduced = list(vector)
        for pivot in sorted(basis):
            factor = reduced[pivot]
            if factor:
                reduced = [
                    value - factor * basis_value
                    for value, basis_value in zip(reduced, basis[pivot])
                ]
        new_pivot = next((index for index, value in enumerate(reduced) if value), None)
        if new_pivot is not None:
            scale = reduced[new_pivot]
            basis[new_pivot] = [value / scale for value in reduced]
        vector = matrix_vector(matrix, vector)
    return len(basis)


def fraction_mod(value: Fraction, prime: int) -> int:
    denominator = value.denominator % prime
    if denominator == 0:
        raise ArithmeticError("selected modular prime divides a denominator")
    return (value.numerator % prime) * pow(denominator, prime - 2, prime) % prime


def modular_krylov_rank(
    matrix: Sequence[Sequence[Fraction]], prime: int, chi_index: int = 0
) -> int:
    size = len(matrix)
    modular = [[fraction_mod(value, prime) for value in row] for row in matrix]
    basis: dict[int, list[int]] = {}
    vector = [1 if row == chi_index else 0 for row in range(size)]
    for _ in range(size):
        reduced = list(vector)
        for pivot in sorted(basis):
            factor = reduced[pivot]
            if factor:
                reduced = [
                    (value - factor * basis_value) % prime
                    for value, basis_value in zip(reduced, basis[pivot])
                ]
        new_pivot = next((index for index, value in enumerate(reduced) if value), None)
        if new_pivot is not None:
            inverse = pow(reduced[new_pivot], prime - 2, prime)
            basis[new_pivot] = [(value * inverse) % prime for value in reduced]
        vector = [
            sum(row[column] * vector[column] for column in range(size)) % prime
            for row in modular
        ]
    return len(basis)


def nonzero_entries(matrix: Sequence[Sequence[Fraction]]) -> int:
    return sum(1 for row in matrix for value in row if value)


def coefficient_payload_bits(matrix: Sequence[Sequence[Fraction]]) -> int:
    return sum(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        for row in matrix
        for value in row
        if value
    )


def exact_mode_fixture(
    name: str, hamiltonian: list[list[Fraction]], expected_green: Fraction
) -> dict[str, object]:
    energy = Fraction(0)
    green = boundary_green(hamiltonian, energy)
    if green != expected_green:
        raise AssertionError(f"{name}: exact Green fixture mismatch")
    phase = cayley_s(green)
    if phase.norm_squared() != 1:
        raise AssertionError(f"{name}: Cayley phase is not unit modulus")
    return {
        "name": name,
        "dimension": len(hamiltonian),
        "energy": fraction_json(energy),
        "coupling_vector": "E1",
        "green": fraction_json(green),
        "cayley_s": gaussian_json(phase),
        "cayley_inverse": gaussian_json(phase.conjugate()),
        "inverse_product_exact_one": phase * phase.conjugate() == GR_ONE,
        "matrix_nonzero_entries": nonzero_entries(hamiltonian),
        "coefficient_payload_bits": coefficient_payload_bits(hamiltonian),
    }


def nonunit_kappa_control() -> dict[str, object]:
    green = Fraction(-1, 3)
    kappa = Fraction(2)
    phase = cayley_s(green, kappa)
    expected = GaussianRational(Fraction(5, 13), Fraction(12, 13))
    if phase != expected or phase.norm_squared() != 1:
        raise AssertionError("non-unit-kappa Cayley control changed")
    return {
        "name": "ONE_MODE_NONUNIT_KAPPA",
        "green": fraction_json(green),
        "kappa": fraction_json(kappa),
        "linear_kappa_times_green": fraction_json(kappa * green),
        "cayley_s": gaussian_json(phase),
        "expected_cayley_s": gaussian_json(expected),
        "linear_kappa_green_rule_exact": True,
    }


def path_hamiltonian(size: int) -> list[list[Fraction]]:
    matrix = zero_matrix(size)
    for index in range(size):
        matrix[index][index] = Fraction(4)
        if index + 1 < size:
            matrix[index][index + 1] = Fraction(1)
            matrix[index + 1][index] = Fraction(1)
    return matrix


def path_continuants(size: int) -> list[int]:
    values = [1, 4]
    for _ in range(2, size + 1):
        values.append(4 * values[-1] - values[-2])
    return values[: size + 1]


def path_lanczos_green(size: int) -> Fraction:
    denominator = Fraction(4)
    for _ in range(size - 1):
        denominator = Fraction(4) - Fraction(1, 1) / denominator
    return -Fraction(1, 1) / denominator


def path_family_evidence() -> list[dict[str, object]]:
    results = []
    for size in (2, 4, 8, 16, 32):
        matrix = path_hamiltonian(size)
        continuants = path_continuants(size)
        continuant_green = -Fraction(continuants[size - 1], continuants[size])
        lanczos_green = path_lanczos_green(size)
        dense_green = boundary_green(matrix, Fraction(0))
        ranks = {
            str(prime): modular_krylov_rank(matrix, prime) for prime in SPLIT_PRIMES
        }
        if dense_green != continuant_green or dense_green != lanczos_green:
            raise AssertionError("path continuant/Lanczos/dense parity failed")
        if any(rank != size for rank in ranks.values()):
            raise AssertionError("path modular Krylov rank did not grow with size")
        results.append(
            {
                "n": size,
                "dimension": size,
                "energy": fraction_json(Fraction(0)),
                "green": fraction_json(dense_green),
                "continuant_green": fraction_json(continuant_green),
                "lanczos_continued_fraction_green": fraction_json(lanczos_green),
                "continuant_lanczos_dense_parity": True,
                "krylov_rank_by_prime": ranks,
                "krylov_rank_certified_exact": size,
                "rank_certificate_law": "FULL_RANK_NONZERO_MINOR_MOD_EACH_GOOD_PRIME_IMPLIES_FULL_RANK_OVER_Q",
                "cayley_s": gaussian_json(cayley_s(dense_green)),
                "matrix_nonzero_entries": nonzero_entries(matrix),
                "exact_comparator": "TRIDIAGONAL_CONTINUANT_OR_SCALAR_LANCZOS_IN_O_N_ARITHMETIC",
            }
        )
    return results


def tilted_field_block(size: int) -> tuple[list[list[Fraction]], Fraction, Fraction]:
    dimension = 1 << size
    matrix = zero_matrix(dimension)
    fields = [Fraction((site + 1) ** 2, size + 2) for site in range(size)]
    interaction_bound = Fraction(size) + Fraction(size - 1, 2) + sum(fields, Fraction(0))
    offset = 2 * interaction_bound + 1
    for basis in range(dimension):
        spins = [1 if ((basis >> site) & 1) == 0 else -1 for site in range(size)]
        diagonal = sum(
            (fields[site] * spins[site] for site in range(size)), Fraction(0)
        )
        diagonal += sum(
            (Fraction(1, 2) * spins[site] * spins[site + 1] for site in range(size - 1)),
            Fraction(0),
        )
        matrix[basis][basis] = offset + diagonal
        for site in range(size):
            matrix[basis][basis ^ (1 << site)] += 1
    return matrix, interaction_bound, offset


def effective_neumann_depth(
    operator_bound: Fraction,
    denominator: Fraction,
    epsilon: Fraction,
    kappa: Fraction = Fraction(1),
    chi_norm_squared: Fraction = Fraction(1),
) -> tuple[int, Fraction]:
    q = operator_bound / denominator
    if not 0 <= q < 1:
        raise ValueError("Neumann bound requires 0 <= q < 1")
    order = 0
    while True:
        phase_tail_bound = (
            2
            * kappa
            * chi_norm_squared
            / denominator
            * q ** (order + 1)
            / (1 - q)
        )
        if phase_tail_bound <= epsilon:
            return order, phase_tail_bound
        order += 1


def interacting_block_evidence() -> list[dict[str, object]]:
    results = []
    for size in range(2, 7):
        matrix, operator_bound, offset = tilted_field_block(size)
        dimension = 1 << size
        ranks = {
            str(prime): modular_krylov_rank(matrix, prime) for prime in SPLIT_PRIMES
        }
        if any(rank != dimension for rank in ranks.values()):
            raise AssertionError("interacting split-prime Krylov certificate not full")
        exact_rank = exact_krylov_rank(matrix) if size <= 4 else dimension
        rank_basis = "DIRECT_FRACTION_ELIMINATION" if size <= 4 else "CERTIFIED_BY_FULL_MODULAR_MINOR"
        if exact_rank != dimension:
            raise AssertionError("interacting exact Krylov rank not full")
        exact_green = boundary_green(matrix, Fraction(0)) if size <= 4 else None
        entry: dict[str, object] = {
            "n": size,
            "dimension": dimension,
            "carrier_preparation": "PRODUCT_VACUUM_FLAG_ZERO_WORK_ALL_ZERO",
            "virtual_block": "FLAG_ONE_RATIONAL_TILTED_FIELD_ISING",
            "interaction_law": "SUM_X_PLUS_ONE_HALF_SUM_ZZ_PLUS_SUM_SITE_SQUARED_OVER_N_PLUS_2_Z",
            "energy": fraction_json(Fraction(0)),
            "operator_norm_upper_bound_j": fraction_json(operator_bound),
            "positive_offset_d": fraction_json(offset),
            "q_upper_bound": fraction_json(operator_bound / offset),
            "krylov_rank_by_prime": ranks,
            "krylov_rank_certified_exact": exact_rank,
            "exact_rank_basis": rank_basis,
            "exact_resolvent_policy": "DENSE_FRACTION_SOLVE_N_LE_4_OTHERWISE_NOT_MATERIALIZED",
            "neumann_degree_bound_applied": False,
            "approximation_work_claim": "NONE",
            "matrix_nonzero_entries": nonzero_entries(matrix),
            "coefficient_payload_bits": coefficient_payload_bits(matrix),
            "non_gaussian_scope": "TILTED_AND_INHOMOGENEOUS_INTERACTING_CONTROL_NOT_A_HARDNESS_PROOF",
        }
        if exact_green is not None:
            entry["green"] = fraction_json(exact_green)
            entry["cayley_s"] = gaussian_json(cayley_s(exact_green))
            entry["exact_resolvent_materialized"] = True
        else:
            entry["green"] = None
            entry["cayley_s"] = None
            entry["exact_resolvent_materialized"] = False
        results.append(entry)
    return results


def fixed_margin_neumann_evidence() -> dict[str, object]:
    operator_bound = Fraction(2)
    denominator = Fraction(4)
    order, tail = effective_neumann_depth(operator_bound, denominator, EPSILON)
    if order != 19 or tail != EPSILON:
        raise AssertionError("canonical dyadic Neumann depth changed")
    return {
        "family": "H_N_EQUALS_4I_PLUS_PATH_ADJACENCY",
        "declared_uniform_operator_bound_j": fraction_json(operator_bound),
        "denominator_d": fraction_json(denominator),
        "q": fraction_json(Fraction(1, 2)),
        "epsilon": fraction_json(EPSILON),
        "smallest_truncation_order_k": order,
        "retained_moment_count": order + 1,
        "phase_tail_bound": fraction_json(tail),
        "law": "ABS_S_MINUS_S_K_LE_2_KAPPA_NORMCHI2_OVER_D_TIMES_Q_POW_K_PLUS_1_OVER_1_MINUS_Q",
        "bound_scope": "PATH_FAMILY_DEGREE_UPPER_BOUND_ONLY",
        "moment_generation_work_counted": False,
        "approximation_lower_bound_established": False,
        "interpretation": "PATH_HAS_AN_EXECUTED_COMPACT_O_N_STREAMED_CONTINUANT_SHADOW_DESPITE_GROWING_EXACT_KRYLOV_RANK",
    }


def finite_bandwidth_evidence() -> dict[str, object]:
    matrix = [
        [Fraction(3), Fraction(1)],
        [Fraction(1), Fraction(3)],
    ]
    energies = (Fraction(0), Fraction(1, 2))
    phases = [cayley_s(boundary_green(matrix, energy)) for energy in energies]
    expected_second = GaussianRational(Fraction(341, 541), Fraction(420, 541))
    if phases[1] != expected_second:
        raise AssertionError("finite-bandwidth second-energy fixture changed")
    mean_phase = (phases[0] + phases[1]) * Fraction(1, 2)
    preservation = mean_phase.norm_squared()
    distortion = 1 - preservation
    if not 0 < distortion < 1:
        raise AssertionError("two-bin finite-bandwidth distortion must be strict")
    return {
        "target": "COUPLED_TWO_MODE_FIXTURE",
        "input_mode": "EQUAL_COHERENT_TWO_ENERGY_BIN_SUPERPOSITION",
        "energies": [fraction_json(value) for value in energies],
        "cayley_phases": [gaussian_json(value) for value in phases],
        "same_spectral_mode_amplitude": gaussian_json(mean_phase),
        "same_spectral_mode_probability": fraction_json(preservation),
        "orthogonal_spectral_distortion_probability": fraction_json(distortion),
        "decision": "NONZERO_ENERGY_DEPENDENCE_PREVENTS_EXACT_UNCHANGED_PROBE_MODE_TIMES_ONE_GLOBAL_PHASE",
    }


def bethe_pair_phase(rapidities: Sequence[Fraction]) -> GaussianRational:
    result = GR_ONE
    for left in range(len(rapidities)):
        for right in range(left + 1, len(rapidities)):
            difference = rapidities[left] - rapidities[right]
            result *= (GaussianRational.scalar(difference) + GR_I) / (
                GaussianRational.scalar(difference) - GR_I
            )
    return result


def bethe_evidence() -> list[dict[str, object]]:
    results = []
    for count in (2, 4, 8):
        rapidities = [Fraction(index * (index + 1)) for index in range(count)]
        phase = bethe_pair_phase(rapidities)
        if phase.norm_squared() != 1:
            raise AssertionError("Bethe pair product lost unit modulus")
        results.append(
            {
                "m": count,
                "rapidities": [fraction_json(value) for value in rapidities],
                "pair_count": count * (count - 1) // 2,
                "factorized_phase": gaussian_json(phase),
                "inverse": gaussian_json(phase.conjugate()),
                "exact_inverse_product_one": phase * phase.conjugate() == GR_ONE,
                "classical_forward_shadow": "PUBLIC_RAPIDITY_PAIR_PRODUCT_IN_O_M_SQUARED_EXACT_FIELD_OPERATIONS",
                "resource_disposition": "KILLED_BY_PREPARATION_ROOT_DESCRIPTOR_AND_REFERENCE_COMPLETE_FACTORIZATION",
            }
        )
    return results


def stationary_reuse_evidence() -> dict[str, object]:
    matrix = [
        [Fraction(3), Fraction(1)],
        [Fraction(1), Fraction(3)],
    ]
    energies = (Fraction(0), Fraction(1, 2))
    transactions = []
    for query_index, energy in enumerate(energies, start=1):
        green = boundary_green(matrix, energy)
        phase = cayley_s(green)
        transactions.append(
            {
                "query": query_index,
                "energy": fraction_json(energy),
                "green": fraction_json(green),
                "phase": gaussian_json(phase),
                "formal_stationary_target_input": "GROUND",
                "formal_stationary_target_output": "GROUND",
            }
        )
    return {
        "transactions": transactions,
        "distinct_energy": energies[0] != energies[1],
        "second_query_uses_no_new_target_descriptor": True,
        "classification": "STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "same_backing_established": False,
        "executed_boundary_law": False,
        "executed_time_domain_restoration": False,
        "physical_restoration_established": False,
    }


def build_evidence() -> dict[str, object]:
    fixtures = [
        exact_mode_fixture("ONE_MODE", [[Fraction(3)]], Fraction(-1, 3)),
        exact_mode_fixture(
            "COUPLED_TWO_MODE",
            [[Fraction(3), Fraction(1)], [Fraction(1), Fraction(3)]],
            Fraction(-3, 8),
        ),
        exact_mode_fixture(
            "TRIDIAGONAL_THREE_MODE",
            [
                [Fraction(4), Fraction(1), Fraction(0)],
                [Fraction(1), Fraction(4), Fraction(1)],
                [Fraction(0), Fraction(1), Fraction(4)],
            ],
            Fraction(-15, 56),
        ),
    ]
    evidence: dict[str, object] = {
        "metadata": {
            "schema": "PHASE_QEMU_V5_SUBGAP_RESOLVENT_EVIDENCE_V1",
            "milestone": "M263",
            "backend": "DETERMINISTIC_EXACT_SOFTWARE_HARDWARE_MODEL",
            "arithmetic": "Q_AND_Q_I_ONLY",
            "claim": CLAIM,
            "claim_ceiling": CLAIM_CEILING,
            "disposition": DISPOSITION,
            "next_mechanism": NEXT_MECHANISM,
            "m257_guardrail": "PRESERVED_EQUAL_ACCESS_FORWARD_ONLY_SOFTWARE_SHADOW_REMAINS_CONTROLLING",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "verification_scope": {
            "science": "SEPARATE_REFERENCE_PARITY",
            "theory": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource": "PACKAGE_SELF_REVIEW",
        },
        "exact_mode_fixtures": fixtures,
        "nonunit_kappa_control": nonunit_kappa_control(),
        "path_family": path_family_evidence(),
        "interacting_flagged_blocks": interacting_block_evidence(),
        "fixed_margin_neumann": fixed_margin_neumann_evidence(),
        "finite_bandwidth_control": finite_bandwidth_evidence(),
        "bethe_factorized_control": bethe_evidence(),
        "stationary_reuse_semantics": stationary_reuse_evidence(),
        "restoration": {
            "classification": "NO_RESTORATION_CLAIM",
            "scope": "STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY",
            "executed_boundary_law": False,
            "executed_time_domain_restoration": False,
            "executed_inverse_or_echo": False,
            "same_backing_established": False,
            "physical_restoration_established": False,
        },
        "resource_accounting": {
            "classification": "PACKAGE_SELF_REVIEW",
            "exact_dense_resolvent_limit": "INTERACTING_N_LE_4",
            "modular_rank_primes": list(SPLIT_PRIMES),
            "coefficient_payload_bits_scope": "INPUT_MATERIALIZED_NONZERO_MATRIX_COEFFICIENT_PAYLOAD_ONLY_NOT_DESCRIPTOR_COPIES_INTERMEDIATES_OR_OUTPUTS",
            "counted": [
                "HILBERT_DIMENSION",
                "MATRIX_NONZERO_ENTRIES",
                "COEFFICIENT_PAYLOAD_BITS",
                "EXACT_KRYLOV_RANK",
                "PAIR_PRODUCT_COUNT",
                "NEUMANN_EFFECTIVE_DEPTH_AT_DECLARED_EPSILON",
            ],
            "not_instrumented": [
                "WHOLE_PROCESS_MEMORY",
                "WHOLE_PROCESS_LIVENESS",
                "PYTHON_ALLOCATOR_STATE",
                "SERIALIZATION_OVERHEAD",
                "INPUT_EXACT_PAYLOAD_HEIGHT_BEYOND_REPORTED_NONZERO_COEFFICIENT_SUM",
                "INTERMEDIATE_EXACT_PAYLOAD_HEIGHT",
                "OUTPUT_EXACT_PAYLOAD_HEIGHT",
                "EXACT_ARITHMETIC_OPERATION_WORK",
                "FORMULA_DESCRIPTOR_SIZE_AND_DESCRIPTOR_COPIES",
                "RETAINED_STATE_AND_RETAINED_HISTORY",
                "REMATERIALIZATION_WORK",
                "CONTROLLER_STATE_AND_DETECTOR_STATE",
                "PER_QUERY_AND_TOTAL_QUERY_WORK",
                "PRECISION_AND_SHOT_COUNT",
                "TIME_DOMAIN_DWELL_OR_WIGNER_DELAY",
                "FINITE_PULSE_DURATION",
                "PHYSICAL_ENERGY_BANDWIDTH_LOSS_NOISE_OR_CALIBRATION",
                "TARGET_PREPARATION_OR_RECONFIGURATION_ENERGY",
            ],
            "strongest_honest_comparators": [
                "EXACT_SCALAR_LANCZOS_CONTINUED_FRACTION",
                "FRACTION_FREE_KRYLOV_RECURRENCE",
                "TRUNCATED_NEUMANN_OR_CHEBYSHEV_MOMENTS",
                "SPARSE_SHIFTED_LINEAR_SOLVER",
                "MPS_CORRECTION_VECTOR_OR_TENSOR_NETWORK_CONTRACTION",
                "INTEGRABLE_BETHE_TQ_OR_PAIR_PRODUCT",
                "FIXED_TARGET_FIXED_ENERGY_SCALAR_RESPONSE_TABLE",
                "EQUAL_ACCESS_FORWARD_ONLY_SOFTWARE_RESOLVENT_SHADOW",
            ],
        },
        "scope_exclusions": [
            "NO_QEMU_DEVICE_EXECUTION",
            "NO_TIME_DOMAIN_SCATTERING",
            "NO_EXECUTED_INVERSE_OR_ECHO",
            "NO_PHYSICAL_CARRIER_OR_OBSERVATION",
            "NO_SAME_BACKING_CUSTODY",
            "NO_PHYSICAL_OR_EXACT_FINITE_TIME_RESTORATION",
            "NO_ADVANTAGE_OR_M257_ESCAPE",
            "NO_SMALL_WALL_CROSSING_OR_UNBOUNDED_COMPUTE",
            "NO_REPLACE_THE_BIT_WITH_PI_CLAIM",
        ],
    }
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = hashlib.sha256(canonical).hexdigest()
    evidence["assertions"] = {
        "status": "SOURCE_SELF_CHECK_PASS",
        "fixtures_exact": True,
        "nonunit_kappa_linear_rule_exact": True,
        "path_continuant_lanczos_parity": True,
        "path_krylov_rank_grows": True,
        "interacting_split_prime_full_ranks": True,
        "path_fixed_margin_degree_bound_checked": True,
        "general_approximation_lower_bound_claimed": False,
        "finite_bandwidth_distortion_nonzero": True,
        "bethe_factorization_resource_killed": True,
        "terminal": False,
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact", action="store_true", help="emit canonical compact JSON"
    )
    arguments = parser.parse_args()
    evidence = build_evidence()
    if arguments.compact:
        print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(evidence, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
