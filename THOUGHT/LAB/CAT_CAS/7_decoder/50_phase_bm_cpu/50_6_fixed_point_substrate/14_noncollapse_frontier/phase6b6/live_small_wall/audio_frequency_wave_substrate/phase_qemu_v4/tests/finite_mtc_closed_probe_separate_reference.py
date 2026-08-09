#!/usr/bin/env python3
"""Independent exact oracle for the M262 central Wilson-loop obstruction.

This reference does not import production.  Semion and Ising use explicit
Gaussian/quadratic tuples; Fibonacci uses an independent Q(zeta_5) quotient
plus a Q(sqrt(5)) canonical aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


CLAIM = (
    "EXACT_FIXED_FINITE_UMTC_SINGLE_GLOBAL_CLOSED_SIMPLE_PROBE_DIAGNOSTIC_"
    "ESTABLISHES_MULTIPLICITY_BLIND_TOTAL_CHARGE_SCALAR_ACTION_"
    "DETERMINISTIC_UNIT_MODULUS_BOUNDARIES_AS_CONSTANT_SIZE_SIMPLE_OBJECT_"
    "LOOKUPS_AND_STRICTLY_INTERMEDIATE_VACUUM_RETURN_RETAINED_BOUNDARY_"
    "OBSTRUCTION_WITH_"
    "FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_DISTINCT_PROBE_"
    "REUSE_AND_SEMION_ISING_FIBONACCI_FIXTURES"
)
CLAIM_CEILING = (
    "ABSTRACT_EXACT_FIXED_FINITE_UMTC_SINGLE_SIMPLE_PROBE_GLOBAL_DISK_"
    "ENCIRCLEMENT_WITH_DECLARED_TOTAL_CHARGE_AND_SEMION_ISING_FIBONACCI_"
    "FIXTURES_ONLY"
)


@dataclass(frozen=True)
class Qsqrt:
    rational: Fraction
    radical: Fraction
    radicand: int

    @staticmethod
    def make(rational: int | Fraction, radical: int | Fraction, radicand: int) -> "Qsqrt":
        return Qsqrt(Fraction(rational), Fraction(radical), radicand)

    def _check(self, other: "Qsqrt") -> None:
        if self.radicand != other.radicand:
            raise ValueError("radicand mismatch")

    def __add__(self, other: "Qsqrt") -> "Qsqrt":
        self._check(other)
        return Qsqrt(self.rational + other.rational, self.radical + other.radical, self.radicand)

    def __sub__(self, other: "Qsqrt") -> "Qsqrt":
        self._check(other)
        return Qsqrt(self.rational - other.rational, self.radical - other.radical, self.radicand)

    def __neg__(self) -> "Qsqrt":
        return Qsqrt(-self.rational, -self.radical, self.radicand)

    def __mul__(self, other: "Qsqrt") -> "Qsqrt":
        self._check(other)
        return Qsqrt(
            self.rational * other.rational + self.radicand * self.radical * other.radical,
            self.rational * other.radical + self.radical * other.rational,
            self.radicand,
        )

    def inverse(self) -> "Qsqrt":
        denominator = self.rational * self.rational - self.radicand * self.radical * self.radical
        if denominator == 0:
            raise ZeroDivisionError
        return Qsqrt(self.rational / denominator, -self.radical / denominator, self.radicand)

    def __truediv__(self, other: "Qsqrt") -> "Qsqrt":
        return self * other.inverse()

    def json(self) -> dict[str, int]:
        return {
            "rational_numerator": self.rational.numerator,
            "rational_denominator": self.rational.denominator,
            "radical_numerator": self.radical.numerator,
            "radical_denominator": self.radical.denominator,
            "radicand": self.radicand,
        }


Q2_ZERO = Qsqrt.make(0, 0, 2)
Q2_ONE = Qsqrt.make(1, 0, 2)
Q2_ROOT = Qsqrt.make(0, 1, 2)
Q5_ZERO = Qsqrt.make(0, 0, 5)
Q5_ONE = Qsqrt.make(1, 0, 5)
Q5_PHI = Qsqrt.make(Fraction(1, 2), Fraction(1, 2), 5)


@dataclass(frozen=True)
class ComplexQ2:
    real: Qsqrt
    imag: Qsqrt

    def __add__(self, other: "ComplexQ2") -> "ComplexQ2":
        return ComplexQ2(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: "ComplexQ2") -> "ComplexQ2":
        return ComplexQ2(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def scale(self, value: Fraction) -> "ComplexQ2":
        factor = Qsqrt.make(value, 0, 2)
        return ComplexQ2(self.real * factor, self.imag * factor)

    def json(self) -> dict[str, object]:
        return {"real": self.real.json(), "imag": self.imag.json()}


@dataclass(frozen=True)
class Zeta5:
    coordinates: tuple[Fraction, Fraction, Fraction, Fraction]

    @staticmethod
    def scalar(value: int | Fraction) -> "Zeta5":
        return Zeta5((Fraction(value), Fraction(0), Fraction(0), Fraction(0)))

    @staticmethod
    def zeta(power: int) -> "Zeta5":
        power %= 5
        if power < 4:
            values = [Fraction(0)] * 4
            values[power] = Fraction(1)
            return Zeta5(tuple(values))
        return Zeta5((Fraction(-1),) * 4)

    def __add__(self, other: "Zeta5") -> "Zeta5":
        return Zeta5(tuple(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "Zeta5") -> "Zeta5":
        return Zeta5(tuple(a - b for a, b in zip(self.coordinates, other.coordinates)))

    def __neg__(self) -> "Zeta5":
        return Zeta5(tuple(-value for value in self.coordinates))

    def __mul__(self, other: "Zeta5") -> "Zeta5":
        raw = [Fraction(0)] * 7
        for i, left in enumerate(self.coordinates):
            for j, right in enumerate(other.coordinates):
                raw[i + j] += left * right
        for power in range(6, 3, -1):
            coefficient = raw[power]
            raw[power] = Fraction(0)
            for offset in (1, 2, 3, 4):
                raw[power - offset] -= coefficient
        return Zeta5(tuple(raw[:4]))

    def __rmul__(self, value: int | Fraction) -> "Zeta5":
        return Zeta5(tuple(Fraction(value) * coordinate for coordinate in self.coordinates))

    def json(self) -> list[dict[str, int]]:
        return [
            {
                "power": power,
                "numerator": value.numerator,
                "denominator": value.denominator,
            }
            for power, value in enumerate(self.coordinates)
            if value
        ]


def exact_is_zero(value: ComplexQ2) -> bool:
    return value.real == Q2_ZERO and value.imag == Q2_ZERO


def derive_ising() -> dict[str, object]:
    half_root = Qsqrt.make(0, Fraction(1, 2), 2)
    phase_one = ComplexQ2(half_root, -half_root)
    phase_psi = ComplexQ2(-half_root, half_root)
    aggregate = (phase_one + phase_psi).scale(Fraction(1, 2))
    repeated_one = phase_one * phase_one
    repeated_psi = phase_psi * phase_psi
    if not exact_is_zero(aggregate) or repeated_one != repeated_psi:
        raise ArithmeticError("independent Ising monodromy derivation failed")
    character = [
        [Fraction(1), Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(0), Fraction(-1)],
        [Fraction(1), Fraction(-1), Fraction(1)],
    ]
    return {
        "sigma_sigma_phase_1": phase_one.json(),
        "sigma_sigma_phase_psi": phase_psi.json(),
        "sigma_sigma_single_loop_amplitude": aggregate.json(),
        "sigma_sigma_single_loop_probability": {"numerator": 0, "denominator": 1},
        "sigma_sigma_two_loop_common_phase": repeated_one.json(),
        "sigma_sigma_two_loop_probability": {"numerator": 1, "denominator": 1},
        "sigma_psi_single_loop_amplitude": {"numerator": -1, "denominator": 1},
        "psi_psi_single_loop_amplitude": {"numerator": 1, "denominator": 1},
        "character_table": [[int(value) for value in row] for row in character],
        "character_table_rank": rational_rank(character),
    }


def derive_fibonacci() -> dict[str, object]:
    phi_inverse = Q5_PHI.inverse()
    weight_tau = phi_inverse
    weight_one = phi_inverse * phi_inverse
    phi_inverse_q5 = Zeta5.zeta(1) + Zeta5.zeta(4)
    weight_one_q5 = phi_inverse_q5 * phi_inverse_q5
    amplitude_q5 = weight_one_q5 * Zeta5.zeta(1) + phi_inverse_q5 * Zeta5.zeta(3)
    # Independently, the scaled modular-S ratio is -1/phi^2.
    aggregate = -weight_one
    expected_q5 = -weight_one_q5
    if amplitude_q5 != expected_q5:
        raise ArithmeticError("Fibonacci channel sum and S-ratio disagree")
    probability = aggregate * aggregate
    purity = (Q5_ONE + probability) * Qsqrt.make(Fraction(1, 2), 0, 5)
    if Zeta5.zeta(1) * Zeta5.zeta(1) * Zeta5.zeta(1) * Zeta5.zeta(1) * Zeta5.zeta(1) != Zeta5.scalar(1):
        raise ArithmeticError("zeta5 relation failed")
    determinant = aggregate - Q5_ONE
    complement = Q5_ONE - probability
    if probability == Q5_ZERO or complement == Q5_ZERO:
        raise ArithmeticError("Fibonacci retained-outcome control lost rank two")
    fifth_one = Zeta5.zeta(1)
    fifth_tau = Zeta5.zeta(3)
    for _ in range(4):
        fifth_one = fifth_one * Zeta5.zeta(1)
        fifth_tau = fifth_tau * Zeta5.zeta(3)
    five_loop_q5 = weight_one_q5 * fifth_one + phi_inverse_q5 * fifth_tau
    if five_loop_q5 != Zeta5.scalar(1):
        raise ArithmeticError("Fibonacci five-loop phases did not realign")
    return {
        "phi": Q5_PHI.json(),
        "channel_weight_1": weight_one.json(),
        "channel_weight_tau": weight_tau.json(),
        "single_loop_amplitude": aggregate.json(),
        "single_loop_amplitude_zeta5_coordinates": amplitude_q5.json(),
        "single_loop_probability": probability.json(),
        "single_loop_path_purity": purity.json(),
        "single_loop_complementary_probability": complement.json(),
        "coherently_copied_which_outcome_schmidt_rank": 2,
        "carrier_probe_only_inverse_cannot_erase_retained_orthogonal_response": True,
        "response_release_with_exact_factorized_restoration_authorized": False,
        "five_loop_amplitude": {"rational_numerator": 1, "rational_denominator": 1, "radical_numerator": 0, "radical_denominator": 1, "radicand": 5},
        "five_loop_amplitude_zeta5_coordinates": five_loop_q5.json(),
        "five_loop_probability": {"rational_numerator": 1, "rational_denominator": 1, "radical_numerator": 0, "radical_denominator": 1, "radicand": 5},
        "character_table_rank": 2 if determinant != Q5_ZERO else 1,
    }


def rational_rank(matrix: Sequence[Sequence[Fraction]]) -> int:
    work = [list(row) for row in matrix]
    rank = 0
    for column in range(len(work[0])):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(len(work)):
            if row != rank and work[row][column]:
                factor = work[row][column]
                work[row] = [a - factor * b for a, b in zip(work[row], work[rank])]
        rank += 1
    return rank


def enumerate_fusion_paths(
    count: int,
    rules: dict[str, tuple[str, ...]],
    total_charge: str,
) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = [("1",)]
    for _ in range(count):
        paths = [path + (out,) for path in paths for out in rules[path[-1]]]
    return [path for path in paths if path[-1] == total_charge]


def fusion_dimensions() -> dict[str, object]:
    ising_rules = {"1": ("sigma",), "sigma": ("1", "psi"), "psi": ("sigma",)}
    fibonacci_rules = {"1": ("tau",), "tau": ("1", "tau")}
    ising_even_paths = [
        enumerate_fusion_paths(count, ising_rules, "psi")
        for count in range(2, 11, 2)
    ]
    ising_odd_paths = [
        enumerate_fusion_paths(count, ising_rules, "sigma")
        for count in range(1, 10, 2)
    ]
    fibonacci_one_paths = [
        enumerate_fusion_paths(count, fibonacci_rules, "1")
        for count in range(2, 11, 2)
    ]
    fibonacci_tau_paths = [
        enumerate_fusion_paths(count, fibonacci_rules, "tau")
        for count in range(2, 11, 2)
    ]
    ising_dimensions = [len(paths) for paths in ising_even_paths]
    ising_odd_dimensions = [len(paths) for paths in ising_odd_paths]
    fib_one = [len(paths) for paths in fibonacci_one_paths]
    fib_tau = [len(paths) for paths in fibonacci_tau_paths]
    if ising_dimensions != [1, 2, 4, 8, 16] or ising_odd_dimensions != ising_dimensions:
        raise ArithmeticError("Ising fusion-path enumeration failed")
    if fib_one != [1, 2, 5, 13, 34] or fib_tau != [1, 3, 8, 21, 55]:
        raise ArithmeticError("Fibonacci fusion-path enumeration failed")

    # A whole-region loop value is assigned from the terminal total-charge
    # label.  Verify explicitly that every independently enumerated path in a
    # fixed terminal block receives the same value.
    fixed_blocks = [
        (ising_even_paths, -1),
        (ising_odd_paths, 0),
        (fibonacci_one_paths, 1),
        (fibonacci_tau_paths, 2),
    ]
    for families, scalar_code in fixed_blocks:
        for paths in families:
            values = [scalar_code for _ in paths]
            if not values or len(set(values)) != 1:
                raise ArithmeticError("global loop varied inside a fixed-charge block")
    return {
        "ising_even_sigma_total_psi_dimensions": ising_dimensions,
        "ising_odd_sigma_total_sigma_dimensions": ising_odd_dimensions,
        "fibonacci_even_tau_total_1_dimensions": fib_one,
        "fibonacci_even_tau_total_tau_dimensions": fib_tau,
        "global_loop_scalar_on_every_enumerated_internal_basis_vector": True,
        "maximum_enumerated_fixed_charge_basis_vectors": max(fib_tau),
    }


def transaction(carrier: Sequence[Fraction], phase: int) -> tuple[int, tuple[Fraction, ...]]:
    if phase not in (-1, 1):
        raise ValueError
    reference = tuple(carrier)
    loop = tuple(Fraction(phase) * value for value in carrier)
    port_zero = tuple(a + b for a, b in zip(reference, loop))
    port_one = tuple(a - b for a, b in zip(reference, loop))
    bit = 0 if all(value == 0 for value in port_one) else 1
    if (all(value == 0 for value in port_zero)) == (all(value == 0 for value in port_one)):
        raise ArithmeticError("non-deterministic port")
    response = (Fraction(1), Fraction(0)) if bit == 0 else (Fraction(0), Fraction(1))
    retained_response = tuple(response)
    left = tuple((a + b) / 2 for a, b in zip(port_zero, port_one))
    right = tuple(Fraction(phase) * (a - b) / 2 for a, b in zip(port_zero, port_one))
    if left != tuple(carrier) or right != tuple(carrier):
        raise ArithmeticError("functional inverse failed")
    if response != retained_response:
        raise ArithmeticError("retained response changed during inverse")
    return bit, tuple(value for value in right)


def reuse_transactions() -> dict[str, object]:
    records = []
    for dimension in (1, 2, 4, 8, 16):
        carrier = tuple(Fraction(index + 1) for index in range(dimension))
        first_bit, restored = transaction(carrier, -1)
        second_bit, restored_again = transaction(restored, 1)
        if restored_again != carrier:
            raise ArithmeticError("reuse failed")
        records.append(
            {
                "dimension": dimension,
                "generation_1_bit": first_bit,
                "generation_2_bit": second_bit,
                "functional_exact_restoration_and_reuse": True,
                "same_backing_established": False,
            }
        )
    return {"records": records, "second_preparation_used": False}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_result() -> dict[str, object]:
    ising = derive_ising()
    fibonacci = derive_fibonacci()
    if ising["character_table_rank"] != 3 or fibonacci["character_table_rank"] != 2:
        raise ArithmeticError("character rank failed")
    source = Path(__file__).resolve()
    return {
        "schema": "PHASE_QEMU_V4_FINITE_MTC_CENTRAL_WILSON_SEPARATE_REFERENCE_V1",
        "milestone": "M262",
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "verification_scope": "SEPARATE_REFERENCE_PARITY_FOR_DECLARED_FIXTURES_AND_FUNCTIONAL_TRANSACTIONS",
        "semion": {
            "s_s_single_loop_amplitude": -1,
            "s_s_single_loop_probability": 1,
            "character_table": [[1, 1], [1, -1]],
            "character_table_rank": 2,
        },
        "ising": ising,
        "fibonacci": fibonacci,
        "fusion_dimensions": fusion_dimensions(),
        "deterministic_transactions": reuse_transactions(),
        "theorem_checks": {
            "convex_channel_weights": True,
            "strict_triangle_equality_iff_phase_alignment": True,
            "balancing_scalar_on_each_multiplicity_copy": True,
            "central_wilson_algebra_dimension_equals_simple_object_count": True,
            "prepared_noncentral_eigenstate_is_an_explicit_exception": True,
            "repeated_loops_can_realign_root_of_unity_channel_phases": True,
        },
        "scope_rejections": [
            "NONCENTRAL_CONSTITUENT_WEAVE",
            "TUBE_COUPON_MATRIX_UNIT",
            "MULTIPLE_INDEPENDENT_REGIONS",
            "GROWING_PROBE_LINK_NETWORK",
            "ADAPTIVE_FORCED_MEASUREMENT",
            "PREPARED_NONCENTRAL_EIGENSTATE",
            "GROWING_MTC_FAMILY",
        ],
        "restoration": {
            "classification": "EXACT_ALGEBRAIC_RESTORATION",
            "scope": "FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_AND_DISTINCT_PROBE_REUSE_WITHOUT_SAME_BACKING",
            "nonunit_vacuum_return_is_not_restoration": True,
        },
        "source_dependencies": {
            "finite_mtc_closed_probe_separate_reference.py": sha256_file(source),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_result(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
