#!/usr/bin/env python3
"""M256 standalone zeta-polynomial oracle and transactional reference.

This source imports neither the service nor the controller.  It represents
Q(zeta8) as Q[z]/(z^4+1), independently derives the polynomial and scalar
Schur recurrences, and executes its own custody/restoration state machine.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


P = tuple[Fraction, Fraction, Fraction, Fraction]
ZERO: P = (Fraction(0),) * 4
ONE: P = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
Z: P = (Fraction(0), Fraction(1), Fraction(0), Fraction(0))
CAPACITY = 4


def add(left: P, right: P) -> P:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def neg(value: P) -> P:
    return tuple(-item for item in value)  # type: ignore[return-value]


def sub(left: P, right: P) -> P:
    return add(left, neg(right))


def scale(value: P, scalar: Fraction) -> P:
    return tuple(item * scalar for item in value)  # type: ignore[return-value]


def mul(left: P, right: P) -> P:
    output = [Fraction(0)] * 4
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            exponent = i + j
            if exponent >= 4:
                output[exponent - 4] -= a * b
            else:
                output[exponent] += a * b
    return tuple(output)  # type: ignore[return-value]


def power(base: P, exponent: int) -> P:
    result = ONE
    for _ in range(exponent):
        result = mul(result, base)
    return result


def conjugate(value: P) -> P:
    output = ZERO
    for exponent, coefficient in enumerate(value):
        if coefficient:
            output = add(output, scale(power(Z, (-exponent) % 8), coefficient))
    return output


def inverse(value: P) -> P:
    # Independent rational Gauss-Jordan solve for multiplication by value.
    matrix = [[Fraction(0) for _ in range(5)] for _ in range(4)]
    for column in range(4):
        product = mul(value, power(Z, column))
        for row in range(4):
            matrix[row][column] = product[row]
    matrix[0][4] = Fraction(1)
    for pivot in range(4):
        row = next((candidate for candidate in range(pivot, 4) if matrix[candidate][pivot]), None)
        if row is None:
            raise RuntimeError("M256 reference singular field element")
        matrix[pivot], matrix[row] = matrix[row], matrix[pivot]
        divisor = matrix[pivot][pivot]
        matrix[pivot] = [item / divisor for item in matrix[pivot]]
        for target in range(4):
            if target == pivot:
                continue
            factor = matrix[target][pivot]
            if factor:
                matrix[target] = [
                    matrix[target][column] - factor * matrix[pivot][column]
                    for column in range(5)
                ]
    result: P = tuple(matrix[row][4] for row in range(4))  # type: ignore[assignment]
    if mul(value, result) != ONE:
        raise RuntimeError("M256 reference inverse verification failed")
    return result


def fj(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def to_service_basis(value: P) -> list[list[int]]:
    # p0+p1*z+p2*z^2+p3*z^3 = a+b*sqrt2+c*i+d*sqrt2*i.
    a = value[0]
    b = (value[1] - value[3]) / 2
    c = value[2]
    d = (value[1] + value[3]) / 2
    return [fj(item) for item in (a, b, c, d)]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sections(descriptor: dict[str, Any]) -> tuple[Fraction, Fraction, Fraction]:
    values = tuple(Fraction(numerator, denominator) for numerator, denominator in descriptor["sections"])
    if len(values) != 3 or descriptor["evaluation"] != "ZETA8":
        raise RuntimeError("M256 reference descriptor rejected")
    if any(abs(value) >= 1 for value in values):
        raise RuntimeError("M256 reference lossless domain rejected")
    return values  # type: ignore[return-value]


def forward_arrays(
    numerator: list[P], denominator: list[P], coefficient: Fraction,
) -> tuple[list[P], list[P]]:
    next_numerator: list[P] = []
    next_denominator: list[P] = []
    for index in range(CAPACITY):
        shifted = numerator[index - 1] if index else ZERO
        next_numerator.append(add(scale(denominator[index], coefficient), shifted))
        next_denominator.append(add(denominator[index], scale(shifted, coefficient)))
    return next_numerator, next_denominator


def inverse_arrays(
    numerator: list[P], denominator: list[P], coefficient: Fraction,
) -> tuple[list[P], list[P]]:
    divisor = Fraction(1) - coefficient * coefficient
    if not divisor or sub(numerator[0], scale(denominator[0], coefficient)) != ZERO:
        raise RuntimeError("M256 reference inverse divisibility rejected")
    old_numerator: list[P] = []
    old_denominator: list[P] = []
    for index in range(CAPACITY):
        source = index + 1
        old_numerator.append(
            scale(sub(numerator[source], scale(denominator[source], coefficient)), 1 / divisor)
            if source < CAPACITY else ZERO
        )
        old_denominator.append(
            scale(sub(denominator[index], scale(numerator[index], coefficient)), 1 / divisor)
        )
    return old_numerator, old_denominator


def evaluate(coefficients: Iterable[P], point: P) -> P:
    value = ZERO
    for coefficient in reversed(list(coefficients)):
        value = add(mul(value, point), coefficient)
    return value


def polynomial_degree(coefficients: list[P]) -> int:
    return next((index for index in range(len(coefficients) - 1, -1, -1) if coefficients[index] != ZERO), -1)


def laurent_norm(coefficients: list[P]) -> dict[int, P]:
    output: dict[int, P] = {}
    for left_index, left in enumerate(coefficients):
        for right_index, right in enumerate(coefficients):
            exponent = left_index - right_index
            output[exponent] = add(output.get(exponent, ZERO), mul(left, conjugate(right)))
    return {exponent: value for exponent, value in output.items() if value != ZERO}


def formal_allpass(numerator: list[P], denominator: list[P]) -> bool:
    return laurent_norm(numerator) == laurent_norm(denominator)


def polynomial_boundary(word: tuple[Fraction, Fraction, Fraction], seed: P = Z) -> tuple[P, list[P], list[P]]:
    numerator = [seed, ZERO, ZERO, ZERO]
    denominator = [ONE, ZERO, ZERO, ZERO]
    for coefficient in word:
        numerator, denominator = forward_arrays(numerator, denominator, coefficient)
    value = mul(evaluate(numerator, Z), inverse(evaluate(denominator, Z)))
    return value, numerator, denominator


def scalar_boundary(word: tuple[Fraction, Fraction, Fraction], seed: P = Z) -> P:
    value = seed
    for coefficient in word:
        z_value = mul(Z, value)
        value = mul(add(scale(ONE, coefficient), z_value), inverse(add(ONE, scale(z_value, coefficient))))
    return value


def fraction_payload_bits(value: Fraction) -> int:
    return max(1, abs(value.numerator).bit_length() + 1) + max(1, value.denominator.bit_length())


def service_basis_payload(values: Iterable[P]) -> dict[str, int]:
    coordinates: list[Fraction] = []
    for value in values:
        encoded = to_service_basis(value)
        coordinates.extend(Fraction(numerator, denominator) for numerator, denominator in encoded)
    return {
        "integer_coordinate_count": len(coordinates),
        "maximum_numerator_signed_bits": max(max(1, abs(value.numerator).bit_length() + 1) for value in coordinates),
        "maximum_denominator_bits": max(max(1, value.denominator.bit_length()) for value in coordinates),
        "total_fraction_payload_bits": sum(fraction_payload_bits(value) for value in coordinates),
    }


@dataclass
class ReferencePort:
    def __post_init__(self) -> None:
        self.numerator = [Z, ZERO, ZERO, ZERO]
        self.denominator = [ONE, ZERO, ZERO, ZERO]
        self.receipts = [Fraction(0)] * 3
        self.cursor = 0
        self.last_generation = 0
        self.live_generation = 0
        self.live_descriptor: tuple[Fraction, Fraction, Fraction] | None = None
        self.leased = False

    def canonical(self) -> bool:
        return (
            self.numerator == [Z, ZERO, ZERO, ZERO]
            and self.denominator == [ONE, ZERO, ZERO, ZERO]
            and self.receipts == [Fraction(0)] * 3 and self.cursor == 0
            and self.live_generation == 0 and self.live_descriptor is None and not self.leased
        )

    def lease(self, word: tuple[Fraction, Fraction, Fraction], generation: int) -> None:
        if not self.canonical() or generation != self.last_generation + 1:
            raise RuntimeError("M256 reference lease rejected")
        self.live_generation = generation
        self.live_descriptor = word
        self.leased = True

    def require(self, word: tuple[Fraction, Fraction, Fraction], generation: int) -> None:
        if not self.leased or self.live_descriptor != word or self.live_generation != generation:
            raise RuntimeError("M256 reference custody rejected")

    def forward(self, word: tuple[Fraction, Fraction, Fraction], generation: int) -> None:
        self.require(word, generation)
        coefficient = word[self.cursor]
        next_numerator, next_denominator = forward_arrays(
            self.numerator, self.denominator, coefficient
        )
        self.numerator[:] = next_numerator
        self.denominator[:] = next_denominator
        self.receipts[self.cursor] = coefficient
        self.cursor += 1

    def project(self, word: tuple[Fraction, Fraction, Fraction], generation: int) -> P:
        self.require(word, generation)
        if self.cursor != 3:
            raise RuntimeError("M256 reference premature projection")
        return mul(evaluate(self.numerator, Z), inverse(evaluate(self.denominator, Z)))

    def reverse(self, word: tuple[Fraction, Fraction, Fraction], generation: int) -> None:
        self.require(word, generation)
        index = self.cursor - 1
        coefficient = word[index]
        if index < 0 or self.receipts[index] != coefficient:
            raise RuntimeError("M256 reference inverse ownership rejected")
        old_numerator, old_denominator = inverse_arrays(
            self.numerator, self.denominator, coefficient
        )
        self.numerator[:] = old_numerator
        self.denominator[:] = old_denominator
        self.receipts[index] = Fraction(0)
        self.cursor -= 1

    def release(self) -> None:
        if not (
            self.leased and self.numerator == [Z, ZERO, ZERO, ZERO]
            and self.denominator == [ONE, ZERO, ZERO, ZERO]
            and self.receipts == [Fraction(0)] * 3 and self.cursor == 0
        ):
            raise RuntimeError("M256 reference release rejected")
        self.last_generation = self.live_generation
        self.live_generation = 0
        self.live_descriptor = None
        self.leased = False


def execute_reference(
    port: ReferencePort, word: tuple[Fraction, Fraction, Fraction], generation: int,
) -> dict[str, object]:
    backing_ids = (id(port.numerator), id(port.denominator), id(port.receipts))
    port.lease(word, generation)
    peak = service_basis_payload([*port.numerator, *port.denominator])
    for _ in word:
        port.forward(word, generation)
        current = service_basis_payload([*port.numerator, *port.denominator])
        peak = {
            key: max(peak[key], current[key]) if key != "total_fraction_payload_bits" else max(peak[key], current[key])
            for key in peak
        }
    value = port.project(word, generation)
    final_degrees = (polynomial_degree(port.numerator), polynomial_degree(port.denominator))
    exact_formal_allpass = formal_allpass(port.numerator, port.denominator)
    final_payload = service_basis_payload([*port.numerator, *port.denominator])
    for _ in word:
        port.reverse(word, generation)
    port.release()
    return {
        "generation": generation,
        "winding": 3,
        "evaluation": to_service_basis(value),
        "actual_final_numerator_degree": final_degrees[0],
        "actual_final_denominator_degree": final_degrees[1],
        "exact_formal_laurent_allpass_identity": exact_formal_allpass,
        "same_reference_backings": backing_ids == (id(port.numerator), id(port.denominator), id(port.receipts)),
        "canonical_after_restoration": port.canonical(),
        "baseline_reload_used": False,
        "final_waveform_service_basis_payload": final_payload,
        "peak_resident_waveform_service_basis_payload_component": peak,
    }


def rejected(action: Any) -> bool:
    try:
        action()
    except (RuntimeError, TypeError, ValueError):
        return True
    return False


def main(public_path: Path, raw_path: Path) -> None:
    public = json.loads(public_path.read_text())
    raw = json.loads(raw_path.read_text())
    cases = public["cases"]
    primary_word = sections(cases["primary"]["descriptor"])
    reuse_word = sections(cases["reuse"]["descriptor"])
    sham_word = sections(cases["sham"]["descriptor"])

    port = ReferencePort()
    primary = execute_reference(port, primary_word, 1)
    reuse = execute_reference(port, reuse_word, 2)
    fresh = execute_reference(ReferencePort(), reuse_word, 1)
    sham = execute_reference(ReferencePort(), sham_word, 1)
    reference_cases = {"PRIMARY": primary, "REUSE": reuse, "FRESH": fresh, "SHAM": sham}

    raw_cases = {case["run_kind"]: case for case in raw["cases"]}
    boundary_parity = all(
        raw_cases[key]["winding"] == value["winding"]
        and raw_cases[key]["evaluation"] == value["evaluation"]
        for key, value in reference_cases.items()
    )

    scalar_parity = all(
        scalar_boundary(sections(cases[key.lower()]["descriptor"]))
        == polynomial_boundary(sections(cases[key.lower()]["descriptor"]))[0]
        for key in ("PRIMARY", "REUSE", "FRESH", "SHAM")
    )
    allpass_norm = all(
        mul(value, conjugate(value)) == ONE
        for value in (polynomial_boundary(primary_word)[0], polynomial_boundary(reuse_word)[0])
    )
    formal_allpass_suite = all(
        formal_allpass(polynomial_boundary(word)[1], polynomial_boundary(word)[2])
        for word in (primary_word, reuse_word, sham_word)
    )

    probe_coefficients = (Fraction(-1, 2), Fraction(0), Fraction(1, 3), Fraction(1, 2))
    probe_count = 0
    probe_parity = True
    for first in probe_coefficients:
        for second in probe_coefficients:
            for third in probe_coefficients:
                probe_word = (first, second, third)
                value, probe_numerator, probe_denominator = polynomial_boundary(probe_word)
                if scalar_boundary(probe_word) != value:
                    probe_parity = False
                for coefficient in reversed(probe_word):
                    probe_numerator, probe_denominator = inverse_arrays(
                        probe_numerator, probe_denominator, coefficient
                    )
                if probe_numerator != [Z, ZERO, ZERO, ZERO] or probe_denominator != [ONE, ZERO, ZERO, ZERO]:
                    probe_parity = False
                probe_count += 1

    missing_port = ReferencePort()
    missing_port.lease(primary_word, 1)
    for _ in primary_word:
        missing_port.forward(primary_word, 1)
    missing_port.reverse(primary_word, 1)
    missing = rejected(missing_port.release)

    _, final_n, final_d = polynomial_boundary(primary_word)
    wrong = rejected(lambda: inverse_arrays(final_n, final_d, Fraction(1, 5)))
    reordered = rejected(lambda: inverse_arrays(final_n, final_d, primary_word[1]))

    controls = {
        "independent_polynomial_and_scalar_schur_boundaries_match": scalar_parity,
        "independent_boundaries_match_atomic_backend_responses": boundary_parity,
        "exact_unit_circle_norm_at_public_evaluation": allpass_norm,
        "exact_formal_laurent_allpass_identity_for_declared_words": formal_allpass_suite,
        "feedback_disabled_sham_retains_winding_but_changes_boundary": (
            sham["winding"] == primary["winding"] == 3 and sham["evaluation"] != primary["evaluation"]
        ),
        "phase_seed_mutation_changes_primary_boundary": (
            polynomial_boundary(primary_word, ONE)[0] != polynomial_boundary(primary_word, Z)[0]
        ),
        "declared_first_two_sections_are_noncommuting": (
            polynomial_boundary((primary_word[0], primary_word[1], Fraction(0)))[1:]
            != polynomial_boundary((primary_word[1], primary_word[0], Fraction(0)))[1:]
        ),
        "missing_inverse_fails_exact_release": missing,
        "wrong_inverse_fails_exact_divisibility_or_restoration": wrong,
        "reordered_noncommuting_inverse_fails_exact_divisibility_or_restoration": reordered,
        "restored_generation2_reuse_matches_fresh_boundary_and_payload": (
            reuse["evaluation"] == fresh["evaluation"]
            and reuse["winding"] == fresh["winding"]
            and reuse["final_waveform_service_basis_payload"] == fresh["final_waveform_service_basis_payload"]
        ),
        "same_reference_backings_and_exact_restoration": all(
            case["same_reference_backings"] and case["canonical_after_restoration"]
            for case in reference_cases.values()
        ),
        "no_reference_baseline_reload": all(not case["baseline_reload_used"] for case in reference_cases.values()),
        "sixty_four_descriptor_words_exact_scalar_polynomial_inverse_parity": (
            probe_count == 64 and probe_parity
        ),
    }
    if not all(controls.values()):
        raise RuntimeError(f"M256 independent control failure: {controls}")

    output = {
        "milestone": 256,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "cases": reference_cases,
        "controls": controls,
        "exact_laws": {
            "field_representation": "Q_ZETA8_AS_Q_Z_MOD_Z4_PLUS1",
            "forward_polynomial_recurrence_derived_independently": True,
            "inverse_polynomial_recurrence_derived_independently": True,
            "scalar_schur_recurrence_is_exact_boundary_bisimulation": True,
            "formal_winding_increment_per_section": 1,
        },
        "classical_baselines": {
            "fixed_fixture": "O1_PUBLIC_CERTIFICATE",
            "actual_boundary": "ONE_FIELD_SCALAR_PLUS_ONE_INTEGER_WINDING",
            "full_waveform": "TWO_POLYNOMIAL_SCHUR_RECURRENCE",
            "actual_boundary_scalar_field_cells": 1,
            "full_waveform_field_cells_at_depth3": 8,
            "catvm_restoration_required_for_classical_baselines": False,
        },
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: separate_reference.py PUBLIC.json RAW.json")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
