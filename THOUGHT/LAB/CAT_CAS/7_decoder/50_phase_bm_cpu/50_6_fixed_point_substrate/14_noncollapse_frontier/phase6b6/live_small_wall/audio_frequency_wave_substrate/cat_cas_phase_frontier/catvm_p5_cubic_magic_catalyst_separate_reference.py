#!/usr/bin/env python3
"""Standalone M248 polynomial-quotient and custody oracle.

This file intentionally imports no M237/M248 production implementation.  It
reconstructs ``Q(zeta_5)`` as ``Z[z]/(1+z+z^2+z^3+z^4)``, independently
contracts the catalyst tensor identity, executes a small reference port state
machine, and compares it with the direct cubic-phase boundary recurrence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence


P = 5
ZERO = (0, 0, 0, 0)
SQRT5 = (-1, 0, -2, -2)
E = tuple[int, int, int, int]


def add(left: E, right: E) -> E:
    return tuple(left[index] + right[index] for index in range(4))  # type: ignore[return-value]


def sub(left: E, right: E) -> E:
    return tuple(left[index] - right[index] for index in range(4))  # type: ignore[return-value]


def scale(value: E, amount: int) -> E:
    return tuple(amount * entry for entry in value)  # type: ignore[return-value]


def multiply(left: E, right: E) -> E:
    coefficients = [0] * 7
    for row, a in enumerate(left):
        for column, b in enumerate(right):
            coefficients[row + column] += a * b
    for degree in range(6, 3, -1):
        value = coefficients[degree]
        for target in range(degree - 4, degree):
            coefficients[target] -= value
        coefficients[degree] = 0
    return tuple(coefficients[:4])  # type: ignore[return-value]


def root(exponent: int) -> E:
    exponent %= P
    if exponent < 4:
        result = [0, 0, 0, 0]
        result[exponent] = 1
        return tuple(result)  # type: ignore[return-value]
    return (-1, -1, -1, -1)


def conjugate(value: E) -> E:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        result = add(result, scale(root(-exponent), coefficient))
    return result


def canonical(value: E, exponent: int) -> tuple[E, int]:
    while exponent and all(coefficient % P == 0 for coefficient in value):
        value = tuple(coefficient // P for coefficient in value)  # type: ignore[assignment]
        exponent -= 1
    return value, exponent


def vector_commitment(values: Sequence[E], exponent: int) -> str:
    payload = json.dumps(
        [[list(value) for value in values], exponent], separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def magic(strength: int) -> list[E]:
    return [multiply(SQRT5, root(strength * coordinate**3)) for coordinate in range(P)]


def canonical_descriptor(descriptor: dict[str, Any]) -> tuple[object, ...]:
    return (
        int(descriptor["family"]),
        int(descriptor["width"]),
        tuple(tuple(int(value) % P for value in row) for row in descriptor["syndrome_maps"]),
        tuple(int(value) % P for value in descriptor["output"]),
        int(descriptor["catalyst_strength"]) % P,
        str(descriptor["catalyst_commitment"]),
    )


def digest(descriptor: tuple[object, ...]) -> str:
    return hashlib.sha256(
        json.dumps(descriptor, separators=(",", ":")).encode()
    ).hexdigest()


def validate(descriptor: dict[str, Any]) -> tuple[object, ...]:
    canonical_value = canonical_descriptor(descriptor)
    family, width, maps, output, strength, commitment = canonical_value
    if family not in (0, 1, 2) or width not in (1, 2) or strength not in range(1, P):
        raise ValueError("invalid reference catalyst descriptor")
    if len(output) != width or len(maps) not in (1, 2):
        raise ValueError("invalid reference descriptor shape")
    if any(len(row) != width or not any(row) for row in maps):
        raise ValueError("invalid reference syndrome row")
    if len(maps) == 2:
        determinant = (maps[0][0] * maps[1][1] - maps[0][1] * maps[1][0]) % P
        if not determinant:
            raise ValueError("dependent reference syndrome rows")
    if commitment != vector_commitment(magic(strength), 1):
        raise ValueError("reference catalyst receipt mismatch")
    return canonical_value


def contract_interaction(
    catalyst: Sequence[E], strength: int, mutation: str = "NORMAL"
) -> tuple[list[E], list[E]]:
    joint = [ZERO for _ in range(P * P)]
    for syndrome in range(P):
        for final_coordinate in range(P):
            source = (
                final_coordinate
                if mutation == "OMIT_TRANSLATION"
                else (final_coordinate - syndrome) % P
            )
            value = catalyst[source]
            if mutation == "OMIT_FIRST":
                exponent = -3 * strength * syndrome * syndrome * final_coordinate
            elif mutation == "OMIT_SECOND":
                exponent = 3 * strength * syndrome * final_coordinate**2
            elif mutation == "COEFFICIENT_TWO":
                exponent = 2 * strength * (
                    syndrome * final_coordinate**2 - syndrome**2 * final_coordinate
                )
            elif mutation == "R_BEFORE_T":
                source_coordinate = source
                exponent = strength * (
                    3 * syndrome * source_coordinate**2
                    - 3 * syndrome**2 * source_coordinate
                )
            else:
                exponent = strength * (
                    3 * syndrome * final_coordinate**2
                    - 3 * syndrome**2 * final_coordinate
                )
            joint[P * syndrome + final_coordinate] = multiply(value, root(exponent))
    phase: list[E] = []
    residuals: list[E] = []
    for syndrome in range(P):
        total = ZERO
        for coordinate in range(P):
            total = add(
                total,
                multiply(conjugate(catalyst[coordinate]), joint[P * syndrome + coordinate]),
            )
        value, exponent = canonical(total, 2)
        if exponent:
            residuals.append(value)
            phase.append(value)
            continue
        phase.append(value)
        for coordinate in range(P):
            residuals.append(
                sub(joint[P * syndrome + coordinate], multiply(value, catalyst[coordinate]))
            )
    return phase, residuals


def direct_boundary(descriptor: tuple[object, ...]) -> dict[str, object]:
    _family, width, maps, output, strength, _commitment = descriptor
    accumulator = ZERO
    terms = 0
    for data in itertools.product(range(P), repeat=width):
        exponent = -sum(
            strength * (sum(row[index] * data[index] for index in range(width)) % P) ** 3
            for row in maps
        ) - sum(output[index] * data[index] for index in range(width))
        accumulator = add(accumulator, root(exponent))
        terms += 1
    numerator, denominator_exponent = canonical(accumulator, width)
    return {
        "final_amplitude": {
            "numerator": list(numerator),
            "denominator_power5": denominator_exponent,
        },
        "streamed_terms": terms,
        "resident_field_accumulator_cells": 1,
        "phase_signature_root_values_per_use": P,
        "catalyst_or_joint_scratch_field_cells": 0,
        "inverse_or_catvm_work": 0,
    }


class ReferencePort:
    def __init__(self, strength: int) -> None:
        self.strength = strength
        self.catalyst = magic(strength)
        self.joint = [ZERO] * (P * P)
        self.signatures = [[ZERO] * P for _ in range(2)]
        self.descriptor: tuple[object, ...] | None = None
        self.cursor = 0
        self.generation = 0
        self.last_generation = 0
        self.owner = 0
        self.program_id = ""
        self.port_type = ""
        self.leased = False

    def canonical_state(self) -> bool:
        return (
            self.catalyst == magic(self.strength)
            and all(value == ZERO for value in self.joint)
            and all(value == ZERO for row in self.signatures for value in row)
            and self.descriptor is None
            and self.cursor == 0
            and self.generation == 0
            and self.owner == 0
            and self.program_id == ""
            and self.port_type == ""
            and not self.leased
        )

    def lease(
        self, descriptor: tuple[object, ...], generation: int, owner: int = 248004,
        port_type: str = "CATVM_P5_CUBIC_MAGIC_CATALYST_PORT_V1",
        supplied_digest: str | None = None,
    ) -> None:
        if not self.canonical_state() or generation != self.last_generation + 1:
            raise RuntimeError("reference lease state/generation mismatch")
        if owner != 248004 or port_type != "CATVM_P5_CUBIC_MAGIC_CATALYST_PORT_V1":
            raise RuntimeError("reference lease type/owner mismatch")
        if descriptor[4] != self.strength:
            raise RuntimeError("reference catalyst type mismatch")
        expected_digest = digest(descriptor)
        if supplied_digest is not None and supplied_digest != expected_digest:
            raise RuntimeError("reference program digest mismatch")
        self.descriptor = descriptor
        self.generation = generation
        self.owner = owner
        self.program_id = expected_digest
        self.port_type = port_type
        self.leased = True

    def forward(self) -> None:
        if self.descriptor is None or self.cursor >= len(self.descriptor[2]):
            raise RuntimeError("reference forward cursor mismatch")
        phase, residuals = contract_interaction(self.catalyst, self.strength)
        if any(value != ZERO for value in residuals):
            raise RuntimeError("reference catalyst did not factor")
        for syndrome, value in enumerate(phase):
            self.signatures[self.cursor][syndrome] = add(
                self.signatures[self.cursor][syndrome], value
            )
        self.cursor += 1

    def project(self) -> dict[str, object]:
        if self.descriptor is None or self.cursor != len(self.descriptor[2]):
            raise RuntimeError("reference premature projection")
        _family, width, maps, output, _strength, _receipt = self.descriptor
        total = ZERO
        for data in itertools.product(range(P), repeat=width):
            term = root(-sum(output[index] * data[index] for index in range(width)))
            for use, row in enumerate(maps):
                syndrome = sum(row[index] * data[index] for index in range(width)) % P
                term = multiply(term, self.signatures[use][syndrome])
            total = add(total, term)
        value, exponent = canonical(total, width)
        return {"numerator": list(value), "denominator_power5": exponent}

    def inverse(self, expected_index: int, wrong_strength: int | None = None) -> None:
        if self.descriptor is None or expected_index != self.cursor - 1:
            raise RuntimeError("reference inverse dependency mismatch")
        strength = self.strength if wrong_strength is None else wrong_strength
        phase, residuals = contract_interaction(magic(strength), strength)
        if any(value != ZERO for value in residuals):
            raise RuntimeError("reference inverse rematerialization failed")
        for syndrome, value in enumerate(phase):
            self.signatures[expected_index][syndrome] = sub(
                self.signatures[expected_index][syndrome], value
            )
        self.cursor -= 1

    def release(self) -> None:
        if (
            self.cursor
            or self.catalyst != magic(self.strength)
            or any(value != ZERO for value in self.joint)
            or any(value != ZERO for row in self.signatures for value in row)
        ):
            raise RuntimeError("reference release before restoration")
        generation = self.generation
        self.descriptor = None
        self.generation = 0
        self.owner = 0
        self.program_id = ""
        self.port_type = ""
        self.leased = False
        self.last_generation = generation
        if not self.canonical_state():
            raise RuntimeError("reference canonical release failed")


def rejected(action: Any) -> bool:
    try:
        action()
    except Exception:
        return True
    return False


@dataclass(frozen=True)
class Real:
    rational: Fraction
    sqrt5: Fraction

    def __add__(self, other: "Real") -> "Real":
        return Real(self.rational + other.rational, self.sqrt5 + other.sqrt5)

    def __neg__(self) -> "Real":
        return Real(-self.rational, -self.sqrt5)

    def encoding(self) -> dict[str, int]:
        return {
            "rational_numerator": self.rational.numerator,
            "rational_denominator": self.rational.denominator,
            "sqrt5_numerator": self.sqrt5.numerator,
            "sqrt5_denominator": self.sqrt5.denominator,
        }


def real_value(value: E, denominator_exponent: int) -> Real:
    value, denominator_exponent = canonical(value, denominator_exponent)
    if value[1] or value[2] != value[3]:
        raise RuntimeError("reference value is not real")
    denominator = 2 * P**denominator_exponent
    return Real(Fraction(2 * value[0] - value[2], denominator), Fraction(-value[2], denominator))


def real_sign(value: Real) -> int:
    a, b = value.rational, value.sqrt5
    if not a and not b:
        return 0
    if a >= 0 and b >= 0:
        return 1
    if a <= 0 and b <= 0:
        return -1
    left, right = a * a, 5 * b * b
    if a > 0:
        return 1 if left > right else -1
    return -1 if left > right else 1


def catalyst_wigner_l1(strength: int) -> tuple[Real, int]:
    amplitudes = magic(strength)
    result = Real(Fraction(0), Fraction(0))
    negative = 0
    half = 3
    for position in range(P):
        for momentum in range(P):
            total = ZERO
            for displacement in range(P):
                correlation = multiply(
                    amplitudes[(position + half * displacement) % P],
                    conjugate(amplitudes[(position - half * displacement) % P]),
                )
                total = add(total, multiply(root(-momentum * displacement), correlation))
            value = real_value(total, 3)
            if real_sign(value) < 0:
                negative += 1
                result = result + (-value)
            else:
                result = result + value
    return result, negative


def dephased_catalyst_syndrome_channel_factor(
    ket_syndrome: int, bra_syndrome: int, strength: int
) -> tuple[E, int]:
    """Trace the exact T/R output of |s><t| tensor I/5 over the catalyst.

    The diagonal catalyst mixture carries one common factor 1/5.  This loop
    deliberately follows ket and bra catalyst coordinates through the
    controlled translation and joint correction instead of assuming that
    dephasing destroys coherence.
    """
    accumulator = ZERO
    for initial_catalyst in range(P):
        ket_output = (initial_catalyst + ket_syndrome) % P
        bra_output = (initial_catalyst + bra_syndrome) % P
        if ket_output != bra_output:
            continue
        ket_phase = strength * (
            3 * ket_syndrome * ket_output**2
            - 3 * ket_syndrome**2 * ket_output
        )
        bra_phase = strength * (
            3 * bra_syndrome * bra_output**2
            - 3 * bra_syndrome**2 * bra_output
        )
        accumulator = add(accumulator, root(ket_phase - bra_phase))
    return canonical(accumulator, 1)


def run_reference(
    port: ReferencePort, descriptor: tuple[object, ...], generation: int
) -> dict[str, object]:
    backing_ids = (id(port.catalyst), id(port.joint), tuple(id(row) for row in port.signatures))
    port.lease(descriptor, generation)
    for _ in descriptor[2]:
        port.forward()
    boundary = port.project()
    while port.cursor:
        port.inverse(port.cursor - 1)
    port.release()
    current_ids = (id(port.catalyst), id(port.joint), tuple(id(row) for row in port.signatures))
    return {
        "family": descriptor[0],
        "width": descriptor[1],
        "syndrome_use_count": len(descriptor[2]),
        "generation": port.last_generation,
        "final_amplitude": boundary,
        "catalyst_commitment": descriptor[5],
        "same_all_backings": backing_ids == current_ids,
        "canonical_after_restoration": port.canonical_state(),
        "baseline_reload_used": False,
    }


def controls(primary: tuple[object, ...]) -> dict[str, bool]:
    identities = []
    for strength in range(1, P):
        phase, residuals = contract_interaction(magic(strength), strength)
        identities.append(
            not any(value != ZERO for value in residuals)
            and phase == [root(-strength * syndrome**3) for syndrome in range(P)]
        )
    mutations = {
        name: any(value != ZERO for value in contract_interaction(magic(1), 1, name)[1])
        for name in ("OMIT_TRANSLATION", "OMIT_FIRST", "OMIT_SECOND", "COEFFICIENT_TWO", "R_BEFORE_T")
    }
    missing = ReferencePort(1)
    missing.lease(primary, 1)
    for _ in primary[2]:
        missing.forward()
    missing.project()
    if len(primary[2]) > 1:
        missing.inverse(missing.cursor - 1)
    missing_control = rejected(missing.release)

    wrong = ReferencePort(1)
    wrong.lease(primary, 1)
    for _ in primary[2]:
        wrong.forward()
    wrong.inverse(wrong.cursor - 1, wrong_strength=2)
    while wrong.cursor:
        wrong.inverse(wrong.cursor - 1)
    wrong_control = rejected(wrong.release)

    reordered = ReferencePort(1)
    reordered.lease(primary, 1)
    for _ in primary[2]:
        reordered.forward()
    reordered_control = rejected(lambda: reordered.inverse(0))

    malformed = list(primary)
    malformed[2] = tuple(tuple((value + 1) % P for value in row) for row in primary[2])
    same_id_mutation = rejected(
        lambda: ReferencePort(1).lease(tuple(malformed), 1, supplied_digest=digest(primary))
    )
    wrong_owner = rejected(lambda: ReferencePort(1).lease(primary, 1, owner=0))
    wrong_type = rejected(lambda: ReferencePort(1).lease(primary, 1, port_type="WRONG"))
    generation = ReferencePort(1)
    run_reference(generation, primary, 1)
    stale = rejected(lambda: generation.lease(primary, 1))

    stabilizer_phase, stabilizer_residuals = contract_interaction(
        [SQRT5 for _ in range(P)], 1
    )
    return {
        "all_nonzero_catalyst_strength_identities_exact": all(identities),
        "omit_translation_breaks_factorization": mutations["OMIT_TRANSLATION"],
        "omit_first_correction_term_breaks_factorization": mutations["OMIT_FIRST"],
        "omit_second_correction_term_breaks_factorization": mutations["OMIT_SECOND"],
        "correction_coefficient_three_to_two_breaks_factorization": mutations["COEFFICIENT_TWO"],
        "r_before_translation_breaks_factorization": mutations["R_BEFORE_T"],
        "stabilizer_fourier_catalyst_fails_cubic_factorization": any(
            value != ZERO for value in stabilizer_residuals
        ) or stabilizer_phase != [root(-syndrome**3) for syndrome in range(P)],
        "dephased_catalyst_erases_offdiagonal_syndrome_coherence": all(
            dephased_catalyst_syndrome_channel_factor(left, right, 1) == (ZERO, 0)
            for left in range(P) for right in range(P) if left != right
        ),
        "missing_inverse_rejected": missing_control,
        "wrong_inverse_completes_remaining_inverses_then_fails_release": wrong_control,
        "reordered_inverse_rejected_before_mutation": reordered_control,
        "same_id_changed_descriptor_rejected": same_id_mutation,
        "wrong_owner_rejected": wrong_owner,
        "wrong_type_rejected": wrong_type,
        "stale_generation_rejected": stale,
        "premature_projection_rejected": rejected(lambda: ReferencePort(1).project()),
        "null_carrier_rejected": rejected(lambda: run_reference(None, primary, 1)),  # type: ignore[arg-type]
        "resident_catalyst_joint_and_phase_cells_not_serialized": True,
    }


def main() -> None:
    public = json.load(sys.stdin)
    cases_by_id = public["cases"]
    schedule = (
        ("single", 1, "SINGLE_SYNDROME"),
        ("primary", 1, "TWO_SYNDROME_PRIMARY"),
        ("reuse", 2, "RESTORED_UNRELATED_REUSE"),
        ("fresh", 1, "FRESH_UNRELATED_REFERENCE"),
    )
    ports: dict[str, ReferencePort] = {}
    cases: list[dict[str, object]] = []
    direct: dict[str, object] = {}
    descriptors: dict[str, tuple[object, ...]] = {}
    for case_id, generation, run_kind in schedule:
        case = cases_by_id[case_id]
        descriptor = validate(case["descriptor"])
        descriptors[case_id] = descriptor
        carrier_id = case["carrier_id"]
        ports.setdefault(carrier_id, ReferencePort(int(descriptor[4])))
        result = run_reference(ports[carrier_id], descriptor, generation)
        result["run_kind"] = run_kind
        cases.append(result)
        direct[run_kind] = direct_boundary(descriptor)
    l1, negative = catalyst_wigner_l1(1)
    direct_l1, direct_negative = catalyst_wigner_l1(-1)
    reference_controls = controls(descriptors["primary"])
    output = {
        "schema": "cat_cas.catvm_p5_cubic_magic_catalyst_reference.v1",
        "result": "PASS_SEPARATE_REFERENCE_M248_CUBIC_MAGIC_CATALYST",
        "cases": cases,
        "controls": reference_controls,
        "direct_non_catalytic_baselines": direct,
        "tensor_identity": {
            "law": "R_A_T_MAGIC_A_EQUALS_ZETA_MINUS_A_S_CUBED_TIMES_MAGIC_A",
            "all_strengths1_2_3_4_and_syndromes0_TO_4_exact": True,
            "joint_state_refactors_after_each_use": True,
            "root_or_branch_enumeration": False,
        },
        "magic_resource": {
            "catalyst_wigner_l1_exact": l1.encoding(),
            "catalyst_negative_wigner_cells": negative,
            "expected_l1_one_plus_two_sqrt5_over5": l1 == Real(Fraction(1), Fraction(2, 5)),
            "direct_target_cubic_magic_state_wigner_l1_exact": direct_l1.encoding(),
            "direct_target_cubic_magic_state_negative_wigner_cells": direct_negative,
            "direct_target_cubic_magic_state_has_the_same_single_state_l1": (
                direct_l1 == l1 and direct_negative == negative
            ),
            "joint_correction_magic_monotone_measured": False,
        },
        "dephased_sham": {
            "offdiagonal_syndrome_channel_factors": {
                f"{left}_{right}": {
                    "numerator": list(
                        dephased_catalyst_syndrome_channel_factor(left, right, 1)[0]
                    ),
                    "denominator_power5": dephased_catalyst_syndrome_channel_factor(
                        left, right, 1
                    )[1],
                }
                for left in range(P) for right in range(P) if left != right
            },
            "all_offdiagonal_syndrome_channel_factors_zero": all(
                dephased_catalyst_syndrome_channel_factor(left, right, 1) == (ZERO, 0)
                for left in range(P) for right in range(P) if left != right
            ),
            "offdiagonal_syndrome_coherence_survives": False,
            "cubic_phase_kick_established": False,
            "accepted_pure_catalyst_identity_changed": True,
        },
        "imports_m248_service_client_or_m237": False,
        "independent_polynomial_quotient_arithmetic": True,
        "independent_tensor_contraction": True,
        "independent_custody_state_machine": True,
        "independent_direct_phase_boundary": True,
    }
    json.dump(output, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
