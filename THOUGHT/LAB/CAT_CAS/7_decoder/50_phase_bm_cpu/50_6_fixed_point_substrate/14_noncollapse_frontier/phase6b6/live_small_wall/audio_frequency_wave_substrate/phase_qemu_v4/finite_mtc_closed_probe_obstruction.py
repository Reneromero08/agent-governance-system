#!/usr/bin/env python3
"""Exact fixed-UMTC central boundary-Wilson obstruction (M262).

This is a theorem diagnostic for one boundary-parallel simple Wilson probe
around one disk-like region of definite total charge.  It is not a general
anyon simulator, a tube-algebra result, QEMU execution, or physical evidence.

Exact fixture arithmetic uses Q(zeta_40), represented in the power basis
modulo Phi_40(x) = x^16 - x^12 + x^8 - x^4 + 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


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
NEXT_MECHANISM = (
    "NONCENTRAL_INTERACTING_SCATTERING_OR_FLOQUET_EIGENPHASE_WITH_"
    "PREPARED_EIGENSTATE_COST_REFERENCE_COMPLETE_FACTORIZATION_AND_"
    "GROWING_RELATIONAL_INVARIANT"
)


def _f(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class K40:
    coordinates: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.coordinates) != 16:
            raise ValueError("K40 requires exactly 16 power-basis coordinates")

    @staticmethod
    def scalar(value: int | Fraction) -> "K40":
        return K40((_f(value),) + (Fraction(0),) * 15)

    @staticmethod
    def basis(power: int) -> "K40":
        power %= 40
        raw = [Fraction(0)] * max(16, power + 1)
        raw[power] = Fraction(1)
        return K40._reduce(raw)

    @staticmethod
    def _reduce(raw_values: Sequence[Fraction]) -> "K40":
        raw = list(raw_values) + [Fraction(0)] * max(0, 31 - len(raw_values))
        for power in range(len(raw) - 1, 15, -1):
            coefficient = raw[power]
            if coefficient == 0:
                continue
            raw[power] = Fraction(0)
            raw[power - 4] += coefficient
            raw[power - 8] -= coefficient
            raw[power - 12] += coefficient
            raw[power - 16] -= coefficient
        return K40(tuple(raw[:16]))

    def __add__(self, other: "K40" | int | Fraction) -> "K40":
        rhs = coerce(other)
        return K40(tuple(a + b for a, b in zip(self.coordinates, rhs.coordinates)))

    def __radd__(self, other: "K40" | int | Fraction) -> "K40":
        return self + other

    def __sub__(self, other: "K40" | int | Fraction) -> "K40":
        rhs = coerce(other)
        return K40(tuple(a - b for a, b in zip(self.coordinates, rhs.coordinates)))

    def __rsub__(self, other: "K40" | int | Fraction) -> "K40":
        return coerce(other) - self

    def __neg__(self) -> "K40":
        return K40(tuple(-value for value in self.coordinates))

    def __mul__(self, other: "K40" | int | Fraction) -> "K40":
        rhs = coerce(other)
        raw = [Fraction(0)] * 31
        for left_power, left in enumerate(self.coordinates):
            if left == 0:
                continue
            for right_power, right in enumerate(rhs.coordinates):
                if right != 0:
                    raw[left_power + right_power] += left * right
        return K40._reduce(raw)

    def __rmul__(self, other: "K40" | int | Fraction) -> "K40":
        return self * other

    def __truediv__(self, other: "K40" | int | Fraction) -> "K40":
        return self * coerce(other).inverse()

    def __pow__(self, exponent: int) -> "K40":
        if exponent < 0:
            return self.inverse() ** (-exponent)
        result = ONE
        base = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * base
            base = base * base
            remaining >>= 1
        return result

    def inverse(self) -> "K40":
        if self == ZERO:
            raise ZeroDivisionError("zero has no inverse")
        matrix: list[list[Fraction]] = []
        columns = [self * K40.basis(power) for power in range(16)]
        for row in range(16):
            matrix.append(
                [columns[column].coordinates[row] for column in range(16)]
                + [Fraction(1 if row == 0 else 0)]
            )
        for column in range(16):
            pivot = next(
                (row for row in range(column, 16) if matrix[row][column] != 0),
                None,
            )
            if pivot is None:
                raise ZeroDivisionError("singular multiplication matrix")
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            scale = matrix[column][column]
            matrix[column] = [value / scale for value in matrix[column]]
            for row in range(16):
                if row == column:
                    continue
                factor = matrix[row][column]
                if factor != 0:
                    matrix[row] = [
                        value - factor * pivot_value
                        for value, pivot_value in zip(matrix[row], matrix[column])
                    ]
        result = K40(tuple(matrix[row][16] for row in range(16)))
        if self * result != ONE:
            raise ArithmeticError("inverse verification failed")
        return result

    def conjugate(self) -> "K40":
        result = ZERO
        for power, coefficient in enumerate(self.coordinates):
            if coefficient != 0:
                result += coefficient * K40.basis(-power)
        return result

    def is_unit_phase(self) -> bool:
        return self * self.conjugate() == ONE


def coerce(value: K40 | int | Fraction) -> K40:
    return value if isinstance(value, K40) else K40.scalar(value)


ZERO = K40.scalar(0)
ONE = K40.scalar(1)
ZETA40 = K40.basis(1)


def k40_json(value: K40) -> dict[str, object]:
    nonzero = []
    for power, coefficient in enumerate(value.coordinates):
        if coefficient:
            nonzero.append(
                {
                    "power": power,
                    "numerator": coefficient.numerator,
                    "denominator": coefficient.denominator,
                }
            )
    return {"basis": "Q_ZETA40_POWER_MOD_PHI40", "nonzero": nonzero}


def k40_payload_bits(value: K40) -> int:
    return sum(
        abs(coefficient.numerator).bit_length()
        + coefficient.denominator.bit_length()
        for coefficient in value.coordinates
        if coefficient
    )


@dataclass(frozen=True)
class Category:
    name: str
    labels: tuple[str, ...]
    dimensions: Mapping[str, K40]
    twists: Mapping[str, K40]
    fusion: Mapping[tuple[str, str], tuple[tuple[str, int], ...]]

    def channels(self, left: str, right: str) -> tuple[tuple[str, int], ...]:
        if left not in self.labels or right not in self.labels:
            raise KeyError("unknown simple-object label")
        return self.fusion[(left, right)]


def _symmetric_fusion(
    entries: Mapping[tuple[str, str], tuple[tuple[str, int], ...]]
) -> dict[tuple[str, str], tuple[tuple[str, int], ...]]:
    result = dict(entries)
    for (left, right), channels in tuple(entries.items()):
        result[(right, left)] = channels
    return result


def categories() -> tuple[Category, ...]:
    imaginary = K40.basis(10)
    minus_one = K40.basis(20)
    sqrt_two = K40.basis(5) + K40.basis(-5)
    phi = ONE + K40.basis(8) + K40.basis(-8)
    semion = Category(
        "SEMION",
        ("1", "s"),
        {"1": ONE, "s": ONE},
        {"1": ONE, "s": imaginary},
        _symmetric_fusion(
            {
                ("1", "1"): (("1", 1),),
                ("1", "s"): (("s", 1),),
                ("s", "s"): (("1", 1),),
            }
        ),
    )
    ising = Category(
        "ISING",
        ("1", "sigma", "psi"),
        {"1": ONE, "sigma": sqrt_two, "psi": ONE},
        {
            "1": ONE,
            "sigma": K40.basis(0),
            "psi": minus_one,
        },
        _symmetric_fusion(
            {
                ("1", "1"): (("1", 1),),
                ("1", "sigma"): (("sigma", 1),),
                ("1", "psi"): (("psi", 1),),
                ("sigma", "sigma"): (("1", 1), ("psi", 1)),
                ("sigma", "psi"): (("sigma", 1),),
                ("psi", "psi"): (("1", 1),),
            }
        ),
    )
    # Only theta_sigma^2 enters every Ising full-monodromy phase.  Store a
    # formal placeholder above and supply exact phases through this square in
    # monodromy_phase(), avoiding an unnecessary zeta_16 field extension.
    fibonacci = Category(
        "FIBONACCI",
        ("1", "tau"),
        {"1": ONE, "tau": phi},
        {"1": ONE, "tau": K40.basis(16)},
        _symmetric_fusion(
            {
                ("1", "1"): (("1", 1),),
                ("1", "tau"): (("tau", 1),),
                ("tau", "tau"): (("1", 1), ("tau", 1)),
            }
        ),
    )
    return semion, ising, fibonacci


def monodromy_phase(category: Category, left: str, right: str, channel: str) -> K40:
    if category.name == "ISING" and (left == "sigma" or right == "sigma"):
        if left == "sigma" and right == "sigma":
            theta_product = K40.basis(5)  # theta_sigma^2 = exp(i*pi/4)
        else:
            # theta_sigma cancels when the other charge is 1 or psi.
            other = right if left == "sigma" else left
            return category.twists[channel] / category.twists[other]
    else:
        theta_product = category.twists[left] * category.twists[right]
    return category.twists[channel] / theta_product


def analyze_pair(
    category: Category, left: str, right: str, loop_count: int = 1
) -> dict[str, object]:
    if loop_count < 1:
        raise ValueError("loop_count must be positive")
    denominator = category.dimensions[left] * category.dimensions[right]
    dimension_sum = ZERO
    phase_records = []
    amplitude = ZERO
    phases: list[K40] = []
    for channel, multiplicity in category.channels(left, right):
        dimension = category.dimensions[channel]
        dimension_sum += multiplicity * dimension
        weight = multiplicity * dimension / denominator
        phase = monodromy_phase(category, left, right, channel) ** loop_count
        if not phase.is_unit_phase():
            raise ArithmeticError("ribbon phase is not unit modulus")
        phases.append(phase)
        amplitude += weight * phase
        phase_records.append(
            {
                "channel": channel,
                "multiplicity": multiplicity,
                "weight": k40_json(weight),
                "phase": k40_json(phase),
            }
        )
    if dimension_sum != denominator:
        raise ArithmeticError("quantum-dimension fusion identity failed")
    return_probability = amplitude * amplitude.conjugate()
    if return_probability.conjugate() != return_probability:
        raise ArithmeticError("return probability is not real")
    aligned = all(phase == phases[0] for phase in phases)
    deterministic = return_probability == ONE
    if aligned != deterministic:
        raise ArithmeticError("strict convexity equality condition failed")
    return {
        "category": category.name,
        "target_charge": left,
        "probe_charge": right,
        "loop_count": loop_count,
        "channels": phase_records,
        "normalized_vacuum_return_amplitude": k40_json(amplitude),
        "vacuum_return_probability": k40_json(return_probability),
        "probe_path_reduced_purity": k40_json((ONE + return_probability) / 2),
        "supported_channel_phases_align": aligned,
        "deterministic_vacuum_return": deterministic,
        "common_full_monodromy_scalar_across_supported_channels": aligned,
        "maximum_amplitude_payload_bits": max(
            [k40_payload_bits(amplitude), k40_payload_bits(return_probability)]
            + [k40_payload_bits(phase) for phase in phases]
        ),
    }


def matrix_rank(matrix: Sequence[Sequence[K40]]) -> int:
    work = [list(row) for row in matrix]
    rows = len(work)
    columns = len(work[0]) if work else 0
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column] != ZERO), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = work[rank][column].inverse()
        work[rank] = [value * inverse for value in work[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = work[row][column]
            if factor != ZERO:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[rank])
                ]
        rank += 1
        if rank == rows:
            break
    return rank


def add_vectors(left: Sequence[K40], right: Sequence[K40]) -> tuple[K40, ...]:
    if len(left) != len(right):
        raise ValueError("vector length mismatch")
    return tuple(a + b for a, b in zip(left, right))


def subtract_vectors(left: Sequence[K40], right: Sequence[K40]) -> tuple[K40, ...]:
    if len(left) != len(right):
        raise ValueError("vector length mismatch")
    return tuple(a - b for a, b in zip(left, right))


def scale_vector(value: K40, vector: Sequence[K40]) -> tuple[K40, ...]:
    return tuple(value * entry for entry in vector)


def zero_vector(vector: Sequence[K40]) -> bool:
    return all(value == ZERO for value in vector)


def vector_commitment(vector: Sequence[K40]) -> str:
    digest = hashlib.sha256()
    for value in vector:
        for coefficient in value.coordinates:
            digest.update(
                f"{coefficient.numerator}/{coefficient.denominator};".encode()
            )
        digest.update(b"|")
    return digest.hexdigest()


def deterministic_scalar_transaction(
    carrier: Sequence[K40], phase: K40, generation: int, probe: str
) -> tuple[dict[str, object], tuple[K40, ...]]:
    if phase not in (ONE, -ONE):
        raise ValueError("the bounded response-copy transaction accepts only +/-1")
    initial = tuple(value for value in carrier)
    initial_commitment = vector_commitment(initial)
    reference_branch = tuple(value for value in initial)
    loop_branch = scale_vector(phase, initial)
    port_zero = add_vectors(reference_branch, loop_branch)
    port_one = subtract_vectors(reference_branch, loop_branch)
    if zero_vector(port_zero) == zero_vector(port_one):
        raise ArithmeticError("response port is not deterministic")
    boundary_bit = 1 if zero_vector(port_zero) else 0
    response = (ONE, ZERO) if boundary_bit == 0 else (ZERO, ONE)
    retained_response = tuple(response)
    # Copy the deterministic computational port to a retained response, then
    # apply H^{-1}=H/2 and the public scalar-loop inverse.
    unrecombined_reference = scale_vector(
        K40.scalar(Fraction(1, 2)), add_vectors(port_zero, port_one)
    )
    unrecombined_loop = scale_vector(
        K40.scalar(Fraction(1, 2)), subtract_vectors(port_zero, port_one)
    )
    restored_loop = scale_vector(phase.conjugate(), unrecombined_loop)
    if unrecombined_reference != initial or restored_loop != initial:
        raise ArithmeticError("public scalar-loop inverse failed")
    restored = tuple(value for value in restored_loop)
    if vector_commitment(restored) != initial_commitment:
        raise ArithmeticError("functional exact restoration failed")
    if response != retained_response:
        raise ArithmeticError("retained response changed during inverse")
    return (
        {
            "generation": generation,
            "probe": probe,
            "phase": k40_json(phase),
            "boundary_bit": boundary_bit,
            "response_copied_only_after_deterministic_port": True,
            "public_inverse_derived_from_phase_conjugation": True,
            "path_and_loop_branches_exactly_restored": True,
            "retained_boundary_survives_inverse": True,
            "retained_response_register": [k40_json(value) for value in response],
            "retained_response_register_unchanged_through_inverse": True,
            "initial_commitment": initial_commitment,
            "restored_commitment": vector_commitment(restored),
            "functional_exact_restoration": True,
            "same_backing_restoration_established": False,
        },
        restored,
    )


def ising_distinct_probe_reuse() -> dict[str, object]:
    cases = []
    for sigma_count in range(2, 12, 2):
        dimension = 1 << (sigma_count // 2 - 1)
        carrier = tuple(K40.scalar(index + 1) for index in range(dimension))
        first, restored = deterministic_scalar_transaction(
            carrier, -ONE, 1, "sigma"
        )
        second, restored_again = deterministic_scalar_transaction(
            restored, ONE, 2, "psi"
        )
        if restored_again != carrier:
            raise ArithmeticError("distinct-probe functional reuse failed")
        cases.append(
            {
                "target": f"sigma^{sigma_count}_total_psi",
                "internal_multiplicity_dimension": dimension,
                "preparation_count": 1,
                "generation_1": first,
                "generation_2": second,
                "distinct_probe_reuse": True,
                "functional_exact_returned_value_reuse": True,
                "same_backing_reuse_established": False,
            }
        )
    return {
        "cases": cases,
        "dimensions": [case["internal_multiplicity_dimension"] for case in cases],
        "boundaries_generation_1": [case["generation_1"]["boundary_bit"] for case in cases],
        "boundaries_generation_2": [case["generation_2"]["boundary_bit"] for case in cases],
        "second_preparation_used": False,
    }


def category_character_table(category: Category) -> dict[str, object]:
    table = []
    for target in category.labels:
        row = []
        for probe in category.labels:
            record = analyze_pair(category, target, probe)
            amplitude = ZERO
            for channel in record["channels"]:
                weight = next(
                    multiplicity * category.dimensions[label]
                    / (category.dimensions[target] * category.dimensions[probe])
                    for label, multiplicity in category.channels(target, probe)
                    if label == channel["channel"]
                )
                phase = monodromy_phase(category, target, probe, channel["channel"])
                amplitude += weight * phase
            row.append(amplitude)
        table.append(row)
    rank = matrix_rank(table)
    if rank != len(category.labels):
        raise ArithmeticError("modular character table lost full simple-charge rank")
    return {
        "category": category.name,
        "simple_object_count_k": len(category.labels),
        "character_table_exact_scalar_cells": len(category.labels) ** 2,
        "character_table_rank": rank,
        "all_boundary_wilson_loops_commute": True,
        "algebra_dimension": rank,
        "acts_as_scalar_on_each_fixed_charge_internal_multiplicity_block": True,
        "table": [[k40_json(value) for value in row] for row in table],
    }


def scope_controls() -> dict[str, object]:
    allowed = "BOUNDARY_PARALLEL_SIMPLE_CLOSED_WILSON_LOOP_ONE_DISK"
    rejected = {
        "NONCENTRAL_CONSTITUENT_WEAVE": "PATH_ENTERS_OR_SEPARATES_TARGET_PUNCTURES",
        "TUBE_COUPON_MATRIX_UNIT": "NOT_IN_CENTRAL_BOUNDARY_WILSON_CHARACTER_ALGEBRA",
        "MULTIPLE_INDEPENDENT_REGIONS": "OBSERVABLE_GEOMETRY_GROWS_BEYOND_ONE_DISK_LABEL",
        "GROWING_PROBE_LINK_NETWORK": "DESCRIPTOR_AND_LINK_INVARIANT_WORK_NOT_CONSTANT",
        "ADAPTIVE_FORCED_MEASUREMENT": "STOCHASTIC_HISTORY_CORRECTIONS_AND_POSTSELECTION_OUTSIDE_SCOPE",
        "PREPARED_NONCENTRAL_EIGENSTATE": "SINGLE_STATE_FACTORIZATION_DOES_NOT_IMPLY_UNIFORM_CENTRALITY",
        "GROWING_MTC_FAMILY": "CATEGORY_TABLE_AND_SIMPLE_OBJECT_COUNT_NOT_FIXED",
    }
    return {
        "accepted_descriptor_class": allowed,
        "rejected_descriptor_classes": rejected,
        "noncentral_two_eigenphase_control": {
            "eigenstate_zero_factorizes": True,
            "eigenstate_one_factorizes": True,
            "equal_superposition_path_purity": {"numerator": 1, "denominator": 2},
            "uniform_factorization_over_whole_space": False,
            "reference_complete_uniform_factorization": False,
        },
        "indefinite_charge_control": {
            "category": "ISING",
            "probe": "sigma",
            "charges": ["1", "psi"],
            "monodromy_phases": [1, -1],
            "equal_superposition_path_purity": {"numerator": 1, "denominator": 2},
            "fixed_charge_hypothesis_required": True,
        },
    }


def strictly_intermediate_return_control(record: Mapping[str, object]) -> dict[str, object]:
    """Classify retained which-outcome correlation without taking square roots.

    Copying vacuum-return versus complementary outcome coherently has Schmidt
    rank two exactly when both exact branch probabilities are nonzero.  An
    inverse acting only on the carrier and probe cannot erase that retained
    orthogonal record.  The p=0 and p=1 endpoints are deliberately excluded.
    """

    probability = ZERO
    encoded = record["vacuum_return_probability"]
    if not isinstance(encoded, Mapping):
        raise TypeError("malformed probability record")
    nonzero = encoded["nonzero"]
    if not isinstance(nonzero, Sequence):
        raise TypeError("malformed probability coordinates")
    for entry in nonzero:
        if not isinstance(entry, Mapping):
            raise TypeError("malformed probability coordinate")
        probability += Fraction(entry["numerator"], entry["denominator"]) * K40.basis(
            entry["power"]
        )
    complement = ONE - probability
    if probability == ZERO or complement == ZERO:
        raise ArithmeticError("strictly intermediate control requires 0<p<1")
    return {
        "fixture": "FIBONACCI_TAU_TAU_SINGLE_LOOP",
        "vacuum_return_probability": k40_json(probability),
        "complementary_outcome_probability": k40_json(complement),
        "both_exact_outcome_weights_nonzero": True,
        "coherently_copied_which_outcome_schmidt_rank": 2,
        "carrier_probe_only_inverse_cannot_erase_retained_orthogonal_response": True,
        "response_release_with_exact_factorized_restoration_authorized": False,
        "result_free_full_unitary_adjoint_can_restore": True,
        "zero_or_unit_probability_endpoints_excluded": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_result() -> dict[str, object]:
    semion, ising, fibonacci = categories()
    calibrations = {
        "semion_s_s_single_loop": analyze_pair(semion, "s", "s"),
        "ising_sigma_psi_single_loop": analyze_pair(ising, "sigma", "psi"),
        "ising_psi_psi_single_loop": analyze_pair(ising, "psi", "psi"),
        "ising_sigma_sigma_single_loop": analyze_pair(ising, "sigma", "sigma"),
        "fibonacci_tau_tau_single_loop": analyze_pair(fibonacci, "tau", "tau"),
    }
    repeated = {
        "ising_sigma_sigma_two_loops": analyze_pair(ising, "sigma", "sigma", 2),
        "fibonacci_tau_tau_five_loops": analyze_pair(fibonacci, "tau", "tau", 5),
    }
    if not calibrations["semion_s_s_single_loop"]["deterministic_vacuum_return"]:
        raise ArithmeticError("Semion calibration must be deterministic")
    if not calibrations["ising_sigma_psi_single_loop"]["deterministic_vacuum_return"]:
        raise ArithmeticError("Ising sigma-psi calibration must be deterministic")
    if calibrations["ising_sigma_sigma_single_loop"]["deterministic_vacuum_return"]:
        raise ArithmeticError("Ising sigma-sigma single loop must be nondeterministic")
    if calibrations["fibonacci_tau_tau_single_loop"]["deterministic_vacuum_return"]:
        raise ArithmeticError("Fibonacci tau-tau single loop must be nondeterministic")
    if not all(record["deterministic_vacuum_return"] for record in repeated.values()):
        raise ArithmeticError("finite-order repeated-loop controls must return exactly")
    character_tables = [category_character_table(category) for category in categories()]
    deterministic_transactions = ising_distinct_probe_reuse()
    script_path = Path(__file__).resolve()
    return {
        "schema": "PHASE_QEMU_V4_FINITE_MTC_CENTRAL_WILSON_OBSTRUCTION_V1",
        "milestone": "M262",
        "claim": CLAIM,
        "claim_ceiling": CLAIM_CEILING,
        "verification_scope": {
            "fixture_arithmetic_dimensions_character_ranks_and_transactions": "SEPARATE_REFERENCE_PARITY",
            "general_fixed_umtc_centrality_and_equality_theorem": "FORMAL_DERIVATION_SOURCE_AUDITED",
            "resource_accounting": "PACKAGE_SELF_REVIEW",
        },
        "theorem": {
            "scope": "FIXED_FINITE_UMTC_ONE_DISK_DEFINITE_SIMPLE_TOTAL_CHARGE_BOUNDARY_PARALLEL_SIMPLE_CLOSED_WILSON_LOOPS",
            "normalized_monodromy_formula": "M_ab=sum_c[N_ab^c*d_c/(d_a*d_b)]*theta_c/(theta_a*theta_b)=S_ab*S_00/(S_0a*S_0b)",
            "channel_weights_positive_and_sum_to_one": True,
            "single_loop_absolute_value_at_most_one": True,
            "single_loop_equality_iff_all_supported_channel_phases_align": True,
            "balancing_identity_is_scalar_on_each_fusion_multiplicity_copy": True,
            "whole_region_wilson_action_is_scalar_on_internal_multiplicity": True,
            "all_simple_boundary_wilson_loops_span_k_dimensional_charge_projector_algebra": True,
            "fixed_charge_internal_multiplicity_observable_rank": 1,
            "declared_internal_multiplicity_dimensions_without_materialized_identity_matrices": deterministic_transactions[
                "dimensions"
            ],
        },
        "calibrations": calibrations,
        "repeated_loop_controls": repeated,
        "character_tables": character_tables,
        "deterministic_transactions": deterministic_transactions,
        "scope_controls": scope_controls(),
        "strictly_intermediate_return_control": strictly_intermediate_return_control(
            calibrations["fibonacci_tau_tau_single_loop"]
        ),
        "restoration": {
            "classification": "EXACT_ALGEBRAIC_RESTORATION",
            "scope": "FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_AND_DISTINCT_PROBE_REUSE_WITHOUT_SAME_BACKING",
            "vacuum_return_probability_is_not_catalytic_restoration": True,
            "result_free_full_monodromy_adjoint_identity": True,
            "response_preserving_bounded_functional_transaction_executed": True,
            "same_backing_restoration_or_reuse_established": False,
        },
        "strongest_classical_comparator": {
            "fixed_category_definite_charge_query": "O1_EXACT_FINITE_MONODROMY_CHARACTER_TABLE_LOOKUP_AFTER_VALIDATION",
            "single_probe_charge_distribution_state": "O_K_EXACT_SCALARS",
            "boundary_charge_coherence_state_if_required": "O_K_SQUARED_EXACT_SCALARS",
            "generic_sequential_charge_recurrence": "O_N_K_SQUARED_ARITHMETIC_O_K_CELLS_WITH_COEFFICIENT_HEIGHT_COUNTED",
            "arbitrary_internal_many_anyon_state_represented_by_comparator": False,
            "deriving_charge_statistics_from_large_input_is_free": False,
            "arbitrary_growing_probe_link_network_constant_work_established": False,
            "fixed_level_noncentral_braid_or_jones_complexity_killed": False,
        },
        "resource_law": {
            "exact_field": "Q_ZETA40_POWER_BASIS_MOD_PHI40",
            "rational_coordinates_per_field_scalar": 16,
            "category_character_table_field_cells": {record["category"]: record["character_table_exact_scalar_cells"] for record in character_tables},
            "accepted_transaction_max_logical_carrier_field_cells": 16,
            "accepted_transaction_two_path_logical_field_cells": 32,
            "retained_boundary_bits_during_inverse": 1,
            "full_internal_multiplicity_identity_matrices_materialized": False,
            "field_inversion_gaussian_matrix_rational_cells": 16 * 17,
            "resource_verification_level": "PACKAGE_SELF_REVIEW",
            "whole_process_liveness_complete": False,
            "python_objects_allocator_hash_json_serialization_timing_rss_uninstrumented": True,
            "physical_energy_noise_bandwidth_latency_area_shots_uninstrumented": True,
        },
        "claim_limits": {
            "full_annular_or_tube_algebra_obstruction": False,
            "noncentral_constituent_braid_obstruction": False,
            "prepared_eigenstate_obstruction": False,
            "adaptive_measurement_or_postselection_obstruction": False,
            "multiple_region_or_growing_genus_obstruction": False,
            "growing_probe_network_constant_work": False,
            "growing_mtc_family_obstruction": False,
            "physical_anyons_or_interferometer": False,
            "phase_qemu_device_execution": False,
            "catvm_custody": False,
            "distinct_phase_resource": False,
            "computational_advantage": False,
            "m257_escape": False,
            "small_wall_crossing": False,
            "unbounded_compute": False,
            "replace_bits_with_pi": False,
        },
        "disposition": "SINGLE_GLOBAL_CLOSED_PROBE_HOLONOMY_IN_A_FIXED_FINITE_UMTC_IS_MULTIPLICITY_BLIND_AND_REDUCES_TO_A_CONSTANT_SIZE_TOTAL_CHARGE_MONODROMY_TABLE",
        "next_mechanism": NEXT_MECHANISM,
        "source_dependencies": {
            "finite_mtc_closed_probe_obstruction.py": sha256_file(script_path),
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
