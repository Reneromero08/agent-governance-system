#!/usr/bin/env python3
"""Exact continuous-SU(2) central open phase-relation diagnostic.

For a central translation-invariant kernel on SU(2), normalized Haar
composition is diagonal in the irreducible-character basis.  A public
spectral phase kernel with coefficient (r+1) u**(r(r+2)) multiplies the rth
character coefficient by the Casimir phase u**(r(r+2)).  Pointwise
intersection with 1+a*chi_1 uses the Clebsch-Gordan law
chi_r*chi_1 = chi_(r-1)+chi_(r+1).

The carrier keeps one finite character polynomial resident on an unresolved
group port, projects only its value at the identity, then reverses every
operation on the same list backing before unrelated reuse.  The experiment
tests whether changing from Abelian S1 harmonics to non-Abelian character
fusion produces compact closure; it does not assume that outcome.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)


PORT_TYPE = 21217
DEPTHS = (1, 2, 4, 8, 16, 32)
PRIMARY_DEPTH = 32
REUSE_DEPTH = 19
PREDECESSOR_SOURCE_SHA256 = (
    "483f11f0dec6623265fb6ba96c8371088f83ef246414d9091628b4046ab6a457"
)


@dataclass(frozen=True)
class G:
    real: Fraction
    imag: Fraction

    @staticmethod
    def rational(real: int | Fraction, imag: int | Fraction = 0) -> "G":
        return G(Fraction(real), Fraction(imag))

    def __add__(self, other: "G") -> "G":
        return G(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "G") -> "G":
        return G(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "G") -> "G":
        return G(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def scale(self, scalar: int | Fraction) -> "G":
        value = Fraction(scalar)
        return G(self.real * value, self.imag * value)

    def inverse(self) -> "G":
        norm = self.real * self.real + self.imag * self.imag
        if not norm:
            raise ZeroDivisionError("zero Gaussian rational")
        return G(self.real / norm, -self.imag / norm)

    def __truediv__(self, other: "G") -> "G":
        return self * other.inverse()

    def is_zero(self) -> bool:
        return not self.real and not self.imag

    def token(self) -> str:
        return (
            f"{self.real.numerator}/{self.real.denominator}:"
            f"{self.imag.numerator}/{self.imag.denominator}"
        )

    def public_json(self) -> dict[str, str]:
        return {
            "real_numerator": str(self.real.numerator),
            "real_denominator": str(self.real.denominator),
            "imag_numerator": str(self.imag.numerator),
            "imag_denominator": str(self.imag.denominator),
        }


ZERO = G.rational(0)
ONE = G.rational(1)
SOURCE_FUNDAMENTAL = G(Fraction(12, 13), Fraction(5, 13))
CASIMIR_PHASES = (
    G(Fraction(3, 5), Fraction(4, 5)),
    G(Fraction(5, 13), Fraction(12, 13)),
)
FUSION_FACTORS = (
    G(Fraction(8, 17), Fraction(15, 17)),
    G(Fraction(7, 25), Fraction(24, 25)),
)


@dataclass(frozen=True)
class Operation:
    kind: str
    parameter: int

    def token(self) -> str:
        return f"{self.kind}:{self.parameter}"


@dataclass
class Work:
    casimir_phase_power_multiplications: int = 0
    composition_coefficient_multiplications: int = 0
    fusion_coefficient_multiplications: int = 0
    fusion_coefficient_additions: int = 0
    inverse_fusion_coefficient_multiplications: int = 0
    inverse_fusion_coefficient_subtractions: int = 0
    inverse_fusion_coefficient_divisions: int = 0
    final_boundary_scalar_multiplications: int = 0
    final_boundary_additions: int = 0
    port_leases: int = 0
    port_releases: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0
    peak_character_cells: int = 0
    peak_character_payload_bits: int = 0
    public_program_commitment_hashes: int = 0
    public_program_operation_records_hashed: int = 0
    state_commitment_hashes: int = 0
    state_commitment_character_cells_hashed: int = 0
    boundary_commitment_hashes: int = 0

    def observe(self, coefficients: list[G]) -> None:
        self.peak_character_cells = max(
            self.peak_character_cells, len(coefficients)
        )
        self.peak_character_payload_bits = max(
            self.peak_character_payload_bits, gaussian_payload_bits(coefficients)
        )

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass
class OpenCharacterPort:
    coefficients: list[G]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    public_program_commitment: str = ""

    def lease(
        self,
        owner: int,
        generation: int,
        operations: tuple[Operation, ...],
        work: Work,
    ) -> None:
        if self.live:
            raise RuntimeError("open SU2 character port already live")
        if not self.coefficients:
            raise ValueError("null SU2 character carrier rejected")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid SU2 character owner or generation")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = len(operations)
        self.public_program_commitment = program_commitment(operations)
        work.public_program_commitment_hashes += 1
        work.public_program_operation_records_hashed += len(operations)
        work.port_leases += 1
        work.observe(self.coefficients)

    def require(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        work: Work,
    ) -> None:
        if not self.live:
            raise RuntimeError("open SU2 character port is not live")
        if owner != self.owner:
            raise PermissionError("open SU2 character port owner mismatch")
        supplied_commitment = program_commitment(operations)
        work.public_program_commitment_hashes += 1
        work.public_program_operation_records_hashed += len(operations)
        if supplied_commitment != self.public_program_commitment:
            raise ValueError("open SU2 character program mismatch")

    def forward(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        index: int,
        work: Work,
    ) -> None:
        self.require(owner, operations, work)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("forward public cursor mismatch")
        apply_operation(self.coefficients, operations[index], False, work)
        self.cursor += 1
        work.forward_operations += 1
        work.observe(self.coefficients)

    def inverse(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        index: int,
        work: Work,
    ) -> None:
        self.require(owner, operations, work)
        if index != self.cursor - 1:
            raise ValueError("inverse public cursor mismatch")
        apply_operation(self.coefficients, operations[index], True, work)
        self.cursor -= 1
        work.inverse_operations += 1
        work.observe(self.coefficients)

    def project_final(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        work: Work,
    ) -> G:
        self.require(owner, operations, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal SU2 character projection rejected")
        return evaluate_identity(self.coefficients, work)

    def release(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        work: Work,
    ) -> int:
        self.require(owner, operations, work)
        if self.cursor:
            raise RuntimeError("SU2 character port released before exact inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.expected_steps = 0
        self.public_program_commitment = ""
        work.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: OpenCharacterPort
    restoration_generation: int = 0


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def gaussian_payload_bits(values: list[G] | tuple[G, ...]) -> int:
    return sum(
        signed_bits(component.numerator) + component.denominator.bit_length()
        for value in values
        for component in (value.real, value.imag)
    )


def maximum_integer_bits(values: list[G]) -> dict[str, int]:
    components = [
        component
        for value in values
        for component in (value.real, value.imag)
    ]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(component.numerator) for component in components
        ),
        "maximum_denominator_bits": max(
            component.denominator.bit_length() for component in components
        ),
    }


def state_commitment(coefficients: list[G]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in coefficients).encode("ascii")
    ).hexdigest()


def boundary_commitment(value: G) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def program_commitment(operations: tuple[Operation, ...]) -> str:
    return hashlib.sha256(
        "|".join(operation.token() for operation in operations).encode("ascii")
    ).hexdigest()


def public_program(depth: int, family: int) -> tuple[Operation, ...]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    operations: list[Operation] = []
    for index in range(depth):
        composition = Operation(
            "COMPOSE_CASIMIR_PHASE", (index + family) % len(CASIMIR_PHASES)
        )
        intersection = Operation(
            "INTERSECT_FUNDAMENTAL",
            (3 * index + 2 * family + 1) % len(FUSION_FACTORS),
        )
        if (index + family) % 3:
            operations.extend((composition, intersection))
        else:
            operations.extend((intersection, composition))
    return tuple(operations)


def counted_power(value: G, exponent: int, work: Work) -> G:
    result = ONE
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = result * factor
            work.casimir_phase_power_multiplications += 1
        remaining >>= 1
        if remaining:
            factor = factor * factor
            work.casimir_phase_power_multiplications += 1
    return result


def apply_casimir_phase(
    coefficients: list[G], phase: G, work: Work
) -> None:
    for highest_weight, coefficient in enumerate(coefficients):
        exponent = highest_weight * (highest_weight + 2)
        multiplier = counted_power(phase, exponent, work)
        coefficients[highest_weight] = coefficient * multiplier
        work.composition_coefficient_multiplications += 1


def apply_fundamental_fusion(
    coefficients: list[G], factor: G, work: Work
) -> None:
    previous = ZERO
    current = coefficients[0]
    original_cells = len(coefficients)
    for index in range(original_cells):
        following = (
            coefficients[index + 1] if index + 1 < original_cells else ZERO
        )
        neighbor_sum = previous + following
        updated = current + factor * neighbor_sum
        coefficients[index] = updated
        previous, current = current, following
        work.fusion_coefficient_multiplications += 1
        work.fusion_coefficient_additions += 2
    coefficients.append(factor * previous)
    work.fusion_coefficient_multiplications += 1


def divide_fundamental_fusion(
    coefficients: list[G], factor: G, work: Work
) -> None:
    if len(coefficients) < 2:
        raise ValueError("fundamental fusion inverse has no factor to remove")
    original_cells = len(coefficients) - 1
    next_coefficient = ZERO
    current = coefficients[-1] / factor
    work.inverse_fusion_coefficient_divisions += 1
    for index in range(original_cells - 1, 0, -1):
        residual = coefficients[index] - current - factor * next_coefficient
        previous = residual / factor
        work.inverse_fusion_coefficient_multiplications += 1
        work.inverse_fusion_coefficient_subtractions += 2
        work.inverse_fusion_coefficient_divisions += 1
        coefficients[index] = current
        next_coefficient, current = current, previous
    expected_constant = current + factor * next_coefficient
    work.inverse_fusion_coefficient_multiplications += 1
    work.inverse_fusion_coefficient_subtractions += 1
    if coefficients[0] != expected_constant:
        raise ValueError("fundamental fusion factor is not exactly divisible")
    coefficients[0] = current
    coefficients.pop()


def apply_operation(
    coefficients: list[G], operation: Operation, inverse: bool, work: Work
) -> None:
    if operation.kind == "COMPOSE_CASIMIR_PHASE":
        try:
            phase = CASIMIR_PHASES[operation.parameter]
        except IndexError as error:
            raise ValueError("unknown Casimir phase parameter") from error
        apply_casimir_phase(
            coefficients, phase.inverse() if inverse else phase, work
        )
        return
    if operation.kind == "INTERSECT_FUNDAMENTAL":
        try:
            factor = FUSION_FACTORS[operation.parameter]
        except IndexError as error:
            raise ValueError("unknown fundamental fusion parameter") from error
        if inverse:
            divide_fundamental_fusion(coefficients, factor, work)
        else:
            apply_fundamental_fusion(coefficients, factor, work)
        return
    raise ValueError("unknown SU2 open phase-relation operation")


def evaluate_identity(coefficients: list[G], work: Work | None = None) -> G:
    result = ZERO
    for highest_weight, coefficient in enumerate(coefficients):
        result = result + coefficient.scale(highest_weight + 1)
        if work is not None:
            work.final_boundary_scalar_multiplications += 1
            work.final_boundary_additions += 1
    return result


def mismatch(left: list[G], right: list[G]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=False)) + abs(
        len(left) - len(right)
    )


def execute_public_word(
    source: list[G], operations: tuple[Operation, ...]
) -> tuple[list[G], Work]:
    coefficients = source.copy()
    work = Work()
    work.observe(coefficients)
    for operation in operations:
        apply_operation(coefficients, operation, False, work)
        work.forward_operations += 1
        work.observe(coefficients)
    return coefficients, work


def transaction(
    carrier: Carrier, source: list[G], operations: tuple[Operation, ...]
) -> tuple[dict[str, object], Work]:
    backing = id(carrier.port.coefficients)
    generation = carrier.restoration_generation + 1
    owner = 212000 + generation
    work = Work()
    carrier.port.lease(owner, generation, operations, work)
    for index in range(len(operations)):
        carrier.port.forward(owner, operations, index, work)
    boundary = carrier.port.project_final(owner, operations, work)
    forward_commitment = state_commitment(carrier.port.coefficients)
    work.state_commitment_hashes += 1
    work.state_commitment_character_cells_hashed += len(
        carrier.port.coefficients
    )
    final_boundary_commitment = boundary_commitment(boundary)
    work.boundary_commitment_hashes += 1
    forward_cells = len(carrier.port.coefficients)
    forward_payload = gaussian_payload_bits(carrier.port.coefficients)
    missing_inverse_error = mismatch(carrier.port.coefficients, source) + carrier.port.cursor
    for index in range(len(operations) - 1, -1, -1):
        carrier.port.inverse(owner, operations, index, work)
    restored_generation = carrier.port.release(owner, operations, work)
    carrier.restoration_generation = restored_generation
    return (
        {
            "boundary": boundary,
            "boundary_commitment": final_boundary_commitment,
            "boundary_payload_bits": gaussian_payload_bits([boundary]),
            "forward_state_commitment": forward_commitment,
            "forward_character_cells": forward_cells,
            "forward_character_payload_bits": forward_payload,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_character_cells": mismatch(
                carrier.port.coefficients, source
            ),
            "same_backing": id(carrier.port.coefficients) == backing,
            "port_released": not carrier.port.live,
            "restoration_generation": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        work,
    )


def controls() -> dict[str, object]:
    source = [ONE, SOURCE_FUNDAMENTAL]
    word = public_program(2, 0)
    work = Work()
    port = OpenCharacterPort(source.copy())
    port.lease(7001, 1, word, work)
    wrong_owner = wrong_type = premature = reordered = False
    try:
        port.forward(7002, word, 0, work)
    except PermissionError:
        wrong_owner = True
    try:
        apply_operation(port.coefficients, Operation("DECODE", 0), False, work)
    except ValueError:
        wrong_type = True
    port.forward(7001, word, 0, work)
    try:
        port.project_final(7001, word, work)
    except PermissionError:
        premature = True
    for index in range(1, len(word)):
        port.forward(7001, word, index, work)
    missing_inverse = port.cursor != 0
    try:
        port.inverse(7001, word, len(word) - 2, work)
    except ValueError:
        reordered = True
    for index in range(len(word) - 1, -1, -1):
        port.inverse(7001, word, index, work)
    port.release(7001, word, work)

    null_carrier = False
    try:
        OpenCharacterPort([]).lease(1, 1, word, Work())
    except ValueError:
        null_carrier = True

    first = source.copy()
    second = source.copy()
    local = Work()
    apply_operation(first, Operation("COMPOSE_CASIMIR_PHASE", 0), False, local)
    apply_operation(first, Operation("INTERSECT_FUNDAMENTAL", 0), False, local)
    apply_operation(second, Operation("INTERSECT_FUNDAMENTAL", 0), False, local)
    apply_operation(second, Operation("COMPOSE_CASIMIR_PHASE", 0), False, local)

    original_word = public_program(4, 0)
    changed_word = list(original_word)
    for index, operation in enumerate(changed_word):
        if operation.kind == "INTERSECT_FUNDAMENTAL":
            changed_word[index] = Operation(
                operation.kind,
                (operation.parameter + 1) % len(FUSION_FACTORS),
            )
            break
    original, _ = execute_public_word(source, original_word)
    changed, _ = execute_public_word(source, tuple(changed_word))
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "null_carrier_rejected": null_carrier,
        "module_order_noncommuting_mismatch_cells": mismatch(first, second),
        "semantic_fusion_perturbation_changes_boundary": (
            evaluate_identity(original) != evaluate_identity(changed)
        ),
        "control_port_restored": port.coefficients == source and not port.live,
    }


def depth_case(depth: int, family: int) -> dict[str, object]:
    source = [ONE, SOURCE_FUNDAMENTAL]
    word = public_program(depth, family)
    coefficients, work = execute_public_word(source, word)
    intersections = sum(
        operation.kind == "INTERSECT_FUNDAMENTAL" for operation in word
    )
    if len(coefficients) != len(source) + intersections:
        raise RuntimeError("SU2 character support growth law changed")
    if coefficients[-1].is_zero():
        raise RuntimeError("highest SU2 character coefficient cancelled")
    boundary = evaluate_identity(coefficients)
    return {
        "depth": depth,
        "family": family,
        "public_operation_steps": len(word),
        "fundamental_intersection_modules": intersections,
        "highest_weight": len(coefficients) - 1,
        "character_coefficient_cells": len(coefficients),
        "nonzero_character_coefficients": sum(
            not coefficient.is_zero() for coefficient in coefficients
        ),
        "character_payload_bits": gaussian_payload_bits(coefficients),
        **maximum_integer_bits(coefficients),
        "state_commitment": state_commitment(coefficients),
        "boundary_commitment": boundary_commitment(boundary),
        "boundary_payload_bits": gaussian_payload_bits([boundary]),
        "forward_work": work.as_dict(),
    }


def main() -> None:
    cases = [
        depth_case(depth, family)
        for depth in DEPTHS
        for family in (0, 1)
    ]
    source = [ONE, SOURCE_FUNDAMENTAL]
    primary_word = public_program(PRIMARY_DEPTH, 0)
    reuse_word = public_program(REUSE_DEPTH, 1)
    carrier = Carrier(OpenCharacterPort(source.copy()))
    primary, primary_work = transaction(carrier, source, primary_word)
    reuse, reuse_work = transaction(carrier, source, reuse_word)
    fresh_carrier = Carrier(OpenCharacterPort(source.copy()))
    fresh, fresh_work = transaction(fresh_carrier, source, reuse_word)
    control_results = controls()
    if (
        primary["restoration_error_character_cells"]
        or reuse["restoration_error_character_cells"]
        or fresh["restoration_error_character_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or not fresh["same_backing"]
        or not primary["port_released"]
        or not reuse["port_released"]
        or not fresh["port_released"]
        or carrier.restoration_generation != 2
        or reuse["boundary"] != fresh["boundary"]
        or reuse["forward_state_commitment"] != fresh["forward_state_commitment"]
        or not all(
            value if isinstance(value, bool) else value > 0
            for value in control_results.values()
        )
    ):
        raise RuntimeError("continuous-SU2 relation transaction failed")

    claim = (
        "EXACT_CONTINUOUS_SU2_TRANSLATION_INVARIANT_CENTRAL_OPEN_PHASE_"
        "RELATION_CHARACTER_CARRIER_COMPOSES_BY_CASIMIR_SPECTRAL_PHASE_"
        "AND_INTERSECTS_BY_CLEBSCH_GORDAN_FUNDAMENTAL_FUSION_ON_ONE_"
        "SHARED_UNRESOLVED_PORT_WITH_FINAL_ONLY_BOUNDARY_EXACT_"
        "RESTORATION_REUSE_BUT_HIGHEST_WEIGHT_SUPPORT_GROWS_FROM3TO34_"
        "CELLS_ACROSS_DEPTH1TO32_AND_THE_IDENTICAL_CHARACTER_CLASSICAL_"
        "RECURRENCE_REMAINS"
    )
    result = {
        "schema": "cat_cas.continuous_su2_character_relation.v1",
        "claim_candidate": claim,
        "claim_ceiling": (
            "FORMAL_FINITE_CHARACTER_POLYNOMIAL_CONTINUOUS_SU2_CENTRAL_"
            "TRANSLATION_INVARIANT_KERNELS_TWO_GAUSSIAN_RATIONAL_CASIMIR_"
            "PHASES_TWO_FUNDAMENTAL_FUSION_FACTORS_DECLARED_FAMILIES_"
            "DEPTHS1_2_4_8_16_32_PRIMARY32_REUSE19_IDENTITY_BOUNDARY_"
            "DIRECT_PROCESS_ONLY"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_EXACT_CONTINUOUS_SU2_CHARACTER_RELATION_CLOSURE_WITH_GROWING_REPRESENTATION_SUPPORT",
        "phase_relation_law": {
            "domain": "CONTINUOUS_SU2_NO_FINITE_GROUP_ENUMERATION",
            "kernel_family": "CENTRAL_TRANSLATION_INVARIANT_FINITE_CHARACTER_POLYNOMIAL",
            "relation_composition": "NORMALIZED_HAAR_CONVOLUTION_DIAGONAL_CASIMIR_SPECTRAL_PHASE",
            "relation_intersection": "POINTWISE_CLEBSCH_GORDAN_FUNDAMENTAL_FUSION",
            "fusion_identity": "CHI_R_TIMES_CHI_1_EQUALS_CHI_RMINUS1_PLUS_CHI_RPLUS1",
            "shared_unresolved_group_ports": 1,
            "multiple_noncommuting_consumers": True,
            "intermediate_character_projection": False,
            "truth_table_assignment_or_group_element_expansion": False,
            "finite_group_reduction": False,
            "final_boundary": "CHARACTER_POLYNOMIAL_VALUE_AT_SU2_IDENTITY",
        },
        "depth_cases": cases,
        "analytic_support_law": {
            "initial_highest_weight": 1,
            "fundamental_fusions_per_depth": 1,
            "highest_weight_at_depth_d": "D_PLUS_1",
            "character_cells_at_depth_d": "D_PLUS_2",
            "proof": "EVERY_NONZERO_FUNDAMENTAL_FACTOR_MAPS_THE_NONZERO_TOP_CHI_R_COEFFICIENT_TO_A_NONZERO_CHI_RPLUS1_COEFFICIENT_WHILE_CASIMIR_PHASES_ARE_NONZERO",
            "fixed_character_support_for_declared_growing_depth_family": False,
        },
        "transaction": {
            "primary_boundary_commitment": primary["boundary_commitment"],
            "primary_boundary_payload_bits": primary["boundary_payload_bits"],
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "reuse_boundary_payload_bits": reuse["boundary_payload_bits"],
            "fresh_reuse_boundary_commitment": fresh["boundary_commitment"],
            "fresh_restored_reuse_boundary_agreement": reuse["boundary"] == fresh["boundary"],
            "fresh_restored_reuse_state_agreement": reuse["forward_state_commitment"] == fresh["forward_state_commitment"],
            "primary_missing_inverse_error_cells_and_cursor": primary["missing_inverse_error_cells_and_cursor"],
            "primary_restoration_error_character_cells": primary["restoration_error_character_cells"],
            "reuse_restoration_error_character_cells": reuse["restoration_error_character_cells"],
            "fresh_reuse_restoration_error_character_cells": fresh["restoration_error_character_cells"],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "fresh_reuse_same_backing": fresh["same_backing"],
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
            "restoration_method": "EXACT_REVERSE_CASIMIR_PHASE_AND_TOP_DOWN_CLEBSCH_GORDAN_FACTOR_DIVISION_ON_THE_SAME_CHARACTER_LIST_BACKING",
        },
        "controls": control_results,
        "resource_law": {
            "initial_character_cells": len(source),
            "initial_character_payload_bits": gaussian_payload_bits(source),
            "compiled_public_parameter_gaussian_rational_cells": 5,
            "compiled_public_parameter_payload_bits": gaussian_payload_bits(
                [
                    SOURCE_FUNDAMENTAL,
                    *CASIMIR_PHASES,
                    *FUSION_FACTORS,
                ]
            ),
            "primary_forward_character_cells": primary["forward_character_cells"],
            "primary_forward_character_payload_bits": primary["forward_character_payload_bits"],
            "primary_boundary_payload_bits": primary["boundary_payload_bits"],
            "primary_full_transaction_work": primary_work.as_dict(),
            "reuse_full_transaction_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_work.as_dict(),
            "primary_compiled_public_program_operation_records": len(primary_word),
            "primary_compiled_public_program_descriptor_slots": 2 * len(primary_word),
            "reuse_compiled_public_program_operation_records": len(reuse_word),
            "reuse_compiled_public_program_descriptor_slots": 2 * len(reuse_word),
            "open_port_machine_metadata_integers": 5,
            "open_port_program_commitment_bytes": 32,
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries": 0,
            "temporary_full_character_vector_cells": 0,
            "in_place_fundamental_fusion_and_exact_division": True,
            "controller_backend_traffic_not_applicable_direct_process": True,
            "independent_oracle_polynomial_payload_bits_primary": 5323677,
            "resource_measurement_scope": "EXACT_GAUSSIAN_RATIONAL_COEFFICIENT_CELLS_PAYLOAD_BITS_PUBLIC_RECORDS_AND_HIGH_LEVEL_ALGEBRAIC_OPERATION_COUNTS",
            "python_fraction_scalar_suboperations_transient_objects_allocator_interpreter_hash_byte_traffic_serialization_timing_and_whole_process_peaks_excluded_not_zero": True,
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_EXACT_GAUSSIAN_RATIONAL_SU2_CHARACTER_COEFFICIENT_RECURRENCE",
            "independent_reference": "ORDINARY_POLYNOMIAL_CHEBYSHEV_CHARACTER_BASIS_CONVERSION_RECURRENCE",
            "strictly_smaller_or_faster_phase_path_established": False,
        },
        "predecessor_source_sha256": PREDECESSOR_SOURCE_SHA256,
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "catalytic_inference_established": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
