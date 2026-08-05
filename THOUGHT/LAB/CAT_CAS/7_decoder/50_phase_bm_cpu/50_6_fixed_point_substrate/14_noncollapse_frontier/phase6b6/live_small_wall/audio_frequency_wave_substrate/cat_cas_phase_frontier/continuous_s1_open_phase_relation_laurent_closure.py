#!/usr/bin/env python3
"""Exact continuous-S1 open phase-relation Laurent carrier diagnostic.

For a translation-invariant circle kernel K(theta_out-theta_in), normalized
relation composition is coefficientwise multiplication in the harmonic basis.
Composition with a phase rotation therefore maps c_k to lambda**k c_k.
Pointwise relation intersection is Laurent-coefficient convolution.

The accepted carrier keeps the analytic Laurent coefficients resident on one
typed open angle port.  Public rotation and one-factor intersection modules
act directly on that unresolved port; only one final phase evaluation is
projected.  Every operation is then inverted on the same backing before an
unrelated program reuses it.

The declared family exposes a strict obstruction: every intersection adds a
nonzero leading harmonic, so its finite-support Hankel rank and reduced
numerator degree grow exactly with depth.  A public-word final-boundary stream
uses only an accumulator and suffix rotation, however, and is the strongest
matched classical baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction


PORT_TYPE = 20817
DEPTHS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class Gaussian:
    real: Fraction
    imag: Fraction

    @staticmethod
    def rational(real: int | Fraction, imag: int | Fraction = 0) -> "Gaussian":
        return Gaussian(Fraction(real), Fraction(imag))

    def __add__(self, other: "Gaussian") -> "Gaussian":
        return Gaussian(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "Gaussian") -> "Gaussian":
        return Gaussian(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "Gaussian") -> "Gaussian":
        return Gaussian(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def inverse(self) -> "Gaussian":
        norm = self.real * self.real + self.imag * self.imag
        if not norm:
            raise ZeroDivisionError("zero Gaussian rational")
        return Gaussian(self.real / norm, -self.imag / norm)

    def __pow__(self, exponent: int) -> "Gaussian":
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


ZERO = Gaussian.rational(0)
ONE = Gaussian.rational(1)
SOURCE_FACTOR = Gaussian(Fraction(12, 13), Fraction(5, 13))
EVALUATION_PHASE = Gaussian(Fraction(20, 29), Fraction(21, 29))
ROTATIONS = (
    Gaussian(Fraction(3, 5), Fraction(4, 5)),
    Gaussian(Fraction(5, 13), Fraction(12, 13)),
)
INTERSECTION_FACTORS = (
    Gaussian(Fraction(8, 17), Fraction(15, 17)),
    Gaussian(Fraction(7, 25), Fraction(24, 25)),
)


@dataclass(frozen=True)
class Operation:
    kind: str
    parameter: int

    def token(self) -> str:
        return f"{self.kind}:{self.parameter}"


@dataclass
class Work:
    rotation_coefficient_multiplications: int = 0
    intersection_coefficient_multiplications: int = 0
    intersection_coefficient_additions: int = 0
    division_coefficient_multiplications: int = 0
    division_coefficient_subtractions: int = 0
    final_boundary_horner_steps: int = 0
    port_leases: int = 0
    port_releases: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0
    peak_relation_coefficient_cells: int = 0
    peak_relation_payload_bits: int = 0

    def observe(self, coefficients: list[Gaussian]) -> None:
        self.peak_relation_coefficient_cells = max(
            self.peak_relation_coefficient_cells, len(coefficients)
        )
        self.peak_relation_payload_bits = max(
            self.peak_relation_payload_bits, gaussian_payload_bits(coefficients)
        )

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass
class OpenAnglePort:
    coefficients: list[Gaussian]
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
            raise RuntimeError("open angle port already live")
        if not self.coefficients:
            raise ValueError("null relation carrier rejected")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid angle-port owner or generation")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = len(operations)
        self.public_program_commitment = program_commitment(operations)
        work.port_leases += 1
        work.observe(self.coefficients)

    def require(
        self,
        owner: int,
        operations: tuple[Operation, ...],
    ) -> None:
        if not self.live:
            raise RuntimeError("open angle port is not live")
        if owner != self.owner:
            raise PermissionError("open angle port owner mismatch")
        if program_commitment(operations) != self.public_program_commitment:
            raise ValueError("open angle port program mismatch")

    def forward(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        index: int,
        work: Work,
    ) -> None:
        self.require(owner, operations)
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
        self.require(owner, operations)
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
        point: Gaussian,
        work: Work,
    ) -> Gaussian:
        self.require(owner, operations)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal open angle port projection rejected")
        work.final_boundary_horner_steps += len(self.coefficients)
        return evaluate(self.coefficients, point)

    def release(
        self,
        owner: int,
        operations: tuple[Operation, ...],
        work: Work,
    ) -> int:
        self.require(owner, operations)
        if self.cursor:
            raise RuntimeError("open angle port released before exact inverse")
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
    port: OpenAnglePort
    restoration_generation: int = 0


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def fraction_payload_bits(value: Fraction) -> int:
    return signed_bits(value.numerator) + value.denominator.bit_length()


def gaussian_payload_bits(values: list[Gaussian] | tuple[Gaussian, ...]) -> int:
    return sum(
        fraction_payload_bits(value.real) + fraction_payload_bits(value.imag)
        for value in values
    )


def gaussian_maximum_integer_bits(values: list[Gaussian]) -> dict[str, int]:
    fractions = [
        component
        for value in values
        for component in (value.real, value.imag)
    ]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(component.numerator) for component in fractions
        ),
        "maximum_denominator_bits": max(
            component.denominator.bit_length() for component in fractions
        ),
    }


def state_commitment(coefficients: list[Gaussian]) -> str:
    payload = "|".join(value.token() for value in coefficients)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def boundary_commitment(value: Gaussian) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def program_commitment(operations: tuple[Operation, ...]) -> str:
    payload = "|".join(operation.token() for operation in operations)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def public_program(depth: int, family: int) -> tuple[Operation, ...]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    operations: list[Operation] = []
    for index in range(depth):
        rotation = Operation("COMPOSE_ROTATION", (index + family) % len(ROTATIONS))
        intersection = Operation(
            "INTERSECT_FACTOR",
            (3 * index + 2 * family + 1) % len(INTERSECTION_FACTORS),
        )
        if (index + family) % 3:
            operations.extend((rotation, intersection))
        else:
            operations.extend((intersection, rotation))
    return tuple(operations)


def apply_rotation(
    coefficients: list[Gaussian],
    rotation: Gaussian,
    work: Work,
) -> None:
    power = ONE
    for index, coefficient in enumerate(coefficients):
        if index:
            power = power * rotation
        coefficients[index] = coefficient * power
        work.rotation_coefficient_multiplications += 1


def apply_intersection(
    coefficients: list[Gaussian],
    factor: Gaussian,
    work: Work,
) -> None:
    coefficients.append(ZERO)
    for index in range(len(coefficients) - 2, -1, -1):
        coefficients[index + 1] = (
            coefficients[index + 1] + coefficients[index] * factor
        )
        work.intersection_coefficient_multiplications += 1
        work.intersection_coefficient_additions += 1


def divide_intersection(
    coefficients: list[Gaussian],
    factor: Gaussian,
    work: Work,
) -> None:
    if len(coefficients) < 2:
        raise ValueError("intersection inverse has no factor to remove")
    for index in range(1, len(coefficients) - 1):
        coefficients[index] = (
            coefficients[index] - coefficients[index - 1] * factor
        )
        work.division_coefficient_multiplications += 1
        work.division_coefficient_subtractions += 1
    expected_leading = coefficients[-2] * factor
    work.division_coefficient_multiplications += 1
    if coefficients[-1] != expected_leading:
        raise ValueError("intersection factor is not exactly divisible")
    coefficients.pop()


def apply_operation(
    coefficients: list[Gaussian],
    operation: Operation,
    inverse: bool,
    work: Work,
) -> None:
    if operation.kind == "COMPOSE_ROTATION":
        try:
            rotation = ROTATIONS[operation.parameter]
        except IndexError as error:
            raise ValueError("unknown rotation parameter") from error
        apply_rotation(
            coefficients,
            rotation.inverse() if inverse else rotation,
            work,
        )
        return
    if operation.kind == "INTERSECT_FACTOR":
        try:
            factor = INTERSECTION_FACTORS[operation.parameter]
        except IndexError as error:
            raise ValueError("unknown intersection parameter") from error
        if inverse:
            divide_intersection(coefficients, factor, work)
        else:
            apply_intersection(coefficients, factor, work)
        return
    raise ValueError("unknown open phase-relation operation")


def evaluate(coefficients: list[Gaussian], point: Gaussian) -> Gaussian:
    result = ZERO
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


def execute_public_word(
    source: list[Gaussian],
    operations: tuple[Operation, ...],
) -> tuple[list[Gaussian], Work]:
    coefficients = source.copy()
    work = Work()
    work.observe(coefficients)
    for operation in operations:
        apply_operation(coefficients, operation, False, work)
        work.forward_operations += 1
        work.observe(coefficients)
    return coefficients, work


def scalar_boundary_stream(
    operations: tuple[Operation, ...],
    point: Gaussian,
) -> tuple[Gaussian, dict[str, int]]:
    suffix_rotation = ONE
    accumulator = ONE
    factor_evaluations = 0
    rotation_updates = 0
    maximum_named_payload = gaussian_payload_bits(
        [accumulator, suffix_rotation]
    )
    for operation in reversed(operations):
        if operation.kind == "COMPOSE_ROTATION":
            updated_suffix = (
                suffix_rotation * ROTATIONS[operation.parameter]
            )
            maximum_named_payload = max(
                maximum_named_payload,
                gaussian_payload_bits(
                    [accumulator, suffix_rotation, updated_suffix]
                ),
            )
            suffix_rotation = updated_suffix
            rotation_updates += 1
        elif operation.kind == "INTERSECT_FACTOR":
            value = (
                ONE
                + INTERSECTION_FACTORS[operation.parameter]
                * suffix_rotation
                * point
            )
            updated_accumulator = accumulator * value
            maximum_named_payload = max(
                maximum_named_payload,
                gaussian_payload_bits(
                    [
                        accumulator,
                        suffix_rotation,
                        value,
                        updated_accumulator,
                    ]
                ),
            )
            accumulator = updated_accumulator
            factor_evaluations += 1
        else:
            raise ValueError("unknown scalar-stream operation")
    source_value = ONE + SOURCE_FACTOR * suffix_rotation * point
    result = source_value * accumulator
    maximum_named_payload = max(
        maximum_named_payload,
        gaussian_payload_bits(
            [accumulator, suffix_rotation, source_value, result]
        ),
    )
    return result, {
        "resident_gaussian_rational_cells": 2,
        "warm_named_gaussian_rational_cell_peak": 4,
        "factor_evaluations": factor_evaluations,
        "rotation_updates": rotation_updates,
        "resident_payload_bits_before_output": gaussian_payload_bits(
            [accumulator, suffix_rotation]
        ),
        "warm_named_payload_bit_peak": maximum_named_payload,
        "final_output_payload_bits": gaussian_payload_bits([result]),
    }


def coefficient_mismatch(
    left: list[Gaussian],
    right: list[Gaussian],
) -> int:
    common = sum(
        first != second
        for first, second in zip(left, right, strict=False)
    )
    return common + abs(len(left) - len(right))


def transaction(
    carrier: Carrier,
    source: list[Gaussian],
    operations: tuple[Operation, ...],
) -> tuple[dict[str, object], Work]:
    backing = id(carrier.port.coefficients)
    generation = carrier.restoration_generation + 1
    owner = 208000 + generation
    work = Work()
    carrier.port.lease(owner, generation, operations, work)
    for index in range(len(operations)):
        carrier.port.forward(owner, operations, index, work)
    projected = carrier.port.project_final(
        owner, operations, EVALUATION_PHASE, work
    )
    forward_commitment = state_commitment(carrier.port.coefficients)
    forward_cells = len(carrier.port.coefficients)
    forward_payload = gaussian_payload_bits(carrier.port.coefficients)
    missing_inverse_error = coefficient_mismatch(
        carrier.port.coefficients, source
    ) + carrier.port.cursor
    for index in range(len(operations) - 1, -1, -1):
        carrier.port.inverse(owner, operations, index, work)
    restored_generation = carrier.port.release(owner, operations, work)
    carrier.restoration_generation = restored_generation
    restoration_error = coefficient_mismatch(
        carrier.port.coefficients, source
    )
    return (
        {
            "boundary": projected,
            "boundary_commitment": boundary_commitment(projected),
            "forward_state_commitment": forward_commitment,
            "forward_relation_coefficient_cells": forward_cells,
            "forward_relation_payload_bits": forward_payload,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_relation_cells": restoration_error,
            "same_backing": id(carrier.port.coefficients) == backing,
            "port_released": not carrier.port.live,
            "restoration_generation": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        work,
    )


def controls() -> dict[str, object]:
    source = [ONE, SOURCE_FACTOR]
    word = public_program(2, 0)
    work = Work()
    port = OpenAnglePort(source.copy())
    port.lease(7001, 1, word, work)
    wrong_owner = False
    wrong_type = False
    premature_projection = False
    reordered_inverse = False
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
        port.project_final(7001, word, EVALUATION_PHASE, work)
    except PermissionError:
        premature_projection = True
    for index in range(1, len(word)):
        port.forward(7001, word, index, work)
    missing_inverse_detected = port.cursor != 0
    try:
        port.inverse(7001, word, len(word) - 2, work)
    except ValueError:
        reordered_inverse = True
    for index in range(len(word) - 1, -1, -1):
        port.inverse(7001, word, index, work)
    port.release(7001, word, work)

    null_carrier = False
    try:
        OpenAnglePort([]).lease(1, 1, word, Work())
    except ValueError:
        null_carrier = True

    initial = source.copy()
    rotate_then_intersect = initial.copy()
    intersect_then_rotate = initial.copy()
    local_work = Work()
    apply_operation(
        rotate_then_intersect, Operation("COMPOSE_ROTATION", 0), False, local_work
    )
    apply_operation(
        rotate_then_intersect, Operation("INTERSECT_FACTOR", 0), False, local_work
    )
    apply_operation(
        intersect_then_rotate, Operation("INTERSECT_FACTOR", 0), False, local_work
    )
    apply_operation(
        intersect_then_rotate, Operation("COMPOSE_ROTATION", 0), False, local_work
    )
    noncommuting_mismatch = coefficient_mismatch(
        rotate_then_intersect, intersect_then_rotate
    )

    original = public_program(4, 0)
    perturbed = list(original)
    for index, operation in enumerate(perturbed):
        if operation.kind == "INTERSECT_FACTOR":
            perturbed[index] = Operation(
                operation.kind,
                (operation.parameter + 1) % len(INTERSECTION_FACTORS),
            )
            break
    original_boundary, _ = scalar_boundary_stream(
        original, EVALUATION_PHASE
    )
    perturbed_boundary, _ = scalar_boundary_stream(
        tuple(perturbed), EVALUATION_PHASE
    )
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature_projection,
        "missing_inverse_detected": missing_inverse_detected,
        "reordered_inverse_rejected": reordered_inverse,
        "null_carrier_rejected": null_carrier,
        "module_order_noncommuting_mismatch_cells": noncommuting_mismatch,
        "semantic_perturbation_changes_boundary": (
            original_boundary != perturbed_boundary
        ),
        "control_port_restored": port.coefficients == source and not port.live,
    }


def depth_case(depth: int, family: int) -> dict[str, object]:
    source = [ONE, SOURCE_FACTOR]
    word = public_program(depth, family)
    coefficients, work = execute_public_word(source, word)
    boundary = evaluate(coefficients, EVALUATION_PHASE)
    scalar_boundary, scalar_work = scalar_boundary_stream(
        word, EVALUATION_PHASE
    )
    if boundary != scalar_boundary:
        raise RuntimeError("sparse Laurent and scalar boundary streams differ")
    intersections = sum(
        operation.kind == "INTERSECT_FACTOR" for operation in word
    )
    degree = len(coefficients) - 1
    if degree != intersections + len(source) - 1:
        raise RuntimeError("Laurent degree growth law changed")
    if coefficients[-1].is_zero():
        raise RuntimeError("leading harmonic unexpectedly cancelled")
    maximum = gaussian_maximum_integer_bits(coefficients)
    return {
        "depth": depth,
        "family": family,
        "public_operation_steps": len(word),
        "intersection_modules": intersections,
        "reduced_rational_numerator_degree": degree,
        "reduced_rational_denominator_degree": 0,
        "finite_support_hankel_rank": degree + 1,
        "relation_coefficient_cells": len(coefficients),
        "nonzero_harmonic_coefficients": sum(
            not coefficient.is_zero() for coefficient in coefficients
        ),
        "relation_payload_bits": gaussian_payload_bits(coefficients),
        **maximum,
        "state_commitment": state_commitment(coefficients),
        "boundary_commitment": boundary_commitment(boundary),
        "scalar_boundary_stream": scalar_work,
        "forward_work": work.as_dict(),
    }


def main() -> None:
    cases = [
        depth_case(depth, family)
        for depth in DEPTHS
        for family in (0, 1)
    ]
    source = [ONE, SOURCE_FACTOR]
    primary_word = public_program(64, 0)
    reuse_word = public_program(37, 1)
    carrier = Carrier(OpenAnglePort(source.copy()))
    primary, primary_work = transaction(carrier, source, primary_word)
    reuse, reuse_work = transaction(carrier, source, reuse_word)
    fresh = Carrier(OpenAnglePort(source.copy()))
    fresh_reuse, fresh_reuse_work = transaction(
        fresh, source, reuse_word
    )
    control_results = controls()

    if (
        primary["restoration_error_relation_cells"]
        or reuse["restoration_error_relation_cells"]
        or fresh_reuse["restoration_error_relation_cells"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or not fresh_reuse["same_backing"]
        or not primary["port_released"]
        or not reuse["port_released"]
        or not fresh_reuse["port_released"]
        or carrier.restoration_generation != 2
        or reuse["boundary"] != fresh_reuse["boundary"]
        or reuse["forward_state_commitment"]
        != fresh_reuse["forward_state_commitment"]
        or not all(
            value if isinstance(value, bool) else value > 0
            for value in control_results.values()
        )
    ):
        raise RuntimeError("continuous-S1 relation transaction failed")

    claim = (
        "EXACT_CONTINUOUS_S1_TRANSLATION_INVARIANT_OPEN_PHASE_RELATION_"
        "LAURENT_CARRIER_COMPOSES_BY_HARMONIC_HADAMARD_ROTATION_AND_"
        "INTERSECTS_BY_LAURENT_CONVOLUTION_ON_ONE_SHARED_UNRESOLVED_PORT_"
        "WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_REUSE_BUT_THE_"
        "DECLARED_DEPTH_D_FAMILY_HAS_REDUCED_NUMERATOR_DEGREE_AND_FINITE_"
        "SUPPORT_HANKEL_RANK_DPLUS2_WHILE_A_MATCHED_PUBLIC_WORD_FINAL_"
        "BOUNDARY_STREAM_USES_TWO_RESIDENT_AND_FOUR_WARM_NAMED_GAUSSIAN_"
        "RATIONAL_CELLS"
    )
    result = {
        "claim_candidate": claim,
        "claim_ceiling": (
            "ANALYTIC_NONNEGATIVE_HARMONIC_TRANSLATION_INVARIANT_S1_"
            "KERNELS_WITH_ONE_PUBLIC_GAUSSIAN_RATIONAL_SOURCE_FACTOR_TWO_"
            "ROTATIONS_TWO_LINEAR_INTERSECTION_FACTORS_DECLARED_PROGRAM_"
            "FAMILIES_DEPTHS1_TO64_PRIMARY64_REUSE37_DIRECT_PROCESS_ONLY"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_EXACT_S1_RELATION_CLOSURE_WITH_GROWING_HARMONIC_RANK",
        "phase_relation_law": {
            "domain": "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING",
            "kernel_coordinate": "Z_EQUALS_EXP_I_THETA",
            "relation_composition": "HARMONIC_HADAMARD_PUBLIC_ROTATION_MULTIPLIER",
            "relation_intersection": "LAURENT_COEFFICIENT_CONVOLUTION",
            "shared_unresolved_angle_ports": 1,
            "multiple_noncommuting_consumers": True,
            "intermediate_coefficient_projection": False,
            "truth_table_or_assignment_expansion": False,
            "finite_cyclic_group_reduction": False,
            "source_factor": SOURCE_FACTOR.public_json(),
            "evaluation_phase": EVALUATION_PHASE.public_json(),
        },
        "depth_cases": cases,
        "analytic_rank_law": {
            "initial_degree": 1,
            "intersection_factors_per_depth": 1,
            "degree_at_depth_d": "D_PLUS_1",
            "coefficient_cells_at_depth_d": "D_PLUS_2",
            "finite_support_hankel_rank_at_depth_d": "D_PLUS_2",
            "proof": (
                "EVERY_NONZERO_LINEAR_INTERSECTION_FACTOR_ADDS_ONE_NONZERO_"
                "LEADING_HARMONIC_AND_ROTATION_MULTIPLIES_IT_BY_A_NONZERO_"
                "UNIT_THE_DPLUS2_SQUARE_FINITE_SEQUENCE_HANKEL_MATRIX_IS_"
                "ANTI_TRIANGULAR_WITH_NONZERO_FINAL_COEFFICIENT"
            ),
            "fixed_bounded_degree_reduced_rational_chart_for_declared_family": False,
            "procedural_public_word_rematerialization_remains": True,
        },
        "transaction": {
            "primary_boundary": primary["boundary"].public_json(),
            "primary_boundary_commitment": primary["boundary_commitment"],
            "reuse_boundary": reuse["boundary"].public_json(),
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "fresh_reuse_boundary_commitment": fresh_reuse[
                "boundary_commitment"
            ],
            "fresh_restored_reuse_boundary_agreement": (
                reuse["boundary"] == fresh_reuse["boundary"]
            ),
            "fresh_restored_reuse_state_agreement": (
                reuse["forward_state_commitment"]
                == fresh_reuse["forward_state_commitment"]
            ),
            "primary_missing_inverse_error_cells_and_cursor": primary[
                "missing_inverse_error_cells_and_cursor"
            ],
            "primary_restoration_error_relation_cells": primary[
                "restoration_error_relation_cells"
            ],
            "reuse_restoration_error_relation_cells": reuse[
                "restoration_error_relation_cells"
            ],
            "fresh_reuse_restoration_error_relation_cells": fresh_reuse[
                "restoration_error_relation_cells"
            ],
            "primary_same_backing": primary["same_backing"],
            "reuse_same_backing": reuse["same_backing"],
            "fresh_reuse_same_backing": fresh_reuse["same_backing"],
            "restoration_generation_after_reuse": (
                carrier.restoration_generation
            ),
            "baseline_reload_used": False,
            "restoration_method": (
                "EXACT_REVERSE_PUBLIC_ROTATION_AND_LINEAR_FACTOR_DIVISION_"
                "ON_THE_SAME_LAURENT_COEFFICIENT_BACKING"
            ),
        },
        "controls": control_results,
        "resource_law": {
            "primary_forward_relation_coefficient_cells": primary[
                "forward_relation_coefficient_cells"
            ],
            "primary_forward_relation_payload_bits": primary[
                "forward_relation_payload_bits"
            ],
            "primary_full_transaction_work": primary_work.as_dict(),
            "reuse_full_transaction_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_reuse_work.as_dict(),
            "primary_compiled_public_program_operation_records": len(
                primary_word
            ),
            "primary_compiled_public_program_descriptor_slots": 2 * len(
                primary_word
            ),
            "reuse_compiled_public_program_operation_records": len(
                reuse_word
            ),
            "reuse_compiled_public_program_descriptor_slots": 2 * len(
                reuse_word
            ),
            "open_port_machine_metadata_integers": 5,
            "open_port_program_commitment_bytes": 32,
            "retained_inverse_history_entries": 0,
            "additional_retained_factor_plan_entries_beyond_public_program": 0,
            "temporary_full_coefficient_vector_cells": 0,
            "temporary_gaussian_rational_cell_peak": 4,
            "in_place_linear_factor_multiply_and_divide": True,
            "python_fraction_object_allocator_interpreter_timing_and_whole_process_peaks_excluded": True,
        },
        "matched_classical_baselines": [
            "IDENTICAL_SPARSE_GAUSSIAN_RATIONAL_LAURENT_COEFFICIENT_RECURRENCE_FOR_FULL_RELATION",
            "TWO_RESIDENT_FOUR_WARM_NAMED_GAUSSIAN_RATIONAL_CELL_REVERSE_PUBLIC_WORD_FINAL_BOUNDARY_STREAM_WITH_LINEAR_WORK",
        ],
        "preserved_subclaims": [
            "M207_ROTOR_ROUTE_CEILING",
            "EXACT_TRANSLATION_INVARIANT_S1_COMPOSITION_INTERSECTION_DUALITY",
            "ONE_SHARED_UNRESOLVED_ANGLE_PORT",
            "FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE",
        ],
        "rejected_interpretations": [
            "FIXED_BOUNDED_DEGREE_RATIONAL_CHART_FOR_THE_DECLARED_INFINITE_DEPTH_FAMILY",
            "GENERAL_S1_RELATION_ALGEBRA",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "CATALYTIC_INFERENCE",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
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
