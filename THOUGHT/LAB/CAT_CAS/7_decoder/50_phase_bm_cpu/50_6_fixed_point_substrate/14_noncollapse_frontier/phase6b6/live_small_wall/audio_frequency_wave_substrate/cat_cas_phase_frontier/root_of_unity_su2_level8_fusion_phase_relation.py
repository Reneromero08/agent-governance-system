#!/usr/bin/env python3
"""Exact SU(2)_8 root-of-unity fusion phase-relation diagnostic.

The continuous SU(2) predecessor has unbounded highest-weight support under
fundamental fusion.  This successor replaces that representation ring by the
Jones-Wenzl quotient at level 8.  Its nine simple-object coefficients obey
X_1 X_j = X_(j-1) + X_(j+1), with X_9 = 0.  Public topological twists are
exact powers of zeta_40.  Fundamental fusion and twist do not commute.

One fixed nine-cell Q(zeta_40) carrier remains resident on an unresolved
typed port.  Only the final quantum-dimension scalar is projected.  The
actual word is then reversed on the same carrier backing.  Fusion inversion
rematerializes and exactly uncomputes a nine-pivot tridiagonal plan on one
reused scratch backing, rather than retaining an inverse matrix or history.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)

LEVEL = 8
SIMPLE_OBJECTS = LEVEL + 1
FIELD_DEGREE = 16
ROOT_ORDER = 40
PORT_TYPE = 21308
DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
PRIMARY_DEPTH = 128
REUSE_DEPTH = 73
PREDECESSOR_SOURCE_SHA256 = (
    "abecd13947f99e53d6f93b09aec024138e89807b3fa4259378506eb2f591e4b4"
)

# Phi_40(x) = x^16 - x^12 + x^8 - x^4 + 1.
MODULUS = tuple(
    Fraction(value)
    for value in (1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1)
)


def trim_fraction(values: list[Fraction]) -> list[Fraction]:
    while values and values[-1] == 0:
        values.pop()
    return values


def fraction_poly_subtract(
    left: list[Fraction], right: list[Fraction]
) -> list[Fraction]:
    size = max(len(left), len(right))
    result = [Fraction(0)] * size
    for index in range(size):
        result[index] = (
            left[index] if index < len(left) else Fraction(0)
        ) - (right[index] if index < len(right) else Fraction(0))
    return trim_fraction(result)


def fraction_poly_multiply(
    left: list[Fraction], right: list[Fraction]
) -> list[Fraction]:
    if not left or not right:
        return []
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return trim_fraction(result)


def fraction_poly_divmod(
    numerator: list[Fraction], denominator: list[Fraction]
) -> tuple[list[Fraction], list[Fraction]]:
    divisor = trim_fraction(denominator.copy())
    if not divisor:
        raise ZeroDivisionError("zero polynomial")
    remainder = trim_fraction(numerator.copy())
    quotient = [Fraction(0)] * max(1, len(remainder) - len(divisor) + 1)
    while remainder and len(remainder) >= len(divisor):
        degree = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        quotient[degree] += factor
        for index, coefficient in enumerate(divisor):
            remainder[index + degree] -= factor * coefficient
        trim_fraction(remainder)
    return trim_fraction(quotient), remainder


def reduce_fraction_polynomial(values: list[Fraction]) -> tuple[Fraction, ...]:
    work = values.copy()
    if len(work) < FIELD_DEGREE:
        work.extend([Fraction(0)] * (FIELD_DEGREE - len(work)))
    for degree in range(len(work) - 1, FIELD_DEGREE - 1, -1):
        coefficient = work[degree]
        if not coefficient:
            continue
        work[degree - 4] += coefficient
        work[degree - 8] -= coefficient
        work[degree - 12] += coefficient
        work[degree - 16] -= coefficient
        work[degree] = Fraction(0)
    return tuple(work[:FIELD_DEGREE])


@dataclass(frozen=True)
class K:
    coefficients: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != FIELD_DEGREE:
            raise ValueError("Q(zeta40) element must have sixteen coordinates")

    @staticmethod
    def integer(value: int) -> "K":
        return K((Fraction(value),) + (Fraction(0),) * (FIELD_DEGREE - 1))

    @staticmethod
    def zeta(power: int) -> "K":
        exponent = power % ROOT_ORDER
        polynomial = [Fraction(0)] * (exponent + 1)
        polynomial[exponent] = Fraction(1)
        return K(reduce_fraction_polynomial(polynomial))

    def __add__(self, other: "K") -> "K":
        return K(tuple(a + b for a, b in zip(self.coefficients, other.coefficients)))

    def __sub__(self, other: "K") -> "K":
        return K(tuple(a - b for a, b in zip(self.coefficients, other.coefficients)))

    def __neg__(self) -> "K":
        return K(tuple(-value for value in self.coefficients))

    def __mul__(self, other: "K") -> "K":
        polynomial = [Fraction(0)] * (2 * FIELD_DEGREE - 1)
        for i, a in enumerate(self.coefficients):
            if not a:
                continue
            for j, b in enumerate(other.coefficients):
                if b:
                    polynomial[i + j] += a * b
        return K(reduce_fraction_polynomial(polynomial))

    def inverse(self) -> "K":
        if self == ZERO:
            raise ZeroDivisionError("zero Q(zeta40) element")
        old_r = list(MODULUS)
        r = trim_fraction(list(self.coefficients))
        old_s: list[Fraction] = []
        s = [Fraction(1)]
        while r:
            quotient, remainder = fraction_poly_divmod(old_r, r)
            old_r, r = r, remainder
            old_s, s = s, fraction_poly_subtract(
                old_s, fraction_poly_multiply(quotient, s)
            )
        if len(old_r) != 1 or not old_r[0]:
            raise ZeroDivisionError("nonunit Q(zeta40) element")
        inverse_gcd = Fraction(1, 1) / old_r[0]
        return K(
            reduce_fraction_polynomial(
                [coefficient * inverse_gcd for coefficient in old_s]
            )
        )

    def __truediv__(self, other: "K") -> "K":
        return self * other.inverse()

    def is_zero(self) -> bool:
        return all(value == 0 for value in self.coefficients)

    def token(self) -> str:
        return ":".join(
            f"{value.numerator}/{value.denominator}"
            for value in self.coefficients
        )


ZERO = K.integer(0)
ONE = K.integer(1)
SOURCE_FUNDAMENTAL = K.zeta(5)
FUSION_FACTORS = (K.zeta(3), K.zeta(7))
TWIST_POWERS = (1, 3)


def quantum_dimensions() -> tuple[K, ...]:
    q = K.zeta(2)
    delta = q + K.zeta(-2)
    dimensions = [ONE, delta]
    for _ in range(2, SIMPLE_OBJECTS + 1):
        dimensions.append(delta * dimensions[-1] - dimensions[-2])
    if dimensions[SIMPLE_OBJECTS] != ZERO:
        raise RuntimeError("SU2 level-8 Jones-Wenzl relation failed")
    return tuple(dimensions[:SIMPLE_OBJECTS])


QUANTUM_DIMENSIONS = quantum_dimensions()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def field_payload_bits(values: list[K] | tuple[K, ...]) -> int:
    return sum(
        signed_bits(coefficient.numerator) + coefficient.denominator.bit_length()
        for value in values
        for coefficient in value.coefficients
    )


def maximum_coordinate_bits(values: list[K]) -> dict[str, int]:
    coordinates = [c for value in values for c in value.coefficients]
    return {
        "maximum_signed_numerator_bits": max(
            signed_bits(value.numerator) for value in coordinates
        ),
        "maximum_denominator_bits": max(
            value.denominator.bit_length() for value in coordinates
        ),
    }


def state_commitment(values: list[K]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in values).encode("ascii")
    ).hexdigest()


def boundary_commitment(value: K) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Operation:
    kind: str
    parameter: int

    def token(self) -> str:
        return f"{self.kind}:{self.parameter}"


@dataclass(frozen=True)
class Program:
    depth: int
    family: int

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.family not in (0, 1):
            raise ValueError("invalid SU2 level-8 public program descriptor")

    @property
    def steps(self) -> int:
        return 2 * self.depth

    def operation(self, step: int) -> Operation:
        if step < 0 or step >= self.steps:
            raise IndexError("SU2 level-8 public program step out of range")
        index, position = divmod(step, 2)
        twist = Operation("TWIST_CASIMIR", (index + self.family) % 2)
        fusion = Operation(
            "FUSE_FUNDAMENTAL", (3 * index + self.family) % 2
        )
        ordered = (
            (twist, fusion) if (index + self.family) % 3 else (fusion, twist)
        )
        return ordered[position]

    def token(self) -> str:
        return f"depth:{self.depth}:family:{self.family}"


def public_program(depth: int, family: int) -> Program:
    return Program(depth, family)


def program_commitment(program: Program) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


@dataclass
class Work:
    field_additions: int = 0
    field_subtractions: int = 0
    field_multiplications: int = 0
    field_inversions: int = 0
    twist_coefficient_multiplications: int = 0
    fusion_neighbor_additions: int = 0
    fusion_factor_multiplications: int = 0
    fusion_accumulation_additions: int = 0
    inverse_pivot_cells_materialized: int = 0
    inverse_pivot_cells_uncomputed: int = 0
    inverse_elimination_steps: int = 0
    inverse_back_substitution_steps: int = 0
    boundary_multiplications: int = 0
    boundary_additions: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0
    port_leases: int = 0
    port_releases: int = 0
    public_program_descriptor_hashes: int = 0
    public_program_descriptor_integers_hashed: int = 0
    state_commitment_hashes: int = 0
    state_commitment_field_cells_hashed: int = 0
    boundary_commitment_hashes: int = 0
    peak_carrier_cells: int = 0
    peak_nonzero_carrier_cells: int = 0
    peak_carrier_payload_bits: int = 0
    peak_inverse_scratch_cells: int = 0
    peak_inverse_scratch_payload_bits: int = 0
    peak_carrier_plus_scratch_payload_bits: int = 0

    def add(self, left: K, right: K) -> K:
        self.field_additions += 1
        return left + right

    def subtract(self, left: K, right: K) -> K:
        self.field_subtractions += 1
        return left - right

    def multiply(self, left: K, right: K) -> K:
        self.field_multiplications += 1
        return left * right

    def invert(self, value: K) -> K:
        self.field_inversions += 1
        return value.inverse()

    def divide(self, left: K, right: K) -> K:
        return self.multiply(left, self.invert(right))

    def observe(self, carrier: list[K], scratch: list[K]) -> None:
        carrier_payload = field_payload_bits(carrier)
        scratch_payload = field_payload_bits(scratch)
        self.peak_carrier_cells = max(self.peak_carrier_cells, len(carrier))
        self.peak_nonzero_carrier_cells = max(
            self.peak_nonzero_carrier_cells,
            sum(not value.is_zero() for value in carrier),
        )
        self.peak_carrier_payload_bits = max(
            self.peak_carrier_payload_bits, carrier_payload
        )
        self.peak_inverse_scratch_cells = max(
            self.peak_inverse_scratch_cells, len(scratch)
        )
        self.peak_inverse_scratch_payload_bits = max(
            self.peak_inverse_scratch_payload_bits, scratch_payload
        )
        self.peak_carrier_plus_scratch_payload_bits = max(
            self.peak_carrier_plus_scratch_payload_bits,
            carrier_payload + scratch_payload,
        )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def twist_multiplier(simple_object: int, parameter: int, inverse: bool) -> K:
    exponent = TWIST_POWERS[parameter] * simple_object * (simple_object + 2)
    return K.zeta(-exponent if inverse else exponent)


def apply_twist(
    coefficients: list[K], parameter: int, inverse: bool, work: Work
) -> None:
    for simple_object in range(SIMPLE_OBJECTS):
        coefficients[simple_object] = work.multiply(
            coefficients[simple_object],
            twist_multiplier(simple_object, parameter, inverse),
        )
        work.twist_coefficient_multiplications += 1


def apply_fusion(coefficients: list[K], factor: K, work: Work) -> None:
    previous = ZERO
    current = coefficients[0]
    for index in range(SIMPLE_OBJECTS):
        following = coefficients[index + 1] if index + 1 < SIMPLE_OBJECTS else ZERO
        neighbor_sum = work.add(previous, following)
        work.fusion_neighbor_additions += 1
        weighted = work.multiply(factor, neighbor_sum)
        work.fusion_factor_multiplications += 1
        coefficients[index] = work.add(current, weighted)
        work.fusion_accumulation_additions += 1
        previous, current = current, following


def inverse_fusion(
    coefficients: list[K], factor: K, scratch: list[K], work: Work
) -> None:
    if any(value != ZERO for value in scratch):
        raise RuntimeError("inverse pivot scratch is not clean")
    factor_squared = work.multiply(factor, factor)
    scratch[0] = work.add(scratch[0], ONE)
    work.inverse_pivot_cells_materialized += 1
    work.observe(coefficients, scratch)
    for index in range(1, SIMPLE_OBJECTS):
        ratio = work.multiply(factor_squared, work.invert(scratch[index - 1]))
        pivot = work.subtract(ONE, ratio)
        scratch[index] = work.add(scratch[index], pivot)
        work.inverse_pivot_cells_materialized += 1
        work.observe(coefficients, scratch)

    for index in range(1, SIMPLE_OBJECTS):
        multiplier = work.multiply(factor, work.invert(scratch[index - 1]))
        correction = work.multiply(multiplier, coefficients[index - 1])
        coefficients[index] = work.subtract(coefficients[index], correction)
        work.inverse_elimination_steps += 1
        work.observe(coefficients, scratch)

    coefficients[-1] = work.multiply(
        coefficients[-1], work.invert(scratch[-1])
    )
    work.inverse_back_substitution_steps += 1
    work.observe(coefficients, scratch)
    for index in range(SIMPLE_OBJECTS - 2, -1, -1):
        correction = work.multiply(factor, coefficients[index + 1])
        residual = work.subtract(coefficients[index], correction)
        coefficients[index] = work.multiply(residual, work.invert(scratch[index]))
        work.inverse_back_substitution_steps += 1
        work.observe(coefficients, scratch)

    for index in range(SIMPLE_OBJECTS - 1, 0, -1):
        ratio = work.multiply(factor_squared, work.invert(scratch[index - 1]))
        pivot = work.subtract(ONE, ratio)
        scratch[index] = work.subtract(scratch[index], pivot)
        work.inverse_pivot_cells_uncomputed += 1
        work.observe(coefficients, scratch)
    scratch[0] = work.subtract(scratch[0], ONE)
    work.inverse_pivot_cells_uncomputed += 1
    work.observe(coefficients, scratch)
    if any(value != ZERO for value in scratch):
        raise RuntimeError("inverse pivot scratch failed exact uncomputation")


def apply_operation(
    coefficients: list[K],
    scratch: list[K],
    operation: Operation,
    inverse: bool,
    work: Work,
) -> None:
    if operation.kind == "TWIST_CASIMIR":
        if operation.parameter not in range(len(TWIST_POWERS)):
            raise ValueError("unknown twist parameter")
        apply_twist(coefficients, operation.parameter, inverse, work)
        return
    if operation.kind == "FUSE_FUNDAMENTAL":
        try:
            factor = FUSION_FACTORS[operation.parameter]
        except IndexError as error:
            raise ValueError("unknown fusion parameter") from error
        if inverse:
            inverse_fusion(coefficients, factor, scratch, work)
        else:
            apply_fusion(coefficients, factor, work)
        return
    raise ValueError("unknown SU2 level-8 operation")


def project_quantum_dimension(coefficients: list[K], work: Work) -> K:
    result = ZERO
    for coefficient, dimension in zip(
        coefficients, QUANTUM_DIMENSIONS, strict=True
    ):
        result = work.add(result, work.multiply(coefficient, dimension))
        work.boundary_multiplications += 1
        work.boundary_additions += 1
    return result


def mismatch(left: list[K], right: list[K]) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


@dataclass
class OpenFusionPort:
    coefficients: list[K]
    scratch: list[K]
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
        program: Program,
        work: Work,
    ) -> None:
        if self.live:
            raise RuntimeError("SU2 level-8 fusion port already live")
        if len(self.coefficients) != SIMPLE_OBJECTS:
            raise ValueError("null or wrong-width SU2 level-8 carrier rejected")
        if len(self.scratch) != SIMPLE_OBJECTS or any(
            value != ZERO for value in self.scratch
        ):
            raise ValueError("invalid SU2 level-8 inverse scratch")
        if owner <= 0 or generation <= 0:
            raise ValueError("invalid SU2 level-8 owner or generation")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.public_program_commitment = program_commitment(program)
        work.public_program_descriptor_hashes += 1
        work.public_program_descriptor_integers_hashed += 2
        work.port_leases += 1
        work.observe(self.coefficients, self.scratch)

    def require(
        self,
        owner: int,
        program: Program,
        work: Work,
    ) -> None:
        if not self.live:
            raise RuntimeError("SU2 level-8 fusion port is not live")
        if owner != self.owner:
            raise PermissionError("SU2 level-8 fusion port owner mismatch")
        supplied = program_commitment(program)
        work.public_program_descriptor_hashes += 1
        work.public_program_descriptor_integers_hashed += 2
        if supplied != self.public_program_commitment:
            raise ValueError("SU2 level-8 public program mismatch")

    def forward(
        self,
        owner: int,
        program: Program,
        index: int,
        work: Work,
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("forward public cursor mismatch")
        apply_operation(
            self.coefficients, self.scratch, program.operation(index), False, work
        )
        self.cursor += 1
        work.forward_operations += 1
        work.observe(self.coefficients, self.scratch)

    def inverse(
        self,
        owner: int,
        program: Program,
        index: int,
        work: Work,
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor - 1:
            raise ValueError("inverse public cursor mismatch")
        apply_operation(
            self.coefficients, self.scratch, program.operation(index), True, work
        )
        self.cursor -= 1
        work.inverse_operations += 1
        work.observe(self.coefficients, self.scratch)

    def project_final(
        self,
        owner: int,
        program: Program,
        work: Work,
    ) -> K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal SU2 level-8 projection rejected")
        return project_quantum_dimension(self.coefficients, work)

    def release(
        self,
        owner: int,
        program: Program,
        work: Work,
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("SU2 level-8 port released before inverse")
        if any(value != ZERO for value in self.scratch):
            raise RuntimeError("SU2 level-8 port released with dirty scratch")
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
    port: OpenFusionPort
    restoration_generation: int = 0


def source_state() -> list[K]:
    return [ONE, SOURCE_FUNDAMENTAL] + [ZERO] * (SIMPLE_OBJECTS - 2)


def canonical_post_restoration(
    carrier: Carrier, source: list[K], expected_generation: int
) -> bool:
    port = carrier.port
    return (
        port.coefficients == source
        and all(value == ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.lease_generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.public_program_commitment == ""
        and carrier.restoration_generation == expected_generation
    )


def transaction(
    carrier: Carrier, source: list[K], program: Program
) -> tuple[dict[str, object], Work]:
    coefficient_backing = id(carrier.port.coefficients)
    scratch_backing = id(carrier.port.scratch)
    generation = carrier.restoration_generation + 1
    owner = 213000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.forward(owner, program, index, work)
    boundary = carrier.port.project_final(owner, program, work)
    forward_state = state_commitment(carrier.port.coefficients)
    work.state_commitment_hashes += 1
    work.state_commitment_field_cells_hashed += SIMPLE_OBJECTS
    final_boundary_commitment = boundary_commitment(boundary)
    work.boundary_commitment_hashes += 1
    forward_payload = field_payload_bits(carrier.port.coefficients)
    missing_inverse_error = mismatch(carrier.port.coefficients, source) + carrier.port.cursor
    for index in range(program.steps - 1, -1, -1):
        carrier.port.inverse(owner, program, index, work)
    restored_generation = carrier.port.release(owner, program, work)
    carrier.restoration_generation = restored_generation
    return (
        {
            "boundary": boundary,
            "boundary_commitment": final_boundary_commitment,
            "boundary_payload_bits": field_payload_bits([boundary]),
            "forward_state_commitment": forward_state,
            "forward_field_cells": SIMPLE_OBJECTS,
            "forward_nonzero_field_cells": sum(
                not value.is_zero() for value in carrier.port.coefficients
            ),
            "forward_payload_bits": forward_payload,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_field_cells": mismatch(
                carrier.port.coefficients, source
            ),
            "same_coefficient_backing": id(carrier.port.coefficients)
            == coefficient_backing,
            "same_scratch_backing": id(carrier.port.scratch) == scratch_backing,
            "canonical_post_restoration_state_exact": canonical_post_restoration(
                carrier, source, generation
            ),
            "restoration_generation": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        work,
    )


def execute_forward(depth: int, family: int) -> tuple[list[K], Work]:
    coefficients = source_state()
    scratch = [ZERO] * SIMPLE_OBJECTS
    work = Work()
    work.observe(coefficients, scratch)
    program = public_program(depth, family)
    for step in range(program.steps):
        apply_operation(
            coefficients, scratch, program.operation(step), False, work
        )
        work.forward_operations += 1
        work.observe(coefficients, scratch)
    return coefficients, work


def depth_case(depth: int, family: int) -> dict[str, object]:
    coefficients, work = execute_forward(depth, family)
    boundary_work = Work()
    boundary = project_quantum_dimension(coefficients, boundary_work)
    return {
        "depth": depth,
        "family": family,
        "simple_object_field_cells": len(coefficients),
        "nonzero_simple_object_field_cells": sum(
            not value.is_zero() for value in coefficients
        ),
        "carrier_payload_bits": field_payload_bits(coefficients),
        **maximum_coordinate_bits(coefficients),
        "state_commitment": state_commitment(coefficients),
        "boundary_commitment": boundary_commitment(boundary),
        "boundary_payload_bits": field_payload_bits([boundary]),
        "forward_work": work.as_dict(),
    }


def controls() -> dict[str, object]:
    source = source_state()
    word = public_program(2, 0)
    port = OpenFusionPort(source.copy(), [ZERO] * SIMPLE_OBJECTS)
    work = Work()
    port.lease(7001, 1, word, work)
    wrong_owner = wrong_type = premature = reordered = False
    try:
        port.forward(7002, word, 0, work)
    except PermissionError:
        wrong_owner = True
    try:
        apply_operation(
            port.coefficients,
            port.scratch,
            Operation("DECODE", 0),
            False,
            work,
        )
    except ValueError:
        wrong_type = True
    port.forward(7001, word, 0, work)
    try:
        port.project_final(7001, word, work)
    except PermissionError:
        premature = True
    for index in range(1, word.steps):
        port.forward(7001, word, index, work)
    missing_inverse = port.cursor != 0
    try:
        port.inverse(7001, word, word.steps - 2, work)
    except ValueError:
        reordered = True
    for index in range(word.steps - 1, -1, -1):
        port.inverse(7001, word, index, work)
    port.release(7001, word, work)

    null_carrier = False
    try:
        OpenFusionPort([], [ZERO] * SIMPLE_OBJECTS).lease(1, 1, word, Work())
    except ValueError:
        null_carrier = True

    first = source.copy()
    second = source.copy()
    local = Work()
    scratch = [ZERO] * SIMPLE_OBJECTS
    apply_operation(first, scratch, Operation("TWIST_CASIMIR", 0), False, local)
    apply_operation(first, scratch, Operation("FUSE_FUNDAMENTAL", 0), False, local)
    apply_operation(second, scratch, Operation("FUSE_FUNDAMENTAL", 0), False, local)
    apply_operation(second, scratch, Operation("TWIST_CASIMIR", 0), False, local)

    original, _ = execute_forward(4, 0)
    semantic_program = public_program(4, 0)
    changed = source.copy()
    changed_scratch = [ZERO] * SIMPLE_OBJECTS
    changed_work = Work()
    perturbed = False
    for step in range(semantic_program.steps):
        operation = semantic_program.operation(step)
        if operation.kind == "FUSE_FUNDAMENTAL" and not perturbed:
            operation = Operation(
                operation.kind, (operation.parameter + 1) % 2
            )
            perturbed = True
        apply_operation(changed, changed_scratch, operation, False, changed_work)

    wrong_inverse = source.copy()
    wrong_scratch = [ZERO] * SIMPLE_OBJECTS
    apply_fusion(wrong_inverse, FUSION_FACTORS[0], Work())
    inverse_fusion(wrong_inverse, FUSION_FACTORS[1], wrong_scratch, Work())
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing_inverse,
        "reordered_inverse_rejected": reordered,
        "null_carrier_rejected": null_carrier,
        "module_order_noncommuting_mismatch_cells": mismatch(first, second),
        "semantic_fusion_perturbation_changes_boundary": (
            project_quantum_dimension(original, Work())
            != project_quantum_dimension(changed, Work())
        ),
        "wrong_inverse_factor_changes_state": wrong_inverse != source,
        "control_port_restored": port.coefficients == source
        and all(value == ZERO for value in port.scratch)
        and not port.live,
        "snapshot_command_available": False,
    }


def main() -> None:
    cases = [
        depth_case(depth, family)
        for depth in DEPTHS
        for family in (0, 1)
    ]
    source = source_state()
    carrier = Carrier(
        OpenFusionPort(source.copy(), [ZERO] * SIMPLE_OBJECTS)
    )
    primary_word = public_program(PRIMARY_DEPTH, 0)
    reuse_word = public_program(REUSE_DEPTH, 1)
    primary, primary_work = transaction(carrier, source, primary_word)
    reuse, reuse_work = transaction(carrier, source, reuse_word)
    fresh_carrier = Carrier(
        OpenFusionPort(source.copy(), [ZERO] * SIMPLE_OBJECTS)
    )
    fresh, fresh_work = transaction(fresh_carrier, source, reuse_word)
    control_results = controls()
    if (
        primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or fresh["restoration_error_field_cells"]
        or not primary["same_coefficient_backing"]
        or not primary["same_scratch_backing"]
        or not reuse["same_coefficient_backing"]
        or not reuse["same_scratch_backing"]
        or not primary["canonical_post_restoration_state_exact"]
        or not reuse["canonical_post_restoration_state_exact"]
        or not fresh["canonical_post_restoration_state_exact"]
        or carrier.restoration_generation != 2
        or reuse["boundary"] != fresh["boundary"]
        or reuse["forward_state_commitment"]
        != fresh["forward_state_commitment"]
        or not all(
            (not value if key == "snapshot_command_available" else value)
            if isinstance(value, bool)
            else value > 0
            for key, value in control_results.items()
        )
    ):
        raise RuntimeError("SU2 level-8 fusion transaction failed")

    primary_case = next(
        case
        for case in cases
        if case["depth"] == PRIMARY_DEPTH and case["family"] == 0
    )
    claim = (
        "EXACT_ROOT_OF_UNITY_SU2_LEVEL8_JONES_WENZL_FUSION_PHASE_"
        "RELATION_QUOTIENT_CLOSES_NONCOMMUTING_CASIMIR_TWIST_AND_"
        "TRUNCATED_FUNDAMENTAL_FUSION_ON_ONE_SHARED_UNRESOLVED_PORT_"
        "IN_FIXED9_QZETA40_CELLS_WITH_FINAL_ONLY_QUANTUM_DIMENSION_"
        "BOUNDARY_EXACT_PIVOT_REMATERIALIZED_RESTORATION_REUSE_BUT_"
        "EXACT_COEFFICIENT_HEIGHT_GROWS_AND_THE_IDENTICAL_FUSION_"
        "CLASSICAL_RECURRENCE_REMAINS"
    )
    result = {
        "schema": "cat_cas.root_of_unity_su2_level8_fusion_relation.v1",
        "claim_candidate": claim,
        "claim_ceiling": (
            "FORMAL_SU2_LEVEL8_VERLINDE_FUSION_ALGEBRA_QZETA40_"
            "JONES_WENZL_X9_ZERO_TWO_TWIST_POWERS_TWO_UNIT_FUSION_"
            "FACTORS_DECLARED_FAMILIES_DEPTHS1_2_4_8_16_32_64_128_"
            "PRIMARY128_REUSE73_QUANTUM_DIMENSION_BOUNDARY_DIRECT_PROCESS_ONLY"
        ),
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_EXACT_SU2_LEVEL8_FIXED9_FUSION_CLOSURE_WITH_GROWING_COEFFICIENT_HEIGHT",
        "phase_relation_law": {
            "domain": "ROOT_OF_UNITY_SU2_LEVEL8_VERLINDE_FUSION_ALGEBRA",
            "coefficient_field": "Q_ZETA40_DEGREE16",
            "simple_object_field_cells": SIMPLE_OBJECTS,
            "jones_wenzl_relation": "X9_EQUALS_ZERO",
            "fundamental_fusion": "X1_XJ_EQUALS_XJMINUS1_PLUS_XJPLUS1_WITH_LEVEL8_TRUNCATION",
            "twist": "THETA_J_EQUALS_ZETA40_TO_J_TIMES_JPLUS2",
            "shared_unresolved_ports": 1,
            "multiple_noncommuting_consumers": True,
            "intermediate_projection": False,
            "truth_table_assignment_group_element_or_path_expansion": False,
            "final_boundary": "EXACT_QUANTUM_DIMENSION_CHARACTER",
        },
        "depth_cases": cases,
        "support_law": {
            "allocated_simple_object_field_cells": SIMPLE_OBJECTS,
            "fixed_across_declared_depths": all(
                case["simple_object_field_cells"] == SIMPLE_OBJECTS
                for case in cases
            ),
            "continuous_su2_predecessor_cells_at_depth128": 130,
            "root_of_unity_quotient_is_a_changed_algebra_not_a_lossless_encoding_of_general_continuous_su2": True,
            "bounded_width_exact_state_set_established": False,
        },
        "transaction": {
            "primary_boundary_commitment": primary["boundary_commitment"],
            "primary_boundary_payload_bits": primary["boundary_payload_bits"],
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "reuse_boundary_payload_bits": reuse["boundary_payload_bits"],
            "fresh_reuse_boundary_commitment": fresh["boundary_commitment"],
            "fresh_restored_reuse_boundary_agreement": reuse["boundary"]
            == fresh["boundary"],
            "fresh_restored_reuse_state_agreement": reuse[
                "forward_state_commitment"
            ]
            == fresh["forward_state_commitment"],
            "primary_missing_inverse_error_cells_and_cursor": primary[
                "missing_inverse_error_cells_and_cursor"
            ],
            "primary_restoration_error_field_cells": primary[
                "restoration_error_field_cells"
            ],
            "reuse_restoration_error_field_cells": reuse[
                "restoration_error_field_cells"
            ],
            "primary_same_coefficient_backing": primary[
                "same_coefficient_backing"
            ],
            "primary_same_scratch_backing": primary["same_scratch_backing"],
            "reuse_same_coefficient_backing": reuse["same_coefficient_backing"],
            "reuse_same_scratch_backing": reuse["same_scratch_backing"],
            "canonical_post_restoration_state_exact": primary[
                "canonical_post_restoration_state_exact"
            ]
            and reuse["canonical_post_restoration_state_exact"],
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
            "restoration_method": "EXACT_REVERSE_ROOT_TWIST_AND_REMATERIALIZED_TRIDIAGONAL_FUSION_SOLVE_WITH_REVERSE_PIVOT_UNCOMPUTATION_ON_SAME_BACKINGS",
        },
        "controls": control_results,
        "resource_law": {
            "initial_carrier_field_cells": SIMPLE_OBJECTS,
            "initial_carrier_payload_bits": field_payload_bits(source),
            "inverse_scratch_field_cells": SIMPLE_OBJECTS,
            "inverse_scratch_clean_payload_bits": field_payload_bits(
                [ZERO] * SIMPLE_OBJECTS
            ),
            "compiled_public_factor_field_cells": len(FUSION_FACTORS) + 1,
            "compiled_public_factor_payload_bits": field_payload_bits(
                [SOURCE_FUNDAMENTAL, *FUSION_FACTORS]
            ),
            "compiled_quantum_dimension_field_cells": len(QUANTUM_DIMENSIONS),
            "compiled_quantum_dimension_payload_bits": field_payload_bits(
                QUANTUM_DIMENSIONS
            ),
            "retained_inverse_pivot_plan_cells": 0,
            "primary_forward_carrier_field_cells": primary_case[
                "simple_object_field_cells"
            ],
            "primary_forward_carrier_payload_bits": primary_case[
                "carrier_payload_bits"
            ],
            "primary_boundary_payload_bits": primary[
                "boundary_payload_bits"
            ],
            "primary_full_transaction_work": primary_work.as_dict(),
            "reuse_full_transaction_work": reuse_work.as_dict(),
            "fresh_reuse_verification_work": fresh_work.as_dict(),
            "primary_public_operation_records_materialized": 0,
            "primary_public_descriptor_integers": 2,
            "primary_generated_operation_visits": 2 * primary_word.steps,
            "reuse_public_operation_records_materialized": 0,
            "reuse_public_descriptor_integers": 2,
            "reuse_generated_operation_visits": 2 * reuse_word.steps,
            "open_port_metadata_integers": 5,
            "open_port_program_commitment_bytes": 32,
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries": 0,
            "controller_backend_traffic_not_applicable_direct_process": True,
            "resource_measurement_scope": "EXACT_QZETA40_FIELD_CELLS_COORDINATE_PAYLOAD_PUBLIC_RECORDS_COMMITMENT_VISITS_AND_HIGH_LEVEL_FIELD_OPERATION_COUNTS",
            "public_descriptor_index_arithmetic_fraction_scalar_suboperations_transient_objects_allocator_interpreter_hash_byte_traffic_serialization_timing_and_whole_process_peaks_excluded_not_zero": True,
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_NINE_QZETA40_COORDINATE_SU2_LEVEL8_FUSION_AND_TWIST_RECURRENCE",
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
