#!/usr/bin/env python3
"""Exact generic-center theta moment Q-jet carrier on continuous S1.

Runtime-resident Gaussian-rational unit phases u_j define wrapped-Gaussian
kernels theta(u_j*x; Q).  The product relation is never sampled on an angle
grid.  Its logarithm is

    sum_j log(theta(u_j*x;Q)) = sum_m p_m L_m(Q) x^m,

where p_m=sum_j u_j^m.  For the first-harmonic jet through Q^J, moments only
through |m| <= floor((J+1)/2) can participate.  The carrier streams runtime
centers into those exact moments, projects the final jet by formal log/exp
series, then reverses the actual moment updates on the same backing.

This removes M209's four-symbol factor-count shortcut, but it exposes two
material laws: the requested moment count grows with precision, and the
runtime center source itself retains one exact phase per factor.  An identical
classical moment recurrence remains.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction


ORDERS = (2, 4, 8, 12, 16, 20, 24)
PRIMARY_CENTERS = 24
REUSE_CENTERS = 17
PORT_TYPE = 21017


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

    def scale(self, scalar: Fraction | int) -> "G":
        return G(self.real * scalar, self.imag * scalar)

    def conjugate(self) -> "G":
        return G(self.real, -self.imag)

    def __pow__(self, exponent: int) -> "G":
        if exponent < 0:
            return (self.conjugate()) ** (-exponent)
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


ZERO = G.rational(0)
ONE = G.rational(1)
ROTATIONS = (
    G(Fraction(-3, 5), Fraction(4, 5)),
    G(Fraction(-4, 5), Fraction(3, 5)),
)


def unit_phase(parameter: int) -> G:
    t = Fraction(parameter)
    denominator = 1 + t * t
    return G((1 - t * t) / denominator, 2 * t / denominator)


def center_family(count: int, family: int) -> list[G]:
    if family == 0:
        parameters = [2 * index + 1 for index in range(1, count + 1)]
    else:
        parameters = [index * index + index + 1 for index in range(1, count + 1)]
    return [unit_phase(parameter) for parameter in parameters]


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def gaussian_payload_bits(values: list[G] | tuple[G, ...]) -> int:
    return sum(
        signed_bits(component.numerator) + component.denominator.bit_length()
        for value in values
        for component in (value.real, value.imag)
    )


def commitment(values: list[G] | tuple[G, ...]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in values).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class Operation:
    kind: str
    parameter: int

    def token(self) -> str:
        return f"{self.kind}:{self.parameter}"


def program(center_count: int, family: int) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for index in range(center_count):
        ingest = Operation("INGEST", index)
        if index % 4 == (family + 1) % 4:
            operations.extend((Operation("ROTATE", (index + family) % 2), ingest))
        elif index % 5 == (family + 2) % 5:
            operations.extend((ingest, Operation("ROTATE", (index + family + 1) % 2)))
        else:
            operations.append(ingest)
    return tuple(operations)


def program_commitment(operations: tuple[Operation, ...]) -> str:
    return hashlib.sha256(
        "|".join(operation.token() for operation in operations).encode("ascii")
    ).hexdigest()


Series = dict[tuple[int, int], G]


@dataclass
class ProjectionWork:
    log_multiplications: int = 0
    exp_multiplications: int = 0
    series_products: int = 0
    series_accumulations: int = 0
    peak_series_cells: int = 0
    universal_log_cells: int = 0
    moment_weighted_log_cells: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


def series_add_scaled(target: Series, source: Series, scalar: Fraction) -> None:
    for key, value in source.items():
        updated = target.get(key, ZERO) + value.scale(scalar)
        if updated.is_zero():
            target.pop(key, None)
        else:
            target[key] = updated


def series_multiply(
    left: Series, right: Series, order: int, work: ProjectionWork
) -> Series:
    result: Series = {}
    for (left_h, left_e), left_value in left.items():
        for (right_h, right_e), right_value in right.items():
            exponent = left_e + right_e
            if exponent > order:
                continue
            harmonic = left_h + right_h
            if abs(harmonic) > order:
                continue
            key = harmonic, exponent
            result[key] = result.get(key, ZERO) + left_value * right_value
            work.series_products += 1
    result = {key: value for key, value in result.items() if not value.is_zero()}
    work.series_accumulations += len(result)
    work.peak_series_cells = max(work.peak_series_cells, len(result))
    return result


def universal_log_theta(order: int, work: ProjectionWork) -> Series:
    radius = math.isqrt(order)
    augmentation: Series = {
        (mode, mode * mode): ONE
        for mode in range(-radius, radius + 1)
        if mode
    }
    term: Series = {(0, 0): ONE}
    logarithm: Series = {}
    for power in range(1, order + 1):
        term = series_multiply(term, augmentation, order, work)
        work.log_multiplications += 1
        series_add_scaled(
            logarithm,
            term,
            Fraction(1 if power % 2 else -1, power),
        )
    work.universal_log_cells = len(logarithm)
    return logarithm


def project_moment_qjet(
    moments: list[G], order: int
) -> tuple[tuple[G, ...], ProjectionWork, dict[str, object]]:
    maximum_moment = (order + 1) // 2
    if len(moments) != maximum_moment + 1:
        raise ValueError("moment chart width does not match Q-jet order")
    work = ProjectionWork()
    logarithm = universal_log_theta(order, work)
    weighted: Series = {}
    for (harmonic, exponent), coefficient in logarithm.items():
        if abs(harmonic) > maximum_moment:
            continue
        moment = moments[harmonic] if harmonic >= 0 else moments[-harmonic].conjugate()
        value = coefficient * moment
        if not value.is_zero():
            weighted[harmonic, exponent] = value
    work.moment_weighted_log_cells = len(weighted)
    exponential: Series = {(0, 0): ONE}
    term: Series = {(0, 0): ONE}
    for power in range(1, order + 1):
        term = series_multiply(term, weighted, order, work)
        term = {key: value.scale(Fraction(1, power)) for key, value in term.items()}
        work.exp_multiplications += 1
        series_add_scaled(exponential, term, Fraction(1))
    jet = tuple(exponential.get((1, exponent), ZERO) for exponent in range(order + 1))
    hierarchy = {
        "maximum_required_positive_moment": maximum_moment,
        "resident_moment_cells": maximum_moment + 1,
        "new_moment_formal_dependence": [
            {
                "moment": moment,
                "first_possible_first_harmonic_q_order": 2 * moment - 1,
                "universal_log_positive_coefficient": logarithm.get(
                    (moment, moment), ZERO
                ).token(),
                "universal_log_negative_coefficient": logarithm.get(
                    (-moment, moment), ZERO
                ).token(),
            }
            for moment in range(1, maximum_moment + 1)
        ],
    }
    for entry in hierarchy["new_moment_formal_dependence"]:
        if entry["universal_log_positive_coefficient"] == ZERO.token():
            raise RuntimeError("theta moment hierarchy coefficient vanished")
    return jet, work, hierarchy


def add_center_to_moments(moments: list[G], center: G, sign: int) -> None:
    moments[0] = moments[0] + G.rational(sign)
    power = ONE
    for moment in range(1, len(moments)):
        power = power * center
        moments[moment] = moments[moment] + power.scale(sign)


def rotate_moments(moments: list[G], rotation: G) -> None:
    power = ONE
    for moment in range(1, len(moments)):
        power = power * rotation
        moments[moment] = moments[moment] * power


@dataclass
class MomentPort:
    source_centers: list[G]
    moments: list[G]
    live: bool = False
    owner: int = 0
    generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    program_hash: str = ""

    def lease(
        self, owner: int, generation: int, operations: tuple[Operation, ...]
    ) -> None:
        if self.live:
            raise RuntimeError("theta moment port already live")
        if not self.source_centers:
            raise ValueError("null generic-center carrier rejected")
        if any(not value.is_zero() for value in self.moments):
            raise RuntimeError("theta moment carrier is not restored")
        self.live = True
        self.owner = owner
        self.generation = generation
        self.cursor = 0
        self.expected_steps = len(operations)
        self.program_hash = program_commitment(operations)

    def require(self, owner: int, operations: tuple[Operation, ...]) -> None:
        if not self.live:
            raise RuntimeError("theta moment port is not live")
        if owner != self.owner:
            raise PermissionError("theta moment owner mismatch")
        if program_commitment(operations) != self.program_hash:
            raise ValueError("theta moment public program mismatch")

    def forward(
        self, owner: int, operations: tuple[Operation, ...], index: int
    ) -> None:
        self.require(owner, operations)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("theta moment forward cursor mismatch")
        operation = operations[index]
        if operation.kind == "INGEST":
            add_center_to_moments(
                self.moments, self.source_centers[operation.parameter], 1
            )
        elif operation.kind == "ROTATE":
            rotate_moments(self.moments, ROTATIONS[operation.parameter])
        else:
            raise ValueError("wrong theta moment operation type")
        self.cursor += 1

    def inverse(
        self, owner: int, operations: tuple[Operation, ...], index: int
    ) -> None:
        self.require(owner, operations)
        if index != self.cursor - 1:
            raise ValueError("theta moment inverse cursor mismatch")
        operation = operations[index]
        if operation.kind == "INGEST":
            add_center_to_moments(
                self.moments, self.source_centers[operation.parameter], -1
            )
        elif operation.kind == "ROTATE":
            rotate_moments(
                self.moments, ROTATIONS[operation.parameter].conjugate()
            )
        else:
            raise ValueError("wrong theta moment operation type")
        self.cursor -= 1

    def project_final(
        self, owner: int, operations: tuple[Operation, ...], order: int
    ) -> tuple[tuple[G, ...], ProjectionWork, dict[str, object]]:
        self.require(owner, operations)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal theta moment projection rejected")
        return project_moment_qjet(self.moments, order)

    def release(self, owner: int, operations: tuple[Operation, ...]) -> int:
        self.require(owner, operations)
        if self.cursor or any(not value.is_zero() for value in self.moments):
            raise RuntimeError("theta moment port released before exact inverse")
        generation = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.expected_steps = 0
        self.program_hash = ""
        return generation


@dataclass
class Carrier:
    port: MomentPort
    restoration_generation: int = 0


def run_transaction(
    carrier: Carrier,
    operations: tuple[Operation, ...],
    order: int,
    owner: int,
) -> dict[str, object]:
    source_backing = id(carrier.port.source_centers)
    moment_backing = id(carrier.port.moments)
    initial_source = carrier.port.source_centers.copy()
    carrier.port.lease(owner, carrier.restoration_generation + 1, operations)
    for index in range(len(operations)):
        carrier.port.forward(owner, operations, index)
    resident_commitment = commitment(carrier.port.moments)
    resident_payload_bits = gaussian_payload_bits(carrier.port.moments)
    jet, work, hierarchy = carrier.port.project_final(owner, operations, order)
    for index in range(len(operations) - 1, -1, -1):
        carrier.port.inverse(owner, operations, index)
    generation = carrier.port.release(owner, operations)
    carrier.restoration_generation = generation
    if carrier.port.source_centers != initial_source:
        raise RuntimeError("runtime theta centers changed")
    return {
        "resident_moment_commitment": resident_commitment,
        "resident_moment_payload_bits": resident_payload_bits,
        "boundary_commitment": commitment(jet),
        "boundary_payload_bits": gaussian_payload_bits(jet),
        "projection_work": work.as_dict(),
        "hierarchy": hierarchy,
        "source_same_backing": id(carrier.port.source_centers) == source_backing,
        "moments_same_backing": id(carrier.port.moments) == moment_backing,
        "restoration_error_moment_cells": sum(
            not value.is_zero() for value in carrier.port.moments
        ),
        "restoration_generation": carrier.restoration_generation,
    }


def case(order: int, family: int) -> dict[str, object]:
    centers = center_family(PRIMARY_CENTERS, family)
    operations = program(PRIMARY_CENTERS, family)
    moments = [ZERO] * ((order + 1) // 2 + 1)
    transaction = run_transaction(
        Carrier(MomentPort(centers, moments)), operations, order, 7000 + family
    )
    return {
        "q_jet_order": order,
        "family": family,
        "runtime_source_center_cells": len(centers),
        "runtime_source_payload_bits": gaussian_payload_bits(centers),
        "resident_moment_cells": (order + 1) // 2 + 1,
        "resident_moment_payload_bits": transaction[
            "resident_moment_payload_bits"
        ],
        "public_operation_records": len(operations),
        "public_descriptor_slots": 2 * len(operations),
        "resident_moment_commitment": transaction[
            "resident_moment_commitment"
        ],
        "boundary_commitment": transaction["boundary_commitment"],
        "boundary_payload_bits": transaction["boundary_payload_bits"],
        "universal_log_cells": transaction["projection_work"][
            "universal_log_cells"
        ],
        "moment_weighted_log_cells": transaction["projection_work"][
            "moment_weighted_log_cells"
        ],
        "projection_peak_series_cells": transaction["projection_work"][
            "peak_series_cells"
        ],
        "projection_series_products": transaction["projection_work"][
            "series_products"
        ],
        "highest_formally_required_moment": transaction["hierarchy"][
            "maximum_required_positive_moment"
        ],
        "exact_same_backing_restoration": transaction["source_same_backing"]
        and transaction["moments_same_backing"],
    }


def controls() -> dict[str, object]:
    centers = center_family(4, 0)
    operations = program(4, 0)
    port = MomentPort(centers, [ZERO] * 5)
    wrong_owner = wrong_type = premature = missing = reordered = null = False
    port.lease(51, 1, operations)
    try:
        port.forward(52, operations, 0)
    except PermissionError:
        wrong_owner = True
    try:
        bogus = (Operation("DECODE", 0),)
        port.program_hash = program_commitment(bogus)
        port.forward(51, bogus, 0)
    except ValueError:
        wrong_type = True
    finally:
        port.program_hash = program_commitment(operations)
    try:
        port.project_final(51, operations, 8)
    except PermissionError:
        premature = True
    for index in range(len(operations)):
        port.forward(51, operations, index)
    try:
        port.release(51, operations)
    except RuntimeError:
        missing = True
    try:
        port.inverse(51, operations, len(operations) - 2)
    except ValueError:
        reordered = True
    for index in range(len(operations) - 1, -1, -1):
        port.inverse(51, operations, index)
    port.release(51, operations)
    try:
        MomentPort([], [ZERO]).lease(1, 1, tuple())
    except ValueError:
        null = True

    first = MomentPort(center_family(2, 0), [ZERO] * 5)
    second = MomentPort(center_family(2, 0), [ZERO] * 5)
    first_ops = (Operation("ROTATE", 0), Operation("INGEST", 0))
    second_ops = (Operation("INGEST", 0), Operation("ROTATE", 0))
    for index, operation in enumerate(first_ops):
        if operation.kind == "ROTATE":
            rotate_moments(first.moments, ROTATIONS[0])
        else:
            add_center_to_moments(first.moments, first.source_centers[0], 1)
    for operation in second_ops:
        if operation.kind == "ROTATE":
            rotate_moments(second.moments, ROTATIONS[0])
        else:
            add_center_to_moments(second.moments, second.source_centers[0], 1)
    jet_a, _, _ = project_moment_qjet(first.moments, 8)
    jet_b, _, _ = project_moment_qjet(second.moments, 8)
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "missing_inverse_detected": missing,
        "reordered_inverse_rejected": reordered,
        "null_carrier_rejected": null,
        "module_order_moments_differ": first.moments != second.moments,
        "module_order_boundary_changes": jet_a != jet_b,
        "control_port_restored": all(value.is_zero() for value in port.moments),
    }


def main() -> None:
    cases = [case(order, family) for order in ORDERS for family in (0, 1)]
    primary_order = 24
    primary_source = center_family(PRIMARY_CENTERS, 0)
    primary_source_payload_bits = gaussian_payload_bits(primary_source)
    carrier = Carrier(
        MomentPort(primary_source, [ZERO] * ((primary_order + 1) // 2 + 1))
    )
    source_backing = id(carrier.port.source_centers)
    moment_backing = id(carrier.port.moments)
    primary = run_transaction(
        carrier, program(PRIMARY_CENTERS, 0), primary_order, 9101
    )
    reuse_source = center_family(REUSE_CENTERS, 1)
    carrier.port.source_centers[:] = reuse_source
    reuse = run_transaction(
        carrier, program(REUSE_CENTERS, 1), primary_order, 9102
    )
    fresh = run_transaction(
        Carrier(
            MomentPort(
                reuse_source.copy(), [ZERO] * ((primary_order + 1) // 2 + 1)
            )
        ),
        program(REUSE_CENTERS, 1),
        primary_order,
        9102,
    )
    if reuse["boundary_commitment"] != fresh["boundary_commitment"]:
        raise RuntimeError("fresh and restored generic-center reuse differ")

    result = {
        "schema": "cat_cas.continuous_s1_generic_center_theta_moment.v1",
        "claim": "EXACT_CONTINUOUS_S1_GENERIC_GAUSSIAN_RATIONAL_CENTER_THETA_FIRST_HARMONIC_QJET_CLOSES_ON_RUNTIME_POWER_MOMENTS_THROUGH_FLOOR_JPLUS1_OVER2_WITH_FINAL_ONLY_PROJECTION_EXACT_RESTORATION_REUSE_BUT_MOMENT_CELLS_GROW_FROM2_TO13_ACROSS_J2_TO24_THE24_CENTER_SOURCE_REMAINS_RESIDENT_AND_IDENTICAL_CLASSICAL_MOMENT_AND_DIRECT_FACTOR_RECURRENCES_REMAIN",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_EXACT_GENERIC_CENTER_MOMENT_QJET_WITH_PRECISION_HIERARCHY_AND_SOURCE_RETENTION_OBSTRUCTION",
        "phase_relation_law": {
            "domain": "CONTINUOUS_S1_NO_FINITE_ANGLE_SAMPLING",
            "runtime_center_type": "GAUSSIAN_RATIONAL_UNIT_PHASE",
            "finite_phase_center_alphabet": False,
            "relation_intersection": "PRODUCT_OF_RUNTIME_CENTERED_WRAPPED_GAUSSIAN_KERNELS",
            "native_moment_law": "LOG_THETA_WEIGHTED_BY_RUNTIME_POWER_SUMS_THEN_FORMAL_EXP",
            "shared_unresolved_angle_ports": 1,
            "multiple_noncommuting_consumers": True,
            "intermediate_moment_projection": False,
            "truth_table_or_assignment_expansion": False,
        },
        "moment_hierarchy": {
            "first_harmonic_qjet_order_j_requires_moments_through": "FLOOR_JPLUS1_OVER_2",
            "resident_moment_cells": "FLOOR_JPLUS1_OVER_2_PLUS_1_INCLUDING_P0",
            "new_positive_moment_m_first_appears_in_first_harmonic_boundary_by_order": "2M_MINUS_1",
            "fixed_moment_count_for_unbounded_precision": False,
            "full_infinite_theta_scalar_evaluated": False,
        },
        "precision_cases": cases,
        "transaction": {
            "primary_boundary_commitment": primary["boundary_commitment"],
            "reuse_boundary_commitment": reuse["boundary_commitment"],
            "fresh_reuse_boundary_commitment": fresh["boundary_commitment"],
            "source_backing_identity_preserved_across_program_reuse": id(carrier.port.source_centers) == source_backing,
            "moment_backing_identity_preserved_across_program_reuse": id(carrier.port.moments) == moment_backing,
            "primary_restoration_error_moment_cells": primary[
                "restoration_error_moment_cells"
            ],
            "reuse_restoration_error_moment_cells": reuse[
                "restoration_error_moment_cells"
            ],
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        "controls": controls(),
        "resource_law": {
            "primary_runtime_source_center_cells": PRIMARY_CENTERS,
            "primary_runtime_source_payload_bits": primary_source_payload_bits,
            "primary_resident_moment_cells": (primary_order + 1) // 2 + 1,
            "primary_resident_moment_payload_bits": primary[
                "resident_moment_payload_bits"
            ],
            "primary_compiled_public_program_operation_records": len(
                program(PRIMARY_CENTERS, 0)
            ),
            "primary_compiled_public_program_descriptor_slots": 2
            * len(program(PRIMARY_CENTERS, 0)),
            "primary_projection_universal_log_cells": primary[
                "projection_work"
            ]["universal_log_cells"],
            "primary_projection_peak_series_cells": primary[
                "projection_work"
            ]["peak_series_cells"],
            "primary_projection_series_products": primary[
                "projection_work"
            ]["series_products"],
            "retained_inverse_history_entries": 0,
            "additional_retained_plan_entries_beyond_public_program": 0,
            "python_fraction_object_allocator_interpreter_serialization_timing_and_whole_process_peaks_excluded_not_zero": True,
        },
        "matched_classical_baselines": {
            "moment_chart": "IDENTICAL_RUNTIME_POWER_MOMENT_LOG_THETA_EXP_RECURRENCE",
            "direct_boundary": "EXACT_FACTOR_BY_FACTOR_SPARSE_QJET_RECURRENCE",
            "strictly_smaller_or_faster_phase_path_established": False,
        },
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
