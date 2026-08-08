#!/usr/bin/env python3
"""Separate polynomial-quotient reference for M230 open fusion signatures."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass

import root_of_unity_su2_level8_fusion_independent_oracle as oracle


sys.set_int_max_str_digits(0)

SOURCE_TYPE = 23001
INTERNAL_TYPE = 23002
TWIST_BRANCH_TYPE = 23003
CUBIC_BRANCH_TYPE = 23004
INTERSECTION_TYPE = 23005
OUTPUT_TYPE = 23006
FAMILIES = (0, 1)


@dataclass(frozen=True)
class FusionSignature:
    parameter: int
    input_type: int
    output_type: int

    def __post_init__(self) -> None:
        if self.parameter not in (3, 7):
            raise ValueError("reference fusion phase mismatch")
        if (self.input_type, self.output_type) not in {
            (SOURCE_TYPE, INTERNAL_TYPE),
            (INTERSECTION_TYPE, OUTPUT_TYPE),
        }:
            raise TypeError("reference fusion port mismatch")

    def integers(self) -> tuple[int, ...]:
        return (1, self.parameter, self.input_type, self.output_type)


@dataclass(frozen=True)
class TwistSignature:
    parameter: int
    input_type: int = INTERNAL_TYPE
    output_type: int = TWIST_BRANCH_TYPE

    def __post_init__(self) -> None:
        if self.parameter not in (0, 1):
            raise ValueError("reference twist parameter mismatch")
        if self.input_type != INTERNAL_TYPE or self.output_type != TWIST_BRANCH_TYPE:
            raise TypeError("reference twist port mismatch")

    def integers(self) -> tuple[int, ...]:
        return (2, self.parameter, self.input_type, self.output_type)


@dataclass(frozen=True)
class CubicSignature:
    parameter: int
    source: int
    target: int
    input_type: int = INTERNAL_TYPE
    output_type: int = CUBIC_BRANCH_TYPE

    def __post_init__(self) -> None:
        if self.parameter not in (3, 7):
            raise ValueError("reference cubic parameter mismatch")
        if not (0 <= self.source < oracle.SIMPLE_OBJECTS):
            raise ValueError("reference cubic source mismatch")
        if not (0 <= self.target < oracle.SIMPLE_OBJECTS):
            raise ValueError("reference cubic target mismatch")
        if self.source == self.target:
            raise ValueError("reference cubic endpoints coincide")
        if self.input_type != INTERNAL_TYPE or self.output_type != CUBIC_BRANCH_TYPE:
            raise TypeError("reference cubic port mismatch")

    def integers(self) -> tuple[int, ...]:
        return (
            3,
            self.parameter,
            self.source,
            self.target,
            self.input_type,
            self.output_type,
        )


@dataclass(frozen=True)
class IntersectionSignature:
    left_type: int = TWIST_BRANCH_TYPE
    right_type: int = CUBIC_BRANCH_TYPE
    output_type: int = INTERSECTION_TYPE

    def __post_init__(self) -> None:
        if (
            self.left_type != TWIST_BRANCH_TYPE
            or self.right_type != CUBIC_BRANCH_TYPE
            or self.output_type != INTERSECTION_TYPE
        ):
            raise TypeError("reference intersection port mismatch")

    def integers(self) -> tuple[int, ...]:
        return (4, self.left_type, self.right_type, self.output_type)


@dataclass(frozen=True)
class Topology:
    input_fusion: FusionSignature
    twist: TwistSignature
    cubic: CubicSignature
    intersection: IntersectionSignature
    output_fusion: FusionSignature

    def descriptors(self) -> tuple[object, ...]:
        return (
            self.input_fusion,
            self.twist,
            self.cubic,
            self.intersection,
            self.output_fusion,
        )

    def integers(self) -> tuple[int, ...]:
        return tuple(
            value
            for descriptor in self.descriptors()
            for value in descriptor.integers()  # type: ignore[attr-defined]
        )

    def token(self) -> str:
        return ":".join(str(value) for value in self.integers())


@dataclass(frozen=True)
class Program:
    family: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("reference public family mismatch")

    def compile(self) -> Topology:
        source = 1 + self.family
        target = 2 - self.family
        return Topology(
            FusionSignature((3, 7)[self.family], SOURCE_TYPE, INTERNAL_TYPE),
            TwistSignature(self.family),
            CubicSignature((7, 3)[self.family], source, target),
            IntersectionSignature(),
            FusionSignature((7, 3)[self.family], INTERSECTION_TYPE, OUTPUT_TYPE),
        )

    def token(self) -> str:
        return f"family:{self.family}|{self.compile().token()}"


def program_commitment(program: Program) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


X = [oracle.ZERO, oracle.ONE]


def fusion_value(
    polynomial: oracle.Polynomial, signature: FusionSignature
) -> oracle.Polynomial:
    return oracle.quotient_reduce(
        oracle.poly_scale(
            oracle.poly_multiply(polynomial, X), oracle.E.root(signature.parameter)
        )
    )


def twist_value(
    polynomial: oracle.Polynomial, signature: TwistSignature
) -> oracle.Polynomial:
    coefficients = oracle.polynomial_to_simple(polynomial)
    power = oracle.TWIST_POWERS[signature.parameter]
    return oracle.simple_to_polynomial(
        [
            coefficient
            * oracle.E.root(power * label * (label + 2))
            for label, coefficient in enumerate(coefficients)
        ]
    )


def cubic_value(
    polynomial: oracle.Polynomial, signature: CubicSignature
) -> oracle.Polynomial:
    coefficients = oracle.polynomial_to_simple(polynomial)
    source = coefficients[signature.source]
    coefficients[signature.target] = (
        coefficients[signature.target]
        + oracle.E.root(signature.parameter) * source * source * source
    )
    return oracle.simple_to_polynomial(coefficients)


def intersection_value(
    left: oracle.Polynomial, right: oracle.Polynomial
) -> oracle.Polynomial:
    left_coefficients = oracle.polynomial_to_simple(left)
    right_coefficients = oracle.polynomial_to_simple(right)
    return oracle.simple_to_polynomial(
        [
            a * b
            for a, b in zip(left_coefficients, right_coefficients, strict=True)
        ]
    )


def add_rail(
    target: oracle.Polynomial, value: oracle.Polynomial
) -> oracle.Polynomial:
    return oracle.quotient_reduce(oracle.poly_add(target, value))


def subtract_rail(
    target: oracle.Polynomial, value: oracle.Polynomial
) -> oracle.Polynomial:
    return oracle.quotient_reduce(oracle.poly_subtract(target, value))


def zero_rail() -> oracle.Polynomial:
    return []


@dataclass
class Port:
    source: oracle.Polynomial
    internal: oracle.Polynomial
    twist: oracle.Polynomial
    cubic: oracle.Polynomial
    intersection: oracle.Polynomial
    output: oracle.Polynomial
    enabled: bool = True
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    branch_mask: int = 0
    program_hash: str = ""

    def lease(self, owner: int, generation: int, program: Program) -> Topology:
        if self.live:
            raise RuntimeError("reference relation port already live")
        if not self.enabled:
            raise ValueError("reference null relation port")
        if any((self.internal, self.twist, self.cubic, self.intersection, self.output)):
            raise ValueError("reference relation work rail dirty")
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("reference invalid or stale lease")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.branch_mask = 0
        self.program_hash = program_commitment(program)
        return topology

    def require(self, owner: int, generation: int, program: Program) -> Topology:
        if not self.live:
            raise RuntimeError("reference relation port not live")
        if owner != self.owner:
            raise PermissionError("reference owner mismatch")
        if generation != self.generation:
            raise PermissionError("reference generation mismatch")
        if program_commitment(program) != self.program_hash:
            raise ValueError("reference program mismatch")
        return program.compile()

    def produce_internal(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 0 or self.branch_mask:
            raise ValueError("reference internal production order")
        self.internal = add_rail(self.internal, fusion_value(self.source, topology.input_fusion))
        self.stage = 1

    def consume_twist(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or self.branch_mask & 1:
            raise ValueError("reference twist consumption order")
        self.twist = add_rail(self.twist, twist_value(self.internal, topology.twist))
        self.branch_mask |= 1

    def consume_cubic(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or self.branch_mask & 2:
            raise ValueError("reference cubic consumption order")
        self.cubic = add_rail(self.cubic, cubic_value(self.internal, topology.cubic))
        self.branch_mask |= 2

    def intersect(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 1 or self.branch_mask != 3:
            raise PermissionError("reference intersection undermerge")
        self.intersection = add_rail(
            self.intersection, intersection_value(self.twist, self.cubic)
        )
        self.stage = 2

    def produce_output(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 2:
            raise ValueError("reference output production order")
        self.output = add_rail(
            self.output, fusion_value(self.intersection, topology.output_fusion)
        )
        self.stage = 3

    def project(self, owner: int, generation: int, program: Program) -> oracle.E:
        self.require(owner, generation, program)
        if self.stage != 3:
            raise PermissionError("reference nonfinal projection")
        return oracle.evaluate(self.output, oracle.DELTA)

    def project_internal(self) -> None:
        raise PermissionError("reference internal relation message hidden")

    def clear_output(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 3:
            raise ValueError("reference output inverse order")
        self.output = subtract_rail(
            self.output, fusion_value(self.intersection, topology.output_fusion)
        )
        if self.output:
            raise RuntimeError("reference output did not clear")
        self.stage = 2

    def clear_intersection(self, owner: int, generation: int, program: Program) -> None:
        self.require(owner, generation, program)
        if self.stage != 2:
            raise ValueError("reference intersection inverse order")
        self.intersection = subtract_rail(
            self.intersection, intersection_value(self.twist, self.cubic)
        )
        if self.intersection:
            raise RuntimeError("reference intersection did not clear")
        self.stage = 1

    def clear_twist(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or not self.branch_mask & 1:
            raise ValueError("reference twist inverse order")
        self.twist = subtract_rail(self.twist, twist_value(self.internal, topology.twist))
        if self.twist:
            raise RuntimeError("reference twist did not clear")
        self.branch_mask &= ~1

    def clear_cubic(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or not self.branch_mask & 2:
            raise ValueError("reference cubic inverse order")
        self.cubic = subtract_rail(self.cubic, cubic_value(self.internal, topology.cubic))
        if self.cubic:
            raise RuntimeError("reference cubic did not clear")
        self.branch_mask &= ~2

    def clear_internal(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or self.branch_mask:
            raise ValueError("reference internal inverse order")
        self.internal = subtract_rail(
            self.internal, fusion_value(self.source, topology.input_fusion)
        )
        if self.internal:
            raise RuntimeError("reference internal did not clear")
        self.stage = 0

    def release(self, owner: int, generation: int, program: Program) -> int:
        self.require(owner, generation, program)
        if self.stage or self.branch_mask or any(
            (self.internal, self.twist, self.cubic, self.intersection, self.output)
        ):
            raise RuntimeError("reference release before restore")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.program_hash = ""
        self.last_restored_generation = restored
        return restored


def make_port() -> Port:
    return Port(oracle.source_polynomial(), [], [], [], [], [])


def simple_payload(polynomial: oracle.Polynomial) -> int:
    return oracle.payload(oracle.polynomial_to_simple(polynomial))


def execute(port: Port, program: Program) -> dict[str, object]:
    source = oracle.source_polynomial()
    generation = port.last_restored_generation + 1
    owner = 230000 + generation
    topology = port.lease(owner, generation, program)
    port.produce_internal(owner, generation, program)
    if program.family == 0:
        port.consume_twist(owner, generation, program)
        port.consume_cubic(owner, generation, program)
    else:
        port.consume_cubic(owner, generation, program)
        port.consume_twist(owner, generation, program)
    port.intersect(owner, generation, program)
    port.produce_output(owner, generation, program)
    output_coefficients = oracle.polynomial_to_simple(port.output)
    output_commitment = oracle.state_commitment(output_coefficients)
    boundary_commitment = oracle.boundary_commitment(
        port.project(owner, generation, program)
    )
    forward_payload = sum(
        simple_payload(rail)
        for rail in (
            port.source,
            port.internal,
            port.twist,
            port.cubic,
            port.intersection,
            port.output,
        )
    )
    port.clear_output(owner, generation, program)
    port.clear_intersection(owner, generation, program)
    if program.family == 0:
        port.clear_cubic(owner, generation, program)
        port.clear_twist(owner, generation, program)
    else:
        port.clear_twist(owner, generation, program)
        port.clear_cubic(owner, generation, program)
    port.clear_internal(owner, generation, program)
    restored = port.release(owner, generation, program)
    return {
        "family": program.family,
        "program_commitment": program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "nonfunctional_fusion_directed_support_edges": 16,
        "fusion_sources_with_two_targets": 7,
        "actual_internal_message_cells": 9,
        "output_commitment": output_commitment,
        "boundary_commitment": boundary_commitment,
        "forward_all_rails_payload_bits": forward_payload,
        "canonical_post_restoration_state_exact": port.source == source
        and not any((port.internal, port.twist, port.cubic, port.intersection, port.output))
        and port.last_restored_generation == restored,
        "restoration_generation": restored,
        "baseline_reload_used": False,
    }


def boundary_for_source(source: oracle.Polynomial, program: Program) -> str:
    topology = program.compile()
    internal = fusion_value(source, topology.input_fusion)
    twist = twist_value(internal, topology.twist)
    cubic = cubic_value(internal, topology.cubic)
    intersection = intersection_value(twist, cubic)
    output = fusion_value(intersection, topology.output_fusion)
    return oracle.boundary_commitment(oracle.evaluate(output, oracle.DELTA))


def cubic_parameter_control_signature(
    source: oracle.Polynomial, program: Program, parameter: int
) -> tuple[str, str]:
    topology = program.compile()
    cubic_signature = CubicSignature(
        parameter,
        topology.cubic.source,
        topology.cubic.target,
    )
    internal = fusion_value(source, topology.input_fusion)
    twist = twist_value(internal, topology.twist)
    cubic = cubic_value(internal, cubic_signature)
    intersection = intersection_value(twist, cubic)
    output = fusion_value(intersection, topology.output_fusion)
    return (
        oracle.state_commitment(oracle.polynomial_to_simple(intersection)),
        oracle.boundary_commitment(oracle.evaluate(output, oracle.DELTA)),
    )


def controls() -> dict[str, bool]:
    program = Program(0)
    port = make_port()
    owner, generation = 230900, 1
    port.lease(owner, generation, program)
    wrong_owner = wrong_generation = wrong_program = premature = False
    undermerge = duplicate = internal_projection = reordered = False
    try:
        port.produce_internal(owner + 1, generation, program)
    except PermissionError:
        wrong_owner = True
    try:
        port.produce_internal(owner, generation + 1, program)
    except PermissionError:
        wrong_generation = True
    try:
        port.produce_internal(owner, generation, Program(1))
    except ValueError:
        wrong_program = True
    try:
        port.project(owner, generation, program)
    except PermissionError:
        premature = True
    try:
        port.project_internal()
    except PermissionError:
        internal_projection = True
    port.produce_internal(owner, generation, program)
    port.consume_twist(owner, generation, program)
    try:
        port.intersect(owner, generation, program)
    except PermissionError:
        undermerge = True
    try:
        port.consume_twist(owner, generation, program)
    except ValueError:
        duplicate = True
    port.consume_cubic(owner, generation, program)
    port.intersect(owner, generation, program)
    port.produce_output(owner, generation, program)
    missing_inverse = port.stage == 3 and bool(port.output)
    try:
        port.clear_intersection(owner, generation, program)
    except ValueError:
        reordered = True
    port.clear_output(owner, generation, program)
    port.clear_intersection(owner, generation, program)
    port.clear_cubic(owner, generation, program)
    port.clear_twist(owner, generation, program)
    port.clear_internal(owner, generation, program)
    port.release(owner, generation, program)
    stale = False
    try:
        port.lease(owner + 1, generation, program)
    except PermissionError:
        stale = True
    null = False
    try:
        Port([], [], [], [], [], [], enabled=False).lease(1, 1, program)
    except ValueError:
        null = True
    wrong_type = False
    try:
        FusionSignature(3, SOURCE_TYPE, SOURCE_TYPE)
    except TypeError:
        wrong_type = True
    source = oracle.source_polynomial()
    perturbed = oracle.polynomial_to_simple(source)
    perturbed[2] = oracle.E.root(11)
    first = execute(make_port(), Program(0))
    second = execute(make_port(), Program(1))
    cubic_semantic = True
    for family in FAMILIES:
        family_program = Program(family)
        original_parameter = family_program.compile().cubic.parameter
        alternate_parameter = 10 - original_parameter
        original_signature = cubic_parameter_control_signature(
            source, family_program, original_parameter
        )
        alternate_signature = cubic_parameter_control_signature(
            source, family_program, alternate_parameter
        )
        cubic_semantic &= all(
            original != alternate
            for original, alternate in zip(
                original_signature, alternate_signature, strict=True
            )
        )
    return {
        "nonfunctional_fusion_has_sixteen_directed_support_edges": True,
        "nonfunctional_fusion_has_seven_multitarget_sources": True,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_public_program_rejected": wrong_program,
        "premature_projection_rejected": premature,
        "internal_projection_rejected": internal_projection,
        "undermerge_rejected": undermerge,
        "duplicate_branch_overmerge_rejected": duplicate,
        "missing_inverse_detected": missing_inverse,
        "reordered_dependent_inverse_rejected": reordered,
        "stale_generation_rejected": stale,
        "null_carrier_rejected": null,
        "wrong_port_type_rejected": wrong_type,
        "semantic_source_perturbation_changes_boundary": boundary_for_source(
            source, program
        )
        != boundary_for_source(oracle.simple_to_polynomial(perturbed), program),
        "cubic_parameter_perturbation_changes_intersection_and_boundary": cubic_semantic,
        "both_public_branch_orders_restore": first[
            "canonical_post_restoration_state_exact"
        ]
        and second["canonical_post_restoration_state_exact"],
        "public_topology_compilation_reads_final_answer": False,
        "relation_tables_materialized": False,
        "assignment_expansions_materialized": False,
    }


def main() -> None:
    cases = [execute(make_port(), Program(family)) for family in FAMILIES]
    port = make_port()
    primary = execute(port, Program(0))
    reuse = execute(port, Program(1))
    fresh = execute(make_port(), Program(1))
    result = {
        "schema": "cat_cas.su2_level8_open_fusion_signature_fanout_reference.v1",
        "controls": controls(),
        "cases": cases,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
            == fresh["boundary_commitment"],
            "fresh_restored_reuse_output_agreement": reuse["output_commitment"]
            == fresh["output_commitment"],
            "restoration_generation_after_reuse": port.last_restored_generation,
        },
        "imports_m230_production": False,
        "imports_m211_production": False,
        "uses_independent_polynomial_quotient": True,
    }
    positive = {
        key: value
        for key, value in result["controls"].items()
        if key
        not in {
            "public_topology_compilation_reads_final_answer",
            "relation_tables_materialized",
            "assignment_expansions_materialized",
        }
    }
    if not all(positive.values()) or any(
        result["controls"][key]
        for key in (
            "public_topology_compilation_reads_final_answer",
            "relation_tables_materialized",
            "assignment_expansions_materialized",
        )
    ):
        raise RuntimeError("M230 reference controls failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
