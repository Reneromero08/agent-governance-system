#!/usr/bin/env python3
"""Standalone polynomial-quotient oracle for the bounded M231 stream."""

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
            raise TypeError("reference fusion port type mismatch")


@dataclass(frozen=True)
class Topology:
    family: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("reference public family mismatch")
        FusionSignature(self.input_parameter, SOURCE_TYPE, INTERNAL_TYPE)
        FusionSignature(self.output_parameter, INTERSECTION_TYPE, OUTPUT_TYPE)

    @property
    def input_parameter(self) -> int:
        return (3, 7)[self.family]

    @property
    def twist_parameter(self) -> int:
        return self.family

    @property
    def cubic_parameter(self) -> int:
        return (7, 3)[self.family]

    @property
    def cubic_source(self) -> int:
        return 1 + self.family

    @property
    def cubic_target(self) -> int:
        return 2 - self.family

    @property
    def output_parameter(self) -> int:
        return (7, 3)[self.family]

    def integers(self) -> tuple[int, ...]:
        return (
            1,
            self.input_parameter,
            SOURCE_TYPE,
            INTERNAL_TYPE,
            2,
            self.twist_parameter,
            INTERNAL_TYPE,
            TWIST_BRANCH_TYPE,
            3,
            self.cubic_parameter,
            self.cubic_source,
            self.cubic_target,
            INTERNAL_TYPE,
            CUBIC_BRANCH_TYPE,
            4,
            TWIST_BRANCH_TYPE,
            CUBIC_BRANCH_TYPE,
            INTERSECTION_TYPE,
            1,
            self.output_parameter,
            INTERSECTION_TYPE,
            OUTPUT_TYPE,
        )


@dataclass(frozen=True)
class Program:
    family: int

    def compile(self) -> Topology:
        return Topology(self.family)

    def token(self) -> str:
        topology = self.compile()
        return "family:" + str(self.family) + "|" + ":".join(
            str(value) for value in topology.integers()
        )


def program_commitment(program: Program) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


X = [oracle.ZERO, oracle.ONE]
QUANTUM_DIMENSIONS = [
    oracle.evaluate(polynomial, oracle.DELTA)
    for polynomial in oracle.SIMPLE_BASIS
]


def fusion_polynomial(
    polynomial: oracle.Polynomial, parameter: int
) -> oracle.Polynomial:
    return oracle.quotient_reduce(
        oracle.poly_scale(
            oracle.poly_multiply(polynomial, X), oracle.E.root(parameter)
        )
    )


def twist_coefficients(
    internal: list[oracle.E], topology: Topology
) -> list[oracle.E]:
    power = oracle.TWIST_POWERS[topology.twist_parameter]
    return [
        value * oracle.E.root(power * label * (label + 2))
        for label, value in enumerate(internal)
    ]


def cubic_coefficients(
    internal: list[oracle.E],
    topology: Topology,
    *,
    parameter_override: int | None = None,
    omit_target: bool = False,
) -> list[oracle.E]:
    result = list(internal)
    if not omit_target:
        source = internal[topology.cubic_source]
        parameter = (
            topology.cubic_parameter
            if parameter_override is None
            else parameter_override
        )
        result[topology.cubic_target] = (
            result[topology.cubic_target]
            + oracle.E.root(parameter) * source * source * source
        )
    return result


def full_materialized_boundary(source: list[oracle.E], program: Program) -> oracle.E:
    """Independently materialize the complete M230 semantic composition."""
    topology = program.compile()
    internal_polynomial = fusion_polynomial(
        oracle.simple_to_polynomial(source), topology.input_parameter
    )
    internal = oracle.polynomial_to_simple(internal_polynomial)
    twist = twist_coefficients(internal, topology)
    cubic = cubic_coefficients(internal, topology)
    intersection = [
        left * right for left, right in zip(twist, cubic, strict=True)
    ]
    output = fusion_polynomial(
        oracle.simple_to_polynomial(intersection), topology.output_parameter
    )
    return oracle.evaluate(output, oracle.DELTA)


def internal_scalar(source: list[oracle.E], label: int, topology: Topology) -> oracle.E:
    lower = source[label - 1] if label else oracle.ZERO
    upper = source[label + 1] if label + 1 < oracle.SIMPLE_OBJECTS else oracle.ZERO
    return oracle.E.root(topology.input_parameter) * (lower + upper)


def twist_scalar(internal: oracle.E, label: int, topology: Topology) -> oracle.E:
    power = oracle.TWIST_POWERS[topology.twist_parameter]
    return oracle.E.root(power * label * (label + 2)) * internal


def cubic_scalar(
    internal: list[oracle.E],
    label: int,
    topology: Topology,
    *,
    parameter_override: int | None = None,
    omit_target: bool = False,
) -> oracle.E:
    result = internal[label]
    if label == topology.cubic_target and not omit_target:
        source = internal[topology.cubic_source]
        parameter = (
            topology.cubic_parameter
            if parameter_override is None
            else parameter_override
        )
        result = result + oracle.E.root(parameter) * source * source * source
    return result


def boundary_weight(
    label: int, topology: Topology, *, output_override: int | None = None
) -> oracle.E:
    lower = QUANTUM_DIMENSIONS[label - 1] if label else oracle.ZERO
    upper = (
        QUANTUM_DIMENSIONS[label + 1]
        if label + 1 < oracle.SIMPLE_OBJECTS
        else oracle.ZERO
    )
    parameter = topology.output_parameter if output_override is None else output_override
    return oracle.E.root(parameter) * (lower + upper)


def stream_boundary_from_internal(
    internal: list[oracle.E],
    topology: Topology,
    *,
    cubic_override: int | None = None,
    output_override: int | None = None,
    omit_target: bool = False,
) -> oracle.E:
    boundary = oracle.ZERO
    for label in range(oracle.SIMPLE_OBJECTS):
        left = twist_scalar(internal[label], label, topology)
        right = cubic_scalar(
            internal,
            label,
            topology,
            parameter_override=cubic_override,
            omit_target=omit_target,
        )
        boundary = boundary + boundary_weight(
            label, topology, output_override=output_override
        ) * left * right
    return boundary


def source_rematerialized_boundary(
    source: list[oracle.E],
    program: Program,
    *,
    cubic_override: int | None = None,
    output_override: int | None = None,
    omit_target: bool = False,
) -> oracle.E:
    topology = program.compile()
    source_internal = internal_scalar(source, topology.cubic_source, topology)
    boundary = oracle.ZERO
    for label in range(oracle.SIMPLE_OBJECTS):
        internal = internal_scalar(source, label, topology)
        left = twist_scalar(internal, label, topology)
        right = internal
        if label == topology.cubic_target and not omit_target:
            parameter = (
                topology.cubic_parameter
                if cubic_override is None
                else cubic_override
            )
            right = (
                right
                + oracle.E.root(parameter)
                * source_internal
                * source_internal
                * source_internal
            )
        boundary = boundary + boundary_weight(
            label, topology, output_override=output_override
        ) * left * right
    return boundary


@dataclass
class Port:
    source: list[oracle.E]
    internal: list[oracle.E]
    twist: list[oracle.E]
    cubic: list[oracle.E]
    intersection: list[oracle.E]
    boundary: list[oracle.E]
    enabled: bool = True
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    cursor: int = 0
    program_hash: str = ""

    def scratch_zero(self) -> bool:
        return all(cell == [oracle.ZERO] for cell in self.rails()[2:5])

    def rails(self) -> tuple[list[oracle.E], ...]:
        return (
            self.source,
            self.internal,
            self.twist,
            self.cubic,
            self.intersection,
            self.boundary,
        )

    def lease(self, owner: int, generation: int, program: Program) -> Topology:
        if self.live:
            raise RuntimeError("reference port already live")
        if not self.enabled:
            raise ValueError("reference null port")
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("reference invalid or stale generation")
        if (
            any(value != oracle.ZERO for value in self.internal)
            or any(value != oracle.ZERO for value in self.boundary)
            or not self.scratch_zero()
        ):
            raise ValueError("reference port dirty at lease")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.cursor = 0
        self.program_hash = program_commitment(program)
        return topology

    def require(self, owner: int, generation: int, program: Program) -> Topology:
        if not self.live:
            raise RuntimeError("reference port not live")
        if owner != self.owner:
            raise PermissionError("reference owner mismatch")
        if generation != self.generation:
            raise PermissionError("reference generation mismatch")
        if program_commitment(program) != self.program_hash:
            raise ValueError("reference program mismatch")
        return program.compile()

    def produce_internal(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 0 or self.cursor or any(
            value != oracle.ZERO for value in self.internal
        ):
            raise ValueError("reference internal production order")
        for label in range(oracle.SIMPLE_OBJECTS):
            self.internal[label] = self.internal[label] + internal_scalar(
                self.source, label, topology
            )
        self.stage = 1

    def produce_label(
        self, owner: int, generation: int, program: Program, label: int
    ) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or label != self.cursor or not self.scratch_zero():
            raise ValueError("reference forward scalar order")
        self.twist[0] = self.twist[0] + twist_scalar(
            self.internal[label], label, topology
        )
        self.cubic[0] = self.cubic[0] + cubic_scalar(
            self.internal, label, topology
        )
        self.intersection[0] = self.intersection[0] + self.twist[0] * self.cubic[0]
        self.boundary[0] = self.boundary[0] + boundary_weight(
            label, topology
        ) * self.intersection[0]
        self.intersection[0] = self.intersection[0] - self.twist[0] * self.cubic[0]
        self.cubic[0] = self.cubic[0] - cubic_scalar(
            self.internal, label, topology
        )
        self.twist[0] = self.twist[0] - twist_scalar(
            self.internal[label], label, topology
        )
        if not self.scratch_zero():
            raise RuntimeError("reference forward scratch did not clear")
        self.cursor += 1
        if self.cursor == oracle.SIMPLE_OBJECTS:
            self.stage = 2

    def project(self, owner: int, generation: int, program: Program) -> oracle.E:
        self.require(owner, generation, program)
        if (
            self.stage != 2
            or self.cursor != oracle.SIMPLE_OBJECTS
            or not self.scratch_zero()
        ):
            raise PermissionError("reference nonfinal or dirty projection")
        return self.boundary[0]

    def project_internal(self) -> None:
        raise PermissionError("reference internal message is hidden")

    def clear_label(
        self, owner: int, generation: int, program: Program, label: int
    ) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 2 or label != self.cursor - 1 or not self.scratch_zero():
            raise ValueError("reference inverse scalar order")
        self.twist[0] = self.twist[0] + twist_scalar(
            self.internal[label], label, topology
        )
        self.cubic[0] = self.cubic[0] + cubic_scalar(
            self.internal, label, topology
        )
        self.intersection[0] = self.intersection[0] + self.twist[0] * self.cubic[0]
        self.boundary[0] = self.boundary[0] - boundary_weight(
            label, topology
        ) * self.intersection[0]
        self.intersection[0] = self.intersection[0] - self.twist[0] * self.cubic[0]
        self.cubic[0] = self.cubic[0] - cubic_scalar(
            self.internal, label, topology
        )
        self.twist[0] = self.twist[0] - twist_scalar(
            self.internal[label], label, topology
        )
        if not self.scratch_zero():
            raise RuntimeError("reference inverse scratch did not clear")
        self.cursor -= 1
        if not self.cursor:
            self.stage = 1

    def clear_internal(self, owner: int, generation: int, program: Program) -> None:
        topology = self.require(owner, generation, program)
        if self.stage != 1 or self.cursor or self.boundary != [oracle.ZERO]:
            raise ValueError("reference internal inverse dependency")
        for label in range(oracle.SIMPLE_OBJECTS):
            self.internal[label] = self.internal[label] - internal_scalar(
                self.source, label, topology
            )
        if any(value != oracle.ZERO for value in self.internal):
            raise RuntimeError("reference internal did not clear")
        self.stage = 0

    def release(self, owner: int, generation: int, program: Program) -> int:
        self.require(owner, generation, program)
        if (
            self.stage
            or self.cursor
            or any(value != oracle.ZERO for value in self.internal)
            or any(value != oracle.ZERO for value in self.boundary)
            or not self.scratch_zero()
        ):
            raise RuntimeError("reference release before exact restoration")
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.program_hash = ""
        self.last_restored_generation = restored
        return restored


def make_port() -> Port:
    source = oracle.polynomial_to_simple(oracle.source_polynomial())
    return Port(
        source,
        [oracle.ZERO for _ in range(oracle.SIMPLE_OBJECTS)],
        [oracle.ZERO],
        [oracle.ZERO],
        [oracle.ZERO],
        [oracle.ZERO],
    )


def restored(port: Port, source: list[oracle.E], generation: int) -> bool:
    return (
        port.source == source
        and not any(value != oracle.ZERO for value in port.internal)
        and port.scratch_zero()
        and port.boundary == [oracle.ZERO]
        and not port.live
        and port.stage == 0
        and port.cursor == 0
        and port.last_restored_generation == generation
    )


def execute(port: Port, program: Program) -> dict[str, object]:
    source = list(port.source)
    backings = tuple(id(rail) for rail in port.rails())
    generation = port.last_restored_generation + 1
    owner = 231000 + generation
    topology = port.lease(owner, generation, program)
    port.produce_internal(owner, generation, program)
    if port.internal != oracle.polynomial_to_simple(
        fusion_polynomial(oracle.simple_to_polynomial(source), topology.input_parameter)
    ):
        raise RuntimeError("reference streamed input fusion differs from polynomial")
    for label in range(oracle.SIMPLE_OBJECTS):
        port.produce_label(owner, generation, program, label)
    boundary = port.project(owner, generation, program)
    if boundary != full_materialized_boundary(source, program):
        raise RuntimeError("reference stream differs from full materialized oracle")
    if boundary != source_rematerialized_boundary(source, program):
        raise RuntimeError("reference stream differs from strongest compact recurrence")
    boundary_commitment = oracle.boundary_commitment(boundary)
    for label in range(oracle.SIMPLE_OBJECTS - 1, -1, -1):
        port.clear_label(owner, generation, program, label)
    port.clear_internal(owner, generation, program)
    restored_generation = port.release(owner, generation, program)
    return {
        "family": program.family,
        "program_commitment": program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "actual_resident_internal_message_cells": len(port.internal),
        "scalar_branch_intersection_backing_cells": 3,
        "final_boundary_backing_cells": len(port.boundary),
        "phase_work_backing_cells_excluding_public_input": sum(
            len(rail) for rail in port.rails()[1:]
        ),
        "phase_total_backing_cells_including_public_input": sum(
            len(rail) for rail in port.rails()
        ),
        "boundary_commitment": boundary_commitment,
        "canonical_post_restoration_state_exact": restored(
            port, source, restored_generation
        )
        and tuple(id(rail) for rail in port.rails()) == backings,
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
    }


def controls() -> dict[str, bool]:
    program = Program(0)
    port = make_port()
    owner, generation = 231900, 1
    port.lease(owner, generation, program)
    wrong_owner = wrong_generation = wrong_program = premature = False
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
    internal_projection = False
    try:
        port.project_internal()
    except PermissionError:
        internal_projection = True
    port.produce_internal(owner, generation, program)
    for label in range(oracle.SIMPLE_OBJECTS):
        port.produce_label(owner, generation, program, label)
    missing_inverse = port.boundary != [oracle.ZERO]
    reordered = False
    try:
        port.clear_internal(owner, generation, program)
    except ValueError:
        reordered = True
    release_before_inverse = False
    try:
        port.release(owner, generation, program)
    except RuntimeError:
        release_before_inverse = True
    port.twist[0] = oracle.ONE
    dirty_projection = False
    try:
        port.project(owner, generation, program)
    except PermissionError:
        dirty_projection = True
    port.twist[0] = oracle.ZERO
    for label in range(oracle.SIMPLE_OBJECTS - 1, -1, -1):
        port.clear_label(owner, generation, program, label)
    port.clear_internal(owner, generation, program)
    port.release(owner, generation, program)
    stale = False
    try:
        port.lease(owner + 1, generation, program)
    except PermissionError:
        stale = True
    null = False
    try:
        null_port = make_port()
        null_port.enabled = False
        null_port.lease(1, 1, program)
    except ValueError:
        null = True
    dirty_forward_port = make_port()
    dirty_forward_port.lease(owner + 1, 1, program)
    dirty_forward_port.produce_internal(owner + 1, 1, program)
    dirty_forward_port.twist[0] = oracle.ONE
    dirty_forward = False
    try:
        dirty_forward_port.produce_label(owner + 1, 1, program, 0)
    except ValueError:
        dirty_forward = True
    wrong_type = False
    try:
        FusionSignature(3, SOURCE_TYPE, SOURCE_TYPE)
    except TypeError:
        wrong_type = True
    source = oracle.polynomial_to_simple(oracle.source_polynomial())
    perturbed = list(source)
    perturbed[2] = oracle.E.root(11)
    cubic_semantic = output_semantic = omit_semantic = True
    both_restore = True
    for family in FAMILIES:
        family_program = Program(family)
        topology = family_program.compile()
        baseline = source_rematerialized_boundary(source, family_program)
        cubic_semantic &= baseline != source_rematerialized_boundary(
            source,
            family_program,
            cubic_override=10 - topology.cubic_parameter,
        )
        output_semantic &= baseline != source_rematerialized_boundary(
            source,
            family_program,
            output_override=10 - topology.output_parameter,
        )
        omit_semantic &= baseline != source_rematerialized_boundary(
            source, family_program, omit_target=True
        )
        both_restore &= bool(execute(make_port(), family_program)[
            "canonical_post_restoration_state_exact"
        ])
    return {
        "assignment_expansions_materialized": False,
        "both_public_families_restore": both_restore,
        "cubic_parameter_perturbation_changes_boundary_all_families": cubic_semantic,
        "dirty_scalar_scratch_projection_rejected": dirty_projection,
        "dirty_scalar_scratch_rejected": dirty_forward,
        "internal_projection_rejected": internal_projection,
        "missing_inverse_detected": missing_inverse,
        "null_carrier_rejected": null,
        "omitted_cubic_target_changes_boundary_all_families": omit_semantic,
        "output_phase_perturbation_changes_boundary_all_families": output_semantic,
        "premature_projection_rejected": premature,
        "public_topology_compilation_reads_final_answer": False,
        "relation_tables_materialized": False,
        "release_before_inverse_rejected": release_before_inverse,
        "reordered_dependent_inverse_rejected": reordered,
        "semantic_source_perturbation_changes_boundary": (
            full_materialized_boundary(source, program)
            != full_materialized_boundary(perturbed, program)
        ),
        "stale_generation_rejected": stale,
        "wrong_generation_rejected": wrong_generation,
        "wrong_owner_rejected": wrong_owner,
        "wrong_port_type_rejected": wrong_type,
        "wrong_public_program_rejected": wrong_program,
    }


def main() -> None:
    cases = [execute(make_port(), Program(family)) for family in FAMILIES]
    port = make_port()
    primary = execute(port, Program(0))
    reuse = execute(port, Program(1))
    fresh = execute(make_port(), Program(1))
    result = {
        "schema": "cat_cas.su2_level8_open_fusion_rematerialized_reference.v1",
        "controls": controls(),
        "cases": cases,
        "reuse": {
            "primary": primary,
            "reuse": reuse,
            "fresh_reuse": fresh,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
            == fresh["boundary_commitment"],
            "restoration_generation_after_reuse": port.last_restored_generation,
        },
        "imports_m231_production": False,
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
        raise RuntimeError("M231 standalone controls failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
