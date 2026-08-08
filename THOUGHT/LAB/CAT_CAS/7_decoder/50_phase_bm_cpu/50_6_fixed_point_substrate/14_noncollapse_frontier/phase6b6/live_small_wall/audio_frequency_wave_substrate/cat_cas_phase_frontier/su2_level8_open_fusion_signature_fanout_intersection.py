#!/usr/bin/env python3
"""M230 exact open SU(2)_8 fusion-signature fanout and intersection.

The fundamental fusion signature is the nonfunctional path relation
N_(1,a)^b.  A reversible accumulator shear produces its actual nine-cell
internal message.  Separate twist and cubic-shear branch backings consume that
same resident message, Hadamard intersection closes the branches, and a final
fusion signature produces the only projectable boundary rail.  Reverse shears
clear every produced rail and restore the same carrier for unrelated reuse.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import root_of_unity_su2_level8_fusion_phase_relation as su2


sys.set_int_max_str_digits(0)

SOURCE_TYPE = 23001
INTERNAL_TYPE = 23002
TWIST_BRANCH_TYPE = 23003
CUBIC_BRANCH_TYPE = 23004
INTERSECTION_TYPE = 23005
OUTPUT_TYPE = 23006
FAMILIES = (0, 1)
PRIMARY_FAMILY = 0
REUSE_FAMILY = 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


@dataclass(frozen=True)
class FusionSignature:
    parameter: int
    input_type: int
    output_type: int

    def __post_init__(self) -> None:
        if self.parameter not in (3, 7):
            raise ValueError("unknown fusion-signature phase")
        if (self.input_type, self.output_type) not in {
            (SOURCE_TYPE, INTERNAL_TYPE),
            (INTERSECTION_TYPE, OUTPUT_TYPE),
        }:
            raise TypeError("fusion signature open-port type mismatch")

    def integers(self) -> tuple[int, ...]:
        return (1, self.parameter, self.input_type, self.output_type)


@dataclass(frozen=True)
class TwistBranchSignature:
    parameter: int
    input_type: int = INTERNAL_TYPE
    output_type: int = TWIST_BRANCH_TYPE

    def __post_init__(self) -> None:
        if self.parameter not in (0, 1):
            raise ValueError("unknown twist branch")
        if self.input_type != INTERNAL_TYPE or self.output_type != TWIST_BRANCH_TYPE:
            raise TypeError("twist branch type mismatch")

    def integers(self) -> tuple[int, ...]:
        return (2, self.parameter, self.input_type, self.output_type)


@dataclass(frozen=True)
class CubicBranchSignature:
    parameter: int
    source: int
    target: int
    input_type: int = INTERNAL_TYPE
    output_type: int = CUBIC_BRANCH_TYPE

    def __post_init__(self) -> None:
        if self.parameter not in (3, 7):
            raise ValueError("unknown cubic branch phase")
        if not (0 <= self.source < su2.SIMPLE_OBJECTS):
            raise ValueError("cubic source outside relation message")
        if not (0 <= self.target < su2.SIMPLE_OBJECTS):
            raise ValueError("cubic target outside relation message")
        if self.source == self.target:
            raise ValueError("cubic branch source equals target")
        if self.input_type != INTERNAL_TYPE or self.output_type != CUBIC_BRANCH_TYPE:
            raise TypeError("cubic branch type mismatch")

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
            raise TypeError("intersection signature type mismatch")

    def integers(self) -> tuple[int, ...]:
        return (4, self.left_type, self.right_type, self.output_type)


@dataclass(frozen=True)
class CompiledTopology:
    input_fusion: FusionSignature
    twist_branch: TwistBranchSignature
    cubic_branch: CubicBranchSignature
    intersection: IntersectionSignature
    output_fusion: FusionSignature

    def descriptors(self) -> tuple[object, ...]:
        return (
            self.input_fusion,
            self.twist_branch,
            self.cubic_branch,
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
class PublicProgram:
    family: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("unknown M230 public family")

    def compile(self) -> CompiledTopology:
        source = 1 + self.family
        target = 2 - self.family
        return CompiledTopology(
            FusionSignature((3, 7)[self.family], SOURCE_TYPE, INTERNAL_TYPE),
            TwistBranchSignature(self.family),
            CubicBranchSignature((7, 3)[self.family], source, target),
            IntersectionSignature(),
            FusionSignature((7, 3)[self.family], INTERSECTION_TYPE, OUTPUT_TYPE),
        )

    def token(self) -> str:
        return f"family:{self.family}|{self.compile().token()}"


def program_commitment(program: PublicProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


@dataclass
class Work(su2.Work):
    fusion_signature_shears: int = 0
    fusion_signature_neighbor_additions: int = 0
    fusion_signature_phase_multiplications: int = 0
    twist_branch_shears: int = 0
    cubic_branch_shears: int = 0
    intersection_shears: int = 0
    branch_field_multiplications: int = 0
    branch_field_additions: int = 0
    branch_field_subtractions: int = 0
    internal_message_productions: int = 0
    internal_message_clears: int = 0
    distinct_branch_backing_mask: int = 0
    typed_topology_checks: int = 0
    owner_checks: int = 0
    generation_checks: int = 0
    program_commitment_checks: int = 0
    premature_projection_rejections: int = 0
    relation_table_cells_materialized: int = 0
    assignment_expansions_materialized: int = 0
    maximum_declared_live_field_cells: int = 0
    maximum_declared_live_payload_bits: int = 0
    maximum_declared_live_context: str = ""
    retained_topology_integers: tuple[int, ...] = field(default_factory=tuple, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "retained_topology_integers"
        }

    def observe_relation(
        self,
        rails: Sequence[list[su2.K]],
        *,
        transients: Sequence[su2.K] = (),
        context: str,
    ) -> None:
        descriptor_bits = sum(
            signed_bits(value) for value in self.retained_topology_integers
        )
        field_cells = sum(len(rail) for rail in rails) + len(transients)
        payload = sum(su2.field_payload_bits(rail) for rail in rails)
        payload += su2.field_payload_bits(tuple(transients)) + descriptor_bits
        self.maximum_declared_live_field_cells = max(
            self.maximum_declared_live_field_cells, field_cells
        )
        if payload > self.maximum_declared_live_payload_bits:
            self.maximum_declared_live_payload_bits = payload
            self.maximum_declared_live_context = context


def require_zero(rail: list[su2.K], name: str) -> None:
    if len(rail) != su2.SIMPLE_OBJECTS or any(value != su2.ZERO for value in rail):
        raise ValueError(f"{name} relation rail is not a clean nine-cell backing")


def fusion_accumulate(
    target: list[su2.K],
    source: list[su2.K],
    signature: FusionSignature,
    *,
    subtract: bool,
    work: Work,
    rails: Sequence[list[su2.K]],
    context: str,
) -> None:
    phase = su2.K.zeta(signature.parameter)
    for label in range(su2.SIMPLE_OBJECTS):
        lower = source[label - 1] if label else su2.ZERO
        upper = source[label + 1] if label + 1 < su2.SIMPLE_OBJECTS else su2.ZERO
        neighbor_sum = work.add(lower, upper)
        weighted = work.multiply(phase, neighbor_sum)
        updated = (
            work.subtract(target[label], weighted)
            if subtract
            else work.add(target[label], weighted)
        )
        work.fusion_signature_neighbor_additions += 1
        work.fusion_signature_phase_multiplications += 1
        work.observe_relation(
            rails,
            transients=(phase, neighbor_sum, weighted, updated),
            context=context,
        )
        target[label] = updated
    work.fusion_signature_shears += 1


def twist_accumulate(
    target: list[su2.K],
    source: list[su2.K],
    signature: TwistBranchSignature,
    *,
    subtract: bool,
    work: Work,
    rails: Sequence[list[su2.K]],
    context: str,
) -> None:
    for label in range(su2.SIMPLE_OBJECTS):
        phase = su2.twist_multiplier(label, signature.parameter, False)
        weighted = work.multiply(phase, source[label])
        updated = (
            work.subtract(target[label], weighted)
            if subtract
            else work.add(target[label], weighted)
        )
        work.branch_field_multiplications += 1
        work.branch_field_subtractions += int(subtract)
        work.branch_field_additions += int(not subtract)
        work.observe_relation(
            rails,
            transients=(phase, weighted, updated),
            context=context,
        )
        target[label] = updated
    work.twist_branch_shears += 1


def cubic_accumulate(
    target: list[su2.K],
    source: list[su2.K],
    signature: CubicBranchSignature,
    *,
    subtract: bool,
    work: Work,
    rails: Sequence[list[su2.K]],
    context: str,
) -> None:
    phase = su2.K.zeta(signature.parameter)
    source_value = source[signature.source]
    square = work.multiply(source_value, source_value)
    cube = work.multiply(square, source_value)
    cubic_term = work.multiply(phase, cube)
    work.branch_field_multiplications += 3
    for label in range(su2.SIMPLE_OBJECTS):
        branch_value = (
            work.add(source[label], cubic_term)
            if label == signature.target
            else source[label]
        )
        if label == signature.target:
            work.branch_field_additions += 1
        updated = (
            work.subtract(target[label], branch_value)
            if subtract
            else work.add(target[label], branch_value)
        )
        work.branch_field_subtractions += int(subtract)
        work.branch_field_additions += int(not subtract)
        work.observe_relation(
            rails,
            transients=(phase, square, cube, cubic_term, branch_value, updated),
            context=context,
        )
        target[label] = updated
    work.cubic_branch_shears += 1


def intersection_accumulate(
    target: list[su2.K],
    left: list[su2.K],
    right: list[su2.K],
    *,
    subtract: bool,
    work: Work,
    rails: Sequence[list[su2.K]],
    context: str,
) -> None:
    for label in range(su2.SIMPLE_OBJECTS):
        product = work.multiply(left[label], right[label])
        updated = (
            work.subtract(target[label], product)
            if subtract
            else work.add(target[label], product)
        )
        work.branch_field_multiplications += 1
        work.branch_field_subtractions += int(subtract)
        work.branch_field_additions += int(not subtract)
        work.observe_relation(
            rails, transients=(product, updated), context=context
        )
        target[label] = updated
    work.intersection_shears += 1


def project_output(output: list[su2.K], work: Work) -> su2.K:
    value = su2.ZERO
    for coefficient, dimension in zip(output, su2.QUANTUM_DIMENSIONS, strict=True):
        weighted = work.multiply(coefficient, dimension)
        updated = work.add(value, weighted)
        work.boundary_multiplications += 1
        work.boundary_additions += 1
        value = updated
    return value


@dataclass
class RelationCarrier:
    source: list[su2.K]
    internal: list[su2.K]
    twist_branch: list[su2.K]
    cubic_branch: list[su2.K]
    intersection: list[su2.K]
    output: list[su2.K]
    live: bool = False
    owner: int = 0
    generation: int = 0
    last_restored_generation: int = 0
    stage: int = 0
    branch_mask: int = 0
    sealed_program_commitment: str = ""

    def rails(self) -> tuple[list[su2.K], ...]:
        return (
            self.source,
            self.internal,
            self.twist_branch,
            self.cubic_branch,
            self.intersection,
            self.output,
        )

    def lease(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> CompiledTopology:
        if self.live:
            raise RuntimeError("relation carrier already live")
        if len(self.source) != su2.SIMPLE_OBJECTS:
            raise ValueError("null or wrong-width source relation rail")
        for name, rail in zip(
            ("internal", "twist", "cubic", "intersection", "output"),
            self.rails()[1:],
            strict=True,
        ):
            require_zero(rail, name)
        if owner <= 0 or generation != self.last_restored_generation + 1:
            raise PermissionError("invalid or stale relation carrier lease")
        topology = program.compile()
        self.live = True
        self.owner = owner
        self.generation = generation
        self.stage = 0
        self.branch_mask = 0
        self.sealed_program_commitment = program_commitment(program)
        work.retained_topology_integers = topology.integers()
        work.port_leases += 1
        work.observe_relation(self.rails(), context="RELATION_LEASE")
        return topology

    def require(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> CompiledTopology:
        if not self.live:
            raise RuntimeError("relation carrier not live")
        work.owner_checks += 1
        if owner != self.owner:
            raise PermissionError("relation owner mismatch")
        work.generation_checks += 1
        if generation != self.generation:
            raise PermissionError("relation generation mismatch")
        work.program_commitment_checks += 1
        if program_commitment(program) != self.sealed_program_commitment:
            raise ValueError("relation public program mismatch")
        work.typed_topology_checks += 1
        return program.compile()

    def produce_internal(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 0 or self.branch_mask:
            raise ValueError("internal relation production out of order")
        fusion_accumulate(
            self.internal,
            self.source,
            topology.input_fusion,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="PRODUCE_NONFUNCTIONAL_FUSION_MESSAGE",
        )
        self.stage = 1
        work.internal_message_productions += 1
        work.forward_operations += 1

    def consume_twist(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or self.branch_mask & 1:
            raise ValueError("twist branch consumption out of order")
        twist_accumulate(
            self.twist_branch,
            self.internal,
            topology.twist_branch,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="CONSUME_SHARED_INTERNAL_TWIST_BRANCH",
        )
        self.branch_mask |= 1
        work.distinct_branch_backing_mask |= 1
        work.forward_operations += 1

    def consume_cubic(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or self.branch_mask & 2:
            raise ValueError("cubic branch consumption out of order")
        cubic_accumulate(
            self.cubic_branch,
            self.internal,
            topology.cubic_branch,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="CONSUME_SHARED_INTERNAL_CUBIC_BRANCH",
        )
        self.branch_mask |= 2
        work.distinct_branch_backing_mask |= 2
        work.forward_operations += 1

    def intersect(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        self.require(owner, generation, program, work)
        if self.stage != 1 or self.branch_mask != 3:
            raise PermissionError("intersection requires both resident branches")
        intersection_accumulate(
            self.intersection,
            self.twist_branch,
            self.cubic_branch,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="NATIVE_BRANCH_HADAMARD_INTERSECTION",
        )
        self.stage = 2
        work.forward_operations += 1

    def produce_output(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 2:
            raise ValueError("output relation production out of order")
        fusion_accumulate(
            self.output,
            self.intersection,
            topology.output_fusion,
            subtract=False,
            work=work,
            rails=self.rails(),
            context="PRODUCE_FINAL_FUSION_RELATION_MESSAGE",
        )
        self.stage = 3
        work.forward_operations += 1

    def project(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> su2.K:
        self.require(owner, generation, program, work)
        if self.stage != 3:
            work.premature_projection_rejections += 1
            raise PermissionError("only final relation output is projectable")
        return project_output(self.output, work)

    def project_internal(self) -> None:
        raise PermissionError("shared internal relation message is never projectable")

    def clear_output(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 3:
            raise ValueError("output inverse out of order")
        fusion_accumulate(
            self.output,
            self.intersection,
            topology.output_fusion,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_FINAL_FUSION_RELATION_MESSAGE",
        )
        require_zero(self.output, "output")
        self.stage = 2
        work.inverse_operations += 1

    def clear_intersection(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        self.require(owner, generation, program, work)
        if self.stage != 2:
            raise ValueError("intersection inverse out of order")
        intersection_accumulate(
            self.intersection,
            self.twist_branch,
            self.cubic_branch,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_NATIVE_BRANCH_INTERSECTION",
        )
        require_zero(self.intersection, "intersection")
        self.stage = 1
        work.inverse_operations += 1

    def clear_twist(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or not self.branch_mask & 1:
            raise ValueError("twist inverse out of order")
        twist_accumulate(
            self.twist_branch,
            self.internal,
            topology.twist_branch,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_TWIST_BRANCH",
        )
        require_zero(self.twist_branch, "twist")
        self.branch_mask &= ~1
        work.inverse_operations += 1

    def clear_cubic(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or not self.branch_mask & 2:
            raise ValueError("cubic inverse out of order")
        cubic_accumulate(
            self.cubic_branch,
            self.internal,
            topology.cubic_branch,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_CUBIC_BRANCH",
        )
        require_zero(self.cubic_branch, "cubic")
        self.branch_mask &= ~2
        work.inverse_operations += 1

    def clear_internal(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> None:
        topology = self.require(owner, generation, program, work)
        if self.stage != 1 or self.branch_mask:
            raise ValueError("internal inverse requires both branches cleared")
        fusion_accumulate(
            self.internal,
            self.source,
            topology.input_fusion,
            subtract=True,
            work=work,
            rails=self.rails(),
            context="CLEAR_NONFUNCTIONAL_FUSION_MESSAGE",
        )
        require_zero(self.internal, "internal")
        self.stage = 0
        work.internal_message_clears += 1
        work.inverse_operations += 1

    def release(
        self, owner: int, generation: int, program: PublicProgram, work: Work
    ) -> int:
        self.require(owner, generation, program, work)
        if self.stage or self.branch_mask:
            raise RuntimeError("relation carrier released before full inverse")
        for name, rail in zip(
            ("internal", "twist", "cubic", "intersection", "output"),
            self.rails()[1:],
            strict=True,
        ):
            require_zero(rail, name)
        restored = self.generation
        self.live = False
        self.owner = 0
        self.generation = 0
        self.sealed_program_commitment = ""
        self.last_restored_generation = restored
        work.port_releases += 1
        return restored


def make_carrier() -> tuple[RelationCarrier, list[su2.K]]:
    source = su2.source_state()
    zero = lambda: [su2.ZERO] * su2.SIMPLE_OBJECTS
    return RelationCarrier(source.copy(), zero(), zero(), zero(), zero(), zero()), source


def carrier_restored(
    carrier: RelationCarrier, source: list[su2.K], generation: int
) -> bool:
    return (
        carrier.source == source
        and all(all(value == su2.ZERO for value in rail) for rail in carrier.rails()[1:])
        and not carrier.live
        and carrier.owner == 0
        and carrier.generation == 0
        and carrier.last_restored_generation == generation
        and carrier.stage == 0
        and carrier.branch_mask == 0
        and carrier.sealed_program_commitment == ""
    )


def compact_classical_boundary(
    source: list[su2.K], program: PublicProgram
) -> dict[str, Any]:
    topology = program.compile()
    input_phase = su2.K.zeta(topology.input_fusion.parameter)
    output_phase = su2.K.zeta(topology.output_fusion.parameter)
    internal = [su2.ZERO] * su2.SIMPLE_OBJECTS
    additions = multiplications = 0
    for label in range(su2.SIMPLE_OBJECTS):
        lower = source[label - 1] if label else su2.ZERO
        upper = source[label + 1] if label + 1 < su2.SIMPLE_OBJECTS else su2.ZERO
        internal[label] = input_phase * (lower + upper)
        additions += 1
        multiplications += 1
    cubic_source = internal[topology.cubic_branch.source]
    cubic_term = (
        su2.K.zeta(topology.cubic_branch.parameter)
        * cubic_source
        * cubic_source
        * cubic_source
    )
    multiplications += 3

    def branch_intersection(label: int) -> su2.K:
        nonlocal additions, multiplications
        twist = su2.twist_multiplier(label, topology.twist_branch.parameter, False)
        twist_value = twist * internal[label]
        cubic_value = internal[label]
        if label == topology.cubic_branch.target:
            cubic_value = cubic_value + cubic_term
            additions += 1
        multiplications += 2
        return twist_value * cubic_value

    boundary = su2.ZERO
    for output_label, dimension in enumerate(su2.QUANTUM_DIMENSIONS):
        lower = branch_intersection(output_label - 1) if output_label else su2.ZERO
        upper = (
            branch_intersection(output_label + 1)
            if output_label + 1 < su2.SIMPLE_OBJECTS
            else su2.ZERO
        )
        output_value = output_phase * (lower + upper)
        boundary = boundary + output_value * dimension
        additions += 2
        multiplications += 2
    return {
        "boundary_commitment": su2.boundary_commitment(boundary),
        "public_input_field_cells": len(source),
        "working_resident_field_cells": len(internal),
        "total_resident_field_cells_including_public_input": len(source)
        + len(internal),
        "public_input_payload_bits": su2.field_payload_bits(source),
        "working_resident_payload_bits": su2.field_payload_bits(internal),
        "total_resident_payload_bits_including_public_input": su2.field_payload_bits(
            source
        )
        + su2.field_payload_bits(internal),
        "materialized_twist_branch_cells": 0,
        "materialized_cubic_branch_cells": 0,
        "materialized_intersection_cells": 0,
        "materialized_output_cells": 0,
        "field_additions": additions,
        "field_multiplications": multiplications,
        "recurrence": "NINE_CELL_INTERNAL_PLUS_STREAMED_BRANCH_INTERSECTION_AND_TRANSPOSE_FUSION_BOUNDARY",
    }


def cubic_parameter_control_signature(
    source: list[su2.K], program: PublicProgram, parameter: int
) -> tuple[str, str]:
    """Control-only full rails for cubic semantic sensitivity."""
    topology = program.compile()
    cubic_signature = CubicBranchSignature(
        parameter,
        topology.cubic_branch.source,
        topology.cubic_branch.target,
    )
    zero = lambda: [su2.ZERO] * su2.SIMPLE_OBJECTS
    internal, twist, cubic, intersection, output = (
        zero(),
        zero(),
        zero(),
        zero(),
        zero(),
    )
    rails = (source, internal, twist, cubic, intersection, output)
    work = Work(retained_topology_integers=topology.integers())
    fusion_accumulate(
        internal,
        source,
        topology.input_fusion,
        subtract=False,
        work=work,
        rails=rails,
        context="CONTROL_INTERNAL",
    )
    twist_accumulate(
        twist,
        internal,
        topology.twist_branch,
        subtract=False,
        work=work,
        rails=rails,
        context="CONTROL_TWIST",
    )
    cubic_accumulate(
        cubic,
        internal,
        cubic_signature,
        subtract=False,
        work=work,
        rails=rails,
        context="CONTROL_CUBIC",
    )
    intersection_accumulate(
        intersection,
        twist,
        cubic,
        subtract=False,
        work=work,
        rails=rails,
        context="CONTROL_INTERSECTION",
    )
    fusion_accumulate(
        output,
        intersection,
        topology.output_fusion,
        subtract=False,
        work=work,
        rails=rails,
        context="CONTROL_OUTPUT",
    )
    return (
        su2.state_commitment(intersection),
        su2.boundary_commitment(project_output(output, work)),
    )


def transaction(
    carrier: RelationCarrier, source: list[su2.K], program: PublicProgram
) -> dict[str, Any]:
    backings = tuple(id(rail) for rail in carrier.rails())
    generation = carrier.last_restored_generation + 1
    owner = 230000 + generation
    work = Work()
    topology = carrier.lease(owner, generation, program, work)
    carrier.produce_internal(owner, generation, program, work)
    internal_backing_at_twist = id(carrier.internal)
    if program.family == 0:
        carrier.consume_twist(owner, generation, program, work)
        carrier.consume_cubic(owner, generation, program, work)
    else:
        carrier.consume_cubic(owner, generation, program, work)
        carrier.consume_twist(owner, generation, program, work)
    internal_backing_at_cubic = id(carrier.internal)
    carrier.intersect(owner, generation, program, work)
    carrier.produce_output(owner, generation, program, work)
    output_commitment = su2.state_commitment(carrier.output)
    boundary = carrier.project(owner, generation, program, work)
    boundary_commitment = su2.boundary_commitment(boundary)
    forward_payload_bits = sum(su2.field_payload_bits(rail) for rail in carrier.rails())
    classical = compact_classical_boundary(source, program)
    if boundary_commitment != classical["boundary_commitment"]:
        raise RuntimeError("M230 compact classical boundary differs")
    carrier.clear_output(owner, generation, program, work)
    carrier.clear_intersection(owner, generation, program, work)
    if program.family == 0:
        carrier.clear_cubic(owner, generation, program, work)
        carrier.clear_twist(owner, generation, program, work)
    else:
        carrier.clear_twist(owner, generation, program, work)
        carrier.clear_cubic(owner, generation, program, work)
    carrier.clear_internal(owner, generation, program, work)
    restored_generation = carrier.release(owner, generation, program, work)
    return {
        "family": program.family,
        "program_commitment": program_commitment(program),
        "topology_descriptor_integers": len(topology.integers()),
        "nonfunctional_fusion_directed_support_edges": 2 * (su2.SIMPLE_OBJECTS - 1),
        "fusion_sources_with_two_targets": su2.SIMPLE_OBJECTS - 2,
        "actual_internal_message_cells": len(carrier.internal),
        "distinct_branch_backings": id(carrier.twist_branch) != id(carrier.cubic_branch),
        "same_internal_backing_seen_by_both_branches": internal_backing_at_twist
        == internal_backing_at_cubic
        == id(carrier.internal),
        "output_commitment": output_commitment,
        "boundary_commitment": boundary_commitment,
        "forward_all_rails_payload_bits": forward_payload_bits,
        "same_rail_backings": tuple(id(rail) for rail in carrier.rails()) == backings,
        "canonical_post_restoration_state_exact": carrier_restored(
            carrier, source, restored_generation
        ),
        "restoration_generation": restored_generation,
        "baseline_reload_used": False,
        "work": work.as_dict(),
        "matched_compact_classical": classical,
    }


def normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "family",
            "program_commitment",
            "topology_descriptor_integers",
            "nonfunctional_fusion_directed_support_edges",
            "fusion_sources_with_two_targets",
            "actual_internal_message_cells",
            "output_commitment",
            "boundary_commitment",
            "forward_all_rails_payload_bits",
            "canonical_post_restoration_state_exact",
            "restoration_generation",
            "baseline_reload_used",
        )
    }


def controls() -> dict[str, bool]:
    program = PublicProgram(0)
    carrier, source = make_carrier()
    work = Work()
    owner, generation = 230900, 1
    carrier.lease(owner, generation, program, work)
    wrong_owner = wrong_generation = wrong_program = premature = False
    undermerge = duplicate_branch = internal_projection = reordered_inverse = False
    try:
        carrier.produce_internal(owner + 1, generation, program, work)
    except PermissionError:
        wrong_owner = True
    try:
        carrier.produce_internal(owner, generation + 1, program, work)
    except PermissionError:
        wrong_generation = True
    try:
        carrier.produce_internal(owner, generation, PublicProgram(1), work)
    except ValueError:
        wrong_program = True
    try:
        carrier.project(owner, generation, program, work)
    except PermissionError:
        premature = True
    try:
        carrier.project_internal()
    except PermissionError:
        internal_projection = True
    carrier.produce_internal(owner, generation, program, work)
    carrier.consume_twist(owner, generation, program, work)
    try:
        carrier.intersect(owner, generation, program, work)
    except PermissionError:
        undermerge = True
    try:
        carrier.consume_twist(owner, generation, program, work)
    except ValueError:
        duplicate_branch = True
    carrier.consume_cubic(owner, generation, program, work)
    carrier.intersect(owner, generation, program, work)
    carrier.produce_output(owner, generation, program, work)
    missing_inverse = carrier.stage == 3 and any(
        value != su2.ZERO for value in carrier.output
    )
    try:
        carrier.clear_intersection(owner, generation, program, work)
    except ValueError:
        reordered_inverse = True
    carrier.clear_output(owner, generation, program, work)
    carrier.clear_intersection(owner, generation, program, work)
    carrier.clear_cubic(owner, generation, program, work)
    carrier.clear_twist(owner, generation, program, work)
    carrier.clear_internal(owner, generation, program, work)
    carrier.release(owner, generation, program, work)
    stale_generation = False
    try:
        carrier.lease(owner + 1, generation, program, Work())
    except PermissionError:
        stale_generation = True

    null_carrier = False
    try:
        RelationCarrier([], [], [], [], [], []).lease(1, 1, program, Work())
    except ValueError:
        null_carrier = True
    wrong_type = False
    try:
        FusionSignature(3, SOURCE_TYPE, SOURCE_TYPE)
    except TypeError:
        wrong_type = True
    original = compact_classical_boundary(source, program)["boundary_commitment"]
    perturbed_source = source.copy()
    perturbed_source[2] = su2.K.zeta(11)
    perturbed = compact_classical_boundary(perturbed_source, program)[
        "boundary_commitment"
    ]
    branch_order_carrier, branch_order_source = make_carrier()
    first = transaction(branch_order_carrier, branch_order_source, PublicProgram(0))
    second_carrier, second_source = make_carrier()
    second = transaction(second_carrier, second_source, PublicProgram(1))
    cubic_semantic = True
    for family in FAMILIES:
        family_program = PublicProgram(family)
        original_parameter = family_program.compile().cubic_branch.parameter
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
        "nonfunctional_fusion_has_sixteen_directed_support_edges": 2
        * (su2.SIMPLE_OBJECTS - 1)
        == 16,
        "nonfunctional_fusion_has_seven_multitarget_sources": su2.SIMPLE_OBJECTS - 2
        == 7,
        "wrong_owner_rejected": wrong_owner,
        "wrong_generation_rejected": wrong_generation,
        "wrong_public_program_rejected": wrong_program,
        "premature_projection_rejected": premature,
        "internal_projection_rejected": internal_projection,
        "undermerge_rejected": undermerge,
        "duplicate_branch_overmerge_rejected": duplicate_branch,
        "missing_inverse_detected": missing_inverse,
        "reordered_dependent_inverse_rejected": reordered_inverse,
        "stale_generation_rejected": stale_generation,
        "null_carrier_rejected": null_carrier,
        "wrong_port_type_rejected": wrong_type,
        "semantic_source_perturbation_changes_boundary": original != perturbed,
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
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: su2_level8_open_fusion_signature_fanout_intersection.py REFERENCE_JSON"
        )
    here = Path(__file__).resolve().parent
    reference_path = Path(sys.argv[1]).resolve()
    if str(reference_path).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M230 reference forbidden")
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") != "cat_cas.su2_level8_open_fusion_signature_fanout_reference.v1":
        raise RuntimeError("M230 reference schema mismatch")
    current_controls = controls()
    if current_controls != reference.get("controls"):
        raise RuntimeError("M230 control parity failed")
    cases = []
    for family in FAMILIES:
        carrier, source = make_carrier()
        cases.append(transaction(carrier, source, PublicProgram(family)))
    if [normalized_case(case) for case in cases] != reference.get("cases"):
        raise RuntimeError("M230 independent case parity failed")

    carrier, source = make_carrier()
    primary = transaction(carrier, source, PublicProgram(PRIMARY_FAMILY))
    reuse = transaction(carrier, source, PublicProgram(REUSE_FAMILY))
    fresh, fresh_source = make_carrier()
    fresh_reuse = transaction(fresh, fresh_source, PublicProgram(REUSE_FAMILY))
    reuse_result = {
        "primary": primary,
        "reuse": reuse,
        "fresh_reuse": fresh_reuse,
        "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"]
        == fresh_reuse["boundary_commitment"],
        "fresh_restored_reuse_output_agreement": reuse["output_commitment"]
        == fresh_reuse["output_commitment"],
        "restoration_generation_after_reuse": carrier.last_restored_generation,
    }
    for key in ("primary", "reuse", "fresh_reuse"):
        if normalized_case(reuse_result[key]) != reference["reuse"][key]:
            raise RuntimeError(f"M230 reuse parity failed: {key}")
    for key in (
        "fresh_restored_reuse_boundary_agreement",
        "fresh_restored_reuse_output_agreement",
        "restoration_generation_after_reuse",
    ):
        if reuse_result[key] != reference["reuse"][key]:
            raise RuntimeError(f"M230 top-level reuse parity failed: {key}")

    primary_case = cases[PRIMARY_FAMILY]
    result = {
        "schema": "cat_cas.su2_level8_open_fusion_signature_fanout.v1",
        "result": "PASS_BOUNDED_EXACT_OPEN_FUSION_SIGNATURE_FANOUT_INTERSECTION_WITH_SMALLER_CLASSICAL_STREAM",
        "claim": "BOUNDED_EXACT_TYPED_OPEN_SU2_LEVEL8_NONFUNCTIONAL_FUNDAMENTAL_FUSION_SIGNATURE_PRODUCES_ONE_ACTUAL_SHARED9_CELL_INTERNAL_RELATION_MESSAGE_CONSUMED_BY_SEPARATE_TWIST_AND_CUBIC_SHEAR_BRANCH_BACKINGS_AND_CLOSED_BY_NATIVE_HADAMARD_INTERSECTION_WITHOUT_RELATION_TABLES_WITH_FINAL_ONLY_BOUNDARY_EXACT_SAME_BACKING_RESTORATION_AND_REUSE_BUT_AN_EXECUTED9_CELL_STREAMED_CLASSICAL_CONTRACTION_IS_SMALLER",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "QZETA40_SU2_LEVEL8_FIXED_NONFUNCTIONAL_FUNDAMENTAL_FUSION_SIGNATURE_TWO_PUBLIC_TWIST_CUBIC_BRANCH_FAMILIES_SIX9_CELL_RAILS_PRIMARY_FAMILY0_REUSE_FAMILY1_DIRECT_PROCESS_ONLY",
        "controls": current_controls,
        "cases": cases,
        "reuse": reuse_result,
        "relation_law": {
            "nonfunctional_open_fusion_signature_constructed": True,
            "fusion_signature_support_materialized_as_table": False,
            "fusion_signature_analytic_neighbor_law": True,
            "actual_produced_internal_relation_message_cells": 9,
            "same_internal_message_consumed_by_two_branches": True,
            "distinct_twist_and_cubic_branch_backings": True,
            "native_hadamard_intersection": True,
            "internal_message_projected": False,
            "final_boundary_only": True,
            "direct_process_logical_custody_only": True,
        },
        "resource_law": {
            "resident_relation_rail_field_cells": 6 * su2.SIMPLE_OBJECTS,
            "public_source_field_cells": su2.SIMPLE_OBJECTS,
            "working_relation_rail_field_cells_excluding_public_source": 5
            * su2.SIMPLE_OBJECTS,
            "primary_forward_all_rails_payload_bits": primary_case[
                "forward_all_rails_payload_bits"
            ],
            "primary_maximum_declared_live_field_cells": primary_case["work"][
                "maximum_declared_live_field_cells"
            ],
            "primary_maximum_declared_live_payload_bits": primary_case["work"][
                "maximum_declared_live_payload_bits"
            ],
            "primary_maximum_declared_context": primary_case["work"][
                "maximum_declared_live_context"
            ],
            "retained_public_topology_descriptor_integers": primary_case[
                "topology_descriptor_integers"
            ],
            "kernel_transients_counted_at_observed_intervals": True,
            "projected_boundary_retention_during_inverse_counted": False,
            "whole_transaction_live_payload_complete": False,
            "excluded_not_zero": "PYTHON_OBJECT_CONTAINER_ALLOCATOR_INTERPRETER_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_RSS",
        },
        "matched_compact_classical": {
            "strongest": "NINE_CELL_INTERNAL_PLUS_STREAMED_BRANCH_INTERSECTION_AND_TRANSPOSE_FUSION_BOUNDARY",
            "public_input_field_cells": primary_case["matched_compact_classical"][
                "public_input_field_cells"
            ],
            "working_resident_field_cells": primary_case["matched_compact_classical"][
                "working_resident_field_cells"
            ],
            "total_resident_field_cells_including_public_input": primary_case[
                "matched_compact_classical"
            ]["total_resident_field_cells_including_public_input"],
            "phase_relation_total_resident_field_cells_including_public_input": 6
            * su2.SIMPLE_OBJECTS,
            "phase_relation_working_field_cells_excluding_public_input": 5
            * su2.SIMPLE_OBJECTS,
            "classical_total_is_strictly_smaller": primary_case[
                "matched_compact_classical"
            ]["total_resident_field_cells_including_public_input"]
            < 6 * su2.SIMPLE_OBJECTS,
            "classical_working_is_strictly_smaller": primary_case[
                "matched_compact_classical"
            ]["working_resident_field_cells"]
            < 5 * su2.SIMPLE_OBJECTS,
            "boundary_agreement_all_cases": True,
            "resource_measurement_verification_level": "PACKAGE_SELF_REVIEW",
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "separate_reference": {
            "imports_m230_production": reference.get("imports_m230_production"),
            "imports_m211_production": reference.get("imports_m211_production"),
            "uses_independent_polynomial_quotient": reference.get(
                "uses_independent_polynomial_quotient"
            ),
            "case_control_restoration_reuse_parity": True,
        },
        "claim_limits": {
            "arbitrary_su2_relation_signatures": False,
            "growing_interface_width": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m211_production_sha256": sha256_file(
                here / "root_of_unity_su2_level8_fusion_phase_relation.py"
            ),
            "m230_production_sha256": sha256_file(Path(__file__).resolve()),
            "m230_reference_code_sha256": sha256_file(
                here
                / "su2_level8_open_fusion_signature_fanout_intersection_separate_reference.py"
            ),
            "m230_reference_result_sha256": sha256_file(reference_path),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
