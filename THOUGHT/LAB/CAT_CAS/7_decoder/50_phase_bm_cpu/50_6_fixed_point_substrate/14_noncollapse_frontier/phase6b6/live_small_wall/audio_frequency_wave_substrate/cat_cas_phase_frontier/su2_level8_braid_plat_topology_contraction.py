#!/usr/bin/env python3
"""Exact public-topology contraction of bounded SU(2)_8 braid plat amplitudes.

This successor to the fusion-path MPS rank no-go contracts only the declared
vacuum-to-vacuum boundary.  Each public braid sweep becomes local sparse
factors over intermediate A9 path labels.  A min-fill order and all support
joins are compiled from topology and structural nonzero rules only; exact
Q(zeta_40) values are not inspected by the compiler.

The computational carrier is a one-cell exact additive accumulator.  Forward
execution rematerializes the topology contraction into that actual cell,
projects only the completed plat boundary, then recomputes the same public
contraction and applies the exact additive inverse.  The same backing is
reused for a different public program.  Factor tables and the public plan are
temporary material resources and are counted; this is direct-process
software, not CATVM custody or a phase/classical separation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

import su2_level8_fusion_path_braid_phase_relation as braid


Variable = tuple[int, int]
Assignment = tuple[int, ...]
STRANDS = (4, 6, 8, 10, 12, 14, 16)
FAMILIES = (0, 1)
ROUNDS = 8
PRIMARY_STRANDS = 16
PRIMARY_FAMILY = 0
REUSE_STRANDS = 12
REUSE_ROUNDS = 5
REUSE_FAMILY = 1
DEPTH_PROFILE_ROUNDS = (1, 2, 4, 6, 8, 10, 12, 14, 16)
M214_SOURCE_SHA256 = (
    "bc54076303fd12d57d9b57974832d451da603e123ea550f2312f81ad67bd9b11"
)


@dataclass(frozen=True)
class LeafSpec:
    round_index: int
    generator: int
    exponent: int
    reverse_sweep: bool
    scope: tuple[Variable, ...]
    support: frozenset[Assignment]


@dataclass(frozen=True)
class PlanOperation:
    kind: str
    left: int
    right: int
    output: int
    variable: Variable | None = None


@dataclass(frozen=True)
class ContractionPlan:
    leaves: tuple[LeafSpec, ...]
    operations: tuple[PlanOperation, ...]
    final_factor: int
    variable_order: tuple[Variable, ...]
    induced_width: int
    peak_support_factor_cells: int
    peak_live_support_cells: int
    peak_live_support_label_cells: int
    primal_nodes: int
    primal_edges: int


@dataclass
class ExactFactor:
    scope: tuple[Variable, ...]
    table: dict[Assignment, braid.K]


@dataclass
class Work:
    plan_compilations: int = 0
    public_leaf_descriptors: int = 0
    public_leaf_descriptor_integer_cells: int = 0
    public_plan_records: int = 0
    public_plan_integer_cells: int = 0
    support_join_pairs: int = 0
    support_projection_rows: int = 0
    peak_support_factor_cells: int = 0
    peak_live_support_cells: int = 0
    peak_live_support_label_cells: int = 0
    exact_leaf_rematerializations: int = 0
    exact_leaf_field_cells_generated: int = 0
    exact_join_field_cells_generated: int = 0
    exact_elimination_field_cells_generated: int = 0
    exact_join_assignment_pairs: int = 0
    exact_elimination_rows: int = 0
    field_additions: int = 0
    field_multiplications: int = 0
    peak_single_exact_factor_cells: int = 0
    peak_live_exact_factor_cells: int = 0
    peak_live_exact_factor_payload_bits: int = 0
    peak_exact_signed_numerator_bits: int = 0
    peak_exact_denominator_bits: int = 0
    contraction_calls: int = 0
    accumulator_additions: int = 0
    accumulator_subtractions: int = 0
    port_leases: int = 0
    port_releases: int = 0

    def merge(self, other: "Work") -> None:
        peak_fields = {
            "peak_support_factor_cells",
            "peak_live_support_cells",
            "peak_live_support_label_cells",
            "peak_single_exact_factor_cells",
            "peak_live_exact_factor_cells",
            "peak_live_exact_factor_payload_bits",
            "peak_exact_signed_numerator_bits",
            "peak_exact_denominator_bits",
        }
        for name in self.__dataclass_fields__:
            if name in peak_fields:
                setattr(self, name, max(getattr(self, name), getattr(other, name)))
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def path_domain(strands: int, position: int) -> tuple[int, ...]:
    maximum = min(position, strands - position, braid.LEVEL)
    return tuple(range(position % 2, maximum + 1, 2))


def fixed_label(strands: int, rounds: int, row: int, position: int) -> int | None:
    if position in (0, strands):
        return 0
    if row in (0, rounds):
        return position % 2
    return None


def gate_coordinates(
    round_index: int, generator: int, reverse_sweep: bool
) -> tuple[Variable, Variable, Variable, Variable]:
    if reverse_sweep:
        return (
            (round_index, generator - 1),
            (round_index, generator),
            (round_index + 1, generator + 1),
            (round_index + 1, generator),
        )
    return (
        (round_index + 1, generator - 1),
        (round_index, generator),
        (round_index, generator + 1),
        (round_index + 1, generator),
    )


def structural_nonzero(left: int, middle: int, right: int, output: int) -> bool:
    if not (
        abs(left - middle) == 1
        and abs(middle - right) == 1
        and abs(left - output) == 1
        and abs(output - right) == 1
    ):
        return False
    return left == right or output == middle


def assignment_labels(
    strands: int,
    rounds: int,
    coordinates: tuple[Variable, ...],
    scope: tuple[Variable, ...],
    assignment: Assignment,
) -> tuple[int, ...]:
    values = dict(zip(scope, assignment, strict=True))
    labels = []
    for row, position in coordinates:
        fixed = fixed_label(strands, rounds, row, position)
        labels.append(fixed if fixed is not None else values[(row, position)])
    return tuple(labels)


def compile_leaves(program: braid.BraidProgram) -> tuple[LeafSpec, ...]:
    leaves = []
    for step in range(program.steps):
        round_index, _offset = divmod(step, program.strands - 1)
        operation = program.operation(step)
        reverse = (round_index + program.family) % 2 == 1
        coordinates = gate_coordinates(round_index, operation.generator, reverse)
        scope = tuple(
            dict.fromkeys(
                coordinate
                for coordinate in coordinates
                if fixed_label(program.strands, program.rounds, *coordinate) is None
            )
        )
        support = set()
        domains = [path_domain(program.strands, position) for _row, position in scope]
        for assignment in product(*domains):
            labels = assignment_labels(
                program.strands, program.rounds, coordinates, scope, assignment
            )
            if structural_nonzero(*labels):
                support.add(assignment)
        leaves.append(
            LeafSpec(
                round_index,
                operation.generator,
                operation.exponent,
                reverse,
                scope,
                frozenset(support),
            )
        )
    return tuple(leaves)


def primal_order(
    leaves: tuple[LeafSpec, ...]
) -> tuple[tuple[Variable, ...], int, int, int]:
    graph: dict[Variable, set[Variable]] = defaultdict(set)
    for leaf in leaves:
        for variable in leaf.scope:
            graph[variable]
        for index, left in enumerate(leaf.scope):
            for right in leaf.scope[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    nodes = len(graph)
    edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    order = []
    induced_width = 0
    while graph:
        def key(variable: Variable) -> tuple[int, int, Variable]:
            neighbors = sorted(graph[variable])
            missing = sum(
                right not in graph[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1 :]
            )
            return missing, len(neighbors), variable

        variable = min(graph, key=key)
        neighbors = sorted(graph[variable])
        induced_width = max(induced_width, len(neighbors))
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
        for neighbor in neighbors:
            graph[neighbor].remove(variable)
        del graph[variable]
        order.append(variable)
    return tuple(order), induced_width, nodes, edges


def support_join(
    left_scope: tuple[Variable, ...],
    left_table: frozenset[Assignment],
    right_scope: tuple[Variable, ...],
    right_table: frozenset[Assignment],
    work: Work,
) -> tuple[tuple[Variable, ...], frozenset[Assignment]]:
    shared = tuple(variable for variable in left_scope if variable in right_scope)
    left_shared = tuple(left_scope.index(variable) for variable in shared)
    right_shared = tuple(right_scope.index(variable) for variable in shared)
    right_add = tuple(
        index for index, variable in enumerate(right_scope) if variable not in shared
    )
    index: dict[Assignment, list[Assignment]] = defaultdict(list)
    for assignment in right_table:
        index[tuple(assignment[position] for position in right_shared)].append(
            assignment
        )
    output = set()
    for left in left_table:
        key = tuple(left[position] for position in left_shared)
        for right in index.get(key, ()):
            output.add(left + tuple(right[position] for position in right_add))
            work.support_join_pairs += 1
    scope = left_scope + tuple(
        variable for variable in right_scope if variable not in shared
    )
    return scope, frozenset(output)


def support_eliminate(
    scope: tuple[Variable, ...],
    table: frozenset[Assignment],
    variable: Variable,
    work: Work,
) -> tuple[tuple[Variable, ...], frozenset[Assignment]]:
    position = scope.index(variable)
    work.support_projection_rows += len(table)
    return (
        scope[:position] + scope[position + 1 :],
        frozenset(
            assignment[:position] + assignment[position + 1 :]
            for assignment in table
        ),
    )


def observe_support(
    active: dict[int, tuple[tuple[Variable, ...], frozenset[Assignment]]],
    work: Work,
) -> None:
    work.peak_live_support_cells = max(
        work.peak_live_support_cells,
        sum(len(table) for _scope, table in active.values()),
    )
    work.peak_live_support_label_cells = max(
        work.peak_live_support_label_cells,
        sum(len(scope) * len(table) for scope, table in active.values()),
    )
    work.peak_support_factor_cells = max(
        work.peak_support_factor_cells,
        max((len(table) for _scope, table in active.values()), default=0),
    )


def compile_plan(program: braid.BraidProgram) -> tuple[ContractionPlan, Work]:
    work = Work(plan_compilations=1)
    leaves = compile_leaves(program)
    work.public_leaf_descriptors = len(leaves)
    work.public_leaf_descriptor_integer_cells = sum(
        4 + 2 * len(leaf.scope) for leaf in leaves
    )
    order, width, nodes, edges = primal_order(leaves)
    active = {index: (leaf.scope, leaf.support) for index, leaf in enumerate(leaves)}
    observe_support(active, work)
    operations = []
    next_id = len(leaves)
    for variable in order:
        bucket = sorted(
            (factor_id for factor_id, (scope, _table) in active.items() if variable in scope),
            key=lambda factor_id: (len(active[factor_id][1]), factor_id),
        )
        while len(bucket) > 1:
            left = bucket.pop(0)
            right = bucket.pop(0)
            output = next_id
            next_id += 1
            joined = support_join(*active[left], *active[right], work)
            del active[left]
            del active[right]
            active[output] = joined
            operations.append(PlanOperation("join", left, right, output))
            observe_support(active, work)
            bucket.append(output)
            bucket.sort(key=lambda factor_id: (len(active[factor_id][1]), factor_id))
        source = bucket.pop()
        output = next_id
        next_id += 1
        projected = support_eliminate(*active[source], variable, work)
        del active[source]
        active[output] = projected
        operations.append(PlanOperation("eliminate", source, -1, output, variable))
        observe_support(active, work)

    remaining = sorted(active, key=lambda factor_id: (len(active[factor_id][1]), factor_id))
    while len(remaining) > 1:
        left = remaining.pop(0)
        right = remaining.pop(0)
        output = next_id
        next_id += 1
        joined = support_join(*active[left], *active[right], work)
        del active[left]
        del active[right]
        active[output] = joined
        operations.append(PlanOperation("join", left, right, output))
        observe_support(active, work)
        remaining.append(output)
        remaining.sort(key=lambda factor_id: (len(active[factor_id][1]), factor_id))
    if len(remaining) != 1 or active[remaining[0]][0]:
        raise RuntimeError("topology contraction did not close to a scalar")
    work.public_plan_records = len(operations)
    work.public_plan_integer_cells = sum(
        4 + (2 if operation.variable is not None else 0)
        for operation in operations
    )
    return (
        ContractionPlan(
            leaves,
            tuple(operations),
            remaining[0],
            order,
            width,
            work.peak_support_factor_cells,
            work.peak_live_support_cells,
            work.peak_live_support_label_cells,
            nodes,
            edges,
        ),
        work,
    )


def local_weight(
    left: int, middle: int, right: int, output: int, exponent: int, work: Work
) -> braid.K:
    if not structural_nonzero(left, middle, right, output):
        return braid.ZERO
    alpha, beta = braid.local_braid_scalars(exponent)
    if left != right:
        return alpha
    temperley = braid.QUANTUM_DIMENSIONS[middle] * braid.INVERSE_DIMENSIONS[left]
    work.field_multiplications += 1
    value = beta * temperley
    work.field_multiplications += 1
    if output == middle:
        value = alpha + value
        work.field_additions += 1
    return value


def exact_leaf(program: braid.BraidProgram, leaf: LeafSpec, work: Work) -> ExactFactor:
    coordinates = gate_coordinates(leaf.round_index, leaf.generator, leaf.reverse_sweep)
    table = {}
    for assignment in leaf.support:
        labels = assignment_labels(
            program.strands, program.rounds, coordinates, leaf.scope, assignment
        )
        value = local_weight(*labels, leaf.exponent, work)
        if not value.is_zero():
            table[assignment] = value
    work.exact_leaf_rematerializations += 1
    work.exact_leaf_field_cells_generated += len(table)
    return ExactFactor(leaf.scope, table)


def exact_join(left: ExactFactor, right: ExactFactor, work: Work) -> ExactFactor:
    shared = tuple(variable for variable in left.scope if variable in right.scope)
    left_shared = tuple(left.scope.index(variable) for variable in shared)
    right_shared = tuple(right.scope.index(variable) for variable in shared)
    right_add = tuple(
        index for index, variable in enumerate(right.scope) if variable not in shared
    )
    index: dict[Assignment, list[tuple[Assignment, braid.K]]] = defaultdict(list)
    for assignment, value in right.table.items():
        index[tuple(assignment[position] for position in right_shared)].append(
            (assignment, value)
        )
    table: dict[Assignment, braid.K] = {}
    for left_assignment, left_value in left.table.items():
        key = tuple(left_assignment[position] for position in left_shared)
        for right_assignment, right_value in index.get(key, ()):
            output = left_assignment + tuple(
                right_assignment[position] for position in right_add
            )
            value = left_value * right_value
            work.field_multiplications += 1
            work.exact_join_assignment_pairs += 1
            if output in table:
                value = table[output] + value
                work.field_additions += 1
            if value.is_zero():
                table.pop(output, None)
            else:
                table[output] = value
    work.exact_join_field_cells_generated += len(table)
    return ExactFactor(
        left.scope + tuple(variable for variable in right.scope if variable not in shared),
        table,
    )


def exact_eliminate(factor: ExactFactor, variable: Variable, work: Work) -> ExactFactor:
    position = factor.scope.index(variable)
    table: dict[Assignment, braid.K] = {}
    for assignment, value in factor.table.items():
        output = assignment[:position] + assignment[position + 1 :]
        work.exact_elimination_rows += 1
        if output in table:
            value = table[output] + value
            work.field_additions += 1
        if value.is_zero():
            table.pop(output, None)
        else:
            table[output] = value
    work.exact_elimination_field_cells_generated += len(table)
    return ExactFactor(factor.scope[:position] + factor.scope[position + 1 :], table)


def observe_exact(
    active: dict[int, ExactFactor],
    work: Work,
    transient: tuple[ExactFactor, ...] = (),
) -> None:
    # Inputs and output coexist during a join/elimination even though consumed
    # derived factors are removed from ``active``.  Count that transient overlap
    # explicitly rather than reporting only the post-operation retained tables.
    factors = tuple(active.values()) + transient
    values = [value for factor in factors for value in factor.table.values()]
    work.peak_single_exact_factor_cells = max(
        work.peak_single_exact_factor_cells,
        max((len(factor.table) for factor in factors), default=0),
    )
    work.peak_live_exact_factor_cells = max(
        work.peak_live_exact_factor_cells, len(values)
    )
    if values:
        work.peak_live_exact_factor_payload_bits = max(
            work.peak_live_exact_factor_payload_bits, braid.field_payload_bits(values)
        )
        widths = braid.maximum_coordinate_bits(values)
        work.peak_exact_signed_numerator_bits = max(
            work.peak_exact_signed_numerator_bits,
            widths["maximum_signed_numerator_bits"],
        )
        work.peak_exact_denominator_bits = max(
            work.peak_exact_denominator_bits,
            widths["maximum_denominator_bits"],
        )


def contract(program: braid.BraidProgram) -> tuple[braid.K, Work, ContractionPlan]:
    plan, work = compile_plan(program)
    active: dict[int, ExactFactor] = {}

    def take(factor_id: int) -> ExactFactor:
        factor = active.pop(factor_id, None)
        if factor is not None:
            return factor
        if factor_id >= len(plan.leaves):
            raise RuntimeError("derived exact factor was not live")
        factor = exact_leaf(program, plan.leaves[factor_id], work)
        observe_exact(active, work, (factor,))
        return factor

    for operation in plan.operations:
        left = take(operation.left)
        if operation.kind == "join":
            right = take(operation.right)
            output = exact_join(left, right, work)
            observe_exact(active, work, (left, right, output))
        elif operation.kind == "eliminate" and operation.variable is not None:
            output = exact_eliminate(left, operation.variable, work)
            observe_exact(active, work, (left, output))
        else:
            raise RuntimeError("invalid public contraction operation")
        active[operation.output] = output
        observe_exact(active, work)
    if set(active) != {plan.final_factor}:
        raise RuntimeError("exact contraction left unresolved factors")
    final = active.pop(plan.final_factor)
    if final.scope or set(final.table) != {()}:
        raise RuntimeError("exact contraction did not produce one scalar")
    work.contraction_calls += 1
    return final.table[()], work, plan


def commitment(value: braid.K) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


@dataclass
class AccumulatorPort:
    cells: list[braid.K]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    program_commitment: str = ""

    def lease(self, owner: int, generation: int, program: braid.BraidProgram) -> None:
        if self.live:
            raise RuntimeError("plat accumulator already live")
        if len(self.cells) != 1 or self.cells[0] != braid.ZERO:
            raise ValueError("null or noncanonical plat accumulator")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.program_commitment = braid.program_commitment(program)

    def require(self, owner: int, program: braid.BraidProgram) -> None:
        if not self.live:
            raise RuntimeError("plat accumulator not live")
        if owner != self.owner:
            raise PermissionError("plat accumulator owner mismatch")
        if braid.program_commitment(program) != self.program_commitment:
            raise ValueError("plat accumulator public program mismatch")

    def forward(self, owner: int, program: braid.BraidProgram) -> Work:
        self.require(owner, program)
        if self.cursor:
            raise ValueError("plat accumulator forward cursor mismatch")
        value, work, _plan = contract(program)
        self.cells[0] = self.cells[0] + value
        work.field_additions += 1
        work.accumulator_additions += 1
        self.cursor = 1
        return work

    def project_final(self, owner: int, program: braid.BraidProgram) -> braid.K:
        self.require(owner, program)
        if self.cursor != 1:
            raise PermissionError("nonfinal plat projection rejected")
        return self.cells[0]

    def inverse(self, owner: int, program: braid.BraidProgram) -> Work:
        self.require(owner, program)
        if self.cursor != 1:
            raise ValueError("plat accumulator inverse cursor mismatch")
        value, work, _plan = contract(program)
        self.cells[0] = self.cells[0] - value
        work.field_additions += 1
        work.accumulator_subtractions += 1
        self.cursor = 0
        return work

    def release(self, owner: int, program: braid.BraidProgram) -> int:
        self.require(owner, program)
        if self.cursor or self.cells[0] != braid.ZERO:
            raise RuntimeError("plat accumulator released before exact inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.program_commitment = ""
        return generation


@dataclass
class Carrier:
    port: AccumulatorPort = field(default_factory=lambda: AccumulatorPort([braid.ZERO]))
    restoration_generation: int = 0


def transaction(carrier: Carrier, program: braid.BraidProgram) -> tuple[dict[str, object], Work]:
    backing = id(carrier.port.cells)
    generation = carrier.restoration_generation + 1
    owner = 216000 + generation
    carrier.port.lease(owner, generation, program)
    forward = carrier.port.forward(owner, program)
    forward.port_leases += 1
    boundary = carrier.port.project_final(owner, program)
    boundary_token = boundary.token()
    inverse = carrier.port.inverse(owner, program)
    released_generation = carrier.port.release(owner, program)
    inverse.port_releases += 1
    carrier.restoration_generation = released_generation
    work = Work()
    work.merge(forward)
    work.merge(inverse)
    canonical = (
        carrier.port.cells == [braid.ZERO]
        and not carrier.port.live
        and carrier.port.owner == 0
        and carrier.port.lease_generation == 0
        and carrier.port.cursor == 0
        and carrier.port.program_commitment == ""
        and carrier.restoration_generation == generation
    )
    return (
        {
            "boundary_commitment": hashlib.sha256(boundary_token.encode("ascii")).hexdigest(),
            "boundary_payload_bits": braid.field_payload_bits([boundary]),
            "same_accumulator_backing": id(carrier.port.cells) == backing,
            "canonical_post_restoration_state_exact": canonical,
            "restoration_error_field_cells": int(carrier.port.cells[0] != braid.ZERO),
            "baseline_reload_used": False,
        },
        work,
    )


def direct_boundary(program: braid.BraidProgram) -> tuple[braid.K, dict[str, int]]:
    topology, coefficients, work = braid.execute_forward(program)
    value = coefficients[topology.rank(braid.vacuum_path(program.strands))]
    return value, work.as_dict()


def exact_case(strands: int, family: int) -> dict[str, object]:
    program = braid.BraidProgram(strands, ROUNDS, family)
    value, work, plan = contract(program)
    direct, direct_work = direct_boundary(program)
    if value != direct:
        raise RuntimeError("topology contraction differs from direct fusion-path boundary")
    return {
        "strands": strands,
        "rounds": ROUNDS,
        "family": family,
        "program_steps": program.steps,
        "boundary_commitment": commitment(value),
        "boundary_payload_bits": braid.field_payload_bits([value]),
        "primal_nodes": plan.primal_nodes,
        "primal_edges": plan.primal_edges,
        "topology_induced_width": plan.induced_width,
        "peak_support_factor_cells": plan.peak_support_factor_cells,
        "peak_live_support_cells": plan.peak_live_support_cells,
        "peak_live_support_label_cells": plan.peak_live_support_label_cells,
        "exact_contraction_work": work.as_dict(),
        "direct_verification_fusion_path_cells": braid.FusionPathTopology.compile(strands).dimension,
        "direct_verification_work": direct_work,
        "exact_direct_boundary_agreement": True,
    }


def topology_depth_profile(strands: int) -> list[dict[str, int]]:
    result = []
    for rounds in DEPTH_PROFILE_ROUNDS:
        plan, work = compile_plan(braid.BraidProgram(strands, rounds, 0))
        result.append(
            {
                "strands": strands,
                "rounds": rounds,
                "program_steps": rounds * (strands - 1),
                "primal_nodes": plan.primal_nodes,
                "primal_edges": plan.primal_edges,
                "topology_induced_width": plan.induced_width,
                "peak_support_factor_cells": plan.peak_support_factor_cells,
                "peak_live_support_cells": plan.peak_live_support_cells,
                "peak_live_support_label_cells": plan.peak_live_support_label_cells,
                "public_plan_records": work.public_plan_records,
            }
        )
    return result


def controls() -> dict[str, object]:
    program = braid.BraidProgram(6, 2, 0)
    carrier = Carrier()
    premature = wrong_owner = wrong_program = missing = null_rejected = False
    owner = 216999
    carrier.port.lease(owner, 1, program)
    try:
        carrier.port.project_final(owner, program)
    except PermissionError:
        premature = True
    try:
        carrier.port.forward(owner + 1, program)
    except PermissionError:
        wrong_owner = True
    carrier.port.forward(owner, program)
    try:
        carrier.port.inverse(owner, braid.BraidProgram(6, 2, 1))
    except ValueError:
        wrong_program = True
    try:
        carrier.port.release(owner, program)
    except RuntimeError:
        missing = True
    carrier.port.inverse(owner, program)
    carrier.port.release(owner, program)
    try:
        bad = Carrier(AccumulatorPort([]))
        bad.port.lease(owner, 1, program)
    except ValueError:
        null_rejected = True
    return {
        "public_topology_compiler_reads_exact_coefficients_or_boundary": False,
        "premature_projection_rejected": premature,
        "wrong_owner_rejected": wrong_owner,
        "wrong_public_program_inverse_rejected": wrong_program,
        "missing_inverse_detected": missing,
        "null_carrier_rejected": null_rejected,
        "reordered_inverse_applicable": False,
        "snapshot_command_available": False,
    }


def main() -> None:
    cases = [exact_case(strands, family) for family in FAMILIES for strands in STRANDS]
    primary_case = next(
        case for case in cases if case["strands"] == PRIMARY_STRANDS and case["family"] == PRIMARY_FAMILY
    )
    primary_program = braid.BraidProgram(PRIMARY_STRANDS, ROUNDS, PRIMARY_FAMILY)
    primary_plan, _primary_compile_work = compile_plan(primary_program)
    carrier = Carrier()
    primary, primary_work = transaction(carrier, primary_program)
    reuse_program = braid.BraidProgram(REUSE_STRANDS, REUSE_ROUNDS, REUSE_FAMILY)
    reuse, reuse_work = transaction(carrier, reuse_program)
    fresh = Carrier()
    fresh_reuse, _fresh_work = transaction(fresh, reuse_program)
    depth_profile = topology_depth_profile(PRIMARY_STRANDS)
    result = {
        "schema": "cat_cas.su2_level8_braid_plat_topology_contraction.v1",
        "result": "PASS_EXACT_SU2_LEVEL8_BRAID_PLAT_PUBLIC_TOPOLOGY_CONTRACTION_WITH_DEPTH_WIDTH_OBSTRUCTION",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_SU2_LEVEL8_TEMPERLEY_LIEB_QZETA40_FIXED_EIGHT_SWEEP_FAMILIES0_1_EVEN_STRANDS4_6_8_10_12_14_16_VACUUM_PLAT_BOUNDARY_MIN_FILL_PUBLIC_TOPOLOGY_CONTRACTION_PRIMARY16_REUSE_STRANDS12_ROUNDS5_FAMILY1_DIRECT_PROCESS_ONLY",
        "phase_relation_law": {
            "domain": "SU2_LEVEL8_TEMPERLEY_LIEB_A9_VACUUM_BRAID_SPACETIME_FACTOR_GRAPH",
            "coefficient_field": "Q_ZETA40_DEGREE16",
            "local_gate_factor_arity": 4,
            "source_and_final_path": "PUBLIC_VACUUM_PAIRING_PATH",
            "compiler_inputs": "PUBLIC_STRANDS_ROUNDS_FAMILY_AND_STRUCTURAL_NONZERO_RULES_ONLY",
            "compiler_reads_exact_coefficients_or_boundary": False,
            "intermediate_fusion_rows_projected": False,
            "complete_fusion_path_vector_materialized_on_accepted_path": False,
            "one_cell_exact_accumulator": True,
            "inverse_rematerializes_contraction_from_public_topology": True,
        },
        "executed_cases": cases,
        "depth_profile": depth_profile,
        "transaction": {
            "primary": primary,
            "primary_full_lifecycle_work": primary_work.as_dict(),
            "reuse": reuse,
            "reuse_full_lifecycle_work": reuse_work.as_dict(),
            "fresh_reuse": fresh_reuse,
            "fresh_restored_reuse_boundary_agreement": reuse["boundary_commitment"] == fresh_reuse["boundary_commitment"],
            "restoration_generation_after_reuse": carrier.restoration_generation,
        },
        "controls": controls(),
        "resource_law": {
            "primary_accumulator_field_cells": 1,
            "primary_peak_single_exact_factor_cells": primary_case["exact_contraction_work"]["peak_single_exact_factor_cells"],
            "primary_peak_live_exact_factor_cells": primary_case["exact_contraction_work"]["peak_live_exact_factor_cells"],
            "primary_peak_live_exact_factor_payload_bits": primary_case["exact_contraction_work"]["peak_live_exact_factor_payload_bits"],
            "primary_retained_public_leaf_descriptors": len(primary_plan.leaves),
            "primary_retained_public_leaf_descriptor_integer_cells": primary_case["exact_contraction_work"]["public_leaf_descriptor_integer_cells"],
            "primary_retained_public_leaf_support_assignment_records": sum(
                len(leaf.support)
                for leaf in primary_plan.leaves
            ),
            "primary_retained_public_leaf_support_label_integer_cells": sum(
                len(leaf.scope) * len(leaf.support)
                for leaf in primary_plan.leaves
            ),
            "primary_peak_compiler_live_support_assignment_records": primary_case["peak_live_support_cells"],
            "primary_peak_compiler_live_support_label_integer_cells": primary_case["peak_live_support_label_cells"],
            "primary_public_plan_records": primary_case["exact_contraction_work"]["public_plan_records"],
            "primary_public_plan_integer_cells": primary_case["exact_contraction_work"]["public_plan_integer_cells"],
            "primary_direct_verification_fusion_path_cells": primary_case["direct_verification_fusion_path_cells"],
            "primary_direct_verification_fusion_path_payload_bits": primary_case["direct_verification_work"]["peak_carrier_payload_bits"],
            "primary_projection_field_cells": 1,
            "primary_projection_payload_bits": primary["boundary_payload_bits"],
            "primary_restoration_verification_field_cells": 1,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "accepted_retained_inverse_history": 0,
            "accepted_retained_complete_path_vectors": 0,
            "inverse_contraction_calls": 1,
            "matched_compact_baseline": "IDENTICAL_EXACT_SPARSE_FACTOR_ELIMINATION_AND_DIRECT_FUSION_PATH_RECURRENCES",
            "excluded_not_zero": "PYTHON_DICT_SET_TUPLE_HASH_STORAGE_ALLOCATOR_INTERPRETER_BYTECODE_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_EXACT_PUBLIC_TOPOLOGY_SPARSE_FACTOR_ELIMINATION",
            "direct_reference": "IDENTICAL_EXACT_FUSION_PATH_VECTOR_RECURRENCE",
            "phase_specific_contraction_reduction": False,
            "computational_advantage": False,
        },
        "source_dependencies": {"m214_production_sha256": M214_SOURCE_SHA256},
        "claim_limits": {
            "fixed_eight_sweep_strand_growth_boundary_contraction": True,
            "fixed_separator_across_growing_sweep_depth": False,
            "full_state_compaction": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
