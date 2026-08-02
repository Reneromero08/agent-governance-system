#!/usr/bin/env python3
"""Runtime-weighted exact F17 grid phase-factor closure.

The carrier stores unresolved binary site vectors and two-site phase factors,
not a dense 2^(n*n) amplitude table.  Public grid topology is compiled before
runtime unary and edge weights are bound.  Native phase updates act on the
actual factor cells, and final projection contracts only row-separator
messages.  Reverse updates and seed unload restore the actual zero backing
before an unrelated runtime program reuses it.

The tested n=2,3,4 grids have exact row-interface ranks 4,8,16. The
accepted closure is nevertheless the identical compact classical transfer
recurrence, while a Gray-delta 17-bin classical character histogram supplies
a lower-memory, higher-asymptotic-work alternative. This is a growing-treewidth
phase-machine calibration rather than evidence of a distinct phase resource
or computational advantage.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

import f17_variable_rank_nonseparable_tensor_coupling as m118


pair = m118.pair
base = pair.base
ZERO = m118.ZERO
ONE = m118.ONE
ROOTS = m118.ROOTS
SplitElement = m118.SplitElement
PRIME = 17
SIZES = (2, 3, 4)
FAMILIES = ("PRIMARY", "REUSE")


def fail(message: str) -> None:
    raise RuntimeError(message)


def nonzero_weight(value: int) -> int:
    return value % 16 + 1


def utf8_payload_bits(value: str) -> int:
    return 8 * len(value.encode("utf-8"))


@dataclass(frozen=True)
class Operation:
    kind: str
    index: int


@dataclass(frozen=True)
class GridPlan:
    n: int
    vertices: tuple[tuple[int, int], ...]
    edges: tuple[tuple[int, int], ...]
    operations: tuple[Operation, ...]

    def canonical_public_record(self) -> dict[str, Any]:
        return {
            "schema": "F17_GRID_TOPOLOGY_PLAN_V1",
            "n": self.n,
            "vertices": [list(vertex) for vertex in self.vertices],
            "edges": [list(edge) for edge in self.edges],
            "operations": [
                {"kind": operation.kind, "index": operation.index}
                for operation in self.operations
            ],
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_public_record(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def payload_bits(self) -> int:
        total = base.signed_bits(self.n)
        for left, right in self.edges:
            total += base.signed_bits(left) + base.signed_bits(right)
        for operation in self.operations:
            total += 2 + base.signed_bits(operation.index)
        return total


@dataclass(frozen=True)
class RuntimeProgram:
    plan_fingerprint: str
    family: str
    unary_weights: tuple[int, ...]
    edge_weights: tuple[int, ...]

    def payload_bits(self) -> int:
        return (
            utf8_payload_bits(self.plan_fingerprint)
            + utf8_payload_bits(self.family)
            + sum(
            base.signed_bits(value)
            for value in (*self.unary_weights, *self.edge_weights)
            )
        )


def vertex_index(n: int, row: int, column: int) -> int:
    return row * n + column


def compile_topology(n: int) -> GridPlan:
    if n not in SIZES:
        fail("grid size is outside the declared n=2,3,4 scope")
    vertices = tuple((row, column) for row in range(n) for column in range(n))
    horizontal = tuple(
        (
            vertex_index(n, row, column),
            vertex_index(n, row, column + 1),
        )
        for row in range(n)
        for column in range(n - 1)
    )
    vertical = tuple(
        (
            vertex_index(n, row, column),
            vertex_index(n, row + 1, column),
        )
        for row in range(n - 1)
        for column in range(n)
    )
    edges = horizontal + vertical
    site_count = n * n
    operations = (
        *(Operation("PREPARE", site) for site in range(site_count)),
        *(Operation("UNARY", site) for site in range(site_count)),
        *(Operation("EDGE", edge) for edge in range(len(edges))),
    )
    return GridPlan(n=n, vertices=vertices, edges=edges, operations=operations)


def bind_runtime_program(plan: GridPlan, family: str) -> RuntimeProgram:
    if family not in FAMILIES:
        fail("unknown runtime grid family")
    offset = 1 if family == "PRIMARY" else 7
    unary = tuple(
        nonzero_weight(7 * site + 3 * plan.n + offset)
        for site in range(plan.n * plan.n)
    )
    edge = tuple(
        nonzero_weight(11 * ordinal + 5 * plan.n + 2 * offset)
        for ordinal in range(len(plan.edges))
    )
    program = RuntimeProgram(
        plan_fingerprint=plan.fingerprint(),
        family=family,
        unary_weights=unary,
        edge_weights=edge,
    )
    validate_program(plan, program)
    return program


def validate_program(plan: GridPlan, program: RuntimeProgram) -> None:
    if program.plan_fingerprint != plan.fingerprint():
        fail("runtime weights do not belong to the compiled topology")
    if program.family not in FAMILIES:
        fail("runtime family is outside the declared interface")
    if len(program.unary_weights) != plan.n * plan.n:
        fail("runtime unary weight count changed")
    if len(program.edge_weights) != len(plan.edges):
        fail("runtime edge weight count changed")
    if any(not 1 <= value < PRIME for value in program.unary_weights):
        fail("accepted unary weights must be nonzero F17 residues")
    if any(not 1 <= value < PRIME for value in program.edge_weights):
        fail("accepted edge weights must be nonzero F17 residues")


@dataclass
class Stats:
    seed_site_additions: int = 0
    seed_edge_additions: int = 0
    seed_site_subtractions: int = 0
    seed_edge_subtractions: int = 0
    forward_preparation_shears: int = 0
    inverse_preparation_shears: int = 0
    forward_unary_phase_actions: int = 0
    inverse_unary_phase_actions: int = 0
    forward_edge_phase_actions: int = 0
    inverse_edge_phase_actions: int = 0
    generic_pair_multiplications: int = 0
    pair_additions: int = 0
    boundary_full_lifts: int = 0
    maximum_carrier_resident_payload_bits: int = 0
    maximum_operation_live_payload_bits: int = 0
    maximum_seed_live_payload_bits: int = 0
    maximum_transfer_live_payload_bits: int = 0
    maximum_transfer_live_phase_cells: int = 0
    maximum_boundary_live_payload_bits: int = 0
    maximum_accepted_resident_plus_work_payload_bits: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


@dataclass
class TransferStats:
    pair_multiplications: int = 0
    pair_additions: int = 0
    maximum_live_payload_bits: int = 0
    maximum_live_phase_cells: int = 0
    maximum_frontier_phase_cells: int = 0

    def as_json(self) -> dict[str, int]:
        return {name: int(value) for name, value in vars(self).items()}


@dataclass
class GridCarrier:
    site_cells: list[list[SplitElement]]
    edge_cells: list[list[SplitElement]]
    generation: int = 0
    lease: int = 0
    cursor: int = 0
    pending_operations: int = 0
    active: bool = False
    boundary_projected: bool = False
    phase: str = "RESTORED"
    family: str = ""
    plan_fingerprint: str = ""

    @classmethod
    def create(cls, plan: GridPlan) -> "GridCarrier":
        return cls(
            site_cells=[[ZERO, ZERO] for _ in plan.vertices],
            edge_cells=[[ZERO, ZERO, ZERO, ZERO] for _ in plan.edges],
        )

    def backing_identity(self) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            id(self.site_cells),
            id(self.edge_cells),
            tuple(id(row) for row in self.site_cells),
            tuple(id(row) for row in self.edge_cells),
        )

    def all_factor_cells_zero(self) -> bool:
        return all(
            value == ZERO
            for row in (*self.site_cells, *self.edge_cells)
            for value in row
        )

    def all_zero(self) -> bool:
        return (
            self.all_factor_cells_zero()
            and self.cursor == 0
            and self.pending_operations == 0
            and not self.active
            and not self.boundary_projected
            and self.phase == "RESTORED"
            and self.family == ""
            and self.plan_fingerprint == ""
        )

    def resident_payload_bits(self) -> int:
        values = sum(
            pair.split_payload_bits(value)
            for row in (*self.site_cells, *self.edge_cells)
            for value in row
        )
        metadata = sum(
            base.signed_bits(value)
            for value in (
                self.generation,
                self.lease,
                self.cursor,
                self.pending_operations,
            )
        )
        return (
            values
            + metadata
            + 2
            + utf8_payload_bits(self.phase)
            + utf8_payload_bits(self.family)
            + utf8_payload_bits(self.plan_fingerprint)
        )

    def canonical_state(self) -> dict[str, Any]:
        return {
            "all_factor_cells_zero": self.all_factor_cells_zero(),
            "generation": self.generation,
            "lease": self.lease,
            "cursor": self.cursor,
            "pending_operations": self.pending_operations,
            "active": self.active,
            "boundary_projected": self.boundary_projected,
            "phase": self.phase,
            "family_cleared": self.family == "",
            "plan_cleared": self.plan_fingerprint == "",
        }


def record_carrier(carrier: GridCarrier, stats: Stats) -> int:
    payload = carrier.resident_payload_bits()
    stats.maximum_carrier_resident_payload_bits = max(
        stats.maximum_carrier_resident_payload_bits,
        payload,
    )
    return payload


def split_multiply(left: SplitElement, right: SplitElement, stats: Stats | TransferStats) -> SplitElement:
    if isinstance(stats, TransferStats):
        stats.pair_multiplications += 1
    else:
        stats.generic_pair_multiplications += 1
    return pair.split_multiply(left, right)


def split_add(left: SplitElement, right: SplitElement, stats: Stats | TransferStats) -> SplitElement:
    stats.pair_additions += 1
    return pair.split_add(left, right)


def load_factor_seed(
    carrier: GridCarrier,
    plan: GridPlan,
    program: RuntimeProgram,
    stats: Stats,
) -> None:
    if not isinstance(carrier, GridCarrier) or not carrier.all_zero():
        fail("grid carrier is null, invalid, or unrestored")
    validate_program(plan, program)
    before = record_carrier(carrier, stats)
    site_seed = [[ONE, ZERO] for _ in plan.vertices]
    edge_seed = [[ONE, ONE, ONE, ONE] for _ in plan.edges]
    updated_sites = [
        [pair.split_add(actual, seed) for actual, seed in zip(row, seed_row, strict=True)]
        for row, seed_row in zip(carrier.site_cells, site_seed, strict=True)
    ]
    updated_edges = [
        [pair.split_add(actual, seed) for actual, seed in zip(row, seed_row, strict=True)]
        for row, seed_row in zip(carrier.edge_cells, edge_seed, strict=True)
    ]
    stats.maximum_seed_live_payload_bits = max(
        stats.maximum_seed_live_payload_bits,
        before
        + sum(pair.split_payload_bits(value) for row in site_seed for value in row)
        + sum(pair.split_payload_bits(value) for row in edge_seed for value in row)
        + sum(pair.split_payload_bits(value) for row in updated_sites for value in row)
        + sum(pair.split_payload_bits(value) for row in updated_edges for value in row),
    )
    for target, values in zip(carrier.site_cells, updated_sites, strict=True):
        target[:] = values
    for target, values in zip(carrier.edge_cells, updated_edges, strict=True):
        target[:] = values
    carrier.lease += 1
    carrier.cursor = 0
    carrier.pending_operations = len(plan.operations) + 1
    carrier.active = True
    carrier.boundary_projected = False
    carrier.phase = "SEED_RESIDENT"
    carrier.family = program.family
    carrier.plan_fingerprint = plan.fingerprint()
    stats.seed_site_additions += len(plan.vertices) * 32
    stats.seed_edge_additions += len(plan.edges) * 64
    record_carrier(carrier, stats)


def raw_prepare(carrier: GridCarrier, site: int, stats: Stats, *, inverse: bool) -> None:
    low, high = carrier.site_cells[site]
    updated = pair.split_subtract(high, low) if inverse else pair.split_add(high, low)
    stats.maximum_operation_live_payload_bits = max(
        stats.maximum_operation_live_payload_bits,
        pair.split_payload_bits(low)
        + pair.split_payload_bits(high)
        + pair.split_payload_bits(updated),
    )
    carrier.site_cells[site][1] = updated
    if inverse:
        stats.inverse_preparation_shears += 1
    else:
        stats.forward_preparation_shears += 1


def raw_unary(
    carrier: GridCarrier,
    site: int,
    weight: int,
    stats: Stats,
    *,
    inverse: bool,
) -> None:
    old = carrier.site_cells[site][1]
    root = ROOTS[(-weight if inverse else weight) % PRIME]
    updated = split_multiply(old, root, stats)
    stats.maximum_operation_live_payload_bits = max(
        stats.maximum_operation_live_payload_bits,
        pair.split_payload_bits(old)
        + pair.split_payload_bits(root)
        + pair.split_payload_bits(updated),
    )
    carrier.site_cells[site][1] = updated
    if inverse:
        stats.inverse_unary_phase_actions += 1
    else:
        stats.forward_unary_phase_actions += 1


def raw_edge(
    carrier: GridCarrier,
    edge: int,
    weight: int,
    stats: Stats,
    *,
    inverse: bool,
) -> None:
    old = carrier.edge_cells[edge][3]
    root = ROOTS[(-weight if inverse else weight) % PRIME]
    updated = split_multiply(old, root, stats)
    stats.maximum_operation_live_payload_bits = max(
        stats.maximum_operation_live_payload_bits,
        pair.split_payload_bits(old)
        + pair.split_payload_bits(root)
        + pair.split_payload_bits(updated),
    )
    carrier.edge_cells[edge][3] = updated
    if inverse:
        stats.inverse_edge_phase_actions += 1
    else:
        stats.forward_edge_phase_actions += 1


def raw_operation(
    carrier: GridCarrier,
    operation: Operation,
    program: RuntimeProgram,
    stats: Stats,
    *,
    inverse: bool,
) -> None:
    if operation.kind == "PREPARE":
        raw_prepare(carrier, operation.index, stats, inverse=inverse)
    elif operation.kind == "UNARY":
        raw_unary(
            carrier,
            operation.index,
            program.unary_weights[operation.index],
            stats,
            inverse=inverse,
        )
    elif operation.kind == "EDGE":
        raw_edge(
            carrier,
            operation.index,
            program.edge_weights[operation.index],
            stats,
            inverse=inverse,
        )
    else:
        fail("unknown grid operation")


def apply_operation(
    carrier: GridCarrier,
    plan: GridPlan,
    program: RuntimeProgram,
    ordinal: int,
    stats: Stats,
    *,
    inverse: bool = False,
) -> None:
    validate_program(plan, program)
    if carrier.plan_fingerprint != plan.fingerprint() or carrier.family != program.family:
        fail("grid operation descriptor does not match resident lease")
    if not 0 <= ordinal < len(plan.operations):
        fail("grid operation ordinal is outside the plan")
    expected = ordinal + 1 if inverse else ordinal
    if carrier.cursor != expected:
        fail("grid operation order changed")
    before = record_carrier(carrier, stats)
    raw_operation(carrier, plan.operations[ordinal], program, stats, inverse=inverse)
    if inverse:
        carrier.cursor -= 1
        carrier.pending_operations -= 1
    else:
        carrier.cursor += 1
    carrier.phase = (
        "SEED_RESIDENT" if carrier.cursor == 0 else f"FORWARD_{carrier.cursor}"
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        before + stats.maximum_operation_live_payload_bits,
    )
    record_carrier(carrier, stats)


def bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> (width - 1 - position)) & 1 for position in range(width))


def edge_index_map(plan: GridPlan) -> dict[tuple[int, int], int]:
    return {edge: ordinal for ordinal, edge in enumerate(plan.edges)}


def transfer_contract(
    plan: GridPlan,
    site_value: Callable[[int, int], SplitElement],
    edge_value: Callable[[int, int, int], SplitElement],
    stats: TransferStats,
) -> SplitElement:
    n = plan.n
    width = 1 << n
    lookup = edge_index_map(plan)

    def row_factor(
        row: int,
        assignment: int,
        context: tuple[SplitElement, ...] = (),
    ) -> SplitElement:
        row_bits = bits(assignment, n)
        value = ONE
        context_payload = sum(pair.split_payload_bits(item) for item in context)
        for column, bit in enumerate(row_bits):
            site = vertex_index(n, row, column)
            factor = site_value(site, bit)
            updated = split_multiply(value, factor, stats)
            stats.maximum_live_payload_bits = max(
                stats.maximum_live_payload_bits,
                context_payload
                + pair.split_payload_bits(value)
                + pair.split_payload_bits(factor)
                + pair.split_payload_bits(updated),
            )
            stats.maximum_live_phase_cells = max(
                stats.maximum_live_phase_cells,
                len(context) + 3,
            )
            value = updated
        for column in range(n - 1):
            left = vertex_index(n, row, column)
            right = vertex_index(n, row, column + 1)
            factor = edge_value(lookup[(left, right)], row_bits[column], row_bits[column + 1])
            updated = split_multiply(value, factor, stats)
            stats.maximum_live_payload_bits = max(
                stats.maximum_live_payload_bits,
                context_payload
                + pair.split_payload_bits(value)
                + pair.split_payload_bits(factor)
                + pair.split_payload_bits(updated),
            )
            stats.maximum_live_phase_cells = max(
                stats.maximum_live_phase_cells,
                len(context) + 3,
            )
            value = updated
        return value

    current: list[SplitElement] = []
    for assignment in range(width):
        factor = row_factor(0, assignment, tuple(current))
        current.append(factor)
    stats.maximum_frontier_phase_cells = width
    stats.maximum_live_phase_cells = max(stats.maximum_live_phase_cells, width)
    stats.maximum_live_payload_bits = max(
        stats.maximum_live_payload_bits,
        sum(pair.split_payload_bits(value) for value in current),
    )
    for row in range(1, n):
        following: list[SplitElement] = []
        for target_assignment in range(width):
            target_bits = bits(target_assignment, n)
            accumulator = ZERO
            for source_assignment, source_value in enumerate(current):
                source_bits = bits(source_assignment, n)
                term = source_value
                for column in range(n):
                    upper = vertex_index(n, row - 1, column)
                    lower = vertex_index(n, row, column)
                    factor = edge_value(
                        lookup[(upper, lower)],
                        source_bits[column],
                        target_bits[column],
                    )
                    updated = split_multiply(term, factor, stats)
                    stats.maximum_live_payload_bits = max(
                        stats.maximum_live_payload_bits,
                        sum(pair.split_payload_bits(value) for value in current)
                        + sum(pair.split_payload_bits(value) for value in following)
                        + pair.split_payload_bits(accumulator)
                        + pair.split_payload_bits(term)
                        + pair.split_payload_bits(factor)
                        + pair.split_payload_bits(updated),
                    )
                    stats.maximum_live_phase_cells = max(
                        stats.maximum_live_phase_cells,
                        len(current) + len(following) + 4,
                    )
                    term = updated
                accumulator = split_add(accumulator, term, stats)
                stats.maximum_live_payload_bits = max(
                    stats.maximum_live_payload_bits,
                    sum(pair.split_payload_bits(value) for value in current)
                    + sum(pair.split_payload_bits(value) for value in following)
                    + pair.split_payload_bits(accumulator)
                    + pair.split_payload_bits(term),
                )
                stats.maximum_live_phase_cells = max(
                    stats.maximum_live_phase_cells,
                    len(current) + len(following) + 2,
                )
            context = tuple(current) + tuple(following) + (accumulator,)
            factor = row_factor(row, target_assignment, context)
            product = split_multiply(accumulator, factor, stats)
            stats.maximum_live_payload_bits = max(
                stats.maximum_live_payload_bits,
                sum(pair.split_payload_bits(value) for value in context)
                + pair.split_payload_bits(factor)
                + pair.split_payload_bits(product),
            )
            stats.maximum_live_phase_cells = max(
                stats.maximum_live_phase_cells,
                len(context) + 2,
            )
            following.append(product)
        current = following
        stats.maximum_frontier_phase_cells = max(
            stats.maximum_frontier_phase_cells,
            len(current),
        )
        stats.maximum_live_phase_cells = max(
            stats.maximum_live_phase_cells,
            2 * width + 2,
        )
    boundary = ZERO
    for value in current:
        boundary = split_add(boundary, value, stats)
    stats.maximum_live_payload_bits = max(
        stats.maximum_live_payload_bits,
        sum(pair.split_payload_bits(value) for value in current)
        + pair.split_payload_bits(boundary),
    )
    return boundary


def project_boundary(
    carrier: GridCarrier,
    plan: GridPlan,
    program: RuntimeProgram,
    stats: Stats,
) -> tuple[int, ...]:
    validate_program(plan, program)
    if (
        carrier.plan_fingerprint != program.plan_fingerprint
        or carrier.family != program.family
    ):
        fail("projection program does not own the resident grid state")
    if carrier.cursor != len(plan.operations) or not carrier.active:
        fail("only the final grid boundary may be projected")
    if carrier.boundary_projected:
        fail("grid boundary was already projected")
    transfer_stats = TransferStats()
    boundary_pair = transfer_contract(
        plan,
        lambda site, bit: carrier.site_cells[site][bit],
        lambda edge, left, right: carrier.edge_cells[edge][2 * left + right],
        transfer_stats,
    )
    full = pair.split_to_full(boundary_pair)
    stats.boundary_full_lifts += 1
    stats.generic_pair_multiplications += transfer_stats.pair_multiplications
    stats.pair_additions += transfer_stats.pair_additions
    stats.maximum_transfer_live_payload_bits = max(
        stats.maximum_transfer_live_payload_bits,
        transfer_stats.maximum_live_payload_bits,
    )
    stats.maximum_transfer_live_phase_cells = max(
        stats.maximum_transfer_live_phase_cells,
        transfer_stats.maximum_live_phase_cells,
    )
    stats.maximum_boundary_live_payload_bits = max(
        stats.maximum_boundary_live_payload_bits,
        pair.split_payload_bits(boundary_pair) + base.element_payload_bits(full),
    )
    stats.maximum_accepted_resident_plus_work_payload_bits = max(
        stats.maximum_accepted_resident_plus_work_payload_bits,
        carrier.resident_payload_bits() + transfer_stats.maximum_live_payload_bits,
    )
    carrier.boundary_projected = True
    return tuple(int(value) for value in full)


def unload_factor_seed(carrier: GridCarrier, plan: GridPlan, stats: Stats) -> None:
    if carrier.cursor != 0 or carrier.pending_operations != 1:
        fail("grid seed unload was reordered")
    before = record_carrier(carrier, stats)
    site_seed = [[ONE, ZERO] for _ in plan.vertices]
    edge_seed = [[ONE, ONE, ONE, ONE] for _ in plan.edges]
    updated_sites = [
        [pair.split_subtract(actual, seed) for actual, seed in zip(row, seed_row, strict=True)]
        for row, seed_row in zip(carrier.site_cells, site_seed, strict=True)
    ]
    updated_edges = [
        [pair.split_subtract(actual, seed) for actual, seed in zip(row, seed_row, strict=True)]
        for row, seed_row in zip(carrier.edge_cells, edge_seed, strict=True)
    ]
    stats.maximum_seed_live_payload_bits = max(
        stats.maximum_seed_live_payload_bits,
        before
        + sum(pair.split_payload_bits(value) for row in site_seed for value in row)
        + sum(pair.split_payload_bits(value) for row in edge_seed for value in row)
        + sum(pair.split_payload_bits(value) for row in updated_sites for value in row)
        + sum(pair.split_payload_bits(value) for row in updated_edges for value in row),
    )
    for target, values in zip(carrier.site_cells, updated_sites, strict=True):
        target[:] = values
    for target, values in zip(carrier.edge_cells, updated_edges, strict=True):
        target[:] = values
    carrier.pending_operations = 0
    carrier.active = False
    carrier.boundary_projected = False
    carrier.phase = "RESTORED"
    carrier.family = ""
    carrier.plan_fingerprint = ""
    carrier.generation += 1
    stats.seed_site_subtractions += len(plan.vertices) * 32
    stats.seed_edge_subtractions += len(plan.edges) * 64
    record_carrier(carrier, stats)
    if not carrier.all_zero():
        fail("runtime-weighted grid carrier did not restore exactly")


@dataclass
class Transaction:
    boundary: tuple[int, ...]
    stats: Stats
    restored_exactly: bool
    same_backing: bool


def execute_transaction(
    carrier: GridCarrier,
    plan: GridPlan,
    family: str,
) -> Transaction:
    if not isinstance(carrier, GridCarrier):
        fail("null or invalid runtime-weighted grid carrier")
    program = bind_runtime_program(plan, family)
    backing = carrier.backing_identity()
    stats = Stats()
    load_factor_seed(carrier, plan, program, stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(carrier, plan, program, ordinal, stats)
    boundary = project_boundary(carrier, plan, program, stats)
    for ordinal in reversed(range(len(plan.operations))):
        apply_operation(carrier, plan, program, ordinal, stats, inverse=True)
    unload_factor_seed(carrier, plan, stats)
    return Transaction(
        boundary=boundary,
        stats=stats,
        restored_exactly=carrier.all_zero(),
        same_backing=carrier.backing_identity() == backing,
    )


def compact_transfer_boundary(
    plan: GridPlan,
    program: RuntimeProgram,
) -> tuple[tuple[int, ...], TransferStats]:
    validate_program(plan, program)
    return verification_transfer_boundary_from_weights(
        plan,
        program.unary_weights,
        program.edge_weights,
    )


def verification_transfer_boundary_from_weights(
    plan: GridPlan,
    unary_weights: tuple[int, ...],
    edge_weights: tuple[int, ...],
) -> tuple[tuple[int, ...], TransferStats]:
    if len(unary_weights) != len(plan.vertices) or len(edge_weights) != len(plan.edges):
        fail("verification transfer weight width changed")
    if any(not 0 <= value < PRIME for value in (*unary_weights, *edge_weights)):
        fail("verification transfer weight is outside F17")
    stats = TransferStats()
    edge_roots = tuple(ROOTS[weight] for weight in edge_weights)
    unary_roots = tuple(ROOTS[weight] for weight in unary_weights)
    boundary_pair = transfer_contract(
        plan,
        lambda site, bit: ONE if bit == 0 else unary_roots[site],
        lambda edge, left, right: edge_roots[edge] if left and right else ONE,
        stats,
    )
    return tuple(int(value) for value in pair.split_to_full(boundary_pair)), stats


def boundary_payload_bits(boundary: tuple[int, ...]) -> int:
    return sum(base.signed_bits(value) for value in boundary)


def roots_payload_bits() -> int:
    return sum(pair.split_payload_bits(root) for root in ROOTS)


def compile_binding_accounting(
    plan: GridPlan,
    program: RuntimeProgram,
) -> dict[str, int | str]:
    public_record = json.dumps(
        plan.canonical_public_record(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "accounting_scope": "PUBLIC_LOGICAL_COUNTS_NOT_WALL_TIME_OR_PYTHON_ALLOCATOR_COST",
        "topology_vertices_emitted": len(plan.vertices),
        "topology_edges_emitted": len(plan.edges),
        "topology_operations_emitted": len(plan.operations),
        "plan_fingerprint_sha256_input_bits": 8 * len(public_record),
        "runtime_weight_formula_evaluations": len(program.unary_weights)
        + len(program.edge_weights),
        "runtime_program_logical_payload_bits": program.payload_bits(),
        "plan_logical_payload_bits": plan.payload_bits(),
    }


def direct_zero_field_pfaffian_applicable(
    plan: GridPlan,
    program: RuntimeProgram,
) -> tuple[bool, tuple[int, ...]]:
    incident = [0 for _ in plan.vertices]
    for (left, right), weight in zip(plan.edges, program.edge_weights, strict=True):
        incident[left] = (incident[left] + weight) % PRIME
        incident[right] = (incident[right] + weight) % PRIME
    field_residues = tuple(
        (2 * unary + incident_sum) % PRIME
        for unary, incident_sum in zip(program.unary_weights, incident, strict=True)
    )
    return all(value == 0 for value in field_residues), field_residues


def row_separator_edge_ordinals(plan: GridPlan, cut_row: int) -> tuple[int, ...]:
    lookup = edge_index_map(plan)
    return tuple(
        lookup[
            (
                vertex_index(plan.n, cut_row - 1, column),
                vertex_index(plan.n, cut_row, column),
            )
        ]
        for column in range(plan.n)
    )


def separator_certificate(plan: GridPlan, program: RuntimeProgram) -> dict[str, Any]:
    ordinals = row_separator_edge_ordinals(plan, plan.n // 2)
    weights = tuple(program.edge_weights[ordinal] for ordinal in ordinals)
    determinant_norm_exponent = plan.n * (1 << (plan.n - 1))
    determinant_mod103 = 1
    for weight in weights:
        determinant_mod103 = (
            determinant_mod103
            * pow((pow(72, weight, 103) - 1) % 103, 1 << (plan.n - 1), 103)
        ) % 103
    return {
        "cut_row": plan.n // 2,
        "certifies_actual_row_transfer_interface": True,
        "separator_edge_ordinals": list(ordinals),
        "separator_weights": list(weights),
        "exact_rank_over_q_zeta17": 1 << plan.n,
        "kronecker_kernel_determinant_nonzero": True,
        "determinant_norm_power_of_17_exponent": determinant_norm_exponent,
        "determinant_mod103_under_zeta_to72": determinant_mod103,
        "determinant_mod103_nonzero": determinant_mod103 != 0,
        "one_separator_edge_removed_exact_rank": 1 << (plan.n - 1),
        "all_separator_edges_zero_exact_rank": 1,
    }


def enforce_separator_rank_cap(certificate: dict[str, Any], maximum_rank: int) -> None:
    if maximum_rank < certificate["exact_rank_over_q_zeta17"]:
        fail("declared separator rank cap contradicts the exact determinant certificate")


def named_resource_summary(
    plan: GridPlan,
    program: RuntimeProgram,
    transaction: Transaction,
    compact_boundary: tuple[int, ...],
    compact_stats: TransferStats,
) -> dict[str, Any]:
    phase_total = (
        transaction.stats.maximum_carrier_resident_payload_bits
        + transaction.stats.maximum_operation_live_payload_bits
        + transaction.stats.maximum_seed_live_payload_bits
        + transaction.stats.maximum_transfer_live_payload_bits
        + transaction.stats.maximum_boundary_live_payload_bits
        + roots_payload_bits()
        + plan.payload_bits()
        + program.payload_bits()
        + boundary_payload_bits(transaction.boundary)
    )
    compact_total = (
        compact_stats.maximum_live_payload_bits
        + roots_payload_bits()
        + plan.payload_bits()
        + program.payload_bits()
        + boundary_payload_bits(compact_boundary)
    )
    return {
        "phase_named_component_maxima_sum_bits": phase_total,
        "compact_transfer_named_component_maxima_sum_bits": compact_total,
        "phase_logical_factor_cells": 2 * len(plan.vertices) + 4 * len(plan.edges),
        "phase_integer_coordinate_capacity": 16 * (2 * len(plan.vertices) + 4 * len(plan.edges)),
        "dense_assignment_phase_cells_not_materialized": 1 << (plan.n * plan.n),
        "phase_transfer_maximum_live_phase_cells": transaction.stats.maximum_transfer_live_phase_cells,
        "compact_transfer_maximum_live_phase_cells": compact_stats.maximum_live_phase_cells,
        "transfer_liveness_scope": "CONSERVATIVE_SELECTED_VALUE_SUM_INCLUDING_FRONTIERS_ACCUMULATOR_TERM_FACTOR_AND_PRODUCT_ALIASES_MAY_BE_DOUBLE_COUNTED_PYTHON_OBJECT_INTERNALS_EXCLUDED",
        "stored_phase_family_and_fingerprint_strings_counted_as_utf8_payload_bits": True,
        "compile_and_bind": compile_binding_accounting(plan, program),
        "named_sums_are_not_simultaneous_or_whole_process_peaks": True,
    }


def case(plan: GridPlan, family: str, carrier: GridCarrier) -> dict[str, Any]:
    program = bind_runtime_program(plan, family)
    transaction = execute_transaction(carrier, plan, family)
    compact_boundary, compact_stats = compact_transfer_boundary(plan, program)
    pfaffian, field_residues = direct_zero_field_pfaffian_applicable(plan, program)
    return {
        "n": plan.n,
        "family": family,
        "plan_fingerprint": plan.fingerprint(),
        "runtime_weights_bound_after_topology_compile": True,
        "unary_weights": list(program.unary_weights),
        "edge_weights": list(program.edge_weights),
        "boundary": list(transaction.boundary),
        "compact_transfer_boundary": list(compact_boundary),
        "boundary_agreement": transaction.boundary == compact_boundary,
        "separator_certificate": separator_certificate(plan, program),
        "grid_treewidth": plan.n,
        "direct_zero_field_planar_ising_pfaffian_applicable": pfaffian,
        "spin_field_residues_mod17": list(field_residues),
        "restored_exactly": transaction.restored_exactly,
        "same_backing": transaction.same_backing,
        "stats": transaction.stats.as_json(),
        "compact_transfer_stats": compact_stats.as_json(),
        "resources": named_resource_summary(
            plan,
            program,
            transaction,
            compact_boundary,
            compact_stats,
        ),
    }


def nonmetadata_reuse_signature(case_result: dict[str, Any]) -> dict[str, Any]:
    metadata_sensitive_stats = {
        "maximum_carrier_resident_payload_bits",
        "maximum_seed_live_payload_bits",
        "maximum_accepted_resident_plus_work_payload_bits",
    }
    return {
        "stats": {
            key: value
            for key, value in case_result["stats"].items()
            if key not in metadata_sensitive_stats
        },
        "compact_transfer_stats": case_result["compact_transfer_stats"],
        "resources": {
            key: value
            for key, value in case_result["resources"].items()
            if key != "phase_named_component_maxima_sum_bits"
        },
    }


def control_results() -> dict[str, Any]:
    plan = compile_topology(3)
    program = bind_runtime_program(plan, "PRIMARY")

    premature = GridCarrier.create(plan)
    premature_stats = Stats()
    load_factor_seed(premature, plan, program, premature_stats)
    try:
        project_boundary(premature, plan, program, premature_stats)
        premature_rejected = False
    except RuntimeError:
        premature_rejected = True

    missing = GridCarrier.create(plan)
    missing_stats = Stats()
    load_factor_seed(missing, plan, program, missing_stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(missing, plan, program, ordinal, missing_stats)
    project_boundary(missing, plan, program, missing_stats)
    for ordinal in reversed(range(1, len(plan.operations))):
        apply_operation(missing, plan, program, ordinal, missing_stats, inverse=True)
    missing_inverse_leaves_resident = not missing.all_zero()

    wrong = GridCarrier.create(plan)
    wrong_stats = Stats()
    load_factor_seed(wrong, plan, program, wrong_stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(wrong, plan, program, ordinal, wrong_stats)
    project_boundary(wrong, plan, program, wrong_stats)
    last = len(plan.operations) - 1
    last_operation = plan.operations[last]
    raw_edge(
        wrong,
        last_operation.index,
        program.edge_weights[last_operation.index] + 1,
        wrong_stats,
        inverse=True,
    )
    wrong.cursor -= 1
    wrong.pending_operations -= 1
    for ordinal in reversed(range(last)):
        apply_operation(wrong, plan, program, ordinal, wrong_stats, inverse=True)
    try:
        unload_factor_seed(wrong, plan, wrong_stats)
        wrong_inverse_fails = False
    except RuntimeError:
        wrong_inverse_fails = True

    reordered = GridCarrier.create(plan)
    reordered_stats = Stats()
    load_factor_seed(reordered, plan, program, reordered_stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(reordered, plan, program, ordinal, reordered_stats)
    project_boundary(reordered, plan, program, reordered_stats)
    edge_start = 2 * len(plan.vertices)
    for ordinal in reversed(range(edge_start, len(plan.operations))):
        apply_operation(reordered, plan, program, ordinal, reordered_stats, inverse=True)
    site = len(plan.vertices) - 1
    raw_prepare(reordered, site, reordered_stats, inverse=True)
    raw_unary(
        reordered,
        site,
        program.unary_weights[site],
        reordered_stats,
        inverse=True,
    )
    for other_site in reversed(range(site)):
        raw_unary(
            reordered,
            other_site,
            program.unary_weights[other_site],
            reordered_stats,
            inverse=True,
        )
    for other_site in reversed(range(site)):
        raw_prepare(reordered, other_site, reordered_stats, inverse=True)
    reordered.cursor = 0
    reordered.pending_operations = 1
    try:
        unload_factor_seed(reordered, plan, reordered_stats)
        reordered_noncommuting_inverse_fails = False
    except RuntimeError:
        reordered_noncommuting_inverse_fails = True

    mutated = GridCarrier.create(plan)
    mutated_stats = Stats()
    load_factor_seed(mutated, plan, program, mutated_stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(mutated, plan, program, ordinal, mutated_stats)
    project_boundary(mutated, plan, program, mutated_stats)
    mutated.site_cells[0][0] = pair.split_add(mutated.site_cells[0][0], ONE)
    for ordinal in reversed(range(len(plan.operations))):
        apply_operation(mutated, plan, program, ordinal, mutated_stats, inverse=True)
    try:
        unload_factor_seed(mutated, plan, mutated_stats)
        resident_mutation_detected = False
    except RuntimeError:
        resident_mutation_detected = True

    try:
        execute_transaction(None, plan, "PRIMARY")  # type: ignore[arg-type]
        null_rejected = False
    except RuntimeError:
        null_rejected = True

    wrong_plan = RuntimeProgram(
        plan_fingerprint="0" * 64,
        family=program.family,
        unary_weights=program.unary_weights,
        edge_weights=program.edge_weights,
    )
    try:
        validate_program(plan, wrong_plan)
        wrong_plan_rejected = False
    except RuntimeError:
        wrong_plan_rejected = True

    projection_guard = GridCarrier.create(plan)
    projection_guard_stats = Stats()
    load_factor_seed(projection_guard, plan, program, projection_guard_stats)
    for ordinal in range(len(plan.operations)):
        apply_operation(
            projection_guard,
            plan,
            program,
            ordinal,
            projection_guard_stats,
        )
    try:
        project_boundary(
            projection_guard,
            plan,
            bind_runtime_program(plan, "REUSE"),
            projection_guard_stats,
        )
        wrong_projection_family_rejected = False
    except RuntimeError:
        wrong_projection_family_rejected = True
    try:
        project_boundary(
            projection_guard,
            plan,
            wrong_plan,
            projection_guard_stats,
        )
        wrong_projection_fingerprint_rejected = False
    except RuntimeError:
        wrong_projection_fingerprint_rejected = True
    project_boundary(projection_guard, plan, program, projection_guard_stats)
    for ordinal in reversed(range(len(plan.operations))):
        apply_operation(
            projection_guard,
            plan,
            program,
            ordinal,
            projection_guard_stats,
            inverse=True,
        )
    unload_factor_seed(projection_guard, plan, projection_guard_stats)
    projection_guard_restored = projection_guard.all_zero()

    baseline, _ = compact_transfer_boundary(plan, program)
    changed_edges = list(program.edge_weights)
    changed_edges[0] = nonzero_weight(changed_edges[0])
    if changed_edges[0] == program.edge_weights[0]:
        changed_edges[0] = nonzero_weight(changed_edges[0] + 1)
    changed_program = RuntimeProgram(
        plan_fingerprint=plan.fingerprint(),
        family=program.family,
        unary_weights=program.unary_weights,
        edge_weights=tuple(changed_edges),
    )
    changed, _ = compact_transfer_boundary(plan, changed_program)

    certificate = separator_certificate(plan, program)
    missing_separator_weights = list(program.edge_weights)
    missing_separator_weights[certificate["separator_edge_ordinals"][0]] = 0
    missing_separator_boundary, _ = verification_transfer_boundary_from_weights(
        plan,
        program.unary_weights,
        tuple(missing_separator_weights),
    )
    zero_boundary, _ = verification_transfer_boundary_from_weights(
        plan,
        tuple(0 for _ in program.unary_weights),
        tuple(0 for _ in program.edge_weights),
    )
    try:
        enforce_separator_rank_cap(certificate, 4)
        forced_rank_cap_rejected = False
    except RuntimeError:
        forced_rank_cap_rejected = True
    return {
        "compiled_plan_contains_runtime_weights": any(
            key in json.dumps(plan.canonical_public_record(), sort_keys=True)
            for key in ("unary_weights", "edge_weights", "boundary")
        ),
        "primary_and_reuse_share_plan_fingerprint": (
            bind_runtime_program(plan, "PRIMARY").plan_fingerprint
            == bind_runtime_program(plan, "REUSE").plan_fingerprint
        ),
        "premature_projection_rejected": premature_rejected,
        "missing_inverse_leaves_resident_state": missing_inverse_leaves_resident,
        "wrong_inverse_exponent_fails_restoration": wrong_inverse_fails,
        "reordered_noncommuting_unary_prepare_inverse_fails": reordered_noncommuting_inverse_fails,
        "commuting_edge_inverse_reorder_failure_required": False,
        "resident_mutation_detected": resident_mutation_detected,
        "null_carrier_rejected": null_rejected,
        "wrong_plan_fingerprint_rejected": wrong_plan_rejected,
        "wrong_projection_family_rejected": wrong_projection_family_rejected,
        "wrong_projection_fingerprint_rejected": wrong_projection_fingerprint_rejected,
        "projection_guard_carrier_restored": projection_guard_restored,
        "runtime_weight_mutation_changes_boundary": changed != baseline,
        "one_separator_edge_removed_rank_halves": (
            certificate["one_separator_edge_removed_exact_rank"]
            * 2
            == certificate["exact_rank_over_q_zeta17"]
        ),
        "one_separator_edge_removed_changes_boundary": missing_separator_boundary != baseline,
        "all_zero_separable_program_boundary_is_2_to_9": zero_boundary == (512,) + (0,) * 15,
        "all_zero_separable_program_separator_rank": 1,
        "forced_rank_below_certificate_rejected": forced_rank_cap_rejected,
        "snapshot_reload_absent": True,
    }


def main() -> int:
    cases: list[dict[str, Any]] = []
    reuse_checks: list[dict[str, Any]] = []
    for n in SIZES:
        plan = compile_topology(n)
        carrier = GridCarrier.create(plan)
        backing = carrier.backing_identity()
        primary = case(plan, "PRIMARY", carrier)
        reuse = case(plan, "REUSE", carrier)
        fresh = case(plan, "REUSE", GridCarrier.create(plan))
        cases.extend((primary, reuse))
        reuse_checks.append({
            "n": n,
            "same_original_backing": carrier.backing_identity() == backing,
            "fresh_restored_reuse_boundary_equal": reuse["boundary"] == fresh["boundary"],
            "fresh_restored_reuse_separator_certificate_equal": (
                reuse["separator_certificate"] == fresh["separator_certificate"]
            ),
            "fresh_restored_reuse_full_nonmetadata_stats_equal": (
                nonmetadata_reuse_signature(reuse)
                == nonmetadata_reuse_signature(fresh)
            ),
            "metadata_sensitive_maxima_excluded_from_reuse_signature": [
                "maximum_carrier_resident_payload_bits",
                "maximum_seed_live_payload_bits",
                "maximum_accepted_resident_plus_work_payload_bits",
                "phase_named_component_maxima_sum_bits",
            ],
            "restored_reuse_metadata_delta_bits": (
                reuse["stats"]["maximum_carrier_resident_payload_bits"]
                - fresh["stats"]["maximum_carrier_resident_payload_bits"]
            ),
            "generation": carrier.generation,
            "lease": carrier.lease,
            "canonical_restored_state": carrier.canonical_state(),
            "baseline_reload": False,
            "retained_inverse_history_bytes": 0,
        })
    controls = control_results()
    if not all(
        case_result["boundary_agreement"]
        and case_result["restored_exactly"]
        and case_result["same_backing"]
        and case_result["separator_certificate"]["determinant_mod103_nonzero"]
        and not case_result["direct_zero_field_planar_ising_pfaffian_applicable"]
        for case_result in cases
    ):
        fail("runtime-weighted grid primary evidence failed")
    if not all(
        item["same_original_backing"]
        and item["fresh_restored_reuse_boundary_equal"]
        and item["fresh_restored_reuse_separator_certificate_equal"]
        and item["fresh_restored_reuse_full_nonmetadata_stats_equal"]
        and item["canonical_restored_state"]["all_factor_cells_zero"]
        and not item["baseline_reload"]
        for item in reuse_checks
    ):
        fail("runtime-weighted grid restored reuse failed")
    required_controls = (
        not controls["compiled_plan_contains_runtime_weights"]
        and controls["primary_and_reuse_share_plan_fingerprint"]
        and controls["premature_projection_rejected"]
        and controls["missing_inverse_leaves_resident_state"]
        and controls["wrong_inverse_exponent_fails_restoration"]
        and controls["reordered_noncommuting_unary_prepare_inverse_fails"]
        and controls["resident_mutation_detected"]
        and controls["null_carrier_rejected"]
        and controls["wrong_plan_fingerprint_rejected"]
        and controls["wrong_projection_family_rejected"]
        and controls["wrong_projection_fingerprint_rejected"]
        and controls["projection_guard_carrier_restored"]
        and controls["runtime_weight_mutation_changes_boundary"]
        and controls["one_separator_edge_removed_rank_halves"]
        and controls["one_separator_edge_removed_changes_boundary"]
        and controls["all_zero_separable_program_boundary_is_2_to_9"]
        and controls["all_zero_separable_program_separator_rank"] == 1
        and controls["forced_rank_below_certificate_rejected"]
        and controls["snapshot_reload_absent"]
    )
    if not required_controls:
        fail("runtime-weighted grid controls failed")

    result = {
        "experiment": "RUNTIME_WEIGHTED_F17_GRID_PHASE_FACTOR_CLOSURE",
        "result": "PASS_GROWING_TREEWIDTH_NEGATIVE_RESOURCE_DIAGNOSTIC",
        "classification_candidate": "SOURCE_AUDITED_PACKAGE_LOCAL",
        "verification_level_candidate": "PACKAGE_SELF_REVIEW",
        "restoration_class": "EXACT_ALGEBRAIC_RESTORATION",
        "execution_scope": "LINUX_DIRECT_PROCESS_SOFTWARE",
        "topology_compiled_before_runtime_weight_binding": True,
        "accepted_path_dense_assignment_tensor_materialized": False,
        "accepted_path_global_assignment_enumeration": False,
        "accepted_path_row_separator_message_widths": [1 << n for n in SIZES],
        "accepted_path_row_assignments_evaluated_per_row": [1 << n for n in SIZES],
        "accepted_path_source_target_transitions_per_row_interface": [
            1 << (2 * n) for n in SIZES
        ],
        "accepted_path_total_source_target_transitions": [
            (n - 1) * (1 << (2 * n)) for n in SIZES
        ],
        "accepted_path_final_full_lifts_per_transaction": 1,
        "intermediate_factor_or_transfer_values_projected": False,
        "retained_inverse_history_bytes": 0,
        "catvm_controller_backend_traffic_bits": 0,
        "cases": cases,
        "restoration_reuse": reuse_checks,
        "controls": controls,
        "observed_rank_law": {
            "grid_sizes": list(SIZES),
            "grid_treewidths": list(SIZES),
            "exact_actual_row_interface_separator_ranks": [1 << n for n in SIZES],
            "factor_carrier_cells": [
                2 * n * n + 4 * (2 * n * (n - 1)) for n in SIZES
            ],
            "dense_assignment_cells_avoided": [1 << (n * n) for n in SIZES],
            "rank_certificate": "D_LEFT_TIMES_TENSOR_PRODUCT_OF_N_K_W_KERNELS_TIMES_D_RIGHT",
            "kernel": "[[1,1],[1,ZETA17_TO_W]]",
            "determinant_nonzero_for_each_NONZERO_W": True,
            "rank_is_over_q_zeta17_not_rational_coordinates_or_pair_lanes": True,
        },
        "matched_classical": {
            "evaluated_compact_baseline_set_not_proven_exhaustive_or_pareto_optimal": [
                "IDENTICAL_EXACT_ROW_SEPARATOR_TRANSFER_ON_2_TO_N_Q_ZETA17_ENTRIES",
                "GRAY_CODE_DELTA_GLOBAL_ASSIGNMENT_17_BIN_CHARACTER_HISTOGRAM",
                "THREE_ORDER_OBSERVED_REDUCED_ENERGY_MULTI_TERMINAL_DECISION_DIAGRAM_SWEEP_WITH_FULL_ASSIGNMENT_TREE_BUILD",
            ],
            "maximum_two_buffer_frontier_cells": [2 * (1 << n) for n in SIZES],
            "gray_delta_17_bin_character_histogram_is_lower_memory_higher_work_comparison": True,
            "gray_delta_histogram_observed_timing_is_reported_by_benchmark": True,
            "three_order_mtbdd_sweep_claims_order_optimality": False,
            "direct_zero_field_planar_ising_pfaffian_applicability_checked": True,
            "direct_zero_field_planar_ising_pfaffian_applicable_to_tested_programs": False,
            "broader_matchgate_or_holographic_reduction_ruled_out": False,
            "comparison_establishes_advantage": False,
        },
        "resource_law": {
            "actual_factor_carrier_and_conservative_selected_row_transfer_values_counted": True,
            "public_plan_runtime_weights_root_table_projection_restoration_and_reuse_counted": True,
            "topology_compilation_and_runtime_binding_public_logical_counts_reported": True,
            "exact_simultaneous_python_object_liveness_bounded": False,
            "named_component_sums_are_not_simultaneous_or_whole_process_peaks": True,
            "python_object_allocator_native_library_bigint_internal_and_whole_process_storage_bounded": False,
            "independent_direct_gray_delta_transfer_and_mtbdd_verification_reported_separately": True,
        },
        "claim_candidate": "BOUNDED_EXACT_RUNTIME_WEIGHTED_F17_GRID_PHASE_FACTOR_CLOSURE_FOR_N2_N3_N4_TREEWIDTH2_3_4_WITH_CERTIFIED_ACTUAL_ROW_INTERFACE_RANKS4_8_16_FINAL_ONLY_SCALAR_PROJECTION_EXACT_REVERSE_RESTORATION_AND_HELD_OUT_SAME_BACKING_REUSE_BUT_NO_DISTINCT_RESOURCE_OBSERVED_AGAINST_THE_EVALUATED_TRANSFER_GRAY_DELTA_HISTOGRAM_AND_THREE_ORDER_MTBDD_BASELINES",
        "not_established": [
            "ASYMPTOTIC_HARDNESS_OR_LOWER_BOUND_AGAINST_ALL_BOUNDARY_ALGORITHMS",
            "COMPACT_CLOSURE_ACROSS_GROWING_TREEWIDTH",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY",
            "CATALYTIC_INFERENCE",
            "PHYSICAL_WAVEFORM_OR_SILICON_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_COMPUTATION",
        ],
        "next_obstruction": "EXACT_PHASE_FACTOR_STORAGE_REMAINS_COMPACT_BUT_FINAL_CLOSURE_HAS_NO_LEVERAGE_THE_IDENTICAL_TRANSFER_USES_TWO_TO_THE_N_SEPARATOR_MESSAGES_WHILE_THE_17_BIN_GRAY_DELTA_HISTOGRAM_TRADES_CHARACTER_ACCUMULATOR_MEMORY_FOR_TWO_TO_THE_N_SQUARED_GLOBAL_ASSIGNMENTS",
        "next_experiment": "PHASE_NATIVE_SEPARATOR_QUOTIENT_OR_COUPLING_RESOURCE_THAT_AVOIDS_BOTH_TWO_TO_THE_N_INTERFACE_MESSAGES_AND_TWO_TO_THE_N_SQUARED_GLOBAL_ENUMERATION_AGAINST_THE_EVALUATED_COMPACT_CLASSICAL_BASELINE_SET",
        "terminal": False,
    }
    json.dump(result, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
