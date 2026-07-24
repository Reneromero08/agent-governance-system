from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


class ParityHolonomyError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ParityConstraint:
    left: str
    right: str
    parity: int

    def __post_init__(self) -> None:
        if not self.left or not self.right or self.left == self.right:
            raise ParityHolonomyError("parity constraints require two distinct vertices")
        if self.parity not in (0, 1):
            raise ParityHolonomyError("parity must be 0 or 1")

    def canonicalized(self) -> "ParityConstraint":
        if self.left <= self.right:
            return self
        return ParityConstraint(self.right, self.left, self.parity)

    @property
    def transport(self) -> int:
        return -1 if self.parity else 1


@dataclass(frozen=True)
class ParityInstance:
    vertices: tuple[str, ...]
    constraints: tuple[ParityConstraint, ...]

    @classmethod
    def build(
        cls,
        vertices: Iterable[str],
        constraints: Iterable[ParityConstraint],
    ) -> "ParityInstance":
        normalized_vertices = tuple(sorted(set(vertices)))
        normalized_constraints = tuple(
            sorted(constraint.canonicalized() for constraint in constraints)
        )
        declared = set(normalized_vertices)
        for constraint in normalized_constraints:
            if constraint.left not in declared or constraint.right not in declared:
                raise ParityHolonomyError("constraint references an undeclared vertex")
        return cls(normalized_vertices, normalized_constraints)

    def pairwise_locally_compatible(self) -> bool:
        seen: dict[tuple[str, str], int] = {}
        for constraint in self.constraints:
            key = (constraint.left, constraint.right)
            previous = seen.get(key)
            if previous is not None and previous != constraint.parity:
                return False
            seen[key] = constraint.parity
        return True


@dataclass(frozen=True)
class TreeTransport:
    parent: str
    child: str
    constraint_index: int


@dataclass(frozen=True)
class Z2TransportProgram:
    roots: tuple[str, ...]
    tree_transports: tuple[TreeTransport, ...]
    cycle_constraint_indices: tuple[int, ...]


def compile_z2_transport(instance: ParityInstance) -> Z2TransportProgram:
    """Compile only graph incidence into a deterministic spanning forest.

    The compiler does not evaluate parity consistency. Constraint parity enters only
    when the borrowed phase carrier executes the transport program.
    """

    adjacency: dict[str, list[tuple[str, int]]] = {
        vertex: [] for vertex in instance.vertices
    }
    for index, constraint in enumerate(instance.constraints):
        adjacency[constraint.left].append((constraint.right, index))
        adjacency[constraint.right].append((constraint.left, index))
    for neighbors in adjacency.values():
        neighbors.sort()

    visited: set[str] = set()
    tree_indices: set[int] = set()
    roots: list[str] = []
    tree_transports: list[TreeTransport] = []

    for root in instance.vertices:
        if root in visited:
            continue
        roots.append(root)
        visited.add(root)
        queue: deque[str] = deque((root,))
        while queue:
            parent = queue.popleft()
            for child, constraint_index in adjacency[parent]:
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)
                tree_indices.add(constraint_index)
                tree_transports.append(
                    TreeTransport(parent, child, constraint_index)
                )

    cycle_indices = tuple(
        index for index in range(len(instance.constraints)) if index not in tree_indices
    )
    return Z2TransportProgram(
        roots=tuple(roots),
        tree_transports=tuple(tree_transports),
        cycle_constraint_indices=cycle_indices,
    )


@dataclass(frozen=True)
class ParityHolonomyResult:
    consistent: bool
    cycle_residues: tuple[int, ...]
    cycle_holonomies: tuple[int, ...]
    pairwise_locally_compatible: bool
    tree_transport_count: int
    cycle_count: int
    initial_carrier_digest: str
    terminal_carrier_digest: str
    restored_carrier_digest: str
    restored: bool
    restoration_scope: str
    obstruction_scope: str
    claim_ceiling: str


class Z2PhaseCarrier:
    """Vertex phase lanes carrying actual Z2 parallel transport."""

    def __init__(self, vertices: tuple[str, ...]) -> None:
        if not vertices:
            raise ParityHolonomyError("carrier must have at least one vertex phase lane")
        self._vertices = vertices
        self._phases = {vertex: 1 for vertex in vertices}

    @property
    def phases(self) -> tuple[tuple[str, int], ...]:
        return tuple((vertex, self._phases[vertex]) for vertex in self._vertices)

    def digest(self) -> str:
        payload = ",".join(
            f"{vertex}:{self._phases[vertex]}" for vertex in self._vertices
        )
        return sha256(payload.encode("ascii")).hexdigest()

    def transport(self, parent: str, child: str, edge_transport: int) -> None:
        if parent not in self._phases or child not in self._phases:
            raise ParityHolonomyError("transport references an unknown carrier lane")
        if edge_transport not in (-1, 1):
            raise ParityHolonomyError("Z2 edge transport must be -1 or +1")
        self._phases[child] *= self._phases[parent] * edge_transport

    def cycle_holonomy(self, constraint: ParityConstraint) -> int:
        return (
            self._phases[constraint.left]
            * constraint.transport
            * self._phases[constraint.right]
        )

    def execute(
        self,
        instance: ParityInstance,
        program: Z2TransportProgram,
    ) -> tuple[int, ...]:
        for operation in program.tree_transports:
            constraint = instance.constraints[operation.constraint_index]
            self.transport(operation.parent, operation.child, constraint.transport)
        return tuple(
            self.cycle_holonomy(instance.constraints[index])
            for index in program.cycle_constraint_indices
        )

    def restore(
        self,
        instance: ParityInstance,
        program: Z2TransportProgram,
    ) -> None:
        for operation in reversed(program.tree_transports):
            constraint = instance.constraints[operation.constraint_index]
            self.transport(operation.parent, operation.child, constraint.transport)


def calibrate_parity_holonomy(instance: ParityInstance) -> ParityHolonomyResult:
    program = compile_z2_transport(instance)
    carrier = Z2PhaseCarrier(instance.vertices)
    initial_digest = carrier.digest()
    cycle_holonomies = carrier.execute(instance, program)
    terminal_digest = carrier.digest()
    carrier.restore(instance, program)
    restored_digest = carrier.digest()
    residues = tuple(0 if holonomy == 1 else 1 for holonomy in cycle_holonomies)

    return ParityHolonomyResult(
        consistent=all(holonomy == 1 for holonomy in cycle_holonomies),
        cycle_residues=residues,
        cycle_holonomies=cycle_holonomies,
        pairwise_locally_compatible=instance.pairwise_locally_compatible(),
        tree_transport_count=len(program.tree_transports),
        cycle_count=len(program.cycle_constraint_indices),
        initial_carrier_digest=initial_digest,
        terminal_carrier_digest=terminal_digest,
        restored_carrier_digest=restored_digest,
        restored=initial_digest == restored_digest,
        restoration_scope="program_derived_inverse_of_executed_z2_tree_transport",
        obstruction_scope="native_cycle_product_on_borrowed_vertex_phase_lanes",
        claim_ceiling="PARITY_HOLONOMY_CALIBRATION_ONLY",
    )
