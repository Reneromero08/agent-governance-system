from __future__ import annotations

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
        normalized_constraints = tuple(constraints)
        declared = set(normalized_vertices)
        for constraint in normalized_constraints:
            if constraint.left not in declared or constraint.right not in declared:
                raise ParityHolonomyError("constraint references an undeclared vertex")
        return cls(normalized_vertices, normalized_constraints)

    def pairwise_locally_compatible(self) -> bool:
        for left_index in range(len(self.constraints)):
            if not _consistent(self.vertices, (self.constraints[left_index],)):
                return False
            for right_index in range(left_index + 1, len(self.constraints)):
                if not _consistent(
                    self.vertices,
                    (self.constraints[left_index], self.constraints[right_index]),
                ):
                    return False
        return True


class _ParityUnionFind:
    def __init__(self, vertices: tuple[str, ...]) -> None:
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}
        self.xor_to_parent = {vertex: 0 for vertex in vertices}

    def find(self, vertex: str) -> tuple[str, int]:
        parent = self.parent[vertex]
        if parent == vertex:
            return vertex, 0
        root, parent_xor = self.find(parent)
        total_xor = self.xor_to_parent[vertex] ^ parent_xor
        self.parent[vertex] = root
        self.xor_to_parent[vertex] = total_xor
        return root, total_xor

    def add(self, constraint: ParityConstraint) -> int | None:
        left_root, left_xor = self.find(constraint.left)
        right_root, right_xor = self.find(constraint.right)
        if left_root == right_root:
            return left_xor ^ right_xor ^ constraint.parity

        relation = left_xor ^ right_xor ^ constraint.parity
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
            self.xor_to_parent[left_root] = relation
        else:
            self.parent[right_root] = left_root
            self.xor_to_parent[right_root] = relation
            if self.rank[left_root] == self.rank[right_root]:
                self.rank[left_root] += 1
        return None


def _consistent(
    vertices: tuple[str, ...], constraints: tuple[ParityConstraint, ...]
) -> bool:
    union_find = _ParityUnionFind(vertices)
    return all(union_find.add(constraint) in (None, 0) for constraint in constraints)


@dataclass(frozen=True)
class ParityHolonomyResult:
    consistent: bool
    cycle_residues: tuple[int, ...]
    pairwise_locally_compatible: bool
    initial_carrier_digest: str
    terminal_carrier_digest: str
    restored_carrier_digest: str
    restored: bool
    restoration_scope: str
    claim_ceiling: str


class Z2PhaseCarrier:
    """Borrowed phase lanes with program-derived self-inverse operations."""

    def __init__(self, lane_count: int) -> None:
        if lane_count < 1:
            raise ParityHolonomyError("carrier must have at least one phase lane")
        self._phases = [1 for _ in range(lane_count)]

    @property
    def phases(self) -> tuple[int, ...]:
        return tuple(self._phases)

    def digest(self) -> str:
        payload = ",".join(str(phase) for phase in self._phases)
        return sha256(payload.encode("ascii")).hexdigest()

    def apply(self, lane: int, parity: int) -> None:
        if lane < 0 or lane >= len(self._phases):
            raise ParityHolonomyError("carrier lane out of range")
        if parity not in (0, 1):
            raise ParityHolonomyError("parity must be 0 or 1")
        if parity:
            self._phases[lane] *= -1

    def execute(self, program: tuple[ParityConstraint, ...]) -> None:
        for lane, constraint in enumerate(program):
            self.apply(lane, constraint.parity)

    def restore(self, program: tuple[ParityConstraint, ...]) -> None:
        for lane in range(len(program) - 1, -1, -1):
            self.apply(lane, program[lane].parity)


def calibrate_parity_holonomy(instance: ParityInstance) -> ParityHolonomyResult:
    union_find = _ParityUnionFind(instance.vertices)
    residues: list[int] = []
    for constraint in instance.constraints:
        residue = union_find.add(constraint)
        if residue is not None:
            residues.append(residue)

    carrier = Z2PhaseCarrier(max(1, len(instance.constraints)))
    initial_digest = carrier.digest()
    carrier.execute(instance.constraints)
    terminal_digest = carrier.digest()
    carrier.restore(instance.constraints)
    restored_digest = carrier.digest()

    return ParityHolonomyResult(
        consistent=all(residue == 0 for residue in residues),
        cycle_residues=tuple(residues),
        pairwise_locally_compatible=instance.pairwise_locally_compatible(),
        initial_carrier_digest=initial_digest,
        terminal_carrier_digest=terminal_digest,
        restored_carrier_digest=restored_digest,
        restored=initial_digest == restored_digest,
        restoration_scope="program_derived_inverse_on_borrowed_z2_phase_lanes",
        claim_ceiling="PARITY_HOLONOMY_CALIBRATION_ONLY",
    )
