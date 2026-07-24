from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = "CONSTRAINT_HOLO_V1"
CLAIM_CEILING = "CONSTRAINT_RELATIONAL_TRACE_REFERENCE_ONLY__CET_NATIVE_OPERATOR_NOT_ESTABLISHED"

FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "answer",
        "candidate",
        "candidate_score",
        "expected_output",
        "hidden_witness",
        "orientation_label",
        "recovered_assignment",
        "selected_branch",
        "verify_pass",
        "winner",
    }
)


class ConstraintHoloError(ValueError):
    """Raised when a public relational object violates its contract."""


@dataclass(frozen=True, order=True)
class Literal:
    variable: str
    positive: bool = True

    def __post_init__(self) -> None:
        if not self.variable or not self.variable.replace("_", "").isalnum():
            raise ConstraintHoloError(f"invalid variable name: {self.variable!r}")

    def evaluate(self, assignment: Mapping[str, bool]) -> bool:
        if self.variable not in assignment:
            raise ConstraintHoloError(f"missing variable in assignment: {self.variable}")
        value = bool(assignment[self.variable])
        return value if self.positive else not value

    def renamed(self, mapping: Mapping[str, str]) -> "Literal":
        return Literal(mapping.get(self.variable, self.variable), self.positive)

    def token(self) -> str:
        return self.variable if self.positive else f"~{self.variable}"


@dataclass(frozen=True)
class ClauseRelation:
    """Constant-size local relation for one three-literal disjunction."""

    literals: tuple[Literal, Literal, Literal]

    def __post_init__(self) -> None:
        if len(self.literals) != 3:
            raise ConstraintHoloError("a ClauseRelation must contain exactly three literals")

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted({literal.variable for literal in self.literals}))

    def accepts(self, assignment: Mapping[str, bool]) -> bool:
        return any(literal.evaluate(assignment) for literal in self.literals)

    def allowed_rows(self) -> tuple[tuple[bool, ...], ...]:
        variables = self.variables
        rows: list[tuple[bool, ...]] = []
        for mask in range(1 << len(variables)):
            assignment = {
                variable: bool((mask >> index) & 1)
                for index, variable in enumerate(variables)
            }
            if self.accepts(assignment):
                rows.append(tuple(assignment[variable] for variable in variables))
        return tuple(rows)

    def canonical_token(self) -> tuple[str, str, str]:
        return tuple(sorted(literal.token() for literal in self.literals))  # type: ignore[return-value]

    def renamed(self, mapping: Mapping[str, str]) -> "ClauseRelation":
        return ClauseRelation(tuple(literal.renamed(mapping) for literal in self.literals))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ConstraintHolo:
    """Public open relational diagram. It contains no witness or answer field."""

    variables: tuple[str, ...]
    clauses: tuple[ClauseRelation, ...]
    schema_version: str = SCHEMA_VERSION
    claim_ceiling: str = CLAIM_CEILING

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConstraintHoloError("unsupported schema version")
        if self.claim_ceiling != CLAIM_CEILING:
            raise ConstraintHoloError("claim ceiling inflation is forbidden")
        if len(set(self.variables)) != len(self.variables):
            raise ConstraintHoloError("variables must be unique")
        if tuple(sorted(self.variables)) != self.variables:
            raise ConstraintHoloError("variables must be stored in sorted order")
        declared = set(self.variables)
        referenced = {literal.variable for clause in self.clauses for literal in clause.literals}
        if not referenced.issubset(declared):
            missing = sorted(referenced - declared)
            raise ConstraintHoloError(f"undeclared variables: {missing}")

    @classmethod
    def build(
        cls,
        variables: Iterable[str],
        clauses: Iterable[ClauseRelation],
    ) -> "ConstraintHolo":
        return cls(tuple(sorted(set(variables))), tuple(clauses))

    @classmethod
    def from_dimacs(cls, text: str) -> "ConstraintHolo":
        declared_count: int | None = None
        declared_clause_count: int | None = None
        raw_clause: list[int] = []
        clauses: list[ClauseRelation] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                fields = line.split()
                if len(fields) != 4 or fields[1] != "cnf":
                    raise ConstraintHoloError("expected 'p cnf <variables> <clauses>'")
                if declared_count is not None:
                    raise ConstraintHoloError("duplicate DIMACS problem line")
                declared_count = int(fields[2])
                declared_clause_count = int(fields[3])
                if declared_count < 0 or declared_clause_count < 0:
                    raise ConstraintHoloError("DIMACS counts must be nonnegative")
                continue
            for token in line.split():
                value = int(token)
                if value == 0:
                    if len(raw_clause) != 3:
                        raise ConstraintHoloError("only exact 3-CNF clauses are accepted")
                    literals = tuple(
                        Literal(f"x{abs(item)}", item > 0) for item in raw_clause
                    )
                    clauses.append(ClauseRelation(literals))  # type: ignore[arg-type]
                    raw_clause = []
                else:
                    raw_clause.append(value)

        if raw_clause:
            raise ConstraintHoloError("unterminated DIMACS clause")
        if declared_count is None or declared_clause_count is None:
            raise ConstraintHoloError("missing DIMACS problem line")
        if len(clauses) != declared_clause_count:
            raise ConstraintHoloError("parsed clause count does not match DIMACS declaration")
        if any(abs(int(literal.variable[1:])) > declared_count for clause in clauses for literal in clause.literals):
            raise ConstraintHoloError("literal index exceeds declared variable count")
        variables = (f"x{index}" for index in range(1, declared_count + 1))
        return cls.build(variables, clauses)

    def accepts(self, assignment: Mapping[str, bool]) -> bool:
        if set(assignment) != set(self.variables):
            raise ConstraintHoloError("assignment domain must equal the public boundary")
        return all(clause.accepts(assignment) for clause in self.clauses)

    def renamed(self, mapping: Mapping[str, str]) -> "ConstraintHolo":
        if set(mapping) != set(self.variables):
            raise ConstraintHoloError("a presentation gauge must rename every variable")
        renamed_values = tuple(mapping[variable] for variable in self.variables)
        if len(set(renamed_values)) != len(renamed_values):
            raise ConstraintHoloError("presentation gauge must be bijective")
        clauses = tuple(clause.renamed(mapping) for clause in self.clauses)
        return ConstraintHolo.build(renamed_values, clauses)

    def with_duplicate_clause(self, index: int) -> "ConstraintHolo":
        if index < 0 or index >= len(self.clauses):
            raise ConstraintHoloError("clause index out of range")
        return ConstraintHolo(self.variables, self.clauses + (self.clauses[index],))

    def semantic_clause_tokens(self) -> tuple[tuple[str, str, str], ...]:
        """Boolean idempotent normal form for order and duplicate controls."""

        return tuple(sorted(set(clause.canonical_token() for clause in self.clauses)))

    def public_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "claim_ceiling": self.claim_ceiling,
            "boundary_variables": list(self.variables),
            "local_relations": [
                {
                    "type": "three_literal_disjunction",
                    "literals": [literal.token() for literal in clause.literals],
                    "context": list(clause.variables),
                    "allowed_rows": [list(row) for row in clause.allowed_rows()],
                }
                for clause in self.clauses
            ],
            "equality_junctions": {
                variable: [
                    [clause_index, literal_index]
                    for clause_index, clause in enumerate(self.clauses)
                    for literal_index, literal in enumerate(clause.literals)
                    if literal.variable == variable
                ]
                for variable in self.variables
            },
            "collapse_boundary": "TOTALIZED_EXISTENTIAL_BOUNDARY_NOT_IMPLEMENTED",
            "native_operator": "CATALYTIC_EXISTENTIAL_TRACE_NOT_ESTABLISHED",
            "restoration_law": "NATIVE_REVERSIBLE_DILATION_NOT_ESTABLISHED",
        }
        audit_public_record(record)
        return record

    def presentation_digest(self) -> str:
        payload = json.dumps(self.public_record(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def semantic_digest(self) -> str:
        payload = json.dumps(
            {
                "variables": list(self.variables),
                "clauses": self.semantic_clause_tokens(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


def audit_public_record(value: object, path: str = "$public") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PUBLIC_FIELDS:
                raise ConstraintHoloError(f"forbidden answer-bearing field at {path}.{key}")
            audit_public_record(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            audit_public_record(child, f"{path}[{index}]")
