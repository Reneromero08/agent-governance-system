from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Mapping

from .constraint_holo import ConstraintHolo, ConstraintHoloError

CLAIM_CEILING = "CONSTRAINT_RELATIONAL_TRACE_REFERENCE_ONLY__CET_NATIVE_OPERATOR_NOT_ESTABLISHED"
REFERENCE_VARIABLE_LIMIT = 20


@dataclass(frozen=True)
class ResourceLedger:
    public_variables: int
    public_clauses: int
    public_literal_occurrences: int
    symbolic_relation_size: int
    reference_basis_states: int
    accepted_basis_states: int
    explicit_provenance_states: int
    native_carrier_coordinates: int | None
    native_readout_cost: int | None
    native_restoration_cost: int | None
    unresolved_resources: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceBoundaryResult:
    valid: bool
    satisfiable: bool
    witness: Mapping[str, bool] | None
    witness_count: int
    ledger: ResourceLedger
    boundary_status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class ReversibleDilationAudit:
    oracle_description_size: int
    basis_states_audited: int
    accepted_basis_states: int
    all_basis_states_restored: bool
    output_flag_is_idempotent: bool
    provenance_retained: bool
    native_existential_trace_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class FactorizedProjectorCandidate:
    """Compact symbolic satisfaction projector with an unresolved nonzero boundary."""

    holo: ConstraintHolo

    @property
    def symbolic_size(self) -> int:
        return len(self.holo.variables) + 3 * len(self.holo.clauses)

    @property
    def native_operator_status(self) -> str:
        return "CATALYTIC_EXISTENTIAL_TRACE_NOT_ESTABLISHED"

    @property
    def unresolved_boundary(self) -> str:
        return "EXACT_NONZERO_NORMALIZATION_WITHOUT_PROVENANCE_EXPANSION"

    def apply_to_basis(self, assignment: Mapping[str, bool]) -> int:
        """Reference action of the factored clause projector on one basis state."""

        return int(self.holo.accepts(assignment))

    def public_contract(self) -> dict[str, object]:
        return {
            "operator_family": "factorized_commuting_clause_projectors",
            "symbolic_size": self.symbolic_size,
            "input_object": "open_solution_relation",
            "basis_action": "retain satisfying basis state; annihilate violating basis state",
            "native_operator_status": self.native_operator_status,
            "unresolved_boundary": self.unresolved_boundary,
            "claim_ceiling": CLAIM_CEILING,
        }


def iter_boundary_assignments(holo: ConstraintHolo) -> Iterator[dict[str, bool]]:
    for values in product((False, True), repeat=len(holo.variables)):
        yield dict(zip(holo.variables, values, strict=True))


def reference_existential_trace(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> ReferenceBoundaryResult:
    """Materialized correctness oracle for tiny objects only.

    This function is a boundary reference. It is not a native CAT_CAS operator and is
    deliberately capped so it cannot be narrated as the proof mechanism.
    """

    if variable_limit < 0 or variable_limit > REFERENCE_VARIABLE_LIMIT:
        raise ConstraintHoloError("invalid materialized reference limit")
    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError(
            "materialized reference refused: variable count exceeds the frozen limit"
        )

    witness: dict[str, bool] | None = None
    witness_count = 0
    basis_states = 0
    for assignment in iter_boundary_assignments(holo):
        basis_states += 1
        if holo.accepts(assignment):
            witness_count += 1
            if witness is None:
                witness = assignment.copy()

    ledger = ResourceLedger(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        public_literal_occurrences=3 * len(holo.clauses),
        symbolic_relation_size=len(holo.variables) + 3 * len(holo.clauses),
        reference_basis_states=basis_states,
        accepted_basis_states=witness_count,
        explicit_provenance_states=basis_states,
        native_carrier_coordinates=None,
        native_readout_cost=None,
        native_restoration_cost=None,
        unresolved_resources=(
            "native_relation_valued_transport",
            "exact_idempotent_existential_boundary",
            "provenance_compactness",
            "native_inverse_restoration",
            "polynomial_standard_model_transfer",
        ),
    )
    return ReferenceBoundaryResult(
        valid=True,
        satisfiable=witness_count > 0,
        witness=witness,
        witness_count=witness_count,
        ledger=ledger,
        boundary_status="REFERENCE_TOTALIZED_BOUNDARY_COMPLETE",
    )


def _reversible_oracle_step(
    holo: ConstraintHolo,
    assignment: Mapping[str, bool],
    output_flag: bool,
) -> bool:
    """Self-inverse oracle dilation for one preserved assignment register."""

    return output_flag ^ holo.accepts(assignment)


def audit_reversible_dilation(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> ReversibleDilationAudit:
    """Prove the small reference oracle is reversible while exposing its limitation.

    The assignment register is never merged. This establishes a lawful reversible
    dilation of evaluation, not an existential trace over the complete relation.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("reversible-dilation audit exceeds reference limit")

    restored = True
    accepted = 0
    audited = 0
    for assignment in iter_boundary_assignments(holo):
        audited += 1
        initial_flag = False
        forward_flag = _reversible_oracle_step(holo, assignment, initial_flag)
        if forward_flag:
            accepted += 1
        reverse_flag = _reversible_oracle_step(holo, assignment, forward_flag)
        restored = restored and reverse_flag == initial_flag

    return ReversibleDilationAudit(
        oracle_description_size=len(holo.variables) + 3 * len(holo.clauses),
        basis_states_audited=audited,
        accepted_basis_states=accepted,
        all_basis_states_restored=restored,
        output_flag_is_idempotent=True,
        provenance_retained=True,
        native_existential_trace_established=False,
        status="REVERSIBLE_EVALUATION_DILATION_ESTABLISHED__EXISTENTIAL_QUOTIENT_UNRESOLVED",
    )
