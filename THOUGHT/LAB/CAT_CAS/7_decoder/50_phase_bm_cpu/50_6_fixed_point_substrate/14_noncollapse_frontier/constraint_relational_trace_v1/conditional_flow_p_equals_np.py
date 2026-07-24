from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import (
    ReferenceClauseFlowRun,
    integrate_reference_until_solution,
)


@dataclass(frozen=True)
class PolynomialFlowDeadline:
    coefficient: int
    exponent: int

    def __post_init__(self) -> None:
        if self.coefficient < 1 or self.exponent < 1:
            raise ConstraintHoloError("polynomial flow deadline parameters must be positive")

    def steps_for_public_length(self, public_length: int) -> int:
        if public_length < 1:
            raise ConstraintHoloError("public length must be positive")
        return self.coefficient * public_length**self.exponent


@dataclass(frozen=True)
class ConditionalFlowTheorem:
    public_compiler_polynomial: bool
    vector_field_coordinates_polynomial: bool
    vector_field_evaluation_polynomial: bool
    witness_verification_polynomial: bool
    assumed_uniform_sat_deadline: str
    assumed_polynomial_precision_simulation: str
    timeout_totalizes_unsat: bool
    conditional_consequence: str
    convergence_theorem_status: str
    precision_theorem_status: str
    restoration_theorem_status: str
    p_equals_np_status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class ConditionalDeadlineRun:
    public_length: int
    step_deadline: int
    reference_run: ReferenceClauseFlowRun
    conditional_boundary: str
    boundary_is_unconditional: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def public_formula_length(holo: ConstraintHolo) -> int:
    return len(holo.variables) + 3 * len(holo.clauses)


def conditional_flow_theorem(holo: ConstraintHolo) -> ConditionalFlowTheorem:
    state_coordinates = len(holo.variables) + 2 * len(holo.clauses)
    local_couplings = 3 * len(holo.clauses)
    return ConditionalFlowTheorem(
        public_compiler_polynomial=True,
        vector_field_coordinates_polynomial=(
            state_coordinates <= 3 * max(1, public_formula_length(holo))
        ),
        vector_field_evaluation_polynomial=(
            local_couplings <= max(1, public_formula_length(holo))
        ),
        witness_verification_polynomial=True,
        assumed_uniform_sat_deadline=(
            "FOR_EVERY_SAT_F_THE_PUBLIC_SEED_FLOW_REACHES_A_VERIFIED_WITNESS_BY_T(|F|)"
        ),
        assumed_polynomial_precision_simulation=(
            "THE_FLOW_TO_T(|F|)_IS_DETERMINISTICALLY_SIMULABLE_WITH_POLYNOMIAL_BITS_AND_WORK"
        ),
        timeout_totalizes_unsat=True,
        conditional_consequence="3SAT_IN_P__THEREFORE_P_EQUALS_NP",
        convergence_theorem_status="NOT_ESTABLISHED",
        precision_theorem_status="NOT_ESTABLISHED",
        restoration_theorem_status="NOT_REQUIRED_FOR_COMPLEXITY_IMPLICATION__REQUIRED_FOR_CAT_CAS_CLOSURE",
        p_equals_np_status="CONDITIONAL_ONLY__NOT_PROVEN",
    )


def run_conditional_deadline_reference(
    holo: ConstraintHolo,
    deadline: PolynomialFlowDeadline,
    step_size: float = 2.0e-3,
) -> ConditionalDeadlineRun:
    """Run the candidate deadline without promoting timeout to proven UNSAT.

    A reached witness is unconditional because it is independently verified. A timeout
    is only conditionally UNSAT until the uniform convergence and simulation theorems
    are established.
    """

    public_length = public_formula_length(holo)
    step_deadline = deadline.steps_for_public_length(public_length)
    run = integrate_reference_until_solution(
        holo,
        step_size=step_size,
        max_steps=step_deadline,
    )
    if run.converged_to_public_solution:
        boundary = "VALID_SAT"
        unconditional = True
        status = "REFERENCE_FLOW_WITNESS_VERIFIED_BEFORE_DEADLINE"
    else:
        boundary = "CONDITIONAL_UNSAT_IF_UNIFORM_DEADLINE_THEOREM_HOLDS"
        unconditional = False
        status = "REFERENCE_FLOW_TIMEOUT__NO_UNSAT_PROMOTION"

    return ConditionalDeadlineRun(
        public_length=public_length,
        step_deadline=step_deadline,
        reference_run=run,
        conditional_boundary=boundary,
        boundary_is_unconditional=unconditional,
        status=status,
    )
