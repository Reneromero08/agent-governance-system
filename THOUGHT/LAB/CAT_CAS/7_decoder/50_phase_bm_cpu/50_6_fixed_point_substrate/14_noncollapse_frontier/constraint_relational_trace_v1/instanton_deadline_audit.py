from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class InstantonDeadlineAudit:
    public_variables: int
    public_clauses: int
    phase_space_dimension: int
    clause_density: float
    maximum_index_descent_steps: int
    fixed_density_dimension_scaling: str
    solvable_instance_scope: str
    fixed_point_solution_correspondence_status: str
    public_seed_basin_status: str
    periodic_orbit_status: str
    chaos_status: str
    instanton_only_low_energy_status: str
    uniform_instanton_width_status: str
    uniform_critical_dwell_status: str
    uniform_tmax_constructive_bound_status: str
    conditional_continuous_time_bound: str
    numerical_discretization_transfer_status: str
    polynomial_precision_status: str
    unsat_deadline_status: str
    native_restoration_status: str
    ordinary_p_equals_np_status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_instanton_deadline_argument(holo: ConstraintHolo) -> InstantonDeadlineAudit:
    """Record the exact proof skeleton of the published continuous-time argument.

    The supplementary analysis for the memory-assisted 3-SAT flow argues:

    1. critical points are connected by instantons that decrease Morse index;
    2. the index cannot exceed the phase-space dimension n + 2m;
    3. at fixed clause density, the number of instanton transitions is O(n);
    4. each instanton width and critical-point dwell time is independent of instance
       size and controlled by the public memory rates;
    5. therefore continuous physical threshold-crossing time is O(n^alpha), alpha<=1.

    The publication explicitly limits this proposition to solvable fixed-density
    instances and says the result does not necessarily transfer to numerical
    integration because discretization breaks the topological supersymmetry.

    This audit does not reject the continuous-time theorem. It isolates what must be
    supplied before the theorem becomes an ordinary deterministic polynomial-time SAT
    algorithm or a complete CAT_CAS closure law.
    """

    variable_count = len(holo.variables)
    clause_count = len(holo.clauses)
    if variable_count < 1:
        raise ConstraintHoloError("instanton deadline audit requires a nonempty public boundary")
    phase_dimension = variable_count + 2 * clause_count
    density = clause_count / variable_count

    return InstantonDeadlineAudit(
        public_variables=variable_count,
        public_clauses=clause_count,
        phase_space_dimension=phase_dimension,
        clause_density=density,
        maximum_index_descent_steps=phase_dimension,
        fixed_density_dimension_scaling="N_PLUS_2M_IS_LINEAR_WHEN_M_OVER_N_IS_FIXED",
        solvable_instance_scope="PUBLISHED_PROPOSITION_APPLIES_TO_SOLVABLE_FIXED_DENSITY_3SAT",
        fixed_point_solution_correspondence_status=(
            "PUBLISHED_ANALYTIC_RESULT__FLOW_TERMINATES_ONLY_AT_SOLUTIONS"
        ),
        public_seed_basin_status=(
            "PUBLISHED_LARGE_HYPERCUBE_BASIN__UNIFORM_COVERAGE_OF_DECLARED_COMPILER_SEED_NOT_MACHINE_CHECKED"
        ),
        periodic_orbit_status="PUBLISHED_ABSENCE_IN_VOLTAGE_DYNAMICS",
        chaos_status="PUBLISHED_ABSENCE_UNDER_STATED_FLOW_HYPOTHESES",
        instanton_only_low_energy_status="PUBLISHED_TOPOLOGICAL_FIELD_THEORY_PREMISE",
        uniform_instanton_width_status=(
            "PUBLISHED_AS_INSTANCE_SIZE_INDEPENDENT_AND_RATE_CONTROLLED__"
            "NO_EXPLICIT_UNIFORM_NUMERIC_BOUND_COMPILED_FROM_PUBLIC_FORMULA"
        ),
        uniform_critical_dwell_status=(
            "PUBLISHED_AS_INSTANCE_SIZE_INDEPENDENT_AND_MEMORY_RATE_CONTROLLED__"
            "NO_EXPLICIT_UNIFORM_NUMERIC_BOUND_COMPILED_FROM_PUBLIC_FORMULA"
        ),
        uniform_tmax_constructive_bound_status=(
            "TMAX_IS_THE_MAXIMUM_WIDTH_PLUS_DWELL_OVER_THE_TRAJECTORY__"
            "POLYNOMIAL_BIT_BOUND_AND_EFFECTIVE_VALUE_NOT_ESTABLISHED_HERE"
        ),
        conditional_continuous_time_bound=(
            "T_PHYS_LE_N_TIMES_ONE_PLUS_TWO_DENSITY_TIMES_TMAX"
        ),
        numerical_discretization_transfer_status=(
            "PUBLICATION_EXPLICITLY_SAYS_CONTINUOUS_TIME_BOUND_NEED_NOT_APPLY_TO_NUMERICAL_INTEGRATION"
        ),
        polynomial_precision_status="NOT_ESTABLISHED",
        unsat_deadline_status=(
            "NOT_COVERED_BY_SOLVABLE_INSTANCE_PROPOSITION__"
            "UNSAT_TIMEOUT_REQUIRES_UNIFORM_SAT_DEADLINE_PLUS_EFFECTIVE_SIMULATION"
        ),
        native_restoration_status=(
            "DISSIPATIVE_PRIMAL_FLOW_NOT_RESTORATIVE__SMOOTH_COTANGENT_LIFT_RETAINS_UNRESOLVED_RANGE_AND_EVENT_COSTS"
        ),
        ordinary_p_equals_np_status="NOT_ESTABLISHED",
    )
