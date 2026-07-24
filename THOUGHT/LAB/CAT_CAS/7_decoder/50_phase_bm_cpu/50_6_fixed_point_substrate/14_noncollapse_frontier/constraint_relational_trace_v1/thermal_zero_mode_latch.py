from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    iter_boundary_assignments,
)
from .clause_hamiltonian import clause_violation_energy
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class ThermalZeroModeLatchAudit:
    public_variables: int
    public_clauses: int
    symbolic_local_terms: int
    physical_binary_coordinates: int
    basis_dimension_reference_only: int
    inverse_temperature: float
    inverse_temperature_scaling: str
    exact_partition_function_reference_only: float
    exact_zero_energy_population_reference_only: float
    guaranteed_sat_zero_population_lower_bound: float
    unsat_zero_population: float | None
    one_sided_sample_count_for_99_percent_detection: int
    constant_population_margin_established: bool
    normalized_state_avoids_unnormalized_partition_readout: bool
    public_clause_compilation_only: bool
    gibbs_preparation_status: str
    worst_case_mixing_status: str
    deterministic_boundary_status: str
    complete_bath_restoration_status: str
    polynomial_total_resources_established: bool
    standard_model_transfer_status: str
    claim_ceiling: str = CLAIM_CEILING


def _sample_count_for_detection(population_lower_bound: float, miss_probability: float) -> int:
    if not 0 < population_lower_bound <= 1:
        raise ConstraintHoloError("population lower bound must lie in (0, 1]")
    if not 0 < miss_probability < 1:
        raise ConstraintHoloError("miss probability must lie in (0, 1)")
    if population_lower_bound == 1:
        return 1
    count = 1
    while (1.0 - population_lower_bound) ** count > miss_probability:
        count += 1
    return count


def audit_thermal_zero_mode_latch(
    holo: ConstraintHolo,
    safety_bits: int = 2,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> ThermalZeroModeLatchAudit:
    """Audit a clause-local Gibbs zero-mode presence latch.

    For H_F equal to the sum of clause-violation projectors, choose

        beta = (n + safety_bits) ln 2.

    If F is satisfiable, every excited state has energy at least one and the total
    excited-state Boltzmann weight is at most 2^n exp(-beta). With safety_bits=2, the
    zero-energy population is therefore at least 4/5, even for a unique witness. If F
    is unsatisfiable, the zero-energy population is exactly zero.

    The result is a constant-margin relational boundary conditional on preparing the
    normalized Gibbs state. It does not establish polynomial Gibbs preparation,
    deterministic readout, or reversible bath closure.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("thermal zero-mode audit exceeds reference limit")
    if safety_bits < 1:
        raise ConstraintHoloError("thermal safety bits must be positive")

    variable_count = len(holo.variables)
    dimension = 1 << variable_count
    inverse_temperature = (variable_count + safety_bits) * log(2.0)
    energies = tuple(
        clause_violation_energy(holo, assignment)
        for assignment in iter_boundary_assignments(holo)
    )
    weights = tuple(exp(-inverse_temperature * energy) for energy in energies)
    partition_function = sum(weights)
    zero_weight = sum(weight for energy, weight in zip(energies, weights, strict=True) if energy == 0)
    zero_population = zero_weight / partition_function
    satisfiable = any(energy == 0 for energy in energies)

    excited_weight_bound = 2.0 ** (-safety_bits)
    guaranteed_lower_bound = 1.0 / (1.0 + excited_weight_bound)
    if not satisfiable:
        guaranteed_lower_bound = 0.0

    return ThermalZeroModeLatchAudit(
        public_variables=variable_count,
        public_clauses=len(holo.clauses),
        symbolic_local_terms=len(holo.clauses),
        physical_binary_coordinates=variable_count,
        basis_dimension_reference_only=dimension,
        inverse_temperature=inverse_temperature,
        inverse_temperature_scaling="LINEAR_IN_PUBLIC_VARIABLE_COUNT",
        exact_partition_function_reference_only=partition_function,
        exact_zero_energy_population_reference_only=zero_population,
        guaranteed_sat_zero_population_lower_bound=guaranteed_lower_bound,
        unsat_zero_population=(0.0 if not satisfiable else None),
        one_sided_sample_count_for_99_percent_detection=(
            _sample_count_for_detection(guaranteed_lower_bound, 0.01)
            if satisfiable
            else 0
        ),
        constant_population_margin_established=(
            (zero_population >= guaranteed_lower_bound >= 0.8)
            if satisfiable
            else zero_population == 0.0
        ),
        normalized_state_avoids_unnormalized_partition_readout=True,
        public_clause_compilation_only=True,
        gibbs_preparation_status="LOW_TEMPERATURE_GIBBS_STATE_PREPARATION_NOT_ESTABLISHED",
        worst_case_mixing_status="POLYNOMIAL_WORST_CASE_MIXING_NOT_ESTABLISHED",
        deterministic_boundary_status=(
            "ONE_SIDED_CONSTANT_PROBABILITY_BOUNDARY_ONLY__"
            "DETERMINISTIC_EXACT_PRESENCE_READOUT_NOT_ESTABLISHED"
        ),
        complete_bath_restoration_status="SYSTEM_PLUS_BATH_NATIVE_INVERSE_NOT_ESTABLISHED",
        polynomial_total_resources_established=False,
        standard_model_transfer_status="NOT_ESTABLISHED",
    )
