from __future__ import annotations

from pathlib import Path
import math
import sys

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.clause_hamiltonian import (  # noqa: E402
    audit_clause_hamiltonian,
    clause_violation_energy,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.zero_mode_amplifier_audit import (  # noqa: E402
    audit_ideal_zero_mode_amplifier,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unique_solution(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    clauses = tuple(
        clause(Literal(variable), Literal(variable), Literal(variable))
        for variable in variables
    )
    return ConstraintHolo.build(variables, clauses)


def test_clause_energy_is_number_of_violated_local_relations() -> None:
    holo = ConstraintHolo.build(
        ("x1", "x2"),
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x2", False), Literal("x2", False), Literal("x2", False)),
        ),
    )
    assert clause_violation_energy(holo, {"x1": True, "x2": False}) == 0
    assert clause_violation_energy(holo, {"x1": False, "x2": True}) == 2


def test_sat_has_zero_ground_energy_and_constant_next_energy_scale() -> None:
    result = audit_clause_hamiltonian(unique_solution(4))
    assert result.satisfiable
    assert result.ground_energy == 0
    assert result.ground_degeneracy == 1
    assert result.basis_dimension == 16
    assert result.normalized_zero_mode_weight == 1 / 16
    assert result.unsat_energy_margin == 1
    assert result.phase_evolution_inverse == "exp(-i t H_F)^-1 = exp(+i t H_F)"


def test_unsat_has_integer_ground_energy_at_least_one() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("x", False), Literal("x", False), Literal("x", False)),
        ),
    )
    result = audit_clause_hamiltonian(holo)
    assert not result.satisfiable
    assert result.ground_energy == 1
    assert result.ground_degeneracy == 0
    assert result.normalized_zero_mode_weight == 0.0
    assert result.unsat_energy_margin == 1


def test_unique_zero_mode_amplification_moves_cost_to_gain_or_modes() -> None:
    hamiltonian = audit_clause_hamiltonian(unique_solution(8))
    amplifier = audit_ideal_zero_mode_amplifier(hamiltonian)
    assert hamiltonian.normalized_zero_mode_weight == 1 / 256
    assert amplifier.required_weight_gain == 255
    assert math.isclose(amplifier.ideal_exponential_gain_time or 0.0, math.log(255))
    assert amplifier.ideal_exponential_gain_time is not None
    assert amplifier.ideal_exponential_gain_time < 8
    assert amplifier.minimum_output_to_input_energy_ratio == 255
    assert amplifier.mode_count_if_materialized == 256
    assert amplifier.polynomial_time_possible_under_ideal_gain
    assert not amplifier.polynomial_energy_established
    assert not amplifier.polynomial_mode_carrier_established


def test_unsat_has_no_zero_mode_for_amplifier_to_seed() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("x", False), Literal("x", False), Literal("x", False)),
        ),
    )
    amplifier = audit_ideal_zero_mode_amplifier(audit_clause_hamiltonian(holo))
    assert amplifier.initial_zero_mode_weight == 0.0
    assert amplifier.required_weight_gain is None
    assert amplifier.ideal_exponential_gain_time is None
