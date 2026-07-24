from __future__ import annotations

from pathlib import Path
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
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.rank_one_resolvent_audit import (  # noqa: E402
    audit_rank_one_resolvent_sensor,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unique_solution(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        tuple(
            clause(Literal(variable), Literal(variable), Literal(variable))
            for variable in variables
        ),
    )


def test_normalized_rank_one_probe_has_exponential_unique_residue() -> None:
    audit = audit_rank_one_resolvent_sensor(
        audit_clause_hamiltonian(unique_solution(8))
    )
    assert audit.basis_dimension == 256
    assert audit.zero_mode_degeneracy == 1
    assert audit.normalized_probe_norm_squared == 1.0
    assert audit.normalized_zero_pole_residue == 1 / 256
    assert audit.unique_witness_normalized_residue == 1 / 256
    assert not audit.polynomial_resolvent_access_established


def test_constant_unique_residue_moves_exponent_to_probe_norm() -> None:
    audit = audit_rank_one_resolvent_sensor(
        audit_clause_hamiltonian(unique_solution(10))
    )
    assert audit.unnormalized_zero_pole_residue == 1.0
    assert audit.unnormalized_probe_norm_squared == 1024
    assert audit.constant_residue_requires_probe_norm_squared == 1024
    assert not audit.polynomial_probe_energy_established


def test_unsat_has_no_zero_pole_residue() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("x", False), Literal("x", False), Literal("x", False)),
        ),
    )
    audit = audit_rank_one_resolvent_sensor(audit_clause_hamiltonian(holo))
    assert audit.zero_mode_degeneracy == 0
    assert audit.normalized_zero_pole_residue == 0.0
    assert audit.unnormalized_zero_pole_residue == 0.0
