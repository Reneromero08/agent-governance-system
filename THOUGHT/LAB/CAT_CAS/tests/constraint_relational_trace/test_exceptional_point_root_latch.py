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

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.exceptional_point_root_latch import (  # noqa: E402
    audit_exceptional_point_root_latch,
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


def unsat(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x1", False), Literal("x1", False), Literal("x1", False)),
        ),
    )


def test_order_n_ep_root_amplifies_unique_clean_amplitude_to_constant_radius() -> None:
    result = audit_exceptional_point_root_latch(unique_solution(8))
    assert result.assignment_dimension == 256
    assert result.satisfying_rank == 1
    assert result.clean_uniform_amplitude == 1 / 256
    assert result.clean_uniform_intensity == 1 / 65536
    assert result.exceptional_point_order == 8
    assert result.cycle_gain == 256
    assert result.effective_cycle_coupling == 1.0
    assert math.isclose(result.spectral_radius, 1.0)
    assert result.presence_margin == 1.0
    assert result.symbolic_sensor_modes == 8
    assert not result.polynomial_total_resources_established


def test_unsat_remains_at_nilpotent_exceptional_point() -> None:
    result = audit_exceptional_point_root_latch(unsat(8))
    assert result.satisfying_rank == 0
    assert result.clean_uniform_amplitude == 0.0
    assert result.effective_cycle_coupling == 0.0
    assert result.spectral_radius == 0.0
    assert result.presence_margin == 0.0


def test_multiple_witnesses_produce_nonzero_root_radius() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c", "d"),
        (clause(Literal("a"), Literal("a"), Literal("a")),),
    )
    result = audit_exceptional_point_root_latch(holo)
    assert result.satisfying_rank == 8
    assert result.effective_cycle_coupling == 8
    assert math.isclose(result.spectral_radius, 8 ** 0.25)
    assert result.spectral_radius >= 1.0


def test_constant_spectral_margin_does_not_erase_pre_gain_dynamic_range() -> None:
    result = audit_exceptional_point_root_latch(unique_solution(10))
    assert result.spectral_radius == 1.0
    assert result.unique_amplitude_dynamic_range == 1024
    assert result.unique_intensity_dynamic_range == 1024**2
    assert result.deterministic_noiseless_gain_status == "NOT_ESTABLISHED"
    assert "ENVIRONMENT" in result.reversible_dilation_status
    assert not result.polynomial_total_resources_established
