from __future__ import annotations

from pathlib import Path
import math
import sys

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.phase_energy_monotonicity_audit import (  # noqa: E402
    audit_phase_energy_monotonicity,
)


def test_clause_product_energy_can_increase_under_boundary_release() -> None:
    audit = audit_phase_energy_monotonicity()

    assert math.isclose(audit.clause_energy, 1.0 / 16.0)
    assert audit.directional_energy_derivative > 0.0
    assert audit.energy_increases
    assert audit.release_term_present
    assert audit.clause_energy_lyapunov_status == (
        "CLAUSE_PRODUCT_ENERGY_NOT_GLOBAL_LYAPUNOV__AUGMENTED_FUNCTIONAL_REQUIRED"
    )


def test_exact_gradient_without_release_descends_clause_energy() -> None:
    audit = audit_phase_energy_monotonicity(boundary_release_rate=1.0e-12)

    assert audit.directional_energy_derivative < 0.0
    assert not audit.energy_increases
    assert audit.clause_energy_lyapunov_status == (
        "CLAUSE_PRODUCT_ENERGY_COUNTEREXAMPLE_NOT_ESTABLISHED"
    )
