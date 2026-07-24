from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.basin_access_audit import (  # noqa: E402
    audit_proven_basin_access,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_unique_solution_holo,
)


def test_unique_solution_proven_basin_fraction_is_exponential() -> None:
    holo = exact_three_unique_solution_holo(6)
    audit = audit_proven_basin_access(holo)

    assert audit.isolated_solution_coordinates == len(holo.variables)
    assert audit.gamma == 0.25
    assert audit.guaranteed_voltage_fraction == 0.25 ** len(holo.variables)
    assert audit.guaranteed_voltage_log2_fraction == -2.0 * len(holo.variables)
    assert audit.short_memory_full_volume_fraction == 0.0
    assert audit.solution_orthant_requires_unknown_signs
    assert "EXPONENTIALLY_SMALL" in audit.unique_solution_voltage_fraction_status
    assert audit.deterministic_public_seed_guarantee_status == (
        "NOT_ESTABLISHED_WITHOUT_KNOWING_A_SOLUTION_ORTHANT"
    )


def test_free_solution_coordinates_reduce_basin_exponent() -> None:
    holo = exact_three_unique_solution_holo(4)
    audit = audit_proven_basin_access(holo, isolated_solution_coordinates=2)

    assert audit.guaranteed_voltage_fraction == 0.25**2
    assert audit.guaranteed_voltage_log2_fraction == -4.0
    assert audit.unique_solution_voltage_fraction_status == (
        "DEPENDS_ON_ISOLATED_INDEX_SIZE"
    )
