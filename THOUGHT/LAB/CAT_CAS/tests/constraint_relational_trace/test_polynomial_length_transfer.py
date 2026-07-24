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

from constraint_relational_trace_v1.polynomial_length_transfer import (  # noqa: E402
    audit_polynomial_length_transfer,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_parity_cycle_holo,
)


def test_polynomial_time_gives_polynomial_resource_bounds() -> None:
    holo = exact_three_parity_cycle_holo(8, total_charge=0)
    audit = audit_polynomial_length_transfer(holo, assumed_physical_time=64.0)

    assert audit.public_variables == 24
    assert audit.public_clauses == 32
    assert audit.maximum_variable_occurrences > 0
    assert audit.long_memory_upper_bound > 1.0
    assert audit.voltage_speed_upper_bound > 0.0
    assert audit.trajectory_length_upper_bound == (
        audit.assumed_physical_time * audit.euclidean_speed_upper_bound
    )
    assert audit.length_status == "POLYNOMIAL_IF_PHYSICAL_TIME_IS_POLYNOMIAL"
    assert audit.transfer_status == "CONDITIONAL__NOT_ESTABLISHED"


def test_length_audit_retains_piecewise_model_gap() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    audit = audit_polynomial_length_transfer(holo, assumed_physical_time=16.0)

    assert audit.continuous_model_status == "PIECEWISE_POLYNOMIAL_CARATHEODORY_FLOW"
    assert "MIN" in audit.selector_status
    assert "PATCHED_BOUNDARY" in audit.boundary_status
