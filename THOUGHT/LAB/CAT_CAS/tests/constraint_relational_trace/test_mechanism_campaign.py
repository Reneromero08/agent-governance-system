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

from constraint_relational_trace_v1.run_mechanism_campaign import (  # noqa: E402
    build_mechanism_record,
)


def test_mechanism_campaign_selects_polynomial_phase_flow_without_promoting_proof() -> None:
    record = build_mechanism_record(variable_count=8)
    assert record["status"] == (
        "MECHANISM_CAMPAIGN_PASS__POLYNOMIAL_PHASE_SELECTOR_THERMAL_LATCH_CANDIDATE__"
        "CET_NOT_ESTABLISHED"
    )
    assert all(record["gates"].values())
    decision = record["decision"]
    assert decision["strongest_candidate"] == (
        "POLYNOMIAL_S1_EXACT_PRODUCT_GRADIENT_CLAUSE_FLOW_WITH_"
        "THERMAL_ZERO_MODE_LATCH"
    )
    assert decision["mathematical_sat_margin"] == "CONSTANT_NORMALIZED_ZERO_MODE_POPULATION"
    assert decision["native_clean_port"] == "DIRECT_PUBLIC_CLAUSE_HAMILTONIAN"
    assert decision["deterministic_noiseless_gain"] == "NOT_REQUIRED_BY_THERMAL_CANDIDATE"
    assert decision["worst_case_native_preparation"] == (
        "UNIFORM_POLYNOMIAL_TRAJECTORY_BOUND_NOT_ESTABLISHED"
    )
    assert decision["deterministic_exact_boundary"] == (
        "SAT_WITNESS_BOUNDARY_REFERENCE__UNSAT_TOTALITY_NOT_ESTABLISHED"
    )
    assert decision["complete_environment_restoration"] == "SMOOTH_REGION_ONLY__GLOBAL_NOT_ESTABLISHED"
    assert decision["polynomial_total_resources"] == "NOT_ESTABLISHED"
    assert decision["p_equals_np"] == "NOT_PROVEN"
