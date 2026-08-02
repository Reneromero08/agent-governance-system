from __future__ import annotations

from pathlib import Path
import sys


CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
    / "phase6b6"
    / "live_small_wall"
    / "carrier_state_tomography"
    / "family10h_holonomy_integration_v1"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from i1_compatibility_audit import run_audit  # noqa: E402


def test_i1_read_only_compatibility_audit_passes() -> None:
    result = run_audit()
    assert result["passed"] is True
    assert result["decision"] == (
        "I1_READ_ONLY_COMPATIBILITY_AUDIT_COMPLETE__H0_PARTIAL__H1_THROUGH_H7_NOT_PASSED"
    )
    assert result["next_gate"] == "I2_RECEIVER_SIDE_GENERATOR_CANDIDATE_CATALOG"
    assert len(result["source_identity_verified"]) == 9
    assert set(result["h_gate_status"]) == {f"H{index}" for index in range(8)}
    assert all("PASSED" not in status for status in result["h_gate_status"].values())


def test_i1_keeps_physical_claims_unpromoted() -> None:
    result = run_audit()
    assert result["h_gate_status"]["H0"] == "PARTIAL_CALIBRATION_ONLY"
    assert result["h_gate_status"]["H4"] == (
        "ORDER_EFFECT_OBSERVED_BUT_CONNECTION_LAW_NOT_ESTABLISHED"
    )
    assert result["h_gate_status"]["H7"] == "DIAGNOSTIC_ADVERSARY_ONLY"
    assert result["claim_ceiling"] == [
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ]


def test_r_squared_is_not_misclassified_as_restoration() -> None:
    result = run_audit()
    evidence = result["retained_evidence_checks"]
    assert evidence["scalar_q_coordinate_confirmed"] is True
    assert evidence["r2_restoration_absent"] is True
