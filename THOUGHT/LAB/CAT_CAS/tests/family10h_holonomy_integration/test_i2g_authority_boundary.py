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

from i2g_authority_boundary import validate_authority_boundary  # noqa: E402


def test_i2g_authority_is_not_granted() -> None:
    result = validate_authority_boundary()
    assert result["passed"] is True
    assert result["authority_granted"] is False
    assert result["target_contact_performed"] is False
    assert result["write_attempt_count"] == 0
    assert result["scientific_measurement_count"] == 0


def test_i2g_scope_and_forbidden_actions_are_complete() -> None:
    result = validate_authority_boundary()
    assert result["read_only_scope_count"] == 10
    assert result["forbidden_action_count"] == 10
    assert result["inventory_output_field_count"] == 19


def test_i2g_blocks_after_github_only_work() -> None:
    result = validate_authority_boundary()
    assert result["decision"] == (
        "I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_CONTRACT_COMPLETE__"
        "AUTHORITY_NOT_GRANTED"
    )
    assert result["next_gate"] == "I2G_AUTHORITY_GRANT_AND_TARGET_ACCESS_REQUIRED"
    assert result["claim_ceiling"] == [
        "READ_ONLY_INVENTORY_AUTHORITY_CONTRACT_ONLY",
        "TARGET_CONTACT_NOT_AUTHORIZED",
        "TARGET_INVENTORY_NOT_ACQUIRED",
        "PHYSICAL_PACKAGE_NOT_FREEZE_READY",
        "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ]
