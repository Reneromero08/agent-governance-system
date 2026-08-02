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

from i2f_interface_spec import validate_i2f_spec  # noqa: E402


def test_i2f_binds_all_predecessor_contracts() -> None:
    result = validate_i2f_spec()
    assert result["passed"] is True
    assert len(result["source_identity_verified"]) == 4


def test_i2f_leaves_topology_and_layout_unassigned() -> None:
    result = validate_i2f_spec()
    assert result["topology"] == {
        "role_count": 5,
        "assigned_role_count": 0,
        "assignment_frozen": False,
    }
    assert result["carrier_and_source"] == {
        "carrier_layout_frozen": False,
        "source_isolation_implementation_frozen": False,
        "fork_only_secrecy_rejected": True,
    }


def test_i2f_backend_and_probe_are_spec_only() -> None:
    result = validate_i2f_spec()
    assert result["backend_specs"] == {
        "backend_method_count": 10,
        "directional_receipt_field_count": 17,
        "probe_channel_count": 7,
        "environment_field_count": 9,
    }


def test_i2f_requires_read_only_target_inventory_next() -> None:
    result = validate_i2f_spec()
    assert result["decision"] == {
        "freeze_blocker_count": 9,
        "physical_package_freeze_ready": False,
        "next_gate": "I2G_READ_ONLY_TARGET_INVENTORY_AUTHORITY_REQUIRED",
    }
    for token in (
        "TOPOLOGY_UNASSIGNED",
        "PHYSICAL_BACKEND_NOT_IMPLEMENTED",
        "NONDESTRUCTIVE_PROBE_NOT_IMPLEMENTED",
        "NUMERICAL_THRESHOLDS_NOT_FROZEN",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ):
        assert token in result["claim_ceiling"]
