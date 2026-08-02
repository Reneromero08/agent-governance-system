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

from i2d_gap_contract import validate_gap_contract  # noqa: E402


def test_i2d_binds_all_predecessor_gates() -> None:
    result = validate_gap_contract()
    assert result["passed"] is True
    assert len(result["source_identity_verified"]) == 5
    assert all(result["inherited_status_checks"].values())


def test_i2d_requires_new_multivariate_observables() -> None:
    result = validate_gap_contract()
    observable = result["observable_contract"]
    assert observable["required_observable_count"] == 7
    assert observable["new_unavailable_observable_count"] == 4
    assert observable["minimum_public_states"] == 3
    assert observable["minimum_heldout_rank"] == 2


def test_i2d_all_physical_reversibility_tests_are_unpassed() -> None:
    result = validate_gap_contract()
    matrix = result["test_matrix"]
    assert matrix["required_test_count"] == 15
    assert matrix["current_pass_count"] == 0
    assert matrix["hard_kill_count"] >= 8


def test_i2d_physical_package_is_not_freeze_ready() -> None:
    result = validate_gap_contract()
    assert result["decision"] == (
        "I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT_COMPLETE__"
        "OBSERVABLE_AND_BACKEND_PREREQUISITES_UNMET"
    )
    assert result["freeze_decision"] == {
        "gap_contract_complete": True,
        "physical_package_freeze_ready": False,
        "next_gate": "I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON",
    }
    for token in (
        "NO_PHYSICAL_REVERSIBLE_GENERATOR_ESTABLISHED",
        "NO_PHYSICAL_INVERSE_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ):
        assert token in result["claim_ceiling"]
