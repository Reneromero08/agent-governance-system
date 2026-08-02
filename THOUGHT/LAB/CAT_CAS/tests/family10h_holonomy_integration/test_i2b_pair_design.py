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

from i2b_pair_design import run_pair_design_audit  # noqa: E402


def test_i2b_binds_retained_pair_attacks() -> None:
    result = run_pair_design_audit()
    assert result["passed"] is True
    assert len(result["source_identity_verified"]) == 5
    assert all(result["retained_attack_checks"].values())
    assert result["claim_boundary"]["attacked_family_count"] == 7


def test_i2b_partial_overlap_fixture_is_design_only() -> None:
    result = run_pair_design_audit()
    geometry = result["synthetic_geometry"]
    assert geometry == {
        "line_set_size": 16,
        "intersection_size": 8,
        "union_size": 24,
    }
    assert result["claim_boundary"]["candidate_id"] == (
        "multi_destination_partial_overlap_ownership_pair"
    )
    assert result["claim_boundary"]["physical_pair_supported"] is False


def test_i2b_keeps_h1_h2_and_live_claims_false() -> None:
    result = run_pair_design_audit()
    assert result["decision"] == (
        "I2B_SECOND_GENERATOR_AND_INVERSE_DESIGN_COMPLETE__NO_PHYSICALLY_SUPPORTED_PAIR"
    )
    assert result["claim_boundary"]["next_gate"] == (
        "I2C_SYNTHETIC_BIDIRECTIONAL_OWNERSHIP_MODEL"
    )
    for token in (
        "SECOND_GENERATOR_NOT_ESTABLISHED",
        "INVERSE_NOT_ESTABLISHED",
        "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ):
        assert token in result["claim_ceiling"]
