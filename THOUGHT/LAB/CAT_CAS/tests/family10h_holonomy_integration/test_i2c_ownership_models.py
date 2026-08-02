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

from i2c_model_audit import validate_contract  # noqa: E402
from synthetic_ownership_model import verify_synthetic_models  # noqa: E402


def test_overwrite_model_exposes_fake_restoration() -> None:
    report = verify_synthetic_models()["overwrite_model"]
    assert report["ab_ba_changed_line_count"] == 8
    assert report["order_effect_localized_to_overlap"] is True
    assert report["baseline_endpoint_restored_after_reclaims_ab"] is True
    assert report["baseline_endpoint_restored_after_reclaims_ba"] is True
    assert report["reclaim_is_left_inverse_on_all_states"] is False
    assert report["reclaim_is_right_inverse_on_all_states"] is False
    assert report["forward_map_injective"] is False


def test_carrier_coupled_readout_defeats_word_recorder_control() -> None:
    report = verify_synthetic_models()["overwrite_model"]
    assert report["carrier_coupled_readout_ab"] == 8
    assert report["carrier_coupled_readout_ba"] == -8
    assert report["carrier_off_readout"] == 0
    assert report["invalid_word_recorder_carrier_off"] == 8


def test_reversible_reference_satisfies_protocol_laws_synthetically() -> None:
    report = verify_synthetic_models()["reversible_reference_model"]
    assert report["A_two_sided_inverse"] is True
    assert report["B_two_sided_inverse"] is True
    assert report["commutator_nontrivial"] is True
    assert report["commutator_output"] == 8
    assert report["reverse_commutator_output"] == -8
    assert report["commutator_then_inverse_restores_carrier"] is True
    assert report["retained_output_after_decoupled_inverse"] == 8
    assert report["contractible_word_is_identity"] is True
    assert report["disjoint_support_commutator_is_identity"] is True
    assert report["carrier_off_output"] == 0
    assert report["physical_realization_established"] is False


def test_i2c_contract_binds_model_and_independent_algebra() -> None:
    result = validate_contract()
    assert result["passed"] is True
    assert result["independent_algebra_checks"] == {
        "A_local_two_sided_inverse": True,
        "B_local_two_sided_inverse": True,
        "nonempty_disjoint_support_commutes": True,
    }
    assert result["next_gate"] == "I2D_PHYSICAL_REVERSIBILITY_GAP_CONTRACT"


def test_i2c_claim_ceiling_remains_synthetic() -> None:
    result = validate_contract()
    assert result["decision"] == (
        "I2C_SYNTHETIC_MODELS_COMPLETE__OVERWRITE_MODEL_REJECTED_FOR_H2_R2__"
        "REVERSIBLE_REFERENCE_PASSES_PROTOCOL_LAWS__PHYSICAL_REALIZATION_NOT_ESTABLISHED"
    )
    assert result["claim_ceiling"] == [
        "SYNTHETIC_PROTOCOL_REFERENCE_ONLY",
        "PHYSICAL_REVERSIBLE_OWNERSHIP_CARRIER_NOT_ESTABLISHED",
        "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ]
