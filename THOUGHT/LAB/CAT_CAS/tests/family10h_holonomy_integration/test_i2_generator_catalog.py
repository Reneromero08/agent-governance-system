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

from i2_generator_catalog import run_catalog_audit  # noqa: E402


def test_i2_catalog_binds_exact_runtime_sources() -> None:
    result = run_catalog_audit()
    assert result["passed"] is True
    assert len(result["source_identity_verified"]) == 6
    assert all(result["runtime_classification_checks"].values())


def test_i2_catalog_admits_no_current_generator_pair() -> None:
    result = run_catalog_audit()
    checks = result["catalog_checks"]
    assert checks["family_count"] == 9
    assert checks["admitted_generator_count"] == 0
    assert checks["admitted_pair_count"] == 0
    assert result["decision"] == (
        "I2_RECEIVER_SIDE_GENERATOR_CATALOG_COMPLETE__NO_H1_ADMISSIBLE_GENERATOR_PAIR"
    )


def test_i2_preserves_only_one_extraction_candidate() -> None:
    result = run_catalog_audit()
    checks = result["catalog_checks"]
    assert checks["extraction_candidate"] == "remote_store_same_value"
    assert checks["next_gate"] == "I2A_POST_SOURCE_OPERATOR_RUNTIME_EXTRACTION_DESIGN"
    assert result["claim_ceiling"] == [
        "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ]
