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

from i2e_harness_audit import validate_harness_contract  # noqa: E402
from measurement_harness import (  # noqa: E402
    build_valid_synthetic_transaction,
    run_harness_self_test,
    validate_transaction,
)


def test_i2e_valid_fixture_is_complete_but_nonphysical() -> None:
    record = build_valid_synthetic_transaction()
    result = validate_transaction(record)
    assert result["state_checkpoint_count"] == 6
    assert result["operation_receipt_count"] == 8
    assert result["physical_package_freeze_ready"] is False
    assert result["live_execution_authorized"] is False
    assert record["identity"]["physical_backend"] is False
    assert record["thresholds"]["frozen"] is False
    assert record["restoration"]["physical_state_equivalence"] is False


def test_i2e_rejects_all_declared_negative_fixtures() -> None:
    report = run_harness_self_test()
    assert report["passed"] is True
    assert len(report["negative_fixture_results"]) == 12
    assert all(
        expected == actual
        for expected, actual in report["negative_fixture_results"].items()
    )


def test_i2e_state_vector_is_multivariate_and_persistent() -> None:
    record = build_valid_synthetic_transaction()
    observations = record["state_observations"]
    assert len({item["carrier_id"] for item in observations.values()}) == 1
    for item in observations.values():
        assert set(item["multi_observer_coherence_vector"]) == {
            "home_core",
            "remote_core_A",
            "remote_core_B",
            "route_control",
        }
        assert set(item["overlap_strata"]) == {
            "A_only",
            "A_intersection_B",
            "B_only",
            "outside_union",
        }


def test_i2e_contract_binds_harness_and_claim_ceiling() -> None:
    result = validate_harness_contract()
    assert result["passed"] is True
    assert len(result["blob_identity_verified"]) == 2
    assert result["schema_counts"]["negative_fixture_classes"] == 12
    assert result["next_gate"] == "I2F_TOPOLOGY_BACKEND_AND_NDESTRUCTIVE_PROBE_SPEC"
    assert result["claim_ceiling"] == [
        "MEASUREMENT_HARNESS_SCHEMA_ONLY",
        "SYNTHETIC_FIXTURES_ONLY",
        "PHYSICAL_BACKEND_NOT_IMPLEMENTED",
        "NUMERICAL_THRESHOLDS_NOT_FROZEN",
        "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
        "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
        "SMALL_WALL_NOT_CROSSED",
        "NO_LIVE_EXECUTION_AUTHORIZED",
    ]
