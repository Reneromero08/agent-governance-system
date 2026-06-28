"""Computed Phase 6B.6 adjudication rules."""

from __future__ import annotations

from typing import Any


VERDICTS = (
    "SHARED_PREDICTIVE_OPERATOR_SUPPORTED",
    "ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY",
    "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
    "PERSISTENT_STATE_CANDIDATE",
    "CONFOUNDED_NO_OPERATOR_CLAIM",
    "INSTRUMENTATION_BOUNDARY_REJECTED",
)


def validate_thresholds(metrics: dict[str, float]) -> bool:
    return (
        metrics["one_step_nrmse_gain"] >= 0.10
        and metrics["eight_step_nrmse_gain"] >= 0.05
        and metrics["one_step_bootstrap_lower"] > 0.0
        and metrics["eight_step_bootstrap_lower"] > 0.0
        and metrics["route_v4s5_complex_corr"] >= 0.80
        and metrics["route_v2s3_complex_corr"] >= 0.80
        and metrics["worst_session_delta_vs_baseline"] >= -0.05
        and metrics["session_lookup_gain_margin"] > 0.05
    )


def derive_adjudication(results: dict[str, Any]) -> dict[str, Any]:
    predictive = validate_thresholds(results["predictive_metrics"])
    transfer = results["route_transfer"]["v4s5_to_v2s3"]["lower_gain"] > 0.0 and results["route_transfer"]["v2s3_to_v4s5"]["lower_gain"] > 0.0
    drive_off = (
        results["drive_off"]["three_consecutive_lower_above_sham"]
        and results["drive_off"]["zero_input_decay_gain"] >= 0.10
        and results["drive_off"]["zero_input_decay_gain_lower"] > 0.0
    )
    confounded = any(results["confounds"].values())
    if confounded:
        verdicts = ["CONFOUNDED_NO_OPERATOR_CLAIM"]
    elif predictive and transfer:
        verdicts = [
            "SHARED_PREDICTIVE_OPERATOR_SUPPORTED",
            "PERSISTENT_STATE_CANDIDATE" if drive_off else "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
        ]
    elif results["within_route_pass"] and not transfer:
        verdicts = ["ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY"]
    else:
        verdicts = ["INSTRUMENTATION_BOUNDARY_REJECTED"]
    return {
        "schema_id": "CAT_CAS_PHASE6B6_ADJUDICATION_RESULT_V1",
        "verdicts": verdicts,
        "predictive_pass": predictive,
        "bidirectional_transfer_pass": transfer,
        "drive_off_persistence_pass": drive_off,
        "confounded": confounded,
        "hardware_authorized": False,
        "scientific_acquisition_authorized": False,
    }


def adjudicate(**flags: bool) -> tuple[str, ...]:
    raise TypeError("adjudication must derive from computed result objects, not caller pass flags")
