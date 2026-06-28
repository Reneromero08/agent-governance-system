"""Frozen Phase 6B.6 verdict rules."""

from __future__ import annotations


VERDICTS = (
    "SHARED_PREDICTIVE_OPERATOR_SUPPORTED",
    "ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY",
    "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
    "PERSISTENT_STATE_CANDIDATE",
    "CONFOUNDED_NO_OPERATOR_CLAIM",
    "INSTRUMENTATION_BOUNDARY_REJECTED",
)


def adjudicate(
    *,
    shared_predictive_pass: bool,
    drive_off_persistence_pass: bool,
    within_route_pass: bool,
    bidirectional_transfer_pass: bool,
    confounded: bool,
) -> tuple[str, ...]:
    if confounded:
        return ("CONFOUNDED_NO_OPERATOR_CLAIM",)
    if shared_predictive_pass and bidirectional_transfer_pass:
        persistence = "PERSISTENT_STATE_CANDIDATE" if drive_off_persistence_pass else "DRIVEN_RELATIONAL_TRANSPORT_ONLY"
        return ("SHARED_PREDICTIVE_OPERATOR_SUPPORTED", persistence)
    if within_route_pass and not bidirectional_transfer_pass:
        return ("ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY",)
    return ("INSTRUMENTATION_BOUNDARY_REJECTED",)


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
