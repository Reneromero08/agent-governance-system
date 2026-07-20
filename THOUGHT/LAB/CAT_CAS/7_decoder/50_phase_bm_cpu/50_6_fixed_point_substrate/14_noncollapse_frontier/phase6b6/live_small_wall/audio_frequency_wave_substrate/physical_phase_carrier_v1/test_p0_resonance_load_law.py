#!/usr/bin/env python3
"""Deterministic offline checks for the prospective P0 resonance/load law."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import p0_scientific_analyzer as analyzer

from p0_resonance_load_law import (
    ACCEPT_MAX_HZ,
    ACCEPT_MIN_HZ,
    MAX_Q,
    MIN_Q,
    MIN_PREPARATION_SECONDS,
    prospective_sanity_document,
)


ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "p0_scientific_analyzer.py"
OPERATIONAL_FUNCTIONS = {
    "validate_metadata",
    "topology_scan_metrics",
    "nonlinear_control_ratio",
    "signal_path_tone_fit",
    "signal_path_transfer",
    "drive_fit",
    "tone_fit",
    "joint_drive_reference_fit",
    "project",
    "arm_metrics",
    "relation_metrics",
}


def hard_coded_frequency_regression() -> None:
    tree = ast.parse(ANALYZER.read_text(encoding="utf-8"), filename=str(ANALYZER))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in OPERATIONAL_FUNCTIONS:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in {"SYNTHETIC_F_CARRIER_HZ", "SYNTHETIC_F_WITNESS_HZ"}:
                    offenders.append(f"{node.name}:{child.lineno}:{child.id}")
                if isinstance(child, ast.Constant) and child.value in {32768, 32768.0, 65536, 65536.0}:
                    offenders.append(f"{node.name}:{child.lineno}:{child.value}")
    if offenders:
        raise AssertionError("operational hard-coded frequency: " + ", ".join(offenders))


def sanity_model_checks() -> None:
    document = prospective_sanity_document()
    assert document["authority"] == "PROSPECTIVE_MODEL_ONLY__NO_PHYSICAL_MEASUREMENT"
    assert document["scope"]["sweep_name"] == "complete binary-corner sweep"
    assert document["scope"]["continuous_uncertainty_envelope_claimed"] is False
    assert document["preparation_law"]["minimum_seconds"] == MIN_PREPARATION_SECONDS
    assert document["selected_load_topology"]["series_limiter_ohm"] == 100_000.0
    assert document["selected_load_topology"]["drive_shunt_ohm"] == 100_000.0
    envelope = document["predicted_envelope"]
    assert ACCEPT_MIN_HZ <= envelope["series_resonance_hz"][0] <= envelope["series_resonance_hz"][1] <= ACCEPT_MAX_HZ
    assert MIN_Q <= envelope["q_factor"][0] <= envelope["q_factor"][1] <= MAX_Q
    assert all(row["loaded_terminal_vpp_at_0p100_vpp"] < 0.100 for row in document["prospective_bvd_binary_corners"])
    assert all(row["ring_up_fraction_after_3s"] > 0.998 for row in document["prospective_bvd_binary_corners"])


def raw_calibration_recomputation_checks() -> None:
    f_carrier_hz = 32_800.0
    decay_seconds = 0.1
    q_factor = math.pi * f_carrier_hz * decay_seconds
    data, spec = analyzer.resonance_calibration_payload("arm_0", f_carrier_hz, q_factor)
    fitted = analyzer.analyze_resonance_calibration_payload(data, spec)
    artifact = {
        "background_imag": analyzer.decimal_text(fitted["background"].imag),
        "background_real": analyzer.decimal_text(fitted["background"].real),
        "canonical_payload": spec,
        "convergence_status": fitted["convergence_status"],
        "decay_seconds": analyzer.decimal_text(fitted["fitted_decay_seconds"]),
        "f_carrier_hz": analyzer.decimal_text(fitted["fitted_f_carrier_hz"]),
        "f_carrier_u95_hz": analyzer.decimal_text(fitted["fitted_u95_hz"]),
        "f_witness_hz": analyzer.decimal_text(2.0 * fitted["fitted_f_carrier_hz"]),
        "fit_condition_number": analyzer.decimal_text(fitted["fit_condition_number"]),
        "frequency_grid_sha256": fitted["frequency_grid_sha256"],
        "gain_imag": analyzer.decimal_text(fitted["gain"].imag),
        "gain_real": analyzer.decimal_text(fitted["gain"].real),
        "off_resonance_probe_hz": analyzer.decimal_text(fitted["probe_frequency_hz"]),
        "off_resonance_response_ratio": analyzer.decimal_text(fitted["response_ratio"]),
        "off_resonance_response_u95": analyzer.decimal_text(fitted["response_u95"]),
        "q_factor": analyzer.decimal_text(fitted["fitted_q_factor"]),
        "q_u95": analyzer.decimal_text(fitted["q_u95"]),
        "reduced_chi_square": analyzer.decimal_text(fitted["reduced_chi_square"]),
        "required_separation_hz": analyzer.decimal_text(fitted["required_separation_hz"]),
        "weighted_residual_rms": analyzer.decimal_text(fitted["weighted_residual_rms"]),
    }
    metrics = analyzer.parse_resonance_calibration_raw(data, "arm_0", artifact)
    assert 0.024 <= metrics["response_ratio"] + metrics["response_u95"] <= 0.030
    assert metrics["required_separation_hz"] >= 20.0
    assert abs(metrics["fitted_q_factor"] - q_factor) <= metrics["q_u95"]
    assert math.isclose(metrics["fitted_decay_seconds"], decay_seconds, rel_tol=1e-5)
    try:
        analyzer.parse_resonance_calibration_raw(b"not calibration data\n", "arm_0", artifact)
    except analyzer.Reject as exc:
        assert exc.code == "RESONANCE_CALIBRATION_PAYLOAD_SIZE"
    else:
        raise AssertionError("self-consistently rebound invalid calibration bytes survived")
    try:
        substituted = dict(artifact)
        substituted["q_factor"] = analyzer.decimal_text(1.01 * fitted["fitted_q_factor"])
        analyzer.parse_resonance_calibration_raw(data, "arm_0", substituted)
    except analyzer.Reject as exc:
        assert exc.code == "RESONANCE_CALIBRATION_RAW"
    else:
        raise AssertionError("artifact Q substituted for raw fitted Q")


def settling_law_checks() -> None:
    tau_max_seconds = 60_000.0 / (math.pi * analyzer.F_CARRIER_MIN_HZ)
    residual_fraction = math.exp(-5.0 / tau_max_seconds)
    assert math.isclose(tau_max_seconds, 0.582842809174, rel_tol=0.0, abs_tol=1e-12)
    assert residual_fraction < 0.0002
    outcomes = analyzer.run_calibration_settling_suite()
    positives = [item for item in outcomes if item["class"] == "calibration_settling_positive"]
    negatives = [item for item in outcomes if item["class"] == "calibration_settling_negative"]
    assert len(positives) == 6
    assert len(negatives) == 11
    assert all(item["outcome"] == "PASS" for item in outcomes)
    assert all(item["minimum_observed_settling_ns"] >= analyzer.CALIBRATION_MIN_SETTLING_NS for item in positives)
    assert {item["case"]: item["rejected_by"] for item in negatives} == analyzer.CALIBRATION_SETTLING_NEGATIVES
    dynamic = next(item for item in positives if item["case"] == "settling_dynamic_32800_375_65600_75")
    assert abs(dynamic["f_carrier_hz"] - 32_800.375) <= dynamic["f_carrier_u95_hz"]
    assert math.isclose(dynamic["f_witness_hz"], 65_600.75, rel_tol=0.0, abs_tol=2.0 * dynamic["f_carrier_u95_hz"])
    data, spec, _ = analyzer._calibration_settling_fixture("settling_mid_q_exact_5s")
    alternate_origin = copy.deepcopy(spec)
    alternate_origin["block_chronology_origin_utc"] = "2041-03-02T04:05:06.123456Z"
    analyzer.analyze_resonance_calibration_payload(data, alternate_origin)
    malformed_origin = copy.deepcopy(spec)
    malformed_origin["block_chronology_origin_utc"] = "2041-03-02T04:05:06Z"
    try:
        analyzer.analyze_resonance_calibration_payload(data, malformed_origin)
    except analyzer.Reject as exc:
        assert exc.code == "RESONANCE_CALIBRATION_CUSTODY"
    else:
        raise AssertionError("noncanonical chronology UTC origin survived")


def main() -> int:
    hard_coded_frequency_regression()
    sanity_model_checks()
    raw_calibration_recomputation_checks()
    settling_law_checks()
    print("P0_RESONANCE_LOAD_LAW_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
