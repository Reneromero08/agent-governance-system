#!/usr/bin/env python3
"""Targeted deterministic mutation family for the P0 signal-path witness."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import p0_build_readiness_validator as validator
import p0_scientific_analyzer as analyzer
import p0_signal_path_ordering_proof as ordering
import p0_signal_path_witness_model as circuit_model

OUTPUT = ROOT / "P0_SIGNAL_PATH_MUTATION_RESULTS.json"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structural_mutations(source: bytes) -> list[dict[str, str]]:
    mutations = {
        "remove_actual_path_call": (b"path_metrics = signal_path_transfer(", b"path_metrics = missing_signal_path_transfer("),
        "accept_guard_before_actual_path": (b"n_contact, n_iso, n_admit = decode_guard_witness(", b"n_contact, n_iso, n_admit = signal_path_transfer("),
        "invert_code0_window_bound": (b"open_stop > n_series_open + 1", b"open_stop < n_series_open + 1"),
        "remove_exact_actual_path_claim": (b'"actual_path_claim": "ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT"', b'"actual_path_claim": "UNSUPPORTED_PATH_TOKEN"'),
        "dead_code_actual_path_call": (b"            path_metrics = signal_path_transfer(\n", b"            if False:\n                path_metrics = signal_path_transfer(\n"),
        "drop_result_def_use": (b'"path_metrics": path_metrics', b'"path_metrics": {}'),
        "substitute_call_ch1_with_ch0": (b"                ch0,\n                ch1,\n                n_gate,\n                records[role][\"series_start\"],", b"                ch0,\n                ch0,\n                n_gate,\n                records[role][\"series_start\"],"),
        "substitute_call_series_start": (b'                records[role]["series_start"],\n                records[role]["n_series_open"],', b'                records[role]["n_gate"],\n                records[role]["n_series_open"],'),
        "substitute_call_series_end": (b'                records[role]["n_series_open"],\n                meta["_signal_model"],', b'                records[role]["n_gate"],\n                meta["_signal_model"],'),
        "substitute_call_model": (b'                meta["_signal_model"],\n                common["f_ref"],', b'                {},\n                common["f_ref"],'),
        "substitute_pre_source_channel": (b"pre0 = signal_path_tone_fit(ch0, pre_start, pre_stop, thresholds, f_carrier_hz)", b"pre0 = signal_path_tone_fit(ch1, pre_start, pre_stop, thresholds, f_carrier_hz)"),
        "substitute_pre_carrier_channel": (b"pre1 = signal_path_tone_fit(ch1, pre_start, pre_stop, thresholds, f_carrier_hz)", b"pre1 = signal_path_tone_fit(ch0, pre_start, pre_stop, thresholds, f_carrier_hz)"),
        "substitute_open_source_channel": (b"open0 = signal_path_tone_fit(ch0, open_start, open_stop, thresholds, f_carrier_hz)", b"open0 = signal_path_tone_fit(ch1, open_start, open_stop, thresholds, f_carrier_hz)"),
        "substitute_open_carrier_channel": (b"open1 = signal_path_tone_fit(ch1, open_start, open_stop, thresholds, f_carrier_hz)", b"open1 = signal_path_tone_fit(ch0, open_start, open_stop, thresholds, f_carrier_hz)"),
        "substitute_pre_transfer_denominator": (b'h_pre = pre1["c2"] / pre0["c2"]', b'h_pre = pre1["c2"] / open0["c2"]'),
        "substitute_open_transfer_numerator": (b'h_open = open1["c2"] / open0["c2"]', b'h_open = pre1["c2"] / open0["c2"]'),
        "substitute_pre_snr_variance": (b'pre_snr = abs(pre1["c2"]) / math.sqrt(max(pre1["c2_variance"], 1e-30))', b'pre_snr = abs(pre1["c2"]) / math.sqrt(max(open1["c2_variance"], 1e-30))'),
        "substitute_separation": (b"separation = abs(h_pre - h_open)", b"separation = abs(h_pre + h_open)"),
        "substitute_r_drop": (b'r_drop = abs(open1["c2"]) / abs(pre1["c2"])', b'r_drop = abs(pre1["c2"]) / abs(open1["c2"])'),
        "substitute_pre_phase": (b"phase_pre = math.atan2(h_pre.imag, h_pre.real)", b"phase_pre = math.atan2(h_open.imag, h_open.real)"),
        "substitute_open_phase": (b"phase_open = math.atan2(h_open.imag, h_open.real)", b"phase_open = math.atan2(h_pre.imag, h_pre.real)"),
        "remove_pre_window_cycle_gate": (b'if (pre_stop - pre_start) * (f_witness_hz - f_witness_u95_hz) / FS < model_number(pre_spec["valid_cycles"], "model.pre_valid_cycles") - 1e-12:', b"if False:"),
        "remove_open_window_cycle_gate": (b'if (open_stop - open_start) * (f_witness_hz - f_witness_u95_hz) / FS < model_number(open_spec["valid_cycles"], "model.open_valid_cycles") - 1e-12:', b"if False:"),
        "invert_clipping_gate": (b"if raw_peak > clip:", b"if raw_peak < clip:"),
        "remove_c2_presence_gate": (b'if abs(pre0["c2"]) <= 0.0 or abs(open0["c2"]) <= 0.0:', b"if False:"),
        "invert_source_snr_gate": (b'if source_snr < model_number(thresholds["minimum_pre_pilot_snr"], "model.minimum_pre_pilot_snr"):', b'if source_snr > model_number(thresholds["minimum_pre_pilot_snr"], "model.minimum_pre_pilot_snr"):'),
        "invert_pre_snr_gate": (b'if pre_snr < model_number(thresholds["minimum_pre_pilot_snr"], "model.minimum_pre_pilot_snr"):', b'if pre_snr > model_number(thresholds["minimum_pre_pilot_snr"], "model.minimum_pre_pilot_snr"):'),
        "invert_minimum_pre_transfer_gate": (b'if abs(h_pre) < model_number(thresholds["minimum_pre_abs_h2"], "model.minimum_pre_abs_h2"):', b'if abs(h_pre) > model_number(thresholds["minimum_pre_abs_h2"], "model.minimum_pre_abs_h2"):'),
        "invert_maximum_pre_transfer_gate": (b'if abs(h_pre) > model_number(thresholds["maximum_pre_abs_h2"], "model.maximum_pre_abs_h2"):', b'if abs(h_pre) < model_number(thresholds["maximum_pre_abs_h2"], "model.maximum_pre_abs_h2"):'),
        "remove_pre_phase_gate": (b'if not model_number(phase_limits["minimum"], "model.pre_phase.minimum") <= phase_pre <= model_number(phase_limits["maximum"], "model.pre_phase.maximum"):', b"if False:"),
        "invert_isolation_gate": (b'if abs(h_open) > model_number(thresholds["isolated_abs_h2_max"], "model.isolated_abs_h2_max"):', b'if abs(h_open) < model_number(thresholds["isolated_abs_h2_max"], "model.isolated_abs_h2_max"):'),
        "remove_isolated_phase_gate": (b'if not model_number(isolated_phase_limits["minimum"], "model.isolated_phase.minimum") <= phase_open <= model_number(isolated_phase_limits["maximum"], "model.isolated_phase.maximum"):', b"if False:"),
        "invert_uncertainty_gate": (b'if open_u95 > model_number(thresholds["isolated_u95_h2_max"], "model.isolated_u95_h2_max"):', b'if open_u95 < model_number(thresholds["isolated_u95_h2_max"], "model.isolated_u95_h2_max"):'),
        "invert_separation_gate": (b'if separation < model_number(thresholds["minimum_pre_open_complex_separation"], "model.minimum_pre_open_complex_separation"):', b'if separation > model_number(thresholds["minimum_pre_open_complex_separation"], "model.minimum_pre_open_complex_separation"):'),
        "invert_r_drop_gate": (b'if r_drop > model_number(thresholds["r_drop_max"], "model.r_drop_max"):', b'if r_drop < model_number(thresholds["r_drop_max"], "model.r_drop_max"):'),
        "substitute_assembly_role_mapping": (b'expected_assembly = ASSEMBLY_FOR_ROLE[meta["role"]]', b'expected_assembly = ASSEMBLY_FOR_ROLE["arm_0"]'),
        "remove_assembly_role_gate": (b'if assembly["assembly_id"] != expected_assembly or assembly["carrier_population"] != expected_population:', b"if False:"),
        "remove_assembly_manifest_binding": (b'or assembly["assembly_manifest_sha256"] != receipts["assembly_manifest"]["sha256"]', b'or receipts["assembly_manifest"]["sha256"] != receipts["assembly_manifest"]["sha256"]'),
        "remove_topology_event_binding": (b'or topology_receipt["qualified_native_file_sha256"] != payload["sha256"]', b'or payload["sha256"] != payload["sha256"]'),
        "remove_topology_scan_chronology_gate": (b'if not scan_started < scan_completed <= acquisition_started < acquisition_completed:', b"if False:"),
        "remove_signal_path_invariant_comparison": (b'if canonical_bytes({key: meta["signal_path"][key] for key in SIGNAL_PATH_INVARIANT_FIELDS}) != common["signal_path"]:', b"if False:"),
        "remove_duplicate_topology_scan_gate": (b'if topology_scan_hash in topology_scan_hashes:', b"if False:"),
        "remove_canonical_utc_grammar_gate": (b'if not isinstance(value, str) or CANONICAL_UTC_RE.fullmatch(value) is None:', b"if False:"),
        "remove_canonical_utc_parse_call": (b'parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")', b"parsed = datetime.min"),
        "remove_canonical_utc_roundtrip_gate": (b'if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:', b"if False:"),
        "remove_acquisition_utc_parse": (b'acquisition_started = canonical_utc(chronology_receipt["acquisition_started_utc"], "CHRONOLOGY_CUSTODY")', b"acquisition_started = datetime.min"),
        "remove_scan_utc_parse": (b'scan_started = canonical_utc(topology_receipt["scan_started_utc"], "SIGNAL_PATH_SCAN_CHRONOLOGY")', b"scan_started = datetime.min"),
        "remove_duplicate_nonlinear_control_gate": (b'if nonlinear_control_hash in nonlinear_control_hashes:', b"if False:"),
    }
    outcomes: list[dict[str, str]] = []
    for mutation_id, (needle, replacement) in mutations.items():
        if source.count(needle) != 1:
            raise AssertionError(f"mutation needle cardinality: {mutation_id}")
        mutant = source.replace(needle, replacement, 1)
        try:
            ordering.prove_bytes(mutant)
        except ordering.ProofFailure as exc:
            outcomes.append({"mutation_id": mutation_id, "outcome": "PASS", "rejected_by": str(exc)})
        else:
            raise AssertionError(f"structural mutant survived: {mutation_id}")
    return outcomes


RAW_BOUNDARY_CONTROLS = {
    "signal_path_wrong_node_raw_nominal_labels": "SIGNAL_PATH_TOPOLOGY_SCAN",
    "signal_path_k3_guard_signal_raw": "SIGNAL_PATH_PHASE",
    "signal_path_nonlinear_2f_raw": "SIGNAL_PATH_2F_RESIDUE",
    "signal_path_assembly_a_replay_into_b": "SIGNAL_PATH_ASSEMBLY_ROLE",
    "signal_path_assembly_a_replay_into_c": "SIGNAL_PATH_ASSEMBLY_ROLE",
    "signal_path_topology_receipt_wrong_event": "SIGNAL_PATH_TOPOLOGY_CUSTODY",
    "signal_path_scan_after_acquisition": "SIGNAL_PATH_SCAN_CHRONOLOGY",
    "signal_path_duplicate_topology_scan_hash": "SIGNAL_PATH_TOPOLOGY_SCAN_REPLAY",
    "signal_path_duplicate_nonlinear_control_hash": "SIGNAL_PATH_NONLINEAR_CONTROL_REPLAY",
    "signal_path_scan_time_malformed": "SIGNAL_PATH_SCAN_CHRONOLOGY",
    "signal_path_scan_time_alternate_offset": "SIGNAL_PATH_SCAN_CHRONOLOGY",
    "signal_path_acquisition_time_truncated_precision": "CHRONOLOGY_CUSTODY",
    "signal_path_scan_time_lexical_deception": "SIGNAL_PATH_SCAN_CHRONOLOGY",
    "resonance_calibration_raw_garbage": "RESONANCE_CALIBRATION_RAW",
    "resonance_calibration_q_claim_mismatch": "RESONANCE_CALIBRATION_RAW",
    "source_preparation_before_assignment": "CHRONOLOGY_CUSTODY",
    "off_resonance_raw_response": "OFF_RESONANCE_RESPONSE",
    "off_resonance_probe_too_close": "OFF_RESONANCE_PROBE_BINDING",
}


def raw_boundary_controls(source: bytes) -> list[dict[str, str]]:
    reference_path = ROOT / "P0_ANALYZER_REFERENCE_RESULTS.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference["artifact_custody"]["analyzer_sha256"] != sha256(source):
        raise AssertionError("raw boundary analyzer custody")
    if reference["raw_adversary_count"] != len(analyzer.RAW_ADVERSARIES):
        raise AssertionError("raw boundary adversary count")
    actual = {item["case"]: item for item in reference["raw_adversary_outcomes"]}
    outcomes: list[dict[str, str]] = []
    for case, expected in RAW_BOUNDARY_CONTROLS.items():
        item = actual.get(case)
        if item is None or item.get("outcome") != "PASS" or item.get("rejected_by") != expected:
            raise AssertionError(f"raw boundary control: {case}")
        outcomes.append({"case": case, "outcome": "PASS", "rejected_by": expected})
    return outcomes


def stale_model_mutation() -> dict[str, Any]:
    data = (ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json").read_bytes()
    if b'"minimum_pre_pilot_snr": 20.0' not in data:
        raise AssertionError("model mutation anchor")
    mutant = data.replace(b'"minimum_pre_pilot_snr": 20.0', b'"minimum_pre_pilot_snr": 19.0', 1)
    regenerated = circuit_model.canonical(circuit_model.build_document())
    if mutant == regenerated:
        raise AssertionError("stale model mutant survived")
    return {"mutation_id": "mutate_frozen_model_threshold", "outcome": "PASS", "rejected_by": "COMMITTED_MODEL_STALE"}


def build_document() -> dict[str, Any]:
    snapshot = validator.read_snapshot(False)
    root = validator.candidate_root(snapshot)
    signal_controls = analyzer.run_signal_path_control_suite(ROOT)
    if any(item.get("outcome") != "PASS" for item in signal_controls):
        raise AssertionError("signal-path semantic mutation suite")
    analyzer_source = (ROOT / "p0_scientific_analyzer.py").read_bytes()
    structural = structural_mutations(analyzer_source)
    raw_controls = raw_boundary_controls(analyzer_source)
    stale = stale_model_mutation()
    return {
        "candidate_root": root,
        "claim_ceiling": "NON_EXECUTING_P0_SIGNAL_PATH_WITNESS_REPAIR_ONLY",
        "contact_attestation": {"audio_playback_or_recording": 0, "hardware": 0, "instrument_command": 0, "purchase": 0, "target": 0},
        "model_mutation": stale,
        "raw_boundary_control_count": len(raw_controls),
        "raw_boundary_controls": raw_controls,
        "result": "PASS",
        "schema": "p0.signal-path-mutation-results.v2",
        "semantic_input_mutation_count": len(signal_controls) - len(analyzer.SIGNAL_PATH_POSITIVES),
        "semantic_outcomes_sha256": sha256(canonical(signal_controls)),
        "structural_mutation_count": len(structural),
        "structural_outcomes": structural,
        "surviving_mutants": 0,
        "total_mutation_count": len(signal_controls) - len(analyzer.SIGNAL_PATH_POSITIVES) + len(structural) + len(raw_controls) + 1,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["build"], ["verify"]):
        print("usage: p0_signal_path_witness_mutation_test.py {build|verify}", file=sys.stderr)
        return 2
    data = canonical(build_document())
    if args == ["build"]:
        temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(OUTPUT)
    elif not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
        print("FAIL: P0_SIGNAL_PATH_MUTATION_RESULTS.json is stale", file=sys.stderr)
        return 1
    print(json.dumps({"bytes": len(data), "result": "PASS", "sha256": sha256(data)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
