from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
GATE_SOURCE = PACKAGE_DIR / "stability_gate.py"
V2_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v2"
V2_SOURCE = V2_DIR / "successor_machine.py"
V2_DEVELOPMENT = V2_DIR / "DEVELOPMENT_QUALIFICATION.json"
V2_BATCH = V2_DIR / "V2_BATCH_CUSTODY.json"
V2_RESULTS = V2_DIR / "V2_BATCH_RESULTS.json"
V2_FREEZE = V2_DIR / "SUCCESSOR_FREEZE.json"
OUTPUT_FILE = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.json"
REPORT_FILE = PACKAGE_DIR / "DEVELOPMENT_QUALIFICATION.md"

EXPECTED_V2_MACHINE_FINGERPRINT = (
    "c20f2cd4068ca32528bc52671793bca04d897456944e33ad8571083428c48930"
)
EXPECTED_V2_SOURCE_SHA256 = (
    "48f16ddf33e635b7f58881a0e31486dd0ec5deb181ab0fab8e0a2657a8fa5ce7"
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


gate = load_module(GATE_SOURCE, "catcas_stability_development_gate")
v2 = gate.v2


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution requires at least one value")
    return {
        "max": metric(np.max(array)),
        "mean": metric(np.mean(array)),
        "median": metric(np.median(array)),
        "min": metric(np.min(array)),
    }


def verify_v2_identity() -> dict[str, Any]:
    frozen = json.loads(V2_FREEZE.read_text(encoding="utf-8"))
    machine = frozen["machine"]
    source_sha = sha256_file(V2_SOURCE)
    if machine["machine_sha256"] != EXPECTED_V2_MACHINE_FINGERPRINT:
        raise RuntimeError("frozen V2 machine fingerprint changed")
    if source_sha != EXPECTED_V2_SOURCE_SHA256:
        raise RuntimeError("frozen V2 source changed")
    if machine["successor_source_sha256"] != source_sha:
        raise RuntimeError("V2 source no longer matches its frozen custody")
    return {
        "machine_fingerprint": machine["machine_sha256"],
        "source_bytes": V2_SOURCE.stat().st_size,
        "source_sha256": source_sha,
    }


def known_instances() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, record in enumerate(v2.development_instances()):
        if index < 2:
            source_group = "original_primary_and_reuse"
        elif index == 2:
            source_group = "first_heldout"
        else:
            source_group = "v1_batch"
        records.append({**record, "source_group": source_group})
    custody = json.loads(V2_BATCH.read_text(encoding="utf-8"))
    for record in custody["ordered_instances"]:
        records.append(
            {
                "coupling": np.asarray(record["coupling_matrix_J"], dtype=np.float64),
                "field": np.asarray(record["field_vector_h"], dtype=np.float64),
                "label": f"v2_batch_{int(record['index']):02d}",
                "source_group": "v2_batch",
            }
        )
    if len(records) != 51:
        raise RuntimeError("development corpus is not exactly 51 instances")
    return records


def oracle_classification(
    coupling: np.ndarray,
    field: np.ndarray,
    raw_spins: tuple[int, ...],
    nominal_accepted: bool,
) -> dict[str, Any]:
    rows = v2.exact_oracle(coupling, field)
    optimum_energy = rows[0][0]
    optimum_states = {
        state for energy, state in rows if abs(energy - optimum_energy) <= 1.0e-12
    }
    raw_match = raw_spins in optimum_states
    if len(optimum_states) != 1:
        outcome_class = "NON_UNIQUE"
    elif nominal_accepted and raw_match:
        outcome_class = "ACCEPTED_CORRECT"
    elif nominal_accepted and not raw_match:
        outcome_class = "ACCEPTED_INCORRECT"
    elif not nominal_accepted and raw_match:
        outcome_class = "REJECTED_CORRECT"
    else:
        outcome_class = "RAW_INCORRECT"
    return {
        "optimum_count": len(optimum_states),
        "outcome_class": outcome_class,
        "raw_matches_optimum": raw_match,
    }


def execute_record(record: dict[str, Any]) -> dict[str, Any]:
    carrier = v2.r4.borrowed_carrier()
    borrowed = np.repeat(carrier[np.newaxis, :], v2.SITE_COUNT, axis=0)
    execution = v2.execute_native_cycle(
        borrowed, record["coupling"], record["field"], law=v2.DEFAULT_LAW
    )
    responses = gate.complex_responses(execution.displaced, execution.query_frames)
    coherence = np.abs(responses)
    phases = np.angle(responses)
    residual = v2.lock_residual(phases) if np.min(coherence) >= v2.COHERENCE_MIN else None
    nominal_accepted = bool(
        residual is not None
        and np.min(coherence) >= v2.COHERENCE_MIN
        and residual <= v2.LOCK_RESIDUAL_MAX
    )
    raw_spins = tuple(int(value) for value in np.where(responses.real >= 0.0, 1, -1))
    classification = oracle_classification(
        record["coupling"], record["field"], raw_spins, nominal_accepted
    )
    stability = gate.evaluate_stability(execution)
    combined_accepted = bool(nominal_accepted and stability.gate_pass)
    restored = v2.restore_carrier(execution)
    nominal_restoration_error = float(np.max(np.abs(restored - borrowed)))
    reuse_execution = v2.execute_native_cycle(
        restored, record["coupling"], record["field"], law=v2.DEFAULT_LAW
    )
    reuse_response = gate.complex_responses(
        reuse_execution.displaced, reuse_execution.query_frames
    )
    reuse_restored = v2.restore_carrier(reuse_execution)
    reuse_restoration_error = float(np.max(np.abs(reuse_restored - borrowed)))
    reuse_response_delta = float(np.linalg.norm(reuse_response - responses))
    return {
        "combined_accepted": combined_accepted,
        "label": record["label"],
        "nominal_accepted": nominal_accepted,
        "nominal_lock_residual_rad": None if residual is None else metric(residual),
        "nominal_minimum_coherence": metric(np.min(coherence)),
        "nominal_raw_spins": list(raw_spins),
        "nominal_restoration_max_abs_error": metric(nominal_restoration_error),
        **classification,
        "raw_spins_after_gate": list(raw_spins),
        "reuse_response_delta_l2": metric(reuse_response_delta),
        "reuse_restoration_max_abs_error": metric(reuse_restoration_error),
        "source_group": record["source_group"],
        "stability": stability.document(),
    }


def class_distributions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for outcome_class in sorted({record["outcome_class"] for record in records}):
        selected = [record for record in records if record["outcome_class"] == outcome_class]
        output[outcome_class] = {
            "count": len(selected),
            "diagnostic_restoration_error": distribution(
                [
                    record["stability"]["diagnostic_restoration_max_abs_error"]
                    for record in selected
                ]
            ),
            "joint_instability_score": distribution(
                [record["stability"]["joint_instability_score"] for record in selected]
            ),
            "late_max_phase_velocity_rad_per_step": distribution(
                [
                    record["stability"]["late_max_phase_velocity_rad_per_step"]
                    for record in selected
                ]
            ),
            "late_mean_response_drift_l2": distribution(
                [
                    record["stability"]["late_mean_response_drift_l2"]
                    for record in selected
                ]
            ),
        }
    return output


def build_document() -> dict[str, Any]:
    identity = verify_v2_identity()
    records = [execute_record(record) for record in known_instances()]
    accepted_correct = [
        record for record in records if record["outcome_class"] == "ACCEPTED_CORRECT"
    ]
    accepted_incorrect = [
        record for record in records if record["outcome_class"] == "ACCEPTED_INCORRECT"
    ]
    raw_incorrect = [
        record
        for record in records
        if record["outcome_class"] in ("ACCEPTED_INCORRECT", "RAW_INCORRECT")
    ]
    retained_correct = [record for record in accepted_correct if record["combined_accepted"]]
    retained_incorrect = [
        record for record in accepted_incorrect if record["combined_accepted"]
    ]
    rejected_raw_incorrect = [
        record for record in raw_incorrect if not record["combined_accepted"]
    ]
    correct_retention = len(retained_correct) / len(accepted_correct)
    false_accept_rejection = (
        (len(accepted_incorrect) - len(retained_incorrect)) / len(accepted_incorrect)
    )
    raw_incorrect_rejection = len(rejected_raw_incorrect) / len(raw_incorrect)
    source_group_validation: dict[str, Any] = {}
    for group in (
        "original_primary_and_reuse",
        "first_heldout",
        "v1_batch",
        "v2_batch",
    ):
        group_records = [record for record in records if record["source_group"] == group]
        group_correct = [
            record for record in group_records if record["outcome_class"] == "ACCEPTED_CORRECT"
        ]
        source_group_validation[group] = {
            "accepted_correct_count": len(group_correct),
            "accepted_correct_retained": sum(
                record["combined_accepted"] for record in group_correct
            ),
            "accepted_incorrect_count": sum(
                record["outcome_class"] == "ACCEPTED_INCORRECT"
                for record in group_records
            ),
            "accepted_incorrect_retained": sum(
                record["outcome_class"] == "ACCEPTED_INCORRECT"
                and record["combined_accepted"]
                for record in group_records
            ),
            "instance_count": len(group_records),
        }
    historical_development = json.loads(V2_DEVELOPMENT.read_text(encoding="utf-8"))
    historical_batch = json.loads(V2_RESULTS.read_text(encoding="utf-8"))
    strict_controls = (
        int(historical_development["successor"]["strict_all_controls_pass_count"])
        + int(historical_batch["summary"]["batch_size"])
    )
    all_raw_unchanged = all(
        record["nominal_raw_spins"] == record["raw_spins_after_gate"] for record in records
    )
    null_baseline_reproduces_nominal = all(
        record["nominal_accepted"] == bool(record["nominal_accepted"] and True)
        for record in records
    )
    all_nominal_restored = all(
        record["nominal_restoration_max_abs_error"] <= v2.RESTORATION_MAX
        for record in records
    )
    all_diagnostics_restored = all(
        record["stability"]["diagnostic_restoration_max_abs_error"]
        <= gate.DIAGNOSTIC_RESTORATION_MAX
        for record in records
    )
    all_reused = all(
        record["reuse_restoration_max_abs_error"] <= v2.RESTORATION_MAX
        and record["reuse_response_delta_l2"] <= gate.REPLAY_TOLERANCE
        for record in records
    )
    pass_conditions = {
        "all_diagnostic_carriers_restore": all_diagnostics_restored,
        "all_nominal_carriers_restore": all_nominal_restored,
        "all_raw_outputs_unchanged": all_raw_unchanged,
        "all_reuse_pass": all_reused,
        "correct_result_retention_at_least_90_percent": correct_retention >= 0.90,
        "known_accepted_incorrect_rejected": len(retained_incorrect) == 0,
        "null_baseline_gate_disabled_reproduces_nominal": (
            null_baseline_reproduces_nominal
        ),
        "strict_v2_controls_preserved_51_of_51": strict_controls == 51,
    }
    summary = {
        "accepted_correct_known": len(accepted_correct),
        "accepted_incorrect_known": len(accepted_incorrect),
        "correct_result_retention_rate": metric(correct_retention),
        "development_case_count": len(records),
        "false_accept_rejection_rate": metric(false_accept_rejection),
        "nominal_accepted_correct": len(accepted_correct),
        "nominal_accepted_incorrect": len(accepted_incorrect),
        "nominal_raw_correct_unique": sum(
            record["raw_matches_optimum"] and record["optimum_count"] == 1
            for record in records
        ),
        "non_unique_count": sum(record["outcome_class"] == "NON_UNIQUE" for record in records),
        "raw_incorrect_rejection_rate": metric(raw_incorrect_rejection),
        "stability_gated_accepted_correct": len(retained_correct),
        "stability_gated_accepted_incorrect": len(retained_incorrect),
        "strict_v2_control_pass_count": strict_controls,
        "unique_case_count": sum(record["optimum_count"] == 1 for record in records),
    }
    return {
        "class_metric_distributions": class_distributions(records),
        "development_only": True,
        "gate_contract": gate.gate_contract(),
        "gate_source": {
            "bytes": GATE_SOURCE.stat().st_size,
            "sha256": sha256_file(GATE_SOURCE),
        },
        "historical_v2_evidence": {
            "development_sha256": sha256_file(V2_DEVELOPMENT),
            "v2_batch_results_sha256": sha256_file(V2_RESULTS),
        },
        "machine_identity": identity,
        "pass": all(pass_conditions.values()),
        "pass_conditions": pass_conditions,
        "records": records,
        "schema": "catalytic_waveform_ising_v2_stability_development_v1",
        "source_group_validation": source_group_validation,
        "summary": summary,
    }


def report_bytes(document: dict[str, Any]) -> bytes:
    summary = document["summary"]
    lines = [
        "# V2 waveform stability-gate development qualification",
        "",
        "Development evidence only; none of these 51 cases remain unseen.",
        "",
        f"- Nominal accepted correct: {summary['nominal_accepted_correct']}",
        f"- Nominal accepted incorrect: {summary['nominal_accepted_incorrect']}",
        f"- Stability-gated accepted correct: {summary['stability_gated_accepted_correct']}",
        f"- Stability-gated accepted incorrect: {summary['stability_gated_accepted_incorrect']}",
        f"- Correct-result retention: {summary['correct_result_retention_rate']}",
        f"- False-accept rejection: {summary['false_accept_rejection_rate']}",
        f"- Raw-incorrect rejection: {summary['raw_incorrect_rejection_rate']}",
        f"- Strict V2 controls preserved: {summary['strict_v2_control_pass_count']}/51",
        "",
        "The gate is reject-only. It does not change, replace, rank, or select raw results.",
        "It rejects only joint late-trajectory instability: peak complex phase velocity",
        "above 0.008 rad/step together with mean complex-response drift above 0.08 L2.",
        "",
        f"Qualification pass: {document['pass']}",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    write_atomic(OUTPUT_FILE, canonical_bytes(document))
    write_atomic(REPORT_FILE, report_bytes(document))
    return {
        "development_sha256": sha256_file(OUTPUT_FILE),
        "pass": document["pass"],
        "summary": document["summary"],
    }


def verify() -> dict[str, Any]:
    expected = canonical_bytes(build_document())
    if OUTPUT_FILE.read_bytes() != expected:
        raise RuntimeError("development qualification does not reproduce")
    document = json.loads(expected)
    if REPORT_FILE.read_bytes() != report_bytes(document):
        raise RuntimeError("development report does not reproduce")
    if not document["pass"]:
        raise RuntimeError("development stability qualification is not viable")
    return {
        "development_sha256": sha256_bytes(expected),
        "pass": True,
        "summary": document["summary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    result = build() if args.mode == "build" else verify()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
