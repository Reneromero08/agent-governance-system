#!/usr/bin/env python3
"""Defect-aware audit of one Phase 6 raw acquisition session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from waveform_reference import (
    MODE_NAMES,
    RECOVERY_HARMONICS,
    acquisition_gate,
    lockin,
    matched_gate_correlation,
    phase_index,
    tone_hz,
)

RAW_DTYPE = np.dtype([("timestamp_tsc", "<u8"), ("ring_period", "<f8")])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_schedule(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("schedule rows must be JSON objects")
    return rows


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def circular_summary(values: np.ndarray) -> dict[str, float]:
    unit = np.exp(1j * values)
    mean = np.mean(unit)
    center = float(np.angle(mean))
    residual = np.angle(np.exp(1j * (values - center)))
    return {
        "concentration_R": float(abs(mean)),
        "mean_angle_rad": center,
        "median_abs_centered_residual_rad": float(np.median(np.abs(residual))),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def validate_schedule_result(
    schedule: list[dict[str, Any]], results: list[dict[str, str]]
) -> list[str]:
    errors: list[str] = []
    if len(schedule) != len(results):
        return [f"row count differs: schedule={len(schedule)} results={len(results)}"]
    fields = (
        "window_index",
        "session_id",
        "stage",
        "block_id",
        "family",
        "actual_mode",
        "declared_mode",
        "executed_tone_order",
        "declared_tone_order",
        "physical_tone_index",
        "codeword_source_index",
        "drive_on",
        "sender_off_required",
        "measurement_mode",
        "amplitude_level",
        "theta_idx",
    )
    integer_fields = {
        "window_index",
        "physical_tone_index",
        "codeword_source_index",
        "amplitude_level",
        "theta_idx",
        "drive_on",
        "sender_off_required",
    }
    for index, (expected, actual) in enumerate(zip(schedule, results)):
        normalized = dict(expected)
        normalized.setdefault(
            "amplitude_level", 3 if normalized.get("drive_on") else 0
        )
        for field in fields:
            expected_value = normalized.get(field)
            if field in integer_fields:
                if field in {"drive_on", "sender_off_required"}:
                    expected_value = int(bool(expected_value))
                elif expected_value is None:
                    expected_value = -1
                else:
                    expected_value = int(expected_value)
                actual_value: Any = int(actual[field])
            else:
                expected_value = "null" if expected_value is None else str(expected_value)
                actual_value = actual[field]
            if expected_value != actual_value:
                errors.append(
                    f"window {index} field {field}: "
                    f"expected={expected_value!r} actual={actual_value!r}"
                )
                break
    return errors


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    session_dir = args.session_dir.resolve()
    raw_path = args.raw.resolve()

    run = load_json(run_dir / "run.json")
    run_manifest = load_json(run_dir / "run_manifest.json")
    session = load_json(session_dir / "session.json")
    schedule = read_schedule(session_dir / "windows.jsonl")
    results = read_results(run_dir / "window_results.csv")

    errors = validate_schedule_result(schedule, results)
    raw_binding = run_manifest.get("files", {}).get("raw_samples.bin")
    if not isinstance(raw_binding, dict):
        raise ValueError("run manifest lacks raw_samples.bin binding")
    raw_size = raw_path.stat().st_size
    raw_sha256 = sha256_file(raw_path)
    if raw_size != raw_binding.get("size"):
        errors.append(
            f"raw size mismatch: expected={raw_binding.get('size')} actual={raw_size}"
        )
    if raw_sha256 != raw_binding.get("sha256"):
        errors.append(
            f"raw SHA-256 mismatch: expected={raw_binding.get('sha256')} "
            f"actual={raw_sha256}"
        )

    counts = np.array([int(row["sample_count"]) for row in results], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    expected_records = int(offsets[-1])
    if raw_size != expected_records * RAW_DTYPE.itemsize:
        errors.append("raw byte size does not equal sample_count total times 16")

    raw = np.memmap(raw_path, dtype=RAW_DTYPE, mode="r")
    if len(raw) != expected_records:
        errors.append(
            f"raw record count mismatch: expected={expected_records} actual={len(raw)}"
        )

    tsc_hz = float(run["tsc_calibration_hz"])
    mode_lookup = {name: index for index, name in enumerate(MODE_NAMES)}
    harmonic_magnitude: dict[str, list[float]] = {
        harmonic.label: [] for harmonic in RECOVERY_HARMONICS
    }
    harmonic_magnitude_off: dict[str, list[float]] = {
        harmonic.label: [] for harmonic in RECOVERY_HARMONICS
    }
    matched_correlations: list[float] = []
    matched_betas: list[float] = []
    phase_residuals: list[float] = []
    reproduction_i_errors: list[float] = []
    reproduction_q_errors: list[float] = []
    timestamp_errors = 0
    nonfinite_samples = 0
    nonmonotonic_windows = 0
    frequency_transition_rows: list[dict[str, Any]] = []

    exact_gate_coefficients = np.full(len(results), np.nan + 1j * np.nan)
    observed_f_over_4 = np.full(len(results), np.nan + 1j * np.nan)

    for index, row in enumerate(results):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        timestamps = np.asarray(raw["timestamp_tsc"][start:stop])
        samples = np.asarray(raw["ring_period"][start:stop])
        if not np.isfinite(samples).all():
            nonfinite_samples += int(np.count_nonzero(~np.isfinite(samples)))
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            nonmonotonic_windows += 1
        if (
            int(timestamps[0]) != int(row["first_sample_tsc"])
            or int(timestamps[-1]) != int(row["last_sample_tsc"])
        ):
            timestamp_errors += 1

        before_khz = int(row["frequency_before_khz"])
        after_khz = int(row["frequency_after_khz"])
        if before_khz != int(run["frequency_policy"]) or after_khz != int(
            run["frequency_policy"]
        ):
            frequency_transition_rows.append(
                {
                    "window_index": index,
                    "stage": row["stage"],
                    "family": row["family"],
                    "frequency_before_khz": before_khz,
                    "frequency_after_khz": after_khz,
                }
            )

        tone_index = int(row["physical_tone_index"])
        if tone_index < 0:
            continue
        frequency = tone_hz(tone_index)
        origin = int(row["slot_start_tsc"])
        coefficients: dict[str, complex] = {}
        for harmonic in RECOVERY_HARMONICS:
            value = lockin(
                timestamps,
                samples,
                origin_tsc=origin,
                tsc_hz=tsc_hz,
                frequency_hz=frequency * harmonic.ratio,
            )
            coefficients[harmonic.label] = value
            target = (
                harmonic_magnitude
                if int(row["drive_on"]) == 1
                else harmonic_magnitude_off
            )
            target[harmonic.label].append(abs(value))

        observed_f_over_4[index] = coefficients["f_over_4"]
        stored_i = row["computed_I"]
        stored_q = row["computed_Q"]
        if stored_i != "null" and stored_q != "null":
            requested = coefficients["requested_f"]
            reproduction_i_errors.append(abs(requested.real - float(stored_i)))
            reproduction_q_errors.append(abs(requested.imag - float(stored_q)))

        if int(row["drive_on"]) != 1:
            continue
        mode_name = row["actual_mode"]
        mode = mode_lookup.get(mode_name)
        source_index = int(row["codeword_source_index"])
        theta_index = int(row["theta_idx"])
        amplitude_level = int(row["amplitude_level"])
        if mode is None or source_index < 0 or theta_index < 0:
            continue
        phase_index_value = phase_index(mode, source_index, theta_index)
        gate = acquisition_gate(
            timestamps,
            origin_tsc=origin,
            tsc_hz=tsc_hz,
            tone_index=tone_index,
            phase_index_value=phase_index_value,
            amplitude_level=amplitude_level,
        )
        beta, correlation = matched_gate_correlation(samples, gate)
        matched_betas.append(beta)
        matched_correlations.append(correlation)
        gate_coefficient = lockin(
            timestamps,
            gate,
            origin_tsc=origin,
            tsc_hz=tsc_hz,
            frequency_hz=frequency / 4.0,
        )
        exact_gate_coefficients[index] = gate_coefficient
        phase_residuals.append(
            float(
                np.angle(
                    coefficients["f_over_4"] * np.conjugate(gate_coefficient)
                )
            )
        )

    result: dict[str, Any] = {
        "schema_id": "CAT_CAS_PHASE6_RAW_SESSION_AUDIT_V1",
        "session_id": run.get("session_id"),
        "partition": session.get("partition"),
        "executor_commit": run.get("executor_git_commit"),
        "raw_file": str(raw_path),
        "raw_size": raw_size,
        "raw_sha256": raw_sha256,
        "raw_manifest_binding": raw_binding,
        "window_count": len(results),
        "record_count": expected_records,
        "timestamp_boundary_errors": timestamp_errors,
        "nonmonotonic_windows": nonmonotonic_windows,
        "nonfinite_samples": nonfinite_samples,
        "schedule_result_errors": errors,
        "stored_lockin_reproduction": {
            "maximum_i_error": max(reproduction_i_errors, default=math.nan),
            "maximum_q_error": max(reproduction_q_errors, default=math.nan),
        },
        "harmonic_magnitude": {},
        "matched_gate": {
            "beta": quantiles(np.asarray(matched_betas)),
            "correlation": quantiles(np.asarray(matched_correlations)),
            "phase_residual": circular_summary(np.asarray(phase_residuals)),
        },
        "frequency_transition_rows": frequency_transition_rows,
    }
    for harmonic in RECOVERY_HARMONICS:
        driven = np.asarray(harmonic_magnitude[harmonic.label])
        off = np.asarray(harmonic_magnitude_off[harmonic.label])
        result["harmonic_magnitude"][harmonic.label] = {
            "ratio": harmonic.ratio,
            "driven": quantiles(driven),
            "off": quantiles(off),
            "median_driven_to_off_ratio": float(
                np.median(driven) / np.median(off)
            ),
        }

    result["verdict"] = {
        "raw_identity_pass": raw_size == raw_binding.get("size")
        and raw_sha256 == raw_binding.get("sha256"),
        "raw_structure_pass": not any(
            (
                timestamp_errors,
                nonmonotonic_windows,
                nonfinite_samples,
                len(errors),
            )
        ),
        "stored_lockin_reproduced": max(
            reproduction_i_errors + reproduction_q_errors, default=1.0
        )
        < 1e-12,
        "requested_frequency_is_dominant": (
            result["harmonic_magnitude"]["requested_f"]["driven"]["median"]
            > result["harmonic_magnitude"]["f_over_4"]["driven"]["median"]
        ),
        "exact_gate_phase_coherent": (
            result["matched_gate"]["phase_residual"]["concentration_R"] > 0.95
        ),
        "scientific_adjudication_complete": False,
    }
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    harmonic_lines = []
    for label, record in result["harmonic_magnitude"].items():
        harmonic_lines.append(
            f"| {label} | {record['ratio']:.2f} | "
            f"{record['driven']['median']:.9g} | "
            f"{record['off']['median']:.9g} | "
            f"{record['median_driven_to_off_ratio']:.6g} |"
        )
    markdown = f"""# Phase 6 Raw Session Audit

**Session:** `{result['session_id']}`  
**Partition:** `{result['partition']}`  
**Executor:** `{result['executor_commit']}`  
**Raw SHA-256:** `{result['raw_sha256']}`

## Integrity

```json
{json.dumps({
    'raw_size': result['raw_size'],
    'record_count': result['record_count'],
    'window_count': result['window_count'],
    'timestamp_boundary_errors': result['timestamp_boundary_errors'],
    'nonmonotonic_windows': result['nonmonotonic_windows'],
    'nonfinite_samples': result['nonfinite_samples'],
    'schedule_result_errors': result['schedule_result_errors'],
    'stored_lockin_reproduction': result['stored_lockin_reproduction'],
}, indent=2, sort_keys=True)}
```

## Harmonic response

| Coordinate | Ratio | Driven median | Off median | Median ratio |
|---|---:|---:|---:|---:|
{chr(10).join(harmonic_lines)}

## Exact executed gate

```json
{json.dumps(result['matched_gate'], indent=2, sort_keys=True)}
```

## Frequency transitions

```json
{json.dumps(result['frequency_transition_rows'], indent=2, sort_keys=True)}
```

## Verdict

```json
{json.dumps(result['verdict'], indent=2, sort_keys=True)}
```
"""
    path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(result, args.output_md)
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    return 0 if result["verdict"]["raw_identity_pass"] and result["verdict"]["raw_structure_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
