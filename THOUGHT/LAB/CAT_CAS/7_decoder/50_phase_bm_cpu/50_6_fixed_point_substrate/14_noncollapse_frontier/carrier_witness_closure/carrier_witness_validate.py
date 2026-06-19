#!/usr/bin/env python3
"""Validate and reconstruct a Phase 6B.5 PDN carrier-witness campaign.

This validator operates below the scored-summary boundary. It verifies run
manifests, reads little-endian raw `(t_tsc, ro_period)` records, recomputes every
window's lock-in I/Q and off-bin floor, rebuilds the legacy-compatible summary
CSV, reruns the existing T300 analyzer, and emits a route-scoped closure report.

It does not authorize observability acquisition or promote the physical claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_ID = "CAT_CAS_PDN_CARRIER_CLOSURE_REPORT_V1"
RAW_RECORD = struct.Struct("<Qd")
ABS_TOL = 1e-9
REL_TOL = 5e-6
REQUIRED_RUN_FILES = {
    "run.json",
    "schedule.json",
    "windows.csv",
    "raw_samples.bin",
    "summary.csv",
    "analysis.json",
    "stdout.log",
    "stderr.log",
    "run_manifest.json",
}
REQUIRED_WINDOW_COLUMNS = {
    "window_index",
    "sample_offset_records",
    "sample_count",
    "symbol_index",
    "bin_index",
    "family",
    "declared_mode",
    "actual_mode",
    "trial",
    "hash_restored",
    "theta_idx",
    "tone_hz",
    "drive_sign",
    "phase_fraction",
    "control",
    "slot_start_tsc",
    "capture_deadline_tsc",
    "first_sample_tsc",
    "last_sample_tsc",
    "temp_before_c",
    "temp_after_c",
    "cur_khz_before",
    "cur_khz_after",
    "cofvid_pstate_before",
    "cofvid_pstate_after",
    "computed_I",
    "computed_Q",
    "computed_magnitude",
    "computed_floor",
}
METRIC_KEYS = (
    "real_accuracy",
    "real_mode_floor",
    "real_vs_pseudo_floor",
    "pseudo_reject_floor",
    "pseudo_declared_match",
    "wrong_actual_match",
    "wrong_declared_match",
    "rho_threshold",
    "phase_corr_true",
    "phase_corr_null",
    "phase_delta",
)


class ValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def close(actual: float, expected: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
        actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL
    )


def lockin(
    timestamps: list[int],
    values: list[float],
    frequency_hz: float,
    origin_tsc: int,
    tsc_hz: float,
) -> tuple[float, float, float]:
    count = len(values)
    if count < 4:
        raise ValidationError("lock-in window has fewer than four samples")
    mean = sum(values) / count
    total_i = 0.0
    total_q = 0.0
    weight_sum = 0.0
    for index, (timestamp, value) in enumerate(zip(timestamps, values, strict=True)):
        weight = 0.5 * (1.0 - math.cos(2.0 * math.pi * index / (count - 1)))
        dt = (timestamp - origin_tsc) / tsc_hz
        phase = 2.0 * math.pi * frequency_hz * dt
        centered = (value - mean) * weight
        total_i += centered * math.cos(phase)
        total_q += centered * math.sin(phase)
        weight_sum += weight
    if weight_sum <= 0.0:
        weight_sum = 1.0
    result_i = 2.0 * total_i / weight_sum
    result_q = 2.0 * total_q / weight_sum
    return result_i, result_q, math.hypot(result_i, result_q)


def verify_manifest(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["run manifest has no files object"]
    for name in sorted(REQUIRED_RUN_FILES - {"run_manifest.json"}):
        record = files.get(name)
        path = run_dir / name
        if not isinstance(record, dict):
            errors.append(f"manifest missing record for {name}")
            continue
        if not path.is_file():
            errors.append(f"missing file {name}")
            continue
        expected_size = record.get("size")
        expected_hash = record.get("sha256")
        if expected_size != path.stat().st_size:
            errors.append(f"size mismatch {name}: {path.stat().st_size} != {expected_size}")
        if expected_hash != sha256_file(path):
            errors.append(f"SHA-256 mismatch {name}")
    return errors


def parse_windows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"empty windows CSV: {path}")
        missing = REQUIRED_WINDOW_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValidationError(f"windows CSV missing columns: {sorted(missing)}")
        return list(reader)


def read_raw_window(
    handle: Any, offset_records: int, sample_count: int
) -> tuple[list[int], list[float]]:
    handle.seek(offset_records * RAW_RECORD.size)
    payload = handle.read(sample_count * RAW_RECORD.size)
    if len(payload) != sample_count * RAW_RECORD.size:
        raise ValidationError("raw sample file ended before declared window completed")
    timestamps: list[int] = []
    periods: list[float] = []
    for timestamp, period in RAW_RECORD.iter_unpack(payload):
        timestamps.append(timestamp)
        periods.append(period)
    return timestamps, periods


def parse_legacy_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    comments: list[str] = []
    data_lines: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            comments.append(line)
        elif line.strip():
            data_lines.append(line)
    if not data_lines:
        raise ValidationError(f"summary has no data: {path}")
    reader = csv.DictReader(data_lines)
    return comments, list(reader)


def compare_summary(
    retained: list[dict[str, str]], reconstructed: list[dict[str, str]], nbin: int
) -> list[str]:
    errors: list[str] = []
    if len(retained) != len(reconstructed):
        return [f"summary row count mismatch {len(retained)} != {len(reconstructed)}"]
    text_fields = (
        "family",
        "declared_mode",
        "actual_mode",
        "trial",
        "hash_restored",
        "theta_idx",
    )
    numeric_fields = [
        field for bin_index in range(nbin)
        for field in (f"b{bin_index:02d}_I", f"b{bin_index:02d}_Q")
    ]
    numeric_fields += [f"fl{bin_index:02d}" for bin_index in range(nbin)]
    for index, (left, right) in enumerate(zip(retained, reconstructed, strict=True)):
        for field in text_fields:
            if left.get(field) != right.get(field):
                errors.append(f"summary row {index} field {field} mismatch")
        for field in numeric_fields:
            try:
                left_value = float(left[field])
                right_value = float(right[field])
            except (KeyError, ValueError) as exc:
                errors.append(f"summary row {index} field {field} invalid: {exc}")
                continue
            if not close(left_value, right_value):
                errors.append(
                    f"summary row {index} field {field} mismatch "
                    f"{left_value:.17g} != {right_value:.17g}"
                )
    return errors


def write_reconstructed_summary(
    path: Path,
    schedule: dict[str, Any],
    rows: list[dict[str, str]],
    nbin: int,
) -> None:
    tones = schedule.get("tones_hz", [])
    codebook = schedule.get("codebook", {})
    fieldnames = [
        "family",
        "declared_mode",
        "actual_mode",
        "trial",
        "hash_restored",
        "theta_idx",
    ]
    fieldnames += [
        field for bin_index in range(nbin)
        for field in (f"b{bin_index:02d}_I", f"b{bin_index:02d}_Q")
    ]
    fieldnames += [f"fl{bin_index:02d}" for bin_index in range(nbin)]
    with path.open("w", newline="") as handle:
        handle.write(
            "# SLOT2 MATRIX reconstructed=true "
            f"nbin={nbin} seed={schedule.get('seed')}\n"
        )
        handle.write(
            "# tones_hz="
            + ",".join(f"{float(value):.4f}" for value in tones)
            + "\n"
        )
        for index, name in enumerate(("basis", "rotation", "residual", "mini")):
            values = codebook.get(name, [])
            handle.write(
                f"# codeword_{index}="
                + ",".join(f"{int(value):+d}" for value in values)
                + "\n"
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_run(run_dir: Path, analyzer: Path, temp_root: Path) -> dict[str, Any]:
    required_missing = sorted(
        name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()
    )
    if required_missing:
        return {
            "run_id": run_dir.name,
            "valid": False,
            "errors": [f"missing {required_missing}"],
        }

    errors: list[str] = []
    run = load_json(run_dir / "run.json")
    schedule = load_json(run_dir / "schedule.json")
    manifest = load_json(run_dir / "run_manifest.json")
    retained_analysis = load_json(run_dir / "analysis.json")
    errors.extend(verify_manifest(run_dir, manifest))

    run_id = str(run.get("run_id", run_dir.name))
    if run_id != run_dir.name:
        errors.append(f"run_id/path mismatch {run_id} != {run_dir.name}")
    tsc_hz = float(run.get("timing", {}).get("tsc_hz", 0.0))
    read_hz = int(run.get("timing", {}).get("read_hz", 0))
    nbin = int(run.get("drive", {}).get("nbin", 0))
    veto = float(run.get("thermal", {}).get("veto_c", float("nan")))
    if tsc_hz <= 0 or read_hz <= 0 or nbin <= 0 or not math.isfinite(veto):
        errors.append("invalid timing/drive/thermal metadata")

    windows = parse_windows(run_dir / "windows.csv")
    raw_path = run_dir / "raw_samples.bin"
    expected_records = 0
    reconstructed_by_symbol: dict[int, dict[str, Any]] = {}
    max_abs_error = 0.0
    max_rel_error = 0.0

    with raw_path.open("rb") as raw_handle:
        for expected_index, row in enumerate(windows):
            try:
                window_index = int(row["window_index"])
                offset = int(row["sample_offset_records"])
                count = int(row["sample_count"])
                symbol_index = int(row["symbol_index"])
                bin_index = int(row["bin_index"])
                tone_hz = float(row["tone_hz"])
                start_tsc = int(row["slot_start_tsc"])
                deadline_tsc = int(row["capture_deadline_tsc"])
            except ValueError as exc:
                errors.append(
                    f"window {expected_index} has invalid numeric metadata: {exc}"
                )
                continue
            if window_index != expected_index:
                errors.append(
                    f"window index discontinuity at {expected_index}: {window_index}"
                )
            if offset != expected_records:
                errors.append(
                    f"raw offset discontinuity at window {window_index}: "
                    f"{offset} != {expected_records}"
                )
            if count < 4:
                errors.append(f"window {window_index} has only {count} samples")
                continue
            if not 0 <= bin_index < nbin:
                errors.append(
                    f"window {window_index} bin index {bin_index} outside nbin={nbin}"
                )
                continue
            timestamps, periods = read_raw_window(raw_handle, offset, count)
            expected_records += count
            if any(not math.isfinite(value) for value in periods):
                errors.append(
                    f"window {window_index} contains non-finite ring periods"
                )
            if any(
                right <= left
                for left, right in zip(timestamps, timestamps[1:])
            ):
                errors.append(
                    f"window {window_index} TSC samples are not strictly increasing"
                )
            if timestamps[0] < start_tsc:
                errors.append(f"window {window_index} first TSC precedes slot start")
            nominal_interval = tsc_hz / read_hz if read_hz else 0.0
            if timestamps[-1] > deadline_tsc + 2.0 * nominal_interval:
                errors.append(f"window {window_index} exceeds deadline tolerance")
            if (
                int(row["first_sample_tsc"]) != timestamps[0]
                or int(row["last_sample_tsc"]) != timestamps[-1]
            ):
                errors.append(
                    f"window {window_index} first/last TSC metadata mismatch"
                )
            temperatures = (
                float(row["temp_before_c"]),
                float(row["temp_after_c"]),
            )
            if any(
                not math.isfinite(value) or value < -100.0
                for value in temperatures
            ):
                errors.append(
                    f"window {window_index} has invalid thermal telemetry"
                )
            if any(value >= veto for value in temperatures):
                errors.append(f"window {window_index} reached temperature veto")

            result_i, result_q, result_magnitude = lockin(
                timestamps,
                periods,
                tone_hz,
                start_tsc,
                tsc_hz,
            )
            _, _, result_floor = lockin(
                timestamps,
                periods,
                1.37 * tone_hz + 0.071,
                start_tsc,
                tsc_hz,
            )
            expected_values = {
                "computed_I": result_i,
                "computed_Q": result_q,
                "computed_magnitude": result_magnitude,
                "computed_floor": result_floor,
            }
            for field, expected in expected_values.items():
                actual = float(row[field])
                absolute = abs(actual - expected)
                relative = absolute / max(abs(expected), ABS_TOL)
                max_abs_error = max(max_abs_error, absolute)
                max_rel_error = max(max_rel_error, relative)
                if not close(actual, expected):
                    errors.append(
                        f"window {window_index} {field} mismatch "
                        f"{actual:.17g} != {expected:.17g}"
                    )

            symbol = reconstructed_by_symbol.setdefault(
                symbol_index,
                {
                    "family": (
                        "real" if row["family"] == "preamble" else row["family"]
                    ),
                    "declared_mode": row["declared_mode"],
                    "actual_mode": row["actual_mode"],
                    "trial": row["trial"],
                    "hash_restored": row["hash_restored"],
                    "theta_idx": row["theta_idx"],
                    "bins": {},
                },
            )
            symbol["bins"][bin_index] = (
                result_i,
                result_q,
                result_floor,
            )

    if raw_path.stat().st_size != expected_records * RAW_RECORD.size:
        errors.append(
            f"raw file size mismatch {raw_path.stat().st_size} != "
            f"{expected_records * RAW_RECORD.size}"
        )

    reconstructed_rows: list[dict[str, str]] = []
    for symbol_index in sorted(reconstructed_by_symbol):
        symbol = reconstructed_by_symbol[symbol_index]
        if set(symbol["bins"]) != set(range(nbin)):
            errors.append(f"symbol {symbol_index} does not contain all {nbin} bins")
            continue
        output = {
            "family": str(symbol["family"]),
            "declared_mode": str(symbol["declared_mode"]),
            "actual_mode": str(symbol["actual_mode"]),
            "trial": str(symbol["trial"]),
            "hash_restored": str(symbol["hash_restored"]),
            "theta_idx": str(symbol["theta_idx"]),
        }
        for bin_index in range(nbin):
            value_i, value_q, floor = symbol["bins"][bin_index]
            output[f"b{bin_index:02d}_I"] = f"{value_i:.17g}"
            output[f"b{bin_index:02d}_Q"] = f"{value_q:.17g}"
            output[f"fl{bin_index:02d}"] = f"{floor:.17g}"
        reconstructed_rows.append(output)

    _, retained_rows = parse_legacy_summary(run_dir / "summary.csv")
    errors.extend(compare_summary(retained_rows, reconstructed_rows, nbin))

    reconstructed_path = temp_root / f"{run_id}_summary.csv"
    analysis_path = temp_root / f"{run_id}_analysis.json"
    write_reconstructed_summary(
        reconstructed_path,
        schedule,
        reconstructed_rows,
        nbin,
    )
    process = subprocess.run(
        [
            sys.executable,
            str(analyzer),
            str(reconstructed_path),
            str(analysis_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode not in (0, 1):
        errors.append(
            f"analyzer failed rc={process.returncode}: {process.stderr.strip()}"
        )
        reconstructed_analysis: dict[str, Any] = {}
    else:
        reconstructed_analysis = load_json(analysis_path)
        for key in METRIC_KEYS:
            try:
                actual = float(retained_analysis[key])
                expected = float(reconstructed_analysis[key])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"analysis metric {key} missing/invalid: {exc}")
                continue
            if not close(actual, expected):
                errors.append(
                    f"analysis metric {key} mismatch {actual} != {expected}"
                )
        if retained_analysis.get("gates") != reconstructed_analysis.get("gates"):
            errors.append("analysis gate map mismatch")
        if retained_analysis.get("verdict") != reconstructed_analysis.get("verdict"):
            errors.append("analysis verdict mismatch")

    route = run.get("route", {})
    return {
        "run_id": run_id,
        "valid": not errors,
        "errors": errors,
        "route": route.get("label"),
        "victim": route.get("victim"),
        "sender": route.get("sender"),
        "seed": run.get("seed"),
        "condition": run.get("condition"),
        "source_commit": run.get("source_commit"),
        "raw_records": expected_records,
        "windows": len(windows),
        "max_abs_reconstruction_error": max_abs_error,
        "max_rel_reconstruction_error": max_rel_error,
        "scientific_gates": reconstructed_analysis.get("gates", {}),
        "scientific_pass": bool(reconstructed_analysis.get("gates"))
        and all(reconstructed_analysis.get("gates", {}).values()),
        "verdict": reconstructed_analysis.get("verdict"),
    }


def validate_campaign(campaign_dir: Path, analyzer: Path) -> dict[str, Any]:
    campaign = load_json(campaign_dir / "campaign.json")
    expected_runs = campaign.get("runs", [])
    if not isinstance(expected_runs, list):
        raise ValidationError("campaign runs must be a list")
    expected_ids = {
        str(item["run_id"]): item
        for item in expected_runs
        if isinstance(item, dict) and "run_id" in item
    }
    runs_root = campaign_dir / "runs"
    if not runs_root.is_dir():
        raise ValidationError("campaign has no runs directory")
    run_dirs = {path.name: path for path in runs_root.iterdir() if path.is_dir()}
    missing = sorted(set(expected_ids) - set(run_dirs))
    unexpected = sorted(set(run_dirs) - set(expected_ids))

    with tempfile.TemporaryDirectory(prefix="carrier-witness-") as temp:
        temp_root = Path(temp)
        results = [
            reconstruct_run(run_dirs[run_id], analyzer, temp_root)
            for run_id in sorted(run_dirs)
        ]

    structural_valid = (
        not missing
        and not unexpected
        and all(result["valid"] for result in results)
    )
    primary_route = str(campaign.get("primary_route", "v4s5"))
    required_seeds = {int(seed) for seed in campaign.get("required_seeds", [])}
    primary_matrix = [
        result
        for result in results
        if result.get("route") == primary_route
        and result.get("condition") == "matrix"
    ]
    primary_seed_map = {
        int(result["seed"]): result
        for result in primary_matrix
        if result.get("seed") is not None
    }
    primary_complete = set(primary_seed_map) == required_seeds
    primary_scientific_pass = primary_complete and all(
        primary_seed_map[seed]["valid"]
        and primary_seed_map[seed]["scientific_pass"]
        for seed in required_seeds
    )
    controls = [
        result
        for result in results
        if result.get("route") == primary_route
        and result.get("condition") in {"silent", "scramble"}
    ]
    control_conditions = {
        result.get("condition") for result in controls if result["valid"]
    }
    controls_complete = {"silent", "scramble"} <= control_conditions
    controls_null = controls_complete and all(
        not result["scientific_pass"] for result in controls
    )

    if structural_valid and primary_scientific_pass and controls_null:
        status = "CLOSED_ROUTE_4_5"
    elif any(result["valid"] for result in results):
        status = "PARTIAL"
    elif results:
        status = "INVALID"
    else:
        status = "PENDING"

    return {
        "schema_id": SCHEMA_ID,
        "campaign_id": campaign.get("campaign_id"),
        "contract_id": campaign.get("contract_id"),
        "source_commit": campaign.get("source_commit"),
        "status": status,
        "claim_ceiling": "RECONSTRUCTABLE_CHANNEL_CARRIER_ONLY",
        "structural_valid": structural_valid,
        "missing_runs": missing,
        "unexpected_runs": unexpected,
        "primary_route": primary_route,
        "primary_complete": primary_complete,
        "primary_scientific_pass": primary_scientific_pass,
        "controls_complete": controls_complete,
        "controls_null": controls_null,
        "runs": results,
        "forbidden_claims": [
            "physical HoloGeometry",
            "observable relational state sufficiency",
            "identified operator",
            "physical restoration",
            "target coupling",
            "orientation recovery",
            "Small Wall crossing",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.campaign_dir.is_dir():
        print(
            f"campaign directory does not exist: {args.campaign_dir}",
            file=sys.stderr,
        )
        return 2
    if not args.analyzer.is_file():
        print(f"analyzer does not exist: {args.analyzer}", file=sys.stderr)
        return 2
    try:
        report = validate_campaign(
            args.campaign_dir.resolve(),
            args.analyzer.resolve(),
        )
    except (ValidationError, OSError, ValueError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(report["status"])
    print(f"structural_valid={report['structural_valid']}")
    print(f"primary_complete={report['primary_complete']}")
    print(f"primary_scientific_pass={report['primary_scientific_pass']}")
    print(f"controls_null={report['controls_null']}")
    for run in report["runs"]:
        print(
            f"{run['run_id']}: valid={run['valid']} "
            f"scientific_pass={run['scientific_pass']} "
            f"windows={run['windows']} raw_records={run['raw_records']}"
        )
        for error in run["errors"]:
            print(f"  ERROR: {error}")
    return (
        0
        if report["status"] in {"CLOSED_ROUTE_4_5", "CLOSED_MULTI_ROUTE"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
