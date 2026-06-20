#!/usr/bin/env python3
"""Finalize one captured Slot 2 witness run without overwriting evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import carrier_witness_manifest as manifests

CAPTURE_FILES = (
    "schedule.json", "windows.csv", "raw_samples.bin", "summary.csv",
    "stdout.log", "stderr.log",
)


def write_exclusive_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def validate_metadata(run_dir: Path, run: dict, source_commit: str) -> None:
    schedule = json.loads((run_dir / "schedule.json").read_text())
    required = ("campaign_id", "run_id", "condition", "route", "seed", "timing",
                "drive", "thermal", "exit", "compiler", "host", "binary_sha256",
                "source_files")
    missing = [key for key in required if key not in run]
    if missing:
        raise ValueError(f"run metadata missing fields: {missing}")
    if run.get("schema_id") != "CAT_CAS_PDN_CARRIER_RUN_V1":
        raise ValueError("invalid run schema_id")
    if run.get("run_id") != run_dir.name:
        raise ValueError("run_id does not match immutable directory")
    if run.get("source_commit") != source_commit:
        raise ValueError("source commit mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be lowercase 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", str(run["binary_sha256"])):
        raise ValueError("binary SHA-256 must be lowercase 64-hex")
    source_files = run["source_files"]
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source file hash map is required")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in source_files.values()):
        raise ValueError("source file hashes must be lowercase SHA-256")
    if schedule.get("run_id") != run["run_id"] or schedule.get("campaign_id") != run["campaign_id"]:
        raise ValueError("schedule identity mismatch")
    t0 = schedule.get("t0_tsc")
    if not isinstance(t0, int) or t0 <= 0 or run["timing"].get("t0_tsc") != t0:
        raise ValueError("persisted t0 mismatch")
    thermal = run["thermal"]
    values = [thermal.get("minimum_c"), thermal.get("maximum_c"), thermal.get("veto_c")]
    if not thermal.get("source") or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("invalid thermal metadata")
    if thermal["minimum_c"] < -100.0 or thermal["maximum_c"] >= thermal["veto_c"]:
        raise ValueError("thermal metadata violates acquisition gate")
    status = run["exit"]
    if any(status.get(name) != 0 for name in ("sender", "receiver", "orchestrator")):
        raise ValueError("cannot finalize failed process exit")
    if status.get("pstate_restored") is not True or status.get("affinity_verified") is not True:
        raise ValueError("cannot finalize without restoration and affinity proof")
    if status.get("temperature_veto_triggered") is not False:
        raise ValueError("cannot finalize a temperature-vetoed run")


def finalize_run(run_dir: Path, metadata_path: Path, analyzer: Path, source_commit: str) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [name for name in CAPTURE_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete run; missing {missing}")
    outputs = [run_dir / name for name in ("run.json", "analysis.json", "run_manifest.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("finalization refuses to overwrite existing outputs")
    run = json.loads(metadata_path.read_text())
    if not isinstance(run, dict):
        raise ValueError("run metadata root must be an object")
    validate_metadata(run_dir, run, source_commit)

    analysis_path = run_dir / "analysis.json"
    process = subprocess.run(
        [sys.executable, str(analyzer), str(run_dir / "summary.csv"), str(analysis_path)],
        text=True, capture_output=True, check=False,
    )
    if process.returncode not in (0, 1) or not analysis_path.is_file():
        raise RuntimeError(f"analyzer failed rc={process.returncode}: {process.stderr.strip()}")
    write_exclusive_json(run_dir / "run.json", run)
    manifest = manifests.make_run_manifest(run_dir, source_commit)
    manifests.write_json(run_dir / "run_manifest.json", manifest)
    errors = manifests.verify_manifest(run_dir / "run_manifest.json")
    if errors:
        raise RuntimeError(f"generated run manifest failed verification: {errors}")
    return run_dir / "run_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        output = finalize_run(args.run_dir.resolve(), args.metadata.resolve(),
                              args.analyzer.resolve(), args.source_commit)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
