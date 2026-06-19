#!/usr/bin/env python3
"""Audit existing T300 evidence without modifying acquisition artifacts.

The audit inventories repository summaries and the historical Phenom run tree,
hashes every discovered file, and reports whether raw per-window timing samples
already exist. It never upgrades a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any

SCHEMA_ID = "CAT_CAS_T300_EXISTING_EVIDENCE_AUDIT_V1"
RAW_REQUIRED_NAMES = {
    "run.json",
    "schedule.json",
    "windows.csv",
    "raw_samples.bin",
    "summary.csv",
    "analysis.json",
    "run_manifest.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def discover_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [file_record(path, root) for path in sorted(root.rglob("*")) if path.is_file()]


def detect_raw_run_dirs(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    candidates: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        names = {entry.name for entry in directory.iterdir() if entry.is_file()}
        found = sorted(RAW_REQUIRED_NAMES & names)
        if found:
            candidates.append(
                {
                    "path": str(directory),
                    "found": found,
                    "missing": sorted(RAW_REQUIRED_NAMES - names),
                    "complete": RAW_REQUIRED_NAMES <= names,
                }
            )
    return candidates


def count_pattern(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.exists() else 0


def thermal_inventory() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.exists():
        return out
    for directory in sorted(hwmon.glob("hwmon*")):
        name_path = directory / "name"
        name = name_path.read_text(errors="replace").strip() if name_path.exists() else "unknown"
        temps = []
        for temp_path in sorted(directory.glob("temp*_input")):
            try:
                raw = temp_path.read_text().strip()
                value_c = int(raw) / 1000.0
            except (OSError, ValueError):
                value_c = None
            temps.append({"path": str(temp_path), "value_c": value_c})
        out.append({"directory": str(directory), "name": name, "temperatures": temps})
    return out


def build_audit(repo_root: Path, host_root: Path) -> dict[str, Any]:
    compact_root = repo_root / (
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
        "50_6_fixed_point_substrate/12_chiral_lane_frontier/"
        "pdn_slot2_t300/results"
    )
    historical_source = repo_root / (
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
        "50_6_fixed_point_substrate/10_cross_core_wormhole/slot2_pdn"
    )

    host_files = discover_files(host_root)
    raw_candidates = detect_raw_run_dirs(host_root)
    matrix_csv = count_pattern(host_root / "matrix", "matrix_v*s*_seed*.csv")
    matrix_logs = count_pattern(host_root / "matrix", "matrix_v*s*_seed*.log")
    control_csv = sum(
        count_pattern(host_root / "matrix", f"control_{name}.csv")
        for name in ("silent", "scramble")
    )
    has_complete_raw = any(candidate["complete"] for candidate in raw_candidates)

    gaps: list[str] = []
    if matrix_csv < 12:
        gaps.append(f"historical matrix CSV count is {matrix_csv}, expected at least 12")
    if control_csv < 2:
        gaps.append(f"historical control CSV count is {control_csv}, expected 2")
    if not has_complete_raw:
        gaps.append("no complete per-window raw bundle found")
    if not any(record["path"].endswith("raw_samples.bin") for record in host_files):
        gaps.append("raw_samples.bin absent")
    if not any(record["path"].endswith("windows.csv") for record in host_files):
        gaps.append("windows.csv absent")
    thermal = thermal_inventory()
    valid_thermal = any(
        temp["value_c"] is not None
        for sensor in thermal
        for temp in sensor["temperatures"]
    )
    if not valid_thermal:
        gaps.append("no live readable hwmon temperature input")

    if has_complete_raw and not gaps:
        status = "CANDIDATE_COMPLETE_REQUIRES_VALIDATION"
    elif matrix_csv or host_files:
        status = "PARTIAL"
    else:
        status = "PENDING"

    return {
        "schema_id": SCHEMA_ID,
        "status": status,
        "claim_ceiling": "EVIDENCE_INVENTORY_ONLY",
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "paths": {
            "repo_root": str(repo_root),
            "host_root": str(host_root),
            "compact_result_root": str(compact_root),
            "historical_source_root": str(historical_source),
        },
        "counts": {
            "historical_matrix_csv": matrix_csv,
            "historical_matrix_logs": matrix_logs,
            "historical_control_csv": control_csv,
            "host_files": len(host_files),
            "raw_candidate_directories": len(raw_candidates),
        },
        "thermal_inventory": thermal,
        "raw_candidates": raw_candidates,
        "gaps": gaps,
        "repository_compact_files": discover_files(compact_root),
        "repository_source_files": discover_files(historical_source),
        "host_files": host_files,
        "forbidden_inference": [
            "inventory completeness is not carrier closure",
            "compact summaries are not raw reconstruction evidence",
            "a historical temperature claim does not replace current sensor telemetry",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--host-root", type=Path, default=Path("/root/slot2_pdn"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    host_root = args.host_root.resolve()
    audit = build_audit(repo_root, host_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(audit["status"])
    print(f"matrix_csv={audit['counts']['historical_matrix_csv']}")
    print(f"raw_candidates={audit['counts']['raw_candidate_directories']}")
    for gap in audit["gaps"]:
        print(f"GAP: {gap}")
    return 0 if audit["status"] == "CANDIDATE_COMPLETE_REQUIRES_VALIDATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
