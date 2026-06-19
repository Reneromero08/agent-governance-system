#!/usr/bin/env python3
"""Generate deterministic run and campaign manifests for carrier evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

RUN_FILES = (
    "run.json",
    "schedule.json",
    "windows.csv",
    "raw_samples.bin",
    "summary.csv",
    "analysis.json",
    "stdout.log",
    "stderr.log",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def make_run_manifest(run_dir: Path, source_commit: str) -> dict[str, Any]:
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a lowercase 40-hex Git SHA")
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run directory missing required files: {missing}")
    run = json.loads((run_dir / "run.json").read_text())
    run_id = run.get("run_id")
    if run_id != run_dir.name:
        raise ValueError(f"run_id/path mismatch: {run_id!r} != {run_dir.name!r}")
    if run.get("source_commit") != source_commit:
        raise ValueError("run.json source_commit does not match manifest source commit")
    return {
        "schema_id": "CAT_CAS_PDN_CARRIER_RUN_MANIFEST_V1",
        "run_id": run_id,
        "source_commit": source_commit,
        "files": {name: record(run_dir / name) for name in RUN_FILES},
    }


def make_campaign_manifest(campaign_dir: Path) -> dict[str, Any]:
    campaign_path = campaign_dir / "campaign.json"
    source_path = campaign_dir / "source_manifest.json"
    aggregate_path = campaign_dir / "aggregate" / "aggregate.json"
    closure_path = campaign_dir / "aggregate" / "closure_report.json"
    for path in (campaign_path, source_path, aggregate_path, closure_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    campaign = json.loads(campaign_path.read_text())
    expected_runs = campaign.get("runs")
    if not isinstance(expected_runs, list):
        raise ValueError("campaign.json runs must be a list")
    run_manifests: dict[str, Any] = {}
    for item in expected_runs:
        if not isinstance(item, dict) or "run_id" not in item:
            raise ValueError("campaign run entry lacks run_id")
        run_id = str(item["run_id"])
        path = campaign_dir / "runs" / run_id / "run_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        run_manifests[run_id] = {
            "path": path.relative_to(campaign_dir).as_posix(),
            **record(path),
        }
    return {
        "schema_id": "CAT_CAS_PDN_CARRIER_CAMPAIGN_MANIFEST_V1",
        "campaign_id": campaign.get("campaign_id"),
        "source_commit": campaign.get("source_commit"),
        "files": {
            "campaign.json": record(campaign_path),
            "source_manifest.json": record(source_path),
            "aggregate/aggregate.json": record(aggregate_path),
            "aggregate/closure_report.json": record(closure_path),
        },
        "run_manifests": run_manifests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("run_dir", type=Path)
    run_parser.add_argument("--source-commit", required=True)

    campaign_parser = subparsers.add_parser("campaign")
    campaign_parser.add_argument("campaign_dir", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def verify_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text())
    base = path.parent
    errors: list[str] = []
    if manifest.get("schema_id") == "CAT_CAS_PDN_CARRIER_RUN_MANIFEST_V1":
        files = manifest.get("files", {})
        for name, expected in files.items():
            candidate = base / name
            if not candidate.is_file():
                errors.append(f"missing {candidate}")
            elif record(candidate) != expected:
                errors.append(f"mismatch {candidate}")
    elif manifest.get("schema_id") == "CAT_CAS_PDN_CARRIER_CAMPAIGN_MANIFEST_V1":
        for name, expected in manifest.get("files", {}).items():
            candidate = base / name
            if not candidate.is_file() or record(candidate) != expected:
                errors.append(f"mismatch {candidate}")
        for run_id, expected in manifest.get("run_manifests", {}).items():
            candidate = base / expected["path"]
            expected_record = {"size": expected["size"], "sha256": expected["sha256"]}
            if not candidate.is_file() or record(candidate) != expected_record:
                errors.append(f"mismatch run manifest {run_id}")
    else:
        errors.append("unknown manifest schema")
    return errors


def main() -> int:
    args = parse_args()
    try:
        if args.command == "run":
            run_dir = args.run_dir.resolve()
            output = run_dir / "run_manifest.json"
            write_json(output, make_run_manifest(run_dir, args.source_commit))
            print(output)
            return 0
        if args.command == "campaign":
            campaign_dir = args.campaign_dir.resolve()
            output = campaign_dir / "campaign_manifest.json"
            write_json(output, make_campaign_manifest(campaign_dir))
            print(output)
            return 0
        errors = verify_manifest(args.manifest.resolve())
        for error in errors:
            print(error, file=sys.stderr)
        return 1 if errors else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
