#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from campaign_plan import make_plan, validate

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(output: Path, source_commit: str, ratification: Path) -> dict[str, Any]:
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be lowercase 40-hex")
    output.mkdir(parents=True, exist_ok=False)
    ratification_sha = sha256_file(ratification)
    plan = make_plan(source_commit, ratification_sha)
    errors = validate(plan)
    if errors:
        raise ValueError("; ".join(errors))
    plan_path = output / "campaign_plan.json"
    write_json(plan_path, plan)
    manifest = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_OBSERVABILITY_MANIFEST_V1",
        "source_commit": source_commit,
        "ratification_sha256": ratification_sha,
        "campaign_plan": {"size": plan_path.stat().st_size, "sha256": sha256_file(plan_path)},
        "summary": {
            "sessions": len(plan["sessions"]),
            "tone_symbols": sum(len(block["symbols"]) for session in plan["sessions"] for block in session["blocks"]["tone_order"]),
            "persistence_events": sum(len(session["blocks"]["persistence"]) for session in plan["sessions"]),
            "trajectory_steps": sum(len(block["steps"]) for session in plan["sessions"] for block in session["blocks"]["trajectories"]),
        },
        "physical_acquisition_authorized_after_preflight": True,
        "restoration_authorized": False,
    }
    write_json(output / "campaign_manifest.json", manifest)
    return manifest


def verify(output: Path) -> list[str]:
    manifest = json.loads((output / "campaign_manifest.json").read_text())
    plan_path = output / "campaign_plan.json"
    errors: list[str] = []
    if not plan_path.is_file():
        return ["missing campaign_plan.json"]
    if plan_path.stat().st_size != manifest["campaign_plan"]["size"]:
        errors.append("plan size mismatch")
    if sha256_file(plan_path) != manifest["campaign_plan"]["sha256"]:
        errors.append("plan hash mismatch")
    errors.extend(validate(json.loads(plan_path.read_text())))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--ratification", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        print(json.dumps(generate(args.output.resolve(), args.source_commit, args.ratification.resolve()), indent=2, sort_keys=True))
        return 0
    errors = verify(args.output.resolve())
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
