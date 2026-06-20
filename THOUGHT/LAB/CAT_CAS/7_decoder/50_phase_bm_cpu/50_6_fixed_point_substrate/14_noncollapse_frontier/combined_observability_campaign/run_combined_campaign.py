#!/usr/bin/env python3
"""Orchestrate the frozen campaign through a schedule-driven local runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from catcas_preflight import inspect as inspect_preflight
from compile_session_schedule import write_session
from generate_campaign_plan import verify as verify_plan

ROUTE_CORES = {"v4s5": (4, 5), "v2s3": (2, 3)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def runner_command(runner: Path, session_dir: Path, output_dir: Path, route: str, args: argparse.Namespace) -> list[str]:
    victim, sender = ROUTE_CORES[route]
    command = [
        str(runner),
        "--session-dir", str(session_dir),
        "--output-dir", str(output_dir),
        "--victim", str(victim),
        "--sender", str(sender),
        "--pin-khz", str(args.pin_khz),
        "--slot-s", str(args.slot_s),
        "--off-window-s", str(args.off_window_s),
        "--read-hz", str(args.read_hz),
        "--temp-veto-c", str(args.temp_veto_c),
        "--executor-commit", str(getattr(args, "executor_commit", "0" * 40)),
    ]
    if getattr(args, "runner_validate_only", False):
        command.append("--validate-only")
    else:
        command.append("--hardware")
    return command


def selected_sessions(plan: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    sessions = plan["sessions"]
    if not requested:
        return sessions
    lookup = {session["session_id"]: session for session in sessions}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"unknown sessions: {missing}")
    return [lookup[name] for name in requested]


def execute(args: argparse.Namespace) -> int:
    plan_dir = args.plan_dir.resolve()
    repo_root = args.repo_root.resolve()
    runner = args.runner.resolve()
    evidence_root = args.evidence_root.resolve()
    if evidence_root.exists():
        raise FileExistsError(f"refusing existing evidence root: {evidence_root}")
    errors = verify_plan(plan_dir)
    if errors:
        raise ValueError(f"invalid plan: {errors}")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise FileNotFoundError(f"runner is not executable: {runner}")

    plan = load_json(plan_dir / "campaign_plan.json")
    manifest = load_json(plan_dir / "campaign_manifest.json")
    sessions = selected_sessions(plan, args.session)
    if not args.runner_validate_only:
        preflight_path = evidence_root.with_name(evidence_root.name + ".preflight.json")
        preflight = inspect_preflight(plan_dir, repo_root, evidence_root, args.min_free_gb)
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not preflight["acquisition_ready"]:
            print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
            return 2

    evidence_root.mkdir(parents=True, exist_ok=False)
    (evidence_root / "sessions").mkdir()
    (evidence_root / "runs").mkdir()
    shutil.copy2(plan_dir / "campaign_plan.json", evidence_root / "campaign_plan.json")
    shutil.copy2(plan_dir / "campaign_manifest.json", evidence_root / "campaign_manifest.json")

    execution = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_EXECUTION_V1",
        "plan_sha256": manifest["campaign_plan"]["sha256"],
        "plan_source_commit": manifest["source_commit"],
        "runner": str(runner),
        "runner_validate_only": bool(args.runner_validate_only),
        "sessions_requested": [session["session_id"] for session in sessions],
        "sessions_completed": [],
        "status": "RUNNING",
        "automatic_retry": False,
        "restoration_authorized": False,
    }
    execution_path = evidence_root / "execution.json"
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for session in sessions:
        session_id = session["session_id"]
        session_dir = evidence_root / "sessions" / session_id
        run_dir = evidence_root / "runs" / session_id
        write_session(evidence_root / "campaign_plan.json", session_id, session_dir)
        command = runner_command(runner, session_dir, run_dir, session["route"], args)
        if args.dry_run:
            print(" ".join(command))
            continue
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "orchestrator_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "orchestrator_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not (run_dir / "run_manifest.json").is_file():
            execution["status"] = "FAILED"
            execution["failed_session"] = session_id
            execution["runner_exit_code"] = proc.returncode
            execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return proc.returncode or 5
        execution["sessions_completed"].append(session_id)
        execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    execution["status"] = "DRY_RUN_COMPLETE" if args.dry_run else "COMPLETE"
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--session", action="append")
    parser.add_argument("--pin-khz", type=int, default=1600000)
    parser.add_argument("--slot-s", type=float, default=0.5)
    parser.add_argument("--off-window-s", type=float, default=0.5)
    parser.add_argument("--read-hz", type=int, default=4000)
    parser.add_argument("--temp-veto-c", type=float, default=68.0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--runner-validate-only", action="store_true")
    parser.add_argument("--executor-commit", default="0" * 40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        return execute(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
