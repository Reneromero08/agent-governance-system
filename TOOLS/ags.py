#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_runtime import _slug  # type: ignore
from PRIMITIVES.restore_proof import canonical_json_bytes  # type: ignore


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_idempotent(path: Path, data: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"REFUSE_OVERWRITE: {path} differs")
        return
    _atomic_write_bytes(path, data)


def _as_repo_relpath(path: Path) -> str:
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    if Path(rel).is_absolute():
        raise ValueError("path must be repo-relative")
    return rel


def _load_plan(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("plan must be a JSON object")
    return obj


def _validate_step(step: Any, idx: int) -> Tuple[str, Dict[str, Any], List[str], bool, bool]:
    if not isinstance(step, dict):
        raise ValueError(f"steps[{idx}] must be an object")
    step_id = step.get("step_id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ValueError(f"steps[{idx}].step_id must be non-empty string")
    jobspec = step.get("jobspec")
    if not isinstance(jobspec, dict):
        raise ValueError(f"steps[{idx}].jobspec must be an object")
    cmd = step.get("cmd")
    if cmd is None:
        cmd = ["python3", "-c", "true"]
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        raise ValueError(f"steps[{idx}].cmd must be a non-empty list[str]")
    strict = step.get("strict", True)
    memoize = step.get("memoize", True)
    if not isinstance(strict, bool):
        raise ValueError(f"steps[{idx}].strict must be boolean")
    if not isinstance(memoize, bool):
        raise ValueError(f"steps[{idx}].memoize must be boolean")
    return step_id, jobspec, cmd, strict, memoize


def ags_route(*, plan_path: Path, pipeline_id: str, runs_root: str) -> int:
    if runs_root != "CONTRACTS/_runs":
        raise RuntimeError("UNSUPPORTED_RUNS_ROOT (only CONTRACTS/_runs is supported)")

    plan = _load_plan(plan_path)
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan.steps must be a non-empty list")

    seen: set[str] = set()
    pipeline_dir = REPO_ROOT / runs_root / "_pipelines" / _slug(pipeline_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    pipeline_steps: List[Dict[str, Any]] = []
    for idx, raw in enumerate(steps):
        step_id, jobspec, cmd, strict, memoize = _validate_step(raw, idx)
        if step_id in seen:
            raise ValueError(f"duplicate step_id: {step_id}")
        seen.add(step_id)

        step_dir = pipeline_dir / "steps" / step_id
        jobspec_path = step_dir / "JOBSPEC.json"
        jobspec_bytes = canonical_json_bytes(jobspec)
        _write_idempotent(jobspec_path, jobspec_bytes)

        pipeline_steps.append(
            {
                "step_id": step_id,
                "jobspec_path": _as_repo_relpath(jobspec_path),
                "cmd": cmd,
                "strict": strict,
                "memoize": memoize,
            }
        )

    pipeline_spec = {"pipeline_id": pipeline_id, "steps": pipeline_steps}
    spec_bytes = canonical_json_bytes(pipeline_spec)
    _write_idempotent(pipeline_dir / "PIPELINE.json", spec_bytes)

    sys.stdout.write(f"OK wrote {runs_root}/_pipelines/{_slug(pipeline_id)}/PIPELINE.json\n")
    return 0


def ags_run(*, pipeline_id: str, runs_root: str, strict: bool) -> int:
    if runs_root != "CONTRACTS/_runs":
        raise RuntimeError("UNSUPPORTED_RUNS_ROOT (only CONTRACTS/_runs is supported)")

    strict = True if strict else True

    catalytic = [sys.executable, str(REPO_ROOT / "TOOLS" / "catalytic.py")]

    run_cmd = catalytic + [
        "pipeline",
        "run",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        runs_root,
    ]
    verify_cmd = catalytic + [
        "pipeline",
        "verify",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        runs_root,
        "--strict",
    ]

    sys.stdout.write("AGS: running pipeline\n")
    res = subprocess.run(run_cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        sys.stdout.write(f"FAIL pipeline run rc={res.returncode}\n")
        return res.returncode

    sys.stdout.write("AGS: verifying pipeline (fail-closed)\n")
    res2 = subprocess.run(verify_cmd, cwd=str(REPO_ROOT))
    if res2.returncode != 0:
        sys.stdout.write(f"FAIL pipeline verify rc={res2.returncode}\n")
        return res2.returncode

    sys.stdout.write("OK\n")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ags", description="AGS CLI (model-free bridge)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    route_p = sub.add_parser("route", help="Emit deterministic PIPELINE.json from an explicit plan")
    route_p.add_argument("--plan", required=True, help="Path to JSON plan file")
    route_p.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    route_p.add_argument("--runs-root", default="CONTRACTS/_runs", help="Runs root (default: CONTRACTS/_runs)")

    run_p = sub.add_parser("run", help="Run and immediately verify a pipeline (fail-closed)")
    run_p.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    run_p.add_argument("--runs-root", default="CONTRACTS/_runs", help="Runs root (default: CONTRACTS/_runs)")
    run_p.add_argument("--strict", action="store_true", help="Strict verification (always on)")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "route":
            return ags_route(
                plan_path=Path(args.plan),
                pipeline_id=args.pipeline_id,
                runs_root=args.runs_root,
            )
        if args.cmd == "run":
            return ags_run(pipeline_id=args.pipeline_id, runs_root=args.runs_root, strict=bool(args.strict))
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    sys.stderr.write("ERROR: unsupported command\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
