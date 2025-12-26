#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_runtime import _slug  # type: ignore
from PRIMITIVES.restore_proof import canonical_json_bytes  # type: ignore
from PRIMITIVES.preflight import PreflightValidator  # type: ignore

from jsonschema import Draft7Validator


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


MAX_PLAN_BYTES_DEFAULT = 262_144
MAX_STEPS = 64
MAX_JOBSPEC_BYTES = 65_536


def _read_bytes_bounded(path: Path, max_bytes: int) -> bytes:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"plan exceeds max bytes: {len(data)} > {max_bytes}")
    return data


def _load_plan(path: Path, *, max_bytes: int = MAX_PLAN_BYTES_DEFAULT) -> Dict[str, Any]:
    raw = _read_bytes_bounded(path, max_bytes)
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("plan must not start with BOM")
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("plan must be a JSON object")
    return obj


def _reject_control_chars(s: str) -> None:
    for ch in s:
        o = ord(ch)
        if o < 32 or o == 127:
            raise ValueError("step_id contains control characters")


def _validate_jobspec_obj(jobspec: Dict[str, Any]) -> None:
    jobspec_bytes = canonical_json_bytes(jobspec)
    if len(jobspec_bytes) > MAX_JOBSPEC_BYTES:
        raise ValueError(f"jobspec exceeds max bytes: {len(jobspec_bytes)} > {MAX_JOBSPEC_BYTES}")
    schema_path = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "jobspec.schema.json"
    preflight = PreflightValidator(schema_path)
    valid, errors = preflight.validate(jobspec, REPO_ROOT)
    if not valid:
        first = errors[0] if errors else {"code": "JOBSPEC_INVALID", "message": "invalid"}
        raise ValueError(f"jobspec invalid: {first.get('code')}: {first.get('message')}")


def _load_ags_plan_schema() -> Draft7Validator:
    schema_path = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "ags_plan.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


_AGS_PLAN_VALIDATOR = _load_ags_plan_schema()


def _validate_plan_schema(obj: Dict[str, Any]) -> None:
    errors = sorted(_AGS_PLAN_VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise ValueError(f"plan schema invalid at {list(first.path)}: {first.message}")


def _parse_step_for_route(step: Any, idx: int) -> Tuple[str, Dict[str, Any], List[str], bool, bool]:
    if not isinstance(step, dict):
        raise ValueError(f"steps[{idx}] must be an object")
    step_id = step.get("step_id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ValueError(f"steps[{idx}].step_id must be non-empty string")
    _reject_control_chars(step_id)
    jobspec = step.get("jobspec")
    if not isinstance(jobspec, dict):
        raise ValueError(f"steps[{idx}].jobspec must be an object")
    _validate_jobspec_obj(jobspec)
    cmd = step.get("command", step.get("cmd"))
    if cmd is None:
        raise ValueError("MISSING_STEP_COMMAND")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        raise ValueError(f"steps[{idx}].command must be a non-empty list[str]")
    strict = step.get("strict", True)
    memoize = step.get("memoize", True)
    if not isinstance(strict, bool):
        raise ValueError(f"steps[{idx}].strict must be boolean")
    if not isinstance(memoize, bool):
        raise ValueError(f"steps[{idx}].memoize must be boolean")
    return step_id, jobspec, cmd, strict, memoize


def _validate_and_extract_steps_for_route(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "plan_version" in plan:
        _validate_plan_schema(plan)
        steps = plan.get("steps")
        assert isinstance(steps, list)
        if len(steps) > MAX_STEPS:
            raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")
        return steps

    # Legacy Phase 6.1 plan format (no plan_version). Keep for backward compatibility.
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan.steps must be a non-empty list")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")
    return steps


def ags_route(*, plan_path: Path, pipeline_id: str, runs_root: str) -> int:
    if runs_root != "CONTRACTS/_runs":
        raise RuntimeError("UNSUPPORTED_RUNS_ROOT (only CONTRACTS/_runs is supported)")

    plan = _load_plan(plan_path, max_bytes=MAX_PLAN_BYTES_DEFAULT)
    steps = _validate_and_extract_steps_for_route(plan)

    seen: set[str] = set()
    pipeline_dir = REPO_ROOT / runs_root / "_pipelines" / _slug(pipeline_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    pipeline_steps: List[Dict[str, Any]] = []
    for idx, raw in enumerate(steps):
        step_id, jobspec, cmd, strict, memoize = _parse_step_for_route(raw, idx)
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


def ags_plan(
    *,
    router: str,
    router_args: List[str],
    out_path: Path,
    pipeline_id: Optional[str],
    max_bytes: int,
) -> int:
    env = dict(os.environ)
    if pipeline_id is not None:
        env["AGS_PIPELINE_ID"] = pipeline_id

    proc = subprocess.run(
        [router, *router_args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
    )
    if proc.stderr not in (b"", None):
        raise RuntimeError("ROUTER_STDERR_NOT_EMPTY")
    if proc.returncode != 0:
        raise RuntimeError(f"ROUTER_EXIT_{proc.returncode}")

    out = proc.stdout or b""
    if len(out) > max_bytes:
        raise RuntimeError(f"ROUTER_OUTPUT_TOO_LARGE: {len(out)} > {max_bytes}")

    text = out.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("router output must not start with BOM")
    plan_obj = json.loads(text)
    if not isinstance(plan_obj, dict):
        raise ValueError("router output must be a JSON object")

    _validate_plan_schema(plan_obj)
    steps = plan_obj.get("steps")
    assert isinstance(steps, list)
    if len(steps) > MAX_STEPS:
        raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")

    seen: set[str] = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{idx}] must be an object")
        sid = step.get("step_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"steps[{idx}].step_id must be non-empty string")
        _reject_control_chars(sid)
        if sid in seen:
            raise ValueError(f"duplicate step_id: {sid}")
        seen.add(sid)
        cmd = step.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
            raise ValueError("MISSING_STEP_COMMAND")
        jobspec = step.get("jobspec")
        if not isinstance(jobspec, dict):
            raise ValueError(f"steps[{idx}].jobspec must be an object")
        _validate_jobspec_obj(jobspec)

    if pipeline_id is not None:
        plan_obj["pipeline_id"] = pipeline_id

    out_bytes = canonical_json_bytes(plan_obj)
    if len(out_bytes) > MAX_PLAN_BYTES_DEFAULT:
        raise RuntimeError(f"PLAN_BYTES_TOO_LARGE_AFTER_CANON: {len(out_bytes)} > {MAX_PLAN_BYTES_DEFAULT}")

    _write_idempotent(out_path, out_bytes)
    sys.stdout.write(f"OK wrote {out_path}\n")
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

    plan_p = sub.add_parser("plan", help="Run router to produce a validated plan JSON (fail-closed)")
    plan_p.add_argument("--router", required=True, help="Router executable path")
    plan_p.add_argument("--router-arg", action="append", default=[], help="Router argument (repeatable)")
    plan_p.add_argument("--out", required=True, help="Output plan.json path (repo-relative recommended)")
    plan_p.add_argument("--pipeline-id", default=None, help="Override pipeline_id")
    plan_p.add_argument("--max-bytes", type=int, default=MAX_PLAN_BYTES_DEFAULT, help="Max router stdout bytes")

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
        if args.cmd == "plan":
            return ags_plan(
                router=args.router,
                router_args=list(args.router_arg),
                out_path=Path(args.out),
                pipeline_id=args.pipeline_id,
                max_bytes=int(args.max_bytes),
            )
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
