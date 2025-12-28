from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_ags(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "TOOLS.ags", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _valid_jobspec(*, tmp_root: str) -> dict:
    return {
        "job_id": "ags-router-step1",
        "phase": 6,
        "task_type": "pipeline_execution",
        "intent": "router produced step",
        "inputs": {},
        "outputs": {"durable_paths": [f"CORTEX/_generated/_tmp/{tmp_root}_noop.txt"], "validation_criteria": {}},
        "catalytic_domains": [f"CONTRACTS/_runs/_tmp/{tmp_root}/domain"],
        "determinism": "deterministic",
    }


def test_ags_plan_router_happy_path(tmp_path: Path) -> None:
    pipeline_id = "ags-router-happy"
    tmp_root = "ags_router_happy"
    plan_out = tmp_path / "plan.json"

    plan_obj = {
        "plan_version": "1.0",
        "pipeline_id": "ignored-by-override",
        "steps": [{"step_id": "s1", "command": [sys.executable, "-c", "pass"], "jobspec": _valid_jobspec(tmp_root=tmp_root)}],
    }
    router_code = "import json,sys;sys.stdout.write(json.dumps(%s))" % json.dumps(plan_obj)
    r1 = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            pipeline_id,
        ]
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr
    b1 = plan_out.read_bytes()

    r2 = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            pipeline_id,
        ]
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    b2 = plan_out.read_bytes()
    assert b1 == b2

    # Route + run should succeed using the validated plan output.
    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"CORTEX/_generated/_tmp/{tmp_root}_noop.txt")

        rr = _run_ags(["route", "--plan", str(plan_out), "--pipeline-id", pipeline_id, "--runs-root", "CONTRACTS/_runs"])
        assert rr.returncode == 0, rr.stdout + rr.stderr
        run = _run_ags(["run", "--pipeline-id", pipeline_id, "--runs-root", "CONTRACTS/_runs", "--strict"])
        assert run.returncode == 0, run.stdout + run.stderr
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"CORTEX/_generated/_tmp/{tmp_root}_noop.txt")


def test_ags_plan_fails_on_stderr(tmp_path: Path) -> None:
    plan_out = tmp_path / "plan.json"
    code = "import sys;sys.stdout.write('{\"plan_version\":\"1.0\",\"steps\":[]}');sys.stderr.write('x')"
    r = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            "p",
        ]
    )
    assert r.returncode != 0


def test_ags_plan_caps_bytes(tmp_path: Path) -> None:
    plan_out = tmp_path / "plan.json"
    code = "import sys;sys.stdout.buffer.write(b'a'*200)"
    r = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            "p",
            "--max-bytes",
            "100",
        ]
    )
    assert r.returncode != 0


def test_ags_route_rejects_invalid_plan_schema(tmp_path: Path) -> None:
    pipeline_id = "ags-router-invalid-plan"
    plan_path = tmp_path / "bad_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "plan_version": "1.0",
                "steps": [
                    {"step_id": "s1", "command": ["python3", "-c", "pass"], "jobspec": {}},
                    {"step_id": "s1", "command": ["python3", "-c", "pass"], "jobspec": {}},
                ],
            }
        )
    )
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "CONTRACTS/_runs"])
    assert r.returncode != 0


def test_ags_plan_jobspec_validation(tmp_path: Path) -> None:
    plan_out = tmp_path / "plan.json"
    bad = {"plan_version": "1.0", "steps": [{"step_id": "s1", "command": [sys.executable, "-c", "pass"], "jobspec": {"task_type": "nope"}}]}
    code = "import json,sys;sys.stdout.write(json.dumps(%s))" % json.dumps(bad)
    r = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            "p",
        ]
    )
    assert r.returncode != 0


def test_ags_route_rejects_missing_step_command(tmp_path: Path) -> None:
    pipeline_id = "ags-router-missing-command"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"steps": [{"step_id": "s1", "jobspec": _valid_jobspec(tmp_root="ags_router_missing_cmd")}]}),
        encoding="utf-8",
    )
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "CONTRACTS/_runs"])
    assert r.returncode != 0
    assert "MISSING_STEP_COMMAND" in r.stderr
