from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_ags(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "CAPABILITY.TOOLS.ags", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _run_catalytic_verify(pipeline_id: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "TOOLS" / "catalytic.py"),
        "pipeline",
        "verify",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        "LAW/CONTRACTS/_runs",
        "--strict",
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _make_plan(*, pipeline_id: str, tmp_root: str) -> dict:
    out1 = f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_ags_step1_out.txt"
    domain = f"LAW/CONTRACTS/_runs/_tmp/{tmp_root}/domain"
    jobspec = {
        "job_id": "ags-step1",
        "phase": 6,
        "task_type": "pipeline_execution",
        "intent": "ags bridge step1",
        "inputs": {},
        "outputs": {"durable_paths": [out1], "validation_criteria": {}},
        "catalytic_domains": [domain],
        "determinism": "deterministic",
    }
    return {
        "pipeline_id": pipeline_id,
        "steps": [
            {
                "step_id": "s1",
                "jobspec": jobspec,
                "memoize": False,
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path;"
                        f"Path('{out1}').parent.mkdir(parents=True, exist_ok=True);"
                        f"Path('{out1}').write_text('ONE', encoding='utf-8')"
                    ),
                ],
            }
        ],
    }


def test_ags_route_deterministic_bytes(tmp_path: Path) -> None:
    pipeline_id = "ags-bridge-route"
    tmp_root = "ags_bridge_route"
    plan_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")

        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "plan.json"
        plan = _make_plan(pipeline_id=pipeline_id, tmp_root=tmp_root)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        r1 = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert r1.returncode == 0, r1.stdout + r1.stderr
        assert not (pipeline_dir / "STATE.json").exists()
        b1 = (pipeline_dir / "PIPELINE.json").read_bytes()

        r2 = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert r2.returncode == 0, r2.stdout + r2.stderr
        assert not (pipeline_dir / "STATE.json").exists()
        b2 = (pipeline_dir / "PIPELINE.json").read_bytes()
        assert b1 == b2
    finally:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_ags_step1_out.txt")


def test_ags_run_calls_verify_ok(tmp_path: Path) -> None:
    pipeline_id = "ags-bridge-run-ok"
    tmp_root = "ags_bridge_run_ok"
    plan_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")

        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "plan.json"
        plan = _make_plan(pipeline_id=pipeline_id, tmp_root=tmp_root)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (pipeline_dir / "STATE.json").exists()

        rr = _run_ags(["run", "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs", "--strict"])
        assert rr.returncode == 0, rr.stdout + rr.stderr
        assert (pipeline_dir / "STATE.json").exists()
    finally:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_ags_step1_out.txt")


def test_ags_run_fails_closed_on_tamper(tmp_path: Path) -> None:
    pipeline_id = "ags-bridge-run-tamper"
    tmp_root = "ags_bridge_run_tamper"
    plan_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")

        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "plan.json"
        plan = _make_plan(pipeline_id=pipeline_id, tmp_root=tmp_root)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert r.returncode == 0, r.stdout + r.stderr
        assert not (pipeline_dir / "STATE.json").exists()

        rr = _run_ags(["run", "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs", "--strict"])
        assert rr.returncode == 0, rr.stdout + rr.stderr
        assert (pipeline_dir / "STATE.json").exists()

        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id = state["step_run_ids"]["s1"]
        proof = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id / "PROOF.json"
        data = proof.read_bytes()
        proof.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))

        v = _run_catalytic_verify(pipeline_id)
        assert v.returncode != 0
    finally:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_ags_step1_out.txt")


def test_pipeline_run_creates_state_when_missing(tmp_path: Path) -> None:
    pipeline_id = "ags-bridge-runtime-init-state"
    tmp_root = "ags_bridge_runtime_init_state"
    plan_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")

        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "plan.json"
        plan = _make_plan(pipeline_id=pipeline_id, tmp_root=tmp_root)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert r.returncode == 0, r.stdout + r.stderr
        assert (pipeline_dir / "PIPELINE.json").exists()
        assert not (pipeline_dir / "STATE.json").exists()

        run_cmd = [
            sys.executable,
            str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "catalytic.py"),
            "pipeline",
            "run",
            "--pipeline-id",
            pipeline_id,
            "--runs-root",
            "LAW/CONTRACTS/_runs",
        ]
        res = subprocess.run(run_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
        assert (pipeline_dir / "STATE.json").exists()
    finally:
        _rm(plan_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_ags_step1_out.txt")
