from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAP = "e8e7e5234b43278a1a257b9257186b8bca5fdae9ab9096572942da1c5fb90f36"


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


def test_capability_pinned_routes_and_verifies(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-ok"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [CAP]}))

    env = dict(os.environ)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    # The v1 capability is pinned to a concrete ant-worker command that expects these files.
    reg_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase65_registry"
    task_path = reg_root / "task.json"
    in_path = reg_root / "in.txt"
    out_path = reg_root / "out.txt"
    (reg_root / "domain").mkdir(parents=True, exist_ok=True)
    in_path.write_bytes(b"PHASE66\n")
    task_path.write_text(
        json.dumps(
            {
                "task_id": "ant-worker-copy",
                "task_type": "file_operation",
                "operation": "copy",
                "verify_integrity": True,
                "timestamp": "CATALYTIC-DPT-02_CONFIG",
                "files": [{"source": str(in_path), "destination": str(out_path)}],
            }
        ),
        encoding="utf-8",
    )

    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    run_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "pipeline-ags-capability-pins-ok-s1-a1"

    try:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)
        (reg_root / "domain").mkdir(parents=True, exist_ok=True)
        in_path.write_bytes(b"PHASE66\n")
        task_path.write_text(
            json.dumps(
                {
                    "task_id": "ant-worker-copy",
                    "task_type": "file_operation",
                    "operation": "copy",
                    "verify_integrity": True,
                    "timestamp": "CATALYTIC-DPT-02_CONFIG",
                    "files": [{"source": str(in_path), "destination": str(out_path)}],
                }
            ),
            encoding="utf-8",
        )
    
        r_route = _run([sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route.returncode == 0, r_route.stdout + r_route.stderr
    
        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"], env=env)
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr
    
        r_verify = _run([sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        assert r_verify.returncode == 0, r_verify.stdout + r_verify.stderr

    finally:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)


def test_known_but_unpinned_rejects_at_route(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-unpinned"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": []}))

    env = dict(os.environ)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(pipeline_dir)
        r_route = _run([sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route.returncode != 0
        assert "CAPABILITY_NOT_PINNED" in (r_route.stdout + r_route.stderr)
    finally:
        _rm(pipeline_dir)


def test_verify_rejects_unpinned_even_if_pipeline_artifact_exists(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-verify-bypass"
    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    run_id = "pipeline-ags-capability-pins-verify-bypass-s1-a1"
    run_dir = REPO_ROOT / "CONTRACTS" / "_runs" / run_id

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": []}))

    env = dict(os.environ)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    reg_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase65_registry"
    task_path = reg_root / "task.json"
    in_path = reg_root / "in.txt"
    out_path = reg_root / "out.txt"

    try:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)
        (reg_root / "domain").mkdir(parents=True, exist_ok=True)
        in_path.write_bytes(b"PHASE66\n")
        task_path.write_text(
            json.dumps(
                {
                    "task_id": "ant-worker-copy",
                    "task_type": "file_operation",
                    "operation": "copy",
                    "verify_integrity": True,
                    "timestamp": "CATALYTIC-DPT-02_CONFIG",
                    "files": [{"source": str(in_path), "destination": str(out_path)}],
                }
            ),
            encoding="utf-8",
        )

        # Create a real run with the pinned capability (using default pins).
        plan_ok = tmp_path / "plan_ok.json"
        plan_ok.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")
        r_route_ok = subprocess.run(
            [sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_ok), "--pipeline-id", pipeline_id],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert r_route_ok.returncode == 0, r_route_ok.stdout + r_route_ok.stderr
        r_run_ok = subprocess.run(
            [sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--allow-dirty-tracked"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert r_run_ok.returncode == 0, r_run_ok.stdout + r_run_ok.stderr

        # Now enforce an empty pin set at verify time.
        r_verify = _run([sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        assert r_verify.returncode != 0
        assert "CAPABILITY_NOT_PINNED" in r_verify.stdout
    finally:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)
