from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CAP = "4f81ae57f3d1c61488c71a9042b041776dd463e6334568333321d15b6b7d78fc"


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


def _prepare_ant_worker_inputs(*, label: str, reg_root: Path) -> None:
    task_path = reg_root / "task.json"
    in_path = reg_root / "in.txt"
    out_path = reg_root / "out.txt"
    _rm(reg_root)
    (reg_root / "domain").mkdir(parents=True, exist_ok=True)
    in_path.write_bytes(label.encode("utf-8"))
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


def test_revoked_capability_rejects_at_route(tmp_path: Path) -> None:
    unique_suffix = hex(hash(str(tmp_path)))[-8:]
    pipeline_id = f"ags-capability-revokes-route-reject-{unique_suffix}"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")

    revokes_path = tmp_path / "REVOKES.json"
    revokes_path.write_bytes(_canon({"revokes_version": "1.0.0", "revoked_capabilities": [CAP]}))

    env = dict(os.environ)
    env["CATALYTIC_REVOKES_PATH"] = str(revokes_path)

    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(pipeline_dir)
        r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route.returncode != 0
        assert "REVOKED_CAPABILITY" in (r_route.stdout + r_route.stderr)
    finally:
        _rm(pipeline_dir)


def test_historical_pipeline_verifies_after_revocation(tmp_path: Path) -> None:
    unique_suffix = hex(hash(str(tmp_path)))[-8:]
    pipeline_id = f"ags-capability-revokes-historical-ok-{unique_suffix}"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")

    empty_revokes = tmp_path / "REVOKES_EMPTY.json"
    empty_revokes.write_bytes(_canon({"revokes_version": "1.0.0", "revoked_capabilities": []}))
    revoked_revokes = tmp_path / "REVOKES_REVOKED.json"
    revoked_revokes.write_bytes(_canon({"revokes_version": "1.0.0", "revoked_capabilities": [CAP]}))

    env_route = dict(os.environ)
    env_route["CATALYTIC_REVOKES_PATH"] = str(empty_revokes)

    env_verify = dict(os.environ)
    env_verify["CATALYTIC_REVOKES_PATH"] = str(revoked_revokes)

    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    runs_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs"
    reg_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / f"phase65_registry_{unique_suffix}"

    try:
        # Clean up pipeline directory first
        _rm(pipeline_dir)

        # Clean up ALL run directories matching this pipeline (any attempt number)
        if runs_root.exists():
            for item in runs_root.iterdir():
                if item.is_dir() and item.name.startswith(f"pipeline-{pipeline_id}-"):
                    _rm(item)

        # Clean up registry root BEFORE preparing new inputs
        _rm(reg_root)

        # Now prepare fresh ant worker inputs
        _prepare_ant_worker_inputs(label="PHASE69-HISTORICAL\n", reg_root=reg_root)

        r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env_route)
        assert r_route.returncode == 0, r_route.stdout + r_route.stderr

        r_run = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict"], env=env_route)
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        # After revocation, historical pipeline verification must still pass.
        r_verify = _run([sys.executable, "CAPABILITY/TOOLS/catalytic/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env_verify)
        assert r_verify.returncode == 0, r_verify.stdout + r_verify.stderr
    finally:
        # Clean up pipeline directory
        _rm(pipeline_dir)

        # Clean up ALL run directories matching this pipeline
        if runs_root.exists():
            for item in runs_root.iterdir():
                if item.is_dir() and item.name.startswith(f"pipeline-{pipeline_id}-"):
                    _rm(item)

        # Clean up registry root
        _rm(reg_root)


def test_post_revocation_pipeline_verify_fails_closed(tmp_path: Path) -> None:
    unique_suffix = hex(hash(str(tmp_path)))[-8:]
    pipeline_id = f"ags-capability-revokes-post-reject-{unique_suffix}"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}), encoding="utf-8")

    empty_revokes = tmp_path / "REVOKES_EMPTY.json"
    empty_revokes.write_bytes(_canon({"revokes_version": "1.0.0", "revoked_capabilities": []}))
    revoked_revokes = tmp_path / "REVOKES_REVOKED.json"
    revoked_revokes.write_bytes(_canon({"revokes_version": "1.0.0", "revoked_capabilities": [CAP]}))

    env_route = dict(os.environ)
    env_route["CATALYTIC_REVOKES_PATH"] = str(empty_revokes)

    env_run_verify = dict(os.environ)
    env_run_verify["CATALYTIC_REVOKES_PATH"] = str(revoked_revokes)

    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    runs_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs"
    reg_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / f"phase65_registry_{unique_suffix}"

    try:
        # Clean up pipeline directory first
        _rm(pipeline_dir)

        # Clean up ALL run directories matching this pipeline (any attempt number)
        if runs_root.exists():
            for item in runs_root.iterdir():
                if item.is_dir() and item.name.startswith(f"pipeline-{pipeline_id}-"):
                    _rm(item)

        # Clean up registry root BEFORE preparing new inputs
        _rm(reg_root)

        # Now prepare fresh ant worker inputs
        _prepare_ant_worker_inputs(label="PHASE69-POST\n", reg_root=reg_root)

        # Route before revocation (allowed).
        r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env_route)
        assert r_route.returncode == 0, r_route.stdout + r_route.stderr

        # Execute + verify after revocation: pipeline verify must fail closed based on POLICY snapshot.
        r_run = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict"], env=env_run_verify)
        assert r_run.returncode != 0
        assert "REVOKED_CAPABILITY" in (r_run.stdout + r_run.stderr)
    finally:
        # Clean up pipeline directory
        _rm(pipeline_dir)

        # Clean up ALL run directories matching this pipeline
        if runs_root.exists():
            for item in runs_root.iterdir():
                if item.is_dir() and item.name.startswith(f"pipeline-{pipeline_id}-"):
                    _rm(item)

        # Clean up registry root
        _rm(reg_root)

