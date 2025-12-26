from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def test_mcp_adapter_ant_worker_e2e_and_tamper_detected(tmp_path: Path) -> None:
    pipeline_id = "ags-mcp-adapter-ant-worker"
    step_id = "s1"
    tmp_root = "ags_mcp_adapter_ant_worker"

    # Workspace for this test's artifacts (all under allowed roots).
    work_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    src_path = work_root / "in.txt"
    dst_path = work_root / "out.txt"
    task_path = tmp_path / "task.json"
    result_path = tmp_path / "result.json"

    src_bytes = b"ant-worker deterministic adapter e2e\n"
    expected_hash = _sha256(src_bytes)

    task = {
        "task_id": "ant-worker-copy",
        "task_type": "file_operation",
        "operation": "copy",
        "verify_integrity": True,
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "files": [{"source": str(src_path), "destination": str(dst_path)}],
    }
    task_path.write_text(json.dumps(task), encoding="utf-8")

    jobspec = {
        "job_id": "ags-mcp-adapter-ant-worker-step1",
        "phase": 6,
        "task_type": "adapter_execution",
        "intent": "mcp adapter ant-worker copy",
        "inputs": {},
        "outputs": {"durable_paths": [str(dst_path.relative_to(REPO_ROOT)).replace("\\", "/")], "validation_criteria": {}},
        "catalytic_domains": [str((work_root / "domain").relative_to(REPO_ROOT)).replace("\\", "/")],
        "determinism": "deterministic",
    }

    adapter = {
        "adapter_version": "1.0.0",
        "name": "ant-worker-copy",
        "command": [
            sys.executable,
            "CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py",
            str(task_path),
            str(result_path),
        ],
        "jobspec": jobspec,
        "inputs": {str(src_path.relative_to(REPO_ROOT)).replace("\\", "/"): expected_hash},
        "outputs": {str(dst_path.relative_to(REPO_ROOT)).replace("\\", "/"): expected_hash},
        "side_effects": {"network": False, "clock": False, "filesystem_unbounded": False, "nondeterministic": False},
        "deref_caps": {"max_bytes": 1024, "max_matches": 1, "max_nodes": 10, "max_depth": 2},
        "artifacts": {"ledger": expected_hash, "proof": expected_hash, "domain_roots": expected_hash},
    }

    plan = {"plan_version": "1.0", "steps": [{"step_id": step_id, "adapter": adapter}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    run_dir = REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-{step_id}-a1"

    try:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(work_root)
        (work_root / "domain").mkdir(parents=True, exist_ok=True)
        src_path.write_bytes(src_bytes)

        r_route = _run([sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
        assert r_route.returncode == 0, r_route.stdout + r_route.stderr

        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict"])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        r_verify_ok = _run(
            [sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"]
        )
        assert r_verify_ok.returncode == 0, r_verify_ok.stdout + r_verify_ok.stderr

        # Tamper with declared durable output; pipeline verify must fail closed.
        out_bytes = dst_path.read_bytes()
        assert out_bytes == src_bytes
        dst_path.write_bytes(out_bytes[:-1] + bytes([out_bytes[-1] ^ 0x01]))

        r_verify_bad = _run(
            [sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"]
        )
        assert r_verify_bad.returncode != 0
        assert "FAIL" in r_verify_bad.stdout
        assert "OUTPUT_HASH_MISMATCH" in r_verify_bad.stdout
    finally:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(work_root)
