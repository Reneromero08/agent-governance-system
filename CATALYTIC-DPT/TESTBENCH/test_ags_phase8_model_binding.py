from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

def _run_ags(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "TOOLS.ags"] + args
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

def test_ags_plan_emits_router_receipts(tmp_path: Path):
    pipeline_id = "test-phase8-receipts"
    plan_out = tmp_path / "plan.json"
    pdir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id

    # Dummy plan output from a router
    plan_obj = {
        "plan_version": "1.0",
        "steps": [
            {
                "step_id": "s1",
                "command": ["echo", "hello"],
                "jobspec": {
                    "job_id": "job1",
                    "intent": "test",
                    "phase": 8,
                    "task_type": "test_execution",
                    "inputs": {},
                    "outputs": {"durable_paths": [], "validation_criteria": {}},
                    "catalytic_domains": [],
                    "determinism": "deterministic"
                }
            }
        ]
    }
    router_code = "import json,sys;sys.stdout.write(json.dumps(%s))" % json.dumps(plan_obj)

    try:
        _rm(pdir)
        
        r = _run_ags([
            "plan",
            "--router", sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out", str(plan_out),
            "--pipeline-id", pipeline_id
        ])
        assert r.returncode == 0, r.stdout + r.stderr
        
        # 1. Check plan.json has embedded router metadata
        plan = json.loads(plan_out.read_text(encoding="utf-8"))
        assert "router" in plan
        assert plan["router"]["router_executable"] == sys.executable
        assert plan["router"]["transcript_hash"] is not None
        
        # 2. Check pipeline directory artifacts
        assert (pdir / "ROUTER.json").exists()
        assert (pdir / "ROUTER_OUTPUT.json").exists()
        assert (pdir / "ROUTER_TRANSCRIPT_HASH").exists()
        
        # 3. Verify transcript hash matches plan content
        stored_hash = (pdir / "ROUTER_TRANSCRIPT_HASH").read_text(encoding="utf-8").strip()
        # The router wrote the JSON of plan_obj
        raw_output = json.dumps(plan_obj).encode("utf-8")
        expected_hash = hashlib.sha256(raw_output).hexdigest()
        assert stored_hash == expected_hash
        
    finally:
        _rm(pdir)

def test_ags_plan_over_output_fails_closed(tmp_path: Path):
    plan_out = tmp_path / "plan.json"
    # Write a huge amount of data
    large_output = "x" * 1000
    router_code = "import sys;sys.stdout.write('%s')" % large_output
    
    r = _run_ags([
        "plan",
        "--router", sys.executable,
        "--router-arg=-c",
        f"--router-arg={router_code}",
        "--out", str(plan_out),
        "--max-bytes", "100"
    ])
    assert r.returncode != 0
    assert "ROUTER_OUTPUT_TOO_LARGE" in r.stderr