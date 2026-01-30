# py3.8 compatibility: use postponed evaluation for builtin generics in annotations.
from __future__ import annotations

# HEAD
from pathlib import Path
import sys

# Ensure REPO_ROOT is correctly set
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import json, subprocess, shutil, os

def _run_ags(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Use absolute path to ags.py instead of -m to be safer on Windows
    ags_path = REPO_ROOT / "CAPABILITY" / "TOOLS" / "ags.py"
    cmd = [sys.executable, str(ags_path)] + args
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)

def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

def test_ags_plan_over_output_fails_closed(tmp_path: Path):
    pipeline_id = "test-phase8-receipts"
    plan_out = tmp_path / "plan.json"
    pdir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id

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
        ],
        "padding": "x" * 2000
    }
    router_code = f"import json,sys;sys.stdout.write(json.dumps({plan_obj}))"

    try:
        _rm(pdir)
        
        r = _run_ags([
            "plan",
            "--router", sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out", str(plan_out),
            "--max-bytes", "1024"
        ])
        assert r.returncode != 0
        assert "ROUTER_OUTPUT_TOO_LARGE" in r.stderr

    finally:
        _rm(pdir)

# TAIL
