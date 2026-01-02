from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path
import pytest



REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

def _run_ags(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(env)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Use -m CAPABILITY.TOOLS.ags to ensure correct module resolution
    cmd = [sys.executable, "-m", "CAPABILITY.TOOLS.ags"] + args
    return subprocess.run(cmd, cwd=str(REPO_ROOT / "CAPABILITY"), capture_output=True, text=True, env=env)

def _canon(obj: object) -> bytes:
    # Use standard canonical JSON (SPECTRUM-04)
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def test_registry_tamper_detected_fail_closed(tmp_path: Path) -> None:
    # Valid adapter
    adapter = {
        "command": ["echo", "ok"],
        "jobspec": {
            "job_id": "j1", "intent": "test", "phase": 6, "task_type": "test",
            "inputs": {}, "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"], "determinism": "deterministic"
        },
        "inputs": {}, "outputs": {}, "side_effects": [], "deref_caps": [], "artifacts": []
    }
    cap = hashlib.sha256(_canon(adapter)).hexdigest()

    # Tamper: change one byte in the adapter but keep the same key (hash) in registry
    tampered_adapter = adapter.copy()
    tampered_adapter["command"] = ["echo", "tampered"]

    # Create registry with correct hash but wrong adapter
    reg_path = tmp_path / "CAPABILITY" / "CAPABILITIES.json"
    obj = {
        "registry_version": "1.0.0",
        "capabilities": {
            cap: {
                "adapter": tampered_adapter,
                "adapter_spec_hash": cap
            }
        }
    }
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_bytes(_canon(obj))

    pins_path = tmp_path / "CAPABILITY" / "PINS.json"
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)
    env.pop("CATALYTIC_REVOKES_PATH", None)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "CAPABILITY" / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", "reg-tamper"], env=env)
    assert r.returncode != 0
    assert "CAPABILITY_HASH_MISMATCH" in (r.stderr + r.stdout)

