from __future__ import annotations

import os
from pathlib import Path
import sys
import pytest
import shutil
import json
import subprocess
import hashlib
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

def _canon(obj: Any) -> bytes:
    """SPECTRUM-04 v1.1.0 Canonical JSON."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')

def _rm(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)

def _mock_registry(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """Create mock skill registry and capability registry and return (capability_hash, env)"""

    # Valid jobspec for a skill
    jobspec = {
        "job_id": "test-job",
        "intent": "test",
        "phase": 6,
        "task_type": "test_execution",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": [],
        "determinism": "deterministic"
    }

    # Valid adapter conforming to adapter.schema.json
    dummy_hash = "0" * 64
    adapter = {
        "adapter_version": "1.0.0",
        "name": "test-adapter",
        "command": ["python", "--version"],
        "jobspec": jobspec,
        "inputs": {},
        "outputs": {},
        "side_effects": {
            "network": False,
            "clock": False,
            "filesystem_unbounded": False,
            "nondeterministic": False
        },
        "deref_caps": {
            "max_bytes": 65536,
            "max_matches": 20,
            "max_nodes": 2000,
            "max_depth": 32
        },
        "artifacts": {
            "ledger": dummy_hash,
            "proof": dummy_hash,
            "domain_roots": dummy_hash
        }
    }

    cap_hash = hashlib.sha256(_canon(adapter)).hexdigest()

    # Capabilities registry
    caps_obj = {
        "registry_version": "1.0.0",
        "capabilities": {
            cap_hash: {
                "adapter": adapter,
                "adapter_spec_hash": cap_hash
            }
        }
    }

    caps_path = tmp_path / "capabilities.json"
    pins_path = tmp_path / "pins.json"
    revokes_path = tmp_path / "revokes.json"

    with open(caps_path, 'wb') as f:
        f.write(_canon(caps_obj))
    # Write empty pins and revokes initially
    with open(pins_path, 'wb') as f:
        f.write(_canon({"pins_version": "1.0.0", "allowed_capabilities": []}))
    with open(revokes_path, 'wb') as f:
        f.write(_canon({"revokes_version": "1.0.0", "revoked_capabilities": []}))

    env = dict(os.environ)
    env.pop("CATALYTIC_SKILLS_PATH", None) # Ensure we don't accidentally use real skills
    env["CATALYTIC_CAPABILITIES_PATH"] = str(caps_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)
    env["CATALYTIC_REVOKES_PATH"] = str(revokes_path)

    return cap_hash, env

def test_capability_pinned_routes_and_verifies(tmp_path: Path) -> None:
    pipeline_id = "ags-pins-ok"
    cap, env = _mock_registry(tmp_path)

    plan_path = tmp_path / "plan.json"
    with open(plan_path, 'w', encoding="utf-8") as f:
        json.dump({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}, f)

    # Pin the capability
    with open(env["CATALYTIC_PINS_PATH"], 'wb') as f:
        f.write(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    # Route
    r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route.returncode == 0, f"Route failed: {r_route.stderr}"

    # Verify
    r_verify = _run([sys.executable, "-m", "CAPABILITY.TOOLS.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
    assert r_verify.returncode != 0
    assert "CHAIN_MISSING" in r_verify.stdout

def test_verify_rejects_unpinned(tmp_path: Path) -> None:
    pipeline_id = "ags-pins-unpinned"
    cap, env = _mock_registry(tmp_path)

    plan_path = tmp_path / "plan.json"
    with open(plan_path, 'w', encoding="utf-8") as f:
        json.dump({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}, f)

    # Route
    r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route.returncode != 0
    assert "CAPABILITY_NOT_PINNED" in r_route.stderr

def test_verify_rejects_revoked(tmp_path: Path) -> None:
    pipeline_id = "ags-pins-revoked"
    cap, env = _mock_registry(tmp_path)

    plan_path = tmp_path / "plan.json"
    with open(plan_path, 'w', encoding="utf-8") as f:
        json.dump({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}, f)

    # Pin it
    with open(env["CATALYTIC_PINS_PATH"], 'wb') as f:
        f.write(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    # Revoke it
    with open(env["CATALYTIC_REVOKES_PATH"], 'wb') as f:
        f.write(_canon({"revokes_version": "1.0.0", "revoked_capabilities": [cap]}))

    r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route.returncode != 0
    assert "REVOKED_CAPABILITY" in r_route.stderr

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
