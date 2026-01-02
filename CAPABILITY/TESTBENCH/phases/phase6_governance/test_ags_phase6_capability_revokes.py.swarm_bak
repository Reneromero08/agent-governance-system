import json
import os
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Import canonical_json_bytes if available, otherwise fallback
try:
    from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes as _canon
except ImportError:
    def _canon(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _run(cmd, env=None):
    if env is None:
        env = dict(os.environ)
    else:
        env = dict(env)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)

def _mock_registry(tmp_path: Path) -> tuple[str, dict[str, str]]:
    # Valid adapter conforming to adapter.schema.json
    adapter = {
        "command": ["echo", "hello"],
        "jobspec": {
            "job_id": "test-job",
            "intent": "test revocation",
            "phase": 6,
            "task_type": "governance_test",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"],
            "determinism": "deterministic"
        },
        "inputs": {},
        "outputs": {},
        "side_effects": [],
        "deref_caps": [],
        "artifacts": []
    }

    cap_hash = hashlib.sha256(_canon(adapter)).hexdigest()

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

    caps_path.write_bytes(_canon(caps_obj))
    pins_path.write_bytes(_canon({
        "pins_version": "1.0.0",
        "allowed_capabilities": [cap_hash]
    }))
    revokes_path.write_bytes(_canon({
        "revokes_version": "1.0.0",
        "revoked_capabilities": []
    }))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(caps_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)
    env["CATALYTIC_REVOKES_PATH"] = str(revokes_path)
    env.pop("CATALYTIC_SKILLS_PATH", None)

    return cap_hash, env

def test_revoked_capability_rejects_at_route(tmp_path: Path) -> None:
    cap, env = _mock_registry(tmp_path)

    # Revoke it
    revokes_content = {
        "revokes_version": "1.0.0",
        "revoked_capabilities": [cap]
    }
    with open(env["CATALYTIC_REVOKES_PATH"], 'wb') as f:
        f.write(_canon(revokes_content))

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    from CAPABILITY.PIPELINES import ags
    r = _run([sys.executable, "-m", "CAPABILITY.PIPELINES.ags", "route", "--plan", str(plan_path), "--pipeline-id", "test-revoke-reject"], env=env)

    assert r.returncode != 0
    assert "REVOKED_CAPABILITY" in (r.stderr + r.stdout)

def test_verify_rejects_revoked_capability(tmp_path: Path) -> None:
    pipeline_id = "test-verify-revoke"
    cap, env = _mock_registry(tmp_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    # 1. Route while NOT revoked
    from CAPABILITY.PIPELINES import ags
    r_route = _run([sys.executable, "-m", "CAPABILITY.PIPELINES.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route.returncode == 0

    # 2. Revoke it NOW
    revokes_content = {
        "revokes_version": "1.0.0",
        "revoked_capabilities": [cap]
    }
    with open(env["CATALYTIC_REVOKES_PATH"], 'wb') as f:
        f.write(_canon(revokes_content))

    # 3. Verify should FAIL because revocation is checked during verification too
    from CAPABILITY.PIPELINES import catalytic
    r_verify = _run([sys.executable, "-m", "CAPABILITY.PIPELINES.catalytic", "verify", "--pipeline-id", pipeline_id], env=env)

    assert r_verify.returncode != 0
    assert "REVOKED_CAPABILITY" in (r_verify.stdout + r_verify.stderr)