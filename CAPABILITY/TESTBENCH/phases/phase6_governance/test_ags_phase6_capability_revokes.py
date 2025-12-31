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
        "adapter_version": "1.0.0",
        "name": "test-adapter",
        "command": [sys.executable, "-c", "print('hello')"],
        "jobspec": {
            "job_id": "test-job",
            "intent": "test revocation",
            "phase": 6,
            "task_type": "test_execution",
            "inputs": {},
            "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/_tmp/out.txt"], "validation_criteria": {}},
            "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"],
            "determinism": "deterministic"
        },
        "inputs": {},
        "outputs": {
            "LAW/CONTRACTS/_runs/_tmp/out.txt": "0"*64
        },
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
            "ledger": "0" * 64,
            "proof": "0" * 64,
            "domain_roots": "0" * 64
        }
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

    r = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", "test-revoke-reject"], env=env)

    assert r.returncode != 0
    assert "REVOKED_CAPABILITY" in (r.stderr + r.stdout)

def test_verify_rejects_revoked_capability(tmp_path: Path) -> None:
    pipeline_id = "test-verify-revoke-v2"
    
    # Cleanup previous run to avoid REFUSE_OVERWRITE
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir)

    cap, env = _mock_registry(tmp_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    # 1. Route while NOT revoked
    # 1. Route while NOT revoked
    r_route = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route.returncode == 0

    # 1.5 Run to establish chain artifacts
    r_run = _run([sys.executable, "-m", "CAPABILITY.TOOLS.catalytic", "pipeline", "run", "--pipeline-id", pipeline_id], env=env)
    assert r_run.returncode == 0

    # 2. Revoke it NOW
    revokes_content = {
        "revokes_version": "1.0.0",
        "revoked_capabilities": [cap]
    }
    with open(env["CATALYTIC_REVOKES_PATH"], 'wb') as f:
        f.write(_canon(revokes_content))

    # 3. Verify should FAIL because revocation is checked during verification too
    r_verify = _run([sys.executable, "-m", "CAPABILITY.TOOLS.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id], env=env)

    assert r_verify.returncode != 0
    assert "REVOKED_CAPABILITY" in (r_verify.stdout + r_verify.stderr)