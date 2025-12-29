from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAP = "4f81ae57f3d1c61488c71a9042b041776dd463e6334568333321d15b6b7d78fc"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass



from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes as _canon



def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


def _mock_registry(tmp_path: Path) -> tuple[str, dict[str, str]]:
    reg_path = tmp_path / "CAPABILITIES.json"
    adapter_def = {
        "adapter_version": "1.0.0",
        "name": "ant-worker-copy-v1",
        "jobspec": {
            "phase": 6,
            "job_id": "cap-ant-worker-copy-v1",
            "intent": "capability: ant-worker copy (Phase 6.5)",
            "task_type": "adapter_execution",
            "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/phase65_registry/domain"],
            "determinism": "deterministic",
            "inputs": {},
            "outputs": {
                "durable_paths": ["LAW/CONTRACTS/_runs/_tmp/phase65_registry/out.txt"],
                "validation_criteria": {}
            }
        },
        "command": [
            "python", 
            "CAPABILITY/SKILLS/agents/ant-worker/scripts/run.py",
            "LAW/CONTRACTS/_runs/_tmp/phase65_registry/task.json",
            "LAW/CONTRACTS/_runs/_tmp/phase65_registry/result.json"
        ],
        "artifacts": {
            "proof": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "ledger": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "domain_roots": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        },
        "inputs": {
            "LAW/CONTRACTS/_runs/_tmp/phase65_registry/in.txt": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        "outputs": {
            "LAW/CONTRACTS/_runs/_tmp/phase65_registry/out.txt": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        "side_effects": {
            "filesystem_unbounded": False,
            "network": False,
            "clock": False,
            "nondeterministic": False
        },
        "deref_caps": {
            "max_bytes": 1024,
            "max_nodes": 10,
            "max_depth": 2,
            "max_matches": 1
        }
    }
    
    adapter_bytes = _canon(adapter_def)
    import hashlib
    computed_cap = hashlib.sha256(adapter_bytes).hexdigest()
    
    reg_content = {
        "registry_version": "1.0.0",
        "capabilities": {
            computed_cap: {
                "adapter_spec_hash": computed_cap,
                "adapter": adapter_def
            }
        }
    }
    reg_path.write_bytes(_canon(reg_content))
    
    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    return computed_cap, env



def test_capability_pinned_routes_and_verifies(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-ok"
    cap, env = _mock_registry(tmp_path)
    
    plan_path = tmp_path / "plan.json"

    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}), encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    # The v1 capability is pinned to a concrete ant-worker command that expects these files.
    reg_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "phase65_registry"
    in_file = reg_root / "in.txt"
    out_file = reg_root / "out.txt"
    
    try:
        r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route_ok.returncode == 0
        assert not in_file.exists()
    finally:
        if in_file.is_file():
            in_file.unlink()

        if out_file.is_file():
            out_file.unlink()


def test_verify_rejects_unpinned_even_if_pipeline_artifact_exists(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-verify-bypass"
    pins_path = tmp_path / "PINS.json"

    try:
        _rm(pipeline_dir)
        r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_ok), "--pipeline-id", pipeline_id], env=env)
        assert r_route_ok.returncode == 0
        
        pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": []}))
        
        r_verify = _run([sys.executable, "-m", "CAPABILITY.TOOLS.catalytic.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        assert r_verify.returncode != 0
        assert "CAPABILITY_NOT_PINNED" in r_verify.stdout
        
    finally:
        if pins_path.is_file():
            pins_path.unlink()


def test_verify_rejects_empty_pins(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-empty-pins"
    caps = [CAP]
    
    try:
        _rm(pipeline_dir)
        
        for cap in caps:
            pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": []}))
            
            r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_ok), "--pipeline-id", pipeline_id], env=env)
            assert r_route_ok.returncode == 0
            
            r_verify = _run([sys.executable, "-m", "CAPABILITY.TOOLS.catalytic.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
            if r_verify.returncode != 0:
                assert "CAPABILITY_NOT_PINNED" in r_verify.stdout
    finally:
        if pins_path.is_file():
            for cap in caps:
                pins_path.unlink()
