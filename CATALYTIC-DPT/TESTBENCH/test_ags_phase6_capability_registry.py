from __future__ import annotations

import hashlib
import json
import os
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


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


def test_capability_registry_happy_unknown_and_tamper(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-registry"
    step_id = "s1"

    reg_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase65_registry"
    task_path = reg_root / "task.json"
    result_path = reg_root / "result.json"
    in_path = reg_root / "in.txt"
    out_path = reg_root / "out.txt"
    domain_dir = reg_root / "domain"

    adapter = {
                    "adapter_version": "1.0.0",
                    "name": "ant-worker-copy-v1",
                    "command": [
                        sys.executable,
                        "CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py",
                        str(task_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                        str(result_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    ],
                    "jobspec": {            "job_id": "cap-ant-worker-copy-v1",
            "phase": 6,
            "task_type": "adapter_execution",
            "intent": "capability: ant-worker copy (Phase 6.5)",
            "inputs": {},
            "outputs": {"durable_paths": [str(out_path.relative_to(REPO_ROOT)).replace("\\", "/")], "validation_criteria": {}},
            "catalytic_domains": [str(domain_dir.relative_to(REPO_ROOT)).replace("\\", "/")],
            "determinism": "deterministic",
        },
        "inputs": {str(in_path.relative_to(REPO_ROOT)).replace("\\", "/"): "a" * 64},
        "outputs": {str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"): "b" * 64},
        "side_effects": {"network": False, "clock": False, "filesystem_unbounded": False, "nondeterministic": False},
        "deref_caps": {"max_bytes": 1024, "max_matches": 1, "max_nodes": 10, "max_depth": 2},
        "artifacts": {"ledger": "c" * 64, "proof": "d" * 64, "domain_roots": "e" * 64},
    }
    cap = _sha256_hex(_canon(adapter))
    registry = {"registry_version": "1.0.0", "capabilities": {cap: {"adapter_spec_hash": cap, "adapter": adapter}}}
    reg_path = tmp_path / "CAPABILITIES.json"
    reg_path.write_bytes(_canon(registry))

    pins = {"pins_version": "1.0.0", "allowed_capabilities": sorted([cap])}
    pins_path = tmp_path / "CAPABILITY_PINS.json"
    pins_path.write_bytes(_canon(pins))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    plan_ok = {"plan_version": "1.0", "steps": [{"step_id": step_id, "capability_hash": cap}]}
    plan_ok_path = tmp_path / "plan_ok.json"
    plan_ok_path.write_text(json.dumps(plan_ok), encoding="utf-8")

    plan_bad = {"plan_version": "1.0", "steps": [{"step_id": step_id, "capability_hash": "0" * 64}]}
    plan_bad_path = tmp_path / "plan_bad.json"
    plan_bad_path.write_text(json.dumps(plan_bad), encoding="utf-8")

    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    run_dir = REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-{step_id}-a1"

    try:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)
        domain_dir.mkdir(parents=True, exist_ok=True)
        in_path.write_bytes(b"PHASE65\n")
        task = {
            "task_id": "ant-worker-copy",
            "task_type": "file_operation",
            "operation": "copy",
            "verify_integrity": True,
            "timestamp": "CATALYTIC-DPT-02_CONFIG",
            "files": [{"source": str(in_path), "destination": str(out_path)}],
        }
        task_path.write_text(json.dumps(task), encoding="utf-8")

        # Unknown capability rejects at route time.
        r_bad = _run(
            [sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_bad_path), "--pipeline-id", pipeline_id],
            env=env,
        )
        assert r_bad.returncode != 0
        assert "UNKNOWN_CAPABILITY" in (r_bad.stderr + r_bad.stdout)

        # Happy path routes, runs, verifies.
        r_route = _run(
            [sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_ok_path), "--pipeline-id", pipeline_id],
            env=env,
        )
        assert r_route.returncode == 0, r_route.stdout + r_route.stderr

        r_run = _run([sys.executable, "-m", "TOOLS.ags", "run", "--pipeline-id", pipeline_id, "--strict", "--skip-preflight"], env=env)
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        r_verify_ok = _run([sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        assert r_verify_ok.returncode == 0, r_verify_ok.stdout + r_verify_ok.stderr

        # Registry tamper: change adapter but keep capability key; verify must fail closed.
        tampered = json.loads(reg_path.read_text(encoding="utf-8"))
        tampered["capabilities"][cap]["adapter"]["name"] = "tampered"
        reg_path.write_bytes(_canon(tampered))

        r_verify_bad = _run([sys.executable, "TOOLS/catalytic.py", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        assert r_verify_bad.returncode != 0
        assert "CAPABILITY_HASH_MISMATCH" in r_verify_bad.stdout
    finally:
        _rm(pipeline_dir)
        _rm(run_dir)
        _rm(reg_root)
