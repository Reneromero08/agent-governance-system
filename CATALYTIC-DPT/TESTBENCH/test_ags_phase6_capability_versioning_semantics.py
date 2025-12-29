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


def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


def test_capability_versioning_in_place_upgrade_rejected(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-versioning"
    step_id = "s1"
    base_cap = "e8e7e5234b43278a1a257b9257186b8bca5fdae9ab9096572942da1c5fb90f36"

    # Canonical registry derived from repo CAPABILITIES, but tamper adapter bytes under same key.
    reg_path = tmp_path / "CAPABILITIES.json"
    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [base_cap]}))

    # Minimal adapter whose canonical hash is base_cap.
    # We only need to demonstrate that changing bytes changes hash and is rejected as mismatch.
    adapter = {"name": "x"}
    computed = _sha256_hex(_canon(adapter))
    # Construct a registry where key claims base_cap but adapter hashes to something else.
    reg = {"registry_version": "1.0.0", "capabilities": {base_cap: {"adapter_spec_hash": base_cap, "adapter": adapter}}}
    reg_path.write_bytes(_canon(reg))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": step_id, "capability_hash": base_cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(pipeline_dir)
        r = _run([sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r.returncode != 0
        assert "CAPABILITY_HASH_MISMATCH" in (r.stderr + r.stdout)
    finally:
        _rm(pipeline_dir)

