from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_ags(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "TOOLS.ags", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)


def _canon(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_registry_duplicate_capability_hash_rejects(tmp_path: Path) -> None:
    cap = "46d06ff771e5857d84895ad4af4ac94196dfa5bf3f60a47140af039985f79e34"
    # Duplicate keys inside capabilities object (must be detected via object_pairs_hook).
    raw = (
        '{"registry_version":"1.0.0","capabilities":{'
        f'"{cap}":{{"adapter_spec_hash":"{cap}","adapter":{{}}}},'
        f'"{cap}":{{"adapter_spec_hash":"{cap}","adapter":{{}}}}'
        "}}"
    )
    reg_path = tmp_path / "CAPABILITIES.json"
    reg_path.write_text(raw, encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", "reg-dupe"], env=env)
    assert r.returncode != 0
    assert "REGISTRY_DUPLICATE_HASH" in (r.stderr + r.stdout)


def test_registry_noncanonical_json_rejects(tmp_path: Path) -> None:
    cap = "46d06ff771e5857d84895ad4af4ac94196dfa5bf3f60a47140af039985f79e34"
    obj = {"registry_version": "1.0.0", "capabilities": {cap: {"adapter_spec_hash": cap, "adapter": {}}}}
    reg_path = tmp_path / "CAPABILITIES.json"
    reg_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", "reg-noncanon"], env=env)
    assert r.returncode != 0
    assert "REGISTRY_NONCANONICAL" in (r.stderr + r.stdout)


def test_registry_tamper_detected_fail_closed(tmp_path: Path) -> None:
    cap = "46d06ff771e5857d84895ad4af4ac94196dfa5bf3f60a47140af039985f79e34"
    reg_path = tmp_path / "CAPABILITIES.json"
    # Canonical bytes but internally inconsistent (hash mismatch) -> tampered.
    obj = {
        "registry_version": "1.0.0",
        "capabilities": {cap: {"adapter_spec_hash": cap, "adapter": {"name": "tampered"}}},
    }
    reg_path.write_bytes(_canon(obj))

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(reg_path)
    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", "reg-tamper"], env=env)
    assert r.returncode != 0
    assert "REGISTRY_TAMPERED" in (r.stderr + r.stdout)

