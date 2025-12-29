from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_ags(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "CAPABILITY.TOOLS.ags", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _hex(ch: str) -> str:
    return ch * 64


def _jobspec(tmp_root: str) -> dict:
    return {
        "job_id": "ags-adapter-step1",
        "phase": 6,
        "task_type": "pipeline_execution",
        "intent": "adapter step",
        "inputs": {},
        "outputs": {"durable_paths": [f"NAVIGATION/CORTEX/_generated/_tmp/{tmp_root}_out.txt"], "validation_criteria": {}},
        "catalytic_domains": [f"LAW/CONTRACTS/_runs/_tmp/{tmp_root}/domain"],
        "determinism": "deterministic",
    }


def _adapter(tmp_root: str, **overrides: object) -> dict:
    base = {
        "adapter_version": "1.0.0",
        "name": "test-adapter",
            "command": [sys.executable, "-c", "print('hello')"],
        "jobspec": _jobspec(tmp_root),
        "inputs": {"LAW/CONTRACTS/_runs/_tmp/%s/in.txt" % tmp_root: _hex("a")},
        "outputs": {"LAW/CONTRACTS/_runs/_tmp/%s/out.txt" % tmp_root: _hex("b")},
        "side_effects": {"network": False, "clock": False, "filesystem_unbounded": False, "nondeterministic": False},
        "deref_caps": {"max_bytes": 1024, "max_matches": 1, "max_nodes": 10, "max_depth": 2},
        "artifacts": {"ledger": _hex("c"), "proof": _hex("d"), "domain_roots": _hex("e")},
    }
    base.update(overrides)
    return base


def test_adapter_happy_path_deterministic(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-happy"
    tmp_root = "ags_adapter_happy"
    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": _adapter(tmp_root)}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    try:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        r1 = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
        assert r1.returncode == 0, r1.stdout + r1.stderr
        pipeline_bytes_1 = (pipeline_dir / "PIPELINE.json").read_bytes()

        r2 = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
        assert r2.returncode == 0, r2.stdout + r2.stderr
        pipeline_bytes_2 = (pipeline_dir / "PIPELINE.json").read_bytes()
        assert pipeline_bytes_1 == pipeline_bytes_2
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"NAVIGATION/CORTEX/_generated/_tmp/{tmp_root}_out.txt")


def test_adapter_reject_missing_command(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-missing-command"
    tmp_root = "ags_adapter_missing_command"
    a = _adapter(tmp_root)
    a.pop("command", None)
    plan = {"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": a}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
    assert r.returncode != 0


def test_adapter_reject_hidden_side_effects(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-side-effects"
    tmp_root = "ags_adapter_side_effects"
    a = _adapter(tmp_root)
    a["side_effects"] = dict(a["side_effects"])
    a["side_effects"]["network"] = True
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": a}]}), encoding="utf-8")
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
    assert r.returncode != 0
    assert "ADAPTER_SIDE_EFFECTS_FORBIDDEN" in r.stderr


def test_adapter_reject_unbounded_deref_caps(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-deref-caps"
    tmp_root = "ags_adapter_deref_caps"
    a = _adapter(tmp_root)
    a["deref_caps"] = {"max_bytes": 10_000_000, "max_matches": 1, "max_nodes": 10, "max_depth": 2}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": a}]}), encoding="utf-8")
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
    assert r.returncode != 0
    assert "DEREF_CAPS_TOO_LARGE" in r.stderr


def test_adapter_reject_non_normalized_paths(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-bad-paths"
    tmp_root = "ags_adapter_bad_paths"
    a = _adapter(tmp_root)
    a["inputs"] = {"../escape.txt": _hex("a")}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": a}]}), encoding="utf-8")
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
    assert r.returncode != 0
    assert "NON_NORMALIZED_PATH" in r.stderr


def test_adapter_reject_nondeterministic_flag_strict(tmp_path: Path) -> None:
    pipeline_id = "ags-adapter-nondet"
    tmp_root = "ags_adapter_nondet"
    a = _adapter(tmp_root)
    a["side_effects"] = dict(a["side_effects"])
    a["side_effects"]["nondeterministic"] = True
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "adapter": a}]}), encoding="utf-8")
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id])
    assert r.returncode != 0
    assert "ADAPTER_SIDE_EFFECTS_FORBIDDEN" in r.stderr

