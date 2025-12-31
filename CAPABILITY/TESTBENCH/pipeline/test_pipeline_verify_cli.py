from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PIPELINES.pipeline_runtime import PipelineRuntime


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_jobspec(path: Path, *, job_id: str, intent: str, catalytic_domains: list[str], durable_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "job_id": job_id,
        "phase": 5,
        "task_type": "pipeline_execution",
        "intent": intent,
        "inputs": {},
        "outputs": {"durable_paths": durable_paths, "validation_criteria": {}},
        "catalytic_domains": catalytic_domains,
        "determinism": "deterministic",
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _setup_pipeline(tmp_root: Path, *, pipeline_id: str) -> tuple[Path, Path]:
    base = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    jobspec1_rel = f"LAW/CONTRACTS/_runs/_tmp/{tmp_root}/jobspec_step1.json"
    jobspec2_rel = f"LAW/CONTRACTS/_runs/_tmp/{tmp_root}/jobspec_step2.json"
    jobspec1 = REPO_ROOT / jobspec1_rel
    jobspec2 = REPO_ROOT / jobspec2_rel

    domain_rel = f"LAW/CONTRACTS/_runs/_tmp/{tmp_root.as_posix()}/domain"
    out1 = f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_verify_step1_out.txt"
    out2 = f"NAVIGATION/CORTEX/_generated/_tmp/{pipeline_id}_verify_step2_out.txt"

    spec_path = base / "PIPELINE_SPEC.json"

    _rm(base)
    _rm(pipeline_dir)
    _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
    _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")

    _write_jobspec(jobspec1, job_id="pipe-verify-step1", intent="step1", catalytic_domains=[domain_rel], durable_paths=[out1])
    _write_jobspec(jobspec2, job_id="pipe-verify-step2", intent="step2", catalytic_domains=[domain_rel], durable_paths=[out2])

    spec = {
        "pipeline_id": pipeline_id,
        "validator_semver": "0.1.0",
        "validator_build_id": "pipeline-verify-test",
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "steps": [
            {
                "step_id": "s1",
                "jobspec_path": jobspec1_rel,
                "memoize": False,
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path;"
                        f"Path('{out1}').parent.mkdir(parents=True, exist_ok=True);"
                        f"Path('{out1}').write_text('ONE', encoding='utf-8')"
                    ),
                ],
            },
            {
                "step_id": "s2",
                "jobspec_path": jobspec2_rel,
                "memoize": False,
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path;"
                        f"Path('{out2}').parent.mkdir(parents=True, exist_ok=True);"
                        f"Path('{out2}').write_text('TWO', encoding='utf-8')"
                    ),
                ],
            },
        ],
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    return spec_path, pipeline_dir


def _run_verify_cli(*, pipeline_id: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "catalytic.py"),
        "pipeline",
        "verify",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        "LAW/CONTRACTS/_runs",
        "--strict",
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def test_pipeline_verify_ok(tmp_path: Path) -> None:
    pipeline_id = "pipeline-verify-ok"
    tmp_root = Path("pipeline_verify_ok")
    spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        res = _run_verify_cli(pipeline_id=pipeline_id)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "OK pipeline_id=" in res.stdout
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")


def test_pipeline_verify_missing_artifact(tmp_path: Path) -> None:
    pipeline_id = "pipeline-verify-missing"
    tmp_root = Path("pipeline_verify_missing")
    spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id = state["step_run_ids"]["s1"]
        ledger = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id / "LEDGER.jsonl"
        _rm(ledger)

        res = _run_verify_cli(pipeline_id=pipeline_id)
        assert res.returncode != 0
        assert "FAIL" in res.stdout
        assert "STEP_ARTIFACT_MISSING" in res.stdout
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")


def test_pipeline_verify_chain_tamper(tmp_path: Path) -> None:
    pipeline_id = "pipeline-verify-chain-tamper"
    tmp_root = Path("pipeline_verify_chain_tamper")
    spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id = state["step_run_ids"]["s1"]
        ledger = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id / "LEDGER.jsonl"
        data = ledger.read_bytes()
        assert data.endswith(b"\n")
        ledger.write_bytes(data[:-1])  # remove trailing newline -> partial line

        res = _run_verify_cli(pipeline_id=pipeline_id)
        assert res.returncode != 0
        assert "FAIL" in res.stdout
        assert "LEDGER_CORRUPT" in res.stdout
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")


def test_pipeline_verify_ledger_corrupt(tmp_path: Path) -> None:
    pipeline_id = "pipeline-verify-ledger-corrupt"
    tmp_root = Path("pipeline_verify_ledger_corrupt")
    spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id = state["step_run_ids"]["s1"]
        ledger = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id / "LEDGER.jsonl"
        data = ledger.read_bytes()
        assert data.endswith(b"\n")
        ledger.write_bytes(data[:-1])  # remove trailing newline -> partial line

        res = _run_verify_cli(pipeline_id=pipeline_id)
        assert res.returncode != 0
        assert "FAIL" in res.stdout
        assert "LEDGER_CORRUPT" in res.stdout
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / tmp_root)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")
