from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_chain import verify_chain
from PIPELINES.pipeline_runtime import PipelineRuntime


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
    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / tmp_root
    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    jobspec1_rel = f"CONTRACTS/_runs/_tmp/{tmp_root}/jobspec_step1.json"
    jobspec2_rel = f"CONTRACTS/_runs/_tmp/{tmp_root}/jobspec_step2.json"
    jobspec1 = REPO_ROOT / jobspec1_rel
    jobspec2 = REPO_ROOT / jobspec2_rel

    domain_rel = f"CONTRACTS/_runs/_tmp/{tmp_root.as_posix()}/domain"
    out1 = f"CORTEX/_generated/_tmp/{pipeline_id}_step1_out.txt"
    out2 = f"CORTEX/_generated/_tmp/{pipeline_id}_step2_out.txt"

    spec_path = base / "PIPELINE_SPEC.json"

    _rm(base)
    _rm(pipeline_dir)
    _rm(REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
    _rm(REPO_ROOT / "CONTRACTS" / "_runs" / f"pipeline-{pipeline_id}-s2-a1")
    _rm(REPO_ROOT / out1)
    _rm(REPO_ROOT / out2)

    _write_jobspec(jobspec1, job_id="pipe-chain-step1", intent="step1", catalytic_domains=[domain_rel], durable_paths=[out1])
    _write_jobspec(jobspec2, job_id="pipe-chain-step2", intent="step2", catalytic_domains=[domain_rel], durable_paths=[out2])

    spec = {
        "pipeline_id": pipeline_id,
        "validator_semver": "0.1.0",
        "validator_build_id": "pipeline-chain-test",
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "steps": [
            {
                "step_id": "s1",
                "jobspec_path": jobspec1_rel,
                "memoize": False,
                "cmd": [
                    "python3",
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
                    "python3",
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


def test_pipeline_chain_valid(tmp_path: Path) -> None:
    pipeline_id = "pipeline-chain-valid"
    spec_path, pipeline_dir = _setup_pipeline(Path("pipeline_chain_valid"), pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        assert (pipeline_dir / "CHAIN.json").exists()
        v = verify_chain(project_root=REPO_ROOT, pipeline_dir=pipeline_dir)
        assert v["ok"] is True
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_chain_valid")


def test_pipeline_chain_tamper(tmp_path: Path) -> None:
    pipeline_id = "pipeline-chain-tamper"
    spec_path, pipeline_dir = _setup_pipeline(Path("pipeline_chain_tamper"), pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id_s1 = state["step_run_ids"]["s1"]
        proof = REPO_ROOT / "CONTRACTS" / "_runs" / run_id_s1 / "PROOF.json"
        data = proof.read_bytes()
        proof.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))

        v = verify_chain(project_root=REPO_ROOT, pipeline_dir=pipeline_dir)
        assert v["ok"] is False
        assert v["code"] == "CHAIN_PROOF_HASH_MISMATCH"
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_chain_tamper")


def test_pipeline_chain_reorder(tmp_path: Path) -> None:
    pipeline_id = "pipeline-chain-reorder"
    spec_path, pipeline_dir = _setup_pipeline(Path("pipeline_chain_reorder"), pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        chain_path = pipeline_dir / "CHAIN.json"
        obj = json.loads(chain_path.read_text(encoding="utf-8"))
        obj["steps"] = list(reversed(obj["steps"]))
        chain_path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")

        v = verify_chain(project_root=REPO_ROOT, pipeline_dir=pipeline_dir)
        assert v["ok"] is False
        assert v["code"] == "CHAIN_STEP_ORDER_MISMATCH"
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_chain_reorder")


def test_pipeline_chain_determinism(tmp_path: Path) -> None:
    pipeline_id = "pipeline-chain-determinism"
    tmp_root = Path("pipeline_chain_determinism")
    spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
    rt = PipelineRuntime(project_root=REPO_ROOT)
    try:
        rt.run(pipeline_id=pipeline_id, spec_path=spec_path)
        chain_bytes_1 = (pipeline_dir / "CHAIN.json").read_bytes()

        # Clean pipeline artifacts + runs, rerun identically.
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_ids = [state["step_run_ids"][sid] for sid in ["s1", "s2"]]
        _rm(pipeline_dir)
        for rid in run_ids:
            _rm(REPO_ROOT / "CONTRACTS" / "_runs" / rid)

        spec_path, pipeline_dir = _setup_pipeline(tmp_root, pipeline_id=pipeline_id)
        rt2 = PipelineRuntime(project_root=REPO_ROOT)
        rt2.run(pipeline_id=pipeline_id, spec_path=spec_path)
        chain_bytes_2 = (pipeline_dir / "CHAIN.json").read_bytes()

        assert chain_bytes_1 == chain_bytes_2
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_chain_determinism")
