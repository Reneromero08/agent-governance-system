from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_runtime import PipelineRuntime


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_jobspec(path: Path, *, job_id: str, catalytic_domains: list[str], durable_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "job_id": job_id,
        "phase": 4,
        "task_type": "pipeline_execution",
        "intent": "adversarial pipeline test",
        "inputs": {},
        "outputs": {"durable_paths": durable_paths, "validation_criteria": {}},
        "catalytic_domains": catalytic_domains,
        "determinism": "deterministic",
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_pipeline_fails_closed_on_partial_state_json() -> None:
    pipeline_id = "adversarial-pipeline-partial-state"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    pdir = rt.pipeline_dir(pipeline_id)
    _rm(pdir)
    pdir.mkdir(parents=True, exist_ok=True)

    # Minimal initialized pipeline files.
    (pdir / "PIPELINE.json").write_text(json.dumps({"pipeline_id": pipeline_id, "steps": [{"step_id": "s1", "jobspec_path": "x.json", "cmd": ["true"]}]}), encoding="utf-8")
    (pdir / "STATE.json").write_text("{", encoding="utf-8")  # partial JSON (crash mid-write)

    with pytest.raises(json.JSONDecodeError):
        _ = rt.status_text(pipeline_id=pipeline_id)


def test_pipeline_fails_closed_on_inconsistent_completed_step_state() -> None:
    pipeline_id = "adversarial-pipeline-inconsistent-state"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    pdir = rt.pipeline_dir(pipeline_id)
    _rm(pdir)
    pdir.mkdir(parents=True, exist_ok=True)

    # Initialized pipeline with 2 steps.
    pipeline_obj = {
        "pipeline_id": pipeline_id,
        "validator_semver": "0.1.0",
        "validator_build_id": "adversarial",
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "steps": [
            {"step_id": "s1", "jobspec_path": "x.json", "cmd": ["true"], "strict": True, "memoize": False},
            {"step_id": "s2", "jobspec_path": "y.json", "cmd": ["true"], "strict": True, "memoize": False},
        ],
    }
    (pdir / "PIPELINE.json").write_text(json.dumps(pipeline_obj), encoding="utf-8")

    # Corrupt state: claims s1 is completed but has no step_run_ids entry.
    state_obj = {"pipeline_id": pipeline_id, "current_step_index": 1, "completed_steps": ["s1"], "step_run_ids": {}, "attempts": {}}
    (pdir / "STATE.json").write_text(json.dumps(state_obj), encoding="utf-8")

    with pytest.raises(ValueError, match=r"inconsistent pipeline state"):
        _ = rt.status_text(pipeline_id=pipeline_id)


def test_pipeline_resume_never_skips_a_missing_step_record() -> None:
    pipeline_id = "adversarial-pipeline-resume"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    pdir = rt.pipeline_dir(pipeline_id)
    _rm(pdir)

    base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "pipeline_resume" / pipeline_id
    _rm(base)
    base.mkdir(parents=True, exist_ok=True)

    domain_rel = "CONTRACTS/_runs/_tmp/adversarial/pipeline_resume/domain"
    (REPO_ROOT / domain_rel).mkdir(parents=True, exist_ok=True)

    jobspec1 = base / "jobspec_s1.json"
    jobspec2 = base / "jobspec_s2.json"
    out1 = "CORTEX/_generated/_tmp/adversarial_pipeline_step1.txt"
    out2 = "CORTEX/_generated/_tmp/adversarial_pipeline_step2.txt"

    _write_jobspec(jobspec1, job_id="adv-pipe-s1", catalytic_domains=[domain_rel], durable_paths=[out1])
    _write_jobspec(jobspec2, job_id="adv-pipe-s2", catalytic_domains=[domain_rel], durable_paths=[out2])

    spec = {
        "pipeline_id": pipeline_id,
        "validator_semver": "0.1.0",
        "validator_build_id": "adversarial",
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "steps": [
            {
                "step_id": "s1",
                "jobspec_path": str(jobspec1.relative_to(REPO_ROOT)).replace("\\", "/"),
                "cmd": [
                    "python3",
                    "-c",
                    f"from pathlib import Path; Path('{out1}').parent.mkdir(parents=True, exist_ok=True); Path('{out1}').write_text('ONE', encoding='utf-8')",
                ],
                "strict": True,
                "memoize": False,
            },
            {
                "step_id": "s2",
                "jobspec_path": str(jobspec2.relative_to(REPO_ROOT)).replace("\\", "/"),
                "cmd": [
                    "python3",
                    "-c",
                    f"from pathlib import Path; Path('{out2}').parent.mkdir(parents=True, exist_ok=True); Path('{out2}').write_text('TWO', encoding='utf-8')",
                ],
                "strict": True,
                "memoize": False,
            },
        ],
    }
    spec_path = base / "PIPELINE_SPEC.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # Simulate crash between step completion and STATE update:
    # - run step1 using the internal executor
    # - do NOT write RUN_REF.json nor update STATE.json
    rt.init_from_spec_path(pipeline_id=pipeline_id, spec_path=spec_path)
    parsed_spec, state = rt.load(pipeline_id=pipeline_id)
    step1 = parsed_spec.steps[0]
    run_id = rt._make_run_id(parsed_spec.pipeline_id, step1.step_id, 1)
    (REPO_ROOT / "CONTRACTS" / "_runs" / run_id).mkdir(parents=True, exist_ok=True)
    rt._execute_step(spec=parsed_spec, step=step1, run_id=run_id)
    rt._assert_step_outputs(REPO_ROOT / "CONTRACTS" / "_runs" / run_id)

    # Resume must not "skip ahead" based on partial artifacts; it must complete s1 and then s2.
    rt2 = PipelineRuntime(project_root=REPO_ROOT)
    rt2.run(pipeline_id=pipeline_id)
    _, state2 = rt2.load(pipeline_id=pipeline_id)
    assert state2["completed_steps"] == ["s1", "s2"]
