from __future__ import annotations

import json
import shutil
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


def test_pipeline_init_is_deterministic_and_resume_safe() -> None:
    import sys

    # sys.path cleanup
    from CAPABILITY.PIPELINES.pipeline_runtime import PipelineRuntime

    pipeline_id = "pipelines-test"
    base = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "pipelines_test"
    pipeline_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id

    jobspec1_rel = "LAW/CONTRACTS/_runs/_tmp/pipelines_test/jobspec_step1.json"
    jobspec2_rel = "LAW/CONTRACTS/_runs/_tmp/pipelines_test/jobspec_step2.json"
    jobspec1 = REPO_ROOT / jobspec1_rel
    jobspec2 = REPO_ROOT / jobspec2_rel

    out1 = "NAVIGATION/CORTEX/_generated/_tmp/pipelines_step1_out.txt"
    out2 = "NAVIGATION/CORTEX/_generated/_tmp/pipelines_step2_out.txt"
    marker1 = "LAW/CONTRACTS/_runs/_tmp/pipelines_test/step1_marker.txt"

    spec_path = base / "PIPELINE_SPEC.json"

    _rm(base)
    _rm(pipeline_dir)
    _rm(REPO_ROOT / out1)
    _rm(REPO_ROOT / out2)
    _rm(REPO_ROOT / marker1)

    try:
        _write_jobspec(jobspec1, job_id="pipe-step1", intent="step1", catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/pipelines_test/domain"], durable_paths=[out1])
        _write_jobspec(jobspec2, job_id="pipe-step2", intent="step2", catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/pipelines_test/domain"], durable_paths=[out2])

        spec = {
            "pipeline_id": pipeline_id,
            "validator_semver": "0.1.0",
            "validator_build_id": "pipeline-test",
            "timestamp": "CATALYTIC-DPT-02_CONFIG",
            "steps": [
                {
                    "step_id": "s1",
                    "jobspec_path": jobspec1_rel,
                    "memoize": False,
                    "cmd": [
                        sys.executable,
                        "-c",                        (
                            "from pathlib import Path;"
                            f"m=Path('{marker1}');"
                            "import sys;"
                            "sys.exit(3) if m.exists() else None;"
                            "m.parent.mkdir(parents=True, exist_ok=True);"
                            "m.write_text('ran', encoding='utf-8');"
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
                        "-c",                        (
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

        rt = PipelineRuntime(project_root=REPO_ROOT)
        rt.init_from_spec_path(pipeline_id=pipeline_id, spec_path=spec_path)

        pipeline_json_1 = (pipeline_dir / "PIPELINE.json").read_bytes()
        state_json_1 = (pipeline_dir / "STATE.json").read_bytes()

        # Deterministic re-init (idempotent bytes).
        rt.init_from_spec_path(pipeline_id=pipeline_id, spec_path=spec_path)
        assert (pipeline_dir / "PIPELINE.json").read_bytes() == pipeline_json_1
        assert (pipeline_dir / "STATE.json").read_bytes() == state_json_1

        # Run only first step (simulated interruption).
        rt.run(pipeline_id=pipeline_id, max_steps=1)

        # Resume should not rerun step1 (would exit 3 if rerun), and should complete step2.
        rt2 = PipelineRuntime(project_root=REPO_ROOT)
        rt2.run(pipeline_id=pipeline_id)

        # Step outputs exist (artifacts in run dirs).
        state = json.loads((pipeline_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id_s1 = state["step_run_ids"]["s1"]
        run_id_s2 = state["step_run_ids"]["s2"]
        for rid in [run_id_s1, run_id_s2]:
            rdir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / rid
            assert (rdir / "PROOF.json").exists()
            assert (rdir / "DOMAIN_ROOTS.json").exists()
            assert (rdir / "LEDGER.jsonl").exists()

        # Deterministic status text.
        status_a = rt2.status_text(pipeline_id=pipeline_id)
        status_b = PipelineRuntime(project_root=REPO_ROOT).status_text(pipeline_id=pipeline_id)
        assert status_a == status_b
        assert "next_step: NONE" in status_a
    finally:
        # Cleanup pipeline + step runs + tmp artifacts.
        _rm(pipeline_dir)
        _rm(base)
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)
        _rm(REPO_ROOT / marker1)
