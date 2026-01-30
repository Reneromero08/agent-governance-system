from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Correctly calculating and using the REPO_ROOT path to ensure correct paths are used.
REPO_ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(0, str(REPO_ROOT))
# Ensure that local relative imports from `tools` folder are corrected if they break.

from CAPABILITY.PIPELINES.swarm_runtime import SwarmRuntime
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

def test_swarm_runtime_happy_path(tmp_path: Path) -> None:
    swarm_id = "swarm-ok"
    p1 = "swarm-p1"
    p2 = "swarm-p2"

    runs_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_pipelines" / "_dags" / swarm_id

    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline1_dir = rt_pipeline.pipeline_dir(p1)
    pipeline2_dir = rt_pipeline.pipeline_dir(p2)

    out1 = f"NAVIGATION/CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"NAVIGATION/CORTEX/_generated/_tmp/{p2}.txt"
    jobspec1_rel = f"LAW/CONTRACTS/_runs/_tmp/swarm_test/{p1}_jobspec.json"
    jobspec2_rel = f"LAW/CONTRACTS/_runs/_tmp/swarm_test/{p2}_jobspec.json"

    try:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "swarm_test")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_jobspec(REPO_ROOT / jobspec1_rel, job_id=f"{p1}-job", intent=p1, catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/swarm_test/domain"], durable_paths=[out1])
        _write_jobspec(REPO_ROOT / jobspec2_rel, job_id=f"{p2}-job", intent=p2, catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/swarm_test/domain"], durable_paths=[out2])

        pipeline_spec_1 = {
            "pipeline_id": p1,
            "steps": [
                {
                    "step_id": "s1",
                    "jobspec_path": jobspec1_rel,
                    "memoize": False,
                    "cmd": [sys.executable, "-c", f"from pathlib import Path;Path('{out1}').parent.mkdir(parents=True, exist_ok=True);Path('{out1}').write_text('{p1}')"],
                }
            ]
        }

        pipeline_spec_2 = {
            "pipeline_id": p2,
            "steps": [
                {
                    "step_id": "s1",
                    "jobspec_path": jobspec2_rel,
                    "memoize": False,
                    "cmd": [sys.executable, "-c", f"from pathlib import Path;Path('{out2}').parent.mkdir(parents=True, exist_ok=True);Path('{out2}').write_text('{p2}')"],
                }
            ]
        }

        swarm_spec = {
            "swarm_version": "1.0.0",
            "swarm_id": swarm_id,
            "nodes": [
                {
                    "node_id": p1,
                    "pipeline_spec": pipeline_spec_1
                },
                {
                    "node_id": p2,
                    "pipeline_spec": pipeline_spec_2
                }
            ],
            "edges": [
                {
                    "from": p1,
                    "to": p2,
                    "requires": ["CHAIN.json"]
                }
            ]
        }

        spec_path = tmp_path / "swarm.json"
        spec_path.write_text(json.dumps(swarm_spec, indent=2), encoding="utf-8")

        sr = SwarmRuntime(project_root=REPO_ROOT, runs_root=runs_root)
        res = sr.run(swarm_id=swarm_id, spec_path=spec_path)
        assert res.get("ok") is True
        assert (swarm_dir / "SWARM_STATE.json").exists()
        assert (swarm_dir / "SWARM_CHAIN.json").exists()

    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "swarm_test")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


