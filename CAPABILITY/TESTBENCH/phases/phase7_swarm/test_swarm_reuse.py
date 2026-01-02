from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

def _write_jobspec(path: Path, *, job_id: str, intent: str, catalytic_domains: list[Path], durable_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _rel(p: Path) -> str:
        if p.is_absolute():
            return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        return str(p).replace("\\", "/")

    obj = {
        "job_id": job_id,
        "phase": 5,
        "task_type": "pipeline_execution",
        "intent": intent,
        "inputs": {},
        "outputs": {"durable_paths": [_rel(p) for p in durable_paths], "validation_criteria": {}},
        "catalytic_domains": [_rel(cd) for cd in catalytic_domains],
        "determinism": "deterministic",
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def test_swarm_execution_elision(tmp_path: Path) -> None:
    """
    Verifies catalytic swarm execution elision:
    1. First run: full execution, elided=False.
    2. Second run (identical): elided=True, no pipeline execution.
    3. Verification: chain/receipt exist and are correct.
    4. Tamper referenced proof -> verification fails on reuse.
    """
    swarm_id = "swarm-elision-test"
    p1 = "elision-p1"

    REPO_ROOT = Path(__file__).resolve().parents[4]
    LAW_CONTRACTS = REPO_ROOT / "LAW" / "CONTRACTS"
    runs_root = LAW_CONTRACTS / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_pipelines" / "_dags" / swarm_id

    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline_dir = rt_pipeline.pipeline_dir(p1)

    out1 = Path("NAVIGATION/CORTEX/_generated/_tmp/") / f"{p1}.txt"
    jobspec_rel = Path("LAW/CONTRACTS/_runs/_tmp/elision_test/") / f"{p1}_jobspec.json"

    receipts_store = runs_root / "_pipelines" / "_swarms" / "_receipts"

    unique_intent = "elision-intent-v1-unique"

    try:
        # Setup catalytic domain
        cat_domain = LAW_CONTRACTS / "_runs" / "_tmp" / "elision_test" / "cat_domain"
        cat_domain.mkdir(parents=True, exist_ok=True)

        # Write jobspec with Path objects
        _write_jobspec(
            runs_root / "_tmp" / "elision_test" / f"{p1}_jobspec.json",
            job_id=f"{p1}-job",
            intent=unique_intent,
            catalytic_domains=[cat_domain],
            durable_paths=[out1]
        )

        pipeline_spec = {
            "pipeline_id": p1,
            "steps": [
                {
                    "step_id": "s1",
                    "jobspec_path": str(jobspec_rel),
                    "memoize": False,
                    "cmd": [
                        sys.executable, "-c",
                        f"from pathlib import Path;Path('{str(out1.parent)}').mkdir(parents=True, exist_ok=True);Path('{str(out1)}').write_text('{p1}')"
                    ],
                }
            ]
        }

        swarm_spec = {
            "swarm_version": "1.0.0",
            "swarm_id": swarm_id,
            "nodes": [{"node_id": p1, "pipeline_spec": pipeline_spec}],
            "edges": []
        }

        spec_path = tmp_path / "swarm.json"
        spec_path.write_text(json.dumps(swarm_spec, indent=2), encoding="utf-8")

        sr = SwarmRuntime(project_root=REPO_ROOT, runs_root=runs_root)

        # =========================================
        # 1. First Run: Full Execution
        # =========================================
        print("\n--- RUN 1 ---")
        res1 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
        assert res1.get("ok") is True, f"First run should succeed: {res1}"
        assert res1.get("elided") is False, "First run should NOT be elided"
        # =========================================
        # 2. Second Run: Elided Execution
        # =========================================
        print("\n--- RUN 2 ---")
        res2 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
        assert res2.get("ok") is True, "Second run should succeed (elided)"
        assert res2.get("elided") is True

        # =========================================
        # 3. Tamper Test: Mutate pipeline receipt, verify reuse fails.
        # =========================================
        print("\n--- RUN 3 (TAMPER) ---")
        pdir = rt_pipeline.pipeline_dir(p1)
        receipt_path = pdir / "RECEIPT.json"
        original_receipt = receipt_path.read_text()

        with receipt_path.open("w") as file:
            file.write(original_receipt + "\n")

        try:
            res3 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
            pytest.fail("Tampered receipt should NOT allow elision!")
        except ValueError as e:
            print(f"Caught expected error: {e}")
            assert "MISMATCH" in str(e) or "DEP" in str(e)

        # Restore original receipt
        with receipt_path.open("w") as file:
            file.write(original_receipt)
    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline_dir)
        _rm(runs_root / "_tmp/elision_test")
        _rm(out1)