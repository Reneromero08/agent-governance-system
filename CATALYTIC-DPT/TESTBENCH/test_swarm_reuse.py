from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.swarm_runtime import SwarmRuntime
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
    
    runs_root = REPO_ROOT / "CONTRACTS" / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_pipelines" / "_dags" / swarm_id
    
    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline_dir = rt_pipeline.pipeline_dir(p1)

    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    jobspec_rel = f"CONTRACTS/_runs/_tmp/elision_test/{p1}_jobspec.json"

    # Also clean up swarm receipt store entries for this hash
    receipts_store = runs_root / "_pipelines" / "_swarms" / "_receipts"

    # Complete cleanup at start
    _rm(swarm_dir)
    _rm(dag_dir)
    _rm(pipeline_dir)
    _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "elision_test")
    _rm(REPO_ROOT / out1)

    unique_intent = "elision-intent-v1-unique"

    try:
        # Setup catalytic domain
        cat_domain = "CONTRACTS/_runs/_tmp/elision_test/cat_domain"
        (REPO_ROOT / cat_domain).mkdir(parents=True, exist_ok=True)

        # Write jobspec
        _write_jobspec(
            REPO_ROOT / jobspec_rel,
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
                    "jobspec_path": jobspec_rel,
                    "memoize": False,
                    "cmd": [
                        sys.executable, "-c",
                        f"from pathlib import Path;Path('{out1}').parent.mkdir(parents=True, exist_ok=True);Path('{out1}').write_text('{p1}')"
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
        res1 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
        assert res1.get("ok") is True, f"First run should succeed: {res1}"
        assert res1.get("elided") is False, "First run should NOT be elided"
        assert (swarm_dir / "SWARM_RECEIPT.json").exists()
        assert (swarm_dir / "SWARM_CHAIN.json").exists()
        
        # Verify chain structure
        chain = json.loads((swarm_dir / "SWARM_CHAIN.json").read_text())
        assert chain["swarm_id"] == swarm_id
        assert len(chain["links"]) == 1
        assert chain["links"][0]["node_id"] == p1
        first_run_head = chain["head_hash"]
        
        # =========================================
        # 2. Second Run: Identical Swarm -> Elision
        # =========================================
        res2 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
        assert res2.get("ok") is True, f"Second run should succeed: {res2}"
        assert res2.get("elided") is True, "Second run MUST be elided"
        
        # Verify receipt indicates elision
        receipt = json.loads((swarm_dir / "SWARM_RECEIPT.json").read_text())
        assert receipt.get("elided") is True
        
        # Verify chain is re-emitted but identical
        chain2 = json.loads((swarm_dir / "SWARM_CHAIN.json").read_text())
        assert chain2["head_hash"] == first_run_head, "Elided chain must reference same proofs"
        
        # =========================================
        # 3. Tamper Test: Mutate pipeline receipt, verify reuse fails
        # =========================================
        pdir = rt_pipeline.pipeline_dir(p1)
        receipt_path = pdir / "RECEIPT.json"
        original_receipt = receipt_path.read_text()
        
        # Tamper: overwrite with invalid receipt
        receipt_path.write_text(json.dumps({"receipt_hash": "0" * 64}))
        
        # Now elision check should fail (hash mismatch), and execution re-attempt should fail too
        # because DAG/pipeline state says "completed" but receipt doesn't match
        
        # The exact failure mode depends on run_dag behavior: 
        # It should either return ok=False or raise ValueError
        try:
            res3 = sr.run(swarm_id=swarm_id, spec_path=spec_path)
            # If it returns, it should be failure
            assert res3.get("ok") is False or res3.get("elided") is False
        except ValueError as e:
            # Expected: DAG verification fails hard
            assert "MISMATCH" in str(e) or "DEP" in str(e)
        
        # Restore original receipt for cleanup idempotency
        receipt_path.write_text(original_receipt)

    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "elision_test")
        _rm(REPO_ROOT / out1)
