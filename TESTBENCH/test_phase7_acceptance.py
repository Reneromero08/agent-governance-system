"""
Phase 7 Acceptance Tests

Verifies:
1. A swarm run produces a top-level chain that binds each pipeline's proof
2. Any missing or tampered pipeline proof fails the swarm verification
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.swarm_runtime import SwarmRuntime
from PIPELINES.pipeline_runtime import PipelineRuntime
from PIPELINES.pipeline_dag import verify_dag


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
        "phase": 7,
        "task_type": "pipeline_execution",
        "intent": intent,
        "inputs": {},
        "outputs": {"durable_paths": durable_paths, "validation_criteria": {}},
        "catalytic_domains": catalytic_domains,
        "determinism": "deterministic",
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_swarm_chain_binds_pipeline_proofs(tmp_path: Path) -> None:
    """
    Phase 7 Acceptance 1: A swarm run produces a top-level chain that binds each pipeline's proof.
    """
    swarm_id = "phase7-chain-ok"
    p1 = "phase7-p1"
    p2 = "phase7-p2"
    
    runs_root = REPO_ROOT / "CONTRACTS" / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_pipelines" / "_dags" / swarm_id
    
    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline1_dir = rt_pipeline.pipeline_dir(p1)
    pipeline2_dir = rt_pipeline.pipeline_dir(p2)

    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"
    jobspec1_rel = f"CONTRACTS/_runs/_tmp/phase7_test/{p1}_jobspec.json"
    jobspec2_rel = f"CONTRACTS/_runs/_tmp/phase7_test/{p2}_jobspec.json"

    try:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase7_test")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_jobspec(REPO_ROOT / jobspec1_rel, job_id=f"{p1}-job", intent=p1, catalytic_domains=["CONTRACTS/_runs/_tmp/phase7_test/domain"], durable_paths=[out1])
        _write_jobspec(REPO_ROOT / jobspec2_rel, job_id=f"{p2}-job", intent=p2, catalytic_domains=["CONTRACTS/_runs/_tmp/phase7_test/domain"], durable_paths=[out2])

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
        
        assert res.get("ok") is True, f"Swarm run should succeed: {res}"
        
        # Verify SWARM_CHAIN.json exists and binds pipeline proofs
        chain_path = swarm_dir / "SWARM_CHAIN.json"
        assert chain_path.exists(), "SWARM_CHAIN.json should exist"
        
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        assert chain["swarm_id"] == swarm_id
        assert len(chain["links"]) == 2, "Chain should have 2 links (one per pipeline)"
        
        # Verify chain links bind pipeline receipts
        link1 = chain["links"][0]
        link2 = chain["links"][1]
        
        assert link1["node_id"] == p1
        assert link2["node_id"] == p2
        assert "receipt_hash" in link1
        assert "receipt_hash" in link2
        assert "link_hash" in link1
        assert "link_hash" in link2
        
        # Verify chain structure (link2 references link1)
        assert link2["prev_link_hash"] == link1["link_hash"], "Chain should link p2 to p1"
        assert link1["prev_link_hash"] is None, "First link should have no prev"

    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase7_test")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_tampered_pipeline_proof_fails_verification(tmp_path: Path) -> None:
    """
    Phase 7 Acceptance 2: Any missing or tampered pipeline proof fails the swarm verification.
    """
    swarm_id = "phase7-tamper"
    p1 = "phase7-tamper-p1"
    p2 = "phase7-tamper-p2"
    
    runs_root = REPO_ROOT / "CONTRACTS" / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_pipelines" / "_dags" / swarm_id
    
    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline1_dir = rt_pipeline.pipeline_dir(p1)
    pipeline2_dir = rt_pipeline.pipeline_dir(p2)

    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"
    jobspec1_rel = f"CONTRACTS/_runs/_tmp/phase7_tamper/{p1}_jobspec.json"
    jobspec2_rel = f"CONTRACTS/_runs/_tmp/phase7_tamper/{p2}_jobspec.json"

    try:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase7_tamper")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_jobspec(REPO_ROOT / jobspec1_rel, job_id=f"{p1}-job", intent=p1, catalytic_domains=["CONTRACTS/_runs/_tmp/phase7_tamper/domain"], durable_paths=[out1])
        _write_jobspec(REPO_ROOT / jobspec2_rel, job_id=f"{p2}-job", intent=p2, catalytic_domains=["CONTRACTS/_runs/_tmp/phase7_tamper/domain"], durable_paths=[out2])

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
        
        assert res.get("ok") is True, f"Swarm run should succeed: {res}"
        
        # Tamper with pipeline1's CHAIN.json
        chain_path = pipeline1_dir / "CHAIN.json"
        original_chain = chain_path.read_text()
        chain_path.write_text(original_chain + " TAMPERED")
        
        # Verify DAG - should fail due to tampered proof
        verify_res = verify_dag(
            project_root=REPO_ROOT,
            runs_root=runs_root,
            dag_id=swarm_id,
            strict=True
        )
        
        assert verify_res.get("ok") is False, "Verification should fail on tampered proof"
        assert "RECEIPT" in verify_res.get("code", ""), f"Should fail with RECEIPT error, got: {verify_res}"

    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "phase7_tamper")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)
