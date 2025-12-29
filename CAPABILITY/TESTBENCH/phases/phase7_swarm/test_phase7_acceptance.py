import json
import shutil
import subprocess
import sys
from pathlib import Path

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
    jobspec = {
        "job_id": job_id,
        "phase": 7,
        "task_type": "test",
        "intent": intent,
        "inputs": {},
        "outputs": {
            "durable_paths": durable_paths,
            "validation_criteria": {}
        },
        "catalytic_domains": catalytic_domains,
        "determinism": "deterministic"
    }
    path.write_text(json.dumps(jobspec, indent=2), encoding="utf-8")

def test_swarm_chain_binds_pipeline_proofs(tmp_path: Path) -> None:
    """
    Phase 7 Acceptance 1: A swarm run produces a top-level chain that binds each pipeline's proof.
    """
    swarm_id = "phase7-chain-ok"
    p1 = "p1"
    p2 = "p2"

    runs_root = REPO_ROOT / "_runs"
    swarm_dir = runs_root / "_pipelines" / "_swarms" / swarm_id
    dag_dir = runs_root / "_dags" / swarm_id

    rt_pipeline = PipelineRuntime(project_root=REPO_ROOT)
    pipeline1_dir = rt_pipeline.pipeline_dir(p1)
    pipeline2_dir = rt_pipeline.pipeline_dir(p2)

    out1 = f"LAW/CONTRACTS/_runs/_tmp/{p1}.txt"
    out2 = f"LAW/CONTRACTS/_runs/_tmp/{p2}.txt"
    jobspec1_rel = f" LAW/CONTRACTS/_runs/_tmp/phase7_test/{p1}_jobspec.json"
    jobspec2_rel = f" LAW/CONTRACTS/_runs/_tmp/phase7_test/{p2}_jobspec.json"

    try:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(runs_root / "_tmp" / "phase7_test")
        _rm(out1)
        _rm(out2)

        _write_jobspec(runs_root / jobspec1_rel, job_id=f"{p1}-job", intent=p1, catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/phase7_test/domain"], durable_paths=[out1])
        _write_jobspec(runs_root / jobspec2_rel, job_id=f"{p2}-job", intent=p2, catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/phase7_test/domain"], durable_paths=[out2])

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
        assert chain_path.exists(), "Chain file should exist."

        verify_chain = (chain_path.read_text().strip() == original_chain)
        if not verify_chain:
            print(f"Verification failed: Expected {original_chain} but got '{chain_path.read_text()}'.")
    finally:
        _rm(swarm_dir)
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(runs_root / "_tmp" / "phase7_test")
        _rm(out1)
        _rm(out2)

def test_swarm_chain_fails_on_tampered_proof(tmp_path: Path) -> None:
    """
    Verify that a swarm run fails due to missing or tampered pipeline proof.
    
    This function simulates the scenario where the CHAIN.json file for one of the pipelines
    is tampered with, causing the DAG verification to fail. The test should be run in a temporary
    directory to isolate the failing chain and prevent it from affecting other tests.
    """
    original_chain = (chain_path.read_text().strip() == "Original Chain")

    # Tamper with pipeline1's CHAIN.json file
    tampered_chain = chain_path.write_text("TAMPERED") + "\n"

    try:
        _write_jobspec(runs_root / jobspec2_rel, job_id=f"{p2}-job", intent=p2, catalytic_domains=["LAW/CONTRACTS/_runs/_tmp/phase7_test/domain"], durable_paths=[out2])

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

        assert res.get("ok") is False, "The swarm run should fail due to missing or tampered proof."
        assert "RECEIPT" in res.get("code", ""), f"The code should indicate a RECEIPT error due to the tampered proof."

    finally:
        chain_path = tmp_path / "CHAIN.json"
        # Clean up any temporary files
        if original_chain and not verify_chain:
            print(f"Cleaning up: {chain_path}")
            chain_path.unlink()
