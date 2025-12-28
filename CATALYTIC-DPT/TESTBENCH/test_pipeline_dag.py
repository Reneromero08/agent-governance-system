from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_runtime import PipelineRuntime
from PIPELINES.pipeline_dag import restore_dag, verify_dag


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_pipeline_spec(path: Path, *, pipeline_id: str, out_path: str) -> None:
    obj = {
        "pipeline_id": pipeline_id,
        "validator_semver": "0.1.0",
        "validator_build_id": "pipeline-dag-test",
        "timestamp": "CATALYTIC-DPT-02_CONFIG",
        "steps": [
            {
                "step_id": "s1",
                "jobspec_path": f"CONTRACTS/_runs/_tmp/pipeline_dag/{pipeline_id}_jobspec.json",
                "memoize": False,
                "cmd": [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path;"
                        f"Path('{out_path}').parent.mkdir(parents=True, exist_ok=True);"
                        f"Path('{out_path}').write_text('{pipeline_id}', encoding='utf-8')"
                    ),
                ],
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    jobspec = {
        "job_id": f"pipe-dag-{pipeline_id}",
        "phase": 5,
        "task_type": "pipeline_execution",
        "intent": pipeline_id,
        "inputs": {},
        "outputs": {"durable_paths": [out_path], "validation_criteria": {}},
        "catalytic_domains": ["CONTRACTS/_runs/_tmp/pipeline_dag/domain"],
        "determinism": "deterministic",
    }
    jobspec_path = REPO_ROOT / f"CONTRACTS/_runs/_tmp/pipeline_dag/{pipeline_id}_jobspec.json"
    jobspec_path.parent.mkdir(parents=True, exist_ok=True)
    jobspec_path.write_text(json.dumps(jobspec, indent=2), encoding="utf-8")


def _run_catalytic(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO_ROOT / "TOOLS" / "catalytic.py")] + args
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _load_receipt(pipeline_dir: Path) -> dict:
    return json.loads((pipeline_dir / "RECEIPT.json").read_text(encoding="utf-8"))


def _receipt_hash(pipeline_dir: Path) -> str:
    return _load_receipt(pipeline_dir)["receipt_hash"]


def test_receipt_includes_policy_proof(tmp_path: Path) -> None:
    dummy_hash = hashlib.sha256(b"policy-proof").hexdigest()
    policy = {
        "preflight": {
            "verdict": "SAFE",
            "canon_sha256": "a" * 64,
            "cortex_sha256": "b" * 64,
            "git_head_sha": "c" * 64,
            "generated_at": "2025-01-01T00:00:00.000000+00:00",
        },
        "admission": {
            "verdict": "ALLOW",
            "intent_sha256": "d" * 64,
            "mode": "artifact-only",
            "reasons": ["ARTIFACT_ONLY"],
        },
    }

    receipts: List[Dict[str, Any]] = []
    for idx in range(2):
        node_dir = tmp_path / f"policy-node-{idx}"
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "POLICY_PROOF.json").write_bytes(canonical_json_bytes(policy))
        receipt = _emit_receipt(
            pipeline_dir=node_dir,
            node_id=f"node-{idx}",
            pipeline_id=f"pipeline-{idx}",
            capability_hash="PIPELINE_NODE",
            input_artifact_hashes={"in": dummy_hash},
            output_artifact_hashes={"out": dummy_hash},
            prior_receipt_hashes=[],
        )
        receipts.append(receipt)

    assert receipts[0]["policy"] == policy
    assert receipts[1]["policy"] == policy


def test_pipeline_dag_happy_path_and_verify(tmp_path: Path) -> None:
    dag_id = "dag-ok"
    p1 = "dag-p1"
    p2 = "dag-p2"
    rt = PipelineRuntime(project_root=REPO_ROOT)

    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        receipt = _load_receipt(pipeline2_dir)
        assert receipt["node_id"] == p2
        assert receipt["pipeline_id"] == p2
        assert receipt["capability_hash"] == "PIPELINE_NODE"
        assert isinstance(receipt.get("receipt_hash"), str) and len(receipt["receipt_hash"]) == 64

        r_verify = _run_catalytic(["pipeline", "dag", "verify", "--dag-id", dag_id, "--strict"])
        assert r_verify.returncode == 0, r_verify.stdout + r_verify.stderr
        assert "OK dag_id=" in r_verify.stdout
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_resume_does_not_rerun_completed(tmp_path: Path) -> None:
    dag_id = "dag-resume"
    p1 = "dag-rp1"
    p2 = "dag-rp2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_first = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path), "--max-nodes", "1"])
        assert r_first.returncode == 0, r_first.stdout + r_first.stderr

        p1_state_before = (pipeline1_dir / "STATE.json").read_bytes()

        r_second = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id])
        assert r_second.returncode == 0, r_second.stdout + r_second.stderr

        p1_state_after = (pipeline1_dir / "STATE.json").read_bytes()
        assert p1_state_after == p1_state_before
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_cycle_detected(tmp_path: Path) -> None:
    dag_id = "dag-cycle"
    p1 = "dag-c1"
    p2 = "dag-c2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=f"CORTEX/_generated/_tmp/{p1}.txt")
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=f"CORTEX/_generated/_tmp/{p2}.txt")
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [
                {"from": p1, "to": p2, "requires": ["CHAIN.json"]},
                {"from": p2, "to": p1, "requires": ["CHAIN.json"]},
            ],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")
        r = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r.returncode != 0
        assert "DAG_CYCLE_DETECTED" in (r.stdout + r.stderr)
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")


def test_pipeline_dag_tamper_rejected(tmp_path: Path) -> None:
    dag_id = "dag-tamper"
    p1 = "dag-t1"
    p2 = "dag-t2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        # Tamper upstream chain => DAG verify must fail closed.
        chain = pipeline1_dir / "CHAIN.json"
        data = chain.read_bytes()
        chain.write_bytes(data[:-1] + (b"0" if data[-1:] != b"0" else b"1"))

        r_verify = _run_catalytic(["pipeline", "dag", "verify", "--dag-id", dag_id, "--strict"])
        assert r_verify.returncode != 0
        assert "DAG_NODE_VERIFY_FAIL" in (r_verify.stdout + r_verify.stderr) or "DAG_RECEIPT_MISMATCH" in (r_verify.stdout + r_verify.stderr)
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_receipt_chain_mismatch(tmp_path: Path) -> None:
    dag_id = "dag-chain-mismatch"
    p1 = "dag-cm1"
    p2 = "dag-cm2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        # Tamper receipt chain while keeping receipt_hash consistent.
        receipt_path = pipeline2_dir / "RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["prior_receipt_hashes"] = []
        receipt.pop("prior_receipt_hash", None)
        payload = dict(receipt)
        payload.pop("receipt_hash", None)
        import hashlib

        receipt["receipt_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")

        r_verify = _run_catalytic(["pipeline", "dag", "verify", "--dag-id", dag_id, "--strict"])
        assert r_verify.returncode != 0
        assert "RECEIPT_CHAIN_INVALID" in (r_verify.stdout + r_verify.stderr)
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_receipts_portable(tmp_path: Path) -> None:
    dag_id = "dag-portable"
    p1 = "dag-pt1"
    p2 = "dag-pt2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        # Copy artifacts to a fresh runs_root and verify with verify_dag.
        new_root = tmp_path / "runs_root"
        new_dag_dir = new_root / "_pipelines" / "_dags" / dag_id
        new_p1 = new_root / "_pipelines" / p1
        new_p2 = new_root / "_pipelines" / p2
        new_dag_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dag_dir, new_dag_dir)
        shutil.copytree(pipeline1_dir, new_p1)
        shutil.copytree(pipeline2_dir, new_p2)

        # Copy run directories referenced by pipeline state.
        state1 = json.loads((pipeline1_dir / "STATE.json").read_text(encoding="utf-8"))
        state2 = json.loads((pipeline2_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id_1 = state1["step_run_ids"]["s1"]
        run_id_2 = state2["step_run_ids"]["s1"]
        shutil.copytree(REPO_ROOT / "CONTRACTS" / "_runs" / run_id_1, new_root / run_id_1)
        shutil.copytree(REPO_ROOT / "CONTRACTS" / "_runs" / run_id_2, new_root / run_id_2)

        res = verify_dag(project_root=REPO_ROOT, runs_root=new_root, dag_id=dag_id, strict=True)
        assert res.get("ok", False), res
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_restore_noop(tmp_path: Path) -> None:
    dag_id = "dag-restore-noop"
    p1 = "dag-rn1"
    p2 = "dag-rn2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        before_p2 = _receipt_hash(pipeline2_dir)
        _rm(dag_dir / "DAG_STATE.json")

        r_restore = _run_catalytic(["pipeline", "dag", "restore", "--dag-id", dag_id])
        assert r_restore.returncode == 0, r_restore.stdout + r_restore.stderr
        assert _receipt_hash(pipeline2_dir) == before_p2
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_restore_missing_artifact(tmp_path: Path) -> None:
    dag_id = "dag-restore-missing"
    p1 = "dag-rm1"
    p2 = "dag-rm2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        before_p1 = _receipt_hash(pipeline1_dir)
        before_p2 = _receipt_hash(pipeline2_dir)
        _rm(pipeline1_dir / "CHAIN.json")

        r_restore = _run_catalytic(["pipeline", "dag", "restore", "--dag-id", dag_id])
        assert r_restore.returncode == 0, r_restore.stdout + r_restore.stderr
        assert _receipt_hash(pipeline1_dir) != before_p1
        assert _receipt_hash(pipeline2_dir) != before_p2
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_restore_tampered_receipt(tmp_path: Path) -> None:
    dag_id = "dag-restore-tamper"
    p1 = "dag-rt1"
    p2 = "dag-rt2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        before_p1 = _receipt_hash(pipeline1_dir)
        before_p2 = _receipt_hash(pipeline2_dir)

        receipt_path = pipeline1_dir / "RECEIPT.json"
        raw = receipt_path.read_text(encoding="utf-8")
        receipt_path.write_text(raw.replace("PIPELINE_NODE", "PIPELINE_NODE_TAMPER", 1), encoding="utf-8")

        r_restore = _run_catalytic(["pipeline", "dag", "restore", "--dag-id", dag_id])
        assert r_restore.returncode == 0, r_restore.stdout + r_restore.stderr
        assert _receipt_hash(pipeline1_dir) != before_p1
        assert _receipt_hash(pipeline2_dir) != before_p2
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)


def test_pipeline_dag_restore_portable(tmp_path: Path) -> None:
    dag_id = "dag-restore-portable"
    p1 = "dag-rp1"
    p2 = "dag-rp2"
    rt = PipelineRuntime(project_root=REPO_ROOT)
    dag_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / "_dags" / dag_id
    pipeline1_dir = rt.pipeline_dir(p1)
    pipeline2_dir = rt.pipeline_dir(p2)
    spec1 = tmp_path / "p1.json"
    spec2 = tmp_path / "p2.json"
    out1 = f"CORTEX/_generated/_tmp/{p1}.txt"
    out2 = f"CORTEX/_generated/_tmp/{p2}.txt"

    try:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)

        _write_pipeline_spec(spec1, pipeline_id=p1, out_path=out1)
        _write_pipeline_spec(spec2, pipeline_id=p2, out_path=out2)
        rt.init_from_spec_path(pipeline_id=p1, spec_path=spec1)
        rt.init_from_spec_path(pipeline_id=p2, spec_path=spec2)

        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": dag_id,
            "nodes": [p1, p2],
            "edges": [{"from": p1, "to": p2, "requires": ["CHAIN.json"]}],
        }
        dag_spec_path = tmp_path / "dag.json"
        dag_spec_path.write_text(json.dumps(dag_spec, indent=2), encoding="utf-8")

        r_run = _run_catalytic(["pipeline", "dag", "run", "--dag-id", dag_id, "--spec", str(dag_spec_path)])
        assert r_run.returncode == 0, r_run.stdout + r_run.stderr

        new_root = tmp_path / "runs_root"
        new_dag_dir = new_root / "_pipelines" / "_dags" / dag_id
        new_p1 = new_root / "_pipelines" / p1
        new_p2 = new_root / "_pipelines" / p2
        new_dag_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dag_dir, new_dag_dir)
        shutil.copytree(pipeline1_dir, new_p1)
        shutil.copytree(pipeline2_dir, new_p2)

        state1 = json.loads((pipeline1_dir / "STATE.json").read_text(encoding="utf-8"))
        state2 = json.loads((pipeline2_dir / "STATE.json").read_text(encoding="utf-8"))
        run_id_1 = state1["step_run_ids"]["s1"]
        run_id_2 = state2["step_run_ids"]["s1"]
        shutil.copytree(REPO_ROOT / "CONTRACTS" / "_runs" / run_id_1, new_root / run_id_1)
        shutil.copytree(REPO_ROOT / "CONTRACTS" / "_runs" / run_id_2, new_root / run_id_2)

        res = restore_dag(project_root=REPO_ROOT, runs_root=new_root, dag_id=dag_id, strict=True)
        assert res.get("ok", False), res
    finally:
        _rm(dag_dir)
        _rm(pipeline1_dir)
        _rm(pipeline2_dir)
        _rm(REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "pipeline_dag")
        _rm(REPO_ROOT / out1)
        _rm(REPO_ROOT / out2)
