from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIPELINES.pipeline_runtime import PipelineRuntime, _slug
from PIPELINES.pipeline_verify import verify_pipeline
from PRIMITIVES.restore_proof import canonical_json_bytes


_HEX64 = "0123456789abcdef"


def _is_hex64(s: str) -> bool:
    return len(s) == 64 and all(ch in _HEX64 for ch in s)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json_obj(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_canon_json(path: Path, obj: Any) -> None:
    _atomic_write(path, canonical_json_bytes(obj))


def _load_policy_proof(pipeline_dir: Path) -> Dict[str, Any]:
    path = pipeline_dir / "POLICY_PROOF.json"
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


@dataclass(frozen=True)
class DagEdge:
    src: str
    dst: str
    requires: List[str]


@dataclass(frozen=True)
class DagSpec:
    dag_id: str
    nodes: List[str]  # pipeline_ids
    edges: List[DagEdge]


def _validate_artifact_name(name: str) -> None:
    # Allow only a small stable set for now; unknown deps are rejected.
    allowed = {"PIPELINE.json", "STATE.json", "CHAIN.json"}
    if name not in allowed:
        raise ValueError("DAG_INVALID_ARTIFACT_DEP")


def _parse_dag_spec(obj: Dict[str, Any]) -> DagSpec:
    if obj.get("dag_version") != "1.0.0":
        raise ValueError("DAG_INVALID_VERSION")
    dag_id = obj.get("dag_id")
    if not isinstance(dag_id, str) or not dag_id.strip():
        raise ValueError("DAG_INVALID_ID")

    nodes = obj.get("nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(x, str) and x.strip() for x in nodes):
        raise ValueError("DAG_INVALID_NODES")
    if len(set(nodes)) != len(nodes):
        raise ValueError("DAG_DUPLICATE_NODE")

    edges_obj = obj.get("edges", [])
    if not isinstance(edges_obj, list):
        raise ValueError("DAG_INVALID_EDGES")
    edges: List[DagEdge] = []
    for raw in edges_obj:
        if not isinstance(raw, dict):
            raise ValueError("DAG_INVALID_EDGES")
        src = raw.get("from")
        dst = raw.get("to")
        requires = raw.get("requires", [])
        if not (isinstance(src, str) and src in nodes):
            raise ValueError("DAG_INVALID_EDGE")
        if not (isinstance(dst, str) and dst in nodes):
            raise ValueError("DAG_INVALID_EDGE")
        if not isinstance(requires, list) or not requires or not all(isinstance(x, str) and x for x in requires):
            raise ValueError("DAG_INVALID_EDGE")
        for r in requires:
            _validate_artifact_name(r)
        edges.append(DagEdge(src=src, dst=dst, requires=list(requires)))

    allowed_top = {"dag_version", "dag_id", "nodes", "edges"}
    if set(obj.keys()) != allowed_top:
        raise ValueError("DAG_INVALID_FIELDS")

    return DagSpec(dag_id=dag_id, nodes=list(nodes), edges=edges)


def topo_sort(spec: DagSpec) -> List[str]:
    """
    Deterministic topological sort.
    - Tie-breaks by lexicographic node id.
    """
    outgoing: Dict[str, List[str]] = {n: [] for n in spec.nodes}
    indeg: Dict[str, int] = {n: 0 for n in spec.nodes}
    for e in spec.edges:
        outgoing[e.src].append(e.dst)
        indeg[e.dst] += 1
    for k in outgoing.keys():
        outgoing[k] = sorted(outgoing[k])

    ready = sorted([n for n, d in indeg.items() if d == 0])
    out: List[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in outgoing[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
                ready.sort()

    if len(out) != len(spec.nodes):
        raise ValueError("DAG_CYCLE_DETECTED")
    return out


def _dag_dir(*, project_root: Path, runs_root: Path, dag_id: str) -> Path:
    return runs_root / "_pipelines" / "_dags" / _slug(dag_id)


def _load_state(path: Path, *, nodes: List[str]) -> Dict[str, Any]:
    if not path.exists():
        return {
            "state_version": "1.0.0",
            "dag_id": None,
            "order": nodes,
            "completed": [],
            "receipts": {},
        }
    obj = _load_json_obj(path)
    if obj.get("state_version") != "1.0.0":
        raise ValueError("DAG_STATE_INVALID_VERSION")
    completed = obj.get("completed")
    receipts = obj.get("receipts")
    if not isinstance(completed, list) or not all(isinstance(x, str) and x for x in completed):
        raise ValueError("DAG_STATE_INVALID")
    if not isinstance(receipts, dict):
        raise ValueError("DAG_STATE_INVALID")
    return obj


def _pipeline_artifact_hashes(*, pipeline_dir: Path, include_receipt: bool = False) -> Dict[str, str]:
    chain = pipeline_dir / "CHAIN.json"
    state = pipeline_dir / "STATE.json"
    spec = pipeline_dir / "PIPELINE.json"
    if not (chain.exists() and state.exists() and spec.exists()):
        raise ValueError("DAG_DEP_MISSING")
    out = {
        "PIPELINE.json": _sha256_file(spec),
        "STATE.json": _sha256_file(state),
        "CHAIN.json": _sha256_file(chain),
    }
    if include_receipt:
        receipt = pipeline_dir / "RECEIPT.json"
        if not receipt.exists():
            raise ValueError("RECEIPT_MISSING")
        out["RECEIPT.json"] = _sha256_file(receipt)
    return out


def _load_restore_report(path: Path) -> Dict[str, Any]:
    obj = _load_json_obj(path)
    if obj.get("restore_version") != "1.0.0":
        raise ValueError("DAG_RESTORE_INVALID")
    return obj


def _verify_receipt_matches(*, pipeline_dir: Path, expected: Dict[str, Any]) -> None:
    if not isinstance(expected, dict):
        raise ValueError("DAG_RECEIPT_MISMATCH")
    for k in ("PIPELINE.json", "STATE.json", "CHAIN.json"):
        v = expected.get(k)
        if not (isinstance(v, str) and _is_hex64(v)):
            raise ValueError("DAG_RECEIPT_MISMATCH")
    # Enforce strict receipt integrity: include RECEIPT.json in hash check
    actual = _pipeline_artifact_hashes(pipeline_dir=pipeline_dir, include_receipt=True)
    
    # If expected doesn't have RECEIPT.json (legacy), fallback to partial check? 
    # No, strict mode requires it. But for upgrade safety during dev, we might conditionally check.
    # However, for this fix, we simply check whatever actual returns.
    
    for k, v in actual.items():
        print(f"DEBUG: Check {k}: Exp={expected.get(k)} Act={v} Pdir={pipeline_dir}")
        if expected.get(k) != v:
            # This covers mismatch AND missing key in expected
             raise ValueError("DAG_RECEIPT_MISMATCH")


def _receipt_executor_id() -> str:
    env = os.environ.get("CATALYTIC_EXECUTOR_ID")
    return env if isinstance(env, str) and env.strip() else "CATALYTIC-DPT-EXECUTOR"


def _compute_receipt_hash(receipt_obj: Dict[str, Any]) -> str:
    payload = dict(receipt_obj)
    payload.pop("receipt_hash", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_receipt(path: Path) -> Dict[str, Any]:
    obj = _load_json_obj(path)
    if not isinstance(obj.get("receipt_hash"), str):
        raise ValueError("RECEIPT_INVALID")
    return obj


def _emit_receipt(
    *,
    pipeline_dir: Path,
    node_id: str,
    pipeline_id: str,
    capability_hash: str,
    input_artifact_hashes: Dict[str, str],
    output_artifact_hashes: Dict[str, str],
    prior_receipt_hashes: List[str],
) -> Dict[str, Any]:
    policy = _load_policy_proof(pipeline_dir)
    receipt = {
        "node_id": node_id,
        "pipeline_id": pipeline_id,
        "capability_hash": capability_hash,
        "input_artifact_hashes": dict(sorted(input_artifact_hashes.items(), key=lambda kv: kv[0])),
        "output_artifact_hashes": dict(sorted(output_artifact_hashes.items(), key=lambda kv: kv[0])),
        "executor_id": _receipt_executor_id(),
        "prior_receipt_hashes": list(sorted(prior_receipt_hashes)),
    }
    if len(prior_receipt_hashes) == 1:
        receipt["prior_receipt_hash"] = prior_receipt_hashes[0]
    if policy:
        receipt["policy"] = policy
    receipt["receipt_hash"] = _compute_receipt_hash(receipt)
    _atomic_write_canon_json(pipeline_dir / "RECEIPT.json", receipt)
    return receipt


def _expected_inputs_and_prior(
    *,
    rt: PipelineRuntime,
    deps: Dict[str, List[DagEdge]],
    node: str,
    completed: Set[str],
) -> Tuple[Dict[str, str], List[str]]:
    expected_inputs: Dict[str, str] = {}
    expected_prior: List[str] = []
    for e in deps.get(node, []):
        if e.src not in completed:
            raise ValueError("DAG_DEP_MISSING")
        src_dir = rt.pipeline_dir(e.src)
        for req in e.requires:
            expected_inputs[f"{e.src}:{req}"] = _sha256_file(src_dir / req)
        src_receipt = _load_receipt(src_dir / "RECEIPT.json")
        src_hash = src_receipt.get("receipt_hash")
        if not (isinstance(src_hash, str) and _is_hex64(src_hash)):
            raise ValueError("RECEIPT_CHAIN_INVALID")
        expected_prior.append(src_hash)
    return expected_inputs, expected_prior


def _should_skip_node(
    *,
    rt: PipelineRuntime,
    deps: Dict[str, List[DagEdge]],
    node: str,
    completed: Set[str],
) -> bool:
    try:
        expected_inputs, expected_prior = _expected_inputs_and_prior(rt=rt, deps=deps, node=node, completed=completed)
        _verify_receipt(
            pipeline_dir=rt.pipeline_dir(node),
            node_id=node,
            pipeline_id=node,
            expected_inputs=expected_inputs,
            expected_outputs=_pipeline_artifact_hashes(pipeline_dir=rt.pipeline_dir(node)),
            expected_prior=expected_prior,
        )
        return True
    except Exception:
        return False


def _verify_receipt(
    *,
    pipeline_dir: Path,
    node_id: str,
    pipeline_id: str,
    expected_inputs: Dict[str, str],
    expected_outputs: Dict[str, str],
    expected_prior: List[str],
) -> None:
    path = pipeline_dir / "RECEIPT.json"
    if not path.exists():
        raise ValueError("RECEIPT_MISSING")
    receipt = _load_receipt(path)
    if receipt.get("node_id") != node_id or receipt.get("pipeline_id") != pipeline_id:
        raise ValueError("RECEIPT_INVALID")
    if receipt.get("capability_hash") != "PIPELINE_NODE":
        raise ValueError("RECEIPT_INVALID")
    computed = _compute_receipt_hash(receipt)
    if receipt.get("receipt_hash") != computed:
        raise ValueError("RECEIPT_HASH_MISMATCH")

    inp = receipt.get("input_artifact_hashes")
    out = receipt.get("output_artifact_hashes")
    if not isinstance(inp, dict) or not isinstance(out, dict):
        raise ValueError("RECEIPT_INVALID")
    if not all(isinstance(v, str) and _is_hex64(v) for v in inp.values()):
        raise ValueError("RECEIPT_INVALID")
    if not all(isinstance(v, str) and _is_hex64(v) for v in out.values()):
        raise ValueError("RECEIPT_INVALID")
    if dict(sorted(inp.items(), key=lambda kv: kv[0])) != dict(sorted(expected_inputs.items(), key=lambda kv: kv[0])):
        raise ValueError("RECEIPT_INPUT_MISMATCH")
    if dict(sorted(out.items(), key=lambda kv: kv[0])) != dict(sorted(expected_outputs.items(), key=lambda kv: kv[0])):
        raise ValueError("RECEIPT_OUTPUT_MISMATCH")

    prior = receipt.get("prior_receipt_hashes", [])
    if not isinstance(prior, list) or not all(isinstance(x, str) and _is_hex64(x) for x in prior):
        raise ValueError("RECEIPT_CHAIN_INVALID")
    if sorted(prior) != sorted(expected_prior):
        raise ValueError("RECEIPT_CHAIN_INVALID")
    if len(expected_prior) == 1:
        if receipt.get("prior_receipt_hash") != expected_prior[0]:
            raise ValueError("RECEIPT_CHAIN_INVALID")


def verify_dag(
    *,
    project_root: Path,
    runs_root: Path,
    dag_id: str,
    strict: bool = True,
) -> Dict[str, Any]:
    dag_dir = _dag_dir(project_root=project_root, runs_root=runs_root, dag_id=dag_id)
    spec_path = dag_dir / "PIPELINE_DAG.json"
    state_path = dag_dir / "DAG_STATE.json"
    if not spec_path.exists():
        return {"ok": False, "code": "DAG_NOT_FOUND", "details": {"dag_dir": str(dag_dir)}}
    try:
        spec = _parse_dag_spec(_load_json_obj(spec_path))
        order = topo_sort(spec)
        state = _load_state(state_path, nodes=order)
    except Exception as e:
        return {"ok": False, "code": str(e) or "DAG_INVALID", "details": {"phase": "LOAD"}}

    completed = state.get("completed", [])
    receipts = state.get("receipts", {})
    if not isinstance(completed, list) or not isinstance(receipts, dict):
        return {"ok": False, "code": "DAG_STATE_INVALID", "details": {"phase": "STATE"}}

    rt = PipelineRuntime(project_root=project_root)
    for node in order:
        pipeline_dir = rt.pipeline_dir(node)
        # Build expected input hashes and prior receipts from DAG edges.
        expected_inputs: Dict[str, str] = {}
        expected_prior: List[str] = []
        for e in spec.edges:
            if e.dst != node:
                continue
            src_dir = rt.pipeline_dir(e.src)
            for req in e.requires:
                expected_inputs[f"{e.src}:{req}"] = _sha256_file(src_dir / req)
            src_receipt = src_dir / "RECEIPT.json"
            if not src_receipt.exists():
                return {"ok": False, "code": "RECEIPT_MISSING", "details": {"phase": "RECEIPT", "node": node}}
            src_obj = _load_receipt(src_receipt)
            src_hash = src_obj.get("receipt_hash")
            if not (isinstance(src_hash, str) and _is_hex64(src_hash)):
                return {"ok": False, "code": "RECEIPT_CHAIN_INVALID", "details": {"phase": "RECEIPT", "node": node}}
            expected_prior.append(src_hash)
        if node in completed:
            try:
                _verify_receipt_matches(pipeline_dir=pipeline_dir, expected=receipts.get(node, {}))
            except Exception as e:
                return {"ok": False, "code": str(e) or "DAG_RECEIPT_MISMATCH", "details": {"phase": "RECEIPT", "node": node}}
            try:
                _verify_receipt(
                    pipeline_dir=pipeline_dir,
                    node_id=node,
                    pipeline_id=node,
                    expected_inputs=expected_inputs,
                    expected_outputs=_pipeline_artifact_hashes(pipeline_dir=pipeline_dir),
                    expected_prior=expected_prior,
                )
            except Exception as e:
                return {"ok": False, "code": str(e) or "RECEIPT_INVALID", "details": {"phase": "RECEIPT", "node": node}}
        # Verify pipeline itself whenever it is completed.
        if node in completed:
            res = verify_pipeline(project_root=project_root, pipeline_id=node, runs_root=runs_root, strict=strict)
            if not res.get("ok", False):
                return {"ok": False, "code": "DAG_NODE_VERIFY_FAIL", "details": {"node": node, "pipeline_code": res.get("code")}}

    restore_path = dag_dir / "DAG_RESTORE.json"
    if restore_path.exists():
        try:
            restore = _load_restore_report(restore_path)
            if restore.get("dag_id") != dag_id:
                raise ValueError("DAG_RESTORE_INVALID")
            skipped = restore.get("skipped", [])
            rerun = restore.get("rerun", [])
            if not (isinstance(skipped, list) and isinstance(rerun, list)):
                raise ValueError("DAG_RESTORE_INVALID")
            if not all(isinstance(x, str) and x for x in skipped + rerun):
                raise ValueError("DAG_RESTORE_INVALID")
            if len(set(skipped + rerun)) != len(skipped) + len(rerun):
                raise ValueError("DAG_RESTORE_INVALID")
            if set(skipped + rerun) != set(completed):
                raise ValueError("DAG_RESTORE_INVALID")
            receipt_hashes = restore.get("receipt_hashes", {})
            if not isinstance(receipt_hashes, dict):
                raise ValueError("DAG_RESTORE_INVALID")
            for node in completed:
                r_path = rt.pipeline_dir(node) / "RECEIPT.json"
                r_obj = _load_receipt(r_path)
                r_hash = r_obj.get("receipt_hash")
                if not (isinstance(r_hash, str) and _is_hex64(r_hash)):
                    raise ValueError("DAG_RESTORE_INVALID")
                if receipt_hashes.get(node) != r_hash:
                    raise ValueError("DAG_RESTORE_INVALID")
        except Exception as e:
            return {"ok": False, "code": str(e) or "DAG_RESTORE_INVALID", "details": {"phase": "RESTORE"}}

    return {"ok": True, "code": "OK", "details": {"dag_id": dag_id, "nodes": len(order), "completed": len(completed)}}


def restore_dag(
    *,
    project_root: Path,
    runs_root: Path,
    dag_id: str,
    strict: bool = True,
) -> Dict[str, Any]:
    dag_dir = _dag_dir(project_root=project_root, runs_root=runs_root, dag_id=dag_id)
    dag_spec_path = dag_dir / "PIPELINE_DAG.json"
    dag_state_path = dag_dir / "DAG_STATE.json"
    restore_path = dag_dir / "DAG_RESTORE.json"
    if not dag_spec_path.exists():
        return {"ok": False, "code": "DAG_NOT_FOUND", "details": {"dag_dir": str(dag_dir)}}
    try:
        dag_spec = _parse_dag_spec(_load_json_obj(dag_spec_path))
        order = topo_sort(dag_spec)
    except Exception as e:
        return {"ok": False, "code": str(e) or "DAG_INVALID", "details": {"phase": "LOAD"}}

    deps: Dict[str, List[DagEdge]] = {n: [] for n in order}
    for e in dag_spec.edges:
        deps[e.dst].append(e)
    for k in deps.keys():
        deps[k] = sorted(deps[k], key=lambda x: (x.src, ",".join(x.requires)))

    rt = PipelineRuntime(project_root=project_root)
    completed: Set[str] = set()
    receipts: Dict[str, Any] = {}
    skipped: List[str] = []
    rerun: List[str] = []

    # Determine which nodes can be skipped based on valid receipts + artifacts.
    for node in order:
        if _should_skip_node(rt=rt, deps=deps, node=node, completed=completed):
            completed.add(node)
            skipped.append(node)
            receipts[node] = _pipeline_artifact_hashes(pipeline_dir=rt.pipeline_dir(node), include_receipt=True)

    # Re-run incomplete nodes deterministically in topo order.
    for node in order:
        if node in completed:
            continue
        try:
            expected_inputs, expected_prior = _expected_inputs_and_prior(rt=rt, deps=deps, node=node, completed=completed)
        except Exception:
            return {"ok": False, "code": "RESTORE_AMBIGUOUS_STATE", "details": {"node": node}}

        pipeline_dir = rt.pipeline_dir(node)
        spec_path = pipeline_dir / "PIPELINE.json"
        if not spec_path.exists():
            return {"ok": False, "code": "PIPELINE_NOT_FOUND", "details": {"node": node}}
        spec_obj = _load_json_obj(spec_path)
        spec_obj = dict(spec_obj)
        spec_obj["pipeline_id"] = node
        try:
            spec = rt._parse_spec(pipeline_id=node, obj=spec_obj)
        except Exception:
            return {"ok": False, "code": "PIPELINE_SPEC_INVALID", "details": {"node": node}}
        _atomic_write_canon_json(pipeline_dir / "STATE.json", rt._initial_state(spec))

        rt.run(pipeline_id=node)
        res = verify_pipeline(project_root=project_root, pipeline_id=node, runs_root=runs_root, strict=strict)
        if not res.get("ok", False):
            return {"ok": False, "code": "DAG_NODE_VERIFY_FAIL", "details": {"node": node, "pipeline_code": res.get("code")}}

        output_hashes = _pipeline_artifact_hashes(pipeline_dir=pipeline_dir)
        _emit_receipt(
            pipeline_dir=pipeline_dir,
            node_id=node,
            pipeline_id=node,
            capability_hash="PIPELINE_NODE",
            input_artifact_hashes=expected_inputs,
            output_artifact_hashes=output_hashes,
            prior_receipt_hashes=expected_prior,
        )
        receipts[node] = _pipeline_artifact_hashes(pipeline_dir=pipeline_dir, include_receipt=True)
        completed.add(node)
        rerun.append(node)

    state = {
        "state_version": "1.0.0",
        "dag_id": dag_spec.dag_id,
        "order": order,
        "completed": [n for n in order if n in completed],
        "receipts": dict(sorted(receipts.items(), key=lambda kv: kv[0])),
    }
    _atomic_write_canon_json(dag_state_path, state)

    receipt_hashes: Dict[str, str] = {}
    for node in completed:
        rec = _load_receipt(rt.pipeline_dir(node) / "RECEIPT.json")
        r_hash = rec.get("receipt_hash")
        if not (isinstance(r_hash, str) and _is_hex64(r_hash)):
            return {"ok": False, "code": "RECEIPT_INVALID", "details": {"node": node}}
        receipt_hashes[node] = r_hash
    restore_obj = {
        "restore_version": "1.0.0",
        "dag_id": dag_spec.dag_id,
        "skipped": [n for n in order if n in skipped],
        "rerun": [n for n in order if n in rerun],
        "receipt_hashes": dict(sorted(receipt_hashes.items(), key=lambda kv: kv[0])),
    }
    _atomic_write_canon_json(restore_path, restore_obj)

    return {
        "ok": True,
        "code": "OK",
        "details": {"dag_id": dag_id, "skipped": len(skipped), "rerun": len(rerun), "completed": len(completed)},
    }


def run_dag(
    *,
    project_root: Path,
    runs_root: Path,
    dag_id: str,
    spec_path: Optional[Path] = None,
    max_nodes: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    dag_dir = _dag_dir(project_root=project_root, runs_root=runs_root, dag_id=dag_id)
    dag_dir.mkdir(parents=True, exist_ok=True)
    dag_spec_path = dag_dir / "PIPELINE_DAG.json"
    dag_state_path = dag_dir / "DAG_STATE.json"

    if spec_path is not None:
        obj = _load_json_obj(spec_path)
        spec = _parse_dag_spec(obj)
        _atomic_write_canon_json(dag_spec_path, obj)
    else:
        if not dag_spec_path.exists():
            return {"ok": False, "code": "DAG_NOT_FOUND", "details": {"dag_dir": str(dag_dir)}}
        spec = _parse_dag_spec(_load_json_obj(dag_spec_path))

    order = topo_sort(spec)
    state = _load_state(dag_state_path, nodes=order)
    state["dag_id"] = spec.dag_id
    state["order"] = order
    state.setdefault("completed", [])
    state.setdefault("receipts", {})
    completed: Set[str] = set(state["completed"])
    receipts: Dict[str, Any] = state["receipts"]
    if not isinstance(receipts, dict):
        return {"ok": False, "code": "DAG_STATE_INVALID", "details": {"phase": "STATE"}}

    # Fail closed if any completed receipt no longer matches current artifacts.
    rt = PipelineRuntime(project_root=project_root)
    for node in sorted(completed):
        pipeline_dir = rt.pipeline_dir(node)
        _verify_receipt_matches(pipeline_dir=pipeline_dir, expected=receipts.get(node, {}))

    executed = 0
    # Pre-compute dependency map
    deps: Dict[str, List[DagEdge]] = {n: [] for n in order}
    for e in spec.edges:
        deps[e.dst].append(e)
    for k in deps.keys():
        deps[k] = sorted(deps[k], key=lambda x: (x.src, ",".join(x.requires)))

    for node in order:
        if node in completed:
            continue
        if max_nodes is not None and executed >= max_nodes:
            break

        # Ensure dependencies are completed and their required artifacts are present + receipt-matching.
        for e in deps.get(node, []):
            if e.src not in completed:
                return {"ok": False, "code": "DAG_DEP_MISSING", "details": {"node": node, "missing": e.src}}
            src_dir = rt.pipeline_dir(e.src)
            exp = receipts.get(e.src, {})
            _verify_receipt_matches(pipeline_dir=src_dir, expected=exp)
            for req in e.requires:
                if not (src_dir / req).exists():
                    return {"ok": False, "code": "DAG_DEP_MISSING", "details": {"node": node, "missing_artifact": req, "src": e.src}}

        # Run and verify the pipeline node (artifact-only).
        rt.run(pipeline_id=node)
        res = verify_pipeline(project_root=project_root, pipeline_id=node, runs_root=runs_root, strict=strict)
        if not res.get("ok", False):
            return {"ok": False, "code": "DAG_NODE_VERIFY_FAIL", "details": {"node": node, "pipeline_code": res.get("code")}}

        pipeline_dir = rt.pipeline_dir(node)
        input_hashes: Dict[str, str] = {}
        prior_hashes: List[str] = []
        for e in deps.get(node, []):
            src_dir = rt.pipeline_dir(e.src)
            for req in e.requires:
                input_hashes[f"{e.src}:{req}"] = _sha256_file(src_dir / req)
            src_receipt = _load_receipt(src_dir / "RECEIPT.json")
            src_hash = src_receipt.get("receipt_hash")
            if not (isinstance(src_hash, str) and _is_hex64(src_hash)):
                return {"ok": False, "code": "RECEIPT_CHAIN_INVALID", "details": {"node": node}}
            prior_hashes.append(src_hash)

        output_hashes = _pipeline_artifact_hashes(pipeline_dir=pipeline_dir)
        _emit_receipt(
            pipeline_dir=pipeline_dir,
            node_id=node,
            pipeline_id=node,
            capability_hash="PIPELINE_NODE",
            input_artifact_hashes=input_hashes,
            output_artifact_hashes=output_hashes,
            prior_receipt_hashes=prior_hashes,
        )

        receipt = _pipeline_artifact_hashes(pipeline_dir=pipeline_dir, include_receipt=True)
        receipts[node] = receipt
        completed.add(node)
        state["completed"] = [n for n in order if n in completed]
        state["receipts"] = dict(sorted(receipts.items(), key=lambda kv: kv[0]))
        _atomic_write_canon_json(dag_state_path, state)
        executed += 1

    return {"ok": True, "code": "OK", "details": {"dag_id": dag_id, "completed": len(completed), "nodes": len(order), "executed": executed}}
