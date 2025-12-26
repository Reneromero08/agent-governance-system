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


def _receipt_for_pipeline(*, pipeline_dir: Path) -> Dict[str, str]:
    chain = pipeline_dir / "CHAIN.json"
    state = pipeline_dir / "STATE.json"
    spec = pipeline_dir / "PIPELINE.json"
    if not (chain.exists() and state.exists() and spec.exists()):
        raise ValueError("DAG_DEP_MISSING")
    return {
        "PIPELINE.json": _sha256_file(spec),
        "STATE.json": _sha256_file(state),
        "CHAIN.json": _sha256_file(chain),
    }


def _verify_receipt_matches(*, pipeline_dir: Path, expected: Dict[str, Any]) -> None:
    if not isinstance(expected, dict):
        raise ValueError("DAG_RECEIPT_MISMATCH")
    for k in ("PIPELINE.json", "STATE.json", "CHAIN.json"):
        v = expected.get(k)
        if not (isinstance(v, str) and _is_hex64(v)):
            raise ValueError("DAG_RECEIPT_MISMATCH")
    actual = _receipt_for_pipeline(pipeline_dir=pipeline_dir)
    if actual != {k: expected[k] for k in actual.keys()}:
        raise ValueError("DAG_RECEIPT_MISMATCH")


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
        if node in completed:
            try:
                _verify_receipt_matches(pipeline_dir=pipeline_dir, expected=receipts.get(node, {}))
            except Exception as e:
                return {"ok": False, "code": str(e) or "DAG_RECEIPT_MISMATCH", "details": {"phase": "RECEIPT", "node": node}}
        # Verify pipeline itself whenever it is completed.
        if node in completed:
            res = verify_pipeline(project_root=project_root, pipeline_id=node, runs_root=runs_root, strict=strict)
            if not res.get("ok", False):
                return {"ok": False, "code": "DAG_NODE_VERIFY_FAIL", "details": {"node": node, "pipeline_code": res.get("code")}}

    return {"ok": True, "code": "OK", "details": {"dag_id": dag_id, "nodes": len(order), "completed": len(completed)}}


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

        receipt = _receipt_for_pipeline(pipeline_dir=rt.pipeline_dir(node))
        receipts[node] = receipt
        completed.add(node)
        state["completed"] = [n for n in order if n in completed]
        state["receipts"] = dict(sorted(receipts.items(), key=lambda kv: kv[0]))
        _atomic_write_canon_json(dag_state_path, state)
        executed += 1

    return {"ok": True, "code": "OK", "details": {"dag_id": dag_id, "completed": len(completed), "nodes": len(order)}}

