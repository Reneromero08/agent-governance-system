from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from CAPABILITY.PIPELINES.pipeline_runtime import PipelineRuntime, _slug
from CAPABILITY.PIPELINES.pipeline_dag import run_dag, verify_dag, topo_sort, _parse_dag_spec
from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes


def _atomic_write_canon_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(canonical_json_bytes(obj))
    os.replace(tmp, path)


class SwarmRuntime:
    """
    Orchestrates a swarm: a DAG of pipelines with catalytic execution elision.
    Expands swarm specifications into a runnable Pipeline DAG.
    """

    def __init__(self, *, project_root: Path | str, runs_root: Path | str):
        self.project_root = Path(project_root)
        self.runs_root = Path(runs_root)
        self.rt = PipelineRuntime(project_root=self.project_root)
        self.receipts_store = self.runs_root / "_pipelines" / "_swarms" / "_receipts"
        self.receipts_store.mkdir(parents=True, exist_ok=True)

    def _swarm_dir(self, swarm_id: str) -> Path:
        return self.runs_root / "_pipelines" / "_swarms" / _slug(swarm_id)

    def _compute_swarm_hash(self, spec: Dict[str, Any]) -> str:
        """
        Computes a deterministic hash of the swarm definition:
        - Canonical swarm spec
        - Resolved pipeline intents and structure
        """
        # 1. Base spec hash
        hasher = hashlib.sha256()
        hasher.update(canonical_json_bytes(spec))

        # 2. Walk nodes to capture deep dependency hashes (intents, jobspecs)
        # Sort nodes by node_id for stability
        nodes = sorted(spec.get("nodes", []), key=lambda x: x.get("node_id", ""))
        
        for node in nodes:
            node_id = node.get("node_id")
            if not node_id: 
                continue
            
            # Resolve pipeline spec to inspect steps/intents
            p_spec = None
            if "pipeline_spec" in node:
                p_spec = self.rt._parse_spec(pipeline_id=node_id, obj=node["pipeline_spec"])
            elif "pipeline_spec_path" in node:
                path = Path(node["pipeline_spec_path"])
                if not path.is_absolute():
                    path = self.project_root / path
                p_spec = self.rt._parse_spec(pipeline_id=node_id, obj=json.loads(path.read_text(encoding="utf-8")))
            elif "pipeline_id" in node:
                # Referenced existing pipeline - load current spec from disk
                try:
                    p_spec, _ = self.rt.load(pipeline_id=node["pipeline_id"])
                except FileNotFoundError:
                    # If not initialized, checking hash is hard without init. 
                    # Assuming for reuse check we only care if it IS initialized.
                    # Behavior: verification fails if missing, so separate hash update is okay.
                    pass
            
            if p_spec:
                # Hash the pipeline spec components
                for step in p_spec.steps:
                    hasher.update(step.step_id.encode("utf-8"))
                    # Read jobspec to get intent (deep property)
                    try:
                        jobspec = self.rt._load_jobspec(step.jobspec_path)
                        intent = jobspec.get("intent", "")
                        hasher.update(intent.encode("utf-8"))
                        # Also hash durable paths outputs definition, determinism flags, etc
                        hasher.update(canonical_json_bytes(jobspec.get("outputs", {})))
                        hasher.update(canonical_json_bytes(jobspec.get("catalytic_domains", [])))
                    except Exception:
                        # If jobspec missing, hash is unstable/invalid, which is fine (execution will fail later)
                        hasher.update(b"JOBSPEC_MISSING")

        return hasher.hexdigest()

    def _emit_swarm_receipt(self, *, swarm_hash: str, swarm_id: str, dag_state: Dict[str, Any], chain: Dict[str, Any]) -> None:
        """
        Persists a receipt allowing future reuse of this execution.
        """
        receipt = {
            "swarm_receipt_version": "1.0.0",
            "swarm_hash": swarm_hash,
            "swarm_id_ref": swarm_id,
            "dag_state": dag_state,
            "swarm_chain_head": chain.get("head_hash"),
            # We embed full chain links to allow reconstruction without full dag traversal
            "swarm_chain_links": chain.get("links", []) 
        }
        path = self.receipts_store / f"{swarm_hash}.json"
        _atomic_write_canon_json(path, receipt)

    def _try_execution_elision(self, swarm_hash: str, swarm_id: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to skip execution by finding a valid prior receipt.
        Fail-closed: if any artifact is missing or tampered, returns None.
        """
        receipt_path = self.receipts_store / f"{swarm_hash}.json"
        if not receipt_path.exists():
            return None

        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("swarm_receipt_version") != "1.0.0":
                return None
                
            # Verify all referenced pipeline proofs still exist and match
            links = receipt.get("swarm_chain_links", [])
            print(f"DEBUG: Checking {len(links)} links for elision of {swarm_hash}")
            for link in links:
                node_id = link["node_id"]
                expected_hash = link["receipt_hash"]
                
                # Check current pipeline state on disk
                pdir = self.rt.pipeline_dir(node_id)
                current_receipt_path = pdir / "RECEIPT.json"
                if not current_receipt_path.exists():
                    print(f"DEBUG: Missing receipt for {node_id}")
                    return None
                
                current = json.loads(current_receipt_path.read_text(encoding="utf-8"))
                current_hash = current.get("receipt_hash")
                print(f"DEBUG: {node_id} expected={expected_hash[:8]}... current={current_hash[:8]}...")
                
                if current_hash != expected_hash:
                    print(f"DEBUG: Hash mismatch for {node_id}")
                    return None # Tampered or overwritten
                    
            # All good - reconstruct artifacts
            print("DEBUG: Elision verified")
            return receipt
        except Exception:
            return None

    def run(self, *, swarm_id: str, spec_path: Path | str) -> Dict[str, Any]:
        """
        1. Compute swarm equivalence hash.
        2. Check for reuse (elision).
        3. If reuse possible: emit fresh chain/state, return success.
        4. Else: Expand node specs, Init pipelines, Run DAG.
        5. Verify & Emit receipts.
        """
        spec_path = Path(spec_path)
        swarm_dir = self._swarm_dir(swarm_id)
        swarm_dir.mkdir(parents=True, exist_ok=True)

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("swarm_version") != "1.0.0":
            raise ValueError("SWARM_INVALID_VERSION")

        # 1. Compute Hash
        swarm_hash = self._compute_swarm_hash(spec)

        # 2. Reuse Check
        cached = self._try_execution_elision(swarm_hash, swarm_id)
        if cached:
            # ELISION ACTIVATED
            # We must still emit the top-level artifacts for THIS run
            dag_state = cached["dag_state"]
            chain_links = cached["swarm_chain_links"]
            
            # Reconstruct chain object
            swarm_chain = {
                "swarm_id": swarm_id,
                "chain_version": "1.0.0",
                "links": chain_links,
                "head_hash": cached["swarm_chain_head"]
            }
            
            _atomic_write_canon_json(swarm_dir / "SWARM_STATE.json", dag_state)
            _atomic_write_canon_json(swarm_dir / "SWARM_CHAIN.json", swarm_chain)
            
            # Emit a "local" receipt for this run pointing to the hash
            _atomic_write_canon_json(swarm_dir / "SWARM_RECEIPT.json", {
                "swarm_id": swarm_id,
                "elided": True,
                "swarm_hash": swarm_hash,
                "original_receipt_path": str(self.receipts_store / f"{swarm_hash}.json")
            })

            return {"ok": True, "swarm_id": swarm_id, "nodes": len(dag_state.get("order", [])), "elided": True}

        # 3. Normal Execution
        nodes = spec.get("nodes", [])
        edges = spec.get("edges", [])

        dag_nodes = []
        for node in nodes:
            node_id = node.get("node_id")
            dag_nodes.append(node_id)

            # Initialize pipeline if spec is provided
            if "pipeline_spec" in node:
                self.rt.init_from_spec_obj(
                    pipeline_id=node_id, 
                    spec_obj=node["pipeline_spec"]
                )
            elif "pipeline_spec_path" in node:
                p_spec_path = Path(node["pipeline_spec_path"])
                if not p_spec_path.is_absolute():
                    p_spec_path = self.project_root / p_spec_path
                self.rt.init_from_spec_path(pipeline_id=node_id, spec_path=p_spec_path)
            elif "pipeline_id" in node:
                # References existing pipeline
                pdir = self.rt.pipeline_dir(node["pipeline_id"])
                if not pdir.exists():
                    raise FileNotFoundError(f"referenced pipeline missing: {node['pipeline_id']}")
                if node_id != node["pipeline_id"]:
                     raise ValueError("ALIASING_NOT_SUPPORTED_YET")

        # Create standard Pipeline DAG spec
        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": swarm_id,
            "nodes": dag_nodes,
            "edges": edges
        }
        
        # Write DAG spec
        dag_dir = self.runs_root / "_pipelines" / "_dags" / _slug(swarm_id)
        _atomic_write_canon_json(dag_dir / "PIPELINE_DAG.json", dag_spec)
        
        # Run DAG
        dag_res = run_dag(
            project_root=self.project_root,
            runs_root=self.runs_root,
            dag_id=swarm_id
        )
        
        if not dag_res.get("ok", False):
            return dag_res

        # Emit SWARM_STATE.json
        dag_state = json.loads((dag_dir / "DAG_STATE.json").read_text(encoding="utf-8"))
        _atomic_write_canon_json(swarm_dir / "SWARM_STATE.json", dag_state)

        # Emit SWARM_CHAIN.json (top-level chain across pipelines)
        swarm_chain = self._build_swarm_chain(swarm_id=swarm_id, dag_state=dag_state)
        _atomic_write_canon_json(swarm_dir / "SWARM_CHAIN.json", swarm_chain)

        details = dag_res.get("details", {})
        executed = details.get("executed", 0)
        
        # Emit SWARM_RECEIPT.json
        receipt = {
            "swarm_id": swarm_id,
            "nodes": len(dag_nodes),
            "elided": executed == 0,
            "state_hash": hashlib.sha256(canonical_json_bytes(dag_state)).hexdigest(),
            "chain_hash": hashlib.sha256(canonical_json_bytes(swarm_chain)).hexdigest()
        }
        _atomic_write_canon_json(swarm_dir / "SWARM_RECEIPT.json", receipt)
        
        return {
            "ok": True, 
            "swarm_id": swarm_id, 
            "nodes": len(dag_nodes),
            "elided": executed == 0
        }

    def _build_swarm_chain(self, swarm_id: str, dag_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Links pipeline receipts into a single chain.
        """
        order = dag_state.get("order", [])
        completed = dag_state.get("completed", [])
        
        entries = []
        prev_hash = None
        for node_id in order:
            if node_id not in completed:
                continue
            
            pdir = self.rt.pipeline_dir(node_id)
            receipt = json.loads((pdir / "RECEIPT.json").read_text(encoding="utf-8"))
            receipt_hash = receipt["receipt_hash"]
            
            entry = {
                "node_id": node_id,
                "receipt_hash": receipt_hash,
                "prev_link_hash": prev_hash
            }
            
            # Link hash binds node + receipt + prev
            link_payload = canonical_json_bytes(entry)
            link_hash = hashlib.sha256(link_payload).hexdigest()
            entry["link_hash"] = link_hash
            
            entries.append(entry)
            prev_hash = link_hash

        return {
            "swarm_id": swarm_id,
            "chain_version": "1.0.0",
            "links": entries,
            "head_hash": prev_hash
        }
