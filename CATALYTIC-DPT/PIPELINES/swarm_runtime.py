from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIPELINES.pipeline_runtime import PipelineRuntime, _slug
from PIPELINES.pipeline_dag import run_dag, verify_dag, topo_sort, _parse_dag_spec
from PRIMITIVES.restore_proof import canonical_json_bytes


def _atomic_write_canon_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(canonical_json_bytes(obj))
    os.replace(tmp, path)


class SwarmRuntime:
    """
    Orchestrates a swarm: a DAG of pipelines.
    Expands swarm specifications into a runnable Pipeline DAG.
    """

    def __init__(self, *, project_root: Path, runs_root: Path):
        self.project_root = project_root
        self.runs_root = runs_root
        self.rt = PipelineRuntime(project_root=project_root)

    def _swarm_dir(self, swarm_id: str) -> Path:
        return self.runs_root / "_pipelines" / "_swarms" / _slug(swarm_id)

    def run(self, *, swarm_id: str, spec_path: Path) -> Dict[str, Any]:
        """
        1. Load and validate swarm spec.
        2. Expand node specs into initialized pipelines.
        3. Convert swarm spec into a standard Pipeline DAG spec.
        4. Execute the DAG.
        5. Emit SWARM_STATE.json and SWARM_CHAIN.json.
        """
        swarm_dir = self._swarm_dir(swarm_id)
        swarm_dir.mkdir(parents=True, exist_ok=True)

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("swarm_version") != "1.0.0":
            raise ValueError("SWARM_INVALID_VERSION")

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
                # References existing pipeline, ensure it exists
                pdir = self.rt.pipeline_dir(node["pipeline_id"])
                if not pdir.exists():
                    raise FileNotFoundError(f"referenced pipeline missing: {node['pipeline_id']}")
                # If node_id != pipeline_id, we might need a symlink or alias.
                # For now, we assume node_id IS the pipeline_id if no spec provided.
                if node_id != node["pipeline_id"]:
                     raise ValueError("ALIASING_NOT_SUPPORTED_YET")

        # Create standard Pipeline DAG spec
        dag_spec = {
            "dag_version": "1.0.0",
            "dag_id": swarm_id,
            "nodes": dag_nodes,
            "edges": edges
        }
        
        # Run the DAG using pipeline_dag logic
        dag_res = run_dag(
            project_root=self.project_root,
            runs_root=self.runs_root,
            dag_id=swarm_id,
            spec_path=None # We'll write it manually to swarm_dir later if needed
        )
        # Actually run_dag expects the spec at _dag_dir/PIPELINE_DAG.json
        # But we want it in swarm_dir.
        
        # Let's fix the run_dag call to point to the right place or just implement it here.
        # For Phase 7, we'll implement a clean Swarm execution that wraps DAG.
        
        # For now, we will write the DAG spec to the expected location for run_dag
        dag_dir = self.runs_root / "_pipelines" / "_dags" / _slug(swarm_id)
        _atomic_write_canon_json(dag_dir / "PIPELINE_DAG.json", dag_spec)
        
        dag_res = run_dag(
            project_root=self.project_root,
            runs_root=self.runs_root,
            dag_id=swarm_id
        )
        
        if not dag_res.get("ok", False):
            return dag_res

        # Emit SWARM_STATE.json (alias of DAG_STATE.json for now)
        dag_state = json.loads((dag_dir / "DAG_STATE.json").read_text(encoding="utf-8"))
        _atomic_write_canon_json(swarm_dir / "SWARM_STATE.json", dag_state)

        # Emit SWARM_CHAIN.json (top-level chain across pipelines)
        swarm_chain = self._build_swarm_chain(swarm_id=swarm_id, dag_state=dag_state)
        _atomic_write_canon_json(swarm_dir / "SWARM_CHAIN.json", swarm_chain)

        return {"ok": True, "swarm_id": swarm_id, "nodes": len(dag_nodes)}

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
