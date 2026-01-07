from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PRIMITIVES.restore_proof import canonical_json_bytes


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _spec_step_ids(pipeline_dir: Path) -> List[str]:
    spec = _load_json(pipeline_dir / "PIPELINE.json")
    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("PIPELINE.json missing steps")
    out: List[str] = []
    for idx, s in enumerate(steps):
        if not isinstance(s, dict):
            raise ValueError(f"PIPELINE.json step[{idx}] not an object")
        step_id = s.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"PIPELINE.json step[{idx}] missing step_id")
        out.append(step_id)
    return out


def chain_path(pipeline_dir: Path) -> Path:
    return pipeline_dir / "CHAIN.json"


def build_chain_obj(
    *,
    project_root: Path,
    pipeline_id: str,
    ordered_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {"pipeline_id": pipeline_id, "steps": ordered_steps}


def compute_step_entry(
    *,
    project_root: Path,
    pipeline_id: str,
    step_id: str,
    run_id: str,
    prev_step_proof_hash: Optional[str],
) -> Dict[str, Any]:
    run_dir = project_root / "CONTRACTS" / "_runs" / run_id
    proof_path = run_dir / "PROOF.json"
    roots_path = run_dir / "DOMAIN_ROOTS.json"
    if not proof_path.exists() or not roots_path.exists():
        missing = [p.name for p in [proof_path, roots_path] if not p.exists()]
        raise FileNotFoundError(f"missing step artifacts for {run_id}: {missing}")

    return {
        "pipeline_id": pipeline_id,
        "step_id": step_id,
        "run_id": run_id,
        "step_proof_hash": _sha256_file(proof_path),
        "prev_step_proof_hash": prev_step_proof_hash,
        "step_domain_roots_hash": _sha256_file(roots_path),
    }


def write_chain(
    *,
    pipeline_dir: Path,
    chain_obj: Dict[str, Any],
) -> None:
    # atomic write is done by pipeline_runtime; this helper keeps bytes canonical.
    chain_path(pipeline_dir).write_bytes(canonical_json_bytes(chain_obj) + b"\n")


@dataclass(frozen=True)
class ChainError:
    code: str
    message: str
    step_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.step_id is not None:
            out["step_id"] = self.step_id
        return out


def verify_chain(*, project_root: Path, pipeline_dir: Path) -> Dict[str, Any]:
    cpath = chain_path(pipeline_dir)
    if not cpath.exists():
        return {"ok": False, "code": "CHAIN_MISSING", "details": {"errors": [ChainError(code="CHAIN_MISSING", message="CHAIN.json missing").to_json()]}}

    try:
        chain = _load_json(cpath)
    except Exception as e:
        return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message=str(e)).to_json()]}}

    pipeline_id = chain.get("pipeline_id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message="missing pipeline_id").to_json()]}}

    steps = chain.get("steps")
    if not isinstance(steps, list):
        return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message="missing steps list").to_json()]}}

    try:
        expected_order = _spec_step_ids(pipeline_dir)
    except Exception as e:
        return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message=str(e)).to_json()]}}

    if len(steps) > len(expected_order):
        return {"ok": False, "code": "CHAIN_STEP_ORDER_MISMATCH", "details": {"errors": [ChainError(code="CHAIN_STEP_ORDER_MISMATCH", message="chain longer than spec").to_json()]}}

    # Ensure exact prefix order matches PIPELINE.json
    step_ids: List[str] = []
    for idx, entry in enumerate(steps):
        if not isinstance(entry, dict):
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message=f"step[{idx}] not object").to_json()]}}
        sid = entry.get("step_id")
        if not isinstance(sid, str) or not sid:
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message=f"step[{idx}] missing step_id").to_json()]}}
        step_ids.append(sid)

    if step_ids != expected_order[: len(step_ids)]:
        return {"ok": False, "code": "CHAIN_STEP_ORDER_MISMATCH", "details": {"errors": [ChainError(code="CHAIN_STEP_ORDER_MISMATCH", message="step order mismatch").to_json()]}}

    # Validate links + recompute hashes.
    prev_hash: Optional[str] = None
    for entry in steps:
        sid = entry.get("step_id")
        rid = entry.get("run_id")
        recorded_prev = entry.get("prev_step_proof_hash")
        recorded_proof = entry.get("step_proof_hash")
        recorded_roots = entry.get("step_domain_roots_hash")

        if entry.get("pipeline_id") != pipeline_id:
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message="pipeline_id mismatch inside step entry", step_id=sid).to_json()]}}

        if recorded_prev != prev_hash:
            return {"ok": False, "code": "CHAIN_LINK_MISMATCH", "details": {"errors": [ChainError(code="CHAIN_LINK_MISMATCH", message="prev_step_proof_hash mismatch", step_id=sid).to_json()]}}

        if not isinstance(rid, str) or not rid:
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"errors": [ChainError(code="CHAIN_INVALID", message="missing run_id", step_id=sid).to_json()]}}

        run_dir = project_root / "CONTRACTS" / "_runs" / rid
        proof_path = run_dir / "PROOF.json"
        roots_path = run_dir / "DOMAIN_ROOTS.json"
        if not proof_path.exists() or not roots_path.exists():
            missing = [p.name for p in [proof_path, roots_path] if not p.exists()]
            return {"ok": False, "code": "CHAIN_MISSING_ARTIFACT", "details": {"errors": [ChainError(code="CHAIN_MISSING_ARTIFACT", message=f"missing artifacts: {missing}", step_id=sid).to_json()]}}

        actual_proof = _sha256_file(proof_path)
        actual_roots = _sha256_file(roots_path)
        if recorded_proof != actual_proof:
            return {"ok": False, "code": "CHAIN_PROOF_HASH_MISMATCH", "details": {"errors": [ChainError(code="CHAIN_PROOF_HASH_MISMATCH", message="PROOF.json hash mismatch", step_id=sid).to_json()]}}
        if recorded_roots != actual_roots:
            return {"ok": False, "code": "CHAIN_DOMAIN_ROOTS_HASH_MISMATCH", "details": {"errors": [ChainError(code="CHAIN_DOMAIN_ROOTS_HASH_MISMATCH", message="DOMAIN_ROOTS.json hash mismatch", step_id=sid).to_json()]}}

        prev_hash = actual_proof

    return {"ok": True, "code": "OK", "details": {}}

