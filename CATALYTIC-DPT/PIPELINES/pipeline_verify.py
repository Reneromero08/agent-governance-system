from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from PIPELINES.pipeline_chain import verify_chain
from PIPELINES.pipeline_runtime import _slug
from PRIMITIVES.ledger import Ledger, _LEDGER_VALIDATOR  # type: ignore


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def _load_json_obj(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _verify_proof_hash(*, proof_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    proof_hash = proof_obj.get("proof_hash")
    if not (isinstance(proof_hash, str) and _HEX64_RE.fullmatch(proof_hash) is not None):
        return None
    proof_without_hash = dict(proof_obj)
    proof_without_hash.pop("proof_hash", None)
    computed = hashlib.sha256(
        json.dumps(proof_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if computed != proof_hash:
        return {"expected": proof_hash, "computed": computed}
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_pipeline(
    *,
    project_root: Path,
    pipeline_id: str,
    runs_root: Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Fail-closed pipeline verifier.

    Verifies:
    - CHAIN.json integrity across steps (hash links + order)
    - Required per-step artifacts exist and are non-empty
    - PROOF.json proof_hash (if present) matches recomputed hash
    - LEDGER.jsonl is valid JSONL and schema-valid, with required fields present
    """
    pipeline_dir = runs_root / "_pipelines" / _slug(pipeline_id)
    if not pipeline_dir.exists():
        return {
            "ok": False,
            "code": "PIPELINE_NOT_FOUND",
            "details": {"pipeline_dir": str(pipeline_dir)},
        }

    chain_result = verify_chain(project_root=project_root, pipeline_dir=pipeline_dir)
    if not chain_result.get("ok", False):
        return {
            "ok": False,
            "code": chain_result.get("code", "CHAIN_INVALID"),
            "details": {"phase": "CHAIN", **chain_result.get("details", {})},
        }

    chain = _load_json_obj(pipeline_dir / "CHAIN.json")
    steps = chain.get("steps")
    if not isinstance(steps, list):
        return {"ok": False, "code": "CHAIN_INVALID", "details": {"phase": "CHAIN", "message": "missing steps"}}

    seen_step_ids: set[str] = set()
    for idx, entry in enumerate(steps):
        if not isinstance(entry, dict):
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"phase": "CHAIN", "message": f"step[{idx}] not object"}}
        step_id = entry.get("step_id")
        run_id = entry.get("run_id")
        if not isinstance(step_id, str) or not step_id:
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"phase": "CHAIN", "message": f"step[{idx}] missing step_id"}}
        if step_id in seen_step_ids:
            return {"ok": False, "code": "CHAIN_DUPLICATE_STEP_ID", "details": {"phase": "CHAIN", "step_id": step_id}}
        seen_step_ids.add(step_id)
        if not isinstance(run_id, str) or not run_id:
            return {"ok": False, "code": "CHAIN_INVALID", "details": {"phase": "CHAIN", "step_id": step_id, "message": "missing run_id"}}

        run_dir = runs_root / run_id
        proof_path = run_dir / "PROOF.json"
        roots_path = run_dir / "DOMAIN_ROOTS.json"
        ledger_path = run_dir / "LEDGER.jsonl"
        outputs_hashes_path = run_dir / "OUTPUT_HASHES.json"

        missing = [
            p.name
            for p in (proof_path, roots_path, ledger_path, outputs_hashes_path)
            if not _is_nonempty_file(p)
        ]
        if missing:
            return {
                "ok": False,
                "code": "STEP_ARTIFACT_MISSING",
                "details": {"phase": "STEP_ARTIFACTS", "step_id": step_id, "run_id": run_id, "missing": missing},
            }

        # PROOF hash integrity (if present).
        try:
            proof_obj = _load_json_obj(proof_path)
        except Exception as e:
            return {
                "ok": False,
                "code": "PROOF_INVALID_JSON",
                "details": {"phase": "PROOF", "step_id": step_id, "run_id": run_id, "message": str(e)},
            }
        mismatch = _verify_proof_hash(proof_obj=proof_obj)
        if mismatch is not None:
            return {
                "ok": False,
                "code": "PROOF_HASH_MISMATCH",
                "details": {"phase": "PROOF", "step_id": step_id, "run_id": run_id, **mismatch},
            }

        # Ledger sanity: fail closed on corruption/partial lines.
        ledger = Ledger(ledger_path)
        if not ledger.verify_append_only():
            return {
                "ok": False,
                "code": "LEDGER_CORRUPT",
                "details": {"phase": "LEDGER", "step_id": step_id, "run_id": run_id},
            }
        try:
            records = ledger.read_all()
        except Exception as e:
            return {
                "ok": False,
                "code": "LEDGER_CORRUPT",
                "details": {"phase": "LEDGER", "step_id": step_id, "run_id": run_id, "message": str(e)},
            }
        if not records:
            return {
                "ok": False,
                "code": "LEDGER_EMPTY",
                "details": {"phase": "LEDGER", "step_id": step_id, "run_id": run_id},
            }
        for line_no, rec in enumerate(records, start=1):
            errors = sorted(_LEDGER_VALIDATOR.iter_errors(rec), key=lambda e: e.path)
            if errors:
                first = errors[0]
                return {
                    "ok": False,
                    "code": "LEDGER_SCHEMA_INVALID",
                    "details": {
                        "phase": "LEDGER",
                        "step_id": step_id,
                        "run_id": run_id,
                        "line": line_no,
                        "message": first.message,
                        "path": list(first.path),
                    },
                }
        if not any(isinstance(r.get("RUN_INFO"), dict) and isinstance(r.get("STATUS"), dict) for r in records):
            return {
                "ok": False,
                "code": "LEDGER_MISSING_REQUIRED_FIELDS",
                "details": {"phase": "LEDGER", "step_id": step_id, "run_id": run_id},
            }

        # Proof-gated acceptance for step runs.
        try:
            from TOOLS.catalytic_validator import CatalyticLedgerValidator  # type: ignore
        except ModuleNotFoundError:
            from catalytic_validator import CatalyticLedgerValidator  # type: ignore

        ok, report = CatalyticLedgerValidator(run_dir).validate()
        if not ok:
            return {
                "ok": False,
                "code": "STEP_RUN_INVALID",
                "details": {"phase": "STEP_VALIDATE", "step_id": step_id, "run_id": run_id, "report": report},
            }

        if strict and isinstance(report, dict) and report.get("valid") is not True:
            return {
                "ok": False,
                "code": "STEP_RUN_INVALID",
                "details": {"phase": "STEP_VALIDATE", "step_id": step_id, "run_id": run_id, "report": report},
            }

        # Durable output tamper detection: OUTPUT_HASHES.json is the source of truth.
        # Recompute hashes over exact bytes and fail closed on any mismatch.
        try:
            output_hashes = _load_json_obj(outputs_hashes_path)
        except Exception as e:
            return {
                "ok": False,
                "code": "OUTPUT_HASHES_INVALID_JSON",
                "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "message": str(e)},
            }

        for rel_path, expected in sorted(output_hashes.items(), key=lambda kv: kv[0]):
            if not isinstance(rel_path, str) or not rel_path:
                return {
                    "ok": False,
                    "code": "OUTPUT_HASHES_INVALID_ENTRY",
                    "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "message": "non-string path"},
                }
            if not isinstance(expected, str):
                return {
                    "ok": False,
                    "code": "OUTPUT_HASHES_INVALID_ENTRY",
                    "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "path": rel_path, "message": "non-string hash"},
                }
            if Path(rel_path).is_absolute():
                return {
                    "ok": False,
                    "code": "OUTPUT_PATH_INVALID",
                    "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "path": rel_path},
                }
            abs_path = project_root / rel_path
            if not abs_path.exists():
                return {
                    "ok": False,
                    "code": "OUTPUT_MISSING",
                    "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "path": rel_path},
                }
            if abs_path.is_dir():
                # Directory outputs are allowed to have empty hashes; presence is the check.
                if expected not in ("",):
                    return {
                        "ok": False,
                        "code": "OUTPUT_HASHES_INVALID_ENTRY",
                        "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "path": rel_path, "message": "dir must have empty hash"},
                    }
                continue
            if not _HEX64_RE.fullmatch(expected):
                return {
                    "ok": False,
                    "code": "OUTPUT_HASHES_INVALID_ENTRY",
                    "details": {"phase": "OUTPUT_HASHES", "step_id": step_id, "run_id": run_id, "path": rel_path, "message": "expected hash must be 64 lowercase hex"},
                }
            computed = _sha256_file(abs_path)
            if computed != expected:
                return {
                    "ok": False,
                    "code": "OUTPUT_HASH_MISMATCH",
                    "details": {
                        "phase": "OUTPUT_HASHES",
                        "step_id": step_id,
                        "run_id": run_id,
                        "path": rel_path,
                        "expected": expected,
                        "computed": computed,
                    },
                }

    return {"ok": True, "code": "OK", "details": {"pipeline_id": pipeline_id, "steps_verified": len(steps)}}
