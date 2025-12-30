#!/usr/bin/env python3
"""
Bundle System (Phase 5)

Deterministic, bounded, fail-closed executable bundle from completed jobs.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import asdict

from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError
from catalytic_chat.section_indexer import SectionIndexer
from catalytic_chat.symbol_registry import SymbolRegistry
from catalytic_chat.slice_resolver import SliceResolver, SliceError
import os


class BundleError(Exception):
    """Bundle system error."""
    pass


def _canonical_json(data: Any) -> str:
    """Serialize data to canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(content: str) -> str:
    """Compute SHA256 hex digest of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _resolve_bundle_path(bundle_path: Path) -> Tuple[Path, Path]:
    """Resolve bundle path and return (bundle_dir, bundle_json_path).
    
    Supports:
    - Directory path: treat as bundle root
    - Path to bundle.json: extract parent as bundle root
    
    Args:
        bundle_path: Path to bundle directory or bundle.json
        
    Returns:
        Tuple of (bundle_dir, bundle_json_path)
        
    Raises:
        BundleError: If path is invalid or bundle.json not found
    """
    bundle_path = Path(bundle_path)
    
    if bundle_path.is_file():
        if bundle_path.name != "bundle.json":
            raise BundleError(f"Bundle file must be named bundle.json: {bundle_path}")
        bundle_json_path = bundle_path
        bundle_dir = bundle_path.parent
    elif bundle_path.is_dir():
        bundle_dir = bundle_path
        bundle_json_path = bundle_dir / "bundle.json"
        if not bundle_json_path.exists():
            raise BundleError(f"bundle.json not found in directory: {bundle_dir}")
    else:
        raise BundleError(f"Bundle path not found: {bundle_path}")
    
    return bundle_dir, bundle_json_path


class BundleBuilder:
    """Build deterministic bundles from completed jobs."""
    
    VERSION = "5.0.0"
    
    def __init__(self, repo_root: Optional[Path] = None):
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = repo_root
        self.cassette = MessageCassette(repo_root=repo_root)
        self.section_indexer = SectionIndexer(repo_root=repo_root)
        self.symbol_registry = SymbolRegistry(repo_root=repo_root)
        self.slice_resolver = SliceResolver()
    
    def _check_job_complete(self, run_id: str, job_id: str) -> Tuple[bool, str]:
        """Verify job completeness: all steps COMMITTED, one receipt per step."""
        conn = self.cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT s.step_id, s.status
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            WHERE m.run_id = ? AND j.job_id = ?
        """, (run_id, job_id))
        
        steps = cursor.fetchall()
        if not steps:
            return False, f"No steps found for run_id={run_id}, job_id={job_id}"
        
        for step in steps:
            if step["status"] != "COMMITTED":
                return False, f"Step {step['step_id']} not COMMITTED (status={step['status']})"
            
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM cassette_receipts
                WHERE step_id = ?
            """, (step["step_id"],))
            
            receipt_count = cursor.fetchone()["count"]
            if receipt_count != 1:
                return False, f"Step {step['step_id']} has {receipt_count} receipts (expected 1)"
        
        return True, "Job complete"
    
    def _extract_artifacts(self, run_id: str, job_id: str) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, Optional[str]]]]:
        """Extract artifacts from completed job with boundedness checks."""
        conn = self.cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT s.step_id, s.ordinal, s.payload_json
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            WHERE m.run_id = ? AND j.job_id = ?
            ORDER BY s.ordinal ASC, s.step_id ASC
        """, (run_id, job_id))
        
        steps = cursor.fetchall()
        artifacts = []
        step_refs = set()
        
        for step in steps:
            step_payload = json.loads(step["payload_json"])
            op = step_payload.get("op")
            refs = step_payload.get("refs", {})
            constraints = step_payload.get("constraints", {})
            
            if op == "READ_SYMBOL":
                symbol_id = refs.get("symbol_id")
                if not symbol_id:
                    raise BundleError(f"READ_SYMBOL step missing symbol_id: {step['step_id']}")
                
                slice_expr = constraints.get("slice")
                
                symbol = self.symbol_registry.get_symbol(symbol_id)
                if symbol is None:
                    raise BundleError(f"Symbol not found: {symbol_id}")
                
                section_id = symbol.target_ref
                
                if not slice_expr:
                    slice_expr = symbol.default_slice
                
                if slice_expr and slice_expr.lower() == "all":
                    raise BundleError(f"slice=ALL forbidden for symbol {symbol_id} in step {step['step_id']}")
                
                step_refs.add((symbol_id, slice_expr))
                
                content, content_hash, applied_slice, _, _ = \
                    self.section_indexer.get_section_content(section_id, slice_expr)
                
                artifact_id = f"{symbol_id}_{slice_expr or 'none'}"
                artifact_id = hashlib.sha256(artifact_id.encode()).hexdigest()[:16]
                
                artifacts.append({
                    "artifact_id": artifact_id,
                    "kind": "SYMBOL_SLICE",
                    "ref": symbol_id,
                    "slice": applied_slice,
                    "content": content,
                    "sha256": content_hash
                })
                
            elif op == "READ_SECTION":
                section_id = refs.get("section_id")
                if not section_id:
                    raise BundleError(f"READ_SECTION step missing section_id: {step['step_id']}")
                
                slice_expr = constraints.get("slice")
                
                if slice_expr and slice_expr.lower() == "all":
                    raise BundleError(f"slice=ALL forbidden for section {section_id} in step {step['step_id']}")
                
                step_refs.add((section_id, slice_expr))
                
                content, content_hash, applied_slice, _, _ = \
                    self.section_indexer.get_section_content(section_id, slice_expr)
                
                artifact_id = f"{section_id}_{slice_expr or 'none'}"
                artifact_id = hashlib.sha256(artifact_id.encode()).hexdigest()[:16]
                
                artifacts.append({
                    "artifact_id": artifact_id,
                    "kind": "SECTION_SLICE",
                    "ref": section_id,
                    "slice": applied_slice,
                    "content": content,
                    "sha256": content_hash
                })
        
        return artifacts, step_refs
    
    def _validate_boundedness(self, artifacts: List[Dict[str, Any]], step_refs: Set[Tuple[str, Optional[str]]]) -> None:
        """Ensure bundle contains only referenced artifacts."""
        for artifact in artifacts:
            ref = artifact["ref"]
            slice_expr = artifact["slice"]
            
            if (ref, slice_expr) not in step_refs:
                raise BundleError(f"Artifact not referenced by steps: {ref} slice={slice_expr}")
    
    def _compute_plan_hash(self, steps: List[Dict[str, Any]]) -> str:
        """Compute plan hash from steps."""
        sorted_steps = sorted(steps, key=lambda x: (x["ordinal"], x["step_id"]))
        
        canonical_steps = []
        for step in sorted_steps:
            step_for_hash = {
                "step_id": step["step_id"],
                "ordinal": step["ordinal"],
                "op": step["op"],
                "refs": step.get("refs", {}),
                "constraints": step.get("constraints", {}),
                "expected_outputs": step.get("expected_outputs", {})
            }
            canonical_steps.append(step_for_hash)
        
        canonical_plan = json.dumps({
            "run_id": steps[0].get("run_id", ""),
            "steps": canonical_steps
        }, sort_keys=True)
        
        return hashlib.sha256(canonical_plan.encode('utf-8')).hexdigest()
    
    def _compute_root_hash(self, artifacts: List[Dict[str, Any]]) -> str:
        """Compute root hash from artifacts."""
        sorted_artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
        
        hash_strings = []
        for artifact in sorted_artifacts:
            hash_strings.append(f"{artifact['artifact_id']}:{artifact['sha256']}")
        
        combined = "\n".join(hash_strings) + "\n"
        return _sha256(combined)
    
    def _extract_inputs(self, step_refs: Set[Tuple[str, Optional[str]]]) -> Dict[str, List[str]]:
        """Extract unique inputs from step refs."""
        symbols = []
        files = []
        slices = []
        
        for ref, slice_expr in step_refs:
            if ref.startswith("@"):
                if ref not in symbols:
                    symbols.append(ref)
            else:
                if ref not in files:
                    files.append(ref)
            
            if slice_expr and slice_expr not in slices:
                slices.append(slice_expr)
        
        return {
            "symbols": sorted(symbols),
            "files": sorted(files),
            "slices": sorted(slices)
        }

    def build(self, run_id: str, job_id: str, output_dir: Path) -> Dict[str, Any]:
        """Build bundle from completed job."""
        output_dir = Path(output_dir)
        
        is_complete, reason = self._check_job_complete(run_id, job_id)
        if not is_complete:
            raise BundleError(f"Job completeness gate failed: {reason}")
        
        conn = self.cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT m.message_id
            FROM cassette_messages m
            JOIN cassette_jobs j ON m.message_id = j.message_id
            WHERE m.run_id = ? AND j.job_id = ?
        """, (run_id, job_id))
        
        message_row = cursor.fetchone()
        if not message_row:
            raise BundleError(f"No message found for run_id={run_id}, job_id={job_id}")
        
        message_id = message_row["message_id"]
        
        cursor = conn.execute("""
            SELECT s.step_id, s.ordinal, s.payload_json
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            WHERE m.run_id = ? AND j.job_id = ?
            ORDER BY s.ordinal ASC, s.step_id ASC
        """, (run_id, job_id))
        
        step_rows = cursor.fetchall()
        steps = []
        for row in step_rows:
            step_payload = json.loads(row["payload_json"])
            steps.append(step_payload)
        
        plan_hash = self._compute_plan_hash(steps)
        
        artifacts, step_refs = self._extract_artifacts(run_id, job_id)
        self._validate_boundedness(artifacts, step_refs)
        
        inputs = self._extract_inputs(step_refs)
        
        sorted_artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
        sorted_steps = sorted(steps, key=lambda x: (x["ordinal"], x["step_id"]))
        
        pre_manifest = {
            "bundle_version": self.VERSION,
            "bundle_id": "",
            "run_id": run_id,
            "job_id": job_id,
            "message_id": message_id,
            "plan_hash": plan_hash,
            "steps": sorted_steps,
            "inputs": inputs,
            "artifacts": [],
            "hashes": {"root_hash": ""},
            "provenance": {}
        }
        
        pre_manifest_json = _canonical_json(pre_manifest)
        bundle_id = _sha256(pre_manifest_json)
        
        artifact_manifests = []
        for artifact in sorted_artifacts:
            content = artifact.pop("content")
            artifact_path = output_dir / "artifacts" / f"{artifact['artifact_id']}.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            content_bytes = content.encode('utf-8')
            artifact_bytes = len(content_bytes)
            
            artifact["path"] = f"artifacts/{artifact['artifact_id']}.txt"
            artifact["bytes"] = artifact_bytes
            artifact_manifests.append(artifact)
            
            with open(artifact_path, 'w', encoding='utf-8') as f:
                f.write(content)
                if not content.endswith('\n'):
                    f.write('\n')
        
        root_hash = self._compute_root_hash(artifact_manifests)
        
        manifest = {
            "bundle_version": self.VERSION,
            "bundle_id": bundle_id,
            "run_id": run_id,
            "job_id": job_id,
            "message_id": message_id,
            "plan_hash": plan_hash,
            "steps": sorted_steps,
            "inputs": inputs,
            "artifacts": sorted(artifact_manifests, key=lambda x: x["artifact_id"]),
            "hashes": {"root_hash": root_hash},
            "provenance": {}
        }
        
        manifest_json = _canonical_json(manifest)
        
        bundle_json_path = output_dir / "bundle.json"
        with open(bundle_json_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(manifest_json)
            if not manifest_json.endswith('\n'):
                f.write('\n')
        
        return {
            "bundle_id": bundle_id,
            "output_dir": str(output_dir),
            "artifact_count": len(artifact_manifests),
            "root_hash": root_hash
        }
    
    def close(self):
        self.cassette.close()


class BundleVerifier:
    """Verify bundle integrity and constraints."""
    
    def __init__(self, bundle_path: Path):
        self.bundle_dir, self.bundle_json_path = _resolve_bundle_path(bundle_path)
        self.artifacts_dir = self.bundle_dir / "artifacts"
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load and validate bundle.json."""
        if not self.bundle_json_path.exists():
            raise BundleError(f"bundle.json not found: {self.bundle_json_path}")
        
        with open(self.bundle_json_path, 'r', encoding='utf-8') as f:
            content = f.read().rstrip('\n')
            manifest = json.loads(content)
        
        return manifest
    
    def _validate_schema(self, manifest: Dict[str, Any]) -> None:
        """Basic schema validation."""
        required = [
            "bundle_version", "bundle_id", "run_id", "job_id", "message_id",
            "plan_hash", "steps", "inputs", "artifacts", "hashes", "provenance"
        ]
        
        for field in required:
            if field not in manifest:
                raise BundleError(f"Missing required field: {field}")
        
        if "root_hash" not in manifest["hashes"]:
            raise BundleError("Missing hashes.root_hash")

        for artifact in manifest["artifacts"]:
            required = ["artifact_id", "kind", "ref", "slice", "path", "sha256", "bytes"]
            for field in required:
                if field not in artifact:
                    raise BundleError(f"Artifact missing required field: {field}")
            
            if artifact["kind"] not in ["SYMBOL_SLICE", "SECTION_SLICE"]:
                raise BundleError(f"Invalid artifact kind: {artifact['kind']}")
    
    def _validate_ordering(self, manifest: Dict[str, Any]) -> None:
        """Ensure steps and artifacts are properly ordered."""
        steps = manifest["steps"]
        sorted_steps = sorted(steps, key=lambda x: (x["ordinal"], x["step_id"]))
        if steps != sorted_steps:
            raise BundleError("Steps not ordered by (ordinal asc, step_id asc)")
        
        artifacts = manifest["artifacts"]
        sorted_artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
        if artifacts != sorted_artifacts:
            raise BundleError("Artifacts not ordered by artifact_id asc")
        
        inputs = manifest["inputs"]
        for key in ["symbols", "files", "slices"]:
            if key in inputs:
                if inputs[key] != sorted(inputs[key]):
                    raise BundleError(f"inputs.{key} not sorted")
    
    def _verify_artifacts(self, manifest: Dict[str, Any]) -> None:
        """Verify artifact content hashes."""
        for artifact in manifest["artifacts"]:
            posix_path = artifact["path"]
            artifact_path = self.bundle_dir / posix_path
            
            if not artifact_path.exists():
                raise BundleError(f"Artifact file missing: {artifact_path}")
            
            resolved = artifact_path.resolve()
            bundle_resolved = self.bundle_dir.resolve()
            try:
                resolved.relative_to(bundle_resolved)
            except ValueError:
                raise BundleError(f"Path traversal attempt: {artifact['path']}")
            
            with open(artifact_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            computed_hash = _sha256(content)
            if computed_hash != artifact["sha256"]:
                raise BundleError(f"Artifact hash mismatch: {artifact['artifact_id']}")
            
            expected_bytes = artifact["bytes"]
            actual_bytes = len(content.encode('utf-8'))
            if actual_bytes != expected_bytes:
                raise BundleError(f"Artifact size mismatch: {artifact['artifact_id']}")
            
            if not content.endswith('\n'):
                raise BundleError(f"Artifact missing trailing newline: {artifact['artifact_id']}")
    
    def _verify_root_hash(self, manifest: Dict[str, Any]) -> None:
        """Recompute and verify root hash."""
        artifacts = manifest["artifacts"]
        sorted_artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
        
        hash_strings = []
        for artifact in sorted_artifacts:
            hash_strings.append(f"{artifact['artifact_id']}:{artifact['sha256']}")
        
        combined = "\n".join(hash_strings) + "\n"
        computed_root_hash = _sha256(combined)
        
        if computed_root_hash != manifest["hashes"]["root_hash"]:
            raise BundleError(f"Root hash mismatch: expected {manifest['hashes']['root_hash']}, computed {computed_root_hash}")
    
    def _verify_bundle_id(self, manifest: Dict[str, Any]) -> None:
        """Recompute bundle_id and verify."""
        pre_manifest = manifest.copy()
        pre_manifest["bundle_id"] = ""
        pre_manifest["hashes"]["root_hash"] = ""
        
        pre_manifest_json = _canonical_json(pre_manifest)
        computed_bundle_id = _sha256(pre_manifest_json)
        
        if computed_bundle_id != manifest["bundle_id"]:
            raise BundleError(f"Bundle ID mismatch: expected {manifest['bundle_id']}, computed {computed_bundle_id}")
    
    def _validate_boundedness(self, manifest: Dict[str, Any]) -> None:
        """Ensure no ALL slices and artifacts match step refs."""
        for artifact in manifest["artifacts"]:
            if artifact["slice"] and artifact["slice"].lower() == "all":
                raise BundleError(f"Artifact has forbidden ALL slice: {artifact['artifact_id']}")
        
        step_refs = set()
        for step in manifest["steps"]:
            op = step.get("op")
            refs = step.get("refs", {})
            constraints = step.get("constraints", {})
            
            if op == "READ_SYMBOL":
                symbol_id = refs.get("symbol_id")
                slice_expr = constraints.get("slice")
                step_refs.add((symbol_id, slice_expr))
            elif op == "READ_SECTION":
                section_id = refs.get("section_id")
                slice_expr = constraints.get("slice")
                step_refs.add((section_id, slice_expr))
        
        for artifact in manifest["artifacts"]:
            ref = artifact["ref"]
            slice_expr = artifact["slice"]
            if (ref, slice_expr) not in step_refs:
                raise BundleError(f"Artifact not referenced by steps: {ref} slice={slice_expr}")
    
    def _reject_forbidden_fields(self, manifest: Dict[str, Any]) -> None:
        """Reject absolute paths and forbidden fields."""
        forbidden_keys = ["timestamp", "created_at", "updated_at", "cwd", "os", "locale"]
        
        for key in forbidden_keys:
            if key in manifest:
                raise BundleError(f"Forbidden top-level field: {key}")
        
        for artifact in manifest["artifacts"]:
            if "\\" in artifact["path"] or artifact["path"].startswith("/"):
                raise BundleError(f"Absolute path forbidden: {artifact['path']}")
    
    def verify(self) -> Dict[str, Any]:
        """Verify bundle integrity."""
        manifest = self._load_manifest()
        
        self._validate_schema(manifest)
        self._validate_ordering(manifest)
        self._verify_artifacts(manifest)
        self._verify_root_hash(manifest)
        self._verify_bundle_id(manifest)
        self._validate_boundedness(manifest)
        self._reject_forbidden_fields(manifest)
        
        return {
            "status": "success",
            "bundle_id": manifest["bundle_id"],
            "run_id": manifest["run_id"],
            "job_id": manifest["job_id"],
            "artifact_count": len(manifest["artifacts"]),
            "root_hash": manifest["hashes"]["root_hash"]
        }
