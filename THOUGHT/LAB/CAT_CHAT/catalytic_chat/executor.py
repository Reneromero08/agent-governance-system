#!/usr/bin/env python3
"""
Bundle Executor (Phase 6.1)

Deterministic, verifier-gated bundle execution with receipt emission.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from catalytic_chat.bundle import BundleVerifier, BundleError, _resolve_bundle_path

try:
    from catalytic_chat.receipt import (
        build_receipt_from_bundle_run,
        write_receipt,
        RECEIPT_VERSION,
        EXECUTOR_VERSION
    )
except ImportError:
    RECEIPT_VERSION = "1.0.0"
    EXECUTOR_VERSION = "1.0.0"


def _sha256(content: str) -> str:
    """Compute SHA256 hex digest of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


class BundleExecutor:
    """Execute verified bundles deterministically and emit receipts."""
    
    SUPPORTED_STEPS = {"READ_SYMBOL", "READ_SECTION"}
    
    def __init__(self, bundle_path: Path, receipt_out: Optional[Path] = None):
        self.bundle_dir, self.bundle_json_path = _resolve_bundle_path(bundle_path)
        self.artifacts_dir = self.bundle_dir / "artifacts"
        
        if receipt_out is None:
            receipt_out = self.bundle_dir / "receipt.json"
        self.receipt_out = Path(receipt_out)
        
        self.manifest = None
        self.artifact_map = {}
    
    def _load_bundle(self):
        """Load and verify bundle before execution."""
        verify_result = None
        
        try:
            verifier = BundleVerifier(self.bundle_dir)
            verify_result = verifier.verify()
        except BundleError as e:
            error = {
                "code": "VERIFY_FAILED",
                "message": f"Bundle verification failed",
                "step_id": None
            }
            receipt = build_receipt_from_bundle_run(
                bundle_manifest={},
                step_results=[],
                outcome="FAILURE",
                error=error
            )
            write_receipt(self.receipt_out, receipt)
            raise BundleError(f"Bundle verification failed")
        
        if verify_result is None or verify_result["status"] != "success":
            if verify_result and "status" in verify_result:
                error_msg = verify_result.get("reason", verify_result.get("error", "Bundle verification failed"))
            error = {
                    "code": "VERIFY_FAILED",
                    "message": error_msg or "Bundle verification failed",
                    "step_id": None
                }
                receipt = build_receipt_from_bundle_run(
                    bundle_manifest={},
                    step_results=[],
                    outcome="FAILURE",
                    error=error
                )
                write_receipt(self.receipt_out, receipt)
                raise BundleError("Bundle verification failed")
        
        if not self.bundle_json_path.exists():
            raise BundleError(f"bundle.json not found: {self.bundle_json_path}")
        
        with open(self.bundle_json_path, 'r', encoding='utf-8') as f:
            content = f.read().rstrip('\n')
            self.manifest = json.loads(content)
        
        self._build_artifact_map()
    
    def _build_artifact_map(self):
        """Build map of (ref, slice) -> artifact_id."""
        for artifact in self.manifest["artifacts"]:
            key = (artifact["ref"], artifact["slice"])
            self.artifact_map[key] = artifact
    
    def _get_artifact_content(self, ref: str, slice_expr: str) -> str:
        """Get artifact content from bundle only (no repo access)."""
        key = (ref, slice_expr)
        if key not in self.artifact_map:
            raise BundleError(f"Artifact not found in bundle: {ref} slice={slice_expr}")
        
        artifact = self.artifact_map[key]
        artifact_path = self.bundle_dir / artifact["path"]
        
        if not artifact_path.exists():
            raise BundleError(f"Artifact file missing: {artifact_path}")
        
        with open(artifact_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    def _execute_read_symbol(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute READ_SYMBOL step using bundle artifacts only."""
        refs = step.get("refs", {})
        constraints = step.get("constraints", {})
        
        symbol_id = refs.get("symbol_id")
        if not symbol_id:
            raise BundleError(f"READ_SYMBOL step missing symbol_id: {step['step_id']}")
        
        slice_expr = constraints.get("slice")
        
        content = self._get_artifact_content(symbol_id, slice_expr)
        content_hash = _sha256(content)
        content_bytes = len(content.encode('utf-8'))
        
        result = {
            "kind": "SYMBOL_SLICE",
            "ref": symbol_id,
            "slice": slice_expr,
            "sha256": content_hash,
            "bytes": content_bytes
        }
        
        return {
            "step_id": step["step_id"],
            "ordinal": step.get("ordinal"),
            "op": "READ_SYMBOL",
            "outcome": "SUCCESS",
            "result": result
        }
    
    def _execute_read_section(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute READ_SECTION step using bundle artifacts only."""
        refs = step.get("refs", {})
        constraints = step.get("constraints", {})
        
        section_id = refs.get("section_id")
        if not section_id:
            raise BundleError(f"READ_SECTION step missing section_id: {step['step_id']}")
        
        slice_expr = constraints.get("slice")
        
        content = self._get_artifact_content(section_id, slice_expr)
        content_hash = _sha256(content)
        content_bytes = len(content.encode('utf-8'))
        
        result = {
            "kind": "SECTION_SLICE",
            "ref": section_id,
            "slice": slice_expr,
            "sha256": content_hash,
            "bytes": content_bytes
        }
        
        return {
            "step_id": step["step_id"],
            "ordinal": step.get("ordinal"),
            "op": "READ_SECTION",
            "outcome": "SUCCESS",
            "result": result
        }
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step."""
        op = step.get("op")
        
        if op not in self.SUPPORTED_STEPS:
            error = {
                "code": "UNSUPPORTED_STEP",
                "message": f"Unsupported step kind: {op}",
                "step_id": step.get("step_id")
            }
            
            receipt = build_receipt_from_bundle_run(
                bundle_manifest=self.manifest,
                step_results=[],
                outcome="FAILURE",
                error=error
            )
            write_receipt(self.receipt_out, receipt)
            raise BundleError(f"Unsupported step kind: {op}")
        
        if op == "READ_SYMBOL":
            return self._execute_read_symbol(step)
        elif op == "READ_SECTION":
            return self._execute_read_section(step)
        else:
            error = {
                "code": "UNSUPPORTED_STEP",
                "message": f"Unsupported step kind: {op}",
                "step_id": step.get("step_id")
            }
            
            receipt = build_receipt_from_bundle_run(
                bundle_manifest=self.manifest,
                step_results=[],
                outcome="FAILURE",
                error=error
            )
            write_receipt(self.receipt_out, receipt)
            raise BundleError(f"Unsupported step kind: {op}")
    
    def execute(self) -> Dict[str, Any]:
        """Execute bundle and emit receipt."""
        try:
            self._load_bundle()
        except BundleError as e:
            if "UNSUPPORTED_STEP" in str(e) or "VERIFY_FAILED" in str(e):
                raise
            error = {
                "code": "LOAD_FAILED",
                "message": f"Failed to load bundle",
                "step_id": None
            }
            receipt = build_receipt_from_bundle_run(
                bundle_manifest={},
                step_results=[],
                outcome="FAILURE",
                error=error
            )
            write_receipt(self.receipt_out, receipt)
            raise
        
        step_results = []
        
        for step in self.manifest["steps"]:
            try:
                result = self._execute_step(step)
                step_results.append(result)
            except BundleError as e:
                if "UNSUPPORTED_STEP" in str(e):
                    raise
                error = {
                    "code": "STEP_FAILED",
                    "message": f"Step execution failed",
                    "step_id": step.get("step_id")
                }
                receipt = build_receipt_from_bundle_run(
                    bundle_manifest=self.manifest,
                    step_results=step_results,
                    outcome="FAILURE",
                    error=error
                )
                write_receipt(self.receipt_out, receipt)
                raise
        
        receipt = build_receipt_from_bundle_run(
            bundle_manifest=self.manifest,
            step_results=step_results,
            outcome="SUCCESS",
            error=None
        )
        write_receipt(self.receipt_out, receipt)
        
        return {
            "bundle_id": self.manifest["bundle_id"],
            "receipt_path": str(self.receipt_out),
            "outcome": "SUCCESS"
        }
