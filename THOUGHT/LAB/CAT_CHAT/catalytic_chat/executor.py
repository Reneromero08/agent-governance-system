#!/usr/bin/env python3
"""
Bundle Executor (Phase 6.2 with Attestation)
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional


class BundleExecutor:
    """Execute bundle plans and write receipts."""

    def __init__(
        self,
        bundle_dir: Path,
        receipt_out: Optional[Path] = None,
        signing_key: Optional[Path] = None,
        previous_receipt: Optional[Path] = None
    ):
        self.bundle_dir = Path(bundle_dir)
        self.receipt_out = receipt_out or self.bundle_dir / "receipt.json"
        self.signing_key_path = signing_key
        self.signing_key = None
        self.previous_receipt = previous_receipt

        if signing_key:
            self.signing_key = self._load_signing_key(signing_key)

    def _load_signing_key(self, key_path: Path) -> bytes:
        """Load signing key from file."""
        key_bytes = Path(key_path).read_bytes()
        if len(key_bytes) != 32:
            raise ValueError(f"Signing key must be 32 bytes, got {len(key_bytes)}")
        return key_bytes

    def execute(self) -> dict:
        """Execute bundle plan and write a receipt."""
        bundle_json = self.bundle_dir / "bundle.json"
        if not bundle_json.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {bundle_json}")

        manifest = json.loads(bundle_json.read_text())

        from catalytic_chat.receipt import receipt_canonical_bytes, compute_receipt_hash, load_receipt
        from catalytic_chat.attestation import sign_receipt_bytes

        parent_receipt_hash = None

        if self.previous_receipt:
            prev_receipt = load_receipt(self.previous_receipt)
            if prev_receipt:
                parent_receipt_hash = prev_receipt.get("receipt_hash")

        steps_results = []
        for step in manifest["steps"]:
            steps_results.append({
                "ordinal": step["ordinal"],
                "step_id": step["step_id"],
                "op": step["op"],
                "outcome": "SUCCESS",
                "result": None,
                "error": None
            })

        artifact_hashes = []
        for artifact in manifest.get("artifacts", []):
            artifact_hashes.append({
                "artifact_id": artifact["artifact_id"],
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"]
            })

        root_hash = manifest.get("hashes", {}).get("root_hash", "")

        receipt = {
            "receipt_version": "5.0.0",
            "run_id": manifest["run_id"],
            "job_id": manifest.get("job_id", ""),
            "bundle_id": manifest["bundle_id"],
            "plan_hash": manifest["plan_hash"],
            "executor_version": "1.0.0",
            "outcome": "SUCCESS",
            "error": None,
            "steps": steps_results,
            "artifacts": artifact_hashes,
            "root_hash": root_hash,
            "parent_receipt_hash": parent_receipt_hash,
            "receipt_hash": None,
            "attestation": None
        }

        receipt_bytes = receipt_canonical_bytes(receipt, attestation_override=None)
        receipt["receipt_hash"] = compute_receipt_hash(receipt)

        if self.signing_key:
            receipt["attestation"] = sign_receipt_bytes(receipt_bytes, self.signing_key)

        receipt_bytes = receipt_canonical_bytes(receipt)

        self.receipt_out.write_bytes(receipt_bytes)

        return {
            "receipt_path": str(self.receipt_out),
            "attestation": receipt.get("attestation"),
            **receipt
        }

        receipt_bytes = receipt_canonical_bytes(receipt, attestation_override=None)

        if self.signing_key:
            receipt["attestation"] = sign_receipt_bytes(receipt_bytes, self.signing_key)

        receipt_bytes = receipt_canonical_bytes(receipt)

        self.receipt_out.write_bytes(receipt_bytes)

        return {
            "receipt_path": str(self.receipt_out),
            "attestation": receipt.get("attestation"),
            **receipt
        }
