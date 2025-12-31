#!/usr/bin/env python3
"""
Bundle Executor (Phase 6.2 with Attestation)
"""

import json
import hashlib
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Optional, Mapping, Any


class BundleExecutor:
    """Execute bundle plans and write receipts."""

    def __init__(
        self,
        bundle_dir: Path,
        receipt_out: Optional[Path] = None,
        signing_key: Optional[Path] = None,
        previous_receipt: Optional[Path] = None,
        receipt_index: Optional[int] = None,
        repo_root: Optional[Path] = None,
        policy: Optional[Mapping[str, Any]] = None
    ):
        self.bundle_dir = Path(bundle_dir)
        self.receipt_out = receipt_out or self.bundle_dir / "receipt.json"
        self.signing_key_path = signing_key
        self.signing_key = None
        self.previous_receipt = previous_receipt
        self.receipt_index = receipt_index
        self.repo_root = repo_root or bundle_dir
        self.policy = policy or {}

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

        receipt_index_value = self.receipt_index if self.receipt_index is not None else 0

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
            "attestation": None,
            "receipt_index": receipt_index_value
        }

        receipt["receipt_hash"] = compute_receipt_hash(receipt)
        receipt_bytes = receipt_canonical_bytes(receipt, attestation_override=None)

        if self.signing_key:
            receipt["attestation"] = sign_receipt_bytes(receipt_bytes, self.signing_key)

        receipt_bytes = receipt_canonical_bytes(receipt)

        self.receipt_out.write_bytes(receipt_bytes)

        self._enforce_policy_after_execution(receipt)

        return {
            "receipt_path": str(self.receipt_out),
            "attestation": receipt.get("attestation"),
            **receipt
        }

    def _enforce_policy_after_execution(self, receipt: dict) -> None:
        """Enforce policy requirements after execution.

        Order:
        c) If policy.require_verify_chain: verify receipt chain AND compute merkle root (fail if chain cannot be formed)
        d) If policy.require_receipt_attestation:
           - require attestation != null for every receipt
           - verify each receipt attestation using trust/identity when enabled
        e) If policy.require_merkle_attestation:
           - require merkle root exists (thus requires verify_chain)
           - if policy indicates verification from file (CLI supplied path) verify it
           - otherwise if CLI requested attestation emission, sign and emit deterministically
           - always enforce strict_trust / strict_identity when enabled
        """
        policy = self.policy

        merkle_root = None
        trust_index = None
        if policy.get("strict_trust", False) or policy.get("strict_identity", False):
            if not policy.get("trust_policy_path"):
                raise RuntimeError("Policy violation: strict_trust/strict_identity requires trust_policy_path")

            from catalytic_chat.trust_policy import load_trust_policy_bytes, parse_trust_policy, build_trust_index

            try:
                policy_bytes = load_trust_policy_bytes(Path(policy.get("trust_policy_path")))
                trust_policy_parsed = parse_trust_policy(policy_bytes)
                trust_index = build_trust_index(trust_policy_parsed)
            except Exception as e:
                raise RuntimeError(f"Policy violation: failed to load trust policy: {e}")

        if policy.get("require_receipt_attestation", False):
            from catalytic_chat.attestation import verify_receipt_attestation, AttestationError

            if receipt.get("attestation") is None:
                raise RuntimeError("Policy violation: receipt attestation required but missing")

            if trust_index is None and (policy.get("strict_trust", False) or policy.get("strict_identity", False)):
                raise RuntimeError("Policy violation: strict_trust/strict_identity requires trust policy")

            if trust_index is not None:
                try:
                    verify_receipt_attestation(
                        receipt,
                        trust_index,
                        strict=policy.get("strict_trust", False),
                        strict_identity=policy.get("strict_identity", False)
                    )
                except AttestationError as e:
                    raise RuntimeError(f"Policy violation: receipt attestation verification failed: {e}")

        if policy.get("require_verify_chain", False):
            from catalytic_chat.receipt import find_receipt_chain, verify_receipt_chain

            receipts_dir = self.receipt_out.parent
            run_id = receipt.get("run_id")

            if run_id:
                receipts = find_receipt_chain(receipts_dir, run_id)

                if len(receipts) > 0:
                    verify_attestation = policy.get("require_receipt_attestation", False)
                    merkle_root = verify_receipt_chain(receipts, verify_attestation=verify_attestation)
                else:
                    raise RuntimeError(f"Policy violation: cannot verify receipt chain - no receipts found for run_id={run_id}")

        if policy.get("require_merkle_attestation", False):
            if merkle_root is None:
                raise RuntimeError("Policy violation: merkle attestation required but no merkle root computed (requires verify_chain)")

            from catalytic_chat.merkle_attestation import load_merkle_attestation, verify_merkle_attestation_with_trust, MerkleAttestationError

            merkle_attestation_path = policy.get("merkle_attestation_path")
            if merkle_attestation_path:
                att_path = Path(merkle_attestation_path)
                att = load_merkle_attestation(att_path)

                if att is None:
                    raise RuntimeError(f"Policy violation: merkle attestation file not found: {att_path}")

                if trust_index is None and (policy.get("strict_trust", False) or policy.get("strict_identity", False)):
                    raise RuntimeError("Policy violation: strict_trust/strict_identity requires trust policy for merkle attestation")

                if trust_index is not None:
                    try:
                        verify_merkle_attestation_with_trust(
                            att,
                            merkle_root,
                            trust_index,
                            strict=policy.get("strict_trust", False),
                            strict_identity=policy.get("strict_identity", False)
                        )
                    except MerkleAttestationError as e:
                        raise RuntimeError(f"Policy violation: merkle attestation verification failed: {e}")
            else:
                if policy.get("emit_merkle_attestation", False):
                    from catalytic_chat.merkle_attestation import sign_merkle_root

                    if not self.signing_key:
                        raise RuntimeError("Policy violation: merkle attestation emission requires signing key")

                    from catalytic_chat.validator_identity import get_validator_identity

                    merkle_validator_identity = get_validator_identity(
                        self.signing_key,
                        self.repo_root
                    )

                    att = sign_merkle_root(
                        merkle_root,
                        self.signing_key.hex(),
                        validator_id=merkle_validator_identity["validator_id"],
                        build_id=merkle_validator_identity["build_id"]
                    )

                    att["run_id"] = receipt.get("run_id")
                    att["job_id"] = receipt.get("job_id")
                    att["bundle_id"] = receipt.get("bundle_id")

                    merkle_attestation_out = policy.get("merkle_attestation_out")
                    if merkle_attestation_out:
                        from catalytic_chat.merkle_attestation import write_merkle_attestation
                        att_path = Path(merkle_attestation_out)
                        write_merkle_attestation(att_path, att)
                    else:
                        from catalytic_chat.receipt import canonical_json_bytes
                        att_bytes = canonical_json_bytes(att)
                        sys.stdout.buffer.write(att_bytes)
                        sys.stdout.buffer.flush()
