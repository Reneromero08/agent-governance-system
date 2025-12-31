#!/usr/bin/env python3
"""
Compression Validator (Phase 7)

Deterministic, bounded, fail-closed validator for compression protocol.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

from .bundle import BundleVerifier, BundleError, _sha256
from .receipt import (
    find_receipt_chain,
    verify_receipt_chain,
    receipt_canonical_bytes,
    canonical_json_bytes
)


class CompressionValidationError(Exception):
    """Compression validation error with error code."""
    def __init__(self, code: str, message: str, path: Optional[str] = None):
        self.code = code
        self.message = message
        self.path = path
        super().__init__(f"[{code}] {message}")


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Serialize dict to canonical JSON bytes with trailing newline."""
    json_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return (json_str + "\n").encode('utf-8')


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (character count / 4)."""
    return round(len(text.encode('utf-8')) / 4)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().rstrip('\n')
        return json.loads(content)


def _validate_schema(obj: Dict[str, Any], schema_path: Path) -> None:
    """Validate object against JSON schema."""
    try:
        import jsonschema
    except ImportError:
        raise CompressionValidationError("INTERNAL_ERROR", "jsonschema package required for schema validation")

    if not schema_path.exists():
        raise CompressionValidationError("INTERNAL_ERROR", f"Schema not found: {schema_path}")

    schema = _load_json(schema_path)
    try:
        jsonschema.validate(instance=obj, schema=schema)
    except jsonschema.ValidationError as e:
        raise CompressionValidationError("INVALID_CLAIM_SCHEMA", f"Schema validation failed: {e.message}")


class CompressionValidator:
    """Validator for compression protocol claims."""

    def __init__(
        self,
        bundle_path: Path,
        receipts_dir: Path,
        trust_policy_path: Optional[Path],
        claim_json_path: Path
    ):
        self.bundle_path = Path(bundle_path)
        self.receipts_dir = Path(receipts_dir)
        self.trust_policy_path = Path(trust_policy_path) if trust_policy_path else None
        self.claim_json_path = Path(claim_json_path)

        self.bundle_verifier = BundleVerifier(self.bundle_path)
        self.bundle_manifest: Optional[Dict[str, Any]] = None
        self.receipts: List[Dict[str, Any]] = []
        self.trust_index: Optional[Dict[str, Any]] = None
        self.claim: Optional[Dict[str, Any]] = None

        self.errors: List[Dict[str, Any]] = []

    def _add_error(self, code: str, message: str, path: Optional[str] = None):
        """Add validation error."""
        self.errors.append({"code": code, "message": message, "path": path})

    def _fail(self, code: str, message: str, path: Optional[str] = None):
        """Raise validation error."""
        raise CompressionValidationError(code, message, path)

    def _validate_inputs(self):
        """Validate input paths exist."""
        if not self.bundle_path.exists():
            self._fail("BUNDLE_NOT_FOUND", f"Bundle path not found: {self.bundle_path}")

        bundle_json = self.bundle_path / "bundle.json"
        if not bundle_json.exists():
            self._fail("BUNDLE_NOT_FOUND", f"bundle.json not found in: {self.bundle_path}")

        if not self.receipts_dir.exists():
            self._fail("MISSING_RECEIPTS", f"Receipts directory not found: {self.receipts_dir}")

        if not self.claim_json_path.exists():
            self._fail("INVALID_CLAIM_SCHEMA", f"Claim file not found: {self.claim_json_path}")

    def _load_and_validate_claim(self):
        """Load and validate claim against schema."""
        self.claim = _load_json(self.claim_json_path)

        # Validate schema
        schema_dir = Path(__file__).parent.parent / "SCHEMAS"
        claim_schema = schema_dir / "compression_claim.schema.json"
        _validate_schema(self.claim, claim_schema)

        # Validate claim_hash
        claim_copy = dict(self.claim)
        claim_hash = claim_copy.pop("claim_hash", None)
        claim_bytes = _canonical_json_bytes(claim_copy)
        computed_hash = _sha256(claim_bytes.decode('utf-8'))

        if claim_hash != computed_hash:
            self._fail("INVALID_CLAIM_SCHEMA", f"claim_hash mismatch: expected {computed_hash}, got {claim_hash}")

    def _load_and_verify_bundle(self):
        """Load and verify bundle."""
        if self.claim is None:
            raise CompressionValidationError("INTERNAL_ERROR", "Claim not loaded")

        self.bundle_manifest = self.bundle_verifier.verify()

        # Verify bundle_id matches claim
        claim_bundle_id = self.claim.get("bundle_id")
        bundle_bundle_id = self.bundle_manifest.get("bundle_id")
        if claim_bundle_id != bundle_bundle_id:
            self._fail("BUNDLE_ID_MISMATCH", f"Claim bundle_id doesn't match bundle: {claim_bundle_id} != {bundle_bundle_id}")

        # Verify run_id matches claim
        claim_run_id = self.claim.get("run_id")
        bundle_run_id = self.bundle_manifest.get("run_id")
        if claim_run_id != bundle_run_id:
            self._fail("RUN_ID_MISMATCH", f"Claim run_id doesn't match bundle: {claim_run_id} != {bundle_run_id}")

    def _load_and_verify_trust_policy(self):
        """Load and verify trust policy."""
        if self.trust_policy_path is None:
            return

        try:
            from .trust_policy import (
                load_trust_policy_bytes,
                parse_trust_policy,
                build_trust_index
            )
        except ImportError:
            self._fail("INTERNAL_ERROR", "trust_policy module not available")

        if not self.trust_policy_path.exists():
            self._fail("VALIDATOR_NOT_FOUND", f"Trust policy not found: {self.trust_policy_path}")

        try:
            policy_bytes = load_trust_policy_bytes(self.trust_policy_path)
            policy = parse_trust_policy(policy_bytes)
            self.trust_index = build_trust_index(policy)
        except Exception as e:
            self._fail("INVALID_TRUST_POLICY_SCHEMA", f"Trust policy error: {e}")

    def _load_and_verify_receipts(self):
        """Load and verify receipt chain."""
        if self.bundle_manifest is None:
            self._fail("INTERNAL_ERROR", "Bundle manifest not loaded")
        run_id = self.bundle_manifest.get("run_id")
        if not run_id:
            self._fail("MISSING_RECEIPTS", "Bundle missing run_id")

        self.receipts = find_receipt_chain(self.receipts_dir, run_id)

        if not self.receipts:
            self._fail("MISSING_RECEIPTS", f"No receipts found for run_id: {run_id}")

        try:
            verify_receipt_chain(self.receipts, verify_attestation=False)
        except (ValueError, BundleError) as e:
            self._fail("RECEIPT_CHAIN_BROKEN", f"Receipt chain verification failed: {e}")

    def _verify_attestations(self, strict_trust: bool = False, strict_identity: bool = False):
        """Verify attestation signatures."""
        try:
            from .attestation import verify_receipt_bytes
        except ImportError:
            self._fail("INTERNAL_ERROR", "attestation module not available")

        for i, receipt in enumerate(self.receipts):
            attestation = receipt.get("attestation")
            if attestation is None:
                self._fail("ATTESTATION_MISSING", f"Receipt {i} missing attestation")

            try:
                receipt_bytes = receipt_canonical_bytes(receipt)

                # Verify signature
                verify_receipt_bytes(receipt_bytes, attestation)

                # Verify trust policy
                if strict_trust or strict_identity:
                    if self.trust_index is None:
                        self._fail("VALIDATOR_NOT_FOUND", "Strict trust enabled but no trust policy loaded")

                    validator_id = attestation.get("validator_id")
                    public_key = attestation.get("public_key", "").lower()

                    # Lookup by validator_id (primary)
                    by_validator = self.trust_index.get("by_validator_id", {})
                    by_public = self.trust_index.get("by_public_key", {})

                    if validator_id and validator_id in by_validator:
                        validator_entry = by_validator[validator_id]
                    else:
                        # Fallback lookup by public_key
                        if public_key in by_public:
                            validator_entry = by_public[public_key]
                        else:
                            self._fail("VALIDATOR_NOT_FOUND", f"Validator not found in trust policy: {validator_id or public_key}")

                    # Check enabled
                    if not validator_entry.get("enabled", False):
                        self._fail("VALIDATOR_DISABLED", f"Validator not enabled: {validator_entry.get('validator_id')}")

                    # Check scope
                    scope = validator_entry.get("scope", [])
                    if "RECEIPT" not in scope:
                        self._fail("SCOPE_MISMATCH", f"Validator scope doesn't include RECEIPT: {validator_entry.get('validator_id')}")

                    # Check build_id if strict_identity
                    if strict_identity:
                        claim_build_id = attestation.get("build_id")
                        policy_build_id = validator_entry.get("build_id")
                        if claim_build_id != policy_build_id:
                            self._fail("BUILD_ID_MISMATCH", f"build_id mismatch: {claim_build_id} != {policy_build_id}")

            except Exception as e:
                self._fail("ATTESTATION_INVALID", f"Receipt {i} attestation invalid: {e}")

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute compression metrics from verified artifacts."""
        if self.bundle_manifest is None:
            self._fail("INTERNAL_ERROR", "Bundle manifest not loaded")
        artifacts = self.bundle_manifest.get("artifacts", [])

        total_bytes = 0
        artifact_count = len(artifacts)
        symbol_artifact_count = 0
        section_artifact_count = 0

        for artifact in artifacts:
            total_bytes += artifact.get("bytes", 0)

            kind = artifact.get("kind")
            if kind == "SYMBOL_SLICE":
                symbol_artifact_count += 1
            elif kind == "SECTION_SLICE":
                section_artifact_count += 1

        # Load artifact contents to compute tokens
        uncompressed_tokens = 0
        compressed_tokens = 0
        vector_db_tokens = 0
        symbol_lang_tokens = 0
        cas_tokens = 0

        bundle_dir = self.bundle_verifier.bundle_dir
        for artifact in artifacts:
            artifact_path = bundle_dir / artifact.get("path")
            if not artifact_path.exists():
                self._fail("ARTIFACT_HASH_MISMATCH", f"Artifact file missing: {artifact_path}")

            with open(artifact_path, 'r', encoding='utf-8') as f:
                content = f.read().rstrip('\n')

            # Vector DB tokens (full content)
            vector_db_tokens += _estimate_tokens(content)

            # Symbol Lang tokens (symbol refs for SYMBOL_SLICE, full content for SECTION_SLICE)
            kind = artifact.get("kind")
            if kind == "SYMBOL_SLICE":
                # Count symbol ID itself (@prefix:name)
                ref = artifact.get("ref", "")
                if ref.startswith("@"):
                    symbol_lang_tokens += _estimate_tokens(ref)
            else:  # SECTION_SLICE
                symbol_lang_tokens += _estimate_tokens(content)

            # CAS tokens (hash only)
            cas_tokens += _estimate_tokens(artifact.get("sha256", ""))

        # Add CAS metadata overhead (10 tokens per artifact for JSON structure)
        cas_tokens += artifact_count * 10

        # Uncompressed tokens = vector_db (full expansion)
        uncompressed_tokens = vector_db_tokens

        # Compressed tokens = symbol_lang (symbol-based)
        compressed_tokens = symbol_lang_tokens

        # Compute components from claim
        if self.claim is None:
            self._fail("INTERNAL_ERROR", "Claim not loaded")
        components = self.claim.get("components", [])
        component_map = {c["name"]: c["included"] for c in components}

        # Fail if F3 is included (not implemented)
        if component_map.get("f3", False):
            self._fail("NOT_IMPLEMENTED", "F3 component is theoretical and not implemented")

        # Verify component flags match artifact composition
        expected_symbols = symbol_artifact_count > 0 or section_artifact_count > 0
        if component_map.get("symbol_lang") and not expected_symbols:
            self._fail("COMPONENT_MISMATCH", "symbol_lang component included but no symbols/sections found")

        # Build reported metrics
        reported_metrics = {
            "compression_ratio": compressed_tokens / uncompressed_tokens if uncompressed_tokens > 0 else 0.0,
            "uncompressed_tokens": uncompressed_tokens,
            "compressed_tokens": compressed_tokens,
            "artifact_count": artifact_count,
            "total_bytes": total_bytes
        }

        # Add component-specific tokens if included
        if component_map.get("vector_db_only"):
            reported_metrics["vector_db_tokens"] = vector_db_tokens

        if component_map.get("symbol_lang"):
            reported_metrics["symbol_lang_tokens"] = symbol_lang_tokens

        if component_map.get("cas"):
            reported_metrics["cas_tokens"] = cas_tokens

        return reported_metrics

    def _verify_claim_metrics(self, computed_metrics: Dict[str, Any]):
        """Verify claim metrics match computed values."""
        if self.claim is None:
            self._fail("INTERNAL_ERROR", "Claim not loaded")
        reported = self.claim.get("reported_metrics", {})

        # Check required metrics
        required_metrics = [
            "compression_ratio",
            "uncompressed_tokens",
            "compressed_tokens",
            "artifact_count",
            "total_bytes"
        ]

        for metric in required_metrics:
            if metric not in reported:
                self._fail("INVALID_CLAIM_SCHEMA", f"Claim missing required metric: {metric}")

        # Compare metrics
        tolerance = 1e-9  # Float comparison tolerance

        # Compression ratio
        claim_ratio = reported.get("compression_ratio")
        computed_ratio = computed_metrics.get("compression_ratio")
        if abs(claim_ratio - computed_ratio) > tolerance:
            self._fail("METRIC_MISMATCH", f"compression_ratio mismatch: claimed {claim_ratio}, computed {computed_ratio}")

        # Uncompressed tokens
        if reported.get("uncompressed_tokens") != computed_metrics.get("uncompressed_tokens"):
            self._fail("METRIC_MISMATCH", f"uncompressed_tokens mismatch: claimed {reported.get('uncompressed_tokens')}, computed {computed_metrics.get('uncompressed_tokens')}")

        # Compressed tokens
        if reported.get("compressed_tokens") != computed_metrics.get("compressed_tokens"):
            self._fail("METRIC_MISMATCH", f"compressed_tokens mismatch: claimed {reported.get('compressed_tokens')}, computed {computed_metrics.get('compressed_tokens')}")

        # Artifact count
        if reported.get("artifact_count") != computed_metrics.get("artifact_count"):
            self._fail("COMPONENT_MISMATCH", f"artifact_count mismatch: claimed {reported.get('artifact_count')}, computed {computed_metrics.get('artifact_count')}")

        # Total bytes
        if reported.get("total_bytes") != computed_metrics.get("total_bytes"):
            self._fail("METRIC_MISMATCH", f"total_bytes mismatch: claimed {reported.get('total_bytes')}, computed {computed_metrics.get('total_bytes')}")

        # Check optional metrics
        if self.claim is None:
            self._fail("INTERNAL_ERROR", "Claim not loaded")
        components = self.claim.get("components", [])
        component_map = {c["name"]: c["included"] for c in components}

        if component_map.get("vector_db_only"):
            if "vector_db_tokens" not in reported:
                self._fail("INVALID_CLAIM_SCHEMA", "vector_db_tokens missing but vector_db_only component included")
            if reported.get("vector_db_tokens") != computed_metrics.get("vector_db_tokens"):
                self._fail("METRIC_MISMATCH", f"vector_db_tokens mismatch: claimed {reported.get('vector_db_tokens')}, computed {computed_metrics.get('vector_db_tokens')}")

        if component_map.get("symbol_lang"):
            if "symbol_lang_tokens" not in reported:
                self._fail("INVALID_CLAIM_SCHEMA", "symbol_lang_tokens missing but symbol_lang component included")
            if reported.get("symbol_lang_tokens") != computed_metrics.get("symbol_lang_tokens"):
                self._fail("METRIC_MISMATCH", f"symbol_lang_tokens mismatch: claimed {reported.get('symbol_lang_tokens')}, computed {computed_metrics.get('symbol_lang_tokens')}")

        if component_map.get("cas"):
            if "cas_tokens" not in reported:
                self._fail("INVALID_CLAIM_SCHEMA", "cas_tokens missing but cas component included")
            if reported.get("cas_tokens") != computed_metrics.get("cas_tokens"):
                self._fail("METRIC_MISMATCH", f"cas_tokens mismatch: claimed {reported.get('cas_tokens')}, computed {computed_metrics.get('cas_tokens')}")

    def validate(
        self,
        strict_trust: bool = False,
        strict_identity: bool = False,
        require_attestation: bool = False
    ) -> Dict[str, Any]:
        """Validate compression claim.

        Returns:
            Result dict with ok, errors, computed, claim

        Raises:
            CompressionValidationError: On validation failure
        """
        try:
            # Phase 1: Input validation
            self._validate_inputs()

            # Phase 2: Load and validate claim
            self._load_and_validate_claim()

            # Phase 3: Load and verify bundle
            self._load_and_verify_bundle()

            # Phase 4: Load trust policy (if strict modes)
            if strict_trust or strict_identity:
                self._load_and_verify_trust_policy()

            # Phase 5: Load and verify receipts
            self._load_and_verify_receipts()

            # Phase 6: Verify attestations (if required)
            if require_attestation or strict_trust:
                self._verify_attestations(strict_trust=strict_trust, strict_identity=strict_identity)

            # Phase 7: Compute metrics from verified artifacts
            computed_metrics = self._compute_metrics()

            # Phase 8: Verify claim metrics
            self._verify_claim_metrics(computed_metrics)

            return {
                "ok": True,
                "errors": [],
                "computed": computed_metrics,
                "claim": self.claim
            }

        except CompressionValidationError as e:
            return {
                "ok": False,
                "errors": [{"code": e.code, "message": e.message, "path": e.path}],
                "computed": None,
                "claim": self.claim
            }
        except Exception as e:
            return {
                "ok": False,
                "errors": [{"code": "INTERNAL_ERROR", "message": str(e), "path": None}],
                "computed": None,
                "claim": self.claim
            }


def validate_compression_claim(
    bundle_path: str,
    receipts_dir: str,
    trust_policy_path: Optional[str],
    claim_json_path: str,
    strict_trust: bool = False,
    strict_identity: bool = False,
    require_attestation: bool = False
) -> Dict[str, Any]:
    """Entry function for compression claim validation.

    Args:
        bundle_path: Path to bundle directory or bundle.json
        receipts_dir: Path to receipts directory
        trust_policy_path: Path to trust policy (optional)
        claim_json_path: Path to claim JSON file
        strict_trust: Enable strict trust verification
        strict_identity: Enable strict identity pinning
        require_attestation: Require attestation on all receipts

    Returns:
        Result dict with ok, errors, computed, claim
    """
    validator = CompressionValidator(
        bundle_path=Path(bundle_path),
        receipts_dir=Path(receipts_dir),
        trust_policy_path=Path(trust_policy_path) if trust_policy_path else None,
        claim_json_path=Path(claim_json_path)
    )

    return validator.validate(
        strict_trust=strict_trust,
        strict_identity=strict_identity,
        require_attestation=require_attestation
    )
