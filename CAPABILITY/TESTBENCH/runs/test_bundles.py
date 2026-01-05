"""
Tests for Run Bundle Contract (Task 2.3)

Validates:
- Bundle creation and determinism
- Bundle verification (dry-run verifier)
- GC rooting semantics
- Immutability and fail-closed behavior
"""

import pytest
import json
from pathlib import Path
from typing import List

from CAPABILITY.RUNS.bundles import (
    run_bundle_create,
    run_bundle_verify,
    get_bundle_roots,
    BundleVerificationReceipt,
    RUN_BUNDLE_VERSION,
    REQUIRED_ARTIFACTS,
)
from CAPABILITY.RUNS.records import (
    put_task_spec,
    put_status,
    put_output_hashes,
)
from CAPABILITY.CAS import cas as cas_mod


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_run_artifacts():
    """Create sample run artifacts in CAS."""
    task_spec = {"action": "test", "input": 42}
    task_spec_hash = put_task_spec(task_spec)
    
    status = {"state": "SUCCESS"}
    status_hash = put_status(status)
    
    # Create some output artifacts
    output1 = cas_mod.cas_put(b"output data 1")
    output2 = cas_mod.cas_put(b"output data 2")
    output_hashes_hash = put_output_hashes([output1, output2])
    
    return {
        "task_spec_hash": task_spec_hash,
        "status_hash": status_hash,
        "output_hashes_hash": output_hashes_hash,
        "output_hashes": [output1, output2],
    }


# ============================================================================
# Bundle Creation Tests (2.3.2)
# ============================================================================

class TestBundleCreation:
    """Test run_bundle_create functionality."""
    
    def test_create_minimal_bundle(self, sample_run_artifacts):
        """Create a minimal valid bundle."""
        bundle_ref = run_bundle_create(
            run_id="test-run-001",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        # Validate reference format
        assert bundle_ref.startswith("sha256:")
        bundle_hash = bundle_ref.split(":", 1)[1]
        assert len(bundle_hash) == 64
        assert all(c in "0123456789abcdef" for c in bundle_hash)
    
    def test_create_bundle_with_receipts(self, sample_run_artifacts):
        """Create bundle with receipt references."""
        receipt1 = cas_mod.cas_put(b'{"receipt": "data1"}')
        receipt2 = cas_mod.cas_put(b'{"receipt": "data2"}')
        
        bundle_ref = run_bundle_create(
            run_id="test-run-002",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            receipts=[receipt1, receipt2],
        )
        
        assert bundle_ref.startswith("sha256:")
    
    def test_create_bundle_with_metadata(self, sample_run_artifacts):
        """Create bundle with metadata."""
        metadata = {
            "timestamp": "2026-01-04T22:00:00Z",
            "executor": "test-agent",
            "duration_ms": 1234,
        }
        
        bundle_ref = run_bundle_create(
            run_id="test-run-003",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            metadata=metadata,
        )
        
        assert bundle_ref.startswith("sha256:")
    
    def test_bundle_determinism(self, sample_run_artifacts):
        """Same inputs produce same bundle hash."""
        bundle_ref1 = run_bundle_create(
            run_id="test-run-004",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        bundle_ref2 = run_bundle_create(
            run_id="test-run-004",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        assert bundle_ref1 == bundle_ref2
    
    def test_different_run_ids_different_bundles(self, sample_run_artifacts):
        """Different run_ids produce different bundles."""
        bundle_ref1 = run_bundle_create(
            run_id="test-run-005a",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        bundle_ref2 = run_bundle_create(
            run_id="test-run-005b",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        assert bundle_ref1 != bundle_ref2
    
    def test_reject_invalid_run_id(self, sample_run_artifacts):
        """Invalid run_id is rejected."""
        with pytest.raises(ValueError, match="run_id"):
            run_bundle_create(
                run_id="",
                task_spec_hash=sample_run_artifacts["task_spec_hash"],
                status_hash=sample_run_artifacts["status_hash"],
                output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            )
    
    def test_reject_invalid_hash_format(self, sample_run_artifacts):
        """Invalid hash format is rejected."""
        with pytest.raises(ValueError):
            run_bundle_create(
                run_id="test-run-006",
                task_spec_hash="not-a-hash",
                status_hash=sample_run_artifacts["status_hash"],
                output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            )


# ============================================================================
# Bundle Verification Tests (2.3.4)
# ============================================================================

class TestBundleVerification:
    """Test run_bundle_verify functionality."""
    
    def test_verify_valid_bundle(self, sample_run_artifacts):
        """Verify a valid bundle."""
        bundle_ref = run_bundle_create(
            run_id="test-run-007",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        receipt = run_bundle_verify(bundle_ref)
        
        assert receipt.verification_status == "VALID"
        assert receipt.manifest_valid is True
        assert receipt.all_artifacts_present is True
        assert len(receipt.errors) == 0
        assert receipt.artifact_status["task_spec"] is True
        assert receipt.artifact_status["status"] is True
        assert receipt.artifact_status["output_hashes"] is True
    
    def test_verify_bundle_with_receipts(self, sample_run_artifacts):
        """Verify bundle with receipts."""
        receipt1 = cas_mod.cas_put(b'{"receipt": "data1"}')
        receipt2 = cas_mod.cas_put(b'{"receipt": "data2"}')
        
        bundle_ref = run_bundle_create(
            run_id="test-run-008",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            receipts=[receipt1, receipt2],
        )
        
        receipt = run_bundle_verify(bundle_ref)
        
        assert receipt.verification_status == "VALID"
        assert receipt.artifact_status["receipt[0]"] is True
        assert receipt.artifact_status["receipt[1]"] is True
    
    def test_verify_invalid_bundle_ref(self):
        """Invalid bundle reference is rejected."""
        receipt = run_bundle_verify("not-a-valid-ref")
        
        assert receipt.verification_status == "INVALID"
        assert len(receipt.errors) > 0
        assert any("Invalid bundle_ref format" in e for e in receipt.errors)
    
    def test_verify_missing_bundle(self):
        """Missing bundle manifest is detected."""
        fake_ref = "sha256:" + ("0" * 64)
        
        receipt = run_bundle_verify(fake_ref)
        
        assert receipt.verification_status == "INVALID"
        assert len(receipt.errors) > 0
        assert any("not found in CAS" in e for e in receipt.errors)
    
    def test_verify_missing_artifact(self, sample_run_artifacts):
        """Missing artifact is detected."""
        # Create bundle with a fake hash
        fake_hash = "f" * 64
        
        bundle_ref = run_bundle_create(
            run_id="test-run-009",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=fake_hash,  # This doesn't exist
        )
        
        receipt = run_bundle_verify(bundle_ref)
        
        assert receipt.verification_status == "INVALID"
        assert receipt.artifact_status["output_hashes"] is False
        assert any("not found in CAS" in e for e in receipt.errors)
    
    def test_verification_determinism(self, sample_run_artifacts):
        """Verification is deterministic."""
        bundle_ref = run_bundle_create(
            run_id="test-run-010",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        receipt1 = run_bundle_verify(bundle_ref)
        receipt2 = run_bundle_verify(bundle_ref)
        
        assert receipt1.verification_status == receipt2.verification_status
        assert receipt1.manifest_valid == receipt2.manifest_valid
        assert receipt1.all_artifacts_present == receipt2.all_artifacts_present
        assert receipt1.artifact_status == receipt2.artifact_status


# ============================================================================
# GC Rooting Tests (2.3.3)
# ============================================================================

class TestGCRooting:
    """Test GC rooting semantics."""
    
    def test_get_bundle_roots_minimal(self, sample_run_artifacts):
        """Get roots from minimal bundle."""
        bundle_ref = run_bundle_create(
            run_id="test-run-011",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        roots = get_bundle_roots(bundle_ref)
        
        # Should include: bundle itself + 3 required artifacts + output hashes
        bundle_hash = bundle_ref.split(":", 1)[1]
        assert bundle_hash in roots
        assert sample_run_artifacts["task_spec_hash"] in roots
        assert sample_run_artifacts["status_hash"] in roots
        assert sample_run_artifacts["output_hashes_hash"] in roots
        assert sample_run_artifacts["output_hashes"][0] in roots
        assert sample_run_artifacts["output_hashes"][1] in roots
    
    def test_get_bundle_roots_with_receipts(self, sample_run_artifacts):
        """Get roots from bundle with receipts."""
        receipt1 = cas_mod.cas_put(b'{"receipt": "data1"}')
        receipt2 = cas_mod.cas_put(b'{"receipt": "data2"}')
        
        bundle_ref = run_bundle_create(
            run_id="test-run-012",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
            receipts=[receipt1, receipt2],
        )
        
        roots = get_bundle_roots(bundle_ref)
        
        # Should include receipts
        assert receipt1 in roots
        assert receipt2 in roots
    
    def test_roots_are_sorted(self, sample_run_artifacts):
        """Roots are returned in sorted order."""
        bundle_ref = run_bundle_create(
            run_id="test-run-013",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        roots = get_bundle_roots(bundle_ref)
        
        assert roots == sorted(roots)
    
    def test_roots_determinism(self, sample_run_artifacts):
        """Root extraction is deterministic."""
        bundle_ref = run_bundle_create(
            run_id="test-run-014",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        roots1 = get_bundle_roots(bundle_ref)
        roots2 = get_bundle_roots(bundle_ref)
        
        assert roots1 == roots2
    
    def test_roots_for_invalid_bundle(self):
        """Gracefully handle invalid bundle."""
        fake_ref = "sha256:" + ("0" * 64)
        
        roots = get_bundle_roots(fake_ref)
        
        # Should at least return the bundle hash itself
        assert "0" * 64 in roots


# ============================================================================
# Integration Tests
# ============================================================================

class TestBundleIntegration:
    """End-to-end integration tests."""
    
    def test_complete_run_lifecycle(self):
        """Complete run: create artifacts, bundle, verify."""
        # 1. Create run artifacts
        task_spec = {
            "action": "process",
            "input": {"file": "data.csv"},
            "params": {"format": "csv"},
        }
        task_spec_hash = put_task_spec(task_spec)
        
        status = {
            "state": "SUCCESS",
            "started_at": "2026-01-04T22:00:00Z",
            "completed_at": "2026-01-04T22:00:05Z",
        }
        status_hash = put_status(status)
        
        # Create outputs
        output1 = cas_mod.cas_put(b"processed result 1")
        output2 = cas_mod.cas_put(b"processed result 2")
        output_hashes_hash = put_output_hashes([output1, output2])
        
        # Create receipt
        receipt_data = {
            "run_id": "integration-test-001",
            "result": "SUCCESS",
        }
        receipt_hash = cas_mod.cas_put(json.dumps(receipt_data).encode())
        
        # 2. Create bundle
        bundle_ref = run_bundle_create(
            run_id="integration-test-001",
            task_spec_hash=task_spec_hash,
            status_hash=status_hash,
            output_hashes_hash=output_hashes_hash,
            receipts=[receipt_hash],
            metadata={
                "timestamp": "2026-01-04T22:00:05Z",
                "executor": "test-agent",
            },
        )
        
        # 3. Verify bundle
        verification = run_bundle_verify(bundle_ref)
        assert verification.verification_status == "VALID"
        
        # 4. Get GC roots
        roots = get_bundle_roots(bundle_ref)
        assert len(roots) >= 6  # bundle + 3 artifacts + 2 outputs + receipt
        
        # 5. Verify all artifacts are reachable
        bundle_hash = bundle_ref.split(":", 1)[1]
        assert bundle_hash in roots
        assert task_spec_hash in roots
        assert status_hash in roots
        assert output_hashes_hash in roots
        assert output1 in roots
        assert output2 in roots
        assert receipt_hash in roots
    
    def test_bundle_is_proof_carrying(self, sample_run_artifacts):
        """Bundle is a proof-carrying record."""
        bundle_ref = run_bundle_create(
            run_id="proof-test-001",
            task_spec_hash=sample_run_artifacts["task_spec_hash"],
            status_hash=sample_run_artifacts["status_hash"],
            output_hashes_hash=sample_run_artifacts["output_hashes_hash"],
        )
        
        # Load the bundle manifest
        bundle_hash = bundle_ref.split(":", 1)[1]
        manifest_bytes = cas_mod.cas_get(bundle_hash)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        
        # Verify manifest structure
        assert manifest["version"] == RUN_BUNDLE_VERSION
        assert manifest["run_id"] == "proof-test-001"
        assert "task_spec_hash" in manifest
        assert "status_hash" in manifest
        assert "output_hashes_hash" in manifest
        
        # Verify all referenced artifacts exist
        verification = run_bundle_verify(bundle_ref)
        assert verification.verification_status == "VALID"
        assert verification.all_artifacts_present is True
