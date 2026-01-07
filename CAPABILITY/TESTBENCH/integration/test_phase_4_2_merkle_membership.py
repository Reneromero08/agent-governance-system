#!/usr/bin/env python3
"""
Phase 4.2: Merkle Membership Proof Integration Tests

Tests for:
- 4.2.1: Merkle membership proofs in restore_proof.py
- 4.2.2: Catalytic runtime --full-proofs integration
- 4.2.3: verify_file.py CLI
- 4.2.4: End-to-end proof generation and verification

Exit Criteria:
- Proofs generated when include_membership_proofs=True
- No proofs when include_membership_proofs=False (default)
- verify_file_membership() validates correctly
- CLI tools work end-to-end
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.restore_proof import (
    RestorationProofValidator,
    verify_file_membership,
    compute_manifest_root_with_proofs,
)
from CAPABILITY.PRIMITIVES.merkle import MerkleProof, build_manifest_with_proofs


def _rm(path: Path) -> None:
    """Cleanup helper."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _sha256_hex(data: bytes) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# 4.2.1 Tests: Merkle Membership in Restore Proofs
# =============================================================================


class TestRestoreProofMembership:
    """Tests for restore_proof.py Merkle membership integration."""

    @pytest.fixture
    def proof_schema_path(self) -> Path:
        return REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"

    @pytest.fixture
    def sample_state(self) -> dict:
        """Sample pre/post state for testing."""
        return {
            "test_domain": {
                "file1.txt": _sha256_hex(b"content1"),
                "file2.txt": _sha256_hex(b"content2"),
                "subdir/file3.txt": _sha256_hex(b"content3"),
            }
        }

    def test_proofs_not_included_by_default(
        self, proof_schema_path: Path, sample_state: dict
    ):
        """Membership proofs should NOT be included when not requested."""
        validator = RestorationProofValidator(proof_schema_path)
        proof = validator.generate_proof(
            run_id="test-no-proofs",
            catalytic_domains=["test_domain"],
            pre_state=sample_state,
            post_state=sample_state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=False,  # explicit default
        )

        # No membership_proofs in pre_state or post_state
        assert "membership_proofs" not in proof["pre_state"]
        assert "membership_proofs" not in proof["post_state"]

    def test_proofs_included_when_requested(
        self, proof_schema_path: Path, sample_state: dict
    ):
        """Membership proofs should be included when requested."""
        validator = RestorationProofValidator(proof_schema_path)
        proof = validator.generate_proof(
            run_id="test-with-proofs",
            catalytic_domains=["test_domain"],
            pre_state=sample_state,
            post_state=sample_state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        # membership_proofs should be present
        assert "membership_proofs" in proof["pre_state"]
        assert "membership_proofs" in proof["post_state"]

        # Should have one proof per file
        pre_proofs = proof["pre_state"]["membership_proofs"]
        assert len(pre_proofs) == 3

        # Each proof should have required fields
        for path, proof_dict in pre_proofs.items():
            assert "path" in proof_dict
            assert "bytes_hash" in proof_dict
            assert "steps" in proof_dict

    def test_proofs_verify_correctly(
        self, proof_schema_path: Path, sample_state: dict
    ):
        """Generated proofs should verify correctly."""
        validator = RestorationProofValidator(proof_schema_path)
        proof = validator.generate_proof(
            run_id="test-verify-proofs",
            catalytic_domains=["test_domain"],
            pre_state=sample_state,
            post_state=sample_state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        pre_proofs = proof["pre_state"]["membership_proofs"]
        root = proof["pre_state"]["domain_root_hash"]

        # Verify each file's membership
        for path, proof_dict in pre_proofs.items():
            bytes_hash = proof_dict["bytes_hash"]
            assert verify_file_membership(path, bytes_hash, proof_dict, root), \
                f"Proof for {path} should verify"

    def test_tampered_proof_rejected(
        self, proof_schema_path: Path, sample_state: dict
    ):
        """Tampered proofs should be rejected."""
        validator = RestorationProofValidator(proof_schema_path)
        proof = validator.generate_proof(
            run_id="test-tampered",
            catalytic_domains=["test_domain"],
            pre_state=sample_state,
            post_state=sample_state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        pre_proofs = proof["pre_state"]["membership_proofs"]
        root = proof["pre_state"]["domain_root_hash"]

        # Get first proof
        path = list(pre_proofs.keys())[0]
        proof_dict = pre_proofs[path].copy()

        # Tamper with bytes_hash
        original_hash = proof_dict["bytes_hash"]
        tampered_hash = original_hash[:-1] + ("0" if original_hash[-1] != "0" else "1")

        assert not verify_file_membership(path, tampered_hash, proof_dict, root), \
            "Tampered hash should be rejected"


# =============================================================================
# 4.2.2 Tests: compute_manifest_root_with_proofs
# =============================================================================


class TestComputeManifestRootWithProofs:
    """Tests for compute_manifest_root_with_proofs function."""

    @pytest.fixture
    def sample_manifest(self) -> dict:
        return {
            "a.txt": _sha256_hex(b"a"),
            "b.txt": _sha256_hex(b"b"),
            "c/d.txt": _sha256_hex(b"cd"),
        }

    def test_no_proofs_by_default(self, sample_manifest: dict):
        """No proofs returned when include_proofs=False."""
        root, proofs = compute_manifest_root_with_proofs(sample_manifest, include_proofs=False)
        assert root
        assert proofs is None

    def test_proofs_when_requested(self, sample_manifest: dict):
        """Proofs returned when include_proofs=True."""
        root, proofs = compute_manifest_root_with_proofs(sample_manifest, include_proofs=True)
        assert root
        assert proofs is not None
        assert len(proofs) == 3

    def test_empty_manifest(self):
        """Empty manifest returns sentinel hash and no proofs."""
        root, proofs = compute_manifest_root_with_proofs({}, include_proofs=True)
        assert root == hashlib.sha256(b"").hexdigest()
        assert proofs is None

    def test_root_matches_build_manifest_with_proofs(self, sample_manifest: dict):
        """Root from compute_manifest_root_with_proofs matches build_manifest_with_proofs."""
        root1, proofs1 = compute_manifest_root_with_proofs(sample_manifest, include_proofs=True)
        root2, proofs2 = build_manifest_with_proofs(sample_manifest)

        assert root1 == root2


# =============================================================================
# 4.2.3 Tests: verify_file_membership function
# =============================================================================


class TestVerifyFileMembership:
    """Tests for verify_file_membership function."""

    def test_valid_proof_verifies(self):
        """Valid proof verifies successfully."""
        manifest = {
            "file1.txt": _sha256_hex(b"content1"),
            "file2.txt": _sha256_hex(b"content2"),
        }
        root, proofs = build_manifest_with_proofs(manifest)

        for path, bytes_hash in manifest.items():
            proof = proofs[path]
            assert verify_file_membership(path, bytes_hash, proof, root)

    def test_wrong_root_rejected(self):
        """Wrong root rejects proof."""
        manifest = {"file.txt": _sha256_hex(b"content")}
        root, proofs = build_manifest_with_proofs(manifest)

        path = "file.txt"
        bytes_hash = manifest[path]
        proof = proofs[path]

        wrong_root = root[:-1] + ("0" if root[-1] != "0" else "1")
        assert not verify_file_membership(path, bytes_hash, proof, wrong_root)

    def test_dict_and_object_both_work(self):
        """Both MerkleProof objects and dicts work."""
        manifest = {"file.txt": _sha256_hex(b"content")}
        root, proofs = build_manifest_with_proofs(manifest)

        path = "file.txt"
        bytes_hash = manifest[path]
        proof_obj = proofs[path]
        proof_dict = proof_obj.to_dict()

        # Object works
        assert verify_file_membership(path, bytes_hash, proof_obj, root)
        # Dict works
        assert verify_file_membership(path, bytes_hash, proof_dict, root)


# =============================================================================
# 4.2.4 Tests: End-to-End Integration
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end tests for the full Merkle membership proof workflow."""

    @pytest.fixture
    def test_dir(self) -> Path:
        """Create and cleanup test directory."""
        test_path = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "test_4_2_e2e"
        _rm(test_path)
        test_path.mkdir(parents=True, exist_ok=True)
        yield test_path
        _rm(test_path)

    def test_full_workflow_with_proofs(self, test_dir: Path):
        """
        Full workflow: generate proof with membership proofs, write to JSON, read back, verify.
        """
        proof_schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"
        validator = RestorationProofValidator(proof_schema_path)

        # Create state
        state = {
            "domain1": {
                "file1.txt": _sha256_hex(b"file 1 content"),
                "file2.txt": _sha256_hex(b"file 2 content"),
            }
        }

        # Generate proof with membership proofs
        proof = validator.generate_proof(
            run_id="e2e-test",
            catalytic_domains=["domain1"],
            pre_state=state,
            post_state=state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        # Write to file
        proof_file = test_dir / "PROOF.json"
        proof_file.write_text(json.dumps(proof, indent=2), encoding="utf-8")

        # Read back
        loaded_proof = json.loads(proof_file.read_text(encoding="utf-8"))

        # Verify each file
        pre_proofs = loaded_proof["pre_state"]["membership_proofs"]
        root = loaded_proof["pre_state"]["domain_root_hash"]

        for path, proof_dict in pre_proofs.items():
            bytes_hash = proof_dict["bytes_hash"]
            assert verify_file_membership(path, bytes_hash, proof_dict, root), \
                f"Loaded proof for {path} should verify"

    def test_proof_roundtrip_deterministic(self, test_dir: Path):
        """
        Same state should produce identical proofs (determinism).
        """
        proof_schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"
        validator = RestorationProofValidator(proof_schema_path)

        state = {
            "domain1": {
                "a.txt": _sha256_hex(b"a"),
                "b.txt": _sha256_hex(b"b"),
            }
        }

        proof1 = validator.generate_proof(
            run_id="det-test-1",
            catalytic_domains=["domain1"],
            pre_state=state,
            post_state=state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        proof2 = validator.generate_proof(
            run_id="det-test-1",  # same run_id
            catalytic_domains=["domain1"],
            pre_state=state,
            post_state=state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        # Domain root hashes should be identical
        assert proof1["pre_state"]["domain_root_hash"] == proof2["pre_state"]["domain_root_hash"]

        # Proof steps should be identical
        for path in state["domain1"]:
            proof1_steps = proof1["pre_state"]["membership_proofs"][path]["steps"]
            proof2_steps = proof2["pre_state"]["membership_proofs"][path]["steps"]
            assert proof1_steps == proof2_steps


# =============================================================================
# CLI Tests (verify_file.py)
# =============================================================================


class TestVerifyFileCLI:
    """Tests for verify_file.py CLI."""

    @pytest.fixture
    def test_proof_file(self, tmp_path: Path) -> Path:
        """Create a test PROOF.json with membership proofs."""
        proof_schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"
        validator = RestorationProofValidator(proof_schema_path)

        state = {
            "domain": {
                "test.txt": _sha256_hex(b"test content"),
            }
        }

        proof = validator.generate_proof(
            run_id="cli-test",
            catalytic_domains=["domain"],
            pre_state=state,
            post_state=state,
            timestamp="2025-01-01T00:00:00Z",
            include_membership_proofs=True,
        )

        proof_file = tmp_path / "PROOF.json"
        proof_file.write_text(json.dumps(proof), encoding="utf-8")
        return proof_file

    def test_cli_verify_success(self, test_proof_file: Path):
        """CLI should return success for valid proof."""
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "verify_file.py"),
                "--proof-file", str(test_proof_file),
                "--path", "test.txt",
                "--state", "post",
                "--json",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI should succeed: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert output["code"] == "VERIFIED"

    def test_cli_verify_nonexistent_file(self, test_proof_file: Path):
        """CLI should fail for file not in proofs."""
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "verify_file.py"),
                "--proof-file", str(test_proof_file),
                "--path", "nonexistent.txt",
                "--state", "post",
                "--json",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["ok"] is False
        assert output["code"] == "FILE_PROOF_NOT_FOUND"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
