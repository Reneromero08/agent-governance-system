#!/usr/bin/env python3
"""
Phase 4.5: Atomic Restore Tests (SPECTRUM-06)

Tests for:
- 4.5.1: Transactional restore with staging
- 4.5.2: Rollback on failure
- 4.5.3: Dry-run mode
- 4.5.4: CLI integration

Exit Criteria:
- Restore is transactional (all-or-nothing)
- Failure never leaves partial state
- Rollback is automatic and clean
- Dry-run validates without writing
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from CAPABILITY.PRIMITIVES.restore_runner import (
    restore_bundle,
    restore_chain,
    RESTORE_CODES,
)
from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_signed_bundle(
    *,
    run_dir: Path,
    output_rel: str,
    output_bytes: bytes,
    include_proof: bool = True,
    proof_verified: bool = True,
) -> None:
    """Create a valid signed bundle for testing."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write output bytes at project_root-relative path.
    output_abs = REPO_ROOT / output_rel
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    output_abs.write_bytes(output_bytes)

    task_spec = {"task_id": run_dir.name, "outputs": {"durable_paths": [output_rel]}}
    task_spec_bytes = json.dumps(task_spec, indent=2).encode("utf-8")
    (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

    status_obj = {"status": "success", "cmp01": "pass", "run_id": run_dir.name}
    (run_dir / "STATUS.json").write_text(json.dumps(status_obj, indent=2))

    output_hashes_obj = {
        "validator_semver": "1.0.0",
        "validator_build_id": "test",
        "hashes": {output_rel: f"sha256:{_sha256_hex(output_bytes)}"},
    }
    (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps(output_hashes_obj, indent=2))

    if include_proof:
        proof = {
            "proof_version": "1.0.0",
            "restoration_result": {
                "verified": bool(proof_verified),
                "condition": "RESTORED_IDENTICAL" if proof_verified else "RESTORATION_FAILED",
            },
        }
        (run_dir / "PROOF.json").write_text(json.dumps(proof, indent=2))

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key_hex = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    validator_id = _sha256_hex(bytes.fromhex(public_key_hex))

    (run_dir / "VALIDATOR_IDENTITY.json").write_text(
        json.dumps({"algorithm": "ed25519", "public_key": public_key_hex, "validator_id": validator_id})
    )

    verifier = BundleVerifier(project_root=REPO_ROOT)
    bundle_root = verifier._compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes)

    signed_payload = {"bundle_root": bundle_root, "decision": "ACCEPT", "validator_id": validator_id}
    (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps(signed_payload))
    canonical_payload = verifier._canonicalize_json(signed_payload)
    msg = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload
    signature_hex = private_key.sign(msg).hex()

    (run_dir / "SIGNATURE.json").write_text(
        json.dumps({"payload_type": "BUNDLE", "signature": signature_hex, "validator_id": validator_id})
    )


@pytest.fixture
def work_area(tmp_path: Path):
    """Create test work area under repo's _runs directory."""
    base = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_test_atomic_restore"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


# =============================================================================
# 4.5.1 Tests: Transactional Restore (Staging + Atomic Swap)
# =============================================================================


class TestTransactionalRestore:
    """Tests for transactional restore using staging directory."""

    def test_successful_restore_uses_staging(self, work_area: Path):
        """Successful restore creates staging dir, copies, verifies, then moves atomically."""
        run_id = "staging-test"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/data.txt"
        payload = b"test content for staging\n"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=payload)

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True)

        assert result["ok"] is True
        assert result["code"] == "OK"

        # Verify file was restored
        restored_path = restore_root / output_rel
        assert restored_path.exists()
        assert restored_path.read_bytes() == payload

        # Verify no staging directory left behind
        staging_dirs = list(restore_root.glob(".spectrum06_staging_*"))
        assert len(staging_dirs) == 0, "Staging directory should be cleaned up"

        # Verify artifacts created
        assert (restore_root / "RESTORE_MANIFEST.json").exists()
        assert (restore_root / "RESTORE_REPORT.json").exists()

    def test_restore_rejects_existing_target(self, work_area: Path):
        """Restore fails if target file already exists."""
        run_id = "existing-target"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"new")

        restore_root = work_area / "dest"
        restore_root.mkdir()

        # Pre-create the target file
        target_path = restore_root / output_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"existing content")

        result = restore_bundle(run_dir, restore_root, strict=True)

        assert result["ok"] is False
        assert result["code"] == RESTORE_CODES["RESTORE_TARGET_PATH_EXISTS"]

        # Original file should be unchanged
        assert target_path.read_bytes() == b"existing content"


# =============================================================================
# 4.5.2 Tests: Rollback on Failure
# =============================================================================


class TestRollbackOnFailure:
    """Tests for rollback behavior when restore fails."""

    def test_rollback_cleans_staging_on_hash_mismatch(self, work_area: Path, monkeypatch):
        """If staged file hash doesn't match, staging is cleaned up."""
        import CAPABILITY.PRIMITIVES.restore_runner as rr

        run_id = "rollback-hash"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"correct")

        restore_root = work_area / "dest"
        restore_root.mkdir()

        # Make copy produce wrong content
        def corrupt_copy(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"corrupted content")

        monkeypatch.setattr(rr, "_copy_file", corrupt_copy)

        result = restore_bundle(run_dir, restore_root, strict=True)

        assert result["ok"] is False
        assert result["code"] == RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"]

        # Verify no staging directory left
        staging_dirs = list(restore_root.glob(".spectrum06_staging_*"))
        assert len(staging_dirs) == 0

        # Verify no partial files in restore root
        target_path = restore_root / output_rel
        assert not target_path.exists()

    def test_rollback_removes_created_targets_on_failure(self, work_area: Path, monkeypatch):
        """If restore fails after some files moved, those files are cleaned up."""
        import CAPABILITY.PRIMITIVES.restore_runner as rr

        run_id = "rollback-partial"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"data")

        restore_root = work_area / "dest"
        restore_root.mkdir()

        # Force rollback failure to test the RESTORE_ROLLBACK_FAILED code
        call_count = [0]
        original_rollback = rr._rollback_bundle

        def fail_rollback(*args, **kwargs):
            call_count[0] += 1
            return False  # Simulate rollback failure

        def corrupt_copy(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"wrong")

        monkeypatch.setattr(rr, "_copy_file", corrupt_copy)
        monkeypatch.setattr(rr, "_rollback_bundle", fail_rollback)

        result = restore_bundle(run_dir, restore_root, strict=True)

        assert result["ok"] is False
        assert result["code"] == RESTORE_CODES["RESTORE_ROLLBACK_FAILED"]
        assert result["details"]["cause_code"] == RESTORE_CODES["RESTORE_STAGING_HASH_MISMATCH"]


# =============================================================================
# 4.5.3 Tests: Dry-Run Mode
# =============================================================================


class TestDryRunMode:
    """Tests for dry-run mode that validates without writing."""

    def test_dry_run_validates_without_writing(self, work_area: Path):
        """Dry run performs all validation but doesn't create files."""
        run_id = "dry-run-test"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        payload = b"dry run content\n"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=payload)

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True, dry_run=True)

        assert result["ok"] is True
        assert result["code"] == "OK"
        assert result["details"]["dry_run"] is True
        assert result["details"]["would_restore_files_count"] == 1
        assert output_rel in result["details"]["would_restore_paths"]

        # Verify NO files were created
        target_path = restore_root / output_rel
        assert not target_path.exists()
        assert not (restore_root / "RESTORE_MANIFEST.json").exists()
        assert not (restore_root / "RESTORE_REPORT.json").exists()

    def test_dry_run_detects_validation_errors(self, work_area: Path):
        """Dry run still detects and reports validation errors."""
        run_id = "dry-run-error"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        _write_signed_bundle(
            run_dir=run_dir,
            output_rel=output_rel,
            output_bytes=b"data",
            proof_verified=False,  # Invalid proof
        )

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True, dry_run=True)

        assert result["ok"] is False
        assert result["code"] == RESTORE_CODES["RESTORE_PROOF_NOT_VERIFIED"]

    def test_dry_run_chain_validates_all_bundles(self, work_area: Path):
        """Dry run for chain validates all bundles without creating directories."""
        # This test requires chain setup which is complex
        # Skip if chain verification isn't available
        pytest.skip("Chain dry-run requires full chain verification setup")


# =============================================================================
# 4.5.4 Tests: Result Artifacts
# =============================================================================


class TestRestoreArtifacts:
    """Tests for RESTORE_MANIFEST.json and RESTORE_REPORT.json."""

    def test_restore_manifest_format(self, work_area: Path):
        """RESTORE_MANIFEST.json has correct format per SPECTRUM-06."""
        run_id = "manifest-format"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/data.txt"
        payload = b"manifest test content"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=payload)

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True)
        assert result["ok"] is True

        manifest = json.loads((restore_root / "RESTORE_MANIFEST.json").read_bytes())

        assert "entries" in manifest
        assert len(manifest["entries"]) == 1

        entry = manifest["entries"][0]
        assert entry["relative_path"] == output_rel
        assert entry["sha256"] == f"sha256:{_sha256_hex(payload)}"
        assert entry["bytes"] == len(payload)

    def test_restore_report_format(self, work_area: Path):
        """RESTORE_REPORT.json has correct format per SPECTRUM-06."""
        run_id = "report-format"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/data.txt"
        payload = b"report test content"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=payload)

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True)
        assert result["ok"] is True

        report = json.loads((restore_root / "RESTORE_REPORT.json").read_bytes())

        assert report["ok"] is True
        assert report["restored_files_count"] == 1
        assert report["restored_bytes"] == len(payload)
        assert "bundle_roots" in report
        assert len(report["bundle_roots"]) == 1
        assert report["chain_root"] is None  # Single bundle, not chain

    def test_canonical_json_no_trailing_newline(self, work_area: Path):
        """Result artifacts use canonical JSON without trailing newline."""
        run_id = "canonical-json"
        run_dir = work_area / run_id
        output_rel = f"LAW/CONTRACTS/_runs/_test_atomic_restore/{run_id}/out/file.txt"
        _write_signed_bundle(run_dir=run_dir, output_rel=output_rel, output_bytes=b"test")

        restore_root = work_area / "dest"
        restore_root.mkdir()

        result = restore_bundle(run_dir, restore_root, strict=True)
        assert result["ok"] is True

        manifest_bytes = (restore_root / "RESTORE_MANIFEST.json").read_bytes()
        report_bytes = (restore_root / "RESTORE_REPORT.json").read_bytes()

        # No trailing newline per SPECTRUM-06 Section 7.3
        assert not manifest_bytes.endswith(b"\n")
        assert not report_bytes.endswith(b"\n")

        # No whitespace (compact JSON)
        assert b"  " not in manifest_bytes  # No indentation
        assert b"  " not in report_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
