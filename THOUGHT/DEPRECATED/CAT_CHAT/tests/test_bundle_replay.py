#!/usr/bin/env python3
"""
Bundle Replay Tests (Phase G - Bundle Replay & Verification)

Tests for:
- G.1: Bundle runner takes only bundle.json + artifacts (offline mode)
- G.2: Verify-before-run with hard fail on mismatch
- G.3: Reproducibility (run twice -> identical receipts)
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path

import pytest

from catalytic_chat.bundle_runner import (
    BundleRunner,
    BundleRunnerError,
    replay_bundle,
    verify_bundle_integrity
)
from catalytic_chat.receipt import compute_receipt_hash, canonical_json_bytes


def create_minimal_bundle(bundle_dir: Path) -> Path:
    """Create a minimal valid bundle for testing.

    Args:
        bundle_dir: Directory to create bundle in

    Returns:
        Path to bundle directory
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Create artifact
    artifact_id = "test_artifact_001"
    artifact_content = "Test content for artifact\n"
    artifact_path = artifacts_dir / f"{artifact_id}.txt"
    artifact_path.write_text(artifact_content)

    content_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()

    # Create step
    step_id = "step_001"
    steps = [{
        "step_id": step_id,
        "ordinal": 1,
        "op": "READ_SECTION",
        "refs": {"section_id": "test_section"},
        "constraints": {"slice": None},
        "expected_outputs": {}
    }]

    # Create artifact manifest
    artifacts = [{
        "artifact_id": artifact_id,
        "kind": "SECTION_SLICE",
        "ref": "test_section",
        "slice": None,
        "path": f"artifacts/{artifact_id}.txt",
        "sha256": content_hash,
        "bytes": len(artifact_content.encode('utf-8'))
    }]

    # Compute plan hash
    plan_hash = hashlib.sha256(
        json.dumps({
            "run_id": "test_run",
            "steps": steps
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()

    # Build manifest (without bundle_id and root_hash first)
    manifest = {
        "bundle_version": "5.0.0",
        "bundle_id": "",
        "run_id": "test_run",
        "job_id": "test_job",
        "message_id": "test_msg",
        "plan_hash": plan_hash,
        "steps": steps,
        "inputs": {"symbols": [], "files": [], "slices": []},
        "artifacts": artifacts,
        "hashes": {"root_hash": ""},
        "provenance": {}
    }

    # Compute bundle_id from pre-manifest
    pre_manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    bundle_id = hashlib.sha256(pre_manifest_json.encode('utf-8')).hexdigest()

    # Compute root_hash
    hash_strings = [f"{artifacts[0]['artifact_id']}:{content_hash}"]
    root_hash = hashlib.sha256(
        ("\n".join(hash_strings) + "\n").encode('utf-8')
    ).hexdigest()

    # Update manifest
    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    # Write bundle.json
    bundle_json = bundle_dir / "bundle.json"
    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return bundle_dir


def create_multi_step_bundle(bundle_dir: Path) -> Path:
    """Create a bundle with multiple steps and artifacts.

    Args:
        bundle_dir: Directory to create bundle in

    Returns:
        Path to bundle directory
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Create multiple artifacts
    artifacts = []
    steps = []

    for i in range(3):
        artifact_id = f"artifact_{i:03d}"
        artifact_content = f"Content for artifact {i}\n"
        artifact_path = artifacts_dir / f"{artifact_id}.txt"
        artifact_path.write_text(artifact_content)

        content_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()

        artifacts.append({
            "artifact_id": artifact_id,
            "kind": "SECTION_SLICE",
            "ref": f"section_{i}",
            "slice": None,
            "path": f"artifacts/{artifact_id}.txt",
            "sha256": content_hash,
            "bytes": len(artifact_content.encode('utf-8'))
        })

        steps.append({
            "step_id": f"step_{i:03d}",
            "ordinal": i + 1,
            "op": "READ_SECTION",
            "refs": {"section_id": f"section_{i}"},
            "constraints": {"slice": None},
            "expected_outputs": {}
        })

    # Sort for determinism
    artifacts = sorted(artifacts, key=lambda x: x["artifact_id"])
    steps = sorted(steps, key=lambda x: (x["ordinal"], x["step_id"]))

    # Compute plan hash
    plan_hash = hashlib.sha256(
        json.dumps({
            "run_id": "test_run",
            "steps": steps
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()

    # Build manifest
    manifest = {
        "bundle_version": "5.0.0",
        "bundle_id": "",
        "run_id": "test_run",
        "job_id": "test_job",
        "message_id": "test_msg",
        "plan_hash": plan_hash,
        "steps": steps,
        "inputs": {"symbols": [], "files": [], "slices": []},
        "artifacts": artifacts,
        "hashes": {"root_hash": ""},
        "provenance": {}
    }

    pre_manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    bundle_id = hashlib.sha256(pre_manifest_json.encode('utf-8')).hexdigest()

    hash_strings = [f"{a['artifact_id']}:{a['sha256']}" for a in artifacts]
    root_hash = hashlib.sha256(
        ("\n".join(hash_strings) + "\n").encode('utf-8')
    ).hexdigest()

    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    bundle_json = bundle_dir / "bundle.json"
    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return bundle_dir


# =============================================================================
# G.1 Tests: Bundle Runner Takes Only Bundle + Artifacts (Offline Mode)
# =============================================================================

class TestG1OfflineMode:
    """G.1: Bundle runner takes only bundle.json + artifacts."""

    def test_runner_uses_only_bundle_artifacts(self):
        """Runner executes without any external file access."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Run in isolated temp directory - no repo access
            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            assert receipt["outcome"] == "SUCCESS"
            assert receipt["bundle_id"] is not None
            assert receipt["receipt_hash"] is not None
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_offline_mode(self):
        """Runner works with no database connection."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # BundleRunner has no repo_root or database parameters
            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            assert receipt["outcome"] == "SUCCESS"
            assert "steps" in receipt
            assert len(receipt["steps"]) == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_resolves_inputs_from_artifacts(self):
        """All step inputs come from artifacts/ directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_multi_step_bundle(Path(tmpdir))

            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            assert receipt["outcome"] == "SUCCESS"
            assert len(receipt["steps"]) == 3

            # Each step should have a result with content_hash
            for step in receipt["steps"]:
                assert step["outcome"] == "SUCCESS"
                assert step["result"] is not None
                assert "content_hash" in step["result"]
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_with_bundle_json_path(self):
        """Runner accepts explicit bundle.json path."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))
            bundle_json = bundle_dir / "bundle.json"

            runner = BundleRunner(bundle_json)
            receipt = runner.run()

            assert receipt["outcome"] == "SUCCESS"
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_writes_receipt_to_default_path(self):
        """Receipt is written to bundle_dir/receipt.json by default."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            receipt_path = bundle_dir / "receipt.json"
            assert receipt_path.exists()

            # Verify receipt file content
            receipt_content = json.loads(receipt_path.read_text().rstrip('\n'))
            assert receipt_content["bundle_id"] == receipt["bundle_id"]
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_writes_receipt_to_custom_path(self):
        """Receipt is written to custom path when specified."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))
            custom_receipt = Path(tmpdir) / "custom_receipt.json"

            runner = BundleRunner(bundle_dir, receipt_out=custom_receipt)
            receipt = runner.run()

            assert custom_receipt.exists()
            receipt_content = json.loads(custom_receipt.read_text().rstrip('\n'))
            assert receipt_content["receipt_hash"] == receipt["receipt_hash"]
        finally:
            shutil.rmtree(tmpdir)


# =============================================================================
# G.2 Tests: Verify-Before-Run Hard Fail
# =============================================================================

class TestG2VerifyBeforeRun:
    """G.2: Verify-before-run with hard fail on mismatch."""

    def test_runner_fails_on_artifact_hash_mismatch(self):
        """Runner hard fails if artifact content doesn't match sha256."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper with artifact content
            artifact_path = bundle_dir / "artifacts" / "test_artifact_001.txt"
            artifact_path.write_text("Tampered content!\n")

            runner = BundleRunner(bundle_dir)
            with pytest.raises(BundleRunnerError, match="verification failed"):
                runner.run()

            # No receipt should be written
            receipt_path = bundle_dir / "receipt.json"
            assert not receipt_path.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_fails_on_root_hash_mismatch(self):
        """Runner hard fails if root_hash is wrong."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper with root_hash in manifest
            bundle_json = bundle_dir / "bundle.json"
            manifest = json.loads(bundle_json.read_text())
            manifest["hashes"]["root_hash"] = "0" * 64  # Invalid hash
            bundle_json.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            runner = BundleRunner(bundle_dir)
            with pytest.raises(BundleRunnerError, match="verification failed"):
                runner.run()
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_fails_on_bundle_id_mismatch(self):
        """Runner hard fails if bundle_id is wrong."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper with bundle_id
            bundle_json = bundle_dir / "bundle.json"
            manifest = json.loads(bundle_json.read_text())
            manifest["bundle_id"] = "corrupted_" + manifest["bundle_id"][:50]
            bundle_json.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            runner = BundleRunner(bundle_dir)
            with pytest.raises(BundleRunnerError, match="verification failed"):
                runner.run()
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_fails_on_missing_artifact(self):
        """Runner hard fails if artifact file is missing."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Delete artifact file
            artifact_path = bundle_dir / "artifacts" / "test_artifact_001.txt"
            artifact_path.unlink()

            runner = BundleRunner(bundle_dir)
            with pytest.raises(BundleRunnerError, match="verification failed"):
                runner.run()
        finally:
            shutil.rmtree(tmpdir)

    def test_verify_before_run_order(self):
        """Verify happens before any execution."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper with artifact to trigger verification failure
            artifact_path = bundle_dir / "artifacts" / "test_artifact_001.txt"
            artifact_path.write_text("Tampered!\n")

            # Create a marker to track if execution happened
            marker_path = bundle_dir / "execution_marker.txt"

            runner = BundleRunner(bundle_dir)

            # Verification should fail before any execution
            with pytest.raises(BundleRunnerError):
                runner.run()

            # No receipt means no execution happened
            receipt_path = bundle_dir / "receipt.json"
            assert not receipt_path.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_no_partial_execution_on_failure(self):
        """No receipt is written on verification failure."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper with bundle
            artifact_path = bundle_dir / "artifacts" / "test_artifact_001.txt"
            artifact_path.write_text("Tampered content\n")

            runner = BundleRunner(bundle_dir)

            try:
                runner.run()
            except BundleRunnerError:
                pass

            # No receipt should exist
            receipt_path = bundle_dir / "receipt.json"
            assert not receipt_path.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_verify_only_without_execution(self):
        """verify_bundle_integrity() checks without executing."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            result = verify_bundle_integrity(bundle_dir)

            assert result["status"] == "verified"
            assert "bundle_id" in result
            assert "root_hash" in result

            # No receipt should be written
            receipt_path = bundle_dir / "receipt.json"
            assert not receipt_path.exists()
        finally:
            shutil.rmtree(tmpdir)

    def test_verify_only_fails_on_tampered_bundle(self):
        """verify_bundle_integrity() fails on tampered bundle."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Tamper
            artifact_path = bundle_dir / "artifacts" / "test_artifact_001.txt"
            artifact_path.write_text("Tampered!\n")

            with pytest.raises(BundleRunnerError, match="verification failed"):
                verify_bundle_integrity(bundle_dir)
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_fails_on_missing_bundle_json(self):
        """Runner fails if bundle.json doesn't exist."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = Path(tmpdir) / "nonexistent"
            bundle_dir.mkdir()

            with pytest.raises(BundleRunnerError, match="bundle.json not found"):
                BundleRunner(bundle_dir)
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_fails_on_invalid_path(self):
        """Runner fails on completely invalid path."""
        with pytest.raises(BundleRunnerError, match="not found"):
            BundleRunner(Path("/nonexistent/path/to/bundle"))


# =============================================================================
# G.3 Tests: Reproducibility (Run Twice -> Identical)
# =============================================================================

class TestG3Reproducibility:
    """G.3: Run twice produces identical receipts."""

    def test_run_twice_identical_receipt_hash(self):
        """Same bundle produces identical receipt_hash on repeat runs."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Run 1
            receipt1_path = Path(tmpdir) / "receipt1.json"
            runner1 = BundleRunner(bundle_dir, receipt_out=receipt1_path)
            receipt1 = runner1.run()

            # Run 2
            receipt2_path = Path(tmpdir) / "receipt2.json"
            runner2 = BundleRunner(bundle_dir, receipt_out=receipt2_path)
            receipt2 = runner2.run()

            assert receipt1["receipt_hash"] == receipt2["receipt_hash"]
        finally:
            shutil.rmtree(tmpdir)

    def test_run_twice_identical_receipt_bytes(self):
        """Same bundle produces byte-identical receipt files."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Run 1
            receipt1_path = Path(tmpdir) / "receipt1.json"
            runner1 = BundleRunner(bundle_dir, receipt_out=receipt1_path)
            runner1.run()

            # Run 2
            receipt2_path = Path(tmpdir) / "receipt2.json"
            runner2 = BundleRunner(bundle_dir, receipt_out=receipt2_path)
            runner2.run()

            # Compare bytes
            receipt1_bytes = receipt1_path.read_bytes()
            receipt2_bytes = receipt2_path.read_bytes()

            assert receipt1_bytes == receipt2_bytes
        finally:
            shutil.rmtree(tmpdir)

    def test_run_twice_identical_merkle_root(self):
        """Same bundle chain produces identical merkle root."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            from catalytic_chat.receipt import compute_merkle_root

            # Run 1
            receipt1_path = Path(tmpdir) / "receipt1.json"
            runner1 = BundleRunner(bundle_dir, receipt_out=receipt1_path)
            receipt1 = runner1.run()

            # Run 2
            receipt2_path = Path(tmpdir) / "receipt2.json"
            runner2 = BundleRunner(bundle_dir, receipt_out=receipt2_path)
            receipt2 = runner2.run()

            # Compute merkle roots
            merkle1 = compute_merkle_root([receipt1["receipt_hash"]])
            merkle2 = compute_merkle_root([receipt2["receipt_hash"]])

            assert merkle1 == merkle2
        finally:
            shutil.rmtree(tmpdir)

    def test_deterministic_across_directories(self):
        """Same bundle in different directories produces identical receipts."""
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        try:
            # Create identical bundles in different directories
            bundle_dir1 = create_minimal_bundle(Path(tmpdir1))
            bundle_dir2 = create_minimal_bundle(Path(tmpdir2))

            # Run in each
            receipt1_path = Path(tmpdir1) / "receipt.json"
            receipt2_path = Path(tmpdir2) / "receipt.json"

            runner1 = BundleRunner(bundle_dir1, receipt_out=receipt1_path)
            runner2 = BundleRunner(bundle_dir2, receipt_out=receipt2_path)

            receipt1 = runner1.run()
            receipt2 = runner2.run()

            # Receipt hashes should be identical (no path leakage)
            assert receipt1["receipt_hash"] == receipt2["receipt_hash"]

            # File bytes should be identical
            assert receipt1_path.read_bytes() == receipt2_path.read_bytes()
        finally:
            shutil.rmtree(tmpdir1)
            shutil.rmtree(tmpdir2)

    def test_deterministic_step_ordering(self):
        """Step results maintain deterministic ordering."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_multi_step_bundle(Path(tmpdir))

            # Run twice
            receipt1_path = Path(tmpdir) / "receipt1.json"
            receipt2_path = Path(tmpdir) / "receipt2.json"

            runner1 = BundleRunner(bundle_dir, receipt_out=receipt1_path)
            runner2 = BundleRunner(bundle_dir, receipt_out=receipt2_path)

            receipt1 = runner1.run()
            receipt2 = runner2.run()

            # Steps should be in identical order
            assert len(receipt1["steps"]) == len(receipt2["steps"])

            for i, (s1, s2) in enumerate(zip(receipt1["steps"], receipt2["steps"])):
                assert s1["step_id"] == s2["step_id"], f"Step {i} ID mismatch"
                assert s1["ordinal"] == s2["ordinal"], f"Step {i} ordinal mismatch"
                assert s1["result"] == s2["result"], f"Step {i} result mismatch"
        finally:
            shutil.rmtree(tmpdir)

    def test_replay_bundle_convenience_function(self):
        """replay_bundle() convenience function produces deterministic output."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            receipt1_path = Path(tmpdir) / "receipt1.json"
            receipt2_path = Path(tmpdir) / "receipt2.json"

            receipt1 = replay_bundle(bundle_dir, receipt_out=receipt1_path)
            receipt2 = replay_bundle(bundle_dir, receipt_out=receipt2_path)

            assert receipt1["receipt_hash"] == receipt2["receipt_hash"]
        finally:
            shutil.rmtree(tmpdir)

    def test_receipt_hash_computation_matches_stored(self):
        """Computed receipt_hash matches stored value."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            # Verify stored hash matches computation
            computed = compute_receipt_hash(receipt)
            assert receipt["receipt_hash"] == computed
        finally:
            shutil.rmtree(tmpdir)

    def test_chained_receipts_produce_same_chain(self):
        """Chained receipts produce identical chains on repeat."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            from catalytic_chat.receipt import compute_merkle_root

            # Chain 1: Run 1 -> Run 2
            receipt1a_path = Path(tmpdir) / "receipt1a.json"
            runner1a = BundleRunner(bundle_dir, receipt_out=receipt1a_path)
            receipt1a = runner1a.run()

            receipt1b_path = Path(tmpdir) / "receipt1b.json"
            runner1b = BundleRunner(
                bundle_dir,
                receipt_out=receipt1b_path,
                previous_receipt_hash=receipt1a["receipt_hash"]
            )
            receipt1b = runner1b.run()

            # Chain 2: Same sequence
            receipt2a_path = Path(tmpdir) / "receipt2a.json"
            runner2a = BundleRunner(bundle_dir, receipt_out=receipt2a_path)
            receipt2a = runner2a.run()

            receipt2b_path = Path(tmpdir) / "receipt2b.json"
            runner2b = BundleRunner(
                bundle_dir,
                receipt_out=receipt2b_path,
                previous_receipt_hash=receipt2a["receipt_hash"]
            )
            receipt2b = runner2b.run()

            # Merkle roots should match
            merkle1 = compute_merkle_root([
                receipt1a["receipt_hash"],
                receipt1b["receipt_hash"]
            ])
            merkle2 = compute_merkle_root([
                receipt2a["receipt_hash"],
                receipt2b["receipt_hash"]
            ])

            assert merkle1 == merkle2
        finally:
            shutil.rmtree(tmpdir)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_runner_with_signing_key(self):
        """Runner can sign receipts with Ed25519 key."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Generate a test signing key (32 bytes)
            signing_key = bytes.fromhex("a" * 64)

            runner = BundleRunner(bundle_dir, signing_key=signing_key)
            receipt = runner.run()

            assert receipt["attestation"] is not None
            assert receipt["attestation"]["scheme"] == "ed25519"
            assert "signature" in receipt["attestation"]
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_rejects_invalid_signing_key(self):
        """Runner rejects signing key of wrong size."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            # Invalid key size (16 bytes instead of 32)
            invalid_key = bytes.fromhex("a" * 32)

            with pytest.raises(BundleRunnerError, match="32 or 64 bytes"):
                BundleRunner(bundle_dir, signing_key=invalid_key)
        finally:
            shutil.rmtree(tmpdir)

    def test_runner_with_previous_receipt_hash(self):
        """Runner includes previous_receipt_hash in chained receipt."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = create_minimal_bundle(Path(tmpdir))

            prev_hash = "abc123" + "0" * 58

            runner = BundleRunner(
                bundle_dir,
                previous_receipt_hash=prev_hash
            )
            receipt = runner.run()

            assert receipt["parent_receipt_hash"] == prev_hash
        finally:
            shutil.rmtree(tmpdir)

    def test_empty_artifacts_bundle(self):
        """Bundle with no artifacts still works."""
        tmpdir = tempfile.mkdtemp()
        try:
            bundle_dir = Path(tmpdir)
            bundle_dir.mkdir(exist_ok=True)

            artifacts_dir = bundle_dir / "artifacts"
            artifacts_dir.mkdir()

            # Bundle with no steps and no artifacts
            manifest = {
                "bundle_version": "5.0.0",
                "bundle_id": "",
                "run_id": "test_run",
                "job_id": "test_job",
                "message_id": "test_msg",
                "plan_hash": hashlib.sha256(
                    json.dumps({"run_id": "test_run", "steps": []}, sort_keys=True).encode()
                ).hexdigest(),
                "steps": [],
                "inputs": {"symbols": [], "files": [], "slices": []},
                "artifacts": [],
                "hashes": {"root_hash": ""},
                "provenance": {}
            }

            pre_manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            bundle_id = hashlib.sha256(pre_manifest_json.encode('utf-8')).hexdigest()
            root_hash = hashlib.sha256("\n".encode('utf-8')).hexdigest()

            manifest["bundle_id"] = bundle_id
            manifest["hashes"]["root_hash"] = root_hash

            bundle_json = bundle_dir / "bundle.json"
            bundle_json.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            runner = BundleRunner(bundle_dir)
            receipt = runner.run()

            assert receipt["outcome"] == "SUCCESS"
            assert len(receipt["steps"]) == 0
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
