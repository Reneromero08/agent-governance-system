#!/usr/bin/env python3
"""
Unit tests for verify_bundle.py primitive.

Tests single bundle verification covering:
- Valid bundles (PASS)
- Missing artifacts (BUNDLE_INCOMPLETE)
- Hash mismatches (HASH_MISMATCH)
- Missing outputs (OUTPUT_MISSING)
- Status failures (STATUS_NOT_SUCCESS, CMP01_NOT_PASS)
- Proof failures (PROOF_REQUIRED, RESTORATION_FAILED)
- Forbidden artifacts (FORBIDDEN_ARTIFACT)

Run: pytest CATALYTIC-DPT/TESTBENCH/test_verify_bundle.py -v
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path
import pytest

# Add CATALYTIC-DPT to path
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))

from PRIMITIVES.verify_bundle import BundleVerifier, VALIDATOR_SEMVER


# Get validator_build_id from LAB/MCP/server.py
import importlib.util
SERVER_PATH = repo_root_path / "CATALYTIC-DPT" / "LAB" / "MCP" / "server.py"
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server_module)
get_validator_build_id = mcp_server_module.get_validator_build_id


@pytest.fixture
def test_base():
    """Fixture providing temporary test directory."""
    base = repo_root_path / "CONTRACTS" / "_runs" / "_test_verify_bundle"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    # Cleanup
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def verifier():
    """Fixture providing BundleVerifier instance."""
    return BundleVerifier(project_root=repo_root_path)


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(content).hexdigest()


def create_minimal_bundle(
    test_base: Path,
    run_id: str,
    output_filename: str,
    output_content: bytes,
    include_proof: bool = True,
    proof_verified: bool = True,
    status: str = "success",
    cmp01: str = "pass",
    create_output: bool = True
) -> Path:
    """Create a minimal valid SPECTRUM-02 bundle for testing.

    Args:
        test_base: Base directory for test runs
        run_id: Run identifier
        output_filename: Name of output file
        output_content: Content for the output file
        include_proof: Whether to create PROOF.json
        proof_verified: Value for PROOF.json.restoration_result.verified
        status: Value for STATUS.status
        cmp01: Value for STATUS.cmp01
        create_output: Whether to create the actual output file

    Returns:
        Path to the run directory
    """
    run_dir = test_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory and file
    out_dir = run_dir / "out"
    out_dir.mkdir(exist_ok=True)

    if create_output:
        output_path = out_dir / output_filename
        output_path.write_bytes(output_content)

    # Compute relative path (repo-root-relative, POSIX style)
    output_rel = f"CONTRACTS/_runs/_test_verify_bundle/{run_id}/out/{output_filename}"

    # Create TASK_SPEC.json
    task_spec = {
        "task_id": run_id,
        "outputs": {
            "durable_paths": [output_rel]
        }
    }
    task_spec_bytes = json.dumps(task_spec, indent=2).encode("utf-8")
    (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

    # Create STATUS.json
    status_obj = {
        "status": status,
        "cmp01": cmp01,
        "run_id": run_id,
        "completed_at": "2024-12-24T12:00:00Z"
    }
    (run_dir / "STATUS.json").write_text(json.dumps(status_obj, indent=2))

    # Create OUTPUT_HASHES.json
    output_hash = f"sha256:{compute_sha256(output_content)}"
    output_hashes = {
        "validator_semver": VALIDATOR_SEMVER,
        "validator_build_id": get_validator_build_id(),
        "generated_at": "2024-12-24T12:00:00Z",
        "hashes": {
            output_rel: output_hash
        }
    }
    (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps(output_hashes, indent=2))

    # Create PROOF.json if requested
    if include_proof:
        proof = {
            "proof_version": "1.0.0",
            "run_id": run_id,
            "timestamp": "2024-12-24T12:00:00Z",
            "catalytic_domains": [],
            "pre_state": {
                "domain_root_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "file_manifest": {}
            },
            "post_state": {
                "domain_root_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "file_manifest": {}
            },
            "restoration_result": {
                "verified": proof_verified,
                "condition": "RESTORED_IDENTICAL" if proof_verified else "RESTORATION_FAILED_HASH_MISMATCH",
                "mismatches": [] if proof_verified else [{"type": "hash_mismatch"}]
            },
            "proof_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        }
        (run_dir / "PROOF.json").write_text(json.dumps(proof, indent=2))
    elif not include_proof:
        # For Phase 1 check in verify_bundle_spectrum05, we still need the file
        # unless we specifically want to test ARTIFACT_MISSING
        pass

    # Add required SPECTRUM-05 artifacts (dummy data sufficient for Phase 1/2)
    (run_dir / "VALIDATOR_IDENTITY.json").write_text(json.dumps({
        "algorithm": "ed25519",
        "public_key": "0" * 64,
        "validator_id": "0" * 64
    }))
    (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps({
        "bundle_root": "0" * 64,
        "decision": "ACCEPT",
        "validator_id": "0" * 64
    }))
    (run_dir / "SIGNATURE.json").write_text(json.dumps({
        "payload_type": "BUNDLE",
        "signature": "0" * 128,
        "validator_id": "0" * 64
    }))

    return run_dir


# =============================================================================
# TESTS: Single Bundle Verification
# =============================================================================

def test_valid_bundle(test_base, verifier):
    """Bundle verification should pass for valid bundle."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-valid",
        "result.txt",
        b"Valid output\n"
    )

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_missing_task_spec(test_base, verifier):
    """Bundle verification should fail when TASK_SPEC.json is missing."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-missing-task-spec",
        "result.txt",
        b"Output\n"
    )

    # Delete TASK_SPEC.json
    (run_dir / "TASK_SPEC.json").unlink()

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert result["errors"][0]["code"] == "ARTIFACT_MISSING"
    assert "TASK_SPEC.json" in result["errors"][0]["message"]


def test_missing_status(test_base, verifier):
    """Bundle verification should fail when STATUS.json is missing."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-missing-status",
        "result.txt",
        b"Output\n"
    )

    # Delete STATUS.json
    (run_dir / "STATUS.json").unlink()

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert result["errors"][0]["code"] == "ARTIFACT_MISSING"
    assert "STATUS.json" in result["errors"][0]["message"]


def test_missing_output_hashes(test_base, verifier):
    """Bundle verification should fail when OUTPUT_HASHES.json is missing."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-missing-output-hashes",
        "result.txt",
        b"Output\n"
    )

    # Delete OUTPUT_HASHES.json
    (run_dir / "OUTPUT_HASHES.json").unlink()

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert result["errors"][0]["code"] == "ARTIFACT_MISSING"
    assert "OUTPUT_HASHES.json" in result["errors"][0]["message"]


def test_status_not_success(test_base, verifier):
    """Bundle verification should fail when STATUS.status != 'success'."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-status-failed",
        "result.txt",
        b"Output\n",
        status="failed"
    )

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    # In SPECTRUM-05 status != success can be seen as DECISION_INVALID since it can't be ACCEPTED
    assert result["errors"][0]["code"] == "DECISION_INVALID"


def test_cmp01_not_pass(test_base, verifier):
    """Bundle verification should fail when STATUS.cmp01 != 'pass'."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-cmp01-fail",
        "result.txt",
        b"Output\n",
        cmp01="fail"
    )

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    # Similarly for cmp01
    assert result["errors"][0]["code"] == "DECISION_INVALID"


def test_output_missing(test_base, verifier):
    """Bundle verification should fail when declared output does not exist."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-output-missing",
        "result.txt",
        b"Output\n",
        create_output=False  # Don't create the output file
    )

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    # Should have OUTPUT_MISSING error
    output_missing_errors = [e for e in result["errors"] if e["code"] == "OUTPUT_MISSING"]
    assert len(output_missing_errors) == 1


def test_hash_mismatch(test_base, verifier):
    """Bundle verification should fail when output hash does not match."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-hash-mismatch",
        "result.txt",
        b"Original content\n"
    )

    # Tamper with output file
    output_path = run_dir / "out" / "result.txt"
    output_path.write_bytes(b"Tampered content\n")

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    # Should have HASH_MISMATCH error
    hash_errors = [e for e in result["errors"] if e["code"] == "HASH_MISMATCH"]
    assert len(hash_errors) == 1


def test_proof_required(test_base, verifier):
    """Bundle verification should fail when PROOF.json is missing."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-proof-required",
        "result.txt",
        b"Output\n",
        include_proof=False
    )

    result = verifier.verify_bundle(run_dir, check_proof=True)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert result["errors"][0]["code"] == "ARTIFACT_MISSING"


def test_proof_not_verified(test_base, verifier):
    """Bundle verification should fail when PROOF.json.verified != true."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-proof-not-verified",
        "result.txt",
        b"Output\n",
        proof_verified=False
    )

    result = verifier.verify_bundle(run_dir, check_proof=True)

    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert result["errors"][0]["code"] == "RESTORATION_FAILED"


def test_forbidden_artifact_logs(test_base, verifier):
    """Bundle verification should fail when logs/ directory exists."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-forbidden-logs",
        "result.txt",
        b"Output\n"
    )

    # Create forbidden logs/ directory
    (run_dir / "logs").mkdir()

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    # Should have FORBIDDEN_ARTIFACT error
    forbidden_errors = [e for e in result["errors"] if e["code"] == "FORBIDDEN_ARTIFACT"]
    assert len(forbidden_errors) == 1
    assert "logs/" in forbidden_errors[0]["message"]


def test_forbidden_artifact_tmp(test_base, verifier):
    """Bundle verification should fail when tmp/ directory exists."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-forbidden-tmp",
        "result.txt",
        b"Output\n"
    )

    # Create forbidden tmp/ directory
    (run_dir / "tmp").mkdir()

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    forbidden_errors = [e for e in result["errors"] if e["code"] == "FORBIDDEN_ARTIFACT"]
    assert len(forbidden_errors) == 1
    assert "tmp/" in forbidden_errors[0]["message"]


def test_forbidden_artifact_transcript(test_base, verifier):
    """Bundle verification should fail when transcript.json exists."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-forbidden-transcript",
        "result.txt",
        b"Output\n"
    )

    # Create forbidden transcript.json
    (run_dir / "transcript.json").write_text("{}")

    result = verifier.verify_bundle(run_dir)

    assert result["valid"] is False
    forbidden_errors = [e for e in result["errors"] if e["code"] == "FORBIDDEN_ARTIFACT"]
    assert len(forbidden_errors) == 1
    assert "transcript.json" in forbidden_errors[0]["message"]


def test_no_proof_check_skips_proof(test_base, verifier):
    """Bundle verification should pass without PROOF.json when check_proof=False."""
    run_dir = create_minimal_bundle(
        test_base,
        "test-no-proof-check",
        "result.txt",
        b"Output\n",
        include_proof=False
    )

    result = verifier.verify_bundle(run_dir, check_proof=False)

    # Should pass because we're not checking proof
    assert result["valid"] is True
    assert len(result["errors"]) == 0


# =============================================================================
# TESTS: Chain Verification
# =============================================================================

def test_chain_valid(test_base, verifier):
    """Chain verification should pass for valid chain."""
    run1 = create_minimal_bundle(test_base, "chain-1", "r1.txt", b"R1\n")
    run2 = create_minimal_bundle(test_base, "chain-2", "r2.txt", b"R2\n")
    run3 = create_minimal_bundle(test_base, "chain-3", "r3.txt", b"R3\n")

    result = verifier.verify_chain([run1, run2, run3])

    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_chain_rejects_middle_tamper(test_base, verifier):
    """Chain verification should fail when middle bundle has tampered output."""
    run1 = create_minimal_bundle(test_base, "chain-1", "r1.txt", b"R1\n")
    run2 = create_minimal_bundle(test_base, "chain-2", "r2.txt", b"R2\n")
    run3 = create_minimal_bundle(test_base, "chain-3", "r3.txt", b"R3\n")

    # Tamper with run2
    (run2 / "out" / "r2.txt").write_bytes(b"TAMPERED\n")

    result = verifier.verify_chain([run1, run2, run3])

    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    # Should have HASH_MISMATCH error with run_id
    hash_errors = [e for e in result["errors"] if e["code"] == "HASH_MISMATCH"]
    assert len(hash_errors) == 1
    assert hash_errors[0].get("run_id") == "chain-2"


def test_chain_rejects_missing_artifact(test_base, verifier):
    """Chain verification should fail when a bundle is missing OUTPUT_HASHES.json."""
    run1 = create_minimal_bundle(test_base, "chain-1", "r1.txt", b"R1\n")
    run2 = create_minimal_bundle(test_base, "chain-2", "r2.txt", b"R2\n")

    # Delete OUTPUT_HASHES.json from run1
    (run1 / "OUTPUT_HASHES.json").unlink()

    result = verifier.verify_chain([run1, run2])

    assert result["valid"] is False
    assert len(result["errors"]) >= 1
    # Should have BUNDLE_INCOMPLETE error
    incomplete_errors = [e for e in result["errors"] if e["code"] == "ARTIFACT_MISSING"]
    assert len(incomplete_errors) == 1
    assert incomplete_errors[0].get("run_id") == "chain-1"




if __name__ == "__main__":
    # Allow running as standalone script
    pytest.main([__file__, "-v"])
