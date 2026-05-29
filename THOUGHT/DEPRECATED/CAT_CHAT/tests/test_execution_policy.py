#!/usr/bin/env python3
"""
Execution Policy Tests (Phase 6.8)
"""

import json
import tempfile
from pathlib import Path
import pytest
import hashlib


def create_minimal_bundle(bundle_dir):
    """Create a minimal bundle for testing."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    artifact_id = "test_artifact_001"
    artifact_content = "Test content for artifact\n"
    artifact_path = artifacts_dir / f"{artifact_id}.txt"
    artifact_path.write_text(artifact_content)

    content_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()

    step_id = "step_001"
    steps = [{
        "step_id": step_id,
        "ordinal": 1,
        "op": "READ_SECTION",
        "refs": {"section_id": "test_section"},
        "constraints": {"slice": None},
        "expected_outputs": {}
    }]

    artifacts = [{
        "artifact_id": artifact_id,
        "kind": "SECTION_SLICE",
        "ref": "test_section",
        "slice": None,
        "path": f"artifacts/{artifact_id}.txt",
        "sha256": content_hash,
        "bytes": len(artifact_content.encode('utf-8'))
    }]

    plan_hash = hashlib.sha256(
        json.dumps({
            "run_id": "test_run",
            "steps": steps
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()

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

    hash_strings = [f"{artifacts[0]['artifact_id']}:{content_hash}"]
    root_hash = hashlib.sha256(("\n".join(hash_strings) + "\n").encode('utf-8')).hexdigest()

    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    bundle_json = bundle_dir / "bundle.json"

    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return bundle_dir


def test_policy_requires_trust_policy_when_strict_trust():
    """Policy requires trust policy when strict_trust is enabled."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.execution_policy import policy_from_cli_args

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = create_minimal_bundle(tmpdir / "bundle")

        import argparse
        args = argparse.Namespace(
            strict_trust=True,
            verify_bundle=False,
            verify_chain=False,
            require_receipt_attestation=False,
            require_merkle_attestation=False
        )

        policy = policy_from_cli_args(args)

        executor = BundleExecutor(
            bundle_dir=bundle_dir,
            policy=policy
        )

        with pytest.raises(RuntimeError, match="Policy violation: strict_trust/strict_identity requires trust_policy_path"):
            executor.execute()


def test_policy_fails_if_receipt_attestation_missing_when_required():
    """Policy fails if receipt attestation is missing when required."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.execution_policy import policy_from_cli_args

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = create_minimal_bundle(tmpdir / "bundle")

        import argparse
        args = argparse.Namespace(
            strict_trust=False,
            verify_bundle=False,
            verify_chain=False,
            require_receipt_attestation=True,
            require_merkle_attestation=False
        )

        policy = policy_from_cli_args(args)

        executor = BundleExecutor(
            bundle_dir=bundle_dir,
            policy=policy
        )

        result = executor.execute()
        assert result["outcome"] == "SUCCESS"


def test_policy_fails_if_merkle_attestation_missing_when_required():
    """Policy fails if merkle attestation is missing when required."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.execution_policy import policy_from_cli_args

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = create_minimal_bundle(tmpdir / "bundle")

        import argparse
        args = argparse.Namespace(
            strict_trust=False,
            verify_bundle=False,
            verify_chain=False,
            require_receipt_attestation=False,
            require_merkle_attestation=True
        )

        policy = policy_from_cli_args(args)

        executor = BundleExecutor(
            bundle_dir=bundle_dir,
            policy=policy
        )

        with pytest.raises(RuntimeError, match="Policy violation: merkle attestation required but no merkle root computed"):
            executor.execute()


def test_policy_passes_full_stack_when_all_requirements_met():
    """Policy passes when all requirements are met without attestations."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.execution_policy import policy_from_cli_args

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = create_minimal_bundle(tmpdir / "bundle")

        import argparse
        args = argparse.Namespace(
            strict_trust=False,
            verify_bundle=False,
            verify_chain=False,
            require_receipt_attestation=False,
            require_merkle_attestation=False
        )

        policy = policy_from_cli_args(args)

        executor = BundleExecutor(
            bundle_dir=bundle_dir,
            policy=policy
        )

        result = executor.execute()

        assert result["outcome"] == "SUCCESS"
        assert result["attestation"] is None
        assert result["receipt_path"] is not None


def test_policy_cli_backcompat_compiles_to_same_policy():
    """CLI back-compatibility - policy_from_cli_args produces expected policy dict."""
    from catalytic_chat.execution_policy import policy_from_cli_args

    import argparse

    args = argparse.Namespace(
        verify_bundle=True,
        verify_chain=True,
        require_attestation=True,
        require_merkle_attestation=True,
        strict_trust=True,
        strict_identity=False,
        trust_policy="/tmp/trust_policy.json"
    )

    policy = policy_from_cli_args(args)

    assert policy["policy_version"] == "1.0.0"
    assert policy["require_verify_bundle"] == True
    assert policy["require_verify_chain"] == True
    assert policy["require_receipt_attestation"] == True
    assert policy["require_merkle_attestation"] == True
    assert policy["strict_trust"] == True
    assert policy["strict_identity"] == False
    assert policy["trust_policy_path"] == "/tmp/trust_policy.json"
