#!/usr/bin/env python3
"""
ARCHIVED: Deprecated Bundle Execution Tests

These tests were skipped placeholders for features that got implemented
in BundleRunner (Phase G) instead of BundleExecutor.

- Verify-before-run: See test_bundle_replay.py::TestG2VerifyBeforeRun
- Unknown step handling: See test_bundle_replay.py::TestEdgeCases

Archived: 2026-01-19
Reason: Functionality moved to BundleRunner class
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path

import pytest

from catalytic_chat.bundle import BundleError
from catalytic_chat.executor import BundleExecutor


def create_minimal_bundle(bundle_dir):
    """Create a minimal bundle for testing executor."""
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


@pytest.mark.skip(reason="ARCHIVED: Implemented in BundleRunner, see test_bundle_replay.py")
def test_bundle_run_fails_without_verify():
    """Corrupt bundle.json, assert run fails before execution.

    ARCHIVED: This feature is now in BundleRunner.verify() and tested in
    test_bundle_replay.py::TestG2VerifyBeforeRun
    """
    tmpdir = tempfile.mkdtemp()
    create_minimal_bundle(tmpdir)

    bundle_json_path = Path(tmpdir) / "bundle.json"
    with open(bundle_json_path, 'r') as f:
        bundle_data = json.load(f)

    bundle_data["bundle_id"] = "corrupted_id"

    with open(bundle_json_path, 'w') as f:
        json.dump(bundle_data, f)

    executor = BundleExecutor(Path(tmpdir))
    with pytest.raises(BundleError):
        executor.execute()

    shutil.rmtree(tmpdir)


@pytest.mark.skip(reason="ARCHIVED: Implemented in BundleRunner, see test_bundle_replay.py")
def test_bundle_run_fails_on_unsupported_step():
    """Create bundle with fake step kind, assert fail-closed.

    ARCHIVED: BundleRunner handles unknown ops gracefully (passthrough).
    A verified bundle is trusted, so unknown ops don't hard-fail.
    See test_bundle_replay.py::TestEdgeCases
    """
    tmpdir = tempfile.mkdtemp()

    bundle_dir = Path(tmpdir)
    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    step_id = "step_001"
    steps = [{
        "step_id": step_id,
        "ordinal": 1,
        "op": "FAKE_OPERATION",
        "refs": {},
        "constraints": {},
        "expected_outputs": {}
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
    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    executor = BundleExecutor(Path(tmpdir))
    with pytest.raises(BundleError, match="Unsupported step kind"):
        executor.execute()

    shutil.rmtree(tmpdir)
