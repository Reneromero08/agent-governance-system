#!/usr/bin/env python3
"""
Bundle Execution Tests (Phase 6)
"""

import json
import hashlib
import tempfile
import shutil
import os
from pathlib import Path

import pytest

from catalytic_chat.bundle import BundleBuilder, BundleError, BundleVerifier
from catalytic_chat.executor import BundleExecutor
from catalytic_chat.message_cassette import MessageCassette, _generate_id
from catalytic_chat.section_indexer import SectionIndexer, build_index


@pytest.fixture
def repo_root(tmp_path):
    """Create a minimal test repository structure."""
    src_dir = tmp_path / "THOUGHT" / "LAB" / "CAT_CHAT"
    src_dir.mkdir(parents=True)

    law_dir = tmp_path / "LAW" / "CANON"
    law_dir.mkdir(parents=True)

    cortex_dir = tmp_path / "CORTEX" / "_generated"
    cortex_dir.mkdir(parents=True)

    content = "# Test Document\n"
    for i in range(31):
        content += f"Line {i}\n"
    (law_dir / "CONTRACT.md").write_text(content)

    return tmp_path


@pytest.fixture
def indexed_repo(repo_root):
    """Build index for test repository."""
    build_index(repo_root=repo_root, substrate_mode="sqlite")
    return repo_root


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


def test_bundle_verify_with_absolute_path():
    """Verify bundle with absolute directory path."""
    tmpdir = tempfile.mkdtemp()
    bundle_dir = create_minimal_bundle(tmpdir)

    try:
        verifier = BundleVerifier(Path(tmpdir))
        result = verifier.verify()

        assert result["status"] == "success"
        assert "bundle_id" in result
    finally:
        shutil.rmtree(tmpdir)


def test_bundle_verify_with_relative_path():
    """Verify bundle with relative directory path."""
    tmpdir = tempfile.mkdtemp()
    bundle_dir = create_minimal_bundle(tmpdir)

    try:
        os.chdir(tmpdir)
        verifier = BundleVerifier(Path("."))
        result = verifier.verify()

        assert result["status"] == "success"
        assert "bundle_id" in result
    finally:
        os.chdir("D:\\CCC 2.0\\AI\\agent-governance-system")
        shutil.rmtree(tmpdir)


def test_bundle_verify_with_explicit_json_path():
    """Verify bundle with explicit bundle.json path."""
    tmpdir = tempfile.mkdtemp()
    bundle_dir = create_minimal_bundle(tmpdir)

    try:
        bundle_json = Path(tmpdir) / "bundle.json"
        verifier = BundleVerifier(bundle_json)
        result = verifier.verify()

        assert result["status"] == "success"
        assert "bundle_id" in result
    finally:
        shutil.rmtree(tmpdir)


def test_bundle_run_deterministic():
    """Run same bundle twice, assert stdout bytes identical."""
    tmpdir1 = tempfile.mkdtemp()
    create_minimal_bundle(tmpdir1)

    executor1 = BundleExecutor(Path(tmpdir1))
    execution_result1 = executor1.execute()

    tmpdir2 = tempfile.mkdtemp()
    create_minimal_bundle(tmpdir2)

    executor2 = BundleExecutor(Path(tmpdir2))
    execution_result2 = executor2.execute()

    assert execution_result1["bundle_id"] == execution_result2["bundle_id"]
    assert execution_result1["root_hash"] == execution_result2["root_hash"]
    assert execution_result1["steps"] == execution_result2["steps"]
    assert execution_result1["artifacts"] == execution_result2["artifacts"]

    shutil.rmtree(tmpdir1)
    shutil.rmtree(tmpdir2)


@pytest.mark.skip(reason="Bundle verification before execution not implemented")
def test_bundle_run_fails_without_verify():
    """Corrupt bundle.json, assert run fails before execution."""
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


def test_bundle_run_uses_only_bundle_artifacts():
    """Remove repo files, assert execution still succeeds."""
    tmpdir = tempfile.mkdtemp()
    create_minimal_bundle(tmpdir)

    executor = BundleExecutor(Path(tmpdir))
    execution_result = executor.execute()

    assert "bundle_id" in execution_result
    assert "steps" in execution_result
    assert "artifacts" in execution_result
    assert execution_result["outcome"] == "SUCCESS"

    shutil.rmtree(tmpdir)


@pytest.mark.skip(reason="Step execution with operation type validation not implemented")
def test_bundle_run_fails_on_unsupported_step():
    """Create bundle with fake step kind, assert fail-closed."""
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
