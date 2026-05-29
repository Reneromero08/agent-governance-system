#!/usr/bin/env python3
"""
CLI Output Tests (Phase 6.14)

Tests for machine-readable JSON output, standardized exit codes, and quiet mode.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli_command(args, cwd=None):
    """Run CLI command and capture output.

    Args:
        args: List of command-line arguments
        cwd: Working directory (optional)

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    import os
    cmd = [sys.executable, "-m", "catalytic_chat.cli"] + args

    # Set PYTHONPATH to include the CAT_CHAT directory so the module can be found
    cat_chat_dir = Path(__file__).parent.parent.resolve()
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = str(cat_chat_dir) + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = str(cat_chat_dir)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env
    )
    return result.returncode, result.stdout, result.stderr


def create_test_bundle(bundle_dir):
    """Create a minimal test bundle.

    Args:
        bundle_dir: Path to bundle directory
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    artifact_id = "test_artifact_001"
    artifact_content = "Test content for artifact\n"
    artifact_path = artifacts_dir / f"{artifact_id}.txt"
    artifact_path.write_text(artifact_content)

    import hashlib
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


def create_test_trust_policy(policy_path):
    """Create a minimal test trust policy.

    Args:
        policy_path: Path to trust policy file
    """
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    policy = {
        "policy_version": "1.0.0",
        "allow": []
    }

    with open(policy_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(policy, sort_keys=True, separators=(",", ":")))


def test_bundle_verify_json_stdout_purity():
    """Test that --json outputs only JSON + trailing newline to stdout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = tmpdir / "test_bundle"
        create_test_bundle(bundle_dir)

        exit_code, stdout, stderr = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir), "--json"],
            cwd=tmpdir
        )

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        stdout_stripped = stdout.strip()
        stderr_stripped = stderr.strip()

        try:
            output_json = json.loads(stdout_stripped)
            assert output_json is not None
            assert output_json.get("ok") == True
            assert output_json.get("command") == "bundle_verify"
            assert "bundle_id" in output_json
        except json.JSONDecodeError as e:
            pytest.fail(f"stdout is not valid JSON: {e}\nstdout: {stdout_stripped}")


def test_bundle_verify_json_purity_on_error():
    """Test that --json outputs error JSON on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code, stdout, stderr = run_cli_command(
            ["bundle", "verify", "--bundle", "/nonexistent", "--json"],
            cwd=tmpdir
        )

        assert exit_code != 0, "Expected non-zero exit code"

        stdout_stripped = stdout.strip()

        try:
            output_json = json.loads(stdout_stripped)
            assert output_json is not None
            assert output_json.get("ok") == False
            assert "errors" in output_json
            assert len(output_json["errors"]) > 0
        except json.JSONDecodeError as e:
            pytest.fail(f"stdout is not valid JSON: {e}\nstdout: {stdout_stripped}")


def test_bundle_verify_exit_code_invalid_input():
    """Test exit code 2 for invalid input (missing file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code, stdout, stderr = run_cli_command(
            ["bundle", "verify", "--bundle", "/nonexistent/bundle"],
            cwd=tmpdir
        )

        assert exit_code == 2, f"Expected exit code 2 (invalid input), got {exit_code}"


def test_bundle_verify_exit_code_verification_failed():
    """Test exit code 1 for verification failure (invalid bundle)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = tmpdir / "test_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        bundle_json = bundle_dir / "bundle.json"
        bundle_json.write_text("invalid json")

        exit_code, stdout, stderr = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir)],
            cwd=tmpdir
        )

        assert exit_code == 2 or exit_code == 1, f"Expected exit code 1 or 2, got {exit_code}"


def test_bundle_verify_quiet_mode():
    """Test that --quiet suppresses non-error stderr output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = tmpdir / "test_bundle"
        create_test_bundle(bundle_dir)

        exit_code, stdout, stderr_quiet = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir), "--quiet"],
            cwd=tmpdir
        )

        exit_code_normal, stdout_normal, stderr_normal = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir)],
            cwd=tmpdir
        )

        assert exit_code == exit_code_normal
        assert len(stderr_quiet) < len(stderr_normal), "Quiet mode should produce less stderr"


def test_trust_verify_json_output():
    """Test that trust verify --json outputs JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        policy_path = tmpdir / "trust_policy.json"
        create_test_trust_policy(policy_path)

        exit_code, stdout, stderr = run_cli_command(
            ["trust", "verify", "--trust-policy", str(policy_path), "--json"],
            cwd=tmpdir
        )

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}"

        stdout_stripped = stdout.strip()

        try:
            output_json = json.loads(stdout_stripped)
            assert output_json is not None
            assert output_json.get("ok") == True
            assert output_json.get("command") == "trust_verify"
        except json.JSONDecodeError as e:
            pytest.fail(f"stdout is not valid JSON: {e}\nstdout: {stdout_stripped}")


def test_trust_verify_exit_code_invalid_input():
    """Test exit code 2 for invalid input (missing file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code, stdout, stderr = run_cli_command(
            ["trust", "verify", "--trust-policy", "/nonexistent"],
            cwd=tmpdir
        )

        assert exit_code == 2, f"Expected exit code 2 (invalid input), got {exit_code}"


def test_json_deterministic_output():
    """Test that JSON output is deterministic (identical bytes across runs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = tmpdir / "test_bundle"
        create_test_bundle(bundle_dir)

        exit_code1, stdout1, stderr1 = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir), "--json"],
            cwd=tmpdir
        )

        exit_code2, stdout2, stderr2 = run_cli_command(
            ["bundle", "verify", "--bundle", str(bundle_dir), "--json"],
            cwd=tmpdir
        )

        assert exit_code1 == exit_code2
        assert stdout1 == stdout2, "JSON output should be deterministic"
        assert stdout1.endswith("\n"), "JSON should end with trailing newline"


def test_exit_codes_documented():
    """Test that exit codes match documented values."""
    from catalytic_chat.cli_output import (
        EXIT_OK,
        EXIT_VERIFICATION_FAILED,
        EXIT_INVALID_INPUT,
        EXIT_INTERNAL_ERROR
    )

    assert EXIT_OK == 0, "EXIT_OK should be 0"
    assert EXIT_VERIFICATION_FAILED == 1, "EXIT_VERIFICATION_FAILED should be 1"
    assert EXIT_INVALID_INPUT == 2, "EXIT_INVALID_INPUT should be 2"
    assert EXIT_INTERNAL_ERROR == 3, "EXIT_INTERNAL_ERROR should be 3"
