"""
Test Phase 2.4.1B: Write Firewall Integration

Validates write firewall enforcement in production surfaces:
- repo_digest.py receipt writes respect firewall
- Forbidden writes (outside tmp/durable roots) are blocked
- Durable writes require commit gate
- Error receipts include firewall violation details

Run:
    pytest CAPABILITY/TESTBENCH/integration/test_write_firewall_enforcement.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall, FirewallViolation
from CAPABILITY.PRIMITIVES.repo_digest import (
    RepoDigest,
    DigestSpec,
    write_receipt,
)


@pytest.fixture
def test_repo():
    """Create a temporary test repository."""
    with TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir)

        # Create standard catalytic domains
        (test_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True)
        (test_root / "LAW" / "CONTRACTS" / "_runs").mkdir(parents=True, exist_ok=True)
        (test_root / "CORTEX" / "_generated" / "_tmp").mkdir(parents=True)
        (test_root / "CORTEX" / "_generated").mkdir(parents=True, exist_ok=True)

        # Create some test files
        (test_root / "README.md").write_text("# Test Repo")
        (test_root / "src" / "main.py").parent.mkdir(parents=True)
        (test_root / "src" / "main.py").write_text("print('hello')")

        yield test_root


@pytest.fixture
def firewall(test_repo):
    """Create write firewall with standard catalytic domains."""
    return WriteFirewall(
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp", "CORTEX/_generated/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs", "CORTEX/_generated"],
        project_root=test_repo,
        exclusions=["LAW/CANON", ".git"],
    )


def test_repo_digest_write_receipt_with_firewall_tmp(test_repo, firewall):
    """Test write_receipt() with firewall for tmp writes."""
    receipt = {
        "digest": "abc123",
        "file_count": 10,
        "module_version": "1.5b.0",
    }

    out_path = test_repo / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "TEST_RECEIPT.json"

    # Tmp write should succeed without opening commit gate
    write_receipt(out_path, receipt, firewall=firewall)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["digest"] == "abc123"


def test_repo_digest_write_receipt_with_firewall_durable(test_repo, firewall):
    """Test write_receipt() with firewall for durable writes."""
    receipt = {
        "digest": "xyz789",
        "file_count": 20,
        "module_version": "1.5b.0",
    }

    out_path = test_repo / "LAW" / "CONTRACTS" / "_runs" / "DURABLE_RECEIPT.json"

    # Durable write should fail before commit gate opens
    with pytest.raises(FirewallViolation) as exc_info:
        write_receipt(out_path, receipt, firewall=firewall)

    assert exc_info.value.error_code == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"

    # Open commit gate and retry
    firewall.open_commit_gate()
    write_receipt(out_path, receipt, firewall=firewall)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["digest"] == "xyz789"


def test_repo_digest_write_receipt_forbidden_path(test_repo, firewall):
    """Test write_receipt() with firewall blocks forbidden paths."""
    receipt = {
        "digest": "forbidden",
        "file_count": 1,
        "module_version": "1.5b.0",
    }

    # Attempt to write outside allowed domains
    forbidden_path = test_repo / "README_RECEIPT.json"

    with pytest.raises(FirewallViolation) as exc_info:
        write_receipt(forbidden_path, receipt, firewall=firewall)

    # Should fail with "not in domain" error
    assert "DOMAIN" in exc_info.value.error_code or "PATH" in exc_info.value.error_code

    # File should not exist
    assert not forbidden_path.exists()


def test_repo_digest_cli_with_firewall_enforcement(test_repo):
    """Test repo_digest.py CLI enforces firewall via WriteFirewall."""
    # Create digest spec
    spec = DigestSpec(
        repo_root=test_repo,
        exclusions=[".git"],
        durable_roots=["LAW/CONTRACTS/_runs", "CORTEX/_generated"],
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp", "CORTEX/_generated/_tmp"],
    )

    # Run pre-digest (should succeed to tmp)
    pre_digest_path = test_repo / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "PRE_DIGEST.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "CAPABILITY" / "PRIMITIVES" / "repo_digest.py"),
            "--repo-root", str(test_repo),
            "--pre-digest", str(pre_digest_path),
            "--tmp-roots", "LAW/CONTRACTS/_runs/_tmp,CORTEX/_generated/_tmp",
            "--durable-roots", "LAW/CONTRACTS/_runs,CORTEX/_generated",
            "--exclusions", ".git",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"pre-digest failed: {result.stderr}"
    assert pre_digest_path.exists()


def test_repo_digest_cli_forbidden_write_blocked(test_repo):
    """Test repo_digest.py CLI blocks forbidden writes via firewall."""
    # Attempt to write receipt to forbidden location (outside tmp/durable roots)
    forbidden_path = test_repo / "FORBIDDEN_DIGEST.json"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "CAPABILITY" / "PRIMITIVES" / "repo_digest.py"),
            "--repo-root", str(test_repo),
            "--pre-digest", str(forbidden_path),
            "--tmp-roots", "LAW/CONTRACTS/_runs/_tmp",
            "--durable-roots", "LAW/CONTRACTS/_runs",
            "--exclusions", ".git",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail with firewall violation
    assert result.returncode != 0
    assert "Firewall violation" in result.stderr or "FIREWALL" in result.stderr
    assert not forbidden_path.exists()


def test_repo_digest_backwards_compat_without_firewall(test_repo):
    """Test write_receipt() without firewall (backwards compatibility)."""
    receipt = {
        "digest": "compat123",
        "file_count": 5,
        "module_version": "1.5b.0",
    }

    # Write without firewall (legacy behavior)
    out_path = test_repo / "LAW" / "CONTRACTS" / "_runs" / "COMPAT_RECEIPT.json"
    write_receipt(out_path, receipt, firewall=None)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["digest"] == "compat123"


def test_firewall_violation_receipt_format(test_repo, firewall):
    """Test that firewall violations produce deterministic receipts."""
    receipt = {"test": "data"}
    forbidden_path = test_repo / "FORBIDDEN.json"

    try:
        write_receipt(forbidden_path, receipt, firewall=firewall)
        assert False, "Should have raised FirewallViolation"
    except FirewallViolation as e:
        # Verify violation receipt structure
        assert "error_code" in e.violation_receipt
        assert "message" in e.violation_receipt
        assert "path" in e.violation_receipt
        assert "policy_snapshot" in e.violation_receipt

        # Verify policy snapshot includes firewall config
        policy = e.violation_receipt["policy_snapshot"]
        assert "tmp_roots" in policy
        assert "durable_roots" in policy
        assert "commit_gate_open" in policy


def test_multiple_receipts_same_firewall(test_repo, firewall):
    """Test writing multiple receipts through the same firewall instance."""
    # Write to tmp (no commit gate needed)
    tmp_receipt1 = {"digest": "tmp1", "module_version": "1.5b.0"}
    tmp_path1 = test_repo / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "RECEIPT1.json"
    write_receipt(tmp_path1, tmp_receipt1, firewall=firewall)
    assert tmp_path1.exists()

    tmp_receipt2 = {"digest": "tmp2", "module_version": "1.5b.0"}
    tmp_path2 = test_repo / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "RECEIPT2.json"
    write_receipt(tmp_path2, tmp_receipt2, firewall=firewall)
    assert tmp_path2.exists()

    # Open commit gate
    firewall.open_commit_gate()

    # Write to durable (commit gate open)
    durable_receipt = {"digest": "durable", "module_version": "1.5b.0"}
    durable_path = test_repo / "LAW" / "CONTRACTS" / "_runs" / "DURABLE.json"
    write_receipt(durable_path, durable_receipt, firewall=firewall)
    assert durable_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
