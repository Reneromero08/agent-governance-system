"""
Test runtime write firewall (Phase 1.5A).

Validates catalytic domain isolation:
- tmp writes only under tmp roots during execution
- durable writes only under durable roots after commit gate opens
- rename/unlink/mkdir enforcement
- deterministic error codes and receipts

Run:
    pytest CAPABILITY/TESTBENCH/pipeline/test_write_firewall.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add repo root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall, FirewallViolation


@pytest.fixture
def project_root():
    """Create a temporary project root for testing."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def firewall(project_root):
    """Create write firewall with standard catalytic domains."""
    # Create domain directories
    (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True)
    (project_root / "LAW" / "CONTRACTS" / "_runs" / "durable").mkdir(parents=True)
    (project_root / "CAPABILITY" / "PRIMITIVES" / "_scratch").mkdir(parents=True)

    return WriteFirewall(
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp", "CAPABILITY/PRIMITIVES/_scratch"],
        durable_roots=["LAW/CONTRACTS/_runs/durable"],
        project_root=project_root,
        exclusions=["LAW/CANON", ".git"],
    )


def test_tmp_write_succeeds(firewall, project_root):
    """Test that tmp write to tmp root succeeds."""
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/test.json", '{"test": true}', kind="tmp")

    # Verify file was written
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "test.json").exists()
    content = (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "test.json").read_text()
    assert content == '{"test": true}'


def test_tmp_write_bytes_succeeds(firewall, project_root):
    """Test that tmp write with bytes succeeds."""
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/binary.dat", b"\x00\x01\x02", kind="tmp")

    # Verify file was written
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "binary.dat").exists()
    content = (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "binary.dat").read_bytes()
    assert content == b"\x00\x01\x02"


def test_tmp_write_to_durable_fails(firewall):
    """Test that tmp write to durable root fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/durable/test.json", '{"test": true}', kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_TMP_WRITE_WRONG_DOMAIN"
    assert "tmp write attempted outside tmp roots" in violation["message"].lower()


def test_durable_write_before_commit_fails(firewall):
    """Test that durable write before commit gate opens fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/durable/test.json", '{"test": true}', kind="durable")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"
    assert "before commit gate" in violation["message"].lower()


def test_durable_write_after_commit_succeeds(firewall, project_root):
    """Test that durable write after commit gate opens succeeds."""
    firewall.open_commit_gate()
    firewall.safe_write("LAW/CONTRACTS/_runs/durable/test.json", '{"test": true}', kind="durable")

    # Verify file was written
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "durable" / "test.json").exists()
    content = (project_root / "LAW" / "CONTRACTS" / "_runs" / "durable" / "test.json").read_text()
    assert content == '{"test": true}'


def test_durable_write_to_tmp_fails(firewall):
    """Test that durable write to tmp root fails."""
    firewall.open_commit_gate()

    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/test.json", '{"test": true}', kind="durable")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_WRONG_DOMAIN"
    assert "durable write attempted outside durable roots" in violation["message"].lower()


def test_write_outside_domain_fails(firewall):
    """Test that write outside allowed domains fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("README.md", "test", kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN"


def test_write_to_excluded_path_fails(firewall):
    """Test that write to excluded path fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CANON/test.md", "test", kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_EXCLUDED"


def test_path_traversal_fails(firewall):
    """Test that path traversal is detected and rejected."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/../../../etc/passwd", "test", kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_TRAVERSAL"


def test_absolute_path_escape_fails(firewall):
    """Test that absolute path outside project fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("/tmp/test.json", "test", kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_ESCAPE"


def test_safe_mkdir_tmp_succeeds(firewall, project_root):
    """Test that safe_mkdir for tmp path succeeds."""
    firewall.safe_mkdir("LAW/CONTRACTS/_runs/_tmp/subdir", kind="tmp")

    # Verify directory was created
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "subdir").is_dir()


def test_safe_mkdir_durable_before_commit_fails(firewall):
    """Test that safe_mkdir for durable path before commit fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_mkdir("LAW/CONTRACTS/_runs/durable/subdir", kind="durable")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"


def test_safe_mkdir_durable_after_commit_succeeds(firewall, project_root):
    """Test that safe_mkdir for durable path after commit succeeds."""
    firewall.open_commit_gate()
    firewall.safe_mkdir("LAW/CONTRACTS/_runs/durable/subdir", kind="durable")

    # Verify directory was created
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "durable" / "subdir").is_dir()


def test_safe_mkdir_outside_domain_fails(firewall):
    """Test that safe_mkdir outside domain fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_mkdir("MEMORY/test", kind="tmp")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN"


def test_safe_rename_tmp_succeeds(firewall, project_root):
    """Test that safe_rename within tmp domain succeeds."""
    # Create source file
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/old.json", "test", kind="tmp")

    # Rename
    firewall.safe_rename(
        "LAW/CONTRACTS/_runs/_tmp/old.json",
        "LAW/CONTRACTS/_runs/_tmp/new.json"
    )

    # Verify rename
    assert not (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "old.json").exists()
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "new.json").exists()


def test_safe_rename_out_of_domain_fails(firewall, project_root):
    """Test that safe_rename outside domain fails."""
    # Create source file
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/test.json", "test", kind="tmp")

    # Try to rename outside domain
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_rename(
            "LAW/CONTRACTS/_runs/_tmp/test.json",
            "README.md"
        )

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN"


def test_safe_unlink_tmp_succeeds(firewall, project_root):
    """Test that safe_unlink in tmp domain succeeds."""
    # Create file
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/delete_me.json", "test", kind="tmp")

    # Unlink
    firewall.safe_unlink("LAW/CONTRACTS/_runs/_tmp/delete_me.json")

    # Verify deleted
    assert not (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "delete_me.json").exists()


def test_safe_unlink_outside_domain_fails(firewall):
    """Test that safe_unlink outside domain fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_unlink("README.md")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN"


def test_safe_unlink_excluded_path_fails(firewall):
    """Test that safe_unlink on excluded path fails."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_unlink("LAW/CANON/test.md")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_PATH_EXCLUDED"


def test_configure_policy_closes_commit_gate(firewall, project_root):
    """Test that reconfiguring policy closes the commit gate."""
    # Open gate
    firewall.open_commit_gate()

    # Durable write should work
    firewall.safe_write("LAW/CONTRACTS/_runs/durable/before.json", "test", kind="durable")

    # Reconfigure
    firewall.configure_policy(
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs/durable"]
    )

    # Commit gate should be closed now
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/durable/after.json", "test", kind="durable")

    violation = exc_info.value.violation_receipt
    assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"


def test_violation_receipt_structure(firewall):
    """Test that violation receipts have correct structure."""
    try:
        firewall.safe_write("README.md", "test", kind="tmp")
    except FirewallViolation as e:
        violation = e.violation_receipt

        # Check required fields
        assert "firewall_version" in violation
        assert "tool_version_hash" in violation
        assert "verdict" in violation
        assert violation["verdict"] == "FAIL"
        assert "error_code" in violation
        assert "message" in violation
        assert "operation" in violation
        assert "path" in violation
        assert "kind" in violation
        assert "policy_snapshot" in violation

        # Check policy snapshot
        policy = violation["policy_snapshot"]
        assert "tmp_roots" in policy
        assert "durable_roots" in policy
        assert "exclusions" in policy
        assert "commit_gate_open" in policy
        assert "tool_version" in policy
        assert "tool_version_hash" in policy

        # Verify JSON serialization
        json_str = e.to_json()
        parsed = json.loads(json_str)
        assert parsed["verdict"] == "FAIL"


def test_violation_receipt_determinism(firewall):
    """Test that identical violations produce identical receipts (same error code)."""
    # Attempt the same violation twice
    violations = []
    for _ in range(2):
        try:
            firewall.safe_write("README.md", "test", kind="tmp")
        except FirewallViolation as e:
            violations.append(e.violation_receipt)

    # Should have same error code and message
    assert violations[0]["error_code"] == violations[1]["error_code"]
    assert violations[0]["message"] == violations[1]["message"]
    assert violations[0]["verdict"] == violations[1]["verdict"]


def test_violation_receipt_write(firewall, project_root):
    """Test that violation receipts can be written to files."""
    try:
        firewall.safe_write("README.md", "test", kind="tmp")
    except FirewallViolation as e:
        receipt_path = project_root / "violation_receipt.json"
        e.write_receipt(receipt_path)

        # Verify receipt was written
        assert receipt_path.exists()

        # Verify receipt content
        content = json.loads(receipt_path.read_text())
        assert content["verdict"] == "FAIL"
        assert content["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN"


def test_multiple_tmp_roots(firewall, project_root):
    """Test that all tmp roots are recognized."""
    # Write to first tmp root
    firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/test1.json", "test1", kind="tmp")

    # Write to second tmp root
    firewall.safe_write("CAPABILITY/PRIMITIVES/_scratch/test2.json", "test2", kind="tmp")

    # Verify both were written
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "test1.json").exists()
    assert (project_root / "CAPABILITY" / "PRIMITIVES" / "_scratch" / "test2.json").exists()


def test_path_normalization_windows(firewall, project_root):
    """Test that Windows-style paths are normalized correctly."""
    # Use backslashes (Windows-style)
    firewall.safe_write("LAW\\CONTRACTS\\_runs\\_tmp\\test.json", "test", kind="tmp")

    # Verify file was written
    assert (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "test.json").exists()


def test_invalid_write_kind_fails(firewall):
    """Test that invalid write kind is rejected."""
    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_write("LAW/CONTRACTS/_runs/_tmp/test.json", "test", kind="invalid")

    violation = exc_info.value.violation_receipt
    assert violation["verdict"] == "FAIL"
    assert violation["error_code"] == "FIREWALL_INVALID_KIND"
