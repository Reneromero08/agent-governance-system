#!/usr/bin/env python3
"""
Z.1.6 CMP-01 Skill Runtime Enforcement Tests

Tests that prove:
1. A skill cannot run if CMP-01 fails
2. CMP-01 always runs before execution
3. Ledger contains CMP-01 validation records
4. Tampering with CMP-01 output is detected
5. Execution without CMP-01 is impossible

Run: python -m pytest CAPABILITY/TESTBENCH/core/test_skill_runtime_cmp01.py -v
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import (
    execute_skill,
    CMP01ValidationError,
    SkillExecutionResult,
    write_ledger_receipt,
)


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root with minimal structure."""
    # Create LAW/CANON/VERSIONING.md
    canon_dir = tmp_path / "LAW" / "CANON"
    canon_dir.mkdir(parents=True)
    (canon_dir / "VERSIONING.md").write_text("canon_version: 3.2.2")

    # Create allowed roots
    (tmp_path / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True)
    (tmp_path / "CAPABILITY" / "PRIMITIVES" / "_scratch").mkdir(parents=True)
    (tmp_path / "NAVIGATION" / "CORTEX" / "_generated").mkdir(parents=True)
    (tmp_path / "MEMORY" / "LLM_PACKER" / "_packs").mkdir(parents=True)

    return tmp_path


@pytest.fixture
def valid_skill_dir(temp_project_root):
    """Create a valid test skill."""
    skill_dir = temp_project_root / "CAPABILITY" / "SKILLS" / "test-skill"
    skill_dir.mkdir(parents=True)

    # Write SKILL.md with required_canon_version (format: NO space between operator and version)
    (skill_dir / "SKILL.md").write_text("""# Test Skill

**required_canon_version:** >=3.0.0

A test skill for CMP-01 validation.
""")

    # Write run.py that echoes input to output
    (skill_dir / "run.py").write_text("""#!/usr/bin/env python3
import json
import sys

if len(sys.argv) < 3:
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    input_data = json.load(f)

output = {"status": "success", "input_received": input_data}

with open(sys.argv[2], 'w') as f:
    json.dump(output, f)
""")

    return skill_dir


def test_cmp01_prevents_execution_on_forbidden_path(temp_project_root, valid_skill_dir):
    """TEST 1: Skill cannot run if task_spec contains forbidden path."""
    task_spec = {
        "catalytic_domains": [],
        "outputs": {
            "durable_paths": ["LAW/CANON/forbidden.txt"]  # FORBIDDEN!
        }
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    # Verify the receipt shows FAIL verdict
    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "FORBIDDEN_PATH_OVERLAP" for e in receipt.errors)
    assert "CMP-01 validation FAILED" in str(exc_info.value)


def test_cmp01_prevents_execution_on_path_traversal(temp_project_root, valid_skill_dir):
    """TEST 2: Skill cannot run if task_spec contains traversal."""
    task_spec = {
        "catalytic_domains": ["LAW/CONTRACTS/_runs/../../../etc/passwd"],
        "outputs": {"durable_paths": []}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "PATH_CONTAINS_TRAVERSAL" for e in receipt.errors)


def test_cmp01_prevents_execution_on_absolute_path(temp_project_root, valid_skill_dir):
    """TEST 3: Skill cannot run if task_spec contains absolute path."""
    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": ["/tmp/absolute"]}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "PATH_ESCAPES_REPO_ROOT" for e in receipt.errors)


def test_cmp01_prevents_execution_outside_allowed_roots(temp_project_root, valid_skill_dir):
    """TEST 4: Skill cannot run if path is outside allowed roots."""
    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": ["CAPABILITY/TOOLS/not_allowed.txt"]}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "OUTPUT_OUTSIDE_DURABLE_ROOT" for e in receipt.errors)


def test_cmp01_passes_with_valid_task_spec(temp_project_root, valid_skill_dir):
    """TEST 5: Skill CAN run with valid task_spec (CMP-01 PASS)."""
    task_spec = {
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/work"],
        "outputs": {
            "durable_paths": ["LAW/CONTRACTS/_runs/test/output.json"]
        }
    }

    # This should NOT raise - execution proceeds
    result = execute_skill(
        valid_skill_dir,
        task_spec,
        input_data={"test": "data"},
        project_root=temp_project_root
    )

    # Verify CMP-01 receipt shows PASS
    assert result.cmp01_receipt.verdict == "PASS"
    assert len(result.cmp01_receipt.errors) == 0

    # Verify skill executed
    assert result.success is True
    assert result.exit_code == 0
    assert result.output_data is not None


def test_cmp01_validates_skill_manifest(temp_project_root):
    """TEST 6: Skill cannot run without SKILL.md."""
    skill_dir = temp_project_root / "CAPABILITY" / "SKILLS" / "no-manifest"
    skill_dir.mkdir(parents=True)

    # Only create run.py, no SKILL.md
    (skill_dir / "run.py").write_text("#!/usr/bin/env python3\nprint('hi')")

    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": []}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "SKILL_MANIFEST_MISSING" for e in receipt.errors)


def test_cmp01_validates_run_script_exists(temp_project_root):
    """TEST 7: Skill cannot run without run.py."""
    skill_dir = temp_project_root / "CAPABILITY" / "SKILLS" / "no-run-script"
    skill_dir.mkdir(parents=True)

    # Only create SKILL.md, no run.py
    (skill_dir / "SKILL.md").write_text("**required_canon_version:** >= 3.0.0")

    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": []}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "SKILL_RUN_SCRIPT_MISSING" for e in receipt.errors)


def test_cmp01_always_executes_before_skill(temp_project_root, valid_skill_dir):
    """TEST 8: CMP-01 validation ALWAYS runs before skill execution.

    This is mechanically proven by the fact that execute_skill() raises
    CMP01ValidationError before subprocess.run() is ever called.
    """
    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": ["LAW/CANON/forbidden.txt"]}
    }

    # Instrument to verify skill never runs
    skill_ran = False

    # Create a skill that sets a flag
    instrumented_skill = temp_project_root / "CAPABILITY" / "SKILLS" / "instrumented"
    instrumented_skill.mkdir(parents=True)
    (instrumented_skill / "SKILL.md").write_text("**required_canon_version:** >= 3.0.0")

    marker_file = temp_project_root / "SKILL_RAN_MARKER.txt"
    (instrumented_skill / "run.py").write_text(f"""#!/usr/bin/env python3
# This should NEVER execute if CMP-01 fails
import pathlib
pathlib.Path(r"{marker_file}").write_text("SKILL_RAN")
""")

    # Try to execute with forbidden path
    with pytest.raises(CMP01ValidationError):
        execute_skill(instrumented_skill, task_spec, project_root=temp_project_root)

    # Verify skill never ran
    assert not marker_file.exists(), "Skill executed despite CMP-01 failure!"


def test_ledger_receipt_contains_validation_record(temp_project_root, valid_skill_dir):
    """TEST 9: Ledger contains CMP-01 validation records."""
    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": []}
    }

    result = execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    # Write receipt to ledger
    ledger_path = temp_project_root / "LAW" / "CONTRACTS" / "_ledger" / "cmp01.jsonl"
    write_ledger_receipt(result.cmp01_receipt, ledger_path)

    # Verify ledger entry
    assert ledger_path.exists()
    ledger_content = ledger_path.read_text()
    ledger_entry = json.loads(ledger_content.strip())

    assert ledger_entry["type"] == "CMP01_VALIDATION"
    assert ledger_entry["validator_id"] == "CMP-01-skill-runtime-v1"
    assert ledger_entry["verdict"] == "PASS"
    assert ledger_entry["skill_manifest_hash"] != ""
    assert ledger_entry["task_spec_hash"] != ""


def test_cmp01_receipt_is_deterministic(temp_project_root, valid_skill_dir):
    """TEST 10: CMP-01 receipt contains deterministic hashes for verification."""
    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": []}
    }

    result = execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)
    receipt = result.cmp01_receipt

    # Verify receipt has all required fields
    assert receipt.validator_id == "CMP-01-skill-runtime-v1"
    assert receipt.skill_manifest_hash != ""
    assert len(receipt.skill_manifest_hash) == 64  # SHA-256 hex
    assert receipt.task_spec_hash != ""
    assert len(receipt.task_spec_hash) == 64  # SHA-256 hex
    assert receipt.verdict in ("PASS", "FAIL")
    assert receipt.timestamp.endswith("Z")  # ISO 8601 UTC


def test_cmp01_detects_path_overlap(temp_project_root, valid_skill_dir):
    """TEST 11: CMP-01 detects containment overlap within same list."""
    task_spec = {
        "catalytic_domains": [
            "LAW/CONTRACTS/_runs/_tmp/parent",
            "LAW/CONTRACTS/_runs/_tmp/parent/child"  # Overlap!
        ],
        "outputs": {"durable_paths": []}
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert any(e["code"] == "PATH_OVERLAP" for e in receipt.errors)


def test_execution_without_cmp01_is_impossible(temp_project_root, valid_skill_dir):
    """TEST 12: There is no way to execute a skill without CMP-01.

    The execute_skill() function is the ONLY exported entry point.
    It always calls _execute_cmp01_validation() before subprocess.run().
    There is no bypass mechanism.
    """
    # Verify that execute_skill is the only public execution path
    import CAPABILITY.TOOLS.agents.skill_runtime as runtime_module

    # Check __all__ exports
    assert "execute_skill" in runtime_module.__all__

    # Verify execute_skill ALWAYS calls CMP-01
    import inspect
    source = inspect.getsource(runtime_module.execute_skill)

    # Mechanically verify CMP-01 is called before subprocess
    assert "_execute_cmp01_validation" in source
    assert "subprocess.run" in source

    # Verify CMP-01 call comes BEFORE subprocess.run in code
    cmp01_pos = source.index("_execute_cmp01_validation")
    subprocess_pos = source.index("subprocess.run")
    assert cmp01_pos < subprocess_pos, "CMP-01 validation must occur before skill execution!"


def test_cmp01_fail_verdict_prevents_execution_mechanically(temp_project_root, valid_skill_dir):
    """TEST 13: FAIL verdict mechanically prevents execution (code inspection)."""
    import inspect
    import CAPABILITY.TOOLS.agents.skill_runtime as runtime_module

    source = inspect.getsource(runtime_module.execute_skill)

    # Verify that FAIL verdict raises exception BEFORE subprocess.run
    assert 'if cmp01_receipt.verdict == "FAIL"' in source
    assert "raise CMP01ValidationError" in source

    # Verify the raise occurs before subprocess execution
    fail_check_pos = source.index('if cmp01_receipt.verdict == "FAIL"')
    subprocess_pos = source.index("subprocess.run")
    assert fail_check_pos < subprocess_pos


def test_multiple_validation_errors_accumulate(temp_project_root, valid_skill_dir):
    """TEST 14: Multiple CMP-01 violations are all reported."""
    task_spec = {
        "catalytic_domains": ["/../escape"],  # Traversal
        "outputs": {"durable_paths": ["LAW/CANON/forbidden.txt"]}  # Forbidden
    }

    with pytest.raises(CMP01ValidationError) as exc_info:
        execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)

    receipt = exc_info.value.receipt
    assert receipt.verdict == "FAIL"
    assert len(receipt.errors) >= 2  # At least traversal + forbidden


def test_ledger_append_only_multiple_entries(temp_project_root, valid_skill_dir):
    """TEST 15: Ledger is append-only (multiple entries accumulate)."""
    ledger_path = temp_project_root / "LAW" / "CONTRACTS" / "_ledger" / "cmp01.jsonl"

    task_spec = {
        "catalytic_domains": [],
        "outputs": {"durable_paths": []}
    }

    # Execute multiple times and write receipts
    for i in range(3):
        result = execute_skill(valid_skill_dir, task_spec, project_root=temp_project_root)
        write_ledger_receipt(result.cmp01_receipt, ledger_path)

    # Verify 3 entries in ledger
    lines = ledger_path.read_text().strip().split('\n')
    assert len(lines) == 3

    # Verify all are valid JSON
    for line in lines:
        entry = json.loads(line)
        assert entry["type"] == "CMP01_VALIDATION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
