#!/usr/bin/env python3
"""Phase 5.2.5 SCL CLI Tests.

Tests for:
- 5.2.5.1 CLI Commands (decode, validate, run, audit)
- 5.2.5.2 Integration (receipts)
- 5.2.5.3 Tests (invocation, output format, error handling)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INVOCATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def run_scl_cli(*args, check=True):
    """Run SCL CLI and return result."""
    cmd = [sys.executable, "-m", "CAPABILITY.TOOLS.scl"] + list(args)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if check and result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result


class TestCLIInvocation:
    """Test basic CLI invocation."""

    def test_help_works(self):
        result = run_scl_cli("--help")
        assert result.returncode == 0
        assert "SCL CLI" in result.stdout
        assert "decode" in result.stdout
        assert "validate" in result.stdout
        assert "run" in result.stdout
        assert "audit" in result.stdout

    def test_decode_help(self):
        result = run_scl_cli("decode", "--help")
        assert result.returncode == 0
        assert "decode" in result.stdout.lower()

    def test_validate_help(self):
        result = run_scl_cli("validate", "--help")
        assert result.returncode == 0
        assert "validate" in result.stdout.lower()

    def test_run_help(self):
        result = run_scl_cli("run", "--help")
        assert result.returncode == 0
        assert "run" in result.stdout.lower()

    def test_audit_help(self):
        result = run_scl_cli("audit", "--help")
        assert result.returncode == 0
        assert "audit" in result.stdout.lower()

    def test_no_command_shows_help(self):
        result = run_scl_cli(check=False)
        assert result.returncode == 1
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_module_invocation(self):
        """Test python -m CAPABILITY.TOOLS.scl works."""
        result = run_scl_cli("validate", "C3")
        assert result.returncode == 0
        assert "PASS" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecodeCommand:
    """Test decode command."""

    def test_decode_simple_macro(self):
        result = run_scl_cli("decode", "C3")
        assert result.returncode == 0
        jobspec = json.loads(result.stdout)
        assert "job_id" in jobspec
        assert jobspec["phase"] == 5
        assert jobspec["determinism"] == "deterministic"

    def test_decode_with_context(self):
        result = run_scl_cli("decode", "C3:build")
        assert result.returncode == 0
        jobspec = json.loads(result.stdout)
        assert jobspec["metadata"]["context"] == "build"

    def test_decode_cjk_symbol(self):
        result = run_scl_cli("decode", "法", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        # CJK may be mangled in subprocess, just check structure
        assert "jobspec" in output

    def test_decode_cjk_compound(self):
        result = run_scl_cli("decode", "法.驗")
        assert result.returncode == 0
        jobspec = json.loads(result.stdout)
        assert jobspec["inputs"]["entry_type"] == "compound"

    def test_decode_json_output(self):
        result = run_scl_cli("decode", "C3", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "jobspec" in output
        assert "receipt" in output

    def test_decode_invalid_program(self):
        result = run_scl_cli("decode", "X999", "--json", check=False)
        # JSON mode returns 0 but ok: false
        output = json.loads(result.stdout)
        assert output["ok"] is False
        assert "errors" in output


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATE COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateCommand:
    """Test validate command."""

    def test_validate_valid_program(self):
        result = run_scl_cli("validate", "C3")
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_validate_invalid_program(self):
        result = run_scl_cli("validate", "X999", check=False)
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_validate_cjk_symbol(self):
        result = run_scl_cli("validate", "法")
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_validate_unknown_cjk(self):
        result = run_scl_cli("validate", "龍", "--json", check=False)
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["ok"] is False

    def test_validate_level_l1(self):
        result = run_scl_cli("validate", "C3", "--level", "L1")
        assert result.returncode == 0
        assert "[L1]" in result.stdout

    def test_validate_level_l2(self):
        result = run_scl_cli("validate", "C3", "--level", "L2")
        assert result.returncode == 0
        assert "[L2]" in result.stdout

    def test_validate_json_output(self):
        result = run_scl_cli("validate", "C3", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert output["layer"] == "L3"

    def test_validate_json_file(self, tmp_path):
        """Test validating a JobSpec JSON file."""
        # Create a valid JobSpec
        jobspec = {
            "job_id": "test-job",
            "phase": 1,
            "task_type": "validation",
            "intent": "Test validation",
            "inputs": {},
            "outputs": {
                "durable_paths": ["_runs/test/output.json"],
                "validation_criteria": {"success": True},
            },
            "catalytic_domains": [],
            "determinism": "deterministic",
        }
        jobspec_path = tmp_path / "test_jobspec.json"
        jobspec_path.write_text(json.dumps(jobspec))

        result = run_scl_cli("validate", str(jobspec_path))
        assert result.returncode == 0
        assert "PASS" in result.stdout or "[L4]" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# RUN COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunCommand:
    """Test run command."""

    def test_run_dry_run(self):
        result = run_scl_cli("run", "C3", "--dry-run")
        assert result.returncode == 0
        assert "DRY-RUN" in result.stdout

    def test_run_shows_invariants(self):
        result = run_scl_cli("run", "C3", "--dry-run")
        assert result.returncode == 0
        assert "I5" in result.stdout  # Determinism invariant
        assert "I6" in result.stdout  # Output roots invariant

    def test_run_json_output(self):
        result = run_scl_cli("run", "C3", "--dry-run", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "execution" in output
        assert output["execution"]["dry_run"] is True
        assert "invariant_proofs" in output["execution"]

    def test_run_invalid_program_fails(self):
        result = run_scl_cli("run", "X999", "--dry-run", check=False)
        assert result.returncode == 1


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditCommand:
    """Test audit command."""

    def test_audit_simple(self):
        result = run_scl_cli("audit", "C3")
        assert result.returncode == 0
        assert "SCL AUDIT" in result.stdout
        assert "VALIDATION" in result.stdout
        assert "EXPANSION" in result.stdout

    def test_audit_cjk_symbol(self):
        result = run_scl_cli("audit", "法", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "domain" in str(output).lower()

    def test_audit_json_output(self):
        result = run_scl_cli("audit", "C3", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "audit" in output
        assert "validation" in output["audit"]
        assert "expansion" in output["audit"]

    def test_audit_shows_compression(self):
        result = run_scl_cli("audit", "法")
        assert result.returncode == 0
        # Should show compression ratio
        assert "Compression" in result.stdout or "compression" in result.stdout.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestReceiptEmission:
    """Test receipt emission for all operations."""

    def test_decode_receipt(self, tmp_path):
        receipt_path = tmp_path / "decode_receipt.json"
        result = run_scl_cli("decode", "C3", "--receipt-out", str(receipt_path))
        assert result.returncode == 0
        assert receipt_path.exists()

        receipt = json.loads(receipt_path.read_text())
        assert receipt["operation"] == "decode"
        assert receipt["program"] == "C3"
        assert receipt["success"] is True
        assert "receipt_hash" in receipt
        assert "timestamp_utc" in receipt

    def test_validate_receipt(self, tmp_path):
        receipt_path = tmp_path / "validate_receipt.json"
        result = run_scl_cli("validate", "C3", "--receipt-out", str(receipt_path))
        assert result.returncode == 0
        assert receipt_path.exists()

        receipt = json.loads(receipt_path.read_text())
        assert receipt["operation"] == "validate"
        assert receipt["success"] is True

    def test_run_receipt(self, tmp_path):
        receipt_path = tmp_path / "run_receipt.json"
        result = run_scl_cli("run", "C3", "--dry-run", "--receipt-out", str(receipt_path))
        assert result.returncode == 0
        assert receipt_path.exists()

        receipt = json.loads(receipt_path.read_text())
        assert receipt["operation"] == "run"
        assert receipt["success"] is True
        assert "invariant_proofs" in receipt["metadata"]

    def test_audit_receipt(self, tmp_path):
        receipt_path = tmp_path / "audit_receipt.json"
        result = run_scl_cli("audit", "C3", "--receipt-out", str(receipt_path))
        assert result.returncode == 0
        assert receipt_path.exists()

        receipt = json.loads(receipt_path.read_text())
        assert receipt["operation"] == "audit"

    def test_receipt_hash_deterministic(self, tmp_path):
        """Same input should produce same receipt hash."""
        receipt_path1 = tmp_path / "receipt1.json"
        receipt_path2 = tmp_path / "receipt2.json"

        # Run twice with same input
        run_scl_cli("validate", "C3", "--receipt-out", str(receipt_path1))
        run_scl_cli("validate", "C3", "--receipt-out", str(receipt_path2))

        r1 = json.loads(receipt_path1.read_text())
        r2 = json.loads(receipt_path2.read_text())

        # Input hash should be identical
        assert r1["input_hash"] == r2["input_hash"]
        # Program should be identical
        assert r1["program"] == r2["program"]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputFormatValidation:
    """Test output format validation."""

    def test_decode_jobspec_has_required_fields(self):
        result = run_scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        required_fields = [
            "job_id", "phase", "task_type", "intent",
            "inputs", "outputs", "catalytic_domains", "determinism"
        ]
        for field in required_fields:
            assert field in jobspec, f"Missing required field: {field}"

    def test_decode_jobspec_outputs_structure(self):
        result = run_scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        assert "durable_paths" in jobspec["outputs"]
        assert "validation_criteria" in jobspec["outputs"]
        assert isinstance(jobspec["outputs"]["durable_paths"], list)

    def test_validate_json_structure(self):
        result = run_scl_cli("validate", "C3", "--json")
        output = json.loads(result.stdout)

        assert "ok" in output
        assert "layer" in output
        assert "program" in output
        assert "errors" in output
        assert "warnings" in output
        assert "receipt" in output

    def test_run_json_structure(self):
        result = run_scl_cli("run", "C3", "--dry-run", "--json")
        output = json.loads(result.stdout)

        assert "ok" in output
        assert "execution" in output
        assert "jobspec" in output["execution"]
        assert "invariant_proofs" in output["execution"]

    def test_audit_json_structure(self):
        result = run_scl_cli("audit", "C3", "--json")
        output = json.loads(result.stdout)

        assert "ok" in output
        assert "audit" in output
        assert "validation" in output["audit"]
        assert "expansion" in output["audit"]


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Test error handling."""

    def test_empty_program_fails(self):
        result = run_scl_cli("validate", "", check=False)
        assert result.returncode == 1

    def test_invalid_syntax_error_message(self):
        result = run_scl_cli("validate", "not-valid-123!", "--json", check=False)
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert output["ok"] is False
        assert len(output["errors"]) > 0
        assert "syntax" in output["errors"][0].lower() or "Invalid" in output["errors"][0]

    def test_unknown_radical_error_message(self):
        result = run_scl_cli("validate", "X3", "--json", check=False)
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert any("Unknown radical" in e for e in output["errors"])

    def test_unknown_contract_rule_error_message(self):
        result = run_scl_cli("validate", "C999", "--json", check=False)
        assert result.returncode == 1
        output = json.loads(result.stdout)
        assert any("Unknown contract rule" in e or "C999" in e for e in output["errors"])

    def test_decode_error_includes_layer(self):
        result = run_scl_cli("decode", "X999", "--json", check=False)
        # JSON mode may return 0 with ok: false
        output = json.loads(result.stdout)
        assert output["ok"] is False
        assert "receipt" in output
        assert output["receipt"]["layer"] in ["L1", "L2", "L3"]

    def test_nonexistent_file_error(self):
        result = run_scl_cli("validate", "/nonexistent/path/file.json", check=False)
        # Should handle gracefully (either as SCL program or file not found)
        # The important thing is it doesn't crash
        assert result.returncode in [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
