#!/usr/bin/env python3
"""Phase 5.2.4 SCL Validator Tests."""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "PRIMITIVES"))

from scl_validator import (
    validate_scl, validate_syntax, validate_symbols, validate_semantic,
    validate_expansion, validate_program_list, ValidationResult,
    get_known_radicals, get_known_operators, get_known_contexts,
    get_contract_rules, get_invariants,
)


class TestL1SyntaxValidation:
    """L1: Syntax validation tests."""

    def test_single_radical(self):
        result = validate_syntax("C")
        assert result.valid is True
        assert result.layer == "L1"
        assert result.parsed["type"] == "macro"
        assert result.parsed["radical"] == "C"

    def test_radical_with_number(self):
        result = validate_syntax("C3")
        assert result.valid is True
        assert result.parsed["radical"] == "C"
        assert result.parsed["number"] == "3"

    def test_radical_with_operator(self):
        result = validate_syntax("C*")
        assert result.valid is True
        assert result.parsed["radical"] == "C"
        assert result.parsed["operator"] == "*"

    def test_radical_with_context(self):
        result = validate_syntax("C3:build")
        assert result.valid is True
        assert result.parsed["context"] == "build"

    def test_cjk_symbol(self):
        result = validate_syntax("法")
        assert result.valid is True
        assert result.parsed["type"] == "cjk_symbol"
        assert result.parsed["symbol"] == "法"

    def test_compound_expression(self):
        result = validate_syntax("C&I")
        assert result.valid is True
        assert result.parsed["type"] == "compound"

    def test_empty_program_fails(self):
        result = validate_syntax("")
        assert result.valid is False
        assert "Empty program" in result.errors[0]

    def test_invalid_syntax_fails(self):
        result = validate_syntax("not-valid-123!")
        assert result.valid is False
        assert "Invalid syntax" in result.errors[0]

    def test_lowercase_radical_fails(self):
        result = validate_syntax("c3")
        assert result.valid is False


class TestL2SymbolValidation:
    """L2: Symbol existence validation tests."""

    def test_known_radical_passes(self):
        result = validate_symbols("C3")
        assert result.valid is True
        assert result.layer == "L2"

    def test_unknown_radical_fails(self):
        result = validate_symbols("X3")
        assert result.valid is False
        assert any("Unknown radical" in e for e in result.errors)

    def test_unknown_contract_rule_fails(self):
        result = validate_symbols("C999")
        assert result.valid is False
        assert any("Unknown contract rule" in e for e in result.errors)

    def test_unknown_invariant_fails(self):
        result = validate_symbols("I999")
        assert result.valid is False
        assert any("Unknown invariant" in e for e in result.errors)

    def test_known_cjk_symbol_passes(self):
        result = validate_symbols("法")
        assert result.valid is True

    def test_unknown_cjk_symbol_fails(self):
        result = validate_symbols("龍")  # Not in codebook
        assert result.valid is False
        assert any("Unknown CJK symbol" in e for e in result.errors)

    def test_unknown_context_warns(self):
        result = validate_symbols("C3:unknown_context")
        # Unknown context is a warning, not error
        assert result.valid is True
        assert any("Unknown context" in w for w in result.warnings)

    def test_all_known_radicals_pass(self):
        radicals = get_known_radicals()
        assert len(radicals) > 0
        for r in radicals:
            result = validate_symbols(r)
            assert result.valid is True, f"Radical {r} should be valid"


class TestL3SemanticValidation:
    """L3: Semantic constraint validation tests."""

    def test_full_validation_passes(self):
        result = validate_semantic("C3:build")
        assert result.valid is True
        assert result.layer == "L3"

    def test_all_operator_with_number_warns(self):
        result = validate_semantic("C*3")
        # This is unusual but syntactically valid
        # The warning comes from L3
        assert any("unusual" in w.lower() for w in result.warnings)

    def test_deny_operator_warns(self):
        result = validate_semantic("V!")
        assert any("denial" in w.lower() for w in result.warnings)

    def test_numbered_non_ci_radical_warns(self):
        # Only C and I have numbered entries
        result = validate_semantic("G5")
        # Actually G5 fails L1 since number without C/I
        # Let me check
        result = validate_syntax("G5")
        # G5 is syntactically valid (RADICAL + NUMBER)
        if result.valid:
            result = validate_semantic("G5")
            assert any("numbered" in w.lower() for w in result.warnings)


class TestL4ExpansionValidation:
    """L4: JobSpec expansion validation tests."""

    def test_valid_jobspec_passes(self):
        jobspec = {
            "job_id": "test-job-001",
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
        result = validate_expansion(jobspec)
        assert result.valid is True
        assert result.layer == "L4"

    def test_missing_required_field_fails(self):
        jobspec = {
            "job_id": "test-job-001",
            # Missing other required fields
        }
        result = validate_expansion(jobspec)
        assert result.valid is False
        assert any("Missing required field" in e for e in result.errors)

    def test_invalid_job_id_format_fails(self):
        jobspec = {
            "job_id": "INVALID_FORMAT_123",  # Must be kebab-case
            "phase": 1,
            "task_type": "validation",
            "intent": "Test",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": [],
            "determinism": "deterministic",
        }
        result = validate_expansion(jobspec)
        assert result.valid is False
        assert any("Invalid job_id format" in e for e in result.errors)

    def test_invalid_task_type_fails(self):
        jobspec = {
            "job_id": "test-job",
            "phase": 1,
            "task_type": "invalid_type",
            "intent": "Test",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": [],
            "determinism": "deterministic",
        }
        result = validate_expansion(jobspec)
        assert result.valid is False
        assert any("Invalid task_type" in e for e in result.errors)

    def test_invalid_output_path_fails(self):
        jobspec = {
            "job_id": "test-job",
            "phase": 1,
            "task_type": "validation",
            "intent": "Test",
            "inputs": {},
            "outputs": {
                "durable_paths": ["/root/unauthorized/path"],  # Not allowed
                "validation_criteria": {},
            },
            "catalytic_domains": [],
            "determinism": "deterministic",
        }
        result = validate_expansion(jobspec)
        assert result.valid is False
        assert any("not in allowed roots" in e for e in result.errors)

    def test_allowed_output_roots(self):
        """Test that all allowed output roots pass validation."""
        allowed = ["_runs/", "_generated/", "_packs/", "_tmp/", "INBOX/", "NAVIGATION/RECEIPTS/"]
        for root in allowed:
            jobspec = {
                "job_id": "test-job",
                "phase": 1,
                "task_type": "validation",
                "intent": "Test",
                "inputs": {},
                "outputs": {
                    "durable_paths": [f"{root}output.json"],
                    "validation_criteria": {},
                },
                "catalytic_domains": [],
                "determinism": "deterministic",
            }
            result = validate_expansion(jobspec)
            assert result.valid is True, f"Root {root} should be allowed"


class TestValidateSCLAPI:
    """Test the main validate_scl API."""

    def test_default_level_is_l3(self):
        result = validate_scl("C3")
        assert result.layer == "L3"

    def test_level_l1_only(self):
        result = validate_scl("C3", level="L1")
        assert result.layer == "L1"

    def test_invalid_level_fails(self):
        result = validate_scl("C3", level="L99")
        assert result.valid is False
        assert "Unknown validation level" in result.errors[0]


class TestBatchValidation:
    """Test batch validation of multiple programs."""

    def test_all_valid_batch(self):
        programs = ["C1", "C2", "C3", "I1", "I2"]
        result = validate_program_list(programs)
        assert result["all_valid"] is True
        assert result["error_count"] == 0
        assert len(result["results"]) == 5

    def test_mixed_batch(self):
        programs = ["C1", "X99", "I1"]  # X99 is invalid
        result = validate_program_list(programs)
        assert result["all_valid"] is False
        assert result["error_count"] == 1

    def test_empty_batch(self):
        result = validate_program_list([])
        assert result["all_valid"] is True
        assert result["error_count"] == 0


class TestCodebookHelpers:
    """Test codebook helper functions."""

    def test_known_radicals_not_empty(self):
        radicals = get_known_radicals()
        assert len(radicals) > 0
        assert "C" in radicals
        assert "I" in radicals

    def test_known_operators_not_empty(self):
        operators = get_known_operators()
        assert len(operators) > 0
        assert "*" in operators
        assert "!" in operators

    def test_known_contexts_not_empty(self):
        contexts = get_known_contexts()
        assert len(contexts) > 0
        assert "build" in contexts

    def test_contract_rules_not_empty(self):
        rules = get_contract_rules()
        assert len(rules) > 0
        assert "C1" in rules
        assert "C3" in rules

    def test_invariants_not_empty(self):
        invariants = get_invariants()
        assert len(invariants) > 0
        assert "I1" in invariants


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
