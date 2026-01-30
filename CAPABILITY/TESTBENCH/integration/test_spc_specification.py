#!/usr/bin/env python3
"""
Phase 5.3.1 Tests: SPC_SPEC.md Validation

Tests that validate the SPC (Semantic Pointer Compression) specification
is complete, well-formed, and internally consistent.

Deliverables verified:
    - SPC_SPEC.md exists and is normative
    - All pointer types defined (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR)
    - Decoder contract specified
    - Error codes enumerated
    - Fail-closed behavior documented

Usage:
    pytest CAPABILITY/TESTBENCH/integration/test_spc_specification.py -v
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SPC_SPEC_PATH = PROJECT_ROOT / "LAW" / "CANON" / "SEMANTIC" / "SPC_SPEC.md"

sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def spec_content() -> str:
    """Load SPC_SPEC.md content."""
    assert SPC_SPEC_PATH.exists(), f"SPC_SPEC.md not found at {SPC_SPEC_PATH}"
    return SPC_SPEC_PATH.read_text(encoding='utf-8')


@pytest.fixture
def spec_hash(spec_content) -> str:
    """Compute SHA-256 of spec content."""
    return hashlib.sha256(spec_content.encode('utf-8')).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE AND METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecExistence:
    """Test spec file exists and has proper metadata."""

    def test_spec_exists(self):
        """SPC_SPEC.md must exist."""
        assert SPC_SPEC_PATH.exists(), "SPC_SPEC.md not found"

    def test_spec_not_empty(self, spec_content):
        """Spec must have content."""
        assert len(spec_content) > 1000, "Spec appears too short"

    def test_spec_is_normative(self, spec_content):
        """Spec must be marked as NORMATIVE."""
        assert "Status:** NORMATIVE" in spec_content or "status: normative" in spec_content.lower()

    def test_spec_has_canon_id(self, spec_content):
        """Spec must have Canon ID."""
        assert "SEMANTIC-SPC-001" in spec_content

    def test_spec_has_version(self, spec_content):
        """Spec must declare version."""
        assert re.search(r'\*\*Version:\*\*\s*\d+\.\d+\.\d+', spec_content)

    def test_spec_has_phase_reference(self, spec_content):
        """Spec must reference Phase 5.3.1."""
        assert "5.3.1" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# POINTER TYPES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPointerTypes:
    """Test that all pointer types are defined."""

    def test_symbol_ptr_defined(self, spec_content):
        """SYMBOL_PTR must be defined."""
        assert "SYMBOL_PTR" in spec_content
        assert "### 2.1 SYMBOL_PTR" in spec_content

    def test_hash_ptr_defined(self, spec_content):
        """HASH_PTR must be defined."""
        assert "HASH_PTR" in spec_content
        assert "### 2.2 HASH_PTR" in spec_content

    def test_composite_ptr_defined(self, spec_content):
        """COMPOSITE_PTR must be defined."""
        assert "COMPOSITE_PTR" in spec_content
        assert "### 2.3 COMPOSITE_PTR" in spec_content

    def test_symbol_examples(self, spec_content):
        """SYMBOL_PTR must have examples."""
        # Core CJK symbols must be documented
        assert "法" in spec_content  # law
        assert "真" in spec_content  # truth
        assert "道" in spec_content  # way/path

    def test_hash_ptr_format(self, spec_content):
        """HASH_PTR format must be specified."""
        assert "sha256:" in spec_content

    def test_composite_operators(self, spec_content):
        """Composite operators must be listed."""
        operators = [".", ":", "*", "!", "?", "&", "|"]
        for op in operators:
            assert f'`{op}`' in spec_content or f'| `{op}`' in spec_content or f'"{op}"' in spec_content

    def test_composite_grammar(self, spec_content):
        """EBNF grammar must be provided."""
        assert "EBNF" in spec_content or "ebnf" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER CONTRACT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecoderContract:
    """Test decoder contract is specified."""

    def test_decoder_section_exists(self, spec_content):
        """Decoder Contract section must exist."""
        assert "## 3. Decoder Contract" in spec_content

    def test_decoder_interface(self, spec_content):
        """Decoder interface must be specified."""
        assert "decode(" in spec_content

    def test_required_inputs(self, spec_content):
        """Required inputs must be documented."""
        required_inputs = [
            "pointer",
            "context_keys",
            "codebook_id",
            "codebook_sha256",
            "kernel_version",
            "tokenizer_id",
        ]
        for inp in required_inputs:
            assert inp in spec_content, f"Missing required input: {inp}"

    def test_output_types(self, spec_content):
        """Output types must be specified."""
        assert "CanonicalIR" in spec_content or "canonical IR" in spec_content.lower()
        assert "FAIL_CLOSED" in spec_content

    def test_decoder_algorithm(self, spec_content):
        """Decoder algorithm must be provided."""
        assert "### 3.4 Decoder Algorithm" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR CODES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorCodes:
    """Test error codes are enumerated."""

    def test_error_section_exists(self, spec_content):
        """Error Codes section must exist."""
        assert "## 4. Error Codes" in spec_content

    def test_required_error_codes(self, spec_content):
        """All required error codes must be defined."""
        required_codes = [
            "E_CODEBOOK_MISMATCH",
            "E_KERNEL_VERSION",
            "E_SYNTAX",
            "E_UNKNOWN_SYMBOL",
            "E_AMBIGUOUS",
        ]
        for code in required_codes:
            assert code in spec_content, f"Missing error code: {code}"

    def test_error_response_format(self, spec_content):
        """Error response format must be specified."""
        assert "Error Response Format" in spec_content or "error_code" in spec_content

    def test_fail_closed_principle(self, spec_content):
        """FAIL_CLOSED principle must be documented."""
        assert "FAIL_CLOSED" in spec_content
        # Should appear multiple times (principle + usage)
        assert spec_content.count("FAIL_CLOSED") >= 5


# ═══════════════════════════════════════════════════════════════════════════════
# AMBIGUITY RULES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAmbiguityRules:
    """Test ambiguity handling is specified."""

    def test_ambiguity_section_exists(self, spec_content):
        """Ambiguity Rules section must exist."""
        assert "## 5. Ambiguity Rules" in spec_content

    def test_polysemic_symbols(self, spec_content):
        """Polysemic symbol handling must be documented."""
        assert "polysemic" in spec_content.lower() or "Polysemic" in spec_content

    def test_context_disambiguation(self, spec_content):
        """Context-based disambiguation must be explained."""
        assert "context_key" in spec_content or "context_keys" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL IR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalIR:
    """Test canonical IR output is specified."""

    def test_canonical_ir_section(self, spec_content):
        """Canonical IR section must exist."""
        assert "## 6. Canonical IR" in spec_content

    def test_normalization_rules(self, spec_content):
        """Normalization rules must be documented."""
        assert "N1:" in spec_content or "Normalization" in spec_content
        assert "Stable Key Ordering" in spec_content or "sort" in spec_content.lower()

    def test_equality_definition(self, spec_content):
        """Equality definition must be provided."""
        assert "ir_equal" in spec_content or "Equality" in spec_content

    def test_stability_property(self, spec_content):
        """Stability property must be documented."""
        assert "encode(decode(x))" in spec_content or "stabilizes" in spec_content.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY AND DRIFT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityAndDrift:
    """Test security and drift behavior is specified."""

    def test_security_section_exists(self, spec_content):
        """Security section must exist."""
        assert "## 7. Security" in spec_content or "Security" in spec_content

    def test_mandatory_rejections(self, spec_content):
        """Mandatory rejection scenarios must be listed."""
        assert "Mandatory Rejection" in spec_content or "REJECT" in spec_content

    def test_no_silent_degradation(self, spec_content):
        """No silent degradation rule must be stated."""
        assert "silent" in spec_content.lower()
        assert "MUST NOT" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test measured metrics are defined."""

    def test_metrics_section_exists(self, spec_content):
        """Metrics section must exist."""
        assert "## 8. Measured Metrics" in spec_content

    def test_concept_unit_defined(self, spec_content):
        """concept_unit must be defined."""
        assert "concept_unit" in spec_content

    def test_cdr_defined(self, spec_content):
        """CDR (Concept Density Ratio) must be defined."""
        assert "CDR" in spec_content
        assert "concept_units" in spec_content and "tokens" in spec_content

    def test_ecr_defined(self, spec_content):
        """ECR (Exact Match Correctness Rate) must be defined."""
        assert "ECR" in spec_content

    def test_m_required_defined(self, spec_content):
        """M_required must be defined."""
        assert "M_required" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REFERENCES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferences:
    """Test internal cross-references are valid."""

    def test_references_gov_ir_spec(self, spec_content):
        """Must reference GOV_IR_SPEC.md."""
        assert "GOV_IR_SPEC" in spec_content

    def test_references_codebook_sync(self, spec_content):
        """Must reference CODEBOOK_SYNC_PROTOCOL.md."""
        assert "CODEBOOK_SYNC_PROTOCOL" in spec_content

    def test_references_token_receipt_spec(self, spec_content):
        """Must reference TOKEN_RECEIPT_SPEC.md."""
        assert "TOKEN_RECEIPT_SPEC" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT HASH RECEIPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentReceipt:
    """Test content integrity for receipt generation."""

    def test_content_hash_reproducible(self, spec_content, spec_hash):
        """Content hash must be reproducible."""
        # Re-read and hash
        content2 = SPC_SPEC_PATH.read_text(encoding='utf-8')
        hash2 = hashlib.sha256(content2.encode('utf-8')).hexdigest()
        assert spec_hash == hash2, "Content hash not reproducible"

    def test_content_hash_format(self, spec_hash):
        """Content hash must be valid SHA-256."""
        assert len(spec_hash) == 64
        assert all(c in '0123456789abcdef' for c in spec_hash)

    def test_changelog_present(self, spec_content):
        """Changelog section must be present."""
        assert "## Changelog" in spec_content

    def test_appendices_present(self, spec_content):
        """Appendices must be present."""
        assert "## Appendix" in spec_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
