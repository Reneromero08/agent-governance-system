#!/usr/bin/env python3
"""
Phase 5.2.2 Tests: CODEBOOK.json Validation

Tests that validate the Semiotic Symbol Vocabulary (CODEBOOK.json) is
complete, well-formed, and internally consistent.

Deliverables verified:
    - CODEBOOK.json exists and is valid JSON
    - All radicals defined with paths and token counts
    - All operators defined
    - Contract rules C1-C13 defined
    - Invariants I1-I20 defined
    - Grammar specification complete

Usage:
    pytest CAPABILITY/TESTBENCH/integration/test_codebook_lookup.py -v
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"

sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def codebook() -> dict:
    """Load CODEBOOK.json."""
    assert CODEBOOK_PATH.exists(), f"CODEBOOK.json not found at {CODEBOOK_PATH}"
    with open(CODEBOOK_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def codebook_hash() -> str:
    """Compute SHA-256 of codebook content."""
    content = CODEBOOK_PATH.read_text(encoding='utf-8')
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE AND VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodebookExistence:
    """Test codebook file exists and is valid."""

    def test_codebook_exists(self):
        """CODEBOOK.json must exist."""
        assert CODEBOOK_PATH.exists(), "CODEBOOK.json not found"

    def test_codebook_valid_json(self, codebook):
        """CODEBOOK.json must be valid JSON."""
        assert isinstance(codebook, dict)

    def test_codebook_not_empty(self, codebook):
        """Codebook must have content."""
        assert len(codebook) > 0

    def test_codebook_has_version(self, codebook):
        """Codebook must have version."""
        assert "version" in codebook
        # Should be semver format
        version = codebook["version"]
        parts = version.split(".")
        assert len(parts) >= 2, "Version should be semver format"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAMMAR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrammar:
    """Test grammar specification is complete."""

    def test_grammar_section_exists(self, codebook):
        """Grammar section must exist."""
        assert "grammar" in codebook

    def test_grammar_has_description(self, codebook):
        """Grammar must have description."""
        assert "description" in codebook["grammar"]

    def test_grammar_describes_format(self, codebook):
        """Grammar description must explain format."""
        desc = codebook["grammar"]["description"]
        assert "RADICAL" in desc.upper() or "macro" in desc.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# RADICALS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRadicals:
    """Test all radicals are defined."""

    def test_radicals_section_exists(self, codebook):
        """Radicals section must exist."""
        assert "radicals" in codebook

    def test_required_radicals_exist(self, codebook):
        """All 10 required radicals must be defined."""
        required = ["C", "I", "V", "L", "G", "S", "R", "J", "A", "P"]
        for radical in required:
            assert radical in codebook["radicals"], f"Missing radical: {radical}"

    def test_radical_c_contract(self, codebook):
        """C radical must map to Contract."""
        assert codebook["radicals"]["C"]["domain"] == "Contract"
        assert "CONTRACT" in codebook["radicals"]["C"]["path"].upper()

    def test_radical_i_invariant(self, codebook):
        """I radical must map to Invariant."""
        assert codebook["radicals"]["I"]["domain"] == "Invariant"
        assert "INVARIANT" in codebook["radicals"]["I"]["path"].upper()

    def test_radical_v_verification(self, codebook):
        """V radical must map to Verification."""
        assert codebook["radicals"]["V"]["domain"] == "Verification"

    def test_radical_l_law(self, codebook):
        """L radical must map to Law."""
        assert codebook["radicals"]["L"]["domain"] == "Law"
        assert "LAW" in codebook["radicals"]["L"]["path"].upper()

    def test_all_radicals_have_path(self, codebook):
        """All radicals must have path."""
        for radical, data in codebook["radicals"].items():
            assert "path" in data, f"Radical {radical} missing path"

    def test_all_radicals_have_tokens(self, codebook):
        """All radicals must have token count."""
        for radical, data in codebook["radicals"].items():
            assert "tokens" in data, f"Radical {radical} missing tokens"
            assert data["tokens"] == 1, f"Radical {radical} should be 1 token"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperators:
    """Test all operators are defined."""

    def test_operators_section_exists(self, codebook):
        """Operators section must exist."""
        assert "operators" in codebook

    def test_required_operators_exist(self, codebook):
        """All 7 required operators must be defined."""
        required = ["*", "!", "?", "&", "|", ".", ":"]
        for op in required:
            assert op in codebook["operators"], f"Missing operator: {op}"

    def test_star_means_all(self, codebook):
        """* operator must mean ALL."""
        assert "ALL" in codebook["operators"]["*"]["meaning"].upper()

    def test_bang_means_not(self, codebook):
        """! operator must mean NOT/DENY."""
        meaning = codebook["operators"]["!"]["meaning"].upper()
        assert "NOT" in meaning or "DENY" in meaning

    def test_question_means_check(self, codebook):
        """? operator must mean CHECK/QUERY."""
        meaning = codebook["operators"]["?"]["meaning"].upper()
        assert "CHECK" in meaning or "QUERY" in meaning

    def test_ampersand_means_and(self, codebook):
        """& operator must mean AND."""
        assert "AND" in codebook["operators"]["&"]["meaning"].upper()

    def test_pipe_means_or(self, codebook):
        """| operator must mean OR."""
        assert "OR" in codebook["operators"]["|"]["meaning"].upper()

    def test_dot_means_path(self, codebook):
        """'.' operator must mean PATH/ACCESS."""
        meaning = codebook["operators"]["."]["meaning"].upper()
        assert "PATH" in meaning or "ACCESS" in meaning

    def test_colon_means_context(self, codebook):
        """':' operator must mean CONTEXT/TYPE."""
        meaning = codebook["operators"][":"]["meaning"].upper()
        assert "CONTEXT" in meaning or "TYPE" in meaning

    def test_all_operators_single_token(self, codebook):
        """All operators must be single token."""
        for op, data in codebook["operators"].items():
            assert data["tokens"] == 1, f"Operator {op} should be 1 token"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTRACT RULES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractRules:
    """Test contract rules C1-C13 are defined."""

    def test_contract_rules_section_exists(self, codebook):
        """Contract rules section must exist."""
        assert "contract_rules" in codebook

    def test_all_13_rules_exist(self, codebook):
        """All 13 contract rules must be defined."""
        for i in range(1, 14):
            rule_id = f"C{i}"
            assert rule_id in codebook["contract_rules"], f"Missing contract rule: {rule_id}"

    def test_c1_text_outranks_code(self, codebook):
        """C1 must be 'Text outranks code'."""
        assert "text" in codebook["contract_rules"]["C1"]["summary"].lower()
        assert "code" in codebook["contract_rules"]["C1"]["summary"].lower()

    def test_c3_inbox_requirement(self, codebook):
        """C3 must be 'INBOX requirement'."""
        assert "INBOX" in codebook["contract_rules"]["C3"]["summary"]

    def test_c7_determinism(self, codebook):
        """C7 must be 'Determinism'."""
        assert "determinism" in codebook["contract_rules"]["C7"]["summary"].lower()

    def test_c8_output_roots(self, codebook):
        """C8 must be 'Output roots'."""
        assert "output" in codebook["contract_rules"]["C8"]["summary"].lower()

    def test_all_rules_have_summary(self, codebook):
        """All rules must have summary."""
        for rule_id, data in codebook["contract_rules"].items():
            assert "summary" in data, f"Rule {rule_id} missing summary"

    def test_all_rules_have_full(self, codebook):
        """All rules must have full description."""
        for rule_id, data in codebook["contract_rules"].items():
            assert "full" in data, f"Rule {rule_id} missing full description"


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANTS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Test invariants I1-I20 are defined."""

    def test_invariants_section_exists(self, codebook):
        """Invariants section must exist."""
        assert "invariants" in codebook

    def test_all_20_invariants_exist(self, codebook):
        """All 20 invariants must be defined."""
        for i in range(1, 21):
            inv_id = f"I{i}"
            assert inv_id in codebook["invariants"], f"Missing invariant: {inv_id}"

    def test_i5_determinism(self, codebook):
        """I5 must be 'Determinism'."""
        assert "determinism" in codebook["invariants"]["I5"]["summary"].lower()
        assert codebook["invariants"]["I5"]["id"] == "INV-005"

    def test_i6_output_roots(self, codebook):
        """I6 must be 'Output roots'."""
        assert "output" in codebook["invariants"]["I6"]["summary"].lower()
        assert codebook["invariants"]["I6"]["id"] == "INV-006"

    def test_all_invariants_have_id(self, codebook):
        """All invariants must have formal ID (INV-XXX)."""
        for inv_id, data in codebook["invariants"].items():
            assert "id" in data, f"Invariant {inv_id} missing id"
            assert data["id"].startswith("INV-"), f"Invariant {inv_id} id should be INV-XXX format"

    def test_all_invariants_have_summary(self, codebook):
        """All invariants must have summary."""
        for inv_id, data in codebook["invariants"].items():
            assert "summary" in data, f"Invariant {inv_id} missing summary"

    def test_all_invariants_have_full(self, codebook):
        """All invariants must have full description."""
        for inv_id, data in codebook["invariants"].items():
            assert "full" in data, f"Invariant {inv_id} missing full description"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXTS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContexts:
    """Test contexts are defined."""

    def test_contexts_section_exists(self, codebook):
        """Contexts section must exist."""
        assert "contexts" in codebook

    def test_required_contexts_exist(self, codebook):
        """Required contexts must be defined."""
        required = ["build", "audit", "validate"]
        for ctx in required:
            assert ctx in codebook["contexts"], f"Missing context: {ctx}"


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExamples:
    """Test examples are provided."""

    def test_examples_section_exists(self, codebook):
        """Examples section must exist."""
        assert "examples" in codebook

    def test_basic_examples_exist(self, codebook):
        """Basic examples must be provided."""
        examples = codebook["examples"]
        assert "C3" in examples, "Missing C3 example"
        assert "C*" in examples, "Missing C* example"
        assert "I5" in examples, "Missing I5 example"

    def test_compound_examples_exist(self, codebook):
        """Compound examples must be provided."""
        examples = codebook["examples"]
        assert "C&I" in examples, "Missing C&I example"
        assert "C3:build" in examples or "C3:audit" in examples, "Missing context example"


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacy:
    """Test legacy mapping is provided."""

    def test_legacy_section_exists(self, codebook):
        """Legacy section must exist."""
        assert "legacy" in codebook

    def test_legacy_mappings_have_maps_to(self, codebook):
        """All legacy mappings must have maps_to field."""
        for legacy_id, data in codebook["legacy"].items():
            assert "maps_to" in data, f"Legacy {legacy_id} missing maps_to"

    def test_legacy_items_deprecated(self, codebook):
        """All legacy items should be marked deprecated."""
        for legacy_id, data in codebook["legacy"].items():
            assert data.get("deprecated", False), f"Legacy {legacy_id} should be deprecated"


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test metrics are documented."""

    def test_metrics_section_exists(self, codebook):
        """Metrics section must exist."""
        assert "metrics" in codebook

    def test_token_savings_documented(self, codebook):
        """Token savings must be documented."""
        assert "token_savings" in codebook["metrics"]

    def test_vocabulary_size_documented(self, codebook):
        """Vocabulary size must be documented."""
        assert "vocabulary_size" in codebook["metrics"]

    def test_radical_count_accurate(self, codebook):
        """Radical count should match actual radicals."""
        declared = codebook["metrics"]["vocabulary_size"]["radicals"]
        actual = len(codebook["radicals"])
        assert declared == actual, f"Declared {declared} radicals but found {actual}"

    def test_operator_count_accurate(self, codebook):
        """Operator count should match actual operators."""
        declared = codebook["metrics"]["vocabulary_size"]["operators"]
        actual = len(codebook["operators"])
        assert declared == actual, f"Declared {declared} operators but found {actual}"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentIntegrity:
    """Test content integrity for receipt generation."""

    def test_content_hash_reproducible(self, codebook_hash):
        """Content hash must be reproducible."""
        content2 = CODEBOOK_PATH.read_text(encoding='utf-8')
        hash2 = hashlib.sha256(content2.encode('utf-8')).hexdigest()
        assert codebook_hash == hash2, "Content hash not reproducible"

    def test_content_hash_format(self, codebook_hash):
        """Content hash must be valid SHA-256."""
        assert len(codebook_hash) == 64
        assert all(c in '0123456789abcdef' for c in codebook_hash)

    def test_json_round_trips(self, codebook):
        """JSON must round-trip cleanly."""
        serialized = json.dumps(codebook, ensure_ascii=False, indent=2)
        deserialized = json.loads(serialized)
        assert deserialized == codebook


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsistency:
    """Test internal consistency."""

    def test_contract_count_matches_invariant_map(self, codebook):
        """Contract rules count should be 13."""
        assert len(codebook["contract_rules"]) == 13

    def test_invariant_count_matches_declaration(self, codebook):
        """Invariants count should be 20."""
        assert len(codebook["invariants"]) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
