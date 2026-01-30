#!/usr/bin/env python3
"""
Test Phase 5.2.1: Macro Grammar Validation

Tests the compact macro notation system for governance rules.
Validates token efficiency and semantic correctness.
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "TOOLS"))

from codebook_lookup import (
    parse_macro,
    lookup_macro,
    lookup_entry,
    load_codebook,
    MACRO_PATTERN,
    RADICALS,
    OPERATORS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tiktoken_encoder():
    """Get tiktoken encoder for cl100k_base."""
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def codebook():
    """Load the codebook."""
    return load_codebook()


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO PARSING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMacroParsing:
    """Test the macro pattern parser."""

    def test_simple_radical(self):
        """Single letter radicals should parse."""
        for radical in RADICALS.keys():
            parsed = parse_macro(radical)
            assert parsed is not None, f"Failed to parse: {radical}"
            assert parsed['radical'] == radical
            assert parsed['number'] is None
            assert parsed['operator'] is None

    def test_radical_with_number(self):
        """Radical + number should parse correctly."""
        cases = [
            ('C3', 'C', 3),
            ('I5', 'I', 5),
            ('C10', 'C', 10),
            ('I20', 'I', 20),
        ]
        for macro, expected_radical, expected_num in cases:
            parsed = parse_macro(macro)
            assert parsed is not None, f"Failed to parse: {macro}"
            assert parsed['radical'] == expected_radical
            assert parsed['number'] == expected_num

    def test_radical_with_operator(self):
        """Radical + operator should parse correctly."""
        cases = [
            ('C*', 'C', '*', 'ALL'),
            ('V!', 'V', '!', 'NOT'),
            ('J?', 'J', '?', 'CHECK'),
        ]
        for macro, expected_radical, expected_op, expected_meaning in cases:
            parsed = parse_macro(macro)
            assert parsed is not None, f"Failed to parse: {macro}"
            assert parsed['radical'] == expected_radical
            assert parsed['operator'] == expected_op
            assert parsed['operator_meaning'] == expected_meaning

    def test_radical_with_context(self):
        """Radical + number + context should parse correctly."""
        cases = [
            ('C3:build', 'C', 3, 'build'),
            ('I5:audit', 'I', 5, 'audit'),
            ('C1:security', 'C', 1, 'security'),
        ]
        for macro, expected_radical, expected_num, expected_ctx in cases:
            parsed = parse_macro(macro)
            assert parsed is not None, f"Failed to parse: {macro}"
            assert parsed['radical'] == expected_radical
            assert parsed['number'] == expected_num
            assert parsed['context'] == expected_ctx

    def test_invalid_macros(self):
        """Invalid patterns should not parse."""
        invalid = [
            'X3',       # Invalid radical
            '3C',       # Number before radical
            'CC3',      # Double radical
            '@C3',      # Legacy @ prefix
            'C-3',      # Invalid operator
        ]
        for macro in invalid:
            parsed = parse_macro(macro)
            assert parsed is None, f"Should not parse: {macro}"


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO LOOKUP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMacroLookup:
    """Test macro resolution."""

    def test_lookup_contract_rules(self, codebook):
        """All contract rules C1-C13 should resolve."""
        for i in range(1, 14):
            result = lookup_macro(f'C{i}')
            assert result['found'], f"C{i} not found"
            assert result['entry']['type'] == 'contract_rule'
            assert 'summary' in result['entry']

    def test_lookup_invariants(self, codebook):
        """All invariants I1-I20 should resolve."""
        for i in range(1, 21):
            result = lookup_macro(f'I{i}')
            assert result['found'], f"I{i} not found"
            assert result['entry']['type'] == 'invariant'
            assert 'summary' in result['entry']

    def test_lookup_all_contracts(self):
        """C* should return all contract rules."""
        result = lookup_macro('C*')
        assert result['found']
        assert result['entry']['type'] == 'collection'
        assert result['entry']['count'] == 13

    def test_lookup_all_invariants(self):
        """I* should return all invariants."""
        result = lookup_macro('I*')
        assert result['found']
        assert result['entry']['type'] == 'collection'
        assert result['entry']['count'] == 20

    def test_lookup_radicals(self, codebook):
        """Domain radicals should resolve to paths."""
        for radical in RADICALS.keys():
            result = lookup_macro(radical)
            if result['found']:
                assert 'path' in result['entry'] or 'domain' in result['entry']

    def test_context_preservation(self):
        """Context tag should be preserved in result."""
        result = lookup_macro('C3:build')
        assert result['found']
        assert result['entry'].get('context') == 'build'


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN EFFICIENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenEfficiency:
    """Validate token savings vs verbose format."""

    def test_radicals_are_single_token(self, tiktoken_encoder):
        """All radicals should be single tokens."""
        for radical in RADICALS.keys():
            tokens = len(tiktoken_encoder.encode(radical))
            assert tokens == 1, f"{radical} is {tokens} tokens, expected 1"

    def test_operators_are_single_token(self, tiktoken_encoder):
        """All operators should be single tokens."""
        for op in OPERATORS.keys():
            tokens = len(tiktoken_encoder.encode(op))
            assert tokens == 1, f"{op} is {tokens} tokens, expected 1"

    def test_compact_vs_verbose_savings(self, tiktoken_encoder):
        """Compact notation should save at least 50% vs verbose."""
        comparisons = [
            ('C3', '@CONTRACT_RULE_3'),
            ('I5', '@INVARIANT_5'),
            ('C*', '@ALL_CONTRACT'),
            ('G', '@DOMAIN_GOVERNANCE'),
            ('V!', '@STRICT_VALIDATION'),
        ]

        total_compact = 0
        total_verbose = 0

        for compact, verbose in comparisons:
            c_tok = len(tiktoken_encoder.encode(compact))
            v_tok = len(tiktoken_encoder.encode(verbose))
            total_compact += c_tok
            total_verbose += v_tok

        savings = (total_verbose - total_compact) / total_verbose
        assert savings >= 0.50, f"Only {savings:.0%} savings, expected >= 50%"

    def test_rule_lookup_tokens(self, tiktoken_encoder):
        """Rule lookups should be 2 tokens (radical + number)."""
        rules = ['C1', 'C10', 'I5', 'I20']
        for rule in rules:
            tokens = len(tiktoken_encoder.encode(rule))
            assert tokens == 2, f"{rule} is {tokens} tokens, expected 2"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH SEMANTIC SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Test integration with CJK semantic symbols."""

    def test_lookup_entry_handles_both(self):
        """lookup_entry should handle both macros and CJK."""
        # Macro
        result = lookup_entry('C3')
        assert result['found']
        assert result['entry']['type'] == 'contract_rule'

        # CJK
        result = lookup_entry('法')
        assert result['found']
        assert result['entry']['type'] == 'domain'

    def test_combined_vocabulary_size(self):
        """Total vocabulary should be 50+ symbols."""
        from codebook_lookup import SEMANTIC_SYMBOLS

        codebook = load_codebook()
        semantic_count = len(SEMANTIC_SYMBOLS)
        contract_count = len(codebook.get('contract_rules', {}))
        invariant_count = len(codebook.get('invariants', {}))
        radical_count = len(codebook.get('radicals', {}))

        total = semantic_count + contract_count + invariant_count + radical_count
        assert total >= 50, f"Only {total} symbols, expected >= 50"


# ═══════════════════════════════════════════════════════════════════════════════
# CODEBOOK VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodebookValidation:
    """Validate codebook structure and content."""

    def test_codebook_version(self, codebook):
        """Codebook should have version 0.2.0+."""
        version = codebook.get('version', '0.0.0')
        major, minor, patch = map(int, version.split('.'))
        assert (major, minor) >= (0, 2), f"Version {version} < 0.2.0"

    def test_grammar_defined(self, codebook):
        """Grammar section should exist."""
        assert 'grammar' in codebook
        assert 'radicals' in codebook['grammar']
        assert 'operators' in codebook['grammar']

    def test_legacy_mappings(self, codebook):
        """Legacy mappings should exist for migration."""
        legacy = codebook.get('legacy', {})
        assert '@DOMAIN_GOVERNANCE' in legacy
        assert legacy['@DOMAIN_GOVERNANCE']['maps_to'] == 'G'
        assert legacy['@DOMAIN_GOVERNANCE']['deprecated'] is True

    def test_metrics_documented(self, codebook):
        """Token savings metrics should be documented."""
        metrics = codebook.get('metrics', {})
        assert 'token_savings' in metrics
        assert 'vocabulary_size' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
