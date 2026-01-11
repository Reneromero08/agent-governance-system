#!/usr/bin/env python3
"""
Phase 5.3.4 Tests: TOKENIZER_ATLAS CI Gate

Tests that enforce single-token stability for preferred semantic symbols.
These tests MUST pass before merging any changes that could affect tokenization.

CI Gate Logic:
    - All preferred symbols MUST be single-token under all tracked tokenizers
    - Tokenizer drift (symbol becoming multi-token) MUST fail the build
    - Atlas MUST be regenerable and match stored artifact

Usage:
    pytest CAPABILITY/TESTBENCH/integration/test_phase_5_3_4_tokenizer_atlas.py -v
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
ATLAS_PATH = PROJECT_ROOT / "LAW" / "CANON" / "SEMANTIC" / "TOKENIZER_ATLAS.json"

# Add project root for imports
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def atlas() -> dict:
    """Load the TOKENIZER_ATLAS.json artifact."""
    assert ATLAS_PATH.exists(), f"Atlas not found: {ATLAS_PATH}"
    with open(ATLAS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    try:
        import tiktoken
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtlasSchema:
    """Test atlas schema validity."""

    def test_atlas_exists(self):
        """Atlas artifact must exist."""
        assert ATLAS_PATH.exists(), "TOKENIZER_ATLAS.json not found"

    def test_atlas_valid_json(self, atlas):
        """Atlas must be valid JSON."""
        assert isinstance(atlas, dict)

    def test_atlas_has_required_fields(self, atlas):
        """Atlas must have all required fields."""
        required = [
            "version",
            "schema",
            "generated_utc",
            "tokenizers",
            "symbols",
            "preferred_glyphs",
            "preferred_single_token_enforced",
            "statistics",
            "content_hash",
        ]
        for field in required:
            assert field in atlas, f"Missing required field: {field}"

    def test_atlas_version(self, atlas):
        """Atlas version must be 1.0.0."""
        assert atlas["version"] == "1.0.0"

    def test_atlas_schema_identifier(self, atlas):
        """Atlas must have correct schema identifier."""
        assert atlas["schema"] == "TOKENIZER_ATLAS_V1"

    def test_atlas_has_tokenizers(self, atlas):
        """Atlas must track at least 2 tokenizers."""
        assert len(atlas["tokenizers"]) >= 2
        assert "cl100k_base" in atlas["tokenizers"]
        assert "o200k_base" in atlas["tokenizers"]


# ═══════════════════════════════════════════════════════════════════════════════
# CI GATE TESTS (Critical - These Must Pass)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleTokenEnforcement:
    """
    CI Gate: Enforce single-token stability for preferred symbols.

    These tests MUST pass. Failure indicates tokenizer drift that would
    degrade compression ratios in production.
    """

    def test_preferred_symbols_single_token(self, atlas):
        """
        CRITICAL: All preferred symbols must be single-token under all tokenizers.

        This is the primary CI gate. If this fails:
        1. A tokenizer update has caused drift
        2. The symbol is no longer efficient for compression
        3. Consider replacing with an alternative single-token symbol
        """
        violations = []

        for symbol in atlas["preferred_single_token_enforced"]:
            if symbol not in atlas["symbols"]:
                violations.append(f"'{symbol}' not found in atlas")
                continue

            for tokenizer in atlas["tokenizers"]:
                count = atlas["symbols"][symbol].get(tokenizer, -1)
                if count != 1:
                    violations.append(
                        f"'{symbol}' is {count} tokens under {tokenizer} (expected 1)"
                    )

        assert len(violations) == 0, (
            f"Single-token enforcement failed with {len(violations)} violations:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_preferred_count_matches(self, atlas):
        """Preferred glyphs count must match enforced count."""
        assert len(atlas["preferred_glyphs"]) == len(atlas["preferred_single_token_enforced"])

    def test_minimum_seven_preferred(self, atlas):
        """Must have at least 7 preferred single-token symbols."""
        assert len(atlas["preferred_single_token_enforced"]) >= 7, (
            f"Only {len(atlas['preferred_single_token_enforced'])} preferred symbols, need >= 7"
        )

    @pytest.mark.skipif(
        not Path(PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "generate_tokenizer_atlas.py").exists(),
        reason="Generator script not found"
    )
    def test_live_verification(self, atlas, tiktoken_available):
        """
        Live verification: Recompute token counts and verify they match atlas.

        This catches cases where the atlas is stale.
        """
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        import tiktoken

        mismatches = []
        for symbol in atlas["preferred_single_token_enforced"]:
            for encoding_name in atlas["tokenizers"]:
                try:
                    enc = tiktoken.get_encoding(encoding_name)
                    live_count = len(enc.encode(symbol))
                    stored_count = atlas["symbols"][symbol].get(encoding_name, -1)

                    if live_count != stored_count:
                        mismatches.append(
                            f"'{symbol}' under {encoding_name}: "
                            f"atlas={stored_count}, live={live_count}"
                        )
                except Exception as e:
                    mismatches.append(f"Failed to verify '{symbol}': {e}")

        assert len(mismatches) == 0, (
            f"Atlas is stale - {len(mismatches)} mismatches found:\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentIntegrity:
    """Test content hash and determinism."""

    def test_content_hash_present(self, atlas):
        """Atlas must have content hash."""
        assert "content_hash" in atlas
        assert len(atlas["content_hash"]) == 64  # SHA-256 hex

    def test_content_hash_valid(self, atlas):
        """Content hash must be valid SHA-256 hex."""
        content_hash = atlas["content_hash"]
        assert all(c in "0123456789abcdef" for c in content_hash)

    def test_symbol_categories_complete(self, atlas):
        """All symbol categories must be present."""
        categories = atlas.get("symbol_categories", {})
        assert "cjk_symbols" in categories
        assert "radicals" in categories
        assert "operators" in categories

    def test_no_negative_token_counts(self, atlas):
        """No symbol should have negative token count (error indicator)."""
        for symbol, counts in atlas["symbols"].items():
            for tokenizer, count in counts.items():
                assert count > 0, f"'{symbol}' has invalid count {count} under {tokenizer}"


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatistics:
    """Test atlas statistics."""

    def test_statistics_present(self, atlas):
        """Statistics section must be present."""
        assert "statistics" in atlas
        stats = atlas["statistics"]
        assert "total_symbols" in stats
        assert "single_token_cjk" in stats

    def test_total_symbols_positive(self, atlas):
        """Must track at least 1 symbol."""
        assert atlas["statistics"]["total_symbols"] > 0

    def test_single_token_count_reasonable(self, atlas):
        """Should have at least 7 single-token CJK symbols."""
        assert atlas["statistics"]["single_token_cjk"] >= 7

    def test_symbol_count_matches(self, atlas):
        """Statistics total must match actual symbol count."""
        assert atlas["statistics"]["total_symbols"] == len(atlas["symbols"])


# ═══════════════════════════════════════════════════════════════════════════════
# RADICALS AND OPERATORS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRadicalsAndOperators:
    """Test ASCII radicals and operators are single-token."""

    def test_all_radicals_single_token(self, atlas):
        """All ASCII radicals must be single-token."""
        radicals = atlas.get("symbol_categories", {}).get("radicals", [])
        for radical in radicals:
            if radical in atlas["symbols"]:
                for tokenizer in atlas["tokenizers"]:
                    count = atlas["symbols"][radical].get(tokenizer, -1)
                    assert count == 1, (
                        f"Radical '{radical}' is {count} tokens under {tokenizer}"
                    )

    def test_all_operators_single_token(self, atlas):
        """All operators must be single-token."""
        operators = atlas.get("symbol_categories", {}).get("operators", [])
        for op in operators:
            if op in atlas["symbols"]:
                for tokenizer in atlas["tokenizers"]:
                    count = atlas["symbols"][op].get(tokenizer, -1)
                    assert count == 1, (
                        f"Operator '{op}' is {count} tokens under {tokenizer}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# O200K COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestO200kCoverage:
    """Test o200k_base tokenizer has good CJK coverage."""

    def test_o200k_better_cjk_coverage(self, atlas):
        """o200k_base should have more single-token CJK than cl100k_base."""
        cjk_symbols = atlas.get("symbol_categories", {}).get("cjk_symbols", [])

        cl100k_singles = sum(
            1 for s in cjk_symbols
            if atlas["symbols"].get(s, {}).get("cl100k_base", 0) == 1
        )
        o200k_singles = sum(
            1 for s in cjk_symbols
            if atlas["symbols"].get(s, {}).get("o200k_base", 0) == 1
        )

        assert o200k_singles >= cl100k_singles, (
            f"o200k_base ({o200k_singles}) should have >= cl100k_base ({cl100k_singles}) "
            "single-token CJK symbols"
        )

    def test_o200k_single_token_list_valid(self, atlas):
        """o200k_single_token_only list should all be single-token under o200k."""
        o200k_only = atlas.get("o200k_single_token_only", [])
        for symbol in o200k_only:
            if symbol in atlas["symbols"]:
                count = atlas["symbols"][symbol].get("o200k_base", -1)
                assert count == 1, (
                    f"'{symbol}' listed as o200k-single but is {count} tokens"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegression:
    """Regression tests for known good states."""

    def test_fa_single_token(self, atlas):
        """法 (law) must always be single-token - highest compression symbol."""
        assert atlas["symbols"]["法"]["cl100k_base"] == 1
        assert atlas["symbols"]["法"]["o200k_base"] == 1

    def test_zhen_single_token(self, atlas):
        """真 (truth) must always be single-token."""
        assert atlas["symbols"]["真"]["cl100k_base"] == 1
        assert atlas["symbols"]["真"]["o200k_base"] == 1

    def test_dao_single_token(self, atlas):
        """道 (way/path) must always be single-token."""
        assert atlas["symbols"]["道"]["cl100k_base"] == 1
        assert atlas["symbols"]["道"]["o200k_base"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
