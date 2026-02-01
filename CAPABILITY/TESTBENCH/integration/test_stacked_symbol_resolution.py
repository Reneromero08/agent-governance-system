#!/usr/bin/env python3
"""Phase 5.2.3.1 Stacked Resolution Tests.

Tests for L1 symbol resolution via codebook lookup.

Note: L1+L2 (FTS) and L1+L3 (semantic) tests removed - these databases
were deprecated and replaced by the cassette network. Archived tests at:
MEMORY/ARCHIVE/deprecated_tests/test_stacked_symbol_resolution_deprecated.py
"""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS"))

from codebook_lookup import (
    stacked_lookup, lookup_entry, _get_domain_paths,
    SEMANTIC_SYMBOLS,
)


class TestL1SymbolResolution:
    def test_cjk_symbol_resolution(self):
        result = lookup_entry("法")
        assert result["found"] is True
        assert result["entry"]["path"] == "LAW/CANON"

    def test_stacked_l1_only(self):
        result = stacked_lookup("法")
        assert result["found"] is True
        assert result["resolution"] == "L1"


class TestDomainPathExtraction:
    def test_single_path_symbol(self):
        paths = _get_domain_paths(SEMANTIC_SYMBOLS["法"])
        assert paths == ["LAW/CANON"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
