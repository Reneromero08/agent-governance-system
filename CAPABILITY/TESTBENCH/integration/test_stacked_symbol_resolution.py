#!/usr/bin/env python3
"""Phase 5.2.3.1 Stacked Resolution Tests."""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS"))

from codebook_lookup import (
    stacked_lookup, lookup_entry, _get_domain_paths, _fts_search_within_paths,
    _get_index_db_for_paths, _semantic_search_within_paths, SEMANTIC_SYMBOLS,
    SYSTEM1_DB, CANON_INDEX_DB,
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


@pytest.mark.skipif(not SYSTEM1_DB.exists(), reason="system1.db deprecated - FTS via cassette network")
class TestL1L2FTSResolution:
    def test_fts_stacked_resolution(self):
        result = stacked_lookup("法", query="verification", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L2"
        assert result["entry"]["chunk_count"] > 0


@pytest.mark.skipif(not CANON_INDEX_DB.exists(), reason="canon_index.db deprecated - semantic via cassette network")
class TestL1L3SemanticResolution:
    def test_semantic_stacked_resolution(self):
        result = stacked_lookup("法", semantic="verification protocols", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L3"
        assert result["entry"]["chunk_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
