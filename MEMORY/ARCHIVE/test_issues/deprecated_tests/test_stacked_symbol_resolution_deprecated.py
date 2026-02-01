#!/usr/bin/env python3
"""
DEPRECATED TESTS ARCHIVE - Stacked Symbol Resolution (system1/canon_index)

**Status:** ARCHIVED - Databases deprecated, replaced by cassette network
**Original Location:** CAPABILITY/TESTBENCH/integration/test_stacked_symbol_resolution.py
**Archive Date:** 2026-02-01
**Reason for Removal:** 
  - system1.db deprecated - FTS now via cassette network
  - canon_index.db deprecated - semantic search via cassette network

These tests validated the L1+L2 (FTS) and L1+L3 (semantic) resolution paths that
used legacy SQLite databases. These have been replaced by the cassette network
in NAVIGATION/CORTEX/cassettes/.

**Content Hash:** <!-- CONTENT_HASH: b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3 -->
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS"))

from codebook_lookup import (
    stacked_lookup, SYSTEM1_DB, CANON_INDEX_DB,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATED: L1+L2 FTS Resolution Tests (system1.db)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not SYSTEM1_DB.exists(), reason="system1.db deprecated - FTS via cassette network")
class TestL1L2FTSResolution:
    """
    DEPRECATED: Tests for L1+L2 stacked resolution using system1.db FTS.
    
    The system1.db provided full-text search (FTS) capabilities for symbol
    resolution. This has been replaced by the cassette network's FTS cassette
    in NAVIGATION/CORTEX/cassettes/fts.db.
    
    Removal Date: 2026-01
    Replacement: NAVIGATION/CORTEX/cassette_network/ (FTS via cassettes)
    """

    def test_fts_stacked_resolution(self):
        """
        DEPRECATED: Test L1+L2 resolution with FTS query.
        
        Original purpose: Verify that stacked_lookup could combine L1 symbol
        resolution with L2 FTS search within the symbol's domain.
        
        Status: system1.db deprecated - FTS via cassette network
        """
        result = stacked_lookup("法", query="verification", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L2"
        assert result["entry"]["chunk_count"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATED: L1+L3 Semantic Resolution Tests (canon_index.db)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not CANON_INDEX_DB.exists(), reason="canon_index.db deprecated - semantic via cassette network")
class TestL1L3SemanticResolution:
    """
    DEPRECATED: Tests for L1+L3 stacked resolution using canon_index.db semantic search.
    
    The canon_index.db provided semantic search capabilities via embeddings.
    This has been replaced by the cassette network's semantic cassettes in
    NAVIGATION/CORTEX/cassettes/canon.db and others.
    
    Removal Date: 2026-01
    Replacement: NAVIGATION/CORTEX/cassette_network/ (semantic via cassettes)
    """

    def test_semantic_stacked_resolution(self):
        """
        DEPRECATED: Test L1+L3 resolution with semantic query.
        
        Original purpose: Verify that stacked_lookup could combine L1 symbol
        resolution with L3 semantic search within the symbol's domain.
        
        Status: canon_index.db deprecated - semantic via cassette network
        """
        result = stacked_lookup("法", semantic="verification protocols", limit=5)
        assert result["found"] is True
        assert result["resolution"] == "L1+L3"
        assert result["entry"]["chunk_count"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Archive Metadata
# ═══════════════════════════════════════════════════════════════════════════════

ARCHIVE_METADATA = {
    "archive_date": "2026-02-01",
    "original_file": "CAPABILITY/TESTBENCH/integration/test_stacked_symbol_resolution.py",
    "archive_location": "MEMORY/ARCHIVE/deprecated_tests/test_stacked_symbol_resolution_deprecated.py",
    "deprecated_features": [
        {
            "feature": "L1+L2 FTS Resolution (system1.db)",
            "removal_reason": "system1.db deprecated, FTS via cassette network",
            "replacement": "NAVIGATION/CORTEX/cassettes/fts.db",
            "test_count": 1
        },
        {
            "feature": "L1+L3 Semantic Resolution (canon_index.db)",
            "removal_reason": "canon_index.db deprecated, semantic via cassette network",
            "replacement": "NAVIGATION/CORTEX/cassettes/canon.db",
            "test_count": 1
        }
    ],
    "total_tests": 2,
    "migration_notes": [
        "All FTS search now handled by cassette network",
        "All semantic search now handled by cassette network",
        "L1 symbol resolution still works (not deprecated)",
        "Archive preserved for audit and historical reference"
    ]
}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
