#!/usr/bin/env python3
"""
Integration Test for Cortex System (Lane C)

Tests:
1. System1DB creation and indexing
2. Search functionality

These tests run serially (xdist_group) and use unique DB paths to avoid conflicts.
"""

import sys
import gc
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.db.system1_builder import System1DB
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


def _create_open_writer() -> GuardedWriter:
    """Create a GuardedWriter with open commit gate for testing."""
    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "NAVIGATION/CORTEX/meta",
            "NAVIGATION/CORTEX/db"
        ]
    )
    writer.open_commit_gate()
    return writer


def _unique_db_path(tmp_path: Path) -> Path:
    """Generate unique DB path under the real project root to avoid parallel conflicts."""
    unique_id = hex(hash(str(tmp_path)))[-8:]
    return PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / f"test_system1_{unique_id}.db"


def _cleanup_db(db: System1DB, db_path: Path) -> None:
    """Safely close DB and clean up, handling Windows file locking."""
    db.close()
    gc.collect()  # Help release file handles on Windows
    try:
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass  # Windows file locking - will be cleaned up later


@pytest.mark.xdist_group("serial_cortex")
def test_system1_db(tmp_path: Path):
    """Test System1DB basic functionality."""
    db_path = _unique_db_path(tmp_path)

    try:
        writer = _create_open_writer()
        db = System1DB(db_path, writer=writer)

        # Add a test file
        test_content = "# Test Document\n\nThis is a test of the resonance and entropy in the system."
        file_id = db.add_file("test/doc.md", test_content)
        assert file_id is not None

        # Search
        results = db.search("resonance")
        assert len(results) > 0, "Search should return results"

        _cleanup_db(db, db_path)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except (PermissionError, FileNotFoundError):
            pass


@pytest.mark.xdist_group("serial_cortex")
def test_search_functionality(tmp_path: Path):
    """Test search across indexed content."""
    db_path = _unique_db_path(tmp_path)

    try:
        writer = _create_open_writer()
        db = System1DB(db_path, writer=writer)

        # Add content with key terms
        content = """# Test Formula Document

## Resonance Section
This discusses the concept of resonance in the system.
Resonance is key to understanding feedback loops.

## Entropy Section
Entropy measures disorder in the system.
Managing entropy is essential for stability.

## Essence Section
The essence of the system is its core purpose.
Understanding essence helps guide decisions.
"""
        db.add_file("test/formula.md", content)

        # Search for key terms
        for query in ["resonance", "entropy", "essence"]:
            results = db.search(query)
            assert len(results) > 0, f"Search for '{query}' should return results"

        _cleanup_db(db, db_path)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except (PermissionError, FileNotFoundError):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
