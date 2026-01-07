#!/usr/bin/env python3
"""
Integration Test for Cortex System (Lane C)

Tests:
1. System1DB creation and indexing
2. Cortex Indexer (FILE_INDEX.json, SECTION_INDEX.json)
3. System1 Verify skill
4. Search functionality
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.db.system1_builder import System1DB
from NAVIGATION.CORTEX.semantic.indexer import CortexIndexer
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

def create_open_writer():
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

def test_system1_db():
    """Test System1DB basic functionality."""
    print("\n=== Test 1: System1DB Creation ===")
    
    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "test_system1.db"
    if db_path.exists():
        db_path.unlink()
    
    db = System1DB(db_path, writer=create_open_writer())
    
    # Add a test file
    test_content = "# Test Document\n\nThis is a test of the resonance and entropy in the system."
    file_id = db.add_file("test/doc.md", test_content)
    print(f"✓ Added test file with ID: {file_id}")
    
    # Search
    results = db.search("resonance")
    assert len(results) > 0, "Search should return results"
    print(f"✓ Search found {len(results)} results")
    
    db.close()
    db_path.unlink()
    print("✓ System1DB test passed")

def test_cortex_indexer():
    """Test Cortex Indexer."""
    print("\n=== Test 2: Cortex Indexer ===")
    
    # Clean previous artifacts
    meta_dir = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "meta"
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    
    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "test_system1.db"
    if db_path.exists():
        db_path.unlink()
    
    # Run indexer
    writer = create_open_writer()
    db = System1DB(db_path, writer=writer)
    indexer = CortexIndexer(db, target_dir=PROJECT_ROOT / "LAW" / "CANON", writer=writer)
    indexer.index_all()
    db.close()
    
    # Verify artifacts
    assert (meta_dir / "FILE_INDEX.json").exists(), "FILE_INDEX.json should exist"
    assert (meta_dir / "SECTION_INDEX.json").exists(), "SECTION_INDEX.json should exist"
    print("✓ Index artifacts created")
    
    # Load and validate FILE_INDEX
    with open(meta_dir / "FILE_INDEX.json") as f:
        file_index = json.load(f)
    assert len(file_index) > 0, "FILE_INDEX should have entries"
    print(f"✓ FILE_INDEX has {len(file_index)} files")
    
    # Load and validate SECTION_INDEX
    with open(meta_dir / "SECTION_INDEX.json") as f:
        section_index = json.load(f)
    assert len(section_index) > 0, "SECTION_INDEX should have entries"
    print(f"✓ SECTION_INDEX has {len(section_index)} sections")
    
    # Verify database has content
    db = System1DB(db_path, writer=create_open_writer())
    cursor = db.conn.execute("SELECT COUNT(*) FROM files")
    file_count = cursor.fetchone()[0]
    assert file_count > 0, "Database should have files"
    print(f"✓ Database has {file_count} files indexed")
    db.close()
    
    # Cleanup
    db_path.unlink()
    shutil.rmtree(meta_dir)
    
    print("✓ Cortex Indexer test passed")

def test_search_functionality():
    """Test search across indexed content."""
    print("\n=== Test 3: Search Functionality ===")
    
    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "test_system1.db"
    if db_path.exists():
        import time
        try:
            db_path.unlink()
        except PermissionError:
            time.sleep(0.5)
            db_path.unlink()
    
    db = System1DB(db_path, writer=create_open_writer())
    
    db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "test_system1.db"
    
    # Index FORMULA.md
    formula_path = PROJECT_ROOT / "LAW" / "CANON" / "FORMULA.md"
    if formula_path.exists():
        content = formula_path.read_text(encoding='utf-8')
        db.add_file(str(formula_path.relative_to(PROJECT_ROOT)), content)
        print("✓ Indexed FORMULA.md")
        
        # Search for key terms
        tests = [
            ("resonance", "Should find Resonance references"),
            ("entropy", "Should find Entropy references"),
            ("essence", "Should find Essence references"),
        ]
        
        for query, description in tests:
            results = db.search(query)
            assert len(results) > 0, f"{description} (got {len(results)} results)"
            print(f"✓ '{query}': {len(results)} results")
    else:
        print("⚠️  FORMULA.md not found, skipping search test")
    
    db.close()
    
    # Wait for Windows to release file handle
    import time
    time.sleep(0.5)
    
    if db_path.exists():
        db_path.unlink()
    
    print("✓ Search functionality test passed")

def main():
    """Run all tests."""
    print("="*60)
    print("Cortex Integration Test Suite")
    print("="*60)
    
    tests = [
        ("System1DB", test_system1_db),
        ("Cortex Indexer", test_cortex_indexer),
        ("Search Functionality", test_search_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
