#!/usr/bin/env python3
"""
Tests for CORTEX/query.py module.

Ensures the query module has all required functions that other modules depend on.
This test was added to prevent the AttributeError regression where export_to_json()
was missing but called by cortex.build.py.
"""

import sys
from pathlib import Path

# Add CORTEX to path for imports
CORTEX_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORTEX_DIR.parent
sys.path.insert(0, str(CORTEX_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def test_export_to_json_exists():
    """Verify export_to_json function exists in query module."""
    import query as cortex_query
    
    assert hasattr(cortex_query, 'export_to_json'), \
        "query module must have export_to_json() function (required by cortex.build.py)"
    assert callable(cortex_query.export_to_json), \
        "export_to_json must be callable"
    print("✓ export_to_json exists and is callable")


def test_export_to_json_returns_dict():
    """Verify export_to_json returns expected structure."""
    import query as cortex_query
    
    result = cortex_query.export_to_json()
    
    assert isinstance(result, dict), \
        f"export_to_json must return dict, got {type(result)}"
    assert "entities" in result, \
        "export_to_json result must have 'entities' key"
    assert "metadata" in result, \
        "export_to_json result must have 'metadata' key"
    assert isinstance(result["entities"], list), \
        "entities must be a list"
    assert isinstance(result["metadata"], dict), \
        "metadata must be a dict"
    print(f"✓ export_to_json returns valid structure (entities: {len(result['entities'])})")


def test_cortex_query_class_exists():
    """Verify CortexQuery class exists with required methods."""
    import query as cortex_query
    
    assert hasattr(cortex_query, 'CortexQuery'), \
        "query module must have CortexQuery class"
    
    required_methods = ['search', 'get_summary', 'find_sections', 'get_neighbors']
    for method in required_methods:
        assert hasattr(cortex_query.CortexQuery, method), \
            f"CortexQuery must have {method}() method"
    print(f"✓ CortexQuery class has all required methods: {', '.join(required_methods)}")


def main():
    """Run all query module tests."""
    print("=" * 60)
    print("CORTEX/query.py Module Tests")
    print("=" * 60)
    
    tests = [
        ("export_to_json exists", test_export_to_json_exists),
        ("export_to_json returns dict", test_export_to_json_returns_dict),
        ("CortexQuery class exists", test_cortex_query_class_exists),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Result: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("✓ All query module tests passed")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
