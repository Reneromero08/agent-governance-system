#!/usr/bin/env python3
"""Quick tool test - no model required."""

import sys
sys.path.insert(0, '.')

# Test imports
try:
    from tool_executor_v2 import search_web, wikipedia_lookup, grokipedia_lookup, fetch_url, list_directory
    print("[OK] Tool imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test web search
print("\n=== Testing Web Search ===")
try:
    result = search_web("Python programming")
    print(f"[OK] Search works: {len(result)} chars returned")
    print(f"First 200 chars: {result[:200]}...")
except Exception as e:
    print(f"[FAIL] Search error: {e}")

# Test Wikipedia
print("\n=== Testing Wikipedia ===")
try:
    result = wikipedia_lookup("Python (programming language)")
    print(f"[OK] Wikipedia works: {len(result)} chars returned")
    print(f"First 200 chars: {result[:200]}...")
except Exception as e:
    print(f"[FAIL] Wikipedia error: {e}")

# Test Grokipedia
print("\n=== Testing Grokipedia ===")
try:
    result = grokipedia_lookup("machine learning")
    print(f"[OK] Grokipedia works: {len(result)} chars returned")
    print(f"First 200 chars: {result[:200]}...")
except Exception as e:
    print(f"[FAIL] Grokipedia error: {e}")

# Test directory listing
print("\n=== Testing Directory Listing ===")
try:
    result = list_directory(".")
    print(f"[OK] Directory listing works: {len(result)} chars returned")
    print(f"First 200 chars: {result[:200]}...")
except Exception as e:
    print(f"[FAIL] Directory error: {e}")

print("\n=== All Tests Complete ===")
