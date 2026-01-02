#!/usr/bin/env python3
"""Test that cassettes are loading correctly."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generic_cassette import load_cassettes_from_json

project_root = Path(__file__).parent.parent.parent
config_path = Path(__file__).parent / "cassettes.json"

print(f"Project root: {project_root}")
print(f"Config path: {config_path}")
print(f"Config exists: {config_path.exists()}")

cassettes = load_cassettes_from_json(config_path, project_root)
print(f"\nLoaded {len(cassettes)} cassettes:")

for cassette in cassettes:
    print(f"\n--- {cassette.cassette_id} ---")
    print(f"  Name: {cassette.name}")
    print(f"  DB path: {cassette.db_path}")
    print(f"  DB exists: {cassette.db_path.exists()}")
    print(f"  Capabilities: {cassette.capabilities}")
    
    if cassette.cassette_id == "cat_chat" and cassette.db_path.exists():
        print(f"  Testing query 'index BLOB'...")
        try:
            results = cassette.query("index BLOB", top_k=1)
            print(f"  Found {len(results)} results")
            if results:
                result = results[0]
                print(f"  First result keys: {list(result.keys())}")
                if 'storage_type' in result:
                    print(f"  Storage type: {result['storage_type']}")
                elif 'content' in result:
                    print(f"  Content preview: {result['content'][:100]}...")
        except Exception as e:
            print(f"  Query error: {e}")