#!/usr/bin/env python3
"""
Initialize CAT_CHAT database with indexing information.

This script adds the database indexing information from the user's prompt
to the cat_chat_index.db database so it can be queried without wasting tokens.
"""

import sys
from pathlib import Path

# Add CORTEX to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORTEX_ROOT = PROJECT_ROOT / "NAVIGATION" / "CORTEX"
NETWORK_ROOT = CORTEX_ROOT / "network"
sys.path.insert(0, str(CORTEX_ROOT))
sys.path.insert(0, str(NETWORK_ROOT))

try:
    from cassettes.cat_chat_cassette import CatChatCassette
    print("[INFO] Successfully imported CatChatCassette")
except ImportError as e:
    print(f"[ERROR] Failed to import CatChatCassette: {e}")
    # Try alternative import
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from NAVIGATION.CORTEX.network.cassettes.cat_chat_cassette import CatChatCassette
        print("[INFO] Successfully imported CatChatCassette via alternative path")
    except ImportError as e2:
        print(f"[ERROR] Alternative import also failed: {e2}")
        sys.exit(1)

def main():
    """Initialize CAT_CHAT database with indexing information."""
    print("Initializing CAT_CHAT database with indexing information...")
    
    # Create cassette instance
    cassette = CatChatCassette()
    
    # Check if database exists
    if not cassette.db_path.exists():
        print(f"[WARNING] Database not found at {cassette.db_path}")
        print("Creating database...")
        cassette.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add indexing information
    print("Adding indexing information to database...")
    cassette.add_indexing_info("indexing_info")
    
    # Get stats to verify
    stats = cassette.get_stats()
    if "error" in stats:
        print(f"[ERROR] Failed to get stats: {stats['error']}")
    else:
        print(f"[SUCCESS] Database initialized successfully!")
        print(f"  - Tables: {list(stats.keys())}")
        if "total_chunks" in stats:
            print(f"  - Total chunks: {stats['total_chunks']}")
        if "total_documents" in stats:
            print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Has FTS: {stats.get('has_fts', False)}")
        print(f"  - With vectors: {stats.get('with_vectors', 0)}")
        print(f"  - Capabilities: {stats.get('capabilities', [])}")
    
    # Test query
    print("\nTesting query for 'indexing'...")
    results = cassette.query("indexing", top_k=3)
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('path', 'Unknown')}")
        if 'content' in result:
            content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
            print(f"     Preview: {content_preview}")

if __name__ == "__main__":
    main()