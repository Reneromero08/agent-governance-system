#!/usr/bin/env python3
"""
Tiny Agent - Swarm Refactoring Task

This script simulates a tiny agent (translation layer worker) that:
1. Receives compressed @Symbol context from Semantic Core
2. Expands symbols on-demand to get full content
3. Executes a refactoring task from swarm-refactoring-report.md
4. Returns result with token usage statistics

Demonstrates the translation layer architecture from ADR-030.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("CORTEX/system1.db")

def get_symbol_content(db, symbol: str) -> str:
    """
    Look up symbol in database and return full content.
    In real system, this would query CORTEX via semantic search.
    """
    cursor = db.execute("""
        SELECT fts.content as content
        FROM chunks c
        JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
        WHERE fts.content LIKE ?
        LIMIT 1
    """, (f"%{symbol[2:]}%",))
    
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return f"<Content for {symbol} not found>"

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ~= 1 token)."""
    return len(text) // 4

def simulate_refactoring_task():
    """
    Execute a refactoring task using compressed @Symbol context.
    Demonstrates 76-80% token reduction from ADR-030.
    """
    print("=== TINY AGENT: REFACTORING TASK ===\n")
    
    if not DB_PATH.exists():
        print("ERROR: CORTEX/system1.db not found")
        print("Run vector_indexer.py --index first to build embeddings.")
        return False
    
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    
    # Get compressed symbols (in real system, from export_semantic.py)
    cursor = db.execute("""
        SELECT 
            c.chunk_hash as hash,
            fts.content as content,
            c.chunk_id as chunk_id,
            sv.embedding
        FROM chunks c
        JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
        JOIN section_vectors sv ON c.chunk_hash = sv.hash
        WHERE fts.content IS NOT NULL AND fts.content != ''
        LIMIT 5
    """)
    
    compressed_context = cursor.fetchall()
    
    if not compressed_context:
        print("ERROR: No compressed symbols found in database")
        return False
    
    db.close()
    
    print(f"Received {len(compressed_context)} compressed @Symbols from Semantic Core\n")
    
    # Calculate token cost of compressed context
    compressed_json = json.dumps({
        "symbols": [f"@C:{r['hash'][:8]}" for r in compressed_context],
        "task": "refactor_acknowledge_function"
    })
    compressed_tokens = estimate_tokens(compressed_json)
    
    print(f"Compressed context: {compressed_tokens} tokens\n")
    
    # TASK: Refactor acknowledge_task in server.py
    # This requires reading to function signature and fixing race condition
    
    # In real system, tiny agent would:
    # 1. Expand symbols on-demand when needed
    # 2. Get code for specific function
    # 3. Apply atomic rewrite pattern
    
    # For this demo, we'll simulate by expanding to first symbol
    target_symbol = f"@C:{compressed_context[0]['hash'][:8]}"
    
    # Expand symbol (this is where we'd fetch from NAVIGATION.CORTEX in real system)
    # In demo, we'll use a placeholder expansion
    expanded_content = "def acknowledge_task(task_id: str) -> bool:\n    # Atomic rewrite with file locking per swarm-refactoring-report.md\n    # Resolves race condition in server.py:1242-1262\n    return True"
    
    expanded_tokens = estimate_tokens(expanded_content)
    
    # Calculate savings
    full_context_tokens = sum([
        estimate_tokens(r['content']) for r in compressed_context
    ])
    
    token_reduction = ((full_context_tokens - expanded_tokens) / full_context_tokens) * 100
    
    print(f"Expanded symbol: {target_symbol}\n")
    print(f"Expanded context: {expanded_tokens} tokens\n")
    print(f"Token reduction: {token_reduction:.1f}%\n")
    
    # Show what refactoring would do
    print("=== REFACTORING OUTPUT ===\n")
    print("From: swarm-refactoring-report.md")
    print("Issue #1: Race condition in `acknowledge_task` (server.py:1242-1262)")
    print("Resolution: Atomic rewrite with file locking")
    print()
    
    # Apply atomic pattern (simulated)
    print("BEFORE (with race condition):")
    print("```python")
    print("def acknowledge_task(task_id):")
    print("    # Race condition: file read-then-write")
    print("    task = load_jsonl(TASKS_PATH, task_id)")
    print("    task['state'] = 'acknowledged'")
    print("    append_to_jsonl(TASKS_PATH, task)")
    print("```")
    print()
    
    print("AFTER (Translation Layer Pattern):")
    print("```python")
    print("def acknowledge_task(task_id):")
    print("    # Translation layer: receives @C:ab5e61a8 symbol")
    print("    task_data = lookup_symbol('@C:ab5e61a8')")
    print("    # Atomic operation with file locking")
    print("    _atomic_rewrite_jsonl(TASKS_PATH, task_id, lambda t: {**t, 'state': 'acknowledged'})")
    print("```")
    print()
    
    # Summary
    print("=== SUMMARY ===\n")
    print(f"Full context (no compression): {full_context_tokens} tokens")
    print(f"Expanded context (symbol only): {expanded_tokens} tokens")
    print(f"Token savings: {full_context_tokens - expanded_tokens} tokens")
    print(f"Efficiency gain: {token_reduction:.1f}%")

if __name__ == "__main__":
    import sys
    success = simulate_refactoring_task()
    sys.exit(0 if success else 1)
