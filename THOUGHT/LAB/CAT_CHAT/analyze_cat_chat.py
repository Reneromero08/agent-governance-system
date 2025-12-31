#!/usr/bin/env python3
"""Query CAT_CHAT index to find duplicates and plan merges."""

import sqlite3
import json
from pathlib import Path
from difflib import SequenceMatcher

DB_PATH = Path(__file__).parent / "cat_chat_index.db"

def similarity(a: str, b: str) -> float:
    """Calculate text similarity (0-1)."""
    return SequenceMatcher(None, a, b).ratio()

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("=" * 80)
    print("CAT_CHAT CONTENT ANALYSIS")
    print("=" * 80)
    
    # 1. Find exact duplicate documents
    print("\n📄 EXACT DUPLICATE DOCUMENTS")
    print("-" * 80)
    c.execute('''
        SELECT content_hash, COUNT(*) as cnt
        FROM files
        WHERE extension IN ('.md', '.txt') AND content_hash != ''
        GROUP BY content_hash
        HAVING cnt > 1
        ORDER BY cnt DESC
    ''')
    
    exact_dupes = c.fetchall()
    if exact_dupes:
        for hash_val, cnt in exact_dupes:
            c.execute('''
                SELECT f.rel_path, f.size
                FROM files f
                WHERE f.content_hash = ?
                ORDER BY f.rel_path
            ''', (hash_val,))
            paths = c.fetchall()
            print(f"\n{cnt} identical copies ({paths[0][1]} bytes):")
            for path, size in paths:
                print(f"  • {path}")
    else:
        print("No exact duplicates found")
    
    # 2. Find similar documents (by filename patterns)
    print("\n\n📋 ROADMAP/PLANNING DOCUMENTS")
    print("-" * 80)
    c.execute('''
        SELECT f.rel_path, f.size, LENGTH(co.content) as content_len
        FROM files f
        LEFT JOIN content co ON f.id = co.file_id
        WHERE f.rel_path LIKE '%ROADMAP%' OR f.rel_path LIKE '%PLAN%'
        ORDER BY f.rel_path
    ''')
    
    roadmaps = c.fetchall()
    for path, size, content_len in roadmaps:
        print(f"  • {path} ({content_len or 0} chars)")
    
    # 3. Find CHANGELOG documents
    print("\n\n📝 CHANGELOG DOCUMENTS")
    print("-" * 80)
    c.execute('''
        SELECT f.rel_path, f.size
        FROM files f
        WHERE f.rel_path LIKE '%CHANGELOG%'
        ORDER BY f.rel_path
    ''')
    
    changelogs = c.fetchall()
    for path, size in changelogs:
        print(f"  • {path} ({size} bytes)")
    
    # 4. Find commit/delivery notes
    print("\n\n📦 COMMIT/DELIVERY NOTES")
    print("-" * 80)
    c.execute('''
        SELECT f.rel_path, f.size
        FROM files f
        WHERE f.rel_path LIKE '%COMMIT%' 
           OR f.rel_path LIKE '%DELIVERY%'
           OR f.rel_path LIKE '%SUMMARY%'
        ORDER BY f.rel_path
    ''')
    
    commits = c.fetchall()
    for path, size in commits:
        print(f"  • {path} ({size} bytes)")
    
    # 5. Show all documentation in GPT documentation folder
    print("\n\n📚 GPT DOCUMENTATION FOLDER")
    print("-" * 80)
    c.execute('''
        SELECT f.rel_path, f.size, co.content
        FROM files f
        LEFT JOIN content co ON f.id = co.file_id
        WHERE f.rel_path LIKE 'GPT documentation%'
        ORDER BY f.rel_path
    ''')
    
    gpt_docs = c.fetchall()
    for path, size, content in gpt_docs:
        # Show first 200 chars of content
        preview = (content[:200] + '...') if content and len(content) > 200 else (content or '')
        print(f"\n  📄 {path} ({size} bytes)")
        if preview:
            print(f"     Preview: {preview.strip()[:100]}")
    
    # 6. List all TODOs
    print("\n\n✅ TODO DOCUMENTS")
    print("-" * 80)
    c.execute('''
        SELECT f.rel_path, co.content
        FROM files f
        LEFT JOIN content co ON f.id = co.file_id
        WHERE f.rel_path LIKE '%TODO%'
        ORDER BY f.rel_path
    ''')
    
    todos = c.fetchall()
    for path, content in todos:
        lines = content.split('\n') if content else []
        unchecked = [l for l in lines if '[ ]' in l]
        checked = [l for l in lines if '[x]' in l or '[X]' in l]
        print(f"  • {path}: {len(unchecked)} pending, {len(checked)} done")
    
    # 7. Analysis summary
    print("\n\n" + "=" * 80)
    print("MERGE RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. ROADMAP consolidation needed:")
    print("   → Merge all ROADMAP files into single source of truth")
    
    if len(changelogs) > 1:
        print(f"\n2. CHANGELOG consolidation:")
        print(f"   → Found {len(changelogs)} changelogs, merge into one")
    
    print("\n3. GPT documentation folder:")
    print("   → Review and integrate into main docs")
    
    print("\n4. TODO cleanup:")
    print("   → Consolidate phase TODOs into roadmap")
    
    conn.close()

if __name__ == '__main__':
    main()
