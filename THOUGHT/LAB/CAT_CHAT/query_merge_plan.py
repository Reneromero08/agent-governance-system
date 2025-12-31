#!/usr/bin/env python3
"""Query cat_chat_index.db to analyze what needs merging - token efficient."""

import sqlite3
from pathlib import Path
import re

DB_PATH = Path(__file__).parent / "cat_chat_index.db"

def query_db():
    """Query the index DB for merge analysis."""
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run index_cat_chat.py first.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    print("=" * 80)
    print("CAT_CHAT MERGE ANALYSIS (from DB)")
    print("=" * 80)
    
    # 1. Find roadmaps
    print("\n📋 ROADMAPS:")
    c.execute('''
        SELECT f.rel_path, co.content
        FROM files f
        JOIN content co ON f.id = co.file_id
        WHERE LOWER(f.rel_path) LIKE '%roadmap%' AND f.extension = '.md'
    ''')
    roadmaps = c.fetchall()
    for path, content in roadmaps:
        lines = content.split('\n')
        headers = [l.strip() for l in lines if l.strip().startswith('#')][:3]
        print(f"\n  📄 {path} ({len(lines)} lines)")
        for h in headers:
            print(f"     {h[:70]}")
    
    # 2. Find changelogs
    print("\n\n📝 CHANGELOGS:")
    c.execute('''
        SELECT f.rel_path, co.content
        FROM files f
        JOIN content co ON f.id = co.file_id
        WHERE LOWER(f.rel_path) LIKE '%changelog%' AND f.extension = '.md'
    ''')
    changelogs = c.fetchall()
    for path, content in changelogs:
        entry_count = len([l for l in content.split('\n') if l.strip().startswith('##')])
        print(f"  📄 {path} ({entry_count} entries)")
    
    # 3. Find TODOs with checkbox counts
    print("\n\n✅ TODO FILES:")
    c.execute('''
        SELECT f.rel_path, co.content
        FROM files f
        JOIN content co ON f.id = co.file_id
        WHERE LOWER(f.rel_path) LIKE '%todo%' AND f.extension = '.md'
    ''')
    todos = c.fetchall()
    total_open = 0
    for path, content in todos:
        checked = len(re.findall(r'\[x\]|\[X\]', content))
        unchecked = len(re.findall(r'\[ \]', content))
        total_open += unchecked
        print(f"  📄 {path}: ✓{checked} done, ☐{unchecked} pending")
    print(f"\n  TOTAL OPEN ITEMS: {total_open}")
    
    # 4. Find commit plans
    print("\n\n📦 COMMIT PLANS:")
    c.execute('''
        SELECT f.rel_path, f.size
        FROM files f
        WHERE (LOWER(f.rel_path) LIKE '%commit%' OR LOWER(f.rel_path) LIKE '%plan%')
          AND f.extension = '.md'
          AND LOWER(f.rel_path) NOT LIKE '%todo%'
          AND LOWER(f.rel_path) NOT LIKE '%roadmap%'
    ''')
    plans = c.fetchall()
    for path, size in plans:
        print(f"  📄 {path} ({size} bytes)")
    
    # 5. Find summaries
    print("\n\n📊 SUMMARIES & REPORTS:")
    c.execute('''
        SELECT f.rel_path, f.size
        FROM files f
        WHERE (LOWER(f.rel_path) LIKE '%summary%' 
           OR LOWER(f.rel_path) LIKE '%report%'
           OR LOWER(f.rel_path) LIKE '%delivery%'
           OR LOWER(f.rel_path) LIKE '%handoff%')
          AND f.extension = '.md'
    ''')
    summaries = c.fetchall()
    for path, size in summaries:
        print(f"  📄 {path} ({size} bytes)")
    
    # 6. Key recommendations
    print("\n\n" + "=" * 80)
    print("MERGE RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n1. CONSOLIDATE {len(roadmaps)} ROADMAPS → single source of truth")
    for path, _ in roadmaps:
        print(f"   • {path}")
    
    if len(changelogs) > 1:
        print(f"\n2. MERGE {len(changelogs)} CHANGELOGS → chronological single file")
        for path, _ in changelogs:
            print(f"   • {path}")
    
    print(f"\n3. EXTRACT {len(todos)} TODO FILES ({total_open} open items)")
    print("   → Integrate into roadmap, mark obsolete ones")
    
    print(f"\n4. ARCHIVE {len(plans)} COMMIT PLANS")
    print("   → Move to archive/ (historical)")
    
    print(f"\n5. REVIEW {len(summaries)} SUMMARIES")
    print("   → Keep recent handoff docs, archive older summaries")
    
    conn.close()

if __name__ == '__main__':
    query_db()
