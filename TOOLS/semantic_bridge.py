#!/usr/bin/env python3
"""
Semantic Bridge App

Connects directly to agent-governance-system CORTEX and AGI CORTEX databases.
Performs real queries on both databases and shows real data.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Database paths
AGENT_DB = Path("CORTEX/system1.db")
AGI_DB = Path("D:/CCC 2.0/AI/AGI/CORTEX/_generated/system1.db")

@dataclass
class DatabaseMatch:
    """A match from one of the databases."""
    source: str  # 'agent' or 'agi'
    chunk_id: str
    heading: str
    content_preview: str
    hash: str

def query_agent_db(query: str, limit: int = 10) -> List[DatabaseMatch]:
    """Query the agent governance system database."""
    if not AGENT_DB.exists():
        print(f"ERROR: Agent DB not found: {AGENT_DB}")
        return []
    
    conn = sqlite3.connect(str(AGENT_DB))
    conn.row_factory = sqlite3.Row
    
    matches = []
    
    try:
        cursor = conn.execute("""
            SELECT 
                c.chunk_id,
                c.chunk_hash,
                fts.content
            FROM chunks c
            JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
            WHERE fts.content LIKE ?
            LIMIT ?
        """, (f"%{query}%", limit))
        
        for row in cursor.fetchall():
            matches.append(DatabaseMatch(
                source="agent",
                chunk_id=str(row['chunk_id']),
                heading=f"chunk_{row['chunk_id']}",
                content_preview=row['content'][:150] if row['content'] else "",
                hash=row['chunk_hash'][:16] if row['chunk_hash'] else ""
            ))
    finally:
        conn.close()
    
    return matches

def query_agi_db(query: str, limit: int = 10) -> List[DatabaseMatch]:
    """Query the AGI research database."""
    if not AGI_DB.exists():
        print(f"ERROR: AGI DB not found: {AGI_DB}")
        return []
    
    try:
        conn = sqlite3.connect(str(AGI_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT 
                rc.chunk_id,
                rc.doc_id,
                rc.heading,
                rc.content
            FROM research_chunks rc
            WHERE rc.content LIKE ?
            LIMIT ?
        """, (f"%{query}%", limit))
        
        for row in cursor.fetchall():
            matches.append(DatabaseMatch(
                source="agi",
                chunk_id=str(row['chunk_id']),
                heading=row['heading'],
                content_preview=row['content'][:150] if row['content'] else "",
                hash=f"doc_{row['doc_id']}"
            ))
        conn.close()
    except Exception as e:
        print(f"Warning: AGI DB query failed: {e}")
        return []
    
    return matches

def get_agent_stats() -> Dict:
    """Get statistics from agent database."""
    if not AGENT_DB.exists():
        return {}
    
    conn = sqlite3.connect(str(AGENT_DB))
    cursor = conn.execute("""
        SELECT 
            (SELECT COUNT(*) FROM chunks) as total_chunks,
            (SELECT COUNT(DISTINCT file_id) FROM chunks) as docs
        """)
    row = cursor.fetchone()
    stats = {
        "total_chunks": row[0],
        "vectors": 0,  # Vectors are in cortex.db, not system1.db
        "documents": row[1]
    }
    
    # Check if vectors exist
    try:
        cur = conn.execute("SELECT COUNT(*) FROM section_vectors")
        stats["vectors"] = cur.fetchone()[0]
    except sqlite3.OperationalError:
        pass
    
    conn.close()
    return stats

def get_agi_stats() -> Dict:
    """Get statistics from AGI database."""
    if not AGI_DB.exists():
        return {}
    
    try:
        conn = sqlite3.connect(str(AGI_DB))
        cursor = conn.execute("""
            SELECT COUNT(*) as total_chunks
            FROM research_chunks
        """)
        row = cursor.fetchone()
        stats = {
            "total_chunks": row[0]
        }
        conn.close()
    except Exception as e:
        print(f"Warning: AGI DB stats failed: {e}")
        return {"total_chunks": 0}
    
    return stats

def cross_database_search(query: str) -> Dict[str, List[DatabaseMatch]]:
    """Search both databases and return results."""
    agent_matches = query_agent_db(query, limit=5)
    agi_matches = query_agi_db(query, limit=5)
    
    return {
        "agent": agent_matches,
        "agi": agi_matches
    }

def main():
    """Main application."""
    print("=" * 70)
    print("SEMANTIC BRIDGE - REAL DATA FROM BOTH DATABASES")
    print("=" * 70)
    print()
    
    # Show stats
    print("DATABASE STATISTICS")
    print("-" * 70)
    
    agent_stats = get_agent_stats()
    print(f"\nAgent Governance System:")
    print(f"  Database: {AGENT_DB.name}")
    print(f"  Total chunks: {agent_stats['total_chunks']}")
    print(f"  With vectors: {agent_stats['vectors']}")
    print(f"  Documents: {agent_stats['documents']}")
    
    agi_stats = get_agi_stats()
    print(f"\nAGI Research:")
    print(f"  Database: {AGI_DB.name}")
    print(f"  Research chunks: {agi_stats['total_chunks']}")
    
    # Perform searches
    queries = ["governance", "memory", "vector", "cortex", "report"]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        results = cross_database_search(query)
        
        print(f"Agent DB matches: {len(results['agent'])}")
        for i, m in enumerate(results['agent'], 1):
            print(f"  {i}. [{m.source}] {m.chunk_id} - {m.heading}")
            if m.content_preview:
                print(f"      {m.content_preview[:100]}...")
        
        print(f"\nAGI DB matches: {len(results['agi'])}")
        for i, m in enumerate(results['agi'], 1):
            print(f"  {i}. [{m.source}] {m.chunk_id} - {m.heading}")
            if m.content_preview:
                print(f"      {m.content_preview[:100]}...")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("This app queries REAL data from your databases.")
    print("No simulation - actual content shown above.")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
