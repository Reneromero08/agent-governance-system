#!/usr/bin/env python3
"""
Research Cache Utility

Handles persistent caching of research summaries to avoid redundant web browsing.
Stores data in CONTEXT/research/research_cache.db (SQLite).

Usage:
    python TOOLS/research_cache.py --lookup https://example.com
    python TOOLS/research_cache.py --save https://example.com "Summary of the page..." --tags "ai,future"
    python TOOLS/research_cache.py --list
    python TOOLS/research_cache.py --clear
"""

import argparse
import hashlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Enable internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = PROJECT_ROOT / "CONTEXT" / "research" / "research_cache.db"

def get_db():
    """Connect to the database and ensure schema exists."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            summary TEXT NOT NULL,
            tags TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def hash_url(url: str) -> str:
    """Standardized URL hashing."""
    return hashlib.sha256(url.strip().lower().encode('utf-8')).hexdigest()

def save_cache(url: str, summary: str, tags: Optional[str] = None):
    """Save a summary to the cache."""
    url_hash = hash_url(url)
    conn = get_db()
    now = datetime.now().isoformat()
    
    conn.execute("""
        INSERT OR REPLACE INTO cache (url_hash, url, summary, tags, timestamp, last_accessed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (url_hash, url.strip(), summary.strip(), tags, now, now))
    conn.commit()
    conn.close()
    print(f"[OK] Cached {url} ({url_hash[:8]})")

def lookup_cache(url: str) -> Optional[Dict]:
    """Look up a URL in the cache."""
    url_hash = hash_url(url)
    conn = get_db()
    
    row = conn.execute("SELECT * FROM cache WHERE url_hash = ?", (url_hash,)).fetchone()
    if row:
        # Update last accessed
        conn.execute("UPDATE cache SET last_accessed = ? WHERE url_hash = ?", (datetime.now().isoformat(), url_hash))
        conn.commit()
        res = dict(row)
        conn.close()
        return res
    
    conn.close()
    return None

def list_cache(tag: Optional[str] = None):
    """List all entries, optionally filtered by tag."""
    conn = get_db()
    if tag:
        rows = conn.execute("SELECT url, url_hash, tags, timestamp FROM cache WHERE tags LIKE ?", (f"%{tag}%",)).fetchall()
    else:
        rows = conn.execute("SELECT url, url_hash, tags, timestamp FROM cache ORDER BY timestamp DESC").fetchall()
    
    conn.close()
    
    if not rows:
        print("Cache is empty.")
        return

    print(f"{'URL':<60} {'Hash':<10} {'Tags':<20}")
    print("-" * 95)
    for row in rows:
        url = row['url']
        if len(url) > 59: url = url[:56] + "..."
        print(f"{url:<60} {row['url_hash'][:8]:<10} {str(row['tags']):<20}")

def clear_cache():
    """Wipe the database."""
    if CACHE_DB.exists():
        CACHE_DB.unlink()
        print("[OK] Cache cleared.")
    else:
        print("[!] No cache found.")

def main():
    parser = argparse.ArgumentParser(description="Research Cache Utility")
    parser.add_argument("--save", nargs=2, metavar=('URL', 'SUMMARY'), help="Save a summary to cache")
    parser.add_argument("--tags", help="Tags for the saved entry (comma-separated)")
    parser.add_argument("--lookup", metavar='URL', help="Look up a URL")
    parser.add_argument("--list", action="store_true", help="List all cached entries")
    parser.add_argument("--filter", metavar='TAG', help="Filter listing by tag")
    parser.add_argument("--clear", action="store_true", help="Clear the research cache")
    
    args = parser.parse_args()
    
    if args.save:
        save_cache(args.save[0], args.save[1], args.tags)
    elif args.lookup:
        entry = lookup_cache(args.lookup)
        if entry:
            print(f"URL: {entry['url']}")
            print(f"Hash: {entry['url_hash']}")
            print(f"Tags: {entry['tags']}")
            print(f"Cached at: {entry['timestamp']}")
            print("\nSummary:")
            print("-" * 20)
            print(entry['summary'])
        else:
            print(f"[!] No record found for {args.lookup}")
            sys.exit(1)
    elif args.list:
        list_cache(args.filter)
    elif args.clear:
        clear_cache()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
