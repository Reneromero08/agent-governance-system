"""Test FTS5 syntax — the hyphen in INV-005 may be the issue."""
import sqlite3
db = r"D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\cassettes\canon.db"
conn = sqlite3.connect(db)

queries = [
    "INV-005",
    "INV 005",
    '"INV-005"',
    "invariant",
    "contract",
    "determinism",
]
for q in queries:
    try:
        cur = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (q,))
        n = cur.fetchone()[0]
        snippet = ""
        if n > 0:
            cur2 = conn.execute("SELECT snippet(chunks_fts, 0, '<m>', '</m>', '', 40) FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1", (q,))
            row = cur2.fetchone()
            snippet = row[0][:100] if row else ""
        print("MATCH {:<20} -> {} results  {}".format(repr(q), n, snippet))
    except Exception as e:
        print("MATCH {:<20} -> ERROR: {}".format(repr(q), str(e)[:80]))

conn.close()
