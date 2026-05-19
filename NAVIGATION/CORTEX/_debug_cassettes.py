"""Debug cassette network FTS and search paths."""
import sqlite3, sys
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system")

def check_db(db_path, name):
    conn = sqlite3.connect(db_path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("=== {} ===".format(name))
    print("  Tables: {}".format([t[0] for t in tables]))

    # Check chunks count
    try:
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print("  Chunks: {}".format(n))
    except:
        print("  Chunks: N/A")

    # Check FTS
    try:
        n = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        print("  FTS entries: {}".format(n))
    except:
        print("  FTS: N/A")

    # Try FTS query
    try:
        cur = conn.execute("SELECT snippet(chunks_fts, 0, '<m>', '</m>', '', 40) FROM chunks_fts WHERE chunks_fts MATCH 'contract' LIMIT 3")
        rows = cur.fetchall()
        print("  FTS 'contract': {} results".format(len(rows)))
        for r in rows:
            print("    {}".format(r[0][:100]))
    except Exception as e:
        print("  FTS query failed: {}".format(e))

    # Check geometric_index
    try:
        n = conn.execute("SELECT COUNT(*) FROM geometric_index").fetchone()[0]
        print("  Geometric index: {} entries".format(n))
    except:
        print("  Geometric index: N/A")

    conn.close()
    print()

# Check all cassettes
import os
cassette_dir = r"D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\cassettes"
for f in sorted(os.listdir(cassette_dir)):
    if f.endswith(".db"):
        check_db(os.path.join(cassette_dir, f), f)
