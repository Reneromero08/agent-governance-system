import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
db = r"D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\cassettes\thought.db"
conn = sqlite3.connect(db)

# Files containing phase4b or VALIDATION
cur = conn.execute("SELECT path FROM files WHERE path LIKE '%phase4b%' OR path LIKE '%VALIDATION%'")
print("Files matching phase4b/VALIDATION:")
for r in cur.fetchall():
    print("  " + r[0])

cur = conn.execute("SELECT COUNT(*) FROM files")
print("Total files: {}".format(cur.fetchone()[0]))

# Content with Phase
cur = conn.execute("SELECT c0 FROM chunks_fts_content WHERE c0 LIKE '%Phase%' LIMIT 3")
print("\nContent samples with 'Phase':")
for r in cur.fetchall():
    txt = r[0]
    # Find position of Phase
    idx = txt.find("Phase")
    if idx >= 0:
        print("  ...{}...".format(txt[max(0,idx-10):idx+40]))

# Content with epistemic  
cur = conn.execute("SELECT COUNT(*) FROM chunks_fts_content WHERE c0 LIKE '%epistemic%'")
print("\nContent rows with 'epistemic': {}".format(cur.fetchone()[0]))

conn.close()
