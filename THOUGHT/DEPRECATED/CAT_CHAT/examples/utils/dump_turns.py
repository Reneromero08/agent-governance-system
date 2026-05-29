import sys
import sqlite3
import json

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT payload_json FROM session_events WHERE event_type='turn_stored' ORDER BY sequence_num")
rows = cursor.fetchall()

print(f"Found {len(rows)} stored turns. Showing last 5 responses:\n")

for i, (payload_json,) in enumerate(rows[-5:]):
    data = json.loads(payload_json)
    summary = data.get("summary", "NO SUMMARY")
    original_tokens = data.get("original_tokens", "N/A")
    print(f"--- Turn {len(rows)-5+i+1} ---")
    print(f"Tokens: {original_tokens}")
    print(f"Summary: {summary[:200]}...")
    print("-" * 50)
