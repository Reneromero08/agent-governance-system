import sys
import sqlite3
import json

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get turn_hydrated events - these contain the FULL turn content
cursor.execute("SELECT payload_json FROM session_events WHERE event_type='turn_hydrated' ORDER BY sequence_num")
rows = cursor.fetchall()

print(f"Found {len(rows)} hydrated turns\n")

for i, (payload_json,) in enumerate(rows[:5]):  # First 5
    data = json.loads(payload_json)
    print(f"=== Hydrated Turn {i+1} ===")
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 500:
            print(f"{key}: {value[:500]}...")
        else:
            print(f"{key}: {value}")
    print()
