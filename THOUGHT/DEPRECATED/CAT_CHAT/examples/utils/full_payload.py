import sys
import sqlite3
import json

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get turn_stored events which contain the full payload with user/assistant messages
cursor.execute("""
    SELECT payload_json FROM session_events 
    WHERE event_type = 'turn_stored' 
    ORDER BY sequence_num DESC 
    LIMIT 5
""")
rows = cursor.fetchall()

print(f"Last 5 turn_stored events:\n")

for i, (payload_json,) in enumerate(rows):
    data = json.loads(payload_json)
    print(f"=== Turn {i+1} ===")
    # Print ALL keys to see what's actually stored
    for key in data.keys():
        print(f"  {key}")
    print()
    # Pretty print the full payload
    print(json.dumps(data, indent=2)[:2000])
    print("\n" + "="*60 + "\n")
