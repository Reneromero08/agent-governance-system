import sys
import sqlite3
import json

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Check all event types
cursor.execute("SELECT DISTINCT event_type FROM session_events")
event_types = cursor.fetchall()
print("Event types:", [e[0] for e in event_types])

# Dump a full turn_stored event to see all fields
cursor.execute("SELECT payload_json FROM session_events WHERE event_type='turn_stored' LIMIT 1")
row = cursor.fetchone()
if row:
    print("\nFull turn_stored payload:")
    print(json.dumps(json.loads(row[0]), indent=2))

# Check if there's a blob_store or turns table
for table in ['blob_store', 'turns', 'content_store', 'cas_store']:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
    if cursor.fetchone():
        print(f"\nFound table: {table}")
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        print(cursor.fetchone())
