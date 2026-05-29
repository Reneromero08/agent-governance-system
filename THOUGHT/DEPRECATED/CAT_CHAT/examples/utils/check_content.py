import sys
import sqlite3
import json

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if content_hash column has values and if payload_json contains full content
cursor.execute("""
    SELECT event_id, content_hash, payload_json 
    FROM session_events 
    WHERE event_type = 'turn_stored' 
    ORDER BY sequence_num DESC 
    LIMIT 3
""")
rows = cursor.fetchall()

for event_id, content_hash_col, payload_json in rows:
    payload = json.loads(payload_json)
    print(f"Event: {event_id}")
    print(f"  content_hash (column): {content_hash_col[:20]}..." if content_hash_col else "  content_hash (column): NULL")
    print(f"  content_hash (payload): {payload.get('content_hash', 'N/A')[:20]}...")
    print(f"  Payload keys: {list(payload.keys())}")
    # Check if user_query or assistant_response exist
    if 'user_query' in payload:
        print(f"  user_query: {payload['user_query'][:100]}...")
    if 'assistant_response' in payload:
        print(f"  assistant_response: {payload['assistant_response'][:200]}...")
    print()
