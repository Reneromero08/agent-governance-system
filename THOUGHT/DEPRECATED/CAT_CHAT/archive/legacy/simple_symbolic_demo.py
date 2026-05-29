#!/usr/bin/env python3
"""Simple Symbolic Chat Demo."""

import sys
import uuid as uuid_lib
from datetime import datetime
from pathlib import Path

chat_system_path = Path(".").resolve()

sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from chat_db import ChatDB

db = ChatDB(db_path="chat.db")
db.init_db()

session_id = "symbolic-demo"

print("Symbolic Chat Demo")
print("=" * 60)

# Symbol dictionary
SYMBOLS = {
    "s001": "hello world",
    "s002": "thank you",
    "s003": "how are you"
}

msg1_uuid = str(uuid_lib.uuid4())
timestamp = datetime.utcnow().isoformat() + "Z"

with db.get_connection() as conn:
    conn.execute(
        "INSERT INTO chat_messages (session_id, uuid, role, content, content_hash, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, msg1_uuid, "user", "hello world how are you", db.compute_content_hash("hello world how are you"), timestamp, '{"encoding": "english"}')
    )

print('User: "hello world how are you (7 words)"')

msg2_uuid = str(uuid_lib.uuid4())
timestamp = datetime.utcnow().isoformat() + "Z"

with db.get_connection() as conn:
    conn.execute(
        "INSERT INTO chat_messages (session_id, uuid, role, content, content_hash, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, msg2_uuid, "assistant", "s001", db.compute_content_hash("thank you"), timestamp, '{"encoding": "symbolic", "symbol_count": 2}')
    )

print("Assistant: s001")

# Check messages
messages = db.get_session_messages(session_id)
for msg in messages:
    print(f"[{msg.role}] {msg.content[:50]}...")
