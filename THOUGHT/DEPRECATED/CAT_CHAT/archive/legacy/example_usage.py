#!/usr/bin/env python3
"""
Example: Using Catalytic Chat System in repository.

Stores data locally in CHAT_SYSTEM directory.
"""

import sys
from pathlib import Path

# Chat system is now in current directory
chat_system_path = Path(__file__).parent
sys.path.insert(0, str(chat_system_path))

from chat_db import ChatDB
from message_writer import MessageWriter

# Local configuration in CHAT_SYSTEM directory
chat_data_dir = chat_system_path
db_path = chat_data_dir / "chat.db"

# Initialize
db = ChatDB(db_path=db_path)
db.init_db()

writer = MessageWriter(db=db, claude_dir=chat_data_dir)

# Write messages
user_uuid = writer.write_message(
    session_id="my-session",
    role="user",
    content="How do I use this chat system?"
)

assistant_uuid = writer.write_message(
    session_id="my-session",
    role="assistant",
    content="Initialize MessageWriter with repo paths, then use write_message()",
    parent_uuid=user_uuid
)

print(f"User UUID: {user_uuid}")
print(f"Assistant UUID: {assistant_uuid}")
print(f"Data stored in: {chat_data_dir}")
print(f"Exports will be in: {chat_data_dir / 'projects'}")

# Query messages
messages = db.get_session_messages("my-session")
print(f"\nSession has {len(messages)} messages:")
for msg in messages:
    print(f"  [{msg.role}]: {msg.content[:50]}...")
