# CAT_CHAT Message Storage Analysis

## Status: FIXED

The system is now truly catalytic - every message (user AND assistant) is stored
individually with its embedding for E-score based recall.

---

## The Fix (Applied)

### What Changed in `finalize_turn()`

The method now:
1. Stores **each message individually** in `_pointer_set` with embedding
2. Logs `user_message` and `assistant_response` events to session_events
3. Logs `turn_stored` event with full content for hydration
4. Uses single capsule connection (no sequence number races)

```python
# BEFORE: Only stored compressed turn blob
compression_result = self.compressor.compress_turn(turn)

# AFTER: Stores individual messages with embeddings
self._pointer_set.append(ContextItem(
    item_id=f"msg_user_{turn_id}",
    content=f"[User] {user_query}",
    embedding=user_embedding,  # <-- KEY: embedding for E-scoring
    item_type="user_message",
))
```

### Result

Now when you ask "Explain catalytic more":
- Messages about catalytic computing -> E=1.0 -> pulled into context
- Messages about weather -> E=0.0 -> stay in pointer set
- Messages about pizza -> E=0.14 -> stay in pointer set

**True catalytic recall: relevant messages surfaced by E-score.**

---

## Original Issues (Fixed)

### Issue 1: Messages Were Hidden in JSON Blobs

**Was:** Messages buried in `session_events.payload_json` with `event_type='turn_stored'`

**Now:** Individual `user_message` and `assistant_response` events, plus messages in `_pointer_set` with embeddings for E-scoring.

### Issue 2: E-Score Wasn't Applied to Messages

**Was:** E-scores computed only on documents and turn pointers (summaries)

**Now:** E-scores computed on individual messages via their embeddings in `_pointer_set`

### Issue 3: No Easy Debugging

**Fixed:** Created `catalytic_chat/debug.py` with:
- `CatChatDebugger` class
- `show_recent_turns(n)`, `show_messages()`, `show_e_score_history()`
- CLI: `python -m catalytic_chat.debug path/to/db --report`

---

## Debug Utilities

### From Code
```python
manager = AutoContextManager(...)

# After some turns...
manager.debug_show_state()    # Print context state
manager.debug_show_turns()    # Print compressed turns
messages = manager.debug_get_all_messages()  # Get all messages
manager.debug_report()        # Full diagnostic
```

### From CLI
```bash
python -m catalytic_chat.debug _generated/cat_chat.db --report
python -m catalytic_chat.debug _generated/cat_chat.db --turns 10
python -m catalytic_chat.debug _generated/cat_chat.db --messages
```

---

## Remaining Gaps (Future Work)

### Gap 1: Embeddings Not Persisted to DB

Currently embeddings are stored in memory (`_pointer_set`) but not in the database.
If the session is closed and reopened, embeddings would need to be recomputed.

### Gap 2: No Direct Messages Table

Messages are events, but a dedicated table would make queries easier:
```sql
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL
);
```

---

## How to Query Messages

### Via SQL (Current)
```sql
-- Get user_message and assistant_response events
SELECT event_type, payload_json
FROM session_events
WHERE event_type IN ('user_message', 'assistant_response')
  AND session_id = 'your_session_id'
ORDER BY sequence_num;

-- Get turn_stored events with full content
SELECT json_extract(payload_json, '$.user_query') as user,
       json_extract(payload_json, '$.assistant_response') as assistant
FROM session_events
WHERE event_type = 'turn_stored'
ORDER BY sequence_num;
```

### Via Python
```python
from catalytic_chat.debug import CatChatDebugger

db = CatChatDebugger("path/to/cat_chat.db")
messages = db.get_all_messages(session_id="...")
for m in messages:
    print(f"[{m.role}] {m.content}")
```
