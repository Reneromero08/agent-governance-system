# CAT_CHAT Write Isolation (Phase B.3)

**Purpose:** Document the isolation model where CAT_CHAT reads from main cassettes but writes only to its sandbox database.

---

## Core Principle

CAT_CHAT is a **consumer** of main cassette content, not a contributor.

- **Reads:** Main cassettes at `NAVIGATION/CORTEX/cassettes/*.db`
- **Writes:** Local sandbox at `THOUGHT/LAB/CAT_CHAT/_generated/cat_chat.db`

All CAT_CHAT-specific state (sessions, events, local symbols, expansion cache) stays in the LAB.

---

## Enforcement Points

### 1. CassetteClient is Read-Only

The `CassetteClient` class in `catalytic_chat/cassette_client.py` has **no write methods by design**.

```python
# This assertion is enforced at module load:
assert not any(m.startswith('write') or m.startswith('save') or m.startswith('create')
               for m in dir(CassetteClient) if not m.startswith('_'))
```

Available methods (all read-only):
- `query()` - Search cassettes
- `resolve_symbol()` - Resolve @SYMBOL references
- `get_network_status()` - Get cassette network info
- `normalize_fts_query()` - Query normalization (static)

### 2. MCP Tool Allowlist

`ChatToolExecutor.ALLOWED_TOOLS` in `catalytic_chat/mcp_integration.py` explicitly excludes write operations:

**Allowed (Read-Only):**
- `cassette_network_query`
- `cortex_query`
- `semantic_search`
- `semantic_stats`
- `context_search`
- `context_review`
- `canon_read`
- `codebook_lookup`

**Explicitly Excluded:**
- `memory_save` - Would write to resident.db
- `memory_promote` - Would modify cassette state
- `session_start` (resident) - Uses resident.db, not CAT_CHAT sessions

### 3. Path Centralization

All write paths in CAT_CHAT are centralized in `catalytic_chat/paths.py`:

```python
def get_generated_dir(repo_root):
    """All writes go here: THOUGHT/LAB/CAT_CHAT/_generated/"""
    return repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "_generated"

def get_cat_chat_db(repo_root):
    """Single database for all CAT_CHAT state."""
    return get_generated_dir(repo_root) / "cat_chat.db"
```

No module in CAT_CHAT writes to paths outside `_generated/`.

---

## What Stays Local

| Data Type | Location | Why Local |
|-----------|----------|-----------|
| Sessions | `_generated/cat_chat.db` | Per-instance state |
| Session Events | `_generated/cat_chat.db` | Hash-chained logs |
| Working Set | `_generated/cat_chat.db` | Active context |
| Pointer Set | `_generated/cat_chat.db` | Offloaded refs |
| Local Symbols | `_generated/cat_chat.db` | Sandbox definitions |
| Expansion Cache | `_generated/cat_chat.db` | Runtime cache |
| Cassette Jobs/Steps | `_generated/cat_chat.db` | Execution logs |
| Cassette Receipts | `_generated/cat_chat.db` | Proof logs |

---

## What is Read from Main Cassettes

| Cassette | Content | Used For |
|----------|---------|----------|
| canon.db | LAW/CANON documents | Constitutional rules |
| governance.db | CONTEXT decisions | Governance context |
| capability.db | CAPABILITY code | Tool definitions |
| thought.db | THOUGHT research | Research content |
| navigation.db | NAVIGATION maps | System structure |
| direction.db | DIRECTION roadmaps | Planning context |
| memory.db | MEMORY archives | Historical context |
| inbox.db | INBOX staging | Temporary content |
| resident.db | AI memories | Agent context |

---

## Guarantees

1. **No Cassette Pollution:** CAT_CHAT never modifies main cassettes
2. **Session Isolation:** Each CAT_CHAT instance has independent state
3. **Deterministic Replay:** Local state can be exported/imported without affecting main cassettes
4. **Safe Experimentation:** LAB isolation allows rapid iteration without risk

---

## Verification

To verify isolation is maintained:

```python
# 1. CassetteClient has no write methods
from catalytic_chat.cassette_client import CassetteClient
write_methods = [m for m in dir(CassetteClient)
                 if not m.startswith('_')
                 and ('write' in m or 'save' in m or 'create' in m)]
assert write_methods == [], f"Found write methods: {write_methods}"

# 2. All paths route to _generated
from catalytic_chat.paths import get_cat_chat_db
db_path = get_cat_chat_db()
assert "_generated" in str(db_path), f"DB not in _generated: {db_path}"
```
