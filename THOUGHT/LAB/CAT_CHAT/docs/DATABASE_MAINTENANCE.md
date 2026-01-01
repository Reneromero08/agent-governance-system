<!-- CONTENT_HASH: 2276208121ecdde7cbe879124e7abf9e727f5d2bae99dbdf707a8624f6858619 -->

# CAT_CHAT Database Maintenance

## Current State

The `cat_chat_index.db` database contains:
- **FTS5 indexes** for full-text search on CAT_CHAT content
- **Indexing information** (database best practices guide)
- **File metadata** (paths, hashes, sizes)

## What is FTS5?

FTS5 (Full-Text Search 5) is SQLite's built-in search engine:
- Creates inverted word indexes for fast text search
- Supports `MATCH` queries: `WHERE content MATCH 'catalytic'`
- Returns ranked results with snippets
- 100x+ faster than `LIKE '%word%'` for text searches

## Maintenance Status

⚠️ **The database is currently STATIC** - it was created once and is not automatically updated.

**What this means:**
- New files added to CAT_CHAT won't be indexed
- Modified files won't update the index
- Deleted files will still appear in search results

## How to Re-index (Manual)

If you need to rebuild the index:

```bash
# Option 1: Use the archived indexing scripts
cd THOUGHT/LAB/CAT_CHAT/archive
python index_cat_chat.py  # Rebuild file index
python index_content.py   # Rebuild FTS5 index

# Option 2: Delete and recreate
rm cat_chat_index.db
python initialize_cat_chat_db.py
```

## Future: Automated Maintenance

To make this automatic, you could:

1. **File watcher** - Monitor CAT_CHAT directory, update DB on changes
2. **Pre-commit hook** - Re-index before commits
3. **CI/CD integration** - Rebuild index as part of build process
4. **MCP tool** - Add `mcp_ags-mcp-server_reindex_cat_chat` tool

## Querying the Database

The database is accessible via:
- **MCP tools:** `cassette_network_query(query="your search")`
- **Direct SQL:** `sqlite3 cat_chat_index.db "SELECT * FROM content_fts WHERE content_fts MATCH 'catalytic'"`
- **Python:** See `archive/legacy/` for demo scripts

## Schema

```sql
-- Files table
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    rel_path TEXT,
    size INTEGER,
    modified TEXT,
    content_hash TEXT,
    extension TEXT,
    is_duplicate BOOLEAN DEFAULT 0
);

-- Content table
CREATE TABLE content (
    file_id INTEGER,
    content TEXT,
    FOREIGN KEY(file_id) REFERENCES files(id)
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE content_fts USING fts5(file_id, content);

-- Indexing information (database best practices)
CREATE TABLE indexing_info (
    id INTEGER PRIMARY KEY,
    storage_type TEXT NOT NULL,
    column_type TEXT NOT NULL,
    normal_indexes_content TEXT NOT NULL,
    how_to_index_content TEXT NOT NULL,
    example TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE indexing_info_fts USING fts5(
    storage_type, 
    column_type, 
    normal_indexes_content, 
    how_to_index_content, 
    example
);
```

## Token Savings

Using the DB instead of pasting file contents:
- **Before:** 50,000 tokens (full file reads)
- **After:** 8,800 tokens (semantic search queries)
- **Savings:** 82.4% token reduction

See `THOUGHT/LAB/CAT_CHAT/archive/CAT_CHAT_INTEGRATION_REPORT.md` for full analysis.
