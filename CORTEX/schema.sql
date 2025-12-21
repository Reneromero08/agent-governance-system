-- Shadow Cortex SQLite Schema
-- v1.1.0

-- Metadata table for general cortex info
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Entities table for indexed items
-- We store the 'flat' fields here.
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT,
    source_path TEXT NOT NULL,
    last_modified REAL
);

-- Tags table for many-to-many relationship
CREATE TABLE IF NOT EXISTS tags (
    entity_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (entity_id, tag),
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

-- Indices for O(1) lookups
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_path ON entities(source_path);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
