-- Vector Embeddings Schema for Semantic Core
-- Created: 2025-12-28
-- Related: ADR-030 (Semantic Core + Translation Layer)

-- Section vectors table
CREATE TABLE IF NOT EXISTS section_vectors (
    hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER NOT NULL DEFAULT 384,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT,
    FOREIGN KEY (hash) REFERENCES sections(hash) ON DELETE CASCADE
);

-- Index for model lookups (for migration/versioning)
CREATE INDEX IF NOT EXISTS idx_section_vectors_model ON section_vectors(model_id);

-- Index for timestamp queries
CREATE INDEX IF NOT EXISTS idx_section_vectors_created ON section_vectors(created_at);

-- Metadata table for embedding model versioning
CREATE TABLE IF NOT EXISTS embedding_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    dimensions INTEGER NOT NULL,
    description TEXT,
    active BOOLEAN DEFAULT 1,
    installed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Insert default model metadata
INSERT OR IGNORE INTO embedding_metadata (model_id, dimensions, description, active)
VALUES ('all-MiniLM-L6-v2', 384, 'Default sentence transformer (384-dim, fast, good quality)', 1);

-- Stats view for monitoring
CREATE VIEW IF NOT EXISTS embedding_stats AS
SELECT
    COUNT(*) as total_embeddings,
    model_id,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM section_vectors
GROUP BY model_id;
