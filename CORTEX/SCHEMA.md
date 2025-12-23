# Cortex Schema

This document defines the data model for the Cortex index database. The Cortex is a shadow index of the repository that enables efficient navigation and governance queries without scanning the filesystem.

## Overview

The Cortex consists of two representations:

1. **SQLite Database** (`cortex.db`) - Primary storage, optimized for O(1) lookups and complex queries.
2. **JSON Index** (`cortex.json`) - Generated snapshot for tooling and CI checks; used as fallback.

Both representations store the same logical schema but with different physical layouts.

## SQLite Schema

### Tables

#### `metadata`

Stores cortex-level configuration and versioning.

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `key` | TEXT | PRIMARY KEY | Configuration key (e.g., `cortex_version`, `canon_version`, `generated_at`) |
| `value` | TEXT | | Configuration value |

**Required metadata keys:**
- `cortex_version` - Version of the Cortex schema (e.g., `1.1.0`)
- `canon_version` - Canon version at build time (e.g., `2.7.2`)
- `generated_at` - ISO 8601 timestamp of last build

#### `entities`

Stores indexed items (files, sections, summaries, etc.).

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `id` | TEXT | PRIMARY KEY | Unique entity identifier. Format: `type:path` or `type:path:anchor` for sections. Must be globally unique and normalized. |
| `type` | TEXT | NOT NULL | Entity type: `file`, `section`, `summary`, `glossary_term`, `skill`, `adr`, etc. |
| `title` | TEXT | | Human-readable title or heading. For sections, the markdown heading text. |
| `source_path` | TEXT | NOT NULL | Relative path to source file in repo. Used for lookups and provenance. |
| `last_modified` | REAL | | Unix timestamp of last modification. For generated entities, may be NULL. |

**Index patterns:**
- Files: `id = file:{rel_path}`
- Sections: `id = section:{rel_path}::{heading_anchor}::01`
- Example: `section:CANON/CONTRACT.md::canon-contract::01`

#### `tags`

Stores many-to-many relationships between entities and tags for classification and filtering.

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `entity_id` | TEXT | NOT NULL, FOREIGN KEY | Reference to `entities.id` |
| `tag` | TEXT | NOT NULL | Tag name (e.g., `governance`, `P0`, `deprecated`, `ADR`) |
| Primary Key | (entity_id, tag) | | Prevents duplicate tags per entity |

### Indexes

Optimizations for common query patterns:

- `idx_entities_type` on `entities(type)` - Fast filtering by entity type
- `idx_entities_path` on `entities(source_path)` - Fast lookups by file path
- `idx_tags_tag` on `tags(tag)` - Fast tag-based searches

### Queries

**Find all sections in a file:**
```sql
SELECT * FROM entities
WHERE type = 'section' AND source_path = 'CANON/CONTRACT.md'
ORDER BY id;
```

**Find all ADRs tagged "governance":**
```sql
SELECT e.* FROM entities e
JOIN tags t ON e.id = t.entity_id
WHERE e.type = 'adr' AND t.tag = 'governance';
```

**Search for entities by title pattern:**
```sql
SELECT * FROM entities
WHERE title LIKE '%context%'
ORDER BY source_path;
```

## JSON Schema

The `cortex.json` file follows a structured format for CI checks and tooling.

```json
{
  "cortex_version": "1.1.0",
  "canon_version": "2.7.2",
  "generated_at": "2025-12-23T10:00:00Z",
  "entities": [
    {
      "id": "file:CANON/CONTRACT.md",
      "type": "file",
      "title": "Canon Contract",
      "tags": ["canon", "governance"],
      "paths": {
        "absolute": "/path/to/repo/CANON/CONTRACT.md",
        "relative": "CANON/CONTRACT.md"
      }
    }
  ]
}
```

### Fields

- `cortex_version` (string, required) - Schema version
- `canon_version` (string, required) - Canon version at build time
- `generated_at` (string, required) - ISO 8601 timestamp
- `entities` (array, required) - Array of entity objects
  - `id` (string, required) - Unique identifier
  - `type` (string, required) - Entity type
  - `title` (string, optional) - Human-readable title
  - `tags` (array, optional) - Classification tags
  - `paths` (object, optional) - Path information (varies by type)

## Entity Types

### Standard Entity Types

| Type | Description | Example ID | Can Have Sections |
|------|-------------|------------|-------------------|
| `file` | Repository file | `file:CANON/CONTRACT.md` | Yes (markdown) |
| `section` | Markdown heading section | `section:CANON/CONTRACT.md::canon-contract::01` | No (leaf) |
| `adr` | Architecture Decision Record | `adr:ADR-015-logging-output-roots` | Yes |
| `skill` | Skill definition | `skill:artifact-escape-hatch` | Yes |
| `glossary_term` | Glossary entry | `glossary:invariant` | No |
| `code_symbol` | Function/class definition | `symbol:packer.py:write_lite_indexes` | No |
| `summary` | AI-generated summary | `summary:section:CANON/CONTRACT.md::rule-1::01` | No |

Entity types are extensible; new types can be added as needed.

## Section ID Format

Sections are identified using a three-part normalized identifier:

```
section:{rel_path}::{anchor}::{part}
```

- `rel_path` - Relative path to markdown file (normalized: lowercase, underscores, no spaces)
- `anchor` - Normalized heading text (lowercase, hyphens, URL-safe)
- `part` - Two-digit partition number for multi-part sections (always `01` for single-part)

**Examples:**
- `section:CANON/CONTRACT.md::non-negotiable-rules::01`
- `section:AGENTS.md::2-authority-gradient::01`

### Heading Extraction Rules

1. Extract markdown headings (`#`, `##`, `###`, etc.)
2. Ignore headings inside fenced code blocks (` ``` `)
3. Normalize to: lowercase, spaces → hyphens, remove special chars
4. Combine with file path for uniqueness

## Determinism and Stability

The schema is designed to be deterministic:

- `id` fields must be stable across builds (content-based, not timestamp-based)
- `source_path` is always relative and platform-normalized (forward slashes)
- `last_modified` is optional and not used for identity
- Generated metadata (`generated_at`) is the only time-dependent field

## Versioning

Schema versions follow semantic versioning:

- **Major version** (e.g., `2.x.x`) - Breaking changes (incompatible with older clients)
- **Minor version** (e.g., `1.2.x`) - New fields/tables (backward compatible)
- **Patch version** (e.g., `1.1.1`) - Bug fixes (fully compatible)

Current version: `1.1.0` (supports files, sections, tags, and basic metadata)

## Building the Index

A single CLI command builds the DB from a repository checkout:

```bash
python CORTEX/cortex.build.py
```

The builder:
1. Scans the repo for markdown, code, and configuration files
2. Parses headings and symbols
3. Generates stable IDs and normalizes paths
4. Writes SQLite DB and JSON snapshot
5. Validates against this schema

## Extending the Schema

To add new entity types or fields:

1. Update `schema.sql` with new table/column definitions
2. Update `cortex.schema.json` with new properties
3. Increment schema version (minor if backward-compatible, major if breaking)
4. Update `cortex.build.py` to populate new fields
5. Document in this file

Example: Adding a new `summaries` table:

```sql
CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL UNIQUE,
    summary_text TEXT,
    freshness_timestamp REAL,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);
```

## Future Work (C3)

The Cortex schema is designed to support future summarization layer:

- Reserve `summaries` table structure (see extending section)
- Define summary freshness policy and max length
- Extend `query.py` with `find()`, `get()`, `neighbors()` methods for navigation without file I/O
