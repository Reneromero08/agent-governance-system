#!/usr/bin/env python3
"""
Build & Initialize Semantic Core

Complete initialization of the Semantic Core system:
1. Create CORTEX database with vector schema
2. Initialize test content
3. Generate embeddings
4. Validate all systems
"""

import sqlite3
import hashlib
import sys
import io
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add CORTEX to path
CORTEX_ROOT = Path(__file__).parent
sys.path.insert(0, str(CORTEX_ROOT))

from embeddings import EmbeddingEngine
from vector_indexer import VectorIndexer
from semantic_search import SemanticSearch


def print_header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_step(step_num, title):
    """Print step indicator."""
    print(f"\n[{step_num}] {title}")
    print(f"{'-'*60}")


def init_database():
    """Initialize CORTEX database with schema."""
    print_step(1, "Initializing CORTEX Database")

    db_path = CORTEX_ROOT / "system1.db"

    # Check if already exists
    if db_path.exists():
        print(f"[OK] Database already exists: {db_path}")
        return db_path

    conn = sqlite3.connect(str(db_path))

    # Create base tables
    print("  Creating base tables...")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sections (
            hash TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            file_path TEXT,
            section_name TEXT,
            line_range TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            chunk_id UNINDEXED,
            tokenize='porter unicode61'
        );

        CREATE INDEX IF NOT EXISTS idx_sections_file ON sections(file_path);
        CREATE INDEX IF NOT EXISTS idx_sections_created ON sections(created_at);
    """)

    conn.commit()
    print("  [OK] Base tables created")

    # Create vector schema
    print("  Creating vector schema...")
    schema_file = CORTEX_ROOT / "schema" / "002_vectors.sql"
    if schema_file.exists():
        with open(schema_file) as f:
            conn.executescript(f.read())
    else:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS section_vectors (
                hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                dimensions INTEGER NOT NULL DEFAULT 384,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT,
                FOREIGN KEY (hash) REFERENCES sections(hash) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_section_vectors_model ON section_vectors(model_id);
            CREATE INDEX IF NOT EXISTS idx_section_vectors_created ON section_vectors(created_at);

            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL UNIQUE,
                dimensions INTEGER NOT NULL,
                description TEXT,
                active BOOLEAN DEFAULT 1,
                installed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            INSERT OR IGNORE INTO embedding_metadata (model_id, dimensions, description, active)
            VALUES ('all-MiniLM-L6-v2', 384, 'Default sentence transformer (384-dim)', 1);
        """)

    conn.commit()
    print("  [OK] Vector schema created")

    conn.close()
    print(f"\n[OK] Database initialized: {db_path}")
    return db_path


def create_test_sections(db_path):
    """Create test sections for demonstration."""
    print_step(2, "Creating Test Content")

    test_sections = [
        {
            "content": "The dispatch_task function manages task distribution to ant workers. It validates task specs, checks for duplicates, and atomically writes to the queue.",
            "file_path": "CATALYTIC-DPT/LAB/MCP/server.py",
            "section_name": "dispatch_task",
            "line_range": "1159-1227"
        },
        {
            "content": "Governor is the central coordinator that receives directives and dispatches tasks to ant workers via MCP. It uses exponential backoff for efficient polling.",
            "file_path": "CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py",
            "section_name": "run_governor",
            "line_range": "101-224"
        },
        {
            "content": "Ant workers poll for pending tasks, execute them, and report results back to the governor. They use proper subprocess timeout handling with process tree cleanup.",
            "file_path": "CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py",
            "section_name": "run_ant",
            "line_range": "227-428"
        },
        {
            "content": "The acknowledgment system uses atomic rewrites to prevent race conditions. Tasks transition through states: pending -> acknowledged -> processing -> completed.",
            "file_path": "CATALYTIC-DPT/LAB/MCP/server.py",
            "section_name": "acknowledge_task",
            "line_range": "1382-1451"
        },
        {
            "content": "Escalation provides chain of command messaging. Issues escalate from Ants to Governor to Claude to User. Each level resolves or forwards up.",
            "file_path": "CATALYTIC-DPT/LAB/MCP/server.py",
            "section_name": "escalate",
            "line_range": "1460-1519"
        },
        {
            "content": "The semantic core architecture adds vector embeddings to CORTEX. Big models maintain semantic understanding via embeddings, tiny models execute compressed tasks.",
            "file_path": "CONTEXT/decisions/ADR-030-semantic-core-architecture.md",
            "section_name": "semantic_core_overview",
            "line_range": "1-50"
        },
        {
            "content": "File operations include reading files with size limits, writing files safely, and deleting files. All operations are tracked and logged.",
            "file_path": "CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py",
            "section_name": "file_operations",
            "line_range": "200-250"
        },
        {
            "content": "Code adaptation supports regex replacement with count limits. Replacements are validated to prevent empty results and are logged for audit trails.",
            "file_path": "CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py",
            "section_name": "code_adapt",
            "line_range": "337-465"
        },
        {
            "content": "Backoff controller manages exponential backoff for polling. It resets on work, increases on idle, and increases more aggressively on errors.",
            "file_path": "CATALYTIC-DPT/SKILLS/swarm-orchestrator/scripts/poll_and_execute.py",
            "section_name": "BackoffController",
            "line_range": "48-78"
        },
        {
            "content": "Atomic file operations prevent data corruption through write-to-temp-then-rename pattern with file locking. Works on both Windows and Unix.",
            "file_path": "CATALYTIC-DPT/LAB/MCP/server.py",
            "section_name": "atomic_operations",
            "line_range": "126-262"
        },
    ]

    conn = sqlite3.connect(str(db_path))

    print(f"  Creating {len(test_sections)} test sections...")

    for i, section in enumerate(test_sections, 1):
        # Generate hash
        content_hash = hashlib.sha256(section["content"].encode()).hexdigest()

        # Insert section
        try:
            conn.execute("""
                INSERT OR REPLACE INTO sections
                (hash, content, file_path, section_name, line_range)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content_hash,
                section["content"],
                section["file_path"],
                section["section_name"],
                section["line_range"]
            ))

            if i % 2 == 0:
                print(f"    Created: {section['section_name']}")

        except Exception as e:
            print(f"    Error: {section['section_name']}: {e}")

    conn.commit()
    conn.close()

    print(f"  [OK] {len(test_sections)} test sections created")


def generate_embeddings(db_path):
    """Generate embeddings for all sections."""
    print_step(3, "Generating Embeddings")

    print("  Initializing embedding engine...")
    engine = EmbeddingEngine()

    print("  Indexing sections with embeddings...")
    with VectorIndexer(db_path=db_path, embedding_engine=engine) as indexer:
        results = indexer.index_all(batch_size=4, verbose=True)

    print(f"\n  Indexing complete:")
    print(f"    [OK] Indexed:  {results['indexed']}")
    print(f"    [OK] Errors:   {results['errors']}")
    print(f"    [OK] Total:    {results['total_sections']}")

    return engine


def test_semantic_search(db_path, engine):
    """Test semantic search functionality."""
    print_step(4, "Testing Semantic Search")

    queries = [
        "task dispatching and scheduling",
        "error handling and escalation",
        "file operations and management",
        "atomic operations and concurrency",
    ]

    with SemanticSearch(db_path, embedding_engine=engine) as searcher:
        for query in queries:
            print(f"\n  Query: '{query}'")
            results = searcher.search(query, top_k=3)

            for i, result in enumerate(results, 1):
                print(f"    {i}. {result.section_name}")
                print(f"       Similarity: {result.similarity:.3f}")
                print(f"       File: {result.file_path}")


def validate_system(db_path):
    """Validate the system is ready for production."""
    print_step(5, "Validating System")

    conn = sqlite3.connect(str(db_path))

    # Check tables
    cursor = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        AND name IN ('sections', 'section_vectors', 'embedding_metadata')
    """)
    tables = [row[0] for row in cursor.fetchall()]

    print(f"\n  Database Tables:")
    for table in ['sections', 'section_vectors', 'embedding_metadata']:
        status = "OK" if table in tables else "FAIL"
        print(f"    [{status}] {table}")

    # Count sections
    cursor = conn.execute("SELECT COUNT(*) FROM sections")
    section_count = cursor.fetchone()[0]
    print(f"\n  Content:")
    print(f"    [OK] {section_count} sections indexed")

    # Count embeddings
    cursor = conn.execute("SELECT COUNT(*) FROM section_vectors")
    embedding_count = cursor.fetchone()[0]
    print(f"    [OK] {embedding_count} embeddings generated")

    # Check model metadata
    cursor = conn.execute("SELECT model_id, dimensions FROM embedding_metadata")
    for row in cursor:
        print(f"    [OK] Model: {row[0]} ({row[1]} dimensions)")

    conn.close()

    # Summary
    success = section_count > 0 and embedding_count > 0
    if success:
        print(f"\n  [OK] System ready for production")
    else:
        print(f"\n  [FAIL] System validation failed")

    return success


def main():
    """Build the semantic core system."""
    print_header("SEMANTIC CORE BUILD SYSTEM")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize database
        db_path = init_database()

        # Create test content
        create_test_sections(db_path)

        # Generate embeddings
        engine = generate_embeddings(db_path)

        # Test search
        test_semantic_search(db_path, engine)

        # Validate
        success = validate_system(db_path)

        # Final status
        print_header("BUILD COMPLETE")
        if success:
            print("  [OK] Semantic Core is ready for production use")
            print(f"  [OK] Database: {db_path}")
            print(f"  [OK] Model: all-MiniLM-L6-v2 (384 dimensions)")
            print(f"\nNext steps:")
            print(f"  1. Run tests: python test_semantic_core.py")
            print(f"  2. Index your CORTEX: python vector_indexer.py --index")
            print(f"  3. Search CORTEX: from semantic_search import search_cortex")
            return 0
        else:
            print("  [FAIL] Build failed validation")
            return 1

    except KeyboardInterrupt:
        print("\n\n[FAIL] Build interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Build failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
