#!/usr/bin/env python3
"""
CORTEX Vector Indexer

Extends CORTEX with vector embedding generation and indexing.
Part of the Semantic Core architecture (ADR-030).

Features:
- Batch embedding generation for efficiency
- Incremental updates (only process changed content)
- Progress tracking
- Database migration support
"""

import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .embeddings import EmbeddingEngine

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None


class VectorIndexer:
    """Index CORTEX sections with vector embeddings."""

    def __init__(
        self,
        db_path: Path = Path("NAVIGATION/CORTEX/db/system1.db"),
        embedding_engine: Optional[EmbeddingEngine] = None,
        writer: Optional[GuardedWriter] = None
    ):
        """Initialize vector indexer.

        Args:
            db_path: Path to CORTEX database
            embedding_engine: Optional pre-initialized embedding engine
        """
        self.db_path = db_path
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.writer = writer
        
        # Enforce usage of GuardedWriter for directory creation
        if self.writer is None:
             # Lazy init default writer if we are running in an env that supports it
            repo_root = Path(__file__).resolve().parents[3]
            self.writer = GuardedWriter(
                project_root=repo_root,
                durable_roots=[
                    "LAW/CONTRACTS/_runs",
                    "NAVIGATION/CORTEX/_generated",
                    "NAVIGATION/CORTEX/db"
                ]
            )
            self.writer.open_commit_gate()

        self.conn = None
        self._connect()
        self._init_schema()

    def _connect(self):
        """Connect to database."""
        if not self.db_path.exists():
            # Create parent directory if needed
            self.writer.mkdir_durable(str(self.db_path.parent))

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def _init_schema(self):
        """Initialize vector schema if not exists."""
        schema_file = Path("NAVIGATION/CORTEX/db/schema/002_vectors.sql")
        if schema_file.exists():
            with open(schema_file) as f:
                self.conn.executescript(f.read())
        else:
            # Fallback inline schema
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS section_vectors (
                    hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                    dimensions INTEGER NOT NULL DEFAULT 384,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT,
                    FOREIGN KEY (hash) REFERENCES chunks(chunk_hash) ON DELETE CASCADE
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
                VALUES ('all-MiniLM-L6-v2', 384, 'Default sentence transformer', 1);
            """)

        self.conn.commit()

    def index_all(
        self,
        batch_size: int = 32,
        force: bool = False,
        verbose: bool = True
    ) -> Dict:
        """Index all sections in CORTEX with embeddings.

        Args:
            batch_size: Number of sections to embed at once
            force: Re-index even if embedding exists
            verbose: Print progress messages

        Returns:
            Dictionary with stats (total_indexed, skipped, errors)
        """
        # Get sections to index
        if force:
            cursor = self.conn.execute("""
                SELECT c.chunk_hash as hash, fts.content
                FROM chunks c
                JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                WHERE fts.content IS NOT NULL AND fts.content != ''
            """)
        else:
            cursor = self.conn.execute("""
                SELECT c.chunk_hash as hash, fts.content
                FROM chunks c
                JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                LEFT JOIN section_vectors sv ON c.chunk_hash = sv.hash
                WHERE fts.content IS NOT NULL
                  AND fts.content != ''
                  AND sv.hash IS NULL
            """)

        sections = cursor.fetchall()
        total = len(sections)

        if verbose:
            print(f"Indexing {total} sections...")

        indexed = 0
        skipped = 0
        errors = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = sections[i:i + batch_size]

            # Extract texts and hashes
            texts = [row['content'] for row in batch]
            hashes = [row['hash'] for row in batch]

            try:
                # Generate embeddings
                embeddings = self.embedding_engine.embed_batch(texts, batch_size=batch_size)

                # Store embeddings
                for hash_val, embedding in zip(hashes, embeddings):
                    try:
                        blob = self.embedding_engine.serialize(embedding)
                        self.conn.execute("""
                            INSERT OR REPLACE INTO section_vectors
                            (hash, embedding, model_id, dimensions, created_at)
                            VALUES (?, ?, ?, ?, datetime('now'))
                        """, (
                            hash_val,
                            blob,
                            self.embedding_engine.MODEL_ID,
                            self.embedding_engine.DIMENSIONS
                        ))
                        indexed += 1
                    except Exception as e:
                        if verbose:
                            print(f"Error storing embedding for {hash_val[:8]}: {e}")
                        errors += 1

                self.conn.commit()

                if verbose and (i + batch_size) % (batch_size * 10) == 0:
                    print(f"  Progress: {i + batch_size}/{total}")

            except Exception as e:
                if verbose:
                    print(f"Batch error at {i}-{i+batch_size}: {e}")
                errors += len(batch)

        if verbose:
            print(f"Indexing complete: {indexed} indexed, {skipped} skipped, {errors} errors")

        return {
            'total_sections': total,
            'indexed': indexed,
            'skipped': skipped,
            'errors': errors
        }

    def index_section(self, content_hash: str, content: str) -> bool:
        """Index a single section.

        Args:
            content_hash: Hash of the section
            content: Text content

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_engine.embed(content)

            # Store
            blob = self.embedding_engine.serialize(embedding)
            self.conn.execute("""
                INSERT OR REPLACE INTO section_vectors
                (hash, embedding, model_id, dimensions, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (
                content_hash,
                blob,
                self.embedding_engine.MODEL_ID,
                self.embedding_engine.DIMENSIONS
            ))
            self.conn.commit()

            return True

        except Exception as e:
            print(f"Error indexing section {content_hash[:8]}: {e}")
            return False

    def delete_embedding(self, content_hash: str) -> bool:
        """Delete embedding for a section.

        Args:
            content_hash: Hash of the section

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.execute("""
            DELETE FROM section_vectors WHERE hash = ?
        """, (content_hash,))
        self.conn.commit()

        return cursor.rowcount > 0

    def get_stats(self) -> Dict:
        """Get indexing statistics.

        Returns:
            Dictionary with stats
        """
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_embeddings,
                model_id,
                MIN(created_at) as first_indexed,
                MAX(created_at) as last_indexed
            FROM section_vectors
            GROUP BY model_id
        """)

        models = []
        for row in cursor:
            models.append({
                'model_id': row['model_id'],
                'total_embeddings': row['total_embeddings'],
                'first_indexed': row['first_indexed'],
                'last_indexed': row['last_indexed']
            })

        # Count sections without embeddings
        cursor = self.conn.execute("""
            SELECT COUNT(*) as unindexed
            FROM chunks c
            JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
            LEFT JOIN section_vectors sv ON c.chunk_hash = sv.hash
            WHERE fts.content IS NOT NULL
              AND fts.content != ''
              AND sv.hash IS NULL
        """)
        unindexed = cursor.fetchone()['unindexed']

        return {
            'models': models,
            'total_embeddings': sum(m['total_embeddings'] for m in models),
            'unindexed_sections': unindexed
        }

    def verify_integrity(self, verbose: bool = True) -> Dict:
        """Verify embedding integrity.

        Args:
            verbose: Print detailed messages

        Returns:
            Dictionary with verification results
        """
        issues = []

        # Check for orphaned embeddings
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count
            FROM section_vectors sv
            LEFT JOIN chunks c ON sv.hash = c.chunk_hash
            WHERE c.chunk_hash IS NULL
        """)
        orphaned = cursor.fetchone()['count']
        if orphaned > 0:
            issues.append(f"{orphaned} orphaned embeddings (no matching section)")

        # Check for malformed embeddings
        cursor = self.conn.execute("SELECT hash, embedding FROM section_vectors")
        malformed = 0
        for row in cursor:
            try:
                self.embedding_engine.deserialize(row['embedding'])
            except Exception:
                malformed += 1

        if malformed > 0:
            issues.append(f"{malformed} malformed embeddings")

        if verbose:
            if issues:
                print("Integrity issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("No integrity issues found")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'orphaned_embeddings': orphaned,
            'malformed_embeddings': malformed
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CORTEX Vector Indexer")
    parser.add_argument("--index", action="store_true", help="Index all sections")
    parser.add_argument("--force", action="store_true", help="Re-index existing embeddings")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--verify", action="store_true", help="Verify integrity")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for indexing")

    args = parser.parse_args()

    # Initialize writer
    repo_root = Path(__file__).resolve().parents[3]
    writer = GuardedWriter(
        project_root=repo_root,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "NAVIGATION/CORTEX/db"
        ]
    )
    writer.open_commit_gate()

    with VectorIndexer(writer=writer) as indexer:
        if args.stats:
            stats = indexer.get_stats()
            print("\nIndexing Statistics:")
            print(f"  Total embeddings: {stats['total_embeddings']}")
            print(f"  Unindexed sections: {stats['unindexed_sections']}")
            print(f"\nModels:")
            for model in stats['models']:
                print(f"  - {model['model_id']}: {model['total_embeddings']} embeddings")
                print(f"    First indexed: {model['first_indexed']}")
                print(f"    Last indexed: {model['last_indexed']}")

        if args.verify:
            print("\nVerifying integrity...")
            results = indexer.verify_integrity(verbose=True)

        if args.index:
            print("\nIndexing sections...")
            results = indexer.index_all(
                batch_size=args.batch_size,
                force=args.force,
                verbose=True
            )

        if not (args.index or args.stats or args.verify):
            parser.print_help()
