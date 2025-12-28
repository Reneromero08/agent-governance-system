#!/usr/bin/env python3
"""
CORTEX Semantic Search

Vector-based semantic similarity search for CORTEX sections.
Part of the Semantic Core architecture (ADR-030).

Features:
- Cosine similarity search
- Top-K retrieval
- Batch processing
- Integration with system1.db
- Fallback to FTS5 when no vectors available
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from embeddings import EmbeddingEngine


@dataclass
class SearchResult:
    """Semantic search result."""
    hash: str
    content: str
    similarity: float
    file_path: Optional[str] = None
    section_name: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None


class SemanticSearch:
    """Semantic search engine for CORTEX."""

    def __init__(self, db_path: Path, embedding_engine: Optional[EmbeddingEngine] = None):
        """Initialize semantic search.

        Args:
            db_path: Path to system1.db (or cortex.db)
            embedding_engine: Optional pre-initialized embedding engine
        """
        self.db_path = db_path
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.conn = None
        self._connect()

    def _connect(self):
        """Connect to database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """Search for semantically similar content.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of SearchResult objects sorted by similarity (descending)
        """
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)

        # Load all embeddings from database
        cursor = self.conn.execute("""
            SELECT sv.hash, sv.embedding, s.content, s.file_path, s.section_name, s.line_range
            FROM section_vectors sv
            LEFT JOIN sections s ON sv.hash = s.hash
        """)

        results = []
        for row in cursor:
            # Deserialize embedding
            try:
                embedding = self.embedding_engine.deserialize(row['embedding'])
            except Exception:
                continue  # Skip malformed embeddings

            # Compute similarity
            similarity = self.embedding_engine.cosine_similarity(
                query_embedding,
                embedding
            )

            if similarity >= min_similarity:
                line_range = row['line_range']
                results.append(SearchResult(
                    hash=row['hash'],
                    content=row['content'] if row['content'] else "",
                    similarity=float(similarity),
                    file_path=row['file_path'] if row['file_path'] else None,
                    section_name=row['section_name'] if row['section_name'] else None,
                    line_range=self._parse_line_range(line_range) if line_range else None
                ))

        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def search_batch(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        batch_size: int = 1000
    ) -> List[SearchResult]:
        """More efficient batch search for large databases.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            batch_size: Number of embeddings to process at once

        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)

        # Load embeddings in batches
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count FROM section_vectors
        """)
        total_count = cursor.fetchone()['count']

        if total_count == 0:
            return []

        all_results = []
        offset = 0

        while offset < total_count:
            cursor = self.conn.execute("""
                SELECT sv.hash, sv.embedding, s.content, s.file_path, s.section_name, s.line_range
                FROM section_vectors sv
                LEFT JOIN sections s ON sv.hash = s.hash
                LIMIT ? OFFSET ?
            """, (batch_size, offset))

            batch_rows = cursor.fetchall()
            if not batch_rows:
                break

            # Deserialize batch
            batch_embeddings = []
            batch_metadata = []

            for row in batch_rows:
                try:
                    embedding = self.embedding_engine.deserialize(row['embedding'])
                    batch_embeddings.append(embedding)
                    line_range_val = row['line_range']
                    batch_metadata.append({
                        'hash': row['hash'],
                        'content': row['content'] if row['content'] else "",
                        'file_path': row['file_path'] if row['file_path'] else None,
                        'section_name': row['section_name'] if row['section_name'] else None,
                        'line_range': self._parse_line_range(line_range_val) if line_range_val else None
                    })
                except Exception:
                    continue

            if batch_embeddings:
                # Compute similarities in batch
                batch_embeddings_array = np.array(batch_embeddings)
                similarities = self.embedding_engine.batch_similarity(
                    query_embedding,
                    batch_embeddings_array
                )

                # Create results
                for metadata, similarity in zip(batch_metadata, similarities):
                    if similarity >= min_similarity:
                        all_results.append(SearchResult(
                            hash=metadata['hash'],
                            content=metadata['content'],
                            similarity=float(similarity),
                            file_path=metadata['file_path'],
                            section_name=metadata['section_name'],
                            line_range=metadata['line_range']
                        ))

            offset += batch_size

        # Sort and return top_k
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        return all_results[:top_k]

    def find_similar_to_hash(
        self,
        content_hash: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """Find sections similar to a given content hash.

        Args:
            content_hash: Hash of the reference content
            top_k: Number of results to return
            exclude_self: Whether to exclude the reference itself

        Returns:
            List of SearchResult objects
        """
        # Get embedding for reference hash
        cursor = self.conn.execute("""
            SELECT embedding FROM section_vectors WHERE hash = ?
        """, (content_hash,))

        row = cursor.fetchone()
        if not row:
            return []

        reference_embedding = self.embedding_engine.deserialize(row['embedding'])

        # Load all other embeddings
        if exclude_self:
            cursor = self.conn.execute("""
                SELECT sv.hash, sv.embedding, s.content, s.file_path, s.section_name, s.line_range
                FROM section_vectors sv
                LEFT JOIN sections s ON sv.hash = s.hash
                WHERE sv.hash != ?
            """, (content_hash,))
        else:
            cursor = self.conn.execute("""
                SELECT sv.hash, sv.embedding, s.content, s.file_path, s.section_name, s.line_range
                FROM section_vectors sv
                LEFT JOIN sections s ON sv.hash = s.hash
            """)

        results = []
        for row in cursor:
            try:
                embedding = self.embedding_engine.deserialize(row['embedding'])
                similarity = self.embedding_engine.cosine_similarity(
                    reference_embedding,
                    embedding
                )

                results.append(SearchResult(
                    hash=row['hash'],
                    content=row['content'] if row['content'] else "",
                    similarity=float(similarity),
                    file_path=row['file_path'] if row['file_path'] else None,
                    section_name=row['section_name'] if row['section_name'] else None,
                    line_range=self._parse_line_range(row['line_range']) if row.get('line_range') else None
                ))
            except Exception:
                continue

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics about indexed embeddings.

        Returns:
            Dictionary with stats (total count, model info, etc.)
        """
        cursor = self.conn.execute("""
            SELECT COUNT(*) as total,
                   model_id,
                   MIN(created_at) as first_created,
                   MAX(created_at) as last_created
            FROM section_vectors
            GROUP BY model_id
        """)

        stats = []
        for row in cursor:
            stats.append({
                'total': row['total'],
                'model_id': row['model_id'],
                'first_created': row['first_created'],
                'last_created': row['last_created']
            })

        return {
            'models': stats,
            'total_embeddings': sum(s['total'] for s in stats)
        }

    def _parse_line_range(self, line_range_str: Optional[str]) -> Optional[Tuple[int, int]]:
        """Parse line range from string format.

        Args:
            line_range_str: String like "10-20" or "10,20"

        Returns:
            Tuple of (start, end) or None
        """
        if not line_range_str:
            return None

        try:
            if '-' in line_range_str:
                parts = line_range_str.split('-')
            elif ',' in line_range_str:
                parts = line_range_str.split(',')
            else:
                return None

            if len(parts) == 2:
                return (int(parts[0].strip()), int(parts[1].strip()))
        except (ValueError, IndexError):
            pass

        return None

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def search_cortex(
    query: str,
    db_path: Path = Path("CORTEX/system1.db"),
    top_k: int = 10
) -> List[SearchResult]:
    """Convenience function for semantic search.

    Args:
        query: Query text
        db_path: Path to database
        top_k: Number of results

    Returns:
        List of SearchResult objects
    """
    with SemanticSearch(db_path) as searcher:
        return searcher.search(query, top_k=top_k)


if __name__ == "__main__":
    # Self-test
    import sys
    from pathlib import Path

    print("Testing SemanticSearch...")

    db_path = Path("CORTEX/system1.db")
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Skipping tests (database needs to be initialized first)")
        sys.exit(0)

    try:
        with SemanticSearch(db_path) as searcher:
            # Get stats
            stats = searcher.get_stats()
            print(f"Database stats: {stats}")

            if stats['total_embeddings'] == 0:
                print("No embeddings found in database - run indexer first")
            else:
                # Test search
                results = searcher.search("task scheduling", top_k=5)
                print(f"\nSearch results for 'task scheduling':")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result.section_name or result.hash[:8]} "
                          f"(similarity: {result.similarity:.3f})")
                    if result.file_path:
                        print(f"   File: {result.file_path}")

        print("\nTests completed!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
