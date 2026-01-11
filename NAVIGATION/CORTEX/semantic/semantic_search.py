#!/usr/bin/env python3
"""
CORTEX Semantic Search

DEPRECATED: system1.db is deprecated. Use the cassette network for search:
    from NAVIGATION.CORTEX.semantic.query import CortexQuery
    cq = CortexQuery()
    results = cq.search("your query")

Or use MCP tool: cassette_network_query

This file is kept for backward compatibility with tests.
"""

import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .embeddings import EmbeddingEngine

# TokenReceipt integration (Phase 5.2.7)
try:
    from CAPABILITY.PRIMITIVES.token_receipt import (
        TokenReceipt,
        TokenizerInfo,
        QueryMetadata,
        get_default_tokenizer,
        count_tokens,
        hash_query,
    )
    TOKEN_RECEIPT_AVAILABLE = True
except ImportError:
    TOKEN_RECEIPT_AVAILABLE = False


@dataclass
class SearchResult:
    """Semantic search result."""
    hash: str
    content: str
    similarity: float
    file_path: Optional[str] = None
    section_name: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None


@dataclass
class SearchResponse:
    """Search response with results and token receipt."""
    results: List[SearchResult]
    receipt: Optional['TokenReceipt'] = None

    def __iter__(self):
        """Allow iteration over results for backwards compatibility."""
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


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

        # TokenReceipt support
        self._tokenizer = get_default_tokenizer() if TOKEN_RECEIPT_AVAILABLE else None
        self._corpus_token_cache: Optional[int] = None

    def _get_corpus_tokens(self) -> int:
        """Get total tokens in corpus for baseline calculation.

        Returns:
            Total token count across all indexed content
        """
        if self._corpus_token_cache is not None:
            return self._corpus_token_cache

        if not TOKEN_RECEIPT_AVAILABLE:
            return 0

        cursor = self.conn.execute("""
            SELECT content FROM chunks_fts
        """)

        total_tokens = 0
        for row in cursor:
            if row['content']:
                total_tokens += count_tokens(row['content'], self._tokenizer)

        self._corpus_token_cache = total_tokens
        return total_tokens

    def _get_corpus_anchor(self) -> Optional[str]:
        """Get SHA-256 hash of corpus state for reproducibility."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count FROM section_vectors
        """)
        count = cursor.fetchone()['count']

        # Simple anchor: hash of vector count + db path
        anchor_data = f"{count}:{self.db_path}"
        return hashlib.sha256(anchor_data.encode()).hexdigest()

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        emit_receipt: bool = True,
        session_id: Optional[str] = None,
    ) -> SearchResponse:
        """Search for semantically similar content.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            emit_receipt: Whether to emit TokenReceipt (default True)
            session_id: Optional session ID for receipt aggregation

        Returns:
            SearchResponse with results and optional TokenReceipt
        """
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)

        # Load all embeddings from database
        cursor = self.conn.execute("""
            SELECT sv.hash, sv.embedding, fts.content, f.path as file_path, c.chunk_index
            FROM section_vectors sv
            LEFT JOIN chunks c ON sv.hash = c.chunk_hash
            LEFT JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
            LEFT JOIN files f ON c.file_id = f.file_id
        """)

        results = []
        total_scanned = 0
        for row in cursor:
            total_scanned += 1
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
                results.append(SearchResult(
                    hash=row['hash'],
                    content=row['content'] if row['content'] else "",
                    similarity=float(similarity),
                    file_path=row['file_path'] if row['file_path'] else None,
                    section_name=f"Chunk {row['chunk_index']}" if row['chunk_index'] is not None else None,
                    line_range=None
                ))

        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = results[:top_k]

        # Emit TokenReceipt
        receipt = None
        if emit_receipt and TOKEN_RECEIPT_AVAILABLE:
            # Count output tokens (sum of result content)
            output_tokens = sum(
                count_tokens(r.content, self._tokenizer)
                for r in final_results
            )

            receipt = TokenReceipt(
                operation="semantic_query",
                tokens_out=output_tokens,
                tokenizer=self._tokenizer,
                tokens_in=count_tokens(query, self._tokenizer),
                baseline_equiv=self._get_corpus_tokens(),
                baseline_method="sum_corpus_tokens",
                corpus_anchor=self._get_corpus_anchor(),
                session_id=session_id,
                query_metadata=QueryMetadata(
                    query_hash=hash_query(query),
                    results_count=len(final_results),
                    threshold_used=min_similarity,
                    top_k=top_k,
                    index_sections_count=total_scanned,
                ),
            )

        return SearchResponse(results=final_results, receipt=receipt)

    def search_batch(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        batch_size: int = 1000,
        emit_receipt: bool = True,
        session_id: Optional[str] = None,
    ) -> SearchResponse:
        """More efficient batch search for large databases.

        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            batch_size: Number of embeddings to process at once
            emit_receipt: Whether to emit TokenReceipt (default True)
            session_id: Optional session ID for receipt aggregation

        Returns:
            SearchResponse with results and optional TokenReceipt
        """
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)

        # Load embeddings in batches
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count FROM section_vectors
        """)
        total_count = cursor.fetchone()['count']

        if total_count == 0:
            return SearchResponse(results=[], receipt=None)

        all_results = []
        offset = 0

        while offset < total_count:
            cursor = self.conn.execute("""
                SELECT sv.hash, sv.embedding, fts.content, f.path as file_path, c.chunk_index
                FROM section_vectors sv
                LEFT JOIN chunks c ON sv.hash = c.chunk_hash
                LEFT JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                LEFT JOIN files f ON c.file_id = f.file_id
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
        final_results = all_results[:top_k]

        # Emit TokenReceipt
        receipt = None
        if emit_receipt and TOKEN_RECEIPT_AVAILABLE:
            output_tokens = sum(
                count_tokens(r.content, self._tokenizer)
                for r in final_results
            )

            receipt = TokenReceipt(
                operation="semantic_query",
                tokens_out=output_tokens,
                tokenizer=self._tokenizer,
                tokens_in=count_tokens(query, self._tokenizer),
                baseline_equiv=self._get_corpus_tokens(),
                baseline_method="sum_corpus_tokens",
                corpus_anchor=self._get_corpus_anchor(),
                session_id=session_id,
                query_metadata=QueryMetadata(
                    query_hash=hash_query(query),
                    results_count=len(final_results),
                    threshold_used=min_similarity,
                    top_k=top_k,
                    index_sections_count=total_count,
                ),
            )

        return SearchResponse(results=final_results, receipt=receipt)

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
                SELECT sv.hash, sv.embedding, fts.content, f.path as file_path, c.chunk_index
                FROM section_vectors sv
                LEFT JOIN chunks c ON sv.hash = c.chunk_hash
                LEFT JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                LEFT JOIN files f ON c.file_id = f.file_id
                WHERE sv.hash != ?
            """, (content_hash,))
        else:
            cursor = self.conn.execute("""
                SELECT sv.hash, sv.embedding, fts.content, f.path as file_path, c.chunk_index
                FROM section_vectors sv
                LEFT JOIN chunks c ON sv.hash = c.chunk_hash
                LEFT JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                LEFT JOIN files f ON c.file_id = f.file_id
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
    db_path: Path = None,
    top_k: int = 10,
    emit_receipt: bool = True,
) -> SearchResponse:
    """Convenience function for semantic search.

    DEPRECATED: Use cassette network instead:
        from NAVIGATION.CORTEX.semantic.query import CortexQuery
        cq = CortexQuery()
        results = cq.search(query)

    Args:
        query: Query text
        db_path: Path to database (deprecated, ignored)
        top_k: Number of results
        emit_receipt: Whether to emit TokenReceipt

    Returns:
        SearchResponse with empty results (system1.db deprecated)
    """
    import warnings
    warnings.warn(
        "search_cortex is deprecated. Use CortexQuery from "
        "NAVIGATION.CORTEX.semantic.query instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Return empty response - system1.db no longer exists
    return SearchResponse(results=[], receipt=None)


if __name__ == "__main__":
    print("SemanticSearch is deprecated.")
    print("Use the cassette network for search:")
    print("  from NAVIGATION.CORTEX.semantic.query import CortexQuery")
    print("  cq = CortexQuery()")
    print("  results = cq.search('your query')")
    print()
    print("Or use MCP tool: cassette_network_query")
