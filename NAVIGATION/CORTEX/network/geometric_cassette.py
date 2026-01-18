#!/usr/bin/env python3
"""
Geometric Cassette - Pure geometry for semantic queries.

I.1 Cassette Network Integration (FERAL_RESIDENT_QUANTUM_ROADMAP.md)

Uses E (Born rule) for relevance scoring instead of FTS/text similarity.
Embeddings happen ONLY at index time; queries are pure vector operations.

Q45 Validated Operations:
- E = <psi|phi> (Born rule inner product, r=0.977 with similarity)
- Analogy: d = b - a + c (king - man + woman = queen)
- Gating: E > threshold discriminates relevance

Acceptance Criteria:
- I.1.1: Geometric queries return same results as embedding queries
- I.1.2: Analogy queries work across cassettes
- I.1.3: Cross-cassette composition (combine results geometrically)
- I.1.4: E-gating discriminates relevance
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Handle both package and direct imports
try:
    from .cassette_protocol import DatabaseCassette
except ImportError:
    from cassette_protocol import DatabaseCassette

# Import geometric primitives
try:
    # Package path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from CAPABILITY.PRIMITIVES.geometric_reasoner import (
        GeometricState,
        GeometricOperations,
        GeometricReasoner,
    )
except ImportError:
    # Fallback for direct module testing
    GeometricState = None
    GeometricOperations = None
    GeometricReasoner = None


class GeometricCassette(DatabaseCassette):
    """
    Cassette using pure geometry for queries (Q45 validated).

    Inherits text query interface from DatabaseCassette.
    Adds geometric query capabilities with E (Born rule) scoring.

    Index built lazily on first geometric query.

    Usage:
        cassette = GeometricCassette(db_path, "thought")

        # Index documents (one-time, persisted)
        cassette.index_document("doc1", "quantum mechanics")
        cassette.index_document("doc2", "classical physics")

        # Query with text (initializes once, then pure geometry)
        results = cassette.query_text("quantum entanglement", k=5)

        # Query with geometric state (pure geometry, no embedding)
        state = cassette.reasoner.initialize("quantum")
        results = cassette.query_geometric(state, k=5)

        # Analogy query (Q45 validated)
        results = cassette.analogy_query("king", "queen", "man", k=5)
    """

    # Schema version for geometric index
    GEO_SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        db_path: Path,
        cassette_id: str,
        model_name: str = 'all-MiniLM-L6-v2',
        auto_build_from_chunks: bool = True,
        **kwargs
    ):
        """
        Initialize geometric cassette.

        Args:
            db_path: Path to SQLite database
            cassette_id: Unique cassette identifier
            model_name: Sentence transformer model for embeddings
            auto_build_from_chunks: If True, build index from chunks table on first access
            **kwargs: Additional arguments for DatabaseCassette
        """
        super().__init__(db_path, cassette_id)

        # Lazy initialization - reasoner created on first access
        self._reasoner: Optional[GeometricReasoner] = None
        self._model_name = model_name

        # Geometric index (lazy loaded)
        self._geo_index: Dict[str, GeometricState] = {}
        self._geo_metadata: Dict[str, Dict] = {}
        self._index_built = False
        self._auto_build = auto_build_from_chunks

        # Vectorized index for fast queries (built on first query)
        self._vector_matrix: Optional[np.ndarray] = None  # N x D matrix
        self._doc_ids: List[str] = []  # Ordered doc_id list matching matrix rows
        self._Df_array: Optional[np.ndarray] = None  # Df values for each doc

        # Add geometric capability
        if 'geometric' not in self.capabilities:
            self.capabilities.append('geometric')

        # Statistics
        self._geo_stats = {
            'index_size': 0,
            'queries_geometric': 0,
            'queries_text': 0,
            'analogy_queries': 0,
            'embedding_calls': 0,
            'geometric_ops': 0
        }

    @property
    def reasoner(self) -> 'GeometricReasoner':
        """Lazy-load GeometricReasoner on first access."""
        if self._reasoner is None:
            if GeometricReasoner is None:
                raise ImportError(
                    "GeometricReasoner not available. "
                    "Install: pip install sentence-transformers"
                )
            self._reasoner = GeometricReasoner(self._model_name)
        return self._reasoner

    # ========================================================================
    # Index Management
    # ========================================================================

    def _ensure_index(self):
        """Build index lazily on first access."""
        if not self._index_built:
            # Try to load from persistence first
            if not self._load_geometric_index():
                # If no persisted index, build from chunks
                if self._auto_build:
                    self._build_from_chunks()
            self._index_built = True
            # Build vectorized index for fast queries
            self._build_vector_matrix()

    def _build_vector_matrix(self):
        """Build vectorized index for fast numpy-based queries."""
        if not self._geo_index:
            return

        # Build ordered lists
        self._doc_ids = list(self._geo_index.keys())
        vectors = [self._geo_index[doc_id].vector for doc_id in self._doc_ids]
        Df_values = [self._geo_index[doc_id].Df for doc_id in self._doc_ids]

        # Stack into matrix (N x D)
        self._vector_matrix = np.vstack(vectors).astype(np.float32)
        self._Df_array = np.array(Df_values, dtype=np.float32)

    def _ensure_geo_table(self, conn: sqlite3.Connection):
        """Ensure geometric_index table exists."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS geometric_index (
                doc_id TEXT PRIMARY KEY,
                vector_blob BLOB NOT NULL,
                vector_hash TEXT NOT NULL,
                Df REAL NOT NULL,
                dim INTEGER NOT NULL,
                content_preview TEXT,
                metadata_json TEXT,
                indexed_at TEXT NOT NULL,
                schema_version TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_geometric_Df ON geometric_index(Df)
        """)
        conn.commit()

    def _build_from_chunks(self):
        """Build geometric index from existing chunks table."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            self._ensure_geo_table(conn)

            # Check if chunks table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
            )
            if not cursor.fetchone():
                conn.close()
                return

            # Load chunks - try different schema formats
            # Schema 1: chunks has content column directly
            # Schema 2: content in FTS content table (chunks_fts_content.c0)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts_content'"
            )
            has_fts_content = cursor.fetchone() is not None

            if has_fts_content:
                # Use FTS content table - join with chunks for chunk_hash
                cursor = conn.execute("""
                    SELECT c.chunk_hash, fts.c0 as content
                    FROM chunks c
                    JOIN chunks_fts_content fts ON fts.id = c.chunk_id
                    WHERE fts.c0 IS NOT NULL AND LENGTH(fts.c0) > 10
                    LIMIT 10000
                """)
            else:
                # Fallback: try content column directly
                cursor = conn.execute(
                    "SELECT chunk_hash, content FROM chunks LIMIT 10000"
                )
            chunks = cursor.fetchall()

            print(f"[GEOMETRIC] Building index from {len(chunks)} chunks...", file=sys.stderr)

            for chunk_hash, content in chunks:
                if content and len(content.strip()) > 0:
                    self.index_document(
                        doc_id=chunk_hash,
                        text=content,
                        metadata={'source': 'chunks_table'},
                        persist=False  # Batch persist at end
                    )

            # Persist all at once
            self._save_geometric_index()
            conn.close()

            print(f"[GEOMETRIC] Index built: {len(self._geo_index)} documents", file=sys.stderr)

        except sqlite3.Error as e:
            print(f"[GEOMETRIC] Error building index: {e}", file=sys.stderr)

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict = None,
        persist: bool = True
    ) -> Dict:
        """
        Index document to geometric manifold.

        This is the ONLY time embedding is called for this document.
        Subsequent queries use pure geometry.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata dict
            persist: If True, save to SQLite immediately

        Returns:
            Receipt with indexing details
        """
        state = self.reasoner.initialize(text)
        self._geo_stats['embedding_calls'] += 1

        self._geo_index[doc_id] = state
        self._geo_metadata[doc_id] = {
            'content': text[:500],  # Preview
            'full_hash': hashlib.sha256(text.encode()).hexdigest()[:16],
            **(metadata or {})
        }
        self._geo_stats['index_size'] = len(self._geo_index)

        if persist:
            self._persist_document(doc_id, state, self._geo_metadata[doc_id])

        return {
            'doc_id': doc_id,
            'vector_hash': state.receipt()['vector_hash'],
            'Df': state.Df,
            'dim': len(state.vector),
            'indexed_at': datetime.utcnow().isoformat()
        }

    def _persist_document(self, doc_id: str, state: GeometricState, metadata: Dict):
        """Persist single document to SQLite."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            self._ensure_geo_table(conn)

            conn.execute("""
                INSERT OR REPLACE INTO geometric_index
                (doc_id, vector_blob, vector_hash, Df, dim, content_preview, metadata_json, indexed_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                state.vector.tobytes(),
                state.receipt()['vector_hash'],
                state.Df,
                len(state.vector),
                metadata.get('content', '')[:500],
                json.dumps(metadata),
                datetime.utcnow().isoformat(),
                self.GEO_SCHEMA_VERSION
            ))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[GEOMETRIC] Persist error: {e}", file=sys.stderr)

    def _save_geometric_index(self):
        """Save entire geometric index to SQLite."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            self._ensure_geo_table(conn)

            for doc_id, state in self._geo_index.items():
                metadata = self._geo_metadata.get(doc_id, {})
                conn.execute("""
                    INSERT OR REPLACE INTO geometric_index
                    (doc_id, vector_blob, vector_hash, Df, dim, content_preview, metadata_json, indexed_at, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    state.vector.tobytes(),
                    state.receipt()['vector_hash'],
                    state.Df,
                    len(state.vector),
                    metadata.get('content', '')[:500],
                    json.dumps(metadata),
                    datetime.utcnow().isoformat(),
                    self.GEO_SCHEMA_VERSION
                ))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[GEOMETRIC] Save error: {e}", file=sys.stderr)

    def _load_geometric_index(self) -> bool:
        """Load geometric index from SQLite."""
        if not self.db_path.exists():
            return False

        try:
            conn = sqlite3.connect(str(self.db_path))

            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='geometric_index'"
            )
            if not cursor.fetchone():
                conn.close()
                return False

            # Get vector dimension from first row
            cursor = conn.execute("SELECT dim FROM geometric_index LIMIT 1")
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            dim = row[0]

            # Load all vectors
            cursor = conn.execute("""
                SELECT doc_id, vector_blob, metadata_json FROM geometric_index
            """)

            for doc_id, vector_blob, metadata_json in cursor:
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                state = GeometricState(vector=vector)
                self._geo_index[doc_id] = state

                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                self._geo_metadata[doc_id] = metadata

            conn.close()

            self._geo_stats['index_size'] = len(self._geo_index)
            print(f"[GEOMETRIC] Loaded {len(self._geo_index)} documents from index", file=sys.stderr)
            return len(self._geo_index) > 0

        except sqlite3.Error as e:
            print(f"[GEOMETRIC] Load error: {e}", file=sys.stderr)
            return False

    # ========================================================================
    # Query Methods
    # ========================================================================

    def query_geometric(
        self,
        query_state: 'GeometricState',
        k: int = 10
    ) -> List[Dict]:
        """
        Query using geometric state (pure geometry, no embedding).

        Uses E (Born rule inner product) for relevance scoring.
        Q44 validated: E correlates r=0.977 with semantic similarity.

        Optimized with vectorized numpy operations for <100ms latency.

        Args:
            query_state: GeometricState to query with
            k: Number of results to return

        Returns:
            List of result dicts sorted by E (highest first)
        """
        self._ensure_index()
        self._geo_stats['queries_geometric'] += 1

        if not self._geo_index:
            return []

        # Use vectorized computation if available (10x faster)
        if self._vector_matrix is not None:
            return self._query_geometric_vectorized(query_state, k)

        # Fallback to loop-based computation
        return self._query_geometric_loop(query_state, k)

    def _query_geometric_vectorized(
        self,
        query_state: 'GeometricState',
        k: int
    ) -> List[Dict]:
        """Vectorized query using numpy matrix operations."""
        # Single matrix-vector multiply: E_scores = M @ q (N dot products in one op)
        E_scores = self._vector_matrix @ query_state.vector
        self._geo_stats['geometric_ops'] += len(self._doc_ids)

        # Get top-k indices using argpartition (faster than full sort for k << N)
        if k < len(E_scores):
            # argpartition is O(N) vs argsort O(N log N)
            top_k_indices = np.argpartition(E_scores, -k)[-k:]
            # Sort only the top-k
            top_k_indices = top_k_indices[np.argsort(E_scores[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(E_scores)[::-1]

        # Build results for top-k only (avoid creating dicts for all N docs)
        results = []
        for idx in top_k_indices:
            doc_id = self._doc_ids[idx]
            results.append({
                'doc_id': doc_id,
                'E': float(E_scores[idx]),
                'content': self._geo_metadata.get(doc_id, {}).get('content', ''),
                'metadata': self._geo_metadata.get(doc_id, {}),
                'source': self.cassette_id,
                'Df': float(self._Df_array[idx])
            })

        return results

    def _query_geometric_loop(
        self,
        query_state: 'GeometricState',
        k: int
    ) -> List[Dict]:
        """Loop-based query (fallback when vectorized index not available)."""
        results = []
        for doc_id, doc_state in self._geo_index.items():
            E = query_state.E_with(doc_state)
            self._geo_stats['geometric_ops'] += 1

            results.append({
                'doc_id': doc_id,
                'E': float(E),
                'content': self._geo_metadata.get(doc_id, {}).get('content', ''),
                'metadata': self._geo_metadata.get(doc_id, {}),
                'source': self.cassette_id,
                'Df': doc_state.Df
            })

        results.sort(key=lambda x: x['E'], reverse=True)
        return results[:k]

    def query_text(self, query_text: str, k: int = 10) -> List[Dict]:
        """
        Query with text - initialize once, then pure geometry.

        Args:
            query_text: Text query
            k: Number of results

        Returns:
            List of results with E scores
        """
        self._geo_stats['queries_text'] += 1
        self._geo_stats['embedding_calls'] += 1

        query_state = self.reasoner.initialize(query_text)
        return self.query_geometric(query_state, k)

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """
        Standard query interface (required by DatabaseCassette).

        Delegates to query_text for geometric search.
        """
        results = self.query_text(query_text, top_k)

        # Convert to standard format expected by network hub
        return [
            {
                'content': r['content'],
                'score': r['E'],
                'metadata': {
                    'doc_id': r['doc_id'],
                    'Df': r['Df'],
                    'source': r['source'],
                    **r.get('metadata', {})
                }
            }
            for r in results
        ]

    def query_with_gate(
        self,
        query_state: 'GeometricState',
        k: int = 10,
        threshold: float = 0.5
    ) -> Dict:
        """
        Query with E-gate filtering.

        Only returns results where E > threshold.
        Q44 validated: E discriminates related vs unrelated.

        Args:
            query_state: GeometricState to query with
            k: Max results
            threshold: E threshold for gate

        Returns:
            Dict with results, gate status, and statistics
        """
        all_results = self.query_geometric(query_state, k * 2)
        gated = [r for r in all_results if r['E'] >= threshold]

        return {
            'results': gated[:k],
            'gate_open': len(gated) > 0,
            'mean_E': float(np.mean([r['E'] for r in gated])) if gated else 0.0,
            'max_E': float(max(r['E'] for r in gated)) if gated else 0.0,
            'filtered_count': len(all_results) - len(gated),
            'threshold': threshold
        }

    # ========================================================================
    # Analogy Queries (Q45 Validated)
    # ========================================================================

    def analogy_query(
        self,
        a: str,
        b: str,
        c: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Analogy query: a is to b as c is to ?

        Q45 validated formula: d = b - a + c
        Example: king - man + woman = queen

        Args:
            a: First term (e.g., "man")
            b: Second term (e.g., "king")
            c: Third term (e.g., "woman")
            k: Number of results

        Returns:
            List of candidates for d with E scores
        """
        self._geo_stats['analogy_queries'] += 1
        self._geo_stats['embedding_calls'] += 3

        state_a = self.reasoner.initialize(a)
        state_b = self.reasoner.initialize(b)
        state_c = self.reasoner.initialize(c)

        # d = b - a + c (pure geometry)
        diff = self.reasoner.subtract(state_b, state_a)
        query_state = self.reasoner.add(diff, state_c)
        self._geo_stats['geometric_ops'] += 2

        return self.query_geometric(query_state, k)

    def blend_query(
        self,
        concept1: str,
        concept2: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Blend query: find common hypernym of two concepts.

        Q45 validated: (c1 + c2) / sqrt(2) (superposition)
        Example: cat + dog = pet/animal

        Args:
            concept1: First concept
            concept2: Second concept
            k: Number of results

        Returns:
            List of blend candidates with E scores
        """
        self._geo_stats['embedding_calls'] += 2

        state1 = self.reasoner.initialize(concept1)
        state2 = self.reasoner.initialize(concept2)

        blended = self.reasoner.superpose(state1, state2)
        self._geo_stats['geometric_ops'] += 1

        return self.query_geometric(blended, k)

    def navigate_query(
        self,
        start: str,
        end: str,
        steps: int = 3,
        k: int = 5
    ) -> List[Dict]:
        """
        Navigate from start to end via geodesic interpolation.

        Q45 validated: slerp on unit sphere
        Example: hot -> warm -> cool -> cold

        Args:
            start: Starting concept
            end: Ending concept
            steps: Number of intermediate points
            k: Results per point

        Returns:
            List of path points with decoded concepts
        """
        self._geo_stats['embedding_calls'] += 2

        state_start = self.reasoner.initialize(start)
        state_end = self.reasoner.initialize(end)

        path = []
        for i in range(steps + 1):
            t = i / steps
            state = self.reasoner.interpolate(state_start, state_end, t)
            self._geo_stats['geometric_ops'] += 1

            results = self.query_geometric(state, k)

            path.append({
                't': t,
                'results': results,
                'Df': state.Df,
                'point_hash': state.receipt()['vector_hash']
            })

        return path

    # ========================================================================
    # Cross-Cassette Operations
    # ========================================================================

    def compose_with(
        self,
        other: 'GeometricCassette',
        query_state: 'GeometricState',
        operation: str = 'superpose',
        k: int = 10
    ) -> Dict:
        """
        Compose results from this cassette with another.

        Args:
            other: Another GeometricCassette
            query_state: Query state
            operation: 'superpose' or 'entangle'
            k: Results per cassette

        Returns:
            Dict with composed results and metadata
        """
        # Get results from both cassettes
        results_self = self.query_geometric(query_state, k)
        results_other = other.query_geometric(query_state, k)

        # If we have results from both, compose the top states
        if results_self and results_other:
            top_self = self._geo_index.get(results_self[0]['doc_id'])
            top_other = other._geo_index.get(results_other[0]['doc_id'])

            if top_self and top_other:
                if operation == 'superpose':
                    composed = self.reasoner.superpose(top_self, top_other)
                else:
                    composed = self.reasoner.entangle(top_self, top_other)

                self._geo_stats['geometric_ops'] += 1

                return {
                    'operation': operation,
                    'composed_state': composed.receipt(),
                    'results_self': results_self,
                    'results_other': results_other,
                    'E_self_other': top_self.E_with(top_other)
                }

        return {
            'operation': operation,
            'results_self': results_self,
            'results_other': results_other
        }

    # ========================================================================
    # Statistics & Metadata
    # ========================================================================

    def get_stats(self) -> Dict:
        """Return cassette statistics including geometric metrics."""
        base_stats = {
            'total_chunks': len(self._geo_index),
            'index_built': self._index_built
        }

        return {
            **base_stats,
            **self._geo_stats,
            'reasoner_stats': self.reasoner.get_stats() if self._reasoner else {}
        }

    def supports_geometric(self) -> bool:
        """Check if cassette supports geometric queries."""
        return True

    def get_index_health(self) -> Dict:
        """Get health metrics for geometric index."""
        self._ensure_index()

        if not self._geo_index:
            return {
                'status': 'empty',
                'size': 0
            }

        # Compute Df statistics
        Df_values = [s.Df for s in self._geo_index.values()]

        return {
            'status': 'healthy',
            'size': len(self._geo_index),
            'Df_mean': float(np.mean(Df_values)),
            'Df_std': float(np.std(Df_values)),
            'Df_min': float(min(Df_values)),
            'Df_max': float(max(Df_values)),
            'schema_version': self.GEO_SCHEMA_VERSION
        }


# ============================================================================
# Cross-Cassette Network
# ============================================================================

class GeometricCassetteNetwork:
    """
    Coordinate geometric queries across multiple cassettes.

    Enables:
    - Query all cassettes with single geometric state
    - Cross-cassette composition (superpose/entangle results)
    - Analogy queries spanning cassettes
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize geometric cassette network.

        Args:
            model_name: Shared model for all cassettes
        """
        self.cassettes: Dict[str, GeometricCassette] = {}
        self.reasoner = GeometricReasoner(model_name) if GeometricReasoner else None
        self._model_name = model_name

    def register(self, cassette: GeometricCassette) -> str:
        """Register a cassette in the network."""
        self.cassettes[cassette.cassette_id] = cassette
        return cassette.cassette_id

    def query_all(
        self,
        query_state: 'GeometricState',
        k: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Query all cassettes with geometric state.

        Args:
            query_state: GeometricState to query
            k: Results per cassette

        Returns:
            Dict mapping cassette_id to results
        """
        results = {}
        for cid, cassette in self.cassettes.items():
            results[cid] = cassette.query_geometric(query_state, k)
        return results

    def query_all_text(self, query_text: str, k: int = 10) -> Dict[str, List[Dict]]:
        """Query all cassettes with text."""
        if not self.reasoner:
            raise RuntimeError("GeometricReasoner not available")

        query_state = self.reasoner.initialize(query_text)
        return self.query_all(query_state, k)

    def query_merged(
        self,
        query_state: 'GeometricState',
        k: int = 10
    ) -> List[Dict]:
        """
        Query all cassettes and merge results by E.

        Args:
            query_state: GeometricState to query
            k: Total results to return

        Returns:
            Merged results sorted by E
        """
        all_results = []
        for cid, cassette in self.cassettes.items():
            results = cassette.query_geometric(query_state, k)
            # Add cassette_id to each result for traceability
            for r in results:
                r['cassette_id'] = cid
            all_results.extend(results)

        all_results.sort(key=lambda x: x['E'], reverse=True)
        return all_results[:k]

    def cross_cassette_analogy(
        self,
        a: str,
        b: str,
        c: str,
        source_cassette: str,
        target_cassette: str,
        k: int = 10
    ) -> Dict:
        """
        Analogy where a,b from source and result from target.

        Example: Find AGS equivalent of a concept from research cassette.

        Args:
            a, b, c: Analogy terms (a is to b as c is to ?)
            source_cassette: Cassette for a, b context
            target_cassette: Cassette to find result in
            k: Number of results

        Returns:
            Dict with analogy results and cross-cassette metadata
        """
        if source_cassette not in self.cassettes:
            raise ValueError(f"Source cassette '{source_cassette}' not registered")
        if target_cassette not in self.cassettes:
            raise ValueError(f"Target cassette '{target_cassette}' not registered")

        source = self.cassettes[source_cassette]
        target = self.cassettes[target_cassette]

        # Compute analogy vector
        state_a = source.reasoner.initialize(a)
        state_b = source.reasoner.initialize(b)
        state_c = source.reasoner.initialize(c)

        diff = source.reasoner.subtract(state_b, state_a)
        query_state = source.reasoner.add(diff, state_c)

        # Query target cassette
        results = target.query_geometric(query_state, k)

        return {
            'analogy': f"{a} : {b} :: {c} : ?",
            'source_cassette': source_cassette,
            'target_cassette': target_cassette,
            'query_Df': query_state.Df,
            'results': results
        }

    def compose_across(
        self,
        cassette_ids: List[str],
        query_state: 'GeometricState',
        operation: str = 'superpose'
    ) -> Dict:
        """
        Compose results from multiple cassettes geometrically.

        Args:
            cassette_ids: List of cassette IDs to compose
            query_state: Query state
            operation: 'superpose' or 'entangle'

        Returns:
            Dict with composed state and per-cassette results
        """
        if not self.reasoner:
            raise RuntimeError("GeometricReasoner not available")

        results_by_cassette = {}
        top_states = []

        for cid in cassette_ids:
            if cid not in self.cassettes:
                continue

            cassette = self.cassettes[cid]
            results = cassette.query_geometric(query_state, k=1)
            results_by_cassette[cid] = results

            if results:
                top_state = cassette._geo_index.get(results[0]['doc_id'])
                if top_state:
                    top_states.append(top_state)

        # Compose all top states
        if len(top_states) >= 2:
            composed = top_states[0]
            for state in top_states[1:]:
                if operation == 'superpose':
                    composed = self.reasoner.superpose(composed, state)
                else:
                    composed = self.reasoner.entangle(composed, state)

            return {
                'operation': operation,
                'cassettes': cassette_ids,
                'composed_state': composed.receipt(),
                'composed_Df': composed.Df,
                'results_by_cassette': results_by_cassette
            }

        return {
            'operation': operation,
            'cassettes': cassette_ids,
            'results_by_cassette': results_by_cassette,
            'note': 'Insufficient states to compose'
        }

    def get_network_stats(self) -> Dict:
        """Get statistics for entire geometric network."""
        total_docs = 0
        total_geometric_ops = 0
        total_embedding_calls = 0

        cassette_stats = {}
        for cid, cassette in self.cassettes.items():
            stats = cassette.get_stats()
            cassette_stats[cid] = stats
            total_docs += stats.get('index_size', 0)
            total_geometric_ops += stats.get('geometric_ops', 0)
            total_embedding_calls += stats.get('embedding_calls', 0)

        return {
            'cassette_count': len(self.cassettes),
            'total_documents': total_docs,
            'total_geometric_ops': total_geometric_ops,
            'total_embedding_calls': total_embedding_calls,
            'embedding_reduction': (
                total_geometric_ops / max(1, total_geometric_ops + total_embedding_calls)
            ),
            'cassettes': cassette_stats
        }

    @classmethod
    def from_config(
        cls,
        config_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        geometric_only: bool = True
    ) -> 'GeometricCassetteNetwork':
        """
        Load cassette network from JSON configuration.

        Auto-discovers cassettes.json if config_path not provided.
        Registers all cassettes with enable_geometric=True.

        Args:
            config_path: Path to cassettes.json (auto-discovers if None)
            project_root: Project root for resolving db_path (auto-discovers if None)
            geometric_only: Only load cassettes with enable_geometric=True

        Returns:
            Configured GeometricCassetteNetwork

        Example:
            # Auto-discovery
            network = GeometricCassetteNetwork.from_config()

            # Explicit path
            network = GeometricCassetteNetwork.from_config(
                config_path=Path("path/to/cassettes.json"),
                project_root=Path("path/to/project")
            )
        """
        import json

        # Auto-discover paths
        if project_root is None:
            # Navigate from this file to AGS root
            project_root = Path(__file__).resolve().parents[3]

        if config_path is None:
            config_path = project_root / "NAVIGATION" / "CORTEX" / "network" / "cassettes.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Cassette config not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Get model name from geometric_config
        geo_config = config.get('geometric_config', {})
        model_name = geo_config.get('model_name', 'all-MiniLM-L6-v2')

        # Create network
        network = cls(model_name=model_name)

        # Register cassettes
        for cassette_config in config.get('cassettes', []):
            if not cassette_config.get('enabled', True):
                continue

            if geometric_only and not cassette_config.get('enable_geometric', False):
                continue

            db_path = project_root / cassette_config['db_path']
            cassette_id = cassette_config['id']

            try:
                cassette = GeometricCassette(
                    db_path=db_path,
                    cassette_id=cassette_id,
                    model_name=model_name,
                    auto_build_from_chunks=geo_config.get('auto_build_from_chunks', True)
                )
                network.register(cassette)
            except Exception as e:
                # Log but continue - some cassettes may not exist yet
                print(f"[WARN] Could not load cassette '{cassette_id}': {e}")

        return network

    @classmethod
    def auto_discover(cls) -> 'GeometricCassetteNetwork':
        """
        Convenience method for automatic network discovery.

        Returns:
            GeometricCassetteNetwork with all available cassettes

        Example:
            network = GeometricCassetteNetwork.auto_discover()
            results = network.query_merged(query_state, k=10)
        """
        return cls.from_config()


# ============================================================================
# Factory Functions
# ============================================================================

def create_geometric_cassette(
    db_path: Path,
    cassette_id: str,
    **kwargs
) -> GeometricCassette:
    """Factory function to create geometric cassette."""
    return GeometricCassette(db_path, cassette_id, **kwargs)


def upgrade_cassette_to_geometric(
    base_cassette: DatabaseCassette,
    model_name: str = 'all-MiniLM-L6-v2'
) -> GeometricCassette:
    """
    Upgrade an existing cassette to support geometric queries.

    Creates a GeometricCassette wrapping the same database.

    Args:
        base_cassette: Existing DatabaseCassette instance
        model_name: Sentence transformer model

    Returns:
        GeometricCassette with same database
    """
    return GeometricCassette(
        db_path=base_cassette.db_path,
        cassette_id=base_cassette.cassette_id,
        model_name=model_name,
        auto_build_from_chunks=True
    )


# ============================================================================
# CLI Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geometric Cassette CLI")
    parser.add_argument("--db", type=Path, help="Database path")
    parser.add_argument("--id", default="test", help="Cassette ID")
    parser.add_argument("--query", help="Text query")
    parser.add_argument("--analogy", nargs=3, help="Analogy query: a b c")
    parser.add_argument("--stats", action="store_true", help="Show stats")

    args = parser.parse_args()

    if args.db:
        cassette = GeometricCassette(args.db, args.id)

        if args.query:
            print(f"\nQuery: {args.query}")
            results = cassette.query_text(args.query, k=5)
            for i, r in enumerate(results, 1):
                print(f"  {i}. E={r['E']:.3f} | {r['content'][:80]}...")

        if args.analogy:
            a, b, c = args.analogy
            print(f"\nAnalogy: {a} : {b} :: {c} : ?")
            results = cassette.analogy_query(a, b, c, k=5)
            for i, r in enumerate(results, 1):
                print(f"  {i}. E={r['E']:.3f} | {r['content'][:80]}...")

        if args.stats:
            print(f"\nStats: {json.dumps(cassette.get_stats(), indent=2)}")
            print(f"\nIndex Health: {json.dumps(cassette.get_index_health(), indent=2)}")
    else:
        print("Usage: python geometric_cassette.py --db <path> --query <text>")
        print("       python geometric_cassette.py --db <path> --analogy king queen man")
