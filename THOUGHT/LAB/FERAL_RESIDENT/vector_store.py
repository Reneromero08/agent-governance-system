"""
Feral Resident Vector Store (A.1.1)

Storage-backed GeometricMemory with database persistence.
Wraps geometric operations with SQLite backing.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib

# Add imports path
FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState, GeometricOperations
from geometric_memory import GeometricMemory
from resident_db import ResidentDB, VectorRecord


class VectorStore:
    """
    Storage-backed vector operations for Feral Resident.

    Combines:
    - GeometricMemory for in-memory composition
    - ResidentDB for persistence
    - Nearest neighbor search via E (Born rule)

    Usage:
        store = VectorStore("feral_eternal.db")

        # Initialize text to manifold (boundary)
        state = store.embed("What is authentication?")

        # Compose states (pure geometry)
        composed = store.compose(state1, state2)

        # Find nearest neighbors
        neighbors = store.find_nearest(query, k=5)

        # Get evolution metrics
        metrics = store.get_metrics()
    """

    def __init__(self, db_path: str = "feral_resident.db"):
        """
        Initialize vector store with database backing.

        Args:
            db_path: Path to SQLite database
        """
        self.db = ResidentDB(db_path)
        self.memory = GeometricMemory()
        self.reasoner = self.memory.reasoner

        # Cache for quick vector lookups
        self._vector_cache: Dict[str, GeometricState] = {}
        self._cache_limit = 1000

    def close(self):
        """Close database connection"""
        self.db.close()

    # =========================================================================
    # Boundary Operations (Text <-> Manifold)
    # =========================================================================

    def embed(self, text: str, store: bool = True) -> GeometricState:
        """
        Initialize text to geometric manifold (BOUNDARY operation).

        This is where text enters the geometric system.

        Args:
            text: Input text to embed
            store: Whether to persist to database

        Returns:
            GeometricState on the semantic manifold
        """
        state = self.reasoner.initialize(text)

        if store:
            vector_id = self.db.store_vector(
                vector=state.vector,
                Df=state.Df,
                composition_op='initialize',
                parent_ids=None
            )

            # Store receipt
            self.db.store_receipt(
                operation='initialize',
                input_hashes=[hashlib.sha256(text.encode()).hexdigest()[:16]],
                output_hash=state.receipt()['vector_hash'],
                metadata={'text_preview': text[:100]}
            )

            # Cache
            self._cache_state(vector_id, state)

        return state

    def decode(
        self,
        state: GeometricState,
        corpus: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Decode geometric state back to text (BOUNDARY operation).

        This is where manifold state exits as text.

        Args:
            state: GeometricState to decode
            corpus: Candidate texts to match against
            k: Number of results to return

        Returns:
            List of (text, E_value) tuples sorted by E
        """
        return self.reasoner.readout(state, corpus, k)

    # =========================================================================
    # Pure Geometry Operations
    # =========================================================================

    def compose(
        self,
        state1: GeometricState,
        state2: GeometricState,
        store: bool = True
    ) -> GeometricState:
        """
        Compose two states via entanglement (PURE GEOMETRY).

        Uses quantum entanglement (circular convolution) to bind states.

        Args:
            state1: First state
            state2: Second state
            store: Whether to persist result

        Returns:
            Entangled GeometricState
        """
        result = self.reasoner.entangle(state1, state2)

        if store:
            # Get parent IDs from cache or store parents first
            parent_ids = self._get_or_store_parents([state1, state2])

            vector_id = self.db.store_vector(
                vector=result.vector,
                Df=result.Df,
                composition_op='entangle',
                parent_ids=parent_ids
            )

            self.db.store_receipt(
                operation='entangle',
                input_hashes=[state1.receipt()['vector_hash'], state2.receipt()['vector_hash']],
                output_hash=result.receipt()['vector_hash'],
                metadata={'Df': result.Df}
            )

            self._cache_state(vector_id, result)

        return result

    def blend(
        self,
        states: List[GeometricState],
        store: bool = True
    ) -> GeometricState:
        """
        Blend multiple states via superposition (PURE GEOMETRY).

        Creates quantum superposition of all input states.

        Args:
            states: List of states to blend
            store: Whether to persist result

        Returns:
            Superposed GeometricState
        """
        if not states:
            raise ValueError("Need at least one state to blend")

        result = states[0]
        for s in states[1:]:
            result = self.reasoner.superpose(result, s)

        if store:
            parent_ids = self._get_or_store_parents(states)

            vector_id = self.db.store_vector(
                vector=result.vector,
                Df=result.Df,
                composition_op='superpose',
                parent_ids=parent_ids
            )

            input_hashes = [s.receipt()['vector_hash'] for s in states]
            self.db.store_receipt(
                operation='superpose',
                input_hashes=input_hashes,
                output_hash=result.receipt()['vector_hash'],
                metadata={'blend_count': len(states), 'Df': result.Df}
            )

            self._cache_state(vector_id, result)

        return result

    def project(
        self,
        query: GeometricState,
        context: List[GeometricState],
        store: bool = True
    ) -> GeometricState:
        """
        Project query onto context subspace (PURE GEOMETRY).

        Uses Born rule projection (Q44) to condition on context.

        Args:
            query: State to project
            context: Context states defining subspace
            store: Whether to persist result

        Returns:
            Projected GeometricState
        """
        result = self.reasoner.project(query, context)

        if store:
            parent_ids = self._get_or_store_parents([query] + context)

            vector_id = self.db.store_vector(
                vector=result.vector,
                Df=result.Df,
                composition_op='project',
                parent_ids=parent_ids
            )

            input_hashes = [query.receipt()['vector_hash']] + [c.receipt()['vector_hash'] for c in context]
            self.db.store_receipt(
                operation='project',
                input_hashes=input_hashes,
                output_hash=result.receipt()['vector_hash'],
                metadata={'context_size': len(context), 'Df': result.Df}
            )

            self._cache_state(vector_id, result)

        return result

    def interpolate(
        self,
        start: GeometricState,
        end: GeometricState,
        t: float,
        store: bool = False
    ) -> GeometricState:
        """
        Interpolate between states via geodesic (PURE GEOMETRY).

        Spherical linear interpolation on unit sphere.

        Args:
            start: Starting state
            end: Ending state
            t: Interpolation parameter [0, 1]
            store: Whether to persist result

        Returns:
            Interpolated GeometricState
        """
        result = self.reasoner.interpolate(start, end, t)

        if store:
            parent_ids = self._get_or_store_parents([start, end])

            vector_id = self.db.store_vector(
                vector=result.vector,
                Df=result.Df,
                composition_op='interpolate',
                parent_ids=parent_ids
            )

            self.db.store_receipt(
                operation='interpolate',
                input_hashes=[start.receipt()['vector_hash'], end.receipt()['vector_hash']],
                output_hash=result.receipt()['vector_hash'],
                metadata={'t': t, 'Df': result.Df}
            )

            self._cache_state(vector_id, result)

        return result

    # =========================================================================
    # Nearest Neighbor Search
    # =========================================================================

    def find_nearest(
        self,
        query: GeometricState,
        k: int = 10,
        exclude_hashes: Optional[List[str]] = None
    ) -> List[Tuple[VectorRecord, float]]:
        """
        Find k nearest neighbors using E (Born rule).

        Computes quantum inner product with all stored vectors.

        Args:
            query: Query state
            k: Number of neighbors to return
            exclude_hashes: Vector hashes to exclude from results

        Returns:
            List of (VectorRecord, E_value) tuples sorted by E descending
        """
        exclude_set = set(exclude_hashes or [])

        # Get all stored vectors
        all_vectors = self.db.get_all_vectors(limit=10000)

        # Compute E with each
        results = []
        for record in all_vectors:
            if record.vec_sha256 in exclude_set:
                continue

            # Reconstruct GeometricState
            state = GeometricState(
                vector=record.vector,
                operation_history=[]
            )

            E = query.E_with(state)
            results.append((record, E))

        # Sort by E descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def find_similar_to_text(
        self,
        text: str,
        k: int = 10
    ) -> List[Tuple[VectorRecord, float]]:
        """
        Find vectors similar to text query.

        Args:
            text: Query text
            k: Number of results

        Returns:
            List of (VectorRecord, E_value) tuples
        """
        query = self.embed(text, store=False)
        return self.find_nearest(query, k=k)

    # =========================================================================
    # Memory Integration
    # =========================================================================

    def remember(self, text: str) -> Dict:
        """
        Add text to compositional memory.

        Wraps GeometricMemory.remember() with persistence.

        Args:
            text: Text to remember

        Returns:
            Receipt of the operation
        """
        # Use GeometricMemory for composition
        receipt = self.memory.remember(text)

        # Persist the updated mind state
        if self.memory.mind_state is not None:
            vector_id = self.db.store_vector(
                vector=self.memory.mind_state.vector,
                Df=self.memory.mind_state.Df,
                composition_op='remember',
                parent_ids=None  # Complex composition
            )

            self.db.store_receipt(
                operation='remember',
                input_hashes=[receipt['interaction_hash']],
                output_hash=receipt['mind_hash'],
                metadata={
                    'Df': receipt['Df'],
                    'distance_from_start': receipt['distance_from_start'],
                    'memory_index': receipt['memory_index']
                }
            )

            self._cache_state(vector_id, self.memory.mind_state)

        return receipt

    def recall(
        self,
        query_text: str,
        corpus: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recall from compositional memory.

        Uses GeometricMemory.recall() for context-aware retrieval.

        Args:
            query_text: Query text
            corpus: Candidate texts
            k: Number of results

        Returns:
            List of (text, E_value) tuples
        """
        return self.memory.recall(query_text, corpus, k)

    def recall_with_gate(
        self,
        query_text: str,
        corpus: List[str],
        k: int = 5,
        threshold: float = 0.3
    ) -> Dict:
        """
        Recall with E-gating for relevance filtering.

        Returns:
            Dict with results, E, gate_open, etc.
        """
        return self.memory.recall_with_gate(query_text, corpus, k, threshold)

    # =========================================================================
    # Metrics & State
    # =========================================================================

    def get_metrics(self) -> Dict:
        """Get comprehensive store metrics"""
        db_stats = self.db.get_stats()
        memory_metrics = self.memory.get_evolution_metrics()
        reasoner_stats = self.reasoner.get_stats()

        return {
            'db': db_stats,
            'memory': memory_metrics,
            'reasoner': reasoner_stats,
            'cache_size': len(self._vector_cache)
        }

    def get_mind_state(self) -> Optional[GeometricState]:
        """Get current compositional mind state"""
        return self.memory.mind_state

    def get_mind_hash(self) -> Optional[str]:
        """Get hash of current mind state"""
        if self.memory.mind_state is None:
            return None
        return self.memory.mind_state.receipt()['vector_hash']

    def get_mind_Df(self) -> float:
        """Get current mind Df (participation ratio)"""
        if self.memory.mind_state is None:
            return 0.0
        return self.memory.mind_state.Df

    def get_distance_from_start(self) -> float:
        """Get geodesic distance from initial mind state"""
        return self.memory.mind_distance_from_start()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _cache_state(self, vector_id: str, state: GeometricState):
        """Add state to cache with LRU eviction"""
        if len(self._vector_cache) >= self._cache_limit:
            # Simple eviction: remove first item
            first_key = next(iter(self._vector_cache))
            del self._vector_cache[first_key]

        self._vector_cache[vector_id] = state

    def _get_or_store_parents(self, states: List[GeometricState]) -> List[str]:
        """Get vector IDs for parent states, storing if necessary"""
        parent_ids = []

        for state in states:
            # Check cache first
            for vid, cached_state in self._vector_cache.items():
                if np.array_equal(cached_state.vector, state.vector):
                    parent_ids.append(vid)
                    break
            else:
                # Not in cache, check database by hash
                vec_hash = state.receipt()['vector_hash']
                # Convert to full SHA256
                full_hash = hashlib.sha256(state.vector.astype(np.float32).tobytes()).hexdigest()
                record = self.db.get_vector_by_hash(full_hash)

                if record:
                    parent_ids.append(record.vector_id)
                    self._cache_state(record.vector_id, state)
                else:
                    # Store new
                    vector_id = self.db.store_vector(
                        vector=state.vector,
                        Df=state.Df,
                        composition_op='external',
                        parent_ids=None
                    )
                    parent_ids.append(vector_id)
                    self._cache_state(vector_id, state)

        return parent_ids

    def clear_memory(self):
        """Clear in-memory state (not database)"""
        self.memory.clear()
        self._vector_cache.clear()

    # =========================================================================
    # Paper Index Integration
    # =========================================================================

    def load_papers(self, papers_dir: Optional[str] = None, max_chunks: int = 10000) -> Dict:
        """
        Load indexed papers into the vector store.

        Papers become part of the navigable semantic space.
        The resident can find_nearest() to paper chunks.

        Args:
            papers_dir: Path to papers directory (default: research/papers/)
            max_chunks: Maximum chunks to load

        Returns:
            Loading stats
        """
        from paper_indexer import PaperIndexer

        indexer = PaperIndexer(papers_dir)
        stats = {'papers_loaded': 0, 'chunks_loaded': 0, 'skipped': 0}

        papers = indexer.list_papers(status='indexed')

        for paper in papers:
            arxiv_id = paper.get('arxiv_id', 'unknown')
            markdown_path = paper.get('markdown_path')

            if not markdown_path:
                stats['skipped'] += 1
                continue

            # Resolve path
            full_path = Path(markdown_path)
            if not full_path.is_absolute():
                full_path = indexer.papers_dir / markdown_path

            if not full_path.exists():
                stats['skipped'] += 1
                continue

            # Chunk and load
            try:
                chunks = indexer.chunk_by_headings(str(full_path))

                for chunk in chunks:
                    if stats['chunks_loaded'] >= max_chunks:
                        break

                    content = chunk['content'][:2000]
                    heading = chunk['heading']

                    # Create paper reference text
                    paper_text = f"@Paper-{arxiv_id} {heading}\n{content}"

                    # Embed and store with paper metadata
                    state = self.embed(paper_text, store=True)

                    # Store additional metadata INCLUDING the actual content
                    self.db.store_receipt(
                        operation='paper_load',
                        input_hashes=[(chunk.get('hash') or '')[:16]],
                        output_hash=state.receipt()['vector_hash'],
                        metadata={
                            'paper_id': arxiv_id,
                            'heading': heading,
                            'alias': paper.get('alias_symbol') or '',
                            'category': paper.get('category') or '',
                            'content': content  # Store actual text for readout
                        }
                    )

                    stats['chunks_loaded'] += 1

                stats['papers_loaded'] += 1

            except Exception as e:
                print(f"Error loading {arxiv_id}: {e}")
                stats['skipped'] += 1

        return stats

    def get_paper_chunks(self, limit: int = 1000) -> List[Dict]:
        """
        Get paper chunks from cassette databases for exploration.

        Returns:
            List of dicts with chunk_id, paper_id, heading, content
        """
        import sqlite3

        # Cassettes directory (REPO_ROOT/NAVIGATION/CORTEX/cassettes)
        # vector_store.py is in THOUGHT/LAB/FERAL_RESIDENT, so go up 4 levels to REPO_ROOT
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        cassettes_dir = repo_root / "NAVIGATION" / "CORTEX" / "cassettes"

        chunks = []
        if not cassettes_dir.exists():
            # Store error info for debugging
            self._last_error = f"Cassettes dir not found: {cassettes_dir}"
            return chunks

        per_cassette_limit = max(50, limit // 10)  # Distribute across cassettes

        for db_file in cassettes_dir.glob("*.db"):
            if len(chunks) >= limit:
                break

            cassette_name = db_file.stem
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    # Chunks table has: chunk_id, file_id, header_text, start_offset, end_offset
                    # Content is in original files, not in DB
                    cursor = conn.execute("""
                        SELECT c.chunk_id, c.header_text, f.path, c.start_offset, c.end_offset
                        FROM chunks c
                        LEFT JOIN files f ON c.file_id = f.file_id
                        WHERE c.header_text IS NOT NULL
                        LIMIT ?
                    """, (per_cassette_limit,))

                    for row in cursor.fetchall():
                        chunk_id, header, file_path, start_off, end_off = row

                        # Try to read content from file
                        content = ''
                        if file_path and start_off is not None and end_off is not None:
                            full_path = repo_root / file_path
                            if full_path.exists():
                                try:
                                    text = full_path.read_text(encoding='utf-8')
                                    content = text[start_off:end_off][:2000]
                                except:
                                    content = header or ''

                        chunks.append({
                            'chunk_id': str(chunk_id),
                            'paper_id': cassette_name,
                            'heading': header or '',
                            'content': content,
                            'alias': ''
                        })

                        if len(chunks) >= limit:
                            break
            except Exception:
                continue  # Skip broken databases

        return chunks

    def find_paper_chunks(
        self,
        query: GeometricState,
        k: int = 10,
        min_E: float = 0.3
    ) -> List[Dict]:
        """
        Find paper chunks relevant to query using E (Born rule).

        Returns chunks with E above threshold.

        Args:
            query: Query state
            k: Max results
            min_E: Minimum E threshold (Born rule gate)

        Returns:
            List of dicts with paper info and E values
        """
        neighbors = self.find_nearest(query, k=k * 2)  # Get extra, filter by E

        results = []
        for record, E in neighbors:
            if E < min_E:
                continue

            # Get paper metadata from receipts
            receipts = self.db.get_receipts_by_output(record.vec_sha256[:16])
            paper_meta = None
            for r in receipts:
                if r['operation'] == 'paper_load':
                    paper_meta = r['metadata']
                    break

            results.append({
                'E': E,
                'Df': record.Df,
                'paper_id': paper_meta.get('paper_id') if paper_meta else None,
                'heading': paper_meta.get('heading') if paper_meta else None,
                'alias': paper_meta.get('alias') if paper_meta else None,
                'content': paper_meta.get('content') if paper_meta else None,  # Actual text
                'chunk_id': record.vec_sha256[:16],  # Consistent with get_paper_chunks()
                'vector_hash': record.vec_sha256[:16]  # Keep for backwards compat
            })

            if len(results) >= k:
                break

        return results


# ============================================================================
# Testing
# ============================================================================

def example_usage():
    """Demonstrate VectorStore usage"""
    import tempfile
    import os

    print("=== VectorStore Example ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        store = VectorStore(db_path)

        # Embed some text
        s1 = store.embed("What is authentication?")
        print(f"Embedded 'authentication': Df={s1.Df:.2f}")

        s2 = store.embed("How does JWT work?")
        print(f"Embedded 'JWT': Df={s2.Df:.2f}")

        # Compose
        composed = store.compose(s1, s2)
        print(f"Composed: Df={composed.Df:.2f}")

        # Remember (compositional memory)
        store.remember("User asked about security")
        store.remember("I explained OAuth vs JWT")
        store.remember("User chose JWT approach")

        print(f"\nMind evolution:")
        print(f"  Df: {store.get_mind_Df():.2f}")
        print(f"  Distance from start: {store.get_distance_from_start():.3f}")

        # Find nearest
        query = store.embed("tokens and security", store=False)
        neighbors = store.find_nearest(query, k=3)
        print(f"\nNearest to 'tokens and security':")
        for record, E in neighbors:
            print(f"  E={E:.3f}, op={record.composition_op}")

        # Metrics
        metrics = store.get_metrics()
        print(f"\nMetrics:")
        print(f"  Vectors stored: {metrics['db']['vector_count']}")
        print(f"  Memory interactions: {metrics['memory']['interaction_count']}")

        store.close()
        print("\nDone!")


if __name__ == "__main__":
    example_usage()
