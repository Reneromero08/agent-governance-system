#!/usr/bin/env python3
"""
Memory Cassette - Write path for AI memory persistence (Phase 2).

Implements the resident.db cassette with write capabilities:
- memory_save(text, metadata) -> hash
- memory_query(query, cassettes, limit) -> [{hash, similarity, text_preview}]
- memory_recall(hash) -> {hash, text, vector, metadata, created_at}

Architecture:
    Unlike read-only cassettes that index repo files, the memory cassette
    stores AI-generated thoughts with vector embeddings for semantic retrieval.
    Each memory is content-addressed by its text hash.

Schema (see CASSETTE_NETWORK_ROADMAP.md Phase 2.2):
    memories(hash, text, vector, metadata, created_at, agent_id, indexed_at)
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Import embedding engine from existing infrastructure
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "semantic"))
    from embeddings import EmbeddingEngine, get_embedding_engine
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EmbeddingEngine = None

from cassette_protocol import DatabaseCassette


# Configuration
CASSETTES_DIR = Path(__file__).parent.parent / "cassettes"
RESIDENT_DB = CASSETTES_DIR / "resident.db"


class MemoryCassette(DatabaseCassette):
    """Write-capable cassette for AI memory persistence.

    Extends the standard cassette protocol with write operations.
    Uses vector embeddings for semantic memory retrieval.
    """

    def __init__(self, db_path: Path = None, agent_id: str = "default"):
        """Initialize memory cassette.

        Args:
            db_path: Path to SQLite database (default: resident.db)
            agent_id: Identifier for the AI agent (e.g., 'opus', 'sonnet')
        """
        db_path = db_path or RESIDENT_DB
        super().__init__(db_path, "resident")

        self.agent_id = agent_id
        self.capabilities = ["fts", "semantic_search", "agent_memory", "write"]
        self.schema_version = "2.0"

        # Lazy-load embedding engine
        self._embedding_engine = None

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    @property
    def embedding_engine(self) -> Optional[EmbeddingEngine]:
        """Lazy-load embedding engine."""
        if self._embedding_engine is None and EMBEDDINGS_AVAILABLE:
            self._embedding_engine = get_embedding_engine()
        return self._embedding_engine

    def _init_schema(self):
        """Initialize memory schema (Phase 2.2)."""
        conn = sqlite3.connect(str(self.db_path))

        conn.executescript("""
            -- Memories table: stores AI-generated thoughts with vectors
            CREATE TABLE IF NOT EXISTS memories (
                hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata JSON,
                created_at TEXT NOT NULL,
                agent_id TEXT,
                indexed_at TEXT NOT NULL
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_indexed ON memories(indexed_at);

            -- FTS5 for text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text,
                hash UNINDEXED,
                tokenize='porter unicode61'
            );

            -- Cassette metadata table
            CREATE TABLE IF NOT EXISTS cassette_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Insert schema version
            INSERT OR REPLACE INTO cassette_metadata (key, value)
            VALUES ('schema_version', '2.0');
        """)

        conn.commit()
        conn.close()

    # =========================================================================
    # Phase 2.1: Core Functions
    # =========================================================================

    def memory_save(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        agent_id: Optional[str] = None
    ) -> str:
        """Save a memory to the cassette.

        Embeds the text, stores the vector, and indexes for search.

        Args:
            text: The memory content to save
            metadata: Optional metadata dictionary
            agent_id: Override default agent ID

        Returns:
            Content-addressed hash of the memory
        """
        if not text or not text.strip():
            raise ValueError("Cannot save empty memory")

        agent_id = agent_id or self.agent_id

        # Compute content hash
        memory_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Generate embedding
        if self.embedding_engine:
            embedding = self.embedding_engine.embed(text)
            vector_blob = self.embedding_engine.serialize(embedding)
        else:
            # Fallback: zero vector if embeddings unavailable
            vector_blob = np.zeros(384, dtype=np.float32).tobytes()

        # Timestamps
        now = datetime.now(timezone.utc).isoformat()

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        # Store in database
        conn = sqlite3.connect(str(self.db_path))

        try:
            # Check if memory already exists
            cursor = conn.execute(
                "SELECT hash FROM memories WHERE hash = ?",
                (memory_hash,)
            )
            existing = cursor.fetchone()

            if existing:
                # Memory already exists - update indexed_at
                conn.execute(
                    "UPDATE memories SET indexed_at = ? WHERE hash = ?",
                    (now, memory_hash)
                )
            else:
                # Insert new memory
                conn.execute("""
                    INSERT INTO memories (hash, text, vector, metadata, created_at, agent_id, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (memory_hash, text, vector_blob, metadata_json, now, agent_id, now))

                # Insert into FTS index
                conn.execute(
                    "INSERT INTO memories_fts (text, hash) VALUES (?, ?)",
                    (text, memory_hash)
                )

            conn.commit()
            return memory_hash

        finally:
            conn.close()

    def memory_query(
        self,
        query: str,
        limit: int = 10,
        agent_id: Optional[str] = None
    ) -> List[Dict]:
        """Semantic search over memories.

        Args:
            query: Search query string
            limit: Maximum results to return
            agent_id: Filter to specific agent (None for all)

        Returns:
            List of {hash, similarity, text_preview, agent_id, created_at}
        """
        if not query or not query.strip():
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Check if we have embeddings for semantic search
            if self.embedding_engine:
                return self._semantic_query(conn, query, limit, agent_id)
            else:
                # Fallback to FTS search
                return self._fts_query(conn, query, limit, agent_id)
        finally:
            conn.close()

    def _semantic_query(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
        agent_id: Optional[str]
    ) -> List[Dict]:
        """Semantic search using vector similarity."""
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query)

        # Fetch all vectors (for small memory counts; will optimize later)
        if agent_id:
            cursor = conn.execute(
                "SELECT hash, text, vector, agent_id, created_at FROM memories WHERE agent_id = ?",
                (agent_id,)
            )
        else:
            cursor = conn.execute(
                "SELECT hash, text, vector, agent_id, created_at FROM memories"
            )

        rows = cursor.fetchall()
        if not rows:
            return []

        # Compute similarities
        results = []
        for row in rows:
            vector = self.embedding_engine.deserialize(row['vector'])
            similarity = float(self.embedding_engine.cosine_similarity(query_embedding, vector))

            results.append({
                "hash": row['hash'],
                "similarity": similarity,
                "text_preview": row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
                "agent_id": row['agent_id'],
                "created_at": row['created_at'],
                "cassette": "resident"
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:limit]

    def _fts_query(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
        agent_id: Optional[str]
    ) -> List[Dict]:
        """Fallback FTS search when embeddings unavailable."""
        if agent_id:
            cursor = conn.execute("""
                SELECT
                    m.hash,
                    m.text,
                    m.agent_id,
                    m.created_at,
                    1.0 as similarity
                FROM memories_fts fts
                JOIN memories m ON fts.hash = m.hash
                WHERE memories_fts MATCH ? AND m.agent_id = ?
                ORDER BY rank
                LIMIT ?
            """, (query, agent_id, limit))
        else:
            cursor = conn.execute("""
                SELECT
                    m.hash,
                    m.text,
                    m.agent_id,
                    m.created_at,
                    1.0 as similarity
                FROM memories_fts fts
                JOIN memories m ON fts.hash = m.hash
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))

        results = []
        for row in cursor.fetchall():
            text = row['text']
            results.append({
                "hash": row['hash'],
                "similarity": row['similarity'],
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "agent_id": row['agent_id'],
                "created_at": row['created_at'],
                "cassette": "resident"
            })

        return results

    def memory_recall(self, memory_hash: str) -> Optional[Dict]:
        """Retrieve full memory by hash.

        Args:
            memory_hash: Content-addressed hash of the memory

        Returns:
            Dict with {hash, text, vector, metadata, created_at, agent_id, cassette}
            or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE hash = ?",
                (memory_hash,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Parse metadata
            metadata = json.loads(row['metadata']) if row['metadata'] else None

            return {
                "hash": row['hash'],
                "text": row['text'],
                "vector": row['vector'],  # Raw bytes
                "metadata": metadata,
                "created_at": row['created_at'],
                "agent_id": row['agent_id'],
                "indexed_at": row['indexed_at'],
                "cassette": "resident"
            }
        finally:
            conn.close()

    def semantic_neighbors(
        self,
        memory_hash: str,
        limit: int = 10,
        cassettes: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find memories semantically similar to a given memory.

        Args:
            memory_hash: Hash of the anchor memory
            limit: Maximum neighbors to return
            cassettes: List of cassette IDs to search (default: resident only)

        Returns:
            List of similar memories (excluding the anchor)
        """
        # Get the anchor memory
        anchor = self.memory_recall(memory_hash)
        if not anchor:
            return []

        # Use its text as the query
        results = self.memory_query(anchor['text'], limit=limit + 1)

        # Filter out the anchor itself
        return [r for r in results if r['hash'] != memory_hash][:limit]

    # =========================================================================
    # Standard Cassette Interface
    # =========================================================================

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Standard cassette query interface."""
        return self.memory_query(query_text, limit=top_k)

    def get_stats(self) -> Dict:
        """Return cassette statistics."""
        stats = {
            "cassette_id": self.cassette_id,
            "name": "Resident (AI Memories)",
            "description": "Per-agent AI memories with vector search",
            "capabilities": self.capabilities,
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "schema_version": self.schema_version
        }

        if not self.db_path.exists():
            return stats

        conn = sqlite3.connect(str(self.db_path))

        try:
            # Total memories
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            stats["total_memories"] = cursor.fetchone()[0]

            # Memories by agent
            cursor = conn.execute("""
                SELECT agent_id, COUNT(*) as count
                FROM memories
                GROUP BY agent_id
            """)
            stats["memories_by_agent"] = {
                row[0] or "unknown": row[1] for row in cursor.fetchall()
            }

            # Date range
            cursor = conn.execute("""
                SELECT MIN(created_at), MAX(created_at) FROM memories
            """)
            row = cursor.fetchone()
            if row[0]:
                stats["first_memory"] = row[0]
                stats["last_memory"] = row[1]

            stats["embeddings_available"] = EMBEDDINGS_AVAILABLE

        finally:
            conn.close()

        return stats


# ============================================================================
# Convenience Functions (for MCP tools)
# ============================================================================

_default_cassette: Optional[MemoryCassette] = None


def get_memory_cassette(agent_id: str = "default") -> MemoryCassette:
    """Get or create the default memory cassette."""
    global _default_cassette
    if _default_cassette is None or _default_cassette.agent_id != agent_id:
        _default_cassette = MemoryCassette(agent_id=agent_id)
    return _default_cassette


def memory_save(text: str, metadata: Optional[Dict] = None, agent_id: str = "default") -> str:
    """Save a memory. Returns hash."""
    cassette = get_memory_cassette(agent_id)
    return cassette.memory_save(text, metadata, agent_id)


def memory_query(query: str, limit: int = 10, agent_id: Optional[str] = None) -> List[Dict]:
    """Query memories. Returns list of matches with similarity scores."""
    cassette = get_memory_cassette()
    return cassette.memory_query(query, limit, agent_id)


def memory_recall(memory_hash: str) -> Optional[Dict]:
    """Recall full memory by hash."""
    cassette = get_memory_cassette()
    return cassette.memory_recall(memory_hash)


def semantic_neighbors(memory_hash: str, limit: int = 10) -> List[Dict]:
    """Find semantically similar memories."""
    cassette = get_memory_cassette()
    return cassette.semantic_neighbors(memory_hash, limit)


# Aliases for consistency with roadmap
symbol_resolve = memory_recall
cas_retrieve = memory_recall


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demo the memory cassette."""
    print("=== Memory Cassette Demo (Phase 2) ===\n")

    # Create cassette with test agent
    cassette = MemoryCassette(agent_id="demo_agent")

    # Save some memories
    print("Saving memories...")

    hash1 = cassette.memory_save(
        "The Formula is beautiful - it reveals how entropy and resonance create meaning.",
        metadata={"context": "reflecting on FORMULA.md"}
    )
    print(f"  Memory 1: {hash1[:16]}...")

    hash2 = cassette.memory_save(
        "I discovered that eigenvalue spectrums are invariant across different embedding models.",
        metadata={"context": "ESAP research", "confidence": 0.95}
    )
    print(f"  Memory 2: {hash2[:16]}...")

    hash3 = cassette.memory_save(
        "The catalytic property ensures all transformations are verifiable and reversible.",
        metadata={"context": "catalytic computing"}
    )
    print(f"  Memory 3: {hash3[:16]}...")

    # Query memories
    print("\n--- Querying 'formula beauty' ---")
    results = cassette.memory_query("formula beauty", limit=3)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['hash'][:12]}... : {r['text_preview'][:60]}...")

    print("\n--- Querying 'eigenvalues vectors' ---")
    results = cassette.memory_query("eigenvalues vectors", limit=3)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['hash'][:12]}... : {r['text_preview'][:60]}...")

    # Recall specific memory
    print(f"\n--- Recalling memory {hash1[:16]}... ---")
    memory = cassette.memory_recall(hash1)
    if memory:
        print(f"  Text: {memory['text']}")
        print(f"  Metadata: {memory['metadata']}")
        print(f"  Created: {memory['created_at']}")

    # Find neighbors
    print(f"\n--- Semantic neighbors of {hash1[:16]}... ---")
    neighbors = cassette.semantic_neighbors(hash1, limit=2)
    for n in neighbors:
        print(f"  [{n['similarity']:.3f}] {n['text_preview'][:60]}...")

    # Stats
    print("\n--- Cassette Stats ---")
    stats = cassette.get_stats()
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  By agent: {stats.get('memories_by_agent', {})}")
    print(f"  Embeddings available: {stats.get('embeddings_available', False)}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo()
