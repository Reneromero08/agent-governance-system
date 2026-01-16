#!/usr/bin/env python3
"""
Memory Cassette - Write path for AI memory persistence (Phase 2 + Phase 3).

Phase 2 - Memory Persistence:
- memory_save(text, metadata) -> hash
- memory_query(query, limit) -> [{hash, similarity, text_preview}]
- memory_recall(hash) -> {hash, text, vector, metadata, created_at}

Phase 3 - Resident Identity:
- agent_register(agent_id, model_name) -> {agent_id, created_at}
- session_start(agent_id) -> {session_id, started_at}
- session_resume(agent_id) -> {recent_thoughts, working_set}
- memory_promote(hash) -> {promoted_at}

Architecture:
    Unlike read-only cassettes that index repo files, the memory cassette
    stores AI-generated thoughts with vector embeddings for semantic retrieval.
    Each memory is content-addressed by its text hash.

    Phase 3 adds persistent agent identity and session continuity, enabling
    agents to resume context across sessions ("what did I think last time?").

Schema (see CASSETTE_NETWORK_ROADMAP.md):
    Phase 2.2: memories(hash, text, vector, metadata, created_at, agent_id, indexed_at)
    Phase 3.1: agents(agent_id, model_name, created_at, last_active, memory_count)
    Phase 3.2: sessions(session_id, agent_id, started_at, working_set)
"""

import sqlite3
import hashlib
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Phase 6 imports for receipt emission and MemoryRecord binding
try:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"))
    from cassette_receipt import (
        CassetteReceipt,
        create_receipt,
        receipt_from_dict,
        verify_receipt_chain,
        compute_session_merkle_root,
        canonical_json,
    )
    from memory_record import (
        create_record as create_memory_record,
        hash_record,
        full_hash as full_memory_hash,
    )
    PHASE6_AVAILABLE = True
except ImportError:
    PHASE6_AVAILABLE = False
    CassetteReceipt = None

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

# Import GuardedWriter for write firewall compliance
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "utilities"))
    from guarded_writer import GuardedWriter
    GUARDED_WRITER_AVAILABLE = True
except ImportError:
    GUARDED_WRITER_AVAILABLE = False
    GuardedWriter = None


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
        self.capabilities = ["fts", "semantic_search", "agent_memory", "write", "sessions", "spc", "receipts"]
        self.schema_version = "5.0"  # Phase 6: receipts + MemoryRecord binding

        # Active session tracking
        self._current_session_id: Optional[str] = None

        # Lazy-load embedding engine
        self._embedding_engine = None

        # Phase 6: Receipt chain tracking
        self._last_receipt_hash: Optional[str] = None
        self._receipt_index: int = 0
        self._session_receipts: List[CassetteReceipt] = [] if PHASE6_AVAILABLE else []

        # Ensure directory exists (use GuardedWriter if available for write firewall compliance)
        if GUARDED_WRITER_AVAILABLE:
            # Get relative path from project root
            rel_path = str(self.db_path.parent.relative_to(PROJECT_ROOT))
            writer = GuardedWriter(
                project_root=PROJECT_ROOT,
                durable_roots=["NAVIGATION/CORTEX/cassettes"]
            )
            writer.open_commit_gate()  # Required before any durable writes
            writer.mkdir_durable(rel_path)
        else:
            # Fallback for contexts without GuardedWriter
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
        """Initialize memory schema (Phase 2.2 + Phase 3)."""
        conn = sqlite3.connect(str(self.db_path))

        conn.executescript("""
            -- =====================================================================
            -- Phase 2.2: Memories table
            -- =====================================================================

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

            -- =====================================================================
            -- Phase 3.1: Agents table (Resident Identity)
            -- =====================================================================

            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                model_name TEXT,
                display_name TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                memory_count INTEGER DEFAULT 0,
                session_count INTEGER DEFAULT 0,
                config JSON
            );

            CREATE INDEX IF NOT EXISTS idx_agents_last_active ON agents(last_active);
            CREATE INDEX IF NOT EXISTS idx_agents_model ON agents(model_name);

            -- =====================================================================
            -- Phase 3.2: Sessions table (Session Continuity)
            -- =====================================================================

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                last_active TEXT NOT NULL,
                memory_count INTEGER DEFAULT 0,
                working_set JSON,
                summary TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(agent_id, ended_at);

            -- =====================================================================
            -- Phase 4.1: Pointers table (SPC Integration)
            -- =====================================================================

            -- Pointers table: caches resolved SPC pointers
            CREATE TABLE IF NOT EXISTS pointers (
                pointer_id TEXT PRIMARY KEY,
                pointer_type TEXT NOT NULL,  -- SYMBOL_PTR | HASH_PTR | COMPOSITE_PTR
                base_ptr TEXT NOT NULL,
                target_hash TEXT,
                qualifiers JSON,
                codebook_id TEXT DEFAULT 'ags-codebook',
                created_at TEXT NOT NULL,
                resolved_count INTEGER DEFAULT 0,
                last_resolved TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_pointers_type ON pointers(pointer_type);
            CREATE INDEX IF NOT EXISTS idx_pointers_base ON pointers(base_ptr);

            -- =====================================================================
            -- Phase 6.0: Receipts table (Cassette Write Receipts)
            -- =====================================================================

            CREATE TABLE IF NOT EXISTS cassette_receipts (
                receipt_hash TEXT PRIMARY KEY,
                receipt_json TEXT NOT NULL,
                cassette_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                record_id TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                parent_receipt_hash TEXT,
                receipt_index INTEGER,
                session_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_receipts_parent ON cassette_receipts(parent_receipt_hash);
            CREATE INDEX IF NOT EXISTS idx_receipts_index ON cassette_receipts(receipt_index);
            CREATE INDEX IF NOT EXISTS idx_receipts_session ON cassette_receipts(session_id);
            CREATE INDEX IF NOT EXISTS idx_receipts_record ON cassette_receipts(record_id);

            -- Insert/update schema version
            INSERT OR REPLACE INTO cassette_metadata (key, value)
            VALUES ('schema_version', '5.0');
        """)

        # Phase 3.3: Extend memories table with session tracking columns
        # Use ALTER TABLE to add columns if they don't exist (SQLite limitation)
        self._migrate_memories_table(conn)

        conn.commit()
        conn.close()

    def _migrate_memories_table(self, conn: sqlite3.Connection):
        """Add Phase 3 + Phase 6 columns to memories and sessions tables if not present."""
        # Migrate memories table
        cursor = conn.execute("PRAGMA table_info(memories)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Phase 3.3 columns
        migrations = [
            ("session_id", "TEXT"),
            ("access_count", "INTEGER DEFAULT 0"),
            ("last_accessed", "TEXT"),
            ("promoted_at", "TEXT"),
            ("source_cassette", "TEXT DEFAULT 'resident'"),
        ]

        for col_name, col_type in migrations:
            if col_name not in existing_columns:
                try:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column may already exist

        # Add indexes for new columns if they don't exist
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(access_count DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_promoted ON memories(promoted_at)")
        except sqlite3.OperationalError:
            pass

        # Phase 6: Migrate sessions table to add merkle_root column
        cursor = conn.execute("PRAGMA table_info(sessions)")
        sessions_columns = {row[1] for row in cursor.fetchall()}

        if "merkle_root" not in sessions_columns:
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN merkle_root TEXT")
            except sqlite3.OperationalError:
                pass  # Column may already exist

    # =========================================================================
    # Phase 2.1: Core Functions
    # =========================================================================

    def memory_save(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        emit_receipt: bool = True
    ) -> Tuple[str, Optional[CassetteReceipt]]:
        """Save a memory to the cassette.

        Embeds the text, stores the vector, and indexes for search.
        Auto-registers agent if not exists (Phase 3.1).
        Phase 6: Emits a CassetteReceipt for the write operation.

        Args:
            text: The memory content to save
            metadata: Optional metadata dictionary
            agent_id: Override default agent ID
            session_id: Optional session to associate with (Phase 3.2)
            emit_receipt: Whether to emit a receipt (Phase 6, default True)

        Returns:
            Tuple of (content-addressed hash, CassetteReceipt or None)
        """
        if not text or not text.strip():
            raise ValueError("Cannot save empty memory")

        agent_id = agent_id or self.agent_id
        session_id = session_id or self._current_session_id

        # Phase 3.1: Auto-register agent
        self._ensure_agent_registered(agent_id)

        # Compute content hash
        memory_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Generate embedding with Phase 6 normalization
        embedding_model = None
        if self.embedding_engine:
            embedding = self.embedding_engine.embed(text)
            # Phase 6.1: Normalize to L2 norm = 1.0 for determinism
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            vector_blob = self.embedding_engine.serialize(embedding)
            embedding_model = getattr(self.embedding_engine, 'model_name', 'all-MiniLM-L6-v2')
        else:
            # Fallback: zero vector if embeddings unavailable
            vector_blob = np.zeros(384, dtype=np.float32).tobytes()

        # Timestamps
        now = datetime.now(timezone.utc).isoformat()

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

        # Phase 6: Compute record hash for receipt (full MemoryRecord hash)
        record_hash = memory_hash  # Default to content hash
        if PHASE6_AVAILABLE:
            try:
                memory_record = create_memory_record(
                    text=text,
                    doc_path=f"resident:/{agent_id}/{memory_hash[:8]}",
                    tags=["resident", agent_id] if agent_id else ["resident"],
                    created_by="memory_cassette.py",
                    tool_version=self.schema_version,
                )
                record_hash = full_memory_hash(memory_record)
            except Exception:
                pass  # Fall back to content hash

        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        is_new_memory = False

        try:
            # Check if memory already exists
            cursor = conn.execute(
                "SELECT hash FROM memories WHERE hash = ?",
                (memory_hash,)
            )
            existing = cursor.fetchone()

            if existing:
                # Memory already exists - update indexed_at and access_count
                conn.execute("""
                    UPDATE memories SET
                        indexed_at = ?,
                        access_count = COALESCE(access_count, 0) + 1,
                        last_accessed = ?
                    WHERE hash = ?
                """, (now, now, memory_hash))
            else:
                is_new_memory = True
                # Insert new memory with Phase 3 columns
                conn.execute("""
                    INSERT INTO memories (hash, text, vector, metadata, created_at, agent_id, indexed_at, session_id, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (memory_hash, text, vector_blob, metadata_json, now, agent_id, now, session_id))

                # Insert into FTS index
                conn.execute(
                    "INSERT INTO memories_fts (text, hash) VALUES (?, ?)",
                    (text, memory_hash)
                )

                # Update agent memory count
                conn.execute("""
                    UPDATE agents SET memory_count = memory_count + 1, last_active = ?
                    WHERE agent_id = ?
                """, (now, agent_id))

                # Update session memory count if session exists
                if session_id:
                    conn.execute("""
                        UPDATE sessions SET memory_count = memory_count + 1, last_active = ?
                        WHERE session_id = ?
                    """, (now, session_id))

            conn.commit()

            # Phase 6.2: Emit receipt after successful commit
            receipt = None
            if emit_receipt and PHASE6_AVAILABLE and is_new_memory:
                receipt = self._emit_receipt(
                    conn=conn,
                    operation="SAVE",
                    record_id=memory_hash,
                    record_hash=record_hash,
                    agent_id=agent_id,
                    session_id=session_id,
                    text_length=len(text.encode('utf-8')),
                    embedding_model=embedding_model,
                )

            return memory_hash, receipt

        finally:
            conn.close()

    def _emit_receipt(
        self,
        conn: sqlite3.Connection,
        operation: str,
        record_id: str,
        record_hash: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        text_length: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> Optional[CassetteReceipt]:
        """Emit and store a CassetteReceipt for a write operation.

        Args:
            conn: Database connection
            operation: Operation type (SAVE, UPDATE, DELETE, etc.)
            record_id: Content hash of the record
            record_hash: Full hash of the MemoryRecord
            agent_id: Agent performing the operation
            session_id: Session during which operation occurred
            text_length: Length of text in bytes
            embedding_model: Model used for embedding

        Returns:
            CassetteReceipt or None if Phase 6 not available
        """
        if not PHASE6_AVAILABLE:
            return None

        try:
            # Create receipt with chain linkage
            receipt = create_receipt(
                cassette_id="resident",
                operation=operation,
                record_id=record_id,
                record_hash=record_hash,
                parent_receipt_hash=self._last_receipt_hash,
                receipt_index=self._receipt_index,
                agent_id=agent_id,
                session_id=session_id,
                text_length=text_length,
                embedding_model=embedding_model,
            )

            # Store receipt in database
            now = datetime.now(timezone.utc).isoformat()
            receipt_json = receipt.to_json(indent=None)

            conn.execute("""
                INSERT INTO cassette_receipts (
                    receipt_hash, receipt_json, cassette_id, operation,
                    record_id, record_hash, parent_receipt_hash,
                    receipt_index, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                receipt.receipt_hash,
                receipt_json,
                receipt.cassette_id,
                receipt.operation,
                receipt.record_id,
                receipt.record_hash,
                receipt.parent_receipt_hash,
                receipt.receipt_index,
                session_id,
                now,
            ))
            conn.commit()

            # Update chain state
            self._last_receipt_hash = receipt.receipt_hash
            self._receipt_index += 1
            self._session_receipts.append(receipt)

            return receipt

        except Exception as e:
            # Log error but don't fail the write
            print(f"[WARNING] Failed to emit receipt: {e}")
            return None

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

    def memory_recall(
        self,
        memory_hash: str,
        update_access: bool = True
    ) -> Optional[Dict]:
        """Retrieve full memory by hash.

        Args:
            memory_hash: Content-addressed hash of the memory
            update_access: Whether to update access stats (Phase 3.3)

        Returns:
            Dict with {hash, text, vector, metadata, created_at, agent_id, cassette,
                       access_count, last_accessed, session_id, promoted_at}
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

            # Phase 3.3: Update access stats
            if update_access:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    UPDATE memories SET
                        access_count = COALESCE(access_count, 0) + 1,
                        last_accessed = ?
                    WHERE hash = ?
                """, (now, memory_hash))
                conn.commit()

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
                "cassette": "resident",
                # Phase 3.3 fields
                "session_id": row['session_id'] if 'session_id' in row.keys() else None,
                "access_count": (row['access_count'] or 0) if 'access_count' in row.keys() else 0,
                "last_accessed": row['last_accessed'] if 'last_accessed' in row.keys() else None,
                "promoted_at": row['promoted_at'] if 'promoted_at' in row.keys() else None,
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
    # Phase 3.1: Agent Registry
    # =========================================================================

    def agent_register(
        self,
        agent_id: str,
        model_name: Optional[str] = None,
        display_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> Dict:
        """Register a new agent or update existing.

        Args:
            agent_id: Unique agent identifier (e.g., 'opus-20260111')
            model_name: Model identifier (e.g., 'claude-opus-4-5-20251101')
            display_name: Human-friendly name
            config: Agent-specific configuration

        Returns:
            Dict with {agent_id, created_at, is_new, last_active}
        """
        now = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(config) if config else None

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Check if agent exists
            cursor = conn.execute(
                "SELECT agent_id, created_at FROM agents WHERE agent_id = ?",
                (agent_id,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing agent
                conn.execute("""
                    UPDATE agents SET
                        last_active = ?,
                        model_name = COALESCE(?, model_name),
                        display_name = COALESCE(?, display_name),
                        config = COALESCE(?, config)
                    WHERE agent_id = ?
                """, (now, model_name, display_name, config_json, agent_id))
                conn.commit()

                return {
                    "agent_id": agent_id,
                    "created_at": existing['created_at'],
                    "last_active": now,
                    "is_new": False
                }
            else:
                # Insert new agent
                conn.execute("""
                    INSERT INTO agents (agent_id, model_name, display_name, created_at, last_active, config)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (agent_id, model_name, display_name, now, now, config_json))
                conn.commit()

                return {
                    "agent_id": agent_id,
                    "created_at": now,
                    "last_active": now,
                    "is_new": True
                }
        finally:
            conn.close()

    def agent_get(self, agent_id: str) -> Optional[Dict]:
        """Get agent info.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with agent info or None if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                "SELECT * FROM agents WHERE agent_id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "agent_id": row['agent_id'],
                "model_name": row['model_name'],
                "display_name": row['display_name'],
                "created_at": row['created_at'],
                "last_active": row['last_active'],
                "memory_count": row['memory_count'],
                "session_count": row['session_count'],
                "config": json.loads(row['config']) if row['config'] else None
            }
        finally:
            conn.close()

    def agent_list(self, model_filter: Optional[str] = None) -> List[Dict]:
        """List all registered agents.

        Args:
            model_filter: Optional filter by model name

        Returns:
            List of agent info dicts
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            if model_filter:
                cursor = conn.execute(
                    "SELECT * FROM agents WHERE model_name = ? ORDER BY last_active DESC",
                    (model_filter,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM agents ORDER BY last_active DESC"
                )

            return [
                {
                    "agent_id": row['agent_id'],
                    "model_name": row['model_name'],
                    "display_name": row['display_name'],
                    "created_at": row['created_at'],
                    "last_active": row['last_active'],
                    "memory_count": row['memory_count'],
                    "session_count": row['session_count']
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def _ensure_agent_registered(self, agent_id: str) -> None:
        """Ensure agent is registered, auto-register if needed."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT 1 FROM agents WHERE agent_id = ?",
                (agent_id,)
            )
            if not cursor.fetchone():
                # Auto-register with model inferred from agent_id
                model_name = agent_id.split('-')[0] if '-' in agent_id else None
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT INTO agents (agent_id, model_name, created_at, last_active)
                    VALUES (?, ?, ?, ?)
                """, (agent_id, model_name, now, now))
                conn.commit()
        finally:
            conn.close()

    def _update_agent_stats(self, agent_id: str) -> None:
        """Update agent memory and session counts."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Count memories
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_id = ?",
                (agent_id,)
            )
            memory_count = cursor.fetchone()[0]

            # Count sessions
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE agent_id = ?",
                (agent_id,)
            )
            session_count = cursor.fetchone()[0]

            # Update agent
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                UPDATE agents SET
                    memory_count = ?,
                    session_count = ?,
                    last_active = ?
                WHERE agent_id = ?
            """, (memory_count, session_count, now, agent_id))
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Phase 3.2: Session Continuity
    # =========================================================================

    def session_start(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        working_set: Optional[Dict] = None
    ) -> Dict:
        """Start a new session for an agent.

        Automatically registers agent if not exists.

        Args:
            agent_id: Agent identifier
            session_id: Optional session ID (auto-generated if not provided)
            working_set: Optional initial working set

        Returns:
            Dict with {session_id, agent_id, started_at, working_set}
        """
        # Ensure agent is registered
        self._ensure_agent_registered(agent_id)

        # Generate session ID if not provided
        session_id = session_id or f"{agent_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        now = datetime.now(timezone.utc).isoformat()
        working_set_json = json.dumps(working_set) if working_set else None

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT INTO sessions (session_id, agent_id, started_at, last_active, working_set)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, agent_id, now, now, working_set_json))

            # Increment agent session count
            conn.execute("""
                UPDATE agents SET session_count = session_count + 1, last_active = ?
                WHERE agent_id = ?
            """, (now, agent_id))

            conn.commit()

            # Track current session
            self._current_session_id = session_id

            return {
                "session_id": session_id,
                "agent_id": agent_id,
                "started_at": now,
                "working_set": working_set
            }
        finally:
            conn.close()

    def session_resume(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> Dict:
        """Resume an existing session or start new one.

        If session_id is None, resumes most recent active session or starts new.

        Args:
            agent_id: Agent identifier
            session_id: Optional specific session to resume
            limit: Max recent thoughts to return

        Returns:
            Dict with {agent_id, session_id, memory_count, recent_thoughts, working_set, last_active}
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Find session to resume
            if session_id:
                cursor = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ? AND agent_id = ?",
                    (session_id, agent_id)
                )
            else:
                # Get most recent session for agent (active or ended)
                cursor = conn.execute("""
                    SELECT * FROM sessions
                    WHERE agent_id = ?
                    ORDER BY last_active DESC
                    LIMIT 1
                """, (agent_id,))

            session = cursor.fetchone()

            # Start new session if none found or if previous session was ended
            if not session or session['ended_at']:
                # Get working set from previous session if any
                prev_working_set = None
                if session and session['working_set']:
                    prev_working_set = json.loads(session['working_set'])
                return self.session_start(agent_id, working_set=prev_working_set)

            # Resume existing session
            session_id = session['session_id']
            now = datetime.now(timezone.utc).isoformat()

            # Update last_active
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (now, session_id)
            )
            conn.commit()

            # Track current session
            self._current_session_id = session_id

            # Get recent thoughts for this agent
            recent_thoughts = self.memory_query("", limit=limit, agent_id=agent_id)

            # Get agent stats
            agent = self.agent_get(agent_id) or {}

            return {
                "agent_id": agent_id,
                "session_id": session_id,
                "memory_count": agent.get("memory_count", 0),
                "recent_thoughts": recent_thoughts,
                "working_set": json.loads(session['working_set']) if session['working_set'] else None,
                "last_active": now,
                "resumed": True
            }
        finally:
            conn.close()

    def session_update(
        self,
        session_id: str,
        working_set: Optional[Dict] = None,
        summary: Optional[str] = None
    ) -> Dict:
        """Update session state.

        Args:
            session_id: Session identifier
            working_set: New working set state
            summary: Optional session summary

        Returns:
            Dict with {session_id, updated_at}
        """
        now = datetime.now(timezone.utc).isoformat()
        working_set_json = json.dumps(working_set) if working_set else None

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE sessions SET
                    last_active = ?,
                    working_set = COALESCE(?, working_set),
                    summary = COALESCE(?, summary)
                WHERE session_id = ?
            """, (now, working_set_json, summary, session_id))
            conn.commit()

            return {
                "session_id": session_id,
                "updated_at": now
            }
        finally:
            conn.close()

    def session_end(self, session_id: str, summary: Optional[str] = None) -> Dict:
        """End a session.

        Phase 6: Computes and stores Merkle root of all session receipts.

        Args:
            session_id: Session identifier
            summary: Optional summary of what was accomplished

        Returns:
            Dict with {session_id, ended_at, duration_minutes, memory_count, merkle_root}
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Get session info
            cursor = conn.execute(
                "SELECT started_at, memory_count FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            session = cursor.fetchone()

            if not session:
                return {"error": "Session not found", "session_id": session_id}

            # Calculate duration
            started = datetime.fromisoformat(session['started_at'].replace('Z', '+00:00'))
            ended = datetime.fromisoformat(now.replace('Z', '+00:00'))
            duration_minutes = (ended - started).total_seconds() / 60

            # Phase 6.2: Compute Merkle root of session receipts
            merkle_root = None
            if PHASE6_AVAILABLE and self._session_receipts:
                receipt_hashes = [r.receipt_hash for r in self._session_receipts]
                merkle_root = compute_session_merkle_root(receipt_hashes)

            # Update session (with merkle_root if available)
            if merkle_root:
                conn.execute("""
                    UPDATE sessions SET
                        ended_at = ?,
                        last_active = ?,
                        summary = COALESCE(?, summary),
                        merkle_root = ?
                    WHERE session_id = ?
                """, (now, now, summary, merkle_root, session_id))
            else:
                conn.execute("""
                    UPDATE sessions SET
                        ended_at = ?,
                        last_active = ?,
                        summary = COALESCE(?, summary)
                    WHERE session_id = ?
                """, (now, now, summary, session_id))
            conn.commit()

            # Clear current session and reset receipt chain
            if self._current_session_id == session_id:
                self._current_session_id = None
                # Reset receipt chain for next session
                self._session_receipts = []

            return {
                "session_id": session_id,
                "ended_at": now,
                "duration_minutes": round(duration_minutes, 2),
                "memory_count": session['memory_count'],
                "merkle_root": merkle_root,
            }
        finally:
            conn.close()

    def session_history(self, agent_id: str, limit: int = 10) -> List[Dict]:
        """Get session history for an agent.

        Args:
            agent_id: Agent identifier
            limit: Max sessions to return

        Returns:
            List of {session_id, started_at, ended_at, memory_count, summary}
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute("""
                SELECT session_id, started_at, ended_at, memory_count, summary, working_set
                FROM sessions
                WHERE agent_id = ?
                ORDER BY started_at DESC
                LIMIT ?
            """, (agent_id, limit))

            return [
                {
                    "session_id": row['session_id'],
                    "started_at": row['started_at'],
                    "ended_at": row['ended_at'],
                    "memory_count": row['memory_count'],
                    "summary": row['summary'],
                    "has_working_set": row['working_set'] is not None
                }
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # =========================================================================
    # Phase 4.1: Pointer Caching (SPC Integration)
    # =========================================================================

    def pointer_register(
        self,
        pointer: str,
        pointer_type: str,
        target_hash: Optional[str] = None,
        qualifiers: Optional[Dict] = None,
        codebook_id: str = "ags-codebook"
    ) -> Dict:
        """Register a resolved pointer in the cache.

        Per SPC_SPEC Section 2 - Pointer Types:
        - SYMBOL_PTR: @GOV:PREAMBLE (single-token)
        - HASH_PTR: sha256:abc123... (content-addressed)
        - COMPOSITE_PTR: @GOV:PREAMBLE:lines=1-10 (pointer + qualifiers)

        Args:
            pointer: The SPC pointer string (e.g., "C3", "sha256:abc...")
            pointer_type: One of "symbol", "hash", "composite"
            target_hash: SHA-256 hash of resolved content (optional)
            qualifiers: Dict of typed qualifiers for composite pointers
            codebook_id: Codebook ID for versioning

        Returns:
            Dict with {pointer_id, created_at, pointer_type}
        """
        import hashlib

        now = datetime.now(timezone.utc).isoformat()
        pointer_id = hashlib.sha256(f"{pointer}:{codebook_id}".encode()).hexdigest()[:16]
        qualifiers_json = json.dumps(qualifiers) if qualifiers else None

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT OR REPLACE INTO pointers
                (pointer_id, pointer_type, base_ptr, target_hash, qualifiers, codebook_id, created_at, resolved_count, last_resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (pointer_id, pointer_type, pointer, target_hash, qualifiers_json, codebook_id, now, now))
            conn.commit()

            return {
                "pointer_id": pointer_id,
                "pointer": pointer,
                "pointer_type": pointer_type,
                "created_at": now
            }
        finally:
            conn.close()

    def pointer_lookup(
        self,
        pointer: str,
        codebook_id: str = "ags-codebook"
    ) -> Optional[Dict]:
        """Look up a cached pointer.

        Args:
            pointer: The SPC pointer to look up
            codebook_id: Codebook ID for versioning

        Returns:
            Dict with cached pointer info or None if not found
        """
        import hashlib

        pointer_id = hashlib.sha256(f"{pointer}:{codebook_id}".encode()).hexdigest()[:16]

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                "SELECT * FROM pointers WHERE pointer_id = ? AND codebook_id = ?",
                (pointer_id, codebook_id)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Update access stats
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                UPDATE pointers SET resolved_count = resolved_count + 1, last_resolved = ?
                WHERE pointer_id = ?
            """, (now, pointer_id))
            conn.commit()

            return {
                "pointer_id": row["pointer_id"],
                "pointer": row["base_ptr"],
                "pointer_type": row["pointer_type"],
                "target_hash": row["target_hash"],
                "qualifiers": json.loads(row["qualifiers"]) if row["qualifiers"] else None,
                "codebook_id": row["codebook_id"],
                "created_at": row["created_at"],
                "resolved_count": row["resolved_count"],
                "last_resolved": row["last_resolved"]
            }
        finally:
            conn.close()

    def pointer_invalidate(
        self,
        codebook_id: Optional[str] = None,
        pointer: Optional[str] = None
    ) -> Dict:
        """Invalidate cached pointers.

        Per CODEBOOK_SYNC_PROTOCOL: Codebook change → invalidate cache.

        Args:
            codebook_id: Invalidate all pointers for this codebook (None = all)
            pointer: Invalidate specific pointer (None = all matching codebook)

        Returns:
            Dict with {invalidated_count}
        """
        conn = sqlite3.connect(str(self.db_path))

        try:
            if pointer and codebook_id:
                import hashlib
                pointer_id = hashlib.sha256(f"{pointer}:{codebook_id}".encode()).hexdigest()[:16]
                cursor = conn.execute(
                    "DELETE FROM pointers WHERE pointer_id = ?",
                    (pointer_id,)
                )
            elif codebook_id:
                cursor = conn.execute(
                    "DELETE FROM pointers WHERE codebook_id = ?",
                    (codebook_id,)
                )
            else:
                cursor = conn.execute("DELETE FROM pointers")

            count = cursor.rowcount
            conn.commit()

            return {"invalidated_count": count}
        finally:
            conn.close()

    def pointer_stats(self) -> Dict:
        """Get pointer cache statistics.

        Returns:
            Dict with cache statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Total pointers
            cursor = conn.execute("SELECT COUNT(*) as count FROM pointers")
            total = cursor.fetchone()["count"]

            # By type
            cursor = conn.execute("""
                SELECT pointer_type, COUNT(*) as count
                FROM pointers GROUP BY pointer_type
            """)
            by_type = {row["pointer_type"]: row["count"] for row in cursor.fetchall()}

            # By codebook
            cursor = conn.execute("""
                SELECT codebook_id, COUNT(*) as count
                FROM pointers GROUP BY codebook_id
            """)
            by_codebook = {row["codebook_id"]: row["count"] for row in cursor.fetchall()}

            # Most resolved
            cursor = conn.execute("""
                SELECT base_ptr, pointer_type, resolved_count
                FROM pointers ORDER BY resolved_count DESC LIMIT 10
            """)
            top_resolved = [
                {"pointer": row["base_ptr"], "type": row["pointer_type"], "count": row["resolved_count"]}
                for row in cursor.fetchall()
            ]

            return {
                "total_pointers": total,
                "by_type": by_type,
                "by_codebook": by_codebook,
                "top_resolved": top_resolved
            }
        finally:
            conn.close()

    # =========================================================================
    # Phase 6.3: Cartridge Export/Import (Restore Guarantee)
    # =========================================================================

    def export_cartridge(self, output_dir: Path) -> Dict[str, Any]:
        """Export cassette as a portable cartridge.

        Creates a directory with:
        - records.jsonl: All memory records (one per line)
        - receipts.jsonl: All receipts in chain order
        - manifest.json: Cartridge metadata with Merkle roots

        Args:
            output_dir: Directory to write cartridge files

        Returns:
            Manifest dict with {cassette_id, record_count, receipt_count, merkle_roots}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Export records
            cursor = conn.execute("""
                SELECT hash, text, metadata, created_at, agent_id, session_id
                FROM memories ORDER BY created_at ASC
            """)
            records = []
            record_hashes = []
            with open(output_dir / "records.jsonl", "w", encoding="utf-8") as f:
                for row in cursor:
                    record = {
                        "id": row["hash"],
                        "text": row["text"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                        "created_at": row["created_at"],
                        "agent_id": row["agent_id"],
                        "session_id": row["session_id"],
                    }
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                    records.append(record)
                    record_hashes.append(row["hash"])

            # Export receipts
            cursor = conn.execute("""
                SELECT receipt_json FROM cassette_receipts
                ORDER BY receipt_index ASC
            """)
            receipt_hashes = []
            with open(output_dir / "receipts.jsonl", "w", encoding="utf-8") as f:
                for row in cursor:
                    f.write(row["receipt_json"] + "\n")
                    data = json.loads(row["receipt_json"])
                    receipt_hashes.append(data.get("receipt_hash", ""))

            # Compute Merkle roots
            content_merkle_root = None
            receipt_merkle_root = None

            if PHASE6_AVAILABLE:
                if record_hashes:
                    content_merkle_root = compute_session_merkle_root(record_hashes)
                if receipt_hashes:
                    receipt_merkle_root = compute_session_merkle_root(receipt_hashes)

            # Write manifest
            now = datetime.now(timezone.utc).isoformat()
            manifest = {
                "cartridge_version": "1.0.0",
                "cassette_id": "resident",
                "schema_version": self.schema_version,
                "record_count": len(records),
                "receipt_count": len(receipt_hashes),
                "content_merkle_root": content_merkle_root,
                "receipt_merkle_root": receipt_merkle_root,
                "exported_at": now,
            }

            with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            return manifest

        finally:
            conn.close()

    def import_cartridge(self, cartridge_dir: Path) -> Dict[str, Any]:
        """Import a cartridge and restore cassette state.

        Validates integrity before importing:
        - Verifies content Merkle root matches
        - Verifies receipt chain integrity

        Args:
            cartridge_dir: Directory containing cartridge files

        Returns:
            Dict with {restored_records, restored_receipts, merkle_verified}
        """
        cartridge_dir = Path(cartridge_dir)

        # Load manifest
        manifest_path = cartridge_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Load records
        records_path = cartridge_dir / "records.jsonl"
        records = []
        record_hashes = []
        if records_path.exists():
            with open(records_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    records.append(record)
                    record_hashes.append(record["id"])

        # Verify content Merkle root
        merkle_verified = True
        if PHASE6_AVAILABLE and manifest.get("content_merkle_root") and record_hashes:
            computed_merkle = compute_session_merkle_root(record_hashes)
            if computed_merkle != manifest["content_merkle_root"]:
                raise ValueError(
                    f"Content Merkle root mismatch: "
                    f"expected={manifest['content_merkle_root']}, computed={computed_merkle}"
                )

        # Load receipts
        receipts_path = cartridge_dir / "receipts.jsonl"
        receipts = []
        if receipts_path.exists():
            with open(receipts_path, "r", encoding="utf-8") as f:
                for line in f:
                    receipt_data = json.loads(line.strip())
                    receipts.append(receipt_data)

        # Verify receipt chain
        if PHASE6_AVAILABLE and receipts:
            receipt_objects = [receipt_from_dict(r) for r in receipts]
            chain_result = verify_receipt_chain(receipt_objects, verify_hashes=True)
            if not chain_result["valid"]:
                raise ValueError(f"Receipt chain invalid: {chain_result['errors']}")

        # Restore records to database
        conn = sqlite3.connect(str(self.db_path))
        restored_records = 0
        restored_receipts = 0

        try:
            for record in records:
                # Check if already exists
                cursor = conn.execute(
                    "SELECT hash FROM memories WHERE hash = ?",
                    (record["id"],)
                )
                if cursor.fetchone():
                    continue  # Skip existing

                # Generate embedding for restored record
                if self.embedding_engine:
                    embedding = self.embedding_engine.embed(record["text"])
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    vector_blob = self.embedding_engine.serialize(embedding)
                else:
                    vector_blob = np.zeros(384, dtype=np.float32).tobytes()

                # Insert record
                metadata_json = json.dumps(record.get("metadata")) if record.get("metadata") else None
                conn.execute("""
                    INSERT INTO memories (hash, text, vector, metadata, created_at, agent_id, indexed_at, session_id, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    record["id"],
                    record["text"],
                    vector_blob,
                    metadata_json,
                    record.get("created_at", datetime.now(timezone.utc).isoformat()),
                    record.get("agent_id"),
                    datetime.now(timezone.utc).isoformat(),
                    record.get("session_id"),
                ))

                # Insert into FTS
                conn.execute(
                    "INSERT INTO memories_fts (text, hash) VALUES (?, ?)",
                    (record["text"], record["id"])
                )

                restored_records += 1

            # Restore receipts
            for receipt_data in receipts:
                receipt_hash = receipt_data.get("receipt_hash")
                cursor = conn.execute(
                    "SELECT receipt_hash FROM cassette_receipts WHERE receipt_hash = ?",
                    (receipt_hash,)
                )
                if cursor.fetchone():
                    continue  # Skip existing

                receipt_json = json.dumps(receipt_data, sort_keys=True, ensure_ascii=False)
                conn.execute("""
                    INSERT INTO cassette_receipts (
                        receipt_hash, receipt_json, cassette_id, operation,
                        record_id, record_hash, parent_receipt_hash,
                        receipt_index, session_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    receipt_hash,
                    receipt_json,
                    receipt_data.get("cassette_id", "resident"),
                    receipt_data.get("operation", "RESTORE"),
                    receipt_data.get("record_id"),
                    receipt_data.get("record_hash"),
                    receipt_data.get("parent_receipt_hash"),
                    receipt_data.get("receipt_index"),
                    receipt_data.get("session_id"),
                    datetime.now(timezone.utc).isoformat(),
                ))
                restored_receipts += 1

            conn.commit()

            return {
                "restored_records": restored_records,
                "restored_receipts": restored_receipts,
                "merkle_verified": merkle_verified,
                "manifest": manifest,
            }

        finally:
            conn.close()

    def restore_from_receipts(
        self,
        receipts: List[Dict[str, Any]],
        source_records: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Restore cassette state from a receipt chain and source records.

        This is the core restore-from-receipts function for Phase 6.3.

        Args:
            receipts: Ordered list of receipt dicts (oldest first)
            source_records: Map of record_id -> record dict with text

        Returns:
            Dict with {restored_count, final_merkle_root, errors}
        """
        if not PHASE6_AVAILABLE:
            return {"error": "Phase 6 not available", "restored_count": 0}

        # Verify receipt chain first
        receipt_objects = [receipt_from_dict(r) for r in receipts]
        chain_result = verify_receipt_chain(receipt_objects, verify_hashes=True)

        if not chain_result["valid"]:
            return {
                "error": f"Invalid receipt chain: {chain_result['errors']}",
                "restored_count": 0,
            }

        conn = sqlite3.connect(str(self.db_path))
        restored_count = 0
        errors = []

        try:
            for receipt in receipts:
                operation = receipt.get("operation")
                record_id = receipt.get("record_id")

                if operation == "SAVE":
                    # Get source record
                    source = source_records.get(record_id)
                    if not source:
                        errors.append(f"Missing source record for {record_id}")
                        continue

                    # Check if already exists
                    cursor = conn.execute(
                        "SELECT hash FROM memories WHERE hash = ?",
                        (record_id,)
                    )
                    if cursor.fetchone():
                        continue  # Already exists

                    # Generate embedding
                    text = source.get("text", "")
                    if self.embedding_engine:
                        embedding = self.embedding_engine.embed(text)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        vector_blob = self.embedding_engine.serialize(embedding)
                    else:
                        vector_blob = np.zeros(384, dtype=np.float32).tobytes()

                    # Insert
                    metadata_json = json.dumps(source.get("metadata")) if source.get("metadata") else None
                    conn.execute("""
                        INSERT INTO memories (hash, text, vector, metadata, created_at, agent_id, indexed_at, access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """, (
                        record_id,
                        text,
                        vector_blob,
                        metadata_json,
                        source.get("created_at", datetime.now(timezone.utc).isoformat()),
                        source.get("agent_id"),
                        datetime.now(timezone.utc).isoformat(),
                    ))

                    # FTS
                    conn.execute(
                        "INSERT INTO memories_fts (text, hash) VALUES (?, ?)",
                        (text, record_id)
                    )

                    restored_count += 1

                elif operation == "DELETE":
                    conn.execute("DELETE FROM memories WHERE hash = ?", (record_id,))
                    conn.execute("DELETE FROM memories_fts WHERE hash = ?", (record_id,))

            conn.commit()

            return {
                "restored_count": restored_count,
                "final_merkle_root": chain_result["merkle_root"],
                "errors": errors,
            }

        finally:
            conn.close()

    def cas_lookup(self, hash_value: str) -> Optional[Dict]:
        """Look up content by hash (CAS interface for SPC decoder).

        This is the callback function for spc_decoder.register_cas_lookup().

        Args:
            hash_value: SHA-256 hash (without 'sha256:' prefix)

        Returns:
            Dict with {text, metadata, type, source} or None if not found
        """
        # First check memories table
        memory = self.memory_recall(hash_value, update_access=True)
        if memory:
            return {
                "text": memory["text"],
                "metadata": memory.get("metadata"),
                "type": "memory",
                "source": "memory_cassette"
            }

        # Could extend to check other content sources here
        return None

    # =========================================================================
    # Phase 3.3: Cross-Session Memory
    # =========================================================================

    def memory_promote(
        self,
        memory_hash: str,
        from_cassette: str = "inbox"
    ) -> Dict:
        """Promote a memory from INBOX to RESIDENT.

        Args:
            memory_hash: Hash of the memory to promote
            from_cassette: Source cassette (for tracking)

        Returns:
            Dict with {hash, promoted_at, source_cassette}
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE memories SET
                    promoted_at = ?,
                    source_cassette = ?
                WHERE hash = ?
            """, (now, from_cassette, memory_hash))
            conn.commit()

            return {
                "hash": memory_hash,
                "promoted_at": now,
                "source_cassette": from_cassette
            }
        finally:
            conn.close()

    def memory_demote(
        self,
        memory_hash: str,
        to_cassette: str = "inbox"
    ) -> Dict:
        """Demote a memory (clear promoted status).

        Args:
            memory_hash: Hash of the memory to demote
            to_cassette: Target cassette (for tracking)

        Returns:
            Dict with {hash, demoted_at, target_cassette}
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                UPDATE memories SET
                    promoted_at = NULL,
                    source_cassette = ?
                WHERE hash = ?
            """, (to_cassette, memory_hash))
            conn.commit()

            return {
                "hash": memory_hash,
                "demoted_at": now,
                "target_cassette": to_cassette
            }
        finally:
            conn.close()

    def get_promotion_candidates(
        self,
        agent_id: Optional[str] = None,
        min_access_count: int = 2,
        min_age_hours: float = 1.0
    ) -> List[Dict]:
        """Get memories eligible for promotion.

        Criteria:
        - Access count >= min_access_count
        - Age >= min_age_hours
        - Not already promoted

        Args:
            agent_id: Optional filter by agent
            min_access_count: Minimum access count
            min_age_hours: Minimum age in hours

        Returns:
            List of {hash, text_preview, access_count, age_hours, created_at}
        """
        cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=min_age_hours)).isoformat()

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            if agent_id:
                cursor = conn.execute("""
                    SELECT hash, text, access_count, created_at
                    FROM memories
                    WHERE agent_id = ?
                      AND promoted_at IS NULL
                      AND access_count >= ?
                      AND created_at <= ?
                    ORDER BY access_count DESC, created_at ASC
                """, (agent_id, min_access_count, cutoff_time))
            else:
                cursor = conn.execute("""
                    SELECT hash, text, access_count, created_at, agent_id
                    FROM memories
                    WHERE promoted_at IS NULL
                      AND access_count >= ?
                      AND created_at <= ?
                    ORDER BY access_count DESC, created_at ASC
                """, (min_access_count, cutoff_time))

            now = datetime.now(timezone.utc)
            results = []
            for row in cursor.fetchall():
                created = datetime.fromisoformat(row['created_at'].replace('Z', '+00:00'))
                age_hours = (now - created).total_seconds() / 3600

                results.append({
                    "hash": row['hash'],
                    "text_preview": row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                    "access_count": row['access_count'] or 0,
                    "age_hours": round(age_hours, 2),
                    "created_at": row['created_at']
                })

            return results
        finally:
            conn.close()

    # =========================================================================
    # Standard Cassette Interface
    # =========================================================================

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Standard cassette query interface."""
        return self.memory_query(query_text, limit=top_k)

    def get_stats(self) -> Dict:
        """Return cassette statistics (enhanced for Phase 3)."""
        stats = {
            "cassette_id": self.cassette_id,
            "name": "Resident (AI Memories)",
            "description": "Per-agent AI memories with vector search and session continuity",
            "capabilities": self.capabilities,
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "schema_version": self.schema_version
        }

        if not self.db_path.exists():
            return stats

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

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

            # Phase 3.1: Agent stats
            cursor = conn.execute("SELECT COUNT(*) FROM agents")
            stats["total_agents"] = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT agent_id, model_name, memory_count, session_count, last_active
                FROM agents
                ORDER BY last_active DESC
                LIMIT 10
            """)
            stats["agents"] = [
                {
                    "agent_id": row['agent_id'],
                    "model_name": row['model_name'],
                    "memory_count": row['memory_count'],
                    "session_count": row['session_count'],
                    "last_active": row['last_active']
                }
                for row in cursor.fetchall()
            ]

            # Phase 3.2: Session stats
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL")
            stats["active_sessions"] = cursor.fetchone()[0]

            # Phase 3.3: Promotion stats
            cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE promoted_at IS NOT NULL")
            promoted_count = cursor.fetchone()[0]
            stats["promotion_stats"] = {
                "promoted": promoted_count,
                "pending": stats["total_memories"] - promoted_count
            }

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


def memory_save(
    text: str,
    metadata: Optional[Dict] = None,
    agent_id: str = "default",
    return_receipt: bool = False
) -> str:
    """Save a memory. Returns hash (or tuple of hash and receipt if return_receipt=True)."""
    cassette = get_memory_cassette(agent_id)
    result = cassette.memory_save(text, metadata, agent_id)
    if return_receipt:
        return result  # Returns (hash, receipt)
    else:
        return result[0]  # Returns just the hash for backwards compatibility


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
# Phase 3 Convenience Functions
# ============================================================================

def agent_register(agent_id: str, model_name: str = None) -> Dict:
    """Register an agent. Returns agent info."""
    cassette = get_memory_cassette()
    return cassette.agent_register(agent_id, model_name)


def agent_get(agent_id: str) -> Optional[Dict]:
    """Get agent info."""
    cassette = get_memory_cassette()
    return cassette.agent_get(agent_id)


def agent_list(model_filter: str = None) -> List[Dict]:
    """List all agents."""
    cassette = get_memory_cassette()
    return cassette.agent_list(model_filter)


def session_start(agent_id: str, working_set: Dict = None) -> Dict:
    """Start a new session."""
    cassette = get_memory_cassette(agent_id)
    return cassette.session_start(agent_id, working_set=working_set)


def session_resume(agent_id: str, limit: int = 10) -> Dict:
    """Resume session with recent thoughts."""
    cassette = get_memory_cassette(agent_id)
    return cassette.session_resume(agent_id, limit=limit)


def session_update(session_id: str, working_set: Dict = None, summary: str = None) -> Dict:
    """Update session state."""
    cassette = get_memory_cassette()
    return cassette.session_update(session_id, working_set, summary)


def session_end(session_id: str, summary: str = None) -> Dict:
    """End a session."""
    cassette = get_memory_cassette()
    return cassette.session_end(session_id, summary)


def session_history(agent_id: str, limit: int = 10) -> List[Dict]:
    """Get session history for an agent."""
    cassette = get_memory_cassette()
    return cassette.session_history(agent_id, limit)


def memory_promote(memory_hash: str, from_cassette: str = "inbox") -> Dict:
    """Promote a memory from INBOX to RESIDENT."""
    cassette = get_memory_cassette()
    return cassette.memory_promote(memory_hash, from_cassette)


def get_promotion_candidates(agent_id: str = None, min_access: int = 2, min_age_hours: float = 1.0) -> List[Dict]:
    """Get memories eligible for promotion."""
    cassette = get_memory_cassette()
    return cassette.get_promotion_candidates(agent_id, min_access, min_age_hours)


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
