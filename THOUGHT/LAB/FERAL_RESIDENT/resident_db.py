"""
Feral Resident Database Schema (A.1.2)

SQLite persistence for quantum state tracking.
Tracks vectors, interactions, threads with Df evolution.
"""

import sqlite3
import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorRecord:
    """Record of a stored vector"""
    vector_id: str
    vec_blob: bytes
    vec_sha256: str
    Df: float
    composition_op: str
    parent_ids: List[str]
    created_at: str

    @property
    def vector(self) -> np.ndarray:
        """Deserialize vector from blob"""
        return np.frombuffer(self.vec_blob, dtype=np.float32)


@dataclass
class InteractionRecord:
    """Record of an interaction"""
    interaction_id: str
    thread_id: str
    input_text: str
    output_text: str
    input_vector_id: str
    output_vector_id: str
    mind_hash: str
    mind_Df: float
    distance_from_start: float
    created_at: str


@dataclass
class ThreadRecord:
    """Record of a thread"""
    thread_id: str
    mind_vector_id: Optional[str]
    interaction_count: int
    current_Df: float
    created_at: str
    updated_at: str


class ResidentDB:
    """
    SQLite database for Feral Resident state persistence.

    Tables:
    - vectors: All GeometricState vectors with composition history
    - interactions: Q/A pairs with mind state snapshots
    - threads: Long-running conversation threads
    """

    SCHEMA = """
    -- Vectors table: stores all GeometricState instances
    CREATE TABLE IF NOT EXISTS vectors (
        vector_id TEXT PRIMARY KEY,
        vec_blob BLOB NOT NULL,
        vec_sha256 TEXT NOT NULL,
        Df REAL,
        composition_op TEXT,  -- 'initialize' | 'entangle' | 'superpose' | 'project' | 'add' | 'subtract'
        parent_ids JSON,      -- List of parent vector IDs for composition
        created_at TEXT NOT NULL
    );

    -- Index for vector hash lookups
    CREATE INDEX IF NOT EXISTS idx_vectors_sha256 ON vectors(vec_sha256);

    -- Interactions table: tracks Q/A with mind state
    CREATE TABLE IF NOT EXISTS interactions (
        interaction_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        input_text TEXT,
        output_text TEXT,
        input_vector_id TEXT,
        output_vector_id TEXT,
        mind_hash TEXT,
        mind_Df REAL,
        distance_from_start REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (thread_id) REFERENCES threads(thread_id),
        FOREIGN KEY (input_vector_id) REFERENCES vectors(vector_id),
        FOREIGN KEY (output_vector_id) REFERENCES vectors(vector_id)
    );

    -- Index for thread lookups
    CREATE INDEX IF NOT EXISTS idx_interactions_thread ON interactions(thread_id);

    -- Threads table: long-running conversation state
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        mind_vector_id TEXT,
        interaction_count INTEGER DEFAULT 0,
        current_Df REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (mind_vector_id) REFERENCES vectors(vector_id)
    );

    -- Receipts table: operation receipts for provenance
    CREATE TABLE IF NOT EXISTS receipts (
        receipt_id TEXT PRIMARY KEY,
        operation TEXT NOT NULL,
        input_hashes JSON NOT NULL,
        output_hash TEXT NOT NULL,
        metadata JSON,
        created_at TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str = "feral_resident.db"):
        """Initialize database connection and schema"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist"""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()

    # =========================================================================
    # Vector Operations
    # =========================================================================

    def store_vector(
        self,
        vector: np.ndarray,
        Df: float,
        composition_op: str = 'initialize',
        parent_ids: Optional[List[str]] = None
    ) -> str:
        """
        Store a vector and return its ID.

        Args:
            vector: The numpy array (should be unit normalized)
            Df: Participation ratio
            composition_op: How this vector was created
            parent_ids: IDs of parent vectors if composed

        Returns:
            vector_id: Unique ID for this vector
        """
        vec_blob = vector.astype(np.float32).tobytes()
        vec_sha256 = hashlib.sha256(vec_blob).hexdigest()

        # Check if vector already exists (content-addressed)
        existing = self.conn.execute(
            "SELECT vector_id FROM vectors WHERE vec_sha256 = ?",
            (vec_sha256,)
        ).fetchone()

        if existing:
            return existing['vector_id']

        vector_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO vectors (vector_id, vec_blob, vec_sha256, Df, composition_op, parent_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (vector_id, vec_blob, vec_sha256, Df, composition_op,
             json.dumps(parent_ids or []), now)
        )
        self.conn.commit()

        return vector_id

    def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Retrieve a vector by ID"""
        row = self.conn.execute(
            "SELECT * FROM vectors WHERE vector_id = ?",
            (vector_id,)
        ).fetchone()

        if not row:
            return None

        return VectorRecord(
            vector_id=row['vector_id'],
            vec_blob=row['vec_blob'],
            vec_sha256=row['vec_sha256'],
            Df=row['Df'],
            composition_op=row['composition_op'],
            parent_ids=json.loads(row['parent_ids']),
            created_at=row['created_at']
        )

    def get_vector_by_hash(self, vec_sha256: str) -> Optional[VectorRecord]:
        """Retrieve a vector by its SHA256 hash"""
        row = self.conn.execute(
            "SELECT * FROM vectors WHERE vec_sha256 = ?",
            (vec_sha256,)
        ).fetchone()

        if not row:
            return None

        return VectorRecord(
            vector_id=row['vector_id'],
            vec_blob=row['vec_blob'],
            vec_sha256=row['vec_sha256'],
            Df=row['Df'],
            composition_op=row['composition_op'],
            parent_ids=json.loads(row['parent_ids']),
            created_at=row['created_at']
        )

    def get_all_vectors(self, limit: int = 1000) -> List[VectorRecord]:
        """Get all vectors (for nearest neighbor search)"""
        rows = self.conn.execute(
            "SELECT * FROM vectors ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()

        return [
            VectorRecord(
                vector_id=row['vector_id'],
                vec_blob=row['vec_blob'],
                vec_sha256=row['vec_sha256'],
                Df=row['Df'],
                composition_op=row['composition_op'],
                parent_ids=json.loads(row['parent_ids']),
                created_at=row['created_at']
            )
            for row in rows
        ]

    def vector_count(self) -> int:
        """Count total vectors"""
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    # =========================================================================
    # Thread Operations
    # =========================================================================

    def create_thread(self, thread_id: str) -> str:
        """Create a new thread"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT OR IGNORE INTO threads (thread_id, interaction_count, current_Df, created_at, updated_at)
            VALUES (?, 0, 0.0, ?, ?)
            """,
            (thread_id, now, now)
        )
        self.conn.commit()
        return thread_id

    def get_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        """Get thread by ID"""
        row = self.conn.execute(
            "SELECT * FROM threads WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()

        if not row:
            return None

        return ThreadRecord(
            thread_id=row['thread_id'],
            mind_vector_id=row['mind_vector_id'],
            interaction_count=row['interaction_count'],
            current_Df=row['current_Df'] or 0.0,
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    def update_thread(
        self,
        thread_id: str,
        mind_vector_id: str,
        current_Df: float
    ):
        """Update thread state after interaction"""
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            UPDATE threads
            SET mind_vector_id = ?,
                current_Df = ?,
                interaction_count = interaction_count + 1,
                updated_at = ?
            WHERE thread_id = ?
            """,
            (mind_vector_id, current_Df, now, thread_id)
        )
        self.conn.commit()

    def list_threads(self) -> List[ThreadRecord]:
        """List all threads"""
        rows = self.conn.execute(
            "SELECT * FROM threads ORDER BY updated_at DESC"
        ).fetchall()

        return [
            ThreadRecord(
                thread_id=row['thread_id'],
                mind_vector_id=row['mind_vector_id'],
                interaction_count=row['interaction_count'],
                current_Df=row['current_Df'] or 0.0,
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            for row in rows
        ]

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    def store_interaction(
        self,
        thread_id: str,
        input_text: str,
        output_text: str,
        input_vector_id: str,
        output_vector_id: str,
        mind_hash: str,
        mind_Df: float,
        distance_from_start: float
    ) -> str:
        """Store an interaction record"""
        interaction_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO interactions
            (interaction_id, thread_id, input_text, output_text,
             input_vector_id, output_vector_id, mind_hash, mind_Df,
             distance_from_start, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (interaction_id, thread_id, input_text, output_text,
             input_vector_id, output_vector_id, mind_hash, mind_Df,
             distance_from_start, now)
        )
        self.conn.commit()

        return interaction_id

    def get_thread_interactions(
        self,
        thread_id: str,
        limit: int = 100
    ) -> List[InteractionRecord]:
        """Get interactions for a thread"""
        rows = self.conn.execute(
            """
            SELECT * FROM interactions
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (thread_id, limit)
        ).fetchall()

        return [
            InteractionRecord(
                interaction_id=row['interaction_id'],
                thread_id=row['thread_id'],
                input_text=row['input_text'],
                output_text=row['output_text'],
                input_vector_id=row['input_vector_id'],
                output_vector_id=row['output_vector_id'],
                mind_hash=row['mind_hash'],
                mind_Df=row['mind_Df'],
                distance_from_start=row['distance_from_start'],
                created_at=row['created_at']
            )
            for row in rows
        ]

    def interaction_count(self, thread_id: Optional[str] = None) -> int:
        """Count interactions (optionally for a specific thread)"""
        if thread_id:
            return self.conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE thread_id = ?",
                (thread_id,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    # =========================================================================
    # Receipt Operations (Provenance)
    # =========================================================================

    def store_receipt(
        self,
        operation: str,
        input_hashes: List[str],
        output_hash: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store an operation receipt"""
        receipt_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO receipts (receipt_id, operation, input_hashes, output_hash, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (receipt_id, operation, json.dumps(input_hashes), output_hash,
             json.dumps(metadata or {}), now)
        )
        self.conn.commit()

        return receipt_id

    def get_receipt_chain(self, output_hash: str) -> List[Dict]:
        """Get the chain of receipts leading to an output"""
        receipts = []
        current_hash = output_hash

        while True:
            row = self.conn.execute(
                "SELECT * FROM receipts WHERE output_hash = ?",
                (current_hash,)
            ).fetchone()

            if not row:
                break

            receipt = {
                'receipt_id': row['receipt_id'],
                'operation': row['operation'],
                'input_hashes': json.loads(row['input_hashes']),
                'output_hash': row['output_hash'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at']
            }
            receipts.append(receipt)

            # Follow the chain
            input_hashes = receipt['input_hashes']
            if input_hashes:
                current_hash = input_hashes[0]
            else:
                break

        return receipts

    # =========================================================================
    # Stats & Diagnostics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'vector_count': self.vector_count(),
            'interaction_count': self.interaction_count(),
            'thread_count': len(self.list_threads()),
            'receipt_count': self.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
            'db_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
        }

    def get_Df_history(self, thread_id: str) -> List[Tuple[str, float]]:
        """Get Df evolution history for a thread"""
        rows = self.conn.execute(
            """
            SELECT created_at, mind_Df FROM interactions
            WHERE thread_id = ?
            ORDER BY created_at ASC
            """,
            (thread_id,)
        ).fetchall()

        return [(row['created_at'], row['mind_Df']) for row in rows]

    def vacuum(self):
        """Compact the database"""
        self.conn.execute("VACUUM")

    # =========================================================================
    # Backup & Restore
    # =========================================================================

    def backup_to(self, backup_path: str):
        """Backup database to another file"""
        backup_conn = sqlite3.connect(backup_path)
        self.conn.backup(backup_conn)
        backup_conn.close()

    def export_thread_receipts(self, thread_id: str) -> List[Dict]:
        """Export all receipts for a thread (for corrupt-and-restore)"""
        interactions = self.get_thread_interactions(thread_id, limit=10000)
        receipts = []

        for interaction in interactions:
            # Get vector receipts
            if interaction.input_vector_id:
                vec = self.get_vector(interaction.input_vector_id)
                if vec:
                    receipts.append({
                        'type': 'vector',
                        'vector_id': vec.vector_id,
                        'sha256': vec.vec_sha256,
                        'Df': vec.Df,
                        'composition_op': vec.composition_op,
                        'parent_ids': vec.parent_ids
                    })

            # Get operation receipts
            chain = self.get_receipt_chain(interaction.mind_hash)
            receipts.extend([{'type': 'receipt', **r} for r in chain])

        return receipts


# ============================================================================
# Testing
# ============================================================================

def example_usage():
    """Demonstrate database usage"""
    import tempfile
    import os

    print("=== ResidentDB Example ===\n")

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_feral.db")
        db = ResidentDB(db_path)

        # Create a thread
        thread_id = "eternal"
        db.create_thread(thread_id)
        print(f"Created thread: {thread_id}")

        # Store some vectors
        v1 = np.random.randn(384).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)  # Normalize
        vid1 = db.store_vector(v1, Df=22.5, composition_op='initialize')
        print(f"Stored vector: {vid1}")

        v2 = np.random.randn(384).astype(np.float32)
        v2 = v2 / np.linalg.norm(v2)
        vid2 = db.store_vector(v2, Df=23.1, composition_op='entangle', parent_ids=[vid1])
        print(f"Stored composed vector: {vid2}")

        # Store an interaction
        iid = db.store_interaction(
            thread_id=thread_id,
            input_text="What is authentication?",
            output_text="[Feral Alpha] E=0.456, Df=22.5",
            input_vector_id=vid1,
            output_vector_id=vid2,
            mind_hash="abc123",
            mind_Df=23.1,
            distance_from_start=0.15
        )
        print(f"Stored interaction: {iid}")

        # Update thread
        db.update_thread(thread_id, vid2, 23.1)

        # Get stats
        stats = db.get_stats()
        print(f"\nStats: {json.dumps(stats, indent=2)}")

        # Get thread
        thread = db.get_thread(thread_id)
        print(f"\nThread: {thread}")

        db.close()
        print("\nDone!")


if __name__ == "__main__":
    example_usage()
