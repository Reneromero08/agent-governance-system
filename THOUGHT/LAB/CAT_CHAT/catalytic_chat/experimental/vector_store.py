#!/usr/bin/env python3
"""
Experimental Vector Store (Phase 2.5 Sandbox)

SQLite-backed vector store for local experiments and tests.
NOT part of canonical Phase 2 or Phase 3.
"""

import hashlib
import json
import sqlite3
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from typing_extensions import deprecated


class VectorStoreError(Exception):
    pass


@dataclass
class VectorResult:
    vector_id: str
    namespace: str
    content_hash: str
    dims: int
    vector: List[float]
    meta: Dict[str, Any]
    created_at: str
    score: float


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same dimensions")
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def _content_hash(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


class VectorStore:
    
    SCHEMA_VERSION = 1
    
    def __init__(self, repo_root: Optional[Path] = None, db_path: Optional[Path] = None):
        if db_path is not None:
            self.db_path = db_path
        else:
            if repo_root is None:
                repo_root = Path.cwd()
            self.db_path = repo_root / "CORTEX" / "db" / "system1.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                vector_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                dims INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_namespace 
            ON vectors(namespace)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_content_hash 
            ON vectors(content_hash)
        """)
        
        conn.commit()
    
    def put_vector(self, namespace: str, content_bytes: bytes, vector: List[float], meta: Dict[str, Any]) -> str:
        if not vector:
            raise VectorStoreError("Vector cannot be empty")
        
        dims = len(vector)
        if dims == 0:
            raise VectorStoreError("Vector dimension cannot be zero")
        
        content_hash = _content_hash(content_bytes)
        vector_id = hashlib.sha256(f"{namespace}:{content_hash}".encode()).hexdigest()[:16]
        
        vector_json = json.dumps(vector)
        meta_json = json.dumps(meta)
        created_at = datetime.now(timezone.utc).isoformat()
        
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO vectors 
            (vector_id, namespace, content_hash, dims, vector_json, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (vector_id, namespace, content_hash, dims, vector_json, meta_json, created_at))
        conn.commit()
        
        return vector_id
    
    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM vectors WHERE vector_id = ?",
            (vector_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            "vector_id": row["vector_id"],
            "namespace": row["namespace"],
            "content_hash": row["content_hash"],
            "dims": row["dims"],
            "vector": json.loads(row["vector_json"]),
            "meta": json.loads(row["meta_json"]),
            "created_at": row["created_at"]
        }
    
    def query_topk(self, namespace: str, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not query_vector:
            raise VectorStoreError("Query vector cannot be empty")
        
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM vectors WHERE namespace = ?",
            (namespace,)
        )
        
        results = []
        for row in cursor.fetchall():
            stored_vector = json.loads(row["vector_json"])
            
            if len(stored_vector) != len(query_vector):
                continue
            
            score = _cosine_similarity(query_vector, stored_vector)
            
            results.append({
                "vector_id": row["vector_id"],
                "namespace": row["namespace"],
                "content_hash": row["content_hash"],
                "dims": row["dims"],
                "vector": stored_vector,
                "meta": json.loads(row["meta_json"]),
                "created_at": row["created_at"],
                "score": score
            })
        
        results.sort(key=lambda x: (-x["score"], x["vector_id"]))
        return results[:k]
    
    def delete_namespace(self, namespace: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM vectors WHERE namespace = ?",
            (namespace,)
        )
        conn.commit()
        return cursor.rowcount
    
    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
