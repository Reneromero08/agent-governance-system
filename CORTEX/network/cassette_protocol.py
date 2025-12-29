#!/usr/bin/env python3
"""
Cassette Protocol - Base interface for all database cassettes.

Defines the standard contract for cassettes in the Semantic Network.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import hashlib
import sqlite3


class DatabaseCassette(ABC):
    """Base class for all database cassettes.

    All cassettes must implement these methods to participate
    in the Semantic Network.
    """

    def __init__(self, db_path: Path, cassette_id: str):
        self.db_path = db_path
        self.cassette_id = cassette_id
        self.capabilities: List[str] = []
        self.schema_version = "1.0"

    def handshake(self) -> Dict:
        """Return cassette metadata for network registration.

        Called during cassette registration to advertise capabilities
        and verify database integrity.
        """
        return {
            "cassette_id": self.cassette_id,
            "db_path": str(self.db_path),
            "db_hash": self._compute_hash(),
            "capabilities": self.capabilities,
            "schema_version": self.schema_version,
            "stats": self.get_stats()
        }

    @abstractmethod
    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Execute query and return results.

        Args:
            query_text: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of result dictionaries with:
                - content: Matched content
                - score: Relevance/similarity score
                - metadata: Additional cassette-specific metadata
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Return cassette statistics.

        Should include:
            - total_chunks: Number of chunks in database
            - Other cassette-specific stats
        """
        pass

    def _compute_hash(self) -> str:
        """Compute DB content hash for verification."""
        if not self.db_path.exists():
            return ""

        with open(self.db_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
