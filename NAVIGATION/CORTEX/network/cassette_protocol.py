#!/usr/bin/env python3
"""
Cassette Protocol - Base interface for all database cassettes.

Defines the standard contract for cassettes in the Semantic Network.

Phase 4 additions:
- sync_tuple: Codebook synchronization per CODEBOOK_SYNC_PROTOCOL
- blanket_status: Markov blanket alignment per Q35
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import sqlite3


# Codebook configuration (Phase 4)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"
KERNEL_VERSION = "1.0.0"
TOKENIZER_ID = "tiktoken/o200k_base"


class DatabaseCassette(ABC):
    """Base class for all database cassettes.

    All cassettes must implement these methods to participate
    in the Semantic Network.

    Phase 4 additions:
    - sync_tuple for codebook synchronization
    - blanket_status for Markov blanket alignment
    """

    def __init__(self, db_path: Path, cassette_id: str):
        self.db_path = db_path
        self.cassette_id = cassette_id
        self.capabilities: List[str] = []
        self.schema_version = "1.0"
        self._codebook_cache: Optional[Dict] = None
        self._codebook_hash_cache: Optional[str] = None

    def _compute_codebook_hash(self) -> str:
        """Compute SHA-256 of canonical codebook JSON.

        Per CODEBOOK_SYNC_PROTOCOL, the hash is computed over the
        canonical (sorted keys, minimal separators) JSON representation.
        """
        if self._codebook_hash_cache is not None:
            return self._codebook_hash_cache

        if not CODEBOOK_PATH.exists():
            return ""

        try:
            with open(CODEBOOK_PATH, 'r', encoding='utf-8') as f:
                codebook = json.load(f)
            # Sort keys recursively for determinism
            canonical = json.dumps(codebook, sort_keys=True, separators=(',', ':'))
            self._codebook_hash_cache = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            self._codebook_cache = codebook
            return self._codebook_hash_cache
        except (IOError, json.JSONDecodeError):
            return ""

    def _get_codebook_version(self) -> str:
        """Get codebook semver from CODEBOOK.json."""
        if self._codebook_cache is not None:
            return self._codebook_cache.get('version', '0.0.0')

        if not CODEBOOK_PATH.exists():
            return "0.0.0"

        try:
            with open(CODEBOOK_PATH, 'r', encoding='utf-8') as f:
                codebook = json.load(f)
            self._codebook_cache = codebook
            return codebook.get('version', '0.0.0')
        except (IOError, json.JSONDecodeError):
            return "0.0.0"

    def get_sync_tuple(self) -> Dict:
        """Return sync tuple for codebook synchronization.

        Per CODEBOOK_SYNC_PROTOCOL Section 6.4:
        - codebook_id: Identifier for the codebook
        - codebook_sha256: Hash for mismatch detection
        - codebook_semver: Version for compatibility checking
        - kernel_version: Decoder kernel version
        - tokenizer_id: Tokenizer for token counting
        """
        return {
            "codebook_id": "ags-codebook",
            "codebook_sha256": self._compute_codebook_hash(),
            "codebook_semver": self._get_codebook_version(),
            "kernel_version": KERNEL_VERSION,
            "tokenizer_id": TOKENIZER_ID
        }

    def get_blanket_status(self) -> str:
        """Return Markov blanket status.

        Per Q35 (Markov Blankets):
        - ALIGNED: R > τ, stable blanket, semantic transfer permitted
        - DISSOLVED: R < τ, blanket broken, resync required
        - PENDING: R ≈ τ, boundary forming, awaiting confirmation
        - UNSYNCED: No sync attempted yet

        Default implementation returns ALIGNED if codebook hash is valid.
        """
        if self._compute_codebook_hash():
            return "ALIGNED"
        return "UNSYNCED"

    def handshake(self) -> Dict:
        """Return cassette metadata for network registration.

        Called during cassette registration to advertise capabilities
        and verify database integrity.

        Phase 4 additions:
        - sync_tuple: For codebook synchronization
        - blanket_status: For Markov blanket alignment (Q35)
        """
        return {
            "cassette_id": self.cassette_id,
            "db_path": str(self.db_path),
            "db_hash": self._compute_hash(),
            "capabilities": self.capabilities,
            "schema_version": self.schema_version,
            "stats": self.get_stats(),
            # Phase 4 additions
            "sync_tuple": self.get_sync_tuple(),
            "blanket_status": self.get_blanket_status()
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
