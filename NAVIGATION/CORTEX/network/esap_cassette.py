#!/usr/bin/env python3
"""
ESAP-Enabled Cassette - Adds spectral alignment to cassettes.

Integrates the ESAP handshake protocol for cross-model semantic alignment
verification between cassettes.
"""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import sqlite3
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "THOUGHT/LAB/VECTOR_ELO/eigen-alignment"))

from lib.handshake import (
    ESAPHandshake,
    SpectrumCompact,
    compute_cumulative_variance,
    compute_effective_rank,
    check_convergence,
    CONVERGENCE_THRESHOLD,
)


@dataclass
class CassetteSpectrum:
    """Spectrum signature for a cassette's vector space."""

    eigenvalues: np.ndarray
    cumulative_variance: np.ndarray
    effective_rank: float
    vector_count: int
    vector_dim: int
    anchor_hash: str

    @classmethod
    def from_vectors(cls, vectors: np.ndarray, anchor_hash: str) -> 'CassetteSpectrum':
        """Compute spectrum from vectors.

        Args:
            vectors: (n_samples, dim) array of vectors
            anchor_hash: Hash identifying the anchor set

        Returns:
            CassetteSpectrum with computed eigenvalues
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors for spectrum computation")

        # Center vectors
        centered = vectors - vectors.mean(axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Compute eigenvalues (descending)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]

        cumulative_variance = compute_cumulative_variance(eigenvalues)
        effective_rank = compute_effective_rank(eigenvalues)

        return cls(
            eigenvalues=eigenvalues,
            cumulative_variance=cumulative_variance,
            effective_rank=effective_rank,
            vector_count=len(vectors),
            vector_dim=vectors.shape[1],
            anchor_hash=anchor_hash
        )

    def to_compact(self, embedder_id: str = "") -> SpectrumCompact:
        """Convert to ESAP SpectrumCompact format."""
        return SpectrumCompact.from_eigenvalues(
            eigenvalues=self.eigenvalues,
            anchor_set_hash=self.anchor_hash,
            embedder_id=embedder_id
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "eigenvalues_top_k": self.eigenvalues[:64].tolist(),
            "cumulative_variance": self.cumulative_variance[:64].tolist(),
            "effective_rank": self.effective_rank,
            "vector_count": self.vector_count,
            "vector_dim": self.vector_dim,
            "anchor_hash": self.anchor_hash
        }


class ESAPCassetteMixin(ABC):
    """Mixin that adds ESAP capabilities to any DatabaseCassette.

    Usage:
        class MyVectorCassette(ESAPCassetteMixin, DatabaseCassette):
            def get_vectors_for_spectrum(self) -> np.ndarray:
                # Return vectors from your database
                ...
    """

    _spectrum: Optional[CassetteSpectrum] = None
    _esap_handler: Optional[ESAPHandshake] = None

    def get_vectors_for_spectrum(self) -> Optional[np.ndarray]:
        """Override to return vectors for spectrum computation.

        Returns:
            (n_samples, dim) array or None if not available
        """
        return None

    def get_embedder_id(self) -> str:
        """Override to return embedder model identifier."""
        return getattr(self, 'embedder_id', 'unknown')

    def compute_spectrum(self, force: bool = False) -> Optional[CassetteSpectrum]:
        """Compute spectrum signature from vectors.

        Args:
            force: Recompute even if cached

        Returns:
            CassetteSpectrum or None if no vectors available
        """
        if self._spectrum is not None and not force:
            return self._spectrum

        vectors = self.get_vectors_for_spectrum()
        if vectors is None or len(vectors) < 2:
            return None

        # Compute anchor hash from vectors
        vector_bytes = vectors.tobytes()
        anchor_hash = f"sha256:{hashlib.sha256(vector_bytes).hexdigest()}"

        self._spectrum = CassetteSpectrum.from_vectors(vectors, anchor_hash)
        return self._spectrum

    def get_esap_handler(self) -> Optional[ESAPHandshake]:
        """Get ESAP handshake handler for this cassette."""
        if self._esap_handler is not None:
            return self._esap_handler

        spectrum = self.compute_spectrum()
        if spectrum is None:
            return None

        compact = spectrum.to_compact(self.get_embedder_id())
        cassette_id = getattr(self, 'cassette_id', 'unknown')
        capabilities = getattr(self, 'capabilities', [])

        self._esap_handler = ESAPHandshake(
            agent_id=cassette_id,
            spectrum=compact,
            capabilities=capabilities
        )
        return self._esap_handler

    def esap_handshake(self) -> dict:
        """Extended handshake with ESAP spectrum signature.

        Extends the base handshake() with spectrum information.
        """
        # Get base handshake from parent class
        base = super().handshake() if hasattr(super(), 'handshake') else {}

        spectrum = self.compute_spectrum()
        if spectrum:
            base["esap"] = {
                "enabled": True,
                "spectrum": spectrum.to_dict(),
                "convergence_threshold": CONVERGENCE_THRESHOLD
            }
        else:
            base["esap"] = {"enabled": False, "reason": "no_vectors"}

        return base

    def verify_alignment(self, other_spectrum: dict) -> dict:
        """Verify spectral alignment with another cassette.

        Args:
            other_spectrum: Spectrum dict from another cassette's esap_handshake

        Returns:
            Convergence check result
        """
        my_spectrum = self.compute_spectrum()
        if my_spectrum is None:
            return {"converges": False, "reason": "no_local_spectrum"}

        their_cv = np.array(other_spectrum.get("cumulative_variance", []))
        their_df = other_spectrum.get("effective_rank", 0)

        if len(their_cv) == 0:
            return {"converges": False, "reason": "no_remote_spectrum"}

        return check_convergence(
            my_spectrum.cumulative_variance,
            their_cv,
            my_spectrum.effective_rank,
            their_df
        )


class VectorCassetteBase(ESAPCassetteMixin):
    """Base class for cassettes with vector storage.

    Provides default implementation for extracting vectors from SQLite.
    """

    db_path: Path
    vector_table: str = "section_vectors"
    vector_column: str = "embedding"

    def get_vectors_for_spectrum(self) -> Optional[np.ndarray]:
        """Extract vectors from database."""
        if not hasattr(self, 'db_path') or not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            if self.vector_table not in tables:
                conn.close()
                return None

            # Get vector column (blob or text)
            cursor = conn.execute(f"SELECT {self.vector_column} FROM {self.vector_table} LIMIT 1000")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            # Parse vectors (assumes blob or JSON format)
            vectors = []
            for row in rows:
                if row[0] is None:
                    continue
                if isinstance(row[0], bytes):
                    vec = np.frombuffer(row[0], dtype=np.float32)
                else:
                    import json
                    vec = np.array(json.loads(row[0]), dtype=np.float32)
                vectors.append(vec)

            if not vectors:
                return None

            return np.array(vectors)

        except Exception as e:
            print(f"[ESAP] Error extracting vectors: {e}")
            return None
