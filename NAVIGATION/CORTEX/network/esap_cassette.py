#!/usr/bin/env python3
"""
ESAP Cassette Mixin - Eigen-Spectrum Alignment Protocol integration.

Enables cross-model semantic alignment via eigenvalue spectrum invariance.
Based on validated research: r=0.99+ correlation across embedding models.

Reference:
- THOUGHT/LAB/VECTOR_ELO/research/vector-substrate/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md
- THOUGHT/LAB/VECTOR_ELO/eigen-alignment/PROTOCOL_SPEC.md
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np


class ESAPCassetteMixin:
    """Mixin providing ESAP spectrum computation for cassettes.

    Computes eigenvalue spectrum from cassette vectors to enable
    cross-model alignment verification.

    The key insight (from validated research): eigenvalue spectrums
    are invariant across embedding models with r=0.99+ correlation.
    This allows us to verify semantic alignment without requiring
    identical embedding models.
    """

    # Configuration
    ESAP_TOP_K = 10  # Number of top eigenvalues to track
    ALIGNMENT_THRESHOLD = 0.95  # Correlation threshold for alignment

    def compute_spectrum_signature(self, vectors: np.ndarray) -> Dict:
        """Compute eigenvalue spectrum signature from vectors.

        Args:
            vectors: Matrix of shape (n_vectors, embedding_dim)

        Returns:
            Spectrum signature dict with eigenvalues, variance, effective rank
        """
        if vectors is None or vectors.shape[0] < 2:
            return self._empty_spectrum()

        # Center vectors
        centered = vectors - vectors.mean(axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Handle 1D case
        if cov.ndim == 0:
            return self._empty_spectrum()

        # Eigendecomposition
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        # Filter negative eigenvalues (numerical precision)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Compute cumulative variance
        total_var = eigenvalues.sum()
        if total_var > 0:
            cumulative = np.cumsum(eigenvalues) / total_var
        else:
            cumulative = np.zeros_like(eigenvalues)

        # Effective rank (entropy-based)
        if total_var > 0:
            probs = eigenvalues / total_var
            probs = probs[probs > 0]  # Filter zeros
            effective_rank = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
        else:
            effective_rank = 0.0

        # Hash for quick comparison
        top_k = eigenvalues[:self.ESAP_TOP_K].tolist()
        anchor_hash = hashlib.sha256(json.dumps(top_k).encode()).hexdigest()[:16]

        return {
            "eigenvalues_top_k": top_k,
            "cumulative_variance": cumulative[:self.ESAP_TOP_K].tolist(),
            "effective_rank": float(effective_rank),
            "anchor_hash": f"sha256:{anchor_hash}",
            "n_vectors": vectors.shape[0],
            "embedding_dim": vectors.shape[1]
        }

    def _empty_spectrum(self) -> Dict:
        """Return empty spectrum for cassettes with insufficient vectors."""
        return {
            "eigenvalues_top_k": [],
            "cumulative_variance": [],
            "effective_rank": 0.0,
            "anchor_hash": "sha256:empty",
            "n_vectors": 0,
            "embedding_dim": 0
        }

    def esap_handshake(self) -> Dict:
        """Extended handshake including ESAP spectrum signature.

        Overrides or extends base handshake() to include spectral info.
        """
        base = self.handshake() if hasattr(self, 'handshake') else {}

        # Get all vectors from cassette
        vectors = self._get_all_vectors()
        spectrum = self.compute_spectrum_signature(vectors)

        base["esap"] = {
            "enabled": True,
            "spectrum": spectrum,
            "alignment_threshold": self.ALIGNMENT_THRESHOLD
        }

        return base

    def _get_all_vectors(self) -> np.ndarray:
        """Retrieve all vectors from cassette.

        Override in subclass to provide actual vectors.

        Returns:
            numpy array of shape (n_vectors, embedding_dim)
        """
        raise NotImplementedError("Subclass must implement _get_all_vectors()")

    @staticmethod
    def compute_spectrum_correlation(spec_a: Dict, spec_b: Dict) -> float:
        """Compute correlation between two spectrum signatures.

        Args:
            spec_a: First spectrum signature
            spec_b: Second spectrum signature

        Returns:
            Correlation coefficient (0-1), where 1 = perfect alignment
        """
        eigs_a = np.array(spec_a.get("eigenvalues_top_k", []))
        eigs_b = np.array(spec_b.get("eigenvalues_top_k", []))

        if len(eigs_a) == 0 or len(eigs_b) == 0:
            return 0.0

        # Pad to same length
        max_len = max(len(eigs_a), len(eigs_b))
        eigs_a = np.pad(eigs_a, (0, max_len - len(eigs_a)))
        eigs_b = np.pad(eigs_b, (0, max_len - len(eigs_b)))

        # Pearson correlation
        if np.std(eigs_a) == 0 or np.std(eigs_b) == 0:
            return 1.0 if np.allclose(eigs_a, eigs_b) else 0.0

        correlation = np.corrcoef(eigs_a, eigs_b)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    @staticmethod
    def verify_alignment(spec_a: Dict, spec_b: Dict, threshold: float = 0.95) -> Dict:
        """Verify if two spectrums are aligned.

        Args:
            spec_a: First spectrum signature
            spec_b: Second spectrum signature
            threshold: Correlation threshold for alignment

        Returns:
            Dict with aligned (bool), correlation, threshold
        """
        correlation = ESAPCassetteMixin.compute_spectrum_correlation(spec_a, spec_b)

        return {
            "aligned": correlation >= threshold,
            "correlation": correlation,
            "threshold": threshold,
            "anchor_a": spec_a.get("anchor_hash", ""),
            "anchor_b": spec_b.get("anchor_hash", "")
        }
