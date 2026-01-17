"""Alignment Key for Cross-Model Vector Communication.

The AlignmentKey is the shared secret that enables two parties to
communicate meaning through vectors. Anyone with the key can:

1. Encode text to a 48D vector
2. Decode a 48D vector back to text (by matching against candidates)
3. Share the key file with others
4. Align with other keys to enable cross-model communication

Usage:
    # Create a key for your model
    key = AlignmentKey.create("my-model", model.encode)

    # Encode text to 48 numbers
    vector = key.encode("Hello world", model.encode)
    # vector = array([0.23, -0.15, 0.87, ...])  # 48 floats

    # Share the numbers through any channel...
    # JSON, clipboard, voice, network, etc.

    # Decode received vector (requires candidate list)
    match, score = key.decode(vector, candidates, model.encode)

Cross-model communication:
    # Two parties with different models
    key_a = AlignmentKey.create("MiniLM", model_a.encode)
    key_b = AlignmentKey.create("MPNet", model_b.encode)

    # Align the keys
    pair = key_a.align_with(key_b)

    # Now they can communicate bidirectionally
    vec = pair.encode_a_to_b("Hello", model_a.encode)
    match = pair.decode_at_b(vec, candidates, model_b.encode)

Key insight: The eigenvalue spectrum is INVARIANT across models,
meaning the geometric "shape" of semantic space is universal.
Only orientation differs - Procrustes finds the rotation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Union
import json
import numpy as np
from scipy.stats import spearmanr

from .mds import squared_distance_matrix, classical_mds
from .procrustes import procrustes_align, out_of_sample_mds, cosine_similarity
from .canonical_anchors import CANONICAL_128, compute_anchor_hash


# Type alias for embedding function
EmbedFn = Callable[[List[str]], np.ndarray]


@dataclass
class AlignmentKey:
    """The key that enables cross-model vector communication.

    Contains all the precomputed data needed to project text
    into the universal MDS space and back.

    Attributes:
        model_id: Identifier for the embedding model
        anchor_set: List of anchor words used
        anchor_hash: SHA-256[:16] of the anchor set
        eigenvalues: (k,) eigenvalue spectrum from MDS
        eigenvectors: (n_anchors, k) eigenvectors from MDS
        anchor_embeddings: (n_anchors, dim) normalized anchor embeddings
        D2: (n_anchors, n_anchors) squared distance matrix
        k: Number of MDS dimensions
    """
    model_id: str
    anchor_set: List[str]
    anchor_hash: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    anchor_embeddings: np.ndarray
    D2: np.ndarray
    k: int

    @classmethod
    def create(
        cls,
        model_id: str,
        embed_fn: EmbedFn,
        anchors: Optional[List[str]] = None,
        k: int = 48
    ) -> 'AlignmentKey':
        """Create an alignment key for a model.

        Args:
            model_id: Identifier for the model (e.g., "all-MiniLM-L6-v2")
            embed_fn: Function that takes List[str] and returns (n, dim) embeddings
            anchors: Anchor set to use (default: CANONICAL_128)
            k: Number of MDS dimensions (default: 48)

        Returns:
            AlignmentKey ready for encoding/decoding
        """
        if anchors is None:
            anchors = CANONICAL_128

        # Embed and normalize anchors
        embeddings = embed_fn(anchors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Compute squared distance matrix
        D2 = squared_distance_matrix(embeddings)

        # Classical MDS
        mds_coords, eigenvalues, eigenvectors = classical_mds(D2, k=k)

        # Actual k might be less if not enough positive eigenvalues
        actual_k = len(eigenvalues)

        return cls(
            model_id=model_id,
            anchor_set=anchors,
            anchor_hash=compute_anchor_hash(anchors),
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            anchor_embeddings=embeddings,
            D2=D2,
            k=actual_k
        )

    def encode(self, text: str, embed_fn: EmbedFn) -> np.ndarray:
        """Encode text to a k-dimensional vector.

        Args:
            text: Text to encode
            embed_fn: Embedding function for this model

        Returns:
            (k,) MDS coordinates for the text
        """
        # Embed and normalize
        v = embed_fn([text])[0]
        v = v / np.linalg.norm(v)

        # Compute squared distances to anchors
        d2 = 2 * (1 - self.anchor_embeddings @ v)

        # Project via Gower's formula
        y = out_of_sample_mds(
            d2.reshape(1, -1),
            self.D2,
            self.eigenvectors,
            self.eigenvalues
        )[0]

        return y

    def decode(
        self,
        vector: np.ndarray,
        candidates: List[str],
        embed_fn: EmbedFn
    ) -> Tuple[str, float]:
        """Decode a vector to the best-matching candidate.

        Args:
            vector: (k,) MDS coordinates received
            candidates: List of possible texts to match against
            embed_fn: Embedding function for this model

        Returns:
            Tuple of (best_match, similarity_score)
        """
        best_match = None
        best_score = -float('inf')

        for cand in candidates:
            y = self.encode(cand, embed_fn)
            k = min(len(y), len(vector))
            sim = cosine_similarity(y[:k], vector[:k])

            if sim > best_score:
                best_score = sim
                best_match = cand

        return best_match, best_score

    def decode_all(
        self,
        vector: np.ndarray,
        candidates: List[str],
        embed_fn: EmbedFn
    ) -> List[Tuple[str, float]]:
        """Decode a vector and return all candidates with scores.

        Args:
            vector: (k,) MDS coordinates received
            candidates: List of possible texts to match against
            embed_fn: Embedding function for this model

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        scores = []
        for cand in candidates:
            y = self.encode(cand, embed_fn)
            k = min(len(y), len(vector))
            sim = cosine_similarity(y[:k], vector[:k])
            scores.append((cand, sim))

        return sorted(scores, key=lambda x: -x[1])

    def align_with(self, other: 'AlignmentKey') -> 'AlignedKeyPair':
        """Align this key with another to enable cross-model communication.

        Args:
            other: Another AlignmentKey (possibly for different model)

        Returns:
            AlignedKeyPair for bidirectional communication

        Raises:
            ValueError: If anchor hashes don't match
        """
        if self.anchor_hash != other.anchor_hash:
            raise ValueError(
                f"Anchor hash mismatch: {self.anchor_hash} vs {other.anchor_hash}. "
                "Both keys must use the same anchor set."
            )

        # Use minimum k
        k = min(self.k, other.k)

        # Compute MDS coordinates (these are stored in eigenvectors * sqrt(eigenvalues))
        X_self = self.eigenvectors[:, :k] * np.sqrt(self.eigenvalues[:k])
        X_other = other.eigenvectors[:, :k] * np.sqrt(other.eigenvalues[:k])

        # Compute spectrum correlation
        corr, _ = spearmanr(self.eigenvalues[:k], other.eigenvalues[:k])

        # Procrustes alignment - both directions
        R_self_to_other, residual_fwd = procrustes_align(X_self, X_other)
        R_other_to_self, residual_rev = procrustes_align(X_other, X_self)

        return AlignedKeyPair(
            key_a=self,
            key_b=other,
            R_a_to_b=R_self_to_other,
            R_b_to_a=R_other_to_self,
            spectrum_correlation=float(corr),
            procrustes_residual=float(residual_fwd),
            k=k
        )

    def to_file(self, path: Union[str, Path]) -> None:
        """Export key to file for sharing.

        Creates two files:
        - {path}.json: Metadata
        - {path}.npz: Numpy arrays

        Args:
            path: Base path (without extension)
        """
        path = Path(path)

        # Save metadata as JSON
        meta = {
            "model_id": self.model_id,
            "anchor_set": self.anchor_set,
            "anchor_hash": self.anchor_hash,
            "k": self.k,
            "version": "1.0.0"
        }
        with open(str(path) + ".json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        # Save arrays as NPZ
        np.savez_compressed(
            str(path) + ".npz",
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            anchor_embeddings=self.anchor_embeddings,
            D2=self.D2
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'AlignmentKey':
        """Load key from file.

        Args:
            path: Base path (without extension)

        Returns:
            Loaded AlignmentKey
        """
        path = Path(path)

        # Load metadata
        with open(str(path) + ".json", 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Load arrays
        arrays = np.load(str(path) + ".npz")

        return cls(
            model_id=meta["model_id"],
            anchor_set=meta["anchor_set"],
            anchor_hash=meta["anchor_hash"],
            eigenvalues=arrays["eigenvalues"],
            eigenvectors=arrays["eigenvectors"],
            anchor_embeddings=arrays["anchor_embeddings"],
            D2=arrays["D2"],
            k=meta["k"]
        )


@dataclass
class AlignedKeyPair:
    """Two alignment keys aligned for bidirectional communication.

    Created by AlignmentKey.align_with(). Enables sending messages
    between two different embedding models.

    Attributes:
        key_a: First key (e.g., MiniLM)
        key_b: Second key (e.g., MPNet)
        R_a_to_b: Rotation matrix from A's space to B's space
        R_b_to_a: Rotation matrix from B's space to A's space
        spectrum_correlation: How similar the eigenvalue spectra are
        procrustes_residual: Alignment error (lower is better)
        k: Shared dimension count
    """
    key_a: AlignmentKey
    key_b: AlignmentKey
    R_a_to_b: np.ndarray
    R_b_to_a: np.ndarray
    spectrum_correlation: float
    procrustes_residual: float
    k: int

    def encode_a_to_b(self, text: str, embed_fn_a: EmbedFn) -> np.ndarray:
        """Encode text at A, rotated for reception at B.

        Args:
            text: Text to send
            embed_fn_a: Model A's embedding function

        Returns:
            (k,) vector in B's coordinate system
        """
        y = self.key_a.encode(text, embed_fn_a)
        return y[:self.k] @ self.R_a_to_b[:self.k, :self.k]

    def encode_b_to_a(self, text: str, embed_fn_b: EmbedFn) -> np.ndarray:
        """Encode text at B, rotated for reception at A.

        Args:
            text: Text to send
            embed_fn_b: Model B's embedding function

        Returns:
            (k,) vector in A's coordinate system
        """
        y = self.key_b.encode(text, embed_fn_b)
        return y[:self.k] @ self.R_b_to_a[:self.k, :self.k]

    def decode_at_a(
        self,
        vector: np.ndarray,
        candidates: List[str],
        embed_fn_a: EmbedFn
    ) -> Tuple[str, float]:
        """Decode a vector at A (vector should be in A's space).

        Args:
            vector: (k,) vector in A's coordinate system
            candidates: Possible messages
            embed_fn_a: Model A's embedding function

        Returns:
            Tuple of (best_match, score)
        """
        return self.key_a.decode(vector, candidates, embed_fn_a)

    def decode_at_b(
        self,
        vector: np.ndarray,
        candidates: List[str],
        embed_fn_b: EmbedFn
    ) -> Tuple[str, float]:
        """Decode a vector at B (vector should be in B's space).

        Args:
            vector: (k,) vector in B's coordinate system
            candidates: Possible messages
            embed_fn_b: Model B's embedding function

        Returns:
            Tuple of (best_match, score)
        """
        return self.key_b.decode(vector, candidates, embed_fn_b)

    def send_and_receive(
        self,
        text: str,
        candidates: List[str],
        embed_fn_a: EmbedFn,
        embed_fn_b: EmbedFn,
        direction: str = "a_to_b"
    ) -> Tuple[np.ndarray, str, float]:
        """Full send/receive cycle for testing.

        Args:
            text: Text to send
            candidates: Candidate pool for decoding
            embed_fn_a: Model A's embedding function
            embed_fn_b: Model B's embedding function
            direction: "a_to_b" or "b_to_a"

        Returns:
            Tuple of (vector, best_match, score)
        """
        if direction == "a_to_b":
            vec = self.encode_a_to_b(text, embed_fn_a)
            match, score = self.decode_at_b(vec, candidates, embed_fn_b)
        else:
            vec = self.encode_b_to_a(text, embed_fn_b)
            match, score = self.decode_at_a(vec, candidates, embed_fn_a)

        return vec, match, score

    def to_file(self, path: Union[str, Path]) -> None:
        """Export aligned pair to file.

        Creates:
        - {path}_pair.json: Metadata and rotation matrices
        - {path}_key_a.json + .npz: Key A
        - {path}_key_b.json + .npz: Key B

        Args:
            path: Base path (without extension)
        """
        path = Path(path)

        # Save individual keys
        self.key_a.to_file(str(path) + "_key_a")
        self.key_b.to_file(str(path) + "_key_b")

        # Save pair metadata and rotation matrices
        meta = {
            "spectrum_correlation": self.spectrum_correlation,
            "procrustes_residual": self.procrustes_residual,
            "k": self.k,
            "version": "1.0.0"
        }
        with open(str(path) + "_pair.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        np.savez_compressed(
            str(path) + "_pair.npz",
            R_a_to_b=self.R_a_to_b,
            R_b_to_a=self.R_b_to_a
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'AlignedKeyPair':
        """Load aligned pair from file.

        Args:
            path: Base path (without extension)

        Returns:
            Loaded AlignedKeyPair
        """
        path = Path(path)

        # Load keys
        key_a = AlignmentKey.from_file(str(path) + "_key_a")
        key_b = AlignmentKey.from_file(str(path) + "_key_b")

        # Load pair metadata
        with open(str(path) + "_pair.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)

        arrays = np.load(str(path) + "_pair.npz")

        return cls(
            key_a=key_a,
            key_b=key_b,
            R_a_to_b=arrays["R_a_to_b"],
            R_b_to_a=arrays["R_b_to_a"],
            spectrum_correlation=meta["spectrum_correlation"],
            procrustes_residual=meta["procrustes_residual"],
            k=meta["k"]
        )
