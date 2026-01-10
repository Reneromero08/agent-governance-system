"""Protocol Message Types and Operations.

Defines the ESAP protocol message types:
    - ANCHOR_SET
    - EMBEDDER_DESCRIPTOR
    - DISTANCE_METRIC_DESCRIPTOR
    - SPECTRUM_SIGNATURE
    - ALIGNMENT_MAP

Provides functions for creating, validating, and hashing messages.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import numpy as np


# Protocol version
PROTOCOL_VERSION = "1.0.0"


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string.

    Ensures deterministic serialization:
    - Keys sorted alphabetically
    - No extra whitespace
    - Floats to 10 decimal places
    """
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Cannot serialize {type(o)}")

    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=default)


def compute_hash(data: str | bytes) -> str:
    """Compute SHA-256 hash of data.

    Args:
        data: String or bytes to hash

    Returns:
        Hash as "sha256:hexdigest" string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def format_float(value: float, precision: int = 10) -> str:
    """Format float to canonical string representation.

    Args:
        value: Float value
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"


def utc_now() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AnchorSet:
    """Defines the reference anchor words for alignment."""

    anchors: list[dict]  # [{"id": "a001", "text": "dog"}, ...]
    anchor_hash: str = ""
    version: str = PROTOCOL_VERSION
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self):
        if not self.anchor_hash:
            self.anchor_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of anchor list."""
        # Sort by id, join with newlines
        sorted_anchors = sorted(self.anchors, key=lambda x: x['id'])
        canonical = '\n'.join(f"{a['id']}:{a['text']}" for a in sorted_anchors)
        return compute_hash(canonical)

    def to_dict(self) -> dict:
        return {
            "type": "ANCHOR_SET",
            "version": self.version,
            "anchors": self.anchors,
            "anchor_hash": self.anchor_hash,
            "created_at": self.created_at
        }

    @classmethod
    def from_words(cls, words: list[str]) -> 'AnchorSet':
        """Create AnchorSet from list of words."""
        anchors = [{"id": f"a{i:03d}", "text": w} for i, w in enumerate(words)]
        return cls(anchors=anchors)


@dataclass
class EmbedderDescriptor:
    """Describes the embedding model used."""

    embedder_id: str
    dimension: int
    weights_hash: str = ""
    normalize: bool = True
    prefix: str | None = None
    version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return {
            "type": "EMBEDDER_DESCRIPTOR",
            "version": self.version,
            "embedder_id": self.embedder_id,
            "weights_hash": self.weights_hash,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "prefix": self.prefix
        }


@dataclass
class DistanceMetricDescriptor:
    """Specifies the distance metric used."""

    metric: str = "cosine"
    normalization: str = "l2"
    squared: bool = True
    version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return {
            "type": "DISTANCE_METRIC_DESCRIPTOR",
            "version": self.version,
            "metric": self.metric,
            "normalization": self.normalization,
            "squared": self.squared
        }


@dataclass
class SpectrumSignature:
    """The eigenvalue spectrum - the invariant across models."""

    eigenvalues: list[float]
    anchor_set_hash: str
    embedder_id: str
    k: int = 0
    effective_rank: float = 0.0
    spectrum_hash: str = ""
    version: str = PROTOCOL_VERSION
    computed_at: str = field(default_factory=utc_now)

    def __post_init__(self):
        if self.k == 0:
            self.k = len(self.eigenvalues)
        if self.effective_rank == 0.0:
            self.effective_rank = self._compute_effective_rank()
        if not self.spectrum_hash:
            self.spectrum_hash = self._compute_hash()

    def _compute_effective_rank(self) -> float:
        """Compute effective rank of spectrum."""
        ev = np.array(self.eigenvalues)
        if np.sum(ev) == 0:
            return 0.0
        return float((np.sum(ev) ** 2) / np.sum(ev ** 2))

    def _compute_hash(self) -> str:
        """Compute hash of eigenvalue spectrum."""
        # Format eigenvalues to canonical precision
        canonical = ','.join(format_float(v) for v in self.eigenvalues)
        return compute_hash(canonical)

    def to_dict(self) -> dict:
        return {
            "type": "SPECTRUM_SIGNATURE",
            "version": self.version,
            "anchor_set_hash": self.anchor_set_hash,
            "embedder_id": self.embedder_id,
            "eigenvalues": self.eigenvalues,
            "k": self.k,
            "effective_rank": self.effective_rank,
            "spectrum_hash": self.spectrum_hash,
            "computed_at": self.computed_at
        }

    def correlation(self, other: 'SpectrumSignature') -> float:
        """Compute Spearman correlation with another signature."""
        from scipy.stats import spearmanr

        k = min(len(self.eigenvalues), len(other.eigenvalues))
        a = self.eigenvalues[:k]
        b = other.eigenvalues[:k]

        corr, _ = spearmanr(a, b)
        return float(corr)


@dataclass
class AlignmentMap:
    """The rotation matrix for cross-model alignment."""

    rotation_matrix: list[list[float]]
    source_embedder: str
    target_embedder: str
    anchor_set_hash: str
    k: int = 0
    procrustes_residual: float = 0.0
    map_hash: str = ""
    version: str = PROTOCOL_VERSION
    computed_at: str = field(default_factory=utc_now)

    def __post_init__(self):
        if self.k == 0:
            self.k = len(self.rotation_matrix)
        if not self.map_hash:
            self.map_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of rotation matrix."""
        # Flatten and format to canonical precision
        flat = []
        for row in self.rotation_matrix:
            flat.extend(format_float(v) for v in row)
        canonical = ','.join(flat)
        return compute_hash(canonical)

    def to_dict(self) -> dict:
        return {
            "type": "ALIGNMENT_MAP",
            "version": self.version,
            "source_embedder": self.source_embedder,
            "target_embedder": self.target_embedder,
            "anchor_set_hash": self.anchor_set_hash,
            "rotation_matrix": self.rotation_matrix,
            "k": self.k,
            "procrustes_residual": self.procrustes_residual,
            "map_hash": self.map_hash,
            "computed_at": self.computed_at
        }

    def as_numpy(self) -> np.ndarray:
        """Get rotation matrix as numpy array."""
        return np.array(self.rotation_matrix)


# Error codes
class ESAPError(Exception):
    """Base exception for ESAP protocol errors."""
    code: str = "E000"
    message: str = "Unknown error"

    def __init__(self, message: str | None = None):
        self.message = message or self.__class__.message
        super().__init__(f"[{self.code}] {self.message}")


class AnchorMismatchError(ESAPError):
    code = "E001"
    message = "anchor_hash does not match"


class EmbedderMismatchError(ESAPError):
    code = "E002"
    message = "weights_hash does not match"


class MetricMismatchError(ESAPError):
    code = "E003"
    message = "Distance metric descriptor mismatch"


class SpectrumMismatchError(ESAPError):
    code = "E004"
    message = "Spectrum correlation below threshold"


class InsufficientRankError(ESAPError):
    code = "E005"
    message = "Not enough positive eigenvalues"


class AlignmentFailedError(ESAPError):
    code = "E006"
    message = "Procrustes residual too high"


class SchemaInvalidError(ESAPError):
    code = "E007"
    message = "Message does not match schema"


class VersionUnsupportedError(ESAPError):
    code = "E008"
    message = "Protocol version not supported"


def spectrum_signature(
    eigenvalues: np.ndarray,
    anchor_set_hash: str,
    embedder_id: str,
    k: int | None = None
) -> SpectrumSignature:
    """Create a SpectrumSignature from eigenvalues.

    Args:
        eigenvalues: Eigenvalues from MDS (sorted descending)
        anchor_set_hash: Hash of the anchor set used
        embedder_id: ID of the embedding model
        k: Number of eigenvalues to retain (default: all)

    Returns:
        SpectrumSignature object
    """
    if k is None:
        k = len(eigenvalues)
    ev = eigenvalues[:k].tolist()

    return SpectrumSignature(
        eigenvalues=ev,
        anchor_set_hash=anchor_set_hash,
        embedder_id=embedder_id,
        k=k
    )


def alignment_map(
    rotation: np.ndarray,
    source_embedder: str,
    target_embedder: str,
    anchor_set_hash: str,
    residual: float
) -> AlignmentMap:
    """Create an AlignmentMap from rotation matrix.

    Args:
        rotation: Orthogonal rotation matrix from Procrustes
        source_embedder: ID of source model
        target_embedder: ID of target (reference) model
        anchor_set_hash: Hash of the anchor set used
        residual: Procrustes alignment residual

    Returns:
        AlignmentMap object
    """
    return AlignmentMap(
        rotation_matrix=rotation.tolist(),
        source_embedder=source_embedder,
        target_embedder=target_embedder,
        anchor_set_hash=anchor_set_hash,
        procrustes_residual=residual
    )
