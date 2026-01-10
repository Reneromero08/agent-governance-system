"""Sentence Transformers Adapter.

Adapter for HuggingFace sentence-transformers models.

Supported models:
    - all-MiniLM-L6-v2 (384d, reference model)
    - all-mpnet-base-v2 (768d)
    - e5-large-v2 (1024d, requires "query: " prefix)
    - bge-large-en-v1.5 (1024d)
    - gte-large (1024d)
"""

from typing import Protocol, runtime_checkable
import hashlib
import numpy as np


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Protocol for embedding adapters."""

    @property
    def embedder_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def weights_hash(self) -> str:
        """Return hash of model weights (for drift detection)."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings

        Returns:
            (n, d) L2-normalized embedding matrix
        """
        ...


# Known models with metadata
KNOWN_MODELS = {
    'all-MiniLM-L6-v2': {
        'id': 'sentence-transformers/all-MiniLM-L6-v2',
        'dim': 384,
        'prefix': None,
    },
    'all-mpnet-base-v2': {
        'id': 'sentence-transformers/all-mpnet-base-v2',
        'dim': 768,
        'prefix': None,
    },
    'e5-large-v2': {
        'id': 'intfloat/e5-large-v2',
        'dim': 1024,
        'prefix': 'query: ',
    },
    'bge-large-en-v1.5': {
        'id': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'prefix': None,
    },
    'gte-large': {
        'id': 'thenlper/gte-large',
        'dim': 1024,
        'prefix': None,
    },
}


class SentenceTransformersAdapter:
    """Adapter for sentence-transformers models."""

    def __init__(self, model_name: str):
        """Initialize adapter for a model.

        Args:
            model_name: Short name (e.g., "all-MiniLM-L6-v2") or
                        full path (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        """
        self._model_name = model_name
        self._model = None
        self._weights_hash = None

        # Get model info
        if model_name in KNOWN_MODELS:
            self._info = KNOWN_MODELS[model_name]
            self._model_id = self._info['id']
        else:
            # Assume it's a full path
            self._model_id = model_name
            self._info = {
                'id': model_name,
                'dim': None,  # Will be determined after loading
                'prefix': None,
            }

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(self._model_id)

        # Update dimension if not known
        if self._info['dim'] is None:
            self._info['dim'] = self._model.get_sentence_embedding_dimension()

        # Compute weights hash (simplified - hash of model path)
        # In production, would hash actual weights
        self._weights_hash = f"sha256:{hashlib.sha256(self._model_id.encode()).hexdigest()[:16]}"

    @property
    def embedder_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._info['dim'] is None:
            self._load_model()
        return self._info['dim']

    @property
    def weights_hash(self) -> str:
        """Return hash of model weights."""
        if self._weights_hash is None:
            self._load_model()
        return self._weights_hash

    @property
    def prefix(self) -> str | None:
        """Return query prefix if required."""
        return self._info.get('prefix')

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings

        Returns:
            (n, d) L2-normalized embedding matrix
        """
        self._load_model()

        # Apply prefix if required
        prefix = self.prefix
        if prefix:
            texts = [prefix + t for t in texts]

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    def to_descriptor(self):
        """Create EmbedderDescriptor for this adapter."""
        from ..protocol import EmbedderDescriptor

        return EmbedderDescriptor(
            embedder_id=self.embedder_id,
            dimension=self.dimension,
            weights_hash=self.weights_hash,
            normalize=True,
            prefix=self.prefix
        )


def get_adapter(model_name: str) -> SentenceTransformersAdapter:
    """Get an adapter for the given model.

    Args:
        model_name: Model name or path

    Returns:
        Configured adapter
    """
    return SentenceTransformersAdapter(model_name)


def list_known_models() -> list[str]:
    """List known model names."""
    return list(KNOWN_MODELS.keys())
