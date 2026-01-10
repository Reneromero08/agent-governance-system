"""Model Adapters for Eigen-Spectrum Alignment.

Provides a pluggable adapter interface for different embedding models.

Available adapters:
    - SentenceTransformersAdapter: For sentence-transformers models
"""

from .sentence_transformers import SentenceTransformersAdapter

__all__ = ["SentenceTransformersAdapter"]
