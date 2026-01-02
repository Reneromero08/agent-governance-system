"""
CAPABILITY/ARTIFACTS - CAS-backed artifact store (Z.2.2)

Provides content-addressed artifact storage with backward compatibility for file paths.
"""

from .store import (
    store_bytes,
    load_bytes,
    store_file,
    materialize,
    ArtifactException,
    InvalidReferenceException,
    ObjectNotFoundException,
)

__all__ = [
    "store_bytes",
    "load_bytes",
    "store_file",
    "materialize",
    "ArtifactException",
    "InvalidReferenceException",
    "ObjectNotFoundException",
]
