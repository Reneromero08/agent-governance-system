"""Eigen-Spectrum Alignment Protocol Library.

Cross-model semantic alignment via eigenvalue spectrum invariance.

Modules:
    mds: Classical Multidimensional Scaling
    procrustes: Procrustes alignment and out-of-sample projection
    protocol: Protocol message types and operations

Example:
    >>> from eigen_alignment.lib import mds, procrustes, protocol
    >>>
    >>> # Compute MDS coordinates
    >>> D2 = mds.squared_distance_matrix(embeddings)
    >>> X, eigenvalues, eigenvectors = mds.classical_mds(D2)
    >>>
    >>> # Compute spectrum signature
    >>> sig = protocol.spectrum_signature(eigenvalues, k=8)
    >>>
    >>> # Align two models
    >>> R, residual = procrustes.procrustes_align(X_source, X_target)
"""

from . import mds
from . import procrustes
from . import protocol

__version__ = "1.0.0"
__all__ = ["mds", "procrustes", "protocol"]
