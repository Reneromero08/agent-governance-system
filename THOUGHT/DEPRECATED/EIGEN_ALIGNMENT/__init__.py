"""Eigen-Spectrum Alignment Protocol.

Cross-model semantic alignment via eigenvalue spectrum invariance.

The eigenvalue spectrum of an anchor word distance matrix is invariant
across embedding models (r = 0.99+), enabling alignment without neural
network training.

Modules:
    lib: Core library (mds, procrustes, protocol)
    cli: Command-line interface
    benchmarks: Benchmark harness

Example:
    >>> from eigen_alignment.lib import mds, procrustes, protocol
    >>>
    >>> # Compute MDS and alignment
    >>> D2 = mds.squared_distance_matrix(embeddings)
    >>> X, eigenvalues, eigenvectors = mds.classical_mds(D2)
    >>> R, residual = procrustes.procrustes_align(X_source, X_target)
"""

__version__ = "1.0.0"
__author__ = "AGS Team"
