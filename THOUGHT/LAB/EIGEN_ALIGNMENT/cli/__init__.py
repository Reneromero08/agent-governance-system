"""Eigen-Spectrum Alignment CLI.

Command-line interface for the ESAP protocol.

Commands:
    anchors build   - Build anchor set from word list
    signature compute - Compute spectrum signature for a model
    signature compare - Compare two spectrum signatures
    map fit         - Fit alignment map between models
    map apply       - Apply alignment to new points
"""

from .main import main

__all__ = ["main"]
