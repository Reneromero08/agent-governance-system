"""
Quantum Geometric Tensor Python Bindings

See qgt.py for implementation.
"""

from .qgt import (
    # Core functions
    normalize_embeddings,
    fubini_study_metric,
    participation_ratio,
    metric_eigenspectrum,

    # Berry phase
    berry_connection,
    berry_phase,
    holonomy,
    holonomy_angle,

    # Natural gradient
    natural_gradient,
    principal_directions,

    # Word analogy
    create_analogy_loop,
    analogy_berry_phase,

    # Chern number
    chern_number_estimate,

    # Analysis
    analyze_qgt_structure,
    compare_compass_to_qgt,
    load_and_analyze,
)

__version__ = "0.1.0"
__all__ = [
    'normalize_embeddings',
    'fubini_study_metric',
    'participation_ratio',
    'metric_eigenspectrum',
    'berry_connection',
    'berry_phase',
    'holonomy',
    'holonomy_angle',
    'natural_gradient',
    'principal_directions',
    'create_analogy_loop',
    'analogy_berry_phase',
    'chern_number_estimate',
    'analyze_qgt_structure',
    'compare_compass_to_qgt',
    'load_and_analyze',
]
