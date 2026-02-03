"""
Quantum Geometric Tensor Python Bindings

See qgt.py for core implementation.
See qgt_phase.py for Q51 phase recovery tools.
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

# Q51 Phase Recovery (optional import - may not have all dependencies)
try:
    from .qgt_phase import (
        # Phase recovery tools
        hilbert_phase_recovery,
        bispectrum_phase_estimate,
        phase_from_covariance,
        unwrap_phases,
        octant_phase_mapping,

        # Zero signature test
        test_zero_signature,

        # Circular statistics
        circular_mean,
        circular_variance,
        circular_correlation,

        # Comprehensive analysis
        analyze_phase_structure,
        validate_all,

        # Result classes
        PhaseRecoveryResult,
        BispectrumPhaseResult,
        CovariancePhaseResult,
        OctantPhaseResult,
        ZeroSignatureResult,

        # Constants
        SEMIOTIC_CONSTANT,
        CRITICAL_ALPHA,
        OCTANT_COUNT,
        SECTOR_WIDTH,
    )
    HAS_PHASE_RECOVERY = True
except ImportError:
    HAS_PHASE_RECOVERY = False

__version__ = "0.2.0"  # Updated for Q51 phase recovery
__all__ = [
    # Core QGT
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
    # Q51 Phase Recovery (if available)
    'hilbert_phase_recovery',
    'bispectrum_phase_estimate',
    'phase_from_covariance',
    'unwrap_phases',
    'octant_phase_mapping',
    'test_zero_signature',
    'circular_mean',
    'circular_variance',
    'circular_correlation',
    'analyze_phase_structure',
    'SEMIOTIC_CONSTANT',
    'CRITICAL_ALPHA',
    'OCTANT_COUNT',
    'SECTOR_WIDTH',
]
