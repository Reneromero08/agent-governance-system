"""
Q18 Tier 3: Cellular Scale Tests

Tests R = E/sigma formula at single-cell resolution using synthetic scRNA-seq-like data.
"""

from .test_tier3_cellular import (
    run_all_tier3_tests,
    test_perturbation_prediction,
    test_critical_transition,
    test_8e_conservation,
    test_edge_cases,
    SyntheticCellularDataGenerator,
    compute_cellular_R,
    compute_cellular_R_advanced,
)

__all__ = [
    'run_all_tier3_tests',
    'test_perturbation_prediction',
    'test_critical_transition',
    'test_8e_conservation',
    'test_edge_cases',
    'SyntheticCellularDataGenerator',
    'compute_cellular_R',
    'compute_cellular_R_advanced',
]
