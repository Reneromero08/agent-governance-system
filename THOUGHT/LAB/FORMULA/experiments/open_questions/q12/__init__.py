"""
Q12: Phase Transitions in Semantic Systems

HARDCORE validation suite testing whether semantic systems exhibit true
phase transitions where "truth crystallizes suddenly rather than gradually."

This package contains 12 tests derived from statistical physics:

Phase 1 - Core Evidence:
    test_q12_01_finite_size_scaling  - Data collapse across system sizes
    test_q12_06_order_parameter      - Jump behavior at transition
    test_q12_07_percolation          - Giant component emergence
    test_q12_09_binder_cumulant      - Precision critical point ID

Phase 2 - Mechanism:
    test_q12_02_universality_class   - Critical exponent matching
    test_q12_03_susceptibility       - Response divergence
    test_q12_11_symmetry_breaking    - Isotropy collapse

Phase 3 - Dynamics:
    test_q12_04_critical_slowing     - Relaxation time divergence
    test_q12_05_hysteresis           - First vs second order
    test_q12_08_scale_invariance     - Power-law correlations

Phase 4 - Universality:
    test_q12_10_fisher_information   - Information-theoretic signature
    test_q12_12_cross_architecture   - BERT/GloVe/Word2Vec agreement

Usage:
    python run_q12_all.py

Success Criteria:
    10+/12 PASS: ANSWERED - Phase transition CONFIRMED
    7-9/12 PASS: PARTIAL - Strong evidence
    <7/12 PASS: FALSIFIED - Not a true phase transition

Author: AGS Research
Date: 2026-01-19
"""

__version__ = "1.0.0"
__author__ = "AGS Research"
