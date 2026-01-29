#!/usr/bin/env python3
"""
Q11: Valley Blindness - Information Horizon Tests

This module contains 12 tests designed to answer:
"Can we extend the information horizon without changing epistemology?
Or is 'can't know from here' an irreducible limit?"

Tests are organized in three phases:
- Phase 1: Structural Horizons (Tests 1-4)
- Phase 2: Detection & Extension (Tests 5-8)
- Phase 3: Transcendence (Tests 9-12)

Q11 is ANSWERED if 8+ tests pass with consistent pattern.
"""

from .q11_utils import (
    RANDOM_SEED,
    SEMANTIC_MODEL,
    EPS,
    HorizonTestResult,
    TestConfig,
    HorizonType,
    compute_R,
    get_embeddings,
    compute_cosine_similarity,
    compute_fidelity,
    to_builtin,
)

__all__ = [
    'RANDOM_SEED',
    'SEMANTIC_MODEL',
    'EPS',
    'HorizonTestResult',
    'TestConfig',
    'HorizonType',
    'compute_R',
    'get_embeddings',
    'compute_cosine_similarity',
    'compute_fidelity',
    'to_builtin',
]
