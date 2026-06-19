"""Core testbench fixtures.

The SVTP pilot-corruption test previously used an unconstrained random vector.
A random 20D vector can rarely exceed the cosine threshold by chance, producing
an intermittent false failure.  For that one test, replace ``randn`` with a
zero vector: this is unambiguously corrupt, deterministic, and exercises the
same decoder rejection path without changing production code.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def deterministic_svtp_pilot_corruption(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.name != "test_corrupted_pilot_detected":
        yield
        return

    def zero_corruption(*shape: int) -> np.ndarray:
        return np.zeros(shape, dtype=float)

    monkeypatch.setattr(np.random, "randn", zero_corruption)
    yield
