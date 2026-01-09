"""
Pytest configuration for TESTBENCH.

Registers custom markers and provides shared fixtures.
"""
import pytest


def pytest_addoption(parser):
    """Add --run-slow option to include slow stress tests."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow stress tests (default: skip)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped unless --run-slow is passed)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        # --run-slow given: don't skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="Slow test - use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
