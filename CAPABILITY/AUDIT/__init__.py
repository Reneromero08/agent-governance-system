"""
CAPABILITY/AUDIT - Root audit and GC safety verification.

Public API:
    root_audit() - Deterministic, fail-closed audit of root completeness and GC safety.
"""

from .root_audit import root_audit

__all__ = ['root_audit']
