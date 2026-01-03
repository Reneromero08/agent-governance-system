"""
Z.2.5 – Garbage Collection for CAS

Provides policy-driven, conservative, auditable, and fail-closed GC for CAS storage.

Public API:
- gc_collect(*, dry_run: bool = True, allow_empty_roots: bool = False) -> dict
"""

from CAPABILITY.GC.gc import gc_collect

__all__ = ['gc_collect']
