from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter, FirewallViolation


class AtomicGuardedWrites:
    """
    Atomic write operations using GuardedWriter for write firewall enforcement.

    This replaces unguarded _atomic_write_* functions with firewall-enforced versions.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.writer = GuardedWriter(
            project_root=self.project_root,
            tmp_roots=[
                "LAW/CONTRACTS/_runs/_tmp",
                "CAPABILITY/PRIMITIVES/_scratch",
                "NAVIGATION/CORTEX/_generated/_tmp",
            ],
            durable_roots=[
                "LAW/CONTRACTS/_runs",
                "NAVIGATION/CORTEX/_generated",
            ],
            exclusions=[
                "LAW/CANON",
                "AGENTS.md",
                "BUILD",
                ".git",
            ],
        )

    def atomic_write_canonical_json(self, path: Path, obj: Any) -> None:
        """
        Atomically write canonical JSON with firewall enforcement.

        Args:
            path: Path to write (relative or absolute)
            obj: Python object to serialize as canonical JSON

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(self.project_root)

        self.writer.write_tmp(path, canonical_json_bytes(obj))

    def atomic_write_bytes(self, path: Path, data: bytes) -> None:
        """
        Atomically write bytes with firewall enforcement.

        Args:
            path: Path to write (relative or absolute)
            data: Bytes to write

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(self.project_root)

        self.writer.write_tmp(path, data)

    def mkdir_tmp(self, path: Path, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory in tmp domain with firewall enforcement.

        Args:
            path: Directory path (relative or absolute)
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates firewall policy
        """
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(self.project_root)

        self.writer.mkdir_tmp(path, parents=parents, exist_ok=exist_ok)

    def open_commit_gate(self) -> None:
        """Open commit gate to allow durable writes."""
        self.writer.open_commit_gate()

    def write_durable_canonical_json(self, path: Path, obj: Any) -> None:
        """
        Write durable canonical JSON (requires commit gate to be open).

        Args:
            path: Path to write (relative or absolute)
            obj: Python object to serialize as canonical JSON

        Raises:
            FirewallViolation: If write violates firewall policy or commit gate is closed
        """
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(self.project_root)

        self.writer.write_durable(path, canonical_json_bytes(obj))

    def mkdir_durable(self, path: Path, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory in durable domain (requires commit gate to be open).

        Args:
            path: Directory path (relative or absolute)
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists
        """
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(self.project_root)

        self.writer.mkdir_durable(path, parents=parents, exist_ok=exist_ok)

