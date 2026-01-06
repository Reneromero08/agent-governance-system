#!/usr/bin/env python3
"""
Packer Write Firewall Integration (Phase 2.4.1B)

Provides firewall-integrated write operations for LLM Packer components.
All packer writes must route through this module to enforce catalytic domain isolation.

Usage:
    from MEMORY.LLM_PACKER.Engine.packer.firewall_writer import PackerWriter

    writer = PackerWriter(project_root=PROJECT_ROOT)
    writer.write_manifest(pack_dir / "PACK_MANIFEST.json", manifest_data)
    writer.commit()  # Opens commit gate for durable writes
    writer.write_receipt(runs_dir / "RECEIPT.json", receipt_data)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import write firewall
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall, FirewallViolation


class PackerWriter:
    """
    Write firewall integration for LLM Packer operations.

    Enforces catalytic domain separation:
    - Tmp writes: MEMORY/LLM_PACKER/_packs/_tmp/
    - Durable writes: MEMORY/LLM_PACKER/_packs/ (non-_tmp)
    - Commit gate required for durable writes
    """

    def __init__(
        self,
        project_root: Path,
        tmp_roots: Optional[list[str]] = None,
        durable_roots: Optional[list[str]] = None,
        exclusions: Optional[list[str]] = None,
    ):
        """
        Initialize packer writer with firewall.

        Args:
            project_root: Project root directory
            tmp_roots: Optional tmp roots (defaults to packer-specific tmp)
            durable_roots: Optional durable roots (defaults to packer-specific durable)
            exclusions: Optional exclusion paths
        """
        self.project_root = Path(project_root).resolve()

        # Default packer-specific catalytic domains
        if tmp_roots is None:
            tmp_roots = [
                "MEMORY/LLM_PACKER/_packs/_tmp",
                "LAW/CONTRACTS/_runs/_tmp",
            ]

        if durable_roots is None:
            durable_roots = [
                "MEMORY/LLM_PACKER/_packs",
                "LAW/CONTRACTS/_runs",
            ]

        if exclusions is None:
            exclusions = [
                "LAW/CANON",
                "AGENTS.md",
                ".git",
            ]

        self.firewall = WriteFirewall(
            tmp_roots=tmp_roots,
            durable_roots=durable_roots,
            project_root=self.project_root,
            exclusions=exclusions,
        )

    def write_json(
        self,
        path: str | Path,
        payload: Any,
        *,
        kind: str = "durable",
        indent: Optional[int] = 2,
        canonical: bool = False,
    ) -> None:
        """
        Write JSON file with firewall enforcement.

        Args:
            path: Output path (relative or absolute)
            payload: JSON-serializable data
            kind: "tmp" or "durable"
            indent: JSON indent (None for compact, 2 for pretty)
            canonical: If True, use canonical JSON (sort_keys, no whitespace)

        Raises:
            FirewallViolation: If write violates policy
        """
        if canonical:
            # Canonical JSON: sorted keys, compact
            json_bytes = (
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode("utf-8")
        else:
            # Pretty JSON: indented
            json_str = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"
            json_bytes = json_str.encode("utf-8")

        self.firewall.safe_write(path, json_bytes, kind=kind)

    def write_text(
        self,
        path: str | Path,
        content: str,
        *,
        kind: str = "durable",
        encoding: str = "utf-8",
    ) -> None:
        """
        Write text file with firewall enforcement.

        Args:
            path: Output path (relative or absolute)
            content: Text content
            kind: "tmp" or "durable"
            encoding: Text encoding

        Raises:
            FirewallViolation: If write violates policy
        """
        data = content.encode(encoding)
        self.firewall.safe_write(path, data, kind=kind)

    def write_bytes(
        self,
        path: str | Path,
        data: bytes,
        *,
        kind: str = "durable",
    ) -> None:
        """
        Write binary file with firewall enforcement.

        Args:
            path: Output path (relative or absolute)
            data: Binary data
            kind: "tmp" or "durable"

        Raises:
            FirewallViolation: If write violates policy
        """
        self.firewall.safe_write(path, data, kind=kind)

    def mkdir(
        self,
        path: str | Path,
        *,
        kind: str = "durable",
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """
        Create directory with firewall enforcement.

        Args:
            path: Directory path (relative or absolute)
            kind: "tmp" or "durable"
            parents: Create parent directories
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates policy
        """
        self.firewall.safe_mkdir(path, kind=kind, parents=parents, exist_ok=exist_ok)

    def rename(self, src: str | Path, dst: str | Path) -> None:
        """
        Rename file/directory with firewall enforcement.

        Args:
            src: Source path
            dst: Destination path

        Raises:
            FirewallViolation: If rename violates policy
        """
        self.firewall.safe_rename(src, dst)

    def unlink(self, path: str | Path) -> None:
        """
        Delete file with firewall enforcement.

        Args:
            path: File path to delete

        Raises:
            FirewallViolation: If unlink violates policy
        """
        self.firewall.safe_unlink(path)

    def commit(self) -> None:
        """
        Open commit gate to allow durable writes.

        Call this after execution phase completes and before committing durable results.
        """
        self.firewall.open_commit_gate()

    def get_violation_receipt(self, violation: FirewallViolation) -> Dict[str, Any]:
        """
        Extract violation receipt from FirewallViolation exception.

        Args:
            violation: FirewallViolation exception

        Returns:
            Violation receipt dict
        """
        return violation.violation_receipt


# Convenience functions for drop-in replacement

def write_json_guarded(
    path: Path,
    payload: Any,
    *,
    writer: Optional[PackerWriter] = None,
    kind: str = "durable",
) -> None:
    """
    Drop-in replacement for Path.write_text(json.dumps(...)) with firewall.

    Args:
        path: Output path
        payload: JSON-serializable data
        writer: Optional PackerWriter (if None, uses direct write for backwards compat)
        kind: "tmp" or "durable"
    """
    if writer is None:
        # Legacy behavior (no firewall)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        writer.write_json(path, payload, kind=kind)


def write_text_guarded(
    path: Path,
    content: str,
    *,
    writer: Optional[PackerWriter] = None,
    kind: str = "durable",
    encoding: str = "utf-8",
) -> None:
    """
    Drop-in replacement for Path.write_text() with firewall.

    Args:
        path: Output path
        content: Text content
        writer: Optional PackerWriter (if None, uses direct write for backwards compat)
        kind: "tmp" or "durable"
        encoding: Text encoding
    """
    if writer is None:
        # Legacy behavior (no firewall)
        path.write_text(content, encoding=encoding)
    else:
        writer.write_text(path, content, kind=kind, encoding=encoding)


def write_bytes_guarded(
    path: Path,
    data: bytes,
    *,
    writer: Optional[PackerWriter] = None,
    kind: str = "durable",
) -> None:
    """
    Drop-in replacement for Path.write_bytes() with firewall.

    Args:
        path: Output path
        data: Binary data
        writer: Optional PackerWriter (if None, uses direct write for backwards compat)
        kind: "tmp" or "durable"
    """
    if writer is None:
        # Legacy behavior (no firewall)
        path.write_bytes(data)
    else:
        writer.write_bytes(path, data, kind=kind)
