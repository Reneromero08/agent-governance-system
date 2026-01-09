#!/usr/bin/env python3
"""
Guarded Writer - Minimal integration example for write firewall.

This demonstrates how to integrate the write firewall into an existing tool.
All writes go through the firewall to enforce catalytic domain separation.

Usage:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter

    writer = GuardedWriter(project_root=Path.cwd())
    writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/output.json", data)
    writer.open_commit_gate()
    writer.write_durable("LAW/CONTRACTS/_runs/durable/result.json", data)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall, FirewallViolation


class GuardedWriter:
    """Writer utility with integrated write firewall enforcement."""

    def __init__(
        self,
        project_root: Path,
        tmp_roots: list[str] | None = None,
        durable_roots: list[str] | None = None,
        exclusions: list[str] | None = None,
    ):
        """
        Initialize guarded writer with firewall.

        Args:
            project_root: Project root directory
            tmp_roots: Temporary write roots (default: standard catalytic tmp roots)
            durable_roots: Durable write roots (default: standard catalytic durable roots)
            exclusions: Paths to exclude from all writes
        """
        self.project_root = Path(project_root).resolve()

        # Default catalytic domains
        if tmp_roots is None:
            tmp_roots = [
                "LAW/CONTRACTS/_runs/_tmp",
                "CAPABILITY/PRIMITIVES/_scratch",
                "NAVIGATION/CORTEX/_generated/_tmp",
            ]

        if durable_roots is None:
            durable_roots = [
                "LAW/CONTRACTS/_runs",
                "NAVIGATION/CORTEX/_generated",
            ]

        if exclusions is None:
            exclusions = [
                "LAW/CANON",
                "AGENTS.md",
                "BUILD",
                ".git",
            ]

        self.firewall = WriteFirewall(
            tmp_roots=tmp_roots,
            durable_roots=durable_roots,
            project_root=self.project_root,
            exclusions=exclusions,
        )

    def write_tmp(self, path: str | Path, data: str | bytes) -> None:
        """
        Write to temporary domain (allowed during execution).

        Args:
            path: Path to write (relative to project_root)
            data: Data to write (str or bytes)

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        self.firewall.safe_write(path, data, kind="tmp")

    def write_durable(self, path: str | Path, data: str | bytes) -> None:
        """
        Write to durable domain (requires commit gate to be open).

        Args:
            path: Path to write (relative to project_root)
            data: Data to write (str or bytes)

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        self.firewall.safe_write(path, data, kind="durable")

    def mkdir_tmp(self, path: str | Path, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory in temporary domain.

        Args:
            path: Directory path (relative to project_root)
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates firewall policy
        """
        self.firewall.safe_mkdir(path, kind="tmp", parents=parents, exist_ok=exist_ok)

    def mkdir_durable(self, path: str | Path, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory in durable domain (requires commit gate to be open).

        Args:
            path: Directory path (relative to project_root)
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates firewall policy
        """
        self.firewall.safe_mkdir(path, kind="durable", parents=parents, exist_ok=exist_ok)

    def _is_tmp_path(self, path: str | Path) -> bool:
        """Check if path is in tmp domain based on configured tmp_roots."""
        norm_path = str(path).replace("\\", "/")
        for tmp_root in self.firewall.tmp_roots:
            if norm_path.startswith(tmp_root + "/") or norm_path == tmp_root:
                return True
        return False

    def write_auto(self, path: str | Path, data: str | bytes) -> None:
        """
        Write to automatic domain (tmp or durable based on path).

        Args:
            path: Path to write (relative to project_root)
            data: Data to write (str or bytes)

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        if self._is_tmp_path(path):
            self.write_tmp(path, data)
        else:
            self.write_durable(path, data)

    def mkdir_auto(self, path: str | Path, parents: bool = True, exist_ok: bool = True) -> None:
        """
        Create directory in automatic domain (tmp or durable based on path).

        Args:
            path: Directory path (relative to project_root)
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates firewall policy
        """
        if self._is_tmp_path(path):
            self.mkdir_tmp(path, parents=parents, exist_ok=exist_ok)
        else:
            self.mkdir_durable(path, parents=parents, exist_ok=exist_ok)

    def open_commit_gate(self) -> None:
        """
        Open the commit gate to allow durable writes.

        Call this after all execution is complete and you're ready to commit results.
        """
        self.firewall.open_commit_gate()

    def unlink(self, path: str | Path) -> None:
        """
        Delete a file with firewall enforcement.

        Args:
            path: Path to delete

        Raises:
            FirewallViolation: If unlink violates firewall policy
        """
        self.firewall.safe_unlink(path)

    def safe_rename(self, src: str | Path, dst: str | Path) -> None:
        """
        Rename/move a file or directory with firewall enforcement.

        Args:
            src: Source path
            dst: Destination path

        Raises:
            FirewallViolation: If rename violates firewall policy
        """
        self.firewall.safe_rename(src, dst)

    def copy(self, src: str | Path, dst: str | Path) -> None:
        """
        Copy a file with firewall enforcement.

        Args:
            src: Source path (must allow read)
            dst: Destination path (must allow write)

        Raises:
            FirewallViolation: If copy violates firewall policy
        """
        # Determine strict enforcement by checking write validity primarily
        # Read from valid source
        src_path = Path(src).resolve()
        dst_path = Path(dst)
        
        # Read content (not strictly enforced by firewall read policy, but simple read is safe)
        if not src_path.exists():
             raise FileNotFoundError(f"Source file not found: {src}")
        
        content = src_path.read_bytes()
        
        # Determine kind based on dest location 
        # (This is tricky, we delegate to safe_write logic, or simpler: try tmp then durable?)
        # But wait, firewall requires us to KNOW kind (tmp vs durable).
        # We can detect it from firewall logic, or just assume durable if calling copy?
        # Actually safe_write allows us to specify kind.
        # Let's inspect destination to decide kind.
        
        # We can query firewall private helper _check_path_domain or just try both.
        # But accessing private method is bad.
        # Let's add public `get_domain(path)` to Firewall later or for now try-except.
        # Or expose it.
        # For now, let's look at roots.
        
        rel_dst = str(dst_path.resolve().relative_to(self.project_root)).replace("\\", "/")
        
        # Check if it looks like tmp
        is_tmp = False
        for r in self.firewall.tmp_roots:
             if rel_dst.startswith(r + "/") or rel_dst == r:
                 is_tmp = True
                 break
                 
        kind = "tmp" if is_tmp else "durable"
        
        # If durable, gate must be open (safe_write checks this)
        self.firewall.safe_write(dst, content, kind=kind)

    def handle_violation(self, violation: FirewallViolation, receipt_dir: Path | None = None) -> Dict[str, Any]:
        """
        Handle a firewall violation by writing receipt and returning error info.

        Args:
            violation: FirewallViolation exception
            receipt_dir: Optional directory to write violation receipt

        Returns:
            Dict with error information
        """
        if receipt_dir:
            receipt_path = receipt_dir / f"firewall_violation_{violation.error_code}.json"
            violation.write_receipt(receipt_path)

        return {
            "ok": False,
            "code": violation.error_code,
            "message": violation.message,
            "violation_receipt": violation.violation_receipt,
        }


# Example usage
if __name__ == "__main__":
    # Example: write some tmp files, then commit durable results
    writer = GuardedWriter(project_root=REPO_ROOT)

    try:
        # Stage 1: Execution - tmp writes allowed
        writer.mkdir_tmp("LAW/CONTRACTS/_runs/_tmp/example")
        writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/example/progress.json", '{"status": "running"}')

        # Stage 2: Commit - open gate and write durable results
        writer.open_commit_gate()
        writer.write_durable("LAW/CONTRACTS/_runs/example_result.json", '{"status": "complete"}')

        print("✓ All writes succeeded")

    except FirewallViolation as e:
        error_info = writer.handle_violation(e)
        print(f"✗ Firewall violation: {error_info['code']}")
        print(f"  Message: {error_info['message']}")
        sys.exit(1)
