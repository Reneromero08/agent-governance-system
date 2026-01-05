"""
Runtime Write Firewall (Phase 1.5A — Catalytic Domains)

Mechanical, fail-closed IO policy layer enforcing catalytic domain isolation:
- tmp writes only under declared tmp roots during execution
- durable writes only under declared durable roots and only after commit gate opens
- block rename/unlink/mkdir outside allowed roots
- deterministic errors + receipts on violation

This is the single source of truth (SSOT) for runtime write policy enforcement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class WriteFirewall:
    """Runtime write firewall enforcing catalytic domain separation."""

    VERSION = "1.0.0"

    def __init__(
        self,
        tmp_roots: List[str],
        durable_roots: List[str],
        project_root: Path,
        exclusions: Optional[List[str]] = None,
    ):
        """
        Initialize the write firewall.

        Args:
            tmp_roots: List of temporary write root paths (relative to project_root)
            durable_roots: List of durable write root paths (relative to project_root)
            project_root: Project root directory (absolute path)
            exclusions: Optional list of paths to exclude from all checks
        """
        self.tmp_roots = [self._normalize_path(r) for r in tmp_roots]
        self.durable_roots = [self._normalize_path(r) for r in durable_roots]
        self.project_root = Path(project_root).resolve()
        self.exclusions = [self._normalize_path(e) for e in (exclusions or [])]

        # Commit gate starts closed
        self._commit_gate_open = False

        # Compute tool version hash
        self._tool_version_hash = self._compute_tool_hash()

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to forward slashes, no trailing slash.

        Args:
            path: Path string to normalize

        Returns:
            Normalized path string
        """
        return str(Path(path)).replace("\\", "/").rstrip("/")

    def _compute_tool_hash(self) -> str:
        """
        Compute SHA256 hash of this module file for receipts.

        Returns:
            SHA256 hex digest of module file bytes
        """
        try:
            module_path = Path(__file__).resolve()
            module_bytes = module_path.read_bytes()
            return hashlib.sha256(module_bytes).hexdigest()
        except Exception:
            # Fallback if __file__ not available or read fails
            return hashlib.sha256(f"write_firewall:{self.VERSION}".encode()).hexdigest()

    def configure_policy(
        self,
        tmp_roots: List[str],
        durable_roots: List[str],
        exclusions: Optional[List[str]] = None
    ) -> None:
        """
        Reconfigure the firewall policy.

        Args:
            tmp_roots: List of temporary write root paths
            durable_roots: List of durable write root paths
            exclusions: Optional list of paths to exclude from all checks
        """
        self.tmp_roots = [self._normalize_path(r) for r in tmp_roots]
        self.durable_roots = [self._normalize_path(r) for r in durable_roots]
        self.exclusions = [self._normalize_path(e) for e in (exclusions or [])]
        # Reconfigure closes the commit gate
        self._commit_gate_open = False

    def open_commit_gate(self) -> None:
        """Open the commit gate to allow durable writes."""
        self._commit_gate_open = True

    def _get_policy_snapshot(self) -> Dict[str, Any]:
        """
        Get current policy configuration snapshot.

        Returns:
            Policy snapshot dict for receipts
        """
        return {
            "tmp_roots": sorted(self.tmp_roots),
            "durable_roots": sorted(self.durable_roots),
            "exclusions": sorted(self.exclusions),
            "commit_gate_open": self._commit_gate_open,
            "tool_version": self.VERSION,
            "tool_version_hash": self._tool_version_hash,
        }

    def _resolve_and_validate_path(self, path: str | Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Resolve path and perform basic validation.

        Args:
            path: Path to validate (relative or absolute)

        Returns:
            Tuple of (valid, normalized_relative_path, error_code)
            - valid: True if path passes basic validation
            - normalized_relative_path: Path relative to project_root (forward slashes)
            - error_code: Error code if validation fails, None otherwise
        """
        path_obj = Path(path)

        # Resolve path to absolute
        if path_obj.is_absolute():
            try:
                resolved = path_obj.resolve()
                relative_path = resolved.relative_to(self.project_root)
                normalized = self._normalize_path(str(relative_path))
            except ValueError:
                return False, None, "FIREWALL_PATH_ESCAPE"
        else:
            try:
                resolved = (self.project_root / path).resolve()
                relative_path = resolved.relative_to(self.project_root)
                normalized = self._normalize_path(str(relative_path))
            except ValueError:
                return False, None, "FIREWALL_PATH_ESCAPE"

        # Check for path traversal in original path
        if ".." in Path(path).parts:
            return False, None, "FIREWALL_PATH_TRAVERSAL"

        return True, normalized, None

    def _check_path_domain(self, normalized_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Check which domain (tmp/durable/excluded/none) a path belongs to.

        Args:
            normalized_path: Normalized relative path

        Returns:
            Tuple of (domain, error_code)
            - domain: "tmp", "durable", "excluded", or None
            - error_code: Error code if path is not in any allowed domain
        """
        # Check exclusions first
        for exclusion in self.exclusions:
            if normalized_path.startswith(exclusion + "/") or normalized_path == exclusion:
                return "excluded", None

        # Check tmp roots
        for tmp_root in self.tmp_roots:
            if normalized_path.startswith(tmp_root + "/") or normalized_path == tmp_root:
                return "tmp", None

        # Check durable roots
        for durable_root in self.durable_roots:
            if normalized_path.startswith(durable_root + "/") or normalized_path == durable_root:
                return "durable", None

        # Not in any allowed domain
        return None, "FIREWALL_PATH_NOT_IN_DOMAIN"

    def _validate_write_operation(
        self,
        path: str | Path,
        kind: str
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a write operation against firewall policy.

        Args:
            path: Path to validate
            kind: "tmp" or "durable"

        Returns:
            Tuple of (valid, violation_receipt)
            - valid: True if write is allowed
            - violation_receipt: Violation receipt dict if write is blocked, None otherwise
        """
        # Resolve and validate path
        valid, normalized_path, error_code = self._resolve_and_validate_path(path)
        if not valid:
            return False, self._build_violation_receipt(
                operation="write",
                path=str(path),
                kind=kind,
                error_code=error_code,
                message=f"Path validation failed: {error_code}"
            )

        # Check path domain
        domain, error_code = self._check_path_domain(normalized_path)

        # Excluded paths always fail
        if domain == "excluded":
            return False, self._build_violation_receipt(
                operation="write",
                path=str(path),
                kind=kind,
                error_code="FIREWALL_PATH_EXCLUDED",
                message=f"Path is in exclusion list"
            )

        # Not in any domain
        if domain is None:
            return False, self._build_violation_receipt(
                operation="write",
                path=str(path),
                kind=kind,
                error_code=error_code,
                message=f"Path not in any allowed write domain"
            )

        # Tmp write requested
        if kind == "tmp":
            if domain != "tmp":
                return False, self._build_violation_receipt(
                    operation="write",
                    path=str(path),
                    kind=kind,
                    error_code="FIREWALL_TMP_WRITE_WRONG_DOMAIN",
                    message=f"Tmp write attempted outside tmp roots (in {domain} domain)"
                )
            return True, None

        # Durable write requested
        if kind == "durable":
            if domain != "durable":
                return False, self._build_violation_receipt(
                    operation="write",
                    path=str(path),
                    kind=kind,
                    error_code="FIREWALL_DURABLE_WRITE_WRONG_DOMAIN",
                    message=f"Durable write attempted outside durable roots (in {domain} domain)"
                )

            # Check commit gate
            if not self._commit_gate_open:
                return False, self._build_violation_receipt(
                    operation="write",
                    path=str(path),
                    kind=kind,
                    error_code="FIREWALL_DURABLE_WRITE_BEFORE_COMMIT",
                    message="Durable write attempted before commit gate opened"
                )

            return True, None

        # Invalid kind
        return False, self._build_violation_receipt(
            operation="write",
            path=str(path),
            kind=kind,
            error_code="FIREWALL_INVALID_KIND",
            message=f"Invalid write kind: {kind} (must be 'tmp' or 'durable')"
        )

    def _build_violation_receipt(
        self,
        operation: str,
        path: str,
        kind: Optional[str] = None,
        error_code: str = "FIREWALL_VIOLATION",
        message: str = ""
    ) -> Dict[str, Any]:
        """
        Build a violation receipt.

        Args:
            operation: Operation type (write, mkdir, rename, unlink)
            path: Path(s) involved in operation
            kind: Write kind if applicable
            error_code: Deterministic error code
            message: Human-readable error message

        Returns:
            Violation receipt dict
        """
        receipt = {
            "firewall_version": self.VERSION,
            "tool_version_hash": self._tool_version_hash,
            "verdict": "FAIL",
            "error_code": error_code,
            "message": message,
            "operation": operation,
            "path": path,
            "policy_snapshot": self._get_policy_snapshot(),
        }

        if kind is not None:
            receipt["kind"] = kind

        return receipt

    def safe_write(
        self,
        path: str | Path,
        data: str | bytes,
        kind: str = "tmp"
    ) -> None:
        """
        Perform a write operation with firewall enforcement.

        Args:
            path: Path to write to
            data: Data to write (str or bytes)
            kind: "tmp" or "durable"

        Raises:
            FirewallViolation: If write violates firewall policy
        """
        valid, violation = self._validate_write_operation(path, kind)
        if not valid:
            raise FirewallViolation(violation)

        # Perform the write
        path_obj = Path(path) if not Path(path).is_absolute() else Path(path)
        if not path_obj.is_absolute():
            path_obj = self.project_root / path_obj

        if isinstance(data, str):
            path_obj.write_text(data, encoding="utf-8")
        else:
            path_obj.write_bytes(data)

    def safe_mkdir(
        self,
        path: str | Path,
        kind: str = "tmp",
        parents: bool = True,
        exist_ok: bool = True
    ) -> None:
        """
        Create a directory with firewall enforcement.

        Args:
            path: Directory path to create
            kind: "tmp" or "durable"
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            FirewallViolation: If mkdir violates firewall policy
        """
        valid, violation = self._validate_write_operation(path, kind)
        if not valid:
            raise FirewallViolation(violation)

        # Perform the mkdir
        path_obj = Path(path) if not Path(path).is_absolute() else Path(path)
        if not path_obj.is_absolute():
            path_obj = self.project_root / path_obj

        path_obj.mkdir(parents=parents, exist_ok=exist_ok)

    def safe_rename(
        self,
        src: str | Path,
        dst: str | Path
    ) -> None:
        """
        Rename a file or directory with firewall enforcement.

        Both src and dst must be in allowed domains and follow write rules.

        Args:
            src: Source path
            dst: Destination path

        Raises:
            FirewallViolation: If rename violates firewall policy
        """
        # Validate src
        _, src_normalized, src_error = self._resolve_and_validate_path(src)
        if src_error:
            violation = self._build_violation_receipt(
                operation="rename",
                path=f"{src} -> {dst}",
                error_code=src_error,
                message=f"Source path validation failed: {src_error}"
            )
            raise FirewallViolation(violation)

        # Validate dst - determine kind from src domain
        src_domain, _ = self._check_path_domain(src_normalized)
        kind = src_domain if src_domain in ["tmp", "durable"] else "tmp"

        valid, violation = self._validate_write_operation(dst, kind)
        if not valid:
            violation["operation"] = "rename"
            violation["path"] = f"{src} -> {dst}"
            raise FirewallViolation(violation)

        # Perform the rename
        src_obj = Path(src) if not Path(src).is_absolute() else Path(src)
        dst_obj = Path(dst) if not Path(dst).is_absolute() else Path(dst)

        if not src_obj.is_absolute():
            src_obj = self.project_root / src_obj
        if not dst_obj.is_absolute():
            dst_obj = self.project_root / dst_obj

        src_obj.rename(dst_obj)

    def safe_unlink(
        self,
        path: str | Path
    ) -> None:
        """
        Delete a file with firewall enforcement.

        Path must be in an allowed domain.

        Args:
            path: Path to delete

        Raises:
            FirewallViolation: If unlink violates firewall policy
        """
        # Validate path
        valid, normalized_path, error_code = self._resolve_and_validate_path(path)
        if not valid:
            violation = self._build_violation_receipt(
                operation="unlink",
                path=str(path),
                error_code=error_code,
                message=f"Path validation failed: {error_code}"
            )
            raise FirewallViolation(violation)

        # Check path domain
        domain, error_code = self._check_path_domain(normalized_path)

        if domain is None or domain == "excluded":
            violation = self._build_violation_receipt(
                operation="unlink",
                path=str(path),
                error_code=error_code or "FIREWALL_PATH_EXCLUDED",
                message="Path not in allowed domain or is excluded"
            )
            raise FirewallViolation(violation)

        # Perform the unlink
        path_obj = Path(path) if not Path(path).is_absolute() else Path(path)
        if not path_obj.is_absolute():
            path_obj = self.project_root / path_obj

        path_obj.unlink()


class FirewallViolation(Exception):
    """Exception raised when a firewall policy is violated."""

    def __init__(self, violation_receipt: Dict[str, Any]):
        """
        Initialize FirewallViolation exception.

        Args:
            violation_receipt: Violation receipt dictionary
        """
        self.violation_receipt = violation_receipt
        self.error_code = violation_receipt["error_code"]
        self.message = violation_receipt["message"]
        super().__init__(f"[{self.error_code}] {self.message}")

    def to_json(self) -> str:
        """
        Serialize violation receipt to JSON.

        Returns:
            JSON string of violation receipt
        """
        return json.dumps(self.violation_receipt, sort_keys=True, indent=2)

    def write_receipt(self, receipt_path: Path) -> None:
        """
        Write violation receipt to a file.

        Args:
            receipt_path: Path to write receipt JSON
        """
        receipt_path.write_text(self.to_json(), encoding="utf-8")
