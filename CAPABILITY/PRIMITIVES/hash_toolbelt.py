from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional

from .cas_store import CatalyticStore


_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


DEFAULT_READ_MAX_BYTES = 4096
DEFAULT_GREP_MAX_BYTES = 65536
DEFAULT_GREP_MAX_MATCHES = 20
DEFAULT_DESCRIBE_MAX_BYTES = 8192
DEFAULT_AST_MAX_BYTES = 65536
DEFAULT_AST_MAX_NODES = 200
DEFAULT_AST_MAX_DEPTH = 6

DEFAULT_GREP_SNIPPET_CHARS = 120


def _validate_hash(hash_hex: str) -> str:
    if not isinstance(hash_hex, str) or _HASH_RE.fullmatch(hash_hex) is None:
        raise ValueError(f"invalid hash: {hash_hex!r}")
    return hash_hex


def _object_path(store: CatalyticStore, hash_hex: str) -> Path:
    # Avoid depending on private CAS internals by re-deriving the deterministic path.
    hash_hex = _validate_hash(hash_hex)
    # Match cas_store.py layout: cas_dir / h[:2] / h
    return store.objects_dir / hash_hex[0:2] / hash_hex


def _get_size(path: Path) -> int:
    return os.stat(path).st_size


def _verify_object_integrity(path: Path, expected_hash_hex: str, *, chunk_size: int = 1024 * 1024) -> None:
    expected_hash_hex = _validate_hash(expected_hash_hex)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected_hash_hex:
        raise ValueError(f"CAS_OBJECT_INTEGRITY_MISMATCH expected={expected_hash_hex} actual={actual}")


def _read_bytes_range(
    store: CatalyticStore,
    hash_hex: str,
    *,
    max_bytes: int,
    start: int = 0,
    end: Optional[int] = None,
) -> tuple[bytes, int, int, int]:
    if max_bytes <= 0:
        raise ValueError("max-bytes must be positive")
    if start < 0:
        raise ValueError("start must be >= 0")

    path = _object_path(store, hash_hex)
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Fail closed on on-disk corruption: the object content must hash to its key.
    _verify_object_integrity(path, hash_hex)

    total_size = _get_size(path)
    effective_end = total_size if end is None else max(0, min(end, total_size))
    effective_start = min(start, total_size)
    if effective_start > effective_end:
        raise ValueError("start must be <= end")

    length = min(max_bytes, effective_end - effective_start)
    with open(path, "rb") as f:
        f.seek(effective_start)
        data = f.read(length)

    return data, effective_start, effective_start + len(data), total_size


def hash_read_text(
    *,
    store: CatalyticStore,
    hash_hex: str,
    max_bytes: int = DEFAULT_READ_MAX_BYTES,
    start: int = 0,
    end: Optional[int] = None,
) -> str:
    data, effective_start, effective_end, _ = _read_bytes_range(
        store, hash_hex, max_bytes=max_bytes, start=start, end=end
    )
    header = f"hash={hash_hex} start={effective_start} end={effective_end} bytes_returned={len(data)}"
    text = data.decode("utf-8", errors="replace")
    return header + "\n" + text


@dataclass(frozen=True)
class GrepMatch:
    line_number: int
    byte_offset: int
    snippet: str

    def format(self) -> str:
        return f"{self.line_number}:{self.byte_offset}:{self.snippet}"


def hash_grep(
    *,
    store: CatalyticStore,
    hash_hex: str,
    pattern: str,
    max_bytes: int = DEFAULT_GREP_MAX_BYTES,
    max_matches: int = DEFAULT_GREP_MAX_MATCHES,
    snippet_chars: int = DEFAULT_GREP_SNIPPET_CHARS,
) -> list[GrepMatch]:
    if max_matches <= 0:
        raise ValueError("max-matches must be positive")
    if snippet_chars <= 0:
        raise ValueError("snippet length must be positive")

    needle = pattern.encode("utf-8", errors="strict")
    data, _, _, _ = _read_bytes_range(store, hash_hex, max_bytes=max_bytes, start=0, end=None)

    matches: list[GrepMatch] = []
    pos = 0
    while len(matches) < max_matches:
        idx = data.find(needle, pos)
        if idx == -1:
            break
        line_number = data.count(b"\n", 0, idx) + 1
        line_start = data.rfind(b"\n", 0, idx)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        preview_bytes = data[line_start : min(len(data), line_start + snippet_chars)]
        snippet = preview_bytes.decode("utf-8", errors="replace").replace("\n", "\\n")
        matches.append(GrepMatch(line_number=line_number, byte_offset=idx, snippet=snippet))
        pos = idx + max(1, len(needle))

    return matches


def hash_describe(
    *,
    store: CatalyticStore,
    hash_hex: str,
    max_bytes: int = DEFAULT_DESCRIBE_MAX_BYTES,
) -> str:
    data, _, _, total_size = _read_bytes_range(store, hash_hex, max_bytes=max_bytes, start=0, end=None)
    is_binary = b"\x00" in data
    if not is_binary:
        # Heuristic: if too many bytes are non-printable (excluding common whitespace), treat as binary.
        printable = 0
        for b in data:
            if b in (9, 10, 13) or 32 <= b <= 126:
                printable += 1
        is_binary = (len(data) > 0) and (printable / len(data) < 0.85)

    obj = {
        "hash": hash_hex,
        "sha256": hash_hex,
        "total_size": total_size,
        "bytes_preview_len": len(data),
        "first_k_bytes_preview": data.decode("utf-8", errors="replace"),
        "type_guess": "binary" if is_binary else "text",
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ast_outline(node: ast.AST, *, max_nodes: int, max_depth: int) -> tuple[list[dict], bool]:
    items: list[dict] = []
    truncated = False

    def visit(n: ast.AST, depth: int) -> None:
        nonlocal truncated
        if truncated:
            return
        if depth > max_depth:
            return
        if len(items) >= max_nodes:
            truncated = True
            return

        if isinstance(n, ast.Import):
            for alias in n.names:
                if len(items) >= max_nodes:
                    truncated = True
                    return
                items.append({"type": "import", "name": alias.name, "asname": alias.asname or ""})
        elif isinstance(n, ast.ImportFrom):
            items.append({"type": "import_from", "module": n.module or "", "level": n.level})
        elif isinstance(n, ast.FunctionDef):
            items.append({"type": "def", "name": n.name})
        elif isinstance(n, ast.AsyncFunctionDef):
            items.append({"type": "async_def", "name": n.name})
        elif isinstance(n, ast.ClassDef):
            items.append({"type": "class", "name": n.name})

        for child in ast.iter_child_nodes(n):
            if isinstance(n, ast.Module):
                visit(child, depth + 1)
            elif depth + 1 <= max_depth:
                # Only traverse into bodies for defs/classes to keep output minimal.
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, depth + 1)

    visit(node, 0)
    return items, truncated


def hash_ast(
    *,
    store: CatalyticStore,
    hash_hex: str,
    max_bytes: int = DEFAULT_AST_MAX_BYTES,
    max_nodes: int = DEFAULT_AST_MAX_NODES,
    max_depth: int = DEFAULT_AST_MAX_DEPTH,
) -> str:
    if max_nodes <= 0:
        raise ValueError("max-nodes must be positive")
    if max_depth < 0:
        raise ValueError("max-depth must be >= 0")

    data, _, _, _ = _read_bytes_range(store, hash_hex, max_bytes=max_bytes, start=0, end=None)

    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("UNSUPPORTED_AST_FORMAT")

    try:
        tree = ast.parse(text)
    except SyntaxError:
        raise ValueError("UNSUPPORTED_AST_FORMAT")

    items, truncated = _ast_outline(tree, max_nodes=max_nodes, max_depth=max_depth)
    if truncated:
        items.append({"type": "TRUNCATED"})

    obj = {
        "hash": hash_hex,
        "max_bytes": max_bytes,
        "max_nodes": max_nodes,
        "max_depth": max_depth,
        "truncated": truncated,
        "outline": items,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _build_dereference_ledger_record(
    *,
    run_id: str,
    timestamp: str,
    command: str,
    hash_hex: str,
    bounds: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a minimal ledger record for a hash dereference event.

    Args:
        run_id: Run identifier
        timestamp: Deterministic timestamp (caller-supplied, may be sentinel)
        command: Command name (read, grep, describe, ast)
        hash_hex: SHA-256 hash being dereferenced
        bounds: Dictionary of bounds used (max_bytes, start, end, max_matches, etc.)

    Returns:
        A ledger record conforming to ledger.schema.json
    """
    # Build minimal JOBSPEC
    jobspec = {
        "job_id": f"deref-{command}-{run_id}",
        "phase": 4,
        "task_type": "adapter_execution",
        "intent": f"expand-by-hash: {command}",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": [],
        "determinism": "deterministic",
    }

    # Build OUTPUTS entry to record dereference metadata
    output_entry = {
        "path": f"deref/{command}/{hash_hex}",
        "type": "file",
        "sha256": hash_hex,
    }

    # Create ledger record
    record = {
        "JOBSPEC": jobspec,
        "RUN_INFO": {
            "run_id": run_id,
            "timestamp": timestamp,
            "intent": f"hash-{command}: {hash_hex} bounds={json.dumps(bounds, sort_keys=True)}",
            "catalytic_domains": [],
            "exit_code": 0,
            "restoration_verified": True,
        },
        "PRE_MANIFEST": {},
        "POST_MANIFEST": {},
        "RESTORE_DIFF": {},
        "OUTPUTS": [output_entry],
        "STATUS": {
            "status": "completed",
            "restoration_verified": True,
            "exit_code": 0,
            "validation_passed": True,
        },
        "VALIDATOR_ID": {
            "validator_semver": "0.1.0",
            "validator_build_id": "phase4-deref-logging",
        },
    }

    return record


def log_dereference_event(
    *,
    run_id: str | None,
    timestamp: str,
    ledger_path: Path,
    command: str,
    hash_hex: str,
    bounds: dict[str, Any],
) -> None:
    """
    Log a dereference event to the ledger if run_id is present.

    Args:
        run_id: Run identifier (if None, skip logging silently)
        timestamp: Deterministic timestamp
        ledger_path: Path to LEDGER.jsonl
        command: Command name (read, grep, describe, ast)
        hash_hex: SHA-256 hash being dereferenced
        bounds: Dictionary of bounds used
    """
    if run_id is None:
        return

    from .ledger import Ledger

    ledger = Ledger(ledger_path)
    record = _build_dereference_ledger_record(
        run_id=run_id,
        timestamp=timestamp,
        command=command,
        hash_hex=hash_hex,
        bounds=bounds,
    )
    ledger.append(record)
