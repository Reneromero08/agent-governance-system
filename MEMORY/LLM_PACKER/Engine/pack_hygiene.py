from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PackLimits:
    max_total_bytes: int
    max_entry_bytes: int
    max_entries: int
    allow_duplicate_hashes: bool


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_state_content_sha256(repo_state: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(repo_state))


def _require_int_limits(limits: PackLimits) -> None:
    if limits.max_total_bytes <= 0:
        raise ValueError("PACK_LIMIT_INVALID:max_total_bytes")
    if limits.max_entry_bytes <= 0:
        raise ValueError("PACK_LIMIT_INVALID:max_entry_bytes")
    if limits.max_entries <= 0:
        raise ValueError("PACK_LIMIT_INVALID:max_entries")


def validate_repo_state_manifest(
    repo_state: Dict[str, Any],
    *,
    allow_duplicate_hashes: bool,
) -> None:
    files = repo_state.get("files")
    if not isinstance(files, list):
        raise ValueError("PACK_MANIFEST_INVALID:files_not_list")

    seen_paths: set[str] = set()
    seen_hash_to_path: Dict[str, str] = {}

    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"PACK_MANIFEST_INVALID:entry_not_object:{idx}")
        path = entry.get("path")
        hash_hex = entry.get("hash")
        size = entry.get("size")
        if not isinstance(path, str) or not path:
            raise ValueError(f"PACK_MANIFEST_INVALID:missing_path:{idx}")
        if not isinstance(hash_hex, str) or len(hash_hex) != 64:
            raise ValueError(f"PACK_MANIFEST_INVALID:missing_hash:{idx}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"PACK_MANIFEST_INVALID:missing_size:{idx}")

        if path in seen_paths:
            raise ValueError(f"PACK_DEDUP_DUPLICATE_PATH:{path}")
        seen_paths.add(path)

        other_path = seen_hash_to_path.get(hash_hex)
        if other_path is not None and other_path != path and not allow_duplicate_hashes:
            raise ValueError(f"PACK_DEDUP_DUPLICATE_HASH:{hash_hex}:{other_path}:{path}")
        seen_hash_to_path.setdefault(hash_hex, path)

    expected = sorted(files, key=lambda e: (e["path"], e["hash"]))
    if files != expected:
        raise ValueError("PACK_MANIFEST_INVALID:unsorted_files")


def enforce_included_repo_limits(
    included_entries: Sequence[Dict[str, Any]],
    *,
    limits: PackLimits,
) -> Dict[str, int]:
    _require_int_limits(limits)

    if len(included_entries) > limits.max_entries:
        raise ValueError("PACK_LIMIT_EXCEEDED:max_entries")

    total_bytes = 0
    max_entry_observed = 0
    for entry in included_entries:
        size = entry.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError("PACK_MANIFEST_INVALID:entry_size")
        if size > limits.max_entry_bytes:
            raise ValueError("PACK_LIMIT_EXCEEDED:max_entry_bytes")
        total_bytes += size
        if size > max_entry_observed:
            max_entry_observed = size

    if total_bytes > limits.max_total_bytes:
        raise ValueError("PACK_LIMIT_EXCEEDED:max_total_bytes")

    return {
        "included_entries": len(included_entries),
        "included_bytes": total_bytes,
        "max_entry_bytes_observed": max_entry_observed,
    }


def pack_dir_total_bytes(pack_dir: Path) -> int:
    total = 0
    for p in pack_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

