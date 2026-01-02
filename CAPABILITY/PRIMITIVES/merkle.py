from __future__ import annotations

import hashlib
import re

from .cas_store import normalize_relpath


_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_sha256_hex(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _leaf_hash(path: str, bytes_hash: str) -> str:
    preimage = f"{path}:{bytes_hash}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def _node_hash(left: str, right: str) -> str:
    preimage = (left + right).encode("ascii")
    return hashlib.sha256(preimage).hexdigest()


def build_manifest_root(manifest: dict[str, str]) -> str:
    """
    Compute a deterministic Merkle root for a domain manifest:
      { normalized_path (posix, repo-relative) -> bytes_hash (sha256 hex) }

    Rules:
    - Paths must be normalized already (normalize_relpath(path) == path), else reject.
    - Reject duplicate hashes bound to different paths.
    - Leaf ordering is strict lexicographic ordering by normalized_path.
    - Leaf hash is sha256(path + ":" + bytes_hash).
    - Internal hash is sha256(left_child_hash + right_child_hash).
    - Odd leaf count: duplicate last leaf (padding).
    - Empty manifest is forbidden.
    """
    if not manifest:
        raise ValueError("empty manifest is forbidden")

    normalized_items: list[tuple[str, str]] = []
    hash_to_path: dict[str, str] = {}

    for raw_path, raw_hash in manifest.items():
        normalized_path = normalize_relpath(raw_path)
        if raw_path != normalized_path:
            raise ValueError(f"non-normalized path: {raw_path!r} (expected {normalized_path!r})")

        bytes_hash = _validate_sha256_hex(raw_hash, label="bytes_hash")
        existing_path = hash_to_path.get(bytes_hash)
        if existing_path is not None and existing_path != normalized_path:
            raise ValueError(f"duplicate bytes_hash bound to multiple paths: {bytes_hash}")
        hash_to_path[bytes_hash] = normalized_path
        normalized_items.append((normalized_path, bytes_hash))

    normalized_items.sort(key=lambda t: t[0])
    level = [_leaf_hash(p, h) for p, h in normalized_items]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]

    return level[0]


def verify_manifest_root(manifest: dict[str, str], expected_root: str) -> bool:
    expected_root = _validate_sha256_hex(expected_root, label="expected_root")
    return build_manifest_root(manifest) == expected_root

