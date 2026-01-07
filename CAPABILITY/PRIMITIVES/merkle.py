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


# =============================================================================
# Merkle Membership Proofs (Phase 1.7.3)
# =============================================================================


class MerkleProof:
    """
    A membership proof for a single file in a Merkle tree.

    Contains the sibling hashes needed to reconstruct the root from a leaf.
    Each step is a tuple of (sibling_hash, is_left) where is_left indicates
    whether the sibling is on the left (True) or right (False).
    """

    __slots__ = ("path", "bytes_hash", "steps")

    def __init__(self, path: str, bytes_hash: str, steps: list[tuple[str, bool]]):
        self.path = path
        self.bytes_hash = bytes_hash
        self.steps = steps  # List of (sibling_hash, sibling_is_left)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "path": self.path,
            "bytes_hash": self.bytes_hash,
            "steps": [{"sibling": s, "sibling_is_left": left} for s, left in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MerkleProof":
        """Deserialize from dict."""
        steps = [(step["sibling"], step["sibling_is_left"]) for step in d["steps"]]
        return cls(d["path"], d["bytes_hash"], steps)


def build_manifest_with_proofs(manifest: dict[str, str]) -> tuple[str, dict[str, MerkleProof]]:
    """
    Build Merkle root AND membership proofs for each file.

    Args:
        manifest: { normalized_path -> bytes_hash (sha256 hex) }

    Returns:
        (root_hash, proofs) where proofs[path] = MerkleProof for that file.

    This enables proving "file X with hash Y was in the manifest" without
    revealing the full manifest - you only need to reveal the sibling hashes
    along the path from leaf to root.
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
    n = len(normalized_items)

    # Build leaf hashes
    leaves = [_leaf_hash(p, h) for p, h in normalized_items]

    # Initialize proof steps for each leaf
    proof_steps: list[list[tuple[str, bool]]] = [[] for _ in range(n)]

    # Track which original leaves each position in current level maps to
    # Each position can map to multiple original leaves (after merging)
    pos_to_leaves: list[set[int]] = [{i} for i in range(n)]

    level = leaves[:]

    while len(level) > 1:
        # Pad if odd
        if len(level) % 2 == 1:
            level.append(level[-1])
            pos_to_leaves.append(pos_to_leaves[-1].copy())

        next_level = []
        next_pos_to_leaves: list[set[int]] = []

        for i in range(0, len(level), 2):
            left_hash = level[i]
            right_hash = level[i + 1]
            left_leaves = pos_to_leaves[i]
            right_leaves = pos_to_leaves[i + 1]

            # For all leaves in the left subtree, their sibling is on the right
            for leaf_idx in left_leaves:
                proof_steps[leaf_idx].append((right_hash, False))

            # For all leaves in the right subtree, their sibling is on the left
            # (only if right subtree has different leaves - not padding duplicate)
            if right_leaves != left_leaves:
                for leaf_idx in right_leaves:
                    proof_steps[leaf_idx].append((left_hash, True))

            parent_hash = _node_hash(left_hash, right_hash)
            next_level.append(parent_hash)
            # Parent position tracks all leaves from both children
            next_pos_to_leaves.append(left_leaves | right_leaves)

        level = next_level
        pos_to_leaves = next_pos_to_leaves

    root = level[0]

    # Build proof objects
    proofs: dict[str, MerkleProof] = {}
    for i, (path, bytes_hash) in enumerate(normalized_items):
        proofs[path] = MerkleProof(path, bytes_hash, proof_steps[i])

    return root, proofs


def verify_membership(
    path: str,
    bytes_hash: str,
    proof: MerkleProof | dict,
    expected_root: str,
) -> bool:
    """
    Verify that a file was in the manifest that produced the given root.

    Args:
        path: The file path (must match proof.path)
        bytes_hash: The file's content hash (must match proof.bytes_hash)
        proof: MerkleProof object or dict representation
        expected_root: The Merkle root to verify against

    Returns:
        True if the proof is valid, False otherwise.

    This allows proving membership without revealing the full manifest.
    """
    expected_root = _validate_sha256_hex(expected_root, label="expected_root")
    bytes_hash = _validate_sha256_hex(bytes_hash, label="bytes_hash")

    if isinstance(proof, dict):
        proof = MerkleProof.from_dict(proof)

    # Verify path and hash match the proof
    if proof.path != path or proof.bytes_hash != bytes_hash:
        return False

    # Start with the leaf hash
    current = _leaf_hash(path, bytes_hash)

    # Walk up the tree using sibling hashes
    for sibling_hash, sibling_is_left in proof.steps:
        sibling_hash = _validate_sha256_hex(sibling_hash, label="sibling_hash")
        if sibling_is_left:
            current = _node_hash(sibling_hash, current)
        else:
            current = _node_hash(current, sibling_hash)

    return current == expected_root

