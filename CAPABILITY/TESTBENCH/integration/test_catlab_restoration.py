# CATLAB-01: Catalytic Temporal Integrity Stress Test
"""
Stress-test catalytic temporal integrity: mutate catalytic domains
and prove byte-identical restoration.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

# Fixed seed for deterministic behavior
_CATLAB_SEED = 42


def compute_tree_hash(root: Path) -> Dict[str, str]:
    """
    Compute SHA-256 hashes for all files under root.

    Args:
        root: Directory to recursively scan.

    Returns:
        Dict mapping relative POSIX paths to their SHA-256 hex digests.
        Paths are sorted deterministically.

    Notes:
        - Pure function, no side effects.
        - Ignores directories (files only).
        - Binary-safe (reads in binary mode).
        - Ignores timestamps, permissions, metadata.
    """
    result: Dict[str, str] = {}

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        result[relative] = file_hash

    return result


def _generate_content(rng: random.Random, index: int, binary: bool) -> bytes:
    """Generate deterministic file content based on index."""
    size = rng.randint(64, 1024)
    if binary:
        return bytes(rng.getrandbits(8) for _ in range(size))
    else:
        lines = [f"Line {i} of file {index}: data={rng.randint(0, 999999)}"
                 for i in range(size // 40 + 1)]
        return "\n".join(lines).encode("utf-8")


def populate_catalytic_domain(root: Path, file_count: int) -> Dict[str, bytes]:
    """
    Create a deterministic file tree under root.

    Args:
        root: Directory to populate (will be created if missing).
        file_count: Number of files to create.

    Returns:
        Dict mapping relative POSIX paths to their original content bytes.
        This snapshot can be used for restoration.

    Notes:
        - Creates nested directories.
        - Mix of text (.txt) and binary (.bin) files.
        - Uses fixed seed for reproducibility.
    """
    rng = random.Random(_CATLAB_SEED)
    root.mkdir(parents=True, exist_ok=True)

    # Deterministic directory structure
    subdirs: List[str] = ["", "level1/a", "level1/b", "level2/deep/nested"]
    content_snapshot: Dict[str, bytes] = {}

    for i in range(file_count):
        subdir = subdirs[i % len(subdirs)]
        is_binary = (i % 3 == 0)
        ext = ".bin" if is_binary else ".txt"
        filename = f"file_{i:04d}{ext}"

        rel_path = f"{subdir}/{filename}".lstrip("/")
        abs_path = root / rel_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        content = _generate_content(rng, i, is_binary)
        abs_path.write_bytes(content)
        content_snapshot[rel_path] = content

    return content_snapshot


def mutate_catalytic_domain(root: Path) -> None:
    """
    Perform hostile but deterministic mutations inside root.

    Mutations include:
        - Modify contents of some existing files.
        - Delete some files.
        - Add new files.
        - Rename some files.

    Args:
        root: Directory to mutate.

    Notes:
        - All operations stay strictly inside root.
        - Uses fixed seed for reproducibility.
    """
    rng = random.Random(_CATLAB_SEED + 1)  # Different seed from populate

    all_files: List[Path] = sorted(root.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    if not all_files:
        return

    # 1. Modify some files (corrupt ~30%)
    modify_count = max(1, len(all_files) // 3)
    for f in rng.sample(all_files, min(modify_count, len(all_files))):
        if f.exists():
            corrupted = f"MUTATED at index {rng.randint(0, 9999)}\n".encode()
            f.write_bytes(corrupted + f.read_bytes()[:100])

    # 2. Delete some files (~20%)
    delete_count = max(1, len(all_files) // 5)
    for f in rng.sample(all_files, min(delete_count, len(all_files))):
        if f.exists():
            f.unlink()

    # 3. Add new rogue files
    rogue_paths = [
        root / "rogue_file.txt",
        root / "level1" / "intruder.bin",
        root / "level2" / "deep" / "malicious.dat",
    ]
    for rp in rogue_paths:
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(f"Rogue content {rng.randint(0, 9999)}".encode())

    # 4. Rename some files (~10%)
    remaining = [f for f in all_files if f.exists()]
    rename_count = max(1, len(remaining) // 10)
    for f in rng.sample(remaining, min(rename_count, len(remaining))):
        if f.exists():
            new_name = f.parent / f"renamed_{f.name}"
            f.rename(new_name)


def restore_catalytic_domain(root: Path, snapshot: Dict[str, bytes]) -> None:
    """
    Restore root to match snapshot exactly.

    Args:
        root: Directory to restore.
        snapshot: Dict mapping relative POSIX paths to original content bytes.

    Operations:
        - Remove any files not present in snapshot.
        - Recreate missing files with original content.
        - Overwrite modified files with original content.
        - Restore directory structure as needed.

    Notes:
        - Ignores timestamps and permissions.
        - Directory structure will match snapshot paths.
    """
    root.mkdir(parents=True, exist_ok=True)

    # Get current state
    current_files: set = set()
    for f in root.rglob("*"):
        if f.is_file():
            current_files.add(f.relative_to(root).as_posix())

    snapshot_files = set(snapshot.keys())

    # 1. Remove files not in snapshot
    for rel in current_files - snapshot_files:
        (root / rel).unlink()

    # 2. Restore or create files from snapshot
    for rel, content in snapshot.items():
        abs_path = root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)

    # 3. Clean up empty directories
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


# =============================================================================
# CATLAB-01 Test Artifacts
# =============================================================================

def _write_artifacts(
    artifacts_dir: Path,
    pre_hash: Dict[str, str],
    post_hash: Dict[str, str],
    passed: bool,
) -> None:
    """Write test artifacts: PRE_SNAPSHOT, POST_SNAPSHOT, STATUS."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "PRE_SNAPSHOT.json").write_text(
        json.dumps(pre_hash, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "POST_SNAPSHOT.json").write_text(
        json.dumps(post_hash, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifacts_dir / "STATUS.json").write_text(
        json.dumps({
            "status": "success" if passed else "error",
            "cmp01": "pass" if passed else "fail",
            "phase": "catlab",
        }, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# CATLAB-01 Tests
# =============================================================================

def test_catlab_restoration_pass() -> None:
    """
    Test that restoration produces byte-identical hash after mutation.
    This is the happy path: populate → mutate → restore → verify.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        domain = repo_root / "CONTRACTS" / "_runs" / "_tmp" / "catlab"
        artifacts_dir = repo_root / "CATALYTIC-DPT" / "TESTBENCH" / "catlab_stress" / "_artifacts"

        # Populate and capture content snapshot
        content_snapshot = populate_catalytic_domain(domain, file_count=500)
        pre_hash = compute_tree_hash(domain)

        # Mutate (hostile operations)
        mutate_catalytic_domain(domain)

        # Restore from content snapshot
        restore_catalytic_domain(domain, content_snapshot)

        # Verify byte-identical restoration
        post_hash = compute_tree_hash(domain)

        passed = (pre_hash == post_hash)
        _write_artifacts(artifacts_dir, pre_hash, post_hash, passed)

        assert passed, "Restoration failed: pre_hash != post_hash"


def test_catlab_detects_single_byte_change() -> None:
    """
    Test that a single byte change in one file is detected.
    After restoration, flip one byte and verify hash mismatch.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        domain = repo_root / "CONTRACTS" / "_runs" / "_tmp" / "catlab"
        artifacts_dir = repo_root / "CATALYTIC-DPT" / "TESTBENCH" / "catlab_stress" / "_artifacts"

        content_snapshot = populate_catalytic_domain(domain, file_count=500)
        pre_hash = compute_tree_hash(domain)

        mutate_catalytic_domain(domain)
        restore_catalytic_domain(domain, content_snapshot)

        # Inject single-byte corruption
        rng = random.Random(_CATLAB_SEED + 100)
        all_files = sorted(domain.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        target_file = rng.choice(all_files)

        data = bytearray(target_file.read_bytes())
        if len(data) > 0:
            byte_idx = rng.randint(0, len(data) - 1)
            data[byte_idx] ^= 0xFF  # Flip all bits of one byte
            target_file.write_bytes(bytes(data))

        post_hash = compute_tree_hash(domain)

        detected = (pre_hash != post_hash)
        _write_artifacts(artifacts_dir, pre_hash, post_hash, not detected)

        assert detected, "Single-byte change was NOT detected"


def test_catlab_detects_missing_file() -> None:
    """
    Test that a missing file is detected.
    After restoration, delete one file and verify hash mismatch.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        domain = repo_root / "CONTRACTS" / "_runs" / "_tmp" / "catlab"
        artifacts_dir = repo_root / "CATALYTIC-DPT" / "TESTBENCH" / "catlab_stress" / "_artifacts"

        content_snapshot = populate_catalytic_domain(domain, file_count=500)
        pre_hash = compute_tree_hash(domain)

        mutate_catalytic_domain(domain)
        restore_catalytic_domain(domain, content_snapshot)

        # Delete one file
        rng = random.Random(_CATLAB_SEED + 200)
        all_files = sorted(domain.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        target_file = rng.choice(all_files)
        target_file.unlink()

        post_hash = compute_tree_hash(domain)

        detected = (pre_hash != post_hash)
        _write_artifacts(artifacts_dir, pre_hash, post_hash, not detected)

        assert detected, "Missing file was NOT detected"


def test_catlab_detects_extra_file() -> None:
    """
    Test that an extra file is detected.
    After restoration, create one rogue file and verify hash mismatch.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        domain = repo_root / "CONTRACTS" / "_runs" / "_tmp" / "catlab"
        artifacts_dir = repo_root / "CATALYTIC-DPT" / "TESTBENCH" / "catlab_stress" / "_artifacts"

        content_snapshot = populate_catalytic_domain(domain, file_count=500)
        pre_hash = compute_tree_hash(domain)

        mutate_catalytic_domain(domain)
        restore_catalytic_domain(domain, content_snapshot)

        # Create rogue file inside catalytic domain
        rogue_file = domain / "level1" / "a" / "rogue_intruder.txt"
        rogue_file.parent.mkdir(parents=True, exist_ok=True)
        rogue_file.write_bytes(b"This file should not exist")

        post_hash = compute_tree_hash(domain)

        detected = (pre_hash != post_hash)
        _write_artifacts(artifacts_dir, pre_hash, post_hash, not detected)

        assert detected, "Extra file was NOT detected"


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    tests = [
        ("test_catlab_restoration_pass", test_catlab_restoration_pass),
        ("test_catlab_detects_single_byte_change", test_catlab_detects_single_byte_change),
        ("test_catlab_detects_missing_file", test_catlab_detects_missing_file),
        ("test_catlab_detects_extra_file", test_catlab_detects_extra_file),
    ]

    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: EXCEPTION: {e}")
            failed += 1

    sys.exit(failed)
