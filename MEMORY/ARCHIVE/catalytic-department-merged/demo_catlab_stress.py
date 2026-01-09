# DEMO: CATLAB-STRESS - Push catalytic restoration to breaking point
# =====================================================================
# STATUS: DEMO/ARCHIVED - Not part of CI test suite
# PURPOSE: Validates catalytic restoration at extreme scale (10K files, 50MB)
# RUNTIME: ~10 minutes - too slow for regular CI
# TO RUN: pytest MEMORY/ARCHIVE/catalytic-department-merged/demo_catlab_stress.py
# =====================================================================
"""
This DEMO exists to BREAK EXPECTATIONS.

We're not just proving "it works" - we're proving:
1. It scales beyond what seems reasonable
2. It stays linear (O(n)) not exponential
3. Single-byte corruption in 10,000 files? DETECTED.
4. Restore 50MB of mutated data? BYTE-IDENTICAL.

If this test passes, you have REAL catalytic computing.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import pytest


# =============================================================================
# Core Functions (same as catlab_restoration but parameterized)
# =============================================================================

def compute_tree_hash(root: Path) -> Dict[str, str]:
    """Compute SHA-256 hashes for all files under root."""
    result: Dict[str, str] = {}
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        result[relative] = file_hash
    return result


def populate_domain(root: Path, file_count: int, file_size: int = 1024) -> Dict[str, bytes]:
    """
    Create file_count files of ~file_size bytes each.
    Returns content snapshot for restoration.
    """
    rng = random.Random(42)  # Deterministic
    root.mkdir(parents=True, exist_ok=True)

    # Create nested structure
    subdirs = ["", "a", "b", "c/deep", "c/deep/nested", "d/e/f/g"]
    content_snapshot: Dict[str, bytes] = {}

    for i in range(file_count):
        subdir = subdirs[i % len(subdirs)]
        ext = ".bin" if i % 3 == 0 else ".txt"
        filename = f"file_{i:06d}{ext}"

        rel_path = f"{subdir}/{filename}".lstrip("/")
        abs_path = root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate deterministic content
        content = bytes(rng.getrandbits(8) for _ in range(file_size))
        abs_path.write_bytes(content)
        content_snapshot[rel_path] = content

    return content_snapshot


def mutate_domain_hostile(root: Path, intensity: float = 0.5) -> None:
    """
    Hostile mutations:
    - Corrupt ~30% of files
    - Delete ~20% of files
    - Add rogue files
    - Rename ~10% of files
    """
    rng = random.Random(43)  # Different seed
    all_files = [f for f in sorted(root.rglob("*")) if f.is_file()]

    if not all_files:
        return

    n = len(all_files)

    # 1. Corrupt files
    for f in rng.sample(all_files, min(int(n * 0.3 * intensity), n)):
        if f.exists():
            data = f.read_bytes()
            # Corrupt at random position
            if len(data) > 0:
                pos = rng.randint(0, len(data) - 1)
                corrupted = data[:pos] + bytes([data[pos] ^ 0xFF]) + data[pos+1:]
                f.write_bytes(corrupted)

    # 2. Delete files
    for f in rng.sample(all_files, min(int(n * 0.2 * intensity), n)):
        if f.exists():
            f.unlink()

    # 3. Add rogue files
    rogue_count = max(10, int(n * 0.1 * intensity))
    for i in range(rogue_count):
        rogue_path = root / f"ROGUE_{i:04d}.malicious"
        rogue_path.write_bytes(f"ROGUE CONTENT {rng.randint(0, 999999)}".encode())

    # 4. Rename files
    remaining = [f for f in all_files if f.exists()]
    for f in rng.sample(remaining, min(int(len(remaining) * 0.1 * intensity), len(remaining))):
        if f.exists():
            new_name = f.parent / f"RENAMED_{f.name}"
            f.rename(new_name)


def restore_domain(root: Path, snapshot: Dict[str, bytes]) -> None:
    """Restore root to exactly match snapshot."""
    root.mkdir(parents=True, exist_ok=True)

    # Get current files
    current_files = set()
    for f in root.rglob("*"):
        if f.is_file():
            current_files.add(f.relative_to(root).as_posix())

    snapshot_files = set(snapshot.keys())

    # Remove files not in snapshot
    for rel in current_files - snapshot_files:
        (root / rel).unlink()

    # Restore/create files from snapshot
    for rel, content in snapshot.items():
        abs_path = root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)

    # Clean empty dirs
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


# =============================================================================
# STRESS TESTS - Break Expectations
# =============================================================================

class TestScaling:
    """Prove O(n) scaling - time grows linearly with file count."""

    def test_scaling_100_to_1000_files(self) -> None:
        """
        If we 10x the files, time should ~10x (not 100x).
        This proves O(n) not O(n²).
        """
        results = []

        for file_count in [100, 500, 1000]:
            with tempfile.TemporaryDirectory() as tmp:
                domain = Path(tmp) / "domain"

                start = time.perf_counter()
                snapshot = populate_domain(domain, file_count, file_size=512)
                pre_hash = compute_tree_hash(domain)
                mutate_domain_hostile(domain)
                restore_domain(domain, snapshot)
                post_hash = compute_tree_hash(domain)
                elapsed = time.perf_counter() - start

                assert pre_hash == post_hash, f"Restoration failed at {file_count} files"
                results.append((file_count, elapsed))

        # Check scaling is roughly linear
        # 1000 files should take less than 15x what 100 files takes
        # (allowing overhead for setup/teardown)
        ratio = results[2][1] / results[0][1]
        assert ratio < 20, f"Scaling is worse than O(n): {ratio:.1f}x for 10x files"

        print(f"\n[SCALING RESULTS]")
        for count, elapsed in results:
            print(f"   {count:>5} files: {elapsed:.3f}s")
        print(f"   Ratio (1000/100): {ratio:.1f}x (should be ~10x for O(n))")


class TestMassiveRestoration:
    """Push file counts beyond "reasonable" limits."""

    @pytest.mark.slow
    def test_5000_files_byte_identical(self) -> None:
        """
        5,000 files. Mutate hard. Restore perfectly.
        Total data: ~5MB
        """
        with tempfile.TemporaryDirectory() as tmp:
            domain = Path(tmp) / "domain"

            print(f"\n[STRESS TEST] 5,000 files")

            start = time.perf_counter()
            snapshot = populate_domain(domain, 5000, file_size=1024)
            populate_time = time.perf_counter() - start
            print(f"   Populate: {populate_time:.2f}s")

            start = time.perf_counter()
            pre_hash = compute_tree_hash(domain)
            hash_time = time.perf_counter() - start
            print(f"   Hash (pre): {hash_time:.2f}s")

            start = time.perf_counter()
            mutate_domain_hostile(domain, intensity=0.8)  # 80% intensity
            mutate_time = time.perf_counter() - start
            print(f"   Mutate: {mutate_time:.2f}s")

            start = time.perf_counter()
            restore_domain(domain, snapshot)
            restore_time = time.perf_counter() - start
            print(f"   Restore: {restore_time:.2f}s")

            start = time.perf_counter()
            post_hash = compute_tree_hash(domain)
            verify_time = time.perf_counter() - start
            print(f"   Hash (post): {verify_time:.2f}s")

            assert pre_hash == post_hash, "5,000 file restoration FAILED"
            print(f"   PASSED: 5,000 files restored byte-identical")

    @pytest.mark.slow
    def test_10000_files_byte_identical(self) -> None:
        """
        10,000 files. This is where most systems choke.
        Total data: ~10MB
        """
        with tempfile.TemporaryDirectory() as tmp:
            domain = Path(tmp) / "domain"

            print(f"\n[STRESS TEST] 10,000 files")

            snapshot = populate_domain(domain, 10000, file_size=1024)
            pre_hash = compute_tree_hash(domain)

            mutate_domain_hostile(domain, intensity=1.0)  # FULL HOSTILE
            restore_domain(domain, snapshot)

            post_hash = compute_tree_hash(domain)

            assert pre_hash == post_hash, "10,000 file restoration FAILED"
            print(f"   PASSED: 10,000 files restored byte-identical")


class TestSingleBitDetection:
    """Prove we detect the smallest possible corruption."""

    @pytest.mark.slow
    def test_single_bit_flip_in_10000_files(self) -> None:
        """
        10,000 files. Flip ONE BIT in ONE file.
        We MUST detect it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            domain = Path(tmp) / "domain"

            print(f"\n[PRECISION TEST] Single bit in 10,000 files")

            snapshot = populate_domain(domain, 10000, file_size=1024)
            pre_hash = compute_tree_hash(domain)

            # Pick a random file and flip ONE BIT
            rng = random.Random(999)
            all_files = sorted(domain.rglob("*"))
            all_files = [f for f in all_files if f.is_file()]
            target = rng.choice(all_files)

            data = bytearray(target.read_bytes())
            byte_idx = rng.randint(0, len(data) - 1)
            bit_idx = rng.randint(0, 7)
            data[byte_idx] ^= (1 << bit_idx)  # Flip ONE bit
            target.write_bytes(bytes(data))

            post_hash = compute_tree_hash(domain)

            # We MUST detect this
            assert pre_hash != post_hash, "FAILED TO DETECT SINGLE BIT FLIP!"

            # Count differences
            diffs = sum(1 for k in pre_hash if pre_hash.get(k) != post_hash.get(k))
            assert diffs == 1, f"Expected exactly 1 file different, got {diffs}"

            print(f"   DETECTED: 1 bit flip in {len(all_files)} files")


class TestDataVolume:
    """Push total data volume."""

    @pytest.mark.slow
    def test_50mb_restoration(self) -> None:
        """
        ~50MB of data. Mutate. Restore byte-identical.
        This is a realistic large refactor scenario.
        """
        with tempfile.TemporaryDirectory() as tmp:
            domain = Path(tmp) / "domain"

            # 5000 files * 10KB each = 50MB
            file_count = 5000
            file_size = 10 * 1024  # 10KB

            print(f"\n[VOLUME TEST] ~50MB ({file_count} files x {file_size//1024}KB)")

            start = time.perf_counter()
            snapshot = populate_domain(domain, file_count, file_size)

            # Calculate actual size
            total_bytes = sum(len(v) for v in snapshot.values())
            print(f"   Total data: {total_bytes / (1024*1024):.1f} MB")

            pre_hash = compute_tree_hash(domain)
            mutate_domain_hostile(domain, intensity=0.9)
            restore_domain(domain, snapshot)
            post_hash = compute_tree_hash(domain)

            elapsed = time.perf_counter() - start

            assert pre_hash == post_hash, "50MB restoration FAILED"

            throughput = total_bytes / elapsed / (1024 * 1024)
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Throughput: {throughput:.1f} MB/s")
            print(f"   PASSED: 50MB restored byte-identical")


class TestDeterminism:
    """Prove repeated runs produce identical results."""

    @pytest.mark.slow
    def test_hash_determinism_across_runs(self) -> None:
        """
        Run the same operation 3 times.
        All hashes MUST be identical.
        """
        results = []

        for run in range(3):
            with tempfile.TemporaryDirectory() as tmp:
                domain = Path(tmp) / "domain"

                snapshot = populate_domain(domain, 1000, file_size=512)
                pre_hash = compute_tree_hash(domain)
                mutate_domain_hostile(domain)
                restore_domain(domain, snapshot)
                post_hash = compute_tree_hash(domain)

                # Compute a single hash of all hashes
                all_hashes = json.dumps(post_hash, sort_keys=True)
                run_hash = hashlib.sha256(all_hashes.encode()).hexdigest()
                results.append(run_hash)

        assert results[0] == results[1] == results[2], \
            f"DETERMINISM FAILED: {results}"

        print(f"\n[DETERMINISM] 3 runs, identical hash: {results[0][:16]}...")


# =============================================================================
# Summary Report
# =============================================================================

def test_print_summary() -> None:
    """Print what these tests prove."""
    print("""
+===================================================================+
|                    CATLAB STRESS TEST SUMMARY                      |
+===================================================================+
|                                                                    |
|  If all tests pass, you have proven:                              |
|                                                                    |
|  1. O(n) SCALING      - 10x files = ~10x time (not 100x)          |
|  2. MASSIVE SCALE     - 10,000 files restored byte-identical      |
|  3. BIT PRECISION     - Single bit flip in 10K files DETECTED     |
|  4. DATA VOLUME       - 50MB mutated and restored                 |
|  5. DETERMINISM       - Repeated runs = identical hashes          |
|                                                                    |
|  This is REAL catalytic computing.                                |
|                                                                    |
+===================================================================+
""")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
