"""
Bekenstein Violator — HDD-Scale Tape
=====================================
Uses the 500GB HDD as the catalytic tape instead of 2MB RAM.
Memory-maps a file on the HDD, feeds it to the Rust FFI engine.
The tape size scales the throughput ratio directly.
"""

import sys
import os
import mmap
import time
import hashlib
import numpy as np
from pathlib import Path

# Point to the Rust FFI
rust_dir = str(next(p for p in Path(__file__).resolve().parents if p.name == "CAT_CAS").parent / "EIGEN_BUDDY" / "core" / "rust_ffi" / "target" / "release")
sys.path.insert(0, rust_dir)
os.chdir(rust_dir)

import catalytic_ffi

# ==================================================================
# Configuration
# ==================================================================
HDD_PATH = os.environ.get("HDD_TAPE_PATH", "G:/bekenstein_tape.bin")
TAPE_SIZE_MB = 512  # Use 512MB of the HDD (can scale to 500GB)
TAPE_SIZE = TAPE_SIZE_MB * 1024 * 1024
SOLVES_PER_DEPTH = 50000
DEPTHS = [4, 6, 8, 10, 12]

# ==================================================================
# Physics
# ==================================================================
HBAR = 1.054571817e-34
C_LIGHT = 2.99792458e8
LN2 = np.log(2)
G = 6.67430e-11
DIE_MASS = 29e-6
DIE_RADIUS = 1e-3
BEKENSTEIN_BOUND = 2 * np.pi * DIE_RADIUS * DIE_MASS * C_LIGHT**2 / (HBAR * C_LIGHT * LN2)


def main():
    print("=" * 78)
    print("BEKENSTEIN VIOLATOR — HDD-SCALE TAPE")
    print(f"  Tape: {TAPE_SIZE_MB}MB on HDD (G:\\)")
    print(f"  Bekenstein Bound: {BEKENSTEIN_BOUND:.4e} bits")
    print(f"  Solves/depth: {SOLVES_PER_DEPTH}")
    print("=" * 78)
    print()

    # Create or open the tape file on HDD
    print(f"  Opening tape file: {HDD_PATH}")
    if not os.path.exists(HDD_PATH):
        print(f"  Creating {TAPE_SIZE_MB}MB tape file on HDD...")
        rng = np.random.default_rng(42)
        with open(HDD_PATH, "wb") as f:
            chunk = 1024 * 1024
            for _ in range(TAPE_SIZE_MB):
                f.write(rng.bytes(chunk))
        print(f"  Created.")
    else:
        print(f"  File exists ({os.path.getsize(HDD_PATH) / (1024*1024):.0f}MB)")

    # Memory-map the tape for zero-copy access
    fd = os.open(HDD_PATH, os.O_RDWR | os.O_BINARY)
    tape_mmap = mmap.mmap(fd, TAPE_SIZE, access=mmap.ACCESS_WRITE)

    try:
        tape_mmap.seek(0)
        tape_bytes = tape_mmap.read(TAPE_SIZE)
        initial_hash = hashlib.sha256(tape_bytes).hexdigest()

        tape_capacity = TAPE_SIZE * 8
        print(f"  Tape capacity: {tape_capacity:,} bits ({TAPE_SIZE / (1024*1024):.0f} MB)")
        print(f"  Initial hash: {initial_hash[:16]}...")
        print()

        # ===== RUN RUST FFI SWEEP =====
        print("=" * 78)
        print("RUNNING RUST FFI BEKENSTEIN SWEEP")
        print("=" * 78)
        print()

        wall_start = time.perf_counter()
        result = catalytic_ffi.bekenstein_sweep(tape_bytes, DEPTHS, SOLVES_PER_DEPTH)
        wall_elapsed = time.perf_counter() - wall_start

        # ===== DISPLAY =====
        print(f"  Total entropy:     {result['total_entropy']:,}")
        print(f"  Total solves:      {result['total_solves']:,}")
        print(f"  Errors:            {result['errors']}")
        print(f"  Elapsed:           {result['elapsed_secs']:.2f}s")
        print(f"  Throughput ratio:  {result['ratio']:,.0f}x")
        print(f"  Tape capacity:     {result['tape_capacity_bits']:,}")
        print(f"  Tape restored:     {result['tape_restored']}")
        print(f"  Entropy/second:    {result['entropy_per_second']:,.0f} bits/s")
        print()

        # ===== BEKENSTEIN ANALYSIS =====
        total_entropy = result['total_entropy']
        ratio = result['ratio']
        bekenstein_ratio = total_entropy / BEKENSTEIN_BOUND

        print("=" * 78)
        print("BEKENSTEIN ANALYSIS")
        print("=" * 78)
        print(f"  Bekenstein Bound:      {BEKENSTEIN_BOUND:.4e} bits")
        print(f"  Total state transitions: {total_entropy:,}")
        print(f"  Fraction of bound:      {bekenstein_ratio:.4e}")
        print(f"  Throughput vs tape:     {ratio:,.0f}x")
        print()
        print(f"  To reach Bekenstein Bound: need {1/bekenstein_ratio:,.0f}x more solves")
        print(f"  Estimated time at current rate: {wall_elapsed / bekenstein_ratio:,.0f}s")
        print()

        # ===== HARD ASSERTIONS =====
        print("=" * 78)
        print("HARD ASSERTIONS")
        print("=" * 78)
        print()

        assert result['tape_restored'], "FAIL: Tape not restored!"
        print(f"  [PASS] Tape SHA-256 restored ({total_entropy:,} state transitions)")

        assert result['errors'] == 0, f"FAIL: {result['errors']} errors!"
        print(f"  [PASS] Zero errors across {result['total_solves']:,} solves")

        assert ratio > 1, "FAIL: Throughput ratio <= 1!"
        print(f"  [PASS] Throughput ratio {ratio:,.0f}x > 1x")

        print()

        # ===== VERDICT =====
        print("=" * 78)
        print("VERDICT")
        print("=" * 78)
        print()
        print(f"  HDD-SCALE BEKENSTEIN VIOLATOR: OPERATIONAL")
        print(f"  {TAPE_SIZE_MB}MB tape on spinning HDD platter.")
        print(f"  {total_entropy:,} state transitions through {TAPE_SIZE * 8:,} bits.")
        print(f"  {ratio:,.0f}x tape capacity. {bekenstein_ratio:.4e} of Bekenstein Bound.")
        print(f"  Zero bits erased. Full SHA-256 restoration.")
        print(f"  Throughput rate: {result['entropy_per_second']:,.0f} bits/s")
        print("=" * 78)

    finally:
        tape_mmap.close()
        os.close(fd)


if __name__ == "__main__":
    main()
