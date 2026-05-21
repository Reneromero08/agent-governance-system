"""Test the Rust FFI bekenstein_sweep"""
import sys
import os
import numpy as np
from pathlib import Path

rust_dir = str(Path(__file__).parent / "target" / "release")
sys.path.insert(0, rust_dir)
os.chdir(rust_dir)

try:
    import catalytic_ffi
    print("catalytic_ffi loaded")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

rng = np.random.RandomState(42)
tape_data = rng.bytes(2 * 1024 * 1024)

print("Running 5000 solves per depth...")
result = catalytic_ffi.bekenstein_sweep(tape_data, [4, 6, 8, 10], 5000)

print("=" * 78)
print("RUST FFI — BEKENSTEIN VIOLATOR")
print("=" * 78)
for k, v in result.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6e}")
    else:
        print(f"  {k}: {v}")
