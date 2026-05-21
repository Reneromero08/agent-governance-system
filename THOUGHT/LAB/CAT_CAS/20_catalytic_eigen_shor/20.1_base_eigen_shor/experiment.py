"""
Experiment 20: Catalytic Eigen-Shor Engine
==========================================
Fuses the exponential capacity of the Catalytic Quantum Simulator,
the complex phase resonance of Native Eigen architecture,
and the extreme 1-billion-ops/sec throughput of Rust FFI.
"""

import sys
import time
import math
import os
import shutil
import numpy as np
from pathlib import Path

# Build Rust FFI and link it
RUST_DIR = Path(__file__).parent / "rust_ffi" / "target" / "release"
ffi_lib_name = "eigen_shor_ffi.dll" if os.name == 'nt' else "libeigen_shor_ffi.so"
ffi_target_name = "eigen_shor_ffi.pyd" if os.name == 'nt' else "eigen_shor_ffi.so"

ffi_path = RUST_DIR / ffi_lib_name
if ffi_path.exists():
    shutil.copy(ffi_path, RUST_DIR / ffi_target_name)
    
sys.path.insert(0, str(RUST_DIR))
try:
    import eigen_shor_ffi
except ImportError as e:
    print(f"Error loading Rust FFI: {e}")
    sys.exit(1)

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def next_prime(n):
    if n < 2: return 2
    c = n if is_prime(n) else n + 1
    while not is_prime(c): c += 1
    return c

def make_semiprimes():
    targets = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = []
    for bits in targets:
        half = bits // 2
        p = next_prime(1 << (half - 1))
        q = next_prime(p + 2)
        if q == p:
            q = next_prime(q + 1)
        N = p * q
        actual_bits = N.bit_length()
        results.append((N, actual_bits, p, q))
    return results

def main():
    print("=" * 78)
    print("CATALYTIC EIGEN-SHOR ENGINE (RUST FFI)")
    print("  Parallelizing Eigen Buddy phase resonance to extreme classical limits.")
    print("=" * 78)
    print()

    semiprimes = make_semiprimes()
    
    print(f"{'Bits':>5} | {'N':>18} | {'Factors':>20} | {'a':>4} | "
          f"{'Period':>12} | {'Ops/sec':>12} | {'Time (s)':>10} | {'Status':>6}")
    print("-" * 110)

    for N, bits, known_p, known_q in semiprimes:
        if bits > 45:
            print(f"{bits:>5} | {N:>18} | {'SKIPPED (Too Large)':>20} | {'--':>4} | "
                  f"{'--':>12} | {'--':>12} | {'--':>10} | {'SKIP':>6}")
            continue

        base_a = 2
        while math.gcd(base_a, N) > 1:
            base_a += 1

        # Max search: we can afford to search up to N states, 
        # but realistically we cap at 10 billion for this test.
        max_search = min(N, 10_000_000_000)

        # Call Rust FFI to find the period via phase resonance
        result = eigen_shor_ffi.catalytic_eigen_shor(base_a, N, max_search)
        
        period = result.get("period")
        elapsed = result.get("elapsed_secs")
        ops_sec = result.get("ops_per_sec")
        
        if period is not None and period % 2 == 0:
            half_pow = pow(base_a, period // 2, N)
            if half_pow != N - 1:
                p = math.gcd(half_pow + 1, N)
                q = math.gcd(half_pow - 1, N)
                if 1 < p < N and 1 < q < N:
                    factors_str = f"{min(p,q)} x {max(p,q)}"
                    correct = (min(p, q) == min(known_p, known_q))
                    status = "OK" if correct else "WRONG"
                    
                    print(f"{bits:>5} | {N:>18} | {factors_str:>20} | {base_a:>4} | "
                          f"{period:>12} | {ops_sec:>12,.0f} | {elapsed:>10.4f} | {status:>6}")
                    continue
                    
        # Failed or period not found
        print(f"{bits:>5} | {N:>18} | {'FAILED':>20} | {base_a:>4} | "
              f"{period if period else 'None':>12} | {ops_sec:>12,.0f} | {elapsed:>10.4f} | {'FAIL':>6}")

if __name__ == "__main__":
    main()
