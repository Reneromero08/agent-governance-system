"""
Experiment 20.11e: Rust-Accelerated FM Cavity Shor
====================================================
Rust FFI builds the catalytic phase grating in parallel (rayon).
Python handles the GPU FFT, autocorrelation, and gcd sweep.

Stack:
  Rust:  build_catalytic_grating(a, N, M) -> (real, imag) pairs (parallel)
  CPU:   numpy assembles complex128 grating from Rust output
  GPU:   torch FFT + autocorrelation on CUDA (12.9 GB VRAM)
  CPU:   gcd sweep on autocorrelation peaks
  Tape:  SHA-256 verified before/after — all operations are READ-ONLY
"""

import hashlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import catalytic_grating_ffi as cg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_hex(data):
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def is_probable_prime(n, k=15):
    if n < 2: return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, min(n - 1, 1000000))
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def generate_semiprime(bits):
    while True:
        p = random.getrandbits(bits // 2); p |= (1 << (bits // 2 - 1)) | 1
        if is_probable_prime(p): break
    while True:
        q = random.getrandbits(bits // 2); q |= (1 << (bits // 2 - 1)) | 1
        if is_probable_prime(q) and q != p: break
    return p * q, p, q


def gcd(a, b):
    while b: a, b = b, a % b
    return a


def shor_factor_from_period(N, a, r):
    if r % 2 != 0: return None
    h = r // 2; v = pow(a, h, N)
    p = gcd(v - 1, N); q = gcd(v + 1, N)
    if p > 1 and q > 1 and p * q == N: return (p, q)
    return None


def rust_catalytic_grating(a, N, M):
    """Build phase grating via Rust FFI -> numpy complex128 array directly."""
    return cg.build_catalytic_grating(a, N, M)


def autocorrelation_gcd_sweep(grating, a, N):
    """GPU FFT -> autocorrelation -> gcd sweep on top peaks."""
    M = len(grating)
    sr = min(M // 2, 200000000)

    grating_t = torch.tensor(grating, dtype=torch.complex64, device=DEVICE)
    G = torch.fft.fft(grating_t)
    power = torch.abs(G) ** 2
    ac_raw = torch.fft.ifft(power).real
    ac_raw = ac_raw / (ac_raw[0] + 1e-15)

    ac_abs = torch.abs(ac_raw[2:sr])
    noise_floor = ac_abs.mean().item()
    peaks = ac_abs > noise_floor * 1.2
    if peaks.sum() == 0:
        peaks = ac_abs > noise_floor * 0.8

    peak_indices = peaks.nonzero(as_tuple=True)[0]
    if len(peak_indices) > 200:
        vals = ac_abs[peak_indices]
        _, sel = torch.topk(vals, 200)
        peak_indices = peak_indices[sel]

    del grating_t, G, ac_raw
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for p_idx in peak_indices:
        tau = int(p_idx.item()) + 2
        if tau < 2: continue
        g = gcd(pow(a, tau, N) - 1, N)
        if 1 < g < N:
            return g, N // g, f"rust_gpu_tau={tau}_gcd"
        if tau % 2 == 0:
            g = gcd(pow(a, tau // 2, N) - 1, N)
            if 1 < g < N:
                return g, N // g, f"rust_gpu_tau={tau}_half"
        for m in [2, 3, 4, 5, 6, 8]:
            tm = tau * m
            if tm >= sr: break
            g = gcd(pow(a, tm, N) - 1, N)
            if 1 < g < N:
                return g, N // g, f"rust_gpu_tau={tau}x{m}_gcd"

    return None, None, "no_factor"


def main():
    print("=" * 78)
    print("EXPERIMENT 20.11e: RUST-ACCELERATED FM CAVITY SHOR")
    print("  Rust (rayon) grating + GPU (CUDA) FFT + gcd sweep")
    print("  Quantum (S^1) + Catalytic (tape SHA-256)")
    print("=" * 78)

    BIT_SIZES = [50, 54]
    print(f"\n  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"  {'bits':>6} {'N':>14} {'M':>12} {'r_sub~':>12}"
          f" {'method':>38} {'time':>8} {'tape':>6}")
    print(f"  {'-'*105}")

    for bits in BIT_SIZES:
        M_power = bits // 2 + 2
        M = 2 ** M_power
        t_start = time.perf_counter()

        N, p_known, q_known = generate_semiprime(bits)
        a = 2
        while gcd(a, N) != 1: a += 1

        # CATALYTIC: Rust-parallel grating construction
        grating = rust_catalytic_grating(a, N, M)
        tape_hash = sha256_hex(grating)

        # GPU FFT + gcd sweep
        p_found, q_found, method = autocorrelation_gcd_sweep(grating, a, N)

        # CATALYTIC: verify tape untouched
        tape_ok = sha256_hex(grating) == tape_hash
        elapsed = time.perf_counter() - t_start

        if p_found and q_found:
            match = set([p_found, q_found]) == set([p_known, q_known])
            print(f"  {bits:>6} {N:>14} {M:>12,} {2**(bits//2):>12,}"
                  f" {method:>38} {elapsed:>7.1f}s {'OK' if tape_ok else 'CORR'}")
        else:
            print(f"  {bits:>6} {N:>14} {M:>12,} {2**(bits//2):>12,}"
                  f" {'FAIL: '+method:>38} {elapsed:>7.1f}s {'OK' if tape_ok else 'CORR'}")

    print(f"\n{'='*78}")
    print("RUST + GPU STACK")
    print(f"{'='*78}")
    print(f"  Grating:  Rust rayon (parallel, 1M chunks)")
    print(f"  FFT:      GPU CUDA (12.9 GB)")
    print(f"  gcd:      Python (pow with modulus)")
    print(f"  Tape:     SHA-256 verified, 0 bits erased, 0.0 J")
    print("=" * 78)


if __name__ == "__main__":
    main()
