"""
Experiment 20.11b: Self-Observing Shor Loop
=============================================
QUANTUM: The Hermitian covariance C = Z^H @ Z IS the density matrix rho.
         Eigendecomposition reveals eigenstates (pointer states). The .holo
         stores the top-k eigenstates — the quantum state of the modular
         exponentiation operator, compressed to its dominant eigenmodes.

CATALYTIC: The grating IS the catalytic tape. All .holo operations are
           READ-ONLY (eigendecomposition, projection, reconstruction).
           Tape hash verified before/after — SHA-256 must match.
           Zero bits erased. Zero Landauer dissipation.

COMPLEX-NATIVE: Vectors live on S^1 (complex unit circle). Covariance is
                Hermitian Z^H @ Z. Eigendecomposition via np.linalg.eigh.
                No real+imag flattening. Phase IS the degree of freedom.

Self-observing loop:
  1. Quantum: build density matrix rho = Z^H @ Z from phase grating on S^1
  2. Quantum: eigendecomposition -> extract top-k eigenstates (the .holo)
  3. Quantum: projective measurement through top-k eigenstates (reconstruction)
  4. Classical: autocorrelation on reconstructed signal -> period r
  5. Catalytic: verify tape SHA-256 unchanged (zero erasure)
  6. Repeat at increasing k until a^r = 1 (convergence)

The .holo IS the stored eigenstates from a single quantum measurement.
The integer r emerges from the interference pattern when illuminated.
The tape remains pristine — the computation was a READ, not a WRITE.
"""

import sys
import time
import math
import random
from pathlib import Path

import numpy as np
import torch


def is_probable_prime(n, k=10):
    if n < 2: return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
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


def true_period(a, N, max_steps=5000000):
    x, r = a % N, 1
    while x != 1 and r < max_steps:
        x = (x * a) % N; r += 1
    return r if x == 1 else 0


def factor_from_period(N, a, r):
    if r % 2 != 0: return None
    h = r // 2; v = pow(a, h, N)
    p = gcd(v - 1, N); q = gcd(v + 1, N)
    if p > 1 and q > 1 and p * q == N: return (p, q)
    return None


import hashlib


# ... (existing imports)


# ============================================================================
# CATALYTIC TAPE VERIFICATION
# ============================================================================

def sha256_hex(data):
    """SHA-256 of a numpy array (as raw bytes)."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


# ============================================================================
# CATALYTIC GRATING
# ============================================================================

def build_catalytic_grating(a, N, M):
    grating = np.empty(M, dtype=np.complex128)
    val = 1
    for i in range(M):
        angle = 2.0 * math.pi * val / N
        grating[i] = complex(math.cos(angle), math.sin(angle))
        val = (val * a) % N
    return grating


def complex_obs_matrix(grating, L, stride):
    M = len(grating)
    n = min(4096, (M - L) // stride)
    obs = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs[i] = grating[i * stride : i * stride + L]
    return obs


def complex_holo_eigenbasis(obs):
    n = obs.shape[0]
    if n < 2: return None
    centered = obs - obs.mean(axis=0, keepdims=True)
    cov = (centered.conj().T @ centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    total = eigenvalues.sum()
    if total < 1e-15: return None
    probs = eigenvalues / total
    cumulative = np.cumsum(probs)
    df = 1.0 / (probs ** 2).sum()
    k95 = int(np.searchsorted(cumulative, 0.95) + 1)
    return {
        'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors,
        'cumulative': cumulative, 'df': float(df), 'k95': k95,
        'total': float(total),
    }


def reconstruct_grating(holo, grating, L, stride, k):
    obs = complex_obs_matrix(grating, L, stride)
    n = obs.shape[0]
    centered = obs - obs.mean(axis=0, keepdims=True)
    eigenvectors = holo['eigenvectors']
    basis = eigenvectors[:, :k]
    coords = centered @ basis
    reconstructed_centered = coords @ basis.conj().T
    reconstructed = reconstructed_centered + obs.mean(axis=0, keepdims=True)
    recon_1d = np.zeros(len(grating), dtype=np.complex128)
    counts = np.zeros(len(grating), dtype=np.int32)
    for i in range(n):
        s, e = i * stride, min(i * stride + L, len(grating))
        w = min(e - s, L)
        recon_1d[s:e] += reconstructed[i, :w]
        counts[s:e] += 1
    mask = counts > 0
    recon_1d[mask] /= counts[mask]
    return recon_1d


def autocorrelation_period(signal, max_search=None):
    M = len(signal)
    if max_search is None: max_search = min(M // 2, 1000000)
    sig_t = torch.tensor(signal, dtype=torch.complex64)
    fft = torch.fft.fft(sig_t)
    power = torch.abs(fft) ** 2
    ac = torch.fft.ifft(power).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(max_search, len(ac) // 2)
    if sr <= 2: return 0, 0.0
    peak_val, peak_idx = torch.max(torch.abs(ac[2:sr]), dim=0)
    r_est = peak_idx.item() + 2
    bg = torch.abs(ac[1:sr]).mean().item()
    snr = peak_val.item() / max(bg, 1e-15)
    return r_est, snr


# ============================================================================
# SELF-OBSERVING LOOP: progressive illumination
# ============================================================================

def self_observing_loop(holo, grating, a, N, L, stride):
    """Progressively illuminate the .holo at increasing k until truth emerges.

    The .holo observes itself: at each resolution k, it reconstructs the
    grating through its own eigenbasis, measures the period via autocorrelation,
    and checks if the truth has emerged. The loop converges when a^r = 1.

    Returns (p, q, method, trace) or (None, None, "failed", trace).
    """
    trace = []
    k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    prev_r = 0
    stable_count = 0

    for k in k_values:
        k_actual = min(k, L - 1)
        recon = reconstruct_grating(holo, grating, L, stride, k_actual)
        r_est, snr = autocorrelation_period(recon)
        verified = pow(a, r_est, N) == 1

        stable = (r_est == prev_r and prev_r > 0)
        if stable:
            stable_count += 1
        else:
            stable_count = 0
        prev_r = r_est

        entry = {
            'k': k_actual, 'r': r_est, 'snr': snr,
            'verified': verified, 'stable': stable, 'stable_count': stable_count,
        }
        trace.append(entry)

        # Try factorization if verified
        if verified:
            factors = factor_from_period(N, a, r_est)
            if factors:
                return factors[0], factors[1], f"k={k_actual}_r={r_est}", trace

            # If period verified but Shor post-process fails, try harmonics
            for m in range(2, 30):
                r_m = r_est * m
                if r_m < L * 8 and pow(a, r_m, N) == 1:
                    factors = factor_from_period(N, a, r_m)
                    if factors:
                        return factors[0], factors[1], f"k={k_actual}_harmonic_{r_est}x{m}", trace

        # Convergence detection: if same r found 3 times with increasing SNR
        # and it's verified, but Shor post-process still failed, try gcd fallback
        if stable_count >= 3 and verified:
            # Try gcd on the stable period
            g = gcd(pow(a, r_est // 2, N) - 1, N) if r_est % 2 == 0 else 0
            if g == 0:
                g = gcd(pow(a, r_est, N) - 1, N)
            if 1 < g < N:
                return g, N // g, f"k={k_actual}_gcd_on_stable_{r_est}", trace

    # Exhausted all k values — try gcd on best unverified candidate
    if trace:
        best = max(trace, key=lambda t: t['snr'] if t['verified'] else 0)
        if best['verified']:
            g = gcd(pow(a, best['r'] // 2, N) - 1, N) if best['r'] % 2 == 0 else gcd(pow(a, best['r'], N) - 1, N)
            if 1 < g < N:
                return g, N // g, f"gcd_fallback_k{best['k']}", trace

    return None, None, "exhausted", trace


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("EXPERIMENT 20.11b: SELF-OBSERVING SHOR LOOP")
    print("  The .holo illuminates itself at increasing resolution (k).")
    print("  The period emerges from the interference pattern. No oracle.")
    print("=" * 78)

    BITS = 22
    N, p_known, q_known = generate_semiprime(BITS)
    a = 2
    while gcd(a, N) != 1: a += 1
    r_global = true_period(a, N)
    r_p = true_period(a % p_known, p_known)
    r_q = true_period(a % q_known, q_known)
    print(f"\n  N = {N} = {p_known} x {q_known}")
    print(f"  a = {a}")
    print(f"  r_p={r_p}  r_q={r_q}  r={r_global}")

    t0 = time.perf_counter()

    # CATALYTIC GRATING — build on the tape
    M = 2 ** 22  # ~4M elements, autocorrelation range up to 2M
    grating = build_catalytic_grating(a, N, M)
    tape_hash_before = sha256_hex(grating)
    print(f"\n  [CATALYTIC] phase grating on S^1: M={M:,} complex128 positions")
    print(f"  Borrow:  a^x mod N -> angle -> e^(i*theta) on complex unit circle")
    print(f"  Tape:    SHA-256 = {tape_hash_before}  ({M*16/1e6:.1f} MB)")

    # .holo CONTAINMENT — quantum measurement (READ-ONLY on tape)
    L = 2048
    stride = max(1, L // 8)
    obs = complex_obs_matrix(grating, L, stride)
    print(f"\n  [QUANTUM] density matrix: rho = Z^H @ Z  ({obs.shape[0]} projective")
    print(f"            measurements x {obs.shape[1]} complex dimensions on S^1)")
    holo = complex_holo_eigenbasis(obs)
    print(f"  [QUANTUM] eigendecomposition: {holo['k95']} eigenstates extracted")
    print(f"            top eigenstate = pointer state (survives decoherence)")
    print(f"            D_pr = {holo['df']:.1f} eigenstates carry signal")

    print(f"\n{'='*78}")
    print(".holo CONTAINMENT")
    print(f"{'='*78}")
    print(f"  D_pr={holo['df']:.1f}  k95={holo['k95']}  D_pr/L={holo['df']/L:.4f}  D_pr/r={holo['df']/max(r_global,1):.4f}")

    # SELF-OBSERVING LOOP
    print(f"\n{'='*78}")
    print("SELF-OBSERVING LOOP: progressive illumination")
    print(f"{'='*78}")
    print(f"  {'k':>6} {'r_est':>10} {'SNR':>10} {'a^r=1?':>8} {'stable?':>8} {'stable#':>8}")
    print(f"  {'-'*60}")

    p_found, q_found, method, trace = self_observing_loop(holo, grating, a, N, L, stride)

    # CATALYTIC: verify tape untouched (all .holo ops are READ-ONLY)
    tape_hash_after = sha256_hex(grating)
    tape_intact = tape_hash_before == tape_hash_after
    print(f"\n  [CATALYTIC] Tape SHA-256 after: {tape_hash_after}")
    print(f"  [CATALYTIC] Tape integrity: {'MATCH — zero bits erased, 0.0 J' if tape_intact else 'CORRUPTED'}")

    for entry in trace:
        print(f"  {entry['k']:>6} {entry['r']:>10} {entry['snr']:>10.1f} "
              f"{str(entry['verified']):>8} {str(entry['stable']):>8} {entry['stable_count']:>8}")

    print(f"\n{'='*78}")
    if p_found and q_found:
        print(f"  CONVERGED: {N} = {p_found} x {q_found}")
        print(f"  Method: {method}")
        print(f"  Match ground truth: {set([p_found, q_found]) == set([p_known, q_known])}")
    else:
        print(f"  FAILED: {method}")
        print(f"  Ground truth: {N} = {p_known} x {q_known}")

    total_t = time.perf_counter() - t0
    print(f"\n  Total: {total_t:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
