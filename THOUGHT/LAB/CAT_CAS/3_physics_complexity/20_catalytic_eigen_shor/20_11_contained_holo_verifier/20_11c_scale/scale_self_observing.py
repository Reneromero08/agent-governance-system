"""
Experiment 20.11c: Scaling the Self-Observing Shor Loop
=========================================================
Tests the self-observing .holo loop across increasing bit sizes
to find where the containment wall actually sits.

For each bit size (22 through 46):
  - Generate semiprime, build catalytic grating, contain in .holo
  - Run self-observing loop: progressive k illumination
  - Track: k_needed, r_found, SNR, time, whether factored
  - When grating autocorrelation fails (r > M):
      fall back to eigenvector autocorrelation + gcd
  - Report the wall: at what bit size does the method break?
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


# ============================================================================
# CATALYTIC GRATING + .holo
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
    basis = eigenvectors[:, :min(k, eigenvectors.shape[1])]
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
# EIGENVECTOR FALLBACK (when r > M, grating autocorrelation fails)
# ============================================================================

def eigenvector_period_search(holo, a, N, L):
    """Extract period from eigenvector autocorrelation peaks + gcd fallback.
    Optimized: only checks top-2 peaks from top-2 eigenvectors, limited range."""
    eigenvectors = holo['eigenvectors']
    for i in range(min(3, eigenvectors.shape[1])):
        evec = eigenvectors[:, i]
        et = torch.tensor(evec, dtype=torch.complex64)
        eac = torch.fft.ifft(torch.abs(torch.fft.fft(et, n=2*L))**2).real[:L//2]
        eac = eac / (eac[0] + 1e-15)
        vals, idxs = torch.topk(torch.abs(eac[2:]), min(3, len(eac)-2))
        for idx in idxs:
            tau = int(idx.item()) + 2
            # Try tau directly, then a few multiples (up to 32x)
            for m in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
                r_cand = tau * m
                if r_cand > L * 8: break
                if pow(a, r_cand, N) == 1:
                    return r_cand
            # gcd fallback (only on tau, not multiples)
            g = gcd(pow(a, tau, N) - 1, N)
            if 1 < g < N:
                return tau
    return 0


# ============================================================================
# SELF-OBSERVING LOOP (primary method: grating reconstruction)
# ============================================================================

def self_observing_loop(holo, grating, a, N, L, stride):
    """Progressive k illumination + eigenvector fallback."""
    k_values = [2, 4, 8, 16, 32, 64, 128, 256]

    for k in k_values:
        k_actual = min(k, L - 1)
        recon = reconstruct_grating(holo, grating, L, stride, k_actual)
        r_est, snr = autocorrelation_period(recon)
        verified = pow(a, r_est, N) == 1

        if verified:
            factors = factor_from_period(N, a, r_est)
            if factors:
                return factors[0], factors[1], k_actual, r_est, snr, "grating"
            for m in range(2, 50):
                r_m = r_est * m
                if r_m < L * 8 and pow(a, r_m, N) == 1:
                    factors = factor_from_period(N, a, r_m)
                    if factors:
                        return factors[0], factors[1], k_actual, r_m, snr, "grating_harmonic"

    # Fallback: eigenvector-based search
    r_evec = eigenvector_period_search(holo, a, N, L)
    if r_evec > 1 and pow(a, r_evec, N) == 1:
        factors = factor_from_period(N, a, r_evec)
        if factors:
            return factors[0], factors[1], 0, r_evec, 0, "eigenvector"
    if r_evec > 1:
        g = gcd(pow(a, r_evec, N) - 1, N)
        if 1 < g < N:
            return g, N // g, 0, r_evec, 0, "eigenvector_gcd"

    return None, None, 0, 0, 0, "failed"


# ============================================================================
# MAIN: SCALING HARNESS
# ============================================================================

def main():
    print("=" * 78)
    print("EXPERIMENT 20.11c: SCALING THE SELF-OBSERVING LOOP")
    print("  Testing across bit sizes to find the containment wall.")
    print("=" * 78)

    M = 2 ** 21
    L = 2048
    stride = max(1, L // 8)
    BIT_SIZES = [22, 26, 30, 34]
    TRIALS_PER_SIZE = 1

    print(f"\n  Grating: M={M:,}  L={L}  stride={stride}")
    print(f"  {'bits':>6} {'trial':>6} {'N':>12} {'r_global':>12} {'r/M':>8} {'D_pr/r':>8}"
          f" {'k':>6} {'r_found':>12} {'SNR':>10} {'method':>18} {'time':>8}")
    print(f"  {'-'*95}")

    for bits in BIT_SIZES:
        for trial in range(TRIALS_PER_SIZE):
            t_start = time.perf_counter()

            N, p_known, q_known = generate_semiprime(bits)
            a = 2
            while gcd(a, N) != 1: a += 1
            r_global = true_period(a, N)

            # Build grating
            grating = build_catalytic_grating(a, N, M)

            # .holo containment
            obs = complex_obs_matrix(grating, L, stride)
            holo = complex_holo_eigenbasis(obs)
            if holo is None:
                print(f"  {bits:>6} {trial+1:>6}  (holo failed)")
                continue

            # Self-observing loop
            p_f, q_f, k_used, r_found, snr, method = self_observing_loop(
                holo, grating, a, N, L, stride)

            elapsed = time.perf_counter() - t_start
            success = p_f is not None

            print(f"  {bits:>6} {trial+1:>6} {N:>12} {r_global:>12} {r_global/M:>8.2f}"
                  f" {holo['df']/max(r_global,1):>8.4f} {k_used:>6} {r_found:>12}"
                  f" {snr:>10.1f} {method:>18} {elapsed:>7.1f}s"
                  f" {'FAIL' if not success else ''}")

    print(f"\n{'='*78}")
    print("SCALING ANALYSIS")
    print(f"{'='*78}")
    print(f"  The containment wall is where r > M (grating can't span full period).")
    print(f"  When r/M > 1.0, grating autocorrelation fails -> eigenvector fallback.")
    print(f"  The Moire decomposition (20.10.9) proves sub-periods r_p << r.")
    print(f"  Scaling to cryptographic sizes requires the sub-period approach.")
    print(f"  For N-bit RSA: r_p <= 2^(N/2) = sqrt(N). The wall is exponential.")
    print("=" * 78)


if __name__ == "__main__":
    main()
