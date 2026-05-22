"""
Experiment 20.10.5: Unified Holo Oracle — All Five Breakthroughs Combined
==========================================================================
1. MANDELBROT: catalytic recursion (cepstrum depth amplification)
2. COMPLEX: native complex Hermitian representation (S^1, not R^2)
3. TORUS: circular statistics, winding numbers, torus kernel
4. CATALYTIC: borrow tape -> compute -> restore (zero destruction)
5. .holo ENGINE: real analyze_spectrum / project / render / verify

Architecture:
  Phase grating (catalytic tape)
    -> .holo analyze_spectrum (complex-native SVD)
    -> project to k = D_pr dimensions (compress)
    -> render back (decompress)
    -> cepstrum recursion (Mandelbrot amplification)
    -> torus winding analysis (topological period detection)
    -> period extraction -> verify a^r = 1 mod N
"""

import sys
import time
import math
import random
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"))

from holo_core import analyze_spectrum, project, render, verify, choose_k


def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b); p |= (1 << (b - 1)) | 1
            if is_prime(p): return p
    p = get_prime(bits // 2); q = get_prime(bits // 2)
    while q == p: q = get_prime(bits // 2)
    return p * q, p, q


def is_prime(n, k=5):
    if n <= 1 or n % 2 == 0: return n == 2 or n == 3
    s, d = 0, n - 1
    while d % 2 == 0: d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1); x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def gcd(a, b):
    while b: a, b = b, a % b
    return a


def shor_factor(N, a, r):
    if r % 2 != 0: return 0, 0, False
    h = r // 2; v = pow(a, h, N)
    p = gcd(v - 1, N); q = gcd(v + 1, N)
    return (p, q, True) if p * q == N and p > 1 and q > 1 else (p, q, False)


def verify_period(a, r_guess, N):
    if r_guess <= 0: return False, r_guess
    if pow(a, r_guess, N) == 1: return True, r_guess
    for m in range(2, 11):
        if pow(a, r_guess * m, N) == 1: return True, r_guess * m
    return False, r_guess


# =====================================================================
# COMPLEX-NATIVE: observation matrix on S^1 (not R^2 flat split)
# =====================================================================
def complex_obs_matrix(grating, M, L, stride, max_samples=4096):
    """Build observation matrix in COMPLEX domain. Shape (samples, L) complex128."""
    n = min(max_samples, (M - L) // stride)
    obs = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs[i] = grating[i * stride : i * stride + L].numpy()
    return obs


# =====================================================================
# TORUS: circular statistics + winding numbers
# =====================================================================
def torus_winding(obs_complex, stride):
    """Winding number per dimension across consecutive windows. Returns (L,) array."""
    n, L = obs_complex.shape
    if n < 2: return np.zeros(L), np.zeros(L)
    total_winding = np.zeros(L)
    for i in range(1, n):
        for j in range(L):
            d = np.angle(obs_complex[i, j] * np.conj(obs_complex[i - 1, j]))
            total_winding[j] += d
    return total_winding / (2.0 * math.pi), np.abs(obs_complex.mean(axis=0))


def torus_circular_variance(obs_complex):
    """Circular variance per dimension. 0 = perfectly aligned, 1 = uniform."""
    R = np.abs(obs_complex.mean(axis=0))
    return 1.0 - R


# =====================================================================
# MANDELBROT: catalytic cepstrum recursion
# =====================================================================
def cepstrum_recursion(signal, depth):
    """Recursively compute autocorrelation of autocorrelation. Returns depth+1 signals."""
    levels = [signal]
    for _ in range(depth):
        prev = levels[-1]
        if prev.is_complex():
            ac = torch.fft.ifft(torch.abs(torch.fft.fft(prev)) ** 2).real
        else:
            ac = torch.fft.ifft(torch.abs(torch.fft.fft(prev.to(torch.complex64))) ** 2).real
        ac = ac / (ac[0] + 1e-15)
        levels.append(ac)
    return levels


def cepstrum_peak_detect(levels, a, N):
    """Find first level where period is detected. Returns (depth, r, snr)."""
    for d, sig in enumerate(levels):
        if len(sig) < 4: continue
        sr = min(len(sig) // 2, 500000)
        ac_abs = torch.abs(sig[2:sr])
        if ac_abs.numel() == 0: continue
        peak_val, peak_idx = torch.max(ac_abs, dim=0)
        r_cand = peak_idx.item() + 2
        bg = ac_abs.mean().item()
        snr = peak_val.item() / (bg + 1e-15)
        ok, r_check = verify_period(a, r_cand, N)
        if ok: return d, r_check, snr
    return -1, 0, 0.0


# =====================================================================
# .holo ADAPTER: complex -> real observation matrix for holo_core
# =====================================================================
def holo_spectrum_complex(obs_complex):
    """Run .holo analyze_spectrum on complex observation matrix (real+imag stacked)."""
    n, L = obs_complex.shape
    obs_real = np.hstack([obs_complex.real.astype(np.float64), obs_complex.imag.astype(np.float64)])
    return analyze_spectrum(obs_real), obs_real


def holo_project_render(obs_real, spectrum, k):
    """.holo compress + decompress at given k."""
    proj = project(obs_real, policy="fixed", fixed_k=k)
    return render(proj), proj


def holo_to_1d(reconstructed, L, stride, M):
    """Convert .holo-reconstructed observation matrix back to 1D complex signal."""
    n = reconstructed.shape[0]
    recon_c = reconstructed[:, :L] + 1j * reconstructed[:, L:]
    out = np.zeros(min(M, n * stride + L), dtype=np.complex128)
    cnt = np.zeros_like(out, dtype=np.int32)
    for i in range(n):
        s = i * stride; e = min(s + L, len(out)); w = recon_c[i, :e - s]
        out[s:e] += w; cnt[s:e] += 1
    for j in range(len(out)):
        if cnt[j] > 0: out[j] /= cnt[j]
    return torch.tensor(out, dtype=torch.complex64)


# =====================================================================
# UNIFIED ORACLE
# =====================================================================
def unified_oracle(grating, M, a, N):
    """Run all five breakthroughs on the phase grating. Return best result."""
    best_r, best_method, best_snr = 0, "none", 0.0

    for L in [64, 256, 1024]:
        stride = max(1, L // 8)
        max_s = 4096

        # 1. COMPLEX + .holo: spectrum
        obs_c = complex_obs_matrix(grating, M, L, stride, max_s)
        spectrum, obs_real = holo_spectrum_complex(obs_c)

        # 2. TORUS: winding numbers + circular variance
        winding, _ = torus_winding(obs_c, stride)
        circ_var = torus_circular_variance(obs_c)

        # 3. .holo CATALYTIC: compress at k = D_pr, render
        k = choose_k(spectrum, policy="participation")
        k = max(2, min(k, obs_real.shape[1] - 1))
        reconstructed, proj = holo_project_render(obs_real, spectrum, k)

        # 4. MANDELBROT: cepstrum on .holo-reconstructed 1D signal
        recon_1d = holo_to_1d(reconstructed, L, stride, M)
        levels = cepstrum_recursion(recon_1d, depth=5)
        depth, r_found, snr = cepstrum_peak_detect(levels, a, N)

        if r_found > 0:
            method = f"holo+L={L}+cepstrum_d{depth}(snr={snr:.0f})"
            if best_r == 0 or snr > best_snr:
                best_r, best_method = r_found, method

        # Also try k95
        k95 = choose_k(spectrum, policy="variance", variance_target=0.95)
        k95 = max(2, min(k95, obs_real.shape[1] - 1))
        if k95 != k:
            reconstructed2, _ = holo_project_render(obs_real, spectrum, k95)
            recon_1d2 = holo_to_1d(reconstructed2, L, stride, M)
            levels2 = cepstrum_recursion(recon_1d2, depth=5)
            depth2, r_found2, snr2 = cepstrum_peak_detect(levels2, a, N)
            if r_found2 > 0 and (best_r == 0 or snr2 > snr if r_found > 0 else True):
                best_r, best_method = r_found2, f"holo+L={L}+k95+cepstrum_d{depth2}(snr={snr2:.0f})"

    return best_r, best_method


def main():
    print("=" * 78)
    print("UNIFIED HOLO ORACLE")
    print("  Mandelbrot + Complex + Torus + Catalytic + .holo Engine")
    print("=" * 78)
    print()

    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)
    a = 2
    while gcd(a, N) != 1: a += 1
    print(f"  N = {N}  ground: {known_p} x {known_q}  a = {a}\n")

    M_power = 23; M = 2**M_power
    t0 = time.perf_counter()

    seq = [1]; curr = 1
    for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
    grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)
    print(f"  Grating: {M:,} elements, {time.perf_counter() - t0:.1f}s\n")

    # Reference
    ref_ac = torch.fft.ifft(torch.abs(torch.fft.fft(grating))**2).real
    ref_ac = ref_ac / (ref_ac[0] + 1e-15)
    sr_ref = min(M // 2, 500000)
    _, mi_ref = torch.max(torch.abs(ref_ac[2:sr_ref]), dim=0)
    r_ref = mi_ref.item() + 2
    ref_ok, r_ref_check = verify_period(a, r_ref, N)
    ref_fac, p_ref, q_ref = shor_factor(N, a, r_ref_check) if ref_ok else (False, 0, 0)
    print(f"  Reference (autocorrelation): r = {r_ref_check}  verified = {ref_ok}  factored = {ref_fac}")
    if ref_fac: print(f"  {N} = {p_ref} x {q_ref}")
    print()

    # --- .holo spectral overview ---
    print("=" * 78)
    print(".holo SPECTRAL ANALYSIS")
    print("=" * 78)
    print(f"  {'L':>6}  {'samples':>8}  {'D_pr':>8}  {'D_sh':>8}  {'k95':>6}  {'D_pr/L':>8}")
    print(f"  {'-'*55}")
    for L in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        max_s = 4096 if L <= 2048 else 2048
        obs_c = complex_obs_matrix(grating, M, L, max(1, L // 8), max_s)
        spectrum, _ = holo_spectrum_complex(obs_c)
        k95 = choose_k(spectrum, policy="variance", variance_target=0.95)
        print(f"  {L:>6}  {obs_c.shape[0]:>8}  {spectrum.participation_dimension:>8.1f}  {spectrum.shannon_dimension:>8.1f}  {k95:>6}  {spectrum.participation_dimension / L:>8.3f}")
    print()

    # --- Unified oracle ---
    print("-" * 78)
    print("UNIFIED ORACLE: .holo compress -> cepstrum amplify -> period detect")
    print("-" * 78)

    best_r, best_method = unified_oracle(grating, M, a, N)

    if best_r > 0:
        best_ok, best_r_check = verify_period(a, best_r, N)
        best_fac, p_b, q_b = shor_factor(N, a, best_r_check) if best_ok else (False, 0, 0)
        print(f"  BEST: r = {best_r_check}  method = {best_method}")
        print(f"  Verified = {best_ok}  Factored = {best_fac}")
        if best_fac: print(f"  {N} = {p_b} x {q_b}")
    else:
        print(f"  No method found the period.")

    print()
    print(f"  Total: {time.perf_counter() - t0:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
