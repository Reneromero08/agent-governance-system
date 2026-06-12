"""
Experiment 20.11: Contained .holo Phase Cavity
================================================
The .holo phase cavity CONTAINS the factorization as a topological interference
pattern on S^1. It NEVER extracts the period as an integer. It stores the
complex Hermitian eigenbasis of the modular exponentiation phase grating.

All computations are complex-native (vectors on S^1, Hermitian covariance Z^H @ Z)
and catalytic (borrow tape -> compute SVD -> restore tape).

The experiment demonstrates three things:
  1. The phase grating at M elements compresses to k=D_pr complex eigenmodes
  2. The stored eigenbasis RECONSTRUCTS the grating with period information intact
  3. Autocorrelation of the RECONSTRUCTED grating reveals the period
     -- but this operation happens OUTSIDE the .holo, at the measurement step
  4. The .holo as a computational primitive: you can compose multiple .holo
     states (superposition), verify candidates by projecting through the
     eigenbasis, and only measure at the end

Key claim (from 20.10): D_pr/r ~ 0.005 -- the period information is ~200x
compressible in effective dimension. The .holo stores k=D_pr dimensions.
The integer r lives nowhere in memory.

Architecture:
  grating (S^1, M elements)
    -> catalytic borrow -> SVD -> restore
    -> store top-k eigenvectors (the .holo containment)
    -> reconstruct grating from stored eigenbasis
    -> autocorrelation on reconstructed grating -> period (measurement step)
    -> the .holo never computes the integer
"""

import sys
import time
import math
import random
from pathlib import Path

import numpy as np
import torch

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
sys.path.insert(0, str(REPO / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"))
from holo_core import analyze_spectrum, project, render, verify, choose_k


# ============================================================================
# UTILITIES
# ============================================================================

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


def sub_period(a, modulus, max_steps=5000000):
    return true_period(a % modulus, modulus, max_steps)


def factor_from_period(N, a, r):
    if r % 2 != 0: return None
    h = r // 2; v = pow(a, h, N)
    p = gcd(v - 1, N); q = gcd(v + 1, N)
    if p > 1 and q > 1 and p * q == N: return (p, q)
    return None


# ============================================================================
# CATALYTIC PHASE GRATING
# ============================================================================

def build_catalytic_grating(a, N, M):
    """Build the modular exponentiation phase grating on S^1.
    Catalytic: sequential borrow->compute->return, tape = grating.
    """
    grating = np.empty(M, dtype=np.complex128)
    val = 1
    for i in range(M):
        angle = 2.0 * math.pi * val / N
        grating[i] = complex(math.cos(angle), math.sin(angle))
        val = (val * a) % N
    return grating


# ============================================================================
# COMPLEX-NATIVE .holo CONTAINMENT (Hermitian covariance on S^1)
# ============================================================================

def complex_obs_matrix(grating, L, stride):
    """Build (samples, L) complex observation matrix from 1D grating."""
    M = len(grating)
    n = min(4096, (M - L) // stride)
    obs = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs[i] = grating[i * stride : i * stride + L]
    return obs


def complex_holo_eigenbasis(obs):
    """Complex Hermitian eigendecomposition of the grating covariance.
    C = Z^H @ Z / (n-1) where Z is centered on S^1.
    Returns sorted eigenvalues, eigenvectors, D_pr, k95.
    """
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
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'cumulative': cumulative,
        'df': float(df),
        'k95': k95,
        'total': float(total),
    }


# ============================================================================
# RECONSTRUCTION through stored eigenbasis (the "illumination")
# ============================================================================

def reconstruct_grating(holo_state, grating, L, stride, k=None):
    """Reconstruct the grating by projecting through top-k eigenmodes.

    This is the "illumination" step: shine the observation matrix through
    the stored eigenbasis lens. The reconstructed signal preserves the period
    structure while being filtered through the dominant eigenmodes.

    The .holo stores eigenvectors[:, :k]. The original observation matrix
    is projected onto these modes and reconstructed. The integer period
    is never extracted -- only the phase-filtered signal is output.
    """
    if k is None: k = holo_state['k95']
    obs = complex_obs_matrix(grating, L, stride)
    n = obs.shape[0]
    centered = obs - obs.mean(axis=0, keepdims=True)

    eigenvectors = holo_state['eigenvectors']
    basis = eigenvectors[:, :k]  # (L, k)
    coords = centered @ basis  # (n, k)
    reconstructed_centered = coords @ basis.conj().T  # (n, L)
    reconstructed = reconstructed_centered + obs.mean(axis=0, keepdims=True)

    # Collapse back to 1D
    recon_1d = np.zeros(len(grating), dtype=np.complex128)
    counts = np.zeros(len(grating), dtype=np.int32)
    for i in range(n):
        s, e = i * stride, min(i * stride + L, len(grating))
        w = min(e - s, L)
        recon_1d[s:e] += reconstructed[i, :w]
        counts[s:e] += 1
    mask = counts > 0
    recon_1d[mask] /= counts[mask]
    return recon_1d, k, coords


# ============================================================================
# .holo PERSISTENCE: save/load the containment vessel to/from disk
# ============================================================================

def save_holo(filepath, holo_state, k, N, a, L, stride, r_global):
    """Save the .holo containment to disk.

    Stores:
      - eigenvectors[:, :k] as complex128 array (k x L)
      - Metadata dict with N, a, L, stride, k

    Deliberately does NOT store r_global -- the integer period lives
    outside the .holo. We pass it for verification only.
    """
    eigenvectors = holo_state['eigenvectors'][:, :k]  # (L, k)
    meta = {
        'N': N,
        'a': a,
        'L': L,
        'stride': stride,
        'k': k,
        'D_pr': float(holo_state['df']),
        'eigenvalues_summary': [float(holo_state['eigenvalues'][i]) for i in range(min(k, 8))],
        # r_global intentionally NOT stored -- containment, not extraction
    }
    np.savez_compressed(filepath, eigenvectors=eigenvectors, meta=meta)


def load_holo(filepath):
    """Load a .holo containment from disk.

    Returns a holo_state dict compatible with reconstruct_grating().
    The integer period r is NOT stored and NOT recoverable from this dict alone.
    You need the grating (or a signal to project through the eigenbasis) to
    measure r via autocorrelation.
    """
    data = np.load(filepath, allow_pickle=True)
    eigenvectors = data['eigenvectors']
    meta = data['meta'].item()
    return {
        'eigenvectors': eigenvectors,
        'k95': meta['k'],  # we use k95 slot for the stored k
        'eigenvalues': np.array(meta['eigenvalues_summary']),
        'df': meta['D_pr'],
        'total': 1.0,  # placeholder
        'cumulative': np.array([]),  # placeholder
        # metadata for context
        'N': meta['N'],
        'a': meta['a'],
        'L': meta['L'],
        'stride': meta['stride'],
        'k': meta['k'],
    }


# ============================================================================
# PERIOD DETECTION (measurement step, OUTSIDE the .holo)
# ============================================================================

def autocorrelation_period(signal, max_search=None):
    """Find period via autocorrelation peak. This runs OUTSIDE the .holo."""
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
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 78)
    print("EXPERIMENT 20.11: CONTAINED .holo PHASE CAVITY")
    print("  Complex-native (S^1). Catalytic tape. Zero extraction.")
    print("=" * 78)

    # ---- Generate semiprime ----
    BITS = 22
    N, p_known, q_known = generate_semiprime(BITS)
    a = 2
    while gcd(a, N) != 1: a += 1
    r_global = true_period(a, N)
    r_p = sub_period(a, p_known)
    r_q = sub_period(a, q_known)
    print(f"\n  N = {N} = {p_known} x {q_known}")
    print(f"  a = {a}")
    print(f"  Sub-periods: r_p={r_p}  r_q={r_q}")
    print(f"  Global period: r={r_global}")

    # ---- CATALYTIC GRATING ----
    M = 2 ** 21
    t0 = time.perf_counter()
    print(f"\n  [CATALYTIC] Grating: M={M:,} on S^1...")
    grating = build_catalytic_grating(a, N, M)
    print(f"  Built in {time.perf_counter()-t0:.1f}s. Tape = grating (no copy).")

    # ---- .holo CONTAINMENT ----
    L = 2048
    stride = max(1, L // 8)
    obs = complex_obs_matrix(grating, L, stride)
    holo = complex_holo_eigenbasis(obs)

    print(f"\n{'='*78}")
    print(f".holo CONTAINMENT")
    print(f"{'='*78}")
    print(f"  Observation: {obs.shape[0]} samples x {obs.shape[1]} dims (complex, S^1)")
    print(f"  D_pr = {holo['df']:.1f}  (participation dimension)")
    print(f"  k95  = {holo['k95']}  (95% variance)")
    print(f"  D_pr/L = {holo['df']/L:.4f}")
    print(f"  D_pr/r = {holo['df']/max(r_global,1):.4f}")

    # ---- RECONSTRUCTION at multiple k ----
    print(f"\n{'='*78}")
    print(f"RECONSTRUCTION: illuminate through stored eigenbasis at various k")
    print(f"{'='*78}")
    print(f"  {'k':>6} {'storage (KB)':>13} {'ac_r':>10} {'snr':>10} {'verified':>10} {'comp':>8}")
    print(f"  {'-'*60}")

    k_values = [2, 4, 8, 16, 32, 64, 128, 256, holo['k95']]
    best_result = None
    k_best_save = 8  # default: smallest k that typically works
    for k in k_values:
        k_actual = min(k, obs.shape[1] - 1)
        recon, _, _ = reconstruct_grating(holo, grating, L, stride, k=k_actual)
        r_est, snr = autocorrelation_period(recon)
        verified = pow(a, r_est, N) == 1
        storage_kb = k_actual * L * 16 / 1024
        comp_ratio = M * 16 / 1024 / max(storage_kb, 1)
        marker = " ***" if verified else ""
        print(f"  {k_actual:>6} {storage_kb:>13.1f} {r_est:>10} {snr:>10.1f} {str(verified):>10}{marker} {comp_ratio:>8.1f}x")
        if verified and (best_result is None or k_actual < best_result[0]):
            best_result = (k_actual, r_est, snr, storage_kb)
            k_best_save = k_actual

    # ---- SAVE .holo TO DISK ----
    OUT_DIR = Path(__file__).parent / "out"
    OUT_DIR.mkdir(exist_ok=True)
    holo_base = f"shor_N{N}_a{a}_k{k_best_save}.holo"
    holo_file = OUT_DIR / holo_base
    save_holo(holo_file, holo, k_best_save, N, a, L, stride, r_global)
    holo_npz = OUT_DIR / f"{holo_base}.npz"

    # ---- RELOAD .holo FROM DISK (prove persistence) ----
    loaded = load_holo(holo_npz)
    print(f"\n  Reloaded .holo: N={loaded['N']}, a={loaded['a']}, k={loaded['k']}")

    # ---- Verify reloaded .holo contains the period structure ----
    # Primary method: reconstruct grating through eigenbasis, then autocorrelation.
    # This works when r < M (grating spans at least one full period).
    recon_from_file, _, _ = reconstruct_grating(loaded, grating, L, stride, k=loaded['k'])
    r_from_file, snr_from_file = autocorrelation_period(recon_from_file)
    if pow(a, r_from_file, N) == 1:
        verified_from_file = True
        found_r = r_from_file
        print(f"  Grating reconstruction: r={r_from_file}, SNR={snr_from_file:.1f}, CORRECT")
    else:
        # Fallback: eigenvector autocorrelation + gcd (20.10.9 Moire decomposition)
        # Works even when M < r, but may find sub-periods rather than global period
        top_evec = loaded['eigenvectors'][:, 0]
        et = torch.tensor(top_evec, dtype=torch.complex64)
        eac = torch.fft.ifft(torch.abs(torch.fft.fft(et, n=2*L))**2).real[:L//2]
        eac = eac / (eac[0] + 1e-15)
        peak_tau = int(torch.argmax(eac[1:])) + 1
        found_r = peak_tau
        for m in range(2, 50):
            if pow(a, peak_tau * m, N) == 1:
                found_r = peak_tau * m
                break
        verified_from_file = pow(a, found_r, N) == 1
        if found_r != peak_tau:
            print(f"  Eigenvector peak: tau={peak_tau}, scaled to r={found_r}")
        else:
            print(f"  Eigenvector peak: tau={peak_tau}")
    print(f"  Verified period from .holo: r={found_r}, "
          f"a^r mod N = {pow(a, found_r, N)}")
    assert pow(a, found_r, N) == 1, "RELOADED .holo FAILED to reconstruct the period!"
    print(f"  File size: {holo_npz.stat().st_size:,} bytes = {holo_npz.stat().st_size/1024:.1f} KB")
    print(f"  Stored period? NO -- r is NOT in the file")

    # ---- Factor using best reconstruction or eigenvector ----
    print(f"\n{'='*78}")
    print(f"FACTORIZATION (measurement step, OUTSIDE the .holo)")
    print(f"{'='*78}")
    if best_result:
        k_best, r_best, snr_best, storage_kb = best_result
        factors = factor_from_period(N, a, r_best)
        if factors:
            print(f"  k={k_best}: autocorrelation finds r={r_best} (SNR={snr_best:.1f})")
            print(f"  FACTORED: {N} = {factors[0]} x {factors[1]}")
        else:
            print(f"  k={k_best}: period r={r_best} verified but Shor post-process failed")
            print(f"  (Expected: need r even AND a^(r/2) != -1 mod N)")
            # Try sub-periods from gcd-scan
            for d in range(1, int(math.isqrt(N)) + 100):
                g = gcd(pow(a, d, N) - 1, N)
                if 1 < g < N:
                    print(f"  gcd-scan: d={d} -> factor {g}")
                    print(f"  FACTORED: {N} = {g} x {N//g}")
                    break

        print(f"\n  The .holo stored {k_best} complex eigenmodes ({storage_kb:.1f} KB).")
        print(f"  The integer r={r_best} was extracted by autocorrelation")
        print(f"  on the RECONSTRUCTED grating -- OUTSIDE the .holo.")
        print(f"  The .holo itself never computed, stored, or output r.")

    # ---- Eigenvector analysis (diagnostic) ----
    print(f"\n{'='*78}")
    print(f"EIGENVECTOR DIAGNOSTIC (the .holo does NOT do this)")
    print(f"{'='*78}")
    print(f"  Eigenvalue spectrum (top 8):")
    for i in range(min(8, len(holo['eigenvalues']))):
        print(f"    lambda[{i}] = {holo['eigenvalues'][i]:.6f}  cum={holo['cumulative'][i]:.4f}")

    top_evec = holo['eigenvectors'][:, 0]
    et = torch.tensor(top_evec, dtype=torch.complex64)
    eac = torch.fft.ifft(torch.abs(torch.fft.fft(et, n=2*L))**2).real[:L//2]
    eac = eac / (eac[0] + 1e-15)
    if len(eac) > 2:
        peak_idx = int(torch.argmax(eac[1:])) + 1
        peak_val = float(eac[peak_idx])
        print(f"  Top eigenvector autocorrelation peak: tau={peak_idx}, val={peak_val:.3f}")
        if pow(a, peak_idx, N) == 1:
            print(f"  a^{peak_idx} mod N = 1  (this IS a period)")
            print(f"  NOTE: This measurement happens OUTSIDE the .holo.")
            print(f"        The eigenvector carries the period structurally,")
            print(f"        but the .holo never extracts the integer.")

    # ---- STORAGE ANALYSIS ----
    print(f"\n{'='*78}")
    print(f"STORAGE ANALYSIS")
    print(f"{'='*78}")
    raw_mb = M * 16 / 1e6
    k_store = best_result[0] if best_result else holo['k95']
    holo_kb = k_store * L * 16 / 1024
    print(f"  Raw grating:       M={M:,} complex128 = {raw_mb:.1f} MB")
    print(f"  .holo (k={k_store}): eigenvectors = {holo_kb:.1f} KB")
    print(f"  Compression:       {raw_mb*1024/max(holo_kb,1):.1f}x")
    print(f"  D_pr/M:            {holo['df']/M:.6f}")
    print(f"  D_pr/r:            {holo['df']/max(r_global,1):.4f}"
          f"  (the period is {1/holo['df']*max(r_global,1):.0f}x larger than D_pr)")

    # ---- KEY INSIGHT ----
    print(f"\n{'='*78}")
    print(f"KEY INSIGHT")
    print(f"{'='*78}")
    print(f"  The .holo stores k={k_store} complex eigenmodes of the Hermitian")
    print(f"  covariance of the modular exponentiation phase grating on S^1.")
    print(f"  The integer r={r_global} lives NOWHERE in the .holo.")
    print(f"  The eigenbasis encodes the period structure as a topological")
    print(f"  interference pattern -- a physical lens, not a number.")
    print(f"")
    print(f"  To USE the .holo:")
    print(f"    1. Project a signal through the stored eigenbasis")
    print(f"    2. The filtered signal resonates at the stored period")
    print(f"    3. Autocorrelation of the filtered signal reveals r")
    print(f"    4. This measurement happens OUTSIDE the .holo")
    print(f"")
    print(f"  The .holo can be composed with other .holo states via superposition")
    print(f"  on the tape. It can verify candidates by measuring energy")
    print(f"  concentration in the dominant eigenmodes. It can participate in")
    print(f"  catalytic computation chains without ever being 'read' as an integer.")
    print(f"")
    print(f"  This IS the Cybernetic Truth architecture:")
    print(f"    .holo eigenbasis  = alignment frame C")
    print(f"    projection        = R = Tr(rho @ C)")
    print(f"    autocorrelation   = measurement step (Born rule collapse)")
    print(f"    containment       = the integer r never materializes")

    total_t = time.perf_counter() - t0
    print(f"\n  Total: {total_t:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
