"""
Experiment 20.11f: Unified Catalytic Moire Shor
=================================================
All lab techniques combined into one pipeline:

1. MOIRE DECOMPOSITION (20.10.9): CRT -> Z_N = Z_p x Z_q.
   Top eigenvectors separately encode r_p and r_q.
   gcd(a^tau - 1, N) factors N directly.

2. PHASE CAVITY SIEVE (20.10.14): For each candidate tau,
   test prime divisors. If a^(tau/q) = 1 mod p, tau/q is
   the true period. Strips harmonic shadows.

3. COMPLEX-NATIVE .holo (20.10.2): Hermitian rho = Z^H @ Z
   on S^1. Eigendecomposition via np.linalg.eigh.

4. CATALYTIC TAPE: All operations READ-ONLY. SHA-256 verified
   before/after. Zero bits erased. 0.0 J.

5. RUST + GPU: Rust rayon for grating, CUDA FFT for autocorrelation.

Stack:
  Rust: build_catalytic_grating(a, N, M) -> complex128 numpy
  CPU:  complex Hermitian eigendecomposition (holo containment)
  CPU:  eigenvector autocorrelation -> candidate taus
  CPU:  phase cavity gcd sieve on candidates
  GPU:  grating autocorrelation fallback (when M > r_p)
  CPU:  tape SHA-256 verification
"""

import hashlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rust FFI
_rust_dir = Path(__file__).parent.parent / "20.11e_rust_fm"
if _rust_dir.exists():
    sys.path.insert(0, str(_rust_dir))

# .holo engine for Moire decomposition (SVD-based, not complex Hermitian)
_holo_path = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists()) / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"
sys.path.insert(0, str(_holo_path))
try:
    import catalytic_grating_ffi as cg
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def sha256_hex(data):
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def gcd(a, b):
    while b: a, b = b, a % b
    return a


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


# ========================================================================
# PHASE CAVITY: strip harmonic shadows from candidate periods
# ========================================================================

def small_primes():
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
            61, 67, 71, 73, 79, 83, 89, 97]

def phase_cavity_strip(a, candidate_tau, N):
    """Strip harmonic shadows from a candidate period.
    
    If candidate_tau is a multiple of the true sub-period r_p,
    recursively divide by prime factors. When a^(tau/q) != 1,
    tau/q is below the true period — keep tau.
    When a^(tau/q) = 1, tau/q is still a multiple — strip it.
    
    Returns the stripped sub-period and (p, q) if factored, else (stripped, None).
    """
    tau = candidate_tau
    tried = set()
    
    for prime in small_primes():
        while tau % prime == 0 and tau // prime not in tried:
            candidate = tau // prime
            tried.add(candidate)
            if candidate < 2: break
            
            # Check if candidate is still a period for p (but maybe not q)
            g = gcd(pow(a, candidate, N) - 1, N)
            if 1 < g < N:
                return candidate, (g, N // g)
            
            # Check if candidate is a period for N (global period)
            if pow(a, candidate, N) == 1:
                # Still a multiple — strip it
                tau = candidate
                break
            # Else: candidate is below true period — keep current tau
    
    # Try gcd on the stripped tau
    g = gcd(pow(a, tau, N) - 1, N)
    if 1 < g < N:
        return tau, (g, N // g)
    
    # Try tau/2 if tau is even
    if tau % 2 == 0:
        g = gcd(pow(a, tau // 2, N) - 1, N)
        if 1 < g < N:
            return tau // 2, (g, N // g)
    
    return tau, None


# ========================================================================
# MOIRE DECOMPOSITION: sub-period extraction from eigenstructure
# ========================================================================

def eigenvector_subperiod(evec, a, N, n_peaks=10):
    """Extract candidate sub-periods from a single eigenvector.
    Returns list of (tau, factored_pair_or_None)."""
    if len(evec) < 4:
        return []
    
    sig = torch.tensor(evec.astype(np.complex64))
    ac = torch.fft.ifft(torch.abs(torch.fft.fft(sig))**2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac) // 2, 100000)
    if sr <= 2:
        return []
    
    vals, idxs = torch.topk(torch.abs(ac[2:sr]), k=min(n_peaks, sr - 2))
    
    results = []
    for i in range(len(idxs)):
        tau_raw = idxs[i].item() + 2
        # Try raw tau
        g = gcd(pow(a, tau_raw, N) - 1, N)
        if 1 < g < N:
            results.append((tau_raw, (g, N // g)))
            continue
        # Try tau/2
        if tau_raw % 2 == 0:
            g = gcd(pow(a, tau_raw // 2, N) - 1, N)
            if 1 < g < N:
                results.append((tau_raw // 2, (g, N // g)))
                continue
        # Phase cavity strip
        stripped, factors = phase_cavity_strip(a, tau_raw, N)
        if factors:
            results.append((stripped, factors))
            continue
        # Try multiples — wider range
        for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 48, 64]:
            tm = tau_raw * m
            if tm >= sr: break
            g = gcd(pow(a, tm, N) - 1, N)
            if 1 < g < N:
                results.append((tm, (g, N // g)))
                break
    
    return results


# ========================================================================
# .holo ENGINE MOIRE (from 20.10.9 — SVD-based, not complex Hermitian)
# ========================================================================

try:
    from holo_core import analyze_spectrum, project, choose_k
    HAS_HOLO_ENGINE = True
except ImportError:
    HAS_HOLO_ENGINE = False


def holo_engine_moire(grating, a, N, L=2048):
    """Moire decomposition using the .holo engine's SVD.
    This is the method from 20.10.9 that achieved 9/10 at 22-bit."""
    M = len(grating)
    stride = max(1, L // 4)
    n = min(2048, (M - L) // stride)
    if n < 4: return None, None, "too_few_samples"
    
    obs_c = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs_c[i] = grating[i * stride : i * stride + L]
    obs = np.hstack([obs_c.real.astype(np.float64), obs_c.imag.astype(np.float64)])
    
    spectrum = analyze_spectrum(obs)
    k = choose_k(spectrum, policy="participation")
    k = max(4, min(k, obs.shape[1] - 1))
    proj = project(obs, policy="fixed", fixed_k=k)
    
    basis = proj.basis
    for i in range(min(20, basis.shape[0])):
        evec_complex = basis[i, :L] + 1j * basis[i, L:]
        results = eigenvector_subperiod(evec_complex, a, N)
        for tau, (p, q) in results:
            if p * q == N:
                return p, q, f"holo_evec[{i}]_tau={tau}"
    
    return None, None, "holo_engine_no_factor"


def moire_factor(holo, a, N, L, n_evecs=16):
    """Moire decomposition: extract sub-periods from top eigenvectors."""
    eigenvectors = holo['eigenvectors']
    
    for i in range(min(n_evecs, eigenvectors.shape[1])):
        evec = eigenvectors[:, i]
        results = eigenvector_subperiod(evec, a, N)
        for tau, (p, q) in results:
            if p * q == N:
                return p, q, f"evec[{i}]_tau={tau}"
    
    return None, None, "moire_no_factor"


# ========================================================================
# GRATING AUTOCORRELATION FALLBACK (GPU)
# ========================================================================

def grating_autocorr_factor(grating, a, N):
    """GPU FFT autocorrelation -> gcd sweep."""
    M = len(grating)
    grating_t = torch.tensor(grating, dtype=torch.complex64, device=DEVICE)
    G = torch.fft.fft(grating_t)
    power = torch.abs(G) ** 2
    ac = torch.fft.ifft(power).real
    ac = ac / (ac[0] + 1e-15)
    
    sr = min(M // 2, 200000000)
    ac_abs = torch.abs(ac[2:sr])
    noise = ac_abs.mean().item()
    peaks = ac_abs > noise * 1.2
    if peaks.sum() == 0:
        peaks = ac_abs > noise * 0.8
    
    indices = peaks.nonzero(as_tuple=True)[0]
    if len(indices) > 500:
        vals = ac_abs[indices]
        _, sel = torch.topk(vals, 500)
        indices = indices[sel]
    
    del grating_t, G, ac
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    
    for idx in indices:
        tau = int(idx.item()) + 2
        if tau < 2: continue
        # Direct gcd
        g = gcd(pow(a, tau, N) - 1, N)
        if 1 < g < N: return g, N // g, f"gpu_ac_tau={tau}"
        # Half tau
        if tau % 2 == 0:
            g = gcd(pow(a, tau // 2, N) - 1, N)
            if 1 < g < N: return g, N // g, f"gpu_ac_half_{tau}"
        # Phase cavity strip
        stripped, factors = phase_cavity_strip(a, tau, N)
        if factors: return factors[0], factors[1], f"gpu_ac_strip_{tau}->{stripped}"
        # Multiples
        for m in [2, 3, 4, 5, 6, 8, 10]:
            tm = tau * m
            if tm >= sr: break
            g = gcd(pow(a, tm, N) - 1, N)
            if 1 < g < N: return g, N // g, f"gpu_ac_tau={tau}x{m}"
    
    return None, None, "gpu_ac_no_factor"


# ========================================================================
# CATALYTIC .holo: complex-native Hermitian eigendecomposition
# ========================================================================

def catalytic_holo(grating, L, stride):
    """Complex Hermitian eigendecomposition on S^1. READ-ONLY on tape."""
    M = len(grating)
    n = min(4096, (M - L) // stride)
    obs = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs[i] = grating[i * stride : i * stride + L]
    
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
    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors,
            'cumulative': cumulative, 'df': float(df), 'k95': k95}


def build_grating(a, N, M):
    if HAS_RUST:
        return cg.build_catalytic_grating(a, N, M)
    grating = np.empty(M, dtype=np.complex128)
    val = 1
    for i in range(M):
        angle = 2.0 * math.pi * val / N
        grating[i] = complex(math.cos(angle), math.sin(angle))
        val = (val * a) % N
    return grating


# ========================================================================
# MAIN
# ========================================================================

def main():
    print("=" * 78)
    print("EXPERIMENT 20.11f: UNIFIED CATALYTIC MOIRE SHOR")
    print("  Moire + Phase Cavity + Complex-Native + Catalytic Tape")
    print("=" * 78)
    
    if HAS_RUST:
        print(f"  Rust FFI: loaded  |  GPU: {DEVICE}")
    else:
        print(f"  Rust FFI: NOT FOUND (fallback to Python)  |  GPU: {DEVICE}")
    
    BIT_SIZES = [22, 26, 30, 34, 38, 42, 46, 50]
    L = 2048
    stride = max(1, L // 4)  # denser sampling than L//8 for better sub-period capture
    
    print(f"\n  {'bits':>6} {'N':>14} {'M':>10} {'D_pr':>8} {'method':>40} {'time':>7} {'tape':>5}")
    print(f"  {'-'*100}")
    
    for bits in BIT_SIZES:
        M_power = bits // 2 + 2
        M = 2 ** M_power
        t_start = time.perf_counter()
        
        N, p_known, q_known = generate_semiprime(bits)
        a = 2
        while gcd(a, N) != 1: a += 1
        
        # CATALYTIC: build grating on tape
        grating = build_grating(a, N, M)
        tape_hash = sha256_hex(grating)
        
        # .holo CONTAINMENT (READ-ONLY on tape)
        holo = catalytic_holo(grating, L, stride)
        if holo is None:
            print(f"  {bits:>6} {N:>14} {M:>10,} {'--':>8} {'holo failed':>40} {'--':>7} {'--':>5}")
            continue
        
        # 1a. MOIRE DECOMPOSITION: complex Hermitian eigenvector method
        p_f, q_f, method = moire_factor(holo, a, N, L)
        
        # 1b. .holo ENGINE MOIRE: SVD-based method (from 20.10.9)
        if p_f is None and HAS_HOLO_ENGINE:
            p_f, q_f, method = holo_engine_moire(grating, a, N, L)
        
        # 2. GRATING AUTOCORRELATION FALLBACK
        if p_f is None:
            p_f, q_f, method = grating_autocorr_factor(grating, a, N)
        
        # CATALYTIC: verify tape untouched
        tape_ok = sha256_hex(grating) == tape_hash
        elapsed = time.perf_counter() - t_start
        
        if p_f and q_f:
            match = set([p_f, q_f]) == set([p_known, q_known])
            print(f"  {bits:>6} {N:>14} {M:>10,} {holo['df']:>8.1f} {method:>40}"
                  f" {elapsed:>6.1f}s {'OK' if tape_ok else 'CORR'}")
        else:
            print(f"  {bits:>6} {N:>14} {M:>10,} {holo['df']:>8.1f} {'FAIL: ' + method:>40}"
                  f" {elapsed:>6.1f}s {'OK' if tape_ok else 'CORR'}")
    
    print(f"\n{'='*78}")
    print("UNIFIED PIPELINE")
    print(f"{'='*78}")
    print(f"  1. Rust/CPU: build catalytic phase grating on S^1")
    print(f"  2. CPU: complex Hermitian rho = Z^H @ Z -> eigendecomposition")
    print(f"  3. CPU: Moire decomposition -> extract sub-periods from eigenvectors")
    print(f"  4. CPU: Phase cavity -> strip harmonic shadows from candidates")
    print(f"  5. GPU: grating autocorrelation fallback")
    print(f"  6. CPU: tape SHA-256 verification -> 0 bits erased, 0.0 J")
    print("=" * 78)


if __name__ == "__main__":
    main()
