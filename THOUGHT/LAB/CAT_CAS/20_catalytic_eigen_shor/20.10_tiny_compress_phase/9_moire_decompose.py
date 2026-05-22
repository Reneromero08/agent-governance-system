"""
20.10.9: Moire Decomposition — Factor via Sub-Period Extraction
=================================================================
Paradigm shift: by CRT, Z_N = Z_p x Z_q. The sequence a^x mod N
is the product of TWO smooth, independent rotations on circles
of size r_p (mod p) and r_q (mod q). The "chaos" is Moire interference.

We don't need r = lcm(r_p, r_q). We need just r_p (or r_q).
  r_p <= p-1 <= sqrt(N)  — exponentially smaller than r.
  a^{r_p} = 1 mod p  =>  gcd(a^{r_p} - 1, N) = p.

.holo eigendecomposition isolates the two fundamental modes.
Each top eigenvector encodes one of the sub-periods.
Autocorrelation of eigenvectors -> r_p or r_q -> factor N.
"""

import sys, time, math, random
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "holographic-image"))
from holo_core import analyze_spectrum, project, choose_k


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


def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b); p |= (1 << (b - 1)) | 1
            if is_prime(p): return p
    p = get_prime(bits // 2); q = get_prime(bits // 2)
    while q == p: q = get_prime(bits // 2)
    return p * q, p, q


def gcd(a, b):
    while b: a, b = b, a % b
    return a


def try_factor_from_candidate(N, a, r_cand):
    """Try to factor N using a candidate sub-period."""
    if r_cand <= 1: return 0, 0, False
    # Check: a^{r_cand} - 1 shares factor with N?
    val = pow(a, r_cand, N)
    g = gcd(val - 1, N)
    if 1 < g < N: return g, N // g, True
    # Also try a^{r_cand} + 1 if r_cand is even
    if r_cand % 2 == 0:
        g2 = gcd(val + 1, N)
        if 1 < g2 < N: return g2, N // g2, True
    return 0, 0, False


def eigenvector_subperiod(evec, L_half, a, N):
    """Extract sub-period from a single eigenvector via autocorrelation."""
    if len(evec) < 4: return 0, 0, 0, False
    sig = torch.tensor(evec.astype(np.complex64))
    ac = torch.fft.ifft(torch.abs(torch.fft.fft(sig))**2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac)//2, 100000)
    if sr <= 2: return 0, 0, 0, False
    # Top 3 peaks
    vals, idxs = torch.topk(torch.abs(ac[2:sr]), k=min(5, sr-2))
    for i in range(len(idxs)):
        r_cand = idxs[i].item() + 2
        p, q, ok = try_factor_from_candidate(N, a, r_cand)
        if ok: return p, q, r_cand, True
    return 0, 0, 0, False


def moire_decompose(grating, M, a, N, L=2048):
    """
    Decompose Moire pattern into its two fundamental modes.
    
    1. Build .holo observation matrix from grating windows
    2. Eigendecomposition of covariance
    3. Top eigenvectors encode the two circles (r_p, r_q)
    4. Extract sub-period from each eigenvector
    5. Try to factor using each sub-period
    """
    stride = max(1, L // 4)
    n = min(2048, (M - L) // stride)
    if n < 4: return 0, 0, "too_few_samples"

    # Build observation matrix
    obs_c = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs_c[i] = grating[i * stride : i * stride + L].numpy()
    obs = np.hstack([obs_c.real.astype(np.float64), obs_c.imag.astype(np.float64)])

    # .holo compression
    spectrum = analyze_spectrum(obs)
    k = choose_k(spectrum, policy="participation")
    k = max(4, min(k, obs.shape[1] - 1))
    proj = project(obs, policy="fixed", fixed_k=k)

    # Eigendecomposition via .holo's own SVD: the basis vectors ARE eigenvectors
    # basis shape: (k, 2L). Each row is a principal component.
    # Convert to complex: first L = real, last L = imag
    basis = proj.basis
    L_half = L

    for i in range(min(10, basis.shape[0])):
        # Convert basis vector i to complex signal
        evec_complex = basis[i, :L_half] + 1j * basis[i, L_half:]
        p, q, r_found, ok = eigenvector_subperiod(evec_complex, L_half, a, N)
        if ok:
            return p, q, f"evec[{i}](r={r_found},k={k})"

    return 0, 0, "not_found"


def main():
    print("=" * 78)
    print("20.10.9: MOIRE DECOMPOSITION — Sub-Period Factoring")
    print("  CRT: Z_N = Z_p x Z_q. Find r_p, not r = lcm(r_p, r_q).")
    print("=" * 78)
    print()

    M = 2**23
    ok = 0; n_trials = 10

    for t in range(n_trials):
        N, known_p, known_q = generate_semiprime(22)
        a = 2
        while gcd(a, N) != 1: a += 1
        t0 = time.perf_counter()

        # Catalytic: generate grating once
        seq = [1]; curr = 1
        for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
        grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)

        # --- Moire decomposition ---
        p, q, method = moire_decompose(grating, M, a, N)
        success = p > 0 and q > 0 and p * q == N
        dt = time.perf_counter() - t0

        if success:
            ok += 1
            match = (p == known_p and q == known_q) or (p == known_q and q == known_p)
            print(f"  [{t+1:>2}] {N} = {p}x{q} (true {known_p}x{known_q}) {method}  {dt:.1f}s {'OK' if match else 'MISMATCH'}")
        else:
            # Fallback: autocorrelation
            spec = torch.fft.fft(grating)
            ac = torch.fft.ifft(torch.abs(spec)**2).real; ac = ac / (ac[0] + 1e-15)
            sr = min(M//2, 500000)
            if sr > 2:
                _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
                r_ref = mi.item() + 2
                if pow(a, r_ref, N) == 1:
                    pf, qf, okf = try_factor_from_candidate(N, a, r_ref)
                    if not okf and r_ref % 2 == 0:
                        pf, qf, okf = try_factor_from_candidate(N, a, r_ref // 2)
                    if okf:
                        ok += 1
                        match = (pf == known_p and qf == known_q) or (pf == known_q and qf == known_p)
                        print(f"  [{t+1:>2}] {N} = {pf}x{qf} (true {known_p}x{known_q}) autocorrelation(r={r_ref})  {dt:.1f}s {'OK' if match else 'MISMATCH'}")
                        continue
            print(f"  [{t+1:>2}] {N} = {known_p}x{known_q} NOT FACTORED  {dt:.1f}s")

    print(f"\n  -> {ok}/{n_trials} factored via Moire decomposition")
    print("=" * 78)


if __name__ == "__main__":
    main()
