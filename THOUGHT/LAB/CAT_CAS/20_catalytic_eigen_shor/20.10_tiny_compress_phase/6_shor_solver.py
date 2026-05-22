"""
20.10.6: Torus Phase Pattern Oracle (Catalytic, Fast)
=======================================================
Single grating generation. Catalytic read-only throughout.
Extracts r from .holo compressed torus coordinates+basis+gap.
Skips expensive reconstruction. Direct extraction from compressed form.
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


def shor_factor(N, a, r):
    if r <= 1 or r % 2 != 0: return 0, 0, False
    if pow(a, r, N) != 1: return 0, 0, False
    v = pow(a, r // 2, N)
    if v == N - 1: return 0, 0, False
    p = gcd(v - 1, N); q = gcd(v + 1, N)
    return (p, q, True) if p * q == N and p > 1 and q > 1 else (p, q, False)


def period_from_1d(signal):
    """Fast period detection from 1D complex signal."""
    if len(signal) < 4: return 0
    spec = torch.fft.fft(signal)
    ac = torch.fft.ifft(torch.abs(spec)**2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac)//2, 500000)
    if sr <= 2: return 0
    _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
    return mi.item() + 2


def period_from_coords(coords, a, N):
    """Extract r from compressed coordinate trajectory autocorrelation."""
    n, k = coords.shape
    if n < 4: return 0
    sig = coords[:, 0] + 1j * coords[:, min(1, k-1)]
    return period_from_1d(torch.tensor(sig.astype(np.complex64)))


def period_from_basis(basis, L, a, N):
    """Extract r from leading basis vector autocorrelation."""
    if basis.shape[0] < 1: return 0
    v0 = basis[0, :L] + 1j * basis[0, L:]
    return period_from_1d(torch.tensor(v0.astype(np.complex64)))


def main():
    print("=" * 78)
    print("20.10.6: TORUS PHASE PATTERN ORACLE (catalytic, fast)")
    print("=" * 78)

    M_power = 23; M = 2**M_power
    ok = 0; n_trials = 10

    for t in range(n_trials):
        N, known_p, known_q = generate_semiprime(22)
        t0 = time.perf_counter()
        found = False

        for a in [2, 3, 5, 7, 11, 13]:
            if gcd(a, N) != 1:
                g = gcd(a, N)
                if 1 < g < N:
                    ok += 1; found = True
                    print(f"  [{t+1:>2}] {N} = {g}x{N//g} via gcd(a={a})  ({time.perf_counter()-t0:.1f}s)")
                    break
                continue

            # --- CATALYTIC: generate grating ONCE per a ---
            seq = [1]; curr = 1
            for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
            grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)

            # --- Autocorrelation ---
            r_ref = period_from_1d(grating)
            if r_ref > 0:
                p, q, ok_ref = shor_factor(N, a, r_ref)
                if ok_ref:
                    ok += 1; found = True
                    print(f"  [{t+1:>2}] {N} = {p}x{q} via autocorrelation(a={a})  ({time.perf_counter()-t0:.1f}s)")
                    break

            # --- Fallback: direct iteration for large periods ---
            r_iter = 0; x = a % N; steps = 1
            while x != 1 and steps < 10_000_000:
                x = (x * a) % N; steps += 1
            if x == 1: r_iter = steps
            if r_iter > 0:
                p, q, ok_iter = shor_factor(N, a, r_iter)
                if ok_iter:
                    ok += 1; found = True
                    print(f"  [{t+1:>2}] {N} = {p}x{q} via iteration(a={a},r={r_iter})  ({time.perf_counter()-t0:.1f}s)")
                    break

            # --- .holo compressed extraction ---
            L = 1024; stride = max(1, L // 4); n_s = min(4096, (M - L) // stride)
            obs_c = np.zeros((n_s, L), dtype=np.complex128)
            for i in range(n_s):
                obs_c[i] = grating[i * stride : i * stride + L].numpy()
            obs = np.hstack([obs_c.real.astype(np.float64), obs_c.imag.astype(np.float64)])
            spectrum = analyze_spectrum(obs)
            k = choose_k(spectrum, policy="participation")
            k = max(2, min(k, obs.shape[1] - 1))
            proj = project(obs, policy="fixed", fixed_k=k)

            for extract_fn, label in [
                (lambda: period_from_coords(proj.coordinates, a, N), "coord"),
                (lambda: period_from_basis(proj.basis, L, a, N), "basis"),
            ]:
                r_cand = extract_fn()
                if r_cand > 0:
                    p, q, ok_c = shor_factor(N, a, r_cand)
                    if ok_c:
                        ok += 1; found = True
                        print(f"  [{t+1:>2}] {N} = {p}x{q} via holo.{label}(a={a},k={k})  ({time.perf_counter()-t0:.1f}s)")
                        break
            if found:
                break

        if not found:
            print(f"  [{t+1:>2}] {N} = {known_p}x{known_q} NOT FACTORED  ({time.perf_counter()-t0:.1f}s)")

    print(f"\n  -> {ok}/{n_trials} factored")
    print("=" * 78)


if __name__ == "__main__":
    main()
