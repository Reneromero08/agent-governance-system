"""20.10.13: Level 1 Scale Push — fixed k, larger L, brute-force eigenvector scan"""
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

def try_factor(N, a, r_cand):
    if r_cand <= 1: return 0, 0, False
    val = pow(a, r_cand, N)
    g = gcd(val - 1, N)
    if 1 < g < N: return g, N // g, True
    if r_cand % 2 == 0:
        g2 = gcd(val + 1, N)
        if 1 < g2 < N: return g2, N // g2, True
    return 0, 0, False

def autocorr_peaks(sig, max_len, topk=10):
    if len(sig) < 4: return []
    ac = torch.fft.ifft(torch.abs(torch.fft.fft(sig))**2).real
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac)//2, max_len)
    if sr <= 2: return []
    vals, idxs = torch.topk(torch.abs(ac[2:sr]), k=min(topk, sr-2))
    return [(i.item() + 2) for i in idxs]

def level1_scan(grating, M, a, N, L, k_fixed=10):
    """Level 1: .holo eigenvector -> sub-period -> factor."""
    stride = max(1, L // 4)
    n = (M - L) // stride
    n = min(n, 8192)
    if n < 4: return 0, 0, "too_few", False

    obs_c = np.zeros((n, L), dtype=np.complex128)
    for i in range(n):
        obs_c[i] = grating[i * stride : i * stride + L].numpy()
    obs = np.hstack([obs_c.real.astype(np.float64), obs_c.imag.astype(np.float64)])

    proj = project(obs, policy="fixed", fixed_k=k_fixed)
    basis = proj.basis

    for i in range(min(k_fixed, basis.shape[0])):
        evec = basis[i, :L] + 1j * basis[i, L:]
        peaks = autocorr_peaks(torch.tensor(evec.astype(np.complex64)), L)
        for r_cand in peaks:
            p, q, ok = try_factor(N, a, r_cand)
            if ok: return p, q, f"evec[{i}](r={r_cand})", True
            for div in [2, 3, 4, 5]:
                if r_cand % div == 0:
                    p, q, ok = try_factor(N, a, r_cand // div)
                    if ok: return p, q, f"evec[{i}](r={r_cand//div})", True
    return 0, 0, "not_found", False

def main():
    print("=" * 78)
    print("20.10.13: LEVEL 1 SCALE PUSH")
    print("=" * 78)

    for bits in [22, 26, 30]:
        M = 2**23 if bits <= 26 else 2**24
        L = 4096 if bits <= 22 else (8192 if bits <= 26 else 16384)
        k = 10

        ok_evec = ok_ref = 0
        print(f"\n--- {bits}-bit, M=2^{int(math.log2(M))}, L={L}, k={k} ---")

        for t in range(5):
            N, kp, kq = generate_semiprime(bits)
            a = 2
            while gcd(a, N) != 1: a += 1
            t0 = time.perf_counter()

            seq = [1]; curr = 1
            for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
            grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)

            p, q, method, evec_ok = level1_scan(grating, M, a, N, L, k)

            if not evec_ok:
                r_ref = autocorr_peaks(grating, M//2, topk=5)
                for r in r_ref:
                    if r > 0 and pow(a, r, N) == 1:
                        p, q, evec_ok = try_factor(N, a, r)
                        if not evec_ok and r % 2 == 0:
                            p, q, evec_ok = try_factor(N, a, r // 2)
                        if evec_ok: method = f"autocorr(r={r})"; break

            dt = time.perf_counter() - t0
            success = p > 0 and q > 0 and p * q == N
            match = success and ((p == kp and q == kq) or (p == kq and q == kp))
            if match and "evec" in method: ok_evec += 1
            if success: ok_ref += 1

            est_rp = min(kp - 1, kq - 1)
            status = f"{p}x{q}" if success else "FAIL"
            print(f"  [{t+1}] {status} via {method} est_rp={est_rp} {dt:.1f}s")

        print(f"  .holo evec: {ok_evec}/5  total: {ok_ref}/5")

    print("\n" + "=" * 78)

if __name__ == "__main__":
    main()
