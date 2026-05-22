"""
20.10.8: Latent Lattice Oracle
================================
Oracle-first architecture. Drop sparse probes across the ENTIRE
global space [0, N], not just a local window [0, M]. Project
into .holo latent space. Use lattice geometry to extract r.

Architecture (user's blueprint):
  1. ORACLE: pow(a, x_i, N) at K random positions x_i in [0, N]
     -> O(K log N) time, O(K) memory, spans global space
  2. .holo MANIFOLD: project sparse samples into compressed latent space
     -> D_pr << r means random points collapse onto structured manifold
  3. LATTICE REDUCTION: measure generator of the latent manifold
     -> torus winding angle -> continued fractions -> r
  4. VERIFY: pow(a, r, N) == 1 in O(log r)

Key insight: the oracle handles PHYSICAL DISTANCE (jumping O(N) steps
instantly via modular exponentiation). The .holo engine handles MEMORY
(compressing phase space so we don't need N^2 elements). The lattice
algorithm connects the dots.

The manifold: g_n = exp(2pi*i * a^n / N). In latent space, points
lie on a 1D curve (closed orbit). The curve's winding number = period r.
"""

import sys, time, math, random
from pathlib import Path
import numpy as np
import torch
from fractions import Fraction

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


# =====================================================================
# STEP 1: Oracle Sparse Probes (global space, not local window)
# =====================================================================

def oracle_sparse_probes(a, N, K=3000):
    """
    Drop K sparse probes across the ENTIRE space [0, N].
    Uses pow(a, x, N) for O(log x) modular exponentiation.
    Returns probes SORTED by position x.
    """
    probes = []
    for i in range(K):
        # Uniformly spaced across full range [0, N] with small random jitter
        x = int(i * N / K) + random.randint(0, max(1, N // K // 10))
        x = min(x, N - 1)
        val = pow(a, x, N)
        phase = 2.0 * math.pi * val / N
        probes.append((x, complex(math.cos(phase), math.sin(phase))))
    # Already sorted by construction (i increases -> x increases)
    return probes


# =====================================================================
# STEP 2: .holo Manifold Projection
# =====================================================================

def holo_manifold_project(probes, L=64):
    """
    Project sparse probes into .holo latent space.
    Build observation matrix from windows of L consecutive probes.
    Returns coordinates in compressed space.
    """
    complex_vals = np.array([p[1] for p in probes], dtype=np.complex128)
    n = len(complex_vals) - L
    if n < 4: return None, None

    obs = np.zeros((n, L * 2), dtype=np.float64)
    for i in range(n):
        w = complex_vals[i : i + L]
        obs[i, :L] = w.real
        obs[i, L:] = w.imag

    spectrum = analyze_spectrum(obs)
    k = choose_k(spectrum, policy="participation")
    k = max(3, min(k, obs.shape[1] - 1))
    proj = project(obs, policy="fixed", fixed_k=k)

    return proj, spectrum


# =====================================================================
# STEP 3: Lattice Geometry — extract generator from latent manifold
# =====================================================================

def latent_lattice_period(proj, a, N):
    """
    Extract period from the latent point cloud geometry.

    The projected points lie on a 1D curve in k-dimensional space.
    The curve wraps around a torus with winding number = r.

    Method: measure pairwise distances in latent space. Points that
    are r apart in the original sequence should be close in latent space.
    Find the fundamental translation that maps the manifold to itself.
    """
    coords = proj.coordinates  # (n, k)
    n, k = coords.shape
    if n < 10: return 0

    # Use first 3 latent dimensions as the manifold embedding
    d = min(3, k)
    points = coords[:, :d]

    # Compute pairwise distances and look for periodic structure
    # For each candidate lag tau, check if points i and i+tau are close
    candidates = set()

    # Method 1: autocorrelation of latent trajectory
    sig = torch.tensor(points[:, 0].astype(np.float32))
    spec = torch.fft.rfft(sig, n=min(len(sig) * 2, 131072))
    ac = torch.fft.irfft(torch.abs(spec)**2)
    ac = ac / (ac[0] + 1e-15)
    sr = min(len(ac)//2, 500000)
    if sr > 2:
        _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
        r_cand = mi.item() + 2
        candidates.add(r_cand)
        for m in range(2, 20):
            candidates.add(r_cand * m)

    # Method 2: measure the dominant direction in latent space
    # The trajectory traces a curve. Its tangent vector reveals the generator.
    if n > 5:
        diffs = np.diff(points, axis=0)  # (n-1, d)
        # Dominant direction via SVD of diffs
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
        direction = vt[0]  # dominant direction

        # Project all points onto this direction
        proj_1d = points @ direction  # (n,)
        # Autocorrelation of 1D projection
        sig_1d = torch.tensor(proj_1d.astype(np.float32))
        spec_1d = torch.fft.rfft(sig_1d, n=min(len(sig_1d) * 2, 131072))
        ac_1d = torch.fft.irfft(torch.abs(spec_1d)**2)
        ac_1d = ac_1d / (ac_1d[0] + 1e-15)
        sr_1d = min(len(ac_1d)//2, 500000)
        if sr_1d > 2:
            _, mi_1d = torch.max(torch.abs(ac_1d[2:sr_1d]), dim=0)
            r_cand2 = mi_1d.item() + 2
            candidates.add(r_cand2)
            for m in range(2, 20):
                candidates.add(r_cand2 * m)

    # Method 3: PAIRWISE SYMMETRY in latent space
    # Period-related points should be close in latent space.
    # Find close pairs, their position differences are multiples of r.
    if n > 20 and d >= 2:
        # Compute all pairwise distances (subsample for speed)
        step = max(1, n // 300)
        subset = points[::step]
        n_sub = len(subset)
        close_pairs = []
        for i in range(0, n_sub, 5):
            for j in range(i + 5, n_sub, 5):
                dist = np.linalg.norm(subset[i] - subset[j])
                if dist < np.percentile(np.linalg.norm(np.diff(subset, axis=0), axis=1), 20):
                    # These points are close in latent space
                    # Their POSITION difference (in original sequence) may relate to r
                    pos_i = i * step
                    pos_j = j * step
                    delta = abs(pos_j - pos_i)
                    if delta > 1:
                        close_pairs.append(delta)

        if len(close_pairs) >= 3:
            # GCD of position differences gives candidate period
            g = close_pairs[0]
            for d_val in close_pairs[1:min(50, len(close_pairs))]:
                g = math.gcd(g, d_val)
            if g > 1:
                candidates.add(g)
                candidates.add(g * 2)

    # Method 4: torus winding angle from phase of complex latent coords
    if k >= 2:
        z = coords[:, 0] + 1j * coords[:, 1]  # complex from first 2 dims
        angles = np.angle(z)
        # Fit unwrapped phase: angle(i) = slope * i + offset
        # Use circular statistics
        X = np.column_stack([np.arange(n), np.ones(n)])
        cos_a = np.cos(angles); sin_a = np.sin(angles)
        cos_coef = np.linalg.lstsq(X, cos_a, rcond=None)[0]
        sin_coef = np.linalg.lstsq(X, sin_a, rcond=None)[0]
        slope_angle = math.atan2(sin_coef[0], cos_coef[0])
        if slope_angle < 0: slope_angle += 2.0 * math.pi

        # slope_angle / (2*pi) is related to 1/r
        # For sequential sampling: slope = 2*pi/r, so r = 2*pi/slope
        # For random sampling: relationship is more complex
        if slope_angle > 1e-10:
            ratio = Fraction(slope_angle / (2.0 * math.pi)).limit_denominator(N)
            candidates.add(ratio.denominator)
            candidates.add(int(2.0 * math.pi / slope_angle))

    # Verify candidates
    for r_cand in sorted(set(int(c) for c in candidates if 1 < c < N)):
        if pow(a, r_cand, N) == 1:
            return r_cand
        for m in range(2, 20):
            if r_cand * m < N and pow(a, r_cand * m, N) == 1:
                return r_cand * m

    return 0


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 78)
    print("20.10.8: LATENT LATTICE ORACLE")
    print("  Oracle probes -> .holo manifold -> lattice reduction -> r")
    print("=" * 78)
    print()

    n_trials = 20
    ok_lattice = 0
    ok_ref = 0

    for t in range(n_trials):
        N, known_p, known_q = generate_semiprime(22)
        a = 2
        while gcd(a, N) != 1: a += 1
        t0 = time.perf_counter()

        # --- Reference: full autocorrelation ---
        M = 2**23
        seq = [1]; curr = 1
        for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
        grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)
        spec = torch.fft.fft(grating)
        ac = torch.fft.ifft(torch.abs(spec)**2).real; ac = ac / (ac[0] + 1e-15)
        sr = min(M//2, 500000)
        r_ref = 0
        if sr > 2:
            _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
            r_ref = mi.item() + 2
        ref_ok = r_ref > 0 and pow(a, r_ref, N) == 1
        if ref_ok:
            p, q, ref_fac = shor_factor(N, a, r_ref)
            if ref_fac: ok_ref += 1

        # --- Latent Lattice Oracle ---
        K = 2000  # sparse probes across global space
        probes = oracle_sparse_probes(a, N, K=K)
        proj, spectrum = holo_manifold_project(probes, L=64)

        r_lattice = 0
        if proj is not None:
            r_lattice = latent_lattice_period(proj, a, N)

        lattice_ok = r_lattice > 0
        if lattice_ok:
            p, q, lattice_fac = shor_factor(N, a, r_lattice)
            if lattice_fac: ok_lattice += 1

        dt = time.perf_counter() - t0
        status = []
        if ref_ok: status.append(f"ref(r={r_ref})")
        if lattice_ok: status.append(f"lattice(r={r_lattice})")

        print(f"  [{t+1:>2}] {N}={known_p}x{known_q}  {', '.join(status) if status else 'ALL FAILED'}  K={K} D_pr={spectrum.participation_dimension:.0f}  {dt:.1f}s")

    print(f"\n  Reference: {ok_ref}/{n_trials}")
    print(f"  Lattice:   {ok_lattice}/{n_trials}")
    print("=" * 78)


if __name__ == "__main__":
    main()
