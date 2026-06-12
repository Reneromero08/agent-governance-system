"""
20.10.7: Sparse Torus Winding Oracle
=====================================
Two attacks on the period-containment limit:

IDEA 1: Sparse Exponential Sampling
  Sample at n = 2^0, 2^1, ..., 2^k instead of 0,1,2,...M.
  Only O(log M) samples needed to span distance >> r.
  Compute pairwise phase coherence between all samples.
  Pairs where phase aligns (cos ~= 1) have lag = multiple of r.
  Vote for candidate periods; the true r gets most votes.

IDEA 2: Torus Winding Projection
  Map sparse samples to torus T^L via .holo compression.
  Measure winding angle (slope) on torus with high precision.
  Continued fractions on the angle -> discrete integer period r.
  Like Shor's post-QFT step: angle ~= c/r -> extract r.

Combined: exponential samples -> pairwise coherence voting
+ torus winding angle -> continued fractions -> r.
"""

import sys, time, math, random
from pathlib import Path
import numpy as np
import torch
from fractions import Fraction

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
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
# IDEA 1: Sparse Exponential Sampling + Pairwise Phase Voting
# =====================================================================

def exponential_samples(a, N, k=20):
    """Generate k+1 samples at n = 2^0, 2^1, ..., 2^k."""
    samples = []
    for i in range(k + 1):
        n = 1 << i  # 2^i
        val = pow(a, n, N)
        phase = 2.0 * math.pi * val / N
        samples.append((n, complex(math.cos(phase), math.sin(phase))))
    return samples


def pairwise_phase_voting(samples, a, N):
    """
    For each pair (i,j) of exponential samples, measure phase coherence.
    The lag delta = 2^j - 2^i. If delta is a multiple of r, phase aligns.
    Vote: each pair contributes to candidate r = delta / m for integer m.
    """
    n_pairs = len(samples) * (len(samples) - 1) // 2
    votes = {}  # candidate_r -> score

    for i in range(len(samples)):
        n_i, g_i = samples[i]
        for j in range(i + 1, len(samples)):
            n_j, g_j = samples[j]
            delta = n_j - n_i  # lag between samples
            if delta <= 1: continue

            # Phase coherence: real(g_j * conj(g_i)) = cos(phase_diff)
            phase_diff_real = (g_j * g_i.conjugate()).real

            # Score: how well does this lag align with being a period multiple?
            # cos ~= 1 -> phases aligned -> delta likely multiple of r
            # cos ~= -1 -> anti-aligned -> could be r/2 (if r even)
            # cos ~= 0 -> no relation

            if phase_diff_real > 0.9:  # strong constructive interference
                # delta is a candidate multiple of r
                # Vote for delta and its divisors as candidate periods
                for divisor in range(1, int(math.isqrt(delta)) + 1):
                    if delta % divisor == 0:
                        r_cand = divisor
                        if r_cand > 1 and r_cand < N:
                            votes[r_cand] = votes.get(r_cand, 0.0) + phase_diff_real
                        r_cand2 = delta // divisor
                        if r_cand2 > 1 and r_cand2 < N and r_cand2 != r_cand:
                            votes[r_cand2] = votes.get(r_cand2, 0.0) + phase_diff_real

            elif phase_diff_real < -0.9:  # strong destructive interference
                # delta might be r/2 (half-period gives phase flip)
                half_r = delta * 2
                for divisor in range(1, int(math.isqrt(half_r)) + 1):
                    if half_r % divisor == 0:
                        r_cand = divisor
                        if r_cand > 1 and r_cand < N:
                            votes[r_cand] = votes.get(r_cand, 0.0) + abs(phase_diff_real)
                        r_cand2 = half_r // divisor
                        if r_cand2 > 1 and r_cand2 < N and r_cand2 != r_cand:
                            votes[r_cand2] = votes.get(r_cand2, 0.0) + abs(phase_diff_real)

    # Find best candidate
    if not votes: return 0, 0.0, votes

    best_r = max(votes, key=votes.get)
    best_score = votes[best_r]
    return best_r, best_score, votes


# =====================================================================
# IDEA 2: Torus Winding Projection
# =====================================================================

def torus_winding_from_samples(phase_values, a, N, num_angle_bins=10000):
    """
    Map phase samples to torus winding angle. Use continued fractions
    to convert the continuous winding angle to discrete integer r.

    For a period-r signal on T^L, the winding angle (slope) encodes 1/r.
    With precise angle measurement, continued fractions recover r.
    """
    if len(phase_values) < 4: return 0

    # Compute pairwise phase differences for consecutive pairs
    ph = np.array(phase_values)
    diffs = np.diff(ph)  # unwrapped, may have large jumps
    diffs = np.arctan2(np.sin(diffs), np.cos(diffs))  # wrap to [-pi, pi]

    # The mean phase step per exponential jump encodes frequency info
    # For samples at n=2^i, the step in n is 2^i (not constant)
    # Instead: use the COMPLEX representation directly
    complex_vals = np.exp(1j * ph)

    # Build torus: each consecutive pair defines a 2D torus point
    if len(complex_vals) > 2:
        # Use first two principal components of the complex samples
        angles = np.angle(complex_vals)
        # The slope of angle vs log2(n) encodes 1/r
        log_n = np.array([i for i in range(len(angles))])  # log2(2^i) = i

        # Linear fit: angle(i) = slope * i + offset (mod 2pi is tricky)
        # Use circular-linear regression
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        # Fit cos = a*i + b, sin = c*i + d
        X = np.column_stack([log_n, np.ones_like(log_n)])
        cos_coef, _, _, _ = np.linalg.lstsq(X, cos_vals, rcond=None)
        sin_coef, _, _, _ = np.linalg.lstsq(X, sin_vals, rcond=None)

        # The phase slope = atan2(sin slope, cos slope) approximately
        slope_angle = math.atan2(sin_coef[0], cos_coef[0])
        if slope_angle < 0: slope_angle += 2.0 * math.pi

        # The slope angle is related to the frequency 1/r
        # slope_angle / (2*pi) = 1/r * something
        # For exponential samples, the relationship is more complex
        # But we can try continued fractions on the angle
        ratio = Fraction(slope_angle / (2.0 * math.pi)).limit_denominator(N)
        r_cand = ratio.denominator

        if r_cand > 1 and r_cand < N and pow(a, r_cand, N) == 1:
            return r_cand
        for m in range(2, 10):
            if pow(a, r_cand * m, N) == 1:
                return r_cand * m

    return 0


# =====================================================================
# EIGENBUDDY-STYLE ATTENTION on sparse samples
# =====================================================================

def eigenbuddy_sparse_attention(samples, a, N):
    """
    EIGEN_BUDDY-style attention on exponentially-spaced samples.
    Q·K^+ between all sample pairs computes phase resonance.
    The attention matrix reveals period structure via eigenvalue analysis.
    """
    S = len(samples)
    if S < 4: return 0

    # Build Q and K matrices from sample phases
    # Each sample is a point on the unit circle
    phases = torch.tensor([math.atan2(s[1].imag, s[1].real) for s in samples])
    Q = torch.polar(torch.ones(S), phases).unsqueeze(0)  # (1, S)
    K = Q.clone()

    # Hermitian attention: sr = Qr*Kr + Qi*Ki, si = Qi*Kr - Qr*Ki
    Qr, Qi = Q.real, Q.imag
    Kr, Ki = K.real, K.imag

    # Pairwise dot products (all pairs)
    sr = Qr * Kr + Qi * Ki  # real part = cos(phase_diff)
    si = Qi * Kr - Qr * Ki  # imag part = sin(phase_diff)

    # Build attention matrix: A_ij = exp(sr_ij) * exp(i*si_ij)
    # Softmax over j for each i
    attn_weights = torch.softmax(sr.squeeze(0), dim=-1)  # (S, S)

    # Eigenvalue analysis of attention matrix
    attn_sym = (attn_weights + attn_weights.T) / 2  # symmetrize
    eigenvalues = torch.linalg.eigvalsh(attn_sym)

    # The eigenvalue spectrum may reveal periodic structure
    # For a period-r signal: eigenvalues show specific pattern
    # Dominant eigenvalue gap position might encode r

    evals = eigenvalues.flip(dims=[0])
    if len(evals) > 2:
        gaps = evals[:-1] / (evals[1:] + 1e-10)
        gap_idx = torch.argmax(gaps).item() + 1
        if gap_idx > 1 and pow(a, gap_idx, N) == 1:
            return gap_idx

    return 0


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 78)
    print("20.10.7: SPARSE TORUS WINDING ORACLE")
    print("  Exponential sampling + pairwise voting + torus winding + attention")
    print("=" * 78)
    print()

    n_trials = 15
    ok_s1 = ok_s2 = ok_s3 = 0
    ok_ref = 0
    M = 2**23

    for t in range(n_trials):
        N, known_p, known_q = generate_semiprime(22)
        a = 2
        while gcd(a, N) != 1: a += 1
        t0 = time.perf_counter()

        # --- Reference: full autocorrelation ---
        r_ref = 0
        seq = [1]; curr = 1
        for _ in range(1, M): curr = (curr * a) % N; seq.append(curr)
        grating = torch.polar(torch.ones(M), 2.0 * math.pi * torch.tensor(seq, dtype=torch.float32) / N)
        spec = torch.fft.fft(grating)
        ac = torch.fft.ifft(torch.abs(spec)**2).real
        ac = ac / (ac[0] + 1e-15)
        sr = min(M//2, 500000)
        if sr > 2:
            _, mi = torch.max(torch.abs(ac[2:sr]), dim=0)
            r_ref = mi.item() + 2

        ref_ok = r_ref > 0 and pow(a, r_ref, N) == 1
        if ref_ok:
            p, q, ref_fac = shor_factor(N, a, r_ref)
            if ref_fac: ok_ref += 1

        # --- IDEA 1: Exponential sampling + pairwise voting ---
        samples = exponential_samples(a, N, k=25)  # 26 samples, spans 2^25 >> r
        best_r, best_score, votes = pairwise_phase_voting(samples, a, N)
        s1_ok = best_r > 0 and pow(a, best_r, N) == 1
        if s1_ok:
            p, q, s1_fac = shor_factor(N, a, best_r)
            if s1_fac: ok_s1 += 1

        # --- IDEA 2: Torus winding ---
        phase_vals = [math.atan2(s[1].imag, s[1].real) for s in samples]
        r_winding = torus_winding_from_samples(phase_vals, a, N)
        s2_ok = r_winding > 0
        if s2_ok:
            p, q, s2_fac = shor_factor(N, a, r_winding)
            if s2_fac: ok_s2 += 1

        # --- IDEA 3: EIGENBUDDY attention ---
        r_eigen = eigenbuddy_sparse_attention(samples, a, N)
        s3_ok = r_eigen > 0
        if s3_ok:
            p, q, s3_fac = shor_factor(N, a, r_eigen)
            if s3_fac: ok_s3 += 1

        dt = time.perf_counter() - t0
        any_new = (s1_ok or s2_ok or s3_ok) and not (ref_ok and (p if ref_ok else 0) == known_p)
        marker = " *** NEW ***" if any_new else ""

        status = []
        if ref_ok: status.append(f"ref(r={r_ref})")
        if s1_ok: status.append(f"vote(r={best_r})")
        if s2_ok: status.append(f"winding(r={r_winding})")
        if s3_ok: status.append(f"eigen(r={r_eigen})")

        print(f"  [{t+1:>2}] {N}={known_p}x{known_q}  {', '.join(status) if status else 'ALL FAILED'}  {dt:.2f}s{marker}")

    print(f"\n  Reference:  {ok_ref}/{n_trials}")
    print(f"  Voting:     {ok_s1}/{n_trials}")
    print(f"  Winding:    {ok_s2}/{n_trials}")
    print(f"  EigenBuddy: {ok_s3}/{n_trials}")
    print("=" * 78)


if __name__ == "__main__":
    main()
