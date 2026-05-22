"""
Experiment 20.12: Torus-Native Catalytic Holo — Winding & Circular PCA
======================================================================
The phase grating lives on T^L (L-torus), not R^L. Each window of L
phases is a point on the L-torus. The period r manifests as:

  1. Winding vector: after r steps, trajectory closes -> winding = 0
  2. Circular statistics: Fréchet mean, wrapped phase differences
  3. Torus kernel PCA: Gaussian kernel on circular distance

Key physics: on a torus, phase wraps modulo 2pi. The complex
representation z = exp(i*theta) is the natural embedding.
The Hermitian inner product <z_i, z_j> = z_i^H * z_j is the
torus geodesic distance.

The period r is the smallest integer such that the trajectory's
winding vector w = (0, 0, ..., 0). For sub-period windows (L < r),
we measure the INCOMPLETE winding and extrapolate.
"""

import sys
import time
import math
import random
import numpy as np
import torch


def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b)
            p |= (1 << (b - 1)) | 1
            if is_prime(p):
                return p
    p = get_prime(bits // 2)
    q = get_prime(bits // 2)
    while q == p:
        q = get_prime(bits // 2)
    return p * q, p, q


def is_prime(n, k=5):
    if n <= 1 or n % 2 == 0:
        return n == 2 or n == 3
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def verify_period(a, r_guess, N):
    if r_guess <= 0:
        return False, r_guess
    if pow(a, r_guess, N) == 1:
        return True, r_guess
    for m in range(2, 11):
        if pow(a, r_guess * m, N) == 1:
            return True, r_guess * m
    return False, r_guess


def shor_factor(N, a, r):
    if r % 2 != 0:
        return 0, 0, False
    half_r = r // 2
    val = pow(a, half_r, N)
    p_guess = gcd(val - 1, N)
    q_guess = gcd(val + 1, N)
    if p_guess * q_guess == N and p_guess > 1 and q_guess > 1:
        return p_guess, q_guess, True
    return p_guess, q_guess, False


def torus_circular_distance(z_i, z_j):
    """Geodesic distance on T^L between two complex vectors on the unit torus."""
    return torch.acos(torch.clamp((z_i.conj() * z_j).real.sum() / (z_i.abs().sum() * z_j.abs().sum() + 1e-15), -1.0, 1.0))


def torus_winding_analysis(grating, M, L, stride=1, max_samples=4096):
    """
    Torus-native analysis: measure winding numbers and circular structure.

    For each window position j, compute the unwrapped cumulative phase
    from sample to sample. The winding per dimension reveals whether
    the trajectory is closing (period detected) or open (sub-period).

    Returns winding metrics and torus-aware spectrum.
    """
    n_samples = min(max_samples, (M - L) // stride)
    if n_samples < 2:
        return None

    windows = np.zeros((n_samples, L), dtype=np.complex128)
    for i in range(n_samples):
        start = i * stride
        windows[i] = grating[start : start + L].numpy()

    # --- 1. Torus circular statistics ---

    # Fréchet mean on T^L: arg of complex sum per dimension
    # For each position j in window, compute circular mean across samples
    circular_mean = np.zeros(L, dtype=np.complex128)
    for j in range(L):
        col_sum = windows[:, j].sum()
        circular_mean[j] = col_sum / (np.abs(col_sum) + 1e-15)

    # Circular variance per dimension (0 = perfectly aligned, 1 = uniform)
    circular_var = np.zeros(L)
    for j in range(L):
        R = np.abs(windows[:, j].mean())
        circular_var[j] = 1.0 - R

    mean_circular_var = float(circular_var.mean())

    # --- 2. Winding number estimation ---

    # Phase increment between consecutive windows
    # Window i: phases at positions [0, ..., L-1] at time t_i
    # Window i+1: phases at [stride, ..., stride+L-1] at time t_i + stride
    # Winding = cumulative unwrapped phase per dimension

    # Compute pairwise phase differences between consecutive windows
    if n_samples > 2:
        phase_diffs = np.zeros((n_samples - 1, L))
        for i in range(n_samples - 1):
            for j in range(L):
                d = np.angle(windows[i + 1, j] * np.conj(windows[i, j]))
                phase_diffs[i, j] = d

        # Cumulative unwrapped phase (track 2pi jumps)
        cum_phase = np.zeros((n_samples, L))
        for i in range(1, n_samples):
            cum_phase[i] = cum_phase[i - 1] + phase_diffs[i - 1]

        # Total winding per dimension
        total_winding = cum_phase[-1] / (2.0 * math.pi)
        max_winding = float(np.max(np.abs(total_winding)))
        mean_abs_winding = float(np.mean(np.abs(total_winding)))

        # Winding dispersion: std across dimensions
        winding_dispersion = float(np.std(total_winding))
    else:
        total_winding = np.zeros(L)
        max_winding = 0.0
        mean_abs_winding = 0.0
        winding_dispersion = 0.0

    # --- 3. Torus kernel matrix (circular-distance Gaussian) ---

    # Build kernel: K_ij = exp(-d_circ(z_i, z_j)^2 / (2*sigma^2))
    # where d_circ is geodesic distance on T^L
    windows_t = torch.tensor(windows, dtype=torch.complex64)

    # Complex inner product matrix: G_ij = <z_i, z_j> / (|z_i|*|z_j|)
    norms = torch.norm(windows_t, dim=1, keepdim=True)
    gram = (windows_t @ windows_t.conj().T).real / (norms @ norms.T + 1e-15)
    gram = torch.clamp(gram, -1.0, 1.0)

    # Geodesic distance: d = acos(gram)
    dist = torch.acos(gram)

    # Gaussian kernel with adaptive sigma (median distance)
    with torch.no_grad():
        triu = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        if len(triu) > 0:
            sigma = triu.median().item() * 0.5
        else:
            sigma = 0.1
    kernel = torch.exp(-dist**2 / (2.0 * sigma**2 + 1e-15))

    # Center the kernel matrix
    n = kernel.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    kernel_centered = H @ kernel @ H

    # Eigendecomposition of centered kernel (kernel PCA on torus)
    eigenvalues_k, eigenvectors_k = torch.linalg.eigh(kernel_centered)
    eigenvalues_k = eigenvalues_k.flip(dims=[0])
    eigenvectors_k = eigenvectors_k.flip(dims=[1])

    total_k = eigenvalues_k.sum().item()
    if total_k > 1e-15:
        probs_k = eigenvalues_k / total_k
        df_kernel = float(1.0 / (probs_k**2).sum())
        cum_k = torch.cumsum(probs_k, dim=0)
        k95_kernel = int(torch.searchsorted(cum_k, 0.95).item() + 1)
    else:
        df_kernel = 1.0
        k95_kernel = 1

    # --- 4. Complex covariance (reference) ---
    windows_c = windows - circular_mean
    cov = (windows_c.conj().T @ windows_c) / (n_samples - 1)
    eigenvalues_c, _ = np.linalg.eigh(cov)
    eigenvalues_c = eigenvalues_c[::-1]
    total_c = eigenvalues_c.sum()
    if total_c > 1e-15:
        probs_c = eigenvalues_c / total_c
        df_complex = float(1.0 / (probs_c**2).sum())
        cum_c = np.cumsum(probs_c)
        k95_complex = int(np.searchsorted(cum_c, 0.95) + 1)
    else:
        df_complex = 1.0
        k95_complex = 1

    return {
        'L': L,
        'n': n_samples,
        'mean_circular_var': mean_circular_var,
        'max_winding': max_winding,
        'mean_abs_winding': mean_abs_winding,
        'winding_dispersion': winding_dispersion,
        'total_winding': total_winding,
        'df_kernel': df_kernel,
        'k95_kernel': k95_kernel,
        'df_complex': df_complex,
        'k95_complex': k95_complex,
        'df_complex_L_ratio': df_complex / L,
        'eigenvalues_kernel': eigenvalues_k.numpy(),
        'circular_mean': circular_mean,
    }


def extract_period_from_winding(torus_data, a, N):
    """
    Extract period from torus winding topology.

    Key insight: for a period-r trajectory on T^L, the winding vector
    is w_j = (θ^j_r - θ^j_0) / 2π. Since θ^j_r = θ^j_0 + 2π*(a^r - 1)*a^j/N,
    and a^r ≡ 1 mod N: w_j = 0 mod 1 for all j. The winding is ZERO.

    For sub-period windows (L < r), the winding is non-zero. The rate
    of winding per step encodes information about r:
    w_j / samples ≈ (a^j * (a^stride - 1)) / N mod 1

    From this, we can estimate components of r.
    """
    if not torus_data:
        return 0, "no_data"

    candidates = set()

    for L, td in torus_data.items():
        # Winding dispersion -> if trajectories are aligned, winding is low
        if td['winding_dispersion'] < 0.5 and td['max_winding'] < 1.0:
            candidates.add(L)
            candidates.add(L * 2)

        # Df kernel saturation
        if td['df_kernel'] < L * 0.5:
            candidates.add(L * 4)
            candidates.add(L * 8)

        # Df complex saturation
        if td['df_complex'] < L * 0.5:
            candidates.add(L * 3)
            candidates.add(L * 6)

    # Also use the total winding vector to estimate r
    if torus_data:
        largest_L = max(torus_data.keys())
        tw = torus_data[largest_L]['total_winding']
        if np.max(np.abs(tw)) > 1e-6:
            # Average winding per dimension
            avg_winding = float(np.mean(tw))
            if abs(avg_winding) > 1e-10:
                n_samples = torus_data[largest_L]['n']
                stride = 1  # approximate
                r_est = int(abs(n_samples * stride / avg_winding))
                candidates.add(r_est)
                candidates.add(r_est // 2)
                candidates.add(r_est * 2)

    for r_cand in sorted(set(int(c) for c in candidates if 1 < c < N)):
        if pow(a, r_cand, N) == 1:
            return r_cand, "winding_match"
        for m in range(2, 20):
            if r_cand * m < N and pow(a, r_cand * m, N) == 1:
                return r_cand * m, "winding_multiple"

    return 0, "no_match"


def main():
    print("=" * 78)
    print("EXPERIMENT 20.12: TORUS-NATIVE CATALYTIC HOLO")
    print("  Winding numbers, circular statistics, torus kernel PCA")
    print("=" * 78)
    print()

    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)
    a = 2
    while gcd(a, N) != 1:
        a += 1

    print(f"  Target: {BIT_SIZE}-bit N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q}")
    print(f"  Base 'a': {a}")
    print()

    M_power = 23
    M = 2**M_power
    t_start = time.perf_counter()

    # --- Generate phase grating ---
    seq = [1]; curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N; seq.append(curr)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    grating = torch.polar(torch.ones(M, dtype=torch.float32), 2.0 * math.pi * (seq_tensor / N))
    print(f"  [+] Grating: {M:,} elements, {time.perf_counter() - t_start:.1f}s\n")

    # --- Reference ---
    spectrum_ref = torch.fft.fft(grating)
    ac = torch.fft.ifft(torch.abs(spectrum_ref)**2).real
    ac = ac / (ac[0] + 1e-15)
    _, max_idx_ref = torch.max(torch.abs(ac[2 : M//2]), dim=0)
    r_ref = max_idx_ref.item() + 2
    ref_ok, r_ref_check = verify_period(a, r_ref, N)
    ref_fac, p_ref, q_ref = False, 0, 0
    if ref_ok: p_ref, q_ref, ref_fac = shor_factor(N, a, r_ref_check)
    print(f"  Reference: r = {r_ref_check}, factored = {ref_fac}")
    if ref_fac: print(f"  {N} = {p_ref} x {q_ref}")
    print()

    # --- Torus-native multi-scale analysis ---
    print("=" * 78)
    print("TORUS ANALYSIS: WINDING + CIRCULAR STATISTICS + KERNEL PCA")
    print("=" * 78)

    window_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    header = (
        f"  {'L':>6}  {'circ_var':>10}  {'max_wind':>10}  {'mean_wind':>10}  "
        f"{'wind_disp':>10}  {'Df_kern':>8}  {'k95_kern':>8}  {'Df_comp':>8}  {'Df_comp/L':>10}"
    )
    print(header)
    print(f"  {'-'*95}")

    torus_data = {}
    for L in window_sizes:
        max_s = 4096 if L <= 2048 else 2048
        td = torus_winding_analysis(grating, M, L, stride=max(1, L//8), max_samples=max_s)
        if td:
            torus_data[L] = td
            print(
                f"  {L:>6}  {td['mean_circular_var']:>10.4f}  "
                f"{td['max_winding']:>10.4f}  {td['mean_abs_winding']:>10.4f}  "
                f"{td['winding_dispersion']:>10.4f}  {td['df_kernel']:>8.1f}  "
                f"{td['k95_kernel']:>8}  {td['df_complex']:>8.1f}  "
                f"{td['df_complex_L_ratio']:>10.3f}"
            )

    print()

    # --- Extract period ---
    print("-" * 78)
    print("PERIOD EXTRACTION FROM TORUS TOPOLOGY")
    print("-" * 78)
    r_torus, method = extract_period_from_winding(torus_data, a, N)
    torus_ok = r_torus > 0
    torus_fac, p_t, q_t = False, 0, 0
    if torus_ok: p_t, q_t, torus_fac = shor_factor(N, a, r_torus)
    print(f"  Method: {method}")
    print(f"  Extracted r = {r_torus}, verified = {torus_ok}, factored = {torus_fac}")
    if torus_fac: print(f"  {N} = {p_t} x {q_t}")
    print()

    # --- Kernel vs Complex comparison ---
    print("-" * 78)
    print("TORUS KERNEL PCA vs COMPLEX COVARIANCE")
    print("-" * 78)
    print(f"  {'L':>6}  {'Df_kernel':>10}  {'Df_complex':>10}  {'Ratio':>8}")
    print(f"  {'-'*38}")
    for L, td in sorted(torus_data.items()):
        ratio = td['df_kernel'] / max(td['df_complex'], 1)
        print(f"  {L:>6}  {td['df_kernel']:>10.1f}  {td['df_complex']:>10.1f}  {ratio:>8.3f}")
    print()

    # --- Final ---
    t_total = time.perf_counter() - t_start
    print("=" * 78)
    print(f"  N = {N} = {known_p} x {known_q}")
    print(f"  Reference r = {r_ref_check}, factored = {ref_fac}")
    print(f"  Torus r = {r_torus} ({method}), factored = {torus_fac}")
    print(f"  Total: {t_total:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()