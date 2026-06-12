"""
Experiment 20.11: Complex-Plane Catalytic Holo Phase Compressor
================================================================
Works in the COMPLEX domain natively. The phase grating lives on the
unit circle (S^1). Each window is an L-dimensional vector of phases.
Uses complex Hermitian covariance C = E[z * z^H] with eigendecomposition.

Previous error: stacking real/imag separately flattened the circle
into a plane, doubling apparent dimension (Df/L ~ 2.0 artifact).
Complex-native SVD preserves the phase topology.

Physics: on the unit circle, only PHASE is a degree of freedom.
The complex covariance captures phase correlations directly.
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


def complex_catalytic_holo(grating, M, L, stride=1, max_samples=4096):
    """
    Complex-native holo: work with complex vectors on the unit circle.
    
    Observation matrix X: (n_samples, L) complex, each row a window.
    Complex Hermitian covariance: C = X^H * X / (n_samples - 1)
    Eigendecomposition of C (Hermitian -> real eigenvalues).
    
    Returns spectral metrics in the COMPLEX domain.
    """
    n_samples = min(max_samples, (M - L) // stride)
    if n_samples < 2:
        return None

    # Extract complex windows directly (no real/imag split)
    windows = np.zeros((n_samples, L), dtype=np.complex128)
    for i in range(n_samples):
        start = i * stride
        windows[i] = grating[start : start + L].numpy()

    # Center (complex mean subtraction)
    mean_complex = windows.mean(axis=0, keepdims=True)
    centered = windows - mean_complex

    # Complex Hermitian covariance: X^H @ X / (n-1)
    # X^H is conjugate transpose: (n_samples, L)^H @ (n_samples, L) -> (L, L)
    cov = (centered.conj().T @ centered) / (n_samples - 1)

    # Eigendecomposition of Hermitian matrix (eigenvalues are real)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    total = eigenvalues.sum()
    if total <= 1e-15:
        return None

    probs = eigenvalues / total
    cumulative = np.cumsum(probs)

    df = 1.0 / (probs**2).sum()
    k95 = int(np.searchsorted(cumulative, 0.95) + 1)
    k50 = int(np.searchsorted(cumulative, 0.50) + 1)

    # Spectral shape metrics
    log_evals = np.log(eigenvalues[: min(100, len(eigenvalues))] + 1e-15)
    if len(log_evals) >= 3:
        curvature = np.diff(log_evals, 2)
        curv_max = np.max(np.abs(curvature))
        curv_pos = np.argmax(np.abs(curvature)) + 1
    else:
        curv_max = 0.0
        curv_pos = 0

    # Gap detection (largest drop between adjacent eigenvalues)
    gaps = eigenvalues[:-1] / (eigenvalues[1:] + 1e-15)
    max_gap_idx = int(np.argmax(gaps)) + 1
    max_gap_val = float(gaps[max_gap_idx - 1])

    # Eigenvector analysis: FFT-based autocorrelation of top eigenvector
    top_evec = eigenvectors[:, 0]  # (L,) complex
    evec_t = torch.tensor(top_evec, dtype=torch.complex64)
    evec_fft = torch.fft.fft(evec_t, n=L*2)  # zero-padded for linear autocorrelation
    evec_ac = torch.fft.ifft(torch.abs(evec_fft)**2).real[:L//2]
    evec_ac = evec_ac / (evec_ac[0] + 1e-15)
    if len(evec_ac) > 2:
        evec_peak_idx = int(torch.argmax(evec_ac[1:])) + 1
        evec_peak_val = float(evec_ac[evec_peak_idx])
    else:
        evec_peak_idx = 0
        evec_peak_val = 0.0

    return {
        'L': L,
        'n': n_samples,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'cumulative': cumulative,
        'df': float(df),
        'k95': k95,
        'k50': k50,
        'df_L_ratio': float(df / L),
        'k95_L_ratio': k95 / L,
        'max_gap_idx': max_gap_idx,
        'max_gap_val': max_gap_val,
        'curv_max': float(curv_max),
        'curv_pos': curv_pos,
        'evec_peak_tau': evec_peak_idx,
        'evec_peak_val': float(evec_peak_val),
    }


def extract_period_complex(spectra, a, N, M):
    """
    Extract period from complex-native spectral topology.
    
    Multi-scale approach: track eigenvector autocorrelation peaks,
    eigenvalue gap positions, and Df saturation across scales.
    The period r manifests where these metrics converge.
    """
    if not spectra:
        return 0, "no spectra"

    candidates = set()

    # 1. Top eigenvector autocorrelation peaks
    for L, s in spectra.items():
        if s['evec_peak_tau'] > 1 and s['evec_peak_val'] > 0.3:
            # The peak in eigenvector autocorrelation may be r or r-related
            tau = s['evec_peak_tau']
            candidates.add(tau)
            candidates.add(tau * 2)
            candidates.add(M // tau if tau > 0 else 0)

    # 2. Eigenvalue gap positions (where rank drops)
    for L, s in spectra.items():
        if s['max_gap_val'] > 10.0:  # significant gap
            candidates.add(s['max_gap_idx'])

    # 3. Df saturation point:
    # Find L where Df transitions from growing linearly to sublinearly
    sorted_L = sorted(spectra.keys())
    df_vals = [spectra[L]['df'] for L in sorted_L]
    for i in range(1, len(sorted_L)):
        ddf = df_vals[i] - df_vals[i-1]
        dL = sorted_L[i] - sorted_L[i-1]
        if dL > 0 and ddf / dL < 0.5:  # sublinear growth starts
            candidates.add(sorted_L[i])
            candidates.add(sorted_L[i] * 2)
            break

    # 4. k50/L saturation: where 50% variance dimension saturates
    for L, s in spectra.items():
        if s['k50'] < L * 0.3:
            candidates.add(L * 4)
            candidates.add(L * 8)

    # Verify candidates
    for r_cand in sorted(set(int(c) for c in candidates if c > 1 and c < N)):
        if pow(a, r_cand, N) == 1:
            return r_cand, "candidate_match"
        for m in range(2, 20):
            if r_cand * m < N and pow(a, r_cand * m, N) == 1:
                return r_cand * m, "candidate_multiple"

    return 0, "no_match"


def main():
    print("=" * 78)
    print("EXPERIMENT 20.11: COMPLEX-PLANE CATALYTIC HOLO")
    print("  Native complex Hermitian covariance on the unit circle")
    print("=" * 78)
    print()

    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)

    a = 2
    while gcd(a, N) != 1:
        a += 1

    print(f"  Target: {BIT_SIZE}-bit Semiprime N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q}")
    print(f"  Base 'a': {a}")
    print()

    M_power = 23
    M = 2**M_power

    t_start = time.perf_counter()

    # --- Generate phase grating ---
    seq = [1]
    curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N
        seq.append(curr)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    phases_val = 2.0 * math.pi * (seq_tensor / N)
    grating = torch.polar(torch.ones(M, dtype=torch.float32), phases_val)
    print(f"  [+] Grating: {M:,} elements, {time.perf_counter() - t_start:.1f}s")
    print()

    # --- Reference ---
    spectrum_ref = torch.fft.fft(grating)
    power = torch.abs(spectrum_ref) ** 2
    ac = torch.fft.ifft(power).real
    ac = ac / (ac[0] + 1e-15)
    ac_abs = torch.abs(ac[2 : M // 2])
    _, max_idx_ref = torch.max(ac_abs, dim=0)
    r_ref = max_idx_ref.item() + 2
    ref_ok, r_ref_check = verify_period(a, r_ref, N)
    ref_fac, p_ref, q_ref = False, 0, 0
    if ref_ok:
        p_ref, q_ref, ref_fac = shor_factor(N, a, r_ref_check)
    print(f"  Reference: r = {r_ref_check}, factored = {ref_fac}")
    if ref_fac:
        print(f"  {N} = {p_ref} x {q_ref}")
    print()

    # --- Complex-native multi-scale holo ---
    print("-" * 78)
    print("COMPLEX HOLO: MULTI-SCALE HERMITIAN COVARIANCE")
    print("  Working on the unit circle (S^1), not flattened plane")
    print("-" * 78)

    window_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(
        f"  {'L':>6}  {'Df':>8}  {'Df/L':>8}  {'k95':>6}  {'k95/L':>8}  "
        f"{'max_gap@':>8}  {'gap':>10}  {'evec_peak@':>10}  {'evec_val':>10}"
    )
    print(f"  {'-'*90}")

    spectra = {}
    for L in window_sizes:
        max_s = 4096 if L <= 2048 else 2048
        result = complex_catalytic_holo(
            grating, M, L, stride=max(1, L // 8), max_samples=max_s
        )
        if result:
            spectra[L] = result
            print(
                f"  {L:>6}  {result['df']:>8.1f}  {result['df_L_ratio']:>8.3f}  "
                f"{result['k95']:>6}  {result['k95_L_ratio']:>8.3f}  "
                f"{result['max_gap_idx']:>8}  {result['max_gap_val']:>10.1f}  "
                f"{result['evec_peak_tau']:>10}  {result['evec_peak_val']:>10.3f}"
            )

    print()

    # --- Extract period ---
    print("-" * 78)
    print("PERIOD EXTRACTION FROM COMPLEX SPECTRAL TOPOLOGY")
    print("-" * 78)

    r_holo, method = extract_period_complex(spectra, a, N, M)
    holo_ok = r_holo > 0
    holo_fac, p_holo, q_holo = False, 0, 0
    if holo_ok:
        p_holo, q_holo, holo_fac = shor_factor(N, a, r_holo)

    print(f"  Method: {method}")
    print(f"  Extracted r = {r_holo}")
    print(f"  Verified (a^r = 1 mod N): {holo_ok}")
    print(f"  Factored: {holo_fac}")
    if holo_fac:
        print(f"  {N} = {p_holo} x {q_holo}")
    print()

    # --- Comparison: real+imag vs complex ---
    print("-" * 78)
    print("COMPARISON: COMPLEX (S^1) vs REAL+IMAG (R^2L)")
    print("-" * 78)
    print(f"  {'L':>6}  {'Df_complex':>10}  {'Df_real+imag':>12}  {'Ratio':>8}")
    print(f"  {'-'*45}")

    # Quick real+imag measurement for comparison
    for L in [64, 256, 1024, 4096]:
        if L not in spectra:
            continue
        # Real+imag SVD (old method)
        complex_result = spectra[L]
        n_samp = complex_result['n']
        windows_ri = np.zeros((n_samp, L * 2), dtype=np.float64)
        for i in range(n_samp):
            start = i * max(1, L // 8)
            w = grating[start : start + L].numpy()
            windows_ri[i, :L] = w.real
            windows_ri[i, L:] = w.imag
        centered_ri = windows_ri - windows_ri.mean(axis=0, keepdims=True)
        _, s_ri, _ = np.linalg.svd(centered_ri, full_matrices=False)
        evals_ri = s_ri**2 / max(1, n_samp - 1)
        total_ri = evals_ri.sum()
        probs_ri = evals_ri / total_ri
        df_ri = 1.0 / (probs_ri**2).sum()

        print(
            f"  {L:>6}  {complex_result['df']:>10.1f}  {df_ri:>12.1f}  "
            f"{complex_result['df'] / df_ri:>8.3f}"
        )

    print()

    # --- Final ---
    t_total = time.perf_counter() - t_start
    print("=" * 78)
    print("FINAL")
    print("=" * 78)
    print(f"  N = {N} = {known_p} x {known_q}")
    print(f"  Reference r = {r_ref_check} (autocorrelation), factored = {ref_fac}")
    print(f"  Complex holo r = {r_holo} ({method}), factored = {holo_fac}")
    print(f"  Total: {t_total:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
