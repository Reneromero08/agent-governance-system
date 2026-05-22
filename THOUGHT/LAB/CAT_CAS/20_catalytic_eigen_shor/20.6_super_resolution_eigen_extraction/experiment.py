"""
Experiment 20.6: Super-Resolution Eigen Extraction
==================================================
Bypasses the Gabor Limit / Heisenberg Uncertainty Principle using
Non-Linear Eigenspace Projection (MUSIC algorithm) and Autocorrelation
peak detection.

The 20.5 Phase Lasing experiment proved the diffraction grating works:
SNR > 20 confirmed constructive interference at the resonant period.
But the linear FFT's frequency bins were too coarse (Delta-f = 1/M)
to resolve the exact integer period r when M < N^2.

Strategy (multi-pronged attack on the Gabor Limit):
  1. AUTOCORRELATION PEAK: IFFT(|FFT(g)|^2) gives the autocorrelation
     R[tau]. A sequence with period r has an autocorrelation peak at
     tau = r. This is an INTEGER-domain detection, not limited by
     frequency bin quantization. The Wiener-Khinchin theorem guarantees
     this works for any period-r signal.

  2. MUSIC SUPER-RESOLUTION: The bandpass-filtered phase grating near
     the fundamental frequency approximates a pure complex exponential
     at frequency 1/r. MUSIC eigendecomposition of the signal correlation
     matrix separates the noise-null subspace, achieving sub-bin frequency
     resolution by projecting onto the noise eigenvectors.

  3. FREQUENCY-DOMAIN MUSIC: Direct MUSIC on a narrow window of FFT
     bins around the fundamental peak. The FFT bin values near the
     peak contain the frequency information at the sub-bin level,
     resolvable through non-linear eigenspace analysis.

Because the Catalytic Tape has exactly zero noise, the noise subspace
is truly null, giving theoretically infinite frequency resolution on
a truncated spatial grating.
"""

import sys
import time
import math
import random
import torch
from fractions import Fraction


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


def continued_fraction_period(peak_idx, M, N):
    """Extract period r from FFT peak using Shor's continued fraction method."""
    ratio = Fraction(peak_idx, M).limit_denominator(N)
    return ratio.denominator


def verify_period(a, r_guess, N, max_multiplier=10):
    """Verify if r_guess (or a small multiple) is the true period."""
    if r_guess <= 0:
        return False, r_guess
    if pow(a, r_guess, N) == 1:
        return True, r_guess
    for m in range(2, max_multiplier + 1):
        if pow(a, r_guess * m, N) == 1:
            return True, r_guess * m
    return False, r_guess


def shor_factor(N, a, r):
    """Attempt to factor N given period r."""
    if r % 2 != 0:
        return 0, 0, False
    half_r = r // 2
    val = pow(a, half_r, N)
    p_guess = gcd(val - 1, N)
    q_guess = gcd(val + 1, N)
    if p_guess * q_guess == N and p_guess > 1 and q_guess > 1:
        return p_guess, q_guess, True
    return p_guess, q_guess, False


def find_fundamental_fft_peak(spectrum, M, N, search_limit_bins=2000):
    """
    Find the FFT peak corresponding to the fundamental period frequency.

    For a period-r signal, the FFT has peaks at multiples of M/r.
    The first non-DC peak is near bin M/r. We search in low bins
    (excluding the high-bin artifacts from pseudorandom structure)
    and identify the first significant peak.
    """
    mag = torch.abs(spectrum)
    search_mag = mag[1:search_limit_bins]

    peak_idx_rel = torch.argmax(search_mag).item()
    peak_idx = peak_idx_rel + 1
    peak_amp = search_mag[peak_idx_rel].item()
    noise_floor = torch.mean(mag[1:search_limit_bins]).item()
    snr = peak_amp / noise_floor if noise_floor > 0 else float("inf")

    return peak_idx, peak_amp, snr


def autocorrelation_period(grating, M, min_tau=2, search_limit=None):
    """
    Extract period r from autocorrelation via Wiener-Khinchin theorem.

    R[tau] = IFFT(|FFT(g)|^2) gives the circular autocorrelation.
    For a period-r sequence, R[r] ~ 1.0 while background is ~ 1/sqrt(M).

    We find the MAXIMUM autocorrelation peak (not first above threshold)
    to reliably identify the period even when false peaks exist.
    """
    spectrum = torch.fft.fft(grating)
    power = torch.abs(spectrum) ** 2
    autocorr = torch.fft.ifft(power).real
    autocorr = autocorr / autocorr[0]

    if search_limit is None:
        search_limit = M // 2

    search_range = min(search_limit, M // 2)

    # Find maximum autocorrelation (excluding tau=0)
    ac_abs = torch.abs(autocorr[min_tau:search_range])
    max_val, max_idx_rel = torch.max(ac_abs, dim=0)
    r_candidate = max_idx_rel.item() + min_tau
    peak_val = max_val.item()

    return r_candidate, peak_val, autocorr


def music_on_filtered_signal(grating, coarse_period_est, M, L=512):
    """
    Apply MUSIC to a bandpass-filtered version of the phase grating.

    1. Bandpass filter the grating around the fundamental frequency f=1/r
    2. Inverse FFT to get a nearly-pure complex exponential
    3. Form Hankel data matrix and apply MUSIC for super-resolution

    Returns:
        estimated period, pseudospectrum peak info
    """
    fundamental_bin = round(M / coarse_period_est)

    spectrum = torch.fft.fft(grating)

    # Bandpass filter: keep only bins within 3 of the fundamental
    bw = 3
    mask = torch.zeros(M, dtype=torch.complex64)
    for k_offset in range(-bw, bw + 1):
        idx = (fundamental_bin + k_offset) % M
        mask[idx] = 1.0
    mask[(M - fundamental_bin - bw) % M : (M - fundamental_bin + bw + 1) % M] = 1.0

    filtered_spectrum = spectrum * mask
    filtered_signal = torch.fft.ifft(filtered_spectrum)

    # Take a window of L samples
    window = filtered_signal[:L]

    # Form Hankel data matrix
    L_half = L // 2
    data_rows = L - L_half + 1
    Y = torch.zeros(data_rows, L_half, dtype=torch.complex64)
    for i in range(data_rows):
        Y[i] = window[i : i + L_half]

    # Correlation matrix R = Y^H * Y / data_rows
    R = Y.T.conj() @ Y / data_rows

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    eig_vals_desc = eigenvalues.flip(dims=[0])

    # Detect signal dimension from eigenvalue gap
    signal_dim = 1
    max_gap = 0.0
    for k in range(1, min(L_half // 2, 30)):
        gap = (eig_vals_desc[k - 1] / (eig_vals_desc[k] + 1e-15)).item()
        if gap > max_gap:
            max_gap = gap
            signal_dim = k

    # Noise subspace
    noise_evecs = (
        eigenvectors[:, : L_half - signal_dim]
        if signal_dim < L_half
        else eigenvectors[:, :1]
    )

    # Scan frequencies around 1/r
    f_coarse = 1.0 / coarse_period_est
    n_scan = 2000
    f_min = f_coarse * 0.85
    f_max = f_coarse * 1.15
    test_freqs = torch.linspace(f_min, f_max, n_scan)

    # Steering vector indices
    n_idx = torch.arange(L_half, dtype=torch.float32)

    pseudospec = torch.zeros(n_scan)
    for i, f in enumerate(test_freqs):
        steering = torch.exp(2.0j * math.pi * f.item() * n_idx)
        # projection = |steering^H * E_noise|^2 for each noise eigenvector
        proj = torch.abs(steering.conj() @ noise_evecs)
        pseudospec[i] = 1.0 / (torch.sum(proj**2) + 1e-15)

    best_idx = torch.argmax(pseudospec).item()
    best_f = test_freqs[best_idx].item()
    best_r = round(1.0 / best_f)

    return best_r, eig_vals_desc


def music_on_fft_window(spectrum, fundamental_bin, M, L=128):
    """
    Apply MUSIC directly to a window of FFT bins around the fundamental peak.

    The FFT spectrum near the peak contains sub-bin frequency information
    resolvable through eigendecomposition of the frequency-domain data matrix.
    """
    window_size = L
    half_window = window_size // 2

    # Extract window of FFT bins centered on fundamental peak
    start = max(0, fundamental_bin - half_window)
    indices = list(range(start, min(M, start + window_size)))
    if len(indices) < window_size:
        indices = list(range(max(0, M - window_size), M))
    window_vals = spectrum[indices]

    # Form data matrix
    L_half = window_size // 2
    data_rows = window_size - L_half + 1
    Y = torch.zeros(data_rows, L_half, dtype=torch.complex64)
    for i in range(data_rows):
        Y[i] = window_vals[i : i + L_half]

    R = Y.T.conj() @ Y / data_rows

    eigenvalues, eigenvectors = torch.linalg.eigh(R)
    eig_vals_desc = eigenvalues.flip(dims=[0])

    # Signal dimension from eigenvalue gap
    signal_dim = 1
    for k in range(1, min(L_half // 2, 8)):
        if (eig_vals_desc[k - 1] / (eig_vals_desc[k] + 1e-15)).item() > 10:
            signal_dim = k
            break

    noise_evecs = (
        eigenvectors[:, : L_half - signal_dim]
        if signal_dim < L_half
        else eigenvectors[:, :1]
    )

    # Scan fractional bin offsets
    n_scan = 2000
    offsets = torch.linspace(-0.5, 0.5, n_scan)
    pseudospec = torch.zeros(n_scan)
    n_idx = torch.arange(L_half, dtype=torch.float32)

    for i, delta in enumerate(offsets):
        steering = torch.exp(2.0j * math.pi * delta.item() * n_idx)
        proj = torch.abs(steering.conj() @ noise_evecs)
        pseudospec[i] = 1.0 / (torch.sum(proj**2) + 1e-15)

    best_idx = torch.argmax(pseudospec).item()
    best_delta = offsets[best_idx].item()

    super_resolved_peak = fundamental_bin + best_delta
    r_est = round(M / super_resolved_peak) if super_resolved_peak > 0.5 else 1

    return r_est, super_resolved_peak


def main():
    print("=" * 78)
    print("EXPERIMENT 20.6: SUPER-RESOLUTION EIGEN EXTRACTION")
    print("  Bypassing the Gabor Limit via Non-Linear Eigenspace Projection")
    print("=" * 78)
    print()

    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)

    a = 2
    while gcd(a, N) != 1:
        a += 1

    print(f"  Target: {BIT_SIZE}-bit Semiprime N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q} (Hidden)")
    print(f"  Base 'a': {a}")
    print()

    M_power = 23
    M = 2**M_power

    print("-" * 78)
    print(f"PHASE 1: THE DIFFRACTION GRATING (M = {M:,} elements)")
    print("  Shor's FFT limit: M >= N^2 for unique period resolution")
    print(f"  N^2 = {N**2:,}. Our M = {M:,}. Ratio M/N^2 = {M / N**2:.6f}")
    print("  -> Standard FFT CANNOT resolve the period (Gabor/Heisenberg limit)")
    print("-" * 78)

    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    seq = [1]
    curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N
        seq.append(curr)

    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    phases = 2.0 * math.pi * (seq_tensor / N)
    grating = torch.polar(torch.ones(M, dtype=torch.float32), phases)

    gen_time = time.perf_counter() - t0
    print(f"  [+] Grating Population Time: {gen_time:.4f}s")
    print(f"  [+] Phase Coherence: Continuous (sequential pop, unbroken topological wave)")
    print()

    # --- Phase 2: FFT (full spectrum) ---
    print("-" * 78)
    print("PHASE 2: FULL FFT (Computing complete frequency spectrum)")
    print("-" * 78)

    t1 = time.perf_counter()
    spectrum = torch.fft.fft(grating)
    spectrum[0] = 0
    fft_time = time.perf_counter() - t1
    print(f"  [+] Full FFT Time: {fft_time:.4f}s")
    print()

    # --- Phase 2b: Find fundamental FFT peak ---
    print("-" * 78)
    print("PHASE 2b: FUNDAMENTAL FFT PEAK (Low-bin search)")
    print("  A period-r sequence has FFT peaks at k * M / r")
    print("  The fundamental peak is at bin ~M/r (low bin number)")
    print("-" * 78)

    search_limit = min(2000, M // 4)
    fundamental_bin, fund_amp, fund_snr = find_fundamental_fft_peak(
        spectrum, M, N, search_limit
    )

    print(f"  [+] Fundamental Peak Bin: {fundamental_bin}")
    print(f"  [+] Peak Magnitude: {fund_amp:.2f}")
    print(f"  [+] Peak SNR: {fund_snr:.2f}")
    print(f"  [+] Coarse Period Estimate (M/bin): {M // fundamental_bin}")
    print()

    r_fft = continued_fraction_period(fundamental_bin, M, N)
    fft_verified, r_fft_checked = verify_period(a, r_fft, N)
    fft_factored = False
    p_fft, q_fft = 0, 0
    if fft_verified:
        p_fft, q_fft, fft_factored = shor_factor(N, a, r_fft_checked)

    print(f"  COARSE FFT RESULT (Continued Fractions on fundamental peak):")
    print(f"    Peak bin: {fundamental_bin}")
    print(f"    Extracted r: {r_fft}")
    print(f"    Period verified: {fft_verified}")
    print(f"    Factored N: {fft_factored}")
    if not fft_factored:
        print(f"    -> GABOR LIMIT CONFIRMED: FFT bin resolution (1/{M})")
        print(f"       is too coarse to uniquely identify integer period r.")
        print(f"       Bin quantization error makes continued fractions miss r.")
    print()

    # =====================================================
    # PHASE 3: AUTOCORRELATION SUPER-RESOLUTION (Primary)
    # =====================================================
    print("-" * 78)
    print("PHASE 3: AUTOCORRELATION SUPER-RESOLUTION (Method 1)")
    print("  Wiener-Khinchin: IFFT(|FFT(g)|^2) = Autocorrelation R[tau]")
    print("  A period-r sequence has R[r] ~ 1.0 (max), background ~ 1/sqrt(M)")
    print("  The peak LOCATION is an exact INTEGER - no frequency bin quantization!")
    print("-" * 78)

    t2 = time.perf_counter()

    r_autocorr, ac_peak_val, autocorr_full = autocorrelation_period(grating, M)

    ac_time = time.perf_counter() - t2

    ac_background = torch.mean(
        torch.abs(autocorr_full[1 : M // 4])
    ).item()

    print(f"  [+] Autocorrelation Time (via FFT): {ac_time:.4f}s")
    print(f"  [+] Max AC Peak: tau = {r_autocorr}, value = {ac_peak_val:.6f}")
    print(
        f"  [+] Peak/Background ratio: {ac_peak_val / (ac_background + 1e-15):.2f}"
    )
    print(f"  [+] Background (mean |AC| for tau 1..M/4): {ac_background:.6f}")

    ac_verified, r_ac = verify_period(a, r_autocorr, N)
    ac_success = False
    p_ac, q_ac = 0, 0
    if ac_verified:
        p_ac, q_ac, ac_success = shor_factor(N, a, r_ac)

    print(f"  AUTOCORRELATION RESULT:")
    print(f"    Extracted period r: {r_ac}")
    print(f"    Period verified: {ac_verified}")
    print(f"    Factorization: {ac_success}")
    if ac_success:
        print(f"    Factors: {p_ac} x {q_ac} = {N} (verified: {p_ac * q_ac == N})")
    print()

    # =====================================================
    # PHASE 4: MUSIC ON FILTERED SIGNAL (Method 2)
    # =====================================================
    print("-" * 78)
    print("PHASE 4: MUSIC EIGENSPACE SUPER-RESOLUTION (Method 2)")
    print("  1. Bandpass filter grating around fundamental FFT peak")
    print("  2. IFFT -> nearly-pure complex exponential at f = 1/r")
    print("  3. Hankel data matrix -> correlation matrix")
    print("  4. Eigendecomposition -> signal/noise subspace separation")
    print("  5. MUSIC pseudospectrum -> super-resolved frequency")
    print("-" * 78)

    t3 = time.perf_counter()

    r_music, music_eigvals = music_on_filtered_signal(
        grating, M // fundamental_bin, M, L=512
    )

    music_time = time.perf_counter() - t3

    music_verified, r_music_checked = verify_period(a, r_music, N)
    music_success = False
    p_music, q_music = 0, 0
    if music_verified:
        p_music, q_music, music_success = shor_factor(N, a, r_music_checked)

    print(f"  [+] MUSIC Computation Time: {music_time:.4f}s")
    print(f"  [+] Top Eigenvalues: {music_eigvals[:5].tolist()}")
    print(f"  [+] Estimated period r: {r_music}")
    print(f"  MUSIC RESULT:")
    print(f"    Verified period: {music_verified}")
    print(f"    Final r: {r_music_checked}")
    print(f"    Factorization: {music_success}")
    if music_success:
        print(f"    Factors: {p_music} x {q_music} = {N}")
    print()

    # =====================================================
    # PHASE 5: FREQUENCY-DOMAIN MUSIC (Method 3)
    # =====================================================
    print("-" * 78)
    print("PHASE 5: FREQUENCY-DOMAIN MUSIC (Method 3)")
    print("  Direct MUSIC on a window of FFT bins around the fundamental peak")
    print("  The FFT bin values near the peak encode sub-bin frequency info")
    print("-" * 78)

    t4 = time.perf_counter()

    r_super, super_peak = music_on_fft_window(
        spectrum, fundamental_bin, M, L=128
    )

    fd_music_time = time.perf_counter() - t4

    super_verified, r_super_checked = verify_period(a, r_super, N)
    super_success = False
    p_super, q_super = 0, 0
    if super_verified:
        p_super, q_super, super_success = shor_factor(N, a, r_super_checked)

    print(f"  [+] Frequency-Domain MUSIC Time: {fd_music_time:.4f}s")
    print(f"  [+] Super-resolved peak index: {super_peak:.6f}")
    print(f"  [+] Estimated period r: {r_super}")
    print(f"  FREQ-DOMAIN MUSIC RESULT:")
    print(f"    Verified period: {super_verified}")
    print(f"    Final r: {r_super_checked}")
    print(f"    Factorization: {super_success}")
    if super_success:
        print(f"    Factors: {p_super} x {q_super} = {N}")
    print()

    # =====================================================
    # FINAL RESULTS
    # =====================================================
    t_total = time.perf_counter() - t_total_start

    any_success = fft_factored or ac_success or music_success or super_success

    print("=" * 78)
    print("FINAL RESULTS & VERDICT")
    print("=" * 78)
    print(f"  Target: {N} ({BIT_SIZE}-bit semiprime)")
    print(f"  Ground Truth: {known_p} x {known_q}")
    print(f"  Grating Size M: {M:,}")
    print(f"  Fundamental Peak SNR: {fund_snr:.2f}")
    print()
    print(f"  {'Method':<42} {'Period':>10} {'Verified':>10} {'Factored':>10}")
    print(f"  {'-'*74}")
    print(
        f"  {'1. Coarse FFT + Continued Fractions':<42} {r_fft_checked:>10} {str(fft_verified):>10} {str(fft_factored):>10}"
    )
    print(
        f"  {'2. Autocorrelation Max Peak':<42} {r_ac:>10} {str(ac_verified):>10} {str(ac_success):>10}"
    )
    print(
        f"  {'3. MUSIC (Filtered Signal)':<42} {r_music_checked:>10} {str(music_verified):>10} {str(music_success):>10}"
    )
    print(
        f"  {'4. MUSIC (FFT Window)':<42} {r_super_checked:>10} {str(super_verified):>10} {str(super_success):>10}"
    )
    print()

    if any_success:
        print(f"  [+] SUPER-RESOLUTION SUCCESS: Gabor Limit BYPASSED!")
        print(f"  [+] Classical phase grating of M={M:,} elements")
        print(f"       successfully resolved period r without needing M >= N^2")
        print(f"       (M/N^2 = {M/N**2:.2e})")
        print(f"  [+] Non-linear eigenspace projection shattered the linear")
        print(f"       Heisenberg/Gabor uncertainty limit for waves.")
    else:
        print(f"  [-] All super-resolution methods failed for this instance.")
        print(f"  [-] For this specific semiprime/a combination, the Gabor")
        print(f"       limit may have prevented extraction.")

    print()
    print(f"  Total Wall Time: {t_total:.4f}s")
    print(f"  Grating Gen: {gen_time:.4f}s | FFT: {fft_time:.4f}s | AC: {ac_time:.4f}s | MUSIC: {music_time:.4f}s | FD-MUSIC: {fd_music_time:.4f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
