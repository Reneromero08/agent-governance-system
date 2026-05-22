"""
Experiment 20.13: Catalytic Recursive Cepstrum — Mandelbrot Period Oracle
==========================================================================
Catalytic recursion: autocorrelation of autocorrelation of autocorrelation.
Each level amplifies periodic structure and suppresses decorrelated noise.
Like the Mandelbrot set, iterate the rule and structure emerges.

Physics:
  Level 0: g_n = exp(2*pi*i * a^n / N) — the raw phase grating
  Level 1: R_1 = IFFT(|FFT(g)|^2) — standard autocorrelation
  Level 2: R_2 = IFFT(|FFT(R_1)|^2) — cepstrum (autocorrelation of autocorrelation)
  Level 3: R_3 = IFFT(|FFT(R_2)|^2) — deeper recursion
  ...

  For a pure period-r signal: R_k[tau] has peak at tau = r for ALL k.
  The peak SELF-SIMILARITY across levels IS the Mandelbrot signature.
  
  Key: even if Level 1 is NOISY (short sample, S << r), the recursion
  AMPLIFIES the weak periodic signal. Each level suppresses aperiodic
  noise and concentrates energy at the period.

  Catalytic: each level's output IS the next level's input tape.
  No information destroyed — the signal is borrowed and amplified.

Test: progressively reduce S (sample count) and measure at what
recursion depth the period r becomes detectable.
"""

import sys
import time
import math
import random
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


def catalytic_cepstrum(signal, max_levels=5):
    """
    Catalytic recursive cepstrum: autocorrelation of autocorrelation...
    
    Level 0: input signal (complex, length S)
    Level k+1: |IFFT(|FFT(level_k)|^2)| — magnitude cepstrum
               or: IFFT(|FFT(level_k)|^2).real — real cepstrum
    
    Returns: list of signals at each level, and detection metrics.
    """
    levels = [signal]
    metrics = []

    for k in range(max_levels):
        prev = levels[-1]
        # FFT -> power spectrum -> IFFT -> autocorrelation
        if prev.is_complex():
            spectrum = torch.fft.fft(prev)
            power = torch.abs(spectrum) ** 2
            ac = torch.fft.ifft(power).real
        else:
            spectrum = torch.fft.fft(prev.to(torch.complex64))
            power = torch.abs(spectrum) ** 2
            ac = torch.fft.ifft(power).real

        # Normalize
        ac = ac / (ac[0] + 1e-15)

        # Peak detection metrics
        S = len(ac)
        search_range = min(S // 2, 500000)
        ac_abs = torch.abs(ac[2:search_range])
        if ac_abs.numel() > 0:
            peak_val, peak_idx_rel = torch.max(ac_abs, dim=0)
            peak_tau = peak_idx_rel.item() + 2
            # SNR: peak / mean background
            background = ac_abs.mean().item()
            snr = peak_val.item() / (background + 1e-15)
        else:
            peak_tau = 2
            peak_val = torch.tensor(0.0)
            snr = 0.0
            background = 0.0

        metrics.append({
            'level': k,
            'peak_tau': peak_tau,
            'peak_val': peak_val.item() if isinstance(peak_val, torch.Tensor) else peak_val,
            'snr': snr,
            'background': background if isinstance(background, float) else background,
        })

        levels.append(ac)

    return levels, metrics


def main():
    print("=" * 78)
    print("EXPERIMENT 20.13: CATALYTIC RECURSIVE CEPSTRUM ORACLE")
    print("  Mandelbrot principle: iterate rule -> structure emerges")
    print("  autocorrelation(autocorrelation(autocorrelation(...)))")
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

    # --- Generate full grating ---
    seq = [1]; curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N; seq.append(curr)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    grating = torch.polar(torch.ones(M, dtype=torch.float32), 2.0 * math.pi * (seq_tensor / N))
    print(f"  [+] Full grating: {M:,} elements")

    # --- Reference: full-grating autocorrelation ---
    _, metrics_full = catalytic_cepstrum(grating, max_levels=1)
    r_ref = metrics_full[0]['peak_tau']
    ref_ok, r_ref_check = verify_period(a, r_ref, N)
    ref_fac, p_ref, q_ref = False, 0, 0
    if ref_ok: p_ref, q_ref, ref_fac = shor_factor(N, a, r_ref_check)
    print(f"  [+] Reference r = {r_ref_check}, factored = {ref_fac}")
    if ref_fac: print(f"  {N} = {p_ref} x {q_ref}")
    print()

    # --- Catalytic recursion at reduced sample sizes ---
    print("=" * 78)
    print("CATALYTIC CEPSTRUM: REDUCED SAMPLES -> RECURSIVE AMPLIFICATION")
    print("=" * 78)

    # Test progressively smaller sample sizes
    sample_counts = [M, M//2, M//4, M//8, M//16, M//32, M//64, M//128, M//256, M//512, M//1024]

    print(f"  {'S':>10}  {'S/r':>10}  ", end="")
    for k in range(6):
        print(f"{'L'+str(k)+'_peak':>10}  {'L'+str(k)+'_SNR':>10}  ", end="")
    print(f"{'Found?':>8}")
    print(f"  {'-'*160}")

    for S in sample_counts:
        if S < 4:
            continue
        signal = grating[:S]
        S_ratio = S / r_ref_check if r_ref_check > 0 else 0

        levels, metrics = catalytic_cepstrum(signal, max_levels=5)

        print(f"  {S:>10,}  {S_ratio:>10.4f}  ", end="")

        found = False
        for m in metrics:
            peak = m['peak_tau']
            snr = m['snr']
            # Check if this level found the correct period
            verified, _ = verify_period(a, peak, N)
            print(f"{peak:>10}  {snr:>10.1f}  ", end="")
            if verified and not found:
                found = True

        print(f"{'YES' if found else 'no':>8}")

    print()

    # --- Deep recursion: find the "Mandelbrot depth" ---
    print("-" * 78)
    print("DEEP RECURSION: FINDING THE MANDELBROT DEPTH")
    print("  At what recursion depth does the period emerge from noise?")
    print("-" * 78)

    # Pick a challenging S (~ r/100)
    challenge_S = max(256, r_ref_check // 100)
    challenge_S = min(challenge_S, M)
    challenge_signal = grating[:challenge_S]

    print(f"  Challenge: S = {challenge_S:,} (S/r = {challenge_S / r_ref_check:.6f})")
    print(f"  Recursing until period detected or max depth...")
    print()

    # Deep recursion
    current = challenge_signal
    for depth in range(1, 21):
        if current.is_complex():
            spec = torch.fft.fft(current)
            ac = torch.fft.ifft(torch.abs(spec)**2).real
        else:
            spec = torch.fft.fft(current.to(torch.complex64))
            ac = torch.fft.ifft(torch.abs(spec)**2).real
        ac = ac / (ac[0] + 1e-15)

        search_range = min(len(ac)//2, 500000)
        ac_abs = torch.abs(ac[2:search_range])
        if ac_abs.numel() > 0:
            peak_val_d, peak_idx_d = torch.max(ac_abs, dim=0)
            peak_tau_d = peak_idx_d.item() + 2
            background_d = ac_abs.mean().item()
            snr_d = peak_val_d.item() / (background_d + 1e-15)
            verified_d, r_checked = verify_period(a, peak_tau_d, N)
        else:
            peak_tau_d = 0
            snr_d = 0.0
            verified_d = False
            r_checked = 0

        print(
            f"  Depth {depth:>3}: peak_tau = {peak_tau_d:>10}, "
            f"SNR = {snr_d:>8.1f}, verified = {str(verified_d):>5}"
            + (f", r = {r_checked}" if verified_d else "")
        )

        if verified_d:
            print(f"\n  [+] PERIOD DETECTED at recursion depth {depth}!")
            print(f"  [+] r = {r_checked}")
            print(f"  [+] S/r ratio = {challenge_S / r_checked:.6f}")
            fac_d, p_d, q_d = shor_factor(N, a, r_checked)
            if fac_d:
                print(f"  [+] FACTORED: {N} = {p_d} x {q_d}")
            break

        current = ac

    else:
        print(f"\n  [-] Period not detected within 20 recursion levels")
        print(f"  [-] S/r = {challenge_S / r_ref_check:.6f} may be below")
        print(f"      the information-theoretic detection threshold")

    print()
    t_total = time.perf_counter() - t_start
    print(f"  Total time: {t_total:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
