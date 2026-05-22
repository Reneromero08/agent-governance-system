"""Q36 HARDENED: Multi-scale unfolding of implicate period from explicate grating.

Uses the catalytic phase lasing grating (20.5/20.6) and applies multi-scale
Feistel decomposition. Each scale provides an independent measurement of the
same period r with different aliasing. Combined, they should resolve r with
smaller total grating than the flat single-scale method.

Bohm connection: the grating is the explicate (observable). The period r is the
implicate (enfolded). The multi-scale Feistel UNFOLDS r by providing multiple
independent projections of it.
"""

import math, random, sys, time
from pathlib import Path
import numpy as np

OUT_DIR = Path(__file__).parent.parent / "q36_bohm" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b)
            p |= (1 << (b - 1)) | 1
            if is_prime(p): return p
    p = get_prime(bits // 2)
    q = get_prime(bits // 2)
    return p * q, p, q


def is_prime(n, k=5):
    if n <= 1 or n % 2 == 0: return n == 2 or n == 3
    s, d = 0, n - 1
    while d % 2 == 0: d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True


def gcd(a, b):
    while b: a, b = b, a % b
    return a


def generate_grating(a, N, M):
    """Generate complex phase grating: f(x) = exp(2*pi*i * a^x mod N / N)."""
    phases = np.zeros(M, dtype=np.float64)
    curr = 1
    for x in range(M):
        phases[x] = 2 * math.pi * curr / N
        curr = (curr * a) % N
    return np.exp(1j * phases)


def autocorrelation_period(grating, N):
    """Extract period via autocorrelation peak (20.6 method)."""
    fft_grating = np.fft.fft(grating)
    power = np.abs(fft_grating) ** 2
    autocorr = np.abs(np.fft.ifft(power))
    autocorr[0] = 0  # ignore DC
    peak_idx = np.argmax(autocorr)
    peak_val = autocorr[peak_idx]
    bg = np.mean(autocorr)
    return peak_idx, peak_val / max(bg, 1e-12)


def multiscale_decompose(grating, R):
    """Decompose grating into R scale channels via multi-scale Feistel pattern.
    
    Scale r: decimate by factor 2^r, measure autocorrelation at that scale.
    Each scale provides aliased period measurement: peak at M * (c / r) mod (M/2^r).
    """
    M = len(grating)
    results = []
    for r in range(R):
        step = 1 << r
        # Decimate: take every step-th element
        sub_grating = grating[::step]
        peak, snr = autocorrelation_period(sub_grating, 0)
        results.append({
            "scale": r,
            "step": step,
            "sub_M": len(sub_grating),
            "peak": int(peak),
            "snr": float(snr),
            "period_candidate": int(peak) * step if peak > 0 else 0,
        })
    return results


def main():
    print("=" * 72)
    print("Q36: MULTI-SCALE UNFOLDING OF IMPLICATE PERIOD")
    print("  Bohm: grating=explicate, period=implicate")
    print("  Multi-scale Feistel = unfolding operator")
    print("=" * 72)
    print()

    # Target a modest semiprime for quick testing
    BIT_SIZE = 16
    N, known_p, known_q = generate_semiprime(BIT_SIZE)
    a = 2
    while gcd(a, N) != 1: a += 1

    # Compute actual period for ground truth
    actual_period = 1
    curr = a
    while curr != 1:
        curr = (curr * a) % N
        actual_period += 1

    print(f"  N = {N} = {known_p} x {known_q}")
    print(f"  a = {a}")
    print(f"  Actual period r = {actual_period}")
    print()

    # Generate grating at various sizes
    grating_sizes = [2**k for k in [10, 12, 14, 16, 18]]
    R_scales = 5

    print(f"  {'M':>8} {'single-scale':>16} {'multi-scale (R={})'.format(R_scales):>22}")
    print(f"  {'M':>8} {'peak':>8} {'snr':>8}  ", end="")
    for r in range(R_scales):
        print(f"{'s'+str(r):>6}", end=" ")
    print(f"  {'combined':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}  ", end="")
    for _ in range(R_scales):
        print(f"{'-'*6}", end=" ")
    print(f"  {'-'*10}")

    for M in grating_sizes:
        grating = generate_grating(a, N, M)

        # Single-scale: standard autocorrelation
        peak_single, snr_single = autocorrelation_period(grating, N)
        single_ok = peak_single == actual_period or (
            peak_single > 0 and actual_period % peak_single == 0)

        # Multi-scale: decompose and combine
        scale_results = multiscale_decompose(grating, R_scales)

        # Combine: find period candidates that are consistent across scales
        candidates = set()
        for sr in scale_results:
            if sr["snr"] > 2.0:  # only use high-SNR scales
                pc = sr["period_candidate"]
                if pc > 0:
                    candidates.add(pc)

        multi_ok = actual_period in candidates or any(
            actual_period % c == 0 for c in candidates if c > 0)

        print(f"  {M:>8} {peak_single:>8} {snr_single:>8.1f}  ", end="")
        for sr in scale_results:
            marker = "*" if sr["snr"] > 2.0 else " "
            print(f"{marker}{sr['peak']:>5}", end=" ")
        combined = "YES" if multi_ok else "NO"
        print(f"  {'single_ok' if single_ok else 'single_fail':>10}")

    print()
    print("=" * 72)
    print("If multi-scale unfolding works:")
    print("  - Multi-scale should find period at SMALLER M than single-scale")
    print("  - Scale channels provide complementary aliasing views")
    print("  - Combined, they resolve the implicate period from the explicate grating")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
