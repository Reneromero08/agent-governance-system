"""
testbed_zeta.py - the REAL-lab extractive testbed: Riemann zeros from primes.

The lab's actual holographic decoder (Exp 34.12 / 34.8): encode the prime
distribution via the explicit-formula grating S(w) = sum_p (1/sqrt(p)) e^{-i w ln p},
sweep frequency w, and the non-trivial Riemann zeros emerge as resonance peaks of
|S(w)|. The zeros are NOT stored anywhere - they are a GLOBAL property of the whole
prime stream (the explicit formula), invisible to any bounded window of primes.

Mechanism copied from 34_zeta_eigenbasis/03_infinity_bootstrap/12_billion_prime_stream.py
and 02_holographic_sieves/8_riemann_harmonic_sieve.py (zeta_zeros used for
cross-validation ONLY - never fed to the decoder).
"""
import math
import numpy as np


def primes_upto_count(n_primes):
    """First n_primes primes (sieve). From 8_riemann_harmonic_sieve.py."""
    if n_primes < 1:
        return np.array([], dtype=np.float64)
    est = int(n_primes * (math.log(n_primes + 2) + math.log(math.log(n_primes + 2)))) + 10
    sieve = np.ones(est, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(est ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:est:i] = False
    return np.where(sieve)[0][:n_primes].astype(np.float64)


def zeta_zeros(n):
    """First n non-trivial zeta zero imaginary parts (cross-validation only)."""
    try:
        import mpmath as mp
        mp.mp.dps = 30
        return [float(mp.zetazero(k).imag) for k in range(1, n + 1)]
    except Exception:
        # fallback known table
        return [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                37.586178, 40.918719, 43.327073, 48.005151, 49.773832][:n]


def prime_grating(primes):
    """Per-prime terms amp * e^{-i w ln p}, amp = von Mangoldt weight ln(p)/sqrt(p).
    The partial sum is the prime part of -zeta'/zeta(1/2 + i w), whose POLES sit at
    the non-trivial zeros w = gamma; so |sum|^2 PEAKS at the Riemann zeros."""
    ln_p = np.log(primes)
    amp = ln_p / np.sqrt(primes)
    return amp, ln_p


def sweep_resonance(amp, ln_p, w_lo=10.0, w_hi=50.0, n_bins=6000):
    """Global coherent sweep: power[w] = |sum amp e^{-i w ln p}|^2 over ALL primes.
    Returns (freqs, power) where power PEAKS = Riemann zeros."""
    freqs = np.linspace(w_lo, w_hi, n_bins)
    power = np.empty(n_bins)
    chunk = 256
    for i in range(0, n_bins, chunk):
        wf = freqs[i:i + chunk][:, None]
        phase = -wf * ln_p[None, :]
        s = (amp[None, :] * np.exp(1j * phase)).sum(axis=1)
        power[i:i + chunk] = np.abs(s) ** 2
    return freqs, power


def find_peaks(freqs, resonance, min_rel=0.05, min_sep=1.0):
    """Local maxima of the resonance curve (= candidate zeros)."""
    r = resonance / (resonance.max() + 1e-30)
    peaks = []
    for i in range(1, len(r) - 1):
        if r[i] > r[i - 1] and r[i] > r[i + 1] and r[i] > min_rel:
            f = freqs[i]
            if not peaks or abs(f - peaks[-1][0]) > min_sep:
                peaks.append((float(f), float(r[i])))
            elif r[i] > peaks[-1][1]:
                peaks[-1] = (float(f), float(r[i]))
    peaks.sort(key=lambda x: x[0])
    return [p[0] for p in peaks]


def extract_zeros(amp, ln_p, n_bins=4000):
    """EXTRACTIVE: global sweep -> resonance peaks = Riemann zeros."""
    freqs, res = sweep_resonance(amp, ln_p, n_bins=n_bins)
    return find_peaks(freqs, res)


def score_zeros(found, true_zeros, tol=0.5):
    """Fraction of true zeros matched by some found peak within tol."""
    if not true_zeros:
        return 0.0
    hit = 0
    for z in true_zeros:
        if found and min(abs(z - f) for f in found) <= tol:
            hit += 1
    return hit / len(true_zeros)
