"""
wrong_answer_control.py - the anti-circularity kill-shot.

Constructs encodings whose STORED VALUE STATISTICS are (near) identical but whose
true answer differs. If the extractive decoder still tracks the true answer while
every statistics-based null returns the same value for both, then the extractive
decoder is provably NOT keying on a stored statistic - it reads a global property
the statistics do not contain.

Two constructions:
  - synth matched pair : same noise realisation, weak tones at k_A != k_B.
                         Value histograms ~identical; FFT peak differs.
  - scrambled zeta     : true prime grating vs per-prime random-phase scramble.
                         Per-term amplitude distribution identical; the random
                         phases destroy the coherent resonance that carries zeros.
"""
import numpy as np
from testbed_synth import build_signal


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic + rough asymptotic p-value."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    p = float(np.exp(-2 * (en * D) ** 2))
    return D, p


def matched_pair_synth(M=4096, k_a=137, k_b=613, snr_global=64.0, seed=7):
    """Two signals sharing the SAME noise realisation, tones at k_a != k_b."""
    rng = np.random.default_rng(seed)
    sigma = 1.0
    A = np.sqrt(snr_global * sigma ** 2 / M)
    n = np.arange(M)
    noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * (sigma / np.sqrt(2))
    E_a = (A * np.exp(2j * np.pi * k_a * n / M) + noise).astype(np.complex128)
    E_b = (A * np.exp(2j * np.pi * k_b * n / M) + noise).astype(np.complex128)
    return (E_a, k_a), (E_b, k_b)


def statistical_identity(E_a, E_b, bins=24):
    """Confirm the two encodings are statistically (near) indistinguishable:
    KS on real/imag/angle + first-4 absolute moment differences."""
    parts = {}
    for name, fa, fb in [
        ("real", E_a.real, E_b.real),
        ("imag", E_a.imag, E_b.imag),
        ("angle", np.angle(E_a), np.angle(E_b)),
    ]:
        D, p = ks_2samp(fa, fb)
        parts[name] = {"ks_D": D, "ks_p": p}
    def moments(x):
        x = np.abs(x)
        return np.array([x.mean(), x.var(), ((x - x.mean()) ** 3).mean(), ((x - x.mean()) ** 4).mean()])
    dm = np.abs(moments(E_a) - moments(E_b))
    parts["max_abs_moment_diff"] = float(dm.max())
    parts["min_ks_p"] = float(min(parts[k]["ks_p"] for k in ("real", "imag", "angle")))
    return parts


def scrambled_zeta(amp, ln_p, seed=11):
    """Per-prime random fixed phase: term = amp e^{i phi_p} e^{-i w ln p}.
    Amplitude distribution (amp) unchanged; coherent zero-resonance destroyed."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, size=len(amp))
    amp_c = amp * np.exp(1j * phi)
    return amp_c, ln_p
