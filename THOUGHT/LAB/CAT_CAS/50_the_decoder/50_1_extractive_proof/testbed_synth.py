"""
testbed_synth.py - the controllable extractive testbed: a weak tone in noise.

A hidden global frequency k0 is buried so that the COHERENT SNR over the full
domain (A^2 * M / sigma^2) is comfortably detectable, while the coherent SNR over
any bounded window w << M (A^2 * w / sigma^2) is below detection. The answer is
therefore recoverable ONLY by integrating over the full domain - a structural
locality barrier, set by physics (integration length), not by a compute handicap.

This is the abelian-HSP / Fourier-sampling advantage made controllable: the
frequency is a global character of the signal, invisible to any bounded-receptive
-field or statistical-order decoder.
"""
import numpy as np


def build_signal(M, k0, snr_global=64.0, sigma=1.0, seed=0):
    """E[n] = A e^{2 pi i k0 n / M} + complex Gaussian noise(sigma).
    A chosen so coherent SNR over full M = snr_global."""
    rng = np.random.default_rng(seed)
    A = np.sqrt(snr_global * sigma ** 2 / M)
    n = np.arange(M)
    tone = A * np.exp(2j * np.pi * k0 * n / M)
    noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * (sigma / np.sqrt(2))
    return (tone + noise).astype(np.complex128), int(k0)


def signal_bank(M=4096, n_signals=40, snr_global=64.0, seed=0):
    """A bank of (E, true_k, meta). window_frac in meta sets the null receptive
    field so per-window coherent SNR = snr_global * window_frac << 1."""
    rng = np.random.default_rng(seed)
    k_lo, k_hi = M // 32, M // 4
    window_frac = 1.0 / 64
    out = []
    for i in range(n_signals):
        k0 = int(rng.integers(k_lo, k_hi))
        E, k = build_signal(M, k0, snr_global, seed=seed * 100003 + i)
        meta = {
            "M": M,
            "snr_global": snr_global,
            "snr_per_window": snr_global * window_frac,
            "window_frac": window_frac,
            "tol_bins": max(2, int(0.01 * M)),  # 1% of M; failure is SNR-driven, not resolution
        }
        out.append((E, k, meta))
    return out


def success(k_est, k_true, meta):
    return abs(int(k_est) - int(k_true)) <= meta["tol_bins"]
