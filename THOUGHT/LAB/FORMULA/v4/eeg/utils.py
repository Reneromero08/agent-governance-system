"""
EEG Phase Coherence Utilities for v4 Wave Mechanics Validation.

Computes PLV (Phase Locking Value), PAC (Phase-Amplitude Coupling),
bandpass filtering, and permutation testing following the Semiotic
Wave Mechanics framework.

PLV measures phase coherence across channels -- the neural correlate
of semiotic resonance R. PAC captures cross-frequency coupling --
the neural correlate of meaningful signal propagation across scales.

All functions are deterministic given fixed seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE-Python not available. Synthetic data mode only.")

# Suppress only known-harmless scipy filter design warnings.
# All other warnings (overflow, invalid value) must not be silenced.
warnings.filterwarnings(
    "ignore", message=".*Badly conditioned filter.*", category=UserWarning
)


# ---------------------------------------------------------------------------
# Core signal processing
# ---------------------------------------------------------------------------

def compute_itc(
    trials: np.ndarray,
    sfreq: float,
    f_band: Tuple[float, float],
) -> np.ndarray:
    """
    Inter-Trial Coherence (ITC): phase consistency across trials.

    ITC(t) = |1/N * sum_n exp(i * phase_n(t))|

    Measures how consistent the phase is across trials at each time point.
    ITC spikes indicate phase reset events (P300, Eureka moments).
    High ITC near 1.0 means all trials have the same phase (reset).
    Low ITC near 0 means random phase across trials.

    Args:
        trials: shape (n_trials, n_channels, n_times)
        sfreq: sampling frequency
        f_band: frequency band for filtering

    Returns:
        ITC array of shape (n_channels, n_times)
    """
    n_trials, n_channels, n_times = trials.shape
    # Bandpass filter each trial
    filtered = np.zeros_like(trials)
    for i in range(n_trials):
        filtered[i] = bandpass_filter(trials[i], sfreq, f_band[0], f_band[1])
    # Hilbert phase
    phases = np.angle(signal.hilbert(filtered, axis=-1))
    # ITC: |mean of complex exponentials| across trials
    complex_sum = np.mean(np.exp(1j * phases), axis=0)
    itc = np.abs(complex_sum)
    return itc  # (n_channels, n_times)

def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase bandpass filter using Butterworth (IIR) design.

    Uses sosfiltfilt. Falls back to order=2 for signals shorter than
    3x the default filter order to avoid padlen errors.

    Args:
        data: shape (n_channels, n_times) or (n_times,)
        sfreq: sampling frequency in Hz
        l_freq: low cutoff in Hz (must be > 0)
        h_freq: high cutoff in Hz (must be < sfreq/2)
        order: Butterworth filter order (default 4)

    Returns:
        Filtered data, same shape as input
    """
    nyq = sfreq / 2.0
    # Clamp to avoid numerical issues. Use absolute frequency floors (0.5 Hz min)
    l_norm = max(0.5, l_freq) / nyq
    h_norm = min(h_freq, nyq - 0.5) / nyq
    l_norm = max(0.005, min(l_norm, 0.995))
    h_norm = max(0.005, min(h_norm, 0.995))
    if l_freq < 0.5 or h_freq > nyq - 0.5:
        warnings.warn(
            f"bandpass_filter: l_freq={l_freq} clamped to {l_norm * nyq:.1f} Hz, "
            f"h_freq={h_freq} clamped to {h_norm * nyq:.1f} Hz",
            stacklevel=2,
        )
    if l_norm >= h_norm:
        raise ValueError(
            f"bandpass_filter: l_freq={l_freq} >= h_freq={h_freq}. "
            f"Low cutoff must be less than high cutoff."
        )

    # Auto-reduce order for short signals to avoid padlen overflow
    n_times = data.shape[-1] if data.ndim > 1 else len(data)
    min_length = 3 * (2 * order + 1)  # approximate padlen
    effective_order = order
    while effective_order > 1 and n_times <= 3 * (2 * effective_order + 1):
        effective_order -= 1

    sos = signal.butter(effective_order, [l_norm, h_norm], btype="band", output="sos")
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)
    return signal.sosfiltfilt(sos, data, axis=-1)


def _hilbert_phase(data: np.ndarray) -> np.ndarray:
    """Compute instantaneous phase via Hilbert transform."""
    analytic = signal.hilbert(data, axis=-1)
    return np.angle(analytic)


def _hilbert_envelope(data: np.ndarray) -> np.ndarray:
    """Compute instantaneous amplitude via Hilbert transform."""
    analytic = signal.hilbert(data, axis=-1)
    return np.abs(analytic)


def compute_plv(
    data: np.ndarray,
    sfreq: float,
    f_band: Tuple[float, float],
    channel_pairs: Optional[List[Tuple[int, int]]] = None,
    use_imaginary: bool = False,
) -> float:
    """
    Compute mean Phase Locking Value across channels.

    PLV = |<exp(i * delta_theta)>| averaged over time and channel pairs.

    If use_imaginary=True, computes iPLV = |Im(<exp(i * delta_theta)>)|,
    which removes volume conduction artifacts (zero-lag phase coupling).

    Args:
        data: shape (n_channels, n_times) -- one epoch/trial
        sfreq: sampling frequency in Hz
        f_band: (low_freq, high_freq) for bandpass
        channel_pairs: list of (ch_i, ch_j) or None for all pairs
        use_imaginary: if True, use imaginary PLV to suppress volume conduction

    Returns:
        Mean PLV (or iPLV) across all channel pairs
    """
    n_channels = data.shape[0]
    filtered = bandpass_filter(data, sfreq, f_band[0], f_band[1])
    phases = _hilbert_phase(filtered)

    if channel_pairs is None:
        pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    else:
        pairs = channel_pairs

    if not pairs:
        return float("nan")

    plv_values = []
    for i, j in pairs:
        delta_phase = phases[i] - phases[j]
        complex_plv = np.mean(np.exp(1j * delta_phase))
        if use_imaginary:
            plv = np.abs(np.imag(complex_plv))  # iPLV: imaginary part only
        else:
            plv = np.abs(complex_plv)
        plv_values.append(plv)

    return float(np.mean(plv_values))


def compute_global_plv(
    data: np.ndarray,
    sfreq: float,
    f_band: Tuple[float, float],
    window_samples: Optional[int] = None,
    step_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Compute sliding-window global PLV.

    Args:
        data: shape (n_channels, n_times)
        sfreq: sampling frequency
        f_band: (low, high) freq band
        window_samples: window size in samples (default: sfreq * 2)
        step_samples: step size (default: window_samples // 4)

    Returns:
        Array of PLV values per window
    """
    n_times = data.shape[1]
    if window_samples is None:
        window_samples = int(sfreq * 2.0)
    if step_samples is None:
        step_samples = max(1, window_samples // 4)

    plv_series = []
    for start in range(0, n_times - window_samples + 1, step_samples):
        window = data[:, start : start + window_samples]
        plv = compute_plv(window, sfreq, f_band)
        plv_series.append(plv)

    return np.array(plv_series)


def compute_pac_tort(
    data: np.ndarray,
    sfreq: float,
    f_phase: Tuple[float, float],
    f_amp: Tuple[float, float],
    n_bins: int = 18,
) -> float:
    """
    Compute Phase-Amplitude Coupling (PAC) using the Tort method.

    1. Filter data into phase and amplitude bands
    2. Extract phase of low-freq and amplitude envelope of high-freq
    3. Bin amplitude by phase, normalize
    4. PAC = KL divergence of amplitude distribution from uniform

    Args:
        data: shape (n_channels, n_times) -- one segment
        sfreq: sampling frequency
        f_phase: (low, high) for phase-providing band (e.g., theta)
        f_amp: (low, high) for amplitude band (e.g., gamma)
        n_bins: number of phase bins

    Returns:
        PAC value (0 = no coupling, higher = more coupling)
    """
    n_channels = data.shape[0]
    pac_per_channel = []

    for ch in range(n_channels):
        ch_data = data[ch]

        # Phase signal (low frequency)
        phase_signal = bandpass_filter(ch_data, sfreq, f_phase[0], f_phase[1])
        phase_angles = _hilbert_phase(phase_signal)

        # Amplitude signal (high frequency)
        amp_signal = bandpass_filter(ch_data, sfreq, f_amp[0], f_amp[1])
        amp_envelope = _hilbert_envelope(amp_signal)

        # Bin amplitude by phase
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        amp_by_bin = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (phase_angles >= bins[b]) & (phase_angles < bins[b + 1])
            if np.any(mask):
                amp_by_bin[b] = np.mean(amp_envelope[mask])

        # Normalize to probability distribution
        amp_sum = amp_by_bin.sum()
        if amp_sum > 0:
            p = amp_by_bin / amp_sum
        else:
            p = np.ones(n_bins) / n_bins

        # KL divergence from uniform: sum(p * log(p / uniform))
        uniform = np.ones(n_bins) / n_bins
        kl = np.sum(p * np.log(p / uniform + 1e-12))
        # Normalize: max KL = log(n_bins)
        pac = kl / np.log(n_bins)
        pac_per_channel.append(pac)

    return float(np.mean(pac_per_channel))


def compute_participation_ratio(data: np.ndarray) -> float:
    """
    Compute fractal depth proxy D_f = (sum lambda)^2 / sum(lambda^2).

    Measures effective dimensionality. Higher = more dimensions participate.
    """
    if data.ndim == 1:
        return 1.0
    centered = data - data.mean(axis=0, keepdims=True)
    n_samples, n_features = centered.shape
    if n_samples < n_features:
        gram = centered @ centered.T
        eigenvalues = np.linalg.eigvalsh(gram)
    else:
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            return 1.0
        eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0
    return float((eigenvalues.sum() ** 2) / (eigenvalues**2).sum())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Two-tailed permutation test for difference in means.

    Args:
        group_a: values for condition A
        group_b: values for condition B
        n_permutations: number of permutations
        seed: random seed for reproducibility

    Returns:
        dict with observed_diff, p_value, n_permutations
    """
    rng = np.random.RandomState(seed)
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "mean_a": float(np.mean(group_a)),
        "mean_b": float(np.mean(group_b)),
        "std_a": float(np.std(group_a, ddof=1)),
        "std_b": float(np.std(group_b, ddof=1)),
        "n_a": int(n_a),
        "n_b": int(len(group_b)),
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_synthetic_eeg_eureka(
    n_channels: int = 64,
    n_times: int = 1250,  # 5 sec at 250 Hz
    sfreq: float = 250.0,
    n_insight: int = 30,
    n_analytic: int = 30,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic EEG with explicit PLV control.

    Design:
    - Pre-solution (0-3s): all trials have independent noise per channel = low PLV
    - Post-solution for insight (3-5s): coherent 40 Hz gamma burst across channels = high PLV
    - Post-solution for analytic (3-5s): independent noise only = low PLV

    The burst is Gaussian-shaped, centered at 3.2s, sigma=100ms.
    This gives a clear PLV spike in the post window for insight trials only.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_times) / sfreq
    burst_center = int(3.2 * sfreq)
    burst_sigma = int(0.10 * sfreq)  # 100 ms

    def _make_trial(is_insight: bool) -> np.ndarray:
        data = np.zeros((n_channels, n_times), dtype=np.float32)

        # Pre: independent noise per channel (mimics 1/f-ish EEG)
        for ch in range(n_channels):
            # Colored-ish noise via filtered white noise + low-freq drift
            base = rng.randn(n_times) * 0.3
            # Add low-freq drift with channel-specific random phase
            drift_freq = rng.uniform(2, 6)
            drift = np.sin(2 * np.pi * drift_freq * t + rng.uniform(0, 2 * np.pi)) * 0.8
            data[ch] = base + drift

        if is_insight:
            # Post (after 3.0s): add coherent gamma burst at 40 Hz
            ref_phase = rng.uniform(0, 2 * np.pi)
            env = np.exp(-0.5 * ((np.arange(n_times) - burst_center) / burst_sigma) ** 2)
            for ch in range(n_channels):
                # Tiny per-channel phase jitter for realism
                jitter = rng.normal(0, 0.1)
                gamma = np.sin(2 * np.pi * 40.0 * t + ref_phase + jitter) * 4.0
                data[ch] += gamma * env
        else:
            # Post: flat independent gamma noise throughout (no burst)
            for ch in range(n_channels):
                gamma = np.sin(2 * np.pi * 40.0 * t + rng.uniform(0, 2 * np.pi)) * 0.3
                data[ch] += gamma

        return data

    insight_data = np.array([_make_trial(True) for _ in range(n_insight)])
    analytic_data = np.array([_make_trial(False) for _ in range(n_analytic)])
    return {"insight": insight_data, "analytic": analytic_data}


def generate_synthetic_eeg_symbols(
    n_channels: int = 63,
    n_times: int = 250,
    sfreq: float = 250.0,
    n_high_sigma: int = 50,
    n_low_sigma: int = 50,
    n_scrambled: int = 50,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic EEG for symbol resonance test.

    High-sigma symbols induce higher PLV in alpha/gamma bands.
    Low-sigma symbols have lower PLV.
    Scrambled controls have minimal PLV.

    Returns:
        dict with 'high_sigma', 'low_sigma', 'scrambled' arrays
        each of shape (n_trials, n_channels, n_times)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_times) / sfreq

    def _make_trial(plv_strength: float) -> np.ndarray:
        """Higher plv_strength = more phase-locked channels."""
        data = np.zeros((n_channels, n_times), dtype=np.float32)
        ref_alpha = rng.uniform(0, 2 * np.pi)
        ref_gamma = rng.uniform(0, 2 * np.pi)
        for ch in range(n_channels):
            # Alpha (10 Hz)
            alpha_phase = ref_alpha + rng.normal(0, (1.0 - plv_strength) * np.pi)
            alpha = np.sin(2 * np.pi * 10.0 * t + alpha_phase) * 1.5
            # Gamma (40 Hz)
            gamma_phase = ref_gamma + rng.normal(0, (1.0 - plv_strength) * np.pi)
            gamma = np.sin(2 * np.pi * 40.0 * t + gamma_phase) * 1.0
            # Noise
            noise = rng.randn(n_times) * 0.5
            data[ch] = alpha + gamma + noise
        return data

    high_sigma = np.array([_make_trial(0.85) for _ in range(n_high_sigma)])
    low_sigma = np.array([_make_trial(0.40) for _ in range(n_low_sigma)])
    scrambled = np.array([_make_trial(0.10) for _ in range(n_scrambled)])
    return {"high_sigma": high_sigma, "low_sigma": low_sigma, "scrambled": scrambled}


def generate_synthetic_eeg_flow(
    n_channels: int = 64,
    n_times: int = 2500,  # 10 sec at 250 Hz
    sfreq: float = 250.0,
    n_high_flow: int = 20,
    n_low_flow: int = 20,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic EEG with known PAC properties and transitions.

    High-flow segments: theta-gamma PAC jumps from ~0.002 (non-flow) to
    ~0.015 (flow) at transition_idx (40% through segment). This creates
    a detectable sudden PAC increase.

    Low-flow segments: uniform low PAC throughout (no transition).

    Returns:
        dict with 'high_flow' and 'low_flow' arrays
        each of shape (n_segments, n_channels, n_times)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_times) / sfreq
    transition_idx = int(n_times * 0.4)

    def _make_segment(pac_strength: float, has_transition: bool) -> np.ndarray:
        data = np.zeros((n_channels, n_times), dtype=np.float32)
        ref_theta_phase = rng.uniform(0, 2 * np.pi)
        for ch in range(n_channels):
            # Theta (6 Hz) - phase provider, present throughout
            theta_phase = ref_theta_phase + rng.normal(0, 0.2)
            theta = np.sin(2 * np.pi * 6.0 * t + theta_phase) * 1.0

            # Gamma (40 Hz) - amplitude modulation
            gamma_phase = rng.uniform(0, 2 * np.pi)

            if has_transition:
                # Pre-transition: weak PAC (gamma amp independent of theta)
                # Post-transition: strong PAC (gamma amp locked to theta peak)
                pac_profile = np.zeros(n_times, dtype=np.float32)
                pac_profile[:transition_idx] = 0.05  # weak coupling
                pac_profile[transition_idx:] = pac_strength  # strong coupling
                gamma_amp = 0.2 + 0.8 * pac_profile * (0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t + theta_phase))
            else:
                # Uniform weak PAC throughout
                gamma_amp = 0.3 + 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t + theta_phase))

            gamma = gamma_amp * np.sin(2 * np.pi * 40.0 * t + gamma_phase)

            noise = rng.randn(n_times) * 0.3
            data[ch] = (theta + gamma + noise).astype(np.float32)
        return data

    high_flow = np.array([_make_segment(0.8, True) for _ in range(n_high_flow)])
    low_flow = np.array([_make_segment(0.1, False) for _ in range(n_low_flow)])
    return {"high_flow": high_flow, "low_flow": low_flow}


# ---------------------------------------------------------------------------
# Data hashing and receipts
# ---------------------------------------------------------------------------

def compute_data_hash(data: np.ndarray) -> str:
    """SHA256 of numpy array bytes."""
    return hashlib.sha256(data.tobytes()).hexdigest().upper()


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Atomic JSON write."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_receipt(
    test_name: str,
    results: Dict[str, Any],
    params: Dict[str, Any],
    data_hashes: Dict[str, str],
    output_dir: str,
) -> str:
    """Write a JSON receipt and return the path."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    receipt_path = os.path.join(output_dir, f"{test_name}_receipt_{ts}.json")
    receipt = {
        "test": test_name,
        "timestamp_utc": ts,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "params": params,
        "data_hashes": data_hashes,
        "results": results,
    }
    write_json(receipt_path, receipt)
    return receipt_path


def check_significant(
    perm_result: Dict[str, float],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Check if permutation test result is significant."""
    p = perm_result["p_value"]
    return p < alpha, p
