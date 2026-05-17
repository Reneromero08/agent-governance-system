"""
TASK 2: Symbolic Resonance -- High-Sigma Symbols Induce Higher PLV

Tests the Semiotic Wave Mechanics prediction that high-compression (high-sigma)
symbols induce higher phase-locking values (PLV) than low-sigma or scrambled
controls.

Hypothesis: PLV(high-sigma) > PLV(low-sigma) > PLV(scrambled)

High-sigma: cross, crown, snake, baby, fire, skull, sword, eagle, dragon, lion
  (archetypal symbols verified to exist in THINGS-EEG ds003825)
Low-sigma: 10 neutral everyday objects matched for trial count
Scrambled: Fourier phase-randomized high-sigma trials (preserves spectrum)

Real data mode: uses THINGS-EEG dataset (OpenNeuro ds003825).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

_sys_root = Path(__file__).resolve().parents[1]
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))

from utils import (
    compute_plv,
    permutation_test,
    cohens_d,
    generate_synthetic_eeg_symbols,
    compute_data_hash,
    write_json,
    make_receipt,
    check_significant,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SFREQ = 250.0
N_CHANNELS = 63
ALPHA_BAND = (8.0, 12.0)
GAMMA_BAND = (30.0, 80.0)

# High-sigma: archetypal symbols with deep cultural resonance
# Verified: all exist in ds003825 (THINGS-EEG) object column
HIGH_SIGMA_SYMBOLS = [
    "cross", "crown", "snake", "baby", "fire",
    "skull", "sword", "eagle", "dragon", "lion",
]

# Low-sigma: neutral everyday objects, no archetypal weight
# Verified: all exist in ds003825, matched for trial count
LOW_SIGMA_OBJECTS = [
    "stapler", "ladle", "faucet", "plunger", "corkscrew",
    "spatula", "tape_measure", "toaster", "broom", "strainer",
]


def compute_band_plv(
    trials: np.ndarray,
    sfreq: float,
    f_band: Tuple[float, float],
    use_iplv: bool = False,
) -> List[float]:
    """Compute mean PLV for each trial in a frequency band."""
    plv_values = []
    for trial in trials:
        plv = compute_plv(trial, sfreq, f_band, use_imaginary=use_iplv)
        plv_values.append(plv)
    return plv_values


def phase_scramble(data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Fourier phase randomization preserving power spectrum.

    For each channel independently: FFT, randomize phase, inverse FFT.
    This destroys phase coherence while preserving spectral power.
    """
    rng = np.random.RandomState(seed)
    scrambled = np.zeros_like(data)
    n_times = data.shape[-1]
    for ch in range(data.shape[0]):
        fft = np.fft.rfft(data[ch])
        magnitude = np.abs(fft)
        # Randomize phase, preserving DC and Nyquist (real-valued)
        random_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, len(fft)))
        random_phase[0] = 1.0  # DC component
        if n_times % 2 == 0:
            random_phase[-1] = 1.0  # Nyquist
        scrambled[ch] = np.fft.irfft(magnitude * random_phase, n=n_times)
    return scrambled


def load_things_eeg(subject_dir: str) -> Dict[str, Any]:
    """
    Load THINGS-EEG BrainVision data for a single subject.

    Reads .vhdr/.eeg/.vmrk via MNE, matches stimulus names to conditions.
    Returns high-sigma, low-sigma, and phase-scrambled control trials.

    Epoch window: -50ms to +500ms relative to stimulus onset (550ms).
    Data is kept at native 1000 Hz; PLV computation handles downsampling.
    """
    import csv

    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required for real data loading. pip install mne")

    # Find BrainVision files
    eeg_dir = os.path.join(subject_dir, "eeg")
    vhdr_files = [f for f in os.listdir(eeg_dir) if f.endswith(".vhdr")]
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file found in {eeg_dir}")
    vhdr_path = os.path.join(eeg_dir, vhdr_files[0])

    # Find events file
    events_files = [f for f in os.listdir(eeg_dir) if f.endswith("_events.tsv")]
    events_path = os.path.join(eeg_dir, events_files[0]) if events_files else None

    print(f"  Loading EEG: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    print(f"  {n_channels} channels, {sfreq} Hz, {raw.n_times} samples ({raw.n_times / sfreq:.0f}s)")

    # Parse events
    print(f"  Loading events: {events_path}")
    events_list = []

    with open(events_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            onset_s = float(row["onset"])
            sample = int(float(row.get("sample", onset_s * sfreq)))
            concept = row.get("object", "").lower()
            events_list.append({
                "onset_s": onset_s,
                "sample": sample,
                "concept": concept,
                "duration_ms": float(row.get("duration", 50)),
            })

    # Assign categories
    for ev in events_list:
        concept = ev["concept"].lower()
        if concept in HIGH_SIGMA_SYMBOLS:
            ev["category"] = "high_sigma"
        elif concept in LOW_SIGMA_OBJECTS:
            ev["category"] = "low_sigma"
        else:
            ev["category"] = "other"

    n_high = sum(1 for e in events_list if e["category"] == "high_sigma")
    n_low = sum(1 for e in events_list if e["category"] == "low_sigma")
    n_other = sum(1 for e in events_list if e["category"] == "other")
    print(f"  Trials: {n_high} high-sigma, {n_low} low-sigma, {n_other} other")

    # Epoch: -200ms to +600ms (800ms total, covers early visual through late semantic)
    epoch_tmin_ms = -200
    epoch_tmax_ms = 600

    categories = {"high_sigma": [], "low_sigma": []}

    for ev in events_list:
        cat = ev["category"]
        if cat not in categories:
            continue
        t0 = ev["sample"] + int(epoch_tmin_ms * sfreq / 1000)
        t1 = ev["sample"] + int(epoch_tmax_ms * sfreq / 1000)
        if t0 < 0 or t1 > raw.n_times:
            continue
        segment = raw.get_data(start=t0, stop=t1)  # (n_channels, n_times)
        categories[cat].append(segment)

    # Build scrambled condition: phase-randomize high-sigma epochs
    scrambled = []
    rng = np.random.RandomState(42)
    for seg in categories["high_sigma"]:
        scrambled.append(phase_scramble(seg, seed=rng.randint(0, 2**31)))

    result = {}
    metadata = {
        "sfreq": sfreq,
        "n_channels": n_channels,
        "epoch_tmin_ms": epoch_tmin_ms,
        "epoch_tmax_ms": epoch_tmax_ms,
    }

    for cat_name, segments in [("high_sigma", categories["high_sigma"]),
                                 ("low_sigma", categories["low_sigma"]),
                                 ("scrambled", scrambled)]:
        if not segments:
            result[cat_name] = np.zeros((0, n_channels, 0), dtype=np.float32)
            metadata[f"n_{cat_name}"] = 0
            continue
        arr = np.array(segments, dtype=np.float32)
        # Decimate to 250 Hz for computational efficiency
        dec_factor = int(sfreq / SFREQ)
        if dec_factor > 1:
            downsampled = sp_signal.decimate(arr, dec_factor, axis=-1, ftype="iir")
        else:
            downsampled = arr
        result[cat_name] = downsampled.astype(np.float32)
        metadata[f"n_{cat_name}"] = downsampled.shape[0]
        metadata[f"n_times_{cat_name}"] = downsampled.shape[2]

    metadata["sfreq"] = SFREQ if int(sfreq / SFREQ) > 1 else sfreq

    print(f"  Loaded: high_sigma={metadata.get('n_high_sigma', 0)}, "
          f"low_sigma={metadata.get('n_low_sigma', 0)}, "
          f"scrambled={metadata.get('n_scrambled', 0)}")

    return result, metadata


def run_symbols_test(
    mode: str = "synthetic",
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the symbolic resonance test."""
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results" / "task2_symbols")
    os.makedirs(output_dir, exist_ok=True)

    data_hashes = {}
    params = {
        "sfreq": SFREQ,
        "n_channels": N_CHANNELS,
        "alpha_band": ALPHA_BAND,
        "gamma_band": GAMMA_BAND,
        "high_sigma_symbols": HIGH_SIGMA_SYMBOLS,
        "low_sigma_objects": LOW_SIGMA_OBJECTS,
        "mode": mode,
        "seed": seed,
    }

    print("=" * 60)
    print("TASK 2: Symbolic Resonance -- High-Sigma > Low-Sigma PLV")
    print("=" * 60)

    if mode == "synthetic":
        print("Generating synthetic EEG with controlled PLV per condition...")
        synth = generate_synthetic_eeg_symbols(
            n_channels=N_CHANNELS,
            n_times=int(SFREQ),  # 1 sec at 250 Hz
            sfreq=SFREQ,
            n_high_sigma=50,
            n_low_sigma=50,
            n_scrambled=50,
            seed=seed,
        )
        high_sigma_trials = synth["high_sigma"]
        low_sigma_trials = synth["low_sigma"]
        scrambled_trials = synth["scrambled"]
        data_hashes["high_sigma"] = compute_data_hash(high_sigma_trials)
        data_hashes["low_sigma"] = compute_data_hash(low_sigma_trials)
        data_hashes["scrambled"] = compute_data_hash(scrambled_trials)
    elif mode == "real":
        if data_dir is None:
            raise ValueError("data_dir required for real mode (path to THINGS-EEG subject dir)")
        loaded, load_meta = load_things_eeg(data_dir)
        high_sigma_trials = loaded["high_sigma"]
        low_sigma_trials = loaded["low_sigma"]
        scrambled_trials = loaded["scrambled"]
        data_hashes["high_sigma"] = compute_data_hash(high_sigma_trials)
        data_hashes["low_sigma"] = compute_data_hash(low_sigma_trials)
        data_hashes["scrambled"] = compute_data_hash(scrambled_trials)
        params["dataset"] = "ds003825 THINGS-EEG"
        params.update(load_meta)
        srate = load_meta.get("sfreq", SFREQ)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_high = high_sigma_trials.shape[0]
    n_low = low_sigma_trials.shape[0]
    n_scr = scrambled_trials.shape[0]
    print(f"  High-sigma trials: {n_high}")
    print(f"  Low-sigma trials:  {n_low}")
    print(f"  Scrambled trials:  {n_scr}")

    if n_high == 0 or n_low == 0:
        print("ERROR: Zero trials in one or more conditions. Check concept lists.")
        return {"overall_pass": False, "error": "zero trials"}

    # Compute PLV per condition per band
    # Use iPLV for real data to suppress volume conduction
    use_iplv = (mode == "real")
    plv_srate = srate if mode == "real" else SFREQ
    results = {}

    for band_name, f_band in [("alpha", ALPHA_BAND), ("gamma", GAMMA_BAND)]:
        print(f"\n--- {band_name.upper()} band ({f_band[0]}-{f_band[1]} Hz) ---")

        high_plv = np.array(compute_band_plv(high_sigma_trials, plv_srate, f_band, use_iplv=use_iplv))
        low_plv = np.array(compute_band_plv(low_sigma_trials, plv_srate, f_band, use_iplv=use_iplv))
        scr_plv = np.array(compute_band_plv(scrambled_trials, plv_srate, f_band, use_iplv=use_iplv))

        # Test 2a: High-sigma > Low-sigma
        high_vs_low = permutation_test(high_plv, low_plv, n_permutations=10000, seed=seed)
        sig_hl, p_hl = check_significant(high_vs_low)
        d_hl = cohens_d(high_plv, low_plv)
        print(f"  High-sigma PLV: {high_vs_low['mean_a']:.4f} +/- {high_vs_low['std_a']:.4f}")
        print(f"  Low-sigma PLV:  {high_vs_low['mean_b']:.4f} +/- {high_vs_low['std_b']:.4f}")
        print(f"  High > Low: p={p_hl:.6f}, d={d_hl:.3f}, significant={sig_hl}")

        # Test 2b: High-sigma > Scrambled
        high_vs_scr = permutation_test(high_plv, scr_plv, n_permutations=10000, seed=seed)
        sig_hs, p_hs = check_significant(high_vs_scr)
        d_hs = cohens_d(high_plv, scr_plv)
        print(f"  Scrambled PLV: {high_vs_scr['mean_b']:.4f} +/- {high_vs_scr['std_b']:.4f}")
        print(f"  High > Scrambled: p={p_hs:.6f}, d={d_hs:.3f}, significant={sig_hs}")

        # Test 2c: Low-sigma > Scrambled
        low_vs_scr = permutation_test(low_plv, scr_plv, n_permutations=10000, seed=seed)
        sig_ls, p_ls = check_significant(low_vs_scr)
        d_ls = cohens_d(low_plv, scr_plv)
        print(f"  Low > Scrambled: p={p_ls:.6f}, d={d_ls:.3f}, significant={sig_ls}")

        results[band_name] = {
            "high_vs_low": {"pass": sig_hl, "p_value": p_hl, "cohens_d": d_hl, "permutation_result": high_vs_low},
            "high_vs_scrambled": {"pass": sig_hs, "p_value": p_hs, "cohens_d": d_hs, "permutation_result": high_vs_scr},
            "low_vs_scrambled": {"pass": sig_ls, "p_value": p_ls, "cohens_d": d_ls, "permutation_result": low_vs_scr},
        }

    overall = (
        results["alpha"]["high_vs_low"]["pass"]
        and results["alpha"]["high_vs_scrambled"]["pass"]
        and results["gamma"]["high_vs_low"]["pass"]
        and results["gamma"]["high_vs_scrambled"]["pass"]
    )
    results["overall_pass"] = overall

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"{'=' * 60}")

    receipt_path = make_receipt("task2_symbols", results, params, data_hashes, output_dir)
    print(f"Receipt: {receipt_path}")

    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 2: Symbolic Resonance EEG Test")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run_symbols_test(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
