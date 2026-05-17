"""
TASK 3: Flow State Phase Transition -- Theta-Gamma PAC in Engaged vs Rest

Tests the Semiotic Wave Mechanics prediction that entry into a focused,
flow-like state produces a sudden increase in theta-gamma coupling.

Real data: EEGBCI motor imagery dataset (PhysioNet/MNE).
Motor imagery requires sustained focused attention -- a flow-state proxy.
Rest periods (T0) are the low-flow control condition.

Hypothesis: Theta-gamma PAC is higher during motor imagery (engaged/flow)
than during rest. Entry into motor imagery shows a sudden PAC transition.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_sys_root = Path(__file__).resolve().parents[1]
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))

from utils import (
    compute_pac_tort,
    permutation_test,
    cohens_d,
    generate_synthetic_eeg_flow,
    compute_data_hash,
    write_json,
    make_receipt,
    check_significant,
    MNE_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SFREQ = 250.0
N_CHANNELS = 64
N_TIMES = 2500  # synthetic only

THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (20.0, 55.0)  # Below Nyquist at 160 Hz (Nyquist=80)

WINDOW_SEC = 1.5  # shorter window for 4-sec segments
STEP_SEC = 0.5


def compute_sliding_pac(
    data: np.ndarray,
    sfreq: float,
    f_phase: Tuple[float, float],
    f_amp: Tuple[float, float],
    window_sec: float = 1.5,
    step_sec: float = 0.5,
) -> np.ndarray:
    """Compute PAC in sliding windows to detect transitions."""
    n_times = data.shape[1]
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    pac_values = []
    for start in range(0, n_times - window_samples + 1, step_samples):
        window = data[:, start : start + window_samples]
        pac = compute_pac_tort(window, sfreq, f_phase, f_amp)
        pac_values.append(pac)
    return np.array(pac_values)


def detect_transition(pac_series: np.ndarray, threshold_factor: float = 3.0) -> Dict[str, Any]:
    """Detect sudden PAC increase. Requires step in middle 40% of windows."""
    n = len(pac_series)
    if n < 4:
        return {"is_sudden": False, "max_step": 0.0, "mean_step": 0.0,
                "transition_bin": -1, "n_windows": n}
    steps = np.abs(np.diff(pac_series))
    max_step = float(np.max(steps))
    mean_step = float(np.mean(steps))
    max_idx = int(np.argmax(steps))
    lo = int(n * 0.2)
    hi = int(n * 0.8)
    in_middle = lo <= max_idx <= hi
    is_sudden = (max_step > threshold_factor * mean_step and in_middle) if mean_step > 0 else False
    return {
        "is_sudden": is_sudden, "max_step": max_step, "mean_step": mean_step,
        "max_step_idx": max_idx, "transition_bin": max_idx,
        "in_expected_region": in_middle, "threshold_factor": threshold_factor,
        "n_windows": n,
    }


def load_eegbci_data(data_dir: str) -> Dict[str, Any]:
    """
    Load EEGBCI motor imagery dataset.

    Motor imagery (T1/T2) requires sustained focused attention = flow proxy.
    Rest (T0) = low-flow control.

    Returns high_flow (motor imagery) and low_flow (rest) segment arrays.
    """
    import mne

    runs = [6, 10, 14]  # Motor imagery runs for subject 1
    subject = "S001"
    base = os.path.join(data_dir, "files", "eegmmidb", "1.0.0", subject)

    high_flow = []
    low_flow = []
    metadata = {}

    for run in runs:
        edf_path = os.path.join(base, f"{subject}R{run:02d}.edf")
        if not os.path.exists(edf_path):
            print(f"  WARNING: {edf_path} not found, skipping")
            continue
        print(f"  Loading {subject}R{run:02d}.edf...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        metadata["sfreq"] = sfreq
        metadata["n_channels"] = len(raw.ch_names)

        ann = raw.annotations
        for onset_s, duration_s, description in zip(ann.onset, ann.duration, ann.description):
            t0 = int(onset_s * sfreq)
            t1 = int((onset_s + duration_s) * sfreq)
            if t0 < 0 or t1 > raw.n_times or t1 <= t0:
                continue
            segment = raw.get_data(start=t0, stop=t1).astype(np.float32)
            if description == "T0":
                low_flow.append(segment)
            else:  # T1 or T2 = motor imagery
                high_flow.append(segment)

    print(f"  Motor imagery segments: {len(high_flow)}")
    print(f"  Rest segments: {len(low_flow)}")

    if not high_flow or not low_flow:
        raise RuntimeError("No segments extracted from EEGBCI data")

    # Trim all segments to uniform length (min across both conditions)
    min_len = min(min(s.shape[1] for s in high_flow), min(s.shape[1] for s in low_flow))
    high_flow = [s[:, :min_len] for s in high_flow]
    low_flow = [s[:, :min_len] for s in low_flow]
    print(f"  Trimmed segments to uniform length: {min_len} samples ({min_len/sfreq:.2f}s)")

    result = {}
    for name, segments in [("high_flow", high_flow), ("low_flow", low_flow)]:
        arr = np.array(segments, dtype=np.float32)
        result[name] = arr
        metadata[f"n_{name}"] = arr.shape[0]
        metadata[f"n_times_{name}"] = arr.shape[2]

    result["sfreq"] = metadata["sfreq"]
    return result, metadata


def run_flow_test(
    mode: str = "synthetic",
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the flow state phase transition test."""
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results" / "task3_flow")
    os.makedirs(output_dir, exist_ok=True)

    data_hashes = {}
    params = {
        "theta_band": THETA_BAND,
        "gamma_band": GAMMA_BAND,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "mode": mode,
        "seed": seed,
    }

    print("=" * 60)
    print("TASK 3: Flow State -- Theta-Gamma PAC in Engaged vs Rest")
    print("=" * 60)

    if mode == "synthetic":
        params["sfreq"] = SFREQ
        params["n_channels"] = N_CHANNELS
        params["n_times"] = N_TIMES
        print("Generating synthetic EEG with known PAC per condition...")
        synth = generate_synthetic_eeg_flow(
            n_channels=N_CHANNELS, n_times=N_TIMES, sfreq=SFREQ,
            n_high_flow=20, n_low_flow=20, seed=seed,
        )
        high_flow_segments = synth["high_flow"]
        low_flow_segments = synth["low_flow"]
        data_hashes["high_flow"] = compute_data_hash(high_flow_segments)
        data_hashes["low_flow"] = compute_data_hash(low_flow_segments)
        srate = SFREQ
    elif mode == "real":
        if data_dir is None:
            raise ValueError("data_dir required (path to eegbci data root)")
        loaded, load_meta = load_eegbci_data(data_dir)
        high_flow_segments = loaded["high_flow"]
        low_flow_segments = loaded["low_flow"]
        data_hashes["high_flow"] = compute_data_hash(high_flow_segments)
        data_hashes["low_flow"] = compute_data_hash(low_flow_segments)
        params["dataset"] = "EEGBCI Motor Imagery (PhysioNet)"
        params.update(load_meta)
        srate = load_meta["sfreq"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_high = high_flow_segments.shape[0]
    n_low = low_flow_segments.shape[0]
    print(f"  Engaged segments: {n_high}, Rest segments: {n_low}")

    print("Computing theta-gamma PAC for each segment...")
    high_pac = []
    low_pac = []
    high_transitions = []
    low_transitions = []

    for i in range(n_high):
        segment = high_flow_segments[i]
        pac_series = compute_sliding_pac(segment, srate, THETA_BAND, GAMMA_BAND,
                                          window_sec=WINDOW_SEC, step_sec=STEP_SEC)
        high_pac.append(float(np.mean(pac_series)))
        high_transitions.append(detect_transition(pac_series))
        if i % 5 == 0:
            print(f"  Engaged {i}/{n_high}: mean PAC = {high_pac[-1]:.6f}")

    for i in range(n_low):
        segment = low_flow_segments[i]
        pac_series = compute_sliding_pac(segment, srate, THETA_BAND, GAMMA_BAND,
                                          window_sec=WINDOW_SEC, step_sec=STEP_SEC)
        low_pac.append(float(np.mean(pac_series)))
        low_transitions.append(detect_transition(pac_series))
        if i % 5 == 0:
            print(f"  Rest {i}/{n_low}: mean PAC = {low_pac[-1]:.6f}")

    high_pac = np.array(high_pac)
    low_pac = np.array(low_pac)

    # Test 3a: PAC higher in engaged vs rest
    print("\n--- Test 3a: PAC(engaged) > PAC(rest) ---")
    pac_compare = permutation_test(high_pac, low_pac, n_permutations=10000, seed=seed)
    sig_3a, p_3a = check_significant(pac_compare)
    d_3a = cohens_d(high_pac, low_pac)
    print(f"  Engaged PAC: {pac_compare['mean_a']:.6f} +/- {pac_compare['std_a']:.6f}")
    print(f"  Rest PAC:    {pac_compare['mean_b']:.6f} +/- {pac_compare['std_b']:.6f}")
    print(f"  p = {p_3a:.6f}, d = {d_3a:.3f}, PASS = {sig_3a}")

    # Test 3b: Sudden transition evidence in engaged
    print("\n--- Test 3b: Sudden PAC increase in engaged ---")
    n_sudden_high = sum(1 for t in high_transitions if t["is_sudden"])
    n_sudden_low = sum(1 for t in low_transitions if t["is_sudden"])
    frac_sudden_high = n_sudden_high / n_high if n_high > 0 else 0.0
    frac_sudden_low = n_sudden_low / n_low if n_low > 0 else 0.0
    print(f"  Engaged sudden: {n_sudden_high}/{n_high} ({frac_sudden_high:.1%})")
    print(f"  Rest sudden:    {n_sudden_low}/{n_low} ({frac_sudden_low:.1%})")

    max_steps_high = np.array([t["max_step"] for t in high_transitions])
    max_steps_low = np.array([t["max_step"] for t in low_transitions])
    step_compare = permutation_test(max_steps_high, max_steps_low, n_permutations=10000, seed=seed)
    sig_3b, p_3b = check_significant(step_compare)
    d_3b = cohens_d(max_steps_high, max_steps_low)
    print(f"  Max step engaged: {step_compare['mean_a']:.6f} +/- {step_compare['std_a']:.6f}")
    print(f"  Max step rest:    {step_compare['mean_b']:.6f} +/- {step_compare['std_b']:.6f}")
    print(f"  p = {p_3b:.6f}, d = {d_3b:.3f}, PASS = {sig_3b}")

    results = {
        "test3a_pac_high_vs_low": {"pass": sig_3a, "p_value": p_3a, "cohens_d": d_3a, "permutation_result": pac_compare},
        "test3b_sudden_transition": {"pass": sig_3b, "p_value": p_3b, "cohens_d": d_3b, "frac_sudden_high": frac_sudden_high, "frac_sudden_low": frac_sudden_low, "permutation_result": step_compare},
        "overall_pass": sig_3a and sig_3b,
    }

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"{'=' * 60}")

    receipt_path = make_receipt("task3_flow", results, params, data_hashes, output_dir)
    print(f"Receipt: {receipt_path}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 3: Flow State Phase Transition EEG Test")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run_flow_test(
        mode=args.mode, data_dir=args.data_dir,
        output_dir=args.output_dir, seed=args.seed,
    )
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
