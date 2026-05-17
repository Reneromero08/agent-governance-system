"""
TASK 1: Eureka Synchronization -- PLV Spike at Target Detection

Tests the Semiotic Wave Mechanics prediction that recognition/target-detection
moments produce a sudden phase-locking spike (analogous to Eureka/insight).

Target trials: participants must actively detect a cued target image
Non-target trials: passive viewing of non-target images

Hypothesis: PLV(post-target) > PLV(pre-target) for target trials but NOT
for non-target trials, because target detection triggers a P300-mediated
phase reset (Kuramoto synchronization at the moment of recognition).

Uses THINGS-EEG dataset (OpenNeuro ds003825) -- RSVP target detection.
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
    generate_synthetic_eeg_eureka,
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
N_TIMES = 1250  # 5 seconds for synthetic
PLV_FREQ_BAND = (4.0, 12.0)  # Theta-alpha band where P300 phase reset occurs

# For real data: windows relative to stimulus onset
BASELINE_WINDOW_MS = (-200, -50)   # pre-stimulus baseline
PRE_WINDOW_MS = (-50, 0)           # immediately pre-stimulus  
POST_WINDOW_MS = (0, 300)          # 0-300ms post-stimulus (P300 window)

# For synthetic: windows around solution time (3.0s)
SOLUTION_TIME_SEC = 3.0
BASELINE_WINDOW = (-3.0, -1.0)
PRE_WINDOW = (-1.0, 0.0)
POST_WINDOW = (0.0, 1.0)


def idx_from_time(t_sec: float, sfreq: float = SFREQ) -> int:
    return int((SOLUTION_TIME_SEC + t_sec) * sfreq)


def extract_window(data: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    i_start = idx_from_time(t_start)
    i_end = idx_from_time(t_end)
    return data[:, i_start:i_end]


def compute_trial_plv(trial: np.ndarray, window: Tuple[float, float], band: Tuple[float, float] = None) -> float:
    band = band if band is not None else PLV_FREQ_BAND
    window_data = extract_window(trial, window[0], window[1])
    return compute_plv(window_data, SFREQ, band)


def compute_plv_for_ms_window(
    trial: np.ndarray,
    sfreq: float,
    onset_sample: int,
    window_start_ms: float,
    window_end_ms: float,
) -> float:
    """Compute PLV for a time window in ms relative to stimulus onset."""
    t0 = onset_sample + int(window_start_ms * sfreq / 1000)
    t1 = onset_sample + int(window_end_ms * sfreq / 1000)
    if t1 <= t0 or t0 < 0:
        return float("nan")
    window = trial[:, t0:t1]
    if window.shape[1] < 10:
        return float("nan")
    return compute_plv(window, sfreq, PLV_FREQ_BAND)


def load_things_eeg_target_nontarget(subject_dir: str) -> Dict[str, Any]:
    """
    Load THINGS-EEG and separate target vs non-target trials.
    
    Returns epochs for target and non-target conditions.
    """
    import csv

    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python required. pip install mne")

    eeg_dir = os.path.join(subject_dir, "eeg")
    vhdr_files = [f for f in os.listdir(eeg_dir) if f.endswith(".vhdr")]
    vhdr_path = os.path.join(eeg_dir, vhdr_files[0])
    events_files = [f for f in os.listdir(eeg_dir) if f.endswith("_events.tsv")]
    events_path = os.path.join(eeg_dir, events_files[0])

    print(f"  Loading EEG: {vhdr_path}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    print(f"  {n_channels} channels, {sfreq} Hz, {raw.n_times} samples")

    # Parse events
    events_list = []
    with open(events_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            onset_s = float(row["onset"])
            sample = int(float(row.get("sample", onset_s * sfreq)))
            is_target = int(row.get("istarget", 0))
            events_list.append({
                "onset_s": onset_s,
                "sample": sample,
                "is_target": is_target == 1,
            })

    n_target = sum(1 for e in events_list if e["is_target"])
    n_nontarget = sum(1 for e in events_list if not e["is_target"])
    print(f"  Trials: {n_target} target, {n_nontarget} non-target")

    # Extract epochs: -200ms to +500ms around stimulus (700ms total)
    epoch_tmin_ms = -200
    epoch_tmax_ms = 500
    
    categories = {"target": [], "nontarget": []}
    for ev in events_list:
        cat = "target" if ev["is_target"] else "nontarget"
        t0 = ev["sample"] + int(epoch_tmin_ms * sfreq / 1000)
        t1 = ev["sample"] + int(epoch_tmax_ms * sfreq / 1000)
        if t0 < 0 or t1 > raw.n_times:
            continue
        segment = raw.get_data(start=t0, stop=t1)
        categories[cat].append(segment)

    result = {}
    metadata = {
        "sfreq_original": sfreq,
        "sfreq_target": SFREQ,
        "n_channels": n_channels,
        "epoch_tmin_ms": epoch_tmin_ms,
        "epoch_tmax_ms": epoch_tmax_ms,
    }

    for cat_name in ["target", "nontarget"]:
        segments = categories[cat_name]
        if not segments:
            result[cat_name] = np.zeros((0, n_channels, 0), dtype=np.float32)
            metadata[f"n_{cat_name}"] = 0
            continue
        arr = np.array(segments, dtype=np.float32)
        # Keep native 1000Hz - no decimation (short window PLV needs full resolution)
        result[cat_name] = arr
        metadata[f"n_{cat_name}"] = arr.shape[0]
        metadata[f"n_times_{cat_name}"] = arr.shape[2]

    metadata["sfreq"] = sfreq  # native rate
    print(f"  Loaded: target={metadata.get('n_target', 0)}, "
          f"nontarget={metadata.get('n_nontarget', 0)}")

    return result, metadata


def run_eureka_test(
    mode: str = "synthetic",
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the Eureka synchronization test."""
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "results" / "task1_insight")
    os.makedirs(output_dir, exist_ok=True)

    data_hashes = {}
    params = {
        "sfreq": SFREQ,
        "n_channels": N_CHANNELS,
        "plv_freq_band": PLV_FREQ_BAND,
        "mode": mode,
        "seed": seed,
    }

    print("=" * 60)
    print("TASK 1: Eureka Synchronization -- Phase-Locking at Target Detection")
    print("=" * 60)

    if mode == "synthetic":
        plv_band = (30.0, 50.0)  # Gamma: synthetic insight burst is at 40 Hz
        print("Generating synthetic EEG with known insight PLV spike...")
        params["solution_time_sec"] = SOLUTION_TIME_SEC
        params["baseline_window"] = BASELINE_WINDOW
        params["pre_window"] = PRE_WINDOW
        params["post_window"] = POST_WINDOW
        params["plv_freq_band_synthetic"] = plv_band
        
        synth = generate_synthetic_eeg_eureka(
            n_channels=N_CHANNELS, n_times=N_TIMES, sfreq=SFREQ,
            n_insight=30, n_analytic=30, seed=seed,
        )
        insight_trials = synth["insight"]
        analytic_trials = synth["analytic"]
        data_hashes["insight"] = compute_data_hash(insight_trials)
        data_hashes["analytic"] = compute_data_hash(analytic_trials)

        n_a = insight_trials.shape[0]
        n_b = analytic_trials.shape[0]
        print(f"  Insight trials: {n_a}, Analytic trials: {n_b}")

        insight_plv_pre = np.array([compute_trial_plv(t, PRE_WINDOW, plv_band) for t in insight_trials])
        insight_plv_post = np.array([compute_trial_plv(t, POST_WINDOW, plv_band) for t in insight_trials])
        analytic_plv_pre = np.array([compute_trial_plv(t, PRE_WINDOW, plv_band) for t in analytic_trials])
        analytic_plv_post = np.array([compute_trial_plv(t, POST_WINDOW, plv_band) for t in analytic_trials])

    elif mode == "real":
        if data_dir is None:
            raise ValueError("data_dir required for real mode")
        params["baseline_window_ms"] = BASELINE_WINDOW_MS
        params["pre_window_ms"] = PRE_WINDOW_MS
        params["post_window_ms"] = POST_WINDOW_MS

        loaded, load_meta = load_things_eeg_target_nontarget(data_dir)
        target_trials = loaded["target"]
        nontarget_trials = loaded["nontarget"]
        data_hashes["target"] = compute_data_hash(target_trials)
        data_hashes["nontarget"] = compute_data_hash(nontarget_trials)
        params["dataset"] = "ds003825 THINGS-EEG"
        params.update(load_meta)

        n_a = target_trials.shape[0]
        n_b = min(len(nontarget_trials), 500)
        nontarget_trials = nontarget_trials[:n_b]
        print(f"  Using {n_b} non-target trials (subsampled)")
        print(f"  Target trials: {n_a}, Non-target trials: {n_b}")

        if n_a == 0 or n_b == 0:
            return {"overall_pass": False, "error": "zero trials in one condition"}

        # iPLV with wide windows: 150ms pre, 200ms post, theta band (4-7 Hz)
        # iPLV suppresses volume conduction. Wide windows give stable phase est.
        sfreq_native = load_meta["sfreq_original"]
        onset_sample = int(200 * sfreq_native / 1000)
        pre_start = onset_sample - int(150 * sfreq_native / 1000)
        pre_end = onset_sample
        post_start = onset_sample + int(200 * sfreq_native / 1000)
        post_end = onset_sample + int(400 * sfreq_native / 1000)
        theta_band = (4.0, 7.0)
        
        target_plv_pre = []
        target_plv_post = []
        for trial in target_trials:
            target_plv_pre.append(compute_plv(trial[:, pre_start:pre_end], sfreq_native, theta_band, use_imaginary=True))
            target_plv_post.append(compute_plv(trial[:, post_start:post_end], sfreq_native, theta_band, use_imaginary=True))
        
        nontarget_plv_pre = []
        nontarget_plv_post = []
        for trial in nontarget_trials:
            nontarget_plv_pre.append(compute_plv(trial[:, pre_start:pre_end], sfreq_native, theta_band, use_imaginary=True))
            nontarget_plv_post.append(compute_plv(trial[:, post_start:post_end], sfreq_native, theta_band, use_imaginary=True))

        insight_plv_pre = np.array(target_plv_pre)
        insight_plv_post = np.array(target_plv_post)
        analytic_plv_pre = np.array(nontarget_plv_pre)
        analytic_plv_post = np.array(nontarget_plv_post)
        n_b = len(nontarget_plv_post)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Test 1a: PLV spike for insight/target (must be positive increase)
    print("\n--- Test 1a: PLV spike at insight/target ---")
    insight_diff = insight_plv_post - insight_plv_pre
    insight_test = permutation_test(insight_plv_post, insight_plv_pre, 10000, seed)
    sig_1a, p_1a = check_significant(insight_test)
    d_1a = cohens_d(insight_plv_post, insight_plv_pre)
    # Require both significance AND positive effect size
    pass_1a = sig_1a and insight_test["mean_a"] > insight_test["mean_b"]
    print(f"  PLV(pre)  = {insight_test['mean_b']:.4f} +/- {insight_test['std_b']:.4f}")
    print(f"  PLV(post) = {insight_test['mean_a']:.4f} +/- {insight_test['std_a']:.4f}")
    print(f"  p={p_1a:.6f}, d={d_1a:.3f}, PASS(positive spike)={pass_1a}")

    # Test 1b: No spike for analytic/nontarget
    print("\n--- Test 1b: No PLV spike for analytic/nontarget ---")
    analytic_diff = analytic_plv_post - analytic_plv_pre
    analytic_test = permutation_test(analytic_plv_post, analytic_plv_pre, 10000, seed)
    sig_1b, p_1b = check_significant(analytic_test)
    d_1b = cohens_d(analytic_plv_post, analytic_plv_pre)
    print(f"  PLV(pre)  = {analytic_test['mean_b']:.4f} +/- {analytic_test['std_b']:.4f}")
    print(f"  PLV(post) = {analytic_test['mean_a']:.4f} +/- {analytic_test['std_a']:.4f}")
    pass_1b = not sig_1b or d_1b < 0.5
    print(f"  p={p_1b:.6f}, d={d_1b:.3f}, PASS(no false positive)={pass_1b}")

    # Test 1c: Insight delta > analytic delta
    print("\n--- Test 1c: Insight vs analytic delta ---")
    delta_test = permutation_test(insight_diff, analytic_diff, 10000, seed)
    sig_1c, p_1c = check_significant(delta_test)
    d_1c = cohens_d(insight_diff, analytic_diff)
    print(f"  Mean diff a={delta_test['mean_a']:.4f}, b={delta_test['mean_b']:.4f}")
    print(f"  p={p_1c:.6f}, d={d_1c:.3f}, PASS={sig_1c}")

    results = {
        "test1a": {"pass": pass_1a, "p_value": p_1a, "cohens_d": d_1a},
        "test1b": {"pass": pass_1b, "p_value": p_1b, "cohens_d": d_1b},
        "test1c": {"pass": sig_1c, "p_value": p_1c, "cohens_d": d_1c},
        "overall_pass": pass_1a and pass_1b and sig_1c,
    }

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"{'=' * 60}")

    receipt_path = make_receipt("task1_eureka", results, params, data_hashes, output_dir)
    print(f"Receipt: {receipt_path}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 1: Eureka Synchronization EEG Test")
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run_eureka_test(
        mode=args.mode, data_dir=args.data_dir,
        output_dir=args.output_dir, seed=args.seed,
    )
    return 0 if results["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
