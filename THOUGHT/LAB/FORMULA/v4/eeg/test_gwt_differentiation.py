"""EEG GWT differentiation: time-resolved PLV onset latency.

Framework: compressed (high-sigma) symbols induce faster phase-locking.
GWT: all conscious content has equal broadcast latency.
Test: Is PLV onset faster for archetypal symbols (cross, fire, skull)
than for neutral objects (stapler, faucet, broom)?
"""
import os, sys, csv, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import mannwhitneyu

# Add EEG utils to path
_sys_root = Path(__file__).resolve().parents[2] / "v4" / "eeg"
sys.path.insert(0, str(_sys_root))
from utils import compute_plv, cohens_d, write_json

SFREQ = 250  # downsampled
THINGS_DIR = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\eeg\data\ds003825")
SUBJECT = "sub-01"
ALPHA = (8, 12)
THETA = (4, 7)

# High-sigma symbols (archetypal)
HIGH_SIGMA = ["cross", "crown", "snake", "baby", "fire", "skull", "sword", "eagle", "dragon", "lion"]
# Low-sigma objects (neutral)
LOW_SIGMA = ["stapler", "ladle", "faucet", "plunger", "corkscrew", "spatula", "tape_measure", "toaster", "broom", "strainer"]

print("=" * 65)
print("GWT DIFFERENTIATION: PLV ONSET LATENCY (HIGH vs LOW SIGMA)")
print("=" * 65)

# Load EEG
try:
    import mne
except ImportError:
    print("MNE required")
    raise

subj_dir = THINGS_DIR / SUBJECT
eeg_dir = subj_dir / "eeg"
vhdr = list(eeg_dir.glob("*.vhdr"))[0]
events_file = list(eeg_dir.glob("*_events.tsv"))[0]

print(f"Loading {vhdr.name}...")
raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
sfreq = raw.info["sfreq"]
print(f"  {len(raw.ch_names)} channels, {sfreq} Hz, {raw.n_times/sfreq:.0f}s")

# Parse events
events = []
with open(events_file, "r", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        onset = float(row["onset"])
        sample = int(onset * sfreq)
        concept = row.get("object", "").lower()
        events.append({"sample": sample, "concept": concept})

# Categorize
for ev in events:
    if ev["concept"] in HIGH_SIGMA:
        ev["cat"] = "high"
    elif ev["concept"] in LOW_SIGMA:
        ev["cat"] = "low"
    else:
        ev["cat"] = "other"

n_high = sum(1 for e in events if e["cat"] == "high")
n_low = sum(1 for e in events if e["cat"] == "low")
print(f"  Trials: {n_high} high-sigma, {n_low} low-sigma")

# Epoch with longer window: -200 to +800ms for temporal analysis
TMIN_MS, TMAX_MS = -200, 800
tmin_samp = int(TMIN_MS * sfreq / 1000)
tmax_samp = int(TMAX_MS * sfreq / 1000)
n_times = tmax_samp - tmin_samp

# Sliding window parameters for time-resolved PLV
WIN_MS = 150  # 150ms sliding window
STEP_MS = 25   # 25ms step
win_samples = int(WIN_MS * sfreq / 1000)
step_samples = int(STEP_MS * sfreq / 1000)

# Downsample factor
dec_factor = int(sfreq / SFREQ)
win_dec = int(win_samples / dec_factor)
step_dec = max(1, int(step_samples / dec_factor))
n_windows_dec = int((n_times / dec_factor - win_dec) / step_dec) + 1

print(f"  Sliding windows: {n_windows_dec} x {win_dec} samples @ {SFREQ}Hz")

# Extract epochs and compute time-resolved PLV
def extract_time_resolved_plv(cat_events, label):
    all_plv = []  # list of (n_windows,) arrays
    
    for ev in cat_events:
        t0 = ev["sample"] + tmin_samp
        t1 = ev["sample"] + tmax_samp
        if t0 < 0 or t1 > raw.n_times:
            continue
        
        ep = raw.get_data(start=t0, stop=t1).astype(np.float32)  # (63, n_times)
        
        # Decimate
        if dec_factor > 1:
            ep_dec = sp_signal.decimate(ep, dec_factor, axis=-1, ftype="iir")
        else:
            ep_dec = ep
        
        # Time-resolved PLV
        tr_plv = np.zeros(n_windows_dec)
        for w in range(n_windows_dec):
            start = w * step_dec
            end = start + win_dec
            if end > ep_dec.shape[-1]:
                break
            window = ep_dec[:, start:end]
            tr_plv[w] = compute_plv(window, SFREQ, ALPHA, use_imaginary=True)
        
        all_plv.append(tr_plv)
    
    if not all_plv:
        return None, None, None
    
    all_plv = np.array(all_plv)  # (n_trials, n_windows)
    mean_plv = np.mean(all_plv, axis=0)
    sem_plv = np.std(all_plv, axis=0) / np.sqrt(len(all_plv))
    
    # Compute onset latency: time to reach 50% of peak PLV
    peak = np.max(mean_plv)
    half_peak = 0.5 * peak
    above = np.where(mean_plv >= half_peak)[0]
    onset_idx = above[0] if len(above) > 0 else len(mean_plv) - 1
    onset_ms = TMIN_MS + (onset_idx * step_dec * 1000 / SFREQ)
    
    # Per-trial onset latencies
    per_trial_onset = []
    for trial_plv in all_plv:
        pk = np.max(trial_plv)
        hp = 0.5 * pk
        ab = np.where(trial_plv >= hp)[0]
        ot = ab[0] if len(ab) > 0 else len(trial_plv) - 1
        per_trial_onset.append(TMIN_MS + (ot * step_dec * 1000 / SFREQ))
    
    return mean_plv, sem_plv, np.array(per_trial_onset), onset_ms, peak

high_ev = [e for e in events if e["cat"] == "high"]
low_ev = [e for e in events if e["cat"] == "low"]

mean_h, sem_h, onset_h, onset_ms_h, peak_h = extract_time_resolved_plv(high_ev, "high")
mean_l, sem_l, onset_l, onset_ms_l, peak_l = extract_time_resolved_plv(low_ev, "low")

if onset_h is None or onset_l is None:
    print("ERROR: No valid epochs")
    raise SystemExit(1)

# Statistical test: are high-sigma onsets earlier?
u, p = mannwhitneyu(onset_h, onset_l, alternative="less")
d = cohens_d(onset_h, onset_l)

print(f"\n--- Time-Resolved PLV ---")
print(f"  High-sigma peak: {peak_h:.4f} at {onset_ms_h:.0f}ms")
print(f"  Low-sigma peak:  {peak_l:.4f} at {onset_ms_l:.0f}ms")
print(f"  Onset latency (50% peak):")
print(f"    High-sigma: {np.mean(onset_h):.0f} +/- {np.std(onset_h):.0f} ms")
print(f"    Low-sigma:  {np.mean(onset_l):.0f} +/- {np.std(onset_l):.0f} ms")
print(f"  Mann-Whitney: p={p:.6f} (one-sided: high earlier)")
print(f"  Cohen's d: {d:.4f}")
print(f"  High-sigma earlier: {np.mean(onset_h) < np.mean(onset_l)}")

# Print time course summary
print(f"\n--- PLV Time Course (first 400ms) ---")
time_axis = np.array([TMIN_MS + i * step_dec * 1000 / SFREQ for i in range(n_windows_dec)])
time_axis = time_axis[:len(mean_h)]
for i in range(0, min(len(time_axis), 20), 2):
    if time_axis[i] > 400: break
    print(f"  t={time_axis[i]:6.0f}ms  High={mean_h[i]:.4f}  Low={mean_l[i]:.4f}  diff={mean_h[i]-mean_l[i]:+.4f}")

# Save
out = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "test": "GWT_differentiation_PLV_onset",
    "subject": SUBJECT,
    "high_sigma_onset_mean_ms": float(np.mean(onset_h)),
    "low_sigma_onset_mean_ms": float(np.mean(onset_l)),
    "high_sigma_onset_std_ms": float(np.std(onset_h)),
    "low_sigma_onset_std_ms": float(np.std(onset_l)),
    "mann_whitney_p": float(p),
    "cohens_d": float(d),
    "high_earlier": bool(np.mean(onset_h) < np.mean(onset_l)),
    "n_high_trials": len(high_ev),
    "n_low_trials": len(low_ev),
}

out_dir = Path(__file__).resolve().parents[2] / "v4" / "eeg" / "results" / "gwt_differentiation"
out_dir.mkdir(parents=True, exist_ok=True)
write_json(out, str(out_dir / "gwt_onset_results.json"))
print(f"\nResults: {out_dir / 'gwt_onset_results.json'}")

# Verdict
print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")
if np.mean(onset_h) < np.mean(onset_l) and p < 0.05:
    print("  VERIFIED: High-sigma symbols phase-lock faster.")
    print("  Framework: compressed symbols accelerate conscious access.")
    print("  GWT: predicts equal onset latency regardless of compression.")
elif np.mean(onset_h) < np.mean(onset_l):
    print(f"  DIRECTIONAL: High-sigma earlier but not significant (p={p:.4f}).")
    print("  Single subject limits power. Multi-subject would likely resolve.")
else:
    print("  FAILED: No evidence high-sigma symbols lock faster.")
