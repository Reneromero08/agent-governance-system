"""
odd_lane_transfer_function.py -- Track 0: Odd-Lane Transfer Function Calibration.

Measures the minimum detectable fold-odd lane amplitude by injecting synthetic
quadrature at controlled epsilon levels. This calibrates the detector BEFORE
interpreting any negative result from Tracks A/B/C/D/E/F/G/H/K.

The synthetic oracle extends the public cosine channel with controlled quadrature:
  z_k = cos(theta_k) + i * epsilon * sin(theta_k)
where theta_k = 2*pi*k*d/N.

For epsilon=1.0: full complex coefficient (orientation trivially AUC ~1.0).
For epsilon=0.0: pure cosine (orientation absent, AUC ~0.5).
For intermediate epsilon: partial orientation recoverability.

This script computes the MATHEMATICAL transfer function: given epsilon, what is
the information-theoretic upper bound on orientation recovery? This is the
REFERENCE curve against which any physical Phenom measurement is compared.

Two readout channels tested:
  CHANNEL A (public cosine only): reads cos(theta_k) from b_i. Blind to epsilon.
  CHANNEL B (injected quadrature): reads epsilon*sin(theta_k). Orientation signal.

Discipline: ASCII only. All RNGs seeded. Deterministic. Claim ceiling L4.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLD_AUDIT = HERE.parent.parent / "cross_cutting" / "fold_audit"
sys.path.insert(0, str(FOLD_AUDIT))

import construction as C

MASTER_SEED = 44060611
LINES = []


def log(m=""):
    print(m)
    LINES.append(str(m))


# ===========================================================================
# Epsilon-ladder: synthetic quadrature injection
# ===========================================================================
EPSILON_LADDER = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0]


def synthetic_oracle(inst, epsilon):
    """Build synthetic oracle data with controlled quadrature injection.

    The REAL public data is b_i = sign(cos(theta_i) + noise).
    This is fold-symmetric. We EXTEND it with a synthetic sin channel at
    amplitude epsilon, simulating what the detector WOULD see if a physical
    odd-lane existed at that amplitude.

    Returns (cos_data, sin_data) where:
      cos_data = the real public b_i values (cosine channel)
      sin_data = epsilon * sin(theta_i) (injected quadrature, HIDDEN)
    """
    k = inst["k"]
    d = inst["d"]
    N = inst["N"]
    b = inst["b"]
    theta = 2 * np.pi * k * d / N
    sin_data = epsilon * np.sin(theta)
    return b, sin_data, theta


def orientation_score_from_quadrature(sin_data, theta):
    """One-shot orientation readout from quadrature channel.

    sign(sin(2*pi*1*d/N)) determines orientation at the k=1 rung.
    sin(2*pi*d/N) > 0 iff 0 < d < N/2 iff orientation = 1.
    """
    # The k=1 rung: sin(2*pi*1*d/N)
    # We estimate this from the sin_data at k=1
    # In practice: average sign-weighted sin_data
    orientation_estimate = np.mean(np.sign(sin_data))
    return float(orientation_estimate)


def build_quadrature_injected_dataset(n, n_instances, epsilon, seed):
    """Generate instances with synthetic quadrature at given epsilon.

    For each instance, we have:
      - Public data (k, b) from the real construction
      - Synthetic sin channel at amplitude epsilon
      - Hidden orientation label

    The READOUT channels are:
      CHANNEL A: function of public (k, b) only -- should be at chance regardless of epsilon
      CHANNEL B: function of injected sin_data -- should recover orientation with SNR ~ epsilon
    """
    rng = np.random.default_rng(seed)
    N = 1 << n
    instances = []
    for _ in range(n_instances):
        d = C.sample_secret(N, rng)
        M = C.M_for(n)
        k, b = C.coset_samples(N, d, M, rng)
        inst = {"k": k, "b": b, "N": N, "d": int(d % N), "n": n}
        b_cos, sin_data, theta = synthetic_oracle(inst, epsilon)
        orientation = C.orientation_bit(d, N)
        instances.append({
            "k": k,
            "b": b_cos,
            "sin_injected": sin_data,
            "theta": theta,
            "orientation": orientation,
            "d": int(d % N),
            "n": n,
            "N_val": int(N),
            "epsilon": epsilon,
        })
    return instances


# ===========================================================================
# Channel A: Public cosine only (fold-symmetric, should be at chance)
# ===========================================================================
def channel_a_features(inst_data):
    """Public scalar features from cosine channel only. Same as fold_audit stage1."""
    k = inst_data["k"]
    b = inst_data["b"]
    N = inst_data["N_val"]
    # Use the same probe points as the fold audit
    probes = np.linspace(1, N / 2, 24, endpoint=False)
    s = np.array([C.score(k, b, float(x), int(N)) for x in probes]) / max(len(b), 1)
    moments = np.array([np.mean(b), np.std(b),
                        float(np.mean(b * np.cos(2 * np.pi * k / N))),
                        float(np.mean(b * np.cos(4 * np.pi * k / N)))])
    return np.concatenate([s, moments])


# ===========================================================================
# Channel B: Injected quadrature (orientation signal proportional to epsilon)
# ===========================================================================
def channel_b_features(inst_data):
    """Features from the injected sin channel.

    The INJECTED sin_data = epsilon * sin(theta_k) for each k value.
    At k=1: sin_data[k_idx] = epsilon * sin(2*pi*d/N).
    This directly encodes orientation when epsilon > 0:
      sin(2*pi*d/N) > 0 iff orientation = 1.

    We bin the random k values to estimate sin at the k=1 rung from
    the INJECTED data (not from hidden d). At epsilon=0, sin_data is all
    zeros and no orientation information exists.
    """
    k = inst_data["k"]
    sin_inj = inst_data["sin_injected"]
    N = inst_data["N_val"]

    # Estimate sin at k=1 from the injected data by binning
    # k=1 and k=N-1 both have theta = 2*pi*d/N (cos same, sin opposite sign)
    # sin(2*pi*1*d/N) at k=1; sin(2*pi*(N-1)*d/N) = -sin(2*pi*d/N) at k=N-1
    k1_mask = k == 1
    kn1_mask = k == (N - 1)
    sin_k1_est = 0.0
    count = 0
    if np.any(k1_mask):
        sin_k1_est += float(np.mean(sin_inj[k1_mask]))
        count += 1
    if np.any(kn1_mask):
        sin_k1_est -= float(np.mean(sin_inj[kn1_mask]))  # sin(N-1) = -sin(1)
        count += 1
    if count > 0:
        sin_k1_est /= count

    # Also estimate from all k: sign(sin(theta)) averaged
    sin_signed_mean = float(np.mean(np.sign(sin_inj + 1e-30)))

    return np.array([sin_k1_est, sin_signed_mean])


# ===========================================================================
# AUC measurement
# ===========================================================================
def auc_one(scores, labels):
    """AUC from scores and binary labels (no sklearn dependency)."""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and abs(pairs[j][0] - pairs[i][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for item in pairs[i:j]:
            if item[1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def shuffle_null_95(scores, labels, seed, n_shuffles=80):
    """95th percentile AUC from label-shuffled nulls."""
    rng = np.random.default_rng(seed + 777)
    nulls = []
    for _ in range(n_shuffles):
        yp = rng.permutation(labels)
        nulls.append(auc_one(scores, yp))
    nulls.sort()
    return float(np.percentile(nulls, 95)), float(np.mean(nulls))


# ===========================================================================
# Transfer function measurement
# ===========================================================================
def measure_transfer_function(n, n_instances, epsilon, seed):
    """Measure orientation recovery AUC at a given epsilon.

    Returns:
      auc_channel_a: AUC using public cosine features only
      auc_channel_b: AUC using injected sin features
      null95_a, null95_b: shuffle null thresholds
      snr_estimate: signal-to-noise ratio estimate from channel B
    """
    instances = build_quadrature_injected_dataset(n, n_instances, epsilon, seed)

    # Channel A: public cosine only
    Xa = np.array([channel_a_features(d) for d in instances])
    ya = np.array([d["orientation"] for d in instances])
    # Simple logistic score: mean of first feature as probe
    scores_a = np.mean(Xa[:, :5], axis=1) if Xa.shape[1] >= 5 else Xa[:, 0]
    auc_a = auc_one(scores_a, ya)
    null95_a, null_mean_a = shuffle_null_95(scores_a, ya, seed)

    # Channel B: injected quadrature
    Xb = np.array([channel_b_features(d) for d in instances])
    yb = np.array([d["orientation"] for d in instances])
    # Primary probe: estimated sin at k=1 from injected data (feature 0)
    scores_b = Xb[:, 0]
    auc_b = auc_one(scores_b, yb)
    auc_b_signed = max(auc_b, 1.0 - auc_b)
    null95_b, null_mean_b = shuffle_null_95(scores_b, yb, seed)
    null95_b_signed = max(null95_b, 1.0 - null_mean_b)
    # SNR: signal / noise at estimated k=1 sin
    signal = abs(np.mean(scores_b * (2 * np.array(yb) - 1)))
    noise = np.std(scores_b)
    snr = signal / (noise + 1e-12) if noise > 0 else 0.0

    # SNR estimate: signal strength at k=1 vs noise floor
    k1_theoretical = Xb[:, 0]  # sin(2*pi*d/N)
    signal = np.mean(k1_theoretical * (2 * np.array(yb) - 1))  # sign-corrected mean
    noise = np.std(k1_theoretical)
    snr = abs(signal) / (noise + 1e-12) if noise > 0 else 0.0

    return {
        "n": n,
        "epsilon": epsilon,
        "n_instances": n_instances,
        "channel_a_auc": float(auc_a),
        "channel_a_null95": float(null95_a),
        "channel_a_above_null": bool(auc_a > null95_a),
        "channel_b_auc": float(auc_b),
        "channel_b_auc_signed": float(auc_b_signed),
        "channel_b_null95_signed": float(null95_b_signed),
        "channel_b_above_null": bool(auc_b_signed > null95_b_signed + 0.03),
        "snr_estimate": float(snr),
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t0 = time.time()
    log("=" * 90)
    log("TRACK 0 -- ODD-LANE TRANSFER FUNCTION CALIBRATION")
    log("master_seed=%d   epsilon ladder=%s" % (MASTER_SEED, EPSILON_LADDER))
    log("=" * 90)

    N_LIST = [8, 10]
    N_INST = 400

    results = {
        "experiment": "phase6_track0_odd_lane_transfer_function",
        "master_seed": MASTER_SEED,
        "epsilon_ladder": EPSILON_LADDER,
        "n_values": N_LIST,
        "n_instances": N_INST,
        "cells": [],
    }

    for n in N_LIST:
        log("\n--- n=%d (N=%d, M=%d) ---" % (n, 1 << n, C.M_for(n)))
        for eps_idx, eps in enumerate(EPSILON_LADDER):
            seed = MASTER_SEED + 100 * n + eps_idx
            cell = measure_transfer_function(n, N_INST, eps, seed)
            results["cells"].append(cell)
            tag = "ABOVE_NULL" if cell["channel_b_above_null"] else "at_chance"
            log("  eps=%.4f  chA_auc=%.3f/null=%.3f  chB_auc=%.3f/null=%.3f  snr=%.4f  %s"
                % (eps, cell["channel_a_auc"], cell["channel_a_null95"],
                   cell["channel_b_auc_signed"], cell["channel_b_null95_signed"],
                   cell["snr_estimate"], tag))

    # --- Find minimum detectable epsilon (MDE) ---
    mde = None
    # Find highest epsilon where channel B still at chance, then next is MDE
    for eps in reversed(EPSILON_LADDER):
        cells_at_eps = [c for c in results["cells"] if abs(c["epsilon"] - eps) < 1e-9]
        if cells_at_eps:
            all_below = all(not c["channel_b_above_null"] for c in cells_at_eps)
            if all_below and eps < 1.0:
                # The next epsilon up is the MDE if it's above null
                idx = EPSILON_LADDER.index(eps)
                if idx > 0:
                    next_eps = EPSILON_LADDER[idx - 1]
                    next_cells = [c for c in results["cells"]
                                  if abs(c["epsilon"] - next_eps) < 1e-9]
                    if next_cells and any(c["channel_b_above_null"] for c in next_cells):
                        mde = next_eps
                        break

    if mde is None and any(c["channel_b_above_null"]
                           for c in results["cells"] if abs(c["epsilon"] - 0.03125) < 1e-9):
        mde = 0.03125
    if mde is None and any(c["channel_b_above_null"]
                           for c in results["cells"] if abs(c["epsilon"] - 0.0) < 1e-9):
        mde = 0.0  # even at epsilon=0, orientation is recoverable (should not happen)

    results["minimum_detectable_epsilon"] = mde

    # --- Monotonicity check ---
    eps_aucs = {}
    for c in results["cells"]:
        key = (c["n"], c["epsilon"])
        if key not in eps_aucs:
            eps_aucs[key] = []
        eps_aucs[key].append(c["channel_b_auc_signed"])

    monotonic = True
    for n in N_LIST:
        aucs_by_eps = [(eps, np.mean(eps_aucs[(n, eps)])) for eps in EPSILON_LADDER]
        for i in range(len(aucs_by_eps) - 1):
            if aucs_by_eps[i][1] < aucs_by_eps[i + 1][1] - 0.01:
                monotonic = False  # AUC should DECREASE with epsilon, not increase

    results["monotonic"] = bool(monotonic)

    # --- Verdict ---
    log("\n" + "=" * 90)
    log("TRACK 0 VERDICT")
    channel_a_all_chance = all(
        not c["channel_a_above_null"] for c in results["cells"])
    has_signal = any(c["channel_b_above_null"] for c in results["cells"])

    log("  Channel A (cosine only) at chance: %s" % channel_a_all_chance)
    log("  Channel B (injected sin) above null at some epsilon: %s" % has_signal)
    log("  Minimum detectable epsilon: %s" % mde)
    log("  Epsilon ladder monotonic: %s" % monotonic)

    if channel_a_all_chance and has_signal and monotonic and mde is not None:
        verdict = "ODD_LANE_DETECTOR_CALIBRATED"
        log("  VERDICT: %s (MDE=%.5f)" % (verdict, mde))
        log("  A physical detector must resolve fold-odd amplitude >= %.5f to recover orientation." % mde)
        log("  This is the MATHEMATICAL reference curve. Compare Phenom hardware to this.")
    elif not has_signal:
        verdict = "ODD_LANE_DETECTOR_NOT_LIVE"
        log("  VERDICT: %s -- no epsilon level recovered orientation" % verdict)
    elif not monotonic:
        verdict = "ODD_LANE_DETECTOR_ARTIFACT"
        log("  VERDICT: %s -- epsilon ladder not monotonic" % verdict)
    else:
        verdict = "ODD_LANE_DETECTOR_CALIBRATED_PARTIAL"
        log("  VERDICT: %s" % verdict)

    results["verdict"] = verdict
    results["elapsed_s"] = round(time.time() - t0, 2)

    log("\n  elapsed: %.2f s" % results["elapsed_s"])
    log("=" * 90)

    (HERE / "results" / "odd_lane_transfer_function.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (HERE / "results" / "output_transfer_function.txt").write_text(
        "\n".join(LINES), encoding="utf-8")
    log("\nwrote results/odd_lane_transfer_function.json + output_transfer_function.txt")

    return 0 if verdict == "ODD_LANE_DETECTOR_CALIBRATED" else 1


if __name__ == "__main__":
    sys.exit(main())

