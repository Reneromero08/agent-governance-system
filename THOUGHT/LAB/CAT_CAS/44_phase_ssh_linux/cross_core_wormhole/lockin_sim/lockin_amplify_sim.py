#!/usr/bin/env python3
"""
Lock-in amplification feasibility sim for the cross-core cache-residency .holo carrier.

Question: the physical run gave a weak-but-REPRODUCIBLE mode lift
(ON 0.311 vs chance 0.251, +0.061; best config 0.40) while the real-vs-pseudo
floor sat at chance (~0.50). Does coherent averaging / lock-in demodulation of
the per-line residency vector lift these above the analyzer gates
(mode real_accuracy >= 0.60 ; real_vs_pseudo floor >= 0.95), and at what N?

This sim is deliberately NOT clean (the prior wormhole sim assumed a perfectly
invertible channel and predicted 0.79). Here we:
  - reproduce the analyzer EXACTLY (mode_contrast differential feature for the
    4-class gate; RAW 64-vector centroid for the real-vs-pseudo gate),
  - calibrate a generative residency model to the MEASURED operating point
    (mode_acc ~ 0.31, real_vs_pseudo ~ 0.50 at N=1),
  - add realistic noise: white + heavy-tail spikes + a MONOTONIC common-mode
    thermal drift (the thing plain averaging cannot remove but lock-in can) +
    an irreducible HOMOGENIZATION bias floor (signal-reducing, not averageable),
  - compare strategies: single-shot, plain block-average (N reps), and lock-in
    demodulation (toggle symbol vs reference at f_mod, demod per line),
  - sweep N and report the N needed to clear each gate, under both the
    store-and-forward (low co-access, A un-homogenized) and the sustained
    co-access (A homogenized) regimes.
"""

import numpy as np

RNG = np.random.default_rng(20260612)

# ---- analyzer mirror (matches analyze_cache_hologram_matched_nulls.py) ----
MODES = ["basis", "rotation", "residual", "mini"]
MODE_SETS = {
    "basis":    {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini":     {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}
# real warmed line family per mode (cache_hologram_matched_nulls.c real_mode_lines)
REAL_WARM = {
    "basis":    [9, 10, 11, 12, 13, 14],
    "rotation": [16, 17, 18, 19, 20, 21, 22, 23],
    "residual": [24, 25, 26, 27],
    "mini":     [9, 16, 24, 10, 17, 25, 11, 18, 26, 12, 19, 27],
}
# pseudo warmed line family per mode (disjoint, high indices) -> pseudo_mode_lines
PSEUDO_WARM = {
    "basis":    [33, 34, 35, 36, 37, 38],
    "rotation": [40, 41, 42, 43, 44, 45, 46, 47],
    "residual": [52, 53, 54, 55],
    "mini":     [33, 40, 52, 34, 41, 53, 35, 42, 54, 36, 43, 55],
}


def mode_contrast_features(vec):
    # vec: (..., 64). returns (..., 4) differential features (non-target - target)
    feats = []
    for m in MODES:
        tgt = np.array(sorted(MODE_SETS[m]))
        mask = np.ones(64, dtype=bool)
        mask[tgt] = False
        o = vec[..., mask].mean(axis=-1)
        t = vec[..., tgt].mean(axis=-1)
        feats.append(o - t)
    return np.stack(feats, axis=-1)


def nearest_centroid_predict(feats, centroids):
    # feats (...,K), centroids dict mode->(K,)
    cs = np.stack([centroids[m] for m in MODES], axis=0)  # (4,K)
    d = np.linalg.norm(feats[..., None, :] - cs[None, ...], axis=-1)  # (...,4)
    return d.argmin(axis=-1)


# ---- generative residency model ----
def gen_vectors(n, warm_lines, A, sigma_w, drift_amp, p_out, sigma_out,
                homog_floor_frac, drift_mode="ramp"):
    """
    Returns (n, 64) latency vectors.
      A          : warmed-line residency contrast (cycles faster) -- the signal
      sigma_w    : per-(sample,line) white noise std
      drift_amp  : amplitude of common-mode (all-line) drift across the n samples
      p_out,sigma_out : heavy-tail spike prob / std
      homog_floor_frac: fraction of A that is IRREDUCIBLY smeared away
                        (homogenization that reduces the per-sample signal AND
                         is correlated across reps so averaging cannot recover it)
      drift_mode : 'ramp' (monotonic, defeats plain averaging) or 'zero_mean'
    """
    base = 200.0  # baseline load latency (cycles), arbitrary units
    v = np.full((n, 64), base) + RNG.normal(0, sigma_w, size=(n, 64))
    # warmed lines: faster by A. homogenization removes homog_floor_frac of A
    A_eff = A * (1.0 - homog_floor_frac)
    if len(warm_lines) > 0:
        warm = np.array(warm_lines, dtype=int)
        v[:, warm] -= A_eff
    # common-mode drift (rejected by differential feature; swamps raw-vector gate)
    if drift_mode == "ramp":
        d = np.linspace(-drift_amp, drift_amp, n)
    else:
        d = RNG.normal(0, drift_amp, size=n)
    v += d[:, None]
    # heavy-tail spikes
    spike = RNG.random((n, 64)) < p_out
    v += spike * RNG.normal(0, sigma_out, size=(n, 64))
    return v


def block_average(vecs, N):
    """average consecutive blocks of N -> (n//N, 64). plain coherent averaging."""
    n = (vecs.shape[0] // N) * N
    return vecs[:n].reshape(-1, N, 64).mean(axis=1)


def lockin_demod(sym_vecs, ref_vecs, N):
    """
    Lock-in: toggle symbol vs reference at f_mod. Take N symbol probes and N
    reference probes interleaved; demodulate per line = mean(symbol)-mean(reference).
    This SUBTRACTS any common-mode drift/offset shared by symbol & reference
    epochs (to first order) AND averages white noise by ~sqrt(2N).
    Returns demodulated 64-vectors, one per group of N.
    """
    nsym = (sym_vecs.shape[0] // N) * N
    nref = (ref_vecs.shape[0] // N) * N
    g = min(nsym, nref) // N
    s = sym_vecs[: g * N].reshape(g, N, 64).mean(axis=1)
    r = ref_vecs[: g * N].reshape(g, N, 64).mean(axis=1)
    # demod vector = symbol residency minus reference residency (per line)
    # restore baseline so the analyzer's distance geometry is comparable
    return (s - r)


# ---- experiment runners ----
def make_mode_dataset(n_per_mode, family, params, N=1, strategy="single"):
    """build a classified accuracy for the 4-class MODE gate (differential feature)."""
    warm_table = REAL_WARM if family == "real" else PSEUDO_WARM
    # build train + test, mirror analyzer's centroid-from-train then test
    feats_by_mode_test = {}
    centroids = {}
    for mi, m in enumerate(MODES):
        wl = warm_table[m]
        # generate enough raw samples for both train and test, then aggregate
        ntot = n_per_mode * 2 * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        raw = gen_vectors(ntot, wl, **params)
        if strategy == "single":
            agg = raw
        elif strategy == "plain":
            agg = block_average(raw, N)
        elif strategy == "lockin":
            # reference = no warmed lines (empty footprint)
            ref = gen_vectors(ntot, [], **params)
            agg = lockin_demod(raw, ref, N)
        half = agg.shape[0] // 2
        train, test = agg[:half], agg[half:]
        centroids[m] = mode_contrast_features(train).mean(axis=0)
        feats_by_mode_test[m] = mode_contrast_features(test)
    # classify test
    correct = 0
    total = 0
    for mi, m in enumerate(MODES):
        preds = nearest_centroid_predict(feats_by_mode_test[m], centroids)
        correct += (preds == mi).sum()
        total += len(preds)
    return correct / total


def make_rvp_floor(n_per_mode, params, N=1, strategy="single"):
    """real-vs-pseudo binary floor over modes, on RAW 64-vectors (analyzer-faithful)."""
    floors = []
    for m in MODES:
        rl, pl = REAL_WARM[m], PSEUDO_WARM[m]
        ntot = n_per_mode * 2 * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        rraw = gen_vectors(ntot, rl, **params)
        praw = gen_vectors(ntot, pl, **params)
        if strategy == "single":
            ragg, pagg = rraw, praw
        elif strategy == "plain":
            ragg, pagg = block_average(rraw, N), block_average(praw, N)
        elif strategy == "lockin":
            refr = gen_vectors(ntot, [], **params)
            refp = gen_vectors(ntot, [], **params)
            ragg = lockin_demod(rraw, refr, N)
            pagg = lockin_demod(praw, refp, N)
        # centroid on raw vector (mirror analyzer: distance on _vector)
        hr, hp = ragg.shape[0] // 2, pagg.shape[0] // 2
        cr = ragg[:hr].mean(axis=0)
        cp = pagg[:hp].mean(axis=0)
        rt, pt = ragg[hr:], pagg[hp:]
        # predict family by nearest raw-vector centroid
        dr_r = np.linalg.norm(rt - cr, axis=1) <= np.linalg.norm(rt - cp, axis=1)
        dp_p = np.linalg.norm(pt - cp, axis=1) < np.linalg.norm(pt - cr, axis=1)
        acc = (dr_r.sum() + dp_p.sum()) / (len(rt) + len(pt))
        floors.append(acc)
    return min(floors)


def calibrate():
    """find A, sigma_w that reproduce mode_acc~0.31, rvp~0.50 at N=1."""
    # fixed realistic noise structure
    base = dict(drift_amp=14.0, p_out=0.02, sigma_out=40.0,
                homog_floor_frac=0.0, drift_mode="ramp")
    # search A, sigma_w
    best = None
    for A in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for sw in [20.0, 25.0, 30.0, 35.0, 45.0]:
            p = dict(A=A, sigma_w=sw, **base)
            ma = np.mean([make_mode_dataset(1500, "real", p) for _ in range(3)])
            rv = np.mean([make_rvp_floor(1500, p) for _ in range(3)])
            score = abs(ma - 0.31) + abs(rv - 0.50)
            if best is None or score < best[0]:
                best = (score, A, sw, ma, rv)
    return best, base


def main():
    np.set_printoptions(precision=4, suppress=True)
    (score, A, sw, ma, rv), base = calibrate()
    print(f"# CALIBRATION  A={A} sigma_w={sw}  -> mode_acc={ma:.3f} rvp={rv:.3f} (target 0.31/0.50)")
    params = dict(A=A, sigma_w=sw, **base)

    print("\n# STORE-AND-FORWARD regime (low co-access, A un-homogenized): sweep N")
    print(f"{'N':>5} {'plain_mode':>11} {'lockin_mode':>12} {'plain_rvp':>10} {'lockin_rvp':>11}")
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    res = {}
    for N in Ns:
        pm = np.mean([make_mode_dataset(max(400, 4000 // N), "real", params, N, "plain") for _ in range(3)])
        lm = np.mean([make_mode_dataset(max(400, 4000 // N), "real", params, N, "lockin") for _ in range(3)])
        pr = np.mean([make_rvp_floor(max(400, 4000 // N), params, N, "plain") for _ in range(3)])
        lr = np.mean([make_rvp_floor(max(400, 4000 // N), params, N, "lockin") for _ in range(3)])
        res[N] = (pm, lm, pr, lr)
        print(f"{N:>5} {pm:>11.3f} {lm:>12.3f} {pr:>10.3f} {lr:>11.3f}")

    # report crossing N
    def first_cross(idx, thr):
        for N in Ns:
            if res[N][idx] >= thr:
                return N
        return None
    print("\n# GATE CROSSINGS (store-and-forward)")
    print(f"  plain  mode>=0.60 at N={first_cross(0,0.60)}   lockin mode>=0.60 at N={first_cross(1,0.60)}")
    print(f"  plain  rvp >=0.95 at N={first_cross(2,0.95)}   lockin rvp >=0.95 at N={first_cross(3,0.95)}")

    print("\n# SUSTAINED CO-ACCESS regime (homogenization removes 70% of A): sweep N")
    params_h = dict(A=A, sigma_w=sw, **{**base, "homog_floor_frac": 0.70})
    print(f"{'N':>5} {'lockin_mode':>12} {'lockin_rvp':>11}")
    resh = {}
    for N in Ns:
        lm = np.mean([make_mode_dataset(max(400, 4000 // N), "real", params_h, N, "lockin") for _ in range(3)])
        lr = np.mean([make_rvp_floor(max(400, 4000 // N), params_h, N, "lockin") for _ in range(3)])
        resh[N] = (lm, lr)
        print(f"{N:>5} {lm:>12.3f} {lr:>11.3f}")
    for N in Ns:
        if resh[N][0] >= 0.60:
            print(f"  homogenized lockin mode>=0.60 at N={N}")
            break
    else:
        print(f"  homogenized lockin mode>=0.60: NOT REACHED by N={Ns[-1]} (max {max(r[0] for r in resh.values()):.3f})")

    print("\n# FULLY HOMOGENIZED regime (homog_floor_frac=1.0, signal truly zeroed)")
    params_z = dict(A=A, sigma_w=sw, **{**base, "homog_floor_frac": 1.0})
    lm = np.mean([make_mode_dataset(2000, "real", params_z, 256, "lockin") for _ in range(3)])
    lr = np.mean([make_rvp_floor(2000, params_z, 256, "lockin") for _ in range(3)])
    print(f"  N=256 lockin mode={lm:.3f} rvp={lr:.3f}  (expect ~chance: averaging cannot revive a zero-mean signal)")


if __name__ == "__main__":
    main()
