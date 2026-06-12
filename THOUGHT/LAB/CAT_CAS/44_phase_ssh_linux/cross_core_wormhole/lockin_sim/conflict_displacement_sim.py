#!/usr/bin/env python3
"""
Conflict-displacement cross-core readout feasibility sim (EXP44 angle vetting).

Replaces the OCCUPANCY/WARMTH carrier (phase4b cross-core: receiver times the
SENDER's warmed lines; +0.061 over chance, homogenizes under co-access, FAILED
the 0.60 gate) with CONFLICT-DISPLACEMENT (receiver fills cache SETS with its
OWN data; sender accesses addresses mapping to the same sets; receiver detects
EVICTION of its own lines by its own re-access timing).

The point of this sim is an HONEST go/no-go, not a clean invertible channel. We:
  - mirror the analyzer (4-class differential-feature MODE gate + raw-vector
    real-vs-pseudo floor gate) used by analyze_cache_hologram_matched_nulls.py,
  - calibrate the WARMTH baseline to the MEASURED operating point
    (mode_acc ~ 0.31, rvp ~ 0.50 at N=1) so we trust the noise model,
  - then swap in the conflict-displacement physics MEASURED in 5.10D:
      same-set aggressor pushes the victim's per-touch time to ~319 cyc vs
      ~270 (no aggressor / compute-only) => ~49-cycle displacement contrast,
      p=0.0005, sign-agreement 1.0, reproduced across swapped cores.
  - model the homogenization HONESTLY. The key physical claim under test:
    occupancy/warmth homogenizes (co-access averages the per-line warmth away,
    homog_floor_frac ~ 0.70). Conflict-displacement does NOT average the same
    way -- it is a per-SET occupancy COMPETITION the receiver feels directly --
    but it has its OWN degradation: under sustained co-access the sender's
    footprint spreads across MANY sets, raising the receiver's baseline miss
    rate on ALL sets (a common-mode pedestal) and shrinking the per-SYMBOL
    contrast between targeted and non-targeted sets. We sweep that shrink.

Honesty stance: we DO NOT assume the channel is invertible. The receiver reads
a per-set displacement vector; there is NO de-permute that recovers a washed-out
signal. If the per-symbol set contrast collapses, the gate fails -- exactly as
the warmth carrier did.
"""

import numpy as np

RNG = np.random.default_rng(20260612)

# ---- analyzer mirror (identical to analyze_cache_hologram_matched_nulls.py) ----
# The .holo schedule has 4 MODES, each a distinct family of "slots". In the
# warmth carrier slots == cache LINES. In the conflict carrier slots == cache
# SETS the receiver pre-fills and the sender targets. The analyzer geometry is
# identical: a 64-slot vector, 4 modes defined by disjoint slot subsets, a
# differential mode-contrast feature, and a raw-vector real-vs-pseudo floor.
MODES = ["basis", "rotation", "residual", "mini"]
MODE_SETS = {
    "basis":    {9, 10, 11, 12, 13, 14},
    "rotation": {16, 17, 18, 19, 20, 21, 22, 23},
    "residual": {24, 25, 26, 27},
    "mini":     {9, 10, 11, 12, 16, 17, 18, 19, 24, 25, 26, 27},
}
REAL_SLOTS = {
    "basis":    [9, 10, 11, 12, 13, 14],
    "rotation": [16, 17, 18, 19, 20, 21, 22, 23],
    "residual": [24, 25, 26, 27],
    "mini":     [9, 16, 24, 10, 17, 25, 11, 18, 26, 12, 19, 27],
}
PSEUDO_SLOTS = {
    "basis":    [33, 34, 35, 36, 37, 38],
    "rotation": [40, 41, 42, 43, 44, 45, 46, 47],
    "residual": [52, 53, 54, 55],
    "mini":     [33, 40, 52, 34, 41, 53, 35, 42, 54, 36, 43, 55],
}


def mode_contrast_features(vec):
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
    cs = np.stack([centroids[m] for m in MODES], axis=0)
    d = np.linalg.norm(feats[..., None, :] - cs[None, ...], axis=-1)
    return d.argmin(axis=-1)


# ---- generative CONFLICT-DISPLACEMENT model ----
def gen_conflict_vectors(n, target_sets, A_disp, sigma_w, drift_amp,
                         p_out, sigma_out, pedestal, contrast_shrink,
                         drift_mode="ramp"):
    """
    Returns (n, 64) per-set re-access latency vectors for the RECEIVER.

      A_disp        : displacement contrast (cycles SLOWER) on a targeted set,
                      i.e. the receiver's own line got evicted and it pays an
                      L3/mem miss on re-access. Measured ~37-49 cyc in 5.10D.
                      NOTE sign is +A on targeted sets (slower), opposite to the
                      warmth model's -A (faster); the analyzer is sign-agnostic
                      (differential + nearest-centroid).
      sigma_w       : per-(sample,set) white timing noise std.
      drift_amp     : common-mode (all-set) monotonic drift (thermal creep).
      p_out,sigma_out: heavy-tail timing spikes (scheduler/SMI hiccups).
      pedestal      : COMMON-MODE eviction pedestal added to ALL sets under
                      sustained co-access (sender footprint spreads). This is
                      REJECTED by the differential mode feature but it INFLATES
                      raw-vector noise and is the realistic homogenization analog
                      for conflict: it does not erase the per-symbol pattern, it
                      raises the floor everyone sits on.
      contrast_shrink: fraction of A_disp lost because sustained co-access
                       spreads the sender across non-targeted sets too, partially
                       evicting them as well -> shrinks targeted-vs-nontargeted
                       contrast. THIS is the conflict analog of homogenization.
                       Correlated across reps => NOT averageable (honest).
    """
    base = 270.0  # measured no-aggressor / compute-only median (cycles/touch)
    v = np.full((n, 64), base) + RNG.normal(0, sigma_w, size=(n, 64))
    A_eff = A_disp * (1.0 - contrast_shrink)
    if len(target_sets) > 0:
        ts = np.array(target_sets, dtype=int)
        v[:, ts] += A_eff  # targeted sets: receiver re-access is SLOWER (evicted)
    # common-mode eviction pedestal (homogenization-as-pedestal, differential-rejected)
    v += pedestal
    # common-mode drift
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
    n = (vecs.shape[0] // N) * N
    return vecs[:n].reshape(-1, N, 64).mean(axis=1)


def lockin_demod(sym_vecs, ref_vecs, N):
    """Lock-in: targeted-set probe minus a CONTROL probe where the receiver
    pre-fills the sets but the sender targets a DIFFERENT family (the measured
    'different_address' control, ~293 cyc). Subtracting the control removes the
    common-mode pedestal + drift to first order and averages white noise."""
    nsym = (sym_vecs.shape[0] // N) * N
    nref = (ref_vecs.shape[0] // N) * N
    g = min(nsym, nref) // N
    s = sym_vecs[: g * N].reshape(g, N, 64).mean(axis=1)
    r = ref_vecs[: g * N].reshape(g, N, 64).mean(axis=1)
    return (s - r)


# ---- experiment runners (mirror the analyzer train/test split) ----
def make_mode_dataset(n_per_mode, family, params, N=1, strategy="single"):
    slot_table = REAL_SLOTS if family == "real" else PSEUDO_SLOTS
    feats_by_mode_test = {}
    centroids = {}
    for mi, m in enumerate(MODES):
        wl = slot_table[m]
        ntot = n_per_mode * 2 * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        raw = gen_conflict_vectors(ntot, wl, **params)
        if strategy == "single":
            agg = raw
        elif strategy == "plain":
            agg = block_average(raw, N)
        elif strategy == "lockin":
            # control = sender targets a DIFFERENT family's sets (measured ctrl)
            ctrl_family = MODES[(mi + 1) % len(MODES)]
            ref = gen_conflict_vectors(ntot, slot_table[ctrl_family], **params)
            agg = lockin_demod(raw, ref, N)
        half = agg.shape[0] // 2
        train, test = agg[:half], agg[half:]
        centroids[m] = mode_contrast_features(train).mean(axis=0)
        feats_by_mode_test[m] = mode_contrast_features(test)
    correct = 0
    total = 0
    for mi, m in enumerate(MODES):
        preds = nearest_centroid_predict(feats_by_mode_test[m], centroids)
        correct += (preds == mi).sum()
        total += len(preds)
    return correct / total


def make_rvp_floor(n_per_mode, params, N=1, strategy="single"):
    floors = []
    for m in MODES:
        rl, pl = REAL_SLOTS[m], PSEUDO_SLOTS[m]
        ntot = n_per_mode * 2 * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        rraw = gen_conflict_vectors(ntot, rl, **params)
        praw = gen_conflict_vectors(ntot, pl, **params)
        if strategy == "single":
            ragg, pagg = rraw, praw
        elif strategy == "plain":
            ragg, pagg = block_average(rraw, N), block_average(praw, N)
        elif strategy == "lockin":
            refr = gen_conflict_vectors(ntot, [], **params)
            refp = gen_conflict_vectors(ntot, [], **params)
            ragg = lockin_demod(rraw, refr, N)
            pagg = lockin_demod(praw, refp, N)
        hr, hp = ragg.shape[0] // 2, pagg.shape[0] // 2
        cr = ragg[:hr].mean(axis=0)
        cp = pagg[:hp].mean(axis=0)
        rt, pt = ragg[hr:], pagg[hp:]
        dr_r = np.linalg.norm(rt - cr, axis=1) <= np.linalg.norm(rt - cp, axis=1)
        dp_p = np.linalg.norm(pt - cp, axis=1) < np.linalg.norm(pt - cr, axis=1)
        acc = (dr_r.sum() + dp_p.sum()) / (len(rt) + len(pt))
        floors.append(acc)
    return min(floors)


def matched_null_wrong(n_per_mode, params, N, strategy):
    """WRONG family: sender runs schedule for ACTUAL mode = (declared+1). The
    classifier must read the ACTUAL mode (gate: wrong_actual_match >= 0.60),
    NOT the declared label (gate: wrong_declared_match <= 0.20). This is the
    discriminating null that the warmth carrier FAILED (0.24, ~chance)."""
    slot_table = REAL_SLOTS
    centroids = {}
    for mi, m in enumerate(MODES):
        ntot = n_per_mode * 2 * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        raw = gen_conflict_vectors(ntot, slot_table[m], **params)
        if strategy == "lockin":
            ctrl_family = MODES[(mi + 1) % len(MODES)]
            ref = gen_conflict_vectors(ntot, slot_table[ctrl_family], **params)
            agg = lockin_demod(raw, ref, N)
        else:
            agg = block_average(raw, N) if strategy == "plain" else raw
        centroids[m] = mode_contrast_features(agg).mean(axis=0)
    actual_match = 0
    declared_match = 0
    total = 0
    for mi, m in enumerate(MODES):
        actual = (mi + 1) % len(MODES)
        ntot = n_per_mode * (N if strategy != "single" else 1) * (2 if strategy == "lockin" else 1)
        raw = gen_conflict_vectors(ntot, slot_table[MODES[actual]], **params)
        if strategy == "lockin":
            ctrl_family = MODES[(actual + 1) % len(MODES)]
            ref = gen_conflict_vectors(ntot, slot_table[ctrl_family], **params)
            agg = lockin_demod(raw, ref, N)
        else:
            agg = block_average(raw, N) if strategy == "plain" else raw
        feats = mode_contrast_features(agg)
        preds = nearest_centroid_predict(feats, centroids)
        actual_match += (preds == actual).sum()
        declared_match += (preds == mi).sum()
        total += len(preds)
    return actual_match / total, declared_match / total


def run_regime(label, params, Ns):
    print(f"\n# {label}", flush=True)
    print(f"{'N':>5} {'plain_mode':>11} {'lockin_mode':>12} {'lockin_rvp':>11} "
          f"{'wrong_act':>10} {'wrong_dec':>10}", flush=True)
    res = {}
    for N in Ns:
        npm = max(150, 1200 // N)
        pm = make_mode_dataset(npm, "real", params, N, "plain")
        lm = make_mode_dataset(npm, "real", params, N, "lockin")
        lr = make_rvp_floor(npm, params, N, "lockin")
        wa, wd = matched_null_wrong(npm, params, N, "lockin")
        res[N] = (pm, lm, lr, wa, wd)
        print(f"{N:>5} {pm:>11.3f} {lm:>12.3f} {lr:>11.3f} {wa:>10.3f} {wd:>10.3f}", flush=True)
    return res


def main():
    np.set_printoptions(precision=4, suppress=True)
    Ns = [1, 4, 16, 64]

    # ---- Noise model anchored to MEASURED 5.10D conflict physics ----
    # base 270 cyc; same-set ~319 => A_disp ~ 49 cyc raw contrast.
    # sigma_w: per-touch timing CV. 5.10D medians were stable; the per-sample
    # spread on this box is dominated by scheduler jitter. We set sigma_w so the
    # single-shot per-set SNR ~ A_disp/sigma_w ~ 49/35 ~ 1.4 (conservative; the
    # measured family-median effect was much cleaner, p=0.0005, but a single
    # un-averaged per-set sample is noisier than a family median).
    base_noise = dict(sigma_w=35.0, drift_amp=14.0, p_out=0.02, sigma_out=40.0,
                      drift_mode="ramp")

    # REGIME 1: store-and-forward conflict (the sender fills sets in a short
    # burst, the receiver probes immediately; LOW sustained co-access).
    # contrast_shrink small, pedestal small. This is the regime the angle is
    # built for: displacement is read before homogenization spreads.
    p_sf = dict(A_disp=49.0, contrast_shrink=0.10, pedestal=6.0, **base_noise)
    res_sf = run_regime("STORE-AND-FORWARD conflict (burst-fill, immediate probe)", p_sf, Ns)

    # REGIME 2: moderate sustained co-access. Sender footprint spreads -> a real
    # pedestal on all sets and meaningful contrast shrink. This is the honest
    # mid-case: conflict degrades but, unlike warmth, does NOT collapse, because
    # the receiver still feels eviction of ITS OWN targeted sets.
    p_mid = dict(A_disp=49.0, contrast_shrink=0.45, pedestal=22.0, **base_noise)
    res_mid = run_regime("MODERATE SUSTAINED CO-ACCESS (footprint spreads)", p_mid, Ns)

    # REGIME 3: heavy sustained co-access (the warmth-killer regime). If conflict
    # behaves like warmth, contrast_shrink -> 0.70+ and it should FAIL too. This
    # is the falsification test: does conflict survive where occupancy died?
    p_heavy = dict(A_disp=49.0, contrast_shrink=0.70, pedestal=40.0, **base_noise)
    res_heavy = run_regime("HEAVY SUSTAINED CO-ACCESS (warmth-killer regime)", p_heavy, Ns)

    # REGIME 4: FALSIFICATION STRESS. If the conflict contrast almost fully
    # collapses (shrink 0.85) AND the per-touch noise is 1.5x worse than measured
    # AND the displacement is only 30 cyc (low end of 5.10D), does it still pass?
    # This is the honest worst-case the box might present.
    stress_noise = dict(sigma_w=52.0, drift_amp=20.0, p_out=0.04, sigma_out=60.0,
                        drift_mode="ramp")
    p_stress = dict(A_disp=30.0, contrast_shrink=0.85, pedestal=55.0, **stress_noise)
    res_stress = run_regime("FALSIFICATION STRESS (30cyc contrast, 0.85 shrink, 1.5x noise)",
                            p_stress, Ns)

    def first_cross(res, idx, thr):
        for N in Ns:
            if res[N][idx] >= thr:
                return N
        return None

    def last_below(res, idx, thr):
        return max(res[N][idx] for N in Ns) <= thr

    print("\n# GATE CROSSINGS (the binding gates)")
    for name, res in (("store_fwd", res_sf), ("moderate", res_mid), ("heavy", res_heavy),
                      ("stress", res_stress)):
        mode_n = first_cross(res, 1, 0.60)     # lockin mode acc >= 0.60
        rvp_n = first_cross(res, 2, 0.95)      # rvp floor >= 0.95
        wa_n = first_cross(res, 3, 0.60)       # wrong_actual >= 0.60
        wd_max = max(res[N][4] for N in Ns)    # wrong_declared (want <= 0.20)
        print(f"  [{name:>9}] mode>=0.60 @N={mode_n}  rvp>=0.95 @N={rvp_n}  "
              f"wrong_act>=0.60 @N={wa_n}  wrong_dec_max={wd_max:.3f}(<=0.20?)")


if __name__ == "__main__":
    main()
