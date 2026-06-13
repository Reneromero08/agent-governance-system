"""
scaling.py - the DELIVERABLE: how does the chiral superradiant array size M(n) needed to
resolve the orientation scale with n, and does the COLLECTIVE chiral coupling beat the
single-copy conjugate read (B1) or the Kuperberg sieve?

Measures, per n in {4,6,8,10,12,...}:
  - chiral-collective orientation AUC vs M (the array-level chiral handedness read),
  - independent single-copy B1 AUC vs M (the control: sum of per-copy conjugate quadratures),
  - M*(n) = smallest M reaching AUC target for each, with multi-seed bands,
  - the one-shot SHOT-NOISE AUC (a physical collective detection, ~M/2 photons),
  - structured frequency sets (dyadic ladder, contiguous matched) with the RESOURCE caveat,
  - controls: achiral (mirror-blind, AUC 0.5 exactly), d-locked homodyne cheat (FAIL_SMUGGLE),
    useless-even (chance),
  - fits of log2 M*(n) against n (exp) and sqrt(n) (subexp).

NO-SMUGGLE: every array is FIXED and d-INDEPENDENT (k0, channel, label set chosen without d);
orientation labels are used ONLY to score AUC (supervised calibration), never inside the
operator. The d-locked cheat is the positive control that MUST be flagged.

ASCII only. RNGs seeded by caller. Reuses construction.py, coset_array.py, chiral_engine.py.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "fold_audit")
for _p in (FOLD, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import coset_array as A


def _auc(y, s):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    a = roc_auc_score(y, np.asarray(s, dtype=float))
    return float(max(a, 1.0 - a))   # orientation sign is a calibration convention


# ===========================================================================
# random-label scaling: chiral collective vs independent single-copy (B1)
# ===========================================================================
def random_label_scaling(ns, k0, target, seeds, n_inst, M_mults):
    """For each n: sweep M = mult * N (random labels, the honest oracle). Record chiral
    collective AUC and independent B1 AUC, averaged over seeds, and M*(n) for each."""
    out = []
    for n in ns:
        N = 1 << n
        Ms = sorted(set(max(8, int(m * N)) for m in M_mults))
        curve = []
        Mstar_ch = Mstar_in = None
        for M in Ms:
            ch_aucs, in_aucs = [], []
            for sd in seeds:
                rng = np.random.default_rng(sd + 977 * n + M)
                Tc, Ti, ys = [], [], []
                for _ in range(n_inst):
                    d = C.sample_secret(N, rng)
                    k = A.freq_set("random", M, n, rng)
                    Tc.append(A.chiral_statistic_fast(k, d, N, k0))
                    Ti.append(A.independent_statistic_fast(k, d, N))
                    ys.append(C.orientation_bit(d, N))
                ch_aucs.append(_auc(ys, Tc))
                in_aucs.append(_auc(ys, Ti))
            ach, sch = float(np.mean(ch_aucs)), float(np.std(ch_aucs))
            ain, sin_ = float(np.mean(in_aucs)), float(np.std(in_aucs))
            curve.append({"M": M, "M_over_N": M / N,
                          "chiral_auc": ach, "chiral_auc_std": sch,
                          "indep_auc": ain, "indep_auc_std": sin_})
            if Mstar_ch is None and ach >= target:
                Mstar_ch = M
            if Mstar_in is None and ain >= target:
                Mstar_in = M
        out.append({"n": n, "N": N, "curve": curve,
                    "Mstar_chiral": Mstar_ch, "Mstar_indep": Mstar_in,
                    "Mstar_chiral_over_N": (Mstar_ch / N if Mstar_ch else None),
                    "Mstar_indep_over_N": (Mstar_in / N if Mstar_in else None)})
    return out


# ===========================================================================
# one-shot SHOT-NOISE model: a physical collective detection of the chiral asymmetry
# ===========================================================================
def one_shot_noise(ns, k0, target, seed, n_inst, M_mults):
    """One collective shot: load M coset states, ~M/2 photons carry the chiral asymmetry
    a(d) = T(d)/R_tot(d); the detected asymmetry is a(d) + Normal(0, sqrt((1-a^2)/(M/2))).
    Superradiant enhancement (a grows with structure) and shot noise (~1/sqrt M) both scale
    with M, so this is the honest physical M(n). Random labels."""
    out = []
    for n in ns:
        N = 1 << n
        Ms = sorted(set(max(8, int(m * N)) for m in M_mults))
        curve = []
        Mstar = None
        for M in Ms:
            rng = np.random.default_rng(seed + 613 * n + M)
            obs, ys = [], []
            for _ in range(n_inst):
                d = C.sample_secret(N, rng)
                k = A.freq_set("random", M, n, rng)
                T, R, a = A.chiral_asymmetry_fast(k, d, N, k0)   # O(M), no M x M matrix
                nph = max(1.0, M * 0.5)
                x = a + rng.normal(0.0, np.sqrt(max(1e-6, 1.0 - a * a) / nph))
                obs.append(x)
                ys.append(C.orientation_bit(d, N))
            au = _auc(ys, obs)
            curve.append({"M": M, "M_over_N": M / N, "oneshot_auc": au})
            if Mstar is None and au >= target:
                Mstar = M
        out.append({"n": n, "N": N, "curve": curve, "Mstar_oneshot": Mstar,
                    "Mstar_oneshot_over_N": (Mstar / N if Mstar else None)})
    return out


# ===========================================================================
# structured frequency sets (dyadic ladder, contiguous matched) + resource caveat
# ===========================================================================
def structured_sets(ns, k0, seeds, n_inst):
    """Chiral-collective AND independent-B1 AUC for the dyadic ladder and the contiguous
    matched comb (M = 4n emitters). These read orientation easily -- but ONLY because they
    contain SMALL labels (k=1,2,... : the B0/B1 resource). Under the standard random-label
    oracle, obtaining chosen small labels costs the Kuperberg sieve (2^{O(sqrt n)}); so the
    total price of the structured read is the sieve, not O(1). Flagged accordingly."""
    out = []
    for n in ns:
        N = 1 << n
        row = {"n": n, "N": N, "M": 4 * n}
        for kind in ("dyadic", "matched"):
            ch_a, in_a = [], []
            for sd in seeds:
                rng = np.random.default_rng(sd + 401 * n + hash(kind) % 1000)
                k = A.freq_set(kind, 4 * n, n, rng)
                G = A.array_decay_matrix(k, n, k0, 1.0)
                Tc, Ti, ys = [], [], []
                for _ in range(n_inst):
                    d = C.sample_secret(N, rng)
                    Tc.append(A.readouts(G, k, d, N)[1])
                    Ti.append(A.independent_statistic_fast(k, d, N))
                    ys.append(C.orientation_bit(d, N))
                ch_a.append(_auc(ys, Tc))
                in_a.append(_auc(ys, Ti))
            row[kind + "_chiral_auc"] = float(np.mean(ch_a))
            row[kind + "_indep_auc"] = float(np.mean(in_a))
        row["resource_caveat"] = ("small labels cost the Kuperberg sieve 2^{O(sqrt n)} under "
                                  "the standard random-label oracle: structured read re-prices to subexp")
        out.append(row)
    return out


# ===========================================================================
# controls (no-smuggle)
# ===========================================================================
def controls(ns, k0, seed, n_inst):
    """achiral (mirror-symmetric, AUC 0.5 EXACTLY -> the structural blind control proving a
    mirror-symmetric operator cannot read the orientation); d-locked homodyne (reads
    sign(-sin(2 pi d/N)) using d -> AUC 1.0 -> FAIL_SMUGGLE); useless-even (reads only |p_j|^2
    = 1, no phase -> chance). Random labels."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 71 * n)
        k = A.freq_set("random", max(8 * n, 32), n, rng)
        Ga = A.array_decay_matrix(k, n, k0, 0.0)     # achiral
        ach, cheat, even, ys = [], [], [], []
        for _ in range(n_inst):
            d = C.sample_secret(N, rng)
            ach.append(A.readouts(Ga, k, d, N)[1])            # achiral chiral-stat (==0)
            cheat.append(-np.sin(2 * np.pi * d / N))          # CHEAT: homodyne LO locked to d
            even.append(float(np.sum(np.abs(A.coset_phases(k, d, N)) ** 2)))  # |p|^2 = M, even
            ys.append(C.orientation_bit(d, N))
        out.append({
            "n": n, "N": N,
            "achiral_auc": _auc(ys, ach), "achiral_verdict": "BLIND (mirror-symmetric)",
            "d_locked_homodyne_auc": _auc(ys, cheat), "d_locked_verdict": "FAIL_SMUGGLE (uses d)",
            "useless_even_auc": _auc(ys, even), "useless_even_verdict": "FAIL_CHANCE",
        })
    return out


# ===========================================================================
# fits
# ===========================================================================
def fit_classes(ns, Mstars):
    pts = [(n, M) for n, M in zip(ns, Mstars) if M]
    if len(pts) < 3:
        return {"note": "too few resolved points to fit", "ns": [p[0] for p in pts],
                "Mstars": [p[1] for p in pts]}
    nn = np.array([p[0] for p in pts], dtype=float)
    y = np.log2(np.array([p[1] for p in pts], dtype=float))

    def fit(x):
        Amat = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
        pred = Amat @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2}

    f_n = fit(nn)
    f_sqrt = fit(np.sqrt(nn))
    f_logn = fit(np.log2(nn))
    better = max([("exp(n)", f_n["r2"]), ("subexp(sqrt n)", f_sqrt["r2"]),
                  ("poly(log-log)", f_logn["r2"])], key=lambda t: t[1])[0]
    return {"fit_log2M_vs_n": f_n, "fit_log2M_vs_sqrtn": f_sqrt, "fit_log2M_vs_logn": f_logn,
            "better_fit": better, "ns": [int(p[0]) for p in pts],
            "Mstars": [int(p[1]) for p in pts]}
