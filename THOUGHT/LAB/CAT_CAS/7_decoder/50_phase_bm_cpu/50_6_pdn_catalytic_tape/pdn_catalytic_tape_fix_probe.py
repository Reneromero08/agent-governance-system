#!/usr/bin/env python3
"""
pdn_catalytic_tape_fix_probe.py  (VET-ONLY probe, no hardware)

Tests the SINGLE fix to the existing pdn_catalytic_tape_sim.py:
  the matched-null pseudo-reject gate failed (~0.76 vs 0.95 gate) at ALL SNR
  including 60 -> NOT an SNR problem. Root cause confirmed: the H8 codebook is
  permutation-degenerate (all-ones row maps to itself under EVERY scramble;
  rows 1,2,3 are mutual permutations), so a "scrambled schedule" lands back ON
  the codebook ~25%+ of the time and cannot be rejected.

FIX (this probe):
  (1) DISTINCT-WEIGHT codebook: 4 codewords over NBIN bins with DIFFERENT numbers
      of sign-flipped bins -> permutation-inequivalent (a scramble can NEVER turn
      one codeword into another; self-collision prob ~ 1/C(NBIN,w) << 0.05).
  (2) SCALE-INVARIANT, BIN-POSITION-SENSITIVE discriminator: de-rotate z by the
      recovered global phase, L2-normalize (kills the 1/f amplitude drift), and
      classify on the FULL NBIN-dim normalized vector (keeps bin positions that
      |CODE@z| threw away). Matched-null statistic = energy concentration on the
      best canonical codeword rho = max_k |<zhat,h_k>|^2 ; accept iff rho>=thresh.

Identical realistic noise to the parent sim: correlated 1/f amp+phase drift,
bin-to-bin crosstalk, cross-process TSC jitter, additive SNR. ASCII only.
"""
import numpy as np

MODES = 4
NBIN = 12
TRIALS = 320
PHASE_LEVELS = 8


def make_codebook(seed=7):
    """4 +/-1 codewords over NBIN bins with DISTINCT flip-weights {4,5,6,7}
    (permutation-inequivalent), random-searched for max-min pairwise Hamming."""
    rng = np.random.default_rng(seed)
    weights = [4, 5, 6, 7]            # distinct -> no cross-codeword perm collision
    best, best_d = None, -1
    for _ in range(4000):
        cw = []
        for w in weights:
            v = np.ones(NBIN)
            v[rng.choice(NBIN, w, replace=False)] = -1.0
            cw.append(v)
        C = np.array(cw)
        # min pairwise Hamming distance
        d = min(int(np.sum(C[i] != C[j]))
                for i in range(MODES) for j in range(i + 1, MODES))
        if d > best_d:
            best_d, best = d, C.copy()
    return best, best_d


CODE, MINHAM = make_codebook()
HN = CODE / np.sqrt(NBIN)            # unit-norm canonical directions


def lockin_vector(rng, codeword, theta, snr, drift_amp, drift_phase, xtalk, jitter):
    z = codeword.astype(complex) * np.exp(1j * theta)
    z = z * (1.0 + drift_amp * rng.standard_normal()) * np.exp(1j * drift_phase * rng.standard_normal())
    if xtalk > 0:
        M = np.eye(NBIN) + xtalk * (rng.standard_normal((NBIN, NBIN)) +
                                    1j * rng.standard_normal((NBIN, NBIN)))
        z = M @ z
    z = z * np.exp(1j * jitter * rng.standard_normal())
    sigma = 1.0 / max(snr, 1e-9)
    z = z + sigma * (rng.standard_normal(NBIN) + 1j * rng.standard_normal(NBIN))
    return z


def feats(z):
    """Scale-invariant, bin-position-sensitive features.
    Returns (zhat_real_imag concat (2*NBIN), rho concentration, proj4)."""
    # de-rotate by the dominant global phase (the shared-TSC carrier phase)
    g = np.angle(np.sum(z))
    zr = z * np.exp(-1j * g)
    n = np.linalg.norm(zr) + 1e-12
    zhat = zr / n
    proj = np.abs(CODE @ z)                      # legacy 4-dim (for reference)
    corr = np.abs(HN @ zhat)                     # |<zhat,h_k>| per mode, in [0,1]
    rho = float(np.max(corr) ** 2)               # energy concentration on best cw
    fvec = np.concatenate([zhat.real, zhat.imag])  # full normalized vector
    return fvec, rho, proj, int(np.argmax(corr)), float(np.angle(CODE[int(np.argmax(corr))] @ z))


def gen_family(rng, kind, snr, p):
    actual = int(rng.integers(0, MODES))
    theta = 2 * np.pi * int(rng.integers(0, PHASE_LEVELS)) / PHASE_LEVELS
    if kind == 'real':
        declared = actual; cw = CODE[actual]
    elif kind == 'wrong':
        declared = (actual + int(rng.integers(1, MODES))) % MODES; cw = CODE[actual]
    else:  # pseudo: codeword driven through a WRONG key permutation (decoy schedule)
        declared = int(rng.integers(0, MODES))
        cw = CODE[actual][rng.permutation(NBIN)]
    z = lockin_vector(rng, cw, theta, snr, **p)
    fvec, rho, proj, mhat, theta_hat = feats(z)
    return dict(fvec=fvec, rho=rho, proj=proj, mhat=mhat,
                actual=actual, declared=declared, theta=theta, theta_hat=theta_hat)


def circ_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ca, cb = np.exp(1j * a), np.exp(1j * b)
    return float(np.real(np.sum(ca * np.conj(cb))) / len(a))


def eval_seed(seed, snr):
    rng = np.random.default_rng(seed)
    p = dict(drift_amp=0.10, drift_phase=0.08, xtalk=0.05, jitter=0.06)
    fam = {k: [gen_family(rng, k, snr, p) for _ in range(TRIALS)]
           for k in ('real', 'pseudo', 'wrong')}

    # ---- mode centroids on EVEN real trials, full normalized vector ----
    train = {m: [] for m in range(MODES)}
    for i, f in enumerate(fam['real']):
        if i % 2 == 0:
            train[f['declared']].append(f['fvec'])
    cent = {m: (np.mean(train[m], axis=0) if train[m] else np.zeros(2 * NBIN))
            for m in range(MODES)}

    def predict(fv):
        return int(min(range(MODES), key=lambda m: np.linalg.norm(fv - cent[m])))

    # ---- real-mode accuracy (odd real trials) ----
    rc = rt = 0; bym = {m: [0, 0] for m in range(MODES)}
    for i, f in enumerate(fam['real']):
        if i % 2 == 1:
            rt += 1; ok = predict(f['fvec']) == f['declared']
            rc += ok; bym[f['declared']][1] += 1; bym[f['declared']][0] += ok
    real_acc = rc / max(rt, 1)
    real_floor = min((c / t if t else 0.0) for c, t in bym.values())

    # ---- wrong: reader must read ACTUAL physical mode, not declared label ----
    wa = wd = wt = 0
    for i, f in enumerate(fam['wrong']):
        if i % 2 == 1:
            wt += 1; pr = predict(f['fvec']); wa += (pr == f['actual']); wd += (pr == f['declared'])
    wrong_actual = wa / max(wt, 1); wrong_declared = wd / max(wt, 1)

    pdm = pt = 0
    for i, f in enumerate(fam['pseudo']):
        if i % 2 == 1:
            pt += 1; pdm += (predict(f['fvec']) == f['declared'])
    pseudo_declared = pdm / max(pt, 1)

    # ---- matched-null real-vs-pseudo via energy-concentration rho threshold ----
    # threshold trained on EVEN real rho (5th percentile) -> accept real, reject pseudo
    rho_real_tr = np.array([f['rho'] for i, f in enumerate(fam['real']) if i % 2 == 0])
    thr = float(np.percentile(rho_real_tr, 5))
    rvp_acc, rvp_rej = [], []
    for m in range(MODES):
        te_real = [f['rho'] for i, f in enumerate(fam['real'])
                   if i % 2 == 1 and f['declared'] == m]
        te_ps = [f['rho'] for i, f in enumerate(fam['pseudo'])
                 if i % 2 == 1 and f['mhat'] == m]   # pseudo binned by its apparent mode
        if not te_real or not te_ps:
            continue
        acc_real = np.mean([r >= thr for r in te_real])
        rej_ps = np.mean([r < thr for r in te_ps])
        rvp_acc.append((np.sum([r >= thr for r in te_real]) + np.sum([r < thr for r in te_ps]))
                       / (len(te_real) + len(te_ps)))
        rvp_rej.append(rej_ps)
    rvp_floor = min(rvp_acc) if rvp_acc else 0.0
    rej_floor = min(rvp_rej) if rvp_rej else 0.0

    # ---- relational phase recovery (differential cancels common 1/f drift) ----
    th_true = np.array([f['theta'] for f in fam['real']])
    th_hat = np.array([f['theta_hat'] for f in fam['real']])
    d_true, d_hat = np.diff(th_true), np.diff(th_hat)
    corr_true = circ_corr(d_hat, d_true)
    sh = np.random.default_rng(seed + 99).permutation(len(d_true))
    corr_null = circ_corr(d_hat, d_true[sh])

    return dict(real_acc=real_acc, real_floor=real_floor,
                wrong_actual=wrong_actual, wrong_declared=wrong_declared,
                pseudo_declared=pseudo_declared,
                rvp_floor=rvp_floor, rej_floor=rej_floor, rho_thr=thr,
                phase_true=corr_true, phase_null=corr_null)


def gates(a):
    return {
        'real_acc>=0.60': a['real_acc'] >= 0.60,
        'real_floor>=0.45': a['real_floor'] >= 0.45,
        'rvp_floor>=0.95': a['rvp_floor'] >= 0.95,
        'rej_floor>=0.95': a['rej_floor'] >= 0.95,
        'pseudo_decl<=0.35': a['pseudo_declared'] <= 0.35,
        'wrong_actual>=0.60': a['wrong_actual'] >= 0.60,
        'wrong_declared<=0.20': a['wrong_declared'] <= 0.20,
        'phase_recovered(>0.30)': (a['phase_true'] - a['phase_null']) > 0.30,
    }


def main():
    seeds = list(range(6))
    print(f"PDN catalytic-tape FIX probe: NBIN={NBIN} distinct-weight codebook, "
          f"min pairwise Hamming={MINHAM}")
    print("scale-invariant full-vector classifier + rho energy-concentration matched null\n")
    for snr in [2, 3, 5, 8, 15, 30, 60]:
        rows = [eval_seed(s, snr) for s in seeds]
        a = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        g = gates(a)
        tag = 'ALL_GATES_PASS' if all(g.values()) else 'partial'
        print(f"SNR_eff={snr:>3}  real_acc={a['real_acc']:.3f}  "
              f"rvp={a['rvp_floor']:.3f}  rej={a['rej_floor']:.3f}  "
              f"ps_decl={a['pseudo_declared']:.3f}  "
              f"wr_act={a['wrong_actual']:.3f} wr_decl={a['wrong_declared']:.3f}  "
              f"ph[T={a['phase_true']:+.2f} N={a['phase_null']:+.2f}]  => {tag}")
    print("\ngate detail at SNR_eff=8 (~7x BELOW proven isolation SNR 50-86):")
    rows = [eval_seed(s, 8) for s in seeds]
    a = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    for k, v in gates(a).items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
