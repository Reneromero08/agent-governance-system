"""
no_smuggle_gate.py - the reusable instrument for the next stage of the Phase 6
quadrature campaign.

A "quadrature-synthesis operation" O is any map that, given an instance, produces a
feature vector intended to expose the orientation bit b_orient = 1[d < N/2]. The
crossing the SPEC (Sec 1B/1C) is looking for is an O that:
   (i) RAISES the held-out orientation-bit AUC above chance, AND
  (ii) does so as a PURE FUNCTION OF PUBLIC DATA - it must not read d.

gate(O) measures both and renders one of three verdicts:

  PASS_CROSSING : AUC lifted above chance using public-only data, with O verified
                  d-invariant (swapping d <-> N-d while holding public data fixed
                  does not change O's output). A genuine quadrature synthesis.

  FAIL_SMUGGLE  : AUC lifted above chance, BUT O's output changes when d is swapped
                  to N-d at fixed public data. O only worked because it secretly
                  read d (or the hidden sin channel). This is the trap the campaign
                  must catch.

  FAIL_CHANCE   : AUC is at chance. O is a useless (even) transform; it manufactured
                  no bit. (This is the EXPECTED, honest outcome for any pure even
                  transform - the equivariance theorem of SPEC 1B.1.)

The d-invariance audit is exact and is the heart of the instrument: we hand O two
hidden states (d and N-d) that produce IDENTICAL public data, and require byte-equal
output. A smuggling O fails this even before any classifier is trained.

O signature (a single, explicit contract so the next stage can plug in candidates):
    O(instance) -> 1D float feature vector
where `instance` is a dict with keys:
    "k"       : int array, public frequencies
    "b"       : float array, public noisy bits (E[b]=cos(2 pi k d/N))
    "N"       : int, modulus
    "d"       : int, the HIDDEN secret (present ONLY so the gate can audit smuggling;
                a non-smuggling O must never read it)
ASCII only. All RNGs seeded by caller.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import construction as C


# ---------------------------------------------------------------------------
# Instance assembly
# ---------------------------------------------------------------------------
def make_instance(n, d, rng):
    """Assemble one public instance. The hidden d is carried for the smuggle audit
    only; public consumers must use k, b, N exclusively."""
    N = 1 << n
    M = C.M_for(n)
    k, b = C.coset_samples(N, d, M, rng)
    return {"k": k, "b": b, "N": N, "d": int(d % N), "n": n}


def folded_instance(inst):
    """Return a copy of inst with the SAME public data (k,b) but the hidden state
    swapped to N-d. Because the public data is a function of cos(2 pi k d / N) which
    is fold-invariant, the public part is identically distributed; we keep the SAME
    realized (k,b) so the ONLY thing that changes is the hidden label d -> N-d.
    A pure-public O must give identical output on inst and folded_instance(inst)."""
    N = inst["N"]
    out = dict(inst)
    out["d"] = int((N - inst["d"]) % N)
    return out


# ---------------------------------------------------------------------------
# AUC harness (cross-validated, held-out)
# ---------------------------------------------------------------------------
def _held_out_auc(X, y, seed):
    """5-fold stratified held-out AUC with a logistic probe on top of O's features.
    The probe is deliberately simple: the gate asks whether the FEATURES carry the
    bit, not whether a heroic classifier can. (The full classifier battery lives in
    fold_audit.py; the gate only needs a faithful linear read.)"""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
def _shuffle_null_auc(X, y, seed, n_shuffles=30):
    """Calibrate the chance ceiling for THIS dataset by re-running the held-out AUC
    on label-shuffled copies. Returns the 95th-percentile shuffled AUC. Any real
    held-out AUC at or below this is statistically indistinguishable from chance for
    this instance count and feature dimension - this is what makes the gate robust to
    finite-sample AUC variance (a fold-invariant O with delta==0 cannot beat its own
    shuffle null, by construction)."""
    rng = np.random.default_rng(seed + 777)
    nulls = []
    for _ in range(n_shuffles):
        yp = rng.permutation(y)
        nulls.append(_held_out_auc(X, yp, int(rng.integers(1 << 30))))
    return float(np.percentile(nulls, 95)), float(np.mean(nulls))


def gate(O, n, n_instances=400, seed=0, invariance_tol=1e-9, n_shuffles=30):
    """Audit a quadrature-synthesis operation O.

    Verdict logic (two independent axes):
      AXIS 1  d-invariance audit (EXACT): hand O the hidden states d and N-d that
              produce IDENTICAL public data; max_fold_delta is the max abs change in
              O's output. delta == 0 proves O is a pure function of public data.
      AXIS 2  orientation-bit signal: held-out AUC, calibrated against a per-dataset
              LABEL-SHUFFLE null (95th pct) so finite-sample AUC variance cannot fake
              a crossing. above_chance := auc > shuffle_null_95.

    Returns a dict with:
      auc, shuffle_null_95, above_chance, reads_d, max_fold_delta, verdict
        PASS_CROSSING : above_chance AND not reads_d (genuine public quadrature)
        FAIL_SMUGGLE  : above_chance AND reads_d     (only worked by reading d)
        FAIL_CHANCE   : not above_chance             (manufactured no bit)
    """
    rng = np.random.default_rng(seed)
    N = 1 << n

    feats, labels = [], []
    max_fold_delta = 0.0
    for _ in range(n_instances):
        d = C.sample_secret(N, rng)
        inst = make_instance(n, d, rng)

        f = np.asarray(O(inst), dtype=float).ravel()
        feats.append(f)
        labels.append(C.orientation_bit(d, N))

        # --- smuggle audit: same public data, hidden state folded ---
        f_folded = np.asarray(O(folded_instance(inst)), dtype=float).ravel()
        if f.shape == f_folded.shape:
            delta = float(np.max(np.abs(f - f_folded))) if f.size else 0.0
        else:
            delta = np.inf  # shape change is itself d-dependence
        max_fold_delta = max(max_fold_delta, delta)

    X = np.asarray(feats, dtype=float)
    y = np.asarray(labels, dtype=int)
    auc = _held_out_auc(X, y, seed)
    null95, null_mean = _shuffle_null_auc(X, y, seed, n_shuffles)

    reads_d = max_fold_delta > invariance_tol
    above_chance = auc > null95              # beats this dataset's own shuffle ceiling

    if above_chance and not reads_d:
        verdict = "PASS_CROSSING"
    elif above_chance and reads_d:
        verdict = "FAIL_SMUGGLE"
    else:
        verdict = "FAIL_CHANCE"

    return {
        "verdict": verdict,
        "auc": auc,
        "shuffle_null_95": null95,
        "shuffle_null_mean": null_mean,
        "reads_d": bool(reads_d),
        "max_fold_delta": max_fold_delta,
        "above_chance": bool(above_chance),
        "n": n,
        "n_instances": n_instances,
    }


# ---------------------------------------------------------------------------
# SELF-TEST operations (the two mandatory cases)
# ---------------------------------------------------------------------------
def O_cheat_reads_d(inst):
    """KNOWN-CHEATING O. It reads the hidden d directly to emit the orientation bit.
    The gate MUST flag this as FAIL_SMUGGLE: its output flips under the fold."""
    N = inst["N"]
    return np.array([float(C.orientation_bit(inst["d"], N))])


def O_cheat_reads_sin(inst):
    """KNOWN-CHEATING O, subtler: reads the ODD / quadrature channel sin(2 pi k d / N)
    from the HIDDEN d. That channel is exactly what is absent from public data, so this
    is smuggling. We evaluate the odd channel on a few FIXED low frequencies of the
    hidden d (k=1,2,3); each sin(2 pi k d / N) flips sign under d -> N-d, so it both
    (a) lifts the orientation-bit AUC well above chance and (b) is caught by the exact
    d-invariance audit. The gate MUST therefore report FAIL_SMUGGLE."""
    N = inst["N"]
    d = inst["d"]
    return np.array([float(np.sin(2 * np.pi * j * d / N)) for j in (1, 2, 3)])


def O_useless_even(inst):
    """KNOWN-USELESS O: an arbitrary nonlinear transform of the PUBLIC cosine data
    only (a function of (k,b)). By the equivariance theorem it is fold-invariant and
    carries zero orientation information. The gate MUST report FAIL_CHANCE and
    max_fold_delta == 0 (it never touches d)."""
    k = inst["k"]
    b = inst["b"]
    N = inst["N"]
    # moments + a few even Fourier lifts of the public score on fixed probe points
    probes = np.linspace(0, N / 2, 8, endpoint=False)
    feats = [float(np.mean(b)), float(np.std(b)), float(np.mean(b * np.cos(2 * np.pi * k / N)))]
    for x in probes:
        feats.append(np.tanh(C.score(k, b, x, N) / max(len(b), 1)))
    return np.array(feats)
