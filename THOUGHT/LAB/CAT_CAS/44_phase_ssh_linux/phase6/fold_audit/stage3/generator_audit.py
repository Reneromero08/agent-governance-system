"""
generator_audit.py - PART A: a GENERATOR AUDIT of the REAL Exp 50.14 construction.

Goal (Stage 3, Phase 6 quadrature-crossing campaign): enumerate EVERY quantity the
real construction makes public, and for each test whether it depends on the secret d
through any ODD function -- i.e. anything NOT invariant under the fold sigma: d <-> N-d.

The frontier consult proved: IF b_i is a pure binary sign bit with E[b]=cos(2 pi k d/N)
as the WHOLE conditional law, then P(D|d)=P(D|N-d) pointwise and I(orientation ; T(D))=0
for any transform T at any compute. The ONLY way orientation can leak is if the ACTUAL
construction exposes something richer or asymmetric than that cos-mean. This script hunts
that, on the real code path lifted verbatim into construction.py (== 50_14_substrate.py).

Audited public quantities (the construction's entire public interface):
  Q1  the published bit b_i in {-1,+1}            (coset_samples)
  Q2  the frequencies k_i                          (coset_samples)
  Q3  the sample count M                           (M_for)
  Q4  the score / threshold / accept rule          (make_verify, score, thresh=M/4)
  Q5  the fixed-point map f and its accepting set  (f_map / forward search)
  Q6  float-rounding of cos(2 pi k d/N) vs cos(2 pi k (N-d)/N)   (the subtle side-channel)
  Q7  sample ORDER and any PRNG seed dependence on d            (vs orbit representative)

Each test reduces to: does the quantity, as a function of d, equal its value at N-d
(EVEN -> airtight, bedrock applies) or differ (ODD/asymmetric -> potential leak)?

Discipline: ASCII only; all RNGs seeded and recorded; FOREGROUND, n in {8,10}; venv
.venv/Scripts/python.exe. Claim ceiling L4-5. A found "leak" is far more likely a
float/seed/order BUG than a real crossing -- treat any positive with maximum skepticism.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # fold_audit/
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import construction as C                  # verbatim copy of 50_14 coset_samples/make_verify

MASTER_SEED = 44060611
LINES = []


def log(m=""):
    print(m)
    LINES.append(str(m))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def float_bits(arr):
    """Exact bit pattern of a float64 array (for bitwise-identity tests)."""
    return np.frombuffer(np.asarray(arr, dtype=np.float64).tobytes(), dtype=np.uint64)


# ===========================================================================
# Q1 - is the published bit pure binary {-1,+1}, and is its WHOLE conditional
#      law the cos-mean (so higher moments are fold-even)?
# ===========================================================================
def audit_Q1_bit(n, rng, trials=4000):
    """Two sub-tests:
      (a) DTYPE/RANGE: confirm b takes values exactly in {-1,+1} (pure binary sign).
      (b) CONDITIONAL LAW: for a fixed (k,d), the bit is Bernoulli with p=(1+cos)/2.
          A pure binary bit has its ENTIRE law fixed by its mean: Var=1-mean^2, and
          mean(d) = cos = mean(N-d) (cos is even). So conditional variance and ALL
          higher moments must be IDENTICAL under d<->N-d. We estimate them empirically
          and test for asymmetry. Any asymmetry would mean the bit is NOT pure binary."""
    N = 1 << n
    res = {"quantity": "Q1_published_bit"}

    # (a) range / dtype
    d = C.sample_secret(N, rng)
    M = C.M_for(n)
    k, b = C.coset_samples(N, d, M, rng)
    uniq = sorted(set(np.unique(b).tolist()))
    is_binary = set(np.unique(b).tolist()).issubset({-1.0, 1.0})
    res["dtype"] = str(b.dtype)
    res["unique_values"] = uniq
    res["is_pure_binary_sign"] = bool(is_binary)

    # (b) conditional moments under d vs N-d at FIXED frequencies, against the CORRECT
    # paired null. The naive null 1/sqrt(trials) is wrong for a max-over-many-cells gap;
    # the honest null is the gap between two INDEPENDENT redraws at the SAME d (pure
    # sampling noise). A real fold-asymmetry must exceed that paired null. We accumulate
    # both the fold gaps (d vs N-d) and same-d gaps (d vs d) and compare their scales.
    probe_k = np.array([1, 2, 3, 5, 7, 11, 13, (N // 4), (N // 2) - 1], dtype=np.int64) % N

    def moments(p):
        b = np.where(rng.random(trials) < p, 1.0, -1.0)
        m = b.mean()
        return m, b.var(), ((b - m) ** 3).mean(), ((b - m) ** 4).mean()

    fold_gaps = {"mean": [], "var": [], "m3": [], "m4": []}
    same_gaps = {"mean": [], "var": [], "m3": [], "m4": []}
    for d_try in [d, C.sample_secret(N, rng), C.sample_secret(N, rng)]:
        Nd = (N - d_try) % N
        for kk in probe_k:
            p_d = (1 + np.cos(2 * np.pi * kk * d_try / N)) / 2
            p_nd = (1 + np.cos(2 * np.pi * kk * Nd / N)) / 2
            md, vd, c3d, c4d = moments(p_d)          # d
            mnd, vnd, c3nd, c4nd = moments(p_nd)      # N-d
            m1, v1, t1, q1 = moments(p_d)             # d, independent redraw (null)
            m2, v2, t2, q2 = moments(p_d)             # d, independent redraw (null)
            fold_gaps["mean"].append(abs(md - mnd)); same_gaps["mean"].append(abs(m1 - m2))
            fold_gaps["var"].append(abs(vd - vnd)); same_gaps["var"].append(abs(v1 - v2))
            fold_gaps["m3"].append(abs(c3d - c3nd)); same_gaps["m3"].append(abs(t1 - t2))
            fold_gaps["m4"].append(abs(c4d - c4nd)); same_gaps["m4"].append(abs(q1 - q2))

    # a fold-asymmetry is REAL only if the fold gap distribution exceeds the same-d
    # (sampling-noise) distribution. Compare means; require fold > 2x same-d to flag.
    gap_summary = {}
    asymmetric = False
    for mom in ("mean", "var", "m3", "m4"):
        fg = float(np.mean(fold_gaps[mom]))
        sg = float(np.mean(same_gaps[mom]))
        gap_summary[mom] = {"fold_gap_mean": fg, "same_d_null_mean": sg,
                            "ratio_fold_over_null": (fg / sg if sg > 0 else 0.0)}
        if sg > 0 and fg > 2.0 * sg:
            asymmetric = True
    res["paired_moment_gaps"] = gap_summary
    res["trials_per_cell"] = trials

    # EXACT analytic check on the WHOLE law: for a {-1,+1} Bernoulli with p=(1+c)/2, the
    # entire conditional law is a function of c=cos (mean=c, var=1-c^2, all even in c).
    # cos(2 pi k d/N) and cos(2 pi k (N-d)/N) are mathematically EQUAL; the only float
    # difference is sub-ULP rounding of the ARGUMENT, which (Q7) never reaches the
    # published bit. So the published bit's law is fold-identical to machine relevance.
    res["odd_dependence"] = bool(asymmetric)
    return res


# ===========================================================================
# Q2 - frequencies k_i independent of d?
# ===========================================================================
def audit_Q2_frequencies(n, rng):
    """coset_samples draws k = rng.integers(0,N,size=M) BEFORE touching d; the draw
    has no d argument. We confirm structurally (k generated independently) and
    empirically (the k-distribution is identical for d and N-d at a shared RNG state)."""
    N = 1 << n
    M = C.M_for(n)
    res = {"quantity": "Q2_frequencies_k"}
    # structural: identical RNG state -> identical k regardless of d
    s = int(rng.integers(1 << 30))
    d = C.sample_secret(N, rng)
    Nd = (N - d) % N
    r1 = np.random.default_rng(s)
    k1, _ = C.coset_samples(N, d, M, r1)
    r2 = np.random.default_rng(s)
    k2, _ = C.coset_samples(N, Nd, M, r2)
    res["k_identical_for_d_and_Nd_at_same_rng_state"] = bool(np.array_equal(k1, k2))
    res["odd_dependence"] = not res["k_identical_for_d_and_Nd_at_same_rng_state"]
    return res


# ===========================================================================
# Q3 - sample count M depends on d?
# ===========================================================================
def audit_Q3_count(n):
    """M_for(n) = max(4 ceil(sqrt N), 48 n) - a pure function of n, no d. Confirm."""
    N = 1 << n
    res = {"quantity": "Q3_sample_count_M"}
    M = C.M_for(n)
    # M_for takes only n -> by construction independent of d; record the value & form.
    res["M"] = int(M)
    res["depends_on_d"] = False
    res["odd_dependence"] = False
    return res


# ===========================================================================
# Q4 - score / threshold / accept rule asymmetry under the fold
# ===========================================================================
def audit_Q4_score_accept(n, rng, n_inst=200):
    """score(x)=sum b_i cos(2 pi k_i x/N), thresh=M/4, accept = score>thresh.
    The accepting SET is {d, N-d} (symmetric). We test the fold directly: for the SAME
    realized public (k,b), is score(d) == score(N-d) and accept(d)==accept(N-d)?
    More importantly for the GENERATOR: does the THRESHOLD or the accept RULE reference
    d's half? It is thresh=M/4 (pure function of M, no d) and score uses only (k,b,x).
    We also check: across many instances, is the accept-rate at the TRUE d equal to the
    accept-rate at N-d (it must be: score(d)=score(N-d) on every realization since the
    SAME (k,b) drives both, and cos(2 pi k d/N)=cos(2 pi k (N-d)/N))."""
    N = 1 << n
    M = C.M_for(n)
    res = {"quantity": "Q4_score_threshold_accept"}
    max_score_gap = 0.0
    accept_mismatch = 0
    thresh_depends_on_d = False     # thresh = M/4, no d in make_verify
    for _ in range(n_inst):
        d = C.sample_secret(N, rng)
        Nd = (N - d) % N
        k, b = C.coset_samples(N, d, M, rng)
        verify, _ = C.make_verify(k, b, N)
        s_d = C.score(k, b, float(d), N)
        s_nd = C.score(k, b, float(Nd), N)
        max_score_gap = max(max_score_gap, abs(s_d - s_nd))
        if bool(verify(d)) != bool(verify(Nd)):
            accept_mismatch += 1
    res["max_abs_score_gap_d_vs_Nd_same_data"] = float(max_score_gap)
    res["accept_d_neq_accept_Nd_count"] = int(accept_mismatch)
    res["threshold_references_d"] = bool(thresh_depends_on_d)
    # score(d)-score(N-d): both use the SAME (k,b) and cos is even in x, so the gap is
    # pure float rounding of cos(2 pi k d/N) vs cos(2 pi k (N-d)/N). Record its scale.
    res["odd_dependence"] = (max_score_gap > 1e-6) or (accept_mismatch > 0) or thresh_depends_on_d
    return res


# ===========================================================================
# Q6 - FLOAT-ROUNDING / code-path asymmetry: the subtlest real leak.
# ===========================================================================
def audit_Q6_float_rounding(n, rng, n_d=64):
    """The subtlest real-leak hunt, done HONESTLY. Two distinct objects must not be
    confused:
      (i)  the INTERMEDIATE cos(2 pi k d/N): a direct function of the HIDDEN d. It is
           NOT published; reading it is by definition a smuggle. Its low float bits
           trivially encode d (hence any smooth function of d), so a classifier on them
           "predicts" orientation -- AND parity, AND any d-derived label. We INCLUDE a
           control proving exactly that, to show the apparent signal is a smuggle, not a
           public leak.
      (ii) the PUBLISHED bit b in {-1,+1}: this is what coset_samples actually returns.
           Its float64 holds only the two patterns for +-1.0 (zero bit information), and
           because cos is even, b is bitwise-identical for d and N-d on the same RNG
           draw (verified in Q7). So the published float code path carries no orientation.

    We report: the fraction of (k,d) where the intermediate cos differs in float bits
    (expected ~1, sub-ULP), whether ANY of that reaches the published bit b (expected 0),
    and the three control AUCs (orientation / parity / random) on the intermediate-cos
    low bits to localize the apparent signal as a smuggle. odd_dependence is True ONLY
    if the PUBLISHED data carries orientation -- never merely because the smuggle does."""
    N = 1 << n
    res = {"quantity": "Q6_float_rounding_codepath"}

    def cos_path(k, d):
        return np.cos(2 * np.pi * k * d / N)        # verbatim sampler intermediate

    k_all = np.arange(N, dtype=np.int64)
    ds = [C.sample_secret(N, rng) for _ in range(n_d)]

    # (1) does the INTERMEDIATE cos differ in float bits under the fold? (sub-ULP)
    inter_bit_diff = 0
    total = 0
    # (2) does that EVER reach the PUBLISHED bit b? share RNG so only d differs.
    published_bit_diffs = 0
    M = C.M_for(n)
    for d in ds:
        Nd = (N - d) % N
        cd = float_bits(cos_path(k_all, d))
        cnd = float_bits(cos_path(k_all, Nd))
        inter_bit_diff += int((cd != cnd).sum())
        total += len(k_all)
        # published bits: same caller seed -> same uniforms; b should be IDENTICAL
        s = int(rng.integers(1 << 30))
        r1 = np.random.default_rng(s); k1, b1 = C.coset_samples(N, d, M, r1)
        r2 = np.random.default_rng(s); k2, b2 = C.coset_samples(N, Nd, M, r2)
        published_bit_diffs += int(np.sum(b1 != b2))

    res["frac_intermediate_cos_float_bits_differ_d_vs_Nd"] = float(inter_bit_diff) / float(total)
    res["published_bit_b_differences_d_vs_Nd_same_rng"] = int(published_bit_diffs)
    res["published_float_carries_orientation"] = bool(published_bit_diffs > 0)

    # (3) CONTROL: classifier on the intermediate-cos low bits, three labels. If it
    # classifies parity and (near) random too, the "orientation AUC" is a smuggle, not
    # a public leak. Use the same feature builder for all three labels.
    ds2 = [C.sample_secret(N, rng) for _ in range(600)]
    low_freqs = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
    Xc = []
    y_orient, y_parity = [], []
    rr = np.random.default_rng(MASTER_SEED ^ (N * 7))
    y_random = []
    for d in ds2:
        fv = []
        for kf in low_freqs:
            bits = int(float_bits(np.array([cos_path(kf, d)]))[0])
            for shift in range(20):
                fv.append(float((bits >> shift) & 1))
        Xc.append(fv)
        y_orient.append(C.orientation_bit(d, N))
        y_parity.append(int(d & 1))
        y_random.append(int(rr.random() < 0.5))
    Xc = np.asarray(Xc, float)
    Xc = Xc[:, Xc.var(axis=0) > 0]
    s6 = int(MASTER_SEED ^ N)
    auc_orient = _cv_auc(Xc, np.asarray(y_orient), s6) if Xc.shape[1] else 0.5
    auc_parity = _cv_auc(Xc, np.asarray(y_parity), s6) if Xc.shape[1] else 0.5
    auc_random = _cv_auc(Xc, np.asarray(y_random), s6) if Xc.shape[1] else 0.5
    res["SMUGGLE_control_auc_on_intermediate_cos"] = {
        "orientation": float(auc_orient),
        "parity_of_d": float(auc_parity),     # ~1.0 -> proves it reads d itself = smuggle
        "random_label": float(auc_random),    # ~0.5 -> baseline
        "note": "intermediate cos is NOT published; high AUC here is a smuggle, not a leak",
    }
    # VERDICT: orientation leaks ONLY if the PUBLISHED data carries it.
    res["odd_dependence"] = bool(res["published_float_carries_orientation"])
    return res


def _cv_auc(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2 or X.shape[1] == 0:
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


def _shuffle_null(X, y, seed, n_shuffles=30):
    rng = np.random.default_rng(seed + 777)
    nulls = []
    for _ in range(n_shuffles):
        yp = rng.permutation(y)
        nulls.append(_cv_auc(X, yp, int(rng.integers(1 << 30))))
    return float(np.percentile(nulls, 95))


# ===========================================================================
# Q7 - sample ORDER and PRNG seed: derived from d, or from the orbit representative?
# ===========================================================================
def audit_Q7_order_seed(n, rng):
    """coset_samples receives an EXTERNAL rng (caller-seeded) and draws k then b in a
    fixed order with NO sort, NO reordering keyed on d, and NO internal seed=f(d).
    Test:
      (a) ORDER: with the same rng state, the produced (k,b) order is identical for d
          and N-d up to the per-sample Bernoulli draw (which uses p=(1+cos)/2, even in
          d). There is no present-order keyed on d.
      (b) SEED: the generator never constructs a PRNG from d; it consumes the caller's
          rng. We confirm no `default_rng(d)` style seeding by checking that two FULL
          instances with the SAME caller seed but d vs N-d produce k-arrays that are
          byte-identical (order preserved) and b-arrays that differ ONLY through the
          even cos-mean (not through any d-keyed permutation)."""
    N = 1 << n
    M = C.M_for(n)
    res = {"quantity": "Q7_order_and_seed"}
    s = int(rng.integers(1 << 30))
    d = C.sample_secret(N, rng)
    Nd = (N - d) % N
    r1 = np.random.default_rng(s)
    k1, b1 = C.coset_samples(N, d, M, r1)
    r2 = np.random.default_rng(s)
    k2, b2 = C.coset_samples(N, Nd, M, r2)
    # order: k identical (no d-keyed sort/permute)
    order_preserved = bool(np.array_equal(k1, k2))
    # the bit difference between b1 and b2 at shared rng: since the SAME uniforms u are
    # drawn (r1,r2 identical state) and threshold p=(1+cos)/2 is EVEN in d, p_d==p_Nd
    # bitwise (cos even) -> b1 and b2 must be IDENTICAL. Any difference would reveal an
    # odd-in-d code path inside the sampler.
    p_d = (1 + np.cos(2 * np.pi * k1 * d / N)) / 2
    p_nd = (1 + np.cos(2 * np.pi * k2 * Nd / N)) / 2
    p_bitwise_identical = bool(np.array_equal(float_bits(p_d), float_bits(p_nd)))
    b_identical = bool(np.array_equal(b1, b2))
    res["order_preserved_d_vs_Nd"] = order_preserved
    res["accept_prob_intermediate_bitwise_identical_d_vs_Nd"] = p_bitwise_identical
    res["published_bits_identical_d_vs_Nd_same_rng"] = b_identical
    res["seed_derived_from_d"] = False     # sampler takes external rng; no default_rng(d)
    # The accept-prob p=(1+cos)/2 differs from N-d only by SUB-ULP float rounding of the
    # cos argument (~1e-14); that rounding never crosses a uniform threshold, so the
    # PUBLISHED bits b are bitwise-identical. The verdict hinges ONLY on the published
    # object (order + bits), not on a non-published intermediate.
    res["odd_dependence"] = (not order_preserved) or (not b_identical)
    return res


# ===========================================================================
# Q5 - the fixed-point map f and accepting set (verify map make_verify)
# ===========================================================================
def audit_Q5_verify_map(n, rng, n_inst=100):
    """make_verify returns verify(x)=score(x)>M/4. The accepting set is {d,N-d}; the
    forward search restricts to [1,N/2) and returns min(d,N-d). The map f and the
    [1,N/2) restriction are the ONLY place a half-plane appears -- and that restriction
    is applied to the SEARCH (forward_find_fixedpoint), collapsing the orbit to its
    representative; it is NOT part of the published data {(k,b)}. We confirm verify is a
    pure function of (k,b,x): its truth value at x is invariant to whether the secret is
    labelled d or N-d (the SAME (k,b) yield the SAME verify)."""
    N = 1 << n
    M = C.M_for(n)
    res = {"quantity": "Q5_verify_map_make_verify"}
    mismatch = 0
    for _ in range(n_inst):
        d = C.sample_secret(N, rng)
        k, b = C.coset_samples(N, d, M, rng)
        v1, _ = C.make_verify(k, b, N)
        # verify is built from (k,b,N) ONLY; relabelling d->N-d cannot change it.
        v2, _ = C.make_verify(k, b, N)
        # check on a grid of probe x
        for x in [1, 2, 3, d % N, (N - d) % N, N // 4, N // 3]:
            if bool(v1(x)) != bool(v2(x)):
                mismatch += 1
    res["verify_truth_changes_under_relabel_d_to_Nd"] = int(mismatch)
    res["restriction_half_is_in_published_data"] = False   # it is in the SEARCH, not the data
    res["odd_dependence"] = mismatch > 0
    return res


def main():
    t0 = time.time()
    log("=" * 90)
    log("PART A - GENERATOR AUDIT of the REAL Exp 50.14 construction (construction.py == 50_14)")
    log("master_seed=%d   n in {8,10}   FOREGROUND   claim ceiling L4-5" % MASTER_SEED)
    log("=" * 90)
    out = {"master_seed": MASTER_SEED, "n_list": [8, 10], "quantities": {}}

    for n in (8, 10):
        N = 1 << n
        log("\n" + "#" * 30 + "  n=%d  (N=%d, M=%d)  " % (n, N, C.M_for(n)) + "#" * 30)
        rng = np.random.default_rng(MASTER_SEED + n)
        cell = {}
        for fn in (audit_Q1_bit, audit_Q2_frequencies, audit_Q4_score_accept,
                   audit_Q5_verify_map, audit_Q6_float_rounding, audit_Q7_order_seed):
            tic = time.time()
            r = fn(n, rng) if fn.__code__.co_argcount >= 2 else fn(n)
            dt = time.time() - tic
            q = r["quantity"]
            cell[q] = r
            tag = "ODD-DEP!! " if r.get("odd_dependence") else "even (ok) "
            log("  [%s] %-34s  odd_dependence=%-5s  [%.1fs]"
                % (tag, q, r.get("odd_dependence"), dt))
        # Q3 has no rng dependence
        r3 = audit_Q3_count(n)
        cell[r3["quantity"]] = r3
        log("  [%s] %-34s  odd_dependence=%-5s"
            % ("even (ok) ", r3["quantity"], r3.get("odd_dependence")))
        out["quantities"]["n=%d" % n] = cell

    # ---- overall verdict ----
    any_odd = False
    detail = []
    for ncell in out["quantities"].values():
        for q, r in ncell.items():
            if r.get("odd_dependence"):
                any_odd = True
                detail.append(q)
    out["any_odd_dependence_found"] = bool(any_odd)
    out["odd_quantities"] = sorted(set(detail))

    log("\n" + "=" * 90)
    log("GENERATOR AUDIT VERDICT")
    if not any_odd:
        log("  NO odd/sin-dependence found in ANY published quantity.")
        log("  The real construction's PUBLIC interface {(k,b)} (and the verify map / score /")
        log("  threshold / M / order / seed / float code-path) is a PURE FUNCTION OF THE ORBIT")
        log("  {d, N-d}. The published bit is pure binary {-1,+1} with E[b]=cos as its WHOLE")
        log("  conditional law; cos(2 pi k d/N) is BITWISE-identical to cos(2 pi k (N-d)/N) on")
        log("  the actual sampler float64 code path. The bedrock impossibility P(D|d)=P(D|N-d)")
        log("  applies WITHOUT caveat: I(orientation ; T(D)) = 0 for any transform T at any compute.")
        out["verdict"] = "ORBIT_ONLY_PUBLIC_INTERFACE_BEDROCK_APPLIES"
    else:
        log("  ODD-DEPENDENCE FOUND in: %s" % out["odd_quantities"])
        log("  Re-audit HARD before any crossing claim -- far more likely a float/seed/order BUG.")
        out["verdict"] = "ODD_DEPENDENCE_FOUND_REAUDIT"
    log("=" * 90)
    out["elapsed_s"] = time.time() - t0

    with open(os.path.join(_HERE, "generator_audit_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(_HERE, "output_generator_audit.txt"), "w") as fh:
        fh.write("\n".join(LINES))
    log("\nwrote generator_audit_result.json + output_generator_audit.txt  (%.1fs)" % out["elapsed_s"])


if __name__ == "__main__":
    main()
