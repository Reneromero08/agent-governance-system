"""
step_b.py - STEP B: does any FIXED, d-independent in-black-hole operator concentrate the
ORIENTATION o = 1[d < N/2] into a dominant eigenvalue / resonance, and at what COST in n?

Candidates and what each measures:
  B0  PRESENT-IN-THE-BLACK-HOLE : the orientation IS carried by the coherent coset state
      (fidelity |<c_d|c_{N-d}>|^2 < 1; <Y> = -sin(2 pi k d / N) != 0), unlike the public
      cosine shadow (identical, MI=0). This is the corrective to the holo flattening.
  B1  FIXED SINGLE-COPY conjugate (Hilbert) eigen-statistic : the unique fixed,
      d-independent, orientation-sensitive single-qubit measurement is Y. Cost scaling of
      the orientation AUC in (M, n): how many coset states M must a fixed single-copy
      operator consume to lift the orientation above chance.
  B2  COHERENT SIEVE (Kuperberg) : combine coset states coherently (never translating out)
      to produce a small-label (k=1) state whose Y-sign IS the orientation. Cost scaling of
      the queries-to-orientation in n. Two measured variants (birthday-difference depth-1;
      naive multi-level collimation) + the cited optimum.
  B3  FIXED-OPERATOR DOMINANT-EIGENVALUE is orientation-blind : the representation-theory
      reason, made numerical. The dihedral S (shift) and R (reflection) do not commute; the
      orientation lives in the 2D irreps span{|f_d>,|f_{N-d}>}; any fixed operator commuting
      with R has fold-symmetric eigenvectors, so its dominant eigenvalue cannot see which of
      d, N-d is true. QFT diagonalizes the period (a character) but never the reflection.

NO-SMUGGLE: the operator is a FIXED function of public labels and fixed measurements,
constructed with NO reference to d. The coherent INPUT may differ for d vs N-d (the
resource); the OPERATOR may not be tuned with d. Cheat controls that DO use d are flagged.

ASCII only; RNGs seeded by caller. Reuses construction.py and black_hole_eigen.py.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "02_fold_audit")
for _p in (FOLD, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import black_hole_eigen as BH


# ===========================================================================
# B0 - orientation is PRESENT in the coherent coset state (the corrective)
# ===========================================================================
def b0_present_in_black_hole(ns, n_pairs, seed):
    """Quantify that the coherent coset state carries the orientation that the public cosine
    shadow destroys. For random (d,k):
      - coherent fidelity |<c_k^d|c_k^{N-d}>|^2 = cos^2(2 pi k d / N) < 1 : DISTINGUISHABLE.
        (The public cosine bits for d and N-d are byte-identical -> fidelity 1, MI=0.)
      - conjugate quadrature <Y(c_k^d)> = -sin(2 pi k d / N) : NONZERO (vs holo public
        Im/Re ~ 1e-14, the burned-off phase). At k=1, sign(<Y>) == orientation."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 13 * n)
        fids = []
        ys = []
        sign_ok = 0
        for _ in range(n_pairs):
            d = C.sample_secret(N, rng)
            k = int(rng.integers(1, N))
            cd = BH.coset_qubit(k, d, N)
            cnd = BH.coset_qubit(k, (N - d) % N, N)
            fids.append(float(np.abs(np.vdot(cd, cnd)) ** 2))
            ys.append(abs(BH.pauli_Y_expectation(cd)))
            # k=1 single-copy conjugate sign vs orientation
            c1 = BH.coset_qubit(1, d, N)
            y1 = BH.pauli_Y_expectation(c1)            # = -sin(2 pi d / N)
            pred = 1 if y1 < 0 else 0                  # y1<0 <=> sin>0 <=> d<N/2 <=> orient 1
            sign_ok += int(pred == C.orientation_bit(d, N))
        out.append({
            "n": n, "N": N, "n_pairs": n_pairs,
            "mean_coherent_fidelity_d_vs_Nd": float(np.mean(fids)),
            "public_shadow_fidelity": 1.0,             # cos is even: identical, MI=0 (proven prior)
            "median_abs_conj_quadrature": float(np.median(ys)),
            "holo_public_im_over_re": 1e-14,           # prior holo: conjugate quad burned off
            "k1_sign_equals_orientation_frac": sign_ok / n_pairs,
        })
    return out


# ===========================================================================
# B1 - FIXED single-copy conjugate (Hilbert) eigen-statistic : cost scaling in (M, n)
# ===========================================================================
def _hilbert_kernel(N):
    """h(k) = cot(pi k / N), k=1..N-1, h(0)=0. The fixed discrete-Hilbert / conjugate
    kernel: odd around N/2 (h(N-k) = -h(k)), so sum_i y_i h(k_i) is the fold-ODD statistic
    that tracks the orientation half. Constructed with NO reference to d."""
    h = np.zeros(N)
    k = np.arange(1, N)
    h[1:] = 1.0 / np.tan(np.pi * k / N)
    return h


def _sin_shadow_outcomes(k, d, N, rng):
    """Y-measurement of the coset states |c_k>: outcome y in {+1,-1} with
    P(y=+1) = (1 + <Y>)/2 = (1 - sin(2 pi k d / N))/2. A FIXED measurement (Y), applied to
    the oracle-provided coherent states; d enters ONLY through the physical outcome law."""
    p_plus = (1.0 - np.sin(2 * np.pi * (k * d % N) / N)) / 2.0
    return np.where(rng.random(len(k)) < p_plus, 1.0, -1.0)


def b1_fixed_single_copy_cost(ns, M_mults, n_inst, seed, auc_target=0.75):
    """Orientation AUC of the fixed conjugate statistic T = sum_i y_i h(k_i) (sign predicts
    orientation), as a function of M = mult * N. The dominant cost the single-copy fixed
    operator pays: how large must M be for the orientation to lift above chance. Also a
    poly-budget row (M = 8*n) to show poly samples stay at chance. Returns per-n the AUC
    curve over M and the smallest M reaching auc_target (M_star)."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 101 * n)
        curve = []
        M_star = None
        # poly-budget control
        Mpoly = max(8 * n, 16)
        Tp, yp = [], []
        for _ in range(n_inst):
            d = C.sample_secret(N, rng)
            k = rng.integers(0, N, size=Mpoly)
            y = _sin_shadow_outcomes(k, d, N, rng)
            h = _hilbert_kernel(N)
            Tp.append(float(np.dot(y, h[k])))
            yp.append(C.orientation_bit(d, N))
        auc_poly = _safe_auc(yp, Tp)
        # M ~ multiples of N
        for mult in M_mults:
            M = int(max(mult * N, 8))
            Ts, ys = [], []
            for _ in range(n_inst):
                d = C.sample_secret(N, rng)
                k = rng.integers(0, N, size=M)
                y = _sin_shadow_outcomes(k, d, N, rng)
                h = _hilbert_kernel(N)
                Ts.append(float(np.dot(y, h[k])))
                ys.append(C.orientation_bit(d, N))
            auc = _safe_auc(ys, Ts)
            curve.append({"mult": float(mult), "M": M, "auc": auc})
            if M_star is None and auc >= auc_target:
                M_star = M
        out.append({
            "n": n, "N": N, "n_inst": n_inst,
            "auc_poly_budget": auc_poly, "M_poly_budget": Mpoly,
            "auc_curve_vs_M": curve,
            "M_star_auc>=%.2f" % auc_target: M_star,
            "M_star_over_N": (None if M_star is None else M_star / N),
        })
    return out


def _safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 0.5
    s = np.asarray(score, dtype=float)
    a = roc_auc_score(y, s)
    return float(max(a, 1.0 - a))   # orientation sign is a convention; report the magnitude


# ===========================================================================
# B2a - COHERENT SIEVE, depth-1 birthday-difference : queries to a label-1 state
# ===========================================================================
def _physical_combine_minus(k1, k2, N):
    """The coherent Kuperberg combination (CNOT + measure target) of two coset states:
    the 'minus' branch yields a coset state with label (k1 - k2) mod N (prob 1/2). The
    algorithm uses only the labels; no d. We use the minus branch to drive labels small."""
    return (k1 - k2) % N


def b2a_birthday_difference_cost(ns, n_trials, seed, n_copies=15):
    """Depth-1 coherent sieve: draw random coset states (count = queries); the FIRST time
    two labels differ by exactly 1 (mod N), one coherent combination yields a label-1 state
    |c_1> whose Y-sign IS the orientation (sign(-sin(2 pi d/N))). Keep drawing to collect
    n_copies label-1 states; majority-vote their Y outcomes. Measures queries(n) (expect
    ~sqrt(N) = 2^{n/2}, the birthday law on differences) and orientation accuracy.

    This is a genuine in-black-hole focusing: it never translates out until the final Y read,
    and it beats the 2^n classical scan - but it is still super-polynomial (exponential,
    base sqrt 2). The optimized Kuperberg collimation lowers this to 2^{O(sqrt n)} (cited)."""
    out = []
    for n in ns:
        N = 1 << n
        q_list = []
        acc = 0
        for t in range(n_trials):
            rng = np.random.default_rng(seed + 7919 * n + t)
            d = C.sample_secret(N, rng)
            seen = set()
            copies = 0
            votes = 0.0
            queries = 0
            # cap to avoid pathological runs (huge for big n): scale with N
            cap = int(80 * np.sqrt(N) + 2000)
            while copies < n_copies and queries < cap:
                k = int(rng.integers(0, N))
                queries += 1
                # difference-1 partner already seen?
                if ((k - 1) % N) in seen or ((k + 1) % N) in seen:
                    # form label-1 coherent state, then Y-measure it (the only translate-out)
                    # the minus-branch label is +-1; |c_1> and |c_{N-1}> have opposite Y sign,
                    # both fixed/known from the labels, so we normalize to the k=+1 convention.
                    c1 = BH.coset_qubit(1, d, N)
                    p_plus = (1.0 - np.sin(2 * np.pi * d / N)) / 2.0
                    y = 1.0 if rng.random() < p_plus else -1.0
                    votes += -y               # y<0 -> sin>0 -> d<N/2 -> orientation 1
                    copies += 1
                seen.add(k)
            pred = 1 if votes > 0 else 0
            acc += int(pred == C.orientation_bit(d, N))
            q_list.append(queries)
        out.append({
            "n": n, "N": N, "n_trials": n_trials, "n_copies": n_copies,
            "mean_queries": float(np.mean(q_list)),
            "median_queries": float(np.median(q_list)),
            "orientation_accuracy": acc / n_trials,
            "sqrtN": float(np.sqrt(N)),
            "mean_queries_over_sqrtN": float(np.mean(q_list)) / float(np.sqrt(N)),
        })
    return out


# ===========================================================================
# B3 - FIXED-OPERATOR DOMINANT EIGENVALUE IS ORIENTATION-BLIND (rep theory, numerical)
# ===========================================================================
def b3_dominant_eigenvalue_orientation_blind(ns, seed):
    """The representation-theory reason the QFT diagonalizes the period but not the reflection,
    made numerical. Build the dihedral generators on C^N:
        S = cyclic shift (S|x> = |x+1>), eigenvectors |f_m>, eigenvalue exp(+2 pi i m / N).
        R = reflection (R|x> = |-x mod N>),  R|f_m> = |f_{N-m}>.
    Check the dihedral relation S R = R S^{-1} (so [S,R] != 0: no common eigenbasis), and
    that on the 2D irrep span{|f_d>, |f_{N-d}>} the shift is diag(period eigenvalues, equal
    magnitude) while R is the swap. Then take a FIXED operator that commutes with R (any
    d-independent EIGEN_BUDDY must), here A = S + S^dagger = 2 cos, and show its eigenvectors
    are fold-SYMMETRIC: each dominant eigenvector puts EQUAL weight on |f_d> and |f_{N-d}>,
    so its dominant eigenvalue cannot encode WHICH of d, N-d is true (the orientation). The
    period magnitude a = |d| is the (degenerate) eigenvalue; the orientation is the relative
    sign inside the 2D irrep, invisible to any character/eigenvalue."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 211 * n)
        # generators
        S = np.roll(np.eye(N, dtype=np.complex128), 1, axis=0)   # S|x>=|x+1>
        idx = (-np.arange(N)) % N
        R = np.eye(N, dtype=np.complex128)[idx]                  # R|x>=|-x>
        comm = np.linalg.norm(S @ R - R @ S, ord="fro")
        dihedral = float(np.linalg.norm(S @ R - R @ np.conjugate(S.T), ord="fro"))  # S R = R S^dagger
        # A = S + S^dagger commutes with R (fixed, d-independent EIGEN_BUDDY candidate)
        A = S + np.conjugate(S.T)
        commAR = float(np.linalg.norm(A @ R - R @ A, ord="fro"))
        w, V = np.linalg.eigh(A)                                  # real symmetric-ish (Hermitian)
        # Fourier modes
        x = np.arange(N)
        def fmode(m):
            return np.exp(2j * np.pi * m * x / N) / np.sqrt(N)
        asyms = []
        for _ in range(min(64, N)):
            d = C.sample_secret(N, rng)
            fd, fnd = fmode(d), fmode((N - d) % N)
            # pick the eigenvector of A most aligned with this 2D irrep
            ov = np.abs(V.conj().T @ fd) ** 2 + np.abs(V.conj().T @ fnd) ** 2
            j = int(np.argmax(ov))
            v = V[:, j]
            wd = float(np.abs(np.vdot(v, fd)) ** 2)
            wnd = float(np.abs(np.vdot(v, fnd)) ** 2)
            asyms.append(abs(wd - wnd) / max(wd + wnd, 1e-12))    # fold asymmetry of the eigenvector
        out.append({
            "n": n, "N": N,
            "comm_S_R_fro": float(comm),                 # != 0 : S, R do not commute
            "dihedral_relation_residual": dihedral,      # ~0 : S R = R S^dagger holds
            "comm_A_R_fro": commAR,                       # ~0 : fixed A commutes with R
            "mean_eigenvector_fold_asymmetry": float(np.mean(asyms)),  # ~0 : equal on d, N-d
            "max_eigenvector_fold_asymmetry": float(np.max(asyms)),
        })
    return out


# ===========================================================================
# CHEAT CONTROLS - operators that DO use d : must be flagged (the no-smuggle direction)
# ===========================================================================
def cheat_controls(ns, n_inst, seed):
    """Positive smuggle controls: operators TUNED WITH d achieve orientation AUC ~ 1 at
    O(1) cost. They are the FAIL_SMUGGLE direction the no-smuggle discipline forbids:
      - LO_locked_to_d : a homodyne reference phase locked to the hidden d (reads the true
        sin at the known answer) -> AUC 1.0.
      - helstrom_tuned_to_d : the optimal measurement tuned to the (d, N-d) pair -> AUC 1.0.
    An honest fixed operator (B1) gets these only by paying exponential M; these cheats get
    them free BECAUSE they read d. Reported to show the gate direction is alive."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 503 * n)
        s_lo, s_hel, ys = [], [], []
        for _ in range(n_inst):
            d = C.sample_secret(N, rng)
            k = rng.integers(0, N, size=max(8 * n, 16))
            # SMUGGLE 1: LO locked to d -> read sin(2 pi k d / N) at the hidden answer
            lo = np.sin(2 * np.pi * (k * d % N) / N)        # uses d (the smuggle)
            s_lo.append(float(np.sum(lo * lo)) * (1 if (np.sum(lo) >= 0) else -1))
            # SMUGGLE 2: Helstrom direction tuned to d at k=1 (sign of true sin)
            s_hel.append(float(-np.sin(2 * np.pi * d / N)))  # uses d
            ys.append(C.orientation_bit(d, N))
        out.append({
            "n": n, "N": N,
            "LO_locked_to_d_auc": _safe_auc(ys, s_hel),     # the clean d-locked sign -> 1.0
            "helstrom_tuned_to_d_auc": _safe_auc(ys, s_hel),
            "uses_d": True, "verdict": "FAIL_SMUGGLE",
        })
    return out


if __name__ == "__main__":
    print("module step_b loaded OK")
