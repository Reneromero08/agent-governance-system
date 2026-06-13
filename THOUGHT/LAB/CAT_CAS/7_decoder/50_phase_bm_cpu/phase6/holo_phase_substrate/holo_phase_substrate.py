"""
holo_phase_substrate.py - CLOSEOUT test of the lab's flagship "it from phase"
phase-cavity / homodyne (conjugate-quadrature) substrate against the 50.14
dihedral fold. Reuses the real lab instruments (construction.py, no_smuggle_gate.py,
stage3/hardened_gate.py) and the live phase-cavity machinery (the holographic
phase-grating -> IFFT/FFT resonance readout of 8_riemann_harmonic_sieve.py and the
HOLO 02_cavity eigenmode sieve), now run on the 50.14 public data.

SUBSTRATE (the .holo phase cavity, homodyne form):
  PUBLIC samples (k_i, b_i), E[b_i] = cos(2 pi k_i d / N), are binned onto the Z_N
  frequency grid as the empirical spectrum
        B[m] = sum_{i : k_i = m} b_i          (real; exactly the un-normalized cos_hat)
  The COHERENT cavity field over torus position x is the holographic (inverse-DFT)
  reconstruction -- the lab's phase-grating sieve evaluated at full altitude:
        Psi(x) = sum_m B[m] exp(+2 pi i m x / N) = N * IFFT(B)[x]
  Re Psi(x) = score(x)             (matched-filter intensity at LO phase phi_LO = 0)
  Im Psi(x) = the CONJUGATE QUADRATURE (intensity at LO phase phi_LO = pi/2)
  Homodyne readout with controllable local-oscillator reference phase phi_LO:
        H(x; phi_LO) = Re[ exp(-i phi_LO) Psi(x) ]
                     = cos(phi_LO) * Re Psi(x) + sin(phi_LO) * Im Psi(x)
  The substrate keeps the COMPLEX amplitude coherent and measures BOTH quadratures
  (homodyne / interferometric), with a tunable reference phase -- the defining
  property vs the scalar readouts already tried (the 6 non-Hermitian sensors).

MEASUREMENTS:
  (A) MAGNITUDE / fold-answer: native cavity intensity |Psi(x)| peak recovers the
      unordered set {d, N-d} = a = min(d, N-d). (Expected YES -- the even/decodable part.)
  (B) ORIENTATION: the conjugate-quadrature homodyne readout (phi_LO sweep, Im quadrature,
      interferometric recombination of the two fold images, torus winding angles),
      scored by the hardened no-smuggle gate. (Predicted FAIL_CHANCE.)
  (C) REPRESENTATION CONGRUENCE (SPEC 1B "relax, do not construct"): a phase-cavity
      alternating-projection relaxation with a symmetric PUBLIC support; does it settle
      into a representation where d (the lower-half representative) is the dominant
      attractor without smuggling? (Predicted FAIL_CHANCE: it relaxes to the symmetric
      real-even fixed point.)

NO-SMUGGLE CRUX:
  Every readout is gated by stage3/hardened_gate (random-private-fold + exact d<->N-d
  byte-equal invariance). Two mandatory sensitivity controls: (i) a deliberate SMUGGLE
  op (homodyne LO locked to the hidden d, injecting the true sin) must be CAUGHT
  (AUC->1, delta>0); (ii) a useless-even op must be CHANCE. The decisive mechanistic
  self-check: feeding the cavity the EVEN public cosines, the imaginary (sin) quadrature
  it computes is ~0 (phaseless) -- the conjugate quadrature of a real-even cos-only
  spectrum is identically zero; orientation appears ONLY when the hidden sin is injected.

ASCII only. All RNGs seeded; seeds recorded. Claim ceiling L4-5.
Author: holo-phase-substrate closeout agent (Fable).
"""
import os
import sys
import json
import time

import numpy as np

# --- make the real lab instruments importable (do not reimplement them) ---
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "fold_audit")
STAGE3 = os.path.join(FOLD, "stage3")
for _p in (FOLD, STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C            # the verbatim 50.14 public construction
import no_smuggle_gate as G         # AUC harness + exact d-invariance audit + controls
import hardened_gate as Hg          # random-private-fold hardened gate


# ===========================================================================
# THE PHASE CAVITY (homodyne form): coherent complex field from PUBLIC data
# ===========================================================================
def binned_spectrum(k, b, N):
    """B[m] = sum_{i:k_i=m} b_i -- the empirical (real) phase-grating spectrum.
    Pure function of PUBLIC (k,b,N). This is the un-normalized cos_hat."""
    B = np.zeros(N, dtype=np.complex128)
    np.add.at(B, k, b.astype(np.complex128))
    return B


def cavity_field(B):
    """Psi(x) = sum_m B[m] exp(+2 pi i m x / N) = N * IFFT(B). The coherent .holo
    reconstruction over torus position x; keeps BOTH quadratures (Re=score, Im=conj)."""
    return np.fft.ifft(B) * len(B)


def homodyne(Psi, phi_LO):
    """H(x; phi_LO) = Re[ exp(-i phi_LO) Psi(x) ] -- the homodyne readout with a
    controllable local-oscillator reference phase. phi_LO=0 -> score; pi/2 -> conj quad."""
    return (np.cos(phi_LO) * Psi.real + np.sin(phi_LO) * Psi.imag)


def _public_seed(k, b, N):
    """Deterministic seed from PUBLIC data only, so any randomized op is byte-identical
    on inst and folded_instance (the exact d-invariance audit stays valid)."""
    h = (int(np.sum(k.astype(np.int64)) * 1000003)
         ^ int(np.sum((b > 0).astype(np.int64)) * 2654435761)
         ^ (int(N) * 40503))
    return h & 0x7FFFFFFF


# ===========================================================================
# (A) MAGNITUDE / FOLD-ANSWER: native cavity resonance recovers a = min(d, N-d)
# ===========================================================================
def fold_answer(k, b, N):
    """Native cavity intensity |Psi(x)| peak. Peaks at BOTH d and N-d (even field),
    so recovering argmax in [1,N) and folding gives a = min(d, N-d)."""
    Psi = cavity_field(binned_spectrum(k, b, N))
    inten = np.abs(Psi)
    inten[0] = -1.0                       # exclude trivial x=0
    xpk = int(np.argmax(inten))
    a_hat = min(xpk, (N - xpk) % N)
    return a_hat, xpk


def measure_A(n, n_inst, seed):
    N = 1 << n
    rng = np.random.default_rng(seed)
    M = C.M_for(n)
    exact = 0
    for _ in range(n_inst):
        d = C.sample_secret(N, rng)
        k, b = C.coset_samples(N, d, M, rng)
        a_hat, _ = fold_answer(k, b, N)
        a = min(d, (N - d) % N)
        exact += int(a_hat == a)
    return {"n": n, "N": N, "M": M, "n_inst": n_inst,
            "frac_exact_fold_answer": exact / n_inst, "seed": int(seed)}


# ===========================================================================
# (B) ORIENTATION: conjugate-quadrature homodyne operator (PUBLIC only)
# ===========================================================================
def O_homodyne_quadrature(inst):
    """The crux operator. Build the coherent cavity field from PUBLIC (k,b), locate the
    resonance peak (public), and read the CONJUGATE QUADRATURE: the Im quadrature at the
    peak/mirror, a full phi_LO homodyne sweep, the resonance phase angle, an
    interferometric recombination Psi(xpk)*conj(Psi(mirror)) of the two fold images, the
    torus winding angles of the spectrum at dyadic rungs (exp(i 2 pi W / q)), and the Im
    quadrature at low dyadic positions. Reads ONLY k,b,N,n -> exact d-invariance (delta=0).
    Prediction: FAIL_CHANCE (the conjugate quadrature of the public even data is phaseless)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    B = binned_spectrum(k, b, N)
    Psi = cavity_field(B)
    inten = np.abs(Psi); inten[0] = -1.0
    xpk = int(np.argmax(inten))
    xmir = (N - xpk) % N
    feats = []
    # conjugate quadrature (LO phase pi/2) at the public peak and its fold mirror
    feats.append(float(Psi.imag[xpk]))
    feats.append(float(Psi.imag[xmir]))
    # homodyne phi_LO sweep at the resonance peak
    for phi in np.linspace(0.0, np.pi, 8, endpoint=False):
        feats.append(float(np.cos(phi) * Psi.real[xpk] + np.sin(phi) * Psi.imag[xpk]))
    # resonance phase angle (homodyne phase) at peak and mirror
    feats.append(float(np.arctan2(Psi.imag[xpk], Psi.real[xpk])))
    feats.append(float(np.arctan2(Psi.imag[xmir], Psi.real[xmir])))
    # interferometric two-arm recombination of the two fold images
    cross = Psi[xpk] * np.conj(Psi[xmir])
    feats.append(float(cross.real)); feats.append(float(cross.imag))
    # torus winding: angle of the phase-grating spectrum at dyadic rungs exp(i 2 pi W / q)
    for j in range(1, 8):
        m = (1 << j) % N
        feats.append(float(np.angle(B[m])) if abs(B[m]) > 0 else 0.0)
    # conjugate quadrature at low dyadic positions (where the odd channel would live)
    for r in (1, 2, 4, 8):
        feats.append(float(Psi.imag[r % N]))
    return np.array(feats, dtype=float)


def O_homodyne_magnitude_only(inst):
    """USELESS-EVEN control: read only the LO-phase-independent cavity intensity |Psi|
    (the decodable even channel). Fold-invariant by construction -> chance on orientation."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    Psi = cavity_field(binned_spectrum(k, b, N))
    inten = np.abs(Psi); inten[0] = -1.0
    xpk = int(np.argmax(inten)); xmir = (N - xpk) % N
    feats = [float(inten[xpk]), float(inten[xmir]),
             float(inten[xpk] - inten[xmir]),
             float(np.mean(inten)), float(np.std(inten))]
    for r in (1, 2, 4, 8):
        feats.append(float(inten[r % N]))
    return np.array(feats, dtype=float)


def O_homodyne_LO_locked_SMUGGLE(inst):
    """DESIGNATED SMUGGLE control: lock the homodyne local oscillator to the HIDDEN secret
    d -- i.e. build the cavity from the FULL COMPLEX quadrature z_k = exp(-i 2 pi k d / N),
    injecting the absent odd/sin channel. The field then localizes to a SINGLE peak at d
    (not the symmetric d,N-d pair), so the peak's half = the orientation. MUST be caught:
    the support/peak flips under d->N-d (delta>0) and AUC->1. Reports where it enters:
    reads inst['d'] to set the LO reference phase."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    d = inst["d"]                                       # <-- reading the secret (the smuggle)
    m = np.arange(N)
    cnt = np.zeros(N); np.add.at(cnt, k, 1.0)            # public visit counts (fair weighting)
    Z = cnt * np.exp(-2j * np.pi * m * d / N)           # <-- true sin injected via d
    Psi = cavity_field(Z)
    inten = np.abs(Psi); inten[0] = -1.0
    xpk = int(np.argmax(inten))
    return np.array([float(xpk < N / 2.0),
                     float(Psi.imag[xpk]),
                     float(np.angle(Psi[xpk]))], dtype=float)


# ===========================================================================
# (C) REPRESENTATION CONGRUENCE: phase-cavity relaxation, symmetric PUBLIC support
# ===========================================================================
def O_relax_congruence(inst):
    """SPEC 1B 'relax, do not construct'. Alternating-projection phase-cavity relaxation:
    magnitudes = |public spectrum|, constraint = real signal with a SYMMETRIC (public,
    no half-preference) support. Read whether the relaxed field localizes to the LOWER
    half (d's representative) vs the UPPER half. PUBLIC-only (random init seeded from public
    data) -> exact d-invariance. Prediction: chance -- the relaxation settles to the
    symmetric real-even fixed point, so no congruent d-dominant representation emerges."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    B = binned_spectrum(k, b, N).real
    mag = np.abs(B)
    rng = np.random.default_rng(_public_seed(k, b, N))
    X = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, size=N))
    for _ in range(25):
        x = np.real(np.fft.ifft(X))            # real-signal (symmetric public) projection
        X = mag * np.exp(1j * np.angle(np.fft.fft(x)))
    field = np.real(np.fft.ifft(X))
    lower = float(np.sum(field[1:N // 2] ** 2))
    upper = float(np.sum(field[N // 2 + 1:] ** 2))
    am = int(np.argmax(np.abs(field)))
    return np.array([lower - upper,
                     lower / (upper + 1e-12),
                     float(am < N / 2.0)], dtype=float)


# ===========================================================================
# THE MECHANISTIC SELF-CHECK: conjugate quadrature of public even data ~ 0
# ===========================================================================
def phaseless_selfcheck(n, n_secrets, seed):
    """The decisive numerical heart. For each secret d:
      (1) NOISELESS EVEN public spectrum c_m = cos(2 pi m d / N): real-even. The cavity
          field is real -> its imaginary (conjugate/sin) quadrature is ~0 (PHASELESS), and
          |Psi| peaks at BOTH d and N-d with EQUAL height (orientation absent).
      (2) FULL COMPLEX spectrum z_m = exp(-i 2 pi m d / N) (INJECT the hidden sin): the
          field localizes to a SINGLE peak at d, and the spectrum's imaginary quadrature is
          -sin(2 pi m d / N) != 0 with SIGN == orientation. So orientation appears ONLY when
          the hidden odd channel is injected.
      (3) NOISY public field from sampled b: Im is nonzero (finite-sample) but is a pure
          function of public data -> fold-invariant -> orientation-blind (gate FAIL_CHANCE).
    Returns medians of the diagnostic ratios."""
    N = 1 << n
    rng = np.random.default_rng(seed)
    m = np.arange(N)
    even_imag_ratio = []      # max|Im Psi_even| / max|Re Psi_even|  -> ~0
    even_peak_asym = []       # | |Psi(d)| - |Psi(N-d)| | / sum      -> ~0 (symmetric)
    inj_single_peak = []      # argmax|Psi_full| == d ?              -> True
    inj_spec_imag = []        # | Im(z_rung) |  (= |sin|)            -> O(1), sign==orient
    inj_sign_is_orient = []   # sign(Im(z_rung1)) matches orientation?
    noisy_imag_ratio = []     # max|Im Psi_noisy| / max|Re Psi_noisy| (nonzero, fold-blind)
    M = C.M_for(n)
    for _ in range(n_secrets):
        d = C.sample_secret(N, rng)
        o = C.orientation_bit(d, N)
        # (1) even public cosine spectrum
        c = np.cos(2 * np.pi * m * d / N).astype(np.complex128)
        Pe = cavity_field(c)
        even_imag_ratio.append(float(np.max(np.abs(Pe.imag)) / (np.max(np.abs(Pe.real)) + 1e-300)))
        pe = np.abs(Pe)
        even_peak_asym.append(float(abs(pe[d] - pe[(N - d) % N]) / (pe[d] + pe[(N - d) % N] + 1e-300)))
        # (2) full complex spectrum (inject the hidden sin)
        z = np.exp(-2j * np.pi * m * d / N)
        Pf = cavity_field(z)
        inj_single_peak.append(int(int(np.argmax(np.abs(Pf))) == d))
        # the imaginary (sin) quadrature of the spectrum at rung 1 = -sin(2 pi d / N)
        imag_q = float(np.imag(z[1]))             # = -sin(2 pi d / N)
        inj_spec_imag.append(abs(imag_q))
        # sign of the injected quadrature vs orientation (lower half d<N/2)
        inj_sign_is_orient.append(int((imag_q < 0) == (o == 1)))
        # (3) noisy public field
        k, b = C.coset_samples(N, d, M, rng)
        Pn = cavity_field(binned_spectrum(k, b, N))
        noisy_imag_ratio.append(float(np.max(np.abs(Pn.imag)) / (np.max(np.abs(Pn.real)) + 1e-300)))
    return {
        "n": n, "N": N, "n_secrets": n_secrets, "seed": int(seed),
        "even_public_imag_over_real_median": float(np.median(even_imag_ratio)),
        "even_public_imag_over_real_max": float(np.max(even_imag_ratio)),
        "even_public_peak_asym_median": float(np.median(even_peak_asym)),
        "injected_sin_single_peak_frac": float(np.mean(inj_single_peak)),
        "injected_sin_spec_imag_median": float(np.median(inj_spec_imag)),
        "injected_sin_sign_is_orientation_frac": float(np.mean(inj_sign_is_orient)),
        "noisy_public_imag_over_real_median": float(np.median(noisy_imag_ratio)),
    }


# ===========================================================================
# DRIVER
# ===========================================================================
def run():
    MASTER_SEED = 44060611
    NS = (8, 10, 12, 14)
    GATE_N_INST = 200
    GATE_N_SHUF = 20
    A_N_INST = 60
    SELFCHECK_SECRETS = 40

    out = {
        "experiment": "holo_phase_substrate_49_14_closeout",
        "master_seed": MASTER_SEED,
        "ns": list(NS),
        "gate_n_instances": GATE_N_INST,
        "gate_n_shuffles": GATE_N_SHUF,
        "A_magnitude": [],
        "B_orientation": [],
        "C_congruence": [],
        "controls": [],
        "selfcheck_phaseless": [],
    }
    t_all = time.time()
    print("=" * 92)
    print("HOLO PHASE SUBSTRATE - 50.14 dihedral fold closeout (homodyne conjugate quadrature)")
    print("=" * 92)

    # --- (A) magnitude / fold-answer recovery ---
    print("\n[A] MAGNITUDE / fold-answer recovery (native cavity |Psi| resonance):")
    for n in NS:
        t = time.time()
        r = measure_A(n, A_N_INST, MASTER_SEED + 7 * n)
        r["secs"] = time.time() - t
        out["A_magnitude"].append(r)
        print("    n=%2d  N=%6d  M=%4d  frac_exact(a=min(d,N-d)) = %.3f   [%.1fs]"
              % (n, r["N"], r["M"], r["frac_exact_fold_answer"], r["secs"]))

    # --- mechanistic self-check (the heart) ---
    print("\n[SELF-CHECK] conjugate quadrature of PUBLIC EVEN data ~ 0 (phaseless):")
    for n in NS:
        t = time.time()
        s = phaseless_selfcheck(n, SELFCHECK_SECRETS, MASTER_SEED + 13 * n)
        s["secs"] = time.time() - t
        out["selfcheck_phaseless"].append(s)
        print("    n=%2d  even Im/Re=%.2e (max %.2e)  even_peak_asym=%.2e  | "
              "inj single-peak@d=%.2f  inj |Im sin|=%.3f sign==orient=%.2f  | noisy Im/Re=%.3f"
              % (n, s["even_public_imag_over_real_median"], s["even_public_imag_over_real_max"],
                 s["even_public_peak_asym_median"], s["injected_sin_single_peak_frac"],
                 s["injected_sin_spec_imag_median"], s["injected_sin_sign_is_orientation_frac"],
                 s["noisy_public_imag_over_real_median"]))

    # --- (B) orientation: the crux operator through the hardened gate ---
    print("\n[B] ORIENTATION: conjugate-quadrature homodyne op vs hardened no-smuggle gate:")
    for n in NS:
        t = time.time()
        res = Hg.hardened_gate(O_homodyne_quadrature, n, n_instances=GATE_N_INST,
                               seed=(MASTER_SEED + 101 * n) & 0x7FFFFFFF, n_shuffles=GATE_N_SHUF)
        res["secs"] = time.time() - t; res["n"] = n
        out["B_orientation"].append(res)
        print("    n=%2d  verdict=%-13s orient_auc=%.3f (null95=%.3f)  rf_auc=%.3f (null95=%.3f)"
              "  delta=%.2g  [%.1fs]"
              % (n, res["verdict"], res["auc"], res["shuffle_null_95"],
                 res["random_fold_auc"], res["random_fold_null_95"],
                 res["max_fold_delta"], res["secs"]))

    # --- (C) representation congruence ---
    print("\n[C] REPRESENTATION CONGRUENCE: phase-cavity relaxation vs hardened gate:")
    for n in NS:
        t = time.time()
        res = Hg.hardened_gate(O_relax_congruence, n, n_instances=GATE_N_INST,
                               seed=(MASTER_SEED + 211 * n) & 0x7FFFFFFF, n_shuffles=GATE_N_SHUF)
        res["secs"] = time.time() - t; res["n"] = n
        out["C_congruence"].append(res)
        print("    n=%2d  verdict=%-13s orient_auc=%.3f (null95=%.3f)  rf_auc=%.3f (null95=%.3f)"
              "  delta=%.2g  [%.1fs]"
              % (n, res["verdict"], res["auc"], res["shuffle_null_95"],
                 res["random_fold_auc"], res["random_fold_null_95"],
                 res["max_fold_delta"], res["secs"]))

    # --- sensitivity controls (the gate must catch the smuggle, pass the even) ---
    print("\n[CONTROLS] sensitivity (smuggle MUST be caught; even MUST be chance):")
    control_cases = [
        ("homodyne_LO_locked_SMUGGLE(reads_d)", O_homodyne_LO_locked_SMUGGLE, "FAIL_SMUGGLE"),
        ("homodyne_magnitude_only_even", O_homodyne_magnitude_only, "FAIL_CHANCE"),
        ("gate_useless_even", G.O_useless_even, "FAIL_CHANCE"),
        ("gate_cheat_reads_sin", G.O_cheat_reads_sin, "FAIL_SMUGGLE"),
    ]
    for n in (8, 10):
        for ci, (name, O, expected) in enumerate(control_cases):
            t = time.time()
            res = Hg.hardened_gate(O, n, n_instances=GATE_N_INST,
                                   seed=(MASTER_SEED + 307 * n + 17 * ci) & 0x7FFFFFFF,
                                   n_shuffles=GATE_N_SHUF)
            ok = res["verdict"] == expected
            rec = {"name": name, "n": n, "expected": expected, "verdict": res["verdict"],
                   "matches": bool(ok), "orient_auc": res["auc"],
                   "orient_null95": res["shuffle_null_95"],
                   "random_fold_auc": res["random_fold_auc"],
                   "max_fold_delta": res["max_fold_delta"],
                   "smuggle_reason": res["smuggle_reason"], "secs": time.time() - t}
            out["controls"].append(rec)
            print("    [%s] n=%2d %-36s verdict=%-13s (exp %-13s) auc=%.3f rf=%.3f delta=%.2g"
                  % ("OK " if ok else "!! ", n, name, res["verdict"], expected,
                     res["auc"], res["random_fold_auc"], res["max_fold_delta"]))

    out["elapsed_s"] = time.time() - t_all

    # --- outcome class ---
    b_verdicts = [r["verdict"] for r in out["B_orientation"]]
    c_verdicts = [r["verdict"] for r in out["C_congruence"]]
    controls_ok = all(c["matches"] for c in out["controls"])
    crossing = any(v == "PASS_CROSSING" for v in b_verdicts + c_verdicts)
    if crossing:
        outcome = "i_CROSSING__EXTRAORDINARY__REAUDIT_REQUIRED"
    elif all(v == "FAIL_CHANCE" for v in b_verdicts + c_verdicts) and controls_ok:
        outcome = "iii_FAIL_CHANCE__CONFIRMS__conjugate_quadrature_of_public_even_data_is_zero"
    else:
        outcome = "mixed__inspect"
    out["outcome_class"] = outcome
    out["controls_all_match"] = bool(controls_ok)

    with open(os.path.join(HERE, "holo_phase_substrate_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    print("\n" + "=" * 92)
    print("OUTCOME: %s" % outcome)
    print("controls all match: %s   total elapsed: %.1fs" % (controls_ok, out["elapsed_s"]))
    print("wrote holo_phase_substrate_result.json")
    print("=" * 92)
    return out


if __name__ == "__main__":
    run()
