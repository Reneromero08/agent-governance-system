"""
operator.py - Coupled-oscillator / Kuramoto-CGLE encoding of the Exp 50.14 public
fixed-point construction, plus the non-Hermitian (non-reciprocal, chiral) variant.

THESIS UNDER TEST (Phase 6 frontier brief):
  Public data has E[b_i] = cos(2 pi k_i d / N): EVEN in d, invariant under the fold
  sigma: d <-> N-d. The orientation bit o = 1[d < N/2] lives ONLY in the ABSENT odd
  channel sin(2 pi k d / N). The brief's KEY HYPOTHESIS is that the MAP f's
  DIRECTIONALITY (the +1 increment, a non-reciprocal flow) might inject the missing odd
  bit as a non-Hermitian / chiral order-parameter-PHASE signature, read for free from a
  synchronizing oscillator network, NOT from the even cosine magnitudes.

  Three encodings, audited by the gate:
    (A) static complex resonance Z(x) = (1/M) sum_i b_i exp(i 2 pi k_i x / N): the
        Kuramoto order parameter vs trial frequency x. |Z(x)| peaks at x in {d, N-d}
        (even; recovers the SET). arg Z(x) is the candidate odd channel under test.
    (B) genuine dynamical mean-field Kuramoto: M oscillators, natural freq 2 pi k_i / N,
        signed weights b_i; integrate to sync; read order param (R, psi).
    (C) NON-RECIPROCAL / chiral Kuramoto (Sakaguchi phase-lag alpha): the directed,
        non-Hermitian version encoding the +1 chirality of f; read the chiral phase drift.

  Controls:
    - O_phase_estimation_dyadic: SPEC-1C one-shot phase estimation on PUBLIC data
      (quadrature absent -> phases 0/pi -> information-empty).
    - O_quadrature_smuggle: reads the genuine sin(2 pi k d / N) from hidden d. A SMUGGLE;
      MUST lift AUC and MUST be caught (delta>0). Proves the harness can SEE orientation
      when it is present (sensitivity control).
    - Null/baseline discipline is supplied by no_smuggle_gate.py: honest operators are
      checked against chance/null orientation recovery, while the smuggle control is the
      positive baseline that must fail the public-data seal.

Contract (no_smuggle_gate.py): O(instance) -> 1D float vector. instance keys: k, b, N,
  d (HIDDEN; only the smuggle control may read it), n. Honest candidates read only
  k, b, N, n. Randomness seeded deterministically from PUBLIC data so the exact
  d-invariance audit holds. ASCII only. Claim ceiling L4-5; the gate renders verdicts.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FOLD_AUDIT = os.path.abspath(os.path.join(_HERE, "..", "..", "fold_audit"))
_STAGE3 = os.path.join(_FOLD_AUDIT, "stage3")
for _p in (_FOLD_AUDIT, _STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C  # noqa: E402  (single source of truth; reimplement nothing)


# ---------------------------------------------------------------------------
# Core primitives (PUBLIC: functions of k, b, N only)
# ---------------------------------------------------------------------------
def complex_resonance(k, b, N, xs):
    """Z(x) = (1/M) sum_i b_i exp(i 2 pi k_i x / N) at trial points xs. The weighted
    Kuramoto order parameter of M oscillators with phases theta_i(x)=2 pi k_i x/N and
    signed weights b_i. PURE PUBLIC.

    Fold identity (explicit): Z(N - x) = conj(Z(x)) because cos is even, sin is odd. The
    resonance curve is symmetric about the real axis; the phase at x=d is the exact
    negative of the phase at x=N-d, so the pair carries no absolute orientation."""
    k = np.asarray(k); b = np.asarray(b, dtype=float); xs = np.asarray(xs, dtype=float)
    M = len(b)
    ph = np.exp(2j * np.pi * np.outer(k.astype(np.float64), xs) / N)  # M x len(xs)
    return (b @ ph) / M


def _public_seed(k, b, N):
    """Deterministic seed from PUBLIC data only (O byte-identical on inst and its fold)."""
    h = (int(np.sum(k.astype(np.int64)) * 1000003)
         ^ int(np.sum((b > 0).astype(np.int64)) * 2654435761)
         ^ (int(N) * 40503))
    return h & 0x7FFFFFFF


def kuramoto_meanfield(omega, weight, K=2.5, alpha=0.0, T=120, dt=0.05, theta0=None):
    """Integrate the weighted mean-field (Sakaguchi-)Kuramoto model:

        Z(t) = (1/M) sum_j weight_j exp(i theta_j)
        dtheta_i/dt = omega_i + K * |Z| * sin(arg Z - theta_i + alpha)

    alpha != 0 is the NON-RECIPROCAL / chiral phase lag (the directed, non-Hermitian
    term); alpha=0 is ordinary reciprocal Kuramoto. Mean-field form is O(M) per step.
    Returns the trajectory of the complex order parameter Z(t) (length T+1)."""
    omega = np.asarray(omega, dtype=float)
    weight = np.asarray(weight, dtype=float)
    M = len(omega)
    theta = np.zeros(M) if theta0 is None else np.array(theta0, dtype=float)
    traj = np.empty(T + 1, dtype=complex)
    for t in range(T + 1):
        Z = np.mean(weight * np.exp(1j * theta))
        traj[t] = Z
        if t == T:
            break
        R = np.abs(Z); psi = np.angle(Z)
        theta = theta + dt * (omega + K * R * np.sin(psi - theta + alpha))
    return traj


def winding_number(zcurve):
    """Point-gap winding W = (1/2pi) * total change of arg(z) around the loop. For a
    conjugation-symmetric curve (Z(N-x)=conj Z(x)) about a real base point W ~ 0 in
    expectation: upper and lower halves cancel."""
    ang = np.unwrap(np.angle(zcurve))
    return float((ang[-1] - ang[0]) / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# CANDIDATE A - static complex-resonance order-parameter PHASE
# ---------------------------------------------------------------------------
def O_resonance_phase(inst):
    """Order parameter Z(x) vs trial frequency x. |Z(x)| peaks at {d, N-d} (even). We
    hand the probe the PHASE features the hypothesis says might carry o: phase and Im at
    the global |Z| argmax over a PUBLIC grid (no [1,N/2) range restriction -- that would
    be the smuggle); the winding number (a non-Hermitian point-gap invariant); Re/Im/arg
    of Z at the dyadic rungs; magnitude moments. Prediction: chance for o."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    G = min(N, 256)
    xs = np.linspace(0.0, N, G, endpoint=False)
    Z = complex_resonance(k, b, N, xs)
    mag = np.abs(Z)
    j = int(np.argmax(mag))
    feats = [
        float(np.angle(Z[j])),            # phase at the dominant resonance
        float(np.imag(Z[j])),             # quadrature at the dominant resonance
        float(np.sign(np.imag(Z[j]))),    # candidate orientation read
        winding_number(Z),                # point-gap winding (non-Hermitian invariant)
        float(np.mean(mag)), float(np.std(mag)), float(np.max(mag)),
    ]
    rungs = np.array([(N >> jj) for jj in range(1, n + 1)], dtype=float)  # N/2 ... 1
    Zr = complex_resonance(k, b, N, rungs)
    feats.extend(list(np.real(Zr)))
    feats.extend(list(np.imag(Zr)))
    feats.extend(list(np.angle(Zr)))
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# CANDIDATE B - genuine dynamical Kuramoto synchronization (reciprocal)
# ---------------------------------------------------------------------------
def O_kuramoto_orderparam(inst):
    """M oscillators: omega_i = 2 pi k_i / N (public), signed weight b_i (public), public
    initial phases. Integrate the reciprocal mean-field Kuramoto to sync and read the
    ORDER PARAMETER (R, psi) -- the headline test of whether psi (odd-looking) recovers o
    or only R (even) does. Prediction: chance for o (omega, b are even in d; the flow law
    is fold-invariant in distribution)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    M = len(b)
    omega = 2.0 * np.pi * k.astype(np.float64) / N
    rng = np.random.default_rng(_public_seed(k, b, N))
    theta0 = rng.uniform(-np.pi, np.pi, size=M)
    traj = kuramoto_meanfield(omega, b, K=2.5, alpha=0.0, T=120, dt=0.05, theta0=theta0)
    Zf = traj[-1]
    R = float(np.abs(Zf)); psi = float(np.angle(Zf))
    half = len(traj) // 2
    late_drift = float(np.angle(traj[-1]) - np.angle(traj[half]))
    feats = [
        R, float(np.cos(psi)), float(np.sin(psi)),
        float(np.real(Zf)), float(np.imag(Zf)),
        float(np.mean(np.abs(traj))), float(np.imag(Zf)),
        late_drift,
    ]
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# CANDIDATE C - NON-RECIPROCAL / chiral Kuramoto (directed, non-Hermitian route)
# ---------------------------------------------------------------------------
def O_nonreciprocal_chiral(inst):
    """The KEY HYPOTHESIS made concrete: encode the +1 directionality of f as a
    NON-RECIPROCAL Sakaguchi phase lag alpha != 0. This makes the synchronized-state
    Jacobian non-normal (non-Hermitian) and produces a CHIRAL, rotating order parameter
    whose phase DRIFTS. We read drift rate and chiral phase, plus the +alpha vs -alpha
    handedness contrast (the hypothesis's strongest shot).

    Measured point: alpha is a GLOBAL constant shared by both fixed points; public omega,
    b are even in d. So the chirality is ORIENTATION-INDEPENDENT -- it reads 'a directed
    current exists' (true for both d and N-d) but not WHICH of {d, N-d} is < N/2.
    Prediction: chance for o."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]
    M = len(b)
    omega = 2.0 * np.pi * k.astype(np.float64) / N
    rng = np.random.default_rng(_public_seed(k, b, N))
    theta0 = rng.uniform(-np.pi, np.pi, size=M)
    alpha = 0.5 * np.pi - 0.3   # strong non-reciprocal chiral lag
    tp = kuramoto_meanfield(omega, b, K=2.5, alpha=+alpha, T=120, dt=0.05, theta0=theta0)
    tm = kuramoto_meanfield(omega, b, K=2.5, alpha=-alpha, T=120, dt=0.05, theta0=theta0)

    def drift(tr):
        a = np.unwrap(np.angle(tr)); h = len(a) // 2
        return float((a[-1] - a[h]) / (len(a) - h))

    feats = [
        drift(tp), drift(tm), drift(tp) + drift(tm),     # handedness contrast
        float(np.abs(tp[-1])), float(np.angle(tp[-1])),
        float(np.imag(tp[-1])), float(np.imag(tm[-1])),
        float(np.imag(tp[-1]) - np.imag(tm[-1])),
    ]
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# CONTROL - SPEC-1C dyadic phase estimation on PUBLIC data (quadrature absent)
# ---------------------------------------------------------------------------
def O_phase_estimation_dyadic(inst):
    """SPEC-1C 'one-shot' recipe: estimate the complex coefficient at each dyadic rung
    kappa in {1,...,N/2} and read its phase to pin a bit of d. On PUBLIC data the estimate
    is cos_hat[kappa]=E[b|k=kappa] (REAL); the imaginary/sin part is ABSENT. The formed
    phase arg(cos_hat + i*0) is in {0, pi}: carries the even bits, NO orientation.
    Prediction: chance for o (phase estimation is information-empty without quadrature)."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; n = inst["n"]
    cos_hat = np.zeros(N); counts = np.zeros(N)
    np.add.at(cos_hat, k, b); np.add.at(counts, k, 1.0)
    nz = counts > 0
    cos_hat[nz] /= counts[nz]
    rungs = [(N >> j) for j in range(1, n + 1)]  # N/2 ... 1
    feats = []
    for r in rungs:
        c = float(cos_hat[r % N])
        feats.append(c)
        feats.append(float(np.angle(c + 0.0j)))   # 0 or pi: information-empty phase
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# SENSITIVITY CONTROL - genuine quadrature read from hidden d (SMUGGLE)
# ---------------------------------------------------------------------------
def O_quadrature_smuggle(inst):
    """KNOWN SMUGGLE. Reads sin(2 pi kappa d / N) at low rungs from the HIDDEN d. Each sin
    flips sign under d -> N-d, so it recovers o AND is caught by the exact d-invariance
    audit (delta>0). The gate MUST report FAIL_SMUGGLE. Proves the harness can SEE
    orientation when present, so a chance result on honest candidates is a real
    information-absence finding, not a blind instrument."""
    N = inst["N"]; d = inst["d"]
    return np.array([float(np.sin(2 * np.pi * j * d / N)) for j in (1, 2, 3, 4)])


# ---------------------------------------------------------------------------
# unoriented-recovery probe: does the resonance recover the SET {d, N-d}? (alive test)
# ---------------------------------------------------------------------------
def resonance_recovers_set(inst):
    """argmax_x |Z(x)| over the FULL grid x=0..N-1; returns (peak_x, hits_set) with
    hits_set = peak within +-1 of d or N-d. Demonstrates the network recovers the
    unoriented secret (even channel works); orientation is the only thing missing. Costs
    O(M*N) -- this is the readout scan priced in run_kuramoto."""
    k = inst["k"]; b = inst["b"]; N = inst["N"]; d = inst["d"]
    xs = np.arange(N, dtype=float)
    Z = complex_resonance(k, b, N, xs)
    px = int(np.argmax(np.abs(Z)))
    nd = (N - d) % N
    dd = d % N
    hit = (min(abs(px - dd), N - abs(px - dd)) <= 1) or (min(abs(px - nd), N - abs(px - nd)) <= 1)
    return px, bool(hit)
