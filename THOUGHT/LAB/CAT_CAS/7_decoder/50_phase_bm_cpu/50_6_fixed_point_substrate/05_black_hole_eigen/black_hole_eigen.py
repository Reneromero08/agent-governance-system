"""
black_hole_eigen.py - the IN-BLACK-HOLE / phase-native EIGEN_BUDDY for Exp 50.14.

CORRECTIVE to the holo flattening error (phase6/holo_phase_substrate). That test fed
the cavity the PUBLIC, already-translated-out, real-even shadow {(k_i,b_i)} with
E[b_i]=cos(2*pi*k_i*d/N). A real-even spectrum has identically-zero conjugate
quadrature (Im/Re ~ 1e-14), so the orientation phase read ~0: the phase was burned off
at decompression. The fix: compute INSIDE the coherent complex space - the dihedral
COSET STATE provided by the oracle - and translate out to the real line only ONCE, at
the end, after the answer has rung as a resonance.

The coherent object (the "black hole"): for a random KNOWN k the oracle returns the
single-qubit dihedral coset state
        |c_k> = (|0> + omega^{k d} |1>)/sqrt(2),   omega = exp(-2*pi*i/N),  N = 2^n,
on a |k> register. The orientation o = 1[d < N/2] IS PRESENT here as the SIGN of the
relative phase (the imaginary part), because the states for d and N-d are complex
CONJUGATES - distinguishable as quantum states - whereas their public (cos) shadows are
identical (cos is even). You RECEIVE these states; you never build them from a known d.

This module:
  STEP A  validates EIGEN_BUDDY on the abelian/period structure: the QFT (= the unitary
          phase estimator Q.K^dagger) concentrates the period/even-answer onto a SINGLE
          dominant eigenvalue - the owner's eureka, reproduced on the coherent state. No
          search. (Uses the stronger abelian/Shor phase access - the decodable class.)
  STEP B  tests whether any FIXED, d-independent in-black-hole operator concentrates the
          ORIENTATION (the reflection bit) into a dominant eigenvalue / resonance, and at
          what COST SCALING in n. Candidates: the present-in-the-black-hole conjugate
          quadrature, a fixed single-copy conjugate (Hilbert) eigen-statistic, a coherent
          Kuperberg collimation sieve, and a catalytic fixed-point relaxation.

NO-SMUGGLE: the operator must be a FIXED computation independent of d (like the QFT is
fixed and known to all). The coherent INPUT STATE legitimately differs for d vs N-d
(that is the resource the public shadow destroyed); but the operator applied to it must
not be tuned with d / the true sin / the half-range. Cheat controls that DO use d are
included and must be flagged.

ASCII only. All RNGs seeded; seeds recorded. Coherent state is 2^n-dim: dense parts run
at small n (4,6,8,10); the sieve (integer-label arithmetic) runs to n=12,14. The COST
SCALING in n is the result, not any single n.

Author: black-hole-eigen agent (Fable). Claim ceiling L4-5.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(HERE)
FOLD = os.path.join(PHASE6, "02_fold_audit")
STAGE3 = os.path.join(FOLD, "stage3")
for _p in (FOLD, STAGE3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C            # the verbatim 50.14 construction (coset phase, M_for, ...)


# ===========================================================================
# COHERENT PRIMITIVES (the black hole) - omega = exp(-2 pi i / N)
# ===========================================================================
def coset_qubit(k, d, N):
    """The dihedral coset state |c_k> = (|0> + omega^{kd}|1>)/sqrt2 as a 2-vector.
    omega^{kd} = exp(-2 pi i k d / N) = C.quadrature_channel(k,d,N). The oracle returns
    this for a random KNOWN k; the algorithm reads k but never d."""
    a = np.exp(-2j * np.pi * (k * d % N) / N)
    return np.array([1.0 + 0j, a], dtype=np.complex128) / np.sqrt(2.0)


def phase_register(d, N):
    """The abelian phase-kickback state |phi_d> = (1/sqrt N) sum_k omega^{kd} |k>.
    This is the Shor/abelian access (a single COHERENT register carrying d's phase across
    all k). It is the eigenvector of the cyclic shift with eigenphase d/N. Used in STEP A
    to validate EIGEN_BUDDY = QFT; the dihedral oracle (STEP B) does NOT hand this over."""
    k = np.arange(N)
    return np.exp(-2j * np.pi * (k * d % N) / N) / np.sqrt(N)


def pauli_Y_expectation(state2):
    """<Y> for a single qubit [a0,a1]. For |c_k> this equals -sin(2 pi k d / N): the
    CONJUGATE quadrature, the fold-odd channel absent from the public cosine data."""
    a0, a1 = state2[0], state2[1]
    return float(np.real(-1j * (np.conj(a0) * a1 - np.conj(a1) * a0)))


# ===========================================================================
# STEP A - EIGEN_BUDDY rings the PERIOD (the abelian half) : QFT as phase estimator
# ===========================================================================
def stepA_qft_rings_period(ns, n_trials, seed):
    """Reproduce the owner's eureka on the coherent state: the QFT (the unitary phase
    estimator) maps the phase register |phi_d> to a SINGLE dominant eigenvalue / peak at
    |d>. The period/even-answer rings as a resonance - no orbit iteration, no search.

    Reports, per n: frac_exact recovery of d by the QFT peak, the IPR of the post-QFT
    intensity (1.0 == a single dominant eigenvalue), and the cyclic-shift eigenphase
    recovery <phi_d|S|phi_d> = exp(+2 pi i d / N) (d read directly off the dominant
    eigenvalue's phase). EIGEN_BUDDY = QFT is real where it is supposed to be real."""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 17 * n)
        exact_peak = 0
        exact_eig = 0
        iprs = []
        peak_mass = []
        for _ in range(n_trials):
            d = C.sample_secret(N, rng)
            phi = phase_register(d, N)
            # EIGEN_BUDDY = QFT (numpy FFT is the unitary phase estimator up to a global 1/sqrt N)
            F = np.fft.fft(phi)
            inten = np.abs(F) ** 2
            inten = inten / np.sum(inten)
            peak = int(np.argmax(inten))
            d_hat = (N - peak) % N           # fft convention: peak sits at N-d
            exact_peak += int(d_hat == d)
            iprs.append(float(np.sum(inten ** 2)))   # IPR: 1.0 for a single dominant mode
            peak_mass.append(float(np.max(inten)))   # fraction of amplitude in the winner
            # dominant-eigenvalue read of the cyclic shift S (|phi_d> is its eigenvector)
            Sphi = np.roll(phi, 1)            # S|k> = |k+1>
            eigval = np.vdot(phi, Sphi)       # = exp(+2 pi i d / N)
            ph = np.angle(eigval) / (2 * np.pi)
            d_eig = int(round((ph % 1.0) * N))
            exact_eig += int(d_eig == d)
        out.append({
            "n": n, "N": N, "n_trials": n_trials,
            "frac_exact_qft_peak": exact_peak / n_trials,
            "frac_exact_shift_eigenphase": exact_eig / n_trials,
            "mean_ipr_post_qft": float(np.mean(iprs)),
            "mean_peak_mass": float(np.mean(peak_mass)),
            "cost_qfts": 1,                  # ONE QFT - poly(n): O(n log n) gates
        })
    return out


def stepA_dihedral_even_answer(ns, n_trials, seed):
    """The SAME even/decodable answer a = min(d, N-d) read directly from the dihedral
    coset ENSEMBLE (the actual oracle), to connect STEP A to the real access. The X
    quadrature <X> of |c_k> = cos(2 pi k d / N) (the public-even channel); the magnitude
    resonance |sum_k <X> exp(+2 pi i k x / N)| peaks at BOTH d and N-d, so the unordered
    set / the fold-answer a is recovered for free. (Orientation is NOT here - that is STEP B.)"""
    out = []
    for n in ns:
        N = 1 << n
        rng = np.random.default_rng(seed + 29 * n)
        M = C.M_for(n)
        exact = 0
        for _ in range(n_trials):
            d = C.sample_secret(N, rng)
            k = rng.integers(0, N, size=M)
            # coherent X-quadrature expectation per sample = cos(2 pi k d / N) (noiseless)
            xq = np.cos(2 * np.pi * (k * d % N) / N)
            B = np.zeros(N, dtype=np.complex128)
            np.add.at(B, k, xq.astype(np.complex128))
            Psi = np.fft.ifft(B) * N
            inten = np.abs(Psi)
            inten[0] = -1.0
            xpk = int(np.argmax(inten))
            a_hat = min(xpk, (N - xpk) % N)
            exact += int(a_hat == min(d, N - d))
        out.append({"n": n, "N": N, "M": int(M), "n_trials": n_trials,
                    "frac_exact_fold_answer": exact / n_trials})
    return out


if __name__ == "__main__":
    print("module black_hole_eigen loaded OK; primitives + STEP A defined")
