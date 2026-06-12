"""
Verify Exp 46.2 hypothesis: CTC Fixed-Point Iterator converges to Exceptional Point
where spectral gap collapses, representing the folding pathway. O(1) convergence.
"""
import numpy as np, os

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
      'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def build_frustrated_H(seq, coupling=1.0):
    """Build a frustrated (unfolded) Hamiltonian with strong non-reciprocal hopping."""
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * 0.1 * KD.get(seq[i], 1.8)
    for i in range(L):
        j = (i + 1) % L
        di, dj = KD.get(seq[i], 1.8), KD.get(seq[j], 1.8)
        delta = abs(di - dj)
        H[j, i] = coupling * (0.1 + delta * 2.0)
        H[i, j] = coupling * 0.1
    return H

def spectral_gap(H):
    evals = np.linalg.eigvals(H)
    return float(np.min(np.abs(evals)))

def eigenvector_overlap(H):
    _, evecs = np.linalg.eig(H)
    if evecs.shape[1] < 2:
        return 0.0
    v0, v1 = evecs[:, 0], evecs[:, 1]
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    return float(np.abs(np.dot(v0.conj(), v1)))

def ctc_folding_iterator(seq, max_iters=50, lr=0.1):
    """CTC iterator: drive coupling -> 0. At coupling=0 (balanced), gap collapses to EP.
    Each step: compute gap, reduce coupling proportionally to gap."""
    coupling = 1.0
    history = []
    for step in range(max_iters):
        H = build_frustrated_H(seq, coupling)
        gap = spectral_gap(H)
        overlap = eigenvector_overlap(H)
        history.append((step, coupling, gap, overlap))

        if gap < 1e-6 or coupling < 1e-6:
            break

        coupling = coupling - lr * gap
        if coupling < 0:
            coupling = 0.0
    return history

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.2 HYPOTHESIS: CTC FIXED-POINT ITERATOR -> EP CONVERGENCE")
    log("=" * 70)
    log("")

    tests = [
        ("GP-repeat (prion)", "GP" * 15),
        ("Random-30", ''.join(np.random.default_rng(42).choice(list(KD.keys()), 30))),
        ("AV-repeat", "AV" * 15),
    ]

    for name, seq in tests:
        log(f"--- {name} (L={len(seq)}) ---")
        history = ctc_folding_iterator(seq)

        log(f"  {'Step':<6s} {'Coupling':<12s} {'Gap':<14s} {'Overlap':<12s}")
        log(f"  {'-'*44}")
        converged_step = None
        for step, coupling, gap, overlap in history:
            log(f"  {step:<6d} {coupling:<12.6f} {gap:<14.6e} {overlap:<12.6f}")
            if gap < 1e-6 or coupling < 1e-6:
                converged_step = step

        log("")
        if converged_step is not None:
            log(f"  CONVERGED at step {converged_step} (gap={history[converged_step][2]:.2e})")
        else:
            log(f"  Did NOT converge in {len(history)} steps.")

        log("")

    log("--- VERDICT ---")
    log("CTC iterator should converge to EP (gap->0) in O(1) steps.")
    log("If convergence requires >>10 steps or never converges, hypothesis is falsified.")
    log("If convergence happens in <20 steps consistently, hypothesis holds.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_47_2_CTC.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"\nSaved: {path}")

if __name__ == "__main__":
    run()
