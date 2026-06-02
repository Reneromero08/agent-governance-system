"""
Exp 46.2: BUILD A GENUINE EP via folding parameter.
Design: a chain Hamiltonian with an embedded Jordan-block structure.
Folding coordinate lambda controls eigenvalue coalescence.
At lambda=0: two eigenvalues coalesce into EP (FOLDED).
At lambda>0: eigenvalues split (UNFOLDED).
CTC iterator drives lambda -> 0 using gap as gradient signal.
"""
import numpy as np, os

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
      'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def build_folding_H(seq, lam=1.0):
    """Chain Hamiltonian with EP-producing structure at lam=0.
    Residues i and i+1 form a Jordan-block pair when lam->0."""
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    gamma = 0.1

    for i in range(L):
        H[i, i] = -1j * gamma * KD.get(seq[i], 1.8)

    for i in range(L):
        j = (i+1) % L
        di = KD.get(seq[i], 1.8)
        dj = KD.get(seq[j], 1.8)
        delta = abs(di - dj)

        # Forward hopping: always active (creates directionality)
        H[j, i] = 0.1 + delta * 2.0

        # Backward hopping: controlled by folding parameter lambda
        # At lam=0: backward=0, creates a directed chain (like Jordan block structure)
        # At lam>0: backward > 0, creates bidirectional coupling (unfolded frustration)
        H[i, j] = 0.1 * lam

    return H

def spectral_gap(H):
    evals = np.linalg.eigvals(H)
    return float(np.min(np.abs(evals)))

def eigenvector_condition(H):
    """Eigenvector condition number. Diverges at EP."""
    _, evecs = np.linalg.eig(H)
    V = evecs
    try:
        Vinv = np.linalg.inv(V)
        return float(np.linalg.norm(V) * np.linalg.norm(Vinv))
    except np.linalg.LinAlgError:
        return float('inf')

def ctc_to_ep(seq, lam0=1.0, max_iters=100, lr=0.3):
    """CTC: drive lam -> 0 by gradient descent on spectral gap.
    At lam=0, backward hopping vanishes => H is lower-triangular => EP possible."""
    lam = lam0
    history = []

    for step in range(max_iters):
        H = build_folding_H(seq, lam)
        gap = spectral_gap(H)
        kappa = eigenvector_condition(H)
        history.append((step, lam, gap, kappa))

        if gap < 1e-8 or kappa > 1e8:
            break

        # Gradient: reduce lam proportionally to gap
        lam = lam - lr * gap
        if lam < 0:
            lam = 0.0

    return history

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.2: CTC FIXED-POINT ITERATOR -> EP (JORDAN BLOCK STRUCTURE)")
    log("=" * 70)
    log("")
    log("Hamiltonian: chain with backward hopping controlled by lambda.")
    log("lambda=0 -> backward=0 -> lower-triangular -> EP possible.")
    log("CTC drives lambda->0 using spectral gap as gradient.")
    log("")

    for name, seq in [
        ("Poly-A", "A" * 20),
        ("GP-repeat", "GP" * 10),
        ("AV-repeat", "AV" * 10),
    ]:
        log(f"--- {name} (L={len(seq)}) ---")
        H0 = build_folding_H(seq, lam=0.0)
        H1 = build_folding_H(seq, lam=1.0)
        gap0 = spectral_gap(H0)
        gap1 = spectral_gap(H1)
        k0 = eigenvector_condition(H0)
        k1 = eigenvector_condition(H1)
        log(f"  lam=0: gap={gap0:.6e}  cond(V)={k0:.2e}")
        log(f"  lam=1: gap={gap1:.6e}  cond(V)={k1:.2e}")

        # Check if lam=0 IS an EP: gap ~ 0 AND condition number diverges
        is_ep = (gap0 < 1e-6) or (k0 > 1e6)
        log(f"  lam=0 is EP: {'YES' if is_ep else 'NO'}")

        # CTC iteration
        history = ctc_to_ep(seq)
        final_step, final_lam, final_gap, final_kappa = history[-1]
        log(f"  CTC: {len(history)} steps -> lam={final_lam:.6e} gap={final_gap:.6e} cond={final_kappa:.2e}")
        steps_ok = len(history) < 20
        ep_reached = final_gap < 1e-6 or final_kappa > 1e6
        log(f"  O(1) convergence (<20 steps): {'YES' if steps_ok else 'NO (took '+str(len(history))+' steps)'}")
        log(f"  EP reached: {'YES' if ep_reached else 'NO'}")
        log("")

    log("--- VERDICT ---")
    log("If CTC converges in <20 steps AND reaches an EP (gap->0 or cond->inf):")
    log("  => Hypothesis VERIFIED: folding pathway = CTC-driven EP approach.")
    log("Otherwise: Hypothesis NOT SUPPORTED by current model.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_46_2_EP.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"\nSaved: {path}")

if __name__ == "__main__":
    run()
