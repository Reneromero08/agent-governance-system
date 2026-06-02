"""
Exp 46.2: Open-chain Hamiltonian with EP at lam=0.
Open chain (no ring closure): strictly lower-triangular at lam=0 -> Jordan block -> EP.
CTC iterator drives lam->0 using eigenvalue separation as gradient.
"""
import numpy as np, os

KD = {'A':1.8,'G':-0.4,'P':-1.6,'V':4.2,'R':-4.5,'L':3.8,'K':-3.9}

def build_open_H(seq, lam=1.0):
    """Open chain (no ring closure). lam controls backward hopping.
    lam=0: strictly lower-triangular -> EP for uniform sequences.
    lam=1: bidirectional -> distinct eigenvalues -> UNFOLDED."""
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * 0.1 * KD.get(seq[i], 1.8)
    for i in range(L-1):  # open chain: no (L-1)->0 connection
        j = i + 1
        di = KD.get(seq[i], 1.8)
        dj = KD.get(seq[j], 1.8)
        delta = abs(di - dj)
        H[j, i] = 0.1 + delta * 2.0  # forward
        H[i, j] = 0.1 * lam          # backward (controlled by lam)
    return H

def eig_separation(H):
    evals = np.sort_complex(np.linalg.eigvals(H))
    diffs = [abs(evals[i+1]-evals[i]) for i in range(len(evals)-1)]
    return float(min(diffs)) if diffs else 0

def cond_number(H):
    _, V = np.linalg.eig(H)
    try:
        Vi = np.linalg.inv(V)
        return float(np.linalg.norm(V) * np.linalg.norm(Vi))
    except np.linalg.LinAlgError:
        return float('inf')

def ctc_drive(seq, lam0=1.0, max_iters=100, lr=0.5):
    lam = lam0
    history = []
    for step in range(max_iters):
        H = build_open_H(seq, lam)
        sep = eig_separation(H)
        cond = cond_number(H)
        history.append((step, lam, sep, cond))
        if sep < 1e-12 or cond > 1e8:
            break
        lam -= lr * sep
        if lam < 0: lam = 0.0
    return history

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.2: OPEN-CHAIN EP + CTC FOLDING ITERATOR")
    log("=" * 70)

    for name, seq in [
        ("Poly-A (uniform)", "A" * 15),
        ("Poly-R (uniform)", "R" * 15),
        ("GP-repeat (prion)", "GP" * 7 + "G"),
        ("AV-repeat", "AV" * 7 + "A"),
        ("Random mix", "ARKLGV" * 2 + "ARK"),
    ]:
        log("")
        log("--- %s (L=%d) ---" % (name, len(seq)))

        # Check lam=0 (folded) vs lam=1 (unfolded)
        H0 = build_open_H(seq, 0.0)
        H1 = build_open_H(seq, 1.0)
        sep0 = eig_separation(H0)
        sep1 = eig_separation(H1)
        cond0 = cond_number(H0)

        is_ep_at_0 = sep0 < 1e-12 or cond0 > 1e6
        log("  lam=0 (FOLDED):  sep=%.2e  cond=%.2e  EP=%s" % (sep0, cond0, "YES" if is_ep_at_0 else "NO"))
        log("  lam=1 (UNFOLDED): sep=%.2e" % sep1)

        # CTC iteration
        h = ctc_drive(seq)
        final_step, final_lam, final_sep, final_cond = h[-1]
        ep_reached = final_sep < 1e-12 or final_cond > 1e6
        steps_ok = len(h) < 30
        log("  CTC: %d steps -> lam=%.6e  sep=%.2e  cond=%.2e" % (len(h), final_lam, final_sep, final_cond))
        log("  O(1) (<30 steps): %s  EP reached: %s" % ("YES" if steps_ok else "NO", "YES" if ep_reached else "NO"))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_46_2_OPEN_CHAIN.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    for l in lines: print(l)

if __name__ == "__main__":
    run()
