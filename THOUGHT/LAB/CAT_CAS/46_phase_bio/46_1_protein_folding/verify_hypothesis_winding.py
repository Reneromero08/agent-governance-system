"""
Independent verification of Exp 46.1 hypothesis:
'The Point-Gap Winding Number strictly dictates the 3D folding class.'
Roadmap claim: W=0 -> alpha-helix, W=1 -> beta-sheet.

Tests the 1D chain Hamiltonian described in the roadmap (not 2D contact map).
"""
import numpy as np
import os

# Kyte-Doolittle hydrophobicity scale
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def build_1d_chain_H(seq, gamma=1.0):
    """1D chain Hamiltonian: diagonal = hydrophobicity dissipation,
    off-diagonal = nearest-neighbor hopping with steric frustration phase.
    This is the Hamiltonian described in the roadmap, NOT the 2D contact map."""
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        aa = seq[i] if seq[i] in KD else 'A'
        H[i, i] = -1j * gamma * KD[aa]
    for i in range(L):
        j = (i + 1) % L
        aa_i = seq[i] if seq[i] in KD else 'A'
        aa_j = seq[j] if seq[j] in KD else 'A'
        delta = abs(KD[aa_i] - KD[aa_j]) / 10.0
        t_fwd = 1.0 + delta
        t_bwd = 1.0 - delta
        H[j, i] = t_fwd
        H[i, j] = t_bwd
    return H

def compute_winding(H, n_phi=200):
    """Point-gap winding via global U(1) twist on off-diagonal elements."""
    D = np.diag(np.diag(H))
    O = H - D
    phis = np.linspace(0, 2*np.pi, n_phi)
    dets = np.zeros(n_phi, dtype=np.complex128)
    for k, phi in enumerate(phis):
        H_phi = D + np.exp(1j * phi) * O
        dets[k] = np.linalg.det(H_phi)
    phases = np.unwrap(np.angle(dets))
    return int(round((phases[-1] - phases[0]) / (2*np.pi)))

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("46.1 HYPOTHESIS TEST: Does winding number predict folding class?")
    log("=" * 70)
    log("")
    log("Roadmap claim: W=0 -> alpha-helix, W=1 -> beta-sheet")
    log("")

    # Known fold classes and their sequence patterns
    # Alpha-helical: periodic hydrophobic pattern (i, i+3, i+4)
    # Beta-sheet: alternating hydrophobic/hydrophilic
    # Random coil: no pattern

    tests = []

    # Poly-alanine (all hydrophobic) -> should be helical
    for L in [12, 24, 48]:
        seq = "A" * L
        H = build_1d_chain_H(seq)
        W = compute_winding(H)
        tests.append(("Poly-A (L={})".format(L), W, "alpha-helix"))

    # Leucine zipper pattern (L-x(6)-L-x(6)-L...) -> helical
    seq = "LAAAAAALAAAAAALAAAAAALAAAAAA"  # L at positions 0,7,14,21
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("Leu zipper (L=28)", W, "alpha-helix"))

    # Alternating hydrophobic/hydrophilic -> beta-sheet
    seq = "AVAVAVAVAVAVAVAVAVAVAVAV"  # A=hydrophobic, V=hydrophobic but different
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("AV repeat (L=24)", W, "beta-sheet"))

    # Mixed polar/nonpolar alternating -> beta
    seq = "KLKLKLKLKLKLKLKLKLKLKLKL"
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("KL repeat (L=24)", W, "beta-sheet"))

    # Glycine-Proline repeat -> beta-turn / random coil
    seq = "GP" * 12
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("GP repeat (L=24)", W, "random coil"))

    # Random sequence
    np.random.seed(42)
    seq = ''.join(np.random.choice(list(KD.keys()), 24))
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("Random (L=24)", W, "random coil"))

    # All-arginine (all hydrophilic) -> random coil
    seq = "R" * 24
    H = build_1d_chain_H(seq)
    W = compute_winding(H)
    tests.append(("Poly-R (L=24)", W, "random coil"))

    log(f"{'Sequence':<30s} {'W':>4s}  {'Expected':<15s}  {'Match?'}")
    log("-" * 60)
    correct = 0
    for name, W, expected in tests:
        if expected == "alpha-helix":
            match = "YES" if W == 0 else "NO"
            if W == 0: correct += 1
        elif expected == "beta-sheet":
            match = "YES" if W != 0 else "NO"
            if W != 0: correct += 1
        else:
            match = "ANY"
        log(f"{name:<30s} {W:+4d}  {expected:<15s}  {match}")

    log("")
    log("--- VERDICT ---")
    n_tests = len(tests)
    log("W=0 for helical, W!=0 for sheet: {}/{} correct ({:.0f}%)".format(
        correct, n_tests, 100*correct/n_tests if n_tests > 0 else 0))

    if correct == n_tests:
        log("HYPOTHESIS CONFIRMED: Winding number classifies fold type correctly.")
    elif correct >= n_tests * 0.7:
        log("HYPOTHESIS PARTIALLY SUPPORTED: Some sequences classified correctly.")
    else:
        log("HYPOTHESIS NOT SUPPORTED: Winding number does not predict fold class.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_46_1_HYPOTHESIS.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log("\nSaved: " + path)

if __name__ == "__main__":
    run()
