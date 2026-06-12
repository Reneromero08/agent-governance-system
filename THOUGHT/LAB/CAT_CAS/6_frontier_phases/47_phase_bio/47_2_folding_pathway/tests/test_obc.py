import numpy as np

KD = {'A': 1.8}
BULK = {'A': 88}

def build_H_obc(gamma):
    L = 30
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * gamma * 1.8
        if i < L - 1:
            phi = (88 + 88) / 500.0 * np.pi
            H[i, i+1] = 2.0 * np.exp(1j * phi)
            H[i+1, i] = 2.0 * np.exp(-1j * phi)
    return H

gammas = np.linspace(0.0, 2.0, 41)
for gamma in gammas:
    H = build_H_obc(gamma)
    evals = np.linalg.eigvals(H)
    gap_orig = np.min(np.abs(evals))
    diffs = np.abs(evals[:, np.newaxis] - evals)
    np.fill_diagonal(diffs, np.inf)
    gap_pair = np.min(diffs)
    print(f"G: {gamma:.2f} | Orig: {gap_orig:.4f} | Pair: {gap_pair:.4f}")
