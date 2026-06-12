import numpy as np

KD = {'A': 1.8}
BULK = {'A': 88}

def build_H(gamma):
    L = 30
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * gamma * 1.8
        next_i = (i + 1) % L
        phi = (88 + 88) / 500.0 * np.pi
        if i == L - 1:
            H[i, next_i] = 2.0 * np.exp(1j * phi)
            H[next_i, i] = 2.0 * np.exp(-1j * phi)
        else:
            H[i, next_i] = 2.0 * np.exp(1j * phi)
            H[next_i, i] = 2.0 * np.exp(-1j * phi)
    return H

gammas = np.linspace(0.0, 2.0, 100)
min_orig = 999
min_gamma_orig = 0

for gamma in gammas:
    H = build_H(gamma)
    evals = np.linalg.eigvals(H)
    gap_orig = np.min(np.abs(evals))
    if gap_orig < min_orig:
        min_orig = gap_orig
        min_gamma_orig = gamma
        
print(f"Min Gap to Origin: {min_orig} at Gamma: {min_gamma_orig}")

