import numpy as np
import matplotlib.pyplot as plt

KD = {'A': 1.8}
BULK = {'A': 88}

def build_H(seq, gamma, theta=0.0):
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * gamma * KD[seq[i]]
        next_i = (i + 1) % L
        frust = abs(BULK[seq[i]] - BULK[seq[next_i]]) / 100.0
        t_fwd = 2.0 * (1.0 + 2.0 * frust)
        t_bwd = 2.0 * (1.0 - 2.0 * frust)
        phi = (BULK[seq[i]] + BULK[seq[next_i]]) / 500.0 * np.pi
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
    return H

seq = "A" * 30
gammas = np.linspace(0.0, 2.0, 41)
for gamma in gammas:
    H = build_H(seq, gamma, 0.0)
    evals = np.linalg.eigvals(H)
    
    # Let's calculate different metrics for "Spectral Gap"
    # 1. Minimum absolute value (Gap to origin)
    gap_origin = np.min(np.abs(evals))
    
    # 2. Minimum real part gap
    real_evals = np.sort(np.real(evals))
    gap_real = np.min(np.diff(real_evals))
    
    # 3. Minimum imaginary part gap
    imag_evals = np.sort(np.imag(evals))
    gap_imag = np.min(np.diff(imag_evals))
    
    # 4. Pairwise absolute gap
    diffs = np.abs(evals[:, np.newaxis] - evals)
    np.fill_diagonal(diffs, np.inf)
    gap_pair = np.min(diffs)
    
    print(f"G: {gamma:.2f} | Orig: {gap_origin:.4f} | Real: {gap_real:.4f} | Imag: {gap_imag:.4f} | Pair: {gap_pair:.4f}")
