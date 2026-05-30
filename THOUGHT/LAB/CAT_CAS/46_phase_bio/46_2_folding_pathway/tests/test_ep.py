import numpy as np

# Kyte-Doolittle Hydrophobicity
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Approximate Amino Acid Volume (Bulk in A^3)
BULK = {
    'G': 60, 'A': 88, 'S': 89, 'C': 108, 'T': 116, 'P': 122, 'D': 111, 'N': 114,
    'V': 140, 'E': 138, 'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168,
    'R': 173, 'F': 189, 'Y': 193, 'W': 227
}

def build_H(seq, gamma, theta=0.0):
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    
    for i in range(L):
        # Aqueous Bath: Gamma scales the imaginary potential
        H[i, i] = -1j * gamma * KD[seq[i]]
        
        next_i = (i + 1) % L
        
        # Steric frustration
        frust = abs(BULK[seq[i]] - BULK[seq[next_i]]) / 100.0
        t_fwd = 2.0 * (1.0 + 2.0 * frust)
        t_bwd = 2.0 * (1.0 - 2.0 * frust)
        
        # Steric bulk -> Complex phase hopping
        phi = (BULK[seq[i]] + BULK[seq[next_i]]) / 500.0 * np.pi
        
        # Twist only on the boundary for Point-Gap Winding
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
            
    return H

def analyze_sweep():
    seq = "A" * 30
    gammas = np.linspace(0.0, 2.0, 41)
    
    for gamma in gammas:
        H0 = build_H(seq, gamma, 0.0)
        evals = np.linalg.eigvals(H0)
        
        # Min pairwise gap (Exceptional Point metric)
        diffs = np.abs(evals[:, np.newaxis] - evals)
        np.fill_diagonal(diffs, np.inf)
        gap_pairwise = np.min(diffs)
        gap_origin = np.min(np.abs(evals))
        
        # Max imaginary part
        max_imag = np.max(np.abs(np.imag(evals)))
        
        # Winding
        thetas = np.linspace(0, 2*np.pi, 100)
        dets = []
        for th in thetas:
            H_th = build_H(seq, gamma, th)
            dets.append(np.linalg.det(H_th))
        phases = np.unwrap(np.angle(dets))
        W = (phases[-1] - phases[0]) / (2 * np.pi)
        
        print(f"Gamma: {gamma:.2f} | Gap Delta E: {gap_origin:.4f} | W: {round(W)}")

analyze_sweep()
