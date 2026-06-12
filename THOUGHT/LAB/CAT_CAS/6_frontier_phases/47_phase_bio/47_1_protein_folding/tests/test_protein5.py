import numpy as np

KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
BULK = {
    'G': 60, 'A': 88, 'S': 89, 'C': 108, 'T': 116, 'P': 122, 'D': 111, 'N': 114,
    'V': 140, 'E': 138, 'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168,
    'R': 173, 'F': 189, 'Y': 193, 'W': 227
}

def build_H(seq, theta=0.0):
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    
    # Calculate sequence specific center to make the origin the topological reference
    mean_kd = np.mean([KD[aa] for aa in seq])
    
    for i in range(L):
        # Center the on-site potential around the origin
        H[i, i] = -1j * (KD[seq[i]] - mean_kd)
        
        next_i = (i + 1) % L
        
        # Steric frustration -> non-reciprocal hopping magnitude
        frust = abs(BULK[seq[i]] - BULK[seq[next_i]]) / 100.0
        t_fwd = 2.0 * (1.0 + frust)
        t_bwd = 2.0 * (1.0 - frust)
        
        phi = (BULK[seq[i]] + BULK[seq[next_i]]) / 500.0 * np.pi
        
        # Twist only on the boundary
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
            
    return H

def get_metrics(seq):
    H0 = build_H(seq, 0.0)
    evals = np.linalg.eigvals(H0)
    gap = np.min(np.abs(evals))
    
    thetas = np.linspace(0, 2*np.pi, 200)
    dets = []
    for th in thetas:
        H = build_H(seq, th)
        dets.append(np.linalg.det(H))
        
    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * np.pi)
    
    print(f"Seq: {seq[:10]}... | Gap: {gap:.4f} | W: {W:.2f}")

get_metrics("A" * 30)
get_metrics(("REWKYD" * 5)[:30])
get_metrics(("GP" * 15)[:30])
