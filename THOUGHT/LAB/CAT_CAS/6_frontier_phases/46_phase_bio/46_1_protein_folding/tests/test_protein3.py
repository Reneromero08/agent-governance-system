import numpy as np

# Kyte-Doolittle Hydrophobicity
KD = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4,
    'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

# Approximate Amino Acid Volume (Bulk)
BULK = {
    'G': 60, 'A': 88, 'S': 89, 'C': 108, 'T': 116, 'P': 122, 'D': 111, 'N': 114,
    'V': 140, 'E': 138, 'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168,
    'R': 173, 'F': 189, 'Y': 193, 'W': 227
}

def build_H(seq, theta=0.0):
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    
    for i in range(L):
        # On-site potential (scaled to bring closer to origin)
        H[i, i] = -1j * (KD[seq[i]])
        
        # Hopping
        next_i = (i + 1) % L
        
        # Steric clash creates non-reciprocity
        frustration = abs(BULK[seq[i]] - BULK[seq[next_i]]) / 100.0
        
        t_fwd = 1.0 + frustration
        t_bwd = 1.0 - frustration
        
        phi = (BULK[seq[i]] + BULK[seq[next_i]]) / 500.0 * np.pi
        
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
            
    return H

def test_seq(seq):
    thetas = np.linspace(0, 2*np.pi, 200)
    dets = []
    
    H0 = build_H(seq, 0.0)
    evals = np.linalg.eigvals(H0)
    gap = np.min(np.abs(evals))
    
    for th in thetas:
        H = build_H(seq, th)
        dets.append(np.linalg.det(H))
        
    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * np.pi)
    
    print(f"Seq: {seq[:10]}... | Gap: {gap:.4f} | W: {W:.2f}")

test_seq("A" * 30)
test_seq("RDEWYK" * 5)
test_seq("GP" * 15)
