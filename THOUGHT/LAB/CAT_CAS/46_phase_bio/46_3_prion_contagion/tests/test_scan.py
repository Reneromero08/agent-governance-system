import numpy as np

def build_lattice(gamma=1.0, theta=0.0):
    L = 15
    H = np.zeros((L, L), dtype=np.complex128)
    
    kd = np.full(L, 1.8)
    bulk = np.full(L, 88.0)
    frust = np.full(L, 0.0)
    
    prion_site = 7
    kd[prion_site] = -1.0
    bulk[prion_site] = 91.0
    frust[prion_site] = 0.62
    
    for i in range(L):
        H[i, i] = -1j * gamma * kd[i]
        next_i = (i + 1) % L
        
        # Test: If Prion infects the whole lattice?
        # Or if it's just the adjacent bonds? Let's just use adjacent bonds but 
        # with strong non-reciprocity.
        
        # What if the Prion has a direct macroscopic non-reciprocal pumping?
        bond_frust = max(frust[i], frust[next_i])
        
        t_fwd = 2.0 * (1.0 + 2.0 * bond_frust)
        t_bwd = 2.0 * (1.0 - 2.0 * bond_frust)
        
        phi = (bulk[i] + bulk[next_i]) / 500.0 * np.pi
        
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
            
    return H

gammas = np.linspace(0.1, 5.0, 50)
for gamma in gammas:
    thetas = np.linspace(0, 2*np.pi, 200)
    dets = []
    for th in thetas:
        H_th = build_lattice(gamma=gamma, theta=th)
        dets.append(np.linalg.det(H_th))
    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * np.pi)
    if abs(W) > 0.1:
        print(f"Gamma: {gamma:.2f} | W: {round(W)}")
