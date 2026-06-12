import numpy as np

def build_lattice(is_infected, gamma=0.2, theta=0.0):
    L = 15
    H = np.zeros((L, L), dtype=np.complex128)
    
    kd = np.full(L, 1.8)
    bulk = np.full(L, 88.0)
    frust = np.full(L, 0.0)
    
    if is_infected:
        prion_site = 7
        kd[prion_site] = -1.0
        bulk[prion_site] = 91.0
        frust[prion_site] = 0.62
        
    for i in range(L):
        H[i, i] = -1j * gamma * kd[i]
        next_i = (i + 1) % L
        
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

def analyze(is_infected):
    H = build_lattice(is_infected, gamma=0.2, theta=0.0)
    evals, evecs = np.linalg.eig(H)
    
    iprs = []
    for i in range(15):
        psi = evecs[:, i]
        ipr = np.sum(np.abs(psi)**4) / (np.sum(np.abs(psi)**2)**2)
        iprs.append(ipr)
    mean_ipr = np.mean(iprs)
    
    thetas = np.linspace(0, 2*np.pi, 200)
    dets = []
    for th in thetas:
        H_th = build_lattice(is_infected, gamma=0.2, theta=th)
        dets.append(np.linalg.det(H_th))
    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * np.pi)
    
    gap = np.min(np.abs(evals))
    
    state = "INFECTED" if is_infected else "HEALTHY"
    print(f"[{state}] Gap: {gap:.4f} | W: {round(W)} | Mean IPR: {mean_ipr:.4f}")

analyze(False)
analyze(True)
