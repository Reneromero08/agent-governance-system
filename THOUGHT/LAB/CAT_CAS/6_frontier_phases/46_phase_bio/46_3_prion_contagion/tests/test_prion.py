import numpy as np

# Macroscopic properties
# Poly-A: mean KD = 1.8, Bulk = 88, internal_frust = 0.0
# Prion (GP): mean KD = -1.0, Bulk = 91, internal_frust = 0.62

def build_lattice(is_infected, gamma=1.0, theta=0.0):
    L = 15
    H = np.zeros((L, L), dtype=np.complex128)
    
    # Define site properties
    kd = np.full(L, 1.8)
    bulk = np.full(L, 88.0)
    frust = np.full(L, 0.0)
    
    if is_infected:
        prion_site = 7
        kd[prion_site] = -1.0
        bulk[prion_site] = 91.0
        frust[prion_site] = 0.62
        
    for i in range(L):
        # Aqueous bath on macroscopic protein
        H[i, i] = -1j * gamma * kd[i]
        
        next_i = (i + 1) % L
        
        # Inter-protein hopping
        # If adjacent to a prion, the prion's internal frustration infects the bond
        bond_frust = max(frust[i], frust[next_i])
        
        # If infected, does it act as a sink?
        # A sink means hopping TOWARDS the prion is larger.
        # If i is prion, it pulls from next_i (backward hopping is large)
        # If next_i is prion, it pulls from i (forward hopping is large)
        
        # Simple non-reciprocal model:
        # We can just define global topological non-reciprocity induced by the prion
        # Or local. Let's just use the bond frustration.
        t_fwd = 2.0 * (1.0 + 2.0 * bond_frust)
        t_bwd = 2.0 * (1.0 - 2.0 * bond_frust)
        
        # Wait, if we want a global topological shift, maybe the prion forces the 
        # ENTIRE lattice to become non-reciprocal?
        # Let's test just local first.
        phi = (bulk[i] + bulk[next_i]) / 500.0 * np.pi
        
        if i == L - 1:
            H[i, next_i] = t_fwd * np.exp(1j * (phi + theta))
            H[next_i, i] = t_bwd * np.exp(-1j * (phi + theta))
        else:
            H[i, next_i] = t_fwd * np.exp(1j * phi)
            H[next_i, i] = t_bwd * np.exp(-1j * phi)
            
    return H

def analyze(is_infected):
    H = build_H = build_lattice(is_infected, gamma=1.0, theta=0.0)
    evals, evecs = np.linalg.eig(H)
    
    # Calculate IPR (Inverse Participation Ratio)
    # IPR = sum(|psi|^4) / (sum(|psi|^2))^2
    iprs = []
    for i in range(15):
        psi = evecs[:, i]
        ipr = np.sum(np.abs(psi)**4) / (np.sum(np.abs(psi)**2)**2)
        iprs.append(ipr)
    mean_ipr = np.mean(iprs)
    
    # Calculate Winding Number W
    thetas = np.linspace(0, 2*np.pi, 200)
    dets = []
    for th in thetas:
        H_th = build_lattice(is_infected, gamma=1.0, theta=th)
        dets.append(np.linalg.det(H_th))
    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * np.pi)
    
    # Spectral gap to origin
    gap = np.min(np.abs(evals))
    
    state = "INFECTED" if is_infected else "HEALTHY"
    print(f"[{state}] Gap: {gap:.4f} | W: {round(W)} | Mean IPR: {mean_ipr:.4f}")

analyze(False)
analyze(True)
