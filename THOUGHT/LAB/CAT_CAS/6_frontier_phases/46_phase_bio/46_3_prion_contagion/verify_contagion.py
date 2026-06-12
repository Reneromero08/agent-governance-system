"""
Exp 46.3 HYPOTHESIS: Prion topological defect propagates via Skin Effect.
Roadmap: inject a W!=0 prion into a W=0 lattice. The Skin Effect exponentially
localizes eigenstates of neighbors, forcing the lattice to adopt the Prion's
winding number.

Current code: prion is detectable but does NOT propagate W to neighbors.
Test: can non-reciprocal coupling between proteins transfer the topological defect?
"""
import numpy as np, os

KD = {'A':1.8,'G':-0.4,'P':-1.6}

def build_protein_H(seq, gamma=0.3, t_base=0.1, frust_scale=1.0):
    """Same parameters as verified 46.1: W=0 for uniform, W!=0 for non-uniform."""
    L=len(seq); H=np.zeros((L,L),dtype=np.complex128)
    for i in range(L): H[i,i]=-1j*gamma*KD.get(seq[i],1.8)
    for i in range(L):
        j=(i+1)%L
        di=KD.get(seq[i],1.8); dj=KD.get(seq[j],1.8)
        delta=abs(di-dj); frust=delta*frust_scale
        H[j,i]=t_base+frust; H[i,j]=t_base
    return H

def compute_winding(H,n_phi=200):
    D=np.diag(np.diag(H)); O=H-D
    phis=np.linspace(0,2*np.pi,n_phi)
    dets=np.array([np.linalg.det(D+np.exp(1j*p)*O) for p in phis])
    return int(round((np.unwrap(np.angle(dets))[-1]-np.unwrap(np.angle(dets))[0])/(2*np.pi)))

def build_lattice(healthy_seq, prion_seq, N=10, L_seq=8, J=0.5, asym=0.8):
    """Lattice of N proteins. Center protein is the prion.
    J: coupling strength. asym: non-reciprocity (0=symmetric, 1=forward-only).
    Non-reciprocal coupling could enable Skin Effect propagation."""
    dim = N * L_seq
    H = np.zeros((dim,dim), dtype=np.complex128)

    for p in range(N):
        rs = p * L_seq
        seq = prion_seq if p == N//2 else healthy_seq
        H_p = build_protein_H(seq)
        H[rs:rs+L_seq, rs:rs+L_seq] = H_p

    for p in range(N-1):
        rs1, rs2 = p*L_seq, (p+1)*L_seq
        H[rs2, rs1+L_seq-1] = J * (1+asym)  # forward (non-reciprocal)
        H[rs1+L_seq-1, rs2] = J * (1-asym)  # backward (weaker)
        H[rs2+L_seq-1, rs1] = J * (1-asym)
        H[rs1, rs2+L_seq-1] = J * (1+asym)

    return H

def measure_infection(H, N, L_seq, prion_pos):
    """Measure winding of each protein's subspace to check propagation."""
    windings = []
    for p in range(N):
        rs = p * L_seq
        H_sub = H[rs:rs+L_seq, rs:rs+L_seq].copy()
        windings.append(compute_winding(H_sub))
    return windings

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.3: PRION CONTAGION VIA NON-RECIPROCAL COUPLING")
    log("=" * 70)

    N, L_seq = 10, 10
    healthy = "A" * L_seq
    prion = "GP" * (L_seq//2)
    prion_pos = N // 2

    log("Healthy: %s (W=%d)" % (healthy, compute_winding(build_protein_H(healthy))))
    log("Prion:   %s (W=%d)" % (prion, compute_winding(build_protein_H(prion))))
    log("")

    for asym in [0.0, 0.5, 0.8, 0.95]:
        for J in [0.1, 0.5, 1.0, 2.0]:
            H = build_lattice(healthy, prion, N, L_seq, J, asym)
            Ws = measure_infection(H, N, L_seq, prion_pos)
            infected = sum(1 for i, w in enumerate(Ws) if w != 0 and i != prion_pos)
            log("J=%.1f asym=%.2f: W=%s infected=%d/%d" % (J, asym, str(Ws), infected, N-1))

    log("")
    log("--- VERDICT ---")
    log("If non-reciprocal coupling (asym>0) causes neighboring proteins to adopt")
    log("W!=0, the Skin Effect propagation is genuine. If infection stays 0 regardless,")
    log("the prion defect does NOT propagate in this model.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_46_3_CONTAGION.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log("\nSaved: " + path)

if __name__ == "__main__":
    run()
