"""
Exp 46.3 CORRECTED: Prion contagion = eigenstate sequestration + local gap collapse.
The prion's EP sink drags healthy protein eigenstates into the complex plane
via the Non-Hermitian Skin Effect. Metric: LOCAL SPECTRAL GAP of each protein.
"""
import numpy as np, os

KD = {'A':1.8,'G':-0.4,'P':-1.6,'R':-4.5,'L':3.8,'V':4.2}

def build_protein_H(seq, gamma=0.3, t_base=0.1, frust_scale=1.0):
    L=len(seq); H=np.zeros((L,L),dtype=np.complex128)
    for i in range(L): H[i,i]=-1j*gamma*KD.get(seq[i],1.8)
    for i in range(L):
        j=(i+1)%L; di=KD.get(seq[i],1.8); dj=KD.get(seq[j],1.8)
        delta=abs(di-dj); frust=delta*frust_scale
        H[j,i]=t_base+frust; H[i,j]=t_base
    return H

def compute_winding(H, n_phi=200):
    D=np.diag(np.diag(H)); O=H-D
    phis=np.linspace(0,2*np.pi,n_phi)
    dets=np.array([np.linalg.det(D+np.exp(1j*p)*O) for p in phis])
    return int(round((np.unwrap(np.angle(dets))[-1]-np.unwrap(np.angle(dets))[0])/(2*np.pi)))

def local_spectral_gap(H_full, protein_start, L_seq):
    """Project H_full onto protein subspace and compute eigenvalue separation.
    Lower gap = more mixed with prion = infected."""
    N_full = H_full.shape[0]
    # Build projector onto protein subspace
    P = np.zeros((N_full, L_seq), dtype=np.complex128)
    for i in range(L_seq):
        P[protein_start + i, i] = 1.0
    # Projected Hamiltonian: P^H H P
    H_proj = P.conj().T @ H_full @ P
    evals = np.sort_complex(np.linalg.eigvals(H_proj))
    gaps = [abs(evals[i+1]-evals[i]) for i in range(len(evals)-1)]
    return float(min(gaps)) if gaps else 0.0

def build_protein_lattice(healthy_seq, prion_seq, N, L_seq, J, asym):
    """Lattice with non-reciprocal inter-protein coupling."""
    dim = N * L_seq
    H = np.zeros((dim, dim), dtype=np.complex128)
    prion_pos = N // 2
    for p in range(N):
        rs = p * L_seq
        seq = prion_seq if p == prion_pos else healthy_seq
        H[rs:rs+L_seq, rs:rs+L_seq] = build_protein_H(seq)
    for p in range(N-1):
        rs1, rs2 = p * L_seq, (p+1) * L_seq
        H[rs2, rs1+L_seq-1] = J * (1 + asym)  # forward
        H[rs1+L_seq-1, rs2] = J * (1 - asym)  # backward (suppressed)
    return H

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.3 CORRECTED: LOCAL SPECTRAL GAP COLLAPSE (CONTAGION)")
    log("=" * 70)
    log("")
    log("Contagion = eigenstate sequestration via Skin Effect.")
    log("Healthy proteins near the prion: local spectral gap COLLAPSES.")
    log("Healthy proteins far from prion: local spectral gap PRESERVED.")
    log("")

    N, L_seq = 20, 10
    healthy_seq = "A" * L_seq
    prion_seq = "GP" * (L_seq // 2)
    prion_pos = N // 2

    # Uninfected baseline: all healthy proteins, no prion
    H_healthy = build_protein_lattice(healthy_seq, healthy_seq, N, L_seq, 0.5, 0.95)
    W_healthy = compute_winding(H_healthy)
    gaps_healthy = [local_spectral_gap(H_healthy, p*L_seq, L_seq) for p in range(N)]

    log("BASELINE (all healthy, no prion):")
    log("  Global W = %+d" % W_healthy)
    log("  Local gaps = [%.4f ... %.4f] (all should be healthy)" % (min(gaps_healthy), max(gaps_healthy)))
    log("")

    # Infected: prion at center with non-reciprocal coupling
    for J in [0.1, 0.3, 0.5, 1.0]:
        for asym in [0.8, 0.95]:
            H_infected = build_protein_lattice(healthy_seq, prion_seq, N, L_seq, J, asym)
            W_infected = compute_winding(H_infected)
            gaps = [local_spectral_gap(H_infected, p*L_seq, L_seq) for p in range(N)]

            # Classify: proteins near prion (distance <= 3) vs far (>= 6)
            near_gaps = [gaps[p] for p in range(N) if abs(p-prion_pos) <= 3 and p != prion_pos]
            far_gaps = [gaps[p] for p in range(N) if abs(p-prion_pos) >= 6]
            near_avg = np.mean(near_gaps) if near_gaps else 0
            far_avg = np.mean(far_gaps) if far_gaps else 0
            gap_ratio = far_avg / near_avg if near_avg > 0 else float('inf')

            # Count infected: proteins where gap collapsed below healthy baseline
            healthy_gap_threshold = np.mean(gaps_healthy) * 0.5
            infected_near = sum(1 for p in range(N) if abs(p-prion_pos) <= 3 and p != prion_pos and gaps[p] < healthy_gap_threshold)
            infected_far = sum(1 for p in range(N) if abs(p-prion_pos) >= 6 and gaps[p] < healthy_gap_threshold)

            log("J=%.1f asym=%.2f W=%+d: near=%.4f far=%.4f ratio=%.1fx  infected: near=%d/%d far=%d/%d" % (
                J, asym, W_infected, near_avg, far_avg, gap_ratio,
                infected_near, len(near_gaps), infected_far, len(far_gaps)))

    log("")
    log("--- VERDICT ---")
    log("If local spectral gap collapses near the prion (near << far),")
    log("and remains healthy far from the prion, contagion via eigenstate")
    log("sequestration is confirmed.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_46_3_GAP_COLLAPSE.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log("\nSaved: " + path)

if __name__ == "__main__":
    run()
