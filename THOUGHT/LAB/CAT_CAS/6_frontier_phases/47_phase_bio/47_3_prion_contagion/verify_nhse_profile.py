"""
Exp 46.3: NHSE EIGENVECTOR LOCALIZATION PROFILE.
Diagonalize full lattice, find prion eigenmode, measure spatial decay.
P_i = block-averaged probability of prion mode on protein i.
Exponential decay = contagion gradient via Skin Effect.
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

def build_lattice(healthy_seq, prion_seq, N, L_seq, J, asym):
    dim = N * L_seq; prion_pos = N // 2
    H = np.zeros((dim, dim), dtype=np.complex128)
    for p in range(N):
        rs = p * L_seq
        seq = prion_seq if p == prion_pos else healthy_seq
        H[rs:rs+L_seq, rs:rs+L_seq] = build_protein_H(seq)
    for p in range(N-1):
        rs1, rs2 = p * L_seq, (p+1) * L_seq
        H[rs2, rs1+L_seq-1] = J * (1 + asym)
        H[rs1+L_seq-1, rs2] = J * (1 - asym)
    return H

def prion_mode_localization(H_full, N, L_seq, prion_pos, prion_EP_eigenvalue):
    """Find eigenmode closest to prion's isolated EP, compute P_i per protein."""
    evals, evecs = np.linalg.eig(H_full)
    # Find eigenstate closest to prion's isolated EP
    dists = np.abs(evals - prion_EP_eigenvalue)
    mode_idx = np.argmin(dists)
    mode = evecs[:, mode_idx]
    prob = np.abs(mode)**2
    prob /= prob.sum()

    P_i = np.zeros(N)
    for p in range(N):
        rs = p * L_seq
        P_i[p] = np.sum(prob[rs:rs+L_seq])
    return P_i, evals[mode_idx]

def fit_exponential(distances, probs):
    """Log-linear fit: log(P) = a - distance/xi. Returns xi (localization length)."""
    valid = probs > 1e-10
    if np.sum(valid) < 3:
        return float('inf'), 0.0
    d = distances[valid]
    p = probs[valid]
    # Linear fit to log(P) vs distance
    A = np.vstack([np.ones_like(d), -d]).T
    coeff, _, _, _ = np.linalg.lstsq(A, np.log(p), rcond=None)
    xi = 1.0 / coeff[1] if abs(coeff[1]) > 1e-10 else float('inf')
    # R^2
    pred = coeff[0] - d * coeff[1]
    ss_res = np.sum((np.log(p) - pred)**2)
    ss_tot = np.sum((np.log(p) - np.mean(np.log(p)))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return xi, r2

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("EXP 46.3: NHSE EIGENVECTOR LOCALIZATION PROFILE")
    log("=" * 70)

    N, L_seq = 30, 10
    healthy_seq = "A" * L_seq
    prion_seq = "GP" * (L_seq // 2)
    prion_pos = N // 2

    # Isolated prion EP eigenvalue (mean eigenvalue of prion Hamiltonian)
    H_prion_iso = build_protein_H(prion_seq)
    prion_EP = np.mean(np.linalg.eigvals(H_prion_iso))

    # Healthy baseline: all healthy proteins
    H_healthy = build_lattice(healthy_seq, healthy_seq, N, L_seq, 0.5, 0.95)
    W_healthy = compute_winding(H_healthy)
    evals_h, evecs_h = np.linalg.eig(H_healthy)
    # Pick a representative eigenmode and check its uniformity
    mid_mode_h = evecs_h[:, len(evals_h)//2]
    prob_h = np.abs(mid_mode_h)**2; prob_h /= prob_h.sum()
    P_h = np.array([np.sum(prob_h[p*L_seq:(p+1)*L_seq]) for p in range(N)])
    log("\nHEALTHY BASELINE (no prion):")
    log("  Global W = %+d" % W_healthy)
    log("  Representative mode P_i: mean=%.4f std=%.4f (uniform=%.4f)" % (
        np.mean(P_h), np.std(P_h), 1.0/N))

    # Infected lattice with prion
    for J in [0.1, 0.3, 0.5, 1.0]:
        for asym in [0.8, 0.95]:
            H = build_lattice(healthy_seq, prion_seq, N, L_seq, J, asym)
            W = compute_winding(H)
            P_i, mode_eval = prion_mode_localization(H, N, L_seq, prion_pos, prion_EP)

            # Exponential fit for proteins NOT at prion
            distances = np.array([abs(p - prion_pos) for p in range(N) if p != prion_pos])
            probs = np.array([P_i[p] for p in range(N) if p != prion_pos])
            xi, r2 = fit_exponential(distances, probs)

            # Check: is prion peak much higher than average?
            prion_prob = P_i[prion_pos]
            avg_near = np.mean([P_i[p] for p in range(N) if 0 < abs(p-prion_pos) <= 3])
            avg_far = np.mean([P_i[p] for p in range(N) if abs(p-prion_pos) >= 8])
            peak_ratio = prion_prob / avg_far if avg_far > 0 else float('inf')

            log("\nJ=%.1f asym=%.2f W=%+d:" % (J, asym, W))
            log("  Prion peak P=%.4f  near(<=3)=%.4f  far(>=8)=%.4f  peak/far=%.1fx" % (
                prion_prob, avg_near, avg_far, peak_ratio))
            log("  Localization length xi=%.1f proteins  R^2=%.3f" % (xi, r2))
            log("  Mode eigenvalue: %.4f%+.4fj" % (mode_eval.real, mode_eval.imag))

            # Show P_i profile
            profile = " ".join("%.3f" % P_i[p] for p in range(N))
            log("  P_i: %s" % profile)

    log("\n--- VERDICT ---")
    log("If P_i is exponentially peaked at the prion (xi < N/2, R^2 > 0.5),")
    log("the NHSE contagion gradient is mathematically confirmed.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_47_3_NHSE.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log("\nSaved: " + path)

if __name__ == "__main__":
    run()
