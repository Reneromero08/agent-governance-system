"""GPU-accelerated superradiance Hamiltonian with PyTorch/CUDA."""
import json, numpy as np, time, sys
import torch

GAMMA=0.00273; LAMBDA_CM=280e-7; K_WAVE=2*np.pi/LAMBDA_CM; A_TO_CM=1e-8; N_TUD=13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def extract_trp(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            if line[17:20].strip() != "TRP": continue
            a, c, r = line[12:16].strip(), line[21:22].strip(), int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            residues.setdefault((c,r), {})[a] = xyz
    return [residues[k] for k in sorted(residues.keys())]

def compute_1la(atoms):
    needed = ["CD2","CE2","CD1","NE1","CG","CE3","CZ2","CZ3","CH2"]
    if not all(a in atoms for a in needed): return np.array([1.,0.,0.])
    mid = (atoms["CD2"] + atoms["CE2"]) / 2
    base = atoms["CD1"] - mid; base = base / np.linalg.norm(base)
    ring_atoms = ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
    ring_pos = np.array([atoms[a] for a in ring_atoms])
    _, _, vh = np.linalg.svd(ring_pos - ring_pos.mean(axis=0))
    normal = vh[2] / np.linalg.norm(vh[2])
    perp = np.cross(normal, base); perp = perp / np.linalg.norm(perp)
    theta = np.radians(46.2)
    c1 = np.cos(theta)*base + np.sin(theta)*perp
    c2 = np.cos(theta)*base - np.sin(theta)*perp
    to_ne1 = atoms["NE1"] - mid
    d = c1 if c1@to_ne1 > c2@to_ne1 else c2
    return d / np.linalg.norm(d)

residues = extract_trp("1JFF.pdb")
trp_pos_ring = np.array([np.mean([atoms[a] for a in ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"] if a in atoms], axis=0) for atoms in residues])
trp_dip = np.array([compute_1la(r) for r in residues])

def assemble_mt(pos, dip, n_spirals):
    ap, ad = [], []
    for s in range(n_spirals):
        for d in range(N_TUD):
            th = d*2*np.pi/N_TUD; ca,sa = np.cos(th),np.sin(th)
            Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            xs = s*80.0 + d*9.0
            for p,dp in zip(pos, dip):
                pr,dr = Rx@p, Rx@dp
                ap.append(np.array([pr[0]+xs, pr[1]+112.0*ca, pr[2]+112.0*sa]))
                ad.append(dr)
    return np.array(ap), np.array(ad)

def build_H_gpu_full(positions_a, dipoles_a):
    """Build full dense Hamiltonian on GPU with vectorized pairwise computation."""
    N = len(positions_a)
    pos_t = torch.tensor(positions_a, dtype=torch.float32, device=DEVICE)
    dip_t = torch.tensor(dipoles_a, dtype=torch.float32, device=DEVICE)
    
    # Compute all pairwise distance vectors: r_ij = pos_j - pos_i
    # pos_t: (N, 3). Expand to (N, N, 3)
    pi = pos_t.unsqueeze(1).expand(-1, N, -1)  # (N, N, 3) -- pos_i for all j
    pj = pos_t.unsqueeze(0).expand(N, -1, -1)  # (N, N, 3) -- pos_j for all i
    r_vecs = pj - pi  # (N, N, 3)
    r = torch.norm(r_vecs, dim=-1)  # (N, N)
    r_hat = r_vecs / (r.unsqueeze(-1) + 1e-16)
    
    # Geometric factors: ui·uj, ui·r_hat, uj·r_hat
    # dip_t: (N, 3). Expand
    ui = dip_t.unsqueeze(1).expand(-1, N, -1)  # (N, N, 3)
    uj = dip_t.unsqueeze(0).expand(N, -1, -1)  # (N, N, 3)
    
    uidot_uj = (ui * uj).sum(dim=-1)  # (N, N)
    uidot_rh = (ui * r_hat).sum(dim=-1)
    ujdot_rh = (uj * r_hat).sum(dim=-1)
    
    k1 = uidot_uj - uidot_rh * ujdot_rh
    k3 = uidot_uj - 3.0 * uidot_rh * ujdot_rh
    
    kr = K_WAVE * r
    
    # Spano-Mukamel (vectorized)
    coskr = torch.cos(kr); sinkr = torch.sin(kr)
    kr_safe = torch.clamp(kr, min=1e-10)
    
    Omega = -(3*GAMMA/4)*k1*coskr/kr_safe + (3*GAMMA/4)*k3*(sinkr/kr_safe**2 + coskr/kr_safe**3)
    Upsilon = (3*GAMMA/2)*k1*sinkr/kr_safe + (3*GAMMA/2)*k3*(coskr/kr_safe**2 - sinkr/kr_safe**3)
    
    # Near-field regularization
    small = kr < 1e-3
    Omega[small] = torch.clamp(3*GAMMA/4*k3[small]/(kr_safe[small]**3), -200, 200)
    Upsilon[small] = 3*GAMMA/2*k1[small]
    
    H = torch.complex(Omega, -Upsilon/2)
    # Diagonal: -i*gamma/2
    H.diagonal().copy_(torch.complex(torch.zeros(N, device=DEVICE), 
                                       torch.full((N,), -GAMMA/2, device=DEVICE)))
    # Zero out NaN/inf
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return H

# Test: single MT with full vectorized build
print("Full vectorized GPU Hamiltonian")
for ns in [1,3,5]:
    mp,md = assemble_mt(trp_pos_ring, trp_dip, ns)
    pos_cm = mp * A_TO_CM; N = len(mp)
    t0=time.time(); H=build_H_gpu_full(pos_cm, md); t1=time.time()
    t2=time.time(); ev=torch.linalg.eigvals(H); t3=time.time()
    gj=torch.clamp(-2*ev.imag,min=0); s=gj.max().item()/GAMMA
    print(f"MT {ns}sp (N={N}): sigma={s:.1f}, build={t1-t0:.1f}s, diag={t3-t2:.1f}s, mem={H.element_size()*H.numel()/1e6:.0f}MB")
