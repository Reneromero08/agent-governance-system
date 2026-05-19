"""Iterative eigensolver for large centriole (N=140k). Finds max eigenvalue without storing full matrix."""
import numpy as np
from scipy.sparse.linalg import eigs

GAMMA=0.00273; LAMBDA_CM=280e-7; K_WAVE=2*np.pi/LAMBDA_CM; A_TO_CM=1e-8; N_TUD=13

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

def V(r_cm, ui, uj, rh):
    kr=K_WAVE*r_cm; k1=ui@uj-(ui@rh)*(uj@rh); k3=ui@uj-3.0*(ui@rh)*(uj@rh)
    if kr<1e-3: O=np.clip(3*GAMMA/4*k3/kr**3,-200,200); U=3*GAMMA/2*k1
    else: sk,ck=np.sin(kr),np.cos(kr); O=-(3*GAMMA/4)*k1*ck/kr+(3*GAMMA/4)*k3*(sk/kr**2+ck/kr**3); U=(3*GAMMA/2)*k1*sk/kr+(3*GAMMA/2)*k3*(ck/kr**2-sk/kr**3)
    return complex(O,-U/2)

# Build centriole positions and dipoles for 1 spiral
print("Building centriole (1sp, 27 MTs)...")
mt_off = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]
all_pos_cm, all_dip = [], []
for t in range(9):
    ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
    for off in mt_off:
        ro = Rx_t @ off
        mp,md = assemble_mt(trp_pos_ring, trp_dip, 1)
        for p,d in zip(mp,md):
            all_pos_cm.append((np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]])) * A_TO_CM)
            all_dip.append(Rx_t @ d)
all_pos_cm = np.array(all_pos_cm, dtype=np.float64)
all_dip = np.array(all_dip, dtype=np.float64)
N = len(all_pos_cm)
print(f"N = {N}")

# Use sparse matrix with physics cutoff instead of dense matvec
# Coupling decays as 1/r^3. Beyond ~10nm (100A), coupling is <0.1% of nearest-neighbor
CUTOFF_CM = 100e-8  # 10 nm

print(f"Building sparse matrix (cutoff={CUTOFF_CM/A_TO_CM:.0f}A)...")
from scipy.sparse import lil_matrix
H_sparse = lil_matrix((N,N), dtype=complex)
for i in range(N):
    H_sparse[i,i] = complex(0, -GAMMA/2)

pair_count = 0
for i in range(N):
    for j in range(i+1, N):
        r_vec = all_pos_cm[j] - all_pos_cm[i]
        r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
        if r > CUTOFF_CM: continue
        r_hat = r_vec / r
        Vij = V(r, all_dip[i], all_dip[j], r_hat)
        H_sparse[i,j] = Vij
        H_sparse[j,i] = Vij
        pair_count += 1
print(f"Non-zero pairs: {pair_count} (sparsity: {1-pair_count/(N*(N-1)/2):.1%})")

print("Diagonalizing sparse matrix with ARPACK...")
import time
from scipy.sparse.linalg import eigs as sparse_eigs
H_csr = H_sparse.tocsr()
t0 = time.time()
eigvals = sparse_eigs(H_csr, k=6, which='LM', maxiter=300, tol=1e-6, return_eigenvectors=False)
t1 = time.time()
print(f"ARPACK done in {t1-t0:.0f}s")
