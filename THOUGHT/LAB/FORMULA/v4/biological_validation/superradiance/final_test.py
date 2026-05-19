"""Final test: centriole with direct 1JFF dipoles (paper's exact method)."""
import json, numpy as np
from scipy.linalg import eigvals

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
# Ring center position
trp_pos = np.array([np.mean([atoms[a] for a in ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"] if a in atoms], axis=0) for atoms in residues])
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

def analyze(pa, da):
    cc=pa*A_TO_CM; N=len(pa); H=np.zeros((N,N),dtype=complex)
    for i in range(N): H[i,i]=complex(0,-GAMMA/2)
    for i in range(N):
        for j in range(i+1,N):
            rv=cc[j]-cc[i]; r=np.linalg.norm(rv)
            if r>LAMBDA_CM*5: continue
            v=V(r,da[i],da[j],rv/r); H[i,j]=v; H[j,i]=v
    return np.max(np.maximum(-2*np.imag(eigvals(H)),0))/GAMMA

# Single MT
for ns in [1,3,5]:
    ap,ad = assemble_mt(trp_pos, trp_dip, ns)
    s = analyze(ap,ad); n = len(ap)
    print(f"MT {ns}sp (N={n}): sigma={s:.1f}, per-chr={s/n:.4f}")

# Centriole
mt_off = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]
print("\nCentriole (direct 1JFF dipoles, 1 spiral/MT):")
for n_trip in [1,3,6,9]:
    all_p, all_d = [], []
    for t in range(n_trip):
        ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
        Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
        for off in mt_off:
            ro = Rx_t @ off
            mt_p, mt_d = assemble_mt(trp_pos, trp_dip, 1)
            for p,d in zip(mt_p, mt_d):
                all_p.append(np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]]))
                all_d.append(Rx_t @ d)
    all_p, all_d = np.array(all_p), np.array(all_d)
    Nc = len(all_p)
    if Nc <= 3000:
        sc = analyze(all_p, all_d)
        n_mt = n_trip * 3
        print(f"  {n_trip} triplets ({n_mt} MTs, N={Nc}): sigma={sc:.1f}, per-chr={sc/Nc:.4f}, per-MT={sc/n_mt:.1f}")

# Compare to paper
print(f"\nPaper (from Table S3):")
print(f"  Single MT (320nm, 40sp): per-chr=0.120, total sigma~0.120*4160=499")
print(f"  Centriole (400nm, 50sp): per-chr=0.028, total sigma~0.028*140400=3931")
print(f"  Drop factor: 0.120/0.028 = {0.120/0.028:.1f}x")
