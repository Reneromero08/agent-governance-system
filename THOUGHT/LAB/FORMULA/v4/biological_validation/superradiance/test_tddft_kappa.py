"""Test inter-MT kappa with TD-DFT dipole vectors vs ring-geometry approximation."""
import json, numpy as np, sys

N_TUD = 13; SPIRAL_A = 80.0

with open("trp_dipoles_tddft.json") as f:
    tddft = json.load(f)
tddft_dips = np.array([np.array(d["dipole_vector"]) for d in tddft])
ring_centers = np.array([np.array(d["ring_center_A"]) for d in tddft])

# Compute ring-geometry dipoles
def ring_dipole(atoms_dict, centers):
    """Same as before but from saved centers and PDB atoms."""
    d = []
    for i in range(len(centers)):
        # Use the saved ring center and compute long axis from PDB
        # Since we don't have PDB atoms here, just use the TD-DFT ring centers
        # and compute an approximate long axis
        pass
    return None

# Actually, load PDB to compute ring dipoles properly
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

def compute_ring_dip(atoms):
    needed = ["CE3","CZ3","CH2","CD2","NE1","CE2","CG"]
    if not all(a in atoms for a in needed):
        return np.array([1.,0.,0.])
    bz = (atoms["CE3"]+atoms["CZ3"]+atoms["CH2"])/3
    py = (atoms["CD2"]+atoms["NE1"]+atoms["CE2"])/3
    d = py - bz
    return d / np.linalg.norm(d)

residues = extract_trp("1JFF.pdb")
ring_dips = np.array([compute_ring_dip(r) for r in residues])

# MT and centriole assembly
def assemble_mt(pos, dip, n_spirals):
    ap, ad = [], []
    ang = 2*np.pi/N_TUD
    for s in range(n_spirals):
        for d in range(N_TUD):
            th=d*ang; ca,sa=np.cos(th),np.sin(th)
            Rx=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            xs=s*SPIRAL_A+d*9.0; rr=112.0
            for p,dp in zip(pos,dip):
                pr,dr=Rx@p,Rx@dp
                ap.append(np.array([pr[0]+xs,pr[1]+rr*ca,pr[2]+rr*sa]))
                ad.append(dr)
    return np.array(ap), np.array(ad)

def assemble_centriole(ref_p, ref_d):
    offsets = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]
    ap, ad, mids = [], [], []
    for t in range(9):
        ang_t = t*(2*np.pi/9); ca,sa=np.cos(ang_t),np.sin(ang_t)
        Rx_t=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        for mi, off in enumerate(offsets):
            ro = Rx_t @ off
            for p,d in zip(ref_p, ref_d):
                ap.append(np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]]))
                ad.append(Rx_t @ d)
                mids.append(t*3+mi)
    return np.array(ap), np.array(ad), np.array(mids)

def kappa(pi,ui,pj,uj):
    rv=pj-pi; r=np.linalg.norm(rv)
    if r<1e-10: return 0.0
    rh=rv/r
    return ui@uj-3.0*(ui@rh)*(uj@rh)

# Test both
for label, dips in [("Ring-geometry", ring_dips), ("TD-DFT", tddft_dips)]:
    mt_p, mt_d = assemble_mt(ring_centers, dips, 1)
    cent_p, cent_d, mids = assemble_centriole(mt_p, mt_d)
    
    intra_k, inter_k = [], []
    step = max(1, len(cent_p)//500)
    for i in range(0, len(cent_p), step):
        for j in range(i+step, min(i+step*20, len(cent_p)), step):
            k = kappa(cent_p[i], cent_d[i], cent_p[j], cent_d[j])
            if mids[i] == mids[j]: intra_k.append(k)
            else: inter_k.append(k)
    
    intra_k = np.array(intra_k); inter_k = np.array(inter_k)
    intra_m = intra_k.mean(); inter_m = inter_k.mean()
    
    print(f"\n{label}:")
    print(f"  Intra-MT kappa: mean={intra_m:.4f}, frac>0={np.mean(intra_k>0):.1%}, std={intra_k.std():.3f}")
    print(f"  Inter-MT kappa: mean={inter_m:.4f}, frac>0={np.mean(inter_k>0):.1%}, std={inter_k.std():.3f}")
    print(f"  Inter/Intra ratio: {abs(inter_m)/max(abs(intra_m),1e-10):.3f}")
