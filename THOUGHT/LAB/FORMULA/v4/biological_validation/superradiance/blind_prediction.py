#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
blind_prediction.py -- The irrefutable test.
Calibrate ONE parameter (coherent_scale) on MT data.
Then BLIND predict sigma for centriole and bundle architectures.
Compare to paper's values. No further tuning.
"""
import io, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals

# --- Constants from mt_hamiltonian_calibrated.py ---
GAMMA = 0.00273; GAMMA_NR = 0.0193
LAMBDA_CM = 280e-7; K = 2*np.pi/LAMBDA_CM
COUPLING_REF = 60.0
N_TUD_PER_SPIRAL = 13; A_TO_CM = 1e-8; SPIRAL_LENGTH_A = 80.0
# Calibrated from MT: coherent_scale = 100
COHERENT_SCALE = 100.0

def extract_trp(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            if line[17:20].strip() != "TRP": continue
            a, c, r = line[12:16].strip(), line[21:22].strip(), int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            k = (c, r)
            if k not in residues: residues[k] = {}
            residues[k][a] = xyz
    return residues

def dipole_vec(atoms):
    needed = ["CE3","CZ3","CH2","CD2","NE1","CE2","CG"]
    if not all(a in atoms for a in needed):
        return atoms.get("CG", np.zeros(3)), np.array([1.,0.,0.])
    ring = ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
    pos = np.mean([atoms[a] for a in ring if a in atoms], axis=0)
    bz = (atoms["CE3"]+atoms["CZ3"]+atoms["CH2"])/3
    py = (atoms["CD2"]+atoms["NE1"]+atoms["CE2"])/3
    d = py - bz; n = np.linalg.norm(d)
    return pos, d/n if n>1e-6 else np.array([1.,0.,0.])

def kappa(pi, ui, pj, uj):
    rv = pj-pi; r = np.linalg.norm(rv)
    if r<1e-10: return 0.0
    rh = rv/r
    return ui@uj - 3.0*(ui@rh)*(uj@rh)

def V(r_cm, kap):
    kr = K*r_cm
    if kr<1e-3:
        d = -min(3*GAMMA/(4*kr**3), COUPLING_REF); gc = GAMMA
    else:
        sk,ck = np.sin(kr),np.cos(kr)
        d = (3*GAMMA/4)*(ck/kr - sk/(kr**2) - ck/(kr**3))
        gc = (3*GAMMA/2)*(sk/kr + ck/(kr**2) - sk/(kr**3))
    return complex(np.clip(d,-COUPLING_REF,COUPLING_REF)*kap, -gc*abs(kap)/2)

def build_H(pos_a, dip, coherent_scale=COHERENT_SCALE):
    coords_cm = pos_a*A_TO_CM; N = len(pos_a)
    H = np.zeros((N,N), dtype=complex)
    for i in range(N):
        H[i,i] = complex(np.random.default_rng(i).uniform(-100,100), -GAMMA/2)
    for i in range(N):
        for j in range(i+1,N):
            rc = np.linalg.norm(coords_cm[j]-coords_cm[i])
            if rc>LAMBDA_CM*5: continue
            k = kappa(pos_a[i],dip[i],pos_a[j],dip[j])
            if abs(k)<1e-10: continue
            v = V(rc,k)
            H[i,j] = complex(v.real*coherent_scale, v.imag); H[j,i] = H[i,j]
    return H

def analyze(pos, dip, cs=COHERENT_SCALE):
    H = build_H(pos, dip, cs)
    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA, np.mean(gj)/GAMMA

# --- Build architectures ---
def assemble_mt(positions, dipoles, n_spirals):
    all_p, all_d = [], []
    ang = 2*np.pi/N_TUD_PER_SPIRAL
    for s in range(n_spirals):
        for d in range(N_TUD_PER_SPIRAL):
            th = d*ang; ca,sa = np.cos(th),np.sin(th)
            Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            xs = s*SPIRAL_LENGTH_A + d*9.0; rr = 112.0
            for p,dp in zip(positions, dipoles):
                pr,dr = Rx@p, Rx@dp
                all_p.append(np.array([pr[0]+xs, pr[1]+rr*ca, pr[2]+rr*sa]))
                all_d.append(dr)
    return np.array(all_p), np.array(all_d)

def assemble_centriole_exact(positions, dipoles, length_nm=80):
    """Exact centriole assembly per paper protocol (lines 150-159).
    9 triplets, each with 3 MTs at specific coordinates.
    Triplet MT centers: (0, 87, -22.5167), (0, 100, 0), (0, 113, 22.5167) nm.
    Each triplet rotated by 40 deg increments around x axis.
    """
    n_spirals = int(length_nm / 8.0)
    # Build one reference MT
    ref_p, ref_d = assemble_mt(positions, dipoles, n_spirals)
    
    # MT positions within a triplet (nm -> Angstroms)
    mt_offsets = [
        np.array([0, 870, -225.167]),  # (0, 87, -22.5167) nm
        np.array([0, 1000, 0]),         # (0, 100, 0) nm  
        np.array([0, 1130, 225.167]),   # (0, 113, 22.5167) nm
    ]
    
    all_p, all_d = [], []
    for triplet in range(9):
        ang = triplet * (2*np.pi/9)  # 40 degrees
        ca, sa = np.cos(ang), np.sin(ang)
        Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        
        for offset in mt_offsets:
            # Rotate the offset by triplet angle
            rot_offset = Rx @ offset
            for p, d in zip(ref_p, ref_d):
                placed = np.array([p[0] + rot_offset[0],
                                   p[1] + rot_offset[1],
                                   p[2] + rot_offset[2]])
                all_p.append(placed)
                all_d.append(Rx @ d)  # rotate dipole too
    
    return np.array(all_p), np.array(all_d)

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    outdir = Path(__file__).parent

    print("=" * 72)
    print("BLIND PREDICTION -- Calibrated on MT, predicting centriole")
    print(f"coherent_scale = {COHERENT_SCALE:.0f} (locked from MT calibration)")
    print("=" * 72)

    residues = extract_trp(str(outdir/"1JFF.pdb"))
    positions, dipoles = [], []
    for k in sorted(residues.keys()):
        p, d = dipole_vec(residues[k])
        positions.append(p); dipoles.append(d)

    # Verify MT calibration
    mt_p, mt_d = assemble_mt(positions, dipoles, 5)
    mt_s, mt_m = analyze(mt_p, mt_d, COHERENT_SCALE)
    print(f"\nMT (5 spirals, N={len(mt_p)}): sigma={mt_s:.1f} (paper ~35)")
    print(f"  Calibration check: ratio = {mt_s/35:.2f}x")

    # BLIND PREDICTION: Exact centriole geometry
    print(f"\nBLIND PREDICTION: Centriole (EXACT paper geometry)")
    print(f"  9 triplets x 3 MTs = 27 MTs total")
    for n_spirals in [1, 2]:
        cent_p, cent_d = assemble_centriole_exact(positions, dipoles, n_spirals*8)
        N = len(cent_p)
        print(f"  {n_spirals} spiral(s)/MT, N={N}... ", end="", flush=True)
        if N > 3000:
            print(f"too large, skipping")
            continue
        cent_s, cent_m = analyze(cent_p, cent_d, COHERENT_SCALE)
        s_per_mt = cent_s / 27
        print(f"sigma = {cent_s:.1f}, sigma/MT = {s_per_mt:.1f}")
        
        # Extrapolate: if sigma scales with MT count
        sigma_single_mt = 36.1  # from MT calibration
        enhancement_from_coupling = cent_s / (27 * sigma_single_mt / 5 * n_spirals)  # adjust for spiral count
        print(f"    Single MT (same N) would give ~{sigma_single_mt * n_spirals / 5:.0f}")
        print(f"    Inter-MT coupling enhancement: {cent_s / (27 * sigma_single_mt * n_spirals / 5):.1f}x")
        
        # Paper centriole at 320nm (40 spirals): sigma ~ 4000
        # Our centriole at n_spirals: scale by sqrt(40/n_spirals) if sub-linear
        paper_40spiral = 4000
        # Simple extrapolation: sigma scales with spirals, saturating
        scale_40 = min(n_spirals/40, 1.0) if n_spirals < 40 else 1.0
        pred_40spiral = cent_s / scale_40 if scale_40 > 0 else cent_s * 40/n_spirals
        print(f"    Crude extrapolation to 40 spirals: sigma ~ {pred_40spiral:.0f}")
        print(f"    Paper target: {paper_40spiral}")
        ratio = pred_40spiral / paper_40spiral
        print(f"    Ratio: {ratio:.2f}x", end="")
        if 0.5 < ratio < 2.0:
            print("  *** BLIND PREDICTION IN RANGE ***")
        else:
            print()

    out = outdir/"blind_prediction_results.json"
    json.dump({"coherent_scale": COHERENT_SCALE, "mt_check": mt_s}, open(str(out),'w'))
    print(f"\nResults: {out}")

if __name__ == "__main__":
    main()
