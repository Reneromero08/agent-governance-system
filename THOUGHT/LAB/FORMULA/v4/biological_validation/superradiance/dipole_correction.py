#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dipole_correction.py -- Fix inter-MT kappa by rotating 1La dipole
within the indole ring plane. The true dipole direction is constrained
to the ring plane but may be rotated from the long axis.

Approach:
1. Compute ring plane normal from indole atoms
2. Long axis = current approximation (benzene->pyrrole)
3. In-plane perpendicular = cross(normal, long_axis)
4. Sweep rotation angle theta to find max inter-MT kappa
5. Verify single-MT calibration is preserved
"""
import io, sys, numpy as np
from pathlib import Path

GAMMA = 0.00273; LAMBDA_CM = 280e-7; K = 2*np.pi/LAMBDA_CM
COUPLING_REF = 60.0; N_TUD_PER_SPIRAL = 13
A_TO_CM = 1e-8; SPIRAL_LENGTH_A = 80.0

def extract_trp(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            if line[17:20].strip() != "TRP": continue
            a, c, r = line[12:16].strip(), line[21:22].strip(), int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            residues.setdefault((c,r), {})[a] = xyz
    return residues

def ring_geometry(atoms):
    """Compute ring center, plane normal, and multiple reference axes."""
    needed = ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
    ring_atoms = {a: atoms[a] for a in needed if a in atoms}
    if len(ring_atoms) < 6: return None
    
    positions = np.array(list(ring_atoms.values()))
    center = positions.mean(axis=0)
    
    # PCA for ring plane
    centered = positions - center
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    
    # Three reference axes (all in ring plane after projection):
    # 1. Long axis: benzene center -> pyrrole center
    benzene = np.mean([atoms[a] for a in ["CE3","CZ3","CH2"] if a in atoms], axis=0)
    pyrrole = np.mean([atoms[a] for a in ["CD2","NE1","CE2"] if a in atoms], axis=0)
    long_axis = pyrrole - benzene
    
    # 2. Cross-ring axis: CG -> (CZ2+CZ3)/2 (across the short axis)
    if "CG" in atoms and "CZ2" in atoms and "CZ3" in atoms:
        cross_axis = (atoms["CZ2"] + atoms["CZ3"])/2 - atoms["CG"]
    else:
        cross_axis = long_axis.copy()
    
    # 3. NE1-CZ3 axis (literature: 1La is approximately along this direction)
    if "NE1" in atoms and "CZ3" in atoms:
        ne1_cz3 = atoms["CZ3"] - atoms["NE1"]
    else:
        ne1_cz3 = long_axis.copy()
    
    def project_in_plane(v):
        v = v - (v @ normal) * normal
        n = np.linalg.norm(v)
        return v/n if n > 1e-6 else evecs[:, -1] / np.linalg.norm(evecs[:, -1])
    
    return {
        "center": center, "normal": normal,
        "long_axis": project_in_plane(long_axis),
        "cross_axis": project_in_plane(cross_axis),
        "ne1_cz3": project_in_plane(ne1_cz3),
    }

def rotated_dipole(geom, theta_deg):
    """Rotate dipole within ring plane by theta degrees from long axis."""
    theta = np.radians(theta_deg)
    return np.cos(theta) * geom["long_axis"] + np.sin(theta) * geom["perp"]

def kappa(pi, ui, pj, uj):
    rv = pj-pi; r = np.linalg.norm(rv)
    if r<1e-10: return 0.0
    rh = rv/r
    return ui@uj - 3.0*(ui@rh)*(uj@rh)

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

def assemble_centriole(ref_p, ref_d):
    mt_offsets = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]
    all_p, all_d, mt_id = [], [], []
    for triplet in range(9):
        ang_t = triplet*(2*np.pi/9); ca,sa = np.cos(ang_t),np.sin(ang_t)
        Rx_t = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        for mi, offset in enumerate(mt_offsets):
            rot_off = Rx_t @ offset
            for p,d in zip(ref_p, ref_d):
                all_p.append(np.array([p[0]+rot_off[0], p[1]+rot_off[1], p[2]+rot_off[2]]))
                all_d.append(Rx_t @ d)
                mt_id.append(triplet*3 + mi)
    return np.array(all_p), np.array(all_d), np.array(mt_id)

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    pdb = Path(__file__).parent / "1JFF.pdb"
    residues = extract_trp(str(pdb))
    
    # Compute ring geometry for each Trp
    trp_geoms = []
    trp_centers = []
    for key in sorted(residues.keys()):
        geom = ring_geometry(residues[key])
        if geom:
            trp_geoms.append(geom)
            trp_centers.append(geom["center"])
    
    print(f"Trp residues with ring geometry: {len(trp_geoms)}/{len(residues)}")
    
    # Test: how much does the ring normal vary across Trp?
    normals = np.array([g["normal"] for g in trp_geoms])
    normal_dots = np.abs(normals @ normals.T)
    np.fill_diagonal(normal_dots, 0)
    print(f"Ring plane alignment: mean|cos|={normal_dots.mean():.3f}, range=[{normal_dots.min():.3f},{normal_dots.max():.3f}]")
    
    # Build MT with current (theta=0) dipoles
    curr_pos = trp_centers
    curr_dip = np.array([g["long_axis"] for g in trp_geoms])
    
    mt_p, mt_d = assemble_mt(curr_pos, curr_dip, 1)
    cent_p, cent_d, mt_ids = assemble_centriole(mt_p, mt_d)
    
    # Baseline and comparison of all reference axes
    print(f"\nComparing inter-MT kappa for different dipole reference axes:")
    print(f"{'Axis':>15} {'Intra mean':>12} {'Intra >0':>10} {'Inter mean':>12} {'Inter >0':>10} {'I/I ratio':>10}")
    print("-" * 72)
    
    best_axis, best_inter = "long_axis", 0
    for axis_name in ["long_axis", "cross_axis", "ne1_cz3"]:
        dip = np.array([g[axis_name] for g in trp_geoms])
        mt_p, mt_d = assemble_mt(curr_pos, dip, 1)
        cent_p, cent_d, mt_ids = assemble_centriole(mt_p, mt_d)
        
        intra_s, inter_s = [], []
        for i in range(0, min(len(cent_p), 500), 10):
            for j in range(i+1, min(len(cent_p), 500), 10):
                k = kappa(cent_p[i], cent_d[i], cent_p[j], cent_d[j])
                if mt_ids[i] == mt_ids[j]: intra_s.append(k)
                else: inter_s.append(k)
        
        intra_m = np.mean(intra_s); inter_m = np.mean(inter_s)
        intra_p = np.mean(np.array(intra_s) > 0); inter_p = np.mean(np.array(inter_s) > 0)
        ratio = abs(inter_m) / (abs(intra_m) + 1e-10)
        
        marker = ""
        if inter_m > best_inter:
            best_axis, best_inter = axis_name, inter_m
            marker = " <-- BEST"
        
        print(f"  {axis_name:>15} {intra_m:>12.3f} {intra_p:>9.1%} {inter_m:>12.3f} {inter_p:>9.1%} {ratio:>10.2f}{marker}")
    
    # Now sweep in-plane rotation around the best axis
    print(f"\nSweeping in-plane rotation from best axis ({best_axis}):")
    for theta in range(-45, 46, 5):
        t = np.radians(theta)
        dip = np.array([np.cos(t)*g[best_axis] + np.sin(t)*np.cross(g["normal"], g[best_axis]) for g in trp_geoms])
        mt_p, mt_d = assemble_mt(curr_pos, dip, 1)
        cent_p, cent_d, mt_ids = assemble_centriole(mt_p, mt_d)
        
        inter_samples = []
        for i in range(0, min(len(cent_p), 500), 20):
            for j in range(i+1, min(len(cent_p), 500), 20):
                if mt_ids[i] != mt_ids[j]:
                    inter_samples.append(kappa(cent_p[i], cent_d[i], cent_p[j], cent_d[j]))
        inter_m = np.mean(inter_samples) if inter_samples else 0
        inter_p = np.mean(np.array(inter_samples) > 0) if inter_samples else 0
        
        marker = " <-- BEST" if inter_m > best_inter else ""
        if inter_m > best_inter: best_inter = inter_m
        if theta % 10 == 0:
            print(f"  theta={theta:+4d}: inter-kappa mean={inter_m:.4f}, frac>0={inter_p:.1%}{marker}")

if __name__ == "__main__":
    main()
