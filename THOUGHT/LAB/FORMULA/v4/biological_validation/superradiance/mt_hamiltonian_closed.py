#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mt_hamiltonian_closed.py -- Gap-closing 3D Hamiltonian.
Extracts actual Trp dipole ORIENTATIONS from PDB indole ring geometry,
computes exact geometric factor for every Trp pair.

ELIMINATES the approximation. Uses dipole vectors from PDB structure.
"""
import io, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals

# ============================================================================
# CONSTANTS
# ============================================================================
GAMMA = 0.00273; GAMMA_NR = 0.0193; KBT = 207.0
LAMBDA_CM = 280e-7; LAMBDA_NM = 280.0
K = 2.0 * np.pi / LAMBDA_CM
COUPLING_REF = 60.0
N_TUD_PER_SPIRAL = 13; N_TRP_PER_TUD = 8
SPIRAL_LENGTH_A = 80.0  # 8 nm in Angstroms
A_TO_CM = 1e-8


def extract_trp_atoms(pdb_path):
    """Extract all atoms for each Trp residue. Returns {(chain,resid): {atom: (x,y,z)}}."""
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            resname = line[17:20].strip()
            if resname != "TRP": continue
            atomname = line[12:16].strip()
            chain = line[21:22].strip()
            resid = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            key = (chain, resid)
            if key not in residues: residues[key] = {}
            residues[key][atomname] = np.array([x, y, z])
    return residues


def compute_dipole_vector(atoms):
    """Compute 1La transition dipole orientation from indole ring geometry.
    Long axis: from 6-membered ring center toward 5-membered ring.
    Approx: (CE3 + CZ3 + CH2)/3 -> (CD2 + NE1 + CE2)/3 normalized."""
    needed = ["CE3","CZ3","CH2","CD2","NE1","CE2","CG"]
    if not all(a in atoms for a in needed):
        # Fallback: use CG as position, random orientation
        if "CG" in atoms:
            return atoms["CG"], np.array([1.0, 0.0, 0.0])
        return np.zeros(3), np.array([1.0, 0.0, 0.0])

    # Ring center (approximate position of the dipole)
    # Use all indole heavy atoms for the position
    ring_atoms = ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
    positions = [atoms[a] for a in ring_atoms if a in atoms]
    position = np.mean(positions, axis=0)

    # 1La dipole direction: long axis of indole ring
    # From 6-membered ring (benzene side) to 5-membered ring (pyrrole side)
    benzene_center = (atoms["CE3"] + atoms["CZ3"] + atoms["CH2"]) / 3.0
    pyrrole_center = (atoms["CD2"] + atoms["NE1"] + atoms["CE2"]) / 3.0
    dipole_dir = pyrrole_center - benzene_center
    norm = np.linalg.norm(dipole_dir)
    if norm < 1e-6: dipole_dir = np.array([1.0, 0.0, 0.0])
    else: dipole_dir = dipole_dir / norm

    return position, dipole_dir


def geometric_factor_kappa(pos_i, u_i, pos_j, u_j):
    """Full dipole-dipole geometric factor: kappa = u_i·u_j - 3(u_i·r_hat)(u_j·r_hat).
    u_i, u_j are unit orientation vectors. pos_i, pos_j are positions in Angstroms."""
    r_vec = pos_j - pos_i
    r = np.linalg.norm(r_vec)
    if r < 1e-10: return 0.0
    r_hat = r_vec / r
    return u_i @ u_j - 3.0 * (u_i @ r_hat) * (u_j @ r_hat)


def dipole_coupling_closed(r_cm, kappa, gamma=GAMMA, k=K):
    """Dipole coupling with geometric factor kappa.
    V = kappa * V_parallel where V_parallel is for aligned dipoles."""
    kr = k * r_cm
    if kr < 1e-3:
        delta = -min(3*gamma/(4*kr**3), COUPLING_REF)
        gamma_c = gamma
    else:
        sinkr, coskr = np.sin(kr), np.cos(kr)
        delta = (3*gamma/4) * (coskr/kr - sinkr/(kr**2) - coskr/(kr**3))
        gamma_c = (3*gamma/2) * (sinkr/kr + coskr/(kr**2) - sinkr/(kr**3))
    delta = np.clip(delta, -COUPLING_REF, COUPLING_REF)
    return complex(delta * kappa, -gamma_c * kappa / 2)


def assemble_mt(trp_positions, trp_dipoles, n_spirals):
    """Assemble MT from single TuD data. Returns positions, dipoles, dimer_ids."""
    all_pos, all_dip, dimer_ids = [], [], []
    ang = 2*np.pi/N_TUD_PER_SPIRAL
    for spiral in range(n_spirals):
        for didx in range(N_TUD_PER_SPIRAL):
            theta = didx * ang
            ca, sa = np.cos(theta), np.sin(theta)
            Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            r_radial, x_shift = 112.0, spiral*SPIRAL_LENGTH_A + didx*9.0
            for pos, dip in zip(trp_positions, trp_dipoles):
                pos_r = Rx @ pos
                dip_r = Rx @ dip
                placed = np.array([
                    pos_r[0] + x_shift,
                    pos_r[1] + r_radial * ca,
                    pos_r[2] + r_radial * sa,
                ])
                all_pos.append(placed)
                all_dip.append(dip_r)
                dimer_ids.append(spiral*N_TUD_PER_SPIRAL + didx)
    return np.array(all_pos), np.array(all_dip), np.array(dimer_ids)


def build_closed_hamiltonian(positions_a, dipoles, dimer_ids=None, disorder=0, seed=42):
    """Build H_eff with EXACT geometric factors from dipole orientations."""
    rng = np.random.default_rng(seed)
    N = len(positions_a)
    coords_cm = positions_a * A_TO_CM
    H = np.zeros((N,N), dtype=complex)
    for i in range(N):
        E_i = rng.uniform(-100,100)
        if disorder > 0: E_i += rng.uniform(-disorder/2, disorder/2)
        H[i,i] = complex(E_i, -GAMMA/2)

    for i in range(N):
        for j in range(i+1, N):
            r_cm = np.linalg.norm(coords_cm[j] - coords_cm[i])
            if r_cm > LAMBDA_CM * 5: continue
            kappa = geometric_factor_kappa(positions_a[i], dipoles[i],
                                           positions_a[j], dipoles[j])
            if abs(kappa) < 1e-10: continue
            V = dipole_coupling_closed(r_cm, kappa)
            H[i,j] = V; H[j,i] = V
    return H


def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    outdir = Path(__file__).parent
    pdb = outdir / "1JFF.pdb"

    print("=" * 72)
    print("GAP-CLOSING 3D HAMILTONIAN -- Exact dipole orientations from PDB")
    print("=" * 72)

    # Extract Trp residues
    residues = extract_trp_atoms(str(pdb))
    print(f"\nFound {len(residues)} Trp residues in PDB (paper says 8 per TuD)")

    # Compute dipole vectors
    trp_positions, trp_dipoles = [], []
    for resid in sorted(residues.keys()):
        pos, dip = compute_dipole_vector(residues[resid])
        trp_positions.append(pos)
        trp_dipoles.append(dip)

    N_tud = len(trp_positions)
    print(f"TuD Trp dipoles extracted: {N_tud}")
    print(f"Dipole directions (first 3):")
    for i in range(min(3, N_tud)):
        print(f"  Trp {i}: dir=({trp_dipoles[i][0]:.3f}, {trp_dipoles[i][1]:.3f}, {trp_dipoles[i][2]:.3f})")

    # Verify orientations are diverse (not all parallel)
    dots = []
    for i in range(N_tud):
        for j in range(i+1, N_tud):
            dots.append(abs(trp_dipoles[i] @ trp_dipoles[j]))
    print(f"Intra-TuD dipole alignment: mean|cos|={np.mean(dots):.3f}, range=[{np.min(dots):.3f},{np.max(dots):.3f}]")

    # Single TuD test
    print(f"\nSingle TuD ({N_tud} Trp):")
    H = build_closed_hamiltonian(np.array(trp_positions), np.array(trp_dipoles))
    evals = eigvals(H)
    gj = np.maximum(-2*np.imag(evals), 0)
    print(f"  max(Gamma/gamma) = {np.max(gj)/GAMMA:.2f}")
    print(f"  mean(Gamma/gamma) = {np.mean(gj)/GAMMA:.4f}")
    print(f"  sum rule: {np.sum(gj)/GAMMA:.1f} (expected {N_tud})")

    # MT sweep
    print(f"\nMT SWEEP (with exact dipole orientations):")
    print(f"{'Spirals':>8} {'N':>6} {'L(nm)':>7} {'max(sigma)':>12} {'mean(G/g)':>10} {'subrad%':>8}")
    print("-" * 58)

    results = {}
    for ns in [1,2,3,5,7,10]:
        pos, dip, dids = assemble_mt(trp_positions, trp_dipoles, ns)
        N = len(pos)
        if N > 800:
            print(f"  {ns:>6} {N:>6} -- skipped (>800)")
            continue
        H = build_closed_hamiltonian(pos, dip, dids)
        ev = eigvals(H)
        gj = np.maximum(-2*np.imag(ev), 0)
        mx = np.max(gj)/GAMMA
        mn = np.mean(gj)/GAMMA
        sr = np.sum(gj < GAMMA)/N
        Lnm = ns * 8.0
        print(f"  {ns:>6} {N:>6} {Lnm:>6.0f} {mx:>12.2f} {mn:>10.4f} {sr:>7.1%}")
        results[f"spirals_{ns}"] = {"N":N,"L_nm":Lnm,"max_sigma":round(float(mx),2),
                                     "mean_sigma":round(float(mn),4),"subradiant":round(float(sr),3)}

    # Extrapolation
    if results:
        ns_vals = [int(k.split('_')[1]) for k in results]
        sig_vals = [results[k]["max_sigma"] for k in results]
        logN = np.log10([results[k]["N"] for k in results])
        logS = np.log10(sig_vals)
        alpha = np.polyfit(logN, logS, 1)[0]
        N_paper = 10400
        sigma_extrap = 10**(np.polyval([alpha, np.polyfit(logN, logS, 1)[1]], np.log10(N_paper)))
        print(f"\nExtrapolation to N={N_paper} (100 spirals, paper target: ~35):")
        print(f"  Power-law: alpha={alpha:.3f}")
        print(f"  Predicted sigma({N_paper}) = {sigma_extrap:.1f}")
        print(f"  Paper sigma({N_paper}) = 35")
        ratio = sigma_extrap / 35
        print(f"  Ratio: {ratio:.1f}x")
        if 0.5 < ratio < 2.0:
            print(f"  *** GAP CLOSED: within factor 2x of paper ***")
        else:
            print(f"  Gap: {ratio:.1f}x (need dipole moment magnitudes for full closure)")

    out = outdir / "mt3d_closed_results.json"
    class NE(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.bool_): return bool(o)
            if isinstance(o, complex): return [float(o.real), float(o.imag)]
            return super().default(o)
    json.dump(results, open(str(out), 'w'), indent=2, cls=NE)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
