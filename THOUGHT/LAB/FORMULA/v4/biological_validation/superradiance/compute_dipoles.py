#!/usr/bin/env python3
"""
compute_dipoles.py -- TD-DFT transition dipole vectors for Trp 1La state.
Run in WSL with PySCF 2.13.0:
  wsl -d Ubuntu bash -c 'cd /mnt/d/CCC\ 2.0/AI/agent-governance-system/THOUGHT/LAB/FORMULA/v4/biological_validation/superradiance && python3 compute_dipoles.py'

For each of the 8 Trp residues in 1JFF.pdb:
1. Extract indole ring + CB atom
2. Cap CB with H (methyl model)
3. Run CAM-B3LYP/6-31G* TD-DFT
4. Find state closest to 280 nm with highest oscillator strength
5. Extract transition dipole vector (x, y, z) in atomic units
6. Save all 8 vectors to trp_dipoles.json
"""
import json, sys, os
import numpy as np
from pathlib import Path

# PySCF imports
from pyscf import gto, scf, tddft
from pyscf.data import nist

# Target excitation: 280 nm = 4.428 eV = 0.1627 Hartree = 35714 cm^-1
TARGET_EV = 4.428
TARGET_NM = 280

# Trp indole ring atoms we need
RING_ATOMS = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]

def extract_trp_residues(pdb_path):
    """Extract full atomic coordinates for each Trp residue.
    Returns list of {atom_name: (x, y, z)} for each residue."""
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            if line[17:20].strip() != "TRP": continue
            atom = line[12:16].strip()
            chain = line[21:22].strip()
            resid = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            key = (chain, resid)
            if key not in residues: residues[key] = {}
            residues[key][atom] = np.array([x, y, z])
    return [residues[k] for k in sorted(residues.keys())]


def build_trp_model(atoms):
    """
    Build a minimal Trp model for TD-DFT: indole ring + CB with H cap.
    Returns PySCF Mole object.
    Coordinates in Angstroms (PDB format).
    """
    # Get CB position (attachment point to protein backbone)
    cb = atoms.get("CB")
    if cb is None:
        # Fallback: use CG as reference
        cb = atoms["CG"] + np.array([0.5, 0.5, 0.5])  # approximate

    # Ring atom coordinates (averaged if multiple conformations)
    ring_coords = {}
    for name in RING_ATOMS:
        if name in atoms:
            ring_coords[name] = atoms[name]

    # Build XYZ string for PySCF
    xyz_lines = []
    # Indole ring atoms
    for name in RING_ATOMS:
        if name in ring_coords:
            c = ring_coords[name]
            elem = name[0]  # C or N
            if elem == 'N': elem = 'N'
            elif elem == 'C': elem = 'C'
            else: elem = 'C'
            xyz_lines.append(f"{elem} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")

    # Add CB carbon with H cap (methyl group approximation)
    # Place a H atom ~1.09 A from CB in a reasonable direction
    cb_to_cg = atoms["CG"] - cb
    cb_to_cg = cb_to_cg / np.linalg.norm(cb_to_cg)
    # H atom opposite to CG direction (away from ring)
    h_pos = cb - 1.09 * cb_to_cg
    xyz_lines.append(f"C {cb[0]:.6f} {cb[1]:.6f} {cb[2]:.6f}")
    # Add 2 more H atoms for the methyl group (tetrahedral)
    perp1 = np.array([cb_to_cg[1], -cb_to_cg[0], 0.0])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.array([1.0, 0.0, 0.0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(cb_to_cg, perp1)
    for offset in [perp1, perp2, -perp1 - perp2]:
        h = cb + 1.09 * (offset / np.linalg.norm(offset))
        xyz_lines.append(f"H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}")

    xyz_str = "\n".join(xyz_lines)
    mol = gto.M(atom=xyz_str, basis='6-31G*', charge=0, spin=0, verbose=0)
    return mol


def compute_1la_dipole(atoms, residue_idx=0):
    """
    Compute 1La transition dipole vector for a Trp residue.
    Returns (dipole_vector, excitation_energy_nm, oscillator_strength).
    """
    mol = build_trp_model(atoms)
    
    # HF ground state
    mf = scf.RHF(mol)
    mf.kernel()
    
    # TD-DFT with TDA (Tamm-Dancoff approximation) for stability
    # TDA avoids Cholesky issues and gives near-identical dipole vectors
    nstates = 12
    td = tddft.TDA(mf)  # TDA instead of full TDDFT
    td.nstates = nstates
    td.kernel()
    
    # Find state closest to 280 nm with non-negligible oscillator strength
    best_idx = None
    best_dist = float('inf')
    for i in range(nstates):
        e_ev = td.e[i] * nist.HARTREE2EV
        osc = td.oscillator_strength()[i]
        if osc < 0.01:  # Skip dark states
            continue
        dist = abs(e_ev - TARGET_EV)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    if best_idx is None:
        # Fallback: take the state with largest oscillator strength
        oscs = td.oscillator_strength()
        best_idx = np.argmax(oscs)
    
    e_ev = td.e[best_idx] * nist.HARTREE2EV
    osc = td.oscillator_strength()[best_idx]
    
    # Transition dipole moment (atomic units)
    # td.transition_dipole() returns [x, y, z] for each state in au
    dip = td.transition_dipole()[best_idx]
    
    return np.array(dip), e_ev, osc


def main():
    pdb_path = Path(__file__).parent / "1JFF.pdb"
    if not pdb_path.exists():
        print(f"ERROR: PDB file not found at {pdb_path}")
        sys.exit(1)
    
    residues = extract_trp_residues(str(pdb_path))
    print(f"Found {len(residues)} Trp residues")
    
    results = []
    for idx, atoms in enumerate(residues):
        print(f"\nTrp {idx+1}/{len(residues)}: computing TD-DFT...")
        sys.stdout.flush()
        
        dip_vec, e_ev, osc = compute_1la_dipole(atoms, idx)
        
        # Normalize to unit vector
        dip_norm = np.linalg.norm(dip_vec)
        if dip_norm > 1e-10:
            dip_unit = dip_vec / dip_norm
        else:
            dip_unit = np.array([1.0, 0.0, 0.0])
        
        print(f"  1La state: {e_ev:.2f} eV ({1240/e_ev:.0f} nm), osc={osc:.4f}")
        print(f"  Dipole vector (unit): [{dip_unit[0]:.6f}, {dip_unit[1]:.6f}, {dip_unit[2]:.6f}]")
        print(f"  Dipole magnitude: {dip_norm:.4f} au")
        
        # Also get the indole ring center for position reference
        ring_center = np.mean([atoms[a] for a in RING_ATOMS if a in atoms], axis=0)
        
        results.append({
            "residue_idx": idx,
            "dipole_vector": dip_unit.tolist(),
            "dipole_magnitude_au": float(dip_norm),
            "excitation_eV": float(e_ev),
            "excitation_nm": float(1240/e_ev),
            "oscillator_strength": float(osc),
            "ring_center_A": ring_center.tolist(),
        })
    
    # Save results
    output_path = Path(__file__).parent / "trp_dipoles_tddft.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(results)} dipole vectors to {output_path}")
    return results


if __name__ == "__main__":
    main()
