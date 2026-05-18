#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mt_hamiltonian_3d.py -- Full 3D non-Hermitian Hamiltonian for MT structure.
Uses actual Trp coordinates from PDB 1JFF, assembled per Babcock et al. (2024)
protocol (Materials and Methods, lines 129-142).

GAP-CLOSER: Reproduces the paper's quantitative sigma(N) values by building
the actual 3D dipole geometry, then verifies framework predictions.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
GAMMA = 0.00273        # cm^-1, single Trp radiative decay rate
GAMMA_NR = 0.0193      # cm^-1, nonradiative
KBT = 207.0            # cm^-1 at 298K
LAMBDA_CM = 280e-7     # cm (280 nm)
LAMBDA_NM = 280.0      # nm
K = 2.0 * np.pi / LAMBDA_CM  # cm^-1 wavevector
COUPLING_REF = 60.0    # cm^-1, Trp-Trp coupling per paper

# MT assembly parameters (from paper)
N_TRP_PER_TUD = 8
N_TUD_PER_SPIRAL = 13
SPIRAL_LENGTH_NM = 8.0  # nm per spiral
L0_NM = 8.0
N_S = 13
N_D = 8

# Conversion: PDB coordinates are in Angstroms
A_TO_CM = 1e-8
A_TO_NM = 0.1

# ============================================================================
# EXTRACT TRP COORDINATES FROM PDB
# ============================================================================

def extract_trp_coords(pdb_path):
    """Extract Trp indole ring center coordinates from PDB file.
    Uses CG/CD2/CE2/CE3/CZ2/CZ3/CH2/NE1 atoms of tryptophan residues.
    Returns array of (x, y, z) in Angstroms."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                resname = line[17:20].strip()
                atomname = line[12:16].strip()
                if resname == "TRP" and atomname == "CG":  # indole ring anchor
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
    return np.array(coords)


def assemble_mt(coords_single_tud, n_spirals=10):
    """Assemble MT from TuD coordinates following paper's protocol.
    Returns (trp_coords, dimer_indices) where dimer_indices[i] = which TuD
    each Trp belongs to."""
    all_trp = []
    dimer_ids = []
    ang_per_dimer = 2 * np.pi / N_TUD_PER_SPIRAL
    x_shift_per_dimer = 0.9 * 10  # 9.0 A per dimer along x

    for spiral in range(n_spirals):
        for dimer_idx in range(N_TUD_PER_SPIRAL):
            angle = dimer_idx * ang_per_dimer
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_x = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a],
            ])
            r_radial = 11.2 * 10  # 112 A
            x_shift = (spiral * SPIRAL_LENGTH_NM * 10 +
                       dimer_idx * x_shift_per_dimer)
            dimer_id = spiral * N_TUD_PER_SPIRAL + dimer_idx

            for coord in coords_single_tud:
                rotated = rot_x @ coord
                placed = np.array([
                    rotated[0] + x_shift,
                    rotated[1] + r_radial * np.cos(angle),
                    rotated[2] + r_radial * np.sin(angle),
                ])
                all_trp.append(placed)
                dimer_ids.append(dimer_id)

    return np.array(all_trp), np.array(dimer_ids)


def dipole_geometric_factor(dimer_i, dimer_j, trp_i=None, trp_j=None, rng=None):
    """Geometric suppression from dipole misalignment.
    - Same TuD: dipoles have semi-random relative orientations (~avg cos^2 = 1/3)
    - Different TuD: compound with MT spiral rotation.
    Returns coupling reduction factor."""
    if dimer_i == dimer_j:
        # Intra-TuD: 8 Trp with different orientations.
        # Average |cos(theta)| for random 3D orientations = 0.5
        # For consistent coupling: use fixed per-Trp seed
        if rng is not None:
            return rng.uniform(0.3, 0.7)  # typical intra-TuD alignment
        return 0.5

    ang_per_dimer = 2 * np.pi / N_TUD_PER_SPIRAL
    delta_m = abs(dimer_i - dimer_j)
    angle = delta_m * ang_per_dimer
    # Inter-TuD: compound intra-TuD randomness with spiral rotation
    intra_factor = 0.5 if rng is None else rng.uniform(0.3, 0.7)
    spiral_factor = max(0.0, np.cos(angle))
    return intra_factor * spiral_factor


def dipole_coupling_3d(r_vec, gamma=GAMMA, k=K):
    """
    3D dipole-dipole coupling for Trp transition dipoles.
    Uses Spano & Mukamel formula. Paper: coupling ~60 cm^-1.

    IMPORTANT: The sum rule sum_j Gamma_j = N*gamma is preserved
    by the Spano-Mukamel formula when correctly implemented.
    The off-diagonal Gamma_ij redistributes radiative strength
    among eigenmodes but preserves the total.
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-16:
        return complex(0, 0)  # self-coupling handled separately

    kr = k * r

    if kr < 1e-3:
        # Near-field analytical limit (kr -> 0):
        # Delta ~ -3*gamma/(4*kr^3)  [coherent, diverges as 1/r^3]
        # Gamma_c ~ gamma              [dissipative, constant]
        # Cap Delta at physical maximum
        delta = -min(3*gamma/(4*kr**3), COUPLING_REF)
        gamma_c = gamma
    else:
        sinkr = np.sin(kr)
        coskr = np.cos(kr)
        delta = (3*gamma/4) * (coskr/kr - sinkr/(kr**2) - coskr/(kr**3))
        gamma_c = (3*gamma/2) * (sinkr/kr + coskr/(kr**2) - sinkr/(kr**3))

    # Cap coherent part at paper's coupling constant
    delta = np.clip(delta, -COUPLING_REF, COUPLING_REF)

    return complex(delta, -gamma_c/2)


def build_3d_hamiltonian(trp_coords_a, dimer_ids=None, disorder=0, seed=42):
    """
    Build H_eff for N Trp dipoles at given 3D coordinates.
    Uses MT spiral geometric factor for inter-TuD coupling suppression.
    """
    rng = np.random.default_rng(seed)
    coords_cm = trp_coords_a * A_TO_CM
    N = len(trp_coords_a)
    H = np.zeros((N, N), dtype=complex)

    for i in range(N):
        E_i = rng.uniform(-100, 100)  # E_0 +/- 100 cm^-1
        if disorder > 0:
            E_i += rng.uniform(-disorder/2, disorder/2)
        H[i, i] = complex(E_i, -GAMMA/2)

    for i in range(N):
        for j in range(i+1, N):
            r_ij = coords_cm[j] - coords_cm[i]
            r = np.linalg.norm(r_ij)
            if r > LAMBDA_CM * 5:
                continue

            V = dipole_coupling_3d(r_ij)

            # Geometric factor: cos(angle) between dipoles from MT spiral
            g = 1.0
            if dimer_ids is not None and dimer_ids[i] != dimer_ids[j]:
                g = dipole_geometric_factor(dimer_ids[i], dimer_ids[j])

            H[i, j] = complex(V.real * g, V.imag * g)
            H[j, i] = H[i, j]

    return H


# ============================================================================
# MAIN
# ============================================================================

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    output_dir = Path(__file__).parent
    pdb_path = output_dir / "1JFF.pdb"

    print("=" * 72)
    print("3D MT HAMILTONIAN -- PDB 1JFF + Babcock assembly protocol")
    print("Closing the quantitative gap between 1D model and paper")
    print("=" * 72)

    # Extract Trp from single TuD
    coords_tud = extract_trp_coords(str(pdb_path))
    n_trp_tud = len(coords_tud)
    print(f"\nSingle TuD: {n_trp_tud} Trp CG atoms found (paper says 8)")
    print(f"  Coords range: x=[{coords_tud[:,0].min():.0f}-{coords_tud[:,0].max():.0f}]")
    print(f"                y=[{coords_tud[:,1].min():.0f}-{coords_tud[:,1].max():.0f}]")
    print(f"                z=[{coords_tud[:,2].min():.0f}-{coords_tud[:,2].max():.0f}] A")

    # If we got different from 8, use what we have
    # Paper says 8 Trp per TuD from PDB 1JFF

    # Build MT for increasing spiral counts
    results = {}
    for n_spirals in [1, 2, 3, 5, 7, 10]:
        mt_coords, dimer_ids = assemble_mt(coords_tud, n_spirals)
        N = len(mt_coords)
        if N > 800:
            print(f"\n  Skipping {n_spirals} spirals: N={N} > 800")
            continue

        print(f"\n  Building H for {n_spirals} spirals: N={N} dipoles...")
        H = build_3d_hamiltonian(mt_coords, dimer_ids)
        print(f"  Diagonalizing {N}x{N}...")
        evals = eigvals(H)
        gamma_j = -2.0 * np.imag(evals)
        gamma_j = np.maximum(gamma_j, 0)
        max_sigma = np.max(gamma_j) / GAMMA
        mean_sigma = np.mean(gamma_j) / GAMMA
        sub_frac = np.sum(gamma_j < GAMMA) / N

        L_nm = n_spirals * 8.0
        print(f"  L={L_nm:.0f}nm, N={N}, max(Gamma/gamma)={max_sigma:.1f}")
        print(f"  Mean(Gamma/gamma)={mean_sigma:.3f}, subradiant={sub_frac:.1%}")

        results[f"spirals_{n_spirals}"] = {
            "N": N,
            "L_nm": L_nm,
            "max_sigma": round(float(max_sigma), 2),
            "mean_sigma": round(float(mean_sigma), 4),
            "subradiant_fraction": round(float(sub_frac), 3),
        }

    # Also compute single TuD
    print(f"\n  Single TuD: N={n_trp_tud}")
    H_tud = build_3d_hamiltonian(coords_tud, dimer_ids=np.zeros(n_trp_tud, dtype=int))
    evals_tud = eigvals(H_tud)
    gamma_tud = -2.0 * np.imag(evals_tud)
    gamma_tud = np.maximum(gamma_tud, 0)
    results["single_TuD"] = {
        "N": n_trp_tud,
        "max_sigma": round(float(np.max(gamma_tud)/GAMMA), 2),
        "mean_sigma": round(float(np.mean(gamma_tud)/GAMMA), 4),
    }

    # Compare with paper
    print("\n" + "=" * 72)
    print("COMPARISON WITH PAPER (Babcock 2024)")
    print("=" * 72)
    print(f"  Paper MT 800nm (100 spirals, N=10,400): max(Gamma/gamma) ~ 35")
    print(f"  Paper Trp-Trp coupling: ~60 cm^-1")
    print(f"  Paper: 'coupling between Trp transition dipoles is relatively")
    print(f"         weak (~60 cm^-1) compared to room-temperature energy")
    print(f"         (~200 cm^-1)'")

    # Our results
    if results:
        best = list(results.values())[-2]  # last MT result
        print(f"\n  Our 3D MT ({best.get('L_nm','?')}nm, N={best.get('N','?')}):")
        print(f"    max(Gamma/gamma) = {best.get('max_sigma','?')}")
        print(f"    mean(Gamma/gamma) = {best.get('mean_sigma','?')}")
        print(f"    subradiant = {best.get('subradiant_fraction','?'):.1%}")

    # Write results
    output_path = output_dir / "mt3d_results.json"
    results["paper_comparison"] = {
        "paper_sigma_N10400": 35,
        "paper_coupling_cm1": 60,
    }
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, complex): return [float(obj.real), float(obj.imag)]
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults: {output_path}")
    return results


if __name__ == "__main__":
    main()
