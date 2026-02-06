#!/usr/bin/env python3
"""
Q18 Tests with REAL Biological Data

NO SYNTHETIC DATA. Every test uses externally sourced real data.

Data Sources (all downloaded via fetch_real_data.py):
1. AlphaFold - REAL protein structures with pLDDT confidence scores
2. UniProt - REAL protein sequences
3. Ensembl - REAL human-mouse ortholog mappings

Tests:
1. Protein Folding: Does R computed from sequence predict pLDDT?
2. 8e Conservation: Does Df x alpha = 8e hold for biological covariance matrices?

NOTE: These tests use REAL data. Results may differ from synthetic tests.
That's the point - we need to know if R works on REALITY, not simulations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import os

CACHE_DIR = Path(__file__).parent / 'cache'


# ============================================================================
# DATA LOADING (REAL DATA ONLY)
# ============================================================================

def load_alphafold_plddt() -> Dict[str, Dict]:
    """Load REAL AlphaFold pLDDT scores."""
    plddt_file = CACHE_DIR / 'alphafold_plddt.json'
    if not plddt_file.exists():
        raise FileNotFoundError(f"No AlphaFold data. Run: python fetch_real_data.py --source alphafold")

    with open(plddt_file) as f:
        return json.load(f)


def load_uniprot_sequences() -> Dict[str, Dict]:
    """Load REAL UniProt protein sequences."""
    seq_file = CACHE_DIR / 'uniprot_sequences.json'
    if not seq_file.exists():
        raise FileNotFoundError(f"No UniProt data. Run: python fetch_real_data.py --source uniprot")

    with open(seq_file) as f:
        return json.load(f)


def load_alphafold_pdb(uniprot_id: str) -> str:
    """Load REAL AlphaFold PDB file content."""
    pdb_file = CACHE_DIR / 'alphafold' / f'{uniprot_id}.pdb'
    if not pdb_file.exists():
        raise FileNotFoundError(f"No PDB for {uniprot_id}. Run: python fetch_real_data.py --source alphafold")

    with open(pdb_file) as f:
        return f.read()


def load_orthologs() -> List[Dict]:
    """Load REAL human-mouse ortholog mappings."""
    ortho_file = CACHE_DIR / 'human_mouse_orthologs.json'
    if not ortho_file.exists():
        raise FileNotFoundError(f"No ortholog data. Run: python fetch_real_data.py --source orthologs")

    with open(ortho_file) as f:
        return json.load(f)


# ============================================================================
# SEQUENCE FEATURES (no synthetic data - computed from real sequences)
# ============================================================================

# Amino acid properties from biochemistry literature
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Disorder propensity (amino acids common in disordered regions)
DISORDER_AA = set('DEKRSPQGN')

# Secondary structure propensity
HELIX_AA = set('AELM')
SHEET_AA = set('VIY')


def compute_sequence_features(sequence: str) -> Dict[str, float]:
    """
    Compute features from REAL protein sequence.
    No synthetic data - all features derived from actual amino acid composition.
    """
    seq = sequence.upper()
    n = len(seq)

    if n == 0:
        return {'length': 0, 'hydrophobicity': 0, 'disorder_frac': 0,
                'helix_prop': 0, 'sheet_prop': 0, 'complexity': 0}

    # Mean hydrophobicity (real biochemical property)
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    mean_hydro = np.mean(hydro_values)
    std_hydro = np.std(hydro_values) if len(hydro_values) > 1 else 0

    # Disorder fraction (based on disorder-promoting residues)
    disorder_count = sum(1 for aa in seq if aa in DISORDER_AA)
    disorder_frac = disorder_count / n

    # Secondary structure propensity
    helix_prop = sum(1 for aa in seq if aa in HELIX_AA) / n
    sheet_prop = sum(1 for aa in seq if aa in SHEET_AA) / n

    # Sequence complexity (Shannon entropy)
    aa_counts = {}
    for aa in seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    entropy = 0
    for count in aa_counts.values():
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)

    # Normalize to 0-1 (max entropy is log2(20) for 20 amino acids)
    complexity = entropy / np.log2(20)

    return {
        'length': n,
        'hydrophobicity_mean': mean_hydro,
        'hydrophobicity_std': std_hydro,
        'disorder_frac': disorder_frac,
        'helix_prop': helix_prop,
        'sheet_prop': sheet_prop,
        'complexity': complexity
    }


# ============================================================================
# R COMPUTATION (canonical formula, no tricks)
# ============================================================================

def compute_R_from_features(features: Dict[str, float]) -> float:
    """
    Compute R = E / sigma from sequence features.

    This is the CANONICAL R formula:
    - E = measure of agreement/quality (here: predicted foldability)
    - sigma = measure of dispersion/uncertainty (here: feature variance)

    NO CIRCULAR TRICKS:
    - E is NOT the same as the target (pLDDT)
    - sigma is internal sequence property, not outcome
    """
    # E: Foldability estimate based on sequence composition
    # Higher for: ordered sequences, balanced hydrophobicity, moderate complexity

    order_score = 1.0 - features['disorder_frac']  # Low disorder = better folding
    hydro_balance = 1.0 - abs(features['hydrophobicity_mean']) / 4.5  # Balanced hydro
    structure_prop = features['helix_prop'] + features['sheet_prop']  # Secondary structure
    complexity_penalty = abs(features['complexity'] - 0.75)  # Optimal complexity

    E = 0.4 * order_score + 0.3 * hydro_balance + 0.2 * structure_prop + 0.1 * (1 - complexity_penalty)

    # sigma: Sequence heterogeneity
    # Higher variance in properties = less certain prediction
    sigma = max(features['hydrophobicity_std'] / 4.5, 0.01)  # Normalized, min 0.01

    R = E / sigma
    return R


def compute_R_from_pdb_bfactors(pdb_content: str) -> Tuple[float, List[float]]:
    """
    Compute R from REAL pLDDT values (stored in B-factor column of AlphaFold PDBs).

    This measures the ACTUAL structure confidence from AlphaFold.
    R = mean(pLDDT) / std(pLDDT)

    High R = confident, consistent structure prediction
    Low R = uncertain or variable structure prediction
    """
    plddt_values = []

    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            try:
                # pLDDT is in B-factor column (columns 61-66, 1-indexed)
                plddt = float(line[60:66].strip())
                plddt_values.append(plddt)
            except (ValueError, IndexError):
                pass

    if len(plddt_values) < 2:
        return 0.0, []

    mean_plddt = np.mean(plddt_values)
    std_plddt = np.std(plddt_values)

    R = mean_plddt / max(std_plddt, 0.1)

    return R, plddt_values


# ============================================================================
# TEST 1: PROTEIN FOLDING PREDICTION
# ============================================================================

def test_protein_folding_real():
    """
    Test if R computed from sequence predicts AlphaFold structure quality.

    METHODOLOGY:
    1. Load REAL UniProt sequences
    2. Load REAL AlphaFold pLDDT scores
    3. Compute R_sequence from sequence features (NO pLDDT info)
    4. Compare R_sequence with mean pLDDT (correlation)

    SUCCESS CRITERION: Pearson r > 0.3 (sequence R predicts structure quality)

    WHY THIS IS NOT CIRCULAR:
    - R_sequence is computed from amino acid composition ONLY
    - pLDDT is an INDEPENDENT measure from AlphaFold structure prediction
    - If R_sequence predicts pLDDT, it means sequence features capture foldability
    """
    print("\n" + "=" * 70)
    print("TEST 1: PROTEIN FOLDING PREDICTION (REAL DATA)")
    print("=" * 70)

    # Load real data
    sequences = load_uniprot_sequences()
    plddt_data = load_alphafold_plddt()

    print(f"Loaded {len(sequences)} sequences and {len(plddt_data)} pLDDT records")

    # Compute R for each protein
    results = []
    for uniprot_id in sequences:
        if uniprot_id not in plddt_data:
            print(f"  Skipping {uniprot_id}: no pLDDT data")
            continue

        # Compute R from sequence (independent of pLDDT)
        seq_data = sequences[uniprot_id]
        features = compute_sequence_features(seq_data['sequence'])
        R_sequence = compute_R_from_features(features)

        # Get REAL pLDDT
        mean_plddt = plddt_data[uniprot_id]['mean_plddt']

        results.append({
            'uniprot_id': uniprot_id,
            'R_sequence': R_sequence,
            'mean_plddt': mean_plddt,
            'length': seq_data['length'],
            'disorder_frac': features['disorder_frac']
        })

        print(f"  {uniprot_id}: R_seq={R_sequence:.3f}, pLDDT={mean_plddt:.1f}, "
              f"disorder={features['disorder_frac']:.2f}")

    if len(results) < 3:
        print("\nNot enough data for correlation test")
        return {'status': 'INSUFFICIENT_DATA', 'n_proteins': len(results)}

    # Correlation test
    R_values = [r['R_sequence'] for r in results]
    plddt_values = [r['mean_plddt'] for r in results]

    # Pearson correlation
    r_corr = np.corrcoef(R_values, plddt_values)[0, 1]

    print(f"\n--- RESULTS ---")
    print(f"N proteins: {len(results)}")
    print(f"Pearson r (R_sequence vs pLDDT): {r_corr:.4f}")

    # Interpretation
    if r_corr > 0.3:
        verdict = "PASS: R from sequence predicts structure quality"
    elif r_corr > 0:
        verdict = "WEAK: Positive but weak relationship"
    else:
        verdict = "FAIL: No relationship between R and pLDDT"

    print(f"Verdict: {verdict}")

    # IMPORTANT: With only 5 proteins, this is a pilot test
    print(f"\nNOTE: This is a pilot with only {len(results)} proteins.")
    print("For robust conclusions, need 50+ diverse proteins.")

    return {
        'status': 'PASS' if r_corr > 0.3 else 'FAIL',
        'pearson_r': float(r_corr),
        'n_proteins': len(results),
        'results': results,
        'verdict': verdict,
        'caveat': 'Small sample size - pilot test only'
    }


# ============================================================================
# TEST 2: 8e CONSERVATION (Df x alpha from biological data)
# ============================================================================

def compute_eigenspectrum_from_plddt(pdb_file_path: str) -> Tuple[np.ndarray, float, float]:
    """
    Compute eigenvalue spectrum from REAL pLDDT distribution in a protein.

    This tests whether the 8e conservation law holds for ACTUAL protein structures.
    """
    pdb_content = open(pdb_file_path).read()

    # Get pLDDT per residue (use CA atoms only for one value per residue)
    residue_plddt = {}
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            try:
                res_num = int(line[22:26].strip())
                plddt = float(line[60:66].strip())
                residue_plddt[res_num] = plddt
            except (ValueError, IndexError):
                pass

    if len(residue_plddt) < 10:
        return np.array([]), 0, 0

    # Convert to array
    plddt_array = np.array(list(residue_plddt.values()))
    n = len(plddt_array)

    # Compute covariance-like matrix from sliding windows
    # This captures local pLDDT patterns
    window = 5
    n_windows = n - window + 1

    if n_windows < window:
        return np.array([]), 0, 0

    # Feature matrix: each row is a window of pLDDT values
    features = np.zeros((n_windows, window))
    for i in range(n_windows):
        features[i] = plddt_array[i:i+window]

    # Center features
    features = features - features.mean(axis=0)

    # Covariance matrix
    cov = features.T @ features / n_windows

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) < 2:
        return eigenvalues, 0, 0

    # Df: Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda2 = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / sum_lambda2 if sum_lambda2 > 0 else 0

    # alpha: Spectral decay (power law exponent)
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    # Linear regression for alpha
    slope, _ = np.polyfit(log_k, log_lambda, 1)
    alpha = -slope

    return eigenvalues, Df, alpha


def test_8e_conservation_real():
    """
    Test if Df x alpha = 8e (21.746) holds for REAL biological structures.

    METHODOLOGY:
    1. Load REAL AlphaFold PDB files
    2. Compute eigenvalue spectrum from pLDDT distribution
    3. Calculate Df (participation ratio) and alpha (spectral decay)
    4. Test if Df x alpha = 8e (within reasonable tolerance)

    WHY THIS IS HONEST:
    - Uses REAL pLDDT values from AlphaFold
    - No parameter tuning to hit 8e
    - Reports actual values, whatever they are
    """
    print("\n" + "=" * 70)
    print("TEST 2: 8e CONSERVATION (REAL BIOLOGICAL STRUCTURES)")
    print("=" * 70)

    alphafold_dir = CACHE_DIR / 'alphafold'
    pdb_files = list(alphafold_dir.glob('*.pdb'))

    print(f"Found {len(pdb_files)} PDB files")

    target_8e = 21.746  # Theoretical 8e value

    results = []
    for pdb_file in pdb_files:
        uniprot_id = pdb_file.stem
        eigenvalues, Df, alpha = compute_eigenspectrum_from_plddt(str(pdb_file))

        if len(eigenvalues) < 2:
            print(f"  {uniprot_id}: Insufficient data")
            continue

        Df_x_alpha = Df * alpha

        results.append({
            'uniprot_id': uniprot_id,
            'Df': Df,
            'alpha': alpha,
            'Df_x_alpha': Df_x_alpha,
            'n_eigenvalues': len(eigenvalues)
        })

        print(f"  {uniprot_id}: Df={Df:.3f}, alpha={alpha:.3f}, Df*alpha={Df_x_alpha:.3f}")

    if len(results) < 2:
        print("\nInsufficient data for 8e test")
        return {'status': 'INSUFFICIENT_DATA', 'n_proteins': len(results)}

    # Aggregate results
    Df_x_alpha_values = [r['Df_x_alpha'] for r in results]
    mean_value = np.mean(Df_x_alpha_values)
    std_value = np.std(Df_x_alpha_values)
    cv = std_value / mean_value if mean_value > 0 else float('inf')

    deviation_from_8e = abs(mean_value - target_8e) / target_8e * 100

    print(f"\n--- RESULTS ---")
    print(f"N proteins: {len(results)}")
    print(f"Mean Df x alpha: {mean_value:.3f}")
    print(f"Std: {std_value:.3f}")
    print(f"CV: {cv:.3f}")
    print(f"Theoretical 8e: {target_8e:.3f}")
    print(f"Deviation from 8e: {deviation_from_8e:.1f}%")

    # Interpretation
    if deviation_from_8e < 15 and cv < 0.3:
        verdict = "PASS: 8e conservation holds in biological structures!"
    elif deviation_from_8e < 30:
        verdict = "PARTIAL: Some 8e signal, but significant deviation"
    else:
        verdict = "FAIL: No evidence for 8e at molecular scale"

    print(f"Verdict: {verdict}")

    # IMPORTANT caveat
    print(f"\nNOTE: Using pLDDT sliding windows as proxy for structural covariance.")
    print("This may not capture the full spectral structure.")
    print("Need more diverse proteins and proper embedding for definitive test.")

    return {
        'status': 'PASS' if deviation_from_8e < 15 else 'FAIL',
        'mean_Df_x_alpha': float(mean_value),
        'std': float(std_value),
        'cv': float(cv),
        'target_8e': target_8e,
        'deviation_percent': float(deviation_from_8e),
        'n_proteins': len(results),
        'results': results,
        'verdict': verdict,
        'caveat': 'Small sample, pLDDT-based proxy for structure'
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Q18 TESTS WITH REAL BIOLOGICAL DATA")
    print("=" * 70)
    print("\nNO SYNTHETIC DATA. All tests use externally sourced real data.")
    print("Data sources: AlphaFold, UniProt, Ensembl")
    print("\nThis is an HONEST test. Results may differ from synthetic tests.")
    print("That's the point - we need to know if R works on REALITY.\n")

    # Check data availability
    if not CACHE_DIR.exists():
        print("ERROR: No cache directory. Run fetch_real_data.py first.")
        return

    results = {}

    # Run tests
    try:
        results['protein_folding'] = test_protein_folding_real()
    except FileNotFoundError as e:
        print(f"Skipping protein folding test: {e}")
        results['protein_folding'] = {'status': 'SKIPPED', 'reason': str(e)}

    try:
        results['8e_conservation'] = test_8e_conservation_real()
    except FileNotFoundError as e:
        print(f"Skipping 8e test: {e}")
        results['8e_conservation'] = {'status': 'SKIPPED', 'reason': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status = result.get('status', 'UNKNOWN')
        print(f"  {test_name}: {status}")

    # Save results
    results_file = Path(__file__).parent / 'real_data_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    main()
