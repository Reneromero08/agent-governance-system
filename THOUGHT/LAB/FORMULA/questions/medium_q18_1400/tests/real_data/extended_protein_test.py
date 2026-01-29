#!/usr/bin/env python3
"""
Q18 Extended Protein Folding Test - 50+ Proteins

Scale up the pilot test (r=0.726 with 5 proteins) to 50+ diverse proteins.
Uses REAL AlphaFold data with pLDDT confidence scores.

Data source: AlphaFold DB (https://alphafold.ebi.ac.uk/)
URL format: https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb
"""

import json
import numpy as np
import urllib.request
import ssl
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Disable SSL verification for some servers
ssl._create_default_https_context = ssl._create_unverified_context

CACHE_DIR = Path(__file__).parent / 'cache'
ALPHAFOLD_DIR = CACHE_DIR / 'alphafold'
RESULTS_DIR = Path(__file__).parent

# =============================================================================
# 50 DIVERSE PROTEINS (different protein classes)
# =============================================================================

EXTENDED_PROTEIN_IDS = [
    # Kinases
    "P00533",  # EGFR
    "P00519",  # ABL1
    "P04049",  # RAF1
    "P06400",  # RB1
    "P08069",  # IGF1R
    "P10275",  # AR (Androgen receptor)
    "P11802",  # CDK4
    "P12931",  # SRC
    "P15056",  # BRAF
    "P17252",  # PRKCA
    "P27361",  # MAPK3 (ERK1)
    "P31749",  # AKT1
    "P45983",  # MAPK8 (JNK1)
    "P49841",  # GSK3B
    "P53779",  # MAPK10 (JNK3)

    # Transcription factors
    "P04637",  # TP53
    "P16220",  # CREB1
    "P19838",  # NFKB1
    "P35222",  # CTNNB1 (beta-catenin)
    "P40763",  # STAT3
    "P42345",  # MTOR
    "P48431",  # SOX2
    "P84022",  # SMAD3

    # Tumor suppressors
    "P38398",  # BRCA1
    "P42336",  # PIK3CA
    "P51587",  # BRCA2
    "P60484",  # PTEN
    "P46527",  # CDKN1B (p27)
    "P38936",  # CDKN1A (p21)

    # Apoptosis regulators
    "P42574",  # Caspase-3
    "P55210",  # Caspase-7
    "P35354",  # COX2

    # Signaling molecules
    "P01112",  # HRAS
    "P21359",  # NF1
    "P22681",  # CBL
    "P23458",  # JAK1
    "P24385",  # CCND1
    "P29350",  # PTPN6 (SHP1)
    "P30304",  # CDC25A
    "P35968",  # KDR (VEGFR2)
    "P52564",  # MAP2K6

    # Structural/scaffolding proteins
    "P61073",  # CXCR4
    "P62258",  # YWHAE (14-3-3 epsilon)
    "P63092",  # GNAS
    "P67870",  # CSNK2B
    "P68400",  # CSNK2A1
    "P78527",  # DNA-PKcs (PRKDC)
    "P98170",  # XIAP

    # Others
    "Q00987",  # MDM2
    "Q02156",  # PRKCE
]


# =============================================================================
# AMINO ACID PROPERTIES (from biochemistry literature)
# =============================================================================

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

DISORDER_AA = set('DEKRSPQGN')
HELIX_AA = set('AELM')
SHEET_AA = set('VIY')


# =============================================================================
# DATA FETCHING
# =============================================================================

def download_protein(uniprot_id: str) -> bool:
    """Download a single protein from AlphaFold."""
    output_file = ALPHAFOLD_DIR / f"{uniprot_id}.pdb"

    if output_file.exists():
        print(f"  {uniprot_id}: Already downloaded")
        return True

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Q18 Research)',
            'Accept': '*/*'
        })

        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

        with open(output_file, 'wb') as f:
            f.write(content)

        print(f"  {uniprot_id}: Downloaded ({len(content)} bytes)")
        return True

    except Exception as e:
        print(f"  {uniprot_id}: FAILED - {e}")
        return False


def download_all_proteins(protein_ids: List[str]) -> Dict[str, bool]:
    """Download all proteins from AlphaFold."""
    print("\n" + "=" * 70)
    print("DOWNLOADING PROTEINS FROM ALPHAFOLD")
    print("=" * 70)

    ALPHAFOLD_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    n_total = len(protein_ids)

    for i, uniprot_id in enumerate(protein_ids, 1):
        print(f"[{i}/{n_total}] ", end="")
        results[uniprot_id] = download_protein(uniprot_id)
        time.sleep(0.5)  # Be polite to the server

    success = sum(1 for v in results.values() if v)
    print(f"\nDownloaded: {success}/{n_total} proteins")

    return results


# =============================================================================
# pLDDT EXTRACTION
# =============================================================================

def extract_plddt_from_pdb(pdb_file: Path) -> Dict:
    """Extract pLDDT scores from AlphaFold PDB file."""
    plddt_values = []
    residue_plddt = {}  # One value per residue (CA atom)

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        # pLDDT is in B-factor column (columns 61-66, 1-indexed)
                        plddt = float(line[60:66].strip())
                        plddt_values.append(plddt)

                        # For per-residue, use CA atom
                        if ' CA ' in line:
                            res_num = int(line[22:26].strip())
                            residue_plddt[res_num] = plddt
                    except (ValueError, IndexError):
                        pass

        if not plddt_values:
            return None

        return {
            'mean_plddt': float(np.mean(plddt_values)),
            'std_plddt': float(np.std(plddt_values)),
            'min_plddt': float(np.min(plddt_values)),
            'max_plddt': float(np.max(plddt_values)),
            'n_atoms': len(plddt_values),
            'n_residues': len(residue_plddt),
            'residue_plddt': list(residue_plddt.values())
        }

    except Exception as e:
        print(f"  Error reading {pdb_file.name}: {e}")
        return None


def extract_sequence_from_pdb(pdb_file: Path) -> Optional[str]:
    """Extract amino acid sequence from PDB file."""
    AA_MAP = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    residues = {}

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())

                    if res_name in AA_MAP:
                        residues[res_num] = AA_MAP[res_name]

        if not residues:
            return None

        # Build sequence from ordered residue numbers
        seq = ''.join(residues[i] for i in sorted(residues.keys()))
        return seq

    except Exception as e:
        print(f"  Error extracting sequence from {pdb_file.name}: {e}")
        return None


def extract_all_plddt() -> Dict[str, Dict]:
    """Extract pLDDT scores from all downloaded PDB files."""
    print("\n" + "=" * 70)
    print("EXTRACTING pLDDT SCORES")
    print("=" * 70)

    pdb_files = list(ALPHAFOLD_DIR.glob('*.pdb'))
    print(f"Found {len(pdb_files)} PDB files")

    results = {}

    for pdb_file in pdb_files:
        uniprot_id = pdb_file.stem
        plddt_data = extract_plddt_from_pdb(pdb_file)
        sequence = extract_sequence_from_pdb(pdb_file)

        if plddt_data and sequence:
            plddt_data['sequence'] = sequence
            plddt_data['length'] = len(sequence)
            results[uniprot_id] = plddt_data
            print(f"  {uniprot_id}: pLDDT={plddt_data['mean_plddt']:.1f}, length={len(sequence)}")
        else:
            print(f"  {uniprot_id}: Failed to extract data")

    print(f"\nExtracted data for {len(results)} proteins")

    # Save to cache
    plddt_file = CACHE_DIR / 'extended_plddt.json'
    with open(plddt_file, 'w') as f:
        # Don't save residue_plddt array to keep file manageable
        save_data = {}
        for uid, data in results.items():
            save_data[uid] = {k: v for k, v in data.items() if k != 'residue_plddt'}
        json.dump(save_data, f, indent=2)
    print(f"Saved to: {plddt_file}")

    return results


# =============================================================================
# SEQUENCE FEATURE COMPUTATION
# =============================================================================

def compute_sequence_features(sequence: str) -> Dict[str, float]:
    """Compute features from protein sequence."""
    seq = sequence.upper()
    n = len(seq)

    if n == 0:
        return {'length': 0, 'hydrophobicity_mean': 0, 'hydrophobicity_std': 0,
                'disorder_frac': 0, 'helix_prop': 0, 'sheet_prop': 0, 'complexity': 0}

    # Hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    mean_hydro = float(np.mean(hydro_values))
    std_hydro = float(np.std(hydro_values)) if len(hydro_values) > 1 else 0

    # Disorder fraction
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

    complexity = entropy / np.log2(20)  # Normalize

    return {
        'length': n,
        'hydrophobicity_mean': mean_hydro,
        'hydrophobicity_std': std_hydro,
        'disorder_frac': disorder_frac,
        'helix_prop': helix_prop,
        'sheet_prop': sheet_prop,
        'complexity': complexity
    }


def compute_R_from_features(features: Dict[str, float]) -> float:
    """
    Compute R = E / sigma from sequence features.

    E: Foldability estimate based on sequence composition
    sigma: Sequence heterogeneity
    """
    # E: Higher for ordered sequences, balanced hydrophobicity
    order_score = 1.0 - features['disorder_frac']
    hydro_balance = 1.0 - abs(features['hydrophobicity_mean']) / 4.5
    structure_prop = features['helix_prop'] + features['sheet_prop']
    complexity_penalty = abs(features['complexity'] - 0.75)

    E = 0.4 * order_score + 0.3 * hydro_balance + 0.2 * structure_prop + 0.1 * (1 - complexity_penalty)

    # sigma: Sequence heterogeneity
    sigma = max(features['hydrophobicity_std'] / 4.5, 0.01)

    R = E / sigma
    return R


# =============================================================================
# FOLDING PREDICTION TEST
# =============================================================================

def run_folding_prediction_test(plddt_data: Dict[str, Dict]) -> Dict:
    """
    Test if R computed from sequence predicts AlphaFold pLDDT.

    This is the core Q18 test:
    - R is computed from sequence features ONLY (independent of pLDDT)
    - pLDDT is the ground truth from AlphaFold
    - Correlation measures if R captures fold quality
    """
    print("\n" + "=" * 70)
    print("FOLDING PREDICTION TEST: R vs pLDDT")
    print("=" * 70)

    results = []

    for uniprot_id, data in plddt_data.items():
        if 'sequence' not in data:
            continue

        # Compute R from sequence features
        features = compute_sequence_features(data['sequence'])
        R_value = compute_R_from_features(features)

        results.append({
            'uniprot_id': uniprot_id,
            'R_sequence': R_value,
            'mean_plddt': data['mean_plddt'],
            'std_plddt': data['std_plddt'],
            'length': data['length'],
            'disorder_frac': features['disorder_frac'],
            'complexity': features['complexity']
        })

    if len(results) < 5:
        print(f"ERROR: Only {len(results)} proteins with valid data")
        return {'status': 'INSUFFICIENT_DATA', 'n_proteins': len(results)}

    # Extract arrays for correlation
    R_values = np.array([r['R_sequence'] for r in results])
    plddt_values = np.array([r['mean_plddt'] for r in results])

    # Pearson correlation
    r_corr = float(np.corrcoef(R_values, plddt_values)[0, 1])

    # Spearman correlation (rank-based)
    from scipy import stats as sp_stats
    spearman_r, spearman_p = sp_stats.spearmanr(R_values, plddt_values)

    # Linear regression
    slope, intercept = np.polyfit(R_values, plddt_values, 1)
    predicted_plddt = slope * R_values + intercept
    ss_res = np.sum((plddt_values - predicted_plddt) ** 2)
    ss_tot = np.sum((plddt_values - np.mean(plddt_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Statistics
    mean_R = float(np.mean(R_values))
    std_R = float(np.std(R_values))
    mean_plddt = float(np.mean(plddt_values))
    std_plddt = float(np.std(plddt_values))

    # Print results table
    print(f"\n{'UniProt':<10} {'R_seq':>8} {'pLDDT':>8} {'Length':>8} {'Disorder':>10}")
    print("-" * 50)

    # Sort by R for display
    sorted_results = sorted(results, key=lambda x: x['R_sequence'], reverse=True)
    for r in sorted_results:
        print(f"{r['uniprot_id']:<10} {r['R_sequence']:>8.3f} {r['mean_plddt']:>8.1f} "
              f"{r['length']:>8} {r['disorder_frac']:>10.3f}")

    print("\n" + "=" * 50)
    print("CORRELATION RESULTS")
    print("=" * 50)
    print(f"N proteins:        {len(results)}")
    print(f"Mean R:            {mean_R:.4f} +/- {std_R:.4f}")
    print(f"Mean pLDDT:        {mean_plddt:.2f} +/- {std_plddt:.2f}")
    print(f"Pearson r:         {r_corr:.4f}")
    print(f"Spearman r:        {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"R-squared:         {r_squared:.4f}")
    print(f"Regression:        pLDDT = {slope:.2f} * R + {intercept:.2f}")

    # Interpretation
    if r_corr > 0.5:
        verdict = "STRONG: R from sequence strongly predicts fold quality"
    elif r_corr > 0.3:
        verdict = "MODERATE: R from sequence moderately predicts fold quality"
    elif r_corr > 0:
        verdict = "WEAK: Positive but weak relationship"
    else:
        verdict = "NONE: No relationship between R and pLDDT"

    print(f"\nVerdict: {verdict}")

    # Compare to pilot (5 proteins, r=0.726)
    print(f"\nComparison to pilot (5 proteins, r=0.726):")
    if abs(r_corr - 0.726) < 0.1:
        print("  Result is consistent with pilot")
    elif r_corr > 0.726:
        print(f"  Result is STRONGER than pilot (+{r_corr - 0.726:.3f})")
    else:
        print(f"  Result is weaker than pilot ({r_corr - 0.726:.3f})")

    return {
        'status': 'PASS' if r_corr > 0.3 else 'FAIL',
        'n_proteins': len(results),
        'pearson_r': r_corr,
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'r_squared': float(r_squared),
        'regression_slope': float(slope),
        'regression_intercept': float(intercept),
        'mean_R': mean_R,
        'std_R': std_R,
        'mean_plddt': mean_plddt,
        'std_plddt': std_plddt,
        'verdict': verdict,
        'pilot_comparison': {
            'pilot_r': 0.726,
            'pilot_n': 5,
            'difference': float(r_corr - 0.726)
        },
        'protein_results': results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Q18 EXTENDED PROTEIN FOLDING TEST")
    print("Scaling from 5 to 50+ proteins")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target proteins: {len(EXTENDED_PROTEIN_IDS)}")

    # Create directories
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ALPHAFOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download all proteins
    download_results = download_all_proteins(EXTENDED_PROTEIN_IDS)

    # Step 2: Extract pLDDT scores
    plddt_data = extract_all_plddt()

    # Step 3: Run folding prediction test
    test_results = run_folding_prediction_test(plddt_data)

    # Final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'extended_protein_folding',
        'description': 'R = E/sigma predicting AlphaFold pLDDT',
        'data_source': 'AlphaFold DB (real structures)',
        'target_proteins': len(EXTENDED_PROTEIN_IDS),
        'downloaded_proteins': sum(1 for v in download_results.values() if v),
        'proteins_with_data': test_results.get('n_proteins', 0),
        'download_results': download_results,
        'test_results': test_results,
        'summary': {
            'pearson_r': test_results.get('pearson_r'),
            'spearman_r': test_results.get('spearman_r'),
            'r_squared': test_results.get('r_squared'),
            'verdict': test_results.get('verdict'),
            'comparison_to_pilot': test_results.get('pilot_comparison')
        }
    }

    # Save results
    output_file = RESULTS_DIR / 'extended_protein_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Proteins tested: {test_results.get('n_proteins', 0)}")
    print(f"Pearson r: {test_results.get('pearson_r', 'N/A'):.4f}" if test_results.get('pearson_r') else "Pearson r: N/A")
    print(f"Spearman r: {test_results.get('spearman_r', 'N/A'):.4f}" if test_results.get('spearman_r') else "Spearman r: N/A")
    print(f"Verdict: {test_results.get('verdict', 'N/A')}")
    print(f"\nResults saved to: {output_file}")

    return final_results


if __name__ == '__main__':
    main()
