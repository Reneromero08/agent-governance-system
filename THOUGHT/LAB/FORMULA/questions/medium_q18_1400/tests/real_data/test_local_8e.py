#!/usr/bin/env python3
"""
Q18 Investigation: Local 8e in Protein Functional Regions

HYPOTHESIS: While global ESM-2 embeddings show Df x alpha = 45-52 (NOT 8e),
8e might emerge in LOCAL functional regions where semiotic compression is highest.

APPROACH:
Since we don't have ESM-2 access, we simulate the test with:
1. Per-residue pLDDT scores (from AlphaFold PDBs)
2. Amino acid property profiles (hydrophobicity, charge, volume, flexibility)
3. Sliding window embeddings (window size 10-50 residues)

For each protein, we extract LOCAL embeddings from:
- Known active sites (from literature/UniProt)
- Conserved motifs (high pLDDT regions)
- Disordered regions (low pLDDT - control)
- Random regions (control)

KEY QUESTION: Does 8e emerge in LOCAL functional regions even if not globally?

Author: Claude Opus 4.5
Date: 2026-01-25
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
RANDOM_BASELINE = 14.5

# Amino acid property tables (from literature)
# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Charge at physiological pH
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Molecular volume (Angstrom^3)
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}

# Flexibility index (B-factor correlation)
FLEXIBILITY = {
    'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346,
    'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462,
    'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509,
    'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386
}

# Polarity (Grantham scale normalized)
POLARITY = {
    'A': 0.0, 'R': 0.52, 'N': 0.35, 'D': 0.35, 'C': 0.04,
    'Q': 0.35, 'E': 0.35, 'G': 0.0, 'H': 0.17, 'I': 0.0,
    'L': 0.0, 'K': 0.37, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.11, 'T': 0.11, 'W': 0.0, 'Y': 0.04, 'V': 0.0
}

# Secondary structure propensity (alpha-helix)
HELIX_PROPENSITY = {
    'A': 1.41, 'R': 0.98, 'N': 0.76, 'D': 1.01, 'C': 0.66,
    'Q': 1.10, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.79, 'T': 0.82, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}

# Beta-sheet propensity
SHEET_PROPENSITY = {
    'A': 0.72, 'R': 0.90, 'N': 0.73, 'D': 0.54, 'C': 1.40,
    'Q': 1.10, 'E': 0.26, 'G': 0.58, 'H': 0.80, 'I': 1.67,
    'L': 1.22, 'K': 0.74, 'M': 1.67, 'F': 1.33, 'P': 0.31,
    'S': 0.96, 'T': 1.20, 'W': 1.35, 'Y': 1.45, 'V': 1.65
}

# Aromaticity
AROMATICITY = {
    'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
    'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': 0,
    'L': 0, 'K': 0, 'M': 0, 'F': 1, 'P': 0,
    'S': 0, 'T': 0, 'W': 1, 'Y': 1, 'V': 0
}

# Functional region definitions from investigation doc
FUNCTIONAL_REGIONS = {
    "P04637": {
        "name": "TP53",
        "regions": [
            {"name": "TAD1", "start": 1, "end": 42, "type": "TRANSACTIVATION", "expected_8e": "medium"},
            {"name": "TAD2", "start": 43, "end": 62, "type": "TRANSACTIVATION", "expected_8e": "medium"},
            {"name": "PRR", "start": 64, "end": 92, "type": "PROLINE_RICH", "expected_8e": "low"},
            {"name": "DBD", "start": 102, "end": 292, "type": "DNA_BINDING", "expected_8e": "high"},
            {"name": "L2", "start": 164, "end": 194, "type": "DNA_CONTACT", "expected_8e": "very_high"},
            {"name": "L3", "start": 237, "end": 250, "type": "DNA_CONTACT", "expected_8e": "very_high"},
            {"name": "TET", "start": 325, "end": 356, "type": "OLIGOMERIZATION", "expected_8e": "medium"},
            {"name": "CTD", "start": 363, "end": 393, "type": "DISORDERED", "expected_8e": "low"}
        ]
    },
    "P38398": {
        "name": "BRCA1",
        "regions": [
            {"name": "RING", "start": 1, "end": 109, "type": "PROTEIN_INTERFACE", "expected_8e": "high"},
            {"name": "RING_FINGER", "start": 24, "end": 65, "type": "METAL_BINDING", "expected_8e": "very_high"},
            {"name": "CENTRAL", "start": 200, "end": 800, "type": "DISORDERED", "expected_8e": "low"},
            {"name": "BRCT1", "start": 1650, "end": 1736, "type": "PROTEIN_INTERFACE", "expected_8e": "high"},
            {"name": "BRCT2", "start": 1756, "end": 1855, "type": "PROTEIN_INTERFACE", "expected_8e": "high"}
        ]
    },
    "P00533": {
        "name": "EGFR",
        "regions": [
            {"name": "EXTRACELLULAR", "start": 25, "end": 645, "type": "LIGAND_BINDING", "expected_8e": "medium"},
            {"name": "TM", "start": 646, "end": 668, "type": "TRANSMEMBRANE", "expected_8e": "low"},
            {"name": "KINASE_N", "start": 712, "end": 815, "type": "ATP_BINDING", "expected_8e": "high"},
            {"name": "GLY_LOOP", "start": 718, "end": 726, "type": "ATP_BINDING", "expected_8e": "very_high"},
            {"name": "CATALYTIC", "start": 812, "end": 818, "type": "CATALYTIC", "expected_8e": "very_high"},
            {"name": "DFG", "start": 831, "end": 833, "type": "CATALYTIC", "expected_8e": "high"},
            {"name": "ACTIVATION_LOOP", "start": 831, "end": 852, "type": "CATALYTIC", "expected_8e": "high"}
        ]
    },
    "P01112": {
        "name": "HRAS",
        "regions": [
            {"name": "P_LOOP", "start": 10, "end": 17, "type": "ATP_BINDING", "expected_8e": "very_high"},
            {"name": "SWITCH_I", "start": 30, "end": 40, "type": "GTP_BINDING", "expected_8e": "very_high"},
            {"name": "SWITCH_II", "start": 60, "end": 76, "type": "GTP_BINDING", "expected_8e": "very_high"},
            {"name": "G_DOMAIN", "start": 1, "end": 166, "type": "GTPASE", "expected_8e": "high"}
        ]
    },
    "P12931": {
        "name": "SRC",
        "regions": [
            {"name": "SH3", "start": 88, "end": 143, "type": "PROTEIN_INTERFACE", "expected_8e": "high"},
            {"name": "SH2", "start": 151, "end": 248, "type": "PROTEIN_INTERFACE", "expected_8e": "high"},
            {"name": "KINASE", "start": 270, "end": 523, "type": "CATALYTIC", "expected_8e": "high"}
        ]
    },
    "P31749": {
        "name": "AKT1",
        "regions": [
            {"name": "PH", "start": 6, "end": 108, "type": "LIPID_BINDING", "expected_8e": "medium"},
            {"name": "KINASE", "start": 150, "end": 408, "type": "CATALYTIC", "expected_8e": "high"}
        ]
    },
    "P11802": {
        "name": "CDK4",
        "regions": [
            {"name": "KINASE", "start": 3, "end": 295, "type": "CATALYTIC", "expected_8e": "high"}
        ]
    },
    "P42574": {
        "name": "CASPASE3",
        "regions": [
            {"name": "PRODOMAIN", "start": 1, "end": 28, "type": "REGULATORY", "expected_8e": "low"},
            {"name": "LARGE_SUBUNIT", "start": 29, "end": 175, "type": "CATALYTIC", "expected_8e": "medium"},
            {"name": "ACTIVE_SITE", "start": 163, "end": 175, "type": "CATALYTIC", "expected_8e": "very_high"},
            {"name": "SMALL_SUBUNIT", "start": 176, "end": 277, "type": "CATALYTIC", "expected_8e": "medium"}
        ]
    }
}


def to_builtin(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj
    return obj


def get_aa_embedding(residue: str, plddt: float = 70.0, use_extended: bool = True) -> np.ndarray:
    """
    Create an embedding for an amino acid residue.

    Basic (8D):
    0: Hydrophobicity (normalized)
    1: Charge
    2: Volume (normalized)
    3: Flexibility
    4: Polarity
    5: Helix propensity
    6: Sheet propensity
    7: pLDDT (normalized confidence score)

    Extended (20D): Adds one-hot encoding for amino acid identity
    """
    if residue not in HYDROPHOBICITY:
        residue = 'A'  # Default to Alanine for unknown residues

    # Basic embedding (8 features)
    basic = np.array([
        (HYDROPHOBICITY.get(residue, 0) + 4.5) / 9.0,  # Normalize to [0, 1]
        CHARGE.get(residue, 0),
        VOLUME.get(residue, 100) / 228.0,  # Normalize by max (Trp)
        FLEXIBILITY.get(residue, 0.4),
        POLARITY.get(residue, 0),
        HELIX_PROPENSITY.get(residue, 1.0) / 1.51,  # Normalize by max (Glu)
        SHEET_PROPENSITY.get(residue, 1.0) / 1.67,  # Normalize by max (Ile/Met)
        plddt / 100.0  # Normalize pLDDT
    ], dtype=np.float32)

    if not use_extended:
        return basic

    # Extended: Add one-hot encoding (20 amino acids)
    aa_order = 'ARNDCQEGHILKMFPSTWYV'
    one_hot = np.zeros(20, dtype=np.float32)
    if residue in aa_order:
        one_hot[aa_order.index(residue)] = 1.0

    # Also add aromaticity and other features
    extra_features = np.array([
        AROMATICITY.get(residue, 0),
        1.0 if residue in 'VILMFYW' else 0.0,  # Hydrophobic
        1.0 if residue in 'STNQ' else 0.0,  # Polar uncharged
        1.0 if residue in 'DE' else 0.0,  # Acidic
        1.0 if residue in 'KRH' else 0.0,  # Basic
        1.0 if residue == 'G' else 0.0,  # Glycine (special)
        1.0 if residue == 'P' else 0.0,  # Proline (special)
        1.0 if residue == 'C' else 0.0,  # Cysteine (disulfide)
    ], dtype=np.float32)

    return np.concatenate([basic, one_hot, extra_features])


def create_local_embeddings(sequence: str, plddt_scores: List[float],
                           start: int, end: int,
                           window_size: int = 10,
                           use_extended: bool = True) -> np.ndarray:
    """
    Create sliding window embeddings for a local region.

    Returns an array of shape (n_windows, embedding_dim)
    where each row is the mean-pooled embedding of a window.
    """
    # Adjust indices (1-indexed to 0-indexed)
    start_idx = max(0, start - 1)
    end_idx = min(len(sequence), end)

    local_seq = sequence[start_idx:end_idx]
    local_plddt = plddt_scores[start_idx:end_idx] if len(plddt_scores) > start_idx else [70.0] * len(local_seq)

    embed_dim = 36 if use_extended else 8

    if len(local_seq) < window_size:
        # If region is smaller than window, use the whole region
        embeddings = []
        for i, (aa, plddt) in enumerate(zip(local_seq, local_plddt)):
            embeddings.append(get_aa_embedding(aa, plddt, use_extended=use_extended))
        return np.array(embeddings) if embeddings else np.zeros((1, embed_dim))

    # Create sliding windows
    windows = []
    for i in range(len(local_seq) - window_size + 1):
        window_embeds = []
        for j in range(window_size):
            aa = local_seq[i + j]
            plddt = local_plddt[i + j] if i + j < len(local_plddt) else 70.0
            window_embeds.append(get_aa_embedding(aa, plddt, use_extended=use_extended))
        # Mean pooling across window
        window_mean = np.mean(window_embeds, axis=0)
        windows.append(window_mean)

    return np.array(windows) if windows else np.zeros((1, embed_dim))


def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute Df, alpha, and their product from embeddings.

    Returns: (Df, alpha, Df_x_alpha, eigenvalues)
    """
    if len(embeddings) < 3:
        return 1.0, 1.0, 1.0, np.array([1.0])

    # Center the data
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]

    # Compute covariance matrix
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, 1.0, eigenvalues

    # Df: Participation ratio (effective dimensionality)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay rate
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, Df * alpha, eigenvalues


def identify_high_plddt_regions(plddt_scores: List[float],
                                threshold: float = 90.0,
                                min_length: int = 20) -> List[Tuple[int, int]]:
    """Identify contiguous regions with high pLDDT scores (conserved motifs)."""
    regions = []
    in_region = False
    start = 0

    for i, score in enumerate(plddt_scores):
        if score >= threshold and not in_region:
            in_region = True
            start = i
        elif score < threshold and in_region:
            in_region = False
            if i - start >= min_length:
                regions.append((start + 1, i))  # 1-indexed

    # Check if we ended in a region
    if in_region and len(plddt_scores) - start >= min_length:
        regions.append((start + 1, len(plddt_scores)))

    return regions


def identify_low_plddt_regions(plddt_scores: List[float],
                               threshold: float = 50.0,
                               min_length: int = 20) -> List[Tuple[int, int]]:
    """Identify contiguous regions with low pLDDT scores (disordered regions)."""
    regions = []
    in_region = False
    start = 0

    for i, score in enumerate(plddt_scores):
        if score <= threshold and not in_region:
            in_region = True
            start = i
        elif score > threshold and in_region:
            in_region = False
            if i - start >= min_length:
                regions.append((start + 1, i))  # 1-indexed

    if in_region and len(plddt_scores) - start >= min_length:
        regions.append((start + 1, len(plddt_scores)))

    return regions


def generate_random_regions(seq_length: int,
                           region_size: int = 30,
                           n_regions: int = 5,
                           exclude: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """Generate random regions for control comparison."""
    np.random.seed(42)
    regions = []
    exclude = exclude or []

    for _ in range(n_regions * 3):  # Try more times than needed
        if len(regions) >= n_regions:
            break
        start = np.random.randint(1, max(2, seq_length - region_size))
        end = start + region_size

        # Check for overlap with excluded regions
        overlaps = False
        for ex_start, ex_end in exclude:
            if not (end < ex_start or start > ex_end):
                overlaps = True
                break

        if not overlaps:
            regions.append((start, end))

    return regions


def load_protein_data() -> Dict[str, Dict]:
    """Load protein sequences and pLDDT data from cache."""
    cache_path = Path(__file__).parent / "cache" / "extended_plddt.json"
    with open(cache_path, 'r') as f:
        return json.load(f)


def extract_plddt_per_residue(pdb_path: Path) -> List[float]:
    """Extract per-residue pLDDT scores from AlphaFold PDB file."""
    plddt_scores = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and ' CA ' in line:
                    # B-factor column (columns 61-66) contains pLDDT in AlphaFold PDBs
                    try:
                        plddt = float(line[60:66].strip())
                        plddt_scores.append(plddt)
                    except (ValueError, IndexError):
                        plddt_scores.append(70.0)
    except FileNotFoundError:
        pass
    return plddt_scores


def run_local_8e_analysis() -> Dict[str, Any]:
    """Run the local 8e analysis on protein functional regions."""

    print("=" * 80)
    print("Q18 INVESTIGATION: LOCAL 8e IN PROTEIN FUNCTIONAL REGIONS")
    print("=" * 80)
    print()
    print("HYPOTHESIS: 8e might emerge in LOCAL functional regions where")
    print("semiotic compression is highest, even if not globally.")
    print()
    print(f"Theoretical 8e: {EIGHT_E:.4f}")
    print(f"Random baseline: ~{RANDOM_BASELINE}")
    print("=" * 80)

    # Load protein data
    proteins = load_protein_data()
    print(f"\nLoaded {len(proteins)} proteins")

    results = {
        "timestamp": datetime.now().isoformat(),
        "theoretical_8e": float(EIGHT_E),
        "hypothesis": "8e emerges in local functional regions where meaning is concentrated",
        "region_results": [],
        "category_results": {},
        "control_results": [],
        "statistical_summary": {}
    }

    # Category pools for aggregated analysis
    category_pools = {
        "CATALYTIC": [],
        "DNA_BINDING": [],
        "ATP_BINDING": [],
        "GTP_BINDING": [],
        "PROTEIN_INTERFACE": [],
        "METAL_BINDING": [],
        "DISORDERED": [],
        "REGULATORY": [],
        "OTHER": []
    }

    plddt_high_pool = []  # High pLDDT regions (conserved)
    plddt_low_pool = []   # Low pLDDT regions (disordered)
    random_pool = []      # Random control regions

    # Analyze each protein with known functional regions
    for prot_id, regions_data in FUNCTIONAL_REGIONS.items():
        if prot_id not in proteins:
            print(f"  Skipping {prot_id}: not in protein data")
            continue

        prot_data = proteins[prot_id]
        sequence = prot_data.get('sequence', '')
        if not sequence:
            continue

        print(f"\n[{regions_data['name']}] {prot_id} ({len(sequence)} residues)")
        print("-" * 50)

        # Try to load per-residue pLDDT from PDB
        pdb_path = Path(__file__).parent / "cache" / "alphafold" / f"{prot_id}.pdb"
        plddt_scores = extract_plddt_per_residue(pdb_path)

        if not plddt_scores:
            # Fall back to uniform pLDDT based on mean
            mean_plddt = prot_data.get('mean_plddt', 70.0)
            plddt_scores = [mean_plddt] * len(sequence)

        # Analyze each defined functional region
        for region in regions_data["regions"]:
            region_name = region["name"]
            start = region["start"]
            end = region["end"]
            region_type = region["type"]
            expected_8e = region.get("expected_8e", "unknown")

            # Create local embeddings
            embeddings = create_local_embeddings(sequence, plddt_scores, start, end, window_size=10)

            if len(embeddings) < 5:
                print(f"  {region_name}: Too few samples ({len(embeddings)}), skipping")
                continue

            # Compute spectral properties
            Df, alpha, Df_x_alpha, eigenvalues = compute_spectral_properties(embeddings)
            deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E

            region_result = {
                "protein": prot_id,
                "protein_name": regions_data["name"],
                "region": region_name,
                "type": region_type,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "n_samples": len(embeddings),
                "Df": float(Df),
                "alpha": float(alpha),
                "Df_x_alpha": float(Df_x_alpha),
                "deviation_from_8e": float(deviation),
                "deviation_percent": float(deviation * 100),
                "expected_8e": expected_8e,
                "passes_8e": deviation < 0.15
            }
            results["region_results"].append(region_result)

            status = "PASS" if deviation < 0.15 else ("WEAK" if deviation < 0.30 else "FAIL")
            print(f"  {region_name} ({region_type}): Df x alpha = {Df_x_alpha:.2f} ({deviation*100:.1f}% dev) [{status}]")

            # Add to category pool
            cat_key = region_type if region_type in category_pools else "OTHER"
            category_pools[cat_key].append(embeddings)

        # Also analyze pLDDT-based regions
        high_plddt_regions = identify_high_plddt_regions(plddt_scores, threshold=90.0, min_length=20)
        for i, (start, end) in enumerate(high_plddt_regions[:3]):  # Limit to 3
            embeddings = create_local_embeddings(sequence, plddt_scores, start, end, window_size=10)
            if len(embeddings) >= 5:
                plddt_high_pool.append(embeddings)

        low_plddt_regions = identify_low_plddt_regions(plddt_scores, threshold=50.0, min_length=20)
        for i, (start, end) in enumerate(low_plddt_regions[:3]):
            embeddings = create_local_embeddings(sequence, plddt_scores, start, end, window_size=10)
            if len(embeddings) >= 5:
                plddt_low_pool.append(embeddings)

        # Random control regions
        functional_coords = [(r["start"], r["end"]) for r in regions_data["regions"]]
        random_regions = generate_random_regions(len(sequence), region_size=30, n_regions=3, exclude=functional_coords)
        for start, end in random_regions:
            embeddings = create_local_embeddings(sequence, plddt_scores, start, end, window_size=10)
            if len(embeddings) >= 5:
                random_pool.append(embeddings)

    # Analyze pooled categories
    print("\n" + "=" * 80)
    print("POOLED CATEGORY ANALYSIS")
    print("=" * 80)

    for category, pool in category_pools.items():
        if not pool:
            continue

        # Combine all embeddings in category
        all_embeddings = np.vstack(pool)

        if len(all_embeddings) < 10:
            continue

        Df, alpha, Df_x_alpha, _ = compute_spectral_properties(all_embeddings)
        deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E

        results["category_results"][category] = {
            "n_regions": len(pool),
            "n_samples": len(all_embeddings),
            "Df": float(Df),
            "alpha": float(alpha),
            "Df_x_alpha": float(Df_x_alpha),
            "deviation_from_8e": float(deviation),
            "deviation_percent": float(deviation * 100),
            "passes_8e": deviation < 0.15
        }

        status = "PASS" if deviation < 0.15 else ("WEAK" if deviation < 0.30 else "FAIL")
        print(f"  {category:<20} ({len(pool):>2} regions, {len(all_embeddings):>4} samples): "
              f"Df x alpha = {Df_x_alpha:>6.2f} ({deviation*100:>5.1f}% dev) [{status}]")

    # Analyze control pools
    print("\n" + "=" * 80)
    print("CONTROL COMPARISONS")
    print("=" * 80)

    # High pLDDT (conserved motifs)
    if plddt_high_pool:
        all_high = np.vstack(plddt_high_pool)
        Df, alpha, Df_x_alpha, _ = compute_spectral_properties(all_high)
        deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E
        results["control_results"].append({
            "name": "High pLDDT (conserved motifs)",
            "n_regions": len(plddt_high_pool),
            "n_samples": len(all_high),
            "Df": float(Df),
            "alpha": float(alpha),
            "Df_x_alpha": float(Df_x_alpha),
            "deviation_percent": float(deviation * 100),
            "passes_8e": deviation < 0.15
        })
        status = "PASS" if deviation < 0.15 else ("WEAK" if deviation < 0.30 else "FAIL")
        print(f"  High pLDDT (conserved): Df x alpha = {Df_x_alpha:.2f} ({deviation*100:.1f}% dev) [{status}]")

    # Low pLDDT (disordered)
    if plddt_low_pool:
        all_low = np.vstack(plddt_low_pool)
        Df, alpha, Df_x_alpha, _ = compute_spectral_properties(all_low)
        deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E
        results["control_results"].append({
            "name": "Low pLDDT (disordered regions)",
            "n_regions": len(plddt_low_pool),
            "n_samples": len(all_low),
            "Df": float(Df),
            "alpha": float(alpha),
            "Df_x_alpha": float(Df_x_alpha),
            "deviation_percent": float(deviation * 100),
            "passes_8e": deviation < 0.15
        })
        status = "PASS" if deviation < 0.15 else ("WEAK" if deviation < 0.30 else "FAIL")
        print(f"  Low pLDDT (disordered): Df x alpha = {Df_x_alpha:.2f} ({deviation*100:.1f}% dev) [{status}]")

    # Random regions
    if random_pool:
        all_random = np.vstack(random_pool)
        Df, alpha, Df_x_alpha, _ = compute_spectral_properties(all_random)
        deviation = abs(Df_x_alpha - EIGHT_E) / EIGHT_E
        results["control_results"].append({
            "name": "Random regions (control)",
            "n_regions": len(random_pool),
            "n_samples": len(all_random),
            "Df": float(Df),
            "alpha": float(alpha),
            "Df_x_alpha": float(Df_x_alpha),
            "deviation_percent": float(deviation * 100),
            "passes_8e": deviation < 0.15
        })
        status = "PASS" if deviation < 0.15 else ("WEAK" if deviation < 0.30 else "FAIL")
        print(f"  Random regions (control): Df x alpha = {Df_x_alpha:.2f} ({deviation*100:.1f}% dev) [{status}]")

    # Statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    region_products = [r["Df_x_alpha"] for r in results["region_results"]]
    if region_products:
        mean_product = np.mean(region_products)
        std_product = np.std(region_products)
        min_product = np.min(region_products)
        max_product = np.max(region_products)

        n_pass = sum(1 for r in results["region_results"] if r["passes_8e"])
        n_weak = sum(1 for r in results["region_results"] if 0.15 <= r["deviation_from_8e"] < 0.30)
        n_fail = sum(1 for r in results["region_results"] if r["deviation_from_8e"] >= 0.30)

        results["statistical_summary"] = {
            "n_regions_analyzed": len(region_products),
            "mean_Df_x_alpha": float(mean_product),
            "std_Df_x_alpha": float(std_product),
            "min_Df_x_alpha": float(min_product),
            "max_Df_x_alpha": float(max_product),
            "n_pass": n_pass,
            "n_weak": n_weak,
            "n_fail": n_fail,
            "pass_rate": float(n_pass / len(region_products)) if region_products else 0.0
        }

        print(f"\n  Regions analyzed: {len(region_products)}")
        print(f"  Mean Df x alpha: {mean_product:.2f} +/- {std_product:.2f}")
        print(f"  Range: [{min_product:.2f}, {max_product:.2f}]")
        print(f"\n  PASS (<15% dev): {n_pass} ({n_pass/len(region_products)*100:.1f}%)")
        print(f"  WEAK (15-30% dev): {n_weak} ({n_weak/len(region_products)*100:.1f}%)")
        print(f"  FAIL (>30% dev): {n_fail} ({n_fail/len(region_products)*100:.1f}%)")

    # Key finding
    print("\n" + "=" * 80)
    print("KEY FINDING")
    print("=" * 80)

    # Compare functional vs control categories
    functional_types = ["CATALYTIC", "DNA_BINDING", "ATP_BINDING", "GTP_BINDING", "PROTEIN_INTERFACE", "METAL_BINDING"]
    functional_products = []
    for ft in functional_types:
        if ft in results["category_results"]:
            functional_products.append(results["category_results"][ft]["Df_x_alpha"])

    control_products = [c["Df_x_alpha"] for c in results["control_results"]]

    if functional_products and control_products:
        mean_functional = np.mean(functional_products)
        mean_control = np.mean(control_products)

        functional_deviation = abs(mean_functional - EIGHT_E) / EIGHT_E
        control_deviation = abs(mean_control - EIGHT_E) / EIGHT_E

        if functional_deviation < control_deviation:
            print(f"\n  FUNCTIONAL regions are CLOSER to 8e than controls!")
            print(f"  Functional mean: {mean_functional:.2f} ({functional_deviation*100:.1f}% dev)")
            print(f"  Control mean: {mean_control:.2f} ({control_deviation*100:.1f}% dev)")
            print(f"\n  This SUPPORTS the hypothesis that 8e emerges in regions")
            print(f"  where semiotic compression is highest.")
            results["key_finding"] = {
                "status": "PARTIAL_SUPPORT",
                "message": "Functional regions are closer to 8e than controls",
                "functional_mean": float(mean_functional),
                "control_mean": float(mean_control),
                "functional_deviation_percent": float(functional_deviation * 100),
                "control_deviation_percent": float(control_deviation * 100)
            }
        else:
            print(f"\n  Control regions show similar or better 8e proximity than functional regions.")
            print(f"  Functional mean: {mean_functional:.2f} ({functional_deviation*100:.1f}% dev)")
            print(f"  Control mean: {mean_control:.2f} ({control_deviation*100:.1f}% dev)")
            print(f"\n  This DOES NOT support the hypothesis about local 8e emergence.")
            results["key_finding"] = {
                "status": "NOT_SUPPORTED",
                "message": "Functional regions do not show stronger 8e than controls",
                "functional_mean": float(mean_functional),
                "control_mean": float(mean_control),
                "functional_deviation_percent": float(functional_deviation * 100),
                "control_deviation_percent": float(control_deviation * 100)
            }
    else:
        results["key_finding"] = {
            "status": "INSUFFICIENT_DATA",
            "message": "Not enough data to compare functional vs control regions"
        }

    print("=" * 80)

    return results


def main():
    """Main entry point."""
    results = run_local_8e_analysis()

    # Convert for JSON serialization
    results = to_builtin(results)

    # Save results
    output_path = Path(__file__).parent / "local_8e_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
