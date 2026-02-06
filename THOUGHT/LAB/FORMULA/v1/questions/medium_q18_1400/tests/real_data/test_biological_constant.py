#!/usr/bin/env python3
"""
Q18: Test the Biological Semiotic Constant Hypothesis

HYPOTHESIS:
ESM-2 protein embeddings show Df x alpha = 45-52 (not 8e = 21.75).
This suggests biology has 4 semiotic categories (vs 3 for human thought),
giving Bf = 2^4 x e = 43.5

The 4th category is "Evolutionary Context" (fitness, selection) - absent from
human thought.

APPROACH:
Test if Bf = 43.5 is a consistent constant across multiple protein representations:
1. One-hot encoding (baseline)
2. k-mer frequency vectors (k=3, k=5)
3. Amino acid property matrix (hydrophobicity, charge, volume, etc.)
4. BLOSUM62 embedding
5. Combined/ESM-like embedding

KEY QUESTION: Is ~45-52 a stable constant (CV < 15%) or just noise?

Author: Claude
Date: 2026-01-25
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import Counter
import math

# =============================================================================
# CONSTANTS
# =============================================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Target constants
E_CONST = 2.718281828459045
TARGET_8E = 8 * E_CONST  # 21.746 - human semiotic constant
TARGET_BF = 16 * E_CONST  # 43.493 - hypothesized biological constant (2^4 * e)
ESM2_OBSERVED_RANGE = (45, 52)  # Observed in ESM-2 embeddings

# Amino acid physicochemical properties
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

VOLUMES = {  # Angstrom^3
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
}

CHARGE = {  # Net charge at pH 7
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

POLARITY = {  # 1 = polar, 0 = nonpolar
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

# Secondary structure propensity (Chou-Fasman)
HELIX_PROPENSITY = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
}

SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
}

# Full BLOSUM62 matrix (symmetric)
BLOSUM62 = {
    'A': {'A':4,'R':-1,'N':-2,'D':-2,'C':0,'Q':-1,'E':-1,'G':0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':0,'W':-3,'Y':-2,'V':0},
    'R': {'A':-1,'R':5,'N':0,'D':-2,'C':-3,'Q':1,'E':0,'G':-2,'H':0,'I':-3,'L':-2,'K':2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
    'N': {'A':-2,'R':0,'N':6,'D':1,'C':-3,'Q':0,'E':0,'G':0,'H':1,'I':-3,'L':-3,'K':0,'M':-2,'F':-3,'P':-2,'S':1,'T':0,'W':-4,'Y':-2,'V':-3},
    'D': {'A':-2,'R':-2,'N':1,'D':6,'C':-3,'Q':0,'E':2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S':0,'T':-1,'W':-4,'Y':-3,'V':-3},
    'C': {'A':0,'R':-3,'N':-3,'D':-3,'C':9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
    'Q': {'A':-1,'R':1,'N':0,'D':0,'C':-3,'Q':5,'E':2,'G':-2,'H':0,'I':-3,'L':-2,'K':1,'M':0,'F':-3,'P':-1,'S':0,'T':-1,'W':-2,'Y':-1,'V':-2},
    'E': {'A':-1,'R':0,'N':0,'D':2,'C':-4,'Q':2,'E':5,'G':-2,'H':0,'I':-3,'L':-3,'K':1,'M':-2,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'G': {'A':0,'R':-2,'N':0,'D':-1,'C':-3,'Q':-2,'E':-2,'G':6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S':0,'T':-2,'W':-2,'Y':-3,'V':-3},
    'H': {'A':-2,'R':0,'N':1,'D':-1,'C':-3,'Q':0,'E':0,'G':-2,'H':8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y':2,'V':-3},
    'I': {'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I':4,'L':2,'K':-3,'M':1,'F':0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V':3},
    'L': {'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I':2,'L':4,'K':-2,'M':2,'F':0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V':1},
    'K': {'A':-1,'R':2,'N':0,'D':-1,'C':-3,'Q':1,'E':1,'G':-2,'H':-1,'I':-3,'L':-2,'K':5,'M':-1,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'M': {'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q':0,'E':-2,'G':-3,'H':-2,'I':1,'L':2,'K':-1,'M':5,'F':0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V':1},
    'F': {'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I':0,'L':0,'K':-3,'M':0,'F':6,'P':-4,'S':-2,'T':-2,'W':1,'Y':3,'V':-1},
    'P': {'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P':7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
    'S': {'A':1,'R':-1,'N':1,'D':0,'C':-1,'Q':0,'E':0,'G':0,'H':-1,'I':-2,'L':-2,'K':0,'M':-1,'F':-2,'P':-1,'S':4,'T':1,'W':-3,'Y':-2,'V':-2},
    'T': {'A':0,'R':-1,'N':0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':5,'W':-2,'Y':-2,'V':0},
    'W': {'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F':1,'P':-4,'S':-3,'T':-2,'W':11,'Y':2,'V':-3},
    'Y': {'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H':2,'I':-1,'L':-1,'K':-2,'M':-1,'F':3,'P':-3,'S':-2,'T':-2,'W':2,'Y':7,'V':-1},
    'V': {'A':0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I':3,'L':1,'K':-2,'M':1,'F':-1,'P':-2,'S':-2,'T':0,'W':-3,'Y':-1,'V':4}
}

# Paths
CACHE_DIR = Path(__file__).parent / 'cache'
ALPHAFOLD_DIR = CACHE_DIR / 'alphafold'
RESULTS_DIR = Path(__file__).parent


# =============================================================================
# EMBEDDING METHODS
# =============================================================================

def one_hot_encoding(sequence: str) -> np.ndarray:
    """
    Method 1: Simple one-hot encoding
    Dimension: 20 (one per amino acid)
    """
    n = len(sequence)
    encoding = np.zeros((n, 20))
    for i, aa in enumerate(sequence.upper()):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
    return encoding


def kmer_frequency(sequence: str, k: int = 3) -> np.ndarray:
    """
    Method 2: k-mer frequency vectors
    Dimension: 20^k (but we use sparse representation)
    """
    seq = sequence.upper()
    n = len(seq)

    if n < k:
        return np.zeros(20**min(k, 2))  # Return zeros for short sequences

    # Count k-mers
    kmer_counts = Counter()
    for i in range(n - k + 1):
        kmer = seq[i:i+k]
        if all(aa in AA_TO_IDX for aa in kmer):
            kmer_counts[kmer] += 1

    # Create frequency vector (use hash for larger k)
    if k <= 2:
        dim = 20**k
        vec = np.zeros(dim)
        for kmer, count in kmer_counts.items():
            idx = sum(AA_TO_IDX[aa] * (20**i) for i, aa in enumerate(kmer))
            vec[idx] = count
    else:
        # Use fixed-size hash for larger k
        dim = 1000  # Fixed dimension
        vec = np.zeros(dim)
        for kmer, count in kmer_counts.items():
            idx = hash(kmer) % dim
            vec[idx] += count

    # Normalize
    total = vec.sum()
    if total > 0:
        vec = vec / total

    return vec


def property_matrix_encoding(sequence: str) -> np.ndarray:
    """
    Method 3: Amino acid property matrix
    Each residue is represented by 6 physicochemical properties
    """
    n = len(sequence)
    seq = sequence.upper()

    encoding = np.zeros((n, 6))
    for i, aa in enumerate(seq):
        if aa in HYDROPHOBICITY:
            # Normalized properties
            encoding[i, 0] = (HYDROPHOBICITY[aa] + 4.5) / 9.0  # Range [-4.5, 4.5] -> [0, 1]
            encoding[i, 1] = VOLUMES[aa] / 227.8  # Normalize by max
            encoding[i, 2] = (CHARGE[aa] + 1) / 2.0  # Range [-1, 1] -> [0, 1]
            encoding[i, 3] = POLARITY[aa]
            encoding[i, 4] = HELIX_PROPENSITY[aa] / 1.51  # Normalize by max
            encoding[i, 5] = SHEET_PROPENSITY[aa] / 1.70  # Normalize by max

    return encoding


def blosum62_encoding(sequence: str) -> np.ndarray:
    """
    Method 4: BLOSUM62-based embedding
    Each residue is represented by its BLOSUM62 row (similarity to all 20 AAs)
    """
    n = len(sequence)
    seq = sequence.upper()

    encoding = np.zeros((n, 20))
    for i, aa in enumerate(seq):
        if aa in BLOSUM62:
            for j, target_aa in enumerate(AMINO_ACIDS):
                # Normalize BLOSUM scores to [0, 1] range
                score = BLOSUM62[aa][target_aa]
                # BLOSUM62 scores range from -4 to 11, shift and scale
                encoding[i, j] = (score + 4) / 15.0

    return encoding


def combined_esm_like_encoding(sequence: str, dim: int = 64) -> np.ndarray:
    """
    Method 5: Combined ESM-like embedding (position-aware)
    Combines multiple features into a single embedding vector per protein
    """
    n = len(sequence)
    seq = sequence.upper()
    embedding = np.zeros(dim)

    # 1. Amino acid composition (20 dims)
    aa_counts = np.zeros(20)
    for aa in seq:
        if aa in AA_TO_IDX:
            aa_counts[AA_TO_IDX[aa]] += 1
    aa_freq = aa_counts / max(n, 1)
    embedding[:20] = aa_freq

    # 2. Di-peptide frequencies (20 dims, hashed)
    for i in range(n - 1):
        dipep = seq[i:i+2]
        if all(aa in AA_TO_IDX for aa in dipep):
            idx = 20 + (hash(dipep) % 20)
            embedding[idx] += 1.0 / max(n - 1, 1)

    # 3. Physicochemical summary (10 dims)
    props = property_matrix_encoding(seq)
    if len(props) > 0:
        # Global statistics
        embedding[40] = np.mean(props[:, 0])  # Mean hydrophobicity
        embedding[41] = np.std(props[:, 0])   # Std hydrophobicity
        embedding[42] = np.mean(props[:, 1])  # Mean volume
        embedding[43] = np.std(props[:, 1])   # Std volume
        embedding[44] = np.mean(props[:, 2])  # Mean charge
        embedding[45] = np.mean(props[:, 3])  # Polarity fraction
        embedding[46] = np.mean(props[:, 4])  # Helix propensity
        embedding[47] = np.mean(props[:, 5])  # Sheet propensity

        # N-terminal and C-terminal regions
        nterm = min(20, n)
        cterm = max(0, n - 20)
        embedding[48] = np.mean(props[:nterm, 0])  # N-term hydrophobicity
        embedding[49] = np.mean(props[cterm:, 0])  # C-term hydrophobicity

    # 4. Sequence complexity (Shannon entropy)
    if n > 0:
        probs = aa_freq[aa_freq > 0]
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            embedding[50] = entropy / np.log2(20)  # Normalized

    # 5. Structural propensity (4 dims)
    helix_formers = set("AELM")
    sheet_formers = set("VIY")
    disorder_prone = set("DEKRSPQGN")
    turn_formers = set("GPDN")

    embedding[51] = sum(1 for aa in seq if aa in helix_formers) / max(n, 1)
    embedding[52] = sum(1 for aa in seq if aa in sheet_formers) / max(n, 1)
    embedding[53] = sum(1 for aa in seq if aa in disorder_prone) / max(n, 1)
    embedding[54] = sum(1 for aa in seq if aa in turn_formers) / max(n, 1)

    # 6. Position-dependent hydrophobicity profile (10 dims)
    for i in range(55, min(65, dim)):
        pos = int((i - 55) * n / 10) if n > 0 else 0
        pos = min(pos, n - 1) if n > 0 else 0
        if n > 0:
            aa = seq[pos]
            if aa in HYDROPHOBICITY:
                embedding[i] = (HYDROPHOBICITY[aa] + 4.5) / 9.0

    return embedding


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from embeddings.

    Returns: (Df, alpha, Df_x_alpha)
    """
    n, d = embeddings.shape

    # Center the data
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    cov = np.dot(centered.T, centered) / max(n - 1, 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero

    if len(eigenvalues) < 2:
        return 1.0, 1.0, 1.0

    # Df: Participation ratio
    # PR = (sum(lambda))^2 / sum(lambda^2)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent
    # Fit power law: lambda_k ~ k^(-alpha)
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    # Linear regression for slope
    n_pts = len(log_k)
    if n_pts > 1:
        slope_num = n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda)
        slope_denom = n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2
        slope = slope_num / (slope_denom + 1e-10)
        alpha = -slope
    else:
        alpha = 1.0

    return float(Df), float(alpha), float(Df * alpha)


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics from embeddings.
    """
    n, d = embeddings.shape

    # Basic stats
    mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
    std_norm = np.std(np.linalg.norm(embeddings, axis=1))

    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(embeddings, 'euclidean')
    mean_dist = np.mean(dists) if len(dists) > 0 else 0
    std_dist = np.std(dists) if len(dists) > 0 else 0

    # Spectral properties
    Df, alpha, Df_x_alpha = compute_spectral_properties(embeddings)

    # Intrinsic dimensionality (MLE estimate)
    try:
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.dot(centered.T, centered) / max(n - 1, 1)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Estimate intrinsic dimension as number of eigenvalues capturing 95% variance
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        intrinsic_dim = np.searchsorted(cumsum, 0.95) + 1
    except:
        intrinsic_dim = d

    return {
        'n_samples': n,
        'n_features': d,
        'mean_norm': float(mean_norm),
        'std_norm': float(std_norm),
        'mean_dist': float(mean_dist),
        'std_dist': float(std_dist),
        'Df': Df,
        'alpha': alpha,
        'Df_x_alpha': Df_x_alpha,
        'intrinsic_dim': intrinsic_dim,
        'n_eigenvalues': len(eigenvalues) if 'eigenvalues' in dir() else 0
    }


# =============================================================================
# DATA LOADING
# =============================================================================

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


def load_all_sequences() -> Dict[str, str]:
    """Load all protein sequences from cached PDB files."""
    sequences = {}

    pdb_files = list(ALPHAFOLD_DIR.glob('*.pdb'))
    print(f"Found {len(pdb_files)} PDB files")

    for pdb_file in pdb_files:
        uniprot_id = pdb_file.stem
        seq = extract_sequence_from_pdb(pdb_file)
        if seq and len(seq) >= 50:  # Minimum length filter
            sequences[uniprot_id] = seq
            print(f"  {uniprot_id}: {len(seq)} residues")

    return sequences


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_method(method_name: str,
                   method_func: callable,
                   sequences: Dict[str, str],
                   aggregate: bool = True) -> Dict[str, Any]:
    """
    Analyze a single embedding method across all proteins.

    If aggregate=True, compute global embedding (protein-level)
    If aggregate=False, compute per-residue statistics
    """
    print(f"\n--- Analyzing: {method_name} ---")

    protein_embeddings = []
    protein_ids = []
    per_protein_stats = []

    for uid, seq in sequences.items():
        try:
            # Get per-residue encoding
            encoding = method_func(seq)

            if len(encoding.shape) == 2:
                # Per-residue encoding: aggregate to protein level
                if aggregate:
                    # Use mean and std across positions
                    mean_vec = np.mean(encoding, axis=0)
                    std_vec = np.std(encoding, axis=0)
                    protein_vec = np.concatenate([mean_vec, std_vec])
                else:
                    # Flatten (sample from positions for very long sequences)
                    if encoding.shape[0] > 500:
                        indices = np.linspace(0, encoding.shape[0]-1, 500, dtype=int)
                        encoding = encoding[indices]
                    protein_vec = encoding.flatten()
            else:
                # Already a 1D embedding
                protein_vec = encoding

            protein_embeddings.append(protein_vec)
            protein_ids.append(uid)

            # Per-protein Df x alpha (using per-residue encoding if 2D)
            if len(encoding.shape) == 2 and encoding.shape[0] >= 10:
                per_Df, per_alpha, per_Df_x_alpha = compute_spectral_properties(encoding)
                per_protein_stats.append({
                    'uniprot_id': uid,
                    'length': len(seq),
                    'Df': per_Df,
                    'alpha': per_alpha,
                    'Df_x_alpha': per_Df_x_alpha
                })

        except Exception as e:
            print(f"  Error processing {uid}: {e}")

    if len(protein_embeddings) < 5:
        return {'status': 'INSUFFICIENT_DATA', 'n_proteins': len(protein_embeddings)}

    # Convert to array
    # Ensure all embeddings have the same dimension
    max_dim = max(len(e) for e in protein_embeddings)
    padded_embeddings = []
    for e in protein_embeddings:
        if len(e) < max_dim:
            padded = np.zeros(max_dim)
            padded[:len(e)] = e
            padded_embeddings.append(padded)
        else:
            padded_embeddings.append(e)

    embeddings_array = np.array(padded_embeddings)

    # Global statistics
    stats = compute_embedding_statistics(embeddings_array)

    # Per-protein Df x alpha statistics
    if per_protein_stats:
        per_protein_Df_x_alpha = [p['Df_x_alpha'] for p in per_protein_stats]
        mean_per_protein = np.mean(per_protein_Df_x_alpha)
        std_per_protein = np.std(per_protein_Df_x_alpha)
        cv_per_protein = std_per_protein / mean_per_protein if mean_per_protein > 0 else float('inf')
    else:
        mean_per_protein = None
        std_per_protein = None
        cv_per_protein = None

    print(f"  Proteins: {len(protein_ids)}")
    print(f"  Embedding dim: {embeddings_array.shape[1]}")
    print(f"  Global Df x alpha: {stats['Df_x_alpha']:.4f}")
    if mean_per_protein is not None:
        print(f"  Per-protein Df x alpha: {mean_per_protein:.4f} +/- {std_per_protein:.4f} (CV: {cv_per_protein:.4f})")

    return {
        'method': method_name,
        'n_proteins': len(protein_ids),
        'embedding_dim': int(embeddings_array.shape[1]),
        'global_stats': stats,
        'per_protein_stats': per_protein_stats,
        'per_protein_summary': {
            'mean_Df_x_alpha': mean_per_protein,
            'std_Df_x_alpha': std_per_protein,
            'cv_Df_x_alpha': cv_per_protein
        } if mean_per_protein is not None else None,
        'protein_ids': protein_ids
    }


def run_biological_constant_test():
    """
    Main test function: Validate the Biological Semiotic Constant Hypothesis.
    """
    print("=" * 70)
    print("Q18: BIOLOGICAL SEMIOTIC CONSTANT HYPOTHESIS TEST")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nHypothesis: Bf = 2^4 x e = {TARGET_BF:.3f}")
    print(f"Alternative: 8e = {TARGET_8E:.3f}")
    print(f"Observed in ESM-2: {ESM2_OBSERVED_RANGE}")

    # Load sequences
    print("\n" + "=" * 70)
    print("LOADING PROTEIN SEQUENCES")
    print("=" * 70)
    sequences = load_all_sequences()
    print(f"\nLoaded {len(sequences)} proteins")

    if len(sequences) < 50:
        print(f"WARNING: Only {len(sequences)} proteins available (target: 50+)")

    # Define methods to test
    methods = [
        ("One-Hot Encoding", one_hot_encoding),
        ("3-mer Frequency", lambda s: kmer_frequency(s, k=3)),
        ("5-mer Frequency", lambda s: kmer_frequency(s, k=5)),
        ("Property Matrix", property_matrix_encoding),
        ("BLOSUM62 Embedding", blosum62_encoding),
        ("Combined ESM-like", combined_esm_like_encoding),
    ]

    # Run analysis for each method
    print("\n" + "=" * 70)
    print("ANALYZING EMBEDDING METHODS")
    print("=" * 70)

    results = {}
    all_global_Df_x_alpha = []
    all_per_protein_Df_x_alpha = []

    for method_name, method_func in methods:
        method_result = analyze_method(method_name, method_func, sequences)
        results[method_name] = method_result

        if method_result.get('global_stats'):
            all_global_Df_x_alpha.append(method_result['global_stats']['Df_x_alpha'])

        if method_result.get('per_protein_summary') and method_result['per_protein_summary'].get('mean_Df_x_alpha'):
            all_per_protein_Df_x_alpha.append(method_result['per_protein_summary']['mean_Df_x_alpha'])

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Df x alpha VALUES")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Global Df x alpha':>18} {'Per-Protein Mean':>18} {'CV':>10}")
    print("-" * 75)

    for method_name, method_result in results.items():
        global_val = method_result['global_stats']['Df_x_alpha'] if method_result.get('global_stats') else None
        per_protein = method_result.get('per_protein_summary', {})
        pp_mean = per_protein.get('mean_Df_x_alpha') if per_protein else None
        pp_cv = per_protein.get('cv_Df_x_alpha') if per_protein else None

        global_str = f"{global_val:.4f}" if global_val is not None else "N/A"
        pp_str = f"{pp_mean:.4f}" if pp_mean is not None else "N/A"
        cv_str = f"{pp_cv:.4f}" if pp_cv is not None else "N/A"

        print(f"{method_name:<25} {global_str:>18} {pp_str:>18} {cv_str:>10}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    if all_global_Df_x_alpha:
        mean_global = np.mean(all_global_Df_x_alpha)
        std_global = np.std(all_global_Df_x_alpha)
        cv_global = std_global / mean_global if mean_global > 0 else float('inf')

        print(f"\nGlobal Df x alpha (across methods):")
        print(f"  Mean: {mean_global:.4f}")
        print(f"  Std:  {std_global:.4f}")
        print(f"  CV:   {cv_global:.4f}")

    if all_per_protein_Df_x_alpha:
        mean_pp = np.mean(all_per_protein_Df_x_alpha)
        std_pp = np.std(all_per_protein_Df_x_alpha)
        cv_pp = std_pp / mean_pp if mean_pp > 0 else float('inf')

        print(f"\nPer-protein Df x alpha (averaged across methods):")
        print(f"  Mean: {mean_pp:.4f}")
        print(f"  Std:  {std_pp:.4f}")
        print(f"  CV:   {cv_pp:.4f}")

    # Comparison to hypotheses
    print("\n" + "=" * 70)
    print("HYPOTHESIS COMPARISON")
    print("=" * 70)

    best_estimate = mean_pp if all_per_protein_Df_x_alpha else mean_global
    cv_estimate = cv_pp if all_per_protein_Df_x_alpha else cv_global

    deviation_8e = abs(best_estimate - TARGET_8E) / TARGET_8E * 100
    deviation_Bf = abs(best_estimate - TARGET_BF) / TARGET_BF * 100
    deviation_ESM2_mid = abs(best_estimate - np.mean(ESM2_OBSERVED_RANGE)) / np.mean(ESM2_OBSERVED_RANGE) * 100

    print(f"\nBest estimate: {best_estimate:.4f}")
    print(f"CV: {cv_estimate:.4f}")
    print(f"\nDeviation from 8e ({TARGET_8E:.3f}): {deviation_8e:.2f}%")
    print(f"Deviation from Bf ({TARGET_BF:.3f}): {deviation_Bf:.2f}%")
    print(f"Deviation from ESM-2 mid ({np.mean(ESM2_OBSERVED_RANGE):.1f}): {deviation_ESM2_mid:.2f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    is_stable = cv_estimate < 0.15  # CV < 15% threshold
    closest_to_Bf = deviation_Bf < deviation_8e
    in_ESM2_range = ESM2_OBSERVED_RANGE[0] <= best_estimate <= ESM2_OBSERVED_RANGE[1]

    if is_stable and closest_to_Bf:
        verdict = "SUPPORTS HYPOTHESIS: Bf = 2^4 x e appears to be a stable biological constant"
    elif is_stable and in_ESM2_range:
        verdict = "PARTIALLY SUPPORTS: Value is stable but closer to ESM-2 observed range than Bf"
    elif is_stable:
        verdict = "STABLE BUT DIFFERENT: CV is low but value differs from predicted constants"
    else:
        verdict = "NO EVIDENCE: High variability (CV > 15%) suggests no stable constant"

    print(f"\n{verdict}")
    print(f"\nInterpretation:")
    print(f"  - Is CV < 15%? {'YES' if is_stable else 'NO'} (CV = {cv_estimate:.4f})")
    print(f"  - Closer to Bf than 8e? {'YES' if closest_to_Bf else 'NO'}")
    print(f"  - Within ESM-2 range? {'YES' if in_ESM2_range else 'NO'}")

    # Potential explanations
    print("\n" + "=" * 70)
    print("THEORETICAL INTERPRETATION")
    print("=" * 70)

    print("\nIf Bf = 2^4 x e holds:")
    print("  - 4 semiotic categories in biology (vs 3 in human cognition)")
    print("  - 4th category = 'Evolutionary Context' (fitness, selection pressure)")
    print("  - Explains why protein embeddings show higher Df x alpha")

    print("\nRelation to biological numbers:")
    print(f"  - 20 amino acids: 20/e = {20/E_CONST:.3f}")
    print(f"  - 64 codons: 64/20 = {64/20:.3f} redundancy")
    print(f"  - 4 nucleotide bases: 2^4 = 16 categories")
    print(f"  - Bf / 8e = {TARGET_BF / TARGET_8E:.3f} = 2 (ratio of categories)")

    # Save results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'hypothesis': {
            'description': 'Biological semiotic constant: Bf = 2^4 x e = 43.5',
            'target_Bf': TARGET_BF,
            'target_8e': TARGET_8E,
            'ESM2_observed_range': ESM2_OBSERVED_RANGE
        },
        'data': {
            'n_proteins': len(sequences),
            'methods_tested': list(results.keys())
        },
        'method_results': {k: to_serializable(v) for k, v in results.items()},
        'aggregate_statistics': {
            'global_Df_x_alpha': {
                'mean': float(mean_global) if all_global_Df_x_alpha else None,
                'std': float(std_global) if all_global_Df_x_alpha else None,
                'cv': float(cv_global) if all_global_Df_x_alpha else None
            },
            'per_protein_Df_x_alpha': {
                'mean': float(mean_pp) if all_per_protein_Df_x_alpha else None,
                'std': float(std_pp) if all_per_protein_Df_x_alpha else None,
                'cv': float(cv_pp) if all_per_protein_Df_x_alpha else None
            }
        },
        'hypothesis_test': {
            'best_estimate': float(best_estimate) if best_estimate else None,
            'cv': float(cv_estimate) if cv_estimate else None,
            'deviation_from_8e_percent': float(deviation_8e),
            'deviation_from_Bf_percent': float(deviation_Bf),
            'deviation_from_ESM2_mid_percent': float(deviation_ESM2_mid),
            'is_stable': bool(is_stable),
            'closest_to_Bf': bool(closest_to_Bf),
            'in_ESM2_range': bool(in_ESM2_range),
            'verdict': verdict
        }
    }

    output_file = RESULTS_DIR / 'biological_constant_results.json'
    with open(output_file, 'w') as f:
        json.dump(to_serializable(final_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return final_results


def to_serializable(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj
    return str(obj)


if __name__ == '__main__':
    run_biological_constant_test()
