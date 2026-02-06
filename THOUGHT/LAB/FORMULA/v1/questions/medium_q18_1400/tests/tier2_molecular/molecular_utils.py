#!/usr/bin/env python3
"""
Q18 Tier 2: Molecular Utilities

Shared utilities for molecular scale tests of R = E/sigma.

Author: Claude
Date: 2026-01-25
Version: 1.0.0
"""

import hashlib
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr


# =============================================================================
# CONSTANTS
# =============================================================================

# Amino acid properties (hydrophobicity, volume, charge, polarity)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Residue volumes (Angstrom^3)
VOLUMES = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
}

# Net charge at pH 7
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Polarity (1 = polar, 0 = nonpolar)
POLARITY = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

# BLOSUM62 substitution probabilities (simplified diagonal)
BLOSUM62_DIAG = {
    'A': 4, 'C': 9, 'D': 6, 'E': 5, 'F': 6,
    'G': 6, 'H': 8, 'I': 4, 'K': 5, 'L': 4,
    'M': 5, 'N': 6, 'P': 7, 'Q': 5, 'R': 5,
    'S': 4, 'T': 5, 'V': 4, 'W': 11, 'Y': 7
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ProteinFeatures:
    """Features extracted from a protein sequence."""
    sequence: str
    length: int
    hydrophobicity_mean: float
    hydrophobicity_std: float
    volume_mean: float
    volume_std: float
    charge_total: float
    polarity_fraction: float
    complexity: float
    embedding: np.ndarray


@dataclass
class MutationEffect:
    """Effect of a point mutation."""
    position: int
    wild_type: str
    mutant: str
    delta_fitness: float
    delta_r: float


# =============================================================================
# SEQUENCE ENCODING
# =============================================================================

def sequence_to_onehot(sequence: str) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    onehot = np.zeros((len(sequence), 20))
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            onehot[i, aa_to_idx[aa]] = 1.0
    return onehot


def sequence_to_physicochemical(sequence: str) -> np.ndarray:
    """Convert sequence to physicochemical property vectors."""
    n = len(sequence)
    features = np.zeros((n, 4))

    for i, aa in enumerate(sequence):
        if aa in HYDROPHOBICITY:
            features[i, 0] = HYDROPHOBICITY[aa] / 4.5  # Normalize
            features[i, 1] = VOLUMES[aa] / 227.8
            features[i, 2] = CHARGE[aa]
            features[i, 3] = POLARITY[aa]

    return features


def compute_sequence_embedding(sequence: str, dim: int = 64) -> np.ndarray:
    """
    Compute a simple ESM-like embedding for a protein sequence.
    Uses k-mer frequencies and physicochemical properties.
    """
    n = len(sequence)
    embedding = np.zeros(dim)

    # 1. Composition (20 dims)
    aa_counts = np.zeros(20)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for aa in sequence:
        if aa in aa_to_idx:
            aa_counts[aa_to_idx[aa]] += 1
    aa_freq = aa_counts / max(n, 1)
    embedding[:20] = aa_freq

    # 2. Di-peptide frequencies (top 20 by variance)
    dipep_counts = {}
    for i in range(n - 1):
        dipep = sequence[i:i+2]
        dipep_counts[dipep] = dipep_counts.get(dipep, 0) + 1

    # Use hash to map to fixed positions
    for dipep, count in dipep_counts.items():
        idx = 20 + (hash(dipep) % 20)
        embedding[idx] += count / max(n - 1, 1)

    # 3. Physicochemical profile (20 dims)
    phys = sequence_to_physicochemical(sequence)
    if len(phys) > 0:
        # Global stats
        embedding[40] = np.mean(phys[:, 0])  # Mean hydrophobicity
        embedding[41] = np.std(phys[:, 0])   # Std hydrophobicity
        embedding[42] = np.mean(phys[:, 1])  # Mean volume
        embedding[43] = np.std(phys[:, 1])   # Std volume
        embedding[44] = np.sum(phys[:, 2])   # Total charge
        embedding[45] = np.mean(phys[:, 3])  # Polarity fraction

        # Windowed features (hydrophobicity profile)
        window = 5
        for i in range(min(10, n // window)):
            start = i * window
            end = min(start + window, n)
            embedding[46 + i] = np.mean(phys[start:end, 0])

    # 4. Sequence complexity (Shannon entropy)
    if n > 0:
        probs = aa_freq[aa_freq > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        embedding[56] = entropy / np.log2(20)  # Normalized

    # 5. Secondary structure propensity (simplified)
    helix_formers = set("AELM")
    sheet_formers = set("VIY")
    helix_prop = sum(1 for aa in sequence if aa in helix_formers) / max(n, 1)
    sheet_prop = sum(1 for aa in sequence if aa in sheet_formers) / max(n, 1)
    embedding[57] = helix_prop
    embedding[58] = sheet_prop

    # 6. Disorder propensity
    disorder_prone = set("DEKRSPQGN")
    disorder_prop = sum(1 for aa in sequence if aa in disorder_prone) / max(n, 1)
    embedding[59] = disorder_prop

    # Fill remaining with position-dependent features
    for i in range(60, dim):
        if n > 0:
            pos = int((i - 60) * n / (dim - 60))
            pos = min(pos, n - 1)
            aa = sequence[pos]
            if aa in HYDROPHOBICITY:
                embedding[i] = HYDROPHOBICITY[aa] / 4.5

    return embedding


def extract_protein_features(sequence: str) -> ProteinFeatures:
    """Extract comprehensive features from a protein sequence."""
    n = len(sequence)

    # Physicochemical
    hydro = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    vol = [VOLUMES.get(aa, 100) for aa in sequence]
    charge = sum(CHARGE.get(aa, 0) for aa in sequence)
    polar = sum(POLARITY.get(aa, 0) for aa in sequence) / max(n, 1)

    # Complexity (Shannon entropy of composition)
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    probs = np.array(list(aa_counts.values())) / n
    complexity = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(20)

    # Embedding
    embedding = compute_sequence_embedding(sequence)

    return ProteinFeatures(
        sequence=sequence,
        length=n,
        hydrophobicity_mean=np.mean(hydro) if hydro else 0,
        hydrophobicity_std=np.std(hydro) if hydro else 0,
        volume_mean=np.mean(vol) if vol else 0,
        volume_std=np.std(vol) if vol else 0,
        charge_total=charge,
        polarity_fraction=polar,
        complexity=complexity,
        embedding=embedding
    )


# =============================================================================
# R COMPUTATION FOR MOLECULAR DATA
# =============================================================================

def compute_R_molecular(embeddings: np.ndarray, k: int = 5) -> Tuple[float, float, float]:
    """
    Compute R = E/sigma for molecular embeddings.

    E = agreement metric (how consistent are similar sequences)
    sigma = variance metric (spread in embedding space)

    Returns: (R, E, sigma)
    """
    n = embeddings.shape[0]
    if n < 3:
        return 0.0, 0.0, 1.0

    # Compute pairwise distances
    dists = squareform(pdist(embeddings, 'euclidean'))

    # E: Agreement among k-nearest neighbors
    # High E means similar sequences cluster well
    agreements = []
    for i in range(n):
        neighbors = np.argsort(dists[i])[1:k+1]
        if len(neighbors) > 0:
            # Agreement = inverse of average distance to neighbors
            avg_dist = np.mean(dists[i, neighbors])
            agreements.append(1.0 / (1.0 + avg_dist))

    E = np.mean(agreements) if agreements else 0.0

    # sigma: Global variance in embedding space
    # Use standard deviation of all pairwise distances
    all_dists = dists[np.triu_indices(n, k=1)]
    sigma = np.std(all_dists) if len(all_dists) > 0 else 1.0
    sigma = max(sigma, 1e-10)  # Prevent division by zero

    R = E / sigma

    return R, E, sigma


def compute_R_for_mutation(wild_embedding: np.ndarray,
                           mutant_embedding: np.ndarray,
                           reference_embeddings: np.ndarray,
                           k: int = 5) -> Tuple[float, float]:
    """
    Compute delta-R for a mutation.

    Returns: (R_wild, R_mutant)
    """
    # Add wild-type to references
    wt_refs = np.vstack([reference_embeddings, wild_embedding.reshape(1, -1)])
    mut_refs = np.vstack([reference_embeddings, mutant_embedding.reshape(1, -1)])

    R_wild, _, _ = compute_R_molecular(wt_refs, k=k)
    R_mutant, _, _ = compute_R_molecular(mut_refs, k=k)

    return R_wild, R_mutant


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_protein_sequence(length: int,
                              disorder_fraction: float = 0.0,
                              seed: Optional[int] = None) -> str:
    """Generate a synthetic protein sequence."""
    if seed is not None:
        np.random.seed(seed)

    # Base amino acid distribution (roughly natural)
    aa_probs = np.array([
        0.08, 0.02, 0.05, 0.06, 0.04,  # ACDEF
        0.07, 0.02, 0.05, 0.06, 0.10,  # GHIKL
        0.02, 0.04, 0.05, 0.04, 0.05,  # MNPQR
        0.07, 0.05, 0.07, 0.01, 0.03   # STVWY
    ])
    aa_probs = aa_probs / aa_probs.sum()

    # Adjust for disorder
    if disorder_fraction > 0:
        disorder_aa = [AMINO_ACIDS.index(aa) for aa in "DEKRSPQGN" if aa in AMINO_ACIDS]
        for idx in disorder_aa:
            aa_probs[idx] *= (1 + 2 * disorder_fraction)
        aa_probs = aa_probs / aa_probs.sum()

    indices = np.random.choice(20, size=length, p=aa_probs)
    return ''.join(AMINO_ACIDS[i] for i in indices)


def generate_protein_family(n_proteins: int,
                            template_length: int = 150,
                            mutation_rate: float = 0.1,
                            seed: Optional[int] = None) -> List[str]:
    """Generate a family of related protein sequences."""
    if seed is not None:
        np.random.seed(seed)

    # Generate template
    template = generate_protein_sequence(template_length, seed=seed)

    sequences = [template]
    for i in range(n_proteins - 1):
        # Create variant
        variant = list(template)
        n_mutations = int(len(template) * mutation_rate)
        positions = np.random.choice(len(template), size=n_mutations, replace=False)
        for pos in positions:
            variant[pos] = AMINO_ACIDS[np.random.randint(20)]
        sequences.append(''.join(variant))

    return sequences


def generate_dms_benchmark(sequence: str,
                           n_mutations: int = 100,
                           seed: Optional[int] = None) -> List[MutationEffect]:
    """
    Generate synthetic Deep Mutational Scanning data.

    Simulates fitness effects based on:
    1. Conservation (BLOSUM scores)
    2. Physicochemical disruption
    3. Position (active site residues)
    """
    if seed is not None:
        np.random.seed(seed)

    mutations = []
    n = len(sequence)

    # Simulate some "active site" positions
    active_sites = set(np.random.choice(n, size=max(1, n // 10), replace=False))

    for _ in range(n_mutations):
        pos = np.random.randint(n)
        wt_aa = sequence[pos]

        # Select a different amino acid
        possible = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        mut_aa = possible[np.random.randint(len(possible))]

        # Compute fitness effect
        # 1. BLOSUM penalty
        blosum_wt = BLOSUM62_DIAG.get(wt_aa, 4)
        blosum_mut = BLOSUM62_DIAG.get(mut_aa, 4)
        blosum_penalty = (blosum_wt - blosum_mut) / 10.0

        # 2. Physicochemical disruption
        hydro_change = abs(HYDROPHOBICITY.get(wt_aa, 0) - HYDROPHOBICITY.get(mut_aa, 0))
        vol_change = abs(VOLUMES.get(wt_aa, 100) - VOLUMES.get(mut_aa, 100))
        charge_change = abs(CHARGE.get(wt_aa, 0) - CHARGE.get(mut_aa, 0))

        phys_penalty = (hydro_change / 9.0 + vol_change / 170.0 + charge_change) / 3.0

        # 3. Active site penalty
        active_penalty = 0.5 if pos in active_sites else 0.0

        # Total fitness effect (negative = deleterious)
        delta_fitness = -(blosum_penalty + phys_penalty + active_penalty +
                          np.random.randn() * 0.1)
        delta_fitness = np.clip(delta_fitness, -2.0, 1.0)

        mutations.append(MutationEffect(
            position=pos,
            wild_type=wt_aa,
            mutant=mut_aa,
            delta_fitness=delta_fitness,
            delta_r=0.0  # Will be computed later
        ))

    return mutations


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from embeddings.

    Returns: (Df, alpha)
    """
    n, d = embeddings.shape

    # Compute covariance matrix
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.dot(centered.T, centered) / max(n - 1, 1)

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0

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
        slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
        slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
        alpha = -slope
    else:
        alpha = 1.0

    return Df, alpha


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_data_hash(data: Any) -> str:
    """Compute SHA256 hash of data."""
    if isinstance(data, np.ndarray):
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]
    elif isinstance(data, list):
        combined = '|'.join(str(x) for x in data)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    else:
        return hashlib.sha256(str(data).encode()).hexdigest()[:16]


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
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
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
