#!/usr/bin/env python3
"""
Q18 Investigation: R-Frequency Modulation Universality Test

CRITICAL HYPOTHESIS:
R as a frequency modulator produces 8e (~21.75):
    embedding[i, d] = sin(d * R[i] / scale) + noise_scale / (R[i] + epsilon) * random

This was discovered on gene expression data. Does it generalize?

TEST DOMAINS:
1. Protein sequences (amino acid properties as R)
2. DMS mutation data (delta-R from property changes)
3. Stock market simulation (Sharpe ratio as R)
4. Random matrices (control - various distributions)

For each domain, we:
- Compute R using domain-appropriate E and sigma
- Create 50D R-modulated sinusoidal embedding
- Compute Df x alpha
- Compare to 8e (21.746)

KEY QUESTION: Is R-frequency modulation a UNIVERSAL method to produce 8e?

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
TOLERANCE = 0.15  # 15% deviation threshold
CACHE_DIR = Path(__file__).parent / 'cache'


@dataclass
class UniversalityResult:
    """Results from testing R-frequency on a specific domain."""
    domain: str
    description: str
    r_formula: str
    n_samples: int
    embedding_dims: int
    R_mean: float
    R_std: float
    R_min: float
    R_max: float
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    deviation_percent: float
    passes_8e: bool
    top_eigenvalues: List[float]
    key_insight: str = ""


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


def compute_spectral_properties(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute Df (participation ratio) and alpha (spectral decay) from embeddings.
    Returns: (Df, alpha, eigenvalues)
    """
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio = (sum(lambda))^2 / sum(lambda^2)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay exponent (power law fit: lambda_k ~ k^(-alpha))
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def create_r_frequency_embedding(R_values: np.ndarray, n_dims: int = 50,
                                   scale: float = 10.0, noise_scale: float = 1.0,
                                   epsilon: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Create R-frequency modulated sinusoidal embedding.

    Formula: embedding[i, d] = sin(d * R[i] / scale) + noise_scale / (R[i] + epsilon) * random

    Args:
        R_values: Array of R values for each sample
        n_dims: Embedding dimensionality
        scale: Frequency scale factor
        noise_scale: Scale of R-modulated noise
        epsilon: Small constant to prevent division by zero
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    n_samples = len(R_values)
    embeddings = np.zeros((n_samples, n_dims))

    for i, r in enumerate(R_values):
        np.random.seed(i + seed)

        # Base sinusoidal position (R modulates frequency)
        dims = np.arange(n_dims)
        base_pos = np.sin(dims * r / scale)

        # R-modulated noise (high R = less noise, low R = more noise)
        noise_amplitude = noise_scale / (r + epsilon)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        noise = noise_amplitude * direction

        embeddings[i] = base_pos + noise

    return embeddings


def make_result(domain: str, description: str, r_formula: str,
                R_values: np.ndarray, embeddings: np.ndarray,
                key_insight: str = "") -> UniversalityResult:
    """Create UniversalityResult from R values and embeddings."""
    Df, alpha, eigenvalues = compute_spectral_properties(embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    return UniversalityResult(
        domain=domain,
        description=description,
        r_formula=r_formula,
        n_samples=len(R_values),
        embedding_dims=embeddings.shape[1],
        R_mean=float(np.mean(R_values)),
        R_std=float(np.std(R_values)),
        R_min=float(np.min(R_values)),
        R_max=float(np.max(R_values)),
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        deviation_percent=deviation * 100,
        passes_8e=deviation < TOLERANCE,
        top_eigenvalues=eigenvalues[:10].tolist(),
        key_insight=key_insight
    )


# =============================================================================
# DOMAIN 1: PROTEIN SEQUENCES
# =============================================================================

# Amino acid property scales
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

AA_VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

AA_CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}


def compute_protein_r_values(sequence: str, window_size: int = 10) -> np.ndarray:
    """
    Compute R values for a protein using sliding windows over amino acid properties.

    R = mean(property) / std(property) for each window

    Uses combination of hydrophobicity, volume, and charge.
    """
    R_values = []

    for start in range(len(sequence) - window_size + 1):
        window = sequence[start:start + window_size]

        # Extract properties for window
        hydro = [HYDROPHOBICITY.get(aa, 0) for aa in window]
        volume = [AA_VOLUME.get(aa, 100) / 100 for aa in window]  # Normalize
        charge = [AA_CHARGE.get(aa, 0) for aa in window]

        # Combined property (normalized)
        properties = np.array(hydro) + np.array(volume) + np.array(charge) * 2

        # R = mean / std (with epsilon for stability)
        E = np.mean(properties)
        sigma = np.std(properties) + 0.1
        R = abs(E) / sigma  # Use absolute to keep R positive

        R_values.append(R)

    return np.array(R_values)


def test_protein_sequences(n_dims: int = 50, seed: int = 42) -> List[UniversalityResult]:
    """Test R-frequency on protein sequence windows."""
    results = []

    # Load protein data
    plddt_path = CACHE_DIR / 'extended_plddt.json'

    if plddt_path.exists():
        with open(plddt_path, 'r') as f:
            proteins = json.load(f)

        # Collect R values from all proteins
        all_R_values = []
        for prot_id, prot_data in list(proteins.items())[:50]:  # Use first 50 proteins
            sequence = prot_data.get('sequence', '')
            if len(sequence) > 20:
                R_values = compute_protein_r_values(sequence)
                all_R_values.extend(R_values)

        if len(all_R_values) > 100:
            R_values = np.array(all_R_values[:5000])  # Cap at 5000 samples

            # Test standard R-frequency embedding
            embeddings = create_r_frequency_embedding(R_values, n_dims=n_dims, seed=seed)

            result = make_result(
                domain="protein_sequences",
                description="Protein sliding windows with R from amino acid properties",
                r_formula="R = |mean(hydro + volume + 2*charge)| / std",
                R_values=R_values,
                embeddings=embeddings,
                key_insight="Testing if amino acid property-based R produces 8e"
            )
            results.append(result)

    # Also test simulated protein-like data
    np.random.seed(seed)
    n_samples = 2000

    # Simulate R from log-normal (common for biological properties)
    R_lognormal = np.random.lognormal(mean=1.0, sigma=0.5, size=n_samples)
    embeddings_ln = create_r_frequency_embedding(R_lognormal, n_dims=n_dims, seed=seed)

    results.append(make_result(
        domain="protein_simulated_lognormal",
        description="Simulated protein-like R (log-normal distribution)",
        r_formula="R ~ LogNormal(1.0, 0.5)",
        R_values=R_lognormal,
        embeddings=embeddings_ln,
        key_insight="Log-normal R simulates biological property distributions"
    ))

    return results


# =============================================================================
# DOMAIN 2: DMS MUTATION DATA
# =============================================================================

def compute_mutation_delta_r(wt_aa: str, mut_aa: str) -> float:
    """
    Compute delta-R for a mutation based on amino acid property changes.

    Higher |delta-R| = more disruptive mutation
    """
    h_wt = HYDROPHOBICITY.get(wt_aa, 0)
    h_mut = HYDROPHOBICITY.get(mut_aa, 0)
    v_wt = AA_VOLUME.get(wt_aa, 100)
    v_mut = AA_VOLUME.get(mut_aa, 100)
    c_wt = AA_CHARGE.get(wt_aa, 0)
    c_mut = AA_CHARGE.get(mut_aa, 0)

    # Compute changes
    delta_h = abs(h_mut - h_wt) / 9.0  # Max hydro change ~9
    delta_v = abs(v_mut - v_wt) / 170  # Max volume change ~170
    delta_c = abs(c_mut - c_wt) / 2.0  # Max charge change ~2

    # Disruption score (0 to 1 range)
    disruption = 0.4 * delta_h + 0.3 * delta_v + 0.3 * delta_c

    # R for wildtype = 1.0 (reference)
    # R for mutant decreases with disruption
    E_mut = max(0.1, 1.0 - disruption)
    sigma_mut = 1.0 + 0.5 * disruption
    R_mut = E_mut / sigma_mut

    return R_mut


def test_dms_mutation_data(n_dims: int = 50, seed: int = 42) -> List[UniversalityResult]:
    """Test R-frequency on DMS mutation data."""
    results = []

    proteins = ['dms_data.json', 'dms_data_ube2i.json', 'dms_data_tp53.json']
    protein_names = ['BRCA1', 'UBE2I', 'TP53']

    all_R_values = []
    all_fitness = []

    for prot_file, prot_name in zip(proteins, protein_names):
        filepath = CACHE_DIR / prot_file
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        mutations = data['mutations']

        for mut in mutations:
            wt = mut['wt']
            mt = mut['mut']
            fitness = mut['fitness']

            if wt in HYDROPHOBICITY and mt in HYDROPHOBICITY and mt != '*':
                R_mut = compute_mutation_delta_r(wt, mt)
                all_R_values.append(R_mut)
                all_fitness.append(fitness)

    if len(all_R_values) > 100:
        R_values = np.array(all_R_values)
        fitness_values = np.array(all_fitness)

        # Create R-frequency embedding
        embeddings = create_r_frequency_embedding(R_values, n_dims=n_dims, seed=seed)

        # Check correlation between R and fitness
        rho, pval = stats.spearmanr(R_values, fitness_values)

        result = make_result(
            domain="dms_mutations",
            description=f"DMS mutations from BRCA1, UBE2I, TP53 (n={len(R_values)})",
            r_formula="R = E_mut / sigma_mut based on AA property changes",
            R_values=R_values,
            embeddings=embeddings,
            key_insight=f"R-fitness correlation: rho={rho:.3f}, p={pval:.2e}"
        )
        results.append(result)

    return results


# =============================================================================
# DOMAIN 3: STOCK MARKET SIMULATION
# =============================================================================

def simulate_stock_market_data(n_stocks: int = 1000, n_days: int = 252,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate stock market returns and compute Sharpe-like R values.

    R = mean_return / volatility (analogous to Sharpe ratio)
    """
    np.random.seed(seed)

    # Simulate returns from a mixture model (realistic market behavior)
    returns = np.zeros((n_stocks, n_days))

    for i in range(n_stocks):
        # Each stock has its own drift and volatility
        drift = np.random.uniform(-0.0005, 0.001)  # Annual drift -12% to +25%
        vol = np.random.uniform(0.01, 0.05)  # Daily vol 1-5%

        # Add regime switches (market crashes)
        crash_days = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)

        daily_returns = drift + vol * np.random.randn(n_days)
        daily_returns[crash_days] *= -3  # Crashes are 3x normal volatility, negative

        returns[i] = daily_returns

    # Compute R = mean_return / volatility for each stock
    mean_returns = np.mean(returns, axis=1)
    volatilities = np.std(returns, axis=1) + 1e-6
    R_values = mean_returns / volatilities

    # Shift to positive values (R should be > 0 for our embedding)
    R_values = R_values - R_values.min() + 0.1

    return R_values, returns


def test_stock_market(n_dims: int = 50, seed: int = 42) -> List[UniversalityResult]:
    """Test R-frequency on simulated stock market data."""
    results = []

    # Standard market simulation
    R_values, returns = simulate_stock_market_data(n_stocks=2000, seed=seed)
    embeddings = create_r_frequency_embedding(R_values, n_dims=n_dims, seed=seed)

    results.append(make_result(
        domain="stock_market",
        description="Simulated stock Sharpe ratios (2000 stocks, 252 days)",
        r_formula="R = mean_return / volatility (shifted positive)",
        R_values=R_values,
        embeddings=embeddings,
        key_insight="Testing if financial signal-to-noise produces 8e"
    ))

    # High volatility market
    np.random.seed(seed + 100)
    R_high_vol, _ = simulate_stock_market_data(n_stocks=2000, seed=seed+100)
    R_high_vol = R_high_vol * 0.5  # Compress R range (higher relative noise)
    embeddings_hv = create_r_frequency_embedding(R_high_vol, n_dims=n_dims, seed=seed)

    results.append(make_result(
        domain="stock_market_high_vol",
        description="High volatility market simulation (compressed R)",
        r_formula="R = mean_return / volatility * 0.5",
        R_values=R_high_vol,
        embeddings=embeddings_hv,
        key_insight="Testing R-frequency with reduced signal-to-noise"
    ))

    return results


# =============================================================================
# DOMAIN 4: RANDOM MATRICES (CONTROL)
# =============================================================================

def test_random_matrices(n_dims: int = 50, n_samples: int = 2000, seed: int = 42) -> List[UniversalityResult]:
    """
    Test R-frequency on random R values from various distributions.

    This is a CONTROL to understand when 8e emerges vs when it doesn't.
    """
    results = []
    np.random.seed(seed)

    distributions = [
        ("uniform", lambda: np.random.uniform(0.1, 10, n_samples)),
        ("normal_positive", lambda: np.abs(np.random.randn(n_samples)) + 0.1),
        ("exponential", lambda: np.random.exponential(2.0, n_samples) + 0.1),
        ("lognormal", lambda: np.random.lognormal(1.0, 0.5, n_samples)),
        ("power_law", lambda: (np.random.pareto(2.0, n_samples) + 1) * 0.5),
        ("bimodal", lambda: np.concatenate([
            np.random.normal(2, 0.5, n_samples // 2),
            np.random.normal(6, 0.5, n_samples // 2)
        ])),
        ("constant", lambda: np.ones(n_samples) + np.random.randn(n_samples) * 0.01),
        ("zipf_like", lambda: 1.0 / (np.arange(1, n_samples + 1) ** 0.5) + np.random.randn(n_samples) * 0.01 + 0.1),
    ]

    for dist_name, dist_func in distributions:
        np.random.seed(seed)
        R_values = dist_func()
        R_values = np.abs(R_values) + 0.01  # Ensure positive

        embeddings = create_r_frequency_embedding(R_values, n_dims=n_dims, seed=seed)

        result = make_result(
            domain=f"random_{dist_name}",
            description=f"Random R values from {dist_name} distribution",
            r_formula=f"R ~ {dist_name}",
            R_values=R_values,
            embeddings=embeddings,
            key_insight=f"Control: testing if {dist_name} R produces 8e"
        )
        results.append(result)

    return results


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

def test_robustness(n_samples: int = 2000, n_dims: int = 50, seed: int = 42) -> List[UniversalityResult]:
    """
    Test robustness of R-frequency embedding to parameter changes.

    Tests:
    1. Different scale factors
    2. Different noise levels
    3. Different R formulas
    4. Different embedding dimensions
    """
    results = []
    np.random.seed(seed)

    # Base R values (log-normal, like gene expression data)
    R_base = np.random.lognormal(1.0, 0.5, n_samples)

    # Test 1: Scale factors
    for scale in [1.0, 5.0, 10.0, 20.0, 50.0]:
        embeddings = create_r_frequency_embedding(R_base, n_dims=n_dims, scale=scale, seed=seed)
        results.append(make_result(
            domain=f"robustness_scale_{scale}",
            description=f"R-frequency with scale={scale}",
            r_formula=f"sin(d * R / {scale}) + noise/R",
            R_values=R_base,
            embeddings=embeddings,
            key_insight=f"Testing scale factor = {scale}"
        ))

    # Test 2: Noise levels
    for noise_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        embeddings = create_r_frequency_embedding(R_base, n_dims=n_dims, noise_scale=noise_scale, seed=seed)
        results.append(make_result(
            domain=f"robustness_noise_{noise_scale}",
            description=f"R-frequency with noise_scale={noise_scale}",
            r_formula=f"sin(d * R / 10) + {noise_scale}/R * random",
            R_values=R_base,
            embeddings=embeddings,
            key_insight=f"Testing noise scale = {noise_scale}"
        ))

    # Test 3: Different R formulas (transformed R)
    r_transforms = [
        ("log_R", np.log(R_base + 0.1)),
        ("sqrt_R", np.sqrt(R_base)),
        ("R_squared", R_base ** 2 / R_base.max()),
        ("inverse_R", 1.0 / (R_base + 0.1)),
        ("tanh_R", np.tanh(R_base / R_base.mean()) * R_base.mean()),
    ]

    for name, R_transformed in r_transforms:
        R_transformed = np.abs(R_transformed) + 0.01  # Ensure positive
        embeddings = create_r_frequency_embedding(R_transformed, n_dims=n_dims, seed=seed)
        results.append(make_result(
            domain=f"robustness_transform_{name}",
            description=f"R-frequency with {name} transformation",
            r_formula=f"sin(d * {name} / 10) + noise/{name}",
            R_values=R_transformed,
            embeddings=embeddings,
            key_insight=f"Testing R transformation: {name}"
        ))

    # Test 4: Different embedding dimensions
    for dims in [10, 25, 50, 100, 200]:
        embeddings = create_r_frequency_embedding(R_base, n_dims=dims, seed=seed)
        results.append(make_result(
            domain=f"robustness_dims_{dims}",
            description=f"R-frequency with {dims}D embedding",
            r_formula=f"sin(d * R / 10) + noise/R, d=0..{dims-1}",
            R_values=R_base,
            embeddings=embeddings,
            key_insight=f"Testing embedding dimensionality = {dims}"
        ))

    return results


# =============================================================================
# GENE EXPRESSION REFERENCE (from original discovery)
# =============================================================================

def test_gene_expression_reference(n_dims: int = 50, seed: int = 42) -> List[UniversalityResult]:
    """Test the original gene expression data that produced 8e."""
    results = []

    filepath = CACHE_DIR / 'gene_expression_sample.json'
    if not filepath.exists():
        return results

    with open(filepath, 'r') as f:
        data = json.load(f)

    genes_data = data['genes']
    R_values = np.array([g['R'] for g in genes_data.values()])

    # Standard R-frequency embedding (the original method)
    embeddings = create_r_frequency_embedding(R_values, n_dims=n_dims, seed=seed)

    results.append(make_result(
        domain="gene_expression_reference",
        description="Original gene expression R values (reference for 8e)",
        r_formula="R = mean_expression / std_expression",
        R_values=R_values,
        embeddings=embeddings,
        key_insight="REFERENCE: This is the original domain where 8e was discovered"
    ))

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_universality_tests(verbose: bool = True) -> Dict[str, Any]:
    """Run all R-frequency universality tests."""

    if verbose:
        print("=" * 80)
        print("Q18 INVESTIGATION: R-FREQUENCY MODULATION UNIVERSALITY")
        print("Testing if R as frequency modulator produces 8e across domains")
        print("=" * 80)
        print(f"\nTheoretical 8e: {EIGHT_E:.4f}")
        print(f"Tolerance: {TOLERANCE*100:.0f}%")
        print()

    all_results = []

    # Domain 1: Gene Expression (reference)
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 0: GENE EXPRESSION (REFERENCE)")
        print("=" * 60)
    gene_results = test_gene_expression_reference()
    all_results.extend(gene_results)
    if verbose:
        for r in gene_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Domain 2: Protein Sequences
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 1: PROTEIN SEQUENCES")
        print("=" * 60)
    protein_results = test_protein_sequences()
    all_results.extend(protein_results)
    if verbose:
        for r in protein_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Domain 3: DMS Mutations
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 2: DMS MUTATION DATA")
        print("=" * 60)
    dms_results = test_dms_mutation_data()
    all_results.extend(dms_results)
    if verbose:
        for r in dms_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Domain 4: Stock Market
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 3: STOCK MARKET SIMULATION")
        print("=" * 60)
    stock_results = test_stock_market()
    all_results.extend(stock_results)
    if verbose:
        for r in stock_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Domain 5: Random Matrices
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 4: RANDOM MATRICES (CONTROL)")
        print("=" * 60)
    random_results = test_random_matrices()
    all_results.extend(random_results)
    if verbose:
        for r in random_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Domain 6: Robustness Tests
    if verbose:
        print("\n" + "=" * 60)
        print("DOMAIN 5: ROBUSTNESS TESTS")
        print("=" * 60)
    robustness_results = test_robustness()
    all_results.extend(robustness_results)
    if verbose:
        for r in robustness_results:
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {r.domain}: Df x alpha = {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Summary Analysis
    passing = [r for r in all_results if r.passes_8e]
    failing = [r for r in all_results if not r.passes_8e]

    # Group by domain category
    domain_groups = {}
    for r in all_results:
        category = r.domain.split('_')[0]
        if category not in domain_groups:
            domain_groups[category] = []
        domain_groups[category].append(r)

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: R-FREQUENCY UNIVERSALITY")
        print("=" * 80)
        print(f"\nTotal experiments: {len(all_results)}")
        print(f"Passing 8e (<15% dev): {len(passing)}")
        print(f"Failing 8e: {len(failing)}")
        print(f"\nPass rate: {100 * len(passing) / len(all_results):.1f}%")

        print("\n" + "-" * 60)
        print("BY DOMAIN:")
        print("-" * 60)
        for category, group_results in domain_groups.items():
            n_pass = sum(1 for r in group_results if r.passes_8e)
            print(f"  {category}: {n_pass}/{len(group_results)} pass 8e")

        print("\n" + "-" * 60)
        print("TOP 10 BY 8e COMPLIANCE:")
        print("-" * 60)
        sorted_results = sorted(all_results, key=lambda x: x.deviation_from_8e)
        for i, r in enumerate(sorted_results[:10], 1):
            status = "PASS" if r.passes_8e else "FAIL"
            print(f"  {i}. {r.domain}: {r.Df_x_alpha:.2f} ({r.deviation_percent:.1f}%) [{status}]")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "theoretical_8e": float(EIGHT_E),
        "tolerance": TOLERANCE,
        "hypothesis": "R-frequency modulation produces 8e universally across domains",
        "results": [],
        "summary": {},
        "key_findings": []
    }

    for r in all_results:
        output["results"].append({
            "domain": r.domain,
            "description": r.description,
            "r_formula": r.r_formula,
            "n_samples": r.n_samples,
            "embedding_dims": r.embedding_dims,
            "R_statistics": {
                "mean": r.R_mean,
                "std": r.R_std,
                "min": r.R_min,
                "max": r.R_max
            },
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_percent,
            "passes_8e": r.passes_8e,
            "top_eigenvalues": r.top_eigenvalues,
            "key_insight": r.key_insight
        })

    output["summary"] = {
        "total_experiments": len(all_results),
        "passing_8e": len(passing),
        "failing_8e": len(failing),
        "pass_rate_percent": 100 * len(passing) / len(all_results),
        "passing_domains": list(set(r.domain.split('_')[0] for r in passing)),
        "best_result": {
            "domain": sorted_results[0].domain,
            "Df_x_alpha": sorted_results[0].Df_x_alpha,
            "deviation_percent": sorted_results[0].deviation_percent
        } if sorted_results else None,
        "domain_pass_rates": {
            cat: sum(1 for r in grp if r.passes_8e) / len(grp)
            for cat, grp in domain_groups.items()
        }
    }

    # Key findings
    key_findings = []

    # Check if gene expression reference passes
    gene_ref = next((r for r in all_results if r.domain == "gene_expression_reference"), None)
    if gene_ref:
        if gene_ref.passes_8e:
            key_findings.append(f"REFERENCE VALIDATES: Gene expression R-frequency produces 8e ({gene_ref.deviation_percent:.1f}% dev)")
        else:
            key_findings.append(f"WARNING: Reference gene expression did NOT produce 8e ({gene_ref.deviation_percent:.1f}% dev)")

    # Check biological domains
    bio_domains = [r for r in all_results if r.domain.startswith(('protein', 'dms'))]
    bio_pass = sum(1 for r in bio_domains if r.passes_8e)
    if bio_domains:
        if bio_pass == len(bio_domains):
            key_findings.append("BIOLOGICAL UNIVERSALITY: R-frequency produces 8e in ALL biological domains!")
        elif bio_pass > len(bio_domains) / 2:
            key_findings.append(f"BIOLOGICAL PARTIAL: R-frequency produces 8e in {bio_pass}/{len(bio_domains)} biological domains")
        else:
            key_findings.append(f"BIOLOGICAL LIMITED: R-frequency produces 8e in only {bio_pass}/{len(bio_domains)} biological domains")

    # Check financial domain
    finance_results = [r for r in all_results if r.domain.startswith('stock')]
    if finance_results:
        finance_pass = sum(1 for r in finance_results if r.passes_8e)
        if finance_pass > 0:
            key_findings.append(f"CROSS-DOMAIN: R-frequency produces 8e in financial data ({finance_pass}/{len(finance_results)} tests)")
        else:
            key_findings.append("DOMAIN-SPECIFIC: R-frequency does NOT produce 8e in financial data")

    # Check robustness
    robust_results = [r for r in all_results if r.domain.startswith('robustness')]
    if robust_results:
        robust_pass = sum(1 for r in robust_results if r.passes_8e)
        robust_rate = robust_pass / len(robust_results)
        if robust_rate > 0.8:
            key_findings.append(f"HIGHLY ROBUST: {robust_rate*100:.0f}% of parameter variations maintain 8e")
        elif robust_rate > 0.5:
            key_findings.append(f"MODERATELY ROBUST: {robust_rate*100:.0f}% of parameter variations maintain 8e")
        else:
            key_findings.append(f"PARAMETER SENSITIVE: Only {robust_rate*100:.0f}% of parameter variations maintain 8e")

    # Check random matrices (control)
    random_results = [r for r in all_results if r.domain.startswith('random')]
    if random_results:
        random_pass = sum(1 for r in random_results if r.passes_8e)
        if random_pass == len(random_results):
            key_findings.append("UNIVERSAL: Even random R distributions produce 8e - it is a property of the EMBEDDING METHOD!")
        elif random_pass > len(random_results) / 2:
            key_findings.append(f"DISTRIBUTION-DEPENDENT: {random_pass}/{len(random_results)} random distributions produce 8e")
        else:
            key_findings.append(f"R-SPECIFIC: Only {random_pass}/{len(random_results)} random distributions produce 8e - R structure matters!")

    output["key_findings"] = key_findings

    if verbose:
        print("\n" + "=" * 80)
        print("KEY FINDINGS")
        print("=" * 80)
        for finding in key_findings:
            print(f"\n  * {finding}")
        print("\n" + "=" * 80)

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_all_universality_tests(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "r_frequency_universal_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
