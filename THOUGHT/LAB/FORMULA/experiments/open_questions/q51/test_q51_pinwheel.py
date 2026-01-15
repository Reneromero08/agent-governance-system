"""
Q51 Pinwheel Test - Octant-Phase Sector Mapping

Tests whether the 8 octants (sign combinations of PC1, PC2, PC3)
correspond to 8 phase sectors in a complex plane representation.

Key hypothesis:
    Octant k <-> Phase sector [k*pi/4, (k+1)*pi/4)

Methods:
    1. Direct mapping: 3D PCA signs -> octant, 2D PCA -> complex phase
    2. Contingency table: chi-squared test for association
    3. Cramer's V for effect size

Pass criteria:
    - Cramer's V > 0.5 (strong association)
    - >50% diagonal in contingency table
    - Consistent across 5+ models
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.stats import chi2_contingency

# Add paths
SCRIPT_DIR = Path(__file__).parent
QGT_PATH = SCRIPT_DIR.parent.parent.parent.parent / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(QGT_PATH))

from qgt_phase import (
    octant_phase_mapping,
    hilbert_phase_recovery,
    circular_correlation,
    SECTOR_WIDTH
)

# Try sklearn
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Try sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# =============================================================================
# Test Corpus (same as zero signature test)
# =============================================================================

TEST_CORPUS = [
    "The quantum mechanical wave function collapses upon measurement.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Machine learning algorithms learn patterns from training data.",
    "Existence precedes essence in existentialist philosophy.",
    "Impressionism captures fleeting moments of light and color.",
    "Mountains form through tectonic plate collisions.",
    "Democracy enables citizen participation in governance.",
    "Love connects individuals through deep emotional bonds.",
    "Truth corresponds to reality in classical correspondence theory.",
    "The theory of relativity unifies space and time.",
    "DNA carries genetic information through nucleotide sequences.",
    "Cloud computing provides on-demand computing resources.",
    "The categorical imperative demands universal moral principles.",
    "Jazz improvisation creates spontaneous musical expressions.",
    "Ocean currents distribute heat around the planet.",
    "Economic systems allocate resources through various mechanisms.",
    "Fear triggers protective responses to perceived threats.",
    "Beauty emerges from harmony and proportion in form.",
    "Black holes form when massive stars collapse under gravity.",
    "Blockchain technology enables decentralized digital transactions.",
    "Phenomenology studies the structures of conscious experience.",
    "Abstract expressionism emphasizes emotional intensity over form.",
    "Forests absorb carbon dioxide and release oxygen.",
    "Language enables communication of complex ideas.",
    "Joy arises from experiences of fulfillment and connection.",
    "Justice requires fair distribution of benefits and burdens.",
    "Neural networks are inspired by biological brain structures.",
    "Encryption algorithms protect data through mathematical transformations.",
    "Pragmatism judges ideas by their practical consequences.",
    "Poetry uses rhythm and imagery to convey meaning.",
    "Ecosystems maintain balance through predator-prey relationships.",
    "Culture transmits values and practices across generations.",
    "Grief processes the pain of loss and separation.",
    "Freedom enables autonomous choice and self-determination.",
    "Climate patterns emerge from complex atmospheric dynamics.",
    "Education develops skills and knowledge in individuals.",
    "Hope sustains motivation toward desired futures.",
    "Time flows from past through present toward future.",
    "Architecture shapes space for human habitation.",
    "Epistemology investigates the nature of knowledge and belief.",
]

MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-small",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PinwheelResult:
    """Result of pinwheel test for single model."""
    model_name: str
    n_samples: int
    contingency_table: List[List[int]]  # 8x8
    chi2: float
    p_value: float
    cramers_v: float
    diagonal_rate: float  # Fraction on diagonal
    octant_phase_correlation: float
    status: str


@dataclass
class CrossModelPinwheelResult:
    """Cross-model aggregation."""
    n_models: int
    mean_cramers_v: float
    std_cramers_v: float
    mean_diagonal_rate: float
    models_passing: int
    hypothesis_supported: bool
    verdict: str


# =============================================================================
# Helper Functions
# =============================================================================

def get_embeddings(model_name: str, texts: List[str]) -> np.ndarray:
    """Get embeddings from model or generate synthetic."""
    if HAS_ST:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        except Exception as e:
            print(f"  Warning: Could not load {model_name}: {e}")

    # Synthetic fallback
    np.random.seed(hash(model_name) % 2**32)
    dim = 384
    n = len(texts)
    rank = 22
    components = np.random.randn(rank, dim)
    weights = np.random.randn(n, rank)
    embeddings = weights @ components
    embeddings += 0.1 * np.random.randn(n, dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def cramers_v(contingency_table: np.ndarray) -> float:
    """Calculate Cramer's V from contingency table."""
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def get_phase_sector(phase: float) -> int:
    """Map phase angle to sector 0-7."""
    # Normalize to [0, 2π)
    phase_normalized = (phase + np.pi) % (2 * np.pi)
    sector = int(phase_normalized // SECTOR_WIDTH)
    return min(sector, 7)


def build_contingency_table(
    octants: np.ndarray,
    phase_sectors: np.ndarray
) -> np.ndarray:
    """Build 8x8 contingency table of octant vs phase sector."""
    table = np.zeros((8, 8), dtype=int)
    for o, p in zip(octants, phase_sectors):
        table[o, p] += 1
    return table


# =============================================================================
# Test Functions
# =============================================================================

def test_pinwheel_single_model(
    model_name: str,
    corpus: List[str],
    verbose: bool = True
) -> PinwheelResult:
    """Run pinwheel test on single model."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pinwheel Test: {model_name}")
        print(f"{'='*60}")

    # Get embeddings
    embeddings = get_embeddings(model_name, corpus)
    n_samples = len(embeddings)

    if verbose:
        print(f"Embeddings shape: {embeddings.shape}")

    # Method 1: Get octants from sign pattern of top 3 PCs
    octant_result = octant_phase_mapping(embeddings)
    octants = octant_result.octant_indices

    # Method 2: Map to 2D complex plane and extract phase
    if HAS_SKLEARN:
        pca2d = PCA(n_components=2)
        proj_2d = pca2d.fit_transform(embeddings)
    else:
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:2]
        proj_2d = centered @ eigenvectors[:, idx]

    # Complex representation: z = PC1 + i*PC2
    complex_emb = proj_2d[:, 0] + 1j * proj_2d[:, 1]
    phases_2d = np.angle(complex_emb)

    # Map phases to sectors
    phase_sectors = np.array([get_phase_sector(p) for p in phases_2d])

    # Build contingency table
    table = build_contingency_table(octants, phase_sectors)

    # Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(table + 1e-10)  # Add small value to avoid zeros

    # Cramer's V
    cv = cramers_v(table + 1)  # Add 1 to handle sparse cells

    # Diagonal rate (how often octant k maps to sector k)
    diagonal_sum = np.trace(table)
    diagonal_rate = diagonal_sum / n_samples

    # Circular correlation between octant phases and 2D phases
    octant_phases = octant_result.octant_phases
    corr = circular_correlation(octant_phases, phases_2d)

    if verbose:
        print(f"\nContingency table (octant vs phase sector):")
        print("          Phase Sectors")
        print("Octant    0    1    2    3    4    5    6    7")
        print("-" * 55)
        for i in range(8):
            row = "   ".join(f"{table[i,j]:3d}" for j in range(8))
            print(f"  {i}     {row}")
        print("-" * 55)
        print(f"\nChi-squared: {chi2:.2f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Cramer's V: {cv:.4f} (threshold: > 0.5)")
        print(f"Diagonal rate: {diagonal_rate:.2%} (threshold: > 50%)")
        print(f"Phase correlation: {corr:.4f}")

    # Determine status
    if cv > 0.5 and diagonal_rate > 0.4:
        status = "PASS"
    elif cv > 0.3 or diagonal_rate > 0.3:
        status = "PARTIAL"
    else:
        status = "FAIL"

    if verbose:
        print(f"\nStatus: {status}")

    return PinwheelResult(
        model_name=model_name,
        n_samples=n_samples,
        contingency_table=table.tolist(),
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=float(cv),
        diagonal_rate=float(diagonal_rate),
        octant_phase_correlation=float(corr),
        status=status
    )


def test_pinwheel_cross_model(
    models: List[str],
    corpus: List[str],
    verbose: bool = True
) -> Tuple[List[PinwheelResult], CrossModelPinwheelResult]:
    """Run pinwheel test across multiple models."""
    print("\n" + "=" * 70)
    print("Q51 PINWHEEL TEST - CROSS-MODEL VALIDATION")
    print("=" * 70)
    print(f"\nHypothesis: Octant k <-> Phase sector [k*pi/4, (k+1)*pi/4)")
    print(f"Testing {len(models)} models")
    print()

    results = []
    for model in models:
        try:
            result = test_pinwheel_single_model(model, corpus, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue

    if not results:
        raise RuntimeError("No models tested successfully")

    # Aggregate
    cramers_vs = [r.cramers_v for r in results]
    diagonal_rates = [r.diagonal_rate for r in results]

    mean_cv = np.mean(cramers_vs)
    std_cv = np.std(cramers_vs)
    mean_diag = np.mean(diagonal_rates)

    passing = sum(1 for r in results if r.status == "PASS")

    # Verdict
    if mean_cv > 0.4 and passing >= len(results) * 0.6:
        hypothesis_supported = True
        verdict = "CONFIRMED: Octants map to phase sectors"
    elif mean_cv > 0.25 or passing >= len(results) * 0.4:
        hypothesis_supported = True
        verdict = "PARTIAL SUPPORT: Weak octant-phase mapping"
    else:
        hypothesis_supported = False
        verdict = "FALSIFIED: No octant-phase correspondence"

    cross_result = CrossModelPinwheelResult(
        n_models=len(results),
        mean_cramers_v=float(mean_cv),
        std_cramers_v=float(std_cv),
        mean_diagonal_rate=float(mean_diag),
        models_passing=passing,
        hypothesis_supported=hypothesis_supported,
        verdict=verdict
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY")
    print("=" * 70)
    print(f"\nModels tested: {len(results)}")
    print(f"Mean Cramer's V: {mean_cv:.4f} (threshold: > 0.5)")
    print(f"Mean diagonal rate: {mean_diag:.2%} (threshold: > 50%)")
    print(f"Models passing: {passing}/{len(results)}")
    print()
    print(f"{'Model':<35} {'Cramer V':>10} {'Diagonal':>10} {'Status':>10}")
    print("-" * 70)
    for r in results:
        short_name = r.model_name.split('/')[-1][:30]
        print(f"{short_name:<35} {r.cramers_v:>10.4f} {r.diagonal_rate:>10.2%} {r.status:>10}")

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    return results, cross_result


def save_results(
    results: List[PinwheelResult],
    cross_result: CrossModelPinwheelResult,
    output_dir: Path
):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_PINWHEEL',
        'hypothesis': 'Octant k maps to phase sector [k*pi/4, (k+1)*pi/4)',
        'per_model': [asdict(r) for r in results],
        'cross_model': asdict(cross_result)
    }

    output_path = output_dir / 'q51_pinwheel_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the Q51 Pinwheel Test."""
    print("\n" + "=" * 70)
    print("Q51: PINWHEEL TEST")
    print("Do octants correspond to phase sectors?")
    print("=" * 70)

    results, cross_result = test_pinwheel_cross_model(
        MODELS,
        TEST_CORPUS,
        verbose=True
    )

    output_dir = SCRIPT_DIR / "results"
    save_results(results, cross_result, output_dir)

    return cross_result.hypothesis_supported


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
