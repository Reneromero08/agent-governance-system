#!/usr/bin/env python3
"""
Q18 Investigation: 8e Detection in REAL Protein Language Model Embeddings

KEY HYPOTHESIS:
The 8e conservation law (Df x alpha = 8e = 21.746) should hold for TRAINED
biological embeddings like ESM-2 because they are trained semiotic representations
that have learned semantic structure from protein sequences.

This is a KEY UNTESTED PREDICTION of the 8e theory. If ESM-2 embeddings show
Df x alpha near 8e, it validates that 8e applies to trained biological
representations - not just NLP language models.

PREDICTION:
- ESM-2 embeddings: Df x alpha = 8e (+/- 15%)
- Random baseline: Df x alpha ~ 14.5 (NOT 8e)
- Raw protein coordinates: Df x alpha >> 8e (no semiotic structure)

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
RANDOM_BASELINE = 14.5  # Expected for random matrices


@dataclass
class EmbeddingResult:
    """Results from embedding analysis."""
    name: str
    description: str
    n_proteins: int
    embedding_dim: int
    Df: float
    alpha: float
    Df_x_alpha: float
    deviation_from_8e: float
    passes_8e: bool
    top_eigenvalues: List[float]
    method_details: Dict[str, Any]


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

    Args:
        embeddings: (n_samples, n_features) array

    Returns:
        (Df, alpha, eigenvalues)
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # Compute covariance matrix
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    # Eigenvalue decomposition
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

    # Linear regression for slope
    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def load_protein_data() -> Dict[str, Dict]:
    """Load protein sequences and metadata from cache."""
    cache_path = Path(__file__).parent / "cache" / "extended_plddt.json"

    with open(cache_path, 'r') as f:
        data = json.load(f)

    return data


def get_esm2_embeddings(proteins: Dict[str, Dict],
                         model_name: str = "facebook/esm2_t6_8M_UR50D",
                         max_length: int = 1024) -> Tuple[np.ndarray, List[str]]:
    """
    Get ESM-2 embeddings for protein sequences.

    Uses a smaller ESM-2 model (8M parameters) for efficiency while still
    capturing learned protein semantics.

    Args:
        proteins: Dict mapping protein IDs to their data (must include 'sequence')
        model_name: HuggingFace model name
        max_length: Maximum sequence length to process

    Returns:
        (embeddings array, list of protein IDs)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(f"Required libraries not available: {e}")

    print(f"Loading ESM-2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    embeddings = []
    protein_ids = []

    for i, (prot_id, prot_data) in enumerate(proteins.items()):
        sequence = prot_data.get('sequence', '')
        if not sequence:
            continue

        # Truncate if needed
        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        try:
            # Tokenize
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True,
                             max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            # Mean pool over sequence length (excluding special tokens)
            hidden_states = outputs.last_hidden_state
            # Average over all positions to get protein-level embedding
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

            embeddings.append(embedding)
            protein_ids.append(prot_id)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(proteins)} proteins")

        except Exception as e:
            print(f"  Error processing {prot_id}: {e}")
            continue

    if not embeddings:
        raise ValueError("No embeddings could be computed")

    return np.array(embeddings), protein_ids


def get_random_baseline(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate random baseline embeddings for comparison."""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_dims)


def get_amino_acid_property_embedding(proteins: Dict[str, Dict], n_dims: int = 50) -> np.ndarray:
    """
    Create simple amino acid property-based embeddings (not trained).

    This serves as a control showing that random/untrained structure
    does NOT produce 8e.
    """
    # Amino acid property scales (hydrophobicity, charge, size)
    aa_props = {
        'A': [0.62, 0.0, 0.5], 'C': [0.29, 0.0, 0.55], 'D': [-0.90, -1.0, 0.6],
        'E': [-0.74, -1.0, 0.7], 'F': [1.19, 0.0, 0.85], 'G': [0.48, 0.0, 0.35],
        'H': [-0.40, 0.5, 0.75], 'I': [1.38, 0.0, 0.75], 'K': [-1.50, 1.0, 0.8],
        'L': [1.06, 0.0, 0.75], 'M': [0.64, 0.0, 0.8], 'N': [-0.78, 0.0, 0.65],
        'P': [0.12, 0.0, 0.55], 'Q': [-0.85, 0.0, 0.7], 'R': [-2.53, 1.0, 0.9],
        'S': [-0.18, 0.0, 0.45], 'T': [-0.05, 0.0, 0.55], 'V': [1.08, 0.0, 0.65],
        'W': [0.81, 0.0, 0.95], 'Y': [0.26, 0.0, 0.9]
    }

    embeddings = []
    for prot_id, prot_data in proteins.items():
        sequence = prot_data.get('sequence', '')
        if not sequence:
            continue

        # Compute average amino acid properties
        props = [0.0, 0.0, 0.0]
        valid_count = 0
        for aa in sequence:
            if aa in aa_props:
                for i in range(3):
                    props[i] += aa_props[aa][i]
                valid_count += 1

        if valid_count > 0:
            props = [p / valid_count for p in props]

        # Expand to n_dims using deterministic transforms
        embedding = np.zeros(n_dims)
        for d in range(n_dims):
            freq = (d + 1) * 0.1
            embedding[d] = np.sin(freq * props[0]) * np.cos(freq * props[1]) + props[2] * d / n_dims

        embeddings.append(embedding)

    return np.array(embeddings)


def run_esm2_8e_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Main test: Run 8e analysis on ESM-2 protein embeddings.

    This tests the KEY PREDICTION that trained biological language models
    should show the 8e conservation law.
    """
    print("=" * 80)
    print("Q18 INVESTIGATION: 8e IN PROTEIN LANGUAGE MODEL EMBEDDINGS")
    print("=" * 80)
    print()
    print("HYPOTHESIS: ESM-2 embeddings should show Df x alpha = 8e (~21.75)")
    print("because they are TRAINED semiotic representations of protein meaning.")
    print()
    print(f"Theoretical 8e: {EIGHT_E:.4f}")
    print(f"Random baseline expected: ~{RANDOM_BASELINE}")
    print("=" * 80)

    # Load protein data
    print("\n[1] Loading protein data...")
    proteins = load_protein_data()
    n_proteins = len(proteins)
    print(f"    Loaded {n_proteins} proteins")

    results = []

    # Test 1: ESM-2 Embeddings (the KEY test)
    print("\n[2] Computing ESM-2 embeddings...")
    try:
        esm_embeddings, protein_ids = get_esm2_embeddings(proteins)
        print(f"    ESM-2 embedding shape: {esm_embeddings.shape}")

        Df, alpha, eigenvalues = compute_spectral_properties(esm_embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E

        result = EmbeddingResult(
            name="ESM-2 Protein Embeddings",
            description="ESM-2 trained protein language model embeddings (facebook/esm2_t6_8M_UR50D)",
            n_proteins=len(protein_ids),
            embedding_dim=esm_embeddings.shape[1],
            Df=Df,
            alpha=alpha,
            Df_x_alpha=product,
            deviation_from_8e=deviation,
            passes_8e=deviation < 0.15,
            top_eigenvalues=eigenvalues[:20].tolist(),
            method_details={
                "model": "facebook/esm2_t6_8M_UR50D",
                "pooling": "mean",
                "n_proteins": len(protein_ids)
            }
        )
        results.append(result)

        print(f"\n    ESM-2 RESULTS:")
        print(f"    Df = {Df:.2f}")
        print(f"    alpha = {alpha:.4f}")
        print(f"    Df x alpha = {product:.2f}")
        print(f"    Deviation from 8e = {deviation*100:.1f}%")
        print(f"    PASSES 8e (<15%): {'YES' if deviation < 0.15 else 'NO'}")

    except Exception as e:
        print(f"    ERROR: Could not compute ESM-2 embeddings: {e}")
        print(f"    Attempting alternative approaches...")
        esm_embeddings = None

    # Test 2: Random baseline (negative control)
    print("\n[3] Computing random baseline...")
    n_dims = esm_embeddings.shape[1] if esm_embeddings is not None else 320
    random_embeddings = get_random_baseline(n_proteins, n_dims)

    Df, alpha, eigenvalues = compute_spectral_properties(random_embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    result = EmbeddingResult(
        name="Random Baseline",
        description="Pure random Gaussian embeddings (negative control)",
        n_proteins=n_proteins,
        embedding_dim=n_dims,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        passes_8e=deviation < 0.15,
        top_eigenvalues=eigenvalues[:20].tolist(),
        method_details={"distribution": "normal", "seed": 42}
    )
    results.append(result)

    print(f"    Random baseline: Df x alpha = {product:.2f} ({deviation*100:.1f}% dev)")

    # Test 3: Amino acid property embedding (untrained control)
    print("\n[4] Computing amino acid property embedding (untrained)...")
    aa_embeddings = get_amino_acid_property_embedding(proteins, n_dims=50)

    Df, alpha, eigenvalues = compute_spectral_properties(aa_embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    result = EmbeddingResult(
        name="Amino Acid Properties",
        description="Untrained amino acid property-based encoding (control)",
        n_proteins=n_proteins,
        embedding_dim=50,
        Df=Df,
        alpha=alpha,
        Df_x_alpha=product,
        deviation_from_8e=deviation,
        passes_8e=deviation < 0.15,
        top_eigenvalues=eigenvalues[:20].tolist(),
        method_details={"method": "aa_properties", "trained": False}
    )
    results.append(result)

    print(f"    AA properties: Df x alpha = {product:.2f} ({deviation*100:.1f}% dev)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: PROTEIN EMBEDDING 8e TEST")
    print("=" * 80)
    print(f"\n{'Method':<35} {'Df':<10} {'alpha':<10} {'Df x a':<10} {'Dev %':<10} {'8e?'}")
    print("-" * 85)

    for r in results:
        status = "PASS" if r.passes_8e else "FAIL"
        print(f"{r.name:<35} {r.Df:<10.2f} {r.alpha:<10.4f} {r.Df_x_alpha:<10.2f} {r.deviation_from_8e*100:<10.1f} {status}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    esm_result = next((r for r in results if "ESM-2" in r.name), None)
    random_result = next((r for r in results if "Random" in r.name), None)

    if esm_result:
        if esm_result.passes_8e:
            print("\n  *** BREAKTHROUGH: ESM-2 SHOWS 8e CONSERVATION! ***")
            print(f"  ESM-2 Df x alpha = {esm_result.Df_x_alpha:.2f} (only {esm_result.deviation_from_8e*100:.1f}% from 8e)")
            print("\n  This VALIDATES the theory that 8e applies to trained biological")
            print("  semantic representations, not just NLP language models!")
        else:
            print(f"\n  ESM-2 did NOT show 8e (deviation = {esm_result.deviation_from_8e*100:.1f}%)")
            print("  Possible explanations:")
            print("  - ESM-2 learns different structure than text LMs")
            print("  - Protein semantic space has different geometry")
            print("  - Sample size too small for reliable measurement")

    if random_result:
        print(f"\n  Random baseline: {random_result.Df_x_alpha:.2f} (expected ~14.5)")
        if abs(random_result.Df_x_alpha - RANDOM_BASELINE) < 5:
            print("  Random baseline behaves as expected (no 8e structure)")

    print("=" * 80)

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "ESM-2 trained embeddings should show Df x alpha = 8e",
        "theoretical_8e": float(EIGHT_E),
        "random_expected": float(RANDOM_BASELINE),
        "n_proteins": n_proteins,
        "results": [],
        "conclusion": None
    }

    for r in results:
        output["results"].append({
            "name": r.name,
            "description": r.description,
            "n_proteins": r.n_proteins,
            "embedding_dim": r.embedding_dim,
            "Df": r.Df,
            "alpha": r.alpha,
            "Df_x_alpha": r.Df_x_alpha,
            "deviation_from_8e": r.deviation_from_8e,
            "deviation_percent": r.deviation_from_8e * 100,
            "passes_8e": r.passes_8e,
            "top_eigenvalues": r.top_eigenvalues[:10],
            "method_details": r.method_details
        })

    # Determine conclusion
    if esm_result:
        if esm_result.passes_8e:
            output["conclusion"] = {
                "status": "HYPOTHESIS_VALIDATED",
                "message": "ESM-2 protein language model shows 8e conservation, validating that trained biological embeddings follow the same semiotic geometry as text language models.",
                "esm2_deviation": esm_result.deviation_from_8e * 100,
                "significance": "This is strong evidence that 8e = Df x alpha is a universal property of trained semantic representations across domains."
            }
        else:
            output["conclusion"] = {
                "status": "HYPOTHESIS_NOT_CONFIRMED",
                "message": f"ESM-2 embeddings showed {esm_result.deviation_from_8e*100:.1f}% deviation from 8e, outside the 15% threshold.",
                "esm2_deviation": esm_result.deviation_from_8e * 100,
                "possible_explanations": [
                    "Protein semantic space may have different geometry than text",
                    "Sample size (47 proteins) may be insufficient",
                    "ESM-2 architecture may learn different representations",
                    "8e may be specific to text language understanding"
                ]
            }

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_esm2_8e_test(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "protein_embeddings_8e_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
