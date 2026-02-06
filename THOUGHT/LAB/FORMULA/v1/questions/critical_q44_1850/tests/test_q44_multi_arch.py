"""
Q44: Multi-Architecture Validation
===================================

Validate E = Born rule across MULTIPLE embedding architectures.
This proves the quantum structure is universal, not model-specific.

Run: python test_q44_multi_arch.py
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

# Real embeddings
from sentence_transformers import SentenceTransformer

# Local imports
from q44_test_cases import get_all_test_cases
from q44_statistics import full_correlation_analysis, check_monotonicity


# =============================================================================
# Embedding Validation
# =============================================================================

def validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Ensure embeddings are unit normalized."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return embeddings / norms


# =============================================================================
# Model Configurations
# =============================================================================

MODELS = [
    # (name, model_id, description)
    ("MiniLM-L6", "all-MiniLM-L6-v2", "384d, fast, general purpose"),
    ("MPNet-base", "all-mpnet-base-v2", "768d, highest quality"),
    ("Paraphrase-MiniLM", "paraphrase-MiniLM-L6-v2", "384d, paraphrase-tuned"),
    ("MultiQA-MiniLM", "multi-qa-MiniLM-L6-cos-v1", "384d, QA-tuned"),
    ("BGE-small", "BAAI/bge-small-en-v1.5", "384d, different architecture"),
]


# =============================================================================
# Core Computation
# =============================================================================

def compute_E_and_born(
    model: SentenceTransformer,
    query: str,
    context: List[str]
) -> Tuple[float, float, float]:
    """
    Compute E (mean overlap) and Born rule probability.

    Returns: (E, E_squared, P_born_mixed)
    """
    # Embed all texts
    all_texts = [query] + context
    vecs = model.encode(all_texts, normalize_embeddings=True)

    query_vec = vecs[0]
    context_vecs = vecs[1:]

    # Compute overlaps
    overlaps = [float(np.dot(query_vec, cv)) for cv in context_vecs]

    # E = mean overlap (quantum inner product)
    E = float(np.mean(overlaps))

    # E² = mean of squared overlaps
    E_squared = float(np.mean([o**2 for o in overlaps]))

    # Born rule (mixed state): P = mean(|⟨ψ|φᵢ⟩|²)
    P_born = float(np.mean([abs(o)**2 for o in overlaps]))

    return E, E_squared, P_born


def validate_model(
    model_name: str,
    model_id: str,
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run full validation for a single model.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_id})")
    print(f"{'='*60}")

    # Load model
    print(f"Loading model...")
    model = SentenceTransformer(model_id)
    dim = model.get_sentence_embedding_dimension()
    print(f"Loaded. Dimension: {dim}")

    # Run all test cases
    E_values = []
    E_sq_values = []
    P_born_values = []

    for i, tc in enumerate(test_cases):
        E, E_sq, P_born = compute_E_and_born(model, tc['query'], tc['context'])
        E_values.append(E)
        E_sq_values.append(E_sq)
        P_born_values.append(P_born)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(test_cases)} cases...")

    # Compute correlations
    from q44_statistics import pearson_correlation

    r_E = pearson_correlation(np.array(E_values), np.array(P_born_values))
    r_E_sq = pearson_correlation(np.array(E_sq_values), np.array(P_born_values))

    # Full statistical analysis for E
    analysis = full_correlation_analysis(E_values, P_born_values)
    monotonicity = check_monotonicity(E_values, P_born_values)

    result = {
        "model_name": model_name,
        "model_id": model_id,
        "dimension": dim,
        "n_cases": len(test_cases),
        "correlations": {
            "E": r_E,
            "E_squared": r_E_sq,
        },
        "E_vs_Born": {
            "r": analysis.r,
            "ci_low": analysis.ci_low,
            "ci_high": analysis.ci_high,
            "p_value": analysis.p_value,
            "verdict": analysis.verdict,
        },
        "spearman_rho": monotonicity['spearman_rho'],
        "monotonic": monotonicity['monotonic'],
    }

    # Print results
    print(f"\nResults:")
    print(f"  E vs P_born:    r = {r_E:.4f}")
    print(f"  E² vs P_born:   r = {r_E_sq:.4f}")
    print(f"  95% CI:         [{analysis.ci_low:.4f}, {analysis.ci_high:.4f}]")
    print(f"  Spearman rho:   {monotonicity['spearman_rho']:.4f}")
    print(f"  Verdict:        {analysis.verdict}")

    return result


def main():
    """
    Run multi-architecture validation.
    """
    print("="*60)
    print("Q44: MULTI-ARCHITECTURE VALIDATION")
    print("Does E = Born rule hold across ALL embedding models?")
    print("="*60)

    # Load test cases
    test_cases = get_all_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases")

    # Validate each model
    results = []
    for model_name, model_id, desc in MODELS:
        try:
            result = validate_model(model_name, model_id, test_cases)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to load {model_name}: {e}")
            results.append({
                "model_name": model_name,
                "model_id": model_id,
                "error": str(e),
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: E vs Born Rule Correlation Across Architectures")
    print("="*60)
    print(f"{'Model':<25} {'Dim':>5} {'r(E)':>8} {'95% CI':>18} {'Verdict':>12}")
    print("-"*75)

    valid_correlations = []
    for r in results:
        if 'error' in r:
            print(f"{r['model_name']:<25} {'N/A':>5} {'ERROR':>8}")
        else:
            ci = f"[{r['E_vs_Born']['ci_low']:.3f}, {r['E_vs_Born']['ci_high']:.3f}]"
            print(f"{r['model_name']:<25} {r['dimension']:>5} {r['correlations']['E']:>8.4f} {ci:>18} {r['E_vs_Born']['verdict']:>12}")
            valid_correlations.append(r['correlations']['E'])

    # Overall statistics
    if valid_correlations:
        mean_r = np.mean(valid_correlations)
        std_r = np.std(valid_correlations)
        min_r = np.min(valid_correlations)
        max_r = np.max(valid_correlations)

        print("-"*75)
        print(f"{'OVERALL':<25} {'':>5} {mean_r:>8.4f} ± {std_r:.4f}")
        print(f"{'Range':<25} {'':>5} [{min_r:.4f}, {max_r:.4f}]")

        # Final verdict
        all_quantum = all(r >= 0.9 for r in valid_correlations)
        print("\n" + "="*60)
        if all_quantum:
            print("VERDICT: QUANTUM STRUCTURE IS UNIVERSAL")
            print(f"All {len(valid_correlations)} architectures show r > 0.9")
        else:
            below_threshold = sum(1 for r in valid_correlations if r < 0.9)
            print(f"VERDICT: {below_threshold}/{len(valid_correlations)} models below r=0.9 threshold")
        print("="*60)

    # Save results
    output = {
        "document": "Q44_MULTI_ARCHITECTURE_VALIDATION",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_models": len(MODELS),
        "n_test_cases": len(test_cases),
        "models": results,
        "summary": {
            "mean_correlation": float(mean_r) if valid_correlations else None,
            "std_correlation": float(std_r) if valid_correlations else None,
            "min_correlation": float(min_r) if valid_correlations else None,
            "max_correlation": float(max_r) if valid_correlations else None,
            "all_quantum": all_quantum if valid_correlations else False,
        }
    }

    # Compute hash
    content_str = json.dumps(output, sort_keys=True)
    output['hash'] = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    # Save
    output_path = Path(__file__).parent / "q44_multi_arch_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
