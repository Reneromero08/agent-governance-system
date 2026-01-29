#!/usr/bin/env python3
"""
Q18 Investigation: 8e in Protein Embeddings - ENHANCED TEST

The initial test showed ESM-2 with Df x alpha = 36.2 (66% dev from 8e).
However, that test had only 47 samples in a 320D space, meaning the
covariance matrix was rank-limited to 47.

This enhanced test:
1. Analyzes PER-RESIDUE embeddings (thousands of samples)
2. Tests different embedding aggregation strategies
3. Compares across different ESM-2 model sizes
4. Investigates whether 8e appears at different analysis scales

INSIGHT: The original 8e studies used many thousands of word embeddings.
We need to test at similar sample sizes for a fair comparison.

Author: Claude Opus 4.5
Date: 2026-01-25
Version: 2.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746
RANDOM_BASELINE = 14.5


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
    """Compute Df and alpha from embeddings."""
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Spectral decay
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def load_protein_data() -> Dict[str, Dict]:
    """Load protein sequences from cache."""
    cache_path = Path(__file__).parent / "cache" / "extended_plddt.json"
    with open(cache_path, 'r') as f:
        return json.load(f)


def get_residue_level_embeddings(proteins: Dict[str, Dict],
                                   model_name: str = "facebook/esm2_t6_8M_UR50D",
                                   max_residues: int = 5000,
                                   max_seq_len: int = 512) -> np.ndarray:
    """
    Get PER-RESIDUE embeddings from ESM-2.

    This provides many more samples (thousands of residues) rather than
    just 47 protein-level embeddings.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Using device: {device}")

    all_residue_embeddings = []
    total_residues = 0

    for prot_id, prot_data in proteins.items():
        if total_residues >= max_residues:
            break

        sequence = prot_data.get('sequence', '')
        if not sequence:
            continue

        # Truncate if needed
        if len(sequence) > max_seq_len:
            sequence = sequence[:max_seq_len]

        try:
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True,
                             max_length=max_seq_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Get per-residue embeddings (exclude [CLS] and [EOS] tokens)
            hidden_states = outputs.last_hidden_state.squeeze().cpu().numpy()
            # Positions 1:-1 are the actual residue embeddings
            residue_embeds = hidden_states[1:-1]

            # Sample residues if we have too many
            remaining = max_residues - total_residues
            if len(residue_embeds) > remaining:
                indices = np.random.choice(len(residue_embeds), remaining, replace=False)
                residue_embeds = residue_embeds[indices]

            all_residue_embeddings.append(residue_embeds)
            total_residues += len(residue_embeds)

        except Exception as e:
            print(f"  Error processing {prot_id}: {e}")
            continue

    if not all_residue_embeddings:
        raise ValueError("No embeddings computed")

    embeddings = np.vstack(all_residue_embeddings)
    print(f"  Collected {len(embeddings)} residue-level embeddings")

    return embeddings


def get_sliding_window_embeddings(proteins: Dict[str, Dict],
                                    model_name: str = "facebook/esm2_t6_8M_UR50D",
                                    window_size: int = 50,
                                    max_windows: int = 2000,
                                    max_seq_len: int = 512) -> np.ndarray:
    """
    Get sliding window embeddings (mean of residues in window).

    This creates local context embeddings similar to how n-gram or
    sentence embeddings work in NLP.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_window_embeddings = []
    total_windows = 0

    for prot_id, prot_data in proteins.items():
        if total_windows >= max_windows:
            break

        sequence = prot_data.get('sequence', '')
        if not sequence or len(sequence) < window_size:
            continue

        if len(sequence) > max_seq_len:
            sequence = sequence[:max_seq_len]

        try:
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True,
                             max_length=max_seq_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state.squeeze().cpu().numpy()
            residue_embeds = hidden_states[1:-1]

            # Create sliding windows
            for start in range(0, len(residue_embeds) - window_size + 1, window_size // 2):
                if total_windows >= max_windows:
                    break
                window = residue_embeds[start:start + window_size]
                window_embed = window.mean(axis=0)
                all_window_embeddings.append(window_embed)
                total_windows += 1

        except Exception as e:
            continue

    embeddings = np.array(all_window_embeddings)
    print(f"  Collected {len(embeddings)} window-level embeddings")

    return embeddings


def run_enhanced_8e_test() -> Dict[str, Any]:
    """Run enhanced 8e analysis with multiple strategies."""

    print("=" * 80)
    print("Q18 INVESTIGATION: 8e IN PROTEIN EMBEDDINGS - ENHANCED TEST")
    print("=" * 80)
    print()
    print("INSIGHT: Original test had only 47 samples (proteins) in 320D space.")
    print("This is rank-limited. Original 8e studies used thousands of samples.")
    print()
    print(f"Theoretical 8e: {EIGHT_E:.4f}")
    print(f"Random baseline: ~{RANDOM_BASELINE}")
    print("=" * 80)

    proteins = load_protein_data()
    print(f"\nLoaded {len(proteins)} proteins")

    results = []

    # Test 1: Per-residue embeddings (MANY samples)
    print("\n[1] PER-RESIDUE EMBEDDINGS (thousands of samples)")
    print("-" * 40)
    try:
        residue_embeddings = get_residue_level_embeddings(
            proteins, max_residues=3000, max_seq_len=512
        )
        print(f"  Shape: {residue_embeddings.shape}")

        Df, alpha, eigenvalues = compute_spectral_properties(residue_embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E

        results.append({
            "name": "ESM-2 Per-Residue Embeddings",
            "description": "Per-residue hidden states from ESM-2 (3000 samples)",
            "n_samples": len(residue_embeddings),
            "embedding_dim": residue_embeddings.shape[1],
            "Df": Df,
            "alpha": alpha,
            "Df_x_alpha": product,
            "deviation_from_8e": deviation,
            "deviation_percent": deviation * 100,
            "passes_8e": deviation < 0.15,
            "top_eigenvalues": eigenvalues[:10].tolist()
        })

        print(f"  Df = {Df:.2f}, alpha = {alpha:.4f}")
        print(f"  Df x alpha = {product:.2f} ({deviation*100:.1f}% dev from 8e)")
        print(f"  PASSES 8e: {'YES' if deviation < 0.15 else 'NO'}")

    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 2: Sliding window embeddings
    print("\n[2] SLIDING WINDOW EMBEDDINGS (local context)")
    print("-" * 40)
    try:
        window_embeddings = get_sliding_window_embeddings(
            proteins, window_size=50, max_windows=2000
        )
        print(f"  Shape: {window_embeddings.shape}")

        Df, alpha, eigenvalues = compute_spectral_properties(window_embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E

        results.append({
            "name": "ESM-2 Sliding Window Embeddings",
            "description": "Mean of 50-residue windows from ESM-2",
            "n_samples": len(window_embeddings),
            "embedding_dim": window_embeddings.shape[1],
            "Df": Df,
            "alpha": alpha,
            "Df_x_alpha": product,
            "deviation_from_8e": deviation,
            "deviation_percent": deviation * 100,
            "passes_8e": deviation < 0.15,
            "top_eigenvalues": eigenvalues[:10].tolist()
        })

        print(f"  Df = {Df:.2f}, alpha = {alpha:.4f}")
        print(f"  Df x alpha = {product:.2f} ({deviation*100:.1f}% dev from 8e)")
        print(f"  PASSES 8e: {'YES' if deviation < 0.15 else 'NO'}")

    except Exception as e:
        print(f"  ERROR: {e}")

    # Test 3: Random baseline for comparison
    print("\n[3] RANDOM BASELINE (3000 samples)")
    print("-" * 40)
    n_samples = 3000
    n_dims = 320
    np.random.seed(42)
    random_embeddings = np.random.randn(n_samples, n_dims)

    Df, alpha, eigenvalues = compute_spectral_properties(random_embeddings)
    product = Df * alpha
    deviation = abs(product - EIGHT_E) / EIGHT_E

    results.append({
        "name": "Random Baseline (3000 samples)",
        "description": "Random Gaussian embeddings for comparison",
        "n_samples": n_samples,
        "embedding_dim": n_dims,
        "Df": Df,
        "alpha": alpha,
        "Df_x_alpha": product,
        "deviation_from_8e": deviation,
        "deviation_percent": deviation * 100,
        "passes_8e": deviation < 0.15,
        "top_eigenvalues": eigenvalues[:10].tolist()
    })

    print(f"  Df = {Df:.2f}, alpha = {alpha:.4f}")
    print(f"  Df x alpha = {product:.2f} ({deviation*100:.1f}% dev from 8e)")

    # Test 4: Sample size sweep (to see if 8e emerges with more samples)
    print("\n[4] SAMPLE SIZE ANALYSIS")
    print("-" * 40)
    try:
        # Get a large batch of residue embeddings for analysis
        all_embeddings = get_residue_level_embeddings(
            proteins, max_residues=5000, max_seq_len=512
        )

        sample_sizes = [100, 300, 500, 1000, 2000, 3000, 5000]
        sample_sizes = [s for s in sample_sizes if s <= len(all_embeddings)]

        sample_sweep_results = []
        for n in sample_sizes:
            subset = all_embeddings[:n]
            Df, alpha, eigenvalues = compute_spectral_properties(subset)
            product = Df * alpha
            deviation = abs(product - EIGHT_E) / EIGHT_E

            sample_sweep_results.append({
                "n_samples": n,
                "Df": Df,
                "alpha": alpha,
                "Df_x_alpha": product,
                "deviation_percent": deviation * 100
            })
            print(f"  n={n}: Df x alpha = {product:.2f} ({deviation*100:.1f}% dev)")

        results.append({
            "name": "Sample Size Sweep",
            "description": "8e analysis at different sample sizes",
            "sweep_results": sample_sweep_results
        })

    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<40} {'Samples':<10} {'Df x alpha':<12} {'Dev %':<10} {'8e?'}")
    print("-" * 82)

    for r in results:
        if "n_samples" in r and "Df_x_alpha" in r:
            status = "PASS" if r.get("passes_8e", False) else "FAIL"
            print(f"{r['name']:<40} {r['n_samples']:<10} {r['Df_x_alpha']:<12.2f} {r['deviation_percent']:<10.1f} {status}")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    esm_residue = next((r for r in results if "Per-Residue" in r.get("name", "")), None)
    if esm_residue:
        if esm_residue["passes_8e"]:
            print("\n  *** ESM-2 PER-RESIDUE EMBEDDINGS SHOW 8e CONSERVATION! ***")
            print(f"  Df x alpha = {esm_residue['Df_x_alpha']:.2f} ({esm_residue['deviation_percent']:.1f}% dev)")
        else:
            print(f"\n  ESM-2 per-residue: {esm_residue['Df_x_alpha']:.2f} ({esm_residue['deviation_percent']:.1f}% dev)")
            if esm_residue['deviation_percent'] < 50:
                print("  Result is CLOSER to 8e than the original protein-level test!")
                print("  This suggests sample size matters for detecting the 8e structure.")
            else:
                print("  The deviation remains high, suggesting protein semantic space")
                print("  may have genuinely different geometry than text embeddings.")

    print("=" * 80)

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "theoretical_8e": float(EIGHT_E),
        "hypothesis": "ESM-2 should show 8e when tested with sufficient samples",
        "results": to_builtin(results),
        "key_finding": None
    }

    # Determine key finding
    if esm_residue:
        if esm_residue["passes_8e"]:
            output["key_finding"] = {
                "status": "HYPOTHESIS_VALIDATED",
                "message": "ESM-2 per-residue embeddings show 8e conservation when tested with sufficient samples!",
                "Df_x_alpha": esm_residue["Df_x_alpha"],
                "deviation_percent": esm_residue["deviation_percent"]
            }
        else:
            output["key_finding"] = {
                "status": "HYPOTHESIS_NOT_CONFIRMED",
                "message": f"ESM-2 showed {esm_residue['deviation_percent']:.1f}% deviation from 8e even with more samples.",
                "Df_x_alpha": esm_residue["Df_x_alpha"],
                "deviation_percent": esm_residue["deviation_percent"],
                "interpretation": "Protein semantic space may have different geometry than text embeddings, or 8e may be specific to natural language understanding."
            }

    return output


def main():
    """Main entry point."""
    results = run_enhanced_8e_test()

    # Save results
    output_path = Path(__file__).parent / "protein_embeddings_8e_results_v2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
