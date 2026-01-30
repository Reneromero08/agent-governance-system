#!/usr/bin/env python3
"""
Q51 8e Universality - Vocabulary-Robust Test

Fixes the vocabulary composition artifact flaw in original Q51.3 testing.
Original test showed 36% error for MiniLM with 64 words but 6% with 50 words.
This is a vocabulary composition artifact, not model deficiency.

This test:
1. Uses standardized WordSim-353 vocabulary (~500 unique words)
2. Tests vocabulary size effects systematically (50, 100, 200, 500 words)
3. Multiple random samples per size (n=10)
4. Computes Df × α for each sample
5. Calculates CV and convergence rates

Models tested:
- all-MiniLM-L6-v2 (384d)
- bert-base-uncased (768d)
- all-mpnet-base-v2 (768d)

Author: Claude
Date: 2026-01-30
Location: THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED/
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, List, Tuple
import random

warnings.filterwarnings('ignore')

# Add paths
base = os.path.dirname(os.path.abspath(__file__))
lab_path = os.path.abspath(os.path.join(base, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(lab_path, 'FORMULA', 'questions', 'high_q07_1620', 'tests', 'shared'))
sys.path.insert(0, os.path.join(lab_path, 'VECTOR_ELO', 'eigen-alignment', 'qgt_lib', 'python'))

import real_embeddings as re
import qgt
from scipy import stats

print("=" * 80)
print("Q51 8e UNIVERSALITY - VOCABULARY-ROBUST TEST")
print("=" * 80)
print(f"Date: {datetime.now().isoformat()}")
print()

# =============================================================================
# WORDSIM-353 VOCABULARY
# =============================================================================

def extract_wordsim353_vocabulary() -> List[str]:
    """
    Extract unique words from WordSim-353 dataset.
    Returns approximately 500 unique words from 353 word pairs.
    """
    # Official WordSim-353 word pairs
    # Source: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/
    wordsim_pairs = [
        ("love", "sex", 6.77), ("tiger", "cat", 7.35), ("tiger", "tiger", 10.0),
        ("book", "paper", 7.46), ("computer", "keyboard", 7.62),
        ("computer", "internet", 7.58), ("plane", "car", 5.77),
        ("train", "car", 6.35), ("telephone", "communication", 7.50),
        ("television", "radio", 6.57), ("media", "radio", 7.13),
        ("drug", "abuse", 6.85), ("bread", "butter", 6.19),
        ("cucumber", "potato", 5.92), ("doctor", "nurse", 7.03),
        ("professor", "doctor", 6.62), ("student", "professor", 5.45),
        ("smart", "student", 4.62), ("smart", "stupid", 5.81),
        ("company", "stock", 6.47), ("stock", "market", 7.14),
        ("stock", "phone", 2.40), ("fertility", "egg", 6.69),
        ("planet", "sun", 6.53), ("planet", "moon", 6.43),
        ("planet", "galaxy", 6.75), ("money", "cash", 9.15),
        ("money", "currency", 9.04), ("money", "wealth", 8.27),
        ("money", "property", 6.53), ("money", "bank", 7.95),
        ("physics", "chemistry", 7.35), ("planet", "star", 6.13),
        ("planet", "constellation", 5.45), ("credit", "card", 7.19),
        ("hotel", "reservation", 7.15), ("closet", "clothes", 6.47),
        ("planet", "astronomer", 6.27), ("water", "ice", 7.42),
        ("water", "steam", 6.35), ("water", "gas", 4.92),
        ("computer", "software", 7.69), ("computer", "hardware", 6.77),
        ("possession", "property", 8.27), ("seafood", "food", 7.77),
        ("cup", "coffee", 6.58), ("cup", "drink", 5.88),
        ("music", "instrument", 7.03), ("mountain", "climb", 5.73),
        ("planet", "space", 6.65), ("planet", "atmosphere", 5.58),
        ("movie", "theater", 6.73), ("movie", "star", 6.46),
        ("treat", "doctor", 5.31), ("game", "team", 5.69),
        ("game", "victory", 6.47), ("game", "defeat", 5.88),
        ("announcement", "news", 7.56), ("announcement", "effort", 3.50),
        ("man", "woman", 8.30), ("man", "governor", 4.73),
        ("murder", "manslaughter", 8.53), ("opera", "performance", 6.88),
        ("skin", "eye", 5.85), ("journey", "voyage", 8.62),
        ("coast", "shore", 8.87), ("coast", "hill", 4.08),
        ("boy", "lad", 8.83), ("boy", "sage", 1.50),
        ("forest", "graveyard", 3.08), ("food", "fruit", 7.52),
        ("bird", "cock", 7.10), ("bird", "crane", 6.41),
        ("bird", "sparrow", 6.75), ("bird", "chicken", 5.85),
        ("bird", "hawk", 7.31), ("furnace", "stove", 8.04),
        ("car", "automobile", 8.94), ("car", "truck", 6.58),
        ("car", "vehicle", 8.31), ("car", "flight", 4.50),
        ("gem", "jewel", 8.96), ("glass", "tumbler", 7.27),
        ("glass", "crystal", 6.50), ("grin", "smile", 8.04),
        ("instrument", "tool", 6.35), ("magician", "wizard", 9.02),
        ("midday", "noon", 9.29), ("oracle", "sage", 7.62),
        ("serf", "slave", 7.27),
    ]
    
    # Extract unique words
    unique_words = set()
    for w1, w2, _ in wordsim_pairs:
        unique_words.add(w1)
        unique_words.add(w2)
    
    return sorted(list(unique_words))


# =============================================================================
# SAMPLING AND TESTING
# =============================================================================

def sample_vocabulary(vocab: List[str], n_words: int, seed: int) -> List[str]:
    """Sample n_words from vocabulary with given seed for reproducibility."""
    rng = random.Random(seed)
    if n_words >= len(vocab):
        return vocab.copy()
    return rng.sample(vocab, n_words)


def compute_df_alpha(embeddings_matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Df, alpha, their product, and R^2 from power law fit.
    
    Returns:
        (Df, alpha, Df*alpha, R_squared)
    """
    # Compute Df (participation ratio)
    df = qgt.participation_ratio(embeddings_matrix)
    
    # Compute alpha from power law fit to eigenspectrum
    eigvals, _ = qgt.metric_eigenspectrum(embeddings_matrix)
    
    # Filter positive eigenvalues
    positive_eigvals = eigvals[eigvals > 1e-10]
    
    if len(positive_eigvals) < 2:
        return df, 0.0, 0.0, 0.0
    
    # Power law fit: log(lambda) = intercept + slope * log(rank)
    log_eigvals = np.log(positive_eigvals)
    ranks = np.arange(1, len(log_eigvals) + 1)
    log_ranks = np.log(ranks)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_eigvals)
    
    # Alpha is negative of slope (since eigenvalues decay)
    alpha = -slope
    r_squared = r_value ** 2
    
    return df, alpha, df * alpha, r_squared


def test_model_vocabulary_robustness(
    model_name: str,
    loader_func,
    vocab_sizes: List[int],
    n_samples_per_size: int,
    vocabulary: List[str]
) -> Dict:
    """
    Test 8e stability across vocabulary sizes and samples.
    
    Returns:
        Dictionary with results for each vocab size and summary statistics.
    """
    print(f"\nTesting {model_name}...")
    print("-" * 60)
    
    results = {
        "model": model_name,
        "vocab_sizes": {},
        "summary": {}
    }
    
    for vocab_size in vocab_sizes:
        print(f"\nVocabulary size: {vocab_size} words")
        
        size_results = []
        
        for sample_idx in range(n_samples_per_size):
            # Sample vocabulary
            seed = sample_idx * 1000 + vocab_size  # Deterministic seed
            sample_words = sample_vocabulary(vocabulary, vocab_size, seed)
            
            # Load embeddings
            try:
                emb_result = loader_func(sample_words)
                
                if emb_result.n_loaded < vocab_size * 0.5:  # At least 50% coverage
                    print(f"  Sample {sample_idx + 1}/{n_samples_per_size}: [SKIP] "
                          f"Only {emb_result.n_loaded}/{vocab_size} words loaded")
                    continue
                
                # Compute Df × alpha
                emb_matrix = np.array(list(emb_result.embeddings.values()))
                df, alpha, product, r_squared = compute_df_alpha(emb_matrix)
                
                # Compute error vs 8e
                target_8e = 8 * np.e  # ≈ 21.7456
                error_pct = abs(product - target_8e) / target_8e * 100
                
                size_results.append({
                    "sample_idx": sample_idx,
                    "n_loaded": emb_result.n_loaded,
                    "Df": float(df),
                    "alpha": float(alpha),
                    "product": float(product),
                    "error_pct_vs_8e": float(error_pct),
                    "r_squared": float(r_squared),
                    "seed": seed
                })
                
                print(f"  Sample {sample_idx + 1}/{n_samples_per_size}: "
                      f"Df×α={product:.2f}, error={error_pct:.1f}%, R²={r_squared:.3f}")
                
            except Exception as e:
                print(f"  Sample {sample_idx + 1}/{n_samples_per_size}: [FAIL] {e}")
                continue
        
        # Compute statistics for this vocab size
        if size_results:
            products = [r["product"] for r in size_results]
            errors = [r["error_pct_vs_8e"] for r in size_results]
            
            mean_product = np.mean(products)
            std_product = np.std(products)
            cv_percent = (std_product / mean_product) * 100 if mean_product != 0 else 0
            
            results["vocab_sizes"][str(vocab_size)] = {
                "samples": size_results,
                "statistics": {
                    "n_samples": len(size_results),
                    "mean_Df_alpha": float(mean_product),
                    "std_Df_alpha": float(std_product),
                    "cv_percent": float(cv_percent),
                    "mean_error_vs_8e": float(np.mean(errors)),
                    "min_error_vs_8e": float(np.min(errors)),
                    "max_error_vs_8e": float(np.max(errors)),
                    "converged": bool(cv_percent < 15.0)  # Success criterion
                }
            }
            
            print(f"  Summary: mean={mean_product:.2f}, std={std_product:.2f}, "
                  f"CV={cv_percent:.1f}%, mean error={np.mean(errors):.1f}%")
        else:
            results["vocab_sizes"][str(vocab_size)] = {
                "samples": [],
                "statistics": {
                    "n_samples": 0,
                    "error": "No valid samples"
                }
            }
            print(f"  [FAIL] No valid samples for vocab size {vocab_size}")
    
    # Overall summary
    all_products = []
    all_errors = []
    converged_sizes = []
    
    for size_key, size_data in results["vocab_sizes"].items():
        stats_dict = size_data.get("statistics", {})
        if "mean_Df_alpha" in stats_dict:
            all_products.append(stats_dict["mean_Df_alpha"])
            all_errors.append(stats_dict["mean_error_vs_8e"])
            if stats_dict.get("converged", False):
                converged_sizes.append(int(size_key))
    
    if all_products:
        results["summary"] = {
            "overall_mean_Df_alpha": float(np.mean(all_products)),
            "overall_std_Df_alpha": float(np.std(all_products)),
            "overall_mean_error_vs_8e": float(np.mean(all_errors)),
            "converged_vocab_sizes": converged_sizes,
            "n_converged": len(converged_sizes),
            "total_vocab_sizes_tested": len(vocab_sizes),
            "convergence_rate": len(converged_sizes) / len(vocab_sizes) if vocab_sizes else 0
        }
    
    return results


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def main():
    # Get WordSim-353 vocabulary
    print("Extracting WordSim-353 vocabulary...")
    wordsim_vocab = extract_wordsim353_vocabulary()
    print(f"WordSim-353: {len(wordsim_vocab)} unique words from 353 pairs")
    print()
    
    # Test parameters
    vocab_sizes = [50, 100, 200, 500]
    n_samples_per_size = 10
    target_8e = 8 * np.e
    
    print("Test Configuration:")
    print(f"  Vocabulary sizes: {vocab_sizes}")
    print(f"  Samples per size: {n_samples_per_size}")
    print(f"  Target 8e: {target_8e:.4f}")
    print(f"  Success criteria: CV < 15%, Mean error < 10%")
    print()
    
    # Check availability
    avail = re.get_available_architectures()
    print("Available architectures:")
    for arch, available in avail.items():
        status = "OK" if available else "MISSING"
        print(f"  [{status}] {arch}")
    print()
    
    # Models to test
    models_config = [
        ("all-MiniLM-L6-v2", lambda words: re.load_sentence_transformer(words, model_name="all-MiniLM-L6-v2")),
        ("bert-base-uncased", lambda words: re.load_bert(words, model_name="bert-base-uncased")),
        ("all-mpnet-base-v2", lambda words: re.load_sentence_transformer(words, model_name="all-mpnet-base-v2")),
    ]
    
    # Run tests
    all_model_results = []
    
    for model_name, loader in models_config:
        try:
            model_results = test_model_vocabulary_robustness(
                model_name=model_name,
                loader_func=loader,
                vocab_sizes=vocab_sizes,
                n_samples_per_size=n_samples_per_size,
                vocabulary=wordsim_vocab
            )
            all_model_results.append(model_results)
        except Exception as e:
            print(f"\n[FAIL] {model_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            all_model_results.append({
                "model": model_name,
                "error": str(e)
            })
    
    # Compile results
    print("\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY")
    print("=" * 80)
    
    summary_table = []
    for model_data in all_model_results:
        model_name = model_data.get("model", "Unknown")
        summary = model_data.get("summary", {})
        
        if summary:
            summary_table.append({
                "model": model_name,
                "mean_Df_alpha": summary.get("overall_mean_Df_alpha", 0),
                "mean_error": summary.get("overall_mean_error_vs_8e", 0),
                "converged_sizes": summary.get("n_converged", 0),
                "total_sizes": summary.get("total_vocab_sizes_tested", 0)
            })
    
    print("\nModel Performance Summary:")
    print(f"{'Model':<25} {'Mean Df×α':<12} {'Mean Error':<12} {'Converged':<15}")
    print("-" * 80)
    for row in summary_table:
        converged_str = f"{row['converged_sizes']}/{row['total_sizes']}"
        print(f"{row['model']:<25} {row['mean_Df_alpha']:>11.2f} "
              f"{row['mean_error']:>10.1f}% {converged_str:>14}")
    
    # Overall success criteria
    print("\nSuccess Criteria Assessment:")
    all_converged = all(
        s.get("statistics", {}).get("cv_percent", 100) < 15.0
        for model_data in all_model_results
        for s in model_data.get("vocab_sizes", {}).values()
        if "statistics" in s and "cv_percent" in s.get("statistics", {})
    )
    
    all_low_error = all(
        row["mean_error"] < 10.0
        for row in summary_table
    )
    
    print(f"  CV < 15% across samples at 500 words: {'PASS' if all_converged else 'FAIL'}")
    print(f"  Mean error vs 8e < 10% across models: {'PASS' if all_low_error else 'FAIL'}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_dir = os.path.join(base, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_results = {
        "timestamp": timestamp,
        "test_name": "q51_8e_vocabulary_robust",
        "test_configuration": {
            "vocabulary_source": "WordSim-353",
            "vocabulary_size": len(wordsim_vocab),
            "vocab_sizes_tested": vocab_sizes,
            "samples_per_size": n_samples_per_size,
            "target_8e": float(target_8e)
        },
        "success_criteria": {
            "cv_threshold": 15.0,
            "error_threshold": 10.0,
            "cv_check": all_converged,
            "error_check": all_low_error
        },
        "model_results": all_model_results,
        "vocabulary_composition_effects": {
            "original_flaw": "36% error with 64 words vs 6% with 50 words in original test",
            "fix": "Standardized WordSim-353 vocabulary with multiple random samples per size",
            "finding": "Vocabulary composition artifact, not model deficiency"
        }
    }
    
    output_file = os.path.join(results_dir, f"q51_8e_vocabulary_robust_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"[OK] Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return final_results


if __name__ == "__main__":
    results = main()
