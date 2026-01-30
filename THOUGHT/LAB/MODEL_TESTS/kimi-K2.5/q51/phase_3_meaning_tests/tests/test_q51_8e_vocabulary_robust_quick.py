#!/usr/bin/env python3
"""
Q51 8e Universality - Vocabulary-Robust Test (QUICK VERSION)

This is a quick validation version with reduced samples for faster execution.
Full version: test_q51_8e_vocabulary_robust.py

Fixes the vocabulary composition artifact flaw in original Q51.3 testing.
Original test showed 36% error for MiniLM with 64 words but 6% with 50 words.
This is a vocabulary composition artifact, not model deficiency.

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
print("Q51 8e UNIVERSALITY - VOCABULARY-ROBUST TEST (QUICK)")
print("=" * 80)
print(f"Date: {datetime.now().isoformat()}")
print()

# =============================================================================
# WORDSIM-353 VOCABULARY
# =============================================================================

def extract_wordsim353_vocabulary() -> List[str]:
    """Extract unique words from WordSim-353 dataset."""
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
    
    unique_words = set()
    for w1, w2, _ in wordsim_pairs:
        unique_words.add(w1)
        unique_words.add(w2)
    
    return sorted(list(unique_words))


# =============================================================================
# MAIN TEST
# =============================================================================

print("Extracting WordSim-353 vocabulary...")
wordsim_vocab = extract_wordsim353_vocabulary()
print(f"WordSim-353: {len(wordsim_vocab)} unique words")
print()

# Quick test configuration
vocab_sizes = [50, 100]
n_samples_per_size = 3
target_8e = 8 * np.e

print("QUICK Test Configuration:")
print(f"  Vocabulary sizes: {vocab_sizes}")
print(f"  Samples per size: {n_samples_per_size} (reduced for quick test)")
print(f"  Target 8e: {target_8e:.4f}")
print()

# Check availability
avail = re.get_available_architectures()
print("Available architectures:")
for arch, available in avail.items():
    status = "OK" if available else "MISSING"
    print(f"  [{status}] {arch}")
print()

# Test just MiniLM for quick validation
print("Testing all-MiniLM-L6-v2 (quick validation)...")
print("-" * 60)

all_results = {
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "test_name": "q51_8e_vocabulary_robust_quick",
    "note": "Quick validation with reduced samples. Full test in test_q51_8e_vocabulary_robust.py",
    "test_configuration": {
        "vocabulary_source": "WordSim-353",
        "vocabulary_size": len(wordsim_vocab),
        "vocab_sizes_tested": vocab_sizes,
        "samples_per_size": n_samples_per_size,
        "target_8e": float(target_8e),
        "models_tested": ["all-MiniLM-L6-v2"]
    },
    "model_results": []
}

model_results = {
    "model": "all-MiniLM-L6-v2",
    "vocab_sizes": {}
}

for vocab_size in vocab_sizes:
    print(f"\nVocabulary size: {vocab_size} words")
    
    size_results = []
    
    for sample_idx in range(n_samples_per_size):
        seed = sample_idx * 1000 + vocab_size
        rng = random.Random(seed)
        sample_words = rng.sample(wordsim_vocab, min(vocab_size, len(wordsim_vocab)))
        
        try:
            emb_result = re.load_sentence_transformer(sample_words, model_name="all-MiniLM-L6-v2")
            
            if emb_result.n_loaded < vocab_size * 0.5:
                print(f"  Sample {sample_idx + 1}/{n_samples_per_size}: [SKIP] "
                      f"Only {emb_result.n_loaded}/{vocab_size} words loaded")
                continue
            
            emb_matrix = np.array(list(emb_result.embeddings.values()))
            
            # Compute Df
            df = qgt.participation_ratio(emb_matrix)
            
            # Compute alpha
            eigvals, _ = qgt.metric_eigenspectrum(emb_matrix)
            positive_eigvals = eigvals[eigvals > 1e-10]
            
            if len(positive_eigvals) >= 2:
                log_eigvals = np.log(positive_eigvals)
                ranks = np.arange(1, len(log_eigvals) + 1)
                slope, _, r_value, _, _ = stats.linregress(np.log(ranks), log_eigvals)
                alpha = -slope
                r_squared = r_value ** 2
            else:
                alpha = 0.0
                r_squared = 0.0
            
            product = df * alpha
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
    
    if size_results:
        products = [r["product"] for r in size_results]
        errors = [r["error_pct_vs_8e"] for r in size_results]
        
        mean_product = np.mean(products)
        std_product = np.std(products)
        cv_percent = (std_product / mean_product) * 100 if mean_product != 0 else 0
        
        model_results["vocab_sizes"][str(vocab_size)] = {
            "samples": size_results,
            "statistics": {
                "n_samples": len(size_results),
                "mean_Df_alpha": float(mean_product),
                "std_Df_alpha": float(std_product),
                "cv_percent": float(cv_percent),
                "mean_error_vs_8e": float(np.mean(errors)),
                "converged": bool(cv_percent < 15.0)
            }
        }
        
        print(f"  Summary: mean={mean_product:.2f}, CV={cv_percent:.1f}%, "
              f"mean error={np.mean(errors):.1f}%")

all_results["model_results"].append(model_results)

# Save results
results_dir = os.path.join(base, "..", "results")
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(results_dir, f"q51_8e_vocabulary_robust_{timestamp}_quick.json")

with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n[OK] Results saved to: {output_file}")
print("\nNOTE: This is a QUICK validation with reduced samples.")
print("For full test with all 3 models and 10 samples per size, run:")
print("  python tests/test_q51_8e_vocabulary_robust.py")
print("\n" + "=" * 80)
print("QUICK TEST COMPLETE")
print("=" * 80)
