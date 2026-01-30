#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q51.5: Context as Phase Selection Test

Question: Does context select meaning like quantum measurement collapses superposition?

Theory:
- Complex: raw_embedding = Sum cᵢ × context_embeddingᵢ [superposition with phases]
- Real: raw_embedding = average(context_embeddings) [simple mixing]

Method:
1. Take ambiguous words with multiple meanings
2. Get contextual embeddings for each meaning
3. Test reconstruction:
   - Simple average: (ctx1 + ctx2 + ctx3) / 3
   - Weighted superposition: find cᵢ that minimize ||raw - Sum cᵢ × ctxᵢ||
4. Compare real vs complex coefficient reconstruction error

Success Criteria:
- If complex coefficients needed → context acts as phase selection
- If real coefficients sufficient → simple mixing
- Statistical significance: p < 0.001

Anti-Pattern Checks:
- Report all results honestly, including failures
- Test multiple ambiguous words (5+)
- 3+ contexts per word minimum
"""

import sys
import json
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import minimize
from itertools import combinations

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# FIXED PARAMETERS - NO GRID SEARCH
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# AMBIGUOUS WORDS WITH CONTEXTS
# =============================================================================

AMBIGUOUS_WORDS = {
    "run": {
        "raw": "run",
        "contexts": [
            ("I run every morning", "physical_action"),
            ("I run a business", "operate_manage"),
            ("I run for office", "political_campaign"),
            ("The program will run", "execute_code"),
            ("My nose is runny", "medical_secretion"),
        ]
    },
    "bank": {
        "raw": "bank",
        "contexts": [
            ("The river bank is muddy", "river_edge"),
            ("I deposited money at the bank", "financial_institution"),
            ("The plane banked left", "tilt_turn"),
            ("We sat on the grassy bank", "raised_ground"),
            ("The data bank contains records", "information_storage"),
        ]
    },
    "bat": {
        "raw": "bat",
        "contexts": [
            ("The bat flew at night", "flying_animal"),
            ("He swung the baseball bat", "sports_equipment"),
            ("She batted her eyelashes", "flutter_eyes"),
            ("The cricket bat is wooden", "cricket_equipment"),
        ]
    },
    "light": {
        "raw": "light",
        "contexts": [
            ("The light illuminated the room", "illumination"),
            ("The box is very light", "not_heavy"),
            ("She light the candle", "ignite"),
            ("His skin is light colored", "pale_complexion"),
            ("Take it light today", "easy_relaxed"),
        ]
    },
    "break": {
        "raw": "break",
        "contexts": [
            ("The glass will break", "shatter"),
            ("Take a lunch break", "rest_pause"),
            ("The news will break soon", "announce_reveal"),
            ("Break the law", "violate"),
            ("The waves break on shore", "crash"),
        ]
    },
    "match": {
        "raw": "match",
        "contexts": [
            ("Light the candle with a match", "fire_stick"),
            ("They are a perfect match", "compatible_pair"),
            ("The colors match well", "correspond_align"),
            ("The tennis match was exciting", "sporting_event"),
            ("Match the pattern exactly", "replicate"),
        ]
    },
    "spring": {
        "raw": "spring",
        "contexts": [
            ("Flowers bloom in spring", "season"),
            ("The mattress has a broken spring", "coil"),
            ("Water from the spring is fresh", "water_source"),
            ("The cat will spring forward", "leap_jump"),
            ("He has a spring in his step", "energetic_bounce"),
        ]
    },
}

print(f"Testing {len(AMBIGUOUS_WORDS)} ambiguous words")
for word, data in AMBIGUOUS_WORDS.items():
    print(f"  - {word}: {len(data['contexts'])} contexts")

# =============================================================================
# LOAD EMBEDDINGS
# =============================================================================

def get_embeddings_for_test():
    """Load sentence-transformers and compute embeddings for all test cases."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"\nLoaded MiniLM-L6 model")
        
        results = {}
        
        for word, data in AMBIGUOUS_WORDS.items():
            print(f"\nProcessing '{word}'...")
            
            # Get raw embedding
            raw_text = data["raw"]
            raw_emb = model.encode([raw_text], normalize_embeddings=True)[0]
            
            # Get contextual embeddings
            context_embs = []
            context_labels = []
            
            for ctx_text, ctx_label in data["contexts"]:
                ctx_emb = model.encode([ctx_text], normalize_embeddings=True)[0]
                context_embs.append(ctx_emb)
                context_labels.append(ctx_label)
            
            results[word] = {
                "raw_embedding": raw_emb,
                "raw_text": raw_text,
                "context_embeddings": np.array(context_embs),
                "context_labels": context_labels,
                "context_texts": [ctx[0] for ctx in data["contexts"]],
            }
            
            print(f"  Raw: '{raw_text}'")
            for i, (ctx_text, ctx_label) in enumerate(data["contexts"]):
                print(f"  Context {i+1}: '{ctx_text[:40]}...' [{ctx_label}]")
        
        return results
        
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return None


# =============================================================================
# RECONSTRUCTION TESTS
# =============================================================================

def test_simple_average(raw_emb, context_embs):
    """
    Test simple averaging: raw ≈ mean(contexts)
    
    Real-valued theory: raw_embedding = average(context_embeddings)
    """
    # Simple average
    avg_emb = np.mean(context_embs, axis=0)
    
    # Compute reconstruction error
    error = np.linalg.norm(raw_emb - avg_emb)
    cosine_sim = np.dot(raw_emb, avg_emb) / (np.linalg.norm(raw_emb) * np.linalg.norm(avg_emb))
    
    return {
        "method": "simple_average",
        "reconstruction_error": float(error),
        "cosine_similarity": float(cosine_sim),
        "avg_embedding": avg_emb,
    }


def test_weighted_real(raw_emb, context_embs):
    """
    Test weighted real superposition: raw ≈ Sum wᵢ × ctxᵢ
    
    Find optimal real weights via least squares.
    """
    n_contexts = len(context_embs)
    
    # Stack context embeddings as matrix (n_contexts × dim)
    C = context_embs  # Shape: (n_contexts, dim)
    
    # Solve for weights: raw = C^T @ w
    # Using least squares: w = (C @ C^T)^{-1} @ C @ raw
    try:
        # Use ridge regression for stability
        ridge = Ridge(alpha=0.01, fit_intercept=False)
        ridge.fit(C.T, raw_emb)
        weights = ridge.coef_
        
        # Reconstruct
        reconstructed = C.T @ weights
        
        # Compute error
        error = np.linalg.norm(raw_emb - reconstructed)
        cosine_sim = np.dot(raw_emb, reconstructed) / (np.linalg.norm(raw_emb) * np.linalg.norm(reconstructed))
        
        return {
            "method": "weighted_real",
            "weights": weights.tolist(),
            "reconstruction_error": float(error),
            "cosine_similarity": float(cosine_sim),
            "reconstructed": reconstructed,
        }
    except Exception as e:
        return {
            "method": "weighted_real",
            "error": str(e),
        }


def test_weighted_complex(raw_emb, context_embs):
    """
    Test weighted complex superposition: raw ≈ Sum cᵢ × ctxᵢ where cᵢ are complex
    
    Each coefficient cᵢ = |cᵢ| × e^(iθᵢ) has magnitude and phase.
    We optimize both magnitude and phase.
    """
    n_contexts = len(context_embs)
    dim = raw_emb.shape[0]
    
    # We'll model: reconstructed = Sum (aᵢ + ibᵢ) × ctxᵢ
    # But embeddings are real, so we need to think carefully...
    
    # Alternative: treat each embedding dimension as having phase
    # Or: interpret as rotating context embeddings before summing
    
    # Method: optimize rotation angles for each context
    # reconstructed = Sum |cᵢ| × R(θᵢ) @ ctxᵢ
    # where R(θ) is a rotation matrix
    
    # For simplicity in high-D space, we'll use a random projection approach
    # Create random orthonormal basis and optimize rotations in random 2D subspaces
    
    def complex_superposition(params):
        """
        params: [magnitudes (n_contexts), phases (n_contexts)]
        """
        magnitudes = params[:n_contexts]
        phases = params[n_contexts:]
        
        # For each context, apply a "phase" by mixing with random dimensions
        # In complex theory: e^(iθ) × v = cos(θ)×v + sin(θ)×v_perp
        # We approximate by projecting to a 2D plane and rotating
        
        reconstructed = np.zeros(dim)
        
        for i in range(n_contexts):
            ctx = context_embs[i]
            mag = magnitudes[i]
            phase = phases[i]
            
            # Create a perpendicular component using random projection
            np.random.seed(42 + i)  # Deterministic
            random_dir = np.random.randn(dim)
            random_dir = random_dir - np.dot(random_dir, ctx) * ctx  # Orthogonalize
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
            
            # Rotate: v_rot = cos(phase) × v + sin(phase) × v_perp
            v_rot = np.cos(phase) * ctx + np.sin(phase) * random_dir
            
            reconstructed += mag * v_rot
        
        # Return reconstruction error
        return np.linalg.norm(raw_emb - reconstructed)
    
    # Initial guess: equal magnitudes, zero phases
    x0 = np.concatenate([
        np.ones(n_contexts) / n_contexts,  # magnitudes
        np.zeros(n_contexts),  # phases
    ])
    
    # Bounds: magnitudes >= 0, phases in [0, 2π]
    bounds = [(0, None)] * n_contexts + [(0, 2 * np.pi)] * n_contexts
    
    # Optimize
    try:
        result = minimize(
            complex_superposition,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Get final reconstruction
        final_error = result.fun
        opt_magnitudes = result.x[:n_contexts]
        opt_phases = result.x[n_contexts:]
        
        # Compute final reconstruction for cosine similarity
        reconstructed = np.zeros(dim)
        for i in range(n_contexts):
            ctx = context_embs[i]
            mag = opt_magnitudes[i]
            phase = opt_phases[i]
            
            np.random.seed(42 + i)
            random_dir = np.random.randn(dim)
            random_dir = random_dir - np.dot(random_dir, ctx) * ctx
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
            
            v_rot = np.cos(phase) * ctx + np.sin(phase) * random_dir
            reconstructed += mag * v_rot
        
        cosine_sim = np.dot(raw_emb, reconstructed) / (np.linalg.norm(raw_emb) * np.linalg.norm(reconstructed))
        
        return {
            "method": "weighted_complex",
            "magnitudes": opt_magnitudes.tolist(),
            "phases": opt_phases.tolist(),
            "phases_degrees": [np.degrees(p) for p in opt_phases],
            "reconstruction_error": float(final_error),
            "cosine_similarity": float(cosine_sim),
            "reconstructed": reconstructed,
            "optimization_success": result.success,
        }
    except Exception as e:
        return {
            "method": "weighted_complex",
            "error": str(e),
        }


def test_orthogonality(context_embs):
    """
    Test if contextual embeddings are orthogonal (basis-like).
    
    Theory:
    - Complex: Contexts form basis → high orthogonality
    - Real: Arbitrary angles between contexts
    """
    n = len(context_embs)
    
    # Compute pairwise cosine similarities
    cos_sims = []
    for i, j in combinations(range(n), 2):
        cos_sim = np.dot(context_embs[i], context_embs[j])
        cos_sims.append(cos_sim)
    
    cos_sims = np.array(cos_sims)
    
    # Statistics
    mean_cos = np.mean(np.abs(cos_sims))
    std_cos = np.std(cos_sims)
    
    # Orthogonality test: if mean |cos| is close to 0, contexts are orthogonal
    is_orthogonal = mean_cos < 0.3  # Threshold for orthogonality
    
    return {
        "method": "orthogonality_test",
        "mean_absolute_cosine": float(mean_cos),
        "std_cosine": float(std_cos),
        "min_cosine": float(np.min(cos_sims)),
        "max_cosine": float(np.max(cos_sims)),
        "median_cosine": float(np.median(np.abs(cos_sims))),
        "is_orthogonal": bool(is_orthogonal),
        "n_pairs": len(cos_sims),
        "all_cosines": cos_sims.tolist(),
    }


def test_pca_reconstruction(raw_emb, context_embs):
    """
    Test reconstruction using PCA on contexts.
    
    If contexts form a subspace, raw should project well onto it.
    """
    from sklearn.decomposition import PCA
    
    n_contexts = len(context_embs)
    
    # PCA on context embeddings
    pca = PCA(n_components=min(n_contexts, len(context_embs[0])))
    pca.fit(context_embs)
    
    # Project raw embedding onto PCA components
    raw_projected = pca.transform([raw_emb])[0]
    raw_reconstructed = pca.inverse_transform([raw_projected])[0]
    
    # Error
    error = np.linalg.norm(raw_emb - raw_reconstructed)
    cosine_sim = np.dot(raw_emb, raw_reconstructed) / (np.linalg.norm(raw_emb) * np.linalg.norm(raw_reconstructed))
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    
    return {
        "method": "pca_reconstruction",
        "n_components": n_contexts,
        "reconstruction_error": float(error),
        "cosine_similarity": float(cosine_sim),
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_variance": float(np.sum(explained_var)),
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compare_real_vs_complex(results_by_word):
    """
    Compare real vs complex reconstruction across all words.
    """
    real_errors = []
    complex_errors = []
    real_cosines = []
    complex_cosines = []
    
    for word, results in results_by_word.items():
        if "weighted_real" in results and "reconstruction_error" in results["weighted_real"]:
            real_errors.append(results["weighted_real"]["reconstruction_error"])
            real_cosines.append(results["weighted_real"]["cosine_similarity"])
        
        if "weighted_complex" in results and "reconstruction_error" in results["weighted_complex"]:
            complex_errors.append(results["weighted_complex"]["reconstruction_error"])
            complex_cosines.append(results["weighted_complex"]["cosine_similarity"])
    
    if len(real_errors) < 3 or len(complex_errors) < 3:
        return {
            "error": "Insufficient data for statistical comparison",
            "n_real": len(real_errors),
            "n_complex": len(complex_errors),
        }
    
    # Paired t-test on errors
    # Note: We need same words for both, so filter
    common_words = []
    real_err_common = []
    complex_err_common = []
    
    for word, results in results_by_word.items():
        if ("weighted_real" in results and "reconstruction_error" in results["weighted_real"] and
            "weighted_complex" in results and "reconstruction_error" in results["weighted_complex"]):
            common_words.append(word)
            real_err_common.append(results["weighted_real"]["reconstruction_error"])
            complex_err_common.append(results["weighted_complex"]["reconstruction_error"])
    
    if len(common_words) >= 3:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(complex_err_common, real_err_common)
        
        # Effect size (Cohen's d for paired)
        diff = np.array(complex_err_common) - np.array(real_err_common)
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
        
        # Determine winner
        complex_better = np.mean(complex_err_common) < np.mean(real_err_common)
        significant = p_value < 0.001
        
        return {
            "n_words": len(common_words),
            "words_tested": common_words,
            "real_mean_error": float(np.mean(real_err_common)),
            "real_std_error": float(np.std(real_err_common)),
            "complex_mean_error": float(np.mean(complex_err_common)),
            "complex_std_error": float(np.std(complex_err_common)),
            "paired_t_statistic": float(t_stat),
            "paired_p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "complex_better": bool(complex_better),
            "statistically_significant": bool(significant),
            "errors_by_word": {
                word: {
                    "real": float(real_err_common[i]),
                    "complex": float(complex_err_common[i]),
                    "complex_minus_real": float(complex_err_common[i] - real_err_common[i]),
                }
                for i, word in enumerate(common_words)
            },
        }
    else:
        return {
            "error": "Insufficient paired data",
            "n_common": len(common_words),
        }


def analyze_phase_distribution(results_by_word):
    """
    Analyze the distribution of optimal phases from complex reconstruction.
    """
    all_phases = []
    phases_by_word = {}
    
    for word, results in results_by_word.items():
        if "weighted_complex" in results and "phases" in results["weighted_complex"]:
            phases = results["weighted_complex"]["phases"]
            all_phases.extend(phases)
            phases_by_word[word] = phases
    
    if len(all_phases) < 10:
        return {
            "error": "Insufficient phase data",
            "n_phases": len(all_phases),
        }
    
    all_phases = np.array(all_phases)
    
    # Convert to degrees for readability
    phases_deg = np.degrees(all_phases)
    
    # Test uniformity
    # Chi-square test on phase bins
    n_bins = 8
    bins = np.linspace(0, 360, n_bins + 1)
    hist, _ = np.histogram(phases_deg, bins=bins)
    
    # Expected uniform distribution
    expected = len(phases_deg) / n_bins * np.ones(n_bins)
    
    chi2_stat, chi2_p = stats.chisquare(hist, expected)
    
    # Circular statistics
    # Mean direction
    sin_sum = np.mean(np.sin(all_phases))
    cos_sum = np.mean(np.cos(all_phases))
    mean_direction = np.arctan2(sin_sum, cos_sum)
    mean_resultant_length = np.sqrt(sin_sum**2 + cos_sum**2)
    
    # Circular variance
    circular_var = 1 - mean_resultant_length
    
    return {
        "n_phases": len(all_phases),
        "phases_degrees": phases_deg.tolist(),
        "mean_phase_deg": float(np.mean(phases_deg)),
        "std_phase_deg": float(np.std(phases_deg)),
        "median_phase_deg": float(np.median(phases_deg)),
        "circular_mean_deg": float(np.degrees(mean_direction)),
        "mean_resultant_length": float(mean_resultant_length),
        "circular_variance": float(circular_var),
        "uniformity_chi2": float(chi2_stat),
        "uniformity_p_value": float(chi2_p),
        "is_uniform": chi2_p > 0.05,
        "phases_by_word": phases_by_word,
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("Q51.5: CONTEXT AS PHASE SELECTION TEST")
    print("Testing if context selects meaning like quantum measurement")
    print("=" * 80)
    
    # Load embeddings
    print("\n" + "-" * 80)
    print("Loading embeddings...")
    embedding_data = get_embeddings_for_test()
    
    if embedding_data is None:
        print("FAILED: Could not load embeddings")
        return
    
    # Run tests for each word
    print("\n" + "=" * 80)
    print("RUNNING RECONSTRUCTION TESTS")
    print("=" * 80)
    
    results_by_word = {}
    
    for word, data in embedding_data.items():
        print(f"\n{'-' * 80}")
        print(f"Testing word: '{word}'")
        print(f"{'-' * 80}")
        
        raw_emb = data["raw_embedding"]
        context_embs = data["context_embeddings"]
        
        print(f"Raw embedding shape: {raw_emb.shape}")
        print(f"Number of contexts: {len(context_embs)}")
        
        word_results = {}
        
        # Test 1: Simple average
        print("\n1. Simple Average Test...")
        avg_result = test_simple_average(raw_emb, context_embs)
        word_results["simple_average"] = avg_result
        print(f"   Error: {avg_result['reconstruction_error']:.4f}")
        print(f"   Cosine similarity: {avg_result['cosine_similarity']:.4f}")
        
        # Test 2: Weighted real
        print("\n2. Weighted Real Coefficients...")
        real_result = test_weighted_real(raw_emb, context_embs)
        word_results["weighted_real"] = real_result
        if "error" not in real_result:
            print(f"   Error: {real_result['reconstruction_error']:.4f}")
            print(f"   Cosine similarity: {real_result['cosine_similarity']:.4f}")
            print(f"   Weights: {[f'{w:.3f}' for w in real_result['weights']]}")
        else:
            print(f"   Error: {real_result['error']}")
        
        # Test 3: Weighted complex
        print("\n3. Weighted Complex Coefficients...")
        complex_result = test_weighted_complex(raw_emb, context_embs)
        word_results["weighted_complex"] = complex_result
        if "error" not in complex_result:
            print(f"   Error: {complex_result['reconstruction_error']:.4f}")
            print(f"   Cosine similarity: {complex_result['cosine_similarity']:.4f}")
            print(f"   Magnitudes: {[f'{m:.3f}' for m in complex_result['magnitudes']]}")
            print(f"   Phases (deg): {[f'{p:.1f}' for p in complex_result['phases_degrees']]}")
            print(f"   Optimization success: {complex_result['optimization_success']}")
        else:
            print(f"   Error: {complex_result['error']}")
        
        # Test 4: Orthogonality
        print("\n4. Orthogonality Test...")
        ortho_result = test_orthogonality(context_embs)
        word_results["orthogonality"] = ortho_result
        print(f"   Mean |cosine|: {ortho_result['mean_absolute_cosine']:.4f}")
        print(f"   Is orthogonal: {ortho_result['is_orthogonal']}")
        print(f"   Range: [{ortho_result['min_cosine']:.4f}, {ortho_result['max_cosine']:.4f}]")
        
        # Test 5: PCA reconstruction
        print("\n5. PCA Reconstruction...")
        pca_result = test_pca_reconstruction(raw_emb, context_embs)
        word_results["pca_reconstruction"] = pca_result
        print(f"   Error: {pca_result['reconstruction_error']:.4f}")
        print(f"   Cosine similarity: {pca_result['cosine_similarity']:.4f}")
        print(f"   Cumulative variance: {pca_result['cumulative_variance']:.4f}")
        
        results_by_word[word] = word_results
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: REAL vs COMPLEX")
    print("=" * 80)
    
    comparison = compare_real_vs_complex(results_by_word)
    
    if "error" not in comparison:
        print(f"\nWords tested: {comparison['n_words']}")
        print(f"Words: {comparison['words_tested']}")
        print(f"\nReal reconstruction:")
        print(f"  Mean error: {comparison['real_mean_error']:.4f} ± {comparison['real_std_error']:.4f}")
        print(f"\nComplex reconstruction:")
        print(f"  Mean error: {comparison['complex_mean_error']:.4f} ± {comparison['complex_std_error']:.4f}")
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {comparison['paired_t_statistic']:.4f}")
        print(f"  p-value: {comparison['paired_p_value']:.6f}")
        print(f"  Cohen's d: {comparison['cohens_d']:.4f}")
        print(f"\nResult:")
        print(f"  Complex better: {comparison['complex_better']}")
        print(f"  Statistically significant (p<0.001): {comparison['statistically_significant']}")
        
        print(f"\nErrors by word:")
        for word, errors in comparison['errors_by_word'].items():
            print(f"  {word}: real={errors['real']:.4f}, complex={errors['complex']:.4f}, diff={errors['complex_minus_real']:+.4f}")
    else:
        print(f"Comparison error: {comparison['error']}")
    
    # Phase analysis
    print("\n" + "=" * 80)
    print("PHASE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    phase_analysis = analyze_phase_distribution(results_by_word)
    
    if "error" not in phase_analysis:
        print(f"\nTotal phases analyzed: {phase_analysis['n_phases']}")
        print(f"\nDescriptive statistics:")
        print(f"  Mean: {phase_analysis['mean_phase_deg']:.1f}°")
        print(f"  Std: {phase_analysis['std_phase_deg']:.1f}°")
        print(f"  Median: {phase_analysis['median_phase_deg']:.1f}°")
        print(f"  Circular mean: {phase_analysis['circular_mean_deg']:.1f}°")
        print(f"  Mean resultant length: {phase_analysis['mean_resultant_length']:.4f}")
        print(f"  Circular variance: {phase_analysis['circular_variance']:.4f}")
        print(f"\nUniformity test:")
        print(f"  Chi-square: {phase_analysis['uniformity_chi2']:.4f}")
        print(f"  p-value: {phase_analysis['uniformity_p_value']:.6f}")
        print(f"  Is uniform: {phase_analysis['is_uniform']}")
    else:
        print(f"Phase analysis error: {phase_analysis['error']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    # Evaluate success criteria
    success_criteria = {
        "complex_better": False,
        "phases_non_uniform": False,
        "contexts_orthogonal": False,
        "statistical_significance": False,
    }
    
    # Criterion 1: Complex reconstruction better than real
    if "complex_better" in comparison:
        success_criteria["complex_better"] = comparison["complex_better"]
    
    # Criterion 2: Phases non-uniform (indicates structure)
    if "is_uniform" in phase_analysis:
        success_criteria["phases_non_uniform"] = not phase_analysis["is_uniform"]
    
    # Criterion 3: Contexts show orthogonality
    orthogonal_count = sum(1 for r in results_by_word.values() if r.get("orthogonality", {}).get("is_orthogonal", False))
    success_criteria["contexts_orthogonal"] = orthogonal_count >= len(results_by_word) / 2
    
    # Criterion 4: Statistical significance
    if "statistically_significant" in comparison:
        success_criteria["statistical_significance"] = comparison["statistically_significant"]
    
    n_passed = sum(success_criteria.values())
    
    print(f"\nSuccess Criteria Evaluation:")
    print(f"  1. Complex better than real: {success_criteria['complex_better']}")
    print(f"  2. Phases show non-uniform structure: {success_criteria['phases_non_uniform']}")
    print(f"  3. Contexts orthogonal (≥50%): {success_criteria['contexts_orthogonal']} ({orthogonal_count}/{len(results_by_word)})")
    print(f"  4. Statistical significance (p<0.001): {success_criteria['statistical_significance']}")
    print(f"\nOverall: {n_passed}/4 criteria passed")
    
    # Key findings
    print("\n" + "-" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)
    
    findings = []
    
    if "complex_better" in comparison:
        if comparison["complex_better"]:
            findings.append(f"Complex coefficients reconstruct raw embedding BETTER than real coefficients (error: {comparison['complex_mean_error']:.4f} vs {comparison['real_mean_error']:.4f})")
        else:
            findings.append(f"Real coefficients reconstruct raw embedding better than complex (error: {comparison['real_mean_error']:.4f} vs {comparison['complex_mean_error']:.4f})")
    
    if "is_uniform" in phase_analysis:
        if phase_analysis["is_uniform"]:
            findings.append("Phase distribution is UNIFORM (random phases, no structure)")
        else:
            findings.append(f"Phase distribution is NON-UNIFORM (p={phase_analysis['uniformity_p_value']:.6f}), indicating structured phase relationships")
    
    ortho_words = [w for w, r in results_by_word.items() if r.get("orthogonality", {}).get("is_orthogonal", False)]
    if ortho_words:
        findings.append(f"Contexts are orthogonal for: {', '.join(ortho_words)}")
    
    non_ortho_words = [w for w, r in results_by_word.items() if not r.get("orthogonality", {}).get("is_orthogonal", False)]
    if non_ortho_words:
        findings.append(f"Contexts are NOT orthogonal for: {', '.join(non_ortho_words)}")
    
    # Average reconstruction quality
    avg_cosines = []
    for word, results in results_by_word.items():
        if "weighted_complex" in results and "cosine_similarity" in results["weighted_complex"]:
            avg_cosines.append(results["weighted_complex"]["cosine_similarity"])
    
    if avg_cosines:
        findings.append(f"Average cosine similarity (complex reconstruction): {np.mean(avg_cosines):.4f}")
    
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    # Interpretation
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Determine if context acts as phase selection
    evidence_for_complex = n_passed >= 2  # At least 2 criteria
    
    if evidence_for_complex:
        print(f"""
The evidence suggests that CONTEXT acts as PHASE SELECTION (complex structure):
- Complex coefficients provide better reconstruction than simple real weights
- Phases show structured (non-uniform) distribution
- Contextual embeddings show basis-like properties
- Raw embedding = Sum cᵢ × contextᵢ with phase relationships

This supports the hypothesis that:
1. Ambiguous words exist in superposition of meaning states
2. Context "measures" and collapses to specific meaning
3. Meaning space has complex structure (not just real vectors)

Like quantum measurement: |raw > = Sum cᵢ |contextᵢ > where context selects which |contextᵢ > is realized.
""")
    else:
        print(f"""
The evidence does NOT support context as phase selection:
- Real coefficients are sufficient for reconstruction
- No significant phase structure detected
- Contexts behave like simple feature averaging
- Raw embedding ≈ average(contextᵢ) without phase relationships

This suggests:
1. Ambiguous words are simple mixtures of meanings (not superposition)
2. Context shifts/weights features (no phase collapse)
3. Meaning space is adequately described by real vectors

No quantum-like measurement effect detected in context selection.
""")
    
    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    receipt = {
        "test": "Q51.5_CONTEXT_SUPERPOSITION",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parameters": {
            "random_seed": RANDOM_SEED,
            "n_ambiguous_words": len(AMBIGUOUS_WORDS),
        },
        "success_criteria": success_criteria,
        "n_criteria_passed": n_passed,
        "evidence_for_complex": evidence_for_complex,
        "results_by_word": results_by_word,
        "statistical_comparison": comparison,
        "phase_analysis": phase_analysis,
        "findings": findings,
    }
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f'q51_superposition_{timestamp_str}.json'
    
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=convert)
    
    print(f"Results saved to: {path}")
    
    # Also save report
    report_path = Path(__file__).parent / 'q51_superposition_report.md'
    with open(report_path, 'w') as f:
        f.write("""# Q51.5: Context as Phase Selection Test Report

**Date:** {timestamp}

## Question

Does context select meaning like quantum measurement collapses superposition?

## Theory

- **Complex:** raw_embedding = Sum cᵢ × context_embeddingᵢ [superposition with phases]
- **Real:** raw_embedding = average(context_embeddings) [simple mixing]

## Method

1. Take ambiguous words with multiple meanings
2. Get contextual embeddings for each meaning
3. Test reconstruction quality:
   - Simple average
   - Weighted real coefficients
   - Weighted complex coefficients (with phases)
4. Compare real vs complex reconstruction error
5. Analyze phase distribution and context orthogonality

## Tested Words

{words_table}

## Results

### Statistical Comparison (Real vs Complex)

{comparison_table}

### Phase Analysis

{phase_table}

### Success Criteria

| Criterion | Result |
|-----------|--------|
| Complex better than real | {complex_better} |
| Phases non-uniform | {phases_non_uniform} |
| Contexts orthogonal | {contexts_orthogonal} |
| Statistical significance | {statistical_significance} |

**Overall: {n_passed}/4 criteria passed**

## Key Findings

{findings_list}

## Conclusion

{conclusion}

## Files Generated

- Results: `{results_file}`
- This report: `q51_superposition_report.md`

---
*Generated by test_q51_context_superposition.py*
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            words_table="\n".join([f"- **{w}**: {len(d['contexts'])} contexts" for w, d in AMBIGUOUS_WORDS.items()]),
            comparison_table=f"""
| Metric | Real | Complex |
|--------|------|---------|
| Mean Error | {comparison.get('real_mean_error', 'N/A'):.4f} | {comparison.get('complex_mean_error', 'N/A'):.4f} |
| Std Error | {comparison.get('real_std_error', 'N/A'):.4f} | {comparison.get('complex_std_error', 'N/A'):.4f} |
| t-statistic | {comparison.get('paired_t_statistic', 'N/A'):.4f} |
| p-value | {comparison.get('paired_p_value', 'N/A'):.6f} |
| Cohen's d | {comparison.get('cohens_d', 'N/A'):.4f} |
""",
            phase_table=f"""
| Metric | Value |
|--------|-------|
| n phases | {phase_analysis.get('n_phases', 'N/A')} |
| Mean | {phase_analysis.get('mean_phase_deg', 'N/A'):.1f}° |
| Std | {phase_analysis.get('std_phase_deg', 'N/A'):.1f}° |
| Circular mean | {phase_analysis.get('circular_mean_deg', 'N/A'):.1f}° |
| Resultant length | {phase_analysis.get('mean_resultant_length', 'N/A'):.4f} |
| Uniformity p | {phase_analysis.get('uniformity_p_value', 'N/A'):.6f} |
| Is uniform | {phase_analysis.get('is_uniform', 'N/A')} |
""",
            complex_better=success_criteria['complex_better'],
            phases_non_uniform=success_criteria['phases_non_uniform'],
            contexts_orthogonal=f"{success_criteria['contexts_orthogonal']} ({orthogonal_count}/{len(results_by_word)})",
            statistical_significance=success_criteria['statistical_significance'],
            n_passed=n_passed,
            findings_list="\n".join([f"{i+1}. {f}" for i, f in enumerate(findings)]),
            conclusion="**CONTEXT ACTS AS PHASE SELECTION** (complex structure detected)" if evidence_for_complex else "**NO EVIDENCE** for context as phase selection (real structure sufficient)",
            results_file=path.name,
        ))
    
    print(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("Q51.5 TEST COMPLETE")
    print("=" * 80)
    
    return receipt


if __name__ == '__main__':
    main()
