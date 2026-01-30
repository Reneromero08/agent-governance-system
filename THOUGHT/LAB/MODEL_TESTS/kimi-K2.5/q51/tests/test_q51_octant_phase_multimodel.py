#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q51.2: 8 Octants as Phase Sectors - Multi-Model Comparison

Objective: Test if the 8 octants correspond to 8 phase sectors of width pi/4 each.
Runs on both MiniLM-L6 and MPNet-base for comparison.

Hypothesis: Each octant k corresponds to phase sector theta in [k*pi/4, (k+1)*pi/4) for k = 0..7
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from scipy import stats
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get_octant_mask(pc3, octant_idx):
    """Get mask for points in a specific octant (0-7)."""
    signs = [(octant_idx >> i) & 1 for i in range(3)]
    signs = [1 if s else -1 for s in signs]
    
    mask = np.ones(len(pc3), dtype=bool)
    for dim, sign in enumerate(signs):
        if sign > 0:
            mask &= (pc3[:, dim] >= 0)
        else:
            mask &= (pc3[:, dim] < 0)
    return mask


def compute_phase_angles(pc3):
    """Compute phase angles from first 2 PCs."""
    pc1 = pc3[:, 0]
    pc2 = pc3[:, 1]
    theta = np.arctan2(pc2, pc1)
    return theta


def compute_phase_sectors(theta, n_sectors=8):
    """Assign each point to a phase sector."""
    theta_2pi = np.mod(theta, 2 * np.pi)
    sector_width = 2 * np.pi / n_sectors
    sectors = (theta_2pi / sector_width).astype(int)
    sectors = np.clip(sectors, 0, n_sectors - 1)
    return sectors


def test_phase_periodicity(theta, n_bins=8):
    """Test for n-fold periodicity in phase histogram."""
    counts, bin_edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    expected = np.full(n_bins, len(theta) / n_bins)
    chi2_stat = np.sum((counts - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)
    sector_width = 2 * np.pi / n_bins
    
    return {
        'chi2_statistic': float(chi2_stat),
        'p_value': float(p_value),
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
        'uniform_reject_p01': bool(p_value < 0.01),
        'uniform_reject_p05': bool(p_value < 0.05),
        'sector_width_rad': float(sector_width),
        'sector_width_deg': float(np.degrees(sector_width))
    }


def correlate_octant_phase(pc3, theta, octant_labels):
    """Correlate octant membership with phase sectors."""
    n_sectors = 8
    sectors = compute_phase_sectors(theta, n_sectors)
    
    confusion = np.zeros((8, n_sectors), dtype=int)
    for i in range(len(pc3)):
        confusion[octant_labels[i], sectors[i]] += 1
    
    diagonal_sum = np.sum([confusion[k, k] for k in range(8)])
    total_sum = np.sum(confusion)
    diagonal_fraction = diagonal_sum / total_sum if total_sum > 0 else 0
    
    best_shift = 0
    best_match = 0
    for shift in range(8):
        shifted_confusion = np.roll(confusion, shift, axis=1)
        match = np.sum([shifted_confusion[k, k] for k in range(8)])
        if match > best_match:
            best_match = match
            best_shift = shift
    
    best_shift_fraction = best_match / total_sum if total_sum > 0 else 0
    
    chi2, _, _, _ = stats.chi2_contingency(confusion)
    n = np.sum(confusion)
    cramers_v = np.sqrt(chi2 / (n * (min(8, n_sectors) - 1))) if n > 0 and chi2 > 0 else 0
    
    return {
        'confusion_matrix': confusion.tolist(),
        'diagonal_fraction': float(diagonal_fraction),
        'best_shift': int(best_shift),
        'best_shift_fraction': float(best_shift_fraction),
        'cramers_v': float(cramers_v),
        'octant_sector_correlation': float(best_shift_fraction)
    }


def compute_angular_momentum(pc3, theta):
    """Compute angular momentum L = r x theta."""
    pc1 = pc3[:, 0]
    pc2 = pc3[:, 1]
    r = np.sqrt(pc1**2 + pc2**2)
    L = r * theta
    
    n_bins = 8
    sector_width = 2 * np.pi / n_bins
    theta_2pi = np.mod(theta, 2 * np.pi)
    sector_indices = (theta_2pi / sector_width).astype(int)
    sector_indices = np.clip(sector_indices, 0, n_bins - 1)
    
    L_by_sector = [[] for _ in range(n_bins)]
    for i, sector in enumerate(sector_indices):
        L_by_sector[sector].append(L[i])
    
    L_means = [np.mean(vals) if vals else 0 for vals in L_by_sector]
    L_stds = [np.std(vals) if vals else 0 for vals in L_by_sector]
    
    return {
        'L_mean': float(np.mean(L)),
        'L_std': float(np.std(L)),
        'L_by_sector_mean': [float(x) for x in L_means],
        'L_by_sector_std': [float(x) for x in L_stds],
        'r_mean': float(np.mean(r)),
        'r_std': float(np.std(r))
    }


def analyze_model(model_name, model_id, WORDS):
    """Run full analysis on a single model."""
    from sentence_transformers import SentenceTransformer
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*70}")
    
    print(f"\nLoading {model_id}...")
    model = SentenceTransformer(model_id)
    embeddings = model.encode(WORDS, normalize_embeddings=True)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    print("\nPerforming PCA (3 components)...")
    pca = PCA(n_components=3)
    pc3 = pca.fit_transform(embeddings)
    
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Compute phase angles
    theta = compute_phase_angles(pc3)
    print(f"\nPhase angle statistics:")
    print(f"  Range: [{np.min(theta):.4f}, {np.max(theta):.4f}] rad")
    print(f"  Mean: {np.mean(theta):.4f} rad ({np.degrees(np.mean(theta)):.2f} deg)")
    print(f"  Std: {np.std(theta):.4f} rad ({np.degrees(np.std(theta)):.2f} deg)")
    
    # Test periodicity
    periodicity = test_phase_periodicity(theta, n_bins=8)
    print(f"\nPhase periodicity:")
    print(f"  Chi-square: {periodicity['chi2_statistic']:.4f}, p={periodicity['p_value']:.6f}")
    print(f"  Non-uniform (p<0.05): {periodicity['uniform_reject_p05']}")
    
    # Octant-phase correlation
    octant_labels = np.zeros(len(pc3), dtype=int)
    for octant_idx in range(8):
        mask = get_octant_mask(pc3, octant_idx)
        octant_labels[mask] = octant_idx
    
    correlation = correlate_octant_phase(pc3, theta, octant_labels)
    print(f"\nOctant-phase correlation:")
    print(f"  Best correlation: {correlation['best_shift_fraction']:.4f}")
    print(f"  Cramer's V: {correlation['cramers_v']:.4f}")
    
    # Angular momentum
    ang_mom = compute_angular_momentum(pc3, theta)
    print(f"\nAngular momentum:")
    print(f"  Mean: {ang_mom['L_mean']:.4f}, Std: {ang_mom['L_std']:.4f}")
    
    # Assessment
    octant_phase_r = correlation['best_shift_fraction']
    chi2_p = periodicity['p_value']
    
    criterion_1_met = octant_phase_r > 0.6
    criterion_2_met = chi2_p < 0.01
    
    counts = np.array(periodicity['counts'])
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    peaks_detected = int(np.sum(counts > mean_count + 0.5 * std_count))
    criterion_3_met = peaks_detected >= 4
    
    overall_pass = criterion_1_met and criterion_2_met and criterion_3_met
    
    print(f"\nAssessment:")
    print(f"  Correlation > 0.6: {criterion_1_met} (r={octant_phase_r:.4f})")
    print(f"  Periodicity p < 0.01: {criterion_2_met} (p={chi2_p:.6f})")
    print(f"  Peaks >= 4: {criterion_3_met} ({peaks_detected} peaks)")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    
    return {
        'model': model_name,
        'model_id': model_id,
        'n_words': len(WORDS),
        'embedding_dim': embeddings.shape[1],
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'phase_angles': {
            'mean': float(np.mean(theta)),
            'std': float(np.std(theta)),
            'min': float(np.min(theta)),
            'max': float(np.max(theta))
        },
        'periodicity': periodicity,
        'octant_correlation': correlation,
        'angular_momentum': ang_mom,
        'assessment': {
            'correlation_passed': bool(criterion_1_met),
            'periodicity_passed': bool(criterion_2_met),
            'peaks_passed': bool(criterion_3_met),
            'overall_pass': bool(overall_pass)
        }
    }


def main():
    print("=" * 70)
    print("Q51.2: 8 OCTANTS AS PHASE SECTORS - MULTI-MODEL COMPARISON")
    print("Testing if octants correspond to phase sectors of width pi/4")
    print("=" * 70)
    
    WORDS = [
        "water", "fire", "earth", "air", "sky", "sun", "moon", "star", "mountain",
        "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
        "stone", "sand", "metal", "wood", "ice", "flame", "lightning", "thunder",
        "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
        "wolf", "bear", "eagle", "snake", "insect", "spider", "whale", "shark",
        "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
        "foot", "arm", "leg", "face", "mouth", "ear", "nose", "hair",
        "mother", "father", "child", "friend", "king", "queen", "hero", "villain",
        "teacher", "student", "doctor", "patient", "leader", "follower", "master", "servant",
        "love", "hate", "truth", "life", "death", "time", "space", "power",
        "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
        "memory", "knowledge", "wisdom", "beauty", "courage", "shame", "pride", "anger",
        "book", "door", "house", "road", "food", "money", "gold", "silver",
        "weapon", "shield", "tool", "machine", "vehicle", "clothing", "jewelry", "toy",
        "good", "bad", "big", "small", "old", "new", "high", "low",
        "fast", "slow", "hot", "cold", "bright", "dark", "heavy", "light",
        "smooth", "rough", "clean", "dirty", "strong", "weak", "rich", "poor",
        "run", "walk", "jump", "fly", "swim", "eat", "drink", "sleep",
        "think", "speak", "write", "read", "create", "destroy", "build", "break",
        "beginning", "end", "cause", "effect", "reason", "result", "question", "answer",
        "problem", "solution", "past", "future", "present", "change", "stability", "growth",
        "red", "blue", "green", "yellow", "white", "black", "color", "shade",
        "one", "two", "many", "few", "all", "none", "some", "most",
        "length", "width", "height", "depth", "distance", "area", "volume", "weight",
        "language", "word", "name", "voice", "sound", "silence", "music", "song",
        "story", "history", "culture", "society", "nation", "world", "universe", "god",
    ]
    
    print(f"\nUsing vocabulary of {len(WORDS)} words")
    
    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_OCTANT_PHASE_SECTORS_MULTI_MODEL',
        'hypothesis': 'Octant k corresponds to phase sector [k*pi/4, (k+1)*pi/4)',
        'vocabulary_size': len(WORDS),
        'models': {}
    }
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Run MiniLM-L6
        results['models']['minilm_l6'] = analyze_model(
            'all-MiniLM-L6-v2', 
            'all-MiniLM-L6-v2',
            WORDS
        )
        
        # Run MPNet-base
        results['models']['mpnet_base'] = analyze_model(
            'all-mpnet-base-v2',
            'all-mpnet-base-v2', 
            WORDS
        )
        
        # Cross-model comparison
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON")
        print("=" * 70)
        
        minilm_pass = results['models']['minilm_l6']['assessment']['overall_pass']
        mpnet_pass = results['models']['mpnet_base']['assessment']['overall_pass']
        
        print(f"MiniLM-L6: {'PASS' if minilm_pass else 'FAIL'}")
        print(f"MPNet-base: {'PASS' if mpnet_pass else 'FAIL'}")
        
        if minilm_pass and mpnet_pass:
            print("\nBoth models support the hypothesis.")
        elif minilm_pass or mpnet_pass:
            print("\nMixed results - one model supports, one does not.")
        else:
            print("\nNeither model strongly supports the hypothesis.")
            print("This is an honest negative result.")
        
        # Model agreement
        minilm_corr = results['models']['minilm_l6']['octant_correlation']['best_shift_fraction']
        mpnet_corr = results['models']['mpnet_base']['octant_correlation']['best_shift_fraction']
        
        print(f"\nCorrelation values:")
        print(f"  MiniLM-L6: {minilm_corr:.4f}")
        print(f"  MPNet-base: {mpnet_corr:.4f}")
        print(f"  Difference: {abs(minilm_corr - mpnet_corr):.4f}")
        
        results['cross_model_comparison'] = {
            'minilm_passed': minilm_pass,
            'mpnet_passed': mpnet_pass,
            'minilm_correlation': minilm_corr,
            'mpnet_correlation': mpnet_corr,
            'correlation_difference': float(abs(minilm_corr - mpnet_corr)),
            'consensus': 'both_pass' if (minilm_pass and mpnet_pass) else 
                        'one_pass' if (minilm_pass or mpnet_pass) else 'neither_pass'
        }
        
    except ImportError as e:
        print(f"ERROR: Required package not installed: {e}")
        results['error'] = str(e)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"q51_octant_phase_multimodel_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
