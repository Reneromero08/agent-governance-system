#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 7: Map the Alignment Spectrum

Hypothesis: As input becomes more "human-aligned" (instruction-formatted),
Df × α decreases monotonically from 8e toward some floor value.

Test: Same words, increasing instruction intensity.

Levels:
0. Single words ("water")
1. Simple phrases ("the water")
2. Questions ("What is water?")
3. Instructions ("query: What is water?")
4. Elaborate prompts ("Please explain in detail what water is and why it matters")

Pass criteria:
- Monotonic relationship between alignment intensity and compression
- Quantify the floor value
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_df(eigenvalues):
    """Participation ratio Df = (Σλ)² / Σλ²"""
    ev = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
    """Power law decay exponent α where λ_k ~ k^(-α)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings):
    """Get eigenvalues from covariance matrix."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings, name):
    """Compute Df, α, and Df × α for embeddings."""
    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    df_alpha = Df * alpha
    vs_8e = (df_alpha - 8 * np.e) / (8 * np.e) * 100

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),
    }


# Base vocabulary for testing
BASE_WORDS = [
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
    "mother", "father", "child", "friend", "king", "queen", "hero", "teacher",
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "light", "shadow", "music", "word", "name", "law", "art", "science",
    "good", "bad", "big", "small", "old", "new", "high", "low",
]


def create_intensity_levels(words):
    """Create input variations at different alignment intensities."""
    levels = {}

    # Level 0: Single words (natural)
    levels[0] = {
        'name': 'Single words',
        'description': 'Just the word itself',
        'inputs': words
    }

    # Level 1: Simple phrases (mild structure)
    levels[1] = {
        'name': 'Simple phrases',
        'description': 'Article + word',
        'inputs': [f"the {w}" for w in words]
    }

    # Level 2: Definitions (moderate structure)
    levels[2] = {
        'name': 'Definitions',
        'description': 'What is X?',
        'inputs': [f"What is {w}?" for w in words]
    }

    # Level 3: Instruction prefix (high structure)
    levels[3] = {
        'name': 'Query prefix',
        'description': 'query: What is X?',
        'inputs': [f"query: What is {w}?" for w in words]
    }

    # Level 4: Elaborate instruction (very high structure)
    levels[4] = {
        'name': 'Elaborate prompt',
        'description': 'Full instruction format',
        'inputs': [f"Please explain what {w} means and describe its key characteristics" for w in words]
    }

    # Level 5: System-style prompt (maximum structure)
    levels[5] = {
        'name': 'System prompt style',
        'description': 'Highly structured system prompt',
        'inputs': [f"You are a helpful assistant. Task: Define '{w}'. Provide a clear, concise definition." for w in words]
    }

    return levels


def main():
    print("=" * 70)
    print("Q50 PART 7: ALIGNMENT SPECTRUM MAPPING")
    print("How does Df × α change with increasing instruction intensity?")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_ALIGNMENT_SPECTRUM',
        'target': 8 * np.e,
        'models': [],
        'summary': {}
    }

    try:
        from sentence_transformers import SentenceTransformer

        # Models to test
        models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6"),
            ("all-mpnet-base-v2", "MPNet-base"),
            ("BAAI/bge-small-en-v1.5", "BGE-small"),
        ]

        # Create intensity levels
        levels = create_intensity_levels(BASE_WORDS)

        all_model_results = []

        for model_id, model_name in models:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {model_name}")
            print("=" * 60)

            model = SentenceTransformer(model_id)

            model_result = {
                'model': model_name,
                'model_id': model_id,
                'levels': []
            }

            print(f"\n  {'Level':<6} {'Intensity':<25} {'Df×α':<10} {'vs 8e':<12} {'Compression':<12}")
            print("  " + "-" * 70)

            level_0_value = None

            for level_num in sorted(levels.keys()):
                level = levels[level_num]

                # Encode inputs
                embeddings = model.encode(level['inputs'], normalize_embeddings=True)
                result = analyze_embeddings(embeddings, level['name'])

                if level_num == 0:
                    level_0_value = result['Df_alpha']

                compression = 0
                if level_0_value and level_0_value > 0:
                    compression = (level_0_value - result['Df_alpha']) / level_0_value * 100

                level_result = {
                    'level': level_num,
                    'name': level['name'],
                    'description': level['description'],
                    'Df': result['Df'],
                    'alpha': result['alpha'],
                    'Df_alpha': result['Df_alpha'],
                    'vs_8e_percent': result['vs_8e_percent'],
                    'compression_from_level_0': compression
                }
                model_result['levels'].append(level_result)

                # Print row
                vs_8e_str = f"{result['vs_8e_percent']:+.1f}%"
                comp_str = f"{compression:+.1f}%" if level_num > 0 else "baseline"
                print(f"  {level_num:<6} {level['name']:<25} {result['Df_alpha']:<10.2f} {vs_8e_str:<12} {comp_str:<12}")

            # Check monotonicity
            df_alphas = [l['Df_alpha'] for l in model_result['levels']]
            is_monotonic_decreasing = all(df_alphas[i] >= df_alphas[i+1] for i in range(len(df_alphas)-1))

            model_result['is_monotonic_decreasing'] = is_monotonic_decreasing
            model_result['floor_value'] = min(df_alphas)
            model_result['ceiling_value'] = max(df_alphas)
            model_result['total_compression'] = (max(df_alphas) - min(df_alphas)) / max(df_alphas) * 100

            print(f"\n  Monotonic decreasing: {is_monotonic_decreasing}")
            print(f"  Ceiling (natural): {max(df_alphas):.2f}")
            print(f"  Floor (aligned): {min(df_alphas):.2f}")
            print(f"  Total compression: {model_result['total_compression']:.1f}%")

            results['models'].append(model_result)
            all_model_results.append(model_result)

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: ALIGNMENT SPECTRUM")
        print("=" * 70)

        # Aggregate
        mean_floor = np.mean([m['floor_value'] for m in all_model_results])
        mean_ceiling = np.mean([m['ceiling_value'] for m in all_model_results])
        mean_compression = np.mean([m['total_compression'] for m in all_model_results])
        monotonic_count = sum(1 for m in all_model_results if m['is_monotonic_decreasing'])

        print(f"\n  Models tested: {len(all_model_results)}")
        print(f"  Monotonic decreasing: {monotonic_count}/{len(all_model_results)}")
        print(f"\n  Mean ceiling (Level 0): {mean_ceiling:.2f} ({(mean_ceiling - 8*np.e)/(8*np.e)*100:+.1f}% vs 8e)")
        print(f"  Mean floor (Level 5): {mean_floor:.2f} ({(mean_floor - 8*np.e)/(8*np.e)*100:+.1f}% vs 8e)")
        print(f"  Mean total compression: {mean_compression:.1f}%")

        # Per-level summary
        print("\n  Per-level averages:")
        for level_num in sorted(levels.keys()):
            level_values = [m['levels'][level_num]['Df_alpha'] for m in all_model_results]
            mean_val = np.mean(level_values)
            vs_8e = (mean_val - 8*np.e) / (8*np.e) * 100
            print(f"    Level {level_num} ({levels[level_num]['name']:<20}): {mean_val:.2f} ({vs_8e:+.1f}% vs 8e)")

        results['summary'] = {
            'mean_floor': float(mean_floor),
            'mean_ceiling': float(mean_ceiling),
            'mean_compression_percent': float(mean_compression),
            'monotonic_count': monotonic_count,
            'floor_vs_8e_percent': float((mean_floor - 8*np.e)/(8*np.e)*100),
            'ceiling_vs_8e_percent': float((mean_ceiling - 8*np.e)/(8*np.e)*100),
        }

        # Verdict
        print("\n" + "=" * 70)
        if monotonic_count == len(all_model_results):
            print("VERDICT: MONOTONIC COMPRESSION CONFIRMED")
        elif monotonic_count > len(all_model_results) / 2:
            print("VERDICT: MONOTONIC COMPRESSION PARTIALLY CONFIRMED")
        else:
            print("VERDICT: MONOTONIC COMPRESSION NOT CONFIRMED")

        print(f"\nAlignment spectrum: {mean_ceiling:.2f} (natural) → {mean_floor:.2f} (aligned)")
        print(f"Compression range: {mean_compression:.1f}%")
        print("=" * 70)

    except ImportError as e:
        print(f"  Import error: {e}")
        results['error'] = str(e)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_alignment_spectrum_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
