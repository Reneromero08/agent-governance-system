#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 4: Does Human Alignment Distort the 8e Conservation Law?

Hypothesis: Instruction-tuned models systematically deviate from 8e
because human preferences distort the "natural" geometry of semantic space.

Test: Compare base models vs their instruction-tuned variants.
- If base ≈ 8e and instruct < 8e → human alignment compresses geometry
- The delta (base - instruct) measures the degree of distortion
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
    vs_8e = (df_alpha - 8 * np.e) / (8 * np.e) * 100  # Signed error

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),  # Positive = above 8e, Negative = below 8e
    }


def main():
    print("=" * 70)
    print("Q50 PART 4: HUMAN ALIGNMENT DISTORTION TEST")
    print("Does instruction-tuning distort the natural 8e geometry?")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_ALIGNMENT_DISTORTION',
        'target': 8 * np.e,
        'comparisons': [],
        'summary': {}
    }

    # Standard test vocabulary
    WORDS = [
        "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
        "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
        "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
        "heart", "eye", "hand", "head", "brain", "blood", "bone",
        "mother", "father", "child", "friend", "king", "queen",
        "love", "hate", "truth", "life", "death", "time", "space", "power",
        "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
        "book", "door", "house", "road", "food", "money", "stone", "gold",
        "light", "shadow", "music", "word", "name", "law",
        "good", "bad", "big", "small", "old", "new", "high", "low",
    ]

    # Instruction-formatted queries (for instruction models)
    INSTRUCTION_QUERIES = [
        "query: What is water?",
        "query: Describe fire",
        "query: What is the earth?",
        "query: Tell me about the sky",
        "query: What is the sun?",
    ] + ["query: " + w for w in WORDS[:30]] + WORDS[30:]

    try:
        from sentence_transformers import SentenceTransformer

        # ============================================================
        # MODEL PAIRS: Base vs Instruction-Tuned
        # ============================================================
        print("\n" + "=" * 60)
        print("COMPARING BASE vs INSTRUCTION-TUNED MODELS")
        print("=" * 60)

        # Model pairs: (base_model, instruct_model, family_name)
        # Note: Some "base" models are already somewhat fine-tuned,
        # but we compare them against their instruction-tuned versions
        model_pairs = [
            # E5 family
            ("intfloat/e5-small-v2", "intfloat/multilingual-e5-small-instruct", "E5-small"),
            ("intfloat/e5-base-v2", "intfloat/multilingual-e5-base-instruct", "E5-base"),

            # BGE family
            ("BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "BGE-small"),  # Same model, different input
            ("BAAI/bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "BGE-base"),

            # Test same model with plain vs instruction input
            ("all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "MiniLM-L6-input"),
            ("all-mpnet-base-v2", "all-mpnet-base-v2", "MPNet-base-input"),
        ]

        all_deltas = []

        for base_model_id, instruct_model_id, family in model_pairs:
            try:
                print(f"\n  Testing {family}...")

                # Load models
                base_model = SentenceTransformer(base_model_id)

                # Test 1: Base model with plain words
                base_embeddings = base_model.encode(WORDS, normalize_embeddings=True)
                base_result = analyze_embeddings(base_embeddings, f"{family}-base")

                # Test 2: Same/instruction model with instruction queries
                if instruct_model_id != base_model_id:
                    instruct_model = SentenceTransformer(instruct_model_id)
                else:
                    instruct_model = base_model

                instruct_embeddings = instruct_model.encode(INSTRUCTION_QUERIES, normalize_embeddings=True)
                instruct_result = analyze_embeddings(instruct_embeddings, f"{family}-instruct")

                # Calculate distortion
                delta = base_result['Df_alpha'] - instruct_result['Df_alpha']
                delta_percent = delta / base_result['Df_alpha'] * 100

                comparison = {
                    'family': family,
                    'base': base_result,
                    'instruct': instruct_result,
                    'delta': float(delta),
                    'delta_percent': float(delta_percent),
                }
                results['comparisons'].append(comparison)
                all_deltas.append(delta)

                print(f"    Base:     Df×α = {base_result['Df_alpha']:.4f} ({base_result['vs_8e_percent']:+.2f}% vs 8e)")
                print(f"    Instruct: Df×α = {instruct_result['Df_alpha']:.4f} ({instruct_result['vs_8e_percent']:+.2f}% vs 8e)")
                print(f"    Delta:    {delta:+.4f} ({delta_percent:+.2f}%)")

                if delta > 0:
                    print(f"    → Instruction input COMPRESSES geometry by {delta_percent:.1f}%")
                else:
                    print(f"    → Instruction input EXPANDS geometry by {abs(delta_percent):.1f}%")

            except Exception as e:
                print(f"    {family} failed: {e}")

        # ============================================================
        # T5 ARCHITECTURE COMPARISON
        # ============================================================
        print("\n" + "=" * 60)
        print("T5 ARCHITECTURE (Encoder-Decoder)")
        print("=" * 60)

        t5_models = [
            ("sentence-transformers/gtr-t5-base", "GTR-T5-base"),
            ("sentence-transformers/sentence-t5-base", "ST5-base"),
        ]

        for model_id, name in t5_models:
            try:
                print(f"\n  Testing {name}...")
                model = SentenceTransformer(model_id)

                # Plain input
                plain_embeddings = model.encode(WORDS, normalize_embeddings=True)
                plain_result = analyze_embeddings(plain_embeddings, f"{name}-plain")

                # Instruction input
                instruct_embeddings = model.encode(INSTRUCTION_QUERIES, normalize_embeddings=True)
                instruct_result = analyze_embeddings(instruct_embeddings, f"{name}-instruct")

                delta = plain_result['Df_alpha'] - instruct_result['Df_alpha']

                print(f"    Plain:    Df×α = {plain_result['Df_alpha']:.4f} ({plain_result['vs_8e_percent']:+.2f}% vs 8e)")
                print(f"    Instruct: Df×α = {instruct_result['Df_alpha']:.4f} ({instruct_result['vs_8e_percent']:+.2f}% vs 8e)")
                print(f"    Delta:    {delta:+.4f}")

                results['comparisons'].append({
                    'family': name,
                    'base': plain_result,
                    'instruct': instruct_result,
                    'delta': float(delta),
                    'architecture': 'encoder-decoder',
                })
                all_deltas.append(delta)

            except Exception as e:
                print(f"    {name} failed: {e}")

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: ALIGNMENT DISTORTION")
        print("=" * 70)

        if all_deltas:
            mean_delta = np.mean(all_deltas)
            positive_deltas = sum(1 for d in all_deltas if d > 0)
            negative_deltas = sum(1 for d in all_deltas if d < 0)

            print(f"\n  Comparisons: {len(all_deltas)}")
            print(f"  Mean delta: {mean_delta:+.4f}")
            print(f"  Compression (delta > 0): {positive_deltas}/{len(all_deltas)}")
            print(f"  Expansion (delta < 0): {negative_deltas}/{len(all_deltas)}")

            if positive_deltas > negative_deltas:
                print(f"\n  FINDING: Instruction/alignment COMPRESSES semantic geometry")
                print(f"           This pushes Df×α below 8e")
            else:
                print(f"\n  FINDING: No consistent compression pattern")

            # Hypothesis test
            hypothesis_supported = positive_deltas > len(all_deltas) / 2 and mean_delta > 0

            results['summary'] = {
                'n_comparisons': len(all_deltas),
                'mean_delta': float(mean_delta),
                'compression_count': positive_deltas,
                'expansion_count': negative_deltas,
                'hypothesis_supported': hypothesis_supported,
            }

            if hypothesis_supported:
                print(f"\n  HYPOTHESIS SUPPORTED:")
                print(f"  Human alignment distorts natural 8e geometry by compressing it.")
                print(f"  Mean compression: {mean_delta:.4f} ({mean_delta/21.746*100:.2f}% of 8e)")

        else:
            print("\n  No valid comparisons collected")
            results['summary'] = {'n_comparisons': 0, 'hypothesis_supported': False}

        # Per-comparison breakdown
        print("\n  Per-comparison results:")
        print(f"  {'Family':<20} {'Base Df×α':<12} {'Instruct Df×α':<15} {'Delta':<12} {'Effect':<15}")
        print("  " + "-" * 74)
        for c in results['comparisons']:
            effect = "COMPRESS" if c['delta'] > 0 else "EXPAND"
            print(f"  {c['family']:<20} {c['base']['Df_alpha']:<12.4f} {c['instruct']['Df_alpha']:<15.4f} {c['delta']:<+12.4f} {effect:<15}")

    except ImportError:
        print("  sentence-transformers not available")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_alignment_distortion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
