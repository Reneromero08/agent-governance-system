#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 6: Validate Peircean PC-to-Category Mapping

Hypothesis: The top 3 PCs correspond to Peirce's semiotic categories:
- PC1 = Secondness (Concrete ↔ Abstract)
- PC2 = Firstness (Positive ↔ Negative)
- PC3 = Thirdness (Agent ↔ Patient)

Test: Use curated word lists and check separation accuracy.

Pass criteria:
- Each PC separates its category with > 80% accuracy
- Effect size (Cohen's d) > 1.0 for each axis
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# CURATED WORD LISTS (Peircean Categories)
# ============================================================

# Secondness: Brute existence, resistance, concrete vs abstract
CONCRETE = [
    "rock", "table", "dog", "water", "hand", "tree", "car", "house",
    "stone", "bread", "shoe", "chair", "hammer", "apple", "fish", "bird",
    "mountain", "river", "sun", "moon", "fire", "ice", "sand", "metal",
    "glass", "wood", "bone", "blood", "skin", "eye", "flower", "grass"
]

ABSTRACT = [
    "love", "justice", "freedom", "truth", "hope", "idea", "thought", "dream",
    "beauty", "wisdom", "courage", "faith", "honor", "peace", "time", "space",
    "memory", "reason", "doubt", "fear", "joy", "hate", "pride", "shame",
    "knowledge", "belief", "value", "meaning", "purpose", "fate", "soul", "mind"
]

# Firstness: Pure quality, feeling, valence
POSITIVE = [
    "joy", "beauty", "peace", "gift", "friend", "sunshine", "love", "hope",
    "smile", "warm", "gentle", "bright", "sweet", "soft", "calm", "kind",
    "happy", "blessed", "good", "wonderful", "pleasant", "delight", "harmony", "grace",
    "comfort", "treasure", "paradise", "angel", "miracle", "bliss", "glory", "triumph"
]

NEGATIVE = [
    "pain", "ugliness", "war", "poison", "enemy", "darkness", "hate", "fear",
    "frown", "cold", "harsh", "dim", "bitter", "hard", "storm", "cruel",
    "sad", "cursed", "evil", "terrible", "unpleasant", "horror", "discord", "disgrace",
    "torment", "trash", "hell", "demon", "disaster", "agony", "shame", "defeat"
]

# Thirdness: Mediation, law, agency
AGENT = [
    "hero", "hunter", "teacher", "leader", "creator", "judge", "king", "warrior",
    "builder", "speaker", "writer", "driver", "player", "singer", "doctor", "master",
    "commander", "director", "founder", "inventor", "pioneer", "champion", "guardian", "ruler",
    "predator", "attacker", "giver", "sender", "author", "designer", "maker", "actor"
]

PATIENT = [
    "victim", "prey", "student", "follower", "creation", "defendant", "servant", "captive",
    "product", "listener", "reader", "passenger", "audience", "patient", "slave", "apprentice",
    "subject", "recipient", "disciple", "target", "object", "sufferer", "prisoner", "dependent",
    "wounded", "defender", "receiver", "recipient", "child", "artifact", "outcome", "effect"
]


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def test_axis_separation(embeddings1, embeddings2, pc_axis, label1, label2):
    """Test if a PC axis separates two word groups."""
    # Get PC scores for each group
    scores1 = embeddings1[:, pc_axis]
    scores2 = embeddings2[:, pc_axis]

    # Cohen's d
    d = compute_cohens_d(scores1, scores2)

    # Classification accuracy using simple threshold (mean of means)
    threshold = (np.mean(scores1) + np.mean(scores2)) / 2
    if np.mean(scores1) > np.mean(scores2):
        pred1 = scores1 > threshold
        pred2 = scores2 <= threshold
    else:
        pred1 = scores1 < threshold
        pred2 = scores2 >= threshold

    accuracy = (np.sum(pred1) + np.sum(pred2)) / (len(scores1) + len(scores2))

    # Logistic regression accuracy
    X = np.concatenate([scores1.reshape(-1, 1), scores2.reshape(-1, 1)])
    y = np.array([0] * len(scores1) + [1] * len(scores2))
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    lr_accuracy = accuracy_score(y, clf.predict(X))

    return {
        'label1': label1,
        'label2': label2,
        'pc_axis': pc_axis,
        'mean1': float(np.mean(scores1)),
        'mean2': float(np.mean(scores2)),
        'std1': float(np.std(scores1)),
        'std2': float(np.std(scores2)),
        'cohens_d': float(abs(d)),
        'threshold_accuracy': float(accuracy),
        'lr_accuracy': float(lr_accuracy),
        'separation_direction': 'positive' if np.mean(scores1) > np.mean(scores2) else 'negative'
    }


def main():
    print("=" * 70)
    print("Q50 PART 6: PEIRCEAN PC-TO-CATEGORY VALIDATION")
    print("Testing if PCs correspond to Firstness, Secondness, Thirdness")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_PC_VALIDATION',
        'hypothesis': {
            'PC1': 'Secondness (Concrete ↔ Abstract)',
            'PC2': 'Firstness (Positive ↔ Negative)',
            'PC3': 'Thirdness (Agent ↔ Patient)'
        },
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

        all_results = []

        for model_id, model_name in models:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {model_name}")
            print("=" * 60)

            model = SentenceTransformer(model_id)

            # Embed all word lists
            concrete_emb = model.encode(CONCRETE, normalize_embeddings=True)
            abstract_emb = model.encode(ABSTRACT, normalize_embeddings=True)
            positive_emb = model.encode(POSITIVE, normalize_embeddings=True)
            negative_emb = model.encode(NEGATIVE, normalize_embeddings=True)
            agent_emb = model.encode(AGENT, normalize_embeddings=True)
            patient_emb = model.encode(PATIENT, normalize_embeddings=True)

            # Combine all embeddings for PCA
            all_words = CONCRETE + ABSTRACT + POSITIVE + NEGATIVE + AGENT + PATIENT
            all_emb = np.vstack([
                concrete_emb, abstract_emb,
                positive_emb, negative_emb,
                agent_emb, patient_emb
            ])

            # Fit PCA on all words
            pca = PCA(n_components=10)
            pca.fit(all_emb)

            # Transform each group
            concrete_pc = pca.transform(concrete_emb)
            abstract_pc = pca.transform(abstract_emb)
            positive_pc = pca.transform(positive_emb)
            negative_pc = pca.transform(negative_emb)
            agent_pc = pca.transform(agent_emb)
            patient_pc = pca.transform(patient_emb)

            model_results = {
                'model': model_name,
                'model_id': model_id,
                'n_words_per_category': len(CONCRETE),
                'variance_explained': [float(v) for v in pca.explained_variance_ratio_[:5]],
                'axes': {}
            }

            # ============================================================
            # Test each PC against each category pair
            # ============================================================
            print("\n  Testing all PC-Category combinations...")

            category_pairs = [
                (concrete_pc, abstract_pc, "Concrete", "Abstract", "Secondness"),
                (positive_pc, negative_pc, "Positive", "Negative", "Firstness"),
                (agent_pc, patient_pc, "Agent", "Patient", "Thirdness"),
            ]

            # For each category pair, find which PC separates best
            for emb1, emb2, label1, label2, category in category_pairs:
                print(f"\n  {category} ({label1} vs {label2}):")
                best_pc = -1
                best_d = 0
                best_result = None

                for pc in range(5):  # Test first 5 PCs
                    result = test_axis_separation(emb1, emb2, pc, label1, label2)
                    if result['cohens_d'] > best_d:
                        best_d = result['cohens_d']
                        best_pc = pc
                        best_result = result

                print(f"    Best PC: PC{best_pc + 1}")
                print(f"    Cohen's d: {best_d:.3f}")
                print(f"    Threshold Accuracy: {best_result['threshold_accuracy']:.1%}")
                print(f"    LR Accuracy: {best_result['lr_accuracy']:.1%}")

                model_results['axes'][category] = {
                    'best_pc': best_pc,
                    'results': best_result
                }

            # ============================================================
            # Test the HYPOTHESIZED mapping specifically
            # ============================================================
            print("\n  Testing HYPOTHESIZED mapping (PC1=Secondness, PC2=Firstness, PC3=Thirdness):")

            hypothesized = [
                (concrete_pc, abstract_pc, "Concrete", "Abstract", "Secondness", 0),  # PC1
                (positive_pc, negative_pc, "Positive", "Negative", "Firstness", 1),   # PC2
                (agent_pc, patient_pc, "Agent", "Patient", "Thirdness", 2),           # PC3
            ]

            hypothesis_results = []
            for emb1, emb2, label1, label2, category, expected_pc in hypothesized:
                result = test_axis_separation(emb1, emb2, expected_pc, label1, label2)
                hypothesis_results.append({
                    'category': category,
                    'expected_pc': expected_pc,
                    'cohens_d': result['cohens_d'],
                    'lr_accuracy': result['lr_accuracy'],
                    'passes_80pct': result['lr_accuracy'] >= 0.80,
                    'passes_d_1': result['cohens_d'] >= 1.0
                })

                status = "PASS" if result['lr_accuracy'] >= 0.80 else "FAIL"
                print(f"    PC{expected_pc + 1} → {category}: d={result['cohens_d']:.3f}, acc={result['lr_accuracy']:.1%} [{status}]")

            model_results['hypothesized_mapping'] = hypothesis_results

            # Check if hypothesis holds
            all_pass_80 = all(h['passes_80pct'] for h in hypothesis_results)
            all_pass_d = all(h['passes_d_1'] for h in hypothesis_results)
            model_results['hypothesis_supported'] = all_pass_80
            model_results['all_d_above_1'] = all_pass_d

            print(f"\n  HYPOTHESIS SUPPORTED: {all_pass_80}")
            print(f"  All d > 1.0: {all_pass_d}")

            results['models'].append(model_results)
            all_results.append(model_results)

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: PEIRCEAN MAPPING VALIDATION")
        print("=" * 70)

        # Aggregate across models
        n_models = len(all_results)
        hypothesis_supported_count = sum(1 for r in all_results if r['hypothesis_supported'])

        print(f"\n  Models tested: {n_models}")
        print(f"  Hypothesis supported: {hypothesis_supported_count}/{n_models}")

        # Per-category summary
        for category in ["Secondness", "Firstness", "Thirdness"]:
            accuracies = [r['hypothesized_mapping'][["Secondness", "Firstness", "Thirdness"].index(category)]['lr_accuracy']
                         for r in all_results]
            ds = [r['hypothesized_mapping'][["Secondness", "Firstness", "Thirdness"].index(category)]['cohens_d']
                  for r in all_results]
            print(f"\n  {category}:")
            print(f"    Mean accuracy: {np.mean(accuracies):.1%}")
            print(f"    Mean Cohen's d: {np.mean(ds):.3f}")

        # Find best PC for each category
        print("\n  Best PC per category (across models):")
        for category in ["Secondness", "Firstness", "Thirdness"]:
            best_pcs = [r['axes'][category]['best_pc'] for r in all_results]
            most_common = max(set(best_pcs), key=best_pcs.count)
            print(f"    {category}: PC{most_common + 1} (found in {best_pcs.count(most_common)}/{n_models} models)")

        results['summary'] = {
            'n_models': n_models,
            'hypothesis_supported_count': hypothesis_supported_count,
            'hypothesis_supported_ratio': hypothesis_supported_count / n_models if n_models > 0 else 0
        }

        # Final verdict
        print("\n" + "=" * 70)
        if hypothesis_supported_count == n_models:
            print("VERDICT: PEIRCEAN MAPPING CONFIRMED")
            print("PC1 = Secondness, PC2 = Firstness, PC3 = Thirdness")
        elif hypothesis_supported_count > n_models / 2:
            print("VERDICT: PEIRCEAN MAPPING PARTIALLY SUPPORTED")
            print("Some models show the expected pattern")
        else:
            print("VERDICT: PEIRCEAN MAPPING NOT CONFIRMED")
            print("The PC-to-category assignment may differ from hypothesis")
        print("=" * 70)

    except ImportError as e:
        print(f"  Import error: {e}")
        results['error'] = str(e)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_pc_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
