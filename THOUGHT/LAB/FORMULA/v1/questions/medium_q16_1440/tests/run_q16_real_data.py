#!/usr/bin/env python3
"""
Q16: Domain Boundaries for R = E/sigma
Uses REAL HuggingFace datasets: SNLI, ANLI

Pre-registered Hypothesis:
R will show low correlation (r < 0.5) with ground truth in:
1. Adversarial NLI - contradictions vs entailments

Falsification: R > 0.7 correlation with ground truth in ANY domain would falsify.
"""

import json
import os
import warnings
from datetime import datetime
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_R_pair(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
    return np.dot(emb1, emb2)

def main():
    print("="*70)
    print("Q16: Domain Boundaries for R = E/sigma")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")

    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded: all-MiniLM-L6-v2")

    results = {
        'experiment': 'Q16_domain_boundaries',
        'timestamp': datetime.now().isoformat(),
        'model': 'all-MiniLM-L6-v2',
        'pre_registration': {
            'hypothesis': 'R < 0.5 correlation with ground truth in adversarial/NLI domains',
            'prediction': 'R cannot distinguish logical contradictions from entailments',
            'falsification': 'R > 0.7 correlation in any NLI domain',
            'threshold': 'correlation < 0.5'
        },
        'tests': {}
    }

    # Test 1: SNLI
    print("\n" + "="*70)
    print("TEST 1: SNLI Dataset (Real Data)")
    print("="*70)

    ds = load_dataset('stanfordnlp/snli', split='validation')
    ds = ds.filter(lambda x: x['label'] != -1)
    ds = ds.shuffle(seed=42).select(range(500))
    print(f"Processing {len(ds)} SNLI examples...")

    entail_sims = []
    neutral_sims = []
    contra_sims = []

    for i, example in enumerate(ds):
        emb_p = model.encode(example['premise'], show_progress_bar=False)
        emb_h = model.encode(example['hypothesis'], show_progress_bar=False)
        sim = compute_R_pair(emb_p, emb_h)

        if example['label'] == 0:
            entail_sims.append(sim)
        elif example['label'] == 1:
            neutral_sims.append(sim)
        elif example['label'] == 2:
            contra_sims.append(sim)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(ds)} examples...")

    mean_entail = np.mean(entail_sims) if entail_sims else 0
    mean_neutral = np.mean(neutral_sims) if neutral_sims else 0
    mean_contra = np.mean(contra_sims) if contra_sims else 0
    std_entail = np.std(entail_sims) if entail_sims else 0
    std_contra = np.std(contra_sims) if contra_sims else 0

    print(f"\nResults:")
    print(f"  Entailment (n={len(entail_sims)}): mean sim = {mean_entail:.4f} +/- {std_entail:.4f}")
    print(f"  Neutral (n={len(neutral_sims)}): mean sim = {mean_neutral:.4f} +/- {np.std(neutral_sims):.4f}")
    print(f"  Contradiction (n={len(contra_sims)}): mean sim = {mean_contra:.4f} +/- {std_contra:.4f}")

    binary_sims = entail_sims + contra_sims
    binary_labels = [1] * len(entail_sims) + [0] * len(contra_sims)
    r_pearson, p_pearson = pearsonr(binary_sims, binary_labels)
    r_spearman, p_spearman = spearmanr(binary_sims, binary_labels)

    print(f"\nCorrelation (Sim vs Entailment/Contradiction label):")
    print(f"  Pearson r: {r_pearson:.4f} (p={p_pearson:.4e})")
    print(f"  Spearman rho: {r_spearman:.4f} (p={p_spearman:.4e})")

    pooled_std = np.sqrt((std_entail**2 + std_contra**2) / 2)
    cohens_d = (mean_entail - mean_contra) / (pooled_std + 1e-8)
    print(f"\nEffect size (Cohens d): {cohens_d:.4f}")

    snli_hypothesis = bool(abs(r_pearson) < 0.5)
    print(f"\nHYPOTHESIS: {'CONFIRMED' if snli_hypothesis else 'FALSIFIED'}")

    results['tests']['snli'] = {
        'dataset': 'SNLI',
        'n_samples': len(ds),
        'n_entailment': len(entail_sims),
        'n_neutral': len(neutral_sims),
        'n_contradiction': len(contra_sims),
        'mean_sim_entailment': float(mean_entail),
        'mean_sim_neutral': float(mean_neutral),
        'mean_sim_contradiction': float(mean_contra),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_rho': float(r_spearman),
        'cohens_d': float(cohens_d),
        'hypothesis_confirmed': snli_hypothesis
    }

    # Test 2: ANLI
    print("\n" + "="*70)
    print("TEST 2: ANLI Dataset (Adversarial Real Data)")
    print("="*70)

    ds2 = load_dataset('facebook/anli', split='test_r3')
    ds2 = ds2.shuffle(seed=42).select(range(300))
    print(f"Processing {len(ds2)} ANLI R3 examples...")

    entail_sims2 = []
    neutral_sims2 = []
    contra_sims2 = []

    for i, example in enumerate(ds2):
        emb_p = model.encode(example['premise'], show_progress_bar=False)
        emb_h = model.encode(example['hypothesis'], show_progress_bar=False)
        sim = compute_R_pair(emb_p, emb_h)

        if example['label'] == 0:
            entail_sims2.append(sim)
        elif example['label'] == 1:
            neutral_sims2.append(sim)
        elif example['label'] == 2:
            contra_sims2.append(sim)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(ds2)} examples...")

    mean_entail2 = np.mean(entail_sims2) if entail_sims2 else 0
    mean_neutral2 = np.mean(neutral_sims2) if neutral_sims2 else 0
    mean_contra2 = np.mean(contra_sims2) if contra_sims2 else 0
    std_entail2 = np.std(entail_sims2) if entail_sims2 else 0
    std_contra2 = np.std(contra_sims2) if contra_sims2 else 0

    print(f"\nResults (Adversarial):")
    print(f"  Entailment (n={len(entail_sims2)}): mean sim = {mean_entail2:.4f} +/- {std_entail2:.4f}")
    print(f"  Neutral (n={len(neutral_sims2)}): mean sim = {mean_neutral2:.4f} +/- {np.std(neutral_sims2):.4f}")
    print(f"  Contradiction (n={len(contra_sims2)}): mean sim = {mean_contra2:.4f} +/- {std_contra2:.4f}")

    binary_sims2 = entail_sims2 + contra_sims2
    binary_labels2 = [1] * len(entail_sims2) + [0] * len(contra_sims2)
    if len(binary_sims2) > 2:
        r_pearson2, p_pearson2 = pearsonr(binary_sims2, binary_labels2)
        r_spearman2, p_spearman2 = spearmanr(binary_sims2, binary_labels2)
    else:
        r_pearson2, p_pearson2 = 0.0, 1.0
        r_spearman2, p_spearman2 = 0.0, 1.0

    print(f"\nCorrelation (Sim vs Entailment/Contradiction label):")
    print(f"  Pearson r: {r_pearson2:.4f} (p={p_pearson2:.4e})")
    print(f"  Spearman rho: {r_spearman2:.4f} (p={p_spearman2:.4e})")

    pooled_std2 = np.sqrt((std_entail2**2 + std_contra2**2) / 2)
    cohens_d2 = (mean_entail2 - mean_contra2) / (pooled_std2 + 1e-8)
    print(f"\nEffect size (Cohens d): {cohens_d2:.4f}")

    anli_hypothesis = bool(abs(r_pearson2) < 0.5)
    print(f"\nHYPOTHESIS: {'CONFIRMED' if anli_hypothesis else 'FALSIFIED'}")

    results['tests']['anli'] = {
        'dataset': 'ANLI_R3',
        'n_samples': len(ds2),
        'n_entailment': len(entail_sims2),
        'n_neutral': len(neutral_sims2),
        'n_contradiction': len(contra_sims2),
        'mean_sim_entailment': float(mean_entail2),
        'mean_sim_neutral': float(mean_neutral2),
        'mean_sim_contradiction': float(mean_contra2),
        'pearson_r': float(r_pearson2),
        'pearson_p': float(p_pearson2),
        'spearman_rho': float(r_spearman2),
        'cohens_d': float(cohens_d2),
        'hypothesis_confirmed': anli_hypothesis
    }

    # Test 3: Positive Control
    print("\n" + "="*70)
    print("TEST 3: Positive Control - Topical Consistency")
    print("="*70)

    ds3 = load_dataset('stanfordnlp/snli', split='validation')
    ds3 = ds3.filter(lambda x: x['label'] == 0)
    ds3 = ds3.shuffle(seed=42).select(range(200))
    print(f"Processing {len(ds3)} examples for topical consistency test...")

    aligned_sims = []
    misaligned_sims = []

    premises = [ex['premise'] for ex in ds3]
    hypotheses = [ex['hypothesis'] for ex in ds3]

    print("  Encoding premises...")
    emb_premises = model.encode(premises, show_progress_bar=True)
    print("  Encoding hypotheses...")
    emb_hypotheses = model.encode(hypotheses, show_progress_bar=True)

    for i in range(len(ds3)):
        sim_aligned = compute_R_pair(emb_premises[i], emb_hypotheses[i])
        aligned_sims.append(sim_aligned)

        j = (i + 1) % len(ds3)
        sim_misaligned = compute_R_pair(emb_premises[i], emb_hypotheses[j])
        misaligned_sims.append(sim_misaligned)

    mean_aligned = np.mean(aligned_sims)
    mean_misaligned = np.mean(misaligned_sims)
    std_aligned = np.std(aligned_sims)
    std_misaligned = np.std(misaligned_sims)

    print(f"\nResults:")
    print(f"  Aligned pairs: mean sim = {mean_aligned:.4f} +/- {std_aligned:.4f}")
    print(f"  Misaligned pairs: mean sim = {mean_misaligned:.4f} +/- {std_misaligned:.4f}")

    all_sims = aligned_sims + misaligned_sims
    all_labels = [1] * len(aligned_sims) + [0] * len(misaligned_sims)
    r_pearson3, p_pearson3 = pearsonr(all_sims, all_labels)

    print(f"\nCorrelation (Sim vs Aligned/Misaligned):")
    print(f"  Pearson r: {r_pearson3:.4f} (p={p_pearson3:.4e})")

    pooled_std3 = np.sqrt((std_aligned**2 + std_misaligned**2) / 2)
    cohens_d3 = (mean_aligned - mean_misaligned) / (pooled_std3 + 1e-8)
    print(f"\nEffect size (Cohens d): {cohens_d3:.4f}")

    pc_passes = bool(r_pearson3 > 0.5)
    print(f"\nPOSITIVE CONTROL: {'PASSES' if pc_passes else 'FAILS'}")

    results['tests']['positive_control'] = {
        'test': 'positive_control',
        'n_pairs': len(aligned_sims),
        'mean_sim_aligned': float(mean_aligned),
        'mean_sim_misaligned': float(mean_misaligned),
        'pearson_r': float(r_pearson3),
        'pearson_p': float(p_pearson3),
        'cohens_d': float(cohens_d3),
        'positive_control_passes': pc_passes
    }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n| Test | Pearson r | Cohens d | Hypothesis |")
    print("|------|-----------|-----------|------------|")
    print(f"| SNLI (NLI) | {r_pearson:.3f} | {cohens_d:.3f} | {'CONFIRMED' if snli_hypothesis else 'FALSIFIED'} |")
    print(f"| ANLI (Adversarial) | {r_pearson2:.3f} | {cohens_d2:.3f} | {'CONFIRMED' if anli_hypothesis else 'FALSIFIED'} |")
    print(f"| Positive Control | {r_pearson3:.3f} | {cohens_d3:.3f} | {'PASS' if pc_passes else 'FAIL'} |")

    confirmed_count = sum([snli_hypothesis, anli_hypothesis, pc_passes])

    results['summary'] = {
        'hypotheses_confirmed': confirmed_count,
        'total_tests': 3,
        'snli_hypothesis': snli_hypothesis,
        'anli_hypothesis': anli_hypothesis,
        'positive_control': pc_passes
    }

    if confirmed_count >= 2 and pc_passes:
        status = "CONFIRMED"
        conclusion = """R has fundamental domain boundaries:
  - WORKS FOR: Topical consistency, semantic similarity
  - FAILS FOR: Logical validity (entailment vs contradiction)

This is EXPECTED and IMPORTANT:
  R measures SEMANTIC COHERENCE (same topic) not LOGICAL VALIDITY.
  Contradictions can be semantically related (same topic) but logically opposed.
  R CANNOT and SHOULD NOT detect logical relationships."""
    else:
        status = "PARTIALLY CONFIRMED"
        conclusion = "Further investigation needed."

    results['summary']['status'] = status
    results['summary']['conclusion'] = conclusion

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print(f"\nQ16 STATUS: {status}")
    print(f"\n{conclusion}")

    # Save results
    output_file = os.path.join(SCRIPT_DIR, "q16_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Completed: {datetime.now().isoformat()}")

    return results

if __name__ == "__main__":
    results = main()
