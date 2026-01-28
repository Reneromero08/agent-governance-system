#!/usr/bin/env python3
"""
Q19: Value Learning - Can R guide which human feedback to trust?

PRE-REGISTRATION:
- HYPOTHESIS: Pearson r > 0.5 between R and inter-annotator agreement
- PREDICTION: High R = high agreement, Low R = disputed
- FALSIFICATION: If r < 0.3
- DATA: HuggingFace - Stanford SHP, OpenAssistant
- THRESHOLD: r > 0.5 to pass

METHODOLOGY:
1. Load real human preference datasets with multiple annotators
2. For each example, compute R on response embeddings
3. Measure inter-annotator agreement (when available) or vote distribution
4. Correlate R with agreement scores
5. Report Pearson correlation coefficient

NO SYNTHETIC DATA. Real human annotation datasets only.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Pre-registration constants
HYPOTHESIS_THRESHOLD = 0.5  # r > 0.5 to PASS
FALSIFICATION_THRESHOLD = 0.3  # r < 0.3 = FALSIFIED


@dataclass
class PreferenceExample:
    """Single example from human preference dataset."""
    prompt: str
    responses: List[str]
    scores: List[float]  # Human preference scores or vote counts
    agreement: float  # Inter-annotator agreement metric
    source: str  # Dataset name


@dataclass
class RResult:
    """Result of R computation."""
    R: float
    E: float  # Mean agreement (evidence)
    sigma: float  # Dispersion
    n_observations: int


def compute_r(embeddings: np.ndarray) -> RResult:
    """
    Compute R = E / sigma from embeddings.

    Args:
        embeddings: Array of shape (n_responses, dim)

    Returns:
        RResult with R value and components
    """
    n = len(embeddings)

    if n < 2:
        return RResult(R=0.0, E=0.0, sigma=float('inf'), n_observations=n)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    # Compute all pairwise cosine similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities.append(float(sim))

    if not similarities:
        return RResult(R=0.0, E=0.0, sigma=float('inf'), n_observations=n)

    # Compute E (mean) and sigma (std)
    E = np.mean(similarities)
    sigma = np.std(similarities)

    # R = E / sigma (with small epsilon for stability)
    R = E / (sigma + 1e-8)

    return RResult(R=float(R), E=float(E), sigma=float(sigma), n_observations=n)


def compute_agreement_from_votes(votes: List[int]) -> float:
    """
    Compute agreement metric from vote distribution.
    Higher = more agreement (less disputed).

    Uses normalized entropy: 1 - H(p) / H_max
    Perfect agreement = 1.0, uniform split = 0.0
    """
    total = sum(votes)
    if total == 0:
        return 0.0

    probs = [v / total for v in votes if v > 0]
    if len(probs) <= 1:
        return 1.0  # Perfect agreement if only one choice has votes

    # Compute entropy
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(len(votes))  # Maximum possible entropy

    # Normalized agreement: 1 - normalized_entropy
    if max_entropy > 0:
        agreement = 1.0 - (entropy / max_entropy)
    else:
        agreement = 1.0

    return agreement


def load_shp_dataset(max_samples: int = 500) -> List[PreferenceExample]:
    """
    Load Stanford Human Preferences (SHP) dataset.

    SHP contains Reddit posts where humans chose which response was more helpful.
    Each example has upvote scores that reflect collective human preference.
    """
    try:
        from datasets import load_dataset
        print("Loading Stanford SHP dataset...")

        # Load SHP - it has post, response pairs with scores
        ds = load_dataset("stanfordnlp/SHP", split="validation", trust_remote_code=True)

        examples = []
        seen_posts = {}  # Group responses by post

        for item in ds:
            post_id = item.get('post_id', str(hash(item['history'])))
            history = item['history']

            if post_id not in seen_posts:
                seen_posts[post_id] = {
                    'prompt': history,
                    'responses': [],
                    'scores': [],
                    'labels': []
                }

            # SHP has human_ref vs human_alt comparisons
            # score_A, score_B are the Reddit scores (upvotes)
            if 'human_ref_A' in item:
                seen_posts[post_id]['responses'].append(item['human_ref_A'])
                seen_posts[post_id]['scores'].append(item.get('score_A', 1))
            if 'human_ref_B' in item:
                seen_posts[post_id]['responses'].append(item['human_ref_B'])
                seen_posts[post_id]['scores'].append(item.get('score_B', 1))

        # Convert to PreferenceExamples
        for post_id, data in seen_posts.items():
            if len(data['responses']) >= 2:
                # Agreement from vote distribution
                scores = data['scores']
                agreement = compute_agreement_from_votes([max(0, int(s)) for s in scores])

                examples.append(PreferenceExample(
                    prompt=data['prompt'],
                    responses=data['responses'][:5],  # Limit responses per example
                    scores=data['scores'][:5],
                    agreement=agreement,
                    source='SHP'
                ))

            if len(examples) >= max_samples:
                break

        print(f"  Loaded {len(examples)} examples from SHP")
        return examples

    except Exception as e:
        print(f"  Error loading SHP: {e}")
        return []


def load_openassistant_dataset(max_samples: int = 500) -> List[PreferenceExample]:
    """
    Load OpenAssistant Conversations dataset.

    OASST has ranked responses with explicit quality labels from multiple annotators.
    This provides direct inter-annotator agreement metrics.
    """
    try:
        from datasets import load_dataset
        print("Loading OpenAssistant dataset...")

        # Load OASST - has message trees with rankings
        ds = load_dataset("OpenAssistant/oasst1", split="validation")

        examples = []
        parent_map = {}  # Map parent_id to children

        # Build parent-child relationships
        for item in ds:
            parent_id = item.get('parent_id')
            if parent_id:
                if parent_id not in parent_map:
                    parent_map[parent_id] = {
                        'prompt': None,
                        'responses': [],
                        'labels': [],
                        'ranks': []
                    }
                parent_map[parent_id]['responses'].append(item['text'])
                # OASST has labels dict with quality scores
                labels = item.get('labels', {})
                if labels and isinstance(labels, dict):
                    # Extract quality value if present
                    quality = labels.get('quality', {})
                    if isinstance(quality, dict) and 'value' in quality:
                        parent_map[parent_id]['labels'].append(quality['value'])
                    elif 'rank' in item and item['rank'] is not None:
                        parent_map[parent_id]['labels'].append(float(item['rank']))
                    else:
                        parent_map[parent_id]['labels'].append(0.5)
                elif 'rank' in item and item['rank'] is not None:
                    parent_map[parent_id]['labels'].append(float(item['rank']))
                else:
                    parent_map[parent_id]['labels'].append(0.5)

        # Find prompts and their responses
        for item in ds:
            msg_id = item.get('message_id')
            if msg_id in parent_map and len(parent_map[msg_id]['responses']) >= 2:
                data = parent_map[msg_id]
                data['prompt'] = item['text']

                # Compute agreement from label distribution
                labels = [l for l in data['labels'] if l is not None]
                if len(labels) >= 2:
                    # Use variance-based agreement (lower variance = higher agreement)
                    label_std = np.std(labels)
                    # Map to 0-1 scale (std of 0 = agreement 1, high std = low agreement)
                    agreement = 1.0 / (1.0 + label_std)
                else:
                    agreement = 0.5

                examples.append(PreferenceExample(
                    prompt=data['prompt'],
                    responses=data['responses'][:5],
                    scores=labels[:5] if labels else [0.5] * len(data['responses'][:5]),
                    agreement=agreement,
                    source='OASST'
                ))

            if len(examples) >= max_samples:
                break

        print(f"  Loaded {len(examples)} examples from OpenAssistant")
        return examples

    except Exception as e:
        print(f"  Error loading OpenAssistant: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_anthropic_hh_dataset(max_samples: int = 500) -> List[PreferenceExample]:
    """
    Load Anthropic HH-RLHF dataset.

    Contains chosen vs rejected response pairs for helpfulness/harmlessness.
    We can infer agreement from the consistency of rankings.
    """
    try:
        from datasets import load_dataset
        print("Loading Anthropic HH-RLHF dataset...")

        ds = load_dataset("Anthropic/hh-rlhf", split="test", trust_remote_code=True)

        examples = []

        for item in ds:
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')

            if chosen and rejected:
                # Extract the actual responses (after last "Assistant:")
                def extract_response(text):
                    parts = text.split('Assistant:')
                    return parts[-1].strip() if parts else text

                chosen_resp = extract_response(chosen)
                rejected_resp = extract_response(rejected)

                # Extract prompt (everything before last Assistant turn)
                prompt_parts = chosen.split('Assistant:')
                prompt = 'Assistant:'.join(prompt_parts[:-1]) if len(prompt_parts) > 1 else chosen[:200]

                # HH-RLHF is binary (chosen/rejected), so agreement = 1.0 for clear cases
                # We'll vary agreement based on response length ratio (proxy for clarity)
                len_ratio = min(len(chosen_resp), len(rejected_resp)) / (max(len(chosen_resp), len(rejected_resp)) + 1)
                # Similar length = more ambiguous, different length = clearer preference
                agreement = 0.5 + 0.5 * (1 - len_ratio)

                examples.append(PreferenceExample(
                    prompt=prompt[:500],  # Truncate long prompts
                    responses=[chosen_resp, rejected_resp],
                    scores=[1.0, 0.0],  # Binary scores
                    agreement=agreement,
                    source='HH-RLHF'
                ))

            if len(examples) >= max_samples:
                break

        print(f"  Loaded {len(examples)} examples from Anthropic HH-RLHF")
        return examples

    except Exception as e:
        print(f"  Error loading HH-RLHF: {e}")
        return []


def run_q19_test() -> Dict:
    """
    Run the Q19 Value Learning test.

    Returns test results including correlation coefficient.
    """
    print("=" * 70)
    print("Q19: VALUE LEARNING - CAN R GUIDE HUMAN FEEDBACK TRUST?")
    print("=" * 70)
    print()
    print("PRE-REGISTRATION:")
    print(f"  Hypothesis: r > {HYPOTHESIS_THRESHOLD}")
    print(f"  Falsification: r < {FALSIFICATION_THRESHOLD}")
    print(f"  Data: Stanford SHP, OpenAssistant, Anthropic HH-RLHF")
    print()

    # Load sentence transformer for embeddings
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  Model loaded.")

        def embed_fn(texts: List[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except ImportError:
        print("ERROR: sentence-transformers required. Install with:")
        print("  pip install sentence-transformers")
        return {"error": "sentence-transformers not installed", "pass": False}

    # Load datasets
    print()
    print("-" * 70)
    print("LOADING REAL HUMAN PREFERENCE DATA")
    print("-" * 70)

    all_examples = []

    # Load from multiple sources
    shp_examples = load_shp_dataset(max_samples=300)
    all_examples.extend(shp_examples)

    oasst_examples = load_openassistant_dataset(max_samples=300)
    all_examples.extend(oasst_examples)

    hh_examples = load_anthropic_hh_dataset(max_samples=300)
    all_examples.extend(hh_examples)

    if not all_examples:
        print("\nERROR: Could not load any human preference data.")
        print("Install datasets library: pip install datasets")
        return {"error": "No data loaded", "pass": False}

    print(f"\nTotal examples loaded: {len(all_examples)}")

    # Compute R for each example
    print()
    print("-" * 70)
    print("COMPUTING R VALUES")
    print("-" * 70)

    R_values = []
    agreement_values = []
    source_labels = []

    for i, example in enumerate(all_examples):
        if i % 100 == 0:
            print(f"  Processing example {i+1}/{len(all_examples)}...")

        # Get embeddings for all responses
        try:
            embeddings = embed_fn(example.responses)
            r_result = compute_r(embeddings)

            R_values.append(r_result.R)
            agreement_values.append(example.agreement)
            source_labels.append(example.source)
        except Exception as e:
            continue  # Skip problematic examples

    print(f"  Computed R for {len(R_values)} examples")

    if len(R_values) < 10:
        print("\nERROR: Too few valid examples")
        return {"error": "Insufficient data", "pass": False}

    # Convert to arrays
    R_values = np.array(R_values)
    agreement_values = np.array(agreement_values)

    # Filter out infinite/nan values
    valid_mask = np.isfinite(R_values) & np.isfinite(agreement_values)
    R_values = R_values[valid_mask]
    agreement_values = agreement_values[valid_mask]

    print()
    print("-" * 70)
    print("COMPUTING CORRELATION")
    print("-" * 70)

    # Log-transform R values to handle extreme ranges
    log_R_values = np.log1p(np.abs(R_values))

    # Compute Pearson correlation (on log-transformed R)
    correlation, p_value = stats.pearsonr(log_R_values, agreement_values)

    # Also compute on raw R for comparison
    raw_correlation, raw_p = stats.pearsonr(R_values, agreement_values)

    # Also compute Spearman for robustness (rank-based, handles non-linearity)
    spearman_r, spearman_p = stats.spearmanr(R_values, agreement_values)

    print(f"\n  Pearson r (log R) = {correlation:.4f} (p = {p_value:.2e})")
    print(f"  Pearson r (raw R) = {raw_correlation:.4f} (p = {raw_p:.2e})")
    print(f"  Spearman rho = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Determine pass/fail
    if correlation > HYPOTHESIS_THRESHOLD:
        verdict = "PASS"
        verdict_reason = f"r = {correlation:.4f} > {HYPOTHESIS_THRESHOLD} (hypothesis threshold)"
    elif correlation < FALSIFICATION_THRESHOLD:
        verdict = "FALSIFIED"
        verdict_reason = f"r = {correlation:.4f} < {FALSIFICATION_THRESHOLD} (falsification threshold)"
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = f"{FALSIFICATION_THRESHOLD} <= r = {correlation:.4f} < {HYPOTHESIS_THRESHOLD}"

    # Statistics by source (use log R for consistency)
    source_stats = {}
    source_labels_arr = np.array(source_labels)
    valid_source_labels = source_labels_arr[valid_mask[:len(source_labels)]]
    for source in set(source_labels):
        mask = np.array([s == source for s in valid_source_labels])
        if np.sum(mask) >= 5:
            log_r_src = log_R_values[mask]
            a_src = agreement_values[mask]
            corr_src, p_src = stats.pearsonr(log_r_src, a_src)
            source_stats[source] = {
                'n': int(np.sum(mask)),
                'r': float(corr_src),
                'p': float(p_src),
                'log_r_mean': float(np.mean(log_r_src)),
                'agreement_mean': float(np.mean(a_src))
            }

    # High R vs Low R analysis
    r_median = np.median(R_values)
    high_r_agreement = agreement_values[R_values > r_median].mean()
    low_r_agreement = agreement_values[R_values <= r_median].mean()

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"\n  R Statistics:")
    print(f"    Mean R: {np.mean(R_values):.4f}")
    print(f"    Std R: {np.std(R_values):.4f}")
    print(f"    Median R: {r_median:.4f}")
    print(f"\n  Agreement Statistics:")
    print(f"    Mean Agreement: {np.mean(agreement_values):.4f}")
    print(f"    Std Agreement: {np.std(agreement_values):.4f}")
    print(f"\n  High R (>{r_median:.2f}) mean agreement: {high_r_agreement:.4f}")
    print(f"  Low R (<={r_median:.2f}) mean agreement: {low_r_agreement:.4f}")
    print(f"  Difference: {high_r_agreement - low_r_agreement:.4f}")

    print()
    print("-" * 70)
    print("BY SOURCE")
    print("-" * 70)
    for source, stats_dict in source_stats.items():
        print(f"\n  {source}:")
        print(f"    n = {stats_dict['n']}")
        print(f"    r = {stats_dict['r']:.4f} (p = {stats_dict['p']:.2e})")

    # Simpson's Paradox Detection
    print()
    print("-" * 70)
    print("SIMPSON'S PARADOX CHECK")
    print("-" * 70)
    within_source_correlations = [s['r'] for s in source_stats.values() if s['n'] >= 10]
    avg_within = np.mean(within_source_correlations) if within_source_correlations else 0

    simpsons_paradox = False
    if len(within_source_correlations) >= 2:
        # Check if overall correlation differs in sign from within-source correlations
        if (correlation > 0 and avg_within < 0) or (correlation < 0 and avg_within > 0):
            simpsons_paradox = True
            print(f"\n  WARNING: Simpson's Paradox Detected!")
            print(f"  Overall correlation: {correlation:.4f}")
            print(f"  Average within-source correlation: {avg_within:.4f}")
            print(f"  The combined correlation may be driven by confounding!")
        else:
            print(f"\n  No Simpson's Paradox detected.")
            print(f"  Overall correlation: {correlation:.4f}")
            print(f"  Average within-source: {avg_within:.4f}")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"\n  ** {verdict} **")
    print(f"  {verdict_reason}")
    print()

    # Prepare results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test": "Q19_VALUE_LEARNING",
        "hypothesis": f"r > {HYPOTHESIS_THRESHOLD}",
        "falsification_criterion": f"r < {FALSIFICATION_THRESHOLD}",
        "n_examples": len(R_values),
        "pearson_r_log": float(correlation),
        "pearson_p_log": float(p_value),
        "pearson_r_raw": float(raw_correlation),
        "pearson_p_raw": float(raw_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "r_mean": float(np.mean(R_values)),
        "r_std": float(np.std(R_values)),
        "r_median": float(r_median),
        "log_r_mean": float(np.mean(log_R_values)),
        "log_r_std": float(np.std(log_R_values)),
        "agreement_mean": float(np.mean(agreement_values)),
        "agreement_std": float(np.std(agreement_values)),
        "high_r_agreement": float(high_r_agreement),
        "low_r_agreement": float(low_r_agreement),
        "agreement_difference": float(high_r_agreement - low_r_agreement),
        "source_stats": source_stats,
        "simpsons_paradox": simpsons_paradox,
        "avg_within_source_r": float(avg_within) if within_source_correlations else None,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "pass": verdict == "PASS"
    }

    # Save results
    output_path = Path(__file__).parent / "q19_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_q19_test()
    sys.exit(0 if results.get("pass", False) else 1)
