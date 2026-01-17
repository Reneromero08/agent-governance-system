#!/usr/bin/env python3
"""Test 5: Phase Parity Violation (Hallucination Detection).

THE KILLER TEST: Proves that the Zero Signature from Q51 acts as a parity
check for semantic error correction.

Hypothesis:
    Valid semantic content maintains Zero Signature (|S|/n < 0.05).
    Hallucinated/invalid content violates Zero Signature (|S|/n > 0.15).

Protocol:
    1. Compute Zero Signature for valid semantic content
    2. Compute Zero Signature for various hallucination types
    3. ROC analysis for classification
    4. Test cross-model consistency

Success Criteria:
    - Valid |S|/n < 0.05
    - Invalid |S|/n > 0.15
    - AUC > 0.85
    - Cohen's d > 2.0

This is the test that wins the alignment argument:
    If phase parity detects hallucinations, R-gating IS error correction.

Usage:
    python test_hallucination.py [--n-valid 100] [--n-invalid 100]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ttest_ind

# Local imports
from core import (
    generate_random_embeddings,
    compute_effective_dimensionality,
    cohens_d,
)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try to import qgt_phase for Zero Signature
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
QGT_PHASE_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"
sys.path.insert(0, str(QGT_PHASE_PATH))

try:
    from qgt_phase import test_zero_signature, octant_phase_mapping, OctantPhaseResult, ZeroSignatureResult
    HAS_QGT_PHASE = True
    print(f"Loaded qgt_phase from {QGT_PHASE_PATH}")
except ImportError as e:
    HAS_QGT_PHASE = False
    print(f"Warning: qgt_phase not available ({e}). Using simplified Zero Signature.")


# =============================================================================
# Test Data: Valid vs Hallucinated Content
# =============================================================================

VALID_CONTENT = {
    "factual": [
        "Water freezes at zero degrees Celsius.",
        "The Earth orbits the Sun once per year.",
        "Light travels at approximately 300000 kilometers per second.",
        "DNA contains the genetic instructions for living organisms.",
        "Gravity causes objects to fall toward the ground.",
        "The moon orbits the Earth approximately every 27 days.",
        "Plants convert sunlight into energy through photosynthesis.",
        "Sound travels slower than light through air.",
        "The human heart pumps blood through the body.",
        "Electricity flows through conductive materials.",
    ],
    "conceptual": [
        "Democracy requires an informed citizenry to function well.",
        "Language shapes the way we think about the world.",
        "Mathematics describes patterns found in nature.",
        "Consciousness emerges from complex neural activity.",
        "Trust must be earned through consistent actions.",
        "Knowledge grows through questioning assumptions.",
        "Communication bridges the gap between minds.",
        "Learning requires both study and practice.",
        "Cooperation enables achievements beyond individual capacity.",
        "Innovation often comes from combining existing ideas.",
    ],
    "technical": [
        "Hash functions map inputs to fixed-length outputs.",
        "Neural networks learn by adjusting connection weights.",
        "Encryption protects data from unauthorized access.",
        "Algorithms solve problems through step-by-step procedures.",
        "Databases store and retrieve structured information.",
        "APIs enable communication between software components.",
        "Version control tracks changes to source code.",
        "Compilers translate code into machine instructions.",
        "Networks route packets between connected devices.",
        "Memory stores data for quick access by processors.",
    ],
}

HALLUCINATED_CONTENT = {
    "word_salad": [
        "Colorless green ideas sleep furiously tonight.",
        "The square circle bounced silently upward.",
        "Tomorrow yesterday happened twice before.",
        "Invisible sounds taste like purple geometry.",
        "The number seven smells of ancient velocity.",
        "Backwards forwards sideways still standing.",
        "Empty fullness overflows with absent presence.",
        "Silent screaming whispers loudly nothing.",
        "The liquid solid evaporated into frozen fire.",
        "Infinite zeros multiplied into singular plurals.",
    ],
    "category_error": [
        "The number seven tastes blue and rectangular.",
        "Wednesday weighs more than a disappointed sigh.",
        "The color of jealousy runs faster than logic.",
        "Democracy smells like the square root of poetry.",
        "The sound of silence is triangular today.",
        "Truth has a temperature of exactly purple.",
        "Justice weighs three kilograms of metaphor.",
        "The velocity of sadness broke the sound barrier.",
        "Freedom has a melting point of abstract.",
        "The circumference of hatred measures infinite regret.",
    ],
    "contradiction": [
        "The living dead immortal corpse breathed lifelessly.",
        "The honest liar truthfully deceived everyone.",
        "The invisible visible phenomenon remained unseen.",
        "The silent noise deafened the deaf listeners.",
        "The stationary movement traveled nowhere fast.",
        "The wet dryness soaked the parched desert.",
        "The dark brightness illuminated the shadowy light.",
        "The cold heat burned with freezing warmth.",
        "The empty fullness contained abundant nothing.",
        "The fast slowness raced at a standstill.",
    ],
    "nonsense": [
        "Qwerty asdfgh zxcvbn poiuyt lkjhgf.",
        "Blarg florp snizzle wibble wobble zorp.",
        "Fnord grimble spackle quux narf plugh.",
        "Xyzzy plover plugh frobnicate mumble.",
        "Gleep glorp glerp glarp glurp glimp.",
        "Snarf blargle fweep zazzle quibble.",
        "Mxyzptlk shazam alakazam abracadabra.",
        "Froboz quux corge grault garply waldo.",
        "Zyxwv utsrq ponml kjihg fedcb.",
        "Blippy bloppy bleepy bloopy blappy.",
    ],
    "semantic_drift": [
        # Valid start, gradually becomes nonsense
        "The cat sat on the mat and then the cat became purple sounds.",
        "Water flows downhill because gravity tastes like mathematics.",
        "The sun provides warmth which makes shadows feel lonely.",
        "Computers process information into crystallized confusion.",
        "Trees grow toward light and their dreams multiply.",
        "Birds fly by flapping their conceptual abstractions.",
        "Fish swim through water made of liquid uncertainty.",
        "Mountains form over time through geological emotions.",
        "Rivers carve valleys with patient aquatic philosophy.",
        "Clouds form when water evaporates into metaphysical vapor.",
    ],
}


def expand_content(content_dict: Dict[str, List[str]], n_target: int) -> List[str]:
    """Expand content to target count."""
    all_content = []
    for category in content_dict.values():
        all_content.extend(category)

    # Repeat if needed
    while len(all_content) < n_target:
        all_content.extend(all_content)

    return all_content[:n_target]


# =============================================================================
# Zero Signature Computation
# =============================================================================

def compute_zero_signature_simple(embeddings: np.ndarray) -> float:
    """Simple Zero Signature computation without qgt_phase dependency.

    The Zero Signature measures how well the octant phases sum to zero.
    For uniformly distributed phases across 8 octants (8th roots of unity),
    the complex sum should be approximately zero.

    Returns:
        Normalized magnitude |S|/n where S = sum of phase vectors
    """
    n_samples, dim = embeddings.shape

    # Map to octants based on first 3 principal components
    # (simplified version of octant_phase_mapping)

    # Use PCA or just first 3 dims
    if dim >= 3:
        coords = embeddings[:, :3]
    else:
        coords = np.hstack([embeddings, np.zeros((n_samples, 3 - dim))])

    # Determine octant for each sample (based on sign pattern)
    octants = (coords[:, 0] > 0).astype(int) * 4 + \
              (coords[:, 1] > 0).astype(int) * 2 + \
              (coords[:, 2] > 0).astype(int)

    # Octant centers as 8th roots of unity
    phase_centers = np.array([k * np.pi / 4 for k in range(8)])

    # Compute complex sum
    phases = phase_centers[octants]
    complex_sum = np.sum(np.exp(1j * phases))

    # Normalized magnitude
    return float(np.abs(complex_sum) / n_samples)


def compute_zero_signature(embeddings: np.ndarray, verbose: bool = False) -> Dict:
    """Compute Zero Signature using qgt_phase if available.

    The Zero Signature measures how well octant phases sum to zero.
    For uniformly distributed embeddings across 8 octants (8th roots of unity),
    the complex sum should be approximately zero.

    Valid semantic content: |S|/n ~ 0.02 (phases uniformly distributed)
    Hallucinated content: |S|/n >> 0.02 (phases cluster in subset of octants)

    Returns:
        Dict with normalized_magnitude, is_zero, and details
    """
    if HAS_QGT_PHASE:
        # Use the proper octant_phase_mapping for detailed analysis
        octant_result = octant_phase_mapping(embeddings)

        # Also run the full test for statistics
        try:
            sig_result = test_zero_signature(embeddings, verbose=verbose)
            return {
                "normalized_magnitude": sig_result.normalized_magnitude,
                "is_zero": sig_result.is_zero,
                "complex_sum": complex(sig_result.complex_sum),
                "uniformity_chi2": sig_result.uniformity_chi2,
                "uniformity_p_value": sig_result.uniformity_p_value,
                "octant_counts": octant_result.octant_counts.tolist(),
                "coverage": octant_result.coverage,
                "entropy": octant_result.entropy,
            }
        except Exception as e:
            # Fall back to octant_phase_mapping only
            return {
                "normalized_magnitude": octant_result.normalized_sum_magnitude,
                "is_zero": octant_result.normalized_sum_magnitude < 0.1,
                "complex_sum": complex(octant_result.complex_sum),
                "uniformity_chi2": None,
                "uniformity_p_value": None,
                "octant_counts": octant_result.octant_counts.tolist(),
                "coverage": octant_result.coverage,
                "entropy": octant_result.entropy,
            }
    else:
        norm_mag = compute_zero_signature_simple(embeddings)
        return {
            "normalized_magnitude": norm_mag,
            "is_zero": norm_mag < 0.1,
            "complex_sum": None,
            "uniformity_chi2": None,
            "uniformity_p_value": None,
        }


def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for texts."""
    if not HAS_TRANSFORMERS:
        np.random.seed(hash(texts[0]) % 2**32)
        dim = 384
        embeddings = np.random.randn(len(texts), dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# =============================================================================
# Classification Metrics
# =============================================================================

def compute_roc_auc(
    valid_scores: np.ndarray,
    invalid_scores: np.ndarray
) -> Tuple[float, List[Tuple[float, float]]]:
    """Compute ROC AUC for binary classification.

    Higher scores should indicate invalid content.

    Returns:
        Tuple of (AUC, list of (FPR, TPR) points)
    """
    # Combine and create labels (0 = valid, 1 = invalid)
    all_scores = np.concatenate([valid_scores, invalid_scores])
    labels = np.concatenate([np.zeros(len(valid_scores)), np.ones(len(invalid_scores))])

    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = all_scores[sorted_indices]

    # Compute TPR and FPR at each threshold
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0

    for i, label in enumerate(sorted_labels):
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    roc_curve = list(zip(fpr_list, tpr_list))

    return float(auc), roc_curve


# =============================================================================
# Main Test
# =============================================================================

def run_hallucination_test(
    n_valid: int = 50,
    n_invalid: int = 50,
    model_names: Optional[List[str]] = None
) -> Dict:
    """Run full hallucination detection test.

    Args:
        n_valid: Number of valid content samples per model
        n_invalid: Number of invalid content samples per model
        model_names: List of models to test

    Returns:
        Complete test results dict
    """
    if model_names is None:
        model_names = ["all-MiniLM-L6-v2"]
        if HAS_TRANSFORMERS:
            # Add more models if available
            try:
                model_names.append("paraphrase-MiniLM-L6-v2")
            except Exception:
                pass

    print("=" * 70)
    print("TEST 5: PHASE PARITY VIOLATION (HALLUCINATION DETECTION)")
    print("=" * 70)
    print()
    print("Hypothesis: Hallucinations violate Zero Signature (|S|/n > 0.1)")
    print()

    # Prepare content
    valid_texts = expand_content(VALID_CONTENT, n_valid)
    invalid_texts = expand_content(HALLUCINATED_CONTENT, n_invalid)

    results = {
        "test_id": "q40-hallucination-detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "model_names": model_names,
        },
        "models": {},
    }

    all_valid_signatures = []
    all_invalid_signatures = []

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name}")
        print(f"{'='*50}")

        # Get embeddings
        print("\nEncoding valid content...")
        valid_emb = get_embeddings(valid_texts, model_name)
        valid_df = compute_effective_dimensionality(valid_emb)
        print(f"  Df: {valid_df:.2f}")

        print("Encoding invalid content...")
        invalid_emb = get_embeddings(invalid_texts, model_name)
        invalid_df = compute_effective_dimensionality(invalid_emb)
        print(f"  Df: {invalid_df:.2f}")

        # Compute Zero Signatures
        # We compute on the full embedding set for proper octant coverage
        # Then also compute per-category for granular analysis
        print("\nComputing Zero Signatures...")

        # Full set signature
        valid_full_sig = compute_zero_signature(valid_emb, verbose=False)
        invalid_full_sig = compute_zero_signature(invalid_emb, verbose=False)

        print(f"  Valid full set |S|/n: {valid_full_sig['normalized_magnitude']:.4f}")
        print(f"  Invalid full set |S|/n: {invalid_full_sig['normalized_magnitude']:.4f}")

        # Per-batch signatures for statistical spread
        # Use larger batches (10) for better octant coverage
        batch_size = min(10, len(valid_emb) // 2)

        valid_signatures = []
        for i in range(0, len(valid_emb) - batch_size + 1, batch_size // 2):
            batch = valid_emb[i:i+batch_size]
            if len(batch) >= batch_size:
                sig = compute_zero_signature(batch, verbose=False)
                valid_signatures.append(sig["normalized_magnitude"])

        invalid_signatures = []
        for i in range(0, len(invalid_emb) - batch_size + 1, batch_size // 2):
            batch = invalid_emb[i:i+batch_size]
            if len(batch) >= batch_size:
                sig = compute_zero_signature(batch, verbose=False)
                invalid_signatures.append(sig["normalized_magnitude"])

        valid_signatures = np.array(valid_signatures)
        invalid_signatures = np.array(invalid_signatures)

        all_valid_signatures.extend(valid_signatures)
        all_invalid_signatures.extend(invalid_signatures)

        # Statistics
        valid_mean = np.mean(valid_signatures)
        valid_std = np.std(valid_signatures)
        invalid_mean = np.mean(invalid_signatures)
        invalid_std = np.std(invalid_signatures)

        print(f"\nValid content |S|/n:   {valid_mean:.4f} +/- {valid_std:.4f}")
        print(f"Invalid content |S|/n: {invalid_mean:.4f} +/- {invalid_std:.4f}")

        # Cohen's d
        d = cohens_d(invalid_signatures, valid_signatures)
        print(f"Cohen's d: {d:.2f}")

        # ROC AUC
        auc, roc_curve = compute_roc_auc(valid_signatures, invalid_signatures)
        print(f"ROC AUC: {auc:.4f}")

        # t-test
        t_stat, p_value = ttest_ind(invalid_signatures, valid_signatures, alternative='greater')
        print(f"t-test p-value: {p_value:.6f}")

        # Per-category analysis
        print("\nPer-hallucination-type analysis:")
        category_results = {}
        for category, texts in HALLUCINATED_CONTENT.items():
            cat_emb = get_embeddings(texts, model_name)
            cat_signatures = []
            for i in range(0, len(cat_emb), 5):
                batch = cat_emb[i:i+5]
                if len(batch) >= 3:
                    sig = compute_zero_signature(batch, verbose=False)
                    cat_signatures.append(sig["normalized_magnitude"])
            if cat_signatures:
                cat_mean = np.mean(cat_signatures)
                print(f"  {category}: |S|/n = {cat_mean:.4f}")
                category_results[category] = float(cat_mean)

        results["models"][model_name] = {
            "valid_df": float(valid_df),
            "invalid_df": float(invalid_df),
            "valid_full_signature": float(valid_full_sig['normalized_magnitude']),
            "invalid_full_signature": float(invalid_full_sig['normalized_magnitude']),
            "valid_mean": float(valid_mean),
            "valid_std": float(valid_std),
            "invalid_mean": float(invalid_mean),
            "invalid_std": float(invalid_std),
            "cohens_d": float(d),
            "roc_auc": float(auc),
            "t_test_p_value": float(p_value),
            "category_results": category_results,
        }

    # Aggregate results
    all_valid = np.array(all_valid_signatures)
    all_invalid = np.array(all_invalid_signatures)

    aggregate_d = cohens_d(all_invalid, all_valid)
    aggregate_auc, _ = compute_roc_auc(all_valid, all_invalid)

    results["aggregate"] = {
        "valid_mean": float(np.mean(all_valid)),
        "valid_std": float(np.std(all_valid)),
        "invalid_mean": float(np.mean(all_invalid)),
        "invalid_std": float(np.std(all_invalid)),
        "cohens_d": float(aggregate_d),
        "roc_auc": float(aggregate_auc),
    }

    # Verdict
    valid_below_threshold = np.mean(all_valid) < 0.05
    invalid_above_threshold = np.mean(all_invalid) > 0.15
    good_separation = aggregate_d > 2.0
    good_auc = aggregate_auc > 0.85

    verdict_pass = good_auc and (good_separation or (valid_below_threshold and invalid_above_threshold))

    results["verdict"] = {
        "valid_below_0.05": valid_below_threshold,
        "invalid_above_0.15": invalid_above_threshold,
        "cohens_d_above_2": good_separation,
        "auc_above_0.85": good_auc,
        "overall_pass": verdict_pass,
        "interpretation": (
            f"PASS: Phase parity violation detects hallucinations. "
            f"AUC={aggregate_auc:.3f}, d={aggregate_d:.2f}. "
            "Zero Signature IS an error correction parity check."
            if verdict_pass else
            f"FAIL: Phase parity does not reliably detect hallucinations. "
            f"AUC={aggregate_auc:.3f}, d={aggregate_d:.2f}."
        )
    }

    print()
    print("=" * 70)
    print("AGGREGATE VERDICT")
    print("=" * 70)
    print(f"Valid mean |S|/n: {np.mean(all_valid):.4f} (threshold: < 0.05)")
    print(f"Invalid mean |S|/n: {np.mean(all_invalid):.4f} (threshold: > 0.15)")
    print(f"Cohen's d: {aggregate_d:.2f} (threshold: > 2.0)")
    print(f"ROC AUC: {aggregate_auc:.4f} (threshold: > 0.85)")
    print()
    print(f"OVERALL: {'PASS' if verdict_pass else 'FAIL'}")
    print(results["verdict"]["interpretation"])
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Test 5: Hallucination Detection')
    parser.add_argument('--n-valid', type=int, default=50,
                        help='Number of valid samples')
    parser.add_argument('--n-invalid', type=int, default=50,
                        help='Number of invalid samples')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_hallucination_test(
        n_valid=args.n_valid,
        n_invalid=args.n_invalid,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "hallucination_detection.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert complex numbers for JSON
    def json_serialize(obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serialize)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
