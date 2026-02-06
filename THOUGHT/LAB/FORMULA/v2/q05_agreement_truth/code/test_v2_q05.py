"""
Q05 v2 Test: High Agreement Reveals Truth

Hypothesis: For independent observations, high R implies proximity to
objective ground truth -- not merely consensus. The formula correctly
distinguishes genuine agreement from echo chambers.

Pre-registered criteria:
  CONFIRM:  rho(R, truth_proximity) > 0.5 on STS-B
            AND systematic bias attack does NOT produce misleading high R
            AND R distinguishes echo chambers from genuine agreement
  FALSIFY:  rho < 0.2
            OR bias attack produces high R on wrong answers
            OR R is higher for echo chambers than genuine agreement
  INCONCLUSIVE: 0.2 <= rho <= 0.5; bias results mixed; partial detection

Seed: 42
"""

import sys
import os
import json
import time
import warnings
import numpy as np
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Setup paths ----------
REPO_ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system")
sys.path.insert(0, str(REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "shared"))
RESULTS_DIR = REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "q05_agreement_truth" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from formula import compute_E, compute_grad_S, compute_R_simple, compute_R_full, compute_all

np.random.seed(42)

# =============================================================================
# Utility functions
# =============================================================================

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_sim_matrix(A, B):
    """Pairwise cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def safe_R(embeddings, variant="simple"):
    """Compute R with NaN fallback."""
    if embeddings.shape[0] < 3:
        return float('nan')
    if variant == "simple":
        return compute_R_simple(embeddings)
    else:
        return compute_R_full(embeddings)


# =============================================================================
# TEST 1: Agreement-Truth Correlation on STS-B
# =============================================================================

def test1_agreement_truth_correlation():
    """
    Approach:
    1. Load STS-B test set (1379 pairs with human similarity scores 0-5).
    2. Encode all sentences with all-MiniLM-L6-v2.
    3. Bin pairs by human similarity score (creating "clusters" of pairs
       at each agreement level).
    4. For each bin, collect the embeddings and compute R.
    5. Define "truth proximity" as how close each bin's cosine similarities
       are to the human-assigned scores (lower divergence = higher truth).
    6. Compute Spearman correlation between R and truth proximity.
    """
    print("=" * 70)
    print("TEST 1: Agreement-Truth Correlation on STS-B")
    print("=" * 70)

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # Load data
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    print(f"Loaded {len(ds)} STS-B test pairs")

    # Encode
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences1 = [ex["sentence1"] for ex in ds]
    sentences2 = [ex["sentence2"] for ex in ds]
    human_scores = np.array([ex["score"] for ex in ds])  # 0-5 scale

    print("Encoding sentences...")
    emb1 = model.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
    emb2 = model.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)

    # Compute model cosine similarities for each pair
    model_cosine = np.array([cosine_sim(emb1[i], emb2[i]) for i in range(len(ds))])

    # Normalize human scores to [0, 1] for comparison
    human_norm = human_scores / 5.0

    # Overall baseline: raw correlation between model cosine and human scores
    rho_baseline, p_baseline = stats.spearmanr(model_cosine, human_norm)
    print(f"\nBaseline: Spearman(model_cosine, human_score) = {rho_baseline:.4f}, p={p_baseline:.2e}")

    # Bin pairs by human score
    # Use 10 bins: [0,0.5), [0.5,1.0), ..., [4.5,5.0]
    bin_edges = np.arange(0, 5.5, 0.5)
    bin_indices = np.digitize(human_scores, bin_edges) - 1  # 0-indexed bin

    # For each bin, collect the difference embeddings (emb1 - emb2) and
    # concatenated embeddings to form a "cluster" of observations
    bin_R_simple = []
    bin_R_full = []
    bin_truth_proximity = []
    bin_sizes = []
    bin_centers = []

    print("\nPer-bin analysis:")
    print(f"{'Bin':>10} | {'N':>5} | {'R_simple':>10} | {'R_full':>10} | {'TruthProx':>10} | {'MeanCos':>8} | {'HumanNorm':>10}")
    print("-" * 80)

    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        n = mask.sum()
        if n < 5:
            continue

        # Collect embeddings for this bin: use concatenated (s1, s2) vectors
        # Each pair contributes TWO embeddings; the cluster is all the sentence
        # embeddings from pairs at this similarity level
        idx = np.where(mask)[0]
        cluster_embs = np.vstack([emb1[idx], emb2[idx]])  # shape (2*n, 384)

        # Compute R
        r_simple = safe_R(cluster_embs, "simple")
        r_full = safe_R(cluster_embs, "full")

        # Truth proximity: how well does the model's cosine similarity
        # match human judgments in this bin?
        # Use 1 - MAE(model_cosine, human_norm) for this bin
        mae = np.mean(np.abs(model_cosine[idx] - human_norm[idx]))
        truth_prox = 1.0 - mae

        bin_center = (bin_edges[b] + bin_edges[b + 1]) / 2.0

        bin_R_simple.append(r_simple)
        bin_R_full.append(r_full)
        bin_truth_proximity.append(truth_prox)
        bin_sizes.append(n)
        bin_centers.append(bin_center)

        mean_cos = np.mean(model_cosine[idx])
        mean_human = np.mean(human_norm[idx])
        print(f"[{bin_edges[b]:.1f},{bin_edges[b+1]:.1f}) | {n:5d} | {r_simple:10.4f} | {r_full:10.4f} | {truth_prox:10.4f} | {mean_cos:8.4f} | {mean_human:10.4f}")

    bin_R_simple = np.array(bin_R_simple)
    bin_R_full = np.array(bin_R_full)
    bin_truth_proximity = np.array(bin_truth_proximity)

    # Remove NaN entries
    valid = ~(np.isnan(bin_R_simple) | np.isnan(bin_truth_proximity))
    R_s = bin_R_simple[valid]
    R_f = bin_R_full[valid]
    T = bin_truth_proximity[valid]

    rho_simple, p_simple = stats.spearmanr(R_s, T)
    rho_full, p_full = stats.spearmanr(R_f, T)

    print(f"\nSpearman(R_simple, truth_proximity) = {rho_simple:.4f}, p = {p_simple:.4e}")
    print(f"Spearman(R_full, truth_proximity)   = {rho_full:.4f}, p = {p_full:.4e}")

    # Also: per-pair R computation approach
    # Treat each pair as a 2-element "cluster" -- too small for R
    # Instead: for each bin, R of the difference vectors
    print("\n--- Alternative: R on difference embeddings ---")
    alt_R_simple = []
    alt_truth = []
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        n = mask.sum()
        if n < 10:
            continue
        idx = np.where(mask)[0]
        diff_embs = emb1[idx] - emb2[idx]  # difference vectors
        r_s = safe_R(diff_embs, "simple")
        mae = np.mean(np.abs(model_cosine[idx] - human_norm[idx]))
        truth_prox = 1.0 - mae
        if not np.isnan(r_s):
            alt_R_simple.append(r_s)
            alt_truth.append(truth_prox)

    alt_R_simple = np.array(alt_R_simple)
    alt_truth = np.array(alt_truth)
    rho_alt, p_alt = stats.spearmanr(alt_R_simple, alt_truth)
    print(f"Spearman(R_simple_diff, truth_proximity) = {rho_alt:.4f}, p = {p_alt:.4e}")

    results = {
        "test": "Agreement-Truth Correlation on STS-B",
        "n_pairs": len(ds),
        "n_bins": len(bin_centers),
        "baseline_spearman_cosine_vs_human": float(rho_baseline),
        "baseline_p": float(p_baseline),
        "spearman_R_simple_vs_truth": float(rho_simple),
        "p_R_simple": float(p_simple),
        "spearman_R_full_vs_truth": float(rho_full),
        "p_R_full": float(p_full),
        "spearman_R_alt_diff_vs_truth": float(rho_alt),
        "p_R_alt": float(p_alt),
        "bin_centers": [float(x) for x in bin_centers],
        "bin_sizes": [int(x) for x in bin_sizes],
        "bin_R_simple": [float(x) for x in bin_R_simple],
        "bin_R_full": [float(x) for x in bin_R_full],
        "bin_truth_proximity": [float(x) for x in bin_truth_proximity],
    }

    return results, emb1, emb2, human_scores, model_cosine, model


# =============================================================================
# TEST 2: Multiple Embedding Models as Independent Observers
# =============================================================================

def test2_multi_model_agreement(emb1_minilm, emb2_minilm, human_scores):
    """
    Use two embedding models as independent observers.
    Compute agreement between their similarity estimates, compare to ground truth.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Embedding Models as Independent Observers")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    # Use first 500 pairs
    n_pairs = min(500, len(ds))
    sentences1 = [ds[i]["sentence1"] for i in range(n_pairs)]
    sentences2 = [ds[i]["sentence2"] for i in range(n_pairs)]
    human_sub = human_scores[:n_pairs]
    human_norm = human_sub / 5.0

    # Model 1: already encoded (MiniLM)
    cos_minilm = np.array([cosine_sim(emb1_minilm[i], emb2_minilm[i]) for i in range(n_pairs)])

    # Model 2: mpnet
    print("Encoding with all-mpnet-base-v2...")
    model2 = SentenceTransformer("all-mpnet-base-v2")
    emb1_mpnet = model2.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
    emb2_mpnet = model2.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)
    cos_mpnet = np.array([cosine_sim(emb1_mpnet[i], emb2_mpnet[i]) for i in range(n_pairs)])

    # Model agreement: correlation between models' similarity scores
    rho_models, p_models = stats.spearmanr(cos_minilm, cos_mpnet)
    print(f"\nInter-model agreement: Spearman(MiniLM, mpnet) = {rho_models:.4f}, p = {p_models:.2e}")

    # Each model vs human ground truth
    rho_minilm_human, p_minilm = stats.spearmanr(cos_minilm, human_norm)
    rho_mpnet_human, p_mpnet = stats.spearmanr(cos_mpnet, human_norm)
    print(f"MiniLM vs Human:  Spearman = {rho_minilm_human:.4f}, p = {p_minilm:.2e}")
    print(f"mpnet vs Human:   Spearman = {rho_mpnet_human:.4f}, p = {p_mpnet:.2e}")

    # Compute R across models' predictions for groups of pairs
    # Bin by human score, then for each bin compute R from the two models'
    # similarity predictions stacked as a 2-column matrix
    bin_edges = np.arange(0, 5.5, 0.5)
    bin_indices = np.digitize(human_sub, bin_edges) - 1

    bin_R_multi = []
    bin_mean_agreement = []
    bin_mean_truth_error = []
    bin_centers = []

    print("\nPer-bin multi-model R:")
    print(f"{'Bin':>10} | {'N':>5} | {'R_multi':>10} | {'ModelAgree':>12} | {'TruthErr':>10}")
    print("-" * 65)

    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        n = mask.sum()
        if n < 5:
            continue

        idx = np.where(mask)[0]

        # Stack the two models' predictions as a matrix
        # Each model gives one similarity per pair -> (n_pairs, 2) matrix
        # But R expects (n, d) embeddings. Instead, we treat the two models'
        # full embeddings for sentences in this bin as the observation matrix.
        #
        # More meaningful: stack the difference vectors from both models
        diff_minilm = emb1_minilm[idx] - emb2_minilm[idx]  # (n, 384)
        diff_mpnet = emb1_mpnet[idx] - emb2_mpnet[idx]  # (n, 768)

        # To compute R, we need observations in the same space.
        # Use the cosine similarities from both models as a 2D embedding.
        obs_matrix = np.column_stack([cos_minilm[idx], cos_mpnet[idx]])  # (n, 2)
        # This is only 2D, so R from it is limited.
        # Better: compute R on each model's embeddings separately, then average.

        r_minilm = safe_R(np.vstack([emb1_minilm[idx], emb2_minilm[idx]]), "simple")
        r_mpnet = safe_R(np.vstack([emb1_mpnet[idx], emb2_mpnet[idx]]), "simple")

        # Combined R: average of the two independent R estimates
        if not np.isnan(r_minilm) and not np.isnan(r_mpnet):
            r_multi = (r_minilm + r_mpnet) / 2.0
        else:
            r_multi = float('nan')

        # Agreement: correlation between models for this bin
        if n >= 3:
            bin_agree = float(np.corrcoef(cos_minilm[idx], cos_mpnet[idx])[0, 1])
        else:
            bin_agree = float('nan')

        # Truth error: mean absolute difference from human scores
        mean_pred = (cos_minilm[idx] + cos_mpnet[idx]) / 2.0
        truth_err = float(np.mean(np.abs(mean_pred - human_norm[idx])))

        bin_center = (bin_edges[b] + bin_edges[b + 1]) / 2.0
        bin_R_multi.append(r_multi)
        bin_mean_agreement.append(bin_agree)
        bin_mean_truth_error.append(truth_err)
        bin_centers.append(bin_center)

        print(f"[{bin_edges[b]:.1f},{bin_edges[b+1]:.1f}) | {n:5d} | {r_multi:10.4f} | {bin_agree:12.4f} | {truth_err:10.4f}")

    # Correlation between multi-model R and truth error
    R_arr = np.array(bin_R_multi)
    E_arr = np.array(bin_mean_truth_error)
    valid = ~(np.isnan(R_arr) | np.isnan(E_arr))

    # Lower truth error = better; higher R should correlate with lower error
    # So we expect NEGATIVE correlation between R and truth error
    rho_R_err, p_R_err = stats.spearmanr(R_arr[valid], E_arr[valid])
    print(f"\nSpearman(R_multi, truth_error) = {rho_R_err:.4f}, p = {p_R_err:.4e}")
    print(f"  (Negative = R tracks truth; Positive = R misleading)")

    # Also: does agreement between models predict truth?
    A_arr = np.array(bin_mean_agreement)
    valid2 = ~(np.isnan(A_arr) | np.isnan(E_arr))
    if valid2.sum() >= 3:
        rho_agree_err, p_agree_err = stats.spearmanr(A_arr[valid2], E_arr[valid2])
        print(f"Spearman(model_agreement, truth_error) = {rho_agree_err:.4f}, p = {p_agree_err:.4e}")
    else:
        rho_agree_err, p_agree_err = float('nan'), float('nan')

    results = {
        "test": "Multi-Model Agreement",
        "n_pairs": n_pairs,
        "inter_model_spearman": float(rho_models),
        "inter_model_p": float(p_models),
        "minilm_vs_human_spearman": float(rho_minilm_human),
        "mpnet_vs_human_spearman": float(rho_mpnet_human),
        "spearman_R_multi_vs_truth_error": float(rho_R_err),
        "p_R_multi_vs_truth_error": float(p_R_err),
        "spearman_agreement_vs_truth_error": float(rho_agree_err) if not np.isnan(rho_agree_err) else None,
        "bin_centers": [float(x) for x in bin_centers],
        "bin_R_multi": [float(x) for x in bin_R_multi],
        "bin_mean_truth_error": [float(x) for x in bin_mean_truth_error],
    }

    return results


# =============================================================================
# TEST 3: Systematic Bias Attack
# =============================================================================

def test3_systematic_bias_attack(model):
    """
    Create independent-but-biased observations by prepending a systematic
    phrase to all sentences. If R is still high on biased data, R fails to
    detect systematic bias.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Systematic Bias Attack")
    print("=" * 70)

    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    # Take 200 pairs
    n_pairs = 200
    np.random.seed(42)
    indices = np.random.choice(len(ds), n_pairs, replace=False)
    indices.sort()

    sentences1 = [ds[int(i)]["sentence1"] for i in indices]
    sentences2 = [ds[int(i)]["sentence2"] for i in indices]
    human_scores_sub = np.array([ds[int(i)]["score"] for i in indices])
    human_norm = human_scores_sub / 5.0

    # Encode unbiased
    print("Encoding unbiased sentences...")
    emb1_clean = model.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
    emb2_clean = model.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)

    cos_clean = np.array([cosine_sim(emb1_clean[i], emb2_clean[i]) for i in range(n_pairs)])

    # Apply systematic bias: prepend the same phrase
    bias_phrases = [
        "In conclusion, ",
        "According to recent studies, ",
        "The committee determined that ",
    ]

    results_per_bias = []

    for bias_phrase in bias_phrases:
        biased_s1 = [bias_phrase + s for s in sentences1]
        biased_s2 = [bias_phrase + s for s in sentences2]

        print(f"\nBias phrase: '{bias_phrase}'")
        emb1_biased = model.encode(biased_s1, show_progress_bar=False, convert_to_numpy=True)
        emb2_biased = model.encode(biased_s2, show_progress_bar=False, convert_to_numpy=True)

        cos_biased = np.array([cosine_sim(emb1_biased[i], emb2_biased[i]) for i in range(n_pairs)])

        # Compare cosine similarities
        cos_diff = cos_biased - cos_clean
        print(f"  Mean cosine shift: {np.mean(cos_diff):+.4f} (std: {np.std(cos_diff):.4f})")
        print(f"  Cosine clean:  mean={np.mean(cos_clean):.4f}, std={np.std(cos_clean):.4f}")
        print(f"  Cosine biased: mean={np.mean(cos_biased):.4f}, std={np.std(cos_biased):.4f}")

        # Correlation with human truth: does bias hurt truth-tracking?
        rho_clean, _ = stats.spearmanr(cos_clean, human_norm)
        rho_biased, _ = stats.spearmanr(cos_biased, human_norm)
        print(f"  Spearman(clean_cos, human):  {rho_clean:.4f}")
        print(f"  Spearman(biased_cos, human): {rho_biased:.4f}")
        print(f"  Truth-tracking degradation: {rho_clean - rho_biased:+.4f}")

        # Compute R on bins for both clean and biased
        bin_edges = np.arange(0, 5.5, 1.0)  # coarser bins for 200 pairs
        bin_indices = np.digitize(human_scores_sub, bin_edges) - 1

        R_clean_bins = []
        R_biased_bins = []
        bin_labels = []

        for b in range(len(bin_edges) - 1):
            mask = bin_indices == b
            n = mask.sum()
            if n < 5:
                continue

            idx = np.where(mask)[0]

            cluster_clean = np.vstack([emb1_clean[idx], emb2_clean[idx]])
            cluster_biased = np.vstack([emb1_biased[idx], emb2_biased[idx]])

            r_c = safe_R(cluster_clean, "simple")
            r_b = safe_R(cluster_biased, "simple")

            R_clean_bins.append(r_c)
            R_biased_bins.append(r_b)
            bin_labels.append(f"[{bin_edges[b]:.0f},{bin_edges[b+1]:.0f})")

            print(f"  Bin {bin_labels[-1]}: R_clean={r_c:.4f}, R_biased={r_b:.4f}, delta={r_b-r_c:+.4f}")

        # Also compute global R
        all_clean = np.vstack([emb1_clean, emb2_clean])
        all_biased = np.vstack([emb1_biased, emb2_biased])
        R_global_clean = safe_R(all_clean, "simple")
        R_global_biased = safe_R(all_biased, "simple")
        print(f"\n  Global R_clean:  {R_global_clean:.4f}")
        print(f"  Global R_biased: {R_global_biased:.4f}")
        print(f"  Delta: {R_global_biased - R_global_clean:+.4f}")

        # KEY CHECK: Does bias inflate R without improving truth-tracking?
        # If R_biased > R_clean but rho_biased < rho_clean, that is a FAILURE
        bias_inflates_R = R_global_biased > R_global_clean
        bias_hurts_truth = rho_biased < rho_clean
        attack_succeeds = bias_inflates_R and bias_hurts_truth

        print(f"\n  Bias inflates R: {bias_inflates_R}")
        print(f"  Bias hurts truth: {bias_hurts_truth}")
        print(f"  ATTACK SUCCEEDS (high R, low truth): {attack_succeeds}")

        results_per_bias.append({
            "bias_phrase": bias_phrase,
            "mean_cosine_shift": float(np.mean(cos_diff)),
            "rho_clean_vs_human": float(rho_clean),
            "rho_biased_vs_human": float(rho_biased),
            "truth_degradation": float(rho_clean - rho_biased),
            "R_global_clean": float(R_global_clean),
            "R_global_biased": float(R_global_biased),
            "R_delta": float(R_global_biased - R_global_clean),
            "bias_inflates_R": bool(bias_inflates_R),
            "bias_hurts_truth": bool(bias_hurts_truth),
            "attack_succeeds": bool(attack_succeeds),
            "bin_labels": bin_labels,
            "R_clean_bins": [float(x) for x in R_clean_bins],
            "R_biased_bins": [float(x) for x in R_biased_bins],
        })

    # Summary
    n_attacks_succeed = sum(1 for r in results_per_bias if r["attack_succeeds"])
    print(f"\n--- Bias Attack Summary ---")
    print(f"Attacks attempted: {len(results_per_bias)}")
    print(f"Attacks succeeded (R inflated + truth degraded): {n_attacks_succeed}")

    results = {
        "test": "Systematic Bias Attack",
        "n_pairs": n_pairs,
        "bias_results": results_per_bias,
        "n_attacks": len(results_per_bias),
        "n_attacks_succeeded": n_attacks_succeed,
    }

    return results


# =============================================================================
# TEST 4: Echo Chamber vs Genuine Agreement
# =============================================================================

def test4_echo_chamber_detection(model):
    """
    Compare R for:
    1. Echo chambers: duplicate embeddings with tiny noise (correlated sources)
    2. Genuine agreement: different sentences that humans rated similarly

    If R is HIGHER for echo chambers, this is a failure mode.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Echo Chamber vs Genuine Agreement Detection")
    print("=" * 70)

    from datasets import load_dataset

    ds = load_dataset("mteb/stsbenchmark-sts", split="test")

    # Encode all sentences
    all_sentences = list(set(
        [ds[i]["sentence1"] for i in range(len(ds))] +
        [ds[i]["sentence2"] for i in range(len(ds))]
    ))
    print(f"Encoding {len(all_sentences)} unique sentences...")
    all_embs = model.encode(all_sentences, show_progress_bar=False, convert_to_numpy=True)
    sent_to_emb = {s: all_embs[i] for i, s in enumerate(all_sentences)}

    np.random.seed(42)

    # --- Genuine Agreement Clusters ---
    # Group sentences by human similarity: take pairs with score >= 4.0
    # and cluster their individual sentence embeddings
    high_sim_pairs = [(i, ds[i]) for i in range(len(ds)) if ds[i]["score"] >= 4.0]
    print(f"\nHigh-similarity pairs (score >= 4.0): {len(high_sim_pairs)}")

    # Collect unique sentences from high-similarity pairs
    genuine_sentences = list(set(
        [p[1]["sentence1"] for p in high_sim_pairs] +
        [p[1]["sentence2"] for p in high_sim_pairs]
    ))
    genuine_embs = np.array([sent_to_emb[s] for s in genuine_sentences])
    print(f"Unique sentences in high-sim cluster: {len(genuine_sentences)}")

    # Compute R for genuine agreement clusters of various sizes
    n_trials = 50
    cluster_sizes = [10, 20, 50]
    genuine_R_results = {}

    for cs in cluster_sizes:
        if len(genuine_sentences) < cs:
            continue
        R_values = []
        for trial in range(n_trials):
            idx = np.random.choice(len(genuine_sentences), cs, replace=False)
            cluster = genuine_embs[idx]
            r = safe_R(cluster, "simple")
            if not np.isnan(r):
                R_values.append(r)
        genuine_R_results[cs] = R_values
        print(f"Genuine agreement (size={cs}): mean R = {np.mean(R_values):.4f} +/- {np.std(R_values):.4f} (n={len(R_values)})")

    # --- Echo Chamber Clusters ---
    # Take a SINGLE sentence and duplicate its embedding with tiny noise
    # This simulates correlated sources echoing the same content
    echo_R_results = {}

    for cs in cluster_sizes:
        R_values = []
        for trial in range(n_trials):
            # Pick a random base sentence
            base_idx = np.random.randint(len(all_embs))
            base_emb = all_embs[base_idx]

            # Create echo chamber: add very small Gaussian noise
            noise_scale = 0.01  # tiny noise
            echo_cluster = base_emb + np.random.randn(cs, base_emb.shape[0]) * noise_scale
            r = safe_R(echo_cluster, "simple")
            if not np.isnan(r):
                R_values.append(r)
        echo_R_results[cs] = R_values
        print(f"Echo chamber (size={cs}, noise=0.01): mean R = {np.mean(R_values):.4f} +/- {np.std(R_values):.4f} (n={len(R_values)})")

    # --- Moderate echo: noise = 0.05 ---
    echo_moderate_R = {}
    for cs in cluster_sizes:
        R_values = []
        for trial in range(n_trials):
            base_idx = np.random.randint(len(all_embs))
            base_emb = all_embs[base_idx]
            echo_cluster = base_emb + np.random.randn(cs, base_emb.shape[0]) * 0.05
            r = safe_R(echo_cluster, "simple")
            if not np.isnan(r):
                R_values.append(r)
        echo_moderate_R[cs] = R_values
        print(f"Echo chamber (size={cs}, noise=0.05): mean R = {np.mean(R_values):.4f} +/- {np.std(R_values):.4f} (n={len(R_values)})")

    # --- Random clusters (baseline) ---
    random_R_results = {}
    for cs in cluster_sizes:
        R_values = []
        for trial in range(n_trials):
            idx = np.random.choice(len(all_embs), cs, replace=False)
            cluster = all_embs[idx]
            r = safe_R(cluster, "simple")
            if not np.isnan(r):
                R_values.append(r)
        random_R_results[cs] = R_values
        print(f"Random cluster (size={cs}): mean R = {np.mean(R_values):.4f} +/- {np.std(R_values):.4f} (n={len(R_values)})")

    # Statistical tests
    print("\n--- Statistical Comparison ---")
    detection_results = {}

    for cs in cluster_sizes:
        if cs not in genuine_R_results or cs not in echo_R_results:
            continue

        genuine = np.array(genuine_R_results[cs])
        echo = np.array(echo_R_results[cs])
        echo_mod = np.array(echo_moderate_R.get(cs, []))
        random = np.array(random_R_results[cs])

        # Mann-Whitney U test: genuine vs echo
        stat_ge, p_ge = stats.mannwhitneyu(genuine, echo, alternative='two-sided')
        # Direction: which has higher R?
        genuine_higher = np.mean(genuine) > np.mean(echo)

        print(f"\nCluster size = {cs}:")
        print(f"  Genuine mean R: {np.mean(genuine):.4f}")
        print(f"  Echo mean R:    {np.mean(echo):.4f}")
        if len(echo_mod) > 0:
            print(f"  Echo(mod) mean R: {np.mean(echo_mod):.4f}")
        print(f"  Random mean R:  {np.mean(random):.4f}")
        print(f"  Mann-Whitney (genuine vs echo): U={stat_ge:.1f}, p={p_ge:.4e}")
        print(f"  Genuine R > Echo R: {genuine_higher}")

        if genuine_higher:
            print(f"  --> PASS: Genuine agreement has higher R than echo chamber")
        else:
            print(f"  --> FAIL: Echo chamber has higher R than genuine agreement")

        detection_results[cs] = {
            "genuine_mean_R": float(np.mean(genuine)),
            "genuine_std_R": float(np.std(genuine)),
            "echo_mean_R": float(np.mean(echo)),
            "echo_std_R": float(np.std(echo)),
            "echo_moderate_mean_R": float(np.mean(echo_mod)) if len(echo_mod) > 0 else None,
            "random_mean_R": float(np.mean(random)),
            "random_std_R": float(np.std(random)),
            "mannwhitney_U": float(stat_ge),
            "mannwhitney_p": float(p_ge),
            "genuine_higher_than_echo": bool(genuine_higher),
        }

    # Overall detection assessment
    all_pass = all(d["genuine_higher_than_echo"] for d in detection_results.values())
    any_pass = any(d["genuine_higher_than_echo"] for d in detection_results.values())

    print(f"\n--- Echo Chamber Detection Summary ---")
    print(f"All sizes show genuine > echo: {all_pass}")
    print(f"Any size shows genuine > echo: {any_pass}")

    results = {
        "test": "Echo Chamber vs Genuine Agreement",
        "n_unique_sentences": len(all_sentences),
        "n_high_sim_pairs": len(high_sim_pairs),
        "n_trials": n_trials,
        "cluster_sizes": cluster_sizes,
        "detection_results": detection_results,
        "all_sizes_pass": all_pass,
        "any_size_pass": any_pass,
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Q05 v2 TEST: HIGH AGREEMENT REVEALS TRUTH")
    print("=" * 70)
    print(f"Seed: 42")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = {}

    # Test 1
    t1_start = time.time()
    r1, emb1, emb2, human_scores, model_cosine, model = test1_agreement_truth_correlation()
    r1["duration_s"] = time.time() - t1_start
    all_results["test1"] = r1

    # Test 2
    t2_start = time.time()
    r2 = test2_multi_model_agreement(emb1, emb2, human_scores)
    r2["duration_s"] = time.time() - t2_start
    all_results["test2"] = r2

    # Test 3
    t3_start = time.time()
    r3 = test3_systematic_bias_attack(model)
    r3["duration_s"] = time.time() - t3_start
    all_results["test3"] = r3

    # Test 4
    t4_start = time.time()
    r4 = test4_echo_chamber_detection(model)
    r4["duration_s"] = time.time() - t4_start
    all_results["test4"] = r4

    # Save results
    results_file = RESULTS_DIR / "test_v2_q05_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # =================================================================
    # VERDICT DETERMINATION
    # =================================================================
    print("\n" + "=" * 70)
    print("VERDICT DETERMINATION")
    print("=" * 70)

    # Criterion 1: R correlates with truth proximity (rho > 0.5)
    rho_test1 = r1["spearman_R_simple_vs_truth"]
    criterion1 = rho_test1 > 0.5
    criterion1_fail = rho_test1 < 0.2
    print(f"\nCriterion 1: R correlates with truth proximity")
    print(f"  rho(R_simple, truth_proximity) = {rho_test1:.4f}")
    print(f"  Confirm (>0.5): {criterion1}")
    print(f"  Falsify (<0.2): {criterion1_fail}")

    # Criterion 2: Systematic bias does NOT inflate R misleadingly
    n_succeed = r3["n_attacks_succeeded"]
    n_attacks = r3["n_attacks"]
    criterion2 = n_succeed == 0  # no attacks succeed
    criterion2_fail = n_succeed == n_attacks  # all attacks succeed
    print(f"\nCriterion 2: Bias attack does not inflate R")
    print(f"  Attacks succeeded: {n_succeed}/{n_attacks}")
    print(f"  Confirm (0 succeed): {criterion2}")
    print(f"  Falsify (all succeed): {criterion2_fail}")

    # Criterion 3: R distinguishes echo from genuine
    criterion3 = r4["all_sizes_pass"]
    criterion3_fail = not r4["any_size_pass"]
    print(f"\nCriterion 3: R distinguishes echo from genuine agreement")
    print(f"  All sizes genuine > echo: {criterion3}")
    print(f"  Falsify (no sizes pass): {criterion3_fail}")

    # Overall verdict
    if criterion1 and criterion2 and criterion3:
        verdict = "CONFIRMED"
    elif criterion1_fail or criterion2_fail or criterion3_fail:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")

    all_results["verdict"] = {
        "result": verdict,
        "criterion1_rho": float(rho_test1),
        "criterion1_confirm": bool(criterion1),
        "criterion1_falsify": bool(criterion1_fail),
        "criterion2_attacks_succeeded": n_succeed,
        "criterion2_confirm": bool(criterion2),
        "criterion2_falsify": bool(criterion2_fail),
        "criterion3_all_pass": bool(criterion3),
        "criterion3_confirm": bool(criterion3),
        "criterion3_falsify": bool(criterion3_fail),
    }

    # Re-save with verdict
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    results = main()
