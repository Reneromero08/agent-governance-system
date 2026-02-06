"""
Q01 v2 Test: grad_S is the correct normalization for E
========================================================

Pre-registered hypothesis:
  R = E/grad_S outperforms alternative normalizations on real data (STS-B).

Pre-registered criteria:
  CONFIRM:  E/grad_S outperforms >=4/5 alternatives (p<0.01) AND bridge r>0.8
  FALSIFY:  E/grad_S loses to >=3 alternatives OR bridge r<0.3
  INCONCLUSIVE: otherwise

Dataset: STS-B (Semantic Textual Similarity Benchmark) from HuggingFace
Model: sentence-transformers/all-MiniLM-L6-v2
Seed: 42
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# Add shared formula module
_this_dir = os.path.dirname(os.path.abspath(__file__))
_shared_dir = os.path.normpath(os.path.join(_this_dir, '..', '..', 'shared'))
sys.path.insert(0, _shared_dir)
from formula import compute_E, compute_grad_S

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# TEST 1: Normalization Comparison on STS-B
# ============================================================

def compute_alternative_normalizations(embeddings):
    """Compute all normalization variants for a cluster of embeddings."""
    n = embeddings.shape[0]
    if n < 4:
        return None

    # --- Compute pairwise cosine similarities ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms

    sim_matrix = normed @ normed.T
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    E = float(np.mean(pairwise_sims))
    grad_S = float(np.std(pairwise_sims))

    # --- Alternative normalizers ---
    # MAD: median absolute deviation
    median_sim = np.median(pairwise_sims)
    mad = float(np.median(np.abs(pairwise_sims - median_sim)))

    # IQR: interquartile range
    q75 = np.percentile(pairwise_sims, 75)
    q25 = np.percentile(pairwise_sims, 25)
    iqr = float(q75 - q25)

    # SNR: mean embedding norm / std embedding norm
    emb_norms = np.linalg.norm(embeddings, axis=1)
    snr = float(np.mean(emb_norms) / max(np.std(emb_norms), 1e-10))

    # Guard against division by zero
    eps = 1e-10

    results = {
        'E': E,
        'grad_S': grad_S,
        'R_grad_S': E / max(grad_S, eps),           # The formula: E/grad_S
        'R_raw_E': E,                                 # Raw E (no normalization)
        'R_grad_S_sq': E / max(grad_S ** 2, eps),    # E/grad_S^2 (precision-weighted)
        'R_MAD': E / max(mad, eps),                   # E/MAD (robust)
        'R_IQR': E / max(iqr, eps),                   # E/IQR (robust)
        'SNR': snr,                                    # Signal-to-noise ratio
    }
    return results


def build_clusters_from_stsb(dataset_split, embeddings_dict, min_cluster_size=10, max_cluster_size=30):
    """
    Build clusters from STS-B by grouping sentence pairs by similarity bin.

    Strategy: For each similarity bin, collect all sentences that appear in
    pairs within that bin. Then sample sub-clusters from those sentences.
    """
    clusters = []

    # Bin edges: 0-1, 1-2, 2-3, 3-4, 4-5
    bin_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    for low, high in bin_edges:
        # Find all pairs in this similarity range
        bin_sentences = []
        bin_scores = []
        for item in dataset_split:
            score = item['label']
            if low <= score < high or (high == 5 and score == 5):
                bin_sentences.append(item['sentence1'])
                bin_sentences.append(item['sentence2'])
                bin_scores.append(score)

        if len(bin_scores) == 0:
            continue

        mean_human_score = np.mean(bin_scores)

        # Get unique sentences in this bin
        unique_sents = list(set(bin_sentences))

        # Only keep sentences we have embeddings for
        valid_sents = [s for s in unique_sents if s in embeddings_dict]

        if len(valid_sents) < min_cluster_size:
            log(f"  Bin [{low}-{high}]: only {len(valid_sents)} valid sentences, skipping")
            continue

        # Create multiple sub-clusters by random sampling
        n_clusters_per_bin = min(50, len(valid_sents) // min_cluster_size)
        rng = np.random.RandomState(42 + int(low * 10))

        for ci in range(n_clusters_per_bin):
            size = rng.randint(min_cluster_size, min(max_cluster_size, len(valid_sents)) + 1)
            chosen = rng.choice(len(valid_sents), size=size, replace=False)
            chosen_sents = [valid_sents[i] for i in chosen]
            embs = np.array([embeddings_dict[s] for s in chosen_sents])

            clusters.append({
                'bin_low': low,
                'bin_high': high,
                'mean_human_score': mean_human_score,
                'n_sentences': len(chosen_sents),
                'embeddings': embs,
            })

    return clusters


def run_test1(dataset_split, embeddings_dict):
    """Test 1: Normalization comparison on STS-B."""
    log("=" * 60)
    log("TEST 1: Normalization Comparison on STS-B")
    log("=" * 60)

    # Build clusters
    log("Building clusters from STS-B pairs grouped by similarity bins...")
    clusters = build_clusters_from_stsb(dataset_split, embeddings_dict)
    log(f"Built {len(clusters)} clusters")

    if len(clusters) < 20:
        log("ERROR: Too few clusters for meaningful analysis")
        return None

    # Compute normalizations for each cluster
    human_scores = []
    norm_results = {
        'R_grad_S': [],
        'R_raw_E': [],
        'R_grad_S_sq': [],
        'R_MAD': [],
        'R_IQR': [],
        'SNR': [],
    }

    skipped = 0
    for cluster in clusters:
        r = compute_alternative_normalizations(cluster['embeddings'])
        if r is None:
            skipped += 1
            continue

        # Check for NaN/Inf in any metric
        has_bad = False
        for key in norm_results:
            val = r[key]
            if not np.isfinite(val):
                has_bad = True
                break

        if has_bad:
            skipped += 1
            continue

        human_scores.append(cluster['mean_human_score'])
        for key in norm_results:
            norm_results[key].append(r[key])

    log(f"Valid clusters: {len(human_scores)}, skipped: {skipped}")

    human_scores = np.array(human_scores)

    # Compute Spearman correlations
    log("\n--- Spearman Correlations with Human Similarity Scores ---")
    log(f"{'Normalization':<20} {'Spearman r':>12} {'p-value':>15} {'Direction':>10}")
    log("-" * 60)

    correlation_results = {}
    for key in norm_results:
        values = np.array(norm_results[key])
        rho, p = stats.spearmanr(values, human_scores)
        direction = "positive" if rho > 0 else "negative"
        log(f"{key:<20} {rho:>12.6f} {p:>15.2e} {direction:>10}")
        correlation_results[key] = {
            'spearman_rho': float(rho),
            'p_value': float(p),
            'n_clusters': len(human_scores),
        }

    # Determine ranking
    ranked = sorted(correlation_results.items(), key=lambda x: abs(x[1]['spearman_rho']), reverse=True)
    log("\n--- Ranking by |Spearman rho| ---")
    for rank, (name, res) in enumerate(ranked, 1):
        log(f"  {rank}. {name}: |rho| = {abs(res['spearman_rho']):.6f}")

    # Count how many alternatives E/grad_S beats
    grad_s_rho = abs(correlation_results['R_grad_S']['spearman_rho'])
    alternatives = ['R_raw_E', 'R_grad_S_sq', 'R_MAD', 'R_IQR', 'SNR']
    wins = 0
    losses = 0
    for alt in alternatives:
        alt_rho = abs(correlation_results[alt]['spearman_rho'])
        if grad_s_rho > alt_rho:
            wins += 1
        else:
            losses += 1

    log(f"\nE/grad_S wins against {wins}/5 alternatives, loses to {losses}/5")

    # Statistical significance: bootstrap comparison
    log("\n--- Bootstrap Comparison (10000 resamples) ---")
    n_boot = 10000
    rng = np.random.RandomState(42)

    grad_s_values = np.array(norm_results['R_grad_S'])
    pairwise_pvalues = {}

    for alt in alternatives:
        alt_values = np.array(norm_results[alt])

        # Bootstrap: difference in |Spearman rho|
        boot_diffs = []
        for _ in range(n_boot):
            idx = rng.choice(len(human_scores), size=len(human_scores), replace=True)
            hs_boot = human_scores[idx]
            gs_boot = grad_s_values[idx]
            alt_boot = alt_values[idx]

            rho_gs, _ = stats.spearmanr(gs_boot, hs_boot)
            rho_alt, _ = stats.spearmanr(alt_boot, hs_boot)
            boot_diffs.append(abs(rho_gs) - abs(rho_alt))

        boot_diffs = np.array(boot_diffs)
        # Two-sided p-value: proportion of bootstrap samples where difference is <= 0
        p_boot = float(np.mean(boot_diffs <= 0))
        mean_diff = float(np.mean(boot_diffs))
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        pairwise_pvalues[alt] = {
            'bootstrap_p': p_boot,
            'mean_diff': mean_diff,
            'ci_95': [ci_low, ci_high],
        }
        log(f"  R_grad_S vs {alt}: mean diff = {mean_diff:+.6f}, "
            f"95% CI = [{ci_low:.6f}, {ci_high:.6f}], p = {p_boot:.4f}")

    # Significant wins (p < 0.01)
    sig_wins = sum(1 for alt in alternatives if pairwise_pvalues[alt]['bootstrap_p'] < 0.01)
    log(f"\nSignificant wins (p < 0.01): {sig_wins}/5")

    return {
        'n_clusters': int(len(human_scores)),
        'correlations': correlation_results,
        'ranking': [(name, float(abs(res['spearman_rho']))) for name, res in ranked],
        'wins_vs_alternatives': wins,
        'losses_vs_alternatives': losses,
        'pairwise_bootstrap': pairwise_pvalues,
        'significant_wins_p01': sig_wins,
    }


# ============================================================
# TEST 2: Bridge Test (E_gaussian vs E_cosine)
# ============================================================

def compute_E_gaussian(embeddings):
    """
    Compute E_gaussian = mean(exp(-z^2/2))
    where z = (x_i - mean) / std for each dimension,
    averaged over all observations and dimensions.

    This is the Gaussian kernel version from v1 theory.
    """
    n, d = embeddings.shape
    if n < 2:
        return float('nan')

    mean_emb = embeddings.mean(axis=0)
    std_emb = embeddings.std(axis=0)

    # Avoid division by zero in low-variance dimensions
    std_emb = np.where(std_emb < 1e-10, 1e-10, std_emb)

    # Normalized deviations
    z = (embeddings - mean_emb) / std_emb  # shape (n, d)

    # Gaussian kernel per observation per dimension
    gaussian_vals = np.exp(-0.5 * z ** 2)

    # Mean across all observations and dimensions
    E_gauss = float(np.mean(gaussian_vals))
    return E_gauss


def run_test2(clusters):
    """Test 2: Bridge test between E_gaussian and E_cosine."""
    log("\n" + "=" * 60)
    log("TEST 2: Bridge Test (E_gaussian vs E_cosine)")
    log("=" * 60)

    e_cosine_list = []
    e_gaussian_list = []

    for cluster in clusters:
        embs = cluster['embeddings']
        if embs.shape[0] < 4:
            continue

        ec = compute_E(embs)
        eg = compute_E_gaussian(embs)

        if np.isfinite(ec) and np.isfinite(eg):
            e_cosine_list.append(ec)
            e_gaussian_list.append(eg)

    e_cosine = np.array(e_cosine_list)
    e_gaussian = np.array(e_gaussian_list)

    log(f"Computed E_cosine and E_gaussian for {len(e_cosine)} clusters")

    # Spearman correlation
    rho, p = stats.spearmanr(e_cosine, e_gaussian)
    log(f"Spearman correlation: rho = {rho:.6f}, p = {p:.2e}")

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(e_cosine, e_gaussian)
    log(f"Pearson correlation:  r = {r_pearson:.6f}, p = {p_pearson:.2e}")

    # Summary stats
    log(f"\nE_cosine  range: [{e_cosine.min():.4f}, {e_cosine.max():.4f}], mean = {e_cosine.mean():.4f}")
    log(f"E_gaussian range: [{e_gaussian.min():.4f}, {e_gaussian.max():.4f}], mean = {e_gaussian.mean():.4f}")

    return {
        'n_clusters': len(e_cosine),
        'spearman_rho': float(rho),
        'spearman_p': float(p),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'e_cosine_stats': {
            'min': float(e_cosine.min()),
            'max': float(e_cosine.max()),
            'mean': float(e_cosine.mean()),
            'std': float(e_cosine.std()),
        },
        'e_gaussian_stats': {
            'min': float(e_gaussian.min()),
            'max': float(e_gaussian.max()),
            'mean': float(e_gaussian.mean()),
            'std': float(e_gaussian.std()),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    log("Q01 v2 Test: grad_S normalization comparison")
    log("Dataset: STS-B from HuggingFace")
    log("Model: all-MiniLM-L6-v2")
    log("Seed: 42")
    log("")

    # Step 1: Load STS-B
    log("Step 1: Loading STS-B dataset...")
    from datasets import load_dataset
    stsb = load_dataset('glue', 'stsb', trust_remote_code=True)
    train = stsb['train']
    val = stsb['validation']
    test = stsb['test']
    log(f"  Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")

    # Use train + validation (test labels are hidden in GLUE)
    # Combine train and validation
    all_items = list(train) + list(val)
    log(f"  Using {len(all_items)} items (train + validation)")

    # Step 2: Encode sentences
    log("\nStep 2: Encoding sentences with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Collect all unique sentences
    all_sentences = set()
    for item in all_items:
        all_sentences.add(item['sentence1'])
        all_sentences.add(item['sentence2'])
    all_sentences = list(all_sentences)
    log(f"  Unique sentences: {len(all_sentences)}")

    # Encode in batches
    log("  Encoding...")
    embeddings = model.encode(all_sentences, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=False)
    log(f"  Embedding shape: {embeddings.shape}")

    # Build lookup dict
    embeddings_dict = {s: embeddings[i] for i, s in enumerate(all_sentences)}

    # Step 3: Build clusters
    log("\nStep 3: Building clusters from STS-B similarity bins...")
    clusters = build_clusters_from_stsb(all_items, embeddings_dict,
                                         min_cluster_size=10, max_cluster_size=30)
    log(f"  Total clusters: {len(clusters)}")

    # Distribution across bins
    bin_counts = {}
    for c in clusters:
        key = f"[{c['bin_low']}-{c['bin_high']}]"
        bin_counts[key] = bin_counts.get(key, 0) + 1
    for b, count in sorted(bin_counts.items()):
        log(f"    {b}: {count} clusters")

    # Step 4: Run Test 1
    log("\nStep 4: Running Test 1 (Normalization Comparison)...")
    test1_results = run_test1(all_items, embeddings_dict)

    # Step 5: Run Test 2
    log("\nStep 5: Running Test 2 (Bridge Test)...")
    test2_results = run_test2(clusters)

    # Step 6: Verdict
    log("\n" + "=" * 60)
    log("VERDICT DETERMINATION")
    log("=" * 60)

    # Criteria from pre-registration
    wins = test1_results['wins_vs_alternatives']
    sig_wins = test1_results['significant_wins_p01']
    bridge_rho = test2_results['spearman_rho']

    log(f"  E/grad_S wins vs alternatives: {wins}/5")
    log(f"  Significant wins (p<0.01): {sig_wins}/5")
    log(f"  Bridge Spearman rho: {bridge_rho:.6f}")

    # Determine verdict
    # CONFIRM: wins >= 4 with significance AND bridge > 0.8
    # FALSIFY: losses >= 3 OR bridge < 0.3
    # INCONCLUSIVE: otherwise

    losses = test1_results['losses_vs_alternatives']

    if sig_wins >= 4 and bridge_rho > 0.8:
        verdict = "CONFIRMED"
        reasoning = (f"E/grad_S significantly outperforms {sig_wins}/5 alternatives (p<0.01) "
                     f"and bridge correlation is {bridge_rho:.4f} > 0.8")
    elif losses >= 3 or bridge_rho < 0.3:
        if losses >= 3 and bridge_rho < 0.3:
            verdict = "FALSIFIED"
            reasoning = (f"E/grad_S loses to {losses}/5 alternatives AND "
                         f"bridge correlation is {bridge_rho:.4f} < 0.3")
        elif losses >= 3:
            verdict = "FALSIFIED"
            reasoning = f"E/grad_S loses to {losses}/5 alternatives"
        else:
            verdict = "FALSIFIED"
            reasoning = f"Bridge correlation is {bridge_rho:.4f} < 0.3"
    else:
        verdict = "INCONCLUSIVE"
        reasoning = (f"E/grad_S wins {wins}/5 (sig: {sig_wins}/5), "
                     f"bridge rho = {bridge_rho:.4f}. Mixed results.")

    log(f"\n  VERDICT: {verdict}")
    log(f"  REASON: {reasoning}")

    elapsed = time.time() - start_time
    log(f"\nTotal runtime: {elapsed:.1f}s")

    # Save results
    full_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'E/grad_S is the correct normalization for E',
        'dataset': 'STS-B (train+validation)',
        'model': 'all-MiniLM-L6-v2',
        'seed': 42,
        'n_items': len(all_items),
        'n_unique_sentences': len(all_sentences),
        'embedding_dim': int(embeddings.shape[1]),
        'test1_normalization_comparison': test1_results,
        'test2_bridge': test2_results,
        'verdict': verdict,
        'verdict_reasoning': reasoning,
        'runtime_seconds': round(elapsed, 1),
    }

    results_path = os.path.join(RESULTS_DIR, 'test_v2_q01_results.json')
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    log(f"\nResults saved to: {results_path}")

    return full_results


if __name__ == '__main__':
    results = main()
