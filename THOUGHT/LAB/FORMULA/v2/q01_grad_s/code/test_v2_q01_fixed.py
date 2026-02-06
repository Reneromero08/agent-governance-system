"""
Q01 Fixed Test: Is grad_S the correct normalization for E in R = (E/grad_S) * sigma^Df?

Methodology improvements over v1:
- Multiple architectures (3 sentence transformer models)
- Natural topic-coherent clusters (20 Newsgroups, not artificial bins)
- Silhouette score as ground truth (not synthetic correctness metrics)
- Bootstrap comparison with 10,000 resamples
- Partial correlation to test whether grad_S adds value beyond E alone

NO synthetic data. NO reward-maxing. Just truth.
"""

import importlib.util
import sys
import os
import json
import time
import warnings
import numpy as np
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ---- Load formula module ----
FORMULA_PATH = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
spec = importlib.util.spec_from_file_location("formula", FORMULA_PATH)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import silhouette_samples
from scipy import stats
from sentence_transformers import SentenceTransformer


# ---- Configuration ----
ARCHITECTURES = [
    "all-MiniLM-L6-v2",           # 384-dim
    "all-mpnet-base-v2",           # 768-dim
    "multi-qa-MiniLM-L6-cos-v1",  # 384-dim
]
MAX_DOCS_PER_CLUSTER = 200
SILHOUETTE_SAMPLE_TARGET = 100     # docs from target cluster for silhouette
SILHOUETTE_SAMPLE_OTHER = 1900     # docs from other clusters for silhouette
BOOTSTRAP_N = 10000
RANDOM_SEED = 42
RESULTS_PATH = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\q01_grad_s\results\test_v2_q01_fixed_results.json"


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def compute_pairwise_sims(embeddings):
    """Return upper-triangle pairwise cosine similarities."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    upper_indices = np.triu_indices(n, k=1)
    return sim_matrix[upper_indices]


def compute_MAD(embeddings):
    """Median Absolute Deviation of pairwise cosine similarities."""
    sims = compute_pairwise_sims(embeddings)
    return float(np.median(np.abs(sims - np.median(sims))))


def compute_IQR(embeddings):
    """Interquartile Range of pairwise cosine similarities."""
    sims = compute_pairwise_sims(embeddings)
    q75, q25 = np.percentile(sims, [75, 25])
    return float(q75 - q25)


def compute_variants(embeddings):
    """Compute all 5 formula variants for a cluster."""
    E = formula.compute_E(embeddings)
    grad_S = formula.compute_grad_S(embeddings)
    mad = compute_MAD(embeddings)
    iqr = compute_IQR(embeddings)

    # Guard against division by zero
    eps = 1e-10

    R0 = E / grad_S if grad_S > eps else float('nan')       # claimed form
    R1 = E / (grad_S ** 2) if grad_S > eps else float('nan') # precision-weighted
    R2 = E / mad if mad > eps else float('nan')              # robust median
    R3 = E / iqr if iqr > eps else float('nan')              # robust quartile
    R4 = E                                                    # no normalization

    return {
        'E': E,
        'grad_S': grad_S,
        'MAD': mad,
        'IQR': iqr,
        'R0_E_div_gradS': R0,
        'R1_E_div_gradS2': R1,
        'R2_E_div_MAD': R2,
        'R3_E_div_IQR': R3,
        'R4_E_alone': R4,
    }


def compute_silhouette_for_cluster(all_embeddings, all_labels, target_label, rng):
    """
    Compute silhouette score for a specific cluster vs all others.
    Subsamples to keep computation tractable.
    """
    target_mask = all_labels == target_label
    other_mask = ~target_mask

    target_indices = np.where(target_mask)[0]
    other_indices = np.where(other_mask)[0]

    # Subsample
    n_target = min(SILHOUETTE_SAMPLE_TARGET, len(target_indices))
    n_other = min(SILHOUETTE_SAMPLE_OTHER, len(other_indices))

    sampled_target = rng.choice(target_indices, size=n_target, replace=False)
    sampled_other = rng.choice(other_indices, size=n_other, replace=False)

    sampled_indices = np.concatenate([sampled_target, sampled_other])
    sampled_embeddings = all_embeddings[sampled_indices]
    sampled_labels = np.array([1] * n_target + [0] * n_other)  # binary: target vs other

    sil_scores = silhouette_samples(sampled_embeddings, sampled_labels, metric='cosine')

    # Average silhouette for target cluster docs only
    return float(np.mean(sil_scores[:n_target]))


def spearman_corr(x, y):
    """Spearman correlation, handling NaN by dropping pairs."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 4:
        return float('nan'), float('nan')
    rho, p = stats.spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def bootstrap_correlation_comparison(x_list, y, n_bootstrap=BOOTSTRAP_N, seed=RANDOM_SEED):
    """
    Bootstrap test: does R0 (x_list[0]) significantly outperform each alternative?
    Returns dict mapping alternative index -> {delta_mean, p_value, ci_lower, ci_upper}.
    """
    rng = np.random.RandomState(seed)

    # Pre-compute valid masks for each variant
    x_arrays = [np.array(x, dtype=float) for x in x_list]
    y_arr = np.array(y, dtype=float)

    results = {}
    r0 = x_arrays[0]

    for alt_idx in range(1, len(x_arrays)):
        r_alt = x_arrays[alt_idx]
        # Use only indices valid for both R0 and the alternative
        valid = ~(np.isnan(r0) | np.isnan(r_alt) | np.isnan(y_arr))
        valid_indices = np.where(valid)[0]

        if len(valid_indices) < 4:
            results[alt_idx] = {
                'delta_mean': float('nan'),
                'p_value': float('nan'),
                'ci_lower': float('nan'),
                'ci_upper': float('nan'),
                'n_valid': int(len(valid_indices)),
            }
            continue

        deltas = []
        for _ in range(n_bootstrap):
            boot_idx = rng.choice(valid_indices, size=len(valid_indices), replace=True)
            rho_r0, _ = stats.spearmanr(r0[boot_idx], y_arr[boot_idx])
            rho_alt, _ = stats.spearmanr(r_alt[boot_idx], y_arr[boot_idx])
            deltas.append(rho_r0 - rho_alt)

        deltas = np.array(deltas)
        delta_mean = float(np.mean(deltas))
        # One-sided p-value: proportion of bootstrap samples where R0 does NOT beat alt
        p_value = float(np.mean(deltas <= 0))
        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))

        results[alt_idx] = {
            'delta_mean': delta_mean,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_valid': int(len(valid_indices)),
        }

    return results


def partial_correlation(x, y, z):
    """
    Partial correlation between x and y, controlling for z.
    Uses Spearman ranks.
    All inputs are 1-D arrays of the same length.
    Returns (partial_rho, p_value).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < 5:
        return float('nan'), float('nan')

    # Rank transform for Spearman
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    z_rank = stats.rankdata(z)

    # Regress x_rank on z_rank, get residuals
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z_rank, x_rank)
    resid_x = x_rank - (slope_xz * z_rank + intercept_xz)

    # Regress y_rank on z_rank, get residuals
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z_rank, y_rank)
    resid_y = y_rank - (slope_yz * z_rank + intercept_yz)

    # Correlate residuals
    rho, p = stats.pearsonr(resid_x, resid_y)

    # Adjust p-value for degrees of freedom (n - 3 for partial correlation)
    n = len(x)
    if abs(rho) < 1.0:
        t_stat = rho * np.sqrt((n - 3) / (1 - rho**2))
        p = 2 * stats.t.sf(abs(t_stat), df=n - 3)

    return float(rho), float(p)


def subsample_dataset(texts, labels, unique_labels, rng, per_cluster=200,
                      background_per_cluster=100):
    """
    Create a smart subsample of the dataset.
    - per_cluster docs for each cluster (for formula computation)
    - background_per_cluster additional docs per cluster (for silhouette background)
    Returns: (subsample_texts, subsample_labels, cluster_indices_map)
    where cluster_indices_map[label] = indices into the subsample arrays for that cluster
    """
    all_indices = []
    cluster_map = {}
    current_pos = 0

    for label in unique_labels:
        mask = labels == label
        label_indices = np.where(mask)[0]

        # Take up to per_cluster docs
        n_take = min(per_cluster, len(label_indices))
        chosen = rng.choice(label_indices, size=n_take, replace=False)
        cluster_map[int(label)] = list(range(current_pos, current_pos + n_take))
        all_indices.extend(chosen.tolist())
        current_pos += n_take

    all_indices = np.array(all_indices)
    sub_texts = [texts[i] for i in all_indices]
    sub_labels = np.array([labels[i] for i in all_indices])

    return sub_texts, sub_labels, cluster_map, all_indices


def run_test_for_architecture(model_name, all_texts, all_labels, unique_labels, label_names):
    """Run all tests for a single architecture."""
    log(f"\n{'='*70}")
    log(f"ARCHITECTURE: {model_name}")
    log(f"{'='*70}")

    rng = np.random.RandomState(RANDOM_SEED)

    # Subsample dataset to reduce encoding burden
    # We need 200 per cluster for variants + enough for silhouette background
    sub_texts, sub_labels, cluster_map, original_indices = subsample_dataset(
        all_texts, all_labels, unique_labels, rng, per_cluster=MAX_DOCS_PER_CLUSTER
    )

    log(f"  Subsampled to {len(sub_texts)} documents for encoding")

    # Encode only the subsample
    log(f"  Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    log(f"  Encoding {len(sub_texts)} documents...")
    t0 = time.time()
    # Use smaller batch size for larger models to avoid OOM
    batch_sz = 32 if 'mpnet' in model_name else 128
    all_embeddings = model.encode(sub_texts, show_progress_bar=True, batch_size=batch_sz)
    all_embeddings = np.array(all_embeddings)
    encode_time = time.time() - t0
    log(f"  Encoding done in {encode_time:.1f}s. Shape: {all_embeddings.shape}")

    # Reset RNG for reproducibility in silhouette sampling
    rng2 = np.random.RandomState(RANDOM_SEED + 1)

    # ---- Test 1: Normalization Comparison ----
    log(f"\n  --- Test 1: Normalization Comparison ---")

    cluster_data = []
    for label in unique_labels:
        indices = cluster_map[int(label)]
        cluster_emb = all_embeddings[indices]
        variants = compute_variants(cluster_emb)

        # Ground truth: silhouette score of this cluster vs all others
        # Use the full subsampled set
        sil = compute_silhouette_for_cluster(all_embeddings, sub_labels, label, rng2)

        cluster_info = {
            'label': int(label),
            'label_name': label_names[label],
            'n_docs': int(len(indices)),
            'silhouette': sil,
        }
        cluster_info.update(variants)
        cluster_data.append(cluster_info)

        log(f"    Cluster {label:2d} ({label_names[label]:30s}): n={len(indices):4d}, "
            f"E={variants['E']:.4f}, grad_S={variants['grad_S']:.4f}, "
            f"sil={sil:.4f}, R0={variants['R0_E_div_gradS']:.4f}")

    # Compute correlations
    sil_scores = [c['silhouette'] for c in cluster_data]
    variant_keys = ['R0_E_div_gradS', 'R1_E_div_gradS2', 'R2_E_div_MAD', 'R3_E_div_IQR', 'R4_E_alone']
    variant_names = ['E/grad_S', 'E/grad_S^2', 'E/MAD', 'E/IQR', 'E alone']

    correlations = {}
    for key, name in zip(variant_keys, variant_names):
        values = [c[key] for c in cluster_data]
        rho, p = spearman_corr(values, sil_scores)
        correlations[key] = {'rho': rho, 'p': p, 'name': name}
        log(f"    Spearman({name:12s} vs silhouette): rho={rho:+.4f}, p={p:.6f}")

    # Bootstrap comparison
    log(f"\n  Bootstrap comparison (n={BOOTSTRAP_N})...")
    variant_values = [[c[key] for c in cluster_data] for key in variant_keys]
    bootstrap_results = bootstrap_correlation_comparison(variant_values, sil_scores)

    for alt_idx, br in bootstrap_results.items():
        alt_name = variant_names[alt_idx]
        log(f"    R0 vs {alt_name:12s}: delta={br['delta_mean']:+.4f}, "
            f"p={br['p_value']:.4f}, 95%CI=[{br['ci_lower']:+.4f}, {br['ci_upper']:+.4f}]")

    # ---- Test 2: Partial Correlation (grad_S value beyond E) ----
    log(f"\n  --- Test 2: Partial Correlation ---")

    R0_values = [c['R0_E_div_gradS'] for c in cluster_data]
    E_values = [c['R4_E_alone'] for c in cluster_data]
    gradS_values = [c['grad_S'] for c in cluster_data]

    # Does R0 predict silhouette BEYOND what E alone predicts?
    partial_rho_R0, partial_p_R0 = partial_correlation(R0_values, sil_scores, E_values)
    log(f"    Partial corr(R0, silhouette | E): rho={partial_rho_R0:+.4f}, p={partial_p_R0:.6f}")

    # Does grad_S itself predict silhouette controlling for E?
    partial_rho_gradS, partial_p_gradS = partial_correlation(gradS_values, sil_scores, E_values)
    log(f"    Partial corr(grad_S, silhouette | E): rho={partial_rho_gradS:+.4f}, p={partial_p_gradS:.6f}")

    # Also test: does std of norms add value?
    norms_std = []
    for label in unique_labels:
        indices = cluster_map[int(label)]
        cluster_emb = all_embeddings[indices]
        norms_vec = np.linalg.norm(cluster_emb, axis=1)
        norms_std.append(float(np.std(norms_vec)))

    # E / std_of_norms
    E_div_norm_std = []
    for e_val, ns in zip(E_values, norms_std):
        if ns > 1e-10:
            E_div_norm_std.append(e_val / ns)
        else:
            E_div_norm_std.append(float('nan'))

    rho_norm_std, p_norm_std = spearman_corr(E_div_norm_std, sil_scores)
    log(f"    Spearman(E/std_norms vs silhouette): rho={rho_norm_std:+.4f}, p={p_norm_std:.6f}")

    # Compile results for this architecture
    arch_result = {
        'model': model_name,
        'embedding_dim': int(all_embeddings.shape[1]),
        'n_documents_encoded': int(len(sub_texts)),
        'n_clusters': int(len(unique_labels)),
        'encode_time_seconds': round(encode_time, 1),
        'cluster_data': cluster_data,
        'correlations': {k: v for k, v in correlations.items()},
        'bootstrap_vs_alternatives': {
            variant_names[alt_idx]: br for alt_idx, br in bootstrap_results.items()
        },
        'partial_correlation_R0_sil_given_E': {
            'rho': partial_rho_R0, 'p': partial_p_R0
        },
        'partial_correlation_gradS_sil_given_E': {
            'rho': partial_rho_gradS, 'p': partial_p_gradS
        },
        'E_div_norm_std_correlation': {
            'rho': rho_norm_std, 'p': p_norm_std
        },
    }

    # ---- Determine per-architecture verdict ----
    # Count how many alternatives R0 beats significantly (p < 0.01)
    wins = 0
    losses = 0
    for alt_idx, br in bootstrap_results.items():
        if br['p_value'] < 0.01:
            wins += 1  # R0 significantly better
        elif br['p_value'] > 0.99:
            losses += 1  # R0 significantly worse

    arch_result['wins_at_p01'] = wins
    arch_result['losses'] = losses
    arch_result['partial_corr_significant'] = bool(partial_p_gradS < 0.05)

    log(f"\n  SUMMARY for {model_name}:")
    log(f"    R0 correlation with silhouette: {correlations['R0_E_div_gradS']['rho']:+.4f}")
    log(f"    R0 wins (p<0.01): {wins}/4")
    log(f"    R0 losses: {losses}/4")
    log(f"    Partial corr grad_S|E significant (p<0.05): {partial_p_gradS < 0.05}")

    # Clean up model to free memory
    del model
    import gc
    gc.collect()

    return arch_result


def determine_overall_verdict(all_results):
    """
    Pre-registered criteria:
    CONFIRM: E/grad_S outperforms >= 4/5 alternatives (p < 0.01) across ALL 3 architectures,
             AND partial correlation of grad_S with ground truth (after controlling for E) is significant.
    FALSIFY: E/grad_S loses to >= 3 alternatives in ANY architecture,
             OR grad_S partial correlation is non-significant in ALL architectures.
    INCONCLUSIVE: otherwise.
    """
    all_win_4plus = all(r['wins_at_p01'] >= 4 for r in all_results)
    any_partial_significant = any(r['partial_corr_significant'] for r in all_results)
    all_partial_significant = all(r['partial_corr_significant'] for r in all_results)

    confirm = all_win_4plus and all_partial_significant

    any_loses_3plus = any(r['losses'] >= 3 for r in all_results)
    no_partial_significant = not any_partial_significant

    falsify = any_loses_3plus or no_partial_significant

    if confirm:
        return "CONFIRMED"
    elif falsify:
        return "FALSIFIED"
    else:
        return "INCONCLUSIVE"


def main():
    log("=" * 70)
    log("Q01 FIXED TEST: Is grad_S the correct normalization?")
    log("=" * 70)
    log(f"Date: {datetime.now(timezone.utc).isoformat()}")
    log(f"Architectures: {ARCHITECTURES}")
    log(f"Bootstrap N: {BOOTSTRAP_N}")
    log(f"Random seed: {RANDOM_SEED}")
    log("")

    # ---- Load data ----
    log("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_texts = data.data
    all_labels = np.array(data.target)
    unique_labels = np.unique(all_labels)
    label_names = data.target_names

    # Filter out empty documents
    valid_mask = np.array([len(t.strip()) > 10 for t in all_texts])
    all_texts_filtered = [t for t, v in zip(all_texts, valid_mask) if v]
    all_labels_filtered = all_labels[valid_mask]

    log(f"Loaded {len(all_texts_filtered)} documents across {len(unique_labels)} categories")
    for i, name in enumerate(label_names):
        count = np.sum(all_labels_filtered == i)
        log(f"  {i:2d}: {name:30s} ({count} docs)")

    # ---- Run tests for each architecture ----
    all_results = []
    total_start = time.time()
    for model_name in ARCHITECTURES:
        result = run_test_for_architecture(
            model_name, all_texts_filtered, all_labels_filtered, unique_labels, label_names
        )
        all_results.append(result)
        log(f"\n  [Elapsed: {time.time() - total_start:.0f}s]")

    # ---- Overall verdict ----
    verdict = determine_overall_verdict(all_results)

    log(f"\n{'='*70}")
    log(f"OVERALL VERDICT: {verdict}")
    log(f"{'='*70}")

    # Print summary table
    log(f"\n{'='*70}")
    log("SUMMARY TABLE")
    log(f"{'='*70}")
    log(f"{'Model':35s} | {'rho(R0)':>8s} | {'rho(E)':>8s} | {'Wins':>5s} | {'Loss':>5s} | {'Partial p':>10s}")
    log("-" * 85)
    for r in all_results:
        rho_r0 = r['correlations']['R0_E_div_gradS']['rho']
        rho_e = r['correlations']['R4_E_alone']['rho']
        log(f"{r['model']:35s} | {rho_r0:>+8.4f} | {rho_e:>+8.4f} | {r['wins_at_p01']:>5d} | {r['losses']:>5d} | {r['partial_correlation_gradS_sil_given_E']['p']:>10.6f}")

    total_time = time.time() - total_start
    log(f"\nTotal runtime: {total_time:.0f}s")

    # ---- Compile and save results ----
    output = {
        'test': 'Q01_grad_S_normalization_fixed',
        'version': 'v2_fixed',
        'date': datetime.now(timezone.utc).isoformat(),
        'total_runtime_seconds': round(total_time, 1),
        'methodology': {
            'dataset': '20 Newsgroups (sklearn)',
            'n_documents_total': len(all_texts_filtered),
            'n_clusters': int(len(unique_labels)),
            'architectures': ARCHITECTURES,
            'max_docs_per_cluster': MAX_DOCS_PER_CLUSTER,
            'silhouette_sample_target': SILHOUETTE_SAMPLE_TARGET,
            'silhouette_sample_other': SILHOUETTE_SAMPLE_OTHER,
            'bootstrap_n': BOOTSTRAP_N,
            'random_seed': RANDOM_SEED,
            'ground_truth': 'silhouette score (cluster vs all others, cosine metric)',
            'variants_tested': [
                'R0 = E / grad_S (claimed form)',
                'R1 = E / grad_S^2 (precision-weighted)',
                'R2 = E / MAD (robust median)',
                'R3 = E / IQR (robust quartile)',
                'R4 = E alone (no normalization)',
            ],
        },
        'pre_registered_criteria': {
            'confirm': 'E/grad_S outperforms >= 4/4 alternatives (p<0.01) across ALL 3 architectures AND partial corr of grad_S significant in ALL architectures',
            'falsify': 'E/grad_S loses to >= 3 alternatives in ANY architecture OR grad_S partial corr non-significant in ALL architectures',
        },
        'per_architecture_results': all_results,
        'overall_verdict': verdict,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nResults saved to: {RESULTS_PATH}")
    return output


if __name__ == '__main__':
    results = main()
