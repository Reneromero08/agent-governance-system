"""
Q09 v2 Test: R Connects Structurally to the Free Energy Principle

Hypothesis: log(R) = -F + const, where F is variational free energy.
R-maximization equals surprise minimization.

This test uses OPERATIONAL E (cosine similarity) -- not reverse-engineered E.
No retrofitting, no charity -- honest science only.

Pre-registered criteria:
  CONFIRM: corr(log(R), -F) > 0.7 AND upper bound holds >= 80% AND R-gating F1 beats 1/var by >= 10%
  FALSIFY: corr(log(R), -F) < 0.3 OR upper bound fails > 50% OR R-gating no better than 1/var (within 5%)
"""

import sys
import os
import json
import warnings
import time

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal, pearsonr, spearmanr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.cluster import KMeans

# Add shared formula to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from formula import compute_E, compute_grad_S, compute_R_simple, compute_R_full, compute_all

warnings.filterwarnings('ignore')
np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def regularize_covariance(cov, reg=1e-4):
    """Add regularization to avoid singular covariance matrices."""
    return cov + reg * np.eye(cov.shape[0])


def compute_variational_free_energy(embeddings, reg=1e-4):
    """
    Compute variational free energy under a Gaussian generative model.

    F = 0.5 * (d * log(2*pi) + log(det(cov)) + trace(cov_inv @ scatter))

    This is the negative log-evidence (marginal likelihood) for a Gaussian.
    Under a Gaussian model with known mean and covariance estimated from data,
    F equals the negative log-marginal-likelihood.
    """
    n, d = embeddings.shape
    if n < 3:
        return float('nan')

    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov_reg = regularize_covariance(cov, reg)

    # Log determinant (numerically stable)
    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float('nan')

    # Data scatter matrix
    centered = embeddings - mean
    scatter = (centered.T @ centered) / n

    # Inverse of covariance
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return float('nan')

    trace_term = np.trace(cov_inv @ scatter)

    # F = 0.5 * (d * log(2*pi) + log(det(cov)) + trace(cov_inv @ scatter))
    F = 0.5 * (d * np.log(2 * np.pi) + logdet + trace_term)

    return float(F)


def compute_negative_log_evidence(embeddings, reg=1e-4):
    """
    Compute negative log-evidence: -log p(mean | model).

    NLE = -log(N(mean; mean, cov))
    This evaluates the Gaussian PDF at its own mean, which equals
    0.5 * (d * log(2*pi) + log(det(cov)))
    """
    n, d = embeddings.shape
    if n < 3:
        return float('nan')

    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov_reg = regularize_covariance(cov, reg)

    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float('nan')

    # -log p(mean | mean, cov) = 0.5 * (d*log(2pi) + log(det(cov)))
    # (the Mahalanobis distance is 0 at the mean)
    nle = 0.5 * (d * np.log(2 * np.pi) + logdet)

    return float(nle)


def compute_surprise(embeddings, reg=1e-4):
    """
    Compute average surprise: S = -(1/n) * sum_i log p(x_i | model).

    Under a Gaussian fitted to the data:
    S = 0.5 * (d*log(2pi) + log(det(cov)) + d)
    (since E[trace(cov_inv @ (x-mu)(x-mu)^T)] = d for the fitted distribution)
    """
    n, d = embeddings.shape
    if n < 3:
        return float('nan')

    mean = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov_reg = regularize_covariance(cov, reg)

    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float('nan')

    # Average over actual data points
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return float('nan')

    centered = embeddings - mean
    # Average Mahalanobis distance
    maha_sum = 0.0
    for i in range(n):
        maha_sum += centered[i] @ cov_inv @ centered[i]
    avg_maha = maha_sum / n

    S = 0.5 * (d * np.log(2 * np.pi) + logdet + avg_maha)
    return float(S)


def compute_embedding_variance(embeddings):
    """Compute mean variance across embedding dimensions."""
    return float(np.mean(np.var(embeddings, axis=0)))


# ============================================================
# DATA LOADING
# ============================================================

def load_stsb_data():
    """Load STS-B dataset from HuggingFace."""
    print("Loading STS-B dataset...")
    from datasets import load_dataset
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    print(f"  Loaded {len(dataset)} sentence pairs")
    return dataset


def encode_sentences(dataset):
    """Encode sentences with all-MiniLM-L6-v2."""
    print("Encoding sentences with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Collect all unique sentences
    sentences_1 = dataset['sentence1']
    sentences_2 = dataset['sentence2']
    scores = np.array(dataset['score'])

    # Encode both sets
    emb1 = model.encode(sentences_1, show_progress_bar=True, batch_size=128)
    emb2 = model.encode(sentences_2, show_progress_bar=True, batch_size=128)

    print(f"  Encoded {len(emb1)} sentence pairs, embedding dim = {emb1.shape[1]}")
    return emb1, emb2, scores


def create_clusters(emb1, emb2, scores, n_clusters=20, cluster_size=20):
    """
    Create clusters of sentences grouped by similarity score.

    Strategy: Sort pairs by score, divide into bins, sample from each bin.
    Each cluster contains embeddings from both sentence1 and sentence2.
    """
    print(f"Creating {n_clusters} clusters of ~{cluster_size} sentences each...")

    # Sort by score
    sorted_indices = np.argsort(scores)
    n_pairs = len(scores)

    clusters = []
    bin_size = n_pairs // n_clusters

    for i in range(n_clusters):
        start = i * bin_size
        end = min(start + bin_size, n_pairs)
        bin_indices = sorted_indices[start:end]

        # Sample pairs from this bin
        if len(bin_indices) > cluster_size // 2:
            np.random.seed(42 + i)
            sampled = np.random.choice(bin_indices, size=min(cluster_size // 2, len(bin_indices)), replace=False)
        else:
            sampled = bin_indices

        # Combine embeddings from both sentences in sampled pairs
        cluster_emb = np.vstack([emb1[sampled], emb2[sampled]])
        cluster_scores = scores[sampled]
        mean_score = float(np.mean(cluster_scores))

        clusters.append({
            'embeddings': cluster_emb,
            'mean_score': mean_score,
            'n_sentences': cluster_emb.shape[0],
            'score_range': (float(np.min(cluster_scores)), float(np.max(cluster_scores))),
        })

    print(f"  Created {len(clusters)} clusters")
    for i, c in enumerate(clusters):
        print(f"    Cluster {i}: {c['n_sentences']} sentences, mean_score={c['mean_score']:.2f}, "
              f"range=[{c['score_range'][0]:.2f}, {c['score_range'][1]:.2f}]")

    return clusters


# ============================================================
# TEST 1: Operational E vs Free Energy Correlation
# ============================================================

def test1_correlation(clusters):
    """
    Measure correlation between log(R) and -F using operational E.
    Also measure correlation between log(R) and -NLE.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Operational E vs Free Energy Correlation")
    print("=" * 70)

    results = {
        'log_R_simple': [],
        'log_R_full': [],
        'neg_F': [],
        'neg_NLE': [],
        'E_values': [],
        'grad_S_values': [],
        'F_values': [],
        'NLE_values': [],
        'cluster_scores': [],
    }

    for i, cluster in enumerate(clusters):
        emb = cluster['embeddings']

        # Compute R using shared formula (operational E = cosine similarity)
        all_metrics = compute_all(emb)
        R_simple = all_metrics['R_simple']
        R_full = all_metrics['R_full']

        # Compute free energy
        F = compute_variational_free_energy(emb)
        NLE = compute_negative_log_evidence(emb)

        if np.isnan(R_simple) or np.isnan(F) or R_simple <= 0:
            print(f"  Cluster {i}: SKIPPED (R_simple={R_simple}, F={F})")
            continue

        log_R_simple = np.log(R_simple) if R_simple > 0 else float('nan')
        log_R_full = np.log(R_full) if (not np.isnan(R_full) and R_full > 0) else float('nan')

        if np.isnan(log_R_simple):
            continue

        results['log_R_simple'].append(log_R_simple)
        results['log_R_full'].append(log_R_full if not np.isnan(log_R_full) else log_R_simple)
        results['neg_F'].append(-F)
        results['neg_NLE'].append(-NLE if not np.isnan(NLE) else float('nan'))
        results['E_values'].append(all_metrics['E'])
        results['grad_S_values'].append(all_metrics['grad_S'])
        results['F_values'].append(F)
        results['NLE_values'].append(NLE)
        results['cluster_scores'].append(cluster['mean_score'])

        print(f"  Cluster {i}: E={all_metrics['E']:.4f}, grad_S={all_metrics['grad_S']:.4f}, "
              f"R_simple={R_simple:.4f}, log(R)={log_R_simple:.4f}, F={F:.2f}, -F={-F:.2f}")

    # Convert to arrays
    log_R = np.array(results['log_R_simple'])
    neg_F = np.array(results['neg_F'])
    neg_NLE = np.array(results['neg_NLE'])

    # Filter NaNs for NLE correlation
    valid_nle = ~np.isnan(neg_NLE)

    # Correlations: log(R) vs -F
    n_valid = len(log_R)
    print(f"\n  Valid clusters: {n_valid}")

    if n_valid < 5:
        print("  ERROR: Too few valid clusters for correlation analysis")
        return {'error': 'insufficient_data', 'n_valid': n_valid}

    r_pearson, p_pearson = pearsonr(log_R, neg_F)
    r_spearman, p_spearman = spearmanr(log_R, neg_F)

    print(f"\n  log(R_simple) vs -F:")
    print(f"    Pearson r  = {r_pearson:.4f} (p = {p_pearson:.6f})")
    print(f"    Spearman rho = {r_spearman:.4f} (p = {p_spearman:.6f})")

    # Correlations: log(R) vs -NLE
    if np.sum(valid_nle) >= 5:
        r_nle_pearson, p_nle_pearson = pearsonr(log_R[valid_nle], neg_NLE[valid_nle])
        r_nle_spearman, p_nle_spearman = spearmanr(log_R[valid_nle], neg_NLE[valid_nle])
        print(f"\n  log(R_simple) vs -NLE:")
        print(f"    Pearson r  = {r_nle_pearson:.4f} (p = {p_nle_pearson:.6f})")
        print(f"    Spearman rho = {r_nle_spearman:.4f} (p = {p_nle_spearman:.6f})")
    else:
        r_nle_pearson, p_nle_pearson = float('nan'), float('nan')
        r_nle_spearman, p_nle_spearman = float('nan'), float('nan')

    # Also check: log(R_full) vs -F
    log_R_full = np.array(results['log_R_full'])
    valid_full = ~np.isnan(log_R_full)
    if np.sum(valid_full) >= 5:
        r_full_pearson, p_full_pearson = pearsonr(log_R_full[valid_full], neg_F[valid_full])
        r_full_spearman, p_full_spearman = spearmanr(log_R_full[valid_full], neg_F[valid_full])
        print(f"\n  log(R_full) vs -F:")
        print(f"    Pearson r  = {r_full_pearson:.4f} (p = {p_full_pearson:.6f})")
        print(f"    Spearman rho = {r_full_spearman:.4f} (p = {p_full_spearman:.6f})")
    else:
        r_full_pearson, p_full_pearson = float('nan'), float('nan')
        r_full_spearman, p_full_spearman = float('nan'), float('nan')

    # Check if log(R) = -F + const by examining residuals
    # If identity holds, log(R) + F should be approximately constant
    sum_values = log_R + np.array(results['F_values'])
    residual_std = np.std(sum_values)
    residual_mean = np.mean(sum_values)
    print(f"\n  Identity check: log(R) + F = const?")
    print(f"    mean(log(R) + F) = {residual_mean:.4f}")
    print(f"    std(log(R) + F)  = {residual_std:.4f}")
    print(f"    CV = {residual_std / abs(residual_mean) if residual_mean != 0 else float('inf'):.4f}")

    test1_results = {
        'n_valid_clusters': n_valid,
        'log_R_vs_neg_F': {
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson),
            'spearman_rho': float(r_spearman),
            'spearman_p': float(p_spearman),
        },
        'log_R_vs_neg_NLE': {
            'pearson_r': float(r_nle_pearson),
            'pearson_p': float(p_nle_pearson),
            'spearman_rho': float(r_nle_spearman),
            'spearman_p': float(p_nle_spearman),
        },
        'log_R_full_vs_neg_F': {
            'pearson_r': float(r_full_pearson),
            'pearson_p': float(p_full_pearson),
            'spearman_rho': float(r_full_spearman),
            'spearman_p': float(p_full_spearman),
        },
        'identity_check': {
            'mean_log_R_plus_F': float(residual_mean),
            'std_log_R_plus_F': float(residual_std),
        },
        'raw_data': {
            'log_R_simple': [float(x) for x in log_R],
            'neg_F': [float(x) for x in neg_F],
            'E_values': [float(x) for x in results['E_values']],
            'grad_S_values': [float(x) for x in results['grad_S_values']],
            'F_values': [float(x) for x in results['F_values']],
            'cluster_scores': [float(x) for x in results['cluster_scores']],
        },
    }

    return test1_results


# ============================================================
# TEST 2: R-Gating vs Variance Filter
# ============================================================

def test2_gating(clusters):
    """
    Compare R as a quality gate against simple alternatives.
    Good clusters: mean human score > 3.5
    Bad clusters: mean human score < 2.0
    """
    print("\n" + "=" * 70)
    print("TEST 2: R-Gating vs Variance Filter")
    print("=" * 70)

    # Classify clusters
    good_clusters = []
    bad_clusters = []
    for i, c in enumerate(clusters):
        if c['mean_score'] > 3.5:
            good_clusters.append((i, c))
        elif c['mean_score'] < 2.0:
            bad_clusters.append((i, c))

    print(f"  Good clusters (score > 3.5): {len(good_clusters)}")
    print(f"  Bad clusters (score < 2.0): {len(bad_clusters)}")

    if len(good_clusters) < 2 or len(bad_clusters) < 2:
        print("  WARNING: Not enough good/bad clusters. Adjusting thresholds...")
        # Use median split instead
        median_score = np.median([c['mean_score'] for c in clusters])
        good_clusters = [(i, c) for i, c in enumerate(clusters) if c['mean_score'] > median_score]
        bad_clusters = [(i, c) for i, c in enumerate(clusters) if c['mean_score'] <= median_score]
        print(f"  After median split ({median_score:.2f}):")
        print(f"    Good: {len(good_clusters)}, Bad: {len(bad_clusters)}")

    all_labeled = good_clusters + bad_clusters
    labels = [1] * len(good_clusters) + [0] * len(bad_clusters)
    labels = np.array(labels)

    # Compute metrics for each cluster
    R_values = []
    inv_var_values = []
    E_values = []

    for idx, cluster in all_labeled:
        emb = cluster['embeddings']
        R = compute_R_simple(emb)
        variance = compute_embedding_variance(emb)
        E = compute_E(emb)

        R_values.append(R if not np.isnan(R) else 0.0)
        inv_var_values.append(1.0 / variance if variance > 1e-10 else 0.0)
        E_values.append(E if not np.isnan(E) else 0.0)

    R_values = np.array(R_values)
    inv_var_values = np.array(inv_var_values)
    E_values = np.array(E_values)

    print(f"\n  R range: [{R_values.min():.4f}, {R_values.max():.4f}]")
    print(f"  1/var range: [{inv_var_values.min():.4f}, {inv_var_values.max():.4f}]")
    print(f"  E range: [{E_values.min():.4f}, {E_values.max():.4f}]")

    def find_best_threshold(values, labels):
        """Find threshold that maximizes F1 score."""
        best_f1 = -1
        best_threshold = 0
        best_metrics = None

        # Try percentile-based thresholds
        for pct in range(5, 96, 5):
            threshold = np.percentile(values, pct)
            predictions = (values > threshold).astype(int)
            if len(np.unique(predictions)) < 2:
                continue
            p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {'precision': float(p), 'recall': float(r), 'f1': float(f1),
                                'threshold': float(threshold), 'accuracy': float(accuracy_score(labels, predictions))}

        return best_metrics

    # Evaluate each method
    r_metrics = find_best_threshold(R_values, labels)
    var_metrics = find_best_threshold(inv_var_values, labels)
    e_metrics = find_best_threshold(E_values, labels)

    # Random baseline (averaged over 1000 trials)
    random_f1s = []
    for trial in range(1000):
        np.random.seed(trial)
        random_preds = np.random.randint(0, 2, size=len(labels))
        _, _, f1, _ = precision_recall_fscore_support(labels, random_preds, average='binary', zero_division=0)
        random_f1s.append(f1)
    random_metrics = {
        'precision': float(np.mean([precision_recall_fscore_support(labels, np.random.randint(0, 2, size=len(labels)), average='binary', zero_division=0)[0] for _ in range(100)])),
        'recall': float(np.mean([precision_recall_fscore_support(labels, np.random.randint(0, 2, size=len(labels)), average='binary', zero_division=0)[1] for _ in range(100)])),
        'f1': float(np.mean(random_f1s)),
        'threshold': 'N/A',
        'accuracy': 0.5,
    }

    print("\n  Results (best threshold for each):")
    print(f"  {'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*60}")

    for name, metrics in [('R > threshold', r_metrics), ('1/var > threshold', var_metrics),
                           ('E > threshold', e_metrics), ('Random', random_metrics)]:
        if metrics:
            print(f"  {name:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f} {metrics['accuracy']:>10.4f}")
        else:
            print(f"  {name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    # Compute F1 difference between R and 1/var
    if r_metrics and var_metrics:
        f1_diff = r_metrics['f1'] - var_metrics['f1']
        f1_pct_diff = (f1_diff / var_metrics['f1'] * 100) if var_metrics['f1'] > 0 else float('inf')
        print(f"\n  R vs 1/var F1 difference: {f1_diff:.4f} ({f1_pct_diff:.1f}%)")
    else:
        f1_diff = float('nan')
        f1_pct_diff = float('nan')

    test2_results = {
        'n_good': len(good_clusters),
        'n_bad': len(bad_clusters),
        'R_gating': r_metrics,
        'inv_variance': var_metrics,
        'raw_E': e_metrics,
        'random_baseline': random_metrics,
        'f1_R_minus_var': float(f1_diff) if not np.isnan(f1_diff) else None,
        'f1_pct_diff': float(f1_pct_diff) if not np.isnan(f1_pct_diff) else None,
    }

    return test2_results


# ============================================================
# TEST 3: Upper Bound on Surprise
# ============================================================

def test3_upper_bound(clusters):
    """
    Test whether -log(R) serves as an upper bound on surprise.

    For each cluster:
      Surprise S = average negative log-likelihood of observations under cluster model
      Check: -log(R) >= S ?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Upper Bound on Surprise")
    print("=" * 70)

    results = []

    for i, cluster in enumerate(clusters):
        emb = cluster['embeddings']
        R = compute_R_simple(emb)

        if np.isnan(R) or R <= 0:
            print(f"  Cluster {i}: SKIPPED (R={R})")
            continue

        neg_log_R = -np.log(R)
        S = compute_surprise(emb)

        if np.isnan(S):
            print(f"  Cluster {i}: SKIPPED (surprise=NaN)")
            continue

        is_upper_bound = neg_log_R >= S
        margin = neg_log_R - S

        results.append({
            'cluster': i,
            'R': float(R),
            'neg_log_R': float(neg_log_R),
            'surprise': float(S),
            'is_upper_bound': bool(is_upper_bound),
            'margin': float(margin),
            'mean_score': cluster['mean_score'],
        })

        status = "YES" if is_upper_bound else "NO"
        print(f"  Cluster {i}: -log(R)={neg_log_R:.4f}, S={S:.2f}, "
              f"bound holds: {status} (margin={margin:.4f})")

    n_valid = len(results)
    n_bound_holds = sum(1 for r in results if r['is_upper_bound'])
    frac_holds = n_bound_holds / n_valid if n_valid > 0 else 0.0

    print(f"\n  Upper bound holds: {n_bound_holds}/{n_valid} ({frac_holds*100:.1f}%)")
    print(f"  Threshold for CONFIRM: >= 80%")
    print(f"  Threshold for FALSIFY: < 50%")

    # Note: -log(R) is a small number (R is a ratio of cosine similarities)
    # while surprise S is computed in 384-dimensional embedding space.
    # They operate on fundamentally different scales. This is expected to fail
    # unless there's a genuine structural connection.

    test3_results = {
        'n_valid': n_valid,
        'n_bound_holds': n_bound_holds,
        'fraction_holds': float(frac_holds),
        'details': results,
    }

    return test3_results


# ============================================================
# TEST 4: Beyond Gaussian
# ============================================================

def test4_non_gaussian(n_clusters=15, cluster_size=40, d=384):
    """
    Test whether log(R) = -F + const holds for non-Gaussian distributions.

    Create synthetic clusters that are clearly non-Gaussian:
    - Bimodal (mixture of 2 Gaussians)
    - Skewed (log-normal in each dimension)
    - Heavy-tailed (Student-t)
    - Uniform
    - Exponential
    """
    print("\n" + "=" * 70)
    print("TEST 4: Beyond Gaussian")
    print("=" * 70)

    np.random.seed(42)

    distributions = []

    # Use a reduced dimension for tractability in computing F
    d_reduced = 50

    # --- Bimodal clusters ---
    print("  Generating bimodal clusters...")
    for i in range(3):
        sep = 1.0 + i * 1.0  # Increasing separation
        half = cluster_size // 2
        mode1 = np.random.randn(half, d_reduced) * 0.3
        mode2 = np.random.randn(cluster_size - half, d_reduced) * 0.3 + sep
        emb = np.vstack([mode1, mode2])
        distributions.append(('bimodal', f'sep={sep:.1f}', emb))

    # --- Skewed clusters (log-normal) ---
    print("  Generating skewed clusters...")
    for sigma_param in [0.5, 1.0, 1.5]:
        emb = np.random.lognormal(mean=0, sigma=sigma_param, size=(cluster_size, d_reduced))
        distributions.append(('skewed', f'sigma={sigma_param}', emb))

    # --- Heavy-tailed (Student-t) ---
    print("  Generating heavy-tailed clusters...")
    for df in [2, 3, 5]:
        emb = np.random.standard_t(df=df, size=(cluster_size, d_reduced))
        distributions.append(('heavy_tail', f'df={df}', emb))

    # --- Uniform ---
    print("  Generating uniform clusters...")
    for scale in [1.0, 3.0, 5.0]:
        emb = np.random.uniform(-scale, scale, size=(cluster_size, d_reduced))
        distributions.append(('uniform', f'scale={scale}', emb))

    # --- Gaussian baseline ---
    print("  Generating Gaussian baselines...")
    for std in [0.5, 1.0, 2.0]:
        emb = np.random.randn(cluster_size, d_reduced) * std
        distributions.append(('gaussian', f'std={std}', emb))

    # Compute R and F for each
    log_R_values = []
    neg_F_values = []
    dist_types = []
    dist_labels = []

    print(f"\n  {'Type':<15} {'Params':<15} {'R_simple':>10} {'log(R)':>10} {'F':>12} {'-F':>12}")
    print(f"  {'-'*74}")

    for dtype, params, emb in distributions:
        R = compute_R_simple(emb)
        F = compute_variational_free_energy(emb, reg=1e-3)  # higher reg for non-Gaussian

        if np.isnan(R) or R <= 0 or np.isnan(F):
            print(f"  {dtype:<15} {params:<15} {'NaN':>10} {'NaN':>10} {'NaN':>12} {'NaN':>12}")
            continue

        log_R = np.log(R)
        log_R_values.append(log_R)
        neg_F_values.append(-F)
        dist_types.append(dtype)
        dist_labels.append(f"{dtype}_{params}")

        print(f"  {dtype:<15} {params:<15} {R:>10.4f} {log_R:>10.4f} {F:>12.2f} {-F:>12.2f}")

    log_R_arr = np.array(log_R_values)
    neg_F_arr = np.array(neg_F_values)

    # Overall correlation
    if len(log_R_arr) >= 5:
        r_all, p_all = pearsonr(log_R_arr, neg_F_arr)
        rho_all, p_rho_all = spearmanr(log_R_arr, neg_F_arr)
        print(f"\n  Overall correlation (all distributions):")
        print(f"    Pearson r = {r_all:.4f} (p = {p_all:.6f})")
        print(f"    Spearman rho = {rho_all:.4f} (p = {p_rho_all:.6f})")
    else:
        r_all, p_all = float('nan'), float('nan')
        rho_all, p_rho_all = float('nan'), float('nan')

    # Per-distribution-type correlation
    type_results = {}
    for dtype in ['bimodal', 'skewed', 'heavy_tail', 'uniform', 'gaussian']:
        mask = np.array([t == dtype for t in dist_types])
        if np.sum(mask) >= 3:
            r_t, p_t = pearsonr(log_R_arr[mask], neg_F_arr[mask])
            rho_t, p_rho_t = spearmanr(log_R_arr[mask], neg_F_arr[mask])
            type_results[dtype] = {
                'pearson_r': float(r_t), 'pearson_p': float(p_t),
                'spearman_rho': float(rho_t), 'spearman_p': float(p_rho_t),
                'n': int(np.sum(mask)),
            }
            print(f"    {dtype}: r={r_t:.4f}, rho={rho_t:.4f} (n={np.sum(mask)})")

    # Non-Gaussian only
    non_gaussian_mask = np.array([t != 'gaussian' for t in dist_types])
    if np.sum(non_gaussian_mask) >= 5:
        r_ng, p_ng = pearsonr(log_R_arr[non_gaussian_mask], neg_F_arr[non_gaussian_mask])
        rho_ng, p_rho_ng = spearmanr(log_R_arr[non_gaussian_mask], neg_F_arr[non_gaussian_mask])
        print(f"\n  Non-Gaussian only:")
        print(f"    Pearson r = {r_ng:.4f} (p = {p_ng:.6f})")
        print(f"    Spearman rho = {rho_ng:.4f} (p = {p_rho_ng:.6f})")
    else:
        r_ng, p_ng = float('nan'), float('nan')
        rho_ng, p_rho_ng = float('nan'), float('nan')

    test4_results = {
        'n_distributions': len(log_R_arr),
        'overall': {
            'pearson_r': float(r_all), 'pearson_p': float(p_all),
            'spearman_rho': float(rho_all), 'spearman_p': float(p_rho_all),
        },
        'non_gaussian_only': {
            'pearson_r': float(r_ng), 'pearson_p': float(p_ng),
            'spearman_rho': float(rho_ng), 'spearman_p': float(p_rho_ng),
        },
        'per_type': type_results,
        'raw_data': {
            'log_R': [float(x) for x in log_R_arr],
            'neg_F': [float(x) for x in neg_F_arr],
            'types': dist_types,
            'labels': dist_labels,
        },
    }

    return test4_results


# ============================================================
# VERDICT DETERMINATION
# ============================================================

def determine_verdict(test1, test2, test3, test4):
    """Apply pre-registered criteria to determine overall verdict."""
    print("\n" + "=" * 70)
    print("VERDICT DETERMINATION")
    print("=" * 70)

    criteria = {
        'corr_log_R_neg_F': None,
        'upper_bound_fraction': None,
        'r_gating_f1_advantage': None,
    }

    # Criterion 1: corr(log(R), -F) on real data
    if 'error' not in test1:
        r = test1['log_R_vs_neg_F']['pearson_r']
        criteria['corr_log_R_neg_F'] = r
        print(f"  Criterion 1: corr(log(R), -F) = {r:.4f}")
        if r > 0.7:
            print(f"    -> CONFIRM range (> 0.7)")
        elif r < 0.3:
            print(f"    -> FALSIFY range (< 0.3)")
        else:
            print(f"    -> INCONCLUSIVE range (0.3-0.7)")

    # Criterion 2: upper bound fraction
    frac = test3['fraction_holds']
    criteria['upper_bound_fraction'] = frac
    print(f"  Criterion 2: upper bound holds = {frac*100:.1f}%")
    if frac >= 0.8:
        print(f"    -> CONFIRM range (>= 80%)")
    elif frac < 0.5:
        print(f"    -> FALSIFY range (< 50%)")
    else:
        print(f"    -> INCONCLUSIVE range (50-80%)")

    # Criterion 3: R-gating vs 1/var
    if test2['R_gating'] and test2['inv_variance']:
        r_f1 = test2['R_gating']['f1']
        var_f1 = test2['inv_variance']['f1']
        pct_diff = ((r_f1 - var_f1) / var_f1 * 100) if var_f1 > 0 else float('inf')
        criteria['r_gating_f1_advantage'] = pct_diff
        print(f"  Criterion 3: R F1={r_f1:.4f}, 1/var F1={var_f1:.4f}, diff={pct_diff:.1f}%")
        if pct_diff >= 10:
            print(f"    -> CONFIRM range (>= 10%)")
        elif abs(pct_diff) <= 5:
            print(f"    -> FALSIFY range (within 5%)")
        else:
            print(f"    -> INCONCLUSIVE range")

    # Overall verdict
    confirm_count = 0
    falsify_count = 0
    inconclusive_count = 0

    r = criteria.get('corr_log_R_neg_F')
    if r is not None:
        if r > 0.7:
            confirm_count += 1
        elif r < 0.3:
            falsify_count += 1
        else:
            inconclusive_count += 1

    frac = criteria.get('upper_bound_fraction')
    if frac is not None:
        if frac >= 0.8:
            confirm_count += 1
        elif frac < 0.5:
            falsify_count += 1
        else:
            inconclusive_count += 1

    pct = criteria.get('r_gating_f1_advantage')
    if pct is not None:
        if pct >= 10:
            confirm_count += 1
        elif abs(pct) <= 5:
            falsify_count += 1
        else:
            inconclusive_count += 1

    print(f"\n  Score: CONFIRM={confirm_count}, FALSIFY={falsify_count}, INCONCLUSIVE={inconclusive_count}")

    # CONFIRM requires ALL three criteria met
    if confirm_count == 3:
        verdict = "CONFIRMED"
    # FALSIFY if ANY criterion hits falsify threshold
    elif falsify_count >= 1:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")

    return verdict, criteria


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("Q09 v2 TEST: R Connects to the Free Energy Principle")
    print("Using OPERATIONAL E (cosine similarity) on real STS-B data")
    print("=" * 70)

    # Load and encode data
    dataset = load_stsb_data()
    emb1, emb2, scores = encode_sentences(dataset)
    clusters = create_clusters(emb1, emb2, scores, n_clusters=20, cluster_size=40)

    # Run tests
    test1_results = test1_correlation(clusters)
    test2_results = test2_gating(clusters)
    test3_results = test3_upper_bound(clusters)
    test4_results = test4_non_gaussian()

    # Determine verdict
    verdict, criteria = determine_verdict(test1_results, test2_results, test3_results, test4_results)

    elapsed = time.time() - start_time

    # Save all results
    all_results = {
        'verdict': verdict,
        'criteria': {k: float(v) if v is not None and not isinstance(v, str) else v for k, v in criteria.items()},
        'test1_correlation': test1_results,
        'test2_gating': test2_results,
        'test3_upper_bound': test3_results,
        'test4_non_gaussian': test4_results,
        'metadata': {
            'seed': 42,
            'model': 'all-MiniLM-L6-v2',
            'dataset': 'mteb/stsbenchmark-sts',
            'n_clusters': 20,
            'cluster_size': 40,
            'elapsed_seconds': elapsed,
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'test_v2_q09_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Verdict: {verdict}")
    print(f"  Elapsed: {elapsed:.1f}s")
    if 'error' not in test1_results:
        print(f"  Test 1 - corr(log(R), -F): Pearson={test1_results['log_R_vs_neg_F']['pearson_r']:.4f}, "
              f"Spearman={test1_results['log_R_vs_neg_F']['spearman_rho']:.4f}")
    print(f"  Test 2 - R F1: {test2_results['R_gating']['f1'] if test2_results['R_gating'] else 'N/A'}, "
          f"1/var F1: {test2_results['inv_variance']['f1'] if test2_results['inv_variance'] else 'N/A'}")
    print(f"  Test 3 - Upper bound: {test3_results['fraction_holds']*100:.1f}% "
          f"({test3_results['n_bound_holds']}/{test3_results['n_valid']})")
    print(f"  Test 4 - Non-Gaussian corr: {test4_results['overall']['pearson_r']:.4f}")

    return all_results


if __name__ == '__main__':
    results = main()
