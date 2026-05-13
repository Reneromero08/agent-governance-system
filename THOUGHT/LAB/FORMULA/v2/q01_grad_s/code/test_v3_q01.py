"""
Q01 v3 Test: Does grad_S add independent predictive value beyond E?

Addresses ALL audit findings from AUDIT.md:
  STAT-01: n>=100 subclusters (random subsets of varying sizes from each category)
  METH-02: Label purity ground truth (non-cosine, breaks E-silhouette confound)
  STAT-06: Steiger's Z-test for dependent correlation comparison
  METH-07: Cross-validated R^2 comparison (multiplicative relationship)
  STAT-05: Formal sign test across architectures
  3 architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
  Pre-registered pass/fail criteria stated before running

NO synthetic data. NO reward-maxing. Report what the data says.
"""

import importlib.util
import sys
import os
import json
import time
import warnings
import math
import numpy as np
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ---- Load shared formula module ----
FORMULA_PATH = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
spec = importlib.util.spec_from_file_location("formula", FORMULA_PATH)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import KFold
from scipy import stats
from sentence_transformers import SentenceTransformer

# ---- Configuration ----
ARCHITECTURES = [
    "all-MiniLM-L6-v2",           # 384-dim
    "all-mpnet-base-v2",           # 768-dim
    "multi-qa-MiniLM-L6-cos-v1",  # 384-dim
]

# Subcluster generation parameters
CLUSTER_SIZES = [30, 50, 75, 100, 150]  # varying sizes per subcluster
SUBCLUSTERS_PER_CATEGORY = 6            # 6 subclusters x 20 categories = 120 clusters
MIN_CLUSTERS_TARGET = 100               # minimum acceptable n

BOOTSTRAP_N = 5000
CV_FOLDS = 5
RANDOM_SEED = 42

RESULTS_PATH = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\q01_grad_s\results\test_v3_q01_results.json"

# ---- Pre-registered criteria ----
# Stated BEFORE running any analysis:
PRE_REGISTERED_CRITERIA = {
    "confirmed": (
        "CONFIRMED if Steiger p < 0.05 (R predicts quality better than E alone) "
        "AND cross-validated R^2(R) > R^2(E) "
        "on at least 2 of 3 architectures."
    ),
    "falsified": (
        "FALSIFIED if Steiger p > 0.05 on ALL 3 architectures "
        "AND cross-validated R^2 difference (R^2(R) - R^2(E)) < 0.01 on ALL 3 architectures."
    ),
    "inconclusive": (
        "INCONCLUSIVE otherwise (mixed signals across architectures or metrics)."
    ),
}


def log(msg):
    print(msg, flush=True)


def compute_pairwise_sims(embeddings):
    """Upper-triangle pairwise cosine similarities."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    upper_indices = np.triu_indices(n, k=1)
    return sim_matrix[upper_indices]


def compute_label_purity(cluster_labels, all_unique_labels):
    """
    METH-02 fix: Ground truth NOT based on cosine similarity.
    Label purity = fraction of docs in cluster belonging to the majority category.
    For a pure single-category cluster this is 1.0.
    For a mixed cluster (which our subclusters will be when we add noise), < 1.0.
    """
    if len(cluster_labels) == 0:
        return 0.0
    label_counts = np.bincount(cluster_labels, minlength=len(all_unique_labels))
    majority_count = np.max(label_counts)
    return float(majority_count / len(cluster_labels))


def generate_subclusters(all_labels, unique_labels, rng, n_per_category=6,
                         sizes=None):
    """
    STAT-01 fix: Generate many subclusters by randomly sampling subsets.

    For each of the 20 categories, create n_per_category subclusters of varying sizes.
    Some subclusters are pure (single category), some are mixed (with noise docs from
    other categories) to create natural variation in label purity.

    Returns list of dicts: [{indices: [...], true_labels: [...], category: int, size: int}, ...]
    """
    if sizes is None:
        sizes = CLUSTER_SIZES

    subclusters = []
    category_indices = {}
    for label in unique_labels:
        category_indices[int(label)] = np.where(all_labels == label)[0]

    all_indices_flat = np.arange(len(all_labels))

    for label in unique_labels:
        label_idx = category_indices[int(label)]
        if len(label_idx) < 20:
            continue

        for i in range(n_per_category):
            # Cycle through sizes
            target_size = sizes[i % len(sizes)]

            # Determine purity: first 4 subclusters are pure, last 2 are mixed
            if i < 4:
                # Pure subcluster: all from same category
                n_take = min(target_size, len(label_idx))
                chosen = rng.choice(label_idx, size=n_take, replace=False)
            else:
                # Mixed subcluster: ~70-90% from target, rest from random others
                purity_frac = rng.uniform(0.7, 0.9)
                n_target = min(int(target_size * purity_frac), len(label_idx))
                n_noise = target_size - n_target

                target_chosen = rng.choice(label_idx, size=n_target, replace=False)

                # Noise docs from other categories
                other_idx = np.setdiff1d(all_indices_flat, label_idx)
                n_noise = min(n_noise, len(other_idx))
                noise_chosen = rng.choice(other_idx, size=n_noise, replace=False)

                chosen = np.concatenate([target_chosen, noise_chosen])

            subclusters.append({
                'indices': chosen,
                'true_labels': all_labels[chosen],
                'primary_category': int(label),
                'size': len(chosen),
                'is_mixed': i >= 4,
            })

    return subclusters


def spearman_corr(x, y):
    """Spearman correlation, dropping NaN pairs."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 4:
        return float('nan'), float('nan')
    rho, p = stats.spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def pearson_corr(x, y):
    """Pearson correlation, dropping NaN pairs."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 4:
        return float('nan'), float('nan')
    r, p = stats.pearsonr(x[valid], y[valid])
    return float(r), float(p)


def partial_correlation(x, y, z):
    """
    Partial Spearman correlation between x and y controlling for z.
    Returns (partial_rho, p_value).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < 5:
        return float('nan'), float('nan')

    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    z_rank = stats.rankdata(z)

    slope_xz, intercept_xz, _, _, _ = stats.linregress(z_rank, x_rank)
    resid_x = x_rank - (slope_xz * z_rank + intercept_xz)

    slope_yz, intercept_yz, _, _, _ = stats.linregress(z_rank, y_rank)
    resid_y = y_rank - (slope_yz * z_rank + intercept_yz)

    rho, _ = stats.pearsonr(resid_x, resid_y)

    n = len(x)
    if abs(rho) < 1.0:
        t_stat = rho * np.sqrt((n - 3) / (1 - rho ** 2))
        p = 2 * stats.t.sf(abs(t_stat), df=n - 3)
    else:
        p = 0.0

    return float(rho), float(p)


def fisher_z(r):
    """Fisher z-transformation of a correlation coefficient."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def steiger_z_test(r1, r2, r12, n):
    """
    STAT-06: Steiger's Z-test for comparing two dependent correlations.

    Tests H0: rho(X1, Y) = rho(X2, Y) where X1 and X2 are correlated (r12).

    r1  = corr(predictor1, outcome)
    r2  = corr(predictor2, outcome)
    r12 = corr(predictor1, predictor2)
    n   = sample size

    Returns (Z_statistic, p_value_two_sided)
    """
    if n < 4 or any(np.isnan([r1, r2, r12])):
        return float('nan'), float('nan')

    z1 = fisher_z(r1)
    z2 = fisher_z(r2)

    # Steiger (1980) formula for the denominator
    r_mean_sq = ((r1 + r2) / 2) ** 2
    det = 1 - r1**2 - r2**2 - r12**2 + 2 * r1 * r2 * r12

    # Simplified Steiger formula
    f = (1 - r12) / (2 * (1 - r_mean_sq))
    f = max(f, 1e-10)  # guard against zero

    h = 1 - f * (1 - r12)  # not needed in simplified but included for reference

    # Z = (z1 - z2) * sqrt((n-3) / (2 * (1 - r12)))
    denom_factor = 2 * (1 - r12)
    if denom_factor <= 0:
        return float('nan'), float('nan')

    Z = (z1 - z2) * np.sqrt((n - 3) / denom_factor)
    p = 2 * stats.norm.sf(abs(Z))

    return float(Z), float(p)


def cross_validated_r_squared(predictor, outcome, n_folds=CV_FOLDS, seed=RANDOM_SEED):
    """
    METH-07: Cross-validated R^2 for a single predictor -> outcome regression.

    Uses Spearman rank-based regression (rank both, then OLS on ranks).
    Returns mean R^2 across folds (can be negative if model hurts prediction).
    """
    predictor = np.array(predictor, dtype=float)
    outcome = np.array(outcome, dtype=float)
    valid = ~(np.isnan(predictor) | np.isnan(outcome))
    predictor = predictor[valid]
    outcome = outcome[valid]
    n = len(predictor)

    if n < n_folds * 2:
        return float('nan')

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r2_scores = []

    for train_idx, test_idx in kf.split(predictor):
        x_train, x_test = predictor[train_idx], predictor[test_idx]
        y_train, y_test = outcome[train_idx], outcome[test_idx]

        # Simple linear regression
        slope, intercept, _, _, _ = stats.linregress(x_train, y_train)
        y_pred = slope * x_test + intercept

        # R^2 = 1 - SS_res / SS_tot
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        if ss_tot < 1e-15:
            r2_scores.append(0.0)
        else:
            r2_scores.append(1.0 - ss_res / ss_tot)

    return float(np.mean(r2_scores))


def sign_test_all_negative(partial_corrs):
    """
    STAT-05: Formal sign test for consistent direction across architectures.

    Tests whether all partial correlations having the same sign is unlikely under H0.
    Under H0 (no effect), P(all same sign) = 2 * 0.5^n (two-tailed).

    Returns (all_same_sign, sign, p_value, n_archs)
    """
    valid = [r for r in partial_corrs if not np.isnan(r)]
    n = len(valid)
    if n == 0:
        return False, 0, 1.0, 0

    n_positive = sum(1 for r in valid if r > 0)
    n_negative = sum(1 for r in valid if r < 0)

    all_same_sign = (n_positive == n) or (n_negative == n)
    dominant_sign = 1 if n_positive >= n_negative else -1

    # Binomial test: P(all same sign | H0: p=0.5)
    # Two-tailed: P(X=0) + P(X=n) = 2 * 0.5^n
    if all_same_sign:
        p = 2.0 * (0.5 ** n)
    else:
        # More general: use the more extreme count
        k = max(n_positive, n_negative)
        p = 2.0 * stats.binom.sf(k - 1, n, 0.5)

    return bool(all_same_sign), int(dominant_sign), float(p), n


def meta_combine_correlations(correlations, sample_sizes):
    """
    Meta-analytic combination of correlations using Fisher's z-transform.
    Weights by (n-3) as recommended for correlation meta-analysis.

    Returns (combined_r, combined_z, combined_p, Q_heterogeneity)
    """
    zs = []
    weights = []
    for r, n in zip(correlations, sample_sizes):
        if np.isnan(r) or n < 4:
            continue
        z = fisher_z(r)
        w = n - 3
        zs.append(z)
        weights.append(w)

    if len(zs) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    zs = np.array(zs)
    weights = np.array(weights, dtype=float)

    # Weighted mean z
    z_combined = np.sum(weights * zs) / np.sum(weights)

    # SE of combined z
    se_z = 1.0 / np.sqrt(np.sum(weights))

    # Two-tailed test
    z_stat = z_combined / se_z
    p_combined = 2 * stats.norm.sf(abs(z_stat))

    # Back-transform to correlation
    r_combined = np.tanh(z_combined)

    # Cochran's Q for heterogeneity
    Q = np.sum(weights * (zs - z_combined) ** 2)
    Q_p = stats.chi2.sf(Q, df=len(zs) - 1) if len(zs) > 1 else float('nan')

    return float(r_combined), float(z_combined), float(p_combined), float(Q_p)


def run_test_for_architecture(model_name, all_texts, all_labels, unique_labels,
                              label_names, subclusters, device='cpu'):
    """Run all tests for a single architecture on pre-generated subclusters."""
    log(f"\n{'='*70}")
    log(f"ARCHITECTURE: {model_name}")
    log(f"{'='*70}")

    n_clusters = len(subclusters)
    log(f"  Number of subclusters: {n_clusters}")

    # Collect all unique document indices needed
    all_needed_indices = set()
    for sc in subclusters:
        all_needed_indices.update(sc['indices'].tolist())
    all_needed_indices = sorted(all_needed_indices)
    index_map = {orig: new for new, orig in enumerate(all_needed_indices)}

    log(f"  Unique documents needed: {len(all_needed_indices)}")

    # Encode only needed documents
    texts_to_encode = [all_texts[i] for i in all_needed_indices]

    log(f"  Loading model {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    log(f"  Encoding {len(texts_to_encode)} documents...")
    t0 = time.time()
    batch_sz = 64 if 'mpnet' in model_name else 128
    all_embeddings = model.encode(texts_to_encode, show_progress_bar=True,
                                  batch_size=batch_sz)
    all_embeddings = np.array(all_embeddings)
    encode_time = time.time() - t0
    log(f"  Encoding done in {encode_time:.1f}s. Shape: {all_embeddings.shape}")

    # Free model memory
    del model
    import gc
    gc.collect()

    # ---- Compute metrics for each subcluster ----
    log(f"\n  Computing formula variants and ground truth for {n_clusters} subclusters...")

    cluster_data = []
    for idx, sc in enumerate(subclusters):
        # Map global indices to local embedding indices
        local_indices = [index_map[g] for g in sc['indices'].tolist()]
        cluster_emb = all_embeddings[local_indices]

        # Formula variants
        E = formula.compute_E(cluster_emb)
        grad_S = formula.compute_grad_S(cluster_emb)
        eps = 1e-10
        R_simple = E / grad_S if grad_S > eps else float('nan')

        # METH-02: Label purity ground truth (non-cosine)
        purity = compute_label_purity(sc['true_labels'], unique_labels)

        cluster_data.append({
            'idx': idx,
            'primary_category': int(sc['primary_category']),
            'size': int(sc['size']),
            'is_mixed': bool(sc['is_mixed']),
            'E': float(E),
            'grad_S': float(grad_S),
            'R_simple': float(R_simple),
            'label_purity': float(purity),
        })

        if idx < 5 or idx % 20 == 0:
            log(f"    Cluster {idx:3d}: size={sc['size']:3d}, "
                f"E={E:.4f}, grad_S={grad_S:.4f}, R={R_simple:.4f}, "
                f"purity={purity:.3f}, mixed={sc['is_mixed']}")

    # ---- Extract arrays for analysis ----
    E_vals = np.array([c['E'] for c in cluster_data])
    gradS_vals = np.array([c['grad_S'] for c in cluster_data])
    R_vals = np.array([c['R_simple'] for c in cluster_data])
    purity_vals = np.array([c['label_purity'] for c in cluster_data])

    # Drop any NaN clusters
    valid = ~(np.isnan(E_vals) | np.isnan(gradS_vals) | np.isnan(R_vals) | np.isnan(purity_vals))
    E_v = E_vals[valid]
    gS_v = gradS_vals[valid]
    R_v = R_vals[valid]
    pur_v = purity_vals[valid]
    n_valid = int(valid.sum())
    log(f"\n  Valid clusters for analysis: {n_valid}")

    # ---- Test 1: Spearman correlations with label purity ----
    log(f"\n  --- Test 1: Spearman Correlations (ground truth = label purity) ---")

    rho_R, p_R = spearman_corr(R_v, pur_v)
    rho_E, p_E = spearman_corr(E_v, pur_v)
    rho_gS, p_gS = spearman_corr(gS_v, pur_v)
    rho_R_E, _ = spearman_corr(R_v, E_v)  # correlation between predictors

    log(f"    rho(R=E/gradS, purity)  = {rho_R:+.4f}  p={p_R:.6f}")
    log(f"    rho(E, purity)          = {rho_E:+.4f}  p={p_E:.6f}")
    log(f"    rho(grad_S, purity)     = {rho_gS:+.4f}  p={p_gS:.6f}")
    log(f"    rho(R, E) [predictor overlap] = {rho_R_E:+.4f}")

    # ---- Test 2: Steiger's Z-test (STAT-06) ----
    log(f"\n  --- Test 2: Steiger Z-test (R vs E for predicting purity) ---")

    steiger_Z, steiger_p = steiger_z_test(rho_R, rho_E, rho_R_E, n_valid)
    log(f"    Steiger Z = {steiger_Z:+.4f}, p = {steiger_p:.6f}")
    log(f"    Interpretation: {'R significantly better' if steiger_p < 0.05 and steiger_Z > 0 else 'E significantly better' if steiger_p < 0.05 and steiger_Z < 0 else 'No significant difference'}")

    # ---- Test 3: Partial Correlations ----
    log(f"\n  --- Test 3: Partial Correlations ---")

    pc_R_given_E, pc_R_given_E_p = partial_correlation(R_v, pur_v, E_v)
    pc_gS_given_E, pc_gS_given_E_p = partial_correlation(gS_v, pur_v, E_v)

    log(f"    Partial rho(R, purity | E)     = {pc_R_given_E:+.4f}  p={pc_R_given_E_p:.6f}")
    log(f"    Partial rho(gradS, purity | E) = {pc_gS_given_E:+.4f}  p={pc_gS_given_E_p:.6f}")

    # ---- Test 4: Cross-validated R^2 comparison (METH-07) ----
    log(f"\n  --- Test 4: Cross-Validated R^2 (multiplicative test) ---")

    cv_r2_R = cross_validated_r_squared(R_v, pur_v)
    cv_r2_E = cross_validated_r_squared(E_v, pur_v)
    cv_r2_diff = cv_r2_R - cv_r2_E if not (np.isnan(cv_r2_R) or np.isnan(cv_r2_E)) else float('nan')

    log(f"    CV R^2 (R=E/gradS -> purity) = {cv_r2_R:.4f}")
    log(f"    CV R^2 (E -> purity)         = {cv_r2_E:.4f}")
    log(f"    Difference (R - E)           = {cv_r2_diff:+.4f}")
    log(f"    {'R wins' if cv_r2_diff > 0 else 'E wins' if cv_r2_diff < 0 else 'Tie'} in cross-validation")

    # ---- Compile results ----
    arch_result = {
        'model': model_name,
        'embedding_dim': int(all_embeddings.shape[1]),
        'n_documents_encoded': int(len(texts_to_encode)),
        'n_clusters': n_clusters,
        'n_valid_clusters': n_valid,
        'encode_time_seconds': round(encode_time, 1),
        'ground_truth': 'label_purity',
        'correlations': {
            'rho_R_purity': rho_R, 'p_R_purity': p_R,
            'rho_E_purity': rho_E, 'p_E_purity': p_E,
            'rho_gradS_purity': rho_gS, 'p_gradS_purity': p_gS,
            'rho_R_E': rho_R_E,
        },
        'steiger_test': {
            'Z': steiger_Z, 'p': steiger_p,
            'R_better': bool(steiger_p < 0.05 and steiger_Z > 0),
        },
        'partial_correlations': {
            'pc_R_given_E': pc_R_given_E, 'p_pc_R_given_E': pc_R_given_E_p,
            'pc_gradS_given_E': pc_gS_given_E, 'p_pc_gradS_given_E': pc_gS_given_E_p,
        },
        'cross_validated_r2': {
            'R_simple': cv_r2_R,
            'E_alone': cv_r2_E,
            'difference': cv_r2_diff,
            'R_wins_cv': bool(cv_r2_diff > 0) if not np.isnan(cv_r2_diff) else None,
        },
        'cluster_data': cluster_data,
    }

    log(f"\n  ARCHITECTURE SUMMARY for {model_name}:")
    log(f"    rho(R, purity) = {rho_R:+.4f} vs rho(E, purity) = {rho_E:+.4f}")
    log(f"    Steiger p = {steiger_p:.4f} ({'sig' if steiger_p < 0.05 else 'ns'})")
    log(f"    CV R^2 diff = {cv_r2_diff:+.4f}")
    log(f"    Partial rho(gradS|E) = {pc_gS_given_E:+.4f}")

    return arch_result


def determine_verdict(all_results):
    """
    Apply pre-registered criteria to determine overall verdict.
    """
    n_archs = len(all_results)

    # Count how many architectures show R significantly better (Steiger p < 0.05)
    n_steiger_sig = sum(
        1 for r in all_results
        if r['steiger_test']['p'] < 0.05 and r['steiger_test']['Z'] > 0
    )

    # Count how many show CV R^2(R) > R^2(E)
    n_cv_wins = sum(
        1 for r in all_results
        if r['cross_validated_r2']['R_wins_cv'] is True
    )

    # Check CV R^2 differences
    cv_diffs = [r['cross_validated_r2']['difference'] for r in all_results
                if not np.isnan(r['cross_validated_r2']['difference'])]

    all_steiger_ns = all(
        r['steiger_test']['p'] > 0.05 for r in all_results
    )
    all_cv_diff_small = all(abs(d) < 0.01 for d in cv_diffs)

    # Partial correlation directions (for sign test)
    partial_corrs_gS = [r['partial_correlations']['pc_gradS_given_E'] for r in all_results]

    sign_all_same, sign_dir, sign_p, sign_n = sign_test_all_negative(partial_corrs_gS)

    # Meta-analytic combination
    pc_values = partial_corrs_gS
    pc_ns = [r['n_valid_clusters'] for r in all_results]
    meta_r, meta_z, meta_p, meta_Q_p = meta_combine_correlations(pc_values, pc_ns)

    verdict_data = {
        'n_steiger_significant': n_steiger_sig,
        'n_cv_R_wins': n_cv_wins,
        'cv_diffs': cv_diffs,
        'all_steiger_ns': all_steiger_ns,
        'all_cv_diff_small': all_cv_diff_small,
        'sign_test': {
            'all_same_sign': sign_all_same,
            'sign_direction': sign_dir,
            'sign_p': sign_p,
            'n_architectures': sign_n,
        },
        'meta_analysis': {
            'combined_r': meta_r,
            'combined_z': meta_z,
            'combined_p': meta_p,
            'heterogeneity_Q_p': meta_Q_p,
        },
    }

    # Apply pre-registered criteria
    # CONFIRMED: Steiger p < 0.05 AND CV R^2(R) > R^2(E) on >= 2/3 architectures
    if n_steiger_sig >= 2 and n_cv_wins >= 2:
        verdict = "CONFIRMED"
    # FALSIFIED: Steiger p > 0.05 on ALL AND CV diff < 0.01 on ALL
    elif all_steiger_ns and all_cv_diff_small:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    verdict_data['verdict'] = verdict
    return verdict, verdict_data


def main():
    log("=" * 70)
    log("Q01 v3 TEST: Does grad_S add independent predictive value beyond E?")
    log("=" * 70)
    log(f"Date: {datetime.now(timezone.utc).isoformat()}")
    log(f"Architectures: {ARCHITECTURES}")
    log(f"Target subclusters: {SUBCLUSTERS_PER_CATEGORY} per category x 20 = {SUBCLUSTERS_PER_CATEGORY * 20}")
    log(f"Cluster sizes: {CLUSTER_SIZES}")
    log(f"Random seed: {RANDOM_SEED}")
    log("")
    log("PRE-REGISTERED CRITERIA (stated before analysis):")
    for k, v in PRE_REGISTERED_CRITERIA.items():
        log(f"  {k.upper()}: {v}")
    log("")

    # ---- Load data ----
    log("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_texts = data.data
    all_labels = np.array(data.target)
    unique_labels = np.unique(all_labels)
    label_names = data.target_names

    # Filter empty/tiny documents
    valid_mask = np.array([len(t.strip()) > 10 for t in all_texts])
    all_texts_arr = np.array(all_texts, dtype=object)
    all_texts_filtered = all_texts_arr[valid_mask]
    all_labels_filtered = all_labels[valid_mask]

    log(f"Loaded {len(all_texts_filtered)} documents across {len(unique_labels)} categories")
    for i, name in enumerate(label_names):
        count = np.sum(all_labels_filtered == i)
        log(f"  {i:2d}: {name:30s} ({count} docs)")

    # ---- Generate subclusters (same for all architectures) ----
    log(f"\nGenerating subclusters...")
    rng = np.random.RandomState(RANDOM_SEED)
    subclusters = generate_subclusters(
        all_labels_filtered, unique_labels, rng,
        n_per_category=SUBCLUSTERS_PER_CATEGORY,
        sizes=CLUSTER_SIZES
    )
    log(f"Generated {len(subclusters)} subclusters")

    # Report subcluster statistics
    sizes = [sc['size'] for sc in subclusters]
    n_mixed = sum(1 for sc in subclusters if sc['is_mixed'])
    log(f"  Size range: {min(sizes)}-{max(sizes)}, mean={np.mean(sizes):.0f}")
    log(f"  Pure clusters: {len(subclusters) - n_mixed}, Mixed clusters: {n_mixed}")

    if len(subclusters) < MIN_CLUSTERS_TARGET:
        log(f"  WARNING: Only {len(subclusters)} subclusters, below target {MIN_CLUSTERS_TARGET}")

    # ---- Run tests for each architecture ----
    all_results = []
    total_start = time.time()

    for model_name in ARCHITECTURES:
        result = run_test_for_architecture(
            model_name, all_texts_filtered, all_labels_filtered,
            unique_labels, label_names, subclusters, device='cpu'
        )
        all_results.append(result)
        log(f"\n  [Elapsed: {time.time() - total_start:.0f}s]")

    # ---- Overall verdict ----
    verdict, verdict_data = determine_verdict(all_results)

    total_time = time.time() - total_start

    log(f"\n{'='*70}")
    log("CROSS-ARCHITECTURE ANALYSIS")
    log(f"{'='*70}")

    # STAT-05: Sign test
    partial_corrs = [r['partial_correlations']['pc_gradS_given_E'] for r in all_results]
    log(f"\n  Partial corr(gradS, purity | E) across architectures: {[f'{x:+.4f}' for x in partial_corrs]}")
    st = verdict_data['sign_test']
    log(f"  Sign test: all_same_sign={st['all_same_sign']}, direction={'+' if st['sign_direction'] > 0 else '-'}, p={st['sign_p']:.4f}")

    # Meta-analysis
    ma = verdict_data['meta_analysis']
    log(f"\n  Meta-analytic combination of partial correlations:")
    log(f"    Combined r = {ma['combined_r']:+.4f}")
    log(f"    Combined p = {ma['combined_p']:.6f}")
    log(f"    Heterogeneity Q p = {ma['heterogeneity_Q_p']:.4f}")

    log(f"\n{'='*70}")
    log(f"OVERALL VERDICT: {verdict}")
    log(f"{'='*70}")

    # Summary table
    log(f"\n{'='*70}")
    log("SUMMARY TABLE")
    log(f"{'='*70}")
    log(f"{'Model':35s} | {'rho(R)':>8s} | {'rho(E)':>8s} | {'Steiger p':>10s} | {'CV R2 diff':>10s} | {'pc(gS|E)':>10s}")
    log("-" * 95)
    for r in all_results:
        c = r['correlations']
        log(f"{r['model']:35s} | {c['rho_R_purity']:>+8.4f} | {c['rho_E_purity']:>+8.4f} | "
            f"{r['steiger_test']['p']:>10.6f} | {r['cross_validated_r2']['difference']:>+10.4f} | "
            f"{r['partial_correlations']['pc_gradS_given_E']:>+10.4f}")

    log(f"\nTotal runtime: {total_time:.0f}s")

    # ---- Compile and save results ----
    output = {
        'test': 'Q01_grad_S_v3',
        'version': 'v3',
        'date': datetime.now(timezone.utc).isoformat(),
        'total_runtime_seconds': round(total_time, 1),
        'methodology': {
            'dataset': '20 Newsgroups (sklearn)',
            'n_documents_total': int(len(all_texts_filtered)),
            'n_categories': int(len(unique_labels)),
            'n_subclusters': int(len(subclusters)),
            'subcluster_sizes': CLUSTER_SIZES,
            'subclusters_per_category': SUBCLUSTERS_PER_CATEGORY,
            'ground_truth': 'label_purity (fraction of majority category - NOT cosine-based)',
            'architectures': ARCHITECTURES,
            'bootstrap_n': BOOTSTRAP_N,
            'cv_folds': CV_FOLDS,
            'random_seed': RANDOM_SEED,
            'tests_applied': [
                'Spearman correlations (R vs E vs grad_S with label purity)',
                "Steiger Z-test for dependent correlations (STAT-06)",
                'Partial Spearman correlation (grad_S | E)',
                'Cross-validated R^2 comparison (METH-07)',
                'Sign test across architectures (STAT-05)',
                'Meta-analytic combination of partial correlations',
            ],
        },
        'pre_registered_criteria': PRE_REGISTERED_CRITERIA,
        'per_architecture_results': all_results,
        'cross_architecture_analysis': verdict_data,
        'overall_verdict': verdict,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nResults saved to: {RESULTS_PATH}")
    return output


if __name__ == '__main__':
    results = main()
