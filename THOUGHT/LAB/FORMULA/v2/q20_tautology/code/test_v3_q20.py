"""
Q20 v3: Is R = (E/grad_S) * sigma^Df tautological?

Fixes from v2 audit (AUDIT.md):
- BUG-1: Dropped 8e conservation test entirely. Under v2 definitions,
  Df = 2/alpha, so Df*alpha = 2 trivially. The conservation law is
  UNTESTABLE without an independent Df measurement (e.g. box-counting).
- STAT-1: Replaced arbitrary 0.05 rho threshold with Steiger's test
  for dependent correlations (proper statistical comparison).
- STAT-3: Increased to 90 clusters (30 pure, 30 mixed, 30 random).
- METH-2: Added cross-validated prediction comparison (out-of-sample R^2).
- METH-3: Added nested model comparison via AIC for component ablation.

Pre-registered criteria (v3):
  NOT TAUTOLOGICAL if:
    Steiger p < 0.05 (R_full > E) on >= 2/3 architectures
    AND cross-validated R^2(R_full) > R^2(E) on >= 2/3 architectures
  TAUTOLOGICAL if:
    Steiger NS (p >= 0.05) on all 3 architectures
    AND cross-validated |R^2(R_full) - R^2(E)| < 0.01 on all 3
  INCONCLUSIVE otherwise.

Ground truth: cluster purity (fraction of dominant category).
NO synthetic data. NO reward-maxing. Just truth.
"""

import importlib.util
import sys
import os
import json
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")


def P(msg=""):
    """Print with flush."""
    print(msg, flush=True)


# Import formula module
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import hashlib


# ============================================================
# STATISTICAL TOOLS
# ============================================================

def steiger_z_test(rho_xz, rho_yz, rho_xy, n):
    """
    Steiger's Z-test for comparing two dependent correlations.

    Tests H0: rho(X,Z) = rho(Y,Z) where X and Y share the same Z.

    Uses the Steiger (1980) method with Hotelling's (1940) correction.

    Args:
        rho_xz: correlation of metric X with ground truth Z
        rho_yz: correlation of metric Y with ground truth Z
        rho_xy: correlation between metric X and metric Y
        n: sample size

    Returns:
        (z_stat, p_value_one_sided):
            z_stat > 0 means rho_xz > rho_yz
            p_value is one-sided: P(rho_xz <= rho_yz)
    """
    if n < 4:
        return float('nan'), float('nan')

    # Fisher z-transform
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    z_xz = fisher_z(rho_xz)
    z_yz = fisher_z(rho_yz)

    # Hotelling's correction factor for dependent correlations
    r_mean_sq = 0.5 * (rho_xz ** 2 + rho_yz ** 2)
    f = (1 - rho_xy) / (2 * (1 - r_mean_sq))
    f = min(f, 1.0)  # bound the correction

    # Denominator: SE of difference
    denom = np.sqrt(2 * (1 - rho_xy) / ((n - 3) * (1 + r_mean_sq * f)))

    if denom < 1e-15:
        return float('nan'), float('nan')

    z_stat = (z_xz - z_yz) / denom
    # One-sided p: probability that rho_xz <= rho_yz
    p_value = 1.0 - norm.cdf(z_stat)

    return float(z_stat), float(p_value)


def cross_validated_r2(metric_values, ground_truth, n_folds=5, seed=42):
    """
    K-fold cross-validated R^2 for a linear model predicting ground_truth
    from metric_values.

    Returns:
        (mean_r2, std_r2, fold_r2s): mean out-of-sample R^2 across folds,
        standard deviation, and per-fold values.
    """
    rng = np.random.RandomState(seed)
    valid = ~(np.isnan(metric_values) | np.isnan(ground_truth))
    X = np.array(metric_values)[valid].reshape(-1, 1)
    y = np.array(ground_truth)[valid]
    n = len(y)

    if n < n_folds * 2:
        return float('nan'), float('nan'), []

    indices = np.arange(n)
    rng.shuffle(indices)
    fold_size = n // n_folds

    fold_r2s = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # R^2 can be negative for poor models
        r2 = r2_score(y_test, y_pred)
        fold_r2s.append(float(r2))

    return float(np.mean(fold_r2s)), float(np.std(fold_r2s)), fold_r2s


def compute_aic(y_true, y_pred, n_params):
    """
    Compute AIC for a regression model.
    AIC = n * ln(RSS/n) + 2*k
    where k = number of parameters (including intercept).
    """
    n = len(y_true)
    residuals = y_true - y_pred
    rss = float(np.sum(residuals ** 2))
    if rss <= 0:
        rss = 1e-15
    aic = n * np.log(rss / n) + 2 * n_params
    return float(aic)


def nested_model_comparison(y, predictors_dict):
    """
    Compare nested models using AIC.

    Args:
        y: ground truth array
        predictors_dict: OrderedDict-like {name: values_array}
            Models are built cumulatively: model_1 uses first predictor,
            model_2 uses first two, etc.

    Returns:
        list of dicts with model name, AIC, delta_AIC, R^2
    """
    valid = ~np.isnan(y)
    for vals in predictors_dict.values():
        valid = valid & ~np.isnan(np.array(vals))

    y_clean = np.array(y)[valid]
    n = len(y_clean)

    if n < 10:
        return []

    results = []
    cumulative_X = []
    predictor_names = list(predictors_dict.keys())

    for i, name in enumerate(predictor_names):
        vals = np.array(predictors_dict[name])[valid]
        cumulative_X.append(vals)

        X = np.column_stack(cumulative_X)
        model = LinearRegression()
        model.fit(X, y_clean)
        y_pred = model.predict(X)

        r2 = r2_score(y_clean, y_pred)
        # +1 for intercept
        aic = compute_aic(y_clean, y_pred, n_params=i + 2)

        results.append({
            'model': ' + '.join(predictor_names[:i+1]),
            'n_predictors': i + 1,
            'r2': float(r2),
            'aic': float(aic),
        })

    # Compute delta_AIC relative to best model
    best_aic = min(r['aic'] for r in results)
    for r in results:
        r['delta_aic'] = r['aic'] - best_aic

    return results


def bootstrap_rho_difference(values_a, values_b, ground_truth,
                             n_bootstrap=10000, seed=42):
    """
    Bootstrap test: is |Spearman rho| of values_a with ground_truth
    significantly greater than that of values_b?
    Returns (mean_diff, p_value, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)

    valid = ~(np.isnan(values_a) | np.isnan(values_b) | np.isnan(ground_truth))
    va = np.array(values_a)[valid]
    vb = np.array(values_b)[valid]
    gt = np.array(ground_truth)[valid]
    n = len(gt)

    if n < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rho_a, _ = spearmanr(va[idx], gt[idx])
        rho_b, _ = spearmanr(vb[idx], gt[idx])
        diffs[i] = abs(rho_a) - abs(rho_b)

    mean_diff = float(np.mean(diffs))
    p_value = float(np.mean(diffs <= 0))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return mean_diff, p_value, ci_lower, ci_upper


# ============================================================
# CLUSTER CREATION
# ============================================================

def compute_purity(labels):
    """Cluster purity: fraction of docs from dominant category."""
    if len(labels) == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts.values()) / len(labels)


def create_clusters(data, rng, n_pure=30, n_mixed=30, n_random=30,
                    docs_per_cluster=200):
    """
    Create 90 clusters from 20 Newsgroups data with continuous purity.

    - 30 pure: single category (purity ~ 1.0)
    - 30 mixed: varying mixtures of 2-5 categories (purity ~ 0.3-0.7)
    - 30 random: random draws from all categories (purity ~ 0.05-0.15)

    The mixed clusters use varying ratios to get continuous purity
    variation rather than the binary pure/50-50 of v2.
    """
    targets = np.array(data.target)
    categories = list(range(20))

    cat_indices = {}
    for cat in categories:
        cat_indices[cat] = np.where(targets == cat)[0]

    clusters = []

    # -- Pure clusters (30): each from one category --
    # Use 20 categories + 10 repeats with different samples
    pure_cats = list(range(20)) + list(rng.choice(20, size=10, replace=True))
    rng.shuffle(pure_cats)
    for cat in pure_cats[:n_pure]:
        available = cat_indices[cat]
        if len(available) >= docs_per_cluster:
            chosen = rng.choice(available, size=docs_per_cluster, replace=False)
        else:
            chosen = rng.choice(available, size=docs_per_cluster, replace=True)
        labels = targets[chosen]
        clusters.append((chosen, labels, 'pure'))

    # -- Mixed clusters (30): varying dominant fractions --
    # Generate dominant fractions from 0.30 to 0.80 (continuous purity)
    dominant_fractions = np.linspace(0.30, 0.80, n_mixed)
    rng.shuffle(dominant_fractions)

    all_pairs = [(i, j) for i in range(20) for j in range(i+1, 20)]
    rng.shuffle(all_pairs)

    for mi in range(n_mixed):
        frac = dominant_fractions[mi]
        n_dominant = int(frac * docs_per_cluster)
        n_other = docs_per_cluster - n_dominant

        pair_idx = mi % len(all_pairs)
        cat_a, cat_b = all_pairs[pair_idx]

        avail_a = cat_indices[cat_a]
        avail_b = cat_indices[cat_b]

        if len(avail_a) >= n_dominant:
            chosen_a = rng.choice(avail_a, size=n_dominant, replace=False)
        else:
            chosen_a = rng.choice(avail_a, size=n_dominant, replace=True)

        if len(avail_b) >= n_other:
            chosen_b = rng.choice(avail_b, size=n_other, replace=False)
        else:
            chosen_b = rng.choice(avail_b, size=n_other, replace=True)

        chosen = np.concatenate([chosen_a, chosen_b])
        labels = targets[chosen]
        clusters.append((chosen, labels, 'mixed'))

    # -- Random clusters (30): from all categories --
    all_indices = np.arange(len(targets))
    for _ in range(n_random):
        chosen = rng.choice(all_indices, size=docs_per_cluster, replace=False)
        labels = targets[chosen]
        clusters.append((chosen, labels, 'random'))

    return clusters


def compute_all_metrics(embeddings):
    """Compute all formula components for a cluster."""
    result = formula.compute_all(embeddings)
    E = result['E']
    grad_S = result['grad_S']
    sigma = result['sigma']
    Df = result['Df']

    result['inv_grad_S'] = 1.0 / grad_S if grad_S > 1e-10 else float('nan')

    if not np.isnan(sigma) and not np.isnan(Df):
        result['sigma_Df'] = sigma ** Df
    else:
        result['sigma_Df'] = float('nan')

    # Also compute E / std (naive SNR) for direct comparison
    result['E_over_std'] = E / grad_S if grad_S > 1e-10 else float('nan')

    return result


def compute_ablation_from_components(E, grad_S, sigma, Df):
    """Compute all 5 ablation forms from pre-computed components."""
    forms = {}

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]) and grad_S > 1e-10:
        forms['R_full'] = (E / grad_S) * (sigma ** Df)
    else:
        forms['R_full'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S]) and grad_S > 1e-10:
        forms['R_simple'] = E / grad_S
    else:
        forms['R_simple'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        forms['R_sub'] = (E - grad_S) * (sigma ** Df)
    else:
        forms['R_sub'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        forms['R_exp'] = E * np.exp(-grad_S) * (sigma ** Df)
    else:
        forms['R_exp'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S]):
        forms['R_log'] = np.log(max(E, 0.01)) / (grad_S + 1)
    else:
        forms['R_log'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        denom = grad_S + sigma ** Df
        forms['R_add'] = E / denom if denom > 1e-10 else float('nan')
    else:
        forms['R_add'] = float('nan')

    return forms


# ============================================================
# MAIN TEST
# ============================================================

def run_all_tests():
    """Run all Q20 v3 tests."""
    results = {
        'metadata': {
            'question': 'Q20',
            'version': 'v3',
            'timestamp': datetime.now().isoformat(),
            'methodology': (
                '20 Newsgroups, 3 architectures, 90 clusters '
                '(30 pure + 30 mixed + 30 random), purity ground truth, '
                'Steiger test + cross-validated R^2 + nested AIC'
            ),
            'seed': 42,
            'pre_registered_criteria': {
                'NOT_TAUTOLOGICAL': (
                    'Steiger p < 0.05 (R_full > E) on >= 2/3 architectures '
                    'AND cross-validated R^2(R_full) > R^2(E) on >= 2/3 architectures'
                ),
                'TAUTOLOGICAL': (
                    'Steiger p >= 0.05 on all 3 architectures '
                    'AND |CV_R^2(R_full) - CV_R^2(E)| < 0.01 on all 3'
                ),
                'INCONCLUSIVE': 'Otherwise',
            },
            'audit_fixes': [
                'BUG-1: Dropped 8e conservation (Df*alpha=2 trivially under v2 defs)',
                'STAT-1: Replaced 0.05 rho threshold with Steiger Z-test',
                'STAT-3: Increased to 90 clusters',
                'METH-2: Added cross-validated prediction comparison',
                'METH-3: Added nested model comparison via AIC',
                'METH-4: Mixed clusters now have continuous purity variation',
            ],
        },
        'test1_steiger_comparison': {},
        'test2_cross_validated_prediction': {},
        'test3_nested_model_comparison': {},
        'test4_ablation': {},
        'test5_novel_predictions': {},
        'verdict': {}
    }

    rng = np.random.RandomState(42)

    # ---- Load 20 Newsgroups ----
    P("=" * 70)
    P("Q20 v3: Is R = (E/grad_S) * sigma^Df tautological?")
    P("=" * 70)
    P()
    P("AUDIT FIXES APPLIED:")
    for fix in results['metadata']['audit_fixes']:
        P(f"  - {fix}")
    P()
    P("PRE-REGISTERED CRITERIA:")
    for k, v in results['metadata']['pre_registered_criteria'].items():
        P(f"  {k}: {v}")
    P()

    P("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(
        subset='all', remove=('headers', 'footers', 'quotes')
    )
    P(f"  Loaded {len(data.data)} documents across "
      f"{len(data.target_names)} categories")

    # Data fingerprint
    text_hash = hashlib.sha256(
        '|'.join(data.data[:100]).encode('utf-8', errors='replace')
    ).hexdigest()[:16]
    results['metadata']['data_hash'] = text_hash
    results['metadata']['n_docs'] = len(data.data)
    results['metadata']['n_categories'] = len(data.target_names)

    # ---- Create clusters ----
    P("Creating 90 clusters (30 pure, 30 mixed, 30 random)...")
    clusters = create_clusters(data, rng, n_pure=30, n_mixed=30, n_random=30)
    purities = [compute_purity(labels) for _, labels, _ in clusters]
    cluster_types = [ctype for _, _, ctype in clusters]
    P(f"  Total clusters: {len(clusters)}")
    P(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
    P(f"  Pure mean: "
      f"{np.mean([p for p, t in zip(purities, cluster_types) if t == 'pure']):.3f}")
    P(f"  Mixed mean: "
      f"{np.mean([p for p, t in zip(purities, cluster_types) if t == 'mixed']):.3f}")
    P(f"  Random mean: "
      f"{np.mean([p for p, t in zip(purities, cluster_types) if t == 'random']):.3f}")

    # Collect unique doc indices
    all_doc_set = set()
    for indices, _, _ in clusters:
        all_doc_set.update(indices.tolist())
    all_doc_list = sorted(all_doc_set)
    all_texts = [data.data[i] for i in all_doc_list]
    P(f"  Unique documents to encode: {len(all_doc_list)}")

    results['metadata']['n_clusters'] = len(clusters)
    results['metadata']['purity_range'] = [float(min(purities)),
                                            float(max(purities))]

    # ---- Architectures ----
    model_names = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1',
    ]

    for model_name in model_names:
        P()
        P("=" * 70)
        P(f"Architecture: {model_name}")
        P("=" * 70)

        t_model_start = time.time()
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        P(f"  Embedding dimension: {dim}")

        encode_batch = 64 if dim > 512 else 128

        # Encode all unique documents
        P(f"  Encoding {len(all_texts)} unique documents "
          f"(batch={encode_batch})...")
        chunk_size = 2000
        all_emb_parts = []
        for chunk_start in range(0, len(all_texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_texts))
            chunk_texts = all_texts[chunk_start:chunk_end]
            chunk_embs = model.encode(
                chunk_texts, show_progress_bar=False, batch_size=encode_batch
            )
            all_emb_parts.append(chunk_embs)
            P(f"    Encoded {chunk_end}/{len(all_texts)} docs...")
        all_embeddings = np.vstack(all_emb_parts)
        del all_emb_parts
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_doc_list)}
        P(f"  Encoding done. Shape: {all_embeddings.shape}")

        # ============================================================
        # Compute metrics for all clusters
        # ============================================================
        P("  Computing metrics for all 90 clusters...")
        metrics_per_cluster = []
        for ci, (indices, labels, ctype) in enumerate(clusters):
            positions = [idx_to_pos[idx] for idx in indices]
            embs = all_embeddings[positions]
            m = compute_all_metrics(embs)
            m['purity'] = compute_purity(labels)
            m['type'] = ctype
            metrics_per_cluster.append(m)
            if (ci + 1) % 30 == 0:
                P(f"    Computed {ci+1}/{len(clusters)} clusters...")

        purity_arr = np.array([m['purity'] for m in metrics_per_cluster])

        # ============================================================
        # TEST 1: Steiger's Test (R_full vs E)
        # ============================================================
        P()
        P("  " + "-" * 60)
        P("  TEST 1: Steiger's Z-test (R_full vs E, sharing ground truth)")
        P("  " + "-" * 60)

        # Get correlation values
        r_full_vals = np.array([m['R_full'] for m in metrics_per_cluster])
        e_vals = np.array([m['E'] for m in metrics_per_cluster])
        r_simple_vals = np.array([m['R_simple'] for m in metrics_per_cluster])

        # Compute all Spearman correlations
        component_names = ['E', 'grad_S', 'sigma', 'Df', 'inv_grad_S',
                           'sigma_Df', 'R_simple', 'R_full', 'E_over_std']
        component_rhos = {}
        for comp in component_names:
            vals = np.array([m[comp] for m in metrics_per_cluster])
            valid = ~np.isnan(vals)
            if np.sum(valid) >= 10:
                rho, pval = spearmanr(vals[valid], purity_arr[valid])
                component_rhos[comp] = {
                    'rho': float(rho),
                    'abs_rho': float(abs(rho)),
                    'p_value': float(pval),
                    'n_valid': int(np.sum(valid))
                }
            else:
                component_rhos[comp] = {
                    'rho': float('nan'), 'abs_rho': float('nan'),
                    'p_value': float('nan'), 'n_valid': int(np.sum(valid))
                }

        P(f"    {'Component':<15} {'rho':>8} {'|rho|':>8} {'p-value':>12}")
        P(f"    {'-'*48}")
        for comp in component_names:
            r = component_rhos[comp]
            P(f"    {comp:<15} {r['rho']:>8.4f} {r['abs_rho']:>8.4f} "
              f"{r['p_value']:>12.6f}")

        # Steiger's test: R_full vs E
        valid_both = ~(np.isnan(r_full_vals) | np.isnan(e_vals) |
                       np.isnan(purity_arr))
        rf_clean = r_full_vals[valid_both]
        e_clean = e_vals[valid_both]
        pur_clean = purity_arr[valid_both]
        n_valid = len(pur_clean)

        rho_rfull_pur, _ = spearmanr(rf_clean, pur_clean)
        rho_e_pur, _ = spearmanr(e_clean, pur_clean)
        rho_rfull_e, _ = spearmanr(rf_clean, e_clean)

        steiger_z, steiger_p = steiger_z_test(
            abs(rho_rfull_pur), abs(rho_e_pur), abs(rho_rfull_e), n_valid
        )

        P(f"\n    Steiger's Z-test (R_full vs E on purity):")
        P(f"      rho(R_full, purity)  = {rho_rfull_pur:.4f}")
        P(f"      rho(E, purity)       = {rho_e_pur:.4f}")
        P(f"      rho(R_full, E)       = {rho_rfull_e:.4f}")
        P(f"      n                    = {n_valid}")
        P(f"      Z-statistic          = {steiger_z:.4f}")
        P(f"      p-value (one-sided)  = {steiger_p:.6f}")
        P(f"      Significant (p<0.05) = {steiger_p < 0.05}")

        # Also Steiger: R_simple vs E
        rs_clean = np.array([m['R_simple'] for m in metrics_per_cluster])[valid_both]
        rho_rsimple_pur, _ = spearmanr(rs_clean, pur_clean)
        rho_rsimple_e, _ = spearmanr(rs_clean, e_clean)

        steiger_z_simple, steiger_p_simple = steiger_z_test(
            abs(rho_rsimple_pur), abs(rho_e_pur), abs(rho_rsimple_e), n_valid
        )

        P(f"\n    Steiger's Z-test (R_simple vs E on purity):")
        P(f"      rho(R_simple, purity) = {rho_rsimple_pur:.4f}")
        P(f"      Z-statistic           = {steiger_z_simple:.4f}")
        P(f"      p-value (one-sided)   = {steiger_p_simple:.6f}")
        P(f"      Significant (p<0.05)  = {steiger_p_simple < 0.05}")

        # Also Steiger: R_full vs R_simple (does sigma^Df add value?)
        rho_rfull_rsimple, _ = spearmanr(rf_clean, rs_clean)
        steiger_z_fullvssimple, steiger_p_fullvssimple = steiger_z_test(
            abs(rho_rfull_pur), abs(rho_rsimple_pur),
            abs(rho_rfull_rsimple), n_valid
        )

        P(f"\n    Steiger's Z-test (R_full vs R_simple on purity):")
        P(f"      rho(R_full, purity)   = {rho_rfull_pur:.4f}")
        P(f"      rho(R_simple, purity) = {rho_rsimple_pur:.4f}")
        P(f"      Z-statistic           = {steiger_z_fullvssimple:.4f}")
        P(f"      p-value (one-sided)   = {steiger_p_fullvssimple:.6f}")
        P(f"      Significant (p<0.05)  = {steiger_p_fullvssimple < 0.05}")

        # Bootstrap confirmation
        P("\n    Bootstrap confirmation (R_full vs E, 10000 resamples)...")
        boot_diff, boot_p, boot_ci_lo, boot_ci_hi = bootstrap_rho_difference(
            r_full_vals, e_vals, purity_arr, n_bootstrap=10000, seed=42
        )
        P(f"      Mean |rho| diff:  {boot_diff:.4f}")
        P(f"      Bootstrap p:      {boot_p:.4f}")
        P(f"      95% CI:           [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")

        steiger_rfull_wins = steiger_p < 0.05

        results['test1_steiger_comparison'][model_name] = {
            'component_rhos': component_rhos,
            'steiger_rfull_vs_e': {
                'rho_rfull_purity': float(rho_rfull_pur),
                'rho_e_purity': float(rho_e_pur),
                'rho_rfull_e': float(rho_rfull_e),
                'n': n_valid,
                'z_stat': float(steiger_z),
                'p_value': float(steiger_p),
                'significant': steiger_rfull_wins,
            },
            'steiger_rsimple_vs_e': {
                'rho_rsimple_purity': float(rho_rsimple_pur),
                'z_stat': float(steiger_z_simple),
                'p_value': float(steiger_p_simple),
                'significant': steiger_p_simple < 0.05,
            },
            'steiger_rfull_vs_rsimple': {
                'z_stat': float(steiger_z_fullvssimple),
                'p_value': float(steiger_p_fullvssimple),
                'significant': steiger_p_fullvssimple < 0.05,
            },
            'bootstrap_rfull_vs_e': {
                'mean_diff': boot_diff,
                'p_value': boot_p,
                'ci_lower': boot_ci_lo,
                'ci_upper': boot_ci_hi,
            },
        }

        # ============================================================
        # TEST 2: Cross-Validated Prediction
        # ============================================================
        P()
        P("  " + "-" * 60)
        P("  TEST 2: Cross-validated prediction (5-fold, out-of-sample R^2)")
        P("  " + "-" * 60)

        cv_metrics = {
            'E': e_vals,
            'R_simple': r_simple_vals,
            'R_full': r_full_vals,
        }

        cv_results = {}
        for metric_name, metric_vals in cv_metrics.items():
            mean_r2, std_r2, fold_r2s = cross_validated_r2(
                metric_vals, purity_arr, n_folds=5, seed=42
            )
            cv_results[metric_name] = {
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'fold_r2s': fold_r2s,
            }
            P(f"    {metric_name:<12}: CV R^2 = {mean_r2:.4f} "
              f"(+/- {std_r2:.4f})")

        cv_rfull_beats_e = (
            cv_results['R_full']['mean_r2'] > cv_results['E']['mean_r2']
        )
        cv_diff = (cv_results['R_full']['mean_r2'] -
                   cv_results['E']['mean_r2'])
        cv_diff_small = abs(cv_diff) < 0.01

        P(f"\n    R_full CV R^2 > E CV R^2: {cv_rfull_beats_e}")
        P(f"    Difference: {cv_diff:.4f}")
        P(f"    |Difference| < 0.01 (negligible): {cv_diff_small}")

        results['test2_cross_validated_prediction'][model_name] = {
            'cv_results': cv_results,
            'rfull_beats_e': cv_rfull_beats_e,
            'cv_r2_difference': float(cv_diff),
            'difference_negligible': cv_diff_small,
        }

        # ============================================================
        # TEST 3: Nested Model Comparison (AIC)
        # ============================================================
        P()
        P("  " + "-" * 60)
        P("  TEST 3: Nested model comparison (AIC)")
        P("  " + "-" * 60)
        P("  Models: E -> E + 1/grad_S -> E + 1/grad_S + sigma^Df")

        inv_gs_vals = np.array([m['inv_grad_S'] for m in metrics_per_cluster])
        sigma_df_vals = np.array([m['sigma_Df'] for m in metrics_per_cluster])

        # Use ordered dict-like structure
        from collections import OrderedDict
        predictors = OrderedDict([
            ('E', e_vals),
            ('1/grad_S', inv_gs_vals),
            ('sigma^Df', sigma_df_vals),
        ])

        nested_results = nested_model_comparison(purity_arr, predictors)

        P(f"\n    {'Model':<35} {'R^2':>8} {'AIC':>10} {'dAIC':>8}")
        P(f"    {'-'*63}")
        for nr in nested_results:
            P(f"    {nr['model']:<35} {nr['r2']:>8.4f} "
              f"{nr['aic']:>10.2f} {nr['delta_aic']:>8.2f}")

        # Check if adding components improves AIC
        if len(nested_results) >= 3:
            aic_e_only = nested_results[0]['aic']
            aic_e_gs = nested_results[1]['aic']
            aic_full = nested_results[2]['aic']
            grad_s_improves = aic_e_gs < aic_e_only - 2  # dAIC > 2 is meaningful
            sigma_df_improves = aic_full < aic_e_gs - 2
            P(f"\n    Adding 1/grad_S improves AIC by > 2: {grad_s_improves} "
              f"(delta = {aic_e_only - aic_e_gs:.2f})")
            P(f"    Adding sigma^Df improves AIC by > 2: {sigma_df_improves} "
              f"(delta = {aic_e_gs - aic_full:.2f})")
        else:
            grad_s_improves = False
            sigma_df_improves = False

        results['test3_nested_model_comparison'][model_name] = {
            'models': nested_results,
            'grad_s_improves_aic': grad_s_improves,
            'sigma_df_improves_aic': sigma_df_improves,
        }

        # ============================================================
        # TEST 4: Ablation (6 functional forms)
        # ============================================================
        P()
        P("  " + "-" * 60)
        P("  TEST 4: Ablation (6 functional forms)")
        P("  " + "-" * 60)

        ablation_per_cluster = []
        for m in metrics_per_cluster:
            forms = compute_ablation_from_components(
                m['E'], m['grad_S'], m['sigma'], m['Df']
            )
            forms['purity'] = m['purity']
            ablation_per_cluster.append(forms)

        ablation_names = ['R_full', 'R_simple', 'R_sub', 'R_exp',
                          'R_log', 'R_add']
        ablation_rhos = {}

        P(f"    {'Form':<12} {'rho':>8} {'|rho|':>8} {'p-value':>12}")
        P(f"    {'-'*42}")
        for form_name in ablation_names:
            vals = np.array([a[form_name] for a in ablation_per_cluster])
            valid = ~np.isnan(vals)
            if np.sum(valid) >= 10:
                rho, pval = spearmanr(vals[valid], purity_arr[valid])
                ablation_rhos[form_name] = {
                    'rho': float(rho),
                    'abs_rho': float(abs(rho)),
                    'p_value': float(pval),
                    'n_valid': int(np.sum(valid))
                }
            else:
                ablation_rhos[form_name] = {
                    'rho': float('nan'), 'abs_rho': float('nan'),
                    'p_value': float('nan'), 'n_valid': int(np.sum(valid))
                }
            r = ablation_rhos[form_name]
            P(f"    {form_name:<12} {r['rho']:>8.4f} {r['abs_rho']:>8.4f} "
              f"{r['p_value']:>12.6f}")

        best_ablation = max(ablation_names,
                            key=lambda f: ablation_rhos[f]['abs_rho'])
        r_full_is_best = best_ablation == 'R_full'
        P(f"\n    Best ablation form: {best_ablation} "
          f"(|rho|={ablation_rhos[best_ablation]['abs_rho']:.4f})")
        P(f"    R_full is best: {r_full_is_best}")

        results['test4_ablation'][model_name] = {
            'form_rhos': ablation_rhos,
            'best_form': best_ablation,
            'R_full_is_best': r_full_is_best,
        }

        # ============================================================
        # TEST 5: Novel Predictions
        # ============================================================
        P()
        P("  " + "-" * 60)
        P("  TEST 5: Novel Predictions")
        P("  " + "-" * 60)

        # P1: R_full correlates meaningfully with purity
        r_full_abs_rho = component_rhos['R_full']['abs_rho']
        pred1_pass = r_full_abs_rho > 0.4
        P(f"    P1: R_full |rho| > 0.4 with purity? "
          f"{r_full_abs_rho:.4f} -> {pred1_pass}")

        # P2: Top quartile R_full has high purity
        valid_mask = ~np.isnan(r_full_vals)
        valid_rfull = r_full_vals[valid_mask]
        valid_purity = purity_arr[valid_mask]
        q75 = np.percentile(valid_rfull, 75)
        top_q_mask = valid_rfull >= q75
        top_q_purity = float(np.mean(valid_purity[top_q_mask]))
        pred2_pass = top_q_purity > 0.7
        P(f"    P2: Top quartile R_full mean purity > 0.7? "
          f"{top_q_purity:.4f} -> {pred2_pass}")

        # P3: R_full is best overall single metric
        all_metrics_combined = {}
        all_metrics_combined.update(component_rhos)
        all_metrics_combined.update(ablation_rhos)
        best_overall = max(all_metrics_combined.keys(),
                           key=lambda k: all_metrics_combined[k]['abs_rho'])
        pred3_pass = best_overall == 'R_full'
        P(f"    P3: R_full is best overall metric? Best={best_overall} "
          f"(|rho|={all_metrics_combined[best_overall]['abs_rho']:.4f}) "
          f"-> {pred3_pass}")

        preds_passed = sum([pred1_pass, pred2_pass, pred3_pass])
        P(f"    Predictions passed: {preds_passed}/3")

        results['test5_novel_predictions'][model_name] = {
            'P1_rho_above_04': {
                'value': r_full_abs_rho,
                'threshold': 0.4,
                'passed': pred1_pass,
            },
            'P2_top_quartile_purity': {
                'value': top_q_purity,
                'threshold': 0.7,
                'passed': pred2_pass,
            },
            'P3_best_metric': {
                'best_metric': best_overall,
                'best_abs_rho': all_metrics_combined[best_overall]['abs_rho'],
                'R_full_abs_rho': r_full_abs_rho,
                'passed': pred3_pass,
            },
            'total_passed': preds_passed,
        }

        # Save per-cluster data
        results['test1_steiger_comparison'][model_name]['cluster_metrics'] = [
            {
                'purity': m['purity'],
                'type': m['type'],
                'E': m['E'],
                'grad_S': m['grad_S'],
                'sigma': m['sigma'],
                'Df': m['Df'],
                'R_simple': m['R_simple'],
                'R_full': m['R_full'],
                'E_over_std': m['E_over_std'],
            }
            for m in metrics_per_cluster
        ]

        t_model_end = time.time()
        P(f"\n  Model {model_name} completed in "
          f"{t_model_end - t_model_start:.1f}s")

        # Free memory
        del model, all_embeddings
        import gc
        gc.collect()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    P()
    P("=" * 70)
    P("FINAL VERDICT COMPUTATION")
    P("=" * 70)

    n_archs = len(model_names)

    # Criterion 1: Steiger significance (R_full > E)
    steiger_sig_count = 0
    for mn in model_names:
        t1 = results['test1_steiger_comparison'][mn]
        if t1['steiger_rfull_vs_e']['significant']:
            steiger_sig_count += 1
    P(f"\n  Steiger significant (R_full > E): {steiger_sig_count}/{n_archs}")

    # Criterion 2: Cross-validated R^2 (R_full > E)
    cv_rfull_wins_count = 0
    cv_negligible_count = 0
    for mn in model_names:
        t2 = results['test2_cross_validated_prediction'][mn]
        if t2['rfull_beats_e']:
            cv_rfull_wins_count += 1
        if t2['difference_negligible']:
            cv_negligible_count += 1
    P(f"  CV R^2 R_full > E: {cv_rfull_wins_count}/{n_archs}")
    P(f"  CV R^2 difference negligible (<0.01): {cv_negligible_count}/{n_archs}")

    # Supplementary: sigma^Df contribution
    sigma_df_helps_count = 0
    for mn in model_names:
        t3 = results['test3_nested_model_comparison'][mn]
        if t3['sigma_df_improves_aic']:
            sigma_df_helps_count += 1
    P(f"  sigma^Df improves AIC: {sigma_df_helps_count}/{n_archs}")

    # Supplementary: grad_S contribution
    grad_s_helps_count = 0
    for mn in model_names:
        t3 = results['test3_nested_model_comparison'][mn]
        if t3['grad_s_improves_aic']:
            grad_s_helps_count += 1
    P(f"  1/grad_S improves AIC: {grad_s_helps_count}/{n_archs}")

    # Supplementary: Steiger R_simple vs E
    steiger_simple_sig_count = 0
    for mn in model_names:
        t1 = results['test1_steiger_comparison'][mn]
        if t1['steiger_rsimple_vs_e']['significant']:
            steiger_simple_sig_count += 1
    P(f"  Steiger significant (R_simple > E): "
      f"{steiger_simple_sig_count}/{n_archs}")

    # Supplementary: Steiger R_full vs R_simple
    steiger_full_vs_simple_sig = 0
    for mn in model_names:
        t1 = results['test1_steiger_comparison'][mn]
        if t1['steiger_rfull_vs_rsimple']['significant']:
            steiger_full_vs_simple_sig += 1
    P(f"  Steiger significant (R_full > R_simple): "
      f"{steiger_full_vs_simple_sig}/{n_archs}")

    # Ablation: R_full best form
    best_ablation_count = 0
    for mn in model_names:
        if results['test4_ablation'][mn]['R_full_is_best']:
            best_ablation_count += 1
    P(f"  R_full is best ablation: {best_ablation_count}/{n_archs}")

    # Novel predictions
    novel_pass_count = 0
    for mn in model_names:
        if results['test5_novel_predictions'][mn]['total_passed'] >= 2:
            novel_pass_count += 1
    P(f"  Novel predictions >= 2/3: {novel_pass_count}/{n_archs}")

    # ---- Apply pre-registered criteria ----
    not_tautological = (
        steiger_sig_count >= 2 and
        cv_rfull_wins_count >= 2
    )

    steiger_all_ns = steiger_sig_count == 0
    cv_all_negligible = cv_negligible_count == n_archs

    tautological = (
        steiger_all_ns and
        cv_all_negligible
    )

    P()
    P("  PRE-REGISTERED CRITERIA CHECK:")
    P(f"    NOT_TAUTOLOGICAL: Steiger sig >= 2/3 [{steiger_sig_count >= 2}] "
      f"AND CV R^2 wins >= 2/3 [{cv_rfull_wins_count >= 2}] "
      f"=> {not_tautological}")
    P(f"    TAUTOLOGICAL: Steiger NS all 3 [{steiger_all_ns}] "
      f"AND CV diff < 0.01 all 3 [{cv_all_negligible}] "
      f"=> {tautological}")

    if not_tautological:
        verdict = "NOT TAUTOLOGICAL"
        explanation = (
            f"R_full significantly outperforms E alone via Steiger's test "
            f"on {steiger_sig_count}/{n_archs} architectures (p < 0.05), "
            f"and cross-validated R^2(R_full) > R^2(E) on "
            f"{cv_rfull_wins_count}/{n_archs} architectures."
        )
    elif tautological:
        verdict = "TAUTOLOGICAL"
        explanation = (
            f"R_full does NOT significantly outperform E alone on any "
            f"architecture (Steiger NS on all {n_archs}), and "
            f"cross-validated R^2 differences are negligible (< 0.01) "
            f"on all {n_archs} architectures. "
            f"R is effectively just E with noise from the other terms."
        )
    else:
        verdict = "INCONCLUSIVE"
        explanation = (
            f"Mixed results: Steiger significant on "
            f"{steiger_sig_count}/{n_archs}, CV R^2 wins on "
            f"{cv_rfull_wins_count}/{n_archs}, CV negligible on "
            f"{cv_negligible_count}/{n_archs}. "
            f"Does not meet either NOT_TAUTOLOGICAL or TAUTOLOGICAL criteria."
        )

    # Add nuance about WHAT R is
    nuance = []
    if grad_s_helps_count >= 2:
        nuance.append(
            "The grad_S (dispersion) component adds significant predictive "
            "value beyond E alone, as measured by AIC improvement on "
            f"{grad_s_helps_count}/{n_archs} architectures."
        )
    else:
        nuance.append(
            "The grad_S component does NOT consistently improve predictions "
            f"beyond E alone (AIC improvement on {grad_s_helps_count}/{n_archs})."
        )

    if sigma_df_helps_count >= 2:
        nuance.append(
            "The sigma^Df (fractal scaling) term adds measurable value "
            f"on {sigma_df_helps_count}/{n_archs} architectures."
        )
    else:
        nuance.append(
            "The sigma^Df term is decorative -- it does NOT consistently "
            f"improve predictions ({sigma_df_helps_count}/{n_archs} by AIC)."
        )

    # E_over_std comparison (is R_simple just mean/std?)
    e_over_std_note = (
        "Note: R_simple = E/grad_S is algebraically identical to "
        "mean(cosine_sims)/std(cosine_sims), i.e., a signal-to-noise ratio. "
        "Whether a well-constructed SNR is 'tautological' or 'explanatory' "
        "is a philosophical question, not an empirical one."
    )
    nuance.append(e_over_std_note)

    P(f"\n  VERDICT: {verdict}")
    P(f"  Explanation: {explanation}")
    P(f"\n  NUANCE:")
    for n_item in nuance:
        P(f"    - {n_item}")

    results['verdict'] = {
        'verdict': verdict,
        'explanation': explanation,
        'nuance': nuance,
        'criteria_results': {
            'steiger_significant_rfull_vs_e': f"{steiger_sig_count}/{n_archs}",
            'cv_r2_rfull_beats_e': f"{cv_rfull_wins_count}/{n_archs}",
            'cv_r2_difference_negligible': f"{cv_negligible_count}/{n_archs}",
            'sigma_df_improves_aic': f"{sigma_df_helps_count}/{n_archs}",
            'grad_s_improves_aic': f"{grad_s_helps_count}/{n_archs}",
            'steiger_significant_rsimple_vs_e':
                f"{steiger_simple_sig_count}/{n_archs}",
            'steiger_significant_rfull_vs_rsimple':
                f"{steiger_full_vs_simple_sig}/{n_archs}",
            'best_ablation_is_rfull': f"{best_ablation_count}/{n_archs}",
            'novel_predictions_pass': f"{novel_pass_count}/{n_archs}",
        }
    }

    return results


if __name__ == '__main__':
    start = time.time()
    results = run_all_tests()
    elapsed = time.time() - start
    results['metadata']['elapsed_seconds'] = elapsed
    P(f"\nTotal time: {elapsed:.1f}s")

    out_path = (
        r"D:\CCC 2.0\AI\agent-governance-system"
        r"\THOUGHT\LAB\FORMULA\v2\q20_tautology\results"
        r"\test_v3_q20_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    P(f"\nResults saved to: {out_path}")
