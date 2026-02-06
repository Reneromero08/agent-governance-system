"""
Q15 v2 FIXED: Does R have a genuine Bayesian interpretation?

Audit fixes applied:
  1. E ~ 1/trace(cov) acknowledged as mathematical identity, not empirical finding.
     Real test: does R correlate with quantities NOT trivially related to E?
  2. Test 4 fixed: R computed on hidden layer activations, not raw input features.
  3. KL divergence was trivially = Fisher * scalar. Removed duplicate; compute properly.
  4. Automated verdict from pre-registered criteria only. No editorial overrides.
  5. Intensive property test: also uses California Housing (8-dim, n >> d) to
     remove rank-deficiency confound.

Data: 20 Newsgroups (text) + California Housing (tabular).
Architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1.

Pre-registered criteria (from README.md, refined in prompt):
  CONFIRM: R_full correlates > 0.7 with a NON-trivial Bayesian quantity
           in >= 2/3 architectures, AND R_full is intensive (CV < 0.15),
           AND R_full gating F1 > E alone by 5%.
  FALSIFY: R_full correlates < 0.3 with all Bayesian quantities
           OR R_full is NOT intensive AND R_full gating worse than E alone.
  INCONCLUSIVE: otherwise.
"""

import sys
import os
import json
import time
import warnings
import traceback
import importlib.util

import numpy as np
from scipy import stats
from sklearn.datasets import fetch_20newsgroups, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ---- Import formula module ----
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

compute_E = formula.compute_E
compute_grad_S = formula.compute_grad_S
compute_sigma = formula.compute_sigma
compute_Df = formula.compute_Df
compute_R_simple = formula.compute_R_simple
compute_R_full = formula.compute_R_full
compute_all = formula.compute_all

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "..", "results")
CACHE_DIR = os.path.join(CODE_DIR, "..", "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

ARCHITECTURES = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
]


# =====================================================================
# UTILITIES
# =====================================================================

def safe_spearman(x, y):
    """Spearman correlation with NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def compute_likelihood_precision_trace(embeddings):
    """Likelihood precision = 1/trace(covariance). Trivially related to E for unit-norm."""
    if embeddings.shape[0] < 3:
        return float("nan")
    cov = np.cov(embeddings, rowvar=False)
    tr = np.trace(cov)
    if tr < 1e-15:
        return float("nan")
    return 1.0 / tr


def compute_bootstrap_posterior_precision(embeddings, n_bootstrap=500, rng=None):
    """
    Bootstrap posterior precision: 1/trace(var of bootstrap centroids).
    This is a proper Bayesian-flavored quantity: it measures how precisely
    we can estimate the population centroid from samples.
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    n = embeddings.shape[0]
    if n < 5:
        return float("nan")
    centroids = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        centroids.append(embeddings[idx].mean(axis=0))
    centroids = np.array(centroids)
    var_trace = np.trace(np.cov(centroids, rowvar=False))
    if var_trace < 1e-15:
        return float("nan")
    return 1.0 / var_trace


def compute_effective_sample_size(embeddings):
    """
    Effective sample size: n * (1 - mean_pairwise_correlation).
    For independent observations, ESS = n. For perfectly correlated, ESS ~ 1.
    """
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    mean_corr = compute_E(embeddings)
    ess = n * (1.0 - mean_corr)
    return float(max(ess, 1.0))


def compute_inv_grad_S(embeddings):
    """1/grad_S -- the trivial baseline."""
    gs = compute_grad_S(embeddings)
    if np.isnan(gs) or gs < 1e-10:
        return float("nan")
    return 1.0 / gs


# =====================================================================
# DATA LOADING WITH CACHING
# =====================================================================

def load_20newsgroups_base():
    """Load raw 20 Newsgroups data (no embeddings)."""
    print("  Loading 20 Newsgroups text data...")
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = newsgroups.data
    labels = newsgroups.target
    categories = newsgroups.target_names
    print(f"    Total documents: {len(texts)}, categories: {len(categories)}")
    return texts, labels, categories


def encode_with_cache(texts, model_name):
    """Encode texts with sentence-transformer, caching to disk."""
    safe_name = model_name.replace("/", "_").replace("-", "_")
    cache_path = os.path.join(CACHE_DIR, f"newsgroups_{safe_name}.npy")

    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings for {model_name}...")
        embeddings = np.load(cache_path)
        print(f"    Loaded from cache: shape={embeddings.shape}")
        return embeddings

    print(f"  Encoding with {model_name} (this may take a while on CPU)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings, dtype=np.float64)
    np.save(cache_path, embeddings)
    print(f"    Saved to cache: shape={embeddings.shape}")
    return embeddings


def create_60_clusters(embeddings, labels, categories, rng):
    """
    Create 60 clusters:
      20 pure (200 docs from one category)
      20 mixed-2 (100 docs from each of 2 categories)
      20 random (200 random docs)
    """
    n_cats = len(categories)
    clusters = []

    cat_indices = {}
    for cat_id in range(n_cats):
        cat_indices[cat_id] = np.where(labels == cat_id)[0]

    # 20 pure clusters
    for i in range(20):
        cat_id = i % n_cats
        available = cat_indices[cat_id]
        if len(available) < 200:
            idx = rng.choice(available, size=200, replace=True)
        else:
            idx = rng.choice(available, size=200, replace=False)
        cluster_labels = labels[idx]
        purity = np.max(np.bincount(cluster_labels, minlength=n_cats)) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "type": "pure",
            "purity": float(purity),
            "category": categories[cat_id],
            "size": len(idx),
        })

    # 20 mixed-2 clusters
    used_pairs = set()
    for i in range(20):
        attempts = 0
        while attempts < 100:
            c1, c2 = rng.choice(n_cats, size=2, replace=False)
            pair = (min(c1, c2), max(c1, c2))
            if pair not in used_pairs or attempts > 50:
                used_pairs.add(pair)
                break
            attempts += 1
        idx1 = rng.choice(cat_indices[c1], size=100, replace=len(cat_indices[c1]) < 100)
        idx2 = rng.choice(cat_indices[c2], size=100, replace=len(cat_indices[c2]) < 100)
        idx = np.concatenate([idx1, idx2])
        cluster_labels = labels[idx]
        purity = np.max(np.bincount(cluster_labels, minlength=n_cats)) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "type": "mixed",
            "purity": float(purity),
            "category": f"{categories[c1]}+{categories[c2]}",
            "size": len(idx),
        })

    # 20 random clusters
    all_indices = np.arange(len(labels))
    for i in range(20):
        idx = rng.choice(all_indices, size=200, replace=False)
        cluster_labels = labels[idx]
        purity = np.max(np.bincount(cluster_labels, minlength=n_cats)) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "type": "random",
            "purity": float(purity),
            "category": "random",
            "size": len(idx),
        })

    print(f"    Created {len(clusters)} clusters: "
          f"{sum(1 for c in clusters if c['type']=='pure')} pure, "
          f"{sum(1 for c in clusters if c['type']=='mixed')} mixed, "
          f"{sum(1 for c in clusters if c['type']=='random')} random")
    return clusters


# =====================================================================
# TEST 1: R vs Bayesian Quantities (per architecture)
# =====================================================================

def test1_bayesian_correlation(texts, labels, categories):
    """
    For each architecture, create 60 clusters from 20 Newsgroups.
    Compute R_full, R_simple, E and compare against:
      - 1/trace(cov): mathematically identical to E for unit-norm (acknowledged).
      - Bootstrap posterior precision: proper non-trivial Bayesian quantity.
      - Effective sample size: n*(1 - mean_pairwise_corr), proper Bayesian quantity.
    The REAL test: does R correlate with quantities NOT trivially related to E?
    """
    print("\n" + "=" * 70)
    print("TEST 1: R vs Bayesian Quantities (per architecture)")
    print("=" * 70)

    results_by_arch = {}

    for arch_idx, model_name in enumerate(ARCHITECTURES):
        print(f"\n--- Architecture: {model_name} ---")
        embeddings = encode_with_cache(texts, model_name)
        rng = np.random.RandomState(SEED + arch_idx)
        clusters = create_60_clusters(embeddings, labels, categories, rng)

        metrics = {
            "R_full": [], "R_simple": [], "E": [], "inv_grad_S": [],
            "lik_prec_trace": [], "bootstrap_post_prec": [], "eff_sample_size": [],
        }

        for c_idx, cluster in enumerate(clusters):
            emb = cluster["embeddings"]
            all_comp = compute_all(emb)
            metrics["R_full"].append(all_comp["R_full"])
            metrics["R_simple"].append(all_comp["R_simple"])
            metrics["E"].append(all_comp["E"])
            metrics["inv_grad_S"].append(compute_inv_grad_S(emb))
            metrics["lik_prec_trace"].append(compute_likelihood_precision_trace(emb))
            boot_rng = np.random.RandomState(SEED + arch_idx * 1000 + c_idx)
            metrics["bootstrap_post_prec"].append(
                compute_bootstrap_posterior_precision(emb, n_bootstrap=500, rng=boot_rng)
            )
            metrics["eff_sample_size"].append(compute_effective_sample_size(emb))

        for k in metrics:
            metrics[k] = np.array(metrics[k])

        formula_keys = ["R_full", "R_simple", "E", "inv_grad_S"]
        bayesian_keys = ["lik_prec_trace", "bootstrap_post_prec", "eff_sample_size"]

        print(f"\n  Spearman correlations (n={len(clusters)} clusters):")
        print(f"  {'':20s} {'lik_prec_trace':>20s} {'boot_post_prec':>20s} {'eff_sample_size':>20s}")
        print("  " + "-" * 80)

        arch_corrs = {}
        for fk in formula_keys:
            row = f"  {fk:20s}"
            arch_corrs[fk] = {}
            for bk in bayesian_keys:
                rho, p = safe_spearman(metrics[fk], metrics[bk])
                row += f"  {rho:+7.3f} (p={p:.3f})" if np.isfinite(rho) else f"  {'NaN':>20s}"
                arch_corrs[fk][bk] = {"rho": float(rho) if np.isfinite(rho) else None,
                                       "p": float(p) if np.isfinite(p) else None}
            print(row)

        e_lik_rho = arch_corrs["E"]["lik_prec_trace"]["rho"]
        print(f"\n  NOTE: E vs lik_prec_trace rho = {e_lik_rho} "
              "(expected ~1.0, mathematical identity for unit-norm embeddings)")

        rfull_boot = arch_corrs["R_full"]["bootstrap_post_prec"]["rho"]
        rfull_ess = arch_corrs["R_full"]["eff_sample_size"]["rho"]
        e_boot = arch_corrs["E"]["bootstrap_post_prec"]["rho"]
        e_ess = arch_corrs["E"]["eff_sample_size"]["rho"]

        print(f"  REAL TEST (non-trivial Bayesian quantities):")
        print(f"    R_full vs bootstrap_post_prec:  {rfull_boot}")
        print(f"    R_full vs eff_sample_size:      {rfull_ess}")
        print(f"    E vs bootstrap_post_prec:       {e_boot}")
        print(f"    E vs eff_sample_size:           {e_ess}")

        results_by_arch[model_name] = {
            "n_clusters": len(clusters),
            "correlations": arch_corrs,
            "cluster_types": [c["type"] for c in clusters],
            "cluster_purities": [c["purity"] for c in clusters],
            "metrics_summary": {
                k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                     "n_valid": int(np.isfinite(v).sum())}
                for k, v in metrics.items()
            },
        }

    # Cross-architecture summary
    print("\n  CROSS-ARCHITECTURE SUMMARY:")
    print("  " + "-" * 80)
    for bk in bayesian_keys:
        print(f"\n  vs {bk}:")
        for fk in formula_keys:
            rhos = []
            for model_name in ARCHITECTURES:
                r = results_by_arch[model_name]["correlations"][fk][bk]["rho"]
                if r is not None:
                    rhos.append(r)
            if rhos:
                mean_rho = np.mean(rhos)
                print(f"    {fk:20s}: mean rho = {mean_rho:+.3f} "
                      f"(across {len(rhos)} architectures: {[f'{r:+.3f}' for r in rhos]})")

    return {"status": "complete", "by_architecture": results_by_arch}


# =====================================================================
# TEST 2: Intensive Property (FIXED)
# =====================================================================

def test2_intensive_property(texts, labels, categories):
    """
    Part A: 20 Newsgroups, 3 largest pure clusters, subsample N = 20,50,100,200,500.
    Part B: California Housing (8-dim), 20 geographic clusters, same N values.
    Threshold: CV < 0.15 for intensive.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Intensive Property (FIXED)")
    print("=" * 70)

    subsample_sizes = [20, 50, 100, 200, 500]
    n_repeats = 30

    # ---- Part A: 20 Newsgroups with first architecture ----
    print("\n  --- Part A: 20 Newsgroups (all-MiniLM-L6-v2) ---")
    embeddings = encode_with_cache(texts, ARCHITECTURES[0])

    cat_sizes = [(cat_id, (labels == cat_id).sum()) for cat_id in range(len(categories))]
    cat_sizes.sort(key=lambda x: -x[1])
    top3_cats = [cat_sizes[i][0] for i in range(3)]
    print(f"    3 largest categories: {[categories[c] for c in top3_cats]} "
          f"(sizes: {[cat_sizes[i][1] for i in range(3)]})")

    text_results = []
    for cat_id in top3_cats:
        cat_mask = labels == cat_id
        cat_emb = embeddings[cat_mask]
        cat_n = cat_emb.shape[0]
        print(f"\n    Category: {categories[cat_id]} (n={cat_n})")

        R_full_means = {}
        R_simple_means = {}

        for N in subsample_sizes:
            if N > cat_n:
                continue
            R_full_samples = []
            R_simple_samples = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(SEED + cat_id * 10000 + N * 100 + rep)
                idx = rng.choice(cat_n, size=N, replace=False)
                sub = cat_emb[idx]
                R_full_samples.append(compute_R_full(sub))
                R_simple_samples.append(compute_R_simple(sub))

            R_full_means[N] = float(np.nanmean(R_full_samples))
            R_simple_means[N] = float(np.nanmean(R_simple_samples))
            print(f"      N={N:4d}: R_full={R_full_means[N]:.4f}, R_simple={R_simple_means[N]:.4f}")

        def cv_across_N(vals_by_N):
            vals = np.array([v for v in vals_by_N.values() if np.isfinite(v)])
            if len(vals) < 2 or np.mean(np.abs(vals)) < 1e-15:
                return float("nan")
            return float(np.std(vals) / np.abs(np.mean(vals)))

        cv_full = cv_across_N(R_full_means)
        cv_simple = cv_across_N(R_simple_means)
        print(f"      CV: R_full={cv_full:.4f}, R_simple={cv_simple:.4f}")

        text_results.append({
            "category": categories[cat_id],
            "n_total": cat_n,
            "cv_R_full": cv_full,
            "cv_R_simple": cv_simple,
            "R_full_by_N": {str(k): v for k, v in R_full_means.items()},
            "R_simple_by_N": {str(k): v for k, v in R_simple_means.items()},
        })

    # ---- Part B: California Housing (8-dim, removes rank deficiency confound) ----
    print("\n  --- Part B: California Housing (8-dim features) ---")
    housing = fetch_california_housing()
    X_housing = StandardScaler().fit_transform(housing.data)
    lat = housing.data[:, -2]
    lon = housing.data[:, -1]
    coords = np.column_stack([lat, lon])

    print(f"    Housing data shape: {X_housing.shape}")
    km = KMeans(n_clusters=20, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(coords)

    housing_results = []
    for cl in range(20):
        cl_mask = cluster_labels == cl
        cl_emb = X_housing[cl_mask]
        cl_n = cl_emb.shape[0]

        if cl_n < 500:
            continue

        R_full_means = {}
        R_simple_means = {}

        for N in subsample_sizes:
            R_full_samples = []
            R_simple_samples = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(SEED + cl * 10000 + N * 100 + rep)
                idx = rng.choice(cl_n, size=N, replace=False)
                sub = cl_emb[idx]
                R_full_samples.append(compute_R_full(sub))
                R_simple_samples.append(compute_R_simple(sub))

            R_full_means[N] = float(np.nanmean(R_full_samples))
            R_simple_means[N] = float(np.nanmean(R_simple_samples))

        def cv_across_N_h(vals_by_N):
            vals = np.array([v for v in vals_by_N.values() if np.isfinite(v)])
            if len(vals) < 2 or np.mean(np.abs(vals)) < 1e-15:
                return float("nan")
            return float(np.std(vals) / np.abs(np.mean(vals)))

        cv_full = cv_across_N_h(R_full_means)
        cv_simple = cv_across_N_h(R_simple_means)
        housing_results.append({
            "cluster": cl,
            "n_total": cl_n,
            "cv_R_full": cv_full,
            "cv_R_simple": cv_simple,
            "R_full_by_N": {str(k): v for k, v in R_full_means.items()},
            "R_simple_by_N": {str(k): v for k, v in R_simple_means.items()},
        })
        print(f"    Cluster {cl:2d} (n={cl_n:5d}): CV R_full={cv_full:.4f}, CV R_simple={cv_simple:.4f}")

    # Summary
    text_cvs = [r["cv_R_full"] for r in text_results if np.isfinite(r["cv_R_full"])]
    housing_cvs = [r["cv_R_full"] for r in housing_results if np.isfinite(r["cv_R_full"])]
    all_cvs = text_cvs + housing_cvs

    text_simple_cvs = [r["cv_R_simple"] for r in text_results if np.isfinite(r["cv_R_simple"])]
    housing_simple_cvs = [r["cv_R_simple"] for r in housing_results if np.isfinite(r["cv_R_simple"])]
    all_simple_cvs = text_simple_cvs + housing_simple_cvs

    print(f"\n  SUMMARY:")
    if text_cvs:
        print(f"    Text (384-dim) R_full CV:   mean={np.mean(text_cvs):.4f}, values={[f'{v:.4f}' for v in text_cvs]}")
    if housing_cvs:
        print(f"    Housing (8-dim) R_full CV:  mean={np.mean(housing_cvs):.4f}, values={[f'{v:.4f}' for v in housing_cvs]}")
    if all_cvs:
        print(f"    Overall R_full CV:          mean={np.mean(all_cvs):.4f}")
        print(f"    Overall R_simple CV:        mean={np.mean(all_simple_cvs):.4f}")
        print(f"    Intensive threshold: CV < 0.15")
        print(f"    R_full intensive?  mean CV {np.mean(all_cvs):.4f} < 0.15 -> "
              f"{'YES' if np.mean(all_cvs) < 0.15 else 'NO'}")
        print(f"    R_simple intensive? mean CV {np.mean(all_simple_cvs):.4f} < 0.15 -> "
              f"{'YES' if np.mean(all_simple_cvs) < 0.15 else 'NO'}")

    return {
        "status": "complete",
        "text_results": text_results,
        "housing_results": housing_results,
        "summary": {
            "text_mean_cv_R_full": float(np.mean(text_cvs)) if text_cvs else None,
            "housing_mean_cv_R_full": float(np.mean(housing_cvs)) if housing_cvs else None,
            "overall_mean_cv_R_full": float(np.mean(all_cvs)) if all_cvs else None,
            "overall_mean_cv_R_simple": float(np.mean(all_simple_cvs)) if all_simple_cvs else None,
        },
    }


# =====================================================================
# TEST 3: R-Gating Quality (per architecture)
# =====================================================================

def test3_gating_quality(texts, labels, categories):
    """
    Using 60 clusters, label as "good" (pure, purity > 0.8) or "bad" (random, purity < 0.3).
    Find optimal threshold on 50% train, evaluate F1 on 50% test.
    Compare: R_simple, R_full, E alone, 1/trace(cov), 1/grad_S.
    """
    print("\n" + "=" * 70)
    print("TEST 3: R-Gating Quality (per architecture)")
    print("=" * 70)

    results_by_arch = {}

    for arch_idx, model_name in enumerate(ARCHITECTURES):
        print(f"\n--- Architecture: {model_name} ---")
        embeddings = encode_with_cache(texts, model_name)
        rng = np.random.RandomState(SEED + arch_idx + 100)
        clusters = create_60_clusters(embeddings, labels, categories, rng)

        feature_names = ["R_full", "R_simple", "E", "lik_prec_trace", "inv_grad_S"]
        features = {k: [] for k in feature_names}
        cluster_labels_binary = []

        for c in clusters:
            emb = c["embeddings"]
            if c["type"] == "pure" and c["purity"] > 0.8:
                cluster_labels_binary.append(1)
            elif c["type"] == "random" and c["purity"] < 0.3:
                cluster_labels_binary.append(0)
            else:
                cluster_labels_binary.append(-1)

            all_comp = compute_all(emb)
            features["R_full"].append(all_comp["R_full"])
            features["R_simple"].append(all_comp["R_simple"])
            features["E"].append(all_comp["E"])
            features["lik_prec_trace"].append(compute_likelihood_precision_trace(emb))
            features["inv_grad_S"].append(compute_inv_grad_S(emb))

        cluster_labels_binary = np.array(cluster_labels_binary)
        valid_mask = cluster_labels_binary >= 0
        for k in features:
            features[k] = np.array(features[k])[valid_mask]
            nan_mask = ~np.isfinite(features[k])
            if nan_mask.any():
                median_val = np.nanmedian(features[k])
                features[k][nan_mask] = median_val

        binary_labels = cluster_labels_binary[valid_mask]
        n_good = int((binary_labels == 1).sum())
        n_bad = int((binary_labels == 0).sum())
        print(f"    Good clusters (pure, purity>0.8): {n_good}")
        print(f"    Bad clusters (random, purity<0.3): {n_bad}")
        print(f"    Total for classification: {len(binary_labels)}")

        from sklearn.model_selection import train_test_split as tts
        try:
            train_idx, test_idx = tts(
                np.arange(len(binary_labels)),
                test_size=0.5, random_state=SEED + arch_idx,
                stratify=binary_labels
            )
        except ValueError:
            all_idx = np.arange(len(binary_labels))
            rng2 = np.random.RandomState(SEED + arch_idx)
            rng2.shuffle(all_idx)
            mid = len(all_idx) // 2
            train_idx, test_idx = all_idx[:mid], all_idx[mid:]

        print(f"    Train: {len(train_idx)} (good={binary_labels[train_idx].sum()}, "
              f"bad={(1-binary_labels[train_idx]).sum()})")
        print(f"    Test:  {len(test_idx)} (good={binary_labels[test_idx].sum()}, "
              f"bad={(1-binary_labels[test_idx]).sum()})")

        arch_results = {}
        print(f"\n    {'Method':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        print("    " + "-" * 55)

        for method_name in feature_names:
            vals = features[method_name]
            train_vals = vals[train_idx]
            train_labels = binary_labels[train_idx]
            test_vals = vals[test_idx]
            test_labels = binary_labels[test_idx]

            best_f1 = -1
            best_thresh = 0
            finite_train = train_vals[np.isfinite(train_vals)]
            if len(finite_train) < 3:
                arch_results[method_name] = {"precision": 0, "recall": 0, "f1": 0}
                print(f"    {method_name:20s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")
                continue

            thresholds = np.percentile(finite_train, np.linspace(5, 95, 50))
            for thresh in thresholds:
                preds = (train_vals > thresh).astype(int)
                tp = ((preds == 1) & (train_labels == 1)).sum()
                fp = ((preds == 1) & (train_labels == 0)).sum()
                fn = ((preds == 0) & (train_labels == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            test_preds = (test_vals > best_thresh).astype(int)
            tp = ((test_preds == 1) & (test_labels == 1)).sum()
            fp = ((test_preds == 1) & (test_labels == 0)).sum()
            fn = ((test_preds == 0) & (test_labels == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            arch_results[method_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "threshold": float(best_thresh),
            }
            print(f"    {method_name:20s} {precision:10.3f} {recall:10.3f} {f1:10.3f}")

        f1_full = arch_results.get("R_full", {}).get("f1", 0)
        f1_E = arch_results.get("E", {}).get("f1", 0)
        diff = f1_full - f1_E
        print(f"\n    KEY: R_full F1={f1_full:.3f}, E alone F1={f1_E:.3f}, diff={diff:+.3f}")
        print(f"    R_full > E + 5%? {'YES' if diff > 0.05 else 'NO'}")

        results_by_arch[model_name] = {
            "n_good": n_good,
            "n_bad": n_bad,
            "results": arch_results,
            "f1_R_full_minus_E": float(diff),
        }

    # Cross-architecture summary
    print("\n  CROSS-ARCHITECTURE SUMMARY:")
    diffs = [results_by_arch[m]["f1_R_full_minus_E"] for m in ARCHITECTURES]
    for m in ARCHITECTURES:
        r = results_by_arch[m]
        print(f"    {m}: R_full F1={r['results'].get('R_full',{}).get('f1',0):.3f}, "
              f"E F1={r['results'].get('E',{}).get('f1',0):.3f}, "
              f"diff={r['f1_R_full_minus_E']:+.3f}")
    n_better = sum(1 for d in diffs if d > 0.05)
    print(f"    Architectures where R_full > E + 5%: {n_better}/{len(diffs)}")

    return {"status": "complete", "by_architecture": results_by_arch}


# =====================================================================
# TEST 4: Neural Network Bayesian Test (FIXED)
# =====================================================================

def test4_neural_bayesian():
    """
    California Housing, 1-hidden-layer NN (8->64->1), 10 seeds.
    FIXED: Compute R on HIDDEN LAYER ACTIVATIONS (64-dim), not raw features.
    Compute Fisher information (diagonal approximation).
    Also compute R on raw 8-dim features as comparison.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Neural Network Bayesian Test (FIXED)")
    print("=" * 70)

    import torch
    import torch.nn as nn

    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lat = housing.data[:, -2]
    lon = housing.data[:, -1]
    coords = np.column_stack([lat, lon])
    km = KMeans(n_clusters=20, random_state=SEED, n_init=10)
    geo_labels = km.fit_predict(coords)

    n_seeds = 10
    results_by_seed = []

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---")
        torch.manual_seed(SEED + seed)
        np.random.seed(SEED + seed)

        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test, train_geo, test_geo = tts(
            X_scaled, y, geo_labels, test_size=0.2, random_state=SEED + seed
        )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 64)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 1)

            def forward(self, x):
                h = self.relu(self.fc1(x))
                return self.fc2(h)

            def hidden(self, x):
                return self.relu(self.fc1(x))

        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = nn.MSELoss()(pred, y_train_t)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"    Final training loss: {final_loss:.4f}")

        model.eval()
        with torch.no_grad():
            hidden_activations = model.hidden(X_train_t).numpy()
        print(f"    Hidden activations shape: {hidden_activations.shape}")

        R_full_hidden = []
        R_full_raw = []
        R_simple_hidden = []
        fisher_info_vals = []
        E_hidden_vals = []

        valid_clusters = []
        for cl in range(20):
            cl_mask = train_geo == cl
            if cl_mask.sum() < 30:
                continue
            valid_clusters.append(cl)

            cl_hidden = hidden_activations[cl_mask]
            cl_raw = X_train[cl_mask]

            R_full_hidden.append(compute_R_full(cl_hidden))
            R_simple_hidden.append(compute_R_simple(cl_hidden))
            E_hidden_vals.append(compute_E(cl_hidden))
            R_full_raw.append(compute_R_full(cl_raw))

            X_cl_t = torch.tensor(cl_raw, dtype=torch.float32)
            y_cl = y_train[cl_mask]
            y_cl_t = torch.tensor(y_cl, dtype=torch.float32).unsqueeze(1)

            fisher_diag = None
            n_cl = len(X_cl_t)
            sample_size = min(n_cl, 200)
            sample_idx = np.random.RandomState(SEED + seed * 100 + cl).choice(
                n_cl, size=sample_size, replace=False
            )
            for i in sample_idx:
                model.zero_grad()
                pred_i = model(X_cl_t[i:i+1])
                loss_i = nn.MSELoss()(pred_i, y_cl_t[i:i+1])
                loss_i.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                grad_vec = torch.cat(grads)

                if fisher_diag is None:
                    fisher_diag = grad_vec ** 2
                else:
                    fisher_diag = fisher_diag + grad_vec ** 2

            fisher_diag = fisher_diag / sample_size
            fisher_trace = fisher_diag.sum().item()
            fisher_info_vals.append(fisher_trace)

        R_full_hidden = np.array(R_full_hidden)
        R_full_raw = np.array(R_full_raw)
        R_simple_hidden = np.array(R_simple_hidden)
        fisher_info = np.array(fisher_info_vals)
        E_hidden = np.array(E_hidden_vals)

        n_valid = len(valid_clusters)
        print(f"    Valid clusters: {n_valid}")

        rho_full_hidden, p_full_hidden = safe_spearman(R_full_hidden, fisher_info)
        rho_full_raw, p_full_raw = safe_spearman(R_full_raw, fisher_info)
        rho_simple_hidden, p_simple_hidden = safe_spearman(R_simple_hidden, fisher_info)
        rho_E_hidden, p_E_hidden = safe_spearman(E_hidden, fisher_info)

        print(f"    R_full(hidden) vs Fisher:   rho={rho_full_hidden:+.3f} (p={p_full_hidden:.3f})")
        print(f"    R_full(raw) vs Fisher:      rho={rho_full_raw:+.3f} (p={p_full_raw:.3f})")
        print(f"    R_simple(hidden) vs Fisher: rho={rho_simple_hidden:+.3f} (p={p_simple_hidden:.3f})")
        print(f"    E(hidden) vs Fisher:        rho={rho_E_hidden:+.3f} (p={p_E_hidden:.3f})")

        results_by_seed.append({
            "seed": seed,
            "n_valid_clusters": n_valid,
            "final_loss": float(final_loss),
            "R_full_hidden_vs_fisher": {"rho": float(rho_full_hidden), "p": float(p_full_hidden)},
            "R_full_raw_vs_fisher": {"rho": float(rho_full_raw), "p": float(p_full_raw)},
            "R_simple_hidden_vs_fisher": {"rho": float(rho_simple_hidden), "p": float(p_simple_hidden)},
            "E_hidden_vs_fisher": {"rho": float(rho_E_hidden), "p": float(p_E_hidden)},
        })

    # Aggregate across seeds
    print("\n  AGGREGATE ACROSS SEEDS:")
    print("  " + "-" * 70)

    aggregate = {}
    for metric_name in ["R_full_hidden_vs_fisher", "R_full_raw_vs_fisher",
                        "R_simple_hidden_vs_fisher", "E_hidden_vs_fisher"]:
        rhos = [s[metric_name]["rho"] for s in results_by_seed
                if np.isfinite(s[metric_name]["rho"])]
        if len(rhos) >= 3:
            mean_rho = float(np.mean(rhos))
            std_rho = float(np.std(rhos))
            t_stat, t_p = stats.ttest_1samp(rhos, 0)
            sig = " *" if t_p < 0.05 else ""
            print(f"    {metric_name:35s}: mean_rho={mean_rho:+.3f} +/- {std_rho:.3f} "
                  f"(t={float(t_stat):.2f}, p={float(t_p):.4f}){sig}")
            aggregate[metric_name] = {
                "mean_rho": mean_rho,
                "std_rho": std_rho,
                "t_stat": float(t_stat),
                "t_p": float(t_p),
                "all_rhos": [float(r) for r in rhos],
            }
        else:
            print(f"    {metric_name:35s}: insufficient data")
            aggregate[metric_name] = {"mean_rho": None}

    return {
        "status": "complete",
        "n_seeds": n_seeds,
        "aggregate": aggregate,
        "by_seed": results_by_seed,
    }


# =====================================================================
# VERDICT
# =====================================================================

def compute_verdict(all_results):
    """
    Pre-registered criteria determine the verdict. No editorial overrides.
    """
    print("\n" + "=" * 70)
    print("VERDICT SYNTHESIS (pre-registered criteria only)")
    print("=" * 70)

    verdict_data = {"criteria": {}, "details": {}}

    # ---- Criterion A: Non-trivial Bayesian correlation ----
    t1 = all_results.get("test1", {})
    n_arch_pass = 0
    max_rhos_by_arch = {}
    all_rho_below_03 = True

    if t1.get("status") == "complete":
        for model_name in ARCHITECTURES:
            arch_data = t1.get("by_architecture", {}).get(model_name, {})
            corrs = arch_data.get("correlations", {})
            nontrivial_keys = ["bootstrap_post_prec", "eff_sample_size"]
            best_rho = 0
            for bk in nontrivial_keys:
                r = corrs.get("R_full", {}).get(bk, {}).get("rho")
                if r is not None:
                    if abs(r) > abs(best_rho):
                        best_rho = r
                    if abs(r) >= 0.3:
                        all_rho_below_03 = False

            max_rhos_by_arch[model_name] = best_rho
            if abs(best_rho) > 0.7:
                n_arch_pass += 1

            all_keys = ["lik_prec_trace", "bootstrap_post_prec", "eff_sample_size"]
            for bk in all_keys:
                r = corrs.get("R_full", {}).get(bk, {}).get("rho")
                if r is not None and abs(r) >= 0.3:
                    all_rho_below_03 = False

    criterion_a_confirm = n_arch_pass >= 2
    criterion_a_falsify = all_rho_below_03

    print(f"\n  Criterion A: Non-trivial Bayesian correlation")
    print(f"    Architectures with |rho| > 0.7 (non-trivial): {n_arch_pass}/3")
    for m, r in max_rhos_by_arch.items():
        print(f"      {m}: best non-trivial |rho| = {abs(r):.3f}")
    print(f"    CONFIRM (>= 2/3 with rho > 0.7): {'YES' if criterion_a_confirm else 'NO'}")
    print(f"    FALSIFY (all rho < 0.3): {'YES' if criterion_a_falsify else 'NO'}")

    verdict_data["criteria"]["A_nontrivial_correlation"] = {
        "confirm": bool(criterion_a_confirm),
        "falsify": bool(criterion_a_falsify),
        "max_rhos_by_arch": {k: float(v) for k, v in max_rhos_by_arch.items()},
    }

    # ---- Criterion B: Intensive property ----
    t2 = all_results.get("test2", {})
    overall_cv = None
    if t2.get("status") == "complete":
        overall_cv = t2.get("summary", {}).get("overall_mean_cv_R_full")

    criterion_b_confirm = overall_cv is not None and overall_cv < 0.15
    criterion_b_falsify = overall_cv is not None and overall_cv > 0.15

    print(f"\n  Criterion B: Intensive property (CV < 0.15)")
    print(f"    Overall mean CV of R_full: {overall_cv}")
    print(f"    CONFIRM (CV < 0.15): {'YES' if criterion_b_confirm else 'NO'}")
    print(f"    Part of FALSIFY (CV > 0.15): {'YES' if criterion_b_falsify else 'NO'}")

    verdict_data["criteria"]["B_intensive"] = {
        "confirm": bool(criterion_b_confirm),
        "falsify_component": bool(criterion_b_falsify),
        "overall_cv": overall_cv,
    }

    # ---- Criterion C: Gating F1 ----
    t3 = all_results.get("test3", {})
    gating_diffs = []
    if t3.get("status") == "complete":
        for model_name in ARCHITECTURES:
            d = t3.get("by_architecture", {}).get(model_name, {}).get("f1_R_full_minus_E", 0)
            gating_diffs.append(d)

    mean_gating_diff = float(np.mean(gating_diffs)) if gating_diffs else 0
    criterion_c_confirm = mean_gating_diff > 0.05
    criterion_c_falsify = mean_gating_diff < 0

    print(f"\n  Criterion C: Gating F1 (R_full > E alone + 5%)")
    for i, m in enumerate(ARCHITECTURES):
        if i < len(gating_diffs):
            print(f"    {m}: R_full - E = {gating_diffs[i]:+.3f}")
    print(f"    Mean diff: {mean_gating_diff:+.3f}")
    print(f"    CONFIRM (> +0.05): {'YES' if criterion_c_confirm else 'NO'}")
    print(f"    Part of FALSIFY (< 0): {'YES' if criterion_c_falsify else 'NO'}")

    verdict_data["criteria"]["C_gating"] = {
        "confirm": bool(criterion_c_confirm),
        "falsify_component": bool(criterion_c_falsify),
        "mean_diff": mean_gating_diff,
        "diffs_by_arch": gating_diffs,
    }

    # ---- Apply pre-registered decision rules ----
    if criterion_a_confirm and criterion_b_confirm and criterion_c_confirm:
        overall = "CONFIRMED"
    elif criterion_a_falsify or (criterion_b_falsify and criterion_c_falsify):
        overall = "FALSIFIED"
    else:
        overall = "INCONCLUSIVE"

    print(f"\n  " + "=" * 50)
    print(f"  PRE-REGISTERED VERDICT: {overall}")
    print(f"  " + "=" * 50)

    # Test 4 informational context
    t4 = all_results.get("test4", {})
    if t4.get("status") == "complete":
        agg = t4.get("aggregate", {})
        rfh = agg.get("R_full_hidden_vs_fisher", {})
        rfr = agg.get("R_full_raw_vs_fisher", {})
        print(f"\n  Test 4 (informational, supplements pre-registered criteria):")
        print(f"    R_full(hidden) vs Fisher: mean_rho = {rfh.get('mean_rho', 'N/A')}, "
              f"p = {rfh.get('t_p', 'N/A')}")
        print(f"    R_full(raw) vs Fisher:    mean_rho = {rfr.get('mean_rho', 'N/A')}, "
              f"p = {rfr.get('t_p', 'N/A')}")

    verdict_data["overall"] = overall
    verdict_data["test4_informational"] = t4.get("aggregate", {}) if t4.get("status") == "complete" else {}

    return verdict_data


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("Q15 v2 FIXED: Does R Have a Genuine Bayesian Interpretation?")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Architectures: {ARCHITECTURES}")
    print()

    all_results = {}

    # Load 20 Newsgroups text data ONCE
    texts, labels, categories = load_20newsgroups_base()

    # Pre-encode all architectures (cached to disk)
    print("\n--- Pre-encoding all architectures ---")
    for model_name in ARCHITECTURES:
        encode_with_cache(texts, model_name)
    print("--- All architectures encoded ---\n")

    # Test 1
    try:
        all_results["test1"] = test1_bayesian_correlation(texts, labels, categories)
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        traceback.print_exc()
        all_results["test1"] = {"status": "error", "error": str(e)}

    # Test 2
    try:
        all_results["test2"] = test2_intensive_property(texts, labels, categories)
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        traceback.print_exc()
        all_results["test2"] = {"status": "error", "error": str(e)}

    # Test 3
    try:
        all_results["test3"] = test3_gating_quality(texts, labels, categories)
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        traceback.print_exc()
        all_results["test3"] = {"status": "error", "error": str(e)}

    # Test 4
    try:
        all_results["test4"] = test4_neural_bayesian()
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        traceback.print_exc()
        all_results["test4"] = {"status": "error", "error": str(e)}

    # Verdict
    try:
        verdict_data = compute_verdict(all_results)
    except Exception as e:
        print(f"  VERDICT COMPUTATION FAILED: {e}")
        traceback.print_exc()
        verdict_data = {"overall": "ERROR", "error": str(e)}

    # Save results
    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_for_json(v) for v in obj]
        return obj

    results_path = os.path.join(RESULTS_DIR, "test_v2_q15_fixed_results.json")
    with open(results_path, "w") as f:
        json.dump(convert_for_json({
            "metadata": {
                "seed": SEED,
                "architectures": ARCHITECTURES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "tests": all_results,
            "verdict": verdict_data,
        }), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return all_results, verdict_data


if __name__ == "__main__":
    all_results, verdict_data = main()
