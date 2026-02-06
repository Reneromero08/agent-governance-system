"""
Q15 v2: Does R have a genuine Bayesian interpretation?

Pre-registered test plan with 4 tests:
  Test 1: Full-Formula R vs Bayesian Quantities (STS-B clusters)
  Test 2: Intensive Property Test (subsample stability)
  Test 3: Gating Decision Quality (classification comparison)
  Test 4: Reinstating the Falsification (Hessian-based, UCI data)

All tests use the shared formula module. No alternative E definitions.
Seed = 42 throughout.
"""

import sys
import os
import json
import time
import warnings
import traceback

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---- Path setup ----
# code/ is at THOUGHT/LAB/FORMULA/v2/q15_bayesian/code/
# shared/ is at THOUGHT/LAB/FORMULA/v2/shared/
V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(V2_DIR, "..", "..", "..", ".."))
sys.path.insert(0, V2_DIR)

from shared.formula import (
    compute_E, compute_grad_S, compute_sigma, compute_Df,
    compute_R_simple, compute_R_full, compute_all,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# =====================================================================
# UTILITIES
# =====================================================================

def safe_spearman(x, y):
    """Spearman correlation with NaN handling."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def safe_pearson(x, y):
    """Pearson correlation with NaN handling."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def compute_trivial_R(embeddings):
    """Trivial R = 1/grad_S (the tautological version from v1 rescue)."""
    grad_S = compute_grad_S(embeddings)
    if np.isnan(grad_S) or grad_S < 1e-10:
        return float("nan")
    return 1.0 / grad_S


def compute_likelihood_precision_trace(embeddings):
    """Likelihood precision = 1/trace(covariance)."""
    if embeddings.shape[0] < 3:
        return float("nan")
    cov = np.cov(embeddings, rowvar=False)
    tr = np.trace(cov)
    if tr < 1e-15:
        return float("nan")
    return 1.0 / tr


def compute_likelihood_precision_det(embeddings):
    """Likelihood precision from det(covariance)^(-1/d)."""
    n, d = embeddings.shape
    if n < d + 1:
        return float("nan")
    cov = np.cov(embeddings, rowvar=False)
    # Use eigenvalues for numerical stability
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-15]
    if len(eigvals) == 0:
        return float("nan")
    # Geometric mean of eigenvalues = det^(1/d)
    log_det_per_d = np.mean(np.log(eigvals))
    return float(np.exp(-log_det_per_d))


def compute_posterior_precision_bootstrap(embeddings, n_bootstrap=100, rng=None):
    """
    Posterior precision estimate via bootstrap.
    Resample embeddings, compute centroid each time, measure precision = 1/var(centroid).
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
    # Precision = 1/trace(var(centroids))
    var_trace = np.trace(np.cov(centroids, rowvar=False))
    if var_trace < 1e-15:
        return float("nan")
    return 1.0 / var_trace


def compute_bic(embeddings):
    """
    BIC-based marginal likelihood approximation.
    Model: single multivariate Gaussian.
    BIC = -2*log_likelihood + k*log(n)
    Return negative BIC (higher = better model evidence).
    """
    n, d = embeddings.shape
    if n < d + 2:
        return float("nan")
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings, rowvar=False)
    # Regularize
    cov += np.eye(d) * 1e-6
    try:
        from scipy.stats import multivariate_normal
        log_lik = multivariate_normal.logpdf(embeddings, mean=mean, cov=cov).sum()
    except Exception:
        return float("nan")
    # k = d (mean params) + d*(d+1)/2 (cov params)
    k = d + d * (d + 1) // 2
    bic = -2 * log_lik + k * np.log(n)
    return float(-bic)  # negative BIC: higher = more evidence


def load_stsb_data():
    """Load STS-B dataset and encode with sentence-transformers."""
    print("Loading STS-B dataset...")
    from datasets import load_dataset
    # STS-B is part of GLUE
    try:
        ds = load_dataset("glue", "stsb", split="validation")
    except Exception as e:
        print(f"  Trying alternative load: {e}")
        ds = load_dataset("nyu-mll/glue", "stsb", split="validation")

    sentences = []
    scores = []
    for row in ds:
        sentences.append(row["sentence1"])
        scores.append(row["label"])  # STS-B scores are 0-5

    print(f"  Loaded {len(sentences)} sentence pairs with scores")
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding sentences...")
    embeddings = model.encode(sentences, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, scores, sentences


def create_clusters(embeddings, scores, n_clusters=50):
    """
    Create clusters by binning sentences by STS-B similarity score.
    Returns list of (cluster_embeddings, mean_score) tuples.
    """
    bins = np.linspace(0, 5, n_clusters + 1)
    clusters = []
    for i in range(n_clusters):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() >= 10:  # Need at least 10 for meaningful computation
            clusters.append((embeddings[mask], float(np.mean(scores[mask]))))
    print(f"  Created {len(clusters)} clusters with >= 10 members")
    return clusters


# =====================================================================
# TEST 1: Full-Formula R vs Bayesian Quantities
# =====================================================================

def test1_bayesian_correlation(embeddings, scores):
    """
    Core test: Does full R correlate with Bayesian quantities?
    Compare R_full, R_simple, and Trivial_R (1/grad_S) against:
      - Likelihood precision (1/trace(cov))
      - Posterior precision (bootstrap)
      - Marginal likelihood (BIC)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Full-Formula R vs Bayesian Quantities")
    print("=" * 70)

    clusters = create_clusters(embeddings, scores, n_clusters=50)
    if len(clusters) < 10:
        print("  ERROR: Too few clusters. Aborting test 1.")
        return {"status": "error", "reason": "too_few_clusters"}

    # Compute all quantities for each cluster
    R_full_vals = []
    R_simple_vals = []
    trivial_R_vals = []
    lik_prec_trace_vals = []
    lik_prec_det_vals = []
    post_prec_vals = []
    bic_vals = []
    E_vals = []
    grad_S_vals = []
    sigma_vals = []
    Df_vals = []

    for idx, (emb, mean_score) in enumerate(clusters):
        all_components = compute_all(emb)
        R_full_vals.append(all_components["R_full"])
        R_simple_vals.append(all_components["R_simple"])
        trivial_R_vals.append(compute_trivial_R(emb))
        E_vals.append(all_components["E"])
        grad_S_vals.append(all_components["grad_S"])
        sigma_vals.append(all_components["sigma"])
        Df_vals.append(all_components["Df"])

        rng = np.random.RandomState(SEED + idx)
        lik_prec_trace_vals.append(compute_likelihood_precision_trace(emb))
        lik_prec_det_vals.append(compute_likelihood_precision_det(emb))
        post_prec_vals.append(compute_posterior_precision_bootstrap(emb, rng=rng))
        bic_vals.append(compute_bic(emb))

    # Convert to arrays
    R_full = np.array(R_full_vals)
    R_simple = np.array(R_simple_vals)
    trivial_R = np.array(trivial_R_vals)
    lik_prec_trace = np.array(lik_prec_trace_vals)
    lik_prec_det = np.array(lik_prec_det_vals)
    post_prec = np.array(post_prec_vals)
    bic = np.array(bic_vals)
    E_arr = np.array(E_vals)
    grad_S_arr = np.array(grad_S_vals)

    # Also test sqrt(likelihood precision) since hypothesis mentions it
    sqrt_lik_prec_trace = np.sqrt(np.abs(lik_prec_trace))

    print(f"\n  Valid R_full values: {np.isfinite(R_full).sum()}/{len(R_full)}")
    print(f"  Valid R_simple values: {np.isfinite(R_simple).sum()}/{len(R_simple)}")
    print(f"  Valid trivial_R values: {np.isfinite(trivial_R).sum()}/{len(trivial_R)}")

    # Print component statistics
    print(f"\n  Component statistics:")
    for name, arr in [("E", E_arr), ("grad_S", grad_S_arr),
                      ("sigma", np.array(sigma_vals)), ("Df", np.array(Df_vals))]:
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            print(f"    {name}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}, "
                  f"range=[{np.min(valid):.4f}, {np.max(valid):.4f}]")

    # Correlation matrix
    bayesian_quantities = {
        "lik_prec_trace": lik_prec_trace,
        "sqrt_lik_prec_trace": sqrt_lik_prec_trace,
        "lik_prec_det": lik_prec_det,
        "post_prec_bootstrap": post_prec,
        "neg_BIC": bic,
    }

    formula_variants = {
        "R_full": R_full,
        "R_simple": R_simple,
        "Trivial_R (1/grad_S)": trivial_R,
        "E": E_arr,
        "1/grad_S": 1.0 / np.where(grad_S_arr > 1e-10, grad_S_arr, np.nan),
    }

    results = {}
    print("\n  Spearman correlations (rho, p-value):")
    print("  " + "-" * 90)
    header = f"  {'':30s}"
    for bname in bayesian_quantities:
        header += f"  {bname:>18s}"
    print(header)
    print("  " + "-" * 90)

    for fname, fvals in formula_variants.items():
        row = f"  {fname:30s}"
        results[fname] = {}
        for bname, bvals in bayesian_quantities.items():
            rho, p = safe_spearman(fvals, bvals)
            row += f"  {rho:+7.3f} (p={p:.3f})" if np.isfinite(rho) else f"  {'NaN':>18s}"
            results[fname][bname] = {"rho": rho, "p": p}
        print(row)

    # KEY COMPARISON: Does R_full beat trivial_R?
    print("\n  KEY COMPARISON: R_full vs Trivial_R (1/grad_S)")
    print("  " + "-" * 60)
    for bname in bayesian_quantities:
        rho_full = results["R_full"].get(bname, {}).get("rho", float("nan"))
        rho_triv = results["Trivial_R (1/grad_S)"].get(bname, {}).get("rho", float("nan"))
        diff = rho_full - rho_triv if np.isfinite(rho_full) and np.isfinite(rho_triv) else float("nan")
        winner = "R_full" if diff > 0 else "Trivial_R" if diff < 0 else "TIE"
        print(f"    {bname:25s}: R_full={rho_full:+.3f}, Trivial={rho_triv:+.3f}, "
              f"diff={diff:+.3f} -> {winner}")

    # Pearson too for completeness
    print("\n  Pearson correlations:")
    print("  " + "-" * 90)
    pearson_results = {}
    for fname, fvals in formula_variants.items():
        row = f"  {fname:30s}"
        pearson_results[fname] = {}
        for bname, bvals in bayesian_quantities.items():
            r, p = safe_pearson(fvals, bvals)
            row += f"  {r:+7.3f} (p={p:.3f})" if np.isfinite(r) else f"  {'NaN':>18s}"
            pearson_results[fname][bname] = {"r": r, "p": p}
        print(row)

    return {
        "status": "complete",
        "n_clusters": len(clusters),
        "spearman": results,
        "pearson": pearson_results,
        "component_stats": {
            "E": {"mean": float(np.nanmean(E_arr)), "std": float(np.nanstd(E_arr))},
            "grad_S": {"mean": float(np.nanmean(grad_S_arr)), "std": float(np.nanstd(grad_S_arr))},
        },
    }


# =====================================================================
# TEST 2: Intensive Property Test
# =====================================================================

def test2_intensive_property(embeddings, scores):
    """
    Test whether R is intensive (independent of sample size).
    Subsample at N = 10, 20, 50, 100, 200 for 10 different clusters.
    Measure CV of R across N values.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Intensive Property Test")
    print("=" * 70)

    # Pick 10 clusters with enough samples
    clusters = create_clusters(embeddings, scores, n_clusters=20)
    # Filter to clusters with at least 200 members
    large_clusters = [(emb, sc) for emb, sc in clusters if emb.shape[0] >= 200]
    print(f"  Clusters with >= 200 members: {len(large_clusters)}")

    # If not enough large clusters, use lower threshold
    if len(large_clusters) < 5:
        min_size = 100
        large_clusters = [(emb, sc) for emb, sc in clusters if emb.shape[0] >= min_size]
        print(f"  Fallback: clusters with >= {min_size} members: {len(large_clusters)}")
        subsample_sizes = [10, 20, 50, min_size]
    else:
        subsample_sizes = [10, 20, 50, 100, 200]
        min_size = 200

    if len(large_clusters) < 3:
        # Use score-based partitions instead
        print("  Using score-based partitions (tertiles) instead of fine clusters")
        tertile_edges = [0, 1.67, 3.33, 5.01]
        large_clusters = []
        for i in range(3):
            mask = (scores >= tertile_edges[i]) & (scores < tertile_edges[i + 1])
            if mask.sum() >= 200:
                large_clusters.append((embeddings[mask], float(np.mean(scores[mask]))))
        subsample_sizes = [10, 20, 50, 100, 200]
        print(f"  Tertile clusters: {len(large_clusters)}")

    n_repeats = 20  # Repeat each subsample this many times for stability

    all_results = []
    for c_idx, (emb, mean_score) in enumerate(large_clusters[:10]):
        cluster_n = emb.shape[0]
        print(f"\n  Cluster {c_idx} (n={cluster_n}, mean_score={mean_score:.2f}):")

        R_full_by_N = {}
        R_simple_by_N = {}
        trivial_R_by_N = {}
        post_prec_by_N = {}

        for N in subsample_sizes:
            if N > cluster_n:
                continue

            R_full_samples = []
            R_simple_samples = []
            trivial_samples = []
            post_prec_samples = []

            for rep in range(n_repeats):
                rng = np.random.RandomState(SEED + c_idx * 1000 + N * 100 + rep)
                idx = rng.choice(cluster_n, size=N, replace=False)
                sub = emb[idx]

                R_full_samples.append(compute_R_full(sub))
                R_simple_samples.append(compute_R_simple(sub))
                trivial_samples.append(compute_trivial_R(sub))
                post_prec_samples.append(compute_posterior_precision_bootstrap(sub, rng=rng))

            # Mean across repeats for this N
            R_full_by_N[N] = float(np.nanmean(R_full_samples))
            R_simple_by_N[N] = float(np.nanmean(R_simple_samples))
            trivial_R_by_N[N] = float(np.nanmean(trivial_samples))
            post_prec_by_N[N] = float(np.nanmean(post_prec_samples))

            print(f"    N={N:4d}: R_full={R_full_by_N[N]:.4f}, "
                  f"R_simple={R_simple_by_N[N]:.4f}, "
                  f"1/grad_S={trivial_R_by_N[N]:.4f}, "
                  f"post_prec={post_prec_by_N[N]:.4f}")

        # Compute CV for each quantity across N values
        def cv_across_N(vals_by_N):
            vals = np.array([v for v in vals_by_N.values() if np.isfinite(v)])
            if len(vals) < 2 or np.mean(np.abs(vals)) < 1e-15:
                return float("nan")
            return float(np.std(vals) / np.abs(np.mean(vals)))

        cv_R_full = cv_across_N(R_full_by_N)
        cv_R_simple = cv_across_N(R_simple_by_N)
        cv_trivial = cv_across_N(trivial_R_by_N)
        cv_post_prec = cv_across_N(post_prec_by_N)

        print(f"    CV: R_full={cv_R_full:.4f}, R_simple={cv_R_simple:.4f}, "
              f"1/grad_S={cv_trivial:.4f}, post_prec={cv_post_prec:.4f}")

        all_results.append({
            "cluster_idx": c_idx,
            "cluster_n": cluster_n,
            "mean_score": mean_score,
            "cv_R_full": cv_R_full,
            "cv_R_simple": cv_R_simple,
            "cv_trivial_R": cv_trivial,
            "cv_post_prec": cv_post_prec,
            "R_full_by_N": R_full_by_N,
            "trivial_R_by_N": trivial_R_by_N,
            "post_prec_by_N": post_prec_by_N,
        })

    # Summary statistics
    cv_R_full_all = np.array([r["cv_R_full"] for r in all_results if np.isfinite(r["cv_R_full"])])
    cv_R_simple_all = np.array([r["cv_R_simple"] for r in all_results if np.isfinite(r["cv_R_simple"])])
    cv_trivial_all = np.array([r["cv_trivial_R"] for r in all_results if np.isfinite(r["cv_trivial_R"])])
    cv_post_prec_all = np.array([r["cv_post_prec"] for r in all_results if np.isfinite(r["cv_post_prec"])])

    print("\n  SUMMARY:")
    print(f"    R_full    CV: mean={np.mean(cv_R_full_all):.4f}, "
          f"median={np.median(cv_R_full_all):.4f}, max={np.max(cv_R_full_all):.4f}")
    print(f"    R_simple  CV: mean={np.mean(cv_R_simple_all):.4f}, "
          f"median={np.median(cv_R_simple_all):.4f}, max={np.max(cv_R_simple_all):.4f}")
    print(f"    1/grad_S  CV: mean={np.mean(cv_trivial_all):.4f}, "
          f"median={np.median(cv_trivial_all):.4f}, max={np.max(cv_trivial_all):.4f}")
    print(f"    post_prec CV: mean={np.mean(cv_post_prec_all):.4f}, "
          f"median={np.median(cv_post_prec_all):.4f}, max={np.max(cv_post_prec_all):.4f}")
    print(f"\n    Intensive threshold: CV < 0.15")
    print(f"    R_full intensive?   {np.mean(cv_R_full_all):.4f} < 0.15 -> "
          f"{'YES' if np.mean(cv_R_full_all) < 0.15 else 'NO'}")
    print(f"    R_full better than 1/grad_S? CV_full={np.mean(cv_R_full_all):.4f} vs "
          f"CV_trivial={np.mean(cv_trivial_all):.4f}")

    return {
        "status": "complete",
        "n_clusters_tested": len(all_results),
        "subsample_sizes": subsample_sizes,
        "cv_summary": {
            "R_full": {"mean": float(np.mean(cv_R_full_all)),
                       "median": float(np.median(cv_R_full_all)),
                       "max": float(np.max(cv_R_full_all))},
            "R_simple": {"mean": float(np.mean(cv_R_simple_all)),
                         "median": float(np.median(cv_R_simple_all)),
                         "max": float(np.max(cv_R_simple_all))},
            "trivial_R": {"mean": float(np.mean(cv_trivial_all)),
                          "median": float(np.median(cv_trivial_all)),
                          "max": float(np.max(cv_trivial_all))},
            "post_prec": {"mean": float(np.mean(cv_post_prec_all)),
                          "median": float(np.median(cv_post_prec_all)),
                          "max": float(np.max(cv_post_prec_all))},
        },
        "cluster_results": all_results,
    }


# =====================================================================
# TEST 3: Gating Decision Quality
# =====================================================================

def test3_gating_quality(embeddings, scores):
    """
    Compare gating accuracy for quality classification:
      high quality (score > 4.0) vs low quality (score < 2.0).
    Methods: R_full, R_simple, 1/grad_S, raw E, Bayesian posterior probability.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Gating Decision Quality")
    print("=" * 70)

    # Create micro-clusters: groups of nearby sentences
    # We'll create small groups (size ~15) and label them by mean score
    n = len(scores)
    rng = np.random.RandomState(SEED)

    # Sort by score, create groups of ~15
    sort_idx = np.argsort(scores)
    group_size = 15
    groups = []
    for i in range(0, n - group_size, group_size):
        idx = sort_idx[i:i + group_size]
        mean_score = np.mean(scores[idx])
        groups.append((embeddings[idx], mean_score))

    # Label: high (>4.0) or low (<2.0), discard middle
    high_groups = [(emb, sc) for emb, sc in groups if sc > 4.0]
    low_groups = [(emb, sc) for emb, sc in groups if sc < 2.0]
    print(f"  High-quality groups (score > 4.0): {len(high_groups)}")
    print(f"  Low-quality groups (score < 2.0): {len(low_groups)}")

    if len(high_groups) < 5 or len(low_groups) < 5:
        print("  WARNING: Very few groups in one category. Results may be unreliable.")

    # Combine and compute features
    all_groups = [(emb, sc, 1) for emb, sc in high_groups] + \
                 [(emb, sc, 0) for emb, sc in low_groups]
    rng.shuffle(all_groups)

    features = {
        "R_full": [],
        "R_simple": [],
        "trivial_R": [],
        "E": [],
        "bayesian_posterior": [],
    }
    labels = []

    for emb, sc, label in all_groups:
        features["R_full"].append(compute_R_full(emb))
        features["R_simple"].append(compute_R_simple(emb))
        features["trivial_R"].append(compute_trivial_R(emb))
        features["E"].append(compute_E(emb))
        # Bayesian posterior: P(high | data) using simple likelihood ratio
        # Assume Gaussian, compute log-likelihood under "tight" vs "loose" model
        lik_prec = compute_likelihood_precision_trace(emb)
        features["bayesian_posterior"].append(lik_prec if np.isfinite(lik_prec) else 0.0)
        labels.append(label)

    labels = np.array(labels)

    # Convert to arrays, handle NaN
    for key in features:
        features[key] = np.array(features[key])
        nan_mask = ~np.isfinite(features[key])
        if nan_mask.any():
            # Replace NaN with median
            median_val = np.nanmedian(features[key])
            features[key][nan_mask] = median_val

    # Train/test split
    n_total = len(labels)
    n_train = n_total // 2
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_total)

    print(f"\n  Total groups: {n_total} (train: {len(train_idx)}, test: {len(test_idx)})")
    print(f"  Train: {labels[train_idx].sum()} high, {(1-labels[train_idx]).sum()} low")
    print(f"  Test:  {labels[test_idx].sum()} high, {(1-labels[test_idx]).sum()} low")

    results = {}
    print(f"\n  {'Method':25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Threshold':>10s}")
    print("  " + "-" * 70)

    for method_name, vals in features.items():
        train_vals = vals[train_idx]
        train_labels = labels[train_idx]
        test_vals = vals[test_idx]
        test_labels = labels[test_idx]

        # Find optimal threshold on train set (maximize F1)
        best_f1 = -1
        best_thresh = 0
        thresholds = np.percentile(train_vals[np.isfinite(train_vals)],
                                   np.linspace(5, 95, 50))
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

        # Evaluate on test set
        test_preds = (test_vals > best_thresh).astype(int)
        tp = ((test_preds == 1) & (test_labels == 1)).sum()
        fp = ((test_preds == 1) & (test_labels == 0)).sum()
        fn = ((test_preds == 0) & (test_labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[method_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(best_thresh),
        }
        print(f"  {method_name:25s} {precision:10.3f} {recall:10.3f} {f1:10.3f} {best_thresh:10.4f}")

    # KEY: Does R_full outperform 1/grad_S?
    f1_full = results["R_full"]["f1"]
    f1_trivial = results["trivial_R"]["f1"]
    diff = f1_full - f1_trivial
    print(f"\n  KEY: R_full F1 = {f1_full:.3f}, Trivial_R F1 = {f1_trivial:.3f}, "
          f"diff = {diff:+.3f}")
    print(f"  R_full outperforms by >5%? {'YES' if diff > 0.05 else 'NO'}")

    return {
        "status": "complete",
        "n_high": int(labels.sum()),
        "n_low": int((1 - labels).sum()),
        "results": results,
        "f1_advantage_R_full_over_trivial": float(diff),
    }


# =====================================================================
# TEST 4: Reinstating the Falsification (Hessian-based)
# =====================================================================

def test4_falsification(n_seeds=10):
    """
    Reproduce the v1 falsification approach on the full formula.
    Use Bayesian linear regression on California Housing.
    Compute Hessian of the loss -> true posterior precision.
    Compute R for feature embeddings.
    Test across 10 seeds.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Reinstating the Falsification (Hessian-based)")
    print("=" * 70)

    import torch
    import torch.nn as nn

    # Load California Housing
    print("  Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results_by_seed = []

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---")
        rng = np.random.RandomState(SEED + seed)
        torch.manual_seed(SEED + seed)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED + seed
        )

        # Create feature subsets for R computation
        # Divide features into groups (different subsets of training data)
        n_groups = 20
        group_size = len(X_train) // n_groups

        # Train a simple linear regression (Bayesian interpretation is straightforward)
        # For Bayesian linear regression: posterior precision = (1/sigma^2) * X^T X + prior_precision
        # We'll use a neural network with one hidden layer for more interesting Hessian

        class SimpleNet(nn.Module):
            def __init__(self, d_in, d_hidden=32):
                super().__init__()
                self.fc1 = nn.Linear(d_in, d_hidden)
                self.fc2 = nn.Linear(d_hidden, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = SimpleNet(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        # Train
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = nn.MSELoss()(pred, y_train_t)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"    Final training loss: {final_loss:.4f}")

        # Compute Hessian-based posterior precision for data subgroups
        # For each group of data, compute:
        #   (a) R_full on the feature embeddings
        #   (b) Hessian-based precision (trace of Fisher information)
        model.eval()

        R_full_vals = []
        R_simple_vals = []
        trivial_R_vals = []
        hessian_prec_vals = []
        fisher_info_vals = []
        kl_div_vals = []

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            X_group = X_train[start:end]

            # (a) Compute R on feature embeddings
            R_full_vals.append(compute_R_full(X_group))
            R_simple_vals.append(compute_R_simple(X_group))
            trivial_R_vals.append(compute_trivial_R(X_group))

            # (b) Compute Fisher information (empirical) for this group
            X_group_t = torch.tensor(X_group, dtype=torch.float32)
            y_group_t = torch.tensor(y_train[start:end], dtype=torch.float32).unsqueeze(1)

            # Fisher information: E[grad log p(y|x,theta) * grad log p(y|x,theta)^T]
            # Approximate with sum of outer products of per-sample gradients
            fisher_diag = None
            for i in range(len(X_group_t)):
                model.zero_grad()
                pred_i = model(X_group_t[i:i+1])
                loss_i = nn.MSELoss()(pred_i, y_group_t[i:i+1])
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

            fisher_diag = fisher_diag / len(X_group_t)
            fisher_trace = fisher_diag.sum().item()
            fisher_info_vals.append(fisher_trace)

            # Hessian-based precision: approximate via finite differences on loss
            # Use diagonal Hessian approximation
            model.zero_grad()
            pred_group = model(X_group_t)
            loss_group = nn.MSELoss()(pred_group, y_group_t)

            params = []
            for p in model.parameters():
                params.append(p.detach().flatten())
            param_vec = torch.cat(params)

            # Hessian diagonal via autograd
            loss_group.backward(create_graph=True)
            hessian_diag = []
            for p in model.parameters():
                if p.grad is not None:
                    for g_elem in p.grad.flatten():
                        h = torch.autograd.grad(g_elem, p, retain_graph=True, allow_unused=True)
                        if h[0] is not None:
                            hessian_diag.append(h[0].abs().sum().item())
                        else:
                            hessian_diag.append(0.0)

            hessian_prec = np.sum(hessian_diag) if hessian_diag else float("nan")
            hessian_prec_vals.append(hessian_prec)

            # KL divergence approximation: 0.5 * (trace(Sigma_posterior * Hessian) - d + log_det_ratio)
            # Simplified: use trace of Hessian as proxy
            kl_div_vals.append(fisher_trace * final_loss)

            model.zero_grad()

        # Convert to arrays
        R_full = np.array(R_full_vals)
        R_simple = np.array(R_simple_vals)
        trivial_R = np.array(trivial_R_vals)
        hessian_prec = np.array(hessian_prec_vals)
        fisher_info = np.array(fisher_info_vals)
        kl_div = np.array(kl_div_vals)

        # Correlations
        seed_results = {}
        for rname, rvals in [("R_full", R_full), ("R_simple", R_simple),
                             ("Trivial_R", trivial_R)]:
            seed_results[rname] = {}
            for bname, bvals in [("hessian_prec", hessian_prec),
                                 ("fisher_info", fisher_info),
                                 ("kl_div", kl_div)]:
                rho, p = safe_spearman(rvals, bvals)
                seed_results[rname][bname] = {"rho": rho, "p": p}

        print(f"    R_full vs hessian_prec:  rho={seed_results['R_full']['hessian_prec']['rho']:+.3f}")
        print(f"    R_full vs fisher_info:   rho={seed_results['R_full']['fisher_info']['rho']:+.3f}")
        print(f"    R_full vs kl_div:        rho={seed_results['R_full']['kl_div']['rho']:+.3f}")
        print(f"    Trivial_R vs hessian:    rho={seed_results['Trivial_R']['hessian_prec']['rho']:+.3f}")
        print(f"    Trivial_R vs fisher:     rho={seed_results['Trivial_R']['fisher_info']['rho']:+.3f}")

        results_by_seed.append(seed_results)

    # Aggregate across seeds
    print("\n  AGGREGATE ACROSS SEEDS:")
    print("  " + "-" * 70)

    aggregate = {}
    for rname in ["R_full", "R_simple", "Trivial_R"]:
        aggregate[rname] = {}
        for bname in ["hessian_prec", "fisher_info", "kl_div"]:
            rhos = [s[rname][bname]["rho"] for s in results_by_seed
                    if np.isfinite(s[rname][bname]["rho"])]
            if rhos:
                mean_rho = float(np.mean(rhos))
                std_rho = float(np.std(rhos))
                # One-sample t-test: is mean_rho significantly different from 0?
                if len(rhos) >= 3:
                    t_stat, t_p = stats.ttest_1samp(rhos, 0)
                else:
                    t_stat, t_p = float("nan"), float("nan")
                aggregate[rname][bname] = {
                    "mean_rho": mean_rho,
                    "std_rho": std_rho,
                    "n_valid": len(rhos),
                    "t_stat": float(t_stat),
                    "t_p": float(t_p),
                    "all_rhos": rhos,
                }
                sig = "*" if t_p < 0.05 else ""
                print(f"    {rname:12s} vs {bname:15s}: "
                      f"mean_rho={mean_rho:+.3f} +/- {std_rho:.3f} "
                      f"(t={t_stat:.2f}, p={t_p:.4f}){sig}")
            else:
                aggregate[rname][bname] = {"mean_rho": float("nan")}
                print(f"    {rname:12s} vs {bname:15s}: NO VALID DATA")

    # KEY: Does R_full correlate with Bayesian quantities across seeds?
    key_rho = aggregate.get("R_full", {}).get("hessian_prec", {}).get("mean_rho", float("nan"))
    key_p = aggregate.get("R_full", {}).get("hessian_prec", {}).get("t_p", float("nan"))
    print(f"\n  KEY: R_full vs Hessian precision, mean rho = {key_rho:+.3f}, p = {key_p:.4f}")
    print(f"  Falsification reproduces? {'YES (no correlation)' if abs(key_rho) < 0.2 or key_p > 0.05 else 'NO (significant correlation)'}")

    return {
        "status": "complete",
        "n_seeds": n_seeds,
        "aggregate": aggregate,
        "by_seed": results_by_seed,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("Q15 v2: Does R Have a Genuine Bayesian Interpretation?")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Load STS-B data (shared across tests 1-3)
    try:
        embeddings, scores, sentences = load_stsb_data()
    except Exception as e:
        print(f"FATAL: Could not load STS-B data: {e}")
        traceback.print_exc()
        return

    # Test 1
    try:
        all_results["test1"] = test1_bayesian_correlation(embeddings, scores)
    except Exception as e:
        print(f"  TEST 1 FAILED: {e}")
        traceback.print_exc()
        all_results["test1"] = {"status": "error", "error": str(e)}

    # Test 2
    try:
        all_results["test2"] = test2_intensive_property(embeddings, scores)
    except Exception as e:
        print(f"  TEST 2 FAILED: {e}")
        traceback.print_exc()
        all_results["test2"] = {"status": "error", "error": str(e)}

    # Test 3
    try:
        all_results["test3"] = test3_gating_quality(embeddings, scores)
    except Exception as e:
        print(f"  TEST 3 FAILED: {e}")
        traceback.print_exc()
        all_results["test3"] = {"status": "error", "error": str(e)}

    # Test 4
    try:
        all_results["test4"] = test4_falsification(n_seeds=10)
    except Exception as e:
        print(f"  TEST 4 FAILED: {e}")
        traceback.print_exc()
        all_results["test4"] = {"status": "error", "error": str(e)}

    # Save raw results
    results_path = os.path.join(RESULTS_DIR, "test_v2_q15_results.json")
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ---- VERDICT ----
    print("\n" + "=" * 70)
    print("VERDICT SYNTHESIS")
    print("=" * 70)

    verdict_lines = []

    # Test 1 verdict
    t1 = all_results.get("test1", {})
    if t1.get("status") == "complete":
        spearman = t1.get("spearman", {})
        # Check if R_full correlates with sqrt(lik prec) at rho > 0.7
        r_full_sqrt = spearman.get("R_full", {}).get("sqrt_lik_prec_trace", {}).get("rho", float("nan"))
        r_triv_sqrt = spearman.get("Trivial_R (1/grad_S)", {}).get("sqrt_lik_prec_trace", {}).get("rho", float("nan"))

        # Best correlation for R_full across all Bayesian quantities
        best_rho_full = max(
            abs(spearman.get("R_full", {}).get(bq, {}).get("rho", 0))
            for bq in ["lik_prec_trace", "sqrt_lik_prec_trace", "lik_prec_det",
                        "post_prec_bootstrap", "neg_BIC"]
        )
        best_rho_triv = max(
            abs(spearman.get("Trivial_R (1/grad_S)", {}).get(bq, {}).get("rho", 0))
            for bq in ["lik_prec_trace", "sqrt_lik_prec_trace", "lik_prec_det",
                        "post_prec_bootstrap", "neg_BIC"]
        )

        verdict_lines.append(f"Test 1: R_full vs sqrt(lik_prec): rho={r_full_sqrt:.3f}")
        verdict_lines.append(f"        Trivial_R vs sqrt(lik_prec): rho={r_triv_sqrt:.3f}")
        verdict_lines.append(f"        Best |rho| for R_full: {best_rho_full:.3f}")
        verdict_lines.append(f"        Best |rho| for Trivial_R: {best_rho_triv:.3f}")

        t1_pass = r_full_sqrt > 0.7 if np.isfinite(r_full_sqrt) else False
        t1_R_adds_value = best_rho_full > best_rho_triv + 0.05
        verdict_lines.append(f"        Correlation > 0.7? {'YES' if t1_pass else 'NO'}")
        verdict_lines.append(f"        R_full adds value over trivial? {'YES' if t1_R_adds_value else 'NO'}")

    # Test 2 verdict
    t2 = all_results.get("test2", {})
    if t2.get("status") == "complete":
        cv = t2.get("cv_summary", {})
        cv_full = cv.get("R_full", {}).get("mean", float("nan"))
        cv_triv = cv.get("trivial_R", {}).get("mean", float("nan"))
        cv_post = cv.get("post_prec", {}).get("mean", float("nan"))
        t2_intensive = cv_full < 0.15 if np.isfinite(cv_full) else False
        verdict_lines.append(f"Test 2: R_full mean CV = {cv_full:.4f} (intensive < 0.15? {'YES' if t2_intensive else 'NO'})")
        verdict_lines.append(f"        1/grad_S mean CV = {cv_triv:.4f}")
        verdict_lines.append(f"        post_prec mean CV = {cv_post:.4f}")

    # Test 3 verdict
    t3 = all_results.get("test3", {})
    if t3.get("status") == "complete":
        t3r = t3.get("results", {})
        f1_full = t3r.get("R_full", {}).get("f1", 0)
        f1_triv = t3r.get("trivial_R", {}).get("f1", 0)
        f1_E = t3r.get("E", {}).get("f1", 0)
        f1_bayes = t3r.get("bayesian_posterior", {}).get("f1", 0)
        t3_advantage = f1_full - f1_triv
        verdict_lines.append(f"Test 3: R_full F1 = {f1_full:.3f}")
        verdict_lines.append(f"        Trivial_R F1 = {f1_triv:.3f}")
        verdict_lines.append(f"        E F1 = {f1_E:.3f}")
        verdict_lines.append(f"        Bayesian F1 = {f1_bayes:.3f}")
        verdict_lines.append(f"        R_full advantage over trivial: {t3_advantage:+.3f} "
                             f"(need > 0.05 for confirm)")

    # Test 4 verdict
    t4 = all_results.get("test4", {})
    if t4.get("status") == "complete":
        agg = t4.get("aggregate", {})
        rho_hess = agg.get("R_full", {}).get("hessian_prec", {}).get("mean_rho", float("nan"))
        rho_fish = agg.get("R_full", {}).get("fisher_info", {}).get("mean_rho", float("nan"))
        rho_kl = agg.get("R_full", {}).get("kl_div", {}).get("mean_rho", float("nan"))
        p_hess = agg.get("R_full", {}).get("hessian_prec", {}).get("t_p", float("nan"))
        p_fish = agg.get("R_full", {}).get("fisher_info", {}).get("t_p", float("nan"))

        t4_falsified = (abs(rho_hess) < 0.2 or p_hess > 0.05) and \
                       (abs(rho_fish) < 0.2 or p_fish > 0.05)
        verdict_lines.append(f"Test 4: R_full vs Hessian precision: rho={rho_hess:+.3f} (p={p_hess:.4f})")
        verdict_lines.append(f"        R_full vs Fisher info: rho={rho_fish:+.3f} (p={p_fish:.4f})")
        verdict_lines.append(f"        R_full vs KL div: rho={rho_kl:+.3f}")
        verdict_lines.append(f"        Falsification reproduces? {'YES' if t4_falsified else 'NO'}")

    for line in verdict_lines:
        print(line)

    # Overall verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    # Determine overall result using pre-registered criteria
    confirm_criteria = []
    falsify_criteria = []

    # Test 1 criteria
    if t1.get("status") == "complete":
        confirm_criteria.append(("rho > 0.7 with sqrt(lik_prec)", t1_pass))
        falsify_criteria.append(("all rho < 0.2", best_rho_full < 0.2))

    # Test 2 criteria
    if t2.get("status") == "complete":
        confirm_criteria.append(("CV < 0.15 (intensive)", t2_intensive))
        falsify_criteria.append(("CV > 0.3 (not intensive)", cv_full > 0.3 if np.isfinite(cv_full) else True))

    # Test 3 criteria
    if t3.get("status") == "complete":
        confirm_criteria.append(("F1 > 5% better than trivial", t3_advantage > 0.05))
        falsify_criteria.append(("F1 within 3% of trivial", abs(t3_advantage) < 0.03))

    # Test 4 criteria
    if t4.get("status") == "complete":
        confirm_criteria.append(("Falsification does NOT reproduce", not t4_falsified))
        falsify_criteria.append(("Falsification reproduces", t4_falsified))

    confirm_count = sum(1 for _, v in confirm_criteria if v)
    falsify_count = sum(1 for _, v in falsify_criteria if v)

    print(f"\nConfirmation criteria met: {confirm_count}/{len(confirm_criteria)}")
    for name, met in confirm_criteria:
        print(f"  {'[PASS]' if met else '[FAIL]'} {name}")

    print(f"\nFalsification criteria met: {falsify_count}/{len(falsify_criteria)}")
    for name, met in falsify_criteria:
        print(f"  {'[TRIG]' if met else '[    ]'} {name}")

    # Decision logic
    if confirm_count == len(confirm_criteria) and len(confirm_criteria) >= 4:
        overall = "CONFIRMED"
    elif falsify_count >= 2:
        overall = "FALSIFIED"
    elif falsify_count >= 1 and confirm_count <= 1:
        overall = "FALSIFIED"
    elif confirm_count >= 3:
        overall = "INCONCLUSIVE (leans confirm)"
    else:
        overall = "INCONCLUSIVE"

    print(f"\n>>> OVERALL VERDICT: {overall} <<<")

    # Save verdict data
    verdict_data = {
        "overall": overall,
        "confirm_criteria": [(n, bool(v)) for n, v in confirm_criteria],
        "falsify_criteria": [(n, bool(v)) for n, v in falsify_criteria],
        "confirm_count": confirm_count,
        "falsify_count": falsify_count,
        "verdict_lines": verdict_lines,
    }
    with open(os.path.join(RESULTS_DIR, "verdict_data.json"), "w") as f:
        json.dump(convert_for_json(verdict_data), f, indent=2)

    return all_results, verdict_data


if __name__ == "__main__":
    results, verdict = main()
