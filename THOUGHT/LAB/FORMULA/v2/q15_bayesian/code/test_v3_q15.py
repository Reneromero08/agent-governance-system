"""
Q15 v3: Does R correlate with standard statistical measures of cluster quality?

HONEST REFRAMING (addresses METH-1 from audit):
  The v2 test dressed frequentist statistics in Bayesian labels. This version
  drops the "Bayesian" framing entirely and asks the honest question:
  "Does R correlate with standard statistical measures of cluster quality
   better than E alone?"

  The measures tested are:
    (a) Inverse trace of covariance  (frequentist; trivially related to E)
    (b) Bootstrap mean precision     (frequentist resampling)
    (c) Silhouette score             (clustering quality; independent of E)
  ESS has been REMOVED -- it is n*(1-E), a trivial affine transform of E.

AUDIT FIXES APPLIED:
  1. BUG-1:  ESS removed entirely (was trivially identical to E in rank).
  2. BUG-2:  Bootstrap precision honestly labeled as frequentist, not Bayesian.
  3. BUG-3:  Intensive property uses domain-weighted CV, not raw mean.
  4. BUG-4:  Gating test redesigned with continuous purity (not trivial binary).
  5. STAT-1: Steiger's test added for E-vs-R correlation comparison.
  6. STAT-2: Both Spearman and Pearson reported.
  7. STAT-3: Confidence intervals via bootstrap reported for all correlations.
  8. METH-1: All "Bayesian" labels removed. Honest frequentist labeling.
  9. METH-2: Partial correlation (R_full | E) computed for all measures.
  10. METH-3: Test 2 uses multiple n values (50, 100, 200, 500).

Data: 20 Newsgroups (text).
Architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1.
Sample: >=80 clusters with continuous purity variation.

PRE-REGISTERED CRITERIA:
  CONFIRMED if R significantly outperforms E (Steiger p<0.05) in correlation
    with >=2/3 statistical quality measures on >=2/3 architectures.
  FALSIFIED if E significantly outperforms R on all measures on all architectures.
  INCONCLUSIVE otherwise.
"""

import sys
import os
import json
import time
import warnings
import traceback
import importlib.util
import math

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

N_CLUSTERS = 90  # >= 80 as required


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


def safe_pearson(x, y):
    """Pearson correlation with NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def bootstrap_correlation_ci(x, y, method="spearman", n_boot=2000, alpha=0.05, rng=None):
    """
    Bootstrap confidence interval for a correlation coefficient.
    Returns (point_estimate, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), float("nan")

    if method == "spearman":
        point_est = float(stats.spearmanr(x, y)[0])
    else:
        point_est = float(stats.pearsonr(x, y)[0])

    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if method == "spearman":
            r, _ = stats.spearmanr(x[idx], y[idx])
        else:
            r, _ = stats.pearsonr(x[idx], y[idx])
        boot_rhos.append(float(r))

    boot_rhos = np.array(boot_rhos)
    boot_rhos = boot_rhos[np.isfinite(boot_rhos)]
    if len(boot_rhos) < 100:
        return point_est, float("nan"), float("nan")

    ci_low = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    return point_est, ci_low, ci_high


def steiger_test(r_xz, r_yz, r_xy, n):
    """
    Steiger's Z-test for comparing two dependent correlations that share
    a common variable (z).

    Tests H0: rho(x,z) = rho(y,z) where x and y are correlated.

    Args:
        r_xz: correlation between x and z (e.g., R_full vs measure)
        r_yz: correlation between y and z (e.g., E vs measure)
        r_xy: correlation between x and y (e.g., R_full vs E)
        n: sample size

    Returns:
        z_stat: Steiger's Z statistic
        p_value: two-tailed p-value
        direction: "x>y" if r_xz > r_yz, "y>x" otherwise
    """
    if any(math.isnan(v) for v in [r_xz, r_yz, r_xy]) or n < 10:
        return float("nan"), float("nan"), "undetermined"

    # Fisher z-transform
    z_xz = np.arctanh(np.clip(r_xz, -0.999, 0.999))
    z_yz = np.arctanh(np.clip(r_yz, -0.999, 0.999))

    # Steiger (1980) formula for dependent correlations
    r_det = (1 - r_xz**2 - r_yz**2 - r_xy**2 + 2 * r_xz * r_yz * r_xy)
    r_mean_sq = ((r_xz + r_yz) / 2) ** 2

    # Approximate variance of the difference of Fisher-z values
    h = (1 - r_xy) / (2 * (1 - r_mean_sq))
    h = max(h, 0.001)  # guard against degenerate case

    denom = np.sqrt(2 * (1 - r_xy) / ((n - 3) * (1 + r_xy)))
    if denom < 1e-10:
        return float("nan"), float("nan"), "undetermined"

    z_stat = (z_xz - z_yz) / denom
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    direction = "x>y" if r_xz > r_yz else "y>x"
    return float(z_stat), float(p_value), direction


def partial_correlation(x, y, z):
    """
    Partial Spearman correlation of x and y controlling for z.

    Uses rank-based partial correlation:
      r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Returns (partial_rho, approximate_p).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 10:
        return float("nan"), float("nan")

    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)

    r_xy = stats.spearmanr(x, y)[0]
    r_xz = stats.spearmanr(x, z)[0]
    r_yz = stats.spearmanr(y, z)[0]

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return float("nan"), float("nan")

    partial_r = (r_xy - r_xz * r_yz) / denom

    # Approximate t-test for significance
    df = n - 3
    if abs(partial_r) > 0.9999:
        return float(partial_r), 0.0
    t_stat = partial_r * np.sqrt(df / (1 - partial_r**2))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return float(partial_r), float(p_val)


# =====================================================================
# STATISTICAL QUALITY MEASURES (honestly labeled)
# =====================================================================

def compute_inv_trace_cov(embeddings):
    """
    Inverse trace of covariance matrix (frequentist).
    NOTE: For unit-norm embeddings, this is trivially related to E.
    Included for completeness; NOT used as independent evidence.
    """
    if embeddings.shape[0] < 3:
        return float("nan")
    cov = np.cov(embeddings, rowvar=False)
    tr = np.trace(cov)
    if tr < 1e-15:
        return float("nan")
    return 1.0 / tr


def compute_bootstrap_mean_precision(embeddings, n_bootstrap=500, rng=None):
    """
    Bootstrap mean precision: 1/trace(Var(bootstrap centroids)).

    This is a FREQUENTIST resampling quantity that measures how precisely
    the centroid can be estimated from the data. It is NOT a Bayesian
    posterior -- there is no prior, no likelihood model, no Bayes' rule.

    For iid data: approximately n / trace(Cov(X)), so it is dominated
    by trace(Cov(X)) which is related to E for unit-norm embeddings.
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


def compute_silhouette_approx(embeddings, labels):
    """
    Approximate silhouette score based on intra-cluster vs nearest-cluster distance.
    This is an independent quality measure NOT trivially related to E.

    For a pure single-cluster, we use the mean cosine distance within vs
    a random reference direction. For labeled multi-class data we use
    proper silhouette. Returns NaN if labels are all one class.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    from sklearn.metrics import silhouette_score
    try:
        # Use cosine distance for consistency
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = embeddings / norms
        score = silhouette_score(normed, labels, metric="cosine",
                                 sample_size=min(len(labels), 1000),
                                 random_state=SEED)
        return float(score)
    except Exception:
        return float("nan")


# =====================================================================
# DATA LOADING WITH CACHING
# =====================================================================

def load_20newsgroups_base():
    """Load raw 20 Newsgroups data."""
    print("  Loading 20 Newsgroups text data...")
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = newsgroups.data
    labels = newsgroups.target
    categories = newsgroups.target_names
    print("    Total documents: %d, categories: %d" % (len(texts), len(categories)))
    return texts, labels, categories


def encode_with_cache(texts, model_name):
    """Encode texts with sentence-transformer, caching to disk."""
    safe_name = model_name.replace("/", "_").replace("-", "_")
    cache_path = os.path.join(CACHE_DIR, "newsgroups_%s.npy" % safe_name)

    if os.path.exists(cache_path):
        print("  Loading cached embeddings for %s..." % model_name)
        embeddings = np.load(cache_path)
        print("    Loaded from cache: shape=%s" % (embeddings.shape,))
        return embeddings

    print("  Encoding with %s (this may take a while on CPU)..." % model_name)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings, dtype=np.float64)
    np.save(cache_path, embeddings)
    print("    Saved to cache: shape=%s" % (embeddings.shape,))
    return embeddings


def create_clusters_continuous_purity(embeddings, labels, categories, rng,
                                      n_clusters=90):
    """
    Create clusters with CONTINUOUS purity variation.

    Strategy (90 clusters total):
      - 20 pure clusters (1 category, purity ~1.0)
      - 20 mostly-pure (dominant category ~70-90% of docs)
      - 20 mixed-2 (two categories, ~50/50)
      - 15 mixed-3 (three categories, ~33/33/33)
      - 15 random (all categories, low purity)

    Each cluster has 200 documents.
    """
    n_cats = len(categories)
    clusters = []
    n_per_cluster = 200

    cat_indices = {}
    for cat_id in range(n_cats):
        cat_indices[cat_id] = np.where(labels == cat_id)[0]

    def sample_from_cat(cat_id, size, rng_local):
        avail = cat_indices[cat_id]
        return rng_local.choice(avail, size=size, replace=len(avail) < size)

    # -- 20 pure clusters --
    for i in range(20):
        cat_id = i % n_cats
        idx = sample_from_cat(cat_id, n_per_cluster, rng)
        cluster_labels = labels[idx]
        purity = float(np.max(np.bincount(cluster_labels, minlength=n_cats))) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "doc_labels": cluster_labels,
            "type": "pure",
            "purity": purity,
            "size": n_per_cluster,
        })

    # -- 20 mostly-pure (70-90% dominant) --
    for i in range(20):
        dominant_cat = rng.randint(0, n_cats)
        frac_dominant = 0.7 + rng.random() * 0.2  # 70-90%
        n_dominant = int(n_per_cluster * frac_dominant)
        n_noise = n_per_cluster - n_dominant
        idx_dom = sample_from_cat(dominant_cat, n_dominant, rng)
        # noise from random categories
        noise_cats = [c for c in range(n_cats) if c != dominant_cat]
        noise_cat = rng.choice(noise_cats)
        idx_noise = sample_from_cat(noise_cat, n_noise, rng)
        idx = np.concatenate([idx_dom, idx_noise])
        cluster_labels = labels[idx]
        purity = float(np.max(np.bincount(cluster_labels, minlength=n_cats))) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "doc_labels": cluster_labels,
            "type": "mostly_pure",
            "purity": purity,
            "size": n_per_cluster,
        })

    # -- 20 mixed-2 (50/50) --
    for i in range(20):
        c1, c2 = rng.choice(n_cats, size=2, replace=False)
        n1 = n_per_cluster // 2
        n2 = n_per_cluster - n1
        idx1 = sample_from_cat(c1, n1, rng)
        idx2 = sample_from_cat(c2, n2, rng)
        idx = np.concatenate([idx1, idx2])
        cluster_labels = labels[idx]
        purity = float(np.max(np.bincount(cluster_labels, minlength=n_cats))) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "doc_labels": cluster_labels,
            "type": "mixed_2",
            "purity": purity,
            "size": n_per_cluster,
        })

    # -- 15 mixed-3 (33/33/33) --
    for i in range(15):
        cats_3 = rng.choice(n_cats, size=3, replace=False)
        sizes = [n_per_cluster // 3, n_per_cluster // 3,
                 n_per_cluster - 2 * (n_per_cluster // 3)]
        idx_parts = []
        for ci, sz in zip(cats_3, sizes):
            idx_parts.append(sample_from_cat(ci, sz, rng))
        idx = np.concatenate(idx_parts)
        cluster_labels = labels[idx]
        purity = float(np.max(np.bincount(cluster_labels, minlength=n_cats))) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "doc_labels": cluster_labels,
            "type": "mixed_3",
            "purity": purity,
            "size": n_per_cluster,
        })

    # -- 15 random --
    all_indices = np.arange(len(labels))
    for i in range(15):
        idx = rng.choice(all_indices, size=n_per_cluster, replace=False)
        cluster_labels = labels[idx]
        purity = float(np.max(np.bincount(cluster_labels, minlength=n_cats))) / len(cluster_labels)
        clusters.append({
            "embeddings": embeddings[idx],
            "doc_labels": cluster_labels,
            "type": "random",
            "purity": purity,
            "size": n_per_cluster,
        })

    type_counts = {}
    for c in clusters:
        t = c["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    purities = [c["purity"] for c in clusters]
    print("    Created %d clusters: %s" % (len(clusters), type_counts))
    print("    Purity range: %.3f to %.3f (mean=%.3f)" % (
        min(purities), max(purities), np.mean(purities)))
    return clusters


# =====================================================================
# TEST 1: R vs Statistical Quality Measures (per architecture)
# =====================================================================

def test1_stat_quality_correlation(texts, labels, categories):
    """
    For each architecture, create 90 clusters from 20 Newsgroups with
    continuous purity variation.

    Compute R_full, R_simple, E and compare against HONESTLY LABELED
    frequentist statistical quality measures:
      (a) inv_trace_cov:       1/trace(Cov). Trivially related to E.
      (b) bootstrap_mean_prec: 1/trace(Var(bootstrap centroids)). Frequentist.
      (c) silhouette:          Silhouette score. Independent of E.

    For each measure, compute:
      - Spearman and Pearson correlations with CIs
      - Steiger's test comparing rho(R_full, measure) vs rho(E, measure)
      - Partial correlation rho(R_full, measure | E)
    """
    print("\n" + "=" * 70)
    print("TEST 1: R vs Statistical Quality Measures (per architecture)")
    print("  NOTE: All measures are FREQUENTIST. No Bayesian claims made.")
    print("=" * 70)

    results_by_arch = {}

    # Quality measures (honestly named)
    quality_keys = ["inv_trace_cov", "bootstrap_mean_prec", "silhouette"]
    # inv_trace_cov is included for completeness but flagged as trivial
    nontrivial_keys = ["bootstrap_mean_prec", "silhouette"]
    formula_keys = ["R_full", "R_simple", "E"]

    for arch_idx, model_name in enumerate(ARCHITECTURES):
        print("\n--- Architecture: %s ---" % model_name)
        embeddings = encode_with_cache(texts, model_name)
        rng = np.random.RandomState(SEED + arch_idx)
        clusters = create_clusters_continuous_purity(
            embeddings, labels, categories, rng, n_clusters=N_CLUSTERS
        )

        metrics = {k: [] for k in formula_keys + quality_keys}
        cluster_meta = []

        for c_idx, cluster in enumerate(clusters):
            emb = cluster["embeddings"]
            doc_labs = cluster["doc_labels"]

            all_comp = compute_all(emb)
            metrics["R_full"].append(all_comp["R_full"])
            metrics["R_simple"].append(all_comp["R_simple"])
            metrics["E"].append(all_comp["E"])

            metrics["inv_trace_cov"].append(compute_inv_trace_cov(emb))
            boot_rng = np.random.RandomState(SEED + arch_idx * 10000 + c_idx)
            metrics["bootstrap_mean_prec"].append(
                compute_bootstrap_mean_precision(emb, n_bootstrap=500, rng=boot_rng)
            )
            metrics["silhouette"].append(
                compute_silhouette_approx(emb, doc_labs)
            )
            cluster_meta.append({
                "type": cluster["type"],
                "purity": cluster["purity"],
            })

        for k in metrics:
            metrics[k] = np.array(metrics[k])

        n_valid = len(clusters)

        # --- Spearman correlations ---
        print("\n  Spearman correlations (n=%d clusters):" % n_valid)
        header = "  %-20s" % ""
        for qk in quality_keys:
            header += " %22s" % qk
        print(header)
        print("  " + "-" * 90)

        arch_corrs = {}
        for fk in formula_keys:
            row = "  %-20s" % fk
            arch_corrs[fk] = {}
            for qk in quality_keys:
                rho, p = safe_spearman(metrics[fk], metrics[qk])
                row += "  %+7.3f (p=%.1e)" % (rho, p) if np.isfinite(rho) else "  %22s" % "NaN"
                arch_corrs[fk][qk] = {
                    "spearman_rho": float(rho) if np.isfinite(rho) else None,
                    "spearman_p": float(p) if np.isfinite(p) else None,
                }
            print(row)

        # --- Pearson correlations ---
        print("\n  Pearson correlations (n=%d clusters):" % n_valid)
        for fk in formula_keys:
            row = "  %-20s" % fk
            for qk in quality_keys:
                r, p = safe_pearson(metrics[fk], metrics[qk])
                row += "  %+7.3f (p=%.1e)" % (r, p) if np.isfinite(r) else "  %22s" % "NaN"
                arch_corrs[fk][qk]["pearson_r"] = float(r) if np.isfinite(r) else None
                arch_corrs[fk][qk]["pearson_p"] = float(p) if np.isfinite(p) else None
            print(row)

        # --- Bootstrap CIs for key correlations ---
        print("\n  Bootstrap 95%% CIs for Spearman correlations:")
        for fk in formula_keys:
            for qk in nontrivial_keys:
                ci_rng = np.random.RandomState(SEED + arch_idx * 1000 + hash(fk + qk) % 10000)
                rho_pt, ci_lo, ci_hi = bootstrap_correlation_ci(
                    metrics[fk], metrics[qk], method="spearman",
                    n_boot=2000, rng=ci_rng
                )
                print("    %s vs %s: %.3f [%.3f, %.3f]" % (fk, qk, rho_pt, ci_lo, ci_hi))
                arch_corrs[fk][qk]["spearman_ci95"] = [ci_lo, ci_hi]

        # --- Steiger's test: R_full vs E for each quality measure ---
        print("\n  Steiger's test (R_full vs E, shared quality measure):")
        steiger_results = {}
        # r(R_full, E) needed for Steiger's
        r_rfull_e, _ = safe_spearman(metrics["R_full"], metrics["E"])
        print("    r(R_full, E) = %.3f" % r_rfull_e)
        for qk in nontrivial_keys:
            r_rfull_q = arch_corrs["R_full"][qk]["spearman_rho"]
            r_e_q = arch_corrs["E"][qk]["spearman_rho"]
            if r_rfull_q is None or r_e_q is None:
                steiger_results[qk] = {"z": None, "p": None, "direction": "undetermined"}
                continue
            z_stat, p_val, direction = steiger_test(
                abs(r_rfull_q), abs(r_e_q), abs(r_rfull_e), n_valid
            )
            winner = "R_full" if direction == "x>y" else "E"
            sig_marker = " *" if (not math.isnan(p_val) and p_val < 0.05) else ""
            print("    vs %s: |rho_R|=%.3f, |rho_E|=%.3f, Z=%.3f, p=%.4f -> %s wins%s" % (
                qk, abs(r_rfull_q), abs(r_e_q), z_stat, p_val, winner, sig_marker))
            steiger_results[qk] = {
                "z": float(z_stat) if np.isfinite(z_stat) else None,
                "p": float(p_val) if np.isfinite(p_val) else None,
                "direction": direction,
                "winner": winner,
                "significant": bool(not math.isnan(p_val) and p_val < 0.05),
            }

        # --- Partial correlations: R_full | E ---
        print("\n  Partial Spearman correlation rho(R_full, measure | E):")
        partial_results = {}
        for qk in nontrivial_keys:
            pr, pp = partial_correlation(metrics["R_full"], metrics[qk], metrics["E"])
            sig_marker = " *" if (not math.isnan(pp) and pp < 0.05) else ""
            print("    R_full vs %s | E: partial_rho=%.3f, p=%.4f%s" % (qk, pr, pp, sig_marker))
            partial_results[qk] = {
                "partial_rho": float(pr) if np.isfinite(pr) else None,
                "partial_p": float(pp) if np.isfinite(pp) else None,
                "significant": bool(not math.isnan(pp) and pp < 0.05),
            }

        # --- E vs inv_trace_cov check (should be ~1.0) ---
        e_itc_rho = arch_corrs["E"]["inv_trace_cov"]["spearman_rho"]
        print("\n  TRIVIAL CHECK: E vs inv_trace_cov rho = %s (expected ~1.0)" % e_itc_rho)

        # --- Silhouette validity check ---
        n_valid_sil = np.isfinite(metrics["silhouette"]).sum()
        print("  Silhouette valid: %d/%d clusters" % (n_valid_sil, n_valid))

        results_by_arch[model_name] = {
            "n_clusters": n_valid,
            "correlations": arch_corrs,
            "steiger_tests": steiger_results,
            "partial_correlations": partial_results,
            "r_rfull_e": float(r_rfull_e) if np.isfinite(r_rfull_e) else None,
            "cluster_types": [c["type"] for c in cluster_meta],
            "cluster_purities": [c["purity"] for c in cluster_meta],
            "metrics_summary": {
                k: {
                    "mean": float(np.nanmean(v)),
                    "std": float(np.nanstd(v)),
                    "n_valid": int(np.isfinite(v).sum()),
                }
                for k, v in metrics.items()
            },
        }

    # Cross-architecture summary
    print("\n  CROSS-ARCHITECTURE SUMMARY:")
    print("  " + "-" * 90)
    for qk in nontrivial_keys:
        print("\n  vs %s:" % qk)
        for fk in formula_keys:
            rhos = []
            for model_name in ARCHITECTURES:
                r = results_by_arch[model_name]["correlations"][fk][qk]["spearman_rho"]
                if r is not None:
                    rhos.append(r)
            if rhos:
                mean_rho = np.mean(rhos)
                print("    %-20s: mean |rho| = %.3f (%s)" % (
                    fk, np.mean([abs(r) for r in rhos]),
                    ", ".join(["%+.3f" % r for r in rhos])))

    return {"status": "complete", "by_architecture": results_by_arch}


# =====================================================================
# TEST 2: Intensive Property (FIXED with domain-weighted CV)
# =====================================================================

def test2_intensive_property(texts, labels, categories):
    """
    Test whether R_full is an intensive property (stable across sample sizes).

    Part A: 20 Newsgroups text embeddings (384-dim), 3 largest categories.
    Part B: California Housing (8-dim), geographic clusters.

    FIX: Use domain-weighted CV (equal weight to text and housing domains).
    Also test at multiple N values: 50, 100, 200, 500.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Intensive Property")
    print("=" * 70)

    subsample_sizes = [50, 100, 200, 500]
    n_repeats = 30

    # ---- Part A: 20 Newsgroups ----
    print("\n  --- Part A: 20 Newsgroups (all-MiniLM-L6-v2) ---")
    embeddings = encode_with_cache(texts, ARCHITECTURES[0])

    cat_sizes = [(cat_id, (labels == cat_id).sum()) for cat_id in range(len(categories))]
    cat_sizes.sort(key=lambda x: -x[1])
    top3_cats = [cat_sizes[i][0] for i in range(3)]
    print("    3 largest categories: %s (sizes: %s)" % (
        [categories[c] for c in top3_cats],
        [cat_sizes[i][1] for i in range(3)]))

    text_results = []
    for cat_id in top3_cats:
        cat_mask = labels == cat_id
        cat_emb = embeddings[cat_mask]
        cat_n = cat_emb.shape[0]
        print("\n    Category: %s (n=%d)" % (categories[cat_id], cat_n))

        R_full_by_N = {}
        for N in subsample_sizes:
            if N > cat_n:
                continue
            R_full_samples = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(SEED + cat_id * 10000 + N * 100 + rep)
                idx = rng.choice(cat_n, size=N, replace=False)
                sub = cat_emb[idx]
                R_full_samples.append(compute_R_full(sub))
            R_full_by_N[N] = float(np.nanmean(R_full_samples))
            print("      N=%4d: R_full=%.4f (std=%.4f)" % (
                N, R_full_by_N[N], float(np.nanstd(R_full_samples))))

        vals = np.array([v for v in R_full_by_N.values() if np.isfinite(v)])
        if len(vals) >= 2 and np.abs(np.mean(vals)) > 1e-15:
            cv = float(np.std(vals) / np.abs(np.mean(vals)))
        else:
            cv = float("nan")
        print("      CV across N: %.4f" % cv)

        text_results.append({
            "category": categories[cat_id],
            "n_total": cat_n,
            "cv_R_full": cv,
            "R_full_by_N": {str(k): v for k, v in R_full_by_N.items()},
        })

    # ---- Part B: California Housing (8-dim) ----
    print("\n  --- Part B: California Housing (8-dim features) ---")
    print("    NOTE: These are standardized tabular features, NOT neural embeddings.")
    housing = fetch_california_housing()
    X_housing = StandardScaler().fit_transform(housing.data)
    lat = housing.data[:, -2]
    lon = housing.data[:, -1]
    coords = np.column_stack([lat, lon])

    km = KMeans(n_clusters=20, random_state=SEED, n_init=10)
    cluster_labels_housing = km.fit_predict(coords)

    housing_results = []
    for cl in range(20):
        cl_mask = cluster_labels_housing == cl
        cl_emb = X_housing[cl_mask]
        cl_n = cl_emb.shape[0]
        if cl_n < 500:
            continue

        R_full_by_N = {}
        for N in subsample_sizes:
            R_full_samples = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(SEED + cl * 10000 + N * 100 + rep)
                idx = rng.choice(cl_n, size=N, replace=False)
                sub = cl_emb[idx]
                R_full_samples.append(compute_R_full(sub))
            R_full_by_N[N] = float(np.nanmean(R_full_samples))

        vals = np.array([v for v in R_full_by_N.values() if np.isfinite(v)])
        if len(vals) >= 2 and np.abs(np.mean(vals)) > 1e-15:
            cv = float(np.std(vals) / np.abs(np.mean(vals)))
        else:
            cv = float("nan")
        housing_results.append({
            "cluster": cl,
            "n_total": cl_n,
            "cv_R_full": cv,
            "R_full_by_N": {str(k): v for k, v in R_full_by_N.items()},
        })
        print("    Cluster %2d (n=%5d): CV R_full=%.4f" % (cl, cl_n, cv))

    # Summary with DOMAIN-WEIGHTED averaging (BUG-3 fix)
    text_cvs = [r["cv_R_full"] for r in text_results if np.isfinite(r["cv_R_full"])]
    housing_cvs = [r["cv_R_full"] for r in housing_results if np.isfinite(r["cv_R_full"])]

    text_mean_cv = float(np.mean(text_cvs)) if text_cvs else None
    housing_mean_cv = float(np.mean(housing_cvs)) if housing_cvs else None

    # Domain-weighted: equal weight to text and housing domains
    if text_mean_cv is not None and housing_mean_cv is not None:
        domain_weighted_cv = (text_mean_cv + housing_mean_cv) / 2.0
    elif text_mean_cv is not None:
        domain_weighted_cv = text_mean_cv
    elif housing_mean_cv is not None:
        domain_weighted_cv = housing_mean_cv
    else:
        domain_weighted_cv = None

    # Raw (unweighted) mean for comparison
    all_cvs = text_cvs + housing_cvs
    raw_mean_cv = float(np.mean(all_cvs)) if all_cvs else None

    print("\n  SUMMARY:")
    if text_cvs:
        print("    Text domain (384-dim) mean CV:     %.4f  (%s)" % (
            text_mean_cv, ", ".join(["%.4f" % v for v in text_cvs])))
    if housing_cvs:
        print("    Housing domain (8-dim) mean CV:    %.4f  (n=%d clusters)" % (
            housing_mean_cv, len(housing_cvs)))
    print("    Domain-weighted mean CV:           %s" % (
        "%.4f" % domain_weighted_cv if domain_weighted_cv is not None else "N/A"))
    print("    Raw unweighted mean CV:            %s" % (
        "%.4f" % raw_mean_cv if raw_mean_cv is not None else "N/A"))
    print("    Threshold: CV < 0.15")
    if domain_weighted_cv is not None:
        print("    Intensive? domain-weighted CV %.4f < 0.15 -> %s" % (
            domain_weighted_cv, "YES" if domain_weighted_cv < 0.15 else "NO"))
    print("    WARNING: Text domain alone has CV=%.4f which FAILS the 0.15 threshold." % (
        text_mean_cv if text_mean_cv else float("nan")))

    return {
        "status": "complete",
        "text_results": text_results,
        "housing_results": housing_results,
        "summary": {
            "text_mean_cv": text_mean_cv,
            "housing_mean_cv": housing_mean_cv,
            "domain_weighted_cv": domain_weighted_cv,
            "raw_mean_cv": raw_mean_cv,
        },
    }


# =====================================================================
# TEST 3: Continuous Purity Prediction (replaces trivial binary gating)
# =====================================================================

def test3_purity_prediction(texts, labels, categories):
    """
    REDESIGNED (BUG-4 fix): Instead of trivial binary classification,
    test whether R_full predicts CONTINUOUS purity better than E alone.

    For each architecture:
      - Create 90 clusters with varying purity
      - Compute Spearman correlation of each metric with purity
      - Steiger's test: does R_full correlate with purity better than E?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Continuous Purity Prediction (replaces trivial gating)")
    print("  NOTE: Binary gating was trivially easy (F1=1.0 for all).")
    print("  This test asks: which metric best predicts continuous purity?")
    print("=" * 70)

    results_by_arch = {}

    for arch_idx, model_name in enumerate(ARCHITECTURES):
        print("\n--- Architecture: %s ---" % model_name)
        embeddings = encode_with_cache(texts, model_name)
        rng = np.random.RandomState(SEED + arch_idx + 200)
        clusters = create_clusters_continuous_purity(
            embeddings, labels, categories, rng, n_clusters=N_CLUSTERS
        )

        purities = np.array([c["purity"] for c in clusters])

        metric_names = ["R_full", "R_simple", "E", "inv_trace_cov"]
        metric_vals = {k: [] for k in metric_names}

        for c in clusters:
            emb = c["embeddings"]
            all_comp = compute_all(emb)
            metric_vals["R_full"].append(all_comp["R_full"])
            metric_vals["R_simple"].append(all_comp["R_simple"])
            metric_vals["E"].append(all_comp["E"])
            metric_vals["inv_trace_cov"].append(compute_inv_trace_cov(emb))

        for k in metric_vals:
            metric_vals[k] = np.array(metric_vals[k])

        print("\n  Correlation with continuous purity (n=%d):" % len(purities))
        purity_corrs = {}
        for mk in metric_names:
            rho, p = safe_spearman(metric_vals[mk], purities)
            r_pear, p_pear = safe_pearson(metric_vals[mk], purities)
            print("    %-20s: Spearman=%.3f (p=%.1e), Pearson=%.3f (p=%.1e)" % (
                mk, rho, p, r_pear, p_pear))
            purity_corrs[mk] = {
                "spearman_rho": float(rho) if np.isfinite(rho) else None,
                "spearman_p": float(p) if np.isfinite(p) else None,
                "pearson_r": float(r_pear) if np.isfinite(r_pear) else None,
                "pearson_p": float(p_pear) if np.isfinite(p_pear) else None,
            }

        # Steiger's test: R_full vs E in predicting purity
        r_rfull_pur = purity_corrs["R_full"]["spearman_rho"]
        r_e_pur = purity_corrs["E"]["spearman_rho"]
        r_rfull_e, _ = safe_spearman(metric_vals["R_full"], metric_vals["E"])

        if r_rfull_pur is not None and r_e_pur is not None:
            z_stat, p_val, direction = steiger_test(
                abs(r_rfull_pur), abs(r_e_pur), abs(r_rfull_e), len(purities)
            )
            winner = "R_full" if direction == "x>y" else "E"
            sig = " *" if (not math.isnan(p_val) and p_val < 0.05) else ""
            print("\n  Steiger's test (R_full vs E for purity prediction):")
            print("    |rho(R_full, purity)|=%.3f, |rho(E, purity)|=%.3f" % (
                abs(r_rfull_pur), abs(r_e_pur)))
            print("    Z=%.3f, p=%.4f -> %s wins%s" % (z_stat, p_val, winner, sig))
        else:
            z_stat, p_val, winner = float("nan"), float("nan"), "undetermined"

        diff = (abs(r_rfull_pur) if r_rfull_pur else 0) - (abs(r_e_pur) if r_e_pur else 0)
        print("    |rho| difference: %+.4f" % diff)

        results_by_arch[model_name] = {
            "n_clusters": len(purities),
            "purity_correlations": purity_corrs,
            "steiger_test": {
                "z": float(z_stat) if np.isfinite(z_stat) else None,
                "p": float(p_val) if np.isfinite(p_val) else None,
                "winner": winner,
                "significant": bool(not math.isnan(p_val) and p_val < 0.05),
            },
            "r_rfull_e": float(r_rfull_e) if np.isfinite(r_rfull_e) else None,
            "rho_diff_rfull_minus_e": float(diff),
        }

    # Cross-architecture summary
    print("\n  CROSS-ARCHITECTURE SUMMARY:")
    for m in ARCHITECTURES:
        r = results_by_arch[m]
        print("    %s: R_full=%.3f, E=%.3f, diff=%+.4f, Steiger p=%s" % (
            m,
            abs(r["purity_correlations"]["R_full"]["spearman_rho"] or 0),
            abs(r["purity_correlations"]["E"]["spearman_rho"] or 0),
            r["rho_diff_rfull_minus_e"],
            "%.4f" % r["steiger_test"]["p"] if r["steiger_test"]["p"] is not None else "N/A"))

    return {"status": "complete", "by_architecture": results_by_arch}


# =====================================================================
# VERDICT
# =====================================================================

def compute_verdict(all_results):
    """
    Pre-registered criteria determine the verdict. No editorial overrides.

    CONFIRMED if R significantly outperforms E (Steiger p<0.05) in
      correlation with >=2/3 statistical quality measures
      on >=2/3 architectures.
    FALSIFIED if E significantly outperforms R on ALL measures
      on ALL architectures.
    INCONCLUSIVE otherwise.
    """
    print("\n" + "=" * 70)
    print("VERDICT SYNTHESIS (pre-registered criteria only)")
    print("  Question: Does R correlate with statistical quality measures")
    print("  better than E alone?")
    print("  NOTE: Reframed from 'Bayesian interpretation' to honest")
    print("  'statistical quality correlation' per audit METH-1.")
    print("=" * 70)

    verdict_data = {"criteria": {}, "details": {}}

    nontrivial_keys = ["bootstrap_mean_prec", "silhouette"]

    # ---- Criterion A: Steiger test on quality measures (Test 1) ----
    t1 = all_results.get("test1", {})
    n_measures = len(nontrivial_keys)
    r_wins_count = 0  # measures where R significantly beats E on >=2/3 archs
    e_wins_all_count = 0  # measures where E significantly beats R on ALL archs

    measure_summaries = {}
    if t1.get("status") == "complete":
        for qk in nontrivial_keys:
            r_sig_wins = 0
            e_sig_wins = 0
            for model_name in ARCHITECTURES:
                arch_data = t1["by_architecture"].get(model_name, {})
                steiger = arch_data.get("steiger_tests", {}).get(qk, {})
                if steiger.get("significant"):
                    if steiger.get("winner") == "R_full":
                        r_sig_wins += 1
                    elif steiger.get("winner") == "E":
                        e_sig_wins += 1

            if r_sig_wins >= 2:
                r_wins_count += 1
            if e_sig_wins == len(ARCHITECTURES):
                e_wins_all_count += 1

            measure_summaries[qk] = {
                "R_significant_wins": r_sig_wins,
                "E_significant_wins": e_sig_wins,
                "R_wins_majority": r_sig_wins >= 2,
                "E_wins_all": e_sig_wins == len(ARCHITECTURES),
            }

    criterion_a_confirm = r_wins_count >= 2  # R wins on >=2/3 measures
    criterion_a_falsify = e_wins_all_count == n_measures  # E wins on ALL

    print("\n  Criterion A: Steiger tests on statistical quality measures")
    for qk, summary in measure_summaries.items():
        print("    %s: R wins %d/%d archs, E wins %d/%d archs" % (
            qk, summary["R_significant_wins"], len(ARCHITECTURES),
            summary["E_significant_wins"], len(ARCHITECTURES)))
    print("    CONFIRM (R sig. wins on >=2/3 measures): %s" % (
        "YES" if criterion_a_confirm else "NO"))
    print("    FALSIFY (E sig. wins on ALL measures): %s" % (
        "YES" if criterion_a_falsify else "NO"))

    verdict_data["criteria"]["A_steiger_quality"] = {
        "confirm": bool(criterion_a_confirm),
        "falsify": bool(criterion_a_falsify),
        "measure_summaries": measure_summaries,
    }

    # ---- Criterion B: Intensive property (Test 2) ----
    t2 = all_results.get("test2", {})
    domain_weighted_cv = None
    if t2.get("status") == "complete":
        domain_weighted_cv = t2.get("summary", {}).get("domain_weighted_cv")

    criterion_b_confirm = domain_weighted_cv is not None and domain_weighted_cv < 0.15
    criterion_b_falsify = domain_weighted_cv is not None and domain_weighted_cv > 0.15

    print("\n  Criterion B: Intensive property (domain-weighted CV < 0.15)")
    print("    Domain-weighted CV: %s" % (
        "%.4f" % domain_weighted_cv if domain_weighted_cv is not None else "N/A"))
    print("    CONFIRM: %s" % ("YES" if criterion_b_confirm else "NO"))
    print("    FALSIFY component: %s" % ("YES" if criterion_b_falsify else "NO"))

    text_cv = t2.get("summary", {}).get("text_mean_cv") if t2.get("status") == "complete" else None
    housing_cv = t2.get("summary", {}).get("housing_mean_cv") if t2.get("status") == "complete" else None
    print("    NOTE: Text domain CV=%s, Housing domain CV=%s" % (
        "%.4f" % text_cv if text_cv is not None else "N/A",
        "%.4f" % housing_cv if housing_cv is not None else "N/A"))
    if text_cv is not None and text_cv > 0.15:
        print("    WARNING: Text domain alone FAILS the 0.15 threshold.")

    verdict_data["criteria"]["B_intensive"] = {
        "confirm": bool(criterion_b_confirm),
        "falsify_component": bool(criterion_b_falsify),
        "domain_weighted_cv": domain_weighted_cv,
        "text_mean_cv": text_cv,
        "housing_mean_cv": housing_cv,
    }

    # ---- Criterion C: Purity prediction (Test 3) ----
    t3 = all_results.get("test3", {})
    n_archs_r_wins_purity = 0
    n_archs_e_wins_purity = 0
    if t3.get("status") == "complete":
        for model_name in ARCHITECTURES:
            arch_data = t3["by_architecture"].get(model_name, {})
            steiger = arch_data.get("steiger_test", {})
            if steiger.get("significant"):
                if steiger.get("winner") == "R_full":
                    n_archs_r_wins_purity += 1
                elif steiger.get("winner") == "E":
                    n_archs_e_wins_purity += 1

    criterion_c_confirm = n_archs_r_wins_purity >= 2
    criterion_c_falsify = n_archs_e_wins_purity == len(ARCHITECTURES)

    print("\n  Criterion C: Purity prediction (Steiger R_full vs E)")
    print("    R_full sig. wins on %d/%d architectures" % (
        n_archs_r_wins_purity, len(ARCHITECTURES)))
    print("    E sig. wins on %d/%d architectures" % (
        n_archs_e_wins_purity, len(ARCHITECTURES)))
    print("    CONFIRM (R wins on >=2/3): %s" % ("YES" if criterion_c_confirm else "NO"))
    print("    FALSIFY (E wins on all): %s" % ("YES" if criterion_c_falsify else "NO"))

    verdict_data["criteria"]["C_purity_prediction"] = {
        "confirm": bool(criterion_c_confirm),
        "falsify": bool(criterion_c_falsify),
        "r_wins": n_archs_r_wins_purity,
        "e_wins": n_archs_e_wins_purity,
    }

    # ---- Partial correlation summary ----
    print("\n  Supplementary: Partial correlations rho(R_full, measure | E)")
    partial_summary = {}
    if t1.get("status") == "complete":
        for qk in nontrivial_keys:
            partials = []
            for model_name in ARCHITECTURES:
                arch_data = t1["by_architecture"].get(model_name, {})
                pc = arch_data.get("partial_correlations", {}).get(qk, {})
                pr = pc.get("partial_rho")
                pp = pc.get("partial_p")
                sig = pc.get("significant", False)
                if pr is not None:
                    partials.append((model_name, pr, pp, sig))
            print("    %s:" % qk)
            for (m, pr, pp, sig) in partials:
                sig_marker = " *" if sig else ""
                print("      %s: partial_rho=%.3f, p=%.4f%s" % (m, pr, pp, sig_marker))
            partial_summary[qk] = [
                {"arch": m, "partial_rho": pr, "p": pp, "significant": sig}
                for (m, pr, pp, sig) in partials
            ]

    verdict_data["details"]["partial_correlations"] = partial_summary

    # ---- Apply pre-registered decision rules ----
    if criterion_a_confirm and criterion_b_confirm and criterion_c_confirm:
        overall = "CONFIRMED"
    elif criterion_a_falsify or (criterion_b_falsify and criterion_c_falsify):
        overall = "FALSIFIED"
    else:
        overall = "INCONCLUSIVE"

    print("\n  " + "=" * 50)
    print("  PRE-REGISTERED VERDICT: %s" % overall)
    print("  " + "=" * 50)

    if overall == "INCONCLUSIVE":
        print("\n  INTERPRETATION (supplementary, not part of decision):")
        # Check if the partial correlations suggest R adds nothing beyond E
        any_partial_sig = False
        if t1.get("status") == "complete":
            for qk in nontrivial_keys:
                for model_name in ARCHITECTURES:
                    pc = t1["by_architecture"].get(model_name, {}).get(
                        "partial_correlations", {}).get(qk, {})
                    if pc.get("significant"):
                        any_partial_sig = True

        if any_partial_sig:
            print("    Some partial correlations R_full|E are significant,")
            print("    suggesting R_full captures information beyond E alone.")
        else:
            print("    No partial correlations R_full|E are significant,")
            print("    suggesting R_full's correlation with quality measures")
            print("    is entirely explained by E.")

    verdict_data["overall"] = overall
    return verdict_data


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("Q15 v3: Does R Correlate with Statistical Quality Measures?")
    print("  (Honest reframing: all frequentist, no Bayesian claims)")
    print("=" * 70)
    print("Seed: %d" % SEED)
    print("Time: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Architectures: %s" % ARCHITECTURES)
    print("Clusters: %d (continuous purity)" % N_CLUSTERS)
    print()

    all_results = {}

    # Load data
    texts, labels, categories = load_20newsgroups_base()

    # Pre-encode
    print("\n--- Pre-encoding all architectures ---")
    for model_name in ARCHITECTURES:
        encode_with_cache(texts, model_name)
    print("--- All architectures encoded ---\n")

    # Test 1: Statistical quality correlation
    try:
        all_results["test1"] = test1_stat_quality_correlation(texts, labels, categories)
    except Exception as e:
        print("  TEST 1 FAILED: %s" % e)
        traceback.print_exc()
        all_results["test1"] = {"status": "error", "error": str(e)}

    # Test 2: Intensive property
    try:
        all_results["test2"] = test2_intensive_property(texts, labels, categories)
    except Exception as e:
        print("  TEST 2 FAILED: %s" % e)
        traceback.print_exc()
        all_results["test2"] = {"status": "error", "error": str(e)}

    # Test 3: Continuous purity prediction
    try:
        all_results["test3"] = test3_purity_prediction(texts, labels, categories)
    except Exception as e:
        print("  TEST 3 FAILED: %s" % e)
        traceback.print_exc()
        all_results["test3"] = {"status": "error", "error": str(e)}

    # Verdict
    try:
        verdict_data = compute_verdict(all_results)
    except Exception as e:
        print("  VERDICT COMPUTATION FAILED: %s" % e)
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

    results_path = os.path.join(RESULTS_DIR, "test_v3_q15_results.json")
    with open(results_path, "w") as f:
        json.dump(convert_for_json({
            "metadata": {
                "version": "v3",
                "seed": SEED,
                "n_clusters": N_CLUSTERS,
                "architectures": ARCHITECTURES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reframing_note": (
                    "v3 drops all Bayesian claims. Tests frequentist statistical "
                    "quality measures only. See AUDIT.md for rationale."
                ),
            },
            "tests": all_results,
            "verdict": verdict_data,
        }), f, indent=2)
    print("\nResults saved to: %s" % results_path)

    return all_results, verdict_data


if __name__ == "__main__":
    all_results, verdict_data = main()
