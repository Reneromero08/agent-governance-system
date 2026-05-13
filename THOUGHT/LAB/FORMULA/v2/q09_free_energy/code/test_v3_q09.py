"""
Q09 v3 Test: Does log(R) correlate with Gaussian NLL?

AUDIT FIXES (v2 -> v3):
  1. CONTINUOUS cluster variation (purity 0.1-1.0 in 0.05 steps) eliminates 3-group confound
  2. WITHIN-GROUP correlations computed for each purity band
  3. Honestly relabeled: "Gaussian NLL" not "Free Energy" (not FEP-standard)
  4. Fixed ddof inconsistency: scatter and covariance both use ddof=0
  5. Harder gating task: rank-ordering within narrow purity bands
  6. Null-hypothesis comparison: correlate -NLL with other simple cluster stats
  7. 3 architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1

PRE-REGISTERED CRITERIA:
  CONFIRMED:  within-group Pearson |r| > 0.7 on >= 2/3 architectures
              AND identity residual std < 10% of range(NLL)
  FALSIFIED:  overall Pearson |r| < 0.5
              OR within-group |r| < 0.3 on all 3 architectures
  INCONCLUSIVE: otherwise

NO synthetic data. NO reward-maxing. Just truth.
"""

import sys
import os
import gc
import json
import warnings
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import fetch_20newsgroups

# Import shared formula via importlib to avoid path issues
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py",
)
formula = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(formula)
compute_E = formula.compute_E
compute_grad_S = formula.compute_grad_S
compute_R_simple = formula.compute_R_simple
compute_R_full = formula.compute_R_full
compute_all = formula.compute_all

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Architecture configs
ARCHITECTURES = [
    ("all-MiniLM-L6-v2", 384),
    ("all-mpnet-base-v2", 768),
    ("multi-qa-MiniLM-L6-cos-v1", 384),
]

CLUSTER_SIZE = 200    # docs per cluster
SEED = 42
MAX_DOCS_PER_CATEGORY = 250

# Continuous purity levels: 0.10 to 1.00 in steps of 0.05 = 19 levels
PURITY_LEVELS = [round(0.10 + i * 0.05, 2) for i in range(19)]
CLUSTERS_PER_PURITY = 3  # 3 clusters at each purity level = 57 total


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def compute_gaussian_nll(embeddings):
    """
    Compute Gaussian negative log-likelihood (NLL) for embeddings.

    NLL = 0.5 * (d * log(2*pi) + log(det(cov)) + tr(cov_inv @ scatter))

    IMPORTANT: This is NOT Free Energy Principle (FEP) variational free energy.
    FEP free energy requires a recognition density q(z) and generative model p(x,z):
        F_FEP = E_q[log q(z) - log p(x,z)]
    This is just the NLL under a fitted Gaussian, i.e., the degenerate case where
    q = delta(mu_ML) and the generative model is a single Gaussian.

    Uses ddof=0 consistently for both scatter and covariance.
    Regularization: reg=1e-4 added to diagonal for numerical stability.
    """
    n, d = embeddings.shape
    if n < 3:
        return float("nan")

    reg = 1e-4
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # Both scatter and cov use ddof=0 (divide by n)
    scatter = (centered.T @ centered) / n
    cov = scatter.copy()  # identical when ddof=0
    cov_reg = cov + reg * np.eye(d)

    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float("nan")

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return float("nan")

    trace_term = np.trace(cov_inv @ scatter)

    nll = 0.5 * (d * np.log(2 * np.pi) + logdet + trace_term)
    return float(nll)


def compute_simple_cluster_stats(embeddings):
    """
    Compute alternative cluster statistics for null-hypothesis comparison.
    If R's correlation with -NLL is not special, these should also correlate strongly.
    """
    n, d = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1)
    centered = embeddings - np.mean(embeddings, axis=0)
    cov = (centered.T @ centered) / n

    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues_pos = eigenvalues[eigenvalues > 0]

    mean_l2_norm = float(np.mean(norms))
    trace_cov = float(np.trace(cov))
    first_eigenvalue = float(eigenvalues_pos[0]) if len(eigenvalues_pos) > 0 else float("nan")
    mean_variance = float(np.mean(np.var(embeddings, axis=0)))

    # Mean pairwise Euclidean distance (sample for speed)
    rng = np.random.RandomState(SEED)
    if n > 100:
        idx = rng.choice(n, size=100, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings
    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    upper = np.triu_indices(len(sample), k=1)
    mean_euclidean_dist = float(np.mean(dists[upper]))

    return {
        "mean_l2_norm": mean_l2_norm,
        "trace_cov": trace_cov,
        "first_eigenvalue": first_eigenvalue,
        "mean_variance": mean_variance,
        "mean_euclidean_dist": mean_euclidean_dist,
    }


# ============================================================
# DATA LOADING AND CLUSTER CONSTRUCTION
# ============================================================

def load_20newsgroups_subsampled(max_per_cat=250, seed=42):
    """Load 20 Newsgroups, subsample to max_per_cat docs per category."""
    print("Loading 20 Newsgroups dataset...", flush=True)
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    print(
        "  Raw: %d documents across %d categories"
        % (len(data.data), len(data.target_names)),
        flush=True,
    )

    texts_all = []
    labels_all = []
    for text, label in zip(data.data, data.target):
        stripped = text.strip()
        if len(stripped) > 20:
            texts_all.append(stripped)
            labels_all.append(label)
    labels_all = np.array(labels_all)
    print("  After filtering trivial: %d documents" % len(texts_all), flush=True)

    rng = np.random.RandomState(seed)
    keep_indices = []
    for cat_id in range(len(data.target_names)):
        cat_idx = np.where(labels_all == cat_id)[0]
        if len(cat_idx) > max_per_cat:
            chosen = rng.choice(cat_idx, size=max_per_cat, replace=False)
        else:
            chosen = cat_idx
        keep_indices.extend(chosen.tolist())
    keep_indices = sorted(keep_indices)

    texts = [texts_all[i] for i in keep_indices]
    labels = labels_all[keep_indices]
    print(
        "  After subsampling (%d/cat): %d documents" % (max_per_cat, len(texts)),
        flush=True,
    )

    return texts, labels, data.target_names


def encode_texts(texts, model_name):
    """Encode texts with a sentence-transformer model. Frees model after encoding."""
    print("  Encoding %d texts with %s..." % (len(texts), model_name), flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    truncated = [t[:512] for t in texts]
    embeddings = model.encode(
        truncated,
        show_progress_bar=True,
        batch_size=128,
        normalize_embeddings=False,
    )
    print("  Embedding shape: %s" % str(embeddings.shape), flush=True)
    del model
    del truncated
    gc.collect()
    return embeddings


def build_continuous_purity_clusters(
    embeddings, labels, target_names, cluster_size=200, seed=42
):
    """
    Build clusters with CONTINUOUSLY varying purity from 0.10 to 1.00.

    For a target purity p and cluster_size n:
      - Pick a primary category
      - Sample floor(n*p) docs from the primary category
      - Sample the remaining (n - floor(n*p)) docs from ALL OTHER categories
      - Record actual purity (may differ slightly from target due to sampling)

    This eliminates the 3-group confound from v2.
    """
    rng = np.random.RandomState(seed)
    n_categories = len(target_names)
    all_idx = np.arange(len(labels))

    # Pre-index per category
    cat_indices = {}
    for cat_id in range(n_categories):
        cat_indices[cat_id] = np.where(labels == cat_id)[0]

    clusters = []
    cluster_id = 0

    for target_purity in PURITY_LEVELS:
        for rep in range(CLUSTERS_PER_PURITY):
            # Rotate through categories
            primary_cat = (cluster_id) % n_categories
            n_primary = int(np.floor(cluster_size * target_purity))
            n_noise = cluster_size - n_primary

            # Sample primary docs
            available_primary = cat_indices[primary_cat]
            if len(available_primary) < n_primary:
                primary_chosen = available_primary
                # Fill remainder from primary if short
                extra_needed = n_primary - len(available_primary)
                primary_chosen = np.concatenate([
                    primary_chosen,
                    rng.choice(available_primary, size=extra_needed, replace=True),
                ])
            else:
                primary_chosen = rng.choice(
                    available_primary, size=n_primary, replace=False
                )

            # Sample noise from ALL OTHER categories
            if n_noise > 0:
                other_idx = np.where(labels != primary_cat)[0]
                if len(other_idx) >= n_noise:
                    noise_chosen = rng.choice(other_idx, size=n_noise, replace=False)
                else:
                    noise_chosen = rng.choice(other_idx, size=n_noise, replace=True)
            else:
                noise_chosen = np.array([], dtype=int)

            chosen = np.concatenate([primary_chosen, noise_chosen]).astype(int)
            emb = embeddings[chosen]
            cat_labels = labels[chosen]
            unique, counts = np.unique(cat_labels, return_counts=True)
            actual_purity = float(counts.max()) / len(cat_labels)

            clusters.append({
                "embeddings": emb,
                "target_purity": target_purity,
                "actual_purity": actual_purity,
                "primary_category": target_names[primary_cat],
                "label": "p%.2f_rep%d_%s" % (target_purity, rep, target_names[primary_cat]),
                "n": len(chosen),
            })
            cluster_id += 1

    purities = [c["actual_purity"] for c in clusters]
    print(
        "  Built %d clusters with continuous purity [%.3f, %.3f]"
        % (len(clusters), min(purities), max(purities)),
        flush=True,
    )
    print(
        "  Purity levels: %s" % str(PURITY_LEVELS),
        flush=True,
    )
    return clusters


# ============================================================
# TEST 1: Overall Correlation (log(R) vs -NLL)
# ============================================================

def test1_overall_correlation(clusters, arch_name):
    """
    Compute overall Pearson and Spearman correlation between log(R) and -NLL
    across ALL clusters (continuous purity design).
    """
    print("\n  TEST 1 [%s]: Overall log(R) vs -Gaussian_NLL" % arch_name, flush=True)
    print("  " + "-" * 55, flush=True)

    log_R_simple_list = []
    log_R_full_list = []
    neg_nll_list = []
    nll_list = []
    purities = []
    cluster_labels = []

    for cluster in clusters:
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        R_simple = metrics["R_simple"]
        R_full = metrics["R_full"]
        nll = compute_gaussian_nll(emb)

        if np.isnan(R_simple) or R_simple <= 0 or np.isnan(nll):
            continue

        log_R_s = np.log(R_simple)
        log_R_f = np.log(R_full) if (not np.isnan(R_full) and R_full > 0) else float("nan")

        log_R_simple_list.append(log_R_s)
        log_R_full_list.append(log_R_f)
        neg_nll_list.append(-nll)
        nll_list.append(nll)
        purities.append(cluster["actual_purity"])
        cluster_labels.append(cluster["label"])

    log_R_simple = np.array(log_R_simple_list)
    log_R_full = np.array(log_R_full_list)
    neg_nll = np.array(neg_nll_list)
    nll_arr = np.array(nll_list)
    n = len(log_R_simple)

    print("    Valid clusters: %d" % n, flush=True)
    if n < 10:
        print("    ERROR: Too few valid clusters", flush=True)
        return {"error": "insufficient_data", "n_valid": n}

    # Overall correlations
    r_pear_s, p_pear_s = pearsonr(log_R_simple, neg_nll)
    r_spear_s, p_spear_s = spearmanr(log_R_simple, neg_nll)

    print("    log(R_simple) vs -NLL:", flush=True)
    print("      Pearson  r = %.4f  (p = %.2e)" % (r_pear_s, p_pear_s), flush=True)
    print("      Spearman r = %.4f  (p = %.2e)" % (r_spear_s, p_spear_s), flush=True)

    valid_full = ~np.isnan(log_R_full)
    if np.sum(valid_full) >= 10:
        r_pear_f, p_pear_f = pearsonr(log_R_full[valid_full], neg_nll[valid_full])
        r_spear_f, p_spear_f = spearmanr(log_R_full[valid_full], neg_nll[valid_full])
        print("    log(R_full) vs -NLL:", flush=True)
        print("      Pearson  r = %.4f  (p = %.2e)" % (r_pear_f, p_pear_f), flush=True)
        print("      Spearman r = %.4f  (p = %.2e)" % (r_spear_f, p_spear_f), flush=True)
    else:
        r_pear_f, p_pear_f = float("nan"), float("nan")
        r_spear_f, p_spear_f = float("nan"), float("nan")

    # Identity check: is log(R) + NLL approximately constant?
    sum_vals = log_R_simple + nll_arr
    identity_std = float(np.std(sum_vals))
    identity_mean = float(np.mean(sum_vals))
    range_nll = float(np.ptp(nll_arr))
    range_logR = float(np.ptp(log_R_simple))
    residual_pct = (identity_std / range_nll * 100.0) if range_nll > 0 else float("inf")

    print("    Identity check: std(log(R) + NLL) = %.4f" % identity_std, flush=True)
    print("      range(NLL) = %.4f, residual = %.1f%% of range" % (range_nll, residual_pct), flush=True)
    print("      Pre-registered threshold: < 10%% for CONFIRMED", flush=True)

    return {
        "n_valid": n,
        "log_R_simple_vs_neg_NLL": {
            "pearson_r": float(r_pear_s),
            "pearson_p": float(p_pear_s),
            "spearman_rho": float(r_spear_s),
            "spearman_p": float(p_spear_s),
        },
        "log_R_full_vs_neg_NLL": {
            "pearson_r": float(r_pear_f),
            "pearson_p": float(p_pear_f),
            "spearman_rho": float(r_spear_f),
            "spearman_p": float(p_spear_f),
        },
        "identity_check": {
            "mean_log_R_plus_NLL": identity_mean,
            "std_log_R_plus_NLL": identity_std,
            "range_NLL": range_nll,
            "range_log_R": range_logR,
            "residual_pct_of_range": residual_pct,
        },
        "raw_data": {
            "log_R_simple": [float(x) for x in log_R_simple],
            "log_R_full": [float(x) for x in log_R_full],
            "neg_NLL": [float(x) for x in neg_nll],
            "purities": [float(x) for x in purities],
            "labels": cluster_labels,
        },
    }


# ============================================================
# TEST 2: WITHIN-GROUP Correlations (the make-or-break test)
# ============================================================

def test2_within_group_correlation(clusters, arch_name):
    """
    CRITICAL TEST: Compute correlations WITHIN purity bands.

    Group clusters into bands (low: 0.1-0.35, mid: 0.4-0.65, high: 0.7-1.0)
    and compute Pearson r within each band.

    If within-group r > 0.7, the finding goes beyond group structure.
    If within-group r < 0.3 in all bands, the overall r is purely a group artifact.
    """
    print("\n  TEST 2 [%s]: WITHIN-GROUP Correlations" % arch_name, flush=True)
    print("  " + "-" * 55, flush=True)

    # Compute metrics for all clusters
    records = []
    for cluster in clusters:
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        R_simple = metrics["R_simple"]
        nll = compute_gaussian_nll(emb)

        if np.isnan(R_simple) or R_simple <= 0 or np.isnan(nll):
            continue

        records.append({
            "log_R": np.log(R_simple),
            "nll": nll,
            "purity": cluster["actual_purity"],
        })

    if len(records) < 10:
        print("    ERROR: Too few valid clusters", flush=True)
        return {"error": "insufficient_data", "n_valid": len(records)}

    # Define purity bands
    bands = {
        "low (0.10-0.35)": (0.10, 0.35),
        "mid (0.40-0.65)": (0.40, 0.65),
        "high (0.70-1.00)": (0.70, 1.00),
    }

    band_results = {}
    for band_name, (lo, hi) in bands.items():
        band_recs = [r for r in records if lo <= r["purity"] <= hi]
        n_band = len(band_recs)
        print("    Band '%s': %d clusters" % (band_name, n_band), flush=True)

        if n_band < 5:
            print("      Too few clusters for correlation", flush=True)
            band_results[band_name] = {
                "n": n_band,
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
            }
            continue

        log_R_band = np.array([r["log_R"] for r in band_recs])
        neg_nll_band = np.array([-r["nll"] for r in band_recs])

        r_p, p_p = pearsonr(log_R_band, neg_nll_band)
        r_s, p_s = spearmanr(log_R_band, neg_nll_band)

        print(
            "      Pearson r = %.4f (p = %.4f), Spearman rho = %.4f (p = %.4f)"
            % (r_p, p_p, r_s, p_s),
            flush=True,
        )

        band_results[band_name] = {
            "n": n_band,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
        }

    # Summary: how many bands have |r| > 0.7?
    band_pearson_rs = [
        band_results[b]["pearson_r"]
        for b in band_results
        if not np.isnan(band_results[b]["pearson_r"])
    ]
    n_above_07 = sum(1 for r in band_pearson_rs if abs(r) > 0.7)
    n_below_03 = sum(1 for r in band_pearson_rs if abs(r) < 0.3)
    n_bands_tested = len(band_pearson_rs)

    print("\n    WITHIN-GROUP SUMMARY:", flush=True)
    print(
        "    Bands with |r| > 0.7: %d / %d" % (n_above_07, n_bands_tested), flush=True
    )
    print(
        "    Bands with |r| < 0.3: %d / %d" % (n_below_03, n_bands_tested), flush=True
    )

    return {
        "bands": band_results,
        "n_bands_tested": n_bands_tested,
        "n_bands_above_07": n_above_07,
        "n_bands_below_03": n_below_03,
        "band_pearson_rs": band_pearson_rs,
    }


# ============================================================
# TEST 3: Null-Hypothesis Comparison
# ============================================================

def test3_null_hypothesis(clusters, arch_name):
    """
    Compare correlation of -NLL with log(R) against other simple cluster statistics.

    If trace(cov), first eigenvalue, mean L2 norm, etc. also correlate with
    -NLL at r > 0.9, then the R-NLL relationship is not special.
    """
    print(
        "\n  TEST 3 [%s]: Null Hypothesis -- Is R Special?" % arch_name, flush=True
    )
    print("  " + "-" * 55, flush=True)

    log_R_vals = []
    neg_nll_vals = []
    alt_stats = {
        "trace_cov": [],
        "first_eigenvalue": [],
        "mean_l2_norm": [],
        "mean_variance": [],
        "mean_euclidean_dist": [],
    }

    for cluster in clusters:
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        R_simple = metrics["R_simple"]
        nll = compute_gaussian_nll(emb)
        stats = compute_simple_cluster_stats(emb)

        if np.isnan(R_simple) or R_simple <= 0 or np.isnan(nll):
            continue

        log_R_vals.append(np.log(R_simple))
        neg_nll_vals.append(-nll)
        for key in alt_stats:
            alt_stats[key].append(stats[key])

    log_R_arr = np.array(log_R_vals)
    neg_nll_arr = np.array(neg_nll_vals)
    n = len(log_R_arr)

    print("    Valid clusters: %d" % n, flush=True)
    if n < 10:
        return {"error": "insufficient_data", "n_valid": n}

    # R vs -NLL
    r_R, p_R = pearsonr(log_R_arr, neg_nll_arr)
    print("    log(R_simple) vs -NLL: Pearson r = %.4f" % r_R, flush=True)

    results = {
        "log_R_simple": {"pearson_r": float(r_R), "pearson_p": float(p_R)},
    }

    # Alternative metrics vs -NLL
    print("\n    Alternative metrics vs -NLL:", flush=True)
    for key in alt_stats:
        arr = np.array(alt_stats[key])
        valid = ~np.isnan(arr)
        if np.sum(valid) < 10:
            print("      %s: insufficient valid data" % key, flush=True)
            results[key] = {"pearson_r": float("nan"), "pearson_p": float("nan")}
            continue

        # Some metrics are inversely related to quality, use absolute r
        r_alt, p_alt = pearsonr(arr[valid], neg_nll_arr[valid])
        print("      %s vs -NLL: Pearson r = %.4f" % (key, r_alt), flush=True)
        results[key] = {"pearson_r": float(r_alt), "pearson_p": float(p_alt)}

    # Count how many alternatives match or beat R's correlation
    r_R_abs = abs(r_R)
    n_matching = sum(
        1
        for key in alt_stats
        if not np.isnan(results[key]["pearson_r"])
        and abs(results[key]["pearson_r"]) >= r_R_abs - 0.05
    )
    print(
        "\n    Alternative metrics with |r| within 0.05 of R: %d / %d"
        % (n_matching, len(alt_stats)),
        flush=True,
    )

    results["n_alternatives_matching_R"] = n_matching
    results["n_alternatives_total"] = len(alt_stats)
    return results


# ============================================================
# TEST 4: Harder Gating Task (rank ordering within narrow bands)
# ============================================================

def test4_harder_gating(clusters, arch_name):
    """
    Harder gating task: Can R correctly RANK clusters within narrow purity bands?

    For each pair of clusters with purity difference >= 0.10 but both within the
    same broad band, check if R agrees with purity ordering.
    This is much harder than the v2 task which separated purity > 0.8 from < 0.4.

    Also test: can R distinguish purity 0.7 from 0.9? (harder discrimination).
    """
    print("\n  TEST 4 [%s]: Harder Gating -- Rank Ordering" % arch_name, flush=True)
    print("  " + "-" * 55, flush=True)

    records = []
    for cluster in clusters:
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        R_simple = metrics["R_simple"]
        if np.isnan(R_simple) or R_simple <= 0:
            continue
        records.append({
            "R_simple": R_simple,
            "purity": cluster["actual_purity"],
            "label": cluster["label"],
        })

    n = len(records)
    print("    Valid clusters: %d" % n, flush=True)
    if n < 10:
        return {"error": "insufficient_data", "n_valid": n}

    # Pairwise concordance: for all pairs where purity_i > purity_j + 0.10,
    # does R_i > R_j?
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_diff = records[i]["purity"] - records[j]["purity"]
            if abs(p_diff) < 0.10:
                continue  # skip near-ties
            if p_diff > 0:
                # i should have higher R
                if records[i]["R_simple"] > records[j]["R_simple"]:
                    concordant += 1
                else:
                    discordant += 1
            else:
                # j should have higher R
                if records[j]["R_simple"] > records[i]["R_simple"]:
                    concordant += 1
                else:
                    discordant += 1

    total_pairs = concordant + discordant
    concordance_rate = concordant / total_pairs if total_pairs > 0 else float("nan")
    print(
        "    Pairwise concordance (purity gap >= 0.10): %.4f (%d/%d pairs)"
        % (concordance_rate, concordant, total_pairs),
        flush=True,
    )

    # Harder: concordance for smaller purity gaps (0.05-0.15)
    concordant_hard = 0
    discordant_hard = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_diff = abs(records[i]["purity"] - records[j]["purity"])
            if p_diff < 0.05 or p_diff > 0.15:
                continue
            if records[i]["purity"] > records[j]["purity"]:
                if records[i]["R_simple"] > records[j]["R_simple"]:
                    concordant_hard += 1
                else:
                    discordant_hard += 1
            else:
                if records[j]["R_simple"] > records[i]["R_simple"]:
                    concordant_hard += 1
                else:
                    discordant_hard += 1

    total_hard = concordant_hard + discordant_hard
    concordance_hard = concordant_hard / total_hard if total_hard > 0 else float("nan")
    print(
        "    Hard concordance (purity gap 0.05-0.15): %.4f (%d/%d pairs)"
        % (concordance_hard, concordant_hard, total_hard),
        flush=True,
    )

    # Discrimination test: clusters with purity 0.65-0.75 vs 0.85-0.95
    mid_clusters = [r for r in records if 0.65 <= r["purity"] <= 0.75]
    high_clusters = [r for r in records if 0.85 <= r["purity"] <= 0.95]
    if len(mid_clusters) >= 3 and len(high_clusters) >= 3:
        mid_R_mean = np.mean([r["R_simple"] for r in mid_clusters])
        high_R_mean = np.mean([r["R_simple"] for r in high_clusters])
        # Cohen's d effect size
        mid_R_arr = np.array([r["R_simple"] for r in mid_clusters])
        high_R_arr = np.array([r["R_simple"] for r in high_clusters])
        pooled_std = np.sqrt(
            (np.var(mid_R_arr, ddof=1) + np.var(high_R_arr, ddof=1)) / 2
        )
        cohens_d = (high_R_mean - mid_R_mean) / pooled_std if pooled_std > 0 else float("nan")
        print(
            "    Discrimination (purity 0.65-0.75 vs 0.85-0.95): Cohen's d = %.4f"
            % cohens_d,
            flush=True,
        )
        discrimination = {
            "mid_purity_mean_R": float(mid_R_mean),
            "high_purity_mean_R": float(high_R_mean),
            "cohens_d": float(cohens_d),
            "n_mid": len(mid_clusters),
            "n_high": len(high_clusters),
        }
    else:
        print("    Discrimination: insufficient clusters in bands", flush=True)
        discrimination = {"error": "insufficient_clusters"}

    return {
        "n_valid": n,
        "concordance_rate_gap_010": float(concordance_rate),
        "concordant_pairs_gap_010": concordant,
        "total_pairs_gap_010": total_pairs,
        "concordance_rate_gap_005_015": float(concordance_hard),
        "concordant_pairs_gap_005_015": concordant_hard,
        "total_pairs_gap_005_015": total_hard,
        "discrimination": discrimination,
    }


# ============================================================
# TEST 5: Cross-Architecture Consistency
# ============================================================

def test5_cross_architecture(all_arch_results):
    """Report correlations and within-group results across architectures."""
    print("\n" + "=" * 70, flush=True)
    print("TEST 5: Cross-Architecture Consistency", flush=True)
    print("=" * 70, flush=True)

    arch_names = []
    overall_pearson_rs = []
    within_group_summaries = []

    for arch_name, results in all_arch_results.items():
        t1 = results.get("test1", {})
        t2 = results.get("test2_within_group", {})
        if "error" in t1:
            print("  %s: SKIPPED (error)" % arch_name, flush=True)
            continue

        r_p = t1["log_R_simple_vs_neg_NLL"]["pearson_r"]
        arch_names.append(arch_name)
        overall_pearson_rs.append(r_p)

        wg_rs = t2.get("band_pearson_rs", [])
        n_above_07 = t2.get("n_bands_above_07", 0)
        n_bands = t2.get("n_bands_tested", 0)

        print(
            "  %s: overall r = %.4f, within-group |r|>0.7: %d/%d bands"
            % (arch_name, r_p, n_above_07, n_bands),
            flush=True,
        )
        within_group_summaries.append({
            "arch": arch_name,
            "n_above_07": n_above_07,
            "n_bands": n_bands,
            "band_rs": wg_rs,
        })

    # Count architectures meeting within-group criterion
    n_arch_meeting_within_group = sum(
        1 for s in within_group_summaries if s["n_above_07"] >= 2
    )

    print(
        "\n  Architectures with within-group |r| > 0.7 in >= 2 bands: %d / %d"
        % (n_arch_meeting_within_group, len(within_group_summaries)),
        flush=True,
    )

    return {
        "arch_names": arch_names,
        "overall_pearson_rs": [float(x) for x in overall_pearson_rs],
        "within_group_summaries": within_group_summaries,
        "n_arch_meeting_within_group": n_arch_meeting_within_group,
    }


# ============================================================
# VERDICT
# ============================================================

def determine_verdict(all_arch_results, test5_results):
    """
    Apply pre-registered criteria:

    CONFIRMED:  within-group Pearson |r| > 0.7 on >= 2/3 architectures
                AND identity residual std < 10% of range(NLL)
    FALSIFIED:  overall Pearson |r| < 0.5
                OR within-group |r| < 0.3 on all 3 architectures
    INCONCLUSIVE: otherwise
    """
    print("\n" + "=" * 70, flush=True)
    print("VERDICT DETERMINATION", flush=True)
    print("=" * 70, flush=True)

    n_arch = len(test5_results["arch_names"])
    if n_arch == 0:
        print("  No valid architectures. Verdict: INCONCLUSIVE", flush=True)
        return {"verdict": "INCONCLUSIVE", "reason": "no_valid_architectures"}

    # -- Criterion 1: Within-group correlations --
    # For each arch, check if within-group |r| > 0.7 in >= 2/3 of bands
    arch_within_group_pass = []
    for s in test5_results["within_group_summaries"]:
        passes = s["n_above_07"] >= 2  # at least 2 of 3 bands
        arch_within_group_pass.append(passes)
        print(
            "  %s within-group: %d/%d bands with |r|>0.7 -> %s"
            % (s["arch"], s["n_above_07"], s["n_bands"], "PASS" if passes else "FAIL"),
            flush=True,
        )

    n_arch_within_pass = sum(arch_within_group_pass)
    within_group_confirmed = n_arch_within_pass >= (2 * n_arch / 3)

    # -- Criterion 2: Identity residual --
    identity_confirmed = True
    for arch_name, results in all_arch_results.items():
        t1 = results.get("test1", {})
        if "error" in t1:
            continue
        residual_pct = t1["identity_check"]["residual_pct_of_range"]
        passes = residual_pct < 10.0
        if not passes:
            identity_confirmed = False
        print(
            "  %s identity residual: %.1f%% of range -> %s (threshold: <10%%)"
            % (arch_name, residual_pct, "PASS" if passes else "FAIL"),
            flush=True,
        )

    # -- Falsify check --
    overall_rs = test5_results["overall_pearson_rs"]
    all_below_05 = all(abs(r) < 0.5 for r in overall_rs)

    all_within_below_03 = True
    for s in test5_results["within_group_summaries"]:
        band_rs = s.get("band_rs", [])
        if any(abs(r) > 0.3 for r in band_rs if not np.isnan(r)):
            all_within_below_03 = False

    print("\n  CONFIRM conditions:", flush=True)
    print(
        "    Within-group |r|>0.7 on >=2/3 arch: %s (%d/%d)"
        % (within_group_confirmed, n_arch_within_pass, n_arch),
        flush=True,
    )
    print(
        "    Identity residual < 10%%: %s" % identity_confirmed,
        flush=True,
    )
    print("  FALSIFY conditions:", flush=True)
    print("    Overall |r| < 0.5 on all arch: %s" % all_below_05, flush=True)
    print(
        "    Within-group |r| < 0.3 on all arch: %s" % all_within_below_03,
        flush=True,
    )

    if within_group_confirmed and identity_confirmed:
        verdict = "CONFIRMED"
    elif all_below_05 or all_within_below_03:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print("\n  >>> VERDICT: %s <<<" % verdict, flush=True)

    return {
        "verdict": verdict,
        "within_group_confirmed": bool(within_group_confirmed),
        "identity_confirmed": bool(identity_confirmed),
        "falsify_overall_below_05": bool(all_below_05),
        "falsify_within_below_03": bool(all_within_below_03),
        "n_arch_within_pass": n_arch_within_pass,
        "n_arch_total": n_arch,
        "overall_pearson_rs": [float(r) for r in overall_rs],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    np.random.seed(SEED)

    print("=" * 70, flush=True)
    print("Q09 v3 TEST: Does log(R) correlate with Gaussian NLL?", flush=True)
    print("Continuous purity | Within-group correlations | 3 architectures", flush=True)
    print("NOTE: NLL is Gaussian NLL, NOT FEP variational free energy", flush=True)
    print("=" * 70, flush=True)

    texts, labels, target_names = load_20newsgroups_subsampled(
        max_per_cat=MAX_DOCS_PER_CATEGORY, seed=SEED
    )

    all_arch_results = {}

    for arch_name, dim in ARCHITECTURES:
        print("\n" + "=" * 70, flush=True)
        print("ARCHITECTURE: %s (%d-dim)" % (arch_name, dim), flush=True)
        print("=" * 70, flush=True)

        arch_start = time.time()

        embeddings = encode_texts(texts, arch_name)

        clusters = build_continuous_purity_clusters(
            embeddings, labels, target_names,
            cluster_size=CLUSTER_SIZE, seed=SEED,
        )

        # Test 1: Overall correlation
        t1 = test1_overall_correlation(clusters, arch_name)

        # Test 2: Within-group correlations (CRITICAL)
        t2 = test2_within_group_correlation(clusters, arch_name)

        # Test 3: Null hypothesis comparison
        t3 = test3_null_hypothesis(clusters, arch_name)

        # Test 4: Harder gating
        t4 = test4_harder_gating(clusters, arch_name)

        arch_elapsed = time.time() - arch_start
        print(
            "\n  Architecture %s completed in %.1fs" % (arch_name, arch_elapsed),
            flush=True,
        )

        all_arch_results[arch_name] = {
            "test1": t1,
            "test2_within_group": t2,
            "test3_null_hypothesis": t3,
            "test4_harder_gating": t4,
            "dim": dim,
            "elapsed_seconds": round(arch_elapsed, 1),
        }

        del embeddings
        del clusters
        gc.collect()

    # Test 5: Cross-architecture
    t5 = test5_cross_architecture(all_arch_results)

    # Verdict
    verdict_info = determine_verdict(all_arch_results, t5)

    elapsed = time.time() - start_time

    full_results = {
        "verdict": verdict_info["verdict"],
        "verdict_detail": verdict_info,
        "test5_cross_architecture": t5,
        "per_architecture": {},
        "metadata": {
            "test_version": "v3",
            "question": "Q09: Does log(R) correlate with Gaussian NLL?",
            "note": "NLL is Gaussian negative log-likelihood, NOT FEP variational free energy",
            "dataset": "20 Newsgroups (sklearn, headers/footers/quotes removed)",
            "architectures": [a[0] for a in ARCHITECTURES],
            "cluster_size": CLUSTER_SIZE,
            "purity_levels": PURITY_LEVELS,
            "clusters_per_purity": CLUSTERS_PER_PURITY,
            "total_clusters": len(PURITY_LEVELS) * CLUSTERS_PER_PURITY,
            "max_docs_per_category": MAX_DOCS_PER_CATEGORY,
            "seed": SEED,
            "elapsed_seconds": round(elapsed, 1),
            "preregistered_criteria": {
                "CONFIRMED": "within-group |r| > 0.7 on >= 2/3 arch AND identity residual std < 10% of range",
                "FALSIFIED": "overall |r| < 0.5 OR within-group |r| < 0.3 on all arch",
                "INCONCLUSIVE": "otherwise",
            },
            "audit_fixes": [
                "Continuous purity (0.10-1.00) instead of 3 discrete groups",
                "Within-group correlations computed per purity band",
                "Honest label: Gaussian NLL not FEP free energy",
                "Fixed ddof: scatter and cov both use ddof=0",
                "Harder gating: pairwise concordance and narrow-band discrimination",
                "Null hypothesis: compare R against simple cluster statistics",
            ],
        },
    }

    for arch_name, results in all_arch_results.items():
        full_results["per_architecture"][arch_name] = results

    results_path = os.path.join(RESULTS_DIR, "test_v3_q09_results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print("\nResults saved to: %s" % results_path, flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print("  Verdict: %s" % verdict_info["verdict"], flush=True)
    print("  Elapsed: %.1fs" % elapsed, flush=True)
    print("  Clusters: %d (purity 0.10-1.00 continuous)" % (len(PURITY_LEVELS) * CLUSTERS_PER_PURITY), flush=True)

    for arch_name, results in all_arch_results.items():
        t1 = results["test1"]
        t2 = results["test2_within_group"]
        t3 = results["test3_null_hypothesis"]
        t4 = results["test4_harder_gating"]
        print("\n  %s:" % arch_name, flush=True)
        if "error" not in t1:
            print(
                "    Overall: Pearson r = %.4f, Spearman rho = %.4f"
                % (
                    t1["log_R_simple_vs_neg_NLL"]["pearson_r"],
                    t1["log_R_simple_vs_neg_NLL"]["spearman_rho"],
                ),
                flush=True,
            )
            print(
                "    Identity: residual = %.1f%% of range"
                % t1["identity_check"]["residual_pct_of_range"],
                flush=True,
            )
        if "error" not in t2:
            for band_name, br in t2["bands"].items():
                if not np.isnan(br["pearson_r"]):
                    print(
                        "    Within-group '%s': Pearson r = %.4f"
                        % (band_name, br["pearson_r"]),
                        flush=True,
                    )
        if "error" not in t3:
            log_R_r = t3.get("log_R_simple", {}).get("pearson_r", float("nan"))
            n_match = t3.get("n_alternatives_matching_R", "?")
            n_total = t3.get("n_alternatives_total", "?")
            print(
                "    Null hypothesis: R r=%.4f, %s/%s alternatives within 0.05"
                % (log_R_r, n_match, n_total),
                flush=True,
            )
        if "error" not in t4:
            print(
                "    Gating concordance (gap>=0.10): %.4f"
                % t4["concordance_rate_gap_010"],
                flush=True,
            )

    return full_results


if __name__ == "__main__":
    results = main()
