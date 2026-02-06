"""
Q09 v2 FIXED Test: Does log(R) = -F + const?
Is R-maximization equivalent to surprise minimization (Free Energy Principle)?

Improvements over v1 test:
  - 20 Newsgroups (natural topic clusters) instead of STS-B similarity bins
  - 3 architectures instead of 1
  - Purity-based quality labels (topically pure vs mixed)
  - Rank-order consistency test
  - Cross-architecture consistency test

Pre-registered criteria (adapted from README):
  CONFIRM: corr(log(R), -F) > 0.5 in >= 2/3 architectures
           AND R-gating outperforms 1/variance by >= 10%
  FALSIFY: corr(log(R), -F) < 0.3 in ALL architectures
           OR R-gating is worse than 1/variance
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import shared formula via importlib to avoid path issues
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "formula",
    os.path.join(os.path.dirname(__file__), "..", "..", "shared", "formula.py"),
)
formula = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(formula)
compute_E = formula.compute_E
compute_grad_S = formula.compute_grad_S
compute_R_simple = formula.compute_R_simple
compute_R_full = formula.compute_R_full
compute_all = formula.compute_all

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Architecture configs
ARCHITECTURES = [
    ("all-MiniLM-L6-v2", 384),
    ("all-mpnet-base-v2", 768),
    ("multi-qa-MiniLM-L6-cos-v1", 384),
]

CLUSTER_SIZE = 200    # docs per cluster
N_MIXED = 20          # number of random-mixed clusters
N_DEGRADED = 20       # number of degraded clusters
SEED = 42
MAX_DOCS_PER_CATEGORY = 250  # subsample before encoding for speed


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def regularize_covariance(cov, reg=1e-4):
    """Add regularization to avoid singular covariance matrices."""
    return cov + reg * np.eye(cov.shape[0])


def compute_variational_free_energy(embeddings, reg=1e-4):
    """
    Compute variational free energy under a fitted Gaussian generative model.

    F = 0.5 * (d * log(2*pi) + log(det(cov + reg*I)) + trace(inv(cov + reg*I) @ scatter))
    where scatter = centered.T @ centered / n
    """
    n, d = embeddings.shape
    if n < 3:
        return float("nan")

    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    scatter = (centered.T @ centered) / n

    cov = np.cov(embeddings, rowvar=False)
    cov_reg = regularize_covariance(cov, reg)

    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        return float("nan")

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return float("nan")

    trace_term = np.trace(cov_inv @ scatter)

    F = 0.5 * (d * np.log(2 * np.pi) + logdet + trace_term)
    return float(F)


def compute_embedding_variance(embeddings):
    """Mean variance across embedding dimensions."""
    return float(np.mean(np.var(embeddings, axis=0)))


# ============================================================
# DATA LOADING AND CLUSTER CONSTRUCTION
# ============================================================

def load_20newsgroups_subsampled(max_per_cat=250, seed=42):
    """
    Load 20 Newsgroups, subsample to max_per_cat documents per category.
    """
    print("Loading 20 Newsgroups dataset...", flush=True)
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    print(f"  Raw: {len(data.data)} documents across {len(data.target_names)} categories",
          flush=True)

    # Filter empty/trivial documents
    texts_all = []
    labels_all = []
    for text, label in zip(data.data, data.target):
        stripped = text.strip()
        if len(stripped) > 20:
            texts_all.append(stripped)
            labels_all.append(label)
    labels_all = np.array(labels_all)
    print(f"  After filtering trivial: {len(texts_all)} documents", flush=True)

    # Subsample per category
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
    print(f"  After subsampling ({max_per_cat}/cat): {len(texts)} documents", flush=True)

    return texts, labels, data.target_names


def encode_texts(texts, model_name):
    """Encode texts with a given sentence-transformer model. Frees model after encoding."""
    print(f"  Encoding {len(texts)} texts with {model_name}...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # Truncate long texts to first 512 chars to reduce memory
    truncated = [t[:512] for t in texts]
    embeddings = model.encode(
        truncated,
        show_progress_bar=True,
        batch_size=128,
        normalize_embeddings=False,
    )
    print(f"  Embedding shape: {embeddings.shape}", flush=True)
    # Free model memory
    del model
    del truncated
    gc.collect()
    return embeddings


def build_clusters(embeddings, labels, target_names, cluster_size=200, seed=42):
    """
    Build 3 types of clusters from 20 Newsgroups:
      1. Pure clusters: cluster_size docs from a single newsgroup (20 clusters)
      2. Mixed clusters: cluster_size random docs from all newsgroups (20 clusters)
      3. Degraded clusters: half pure + half random (20 clusters)
    """
    rng = np.random.RandomState(seed)
    clusters = []
    n_categories = len(target_names)
    all_idx = np.arange(len(labels))

    # Pure clusters
    for cat_id in range(n_categories):
        idx = np.where(labels == cat_id)[0]
        if len(idx) < cluster_size:
            chosen = idx
        else:
            chosen = rng.choice(idx, size=cluster_size, replace=False)
        emb = embeddings[chosen]
        cat_labels = labels[chosen]
        unique, counts = np.unique(cat_labels, return_counts=True)
        purity = float(counts.max()) / len(cat_labels)
        clusters.append({
            "embeddings": emb,
            "purity": purity,
            "type": "pure",
            "label": target_names[cat_id],
            "n": len(chosen),
        })

    # Mixed clusters
    for i in range(N_MIXED):
        chosen = rng.choice(all_idx, size=cluster_size, replace=False)
        emb = embeddings[chosen]
        cat_labels = labels[chosen]
        unique, counts = np.unique(cat_labels, return_counts=True)
        purity = float(counts.max()) / len(cat_labels)
        clusters.append({
            "embeddings": emb,
            "purity": purity,
            "type": "mixed",
            "label": f"mixed_{i}",
            "n": cluster_size,
        })

    # Degraded clusters
    for i in range(N_DEGRADED):
        cat_id = i % n_categories
        cat_idx = np.where(labels == cat_id)[0]
        n_pure_half = cluster_size // 2
        n_random_half = cluster_size - n_pure_half
        if len(cat_idx) < n_pure_half:
            pure_chosen = cat_idx
        else:
            pure_chosen = rng.choice(cat_idx, size=n_pure_half, replace=False)
        random_chosen = rng.choice(all_idx, size=n_random_half, replace=False)
        chosen = np.concatenate([pure_chosen, random_chosen])
        emb = embeddings[chosen]
        cat_labels = labels[chosen]
        unique, counts = np.unique(cat_labels, return_counts=True)
        purity = float(counts.max()) / len(cat_labels)
        clusters.append({
            "embeddings": emb,
            "purity": purity,
            "type": "degraded",
            "label": f"degraded_{target_names[cat_id]}_{i}",
            "n": len(chosen),
        })

    print(f"  Built {len(clusters)} clusters: "
          f"{n_categories} pure, {N_MIXED} mixed, {N_DEGRADED} degraded", flush=True)
    purities = [c["purity"] for c in clusters]
    print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]", flush=True)
    return clusters


# ============================================================
# TEST 1: log(R) vs -F Correlation
# ============================================================

def test1_correlation(clusters, arch_name):
    """
    For each cluster, compute R_simple, R_full, F.
    Correlate log(R) with -F across all 60 clusters.
    """
    print(f"\n  TEST 1 [{arch_name}]: log(R) vs -F Correlation", flush=True)
    print(f"  {'-'*50}", flush=True)

    log_R_simple_list = []
    log_R_full_list = []
    neg_F_list = []
    F_list = []
    purities = []
    types = []
    cluster_labels = []

    for i, cluster in enumerate(clusters):
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        R_simple = metrics["R_simple"]
        R_full = metrics["R_full"]
        F = compute_variational_free_energy(emb, reg=1e-4)

        if np.isnan(R_simple) or R_simple <= 0 or np.isnan(F):
            continue

        log_R_s = np.log(R_simple)
        log_R_f = np.log(R_full) if (not np.isnan(R_full) and R_full > 0) else float("nan")

        log_R_simple_list.append(log_R_s)
        log_R_full_list.append(log_R_f)
        neg_F_list.append(-F)
        F_list.append(F)
        purities.append(cluster["purity"])
        types.append(cluster["type"])
        cluster_labels.append(cluster["label"])

    log_R_simple = np.array(log_R_simple_list)
    log_R_full = np.array(log_R_full_list)
    neg_F = np.array(neg_F_list)
    F_arr = np.array(F_list)
    n = len(log_R_simple)

    print(f"    Valid clusters: {n}", flush=True)
    if n < 10:
        print("    ERROR: Too few valid clusters", flush=True)
        return {"error": "insufficient_data", "n_valid": n}

    r_pear_s, p_pear_s = pearsonr(log_R_simple, neg_F)
    r_spear_s, p_spear_s = spearmanr(log_R_simple, neg_F)

    print(f"    log(R_simple) vs -F:", flush=True)
    print(f"      Pearson  r = {r_pear_s:.4f}  (p = {p_pear_s:.6f})", flush=True)
    print(f"      Spearman r = {r_spear_s:.4f}  (p = {p_spear_s:.6f})", flush=True)

    valid_full = ~np.isnan(log_R_full)
    if np.sum(valid_full) >= 10:
        r_pear_f, p_pear_f = pearsonr(log_R_full[valid_full], neg_F[valid_full])
        r_spear_f, p_spear_f = spearmanr(log_R_full[valid_full], neg_F[valid_full])
        print(f"    log(R_full) vs -F:", flush=True)
        print(f"      Pearson  r = {r_pear_f:.4f}  (p = {p_pear_f:.6f})", flush=True)
        print(f"      Spearman r = {r_spear_f:.4f}  (p = {p_spear_f:.6f})", flush=True)
    else:
        r_pear_f, p_pear_f = float("nan"), float("nan")
        r_spear_f, p_spear_f = float("nan"), float("nan")

    sum_vals = log_R_simple + F_arr
    identity_std = float(np.std(sum_vals))
    identity_mean = float(np.mean(sum_vals))
    range_logR = float(np.ptp(log_R_simple))
    range_F = float(np.ptp(F_arr))
    print(f"    Identity check: std(log(R)+F) = {identity_std:.4f}", flush=True)
    print(f"      range(log(R)) = {range_logR:.4f}, range(F) = {range_F:.4f}", flush=True)

    return {
        "n_valid": n,
        "log_R_simple_vs_neg_F": {
            "pearson_r": float(r_pear_s),
            "pearson_p": float(p_pear_s),
            "spearman_rho": float(r_spear_s),
            "spearman_p": float(p_spear_s),
        },
        "log_R_full_vs_neg_F": {
            "pearson_r": float(r_pear_f),
            "pearson_p": float(p_pear_f),
            "spearman_rho": float(r_spear_f),
            "spearman_p": float(p_spear_f),
        },
        "identity_check": {
            "mean_log_R_plus_F": identity_mean,
            "std_log_R_plus_F": identity_std,
            "range_log_R": range_logR,
            "range_F": range_F,
        },
        "raw_data": {
            "log_R_simple": [float(x) for x in log_R_simple],
            "log_R_full": [float(x) for x in log_R_full],
            "neg_F": [float(x) for x in neg_F],
            "purities": [float(x) for x in purities],
            "types": types,
            "labels": cluster_labels,
        },
    }


# ============================================================
# TEST 2: R-Gating vs Alternatives
# ============================================================

def test2_gating(clusters, arch_name):
    """
    High quality = purity > 0.8, low quality = purity < 0.4.
    Compare R_simple, R_full, E alone, 1/variance, 1/grad_S.
    """
    print(f"\n  TEST 2 [{arch_name}]: R-Gating vs Alternatives", flush=True)
    print(f"  {'-'*50}", flush=True)

    high_quality = [c for c in clusters if c["purity"] > 0.8]
    low_quality = [c for c in clusters if c["purity"] < 0.4]

    print(f"    High quality (purity > 0.8): {len(high_quality)}", flush=True)
    print(f"    Low quality (purity < 0.4): {len(low_quality)}", flush=True)

    if len(high_quality) < 3 or len(low_quality) < 3:
        print("    WARNING: Insufficient clusters for gating test", flush=True)
        return {"error": "insufficient_clusters",
                "n_high": len(high_quality), "n_low": len(low_quality)}

    labeled = high_quality + low_quality
    true_labels = np.array([1] * len(high_quality) + [0] * len(low_quality))

    R_simple_vals = []
    R_full_vals = []
    E_vals = []
    inv_var_vals = []
    inv_grad_S_vals = []

    for cluster in labeled:
        emb = cluster["embeddings"]
        metrics = compute_all(emb)
        var = compute_embedding_variance(emb)

        R_s = metrics["R_simple"] if not np.isnan(metrics["R_simple"]) else 0.0
        R_f = metrics["R_full"] if not np.isnan(metrics["R_full"]) else 0.0
        E_val = metrics["E"] if not np.isnan(metrics["E"]) else 0.0
        gs = metrics["grad_S"] if not np.isnan(metrics["grad_S"]) else 1e10
        iv = 1.0 / var if var > 1e-10 else 0.0
        igs = 1.0 / gs if gs > 1e-10 else 0.0

        R_simple_vals.append(R_s)
        R_full_vals.append(R_f)
        E_vals.append(E_val)
        inv_var_vals.append(iv)
        inv_grad_S_vals.append(igs)

    R_simple_vals = np.array(R_simple_vals)
    R_full_vals = np.array(R_full_vals)
    E_vals = np.array(E_vals)
    inv_var_vals = np.array(inv_var_vals)
    inv_grad_S_vals = np.array(inv_grad_S_vals)

    def find_best_f1(values, true_lab):
        best_f1 = -1
        best_metrics = None
        for pct in range(5, 96, 1):
            threshold = np.percentile(values, pct)
            preds = (values > threshold).astype(int)
            if len(np.unique(preds)) < 2:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                true_lab, preds, average="binary", zero_division=0
            )
            acc = float(accuracy_score(true_lab, preds))
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "accuracy": acc,
                    "threshold_pct": pct,
                }
        return best_metrics

    methods = {
        "R_simple": R_simple_vals,
        "R_full": R_full_vals,
        "E_alone": E_vals,
        "1/variance": inv_var_vals,
        "1/grad_S": inv_grad_S_vals,
    }

    method_results = {}
    print(f"    {'Method':<15} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8}", flush=True)
    print(f"    {'-'*47}", flush=True)

    for name, vals in methods.items():
        best = find_best_f1(vals, true_labels)
        method_results[name] = best
        if best:
            print(f"    {name:<15} {best['precision']:>8.4f} {best['recall']:>8.4f} "
                  f"{best['f1']:>8.4f} {best['accuracy']:>8.4f}", flush=True)
        else:
            print(f"    {name:<15} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}",
                  flush=True)

    # Random baseline
    random_f1s = []
    rng = np.random.RandomState(SEED)
    for _ in range(1000):
        preds = rng.randint(0, 2, size=len(true_labels))
        _, _, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average="binary", zero_division=0
        )
        random_f1s.append(f1)
    random_f1 = float(np.mean(random_f1s))
    method_results["random"] = {"f1": random_f1}
    print(f"    {'random':<15} {'':>8} {'':>8} {random_f1:>8.4f}", flush=True)

    r_f1 = method_results["R_simple"]["f1"] if method_results["R_simple"] else 0.0
    var_f1 = method_results["1/variance"]["f1"] if method_results["1/variance"] else 0.0
    if var_f1 > 0:
        advantage_pct = ((r_f1 - var_f1) / var_f1) * 100.0
    else:
        advantage_pct = float("inf") if r_f1 > 0 else 0.0
    print(f"\n    R_simple F1 vs 1/variance F1: {r_f1:.4f} vs {var_f1:.4f} "
          f"(advantage = {advantage_pct:+.1f}%)", flush=True)

    return {
        "n_high": len(high_quality),
        "n_low": len(low_quality),
        "methods": method_results,
        "R_vs_invvar_advantage_pct": float(advantage_pct),
    }


# ============================================================
# TEST 3: Rank-Order Consistency
# ============================================================

def test3_rank_order(clusters, arch_name):
    """Spearman correlation between R and -F across all clusters."""
    print(f"\n  TEST 3 [{arch_name}]: Rank-Order Consistency", flush=True)
    print(f"  {'-'*50}", flush=True)

    R_simple_vals = []
    R_full_vals = []
    neg_F_vals = []

    for cluster in clusters:
        emb = cluster["embeddings"]
        R_s = compute_R_simple(emb)
        R_f = compute_R_full(emb)
        F = compute_variational_free_energy(emb, reg=1e-4)

        if np.isnan(R_s) or R_s <= 0 or np.isnan(F):
            continue

        R_simple_vals.append(R_s)
        R_full_vals.append(R_f if (not np.isnan(R_f) and R_f > 0) else float("nan"))
        neg_F_vals.append(-F)

    R_simple_vals = np.array(R_simple_vals)
    R_full_vals = np.array(R_full_vals)
    neg_F_vals = np.array(neg_F_vals)
    n = len(R_simple_vals)

    print(f"    Valid clusters: {n}", flush=True)
    if n < 10:
        return {"error": "insufficient_data", "n_valid": n}

    rho_s, p_s = spearmanr(R_simple_vals, neg_F_vals)
    print(f"    R_simple vs -F: Spearman rho = {rho_s:.4f} (p = {p_s:.6f})", flush=True)

    valid_full = ~np.isnan(R_full_vals)
    if np.sum(valid_full) >= 10:
        rho_f, p_f = spearmanr(R_full_vals[valid_full], neg_F_vals[valid_full])
        print(f"    R_full   vs -F: Spearman rho = {rho_f:.4f} (p = {p_f:.6f})",
              flush=True)
    else:
        rho_f, p_f = float("nan"), float("nan")

    return {
        "n_valid": n,
        "R_simple_vs_neg_F": {
            "spearman_rho": float(rho_s),
            "spearman_p": float(p_s),
        },
        "R_full_vs_neg_F": {
            "spearman_rho": float(rho_f),
            "spearman_p": float(p_f),
        },
    }


# ============================================================
# TEST 4: Cross-Architecture Consistency
# ============================================================

def test4_cross_architecture(all_arch_results):
    """Report Pearson r between log(R) and -F for each architecture."""
    print("\n" + "=" * 70, flush=True)
    print("TEST 4: Cross-Architecture Consistency", flush=True)
    print("=" * 70, flush=True)

    arch_names = []
    pearson_rs = []
    spearman_rs = []

    for arch_name, results in all_arch_results.items():
        t1 = results.get("test1", {})
        if "error" in t1:
            print(f"  {arch_name}: SKIPPED (error)", flush=True)
            continue
        r_p = t1["log_R_simple_vs_neg_F"]["pearson_r"]
        r_s = t1["log_R_simple_vs_neg_F"]["spearman_rho"]
        arch_names.append(arch_name)
        pearson_rs.append(r_p)
        spearman_rs.append(r_s)
        print(f"  {arch_name}: Pearson r = {r_p:.4f}, Spearman rho = {r_s:.4f}",
              flush=True)

    if len(pearson_rs) >= 2:
        range_pearson = max(pearson_rs) - min(pearson_rs)
        range_spearman = max(spearman_rs) - min(spearman_rs)
        mean_pearson = float(np.mean(pearson_rs))
        mean_spearman = float(np.mean(spearman_rs))
    else:
        range_pearson = float("nan")
        range_spearman = float("nan")
        mean_pearson = float(pearson_rs[0]) if pearson_rs else float("nan")
        mean_spearman = float(spearman_rs[0]) if spearman_rs else float("nan")

    n_above_05 = sum(1 for r in pearson_rs if r > 0.5)
    n_below_03 = sum(1 for r in pearson_rs if r < 0.3)

    print(f"\n  Architectures with Pearson r > 0.5: {n_above_05}/{len(pearson_rs)}",
          flush=True)
    print(f"  Architectures with Pearson r < 0.3: {n_below_03}/{len(pearson_rs)}",
          flush=True)

    return {
        "arch_names": arch_names,
        "pearson_rs": [float(x) for x in pearson_rs],
        "spearman_rs": [float(x) for x in spearman_rs],
        "range_pearson": float(range_pearson) if not np.isnan(range_pearson) else None,
        "range_spearman": float(range_spearman) if not np.isnan(range_spearman) else None,
        "mean_pearson": float(mean_pearson),
        "mean_spearman": float(mean_spearman),
        "n_above_05": n_above_05,
        "n_below_03": n_below_03,
        "n_architectures": len(pearson_rs),
    }


# ============================================================
# VERDICT
# ============================================================

def determine_verdict(all_arch_results, test4_results):
    """
    Apply pre-registered criteria.
    """
    print("\n" + "=" * 70, flush=True)
    print("VERDICT DETERMINATION", flush=True)
    print("=" * 70, flush=True)

    pearson_rs = []
    r_advantages = []
    arch_names_corr = []
    arch_names_gate = []

    for arch_name, results in all_arch_results.items():
        t1 = results.get("test1", {})
        t2 = results.get("test2", {})
        if "error" not in t1:
            pearson_rs.append(t1["log_R_simple_vs_neg_F"]["pearson_r"])
            arch_names_corr.append(arch_name)
        if "error" not in t2:
            r_advantages.append(t2.get("R_vs_invvar_advantage_pct", float("nan")))
            arch_names_gate.append(arch_name)

    # Criterion 1: correlation threshold
    n_above_05 = sum(1 for r in pearson_rs if r > 0.5)
    n_below_03 = sum(1 for r in pearson_rs if r < 0.3)
    all_below_03 = (n_below_03 == len(pearson_rs)) and len(pearson_rs) > 0

    print(f"  Criterion 1: corr(log(R), -F)", flush=True)
    for name, r in zip(arch_names_corr, pearson_rs):
        zone = "CONFIRM" if r > 0.5 else ("FALSIFY" if r < 0.3 else "INCONCLUSIVE")
        print(f"    {name}: Pearson r = {r:.4f} [{zone}]", flush=True)
    print(f"    Above 0.5: {n_above_05}/{len(pearson_rs)}", flush=True)
    print(f"    Below 0.3: {n_below_03}/{len(pearson_rs)}", flush=True)
    confirm_corr = n_above_05 >= (2 * len(pearson_rs) / 3)
    falsify_corr = all_below_03

    # Criterion 2: R-gating advantage
    print(f"\n  Criterion 2: R-gating vs 1/variance", flush=True)
    r_worse_count = 0
    r_better_10_count = 0
    for name, adv in zip(arch_names_gate, r_advantages):
        print(f"    {name}: advantage = {adv:+.1f}%", flush=True)
        if adv < 0:
            r_worse_count += 1
        if adv >= 10:
            r_better_10_count += 1

    r_worse_majority = r_worse_count > len(r_advantages) / 2
    confirm_gating = (r_better_10_count >= (2 * len(r_advantages) / 3)
                      if r_advantages else False)
    falsify_gating = r_worse_majority

    print(f"\n  Confirm conditions:", flush=True)
    print(f"    Correlation > 0.5 in 2/3 arch: {confirm_corr}", flush=True)
    print(f"    R-gating beats 1/var by 10%:   {confirm_gating}", flush=True)
    print(f"  Falsify conditions:", flush=True)
    print(f"    Correlation < 0.3 in ALL arch:  {falsify_corr}", flush=True)
    print(f"    R-gating worse than 1/var:      {falsify_gating}", flush=True)

    if confirm_corr and confirm_gating:
        verdict = "CONFIRMED"
    elif falsify_corr or falsify_gating:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  >>> VERDICT: {verdict} <<<", flush=True)

    return {
        "verdict": verdict,
        "confirm_corr": bool(confirm_corr),
        "confirm_gating": bool(confirm_gating),
        "falsify_corr": bool(falsify_corr),
        "falsify_gating": bool(falsify_gating),
        "pearson_rs": [float(r) for r in pearson_rs],
        "r_advantages": [float(a) for a in r_advantages],
        "arch_names_corr": arch_names_corr,
        "arch_names_gate": arch_names_gate,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    np.random.seed(SEED)

    print("=" * 70, flush=True)
    print("Q09 v2 FIXED TEST: Does log(R) = -F + const?", flush=True)
    print("20 Newsgroups | 3 architectures | NO synthetic data", flush=True)
    print("=" * 70, flush=True)

    # Load data once (subsampled for encoding speed)
    texts, labels, target_names = load_20newsgroups_subsampled(
        max_per_cat=MAX_DOCS_PER_CATEGORY, seed=SEED
    )

    all_arch_results = {}

    for arch_name, dim in ARCHITECTURES:
        print("\n" + "=" * 70, flush=True)
        print(f"ARCHITECTURE: {arch_name} ({dim}-dim)", flush=True)
        print("=" * 70, flush=True)

        arch_start = time.time()

        # Encode
        embeddings = encode_texts(texts, arch_name)

        # Build clusters
        clusters = build_clusters(embeddings, labels, target_names,
                                  cluster_size=CLUSTER_SIZE, seed=SEED)

        type_counts = {}
        for c in clusters:
            t = c["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Cluster breakdown: {type_counts}", flush=True)

        # Run tests 1-3
        t1 = test1_correlation(clusters, arch_name)
        t2 = test2_gating(clusters, arch_name)
        t3 = test3_rank_order(clusters, arch_name)

        arch_elapsed = time.time() - arch_start
        print(f"\n  Architecture {arch_name} completed in {arch_elapsed:.1f}s", flush=True)

        all_arch_results[arch_name] = {
            "test1": t1,
            "test2": t2,
            "test3": t3,
            "dim": dim,
            "elapsed_seconds": round(arch_elapsed, 1),
        }

        # Free embeddings to save memory before next architecture
        del embeddings
        del clusters
        gc.collect()

    # Test 4: cross-architecture
    t4 = test4_cross_architecture(all_arch_results)

    # Verdict
    verdict_info = determine_verdict(all_arch_results, t4)

    elapsed = time.time() - start_time

    # Assemble full results
    full_results = {
        "verdict": verdict_info["verdict"],
        "verdict_detail": verdict_info,
        "test4_cross_architecture": t4,
        "per_architecture": {},
        "metadata": {
            "dataset": "20 Newsgroups (sklearn, headers/footers/quotes removed)",
            "architectures": [a[0] for a in ARCHITECTURES],
            "cluster_size": CLUSTER_SIZE,
            "n_pure_clusters": 20,
            "n_mixed_clusters": N_MIXED,
            "n_degraded_clusters": N_DEGRADED,
            "total_clusters": 60,
            "max_docs_per_category": MAX_DOCS_PER_CATEGORY,
            "total_docs_encoded": len(texts),
            "seed": SEED,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    for arch_name, results in all_arch_results.items():
        full_results["per_architecture"][arch_name] = results

    # Save results
    results_path = os.path.join(RESULTS_DIR, "test_v2_q09_fixed_results.json")
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}", flush=True)

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Verdict: {verdict_info['verdict']}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  Dataset: 20 Newsgroups ({len(texts)} docs)", flush=True)
    print(f"  Architectures: {len(ARCHITECTURES)}", flush=True)

    for arch_name, results in all_arch_results.items():
        t1 = results["test1"]
        t2 = results["test2"]
        t3 = results["test3"]
        print(f"\n  {arch_name}:", flush=True)
        if "error" not in t1:
            print(f"    Test 1 - corr(log(R), -F): "
                  f"Pearson={t1['log_R_simple_vs_neg_F']['pearson_r']:.4f}, "
                  f"Spearman={t1['log_R_simple_vs_neg_F']['spearman_rho']:.4f}",
                  flush=True)
        if "error" not in t2:
            r_f1 = t2["methods"].get("R_simple", {})
            v_f1 = t2["methods"].get("1/variance", {})
            r_f1_val = r_f1["f1"] if r_f1 else "N/A"
            v_f1_val = v_f1["f1"] if v_f1 else "N/A"
            print(f"    Test 2 - R F1: {r_f1_val}, 1/var F1: {v_f1_val}, "
                  f"advantage: {t2['R_vs_invvar_advantage_pct']:+.1f}%", flush=True)
        if "error" not in t3:
            print(f"    Test 3 - Rank order: "
                  f"Spearman={t3['R_simple_vs_neg_F']['spearman_rho']:.4f}", flush=True)

    print(f"\n  Cross-arch consistency:", flush=True)
    print(f"    Pearson r values: {t4['pearson_rs']}", flush=True)
    print(f"    Mean Pearson r: {t4['mean_pearson']:.4f}", flush=True)

    return full_results


if __name__ == "__main__":
    results = main()
