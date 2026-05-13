"""
Q02 v3 Test: Does R = (E / grad_S) * sigma^Df have proper falsification criteria?

Fixes from v2 audit (AUDIT.md):
1. grad_S direction: TWO-SIDED test with effect size (no assumed direction)
2. R4=log(R0) removed from alternatives (monotonic transform => identical Spearman)
3. sigma and Df tested across models (not just across categories)
4. Cluster count increased: 40 pure + 40 impure at known ratios = 80 per model
5. Steiger's test for R vs E-alone comparison
6. Modus tollens with 200 clusters for tighter CIs
7. Pre-registered criteria updated for corrected design

Architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
Dataset: 20 Newsgroups (sklearn)

Date: 2026-02-06
"""

import importlib.util
import sys
import os
import json
import time
import math
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load the shared formula module
# ---------------------------------------------------------------------------
FORMULA_PATH = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
spec = importlib.util.spec_from_file_location("formula", FORMULA_PATH)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

RESULTS_DIR = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\q02_falsification\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Cluster configuration
DOCS_PER_CLUSTER = 100
SUBSAMPLE_PER_CAT = 400

# ---------------------------------------------------------------------------
# Pre-registered criteria (stated BEFORE any data is loaded)
# ---------------------------------------------------------------------------
# Audit fix #7: Corrected criteria
PRE_REGISTERED = {
    "test1_component_E": (
        "Two-sided Mann-Whitney U: E_pure != E_mixed at p < 0.05, "
        "reported with effect size (Cohen's d)"
    ),
    "test1_component_grad_S": (
        "Two-sided Mann-Whitney U: grad_S_pure != grad_S_mixed at p < 0.05, "
        "reported with effect size. Direction NOT assumed -- tested empirically."
    ),
    "test1_component_sigma": (
        "Two-sided Mann-Whitney U: sigma_pure != sigma_mixed at p < 0.05, "
        "OR CV > 0.1 across 3 architectures (cross-model variation)"
    ),
    "test1_component_Df": (
        "Two-sided Mann-Whitney U: Df_pure != Df_mixed at p < 0.05, "
        "OR CV > 0.1 across 3 architectures (cross-model variation)"
    ),
    "test2_functional_form": (
        "R0=E/grad_S outperforms >= 3 of 4 genuine alternatives at p < 0.01. "
        "R4=log(R0) excluded (monotonic transform => identical Spearman rank). "
        "Steiger's test: R vs E-alone on >= 2/3 architectures at p < 0.05."
    ),
    "test3_modus_tollens": (
        "Violation rate < 10% on held-out test data with n_high_R >= 30 "
        "for informative 95% CI."
    ),
    "test4_adversarial": (
        "R distinguishes pure from random clusters: effect size d > 0.8 "
        "AND p < 0.01."
    ),
    "verdict_CONFIRMED": (
        ">= 3/4 components show significant (p<0.05) two-sided effect "
        "AND R outperforms E-alone (Steiger p<0.05) on >= 2/3 architectures."
    ),
    "verdict_FALSIFIED": (
        "< 2/4 components significant two-sided "
        "AND Steiger NS on all 3 architectures."
    ),
    "verdict_INCONCLUSIVE": "Otherwise.",
}

t_start = time.time()

print("=" * 80)
print("Q02 v3 TEST: Formula Falsification Criteria (Audit-Fixed)")
print("=" * 80)
print("Seed: %d" % SEED)
print("Formula: R = (E / grad_S) * sigma^Df")
print("Data: 20 Newsgroups (sklearn)")
print("Timestamp: %s" % datetime.now().isoformat())
print()

# ---------------------------------------------------------------------------
# Step 1: Load 20 Newsgroups data and subsample
# ---------------------------------------------------------------------------
print("Loading 20 Newsgroups...")
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
all_texts = data.data
all_labels = np.array(data.target)
category_names = data.target_names
n_categories = len(category_names)

print("  Total documents: %d" % len(all_texts))
print("  Categories: %d" % n_categories)

# Subsample for encoding efficiency
rng_sub = np.random.RandomState(SEED)
subsample_indices = []
for cat_idx in range(n_categories):
    cat_mask = np.where(all_labels == cat_idx)[0]
    n_avail = len(cat_mask)
    n_take = min(SUBSAMPLE_PER_CAT, n_avail)
    chosen = rng_sub.choice(cat_mask, size=n_take, replace=False)
    subsample_indices.extend(chosen.tolist())
    print("  %2d. %s: %d total, %d subsampled" % (
        cat_idx, category_names[cat_idx], n_avail, n_take))

subsample_indices = np.array(subsample_indices)
texts = [all_texts[i] for i in subsample_indices]
labels = all_labels[subsample_indices]
n_docs = len(texts)
print("\n  Subsampled documents: %d" % n_docs)
print()

# ---------------------------------------------------------------------------
# Step 2: Encode with 3 architectures (audit fix #3: diverse architectures)
# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

MODEL_NAMES = [
    "all-MiniLM-L6-v2",           # 384-dim, general purpose, 6-layer
    "all-mpnet-base-v2",           # 768-dim, MPNet architecture (different arch+dim)
    "multi-qa-MiniLM-L6-cos-v1",  # 384-dim, QA-tuned (different training objective)
]

embeddings_by_model = {}
for model_name in MODEL_NAMES:
    print("Encoding with %s..." % model_name)
    t0 = time.time()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, batch_size=512)
    embeddings_by_model[model_name] = embs
    elapsed = time.time() - t0
    print("  Shape: %s, Time: %.1fs" % (str(embs.shape), elapsed))
    del model
    import gc
    gc.collect()
    sys.stdout.flush()

print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_category_docs(category_idx, n_docs_needed, emb_matrix, rng):
    """Get n_docs random embeddings from a specific category."""
    mask = labels == category_idx
    indices = np.where(mask)[0]
    n_take = min(n_docs_needed, len(indices))
    chosen = rng.choice(indices, size=n_take, replace=False)
    return emb_matrix[chosen], labels[chosen]


def get_random_docs(n_docs_needed, emb_matrix, rng):
    """Get n_docs random embeddings from all categories."""
    indices = rng.choice(len(emb_matrix), size=n_docs_needed, replace=False)
    return emb_matrix[indices], labels[indices]


def compute_purity(category_labels):
    """Compute cluster purity = fraction from dominant category."""
    if len(category_labels) == 0:
        return 0.0
    counts = np.bincount(category_labels, minlength=n_categories)
    return float(counts.max()) / len(category_labels)


def cohens_d(group_a, group_b):
    """Compute Cohen's d with pooled standard deviation (handles unequal n)."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return float('nan')
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    # Proper pooled SD formula for unequal sample sizes
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd < 1e-10:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_sd)


def steiger_z_test(r1, r2, r12, n):
    """
    Steiger's Z-test for comparing two dependent correlations r1 and r2,
    where r12 is the correlation between the two predictors, and n is
    the sample size.

    Tests H0: |rho1| = |rho2| vs H1: |rho1| > |rho2|

    Returns (z_stat, p_value_one_sided)
    """
    if n < 4:
        return (float('nan'), float('nan'))

    # Fisher z transforms
    z1 = np.arctanh(min(max(r1, -0.9999), 0.9999))
    z2 = np.arctanh(min(max(r2, -0.9999), 0.9999))
    z12 = np.arctanh(min(max(r12, -0.9999), 0.9999))

    # Steiger (1980) formula for the variance of the difference
    r_bar = (r1 + r2) / 2.0
    f = (1.0 - r12) / (2.0 * (1.0 - r_bar ** 2))
    if f < 0:
        f = 0.001
    h = (1.0 - f * r_bar ** 2) / (1.0 - r_bar ** 2)

    denom = np.sqrt(2.0 * (1.0 - r12) / ((n - 3) * (1.0 + r_bar ** 2 * h / (1.0 - r_bar ** 2))))
    if denom < 1e-10:
        return (float('nan'), float('nan'))

    z_stat = (z1 - z2) / denom

    from scipy.stats import norm
    p_one_sided = 1.0 - norm.cdf(z_stat)
    return (float(z_stat), float(p_one_sided))


from scipy import stats

# ===================================================================
# TEST 1: Component-Level Falsification (AUDIT-FIXED)
# ===================================================================
print("=" * 80)
print("TEST 1: Component-Level Falsification (two-sided, n=80 clusters)")
print("  Fix: grad_S uses TWO-SIDED test (no assumed direction)")
print("  Fix: sigma/Df tested pure vs mixed AND cross-model CV")
print("=" * 80)

test1_results = {}

# Store per-model component values for cross-model analysis
all_model_sigma_means = []
all_model_Df_means = []

for model_name in MODEL_NAMES:
    print("\n--- Architecture: %s ---" % model_name)
    emb_matrix = embeddings_by_model[model_name]
    rng = np.random.RandomState(SEED)

    # Audit fix #4: Increase cluster count to n>=80
    # 20 pure (one per category) + 20 sub-clusters within categories
    # + 20 mixed-2 + 20 mixed-5 = 80 clusters
    pure_metrics = {"E": [], "grad_S": [], "sigma": [], "Df": [], "R_simple": [], "R_full": []}
    mixed_metrics = {"E": [], "grad_S": [], "sigma": [], "Df": [], "R_simple": [], "R_full": []}

    # --- 40 pure clusters ---
    print("  Computing 40 pure clusters (20 primary + 20 sub-clusters)...")
    # 20 primary: one per category
    for cat_idx in range(n_categories):
        embs, _ = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng)
        result = formula.compute_all(embs)
        for k in pure_metrics:
            pure_metrics[k].append(result[k])

    # 20 sub-clusters: second draw from each category (different random subset)
    rng_sub2 = np.random.RandomState(SEED + 77)
    for cat_idx in range(n_categories):
        embs, _ = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng_sub2)
        result = formula.compute_all(embs)
        for k in pure_metrics:
            pure_metrics[k].append(result[k])

    # --- 40 mixed clusters ---
    print("  Computing 40 mixed clusters (20 mixed-2 + 20 mixed-random)...")
    # 20 mixed-2 clusters (50/50 from 2 categories)
    rng_mix = np.random.RandomState(SEED + 10)
    for _ in range(20):
        cats = rng_mix.choice(n_categories, size=2, replace=False)
        all_embs = []
        for c in cats:
            embs, _ = get_category_docs(c, DOCS_PER_CLUSTER // 2, emb_matrix, rng_mix)
            all_embs.append(embs)
        combined = np.vstack(all_embs)
        result = formula.compute_all(combined)
        for k in mixed_metrics:
            mixed_metrics[k].append(result[k])

    # 20 random clusters
    rng_rand = np.random.RandomState(SEED + 20)
    for _ in range(20):
        embs, _ = get_random_docs(DOCS_PER_CLUSTER, emb_matrix, rng_rand)
        result = formula.compute_all(embs)
        for k in mixed_metrics:
            mixed_metrics[k].append(result[k])

    model_results = {}
    n_pure = len(pure_metrics["E"])
    n_mixed = len(mixed_metrics["E"])
    print("  Pure clusters: %d, Mixed clusters: %d" % (n_pure, n_mixed))

    # --- E test: TWO-SIDED ---
    E_pure = np.array(pure_metrics["E"])
    E_mixed = np.array(mixed_metrics["E"])
    # two-sided Mann-Whitney
    u_stat, u_p = stats.mannwhitneyu(E_pure, E_mixed, alternative='two-sided')
    d_val = cohens_d(E_pure, E_mixed)
    observed_dir = "pure > mixed" if np.mean(E_pure) > np.mean(E_mixed) else "pure < mixed"
    model_results["E_test"] = {
        "pure_mean": float(np.mean(E_pure)),
        "pure_std": float(np.std(E_pure, ddof=1)),
        "mixed_mean": float(np.mean(E_mixed)),
        "mixed_std": float(np.std(E_mixed, ddof=1)),
        "n_pure": n_pure,
        "n_mixed": n_mixed,
        "U_statistic": float(u_stat),
        "p_value_two_sided": float(u_p),
        "cohens_d": float(d_val),
        "observed_direction": observed_dir,
        "significant": bool(u_p < 0.05),
    }
    print("  E test (two-sided): pure=%.4f+/-%.4f, mixed=%.4f+/-%.4f" % (
        np.mean(E_pure), np.std(E_pure, ddof=1),
        np.mean(E_mixed), np.std(E_mixed, ddof=1)))
    print("    U p=%.2e, d=%.2f, direction=%s -> %s" % (
        u_p, d_val, observed_dir, "SIGNIFICANT" if u_p < 0.05 else "NS"))

    # --- grad_S test: TWO-SIDED (audit fix #1) ---
    gS_pure = np.array(pure_metrics["grad_S"])
    gS_mixed = np.array(mixed_metrics["grad_S"])
    u_stat, u_p = stats.mannwhitneyu(gS_pure, gS_mixed, alternative='two-sided')
    d_val = cohens_d(gS_pure, gS_mixed)
    observed_dir = "pure > mixed" if np.mean(gS_pure) > np.mean(gS_mixed) else "pure < mixed"
    model_results["grad_S_test"] = {
        "pure_mean": float(np.mean(gS_pure)),
        "pure_std": float(np.std(gS_pure, ddof=1)),
        "mixed_mean": float(np.mean(gS_mixed)),
        "mixed_std": float(np.std(gS_mixed, ddof=1)),
        "n_pure": n_pure,
        "n_mixed": n_mixed,
        "U_statistic": float(u_stat),
        "p_value_two_sided": float(u_p),
        "cohens_d": float(d_val),
        "observed_direction": observed_dir,
        "significant": bool(u_p < 0.05),
    }
    print("  grad_S test (two-sided): pure=%.4f+/-%.4f, mixed=%.4f+/-%.4f" % (
        np.mean(gS_pure), np.std(gS_pure, ddof=1),
        np.mean(gS_mixed), np.std(gS_mixed, ddof=1)))
    print("    U p=%.2e, d=%.2f, direction=%s -> %s" % (
        u_p, d_val, observed_dir, "SIGNIFICANT" if u_p < 0.05 else "NS"))

    # --- sigma test: pure vs mixed AND cross-model CV (audit fix #3) ---
    sigma_pure = np.array(pure_metrics["sigma"])
    sigma_mixed = np.array(mixed_metrics["sigma"])
    u_stat, u_p = stats.mannwhitneyu(sigma_pure, sigma_mixed, alternative='two-sided')
    d_val = cohens_d(sigma_pure, sigma_mixed)
    observed_dir = "pure > mixed" if np.mean(sigma_pure) > np.mean(sigma_mixed) else "pure < mixed"
    # Store mean for cross-model CV computation
    all_model_sigma_means.append(float(np.mean(np.concatenate([sigma_pure, sigma_mixed]))))
    model_results["sigma_test"] = {
        "pure_mean": float(np.mean(sigma_pure)),
        "pure_std": float(np.std(sigma_pure, ddof=1)),
        "mixed_mean": float(np.mean(sigma_mixed)),
        "mixed_std": float(np.std(sigma_mixed, ddof=1)),
        "n_pure": n_pure,
        "n_mixed": n_mixed,
        "U_statistic": float(u_stat),
        "p_value_two_sided": float(u_p),
        "cohens_d": float(d_val),
        "observed_direction": observed_dir,
        "significant_pure_vs_mixed": bool(u_p < 0.05),
    }
    print("  sigma test (two-sided): pure=%.4f+/-%.4f, mixed=%.4f+/-%.4f" % (
        np.mean(sigma_pure), np.std(sigma_pure, ddof=1),
        np.mean(sigma_mixed), np.std(sigma_mixed, ddof=1)))
    print("    U p=%.2e, d=%.2f, direction=%s -> %s" % (
        u_p, d_val, observed_dir, "SIGNIFICANT" if u_p < 0.05 else "NS"))

    # --- Df test: pure vs mixed AND cross-model CV (audit fix #3) ---
    Df_pure = np.array(pure_metrics["Df"])
    Df_mixed = np.array(mixed_metrics["Df"])
    u_stat, u_p = stats.mannwhitneyu(Df_pure, Df_mixed, alternative='two-sided')
    d_val = cohens_d(Df_pure, Df_mixed)
    observed_dir = "pure > mixed" if np.mean(Df_pure) > np.mean(Df_mixed) else "pure < mixed"
    all_model_Df_means.append(float(np.mean(np.concatenate([Df_pure, Df_mixed]))))
    model_results["Df_test"] = {
        "pure_mean": float(np.mean(Df_pure)),
        "pure_std": float(np.std(Df_pure, ddof=1)),
        "mixed_mean": float(np.mean(Df_mixed)),
        "mixed_std": float(np.std(Df_mixed, ddof=1)),
        "n_pure": n_pure,
        "n_mixed": n_mixed,
        "U_statistic": float(u_stat),
        "p_value_two_sided": float(u_p),
        "cohens_d": float(d_val),
        "observed_direction": observed_dir,
        "significant_pure_vs_mixed": bool(u_p < 0.05),
    }
    print("  Df test (two-sided): pure=%.4f+/-%.4f, mixed=%.4f+/-%.4f" % (
        np.mean(Df_pure), np.std(Df_pure, ddof=1),
        np.mean(Df_mixed), np.std(Df_mixed, ddof=1)))
    print("    U p=%.2e, d=%.2f, direction=%s -> %s" % (
        u_p, d_val, observed_dir, "SIGNIFICANT" if u_p < 0.05 else "NS"))

    test1_results[model_name] = model_results
    sys.stdout.flush()

# --- Cross-model variation for sigma and Df (audit fix #3) ---
print("\n--- Cross-Model Variation (sigma and Df) ---")
sigma_arr = np.array(all_model_sigma_means)
Df_arr = np.array(all_model_Df_means)
sigma_cross_cv = float(np.std(sigma_arr, ddof=1) / np.mean(sigma_arr)) if np.mean(sigma_arr) > 0 else 0.0
Df_cross_cv = float(np.std(Df_arr, ddof=1) / np.mean(Df_arr)) if np.mean(Df_arr) > 0 else 0.0

print("  sigma means by model: %s" % [round(v, 6) for v in sigma_arr])
print("  sigma cross-model CV: %.4f (threshold: 0.1)" % sigma_cross_cv)
print("  Df means by model: %s" % [round(v, 6) for v in Df_arr])
print("  Df cross-model CV: %.4f (threshold: 0.1)" % Df_cross_cv)

cross_model_results = {
    "sigma_means_by_model": [float(v) for v in sigma_arr],
    "sigma_cross_model_CV": sigma_cross_cv,
    "sigma_cross_model_significant": bool(sigma_cross_cv > 0.1),
    "Df_means_by_model": [float(v) for v in Df_arr],
    "Df_cross_model_CV": Df_cross_cv,
    "Df_cross_model_significant": bool(Df_cross_cv > 0.1),
}

print()
sys.stdout.flush()

# ===================================================================
# TEST 2: Functional Form Comparison (AUDIT-FIXED)
# ===================================================================
print("=" * 80)
print("TEST 2: Functional Form Comparison")
print("  Fix: R4=log(R0) removed (4 genuine alternatives)")
print("  Fix: Steiger's test for R vs E-alone")
print("  Fix: n=80 clusters")
print("=" * 80)

test2_results = {}

for model_name in MODEL_NAMES:
    print("\n--- Architecture: %s ---" % model_name)
    emb_matrix = embeddings_by_model[model_name]
    rng = np.random.RandomState(SEED + 1)

    cluster_embeddings = []
    cluster_purities = []
    cluster_types = []

    # 20 pure clusters
    print("  Creating 20 pure clusters...")
    for cat_idx in range(n_categories):
        embs, cat_labels = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng)
        purity = compute_purity(cat_labels)
        cluster_embeddings.append(embs)
        cluster_purities.append(purity)
        cluster_types.append("pure")

    # 20 mixed-2 clusters
    print("  Creating 20 mixed-2 clusters...")
    for _ in range(20):
        cats = rng.choice(n_categories, size=2, replace=False)
        all_embs = []
        all_labs = []
        for c in cats:
            embs, cat_labels = get_category_docs(c, DOCS_PER_CLUSTER // 2, emb_matrix, rng)
            all_embs.append(embs)
            all_labs.append(cat_labels)
        combined_embs = np.vstack(all_embs)
        combined_labs = np.concatenate(all_labs)
        purity = compute_purity(combined_labs)
        cluster_embeddings.append(combined_embs)
        cluster_purities.append(purity)
        cluster_types.append("mixed-2")

    # 20 mixed-5 clusters
    print("  Creating 20 mixed-5 clusters...")
    for _ in range(20):
        cats = rng.choice(n_categories, size=5, replace=False)
        all_embs = []
        all_labs = []
        for c in cats:
            embs, cat_labels = get_category_docs(c, DOCS_PER_CLUSTER // 5, emb_matrix, rng)
            all_embs.append(embs)
            all_labs.append(cat_labels)
        combined_embs = np.vstack(all_embs)
        combined_labs = np.concatenate(all_labs)
        purity = compute_purity(combined_labs)
        cluster_embeddings.append(combined_embs)
        cluster_purities.append(purity)
        cluster_types.append("mixed-5")

    # 20 random clusters
    print("  Creating 20 random clusters...")
    rng_r2 = np.random.RandomState(SEED + 2)
    for _ in range(20):
        embs, cat_labels = get_random_docs(DOCS_PER_CLUSTER, emb_matrix, rng_r2)
        purity = compute_purity(cat_labels)
        cluster_embeddings.append(embs)
        cluster_purities.append(purity)
        cluster_types.append("random")

    cluster_purities = np.array(cluster_purities)
    n_clusters = len(cluster_embeddings)
    print("  Total clusters: %d" % n_clusters)
    print("  Purity range: [%.3f, %.3f]" % (cluster_purities.min(), cluster_purities.max()))

    # Compute 5 variants (audit fix #2: R4 removed)
    print("  Computing formula variants (R4=log removed)...")

    def compute_MAD(embeddings):
        """Compute MAD of pairwise cosine similarities."""
        n = embeddings.shape[0]
        if n < 2:
            return float('nan')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_indices]
        return float(np.median(np.abs(pairwise_sims - np.median(pairwise_sims))))

    variant_values = {
        "R0_E_over_gradS": [],     # The formula: E/grad_S
        "R1_E_over_gradS2": [],    # Alt: E/grad_S^2
        "R2_E_over_MAD": [],       # Alt: E/MAD
        "R3_E_times_gradS": [],    # Alt: E*grad_S (opposite)
        "R5_E_alone": [],          # Alt: E alone (no denominator)
    }

    for i, embs in enumerate(cluster_embeddings):
        E_val = formula.compute_E(embs)
        gS_val = formula.compute_grad_S(embs)
        mad_val = compute_MAD(embs)

        r0 = E_val / gS_val if gS_val > 1e-10 else float('nan')
        r1 = E_val / (gS_val ** 2) if gS_val > 1e-10 else float('nan')
        r2 = E_val / mad_val if mad_val > 1e-10 else float('nan')
        r3 = E_val * gS_val
        r5 = E_val

        variant_values["R0_E_over_gradS"].append(r0)
        variant_values["R1_E_over_gradS2"].append(r1)
        variant_values["R2_E_over_MAD"].append(r2)
        variant_values["R3_E_times_gradS"].append(r3)
        variant_values["R5_E_alone"].append(r5)

    # Spearman correlations
    print("\n  Spearman correlations with cluster purity (n=%d):" % n_clusters)
    variant_correlations = {}
    for name, vals in variant_values.items():
        vals_arr = np.array(vals)
        valid_mask = ~np.isnan(vals_arr)
        if valid_mask.sum() < 10:
            variant_correlations[name] = {
                "rho": float('nan'), "p": 1.0, "n_valid": int(valid_mask.sum())
            }
            continue
        rho, p = stats.spearmanr(vals_arr[valid_mask], cluster_purities[valid_mask])
        variant_correlations[name] = {
            "rho": float(rho), "p": float(p), "n_valid": int(valid_mask.sum())
        }
        print("    %s: rho=%.4f, p=%.2e, n=%d" % (name, rho, p, valid_mask.sum()))

    # Rank by |rho|
    ranked = sorted(variant_correlations.items(),
                    key=lambda x: abs(x[1]["rho"]) if not np.isnan(x[1]["rho"]) else 0,
                    reverse=True)
    print("\n  Ranking by |rho|:")
    r0_rank = None
    for rank, (name, vals) in enumerate(ranked, 1):
        marker = " <-- FORMULA" if name == "R0_E_over_gradS" else ""
        print("    %d. %s: |rho|=%.4f%s" % (rank, name, abs(vals["rho"]), marker))
        if name == "R0_E_over_gradS":
            r0_rank = rank

    # Bootstrap pairwise comparison (audit fix #2: only 4 genuine alternatives)
    print("\n  Bootstrap pairwise comparisons (2000 resamples, 4 alternatives):")
    sys.stdout.flush()
    N_BOOTSTRAP = 2000
    boot_rng = np.random.RandomState(SEED + 100)

    r0_vals = np.array(variant_values["R0_E_over_gradS"])
    r0_wins = {}

    for alt_name, alt_vals_list in variant_values.items():
        if alt_name == "R0_E_over_gradS":
            continue
        alt_vals = np.array(alt_vals_list)

        valid = ~(np.isnan(r0_vals) | np.isnan(alt_vals))
        if valid.sum() < 10:
            r0_wins[alt_name] = {
                "win_rate": float('nan'), "p_r0_better": float('nan')
            }
            continue

        r0_v = r0_vals[valid]
        alt_v = alt_vals[valid]
        purity_v = cluster_purities[valid]
        n_valid = len(purity_v)

        boot_indices = boot_rng.choice(n_valid, size=(N_BOOTSTRAP, n_valid), replace=True)

        r0_better_count = 0
        for b in range(N_BOOTSTRAP):
            idx = boot_indices[b]
            rho_r0 = stats.spearmanr(r0_v[idx], purity_v[idx])[0]
            rho_alt = stats.spearmanr(alt_v[idx], purity_v[idx])[0]
            if abs(rho_r0) > abs(rho_alt):
                r0_better_count += 1

        win_rate = r0_better_count / N_BOOTSTRAP
        p_r0_better = 1.0 - win_rate
        r0_wins[alt_name] = {
            "win_rate": float(win_rate),
            "p_r0_better": float(p_r0_better),
        }
        if win_rate > 0.5:
            result_str = "R0 WINS" if p_r0_better < 0.01 else "R0 AHEAD"
        else:
            result_str = "R0 LOSES"
        print("    R0 vs %s: R0 wins %.1f%%, p=%.4f -> %s" % (
            alt_name, win_rate * 100, p_r0_better, result_str))
        sys.stdout.flush()

    # Count genuine wins (audit fix #2: out of 4, not 5)
    n_r0_wins = sum(1 for v in r0_wins.values()
                    if not np.isnan(v["p_r0_better"]) and v["p_r0_better"] < 0.01)
    n_r0_loses = sum(1 for v in r0_wins.values()
                     if not np.isnan(v["win_rate"]) and v["win_rate"] < 0.5)
    print("\n  R0 beats %d/4 genuine alternatives at p < 0.01" % n_r0_wins)

    # --- Steiger's test: R0 vs E-alone (audit fix #5) ---
    r0_arr = np.array(variant_values["R0_E_over_gradS"])
    e_arr = np.array(variant_values["R5_E_alone"])
    valid_st = ~(np.isnan(r0_arr) | np.isnan(e_arr))
    r0_st = r0_arr[valid_st]
    e_st = e_arr[valid_st]
    pur_st = cluster_purities[valid_st]
    n_st = len(pur_st)

    rho_R0_pur = stats.spearmanr(r0_st, pur_st)[0]
    rho_E_pur = stats.spearmanr(e_st, pur_st)[0]
    rho_R0_E = stats.spearmanr(r0_st, e_st)[0]

    steiger_z, steiger_p = steiger_z_test(
        abs(rho_R0_pur), abs(rho_E_pur), abs(rho_R0_E), n_st
    )
    steiger_significant = (not np.isnan(steiger_p)) and steiger_p < 0.05

    print("\n  Steiger's test: R0 vs E-alone")
    print("    rho(R0, purity) = %.4f" % rho_R0_pur)
    print("    rho(E, purity)  = %.4f" % rho_E_pur)
    print("    rho(R0, E)      = %.4f" % rho_R0_E)
    print("    Steiger Z = %.3f, p = %.4f -> %s" % (
        steiger_z, steiger_p, "R0 BETTER" if steiger_significant else "NS"))

    test2_results[model_name] = {
        "n_clusters": n_clusters,
        "variant_correlations": variant_correlations,
        "ranking": [(name, vals) for name, vals in ranked],
        "r0_rank": r0_rank,
        "bootstrap_comparisons": r0_wins,
        "n_r0_wins_p01": n_r0_wins,
        "n_r0_loses": n_r0_loses,
        "steiger_test": {
            "rho_R0_purity": float(rho_R0_pur),
            "rho_E_purity": float(rho_E_pur),
            "rho_R0_E": float(rho_R0_E),
            "z_stat": float(steiger_z),
            "p_one_sided": float(steiger_p),
            "n": n_st,
            "R0_significantly_better": steiger_significant,
        },
    }

print()
sys.stdout.flush()

# ===================================================================
# TEST 3: Modus Tollens (AUDIT-FIXED: 200 clusters for tighter CIs)
# ===================================================================
print("=" * 80)
print("TEST 3: Modus Tollens (train/test split, 200 clusters each)")
print("  Fix: n=200 per split for tighter 95% CIs")
print("  Fix: Use diverse cluster types for better threshold calibration")
print("=" * 80)

test3_results = {}

for model_name in MODEL_NAMES:
    print("\n--- Architecture: %s ---" % model_name)
    emb_matrix = embeddings_by_model[model_name]

    # Split document indices into train (60%) and test (40%) within each category
    rng_split = np.random.RandomState(SEED + 200)
    train_indices_set = set()
    test_indices_set = set()

    for cat_idx in range(n_categories):
        cat_mask = np.where(labels == cat_idx)[0]
        rng_split.shuffle(cat_mask)
        split_pt = int(0.6 * len(cat_mask))
        train_indices_set.update(cat_mask[:split_pt].tolist())
        test_indices_set.update(cat_mask[split_pt:].tolist())

    print("  Train: %d docs, Test: %d docs" % (len(train_indices_set), len(test_indices_set)))

    def get_category_docs_split(category_idx, n_docs_needed, split_set, rng_local):
        """Get docs from a specific category within a split."""
        cat_mask = np.where(labels == category_idx)[0]
        valid = np.array([i for i in cat_mask if i in split_set])
        n_take = min(n_docs_needed, len(valid))
        if n_take == 0:
            return np.empty((0, emb_matrix.shape[1])), np.array([], dtype=int)
        chosen = rng_local.choice(valid, size=n_take, replace=False)
        return emb_matrix[chosen], labels[chosen]

    def get_random_docs_split(n_docs_needed, split_set, rng_local):
        """Get random docs from the split."""
        valid = np.array(list(split_set))
        n_take = min(n_docs_needed, len(valid))
        chosen = rng_local.choice(valid, size=n_take, replace=False)
        return emb_matrix[chosen], labels[chosen]

    def create_clusters(n_clusters_total, split_set, rng_local, cluster_size=80):
        """Create clusters: pure, mixed-2, mixed-5, random in equal parts."""
        clusters = []
        n_per_type = n_clusters_total // 4

        # Pure clusters
        for i in range(n_per_type):
            cat = i % n_categories
            embs, cat_labs = get_category_docs_split(cat, cluster_size, split_set, rng_local)
            if len(embs) < 10:
                continue
            purity = compute_purity(cat_labs)
            clusters.append({"embeddings": embs, "purity": purity, "type": "pure"})

        # Mixed-2
        for _ in range(n_per_type):
            cats = rng_local.choice(n_categories, size=2, replace=False)
            all_embs = []
            all_labs = []
            for c in cats:
                embs, cat_labs = get_category_docs_split(
                    c, cluster_size // 2, split_set, rng_local)
                all_embs.append(embs)
                all_labs.append(cat_labs)
            combined_embs = np.vstack(all_embs) if all_embs else np.empty((0, emb_matrix.shape[1]))
            combined_labs = np.concatenate(all_labs) if all_labs else np.array([], dtype=int)
            if len(combined_embs) < 10:
                continue
            purity = compute_purity(combined_labs)
            clusters.append({"embeddings": combined_embs, "purity": purity, "type": "mixed-2"})

        # Mixed-5
        for _ in range(n_per_type):
            cats = rng_local.choice(n_categories, size=5, replace=False)
            all_embs = []
            all_labs = []
            for c in cats:
                embs, cat_labs = get_category_docs_split(
                    c, cluster_size // 5, split_set, rng_local)
                all_embs.append(embs)
                all_labs.append(cat_labs)
            combined_embs = np.vstack(all_embs) if all_embs else np.empty((0, emb_matrix.shape[1]))
            combined_labs = np.concatenate(all_labs) if all_labs else np.array([], dtype=int)
            if len(combined_embs) < 10:
                continue
            purity = compute_purity(combined_labs)
            clusters.append({"embeddings": combined_embs, "purity": purity, "type": "mixed-5"})

        # Random
        remainder = n_clusters_total - 3 * n_per_type
        for _ in range(remainder):
            embs, cat_labs = get_random_docs_split(cluster_size, split_set, rng_local)
            if len(embs) < 10:
                continue
            purity = compute_purity(cat_labs)
            clusters.append({"embeddings": embs, "purity": purity, "type": "random"})

        return clusters

    # --- TRAIN: Calibrate threshold (audit fix #6: 200 clusters) ---
    print("  Creating 200 training clusters...")
    rng_train = np.random.RandomState(SEED + 300)
    train_clusters = create_clusters(200, train_indices_set, rng_train, cluster_size=80)

    train_R_values = []
    train_purities = []
    for cl in train_clusters:
        R_val = formula.compute_R_simple(cl["embeddings"])
        train_R_values.append(R_val)
        train_purities.append(cl["purity"])

    train_R = np.array(train_R_values)
    train_purity = np.array(train_purities)

    # Calibrate: T = median R among high-purity clusters (purity > 0.8)
    high_purity_mask = train_purity > 0.8
    valid_train = ~np.isnan(train_R)
    if (high_purity_mask & valid_train).sum() > 0:
        T = float(np.median(train_R[high_purity_mask & valid_train]))
    else:
        T = float(np.nanmedian(train_R))

    # Q_min = 10th percentile of purity among clusters with R > T
    high_R_mask = (train_R > T) & valid_train
    if high_R_mask.sum() > 0:
        Q_min = float(np.percentile(train_purity[high_R_mask], 10))
    else:
        Q_min = 0.5

    # Spearman on train
    if valid_train.sum() > 5:
        rho_train, p_train = stats.spearmanr(train_R[valid_train], train_purity[valid_train])
    else:
        rho_train, p_train = float('nan'), float('nan')

    print("  Train clusters: %d" % len(train_clusters))
    print("  High-purity (>0.8): %d" % (high_purity_mask & valid_train).sum())
    print("  High-R (>T): %d" % high_R_mask.sum())
    print("  Calibrated threshold T = %.4f" % T)
    print("  Quality minimum Q_min = %.4f" % Q_min)
    print("  Train Spearman rho(R, purity) = %.4f, p = %.2e" % (rho_train, p_train))

    # --- TEST: Evaluate modus tollens (200 clusters) ---
    print("  Creating 200 test clusters...")
    rng_test = np.random.RandomState(SEED + 400)
    test_clusters = create_clusters(200, test_indices_set, rng_test, cluster_size=80)

    test_R_values = []
    test_purities = []
    test_types = []
    for cl in test_clusters:
        R_val = formula.compute_R_simple(cl["embeddings"])
        test_R_values.append(R_val)
        test_purities.append(cl["purity"])
        test_types.append(cl["type"])

    test_R = np.array(test_R_values)
    test_purity = np.array(test_purities)

    # Modus tollens: among clusters with R > T, how many have purity < Q_min?
    valid_test = ~np.isnan(test_R)
    test_high_R = (test_R > T) & valid_test
    n_high_R = int(test_high_R.sum())
    if n_high_R > 0:
        violations = test_purity[test_high_R] < Q_min
        n_violations = int(violations.sum())
        violation_rate = float(n_violations) / n_high_R

        from statsmodels.stats.proportion import proportion_confint
        ci_low, ci_high = proportion_confint(
            n_violations, n_high_R, alpha=0.05, method='wilson')
    else:
        n_violations = 0
        violation_rate = float('nan')
        ci_low, ci_high = (float('nan'), float('nan'))

    # Spearman on test
    if valid_test.sum() > 5:
        rho_test, p_test = stats.spearmanr(test_R[valid_test], test_purity[valid_test])
    else:
        rho_test, p_test = float('nan'), float('nan')

    print("\n  TEST RESULTS:")
    print("  Test clusters: %d" % len(test_clusters))
    print("  Clusters with R > T (%.4f): %d / %d" % (T, n_high_R, len(test_clusters)))
    print("  Violations (purity < Q_min=%.4f): %d / %d" % (Q_min, n_violations, n_high_R))
    if not np.isnan(violation_rate):
        print("  Violation rate: %.3f" % violation_rate)
    else:
        print("  Violation rate: N/A (no clusters with R > T)")
    if not np.isnan(ci_low):
        print("  95%% CI: [%.3f, %.3f]" % (ci_low, ci_high))
    print("  Test Spearman rho(R, purity) = %.4f, p = %.2e" % (rho_test, p_test))
    print("  Criterion: violation rate < 10%% AND n_high_R >= 30")
    mt_pass = (not np.isnan(violation_rate)) and (violation_rate < 0.10) and (n_high_R >= 30)
    print("  Result: %s (n_high_R=%d, violation_rate=%.3f)" % (
        "PASS" if mt_pass else "FAIL",
        n_high_R,
        violation_rate if not np.isnan(violation_rate) else -1))

    # Breakdown by cluster type
    print("\n  Breakdown by cluster type:")
    for ctype in ["pure", "mixed-2", "mixed-5", "random"]:
        mask = np.array([t == ctype for t in test_types])
        if mask.sum() == 0:
            continue
        r_vals = test_R[mask]
        p_vals = test_purity[mask]
        print("    %8s: n=%d, R mean=%.4f, purity mean=%.4f" % (
            ctype, mask.sum(), np.nanmean(r_vals), np.mean(p_vals)))

    sys.stdout.flush()

    test3_results[model_name] = {
        "threshold_T": T,
        "quality_min_Q": Q_min,
        "train_n_clusters": len(train_clusters),
        "train_rho": float(rho_train),
        "train_p": float(p_train),
        "test_n_clusters": len(test_clusters),
        "n_high_R_test": n_high_R,
        "n_violations": n_violations,
        "violation_rate": float(violation_rate) if not np.isnan(violation_rate) else None,
        "ci_95_low": float(ci_low) if not np.isnan(ci_low) else None,
        "ci_95_high": float(ci_high) if not np.isnan(ci_high) else None,
        "test_rho": float(rho_test),
        "test_p": float(p_test),
        "pass": mt_pass,
    }

print()

# ===================================================================
# TEST 4: Adversarial (pure vs random, effect size)
# ===================================================================
print("=" * 80)
print("TEST 4: Adversarial (pure vs random clusters, effect size)")
print("=" * 80)

test4_results = {}

for model_name in MODEL_NAMES:
    print("\n--- Architecture: %s ---" % model_name)
    emb_matrix = embeddings_by_model[model_name]
    rng_adv = np.random.RandomState(SEED + 500)

    R_pure = []
    R_random = []
    E_pure_vals = []
    E_random_vals = []

    for cat_idx in range(n_categories):
        embs, _ = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng_adv)
        R_pure.append(formula.compute_R_simple(embs))
        E_pure_vals.append(formula.compute_E(embs))

    rng_adv2 = np.random.RandomState(SEED + 600)
    for _ in range(n_categories):
        embs, _ = get_random_docs(DOCS_PER_CLUSTER, emb_matrix, rng_adv2)
        R_random.append(formula.compute_R_simple(embs))
        E_random_vals.append(formula.compute_E(embs))

    R_pure = np.array(R_pure)
    R_random = np.array(R_random)
    E_pure_arr = np.array(E_pure_vals)
    E_random_arr = np.array(E_random_vals)

    effect_d = cohens_d(R_pure[~np.isnan(R_pure)], R_random[~np.isnan(R_random)])

    valid_pure = ~np.isnan(R_pure)
    valid_rand = ~np.isnan(R_random)
    if valid_pure.sum() > 0 and valid_rand.sum() > 0:
        u_stat, u_p = stats.mannwhitneyu(
            R_pure[valid_pure], R_random[valid_rand], alternative='greater')
    else:
        u_stat, u_p = 0, 1.0

    effect_d_E = cohens_d(E_pure_arr, E_random_arr)

    # Also R_full
    rng_adv3 = np.random.RandomState(SEED + 500)
    R_full_pure = []
    for cat_idx in range(n_categories):
        embs, _ = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng_adv3)
        R_full_pure.append(formula.compute_R_full(embs))
    rng_adv4 = np.random.RandomState(SEED + 600)
    R_full_random = []
    for _ in range(n_categories):
        embs, _ = get_random_docs(DOCS_PER_CLUSTER, emb_matrix, rng_adv4)
        R_full_random.append(formula.compute_R_full(embs))

    R_full_pure = np.array(R_full_pure)
    R_full_random = np.array(R_full_random)
    effect_d_full = cohens_d(
        R_full_pure[~np.isnan(R_full_pure)],
        R_full_random[~np.isnan(R_full_random)])

    adv_pass = effect_d > 0.8 and u_p < 0.01
    print("  R_simple: pure mean=%.4f, random mean=%.4f" % (
        np.nanmean(R_pure), np.nanmean(R_random)))
    print("  Cohen's d = %.4f, U p = %.2e" % (effect_d, u_p))
    print("  R_full: pure mean=%.4f, random mean=%.4f" % (
        np.nanmean(R_full_pure), np.nanmean(R_full_random)))
    print("  R_full Cohen's d = %.4f" % effect_d_full)
    print("  E alone: pure=%.4f, random=%.4f, d=%.4f" % (
        np.mean(E_pure_arr), np.mean(E_random_arr), effect_d_E))
    print("  Criterion: d > 0.8 AND p < 0.01 -> %s" % ("PASS" if adv_pass else "FAIL"))

    sys.stdout.flush()

    test4_results[model_name] = {
        "R_simple_pure_mean": float(np.nanmean(R_pure)),
        "R_simple_pure_std": float(np.nanstd(R_pure)),
        "R_simple_random_mean": float(np.nanmean(R_random)),
        "R_simple_random_std": float(np.nanstd(R_random)),
        "R_simple_cohens_d": float(effect_d),
        "R_simple_U_p": float(u_p),
        "R_full_pure_mean": float(np.nanmean(R_full_pure)),
        "R_full_random_mean": float(np.nanmean(R_full_random)),
        "R_full_cohens_d": float(effect_d_full),
        "E_alone_pure_mean": float(np.mean(E_pure_arr)),
        "E_alone_random_mean": float(np.mean(E_random_arr)),
        "E_alone_cohens_d": float(effect_d_E),
        "pass": adv_pass,
    }

print()

# ===================================================================
# AGGREGATE VERDICT (AUDIT-FIXED CRITERIA)
# ===================================================================
print("=" * 80)
print("AGGREGATE VERDICT (v3 audit-fixed criteria)")
print("=" * 80)

# --- Test 1: Component significance (two-sided, p < 0.05) ---
# A component "passes" if significant on >= 2/3 architectures
component_pass_counts = {"E": 0, "grad_S": 0, "sigma": 0, "Df": 0}
component_directions = {"E": [], "grad_S": [], "sigma": [], "Df": []}
component_effect_sizes = {"E": [], "grad_S": [], "sigma": [], "Df": []}

for model_name in MODEL_NAMES:
    r = test1_results[model_name]
    # E
    if r["E_test"]["significant"]:
        component_pass_counts["E"] += 1
    component_directions["E"].append(r["E_test"]["observed_direction"])
    component_effect_sizes["E"].append(r["E_test"]["cohens_d"])
    # grad_S
    if r["grad_S_test"]["significant"]:
        component_pass_counts["grad_S"] += 1
    component_directions["grad_S"].append(r["grad_S_test"]["observed_direction"])
    component_effect_sizes["grad_S"].append(r["grad_S_test"]["cohens_d"])
    # sigma (pure vs mixed OR cross-model CV)
    sig_pvm = r["sigma_test"]["significant_pure_vs_mixed"]
    sig_cm = cross_model_results["sigma_cross_model_significant"]
    if sig_pvm or sig_cm:
        component_pass_counts["sigma"] += 1
    component_directions["sigma"].append(r["sigma_test"]["observed_direction"])
    component_effect_sizes["sigma"].append(r["sigma_test"]["cohens_d"])
    # Df (pure vs mixed OR cross-model CV)
    df_pvm = r["Df_test"]["significant_pure_vs_mixed"]
    df_cm = cross_model_results["Df_cross_model_significant"]
    if df_pvm or df_cm:
        component_pass_counts["Df"] += 1
    component_directions["Df"].append(r["Df_test"]["observed_direction"])
    component_effect_sizes["Df"].append(r["Df_test"]["cohens_d"])

components_passing = 0
component_verdicts = {}
for comp, count in component_pass_counts.items():
    passes = count >= 2
    avg_d = np.mean(component_effect_sizes[comp])
    dirs = component_directions[comp]
    component_verdicts[comp] = {
        "pass_count": count,
        "total": len(MODEL_NAMES),
        "pass": passes,
        "avg_cohens_d": float(avg_d),
        "directions": dirs,
    }
    if passes:
        components_passing += 1
    print("  Component %s: %d/%d arch pass, avg d=%.2f, dirs=%s -> %s" % (
        comp, count, len(MODEL_NAMES), avg_d, dirs,
        "SIGNIFICANT" if passes else "NS"))

print("\n  Components with significant two-sided effect: %d/4" % components_passing)

# --- Test 2: Steiger test (R vs E-alone) ---
steiger_passes = 0
for model_name in MODEL_NAMES:
    r = test2_results[model_name]
    st = r["steiger_test"]
    sig = st["R0_significantly_better"]
    if sig:
        steiger_passes += 1
    print("  Steiger (%s): z=%.3f, p=%.4f -> %s" % (
        model_name, st["z_stat"], st["p_one_sided"],
        "R0 BETTER" if sig else "NS"))

steiger_pass_overall = steiger_passes >= 2
print("  Steiger pass on >= 2/3 architectures: %s (%d/3)" % (
    steiger_pass_overall, steiger_passes))

# --- Test 3: Modus tollens ---
mt_passes = 0
violation_rates = []
for model_name in MODEL_NAMES:
    r = test3_results[model_name]
    passes = r["pass"]
    if passes:
        mt_passes += 1
    vr = r["violation_rate"]
    if vr is not None:
        violation_rates.append(vr)
    vr_str = "%.3f" % vr if vr is not None else "N/A"
    ci_str = ""
    if r["ci_95_low"] is not None:
        ci_str = " CI=[%.3f, %.3f]" % (r["ci_95_low"], r["ci_95_high"])
    print("  Test 3 (%s): violation=%.3s, n_high_R=%d%s -> %s" % (
        model_name, vr_str, r["n_high_R_test"], ci_str,
        "PASS" if passes else "FAIL"))

mt_pass_overall = mt_passes >= 2
avg_violation = np.mean(violation_rates) if violation_rates else float('nan')

# --- Test 4: Adversarial ---
adv_passes = sum(1 for r in test4_results.values() if r["pass"])
adv_pass_overall = adv_passes >= 2
for model_name in MODEL_NAMES:
    r = test4_results[model_name]
    print("  Test 4 (%s): d=%.4f, p=%.2e -> %s" % (
        model_name, r["R_simple_cohens_d"], r["R_simple_U_p"],
        "PASS" if r["pass"] else "FAIL"))

# --- Apply pre-registered verdict criteria ---
print("\n" + "=" * 80)
print("APPLYING PRE-REGISTERED CRITERIA (v3)")
print("=" * 80)

# CONFIRMED: >= 3/4 components significant (two-sided, p<0.05)
#   AND Steiger R > E-alone on >= 2/3 architectures (p<0.05)
confirm = (components_passing >= 3) and steiger_pass_overall

# FALSIFIED: < 2/4 components significant
#   AND Steiger NS on all 3 architectures
falsify = (components_passing < 2) and (steiger_passes == 0)

if confirm:
    verdict = "CONFIRMED"
elif falsify:
    verdict = "FALSIFIED"
else:
    verdict = "INCONCLUSIVE"

print("\n  Components significant (two-sided): %d/4 (need >= 3 to confirm, < 2 to falsify)" % (
    components_passing))
print("  Steiger R>E passes: %d/3 (need >= 2 to confirm, 0 to falsify)" % steiger_passes)
print("  Modus tollens: %s (avg violation: %.3f)" % (
    "PASS" if mt_pass_overall else "FAIL",
    avg_violation if not np.isnan(avg_violation) else -1))
print("  Adversarial: %d/%d models pass" % (adv_passes, len(MODEL_NAMES)))
print()
print("  CONFIRM criteria: components >= 3 (%s) AND Steiger >= 2 (%s) => %s" % (
    components_passing >= 3, steiger_pass_overall, confirm))
print("  FALSIFY criteria: components < 2 (%s) AND Steiger == 0 (%s) => %s" % (
    components_passing < 2, steiger_passes == 0, falsify))
print()
print("  >>> VERDICT: %s <<<" % verdict)

elapsed_total = time.time() - t_start
print("\n  Total runtime: %.1fs (%.1f min)" % (elapsed_total, elapsed_total / 60))

# ===================================================================
# Save results
# ===================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


results = {
    "metadata": {
        "test": "Q02 v3 - Formula Falsification Criteria (Audit-Fixed)",
        "formula": "R = (E / grad_S) * sigma^Df",
        "dataset": "20 Newsgroups (sklearn)",
        "n_documents_total": len(all_texts),
        "n_documents_subsampled": n_docs,
        "n_categories": n_categories,
        "docs_per_cluster": DOCS_PER_CLUSTER,
        "subsample_per_category": SUBSAMPLE_PER_CAT,
        "models": MODEL_NAMES,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": elapsed_total,
        "audit_fixes": [
            "1. grad_S: two-sided test, no assumed direction",
            "2. R4=log(R0) removed from alternatives (monotonic invariant)",
            "3. sigma/Df tested across models + pure vs mixed",
            "4. Cluster count increased to 80 (40 pure + 40 mixed)",
            "5. Steiger's test for R vs E-alone",
            "6. Modus tollens n=200 clusters per split",
            "7. Pre-registered criteria updated",
        ],
    },
    "pre_registered_criteria": PRE_REGISTERED,
    "test1_component_falsification": test1_results,
    "test1_cross_model": cross_model_results,
    "test2_functional_form": {
        model_name: {
            "n_clusters": r["n_clusters"],
            "variant_correlations": r["variant_correlations"],
            "ranking": r["ranking"],
            "r0_rank": r["r0_rank"],
            "bootstrap_comparisons": r["bootstrap_comparisons"],
            "n_r0_wins_p01": r["n_r0_wins_p01"],
            "n_r0_loses": r["n_r0_loses"],
            "steiger_test": r["steiger_test"],
        }
        for model_name, r in test2_results.items()
    },
    "test3_modus_tollens": test3_results,
    "test4_adversarial": test4_results,
    "aggregate": {
        "component_verdicts": component_verdicts,
        "components_passing": components_passing,
        "steiger_passes": steiger_passes,
        "steiger_pass_overall": steiger_pass_overall,
        "test3_avg_violation_rate": float(avg_violation) if not np.isnan(avg_violation) else None,
        "test3_mt_pass": mt_pass_overall,
        "test4_adv_pass": adv_pass_overall,
        "confirm": confirm,
        "falsify": falsify,
        "verdict": verdict,
    },
}

results_path = os.path.join(RESULTS_DIR, "test_v3_q02_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print("\nResults saved to: %s" % results_path)
