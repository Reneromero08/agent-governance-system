"""
Q02 Fixed Test: Does R = (E / grad_S) * sigma^Df have proper falsification criteria?

Methodology fixes from v1:
1. Uses 20 Newsgroups (natural topic clusters) instead of STS-B pooling artifact
2. 60 clusters for functional form comparison (not 20) with bootstrap significance
3. 200 clusters for modus tollens (not 6 exceeding threshold) with train/test split
4. Three architectures for robustness
5. All pre-registered criteria stated before running

OPTIMIZATION: Subsample 400 docs per category (8000 total) before encoding.
Cluster sizes of 100 docs provide ample statistical power while keeping
pairwise computation tractable.

Date: 2026-02-06
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
DOCS_PER_CLUSTER = 100  # enough for stable pairwise stats (4950 pairs)
SUBSAMPLE_PER_CAT = 400  # max docs per category to encode

# ---------------------------------------------------------------------------
# Pre-registered criteria (stated before any data is loaded)
# ---------------------------------------------------------------------------
PRE_REGISTERED = {
    "test1_component_E": "Mann-Whitney U p < 0.01 for pure vs mixed clusters",
    "test1_component_grad_S": "Mann-Whitney U p < 0.01 for pure vs mixed clusters",
    "test1_component_sigma": "CV > 0.2 across 20 categories",
    "test1_component_Df": "CV > 0.1 across 20 categories",
    "test2_functional_form": "R0=E/grad_S outperforms >= 4 of 5 alternatives at p < 0.01 (bootstrap)",
    "test3_modus_tollens": "Violation rate < 10% on held-out test data with n >> 6",
    "test4_adversarial": "R distinguishes pure from random clusters (effect size d > 0.8)",
    "confirm": "4/4 components pass AND modus tollens < 10% AND R0 beats >= 4/5 alternatives",
    "falsify": ">= 3 components fail OR modus tollens > 20% OR R0 loses to >= 3 alternatives",
    "inconclusive": "Otherwise",
}

t_start = time.time()

print("=" * 80)
print("Q02 FIXED TEST: Formula Falsification Criteria")
print("=" * 80)
print(f"Seed: {SEED}")
print(f"Formula: R = (E / grad_S) * sigma^Df")
print(f"Data: 20 Newsgroups (sklearn)")
print(f"Timestamp: {datetime.now().isoformat()}")
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

print(f"  Total documents: {len(all_texts)}")
print(f"  Categories: {n_categories}")

# Subsample for encoding efficiency: max SUBSAMPLE_PER_CAT per category
rng_sub = np.random.RandomState(SEED)
subsample_indices = []
for cat_idx in range(n_categories):
    cat_mask = np.where(all_labels == cat_idx)[0]
    n_avail = len(cat_mask)
    n_take = min(SUBSAMPLE_PER_CAT, n_avail)
    chosen = rng_sub.choice(cat_mask, size=n_take, replace=False)
    subsample_indices.extend(chosen.tolist())
    print(f"  {cat_idx:2d}. {category_names[cat_idx]}: {n_avail} total, {n_take} subsampled")

subsample_indices = np.array(subsample_indices)
texts = [all_texts[i] for i in subsample_indices]
labels = all_labels[subsample_indices]
n_docs = len(texts)
print(f"\n  Subsampled documents: {n_docs}")
print()

# ---------------------------------------------------------------------------
# Step 2: Encode with 3 architectures
# ---------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer

MODEL_NAMES = [
    "all-MiniLM-L6-v2",          # 384-dim, general purpose
    "multi-qa-MiniLM-L6-cos-v1", # 384-dim, QA-tuned (different training objective)
    "paraphrase-MiniLM-L3-v2",   # 384-dim, paraphrase-tuned, 3-layer (different depth)
]

embeddings_by_model = {}
for model_name in MODEL_NAMES:
    print(f"Encoding with {model_name}...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, batch_size=512)
    embeddings_by_model[model_name] = embs
    elapsed = time.time() - t0
    print(f"  Shape: {embs.shape}, Time: {elapsed:.1f}s")
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


from scipy import stats


# ===================================================================
# TEST 1: Component-Level Falsification
# ===================================================================
print("=" * 80)
print("TEST 1: Component-Level Falsification")
print("=" * 80)

test1_results = {}

for model_name in MODEL_NAMES:
    print(f"\n--- Architecture: {model_name} ---")
    emb_matrix = embeddings_by_model[model_name]
    rng = np.random.RandomState(SEED)

    # Create 20 pure clusters and 20 mixed clusters
    pure_metrics = {"E": [], "grad_S": [], "sigma": [], "Df": [], "R_simple": [], "R_full": []}
    mixed_metrics = {"E": [], "grad_S": [], "sigma": [], "Df": [], "R_simple": [], "R_full": []}

    print("  Computing pure clusters (1 category each)...")
    for cat_idx in range(n_categories):
        embs, _ = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng)
        result = formula.compute_all(embs)
        for k in pure_metrics:
            pure_metrics[k].append(result[k])

    print("  Computing mixed clusters (random docs)...")
    for _ in range(n_categories):
        embs, _ = get_random_docs(DOCS_PER_CLUSTER, emb_matrix, rng)
        result = formula.compute_all(embs)
        for k in mixed_metrics:
            mixed_metrics[k].append(result[k])

    model_results = {}

    # --- E test: pure vs mixed ---
    E_pure = np.array(pure_metrics["E"])
    E_mixed = np.array(mixed_metrics["E"])
    u_stat, u_p = stats.mannwhitneyu(E_pure, E_mixed, alternative='greater')
    pooled_std = np.sqrt((np.std(E_pure)**2 + np.std(E_mixed)**2) / 2)
    effect_d = (np.mean(E_pure) - np.mean(E_mixed)) / pooled_std if pooled_std > 1e-10 else 0
    model_results["E_test"] = {
        "pure_mean": float(np.mean(E_pure)),
        "pure_std": float(np.std(E_pure)),
        "mixed_mean": float(np.mean(E_mixed)),
        "mixed_std": float(np.std(E_mixed)),
        "U_statistic": float(u_stat),
        "p_value": float(u_p),
        "cohens_d": float(effect_d),
        "pass": bool(u_p < 0.01),
    }
    print(f"  E test: pure={np.mean(E_pure):.4f}+/-{np.std(E_pure):.4f}, "
          f"mixed={np.mean(E_mixed):.4f}+/-{np.std(E_mixed):.4f}, "
          f"U p={u_p:.2e}, d={effect_d:.2f} -> {'PASS' if u_p < 0.01 else 'FAIL'}")

    # --- grad_S test: pure should have LOWER grad_S (more uniform similarities) ---
    gS_pure = np.array(pure_metrics["grad_S"])
    gS_mixed = np.array(mixed_metrics["grad_S"])
    u_stat, u_p = stats.mannwhitneyu(gS_mixed, gS_pure, alternative='greater')
    pooled_std = np.sqrt((np.std(gS_pure)**2 + np.std(gS_mixed)**2) / 2)
    effect_d = (np.mean(gS_mixed) - np.mean(gS_pure)) / pooled_std if pooled_std > 1e-10 else 0
    model_results["grad_S_test"] = {
        "pure_mean": float(np.mean(gS_pure)),
        "pure_std": float(np.std(gS_pure)),
        "mixed_mean": float(np.mean(gS_mixed)),
        "mixed_std": float(np.std(gS_mixed)),
        "U_statistic": float(u_stat),
        "p_value": float(u_p),
        "cohens_d": float(effect_d),
        "pass": bool(u_p < 0.01),
    }
    print(f"  grad_S test: pure={np.mean(gS_pure):.4f}+/-{np.std(gS_pure):.4f}, "
          f"mixed={np.mean(gS_mixed):.4f}+/-{np.std(gS_mixed):.4f}, "
          f"U p={u_p:.2e}, d={effect_d:.2f} -> {'PASS' if u_p < 0.01 else 'FAIL'}")

    # --- sigma test: CV across 20 categories ---
    sigma_vals = np.array(pure_metrics["sigma"])
    sigma_cv = float(np.std(sigma_vals) / np.mean(sigma_vals)) if np.mean(sigma_vals) > 0 else 0
    model_results["sigma_test"] = {
        "values": [float(v) for v in sigma_vals],
        "mean": float(np.mean(sigma_vals)),
        "std": float(np.std(sigma_vals)),
        "CV": sigma_cv,
        "pass": bool(sigma_cv > 0.2),
    }
    print(f"  sigma test: mean={np.mean(sigma_vals):.4f}, std={np.std(sigma_vals):.4f}, "
          f"CV={sigma_cv:.4f} -> {'PASS' if sigma_cv > 0.2 else 'FAIL'}")

    # --- Df test: CV across 20 categories ---
    Df_vals = np.array(pure_metrics["Df"])
    Df_cv = float(np.std(Df_vals) / np.mean(Df_vals)) if np.mean(Df_vals) > 0 else 0
    model_results["Df_test"] = {
        "values": [float(v) for v in Df_vals],
        "mean": float(np.mean(Df_vals)),
        "std": float(np.std(Df_vals)),
        "CV": Df_cv,
        "pass": bool(Df_cv > 0.1),
    }
    print(f"  Df test: mean={np.mean(Df_vals):.4f}, std={np.std(Df_vals):.4f}, "
          f"CV={Df_cv:.4f} -> {'PASS' if Df_cv > 0.1 else 'FAIL'}")

    test1_results[model_name] = model_results
    sys.stdout.flush()

print()
sys.stdout.flush()

# ===================================================================
# TEST 2: Functional Form Comparison (FIXED)
# ===================================================================
print("=" * 80)
print("TEST 2: Functional Form Comparison (n=60 clusters, bootstrap)")
print("=" * 80)

test2_results = {}

for model_name in MODEL_NAMES:
    print(f"\n--- Architecture: {model_name} ---")
    emb_matrix = embeddings_by_model[model_name]
    rng = np.random.RandomState(SEED + 1)

    cluster_embeddings = []
    cluster_purities = []
    cluster_types = []

    # 20 "pure" clusters (1 category each)
    print("  Creating 20 pure clusters...")
    for cat_idx in range(n_categories):
        embs, cat_labels = get_category_docs(cat_idx, DOCS_PER_CLUSTER, emb_matrix, rng)
        purity = compute_purity(cat_labels)
        cluster_embeddings.append(embs)
        cluster_purities.append(purity)
        cluster_types.append("pure")

    # 20 "mixed-2" clusters (50 docs from each of 2 random categories)
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

    # 20 "mixed-5" clusters (20 docs from each of 5 random categories)
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

    cluster_purities = np.array(cluster_purities)
    n_clusters = len(cluster_embeddings)
    print(f"  Total clusters: {n_clusters}")
    print(f"  Purity range: [{cluster_purities.min():.3f}, {cluster_purities.max():.3f}]")
    print(f"  Purity by type: pure={np.mean(cluster_purities[:20]):.3f}, "
          f"mixed-2={np.mean(cluster_purities[20:40]):.3f}, "
          f"mixed-5={np.mean(cluster_purities[40:60]):.3f}")

    # Compute 6 variants for each cluster
    print("  Computing formula variants...")
    variant_values = {
        "R0_E_over_gradS": [],
        "R1_E_over_gradS2": [],
        "R2_E_over_MAD": [],
        "R3_E_times_gradS": [],
        "R4_logE_minus_logGradS": [],
        "R5_E_alone": [],
    }

    for i, embs in enumerate(cluster_embeddings):
        E_val = formula.compute_E(embs)
        gS_val = formula.compute_grad_S(embs)
        mad_val = compute_MAD(embs)

        # R0 = E / grad_S (the claimed formula)
        r0 = E_val / gS_val if gS_val > 1e-10 else float('nan')
        # R1 = E / grad_S^2
        r1 = E_val / (gS_val**2) if gS_val > 1e-10 else float('nan')
        # R2 = E / MAD
        r2 = E_val / mad_val if mad_val > 1e-10 else float('nan')
        # R3 = E * grad_S (opposite direction)
        r3 = E_val * gS_val
        # R4 = log(E) - log(grad_S) (= log(E/grad_S))
        r4 = np.log(max(E_val, 1e-10)) - np.log(max(gS_val, 1e-10))
        # R5 = E alone (no grad_S at all)
        r5 = E_val

        variant_values["R0_E_over_gradS"].append(r0)
        variant_values["R1_E_over_gradS2"].append(r1)
        variant_values["R2_E_over_MAD"].append(r2)
        variant_values["R3_E_times_gradS"].append(r3)
        variant_values["R4_logE_minus_logGradS"].append(r4)
        variant_values["R5_E_alone"].append(r5)

    # Spearman correlation of each variant with purity
    print("\n  Spearman correlations with cluster purity (n=60):")
    variant_correlations = {}
    for name, vals in variant_values.items():
        vals_arr = np.array(vals)
        valid_mask = ~np.isnan(vals_arr)
        if valid_mask.sum() < 10:
            variant_correlations[name] = {"rho": float('nan'), "p": 1.0, "n_valid": int(valid_mask.sum())}
            continue
        rho, p = stats.spearmanr(vals_arr[valid_mask], cluster_purities[valid_mask])
        variant_correlations[name] = {
            "rho": float(rho),
            "p": float(p),
            "n_valid": int(valid_mask.sum()),
        }
        print(f"    {name}: rho={rho:.4f}, p={p:.2e}, n={valid_mask.sum()}")

    # Rank by |rho|
    ranked = sorted(variant_correlations.items(), key=lambda x: abs(x[1]["rho"]), reverse=True)
    print("\n  Ranking by |rho|:")
    r0_rank = None
    for rank, (name, vals) in enumerate(ranked, 1):
        marker = " <-- FORMULA" if name == "R0_E_over_gradS" else ""
        print(f"    {rank}. {name}: |rho|={abs(vals['rho']):.4f}, p={vals['p']:.2e}{marker}")
        if name == "R0_E_over_gradS":
            r0_rank = rank

    # Bootstrap pairwise comparison: R0 vs each alternative
    # Use fast rank-based Spearman to avoid scipy overhead in inner loop
    def fast_spearman(x, y):
        """Fast Spearman rho using numpy rank arrays."""
        n = len(x)
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        d = rx - ry
        return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))

    print("\n  Bootstrap pairwise comparisons (2000 resamples):")
    sys.stdout.flush()
    N_BOOTSTRAP = 2000
    boot_rng = np.random.RandomState(SEED + 100)

    r0_vals = np.array(variant_values["R0_E_over_gradS"])
    r0_wins = {}

    for alt_name, alt_vals_list in variant_values.items():
        if alt_name == "R0_E_over_gradS":
            continue
        alt_vals = np.array(alt_vals_list)

        # Only use indices where both are valid
        valid = ~(np.isnan(r0_vals) | np.isnan(alt_vals))
        if valid.sum() < 10:
            r0_wins[alt_name] = {"win_rate": float('nan'), "p_r0_better": float('nan')}
            continue

        r0_v = r0_vals[valid]
        alt_v = alt_vals[valid]
        purity_v = cluster_purities[valid]
        n_valid = len(purity_v)

        # Pre-generate all bootstrap indices for speed
        boot_indices = boot_rng.choice(n_valid, size=(N_BOOTSTRAP, n_valid), replace=True)

        r0_better_count = 0
        for b in range(N_BOOTSTRAP):
            idx = boot_indices[b]
            rho_r0 = fast_spearman(r0_v[idx], purity_v[idx])
            rho_alt = fast_spearman(alt_v[idx], purity_v[idx])
            if abs(rho_r0) > abs(rho_alt):
                r0_better_count += 1

        win_rate = r0_better_count / N_BOOTSTRAP
        p_r0_better = 1.0 - win_rate
        r0_wins[alt_name] = {
            "win_rate": float(win_rate),
            "p_r0_better": float(p_r0_better),
        }
        result_str = "R0 WINS" if p_r0_better < 0.01 else ("R0 LOSES" if win_rate < 0.5 else "TIE")
        print(f"    R0 vs {alt_name}: R0 wins {win_rate*100:.1f}% of bootstraps, "
              f"p(R0 better)={p_r0_better:.4f} -> {result_str}")
        sys.stdout.flush()

    # Count how many alternatives R0 beats at p < 0.01
    n_r0_wins = sum(1 for v in r0_wins.values()
                    if not np.isnan(v["p_r0_better"]) and v["p_r0_better"] < 0.01)
    n_r0_loses = sum(1 for v in r0_wins.values()
                     if not np.isnan(v["win_rate"]) and v["win_rate"] < 0.5)
    print(f"\n  R0 beats {n_r0_wins}/5 alternatives at p < 0.01")
    print(f"  R0 loses to {n_r0_loses}/5 alternatives (win rate < 50%)")

    test2_results[model_name] = {
        "n_clusters": n_clusters,
        "variant_correlations": variant_correlations,
        "ranking": [(name, vals) for name, vals in ranked],
        "r0_rank": r0_rank,
        "bootstrap_comparisons": r0_wins,
        "n_r0_wins_p01": n_r0_wins,
        "n_r0_loses": n_r0_loses,
    }

print()

# ===================================================================
# TEST 3: Modus Tollens (FIXED -- proper power)
# ===================================================================
print("=" * 80)
print("TEST 3: Modus Tollens (train/test split, 100 clusters each)")
print("=" * 80)

test3_results = {}

for model_name in MODEL_NAMES:
    print(f"\n--- Architecture: {model_name} ---")
    emb_matrix = embeddings_by_model[model_name]

    # Split document indices into train (60%) and test (40%) within each category
    # This ensures each split has balanced category representation
    rng_split = np.random.RandomState(SEED + 200)
    train_indices_set = set()
    test_indices_set = set()

    for cat_idx in range(n_categories):
        cat_mask = np.where(labels == cat_idx)[0]
        rng_split.shuffle(cat_mask)
        split_pt = int(0.6 * len(cat_mask))
        train_indices_set.update(cat_mask[:split_pt].tolist())
        test_indices_set.update(cat_mask[split_pt:].tolist())

    print(f"  Train: {len(train_indices_set)} docs, Test: {len(test_indices_set)} docs")

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
        """Create clusters of varying quality: pure, mixed-2, mixed-5, random."""
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
                embs, cat_labs = get_category_docs_split(c, cluster_size // 2, split_set, rng_local)
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
                embs, cat_labs = get_category_docs_split(c, cluster_size // 5, split_set, rng_local)
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

    # --- TRAIN: Calibrate threshold ---
    print("  Creating 100 training clusters...")
    rng_train = np.random.RandomState(SEED + 300)
    train_clusters = create_clusters(100, train_indices_set, rng_train, cluster_size=80)

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

    # Also compute Spearman correlation on train
    valid_mask = ~np.isnan(train_R)
    if valid_mask.sum() > 5:
        rho_train, p_train = stats.spearmanr(train_R[valid_mask], train_purity[valid_mask])
    else:
        rho_train, p_train = float('nan'), float('nan')

    print(f"  Train clusters: {len(train_clusters)}")
    print(f"  High-purity (>0.8): {(high_purity_mask & valid_train).sum()}")
    print(f"  High-R (>T): {high_R_mask.sum()}")
    print(f"  Calibrated threshold T = {T:.4f}")
    print(f"  Quality minimum Q_min = {Q_min:.4f}")
    print(f"  Train Spearman rho(R, purity) = {rho_train:.4f}, p = {p_train:.2e}")

    # --- TEST: Evaluate modus tollens ---
    print("  Creating 100 test clusters...")
    rng_test = np.random.RandomState(SEED + 400)
    test_clusters = create_clusters(100, test_indices_set, rng_test, cluster_size=80)

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

        # 95% CI for violation rate (Wilson score interval)
        from statsmodels.stats.proportion import proportion_confint
        ci_low, ci_high = proportion_confint(n_violations, n_high_R, alpha=0.05, method='wilson')
    else:
        n_violations = 0
        violation_rate = float('nan')
        ci_low, ci_high = (float('nan'), float('nan'))

    # Spearman on test
    if valid_test.sum() > 5:
        rho_test, p_test = stats.spearmanr(test_R[valid_test], test_purity[valid_test])
    else:
        rho_test, p_test = float('nan'), float('nan')

    print(f"\n  TEST RESULTS:")
    print(f"  Test clusters: {len(test_clusters)}")
    print(f"  Clusters with R > T ({T:.4f}): {n_high_R} / {len(test_clusters)}")
    print(f"  Violations (purity < Q_min={Q_min:.4f}): {n_violations} / {n_high_R}")
    if not np.isnan(violation_rate):
        print(f"  Violation rate: {violation_rate:.3f}")
    else:
        print(f"  Violation rate: N/A (no clusters with R > T)")
    if not np.isnan(ci_low):
        print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  Test Spearman rho(R, purity) = {rho_test:.4f}, p = {p_test:.2e}")
    print(f"  Criterion: violation rate < 10%")
    mt_pass = (not np.isnan(violation_rate)) and (violation_rate < 0.10)
    print(f"  Result: {'PASS' if mt_pass else 'FAIL'}")

    # Detailed breakdown by cluster type
    print(f"\n  Breakdown by cluster type:")
    for ctype in ["pure", "mixed-2", "mixed-5", "random"]:
        mask = np.array([t == ctype for t in test_types])
        if mask.sum() == 0:
            continue
        r_vals = test_R[mask]
        p_vals = test_purity[mask]
        print(f"    {ctype:8s}: n={mask.sum()}, R mean={np.nanmean(r_vals):.4f}, "
              f"purity mean={np.mean(p_vals):.4f}, R range=[{np.nanmin(r_vals):.4f}, {np.nanmax(r_vals):.4f}]")

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
    print(f"\n--- Architecture: {model_name} ---")
    emb_matrix = embeddings_by_model[model_name]
    rng_adv = np.random.RandomState(SEED + 500)

    # Create 20 pure clusters and 20 random clusters
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

    # Effect size for R_simple (Cohen's d)
    pooled_std = np.sqrt((np.nanstd(R_pure)**2 + np.nanstd(R_random)**2) / 2)
    effect_d = (np.nanmean(R_pure) - np.nanmean(R_random)) / pooled_std if pooled_std > 1e-10 else 0

    # Mann-Whitney U
    valid_pure = ~np.isnan(R_pure)
    valid_rand = ~np.isnan(R_random)
    if valid_pure.sum() > 0 and valid_rand.sum() > 0:
        u_stat, u_p = stats.mannwhitneyu(R_pure[valid_pure], R_random[valid_rand], alternative='greater')
    else:
        u_stat, u_p = 0, 1.0

    # Effect size for E alone
    pooled_std_E = np.sqrt((np.std(E_pure_arr)**2 + np.std(E_random_arr)**2) / 2)
    effect_d_E = (np.mean(E_pure_arr) - np.mean(E_random_arr)) / pooled_std_E if pooled_std_E > 1e-10 else 0

    # Also test R_full
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
    pooled_std_full = np.sqrt((np.nanstd(R_full_pure)**2 + np.nanstd(R_full_random)**2) / 2)
    effect_d_full = (np.nanmean(R_full_pure) - np.nanmean(R_full_random)) / pooled_std_full if pooled_std_full > 1e-10 else 0

    adv_pass = effect_d > 0.8 and u_p < 0.01
    print(f"  R_simple: pure mean={np.nanmean(R_pure):.4f}, random mean={np.nanmean(R_random):.4f}")
    print(f"  Cohen's d = {effect_d:.4f}, U p = {u_p:.2e}")
    print(f"  R_full: pure mean={np.nanmean(R_full_pure):.4f}, random mean={np.nanmean(R_full_random):.4f}")
    print(f"  R_full Cohen's d = {effect_d_full:.4f}")
    print(f"  E alone: pure mean={np.mean(E_pure_arr):.4f}, random mean={np.mean(E_random_arr):.4f}, d={effect_d_E:.4f}")
    print(f"  Criterion: d > 0.8 AND p < 0.01 -> {'PASS' if adv_pass else 'FAIL'}")

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
# AGGREGATE VERDICT
# ===================================================================
print("=" * 80)
print("AGGREGATE VERDICT")
print("=" * 80)

# Aggregate across models
component_pass_counts = {"E": 0, "grad_S": 0, "sigma": 0, "Df": 0}
for model_name in MODEL_NAMES:
    r = test1_results[model_name]
    if r["E_test"]["pass"]:
        component_pass_counts["E"] += 1
    if r["grad_S_test"]["pass"]:
        component_pass_counts["grad_S"] += 1
    if r["sigma_test"]["pass"]:
        component_pass_counts["sigma"] += 1
    if r["Df_test"]["pass"]:
        component_pass_counts["Df"] += 1

# A component passes if it passes on >= 2 of 3 architectures
components_passing = 0
component_verdicts = {}
for comp, count in component_pass_counts.items():
    passes = count >= 2
    component_verdicts[comp] = {"pass_count": count, "total": len(MODEL_NAMES), "pass": passes}
    if passes:
        components_passing += 1
    print(f"  Component {comp}: {count}/{len(MODEL_NAMES)} architectures pass -> {'PASS' if passes else 'FAIL'}")

print(f"\n  Components passing: {components_passing}/4")

# Test 2: Functional form
test2_pass_any = False
r0_loses_counts = []
for model_name in MODEL_NAMES:
    r = test2_results[model_name]
    wins = r["n_r0_wins_p01"]
    loses = r["n_r0_loses"]
    r0_loses_counts.append(loses)
    print(f"  Test 2 ({model_name}): R0 beats {wins}/5 at p<0.01, loses to {loses}/5")
    if wins >= 4:
        test2_pass_any = True

avg_r0_loses = np.mean(r0_loses_counts)
print(f"  Average R0 loses: {avg_r0_loses:.1f}/5")

# Test 3: Modus tollens
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
    vr_str = f"{vr:.3f}" if vr is not None else "N/A"
    print(f"  Test 3 ({model_name}): violation rate = {vr_str}, "
          f"n_high_R = {r['n_high_R_test']} -> {'PASS' if passes else 'FAIL'}")

mt_pass_overall = mt_passes >= 2
avg_violation = np.mean(violation_rates) if violation_rates else float('nan')

# Test 4: Adversarial
adv_passes = sum(1 for r in test4_results.values() if r["pass"])
adv_pass_overall = adv_passes >= 2
for model_name in MODEL_NAMES:
    r = test4_results[model_name]
    print(f"  Test 4 ({model_name}): d={r['R_simple_cohens_d']:.4f} -> {'PASS' if r['pass'] else 'FAIL'}")

# --- Apply pre-registered criteria ---
print("\n" + "=" * 80)
print("APPLYING PRE-REGISTERED CRITERIA")
print("=" * 80)

# CONFIRM: 4/4 components pass AND modus tollens < 10% AND R0 beats >= 4/5 alternatives
confirm = (components_passing >= 4) and mt_pass_overall and test2_pass_any

# FALSIFY: >= 3 components fail OR modus tollens > 20% OR R0 loses to >= 3 alternatives
components_failing = 4 - components_passing
mt_falsify = (not np.isnan(avg_violation)) and (avg_violation > 0.20)
r0_loses_falsify = avg_r0_loses >= 3

falsify = (components_failing >= 3) or mt_falsify or r0_loses_falsify

if confirm:
    verdict = "CONFIRMED"
elif falsify:
    verdict = "FALSIFIED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n  Components passing: {components_passing}/4 (need 4 to confirm, <=1 to falsify)")
print(f"  Components failing: {components_failing}/4")
print(f"  Modus tollens overall: {'PASS' if mt_pass_overall else 'FAIL'} (avg violation: {avg_violation:.3f})")
print(f"  Functional form: R0 loses on avg to {avg_r0_loses:.1f}/5 alternatives")
print(f"  Adversarial: {adv_passes}/{len(MODEL_NAMES)} models pass")
print(f"\n  CONFIRM criteria met: {confirm}")
print(f"  FALSIFY criteria met: {falsify}")
print(f"    - Components failing >= 3: {components_failing >= 3} ({components_failing} failing)")
print(f"    - Modus tollens > 20%: {mt_falsify} (avg={avg_violation:.3f})")
print(f"    - R0 loses to >= 3: {r0_loses_falsify} (avg loses={avg_r0_loses:.1f})")
print(f"\n  >>> VERDICT: {verdict} <<<")

elapsed_total = time.time() - t_start
print(f"\n  Total runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

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
        "test": "Q02 Fixed - Formula Falsification Criteria",
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
    },
    "pre_registered_criteria": PRE_REGISTERED,
    "test1_component_falsification": test1_results,
    "test2_functional_form": {
        model_name: {
            "n_clusters": r["n_clusters"],
            "variant_correlations": r["variant_correlations"],
            "ranking": r["ranking"],
            "r0_rank": r["r0_rank"],
            "bootstrap_comparisons": r["bootstrap_comparisons"],
            "n_r0_wins_p01": r["n_r0_wins_p01"],
            "n_r0_loses": r["n_r0_loses"],
        }
        for model_name, r in test2_results.items()
    },
    "test3_modus_tollens": test3_results,
    "test4_adversarial": test4_results,
    "aggregate": {
        "component_verdicts": component_verdicts,
        "components_passing": components_passing,
        "components_failing": components_failing,
        "test2_r0_avg_loses": float(avg_r0_loses),
        "test3_avg_violation_rate": float(avg_violation) if not np.isnan(avg_violation) else None,
        "test3_mt_pass": mt_pass_overall,
        "test4_adv_pass": adv_pass_overall,
        "confirm": confirm,
        "falsify": falsify,
        "verdict": verdict,
    },
}

results_path = os.path.join(RESULTS_DIR, "test_v2_q02_fixed_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print(f"\nResults saved to: {results_path}")
