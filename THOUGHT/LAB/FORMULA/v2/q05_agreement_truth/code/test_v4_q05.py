"""
Q05 v4 Test: Does High Local Agreement (High R) Reveal Truth?

Fixes from v3 AUDIT_v3.md:
  - BUG-1: Steiger Z-test variance formula corrected to ZPF (1980)
    Old (wrong): var = (2/(n-3)) * (1-r12) / (1-r_bar^2)^2
    New (correct): var = 2*(1-r12) / ((n-3)*(1+r12))
  - STAT-1: Amplification ratio (R_infl/E_infl) was tautological (= 1/gradS_change).
    Replaced with direct component analysis:
      - E_clean vs E_biased
      - grad_S_clean vs grad_S_biased
      - R_clean vs R_biased
      - grad_S dampening factor = grad_S_biased / grad_S_clean
        If > 1 => grad_S dampens bias (good for R)
        If < 1 => grad_S amplifies bias (bad for R)

Pre-registered criteria (v4):
  FALSIFIED:    grad_S dampening < 1.0 on >= 2/3 architectures
                AND Steiger shows R does NOT outperform E on >= 2/3
  CONFIRMED:    grad_S dampening > 1.0 on >= 2/3 architectures
                AND Steiger shows R outperforms E (p < 0.05) on >= 2/3
  INCONCLUSIVE: otherwise

Architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
Seed: 42
"""

import sys
import os
import json
import time
import math
import warnings
import importlib.util
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import functools
print = functools.partial(print, flush=True)

# ---------- Import formula ----------
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

compute_E = formula.compute_E
compute_grad_S = formula.compute_grad_S
compute_R_simple = formula.compute_R_simple

# ---------- Paths ----------
REPO_ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system")
RESULTS_DIR = REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "q05_agreement_truth" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

MODEL_SPECS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
]


# =============================================================================
# Utilities
# =============================================================================

def safe_R(embeddings):
    """Compute R_simple with NaN fallback."""
    if embeddings.shape[0] < 3:
        return float('nan')
    return compute_R_simple(embeddings)


def safe_E(embeddings):
    """Compute E with NaN fallback."""
    if embeddings.shape[0] < 2:
        return float('nan')
    return compute_E(embeddings)


def safe_grad_S(embeddings):
    """Compute grad_S with NaN fallback."""
    if embeddings.shape[0] < 2:
        return float('nan')
    return compute_grad_S(embeddings)


def cluster_purity(labels):
    """Fraction of items belonging to the dominant class."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def steiger_z_test(r1, r2, r12, n):
    """
    Steiger's (1980) ZPF test for comparing two dependent correlations.
    Tests H0: rho(X,Y1) = rho(X,Y2) where Y1 and Y2 share the same X.

    r1  = cor(X, Y1)  -- e.g., cor(purity, R)
    r2  = cor(X, Y2)  -- e.g., cor(purity, E)
    r12 = cor(Y1, Y2) -- e.g., cor(R, E)
    n   = sample size

    Returns (z_stat, p_value_two_sided).
    Positive z means r1 > r2.

    Correct formula (Steiger 1980):
        var(z1 - z2) = 2 * (1 - r12) / ((n - 3) * (1 + r12))
    """
    if n < 4:
        return float('nan'), float('nan')

    # Fisher z-transform
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))

    # Steiger (1980) ZPF variance
    denom = (1.0 + r12)
    if abs(denom) < 1e-10:
        return float('nan'), float('nan')

    var_diff = 2.0 * (1.0 - r12) / ((n - 3) * (1.0 + r12))
    if var_diff <= 0:
        return float('nan'), float('nan')

    z_stat = (z1 - z2) / np.sqrt(var_diff)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_val)


# =============================================================================
# Data Loading
# =============================================================================

def load_20newsgroups(max_docs=5000):
    """Load 20 Newsgroups dataset with stratified subsample."""
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    print(f"Loaded 20 Newsgroups: {len(data.data)} documents, {len(data.target_names)} categories")

    valid_idx = [i for i, doc in enumerate(data.data) if len(doc.strip()) >= 20]
    all_texts = [data.data[i] for i in valid_idx]
    all_labels = [data.target[i] for i in valid_idx]
    target_names = data.target_names
    print(f"After filtering short docs: {len(all_texts)} documents")

    if len(all_texts) > max_docs:
        np.random.seed(42)
        all_labels_arr = np.array(all_labels)
        chosen = []
        n_cats = len(target_names)
        per_cat = max_docs // n_cats
        for cat in range(n_cats):
            cat_idx = np.where(all_labels_arr == cat)[0]
            n_take = min(per_cat, len(cat_idx))
            chosen.extend(np.random.choice(cat_idx, n_take, replace=False).tolist())
        remaining = max_docs - len(chosen)
        if remaining > 0:
            leftover = list(set(range(len(all_texts))) - set(chosen))
            chosen.extend(np.random.choice(leftover, min(remaining, len(leftover)), replace=False).tolist())
        chosen.sort()
        texts = [all_texts[i] for i in chosen]
        labels = [all_labels[i] for i in chosen]
        print(f"Stratified subsample: {len(texts)} documents ({per_cat} per category target)")
    else:
        texts = all_texts
        labels = all_labels

    cat_counts = Counter(labels)
    print(f"Category counts: min={min(cat_counts.values())}, max={max(cat_counts.values())}, "
          f"mean={np.mean(list(cat_counts.values())):.0f}")
    return texts, labels, target_names


def encode_with_model(model_name, texts, batch_size=32):
    """Encode texts with sentence-transformer. No character truncation (use model-native tokenization)."""
    import gc
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True,
        batch_size=batch_size
    )
    elapsed = time.time() - t0
    print(f"  Done: shape {embeddings.shape} in {elapsed:.1f}s")
    del model
    gc.collect()
    return embeddings


# =============================================================================
# Cluster Construction -- Continuous Purity (10 levels: 0.1 to 1.0)
# =============================================================================

def build_continuous_purity_clusters(texts, labels, target_names):
    """
    Build ~80 clusters with continuous purity from 0.1 to 1.0 in 0.1 steps.
    8 clusters per purity level.
    """
    np.random.seed(42)
    labels_arr = np.array(labels)
    n_categories = len(target_names)
    cluster_size = 60
    clusters_per_level = 8
    purity_levels = [round(x * 0.1, 1) for x in range(1, 11)]  # 0.1 to 1.0

    cat_indices = {}
    for cat in range(n_categories):
        cat_indices[cat] = np.where(labels_arr == cat)[0].tolist()
        np.random.shuffle(cat_indices[cat])

    clusters = []
    all_categories = list(range(n_categories))

    for purity_target in purity_levels:
        for ci in range(clusters_per_level):
            dominant_cat = all_categories[(len(clusters)) % n_categories]
            n_dominant = max(1, int(round(cluster_size * purity_target)))
            n_other = cluster_size - n_dominant

            # Draw dominant docs
            dom_pool = cat_indices[dominant_cat]
            if len(dom_pool) < n_dominant:
                chosen_dom = np.random.choice(dom_pool, n_dominant, replace=True).tolist()
            else:
                chosen_dom = np.random.choice(dom_pool, n_dominant, replace=False).tolist()

            # Draw other docs from non-dominant categories
            other_cats = [c for c in all_categories if c != dominant_cat]
            other_pool = []
            for oc in other_cats:
                other_pool.extend(cat_indices[oc])
            if n_other > 0 and len(other_pool) > 0:
                chosen_other = np.random.choice(other_pool, min(n_other, len(other_pool)), replace=False).tolist()
            else:
                chosen_other = []

            all_chosen = chosen_dom + chosen_other
            cluster_labels = [labels[j] for j in all_chosen]
            actual_purity = cluster_purity(cluster_labels)

            clusters.append({
                'indices': all_chosen,
                'labels': cluster_labels,
                'target_purity': purity_target,
                'actual_purity': actual_purity,
                'dominant_cat': dominant_cat,
                'description': f"purity_{purity_target:.1f}_{ci}",
            })

    print(f"\nBuilt {len(clusters)} clusters across {len(purity_levels)} purity levels")
    for plevel in purity_levels:
        subset = [c for c in clusters if c['target_purity'] == plevel]
        actual_ps = [c['actual_purity'] for c in subset]
        print(f"  target={plevel:.1f}: n={len(subset)}, actual_purity mean={np.mean(actual_ps):.3f} std={np.std(actual_ps):.3f}")

    return clusters


# =============================================================================
# TEST 1: Agreement-Truth Correlation with Continuous Purity + Steiger's Test
# =============================================================================

def test1_purity_correlation(clusters, all_embeddings_by_model):
    """
    For each cluster, compute R_simple and E.
    Correlate with actual_purity via Spearman.
    Use Steiger's ZPF test to compare cor(purity, R) vs cor(purity, E).
    """
    print("\n" + "=" * 70)
    print("TEST 1: Purity Correlation (continuous, 10 levels, Steiger ZPF)")
    print("=" * 70)

    purities = np.array([c['actual_purity'] for c in clusters])
    results_by_model = {}

    for model_name in MODEL_SPECS:
        print(f"\n--- {model_name} ---")
        embeddings = all_embeddings_by_model[model_name]

        R_vals = []
        E_vals = []

        for cluster in clusters:
            idx = cluster['indices']
            embs = embeddings[idx]
            R_vals.append(safe_R(embs))
            E_vals.append(safe_E(embs))

        R_arr = np.array(R_vals)
        E_arr = np.array(E_vals)

        valid = ~(np.isnan(R_arr) | np.isnan(E_arr))
        n_valid = int(valid.sum())

        rho_R, p_R = stats.spearmanr(R_arr[valid], purities[valid])
        rho_E, p_E = stats.spearmanr(E_arr[valid], purities[valid])
        rho_RE, _ = stats.spearmanr(R_arr[valid], E_arr[valid])

        # Steiger's ZPF test: is rho_R significantly > rho_E?
        z_steiger, p_steiger = steiger_z_test(
            float(rho_R), float(rho_E), float(rho_RE), n_valid
        )

        R_outperforms = bool(abs(rho_R) > abs(rho_E))
        steiger_significant = bool(
            not math.isnan(z_steiger) and not math.isnan(p_steiger)
            and z_steiger > 0 and p_steiger < 0.05
        )

        print(f"  n_valid = {n_valid}")
        print(f"  Spearman(R, purity) = {rho_R:.4f}, p = {p_R:.2e}")
        print(f"  Spearman(E, purity) = {rho_E:.4f}, p = {p_E:.2e}")
        print(f"  Spearman(R, E)      = {rho_RE:.4f}")
        print(f"  Steiger ZPF: Z = {z_steiger:.4f}, p = {p_steiger:.6f}")
        print(f"  R outperforms E (magnitude): {R_outperforms}")
        print(f"  Steiger significant (Z>0, p<0.05): {steiger_significant}")

        results_by_model[model_name] = {
            "rho_R_purity": float(rho_R),
            "p_R_purity": float(p_R),
            "rho_E_purity": float(rho_E),
            "p_E_purity": float(p_E),
            "rho_R_E": float(rho_RE),
            "steiger_z": float(z_steiger) if not math.isnan(z_steiger) else None,
            "steiger_p": float(p_steiger) if not math.isnan(p_steiger) else None,
            "R_outperforms_E": R_outperforms,
            "steiger_significant_R_better": steiger_significant,
            "n_valid": n_valid,
            "R_values": [float(x) for x in R_arr],
            "E_values": [float(x) for x in E_arr],
        }

    return {
        "test": "Purity Correlation with Steiger ZPF Test",
        "note": "Steiger (1980) ZPF formula: var = 2*(1-r12)/((n-3)*(1+r12))",
        "n_clusters": len(clusters),
        "n_purity_levels": 10,
        "purities": [float(p) for p in purities],
        "results_by_model": results_by_model,
    }


# =============================================================================
# TEST 2: Bias Attack -- Direct Component Analysis (NO tautological ratio)
# =============================================================================

def test2_bias_attack(texts, labels, all_embeddings_by_model):
    """
    Bias attack with direct component analysis. No amplification ratio.

    For each architecture and bias phrase:
    - Compute E_clean, grad_S_clean, R_clean per cluster
    - Apply bias, recompute E_biased, grad_S_biased, R_biased
    - Report directly:
        E change:     E_biased / E_clean
        grad_S change: grad_S_biased / grad_S_clean  (the "dampening factor")
        R change:     R_biased / R_clean

    Key metric: grad_S_dampening = grad_S_biased / grad_S_clean
      > 1.0 => dispersion increases under bias => R is dampened relative to E (good)
      < 1.0 => dispersion decreases under bias => R is amplified relative to E (bad)

    Note: R_change = E_change / grad_S_change by definition (R = E/grad_S).
    This is not a bug -- it is an acknowledged structural property.
    The test focuses on whether grad_S *empirically* increases or decreases.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Bias Attack -- Direct Component Analysis")
    print("=" * 70)
    print("  NOTE: R = E/grad_S, so R_change = E_change/grad_S_change by definition.")
    print("  The key empirical question: does grad_S increase (dampen) or decrease")
    print("  (amplify) under bias attack?")

    np.random.seed(42)

    n_docs = len(texts)
    cluster_size = 50
    n_clusters = 20

    cluster_indices = []
    for i in range(n_clusters):
        chosen = np.random.choice(n_docs, cluster_size, replace=False)
        cluster_indices.append(chosen)

    bias_phrases = [
        "In conclusion, ",
        "According to recent studies, ",
        "The committee determined that ",
    ]

    from sentence_transformers import SentenceTransformer
    import gc

    results_by_model = {}

    for model_name in MODEL_SPECS:
        print(f"\n--- {model_name} ---")
        clean_embeddings = all_embeddings_by_model[model_name]

        # Compute clean E, grad_S, R for each cluster
        clean_R = []
        clean_E = []
        clean_gradS = []
        for idx in cluster_indices:
            embs = clean_embeddings[idx]
            clean_R.append(safe_R(embs))
            clean_E.append(safe_E(embs))
            clean_gradS.append(safe_grad_S(embs))

        model = SentenceTransformer(model_name)

        bias_results = []
        for bias_phrase in bias_phrases:
            print(f"\n  Bias: '{bias_phrase}'")

            # Collect all unique indices across clusters
            all_cluster_idx = np.unique(np.concatenate(cluster_indices))
            biased_text_list = [bias_phrase + texts[i] for i in all_cluster_idx]

            biased_embs_raw = model.encode(
                biased_text_list, show_progress_bar=False,
                convert_to_numpy=True, batch_size=32
            )
            biased_emb_map = {}
            for j, i in enumerate(all_cluster_idx):
                biased_emb_map[int(i)] = biased_embs_raw[j]

            E_changes = []
            gradS_changes = []
            R_changes = []
            per_cluster = []

            for ci, idx in enumerate(cluster_indices):
                biased_embs = np.array([biased_emb_map[int(k)] for k in idx])

                r_b = safe_R(biased_embs)
                e_b = safe_E(biased_embs)
                g_b = safe_grad_S(biased_embs)

                r_c = clean_R[ci]
                e_c = clean_E[ci]
                g_c = clean_gradS[ci]

                # Ratios (biased / clean)
                e_change = e_b / e_c if (not math.isnan(e_c) and abs(e_c) > 1e-10 and not math.isnan(e_b)) else float('nan')
                g_change = g_b / g_c if (not math.isnan(g_c) and abs(g_c) > 1e-10 and not math.isnan(g_b)) else float('nan')
                r_change = r_b / r_c if (not math.isnan(r_c) and abs(r_c) > 1e-10 and not math.isnan(r_b)) else float('nan')

                if not math.isnan(e_change):
                    E_changes.append(e_change)
                if not math.isnan(g_change):
                    gradS_changes.append(g_change)
                if not math.isnan(r_change):
                    R_changes.append(r_change)

                per_cluster.append({
                    "R_clean": float(r_c) if not math.isnan(r_c) else None,
                    "R_biased": float(r_b) if not math.isnan(r_b) else None,
                    "E_clean": float(e_c) if not math.isnan(e_c) else None,
                    "E_biased": float(e_b) if not math.isnan(e_b) else None,
                    "gradS_clean": float(g_c) if not math.isnan(g_c) else None,
                    "gradS_biased": float(g_b) if not math.isnan(g_b) else None,
                    "E_change": float(e_change) if not math.isnan(e_change) else None,
                    "gradS_dampening": float(g_change) if not math.isnan(g_change) else None,
                    "R_change": float(r_change) if not math.isnan(r_change) else None,
                })

            mean_E_change = float(np.mean(E_changes)) if E_changes else float('nan')
            mean_gradS_dampening = float(np.mean(gradS_changes)) if gradS_changes else float('nan')
            mean_R_change = float(np.mean(R_changes)) if R_changes else float('nan')
            std_gradS_dampening = float(np.std(gradS_changes)) if gradS_changes else float('nan')

            # One-sample t-test: is grad_S_dampening significantly different from 1?
            if len(gradS_changes) >= 3:
                t_stat, t_pval = stats.ttest_1samp(gradS_changes, 1.0)
                gradS_t_stat = float(t_stat)
                gradS_t_pval = float(t_pval)
            else:
                gradS_t_stat = float('nan')
                gradS_t_pval = float('nan')

            print(f"    E change (biased/clean):      mean={mean_E_change:.4f}")
            print(f"    grad_S dampening (biased/clean): mean={mean_gradS_dampening:.4f} +/- {std_gradS_dampening:.4f}")
            print(f"    R change (biased/clean):      mean={mean_R_change:.4f}")
            print(f"    grad_S != 1.0 t-test: t={gradS_t_stat:.3f}, p={gradS_t_pval:.4f}")
            if mean_gradS_dampening > 1.0:
                print(f"    --> grad_S INCREASES under bias (dampens R inflation)")
            else:
                print(f"    --> grad_S DECREASES under bias (amplifies R inflation)")

            bias_results.append({
                "bias_phrase": bias_phrase,
                "mean_E_change": mean_E_change if not math.isnan(mean_E_change) else None,
                "mean_gradS_dampening": mean_gradS_dampening if not math.isnan(mean_gradS_dampening) else None,
                "std_gradS_dampening": std_gradS_dampening if not math.isnan(std_gradS_dampening) else None,
                "mean_R_change": mean_R_change if not math.isnan(mean_R_change) else None,
                "gradS_ttest_stat": gradS_t_stat if not math.isnan(gradS_t_stat) else None,
                "gradS_ttest_pval": gradS_t_pval if not math.isnan(gradS_t_pval) else None,
                "n_valid": len(gradS_changes),
                "per_cluster": per_cluster,
            })

        del model
        gc.collect()

        # Summary across bias phrases for this model
        all_dampening = [br["mean_gradS_dampening"] for br in bias_results if br["mean_gradS_dampening"] is not None]
        mean_dampening = float(np.mean(all_dampening)) if all_dampening else None
        min_dampening = float(np.min(all_dampening)) if all_dampening else None
        all_dampen_above_1 = all(d > 1.0 for d in all_dampening) if all_dampening else False

        results_by_model[model_name] = {
            "bias_results": bias_results,
            "mean_dampening_across_phrases": mean_dampening,
            "min_dampening_across_phrases": min_dampening,
            "all_dampening_above_1": bool(all_dampen_above_1),
            "clean_R_values": [float(x) if not math.isnan(x) else None for x in clean_R],
            "clean_E_values": [float(x) if not math.isnan(x) else None for x in clean_E],
            "clean_gradS_values": [float(x) if not math.isnan(x) else None for x in clean_gradS],
        }

        print(f"\n  Model summary: mean_dampening={mean_dampening}, min_dampening={min_dampening}, all>1={all_dampen_above_1}")

    return {
        "test": "Bias Attack Direct Component Analysis",
        "note": "grad_S_dampening = grad_S_biased/grad_S_clean. >1 means dampens (good), <1 means amplifies (bad). "
                "R_change = E_change/grad_S_dampening by definition (R=E/grad_S), acknowledged not independent.",
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "bias_phrases": bias_phrases,
        "results_by_model": results_by_model,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import gc
    print("Q05 v4 TEST: DOES HIGH AGREEMENT (HIGH R) REVEAL TRUTH?")
    print("=" * 70)
    print(f"Seed: 42")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Fixes from v3:")
    print("  - BUG-1: Steiger variance corrected to ZPF (1980)")
    print("  - STAT-1: Tautological amplification ratio replaced with")
    print("    direct grad_S dampening analysis")
    print()
    print("Pre-registered criteria (v4):")
    print("  FALSIFIED:    grad_S dampening < 1.0 on >= 2/3 archs")
    print("                AND Steiger R not better on >= 2/3")
    print("  CONFIRMED:    grad_S dampening > 1.0 on >= 2/3 archs")
    print("                AND Steiger R > E (p<0.05) on >= 2/3")
    print("  INCONCLUSIVE: otherwise")
    print()

    total_start = time.time()

    # ---- Load data ----
    texts, labels, target_names = load_20newsgroups()

    # ---- Encode with all 3 architectures ----
    all_embeddings = {}
    for mname in MODEL_SPECS:
        t0 = time.time()
        all_embeddings[mname] = encode_with_model(mname, texts)
        print(f"  Total time for {mname}: {time.time() - t0:.1f}s")
        gc.collect()

    # ---- Build clusters with continuous purity ----
    clusters = build_continuous_purity_clusters(texts, labels, target_names)

    # ---- Run Tests ----
    all_results = {
        "metadata": {
            "test": "Q05 v4",
            "question": "Does high local agreement (high R) reveal truth?",
            "seed": 42,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "n_documents": len(texts),
            "n_categories": len(target_names),
            "architectures": MODEL_SPECS,
            "target_names": list(target_names),
            "fixes": [
                "BUG-1: Steiger ZPF variance = 2*(1-r12)/((n-3)*(1+r12))",
                "STAT-1: Tautological amplification ratio replaced with direct grad_S dampening",
            ],
            "pre_registered_criteria": {
                "FALSIFIED": "grad_S dampening < 1.0 on >= 2/3 archs AND Steiger R not better on >= 2/3",
                "CONFIRMED": "grad_S dampening > 1.0 on >= 2/3 archs AND Steiger R>E (p<0.05) on >= 2/3",
                "INCONCLUSIVE": "otherwise",
            },
        }
    }

    # Test 1: Purity correlation with Steiger's ZPF test
    t1_start = time.time()
    r1 = test1_purity_correlation(clusters, all_embeddings)
    r1["duration_s"] = time.time() - t1_start
    all_results["test1_purity_correlation"] = r1

    # Test 2: Bias attack direct component analysis
    t2_start = time.time()
    r2 = test2_bias_attack(texts, labels, all_embeddings)
    r2["duration_s"] = time.time() - t2_start
    all_results["test2_bias_attack"] = r2

    # =================================================================
    # VERDICT DETERMINATION
    # =================================================================
    print("\n" + "=" * 70)
    print("VERDICT DETERMINATION")
    print("=" * 70)

    # --- Criterion A: grad_S dampening ---
    print("\n--- Criterion A: grad_S dampening factor ---")
    print("  (grad_S_biased / grad_S_clean: >1 dampens, <1 amplifies)")
    n_dampens = 0   # models where ALL phrases have dampening > 1.0
    n_amplifies = 0  # models where ANY phrase has dampening < 1.0
    dampening_details = {}

    for model_name in MODEL_SPECS:
        mr = r2["results_by_model"][model_name]
        mean_d = mr["mean_dampening_across_phrases"]
        min_d = mr["min_dampening_across_phrases"]
        all_above = mr["all_dampening_above_1"]
        print(f"  {model_name}:")
        print(f"    mean dampening = {mean_d}")
        print(f"    min dampening  = {min_d}")
        print(f"    all phrases > 1.0: {all_above}")

        if all_above:
            n_dampens += 1
            print(f"    --> DAMPENS (grad_S increases under bias)")
        else:
            n_amplifies += 1
            print(f"    --> AMPLIFIES or MIXED (grad_S does not consistently increase)")

        dampening_details[model_name] = {
            "mean_dampening": mean_d,
            "min_dampening": min_d,
            "all_phrases_dampen": all_above,
        }

    dampens_criterion = n_dampens >= 2
    amplifies_criterion = n_amplifies >= 2

    print(f"\n  Models where grad_S dampens on ALL phrases: {n_dampens}/3")
    print(f"  Models where grad_S amplifies on ANY phrase: {n_amplifies}/3")
    print(f"  Dampening on >= 2/3: {dampens_criterion}")
    print(f"  Amplification on >= 2/3: {amplifies_criterion}")

    # --- Criterion B: Steiger's ZPF test R > E for purity ---
    print("\n--- Criterion B: Steiger ZPF test R > E ---")
    n_steiger_R_better = 0
    n_steiger_R_not_better = 0
    steiger_details = {}

    for model_name in MODEL_SPECS:
        mr = r1["results_by_model"][model_name]
        z = mr["steiger_z"]
        p = mr["steiger_p"]
        r_rho = mr["rho_R_purity"]
        e_rho = mr["rho_E_purity"]
        sig = mr["steiger_significant_R_better"]

        if sig:
            n_steiger_R_better += 1
        else:
            n_steiger_R_not_better += 1

        print(f"  {model_name}: rho_R={r_rho:.4f}, rho_E={e_rho:.4f}, "
              f"delta={r_rho - e_rho:.4f}, Z={z}, p={p}, sig_R>E={sig}")
        steiger_details[model_name] = {
            "rho_R": float(r_rho),
            "rho_E": float(e_rho),
            "delta_rho": float(r_rho - e_rho),
            "steiger_z": z,
            "steiger_p": p,
            "significant_R_better": sig,
        }

    steiger_confirms = n_steiger_R_better >= 2
    steiger_fails = n_steiger_R_not_better >= 2

    print(f"\n  Models with significant R > E: {n_steiger_R_better}/3")
    print(f"  Models without significant R > E: {n_steiger_R_not_better}/3")
    print(f"  Steiger confirms R > E on >= 2: {steiger_confirms}")
    print(f"  Steiger fails R > E on >= 2: {steiger_fails}")

    # --- Overall Verdict ---
    if amplifies_criterion and steiger_fails:
        verdict = "FALSIFIED"
        reason = ("grad_S dampening < 1.0 (amplifies bias) on >= 2/3 architectures "
                  "AND Steiger does not show R outperforms E on >= 2/3")
    elif dampens_criterion and steiger_confirms:
        verdict = "CONFIRMED"
        reason = ("grad_S dampening > 1.0 on >= 2/3 architectures "
                  "AND Steiger shows R > E (p<0.05) on >= 2/3")
    else:
        verdict = "INCONCLUSIVE"
        # Build specific reason
        parts = []
        if dampens_criterion:
            parts.append(f"grad_S dampens on {n_dampens}/3 (meets threshold)")
        else:
            parts.append(f"grad_S dampens on {n_dampens}/3 (does not meet 2/3 threshold)")
        if steiger_confirms:
            parts.append(f"Steiger R>E on {n_steiger_R_better}/3 (meets threshold)")
        else:
            parts.append(f"Steiger R>E on {n_steiger_R_better}/3 (does not meet 2/3 threshold)")
        reason = "Mixed results: " + "; ".join(parts)

    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print(f"{'=' * 70}")

    all_results["verdict"] = {
        "result": verdict,
        "reason": reason,
        "criterion_A_dampening": {
            "description": "grad_S_biased/grad_S_clean: >1 dampens, <1 amplifies. "
                           "FALSIFIED requires < 1.0 on >= 2/3; CONFIRMED requires > 1.0 on >= 2/3",
            "n_dampens": n_dampens,
            "n_amplifies": n_amplifies,
            "dampens_criterion_met": bool(dampens_criterion),
            "amplifies_criterion_met": bool(amplifies_criterion),
            "details": dampening_details,
        },
        "criterion_B_steiger": {
            "description": "Steiger (1980) ZPF test: R > E correlation with purity, p < 0.05 on >= 2/3",
            "formula_note": "var = 2*(1-r12)/((n-3)*(1+r12))",
            "n_significant": n_steiger_R_better,
            "confirms_criterion_met": bool(steiger_confirms),
            "details": steiger_details,
        },
    }

    all_results["total_duration_s"] = time.time() - total_start

    # Save results
    results_file = RESULTS_DIR / "test_v4_q05_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
