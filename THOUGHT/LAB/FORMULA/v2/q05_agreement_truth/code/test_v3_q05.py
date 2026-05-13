"""
Q05 v3 Test: Does High Local Agreement (High R) Reveal Truth?

Addresses ALL audit findings from AUDIT.md:
  - METHOD-1: Compute E inflation alongside R inflation under same bias attack
  - BUG-1: Use bool() for numpy bools before JSON serialization
  - STAT-4: Continuous purity (0.1 to 1.0 in 0.1 steps) instead of 4 discrete levels
  - METHOD-2: R-specific vulnerability analysis (E, grad_S, R decomposition)
  - Steiger's test for R vs E correlation with purity
  - 3 architectures: all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1

Pre-registered criteria:
  FALSIFIED:    R_inflation / E_inflation > 1.5 on >= 2/3 architectures
  CONFIRMED:    ratio < 1.2 AND Steiger shows R > E for purity on >= 2/3 archs
  INCONCLUSIVE: otherwise

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
    Steiger's Z-test for comparing two dependent correlations.
    Tests H0: rho(X,Y1) = rho(X,Y2) where Y1 and Y2 share the same X.

    r1  = cor(X, Y1)  -- e.g., cor(purity, R)
    r2  = cor(X, Y2)  -- e.g., cor(purity, E)
    r12 = cor(Y1, Y2) -- e.g., cor(R, E)
    n   = sample size

    Returns (z_stat, p_value_two_sided).
    Positive z means r1 > r2.
    """
    if n < 4:
        return float('nan'), float('nan')
    # Fisher z-transform
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))

    # Determinant of the 2x2 correlation matrix of Y1, Y2
    det = 1.0 - r1**2 - r2**2 - r12**2 + 2*r1*r2*r12
    r_bar = (r1 + r2) / 2.0

    # Steiger's formula for the variance of z1 - z2
    denom = (1.0 - r12)
    if abs(denom) < 1e-10:
        return float('nan'), float('nan')

    # Simplified Steiger formula (Dunn & Clark modification)
    var_diff = (2.0 / (n - 3)) * (1.0 - r12) / (1.0 - r_bar**2)**2
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

    For purity p, a cluster of size N has floor(N*p) docs from the dominant
    category and the rest drawn uniformly from other categories.
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
    Use Steiger's test to compare cor(purity, R) vs cor(purity, E).
    """
    print("\n" + "=" * 70)
    print("TEST 1: Purity Correlation (continuous, 10 levels, Steiger's test)")
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

        # Steiger's test: is rho_R significantly > rho_E?
        z_steiger, p_steiger = steiger_z_test(
            float(rho_R), float(rho_E), float(rho_RE), n_valid
        )

        R_outperforms = bool(abs(rho_R) > abs(rho_E))

        print(f"  n_valid = {n_valid}")
        print(f"  Spearman(R, purity) = {rho_R:.4f}, p = {p_R:.2e}")
        print(f"  Spearman(E, purity) = {rho_E:.4f}, p = {p_E:.2e}")
        print(f"  Spearman(R, E)      = {rho_RE:.4f}")
        print(f"  Steiger Z = {z_steiger:.4f}, p = {p_steiger:.4f}")
        print(f"  R outperforms E: {R_outperforms}")

        results_by_model[model_name] = {
            "rho_R_purity": float(rho_R),
            "p_R_purity": float(p_R),
            "rho_E_purity": float(rho_E),
            "p_E_purity": float(p_E),
            "rho_R_E": float(rho_RE),
            "steiger_z": float(z_steiger) if not math.isnan(z_steiger) else None,
            "steiger_p": float(p_steiger) if not math.isnan(p_steiger) else None,
            "R_outperforms_E": R_outperforms,
            "n_valid": n_valid,
            "R_values": [float(x) for x in R_arr],
            "E_values": [float(x) for x in E_arr],
        }

    return {
        "test": "Purity Correlation with Steiger's Test",
        "n_clusters": len(clusters),
        "n_purity_levels": 10,
        "purities": [float(p) for p in purities],
        "results_by_model": results_by_model,
    }


# =============================================================================
# TEST 2: Bias Attack -- R vs E Inflation Comparison (CRITICAL FIX)
# =============================================================================

def test2_bias_attack(texts, labels, all_embeddings_by_model):
    """
    The key test: run the SAME bias attack and measure E inflation AND R inflation.

    For each architecture:
    - Take 20 random clusters of 50 docs
    - For each bias phrase, prepend to all docs, re-encode, compute R and E
    - Report: R_inflation = R_biased / R_clean, E_inflation = E_biased / E_clean
    - Report: inflation_ratio = R_inflation / E_inflation
    - If inflation_ratio > 1.5 => R amplifies the vulnerability
    - If inflation_ratio < 1.2 => R merely inherits it from E
    """
    print("\n" + "=" * 70)
    print("TEST 2: Bias Attack -- R vs E Inflation (all 3 architectures)")
    print("=" * 70)

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

        # Compute clean R, E, grad_S for each cluster
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

            R_inflations = []
            E_inflations = []
            gradS_changes = []
            per_cluster = []

            for ci, idx in enumerate(cluster_indices):
                biased_embs = np.array([biased_emb_map[int(k)] for k in idx])

                r_b = safe_R(biased_embs)
                e_b = safe_E(biased_embs)
                g_b = safe_grad_S(biased_embs)

                r_c = clean_R[ci]
                e_c = clean_E[ci]
                g_c = clean_gradS[ci]

                # Inflation ratios (biased / clean)
                r_infl = r_b / r_c if (not math.isnan(r_c) and abs(r_c) > 1e-10 and not math.isnan(r_b)) else float('nan')
                e_infl = e_b / e_c if (not math.isnan(e_c) and abs(e_c) > 1e-10 and not math.isnan(e_b)) else float('nan')
                g_change = g_b / g_c if (not math.isnan(g_c) and abs(g_c) > 1e-10 and not math.isnan(g_b)) else float('nan')

                if not math.isnan(r_infl):
                    R_inflations.append(r_infl)
                if not math.isnan(e_infl):
                    E_inflations.append(e_infl)
                if not math.isnan(g_change):
                    gradS_changes.append(g_change)

                per_cluster.append({
                    "R_clean": float(r_c) if not math.isnan(r_c) else None,
                    "R_biased": float(r_b) if not math.isnan(r_b) else None,
                    "E_clean": float(e_c) if not math.isnan(e_c) else None,
                    "E_biased": float(e_b) if not math.isnan(e_b) else None,
                    "gradS_clean": float(g_c) if not math.isnan(g_c) else None,
                    "gradS_biased": float(g_b) if not math.isnan(g_b) else None,
                    "R_inflation": float(r_infl) if not math.isnan(r_infl) else None,
                    "E_inflation": float(e_infl) if not math.isnan(e_infl) else None,
                    "gradS_change": float(g_change) if not math.isnan(g_change) else None,
                })

            mean_R_infl = float(np.mean(R_inflations)) if R_inflations else float('nan')
            mean_E_infl = float(np.mean(E_inflations)) if E_inflations else float('nan')
            mean_gradS_change = float(np.mean(gradS_changes)) if gradS_changes else float('nan')

            # THE KEY METRIC: does R amplify beyond E?
            if not math.isnan(mean_R_infl) and not math.isnan(mean_E_infl) and abs(mean_E_infl) > 1e-10:
                amplification_ratio = mean_R_infl / mean_E_infl
            else:
                amplification_ratio = float('nan')

            print(f"    Mean E inflation:  {mean_E_infl:.3f}x")
            print(f"    Mean R inflation:  {mean_R_infl:.3f}x")
            print(f"    Mean gradS change: {mean_gradS_change:.3f}x")
            print(f"    Amplification (R_infl / E_infl): {amplification_ratio:.3f}x")

            bias_results.append({
                "bias_phrase": bias_phrase,
                "mean_R_inflation": mean_R_infl if not math.isnan(mean_R_infl) else None,
                "mean_E_inflation": mean_E_infl if not math.isnan(mean_E_infl) else None,
                "mean_gradS_change": mean_gradS_change if not math.isnan(mean_gradS_change) else None,
                "amplification_ratio": float(amplification_ratio) if not math.isnan(amplification_ratio) else None,
                "n_valid": len(R_inflations),
                "per_cluster": per_cluster,
            })

        del model
        gc.collect()

        # Summary across bias phrases for this model
        amp_ratios = [br["amplification_ratio"] for br in bias_results if br["amplification_ratio"] is not None]
        mean_amp = float(np.mean(amp_ratios)) if amp_ratios else None
        max_amp = float(np.max(amp_ratios)) if amp_ratios else None

        results_by_model[model_name] = {
            "bias_results": bias_results,
            "mean_amplification_across_phrases": mean_amp,
            "max_amplification_across_phrases": max_amp,
            "clean_R_values": [float(x) if not math.isnan(x) else None for x in clean_R],
            "clean_E_values": [float(x) if not math.isnan(x) else None for x in clean_E],
        }

        print(f"\n  Model summary: mean_amplification={mean_amp}, max_amplification={max_amp}")

    return {
        "test": "Bias Attack R vs E Inflation",
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
    print("Q05 v3 TEST: DOES HIGH AGREEMENT (HIGH R) REVEAL TRUTH?")
    print("=" * 70)
    print(f"Seed: 42")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Pre-registered criteria:")
    print("  FALSIFIED:    R_inflation/E_inflation > 1.5 on >= 2/3 architectures")
    print("  CONFIRMED:    ratio < 1.2 AND Steiger R > E on >= 2/3 architectures")
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
            "test": "Q05 v3",
            "question": "Does high local agreement (high R) reveal truth?",
            "seed": 42,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "n_documents": len(texts),
            "n_categories": len(target_names),
            "architectures": MODEL_SPECS,
            "target_names": list(target_names),
            "pre_registered_criteria": {
                "FALSIFIED": "R_inflation/E_inflation > 1.5 on >= 2/3 architectures",
                "CONFIRMED": "ratio < 1.2 AND Steiger R > E on >= 2/3 architectures",
                "INCONCLUSIVE": "otherwise",
            },
        }
    }

    # Test 1: Purity correlation with Steiger's test
    t1_start = time.time()
    r1 = test1_purity_correlation(clusters, all_embeddings)
    r1["duration_s"] = time.time() - t1_start
    all_results["test1_purity_correlation"] = r1

    # Test 2: Bias attack R vs E inflation
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

    # --- Criterion A: R_inflation / E_inflation (across all 3 architectures) ---
    print("\n--- Criterion A: R amplification ratio ---")
    n_amplifies = 0  # models where ratio > 1.5
    n_inherits = 0   # models where ratio < 1.2
    amp_details = {}

    for model_name in MODEL_SPECS:
        mr = r2["results_by_model"][model_name]
        max_amp = mr["max_amplification_across_phrases"]
        mean_amp = mr["mean_amplification_across_phrases"]
        print(f"  {model_name}:")
        print(f"    mean amplification = {mean_amp}")
        print(f"    max amplification  = {max_amp}")

        # Use max across bias phrases (worst case)
        if max_amp is not None and max_amp > 1.5:
            n_amplifies += 1
            print(f"    --> AMPLIFIES (ratio > 1.5)")
        elif max_amp is not None and max_amp < 1.2:
            n_inherits += 1
            print(f"    --> INHERITS (ratio < 1.2)")
        else:
            print(f"    --> INTERMEDIATE")

        amp_details[model_name] = {
            "mean_amplification": mean_amp,
            "max_amplification": max_amp,
        }

    falsified_A = n_amplifies >= 2
    inherited_A = n_inherits >= 2

    print(f"\n  Models with amplification > 1.5: {n_amplifies}/3")
    print(f"  Models with amplification < 1.2: {n_inherits}/3")
    print(f"  FALSIFIED (>= 2 amplify): {falsified_A}")
    print(f"  Inherited (<= 1.2 on >= 2): {inherited_A}")

    # --- Criterion B: Steiger's test R > E for purity ---
    print("\n--- Criterion B: Steiger's test R > E ---")
    n_steiger_R_better = 0
    steiger_details = {}

    for model_name in MODEL_SPECS:
        mr = r1["results_by_model"][model_name]
        z = mr["steiger_z"]
        p = mr["steiger_p"]
        r_rho = mr["rho_R_purity"]
        e_rho = mr["rho_E_purity"]
        outperforms = mr["R_outperforms_E"]

        sig = False
        if z is not None and p is not None:
            sig = bool(z > 0 and p < 0.05)
        if sig:
            n_steiger_R_better += 1

        print(f"  {model_name}: rho_R={r_rho:.4f}, rho_E={e_rho:.4f}, Z={z}, p={p}, sig_R>E={sig}")
        steiger_details[model_name] = {
            "rho_R": r_rho,
            "rho_E": e_rho,
            "steiger_z": z,
            "steiger_p": p,
            "significant_R_better": sig,
        }

    confirmed_B = n_steiger_R_better >= 2

    print(f"\n  Models with significant R > E: {n_steiger_R_better}/3")
    print(f"  Steiger confirms R > E on >= 2: {confirmed_B}")

    # --- Overall Verdict ---
    if falsified_A:
        verdict = "FALSIFIED"
        reason = "R amplifies bias vulnerability beyond E in >= 2/3 architectures"
    elif inherited_A and confirmed_B:
        verdict = "CONFIRMED"
        reason = "R inherits (not amplifies) E vulnerability AND Steiger shows R > E for purity"
    elif inherited_A and not confirmed_B:
        verdict = "INCONCLUSIVE"
        reason = "R inherits E vulnerability but Steiger does not confirm R > E for purity"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed results across architectures"

    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print(f"{'=' * 70}")

    all_results["verdict"] = {
        "result": verdict,
        "reason": reason,
        "criterion_A_amplification": {
            "description": "R_inflation/E_inflation > 1.5 on >= 2/3 architectures => FALSIFIED",
            "n_amplifies": n_amplifies,
            "n_inherits": n_inherits,
            "falsified": bool(falsified_A),
            "details": amp_details,
        },
        "criterion_B_steiger": {
            "description": "Steiger's test: R > E correlation with purity, significant on >= 2/3",
            "n_significant": n_steiger_R_better,
            "confirmed": bool(confirmed_B),
            "details": steiger_details,
        },
    }

    all_results["total_duration_s"] = time.time() - total_start

    # Save results
    results_file = RESULTS_DIR / "test_v3_q05_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
