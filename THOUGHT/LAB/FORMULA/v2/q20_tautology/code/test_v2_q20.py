"""
Q20 v2: Is R Tautological?
===========================
Tests whether R = (E / grad_S) * sigma^Df is explanatory or merely descriptive.

4 tests:
  1. Component comparison: Does R_full outperform its parts?
  2. 8e conservation on in-distribution data
  3. Ablation study on functional form
  4. Novel prediction test (pre-registered)

Uses STS-B dataset with sentence-transformer embeddings.
Seed: 42

IMPORTANT NAMING NOTE:
  The v2 shared formula defines:
    - sigma = PR / d  (participation ratio normalized by ambient dim)
    - Df = 2 / alpha  (fractal dimension from eigenvalue decay)
  The v1 8e conservation claim uses:
    - Df = PR  (raw participation ratio, NOT normalized)
    - alpha = eigenvalue decay exponent (fit on top half of spectrum)
  These are DIFFERENT quantities with the SAME name.
  This test uses the v1 definitions for the 8e test, and v2 definitions
  for the formula comparison tests, and reports both clearly.
"""

import sys
import os
import json
import time
import warnings
import traceback

import numpy as np
from scipy import stats

np.random.seed(42)
warnings.filterwarnings("ignore")

# ---------- path setup ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", ".."))
V2_DIR = os.path.join(REPO_ROOT, "THOUGHT", "LAB", "FORMULA", "v2")
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared.formula import (
    compute_E, compute_grad_S, compute_sigma, compute_Df,
    compute_R_simple, compute_R_full, compute_all,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "THOUGHT", "LAB", "FORMULA", "v2", "q20_tautology", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Utility functions
# =============================================================================

def compute_SNR(embeddings):
    """Signal-to-noise ratio: mean(norms) / std(norms)."""
    norms = np.linalg.norm(embeddings, axis=1)
    s = np.std(norms)
    if s < 1e-10:
        return float("nan")
    return float(np.mean(norms) / s)


def compute_E_times_sigmaDf(embeddings):
    """E * sigma^Df -- without dispersion normalization."""
    E = compute_E(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)
    if any(np.isnan(x) for x in [E, sigma, Df]):
        return float("nan")
    return float(E * (sigma ** Df))


def compute_inv_grad_S(embeddings):
    """1 / grad_S."""
    g = compute_grad_S(embeddings)
    if np.isnan(g) or g < 1e-10:
        return float("nan")
    return 1.0 / g


def compute_sigma_Df(embeddings):
    """sigma^Df alone."""
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)
    if any(np.isnan(x) for x in [sigma, Df]):
        return float("nan")
    return float(sigma ** Df)


# ---------- Ablation alternative forms ----------

def compute_R_sub(embeddings):
    """R_sub = (E - grad_S) * sigma^Df."""
    E = compute_E(embeddings)
    g = compute_grad_S(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)
    if any(np.isnan(x) for x in [E, g, sigma, Df]):
        return float("nan")
    return float((E - g) * (sigma ** Df))


def compute_R_exp(embeddings):
    """R_exp = E * exp(-grad_S) * sigma^Df."""
    E = compute_E(embeddings)
    g = compute_grad_S(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)
    if any(np.isnan(x) for x in [E, g, sigma, Df]):
        return float("nan")
    return float(E * np.exp(-g) * (sigma ** Df))


def compute_R_log(embeddings):
    """R_log = log(max(E, 0.01)) / (grad_S + 1)."""
    E = compute_E(embeddings)
    g = compute_grad_S(embeddings)
    if any(np.isnan(x) for x in [E, g]):
        return float("nan")
    return float(np.log(max(E, 0.01)) / (g + 1.0))


def compute_R_add(embeddings):
    """R_add = E / (grad_S + sigma^Df)."""
    E = compute_E(embeddings)
    g = compute_grad_S(embeddings)
    sigma = compute_sigma(embeddings)
    Df = compute_Df(embeddings)
    if any(np.isnan(x) for x in [E, g, sigma, Df]):
        return float("nan")
    denom = g + sigma ** Df
    if denom < 1e-10:
        return float("nan")
    return float(E / denom)


# =============================================================================
# Data loading
# =============================================================================

def load_stsb():
    """Load STS-B dataset from HuggingFace."""
    print("[DATA] Loading STS-B from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences1 = ds["sentence1"]
    sentences2 = ds["sentence2"]
    scores = np.array(ds["score"], dtype=float)
    print(f"  Loaded {len(scores)} sentence pairs, score range [{scores.min():.2f}, {scores.max():.2f}]")
    return sentences1, sentences2, scores


def encode_sentences(model_name, sentences1, sentences2):
    """Encode sentence pairs with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    print(f"  Encoding with {model_name}...")
    t0 = time.time()
    st = SentenceTransformer(model_name)
    emb1 = st.encode(sentences1, show_progress_bar=False, batch_size=256)
    emb2 = st.encode(sentences2, show_progress_bar=False, batch_size=256)
    elapsed = time.time() - t0
    print(f"  Encoded {len(sentences1)} pairs in {elapsed:.1f}s, dim={emb1.shape[1]}")
    return np.array(emb1), np.array(emb2)


# =============================================================================
# Test 1: Component Comparison (THE KEY TEST)
# =============================================================================

def test1_component_comparison(emb1, emb2, scores, n_clusters=100):
    """
    Cluster sentence pairs by similarity score (quantile-based),
    compute formula metrics per cluster, correlate with mean human score.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Component Comparison -- Does R Outperform Its Parts?")
    print("=" * 70)

    # Use quantile-based binning for even distribution
    percentiles = np.linspace(0, 100, n_clusters + 1)
    score_bins = np.percentile(scores, percentiles)
    # Remove duplicate bin edges
    score_bins = np.unique(score_bins)
    actual_bins = len(score_bins) - 1
    print(f"  Using {actual_bins} quantile-based bins (from {n_clusters} requested)")

    cluster_metrics = []
    cluster_human_scores = []
    cluster_R_values = []  # For prediction 2

    for i in range(actual_bins):
        lo, hi = score_bins[i], score_bins[i + 1]
        if i < actual_bins - 1:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)

        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue

        # Combine both sides of pairs into one embedding matrix per cluster
        cluster_emb = np.vstack([emb1[idx], emb2[idx]])
        mean_score = float(np.mean(scores[idx]))

        # Compute all metrics
        try:
            metrics = {}
            metrics["R_full"] = compute_R_full(cluster_emb)
            metrics["E"] = compute_E(cluster_emb)
            metrics["inv_grad_S"] = compute_inv_grad_S(cluster_emb)
            metrics["sigma_Df"] = compute_sigma_Df(cluster_emb)
            metrics["E_over_grad_S"] = compute_R_simple(cluster_emb)
            metrics["E_times_sigmaDf"] = compute_E_times_sigmaDf(cluster_emb)
            metrics["SNR"] = compute_SNR(cluster_emb)

            # Ablation alternatives
            metrics["R_sub"] = compute_R_sub(cluster_emb)
            metrics["R_exp"] = compute_R_exp(cluster_emb)
            metrics["R_log"] = compute_R_log(cluster_emb)
            metrics["R_add"] = compute_R_add(cluster_emb)

            # Skip clusters with NaN in any core metric
            core = ["R_full", "E", "inv_grad_S", "sigma_Df", "E_over_grad_S",
                     "E_times_sigmaDf", "SNR"]
            if any(np.isnan(metrics[k]) for k in core):
                continue

            cluster_metrics.append(metrics)
            cluster_human_scores.append(mean_score)
            cluster_R_values.append(metrics["R_full"])
        except Exception as e:
            print(f"  Cluster {i}: skipped ({e})")
            continue

    n_valid = len(cluster_metrics)
    print(f"\n  Valid clusters: {n_valid} / {actual_bins}")
    print(f"  Human score range in clusters: [{min(cluster_human_scores):.2f}, {max(cluster_human_scores):.2f}]")

    if n_valid < 10:
        print("  ERROR: Too few valid clusters for meaningful correlation.")
        return None

    human = np.array(cluster_human_scores)

    # Compute Spearman correlations
    metric_names = ["R_full", "E", "inv_grad_S", "sigma_Df", "E_over_grad_S",
                    "E_times_sigmaDf", "SNR", "R_sub", "R_exp", "R_log", "R_add"]

    results = {}
    print(f"\n  {'Metric':<20} {'Spearman rho':>14} {'p-value':>12}")
    print("  " + "-" * 48)

    for name in metric_names:
        values = np.array([m[name] for m in cluster_metrics])
        valid = ~np.isnan(values)
        if valid.sum() < 10:
            print(f"  {name:<20} {'N/A':>14} {'N/A':>12}  (too few valid)")
            results[name] = {"rho": float("nan"), "p": float("nan"), "n": int(valid.sum())}
            continue
        rho, p = stats.spearmanr(values[valid], human[valid])
        print(f"  {name:<20} {rho:>14.4f} {p:>12.2e}  (n={valid.sum()})")
        results[name] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}

    # Evaluate: R_full must outperform ALL core components by >= 0.05
    r_full_rho = results["R_full"]["rho"]
    core_components = ["E", "inv_grad_S", "sigma_Df", "E_over_grad_S",
                       "E_times_sigmaDf", "SNR"]
    beats_all = True
    margins = {}
    for name in core_components:
        comp_rho = results[name]["rho"]
        margin = abs(r_full_rho) - abs(comp_rho)
        margins[name] = float(margin)
        if margin < 0.05:
            beats_all = False

    print(f"\n  R_full rho = {r_full_rho:.4f}")
    print(f"  Margins over components (need >= 0.05 each):")
    for name, margin in margins.items():
        status = "PASS" if margin >= 0.05 else "FAIL"
        print(f"    vs {name:<20}: margin = {margin:+.4f}  [{status}]")

    print(f"\n  R_full outperforms ALL components by >= 0.05: {'YES' if beats_all else 'NO'}")

    # Also compute: which metric actually correlates best?
    best_metric = max(results.keys(), key=lambda k: abs(results[k]["rho"]) if not np.isnan(results[k]["rho"]) else -1)
    print(f"\n  BEST correlating metric: {best_metric} (rho = {results[best_metric]['rho']:.4f})")

    return {
        "n_clusters": n_valid,
        "correlations": results,
        "r_full_rho": float(r_full_rho),
        "margins": margins,
        "beats_all": beats_all,
        "best_metric": best_metric,
        "cluster_R_values": [float(x) for x in cluster_R_values],
        "cluster_human_scores": [float(x) for x in cluster_human_scores],
    }


# =============================================================================
# Test 2: 8e Conservation
# =============================================================================

def compute_8e_product(embeddings, label=""):
    """
    Compute PR * alpha using v1 definitions (the 8e claim).

    v1 definitions:
      - PR = (sum(eig))^2 / sum(eig^2)  -- raw participation ratio (NOT normalized)
      - alpha = power law exponent, fit on TOP HALF of eigenvalues only

    Also reports v2 formula's Df (= 2/alpha) and sigma (= PR/d) for comparison.
    """
    n, d = embeddings.shape
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 10:
        return {"alpha": float("nan"), "PR": float("nan"), "product": float("nan"),
                "error_vs_8e": float("nan"), "n_eigenvalues": 0}

    # v1 alpha: fit on top half of eigenvalues
    n_fit = len(eigenvalues) // 2
    if n_fit < 5:
        n_fit = min(len(eigenvalues), 10)
    k_fit = np.arange(1, n_fit + 1)
    log_k = np.log(k_fit)
    log_eig = np.log(eigenvalues[:n_fit])
    slope, _ = np.polyfit(log_k, log_eig, 1)
    alpha = -slope

    # v1 Df: raw participation ratio
    PR = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    # v2 formula definitions for comparison
    sigma_v2 = PR / d  # what v2 calls sigma
    Df_v2 = 2.0 / alpha if alpha > 0 else float("nan")  # what v2 calls Df

    # Also compute alpha using ALL eigenvalues (full spectrum) for comparison
    k_all = np.arange(1, len(eigenvalues) + 1)
    log_k_all = np.log(k_all)
    log_eig_all = np.log(eigenvalues)
    slope_all, _ = np.polyfit(log_k_all, log_eig_all, 1)
    alpha_full = -slope_all

    product = PR * alpha
    product_full = PR * alpha_full
    eight_e = 8 * np.e  # 21.746
    error = abs(product - eight_e) / eight_e * 100
    error_full = abs(product_full - eight_e) / eight_e * 100

    return {
        "alpha_top_half": float(alpha),
        "alpha_full_spectrum": float(alpha_full),
        "PR": float(PR),
        "PR_x_alpha_top_half": float(product),
        "PR_x_alpha_full": float(product_full),
        "error_top_half_vs_8e": float(error),
        "error_full_vs_8e": float(error_full),
        "sigma_v2": float(sigma_v2),
        "Df_v2": float(Df_v2),
        "n_eigenvalues": int(len(eigenvalues)),
        "n_fit": int(n_fit),
        "n_samples": int(n),
        "dim": int(d),
    }


def test2_8e_conservation(sentences1, sentences2, scores):
    """Test 8e conservation on real text data with multiple models."""
    print("\n" + "=" * 70)
    print("TEST 2: 8e Conservation on In-Distribution Data")
    print("=" * 70)
    print("  NOTE: Using v1 definitions: PR = raw participation ratio,")
    print("  alpha fit on top half of eigenvalues. Also reports full-spectrum alpha.")

    from sentence_transformers import SentenceTransformer

    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ]

    # Combine all unique sentences for embedding
    all_sentences = list(set(list(sentences1) + list(sentences2)))
    np.random.seed(42)
    if len(all_sentences) > 2000:
        idx = np.random.choice(len(all_sentences), 2000, replace=False)
        all_sentences = [all_sentences[i] for i in idx]

    print(f"  Using {len(all_sentences)} unique sentences")

    results = {}
    eight_e = 8 * np.e

    for model_name in models:
        print(f"\n  Model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            embs = model.encode(all_sentences, show_progress_bar=False, batch_size=256)
            embs = np.array(embs)
            print(f"    Shape: {embs.shape}")

            r = compute_8e_product(embs, label=model_name)
            print(f"    alpha (top half) = {r['alpha_top_half']:.4f}")
            print(f"    alpha (full)     = {r['alpha_full_spectrum']:.4f}")
            print(f"    PR               = {r['PR']:.4f}")
            print(f"    PR * alpha (top) = {r['PR_x_alpha_top_half']:.4f}  (8e = {eight_e:.4f})")
            print(f"    PR * alpha (full)= {r['PR_x_alpha_full']:.4f}  (8e = {eight_e:.4f})")
            print(f"    Error (top half) vs 8e: {r['error_top_half_vs_8e']:.2f}%")
            print(f"    Error (full)     vs 8e: {r['error_full_vs_8e']:.2f}%")
            print(f"    (v2 sigma = {r['sigma_v2']:.4f}, v2 Df = {r['Df_v2']:.4f})")
            results[model_name] = r
        except Exception as e:
            print(f"    SKIPPED: {e}")
            traceback.print_exc()
            results[model_name] = {"error": str(e), "skipped": True}

    # Random matrix negative control
    print(f"\n  Negative Control: Random Matrices (Gaussian iid)")
    random_results = []
    for trial in range(5):
        shape = (2000, 384)
        rng = np.random.RandomState(42 + trial)
        rand_emb = rng.randn(*shape)
        r = compute_8e_product(rand_emb, label=f"random_{trial}")
        random_results.append(r)
        print(f"    Trial {trial}: PR*alpha(top) = {r['PR_x_alpha_top_half']:.4f}, "
              f"error = {r['error_top_half_vs_8e']:.2f}%  |  "
              f"PR*alpha(full) = {r['PR_x_alpha_full']:.4f}, "
              f"error = {r['error_full_vs_8e']:.2f}%")

    mean_random_error_top = np.mean([r["error_top_half_vs_8e"] for r in random_results])
    mean_random_error_full = np.mean([r["error_full_vs_8e"] for r in random_results])
    print(f"    Mean random error (top half alpha): {mean_random_error_top:.2f}%")
    print(f"    Mean random error (full alpha): {mean_random_error_full:.2f}%")

    results["random_control"] = {
        "trials": random_results,
        "mean_error_top_half": float(mean_random_error_top),
        "mean_error_full": float(mean_random_error_full),
    }

    return results


# =============================================================================
# Test 3: Ablation Study
# =============================================================================

def test3_ablation(test1_results):
    """Compare R_full against alternative functional forms."""
    print("\n" + "=" * 70)
    print("TEST 3: Ablation Study on Functional Form")
    print("=" * 70)

    if test1_results is None:
        print("  SKIPPED: Test 1 did not produce results.")
        return None

    corrs = test1_results["correlations"]

    alternatives = {
        "R_sub": "(E - grad_S) * sigma^Df",
        "R_exp": "E * exp(-grad_S) * sigma^Df",
        "R_log": "log(max(E, 0.01)) / (grad_S + 1)",
        "R_add": "E / (grad_S + sigma^Df)",
    }

    print(f"\n  {'Form':<12} {'Formula':<35} {'|rho|':>8} {'vs R_full':>10}")
    print("  " + "-" * 67)

    r_full_abs = abs(corrs["R_full"]["rho"])
    print(f"  {'R_full':<12} {'(E/grad_S) * sigma^Df':<35} {r_full_abs:>8.4f} {'baseline':>10}")

    n_beaten = 0
    comparison_results = {"R_full": {"rho": corrs["R_full"]["rho"], "formula": "(E/grad_S) * sigma^Df"}}

    for name, formula in alternatives.items():
        if name in corrs and not np.isnan(corrs[name]["rho"]):
            alt_abs = abs(corrs[name]["rho"])
            diff = r_full_abs - alt_abs
            status = "WINS" if diff > 0 else "LOSES"
            print(f"  {name:<12} {formula:<35} {alt_abs:>8.4f} {diff:>+10.4f} [{status}]")
            comparison_results[name] = {"rho": float(corrs[name]["rho"]), "formula": formula, "diff": float(diff)}
            if diff > 0:
                n_beaten += 1
        else:
            print(f"  {name:<12} {formula:<35} {'N/A':>8}")
            comparison_results[name] = {"rho": float("nan"), "formula": formula}

    total_alts = len(alternatives)
    print(f"\n  R_full beats {n_beaten}/{total_alts} alternatives")
    print(f"  Criterion: must beat >= 3/4 alternatives")
    passes = n_beaten >= 3

    return {
        "comparisons": comparison_results,
        "n_beaten": n_beaten,
        "total": total_alts,
        "passes": passes,
    }


# =============================================================================
# Test 4: Novel Prediction Test
# =============================================================================

def test4_novel_predictions(test1_results):
    """Test pre-registered predictions using data from test 1."""
    print("\n" + "=" * 70)
    print("TEST 4: Novel Prediction Test (Pre-Registered)")
    print("=" * 70)

    if test1_results is None:
        print("  SKIPPED: Test 1 did not produce results.")
        return None

    results = {}

    # Prediction 1: R_full correlates with human scores at |rho| > 0.4
    r_full_rho = test1_results["correlations"]["R_full"]["rho"]
    p_val = test1_results["correlations"]["R_full"]["p"]
    p1_pass = abs(r_full_rho) > 0.4
    print(f"\n  PREDICTION 1: R_full will correlate with human scores at |rho| > 0.4")
    print(f"    Observed: rho = {r_full_rho:.4f}, |rho| = {abs(r_full_rho):.4f}, p = {p_val:.2e}")
    print(f"    Result: {'PASS' if p1_pass else 'FAIL'}")
    results["prediction_1"] = {
        "description": "R_full correlates with human scores at |rho| > 0.4",
        "threshold": 0.4,
        "observed_rho": float(r_full_rho),
        "observed_abs_rho": float(abs(r_full_rho)),
        "p_value": float(p_val),
        "passed": p1_pass,
    }

    # Prediction 2: Top quartile R_full clusters have mean human score > 3.5
    print(f"\n  PREDICTION 2: Top quartile R_full clusters have mean human score > 3.5")
    cluster_R = np.array(test1_results["cluster_R_values"])
    cluster_human = np.array(test1_results["cluster_human_scores"])

    q75 = np.percentile(cluster_R, 75)
    top_mask = cluster_R >= q75
    top_mean_human = float(np.mean(cluster_human[top_mask]))
    bottom_mask = cluster_R <= np.percentile(cluster_R, 25)
    bottom_mean_human = float(np.mean(cluster_human[bottom_mask]))

    p2_pass = top_mean_human > 3.5
    print(f"    Top quartile R_full threshold: {q75:.4f}")
    print(f"    N in top quartile: {int(top_mask.sum())}")
    print(f"    Mean human score in top quartile: {top_mean_human:.4f}")
    print(f"    Mean human score in bottom quartile: {bottom_mean_human:.4f}")
    print(f"    Result: {'PASS' if p2_pass else 'FAIL'}")

    results["prediction_2"] = {
        "description": "Top quartile R_full clusters have mean human score > 3.5",
        "threshold": 3.5,
        "q75_R_threshold": float(q75),
        "top_quartile_mean_human": top_mean_human,
        "bottom_quartile_mean_human": bottom_mean_human,
        "n_top": int(top_mask.sum()),
        "n_total": len(cluster_R),
        "passed": p2_pass,
    }

    return results


# =============================================================================
# Main execution
# =============================================================================

def main():
    print("=" * 70)
    print("Q20 v2: Is R Tautological? -- Rigorous Scientific Test")
    print("=" * 70)
    print(f"Seed: 42")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"8e = {8 * np.e:.6f}")
    print()

    all_results = {"seed": 42, "date": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------- Load data ----------
    sentences1, sentences2, scores = load_stsb()
    all_results["dataset"] = {
        "name": "STS-B (mteb/stsbenchmark-sts)",
        "split": "test",
        "n_pairs": len(scores),
        "score_range": [float(scores.min()), float(scores.max())],
    }

    # ---------- Encode with primary model ----------
    primary_model = "all-MiniLM-L6-v2"
    emb1, emb2 = encode_sentences(primary_model, sentences1, sentences2)
    all_results["primary_model"] = primary_model
    all_results["embedding_dim"] = int(emb1.shape[1])

    # ---------- Test 1: Component Comparison ----------
    t1 = test1_component_comparison(emb1, emb2, scores, n_clusters=100)
    all_results["test1_component_comparison"] = t1

    # ---------- Test 2: 8e Conservation ----------
    t2 = test2_8e_conservation(sentences1, sentences2, scores)
    all_results["test2_8e_conservation"] = t2

    # ---------- Test 3: Ablation ----------
    t3 = test3_ablation(t1)
    all_results["test3_ablation"] = t3

    # ---------- Test 4: Novel Predictions ----------
    t4 = test4_novel_predictions(t1)
    all_results["test4_novel_predictions"] = t4

    # ==========================================================================
    # OVERALL VERDICT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("OVERALL VERDICT DETERMINATION")
    print("=" * 70)

    verdicts = []

    # Criterion 1: R_full outperforms ALL components by >= 0.05
    if t1 is not None:
        c1 = t1["beats_all"]
        verdicts.append(("Component superiority (>= 0.05 margin over ALL)", c1))
        print(f"  C1 - R_full beats all components by >= 0.05: {'PASS' if c1 else 'FAIL'}")
    else:
        verdicts.append(("Component superiority", False))
        print(f"  C1 - Component comparison: SKIPPED")

    # Criterion 2: Ablation -- R_full beats >= 3/4 alternatives
    if t3 is not None:
        c2 = t3["passes"]
        verdicts.append(("Ablation (beats >= 3/4)", c2))
        print(f"  C2 - R_full beats >= 3/4 ablation alternatives: {'PASS' if c2 else 'FAIL'}")
    else:
        verdicts.append(("Ablation", False))
        print(f"  C2 - Ablation: SKIPPED")

    # Criterion 3: Novel predictions succeed (both)
    if t4 is not None:
        p1 = t4["prediction_1"]["passed"]
        p2 = t4["prediction_2"]["passed"]
        c3 = p1 and p2
        verdicts.append(("Novel predictions (both pass)", c3))
        print(f"  C3 - Novel predictions: P1={'PASS' if p1 else 'FAIL'}, P2={'PASS' if p2 else 'FAIL'} -> {'PASS' if c3 else 'FAIL'}")
    else:
        verdicts.append(("Novel predictions", False))
        print(f"  C3 - Novel predictions: SKIPPED")

    # Decision logic
    all_pass = all(v[1] for v in verdicts)
    any_pass = any(v[1] for v in verdicts)

    # Check falsification conditions
    falsified = False
    falsification_reasons = []

    if t1 is not None:
        corrs = t1["correlations"]
        r_full_abs = abs(corrs["R_full"]["rho"])
        e_abs = abs(corrs["E"]["rho"])
        inv_gs_abs = abs(corrs["inv_grad_S"]["rho"])
        snr_abs = abs(corrs["SNR"]["rho"])
        e_gs_abs = abs(corrs["E_over_grad_S"]["rho"])

        if r_full_abs <= e_abs:
            falsified = True
            falsification_reasons.append(
                f"R_full (|rho|={r_full_abs:.4f}) does not outperform E alone (|rho|={e_abs:.4f})")
        if r_full_abs <= inv_gs_abs:
            falsified = True
            falsification_reasons.append(
                f"R_full (|rho|={r_full_abs:.4f}) does not outperform 1/grad_S (|rho|={inv_gs_abs:.4f})")
        if r_full_abs <= snr_abs:
            falsification_reasons.append(
                f"R_full (|rho|={r_full_abs:.4f}) does not outperform SNR (|rho|={snr_abs:.4f})")

        # Check if E/grad_S is essentially the same as R_full
        # (meaning sigma^Df adds nothing)
        if abs(r_full_abs - e_gs_abs) < 0.02:
            falsification_reasons.append(
                f"R_full and E/grad_S have nearly identical correlation "
                f"(diff={abs(r_full_abs - e_gs_abs):.4f}), "
                f"suggesting sigma^Df adds no value")

    if t3 is not None:
        # Check if all alternatives perform equally (within 0.02)
        alt_rhos = []
        for name in ["R_sub", "R_exp", "R_log", "R_add"]:
            if name in t3["comparisons"] and not np.isnan(t3["comparisons"][name].get("rho", float("nan"))):
                alt_rhos.append(abs(t3["comparisons"][name]["rho"]))
        if alt_rhos and t1 is not None:
            spread = max(alt_rhos) - min(alt_rhos)
            all_similar = all(abs(r_full_abs - ar) < 0.05 for ar in alt_rhos)
            if all_similar:
                falsification_reasons.append(
                    f"All alternative forms have similar correlation to R_full (spread={spread:.4f})")

    if all_pass:
        verdict = "CONFIRMED"
    elif falsified:
        verdict = "FALSIFIED"
    elif any_pass:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FALSIFIED"

    print(f"\n  VERDICT: {verdict}")
    if falsification_reasons:
        print(f"  Falsification reasons:")
        for r in falsification_reasons:
            print(f"    - {r}")

    all_results["verdict"] = verdict
    all_results["falsification_reasons"] = falsification_reasons
    all_results["criteria_results"] = [(v[0], v[1]) for v in verdicts]

    # Save results
    results_path = os.path.join(RESULTS_DIR, "test_v2_q20_results.json")

    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return "NaN"
        else:
            return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    results = main()
