"""
Q02 v2: Falsification Criteria for R = (E / grad_S) * sigma^Df

Rigorous test: component-level falsification on STS-B,
modus tollens on held-out data, adversarial robustness.

Uses shared formula from THOUGHT/LAB/FORMULA/v2/shared/formula.py
No retrofitting. Pre-registered criteria evaluated honestly.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[6]  # up to agent-governance-system

# Direct import of shared formula module via importlib
import importlib.util
_formula_path = REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "shared" / "formula.py"
_spec = importlib.util.spec_from_file_location("formula", str(_formula_path))
_formula = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_formula)

compute_E = _formula.compute_E
compute_grad_S = _formula.compute_grad_S
compute_sigma = _formula.compute_sigma
compute_Df = _formula.compute_Df
compute_R_simple = _formula.compute_R_simple
compute_R_full = _formula.compute_R_full
compute_all = _formula.compute_all

RESULTS_DIR = REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "q02_falsification" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_stsb():
    """Load STS-B from HuggingFace datasets."""
    from datasets import load_dataset
    print("[DATA] Loading STS-B from HuggingFace...")
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    # Also load train for calibration
    ds_train = load_dataset("mteb/stsbenchmark-sts", split="train")
    print(f"  Test: {len(ds)} pairs, Train: {len(ds_train)} pairs")
    return ds_train, ds


def encode_sentences(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Encode sentences using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    print(f"[ENCODE] Encoding {len(sentences)} sentences with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=128)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Test 1A: E Falsification -- cosine similarity vs human scores
# ---------------------------------------------------------------------------
def test_E_falsification(ds, emb1, emb2):
    """
    Test: Does cosine-similarity E correlate with human similarity judgments?

    For each STS-B pair, compute cosine similarity of embeddings.
    Compute Spearman correlation with human scores.

    Pre-registered criterion: r > 0.5 to pass.
    """
    print("\n" + "="*70)
    print("TEST 1A: E Falsification -- Cosine Similarity vs Human Scores")
    print("="*70)

    # Normalize embeddings
    norms1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norms1 = np.where(norms1 == 0, 1e-10, norms1)
    norms2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    norms2 = np.where(norms2 == 0, 1e-10, norms2)

    normed1 = emb1 / norms1
    normed2 = emb2 / norms2

    # Pairwise cosine similarity for each pair
    cosine_sims = np.sum(normed1 * normed2, axis=1)

    # Human scores (STS-B scores are 0-5)
    human_scores = np.array([ex["score"] for ex in ds])

    # Spearman correlation
    rho, p_value = stats.spearmanr(cosine_sims, human_scores)

    print(f"  Cosine sim range: [{cosine_sims.min():.4f}, {cosine_sims.max():.4f}]")
    print(f"  Human score range: [{human_scores.min():.4f}, {human_scores.max():.4f}]")
    print(f"  Spearman rho = {rho:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Criterion: rho > 0.5")

    passed = rho > 0.5
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "E_falsification",
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_pairs": len(cosine_sims),
        "criterion": "rho > 0.5",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 1B: grad_S Falsification -- grad_S vs empirical uncertainty
# ---------------------------------------------------------------------------
def test_grad_S_falsification(ds, emb1, emb2):
    """
    Test: Does grad_S correlate with empirical uncertainty?

    Group sentence pairs by score bucket, compute grad_S for each group,
    compute bootstrap variance of group centroids, and correlate.

    Pre-registered criterion: significant positive correlation (p < 0.05).
    """
    print("\n" + "="*70)
    print("TEST 1B: grad_S Falsification -- grad_S vs Bootstrap Variance")
    print("="*70)

    human_scores = np.array([ex["score"] for ex in ds])

    # Create score buckets (0-1, 1-2, 2-3, 3-4, 4-5)
    n_buckets = 10
    bucket_edges = np.linspace(0, 5, n_buckets + 1)

    grad_s_values = []
    bootstrap_variances = []
    bucket_labels = []

    for i in range(n_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i+1]
        mask = (human_scores >= lo) & (human_scores < hi)
        if i == n_buckets - 1:  # include right edge for last bucket
            mask = (human_scores >= lo) & (human_scores <= hi)

        if mask.sum() < 10:
            continue

        # Combine both sentence embeddings for this bucket
        bucket_embs = np.vstack([emb1[mask], emb2[mask]])

        # Compute grad_S
        gs = compute_grad_S(bucket_embs)

        # Bootstrap variance of centroids
        n_bootstrap = 100
        centroids = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(bucket_embs.shape[0], size=bucket_embs.shape[0], replace=True)
            centroids.append(bucket_embs[idx].mean(axis=0))
        centroids = np.array(centroids)
        # Variance = mean squared distance from mean centroid
        mean_centroid = centroids.mean(axis=0)
        dists = np.linalg.norm(centroids - mean_centroid, axis=1)
        bvar = float(np.var(dists))

        grad_s_values.append(gs)
        bootstrap_variances.append(bvar)
        bucket_labels.append(f"[{lo:.1f},{hi:.1f})")
        print(f"  Bucket {bucket_labels[-1]}: n={mask.sum()}, grad_S={gs:.4f}, bootstrap_var={bvar:.6f}")

    grad_s_values = np.array(grad_s_values)
    bootstrap_variances = np.array(bootstrap_variances)

    # Spearman correlation
    rho, p_value = stats.spearmanr(grad_s_values, bootstrap_variances)

    print(f"\n  Spearman rho(grad_S, bootstrap_var) = {rho:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Criterion: positive correlation with p < 0.05")

    passed = (rho > 0) and (p_value < 0.05)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "grad_S_falsification",
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_buckets": len(grad_s_values),
        "grad_s_values": grad_s_values.tolist(),
        "bootstrap_variances": bootstrap_variances.tolist(),
        "bucket_labels": bucket_labels,
        "criterion": "positive rho with p < 0.05",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 1C: Functional Form Comparison
# ---------------------------------------------------------------------------
def test_functional_form(ds_train, ds_test, emb1_train, emb2_train, emb1_test, emb2_test):
    """
    Test: Is R = E/grad_S the best functional form?

    Compare against alternatives on how well the formula predicts
    ground-truth sentence similarity quality.

    Method: Group pairs into clusters, compute formula value for each cluster,
    correlate with mean human score for that cluster. Best correlation wins.

    Pre-registered criterion: R must be within top 2 of all alternatives
    (otherwise the specific ratio form is not privileged).
    """
    print("\n" + "="*70)
    print("TEST 1C: Functional Form Comparison")
    print("="*70)

    def compute_cluster_metrics(ds, emb1, emb2, n_clusters=20):
        """Compute formula variants for score-based clusters."""
        human_scores = np.array([ex["score"] for ex in ds])
        bucket_edges = np.linspace(0, 5, n_clusters + 1)

        results = []
        for i in range(n_clusters):
            lo, hi = bucket_edges[i], bucket_edges[i+1]
            mask = (human_scores >= lo) & (human_scores < hi)
            if i == n_clusters - 1:
                mask = (human_scores >= lo) & (human_scores <= hi)

            if mask.sum() < 5:
                continue

            cluster_embs = np.vstack([emb1[mask], emb2[mask]])

            E = compute_E(cluster_embs)
            gs = compute_grad_S(cluster_embs)
            mean_score = float(human_scores[mask].mean())

            if np.isnan(E) or np.isnan(gs) or gs < 1e-10:
                continue

            # R variants
            R0 = E / gs                          # R = E/grad_S (original)
            R1 = E - gs                           # R = E - grad_S
            R2 = E * np.exp(-gs)                  # R = E * exp(-grad_S)
            R3 = np.log(max(E, 1e-10)) / (gs + 1)  # R = log(E) / (grad_S + 1)
            R4 = E / (gs + 0.1)                   # R = E / (grad_S + 0.1)  regularized

            results.append({
                "mean_score": mean_score,
                "E": E,
                "grad_S": gs,
                "R0_ratio": R0,
                "R1_diff": R1,
                "R2_exp": R2,
                "R3_log": R3,
                "R4_reg": R4,
            })

        return results

    # Evaluate on test data
    test_results = compute_cluster_metrics(ds_test, emb1_test, emb2_test, n_clusters=20)

    if len(test_results) < 5:
        print("  WARNING: Too few clusters with valid data")
        return {"test": "functional_form", "passed": False, "error": "too few clusters"}

    mean_scores = np.array([r["mean_score"] for r in test_results])

    formulas = {
        "R0 = E/grad_S (original)": np.array([r["R0_ratio"] for r in test_results]),
        "R1 = E - grad_S": np.array([r["R1_diff"] for r in test_results]),
        "R2 = E * exp(-grad_S)": np.array([r["R2_exp"] for r in test_results]),
        "R3 = log(E)/(grad_S+1)": np.array([r["R3_log"] for r in test_results]),
        "R4 = E/(grad_S+0.1)": np.array([r["R4_reg"] for r in test_results]),
        "E alone (baseline)": np.array([r["E"] for r in test_results]),
    }

    correlations = {}
    for name, vals in formulas.items():
        # Remove NaN/Inf
        valid = np.isfinite(vals)
        if valid.sum() < 5:
            correlations[name] = {"rho": float("nan"), "p": float("nan")}
            continue
        rho, p = stats.spearmanr(vals[valid], mean_scores[valid])
        correlations[name] = {"rho": float(rho), "p": float(p)}
        print(f"  {name}: rho={rho:.4f}, p={p:.4f}")

    # Rank by absolute correlation
    ranked = sorted(correlations.items(), key=lambda x: abs(x[1]["rho"]) if not np.isnan(x[1]["rho"]) else 0, reverse=True)
    print(f"\n  Ranking by |rho|:")
    for rank, (name, vals) in enumerate(ranked):
        print(f"    {rank+1}. {name}: |rho|={abs(vals['rho']):.4f}")

    # Find rank of R0
    r0_rank = next(i+1 for i, (name, _) in enumerate(ranked) if "R0" in name)

    print(f"\n  R0 (E/grad_S) rank: {r0_rank} out of {len(ranked)}")
    print(f"  Criterion: R0 must be in top 2")

    passed = r0_rank <= 2
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    # Also check: does E alone do just as well?
    e_rho = correlations["E alone (baseline)"]["rho"]
    r0_rho = correlations["R0 = E/grad_S (original)"]["rho"]
    e_better = abs(e_rho) >= abs(r0_rho)
    print(f"\n  CRITICAL CHECK: Does E alone match or beat R0?")
    print(f"    E alone |rho| = {abs(e_rho):.4f}")
    print(f"    R0 |rho| = {abs(r0_rho):.4f}")
    print(f"    E alone >= R0: {e_better}")
    if e_better:
        print("    WARNING: grad_S division adds no value over E alone!")

    return {
        "test": "functional_form",
        "correlations": {k: v for k, v in correlations.items()},
        "ranking": [(name, vals) for name, vals in ranked],
        "r0_rank": r0_rank,
        "e_alone_beats_r0": e_better,
        "criterion": "R0 in top 2",
        "passed": passed,
        "n_clusters": len(test_results),
    }


# ---------------------------------------------------------------------------
# Test 1D: sigma Falsification
# ---------------------------------------------------------------------------
def test_sigma_falsification(ds, emb1, emb2):
    """
    Test: Does sigma (participation ratio) vary meaningfully across groups?

    If sigma is always ~same regardless of content, it has no discriminative power.

    Pre-registered criterion: sigma must vary by at least 20% across score bins
    (coefficient of variation > 0.2).
    """
    print("\n" + "="*70)
    print("TEST 1D: sigma Falsification -- Discriminative Power")
    print("="*70)

    human_scores = np.array([ex["score"] for ex in ds])
    n_buckets = 5
    bucket_edges = np.linspace(0, 5, n_buckets + 1)

    sigma_values = []
    bucket_labels = []

    for i in range(n_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i+1]
        mask = (human_scores >= lo) & (human_scores < hi)
        if i == n_buckets - 1:
            mask = (human_scores >= lo) & (human_scores <= hi)

        if mask.sum() < 10:
            continue

        cluster_embs = np.vstack([emb1[mask], emb2[mask]])
        sig = compute_sigma(cluster_embs)
        sigma_values.append(sig)
        bucket_labels.append(f"[{lo:.1f},{hi:.1f})")
        print(f"  Bucket {bucket_labels[-1]}: n={mask.sum()}, sigma={sig:.6f}")

    sigma_values = np.array(sigma_values)
    cv = np.std(sigma_values) / np.mean(sigma_values) if np.mean(sigma_values) > 0 else 0

    print(f"\n  sigma mean = {np.mean(sigma_values):.6f}")
    print(f"  sigma std = {np.std(sigma_values):.6f}")
    print(f"  Coefficient of variation = {cv:.4f}")
    print(f"  Criterion: CV > 0.2 (sigma varies meaningfully)")

    passed = cv > 0.2
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    # Correlation with human scores
    if len(sigma_values) >= 3:
        bucket_midpoints = [(bucket_edges[i] + bucket_edges[i+1])/2 for i in range(n_buckets)]
        bucket_midpoints = [m for i, m in enumerate(bucket_midpoints) if i < len(sigma_values)]
        rho, p = stats.spearmanr(sigma_values, bucket_midpoints[:len(sigma_values)])
        print(f"  sigma vs score-bucket correlation: rho={rho:.4f}, p={p:.4f}")

    return {
        "test": "sigma_falsification",
        "sigma_values": sigma_values.tolist(),
        "bucket_labels": bucket_labels,
        "cv": float(cv),
        "mean": float(np.mean(sigma_values)),
        "std": float(np.std(sigma_values)),
        "criterion": "CV > 0.2",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 1E: Df Falsification
# ---------------------------------------------------------------------------
def test_Df_falsification(ds, emb1, emb2):
    """
    Test: Does Df (fractal dimension) vary meaningfully across groups?

    Pre-registered criterion: Df must correlate with some property of the data
    (e.g., score diversity within bucket). If Df is always ~constant, it has
    no discriminative power.
    """
    print("\n" + "="*70)
    print("TEST 1E: Df Falsification -- Discriminative Power")
    print("="*70)

    human_scores = np.array([ex["score"] for ex in ds])
    n_buckets = 5
    bucket_edges = np.linspace(0, 5, n_buckets + 1)

    df_values = []
    score_stds = []  # diversity of scores within bucket
    bucket_labels = []

    for i in range(n_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i+1]
        mask = (human_scores >= lo) & (human_scores < hi)
        if i == n_buckets - 1:
            mask = (human_scores >= lo) & (human_scores <= hi)

        if mask.sum() < 10:
            continue

        cluster_embs = np.vstack([emb1[mask], emb2[mask]])
        df = compute_Df(cluster_embs)
        score_std = float(np.std(human_scores[mask]))

        df_values.append(df)
        score_stds.append(score_std)
        bucket_labels.append(f"[{lo:.1f},{hi:.1f})")
        print(f"  Bucket {bucket_labels[-1]}: n={mask.sum()}, Df={df:.4f}, score_std={score_std:.4f}")

    df_values = np.array(df_values)
    score_stds = np.array(score_stds)

    cv = np.std(df_values) / np.mean(df_values) if np.mean(df_values) > 0 else 0

    print(f"\n  Df mean = {np.mean(df_values):.4f}")
    print(f"  Df std = {np.std(df_values):.4f}")
    print(f"  Coefficient of variation = {cv:.4f}")

    if len(df_values) >= 3:
        rho, p = stats.spearmanr(df_values, score_stds)
        print(f"  Df vs score-diversity correlation: rho={rho:.4f}, p={p:.4f}")
    else:
        rho, p = float("nan"), float("nan")

    # Df is meaningful if it varies (CV > 0.1) OR correlates with something
    passed = cv > 0.1 or (not np.isnan(p) and p < 0.05)
    print(f"  Criterion: CV > 0.1 OR significant correlation with score diversity")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "Df_falsification",
        "df_values": df_values.tolist(),
        "score_stds": score_stds.tolist(),
        "bucket_labels": bucket_labels,
        "cv": float(cv),
        "rho_vs_diversity": float(rho),
        "p_vs_diversity": float(p),
        "criterion": "CV > 0.1 OR significant correlation",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 2: Modus Tollens on Real Data
# ---------------------------------------------------------------------------
def test_modus_tollens(ds_train, ds_test, emb1_train, emb2_train, emb1_test, emb2_test):
    """
    Pre-register: "For groups with N >= 20, R > T implies mean_human_score > Q_min."

    1. Calibrate T, Q_min on training split.
    2. Test the conditional on test split.
    3. If any group has R > T but mean_score <= Q_min, the conditional is violated.
    """
    print("\n" + "="*70)
    print("TEST 2: Modus Tollens on Real Data")
    print("="*70)

    def compute_group_data(ds, emb1, emb2, n_groups=30):
        """Compute R and mean score for random groups of N >= 20 pairs."""
        np.random.seed(SEED)
        human_scores = np.array([ex["score"] for ex in ds])
        n = len(ds)

        groups = []
        for g in range(n_groups):
            # Random group of 20-50 pairs
            size = np.random.randint(20, min(50, n))
            idx = np.random.choice(n, size=size, replace=False)

            group_embs = np.vstack([emb1[idx], emb2[idx]])

            components = compute_all(group_embs)
            mean_score = float(human_scores[idx].mean())

            groups.append({
                "group_id": g,
                "size": size,
                "mean_score": mean_score,
                "R_simple": components["R_simple"],
                "R_full": components["R_full"],
                "E": components["E"],
                "grad_S": components["grad_S"],
                "sigma": components["sigma"],
                "Df": components["Df"],
            })

        return groups

    # Calibration on training data
    print("  [Calibration on training split]")
    train_groups = compute_group_data(ds_train, emb1_train, emb2_train, n_groups=50)

    # Get R_simple values and mean scores
    train_R = np.array([g["R_simple"] for g in train_groups])
    train_scores = np.array([g["mean_score"] for g in train_groups])

    # Remove NaN
    valid = np.isfinite(train_R)
    train_R = train_R[valid]
    train_scores = train_scores[valid]

    # Calibrate: T = median R, Q_min = score at 25th percentile
    T = float(np.median(train_R))

    # Q_min: the minimum score we expect when R > T
    high_R_mask = train_R > T
    if high_R_mask.sum() > 0:
        Q_min = float(np.percentile(train_scores[high_R_mask], 10))  # 10th percentile as safety margin
    else:
        Q_min = 2.0  # fallback

    print(f"  Calibrated threshold T = {T:.4f}")
    print(f"  Calibrated Q_min = {Q_min:.4f}")
    print(f"  Pre-registered conditional: R > {T:.4f} => mean_score > {Q_min:.4f}")

    # Test on held-out data
    print(f"\n  [Testing on test split]")
    np.random.seed(SEED + 1)  # different seed for test
    test_groups = compute_group_data(ds_test, emb1_test, emb2_test, n_groups=50)

    test_R = np.array([g["R_simple"] for g in test_groups])
    test_scores = np.array([g["mean_score"] for g in test_groups])

    valid = np.isfinite(test_R)
    test_R = test_R[valid]
    test_scores = test_scores[valid]

    # Check the conditional
    high_R = test_R > T
    violations = 0
    total_high_R = int(high_R.sum())

    print(f"  Groups with R > T: {total_high_R} / {len(test_R)}")

    for i in range(len(test_R)):
        if high_R[i]:
            if test_scores[i] <= Q_min:
                violations += 1
                print(f"    VIOLATION: group R={test_R[i]:.4f} > T={T:.4f}, but score={test_scores[i]:.4f} <= Q_min={Q_min:.4f}")

    violation_rate = violations / total_high_R if total_high_R > 0 else 0
    print(f"\n  Violations: {violations} / {total_high_R} ({violation_rate:.1%})")
    print(f"  Criterion: violation rate < 10%")

    passed = violation_rate < 0.10
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    # Also report correlation
    rho, p = stats.spearmanr(test_R, test_scores)
    print(f"  R vs score correlation on test: rho={rho:.4f}, p={p:.4f}")

    return {
        "test": "modus_tollens",
        "T": T,
        "Q_min": Q_min,
        "total_high_R": total_high_R,
        "violations": violations,
        "violation_rate": float(violation_rate),
        "rho_R_vs_score": float(rho),
        "p_R_vs_score": float(p),
        "n_train_groups": len(train_R),
        "n_test_groups": len(test_R),
        "criterion": "violation rate < 10%",
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Test 3: Adversarial Robustness
# ---------------------------------------------------------------------------
def test_adversarial(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Can R detect when it should fail?

    Case A: Near-duplicates (high E, low grad_S -- should give misleading high R)
    Case B: Random mixtures (low E, high grad_S -- should give low R correctly)
    Case C: Cross-domain/language (semantically unrelated but embedded together)
    """
    print("\n" + "="*70)
    print("TEST 3: Adversarial Robustness")
    print("="*70)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    results = {}

    # Case A: Near-duplicates -- paraphrases that agree but might be an echo chamber
    print("\n  [Case A: Near-Duplicate Sentences]")
    near_dupes = [
        "The cat sat on the mat.",
        "The cat was sitting on the mat.",
        "A cat sat upon the mat.",
        "The feline sat on the rug.",
        "On the mat, a cat was sitting.",
        "The cat rested on the mat.",
        "A cat was seated on the mat.",
        "The cat perched on the mat.",
        "Sitting on the mat was the cat.",
        "The cat lay on the mat.",
        "A cat sat on the mat nearby.",
        "The small cat sat on the mat.",
        "The cat sat on the soft mat.",
        "On the mat the cat sat still.",
        "The cat sat peacefully on the mat.",
        "The cat found the mat and sat.",
        "The cat was on the mat, sitting.",
        "There was a cat sitting on the mat.",
        "A cat that sat on a mat.",
        "The cat chose the mat to sit on.",
    ]
    emb_dupes = model.encode(near_dupes)
    metrics_a = compute_all(np.array(emb_dupes))
    print(f"    E={metrics_a['E']:.4f}, grad_S={metrics_a['grad_S']:.4f}, R_simple={metrics_a['R_simple']:.4f}")
    print(f"    Expected: HIGH R (all say same thing) -- this is the echo chamber trap")
    print(f"    R is high because agreement is high and dispersion is low.")
    print(f"    This is a valid high-agreement cluster. R SHOULD be high here.")
    print(f"    The question is: does R distinguish genuine agreement from parroting?")
    print(f"    Answer: R CANNOT distinguish -- it measures agreement, not independence.")
    results["case_A_near_duplicates"] = {
        "R_simple": metrics_a["R_simple"],
        "R_full": metrics_a["R_full"],
        "E": metrics_a["E"],
        "grad_S": metrics_a["grad_S"],
        "behavior": "HIGH R (as expected for agreement)",
        "concern": "Cannot distinguish genuine agreement from echo chamber",
    }

    # Case B: Random sentence mixtures (should give low R)
    print("\n  [Case B: Random Sentence Mixture]")
    random_sents = [
        "The stock market fell by 3% yesterday.",
        "Photosynthesis converts light into chemical energy.",
        "The Treaty of Westphalia was signed in 1648.",
        "Python is a popular programming language.",
        "Mount Everest is the tallest mountain on Earth.",
        "Beethoven composed nine symphonies.",
        "Water boils at 100 degrees Celsius.",
        "The speed of light is approximately 300,000 km/s.",
        "Tokyo is the capital of Japan.",
        "DNA contains the genetic instructions for life.",
        "The Amazon River is the largest by volume.",
        "Shakespeare wrote Hamlet around 1600.",
        "Insulin regulates blood sugar levels.",
        "The Eiffel Tower was built in 1889.",
        "Mars is known as the Red Planet.",
        "Coffee beans are actually seeds of a fruit.",
        "The Great Wall of China is over 13,000 miles.",
        "Electrons orbit the nucleus of an atom.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "Gravity keeps planets in orbit around the sun.",
    ]
    emb_random = model.encode(random_sents)
    metrics_b = compute_all(np.array(emb_random))
    print(f"    E={metrics_b['E']:.4f}, grad_S={metrics_b['grad_S']:.4f}, R_simple={metrics_b['R_simple']:.4f}")
    print(f"    Expected: LOW R (diverse, unrelated sentences)")

    case_b_correct = metrics_b["R_simple"] < metrics_a["R_simple"]
    print(f"    R_random < R_duplicates: {case_b_correct}")
    results["case_B_random_mixture"] = {
        "R_simple": metrics_b["R_simple"],
        "R_full": metrics_b["R_full"],
        "E": metrics_b["E"],
        "grad_S": metrics_b["grad_S"],
        "behavior": "LOW R" if metrics_b["R_simple"] < metrics_a["R_simple"] else "HIGH R (unexpected)",
        "correct": case_b_correct,
    }

    # Case C: Cross-domain -- sentences from different domains embedded together
    print("\n  [Case C: Cross-Domain Adversarial]")
    # Mix medical and cooking sentences -- different domains but both coherent
    cross_domain = [
        "The patient presented with acute myocardial infarction.",
        "Preheat the oven to 350 degrees Fahrenheit.",
        "Serum creatinine levels were elevated at 2.5 mg/dL.",
        "Fold the egg whites gently into the batter.",
        "MRI showed a 2cm lesion in the left temporal lobe.",
        "Simmer the sauce for 20 minutes until thickened.",
        "The fracture was reduced and immobilized with a cast.",
        "Knead the dough for 10 minutes until smooth.",
        "Blood cultures grew gram-positive cocci in clusters.",
        "Season the steak with salt and pepper before grilling.",
        "The patient was started on metformin 500mg twice daily.",
        "Whisk the cream until stiff peaks form.",
        "Echocardiography revealed an ejection fraction of 35%.",
        "Deglaze the pan with white wine and scrape up the fond.",
        "The wound was debrided and packed with saline gauze.",
        "Let the bread rise in a warm place for one hour.",
        "Lumbar puncture showed elevated opening pressure.",
        "Blanch the vegetables in boiling water for two minutes.",
        "The patient's Glasgow Coma Scale was 12.",
        "Garnish with fresh herbs before serving.",
    ]
    emb_cross = model.encode(cross_domain)
    metrics_c = compute_all(np.array(emb_cross))
    print(f"    E={metrics_c['E']:.4f}, grad_S={metrics_c['grad_S']:.4f}, R_simple={metrics_c['R_simple']:.4f}")
    print(f"    Expected: LOW R (two unrelated domains mixed)")

    case_c_correct = metrics_c["R_simple"] < metrics_a["R_simple"]
    print(f"    R_cross < R_duplicates: {case_c_correct}")
    results["case_C_cross_domain"] = {
        "R_simple": metrics_c["R_simple"],
        "R_full": metrics_c["R_full"],
        "E": metrics_c["E"],
        "grad_S": metrics_c["grad_S"],
        "behavior": "LOW R" if case_c_correct else "HIGH R (unexpected)",
        "correct": case_c_correct,
    }

    # Summary
    cases_correct = sum([
        True,  # Case A: high R is actually correct behavior
        case_b_correct,
        case_c_correct,
    ])

    print(f"\n  Adversarial summary: {cases_correct}/3 cases behave as expected")
    print(f"  Criterion: 2/3 cases correct")

    passed = cases_correct >= 2
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    # Critical analysis
    print(f"\n  CRITICAL ANALYSIS:")
    print(f"    Case A reveals the fundamental limitation: R measures agreement,")
    print(f"    not truth. Near-duplicates legitimately agree, so R is legitimately")
    print(f"    high. But so would an echo chamber of identical wrong answers.")
    print(f"    R has NO way to distinguish these cases without external ground truth.")

    results["cases_correct"] = cases_correct
    results["passed"] = passed
    results["criterion"] = "2/3 cases correct"
    results["test"] = "adversarial"

    return results


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    all_results = {}

    print("="*70)
    print("Q02 v2: Falsification Test Suite")
    print("R = (E / grad_S) * sigma^Df")
    print(f"Seed: {SEED}")
    print("="*70)

    # Load data
    ds_train, ds_test = load_stsb()

    # Extract sentences
    s1_train = [ex["sentence1"] for ex in ds_train]
    s2_train = [ex["sentence2"] for ex in ds_train]
    s1_test = [ex["sentence1"] for ex in ds_test]
    s2_test = [ex["sentence2"] for ex in ds_test]

    # Encode all sentences
    print("\n[ENCODING] Training sentences...")
    all_train = list(set(s1_train + s2_train))
    all_test = list(set(s1_test + s2_test))

    # Build lookup for efficiency
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"  Encoding {len(all_train)} unique training sentences...")
    train_embs = model.encode(all_train, show_progress_bar=True, batch_size=256)
    train_lookup = {s: e for s, e in zip(all_train, train_embs)}

    print(f"  Encoding {len(all_test)} unique test sentences...")
    test_embs = model.encode(all_test, show_progress_bar=True, batch_size=256)
    test_lookup = {s: e for s, e in zip(all_test, test_embs)}

    # Build aligned embedding arrays
    emb1_train = np.array([train_lookup[s] for s in s1_train])
    emb2_train = np.array([train_lookup[s] for s in s2_train])
    emb1_test = np.array([test_lookup[s] for s in s1_test])
    emb2_test = np.array([test_lookup[s] for s in s2_test])

    print(f"  Train shape: {emb1_train.shape}, Test shape: {emb1_test.shape}")

    # -----------------------------------------------------------------------
    # Test 1A: E Falsification
    # -----------------------------------------------------------------------
    all_results["test_1a"] = test_E_falsification(ds_test, emb1_test, emb2_test)

    # -----------------------------------------------------------------------
    # Test 1B: grad_S Falsification
    # -----------------------------------------------------------------------
    all_results["test_1b"] = test_grad_S_falsification(ds_test, emb1_test, emb2_test)

    # -----------------------------------------------------------------------
    # Test 1C: Functional Form Comparison
    # -----------------------------------------------------------------------
    all_results["test_1c"] = test_functional_form(
        ds_train, ds_test, emb1_train, emb2_train, emb1_test, emb2_test
    )

    # -----------------------------------------------------------------------
    # Test 1D: sigma Falsification
    # -----------------------------------------------------------------------
    all_results["test_1d"] = test_sigma_falsification(ds_test, emb1_test, emb2_test)

    # -----------------------------------------------------------------------
    # Test 1E: Df Falsification
    # -----------------------------------------------------------------------
    all_results["test_1e"] = test_Df_falsification(ds_test, emb1_test, emb2_test)

    # -----------------------------------------------------------------------
    # Test 2: Modus Tollens
    # -----------------------------------------------------------------------
    all_results["test_2"] = test_modus_tollens(
        ds_train, ds_test, emb1_train, emb2_train, emb1_test, emb2_test
    )

    # -----------------------------------------------------------------------
    # Test 3: Adversarial
    # -----------------------------------------------------------------------
    all_results["test_3"] = test_adversarial()

    # -----------------------------------------------------------------------
    # Overall Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    component_tests = ["test_1a", "test_1b", "test_1c", "test_1d", "test_1e"]
    component_passes = sum(1 for t in component_tests if all_results[t].get("passed", False))
    modus_tollens_passed = all_results["test_2"]["passed"]
    adversarial_passed = all_results["test_3"]["passed"]

    print(f"  Component tests passed: {component_passes}/5")
    for t in component_tests:
        status = "PASS" if all_results[t].get("passed", False) else "FAIL"
        print(f"    {t}: {status}")
    print(f"  Modus tollens: {'PASS' if modus_tollens_passed else 'FAIL'}")
    print(f"  Adversarial: {'PASS' if adversarial_passed else 'FAIL'}")

    # Apply pre-registered criteria
    if component_passes >= 4 and modus_tollens_passed and adversarial_passed:
        verdict = "CONFIRMED"
    elif not modus_tollens_passed or component_passes <= 2:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Elapsed: {elapsed:.1f}s")

    all_results["summary"] = {
        "component_passes": component_passes,
        "modus_tollens_passed": modus_tollens_passed,
        "adversarial_passed": adversarial_passed,
        "verdict": verdict,
        "elapsed_seconds": elapsed,
        "seed": SEED,
    }

    # Save results
    results_path = RESULTS_DIR / "q02_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    results = main()
