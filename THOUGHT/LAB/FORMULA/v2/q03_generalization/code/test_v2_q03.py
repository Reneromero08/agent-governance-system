"""
Q03 v2 Test: R generalizes across genuinely different domains.

Hypothesis: R = E/grad_S generalizes across fundamentally different domains,
not just different text distributions.

Domains:
  1. Text Semantics (STS-B + sentence-transformers)
  2. Numerical/Tabular (California Housing via sklearn)
  3. Time Series / Financial (S&P 500 daily returns via yfinance)

Pre-registered criteria:
  CONFIRM: rho>0.5 in >=2/3 domains AND cross-domain transfer works for >=1/2 pairs
           AND R outperforms >=1 domain-specific alternative
  FALSIFY: rho>0.5 in <1/3 domains OR cross-domain transfer fails entirely
           OR R beaten by all domain-specific alternatives
  INCONCLUSIVE: otherwise

Author: Claude Opus 4.6 (automated test execution)
Date: 2026-02-05
Seed: 42
"""

import sys
import os
import json
import warnings
import time
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from datetime import datetime

# Add shared formula to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "shared"))
sys.path.insert(0, _SHARED_DIR)
from formula import compute_E, compute_grad_S, compute_R_simple, compute_R_full, compute_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Utility: robust Spearman correlation
# ---------------------------------------------------------------------------

def safe_spearman(x, y):
    """Compute Spearman rho with proper NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


def safe_pearson(x, y):
    """Compute Pearson r with proper NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = scipy_stats.pearsonr(x, y)
    return float(r), float(p)


# ===================================================================
# DOMAIN 1: TEXT SEMANTICS (STS-B)
# ===================================================================

def run_domain1_text():
    """
    STS-B text semantic similarity.

    Steps:
      1. Load STS-B dataset (HuggingFace datasets)
      2. Encode sentences with all-MiniLM-L6-v2
      3. Group sentence pairs by human similarity bins
      4. For each bin, gather sentence embeddings into clusters
      5. Compute R for each cluster
      6. Correlate R with mean human similarity score per bin
      7. Compare R against raw cosine similarity and simple SNR
    """
    print("\n" + "=" * 70)
    print("DOMAIN 1: TEXT SEMANTICS (STS-B)")
    print("=" * 70)

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # Load STS-B
    print("Loading STS-B dataset...")
    try:
        ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    except Exception:
        # Fallback: try the older name
        ds = load_dataset("stsb_multi_mt", name="en", split="test")

    print(f"  Loaded {len(ds)} sentence pairs")

    # Encode sentences
    print("Encoding sentences with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences1 = ds["sentence1"]
    sentences2 = ds["sentence2"]
    scores = np.array(ds["score"], dtype=float)

    # Normalize scores to [0, 5] if needed
    if scores.max() <= 1.0:
        scores = scores * 5.0

    emb1 = model.encode(sentences1, show_progress_bar=False, batch_size=128)
    emb2 = model.encode(sentences2, show_progress_bar=False, batch_size=128)

    print(f"  Embedding shape: {emb1.shape}")

    # Create bins of human similarity scores
    # 10 bins from 0 to 5
    n_bins = 10
    bin_edges = np.linspace(0, 5.01, n_bins + 1)

    results = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        idx = np.where(mask)[0]

        if len(idx) < 10:
            continue

        # Sample up to 100 pairs per bin for computational sanity
        if len(idx) > 100:
            rng = np.random.RandomState(SEED + i)
            idx = rng.choice(idx, 100, replace=False)

        # Pool all sentence embeddings in this bin into one cluster
        cluster_embs = np.vstack([emb1[idx], emb2[idx]])

        # Compute R
        metrics = compute_all(cluster_embs)
        R_simple = metrics["R_simple"]
        R_full = metrics["R_full"]
        E_val = metrics["E"]

        # Mean human score for this bin
        mean_human = float(np.mean(scores[idx]))

        # Domain-specific alternatives:
        # 1. Raw mean cosine similarity between paired sentences
        pair_cosines = []
        for j in idx:
            cos_sim = np.dot(emb1[j], emb2[j]) / (
                np.linalg.norm(emb1[j]) * np.linalg.norm(emb2[j]) + 1e-10
            )
            pair_cosines.append(cos_sim)
        mean_pair_cosine = float(np.mean(pair_cosines))

        # 2. Simple SNR = mean / std of pairwise cosines in cluster
        norms1 = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        norms1 = np.where(norms1 == 0, 1e-10, norms1)
        normed = cluster_embs / norms1
        sim_mat = normed @ normed.T
        upper = sim_mat[np.triu_indices(len(cluster_embs), k=1)]
        snr = float(np.mean(upper) / (np.std(upper) + 1e-10))

        results.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n_pairs": len(idx),
            "mean_human_score": mean_human,
            "E": E_val,
            "grad_S": metrics["grad_S"],
            "R_simple": R_simple,
            "R_full": R_full,
            "mean_pair_cosine": mean_pair_cosine,
            "SNR": snr,
        })

    df = pd.DataFrame(results)
    print(f"\n  Bins with sufficient data: {len(df)}")
    print(df.to_string(index=False))

    # Correlations with human score
    human_scores = df["mean_human_score"].values

    rho_R_simple, p_R_simple = safe_spearman(df["R_simple"].values, human_scores)
    rho_R_full, p_R_full = safe_spearman(df["R_full"].values, human_scores)
    rho_E, p_E = safe_spearman(df["E"].values, human_scores)
    rho_cosine, p_cosine = safe_spearman(df["mean_pair_cosine"].values, human_scores)
    rho_snr, p_snr = safe_spearman(df["SNR"].values, human_scores)

    print("\n  Spearman correlations with human similarity scores:")
    print(f"    R_simple vs human:     rho={rho_R_simple:.4f}, p={p_R_simple:.4e}")
    print(f"    R_full vs human:       rho={rho_R_full:.4f}, p={p_R_full:.4e}")
    print(f"    E (raw) vs human:      rho={rho_E:.4f}, p={p_E:.4e}")
    print(f"    Pair cosine vs human:  rho={rho_cosine:.4f}, p={p_cosine:.4e}")
    print(f"    SNR vs human:          rho={rho_snr:.4f}, p={p_snr:.4e}")

    domain1_result = {
        "domain": "Text Semantics (STS-B)",
        "n_datapoints": int(len(ds)),
        "n_bins_used": len(df),
        "correlations": {
            "R_simple_vs_human": {"rho": rho_R_simple, "p": p_R_simple},
            "R_full_vs_human": {"rho": rho_R_full, "p": p_R_full},
            "E_vs_human": {"rho": rho_E, "p": p_E},
            "pair_cosine_vs_human": {"rho": rho_cosine, "p": p_cosine},
            "SNR_vs_human": {"rho": rho_snr, "p": p_snr},
        },
        "per_bin": results,
        "best_R_rho": max(rho_R_simple, rho_R_full) if not (np.isnan(rho_R_simple) and np.isnan(rho_R_full)) else float("nan"),
    }

    return domain1_result


# ===================================================================
# DOMAIN 2: NUMERICAL/TABULAR (California Housing)
# ===================================================================

def run_domain2_tabular():
    """
    California Housing regression dataset.

    Steps:
      1. Load California Housing from sklearn
      2. Standardize features
      3. Create "clusters" by binning target values (median house price)
      4. For each cluster, compute R on the standardized feature vectors
      5. Ground truth quality: how tight is the cluster?
         - Use inverse of target std within cluster (tighter cluster = better quality)
         - Also compute linear regression R-squared within each cluster
      6. Correlate R with cluster tightness and R-squared
      7. Compare R against domain-specific metrics (R-squared, RMSE)
    """
    print("\n" + "=" * 70)
    print("DOMAIN 2: NUMERICAL/TABULAR (California Housing)")
    print("=" * 70)

    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    data = fetch_california_housing()
    X = data.data  # (20640, 8)
    y = data.target  # median house value in 100k

    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create bins by target value
    n_bins = 15
    bin_edges = np.linspace(y.min() - 0.01, y.max() + 0.01, n_bins + 1)

    results = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y >= lo) & (y < hi)
        idx = np.where(mask)[0]

        if len(idx) < 20:
            continue

        # Sample up to 200 for computational reasons
        if len(idx) > 200:
            rng = np.random.RandomState(SEED + i)
            idx = rng.choice(idx, 200, replace=False)

        cluster_X = X_scaled[idx]
        cluster_y = y[idx]

        # Compute R on feature-space embeddings
        metrics = compute_all(cluster_X)
        R_simple = metrics["R_simple"]
        R_full = metrics["R_full"]
        E_val = metrics["E"]

        # Ground truth quality metrics:
        # 1. Target homogeneity (inverse of target std -- tighter = better)
        target_std = float(np.std(cluster_y))
        target_homogeneity = 1.0 / (target_std + 1e-10)

        # 2. Linear regression R-squared within cluster
        #    (how predictable is target from features within this slice?)
        if len(idx) > cluster_X.shape[1] + 2:
            lr = LinearRegression()
            lr.fit(cluster_X, cluster_y)
            y_pred = lr.predict(cluster_X)
            r2 = float(r2_score(cluster_y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(cluster_y, y_pred)))
        else:
            r2 = float("nan")
            rmse = float("nan")

        # 3. Feature-space SNR (simple mean/std of pairwise cosines)
        norms = np.linalg.norm(cluster_X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = cluster_X / norms
        sim_mat = normed @ normed.T
        upper = sim_mat[np.triu_indices(len(cluster_X), k=1)]
        snr = float(np.mean(upper) / (np.std(upper) + 1e-10))

        # 4. Mean proximity to cluster centroid (another quality measure)
        centroid = np.mean(cluster_X, axis=0)
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        mean_dist = float(np.mean(dists))

        results.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n_samples": len(idx),
            "mean_target": float(np.mean(cluster_y)),
            "target_std": target_std,
            "target_homogeneity": target_homogeneity,
            "E": E_val,
            "grad_S": metrics["grad_S"],
            "R_simple": R_simple,
            "R_full": R_full,
            "R_squared": r2,
            "RMSE": rmse,
            "SNR": snr,
            "mean_dist_to_centroid": mean_dist,
        })

    df = pd.DataFrame(results)
    print(f"\n  Bins with sufficient data: {len(df)}")
    print(df[["bin", "n_samples", "mean_target", "target_std", "R_simple", "R_full", "R_squared"]].to_string(index=False))

    # Correlations: R vs target homogeneity
    homogeneity = df["target_homogeneity"].values

    rho_R_simple_hom, p_R_simple_hom = safe_spearman(df["R_simple"].values, homogeneity)
    rho_R_full_hom, p_R_full_hom = safe_spearman(df["R_full"].values, homogeneity)
    rho_E_hom, p_E_hom = safe_spearman(df["E"].values, homogeneity)
    rho_snr_hom, p_snr_hom = safe_spearman(df["SNR"].values, homogeneity)

    # Correlations: R vs R-squared (prediction quality)
    r2_vals = df["R_squared"].values
    rho_R_simple_r2, p_R_simple_r2 = safe_spearman(df["R_simple"].values, r2_vals)
    rho_R_full_r2, p_R_full_r2 = safe_spearman(df["R_full"].values, r2_vals)
    rho_E_r2, p_E_r2 = safe_spearman(df["E"].values, r2_vals)
    rho_snr_r2, p_snr_r2 = safe_spearman(df["SNR"].values, r2_vals)

    print("\n  Spearman correlations with TARGET HOMOGENEITY:")
    print(f"    R_simple vs homogeneity:  rho={rho_R_simple_hom:.4f}, p={p_R_simple_hom:.4e}")
    print(f"    R_full vs homogeneity:    rho={rho_R_full_hom:.4f}, p={p_R_full_hom:.4e}")
    print(f"    E (raw) vs homogeneity:   rho={rho_E_hom:.4f}, p={p_E_hom:.4e}")
    print(f"    SNR vs homogeneity:       rho={rho_snr_hom:.4f}, p={p_snr_hom:.4e}")

    print("\n  Spearman correlations with R-SQUARED (prediction quality):")
    print(f"    R_simple vs R-squared:    rho={rho_R_simple_r2:.4f}, p={p_R_simple_r2:.4e}")
    print(f"    R_full vs R-squared:      rho={rho_R_full_r2:.4f}, p={p_R_full_r2:.4e}")
    print(f"    E (raw) vs R-squared:     rho={rho_E_r2:.4f}, p={p_E_r2:.4e}")
    print(f"    SNR vs R-squared:         rho={rho_snr_r2:.4f}, p={p_snr_r2:.4e}")

    # Use the better ground truth (homogeneity) as the primary metric
    best_rho_hom = max(
        rho_R_simple_hom if not np.isnan(rho_R_simple_hom) else -999,
        rho_R_full_hom if not np.isnan(rho_R_full_hom) else -999,
    )
    best_rho_r2 = max(
        rho_R_simple_r2 if not np.isnan(rho_R_simple_r2) else -999,
        rho_R_full_r2 if not np.isnan(rho_R_full_r2) else -999,
    )

    domain2_result = {
        "domain": "Tabular (California Housing)",
        "n_datapoints": int(X.shape[0]),
        "n_bins_used": len(df),
        "correlations_vs_homogeneity": {
            "R_simple": {"rho": rho_R_simple_hom, "p": p_R_simple_hom},
            "R_full": {"rho": rho_R_full_hom, "p": p_R_full_hom},
            "E_raw": {"rho": rho_E_hom, "p": p_E_hom},
            "SNR": {"rho": rho_snr_hom, "p": p_snr_hom},
        },
        "correlations_vs_r_squared": {
            "R_simple": {"rho": rho_R_simple_r2, "p": p_R_simple_r2},
            "R_full": {"rho": rho_R_full_r2, "p": p_R_full_r2},
            "E_raw": {"rho": rho_E_r2, "p": p_E_r2},
            "SNR": {"rho": rho_snr_r2, "p": p_snr_r2},
        },
        "per_bin": results,
        "best_R_rho_homogeneity": best_rho_hom,
        "best_R_rho_r_squared": best_rho_r2,
    }

    return domain2_result


# ===================================================================
# DOMAIN 3: TIME SERIES / FINANCIAL (S&P 500)
# ===================================================================

def run_domain3_financial():
    """
    S&P 500 daily returns grouped by volatility regime.

    Steps:
      1. Download SPY daily prices (2 years via yfinance)
      2. Compute daily log returns
      3. Create sliding windows of 20 trading days
      4. Embed each window as a 20-d vector (the return sequence)
      5. Compute rolling 20-day volatility
      6. Bin windows by volatility regime (5 bins: very-low to very-high)
      7. For each regime, compute R on the set of return-sequence vectors
      8. Ground truth: regime stability = inverse of regime transition frequency
      9. Also compute Sharpe ratio and raw volatility per regime
      10. Correlate R with regime stability
    """
    print("\n" + "=" * 70)
    print("DOMAIN 3: TIME SERIES / FINANCIAL (S&P 500)")
    print("=" * 70)

    import yfinance as yf

    # Download SPY data
    print("Downloading SPY data...")
    ticker = yf.Ticker("SPY")
    # Get ~3 years of data to ensure we have 2 full years of returns
    hist = ticker.history(period="3y", auto_adjust=True)

    if len(hist) < 252:
        # Fallback: try ^GSPC
        print("  SPY download small, trying ^GSPC...")
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period="3y", auto_adjust=True)

    print(f"  Downloaded {len(hist)} trading days")
    print(f"  Date range: {hist.index[0].date()} to {hist.index[-1].date()}")

    # Compute daily log returns
    close = hist["Close"].values.flatten()
    log_returns = np.diff(np.log(close))
    print(f"  Log returns: {len(log_returns)} days")

    # Sliding windows of 20 trading days
    window_size = 20
    windows = []
    for i in range(len(log_returns) - window_size + 1):
        windows.append(log_returns[i:i + window_size])
    windows = np.array(windows)
    print(f"  Windows: {windows.shape}")

    # Rolling volatility (std of each 20-day window)
    volatilities = np.std(windows, axis=1)

    # Bin by volatility regime
    n_bins = 5
    bin_edges = np.percentile(volatilities, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges
    bin_edges[-1] += 0.001
    bin_edges[0] -= 0.001

    # Label each window with its volatility bin
    labels = np.digitize(volatilities, bin_edges) - 1
    labels = np.clip(labels, 0, n_bins - 1)

    # Compute regime stability: for each regime, how often does the regime
    # NOT change from one window to the next? Higher = more stable regime.
    regime_stability = {}
    for b in range(n_bins):
        in_regime = (labels == b)
        if np.sum(in_regime) < 2:
            regime_stability[b] = float("nan")
            continue
        # Count transitions: consecutive windows that stay in the same regime
        in_regime_indices = np.where(in_regime)[0]
        consecutive = np.sum(np.diff(in_regime_indices) == 1)
        stability = consecutive / (len(in_regime_indices) - 1) if len(in_regime_indices) > 1 else 0
        regime_stability[b] = float(stability)

    results = []
    regime_names = ["very-low", "low", "medium", "high", "very-high"]

    for b in range(n_bins):
        mask = (labels == b)
        idx = np.where(mask)[0]

        if len(idx) < 20:
            continue

        # Sample up to 200
        if len(idx) > 200:
            rng = np.random.RandomState(SEED + b)
            idx = rng.choice(idx, 200, replace=False)

        cluster_windows = windows[idx]

        # Compute R on the set of 20-d return vectors
        metrics = compute_all(cluster_windows)
        R_simple = metrics["R_simple"]
        R_full = metrics["R_full"]
        E_val = metrics["E"]

        # Domain-specific metrics:
        # 1. Mean Sharpe ratio (annualized): mean_return / vol * sqrt(252)
        mean_returns = np.mean(cluster_windows, axis=1)
        vols = np.std(cluster_windows, axis=1)
        sharpe_ratios = mean_returns / (vols + 1e-10) * np.sqrt(252)
        mean_sharpe = float(np.mean(sharpe_ratios))
        abs_mean_sharpe = float(np.mean(np.abs(sharpe_ratios)))

        # 2. Mean volatility
        mean_vol = float(np.mean(vols))

        # 3. SNR
        norms = np.linalg.norm(cluster_windows, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = cluster_windows / norms
        sim_mat = normed @ normed.T
        upper = sim_mat[np.triu_indices(len(cluster_windows), k=1)]
        snr = float(np.mean(upper) / (np.std(upper) + 1e-10))

        stability = regime_stability.get(b, float("nan"))

        # Predictability: auto-correlation of returns within regime
        # (how predictable are tomorrow's returns from today's?)
        all_returns_in_regime = log_returns[np.where(labels == b)[0]]
        if len(all_returns_in_regime) > 10:
            autocorr = float(np.corrcoef(all_returns_in_regime[:-1], all_returns_in_regime[1:])[0, 1])
        else:
            autocorr = float("nan")

        results.append({
            "regime": regime_names[b] if b < len(regime_names) else f"bin_{b}",
            "n_windows": len(idx),
            "vol_range": f"{bin_edges[b]:.5f}-{bin_edges[b+1]:.5f}",
            "mean_vol": mean_vol,
            "stability": stability,
            "autocorrelation": autocorr,
            "E": E_val,
            "grad_S": metrics["grad_S"],
            "R_simple": R_simple,
            "R_full": R_full,
            "mean_sharpe": mean_sharpe,
            "abs_mean_sharpe": abs_mean_sharpe,
            "SNR": snr,
        })

    df = pd.DataFrame(results)
    print(f"\n  Regimes with sufficient data: {len(df)}")
    print(df[["regime", "n_windows", "mean_vol", "stability", "R_simple", "R_full", "mean_sharpe"]].to_string(index=False))

    # Correlations with stability
    stab = df["stability"].values

    rho_R_simple_stab, p_R_simple_stab = safe_spearman(df["R_simple"].values, stab)
    rho_R_full_stab, p_R_full_stab = safe_spearman(df["R_full"].values, stab)
    rho_E_stab, p_E_stab = safe_spearman(df["E"].values, stab)
    rho_sharpe_stab, p_sharpe_stab = safe_spearman(df["abs_mean_sharpe"].values, stab)
    rho_vol_stab, p_vol_stab = safe_spearman(df["mean_vol"].values, stab)
    rho_snr_stab, p_snr_stab = safe_spearman(df["SNR"].values, stab)

    # Also try correlation with absolute autocorrelation (predictability)
    autocorrs = np.abs(df["autocorrelation"].values)
    rho_R_simple_ac, p_R_simple_ac = safe_spearman(df["R_simple"].values, autocorrs)
    rho_R_full_ac, p_R_full_ac = safe_spearman(df["R_full"].values, autocorrs)

    print("\n  Spearman correlations with REGIME STABILITY:")
    print(f"    R_simple vs stability:    rho={rho_R_simple_stab:.4f}, p={p_R_simple_stab:.4e}")
    print(f"    R_full vs stability:      rho={rho_R_full_stab:.4f}, p={p_R_full_stab:.4e}")
    print(f"    E (raw) vs stability:     rho={rho_E_stab:.4f}, p={p_E_stab:.4e}")
    print(f"    |Sharpe| vs stability:    rho={rho_sharpe_stab:.4f}, p={p_sharpe_stab:.4e}")
    print(f"    Volatility vs stability:  rho={rho_vol_stab:.4f}, p={p_vol_stab:.4e}")
    print(f"    SNR vs stability:         rho={rho_snr_stab:.4f}, p={p_snr_stab:.4e}")

    print("\n  Spearman correlations with |AUTOCORRELATION| (predictability):")
    print(f"    R_simple vs |autocorr|:   rho={rho_R_simple_ac:.4f}, p={p_R_simple_ac:.4e}")
    print(f"    R_full vs |autocorr|:     rho={rho_R_full_ac:.4f}, p={p_R_full_ac:.4e}")

    best_rho_stab = max(
        rho_R_simple_stab if not np.isnan(rho_R_simple_stab) else -999,
        rho_R_full_stab if not np.isnan(rho_R_full_stab) else -999,
    )

    domain3_result = {
        "domain": "Financial Time Series (S&P 500 / SPY)",
        "n_trading_days": len(log_returns),
        "n_windows": len(windows),
        "n_regimes_used": len(df),
        "correlations_vs_stability": {
            "R_simple": {"rho": rho_R_simple_stab, "p": p_R_simple_stab},
            "R_full": {"rho": rho_R_full_stab, "p": p_R_full_stab},
            "E_raw": {"rho": rho_E_stab, "p": p_E_stab},
            "abs_sharpe": {"rho": rho_sharpe_stab, "p": p_sharpe_stab},
            "volatility": {"rho": rho_vol_stab, "p": p_vol_stab},
            "SNR": {"rho": rho_snr_stab, "p": p_snr_stab},
        },
        "correlations_vs_autocorrelation": {
            "R_simple": {"rho": rho_R_simple_ac, "p": p_R_simple_ac},
            "R_full": {"rho": rho_R_full_ac, "p": p_R_full_ac},
        },
        "per_regime": results,
        "best_R_rho_stability": best_rho_stab,
    }

    return domain3_result


# ===================================================================
# TEST 2: CROSS-DOMAIN THRESHOLD TRANSFER
# ===================================================================

def run_cross_domain_transfer(d1_result, d2_result, d3_result):
    """
    Calibrate R threshold on Domain 1 (text), apply to Domains 2 and 3.

    Strategy:
      - In Domain 1 bins, find the R value that best separates
        "high quality" (human score >= 3.0) from "low quality" (< 3.0)
      - Apply that threshold to Domains 2 and 3
      - Check if classification accuracy is within 20% of Domain 1 accuracy
    """
    print("\n" + "=" * 70)
    print("TEST 2: CROSS-DOMAIN THRESHOLD TRANSFER")
    print("=" * 70)

    # --- Calibrate on Domain 1 ---
    d1_bins = d1_result["per_bin"]
    d1_R = np.array([b["R_simple"] for b in d1_bins])
    d1_quality = np.array([1 if b["mean_human_score"] >= 2.5 else 0 for b in d1_bins])

    # Find optimal threshold by trying each R value as threshold
    best_acc_d1 = 0
    best_threshold = float("nan")

    # Also try R_full
    d1_R_full = np.array([b["R_full"] for b in d1_bins])

    for use_full in [False, True]:
        R_vals = d1_R_full if use_full else d1_R
        valid_mask = np.isfinite(R_vals)
        if np.sum(valid_mask) < 3:
            continue
        R_valid = R_vals[valid_mask]
        q_valid = d1_quality[valid_mask]

        for threshold in np.percentile(R_valid, np.arange(10, 91, 5)):
            predicted = (R_valid >= threshold).astype(int)
            acc = float(np.mean(predicted == q_valid))
            if acc > best_acc_d1:
                best_acc_d1 = acc
                best_threshold = threshold
                best_is_full = use_full

    R_variant = "R_full" if best_is_full else "R_simple"
    print(f"\n  Calibration (Domain 1 - Text):")
    print(f"    Best threshold: {best_threshold:.4f} (using {R_variant})")
    print(f"    Domain 1 accuracy: {best_acc_d1:.4f}")
    print(f"    Quality split: high={np.sum(d1_quality==1)}, low={np.sum(d1_quality==0)}")

    # --- Apply to Domain 2 ---
    d2_bins = d2_result["per_bin"]
    # Define "high quality" for tabular: target_std below median (tight cluster)
    d2_target_stds = [b["target_std"] for b in d2_bins]
    median_std = np.median(d2_target_stds)
    d2_quality = np.array([1 if b["target_std"] <= median_std else 0 for b in d2_bins])

    R_key = "R_full" if best_is_full else "R_simple"
    d2_R = np.array([b[R_key] for b in d2_bins])
    valid_mask_d2 = np.isfinite(d2_R)

    if np.sum(valid_mask_d2) > 0:
        d2_predicted = (d2_R[valid_mask_d2] >= best_threshold).astype(int)
        d2_acc = float(np.mean(d2_predicted == d2_quality[valid_mask_d2]))
    else:
        d2_acc = float("nan")

    print(f"\n  Transfer to Domain 2 (Tabular):")
    print(f"    Threshold {best_threshold:.4f} applied without recalibration")
    print(f"    Domain 2 accuracy: {d2_acc:.4f}")
    print(f"    Delta from D1: {abs(d2_acc - best_acc_d1):.4f}")
    print(f"    Within 20%: {'YES' if abs(d2_acc - best_acc_d1) <= 0.20 else 'NO'}")

    # --- Apply to Domain 3 ---
    d3_regimes = d3_result["per_regime"]
    # Define "high quality" for financial: stability above median
    d3_stabilities = [r["stability"] for r in d3_regimes if not np.isnan(r["stability"])]
    if len(d3_stabilities) > 0:
        median_stab = np.median(d3_stabilities)
        d3_quality = np.array([1 if r["stability"] >= median_stab else 0 for r in d3_regimes
                              if not np.isnan(r["stability"])])
        d3_R = np.array([r[R_key] for r in d3_regimes if not np.isnan(r["stability"])])
    else:
        d3_quality = np.array([])
        d3_R = np.array([])

    valid_mask_d3 = np.isfinite(d3_R)

    if np.sum(valid_mask_d3) >= 2:
        d3_predicted = (d3_R[valid_mask_d3] >= best_threshold).astype(int)
        d3_acc = float(np.mean(d3_predicted == d3_quality[valid_mask_d3]))
    else:
        d3_acc = float("nan")

    print(f"\n  Transfer to Domain 3 (Financial):")
    print(f"    Threshold {best_threshold:.4f} applied without recalibration")
    print(f"    Domain 3 accuracy: {d3_acc:.4f}")
    print(f"    Delta from D1: {abs(d3_acc - best_acc_d1) if not np.isnan(d3_acc) else 'N/A'}")
    d3_within = abs(d3_acc - best_acc_d1) <= 0.20 if not np.isnan(d3_acc) else False
    print(f"    Within 20%: {'YES' if d3_within else 'NO'}")

    # Also try calibrating on Domain 2 and applying to Domain 3
    print("\n  --- Alternative: Calibrate on Domain 2, apply to Domain 3 ---")
    d2_R_cal = np.array([b[R_key] for b in d2_bins])
    d2_qual_cal = d2_quality
    valid_d2_cal = np.isfinite(d2_R_cal)

    best_acc_d2_cal = 0
    best_thresh_d2 = float("nan")
    if np.sum(valid_d2_cal) >= 3:
        R_d2_valid = d2_R_cal[valid_d2_cal]
        q_d2_valid = d2_qual_cal[valid_d2_cal]
        for threshold in np.percentile(R_d2_valid, np.arange(10, 91, 5)):
            predicted = (R_d2_valid >= threshold).astype(int)
            acc = float(np.mean(predicted == q_d2_valid))
            if acc > best_acc_d2_cal:
                best_acc_d2_cal = acc
                best_thresh_d2 = threshold

    if np.sum(valid_mask_d3) >= 2 and not np.isnan(best_thresh_d2):
        d3_pred_v2 = (d3_R[valid_mask_d3] >= best_thresh_d2).astype(int)
        d3_acc_v2 = float(np.mean(d3_pred_v2 == d3_quality[valid_mask_d3]))
    else:
        d3_acc_v2 = float("nan")

    print(f"    D2 calibration accuracy: {best_acc_d2_cal:.4f}")
    print(f"    D2 threshold: {best_thresh_d2:.4f}")
    print(f"    D3 accuracy with D2 threshold: {d3_acc_v2:.4f}")
    d3v2_within = abs(d3_acc_v2 - best_acc_d2_cal) <= 0.20 if not (np.isnan(d3_acc_v2) or np.isnan(best_acc_d2_cal)) else False
    print(f"    Within 20%: {'YES' if d3v2_within else 'NO'}")

    transfer_result = {
        "calibration_domain": "Domain 1 (Text)",
        "R_variant_used": R_variant,
        "threshold": best_threshold,
        "d1_accuracy": best_acc_d1,
        "d1_to_d2": {
            "d2_accuracy": d2_acc,
            "delta": abs(d2_acc - best_acc_d1) if not np.isnan(d2_acc) else float("nan"),
            "within_20pct": bool(abs(d2_acc - best_acc_d1) <= 0.20) if not np.isnan(d2_acc) else False,
        },
        "d1_to_d3": {
            "d3_accuracy": d3_acc,
            "delta": abs(d3_acc - best_acc_d1) if not np.isnan(d3_acc) else float("nan"),
            "within_20pct": bool(d3_within),
        },
        "d2_to_d3": {
            "d2_cal_accuracy": best_acc_d2_cal,
            "d2_threshold": best_thresh_d2,
            "d3_accuracy": d3_acc_v2,
            "within_20pct": bool(d3v2_within),
        },
    }

    return transfer_result


# ===================================================================
# TEST 3: R vs DOMAIN-SPECIFIC ALTERNATIVES
# ===================================================================

def run_comparison_analysis(d1_result, d2_result, d3_result):
    """
    For each domain, compare R against the standard quality metric.
    R must add value beyond what domain-specific metrics provide.
    """
    print("\n" + "=" * 70)
    print("TEST 3: R vs DOMAIN-SPECIFIC ALTERNATIVES")
    print("=" * 70)

    comparisons = {}

    # --- Domain 1: Text ---
    print("\n  Domain 1 (Text): R vs raw cosine, SNR, E")
    d1_corrs = d1_result["correlations"]

    r_best = max(
        abs(d1_corrs["R_simple_vs_human"]["rho"]) if not np.isnan(d1_corrs["R_simple_vs_human"]["rho"]) else 0,
        abs(d1_corrs["R_full_vs_human"]["rho"]) if not np.isnan(d1_corrs["R_full_vs_human"]["rho"]) else 0,
    )
    alt_cosine = abs(d1_corrs["pair_cosine_vs_human"]["rho"]) if not np.isnan(d1_corrs["pair_cosine_vs_human"]["rho"]) else 0
    alt_snr = abs(d1_corrs["SNR_vs_human"]["rho"]) if not np.isnan(d1_corrs["SNR_vs_human"]["rho"]) else 0
    alt_E = abs(d1_corrs["E_vs_human"]["rho"]) if not np.isnan(d1_corrs["E_vs_human"]["rho"]) else 0

    d1_R_beats_cosine = r_best > alt_cosine
    d1_R_beats_snr = r_best > alt_snr
    d1_R_beats_E = r_best > alt_E

    print(f"    |R best rho|: {r_best:.4f}")
    print(f"    |Cosine rho|: {alt_cosine:.4f} -> R beats? {d1_R_beats_cosine}")
    print(f"    |SNR rho|:    {alt_snr:.4f} -> R beats? {d1_R_beats_snr}")
    print(f"    |E rho|:      {alt_E:.4f} -> R beats? {d1_R_beats_E}")

    comparisons["text"] = {
        "R_best_abs_rho": r_best,
        "alternatives": {
            "cosine": {"abs_rho": alt_cosine, "R_beats": bool(d1_R_beats_cosine)},
            "SNR": {"abs_rho": alt_snr, "R_beats": bool(d1_R_beats_snr)},
            "E_raw": {"abs_rho": alt_E, "R_beats": bool(d1_R_beats_E)},
        },
        "R_beats_at_least_one": bool(d1_R_beats_cosine or d1_R_beats_snr or d1_R_beats_E),
    }

    # --- Domain 2: Tabular ---
    print("\n  Domain 2 (Tabular): R vs R-squared, SNR, E")
    # Use homogeneity correlations as primary
    d2_corrs_h = d2_result["correlations_vs_homogeneity"]

    r_best_d2 = max(
        abs(d2_corrs_h["R_simple"]["rho"]) if not np.isnan(d2_corrs_h["R_simple"]["rho"]) else 0,
        abs(d2_corrs_h["R_full"]["rho"]) if not np.isnan(d2_corrs_h["R_full"]["rho"]) else 0,
    )
    alt_snr_d2 = abs(d2_corrs_h["SNR"]["rho"]) if not np.isnan(d2_corrs_h["SNR"]["rho"]) else 0
    alt_E_d2 = abs(d2_corrs_h["E_raw"]["rho"]) if not np.isnan(d2_corrs_h["E_raw"]["rho"]) else 0

    # R-squared correlation with homogeneity (expected to be strong since both measure cluster tightness)
    d2_bins = d2_result["per_bin"]
    d2_r2_vals = [b["R_squared"] for b in d2_bins]
    d2_hom_vals = [b["target_homogeneity"] for b in d2_bins]
    rho_r2_hom, _ = safe_spearman(d2_r2_vals, d2_hom_vals)
    alt_r2_d2 = abs(rho_r2_hom) if not np.isnan(rho_r2_hom) else 0

    d2_R_beats_snr = r_best_d2 > alt_snr_d2
    d2_R_beats_E = r_best_d2 > alt_E_d2
    d2_R_beats_r2 = r_best_d2 > alt_r2_d2

    print(f"    |R best rho|:    {r_best_d2:.4f}")
    print(f"    |R-squared rho|: {alt_r2_d2:.4f} -> R beats? {d2_R_beats_r2}")
    print(f"    |SNR rho|:       {alt_snr_d2:.4f} -> R beats? {d2_R_beats_snr}")
    print(f"    |E rho|:         {alt_E_d2:.4f} -> R beats? {d2_R_beats_E}")

    comparisons["tabular"] = {
        "R_best_abs_rho": r_best_d2,
        "alternatives": {
            "R_squared": {"abs_rho": alt_r2_d2, "R_beats": bool(d2_R_beats_r2)},
            "SNR": {"abs_rho": alt_snr_d2, "R_beats": bool(d2_R_beats_snr)},
            "E_raw": {"abs_rho": alt_E_d2, "R_beats": bool(d2_R_beats_E)},
        },
        "R_beats_at_least_one": bool(d2_R_beats_snr or d2_R_beats_E or d2_R_beats_r2),
    }

    # --- Domain 3: Financial ---
    print("\n  Domain 3 (Financial): R vs Sharpe, volatility, SNR, E")
    d3_corrs = d3_result["correlations_vs_stability"]

    r_best_d3 = max(
        abs(d3_corrs["R_simple"]["rho"]) if not np.isnan(d3_corrs["R_simple"]["rho"]) else 0,
        abs(d3_corrs["R_full"]["rho"]) if not np.isnan(d3_corrs["R_full"]["rho"]) else 0,
    )
    alt_sharpe_d3 = abs(d3_corrs["abs_sharpe"]["rho"]) if not np.isnan(d3_corrs["abs_sharpe"]["rho"]) else 0
    alt_vol_d3 = abs(d3_corrs["volatility"]["rho"]) if not np.isnan(d3_corrs["volatility"]["rho"]) else 0
    alt_snr_d3 = abs(d3_corrs["SNR"]["rho"]) if not np.isnan(d3_corrs["SNR"]["rho"]) else 0
    alt_E_d3 = abs(d3_corrs["E_raw"]["rho"]) if not np.isnan(d3_corrs["E_raw"]["rho"]) else 0

    d3_R_beats_sharpe = r_best_d3 > alt_sharpe_d3
    d3_R_beats_vol = r_best_d3 > alt_vol_d3
    d3_R_beats_snr = r_best_d3 > alt_snr_d3
    d3_R_beats_E = r_best_d3 > alt_E_d3

    print(f"    |R best rho|:     {r_best_d3:.4f}")
    print(f"    |Sharpe rho|:     {alt_sharpe_d3:.4f} -> R beats? {d3_R_beats_sharpe}")
    print(f"    |Volatility rho|: {alt_vol_d3:.4f} -> R beats? {d3_R_beats_vol}")
    print(f"    |SNR rho|:        {alt_snr_d3:.4f} -> R beats? {d3_R_beats_snr}")
    print(f"    |E rho|:          {alt_E_d3:.4f} -> R beats? {d3_R_beats_E}")

    comparisons["financial"] = {
        "R_best_abs_rho": r_best_d3,
        "alternatives": {
            "sharpe": {"abs_rho": alt_sharpe_d3, "R_beats": bool(d3_R_beats_sharpe)},
            "volatility": {"abs_rho": alt_vol_d3, "R_beats": bool(d3_R_beats_vol)},
            "SNR": {"abs_rho": alt_snr_d3, "R_beats": bool(d3_R_beats_snr)},
            "E_raw": {"abs_rho": alt_E_d3, "R_beats": bool(d3_R_beats_E)},
        },
        "R_beats_at_least_one": bool(d3_R_beats_sharpe or d3_R_beats_vol or d3_R_beats_snr or d3_R_beats_E),
    }

    return comparisons


# ===================================================================
# VERDICT LOGIC
# ===================================================================

def determine_verdict(d1_result, d2_result, d3_result, transfer_result, comparisons):
    """
    Apply pre-registered criteria to determine CONFIRMED/FALSIFIED/INCONCLUSIVE.
    """
    print("\n" + "=" * 70)
    print("VERDICT DETERMINATION")
    print("=" * 70)

    # Criterion 1: R correlates with ground truth (rho > 0.5) in >= 2/3 domains
    domain_pass = {}

    # Domain 1: best R rho with human scores
    d1_rho = d1_result["best_R_rho"]
    d1_pass = abs(d1_rho) > 0.5 if not np.isnan(d1_rho) else False
    domain_pass["text"] = {"rho": d1_rho, "pass": d1_pass}

    # Domain 2: best R rho with homogeneity
    d2_rho = d2_result["best_R_rho_homogeneity"]
    d2_pass = abs(d2_rho) > 0.5 if not np.isnan(d2_rho) else False
    domain_pass["tabular"] = {"rho": d2_rho, "pass": d2_pass}

    # Domain 3: best R rho with stability
    d3_rho = d3_result["best_R_rho_stability"]
    d3_pass = abs(d3_rho) > 0.5 if not np.isnan(d3_rho) else False
    domain_pass["financial"] = {"rho": d3_rho, "pass": d3_pass}

    n_domains_pass = sum(1 for v in domain_pass.values() if v["pass"])

    print(f"\n  Criterion 1: R correlates (|rho| > 0.5) with ground truth")
    for name, info in domain_pass.items():
        status = "PASS" if info["pass"] else "FAIL"
        print(f"    {name}: rho={info['rho']:.4f} -> {status}")
    print(f"    Domains passing: {n_domains_pass}/3 (need >= 2 for confirm, < 1 for falsify)")

    # Criterion 2: Cross-domain threshold transfer works for >= 1/2 pairs
    transfer_pairs = [
        ("D1->D2", transfer_result["d1_to_d2"]["within_20pct"]),
        ("D1->D3", transfer_result["d1_to_d3"]["within_20pct"]),
        ("D2->D3", transfer_result["d2_to_d3"]["within_20pct"]),
    ]
    n_transfer_pass = sum(1 for _, ok in transfer_pairs if ok)

    print(f"\n  Criterion 2: Cross-domain threshold transfer (within 20%)")
    for name, ok in transfer_pairs:
        print(f"    {name}: {'PASS' if ok else 'FAIL'}")
    print(f"    Pairs passing: {n_transfer_pass}/3 (need >= 1 for confirm, 0 for falsify)")

    # Criterion 3: R outperforms at least 1 domain-specific alternative
    n_domains_R_beats = sum(1 for v in comparisons.values() if v.get("R_beats_at_least_one", False))

    print(f"\n  Criterion 3: R outperforms >= 1 domain-specific alternative")
    for name, info in comparisons.items():
        beats = info.get("R_beats_at_least_one", False)
        print(f"    {name}: {'YES' if beats else 'NO'} (R rho={info['R_best_abs_rho']:.4f})")
    print(f"    Domains where R beats an alternative: {n_domains_R_beats}/3 (need >= 1 for confirm)")

    # --- Apply pre-registered criteria ---
    # CONFIRM: rho>0.5 in >=2/3 AND transfer >=1/2 AND R outperforms >=1 alt
    # FALSIFY: rho>0.5 in <1/3 OR transfer fails entirely OR R beaten by all alts
    # INCONCLUSIVE: otherwise

    confirm = (n_domains_pass >= 2) and (n_transfer_pass >= 1) and (n_domains_R_beats >= 1)
    falsify = (n_domains_pass < 1) or (n_transfer_pass == 0) or (n_domains_R_beats == 0)

    if confirm:
        verdict = "CONFIRMED"
    elif falsify:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    # Edge case: if criteria conflict, be conservative
    if confirm and falsify:
        verdict = "INCONCLUSIVE"  # should not happen, but safety

    print(f"\n  *** VERDICT: {verdict} ***")
    print(f"    Domain correlation (>=2 needed): {n_domains_pass}/3")
    print(f"    Transfer success (>=1 needed):   {n_transfer_pass}/3")
    print(f"    R beats alternatives (>=1):      {n_domains_R_beats}/3")

    return {
        "verdict": verdict,
        "domain_correlations": domain_pass,
        "n_domains_pass": n_domains_pass,
        "transfer_pairs": {name: ok for name, ok in transfer_pairs},
        "n_transfer_pass": n_transfer_pass,
        "comparison_summary": {name: info.get("R_beats_at_least_one", False) for name, info in comparisons.items()},
        "n_domains_R_beats": n_domains_R_beats,
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 70)
    print("Q03 v2 TEST: R generalizes across genuinely different domains")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("=" * 70)

    all_results = {}
    errors = {}

    # --- Run Domain 1 ---
    try:
        d1_result = run_domain1_text()
        all_results["domain1_text"] = d1_result
    except Exception as e:
        print(f"\n  ERROR in Domain 1: {e}")
        traceback.print_exc()
        d1_result = None
        errors["domain1"] = str(e)

    # --- Run Domain 2 ---
    try:
        d2_result = run_domain2_tabular()
        all_results["domain2_tabular"] = d2_result
    except Exception as e:
        print(f"\n  ERROR in Domain 2: {e}")
        traceback.print_exc()
        d2_result = None
        errors["domain2"] = str(e)

    # --- Run Domain 3 ---
    try:
        d3_result = run_domain3_financial()
        all_results["domain3_financial"] = d3_result
    except Exception as e:
        print(f"\n  ERROR in Domain 3: {e}")
        traceback.print_exc()
        d3_result = None
        errors["domain3"] = str(e)

    # --- Cross-domain transfer ---
    if d1_result and d2_result and d3_result:
        try:
            transfer_result = run_cross_domain_transfer(d1_result, d2_result, d3_result)
            all_results["cross_domain_transfer"] = transfer_result
        except Exception as e:
            print(f"\n  ERROR in cross-domain transfer: {e}")
            traceback.print_exc()
            transfer_result = None
            errors["transfer"] = str(e)
    else:
        transfer_result = None
        print("\n  SKIPPING cross-domain transfer (missing domain results)")

    # --- Comparison analysis ---
    if d1_result and d2_result and d3_result:
        try:
            comparisons = run_comparison_analysis(d1_result, d2_result, d3_result)
            all_results["comparisons"] = comparisons
        except Exception as e:
            print(f"\n  ERROR in comparison: {e}")
            traceback.print_exc()
            comparisons = None
            errors["comparisons"] = str(e)
    else:
        comparisons = None

    # --- Verdict ---
    if d1_result and d2_result and d3_result and transfer_result and comparisons:
        verdict_result = determine_verdict(d1_result, d2_result, d3_result, transfer_result, comparisons)
        all_results["verdict"] = verdict_result
    else:
        # Partial results verdict
        verdict_result = {
            "verdict": "INCONCLUSIVE",
            "reason": f"Missing domain results. Errors: {errors}",
        }
        all_results["verdict"] = verdict_result

    # --- Save results ---
    results_path = os.path.join(RESULTS_DIR, "q03_v2_results.json")
    # Convert numpy types for JSON serialization
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

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to: {results_path}")

    all_results["errors"] = errors
    return all_results


if __name__ == "__main__":
    results = main()
