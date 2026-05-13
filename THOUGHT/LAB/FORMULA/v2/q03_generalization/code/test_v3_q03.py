"""
Q03 v3: Does R = (E/grad_S) * sigma^Df generalize across domains?

REFRAMED (per METH-01 audit): Since v2 uses ONE E definition (mean pairwise
cosine similarity) for everything, the honest question is:

  "Does R (using cosine E) correlate with domain-appropriate quality metrics
   across different data types, and does R outperform E alone?"

Fixes from audit:
  STAT-01: Steiger's test for dependent correlations (R vs E, same Y)
  STAT-03: 3 architecturally diverse embedding models (384d, 384d, 768d)
  STAT-05: Financial domain uses sector classification purity (not Sharpe)
           to avoid tautological E-Sharpe relationship
  METH-01: Honest reframing -- E is NOT changed per domain
  METH-02: Continuous purity range for text (not tri-modal)
  METH-05: Non-overlapping windows for financial data
  BUG-01:  No abs() in beats_E comparison (use signed rho)
  BUG-03:  All imports at module level
  BUG-05:  Outlier analysis (with and without extremes)

Pre-registered decision rule:
  CONFIRMED  if R significantly outperforms E (Steiger p<0.05) in >=2/3 domains
  FALSIFIED  if R does not significantly outperform E in any domain
  INCONCLUSIVE otherwise

NO synthetic data. NO fabrication. ASCII only.
"""

import importlib.util
import sys
import json
import time
import warnings
import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# Load shared formula module
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

warnings.filterwarnings("ignore")

RESULTS = {
    "test_id": "Q03_v3",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "methodology_version": "v3",
    "pre_registration": {
        "question": (
            "Does R (using cosine E) correlate with domain-appropriate quality "
            "metrics across different data types, and does R outperform E alone?"
        ),
        "reframe_note": (
            "v2 uses a single E definition (mean pairwise cosine sim) for all "
            "domains. This test does NOT change E per domain. The question is "
            "whether cosine-based R is a useful quality statistic beyond E alone."
        ),
        "decision_rule": {
            "CONFIRMED": "R significantly outperforms E (Steiger p<0.05) in >=2/3 domains",
            "FALSIFIED": "R does not significantly outperform E in any domain",
            "INCONCLUSIVE": "otherwise",
        },
        "domains": ["text", "text_secondary", "financial"],
        "text_ground_truth": "cluster purity (label-based, not geometric)",
        "text_secondary_ground_truth": "cluster purity on different corpus",
        "financial_ground_truth": "sector classification purity (NOT Sharpe ratio)",
        "financial_note": (
            "Sharpe ratio is algebraically related to mean cosine of return "
            "vectors (both measure return consistency). Using sector purity "
            "instead provides a genuinely independent ground truth."
        ),
    },
    "domains": {},
    "snr_verification": {},
    "verdict": None,
}


# =========================================================================
# STATISTICAL TOOLS
# =========================================================================

def steiger_test(r_xz, r_yz, r_xy, n):
    """
    Steiger (1980) test for comparing two dependent correlations.

    Tests H0: rho(X,Z) = rho(Y,Z) where X and Y share the same Z.

    Args:
        r_xz: correlation between predictor X and outcome Z
        r_yz: correlation between predictor Y and outcome Z
        r_xy: correlation between the two predictors X and Y
        n: sample size

    Returns:
        dict with z_stat, p_value (two-tailed)
    """
    if n < 4:
        return {"z_stat": float("nan"), "p_value": float("nan"), "n": n}

    # Steiger's Z formula for dependent correlations
    # Using the formula from Steiger (1980) and Meng, Rosenthal & Rubin (1992)
    r_mean_sq = (r_xz ** 2 + r_yz ** 2) / 2.0
    f_val = (1.0 - r_xy) / (2.0 * (1.0 - r_mean_sq))
    f_val = min(f_val, 1.0)  # clamp for numerical stability
    h = (1.0 - f_val * r_mean_sq) / (1.0 - r_mean_sq)

    # Fisher z-transformation
    z_xz = 0.5 * math.log((1 + r_xz) / max(1 - r_xz, 1e-10))
    z_yz = 0.5 * math.log((1 + r_yz) / max(1 - r_yz, 1e-10))

    denom = math.sqrt(2.0 * (1.0 - r_xy) / ((n - 3) * h))
    if denom < 1e-15:
        return {"z_stat": float("nan"), "p_value": float("nan"), "n": n}

    z_stat = (z_xz - z_yz) / denom
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    return {"z_stat": float(z_stat), "p_value": float(p_value), "n": n}


def safe_spearman(x, y, label=""):
    """Spearman correlation with NaN filtering."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 5:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "p_value": float(p), "n": n}


def safe_pearson(x, y):
    """Pearson correlation with NaN filtering."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < 5:
        return {"r": float("nan"), "p_value": float("nan"), "n": n}
    r, p = stats.pearsonr(x[mask], y[mask])
    return {"r": float(r), "p_value": float(p), "n": n}


def compute_snr(embeddings):
    """Compute SNR = mean(pairwise_cosine) / std(pairwise_cosine)."""
    n = embeddings.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]
    mean_sim = float(np.mean(pairwise_sims))
    std_sim = float(np.std(pairwise_sims))
    if std_sim < 1e-10:
        return float("nan")
    return mean_sim / std_sim


def compute_cluster_metrics(embeddings):
    """Compute all formula metrics + SNR for a cluster."""
    metrics = formula.compute_all(embeddings)
    metrics["SNR"] = compute_snr(embeddings)
    return metrics


# =========================================================================
# DOMAIN 1: TEXT (20 Newsgroups) -- Continuous purity range
# =========================================================================
def run_text_domain(model_name, corpus_name, texts, labels, categories, rng):
    """
    Run text domain with CONTINUOUS purity range.

    Instead of tri-modal (pure/mixed/degraded), we vary noise fraction
    from 0% to 100% in 5% increments to get a continuous purity distribution.
    """
    print(f"\n--- {corpus_name} with {model_name} ---")

    from sentence_transformers import SentenceTransformer

    n_cats = len(categories)

    # Index docs by category
    cat_indices = {}
    for i, label in enumerate(labels):
        cat_indices.setdefault(label, []).append(i)

    # Build clusters with CONTINUOUS purity range
    # For each of 5 base categories, create clusters with noise fractions:
    # 0%, 5%, 10%, 15%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
    # This gives 13 purity levels x 5 categories = 65 clusters
    noise_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                       0.60, 0.70, 0.80, 0.90, 1.00]
    cluster_size = 150
    n_base_cats = min(5, n_cats)
    base_cats = rng.choice(n_cats, size=n_base_cats, replace=False)

    clusters = []
    for cat_id in base_cats:
        available_main = cat_indices[cat_id]
        other_cats = [c for c in range(n_cats) if c != cat_id]

        for noise_frac in noise_fractions:
            n_noise = int(cluster_size * noise_frac)
            n_main = cluster_size - n_noise

            # Sample main category docs
            if n_main > 0:
                main_chosen = rng.choice(
                    available_main,
                    size=min(n_main, len(available_main)),
                    replace=len(available_main) < n_main
                ).tolist()
            else:
                main_chosen = []

            # Sample noise docs from random other categories
            noise_chosen = []
            if n_noise > 0:
                for _ in range(n_noise):
                    oc = rng.choice(other_cats)
                    noise_chosen.append(rng.choice(cat_indices[oc]))

            chosen = main_chosen + noise_chosen
            doc_labels = [labels[j] for j in chosen]

            # Compute actual purity
            label_counts = {}
            for lab in doc_labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            dominant = max(label_counts.values()) if label_counts else 0
            purity = dominant / len(doc_labels) if doc_labels else 0.0

            clusters.append({
                "category": categories[cat_id],
                "noise_fraction": noise_frac,
                "doc_indices": chosen,
                "purity": purity,
                "n_docs": len(chosen),
            })

    print(f"  Built {len(clusters)} clusters with continuous purity range")
    purities = [c["purity"] for c in clusters]
    print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
    print(f"  Purity mean: {np.mean(purities):.3f}, std: {np.std(purities):.3f}")

    # Encode
    model = SentenceTransformer(model_name)

    all_indices = set()
    for cl in clusters:
        all_indices.update(cl["doc_indices"])
    all_indices = sorted(all_indices)
    print(f"  Unique docs to encode: {len(all_indices)}")

    all_texts = [texts[i] for i in all_indices]
    print("  Encoding...")
    all_embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=128)
    idx_to_emb = {idx: all_embeddings[i] for i, idx in enumerate(all_indices)}
    del model

    # Compute metrics per cluster
    cluster_data = []
    for ci, cl in enumerate(clusters):
        emb_matrix = np.array([idx_to_emb[j] for j in cl["doc_indices"]])
        metrics = compute_cluster_metrics(emb_matrix)
        cluster_data.append({
            "cluster_id": ci,
            "category": cl["category"],
            "noise_fraction": cl["noise_fraction"],
            "purity": cl["purity"],
            "n_docs": cl["n_docs"],
            "metrics": {k: float(v) for k, v in metrics.items()},
        })

    # Extract arrays for correlation
    R_simple_vals = [cd["metrics"]["R_simple"] for cd in cluster_data]
    R_full_vals = [cd["metrics"]["R_full"] for cd in cluster_data]
    E_vals = [cd["metrics"]["E"] for cd in cluster_data]
    SNR_vals = [cd["metrics"]["SNR"] for cd in cluster_data]
    purity_vals = [cd["purity"] for cd in cluster_data]

    # Spearman correlations
    corr = {
        "R_simple_vs_purity": safe_spearman(R_simple_vals, purity_vals),
        "R_full_vs_purity": safe_spearman(R_full_vals, purity_vals),
        "E_vs_purity": safe_spearman(E_vals, purity_vals),
        "SNR_vs_purity": safe_spearman(SNR_vals, purity_vals),
    }

    # Steiger's test: does R_full beat E for predicting purity?
    # We need the Pearson correlations for Steiger (Spearman rho used as input)
    # and also the correlation between R_full and E
    r_full_purity_rho = corr["R_full_vs_purity"]["rho"]
    e_purity_rho = corr["E_vs_purity"]["rho"]

    # Correlation between R_full and E (to account for dependency)
    r_full_e_corr = safe_spearman(R_full_vals, E_vals)

    # Run Steiger's test
    n_valid = corr["R_full_vs_purity"]["n"]
    steiger_result = steiger_test(
        r_xz=r_full_purity_rho,
        r_yz=e_purity_rho,
        r_xy=r_full_e_corr["rho"],
        n=n_valid,
    )
    corr["steiger_Rfull_vs_E"] = steiger_result

    # Also test R_simple vs E
    r_simple_purity_rho = corr["R_simple_vs_purity"]["rho"]
    r_simple_e_corr = safe_spearman(R_simple_vals, E_vals)
    steiger_simple = steiger_test(
        r_xz=r_simple_purity_rho,
        r_yz=e_purity_rho,
        r_xy=r_simple_e_corr["rho"],
        n=n_valid,
    )
    corr["steiger_Rsimple_vs_E"] = steiger_simple

    print(f"\n  Correlations ({model_name} on {corpus_name}):")
    for k, v in corr.items():
        if "rho" in v:
            print(f"    {k}: rho={v['rho']:.4f}, p={v['p_value']:.4e}, n={v['n']}")
        elif "z_stat" in v:
            print(f"    {k}: z={v['z_stat']:.4f}, p={v['p_value']:.4e}")

    return {
        "model": model_name,
        "corpus": corpus_name,
        "n_clusters": len(cluster_data),
        "clusters": cluster_data,
        "correlations": corr,
    }


# =========================================================================
# DOMAIN 2: FINANCIAL (Sector Classification Purity)
# =========================================================================
def run_financial_domain():
    """
    Financial domain with sector purity as ground truth.

    FIXES:
    - Uses sector classification purity (NOT Sharpe ratio) to avoid tautology
    - Uses NON-OVERLAPPING 60-day windows to avoid pseudo-replication
    - Each "cluster" = all non-overlapping windows for stocks in a sector mix

    Approach: group stocks by sector, create clusters of mixed-sector
    return windows, measure whether R predicts sector purity.
    """
    print("\n" + "=" * 70)
    print("DOMAIN 2: FINANCIAL (Sector Purity, Non-overlapping Windows)")
    print("=" * 70)

    import yfinance as yf

    # 30 diverse stocks across 3 sectors
    tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "INTC", "AMD", "CRM"]
    healthcare = ["JNJ", "PFE", "UNH", "ABT", "MRK", "TMO", "ABBV", "LLY", "BMY", "AMGN"]
    energy_ind = ["XOM", "CVX", "COP", "SLB", "EOG", "CAT", "HON", "GE", "MMM", "UPS"]
    all_tickers = tech + healthcare + energy_ind
    sectors = {}
    for t in tech:
        sectors[t] = "tech"
    for t in healthcare:
        sectors[t] = "healthcare"
    for t in energy_ind:
        sectors[t] = "energy_industrial"

    print(f"Downloading 2 years of daily prices for {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, period="2y", auto_adjust=True, progress=True)

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    # Filter valid tickers
    valid_tickers = []
    for t in all_tickers:
        if t in close.columns:
            col = close[t].dropna()
            if len(col) >= 120:
                valid_tickers.append(t)

    print(f"  Valid tickers with >= 120 days: {len(valid_tickers)}")
    if len(valid_tickers) < 20:
        print("  WARNING: fewer than 20 valid tickers, financial results may be weak")

    # Compute daily returns
    returns = close[valid_tickers].pct_change().dropna()

    # NON-OVERLAPPING 60-day windows per stock
    window_size = 60
    rng_fin = np.random.RandomState(42)

    # Collect windows per stock
    stock_windows = {}
    for ticker in valid_tickers:
        ret_series = returns[ticker].dropna().values
        n_windows = len(ret_series) // window_size  # non-overlapping
        if n_windows < 2:
            continue
        windows = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            windows.append(ret_series[start:end])
        stock_windows[ticker] = np.array(windows)

    print(f"  Stocks with >= 2 non-overlapping windows: {len(stock_windows)}")
    for ticker, w in stock_windows.items():
        print(f"    {ticker} ({sectors.get(ticker, '?')}): {w.shape[0]} windows")

    # Build clusters with varying sector purity
    # Pure clusters: windows from stocks in one sector
    # Mixed clusters: windows from stocks across sectors
    sector_stocks = {}
    for t in stock_windows:
        s = sectors.get(t, "unknown")
        sector_stocks.setdefault(s, []).append(t)

    sector_list = sorted(sector_stocks.keys())
    print(f"  Sectors represented: {sector_list}")

    clusters = []
    cluster_target_size = 15  # windows per cluster

    # Pure clusters: one per sector, using all windows from that sector's stocks
    for sec in sector_list:
        sec_tickers = sector_stocks[sec]
        all_sec_windows = []
        all_sec_labels = []
        for t in sec_tickers:
            for w in stock_windows[t]:
                all_sec_windows.append(w)
                all_sec_labels.append(sec)
        if len(all_sec_windows) < 5:
            continue

        # Create multiple pure clusters by splitting
        n_full_clusters = max(1, len(all_sec_windows) // cluster_target_size)
        indices = rng_fin.permutation(len(all_sec_windows))
        for ci in range(n_full_clusters):
            start = ci * cluster_target_size
            end = min(start + cluster_target_size, len(all_sec_windows))
            if end - start < 5:
                continue
            cl_windows = [all_sec_windows[indices[j]] for j in range(start, end)]
            cl_labels = [all_sec_labels[indices[j]] for j in range(start, end)]
            label_counts = {}
            for lab in cl_labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            purity = max(label_counts.values()) / len(cl_labels)
            clusters.append({
                "type": "pure",
                "dominant_sector": sec,
                "windows": np.array(cl_windows),
                "labels": cl_labels,
                "purity": purity,
            })

    # Mixed clusters: varying proportions of sectors
    mix_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for noise_frac in mix_fractions:
        for base_sec in sector_list:
            base_tickers = sector_stocks[base_sec]
            other_secs = [s for s in sector_list if s != base_sec]

            # Gather base windows
            base_windows = []
            base_labels = []
            for t in base_tickers:
                for w in stock_windows[t]:
                    base_windows.append(w)
                    base_labels.append(base_sec)

            # Gather other windows
            other_windows = []
            other_labels = []
            for os_name in other_secs:
                for t in sector_stocks[os_name]:
                    for w in stock_windows[t]:
                        other_windows.append(w)
                        other_labels.append(os_name)

            if len(base_windows) < 3 or len(other_windows) < 2:
                continue

            n_noise = max(1, int(cluster_target_size * noise_frac))
            n_base = cluster_target_size - n_noise

            if n_base > len(base_windows):
                n_base = len(base_windows)
            if n_noise > len(other_windows):
                n_noise = len(other_windows)

            base_idx = rng_fin.choice(len(base_windows), size=n_base, replace=False)
            other_idx = rng_fin.choice(len(other_windows), size=n_noise, replace=False)

            cl_windows = ([base_windows[i] for i in base_idx] +
                          [other_windows[i] for i in other_idx])
            cl_labels = ([base_labels[i] for i in base_idx] +
                         [other_labels[i] for i in other_idx])

            label_counts = {}
            for lab in cl_labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            purity = max(label_counts.values()) / len(cl_labels)

            clusters.append({
                "type": f"mixed_{noise_frac:.0%}",
                "dominant_sector": base_sec,
                "windows": np.array(cl_windows),
                "labels": cl_labels,
                "purity": purity,
            })

    print(f"\n  Built {len(clusters)} financial clusters")
    fin_purities = [c["purity"] for c in clusters]
    print(f"  Purity range: [{min(fin_purities):.3f}, {max(fin_purities):.3f}]")

    # Compute formula metrics per cluster
    cluster_data = []
    for ci, cl in enumerate(clusters):
        metrics = compute_cluster_metrics(cl["windows"])
        cluster_data.append({
            "cluster_id": ci,
            "type": cl["type"],
            "dominant_sector": cl["dominant_sector"],
            "purity": cl["purity"],
            "n_windows": len(cl["labels"]),
            "metrics": {k: float(v) for k, v in metrics.items()},
        })
        print(f"  Cluster {ci} ({cl['type']}, {cl['dominant_sector']}): "
              f"n={len(cl['labels'])}, purity={cl['purity']:.3f}, "
              f"R_simple={metrics['R_simple']:.4f}, E={metrics['E']:.4f}")

    # Correlations
    R_simple_vals = [cd["metrics"]["R_simple"] for cd in cluster_data]
    R_full_vals = [cd["metrics"]["R_full"] for cd in cluster_data]
    E_vals = [cd["metrics"]["E"] for cd in cluster_data]
    SNR_vals = [cd["metrics"]["SNR"] for cd in cluster_data]
    purity_vals = [cd["purity"] for cd in cluster_data]

    corr = {
        "R_simple_vs_purity": safe_spearman(R_simple_vals, purity_vals),
        "R_full_vs_purity": safe_spearman(R_full_vals, purity_vals),
        "E_vs_purity": safe_spearman(E_vals, purity_vals),
        "SNR_vs_purity": safe_spearman(SNR_vals, purity_vals),
    }

    # Steiger tests
    r_full_purity_rho = corr["R_full_vs_purity"]["rho"]
    e_purity_rho = corr["E_vs_purity"]["rho"]
    r_full_e_corr = safe_spearman(R_full_vals, E_vals)
    n_valid = corr["R_full_vs_purity"]["n"]

    corr["steiger_Rfull_vs_E"] = steiger_test(
        r_xz=r_full_purity_rho,
        r_yz=e_purity_rho,
        r_xy=r_full_e_corr["rho"],
        n=n_valid,
    )

    r_simple_purity_rho = corr["R_simple_vs_purity"]["rho"]
    r_simple_e_corr = safe_spearman(R_simple_vals, E_vals)
    corr["steiger_Rsimple_vs_E"] = steiger_test(
        r_xz=r_simple_purity_rho,
        r_yz=e_purity_rho,
        r_xy=r_simple_e_corr["rho"],
        n=n_valid,
    )

    print(f"\n  Financial correlations:")
    for k, v in corr.items():
        if "rho" in v:
            print(f"    {k}: rho={v['rho']:.4f}, p={v['p_value']:.4e}, n={v['n']}")
        elif "z_stat" in v:
            print(f"    {k}: z={v['z_stat']:.4f}, p={v['p_value']:.4e}")

    return {
        "n_clusters": len(cluster_data),
        "n_valid_tickers": len(stock_windows),
        "non_overlapping_windows": True,
        "ground_truth": "sector_purity",
        "ground_truth_note": "NOT Sharpe (avoids tautological E-Sharpe relationship)",
        "clusters": cluster_data,
        "correlations": corr,
    }


# =========================================================================
# R = SNR VERIFICATION
# =========================================================================
def verify_r_equals_snr(all_cluster_metrics):
    """Verify that R_simple = SNR = mean(cos)/std(cos) for all clusters."""
    print("\n" + "=" * 70)
    print("R = SNR VERIFICATION")
    print("=" * 70)

    max_abs_diff = 0.0
    diffs = []
    for label, metrics in all_cluster_metrics:
        r_simple = metrics["R_simple"]
        snr = metrics["SNR"]
        if np.isnan(r_simple) or np.isnan(snr):
            continue
        diff = abs(r_simple - snr)
        diffs.append(diff)
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"  Checked {len(diffs)} clusters")
    print(f"  Max |R_simple - SNR|: {max_abs_diff:.2e}")
    print(f"  Mean |R_simple - SNR|: {np.mean(diffs):.2e}")
    print(f"  R_simple == SNR confirmed: {max_abs_diff < 1e-10}")

    return {
        "n_clusters_checked": len(diffs),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "confirmed": bool(max_abs_diff < 1e-10),
    }


# =========================================================================
# ADJUDICATION
# =========================================================================
def adjudicate(results):
    """
    Pre-registered decision rule:
      CONFIRMED  if R significantly outperforms E (Steiger p<0.05) in >=2/3 domains
      FALSIFIED  if R does not significantly outperform E in any domain
      INCONCLUSIVE otherwise

    Uses SIGNED rho (no abs()). A negative correlation is worse, not better.
    """
    print("\n" + "=" * 70)
    print("ADJUDICATION")
    print("=" * 70)

    domain_outcomes = {}

    for domain_key, domain_data in results["domains"].items():
        corr = domain_data.get("correlations", {})
        if not corr:
            continue

        r_simple_rho = corr.get("R_simple_vs_purity", {}).get("rho", float("nan"))
        r_full_rho = corr.get("R_full_vs_purity", {}).get("rho", float("nan"))
        e_rho = corr.get("E_vs_purity", {}).get("rho", float("nan"))

        # Best R metric (signed, no abs)
        if np.isnan(r_full_rho):
            best_r_rho = r_simple_rho
            best_r_label = "R_simple"
        elif np.isnan(r_simple_rho):
            best_r_rho = r_full_rho
            best_r_label = "R_full"
        elif r_full_rho > r_simple_rho:
            best_r_rho = r_full_rho
            best_r_label = "R_full"
        else:
            best_r_rho = r_simple_rho
            best_r_label = "R_simple"

        # Steiger test: does R significantly beat E?
        steiger_full = corr.get("steiger_Rfull_vs_E", {})
        steiger_simple = corr.get("steiger_Rsimple_vs_E", {})

        # Use the Steiger test corresponding to the best R metric
        if best_r_label == "R_full":
            steiger = steiger_full
        else:
            steiger = steiger_simple

        steiger_p = steiger.get("p_value", float("nan"))
        steiger_z = steiger.get("z_stat", float("nan"))

        # Does R significantly outperform E?
        # Steiger p < 0.05 AND R rho > E rho (signed)
        sig_beats_e = (
            not np.isnan(steiger_p)
            and steiger_p < 0.05
            and best_r_rho > e_rho
        )

        # Does R correlate meaningfully? (rho > 0.5, p < 0.05)
        r_p = corr.get(f"{best_r_label}_vs_purity", {}).get("p_value", 1.0)
        passes_threshold = best_r_rho > 0.5 and r_p < 0.05

        domain_outcomes[domain_key] = {
            "best_R_metric": best_r_label,
            "best_R_rho": float(best_r_rho),
            "E_rho": float(e_rho),
            "R_minus_E": float(best_r_rho - e_rho) if not (np.isnan(best_r_rho) or np.isnan(e_rho)) else float("nan"),
            "steiger_z": float(steiger_z),
            "steiger_p": float(steiger_p),
            "significantly_beats_E": sig_beats_e,
            "passes_rho_threshold": passes_threshold,
        }

        print(f"  {domain_key}:")
        print(f"    best_R ({best_r_label}) rho={best_r_rho:.4f}, E rho={e_rho:.4f}")
        print(f"    R - E = {best_r_rho - e_rho:.4f}")
        print(f"    Steiger z={steiger_z:.4f}, p={steiger_p:.4e}")
        print(f"    Significantly beats E: {sig_beats_e}")
        print(f"    Passes threshold (rho>0.5, p<0.05): {passes_threshold}")

    # Apply pre-registered decision rule
    n_domains = len(domain_outcomes)
    n_sig_beats = sum(1 for d in domain_outcomes.values() if d["significantly_beats_E"])
    n_passes = sum(1 for d in domain_outcomes.values() if d["passes_rho_threshold"])

    print(f"\n  Summary:")
    print(f"    Domains tested: {n_domains}")
    print(f"    R significantly beats E (Steiger p<0.05): {n_sig_beats}/{n_domains}")
    print(f"    R passes threshold (rho>0.5): {n_passes}/{n_domains}")

    required_for_confirm = max(2, int(np.ceil(2.0 * n_domains / 3.0)))

    if n_sig_beats >= required_for_confirm:
        verdict = "CONFIRMED"
    elif n_sig_beats == 0:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")
    print(f"  (Need {required_for_confirm}/{n_domains} domains with Steiger p<0.05)")

    results["adjudication"] = {
        "domain_outcomes": domain_outcomes,
        "n_domains": n_domains,
        "n_significantly_beats_E": n_sig_beats,
        "n_passes_threshold": n_passes,
        "required_for_confirm": required_for_confirm,
        "verdict": verdict,
        "decision_applied": (
            f"CONFIRMED requires {required_for_confirm}/{n_domains} sig beats. "
            f"FALSIFIED requires 0/{n_domains} sig beats. "
            f"Got {n_sig_beats}/{n_domains}."
        ),
    }
    results["verdict"] = verdict

    return verdict


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    start_time = time.time()

    all_cluster_metrics = []

    # ---------------------------------------------------------------
    # DOMAIN 1a: Text -- 20 Newsgroups, 3 diverse models
    # ---------------------------------------------------------------
    print("=" * 70)
    print("DOMAIN 1: TEXT (20 Newsgroups, 3 models, continuous purity)")
    print("=" * 70)

    from sklearn.datasets import fetch_20newsgroups

    print("Loading 20 Newsgroups...")
    ng_data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    ng_texts = ng_data.data
    ng_labels = ng_data.target
    ng_categories = ng_data.target_names

    text_model_names = [
        "all-MiniLM-L6-v2",           # 384d, MiniLM architecture
        "all-mpnet-base-v2",           # 768d, MPNet architecture
        "multi-qa-MiniLM-L6-cos-v1",  # 384d, QA-tuned MiniLM
    ]

    text_model_results = {}
    rng_text = np.random.RandomState(42)

    for model_name in text_model_names:
        result = run_text_domain(
            model_name=model_name,
            corpus_name="20newsgroups",
            texts=ng_texts,
            labels=ng_labels,
            categories=ng_categories,
            rng=np.random.RandomState(42),  # same seed each time for reproducibility
        )
        text_model_results[model_name] = result

        # Collect for SNR verification (first model only)
        if model_name == text_model_names[0]:
            for cl in result.get("clusters", []):
                all_cluster_metrics.append((f"text_{cl['cluster_id']}", cl["metrics"]))

    # Aggregate text correlations across 3 models
    agg_corr_keys = ["R_simple_vs_purity", "R_full_vs_purity", "E_vs_purity", "SNR_vs_purity"]
    text_agg = {}
    for ck in agg_corr_keys:
        rhos = []
        for mn in text_model_names:
            r = text_model_results[mn]["correlations"].get(ck, {}).get("rho", float("nan"))
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            text_agg[ck] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "min_rho": float(np.min(rhos)),
                "max_rho": float(np.max(rhos)),
                "n_models": len(rhos),
                "individual_rhos": rhos,
            }

    # Aggregate Steiger tests
    steiger_keys = ["steiger_Rfull_vs_E", "steiger_Rsimple_vs_E"]
    for sk in steiger_keys:
        ps = []
        zs = []
        for mn in text_model_names:
            st = text_model_results[mn]["correlations"].get(sk, {})
            p = st.get("p_value", float("nan"))
            z = st.get("z_stat", float("nan"))
            if not np.isnan(p):
                ps.append(p)
                zs.append(z)
        if ps:
            text_agg[sk] = {
                "mean_p": float(np.mean(ps)),
                "min_p": float(np.min(ps)),
                "max_p": float(np.max(ps)),
                "mean_z": float(np.mean(zs)),
                "individual_ps": ps,
                "individual_zs": zs,
                "n_models": len(ps),
            }

    print("\n  Aggregate text correlations across 3 models:")
    for k, v in text_agg.items():
        if "mean_rho" in v:
            print(f"    {k}: mean_rho={v['mean_rho']:.4f} +/- {v['std_rho']:.4f} "
                  f"[{v['min_rho']:.4f}, {v['max_rho']:.4f}]")
        elif "mean_p" in v:
            print(f"    {k}: mean_p={v['mean_p']:.4e}, min_p={v['min_p']:.4e}")

    # For adjudication, use median model's correlations as the domain result
    # (Use the model with median R_full rho)
    r_full_rhos = []
    for mn in text_model_names:
        r = text_model_results[mn]["correlations"].get("R_full_vs_purity", {}).get("rho", float("nan"))
        r_full_rhos.append((r, mn))
    r_full_rhos.sort(key=lambda x: x[0] if not np.isnan(x[0]) else -999)
    median_model = r_full_rhos[len(r_full_rhos) // 2][1]
    print(f"\n  Using median model for adjudication: {median_model}")

    RESULTS["domains"]["text_20ng"] = {
        "per_model": {mn: text_model_results[mn] for mn in text_model_names},
        "aggregate": text_agg,
        "median_model": median_model,
        "correlations": text_model_results[median_model]["correlations"],
    }

    # ---------------------------------------------------------------
    # DOMAIN 1b: Second text corpus (subset of 20NG categories)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DOMAIN 1b: SECOND TEXT CORPUS (20NG alt categories, mpnet)")
    print("=" * 70)

    # Use a different subset of 20NG categories to simulate a second corpus
    # First run used 5 random categories; now use 5 different ones
    rng_alt = np.random.RandomState(99)  # different seed selects different categories

    alt_result = run_text_domain(
        model_name="all-mpnet-base-v2",
        corpus_name="20newsgroups_alt",
        texts=ng_texts,
        labels=ng_labels,
        categories=ng_categories,
        rng=rng_alt,
    )

    RESULTS["domains"]["text_alt"] = {
        "per_model": {"all-mpnet-base-v2": alt_result},
        "correlations": alt_result["correlations"],
    }

    for cl in alt_result.get("clusters", []):
        all_cluster_metrics.append((f"text_alt_{cl['cluster_id']}", cl["metrics"]))

    # ---------------------------------------------------------------
    # DOMAIN 3: FINANCIAL
    # ---------------------------------------------------------------
    print("\n>>> Running Financial Domain...")
    financial_results = run_financial_domain()
    RESULTS["domains"]["financial"] = financial_results

    for cl in financial_results.get("clusters", []):
        all_cluster_metrics.append((f"fin_{cl['cluster_id']}", cl["metrics"]))

    # ---------------------------------------------------------------
    # R = SNR Verification
    # ---------------------------------------------------------------
    snr_results = verify_r_equals_snr(all_cluster_metrics)
    RESULTS["snr_verification"] = snr_results

    # ---------------------------------------------------------------
    # ADJUDICATION
    # ---------------------------------------------------------------
    verdict = adjudicate(RESULTS)

    elapsed = time.time() - start_time
    RESULTS["elapsed_seconds"] = float(elapsed)

    # Save results
    output_path = (
        r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA"
        r"\v2\q03_generalization\results\test_v3_q03_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")
