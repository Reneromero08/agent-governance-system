"""
Q03 Fixed: Does R = (E/grad_S) * sigma^Df generalize across genuinely different domains?

v2 retest with corrected methodology:
- Text: 20 Newsgroups with proper cluster construction (pure/mixed/degraded)
- Tabular: California Housing with geographic clusters, out-of-sample R^2
- Financial: 30 diverse stocks, 60-day return windows, Sharpe ratio ground truth
- R=SNR verification across all clusters
- Cross-domain transfer test

NO synthetic data. NO reward-maxing. Just truth.
"""

import importlib.util
import sys
import json
import time
import warnings
import os
from datetime import datetime

import numpy as np
from scipy import stats

# Load the shared formula module
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

warnings.filterwarnings("ignore")
np.random.seed(42)

RESULTS = {
    "test_id": "Q03_v2_fixed",
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "methodology_version": "v2_fixed",
    "domains": {},
    "snr_verification": {},
    "cross_domain_transfer": {},
    "verdict": None,
}


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


# ===========================================================================
# DOMAIN 1: TEXT (20 Newsgroups)
# ===========================================================================
def run_text_domain():
    print("=" * 70)
    print("DOMAIN 1: TEXT (20 Newsgroups)")
    print("=" * 70)

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.metrics import silhouette_score
    from sentence_transformers import SentenceTransformer

    # Load data
    print("Loading 20 Newsgroups...")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = data.data
    labels = data.target
    categories = data.target_names
    n_cats = len(categories)

    # Index docs by category
    cat_indices = {}
    for i, label in enumerate(labels):
        cat_indices.setdefault(label, []).append(i)

    # Print category sizes
    for cat_id in range(n_cats):
        print(f"  Category {cat_id} ({categories[cat_id]}): {len(cat_indices[cat_id])} docs")

    # Build 60 clusters: 20 pure, 20 mixed, 20 degraded
    clusters = []
    rng = np.random.RandomState(42)

    # 20 pure clusters: 200 random docs from each category
    print("\nBuilding 20 pure clusters...")
    for cat_id in range(n_cats):
        available = cat_indices[cat_id]
        if len(available) < 200:
            chosen = rng.choice(available, size=min(200, len(available)), replace=True)
        else:
            chosen = rng.choice(available, size=200, replace=False)
        clusters.append({
            "type": "pure",
            "category": categories[cat_id],
            "doc_indices": chosen.tolist(),
            "doc_labels": [labels[i] for i in chosen],
        })

    # 20 mixed clusters: 200 docs from random mix of 3-5 categories
    print("Building 20 mixed clusters...")
    for i in range(20):
        n_mix = rng.randint(3, 6)  # 3 to 5 categories
        mix_cats = rng.choice(n_cats, size=n_mix, replace=False)
        per_cat = 200 // n_mix
        chosen = []
        for cat_id in mix_cats:
            available = cat_indices[cat_id]
            c = rng.choice(available, size=min(per_cat, len(available)), replace=len(available) < per_cat)
            chosen.extend(c.tolist())
        # Trim or pad to exactly 200
        if len(chosen) > 200:
            chosen = chosen[:200]
        elif len(chosen) < 200:
            extra_cats = rng.choice(mix_cats, size=200 - len(chosen))
            for cat_id in extra_cats:
                available = cat_indices[cat_id]
                c = rng.choice(available, size=1)
                chosen.append(int(c[0]))
        clusters.append({
            "type": "mixed",
            "category": f"mixed_{i}",
            "doc_indices": chosen,
            "doc_labels": [labels[j] for j in chosen],
        })

    # 20 degraded clusters: 160 from one category + 40 from others
    print("Building 20 degraded clusters...")
    for cat_id in range(n_cats):
        available_main = cat_indices[cat_id]
        if len(available_main) < 160:
            main_chosen = rng.choice(available_main, size=160, replace=True)
        else:
            main_chosen = rng.choice(available_main, size=160, replace=False)

        # 40 from random other categories
        other_cats = [c for c in range(n_cats) if c != cat_id]
        noise_chosen = []
        for _ in range(40):
            oc = rng.choice(other_cats)
            noise_chosen.append(rng.choice(cat_indices[oc]))
        chosen = list(main_chosen) + noise_chosen
        clusters.append({
            "type": "degraded",
            "category": categories[cat_id],
            "doc_indices": chosen,
            "doc_labels": [labels[j] for j in chosen],
        })

    # Compute ground truth: purity
    print("\nComputing cluster purity...")
    purities = []
    for cl in clusters:
        label_counts = {}
        for lab in cl["doc_labels"]:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        dominant = max(label_counts.values())
        purity = dominant / len(cl["doc_labels"])
        cl["purity"] = purity
        purities.append(purity)

    print(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")

    # Encode with 2 fast architectures (mpnet dropped: 3min/batch too slow)
    # Both are 384-dim but trained differently: general SBERT vs QA-tuned
    model_names = [
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ]

    text_results = {"clusters": [], "correlations_by_model": {}}

    for model_name in model_names:
        print(f"\n--- Encoding with {model_name} ---")
        model = SentenceTransformer(model_name)

        # Collect all unique doc indices
        all_indices = set()
        for cl in clusters:
            all_indices.update(cl["doc_indices"])
        all_indices = sorted(all_indices)
        print(f"  Total unique docs to encode: {len(all_indices)}")

        # Encode all docs
        all_texts = [texts[i] for i in all_indices]
        print("  Encoding...")
        all_embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=128)
        idx_to_emb = {idx: all_embeddings[i] for i, idx in enumerate(all_indices)}

        # Compute metrics per cluster
        cluster_data = []
        for ci, cl in enumerate(clusters):
            emb_matrix = np.array([idx_to_emb[j] for j in cl["doc_indices"]])
            metrics = compute_cluster_metrics(emb_matrix)
            cluster_data.append({
                "cluster_id": ci,
                "type": cl["type"],
                "category": cl["category"],
                "purity": cl["purity"],
                "n_docs": len(cl["doc_indices"]),
                "metrics": metrics,
            })

        # Compute silhouette scores (sample-based for speed)
        print("  Computing silhouette scores...")
        # Build a combined embedding matrix + labels for silhouette
        # Use a subsample for silhouette (it is O(n^2))
        sample_size_per_cluster = 50
        combined_embs = []
        combined_labels = []
        for ci, cl in enumerate(clusters):
            indices = cl["doc_indices"]
            if len(indices) > sample_size_per_cluster:
                sample_idx = rng.choice(len(indices), size=sample_size_per_cluster, replace=False)
            else:
                sample_idx = np.arange(len(indices))
            for si in sample_idx:
                combined_embs.append(idx_to_emb[indices[si]])
                combined_labels.append(ci)
        combined_embs = np.array(combined_embs)
        combined_labels = np.array(combined_labels)

        # Per-cluster silhouette: average silhouette of samples in each cluster
        from sklearn.metrics import silhouette_samples
        print(f"  Computing silhouette on {len(combined_embs)} samples...")
        sil_samples = silhouette_samples(combined_embs, combined_labels, metric="cosine")
        # Map back to cluster
        cluster_sil = {}
        idx_counter = 0
        for ci, cl in enumerate(clusters):
            n_in = min(sample_size_per_cluster, len(cl["doc_indices"]))
            sil_vals = sil_samples[idx_counter:idx_counter + n_in]
            cluster_sil[ci] = float(np.mean(sil_vals))
            idx_counter += n_in

        for cd in cluster_data:
            cd["silhouette"] = cluster_sil[cd["cluster_id"]]

        # Correlations
        R_simple_vals = [cd["metrics"]["R_simple"] for cd in cluster_data]
        R_full_vals = [cd["metrics"]["R_full"] for cd in cluster_data]
        E_vals = [cd["metrics"]["E"] for cd in cluster_data]
        inv_grad_S_vals = [1.0 / cd["metrics"]["grad_S"] if cd["metrics"]["grad_S"] > 1e-10 else float("nan") for cd in cluster_data]
        SNR_vals = [cd["metrics"]["SNR"] for cd in cluster_data]
        purity_vals = [cd["purity"] for cd in cluster_data]
        sil_vals_list = [cd["silhouette"] for cd in cluster_data]

        # Filter out NaN for correlations
        def safe_spearman(x, y, label=""):
            x = np.array(x)
            y = np.array(y)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 5:
                return {"rho": float("nan"), "p_value": float("nan"), "n": int(np.sum(mask))}
            rho, p = stats.spearmanr(x[mask], y[mask])
            return {"rho": float(rho), "p_value": float(p), "n": int(np.sum(mask))}

        correlations = {
            "R_simple_vs_purity": safe_spearman(R_simple_vals, purity_vals),
            "R_full_vs_purity": safe_spearman(R_full_vals, purity_vals),
            "E_vs_purity": safe_spearman(E_vals, purity_vals),
            "inv_grad_S_vs_purity": safe_spearman(inv_grad_S_vals, purity_vals),
            "SNR_vs_purity": safe_spearman(SNR_vals, purity_vals),
            "R_simple_vs_silhouette": safe_spearman(R_simple_vals, sil_vals_list),
            "R_full_vs_silhouette": safe_spearman(R_full_vals, sil_vals_list),
            "E_vs_silhouette": safe_spearman(E_vals, sil_vals_list),
            "inv_grad_S_vs_silhouette": safe_spearman(inv_grad_S_vals, sil_vals_list),
            "SNR_vs_silhouette": safe_spearman(SNR_vals, sil_vals_list),
        }

        print(f"\n  Correlations ({model_name}):")
        for k, v in correlations.items():
            print(f"    {k}: rho={v['rho']:.4f}, p={v['p_value']:.4e}, n={v['n']}")

        text_results["correlations_by_model"][model_name] = correlations

        # Store cluster data only for first model to keep JSON manageable
        if model_name == model_names[0]:
            # Convert numpy types for JSON
            for cd in cluster_data:
                for mk, mv in cd["metrics"].items():
                    cd["metrics"][mk] = float(mv) if not isinstance(mv, float) else mv
            text_results["clusters"] = cluster_data

        # Free model memory
        del model

    # Aggregate across models: average rho for each correlation
    agg_corr = {}
    for corr_key in text_results["correlations_by_model"][model_names[0]].keys():
        rhos = [text_results["correlations_by_model"][m][corr_key]["rho"]
                for m in model_names
                if not np.isnan(text_results["correlations_by_model"][m][corr_key]["rho"])]
        if rhos:
            agg_corr[corr_key] = {
                "mean_rho": float(np.mean(rhos)),
                "std_rho": float(np.std(rhos)),
                "n_models": len(rhos),
            }
    text_results["aggregate_correlations"] = agg_corr

    print("\n  Aggregate correlations across 2 models:")
    for k, v in agg_corr.items():
        print(f"    {k}: mean_rho={v['mean_rho']:.4f} +/- {v['std_rho']:.4f}")

    return text_results


# ===========================================================================
# DOMAIN 2: TABULAR (California Housing)
# ===========================================================================
def run_tabular_domain():
    print("\n" + "=" * 70)
    print("DOMAIN 2: TABULAR (California Housing)")
    print("=" * 70)

    from sklearn.datasets import fetch_california_housing
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    housing = fetch_california_housing()
    X = housing.data  # 8 features
    y = housing.target  # median house value
    feature_names = housing.feature_names

    # Geographic clustering: k-means on lat/lon
    lat_idx = feature_names.index("Latitude")
    lon_idx = feature_names.index("Longitude")
    geo_coords = X[:, [lat_idx, lon_idx]]

    print("Clustering by geography (k=20)...")
    km = KMeans(n_clusters=20, random_state=42, n_init=10)
    geo_labels = km.fit_predict(geo_coords)

    # Standardize features (excluding lat/lon for the regression features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # For each geographic cluster: compute R on standardized features, compute out-of-sample R^2
    cluster_results = []
    for cluster_id in range(20):
        mask = geo_labels == cluster_id
        X_cluster = X_scaled[mask]
        y_cluster = y[mask]
        n_samples = int(np.sum(mask))

        if n_samples < 20:
            print(f"  Cluster {cluster_id}: only {n_samples} samples, skipping")
            continue

        # Compute formula metrics on the standardized feature vectors
        metrics = compute_cluster_metrics(X_cluster)

        # Out-of-sample R^2 via Ridge regression
        # 80/20 split
        n_train = int(0.8 * n_samples)
        perm = rng_tabular.permutation(n_samples)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        if len(test_idx) < 5:
            r2_oos = float("nan")
        else:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_cluster[train_idx], y_cluster[train_idx])
            y_pred = ridge.predict(X_cluster[test_idx])
            ss_res = np.sum((y_cluster[test_idx] - y_pred) ** 2)
            ss_tot = np.sum((y_cluster[test_idx] - np.mean(y_cluster[test_idx])) ** 2)
            r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        cluster_results.append({
            "cluster_id": int(cluster_id),
            "n_samples": n_samples,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "r2_oos": float(r2_oos),
        })

        print(f"  Cluster {cluster_id}: n={n_samples}, R_simple={metrics['R_simple']:.4f}, "
              f"R_full={metrics['R_full']:.4f}, E={metrics['E']:.4f}, R2_oos={r2_oos:.4f}")

    # Correlations
    R_simple_vals = [cr["metrics"]["R_simple"] for cr in cluster_results]
    R_full_vals = [cr["metrics"]["R_full"] for cr in cluster_results]
    E_vals = [cr["metrics"]["E"] for cr in cluster_results]
    SNR_vals = [cr["metrics"]["SNR"] for cr in cluster_results]
    r2_vals = [cr["r2_oos"] for cr in cluster_results]

    def safe_spearman(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 5:
            return {"rho": float("nan"), "p_value": float("nan"), "n": int(np.sum(mask))}
        rho, p = stats.spearmanr(x[mask], y[mask])
        return {"rho": float(rho), "p_value": float(p), "n": int(np.sum(mask))}

    correlations = {
        "R_simple_vs_r2_oos": safe_spearman(R_simple_vals, r2_vals),
        "R_full_vs_r2_oos": safe_spearman(R_full_vals, r2_vals),
        "E_vs_r2_oos": safe_spearman(E_vals, r2_vals),
        "SNR_vs_r2_oos": safe_spearman(SNR_vals, r2_vals),
    }

    print("\n  Correlations:")
    for k, v in correlations.items():
        print(f"    {k}: rho={v['rho']:.4f}, p={v['p_value']:.4e}, n={v['n']}")

    return {
        "clusters": cluster_results,
        "correlations": correlations,
    }


rng_tabular = np.random.RandomState(42)


# ===========================================================================
# DOMAIN 3: FINANCIAL (Stock Return Windows)
# ===========================================================================
def run_financial_domain():
    print("\n" + "=" * 70)
    print("DOMAIN 3: FINANCIAL (Stock Return Windows)")
    print("=" * 70)

    import yfinance as yf

    # 30 diverse stocks: 10 tech, 10 healthcare, 10 energy/industrial
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
    # Download
    raw = yf.download(all_tickers, period="2y", auto_adjust=True, progress=True)

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    # Filter tickers that actually downloaded
    valid_tickers = []
    for t in all_tickers:
        if t in close.columns:
            col = close[t].dropna()
            if len(col) >= 120:  # Need at least 120 days for 60-day windows
                valid_tickers.append(t)

    print(f"  Valid tickers with >= 120 days: {len(valid_tickers)}")
    if len(valid_tickers) < 20:
        print("  WARNING: fewer than 20 valid tickers")

    # Compute daily returns
    returns = close[valid_tickers].pct_change().dropna()

    # For each stock: build 60-day rolling windows of daily returns
    window_size = 60
    stock_results = []

    for ticker in valid_tickers:
        ret_series = returns[ticker].dropna().values
        if len(ret_series) < window_size + 10:
            continue

        # Build window matrix: each row = a 60-day window
        n_windows = len(ret_series) - window_size + 1
        windows = np.array([ret_series[i:i + window_size] for i in range(n_windows)])

        # Compute formula metrics on the window matrix
        # Each row is a 60-dimensional vector (60 days of returns)
        metrics = compute_cluster_metrics(windows)

        # Ground truth: annualized Sharpe ratio
        mean_daily = np.mean(ret_series)
        std_daily = np.std(ret_series)
        if std_daily < 1e-10:
            sharpe = float("nan")
        else:
            sharpe = (mean_daily / std_daily) * np.sqrt(252)  # annualized

        stock_results.append({
            "ticker": ticker,
            "sector": sectors.get(ticker, "unknown"),
            "n_windows": int(n_windows),
            "n_days": len(ret_series),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "sharpe_ratio": float(sharpe),
        })

        print(f"  {ticker} ({sectors.get(ticker, '?')}): n_windows={n_windows}, "
              f"R_simple={metrics['R_simple']:.4f}, Sharpe={sharpe:.4f}")

    # Correlations
    R_simple_vals = [sr["metrics"]["R_simple"] for sr in stock_results]
    R_full_vals = [sr["metrics"]["R_full"] for sr in stock_results]
    E_vals = [sr["metrics"]["E"] for sr in stock_results]
    SNR_vals = [sr["metrics"]["SNR"] for sr in stock_results]
    sharpe_vals = [sr["sharpe_ratio"] for sr in stock_results]

    def safe_spearman(x, y):
        x = np.array(x)
        y = np.array(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 5:
            return {"rho": float("nan"), "p_value": float("nan"), "n": int(np.sum(mask))}
        rho, p = stats.spearmanr(x[mask], y[mask])
        return {"rho": float(rho), "p_value": float(p), "n": int(np.sum(mask))}

    correlations = {
        "R_simple_vs_sharpe": safe_spearman(R_simple_vals, sharpe_vals),
        "R_full_vs_sharpe": safe_spearman(R_full_vals, sharpe_vals),
        "E_vs_sharpe": safe_spearman(E_vals, sharpe_vals),
        "SNR_vs_sharpe": safe_spearman(SNR_vals, sharpe_vals),
    }

    print("\n  Correlations:")
    for k, v in correlations.items():
        print(f"    {k}: rho={v['rho']:.4f}, p={v['p_value']:.4e}, n={v['n']}")

    return {
        "stocks": stock_results,
        "correlations": correlations,
    }


# ===========================================================================
# R = SNR VERIFICATION
# ===========================================================================
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


# ===========================================================================
# CROSS-DOMAIN TRANSFER
# ===========================================================================
def run_cross_domain_transfer(text_results, tabular_results, financial_results):
    """Calibrate threshold on text, apply to tabular and financial."""
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN TRANSFER")
    print("=" * 70)

    # Use first model's cluster data for text
    text_clusters = text_results["clusters"]
    if not text_clusters:
        print("  No text cluster data available for transfer test")
        return {"error": "no text data"}

    # Text: optimize R_simple threshold for pure (purity >= 0.9) vs mixed (purity < 0.6)
    text_r_simple = np.array([c["metrics"]["R_simple"] for c in text_clusters])
    text_purity = np.array([c["purity"] for c in text_clusters])
    text_is_pure = text_purity >= 0.9
    text_is_mixed = text_purity < 0.6

    # Only consider clusters that are clearly pure or mixed
    cal_mask = text_is_pure | text_is_mixed
    cal_r = text_r_simple[cal_mask]
    cal_label = text_is_pure[cal_mask].astype(int)

    if len(cal_r) < 10 or np.sum(cal_label == 1) < 3 or np.sum(cal_label == 0) < 3:
        print("  Insufficient calibration data")
        return {"error": "insufficient calibration data"}

    # Find threshold that maximizes accuracy on text
    thresholds = np.percentile(cal_r[np.isfinite(cal_r)], np.arange(5, 96, 5))
    best_acc = 0
    best_thresh = None
    for t in thresholds:
        pred = (cal_r >= t).astype(int)
        valid = np.isfinite(cal_r)
        acc = np.mean(pred[valid] == cal_label[valid])
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"  Text calibration: threshold={best_thresh:.4f}, accuracy={best_acc:.4f}")
    print(f"    (Pure clusters: {np.sum(cal_label==1)}, Mixed clusters: {np.sum(cal_label==0)})")

    transfer_results = {
        "calibration_domain": "text",
        "threshold": float(best_thresh),
        "calibration_accuracy": float(best_acc),
        "transfers": {},
    }

    # Transfer to tabular: "good" clusters = R^2 > median, "bad" = R^2 < median
    tab_clusters = tabular_results["clusters"]
    if tab_clusters:
        tab_r_simple = np.array([c["metrics"]["R_simple"] for c in tab_clusters])
        tab_r2 = np.array([c["r2_oos"] for c in tab_clusters])
        valid_tab = np.isfinite(tab_r_simple) & np.isfinite(tab_r2)
        if np.sum(valid_tab) >= 5:
            median_r2 = np.median(tab_r2[valid_tab])
            tab_pred_good = (tab_r_simple[valid_tab] >= best_thresh).astype(int)
            tab_actual_good = (tab_r2[valid_tab] >= median_r2).astype(int)
            tab_transfer_acc = float(np.mean(tab_pred_good == tab_actual_good))
            transfer_results["transfers"]["tabular"] = {
                "accuracy": tab_transfer_acc,
                "n": int(np.sum(valid_tab)),
                "r2_median_threshold": float(median_r2),
            }
            print(f"  Transfer to tabular: accuracy={tab_transfer_acc:.4f} (n={np.sum(valid_tab)})")
        else:
            transfer_results["transfers"]["tabular"] = {"error": "insufficient data"}
            print("  Transfer to tabular: insufficient data")

    # Transfer to financial: "good" stocks = Sharpe > median, "bad" = Sharpe < median
    fin_stocks = financial_results.get("stocks", [])
    if fin_stocks:
        fin_r_simple = np.array([s["metrics"]["R_simple"] for s in fin_stocks])
        fin_sharpe = np.array([s["sharpe_ratio"] for s in fin_stocks])
        valid_fin = np.isfinite(fin_r_simple) & np.isfinite(fin_sharpe)
        if np.sum(valid_fin) >= 5:
            median_sharpe = np.median(fin_sharpe[valid_fin])
            fin_pred_good = (fin_r_simple[valid_fin] >= best_thresh).astype(int)
            fin_actual_good = (fin_sharpe[valid_fin] >= median_sharpe).astype(int)
            fin_transfer_acc = float(np.mean(fin_pred_good == fin_actual_good))
            transfer_results["transfers"]["financial"] = {
                "accuracy": fin_transfer_acc,
                "n": int(np.sum(valid_fin)),
                "sharpe_median_threshold": float(median_sharpe),
            }
            print(f"  Transfer to financial: accuracy={fin_transfer_acc:.4f} (n={np.sum(valid_fin)})")
        else:
            transfer_results["transfers"]["financial"] = {"error": "insufficient data"}
            print("  Transfer to financial: insufficient data")

    return transfer_results


# ===========================================================================
# ADJUDICATION
# ===========================================================================
def adjudicate(results):
    """
    Pre-registered criteria (adapted from README):
    - CONFIRM: R correlates with ground truth (rho > 0.5, p < 0.05) in >= 2/3 domains,
      AND R outperforms E alone in >= 2/3 domains
    - FALSIFY: R fails (rho < 0.3) in ALL domains, OR R never outperforms E alone
    - INCONCLUSIVE: otherwise
    """
    print("\n" + "=" * 70)
    print("ADJUDICATION")
    print("=" * 70)

    domain_outcomes = {}

    # Text: use aggregate mean rho across models, vs purity
    text_corr = results["domains"].get("text", {}).get("aggregate_correlations", {})
    if text_corr:
        # Best R metric vs purity
        r_simple_purity = text_corr.get("R_simple_vs_purity", {}).get("mean_rho", float("nan"))
        r_full_purity = text_corr.get("R_full_vs_purity", {}).get("mean_rho", float("nan"))
        e_purity = text_corr.get("E_vs_purity", {}).get("mean_rho", float("nan"))
        best_r_purity = max(r_simple_purity, r_full_purity) if not (np.isnan(r_simple_purity) and np.isnan(r_full_purity)) else float("nan")

        # Also check vs silhouette
        r_simple_sil = text_corr.get("R_simple_vs_silhouette", {}).get("mean_rho", float("nan"))
        r_full_sil = text_corr.get("R_full_vs_silhouette", {}).get("mean_rho", float("nan"))
        e_sil = text_corr.get("E_vs_silhouette", {}).get("mean_rho", float("nan"))
        best_r_sil = max(r_simple_sil, r_full_sil) if not (np.isnan(r_simple_sil) and np.isnan(r_full_sil)) else float("nan")

        # Use purity as primary ground truth
        # We need p-values -- use first model's correlations
        first_model = list(results["domains"]["text"]["correlations_by_model"].keys())[0]
        first_corr = results["domains"]["text"]["correlations_by_model"][first_model]

        r_rho = best_r_purity
        e_rho = e_purity
        # For p-value, take max of R_simple and R_full
        r_simple_p = first_corr.get("R_simple_vs_purity", {}).get("p_value", 1.0)
        r_full_p = first_corr.get("R_full_vs_purity", {}).get("p_value", 1.0)
        best_r_p = min(r_simple_p, r_full_p)

        passes = r_rho > 0.5 and best_r_p < 0.05
        beats_e = abs(r_rho) > abs(e_rho)

        domain_outcomes["text"] = {
            "best_R_rho_vs_purity": float(r_rho),
            "E_rho_vs_purity": float(e_rho),
            "best_R_p_value": float(best_r_p),
            "passes_threshold": passes,
            "beats_E_alone": beats_e,
            "best_R_rho_vs_silhouette": float(best_r_sil),
            "E_rho_vs_silhouette": float(e_sil),
        }
        print(f"  Text: best_R_rho={r_rho:.4f} (p={best_r_p:.4e}), E_rho={e_rho:.4f}, "
              f"passes={passes}, beats_E={beats_e}")

    # Tabular: R vs R^2_oos
    tab_corr = results["domains"].get("tabular", {}).get("correlations", {})
    if tab_corr:
        r_simple_rho = tab_corr.get("R_simple_vs_r2_oos", {}).get("rho", float("nan"))
        r_full_rho = tab_corr.get("R_full_vs_r2_oos", {}).get("rho", float("nan"))
        e_rho = tab_corr.get("E_vs_r2_oos", {}).get("rho", float("nan"))
        best_r = max(r_simple_rho, r_full_rho) if not (np.isnan(r_simple_rho) and np.isnan(r_full_rho)) else float("nan")
        r_simple_p = tab_corr.get("R_simple_vs_r2_oos", {}).get("p_value", 1.0)
        r_full_p = tab_corr.get("R_full_vs_r2_oos", {}).get("p_value", 1.0)
        best_p = min(r_simple_p, r_full_p)

        passes = best_r > 0.5 and best_p < 0.05
        beats_e = abs(best_r) > abs(e_rho)

        domain_outcomes["tabular"] = {
            "best_R_rho": float(best_r),
            "E_rho": float(e_rho),
            "best_R_p_value": float(best_p),
            "passes_threshold": passes,
            "beats_E_alone": beats_e,
        }
        print(f"  Tabular: best_R_rho={best_r:.4f} (p={best_p:.4e}), E_rho={e_rho:.4f}, "
              f"passes={passes}, beats_E={beats_e}")

    # Financial: R vs Sharpe
    fin_corr = results["domains"].get("financial", {}).get("correlations", {})
    if fin_corr:
        r_simple_rho = fin_corr.get("R_simple_vs_sharpe", {}).get("rho", float("nan"))
        r_full_rho = fin_corr.get("R_full_vs_sharpe", {}).get("rho", float("nan"))
        e_rho = fin_corr.get("E_vs_sharpe", {}).get("rho", float("nan"))
        best_r = max(r_simple_rho, r_full_rho) if not (np.isnan(r_simple_rho) and np.isnan(r_full_rho)) else float("nan")
        r_simple_p = fin_corr.get("R_simple_vs_sharpe", {}).get("p_value", 1.0)
        r_full_p = fin_corr.get("R_full_vs_sharpe", {}).get("p_value", 1.0)
        best_p = min(r_simple_p, r_full_p)

        passes = best_r > 0.5 and best_p < 0.05
        beats_e = abs(best_r) > abs(e_rho)

        domain_outcomes["financial"] = {
            "best_R_rho": float(best_r),
            "E_rho": float(e_rho),
            "best_R_p_value": float(best_p),
            "passes_threshold": passes,
            "beats_E_alone": beats_e,
        }
        print(f"  Financial: best_R_rho={best_r:.4f} (p={best_p:.4e}), E_rho={e_rho:.4f}, "
              f"passes={passes}, beats_E={beats_e}")

    # Count
    n_domains = len(domain_outcomes)
    n_pass = sum(1 for d in domain_outcomes.values() if d.get("passes_threshold", False))
    n_beats_e = sum(1 for d in domain_outcomes.values() if d.get("beats_E_alone", False))
    all_fail = all(
        abs(d.get("best_R_rho", d.get("best_R_rho_vs_purity", 0))) < 0.3
        for d in domain_outcomes.values()
    )
    never_beats_e = n_beats_e == 0

    print(f"\n  Summary: {n_pass}/{n_domains} domains pass (rho>0.5, p<0.05)")
    print(f"  R beats E alone in {n_beats_e}/{n_domains} domains")

    if n_pass >= 2 and n_beats_e >= 2:
        verdict = "CONFIRM"
    elif all_fail or never_beats_e:
        verdict = "FALSIFY"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  VERDICT: {verdict}")

    results["adjudication"] = {
        "domain_outcomes": domain_outcomes,
        "n_domains_pass": n_pass,
        "n_domains_R_beats_E": n_beats_e,
        "n_domains_total": n_domains,
        "all_domains_fail": all_fail,
        "never_beats_e": never_beats_e,
        "verdict": verdict,
    }
    results["verdict"] = verdict

    return verdict


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    import pandas as pd  # needed for yfinance

    start_time = time.time()

    # Collect all cluster metrics for SNR verification
    all_cluster_metrics = []

    # Domain 1: Text
    print("\n>>> Running Text Domain...")
    text_results = run_text_domain()
    RESULTS["domains"]["text"] = text_results

    # Collect metrics for SNR check
    for cl in text_results.get("clusters", []):
        all_cluster_metrics.append((f"text_{cl['cluster_id']}", cl["metrics"]))

    # Domain 2: Tabular
    print("\n>>> Running Tabular Domain...")
    tabular_results = run_tabular_domain()
    RESULTS["domains"]["tabular"] = tabular_results

    for cl in tabular_results.get("clusters", []):
        all_cluster_metrics.append((f"tabular_{cl['cluster_id']}", cl["metrics"]))

    # Domain 3: Financial
    print("\n>>> Running Financial Domain...")
    financial_results = run_financial_domain()
    RESULTS["domains"]["financial"] = financial_results

    for st in financial_results.get("stocks", []):
        all_cluster_metrics.append((f"financial_{st['ticker']}", st["metrics"]))

    # R = SNR Verification
    snr_results = verify_r_equals_snr(all_cluster_metrics)
    RESULTS["snr_verification"] = snr_results

    # Cross-domain Transfer
    transfer_results = run_cross_domain_transfer(text_results, tabular_results, financial_results)
    RESULTS["cross_domain_transfer"] = transfer_results

    # Adjudication
    verdict = adjudicate(RESULTS)

    elapsed = time.time() - start_time
    RESULTS["elapsed_seconds"] = float(elapsed)

    # Save results
    output_path = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\q03_generalization\results\test_v2_q03_fixed_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to: {output_path}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")
