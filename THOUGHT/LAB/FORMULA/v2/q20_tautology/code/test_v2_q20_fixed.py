"""
Q20 v2 Fixed: Is R = (E/grad_S) * sigma^Df tautological?

Methodology:
- 20 Newsgroups dataset (real text, natural topic clusters)
- 3 architectures: MiniLM-L6, MPNet-base, multi-qa-MiniLM
- 60 clusters per architecture (20 pure, 20 mixed-2, 20 random)
- Ground truth: cluster purity
- Pre-registered criteria from README.md

Tests:
1. Component comparison (R_full vs each component, Spearman rho with purity)
2. 8e conservation (PR * alpha across architectures and sample sizes)
3. Ablation (5 functional forms)
4. Novel predictions (3 pre-registered predictions)

NO synthetic data. NO reward-maxing. Just truth.
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


def P(msg=""):
    """Print with flush."""
    print(msg, flush=True)


# Import formula module
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import hashlib


def compute_purity(labels):
    """Cluster purity: fraction of docs from dominant category."""
    if len(labels) == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts.values()) / len(labels)


def compute_all_metrics(embeddings):
    """Compute all formula components for a cluster."""
    result = formula.compute_all(embeddings)
    E = result['E']
    grad_S = result['grad_S']
    sigma = result['sigma']
    Df = result['Df']

    result['inv_grad_S'] = 1.0 / grad_S if grad_S > 1e-10 else float('nan')
    result['E_times_grad_S'] = E * grad_S if not np.isnan(grad_S) else float('nan')

    if not np.isnan(sigma) and not np.isnan(Df):
        result['sigma_Df'] = sigma ** Df
    else:
        result['sigma_Df'] = float('nan')

    return result


def compute_ablation_from_components(E, grad_S, sigma, Df):
    """Compute all 5 ablation forms from pre-computed components."""
    forms = {}

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]) and grad_S > 1e-10:
        forms['R_full'] = (E / grad_S) * (sigma ** Df)
    else:
        forms['R_full'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        forms['R_sub'] = (E - grad_S) * (sigma ** Df)
    else:
        forms['R_sub'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        forms['R_exp'] = E * np.exp(-grad_S) * (sigma ** Df)
    else:
        forms['R_exp'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S]):
        forms['R_log'] = np.log(max(E, 0.01)) / (grad_S + 1)
    else:
        forms['R_log'] = float('nan')

    if not any(np.isnan(x) for x in [E, grad_S, sigma, Df]):
        denom = grad_S + sigma ** Df
        forms['R_add'] = E / denom if denom > 1e-10 else float('nan')
    else:
        forms['R_add'] = float('nan')

    return forms


def bootstrap_rho_difference(values_a, values_b, ground_truth, n_bootstrap=10000, seed=42):
    """
    Bootstrap test: is |Spearman rho| of values_a with ground_truth
    significantly greater than that of values_b?
    Returns (mean_diff, p_value, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)

    valid = ~(np.isnan(values_a) | np.isnan(values_b) | np.isnan(ground_truth))
    va = np.array(values_a)[valid]
    vb = np.array(values_b)[valid]
    gt = np.array(ground_truth)[valid]
    n = len(gt)

    if n < 10:
        return float('nan'), float('nan'), float('nan'), float('nan')

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rho_a, _ = spearmanr(va[idx], gt[idx])
        rho_b, _ = spearmanr(vb[idx], gt[idx])
        diffs[i] = abs(rho_a) - abs(rho_b)

    mean_diff = float(np.mean(diffs))
    p_value = float(np.mean(diffs <= 0))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    return mean_diff, p_value, ci_lower, ci_upper


def create_clusters(data, rng, n_pure=20, n_mixed=20, n_random=20, docs_per_cluster=200):
    """
    Create 60 clusters from 20 Newsgroups data.
    - 20 pure: 200 docs from one category
    - 20 mixed-2: 100 docs from each of 2 categories
    - 20 random: 200 docs from all categories
    """
    targets = np.array(data.target)
    categories = list(range(20))

    cat_indices = {}
    for cat in categories:
        cat_indices[cat] = np.where(targets == cat)[0]

    clusters = []

    # Pure clusters
    for cat in categories:
        available = cat_indices[cat]
        if len(available) >= docs_per_cluster:
            chosen = rng.choice(available, size=docs_per_cluster, replace=False)
        else:
            chosen = rng.choice(available, size=docs_per_cluster, replace=True)
        labels = targets[chosen]
        clusters.append((chosen, labels, 'pure'))

    # Mixed-2 clusters
    all_pairs = list((i, j) for i in range(20) for j in range(i+1, 20))
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[:n_mixed]

    for cat_a, cat_b in selected_pairs:
        half = docs_per_cluster // 2
        avail_a = cat_indices[cat_a]
        avail_b = cat_indices[cat_b]

        if len(avail_a) >= half:
            chosen_a = rng.choice(avail_a, size=half, replace=False)
        else:
            chosen_a = rng.choice(avail_a, size=half, replace=True)

        if len(avail_b) >= half:
            chosen_b = rng.choice(avail_b, size=half, replace=False)
        else:
            chosen_b = rng.choice(avail_b, size=half, replace=True)

        chosen = np.concatenate([chosen_a, chosen_b])
        labels = targets[chosen]
        clusters.append((chosen, labels, 'mixed'))

    # Random clusters
    all_indices = np.arange(len(targets))
    for _ in range(n_random):
        chosen = rng.choice(all_indices, size=docs_per_cluster, replace=False)
        labels = targets[chosen]
        clusters.append((chosen, labels, 'random'))

    return clusters


def compute_8e_metrics(embeddings):
    """
    Compute PR and alpha for 8e conservation test.
    PR = participation ratio of covariance eigenvalues
    alpha = power-law exponent from top-half eigenvalue fit
    """
    n, d = embeddings.shape
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    PR = float((np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2))

    half = len(eigenvalues) // 2
    top_eigs = eigenvalues[:half]
    if len(top_eigs) < 3:
        return PR, float('nan'), float('nan')

    k = np.arange(1, len(top_eigs) + 1)
    log_k = np.log(k)
    log_eig = np.log(top_eigs)

    A = np.vstack([log_k, np.ones(len(log_k))]).T
    result = np.linalg.lstsq(A, log_eig, rcond=None)
    alpha = float(-result[0][0])

    product = PR * alpha
    return PR, alpha, product


def run_all_tests():
    """Run all Q20 v2 tests."""
    results = {
        'metadata': {
            'question': 'Q20',
            'version': 'v2_fixed',
            'timestamp': datetime.now().isoformat(),
            'methodology': '20 Newsgroups, 3 architectures, 60 clusters, purity ground truth',
            'seed': 42,
        },
        'test1_component_comparison': {},
        'test2_8e_conservation': {},
        'test3_ablation': {},
        'test4_novel_predictions': {},
        'verdict': {}
    }

    rng = np.random.RandomState(42)
    EIGHT_E = 8 * np.e  # 21.746

    # ---- Load 20 Newsgroups ----
    P("=" * 70)
    P("Q20 v2 FIXED: Is R tautological?")
    P("=" * 70)
    P()
    P("Loading 20 Newsgroups dataset...")
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    P(f"  Loaded {len(data.data)} documents across {len(data.target_names)} categories")

    # Data fingerprint
    text_hash = hashlib.sha256('|'.join(data.data[:100]).encode('utf-8', errors='replace')).hexdigest()[:16]
    results['metadata']['data_hash'] = text_hash
    results['metadata']['n_docs'] = len(data.data)
    results['metadata']['n_categories'] = len(data.target_names)

    # ---- Create clusters ----
    P("Creating 60 clusters (20 pure, 20 mixed, 20 random)...")
    clusters = create_clusters(data, rng)
    purities = [compute_purity(labels) for _, labels, _ in clusters]
    cluster_types = [ctype for _, _, ctype in clusters]
    P(f"  Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
    P(f"  Pure mean: {np.mean([p for p, t in zip(purities, cluster_types) if t == 'pure']):.3f}")
    P(f"  Mixed mean: {np.mean([p for p, t in zip(purities, cluster_types) if t == 'mixed']):.3f}")
    P(f"  Random mean: {np.mean([p for p, t in zip(purities, cluster_types) if t == 'random']):.3f}")

    # Collect unique doc indices across all clusters
    all_doc_set = set()
    for indices, _, _ in clusters:
        all_doc_set.update(indices.tolist())
    all_doc_list = sorted(all_doc_set)
    all_texts = [data.data[i] for i in all_doc_list]
    P(f"  Unique documents to encode: {len(all_doc_list)}")

    # ---- Architectures ----
    model_names = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1',
    ]

    for model_name in model_names:
        P()
        P("-" * 70)
        P(f"Architecture: {model_name}")
        P("-" * 70)

        t_model_start = time.time()
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        P(f"  Embedding dimension: {dim}")

        # Adjust batch size based on model dimension
        if dim > 512:
            encode_batch = 64
        else:
            encode_batch = 128

        # Encode all unique documents in chunks to manage memory
        P(f"  Encoding {len(all_texts)} unique documents (batch={encode_batch})...")
        chunk_size = 2000
        all_emb_parts = []
        for chunk_start in range(0, len(all_texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_texts))
            chunk_texts = all_texts[chunk_start:chunk_end]
            chunk_embs = model.encode(chunk_texts, show_progress_bar=False, batch_size=encode_batch)
            all_emb_parts.append(chunk_embs)
            P(f"    Encoded {chunk_end}/{len(all_texts)} docs...")
        all_embeddings = np.vstack(all_emb_parts)
        del all_emb_parts
        idx_to_pos = {idx: pos for pos, idx in enumerate(all_doc_list)}
        P(f"  Encoding done. Shape: {all_embeddings.shape}")

        # ============================================================
        # TEST 1: Component Comparison
        # ============================================================
        P("  Running Test 1: Component Comparison...")
        metrics_per_cluster = []
        for ci, (indices, labels, ctype) in enumerate(clusters):
            positions = [idx_to_pos[idx] for idx in indices]
            embs = all_embeddings[positions]
            m = compute_all_metrics(embs)
            m['purity'] = compute_purity(labels)
            m['type'] = ctype
            metrics_per_cluster.append(m)
            if (ci + 1) % 20 == 0:
                P(f"    Computed {ci+1}/60 clusters...")

        purity_arr = np.array([m['purity'] for m in metrics_per_cluster])

        component_names = ['E', 'grad_S', 'sigma', 'Df', 'inv_grad_S',
                           'sigma_Df', 'R_simple', 'R_full']

        component_rhos = {}
        for comp in component_names:
            vals = np.array([m[comp] for m in metrics_per_cluster])
            valid = ~np.isnan(vals)
            if np.sum(valid) >= 10:
                rho, pval = spearmanr(vals[valid], purity_arr[valid])
                component_rhos[comp] = {
                    'rho': float(rho),
                    'abs_rho': float(abs(rho)),
                    'p_value': float(pval),
                    'n_valid': int(np.sum(valid))
                }
            else:
                component_rhos[comp] = {
                    'rho': float('nan'),
                    'abs_rho': float('nan'),
                    'p_value': float('nan'),
                    'n_valid': int(np.sum(valid))
                }

        P(f"    {'Component':<15} {'rho':>8} {'|rho|':>8} {'p-value':>12}")
        P(f"    {'-'*45}")
        for comp in component_names:
            r = component_rhos[comp]
            P(f"    {comp:<15} {r['rho']:>8.4f} {r['abs_rho']:>8.4f} {r['p_value']:>12.6f}")

        base_components = ['E', 'grad_S', 'sigma', 'Df', 'inv_grad_S', 'sigma_Df']
        best_base_name = max(base_components, key=lambda c: component_rhos[c]['abs_rho'])
        best_base_rho = component_rhos[best_base_name]['abs_rho']
        r_full_rho = component_rhos['R_full']['abs_rho']

        margin = r_full_rho - best_base_rho
        r_full_outperforms_all = all(
            r_full_rho > component_rhos[c]['abs_rho'] + 0.05
            for c in base_components
        )

        P(f"\n    R_full |rho| = {r_full_rho:.4f}")
        P(f"    Best base component: {best_base_name} |rho| = {best_base_rho:.4f}")
        P(f"    Margin: {margin:.4f}")
        P(f"    R_full outperforms ALL by >= 0.05: {r_full_outperforms_all}")

        r_full_sign = 'positive' if component_rhos['R_full']['rho'] > 0 else 'negative'
        P(f"    R_full correlation sign: {r_full_sign}")

        # Bootstrap: R_full vs best base
        r_full_vals = np.array([m['R_full'] for m in metrics_per_cluster])
        best_base_vals = np.array([m[best_base_name] for m in metrics_per_cluster])
        P("    Running bootstrap (R_full vs best base)...")
        boot_diff, boot_p, boot_ci_lo, boot_ci_hi = bootstrap_rho_difference(
            r_full_vals, best_base_vals, purity_arr, n_bootstrap=10000, seed=42
        )
        P(f"    Bootstrap |rho| diff (R_full - {best_base_name}): {boot_diff:.4f}")
        P(f"    Bootstrap p-value: {boot_p:.4f}")
        P(f"    Bootstrap 95% CI: [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")

        # Bootstrap R_full vs EACH base component
        P("    Running bootstrap (R_full vs each component)...")
        bootstrap_vs_all = {}
        for ci2, comp in enumerate(base_components):
            comp_vals = np.array([m[comp] for m in metrics_per_cluster])
            bd, bp, bcl, bch = bootstrap_rho_difference(
                r_full_vals, comp_vals, purity_arr, n_bootstrap=10000, seed=42
            )
            bootstrap_vs_all[comp] = {
                'mean_diff': bd, 'p_value': bp,
                'ci_lower': bcl, 'ci_upper': bch
            }
            P(f"      vs {comp}: diff={bd:.4f}, p={bp:.4f}")

        all_significant = all(
            bootstrap_vs_all[c]['p_value'] < 0.05
            for c in base_components
        )
        P(f"    R_full significantly beats ALL components (p<0.05): {all_significant}")

        results['test1_component_comparison'][model_name] = {
            'component_rhos': component_rhos,
            'best_base_component': best_base_name,
            'best_base_abs_rho': best_base_rho,
            'R_full_abs_rho': r_full_rho,
            'margin_over_best': margin,
            'R_full_outperforms_all_by_005': r_full_outperforms_all,
            'R_full_sign': r_full_sign,
            'bootstrap_vs_best': {
                'mean_diff': boot_diff, 'p_value': boot_p,
                'ci_lower': boot_ci_lo, 'ci_upper': boot_ci_hi
            },
            'bootstrap_vs_all_components': bootstrap_vs_all,
            'all_bootstrap_significant': all_significant,
            'cluster_metrics': [
                {
                    'purity': m['purity'],
                    'type': m['type'],
                    'E': m['E'],
                    'grad_S': m['grad_S'],
                    'sigma': m['sigma'],
                    'Df': m['Df'],
                    'R_simple': m['R_simple'],
                    'R_full': m['R_full'],
                }
                for m in metrics_per_cluster
            ]
        }

        # ============================================================
        # TEST 2: 8e Conservation
        # ============================================================
        P("\n  Running Test 2: 8e Conservation...")

        sample_2000 = rng.choice(len(data.data), size=2000, replace=False)
        texts_2000 = [data.data[i] for i in sample_2000]
        P("    Encoding 2000 sample docs...")
        embs_2000 = model.encode(texts_2000, show_progress_bar=False, batch_size=encode_batch)
        P("    Encoding done.")

        sample_sizes = [100, 200, 500, 1000, 2000]
        eight_e_results = {}

        for ss in sample_sizes:
            embs_ss = embs_2000[:ss]
            PR, alpha, product = compute_8e_metrics(embs_ss)
            error_pct = abs(product - EIGHT_E) / EIGHT_E * 100
            eight_e_results[str(ss)] = {
                'PR': PR,
                'alpha': alpha,
                'product': product,
                'target_8e': EIGHT_E,
                'error_pct': error_pct
            }
            P(f"    n={ss:>5}: PR={PR:.4f}, alpha={alpha:.4f}, "
              f"PR*alpha={product:.4f}, error={error_pct:.1f}%")

        results['test2_8e_conservation'][model_name] = eight_e_results

        # ============================================================
        # TEST 3: Ablation (reuse pre-computed components)
        # ============================================================
        P("\n  Running Test 3: Ablation (5 functional forms)...")

        ablation_per_cluster = []
        for ci, m in enumerate(metrics_per_cluster):
            forms = compute_ablation_from_components(
                m['E'], m['grad_S'], m['sigma'], m['Df']
            )
            forms['purity'] = m['purity']
            ablation_per_cluster.append(forms)

        ablation_names = ['R_full', 'R_sub', 'R_exp', 'R_log', 'R_add']
        ablation_rhos = {}

        P(f"    {'Form':<12} {'rho':>8} {'|rho|':>8} {'p-value':>12}")
        P(f"    {'-'*40}")
        for form_name in ablation_names:
            vals = np.array([a[form_name] for a in ablation_per_cluster])
            valid = ~np.isnan(vals)
            if np.sum(valid) >= 10:
                rho, pval = spearmanr(vals[valid], purity_arr[valid])
                ablation_rhos[form_name] = {
                    'rho': float(rho),
                    'abs_rho': float(abs(rho)),
                    'p_value': float(pval),
                    'n_valid': int(np.sum(valid))
                }
            else:
                ablation_rhos[form_name] = {
                    'rho': float('nan'), 'abs_rho': float('nan'),
                    'p_value': float('nan'), 'n_valid': int(np.sum(valid))
                }
            r = ablation_rhos[form_name]
            P(f"    {form_name:<12} {r['rho']:>8.4f} {r['abs_rho']:>8.4f} {r['p_value']:>12.6f}")

        best_ablation = max(ablation_names, key=lambda f: ablation_rhos[f]['abs_rho'])
        r_full_is_best_ablation = best_ablation == 'R_full'
        P(f"\n    Best ablation form: {best_ablation} "
          f"(|rho|={ablation_rhos[best_ablation]['abs_rho']:.4f})")
        P(f"    R_full is best: {r_full_is_best_ablation}")

        results['test3_ablation'][model_name] = {
            'form_rhos': ablation_rhos,
            'best_form': best_ablation,
            'R_full_is_best': r_full_is_best_ablation,
        }

        # ============================================================
        # TEST 4: Novel Predictions
        # ============================================================
        P("\n  Running Test 4: Novel Predictions...")

        pred1_pass = r_full_rho > 0.4
        P(f"    P1: R_full |rho| > 0.4? {r_full_rho:.4f} -> {pred1_pass}")

        valid_mask = ~np.isnan(r_full_vals)
        valid_rfull = r_full_vals[valid_mask]
        valid_purity = purity_arr[valid_mask]
        q75 = np.percentile(valid_rfull, 75)
        top_quartile_mask = valid_rfull >= q75
        top_quartile_purity = float(np.mean(valid_purity[top_quartile_mask]))
        pred2_pass = top_quartile_purity > 0.7
        P(f"    P2: Top quartile R_full mean purity > 0.7? "
          f"{top_quartile_purity:.4f} -> {pred2_pass}")

        all_metrics_dict = {}
        all_metrics_dict.update(component_rhos)
        all_metrics_dict.update(ablation_rhos)
        best_overall = max(all_metrics_dict.keys(), key=lambda k: all_metrics_dict[k]['abs_rho'])
        pred3_pass = best_overall == 'R_full'
        P(f"    P3: R_full is best overall metric? Best={best_overall} "
          f"(|rho|={all_metrics_dict[best_overall]['abs_rho']:.4f}) -> {pred3_pass}")

        preds_passed = sum([pred1_pass, pred2_pass, pred3_pass])
        P(f"    Predictions passed: {preds_passed}/3")

        results['test4_novel_predictions'][model_name] = {
            'P1_rho_above_04': {
                'R_full_abs_rho': r_full_rho,
                'threshold': 0.4,
                'passed': pred1_pass
            },
            'P2_top_quartile_purity': {
                'top_quartile_mean_purity': top_quartile_purity,
                'threshold': 0.7,
                'passed': pred2_pass
            },
            'P3_best_metric': {
                'best_metric': best_overall,
                'best_abs_rho': all_metrics_dict[best_overall]['abs_rho'],
                'R_full_abs_rho': r_full_rho,
                'passed': pred3_pass
            },
            'total_passed': preds_passed,
        }

        t_model_end = time.time()
        P(f"\n  Model {model_name} completed in {t_model_end - t_model_start:.1f}s")

        # Free memory before loading next model
        del model, all_embeddings
        import gc
        gc.collect()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    P()
    P("=" * 70)
    P("FINAL VERDICT COMPUTATION")
    P("=" * 70)

    n_archs = len(model_names)

    outperforms_count = 0
    positive_sign_count = 0
    significant_count = 0
    for mn in model_names:
        t1 = results['test1_component_comparison'][mn]
        if t1['R_full_outperforms_all_by_005']:
            outperforms_count += 1
        if t1['R_full_sign'] == 'positive':
            positive_sign_count += 1
        if t1['all_bootstrap_significant']:
            significant_count += 1

    eight_e_errors = []
    for mn in model_names:
        err = results['test2_8e_conservation'][mn]['2000']['error_pct']
        eight_e_errors.append(err)
    mean_8e_error = np.mean(eight_e_errors)
    eight_e_within_30 = mean_8e_error < 30
    eight_e_within_100 = mean_8e_error < 100

    best_ablation_count = 0
    for mn in model_names:
        if results['test3_ablation'][mn]['R_full_is_best']:
            best_ablation_count += 1

    novel_pass_count = 0
    for mn in model_names:
        if results['test4_novel_predictions'][mn]['total_passed'] >= 2:
            novel_pass_count += 1

    P(f"\n  Outperforms all components (>=0.05 margin): {outperforms_count}/{n_archs}")
    P(f"  Positive correlation sign: {positive_sign_count}/{n_archs}")
    P(f"  Bootstrap significant (all components): {significant_count}/{n_archs}")
    P(f"  8e mean error at n=2000: {mean_8e_error:.1f}%")
    P(f"  8e within 30%: {eight_e_within_30}")
    P(f"  8e within 100%: {eight_e_within_100}")
    P(f"  R_full is best ablation: {best_ablation_count}/{n_archs}")
    P(f"  Novel predictions >= 2/3: {novel_pass_count}/{n_archs}")

    # CONFIRM criteria
    confirm = (
        outperforms_count == n_archs and
        positive_sign_count == n_archs and
        significant_count == n_archs and
        eight_e_within_30 and
        best_ablation_count == n_archs and
        novel_pass_count >= 2
    )

    # FALSIFY criteria
    falsify = (
        (outperforms_count == 0) or
        (not eight_e_within_100) or
        (best_ablation_count == 0 and novel_pass_count == 0)
    )

    if confirm:
        verdict = "CONFIRMED"
        explanation = ("R_full outperforms all components significantly with positive correlation, "
                       "8e conservation within 30%, R_full is uniquely best ablation form, "
                       "and novel predictions pass.")
    elif falsify:
        verdict = "FALSIFIED"
        reasons = []
        if outperforms_count == 0:
            reasons.append("R_full does not outperform components in any architecture")
        if not eight_e_within_100:
            reasons.append(f"8e error ({mean_8e_error:.1f}%) exceeds 100%")
        if best_ablation_count == 0 and novel_pass_count == 0:
            reasons.append("R_full is not best ablation form AND all novel predictions fail")
        explanation = "; ".join(reasons)
    else:
        verdict = "INCONCLUSIVE"
        explanation = ("Mixed results: R_full shows some advantages but does not meet all "
                       "CONFIRM criteria and does not fail all FALSIFY criteria.")

    P(f"\n  VERDICT: {verdict}")
    P(f"  Explanation: {explanation}")

    results['verdict'] = {
        'verdict': verdict,
        'explanation': explanation,
        'criteria': {
            'outperforms_all_components': f"{outperforms_count}/{n_archs}",
            'positive_correlation': f"{positive_sign_count}/{n_archs}",
            'bootstrap_significant': f"{significant_count}/{n_archs}",
            'eight_e_mean_error_pct': float(mean_8e_error),
            'eight_e_within_30_pct': eight_e_within_30,
            'eight_e_within_100_pct': eight_e_within_100,
            'best_ablation_form': f"{best_ablation_count}/{n_archs}",
            'novel_predictions_pass': f"{novel_pass_count}/{n_archs}",
        }
    }

    return results


if __name__ == '__main__':
    start = time.time()
    results = run_all_tests()
    elapsed = time.time() - start
    results['metadata']['elapsed_seconds'] = elapsed
    P(f"\nTotal time: {elapsed:.1f}s")

    out_path = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\q20_tautology\results\test_v2_q20_fixed_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    P(f"\nResults saved to: {out_path}")
