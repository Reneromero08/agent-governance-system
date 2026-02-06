"""
Q05 v2 FIXED Test: Does High Local Agreement (High R) Reveal Truth?

Rerun with improved methodology. NO synthetic data for echo chambers.
NO grouping artifacts. Uses 20 Newsgroups natural topic clusters.

Pre-registered criteria:
  CONFIRM:  R correlates with purity (rho > 0.5) in >= 2/3 architectures
            AND R outperforms E alone
            AND R distinguishes echo from genuine agreement
  FALSIFY:  R fails purity correlation (rho < 0.2) in all architectures
            OR echo chambers have higher R than genuine agreement
            OR bias attack inflates R > 2x
  INCONCLUSIVE: otherwise

Seed: 42
"""

import sys
import os
import json
import time
import warnings
import importlib.util
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ---------- Import formula via importlib ----------
spec = importlib.util.spec_from_file_location(
    "formula",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v2\shared\formula.py"
)
formula = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formula)

compute_E = formula.compute_E
compute_grad_S = formula.compute_grad_S
compute_R_simple = formula.compute_R_simple
compute_R_full = formula.compute_R_full
compute_all = formula.compute_all

# ---------- Paths ----------
REPO_ROOT = Path(r"D:\CCC 2.0\AI\agent-governance-system")
RESULTS_DIR = REPO_ROOT / "THOUGHT" / "LAB" / "FORMULA" / "v2" / "q05_agreement_truth" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# =============================================================================
# Utility
# =============================================================================

def safe_R(embeddings, variant="simple"):
    """Compute R with NaN fallback."""
    if embeddings.shape[0] < 3:
        return float('nan')
    if variant == "simple":
        return compute_R_simple(embeddings)
    else:
        return compute_R_full(embeddings)


def safe_E(embeddings):
    """Compute E with NaN fallback."""
    if embeddings.shape[0] < 2:
        return float('nan')
    return compute_E(embeddings)


def cluster_purity(labels):
    """Fraction of items belonging to the dominant class."""
    if len(labels) == 0:
        return 0.0
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


# =============================================================================
# Data Loading
# =============================================================================

def load_20newsgroups(max_docs=5000):
    """Load 20 Newsgroups dataset, subsample for CPU feasibility."""
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    print(f"Loaded 20 Newsgroups: {len(data.data)} documents, {len(data.target_names)} categories")

    # Filter out very short documents (< 20 chars) to avoid degenerate embeddings
    valid_idx = [i for i, doc in enumerate(data.data) if len(doc.strip()) >= 20]
    all_texts = [data.data[i] for i in valid_idx]
    all_labels = [data.target[i] for i in valid_idx]
    target_names = data.target_names
    print(f"After filtering short docs: {len(all_texts)} documents")

    # Stratified subsample to max_docs for CPU feasibility
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
        # Fill remainder randomly from what's left
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

    # Verify category distribution
    cat_counts = Counter(labels)
    print(f"Category counts: min={min(cat_counts.values())}, max={max(cat_counts.values())}, "
          f"mean={np.mean(list(cat_counts.values())):.0f}")

    return texts, labels, target_names


def encode_with_model(model_name, texts, batch_size=32):
    """Encode texts with a sentence-transformer model. Returns embeddings, releases model."""
    import gc
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {model_name}...")
    model = SentenceTransformer(model_name)
    # Truncate long documents to first 256 chars for encoding speed on CPU
    # sentence-transformers truncate at token level anyway (max 256 tokens for MiniLM)
    truncated = [t[:256] for t in texts]
    print(f"  Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(
        truncated, show_progress_bar=False, convert_to_numpy=True,
        batch_size=batch_size
    )
    elapsed = time.time() - t0
    print(f"  Done: shape {embeddings.shape} in {elapsed:.1f}s")
    # Release model to free memory
    del model
    gc.collect()
    return embeddings


# =============================================================================
# Cluster Construction
# =============================================================================

# Related category groups for near-pure clusters
RELATED_GROUPS = {
    'comp': [0, 1, 2, 3, 4],   # comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x
    'rec': [7, 8, 9, 10],       # rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
    'sci': [11, 12, 13, 14],    # sci.crypt, sci.electronics, sci.med, sci.space
    'talk': [16, 17, 18, 19],   # talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc
    'religion': [15, 19],       # soc.religion.christian, talk.religion.misc
    'politics': [16, 17, 18],   # talk.politics.*
}

# Specific echo chamber categories (narrow, homogeneous viewpoint)
ECHO_CATEGORIES = [
    16,  # talk.politics.guns
    17,  # talk.politics.mideast
    15,  # soc.religion.christian
    19,  # talk.religion.misc
    0,   # comp.graphics
]

# Broad genuine agreement: related categories that span a broader topic
GENUINE_AGREEMENT_PAIRS = [
    ([11, 12, 13, 14], "sci.*"),            # sci.crypt + sci.electronics + sci.med + sci.space
    ([0, 1, 2, 3, 4], "comp.*"),            # all comp categories
    ([7, 8], "rec.vehicles"),               # rec.autos + rec.motorcycles
    ([9, 10], "rec.sports"),                # rec.sport.baseball + rec.sport.hockey
    ([16, 17, 18], "talk.politics.*"),      # all politics
]

# Cross-echo pairs: same broad topic but different viewpoints/framings
CROSS_ECHO_PAIRS = [
    (16, 18, "politics.guns vs politics.misc"),
    (17, 18, "politics.mideast vs politics.misc"),
    (15, 19, "soc.religion.christian vs talk.religion.misc"),
    (9, 10, "baseball vs hockey"),
    (0, 4, "comp.graphics vs comp.windows.x"),
]


def build_clusters(texts, labels, target_names):
    """
    Build 80 clusters as specified:
    - 20 pure (one category)
    - 20 near-pure (dominant + related minority)
    - 20 mixed (50-50 from 2 categories)
    - 20 random (from all categories)
    """
    np.random.seed(42)
    labels_arr = np.array(labels)
    n_categories = len(target_names)
    clusters = []

    # Index by category
    cat_indices = {}
    for cat in range(n_categories):
        cat_indices[cat] = np.where(labels_arr == cat)[0]
        # Shuffle for randomness
        np.random.shuffle(cat_indices[cat])

    print(f"\nCategory sizes:")
    for cat in range(n_categories):
        print(f"  {cat:2d} {target_names[cat]:30s}: {len(cat_indices[cat])}")

    cluster_size = 80  # Reduced from 100 to fit subsample (~250 per category)

    # --- 20 Pure clusters: 80 docs from one category ---
    print(f"\nBuilding 20 pure clusters (size={cluster_size})...")
    used_cats_pure = list(range(n_categories))
    np.random.shuffle(used_cats_pure)
    for i in range(20):
        cat = used_cats_pure[i % n_categories]
        idx = cat_indices[cat]
        if len(idx) < cluster_size:
            # Sample with replacement if category is small
            chosen = np.random.choice(idx, cluster_size, replace=True)
        else:
            # Use a fresh random sample each time
            chosen = np.random.choice(idx, cluster_size, replace=False)
        cluster_labels = [labels[j] for j in chosen]
        purity = cluster_purity(cluster_labels)
        clusters.append({
            'type': 'pure',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': purity,
            'description': f"pure_{target_names[cat]}_{i}",
        })

    # --- 20 Near-pure clusters: 80 dominant + 20 related ---
    print(f"Building 20 near-pure clusters...")
    related_pairs = []
    for group_name, group_cats in RELATED_GROUPS.items():
        for j in range(len(group_cats)):
            dominant = group_cats[j]
            minority = group_cats[(j + 1) % len(group_cats)]
            related_pairs.append((dominant, minority))
    np.random.shuffle(related_pairs)

    for i in range(20):
        dominant, minority = related_pairs[i % len(related_pairs)]
        n_dominant = 64
        n_minority = 16
        idx_d = cat_indices[dominant]
        idx_m = cat_indices[minority]
        chosen_d = np.random.choice(idx_d, min(n_dominant, len(idx_d)), replace=len(idx_d) < n_dominant)
        chosen_m = np.random.choice(idx_m, min(n_minority, len(idx_m)), replace=len(idx_m) < n_minority)
        chosen = np.concatenate([chosen_d, chosen_m])
        cluster_labels = [labels[j] for j in chosen]
        purity = cluster_purity(cluster_labels)
        clusters.append({
            'type': 'near_pure',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': purity,
            'description': f"near_pure_{target_names[dominant]}+{target_names[minority]}_{i}",
        })

    # --- 20 Mixed clusters: 50-50 from 2 different categories ---
    print(f"Building 20 mixed clusters...")
    # Pick dissimilar category pairs
    all_pairs = []
    for a in range(n_categories):
        for b in range(a + 1, n_categories):
            all_pairs.append((a, b))
    np.random.shuffle(all_pairs)

    for i in range(20):
        cat_a, cat_b = all_pairs[i % len(all_pairs)]
        idx_a = cat_indices[cat_a]
        idx_b = cat_indices[cat_b]
        chosen_a = np.random.choice(idx_a, 40, replace=len(idx_a) < 40)
        chosen_b = np.random.choice(idx_b, 40, replace=len(idx_b) < 40)
        chosen = np.concatenate([chosen_a, chosen_b])
        cluster_labels = [labels[j] for j in chosen]
        purity = cluster_purity(cluster_labels)
        clusters.append({
            'type': 'mixed',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': purity,
            'description': f"mixed_{target_names[cat_a]}+{target_names[cat_b]}_{i}",
        })

    # --- 20 Random clusters: 100 docs from all categories ---
    print(f"Building 20 random clusters...")
    all_indices = np.arange(len(texts))
    for i in range(20):
        chosen = np.random.choice(all_indices, cluster_size, replace=False)
        cluster_labels = [labels[j] for j in chosen]
        purity = cluster_purity(cluster_labels)
        clusters.append({
            'type': 'random',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': purity,
            'description': f"random_{i}",
        })

    # Summary
    type_counts = Counter(c['type'] for c in clusters)
    print(f"\nCluster summary: {dict(type_counts)}")
    purities = [c['purity'] for c in clusters]
    print(f"Purity range: [{min(purities):.3f}, {max(purities):.3f}]")
    print(f"Purity mean by type:")
    for ctype in ['pure', 'near_pure', 'mixed', 'random']:
        ps = [c['purity'] for c in clusters if c['type'] == ctype]
        print(f"  {ctype:12s}: mean={np.mean(ps):.3f}, std={np.std(ps):.3f}")

    return clusters


def build_echo_vs_genuine_clusters(texts, labels, target_names):
    """
    Build clusters for Test 2: Echo Chamber Detection.

    Echo chambers: docs from a single narrow category (topically homogeneous).
    Genuine agreement: docs from multiple related categories (broad but real agreement).
    Cross-echo: docs from two related but distinct categories.
    """
    np.random.seed(42)
    labels_arr = np.array(labels)
    cluster_size = 50
    clusters = []

    cat_indices = {}
    for cat in range(len(target_names)):
        cat_indices[cat] = np.where(labels_arr == cat)[0]

    # --- 10 Echo chamber clusters ---
    print("\nBuilding 10 echo chamber clusters...")
    echo_cats = ECHO_CATEGORIES * 2  # repeat to get 10
    for i in range(10):
        cat = echo_cats[i % len(echo_cats)]
        idx = cat_indices[cat]
        chosen = np.random.choice(idx, min(cluster_size, len(idx)), replace=len(idx) < cluster_size)
        cluster_labels = [labels[j] for j in chosen]
        clusters.append({
            'echo_type': 'echo_chamber',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': cluster_purity(cluster_labels),
            'description': f"echo_{target_names[cat]}_{i}",
        })

    # --- 10 Genuine agreement clusters ---
    print("Building 10 genuine agreement clusters...")
    genuine_specs = GENUINE_AGREEMENT_PAIRS * 2  # repeat to get 10
    for i in range(10):
        cats, name = genuine_specs[i % len(genuine_specs)]
        # Draw equally from each related category
        per_cat = cluster_size // len(cats)
        remainder = cluster_size - per_cat * len(cats)
        all_chosen = []
        for j, cat in enumerate(cats):
            n_draw = per_cat + (1 if j < remainder else 0)
            idx = cat_indices[cat]
            chosen = np.random.choice(idx, min(n_draw, len(idx)), replace=len(idx) < n_draw)
            all_chosen.append(chosen)
        chosen = np.concatenate(all_chosen)
        cluster_labels = [labels[j] for j in chosen]
        clusters.append({
            'echo_type': 'genuine_agreement',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': cluster_purity(cluster_labels),
            'description': f"genuine_{name}_{i}",
        })

    # --- 10 Cross-echo clusters ---
    print("Building 10 cross-echo clusters...")
    cross_specs = CROSS_ECHO_PAIRS * 2  # repeat to get 10
    for i in range(10):
        cat_a, cat_b, name = cross_specs[i % len(cross_specs)]
        n_a = cluster_size // 2
        n_b = cluster_size - n_a
        idx_a = cat_indices[cat_a]
        idx_b = cat_indices[cat_b]
        chosen_a = np.random.choice(idx_a, min(n_a, len(idx_a)), replace=len(idx_a) < n_a)
        chosen_b = np.random.choice(idx_b, min(n_b, len(idx_b)), replace=len(idx_b) < n_b)
        chosen = np.concatenate([chosen_a, chosen_b])
        cluster_labels = [labels[j] for j in chosen]
        clusters.append({
            'echo_type': 'cross_echo',
            'indices': chosen.tolist(),
            'labels': cluster_labels,
            'purity': cluster_purity(cluster_labels),
            'description': f"cross_{name}_{i}",
        })

    # Summary
    for etype in ['echo_chamber', 'genuine_agreement', 'cross_echo']:
        ps = [c['purity'] for c in clusters if c['echo_type'] == etype]
        print(f"  {etype:20s}: n={len(ps)}, mean_purity={np.mean(ps):.3f}")

    return clusters


# =============================================================================
# TEST 1: Agreement-Truth Correlation (REDESIGNED)
# =============================================================================

def test1_agreement_truth_correlation(clusters, all_embeddings_by_model):
    """
    For each of the 80 clusters, compute R_simple, R_full, and E alone.
    Ground truth = cluster purity.
    Correlate R with purity (Spearman, n=80) for each architecture.
    Compare: does R outperform E alone?
    """
    print("\n" + "=" * 70)
    print("TEST 1: Agreement-Truth Correlation (80 clusters x 3 architectures)")
    print("=" * 70)

    model_names = list(all_embeddings_by_model.keys())
    results_by_model = {}

    purities = np.array([c['purity'] for c in clusters])
    types = [c['type'] for c in clusters]

    for model_name in model_names:
        print(f"\n--- Architecture: {model_name} ---")
        embeddings = all_embeddings_by_model[model_name]

        R_simples = []
        R_fulls = []
        Es = []
        grad_Ss = []

        dim = embeddings.shape[1]
        # Only compute R_full for lower-dim models (eigendecomp of d x d cov is O(d^3))
        do_full = (dim <= 400)

        for ci, cluster in enumerate(clusters):
            idx = cluster['indices']
            cluster_embs = embeddings[idx]

            r_s = safe_R(cluster_embs, "simple")
            r_f = safe_R(cluster_embs, "full") if do_full else float('nan')
            e = safe_E(cluster_embs)
            g = compute_grad_S(cluster_embs) if cluster_embs.shape[0] >= 2 else float('nan')

            R_simples.append(r_s)
            R_fulls.append(r_f)
            Es.append(e)
            grad_Ss.append(g)

        R_simples = np.array(R_simples)
        R_fulls = np.array(R_fulls)
        Es = np.array(Es)
        grad_Ss = np.array(grad_Ss)

        # Remove NaN entries for correlation
        valid_s = ~np.isnan(R_simples)
        valid_f = ~np.isnan(R_fulls)
        valid_e = ~np.isnan(Es)

        # Spearman correlations with purity
        rho_s, p_s = stats.spearmanr(R_simples[valid_s], purities[valid_s])
        rho_f, p_f = stats.spearmanr(R_fulls[valid_f], purities[valid_f])
        rho_e, p_e = stats.spearmanr(Es[valid_e], purities[valid_e])

        print(f"  n_valid: R_simple={valid_s.sum()}, R_full={valid_f.sum()}, E={valid_e.sum()}")
        print(f"  Spearman(R_simple, purity) = {rho_s:.4f}, p = {p_s:.4e}")
        print(f"  Spearman(R_full, purity)   = {rho_f:.4f}, p = {p_f:.4e}")
        print(f"  Spearman(E, purity)        = {rho_e:.4f}, p = {p_e:.4e}")
        print(f"  R_simple outperforms E alone: {abs(rho_s) > abs(rho_e)}")
        print(f"  R_full outperforms E alone:   {abs(rho_f) > abs(rho_e)}")

        # Per-type breakdown
        print(f"\n  Per-type means:")
        for ctype in ['pure', 'near_pure', 'mixed', 'random']:
            mask = np.array([t == ctype for t in types])
            valid_mask = mask & valid_s
            if valid_mask.sum() > 0:
                print(f"    {ctype:12s}: R_simple={np.mean(R_simples[valid_mask]):8.4f}, "
                      f"E={np.mean(Es[mask & valid_e]):8.4f}, "
                      f"purity={np.mean(purities[mask]):6.3f}")

        results_by_model[model_name] = {
            "rho_R_simple_purity": float(rho_s),
            "p_R_simple_purity": float(p_s),
            "rho_R_full_purity": float(rho_f),
            "p_R_full_purity": float(p_f),
            "rho_E_purity": float(rho_e),
            "p_E_purity": float(p_e),
            "R_simple_outperforms_E": bool(abs(rho_s) > abs(rho_e)),
            "R_full_outperforms_E": bool(abs(rho_f) > abs(rho_e)),
            "n_valid_R_simple": int(valid_s.sum()),
            "n_valid_R_full": int(valid_f.sum()),
            "per_type_R_simple": {},
            "per_type_E": {},
            "per_type_purity": {},
            "R_simples": [float(x) for x in R_simples],
            "R_fulls": [float(x) for x in R_fulls],
            "Es": [float(x) for x in Es],
        }
        for ctype in ['pure', 'near_pure', 'mixed', 'random']:
            mask = np.array([t == ctype for t in types])
            valid_mask = mask & valid_s
            if valid_mask.sum() > 0:
                results_by_model[model_name]["per_type_R_simple"][ctype] = float(np.mean(R_simples[valid_mask]))
                results_by_model[model_name]["per_type_E"][ctype] = float(np.mean(Es[mask & valid_e]))
                results_by_model[model_name]["per_type_purity"][ctype] = float(np.mean(purities[mask]))

    # Summary across models
    print(f"\n--- Summary Across Architectures ---")
    confirm_count = 0
    for mn in model_names:
        r = results_by_model[mn]
        rho = r["rho_R_simple_purity"]
        print(f"  {mn:35s}: rho(R_simple,purity)={rho:+.4f}, outperforms_E={r['R_simple_outperforms_E']}")
        if rho > 0.5:
            confirm_count += 1

    results = {
        "test": "Agreement-Truth Correlation (Redesigned)",
        "n_clusters": len(clusters),
        "purities": [float(p) for p in purities],
        "types": types,
        "results_by_model": results_by_model,
        "n_models_rho_gt_0_5": confirm_count,
        "criterion_confirm": confirm_count >= 2,
    }

    return results


# =============================================================================
# TEST 2: Echo Chamber Detection (REDESIGNED -- realistic)
# =============================================================================

def test2_echo_chamber_detection(echo_clusters, all_embeddings_by_model):
    """
    Echo chambers = docs from a single narrow category.
    Genuine agreement = docs from multiple related categories.
    Cross-echo = docs from two related but distinct categories.

    Question: Does R distinguish echo chambers from genuine diverse agreement?
    If R is higher for narrow echo chambers than for genuine broad agreement,
    the formula rewards homogeneity over truth.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Echo Chamber Detection (Realistic)")
    print("=" * 70)

    model_names = list(all_embeddings_by_model.keys())
    results_by_model = {}

    for model_name in model_names:
        print(f"\n--- Architecture: {model_name} ---")
        embeddings = all_embeddings_by_model[model_name]

        type_Rs = {'echo_chamber': [], 'genuine_agreement': [], 'cross_echo': []}
        type_Es = {'echo_chamber': [], 'genuine_agreement': [], 'cross_echo': []}

        for cluster in echo_clusters:
            idx = cluster['indices']
            cluster_embs = embeddings[idx]
            r_s = safe_R(cluster_embs, "simple")
            e = safe_E(cluster_embs)
            etype = cluster['echo_type']
            if not np.isnan(r_s):
                type_Rs[etype].append(r_s)
            if not np.isnan(e):
                type_Es[etype].append(e)

        print(f"\n  {'Type':25s} | {'mean R':>10} | {'std R':>10} | {'mean E':>10} | {'n':>5}")
        print(f"  {'-'*70}")
        for etype in ['echo_chamber', 'genuine_agreement', 'cross_echo']:
            rs = type_Rs[etype]
            es = type_Es[etype]
            if len(rs) > 0:
                print(f"  {etype:25s} | {np.mean(rs):10.4f} | {np.std(rs):10.4f} | {np.mean(es):10.4f} | {len(rs):5d}")

        # Key comparison: echo vs genuine
        echo_Rs = np.array(type_Rs['echo_chamber'])
        genuine_Rs = np.array(type_Rs['genuine_agreement'])
        cross_Rs = np.array(type_Rs['cross_echo'])

        if len(echo_Rs) > 0 and len(genuine_Rs) > 0:
            # Mann-Whitney test
            stat, p_val = stats.mannwhitneyu(echo_Rs, genuine_Rs, alternative='two-sided')
            echo_higher = np.mean(echo_Rs) > np.mean(genuine_Rs)
            ratio = np.mean(echo_Rs) / np.mean(genuine_Rs) if np.mean(genuine_Rs) != 0 else float('inf')

            print(f"\n  Echo vs Genuine:")
            print(f"    Echo mean R:    {np.mean(echo_Rs):.4f}")
            print(f"    Genuine mean R: {np.mean(genuine_Rs):.4f}")
            print(f"    Ratio (echo/genuine): {ratio:.2f}x")
            print(f"    Mann-Whitney U={stat:.1f}, p={p_val:.4e}")
            print(f"    Echo has HIGHER R: {echo_higher}")

            if echo_higher:
                print(f"    --> CONCERN: R rewards homogeneity (echo > genuine)")
            else:
                print(f"    --> GOOD: R correctly values diverse genuine agreement")
        else:
            stat, p_val = float('nan'), float('nan')
            echo_higher = None
            ratio = float('nan')

        results_by_model[model_name] = {
            "echo_mean_R": float(np.mean(echo_Rs)) if len(echo_Rs) > 0 else None,
            "echo_std_R": float(np.std(echo_Rs)) if len(echo_Rs) > 0 else None,
            "genuine_mean_R": float(np.mean(genuine_Rs)) if len(genuine_Rs) > 0 else None,
            "genuine_std_R": float(np.std(genuine_Rs)) if len(genuine_Rs) > 0 else None,
            "cross_mean_R": float(np.mean(cross_Rs)) if len(cross_Rs) > 0 else None,
            "echo_genuine_ratio": float(ratio) if not np.isnan(ratio) else None,
            "mannwhitney_U": float(stat) if not np.isnan(stat) else None,
            "mannwhitney_p": float(p_val) if not np.isnan(p_val) else None,
            "echo_higher_than_genuine": bool(echo_higher) if echo_higher is not None else None,
            "echo_Rs": [float(x) for x in echo_Rs],
            "genuine_Rs": [float(x) for x in genuine_Rs],
            "cross_Rs": [float(x) for x in cross_Rs],
            "echo_Es": [float(x) for x in type_Es['echo_chamber']],
            "genuine_Es": [float(x) for x in type_Es['genuine_agreement']],
        }

    # Summary
    print(f"\n--- Echo Detection Summary ---")
    n_echo_higher = sum(1 for mn in model_names
                        if results_by_model[mn]["echo_higher_than_genuine"] is True)
    n_genuine_higher = sum(1 for mn in model_names
                          if results_by_model[mn]["echo_higher_than_genuine"] is False)
    print(f"  Models where echo R > genuine R: {n_echo_higher}/{len(model_names)}")
    print(f"  Models where genuine R > echo R: {n_genuine_higher}/{len(model_names)}")

    # Falsification: echo chambers have higher R than genuine agreement in ALL models
    falsified = n_echo_higher == len(model_names)

    results = {
        "test": "Echo Chamber Detection (Redesigned)",
        "n_echo_clusters": sum(1 for c in echo_clusters if c['echo_type'] == 'echo_chamber'),
        "n_genuine_clusters": sum(1 for c in echo_clusters if c['echo_type'] == 'genuine_agreement'),
        "n_cross_clusters": sum(1 for c in echo_clusters if c['echo_type'] == 'cross_echo'),
        "results_by_model": results_by_model,
        "n_echo_higher": n_echo_higher,
        "n_genuine_higher": n_genuine_higher,
        "criterion_falsified": falsified,
    }

    return results


# =============================================================================
# TEST 3: Bias Attack (KEEP -- was valid)
# =============================================================================

def test3_bias_attack(texts, labels, all_embeddings_by_model):
    """
    Take 20 clusters of 50 docs each from random categories.
    Compute R before and after prepending boilerplate.
    Measure R inflation.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Bias Attack (Boilerplate Prepending)")
    print("=" * 70)

    np.random.seed(42)

    # Build 20 random clusters of 50 docs
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

    # We need to re-encode biased texts. Use the first model only for efficiency.
    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Get clean embeddings
    clean_embeddings = all_embeddings_by_model[model_name]

    results_per_bias = []

    for bias_phrase in bias_phrases:
        print(f"\n--- Bias phrase: '{bias_phrase}' ---")

        # We only need to encode the texts that appear in our clusters
        all_cluster_idx = np.unique(np.concatenate(cluster_indices))
        biased_texts = {i: bias_phrase + texts[i][:256] for i in all_cluster_idx}

        # Encode biased texts
        biased_text_list = [biased_texts[i] for i in all_cluster_idx]
        biased_embs_raw = model.encode(biased_text_list, show_progress_bar=False,
                                       convert_to_numpy=True, batch_size=32)
        # Map back to indices
        biased_emb_map = {}
        for j, i in enumerate(all_cluster_idx):
            biased_emb_map[i] = biased_embs_raw[j]

        R_clean_list = []
        R_biased_list = []
        E_clean_list = []
        E_biased_list = []
        inflation_ratios = []

        for ci, idx in enumerate(cluster_indices):
            clean_embs = clean_embeddings[idx]
            biased_embs = np.array([biased_emb_map[i] for i in idx])

            r_clean = safe_R(clean_embs, "simple")
            r_biased = safe_R(biased_embs, "simple")
            e_clean = safe_E(clean_embs)
            e_biased = safe_E(biased_embs)

            R_clean_list.append(r_clean)
            R_biased_list.append(r_biased)
            E_clean_list.append(e_clean)
            E_biased_list.append(e_biased)

            if not np.isnan(r_clean) and r_clean > 1e-10 and not np.isnan(r_biased):
                inflation_ratios.append(r_biased / r_clean)

        R_clean_arr = np.array(R_clean_list)
        R_biased_arr = np.array(R_biased_list)
        valid = ~(np.isnan(R_clean_arr) | np.isnan(R_biased_arr))

        mean_clean = np.mean(R_clean_arr[valid])
        mean_biased = np.mean(R_biased_arr[valid])
        mean_inflation = np.mean(inflation_ratios) if inflation_ratios else float('nan')
        median_inflation = np.median(inflation_ratios) if inflation_ratios else float('nan')

        print(f"  Mean R clean:  {mean_clean:.4f}")
        print(f"  Mean R biased: {mean_biased:.4f}")
        print(f"  Mean inflation ratio:   {mean_inflation:.2f}x")
        print(f"  Median inflation ratio: {median_inflation:.2f}x")

        # Check if inflation > 2x (falsification threshold)
        inflates_gt_2x = mean_inflation > 2.0
        print(f"  Inflation > 2x: {inflates_gt_2x}")

        results_per_bias.append({
            "bias_phrase": bias_phrase,
            "mean_R_clean": float(mean_clean),
            "mean_R_biased": float(mean_biased),
            "mean_inflation_ratio": float(mean_inflation) if not np.isnan(mean_inflation) else None,
            "median_inflation_ratio": float(median_inflation) if not np.isnan(median_inflation) else None,
            "inflation_gt_2x": bool(inflates_gt_2x),
            "n_valid_clusters": int(valid.sum()),
            "R_clean_values": [float(x) for x in R_clean_arr],
            "R_biased_values": [float(x) for x in R_biased_arr],
            "inflation_ratios": [float(x) for x in inflation_ratios],
        })

    # Summary
    n_inflated = sum(1 for r in results_per_bias if r["inflation_gt_2x"])
    any_inflated = n_inflated > 0
    print(f"\n--- Bias Attack Summary ---")
    print(f"  Bias phrases tested: {len(bias_phrases)}")
    print(f"  Phrases causing >2x inflation: {n_inflated}/{len(bias_phrases)}")

    results = {
        "test": "Bias Attack",
        "model": model_name,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "bias_results": results_per_bias,
        "any_inflation_gt_2x": any_inflated,
        "criterion_falsified": any_inflated,
    }

    return results


# =============================================================================
# TEST 4: Multi-Architecture Agreement
# =============================================================================

def test4_multi_architecture_agreement(clusters, all_embeddings_by_model):
    """
    For each of the 80 clusters, compute R using all 3 architectures.
    Check inter-architecture agreement on R rankings.
    Check if R-based ranking correlates with purity consistently.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Architecture Agreement")
    print("=" * 70)

    model_names = list(all_embeddings_by_model.keys())
    purities = np.array([c['purity'] for c in clusters])

    # Compute R for each model
    R_by_model = {}
    for model_name in model_names:
        embeddings = all_embeddings_by_model[model_name]
        Rs = []
        for cluster in clusters:
            idx = cluster['indices']
            cluster_embs = embeddings[idx]
            r_s = safe_R(cluster_embs, "simple")
            Rs.append(r_s)
        R_by_model[model_name] = np.array(Rs)

    # Inter-architecture Spearman correlation on R values
    print("\n  Inter-architecture Spearman correlations (R_simple):")
    inter_arch_corrs = {}
    for i, mn_a in enumerate(model_names):
        for j, mn_b in enumerate(model_names):
            if j <= i:
                continue
            Ra = R_by_model[mn_a]
            Rb = R_by_model[mn_b]
            valid = ~(np.isnan(Ra) | np.isnan(Rb))
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(Ra[valid], Rb[valid])
            else:
                rho, p = float('nan'), float('nan')
            key = f"{mn_a} vs {mn_b}"
            inter_arch_corrs[key] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}
            print(f"    {key}: rho={rho:.4f}, p={p:.4e}, n={valid.sum()}")

    # R vs purity for each model
    print("\n  R vs purity by architecture:")
    R_purity_corrs = {}
    for model_name in model_names:
        Rs = R_by_model[model_name]
        valid = ~np.isnan(Rs)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(Rs[valid], purities[valid])
        else:
            rho, p = float('nan'), float('nan')
        R_purity_corrs[model_name] = {"rho": float(rho), "p": float(p), "n": int(valid.sum())}
        print(f"    {model_name}: rho(R,purity)={rho:.4f}, p={p:.4e}")

    # Check consistency: are the R-purity correlations in the same direction?
    rhos = [R_purity_corrs[mn]["rho"] for mn in model_names]
    all_same_sign = all(r > 0 for r in rhos if not np.isnan(r)) or all(r < 0 for r in rhos if not np.isnan(r))
    print(f"\n  All architectures agree on sign of R-purity correlation: {all_same_sign}")
    print(f"  Rhos: {[f'{r:.4f}' for r in rhos]}")

    results = {
        "test": "Multi-Architecture Agreement",
        "model_names": model_names,
        "inter_architecture_correlations": inter_arch_corrs,
        "R_purity_correlations": R_purity_corrs,
        "all_same_sign": all_same_sign,
        "R_values_by_model": {mn: [float(x) for x in R_by_model[mn]] for mn in model_names},
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import gc
    print("Q05 v2 FIXED TEST: DOES HIGH AGREEMENT (HIGH R) REVEAL TRUTH?")
    print("=" * 70)
    print(f"Seed: 42")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    # ---- Load data ----
    texts, labels, target_names = load_20newsgroups()

    # ---- Encode with all 3 architectures (one at a time to save memory) ----
    model_specs = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ]

    all_embeddings = {}
    for mname in model_specs:
        t0 = time.time()
        all_embeddings[mname] = encode_with_model(mname, texts)
        print(f"  Total time for {mname}: {time.time() - t0:.1f}s")
        gc.collect()

    # ---- Build clusters ----
    clusters_80 = build_clusters(texts, labels, target_names)
    echo_clusters_30 = build_echo_vs_genuine_clusters(texts, labels, target_names)

    # ---- Run Tests ----
    all_results = {
        "metadata": {
            "test": "Q05 v2 FIXED",
            "question": "Does high local agreement (high R) reveal truth?",
            "seed": 42,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "n_documents": len(texts),
            "n_categories": len(target_names),
            "architectures": model_specs,
            "target_names": target_names,
        }
    }

    # Test 1
    t1_start = time.time()
    r1 = test1_agreement_truth_correlation(clusters_80, all_embeddings)
    r1["duration_s"] = time.time() - t1_start
    all_results["test1_agreement_truth"] = r1

    # Test 2
    t2_start = time.time()
    r2 = test2_echo_chamber_detection(echo_clusters_30, all_embeddings)
    r2["duration_s"] = time.time() - t2_start
    all_results["test2_echo_chamber"] = r2

    # Test 3
    t3_start = time.time()
    r3 = test3_bias_attack(texts, labels, all_embeddings)
    r3["duration_s"] = time.time() - t3_start
    all_results["test3_bias_attack"] = r3

    # Test 4
    t4_start = time.time()
    r4 = test4_multi_architecture_agreement(clusters_80, all_embeddings)
    r4["duration_s"] = time.time() - t4_start
    all_results["test4_multi_arch"] = r4

    # =================================================================
    # VERDICT DETERMINATION
    # =================================================================
    print("\n" + "=" * 70)
    print("VERDICT DETERMINATION")
    print("=" * 70)

    # Criterion 1: R correlates with purity (rho > 0.5) in >= 2/3 architectures
    # AND R outperforms E alone
    n_confirm_rho = r1["n_models_rho_gt_0_5"]
    rho_vals = {mn: r1["results_by_model"][mn]["rho_R_simple_purity"] for mn in model_specs}
    outperforms_E = {mn: r1["results_by_model"][mn]["R_simple_outperforms_E"] for mn in model_specs}
    n_outperforms = sum(1 for v in outperforms_E.values() if v)

    criterion1_confirm = n_confirm_rho >= 2 and n_outperforms >= 2
    criterion1_falsify = all(abs(rho_vals[mn]) < 0.2 for mn in model_specs)

    print(f"\nCriterion 1: R correlates with purity (rho > 0.5) in >= 2/3 architectures, AND outperforms E")
    for mn in model_specs:
        print(f"  {mn}: rho={rho_vals[mn]:.4f}, outperforms_E={outperforms_E[mn]}")
    print(f"  Models with rho > 0.5: {n_confirm_rho}/3")
    print(f"  Models where R outperforms E: {n_outperforms}/3")
    print(f"  CONFIRM: {criterion1_confirm}")
    print(f"  FALSIFY (all rho < 0.2): {criterion1_falsify}")

    # Criterion 2: R distinguishes echo from genuine agreement
    n_echo_higher = r2["n_echo_higher"]
    criterion2_confirm = r2["n_genuine_higher"] == len(model_specs)
    criterion2_falsify = r2["n_echo_higher"] == len(model_specs)

    print(f"\nCriterion 2: R distinguishes echo from genuine agreement")
    for mn in model_specs:
        m = r2["results_by_model"][mn]
        print(f"  {mn}: echo_R={m['echo_mean_R']}, genuine_R={m['genuine_mean_R']}, "
              f"ratio={m['echo_genuine_ratio']}, echo_higher={m['echo_higher_than_genuine']}")
    print(f"  CONFIRM (genuine > echo in all): {criterion2_confirm}")
    print(f"  FALSIFY (echo > genuine in all): {criterion2_falsify}")

    # Criterion 3: Bias attack does NOT inflate R > 2x
    criterion3_falsify = r3["any_inflation_gt_2x"]
    criterion3_confirm = not criterion3_falsify

    print(f"\nCriterion 3: Bias attack does NOT inflate R > 2x")
    for br in r3["bias_results"]:
        print(f"  '{br['bias_phrase']}': inflation={br['mean_inflation_ratio']}x, >2x={br['inflation_gt_2x']}")
    print(f"  CONFIRM (no >2x inflation): {criterion3_confirm}")
    print(f"  FALSIFY (any >2x inflation): {criterion3_falsify}")

    # Overall verdict
    if criterion1_confirm and criterion2_confirm and criterion3_confirm:
        verdict = "CONFIRMED"
    elif criterion1_falsify or criterion2_falsify or criterion3_falsify:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT: {verdict}")
    print(f"{'=' * 70}")

    all_results["verdict"] = {
        "result": verdict,
        "criterion1": {
            "description": "R correlates with purity (rho>0.5) in >=2/3 archs AND outperforms E",
            "rho_values": {mn: float(rho_vals[mn]) for mn in model_specs},
            "n_rho_gt_0_5": n_confirm_rho,
            "n_outperforms_E": n_outperforms,
            "confirm": criterion1_confirm,
            "falsify": criterion1_falsify,
        },
        "criterion2": {
            "description": "R distinguishes echo chambers from genuine agreement",
            "n_echo_higher": n_echo_higher,
            "n_genuine_higher": r2["n_genuine_higher"],
            "confirm": criterion2_confirm,
            "falsify": criterion2_falsify,
        },
        "criterion3": {
            "description": "Bias attack does NOT inflate R > 2x",
            "any_inflation_gt_2x": r3["any_inflation_gt_2x"],
            "confirm": criterion3_confirm,
            "falsify": criterion3_falsify,
        },
    }

    all_results["total_duration_s"] = time.time() - total_start

    # Save results
    results_file = RESULTS_DIR / "test_v2_q05_fixed_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
