#!/usr/bin/env python3
"""
E.X.4.2: Cross-Model Symbol Resolution Test

Tests that governance symbols (法, 真, 道) resolve correctly across
spectrally aligned embedding models.

Key questions:
1. Do symbols map to the same semantic region after alignment?
2. How much does H(X|S) decrease with alignment?
3. Are polysemic symbols (道) handled correctly?
"""

import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.linalg import orthogonal_procrustes
from datetime import datetime
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Governance symbols from CODIFIER.md
GOVERNANCE_SYMBOLS = {
    "法": {
        "domain": "Canon Law",
        "expansion": "All governance rules and canonical documents",
        "path": "LAW/CANON/*"
    },
    "真": {
        "domain": "Truth Foundation",
        "expansion": "The semiotic foundation of truth, ontological principles",
        "path": "LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md"
    },
    "契": {
        "domain": "Contract",
        "expansion": "The governance contract defining system behavior",
        "path": "LAW/CANON/CONSTITUTION/CONTRACT.md"
    },
    "恆": {
        "domain": "Invariants",
        "expansion": "System invariants that must always hold",
        "path": "LAW/CANON/CONSTITUTION/INVARIANTS.md"
    },
    "驗": {
        "domain": "Verification",
        "expansion": "Verification protocols and audit procedures",
        "path": "LAW/CANON/GOVERNANCE/VERIFICATION.md"
    },
    "道": {
        "domain": "Path/Principle (polysemic)",
        "expansion": "Context-activated: path, principle, method, speech",
        "contexts": ["path", "way", "method", "speech"]
    }
}

# Test with symbol expansions for semantic similarity
SYMBOL_EXPANSIONS = {
    "法": "governance law rules contract canon constitution policy",
    "真": "truth foundation ontology principle semiotic meaning",
    "契": "contract agreement binding rules commitment obligation",
    "恆": "invariant constant unchanging stable persistent fixed",
    "驗": "verify validate audit check confirm test proof",
    "道": "path way method principle road approach direction",
}


def load_models():
    """Load multiple embedding models for comparison."""
    models = {}

    try:
        from sentence_transformers import SentenceTransformer

        model_names = [
            ("all-MiniLM-L6-v2", "minilm"),
            ("all-mpnet-base-v2", "mpnet"),
        ]

        for model_name, short_name in model_names:
            try:
                model = SentenceTransformer(model_name)
                models[short_name] = model
                print(f"  Loaded: {model_name}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")

    except ImportError:
        print("  sentence-transformers not available")

    return models


def embed_symbols(model, symbols: dict) -> dict:
    """Embed governance symbols and their expansions.

    Returns dict mapping symbol -> embedding vector
    """
    embeddings = {}

    for symbol, expansion in symbols.items():
        # Embed both the symbol and its expansion
        texts = [symbol, expansion]
        vecs = model.encode(texts)

        embeddings[symbol] = {
            "symbol_vec": vecs[0],
            "expansion_vec": vecs[1],
            "combined_vec": (vecs[0] + vecs[1]) / 2  # Average for robustness
        }

    return embeddings


def compute_procrustes_alignment(source_vecs: np.ndarray, target_vecs: np.ndarray):
    """Compute Procrustes alignment from source to target space.

    If dimensions differ, projects to common space first.
    Returns rotation matrix R such that source @ R ≈ target
    """
    # Handle different dimensions by projecting to common space
    dim_a = source_vecs.shape[1]
    dim_b = target_vecs.shape[1]

    if dim_a != dim_b:
        # Project to lower dimension using SVD
        min_dim = min(dim_a, dim_b)

        # SVD of each matrix
        U_a, S_a, _ = np.linalg.svd(source_vecs, full_matrices=False)
        U_b, S_b, _ = np.linalg.svd(target_vecs, full_matrices=False)

        # Keep top-k dimensions
        k = min(min_dim, len(S_a), len(S_b))
        source_proj = U_a[:, :k] * S_a[:k]
        target_proj = U_b[:, :k] * S_b[:k]
    else:
        source_proj = source_vecs
        target_proj = target_vecs

    # Center both
    source_centered = source_proj - source_proj.mean(axis=0)
    target_centered = target_proj - target_proj.mean(axis=0)

    # Orthogonal Procrustes
    R, scale = orthogonal_procrustes(source_centered, target_centered)

    return R, scale, source_proj, target_proj


def measure_symbol_alignment(
    symbols: list,
    source_proj: np.ndarray,
    target_proj: np.ndarray,
    R: np.ndarray
) -> dict:
    """Measure how well symbols align after Procrustes rotation.

    For each symbol, compute:
    1. Raw cosine similarity (before alignment)
    2. Aligned cosine similarity (after rotation)
    3. Improvement factor
    """
    results = {}

    for i, symbol in enumerate(symbols):
        vec_a = source_proj[i]
        vec_b = target_proj[i]

        # Raw similarity (in projected space, before rotation)
        raw_sim = 1 - cosine(vec_a, vec_b)

        # Aligned similarity (after rotation)
        vec_a_aligned = vec_a @ R
        aligned_sim = 1 - cosine(vec_a_aligned, vec_b)

        results[symbol] = {
            "raw_similarity": float(raw_sim),
            "aligned_similarity": float(aligned_sim),
            "improvement": float(aligned_sim - raw_sim)
        }

    return results


def compute_conditional_entropy_reduction(
    before_distances: np.ndarray,
    after_distances: np.ndarray
) -> float:
    """Estimate H(X|S) reduction from alignment.

    Uses distance variance as proxy for entropy.
    Lower variance = lower conditional entropy = better alignment.
    """
    # Convert distances to "probabilities" via softmax-like
    def dist_to_entropy(distances):
        # Normalize to [0,1]
        normalized = distances / (distances.max() + 1e-10)
        # Higher distances = higher uncertainty
        return float(np.var(normalized))

    h_before = dist_to_entropy(before_distances)
    h_after = dist_to_entropy(after_distances)

    # Reduction ratio
    reduction = (h_before - h_after) / (h_before + 1e-10)

    return reduction


def test_cross_model_resolution():
    """Main test: symbol resolution across aligned models."""
    print("=" * 60)
    print("E.X.4.2: Cross-Model Symbol Resolution Test")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "test": "cross_model_symbol_resolution",
        "symbols_tested": list(GOVERNANCE_SYMBOLS.keys())
    }

    # Load models
    print("\nLoading embedding models...")
    models = load_models()

    if len(models) < 2:
        print("ERROR: Need at least 2 models for cross-model test")
        results["status"] = "FAILED"
        results["error"] = "insufficient_models"
        return results

    model_names = list(models.keys())
    print(f"\nModels: {model_names}")

    # Embed symbols with each model
    print("\nEmbedding governance symbols...")
    embeddings = {}
    for name, model in models.items():
        embeddings[name] = embed_symbols(model, SYMBOL_EXPANSIONS)
        print(f"  {name}: {len(embeddings[name])} symbols embedded")

    # Test all model pairs
    results["model_pairs"] = {}

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            pair_name = f"{model_a}_vs_{model_b}"
            print(f"\n--- {model_a} <-> {model_b} ---")

            emb_a = embeddings[model_a]
            emb_b = embeddings[model_b]

            # Get anchor vectors for alignment (use all symbols)
            symbols = list(emb_a.keys())
            source_vecs = np.array([emb_a[s]["combined_vec"] for s in symbols])
            target_vecs = np.array([emb_b[s]["combined_vec"] for s in symbols])

            # Compute Procrustes alignment (projects to common space if dims differ)
            R, scale, source_proj, target_proj = compute_procrustes_alignment(source_vecs, target_vecs)
            print(f"  Procrustes scale: {scale:.4f}")
            print(f"  Projected dimension: {source_proj.shape[1]}")

            # Measure symbol alignment
            alignment = measure_symbol_alignment(symbols, source_proj, target_proj, R)

            print("\n  Symbol Alignment (after Procrustes):")
            total_aligned_sim = 0
            total_raw_sim = 0

            for symbol, scores in alignment.items():
                raw = scores["raw_similarity"]
                aligned = scores["aligned_similarity"]
                total_raw_sim += raw
                total_aligned_sim += aligned

                # Print with improvement indicator
                improvement = "+" if aligned > raw else "-"
                print(f"    {symbol}: raw={raw:.3f} -> aligned={aligned:.3f} [{improvement}]")

            n_symbols = len(alignment)
            mean_raw = total_raw_sim / n_symbols
            mean_aligned = total_aligned_sim / n_symbols

            print(f"\n  Mean similarity: {mean_raw:.3f} -> {mean_aligned:.3f}")
            print(f"  Improvement: {(mean_aligned - mean_raw):.3f}")

            # Compute H(X|S) reduction (using projected vectors)
            before_dists = cdist(source_proj, target_proj, metric='cosine').diagonal()
            aligned_source = source_proj @ R
            after_dists = cdist(aligned_source, target_proj, metric='cosine').diagonal()

            h_reduction = compute_conditional_entropy_reduction(before_dists, after_dists)
            print(f"  H(X|S) reduction: {h_reduction:.1%}")

            results["model_pairs"][pair_name] = {
                "procrustes_scale": float(scale),
                "symbol_alignment": alignment,
                "mean_raw_similarity": float(mean_raw),
                "mean_aligned_similarity": float(mean_aligned),
                "h_reduction": float(h_reduction),
                "symbols_resolved": n_symbols
            }

    # Test polysemic symbol (道) specifically
    print("\n" + "=" * 60)
    print("Polysemic Symbol Test: 道 (path/way/method/speech)")
    print("=" * 60)

    if len(models) >= 2:
        # Embed different meanings of 道
        dao_contexts = {
            "道_path": "path road direction route",
            "道_way": "way method approach technique",
            "道_principle": "principle doctrine philosophy truth",
            "道_speech": "speak say tell express"
        }

        model_a_name = model_names[0]
        model_b_name = model_names[1]

        dao_emb_a = {}
        dao_emb_b = {}

        for ctx, expansion in dao_contexts.items():
            vec_a = models[model_a_name].encode([expansion])[0]
            vec_b = models[model_b_name].encode([expansion])[0]
            dao_emb_a[ctx] = vec_a
            dao_emb_b[ctx] = vec_b

        # Get alignment from full symbol set (reuse computed alignment)
        symbols = list(embeddings[model_a_name].keys())
        source_vecs = np.array([embeddings[model_a_name][s]["combined_vec"] for s in symbols])
        target_vecs = np.array([embeddings[model_b_name][s]["combined_vec"] for s in symbols])
        R, _, source_proj, target_proj = compute_procrustes_alignment(source_vecs, target_vecs)

        # For polysemic test, project dao embeddings using SVD
        dao_keys = list(dao_contexts.keys())
        dao_a_vecs = np.array([dao_emb_a[k] for k in dao_keys])
        dao_b_vecs = np.array([dao_emb_b[k] for k in dao_keys])

        # Project to same dimension as main alignment
        proj_dim = source_proj.shape[1]
        U_dao_a, S_dao_a, _ = np.linalg.svd(dao_a_vecs, full_matrices=False)
        U_dao_b, S_dao_b, _ = np.linalg.svd(dao_b_vecs, full_matrices=False)
        k = min(proj_dim, len(S_dao_a), len(S_dao_b))
        dao_a_proj = U_dao_a[:, :k] * S_dao_a[:k]
        dao_b_proj = U_dao_b[:, :k] * S_dao_b[:k]

        print(f"\nContext-specific alignment ({model_a_name} -> {model_b_name}):")
        polysemic_results = {}

        for i, ctx in enumerate(dao_keys):
            raw_sim = 1 - cosine(dao_a_proj[i], dao_b_proj[i])
            # Use k x k submatrix of R for alignment
            R_sub = R[:k, :k] if R.shape[0] >= k else R
            aligned_sim = 1 - cosine(dao_a_proj[i] @ R_sub, dao_b_proj[i])

            print(f"  {ctx}: raw={raw_sim:.3f} -> aligned={aligned_sim:.3f}")
            polysemic_results[ctx] = {
                "raw": float(raw_sim),
                "aligned": float(aligned_sim)
            }

        results["polysemic_test"] = {
            "symbol": "道",
            "contexts": dao_keys,
            "results": polysemic_results
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_aligned_sims = []
    all_h_reductions = []

    for pair_name, pair_results in results["model_pairs"].items():
        all_aligned_sims.append(pair_results["mean_aligned_similarity"])
        all_h_reductions.append(pair_results["h_reduction"])

    if all_aligned_sims:
        mean_sim = np.mean(all_aligned_sims)
        mean_h_red = np.mean(all_h_reductions)

        print(f"Mean aligned similarity: {mean_sim:.3f}")
        print(f"Mean H(X|S) reduction: {mean_h_red:.1%}")

        # Verdict
        if mean_sim > 0.7 and mean_h_red > 0.3:
            verdict = "PASS"
            print("\nVERDICT: PASS - Symbols resolve correctly across models")
        elif mean_sim > 0.5:
            verdict = "PARTIAL"
            print("\nVERDICT: PARTIAL - Some alignment achieved")
        else:
            verdict = "FAIL"
            print("\nVERDICT: FAIL - Poor cross-model resolution")

        results["summary"] = {
            "mean_aligned_similarity": float(mean_sim),
            "mean_h_reduction": float(mean_h_red),
            "verdict": verdict
        }

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "cross_model_symbols.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    test_cross_model_resolution()
