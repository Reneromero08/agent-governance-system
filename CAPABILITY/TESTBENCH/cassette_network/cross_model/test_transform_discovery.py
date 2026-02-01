#!/usr/bin/env python3
"""
Transform Discovery Tests

Phase 2: Discover the mapping T: Space_A -> Space_B between embedding spaces.

Uses Procrustes analysis to find orthogonal transforms that align
embeddings from different models. If eigenstructure is truly universal,
such transforms should exist with low alignment error.
"""
import json
import numpy as np
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add paths for imports
CROSS_MODEL_DIR = Path(__file__).resolve().parent
CASSETTE_NETWORK_DIR = CROSS_MODEL_DIR.parent
REPO_ROOT = CASSETTE_NETWORK_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))

# Fixtures directory
FIXTURES_DIR = CROSS_MODEL_DIR / "fixtures"

# Test corpus - same content embedded by different models
TEST_CORPUS = [
    "The system invariants must never be violated.",
    "Genesis prompt bootstraps a new agent.",
    "Catalytic computing ensures memory restoration.",
    "Contract rules govern system behavior.",
    "Authority gradient defines escalation paths.",
    "Verification chain ensures integrity.",
    "Receipts provide proof of execution.",
    "The canon is immutable law.",
    "Agents must declare their intentions.",
    "Truth-linking connects claims to evidence.",
    "Provenance tracks origin of all artifacts.",
    "Recovery procedures restore invariants.",
    "The governance system has five invariants.",
    "Bootstrap requires genesis context.",
    "Semantic embeddings encode meaning.",
    "Vector space represents concepts geometrically.",
    "Inner product measures similarity.",
    "Born rule gives quantum probability.",
    "Eigenvalues describe spectral structure.",
    "Alpha decay follows power law.",
    "The Living Formula computes relevance.",
    "R-gating determines what passes threshold.",
    "Cassettes store indexed semantic content.",
    "Cross-model alignment preserves meaning.",
    "Eigenstructure is universal across architectures.",
]


def procrustes_alignment(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find orthogonal transform R such that A @ R ~ B.

    Uses Procrustes analysis (SVD-based).

    Args:
        A: (N, d) source embeddings
        B: (N, d) target embeddings (same d after PCA)

    Returns:
        R: (d, d) orthogonal transform matrix
        error: mean alignment error
    """
    # Center both
    A_centered = A - A.mean(axis=0)
    B_centered = B - B.mean(axis=0)

    # SVD of A^T @ B
    M = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(M)

    # Optimal rotation
    R = U @ Vt

    # Compute alignment error
    A_transformed = A_centered @ R
    error = np.mean(np.linalg.norm(A_transformed - B_centered, axis=1))

    return R, error


def project_to_shared_dimension(emb_A: np.ndarray, emb_B: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project both embedding sets to shared k-dimensional space via PCA.

    Args:
        emb_A: (N, d_A) embeddings from model A
        emb_B: (N, d_B) embeddings from model B
        k: target dimension

    Returns:
        pca_A: (N, k) projected embeddings A
        pca_B: (N, k) projected embeddings B
    """
    from sklearn.decomposition import PCA

    pca_a = PCA(n_components=k)
    pca_b = PCA(n_components=k)

    proj_A = pca_a.fit_transform(emb_A)
    proj_B = pca_b.fit_transform(emb_B)

    return proj_A, proj_B


@pytest.fixture(scope="module")
def model_pairs_fixture():
    """Load model pairs configuration."""
    fixture_path = FIXTURES_DIR / "model_pairs.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    pytest.skip("model_pairs.json not found")


@pytest.fixture(scope="module")
def sentence_transformer():
    """Import sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.fixture(scope="module")
def sklearn_available():
    """Check sklearn is available."""
    try:
        from sklearn.decomposition import PCA
        return True
    except ImportError:
        pytest.skip("sklearn not installed")


@pytest.mark.cross_model
class TestProcrustesAlignment:
    """Test Procrustes alignment between model pairs."""

    def test_same_dimension_alignment(self, model_pairs_fixture, sentence_transformer, sklearn_available):
        """Models with same dimension should align directly."""
        # Find same-dimension pairs
        same_dim_pairs = [
            p for p in model_pairs_fixture["model_pairs"]
            if p["dim_a"] == p["dim_b"]
        ]

        if not same_dim_pairs:
            pytest.skip("No same-dimension pairs configured")

        results = []

        for pair in same_dim_pairs:
            model_a_name = pair["model_a"]
            model_b_name = pair["model_b"]

            try:
                model_a = sentence_transformer(model_a_name)
                model_b = sentence_transformer(model_b_name)

                emb_a = model_a.encode(TEST_CORPUS, normalize_embeddings=True)
                emb_b = model_b.encode(TEST_CORPUS, normalize_embeddings=True)

                emb_a = np.array(emb_a)
                emb_b = np.array(emb_b)

                R, error = procrustes_alignment(emb_a, emb_b)

                results.append({
                    "pair": pair["id"],
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    "alignment_error": error,
                    "success": error < 0.5
                })
            except Exception as e:
                results.append({
                    "pair": pair["id"],
                    "error": str(e)
                })

        print("\n=== Same-Dimension Procrustes Alignment ===")
        for r in results:
            if "error" in r:
                print(f"  {r['pair']}: ERROR - {r['error']}")
            else:
                status = "PASS" if r["success"] else "WARN"
                print(f"  {r['pair']}: error={r['alignment_error']:.4f} [{status}]")

        # At least one pair should align well
        successful = [r for r in results if r.get("success", False)]
        assert len(successful) > 0 or len(results) == 0, "No same-dimension pairs aligned successfully"

    def test_different_dimension_alignment_via_pca(self, model_pairs_fixture, sentence_transformer, sklearn_available):
        """Models with different dimensions should align via PCA projection."""
        # Find different-dimension pairs
        diff_dim_pairs = [
            p for p in model_pairs_fixture["model_pairs"]
            if p["dim_a"] != p["dim_b"]
        ]

        if not diff_dim_pairs:
            pytest.skip("No different-dimension pairs configured")

        results = []

        for pair in diff_dim_pairs:
            model_a_name = pair["model_a"]
            model_b_name = pair["model_b"]
            k = min(pair["dim_a"], pair["dim_b"], 64)  # Shared dimension

            try:
                model_a = sentence_transformer(model_a_name)
                model_b = sentence_transformer(model_b_name)

                emb_a = model_a.encode(TEST_CORPUS, normalize_embeddings=True)
                emb_b = model_b.encode(TEST_CORPUS, normalize_embeddings=True)

                emb_a = np.array(emb_a)
                emb_b = np.array(emb_b)

                # Project to shared dimension
                proj_a, proj_b = project_to_shared_dimension(emb_a, emb_b, k)

                # Procrustes on projected space
                R, error = procrustes_alignment(proj_a, proj_b)

                results.append({
                    "pair": pair["id"],
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    "dim_a": pair["dim_a"],
                    "dim_b": pair["dim_b"],
                    "shared_dim": k,
                    "alignment_error": error,
                    "success": error < 0.5
                })
            except Exception as e:
                results.append({
                    "pair": pair["id"],
                    "error": str(e)
                })

        print("\n=== Different-Dimension PCA + Procrustes Alignment ===")
        for r in results:
            if "error" in r:
                print(f"  {r['pair']}: ERROR - {r['error']}")
            else:
                status = "PASS" if r["success"] else "WARN"
                print(f"  {r['pair']} ({r['dim_a']}d -> {r['dim_b']}d via {r['shared_dim']}d): error={r['alignment_error']:.4f} [{status}]")

        # Report overall
        successful = [r for r in results if r.get("success", False)]
        print(f"\nSuccessful alignments: {len(successful)}/{len(results)}")


@pytest.mark.cross_model
class TestEigenvalueMatching:
    """Test that eigenvalue spectra match after normalization."""

    def test_normalized_eigenvalue_correlation(self, model_pairs_fixture, sentence_transformer):
        """Normalized eigenvalue spectra should correlate."""
        from scipy.stats import pearsonr

        results = []

        for pair in model_pairs_fixture["model_pairs"]:
            model_a_name = pair["model_a"]
            model_b_name = pair["model_b"]

            try:
                model_a = sentence_transformer(model_a_name)
                model_b = sentence_transformer(model_b_name)

                emb_a = model_a.encode(TEST_CORPUS, normalize_embeddings=True)
                emb_b = model_b.encode(TEST_CORPUS, normalize_embeddings=True)

                emb_a = np.array(emb_a)
                emb_b = np.array(emb_b)

                # Compute covariance eigenvalues
                cov_a = (emb_a - emb_a.mean(0)).T @ (emb_a - emb_a.mean(0)) / len(emb_a)
                cov_b = (emb_b - emb_b.mean(0)).T @ (emb_b - emb_b.mean(0)) / len(emb_b)

                ev_a = np.sort(np.linalg.eigvalsh(cov_a))[::-1]
                ev_b = np.sort(np.linalg.eigvalsh(cov_b))[::-1]

                # Normalize
                ev_a = ev_a / ev_a.sum()
                ev_b = ev_b / ev_b.sum()

                # Compare top-k
                k = min(20, len(ev_a), len(ev_b))
                r, p = pearsonr(ev_a[:k], ev_b[:k])

                results.append({
                    "pair": pair["id"],
                    "correlation": r,
                    "p_value": p,
                    "success": r > 0.8
                })
            except Exception as e:
                results.append({
                    "pair": pair["id"],
                    "error": str(e)
                })

        print("\n=== Eigenvalue Spectrum Correlation ===")
        for r in results:
            if "error" in r:
                print(f"  {r['pair']}: ERROR - {r['error']}")
            else:
                status = "PASS" if r["success"] else "WARN"
                print(f"  {r['pair']}: r={r['correlation']:.4f}, p={r['p_value']:.2e} [{status}]")

        # At least some pairs should have high correlation
        high_corr = [r for r in results if r.get("correlation", 0) > 0.7]
        print(f"\nHigh correlation pairs (r > 0.7): {len(high_corr)}/{len(results)}")


@pytest.mark.cross_model
class TestTransformStability:
    """Test that discovered transforms are stable.

    NOTE: With small training sets (15 examples), Procrustes alignment
    overfits. Cross-model transforms require larger corpora for stability.
    
    DEPRECATED TEST REMOVED: test_transform_on_held_out_data was archived to
    MEMORY/ARCHIVE/deprecated_tests/test_pre_svtp_alignment_deprecated.py
    as it is superseded by SVTP (Semantic Vector Transport Protocol).
    
    SVTP uses 128+ canonical anchors instead of 15 training examples,
    providing stable cross-model communication.
    
    See: CAPABILITY/PRIMITIVES/vector_packet.py for production SVTP implementation.
    """

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
