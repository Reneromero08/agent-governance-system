"""Anchor Set Analysis and Optimization.

Analyzes anchor sets to find optimal words for vector communication.
The goal is to find anchors that:
1. Maximally span the semantic space (high effective rank)
2. Cover all 8 pinwheel sectors (octants)
3. Have stable relative distances across models

Key insight from Q51: Semantic space has 8 octants that map to 8 phase
sectors in the complex plane representation. Optimal anchors should
have good coverage of all 8 sectors.

Usage:
    analyzer = AnchorAnalyzer(embed_fn)
    analysis = analyzer.analyze(CANONICAL_128)
    print(analysis.report())

    # Find which sector each anchor occupies
    sectors = analyzer.get_sector_distribution(CANONICAL_128)

    # Suggest optimized anchor set
    optimized = analyzer.optimize(CANONICAL_128, target_k=64)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
from scipy.stats import spearmanr

from .mds import squared_distance_matrix, classical_mds, effective_rank
from .canonical_anchors import CANONICAL_128


@dataclass
class AnchorAnalysis:
    """Results of anchor set analysis."""
    anchors: List[str]
    n_anchors: int
    embedding_dim: int

    # MDS analysis
    eigenvalues: np.ndarray
    effective_rank: float
    explained_variance: List[float]  # Cumulative at k=8, 16, 32, 48

    # Sector coverage (pinwheel)
    sector_counts: List[int]  # Count per sector (0-7)
    sector_balance: float  # Entropy / max_entropy (1.0 = perfectly balanced)
    uncovered_sectors: List[int]

    # Word-level analysis
    sector_assignments: Dict[str, int]  # word -> sector
    outliers: List[str]  # Words far from others

    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "ANCHOR SET ANALYSIS",
            "=" * 60,
            "",
            f"Anchors: {self.n_anchors}",
            f"Embedding dimension: {self.embedding_dim}",
            "",
            "--- MDS Analysis ---",
            f"Effective rank: {self.effective_rank:.2f}",
            f"Explained variance:",
            f"  k=8:  {self.explained_variance[0]*100:.1f}%",
            f"  k=16: {self.explained_variance[1]*100:.1f}%",
            f"  k=32: {self.explained_variance[2]*100:.1f}%",
            f"  k=48: {self.explained_variance[3]*100:.1f}%",
            "",
            "--- Sector Coverage (Pinwheel) ---",
            f"Sector balance: {self.sector_balance:.3f} (1.0 = perfect)",
            "Sector distribution:",
        ]

        for i, count in enumerate(self.sector_counts):
            bar = "*" * (count // 2)
            lines.append(f"  Sector {i}: {count:3d} {bar}")

        if self.uncovered_sectors:
            lines.append(f"\nUncovered sectors: {self.uncovered_sectors}")
        else:
            lines.append("\nAll 8 sectors covered!")

        if self.outliers:
            lines.append(f"\nOutliers (far from centroid): {self.outliers[:5]}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class AnchorAnalyzer:
    """Analyzes and optimizes anchor sets."""

    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray]):
        """Initialize with an embedding function.

        Args:
            embed_fn: Function that takes List[str] and returns (n, dim) embeddings
        """
        self.embed_fn = embed_fn

    def _get_embeddings(self, anchors: List[str]) -> np.ndarray:
        """Get normalized embeddings."""
        embeddings = self.embed_fn(anchors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _compute_sectors(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign each embedding to a sector (0-7) using 3D PCA signs.

        The pinwheel structure maps the 8 octants (sign combinations of
        PC1, PC2, PC3) to 8 phase sectors.
        """
        # Center
        centered = embeddings - embeddings.mean(axis=0)

        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Project to top 3 PCs
        proj = centered @ Vt[:3].T  # (n, 3)

        # Octant = sign pattern
        # Octant k = (sign(PC1), sign(PC2), sign(PC3)) encoded as binary
        signs = (proj > 0).astype(int)  # 0 or 1 for each component
        octants = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]

        return octants

    def analyze(self, anchors: List[str]) -> AnchorAnalysis:
        """Analyze an anchor set.

        Args:
            anchors: List of anchor words

        Returns:
            AnchorAnalysis with detailed metrics
        """
        n = len(anchors)

        # Get embeddings
        embeddings = self._get_embeddings(anchors)
        dim = embeddings.shape[1]

        # MDS
        D2 = squared_distance_matrix(embeddings)
        _, eigenvalues, _ = classical_mds(D2, k=min(n-1, 64))

        # Effective rank
        eff_rank = effective_rank(eigenvalues)

        # Explained variance at different k
        total_var = eigenvalues.sum()
        explained = []
        for k in [8, 16, 32, 48]:
            if k <= len(eigenvalues):
                explained.append(eigenvalues[:k].sum() / total_var)
            else:
                explained.append(1.0)

        # Sector assignment
        sectors = self._compute_sectors(embeddings)

        # Sector counts
        sector_counts = [0] * 8
        for s in sectors:
            sector_counts[s] += 1

        # Sector balance (entropy-based)
        probs = np.array(sector_counts) / n
        probs = probs[probs > 0]  # Remove zeros for log
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(8)  # Perfect balance
        sector_balance = entropy / max_entropy

        # Uncovered sectors
        uncovered = [i for i, c in enumerate(sector_counts) if c == 0]

        # Word -> sector mapping
        sector_assignments = {anchors[i]: int(sectors[i]) for i in range(n)}

        # Find outliers (words far from centroid in MDS space)
        _, _, eigenvectors = classical_mds(D2, k=min(n-1, 16))
        coords = eigenvectors[:, :16] * np.sqrt(eigenvalues[:16])
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        outlier_idx = np.argsort(distances)[-5:][::-1]
        outliers = [anchors[i] for i in outlier_idx]

        return AnchorAnalysis(
            anchors=anchors,
            n_anchors=n,
            embedding_dim=dim,
            eigenvalues=eigenvalues,
            effective_rank=eff_rank,
            explained_variance=explained,
            sector_counts=sector_counts,
            sector_balance=sector_balance,
            uncovered_sectors=uncovered,
            sector_assignments=sector_assignments,
            outliers=outliers
        )

    def get_sector_distribution(self, anchors: List[str]) -> Dict[int, List[str]]:
        """Get which words are in each sector.

        Args:
            anchors: List of anchor words

        Returns:
            Dict mapping sector (0-7) to list of words in that sector
        """
        embeddings = self._get_embeddings(anchors)
        sectors = self._compute_sectors(embeddings)

        distribution = {i: [] for i in range(8)}
        for i, word in enumerate(anchors):
            distribution[sectors[i]].append(word)

        return distribution

    def suggest_additions(
        self,
        current_anchors: List[str],
        candidates: List[str],
        target_count: int = 16
    ) -> List[Tuple[str, int, float]]:
        """Suggest words to add to improve sector balance.

        Args:
            current_anchors: Current anchor set
            candidates: Potential words to add
            target_count: How many suggestions to return

        Returns:
            List of (word, target_sector, improvement_score)
        """
        # Analyze current
        analysis = self.analyze(current_anchors)

        # Find underrepresented sectors
        min_count = min(analysis.sector_counts)
        underrep = [i for i, c in enumerate(analysis.sector_counts)
                    if c <= min_count + 1]

        # Get candidate embeddings
        all_words = current_anchors + candidates
        all_embeddings = self._get_embeddings(all_words)

        # Get sectors for all
        all_sectors = self._compute_sectors(all_embeddings)

        # Score candidates by whether they fill underrepresented sectors
        suggestions = []
        for i, word in enumerate(candidates):
            idx = len(current_anchors) + i
            sector = all_sectors[idx]

            if sector in underrep:
                # Higher score for more underrepresented sectors
                deficit = max(analysis.sector_counts) - analysis.sector_counts[sector]
                score = deficit / max(analysis.sector_counts)
                suggestions.append((word, int(sector), score))

        # Sort by score
        suggestions.sort(key=lambda x: -x[2])

        return suggestions[:target_count]

    def optimize(
        self,
        anchors: List[str],
        target_k: int = 64,
        iterations: int = 100
    ) -> List[str]:
        """Optimize anchor set for better sector balance.

        Uses greedy selection to pick anchors that maximize sector coverage.

        Args:
            anchors: Initial anchor set
            target_k: Target number of anchors
            iterations: Number of optimization iterations

        Returns:
            Optimized subset of anchors
        """
        if target_k >= len(anchors):
            return anchors

        embeddings = self._get_embeddings(anchors)
        sectors = self._compute_sectors(embeddings)

        # Greedy selection: pick one from each sector first
        selected_idx = []
        for sector in range(8):
            candidates = [i for i in range(len(anchors))
                          if sectors[i] == sector and i not in selected_idx]
            if candidates:
                # Pick the one closest to sector centroid
                selected_idx.append(candidates[0])

        # Fill remaining slots with diverse picks
        while len(selected_idx) < target_k:
            remaining = [i for i in range(len(anchors)) if i not in selected_idx]
            if not remaining:
                break

            # Pick word that maximizes minimum distance to already selected
            best_idx = None
            best_min_dist = -1

            for i in remaining:
                min_dist = float('inf')
                for j in selected_idx:
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    min_dist = min(min_dist, dist)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx is not None:
                selected_idx.append(best_idx)
            else:
                break

        return [anchors[i] for i in selected_idx]

    def cross_model_stability(
        self,
        anchors: List[str],
        embed_fns: List[Callable[[List[str]], np.ndarray]],
        model_names: List[str]
    ) -> Dict:
        """Analyze anchor stability across multiple models.

        Args:
            anchors: Anchor set to analyze
            embed_fns: List of embedding functions
            model_names: Names for each model

        Returns:
            Dict with stability metrics
        """
        n_models = len(embed_fns)

        # Get distance matrices for each model
        D2_matrices = []
        for embed_fn in embed_fns:
            embeddings = embed_fn(anchors)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            D2 = squared_distance_matrix(embeddings)
            D2_matrices.append(D2)

        # Pairwise correlations
        correlations = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i+1, n_models):
                flat_i = D2_matrices[i].flatten()
                flat_j = D2_matrices[j].flatten()
                corr, _ = spearmanr(flat_i, flat_j)
                correlations[i, j] = corr
                correlations[j, i] = corr

        # Per-word stability
        word_stability = []
        for w_idx, word in enumerate(anchors):
            # Get distances from this word to all others across models
            distances = []
            for D2 in D2_matrices:
                distances.append(D2[w_idx])
            distances = np.array(distances)

            # Stability = correlation across models
            if n_models > 1:
                corrs = []
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        c, _ = spearmanr(distances[i], distances[j])
                        corrs.append(c)
                word_stability.append((word, np.mean(corrs)))
            else:
                word_stability.append((word, 1.0))

        # Sort by stability
        word_stability.sort(key=lambda x: -x[1])

        return {
            'model_names': model_names,
            'correlation_matrix': correlations.tolist(),
            'mean_correlation': float(np.mean(correlations[np.triu_indices(n_models, k=1)])),
            'word_stability': word_stability,
            'most_stable': word_stability[:10],
            'least_stable': word_stability[-10:]
        }


def demo_analysis():
    """Demo anchor analysis."""
    print("=" * 60)
    print("ANCHOR SET ANALYSIS DEMO")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embed_fn = model.encode
    except ImportError:
        print("sentence-transformers not installed, using mock embeddings")
        def embed_fn(texts):
            np.random.seed(42)
            return np.random.randn(len(texts), 384)

    analyzer = AnchorAnalyzer(embed_fn)

    # Analyze canonical set
    print("\nAnalyzing CANONICAL_128...")
    analysis = analyzer.analyze(CANONICAL_128)
    print(analysis.report())

    # Show sector distribution
    print("\nSector Distribution:")
    dist = analyzer.get_sector_distribution(CANONICAL_128)
    for sector, words in dist.items():
        print(f"  Sector {sector}: {words[:5]}..." if len(words) > 5 else f"  Sector {sector}: {words}")

    # Optimize to 64 anchors
    print("\nOptimizing to 64 anchors...")
    optimized = analyzer.optimize(CANONICAL_128, target_k=64)
    print(f"Selected {len(optimized)} anchors")

    opt_analysis = analyzer.analyze(optimized)
    print(f"Optimized sector balance: {opt_analysis.sector_balance:.3f}")
    print(f"Original sector balance: {analysis.sector_balance:.3f}")


if __name__ == "__main__":
    demo_analysis()
