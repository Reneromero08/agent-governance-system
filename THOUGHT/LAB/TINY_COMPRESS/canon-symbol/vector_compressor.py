#!/usr/bin/env python3
"""
Vector Compressor - Compress canon embeddings using Q34 spectral method

Takes existing embeddings from canon_index.db (384 dims) and compresses
them to k dimensions (typically 9) using proven Df discovery.

This is practical compression you can use NOW with any LLM API:
- Query with compressed vectors (42x smaller)
- Retrieve matching canon files
- Send to Minimax/Claude/GPT-4

Usage:
    python vector_compressor.py --compress
    python vector_compressor.py --query "catalytic computing"
    python vector_compressor.py --stats
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class CompressionConfig:
    """Configuration for vector compression"""
    original_dims: int
    compressed_dims: int  # k from Q34
    effective_rank: float  # Df
    compression_ratio: float
    variance_captured: float  # % variance in k dims
    eigenvalues: List[float]
    cumulative_variance: List[float]


class VectorCompressor:
    """Compress embeddings using Q34 spectral method"""

    def __init__(self, repo_root: str = None):
        if repo_root is None:
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "NAVIGATION" / "CORTEX").exists():
                    repo_root = str(current)
                    break
                current = current.parent
            else:
                raise RuntimeError("Could not find repo root")

        self.repo_root = Path(repo_root)
        self.canon_db = self.repo_root / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"
        self.output_dir = self.repo_root / "NAVIGATION" / "CORTEX" / "_generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compressed_db = self.output_dir / "canon_compressed.db"
        self.config_path = self.output_dir / "compression_config.json"

        self.projection_matrix: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.config: Optional[CompressionConfig] = None

    def load_embeddings(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load embeddings from canon_index.db"""
        if not self.canon_db.exists():
            raise FileNotFoundError(f"Canon DB not found: {self.canon_db}")

        conn = sqlite3.connect(self.canon_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding, file_path, id
            FROM canon_records
            WHERE embedding IS NOT NULL
            ORDER BY file_path
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise ValueError("No embeddings found in canon_index.db")

        # Deserialize embeddings
        embeddings = []
        file_paths = []
        ids = []

        for emb_blob, file_path, record_id in rows:
            # Embeddings stored as float32 blobs
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            embeddings.append(emb)
            file_paths.append(file_path)
            ids.append(record_id)

        embeddings_array = np.vstack(embeddings)

        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Shape: {embeddings_array.shape}")

        return embeddings_array, file_paths, ids

    def compute_spectrum(self, embeddings: np.ndarray) -> CompressionConfig:
        """Compute spectrum and determine optimal k (Q34 method)"""
        # Center embeddings
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # Covariance eigendecomposition
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # Descending order
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter numerical noise

        # Compute Df (effective rank) - Q34 formula
        df = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

        # Cumulative variance
        total_var = eigenvalues.sum()
        cumulative_var = np.cumsum(eigenvalues) / total_var

        # Find k for 95% variance (typical threshold)
        k = np.searchsorted(cumulative_var, 0.95) + 1
        k = max(k, int(np.ceil(df)))  # At least ceil(Df)

        print(f"\n=== Spectrum Analysis ===")
        print(f"Effective rank (Df): {df:.2f}")
        print(f"Recommended k (95% variance): {k}")
        print(f"Compression ratio: {embeddings.shape[1] / k:.1f}x")

        print(f"\nCumulative variance:")
        for i in [1, 2, 3, 5, 9, 10, 15, 20]:
            if i <= len(cumulative_var):
                print(f"  k={i:2d}: {cumulative_var[i-1]:.1%}")

        return CompressionConfig(
            original_dims=embeddings.shape[1],
            compressed_dims=k,
            effective_rank=df,
            compression_ratio=embeddings.shape[1] / k,
            variance_captured=cumulative_var[k-1] if k <= len(cumulative_var) else 1.0,
            eigenvalues=eigenvalues.tolist(),
            cumulative_variance=cumulative_var.tolist()
        )

    def compute_projection_matrix(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PCA projection matrix to k dimensions"""
        # Center
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # SVD for projection matrix
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Projection matrix: first k principal components
        projection = Vt[:k, :]  # Shape: (k, original_dims)

        return projection, mean

    def compress_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress embeddings using projection matrix"""
        if self.projection_matrix is None or self.mean is None:
            raise RuntimeError("Must compute projection matrix first")

        # Center and project
        centered = embeddings - self.mean
        compressed = centered @ self.projection_matrix.T  # (n, k)

        return compressed

    def decompress_embeddings(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress embeddings (lossy reconstruction)"""
        if self.projection_matrix is None or self.mean is None:
            raise RuntimeError("Must compute projection matrix first")

        # Project back to original space
        reconstructed = compressed @ self.projection_matrix + self.mean

        return reconstructed

    def compress(self, target_k: Optional[int] = None):
        """Compress canon embeddings and save to database"""
        print("=== Canon Vector Compression ===\n")

        # Load embeddings
        embeddings, file_paths, ids = self.load_embeddings()

        # Analyze spectrum
        config = self.compute_spectrum(embeddings)

        # Use target k if specified, otherwise use computed k
        k = target_k if target_k is not None else config.compressed_dims

        # Compute projection matrix
        self.projection_matrix, self.mean = self.compute_projection_matrix(embeddings, k)

        # Update config with actual k
        config.compressed_dims = k
        config.compression_ratio = config.original_dims / k
        config.variance_captured = config.cumulative_variance[k-1] if k <= len(config.cumulative_variance) else 1.0

        self.config = config

        # Compress all embeddings
        compressed = self.compress_embeddings(embeddings)

        print(f"\n=== Compression Results ===")
        print(f"Original: {embeddings.shape[1]} dims")
        print(f"Compressed: {compressed.shape[1]} dims")
        print(f"Compression: {config.compression_ratio:.1f}x")
        print(f"Variance captured: {config.variance_captured:.1%}")

        # Test reconstruction
        reconstructed = self.decompress_embeddings(compressed)
        reconstruction_error = np.linalg.norm(embeddings - reconstructed, axis=1).mean()
        relative_error = reconstruction_error / np.linalg.norm(embeddings, axis=1).mean()

        print(f"\nReconstruction error: {relative_error:.4f} (relative)")

        # Save compressed embeddings to new database
        self._save_compressed_db(compressed, file_paths, ids)

        # Save config
        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        print(f"\nCompressed DB: {self.compressed_db}")
        print(f"Config: {self.config_path}")

        # Save projection matrix for future queries
        projection_path = self.output_dir / "projection_matrix.npz"
        np.savez(projection_path,
                 projection=self.projection_matrix,
                 mean=self.mean,
                 k=k,
                 original_dims=embeddings.shape[1])

        print(f"Projection matrix: {projection_path}")

    def _save_compressed_db(self, compressed: np.ndarray, file_paths: List[str], ids: List[str]):
        """Save compressed embeddings to SQLite"""
        conn = sqlite3.connect(self.compressed_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compressed_vectors (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                original_id TEXT NOT NULL
            )
        """)

        # Insert compressed vectors
        for i, (emb, file_path, original_id) in enumerate(zip(compressed, file_paths, ids)):
            emb_blob = emb.astype(np.float32).tobytes()

            cursor.execute("""
                INSERT OR REPLACE INTO compressed_vectors
                (id, file_path, embedding, dimensions, original_id)
                VALUES (?, ?, ?, ?, ?)
            """, (f"compressed_{i}", file_path, emb_blob, len(emb), original_id))

        conn.commit()
        conn.close()

        print(f"\nSaved {len(compressed)} compressed vectors")

    def load_projection(self):
        """Load saved projection matrix"""
        projection_path = self.output_dir / "projection_matrix.npz"

        if not projection_path.exists():
            raise FileNotFoundError("Projection matrix not found. Run --compress first.")

        data = np.load(projection_path)
        self.projection_matrix = data['projection']
        self.mean = data['mean']

        # Load config
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
            self.config = CompressionConfig(**config_dict)

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Query compressed database with embedding"""
        if self.projection_matrix is None:
            self.load_projection()

        # Compress query
        query_compressed = self.compress_embeddings(query_embedding.reshape(1, -1))[0]

        # Load compressed vectors
        conn = sqlite3.connect(self.compressed_db)
        cursor = conn.cursor()

        cursor.execute("SELECT file_path, embedding FROM compressed_vectors")
        rows = cursor.fetchall()
        conn.close()

        # Compute similarities
        results = []
        for file_path, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = np.dot(query_compressed, emb) / (np.linalg.norm(query_compressed) * np.linalg.norm(emb))
            results.append((file_path, float(similarity)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def stats(self):
        """Show compression statistics"""
        if not self.config_path.exists():
            print("No compression config found. Run --compress first.")
            return

        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)

        config = CompressionConfig(**config_dict)

        print("=== Vector Compression Statistics ===\n")
        print(f"Original dimensions: {config.original_dims}")
        print(f"Compressed dimensions: {config.compressed_dims}")
        print(f"Effective rank (Df): {config.effective_rank:.2f}")
        print(f"Compression ratio: {config.compression_ratio:.1f}x")
        print(f"Variance captured: {config.variance_captured:.1%}")

        print("\n=== Storage Savings ===")
        original_size = 32 * config.original_dims * 4  # 32 vectors, float32
        compressed_size = 32 * config.compressed_dims * 4

        print(f"Original: {original_size:,} bytes ({original_size / 1024:.1f} KB)")
        print(f"Compressed: {compressed_size:,} bytes ({compressed_size / 1024:.1f} KB)")
        print(f"Savings: {original_size - compressed_size:,} bytes ({(1 - compressed_size/original_size)*100:.1f}%)")

        print("\n=== Cumulative Variance (first 20 dims) ===")
        for i in range(min(20, len(config.cumulative_variance))):
            var = config.cumulative_variance[i]
            marker = " <-- k selected" if i == config.compressed_dims - 1 else ""
            print(f"k={i+1:2d}: {var:.1%}{marker}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compress canon vectors using Q34 spectral method")
    parser.add_argument('--compress', action='store_true', help="Compress embeddings")
    parser.add_argument('--stats', action='store_true', help="Show compression statistics")
    parser.add_argument('--k', type=int, help="Target compression dimension (default: auto)")

    args = parser.parse_args()

    compressor = VectorCompressor()

    if args.compress:
        compressor.compress(target_k=args.k)
    elif args.stats:
        compressor.stats()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
