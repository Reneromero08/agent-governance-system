"""
Manifold addressing for text: prove 100x compression
"""
import numpy as np

# 10 sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming every industry",
    "Python is a great programming language",
    "The weather today is sunny and warm",
    "I love coffee in the morning",
    "Mathematics is the language of the universe",
    "Compression reduces storage requirements",
    "Neural networks learn patterns from data",
    "The cat sat on the mat",
    "Artificial intelligence will change everything"
]

# Use sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, fast
    embeddings = model.encode(sentences)
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.run(['pip', 'install', 'sentence-transformers', '-q'])
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")
print(f"Original size per sentence: {embeddings.shape[1] * 4} bytes (float32)")

# Measure Df using participation ratio
def participation_ratio(X):
    """Df = (Σλ)² / Σλ²"""
    centered = X - X.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

Df = participation_ratio(embeddings)
print(f"\nDf (effective dimensionality): {Df:.1f}")

# SVD to find principal components
U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(axis=0), full_matrices=False)

# Compress to k dimensions
k = max(int(np.ceil(Df)), 2)
print(f"Using k = {k} dimensions")

# Project to k dimensions
compressed = (embeddings - embeddings.mean(axis=0)) @ Vt[:k].T

# Storage calculation
original_bytes = len(sentences) * embeddings.shape[1] * 4  # 10 * 384 * 4
compressed_bytes = len(sentences) * k * 2  # using float16
basis_bytes = k * embeddings.shape[1] * 2  # basis vectors (shared)
mean_bytes = embeddings.shape[1] * 2  # mean vector (shared)

total_compressed = compressed_bytes + basis_bytes + mean_bytes

print(f"\n--- Storage ---")
print(f"Original: {original_bytes:,} bytes")
print(f"Compressed coefficients: {compressed_bytes} bytes")
print(f"Basis (shared): {basis_bytes} bytes")
print(f"Mean (shared): {mean_bytes} bytes")
print(f"Total compressed: {total_compressed} bytes")
print(f"Compression ratio: {original_bytes / total_compressed:.1f}x")

# Verify reconstruction quality
reconstructed = compressed @ Vt[:k] + embeddings.mean(axis=0)
mse = ((embeddings - reconstructed) ** 2).mean()
print(f"\nReconstruction MSE: {mse:.6f}")

# Verify we can identify sentences from compressed form
print(f"\n--- Identification Test ---")
for i, sent in enumerate(sentences):
    # Find nearest neighbor in compressed space
    query = compressed[i]
    distances = np.linalg.norm(compressed - query, axis=1)
    distances[i] = np.inf  # exclude self
    nearest = np.argmin(distances)

    # Cosine similarity with original embeddings
    cos_sim = np.dot(embeddings[i], reconstructed[i]) / (
        np.linalg.norm(embeddings[i]) * np.linalg.norm(reconstructed[i])
    )
    print(f"[{i}] cos_sim={cos_sim:.4f} | {sent[:40]}...")

# Per-sentence compression if we only store coefficients
print(f"\n--- Per-Sentence (if basis is universal) ---")
per_sentence_bytes = k * 2  # just the k coefficients in float16
original_per_sentence = np.mean([len(s) for s in sentences])
print(f"Coefficients per sentence: {per_sentence_bytes} bytes")
print(f"Average sentence length: {original_per_sentence:.0f} characters")
print(f"Compression vs text: {original_per_sentence / per_sentence_bytes:.1f}x")

# The BIG compression: if we have a diffuser
print(f"\n--- With Manifold Diffuser (theoretical) ---")
print(f"Df = {Df:.1f} dimensions needed")
print(f"At 16-bit precision: {int(Df) * 2} bytes per sentence")
print(f"This could address ANY sentence on the learned manifold")
