"""
Direct text compression using Df formula
No embeddings, just bytes → compressed bytes
"""
import numpy as np

# 10 sentences as raw text
texts = [
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

# Convert to byte matrix (pad to same length)
max_len = max(len(t) for t in texts)
byte_matrix = np.zeros((len(texts), max_len), dtype=np.float32)
for i, t in enumerate(texts):
    byte_matrix[i, :len(t)] = [ord(c) for c in t]

print(f"Original: {len(texts)} texts, {max_len} bytes max")
print(f"Matrix shape: {byte_matrix.shape}")
original_bytes = sum(len(t) for t in texts)
print(f"Total raw bytes: {original_bytes}")

# Df formula
def participation_ratio(X):
    centered = X - X.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

Df = participation_ratio(byte_matrix)
print(f"\nDf = {Df:.1f}")

# SVD compress
mean = byte_matrix.mean(axis=0)
U, S, Vt = np.linalg.svd(byte_matrix - mean, full_matrices=False)

# Keep k = ceil(Df) components
k = max(int(np.ceil(Df)), 1)
print(f"Using k = {k} components")

# Compressed representation
coefficients = U[:, :k] * S[:k]  # n_texts × k

# Storage
coef_bytes = coefficients.size * 2  # float16
basis_bytes = Vt[:k].size * 2  # k × max_len
mean_bytes = mean.size * 2

total = coef_bytes + basis_bytes + mean_bytes
print(f"\n--- Storage ---")
print(f"Coefficients: {coef_bytes} bytes")
print(f"Basis: {basis_bytes} bytes")
print(f"Mean: {mean_bytes} bytes")
print(f"Total: {total} bytes")
print(f"Compression: {original_bytes / total:.2f}x")

# Reconstruct and verify
reconstructed = (coefficients @ Vt[:k]) + mean
errors = []
for i, t in enumerate(texts):
    recon_bytes = reconstructed[i, :len(t)]
    recon_chars = ''.join(chr(max(0, min(127, int(round(b))))) for b in recon_bytes)
    match = sum(a == b for a, b in zip(t, recon_chars)) / len(t)
    errors.append(1 - match)
    print(f"\n[{i}] {match*100:.0f}% match")
    print(f"  Original: {t[:50]}")
    print(f"  Recon:    {recon_chars[:50]}")

print(f"\nAverage accuracy: {(1-np.mean(errors))*100:.1f}%")
