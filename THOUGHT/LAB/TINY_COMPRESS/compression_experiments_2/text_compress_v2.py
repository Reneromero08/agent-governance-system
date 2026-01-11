"""
Text compression - find data with actual redundancy
"""
import numpy as np

# Take a real document with repetition/structure
document = """
The cat sat on the mat. The cat was happy.
The dog ran in the park. The dog was fast.
The bird flew in the sky. The bird was free.
The fish swam in the sea. The fish was calm.
The cat sat on the mat. The cat was happy.
The dog ran in the park. The dog was fast.
The bird flew in the sky. The bird was free.
The fish swam in the sea. The fish was calm.
""" * 10  # Repeat for more data

print(f"Document size: {len(document)} bytes")

# Convert to overlapping 8-byte windows (like image patches)
window_size = 8
windows = []
for i in range(0, len(document) - window_size, 1):
    window = [ord(c) for c in document[i:i+window_size]]
    windows.append(window)

data = np.array(windows, dtype=np.float32)
print(f"Windows: {data.shape[0]} × {window_size} bytes")

# Df formula
centered = data - data.mean(axis=0)
cov = np.cov(centered.T)
eigenvalues = np.linalg.eigvalsh(cov)
eigenvalues = eigenvalues[eigenvalues > 1e-10]
Df = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
print(f"\nDf = {Df:.2f}")

# SVD
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
k = max(int(np.ceil(Df)), 1)
print(f"Using k = {k} components")

# Vector Quantization on projected windows
from sklearn.cluster import MiniBatchKMeans

# Project to k dimensions
projected = centered @ Vt[:k].T

# Cluster into archetypes
n_clusters = 32  # Small codebook
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
labels = kmeans.fit_predict(projected)
codebook = kmeans.cluster_centers_

# Storage calculation
label_bits = int(np.ceil(np.log2(n_clusters)))  # bits per label
label_bytes = (len(labels) * label_bits + 7) // 8
codebook_bytes = codebook.size * 2  # float16
basis_bytes = Vt[:k].size * 2
mean_bytes = data.mean(axis=0).size * 2

total = label_bytes + codebook_bytes + basis_bytes + mean_bytes

print(f"\n--- Storage ---")
print(f"Original: {len(document)} bytes")
print(f"Labels ({label_bits} bits each): {label_bytes} bytes")
print(f"Codebook ({n_clusters} × {k}): {codebook_bytes} bytes")
print(f"Basis ({k} × {window_size}): {basis_bytes} bytes")
print(f"Mean: {mean_bytes} bytes")
print(f"Total: {total} bytes")
print(f"Compression: {len(document) / total:.1f}x")

# Reconstruct
reconstructed_proj = codebook[labels]
reconstructed = (reconstructed_proj @ Vt[:k]) + data.mean(axis=0)

# Check reconstruction quality
errors = np.abs(data - reconstructed)
print(f"\nMean absolute error per byte: {errors.mean():.2f}")
print(f"Max error: {errors.max():.2f}")

# Show some reconstructions
print(f"\n--- Sample Reconstructions ---")
for i in range(0, 50, 10):
    orig = ''.join(chr(int(c)) for c in data[i])
    recon = ''.join(chr(max(32, min(126, int(round(c))))) for c in reconstructed[i])
    print(f"'{orig}' → '{recon}'")
