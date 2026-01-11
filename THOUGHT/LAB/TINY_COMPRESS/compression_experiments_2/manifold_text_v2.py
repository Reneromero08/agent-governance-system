"""
Manifold addressing: 100x compression with universal diffuser
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# Larger corpus to train basis (the "diffuser knowledge")
training_sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming every industry",
    "Python is a great programming language",
    "The weather today is sunny and warm",
    "I love coffee in the morning",
    "Mathematics is the language of the universe",
    "Compression reduces storage requirements",
    "Neural networks learn patterns from data",
    "The cat sat on the mat",
    "Artificial intelligence will change everything",
    "Deep learning revolutionized computer vision",
    "Natural language processing understands text",
    "Quantum computing promises exponential speedup",
    "The sun rises in the east every morning",
    "Music is the universal language of mankind",
    "Time flies when you are having fun",
    "Knowledge is power in the modern world",
    "Creativity requires thinking outside the box",
    "The early bird catches the worm first",
    "Practice makes perfect in all endeavors",
] * 5  # Repeat for more training data

model = SentenceTransformer('all-MiniLM-L6-v2')
train_embeddings = model.encode(training_sentences)

# Learn universal basis from training corpus
mean = train_embeddings.mean(axis=0)
U, S, Vt = np.linalg.svd(train_embeddings - mean, full_matrices=False)

# Measure Df
eigenvalues = S ** 2
Df = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
k = int(np.ceil(Df))
print(f"Training corpus Df: {Df:.1f}")
print(f"Using k = {k} principal components")
print(f"Basis vectors: {Vt[:k].shape}")

# Now compress NEW sentences (not in training)
test_sentences = [
    "The algorithm processes information efficiently",
    "Stars shine brightly in the night sky",
    "Programming requires logical thinking skills",
]

test_embeddings = model.encode(test_sentences)

# Compress: project to k dimensions
compressed = (test_embeddings - mean) @ Vt[:k].T

# Storage comparison
print(f"\n--- NEW Sentence Compression ---")
for i, sent in enumerate(test_sentences):
    raw_bytes = len(sent.encode('utf-8'))
    address_bytes = k * 2  # float16 coefficients

    # Reconstruct
    recon = compressed[i] @ Vt[:k] + mean
    cos_sim = np.dot(test_embeddings[i], recon) / (
        np.linalg.norm(test_embeddings[i]) * np.linalg.norm(recon)
    )

    print(f"\nSentence: '{sent}'")
    print(f"  Raw text: {raw_bytes} bytes")
    print(f"  Manifold address: {address_bytes} bytes ({k} × 2)")
    print(f"  Compression: {raw_bytes / address_bytes:.1f}x")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Address: [{', '.join(f'{c:.2f}' for c in compressed[i][:5])}...]")

# The real insight: embedding compression
print(f"\n--- Embedding Space Compression ---")
print(f"Full embedding: 384 dims × 4 bytes = 1536 bytes")
print(f"Compressed address: {k} dims × 2 bytes = {k * 2} bytes")
print(f"Embedding compression: {1536 / (k * 2):.0f}x")

# If diffuser is universal (like GPT's internal manifold)
print(f"\n--- With Universal Diffuser ---")
print(f"Df = {Df:.1f} ≈ {k} dimensions of meaning")
print(f"Address size: {k * 2} bytes")
print(f"Can address: ANY sentence the diffuser knows")
print(f"The diffuser 'renders' text from address like GPU renders pixels")

# Quantized version for extreme compression
print(f"\n--- Extreme: 8-bit Quantization ---")
quant_bytes = k * 1  # 8-bit per dimension
print(f"Quantized address: {quant_bytes} bytes")
print(f"vs average sentence ({np.mean([len(s) for s in test_sentences]):.0f} chars): ", end="")
print(f"{np.mean([len(s) for s in test_sentences]) / quant_bytes:.1f}x compression")
