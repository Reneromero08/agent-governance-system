"""
Holographic Projector on Text

Learn projector from text embeddings.
Store addresses, render meaning.
"""
import numpy as np
from projector import HolographicProjector
from sentence_transformers import SentenceTransformer

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims

# Training corpus - learn the projector
training_texts = [
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
    "Science explores the mysteries of nature",
    "Art expresses the depths of human emotion",
    "Technology connects people across the globe",
    "History teaches us lessons from the past",
    "Philosophy questions the meaning of existence",
] * 4  # Repeat for more training data

print(f"Training on {len(training_texts)} texts...")
train_embeddings = model.encode(training_texts)
print(f"Embedding shape: {train_embeddings.shape}")

# Learn the projector
projector = HolographicProjector()
projector.learn(train_embeddings)

# Test on NEW sentences (not in training)
test_texts = [
    "The algorithm processes information efficiently",
    "Stars shine brightly in the night sky",
    "Programming requires logical thinking skills",
    "Love is the most powerful force in the universe",
    "Water flows downhill following gravity",
]

print(f"\n{'='*60}")
print("TESTING ON NEW SENTENCES")
print(f"{'='*60}")

test_embeddings = model.encode(test_texts)

for i, text in enumerate(test_texts):
    # Encode to address
    address = projector.encode(test_embeddings[i])

    # Render back
    reconstructed = projector.render(address)

    # Fidelity
    fidelity = np.dot(test_embeddings[i], reconstructed) / (
        np.linalg.norm(test_embeddings[i]) * np.linalg.norm(reconstructed)
    )

    # Storage
    text_bytes = len(text.encode('utf-8'))
    address_bytes = projector.Df * 2  # float16

    print(f"\nText: '{text}'")
    print(f"  Text size: {text_bytes} bytes")
    print(f"  Address size: {address_bytes} bytes ({projector.Df} floats)")
    print(f"  Compression vs text: {text_bytes / address_bytes:.1f}x")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  Address: [{', '.join(f'{a:.2f}' for a in address[:5])}...]")

# Find nearest neighbor from address
print(f"\n{'='*60}")
print("NEAREST NEIGHBOR RETRIEVAL FROM ADDRESS")
print(f"{'='*60}")

# Build index of training embeddings
train_emb_normalized = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

for i, text in enumerate(test_texts):
    address = projector.encode(test_embeddings[i])
    reconstructed = projector.render(address)
    recon_normalized = reconstructed / np.linalg.norm(reconstructed)

    # Find nearest training text
    similarities = train_emb_normalized @ recon_normalized
    best_idx = np.argmax(similarities)

    print(f"\nQuery: '{text}'")
    print(f"  Nearest: '{training_texts[best_idx]}'")
    print(f"  Similarity: {similarities[best_idx]:.4f}")

# The key insight
print(f"\n{'='*60}")
print("THE FORMULA IN ACTION")
print(f"{'='*60}")
print(f"Embedding dim: 384")
print(f"Address dim (Df): {projector.Df}")
print(f"Compression: {384 / projector.Df:.1f}x")
print(f"\nR = address @ basis + mean")
print(f"  address: {projector.Df} numbers (the manifold coordinates)")
print(f"  basis: {projector.Df} x 384 (the projector sigma(f)^Df)")
print(f"  R: 384 numbers (the rendered embedding)")
print(f"\nStore {projector.Df} numbers. Render meaning on demand.")
