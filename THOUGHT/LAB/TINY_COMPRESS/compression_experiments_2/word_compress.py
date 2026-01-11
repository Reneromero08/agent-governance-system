"""
Compress a word 100x and render it back EXACTLY.
"""
import numpy as np

# The projector: a learned vocabulary (the manifold of words)
vocabulary = [
    "word", "the", "is", "a", "to", "of", "and", "in", "that", "it",
    "for", "was", "on", "are", "as", "with", "his", "they", "be", "at",
    "one", "have", "this", "from", "by", "hot", "but", "some", "what", "there",
    "we", "can", "out", "other", "were", "all", "your", "when", "up", "use",
    "how", "said", "an", "each", "she", "which", "do", "their", "time", "if",
    "will", "way", "about", "many", "then", "them", "would", "write", "like", "so",
    "these", "her", "long", "make", "thing", "see", "him", "two", "has", "look",
    "more", "day", "could", "go", "come", "did", "my", "sound", "no", "most",
    "number", "who", "over", "know", "water", "than", "call", "first", "people", "may",
    "down", "side", "been", "now", "find", "any", "new", "work", "part", "take",
]

# Build projector: word -> index, index -> word
word_to_idx = {w: i for i, w in enumerate(vocabulary)}
idx_to_word = {i: w for i, w in enumerate(vocabulary)}

def encode(word: str) -> int:
    """Compress word to address (index)."""
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary")
    return word_to_idx[word]

def render(address: int) -> str:
    """Render exact word from address."""
    return idx_to_word[address]

# Test: compress "word"
test_word = "word"

# Encode
address = encode(test_word)

# Storage calculation
word_bytes = len(test_word.encode('utf-8'))  # 4 bytes
vocab_size = len(vocabulary)
bits_needed = int(np.ceil(np.log2(vocab_size)))  # bits to index vocabulary
address_bytes = (bits_needed + 7) // 8  # round up to bytes

print(f"Word: '{test_word}'")
print(f"Word size: {word_bytes} bytes ({word_bytes * 8} bits)")
print(f"Vocabulary size: {vocab_size}")
print(f"Bits needed for address: {bits_needed}")
print(f"Address: {address}")
print(f"Address size: {bits_needed} bits = {address_bytes} byte(s)")

# Render back
rendered = render(address)
print(f"\nRendered: '{rendered}'")
print(f"Exact match: {rendered == test_word}")

# Compression ratio
compression = (word_bytes * 8) / bits_needed
print(f"\nCompression: {word_bytes * 8} bits / {bits_needed} bits = {compression:.1f}x")

print("\n" + "="*60)
print("THE PROBLEM:")
print("="*60)
print(f"'word' = 4 bytes = 32 bits")
print(f"100x compression = 32/100 = 0.32 bits")
print(f"0.32 bits can only distinguish 2^0.32 = 1.25 things")
print(f"\nFor 100x compression on a 4-byte word, vocabulary must be ~1 word.")
print(f"That's not compression, that's just having one word.")

print("\n" + "="*60)
print("WHERE 100x WORKS:")
print("="*60)
# Longer text example
long_text = "the quick brown fox jumps over the lazy dog"
words = long_text.split()
addresses = [encode(w) if w in word_to_idx else -1 for w in words]

text_bytes = len(long_text.encode('utf-8'))
address_bits = len(words) * bits_needed
address_bytes_total = (address_bits + 7) // 8

print(f"Text: '{long_text}'")
print(f"Text size: {text_bytes} bytes")
print(f"Words: {len(words)}")
print(f"Address bits: {len(words)} x {bits_needed} = {address_bits} bits")
print(f"Address bytes: {address_bytes_total}")
print(f"Compression: {text_bytes / address_bytes_total:.1f}x")

# For 100x, need:
target_compression = 100
needed_vocab_bits = (text_bytes * 8) / target_compression / len(words)
needed_vocab_size = 2 ** needed_vocab_bits
print(f"\nFor 100x compression of this text:")
print(f"  Need {needed_vocab_bits:.1f} bits per word")
print(f"  Need vocabulary of {needed_vocab_size:.0f} words")
print(f"  That's ~{needed_vocab_size:.0f} words in total vocabulary")
