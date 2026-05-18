"""Q45: Complex geometry vs. real geometry for semantic navigation.
FS geodesic distance vs. cosine similarity — which ranks related words better?
"""
import sys, time
import numpy as np
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS = list(ANCHOR_1024)

# Semantic pairs: word A and a set of words that SHOULD be closer to A than others
SEMANTIC_GROUPS = {
    "dog": ["cat", "puppy", "bark", "pet", "animal", "tail", "bone", "wolf"],
    "king": ["queen", "prince", "castle", "kingdom", "crown", "rule", "royal", "throne"],
    "water": ["river", "ocean", "sea", "lake", "rain", "wave", "stream", "drink"],
    "fire": ["flame", "burn", "hot", "heat", "smoke", "ash", "blaze", "spark"],
    "love": ["heart", "romance", "passion", "affection", "kiss", "emotion", "desire", "caring"],
    "war": ["battle", "fight", "peace", "army", "weapon", "combat", "soldier", "conflict"],
    "book": ["page", "read", "story", "author", "chapter", "novel", "write", "library"],
    "tree": ["forest", "leaf", "wood", "branch", "root", "plant", "bark", "grow"],
}

def evaluate_ranking(query_word, related_words, vectors, metric="cosine"):
    """For each related word, compute its rank among all words under given metric.
    Lower mean rank = better metric for semantic navigation."""
    ai = WORDS.index(query_word)
    ranks = []
    for rw in related_words:
        if rw not in WORDS: continue
        ri = WORDS.index(rw)
        if metric == "cosine":
            sims = vectors @ vectors[ai]  # real dot products
        elif metric == "complex_abs":
            sims = np.abs(vectors @ np.conj(vectors[ai]))  # FS overlap
        elif metric == "real":
            sims = vectors.real @ vectors[ai].real  # real part only
        sims[ai] = -np.inf
        rank = np.sum(sims > sims[ri])
        ranks.append(rank)
    return np.mean(ranks) if ranks else np.inf

for mid, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()

    print(f"\n{'='*64}")
    print(f"{name} — Semantic ranking: cosine vs. complex FS")
    print(f"{'='*64}")

    for K in [96, D]:
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
        proj = centered @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms

        # Real vectors (cosine)
        real_ranks = []
        # Complex vectors (FS abs overlap)
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
        complex_ranks = []

        for qword, rwords in SEMANTIC_GROUPS.items():
            if qword not in WORDS: continue
            rr = evaluate_ranking(qword, rwords, proj, "cosine")
            cr = evaluate_ranking(qword, rwords, z, "complex_abs")
            if not np.isinf(rr): real_ranks.append(rr)
            if not np.isinf(cr): complex_ranks.append(cr)

        if real_ranks and complex_ranks:
            print(f"  K={K:3d}: cosine_mean_rank={np.mean(real_ranks):.1f}  complex_mean_rank={np.mean(complex_ranks):.1f}  delta={np.mean(real_ranks)-np.mean(complex_ranks):+.1f}")
            better = "COMPLEX" if np.mean(complex_ranks) < np.mean(real_ranks) else "COSINE"
            print(f"         Better: {better}")
            # Paired t-test
            from scipy import stats
            t, p = stats.ttest_rel(real_ranks, complex_ranks)
            print(f"         Paired t-test p={p:.4f}")

print(f"\n{'='*64}")
print("Verdict:")
print("  If complex FS ranks related words HIGHER (lower mean rank),")
print("  complex geometry improves semantic navigation over cosine similarity.")
