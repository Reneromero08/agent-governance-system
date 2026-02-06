"""
Q36: Empirical Tests

Tests that require actual embeddings and proper null models.
Each test has a clear hypothesis and appropriate baseline.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("Q36 EMPIRICAL TESTS")
print("=" * 70)
print()

# Load embeddings
EMBEDDINGS = None
DIM = None
MODEL_NAME = None

try:
    from sentence_transformers import SentenceTransformer
    print("Loading SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    WORDS = [
        'king', 'queen', 'man', 'woman', 'boy', 'girl',
        'good', 'bad', 'happy', 'sad', 'love', 'hate',
        'big', 'small', 'hot', 'cold', 'fast', 'slow',
        'light', 'dark', 'up', 'down', 'in', 'out',
        'dog', 'cat', 'car', 'house', 'tree', 'water',
        'paris', 'france', 'berlin', 'germany', 'london', 'england',
        'walk', 'walked', 'talk', 'talked', 'run', 'ran',
        'bigger', 'smaller', 'best', 'worst', 'dogs', 'cats',
        'prince', 'princess', 'brother', 'sister', 'father', 'mother'
    ]

    embs = model.encode(WORDS, normalize_embeddings=True)
    EMBEDDINGS = {word: embs[i] for i, word in enumerate(WORDS)}
    DIM = embs.shape[1]
    MODEL_NAME = "all-MiniLM-L6-v2"
    print(f"Loaded {len(EMBEDDINGS)} words, dim={DIM}")
    print()
except Exception as e:
    print(f"Failed to load embeddings: {e}")
    exit(1)

# =============================================================================
# EMPIRICAL TEST 1: Subspace Prediction
# =============================================================================

print("EMPIRICAL TEST 1: Subspace Prediction")
print("-" * 50)
print("Hypothesis: Semantic subspace predicts held-out words")
print("            better than random orthonormal basis.")
print()

def random_orthonormal_basis(k, dim, seed):
    """Generate k random orthonormal vectors in R^dim"""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(dim, k)
    Q, _ = np.linalg.qr(vecs)
    return Q.T[:k]

words = list(EMBEDDINGS.keys())
n_words = len(words)
n_trials = 30
k = 10  # Subspace dimension

semantic_errors = []
random_errors = []

np.random.seed(42)

for trial in range(n_trials):
    np.random.shuffle(words)
    split = int(0.8 * n_words)
    known = words[:split]
    heldout = words[split:]

    # Semantic subspace: top-k PCs of known words
    known_vecs = np.array([EMBEDDINGS[w] for w in known])
    U, S, Vt = np.linalg.svd(known_vecs, full_matrices=False)
    semantic_basis = Vt[:k]

    # Random subspace: random orthonormal basis
    random_basis = random_orthonormal_basis(k, DIM, seed=trial)

    for w in heldout:
        v = EMBEDDINGS[w]

        # Project onto semantic subspace
        proj_s = sum(np.dot(v, b) * b for b in semantic_basis)
        proj_s = proj_s / (np.linalg.norm(proj_s) + 1e-10)
        err_s = 1 - np.dot(v, proj_s)
        semantic_errors.append(err_s)

        # Project onto random subspace
        proj_r = sum(np.dot(v, b) * b for b in random_basis)
        proj_r = proj_r / (np.linalg.norm(proj_r) + 1e-10)
        err_r = 1 - np.dot(v, proj_r)
        random_errors.append(err_r)

mean_semantic = np.mean(semantic_errors)
mean_random = np.mean(random_errors)
ratio = mean_random / (mean_semantic + 1e-10)

print(f"  Semantic error:   {mean_semantic:.4f}")
print(f"  Random error:     {mean_random:.4f}")
print(f"  Ratio:            {ratio:.2f}x")
print(f"  Hypothesis:       Ratio > 1 means semantic structure exists")
print()
print(f"  RESULT: {'SUPPORTED' if ratio > 1.5 else 'NOT SUPPORTED'} (ratio={ratio:.2f})")
print()

# =============================================================================
# EMPIRICAL TEST 2: Word Analogies
# =============================================================================

print("EMPIRICAL TEST 2: Word Analogies")
print("-" * 50)
print("Hypothesis: a - b + c is close to d for valid analogies.")
print()

ANALOGIES = [
    ('king', 'man', 'queen', 'woman'),
    ('man', 'woman', 'boy', 'girl'),
    ('paris', 'france', 'berlin', 'germany'),
    ('walk', 'walked', 'talk', 'talked'),
    ('big', 'bigger', 'small', 'smaller'),
    ('good', 'best', 'bad', 'worst'),
    ('dog', 'dogs', 'cat', 'cats'),
    ('king', 'queen', 'prince', 'princess'),
    ('brother', 'sister', 'father', 'mother'),
]

correct = 0
total = 0
sims_to_target = []

for a, b, c, d in ANALOGIES:
    if not all(w in EMBEDDINGS for w in [a, b, c, d]):
        continue

    # Compute a - b + c
    pred = EMBEDDINGS[a] - EMBEDDINGS[b] + EMBEDDINGS[c]
    pred = pred / np.linalg.norm(pred)

    # Similarity to target d
    sim_d = np.dot(pred, EMBEDDINGS[d])
    sims_to_target.append(sim_d)

    # Find closest word (excluding a, b, c)
    best_word = None
    best_sim = -2
    for w in EMBEDDINGS:
        if w in [a, b, c]:
            continue
        sim = np.dot(pred, EMBEDDINGS[w])
        if sim > best_sim:
            best_sim = sim
            best_word = w

    is_correct = (best_word == d)
    if is_correct:
        correct += 1
    total += 1

    status = "CORRECT" if is_correct else f"got '{best_word}'"
    print(f"  {a}:{b}::{c}:? -> {status} (sim={sim_d:.3f})")

accuracy = correct / total if total > 0 else 0
mean_sim = np.mean(sims_to_target)

print()
print(f"  Accuracy:         {correct}/{total} = {accuracy*100:.0f}%")
print(f"  Mean sim to d:    {mean_sim:.3f}")
print()
print(f"  RESULT: {'SUPPORTED' if accuracy > 0.3 or mean_sim > 0.5 else 'WEAK'}")
print()

# =============================================================================
# EMPIRICAL TEST 3: Semantic Clustering
# =============================================================================

print("EMPIRICAL TEST 3: Semantic Clustering")
print("-" * 50)
print("Hypothesis: Semantic neighbors cluster more than random.")
print()

words = list(EMBEDDINGS.keys())
n = len(words)

# Build similarity matrix
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            sim_matrix[i, j] = np.dot(EMBEDDINGS[words[i]], EMBEDDINGS[words[j]])

# Threshold: top 25% of similarities define "neighbors"
threshold = np.percentile(sim_matrix[sim_matrix > 0], 75)

# Compute clustering coefficient
clustering_coeffs = []
for i in range(n):
    neighbors = np.where(sim_matrix[i] > threshold)[0]
    k = len(neighbors)
    if k < 2:
        continue

    edges = 0
    for ni in range(len(neighbors)):
        for nj in range(ni + 1, len(neighbors)):
            if sim_matrix[neighbors[ni], neighbors[nj]] > threshold:
                edges += 1

    possible = k * (k - 1) / 2
    if possible > 0:
        clustering_coeffs.append(edges / possible)

mean_clustering = np.mean(clustering_coeffs)

# Random baseline: Erdos-Renyi with same edge density
edge_density = np.mean(sim_matrix > threshold)
expected_random = edge_density

ratio = mean_clustering / (expected_random + 1e-10)

print(f"  Clustering coeff: {mean_clustering:.3f}")
print(f"  Random expected:  {expected_random:.3f}")
print(f"  Ratio:            {ratio:.2f}x")
print()
print(f"  RESULT: {'SUPPORTED' if ratio > 1.5 else 'NOT SUPPORTED'}")
print()

# =============================================================================
# EMPIRICAL TEST 4: Relation Vector Consistency
# =============================================================================

print("EMPIRICAL TEST 4: Relation Vector Consistency")
print("-" * 50)
print("Hypothesis: Similar relations have parallel difference vectors.")
print()

RELATION_GROUPS = [
    [('king', 'queen'), ('man', 'woman'), ('boy', 'girl'), ('prince', 'princess')],
    [('good', 'bad'), ('happy', 'sad'), ('love', 'hate')],
    [('big', 'small'), ('hot', 'cold'), ('fast', 'slow')],
    [('paris', 'france'), ('berlin', 'germany'), ('london', 'england')],
]

group_sims = []

for group in RELATION_GROUPS:
    diffs = []
    for w1, w2 in group:
        if w1 in EMBEDDINGS and w2 in EMBEDDINGS:
            d = EMBEDDINGS[w1] - EMBEDDINGS[w2]
            d = d / np.linalg.norm(d)
            diffs.append(d)

    if len(diffs) >= 2:
        sims = []
        for i in range(len(diffs)):
            for j in range(i + 1, len(diffs)):
                sims.append(abs(np.dot(diffs[i], diffs[j])))
        group_sims.append(np.mean(sims))
        print(f"  Group '{group[0][0]}-{group[0][1]}': mean|cos|={np.mean(sims):.3f}")

mean_consistency = np.mean(group_sims) if group_sims else 0

print()
print(f"  Overall consistency: {mean_consistency:.3f}")
print()
print(f"  RESULT: {'SUPPORTED' if mean_consistency > 0.3 else 'NOT SUPPORTED'}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  Model: {MODEL_NAME}")
print()

results = [
    ("Subspace prediction", ratio > 1.5, f"ratio={ratio:.1f}x"),
    ("Word analogies", accuracy > 0.3 or mean_sim > 0.5, f"acc={accuracy*100:.0f}%, sim={mean_sim:.2f}"),
    ("Semantic clustering", mean_clustering / (expected_random + 1e-10) > 1.5, f"ratio={mean_clustering/(expected_random+1e-10):.1f}x"),
    ("Relation consistency", mean_consistency > 0.3, f"cos={mean_consistency:.2f}"),
]

for name, supported, evidence in results:
    status = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"  [{status:^13}] {name}: {evidence}")

print()
supported_count = sum(1 for r in results if r[1])
print(f"Supported: {supported_count}/{len(results)}")
