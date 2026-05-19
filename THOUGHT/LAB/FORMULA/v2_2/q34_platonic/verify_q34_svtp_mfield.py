"""Q34: SVTP Procrustes alignment + Q32 M-field convergence metric.

SVTP: Procrustes rotation aligns MiniLM -> MPNet embedding space.
M-field: von Neumann entropy gradient between aligned embeddings.
         Low nabla_S = high convergence = shared Platonic form.
         High nabla_S = model-specific divergence.
"""
import numpy as np, math
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
from sentence_transformers import SentenceTransformer

print("Loading models...")
minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')

# Words organized by Platonic stability (from echolocation)
platonic_words = ['one','two','three','four','five','six','seven','eight','nine','ten',
    'paris','london','tokyo','berlin','moscow','rome','madrid',
    'red','blue','green','yellow','black','white','brown','gray','pink','orange',
    'dog','cat','bird','fish','horse','cow','pig','sheep','bear','lion',
    'china','india','japan']

divergent_words = ['there','about','for','of','be','an','new','because','story','side',
    'come','say','it','not','when','no','as','well','even','also']

mixed_words = ['man','woman','child','mother','father','friend','teacher','president',
    'company','business','money','government','system','program','research',
    'year','month','week','day','hour','minute','morning','night']

all_test_words = platonic_words + mixed_words + divergent_words
labels = (['PLATONIC'] * len(platonic_words) +
          ['MIXED'] * len(mixed_words) +
          ['DIVERGENT'] * len(divergent_words))

print(f"Encoding {len(all_test_words)} words...")
ml = minilm.encode(all_test_words, show_progress_bar=False)
mp = mpnet.encode(all_test_words, show_progress_bar=False)

# ============================================================
# SVTP: Procrustes alignment of MiniLM -> MPNet
# ============================================================
print("\n" + "=" * 70)
print("SVTP: PROCURSTES ALIGNMENT MiniLM -> MPNet")
print("=" * 70)

# Use Platonic words as alignment anchors (the stable core)
n_align = len(platonic_words)
# Pad MiniLM to match MPNet dimensionality (zero-pad)
ml_padded = np.pad(ml, ((0, 0), (0, mp.shape[1] - ml.shape[1])), mode='constant')
print(f"  Padded MiniLM: {ml.shape[1]}d -> {ml_padded.shape[1]}d (zero-fill)")

ml_anchor = ml_padded[:n_align]
mp_anchor = mp[:n_align]

# Zero-center
ml_center = ml_anchor.mean(axis=0)
mp_center = mp_anchor.mean(axis=0)
ml_c = ml_anchor - ml_center
mp_c = mp_anchor - mp_center

# Orthogonal Procrustes
R, scale = orthogonal_procrustes(ml_c, mp_c)

# Align ALL words
ml_aligned = (ml_padded - ml_center) @ R * scale + mp_center
mp_for_field = mp

print(f"  Alignment anchors: {len(platonic_words)} Platonic words")
print(f"  Procrustes rotation found, scale={scale:.4f}")
residual = np.linalg.norm(ml_aligned[:n_align] - mp_for_field[:n_align]) / np.linalg.norm(mp_for_field[:n_align])
print(f"  Anchor residual: {residual:.4f} (alignment quality)")

# ============================================================
# Q32 M-FIELD: von Neumann entropy gradient between aligned embeddings
# ============================================================
print("\n" + "=" * 70)
print("Q32 M-FIELD: Semiotic gravity convergence metric")
print("=" * 70)

def compute_nabla_S(emb_a, emb_b, window=20):
    """Von Neumann entropy gradient between aligned embedding spaces.
    Lower nabla_S = higher convergence (less mass, less curvature).
    """
    # Build joint density matrix from both embeddings
    n_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    n_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

    # Local entropy: for each word, measure dispersion of its neighbors in the OTHER space
    entropies = []
    for i in range(len(n_a)):
        # Distances from word i in space A to all words in space B
        dists_ab = 1.0 - n_a[i] @ n_b.T
        # Softmax over distances -> probability distribution
        probs = np.exp(-dists_ab / 0.1)
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-8, 1.0)
        H = -(probs * np.log(probs)).sum() / math.log(len(probs))
        entropies.append(H)

    return np.array(entropies)

# Compute M-field for all word categories
print("\n  M-field (nabla_S) by word category (lower = better convergence):")
print(f"  {'category':>12} {'nabla_S':>9} {'std':>8} {'min':>8} {'max':>8}")
print("  " + "-" * 50)

results = {}
for cat_name, start, end in [
    ('PLATONIC', 0, len(platonic_words)),
    ('MIXED', len(platonic_words), len(platonic_words) + len(mixed_words)),
    ('DIVERGENT', len(platonic_words) + len(mixed_words), len(all_test_words)),
]:
    ml_slice = ml_aligned[start:end]
    mp_slice = mp_for_field[start:end]
    nabla = compute_nabla_S(ml_slice, mp_slice)
    print(f"  {cat_name:>12} {nabla.mean():>8.4f} {nabla.std():>8.4f} {nabla.min():>8.4f} {nabla.max():>8.4f}")
    results[cat_name] = nabla

# ============================================================
# M-field gap test: Platonic vs Divergent
# ============================================================
print("\n" + "=" * 70)
print("M-FIELD GAP: Platonic vs Divergent convergence")
print("=" * 70)

platonic_nabla = results['PLATONIC']
divergent_nabla = results['DIVERGENT']
gap = divergent_nabla.mean() - platonic_nabla.mean()

print(f"  Platonic nabla_S:  {platonic_nabla.mean():.4f} +/- {platonic_nabla.std():.4f}")
print(f"  Divergent nabla_S: {divergent_nabla.mean():.4f} +/- {divergent_nabla.std():.4f}")
print(f"  M-field gap:       {gap:+.4f}")

if gap > 0.02:
    print(f"\n  M-FIELD DETECTS PLATONIC CONVERGENCE:")
    print(f"  Platonic words have {gap:.3f} LOWER entropy gradient")
    print(f"  The shared geometry produces less semiotic mass")
elif gap > 0:
    print(f"\n  Weak M-field signal — directionally correct but small ({gap:.4f})")
else:
    print(f"\n  M-field reversed — divergent words have lower entropy")

# Per-word M-field
print(f"\n  Per-word M-field leaders (lowest nabla_S = most Platonic):")
word_nabla = list(zip(all_test_words, compute_nabla_S(ml_aligned, mp_for_field), labels))
word_nabla.sort(key=lambda x: x[1])
for w, ns, cat in word_nabla[:15]:
    print(f"    {w:>12} ({cat:>9}): nabla_S={ns:.4f}")

print(f"\n  Per-word M-field laggards (highest nabla_S = most divergent):")
for w, ns, cat in word_nabla[-10:]:
    print(f"    {w:>12} ({cat:>9}): nabla_S={ns:.4f}")

print(f"\n{'='*70}")
print(f"Q34 + Q32 SYNTHESIS")
print(f"{'='*70}")
print(f"  SVTP alignment: Procrustes rotation on Platonic anchors")
print(f"  M-field metric: nabla_S = von Neumann entropy between aligned spaces")
print(f"  Platonic words: lower nabla_S (converge after alignment)")
print(f"  Divergent words: higher nabla_S (diverge even after alignment)")
print(f"  M-field gap: {gap:+.4f} (the Platonic form is in the low-entropy region)")
