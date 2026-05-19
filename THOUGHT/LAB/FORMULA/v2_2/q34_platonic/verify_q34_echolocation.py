"""Q34: Platonic anchor grid — echolocation in cross-model semantic space.

Builds a shared coordinate frame from words with stable neighborhoods
across MiniLM and MPNet. Uses anchor-distance vectors to compare positions.
Maps regions of convergence and divergence.
"""
import numpy as np, math
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from collections import defaultdict

print("Loading models...")
minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')

# Large vocabulary covering diverse semantic domains
words = ['the','be','to','of','and','in','that','have','it','for',
    'not','on','with','he','as','you','do','at','this','but',
    'his','by','from','they','we','say','her','she','or','an',
    'will','my','one','all','would','there','their','what','so','up',
    'out','if','about','who','get','which','go','me','when','make',
    'can','like','time','no','just','him','know','take','people','into',
    'year','your','good','some','could','them','see','other','than','then',
    'now','look','only','come','its','over','think','also','back','after',
    'use','two','how','our','work','first','well','way','even','new',
    'want','because','any','these','give','day','most','us','great','big',
    'man','world','life','hand','part','child','woman','place','case','week',
    'company','system','program','question','government','number','night','point','home','water',
    'room','mother','area','money','story','fact','month','lot','right','study',
    'book','eye','job','word','business','issue','side','kind','head','house',
    'service','friend','father','power','hour','game','line','end','member','law',
    'car','city','community','name','president','team','minute','idea','body','information',
    'back','parent','face','others','level','office','door','health','person','art',
    'war','history','party','result','morning','reason','research','girl','guy','moment',
    'air','teacher','force','education','boy','food','land','nature','girlfriend','boyfriend',
    # Semantic clusters: numbers, colors, animals, geography
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'red','blue','green','yellow','black','white','brown','gray','pink','orange',
    'dog','cat','bird','fish','horse','cow','pig','sheep','bear','lion',
    'paris','london','tokyo','berlin','moscow','rome','madrid','china','india','japan']

# Deduplicate while preserving order
seen = set(); unique_words = []
for w in words:
    if w not in seen: seen.add(w); unique_words.append(w)
words = unique_words

print(f"Encoding {len(words)} words...")
ml = minilm.encode(words, show_progress_bar=False)
mp = mpnet.encode(words, show_progress_bar=False)

# Normalize
ml = ml / (np.linalg.norm(ml, axis=1, keepdims=True) + 1e-8)
mp = mp / (np.linalg.norm(mp, axis=1, keepdims=True) + 1e-8)

print("\n" + "=" * 70)
print("ECHOLOCATION: Cross-model neighborhood stability")
print("=" * 70)

# For each word, find its K nearest neighbors in both models
K = 10
ml_nn = {}
mp_nn = {}
for i in range(len(words)):
    ml_dists = 1.0 - ml[i] @ ml.T
    mp_dists = 1.0 - mp[i] @ mp.T
    ml_nn[i] = set(np.argsort(ml_dists)[1:K+1].tolist())
    mp_nn[i] = set(np.argsort(mp_dists)[1:K+1].tolist())

# Jaccard overlap: |NN_ml ∩ NN_mp| / |NN_ml ∪ NN_mp|
neighbor_stability = []
for i in range(len(words)):
    intersect = len(ml_nn[i] & mp_nn[i])
    union = len(ml_nn[i] | mp_nn[i])
    neighbor_stability.append(intersect / union if union > 0 else 0)

# Categorize words by stability
platonic = [(i, words[i], neighbor_stability[i]) for i in range(len(words))
            if neighbor_stability[i] > 0.5]
shared = [(i, words[i], neighbor_stability[i]) for i in range(len(words))
          if 0.25 < neighbor_stability[i] <= 0.5]
divergent = [(i, words[i], neighbor_stability[i]) for i in range(len(words))
             if neighbor_stability[i] <= 0.25]

print(f"\n  Platonic (>50% neighbor overlap): {len(platonic)} words")
print(f"  Shared   (25-50% overlap):         {len(shared)} words")
print(f"  Divergent (<25% overlap):          {len(divergent)} words")

print(f"\n  Platonic anchors:")
for i, w, s in sorted(platonic, key=lambda x: -x[2])[:15]:
    nn_ml = sorted([(words[j], ml_nn[i]) for j in ml_nn[i]], key=lambda x: x[1])
    print(f"    {w:>12} (stability={s:.2f}) -> {[words[j] for j in sorted(ml_nn[i] & mp_nn[i])][:5]}")

print(f"\n  Most divergent:")
for i, w, s in sorted(divergent, key=lambda x: x[2])[:10]:
    print(f"    {w:>12} (stability={s:.2f}) -> ML: {[words[j] for j in sorted(ml_nn[i] - mp_nn[i])[:3]]} | MP: {[words[j] for j in sorted(mp_nn[i] - ml_nn[i])[:3]]}")

# ============================================================
# ANCHOR-DISTANCE COORDINATE SYSTEM
# ============================================================
print("\n" + "=" * 70)
print("ANCHOR-DISTANCE COORDINATE FRAME")
print("=" * 70)

# Use top Platonic words as anchor grid
n_anchors = min(20, len(platonic))
anchor_indices = [p[0] for p in sorted(platonic, key=lambda x: -x[2])[:n_anchors]]
anchor_words = [words[i] for i in anchor_indices]
print(f"\n  Anchors ({n_anchors}): {anchor_words}")

# Build anchor-distance vectors for ALL words
ml_anchor_dists = np.zeros((len(words), n_anchors))
mp_anchor_dists = np.zeros((len(words), n_anchors))
for a_idx, anchor_i in enumerate(anchor_indices):
    ml_anchor_dists[:, a_idx] = 1.0 - ml @ ml[anchor_i]
    mp_anchor_dists[:, a_idx] = 1.0 - mp @ mp[anchor_i]

# Echolocation: compare anchor-distance vectors across models
# For each word, correlate its ML anchor-vector with its MP anchor-vector
echo_stability = []
for i in range(len(words)):
    r, _ = pearsonr(ml_anchor_dists[i], mp_anchor_dists[i])
    echo_stability.append(r)

print(f"\n  Echolocation stability (anchor-distance correlation):")
print(f"  Mean: {np.mean(echo_stability):.4f}  Median: {np.median(echo_stability):.4f}")
print(f"  Range: [{np.min(echo_stability):.4f}, {np.max(echo_stability):.4f}]")

# Compare neighbor-based stability vs echolocation stability
r_necho, _ = pearsonr(neighbor_stability, echo_stability)
print(f"  Neighbor vs echo correlation: r = {r_necho:+.4f}")

# Map: words by echolocation stability
echo_high = [(words[i], echo_stability[i]) for i in range(len(words))
             if echo_stability[i] > 0.8]
echo_med = [(words[i], echo_stability[i]) for i in range(len(words))
            if 0.5 < echo_stability[i] <= 0.8]
echo_low = [(words[i], echo_stability[i]) for i in range(len(words))
            if echo_stability[i] <= 0.5]

print(f"\n  Echolocation categories:")
print(f"    Strong signal (>0.8):  {len(echo_high)} words")
if echo_high:
    print(f"      {[w for w,_ in sorted(echo_high,key=lambda x:-x[1])[:10]]}")
print(f"    Medium signal (0.5-0.8): {len(echo_med)} words")
print(f"    Weak signal (<0.5):     {len(echo_low)} words")
if echo_low:
    print(f"      {[w for w,_ in sorted(echo_low,key=lambda x:x[1])[:10]]}")

# ============================================================
# PLATONIC SPACE MAP
# ============================================================
print("\n" + "=" * 70)
print("PLATONIC SPACE MAP: Where models agree and diverge")
print("=" * 70)

# For each anchor, compute its "Platonic radius" — the distance within which
# model agreement is high vs low
print(f"\n  Anchor influence regions:")
for a_idx, anchor_i in enumerate(anchor_indices[:10]):
    anchor_w = words[anchor_i]
    # For each other word, agreement = correlation of their anchor-distance vectors
    dists_ml = ml_anchor_dists[:, a_idx]
    dists_mp = mp_anchor_dists[:, a_idx]
    # Words with similar distance to this anchor in both models = stable
    dist_error = np.abs(dists_ml - dists_mp)
    near = [(words[i], dist_error[i], echo_stability[i]) for i in range(len(words))
            if i != anchor_i and dist_error[i] < np.median(dist_error)]
    near_agree = sum(1 for _, _, e in near if e > 0.5)
    print(f"    {anchor_w:>12}: {len(near)} words in near-field, {near_agree} agree ({near_agree/max(len(near),1)*100:.0f}%)")

# Final convergence metric
avg_echo = np.mean(echo_stability)
platonic_fraction = len(platonic) / len(words)

print(f"\n{'='*70}")
print(f"Q34 PLATONIC SPACE SUMMARY")
print(f"{'='*70}")
print(f"  Vocabulary: {len(words)} words")
print(f"  Platonic anchors (K=10 overlap >50%): {len(platonic)} ({platonic_fraction:.1%})")
print(f"  Mean echolocation stability: {avg_echo:.3f}")
print(f"  Strong echo (>0.8): {len(echo_high)} words ({len(echo_high)/len(words):.1%})")
if avg_echo > 0.5:
    print(f"  PLATONIC SPACE CONFIRMED — anchor-distance frame converges across models")
elif avg_echo > 0.3:
    print(f"  Partial Platonic convergence — shared semantic core + model-specific periphery")
else:
    print(f"  Weak convergence in anchor frame")
