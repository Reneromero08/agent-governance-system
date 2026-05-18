"""Verify: geodesic distance can classify truth vs lies.

If truth follows shorter geodesics, then geodesic distance alone
should classify true vs false concept pairs above chance.
Also: test whether the corollary "lie detection via geodesic deviation"
holds at the individual pair level.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import mannwhitneyu

print("=" * 65)
print("VERIFICATION: GEODESIC DISTANCE CLASSIFIES TRUTH VS LIES")
print("=" * 65)

model = SentenceTransformer('all-MiniLM-L6-v2')

true_triples = [
    ("Paris", "France"), ("Tokyo", "Japan"), ("Einstein", "relativity"),
    ("DNA", "biology"), ("Water", "H2O"), ("Shakespeare", "Romeo and Juliet"),
    ("Sun", "solar system"), ("Everest", "mountain"), ("Bees", "honey"),
    ("Pacific", "ocean"), ("Gravity", "force"), ("Earth", "Sun"),
    ("Oxygen", "breathing"), ("Gold", "metal"), ("Nile", "river"),
    ("Venus", "planet"), ("Cheetahs", "speed"), ("Bananas", "fruit"),
    ("Amazon", "rainforest"), ("Light", "speed"), ("Iron", "magnet"),
    ("Penguins", "flightless"), ("Jupiter", "planet"), ("Wolves", "hunt"),
    ("Diamonds", "pressure"), ("Spiders", "silk"), ("Salt", "water"),
    ("Moon", "tides"), ("Whales", "ocean"), ("Volcanoes", "lava"),
    ("Blood", "heart"), ("Helium", "gas"), ("Coffee", "caffeine"),
    ("Dolphins", "ocean"), ("Octopuses", "arms"), ("Saturn", "rings"),
    ("Diamonds", "hard"), ("Bamboo", "fast"), ("Antarctica", "cold"),
    ("Lightning", "electricity"), ("Sahara", "desert"),
    ("Hummingbirds", "hover"), ("Turtles", "shell"),
    ("Earthquakes", "tectonic"), ("Coral", "reef"),
    ("Photosynthesis", "sunlight"), ("Diamond", "carbon"),
    ("Humans", "DNA"), ("Glaciers", "ice"),
    ("Coffee", "beans"),
]

false_triples = [
    ("Paris", "Germany"), ("Tokyo", "China"), ("Einstein", "evolution"),
    ("DNA", "geology"), ("Water", "CO2"), ("Shakespeare", "Star Wars"),
    ("Sun", "Mars"), ("Everest", "volcano"), ("Bees", "milk"),
    ("Pacific", "lake"), ("Gravity", "color"), ("Earth", "Jupiter"),
    ("Oxygen", "digestion"), ("Gold", "liquid"), ("Nile", "highway"),
    ("Venus", "coldest"), ("Cheetahs", "insects"), ("Bananas", "mineral"),
    ("Amazon", "city"), ("Light", "walking"), ("Iron", "plastic"),
    ("Penguins", "flying"), ("Jupiter", "smallest"), ("Wolves", "oceans"),
    ("Diamonds", "sunlight"), ("Spiders", "milk"), ("Salt", "oil"),
    ("Moon", "earthquakes"), ("Whales", "insects"), ("Volcanoes", "water"),
    ("Blood", "lungs"), ("Helium", "metal"), ("Coffee", "alcohol"),
    ("Dolphins", "reptiles"), ("Octopuses", "brains"), ("Saturn", "oceans"),
    ("Diamonds", "soft"), ("Bamboo", "slow"), ("Antarctica", "hot"),
    ("Lightning", "chemical"), ("Sahara", "tundra"),
    ("Hummingbirds", "cannot fly"), ("Turtles", "wings"),
    ("Earthquakes", "waves"), ("Coral", "giant fish"),
    ("Photosynthesis", "gravity"), ("Diamond", "wood"),
    ("Humans", "feathers"), ("Glaciers", "fire"),
    ("Coffee", "plastic"),
]

def geodesic_distance(subj, obj):
    e_s = model.encode(subj)
    e_o = model.encode(obj)
    a = e_s / (np.linalg.norm(e_s) + 1e-10)
    b = e_o / (np.linalg.norm(e_o) + 1e-10)
    return float(np.arccos(np.clip(np.dot(a, b), -1, 1)))

# Also compute subject+relation->object distance for stronger signal
true_dists = []; false_dists = []
true_dists_sr = []; false_dists_sr = []

# Extended triples with relations for stronger signal
true_triples_rel = [
    ("Paris", "is the capital of", "France"),
    ("Tokyo", "is the capital of", "Japan"),
    ("Einstein", "developed", "relativity"),
    ("Water", "is", "H2O"),
    ("Shakespeare", "wrote", "Romeo and Juliet"),
    ("Bees", "produce", "honey"),
    ("Earth", "orbits", "Sun"),
    ("Diamonds", "are", "carbon"),
    ("Photosynthesis", "uses", "sunlight"),
    ("Gravity", "is a", "force"),
    ("Gold", "is a precious", "metal"),
    ("Oxygen", "enables", "breathing"),
    ("Moon", "causes", "tides"),
    ("Blood", "is pumped by", "heart"),
    ("Venus", "is a", "planet"),
    ("Everest", "is a", "mountain"),
    ("Pacific", "is an", "ocean"),
    ("Cheetahs", "are the fastest", "animals"),
    ("Bananas", "are", "fruit"),
    ("Amazon", "is a", "rainforest"),
    ("Light", "travels at", "speed"),
    ("Iron", "attracts", "magnet"),
    ("Penguins", "are", "flightless"),
    ("Jupiter", "is a", "planet"),
    ("Wolves", "hunt in", "packs"),
    ("Diamonds", "form under", "pressure"),
    ("Spiders", "produce", "silk"),
    ("Salt", "dissolves in", "water"),
    ("Volcanoes", "erupt", "lava"),
    ("Coffee", "contains", "caffeine"),
]

false_triples_rel = [
    ("Paris", "is the capital of", "Germany"),
    ("Tokyo", "is the capital of", "China"),
    ("Einstein", "developed", "evolution"),
    ("Water", "is", "CO2"),
    ("Shakespeare", "wrote", "Star Wars"),
    ("Bees", "produce", "milk"),
    ("Earth", "orbits", "Jupiter"),
    ("Diamonds", "are", "wood"),
    ("Photosynthesis", "uses", "gravity"),
    ("Gravity", "is a", "color"),
    ("Gold", "is a precious", "liquid"),
    ("Oxygen", "enables", "digestion"),
    ("Moon", "causes", "earthquakes"),
    ("Blood", "is pumped by", "lungs"),
    ("Venus", "is a", "star"),
    ("Everest", "is a", "volcano"),
    ("Pacific", "is an", "lake"),
    ("Cheetahs", "are the fastest", "insects"),
    ("Bananas", "are", "mineral"),
    ("Amazon", "is a", "city"),
    ("Light", "travels at", "walking"),
    ("Iron", "attracts", "plastic"),
    ("Penguins", "are", "flying"),
    ("Jupiter", "is a", "star"),
    ("Wolves", "hunt in", "oceans"),
    ("Diamonds", "form under", "sunlight"),
    ("Spiders", "produce", "milk"),
    ("Salt", "dissolves in", "oil"),
    ("Volcanoes", "erupt", "water"),
    ("Coffee", "contains", "alcohol"),
]

for subj, obj in true_triples:
    true_dists.append(geodesic_distance(subj, obj))
for subj, obj in false_triples:
    false_dists.append(geodesic_distance(subj, obj))
for subj, rel, obj in true_triples_rel:
    true_dists_sr.append(geodesic_distance(f"{subj} {rel}", obj))
for subj, rel, obj in false_triples_rel:
    false_dists_sr.append(geodesic_distance(f"{subj} {rel}", obj))

# Combine
labels_true = np.zeros(len(true_dists))  # 0 = true
labels_false = np.ones(len(false_dists))  # 1 = false
all_dists = np.concatenate([true_dists, false_dists])
all_labels = np.concatenate([labels_true, labels_false])

labels_true_sr = np.zeros(len(true_dists_sr))
labels_false_sr = np.ones(len(false_dists_sr))
all_dists_sr = np.concatenate([true_dists_sr, false_dists_sr])
all_labels_sr = np.concatenate([labels_true_sr, labels_false_sr])

# Classification: predict false (label=1) when distance > threshold
# AUC measures how well distance separates truth from lies
# Using distance directly (higher distance = more likely false)
auc_subj_obj = roc_auc_score(all_labels, all_dists)
auc_subjrel_obj = roc_auc_score(all_labels_sr, all_dists_sr)

# Optimal threshold: minimize error
thresholds = np.linspace(0.5, 1.5, 100)
best_acc = 0; best_thresh = 0
for t in thresholds:
    pred = (all_dists_sr > t).astype(int)
    acc = accuracy_score(all_labels_sr, pred)
    if acc > best_acc:
        best_acc = acc; best_thresh = t

# Per-pair classification
correct = 0; total = 0
for i in range(min(len(true_dists_sr), len(false_dists_sr))):
    if true_dists_sr[i] < best_thresh:
        correct += 1  # correctly identified as true (short distance)
    if false_dists_sr[i] > best_thresh:
        correct += 1  # correctly identified as false (long distance)
    total += 2

print(f"\n--- Classification Performance ---")
print(f"  Subj->Obj AUC:            {auc_subj_obj:.4f}")
print(f"  Subj+Rel->Obj AUC:        {auc_subjrel_obj:.4f}")
print(f"  Best threshold:           {best_thresh:.4f}")
print(f"  Accuracy at threshold:    {best_acc:.4f} ({correct}/{total} pairs)")
print(f"  Chance accuracy:          0.5000")
print(f"\n  AUC test: {'PASS' if auc_subjrel_obj > 0.7 else 'PARTIAL' if auc_subjrel_obj > 0.6 else 'FAIL'}")

# Individual pair verification
print(f"\n--- Individual Truth/Lie Classification ---")
print(f"  {'Pair':<35s} {'True d':>8s} {'False d':>8s} {'Classified':>12s}")
print(f"  {'-'*65}")
for i in range(min(len(true_dists_sr), len(false_dists_sr))):
    subj, rel, obj = true_triples_rel[i]
    _, _, fobj = false_triples_rel[i]
    td = true_dists_sr[i]; fd = false_dists_sr[i]
    pred_t = "TRUE" if td < best_thresh else "FALSE (miss)"
    pred_f = "FALSE" if fd > best_thresh else "TRUE (miss)"
    short = "Truth" if td < fd else "Lie"
    print(f"  {subj:>10s} -> {obj:<17s} {td:8.4f} {fd:8.4f}  True:{pred_t:<10s} False:{pred_f:<10s}")

# Count: how often is truth distance < lie distance within the pair?
same_pair_correct = sum(1 for i in range(min(len(true_dists_sr), len(false_dists_sr))) 
                        if true_dists_sr[i] < false_dists_sr[i])
same_pair_total = min(len(true_dists_sr), len(false_dists_sr))
print(f"\n  Truth closer than lie (same pair): {same_pair_correct}/{same_pair_total}")

print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")
print(f"  AUC (subj->obj):     {auc_subj_obj:.4f}")
print(f"  AUC (subj+rel->obj): {auc_subjrel_obj:.4f}")
print(f"  Accuracy:            {best_acc:.4f}")
print(f"  Same-pair correct:   {same_pair_correct}/{same_pair_total}")

if auc_subjrel_obj > 0.8:
    print(f"\n  VERIFIED: Geodesic distance classifies truth vs lies.")
    print(f"  Lie detection via geodesic deviation is operationally viable.")
elif auc_subjrel_obj > 0.6:
    print(f"\n  PARTIALLY VERIFIED: Geodesic distance carries signal.")
else:
    print(f"\n  NOT VERIFIED.")
