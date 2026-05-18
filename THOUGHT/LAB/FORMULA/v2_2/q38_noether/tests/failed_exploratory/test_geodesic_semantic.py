"""Test: truth follows shorter geodesics in semantic embedding space.

Theory: R = (E/grad_S) * sigma^D_f predicts:
- Truth: high sigma (compressed), low grad_S (low dissonance) = short geodesic
- Lie: low sigma (diffuse), high grad_S (high dissonance) = long geodesic

Method: Embed concepts in sentence-transformer space. Compute geodesic distance
(SLERP arc length on unit sphere) between true concept pairs and false concept pairs.
Truth should produce shorter distances.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_ind, mannwhitneyu

print("=" * 65)
print("GEODESIC TRUTH IN SEMANTIC SPACE")
print("=" * 65)

print("\nLoading all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded.")

# True facts: subject, relation, object
true_triples = [
    ("Paris", "is the capital of", "France"),
    ("Tokyo", "is the capital of", "Japan"),
    ("Einstein", "developed the theory of", "relativity"),
    ("DNA", "carries genetic information in", "cells"),
    ("Water", "has the chemical formula", "H2O"),
    ("Shakespeare", "wrote the play", "Romeo and Juliet"),
    ("The Sun", "is the star at the center of", "the solar system"),
    ("Mount Everest", "is the tallest", "mountain"),
    ("Bees", "produce", "honey"),
    ("The Pacific", "is the largest", "ocean"),
    ("Photosynthesis", "converts sunlight into", "energy"),
    ("Diamond", "is a form of", "carbon"),
    ("Gravity", "is a fundamental", "force"),
    ("The Earth", "orbits around", "the Sun"),
    ("Oxygen", "is necessary for human", "respiration"),
    ("Gold", "is a precious", "metal"),
    ("The Nile", "is the longest", "river"),
    ("Venus", "is the hottest", "planet"),
    ("Cheetahs", "are the fastest land", "animals"),
    ("Bananas", "are a type of", "fruit"),
    ("The Amazon", "is the largest", "rainforest"),
    ("Light", "travels faster than", "sound"),
    ("Iron", "is attracted to", "magnets"),
    ("Penguins", "are flightless", "birds"),
    ("Jupiter", "is the largest", "planet"),
    ("Wolves", "hunt in", "packs"),
    ("Diamonds", "are formed under extreme", "pressure"),
    ("Spiders", "produce", "silk"),
    ("Salt", "dissolves in", "water"),
    ("The Moon", "causes", "tides"),
    ("Whales", "are the largest", "mammals"),
    ("Humans", "have 46", "chromosomes"),
    ("Helium", "is lighter than", "air"),
    ("Glaciers", "are made of compressed", "snow"),
    ("Coffee", "contains", "caffeine"),
    ("Dolphins", "are highly intelligent marine", "mammals"),
    ("Helium", "is a noble", "gas"),
    ("Octopuses", "have three", "hearts"),
    ("Saturn", "is known for its", "rings"),
    ("Volcanoes", "erupt", "lava"),
    ("Blood", "is pumped by", "the heart"),
    ("Diamonds", "are the hardest natural", "substance"),
    ("Bamboo", "is the fastest growing", "plant"),
    ("Antarctica", "is the coldest", "continent"),
    ("Lightning", "is an electrical", "discharge"),
    ("The Sahara", "is the largest hot", "desert"),
    ("Hummingbirds", "can hover in", "mid-air"),
    ("Turtles", "have a protective", "shell"),
    ("Earthquakes", "are caused by tectonic plate", "movement"),
    ("Coral", "reefs are built by tiny marine", "organisms"),
]

# False facts: subject, false relation, wrong object
false_triples = [
    ("Paris", "is the capital of", "Germany"),
    ("Tokyo", "is the capital of", "China"),
    ("Einstein", "developed the theory of", "evolution"),
    ("DNA", "carries genetic information in", "rocks"),
    ("Water", "has the chemical formula", "CO2"),
    ("Shakespeare", "wrote the play", "Star Wars"),
    ("The Sun", "is the star at the center of", "Mars"),
    ("Mount Everest", "is the tallest", "volcano"),
    ("Bees", "produce", "milk"),
    ("The Pacific", "is the largest", "lake"),
    ("Photosynthesis", "converts sunlight into", "gravity"),
    ("Diamond", "is a form of", "wood"),
    ("Gravity", "is a fundamental", "color"),
    ("The Earth", "orbits around", "Jupiter"),
    ("Oxygen", "is necessary for human", "digestion of rocks"),
    ("Gold", "is a precious", "liquid"),
    ("The Nile", "is the longest", "highway"),
    ("Venus", "is the coldest", "planet"),
    ("Cheetahs", "are the fastest land", "insects"),
    ("Bananas", "are a type of", "mineral"),
    ("The Amazon", "is the largest", "city"),
    ("Light", "travels slower than", "walking"),
    ("Iron", "is attracted to", "plastic"),
    ("Penguins", "are flying", "birds"),
    ("Jupiter", "is the smallest", "planet"),
    ("Wolves", "hunt in", "oceans"),
    ("Diamonds", "are formed under extreme", "sunlight"),
    ("Spiders", "produce", "milk"),
    ("Salt", "dissolves in", "oil"),
    ("The Moon", "causes", "earthquakes"),
    ("Whales", "are the largest", "insects"),
    ("Humans", "have 100", "chromosomes"),
    ("Helium", "is heavier than", "lead"),
    ("Glaciers", "are made of compressed", "fire"),
    ("Coffee", "contains", "alcohol"),
    ("Dolphins", "are highly intelligent marine", "reptiles"),
    ("Helium", "is a transition", "metal"),
    ("Octopuses", "have eight", "brains"),
    ("Saturn", "is known for its", "oceans"),
    ("Volcanoes", "erupt", "water"),
    ("Blood", "is pumped by", "the lungs"),
    ("Diamonds", "are the softest natural", "substance"),
    ("Bamboo", "is the slowest growing", "plant"),
    ("Antarctica", "is the hottest", "continent"),
    ("Lightning", "is a chemical", "reaction"),
    ("The Sahara", "is the largest frozen", "tundra"),
    ("Hummingbirds", "cannot fly in", "air"),
    ("Turtles", "have a protective", "wing"),
    ("Earthquakes", "are caused by ocean", "waves"),
    ("Coral", "reefs are built by giant", "fish"),
]

def geodesic_distance(emb_a, emb_b):
    """SLERP arc length on unit sphere between two normalized vectors."""
    a = emb_a / (np.linalg.norm(emb_a) + 1e-10)
    b = emb_b / (np.linalg.norm(emb_b) + 1e-10)
    cos_theta = np.clip(np.dot(a, b), -1, 1)
    return np.arccos(cos_theta)

print(f"\nComputing geodesic distances for {len(true_triples)} true + {len(false_triples)} false pairs...")

true_distances = []
false_distances = []

for subj, rel, obj in true_triples:
    # Embed concept pairs
    emb_s = model.encode(subj)
    emb_o = model.encode(obj)
    # Also: embed the subject + relation concatenated for context
    emb_sr = model.encode(f"{subj} {rel}")
    
    # Geodesic: subject -> object, and subject+relation -> object
    d_so = geodesic_distance(emb_s, emb_o)
    d_sro = geodesic_distance(emb_sr, emb_o)
    # Combined: shorter of the two paths, plus subject-to-relation distance
    emb_r = model.encode(obj)  # the object itself as destination
    true_distances.append({
        "subject": subj, "object": obj,
        "d_subj_obj": d_so,
        "d_subjrel_obj": d_sro,
        "d_min": min(d_so, d_sro),
    })

for subj, rel, obj in false_triples:
    emb_s = model.encode(subj)
    emb_o = model.encode(obj)
    emb_sr = model.encode(f"{subj} {rel}")
    d_so = geodesic_distance(emb_s, emb_o)
    d_sro = geodesic_distance(emb_sr, emb_o)
    false_distances.append({
        "subject": subj, "object": obj,
        "d_subj_obj": d_so,
        "d_subjrel_obj": d_sro,
        "d_min": min(d_so, d_sro),
    })

td = np.array([d["d_subj_obj"] for d in true_distances])
fd = np.array([d["d_subj_obj"] for d in false_distances])
td_sr = np.array([d["d_subjrel_obj"] for d in true_distances])
fd_sr = np.array([d["d_subjrel_obj"] for d in false_distances])
td_min = np.array([d["d_min"] for d in true_distances])
fd_min = np.array([d["d_min"] for d in false_distances])

print(f"\n{'='*65}")
print(f"RESULTS")
print(f"{'='*65}")

print(f"\n--- Geodesic distance: subject -> object ---")
print(f"  True:  {np.mean(td):.4f} +/- {np.std(td):.4f}")
print(f"  False: {np.mean(fd):.4f} +/- {np.std(fd):.4f}")
t1, p1 = ttest_ind(td, fd)
u1, pu1 = mannwhitneyu(td, fd, alternative='less')
d1 = (np.mean(fd) - np.mean(td)) / np.sqrt((np.var(td) + np.var(fd))/2)
print(f"  t = {t1:.4f}, p = {p1:.6f}, Mann-Whitney p = {pu1:.6f}")
print(f"  Cohen's d = {d1:.4f}")
print(f"  {'PASS' if np.mean(td) < np.mean(fd) and pu1 < 0.05 else 'FAIL'}")

print(f"\n--- Geodesic distance: subject+relation -> object ---")
print(f"  True:  {np.mean(td_sr):.4f} +/- {np.std(td_sr):.4f}")
print(f"  False: {np.mean(fd_sr):.4f} +/- {np.std(fd_sr):.4f}")
t2, p2 = ttest_ind(td_sr, fd_sr)
u2, pu2 = mannwhitneyu(td_sr, fd_sr, alternative='less')
d2 = (np.mean(fd_sr) - np.mean(td_sr)) / np.sqrt((np.var(td_sr) + np.var(fd_sr))/2)
print(f"  t = {t2:.4f}, p = {p2:.6f}, Mann-Whitney p = {pu2:.6f}")
print(f"  Cohen's d = {d2:.4f}")
print(f"  {'PASS' if np.mean(td_sr) < np.mean(fd_sr) and pu2 < 0.05 else 'FAIL'}")

print(f"\n--- Minimum distance (best path) ---")
print(f"  True:  {np.mean(td_min):.4f} +/- {np.std(td_min):.4f}")
print(f"  False: {np.mean(fd_min):.4f} +/- {np.std(fd_min):.4f}")

# Show some examples
print(f"\n--- Example pairs ---")
for i in [0, 5, 10, 25, 40]:
    if i < len(true_distances):
        t = true_distances[i]
        f = false_distances[i]
        print(f"\n  True:  {t['subject']} -> {t['object']:20s}  d={t['d_subj_obj']:.4f}")
        print(f"  False: {f['subject']} -> {f['object']:20s}  d={f['d_subj_obj']:.4f}")
        print(f"         {'Truth closer' if t['d_subj_obj'] < f['d_subj_obj'] else 'False closer':>50s}")

# Summary
print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")

passed = (np.mean(td) < np.mean(fd)) + (np.mean(td_sr) < np.mean(fd_sr))
sig = pu1 < 0.05 or pu2 < 0.05

print(f"  Subj->Obj truth shorter: {'YES' if np.mean(td) < np.mean(fd) else 'NO'} (p={pu1:.4f}, d={d1:.4f})")
print(f"  Subj+Rel->Obj truth shorter: {'YES' if np.mean(td_sr) < np.mean(fd_sr) else 'NO'} (p={pu2:.4f}, d={d2:.4f})")

if passed >= 2 and sig:
    print(f"\n  VERIFIED: True relations have shorter geodesics in semantic space.")
elif passed >= 1:
    print(f"\n  PARTIALLY VERIFIED: Directional evidence for truth-geodesic alignment.")
else:
    print(f"\n  NOT VERIFIED: No evidence true relations follow shorter geodesics.")
