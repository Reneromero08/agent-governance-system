"""Latent Lattice Oracle for Catalytic Hidden States.

From 20.10.8: sparse probes + .holo manifold + lattice geometry.
Key insight: tokens close in latent space should map to similar gold tokens.
Uses k-NN in latent space instead of training a classifier.

Pipeline:
  1. Sparse probes: run catalytic engine on K token embeddings
  2. .holo manifold: project hidden states to latent space
  3. Latent lattice: for each hidden state, find nearest gold token in latent space
  4. Verify: does predicted token match oracle?
"""
import sys, os, time, math, numpy as np, torch
from pathlib import Path
from collections import Counter

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
CAT_CAS = REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference'
EIGEN = CAT_CAS.parent.parent / 'EIGEN_BUDDY'
sys.path.insert(0, str(REPO / 'THOUGHT' / 'LAB' / 'TINY_COMPRESS' / 'holographic-image'))
from holo_core import analyze_spectrum, project, choose_k
sys.path.insert(0, str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
os.chdir(str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
import catalytic_ffi

# Load gold data
data = torch.load(CAT_CAS / 'gold_training_data' / 'gold_pairs_quick.pt', weights_only=True)
sr = data['states_real'].numpy(); si = data['states_imag'].numpy()
targets = data['targets'].tolist()
# Fix Inf
sr[~np.isfinite(sr)] = 0; si[~np.isfinite(si)] = 0

N = len(targets)
print(f'Loaded {N} samples, {len(set(targets))} unique gold tokens')

# ── Step 1: Build complex observation matrix (per-sample normalized) ──
Z = (sr + 1j * si).astype(np.complex128)
norms = np.sqrt((Z.conj() * Z).real.sum(axis=1))
norms = np.maximum(norms, 1e-15)
Z_norm = Z / norms[:, np.newaxis]

# ── Step 2: .holo manifold projection ──
# Use neighbor windows: each row = 4 consecutive hidden states
window_size = min(4, N - 1)
if window_size >= 2:
    obs_list = []
    for i in range(N - window_size):
        w = Z_norm[i:i+window_size].flatten()
        obs_list.append(np.hstack([w.real.astype(np.float64), w.imag.astype(np.float64)]))
    obs = np.array(obs_list)
    actual_N = N - window_size
    targets_w = targets[window_size:]  # next token after window
else:
    obs = np.hstack([Z_norm.real.astype(np.float64), Z_norm.imag.astype(np.float64)])
    actual_N = N
    targets_w = targets

print(f'Observation matrix: {obs.shape}')
spec = analyze_spectrum(obs)
k = max(4, min(choose_k(spec, policy="participation"), obs.shape[1]-1))
proj = project(obs, policy="fixed", fixed_k=k)
coords = proj.coordinates  # (n, k) latent positions
print(f'D_pr={spec.participation_dimension:.1f}, K={k}, Coords: {coords.shape}')

# ── Step 3: Latent k-NN prediction ──
# Train on first 80%, test on last 20%
split = int(actual_N * 0.8)
if split < 1: split = 1
train_coords = coords[:split]
train_targets = [targets_w[i] for i in range(split)]
test_coords = coords[split:]
test_targets = [targets_w[i] for i in range(split, actual_N)]

# Build per-class centroids in latent space
class_centroids = {}
class_members = {}
for i, t in enumerate(train_targets):
    if t not in class_centroids:
        class_centroids[t] = []
        class_members[t] = []
    class_centroids[t].append(train_coords[i])
    class_members[t].append(train_coords[i])

# Average centroids
for t in class_centroids:
    class_centroids[t] = np.mean(class_centroids[t], axis=0)

print(f'\n--- k-NN Latent Prediction ---')
print(f'Train classes: {len(class_centroids)}, Test pairs: {len(test_targets)}')

correct = 0
top5_correct = 0
for i, (test_pt, true_tok) in enumerate(zip(test_coords, test_targets)):
    # Distance to each class centroid
    dists = [(t, np.linalg.norm(test_pt - c)) for t, c in class_centroids.items()]
    dists.sort(key=lambda x: x[1])
    pred = dists[0][0]
    top5 = {d[0] for d in dists[:5]}
    if pred == true_tok:
        correct += 1
    if true_tok in top5:
        top5_correct += 1
    if i < 10:
        print(f'  [{split+i}] true={true_tok:>5} pred={pred:>5} '
              f'correct={pred==true_tok} dist={dists[0][1]:.4f}')

print(f'\nTop-1: {correct}/{len(test_targets)} ({correct/max(len(test_targets),1)*100:.1f}%)')
print(f'Top-5: {top5_correct}/{len(test_targets)} ({top5_correct/max(len(test_targets),1)*100:.1f}%)')

# ── Step 4: Compare with raw complex (baseline) ──
print(f'\n--- Baseline: Raw Complex k-NN ---')
raw_obs = np.hstack([Z_norm.real.astype(np.float64), Z_norm.imag.astype(np.float64)])
raw_train = raw_obs[:split]; raw_test = raw_obs[split:]
correct_r = 0; top5_r = 0
raw_centroids = {}
for i, t in enumerate(train_targets):
    if t not in raw_centroids: raw_centroids[t] = []
    raw_centroids[t].append(raw_train[i])
for t in raw_centroids:
    raw_centroids[t] = np.mean(raw_centroids[t], axis=0)
for i, (pt, true_tok) in enumerate(zip(raw_test, test_targets)):
    dists = [(t, np.linalg.norm(pt - c)) for t, c in raw_centroids.items()]
    dists.sort(key=lambda x: x[1])
    pred = dists[0][0]
    top5 = {d[0] for d in dists[:5]}
    if pred == true_tok: correct_r += 1
    if true_tok in top5: top5_r += 1
print(f'Top-1: {correct_r}/{len(test_targets)} ({correct_r/max(len(test_targets),1)*100:.1f}%)')
print(f'Top-5: {top5_r}/{len(test_targets)} ({top5_r/max(len(test_targets),1)*100:.1f}%)')
