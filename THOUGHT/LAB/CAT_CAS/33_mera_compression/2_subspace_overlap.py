"""
MERA Compression v2 — Subspace Overlap + Differential Encoding
===============================================================
Instead of stacking + SVD (which fails because layers are independent),
measure subspace overlap between adjacent layers. If U_k(L) and U_k(L+1)
span similar subspaces, store only the DIFFERENCE (a rotation matrix).

Also: measure the Grassmann distance between consecutive layers.
Small distance -> share basis. Large distance -> store independently.
"""
import torch, time, os, math, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(r"d:\CCC 2.0\AI\agent-governance-system")
HOLO_05B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")
HOLO_27B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_27b_catalytic_k256.holo")

def subspace_overlap(U1, U2):
    """Measure overlap between subspaces spanned by U1 and U2 columns.
    Uses principal angles: cos(theta_i) = singular values of U1^T @ U2.
    Returns mean cosine similarity and Grassmann distance."""
    # U1: (m, k1), U2: (m, k2)
    k = min(U1.shape[1], U2.shape[1])
    M = U1[:, :k].T @ U2[:, :k]  # (k, k)
    _, S, _ = torch.linalg.svd(M.float(), full_matrices=False)
    # cos(theta_i) = S_i. theta_i = acos(S_i)
    cos_mean = S.mean().item()
    # Grassmann distance: sqrt(sum(theta_i^2))
    theta = torch.acos(torch.clamp(S, 0.0, 1.0))
    grassmann = torch.norm(theta).item()
    return cos_mean, grassmann, S

def compress_differential(tensors_dict, similarity_threshold=0.95):
    """
    Chain: store first layer fully, subsequent layers as ROTATIONS from previous.
    If U_{L+1} = R @ U_L for some rotation R, store only R. R is k x k -> small.
    """
    sorted_l = sorted(tensors_dict.keys())
    if len(sorted_l) < 2: return None
    
    L = len(sorted_l)
    first_tensor = tensors_dict[sorted_l[0]]
    m, k = first_tensor.shape
    
    rotations = []
    overlaps = []
    independent_layers = [sorted_l[0]]  # layers stored fully
    rotation_layers = []  # layers stored as rotations
    
    original_bytes = L * m * k * 2  # float16
    compressed_bytes = m * k * 2  # first layer fully stored
    
    for i in range(1, L):
        prev = tensors_dict[sorted_l[i-1]]
        curr = tensors_dict[sorted_l[i]]
        cos_mean, grassmann, S_vals = subspace_overlap(prev, curr)
        overlaps.append((sorted_l[i], cos_mean, grassmann))
        
        if cos_mean > similarity_threshold:
            # Store as rotation: R = prev^T @ curr (approximately, via SVD)
            # Actually: R = U_prev^T @ U_curr where U_prev has orthogonal columns
            # But U matrices from SVD ARE orthogonal, so R is k x k orthonormal
            R = prev[:, :k].T @ curr[:, :k]  # (k, k)
            rotations.append(R)
            rotation_layers.append(sorted_l[i])
            compressed_bytes += k * k * 2
        else:
            # Store fully
            independent_layers.append(sorted_l[i])
            compressed_bytes += m * k * 2
    
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    
    return {
        'L': L,
        'm': m, 'k': k,
        'overlaps': overlaps,
        'independent': len(independent_layers),
        'rotation': len(rotation_layers),
        'ratio': ratio,
        'original_MB': original_bytes / 1024**2,
        'compressed_MB': compressed_bytes / 1024**2,
    }


print("=" * 78)
print("MERA v2: Subspace Overlap + Differential Encoding")
print("=" * 78)

for holo_path, label in [(HOLO_05B, "0.5B k128"), (HOLO_27B, "27B k256")]:
    if not os.path.exists(holo_path):
        print(f"  {label}: .holo not found, skipping")
        continue
    
    print(f"\n  {label}:")
    holo = torch.load(holo_path, weights_only=False)
    
    u_groups = defaultdict(dict)
    for key, val in holo.items():
        if not key.endswith('.U') or val.ndim != 2: continue
        parts = key.split('.')
        layer_idx = None; wt = None
        for i, p in enumerate(parts):
            if p == 'layers' and i+1 < len(parts):
                try: layer_idx = int(parts[i+1])
                except: pass
            if p in ('mlp', 'self_attn', 'attn') and i+1 < len(parts):
                wt = '.'.join(parts[i:-1])
        if layer_idx is not None and wt is not None:
            u_groups[wt][layer_idx] = val
    
    total_orig_MB = 0; total_comp_MB = 0
    for wt, tensors in sorted(u_groups.items()):
        if len(tensors) < 2: continue
        result = compress_differential(tensors, similarity_threshold=0.95)
        if result:
            # Print first few overlaps
            first_overlaps = result['overlaps'][:3]
            ol_str = ' '.join([f"L{l}:{c:.3f}" for l, c, g in first_overlaps])
            print(f"    {wt}: L={result['L']} indep={result['independent']} rot={result['rotation']} ratio={result['ratio']:.1f}x overlaps=[{ol_str}...]")
            total_orig_MB += result['original_MB']
            total_comp_MB += result['compressed_MB']
    
    if total_orig_MB > 0:
        print(f"    OVERALL: {total_orig_MB:.0f}MB -> {total_comp_MB:.0f}MB ({total_orig_MB/total_comp_MB:.1f}x)")

# Show full overlap distribution for one group
print(f"\n  OVERLAP DISTRIBUTION (0.5B mlp.down_proj):")
tensors = {}
for key, val in torch.load(HOLO_05B, weights_only=False).items():
    if 'mlp.down_proj' in key and key.endswith('.U') and val.ndim == 2:
        parts = key.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i+1 < len(parts):
                try: tensors[int(parts[i+1])] = val
                except: pass
sorted_l = sorted(tensors.keys())
prev = tensors[sorted_l[0]]
for i in range(1, len(sorted_l)):
    curr = tensors[sorted_l[i]]
    cos_mean, grassmann, _ = subspace_overlap(prev, curr)
    bar = '#' * int(cos_mean * 50)
    print(f"    L{sorted_l[i-1]:>2}->L{sorted_l[i]:>2}: cos={cos_mean:.4f} grassmann={grassmann:.3f} [{bar}]")
    prev = curr

print("=" * 78)
