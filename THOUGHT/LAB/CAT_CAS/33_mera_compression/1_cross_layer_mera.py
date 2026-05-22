"""
MERA Cross-Layer Compression — Holographic Model Squeeze
==========================================================
Instead of per-layer SVD, build a MERA tensor network across ALL
layers simultaneously. The gapped topological phase (Q57: constant
min-cut ~4.2) means only O(log L) tensors store the entire model.

Method: SVD across the LAYER dimension. Same-type layers share
a cross-layer eigenbasis. Compresses 28 layers into k << 28 bases.

Applies to .holo files: reads U_k from each layer, stacks into 3D
tensor, SVD across layers, stores shared basis + per-layer coefficients.

Target: 27B .holo (3.65 GB) -> MERA compressed (~50 MB theoretcial).
"""
import torch, time, os, math, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(r"d:\CCC 2.0\AI\agent-governance-system")
HOLO_27B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_27b_catalytic_k256.holo")
HOLO_05B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")

def group_by_type(holo_dict):
    """Group .holo keys by weight type (mlp.down_proj, etc)."""
    groups = defaultdict(lambda: {'U': {}, 'SVh': {}})
    for key, val in holo_dict.items():
        if not (key.endswith('.U') or key.endswith('.SVh')): continue
        if val.ndim != 2: continue
        # Extract layer number and weight type
        parts = key.split('.')
        layer_idx = None; wt = None
        for i, p in enumerate(parts):
            if p in ('layers', 'blocks') and i+1 < len(parts):
                try: layer_idx = int(parts[i+1])
                except: pass
            if p in ('mlp', 'self_attn', 'attn') and i+1 < len(parts):
                wt = '.'.join(parts[i:-1])
        if layer_idx is None or wt is None: continue
        suffix = '.U' if key.endswith('.U') else '.SVh'
        groups[wt][suffix[1:]][layer_idx] = val
    return groups

def mera_compress_group(tensors_dict, cross_layer_k=8):
    """
    Compress a group of same-type layers using cross-layer SVD.
    
    tensors_dict: {layer_idx: tensor_of_shape(m, n)}
    Returns: shared_basis, per_layer_coeffs, compression_ratio
    """
    if len(tensors_dict) < 2: return None, None, 1.0
    
    # Stack layers into 3D tensor: (num_layers, m, n)
    sorted_layers = sorted(tensors_dict.keys())
    shapes = set(t.shape for t in tensors_dict.values())
    if len(shapes) > 1:
        # Different shapes — can't stack directly
        # Use the most common shape
        from collections import Counter
        most_common = Counter(t.shape for t in tensors_dict.values()).most_common(1)[0][0]
        filtered = {k: v for k, v in tensors_dict.items() if v.shape == most_common}
        if len(filtered) < 2: return None, None, 1.0
        sorted_layers = sorted(filtered.keys())
        stack = torch.stack([filtered[l] for l in sorted_layers])  # (L, m, n)
    else:
        stack = torch.stack([tensors_dict[l] for l in sorted_layers])
    
    L, m, n = stack.shape
    
    # Flatten each layer: (L, m*n)
    flat = stack.reshape(L, m * n)
    
    # SVD across layers: find cross-layer principal components
    U, S, Vh = torch.linalg.svd(flat.float(), full_matrices=False)
    
    # Cross-layer effective dimension
    S2 = S**2; S2 = S2 / S2.sum()
    D_pr = 1.0 / (S2**2).sum().item()
    
    k = min(cross_layer_k, L, len(S))
    k = max(k, int(np.ceil(D_pr)))
    k = min(k, L)
    
    # Shared basis: top-k cross-layer components
    shared_basis = Vh[:k, :]  # (k, m*n)
    per_layer_coeffs = U[:, :k]  # (L, k)
    singular_values = S[:k]
    
    # Compression ratio
    original_bytes = L * m * n * 2  # float16
    compressed_bytes = (k * m * n + L * k + k) * 2
    ratio = original_bytes / compressed_bytes
    
    return {
        'shared_basis': shared_basis,
        'per_layer_coeffs': per_layer_coeffs,
        'singular_values': singular_values,
        'D_pr': D_pr,
        'k': k,
        'L': L,
        'compression_ratio': ratio,
        'layers': sorted_layers,
    }, per_layer_coeffs, ratio

print("=" * 78)
print("MERA CROSS-LAYER COMPRESSION")
print("=" * 78)

# Test on 0.5B first (smaller, faster)
for holo_path, label in [(HOLO_05B, "0.5B k128"), (HOLO_27B, "27B k256")]:
    if not os.path.exists(holo_path):
        print(f"  {label}: .holo not found, skipping")
        continue
    
    print(f"\n  {label}:")
    t0 = time.perf_counter()
    holo = torch.load(holo_path, weights_only=False)
    
    # Group U matrices by type
    u_groups = defaultdict(dict)
    for key, val in holo.items():
        if not key.endswith('.U') or val.ndim != 2: continue
        parts = key.split('.')
        layer_idx = None; wt = None
        for i, p in enumerate(parts):
            if p in ('layers',) and i+1 < len(parts):
                try: layer_idx = int(parts[i+1])
                except: pass
            if p in ('mlp', 'self_attn', 'attn') and i+1 < len(parts):
                wt = '.'.join(parts[i:-1])
        if layer_idx is not None and wt is not None:
            u_groups[wt][layer_idx] = val
    
    # Compress each group
    total_orig = 0; total_compressed = 0
    for wt, tensors in sorted(u_groups.items()):
        if len(tensors) < 2: continue
        result, _, ratio = mera_compress_group(tensors, cross_layer_k=8)
        if result:
            total_orig += result['L'] * tensors[list(tensors.keys())[0]].numel() * 2
            total_compressed += (result['k'] * tensors[list(tensors.keys())[0]].numel() + result['L'] * result['k'] + result['k']) * 2
            print(f"    {wt}: L={result['L']} k={result['k']} D_pr={result['D_pr']:.1f} ratio={result['compression_ratio']:.1f}x")
        else:
            print(f"    {wt}: not compressible (varying shapes or <2 layers)")
    
    if total_orig > 0:
        overall = total_orig / total_compressed
        print(f"    Overall cross-layer compression: {overall:.1f}x")
    
    dt = time.perf_counter() - t0
    print(f"    Time: {dt:.1f}s")

# MERA depth: how many cross-layer modes are needed?
print(f"\n  MERA DEPTH ANALYSIS (0.5B, mlp layers):")
holo_05b = torch.load(HOLO_05B, weights_only=False)
for wt_name in ['mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj']:
    tensors = {}
    for key, val in holo_05b.items():
        if not key.endswith('.U') or val.ndim != 2: continue
        if wt_name in key:
            parts = key.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i+1 < len(parts):
                    try: tensors[int(parts[i+1])] = val
                    except: pass
    if len(tensors) < 2: continue
    
    sorted_l = sorted(tensors.keys())
    stack = torch.stack([tensors[l] for l in sorted_l])
    L, m, n = stack.shape
    flat = stack.reshape(L, m*n)
    _, S, _ = torch.linalg.svd(flat.float(), full_matrices=False)
    S2 = S**2; S2 = S2 / S2.sum()
    cum = torch.cumsum(S2, dim=0)
    
    k50 = int((cum > 0.50).float().argmax().item() + 1)
    k90 = int((cum > 0.90).float().argmax().item() + 1)
    k95 = int((cum > 0.95).float().argmax().item() + 1)
    D_pr = 1.0 / (S2**2).sum().item()
    print(f"    {wt_name}: L={L} D_pr={D_pr:.1f} k50={k50} k90={k90} k95={k95} -> compress={L/k95:.1f}x across layers")

print("=" * 78)
