"""
Wormhole .holo Compression — Teleport weights through entanglement
====================================================================
Pair consecutive layers. Store first U_k fully. For subsequent layers,
store R = U_prev^T @ U_curr (k x k). Reconstruct: U_curr = U_prev @ R.

Compression: (m*k) vs (k*k) per layer = m/k ratio.
Quality: how much of U_curr lies in U_prev's subspace.
"""
import torch, time, os, math, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(r"d:\CCC 2.0\AI\agent-governance-system")
HOLO_05B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_0_5b_k128.holo")

def wormhole_compress(tensors_dict):
    """Pair layers: store first U + rotation matrices R for the rest."""
    sorted_l = sorted(tensors_dict.keys())
    if len(sorted_l) < 2: return None
    
    U_first = tensors_dict[sorted_l[0]].float()
    m, k = U_first.shape
    
    rotations = {}
    reconstructions = {}
    fidelities = {}
    residual_norms = {}
    
    prev = U_first
    for i in range(1, len(sorted_l)):
        l = sorted_l[i]
        curr = tensors_dict[l].float()
        
        # R = U_prev^T @ U_curr (optimal in least-squares sense)
        R = prev.T @ curr  # (k, k)
        rotations[l] = R
        
        # Reconstruct
        recon = prev @ R  # (m, k)
        reconstructions[l] = recon
        
        # Fidelity: cosine similarity
        fid = torch.nn.functional.cosine_similarity(
            curr.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)
        ).item()
        fidelities[l] = fid
        
        # Residual norm
        residual = curr - recon
        res_norm = residual.norm().item() / (curr.norm().item() + 1e-10)
        residual_norms[l] = res_norm
        
        prev = curr  # chain from actual for next step
    
    # Compression
    L = len(sorted_l)
    original = L * m * k * 2  # float16 U matrices
    compressed = m * k * 2 + (L-1) * k * k * 2  # first U + (L-1) rotations
    ratio = original / compressed if compressed > 0 else 1.0
    
    avg_fid = np.mean(list(fidelities.values()))
    avg_res = np.mean(list(residual_norms.values()))
    
    return {
        'L': L, 'm': m, 'k': k, 'ratio': ratio,
        'avg_fidelity': avg_fid, 'avg_residual': avg_res,
        'fidelities': fidelities, 'residuals': residual_norms,
        'original_MB': original/1024**2, 'compressed_MB': compressed/1024**2,
    }

print("=" * 78)
print("WORMHOLE .holo COMPRESSION")
print("=" * 78)

holo = torch.load(HOLO_05B, weights_only=False)
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

total_orig = 0; total_comp = 0; all_fids = []
for wt, tensors in sorted(u_groups.items()):
    result = wormhole_compress(tensors)
    if result:
        total_orig += result['original_MB']
        total_comp += result['compressed_MB']
        all_fids.extend(result['fidelities'].values())
        f1 = list(result['fidelities'].values())[:3]
        print(f"  {wt}: L={result['L']} m={result['m']} k={result['k']} ratio={result['ratio']:.1f}x fid={result['avg_fidelity']:.3f} res={result['avg_residual']:.3f} [{', '.join(f'{f:.3f}' for f in f1)}...]")

print(f"\n  OVERALL: {total_orig:.1f}MB -> {total_comp:.1f}MB ({total_orig/total_comp:.1f}x)")
print(f"  Mean fidelity: {np.mean(all_fids):.3f}")
print(f"  Min fidelity:  {np.min(all_fids):.3f}")
print(f"  Max fidelity:  {np.max(all_fids):.3f}")
print("=" * 78)
