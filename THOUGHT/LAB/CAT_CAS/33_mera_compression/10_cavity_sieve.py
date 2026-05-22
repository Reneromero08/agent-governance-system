"""
Phase Cavity Eigenmode Sieve — Pre-Compression Eigenmode Pruning
==================================================================
ROADMAP_2 Track A5: Before wormhole compression, run Phase Cavity
on each weight type's U·SVh decomposition to detect and drop
dispersion eigenmodes. Uses shared mask across all layers of a
weight type so the wormhole rotation R = U_prev^T @ U_curr remains valid.

Algorithm:
  1. Load catalytic .holo
  2. For each weight type, reconstruct W = U[l] @ SVh[l] for a few sample layers
  3. Test each eigenmode: if dropping it keeps cos_sim > 0.99, it's dispersion
  4. Take intersection across sample layers → shared mask
  5. Prune U and SVh to reduced k'
  6. Output reduced .holo (ready for wormhole compression)

Usage:
  python 10_cavity_sieve.py <catalytic.holo> <output_reduced.holo>
"""
import torch, os, sys
from collections import defaultdict
from pathlib import Path

def cavity_sieve_weight_type(U_list, SVh_list, threshold=0.99, n_test=50):
    """
    Phase Cavity sieve on a weight type across multiple sampled layers.
    
    Uses the identity: (W - u_i @ vh_i) @ X = W@X - u_i @ (vh_i @ X)
    Precomputes Y = W@X once per layer, then tests each eigenmode
    via rank-1 subtraction on the projected space — O(k*n_test*m) 
    instead of O(k*m*n) per test.
    
    Returns: (kept_indices, stats)
    """
    if not U_list:
        return [], {}
    
    k = U_list[0].shape[1]
    kept = set(range(k))
    L = len(U_list)
    
    # Precompute Y_orig = W @ X for each sampled layer
    Y_orig_list = []
    X_list = []
    vhX_list = []  # SVh @ X for each layer
    for U, SVh in zip(U_list, SVh_list):
        Uf = U.float()
        SVhf = SVh.float()
        X = torch.randn(n_test, SVhf.shape[1])  # [n_test, n_in]
        Y = (Uf @ (SVhf @ X.T))  # [m, k] @ [k, n_test] = [m, n_test]
        vhX = SVhf @ X.T  # [k, n_test] — precomputed for rank-1 subtraction
        Y_orig_list.append(Y)
        vhX_list.append(vhX)
    
    for i in range(k - 1, -1, -1):
        if i not in kept: continue
        all_good = True
        for l in range(L):
            U = U_list[l].float()
            # Rank-1 subtraction in projected space:
            # Y_reduced = (W - u_i @ vh_i) @ X = W@X - u_i @ (vh_i @ X)
            Y_reduced = Y_orig_list[l] - U[:, i:i+1] @ vhX_list[l][i:i+1, :]
            
            # Cosine similarity between Y_orig and Y_reduced columns
            d = (Y_orig_list[l] * Y_reduced).sum(dim=0)
            denom = Y_orig_list[l].norm(dim=0) * Y_reduced.norm(dim=0) + 1e-9
            sim = (d / denom).mean().item()
            
            if sim < threshold:
                all_good = False
                break
        
        if all_good:
            kept.discard(i)
    
    kept_list = sorted(kept)
    stats = {
        'original_k': k,
        'kept_k': len(kept_list),
        'removed': k - len(kept_list),
        'compression': k / len(kept_list) if len(kept_list) > 0 else float('inf'),
        'kept_indices': kept_list,
    }
    return kept_list, stats


def cavity_sieve_holo(holo_path, output_path, threshold=0.99, sample_layers=5):
    """
    Load catalytic .holo, sieve all weight types, output reduced .holo.
    
    For each weight type, samples up to `sample_layers` evenly-spaced layers
    to compute the shared eigenmode mask.
    """
    print(f"Loading {holo_path}...")
    holo = torch.load(holo_path, map_location='cpu', weights_only=True)
    
    # Group U and SVh by weight type
    groups = defaultdict(lambda: {'U': {}, 'SVh': {}, 'layers': []})
    for key, val in holo.items():
        parts = key.split('.')
        if 'layers' not in parts:
            continue
        try:
            layer_idx = int(parts[parts.index('layers') + 1])
            wt = '.'.join(parts[parts.index('layers') + 2:-1])
        except (ValueError, IndexError):
            continue
        
        if key.endswith('.U'):
            groups[wt]['U'][layer_idx] = val
            groups[wt]['layers'] = sorted(set(groups[wt]['layers']) | {layer_idx})
        elif key.endswith('.SVh'):
            groups[wt]['SVh'][layer_idx] = val
    
    print(f"  Found {len(groups)} unique weight types")
    
    # Sieve each weight type
    all_stats = {}
    kept_per_wt = {}
    
    for wt, g in sorted(groups.items()):
        layers = sorted(g['U'].keys())
        n_layers = len(layers)
        
        if n_layers < 1:
            continue
        
        # Sample layers evenly
        n_sample = min(sample_layers, n_layers)
        if n_sample == 1:
            sampled = layers[:1]
        else:
            step = max(1, (n_layers - 1) // (n_sample - 1))
            sampled = [layers[i] for i in range(0, n_layers, step)][:n_sample]
        
        U_sample = [g['U'][l] for l in sampled if l in g['U']]
        SVh_sample = [g['SVh'][l] for l in sampled if l in g['SVh']]
        
        if not U_sample or not SVh_sample:
            continue
        
        kept, stats = cavity_sieve_weight_type(U_sample, SVh_sample, threshold)
        kept_per_wt[wt] = sorted(kept)
        all_stats[wt] = stats
    
    # Apply sieved masks to ALL layers and build output
    output = {}
    total_orig_k = 0
    total_new_k = 0
    
    for key, val in holo.items():
        parts = key.split('.')
        if 'layers' not in parts:
            output[key] = val  # pass-through non-layer entries
            continue
        
        try:
            wt = '.'.join(parts[parts.index('layers') + 2:-1])
        except (ValueError, IndexError):
            output[key] = val
            continue
        
        if wt not in kept_per_wt:
            output[key] = val  # no sieve data for this type
            continue
        
        kept = torch.tensor(kept_per_wt[wt], dtype=torch.long)
        
        if key.endswith('.U'):
            output[key] = val[:, kept].contiguous()
            if val.ndim == 2:
                total_orig_k += val.shape[1]
                total_new_k += len(kept)
        elif key.endswith('.SVh'):
            output[key] = val[kept, :].contiguous()
        else:
            output[key] = val
    
    # Stats
    kp_label = "K'"
    print(f"\n  {'Weight Type':<35} {'K':>6} {kp_label:>6} {'Removed':>8} {'Ratio':>6}")
    print(f"  {'-'*65}")
    avg_ratio = 0; count = 0
    for wt, stats in sorted(all_stats.items()):
        k, kp, rm = stats['original_k'], stats['kept_k'], stats['removed']
        ratio = k / kp if kp > 0 else 1.0
        print(f"  {wt:<35} {k:>6} {kp:>6} {rm:>8}  {ratio:>5.1f}x")
        avg_ratio += ratio; count += 1
    
    if count > 0:
        print(f"\n  OVERALL: K total span {total_orig_k} -> {total_new_k} ({total_orig_k/total_new_k:.1f}x" if total_new_k > 0 else "")
        print(f"  Mean eigenmode compression: {avg_ratio/count:.1f}x")
        print(f"  Sieved types: {count}/{len(groups)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output, output_path)
    in_mb = os.path.getsize(holo_path) / 1024**2
    out_mb = os.path.getsize(output_path) / 1024**2
    print(f"\n  {in_mb:.0f} MB -> {out_mb:.0f} MB ({out_mb/in_mb*100:.1f}% of original)")
    
    return kept_per_wt, all_stats


if __name__ == "__main__":
    REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
    default_in = REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo"
    default_out = REPO / "THOUGHT/LAB/CAT_CAS/33_mera_compression/qwen_27b_cavitated.holo"
    
    in_path = sys.argv[1] if len(sys.argv) > 1 else str(default_in)
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(default_out)
    
    cavity_sieve_holo(in_path, out_path, threshold=0.99, sample_layers=5)
