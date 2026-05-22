"""
Wormhole .holo Compressor — Rotation + Quantized Residual
===========================================================
Compresses .holo U_k matrices using:
  1. Wormhole rotation: R = U_prev^T @ U_curr (k x k, small)
  2. 2-bit quantized residual: preserves layer individuality
  
Compression: 3.4-5.4x at 0.81-0.88 fidelity.
Usage: python 5_wormhole_compressor.py <input.holo> <output.holo>
"""
import torch, math, numpy as np, os, sys
from collections import defaultdict
from pathlib import Path

def compress_holo(holo_dict, rotation_threshold=0.5, quant_bits=2):
    """
    Compress .holo dict using wormhole rotation + quantized residual.
    
    Returns: compressed_dict, stats
    """
    # Group U matrices by weight type
    u_groups = defaultdict(dict)
    for key, val in holo_dict.items():
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
            u_groups[wt][layer_idx] = val.float()
    
    compressed = {}
    stats = {'groups': {}, 'total_orig_MB': 0, 'total_comp_MB': 0, 'fidelities': {}}
    
    # Process each group
    for wt, tensors in sorted(u_groups.items()):
        sorted_l = sorted(tensors.keys())
        if len(sorted_l) < 2:
            # Single layer — store as-is
            for l in sorted_l:
                compressed[f"{wt}.L{l}.U"] = tensors[l]
            continue
        
        first = tensors[sorted_l[0]]
        m, k = first.shape
        L = len(sorted_l)
        
        # Store first layer fully
        compressed[f"{wt}.L{sorted_l[0]}.U"] = first.half()
        
        prev = first
        fids_rot = []; fids_quant = []
        orig_bits = L * m * k * 16
        comp_bits = m * k * 16  # first layer
        
        for i in range(1, L):
            l = sorted_l[i]
            curr = tensors[l]
            
            # Rotation
            R = prev.T @ curr  # (k, k)
            recon_rot = prev @ R
            fid_rot = torch.nn.functional.cosine_similarity(
                curr.flatten().unsqueeze(0), recon_rot.flatten().unsqueeze(0)
            ).item()
            
            # Residual
            residual = curr - recon_rot
            res_max = residual.abs().max().item()
            
            if fid_rot > rotation_threshold:
                # High fidelity — rotation only, no residual needed
                compressed[f"{wt}.L{l}.R"] = R.half()
                comp_bits += k * k * 16
                fid_quant = fid_rot
            else:
                # Quantize and reconstruct
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(res_max, 1e-6)
                residual_norm = residual / max(res_max, 1e-6)
                diffs = residual_norm.unsqueeze(-1) - levels.view(1, 1, -1)
                idx = diffs.abs().argmin(dim=-1).to(torch.uint8)
                residual_quant = levels[idx.long()]  # quantized residual
                
                compressed[f"{wt}.L{l}.R"] = R.half()
                compressed[f"{wt}.L{l}.res_idx"] = idx
                compressed[f"{wt}.L{l}.res_max"] = torch.tensor(res_max)
                comp_bits += k * k * 16 + m * k * quant_bits + 16
                
                recon_quant = recon_rot + residual_quant
                fid_quant = torch.nn.functional.cosine_similarity(
                    curr.flatten().unsqueeze(0), recon_quant.flatten().unsqueeze(0)
                ).item()
            
            fids_rot.append(fid_rot)
            fids_quant.append(fid_quant)
            prev = curr
        
        ratio = orig_bits / comp_bits if comp_bits > 0 else 1.0
        fid_rot_avg = np.mean(fids_rot)
        fid_quant_avg = np.mean(fids_quant)
        
        stats['groups'][wt] = {
            'L': L, 'm': m, 'k': k, 'ratio': ratio,
            'fid_rot': fid_rot_avg, 'fid_quant': fid_quant_avg,
            'orig_MB': orig_bits / 8 / 1024**2,
            'comp_MB': comp_bits / 8 / 1024**2,
        }
        stats['total_orig_MB'] += orig_bits / 8 / 1024**2
        stats['total_comp_MB'] += comp_bits / 8 / 1024**2
        stats['fidelities'][wt] = fids_quant
    
    # Copy non-U entries as-is
    for key, val in holo_dict.items():
        if not key.endswith('.U') or key in compressed:
            if not any(key.startswith(k.split('.L')[0] + '.') for k in compressed if '.L' in k):
                if key not in compressed and '.U' not in key and '.SVh' not in key:
                    compressed[key] = val
    
    return compressed, stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python 5_wormhole_compressor.py <input.holo> <output.holo>")
        REPO = Path(r"d:\CCC 2.0\AI\agent-governance-system")
        input_path = str(REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_0_5b_k128.holo")
        output_path = str(REPO / "THOUGHT/LAB/CAT_CAS/33_mera_compression/qwen_0_5b_wormhole.holo")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return
    
    print(f"Loading {input_path}...")
    holo = torch.load(input_path, weights_only=False)
    
    print(f"Compressing...")
    compressed, stats = compress_holo(holo)
    
    print(f"\n  {'Group':<30} {'L':>4} {'fid_rot':>8} {'fid+res':>8} {'ratio':>6}")
    print(f"  {'-'*60}")
    for wt, s in sorted(stats['groups'].items()):
        print(f"  {wt:<30} {s['L']:>4} {s['fid_rot']:>8.3f} {s['fid_quant']:>8.3f} {s['ratio']:>5.1f}x")
    
    print(f"\n  OVERALL: {stats['total_orig_MB']:.0f}MB -> {stats['total_comp_MB']:.0f}MB ({stats['total_orig_MB']/stats['total_comp_MB']:.1f}x)")
    print(f"  Mean fidelity: {np.mean([np.mean(f) for f in stats['fidelities'].values()]):.3f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(compressed, output_path)
    out_size = os.path.getsize(output_path) / 1024**2
    print(f"\n  Saved: {output_path} ({out_size:.1f} MB)")

if __name__ == "__main__":
    main()
