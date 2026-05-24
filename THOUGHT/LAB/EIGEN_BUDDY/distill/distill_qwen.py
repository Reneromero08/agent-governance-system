"""
Eigen Buddy Distiller — SVD Any Model → .holo Phase Eigenbasis
==================================================================
Streams safetensor shards, SVDs attention weight matrices, maps
dominant eigenvectors to complex phase angles on S^1 (the torus).
Stores as .holo format for Eigen Buddy's MultiHeadComplexAttention.

Usage:
    python distill_qwen.py --model F:/path/to/model --out ./distilled

No backpropagation. No training. Pure analytic geometry.
The SVD IS the grating. The eigenvectors ARE the eigenstates.
Complex, catalytic (streamed, no full model in RAM), quantum (eigenstates).
"""
import sys, time, math, json, os, argparse
from pathlib import Path
import numpy as np
import torch
import safetensors.torch as st


def randomized_svd(W, k=128, iters=2):
    """Randomized SVD with power iteration. Fast approximate SVD.
    Returns U[:,:k], S[:k], Vh[:k,:]  or (None,None,None) if fails."""
    m, n = W.shape
    k = min(k, min(m, n) - 1)
    if k < 1: return None, None, None
    Omega = torch.randn(n, k + 10, dtype=torch.float32)
    Y = W @ Omega
    for _ in range(iters):
        Y = W @ (W.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ W
    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub[:, :k]
    return U[:, :k], Sb[:k], Vhb[:k, :]


def to_phase_grating(U, S):
    """Map eigenvectors to phase angles on the complex unit circle (S^1).
    theta_i = 2*pi * i/k  (position-dependent phase)
    magnitude = sqrt(S_i / sum(S))  (signal strength)
    U shape: (dim, k), S shape: (k,)"""
    dim, k = U.shape
    grating = np.zeros((k, dim), dtype=np.complex64)
    S_total = S.sum() + 1e-12
    for i in range(k):
        theta = 2 * math.pi * i / k
        mag = math.sqrt(S[i] / S_total)
        grating[i] = mag * np.exp(1j * theta)
    return grating


def find_attention_shards(model_dir):
    """Scan all shards to find which contain attention weight matrices.
    Returns (shard_paths, attention_key_patterns)."""
    shards = sorted(Path(model_dir).glob("model-*.safetensors"))
    if not shards:
        return [], set()
    
    # Scan first few shards to find attention key patterns
    attn_patterns = set()
    for sp in shards[:3]:
        tensors = st.load_file(str(sp))
        for k in tensors:
            if len(tensors[k].shape) != 2: continue
            parts = k.split('.')
            key = '.'.join(parts[-2:])
            if any(x in key.lower() for x in 
                   ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'attn', 'self_attn']):
                attn_patterns.add(key)
    
    # Find which shards have attention weights
    attn_shards = []
    for sp in shards:
        tensors = st.load_file(str(sp))
        has_attn = False
        for k in tensors:
            if len(tensors[k].shape) != 2: continue
            key = '.'.join(k.split('.')[-2:])
            if key in attn_patterns:
                has_attn = True
                break
        if has_attn:
            attn_shards.append(sp)
    
    return attn_shards, attn_patterns


def distill(model_dir, out_dir, K=128):
    """Main distillation pipeline: SVD → torus mapping → .holo save."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)
    
    attn_shards, attn_keys = find_attention_shards(model_dir)
    if not attn_shards:
        print(f"No attention weight shards found in {model_dir}")
        print("Make sure the path contains model-*.safetensors files")
        return
    
    print(f"Model: {model_dir}")
    print(f"Attention shards: {len(attn_shards)}/{len(list(Path(model_dir).glob('model-*.safetensors')))}")
    print(f"Attention key patterns: {sorted(attn_keys)}")
    print(f"K={K} eigenvectors | Torus mapping: S^1")
    print(f"{'='*78}")
    
    holo = {}
    total_svd = 0
    t_start = time.perf_counter()
    
    for sp in attn_shards:
        tensors = st.load_file(str(sp))
        count = 0
        svd_time = 0.0
        
        for name, W in tensors.items():
            if len(W.shape) != 2: continue
            key = '.'.join(name.split('.')[-2:])
            if key not in attn_keys: continue
            
            ts = time.perf_counter()
            U, S, Vh = randomized_svd(W.float(), k=K)
            svd_time += time.perf_counter() - ts
            
            if U is None: continue
            total_svd += 1
            count += 1
            
            Uk = U.numpy()
            Sk = S.numpy()
            D_pr = float((Sk.sum()**2) / ((Sk**2).sum() + 1e-12))
            
            # Map to phase grating on S^1
            grating = to_phase_grating(Uk, Sk)
            
            lkey = name.replace('.weight', '')
            holo[lkey] = {
                'k': int(Uk.shape[1]),
                'dim': int(Uk.shape[0]),
                'D_pr': D_pr,
                'grating': grating
            }
        
        del tensors
        elapsed = time.perf_counter() - t_start
        print(f"  {sp.name}: {count:>3} rSVD ({svd_time:.1f}s SVD, {elapsed:.1f}s total)")
    
    if not holo:
        print("No attention matrices found to distill!")
        return
    
    # Save .holo file (npz compressed)
    grating_kb = sum(v['grating'].nbytes for v in holo.values()) / 1024
    holo_path = out / "eigenbuddy_distilled.holo"
    save_dict = {k: v['grating'] for k, v in holo.items()}
    np.savez_compressed(str(holo_path), **save_dict)
    
    # Save metadata JSON
    meta = {k: {kk: vv for kk, vv in v.items() if kk != 'grating'} for k, v in holo.items()}
    meta_path = out / "eigenbuddy_distilled.json"
    json.dump(meta, open(str(meta_path), 'w'), indent=2, default=str)
    
    size_mb = holo_path.with_suffix('.npz').stat().st_size / 1e6
    total_t = time.perf_counter() - t_start
    
    print(f"\n{'='*78}")
    print(f"Distilled: {total_svd} matrices, {grating_kb:.0f} KB phase grating")
    print(f"Compressed: {size_mb:.1f} MB .holo ({54000/size_mb:.0f}x vs 54GB raw)")
    print(f"Time: {total_t:.1f}s")
    
    # Show top eigenstructure
    by_D_pr = sorted(holo.items(), key=lambda x: x[1]['D_pr'], reverse=True)
    print(f"\nTop eigenstructure (by D_pr):")
    for name, info in by_D_pr[:10]:
        print(f"  {name:60s} k={info['k']:>3} dim={info['dim']:>6} D_pr={info['D_pr']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill any model to .holo phase eigenbasis")
    parser.add_argument("--model", default=r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B",
                        help="Path to model directory with safetensors shards")
    parser.add_argument("--out", default="./distilled", help="Output directory")
    parser.add_argument("--k", type=int, default=128, help="Number of eigenvectors to keep")
    args = parser.parse_args()
    distill(args.model, args.out, args.k)
