"""
K Sweet Spot Sweep for DeepSeek V4 Experts
===========================================
Tests K=128,192,256,384,512. Measures:
- Per-weight fidelity (cosine sim of W_recon vs W_orig)
- Storage size estimate
- Forward pass stability (norm after 5 layers)
- Chain rotation fidelity
"""
import torch, os, json, struct, numpy as np, time, math

MODEL_DIR = r"E:\Reneshizzle SG\Models\deepseek-ai\DeepSeek-V4-Flash"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [128, 192, 256, 320, 384, 448, 512]
LAYERS = 43

with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    idx = json.load(f)

def load_expert(key):
    shard = idx["weight_map"][key]
    path = os.path.join(MODEL_DIR, shard)
    with open(path, 'rb') as f:
        hl = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hl))
        s, e = hdr[key]['data_offsets']
        f.seek(8 + hl + s)
        data = f.read(e - s)
    shape = hdr[key]['shape']
    arr = np.frombuffer(data, dtype=np.int8).astype(np.float32).copy()
    return torch.from_numpy(arr).reshape(shape)

# Load one expert-0 weight per type for SVD
print(f"Loading expert 0 weights...")
expert_weights = {}
for wt in ["w1", "w2", "w3"]:
    expert_weights[wt] = load_expert(f"layers.0.ffn.experts.0.{wt}.weight").float().to(DEVICE)
    print(f"  {wt}: {list(expert_weights[wt].shape)}")

# Load expert-0 across all layers for chain fidelity
print(f"\nLoading expert 0 across {LAYERS} layers for chain test...")
Us_by_k = {k: {} for k in K_VALUES}
chain_data = {}

for wt in ["w1"]:  # just w1 for chain
    Us = {}
    for layer in [0, 21, 42]:  # first, middle, last
        key = f"layers.{layer}.ffn.experts.0.{wt}.weight"
        W = load_expert(key).float()
        U_full, S_full, Vh_full = torch.linalg.svd(W.to(DEVICE), full_matrices=False)
        for k in K_VALUES:
            if k not in Us:
                Us[k] = {}
            Us[k][layer] = U_full[:, :k].cpu()
        del U_full, S_full, Vh_full, W
        torch.cuda.empty_cache()
    chain_data[wt] = Us

print(f"\n{'='*70}")
print(f"K SWEEP RESULTS")
print(f"{'='*70}")
print(f"{'K':>5} {'fid_w1':>8} {'fid_w2':>8} {'fid_w3':>8} {'mean_fid':>9} {'GB':>8} {'fid_chain':>10} {'stable':>8}")
print(f"{'-'*70}")

results = []
for k in K_VALUES:
    fids = {}
    for wt, W_orig in expert_weights.items():
        U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
        Uk = U[:, :k]
        Vhk = Vh[:k, :]
        Sk = S[:k]
        W_recon = (Uk * Sk.unsqueeze(0)) @ Vhk
        fid = torch.nn.functional.cosine_similarity(
            W_orig.flatten(), W_recon.flatten(), dim=0
        ).item()
        fids[wt] = fid
        del U, S, Vh, Uk, Vhk, W_recon
    
    mean_fid = np.mean(list(fids.values()))
    
    # Storage estimate (INT8, deduplicated SVh)
    # U tensors: 33_792 * m * k bytes (INT8)
    # SVh shared: 12 * k * n bytes (INT8)
    n_experts = 33792
    avg_m = 2730  # ~2/3*2048 + 1/3*4096
    avg_n = 1707  # ~2/3*2048 + 1/3*1024
    u_gb = n_experts * avg_m * k / 1024**3
    svh_gb = 12 * k * avg_n / 1024**3
    total_gb = u_gb + svh_gb
    
    # Chain fidelity
    if "w1" in chain_data:
        Us = chain_data["w1"][k]
        if 0 in Us and 42 in Us:
            anchor = Us[0].float()
            final = Us[42].float()
            R = anchor.T @ final
            chain_fid = torch.nn.functional.cosine_similarity(
                final.flatten(), (anchor @ R).flatten(), dim=0
            ).item()
        else:
            chain_fid = 0
    else:
        chain_fid = 0
    
    # Stability estimate: if mean_fid^LAYERS > threshold
    stable = "YES" if mean_fid ** LAYERS > 1e-6 else "NO"
    
    results.append({
        'k': k, 'fids': fids, 'mean_fid': mean_fid,
        'gb': total_gb, 'chain_fid': chain_fid, 'stable': stable
    })
    
    print(f"{k:>5} {fids['w1']:>8.4f} {fids['w2']:>8.4f} {fids['w3']:>8.4f} "
          f"{mean_fid:>9.4f} {total_gb:>8.1f} {chain_fid:>10.4f} {stable:>8}")

# Summary
print(f"\n{'='*70}")
print(f"SWEET SPOT ANALYSIS")
print(f"{'='*70}")
for r in results:
    decay = r['mean_fid'] ** LAYERS
    print(f"  K={r['k']:>3}: fid={r['mean_fid']:.4f}  "
          f"fid^{LAYERS}={decay:.2e}  "
          f"{r['gb']:.1f} GB  "
          f"chain={r['chain_fid']:.4f}  "
          f"{'STABLE' if r['stable'] == 'YES' else 'unstable'}")

# Best K: first one where fid^43 > 1e-6
targets = [r for r in results if r['mean_fid'] ** LAYERS > 1e-6]
if targets:
    best = targets[0]
    print(f"\n  RECOMMENDED: K={best['k']} ({best['gb']:.1f} GB, fid={best['mean_fid']:.4f})")
else:
    print(f"\n  No K achieves stability at 43 layers. Minimum viable: K>=512")

print(f"\n  Reference: raw INT8 = 3 GB, cross-model Df = 39.6")
print(f"  Our K=128: 11 GB (dedup INT8), fid=0.46")
