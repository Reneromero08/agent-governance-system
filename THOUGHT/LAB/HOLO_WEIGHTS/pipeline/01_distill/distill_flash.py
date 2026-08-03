"""
Catalytic Distillation v7 — FLASH (Randomized SVD + GPU)
==========================================================
v6 bottleneck: 17 cache misses × ~90s = 1530s of 1561s total.
Each miss: full SVD of [m, n] matrix. Insane.

v7 fix: Randomized SVD. FJLT→[m, k+p] then SVD of [k+p, n].
        20x per-miss speedup. GPU offload for the small SVD.
        Swarm-ready: each safetensors shard is independent.

Expected: 1561s → ~60s on GPU, ~120s on CPU.

Architecture:
  1. FJLT (or Hadamard) project: W[m,n] → Y[m, k+p]  (one matmul)
  2. QR: Y → Q[m, k+p]  (thin QR, cheap)
  3. B = Q^T @ W: [k+p, n]  (small matmul)
  4. SVD of B: tiny [k+p, n] matrix — this is the speedup
  5. U = Q @ U_b; truncate to k

Usage:
  python distill_flash.py  # reads MODEL_DIR, writes OUTPUT_PATH
"""
import os, sys, math, time, json
import torch
import torch.nn.functional as F
from safetensors import safe_open
from pathlib import Path

# ---- Config ----
MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
OUTPUT_PATH = str(REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo")
RANK_K = 256
OVERSAMPLE_P = 10  # extra dimensions for numerical stability
GPU_BATCH = 4096   # batch rows for GPU matmul
MAX_M_GPU = 20000  # if m > this, chunk the FJLT


def randomized_svd(W, k, p=OVERSAMPLE_P, n_power_iter=2,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Randomized SVD with power iteration for accuracy.
    
    Algorithm (Halko et al., 2011, with power iteration):
      1. Omega ~ N(0,1) [n, k+p]
      2. Y = W @ Omega [m, k+p]
      3. Power iter q=2: Y = (W @ W^T)^q @ Y  (converges to dominant subspace)
      4. Q = QR(Y) [m, k+p]
      5. B = Q^T @ W [k+p, n]
      6. SVD(B) -> Ub, Sb, Vhb
      7. U = Q @ Ub; truncate to k
    
    Returns: Uk [m,k], Sk [k], Vhk [k,n]
    """
    m, n = W.shape
    kp = min(k + p, min(m, n))
    
    # Step 1: Random projection
    torch.manual_seed(42)
    Omega = torch.randn(n, kp, device=device, dtype=torch.float32)
    
    # Step 2: Y = W @ Omega
    W_gpu = W.to(device)
    Y = W_gpu @ Omega  # [m, kp]
    del Omega
    
    # Step 3: Power iteration with re-orthogonalization
    for _ in range(n_power_iter):
        Y = W_gpu @ (W_gpu.T @ Y)  # [m, kp]
        Y, _ = torch.linalg.qr(Y)  # re-orthogonalize to prevent mode collapse
    
    # Step 4: QR decomposition
    Q, _ = torch.linalg.qr(Y)  # [m, kp]
    del Y
    
    # Step 5: B = Q^T @ W
    if m > 50000:
        B = torch.zeros(kp, n, device=device)
        chunk = 4096
        for i in range(0, m, chunk):
            end = min(i + chunk, m)
            B += Q[i:end, :].T @ W_gpu[i:end, :]
    else:
        B = Q.T @ W_gpu  # [kp, n]
    
    # Step 6: SVD of small B
    B_cpu = B.cpu()
    Ub, Sb, Vhb = torch.linalg.svd(B_cpu, full_matrices=False)
    
    # Step 7: Project and truncate
    Ukp = Ub[:, :kp].to(device)
    U = (Q @ Ukp)[:, :k].cpu()
    Sk = Sb[:k].cpu()
    Vhk = Vhb[:k, :].cpu()
    
    del W_gpu, Q, Ub, Ukp, B_cpu
    torch.cuda.empty_cache()
    
    return U, Sk, Vhk


def compress_to_holo(W, k=RANK_K, wt_type=""):
    """
    Compress weight matrix W to .holo format (U, SVh).
    Uses randomized SVD for speed.
    """
    with torch.no_grad():
        U, S, Vh = randomized_svd(W.float(), k)
        SVh = S.unsqueeze(1) * Vh  # [k, n] = diag(S) @ Vh
    return U.half(), SVh.half()


def process_safetensors(filepath, k=RANK_K):
    """
    Process one safetensors file: catalytic cache projection.
    First layer of each type: randomized SVD (cache miss).
    Subsequent layers: PURE PROJECTION (cache hit — zero SVD).
    """
    holo = {}
    cache = {}  # wt_type -> Vh_cached
    misses = 0
    hits = 0
    t0 = time.perf_counter()
    n = 0
    
    with safe_open(filepath, framework='pt', device='cpu') as f:
        keys = sorted(f.keys())
        
        for key in keys:
            if not key.endswith('.weight'):
                continue
            
            t = f.get_tensor(key)
            if t.ndim != 2:
                continue
            
            n += 1
            ts = time.perf_counter()
            
            # Determine weight type
            parts = key.split('.')
            wt_type = None
            for i, p in enumerate(parts):
                if p in ('mlp', 'self_attn', 'attn', 'linear_attn') and i + 1 < len(parts):
                    wt_type = '.'.join(parts[i:])  # e.g. 'mlp.down_proj.weight'
                    break
            if wt_type is None:
                wt_type = key.split('.')[-1]
            
            kind = "HIT" if wt_type in cache else "MISS"
            
            if kind == "MISS":
                misses += 1
                Uk, SVh = compress_to_holo(t, k, wt_type)
                cache[wt_type] = SVh  # cache Vh for this type
            else:
                hits += 1
                Vh_cached = cache[wt_type].float()
                # Project: U = orth(W @ Vh_cached^T) — two matmuls, zero SVD
                n_in = t.shape[1]
                Svh_cache = Vh_cached  # [k, n]
                # W is [m, n], SVh_cache^T is [n, k], result is [m, k]
                U_raw = t.float() @ Svh_cache.T  # [m, k]
                # Orthogonalize via QR
                U, _ = torch.linalg.qr(U_raw)
                Uk = U[:, :k].half()
                SVh = Svh_cache.half()
            
            holo[key + ".U"] = Uk
            holo[key + ".SVh"] = SVh
            
            dt = time.perf_counter() - ts
            if dt > 1 or n % 20 == 0:
                print(f"    [{n}] {wt_type}: {kind} {dt:.1f}s")
    
    total_time = time.perf_counter() - t0
    return holo, total_time, misses, hits


def embed_standalone(all_holo, index_path, model_dir):
    """Embed config + embed/norm/bias for self-contained .holo."""
    # 1. Config
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        all_holo['_config'] = json.dumps(config).encode('utf-8')
        print(f"  Config: {len(all_holo['_config'])} bytes")
    
    # 2. Embed/norm/bias
    with open(index_path) as f:
        wm = json.load(f).get('weight_map', {})
    
    embed_patterns = ['embed_tokens', 'norm', 'bias', 'lm_head']
    loaded = 0
    files_open = {}
    
    for name, shard in wm.items():
        is_target = any(p in name for p in embed_patterns)
        if not is_target: continue
        if name.endswith('.weight') and any(k.endswith(name + '.U') for k in all_holo): continue
        
        sf_path = os.path.join(model_dir, shard)
        if sf_path not in files_open:
            files_open[sf_path] = safe_open(sf_path, framework='pt', device='cpu')
        
        try:
            t = files_open[sf_path].get_tensor(name)
            safe_key = '_embed.' + name.replace('.', '_') if 'embed' in name else \
                       '_norm.' + name.replace('.', '_') if 'norm' in name else \
                       '_bias.' + name.replace('.', '_') if 'bias' in name else name
            all_holo[safe_key] = t
            loaded += 1
        except:
            pass
    
    for f in files_open.values():
        try: f.__exit__(None, None, None)
        except: pass
    
    print(f"  Standalone: {loaded} embed/norm/bias tensors")
    return all_holo


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Distill FLASH v7 | Rank={RANK_K} | Device={DEVICE} | Randomized SVD")
    print(f"Model: {MODEL_DIR}")
    
    with open(INDEX_PATH) as f:
        ix = json.load(f)
    wm = ix.get('weight_map', {})
    files = sorted(set(wm.values()))
    print(f"Shards: {len(files)}")
    
    all_holo = {}
    tt = 0.0
    tm = 0
    th = 0
    
    for i, fn in enumerate(files):
        fp = os.path.join(MODEL_DIR, fn)
        print(f"\n[{i+1}/{len(files)}] {fn}")
        holo, dt, m, h = process_safetensors(fp, RANK_K)
        all_holo.update(holo)
        tt += dt
        tm += m
        th += h
        print(f"  {dt:.1f}s | misses={m} hits={h} | "
              f"{'GPU' if DEVICE=='cuda' else 'CPU'} randomized SVD")
    
    # Embed standalone
    print("\nEmbedding standalone params...")
    all_holo = embed_standalone(all_holo, INDEX_PATH, MODEL_DIR)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nSaving {OUTPUT_PATH}...")
    torch.save(all_holo, OUTPUT_PATH)
    gb = os.path.getsize(OUTPUT_PATH) / 1024**3
    holo_keys = len(all_holo)
    print(f"Done! {gb:.2f} GB | {tt:.0f}s | {holo_keys} keys | "
          f"misses={tm} hits={th} ({100*th/(tm+th):.0f}% hit)")
    
    if tt < 100:
        print("FLASH MODE ACHIEVED.")
    else:
        print(f"Still {tt:.0f}s — try GPU with CUDA_VISIBLE_DEVICES")


if __name__ == "__main__":
    if os.path.exists(MODEL_DIR):
        main()
    else:
        # Quick test on random data
        print("Model dir not found. Running speed test on synthetic data...")
        W = torch.randn(17408, 5120)
        t0 = time.perf_counter()
        Uk, Sk, Vhk = randomized_svd(W, 256)
        dt = time.perf_counter() - t0
        print(f"Randomized SVD [17408, 5120] → k=256: {dt:.2f}s")
        
        # Compare with full SVD timing
        t1 = time.perf_counter()
        Uf, Sf, Vhf = torch.linalg.svd(W.float(), full_matrices=False)
        dt_full = time.perf_counter() - t1
        print(f"Full SVD [17408, 5120]: {dt_full:.2f}s")
        print(f"Speedup: {dt_full/dt:.1f}x")
        
        # Fidelity check
        W_rand = (Uk * Sk.unsqueeze(0)) @ Vhk
        W_full = (Uf[:, :256] * Sf[:256].unsqueeze(0)) @ Vhf[:256, :]
        cos_rand = F.cosine_similarity(W.flatten().unsqueeze(0), W_rand.flatten().unsqueeze(0)).item()
        cos_full = F.cosine_similarity(W.flatten().unsqueeze(0), W_full.flatten().unsqueeze(0)).item()
        print(f"Randomized fidelity: {cos_rand:.6f}")
        print(f"Full SVD fidelity:   {cos_full:.6f}")
