"""
GGUF MTP Distiller — Qwen 3.6 27B MTP (F16)
==============================================
MTP-trained eigenmodes have deeper temporal coherence.
Direct GGUF → .holo conversion with catalytic cache.
No dequant needed — F16 native.

Output: qwen_27b_mtp_k128.holo
"""
import os, sys, math, time, json, re
import torch, torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---- Config ----
GGUF_PATH = r"F:\LLM_Models\lmstudio-models\Qwen3.6-27B\Qwen3.6-27B-F16-mtp.gguf"
REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
OUTPUT_PATH = str(REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_mtp_k128.holo")
RANK_K = 128
OVERSAMPLE_P = 10
N_POWER_ITER = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Randomized SVD (from distill_flash.py) ----
def randomized_svd(W, k, p=OVERSAMPLE_P, n_power_iter=N_POWER_ITER):
    m, n = W.shape
    kp = min(k + p, min(m, n))
    torch.manual_seed(42)
    Omega = torch.randn(n, kp, device=DEVICE, dtype=torch.float32)
    W_gpu = W.to(DEVICE)
    Y = W_gpu @ Omega
    del Omega
    for _ in range(n_power_iter):
        Y = W_gpu @ (W_gpu.T @ Y)
        Y, _ = torch.linalg.qr(Y)
    Q, _ = torch.linalg.qr(Y)
    del Y
    if m > 50000:
        B = torch.zeros(kp, n, device=DEVICE)
        for i in range(0, m, 4096):
            end = min(i+4096, m)
            B += Q[i:end,:].T @ W_gpu[i:end,:]
    else:
        B = Q.T @ W_gpu
    B_cpu = B.cpu()
    Ub, Sb, Vhb = torch.linalg.svd(B_cpu, full_matrices=False)
    Ukp = Ub[:,:kp].to(DEVICE)
    U = (Q @ Ukp)[:,:k].cpu()
    Sk = Sb[:k].cpu()
    Vhk = Vhb[:k,:].cpu()
    del W_gpu, Q, Ub, Ukp, B_cpu
    torch.cuda.empty_cache()
    return U, Sk, Vhk


def compress(W, k=RANK_K):
    with torch.no_grad():
        U, S, Vh = randomized_svd(W.float(), k)
        SVh = S.unsqueeze(1) * Vh
    return U.half(), SVh.half()


def extract_wt(key):
    """GGUF key -> weight type for cache: blk.0.ffn_down.weight -> ffn_down.weight"""
    parts = key.split('.')
    for i, p in enumerate(parts):
        if p.isdigit() or p == 'blk':
            return '.'.join(parts[i+1:])
    return key


def main():
    from gguf import GGUFReader
    import numpy as np
    
    reader = GGUFReader(GGUF_PATH)
    data = reader.data  # memmap to raw bytes
    
    # Collect weight tensors
    weight_tensors = []
    for t in reader.tensors:
        if '.weight' in t.name and t.n_elements > 0 and len(t.shape) == 2:
            weight_tensors.append(t)
    
    print(f"Qwen 3.6 27B MTP — GGUF Distiller")
    print(f"Rank: {RANK_K} | Device: {DEVICE} | Tensors: {len(weight_tensors)}")
    print(f"MTP mode: eigenmodes trained for deeper temporal coherence")
    print()
    
    holo = {}
    cache = {}
    misses = 0; hits = 0
    t0 = time.perf_counter()
    
    for i, t in enumerate(weight_tensors):
        ts = time.perf_counter()
        
        # Read tensor from GGUF memmap
        name = t.name
        off = t.data_offset
        size = t.n_elements * 2  # F16
        raw = data[off:off+size]
        buf = np.frombuffer(raw.tobytes() if hasattr(raw,'tobytes') else bytes(raw), 
                            dtype=np.float16).astype(np.float32).copy()
        shape = tuple(int(x) for x in t.shape)
        W = torch.from_numpy(buf).reshape(shape).float()
        
        wt = extract_wt(name)
        kind = "HIT" if wt in cache else "MISS"
        
        if kind == "MISS":
            misses += 1
            Uk, SVh = compress(W)
            cache[wt] = SVh
        else:
            hits += 1
            SVh_cache = cache[wt].float()
            U_raw = W @ SVh_cache.T
            U, _ = torch.linalg.qr(U_raw)
            Uk = U[:, :RANK_K].half()
            SVh = SVh_cache.half()
        
        holo[f"{name}.U"] = Uk
        holo[f"{name}.SVh"] = SVh
        
        dt = time.perf_counter() - ts
        if dt > 2 or (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(weight_tensors)}] {wt}: {kind} {dt:.1f}s "
                  f"({time.perf_counter()-t0:.0f}s total)")

    
    total_time = time.perf_counter() - t0
    print(f"\nDistilled: {len(holo)} keys, {misses} misses, {hits} hits")
    print(f"Cache hit rate: {100*hits/max(1,misses+hits):.1f}%")
    print(f"Time: {total_time:.0f}s")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(holo, OUTPUT_PATH)
    gb = os.path.getsize(OUTPUT_PATH) / 1024**3
    print(f"Saved: {OUTPUT_PATH} ({gb:.2f} GB)")

if __name__ == "__main__":
    main()
