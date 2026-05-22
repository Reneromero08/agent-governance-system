"""
DeepSeek V4 Flash Distiller — Modular + Wormhole
==================================================
Modules:
  attention    wq_a/b, wkv, wo_a/b, kv_norm, q_norm, attn_norm
  experts      256 MoE experts × w1/w2/w3 + shared expert
  compressor   wgate, wkv, norm, ape (KV cache compression)
  indexer      wq_b, weights_proj (token indexing)
  embed_head   embed, head, hc_head_base/fn/scale

Wormhole: each module's weights are connected across layers via rotation chain.
Catalytic cache: first expert of each type gets SVD, all 255 others reuse V.
FP4 experts: dequantized on-the-fly, compressed as FP16 in .holo.
"""
import os, sys, math, time, json, re, struct
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path
from collections import defaultdict

# ---- Config ----
MODEL_DIR = r"E:\Reneshizzle SG\Models\deepseek-ai\DeepSeek-V4-Flash"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
OUTPUT_DIR = Path(r"E:\Reneshizzle SG\Models\deepseek-ai\_holo")

RANK_K = 128  # DeepSeek experts are [2048, 2048] — 128 is plenty
OVERSAMPLE_P = 10
N_POWER_ITER = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- FP4 Dequantization ----
# DeepSeek V4 uses custom FP4 format: E2M1 (2-bit exponent, 1-bit mantissa)
# Stored as packed uint8: 2 FP4 values per byte

FP4_E2M1_TABLE = None

def build_fp4_lut():
    """Build E2M1 FP4 lookup table: 16 values -> float16."""
    global FP4_E2M1_TABLE
    if FP4_E2M1_TABLE is not None:
        return FP4_E2M1_TABLE
    
    # E2M1: 2 exponent bits, 1 mantissa bit, no sign (or sign bit?)
    # Format: seeee_m where s=sign, e=exponent, m=mantissa
    # Actually DeepSeek V4 uses unsigned E2M1 for weights
    lut = np.zeros(16, dtype=np.float32)
    for i in range(16):
        sign = (i >> 3) & 1
        exp = (i >> 1) & 3
        mant = i & 1
        if exp == 0:
            # Subnormal: (-1)^sign * 2^(-1) * 0.mant
            val = ((-1)**sign) * (0.5) * (mant * 0.5)
        elif exp == 3:
            # Special values: NaN/Inf for mant=1, max for mant=0
            if mant == 0:
                val = ((-1)**sign) * 6.0  # Max normal
            else:
                val = float('nan')
        else:
            # Normal: (-1)^sign * 2^(exp-2) * 1.mant
            val = ((-1)**sign) * (2.0**(exp-2)) * (1.0 + mant * 0.5)
        lut[i] = val
    FP4_E2M1_TABLE = torch.from_numpy(lut).to(torch.float32)
    return FP4_E2M1_TABLE


def dequantize_fp4(packed_bytes, shape, dtype_str=None):
    """
    Dequantize packed FP4 uint8 tensor to float32.
    Each byte contains 2 FP4 values: high nibble first, low nibble second.
    """
    lut = build_fp4_lut()
    
    if isinstance(packed_bytes, torch.Tensor):
        packed = packed_bytes.numpy()
    else:
        packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    
    total = np.prod(shape)
    # Unpack nibbles
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    
    # Interleave: high[0], low[0], high[1], low[1], ...
    unpacked = np.empty(total, dtype=np.uint8)
    unpacked[0::2] = high[:total//2 + 1][:total//2 + (total%2)]
    unpacked[1::2] = low[:total//2]
    
    # Lookup
    result = lut[unpacked[:total]].numpy()
    return torch.from_numpy(result.reshape(shape)).float()


# ---- Randomized SVD ----

def randomized_svd(W, k, p=OVERSAMPLE_P, n_power_iter=N_POWER_ITER):
    """Randomized SVD with power iteration + QR re-orthogonalization."""
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
            end = min(i + 4096, m)
            B += Q[i:end, :].T @ W_gpu[i:end, :]
    else:
        B = Q.T @ W_gpu
    
    B_cpu = B.cpu()
    Ub, Sb, Vhb = torch.linalg.svd(B_cpu, full_matrices=False)
    
    Ukp = Ub[:, :kp].to(DEVICE)
    U = (Q @ Ukp)[:, :k].cpu()
    Sk = Sb[:k].cpu()
    Vhk = Vhb[:k, :].cpu()
    
    del W_gpu, Q, Ub, Ukp, B_cpu
    torch.cuda.empty_cache()
    
    return U, Sk, Vhk


# ---- Modular Distillation ----

MODULES = {
    "attention": {
        "patterns": ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
                     "attn.kv_norm", "attn.q_norm", "attn_norm"],
    },
    "experts": {
        "patterns": ["ffn.experts.", "ffn.shared_expert"],
        "is_expert": True,
    },
    "compressor": {
        "patterns": ["attn.compressor"],
    },
    "indexer": {
        "patterns": ["attn.indexer"],
    },
    "embed_head": {
        "patterns": ["embed.", "head.", "hc_head"],
        "is_single": True,
    },
    "aux": {
        "patterns": ["attn_sink", "scale"],  # scales, sinks
        "is_lightweight": True,
    },
}


def get_module_for_key(key):
    """Determine which module a weight key belongs to."""
    for mod_name, cfg in MODULES.items():
        for pat in cfg["patterns"]:
            if pat in key:
                return mod_name
    return "aux"


def extract_weight_type(key, module_name):
    """
    Extract normalized weight type for catalytic cache grouping.
    
    attention:  layers.0.attn.wq_a.weight -> attn.wq_a.weight
    experts:    layers.0.ffn.experts.0.w1.weight -> ffn.experts.w1.weight
    compressor: layers.0.attn.compressor.wgate.weight -> attn.compressor.wgate.weight
    """
    parts = key.split('.')
    
    # Remove layer index: layers.N -> layers
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try: int(parts[i+1])
            except: continue
            parts.pop(i+1)
            break
    
    # Remove expert index: experts.N -> experts
    for i, p in enumerate(parts):
        if p == 'experts' and i + 1 < len(parts):
            try: int(parts[i+1])
            except: continue
            parts.pop(i+1)
            break
    
    # Remove indexer compressor's layers.N
    for i, p in enumerate(parts):
        if p == 'indexer' and i + 1 < len(parts) and parts[i+1] == 'compressor':
            pass  # keep it
    
    return '.'.join(parts)


def distill_module(module_name, safetensors_files, output_path, rank_k=RANK_K):
    """
    Distill one module from safetensors files.
    Uses catalytic cache: first occurrence of each weight type gets randomized SVD.
    All subsequent occurrences reuse the cached Vh.
    """
    print(f"\n{'='*60}")
    print(f"Module: {module_name}")
    print(f"{'='*60}")
    
    holo = {}
    cache = {}  # wt_type -> (Vh_k, Sk) cached from first SVD
    misses = 0
    hits = 0
    skipped = 0
    n = 0
    t0 = time.perf_counter()
    
    for filepath in safetensors_files:
        fn = os.path.basename(filepath)
        
        with safe_open(filepath, framework='pt', device='cpu') as f:
            keys = [k for k in f.keys() if get_module_for_key(k) == module_name]
            
            for key in keys:
                if not key.endswith('.weight'):
                    continue
                
                n += 1
                ts = time.perf_counter()
                
                # Load tensor
                try:
                    t = f.get_tensor(key)
                except:
                    # Float8 tensors fail on .min() — read raw
                    info = f.get_slice(key)
                    shape = tuple(int(x) for x in info.get_shape())
                    raw = info[:]
                    buf = raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw)
                    dt = str(info.get_dtype())
                    if 'I8' in dt:
                        arr = np.frombuffer(buf, dtype=np.int8).astype(np.float32).copy()
                    elif 'F8' in dt or 'FP8' in dt:
                        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).copy()
                    else:
                        arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32).copy()
                    t = torch.from_numpy(arr).reshape(shape)
                
                if t.ndim != 2:
                    skipped += 1
                    continue
                
                # Dequantization: apply scale for I8/F8 quantized expert weights
                try:
                    info = f.get_slice(key)
                    dt = str(info.get_dtype())
                    if 'I8' in dt or 'F8' in dt:
                        scale_key = key.replace('.weight', '.scale')
                        try:
                            # Read scale tensor  
                            s_info = f.get_slice(scale_key)
                            s_raw = s_info[:]
                            s_buf = s_raw.tobytes() if hasattr(s_raw, 'tobytes') else bytes(s_raw)
                            s_dt = str(s_info.get_dtype())
                            
                            if 'F32' in s_dt:
                                s_arr = np.frombuffer(s_buf, dtype=np.float32)
                            elif 'F16' in s_dt or 'BF16' in s_dt:
                                s_arr = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32)
                            elif 'F8' in s_dt:
                                s_arr = np.frombuffer(s_buf, dtype=np.uint8).astype(np.float32)
                            else:
                                s_arr = np.frombuffer(s_buf, dtype=np.float32)
                            
                            if s_arr.size == 1:
                                t = t.float() * float(s_arr[0])
                            elif s_arr.ndim == 1:
                                t = t.float() * torch.from_numpy(s_arr).float()
                            else:
                                t = t.float() * torch.from_numpy(s_arr.reshape(-1)).float()
                        except Exception:
                            t = t.float()  # no scale, just cast
                except Exception:
                    t = t.float()
                
                if t.ndim != 2:
                    skipped += 1
                    continue
                
                # Determine weight type for cache
                wt = extract_weight_type(key, module_name)
                
                kind = "HIT" if wt in cache else "MISS"
                
                if kind == "MISS":
                    misses += 1
                    Uk, Sk, Vhk = randomized_svd(t.float(), rank_k)
                    cache[wt] = (Vhk, Sk)  # cache for reuse
                else:
                    hits += 1
                    Vh_cached, Sk_cached = cache[wt]
                    # Pure projection: U = orth(W @ Vh_cached^T)
                    U_raw = t.float() @ Vh_cached.T
                    U, _ = torch.linalg.qr(U_raw)
                    Uk = U[:, :rank_k].half()
                    Vhk = Vh_cached
                    Sk = Sk_cached
                
                holo[key + ".U"] = Uk.half()
                holo[key + ".SVh"] = (Sk.unsqueeze(1) * Vhk).half()
                
                dt = time.perf_counter() - ts
                if dt > 1 or n % 500 == 0:
                    print(f"  [{n}] {wt}: {kind} {dt:.1f}s")
    
    total_time = time.perf_counter() - t0
    print(f"  Done: {n} weights, {misses} misses, {hits} hits "
          f"({100*hits/max(1,misses+hits):.0f}% hit), {skipped} skipped")
    print(f"  Time: {total_time:.0f}s")
    
    return holo, total_time, misses, hits


def compress_to_holo(W, k=RANK_K):
    """Compress weight matrix to .holo format via randomized SVD."""
    with torch.no_grad():
        U, S, Vh = randomized_svd(W.float(), k)
        SVh = S.unsqueeze(1) * Vh
    return U.half(), SVh.half()


def run_full_distill():
    """Distill all modules from DeepSeek V4 Flash — SINGLE PASS per shard."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"DeepSeek V4 Flash Distiller — Single Pass Multi-Module")
    print(f"Model: {MODEL_DIR} | Output: {OUTPUT_DIR} | Rank: {RANK_K} | Device: {DEVICE}")
    
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    wm = idx.get('weight_map', {})
    all_files = sorted(set(os.path.join(MODEL_DIR, s) for s in wm.values()))
    print(f"Shards: {len(all_files)} — SINGLE PASS through each")
    
    # Check which modules already have valid .holo files — skip re-distilling those
    active_modules = {}
    for m in MODULES:
        existing = OUTPUT_DIR / f"deepseek_v4_flash_{m}_k{RANK_K}.holo"
        if existing.exists():
            try:
                torch.load(str(existing), map_location='cpu', weights_only=True)
                print(f"  {m}: already distilled ({existing.stat().st_size/1024**2:.0f} MB) — skipping")
                continue
            except:
                print(f"  {m}: corrupted, re-distilling")
        active_modules[m] = MODULES[m]
    
    if not active_modules:
        print("All modules already distilled. Done.")
        return
    
    # Per-module state for active modules only
    holo = {m: {} for m in active_modules}
    cache = {m: {} for m in active_modules}
    stats = {m: {'misses': 0, 'hits': 0, 'skipped': 0, 'n': 0} for m in active_modules}
    
    MODULES_ACTIVE = active_modules  # override for get_module_for_key
    
    t0 = time.perf_counter()
    
    for si, filepath in enumerate(all_files):
        fn = os.path.basename(filepath)
        print(f"\n[{si+1}/{len(all_files)}] {fn}")
        
        with safe_open(filepath, framework='pt', device='cpu') as f:
            keys = sorted(f.keys())
            
            for key in keys:
                if not key.endswith('.weight'):
                    continue
                
                module_name = get_module_for_key(key)
                if module_name not in active_modules:
                    continue  # skip keys for already-distilled modules
                
                st = stats[module_name]
                st['n'] += 1
                ts = time.perf_counter()
                
                # Load tensor
                try:
                    t = f.get_tensor(key)
                except:
                    st['skipped'] += 1
                    continue
                
                if t.ndim != 2:
                    st['skipped'] += 1
                    continue
                
                # INT8 dequant for expert weights
                try:
                    info = f.get_slice(key)
                    if str(info.get_dtype()) == 'I8':
                        raw = info[:]
                        import numpy as np
                        arr = np.frombuffer(raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw), dtype=np.int8)
                        t = torch.from_numpy(arr.astype(np.float32).copy()).reshape(tuple(int(x) for x in info.get_shape()))
                        scale_key = key.replace('.weight', '.scale')
                        try:
                            s_raw = f.get_slice(scale_key)[:]
                            s_arr = np.frombuffer(s_raw.tobytes() if hasattr(s_raw, 'tobytes') else bytes(s_raw), dtype=np.float32)
                            sv = s_arr.reshape(-1)
                            if sv.shape[0] == 1:
                                t = t * float(sv[0])
                            else:
                                t = t * torch.from_numpy(sv).float()
                        except:
                            pass
                except:
                    pass
                
                wt = extract_weight_type(key, module_name)
                ch = cache[module_name]
                kind = "HIT" if wt in ch else "MISS"
                
                if kind == "MISS":
                    st['misses'] += 1
                    Uk, SVh = compress_to_holo(t, RANK_K)
                    ch[wt] = SVh
                else:
                    st['hits'] += 1
                    SVh_cache = ch[wt].float()
                    U_raw = t.float() @ SVh_cache.T
                    U, _ = torch.linalg.qr(U_raw)
                    Uk = U[:, :RANK_K].half()
                    SVh = SVh_cache.half()
                
                holo[module_name][f"{key}.U"] = Uk
                holo[module_name][f"{key}.SVh"] = SVh
                
                dt = time.perf_counter() - ts
                if dt > 2 or st['n'] % 500 == 0:
                    total_n = sum(s['n'] for s in stats.values())
                    total_m = sum(s['misses'] for s in stats.values())
                    total_h = sum(s['hits'] for s in stats.values())
                    elapsed = time.perf_counter() - t0
                    print(f"  [{total_n}] {wt}: {kind} {dt:.1f}s | "
                          f"misses={total_m} hits={total_h} | {elapsed:.0f}s")
    
    total_time = time.perf_counter() - t0
    
    # Save each module
    print(f"\n{'='*60}")
    print(f"Single-pass complete: {total_time:.0f}s")
    print(f"{'='*60}")
    
    # Delete old corrupted files
    for m in MODULES:
        old = OUTPUT_DIR / f"deepseek_v4_flash_{m}_k{RANK_K}.holo"
        if old.exists() and m == 'experts':
            try:
                torch.load(str(old), map_location='cpu', weights_only=True)
                print(f"  {m}: keeping valid file")
            except:
                print(f"  {m}: removing corrupted file, will save fresh")
                old.unlink(missing_ok=True)
    
    for m in list(holo.keys()):
        d = holo[m]
        if not d:
            continue
        out = OUTPUT_DIR / f"deepseek_v4_flash_{m}_k{RANK_K}.holo"
        # Atomic save: write to temp, then rename
        tmp = str(out) + ".tmp"
        torch.save(d, tmp)
        os.replace(tmp, str(out))
        gb = os.path.getsize(out) / 1024**3
        s = stats[m]
        print(f"  {m:<15}: {len(d):>6} keys, {gb:.2f} GB | "
              f"misses={s['misses']}, hits={s['hits']} ({100*s['hits']/max(1,s['misses']+s['hits']):.0f}%)")
    
    # Embed config
    config_path = os.path.join(MODEL_DIR, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            torch.save({'_config': json.dumps(json.load(f)).encode('utf-8')},
                      str(OUTPUT_DIR / "deepseek_v4_flash_config.holo"))
    
    total_gb = sum(os.path.getsize(str(OUTPUT_DIR / f)) / 1024**3
                   for f in os.listdir(OUTPUT_DIR) if f.endswith('.holo') and '.tmp' not in f)
    print(f"\n  Total: {total_gb:.2f} GB in {OUTPUT_DIR}")


if __name__ == "__main__":
    run_full_distill()
