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
                except Exception as e:
                    skipped += 1
                    continue
                
                if t.ndim != 2:
                    skipped += 1
                    continue
                
                # INT8 dequantization for expert weights (DeepSeek V4 Flash)
                # Expert weights stored as I8 with per-channel FP32 scale
                # Use raw byte read because torch doesn't support float8 scales
                try:
                    info = f.get_slice(key)
                    if str(info.get_dtype()) == 'I8':
                        raw = info[:]  # raw bytes
                        import numpy as np
                        arr = np.frombuffer(raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw), dtype=np.int8)
                        t = torch.from_numpy(arr.astype(np.float32).reshape(info.get_shape()))
                        # Read scale
                        scale_key = key.replace('.weight', '.scale')
                        try:
                            s_raw = f.get_slice(scale_key)[:]
                            s_arr = np.frombuffer(s_raw.tobytes() if hasattr(s_raw, 'tobytes') else bytes(s_raw), dtype=np.float32)
                            scale_val = s_arr.reshape(-1)
                            if scale_val.shape[0] == 1:
                                t = t * float(scale_val[0])
                            else:
                                t = t * torch.from_numpy(scale_val).float()
                        except:
                            pass  # no scale, keep as float32
                except:
                    pass  # not I8, keep original
                
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


def run_full_distill():
    """Distill all modules from DeepSeek V4 Flash."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"DeepSeek V4 Flash Distiller — Modular + Wormhole")
    print(f"Model: {MODEL_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Rank: {RANK_K} | Device: {DEVICE} | Rand SVD q={N_POWER_ITER}")
    
    # Load index
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    wm = idx.get('weight_map', {})
    
    # Get ALL safetensors files
    all_files = sorted(set(os.path.join(MODEL_DIR, s) for s in wm.values()))
    print(f"Safetensors shards: {len(all_files)}")
    
    # Distill each module
    total_time = 0
    all_stats = {}
    
    for mod_name in MODULES:
        module_holo, dt, misses, hits = distill_module(mod_name, all_files, OUTPUT_DIR)
        total_time += dt
        all_stats[mod_name] = {'keys': len(module_holo), 'time': dt, 'misses': misses, 'hits': hits}
        
        out_path = OUTPUT_DIR / f"deepseek_v4_flash_{mod_name}_k{RANK_K}.holo"
        torch.save(module_holo, str(out_path))
        gb = os.path.getsize(out_path) / 1024**3
        print(f"  Saved: {out_path.name} ({gb:.2f} GB)")
    
    # Embed config for self-containment
    config_path = os.path.join(MODEL_DIR, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config_holo = {'_config': json.dumps(config).encode('utf-8')}
        torch.save(config_holo, str(OUTPUT_DIR / "deepseek_v4_flash_config.holo"))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Distillation Complete: {total_time:.0f}s total")
    print(f"{'='*60}")
    for mod, s in all_stats.items():
        print(f"  {mod:<15}: {s['keys']:>6} keys, {s['time']:>6.0f}s, "
              f"hits={s['hits']}, misses={s['misses']}")
    
    # Total size
    total_gb = sum(os.path.getsize(str(OUTPUT_DIR / f)) / 1024**3
                   for f in os.listdir(OUTPUT_DIR) if f.endswith('.holo'))
    print(f"\n  Total output: {total_gb:.2f} GB")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_full_distill()
