"""
DeepSeek V4 Flash Distiller v2 — Deduplicated SVh + INT8 Quantization
=======================================================================
v1 improvements:
  - SVh deduplication: store once per weight type (catalytic cache), not 33,792 copies
  - INT8 quantization: U + SVh at 8-bit with scale, 2x smaller at 0.9999 fidelity
  - Modular single-pass: all 6 modules in one sweep
  - QR blacklist: experts require QR, other modules use fast NO_QR

Output format:
  {module}_k{K}.holo:
    _svh: {wt: int8_tensor}           — shared SVh per weight type
    _scales: {wt: {svh_scale: float}}  — dequant scales
    key.U: int8_tensor                 — quantized U per key  
    key.scale: float                   — dequant scale for this U
"""
import os, sys, time, json, re, struct
import torch, torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path
from collections import defaultdict

# ---- Config ----
MODEL_DIR = r"E:\Reneshizzle SG\Models\deepseek-ai\DeepSeek-V4-Flash"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
OUTPUT_DIR = Path(r"E:\Reneshizzle SG\Models\deepseek-ai\_holo")

RANK_K = 128
OVERSAMPLE_P = 10
N_POWER_ITER = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NO_QR_BLACKLIST = {'experts'}  # QR required for experts
INT8_BITS = 8

# ---- Module Definitions ----
MODULES = {
    "attention": {"patterns": ["attn.wq_a","attn.wq_b","attn.wkv","attn.wo_a","attn.wo_b",
                                "attn.kv_norm","attn.q_norm","attn_norm"]},
    "experts":   {"patterns": ["ffn.experts.","ffn.shared_expert"], "is_expert": True},
    "compressor":{"patterns": ["attn.compressor"]},
    "indexer":   {"patterns": ["attn.indexer"]},
    "embed_head":{"patterns": ["embed.","head.","hc_head"], "is_single": True},
    "aux":       {"patterns": ["attn_sink","scale"], "is_lightweight": True},
}

def get_module_for_key(key):
    for m, cfg in MODULES.items():
        for pat in cfg["patterns"]:
            if pat in key: return m
    return "aux"

def extract_weight_type(key, module_name):
    parts = key.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            try: int(parts[i+1])
            except: continue
            parts.pop(i+1); break
    for i, p in enumerate(parts):
        if p == 'experts' and i+1 < len(parts):
            try: int(parts[i+1])
            except: continue
            parts.pop(i+1); break
    return '.'.join(parts)

# ---- Randomized SVD (FLASH) ----
def randomized_svd(W, k):
    m, n = W.shape; kp = min(k + OVERSAMPLE_P, min(m, n))
    torch.manual_seed(42)
    Omega = torch.randn(n, kp, device=DEVICE, dtype=torch.float32)
    W_gpu = W.to(DEVICE); Y = W_gpu @ Omega; del Omega
    for _ in range(N_POWER_ITER):
        Y = W_gpu @ (W_gpu.T @ Y); Y, _ = torch.linalg.qr(Y)
    Q, _ = torch.linalg.qr(Y); del Y
    if m > 50000:
        B = torch.zeros(kp, n, device=DEVICE)
        for i in range(0, m, 4096):
            B += Q[i:min(i+4096,m),:].T @ W_gpu[i:min(i+4096,m),:]
    else:
        B = Q.T @ W_gpu
    B_cpu = B.cpu(); Ub, Sb, Vhb = torch.linalg.svd(B_cpu, full_matrices=False)
    Ukp = Ub[:,:kp].to(DEVICE); U = (Q @ Ukp)[:,:k].cpu()
    Sk = Sb[:k].cpu(); Vhk = Vhb[:k,:].cpu()
    del W_gpu, Q, Ub, Ukp, B_cpu; torch.cuda.empty_cache()
    return U, Sk, Vhk

def compress_to_holo(W, k=RANK_K):
    U, S, Vh = randomized_svd(W.float(), k)
    SVh = S.unsqueeze(1) * Vh
    return U, SVh

# ---- INT8 Quantization ----
def int8_quantize(t):
    """Quantize float32 tensor to int8 with scale. Returns (q_int8, scale)."""
    t_f = t.float()
    vmax = t_f.abs().max().item()
    if vmax < 1e-8:
        return torch.zeros_like(t_f, dtype=torch.int8), 1.0
    scale = vmax / 127.0
    q = torch.clamp(torch.round(t_f / scale), -127, 127).to(torch.int8)
    return q, float(scale)

def int8_dequant(q, scale):
    """Dequantize int8 to float32."""
    return q.float() * scale

# ---- Main Distiller ----
def run_full_distill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(min(4, os.cpu_count()))
    
    print(f"DeepSeek V4 Flash Distiller v2 — Deduplicated SVh + INT8")
    print(f"Model: {MODEL_DIR} | Rank: {RANK_K} | Device: {DEVICE}")
    print(f"Threads: {torch.get_num_threads()} | Quant: INT{INT8_BITS}")
    
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    wm = idx.get('weight_map', {})
    all_files = sorted(set(os.path.join(MODEL_DIR, s) for s in wm.values()))
    print(f"Shards: {len(all_files)}")
    
    # Check existing modules
    active_modules = {}
    for m in MODULES:
        existing = OUTPUT_DIR / f"deepseek_v4_flash_{m}_k{RANK_K}.holo"
        if existing.exists():
            try:
                d = torch.load(str(existing), map_location='cpu', weights_only=True)
                if '_svh' in d:
                    print(f"  {m}: already distilled ({existing.stat().st_size/1024**2:.0f} MB) — skipping")
                    continue
            except: pass
        active_modules[m] = MODULES[m]
    
    if not active_modules:
        print("All modules already distilled. Done.")
        return
    
    # Per-module state
    holo_U = {m: {} for m in active_modules}      # key -> int8 U
    holo_scales = {m: {} for m in active_modules}  # key -> scale
    svh_shared = {m: {} for m in active_modules}   # wt -> int8 SVh
    svh_scales = {m: {} for m in active_modules}   # wt -> scale
    cache = {m: {} for m in active_modules}         # wt -> float32 SVh (working)
    stats = {m: {'misses':0, 'hits':0, 'skipped':0, 'n':0} for m in active_modules}
    
    t0 = time.perf_counter()
    
    for si, filepath in enumerate(all_files):
        fn = os.path.basename(filepath)
        print(f"\n[{si+1}/{len(all_files)}] {fn}")
        
        with safe_open(filepath, framework='pt', device='cpu') as f:
            keys = sorted(f.keys())
            
            for key in keys:
                if not key.endswith('.weight'): continue
                module_name = get_module_for_key(key)
                if module_name not in active_modules: continue
                
                st = stats[module_name]; st['n'] += 1
                ts = time.perf_counter()
                
                # ---- Load tensor (INT8 fallback) ----
                try:
                    t = f.get_tensor(key)
                except:
                    info = f.get_slice(key)
                    shape = tuple(int(x) for x in info.get_shape())
                    raw = info[:]
                    buf = raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw)
                    dt = str(info.get_dtype())
                    if 'I8' in dt.upper():
                        arr = np.frombuffer(buf, dtype=np.int8).astype(np.float32).copy()
                    elif 'F8' in dt:
                        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).copy()
                    else:
                        arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32).copy()
                    t = torch.from_numpy(arr).reshape(shape).float()
                    try:
                        scale_key = key.replace('.weight', '.scale')
                        s_info = f.get_slice(scale_key)
                        s_raw = s_info[:]
                        s_buf = s_raw.tobytes() if hasattr(s_raw, 'tobytes') else bytes(raw)
                        s_dt = str(s_info.get_dtype())
                        if 'F32' in s_dt: s_arr = np.frombuffer(s_buf, dtype=np.float32)
                        elif 'F16' in s_dt: s_arr = np.frombuffer(s_buf, dtype=np.float16).astype(np.float32)
                        else: s_arr = np.frombuffer(s_buf, dtype=np.float32)
                        if s_arr.size == 1: t = t * float(s_arr[0])
                        else: t = t * torch.from_numpy(s_arr).float().view(-1, 1)
                    except: pass
                
                if t.ndim != 2: st['skipped'] += 1; continue
                
                # ---- Catalytic SVD or Projection ----
                wt = extract_weight_type(key, module_name)
                ch = cache[module_name]
                kind = "HIT" if wt in ch else "MISS"
                
                if kind == "MISS":
                    st['misses'] += 1
                    Uk, SVh = compress_to_holo(t, RANK_K)
                    ch[wt] = SVh  # cache float32 SVh
                    # Quantize and store shared SVh
                    q_svh, sc_svh = int8_quantize(SVh)
                    svh_shared[module_name][wt] = q_svh
                    svh_scales[module_name][wt] = sc_svh
                else:
                    st['hits'] += 1
                    SVh_cache = ch[wt].float()
                    U_raw = t.float() @ SVh_cache.T
                    do_qr = module_name in NO_QR_BLACKLIST
                    if do_qr:
                        U, _ = torch.linalg.qr(U_raw)
                        Uk = U[:, :RANK_K]
                    else:
                        Uk = F.normalize(U_raw.float(), p=2, dim=0)[:, :RANK_K]
                    # SVh already stored once for this weight type
                
                # Quantize U to int8
                q_u, sc_u = int8_quantize(Uk)
                holo_U[module_name][f"{key}.U"] = q_u
                holo_scales[module_name][f"{key}.scale"] = sc_u
                
                dt = time.perf_counter() - ts
                if dt > 2 or st['n'] % 500 == 0:
                    tn = sum(s['n'] for s in stats.values())
                    tm = sum(s['misses'] for s in stats.values())
                    th = sum(s['hits'] for s in stats.values())
                    el = time.perf_counter() - t0
                    print(f"  [{tn}] {wt}: {kind} {dt:.1f}s | m={tm} h={th} | {el:.0f}s")
    
    total_time = time.perf_counter() - t0
    
    # ---- Save each module (deduplicated + quantized) ----
    print(f"\n{'='*60}")
    print(f"Single-pass complete: {total_time:.0f}s")
    print(f"{'='*60}")
    
    for m in list(holo_U.keys()):
        if not holo_U[m]: continue
        
        out = OUTPUT_DIR / f"deepseek_v4_flash_{m}_k{RANK_K}.holo"
        
        # Build output dict
        d = {
            '_svh': svh_shared[m],
            '_svh_scales': svh_scales[m],
            '_format': 'int8_dedup',
            '_k': RANK_K,
        }
        d.update(holo_U[m])
        d.update(holo_scales[m])
        
        # Also include SVh reference mapping: key -> wt for reconstruction
        svh_ref = {}
        for key in holo_U[m]:
            wt = extract_weight_type(key.replace('.U', ''), m)
            svh_ref[key] = wt
        d['_svh_ref'] = svh_ref
        
        tmp = str(out) + ".tmp"
        torch.save(d, tmp)
        os.replace(tmp, str(out))
        
        s = stats[m]
        gb = os.path.getsize(out) / 1024**3
        print(f"  {m:<15}: {len(holo_U[m]):>6} U tensors, {len(svh_shared[m])} shared SVh, "
              f"{gb:.2f} GB | m={s['misses']} h={s['hits']}")
    
    # Config
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
