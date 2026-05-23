"""Extract original attention weights from safetensors (bypass holo for inference)."""
import torch, os, json, struct, numpy as np
from pathlib import Path

MODEL_DIR = r"E:\Reneshizzle SG\Models\deepseek-ai\DeepSeek-V4-Flash"
OUT = Path(r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\HOLO\_models\ds_attn_original.pt")

with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    idx = json.load(f)

attn = {}
shard_cache = {}
total_bytes = 0

for k in sorted(idx["weight_map"]):
    if "attn." not in k or not k.endswith(".weight"): continue
    
    shard_name = idx["weight_map"][k]
    if shard_name not in shard_cache:
        path = os.path.join(MODEL_DIR, shard_name)
        with open(path, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            shard_cache[shard_name] = (hl, json.loads(f.read(hl)), path)
    
    hl, hdr, path = shard_cache[shard_name]
    if k not in hdr: continue
    
    info = hdr[k]
    s, e = info['data_offsets']
    shape = info['shape']
    dtype = info['dtype']
    
    with open(path, 'rb') as f:
        f.seek(8 + hl + s)
        data = f.read(e - s)
    
    # Handle different dtypes
    if 'I8' in dtype:
        arr = np.frombuffer(data, dtype=np.int8).astype(np.float32).copy()
    elif 'F16' in dtype:
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
    elif 'F32' in dtype:
        arr = np.frombuffer(data, dtype=np.float32).copy()
    else:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32).copy()
    
    t = torch.from_numpy(arr)
    try:
        t = t.reshape(shape)
    except:
        # Shape mismatch: try to infer from data size
        total = t.numel()
        if len(shape) == 2:
            r, c = shape
            if r * c != total:
                # Try alternative: data might be FP16 stored at half the reported size
                if total * 2 == r * c:
                    arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
                    t = torch.from_numpy(arr).reshape(shape)
                elif total == r * c // 2:
                    t = t.reshape(-1, c) if c <= total else t.reshape(r, -1)
        if t.numel() > 0:
            pass  # use whatever shape we got
    
    attn[k] = t
    total_bytes += e - s

torch.save(attn, str(OUT))
gb = os.path.getsize(OUT) / 1024**3
print(f"Saved: {OUT.name} ({gb:.2f} GB, {len(attn)} weights)")

# Group by layer
layers = set()
for k in attn:
    parts = k.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            try: layers.add(int(parts[i+1]))
            except: pass
print(f"Layers: {sorted(layers)[:3]}...{sorted(layers)[-3:]} ({len(layers)} total)")
