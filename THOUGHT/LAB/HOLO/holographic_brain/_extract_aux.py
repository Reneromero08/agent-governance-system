"""Extract norm weights, embed, and lm_head from safetensors to local .holo."""
import torch, os, json, struct, numpy as np

MODEL_DIR = r"E:\Reneshizzle SG\Models\deepseek-ai\DeepSeek-V4-Flash"
OUT = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\HOLO\_models\ds_aux_weights.holo"

with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
    idx = json.load(f)

def load(key):
    shard = idx["weight_map"][key]
    path = os.path.join(MODEL_DIR, shard)
    with open(path, 'rb') as f:
        hl = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hl))
        s, e = hdr[key]['data_offsets']
        f.seek(8 + hl + s)
        data = f.read(e - s)
    dtype = hdr[key]['dtype']
    if 'F32' in dtype:
        arr = np.frombuffer(data, dtype=np.float32).copy()
    else:
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
    return torch.from_numpy(arr).reshape(hdr[key]['shape'])

out = {}

# Embed and lm_head
out['embed.weight'] = load('embed.weight')
out['head.weight'] = load('head.weight')
print(f"Embed: {list(out['embed.weight'].shape)}")
print(f"Head:  {list(out['head.weight'].shape)}")

# Norm weights for all layers
for layer in range(43):
    for nt in ['attn_norm', 'ffn_norm', 'attn.q_norm', 'attn.kv_norm']:
        key = f"layers.{layer}.{nt}.weight"
        if key in idx["weight_map"]:
            out[key] = load(key)

# Output norm
if 'norm.weight' in idx["weight_map"]:
    out['norm.weight'] = load('norm.weight')

torch.save(out, OUT)
mb = os.path.getsize(OUT) / 1024**2
print(f"\nSaved: {OUT} ({mb:.0f} MB, {len(out)} tensors)")
