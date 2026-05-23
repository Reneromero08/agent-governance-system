"""
Shard the 11 GB experts monolith into per-layer files (~260 MB each).
Layer 0 = all 768 U tensors for layer 0, plus shared SVh once.
Inference loads only current layer, never the full 11 GB.
"""
import torch, os, time
from pathlib import Path
from collections import defaultdict

INPUT = Path(r"THOUGHT\LAB\HOLO\_models\deepseek_v4_flash_experts_k128.holo")
OUTDIR = INPUT.parent / "experts_shards"
OUTDIR.mkdir(exist_ok=True)

print(f"Loading {os.path.getsize(INPUT)/1024**3:.1f} GB...")
t0 = time.perf_counter()
d = torch.load(str(INPUT), weights_only=False, map_location="cpu")
print(f"  Loaded in {time.perf_counter()-t0:.0f}s")

# Extract shared SVh (tiny, save once)
svh_shared = {}
for wt, svh_i8 in d["_svh"].items():
    svh_shared[wt] = {
        "data": svh_i8,
        "scale": d["_svh_scales"][wt],
    }
torch.save(svh_shared, str(OUTDIR / "svh_shared.holo"))
print(f"  Shared SVh: {len(svh_shared)} types saved")

# Group U tensors by layer
layers = defaultdict(dict)
scales = defaultdict(dict)
svh_ref = defaultdict(dict)

for key in d:
    if not key.endswith(".U"): continue
    
    # Parse layer from key
    parts = key.split(".")
    layer = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try: layer = int(parts[i+1])
            except: pass
            break
    
    if layer is None: continue
    
    layers[layer][key] = d[key]
    scale_key = key.replace(".U", ".scale")
    if scale_key in d:
        scales[layer][scale_key] = d[scale_key]
    
    # Per-layer svh_ref
    wt = d["_svh_ref"].get(key, "")
    wt = wt.replace(".weight.weight", ".weight")  # fix double weight bug
    if wt:
        svh_ref[layer][key] = wt

# Save each layer
total_mb = 0
for layer in sorted(layers.keys()):
    out = {
        "_U": layers[layer],
        "_scales": scales[layer],
        "_svh_ref": svh_ref[layer],
        "_layer": layer,
        "_format": "shard_v1",
        "_k": d.get("_k", 128),
    }
    out.update(layers[layer])
    for k, v in scales[layer].items():
        out[k] = v
    
    path = OUTDIR / f"experts_layer_{layer:02d}.holo"
    torch.save(out, str(path))
    mb = os.path.getsize(path) / 1024**2
    total_mb += mb
    print(f"  Layer {layer:02d}: {len(layers[layer])} U tensors, {mb:.0f} MB")

# Save metadata
meta = {
    "num_layers": len(layers),
    "num_experts_per_layer": len(layers[0]) // 3,
    "weight_types": ["w1", "w2", "w3"],
    "shared_svh_file": "svh_shared.holo",
    "total_layers": len(layers),
}
torch.save(meta, str(OUTDIR / "manifest.holo"))

total_gb = total_mb / 1024
print(f"\n  Sharded: {len(layers)} layers, {total_gb:.1f} GB total")
print(f"  Saved to: {OUTDIR}")
print(f"  Inference: load svh_shared.holo once + one layer at a time")
