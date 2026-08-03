"""Pre-extract shared expert FFN weights (w1,w2,w3) for all 43 layers.
Replaces 257 MB shard loads with 50 MB per layer. 5x speedup."""
import torch, os, time, numpy as np
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
SHARDS = HOLO / "experts_shards"
OUT = HOLO / "ds_shared_ffn.holo"

print("Loading SVh...")
svh_data = torch.load(str(SHARDS / "svh_shared.holo"), weights_only=False, map_location="cpu")
svh = {wt: (info["data"].float() * info["scale"]) for wt, info in svh_data.items()}

ffn = {}
t0 = time.perf_counter()

for layer in range(43):
    path = SHARDS / f"experts_layer_{layer:02d}.holo"
    d = torch.load(str(path), weights_only=False, map_location="cpu")
    
    for wt_suffix in ['w1', 'w2', 'w3']:
        ukey = f"layers.{layer}.ffn.shared_experts.{wt_suffix}.weight.U"
        skey = ukey.replace('.U', '.scale')
        svh_wt = f"layers.ffn.shared_experts.{wt_suffix}.weight"
        if ukey in d and svh_wt in svh:
            U = d[ukey].float() * d.get(skey, 1.0)
            W = U @ svh[svh_wt]
            ffn[f"layers.{layer}.ffn.shared_experts.{wt_suffix}.weight"] = W.half()
    
    del d

torch.save(ffn, str(OUT))
mb = os.path.getsize(OUT) / 1024**2
print(f"Saved: {OUT.name} ({mb:.0f} MB, {len(ffn)} weights, {time.perf_counter()-t0:.0f}s)")

# Per-layer estimate
per_layer_mb = mb / 43
print(f"Per layer: {per_layer_mb:.0f} MB (was 257 MB — {257/per_layer_mb:.0f}x reduction)")
