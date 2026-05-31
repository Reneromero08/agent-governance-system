"""
Catalytic Offloading Inference — DeepSeek V4 Flash
===================================================
Exp 16 pattern: borrow layer weights to GPU, compute, return to CPU.
Loads per-layer shards from experts_shards/ + attention v2 holo.
One layer at a time in GPU. O(1 layer) VRAM.
"""
import torch, os, time, json
from pathlib import Path
from collections import defaultdict

REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
EXPERT_SHARDS = HOLO / "experts_shards"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 128

def load_shard(layer_idx):
    """Load one layer's U tensors + reconstruct weights via shared SVh."""
    path = EXPERT_SHARDS / f"experts_layer_{layer_idx:02d}.holo"
    d = torch.load(str(path), weights_only=False, map_location="cpu")
    
    weights = {}
    for key in d:
        if not key.endswith(".U") or key.startswith("_"):
            continue
        scale_key = key.replace(".U", ".scale")
        U = d[key].float() * d.get(scale_key, 1.0)
        
        wt = d["_svh_ref"].get(key, "").replace(".weight.weight", ".weight")
        if wt not in svh_cache:
            continue
        
        SVh = svh_cache[wt]  # on GPU
        weights[key.replace(".U", "")] = (U.to(DEVICE) @ SVh).to(DEVICE)
    
    return weights

def load_attention():
    """Load attention v2 holo, build per-layer weight dict."""
    path = HOLO / "deepseek_v4_flash_attention_k128.holo"
    d = torch.load(str(path), weights_only=False, map_location="cpu")
    
    # Dequant shared SVh
    svh_deq = {}
    for wt in d["_svh"]:
        svh_deq[wt] = d["_svh"][wt].float() * d["_svh_scales"][wt]
    
    # Build weights per layer
    layers = defaultdict(dict)
    for key in d:
        if not key.endswith(".U") or key.startswith("_"):
            continue
        parts = key.split(".")
        layer = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try: layer = int(parts[i+1])
                except Exception: pass; break
        
        if layer is None: continue
        
        U = d[key].float() * d.get(key.replace(".U", ".scale"), 1.0)
        wt = d["_svh_ref"].get(key, "").replace(".weight.weight", ".weight")
        if wt not in svh_deq: continue
        
        SVh = svh_deq[wt]
        layers[layer][key.replace(".U", "")] = U @ SVh
    
    return dict(layers)

# ==== INIT ====
print(f"Device: {DEVICE}")
t0 = time.perf_counter()

# Load shared SVh (tiny, ~4 MB, stays on GPU)
print("Loading shared SVh...")
svh_path = EXPERT_SHARDS / "svh_shared.holo"
svh_data = torch.load(str(svh_path), weights_only=False)
svh_cache = {}
for wt, info in svh_data.items():
    svh_cache[wt] = (info["data"].float() * info["scale"]).to(DEVICE)
print(f"  {len(svh_cache)} SVh tensors on {DEVICE}")

# Load attention weights (all layers, stay on CPU, move to GPU per layer)
print("Loading attention...")
attn_layers = load_attention()
num_layers = len(attn_layers)
print(f"  {num_layers} attention layers (on CPU)")

# ==== CATALYTIC LOOP ====
print(f"\nCatalytic offloading over {num_layers} layers...")
print(f"{'Layer':>6} {'AttnW':>7} {'ExpW':>7} {'GPU_GB':>8} {'Time':>8} {'Tape':>8}")
print(f"{'-'*50}")

total_weights = 0
peak_gpu = 0
batch, seq = 1, 4
hidden = 4096

# Dummy input
x = torch.randn(batch, seq, hidden, device=DEVICE)

for layer in range(min(5, num_layers)):  # POC: first 5 layers
    ts = time.perf_counter()
    torch.cuda.empty_cache()
    gpu_before = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
    
    # BORROW: load attention weights to GPU
    attn_w = {}
    for k, w in attn_layers[layer].items():
        attn_w[k] = w.to(DEVICE)
    
    # BORROW: load expert weights to GPU
    exp_w = load_shard(layer) if (EXPERT_SHARDS / f"experts_layer_{layer:02d}.holo").exists() else {}
    
    gpu_after = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
    peak_gpu = max(peak_gpu, gpu_after)
    
    # COMPUTE: simple forward pass (just wq_a projection as POC)
    if "layers.0.attn.wq_a.weight" in attn_w:
        wq = attn_w[f"layers.{layer}.attn.wq_a.weight"]
        x = x.float() @ wq.T  # [B, S, 4096] @ [4096, 1024] = [B, S, 1024]
        # Project back to hidden for next layer
        if f"layers.{layer}.attn.wq_b.weight" in attn_w:
            wqb = attn_w[f"layers.{layer}.attn.wq_b.weight"]
            x = x @ wqb.T  # [B, S, 1024] @ [1024, 4096] = [B, S, 4096]
    
    n_attn = sum(w.numel() for w in attn_w.values())
    n_exp = sum(w.numel() for w in exp_w.values())
    total_weights += n_attn + n_exp
    
    dt = time.perf_counter() - ts
    
    # RETURN: free GPU
    del attn_w, exp_w
    torch.cuda.empty_cache()
    gpu_after_free = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
    
    tape_status = "CLEAN" if abs(gpu_after_free - gpu_before) < 0.01 else "LEAK"
    
    print(f"{layer:>6} {n_attn/1e6:>6.0f}M {n_exp/1e6:>6.0f}M "
          f"{gpu_after:>8.2f} {dt:>7.2f}s {tape_status:>8}")

print(f"\n  Total weights processed: {total_weights/1e9:.2f}B params")
print(f"  Peak GPU: {peak_gpu:.2f} GB")
print(f"  Time: {time.perf_counter()-t0:.1f}s")
print(f"  Output shape: {list(x.shape)}")
print(f"\n  Catalytic tape: GPU memory returned after each layer")
print(f"  Ready for full forward pass wiring.")
