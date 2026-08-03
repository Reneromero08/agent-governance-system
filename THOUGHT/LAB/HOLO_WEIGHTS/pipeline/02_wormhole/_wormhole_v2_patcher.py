"""V2 Wormhole Patcher — reconstructs full .holo from rotations + SVh.

Format:
  {wt}.weight.SVh — shared SVh per weight type
  {wt}.weight.L0.U — first layer's U
  {wt}.weight.L{n}.R — rotation R = U_prev^T @ U_curr (k×k)
  {wt}.weight.L{n}.res_idx — 2-bit residual indices
  {wt}.weight.L{n}.res_max — residual max
  
Reconstructs: U_curr = U_prev @ R + dequantize(residual)
Creates HoloLinear from U + SVh, patches 27B model.
"""
import torch, math, os, sys, time, re
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM

WH_PATH = str(Path(__file__).resolve().parent / "qwen_27b_wormhole_v2.holo")
MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"

class HoloLinear(nn.Module):
    def __init__(self, U, SVh, bias=None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
    def forward(self, x):
        out = x @ self.SVh.t() @ self.U.t()
        if self.bias is not None: out += self.bias
        return out

print("Loading V2 wormhole...", flush=True)
t0 = time.perf_counter()
wh = torch.load(WH_PATH, weights_only=False, map_location='cpu')
print(f"  Loaded {len(wh)} keys in {time.perf_counter()-t0:.1f}s", flush=True)

# Parse: group by weight type
groups = defaultdict(dict)  # {wt: {layer: (U or R+res)}}
svh_dict = {}  # {wt: SVh}

for key, val in wh.items():
    if key.endswith('.SVh'):
        wt = key[:-4]  # remove .SVh
        svh_dict[wt] = val.float()
    elif '.L' in key:
        # Parse {wt}.weight.L{layer}.{suffix}
        parts = key.split('.L')
        wt_base = parts[0]  # e.g., mlp.down_proj.weight
        rest = parts[1]     # e.g., 0.U or 1.R
        layer_str, suffix = rest.split('.', 1) if '.' in rest else (rest, '')
        layer = int(layer_str)
        
        if suffix == 'U':
            groups[wt_base][layer] = ('U', val.float())
        elif suffix == 'R':
            # Find matching res_idx and res_max
            base_key = f"{wt_base}.L{layer}"
            res_idx = wh.get(f"{base_key}.res_idx")
            res_max = wh.get(f"{base_key}.res_max")
            groups[wt_base][layer] = ('R', val.float(), res_idx, res_max)

print(f"  Groups: {len(groups)} weight types, {len(svh_dict)} SVh entries", flush=True)

# Reconstruct U for each layer
reconstructed = {}

for wt_base, layers in sorted(groups.items()):
    svh_key = wt_base
    svh = svh_dict.get(svh_key)
    if svh is None:
        print(f"  WARNING: no SVh for {wt_base}, skipping")
        continue

    sorted_l = sorted(layers.keys())
    prev_U = None
    layer_us = {}  # l -> U tensor

    for l in sorted_l:
        data = layers[l]
        if data[0] == 'U':
            prev_U = data[1]
            layer_us[l] = prev_U
        else:
            R = data[1]; res_idx = data[2]; res_max = data[3]
            U_rot = prev_U @ R
            if res_idx is not None and res_max is not None:
                rm = max(res_max.item(), 1e-6)
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0], device=U_rot.device) * rm
                residual_quant = levels[res_idx.long()]
                prev_U = U_rot + residual_quant
            else:
                prev_U = U_rot
            layer_us[l] = prev_U

    for l in sorted_l:
        wt_clean = wt_base.replace('.weight', '')
        hf_key = f"model.layers.{l}.{wt_clean}.weight"
        reconstructed[f"{hf_key}.U"] = layer_us[l].half()
        reconstructed[f"{hf_key}.SVh"] = svh.half()

print(f"  Reconstructed: {len(reconstructed)} entries", flush=True)

# Patch 27B model
print("\nInitializing 27B model...", flush=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Materialize all meta params (embeddings, norms, etc.)
print("  Materializing meta parameters...", flush=True)
meta_count = 0
for name, param in model.named_parameters():
    if param.device.type == "meta":
        dt = param.dtype if param.dtype != torch.bfloat16 else torch.bfloat16
        try:
            data = torch.zeros_like(param, device=device)
            if 'embed' in name or 'lm_head' in name:
                data.normal_(std=0.02)
            param.data = data
        except RuntimeError:
            data = torch.zeros(param.shape, device=device, dtype=torch.float32)
            if 'embed' in name or 'lm_head' in name:
                data.normal_(std=0.02)
            param.data = data
        meta_count += 1
print(f"  Materialized {meta_count} meta parameters", flush=True)

# Patch with HoloLinear
patched = 0; skipped_linear = 0; found_u = 0; found_svh = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        skipped_linear += 1
        weight_key = name + ".weight"
        u_key = f"{weight_key}.U"
        svh_key = f"{weight_key}.SVh"
        
        has_u = u_key in reconstructed
        has_svh = svh_key in reconstructed
        if has_u: found_u += 1
        if has_svh: found_svh += 1
        
        if has_u and has_svh:
            U = reconstructed[u_key].to(device, dtype=torch.bfloat16)
            SVh = reconstructed[svh_key].to(device, dtype=torch.bfloat16)
            bias = module.bias.data if module.bias is not None else None
            if bias is not None: bias = bias.to(device, dtype=torch.bfloat16)
            
            holo = HoloLinear(U, SVh, bias)
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                setattr(parent, parts[1], holo)
                patched += 1
        elif patched < 3 and skipped_linear < 10:
            print(f"  DEBUG: {name} -> u={has_u} svh={has_svh}")

print(f"  Patched: {patched} HoloLinear layers", flush=True)
model.eval()

# Count
meta_params = sum(1 for p in model.parameters() if p.device.type == 'meta')
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")
print(f"  Still on meta: {meta_params}")
print(f"\n  V2 WORMHOLE LOADED: 1.13 GB -> 27B model with {patched} HoloLinear layers")

# Stats after patching
meta_after = sum(1 for p in model.parameters() if p.device.type == 'meta')
holo_params = sum(p.numel() for n, p in model.named_parameters() if 'U' in n or 'SVh' in n)
print(f"  Still on meta: {meta_after} | Holo params: {holo_params:,}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
ids = tokenizer("The catalytic computing paradigm demonstrates that", return_tensors="pt")['input_ids']
print(f"\n  Inference test...", flush=True)
with torch.no_grad():
    try:
        out = model(ids.to(device), output_hidden_states=True)
        top5 = torch.topk(out.logits[:, -1, :].float(), 5).indices[0]
        words = [tokenizer.decode([t]) for t in top5]
        print(f"  Top-5: {words}")
        print(f"  Hidden norm: {out.hidden_states[-1].norm():.2f}")
    except Exception as e:
        print(f"  Crash: {e}")
        import traceback; traceback.print_exc()
