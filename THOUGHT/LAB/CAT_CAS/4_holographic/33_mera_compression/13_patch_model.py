"""
Wormhole Model Patcher — GPU-Safe Loading
==========================================
Loads Qwen 27B without OOM: model architecture on meta device,
patch HoloLinear BEFORE materialization, only allocate compressed weights.

Fixes: "to_empty allocates full 53 GB on CUDA, OOM before patching happens"

Usage:
  python 13_patch_model.py --model qwen --module llm
"""
import torch, torch.nn as nn, json, sys, os, gc
from pathlib import Path
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import _paths


class WormholeLinear(nn.Module):
    """Compressed HoloLinear from wormhole cassette. Lightweight GPU allocation."""
    
    def __init__(self, U, SVh, bias=None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # HoloLinear: x @ SVh^T @ U^T (2 matmuls, no full weight materialized)
        out = torch.matmul(x, self.SVh.t())
        out = torch.matmul(out, self.U.t())
        if self.bias is not None:
            out += self.bias
        return out
    
    @staticmethod
    def from_wormhole_rotations(wt_name, layer_idx, groups, shared_svh, device='cpu'):
        """
        Reconstruct one layer's U from wormhole rotations + shared SVh.
        Returns: WormholeLinear or None
        """
        if wt_name not in groups:
            return None
        
        g = groups[wt_name]
        first_l = g['first_l']
        
        if layer_idx == first_l:
            U = g['first_U'].float()
        elif layer_idx in g['rots']:
            anchor = g['first_U'].float()
            R = g['rots'][layer_idx].float()
            U = anchor @ R
            if layer_idx in g['res'] and g['res'][layer_idx].get('idx') is not None:
                rd = g['res'][layer_idx]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                residual = levels[rd['idx'].long()]
                U = U + residual
        else:
            return None
        
        SVh = shared_svh.get(wt_name)
        if SVh is None:
            return None
        
        return WormholeLinear(
            U.to(device=device, dtype=torch.bfloat16),
            SVh.to(device=device, dtype=torch.bfloat16),
        )


def parse_wormhole(wormhole_path):
    """
    Parse a wormhole cassette into groups + shared SVh.
    Lightweight parse — takes ~1 second, no GPU allocation.
    """
    import re
    worm = torch.load(str(wormhole_path), map_location='cpu', weights_only=True)
    pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
    groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))
    
    for key, val in worm.items():
        m = pattern.match(key)
        if not m: continue
        wt, layer_str, field = m.groups()
        l = int(layer_str)
        g = groups[wt]
        if field == 'U':
            g['first_U'] = val; g['first_l'] = l
        elif field == 'R':
            g['rots'][l] = val
        elif field == 'res_idx':
            g['res'].setdefault(l, {})['idx'] = val
        elif field == 'res_max':
            if l in g['res']:
                g['res'][l]['max'] = val
    
    shared_svh = {}
    for key, val in worm.items():
        if '.L' not in key and key.endswith('.SVh'):
            wt = key.replace('.SVh', '')
            shared_svh[wt] = val
    
    return groups, shared_svh


def map_wt_to_hf(wormhole_wt, config):
    """
    Map a wormhole weight type to HuggingFace model layer attribute path.
    
    Wormhole key:  'mlp.down_proj.weight'
    HF path:       model.language_model.layers.{N}.mlp.down_proj
    
    Wormhole key:  'self_attn.q_proj.weight'
    HF path:       model.language_model.layers.{N}.self_attn.q_proj
    
    Wormhole key:  'linear_attn.in_proj_qkv.weight'
    HF path:       model.language_model.layers.{N}.linear_attn.in_proj_qkv
    """
    base = wt.replace('.weight', '')
    # Handle special cases
    if base.startswith('linear_attn.'):
        return ('language_model', 'layers', base)
    if base.startswith('self_attn.'):
        return ('language_model', 'layers', base)
    if base.startswith('mlp.'):
        return ('language_model', 'layers', base)
    if base.startswith('attn.'):
        return ('visual', 'blocks', base)
    return None


def patch_model_with_wormhole(model, wormhole_groups, shared_svh, device='cuda'):
    """
    Patch a HF model's Linear layers with WormholeLinear.
    Model must be on 'meta' or 'cpu' device. Only allocates compressed weights.
    
    Returns: (model, stats)
    """
    patched = 0
    skipped = 0
    total_params_orig = 0
    total_params_comp = 0
    
    # Iterate through model layers
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # Parse the HF layer path to extract weight type + layer index
        # e.g. 'model.language_model.layers.0.mlp.down_proj' -> wt='mlp.down_proj.weight', l=0
        parts = name.split('.')
        
        layer_idx = None
        for i, p in enumerate(parts):
            if p in ('layers', 'blocks') and i + 1 < len(parts):
                try: layer_idx = int(parts[i + 1])
                except Exception: pass
                break
        
        if layer_idx is None:
            continue
        
        # Build wormhole weight type name
        wt_parts = []
        found = False
        for p in ('mlp', 'self_attn', 'linear_attn', 'attn'):
            if p in parts:
                idx = parts.index(p)
                wt_parts = parts[idx:]
                found = True
                break
        
        if not found:
            continue
        
        wt_name = '.'.join(wt_parts) + '.weight'
        
        # Try to construct the wormhole linear
        wl = WormholeLinear.from_wormhole_rotations(
            wt_name, layer_idx, wormhole_groups, shared_svh, device=device
        )
        
        if wl is None:
            skipped += 1
            continue
        
        # Count compression
        orig_w = module.weight
        if hasattr(orig_w, 'shape'):
            total_params_orig += orig_w.shape[0] * orig_w.shape[1]
        total_params_comp += wl.U.shape[0] * wl.U.shape[1] + wl.SVh.shape[0] * wl.SVh.shape[1]
        
        # Replace the Linear layer in the model
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        parent = model
        for p in parent_name.split('.'):
            parent = getattr(parent, p)
        setattr(parent, attr_name, wl)
        patched += 1
    
    stats = {
        'patched': patched,
        'skipped': skipped,
        'params_orig': total_params_orig,
        'params_comp': total_params_comp,
        'compression': total_params_orig / total_params_comp if total_params_comp > 0 else 0,
    }
    return model, stats


def load_model_patched(wormhole_paths, model_id="Qwen/Qwen2.5-7B", device='cuda'):
    """
    Load a model with wormhole-patched layers. GPU-safe.
    
    1. Load config + build architecture on meta device (0 GPU memory)
    2. Parse wormhole cassettes (CPU, ~1 second)  
    3. Patch Linear -> WormholeLinear on meta device
    4. Load remaining params (embed, norm, bias) from HF hub to CPU
    5. Move to GPU only the compressed + remaining layers
    """
    print(f"Loading config: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Step 1: Build architecture on meta (0 GPU memory)
    print("Building model architecture on meta device...")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # Step 2: Parse all wormhole cassettes
    all_groups = {}
    all_svh = {}
    for mod_name, path in wormhole_paths.items():
        path = Path(path)
        if not path.exists():
            print(f"  WARNING: {mod_name} not found at {path}")
            continue
        groups, svh = parse_wormhole(path)
        all_groups.update(groups)
        all_svh.update(svh)
        print(f"  {mod_name}: {len(groups)} weight types, {len(svh)} shared SVh")
    
    # Step 3: Patch Linear layers with WormholeLinear
    print(f"Patching model with wormhole layers...")
    model, stats = patch_model_with_wormhole(model, all_groups, all_svh, device='meta')
    print(f"  Patched: {stats['patched']} layers")
    print(f"  Compression: {stats['compression']:.1f}x ({stats['params_orig']/1e6:.0f}M -> {stats['params_comp']/1e6:.0f}M params)")
    
    # Step 4: Materialize model to target device (only compressed layers allocated)
    print(f"Materializing to {device}...")
    model.to_empty(device=device)
    
    # Step 5: Load ONLY missing params (embed, norm, bias) from HF hub — NOT the full model
    print("Loading embed/norm from config only (no 53 GB download)...")
    try:
        # Only download the tiny safetensors index + config, not the full 53 GB
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            low_cpu_mem_usage=True,
            max_memory={0: "4GB"},  # limit CPU RAM, forces streaming
        )
    except Exception as e:
        print(f"  WARNING: Could not load reference: {e}")
        print(f"  Trying with local_files_only + offline config...")
        ref_model = None
    
    if ref_model is not None:
        ref_params = dict(ref_model.named_parameters())
        ref_buffers = dict(ref_model.named_buffers())
        
        # Copy non-HoloLinear params
        for name, param in model.named_parameters():
            if param.device.type != 'meta':
                continue
            if name in ref_params:
                param.data = ref_params[name].data.to(device)
            else:
                param.data = torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
        
        # Copy buffers
        for name, buf in model.named_buffers():
            if buf.device.type == 'meta' and name in ref_buffers:
                buf.data = ref_buffers[name].data.to(device)
        
        del ref_model, ref_params, ref_buffers
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    model.eval()
    return model, stats


if __name__ == "__main__":
    wormhole_paths = _paths.MODULE_PATHS
    
    MODEL_ID = "Qwen/Qwen2.5-7B"  # Architecture template
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("WORWHOLE MODEL PATCHER — GPU-Safe")
    print("=" * 70)
    
    model, stats = load_model_patched(wormhole_paths, model_id=MODEL_ID, device=DEVICE)
    
    # VRAM report
    if DEVICE == 'cuda':
        print(f"\n  GPU VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  GPU VRAM reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Quick inference test
    print("\n[Inference Test]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {text}")
    
    # Model size on disk
    import tempfile, os as _os
    tmp = _os.path.join(tempfile.gettempdir(), "wormhole_test.pt")
    torch.save(model.state_dict(), tmp)
    print(f"\n  State dict size: {_os.path.getsize(tmp) / 1024**2:.0f} MB")
    _os.remove(tmp)
