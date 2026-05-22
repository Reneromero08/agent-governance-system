"""
MTP Wormhole Inference — Qwen 3.6 27B with K=128 compression
=============================================================
Loads Qwen 3.5 config, builds model, replaces Linear with WormholeLinear
using MTP K=128 wormhole. Loads only non-linear params from safetensors.

Fix: HoloLinear forward uses x @ U (input→K) then h @ SVh (K→output).
The existing convention stored U(in,K) and SVh(K,out) but computed in wrong order.
"""
import torch, torch.nn as nn, json, sys, os, gc, re, time
from pathlib import Path
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_DIR = r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
WORMHOLE_PATH = Path(r'THOUGHT/LAB/HOLO/_models/qwen_27b_mtp_wormhole_k128.holo')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_NEW_TOKENS = 50

# GGUF weight type -> HF module attribute mapping
GGUF_TO_HF = {
    'ffn_gate.weight':     ('mlp', 'gate_proj'),
    'ffn_up.weight':       ('mlp', 'up_proj'),
    'ffn_down.weight':     ('mlp', 'down_proj'),
    'attn_qkv.weight':     ('linear_attn', 'in_proj_qkv'),
    'attn_gate.weight':    ('linear_attn', 'in_proj_z'),
    'ssm_alpha.weight':    ('linear_attn', 'in_proj_a'),
    'ssm_beta.weight':     ('linear_attn', 'in_proj_b'),
    'ssm_out.weight':      ('linear_attn', 'out_proj'),
    'attn_q.weight':       ('self_attn', 'q_proj'),
    'attn_k.weight':       ('self_attn', 'k_proj'),
    'attn_v.weight':       ('self_attn', 'v_proj'),
    'attn_output.weight':  ('self_attn', 'o_proj'),
    'nextn.eh_proj.weight': ('nextn', 'eh_proj'),
}

# ============================================================================
# Wormhole Parser
# ============================================================================
def parse_wormhole(path):
    """Parse wormhole cassette into groups + shared SVh. Lightweight."""
    worm = torch.load(str(path), map_location='cpu', weights_only=True)
    pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
    groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))
    for key, val in worm.items():
        m = pattern.match(key)
        if not m: continue
        wt, layer_str, field = m.groups()
        l = int(layer_str)
        g = groups[wt]
        if field == 'U': g['first_U'] = val; g['first_l'] = l
        elif field == 'R': g['rots'][l] = val
        elif field == 'res_idx': g['res'].setdefault(l, {})['idx'] = val
        elif field == 'res_max':
            if l in g['res']: g['res'][l]['max'] = val
    shared_svh = {}
    for key, val in worm.items():
        if '.L' not in key and key.endswith('.SVh'):
            shared_svh[key.replace('.SVh', '')] = val
    return groups, shared_svh

# ============================================================================
# Fixed WormholeLinear — correct forward pass
# ============================================================================
class MTPWormholeLinear(nn.Module):
    """
    Fixed HoloLinear: U maps from input to K, SVh maps from K to output.
    Forward: x @ U (input→K) then h @ SVh (K→output).
    The original HoloLinear computed the transpose, which was wrong for real models.
    """
    def __init__(self, U, SVh, bias=None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # U: (in_dim, K)    x: (B, S, in_dim)    -> h: (B, S, K)
        h = torch.matmul(x, self.U)
        # SVh: (K, out_dim)                       -> out: (B, S, out_dim)
        out = torch.matmul(h, self.SVh)
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def from_wormhole(wt_name, layer_idx, groups, shared_svh, device='cpu'):
        """Reconstruct one layer's compressed weights from wormhole rotations."""
        if wt_name not in groups: return None
        g = groups[wt_name]
        if layer_idx == g['first_l']:
            U_anchor = g['first_U'].float()
        elif layer_idx in g['rots']:
            U_anchor = g['first_U'].float() @ g['rots'][layer_idx].float()
            if layer_idx in g['res'] and g['res'][layer_idx].get('idx') is not None:
                rd = g['res'][layer_idx]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                U_anchor = U_anchor + levels[rd['idx'].long()]
        else:
            return None
        SVh = shared_svh.get(wt_name)
        if SVh is None: return None
        return MTPWormholeLinear(
            U_anchor.to(device=device, dtype=torch.bfloat16),
            SVh.to(device=device, dtype=torch.bfloat16),
        )

# ============================================================================
# HF Path Mapping
# ============================================================================
def find_wt_name(hf_path, layer_idx):
    """Map HF module path to GGUF wormhole weight type name."""
    # hf_path: model.language_model.layers.N.mlp.gate_proj
    # or: model.layers.N.mlp.gate_proj (direct)
    parts = hf_path.split('.')
    # Find the module type and attr after the layer index
    kit = None
    for i, p in enumerate(parts):
        if p in ('mlp', 'self_attn', 'linear_attn', 'nextn', 'attn'):
            if i + 1 < len(parts):
                kit = p, parts[i + 1]
            break
    if kit is None: return None
    # Reverse lookup: (module_type, attr) -> GGUF weight type
    reverse = {v: k for k, v in GGUF_TO_HF.items()}
    return reverse.get(kit)

# ============================================================================
# Main: Patch and Infer
# ============================================================================
def main():
    print("=" * 60)
    print("MTP WORMHOLE INFERENCE — Qwen 3.6 27B K=128")
    print("=" * 60)

    # 1. Parse wormhole
    print(f"\n[1] Parsing wormhole: {WORMHOLE_PATH}")
    groups, shared_svh = parse_wormhole(WORMHOLE_PATH)
    n_weights = sum(1 + len(g['rots']) for g in groups.values())
    print(f"    {len(groups)} weight types, {n_weights} layers, {len(shared_svh)} SVh entries")

    # 2. Load config and build model on meta
    print(f"\n[2] Loading config from {MODEL_DIR}")
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    tc = config.text_config
    print(f"    Layers: {getattr(tc, 'num_hidden_layers', '?')}")
    print(f"    Hidden:  {getattr(tc, 'hidden_size', '?')}")
    print(f"    Layer types: {getattr(tc, 'layer_types', [])[:8]}...")

    print("    Building model on meta device...")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # 3. Patch Linear layers
    print(f"\n[3] Patching Linear -> MTPWormholeLinear")
    patched = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Extract layer index
        parts = name.split('.')
        layer_idx = None
        for i, p in enumerate(parts):
            if p in ('layers', 'blocks') and i + 1 < len(parts):
                try: layer_idx = int(parts[i + 1])
                except: pass
                break
        if layer_idx is None: continue

        # Map to wormhole weight type
        wt_name = find_wt_name(name, layer_idx)
        if wt_name is None:
            skipped += 1
            continue

        wl = MTPWormholeLinear.from_wormhole(wt_name, layer_idx, groups, shared_svh, device='meta')
        if wl is None:
            skipped += 1
            continue

        # Replace the Linear layer
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        parent = model
        for p in parent_name.split('.'):
            parent = getattr(parent, p)
        setattr(parent, attr_name, wl)
        patched += 1

    print(f"    Patched: {patched}  Skipped: {skipped}")

    # 4. Materialize to GPU
    print(f"\n[4] Materializing to {DEVICE}...")
    model.to_empty(device=DEVICE)
    model.eval()
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"    GPU allocated: {alloc:.2f} GB")

    # 5. Load missing params (norms, embeddings, SSM conv, etc.) from safetensors
    print(f"\n[5] Loading norms/embeddings from safetensors...")
    wt_mods = set()
    for name, mod in model.named_modules():
        if isinstance(mod, MTPWormholeLinear):
            wt_mods.add(name)

    index_path = Path(MODEL_DIR) / 'model.safetensors.index.json'
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    # Collect all non-wormhole params and buffers that need loading
    shard_keys = defaultdict(list)

    def collect(name, is_param=True):
        mod_name = '.'.join(name.split('.')[:-1])
        if mod_name in wt_mods:
            return  # skip WormholeLinear params
        sf_key = name.replace('model.', 'model.language_model.', 1)
        if sf_key in weight_map:
            shard_keys[weight_map[sf_key]].append((name, sf_key, is_param))
        elif name in weight_map:
            shard_keys[weight_map[name]].append((name, name, is_param))

    for name, _ in model.named_parameters():
        collect(name, True)
    for name, _ in model.named_buffers():
        collect(name, False)

    from safetensors import safe_open
    loaded = 0
    for shard_name, items in shard_keys.items():
        shard_path = Path(MODEL_DIR) / shard_name
        if not shard_path.exists():
            print(f"    WARNING: {shard_name} not found")
            continue
        with safe_open(str(shard_path), framework='pt', device='cpu') as sf:
            for param_name, sf_key, is_param in items:
                try:
                    val = sf.get_tensor(sf_key)
                    parts = param_name.split('.')
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    target = getattr(parent, parts[-1])
                    target.data = val.to(DEVICE, dtype=target.dtype)
                    loaded += 1
                except Exception:
                    pass

    print(f"    Loaded: {loaded} params+buffers")
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"    GPU after loading: {alloc:.2f} GB")

    # 6. Inference test
    print(f"\n[6] Running inference...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "The capital of France is",
    ]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            text = text.encode('ascii', errors='replace').decode('ascii')
        except:
            pass
        print(f"    {new_tokens/dt:.1f} tok/s | {repr(text[:120])}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == '__main__':
    main()
