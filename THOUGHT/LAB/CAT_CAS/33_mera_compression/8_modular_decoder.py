"""
Modular Wormhole Decoder -- Multi-File Loader
===============================================
Loads any combination of wormhole module files and reconstructs
the full .holo state dict. MoE-style: load only what's requested.

Modules:
  llm     -- qwen_27b_llm_wormhole.holo
  visual  -- qwen_27b_visual_wormhole.holo
  full    -- qwen_27b_full_wormhole.holo (all in one)

Usage:
  from modular_decoder import load_modules
  sd = load_modules(["llm"])           # text-only
  sd = load_modules(["llm", "visual"])  # multimodal
  sd = load_modules(["full"])           # everything
"""
import torch, re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("THOUGHT/LAB/CAT_CAS/33_mera_compression")

MODULE_MAP = {
    "llm":    OUT_DIR / "qwen_27b_llm_wormhole.holo",
    "visual": OUT_DIR / "qwen_27b_visual_wormhole.holo",
    "full":   OUT_DIR / "qwen_27b_full_wormhole.holo",
}

PREFIX_MAP = {
    "layers": "model.language_model.layers",
    "blocks": "model.visual.blocks",
    "merger": "model.visual.merger",
}

def _parse_wormhole(worm, all_layers, rec_u, shared_svh):
    """Parse one wormhole file's groups into shared structures."""
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

    compressed_prefixes = set(groups.keys())

    # Reconstruct U
    for wt, g in groups.items():
        if g['first_U'] is None: continue
        first_l = g['first_l']
        rec_u[(wt, first_l)] = g['first_U'].float()
        all_layers[wt] = {first_l} | set(g['rots'].keys())

        anchor = g['first_U'].float()
        for l in sorted(g['rots'].keys()):
            R = g['rots'][l].float()
            recon = anchor @ R
            if l in g['res'] and g['res'][l].get('idx') is not None:
                rd = g['res'][l]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                residual = levels[rd['idx'].long()]
                recon = recon + residual
            rec_u[(wt, l)] = recon

    # Collect shared SVh
    for key, val in worm.items():
        if '.L' in key: continue
        if key.endswith('.SVh'):
            wt = key.replace('.SVh', '')
            if wt in compressed_prefixes:
                shared_svh[wt] = val

    return compressed_prefixes


def _resolve_prefix(wt):
    """Determine catalytic key prefix for a weight type by scanning known prefixes."""
    # Check if weight type belongs to a known tag
    for tag, prefix in [("layers", "model.language_model.layers"),
                         ("blocks", "model.visual.blocks"),
                         ("merger", "model.visual.merger")]:
        # Weight types don't have the tag directly. Heuristic: known types
        pass

    # Fallback: try to match known patterns
    known_merger = {"merger.linear_fc1.weight", "merger.linear_fc2.weight"}
    if wt in known_merger:
        return "model.visual.merger"

    # For multilayered types: default to language_model.layers
    return "model.language_model.layers"


def load_modules(module_names, module_dir=None):
    """
    Load one or more wormhole modules and return a full state dict.
    
    Args:
        module_names: list of module keys ("llm", "visual", "full")
        module_dir: override module file directory
    
    Returns:
        state_dict: dict mapping catalytic key -> tensor (fp16)
    """
    base = Path(module_dir) if module_dir else OUT_DIR
    all_layers = {}
    rec_u = {}
    shared_svh = {}
    passthrough = {}

    for name in module_names:
        path = base / f"qwen_27b_{name}_wormhole.holo" if module_dir else MODULE_MAP.get(name, base / f"qwen_27b_{name}_wormhole.holo")
        if not path.exists():
            print(f"  WARNING: module '{name}' not found at {path}, skipping")
            continue

        worm = torch.load(str(path), map_location='cpu', weights_only=True)
        compressed_prefixes = _parse_wormhole(worm, all_layers, rec_u, shared_svh)

        # Pass-through entries (non-wormhole SVh/U/other)
        for key, val in worm.items():
            if '.L' in key: continue
            parts = key.split('.')
            wt = key.replace('.SVh', '') if key.endswith('.SVh') else '.'.join(parts[:-1]) if key.endswith('.U') else None

            if key.endswith('.SVh'):
                wt_s = key.replace('.SVh', '')
                if wt_s in compressed_prefixes:
                    continue  # already in shared_svh
            elif key.endswith('.U'):
                if wt and wt in compressed_prefixes:
                    continue  # already in rec_u
            passthrough[key] = val

    # Build output state dict
    output = {}

    # 1. Reconstructed U (wormhole-compressed)
    for (wt, l), u_val in rec_u.items():
        prefix = _resolve_prefix(wt)
        if "merger" in prefix:
            key = f"{prefix}.{wt}.U"
        else:
            key = f"{prefix}.{l}.{wt}.U"
        output[key] = u_val.half()

    # 2. Shared SVh -> replicate to exact layers
    for wt, svh_val in shared_svh.items():
        if wt in all_layers:
            prefix = _resolve_prefix(wt)
            for l in sorted(all_layers[wt]):
                key = f"{prefix}.{l}.{wt}.SVh"
                output[key] = svh_val
        else:
            # Single-layer non-compressed SVh
            output[f"{wt}.SVh"] = svh_val

    # 3. Pass-through entries
    for key, val in passthrough.items():
        # Avoid duplicates with shared SVh
        if key not in output:
            output[key] = val

    return output


def decode_to_file(module_names, output_path, module_dir=None):
    """Load modules and save decoded .holo to disk."""
    sd = load_modules(module_names, module_dir)
    torch.save(sd, output_path)
    size_mb = Path(output_path).stat().st_size / 1024**2
    n_u = sum(1 for k in sd if k.endswith('.U'))
    n_svh = sum(1 for k in sd if k.endswith('.SVh'))
    print(f"  Decoded: {len(sd)} keys ({n_u} U, {n_svh} SVh) -> {output_path} ({size_mb:.0f} MB)")
    return sd


# ---- CLI quick test ----
if __name__ == "__main__":
    import sys
    modules = sys.argv[1:] if len(sys.argv) > 1 else ["llm", "visual"]
    print(f"Modules: {modules}")
    sd = load_modules(modules)
    n_u = sum(1 for k in sd if k.endswith('.U'))
    n_svh = sum(1 for k in sd if k.endswith('.SVh'))
    print(f"Total: {len(sd)} keys ({n_u} U, {n_svh} SVh)")
