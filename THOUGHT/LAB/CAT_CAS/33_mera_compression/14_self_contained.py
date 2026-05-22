"""
Self-Contained .holo Builder
=============================
Makes wormhole files truly standalone — no HF cache, no model directory,
no config download. Everything the loader needs is embedded in the .holo.

Adds:
  _config     — model config as JSON bytes (arch, vocab, hidden dim, etc.)
  embed       — token embedding weight (from safetensors)
  norm        — layer norm weights (from safetensors)
  bias        — Linear layer biases (from safetensors, optional)

Reads these from the original safetensors ONCE, then they're in the .holo forever.

Usage:
  python 14_self_contained.py <cavitated.holo> <safetensors_dir> <output.holo>
  python 14_self_contained.py --module llm  (existing files)
"""
import torch, json, struct, mmap, os, sys, re
from pathlib import Path
from collections import defaultdict
import _paths


def read_safetensors_metadata(path):
    """Read safetensors header to get tensor list + shapes without loading data."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header


def load_tensor_from_safetensors(filepath, tensor_name, header=None):
    """Load a single tensor from safetensors without loading the whole file."""
    if header is None:
        header = read_safetensors_metadata(filepath)
    
    if tensor_name not in header:
        return None
    
    info = header[tensor_name]
    start, end = info['data_offsets']
    dtype_str = info.get('dtype', 'F32')
    shape = info['shape']
    
    with open(filepath, 'rb') as f:
        f.seek(8 + len(json.dumps(header)) + start)
        raw = f.read(end - start)
    
    import numpy as np
    if dtype_str == 'BF16':
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (arr << 16).view(np.float32).reshape(shape)
        return torch.from_numpy(arr.copy()).to(torch.bfloat16)
    elif dtype_str == 'F16':
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).reshape(shape).copy())
    else:
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(shape).copy())


def embed_config(holo_dict, model_dir_or_config):
    """
    Embed model config into .holo as _config key.
    Accepts: directory path (reads config.json), or a dict.
    """
    if isinstance(model_dir_or_config, dict):
        config = model_dir_or_config
    elif isinstance(model_dir_or_config, (str, Path)):
        config_path = Path(model_dir_or_config) / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            print(f"  WARNING: config.json not found at {config_path}")
            return holo_dict
    else:
        return holo_dict
    
    config_bytes = json.dumps(config).encode('utf-8')
    # Store as uint8 tensor so torch.save handles it natively
    holo_dict['_config'] = torch.ByteTensor(list(config_bytes))
    print(f"  Embedded config: {len(config_bytes)} bytes, keys: {list(config.keys())[:8]}...")
    return holo_dict


def embed_embed_norm(holo_dict, safetensors_path, config=None):
    """
    Read embed_tokens + norm weights from safetensors and embed into .holo.
    Keys stored as: _embed.weight, _norm.{name}.weight
    
    Only loads these specific tensors — NOT the full model.
    """
    header = read_safetensors_metadata(safetensors_path)
    all_keys = list(header.keys())
    
    # Find embed tokens
    embed_keys = [k for k in all_keys if 'embed_tokens' in k and 'weight' in k]
    norm_keys = [k for k in all_keys if 'norm' in k and 'weight' in k]
    
    loaded = 0
    skipped = 0
    
    for key in embed_keys:
        t = load_tensor_from_safetensors(safetensors_path, key, header)
        if t is not None:
            safe_key = '_embed.' + key.replace('.', '_')
            holo_dict[safe_key] = t
            loaded += 1
            print(f"  Embed: {key} -> {safe_key} [{list(t.shape)}]")
    
    for key in norm_keys:
        # Skip norms that are just for visual blocks (they're tiny, handle separately)
        t = load_tensor_from_safetensors(safetensors_path, key, header)
        if t is not None:
            safe_key = '_norm.' + key.replace('.', '_')
            holo_dict[safe_key] = t
            loaded += 1
            print(f"  Norm:  {key} -> {safe_key} [{list(t.shape)}]")
    
    print(f"  Embedded {loaded} tensors ({skipped} skipped)")
    return holo_dict


def embed_bias(holo_dict, safetensors_path):
    """Read Linear bias terms from safetensors and embed into .holo."""
    header = read_safetensors_metadata(safetensors_path)
    bias_keys = [k for k in header if 'bias' in k]
    
    loaded = 0
    for key in bias_keys:
        t = load_tensor_from_safetensors(safetensors_path, key, header)
        if t is not None:
            safe_key = '_bias.' + key.replace('.', '_')
            holo_dict[safe_key] = t
            loaded += 1
    
    print(f"  Embedded {loaded} bias tensors")
    return holo_dict


def extract_config_from_holo(holo_dict):
    """Extract config JSON from _config key in .holo."""
    if '_config' in holo_dict:
        config_tensor = holo_dict['_config']
        if isinstance(config_tensor, torch.Tensor):
            config_bytes = bytes(config_tensor.tolist())
        else:
            config_bytes = config_tensor
        return json.loads(config_bytes.decode('utf-8'))
    return None


def extract_embed_norm_from_holo(holo_dict):
    """Extract embed and norm tensors back to HF key format."""
    result = {}
    for key, val in holo_dict.items():
        if key.startswith('_embed.'):
            orig_key = key.replace('_embed.', '').replace('_', '.')
            result[orig_key] = val
        elif key.startswith('_norm.'):
            orig_key = key.replace('_norm.', '').replace('_', '.')
            result[orig_key] = val
        elif key.startswith('_bias.'):
            orig_key = key.replace('_bias.', '').replace('_', '.')
            result[orig_key] = val
    return result


def build_self_contained(cavitated_path, model_dir, output_path):
    """
    Make a cavitated .holo self-contained by adding config + embed + norm.
    
    Args:
        cavitated_path: existing cavitated .holo
        model_dir: directory with config.json + model.safetensors
        output_path: output .holo path
    """
    print(f"Loading cavitated: {cavitated_path}")
    holo = torch.load(cavitated_path, map_location='cpu', weights_only=True)
    
    model_dir = Path(model_dir)
    
    # 1. Embed config
    print(f"Reading config from: {model_dir}")
    holo = embed_config(holo, model_dir)
    
    # 2. Embed embed + norm
    safetensors_files = list(model_dir.glob('*.safetensors'))
    if not safetensors_files:
        safetensors_files = list(model_dir.glob('model*.safetensors'))
    
    if safetensors_files:
        sf_path = str(safetensors_files[0])
        print(f"Reading embed/norm from: {sf_path}")
        holo = embed_embed_norm(holo, sf_path)
        holo = embed_bias(holo, sf_path)
    else:
        print("  WARNING: No safetensors found in model dir")
    
    # 3. Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(holo, output_path)
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Saved: {output_path} ({size_mb:.0f} MB)")
    
    return holo


if __name__ == "__main__":
    # Paths
    cavitated = _paths.CAVITATED_27B
    
    # Try common model paths
    model_dirs = [
        _paths.REPO / "THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/gemini_update/qwen_27b",
        _paths.REPO / "THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/gemini_update/qwen_0.5b",
        _paths.REPO / "THOUGHT/LAB/HOLO/holographic_brain",
    ]
    
    model_dir = None
    for md in model_dirs:
        if md.exists() and (list(md.glob('*.safetensors')) or (md / 'config.json').exists()):
            model_dir = str(md)
            break
    
    if not model_dir:
        print("No model directory found. Usage:")
        print("  python 14_self_contained.py <cavitated.holo> <model_dir> <output.holo>")
        sys.exit(1)
    
    output = _paths.HOLO_MODELS / "qwen_27b_standalone.holo"
    
    if len(sys.argv) >= 3:
        cavitated = Path(sys.argv[1])
        model_dir = sys.argv[2]
        output = Path(sys.argv[3]) if len(sys.argv) >= 4 else cavitated.parent / f"{cavitated.stem}_standalone.holo"
    
    build_self_contained(str(cavitated), model_dir, str(output))
