"""
Modular Wormhole Compressor -- EIGEN_BUDDY-style
==================================================
Splits .holo into independent modules, each wormhole-compressed.
No monolith. Load only what the task needs.

Modules:
  llm     -- 12 types: mlp(down/gate/up) + self_attn(k/q/v/o) + linear_attn(in_a/b/qkv/z/out)
  visual  --  6 types: attn(qkv/proj) + mlp(fc1/fc2) + merger(fc1/fc2)

Usage:
  python 7_modular_compress.py <input.holo> --module llm    > llm_wormhole.holo
  python 7_modular_compress.py <input.holo> --module visual > visual_wormhole.holo
  python 7_modular_compress.py <input.holo> --module all    > _full_wormhole.holo
"""
import torch, math, numpy as np, os, sys, re
from collections import defaultdict
from pathlib import Path
import _paths

MODULES = {
    "llm": {
        "tag": ("layers", "blk"),
        "types": ("mlp", "self_attn", "linear_attn", "ffn", "attn", "ssm", "nextn"),
    },
    "visual": {
        "tag": ("blocks",),
        "types": ("attn", "mlp", "merger"),
    },
    "aux": {
        "tag": (),
        "types": (),
        "catch_all": False,
        "explicit_keys": ["lm_head", "embed_tokens", "norm", "model.language_model.norm",
                          "_config", "_embed", "_norm", "_bias"],  # self-contained holos
    },
}

def group_u_matrices(holo_dict, modules):
    """Group U matrices by weight type, filtered by module tags/types."""
    u_groups = defaultdict(dict)
    for key, val in holo_dict.items():
        if not key.endswith('.U') or val.ndim != 2: continue
        parts = key.split('.')
        layer_idx = None; wt = None; ok = False
        for mod_name, cfg in modules.items():
            for tag in cfg["tag"]:
                if not tag: continue
                if tag in parts:
                    i = parts.index(tag)
                    try: layer_idx = int(parts[i + 1])
                    except: pass
                    for t in cfg["types"]:
                        if t in parts or any(t in p for p in parts):
                            wt = '.'.join(parts[i + 2:-1])
                            ok = True
                            break
                    break
            if ok: break
        if layer_idx is not None and wt is not None:
            u_groups[wt][layer_idx] = val.float()
    return u_groups


def compress_group(wt, tensors, compressed, stats, rotation_threshold=0.5, quant_bits=2, skip_threshold=0.01):
    """Wormhole-compress a single weight-type group with Skip-R identity detection."""
    sorted_l = sorted(tensors.keys())
    if len(sorted_l) < 2:
        for l in sorted_l:
            compressed[f"{wt}.L{l}.U"] = tensors[l].half()
        return

    first = tensors[sorted_l[0]]
    m, k = first.shape
    L = len(sorted_l)
    compressed[f"{wt}.L{sorted_l[0]}.U"] = first.half()

    prev = first
    fids_rot = []; fids_quant = []
    skipped = 0
    orig_bits = L * m * k * 16
    comp_bits = m * k * 16

    for i in range(1, L):
        l = sorted_l[i]
        curr = tensors[l]

        R = prev.T @ curr
        recon_rot = prev @ R
        fid_rot = torch.nn.functional.cosine_similarity(
            curr.flatten().unsqueeze(0), recon_rot.flatten().unsqueeze(0)
        ).item()
        
        # B2: Skip-R detection — near-identity rotation, skip entirely
        # Uses absolute Frobenius norm per PUSHED_REPORT: ||R - I|| < skip_threshold
        identity_dist = torch.norm(R - torch.eye(k, device=R.device)).item()
        if identity_dist < skip_threshold:
            # Zero-copy skip: don't store R or residual. Decoder reuses anchor.
            compressed[f"{wt}.L{l}.skip"] = torch.tensor(1)  # skip token
            skipped += 1
            fids_rot.append(1.0); fids_quant.append(1.0)
            continue

        residual = curr - recon_rot
        res_max = residual.abs().max().item()

        if fid_rot > rotation_threshold:
            compressed[f"{wt}.L{l}.R"] = R.half()
            comp_bits += k * k * 16
            fid_quant = fid_rot
        else:
            levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(res_max, 1e-6)
            residual_norm = residual / max(res_max, 1e-6)
            diffs = residual_norm.unsqueeze(-1) - levels.view(1, 1, -1)
            idx = diffs.abs().argmin(dim=-1).to(torch.uint8)

            compressed[f"{wt}.L{l}.R"] = R.half()
            compressed[f"{wt}.L{l}.res_idx"] = idx
            compressed[f"{wt}.L{l}.res_max"] = torch.tensor(res_max)
            comp_bits += k * k * 16 + m * k * quant_bits + 16

            recon_quant = recon_rot + levels[idx.long()]
            fid_quant = torch.nn.functional.cosine_similarity(
                curr.flatten().unsqueeze(0), recon_quant.flatten().unsqueeze(0)
            ).item()

        fids_rot.append(fid_rot); fids_quant.append(fid_quant)
        prev = curr

    ratio = orig_bits / comp_bits if comp_bits > 0 else 1.0
    stats['groups'][wt] = {
        'L': L, 'm': m, 'k': k, 'ratio': ratio,
        'fid_rot': np.mean(fids_rot), 'fid_quant': np.mean(fids_quant),
        'orig_MB': orig_bits / 8 / 1024**2,
        'comp_MB': comp_bits / 8 / 1024**2,
        'skipped': skipped,
    }
    stats['total_orig_MB'] += orig_bits / 8 / 1024**2
    stats['total_comp_MB'] += comp_bits / 8 / 1024**2
    stats['fidelities'][wt] = fids_quant


def compress_holo_modular(holo_dict, modules, rotation_threshold=0.5, quant_bits=2, skip_threshold=0.3):
    """Compress .holo dict into modular wormhole format."""
    u_groups = group_u_matrices(holo_dict, modules)

    compressed = {}
    stats = {'groups': {}, 'total_orig_MB': 0, 'total_comp_MB': 0, 'fidelities': {}}

    for wt, tensors in sorted(u_groups.items()):
        compress_group(wt, tensors, compressed, stats, rotation_threshold, quant_bits, skip_threshold)

    # Collect compressed prefixes for SVh dedup
    compressed_prefixes = set()
    for key in compressed:
        m = re.match(r'(.+)\.L\d+\.', key)
        if m: compressed_prefixes.add(m.group(1))

    # Fallthrough: SVh, non-compressed U, other -- FILTERED by module
    # Only keep entries whose prefix matches an active module's tag
    active_tags = []
    active_types = []
    for mod_name, cfg in modules.items():
        active_tags.extend(cfg['tag'])
        active_types.extend(cfg['types'])
    
    def _key_belongs_to_module(key):
        """Check if a catalytic key belongs to any active module."""
        parts = key.split('.')
        # Explicit key matching (aux: lm_head, embed_tokens, norm)
        for mod_name, cfg in modules.items():
            if 'explicit_keys' in cfg:
                for ek in cfg['explicit_keys']:
                    if ek in key:
                        return True
        for tag in active_tags:
            if not tag: continue
            if tag in parts:
                idx = parts.index(tag)
                if idx + 1 < len(parts):
                    try: int(parts[idx + 1])
                    except: continue
                    for t in active_types:
                        if t in parts:
                            return True
        return False
    
    svh_stored = set()
    for key, val in holo_dict.items():
        if key in compressed: continue

        parts = key.split('.')
        wt = None

        # Determine weight type from key
        for tag_candidate in ('layers', 'blocks', 'blk'):
            if tag_candidate in parts:
                idx = parts.index(tag_candidate)
                try: int(parts[idx + 1])
                except: continue
                wt = '.'.join(parts[idx + 2:-1])
                break
        if wt is None:
            wt = '.'.join(parts[:-1])

        if key.endswith('.SVh'):
            if wt in compressed_prefixes:
                shared_key = f"{wt}.SVh"
                if shared_key not in svh_stored:
                    compressed[shared_key] = val
                    svh_stored.add(shared_key)
            elif _key_belongs_to_module(key):
                compressed[key] = val
            # else: SKIP -- not in this module

        elif key.endswith('.U'):
            if wt not in compressed_prefixes and _key_belongs_to_module(key):
                compressed[key] = val
            # else: SKIP -- not in this module

        elif 'weight' in parts:
            # Other weight-like entries (embed, norm) -- keep only if in module
            if _key_belongs_to_module(key):
                compressed[key] = val
        # else: skip non-weight non-module entries

    return compressed, stats


# ---- CLI ----

DEFAULT_INPUT = _paths.CATALYTIC_27B
OUT_DIR = _paths.HOLO_MODELS

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    ap.add_argument("--module", "-m", choices=["llm", "visual", "aux", "all"], default="all")
    ap.add_argument("--output", "-o", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        return

    if args.module == "all":
        modules = MODULES
        suffix = "full"
    else:
        modules = {args.module: MODULES[args.module]}
        suffix = args.module

    if args.output:
        out_path = args.output
    else:
        out_path = str(OUT_DIR / f"qwen_27b_{suffix}_wormhole.holo")

    print(f"Module: {args.module} ({', '.join(modules.keys())})")
    print(f"Loading {args.input}...")
    holo = torch.load(args.input, weights_only=False)

    print(f"Compressing...")
    compressed, stats = compress_holo_modular(holo, modules)

    print(f"\n  {'Group':<30} {'L':>4} {'fid_rot':>8} {'fid+res':>8} {'ratio':>6} {'skip':>5}")
    print(f"  {'-'*65}")
    for wt, s in sorted(stats['groups'].items()):
        print(f"  {wt:<30} {s['L']:>4} {s['fid_rot']:>8.3f} {s['fid_quant']:>8.3f} {s['ratio']:>5.1f}x {s.get('skipped',0):>5}")

    overall_ratio = stats['total_orig_MB'] / stats['total_comp_MB'] if stats['total_comp_MB'] > 0 else 1.0
    mean_fid = np.mean([np.mean(f) for f in stats['fidelities'].values()]) if stats['fidelities'] else 0
    print(f"\n  OVERALL: {stats['total_orig_MB']:.0f}MB -> {stats['total_comp_MB']:.0f}MB ({overall_ratio:.1f}x)")
    print(f"  Mean fidelity: {mean_fid:.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(compressed, out_path)
    out_size = os.path.getsize(out_path) / 1024**2
    print(f"\n  Saved: {out_path} ({out_size:.1f} MB)")

if __name__ == "__main__":
    main()
