"""
Wormhole .holo Decoder -- Self-Contained (v2)
===============================================
Reconstructs full .holo from wormhole file alone.
Shared SVh replicated to exact layers, not ranges.
"""
import torch, re
from collections import defaultdict
import _paths

CAT_PREFIX = "model.language_model.layers"

def decode_wormhole_holo(wormhole_path):
    worm = torch.load(wormhole_path, map_location='cpu', weights_only=True)

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

    # Exact layer set for each weight type (from wormhole groups)
    all_layers = {}
    for wt, g in groups.items():
        all_layers[wt] = {g['first_l']} | set(g['rots'].keys())

    # Reconstruct U per (wt, layer)
    rec_u = {}
    rec_svh = {}
    for wt, g in groups.items():
        if g['first_U'] is None: continue
        first_l = g['first_l']
        rec_u[(wt, first_l)] = g['first_U'].float()
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
    compressed_prefixes = set(groups.keys())
    for key, val in worm.items():
        if '.L' in key: continue
        if key.endswith('.SVh'):
            wt = key.replace('.SVh', '')
            if wt in compressed_prefixes:
                rec_svh[wt] = val

    # Build output
    output = {}
    wt_to_layers = all_layers  # exact layer mapping

    for key, val in worm.items():
        if '.L' in key: continue

        if key.endswith('.SVh'):
            wt = key.replace('.SVh', '')
            if wt in wt_to_layers:
                # Shared SVh -- replicate only to exact layers
                for l in sorted(wt_to_layers[wt]):
                    cat_key = f"{CAT_PREFIX}.{l}.{wt}.SVh"
                    output[cat_key] = val
            else:
                # Non-compressed SVh (visual, lm_head etc.) -- keep as-is
                output[key] = val

        elif key.endswith('.U'):
            # Non-compressed U -- keep as-is
            output[key] = val

        else:
            # Everything else
            output[key] = val

    # Insert reconstructed U
    for (wt, l), u_val in rec_u.items():
        cat_key = f"{CAT_PREFIX}.{l}.{wt}.U"
        output[cat_key] = u_val.half()

    return output, groups, all_layers, rec_svh


if __name__ == "__main__":
    import sys, os
    worm_path = sys.argv[1] if len(sys.argv) > 1 else str(_paths.HOLO_MODELS / "qwen_27b_wormhole_v2.holo")
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(_paths.HOLO_MODELS / "qwen_27b_decoded.holo")

    in_size = os.path.getsize(worm_path) / 1024**2
    print(f"Wormhole: {os.path.basename(worm_path)} ({in_size:.0f} MB)")

    output, groups, all_layers, rec_svh = decode_wormhole_holo(worm_path)

    n_u = sum(1 for k in output if k.endswith('.U'))
    n_svh = sum(1 for k in output if k.endswith('.SVh'))
    n_other = len(output) - n_u - n_svh
    print(f"  Groups: {len(groups)} (wormhole-compressed)")
    print(f"  Shared SVh: {len(rec_svh)} types -> {sum(len(v) for v in all_layers.values())} per-layer replicas")
    print(f"  Output: {n_u} U, {n_svh} SVh, {n_other} other = {len(output)} keys total")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(output, out_path)
    out_size = os.path.getsize(out_path) / 1024**2
    print(f"  Saved: {out_path} ({out_size:.0f} MB)")

    # Fidelity spot-check
    if 'THOUGHT' in output.get('', ''): pass  # dummy
    cat_path = str(_paths.CATALYTIC_27B)
    if os.path.exists(cat_path):
        cat = torch.load(cat_path, map_location='cpu', weights_only=True)
        coses = []
        for (wt, l), u_rec in rec_u.items():
            cat_key = f"{CAT_PREFIX}.{l}.{wt}.U"
            if cat_key in cat:
                orig = cat[cat_key].float()
                cos = torch.nn.functional.cosine_similarity(
                    u_rec.flatten().unsqueeze(0), orig.flatten().unsqueeze(0)
                ).item()
                coses.append(cos)
        if coses:
            import numpy as np
            print(f"  Fidelity vs catalytic: mean={np.mean(coses):.4f} min={np.min(coses):.4f} (N={len(coses)})")
