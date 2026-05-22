"""Verify wormhole U reconstruction fidelity."""
import torch, re
from collections import defaultdict

worm = torch.load('THOUGHT/LAB/CAT_CAS/33_mera_compression/qwen_27b_wormhole.holo', map_location='cpu', weights_only=True)
cat = torch.load('THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo', map_location='cpu', weights_only=True)

pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))

for key, val in worm.items():
    m = pattern.match(key)
    if not m: continue
    wt, layer_str, field = m.groups()
    l = int(layer_str)
    g = groups[wt]
    if field == 'U':
        g['first_U'] = val
        g['first_l'] = l
    elif field == 'R':
        g['rots'][l] = val
    elif field == 'res_idx':
        g['res'].setdefault(l, {})['idx'] = val
    elif field == 'res_max':
        if l in g['res']:
            g['res'][l]['max'] = val

print(f"Groups: {len(groups)}")

for wt, g in sorted(groups.items()):
    prev_U = g['first_U'].float()
    src_key0 = f'model.language_model.layers.{g["first_l"]}.{wt}.U'
    if src_key0 in cat:
        orig0 = cat[src_key0].float()
        m0 = torch.allclose(prev_U, orig0, rtol=1e-3)
    else:
        m0 = 'N/A'

    coses = []
    for l in sorted(g['rots'].keys()):
        R = g['rots'][l].float()
        recon = prev_U @ R
        if l in g['res']:
            rd = g['res'][l]
            mval = rd.get('max', torch.tensor(1e-6)).item()
            levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
            residual = levels[rd['idx'].long()]
            recon = recon + residual
        
        src_key = f'model.language_model.layers.{l}.{wt}.U'
        if src_key in cat:
            orig = cat[src_key].float()
            cos = torch.nn.functional.cosine_similarity(recon.flatten().unsqueeze(0), orig.flatten().unsqueeze(0)).item()
            coses.append(cos)
        
        prev_U = g['first_U'].float()  # anchor from first
    
    if coses:
        avg = sum(coses) / len(coses)
        shp = list(g['first_U'].shape)
        print(f"  {wt}: shape={shp} layers={len(g['rots'])} first_layer={g['first_l']} base_match={m0} cos_mean={avg:.4f} cos_min={min(coses):.4f}")
    else:
        shp = list(g['first_U'].shape)
        print(f"  {wt}: shape={shp} layers={len(g['rots'])} first_layer={g['first_l']} base_match={m0} (no rotations)")
