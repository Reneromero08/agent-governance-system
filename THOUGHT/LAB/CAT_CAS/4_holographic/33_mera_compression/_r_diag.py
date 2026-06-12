import torch, re, numpy as np
from pathlib import Path; from collections import defaultdict
import sys; sys.path.insert(0, str(Path('THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression')))
from _paths import LLM_WORMHOLE

worm = torch.load(str(LLM_WORMHOLE), map_location='cpu', weights_only=True)
pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))
for key, val in worm.items():
    m = pattern.match(key)
    if not m: continue
    wt, ls, field = m.groups()
    l = int(ls)
    g = groups[wt]
    if field == 'U': g['first_U'] = val; g['first_l'] = l
    elif field == 'R': g['rots'][l] = val
    elif field == 'res_idx': g['res'].setdefault(l,{})['idx'] = val
    elif field == 'res_max':
        if l in g['res']: g['res'][l]['max'] = val

print('R = Tr(rho*C) per weight type')
print('Weight Type                        R_mean   R_min    R_std    Verdict')
print('-' * 75)
for wt, g in sorted(groups.items()):
    anchor = g['first_U'].float()
    R_vals = []
    raw_vals = []  # fidelity of U_recon vs original per layer
    for l in sorted(g['rots'].keys()):
        U_recon = anchor @ g['rots'][l].float()
        # R = cos(U_recon, U_anchor) — how much anchor subspace survives rotation
        cos_anchor = torch.nn.functional.cosine_similarity(
            anchor.flatten().unsqueeze(0), U_recon.flatten().unsqueeze(0)).item()
        R_vals.append(cos_anchor)
        
        # Fidelity = how much rotation retains info (proxy for reconstruction quality)
        # If R is near identity: cos ≈ 1. If R rotates aggressively: cos ≈ 0
        # But this is SUBSPACE preservation, not reconstruction quality
        n_dims = anchor.shape[1]
        raw_vals.append((anchor @ g['rots'][l].float()).norm().item() / anchor.norm().item())
    
    if R_vals:
        rm, rs = np.mean(R_vals), np.std(R_vals)
        fm = np.mean(raw_vals)  # average energy preservation
        # R-guided verdict: if energy isn't preserved (R < 0.7), the rotation is too aggressive
        v = 'KEEP' if fm > 0.7 else ('NOISY' if fm > 0.5 else 'PRUNE')
        print(f'{wt:<35} R={rm:>6.4f} E={fm:>6.4f} {v:>12}')
