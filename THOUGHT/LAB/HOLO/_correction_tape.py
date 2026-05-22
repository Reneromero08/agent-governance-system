"""Hayden-Preskill Correction Tape — 5-step fix chain for gibberish.

1. ANALYTIC: dR = U_anchor^T @ U_teacher - R_base       (instant, optimal)
2. CORRECTION TAPE: per-layer teacher-student delta      (weight-level)
3. RANK-1 COMPRESS: SVD -> dominant phase vector          (2 KB per layer)
4. SAVE TAPE: compressed correction to disk              (~10 MB total)
5. AT INFERENCE: WormholeLinear += decompress_correction  (unscramble noise)

The gibberish is Hawking radiation from the wormhole. The correction tape
recovers the thrown-in diary — Exp 32's Hayden-Preskill protocol.
"""
import torch,os,math
from collections import defaultdict
from pathlib import Path

HOLO = Path(r'THOUGHT/LAB/HOLO/_models')
teacher_path = HOLO / 'qwen_27b_cavitated.holo'
student_path = HOLO / 'qwen_27b_llm_cavity_wormhole.holo'
out_path = HOLO / 'qwen_27b_correction_tape.pt'

print('Loading .holo files...')
teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
student = torch.load(student_path, map_location='cpu', weights_only=True)
print(f'  Teacher: {len(teacher)} keys')
print(f'  Student: {len(student)} keys')

# Parse student wormhole
import re
pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
s_groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))

for key, val in student.items():
    m = pattern.match(key)
    if not m: continue
    wt, ls, field = m.groups()
    l = int(ls)
    g = s_groups[wt]
    if field == 'U': g['first_U'] = val; g['first_l'] = l
    elif field == 'R': g['rots'][l] = val
    elif field == 'res_idx': g['res'].setdefault(l, {})['idx'] = val
    elif field == 'res_max':
        if l in g['res']: g['res'][l]['max'] = val

CAT = 'model.language_model.layers'
tape = {}
total_entries = 0

for wt, g in sorted(s_groups.items()):
    all_layers = sorted(set([g['first_l']] + list(g['rots'].keys())))
    
    for l in all_layers:
        # STEP 1: Get teacher U for this layer
        t_key = f'{CAT}.{l}.{wt}.U'
        if t_key not in teacher: continue
        t_U = teacher[t_key].float()
        
        # Get student U (reconstructed from rotations)
        if l == g['first_l']:
            s_U = g['first_U'].float()
        elif l in g['rots']:
            anchor = g['first_U'].float()
            R = g['rots'][l].float()
            s_U = anchor @ R
            if l in g['res'] and g['res'][l].get('idx') is not None:
                rd = g['res'][l]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                s_U = s_U + levels[rd['idx'].long()]
        else: continue
        
        # STEP 2: Correction tape = teacher_U - student_U  (U-level)
        # This is the Hayden-Preskill diary — what the wormhole lost
        delta_U = t_U - s_U  # (out_dim, k)
        
        # STEP 3: Rank-1 compression via SVD dominant component
        U_d, S_d, Vh_d = torch.linalg.svd(delta_U.float(), full_matrices=False)
        # Keep only the dominant singular vector (rank-1 approximation)
        u1 = U_d[:, 0] * math.sqrt(S_d[0].item())  # (out_dim,) dominant left vector
        v1 = Vh_d[0, :] * math.sqrt(S_d[0].item()) # (k,) dominant right vector
        
        # Store compressed correction: only 2 vectors per layer
        tape[f'{wt}.L{l}'] = {
            'u': u1.half(),  # (out_dim,) ~ out_dim * 2 bytes
            'v': v1.half(),  # (k,) ~ k * 2 bytes
            's': S_d[0].item(),  # scalar
        }
        total_entries += 1

torch.save(tape, out_path)
entries = len(tape)
total_size = os.path.getsize(out_path)
total_params = sum(v['u'].numel() + v['v'].numel() for v in tape.values())

print(f'\n=== CORRECTION TAPE ===')
print(f'  Entries: {entries} layers')
print(f'  Total params: {total_params:,}')
print(f'  File size: {total_size/1024:.0f} KB')
print(f'  Per-layer: ~{total_params/entries:.0f} params ({total_params*2/entries:.0f} bytes)')
print(f'  Saved: {out_path}')

# Show a few entries
for key in sorted(tape.keys())[:3]:
    v = tape[key]
    print(f'  {key}: u={list(v["u"].shape)}, v={list(v["v"].shape)}, s={v["s"]:.4f}')
