"""MTP Wormhole Correction Tape — K=128 student vs K=256 catalytic teacher."""
import torch, re, math
from collections import defaultdict
from pathlib import Path

HOLO = Path(r'THOUGHT/LAB/HOLO/_models')
CATALYTIC = Path(r'THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo')
MTP = HOLO / 'qwen_27b_mtp_wormhole_k128.holo'
OUT = HOLO / 'qwen_27b_mtp_correction_tape.pt'

# Qwen 3.5 -> Qwen 2.5 weight name mapping
NAME_MAP = {
    'mlp.down_proj': 'ffn_down',
    'mlp.gate_proj': 'ffn_gate', 
    'mlp.up_proj':   'ffn_up',
    'self_attn.q_proj': 'attn_q',
    'self_attn.k_proj': 'attn_k',
    'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output',
    'linear_attn.in_proj_qkv': 'attn_qkv',
    'linear_attn.out_proj':   'ssm_out',
    'linear_attn.in_proj_a':  'ssm_alpha',
    'linear_attn.in_proj_b':  'ssm_beta',
    'linear_attn.in_proj_z':  'attn_gate',
    'linear_attn.conv1d.weight': 'ssm_conv1d',
}

print(f'Loading teacher: {CATALYTIC}')
teacher = torch.load(CATALYTIC, map_location='cpu', weights_only=True)
print(f'Loading student: {MTP}')
student = torch.load(MTP, map_location='cpu', weights_only=True)
print(f'  Teacher: {len(teacher)} keys')
print(f'  Student: {len(student)} keys')

# Parse student MTP wormhole
p = re.compile(r'(.+)\.L(\d+)\.(.+)')
s_groups = defaultdict(lambda: dict(first_U=None, first_l=-1, rots={}, res={}))
s_svh = {}
for k, v in student.items():
    m = p.match(k)
    if m:
        wt, ls, f = m.groups(); l = int(ls)
        g = s_groups[wt]
        if f == 'U': g['first_U'] = v; g['first_l'] = l
        elif f == 'R': g['rots'][l] = v
        elif f == 'res_idx': g['res'].setdefault(l, {})['idx'] = v
        elif f == 'res_max':
            if l in g['res']: g['res'][l]['max'] = v
    elif k.endswith('.SVh') and '.L' not in k:
        s_svh[k.replace('.SVh', '')] = v

CAT_PREFIX = 'model.language_model.layers'
tape = {}
total_entries = 0
mapped = 0
unmapped = 0

# Reverse map: Qwen 2.5 name -> Qwen 3.5 name
REV_MAP = {v: k for k, v in NAME_MAP.items()}

for wt_25, g in sorted(s_groups.items()):
    # Map to catalytic (Qwen 3.5) name
    wt_35 = REV_MAP.get(wt_25.replace('.weight', ''))
    if wt_35 is None:
        # Try direct match
        wt_35 = wt_25.replace('.weight', '')
    
    all_layers = sorted(set([g['first_l']] + list(g['rots'].keys())))
    matched = False
    
    for l in all_layers:
        # Teacher U (Qwen 3.5 naming)
        t_key = f'{CAT_PREFIX}.{l}.{wt_35}.weight.U' if wt_35 else None
        if t_key is None or t_key not in teacher:
            # Try alternate mappings
            for key in teacher:
                if f'.{l}.' in key and wt_35 and wt_35 in key and key.endswith('.U'):
                    t_key = key
                    break
        
        if t_key not in teacher:
            continue
        
        t_U = teacher[t_key].float()
        matched = True
        
        # Student U (reconstructed from MTP wormhole)
        if l == g['first_l']:
            s_U = g['first_U'].float()
        elif l in g['rots']:
            anchor = g['first_U'].float()
            R = g['rots'][l].float()
            s_U = anchor @ R
            if l in g['res'] and g['res'][l].get('idx') is not None:
                rd = g['res'][l]
                mval = rd.get('max', torch.tensor(1e-6)).item()
                levels = torch.tensor([-1., -0.333, 0.333, 1.]) * max(abs(mval), 1e-6)
                s_U = s_U + levels[rd['idx'].long()]
        else:
            continue
        
        # Compare at WEIGHT level: W_teacher - W_student
        # Teacher: W_t = U_t @ SVh_t^T (K=256)
        # Student: W_s = U_s @ SVh_s^T (K=128)
        t_svh_key = f'{CAT_PREFIX}.{l}.{wt_35}.weight.SVh'
        t_SVh = teacher.get(t_svh_key)
        if t_SVh is None: 
            t_SVh = s_svh.get(wt_25.replace('.weight',''))
        if t_SVh is None: continue
        t_SVh = t_SVh.float()
        
        s_SVh_mat = s_svh.get(wt_25.replace('.weight',''))
        if s_SVh_mat is None: continue
        s_SVh_mat = s_SVh_mat.float()
        
        # Full weight matrices
        W_t = t_U @ t_SVh.T  # (out_dim, in_dim) at K=256
        W_s = s_U @ s_SVh_mat.T  # (out_dim, in_dim) at K=128
        
        if W_t.shape != W_s.shape:
            # Project to common dimension
            min_in = min(W_t.shape[1], W_s.shape[1])
            W_t = W_t[:, :min_in]
            W_s = W_s[:, :min_in]
        
        delta_W = W_t - W_s  # (out_dim, in_dim)
        
        # Rank-1 compression of weight delta
        U_d, S_d, Vh_d = torch.linalg.svd(delta_W.float(), full_matrices=False)
        u1 = U_d[:, 0] * math.sqrt(max(S_d[0].item(), 1e-9))
        v1 = Vh_d[0, :] * math.sqrt(max(S_d[0].item(), 1e-9))
        
        tape[f'{wt_25}.L{l}'] = {
            'u': u1.half(), 'v': v1.half(), 's': S_d[0].item()
        }
        total_entries += 1
    
    if matched: mapped += 1
    else: unmapped += 1

torch.save(tape, OUT)
print(f'\n  Mapped: {mapped}, Unmapped: {unmapped}')
print(f'  Tape entries: {total_entries} layers')
print(f'  File: {OUT} ({Path(OUT).stat().st_size/1024:.0f} KB)')

# Show signal strength
svals = [v['s'] for v in tape.values()]
if svals:
    print(f'  Signal: min={min(svals):.2f} max={max(svals):.2f} mean={sum(svals)/len(svals):.2f}')
    print(f'  Strong signal (>1.0): {sum(1 for s in svals if s>1.0)}/{len(svals)} layers')
