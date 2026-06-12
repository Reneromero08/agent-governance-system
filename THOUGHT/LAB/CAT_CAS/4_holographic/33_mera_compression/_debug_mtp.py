import torch, importlib, sys
from collections import defaultdict
sys.path.insert(0, 'THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression')
mod = importlib.import_module('7_modular_compress')

cfg = mod.MODULES['llm']
print('LLM config:', cfg)

h = torch.load('THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_mtp_k128.holo', map_location='cpu', weights_only=True)

# Test matching logic on one key
import random
u_keys = [k for k in h if k.endswith('.U') and h[k].ndim == 2]
random.shuffle(u_keys)

for i, k in enumerate(u_keys[:5]):
    parts = k.split('.')
    print(f'\nKey: {k}')
    print(f'  Parts: {parts}')
    
    for tag in cfg['tag']:
        if tag in parts:
            idx = parts.index(tag)
            try: layer = int(parts[idx+1])
            except Exception: layer = None
            print(f'  Tag {tag}: idx={idx} layer={layer}')
            
            for t in cfg['types']:
                if t in parts or any(t in p for p in parts):
                    wt = '.'.join(parts[idx+2:-1])
                    print(f'  Type {t}: MATCHED -> wt={wt}')
                    break
            else:
                print(f'  NO type matched!')
            break
    else:
        print(f'  NO tag matched!')

# Full group
groups = mod.group_u_matrices(h, {'llm': cfg})
print(f'\nTotal groups: {len(groups)}')
for wt, tens in sorted(groups.items()):
    layers = sorted(tens.keys())
    sh = list(tens[layers[0]].shape)
    print(f'  {wt}: {len(layers)} layers, shape={sh}')
