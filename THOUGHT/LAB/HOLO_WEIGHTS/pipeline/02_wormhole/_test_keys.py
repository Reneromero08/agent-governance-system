import torch, re
d = torch.load('THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/qwen_27b_wormhole_v2.holo', map_location='cpu', weights_only=False)
keys = sorted(d.keys())

# Show all keys with .L0.U (layer 0 base U)
l0 = [k for k in keys if '.L0.U' in k]
print("=== L0.U keys ===")
for k in l0:
    print(f"  {k}: {list(d[k].shape)}")

# Show all keys with L that are NOT L0
ln = [k for k in keys if '.L' in k and '.L0.U' not in k]
print(f"\n=== Other L keys (first 5) ===")
for k in ln[:5]:
    print(f"  {k}: {list(d[k].shape)}")

# Parse to understand wt_base structure
print("\n=== Parsing L0.U keys ===")
for k in l0:
    parts = k.split('.L')
    wt_base = parts[0]
    print(f"  key={k}")
    print(f"    parts[0]='{wt_base}'")
    print(f"    svh_key = '{wt_base}'")
    print(f"    in svh_dict? {wt_base in d}")

# Check SVh keys that match
print("\n=== Matching .L0.U wt_base with .SVh ===")
for k in l0:
    wt_base = k.split('.L')[0]
    svh_candidate = wt_base + '.SVh'
    # Try without trailing dot
    if wt_base.endswith('.'):
        wt_base_nodot = wt_base[:-1]
        svh_candidate2 = wt_base_nodot + '.SVh'
        print(f"  wt_base='{wt_base}'")
        print(f"    svh_candidate='{svh_candidate}' exists={svh_candidate in d}")
        print(f"    svh_candidate2='{svh_candidate2}' exists={svh_candidate2 in d}")
    else:
        print(f"  wt_base='{wt_base}'  svh='{svh_candidate}' exists={svh_candidate in d}")

# Show all .SVh keys that start with self_attn or mlp
svh_self = [k for k in keys if k.endswith('.SVh') and ('self_attn' in k or 'mlp' in k or 'linear_attn' in k)]
print(f"\n=== Relevant SVh keys ===")
for k in svh_self:
    print(f"  {k}")

del d
