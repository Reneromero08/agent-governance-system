import torch
cat = torch.load("THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo", map_location="cpu", weights_only=True)
svh = [k for k in cat if k.endswith(".SVh")]

types = {}
for k in svh:
    parts = k.split(".")
    try:
        idx = parts.index("layers")
        wt = ".".join(parts[idx+2:-1])
    except ValueError:
        wt = k
    if wt not in types:
        types[wt] = {"count": 0, "shape": list(cat[k].shape)}
    types[wt]["count"] += 1

print(f"SVh: {len(svh)} keys total")
print(f"Unique types: {len(types)}")
for wt, info in sorted(types.items()):
    mb = info["count"] * info["shape"][0] * info["shape"][1] * 2 / 1024 / 1024
    print(f"  {wt}: {info['count']}x {info['shape']} = {mb:.0f} MB total ({mb/info['count']:.1f} MB each)")
print(f"  Shared SVh (1/type): {sum(info['shape'][0]*info['shape'][1]*2/1024/1024 for info in types.values()):.0f} MB")
print(f"  Per-layer SVh: {sum(info['count']*info['shape'][0]*info['shape'][1]*2/1024/1024 for info in types.values()):.0f} MB")
