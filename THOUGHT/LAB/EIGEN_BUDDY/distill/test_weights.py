"""Test: load distilled .holo, verify structure, check Eigen Buddy compatibility."""
import sys, json, numpy as np, torch
sys.path.insert(0, r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY')
from core.attention import MultiHeadComplexAttention

meta = json.load(open(r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\distill\distilled\eigenbuddy_qwen27b.json'))
print(f"Loaded {len(meta)} weight matrices from .holo")

# Show top entries
by_D_pr = sorted(meta.items(), key=lambda x: -x[1]['D_pr'])
print("\nTop 10 by D_pr (participation dimension = signal rank):")
for name, v in by_D_pr[:10]:
    print(f"  {name:60s} k={v['k']:>3} dim={v['dim']:>6} D_pr={v['D_pr']:.1f}")

# Dimension analysis
dims = sorted(set(v['dim'] for v in meta.values()))
ks = sorted(set(v['k'] for v in meta.values()))
D_prs = [v['D_pr'] for v in meta.values()]
print(f"\nDimension ranges: dim={dims} k={ks}")
print(f"D_pr range: {min(D_prs):.1f} - {max(D_prs):.1f}  mean={np.mean(D_prs):.1f}")

# Compatibility check
print(f"\n=== COMPATIBILITY CHECK ===")
print(f"Distilled weights: dim ~ {dims[0]}-{dims[-1]}")
print(f"Eigen Buddy attention: d_model = 16-32 (configurable)")
print(f"k = {ks[0]} (number of eigenmodes kept)")

if max(dims) <= 64:
    print("COMPATIBLE: dimensions fit within Eigen Buddy's range")
elif max(dims) <= 512:
    print("ADAPTABLE: can use top eigenmodes with dimension reduction")
else:
    print(f"GAP: distilled dim={max(dims)} >> Eigen Buddy d_model=32")
    print(f"  Fix: use only the top D_pr eigenmodes (D_pr mean={np.mean(D_prs):.1f})")
    print(f"  Or: scale Eigen Buddy d_model to match distilled dims")
    print(f"  Or: project distilled eigenvectors to target dimension via interpolation")

# Test: can we load the phase grating and use it as attention weight init?
print(f"\n=== WEIGHT INJECTION TEST ===")
# Pick a matrix with reasonable dimensions
for name, v in by_D_pr:
    if v['dim'] <= 128 and v['k'] <= 64:
        print(f"Candidate: {name} dim={v['dim']} k={v['k']}")
        break
else:
    print(f"All matrices have dim > 128. Need dimension reduction.")
    # Show the smallest
    smallest = min(meta.items(), key=lambda x: x[1]['dim'])
    print(f"Smallest: {smallest[0]} dim={smallest[1]['dim']}")
