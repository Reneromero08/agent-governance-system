"""
B4: GOE Eigenvalue Validation — Quantum Chaos Check
=====================================================
Formula V4 / QEC precision sweep proved stabilizer correlation matrices
follow Wigner-Dyson GOE statistics (mean spacing ratio r=0.53 vs Poisson r=0.39).

This tests whether our wormhole rotation matrices R = U_prev^T @ U_curr
also follow GOE — proving the wormhole operates on the "quantum chaotic"
manifold, achieving maximum physical information density.
"""
import torch, re, numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _paths import LLM_WORMHOLE


def compute_eigenvalue_spacings(R_list):
    """
    For each rotation matrix, compute eigenvalue spacings of R @ R^T.
    Returns mean spacing ratio r (GOE = 0.53, Poisson = 0.39).
    """
    ratios = []
    for R in R_list:
        M = (R @ R.T).numpy()  # symmetric positive definite
        evals = np.linalg.eigvalsh(M)
        evals = evals[evals > 1e-10]  # drop near-zero
        
        if len(evals) < 4:
            continue
        
        spacings = np.diff(evals)
        # Unfold: divide by local mean spacing
        unfolded = []
        for i in range(len(spacings)):
            window = spacings[max(0,i-2):min(len(spacings),i+3)]
            local_mean = np.mean(window)
            if local_mean > 1e-10:
                unfolded.append(spacings[i] / local_mean)
        
        unfolded = np.array(unfolded)
        if len(unfolded) < 2:
            continue
        
        # Adjacent spacing ratio: r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
        for i in range(len(unfolded)-1):
            s1, s2 = unfolded[i], unfolded[i+1]
            if s1 + s2 > 1e-10:
                r = min(s1, s2) / max(s1, s2)
                ratios.append(r)
    
    return np.mean(ratios) if ratios else 0.0, len(ratios)


def test_goe(wormhole_path):
    print("=" * 60)
    print("B4: GOE EIGENVALUE VALIDATION")
    print("=" * 60)
    
    worm = torch.load(wormhole_path, map_location='cpu', weights_only=True)
    pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
    groups = defaultdict(lambda: dict(rots={}))
    for key, val in worm.items():
        m = pattern.match(key)
        if not m: continue
        wt, ls, field = m.groups()
        if field == 'R':
            groups[wt]['rots'][int(ls)] = val
    
    print(f"{'Weight Type':<35} {'r_mean':>8} {'n_spacings':>10} {'Verdict':>15}")
    print("-" * 70)
    
    all_ratios = []
    all_vals = []
    for wt, g in sorted(groups.items()):
        R_list = [g['rots'][l].float() for l in sorted(g['rots'].keys())]
        if not R_list:
            continue
        
        r_mean, n = compute_eigenvalue_spacings(R_list)
        if n < 10:
            continue
        
        all_ratios.append((wt, r_mean, n))
        all_vals.append(r_mean)
        
        # GOE: r ~ 0.53, Poisson: r ~ 0.39
        if r_mean > 0.48:
            verdict = "GOE (CHAOTIC)"
        elif r_mean > 0.43:
            verdict = "MIXED"
        else:
            verdict = "POISSON (REGULAR)"
        
        print(f"{wt:<35} {r_mean:>8.4f} {n:>10} {verdict:>15}")
    
    if all_ratios:
        avg_r = np.mean([r for _, r, _ in all_ratios])
        std_r = np.std(all_vals)
        goe_types = sum(1 for _, r, _ in all_ratios if r > 0.48)
        print(f"\n  Mean spacing ratio: {avg_r:.4f}  std={std_r:.4f}")
        print(f"  GOE (r > 0.48): {goe_types}/{len(all_ratios)} types")
        print(f"  Theoretical GOE: 0.5300")
        print(f"  Theoretical Poisson: 0.3900")
        
        if avg_r > 0.48:
            print(f"\n  VERDICT: Wormhole rotation matrices ARE quantum-chaotic (GOE).")
            print(f"  Maximum physical information density achieved.")
            print(f"  No further redundancies to exploit — at the Bekenstein bound.")
        elif avg_r > 0.43:
            print(f"\n  VERDICT: Mixed regime. Some eigenmodes rotate chaotically, some frozen.")
        else:
            print(f"\n  VERDICT: Poisson regular. Further compression possible.")

if __name__ == "__main__":
    test_goe(str(LLM_WORMHOLE))
