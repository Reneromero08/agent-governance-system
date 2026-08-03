"""
B5: Living Formula Pre-Compression Quality Predictor
=====================================================
Predicts wormhole compression quality BEFORE running full compress.
Uses the Living Formula R = (E/nabla_S) * sigma^D_f with quick
sampled metrics from a few layers.

If R > 0.7: COMPRESS (high confidence)
If 0.3 < R < 0.7: CONSERVATIVE (use more residual bits)
If R < 0.3: SKIP (silence protocol — fall back to as-is storage)

Run on catalytic .holo BEFORE cavity sieve or wormhole.
"""
import torch, re, numpy as np
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _paths import CATALYTIC_27B, LLM_WORMHOLE


def predict_compression_quality(holo_path, sample_layers=5):
    """
    Living Formula pre-compression predictor.
    
    For each weight type, samples `sample_layers` to estimate:
      E      = signal energy (mean anchor singular value / total)
      nabla_S = cross-layer entropy gradient (std of subspace overlap decay)
      sigma   = predicted fidelity (cosine preservation across samples)
      D_f     = independent rotation blocks (estimated)
    
    Returns: {wt: {'R': predicted_resonance, 'verdict': 'COMPRESS'|'CONSERVATIVE'|'SKIP'}}
    """
    print("=" * 65)
    print("B5: LIVING FORMULA PRE-COMPRESSION PREDICTOR")
    print("R = (E / nabla_S) * sigma^D_f")
    print("=" * 65)
    
    holo = torch.load(holo_path, map_location='cpu', weights_only=True)
    
    # Group U matrices by weight type
    groups = defaultdict(dict)
    for key, val in holo.items():
        parts = key.split('.')
        if not key.endswith('.U') or val.ndim != 2:
            continue
        try:
            idx = parts.index('layers')
            l = int(parts[idx + 1])
            wt = '.'.join(parts[idx + 2:-1])
        except (ValueError, IndexError):
            continue
        groups[wt][l] = val.float()
    
    predictions = {}
    
    for wt, tensors in sorted(groups.items()):
        sorted_l = sorted(tensors.keys())
        if len(sorted_l) < 2:
            continue
        
        n_sample = min(sample_layers, len(sorted_l))
        sampled = sorted_l[:n_sample]
        
        # E: signal energy = mean(||U_i||^2) / (m * k)
        energies = [(tensors[l] * tensors[l]).sum().item() for l in sampled]
        m, k = tensors[sampled[0]].shape
        E = np.mean(energies) / (m * k)
        
        # nabla_S: cross-layer entropy gradient
        # Measure how fast the subspace rotates across layers
        U0 = tensors[sampled[0]]
        subspace_overlaps = []
        for l in sampled[1:]:
            Ul = tensors[l]
            # Quick subspace overlap via Frobenius projection
            proj = (U0.T @ Ul).norm().item() / (U0.norm().item() * Ul.norm().item() + 1e-9)
            subspace_overlaps.append(proj)
        
        nabla_S = np.std(subspace_overlaps) if subspace_overlaps else 1.0
        # Prevent division by zero
        nabla_S = max(nabla_S, 1e-6)
        
        # sigma: predicted fidelity factor
        sigma = np.mean(subspace_overlaps) if subspace_overlaps else 0.5
        sigma = max(sigma, 0.01)
        
        # D_f: estimated independent rotation blocks
        threshold_crosses = 0
        running_mean = subspace_overlaps[0] if subspace_overlaps else 1.0
        for ov in subspace_overlaps[1:]:
            if abs(ov - running_mean) / max(running_mean, 1e-6) > 0.3:
                threshold_crosses += 1
            running_mean = 0.9 * running_mean + 0.1 * ov
        
        # D_f: effective independent blocks (corrected)
        # Formula V4: D_f = t = floor((d-1)/2) — not raw layer count
        D_f = max(1, len(sorted_l) // max(1, threshold_crosses + 1))
        
        # Living Formula
        R = (E / nabla_S) * (sigma ** D_f)
        
        # Verdict — adjusted thresholds for compression prediction
        if R > 1.0:
            verdict = "COMPRESS"
        elif R > 0.1:
            verdict = "CONSERVATIVE"
        else:
            verdict = "SKIP"
        
        predictions[wt] = {
            'E': E, 'nabla_S': nabla_S, 'sigma': sigma, 'D_f': D_f,
            'R': R, 'verdict': verdict, 'layers': len(sorted_l)
        }
    
    # Print results
    print(f"{'Weight Type':<35} {'E':>8} {'grad':>8} {'sigma':>8} {'D_f':>6} {'R':>8} {'Verdict':>14}")
    print("-" * 85)
    for wt, p in sorted(predictions.items()):
        print(f"{wt:<35} {p['E']:>8.4f} {p['nabla_S']:>8.4f} {p['sigma']:>8.4f} "
              f"{p['D_f']:>6} {p['R']:>8.4f} {p['verdict']:>14}")
    
    compress = sum(1 for p in predictions.values() if p['verdict'] == 'COMPRESS')
    conservative = sum(1 for p in predictions.values() if p['verdict'] == 'CONSERVATIVE')
    skip = sum(1 for p in predictions.values() if p['verdict'] == 'SKIP')
    
    print(f"\n  COMPRESS:     {compress}")
    print(f"  CONSERVATIVE: {conservative}")
    print(f"  SKIP:         {skip}")
    print(f"  Total types:  {len(predictions)}")
    
    # Validate against actual wormhole fidelity if available
    worm_path = str(LLM_WORMHOLE) if LLM_WORMHOLE.exists() else None
    if worm_path and Path(worm_path).exists():
        print(f"\n  Validation against actual wormhole fidelity:")
        worm = torch.load(worm_path, map_location='cpu', weights_only=True)
        w_pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
        w_groups = defaultdict(lambda: dict(first_U=None, rots={}))
        for key, val in worm.items():
            m = w_pattern.match(key)
            if not m: continue
            wt, ls, field = m.groups()
            if field == 'U': w_groups[wt]['first_U'] = val; w_groups[wt]['first_l'] = int(ls)
            elif field == 'R': w_groups[wt]['rots'][int(ls)] = val
        
        for wt, p in sorted(predictions.items()):
            if wt in w_groups:
                g = w_groups[wt]
                anchor = g['first_U'].float()
                fids = []
                for l in sorted(g['rots'].keys())[:5]:
                    U_recon = anchor @ g['rots'][l].float()
                    cos = torch.nn.functional.cosine_similarity(
                        anchor.flatten().unsqueeze(0), U_recon.flatten().unsqueeze(0)).item()
                    fids.append(cos)
                actual_fid = np.mean(fids)
                match = "OK" if (p['verdict'] == 'COMPRESS' and actual_fid > 0.5) or \
                              (p['verdict'] == 'SKIP' and actual_fid < 0.3) else "MISMATCH"
                print(f"    {wt:<35} pred_R={p['R']:.4f} actual_fid={actual_fid:.4f} [{match}]")
    
    return predictions


if __name__ == "__main__":
    predict_compression_quality(str(CATALYTIC_27B))
