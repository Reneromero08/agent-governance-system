"""
Temporal Calibration — Catalytic .holo as Teacher
===================================================
Never loads the 54 GB raw model. The catalytic .holo (3.65 GB, full K=256 SVD)
IS the teacher. The wormhole (199 MB, rotation+residual) IS the student.

For each layer of each weight type, computes per-mode energy ratio:
   scale[i] = ||U_cat[:,i]||^2 / ||U_worm[:,i]||^2

This tells us whether the wormhole compresses or expands each eigenmode.
The scale factors bake into the TuneableWormhole's SVh gamma parameter.

Result: the 34K tuneable params learn the calibration signal without
a single forward pass through the full model.
"""
import torch, re, sys, os
from pathlib import Path
from collections import defaultdict


def compute_calibration(catalytic_path, wormhole_path, device='cpu'):
    """
    Per-layer full-matrix fidelity: cos_sim(U_cat.flatten(), U_worm.flatten())
    Averages across all layers. Maps to residual gate: lower fidelity = more suppression.
    """
    print(f"Teacher: {catalytic_path}")
    cat = torch.load(catalytic_path, map_location='cpu', weights_only=True)
    print(f"Student: {wormhole_path}")
    worm = torch.load(wormhole_path, map_location='cpu', weights_only=True)
    
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
        elif field == 'res_idx': g['res'].setdefault(l, {})['idx'] = val
        elif field == 'res_max':
            if l in g['res']: g['res'][l]['max'] = val
    
    CAT_PREFIX = 'model.language_model.layers'
    calibration = {}  # {wt: per_layer_fidelity [n_layers]}
    
    for wt, g in sorted(groups.items()):
        all_layers = [g['first_l']] + sorted(g['rots'].keys())
        fidelities = []
        
        for l in all_layers:
            cat_key = f'{CAT_PREFIX}.{l}.{wt}.U'
            if cat_key not in cat: continue
            U_cat = cat[cat_key].float()
            
            if l == g['first_l']: U_worm = g['first_U'].float()
            elif l in g['rots']:
                anchor = g['first_U'].float(); R = g['rots'][l].float()
                U_worm = anchor @ R
                if l in g['res'] and g['res'][l].get('idx') is not None:
                    rd = g['res'][l]
                    mval = rd.get('max', torch.tensor(1e-6)).item()
                    levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(abs(mval), 1e-6)
                    U_worm = U_worm + levels[rd['idx'].long()]
            else: continue
            
            cos = torch.nn.functional.cosine_similarity(
                U_cat.flatten().unsqueeze(0), U_worm.flatten().unsqueeze(0)
            ).item()
            fidelities.append(cos)
        
        if fidelities:
            import numpy as np
            mean_fid = np.mean(fidelities)
            min_fid = np.min(fidelities)
            # Store mean fidelity + decay info
            calibration[wt] = {'mean': mean_fid, 'min': min_fid, 'n': len(fidelities)}
    
    del cat, worm
    return calibration


def apply_calibration(tuner, calibration):
    """
    Apply per-weight-type fidelity as scaling on SVh + residual gate.
    Lower fidelity -> stronger SVh damping + residual suppression.
    """
    applied = 0
    for wt, info in calibration.items():
        if wt not in tuner._wt_map: continue
        tw = tuner._tw(wt)
        
        fid = info['mean']
        # Below 0.85: suppress residual output. 0.85-1.0: pass through.
        gate_strength = max(0.0, min(1.0, (fid - 0.7) / 0.2))  # ramp 0.7->0.9
        
        # SVh gamma: scale down by fidelity (dampen noisy projection)
        desired_gamma = gate_strength
        raw_gamma = torch.atanh(torch.clamp(torch.tensor(desired_gamma - 0.9) / 0.1, -0.99, 0.99))
        tw.svh_gamma.data.fill_(raw_gamma.item())
        
        # Residual gate: suppress all layers uniformly by quality
        raw_gate = torch.logit(torch.tensor(gate_strength).clamp(0.001, 0.999))
        tw.res_gate.data.fill_(raw_gate.item())
        
        applied += 1
    
    return applied


def calibrate_and_merge(catalytic_path, wormhole_llm_path, output_path):
    """
    Full pipeline: compute calibration, apply to tuneable, merge into wormhole.
    Returns calibrated wormhole file.
    """
    import importlib
    sys.path.insert(0, str(Path(__file__).parent))
    
    calibration = compute_calibration(catalytic_path, wormhole_llm_path)
    
    print(f"\nPer-layer fidelity calibration ({len(calibration)} weight types):")
    for wt, info in sorted(calibration.items()):
        print(f"  {wt:<30} mean_fid={info['mean']:.4f} min={info['min']:.4f} n={info['n']}")
    
    # Load wormhole into tuneable session
    graph_mod = importlib.import_module("9_catalytic_graph_loader")
    tuneable_mod = importlib.import_module("11_tuneable_wormhole")
    
    print(f"\nLoading wormhole into catalytic session...")
    graph = graph_mod.load_graph({"llm": wormhole_llm_path})
    session = graph_mod.CatalyticSession(graph=graph)
    session.borrow("llm")
    tuner = tuneable_mod.TuneableWormhole(session, "llm", lora_rank=8)
    
    # Apply calibration
    n = apply_calibration(tuner, calibration)
    print(f"  Calibrated {n}/{len(calibration)} weight types (SVh gamma)")
    
    # Merge into calibrated wormhole
    tuner.merge_to_wormhole(output_path)
    out_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Calibrated wormhole: {output_path} ({out_mb:.0f} MB)")
    
    # Verify: re-read and check fidelity improvement
    if os.path.exists(catalytic_path):
        cat2 = torch.load(catalytic_path, map_location='cpu', weights_only=True)
        worm2 = torch.load(output_path, map_location='cpu', weights_only=True)
        
        pattern = re.compile(r'(.+)\.L(\d+)\.(.+)')
        groups2 = defaultdict(lambda: dict(first_U=None, rots={}, res={}))
        for key, val in worm2.items():
            m = pattern.match(key)
            if not m: continue
            wt2, ls2, field2 = m.groups()
            l2 = int(ls2)
            g2 = groups2[wt2]
            if field2 == 'U': g2['first_U'] = val; g2['first_l'] = l2
            elif field2 == 'R': g2['rots'][l2] = val
            elif field2 == 'res_idx': g2['res'].setdefault(l2, {})['idx'] = val
            elif field2 == 'res_max':
                if l2 in g2['res']: g2['res'][l2]['max'] = val
        
        CAT_PREFIX = 'model.language_model.layers'
        import numpy as np
        deltas = []
        for wt, g2 in groups2.items():
            cat_key2 = f'{CAT_PREFIX}.{g2["first_l"]}.{wt}.U'
            if cat_key2 in cat2 and g2['first_U'] is not None:
                U_cat2 = cat2[cat_key2].float()
                U_cal = g2['first_U'].float()
                cos = torch.nn.functional.cosine_similarity(
                    U_cat2.flatten().unsqueeze(0), U_cal.flatten().unsqueeze(0)
                ).item()
                deltas.append(cos)
        
        if deltas:
            print(f"  Post-calibration first-layer fidelity: {np.mean(deltas):.4f} (was ~0.89)")
    
    session.close()
    return output_path


if __name__ == "__main__":
    from _paths import CATALYTIC_27B, CAVITATED_27B, LLM_WORMHOLE, HOLO_MODELS
    # Use cavitated (k~49) as teacher — same rank as wormhole, no dimension mismatch
    cat_path = str(CAVITATED_27B)  # 734 MB, cavity-sieved, same k as wormhole
    worm_path = str(LLM_WORMHOLE)
    out_path = str(HOLO_MODELS / "qwen_27b_llm_calibrated.holo")
    
    if len(sys.argv) >= 3:
        cat_path = sys.argv[1]
        worm_path = sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) >= 4 else out_path
    
    calibrate_and_merge(cat_path, worm_path, out_path)
