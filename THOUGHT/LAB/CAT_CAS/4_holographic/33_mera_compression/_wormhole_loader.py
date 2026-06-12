"""Wormhole .holo Loader — reconstructs U from rotations + SVh from catalytic."""
import torch, math, os
from collections import defaultdict

def load_wormhole_holo(wormhole_path, catalytic_path=None):
    """Load wormhole-compressed .holo, supplementing SVh from catalytic if needed."""
    wh = torch.load(wormhole_path, weights_only=False, map_location='cpu')
    
    if catalytic_path and os.path.exists(catalytic_path):
        cat = torch.load(catalytic_path, weights_only=False, map_location='cpu')
        print(f"  Loaded catalytic supplement: {len(cat)} keys")
    else:
        cat = {}
    
    # Parse wormhole keys
    u_groups = defaultdict(dict)  # {wt: {layer: U or (R, res)}}
    svh_keys = {}  # {wt: SVh_tensor} — shared SVh per weight type
    
    for key, val in wh.items():
        parts = key.split('.')
        if key.endswith('.U'):
            # First layer U
            wt = '.'.join(parts[:-2])  # e.g., mlp.down_proj.weight
            layer = int(parts[-2][1:])  # L0 -> 0
            u_groups[wt][layer] = ('U', val.float())
        elif key.endswith('.R'):
            wt = '.'.join(parts[:-2])
            layer = int(parts[-2][1:])
            res_idx_key = key.replace('.R', '.res_idx')
            res_max_key = key.replace('.R', '.res_max')
            res_idx = wh.get(res_idx_key)
            res_max = wh.get(res_max_key)
            u_groups[wt][layer] = ('R', val.float(), res_idx, res_max)
    
    # Collect SVh from catalytic
    for key, val in cat.items():
        if key.endswith('.SVh'):
            # Match weight type
            for wt in u_groups:
                if key.startswith(wt):
                    # Store first SVh found
                    if wt not in svh_keys:
                        svh_keys[wt] = val.float()
                    break
    
    # Reconstruct full .holo dict
    reconstructed = {}
    stats = []
    
    for wt, layers in sorted(u_groups.items()):
        sorted_l = sorted(layers.keys())
        prev_U = None
        
        for l in sorted_l:
            data = layers[l]
            if data[0] == 'U':
                U = data[1]
                prev_U = U
            else:
                # Reconstruct: U_curr = U_prev @ R + residual
                R = data[1]
                res_idx = data[2]
                res_max = data[3]
                U_rot = prev_U @ R
                
                if res_idx is not None and res_max is not None:
                    # Dequantize 2-bit residual
                    levels = torch.tensor([-1.0, -0.333, 0.333, 1.0]) * max(res_max.item(), 1e-6)
                    residual_quant = levels[res_idx.long()]
                    U = U_rot + residual_quant
                else:
                    U = U_rot
                
                prev_U = U  # use RECONSTRUCTED U for next rotation
            
            # Full key reconstruction
            # Wormhole format: mlp.down_proj.weight.L0.U
            # Original format: model.language_model.layers.0.mlp.down_proj.weight.U
            # We need to map wormhole names to original .holo names
            
            # For loading into HoloLinear patcher, we need the original key format
            # The catalytic SVh has keys like: model.language_model.layers.0.mlp.down_proj.weight.SVh
            # We need to match U to SVh
            
            # Store with a key that patch_model_with_holo can find
            # The patcher looks for name + ".weight.U" in the dict
            # We need the weight type + layer number in the ORIGINAL format
            
            # For now, store all U matrices. The SVh comes from catalytic.
            orig_key = f"{wt}.U.L{l}"  # temp key
            reconstructed[orig_key] = U.half()
            
        stats.append(f"{wt}: {len(sorted_l)} layers, U shapes: {list(layers.values())[0][1].shape if 'U' in [x[0] for x in layers.values()] else 'rotations'}")
    
    # Copy SVh from catalytic
    for wt, svh in svh_keys.items():
        reconstructed[f"{wt}.SVh"] = svh.half()
    
    print(f"  Reconstructed: {len(reconstructed)} entries")
    for s in stats[:10]: print(f"    {s}")
    
    return reconstructed, svh_keys

if __name__ == "__main__":
    wormhole = r"THOUGHT\LAB\CAT_CAS\4_holographic\33_mera_compression\qwen_27b_wormhole.holo"
    catalytic = r"THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_27b_catalytic_k256.holo"
    rec, svh = load_wormhole_holo(wormhole, catalytic)
