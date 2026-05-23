"""
v2 .holo Loader — Reconstructs U + SVh from deduplicated INT8 format.

Format:
  _svh: {wt: int8_tensor}          — shared SVh per weight type
  _svh_scales: {wt: float}          — dequant scales
  _svh_ref: {key.U: wt}            — which SVh each U belongs to
  _k: int                           — eigenbasis rank
  key.U: int8_tensor                — quantized U
  key.scale: float                  — dequant scale
"""
import torch

def load_holo_v2(path, map_location='cpu'):
    """Load v2 .holo and reconstruct full U + SVh tensors."""
    d = torch.load(path, weights_only=False, map_location=map_location)
    
    if '_format' not in d or d['_format'] != 'int8_dedup':
        # Not v2 format, return as-is
        return d
    
    svh_shared = d['_svh']
    svh_scales = d['_svh_scales']
    svh_ref = d.get('_svh_ref', {})
    k = d.get('_k', 128)
    
    result = {}
    
    # Dequantize shared SVh
    svh_dequant = {}
    for wt in svh_shared:
        svh_dequant[wt] = svh_shared[wt].float() * svh_scales[wt]
    
    # Reconstruct per-key tensors
    for key, val in d.items():
        if key.startswith('_'): continue
        
        if key.endswith('.U'):
            scale_key = key.replace('.U', '.scale')
            scale = d.get(scale_key, 1.0)
            result[key] = val.float() * scale  # dequantize U
            
            # Attach corresponding SVh
            wt = svh_ref.get(key, None)
            if wt is None:
                # Infer from key name
                parts = key.rsplit('.', 1)[0].split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i+1 < len(parts):
                        for j in range(i+2, len(parts)):
                            if parts[j] in ('w1','w2','w3'):
                                wt = '.'.join(parts[i:j+1])
                                break
            if wt and wt in svh_dequant:
                svh_key = key.replace('.U', '.SVh')
                result[svh_key] = svh_dequant[wt].half()
            elif wt and wt.replace('.weight.weight', '.weight') in svh_dequant:
                svh_key = key.replace('.U', '.SVh')
                result[svh_key] = svh_dequant[wt.replace('.weight.weight', '.weight')].half()
        
        elif key.endswith('.SVh'):
            pass  # handled by reference
        elif '.scale' in key:
            pass  # metadata
    
    result['_k'] = k
    result['_format'] = 'reconstructed'
    return result

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "NOT FOUND"
    d = load_holo_v2(path)
    u_keys = [k for k in d if k.endswith('.U')]
    svh_keys = [k for k in d if k.endswith('.SVh')]
    size_mb = sum(v.numel() * v.element_size() for v in d.values() if hasattr(v, 'numel')) / 1024**2
    print(f"Loaded: {len(u_keys)} U tensors, {len(svh_keys)} SVh tensors, {size_mb:.0f} MB reconstructed")
