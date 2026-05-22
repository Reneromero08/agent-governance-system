"""
0.5B Holographic Distiller
==========================
Rapid prototyping script to test the holographic engine on the 0.5B model.
"""

import os
import torch
from safetensors import safe_open
from pathlib import Path

MODEL_FILE = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gemini_update\qwen_0.5b\model.safetensors"
OUTPUT_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_0_5b_k128.holo"

RANK_K = 128  # Even smaller rank for 0.5B

def compress_tensor(tensor: torch.Tensor, k: int) -> tuple:
    if tensor.ndim != 2:
        return tensor
    
    orig_dtype = tensor.dtype
    t_f32 = tensor.to(torch.float32)
    U, S, Vh = torch.linalg.svd(t_f32, full_matrices=False)
    
    k = min(k, U.size(1))
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    
    SVh_k = (S_k.unsqueeze(1) * Vh_k).to(orig_dtype)
    U_k = U_k.to(orig_dtype)
    return U_k, SVh_k

def main():
    print(f"[*] Compressing {MODEL_FILE}")
    holo_state_dict = {}
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with safe_open(MODEL_FILE, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"[*] Found {len(keys)} tensors.")
        
        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            if tensor.ndim == 2:
                U_k, SVh_k = compress_tensor(tensor, RANK_K)
                holo_state_dict[key + ".U"] = U_k.clone()
                holo_state_dict[key + ".SVh"] = SVh_k.clone()
            else:
                holo_state_dict[key] = tensor.clone()
                
            if i % 50 == 0:
                print(f"  Processed {i}/{len(keys)}")
                
    print(f"[*] Saving holographic dict to {OUTPUT_PATH}...")
    torch.save(holo_state_dict, OUTPUT_PATH)
    
    final_size = os.path.getsize(OUTPUT_PATH) / 1024**2
    print(f"[*] Distillation Complete! Final Holo Size: {final_size:.2f} MB")

if __name__ == "__main__":
    main()
