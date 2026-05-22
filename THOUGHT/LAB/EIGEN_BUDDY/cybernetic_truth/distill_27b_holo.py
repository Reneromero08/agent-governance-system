"""
Out-of-Core Holographic Distillation for Qwen 27B
=================================================
This script reads the massive 27B safetensors model layer-by-layer,
applies Holographic SVD compression (dropping rank), and saves the
tiny resulting eigenvectors to a `.holo` file.
This guarantees minimal RAM usage.
"""

import os
import json
import torch
from safetensors import safe_open
from pathlib import Path

MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
OUTPUT_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_27b_k256.holo"

RANK_K = 256  # Massive rank drop to fit in 2GB-4GB total

def compress_tensor(tensor: torch.Tensor, k: int) -> tuple:
    """Compresses a 2D matrix into two smaller matrices via SVD."""
    if tensor.ndim != 2:
        return tensor  # Don't compress 1D biases or norms
    
    # Ensure float32 for stable SVD
    orig_dtype = tensor.dtype
    t_f32 = tensor.to(torch.float32)
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(t_f32, full_matrices=False)
    
    # Truncate
    k = min(k, U.size(1))
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    
    # Absorb S into Vh
    SVh_k = (S_k.unsqueeze(1) * Vh_k).to(orig_dtype)
    U_k = U_k.to(orig_dtype)
    
    return U_k, SVh_k

def main():
    print(f"Loading index from: {INDEX_PATH}")
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        index = json.load(f)
        
    weight_map = index.get("weight_map", {})
    
    holo_state_dict = {}
    
    # Identify unique files
    unique_files = list(set(weight_map.values()))
    
    # To be extremely memory efficient, we process file by file.
    # We will open one safetensor file, process all weights in it, and then close it.
    print(f"Found {len(unique_files)} safetensor files to process.")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    for i, file_name in enumerate(unique_files):
        file_path = os.path.join(MODEL_DIR, file_name)
        print(f"[{i+1}/{len(unique_files)}] Processing {file_name}...")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Skip vision or MTP if they exist to save space (we just want language model)
                if "vision" in key or "mtp" in key:
                    continue
                
                tensor = f.get_tensor(key)
                
                if tensor.ndim == 2:
                    # Compress
                    U_k, SVh_k = compress_tensor(tensor, RANK_K)
                    holo_state_dict[key + ".U"] = U_k.clone()
                    holo_state_dict[key + ".SVh"] = SVh_k.clone()
                    # Calculate savings
                    orig_size = tensor.numel() * tensor.element_size()
                    new_size = (U_k.numel() + SVh_k.numel()) * U_k.element_size()
                    print(f"  {key}: Compressed {(orig_size - new_size) / 1024**2:.1f} MB (to {new_size/1024**2:.1f} MB)")
                else:
                    holo_state_dict[key] = tensor.clone()
                    
    print(f"Saving holographic dict to {OUTPUT_PATH}...")
    torch.save(holo_state_dict, OUTPUT_PATH)
    
    final_size = os.path.getsize(OUTPUT_PATH) / 1024**3
    print(f"Distillation Complete! Final Holo Size: {final_size:.2f} GB")

if __name__ == "__main__":
    main()
