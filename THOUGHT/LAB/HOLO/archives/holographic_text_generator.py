"""
HOLO 4.4: Holographic Text Generator
====================================
Loads Qwen 0.5B via HuggingFace, intercepts the discrete weights,
maps them to the Unit Circle phase plane, extracts the top K geometric waves,
reconstructs the weights from pure continuous phase, and generates text.
"""

import sys, time, math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = str(REPO / "THOUGHT" / "LAB" / "CAT_CAS" / "3_physics_complexity" / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b")

def holographic_compress_tensor(weight_tensor, k=128):
    """
    Compresses a PyTorch weight tensor via Optical Phase Grating SVD.
    """
    device = weight_tensor.device
    dtype = weight_tensor.dtype
    
    # 1. Cast to float32 for math
    W = weight_tensor.float()
    
    # 2. Phase Normalization
    max_val = torch.max(torch.abs(W)) + 1e-9
    phase_angles = (W / max_val) * math.pi
    
    # 3. Map to Unit Circle (Complex)
    # e^(i * theta) = cos(theta) + i*sin(theta)
    grating = torch.complex(torch.cos(phase_angles), torch.sin(phase_angles))
    
    # 4. Topological Extraction (SVD over C)
    U, S, Vh = torch.linalg.svd(grating, full_matrices=False)
    
    # 5. Truncate to top K fundamental frequencies
    k_actual = min(k, U.shape[1])
    U_k = U[:, :k_actual]
    S_k = S[:k_actual]
    Vh_k = Vh[:k_actual, :]
    
    # 6. Reconstruct geometric wave
    grating_recon = (U_k * S_k.unsqueeze(0)) @ Vh_k
    
    # 7. Map back to discrete parameter space
    recon_angles = torch.angle(grating_recon)
    W_recon = (recon_angles / math.pi) * max_val
    
    # Calculate compression
    params_orig = W.numel()
    params_holo = k_actual * W.shape[0] + k_actual * W.shape[1] + k_actual
    compression = params_orig / params_holo
    
    return W_recon.to(dtype=dtype, device=device), compression, params_orig, params_holo

def apply_holographic_distillation(model, k=128):
    """
    Walks the model architecture and replaces every Linear layer's weights
    with the Holographically reconstructed weights.
    """
    total_params_orig = 0
    total_params_holo = 0
    
    print(f"[*] Commencing Holographic Wave Extraction (K={k})...")
    start = time.time()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only compress if the matrix is large enough
            if module.weight.shape[0] <= k or module.weight.shape[1] <= k:
                continue
                
            W_recon, comp, p_orig, p_holo = holographic_compress_tensor(module.weight.data, k)
            module.weight.data = W_recon
            
            total_params_orig += p_orig
            total_params_holo += p_holo
            
    print(f"[+] Distillation Complete in {time.time()-start:.2f}s")
    print(f"    Original Discrete Parameters:  {total_params_orig:,}")
    print(f"    Holographic Wave Parameters:   {total_params_holo:,}")
    print(f"    Total Structural Compression:  {total_params_orig / max(1, total_params_holo):.2f}x")
    print()

def main():
    print("=" * 78)
    print("HOLO 4.4: HOLOGRAPHIC TEXT GENERATOR")
    print("  Distilling Intelligence into Continuous Wave Topologies")
    print("=" * 78)
    
    print("[1] Loading Qwen 0.5B...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, device_map="cpu")
    except Exception as e:
        print(f"[-] Failed to load model from {MODEL_DIR}: {e}")
        return
        
    print("[2] Distilling Model into Optical Phase Waves...")
    # Using K=128 as recommended for reliable testing.
    # Qwen 0.5B hidden_size=896. K=128 gives roughly 3.5x compression per matrix.
    apply_holographic_distillation(model, k=128)
    
    print("[3] Testing Holographic Intelligence...")
    prompts = [
        "The holographic computing paradigm demonstrates that",
        "If you compress a neural network into an optical wave, the result is",
    ]
    
    for p in prompts:
        print("-" * 78)
        print(f"Prompt: {p}")
        inputs = tokenizer(p, return_tensors="pt")
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print(f"Speed: {(len(outputs[0]) - inputs.input_ids.shape[1]) / elapsed:.2f} tok/s")
    
    print("=" * 78)

if __name__ == "__main__":
    main()
