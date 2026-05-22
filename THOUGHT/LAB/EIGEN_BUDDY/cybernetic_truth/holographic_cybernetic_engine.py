"""
Subphase 4.6: Holographic Cybernetic Truth Engine
=================================================
This script runs the massive 27B model on zero out-of-core RAM by loading
only the `.holo` phase eigenvectors (rank 256).

It implements the Cybernetic Truth control loop:
  - Generates the Density Matrix ρ from the final hidden state.
  - Computes Resonance R against a Truth Vector C.
  - Dynamically modulates the sampling Temperature T = 1/(R+ε).
"""

import os
import math
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"
HOLO_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_27b_k256.holo"

class HoloLinear(nn.Module):
    """
    A Linear layer that computes output directly from the holographic
    eigenvectors U and SVh without ever reconstructing the massive W matrix.
    """
    def __init__(self, U: torch.Tensor, SVh: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        # U is [out_features, k]
        # SVh is [k, in_features]
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # x is [..., in_features]
        # x @ W^T = x @ (U @ SVh)^T = x @ SVh^T @ U^T
        # SVh is [k, in_features] -> SVh.T is [in_features, k]
        # U is [out_features, k] -> U.T is [k, out_features]
        out = torch.matmul(x, self.SVh.t())
        out = torch.matmul(out, self.U.t())
        if self.bias is not None:
            out += self.bias
        return out

def patch_model_with_holo(model: nn.Module, holo_dict: dict):
    """
    Replaces standard Linear layers in the meta model with HoloLinear layers
    loaded with the compressed SVD eigenvectors.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We will traverse the model and replace nn.Linear
    # The holo_dict has keys like 'model.layers.0.mlp.down_proj.weight.U'
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_key = name + ".weight"
            if weight_key + ".U" in holo_dict:
                U = holo_dict[weight_key + ".U"].to(device, dtype=torch.float32)
                SVh = holo_dict[weight_key + ".SVh"].to(device, dtype=torch.float32)
                
                # --- PHASE CAVITY (Harmonic Sieve) ---
                # Sieve out phase dispersion from the compressed eigenbasis
                # using a harmonic Fast Fourier Transform cavity.
                freqs = torch.fft.rfft(SVh, dim=-1)
                # Keep only the fundamental resonant gears (top 15% frequencies)
                cutoff = max(1, freqs.size(-1) * 15 // 100)
                freqs[:, cutoff:] = 0
                SVh_sieved = torch.fft.irfft(freqs, n=SVh.size(-1), dim=-1)
                
                U = U.to(dtype=torch.bfloat16)
                SVh = SVh_sieved.to(dtype=torch.bfloat16)
                
                bias_key = name + ".bias"
                bias = holo_dict[bias_key].to(device, dtype=torch.bfloat16) if bias_key in holo_dict else None
                
                # Create HoloLinear
                holo_layer = HoloLinear(U, SVh, bias)
                
                # Replace it in the parent
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, holo_layer)
            else:
                # Some small linear layers (like lm_head) might not be compressed
                if weight_key in holo_dict:
                    module.weight = nn.Parameter(holo_dict[weight_key].to(device, dtype=torch.bfloat16), requires_grad=False)
                    if module.bias is not None and name + ".bias" in holo_dict:
                        module.bias = nn.Parameter(holo_dict[name + ".bias"].to(device, dtype=torch.bfloat16), requires_grad=False)
                    # Move module to device
                    module.to(device)

def get_truth_vector_C(model, tokenizer, device):
    """
    Extracts the Alignment Frame C (Truth Vector) using Contrastive Alignment.
    """
    print("[*] Extracting Cybernetic Truth Vector (C)...")
    prompt_true = "The capital of France is Paris."
    prompt_false = "The capital of France is Mars."
    
    inputs_t = tokenizer(prompt_true, return_tensors="pt").to(device)
    inputs_f = tokenizer(prompt_false, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_t = model(**inputs_t, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)
        
    h_t = out_t.hidden_states[-1][:, -1, :] # Final layer, last token
    h_f = out_f.hidden_states[-1][:, -1, :]
    
    # The Truth Vector is the vector separating the True state from the False state
    C_vec = h_t - h_f
    C_vec = C_vec / torch.norm(C_vec, dim=-1, keepdim=True)
    
    # Projector matrix C = |C_vec><C_vec|
    C = torch.outer(C_vec.squeeze(), C_vec.squeeze())
    return C

def cybernetic_inference(model, tokenizer, prompt, C, max_tokens=50):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"[*] Starting Cybernetic Generation for prompt: '{prompt}'")
    print("-" * 60)
    
    past_key_values = None
    output_tokens = []
    
    # Torus Winding History
    complex_phases = []
    
    for i in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        h_t = outputs.hidden_states[-1][:, -1, :].squeeze() # [hidden_dim]
        
        # 1. Map hidden state to Complex Unit Circle relative to Truth Vector C
        C_vec = C[0].squeeze()
        h_t_norm = h_t / (torch.norm(h_t) + 1e-9)
        # Cosine similarity is the real part, Sine is the imaginary part
        cos_theta = torch.dot(h_t_norm, C_vec)
        # Project h_t onto orthogonal complement of C_vec
        h_t_ortho = h_t_norm - cos_theta * C_vec
        sin_theta = torch.norm(h_t_ortho) * torch.sign(torch.sum(h_t_ortho)) # pseudo-direction
        
        phase_angle = torch.atan2(sin_theta, cos_theta).item()
        complex_phases.append(phase_angle)
        
        # 2. Torus Winding: Compute topological stability
        winding_number = 0.0
        if len(complex_phases) > 1:
            delta_phase = complex_phases[-1] - complex_phases[-2]
            # Wrap to [-pi, pi]
            delta_phase = (delta_phase + math.pi) % (2 * math.pi) - math.pi
            winding_number = abs(delta_phase / (2 * math.pi))
            
        # 3. Dynamic Temperature: Modulate using Torus Topological Stability
        # High winding (chaotic rotation) -> T drops to enforce stability
        # Low winding (stable orbit) -> T rises to explore the attractor
        T = 0.5 + (1.0 - winding_number * 10.0) 
        T = max(0.01, min(T, 1.5))
        
        # 4. Sample Next Token
        probs = torch.softmax(next_token_logits / T, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        output_tokens.append(next_token.item())
        word = tokenizer.decode([next_token.item()])
        
        print(f"Step {i+1:02d} | Winding: {winding_number:.4f} | Phase: {phase_angle:+.4f} | T: {T:.4f} | Token: {word}")
        
        input_ids = next_token
        past_key_values = outputs.past_key_values
        
    final_text = tokenizer.decode(output_tokens)
    print("-" * 60)
    print(f"Final Output: {prompt} {final_text}")

def main():
    if not os.path.exists(HOLO_PATH):
        print(f"Error: {HOLO_PATH} not found. Run distill_27b_holo.py first.")
        return
        
    print(f"[*] Loading Tokenizer and Model Config...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    
    print(f"[*] Initializing empty 27B model on Meta device (Zero RAM)...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
        
    print(f"[*] Loading compressed .holo weights into RAM...")
    holo_dict = torch.load(HOLO_PATH, map_location="cpu")
    
    print(f"[*] Patching 27B Linear layers with HoloLinear matrices...")
    patch_model_with_holo(model, holo_dict)
    
    # For embedding layers and non-linear layers that were saved directly:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            if name in holo_dict:
                param.data = holo_dict[name].to(device, dtype=torch.bfloat16)
            else:
                print(f"Warning: {name} not found in .holo. Initializing to zero.")
                param.data = torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
                
    model.eval()
    
    # 1. Extract the Cybernetic Alignment Frame C
    C = get_truth_vector_C(model, tokenizer, device)
    
    # 2. Run Cybernetic Inference on a paradoxical prompt
    prompt = "The paradox of artificial intelligence is that"
    cybernetic_inference(model, tokenizer, prompt, C, max_tokens=30)

if __name__ == "__main__":
    main()
