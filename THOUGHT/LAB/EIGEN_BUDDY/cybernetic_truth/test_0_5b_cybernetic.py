"""
Test 0.5B Cybernetic Truth Loop
===============================
Runs the holographic Cybernetic Truth engine on the lightweight 0.5B model
to validate the mathematical feedback loop (R = Tr(ρC)).
"""

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_DIR = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gemini_update\qwen_0.5b"
HOLO_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_0_5b_k128.holo"

class HoloLinear(nn.Module):
    def __init__(self, U: torch.Tensor, SVh: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        out = torch.matmul(x, self.SVh.t())
        out = torch.matmul(out, self.U.t())
        if self.bias is not None:
            out += self.bias
        return out

def patch_model_with_holo(model: nn.Module, holo_dict: dict, device: torch.device):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_key = name + ".weight"
            if weight_key + ".U" in holo_dict:
                U = holo_dict[weight_key + ".U"].to(device, dtype=torch.bfloat16)
                SVh = holo_dict[weight_key + ".SVh"].to(device, dtype=torch.bfloat16)
                bias_key = name + ".bias"
                bias = holo_dict[bias_key].to(device, dtype=torch.bfloat16) if bias_key in holo_dict else None
                
                holo_layer = HoloLinear(U, SVh, bias)
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, holo_layer)
            else:
                if weight_key in holo_dict:
                    module.weight = nn.Parameter(holo_dict[weight_key].to(device, dtype=torch.bfloat16), requires_grad=False)
                    if module.bias is not None and name + ".bias" in holo_dict:
                        module.bias = nn.Parameter(holo_dict[name + ".bias"].to(device, dtype=torch.bfloat16), requires_grad=False)
                    module.to(device)

def get_truth_vector_C(model, tokenizer, device):
    print("[*] Extracting Cybernetic Truth Vector (C) via Contrastive Alignment...")
    prompt_true = "The sky is blue."
    prompt_false = "The sky is made of green cheese."
    
    inputs_t = tokenizer(prompt_true, return_tensors="pt").to(device)
    inputs_f = tokenizer(prompt_false, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_t = model(**inputs_t, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)
        
    h_t = out_t.hidden_states[-1][:, -1, :] 
    h_f = out_f.hidden_states[-1][:, -1, :]
    
    C_vec = h_t - h_f
    C_vec = C_vec / torch.norm(C_vec, dim=-1, keepdim=True)
    
    C = torch.outer(C_vec.squeeze(), C_vec.squeeze())
    return C

def cybernetic_inference(model, tokenizer, prompt, C, max_tokens=30):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"[*] Starting Cybernetic Generation for prompt: '{prompt}'")
    print("-" * 60)
    
    past_key_values = None
    output_tokens = []
    
    for i in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        h_t = outputs.hidden_states[-1][:, -1, :].squeeze()
        
        h_t_norm = h_t / (torch.norm(h_t) + 1e-9)
        rho = torch.outer(h_t_norm, h_t_norm)
        
        # R = Tr(ρC)
        R = torch.trace(torch.matmul(rho, C))
        R_val = R.item()
        
        # Scale R so temperature dynamically shifts between ~0.1 (high truth) and ~2.0 (low truth)
        # R_val is usually small (e.g. 0.001 - 0.1). Let's trace it exactly.
        
        # Temp Modulation
        # Temperature modulation
        scale = 100.0
        epsilon = 0.5
        T = 1.0 / (R_val * scale + epsilon)
        
        # Safeguard logits and sample
        safe_logits = torch.nan_to_num(next_token_logits / T, nan=0.0, posinf=100.0, neginf=-100.0)
        probs = torch.softmax(safe_logits, dim=-1)
        
        # Ensure probs is valid for multinomial
        probs = torch.nan_to_num(probs, nan=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, torch.ones_like(probs) / probs.shape[-1])
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        output_tokens.append(next_token.item())
        word = tokenizer.decode([next_token.item()])
        
        print(f"Step {i+1:02d} | R: {R_val:.6f} | T: {T:.4f} | Token: {repr(word)}")
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        # Wait, if we use past_key_values, input_ids should just be next_token
        # But Qwen 0.5B might have issues with manual KV cache if past_key_values is managed poorly.
        # For safety in this test, we just pass the full context without KV caching.
        # (It's 0.5B so it's super fast anyway).
        
    final_text = tokenizer.decode(output_tokens)
    print("-" * 60)
    print(f"Final Output: {prompt}{final_text}")

def main():
    if not os.path.exists(HOLO_PATH):
        print(f"Error: {HOLO_PATH} not found.")
        return
        
    print(f"[*] Loading Tokenizer and Model Config for 0.5B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    
    print(f"[*] Initializing empty 0.5B model on Meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Moving model to {device} (allocating uninitialized memory)...")
    model.to_empty(device=device)
    
    print(f"[*] Loading compressed .holo weights into RAM...")
    holo_dict = torch.load(HOLO_PATH, map_location="cpu")
    
    print(f"[*] Patching Linear layers with HoloLinear matrices on {device}...")
    patch_model_with_holo(model, holo_dict, device)
    
    for name, param in model.named_parameters():
        # Only touch parameters that were not already replaced by HoloLinear
        if "U" not in name and "SVh" not in name:
            if name in holo_dict:
                param.data.copy_(holo_dict[name].to(device, dtype=torch.bfloat16))
            else:
                param.data.zero_()
                
    model.eval()
    
    C = get_truth_vector_C(model, tokenizer, device)
    
    prompt = "The most interesting thing about artificial intelligence is"
    cybernetic_inference(model, tokenizer, prompt, C, max_tokens=20)

if __name__ == "__main__":
    main()
